"""Faithful one-block Anima/Cosmos CPU oracle and XDNA adapter."""

from dataclasses import dataclass
from contextlib import nullcontext
import math
import time
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F

from .block_checkpoint import AnimaBlockConfig, AnimaBlockWeights
from .errors import NumericalMismatch, PrototypeError, UnsupportedTensor
from .linear import cpu_linear, prepare_linear


LinearRunner = Callable[
    [str, torch.Tensor, torch.Tensor, Optional[str]], torch.Tensor
]
BatchedRunner = Callable[[str, torch.Tensor, torch.Tensor], torch.Tensor]
QKVRunner = Callable[
    [str, torch.Tensor, torch.Tensor, tuple[str, str, str], tuple[torch.Tensor, ...]],
    dict[str, Any],
]


@dataclass(frozen=True)
class BlockInputs:
    hidden_states: torch.Tensor
    encoder_hidden_states: torch.Tensor
    embedded_timestep: torch.Tensor
    temb: torch.Tensor
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor]
    attention_mask: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class StageMetric:
    name: str
    device: str
    wall_ms: float
    dispatches: int
    estimated_flops: int
    transfer_bytes: int
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    allocation_count: int = 0
    resident_hits: int = 0
    host_copy_ms: float = 0.0
    h2d_sync_ms: float = 0.0
    kernel_sync_ms: float = 0.0
    d2h_sync_ms: float = 0.0
    weight_population_ms: float = 0.0
    weight_population_bytes: int = 0


@dataclass(frozen=True)
class BlockResult:
    output: torch.Tensor
    metrics: tuple[StageMetric, ...]
    artifact_caches: tuple[str, ...] = ()
    checkpoints: tuple[tuple[str, torch.Tensor], ...] = ()

    @property
    def dispatch_count(self) -> int:
        return sum(metric.dispatches for metric in self.metrics)


def deterministic_block_inputs(
    config: AnimaBlockConfig,
    image_tokens: int,
    context_tokens: int,
    seed: int = 0,
    masked_context_tokens: int = 0,
) -> BlockInputs:
    if image_tokens <= 0 or context_tokens <= 0:
        raise UnsupportedTensor("image and context token counts must be positive")
    if not 0 <= masked_context_tokens < context_tokens:
        raise UnsupportedTensor(
            "masked context token count must be non-negative and smaller than context tokens"
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)

    def sample(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16)

    hidden = sample(1, image_tokens, config.hidden_size, scale=0.02)
    context = sample(1, context_tokens, config.context_dim, scale=0.02)
    embedded = sample(1, config.hidden_size, scale=0.02)
    temb = sample(1, 3 * config.hidden_size, scale=0.02)
    cos, sin = rotary_embedding(config, image_tokens)
    mask = None
    if masked_context_tokens:
        mask = torch.ones(1, 1, 1, context_tokens, dtype=torch.bool)
        mask[..., -masked_context_tokens:] = False
    return BlockInputs(hidden, context, embedded, temb, (cos, sin), mask)


def rotary_embedding(
    config: AnimaBlockConfig,
    image_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    side = math.isqrt(image_tokens)
    if side * side != image_tokens:
        raise UnsupportedTensor(
            "image token count must be a square for the image-only Cosmos RoPE fixture"
        )
    dim_h = config.head_dim // 6 * 2
    dim_w = config.head_dim // 6 * 2
    dim_t = config.head_dim - dim_h - dim_w
    dims = (dim_t, dim_h, dim_w)
    factors = tuple(
        scale ** (dimension / (dimension - 2))
        for scale, dimension in zip(config.rope_scale, dims)
    )
    coordinates = (
        torch.zeros(image_tokens, dtype=torch.float32),
        torch.arange(side, dtype=torch.float32).view(side, 1).expand(side, side).flatten(),
        torch.arange(side, dtype=torch.float32).view(1, side).expand(side, side).flatten(),
    )
    pieces = []
    for coordinate, dimension, factor in zip(coordinates, dims, factors):
        powers = torch.arange(0, dimension, 2, dtype=torch.float32) / dimension
        pieces.append(coordinate[:, None] / ((10000.0 * factor) ** powers[None, :]))
    frequencies = torch.cat(pieces * 2, dim=-1)
    return frequencies.cos(), frequencies.sin()


def _validate_inputs(config: AnimaBlockConfig, inputs: BlockInputs) -> None:
    expected = {
        "hidden_states": (1, inputs.hidden_states.shape[1], config.hidden_size),
        "encoder_hidden_states": (1, inputs.encoder_hidden_states.shape[1], config.context_dim),
        "embedded_timestep": (1, config.hidden_size),
        "temb": (1, 3 * config.hidden_size),
    }
    for name, shape in expected.items():
        tensor = getattr(inputs, name)
        if tuple(tensor.shape) != shape:
            raise UnsupportedTensor(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
    sequence = inputs.hidden_states.shape[1]
    for name, tensor in zip(("cos", "sin"), inputs.image_rotary_emb):
        if tuple(tensor.shape) != (sequence, config.head_dim):
            raise UnsupportedTensor(
                f"RoPE {name} has shape {tuple(tensor.shape)}, "
                f"expected {(sequence, config.head_dim)}"
            )
    if inputs.attention_mask is not None:
        expected_mask = (1, 1, 1, inputs.encoder_hidden_states.shape[1])
        if tuple(inputs.attention_mask.shape) != expected_mask:
            raise UnsupportedTensor(
                f"attention_mask has shape {tuple(inputs.attention_mask.shape)}, "
                f"expected {expected_mask}"
            )


def _cpu_runner(_: str, inputs: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    shape = inputs.shape
    prepared = prepare_linear(inputs.reshape(-1, shape[-1]), weight)
    return cpu_linear(prepared).reshape(*shape[:-1], weight.shape[0])


class _BlockExecution:
    def __init__(
        self,
        config: AnimaBlockConfig,
        weights: AnimaBlockWeights,
        runner: LinearRunner,
        runner_device: str,
        batched_runner: Optional[BatchedRunner] = None,
        qkv_runner: Optional[QKVRunner] = None,
    ):
        self.config = config
        self.weights = weights
        self.runner = runner
        self.runner_device = runner_device
        self.batched_runner = batched_runner
        self.qkv_runner = qkv_runner
        self.metrics: list[StageMetric] = []

    def host(self, name: str, operation: Callable[[], Any]) -> Any:
        started = time.perf_counter()
        result = operation()
        self.metrics.append(
            StageMetric(name, "host", (time.perf_counter() - started) * 1000, 0, 0, 0)
        )
        return result

    def linear(
        self,
        name: str,
        inputs: torch.Tensor,
        weight_name: Union[str, "_AttentionWeight"],
    ) -> torch.Tensor:
        weight = (
            weight_name.tensor
            if isinstance(weight_name, _AttentionWeight)
            else self.weights[weight_name]
        )
        rows = inputs.numel() // inputs.shape[-1]
        started = time.perf_counter()
        try:
            result = self.runner(
                name,
                inputs,
                weight,
                None if isinstance(weight_name, _AttentionWeight) else weight_name,
            )
        except PrototypeError as error:
            raise type(error)(f"{name}: {error}") from error
        elapsed = (time.perf_counter() - started) * 1000
        flops = 2 * rows * weight.shape[0] * weight.shape[1]
        transfer = (inputs.numel() + weight.numel() + result.numel()) * 2
        self.metrics.append(
            StageMetric(
                name,
                self.runner_device,
                elapsed,
                getattr(self.runner, "last_dispatches", 1),
                flops,
                transfer,
                getattr(self.runner, "last_h2d_bytes", 0),
                getattr(self.runner, "last_d2h_bytes", 0),
                getattr(self.runner, "last_allocation_count", 0),
                getattr(self.runner, "last_resident_hits", 0),
                getattr(self.runner, "last_host_copy_ms", 0.0),
                getattr(self.runner, "last_h2d_sync_ms", 0.0),
                getattr(self.runner, "last_kernel_sync_ms", 0.0),
                getattr(self.runner, "last_d2h_sync_ms", 0.0),
                getattr(self.runner, "last_weight_population_ms", 0.0),
                getattr(self.runner, "last_weight_population_bytes", 0),
            )
        )
        return result

    def qkv(
        self,
        prefix: str,
        hidden: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        names = (
            prefix + ".to_q.weight",
            prefix + ".to_k.weight",
            prefix + ".to_v.weight",
        )
        if self.qkv_runner is None:
            return (
                self.linear(prefix + ".q", hidden, names[0]),
                self.linear(prefix + ".k", context, names[1]),
                self.linear(prefix + ".v", context, names[2]),
            )
        weights = tuple(self.weights[name] for name in names)
        started = time.perf_counter()
        try:
            profile = self.qkv_runner(
                prefix, hidden, context, names, weights
            )
        except PrototypeError as error:
            raise type(error)(f"{prefix}.qkv_chain: {error}") from error
        elapsed = (time.perf_counter() - started) * 1000
        rows = (
            hidden.numel() // hidden.shape[-1],
            context.numel() // context.shape[-1],
            context.numel() // context.shape[-1],
        )
        flops = sum(
            2 * row_count * weight.shape[0] * weight.shape[1]
            for row_count, weight in zip(rows, weights)
        )
        transfer = sum(
            weight.numel() * 2 for weight in weights
        )
        transfer += hidden.numel() * 2
        if hidden is not context:
            transfer += context.numel() * 2
        transfer += sum(output.numel() * 2 for output in profile["outputs"])
        self.metrics.append(
            StageMetric(
                prefix + ".qkv_chain",
                self.runner_device,
                elapsed,
                profile["dispatches"],
                flops,
                transfer,
                profile["h2d_bytes"],
                profile["d2h_bytes"],
                profile["allocation_count"],
                profile["resident_hits"],
                weight_population_bytes=profile["weight_population_bytes"],
            )
        )
        return profile["outputs"]

    def adaln(
        self,
        prefix: str,
        hidden: torch.Tensor,
        embedded: torch.Tensor,
        temb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        activated = self.host(prefix + ".silu", lambda: F.silu(embedded))
        modulation = self.linear(prefix + ".linear_1", activated, prefix + ".linear_1.weight")
        modulation = self.linear(prefix + ".linear_2", modulation, prefix + ".linear_2.weight")

        def normalize() -> tuple[torch.Tensor, torch.Tensor]:
            combined = modulation + temb
            shift, scale, gate = combined.chunk(3, dim=-1)
            normalized = F.layer_norm(hidden, (self.config.hidden_size,), eps=1e-6)
            return (
                normalized * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1),
                gate.unsqueeze(1),
            )

        return self.host(prefix + ".layer_norm_modulation", normalize)

    def rms_norm(self, name: str, value: torch.Tensor, weight_name: str) -> torch.Tensor:
        weight = self.weights[weight_name]

        def normalize() -> torch.Tensor:
            variance = value.float().pow(2).mean(-1, keepdim=True)
            normalized = value * torch.rsqrt(variance + 1e-6)
            return normalized.to(torch.bfloat16) * weight

        return self.host(name, normalize)

    def attention(
        self,
        prefix: str,
        hidden: torch.Tensor,
        context: torch.Tensor,
        rotary: Optional[tuple[torch.Tensor, torch.Tensor]],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.config
        query, key, value = self.qkv(prefix, hidden, context)
        query, key, value = self.host(
            prefix + ".qkv_layout",
            lambda: tuple(
                tensor.view(1, -1, cfg.num_heads, cfg.head_dim).transpose(1, 2)
                for tensor in (query, key, value)
            ),
        )
        query = self.rms_norm(prefix + ".q_rms_norm", query, prefix + ".norm_q.weight")
        key = self.rms_norm(prefix + ".k_rms_norm", key, prefix + ".norm_k.weight")

        if rotary is not None:
            def apply_rope(value: torch.Tensor) -> torch.Tensor:
                cos, sin = (part[None, None] for part in rotary)
                first, second = value.reshape(*value.shape[:-1], 2, -1).unbind(-2)
                rotated = torch.cat((-second, first), dim=-1)
                return (value.float() * cos + rotated.float() * sin).to(value.dtype)

            query = self.host(prefix + ".q_rope", lambda: apply_rope(query))
            key = self.host(prefix + ".k_rope", lambda: apply_rope(key))

        scale = cfg.head_dim**-0.5
        if self.batched_runner is None:
            score_heads = []
            for head in range(cfg.num_heads):
                scores = self.linear(
                    f"{prefix}.qk.head{head}",
                    query[:, head],
                    _AttentionWeight(key[:, head].squeeze(0)),
                )
                score_heads.append(scores)
            scores = self.host(
                prefix + ".qk_scale_stack",
                lambda: torch.stack(score_heads, dim=1) * scale,
            )
        else:
            scores = self.batched(
                prefix + ".qk_batched",
                query.squeeze(0),
                key.squeeze(0).transpose(1, 2),
            ).unsqueeze(0)
            scores = self.host(prefix + ".qk_scale", lambda: scores * scale)
        if mask is not None:
            scores = self.host(
                prefix + ".mask",
                lambda: scores.masked_fill(~mask, float("-inf")),
            )
        probabilities = self.host(
            prefix + ".softmax",
            lambda: torch.softmax(scores.float(), dim=-1).to(torch.bfloat16),
        )
        if self.batched_runner is None:
            output_heads = []
            for head in range(cfg.num_heads):
                output_heads.append(
                    self.linear(
                        f"{prefix}.av.head{head}",
                        probabilities[:, head],
                        _AttentionWeight(value[:, head].transpose(1, 2).squeeze(0)),
                    )
                )
            attended = self.host(
                prefix + ".av_layout",
                lambda: torch.stack(output_heads, dim=1)
                .transpose(1, 2)
                .reshape(1, hidden.shape[1], cfg.hidden_size),
            )
        else:
            output = self.batched(
                prefix + ".av_batched",
                probabilities.squeeze(0),
                value.squeeze(0),
            )
            attended = self.host(
                prefix + ".av_layout",
                lambda: output.transpose(0, 1).reshape(
                    1, hidden.shape[1], cfg.hidden_size
                ),
            )
        return self.linear(prefix + ".out", attended, prefix + ".to_out.0.weight")

    def batched(
        self,
        name: str,
        inputs: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        started = time.perf_counter()
        try:
            result = self.batched_runner(name, inputs, weights)
        except PrototypeError as error:
            raise type(error)(f"{name}: {error}") from error
        elapsed = (time.perf_counter() - started) * 1000
        heads, rows, inner = inputs.shape
        columns = weights.shape[-1]
        self.metrics.append(
            StageMetric(
                name,
                self.runner_device,
                elapsed,
                1,
                2 * heads * rows * inner * columns,
                (inputs.numel() + weights.numel() + result.numel()) * 2,
                getattr(self.batched_runner, "last_h2d_bytes", 0),
                getattr(self.batched_runner, "last_d2h_bytes", 0),
                getattr(self.batched_runner, "last_allocation_count", 0),
                getattr(self.batched_runner, "last_resident_hits", 0),
                getattr(self.batched_runner, "last_host_copy_ms", 0.0),
                getattr(self.batched_runner, "last_h2d_sync_ms", 0.0),
                getattr(self.batched_runner, "last_kernel_sync_ms", 0.0),
                getattr(self.batched_runner, "last_d2h_sync_ms", 0.0),
                getattr(self.batched_runner, "last_weight_population_ms", 0.0),
                getattr(self.batched_runner, "last_weight_population_bytes", 0),
            )
        )
        return result

    def run(self, inputs: BlockInputs) -> BlockResult:
        hidden = inputs.hidden_states
        checkpoints = []
        normalized, gate = self.adaln(
            "norm1", hidden, inputs.embedded_timestep, inputs.temb
        )
        attended = self.attention(
            "attn1", normalized, normalized, inputs.image_rotary_emb, None
        )
        hidden = self.host("attn1.gated_residual", lambda: hidden + gate * attended)
        checkpoints.append(("self_attention", hidden))

        normalized, gate = self.adaln(
            "norm2", hidden, inputs.embedded_timestep, inputs.temb
        )
        attended = self.attention(
            "attn2",
            normalized,
            inputs.encoder_hidden_states,
            None,
            inputs.attention_mask,
        )
        hidden = self.host("attn2.gated_residual", lambda: hidden + gate * attended)
        checkpoints.append(("cross_attention", hidden))

        normalized, gate = self.adaln(
            "norm3", hidden, inputs.embedded_timestep, inputs.temb
        )
        feed_forward = self.linear(
            "ff.proj_in", normalized, "ff.net.0.proj.weight"
        )
        feed_forward = self.host(
            "ff.gelu",
            lambda: F.gelu(feed_forward, approximate="none"),
        )
        feed_forward = self.linear(
            "ff.proj_out", feed_forward, "ff.net.2.weight"
        )
        hidden = self.host("ff.gated_residual", lambda: hidden + gate * feed_forward)
        checkpoints.append(("feed_forward", hidden))
        return BlockResult(
            hidden,
            tuple(self.metrics),
            checkpoints=tuple(checkpoints),
        )


class _AttentionWeight:
    """Marker carrying a dynamic attention matrix through the Linear runner API."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor


def run_cpu_block(
    config: AnimaBlockConfig,
    weights: AnimaBlockWeights,
    inputs: BlockInputs,
) -> BlockResult:
    _validate_inputs(config, inputs)

    def runner(
        name: str,
        value: torch.Tensor,
        weight,
        weight_name: Optional[str],
    ) -> torch.Tensor:
        return _cpu_runner(name, value, weight)

    return _BlockExecution(config, weights, runner, "host").run(inputs)


def run_xdna_block(
    config: AnimaBlockConfig,
    weights: AnimaBlockWeights,
    inputs: BlockInputs,
    session=None,
    resident_key: Optional[str] = None,
    batched_attention: bool = True,
    packed_cache=None,
    block_index: Optional[int] = None,
    qkv_chaining: bool = True,
) -> BlockResult:
    _validate_inputs(config, inputs)
    from .xdna import XDNASession

    caches: set[str] = set()
    session_scope = XDNASession() if session is None else nullcontext(session)
    with session_scope as active_session:
        def runner(
            name: str,
            value: torch.Tensor,
            weight: torch.Tensor,
            weight_name: Optional[str],
        ) -> torch.Tensor:
            shape = value.shape
            flattened = value.reshape(-1, shape[-1])
            partials = []
            profiles = []
            for start in range(0, shape[-1], 2048):
                end = min(start + 2048, shape[-1])
                operation_key = f"{resident_key}:{name}:k{start}"
                if (
                    resident_key is not None
                    and hasattr(active_session, "has_resident")
                    and active_session.has_resident(operation_key)
                ):
                    dispatched = active_session.dispatch_resident_input(
                        flattened[:, start:end],
                        operation_key,
                    )
                else:
                    if (
                        packed_cache is not None
                        and block_index is not None
                        and weight_name is not None
                    ):
                        prepared = packed_cache.prepared(
                            block_index,
                            weight_name,
                            start,
                            flattened[:, start:end],
                            weight.shape[0],
                        )
                    else:
                        prepared = prepare_linear(
                            flattened[:, start:end],
                            weight[:, start:end],
                        )
                    if resident_key is not None and hasattr(
                        active_session, "dispatch_resident"
                    ):
                        dispatched = active_session.dispatch_resident(
                            prepared,
                            operation_key,
                        )
                    else:
                        dispatched = active_session.dispatch(prepared)
                partials.append(
                    dispatched.output_fp32.clone()
                    if shape[-1] > 2048
                    else dispatched.output_fp32
                )
                profiles.append(dispatched)
                if dispatched.artifact_cache:
                    caches.add(dispatched.artifact_cache)
            runner.last_dispatches = len(partials)
            for field in (
                "h2d_bytes",
                "d2h_bytes",
                "allocation_count",
                "host_copy_ms",
                "h2d_sync_ms",
                "kernel_sync_ms",
                "d2h_sync_ms",
                "weight_population_ms",
                "weight_population_bytes",
            ):
                setattr(runner, "last_" + field, sum(getattr(p, field) for p in profiles))
            runner.last_resident_hits = sum(p.resident_hit for p in profiles)
            output = torch.stack(partials).sum(dim=0).to(torch.bfloat16)
            return output.reshape(*shape[:-1], weight.shape[0])

        def batched_runner(
            _: str,
            value: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            if hasattr(active_session, "dispatch_head_chain"):
                dispatched = active_session.dispatch_head_chain(
                    value,
                    weight,
                    f"{resident_key or 'block'}:{_}",
                )
            elif resident_key is not None and hasattr(
                active_session, "dispatch_batched_resident"
            ):
                dispatched = active_session.dispatch_batched_resident(
                    value,
                    weight,
                    f"{resident_key}:{_}",
                )
            else:
                dispatched = active_session.dispatch_batched(value, weight)
            if dispatched.artifact_cache:
                caches.add(dispatched.artifact_cache)
            for field in (
                "h2d_bytes",
                "d2h_bytes",
                "allocation_count",
                "host_copy_ms",
                "h2d_sync_ms",
                "kernel_sync_ms",
                "d2h_sync_ms",
                "weight_population_ms",
                "weight_population_bytes",
            ):
                setattr(batched_runner, "last_" + field, getattr(dispatched, field))
            batched_runner.last_resident_hits = int(dispatched.resident_hit)
            return dispatched.output_bf16

        def qkv_runner(
            prefix: str,
            hidden: torch.Tensor,
            context: torch.Tensor,
            weight_names: tuple[str, str, str],
            projection_weights: tuple[torch.Tensor, ...],
        ) -> dict[str, Any]:
            if resident_key is None or not hasattr(
                active_session, "dispatch_qkv_chain"
            ):
                raise UnsupportedTensor(
                    "resident Q/K/V chaining requires a fingerprint-scoped XDNA session"
                )
            sources = (hidden, context, context)
            prepared = []
            for source, weight, weight_name in zip(
                sources, projection_weights, weight_names
            ):
                flattened = source.reshape(-1, source.shape[-1])
                if flattened.shape[1] > 2048:
                    raise UnsupportedTensor(
                        "resident Q/K/V chaining supports input widths up to 2048"
                    )
                if (
                    packed_cache is not None
                    and block_index is not None
                ):
                    item = packed_cache.prepared(
                        block_index,
                        weight_name,
                        0,
                        flattened,
                        weight.shape[0],
                    )
                else:
                    item = prepare_linear(flattened, weight)
                prepared.append(item)
            result = active_session.dispatch_qkv_chain(
                tuple(prepared),
                f"{packed_cache.manifest['cache_key'] if packed_cache is not None else resident_key}:"
                f"{resident_key}:{prefix}:qkv",
                shared_input=hidden is context,
            )
            return result

        result = _BlockExecution(
            config,
            weights,
            runner,
            "xdna2",
            batched_runner if batched_attention else None,
            qkv_runner
            if resident_key is not None and qkv_chaining
            else None,
        ).run(inputs)
    return BlockResult(
        result.output,
        result.metrics,
        tuple(sorted(caches)),
        result.checkpoints,
    )


def compare_block_outputs(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
) -> tuple[float, float]:
    difference = (actual.float() - expected.float()).abs()
    max_error = difference.max().item()
    mean_error = difference.mean().item()
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as error:
        raise NumericalMismatch(
            f"NPU block output differs from the CPU oracle "
            f"(max_abs_error={max_error:.6g}, mean_abs_error={mean_error:.6g}, "
            f"rtol={rtol}, atol={atol}): {error}"
        ) from error
    return max_error, mean_error
