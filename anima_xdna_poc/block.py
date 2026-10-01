"""Faithful one-block Anima/Cosmos CPU oracle and XDNA adapter."""

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F

from .block_checkpoint import AnimaBlockConfig, AnimaBlockWeights
from .errors import NumericalMismatch, PrototypeError, UnsupportedTensor
from .linear import cpu_linear, prepare_linear


LinearRunner = Callable[[str, torch.Tensor, torch.Tensor], torch.Tensor]


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
    ):
        self.config = config
        self.weights = weights
        self.runner = runner
        self.runner_device = runner_device
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
            result = self.runner(name, inputs, weight)
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
            )
        )
        return result

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
        query = self.linear(prefix + ".q", hidden, prefix + ".to_q.weight")
        key = self.linear(prefix + ".k", context, prefix + ".to_k.weight")
        value = self.linear(prefix + ".v", context, prefix + ".to_v.weight")
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

        score_heads = []
        scale = cfg.head_dim**-0.5
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
        if mask is not None:
            scores = self.host(
                prefix + ".mask",
                lambda: scores.masked_fill(~mask, float("-inf")),
            )
        probabilities = self.host(
            prefix + ".softmax",
            lambda: torch.softmax(scores.float(), dim=-1).to(torch.bfloat16),
        )
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
        return self.linear(prefix + ".out", attended, prefix + ".to_out.0.weight")

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

    def runner(name: str, value: torch.Tensor, weight) -> torch.Tensor:
        return _cpu_runner(name, value, weight)

    return _BlockExecution(config, weights, runner, "host").run(inputs)


def run_xdna_block(
    config: AnimaBlockConfig,
    weights: AnimaBlockWeights,
    inputs: BlockInputs,
) -> BlockResult:
    _validate_inputs(config, inputs)
    from .xdna import XDNASession

    caches: set[str] = set()
    with XDNASession() as session:
        def runner(_: str, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            shape = value.shape
            flattened = value.reshape(-1, shape[-1])
            partials = []
            for start in range(0, shape[-1], 2048):
                end = min(start + 2048, shape[-1])
                prepared = prepare_linear(flattened[:, start:end], weight[:, start:end])
                dispatched = session.dispatch(prepared)
                partials.append(dispatched.output_fp32)
                if dispatched.artifact_cache:
                    caches.add(dispatched.artifact_cache)
            runner.last_dispatches = len(partials)
            output = torch.stack(partials).sum(dim=0).to(torch.bfloat16)
            return output.reshape(*shape[:-1], weight.shape[0])

        result = _BlockExecution(config, weights, runner, "xdna2").run(inputs)
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
