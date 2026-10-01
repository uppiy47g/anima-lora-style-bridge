"""Guarded ComfyUI MODEL wrapper replacing only Anima's 28-block loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any, Optional
import weakref

import torch

from anima_xdna_poc.block import BlockInputs, rotary_embedding
from anima_xdna_poc.chain import AnimaXDNAChainRuntime
from anima_xdna_poc.errors import PrototypeError, UnsupportedTensor


SUPPORTED_COMFY_COMMIT = "170594057a22673349ddf0a3d88624b7fa5865bb"
WRAPPER_KEY = "anima_xdna2_resident"
ATTACHMENT_KEY = "anima_xdna2_runtime"
SUPPORTED_SHAPE = (1, 16, 1, 64, 64)


@dataclass
class RuntimeDiagnostics:
    state: str = "created"
    calls: int = 0
    chain_runs: int = 0
    resident_reuses: int = 0
    cold_setup_ms: float = 0.0
    first_call_total_ms: float = 0.0
    first_chain_ms: float = 0.0
    last_total_ms: float = 0.0
    last_preprocess_ms: float = 0.0
    last_chain_ms: float = 0.0
    last_postprocess_ms: float = 0.0
    last_input_bytes: int = 0
    last_output_bytes: int = 0
    last_dispatches: int = 0
    last_h2d_bytes: int = 0
    last_d2h_bytes: int = 0
    last_allocations: int = 0
    last_resident_hits: int = 0
    last_weight_population_ms: float = 0.0
    last_weight_population_bytes: int = 0
    last_error: Optional[str] = None
    cache: Optional[dict[str, Any]] = None


class SharedRuntime:
    """One single-tenant XRT runtime shared by ModelPatcher clones."""

    _npu_lock = threading.RLock()

    def __init__(
        self,
        checkpoint: Path,
        cache_dir: Optional[Path] = None,
        rebuild_cache: bool = False,
    ):
        self.checkpoint = Path(checkpoint)
        self.cache_dir = cache_dir
        self.rebuild_cache = rebuild_cache
        self._runtime: Optional[AnimaXDNAChainRuntime] = None
        self._refs = 1
        self._lock = threading.RLock()
        self.diagnostics = RuntimeDiagnostics()

    def acquire(self):
        with self._lock:
            if self._refs <= 0:
                raise RuntimeError("Anima XDNA runtime is already closed")
            self._refs += 1
        return self

    def release(self):
        with self._lock:
            self._refs -= 1
            if self._refs <= 0:
                self.close()

    def open(self) -> AnimaXDNAChainRuntime:
        with self._lock:
            if self._runtime is None:
                started = time.perf_counter()
                runtime = AnimaXDNAChainRuntime(
                    self.checkpoint,
                    cache_host_weights=False,
                    weight_cache=True,
                    cache_dir=self.cache_dir,
                    rebuild_cache=self.rebuild_cache,
                )
                runtime.__enter__()
                self._runtime = runtime
                self.diagnostics.cold_setup_ms = (
                    time.perf_counter() - started
                ) * 1000
                self.diagnostics.state = "open"
            return self._runtime

    def close(self):
        with self._lock:
            if self._runtime is not None:
                self._runtime.close()
                self._runtime = None
            self.diagnostics.state = "closed"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            payload = asdict(self.diagnostics)
            payload.update(
                {
                    "checkpoint": str(self.checkpoint),
                    "reference_count": self._refs,
                    "runtime_open": self._runtime is not None,
                    "supported_shape": list(SUPPORTED_SHAPE),
                    "replacement_boundary": (
                        "after ComfyUI patch/time/Qwen/RoPE preprocessing; "
                        "before final_layer/unpatchify"
                    ),
                }
            )
            return payload

    @staticmethod
    def _interrupt():
        try:
            import comfy.model_management

            comfy.model_management.throw_exception_if_processing_interrupted()
        except ImportError:
            pass

    @staticmethod
    def _validate_model(model, x, context, kwargs):
        class_name = type(model).__name__
        module = type(model).__module__
        if class_name != "Anima" or not module.endswith("ldm.anima.model"):
            raise UnsupportedTensor(
                "Anima XDNA wrapper requires ComfyUI's comfy.ldm.anima.model.Anima; "
                f"received {module}.{class_name}"
            )
        if x.ndim != 5 or tuple(x.shape[1:]) != SUPPORTED_SHAPE[1:] or x.shape[0] not in (1, 2):
            raise UnsupportedTensor(
                "Anima XDNA currently supports a batch-1 latent (or the "
                "sampler's two-way CFG batch) with shape [B,16,1,64,64], "
                f"got {tuple(x.shape)}"
            )
        if tuple(context.shape) != (x.shape[0], 512, 1024):
            raise UnsupportedTensor(
                "Anima XDNA requires one adapted Qwen context per sampler "
                f"batch item with shape [B,512,1024], got {tuple(context.shape)}"
            )
        transformer_options = kwargs.get("transformer_options", {})
        patches = transformer_options.get("patches", {})
        replacements = transformer_options.get("patches_replace", {})
        if patches or replacements:
            names = sorted(set(patches) | set(replacements))
            raise UnsupportedTensor(
                "Anima XDNA cannot execute transformer patches/LoRAs yet; "
                "remove them before attaching the wrapper. Active patches: "
                + ", ".join(names)
            )
        required = (
            "prepare_embedded_sequence",
            "t_embedder",
            "t_embedding_norm",
            "final_layer",
            "unpatchify",
            "blocks",
        )
        missing = [name for name in required if not hasattr(model, name)]
        if missing or len(model.blocks) != 28:
            raise UnsupportedTensor(
                "unsupported ComfyUI Anima API/version; expected 28 blocks and "
                f"pre/post methods, missing={missing}"
            )

    def forward(
        self,
        model,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        fps=None,
        padding_mask=None,
        **kwargs,
    ) -> torch.Tensor:
        self._validate_model(model, x, context, kwargs)
        self._interrupt()
        total_started = time.perf_counter()
        original_device = x.device
        original_dtype = x.dtype
        orig_shape = list(x.shape)

        pre_started = time.perf_counter()
        # The fixed 512x512 image shape is already divisible by [1,2,2].
        embedded, _comfy_rope, extra_position = model.prepare_embedded_sequence(
            x,
            fps=fps,
            padding_mask=padding_mask,
        )
        if extra_position is not None:
            raise UnsupportedTensor(
                "extra per-block absolute position embeddings are unsupported"
            )
        timestep_tokens = timesteps.unsqueeze(1) if timesteps.ndim == 1 else timesteps
        embedded_timestep, temb = model.t_embedder[1](
            model.t_embedder[0](timestep_tokens).to(embedded.dtype)
        )
        embedded_timestep = model.t_embedding_norm(embedded_timestep)
        if temb is None:
            raise UnsupportedTensor("Anima XDNA requires AdaLN-LoRA timestep modulation")
        grid_shape = embedded.shape[1:4]
        if tuple(grid_shape) != (1, 32, 32):
            raise UnsupportedTensor(
                f"expected patch grid [1,32,32], got {tuple(grid_shape)}"
            )
        batch = x.shape[0]
        hidden = embedded.reshape(batch, 1024, 2048).to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
        cpu_context = context.to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
        cpu_timestep = embedded_timestep.reshape(batch, 2048).to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
        cpu_temb = temb.reshape(batch, 6144).to(
            device="cpu", dtype=torch.bfloat16
        ).contiguous()
        rope = rotary_embedding(self.open().config, 1024)
        preprocess_ms = (time.perf_counter() - pre_started) * 1000

        self._interrupt()
        with self._npu_lock:
            runtime = self.open()
            chains = []
            for index in range(batch):
                block_inputs = BlockInputs(
                    hidden[index : index + 1],
                    cpu_context[index : index + 1],
                    cpu_timestep[index : index + 1],
                    cpu_temb[index : index + 1],
                    rope,
                    None,
                )
                chains.append(runtime.run_range(block_inputs))
                self._interrupt()
        self._interrupt()

        post_started = time.perf_counter()
        chain_output = torch.cat([chain.output for chain in chains], dim=0)
        residual = chain_output.reshape(batch, 1, 32, 32, 2048).to(
            device=original_device,
            dtype=context.dtype,
        )
        output_patches = model.final_layer(
            residual,
            embedded_timestep,
            adaln_lora_B_T_3D=temb,
        )
        output = model.unpatchify(output_patches)[
            :, :, : orig_shape[-3], : orig_shape[-2], : orig_shape[-1]
        ]
        output = output.to(device=original_device, dtype=original_dtype)
        postprocess_ms = (time.perf_counter() - post_started) * 1000

        with self._lock:
            metrics = tuple(
                metric
                for chain in chains
                for block in chain.blocks
                for metric in block.result.metrics
            )
            chain_wall_ms = sum(chain.wall_ms for chain in chains)
            dispatch_count = sum(chain.dispatch_count for chain in chains)
            call_total_ms = (time.perf_counter() - total_started) * 1000
            if self.diagnostics.calls == 0:
                self.diagnostics.first_call_total_ms = call_total_ms
                self.diagnostics.first_chain_ms = chain_wall_ms
            self.diagnostics.calls += 1
            self.diagnostics.chain_runs += batch
            self.diagnostics.resident_reuses = max(
                0, self.diagnostics.chain_runs - 1
            )
            self.diagnostics.last_preprocess_ms = preprocess_ms
            self.diagnostics.last_chain_ms = chain_wall_ms
            self.diagnostics.last_postprocess_ms = postprocess_ms
            self.diagnostics.last_total_ms = call_total_ms
            self.diagnostics.last_input_bytes = (
                hidden.numel() * hidden.element_size()
                + cpu_context.numel() * cpu_context.element_size()
                + batch * 2048 * 2
                + batch * 6144 * 2
            )
            self.diagnostics.last_output_bytes = (
                chain_output.numel() * chain_output.element_size()
            )
            self.diagnostics.last_dispatches = dispatch_count
            self.diagnostics.last_h2d_bytes = sum(
                metric.h2d_bytes for metric in metrics
            )
            self.diagnostics.last_d2h_bytes = sum(
                metric.d2h_bytes for metric in metrics
            )
            self.diagnostics.last_allocations = sum(
                metric.allocation_count for metric in metrics
            )
            self.diagnostics.last_resident_hits = sum(
                metric.resident_hits for metric in metrics
            )
            self.diagnostics.last_weight_population_ms = sum(
                metric.weight_population_ms for metric in metrics
            )
            self.diagnostics.last_weight_population_bytes = sum(
                metric.weight_population_bytes for metric in metrics
            )
            self.diagnostics.cache = (
                {
                    **asdict(runtime.cache_status),
                    "path": str(runtime.cache_status.path),
                }
                if runtime.cache_status is not None
                else None
            )
            self.diagnostics.last_error = None
        return output


class RuntimeAttachment:
    def __init__(self, runtime: SharedRuntime):
        self.runtime = runtime
        self._finalizer = weakref.finalize(self, runtime.release)

    def on_model_patcher_clone(self):
        return RuntimeAttachment(self.runtime.acquire())

    def cleanup(self):
        if self._finalizer.alive:
            self._finalizer()


class AnimaXDNADiffusionWrapper:
    def __init__(self, runtime: SharedRuntime):
        self.runtime = runtime

    def __call__(self, executor, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
        try:
            return self.runtime.forward(
                executor.class_obj,
                x,
                timesteps,
                context,
                fps,
                padding_mask,
                **kwargs,
            )
        except PrototypeError as error:
            self.runtime.diagnostics.last_error = str(error)
            raise RuntimeError(f"Anima XDNA execution failed: {error}") from error

    def to(self, _device):
        return self

    def cleanup(self, **_kwargs):
        # ModelPatcher attachment owns the reference and cleanup lifecycle.
        pass


def _validate_patcher(model):
    required = (
        "clone",
        "add_wrapper_with_key",
        "set_attachments",
    )
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise RuntimeError(
            "unsupported ComfyUI ModelPatcher API; missing " + ", ".join(missing)
        )
    if getattr(model, "patches", None):
        raise RuntimeError(
            "Anima XDNA does not support LoRA/model weight patches yet. "
            "Attach it directly to the unpatched Base v1.0 MODEL."
        )
    transformer_options = getattr(model, "model_options", {}).get(
        "transformer_options", {}
    )
    if transformer_options.get("patches") or transformer_options.get(
        "patches_replace"
    ):
        raise RuntimeError(
            "Anima XDNA cannot silently ignore active transformer patches"
        )
    diffusion_model = getattr(getattr(model, "model", None), "diffusion_model", None)
    if diffusion_model is None:
        raise RuntimeError("MODEL does not expose model.diffusion_model")
    if type(diffusion_model).__name__ != "Anima":
        raise RuntimeError(
            "Load/Attach Anima XDNA Model requires a ComfyUI Anima MODEL"
        )


class LoadAttachAnimaXDNAModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "checkpoint": (
                    "STRING",
                    {"default": "path/to/Anima-Base-v1.0/transformer/diffusion_pytorch_model.safetensors"},
                ),
                "rebuild_cache": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "cache_dir": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "runtime_status")
    FUNCTION = "attach"
    CATEGORY = "Anima XDNA 2"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return time.time_ns()

    def attach(self, model, checkpoint, rebuild_cache, cache_dir=""):
        _validate_patcher(model)
        path = Path(checkpoint)
        if not path.is_file():
            raise RuntimeError(f"Anima checkpoint does not exist: {path}")
        patched = model.clone()
        runtime = SharedRuntime(
            path,
            Path(cache_dir) if cache_dir.strip() else None,
            rebuild_cache,
        )
        attachment = RuntimeAttachment(runtime)
        patched.set_attachments(ATTACHMENT_KEY, attachment)
        try:
            import comfy.patcher_extension
        except ImportError as error:
            attachment.cleanup()
            raise RuntimeError(
                "ComfyUI patcher_extension is unavailable; use a supported "
                f"ComfyUI checkout (validated commit {SUPPORTED_COMFY_COMMIT})"
            ) from error
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
            AnimaXDNADiffusionWrapper(runtime),
        )
        return patched, json.dumps(runtime.snapshot(), indent=2)


class AnimaXDNARuntimeStatus:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "runtime_status")
    OUTPUT_NODE = True
    FUNCTION = "status"
    CATEGORY = "Anima XDNA 2"

    def status(self, model):
        attachment = (
            model.get_attachment(ATTACHMENT_KEY)
            if hasattr(model, "get_attachment")
            else None
        )
        if not isinstance(attachment, RuntimeAttachment):
            return model, json.dumps(
                {"state": "not attached", "error": "MODEL has no Anima XDNA runtime"},
                indent=2,
            )
        return model, json.dumps(attachment.runtime.snapshot(), indent=2)


class AnimaXDNAUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "status")
    OUTPUT_NODE = True
    FUNCTION = "unload"
    CATEGORY = "Anima XDNA 2"

    def unload(self, model):
        attachment = (
            model.get_attachment(ATTACHMENT_KEY)
            if hasattr(model, "get_attachment")
            else None
        )
        if isinstance(attachment, RuntimeAttachment):
            attachment.cleanup()
            return model, "Anima XDNA runtime reference released"
        return model, "MODEL had no Anima XDNA runtime"


NODE_CLASS_MAPPINGS = {
    "LoadAttachAnimaXDNAModel": LoadAttachAnimaXDNAModel,
    "AnimaXDNARuntimeStatus": AnimaXDNARuntimeStatus,
    "AnimaXDNAUnload": AnimaXDNAUnload,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadAttachAnimaXDNAModel": "Load/Attach Anima XDNA Model",
    "AnimaXDNARuntimeStatus": "Anima XDNA Runtime Status",
    "AnimaXDNAUnload": "Unload Anima XDNA Runtime",
}
