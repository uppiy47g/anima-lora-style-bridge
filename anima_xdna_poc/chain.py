"""Resident multi-block Anima execution for future ComfyUI integration."""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

import torch

from .block import BlockInputs, BlockResult, run_cpu_block, run_xdna_block
from .block_checkpoint import (
    AnimaBlockConfig,
    AnimaBlockWeights,
    LINEAR_SHAPES,
    load_block_config,
    load_block_weights,
)
from .errors import NumericalMismatch, UnsupportedTensor
from .xdna import ResidentXDNASession
from .weight_cache import CacheStatus, PackedWeightCache


@dataclass(frozen=True)
class ChainBlockProfile:
    index: int
    wall_ms: float
    result: BlockResult


@dataclass(frozen=True)
class ChainResult:
    output: torch.Tensor
    blocks: tuple[ChainBlockProfile, ...]

    @property
    def wall_ms(self) -> float:
        return sum(block.wall_ms for block in self.blocks)

    @property
    def dispatch_count(self) -> int:
        return sum(block.result.dispatch_count for block in self.blocks)


@dataclass(frozen=True)
class BlockError:
    index: int
    max_abs_error: float
    mean_abs_error: float
    rms_error: float
    reference_rms: float
    normalized_rms_error: float


@dataclass(frozen=True)
class ChainValidation:
    cpu: ChainResult
    xdna: ChainResult
    errors: tuple[BlockError, ...]


class AnimaXDNAChainRuntime:
    """Load one checkpoint/config and execute block ranges with resident XRT BOs."""

    def __init__(
        self,
        checkpoint: Path,
        config: Optional[Path] = None,
        cache_host_weights: bool = True,
        weight_cache: bool = True,
        cache_dir: Optional[Path] = None,
        rebuild_cache: bool = False,
    ):
        self.checkpoint = Path(checkpoint)
        self.config: AnimaBlockConfig = load_block_config(
            config or self.checkpoint.parent
        )
        self.cache_host_weights = cache_host_weights
        self.weight_cache_enabled = weight_cache
        self.cache_dir = cache_dir
        self.rebuild_cache = rebuild_cache
        self._weights: dict[int, AnimaBlockWeights] = {}
        self._session: Optional[ResidentXDNASession] = None
        self._packed_cache: Optional[PackedWeightCache] = None
        self.cache_status: Optional[CacheStatus] = None

    def __enter__(self):
        if self._session is not None:
            raise RuntimeError("AnimaXDNAChainRuntime is already open")
        self._session = ResidentXDNASession()
        self._session.__enter__()
        return self

    def close(self):
        if self._session is not None:
            self._session.__exit__(None, None, None)
            self._session = None
        if self._packed_cache is not None:
            self._packed_cache.close()
            self._packed_cache = None
        self._weights.clear()

    def __exit__(self, exc_type, exc_value, traceback):
        if self._session is not None:
            self._session.__exit__(exc_type, exc_value, traceback)
            self._session = None
        if self._packed_cache is not None:
            self._packed_cache.close()
            self._packed_cache = None
        self._weights.clear()

    def _ensure_packed_cache(self, start: int, end: int) -> None:
        if not self.weight_cache_enabled:
            return
        if self._packed_cache is not None:
            return
        cache = PackedWeightCache(
            self.checkpoint,
            self.config,
            0,
            28,
            self.cache_dir,
        )
        self.cache_status = cache.open(self.rebuild_cache)
        self._packed_cache = cache

    def weights(self, index: int, source: bool = False) -> AnimaBlockWeights:
        if not 0 <= index < 28:
            raise UnsupportedTensor(f"block index must be in [0, 27], got {index}")
        if self._packed_cache is not None and not source:
            return self._packed_cache.block_weights(index)
        loaded = self._weights.get(index)
        if loaded is None:
            loaded = load_block_weights(self.checkpoint, self.config, index)
            if self.cache_host_weights:
                self._weights[index] = loaded
        return loaded

    def _retain_resident_metadata(
        self,
        index: int,
        weights: AnimaBlockWeights,
    ) -> None:
        tensors = {}
        for name, tensor in weights.tensors.items():
            if name in LINEAR_SHAPES:
                tensors[name] = torch.empty(
                    tensor.shape,
                    dtype=tensor.dtype,
                    device="meta",
                )
            else:
                tensors[name] = tensor.clone()
        self._weights[index] = AnimaBlockWeights(weights.prefix, tensors)

    @staticmethod
    def _next_inputs(inputs: BlockInputs, hidden: torch.Tensor) -> BlockInputs:
        return BlockInputs(
            hidden,
            inputs.encoder_hidden_states,
            inputs.embedded_timestep,
            inputs.temb,
            inputs.image_rotary_emb,
            inputs.attention_mask,
        )

    @staticmethod
    def _validate_range(start: int, end: int) -> None:
        if not 0 <= start < end <= 28:
            raise UnsupportedTensor(
                f"block range must satisfy 0 <= start < end <= 28, got [{start}, {end})"
            )

    def run_range(
        self,
        inputs: BlockInputs,
        start: int = 0,
        end: int = 28,
    ) -> ChainResult:
        self._validate_range(start, end)
        if self._session is None:
            raise RuntimeError("AnimaXDNAChainRuntime must be entered before execution")
        self._ensure_packed_cache(start, end)
        hidden = inputs.hidden_states
        profiles = []
        for index in range(start, end):
            block_inputs = self._next_inputs(inputs, hidden)
            weights = self.weights(index)
            started = time.perf_counter()
            result = run_xdna_block(
                self.config,
                weights,
                block_inputs,
                session=self._session,
                resident_key=f"block{index}",
                packed_cache=self._packed_cache,
                block_index=index,
            )
            wall_ms = (time.perf_counter() - started) * 1000
            hidden = result.output
            profiles.append(ChainBlockProfile(index, wall_ms, result))
            if self._packed_cache is None:
                self._retain_resident_metadata(index, weights)
        return ChainResult(hidden, tuple(profiles))

    def run_cpu_range(
        self,
        inputs: BlockInputs,
        start: int = 0,
        end: int = 28,
    ) -> ChainResult:
        self._validate_range(start, end)
        hidden = inputs.hidden_states
        profiles = []
        for index in range(start, end):
            block_inputs = self._next_inputs(inputs, hidden)
            started = time.perf_counter()
            result = run_cpu_block(
                self.config, self.weights(index, source=True), block_inputs
            )
            wall_ms = (time.perf_counter() - started) * 1000
            hidden = result.output
            profiles.append(ChainBlockProfile(index, wall_ms, result))
        return ChainResult(hidden, tuple(profiles))

    def validate_range(
        self,
        inputs: BlockInputs,
        start: int = 0,
        end: int = 28,
        max_normalized_rms_error: float = 5e-2,
    ) -> ChainValidation:
        self._validate_range(start, end)
        if self._session is None:
            raise RuntimeError("AnimaXDNAChainRuntime must be entered before execution")
        self._ensure_packed_cache(start, end)
        cpu_hidden = inputs.hidden_states
        xdna_hidden = inputs.hidden_states
        cpu_profiles = []
        xdna_profiles = []
        errors = []
        for index in range(start, end):
            cpu_inputs = self._next_inputs(inputs, cpu_hidden)
            xdna_inputs = self._next_inputs(inputs, xdna_hidden)
            weights = self.weights(index, source=True)
            started = time.perf_counter()
            cpu_result = run_cpu_block(self.config, weights, cpu_inputs)
            cpu_ms = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            xdna_result = run_xdna_block(
                self.config,
                weights,
                xdna_inputs,
                session=self._session,
                resident_key=f"block{index}",
                packed_cache=self._packed_cache,
                block_index=index,
            )
            xdna_ms = (time.perf_counter() - started) * 1000
            cpu_hidden = cpu_result.output
            xdna_hidden = xdna_result.output
            difference = (xdna_hidden.float() - cpu_hidden.float()).abs()
            rms = difference.square().mean().sqrt().item()
            reference_rms = cpu_hidden.float().square().mean().sqrt().item()
            normalized = rms / max(reference_rms, 1e-12)
            errors.append(
                BlockError(
                    index,
                    difference.max().item(),
                    difference.mean().item(),
                    rms,
                    reference_rms,
                    normalized,
                )
            )
            cpu_profiles.append(ChainBlockProfile(index, cpu_ms, cpu_result))
            xdna_profiles.append(ChainBlockProfile(index, xdna_ms, xdna_result))
            if self._packed_cache is None:
                self._retain_resident_metadata(index, weights)
        worst = max(errors, key=lambda item: item.normalized_rms_error)
        if worst.normalized_rms_error > max_normalized_rms_error:
            raise NumericalMismatch(
                f"28-block normalized RMS error exceeded "
                f"{max_normalized_rms_error:.4g} at block {worst.index}: "
                f"{worst.normalized_rms_error:.6g} "
                f"(max_abs={worst.max_abs_error:.6g})"
            )
        return ChainValidation(
            ChainResult(cpu_hidden, tuple(cpu_profiles)),
            ChainResult(xdna_hidden, tuple(xdna_profiles)),
            tuple(errors),
        )
