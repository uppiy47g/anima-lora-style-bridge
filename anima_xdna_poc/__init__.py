"""Isolated Anima Linear and transformer-block proof of concept for XDNA 2."""

from .block import (
    BlockInputs,
    BlockResult,
    deterministic_block_inputs,
    run_cpu_block,
    run_xdna_block,
)
from .block_checkpoint import (
    AnimaBlockConfig,
    AnimaBlockWeights,
    load_block_config,
    load_block_weights,
)
from .checkpoint import LinearWeight, discover_linear_weight
from .chain import AnimaXDNAChainRuntime, ChainResult, ChainValidation
from .linear import PreparedLinear, cpu_linear, deterministic_input, prepare_linear
from .weight_cache import CacheStatus, PackedWeightCache

__all__ = [
    "AnimaBlockConfig",
    "AnimaBlockWeights",
    "AnimaXDNAChainRuntime",
    "BlockInputs",
    "BlockResult",
    "ChainResult",
    "ChainValidation",
    "LinearWeight",
    "PreparedLinear",
    "CacheStatus",
    "PackedWeightCache",
    "cpu_linear",
    "deterministic_block_inputs",
    "deterministic_input",
    "discover_linear_weight",
    "load_block_config",
    "load_block_weights",
    "prepare_linear",
    "run_cpu_block",
    "run_xdna_block",
]
