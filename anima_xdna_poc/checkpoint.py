"""Lazy discovery of one representative Anima Linear weight."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
from safetensors import safe_open

from .errors import UnsupportedTensor


CANONICAL_ALIASES = (
    "diffusion_model.blocks.0.self_attn.q_proj.weight",
    "model.diffusion_model.blocks.0.self_attn.q_proj.weight",
    "blocks.0.self_attn.q_proj.weight",
    "transformer_blocks.0.attn1.to_q.weight",
    "diffusion_model.transformer_blocks.0.attn1.to_q.weight",
    "diffusion_model.transformer_blocks.0.self_attn.q_proj.weight",
    "transformer_blocks.0.self_attn.q_proj.weight",
)
SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


@dataclass(frozen=True)
class LinearWeight:
    key: str
    tensor: torch.Tensor

    @property
    def out_features(self) -> int:
        return self.tensor.shape[0]

    @property
    def in_features(self) -> int:
        return self.tensor.shape[1]


def select_weight_key(keys: Iterable[str], aliases: Sequence[str] = CANONICAL_ALIASES) -> str:
    available = set(keys)
    for alias in aliases:
        if alias in available:
            return alias

    suffixes = (
        "blocks.0.self_attn.q_proj.weight",
        "transformer_blocks.0.self_attn.q_proj.weight",
        "transformer_blocks.0.attn1.to_q.weight",
    )
    matches = sorted(
        key for key in available if any(key.endswith(suffix) for suffix in suffixes)
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise UnsupportedTensor(
            "multiple candidate q_proj weights were found; pass --key explicitly: "
            + ", ".join(matches)
        )
    raise UnsupportedTensor(
        "no block-0 self-attention q_proj weight was found. Tried canonical aliases: "
        + ", ".join(aliases)
    )


def validate_weight(key: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 2:
        raise UnsupportedTensor(
            f"{key!r} must be a rank-2 Linear weight [out_features, in_features], "
            f"but has shape {tuple(tensor.shape)}"
        )
    if tensor.dtype not in SUPPORTED_DTYPES:
        supported = ", ".join(str(dtype).removeprefix("torch.") for dtype in SUPPORTED_DTYPES)
        raise UnsupportedTensor(
            f"{key!r} has dtype {tensor.dtype}; supported checkpoint dtypes are {supported}"
        )
    if min(tensor.shape) <= 0:
        raise UnsupportedTensor(f"{key!r} has an empty dimension: {tuple(tensor.shape)}")


def discover_linear_weight(path: Path, key: Optional[str] = None) -> LinearWeight:
    path = Path(path)
    if not path.is_file():
        raise UnsupportedTensor(f"checkpoint does not exist or is not a file: {path}")
    try:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            selected = key or select_weight_key(checkpoint.keys())
            if selected not in checkpoint.keys():
                raise UnsupportedTensor(f"requested weight key was not found: {selected}")
            tensor = checkpoint.get_tensor(selected)
    except UnsupportedTensor:
        raise
    except Exception as error:
        raise UnsupportedTensor(f"cannot read safetensors checkpoint {path}: {error}") from error

    validate_weight(selected, tensor)
    return LinearWeight(selected, tensor)
