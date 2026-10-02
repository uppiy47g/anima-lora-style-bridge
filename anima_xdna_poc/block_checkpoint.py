"""Lazy loading and validation for one Diffusers Anima/Cosmos block."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open

from .errors import UnsupportedTensor


@dataclass(frozen=True)
class AnimaBlockConfig:
    hidden_size: int
    num_heads: int
    head_dim: int
    context_dim: int
    adaln_dim: int
    mlp_dim: int
    patch_size: tuple[int, int, int]
    rope_scale: tuple[float, float, float]
    max_size: tuple[int, int, int]


@dataclass(frozen=True)
class AnimaBlockWeights:
    prefix: str
    tensors: Mapping[str, torch.Tensor]

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.tensors[name]


LINEAR_SHAPES = {
    "norm1.linear_1.weight": ("adaln", "hidden"),
    "norm1.linear_2.weight": ("modulation", "adaln"),
    "attn1.to_q.weight": ("hidden", "hidden"),
    "attn1.to_k.weight": ("hidden", "hidden"),
    "attn1.to_v.weight": ("hidden", "hidden"),
    "attn1.to_out.0.weight": ("hidden", "hidden"),
    "norm2.linear_1.weight": ("adaln", "hidden"),
    "norm2.linear_2.weight": ("modulation", "adaln"),
    "attn2.to_q.weight": ("hidden", "hidden"),
    "attn2.to_k.weight": ("hidden", "context"),
    "attn2.to_v.weight": ("hidden", "context"),
    "attn2.to_out.0.weight": ("hidden", "hidden"),
    "norm3.linear_1.weight": ("adaln", "hidden"),
    "norm3.linear_2.weight": ("modulation", "adaln"),
    "ff.net.0.proj.weight": ("mlp", "hidden"),
    "ff.net.2.weight": ("hidden", "mlp"),
}

NORM_SHAPES = {
    "attn1.norm_q.weight": "head",
    "attn1.norm_k.weight": "head",
    "attn2.norm_q.weight": "head",
    "attn2.norm_k.weight": "head",
}


def load_block_config(path: Path) -> AnimaBlockConfig:
    path = Path(path)
    if path.is_dir():
        path = path / "config.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise UnsupportedTensor(f"cannot read transformer config {path}: {error}") from error
    if data.get("_class_name", "CosmosTransformer3DModel") != "CosmosTransformer3DModel":
        raise UnsupportedTensor(
            f"unsupported transformer class {data.get('_class_name')!r}; "
            "expected CosmosTransformer3DModel"
        )
    if data.get("img_context_dim_in") is not None:
        raise UnsupportedTensor("image-context Cosmos blocks are not supported by this milestone")
    if data.get("use_crossattn_projection", False):
        raise UnsupportedTensor(
            "Cosmos cross-attention projection is not supported by this milestone"
        )

    required = (
        "num_attention_heads",
        "attention_head_dim",
        "adaln_lora_dim",
        "mlp_ratio",
        "patch_size",
        "rope_scale",
        "max_size",
    )
    missing = [name for name in required if name not in data]
    if missing:
        raise UnsupportedTensor(
            f"transformer config is missing required Cosmos fields: {', '.join(missing)}"
        )
    context_value = data.get(
        "encoder_hidden_states_channels",
        data.get("cross_attention_dim"),
    )
    if context_value is None:
        raise UnsupportedTensor(
            "transformer config is missing required Cosmos field "
            "encoder_hidden_states_channels"
        )
    heads = int(data["num_attention_heads"])
    head_dim = int(data["attention_head_dim"])
    hidden = heads * head_dim
    config = AnimaBlockConfig(
        hidden_size=hidden,
        num_heads=heads,
        head_dim=head_dim,
        context_dim=int(context_value),
        adaln_dim=int(data["adaln_lora_dim"]),
        mlp_dim=int(hidden * float(data["mlp_ratio"])),
        patch_size=tuple(int(value) for value in data["patch_size"]),
        rope_scale=tuple(float(value) for value in data["rope_scale"]),
        max_size=tuple(int(value) for value in data["max_size"]),
    )
    if config.head_dim % 2:
        raise UnsupportedTensor(f"attention_head_dim must be even, got {config.head_dim}")
    return config


def infer_native_block_config(checkpoint_path: Path) -> AnimaBlockConfig:
    from .checkpoint_schema import detect_checkpoint_schema

    path = Path(checkpoint_path)
    try:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            schema = detect_checkpoint_schema(checkpoint.keys(), range(1))
            mapping = schema.canonical_to_source
            shapes = {
                key: tuple(checkpoint.get_slice(source).get_shape())
                for key, source in mapping.items()
            }
    except UnsupportedTensor:
        raise
    except Exception as error:
        raise UnsupportedTensor(
            f"cannot inspect native Anima checkpoint {path}: {error}"
        ) from error
    expected = {
        "transformer_blocks.0.attn1.to_q.weight": (2048, 2048),
        "transformer_blocks.0.attn1.norm_q.weight": (128,),
        "transformer_blocks.0.attn2.to_k.weight": (2048, 1024),
        "transformer_blocks.0.norm1.linear_1.weight": (256, 2048),
        "transformer_blocks.0.norm1.linear_2.weight": (6144, 256),
        "transformer_blocks.0.ff.net.0.proj.weight": (8192, 2048),
        "transformer_blocks.0.ff.net.2.weight": (2048, 8192),
    }
    mismatches = [
        f"{key}: {shapes.get(key)} != {shape}"
        for key, shape in expected.items()
        if shapes.get(key) != shape
    ]
    if mismatches:
        raise UnsupportedTensor(
            "native checkpoint is not the validated Anima 2B architecture: "
            + "; ".join(mismatches)
        )
    return AnimaBlockConfig(
        hidden_size=2048,
        num_heads=16,
        head_dim=128,
        context_dim=1024,
        adaln_dim=256,
        mlp_dim=8192,
        patch_size=(1, 2, 2),
        rope_scale=(1.0, 4.0, 4.0),
        max_size=(128, 240, 240),
    )


def load_checkpoint_config(
    checkpoint_path: Path,
    config_path: Path | None = None,
) -> AnimaBlockConfig:
    if config_path is not None:
        return load_block_config(config_path)
    adjacent = Path(checkpoint_path).parent / "config.json"
    if adjacent.is_file():
        return load_block_config(adjacent)
    return infer_native_block_config(checkpoint_path)


def _dimension_map(config: AnimaBlockConfig) -> dict[str, int]:
    return {
        "hidden": config.hidden_size,
        "head": config.head_dim,
        "context": config.context_dim,
        "adaln": config.adaln_dim,
        "modulation": 3 * config.hidden_size,
        "mlp": config.mlp_dim,
    }


def load_block_weights(
    checkpoint_path: Path,
    config: AnimaBlockConfig,
    block_index: int = 0,
) -> AnimaBlockWeights:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise UnsupportedTensor(f"checkpoint does not exist or is not a file: {path}")
    from .checkpoint_schema import detect_checkpoint_schema

    prefix = f"transformer_blocks.{block_index}."
    dimensions = _dimension_map(config)
    expected = {
        **{
            name: (dimensions[out_name], dimensions[in_name])
            for name, (out_name, in_name) in LINEAR_SHAPES.items()
        },
        **{name: (dimensions[size_name],) for name, size_name in NORM_SHAPES.items()},
    }
    tensors: dict[str, torch.Tensor] = {}
    try:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            available = set(checkpoint.keys())
            schema = detect_checkpoint_schema(available, range(block_index, block_index + 1))
            for name, shape in expected.items():
                canonical_key = prefix + name
                source_key = schema.canonical_to_source[canonical_key]
                tensor = checkpoint.get_tensor(source_key)
                if tuple(tensor.shape) != shape:
                    raise UnsupportedTensor(
                        f"{source_key!r} has shape {tuple(tensor.shape)}, expected {shape}"
                    )
                if tensor.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                    raise UnsupportedTensor(
                        f"{source_key!r} has unsupported dtype {tensor.dtype}"
                    )
                tensors[name] = tensor.to(torch.bfloat16).contiguous()
    except UnsupportedTensor:
        raise
    except Exception as error:
        raise UnsupportedTensor(f"cannot read safetensors checkpoint {path}: {error}") from error
    return AnimaBlockWeights(prefix, tensors)
