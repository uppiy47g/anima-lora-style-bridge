"""Canonical Anima block keys across Diffusers and native ComfyUI checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping

import torch

from .block_checkpoint import LINEAR_SHAPES, NORM_SHAPES
from .errors import UnsupportedTensor


NATIVE_NAMES = {
    "norm1.linear_1.weight": "adaln_modulation_self_attn.1.weight",
    "norm1.linear_2.weight": "adaln_modulation_self_attn.2.weight",
    "attn1.to_q.weight": "self_attn.q_proj.weight",
    "attn1.to_k.weight": "self_attn.k_proj.weight",
    "attn1.to_v.weight": "self_attn.v_proj.weight",
    "attn1.to_out.0.weight": "self_attn.output_proj.weight",
    "attn1.norm_q.weight": "self_attn.q_norm.weight",
    "attn1.norm_k.weight": "self_attn.k_norm.weight",
    "norm2.linear_1.weight": "adaln_modulation_cross_attn.1.weight",
    "norm2.linear_2.weight": "adaln_modulation_cross_attn.2.weight",
    "attn2.to_q.weight": "cross_attn.q_proj.weight",
    "attn2.to_k.weight": "cross_attn.k_proj.weight",
    "attn2.to_v.weight": "cross_attn.v_proj.weight",
    "attn2.to_out.0.weight": "cross_attn.output_proj.weight",
    "attn2.norm_q.weight": "cross_attn.q_norm.weight",
    "attn2.norm_k.weight": "cross_attn.k_norm.weight",
    "norm3.linear_1.weight": "adaln_modulation_mlp.1.weight",
    "norm3.linear_2.weight": "adaln_modulation_mlp.2.weight",
    "ff.net.0.proj.weight": "mlp.layer1.weight",
    "ff.net.2.weight": "mlp.layer2.weight",
}

CANONICAL_NAMES = tuple(LINEAR_SHAPES) + tuple(NORM_SHAPES)
_DTYPE_NAMES = {
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
}
VALIDATED_VARIANT_FINGERPRINTS = {
    "083d6f88949dec38ec35579dfd16ae96f4d453074b4fdfb6c0ed965c9cd76757": (
        "Base v1.0"
    ),
    "066b4281037504b1b7200ecd65b4182fc765ed882ecd5ca650db7308246a8dee": (
        "Turbo V1.1"
    ),
}


@dataclass(frozen=True)
class CheckpointSchema:
    name: str
    canonical_to_source: Mapping[str, str]


def canonical_keys(blocks: range = range(28)) -> list[str]:
    return [
        f"transformer_blocks.{block}.{name}"
        for block in blocks
        for name in CANONICAL_NAMES
    ]


def _mapping_for_diffusers(blocks: range) -> dict[str, str]:
    return {key: key for key in canonical_keys(blocks)}


def _mapping_for_native(prefix: str, blocks: range) -> dict[str, str]:
    return {
        f"transformer_blocks.{block}.{name}": (
            f"{prefix}blocks.{block}.{NATIVE_NAMES[name]}"
        )
        for block in blocks
        for name in CANONICAL_NAMES
    }


def detect_checkpoint_schema(
    available_keys: Iterable[str],
    blocks: range = range(28),
) -> CheckpointSchema:
    available = set(available_keys)
    candidates = (
        ("diffusers-transformer", _mapping_for_diffusers(blocks)),
        ("native-net", _mapping_for_native("net.", blocks)),
        (
            "native-model-diffusion-model",
            _mapping_for_native("model.diffusion_model.", blocks),
        ),
        ("comfyui-anima-module", _mapping_for_native("", blocks)),
        ("comfyui-model-wrapper", _mapping_for_native("diffusion_model.", blocks)),
    )
    matches = [
        CheckpointSchema(name, mapping)
        for name, mapping in candidates
        if all(source_key in available for source_key in mapping.values())
    ]
    if len(matches) != 1:
        names = ", ".join(match.name for match in matches) or "none"
        raise UnsupportedTensor(
            "checkpoint does not match exactly one supported Anima key schema; "
            f"matches={names}"
        )
    return matches[0]


def canonical_block_fingerprint(records: Iterable[Mapping]) -> str:
    normalized = [
        {
            "key": record["key"],
            "dtype": record["dtype"],
            "shape": list(record["shape"]),
            "sha256": record["sha256"],
        }
        for record in records
    ]
    normalized.sort(key=lambda record: record["key"])
    canonical = json.dumps(
        normalized, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def validated_variant(block_fingerprint: str) -> str:
    return VALIDATED_VARIANT_FINGERPRINTS.get(
        block_fingerprint, "matched Anima checkpoint (unclassified)"
    )


def fingerprint_model_blocks(diffusion_model) -> tuple[str, str]:
    state_dict = diffusion_model.state_dict()
    schema = detect_checkpoint_schema(state_dict.keys())
    records = []
    for canonical_key in canonical_keys():
        source_key = schema.canonical_to_source[canonical_key]
        tensor = state_dict[source_key]
        if not isinstance(tensor, torch.Tensor):
            raise UnsupportedTensor(
                f"MODEL tensor {source_key!r} is not a torch.Tensor"
            )
        dtype = _DTYPE_NAMES.get(tensor.dtype)
        if dtype is None:
            raise UnsupportedTensor(
                f"MODEL tensor {source_key!r} has unsupported dtype {tensor.dtype}"
            )
        contiguous = tensor.detach().to(device="cpu").contiguous()
        if contiguous.dtype == torch.float32:
            contiguous = contiguous.to(torch.bfloat16)
            dtype = "BF16"
        raw = contiguous.view(torch.uint8).numpy()
        records.append(
            {
                "key": canonical_key,
                "dtype": dtype,
                "shape": list(contiguous.shape),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return canonical_block_fingerprint(records), schema.name
