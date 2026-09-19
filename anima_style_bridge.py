#!/usr/bin/env python3
"""
SD 1.5/SDXL LoRA -> Anima LoRA experimental style bridge.

Features:
  * Interpolates source transformer blocks across all 28 Anima blocks.
  * Blends collisions as weighted low-rank deltas instead of overwriting them.
  * Supports optional Procrustes/CCA projections fitted from paired activations.
  * Preserves each source delta's Frobenius norm before block blending.

Calibration NPZ format:
  <name>.source = [samples, source_dim]
  <name>.target = [samples, target_dim]

Projection names used by conversion, from most to least specific:
  self_attn.q_proj.in.projection
  in.320x2048.projection

Use the same convention with "out" for output activations.
"""

import argparse
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


VERSION = "3.1.0"
ANIMA_BLOCKS = 28
ANIMA_DIMS = {
    "cross_attn.k_proj": (1024, 2048),
    "cross_attn.q_proj": (2048, 2048),
    "cross_attn.v_proj": (1024, 2048),
    "cross_attn.output_proj": (2048, 2048),
    "self_attn.k_proj": (2048, 2048),
    "self_attn.q_proj": (2048, 2048),
    "self_attn.v_proj": (2048, 2048),
    "self_attn.output_proj": (2048, 2048),
    "mlp.layer1": (2048, 8192),
    "mlp.layer2": (8192, 2048),
}
SD_LAYER_MAP = {
    "attn1_to_k": "self_attn.k_proj",
    "attn1_to_q": "self_attn.q_proj",
    "attn1_to_v": "self_attn.v_proj",
    "attn1_to_out_0": "self_attn.output_proj",
    "attn2_to_k": "cross_attn.k_proj",
    "attn2_to_q": "cross_attn.q_proj",
    "attn2_to_v": "cross_attn.v_proj",
    "attn2_to_out_0": "cross_attn.output_proj",
    "ff_net_0_proj": "mlp.layer1",
    "ff_net_2": "mlp.layer2",
}
PAIR_SUFFIXES = {
    ".lora_down.weight": "down",
    ".lora_up.weight": "up",
    ".alpha": "alpha",
}
DIFFUSERS_SD_KEY_RE = re.compile(
    r"^lora_unet_"
    r"(?P<section>down_blocks_\d+|up_blocks_\d+|mid_block)_"
    r"attentions_(?P<attention>\d+)_transformer_blocks_(?P<transformer>\d+)_"
    r"(?P<layer>attn[12]_to_[qkv]|attn[12]_to_out_0|ff_net_0_proj|ff_net_2)$"
)
SGM_SD_KEY_RE = re.compile(
    r"^lora_unet_"
    r"(?P<section>input_blocks|output_blocks|middle_block)_"
    r"(?P<block>\d+)(?:_(?P<attention>\d+))?_"
    r"transformer_blocks_(?P<transformer>\d+)_"
    r"(?P<layer>attn[12]_to_[qkv]|attn[12]_to_out_0|ff_net_0_proj|ff_net_2)$"
)
SOURCE_CAPTURE_KEY_RE = re.compile(
    r"^diffusion_model\."
    r"(?P<section>input_blocks|output_blocks|middle_block)\."
    r"(?P<block>\d+)(?:\.(?P<attention>\d+))?"
    r"\.transformer_blocks\.(?P<transformer>\d+)\."
    r"(?P<layer>attn[12]\.to_[qkv]|attn[12]\.to_out\.0)"
    r"\.(?P<side>input|output)$"
)
TARGET_CAPTURE_KEY_RE = re.compile(
    r"^(?P<prefix>diffusion_model\.blocks\.(?P<block>\d+)\."
    r"(?P<layer>(?:self|cross)_attn\."
    r"(?:[qkv]_proj|output_proj)|mlp\.layer[12]))"
    r"\.(?P<side>input|output)$"
)
CAPTURE_LAYER_MAP = {
    "attn1.to_k": "self_attn.k_proj",
    "attn1.to_q": "self_attn.q_proj",
    "attn1.to_v": "self_attn.v_proj",
    "attn1.to_out.0": "self_attn.output_proj",
    "attn2.to_k": "cross_attn.k_proj",
    "attn2.to_q": "cross_attn.q_proj",
    "attn2.to_v": "cross_attn.v_proj",
    "attn2.to_out.0": "cross_attn.output_proj",
}


def tensor_2d(tensor):
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim >= 2 and all(size == 1 for size in tensor.shape[2:]):
        return tensor.reshape(tensor.shape[0], tensor.shape[1])
    raise ValueError(f"Unsupported LoRA tensor shape: {tuple(tensor.shape)}")


def delta_norm(a, b):
    a = a.float()
    b = b.float()
    gram_a = a @ a.T
    gram_b = b.T @ b
    return torch.sqrt(torch.clamp((gram_b * gram_a.T).sum(), min=0.0))


def resample_axis(matrix, size, axis):
    if matrix.shape[axis] == size:
        return matrix.float()
    if axis == 1:
        values = matrix.float().unsqueeze(0)
        return F.interpolate(values, size=size, mode="linear", align_corners=True).squeeze(0)
    values = matrix.float().T.unsqueeze(0)
    return F.interpolate(values, size=size, mode="linear", align_corners=True).squeeze(0).T


class ProjectionStore:
    def __init__(self, path=None):
        self.projections = {}
        self.inverses = {}
        if path:
            with np.load(path) as data:
                self.projections = {
                    key: torch.from_numpy(data[key]).float()
                    for key in data.files
                    if key.endswith(".projection")
                }

    def find(self, layer, side, source_dim, target_dim):
        candidates = (
            f"{layer}.{side}.projection",
            f"{side}.{source_dim}x{target_dim}.projection",
        )
        for key in candidates:
            projection = self.projections.get(key)
            if projection is None:
                continue
            expected = (target_dim, source_dim)
            if tuple(projection.shape) != expected:
                raise ValueError(
                    f"{key} has shape {tuple(projection.shape)}, expected {expected}"
                )
            return projection
        return None

    def inverse(self, projection):
        key = projection.data_ptr()
        if key not in self.inverses:
            self.inverses[key] = torch.linalg.pinv(projection)
        return self.inverses[key]


def project_factors(a, b, target_in, target_out, layer, projections):
    a = tensor_2d(a).float()
    b = tensor_2d(b).float()
    source_norm = delta_norm(a, b)

    p_in = projections.find(layer, "in", a.shape[1], target_in)
    p_out = projections.find(layer, "out", b.shape[0], target_out)
    projected_a = (
        a @ projections.inverse(p_in)
        if p_in is not None
        else resample_axis(a, target_in, 1)
    )
    projected_b = p_out @ b if p_out is not None else resample_axis(b, target_out, 0)

    projected_norm = delta_norm(projected_a, projected_b)
    if source_norm > 0 and projected_norm > 0:
        projected_b *= source_norm / projected_norm
    return projected_a, projected_b


def combine_factors(terms, rank, dtype):
    """Compress weighted deltas and retain their combined Frobenius magnitude."""
    a_cat = torch.cat([a for _, a, _ in terms], dim=0)
    b_cat = torch.cat([b * weight for weight, _, b in terms], dim=1)
    combined_norm = delta_norm(a_cat, b_cat)

    q_b, r_b = torch.linalg.qr(b_cat.float(), mode="reduced")
    q_a, r_a = torch.linalg.qr(a_cat.float().T, mode="reduced")
    core = r_b @ r_a.T
    u, singular, vh = torch.linalg.svd(core, full_matrices=False)
    used_rank = min(rank, singular.numel())
    scale = singular[:used_rank].clamp_min(0).sqrt()
    lora_b = (q_b @ u[:, :used_rank]) * scale
    lora_a = scale[:, None] * (vh[:used_rank] @ q_a.T)
    compressed_norm = delta_norm(lora_a, lora_b)
    if combined_norm > 0 and compressed_norm > 0:
        lora_b *= combined_norm / compressed_norm
    return lora_a.to(dtype).contiguous(), lora_b.to(dtype).contiguous()


def source_block_order(section, block, attention, transformer):
    if section == "input_blocks":
        return (0, block, attention, transformer)
    if section.startswith("down_blocks_"):
        stage = int(section.rsplit("_", 1)[1])
        return (0, stage, attention, transformer)
    if section in {"mid_block", "middle_block"}:
        return (1, 0, attention, transformer)
    if section == "output_blocks":
        return (2, block, attention, transformer)
    stage = int(section.rsplit("_", 1)[1])
    return (2, stage, attention, transformer)


def parse_sd_base(base):
    match = DIFFUSERS_SD_KEY_RE.match(base)
    if match:
        section = match.group("section")
        block = int(section.rsplit("_", 1)[1]) if section != "mid_block" else 0
        return (
            section,
            block,
            int(match.group("attention")),
            int(match.group("transformer")),
            SD_LAYER_MAP[match.group("layer")],
        )

    match = SGM_SD_KEY_RE.match(base)
    if match:
        return (
            match.group("section"),
            int(match.group("block")),
            int(match.group("attention") or 0),
            int(match.group("transformer")),
            SD_LAYER_MAP[match.group("layer")],
        )
    return None


def parse_sd_groups(state):
    raw_groups = defaultdict(dict)
    for key, tensor in state.items():
        for suffix, name in PAIR_SUFFIXES.items():
            if key.endswith(suffix):
                raw_groups[key[: -len(suffix)]][name] = tensor
                break

    blocks = defaultdict(dict)
    skipped = 0
    for base, tensors in raw_groups.items():
        parsed = parse_sd_base(base)
        if parsed is None or "down" not in tensors or "up" not in tensors:
            skipped += 1
            continue
        section, stage, attention, transformer, layer = parsed
        block = (section, stage, attention, transformer)
        blocks[block][layer] = tensors
    return blocks, skipped


def parse_source_capture_key(key):
    match = SOURCE_CAPTURE_KEY_RE.match(key)
    if match is None:
        return None
    return (
        match.group("section"),
        int(match.group("block")),
        int(match.group("attention") or 0),
        int(match.group("transformer")),
        CAPTURE_LAYER_MAP[match.group("layer")],
        match.group("side"),
    )


def npz_scalar(data, key):
    if key not in data:
        return None
    value = data[key]
    return str(value.item()) if value.ndim == 0 else None


def captured_source_modules(data):
    modules = defaultdict(dict)
    for key in data.files:
        parsed = parse_source_capture_key(key)
        if parsed is None:
            continue
        section, block, attention, transformer, layer, side = parsed
        identity = (section, block, attention, transformer)
        modules[(identity, layer)][side] = key
    return modules


def source_lora_modules(state):
    blocks, _skipped = parse_sd_groups(state)
    return {
        (identity, layer): tensors
        for identity, layers in blocks.items()
        for layer, tensors in layers.items()
    }


def pair_source_capture_and_lora(captures, loras):
    pairs = []
    layers = sorted({key[1] for key in captures} & {key[1] for key in loras})
    for layer in layers:
        capture_items = sorted(
            (
                (identity, arrays)
                for (identity, item_layer), arrays in captures.items()
                if item_layer == layer and {"input", "output"} <= arrays.keys()
            ),
            key=lambda item: source_block_order(*item[0]),
        )
        lora_items = sorted(
            (
                (identity, tensors)
                for (identity, item_layer), tensors in loras.items()
                if item_layer == layer
            ),
            key=lambda item: source_block_order(*item[0]),
        )
        if not lora_items:
            continue

        capture_by_identity = dict(capture_items)
        exact = all(identity in capture_by_identity for identity, _ in lora_items)
        if exact:
            matched = [
                (identity, capture_by_identity[identity], tensors)
                for identity, tensors in lora_items
            ]
        elif len(capture_items) == len(lora_items):
            matched = [
                (capture_identity, arrays, tensors)
                for (capture_identity, arrays), (_lora_identity, tensors)
                in zip(capture_items, lora_items)
            ]
        else:
            raise ValueError(
                f"{layer}: cannot safely pair {len(lora_items)} LoRA modules with "
                f"{len(capture_items)} captured modules. Use an SGM-key LoRA for "
                "exact matching, or capture exactly the LoRA-targeted modules."
            )
        pairs.extend((layer, *item) for item in matched)
    if not pairs:
        raise ValueError("No captured SDXL modules could be paired with LoRA weights")
    return pairs


def local_lora_delta(inputs, tensors, strength):
    down = tensor_2d(tensors["down"]).float()
    up = tensor_2d(tensors["up"]).float()
    if inputs.shape[1] != down.shape[1]:
        raise ValueError(
            f"Captured input width {inputs.shape[1]} does not match "
            f"LoRA input width {down.shape[1]}"
        )
    rank = down.shape[0]
    alpha = float(tensors.get("alpha", torch.tensor(rank)).item())
    return ((inputs.float() @ down.T) @ up.T) * (strength * alpha / rank)


def prepare_v3_activations(
    source_capture_path,
    target_capture_path,
    lora_path,
    output_path,
    lora_strength,
    truncate_unpaired_tail=False,
):
    lora_state = load_file(lora_path)
    loras = source_lora_modules(lora_state)
    output = {}

    with np.load(source_capture_path) as source, np.load(
        target_capture_path
    ) as target:
        source_aggregation = npz_scalar(source, "__aggregation__")
        target_aggregation = npz_scalar(target, "__aggregation__")
        if source_aggregation != target_aggregation:
            raise ValueError(
                "Capture aggregation differs: "
                f"{source_aggregation!r} != {target_aggregation!r}"
            )
        if source_aggregation == "tokens":
            raise ValueError(
                "v3 preprocessing requires aggregation='mean'; token counts differ "
                "across SDXL and Anima layers"
            )

        captures = captured_source_modules(source)
        pairs = pair_source_capture_and_lora(captures, loras)
        source_indices = defaultdict(int)
        for layer, _identity, arrays, tensors in pairs:
            input_values = flatten_activation(source[arrays["input"]], arrays["input"])
            styled = flatten_activation(source[arrays["output"]], arrays["output"])
            delta = local_lora_delta(input_values, tensors, lora_strength)
            if styled.shape != delta.shape:
                raise ValueError(
                    f"{arrays['output']} has shape {tuple(styled.shape)}, "
                    f"but its LoRA delta has shape {tuple(delta.shape)}"
                )
            source_block = source_indices[layer]
            source_indices[layer] += 1
            prefix = f"source.{source_block}.{layer}"
            output[f"{prefix}.base"] = (styled - delta).cpu().numpy()
            output[f"{prefix}.styled"] = styled.cpu().numpy()

        target_modules = defaultdict(dict)
        for key in target.files:
            match = TARGET_CAPTURE_KEY_RE.match(key)
            if match is None:
                continue
            module = (int(match.group("block")), match.group("layer"))
            target_modules[module][match.group("side")] = key
        for (block, layer), arrays in sorted(target_modules.items()):
            if not {"input", "output"} <= arrays.keys():
                continue
            prefix = f"target.{block}.{layer}"
            output[f"{prefix}.input"] = flatten_activation(
                target[arrays["input"]], arrays["input"]
            ).cpu().numpy()
            output[f"{prefix}.base"] = flatten_activation(
                target[arrays["output"]], arrays["output"]
            ).cpu().numpy()

    source_count = sum(
        key.startswith("source.") and key.endswith(".base") for key in output
    )
    target_count = sum(
        key.startswith("target.") and key.endswith(".base") for key in output
    )
    if not target_count:
        raise ValueError("No supported Anima modules were found in the target capture")
    row_counts = {values.shape[0] for values in output.values()}
    if len(row_counts) != 1:
        if not truncate_unpaired_tail:
            raise ValueError(
                f"Prepared arrays have different row counts: {sorted(row_counts)}. "
                "Capture both models with aggregation='mean', identical sampler call "
                "counts, and identical append order. If the extra executions were "
                "appended only at the end, retry with --truncate-unpaired-tail."
            )
        paired_rows = min(row_counts)
        output = {
            key: values[:paired_rows]
            for key, values in output.items()
        }
        print(
            f"Warning: truncated unpaired tail rows {sorted(row_counts)} "
            f"to {paired_rows}"
        )
        row_counts = {paired_rows}
    np.savez_compressed(output_path, **output)
    print(
        f"Prepared {source_count} SDXL and {target_count} Anima modules "
        f"with {next(iter(row_counts))} paired rows"
    )
    print(f"LoRA strength: {lora_strength}")
    print(f"Saved: {output_path}")


def interpolation_terms(items, target_index):
    if len(items) == 1:
        return [(1.0, items[0])]
    position = target_index * (len(items) - 1) / (ANIMA_BLOCKS - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(items) - 1)
    fraction = position - lower
    if lower == upper or fraction == 0:
        return [(1.0, items[lower])]
    return [(1.0 - fraction, items[lower]), (fraction, items[upper])]


def convert_sd(input_path, output_path, rank, calibration, output_dtype):
    state = load_file(input_path)
    blocks, skipped = parse_sd_groups(state)
    if not blocks:
        raise ValueError("No supported SD 1.5/SDXL transformer LoRA layers were found")

    ordered_blocks = sorted(
        blocks,
        key=lambda block: source_block_order(*block),
    )
    by_layer = {
        layer: [(block, blocks[block][layer]) for block in ordered_blocks if layer in blocks[block]]
        for layer in ANIMA_DIMS
    }
    projections = ProjectionStore(calibration)
    output = {}
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    target_dtype = dtype_map[output_dtype]

    for target_block in range(ANIMA_BLOCKS):
        for layer, (target_in, target_out) in ANIMA_DIMS.items():
            sources = by_layer[layer]
            if not sources:
                continue
            terms = []
            for weight, (_, tensors) in interpolation_terms(sources, target_block):
                source_rank = tensor_2d(tensors["down"]).shape[0]
                alpha = float(tensors.get("alpha", torch.tensor(source_rank)).item())
                effective_weight = weight * alpha / source_rank
                a, b = project_factors(
                    tensors["down"],
                    tensors["up"],
                    target_in,
                    target_out,
                    layer,
                    projections,
                )
                terms.append((effective_weight, a, b))

            lora_a, lora_b = combine_factors(terms, rank, target_dtype)
            prefix = f"diffusion_model.blocks.{target_block}.{layer}"
            output[f"{prefix}.lora_A.weight"] = lora_a
            output[f"{prefix}.lora_B.weight"] = lora_b
            output[f"{prefix}.lora_alpha"] = torch.tensor(float(lora_a.shape[0]))

    metadata = {
        "converter": "anima_lora_style_bridge",
        "source": str(Path(input_path).name),
        "source_blocks": str(len(ordered_blocks)),
        "target_blocks": str(ANIMA_BLOCKS),
        "projection": "calibrated" if calibration else "deterministic-resample",
        "dtype": output_dtype,
    }
    save_file(output, output_path, metadata=metadata)
    modules = sum(key.endswith(".lora_A.weight") for key in output)
    print(f"Source transformer blocks: {len(ordered_blocks)}")
    print(f"Anima modules written: {modules} ({modules // ANIMA_BLOCKS} per block)")
    print(f"Skipped unsupported source groups: {skipped}")
    print(f"Projection: {metadata['projection']}")
    print(f"Saved: {output_path}")


def inverse_sqrt(covariance, epsilon):
    values, vectors = torch.linalg.eigh(covariance)
    values = values.clamp_min(epsilon)
    return (vectors * values.rsqrt()) @ vectors.T


def matrix_sqrt(covariance, epsilon):
    values, vectors = torch.linalg.eigh(covariance)
    values = values.clamp_min(epsilon)
    return (vectors * values.sqrt()) @ vectors.T


def fit_projection(source, target, method, regularization):
    if source.ndim != 2 or target.ndim != 2 or source.shape[0] != target.shape[0]:
        raise ValueError("Paired activations must be 2D arrays with equal sample counts")
    source = source.float() - source.float().mean(dim=0, keepdim=True)
    target = target.float() - target.float().mean(dim=0, keepdim=True)
    samples = max(source.shape[0] - 1, 1)

    if method == "procrustes":
        cross = target.T @ source
        u, _, vh = torch.linalg.svd(cross, full_matrices=False)
        return u @ vh

    eye_source = torch.eye(source.shape[1])
    eye_target = torch.eye(target.shape[1])
    cov_source = source.T @ source / samples + regularization * eye_source
    cov_target = target.T @ target / samples + regularization * eye_target
    source_inv_sqrt = inverse_sqrt(cov_source, regularization)
    target_inv_sqrt = inverse_sqrt(cov_target, regularization)
    target_sqrt = matrix_sqrt(cov_target, regularization)
    cross = target_inv_sqrt @ (target.T @ source / samples) @ source_inv_sqrt
    u, _, vh = torch.linalg.svd(cross, full_matrices=False)
    return target_sqrt @ (u @ vh) @ source_inv_sqrt


def fit_calibration(input_path, output_path, method, regularization):
    output = {}
    with np.load(input_path) as data:
        source_keys = [key for key in data.files if key.endswith(".source")]
        for source_key in source_keys:
            name = source_key[: -len(".source")]
            target_key = f"{name}.target"
            if target_key not in data:
                raise ValueError(f"Missing paired array: {target_key}")
            projection = fit_projection(
                torch.from_numpy(data[source_key]),
                torch.from_numpy(data[target_key]),
                method,
                regularization,
            )
            output[f"{name}.projection"] = projection.cpu().numpy()
            print(
                f"{name}: {tuple(data[source_key].shape)} -> "
                f"{tuple(data[target_key].shape)}"
            )
    if not output:
        raise ValueError("No '<name>.source'/'<name>.target' activation pairs found")
    np.savez_compressed(output_path, **output)
    print(f"Saved {len(output)} {method} projections: {output_path}")


V3_ACTIVATION_RE = re.compile(
    r"^(?P<model>source|target)\.(?P<block>\d+)\."
    r"(?P<layer>cross_attn\.(?:[kqv]_proj|output_proj)|"
    r"self_attn\.(?:[kqv]_proj|output_proj)|mlp\.layer[12])\."
    r"(?P<kind>base|styled|input)$"
)


def flatten_activation(array, name):
    tensor = torch.from_numpy(np.asarray(array).copy()).float()
    if tensor.ndim < 2:
        raise ValueError(f"{name} must have shape [..., features]")
    tensor = tensor.reshape(-1, tensor.shape[-1])
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def load_v3_activations(path):
    modules = {"source": defaultdict(dict), "target": defaultdict(dict)}
    with np.load(path) as data:
        for key in data.files:
            match = V3_ACTIVATION_RE.match(key)
            if match is None:
                continue
            module = (int(match.group("block")), match.group("layer"))
            if match.group("model") == "target" and not 0 <= module[0] < ANIMA_BLOCKS:
                raise ValueError(
                    f"Target block {module[0]} is outside Anima range "
                    f"0-{ANIMA_BLOCKS - 1}"
                )
            modules[match.group("model")][module][match.group("kind")] = (
                flatten_activation(data[key], key)
            )

    if not modules["source"] or not modules["target"]:
        raise ValueError("No v3 source/target activation modules were found")

    for module, arrays in modules["source"].items():
        missing = {"base", "styled"} - arrays.keys()
        if missing:
            raise ValueError(f"source.{module[0]}.{module[1]} is missing {sorted(missing)}")
        if arrays["base"].shape != arrays["styled"].shape:
            raise ValueError(f"source.{module[0]}.{module[1]} base/styled shapes differ")
    for module, arrays in modules["target"].items():
        missing = {"base", "input"} - arrays.keys()
        if missing:
            raise ValueError(f"target.{module[0]}.{module[1]} is missing {sorted(missing)}")
        expected = ANIMA_DIMS[module[1]]
        if arrays["input"].shape[1] != expected[0]:
            raise ValueError(
                f"target.{module[0]}.{module[1]}.input has "
                f"{arrays['input'].shape[1]} features, expected {expected[0]}"
            )
        if arrays["base"].shape[1] != expected[1]:
            raise ValueError(
                f"target.{module[0]}.{module[1]}.base has "
                f"{arrays['base'].shape[1]} features, expected {expected[1]}"
            )
        if arrays["input"].shape[0] != arrays["base"].shape[0]:
            raise ValueError(f"target.{module[0]}.{module[1]} sample counts differ")
    sample_counts = {
        array.shape[0]
        for model_modules in modules.values()
        for arrays in model_modules.values()
        for array in arrays.values()
    }
    if len(sample_counts) != 1:
        raise ValueError(
            "All v3 activation arrays must have the same flattened sample count"
        )
    return modules


def linear_cka(left, right, epsilon=1e-12):
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise ValueError("CKA inputs must be 2D with equal sample counts")
    left = left.float() - left.float().mean(dim=0, keepdim=True)
    right = right.float() - right.float().mean(dim=0, keepdim=True)
    samples, left_features = left.shape
    right_features = right.shape[1]
    gram_cost = 2 * samples * samples
    feature_cost = (
        left_features * left_features
        + right_features * right_features
        + left_features * right_features
    )
    if gram_cost <= feature_cost:
        left_gram = left @ left.T
        right_gram = right @ right.T
        cross_norm = (left_gram * right_gram).sum()
        denominator = (
            torch.linalg.matrix_norm(left_gram)
            * torch.linalg.matrix_norm(right_gram)
        )
    else:
        cross_norm = torch.linalg.matrix_norm(left.T @ right).square()
        denominator = (
            torch.linalg.matrix_norm(left.T @ left)
            * torch.linalg.matrix_norm(right.T @ right)
        )
    if denominator <= epsilon:
        return 0.0
    return float((cross_norm / denominator).clamp(0.0, 1.0))


def monotonic_cka_matches(source_items, target_items):
    if not source_items or not target_items:
        return {}
    scores = torch.tensor(
        [
            [linear_cka(source[1], target[1]) for source in source_items]
            for target in target_items
        ],
        dtype=torch.float64,
    )
    target_count, source_count = scores.shape
    totals = torch.empty_like(scores)
    parents = torch.zeros((target_count, source_count), dtype=torch.long)
    totals[0] = scores[0]
    for target_index in range(1, target_count):
        best_values, best_indices = torch.cummax(totals[target_index - 1], dim=0)
        totals[target_index] = scores[target_index] + best_values
        parents[target_index] = best_indices

    source_index = int(torch.argmax(totals[-1]))
    matches = {}
    for target_index in range(target_count - 1, -1, -1):
        target_id = target_items[target_index][0]
        source_id = source_items[source_index][0]
        matches[target_id] = (source_id, float(scores[target_index, source_index]))
        source_index = int(parents[target_index, source_index])
    return matches


def depth_matches(source_items, target_items):
    if not source_items or not target_items:
        return {}
    matches = {}
    for target_index, (target_id, _target_values) in enumerate(target_items):
        if len(target_items) == 1:
            source_index = 0
        else:
            position = target_index * (len(source_items) - 1) / (
                len(target_items) - 1
            )
            source_index = int(round(position))
        matches[target_id] = source_items[source_index][0]
    return matches


def fit_projection_batches(pairs, method, regularization):
    if not pairs:
        raise ValueError("At least one activation pair is required")
    source_dim = pairs[0][0].shape[1]
    target_dim = pairs[0][1].shape[1]
    total = 0
    source_sum = torch.zeros(source_dim)
    target_sum = torch.zeros(target_dim)
    cross_sum = torch.zeros(target_dim, source_dim)
    source_square_sum = None
    target_square_sum = None
    if method == "cca":
        source_square_sum = torch.zeros(source_dim, source_dim)
        target_square_sum = torch.zeros(target_dim, target_dim)

    for source, target in pairs:
        if (
            source.ndim != 2
            or target.ndim != 2
            or source.shape[0] != target.shape[0]
            or source.shape[1] != source_dim
            or target.shape[1] != target_dim
        ):
            raise ValueError("Hybrid bridge activation pair shapes are inconsistent")
        source = source.float()
        target = target.float()
        total += source.shape[0]
        source_sum += source.sum(dim=0)
        target_sum += target.sum(dim=0)
        cross_sum += target.T @ source
        if method == "cca":
            source_square_sum += source.T @ source
            target_square_sum += target.T @ target

    source_mean = source_sum / total
    target_mean = target_sum / total
    cross = cross_sum - total * torch.outer(target_mean, source_mean)
    if method == "procrustes":
        u, _, vh = torch.linalg.svd(cross, full_matrices=False)
        return u @ vh

    samples = max(total - 1, 1)
    cov_source = (
        source_square_sum - total * torch.outer(source_mean, source_mean)
    ) / samples
    cov_target = (
        target_square_sum - total * torch.outer(target_mean, target_mean)
    ) / samples
    cross_covariance = cross / samples
    cov_source += regularization * torch.eye(source_dim)
    cov_target += regularization * torch.eye(target_dim)
    source_inv_sqrt = inverse_sqrt(cov_source, regularization)
    target_inv_sqrt = inverse_sqrt(cov_target, regularization)
    target_sqrt = matrix_sqrt(cov_target, regularization)
    whitened_cross = (
        target_inv_sqrt @ cross_covariance @ source_inv_sqrt
    )
    u, _, vh = torch.linalg.svd(whitened_cross, full_matrices=False)
    return target_sqrt @ (u @ vh) @ source_inv_sqrt


def hybrid_projection_key(layer, source_dim, target_dim):
    return f"hybrid.{layer}.{source_dim}x{target_dim}.projection"


def hybrid_module_pairs(modules, layer):
    source_items = sorted(
        (
            (block, arrays["base"])
            for (block, module_layer), arrays in modules["source"].items()
            if module_layer == layer
        ),
        key=lambda item: item[0],
    )
    target_items = sorted(
        (
            (block, arrays["base"])
            for (block, module_layer), arrays in modules["target"].items()
            if module_layer == layer
        ),
        key=lambda item: item[0],
    )
    if not source_items:
        raise ValueError(f"No source activations found for target layer {layer}")
    matches = depth_matches(source_items, target_items)
    return [
        (
            target_block,
            matches[target_block],
            target_base,
        )
        for target_block, target_base in target_items
    ]


def fit_v3_hybrid_bridge(input_path, output_path, method, regularization):
    modules = load_v3_activations(input_path)
    grouped_pairs = defaultdict(list)
    mappings = []
    layers = sorted({module[1] for module in modules["target"]})
    for layer in layers:
        for target_block, source_block, target_base in hybrid_module_pairs(
            modules, layer
        ):
            source_base = modules["source"][(source_block, layer)]["base"]
            signature = (layer, source_base.shape[1], target_base.shape[1])
            grouped_pairs[signature].append((source_base, target_base))
            mappings.append((target_block, layer, source_block))

    output = {
        hybrid_projection_key(layer, source_dim, target_dim): (
            fit_projection_batches(pairs, method, regularization).cpu().numpy()
        )
        for (layer, source_dim, target_dim), pairs in grouped_pairs.items()
    }
    output["__format__"] = np.array("anima_v3_hybrid_bridge")
    output["__method__"] = np.array(method)
    output["__regularization__"] = np.array(regularization)
    np.savez_compressed(output_path, **output)
    print(
        f"Saved {len(grouped_pairs)} reusable {method} projections "
        f"for {len(mappings)} depth-matched modules"
    )
    print(f"Saved: {output_path}")


def load_v3_hybrid_bridge(path):
    with np.load(path) as data:
        if npz_scalar(data, "__format__") != "anima_v3_hybrid_bridge":
            raise ValueError("Not an Anima v3 hybrid bridge archive")
        projections = {
            key: torch.from_numpy(data[key]).float()
            for key in data.files
            if key.endswith(".projection")
        }
        method = npz_scalar(data, "__method__")
    if not projections:
        raise ValueError("Hybrid bridge contains no projections")
    return projections, method


def ridge_map(inputs, outputs, regularization, center=False):
    if inputs.ndim != 2 or outputs.ndim != 2 or inputs.shape[0] != outputs.shape[0]:
        raise ValueError("Ridge inputs/outputs must be 2D with equal sample counts")
    x = inputs.float()
    y = outputs.float()
    if center:
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)

    if x.shape[0] <= x.shape[1]:
        gram = x @ x.T
        scale = gram.diagonal().mean().clamp_min(torch.finfo(gram.dtype).eps)
        system = gram + regularization * scale * torch.eye(gram.shape[0])
        return x.T @ torch.linalg.solve(system, y)

    gram = x.T @ x
    scale = gram.diagonal().mean().clamp_min(torch.finfo(gram.dtype).eps)
    system = gram + regularization * scale * torch.eye(gram.shape[0])
    return torch.linalg.solve(system, x.T @ y)


def ridge_transform_directions(inputs, outputs, directions, regularization):
    if directions.ndim != 2 or directions.shape[1] != inputs.shape[1]:
        raise ValueError("Directions must be 2D and match the input feature dimension")
    x = inputs.float() - inputs.float().mean(dim=0, keepdim=True)
    y = outputs.float() - outputs.float().mean(dim=0, keepdim=True)
    gram = x @ x.T
    scale = gram.diagonal().mean().clamp_min(torch.finfo(gram.dtype).eps)
    system = gram + regularization * scale * torch.eye(gram.shape[0])
    coefficients = torch.linalg.solve(system, y)
    return directions.float() @ x.T @ coefficients


def factorize_distilled_delta(
    target_input,
    desired_delta,
    rank,
    regularization,
    dtype,
    calibration_input=None,
    calibration_delta=None,
):
    x = target_input.float()
    gram = x @ x.T
    ridge_scale = gram.diagonal().mean().clamp_min(torch.finfo(gram.dtype).eps)
    system = gram + regularization * ridge_scale * torch.eye(gram.shape[0])
    coefficients = torch.linalg.solve(system, desired_delta.float())

    q_b, r_b = torch.linalg.qr(coefficients.T, mode="reduced")
    q_a, r_a = torch.linalg.qr(x.T, mode="reduced")
    core = r_b @ r_a.T
    u, singular, vh = torch.linalg.svd(core, full_matrices=False)
    used_rank = min(rank, singular.numel())
    roots = singular[:used_rank].clamp_min(0).sqrt()
    lora_b = (q_b @ u[:, :used_rank]) * roots
    lora_a = roots[:, None] * (vh[:used_rank] @ q_a.T)

    if (calibration_input is None) != (calibration_delta is None):
        raise ValueError("Calibration input and delta must be provided together")
    evaluation_input = (
        target_input if calibration_input is None else calibration_input
    ).float()
    evaluation_delta = (
        desired_delta if calibration_delta is None else calibration_delta
    ).float()
    predicted = evaluation_input @ (lora_b @ lora_a).T
    predicted_rms = predicted.square().mean().sqrt()
    desired_rms = evaluation_delta.square().mean().sqrt()
    scale = 1.0
    if desired_rms > 0 and predicted_rms > 0:
        scale = float(desired_rms / predicted_rms)
        lora_b *= scale
        predicted *= scale
    residual_rms = (predicted - evaluation_delta).square().mean().sqrt()
    normalized_error = float(residual_rms / desired_rms) if desired_rms > 0 else 0.0
    return (
        lora_a.to(dtype).contiguous(),
        lora_b.to(dtype).contiguous(),
        scale,
        normalized_error,
    )


def distill_v3(input_path, output_path, rank, alignment_regularization,
               distillation_regularization, min_cka, max_scale, max_nrmse,
               validation_fraction, seed, output_dtype):
    modules = load_v3_activations(input_path)
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    output = {}
    match_records = []
    sample_count = next(
        iter(next(iter(modules["source"].values())).values())
    ).shape[0]
    validation_count = max(1, round(sample_count * validation_fraction))
    if validation_count >= sample_count:
        raise ValueError("Not enough samples for the requested validation fraction")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(sample_count, generator=generator)
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]

    layers = sorted({module[1] for module in modules["target"]})
    for layer in layers:
        source_items = sorted(
            (
                (block, arrays["base"][train_indices])
                for (block, module_layer), arrays in modules["source"].items()
                if module_layer == layer
            ),
            key=lambda item: item[0],
        )
        target_items = sorted(
            (
                (block, arrays["base"][train_indices])
                for (block, module_layer), arrays in modules["target"].items()
                if module_layer == layer
            ),
            key=lambda item: item[0],
        )
        if not source_items:
            raise ValueError(f"No source activations found for target layer {layer}")
        matches = monotonic_cka_matches(source_items, target_items)

        for target_block, target_base_train in target_items:
            source_block, score = matches[target_block]
            if score < min_cka:
                raise ValueError(
                    f"CKA {score:.4f} for target.{target_block}.{layer} is below "
                    f"--min-cka {min_cka:.4f}"
                )
            source = modules["source"][(source_block, layer)]
            target = modules["target"][(target_block, layer)]
            source_base_train = source["base"][train_indices]
            teacher_delta = source["styled"] - source["base"]
            desired_delta_train = ridge_transform_directions(
                source_base_train,
                target_base_train,
                teacher_delta[train_indices],
                alignment_regularization,
            )
            desired_delta_validation = ridge_transform_directions(
                source_base_train,
                target_base_train,
                teacher_delta[validation_indices],
                alignment_regularization,
            )
            lora_a, lora_b, scale, normalized_error = factorize_distilled_delta(
                target["input"][train_indices],
                desired_delta_train,
                rank,
                distillation_regularization,
                dtype_map[output_dtype],
                target["input"][validation_indices],
                desired_delta_validation,
            )
            if scale > max_scale:
                raise ValueError(
                    f"Scale {scale:.4f} for target.{target_block}.{layer} exceeds "
                    f"--max-scale {max_scale:.4f}; improve activation coverage or "
                    "increase the output rank"
                )
            if normalized_error > max_nrmse:
                raise ValueError(
                    f"NRMSE {normalized_error:.4f} for target.{target_block}.{layer} "
                    f"exceeds --max-nrmse {max_nrmse:.4f}; the target input cannot "
                    "reproduce the aligned teacher difference"
                )
            prefix = f"diffusion_model.blocks.{target_block}.{layer}"
            output[f"{prefix}.lora_A.weight"] = lora_a
            output[f"{prefix}.lora_B.weight"] = lora_b
            output[f"{prefix}.lora_alpha"] = torch.tensor(float(lora_a.shape[0]))
            match_records.append(
                (target_block, layer, source_block, score, scale, normalized_error)
            )

    metadata = {
        "converter": "anima_lora_style_bridge_v3",
        "source": str(Path(input_path).name),
        "method": "cka+ridge-stitching+delta-distillation+rms-calibration",
        "requested_rank": str(rank),
        "output_ranks": ",".join(
            str(value)
            for value in sorted(
                {
                    tensor.shape[0]
                    for key, tensor in output.items()
                    if key.endswith(".lora_A.weight")
                }
            )
        ),
        "dtype": output_dtype,
        "validation_fraction": str(validation_fraction),
        "seed": str(seed),
    }
    save_file(output, output_path, metadata=metadata)
    for (
        target_block,
        layer,
        source_block,
        score,
        scale,
        normalized_error,
    ) in match_records:
        warning = " warning=low-cka" if score < 0.1 else ""
        print(
            f"target.{target_block}.{layer} <- source.{source_block}.{layer} "
            f"cka={score:.4f} scale={scale:.4f} nrmse={normalized_error:.4f}"
            f"{warning}"
        )
    print(f"Anima modules distilled: {len(match_records)}")
    print(f"Saved: {output_path}")


def distill_v3_hybrid(
    input_path,
    bridge_path,
    output_path,
    rank,
    distillation_regularization,
    max_scale,
    max_nrmse,
    validation_fraction,
    seed,
    output_dtype,
):
    modules = load_v3_activations(input_path)
    projections, bridge_method = load_v3_hybrid_bridge(bridge_path)
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    sample_count = next(
        iter(next(iter(modules["source"].values())).values())
    ).shape[0]
    validation_count = max(1, round(sample_count * validation_fraction))
    if validation_count >= sample_count:
        raise ValueError("Not enough samples for the requested validation fraction")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(sample_count, generator=generator)
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    output = {}
    records = []

    layers = sorted({module[1] for module in modules["target"]})
    for layer in layers:
        for target_block, source_block, _target_base in hybrid_module_pairs(
            modules, layer
        ):
            source = modules["source"][(source_block, layer)]
            target = modules["target"][(target_block, layer)]
            source_dim = source["base"].shape[1]
            target_dim = target["base"].shape[1]
            key = hybrid_projection_key(layer, source_dim, target_dim)
            projection = projections.get(key)
            if projection is None:
                raise ValueError(f"Hybrid bridge is missing projection: {key}")
            expected = (target_dim, source_dim)
            if tuple(projection.shape) != expected:
                raise ValueError(
                    f"{key} has shape {tuple(projection.shape)}, expected {expected}"
                )

            teacher_delta = source["styled"] - source["base"]
            desired_delta_train = teacher_delta[train_indices] @ projection.T
            desired_delta_validation = (
                teacher_delta[validation_indices] @ projection.T
            )
            lora_a, lora_b, scale, normalized_error = factorize_distilled_delta(
                target["input"][train_indices],
                desired_delta_train,
                rank,
                distillation_regularization,
                dtype_map[output_dtype],
                target["input"][validation_indices],
                desired_delta_validation,
            )
            if scale > max_scale:
                raise ValueError(
                    f"Scale {scale:.4f} for target.{target_block}.{layer} exceeds "
                    f"--max-scale {max_scale:.4f}"
                )
            if normalized_error > max_nrmse:
                raise ValueError(
                    f"NRMSE {normalized_error:.4f} for target.{target_block}.{layer} "
                    f"exceeds --max-nrmse {max_nrmse:.4f}"
                )
            prefix = f"diffusion_model.blocks.{target_block}.{layer}"
            output[f"{prefix}.lora_A.weight"] = lora_a
            output[f"{prefix}.lora_B.weight"] = lora_b
            output[f"{prefix}.lora_alpha"] = torch.tensor(float(lora_a.shape[0]))
            records.append(
                (target_block, layer, source_block, scale, normalized_error)
            )

    metadata = {
        "converter": "anima_lora_style_bridge_v3_hybrid",
        "source": str(Path(input_path).name),
        "bridge": str(Path(bridge_path).name),
        "method": (
            f"depth+{bridge_method}-bridge+delta-distillation+rms-calibration"
        ),
        "requested_rank": str(rank),
        "output_ranks": ",".join(
            str(value)
            for value in sorted(
                {
                    tensor.shape[0]
                    for key, tensor in output.items()
                    if key.endswith(".lora_A.weight")
                }
            )
        ),
        "dtype": output_dtype,
        "validation_fraction": str(validation_fraction),
        "seed": str(seed),
    }
    save_file(output, output_path, metadata=metadata)
    for target_block, layer, source_block, scale, normalized_error in records:
        print(
            f"target.{target_block}.{layer} <- source.{source_block}.{layer} "
            f"depth scale={scale:.4f} nrmse={normalized_error:.4f}"
        )
    print(f"Anima modules distilled: {len(records)}")
    print(f"Saved: {output_path}")


def validate_output(path):
    state = load_file(path)
    errors = []
    modules = 0
    blocks = set()
    for key, a in state.items():
        if not key.endswith(".lora_A.weight"):
            continue
        base = key[: -len(".lora_A.weight")]
        b_key = f"{base}.lora_B.weight"
        if b_key not in state:
            errors.append(f"Missing {b_key}")
            continue
        layer = ".".join(base.split(".")[3:])
        block = int(base.split(".")[2])
        expected = ANIMA_DIMS.get(layer)
        b = state[b_key]
        if expected and (a.shape[1], b.shape[0]) != expected:
            errors.append(
                f"{base}: {(a.shape[1], b.shape[0])}, expected {expected}"
            )
        if a.shape[0] != b.shape[1]:
            errors.append(f"{base}: rank mismatch")
        modules += 1
        blocks.add(block)
    if errors:
        raise ValueError("\n".join(errors[:20]))
    print(f"Valid Anima dimensions: {modules} modules across {len(blocks)} blocks")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert", help="Convert an SD LoRA to Anima")
    convert.add_argument("input")
    convert.add_argument("-o", "--output")
    convert.add_argument(
        "--rank",
        type=int,
        default=16,
        help="Output rank (default: 16 for broad loader compatibility)",
    )
    convert.add_argument("--calibration", help="NPZ containing fitted projections")
    convert.add_argument(
        "--dtype",
        choices=("float16", "float32", "bfloat16"),
        default="float16",
        help="Output weight dtype (default: float16 for broad loader compatibility)",
    )

    fit = subparsers.add_parser("fit-calibration", help="Fit projections from activations")
    fit.add_argument("input", help="NPZ containing paired source/target activations")
    fit.add_argument("-o", "--output", default="anima_alignment.npz")
    fit.add_argument("--method", choices=("procrustes", "cca"), default="procrustes")
    fit.add_argument("--regularization", type=float, default=1e-4)

    distill = subparsers.add_parser(
        "distill-v3",
        help="Distill paired SDXL LoRA activation differences into an Anima LoRA",
    )
    distill.add_argument("input", help="NPZ containing v3 paired activations")
    distill.add_argument("-o", "--output", default="anima_style_v3.safetensors")
    distill.add_argument("--rank", type=int, default=16)
    distill.add_argument("--alignment-regularization", type=float, default=1e-3)
    distill.add_argument("--distillation-regularization", type=float, default=1e-3)
    distill.add_argument(
        "--min-cka",
        type=float,
        default=0.0,
        help="Reject matched modules below this CKA score (default: 0)",
    )
    distill.add_argument(
        "--max-scale",
        type=float,
        default=10.0,
        help="Reject unstable RMS calibration above this multiplier (default: 10)",
    )
    distill.add_argument(
        "--max-nrmse",
        type=float,
        default=1.0,
        help="Reject low-rank fits above this normalized RMSE (default: 1)",
    )
    distill.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Held-out fraction used for scale calibration and NRMSE (default: 0.2)",
    )
    distill.add_argument("--seed", type=int, default=0)
    distill.add_argument(
        "--dtype",
        choices=("float16", "float32", "bfloat16"),
        default="float16",
    )

    fit_hybrid = subparsers.add_parser(
        "fit-v3-hybrid-bridge",
        help="Fit a reusable depth-matched Procrustes/CCA activation bridge",
    )
    fit_hybrid.add_argument("input", help="NPZ containing v3 paired activations")
    fit_hybrid.add_argument("-o", "--output", default="anima_v3_hybrid_bridge.npz")
    fit_hybrid.add_argument(
        "--method",
        choices=("procrustes", "cca"),
        default="procrustes",
    )
    fit_hybrid.add_argument("--regularization", type=float, default=1e-4)

    hybrid = subparsers.add_parser(
        "distill-v3-hybrid",
        help="Distill with depth matching and a reusable activation bridge",
    )
    hybrid.add_argument("input", help="NPZ containing v3 paired activations")
    hybrid.add_argument("--bridge", required=True, help="Fitted hybrid bridge NPZ")
    hybrid.add_argument("-o", "--output", default="anima_style_v3_hybrid.safetensors")
    hybrid.add_argument("--rank", type=int, default=16)
    hybrid.add_argument("--distillation-regularization", type=float, default=1e-3)
    hybrid.add_argument("--max-scale", type=float, default=10.0)
    hybrid.add_argument("--max-nrmse", type=float, default=1.0)
    hybrid.add_argument("--validation-fraction", type=float, default=0.2)
    hybrid.add_argument("--seed", type=int, default=0)
    hybrid.add_argument(
        "--dtype",
        choices=("float16", "float32", "bfloat16"),
        default="float16",
    )

    prepare = subparsers.add_parser(
        "prepare-v3-activations",
        help="Convert v2 ComfyUI captures and an SDXL LoRA into a v3 activation NPZ",
    )
    prepare.add_argument("source_capture", help="SDXL+LoRA v2 capture NPZ")
    prepare.add_argument("target_capture", help="Anima-Base v2 capture NPZ")
    prepare.add_argument("lora", help="The SDXL LoRA used during source capture")
    prepare.add_argument("-o", "--output", default="activations_v3.npz")
    prepare.add_argument(
        "--lora-strength",
        type=float,
        default=1.0,
        help="Model strength used by the ComfyUI LoRA loader (default: 1)",
    )
    prepare.add_argument(
        "--truncate-unpaired-tail",
        action="store_true",
        help=(
            "Use the common row prefix when one capture has extra executions "
            "appended at the end"
        ),
    )

    validate = subparsers.add_parser("validate", help="Validate converted dimensions")
    validate.add_argument("input")
    return parser


def main():
    argv = sys.argv[1:]
    commands = {
        "convert",
        "fit-calibration",
        "prepare-v3-activations",
        "fit-v3-hybrid-bridge",
        "distill-v3",
        "distill-v3-hybrid",
        "validate",
    }
    if argv and argv[0] not in commands and not argv[0].startswith("-"):
        argv.insert(0, "convert")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fit-calibration":
        if args.regularization <= 0:
            parser.error("--regularization must be greater than zero")
        fit_calibration(args.input, args.output, args.method, args.regularization)
        return
    if args.command == "validate":
        validate_output(args.input)
        return
    if args.command == "prepare-v3-activations":
        if args.lora_strength == 0:
            parser.error("--lora-strength must not be zero")
        if Path(args.output).resolve() in {
            Path(args.source_capture).resolve(),
            Path(args.target_capture).resolve(),
            Path(args.lora).resolve(),
        }:
            parser.error("output path must be different from all input paths")
        prepare_v3_activations(
            args.source_capture,
            args.target_capture,
            args.lora,
            args.output,
            args.lora_strength,
            args.truncate_unpaired_tail,
        )
        load_v3_activations(args.output)
        return
    if args.command == "fit-v3-hybrid-bridge":
        if args.regularization <= 0:
            parser.error("--regularization must be greater than zero")
        if Path(args.input).resolve() == Path(args.output).resolve():
            parser.error("input and output paths must be different")
        fit_v3_hybrid_bridge(
            args.input,
            args.output,
            args.method,
            args.regularization,
        )
        load_v3_hybrid_bridge(args.output)
        return
    if args.command == "distill-v3-hybrid":
        if args.rank <= 0:
            parser.error("--rank must be greater than zero")
        if args.distillation_regularization <= 0:
            parser.error("--distillation-regularization must be greater than zero")
        if args.max_scale <= 0:
            parser.error("--max-scale must be greater than zero")
        if args.max_nrmse <= 0:
            parser.error("--max-nrmse must be greater than zero")
        if not 0 < args.validation_fraction < 1:
            parser.error("--validation-fraction must be between zero and one")
        if Path(args.input).resolve() == Path(args.output).resolve():
            parser.error("input and output paths must be different")
        distill_v3_hybrid(
            args.input,
            args.bridge,
            args.output,
            args.rank,
            args.distillation_regularization,
            args.max_scale,
            args.max_nrmse,
            args.validation_fraction,
            args.seed,
            args.dtype,
        )
        validate_output(args.output)
        return
    if args.command == "distill-v3":
        if args.rank <= 0:
            parser.error("--rank must be greater than zero")
        if args.alignment_regularization <= 0:
            parser.error("--alignment-regularization must be greater than zero")
        if args.distillation_regularization <= 0:
            parser.error("--distillation-regularization must be greater than zero")
        if not 0 <= args.min_cka <= 1:
            parser.error("--min-cka must be between zero and one")
        if args.max_scale <= 0:
            parser.error("--max-scale must be greater than zero")
        if args.max_nrmse <= 0:
            parser.error("--max-nrmse must be greater than zero")
        if not 0 < args.validation_fraction < 1:
            parser.error("--validation-fraction must be between zero and one")
        if Path(args.input).resolve() == Path(args.output).resolve():
            parser.error("input and output paths must be different")
        distill_v3(
            args.input,
            args.output,
            args.rank,
            args.alignment_regularization,
            args.distillation_regularization,
            args.min_cka,
            args.max_scale,
            args.max_nrmse,
            args.validation_fraction,
            args.seed,
            args.dtype,
        )
        validate_output(args.output)
        return

    output = args.output or f"{Path(args.input).stem}_anima_style.safetensors"
    if args.rank <= 0:
        parser.error("--rank must be greater than zero")
    if Path(args.input).resolve() == Path(output).resolve():
        parser.error("input and output paths must be different")
    convert_sd(args.input, output, args.rank, args.calibration, args.dtype)
    validate_output(output)


if __name__ == "__main__":
    main()
