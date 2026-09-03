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


VERSION = "1.0.0"
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

    validate = subparsers.add_parser("validate", help="Validate converted dimensions")
    validate.add_argument("input")
    return parser


def main():
    argv = sys.argv[1:]
    commands = {"convert", "fit-calibration", "validate"}
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

    output = args.output or f"{Path(args.input).stem}_anima_style.safetensors"
    if args.rank <= 0:
        parser.error("--rank must be greater than zero")
    if Path(args.input).resolve() == Path(output).resolve():
        parser.error("input and output paths must be different")
    convert_sd(args.input, output, args.rank, args.calibration, args.dtype)
    validate_output(output)


if __name__ == "__main__":
    main()
