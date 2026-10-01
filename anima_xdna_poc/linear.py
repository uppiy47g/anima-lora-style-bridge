"""Shared Linear layout and CPU oracle."""

from dataclasses import dataclass
import math

import torch

from .errors import UnsupportedTensor


ALIGNMENT = 8
BLOCK_M = 256
BLOCK_N = 256
BLOCK_K = 64


@dataclass(frozen=True)
class PreparedLinear:
    input_bf16: torch.Tensor
    weight_k_n_bf16: torch.Tensor
    rows: int
    in_features: int
    out_features: int

    @property
    def padded_shape(self) -> tuple[int, int, int]:
        return (
            self.input_bf16.shape[0],
            self.weight_k_n_bf16.shape[1],
            self.input_bf16.shape[1],
        )


def _aligned(value: int, alignment: int = ALIGNMENT) -> int:
    return math.ceil(value / alignment) * alignment


def deterministic_input(rows: int, in_features: int, seed: int = 0) -> torch.Tensor:
    if rows <= 0 or in_features <= 0:
        raise UnsupportedTensor(
            f"input dimensions must be positive, got rows={rows}, in_features={in_features}"
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scale = 1.0 / math.sqrt(in_features)
    return (torch.randn(rows, in_features, generator=generator) * scale).to(
        torch.bfloat16
    )


def prepare_linear(input_tensor: torch.Tensor, weight_out_in: torch.Tensor) -> PreparedLinear:
    if input_tensor.ndim != 2 or weight_out_in.ndim != 2:
        raise UnsupportedTensor(
            "Linear input and weight must both be rank 2; "
            f"got {tuple(input_tensor.shape)} and {tuple(weight_out_in.shape)}"
        )
    rows, in_features = input_tensor.shape
    out_features, weight_in = weight_out_in.shape
    if in_features != weight_in:
        raise UnsupportedTensor(
            f"input feature count {in_features} does not match weight feature count {weight_in}"
        )
    if min(rows, in_features, out_features) <= 0:
        raise UnsupportedTensor("Linear dimensions must all be positive")

    padded_rows = _aligned(rows, BLOCK_M)
    padded_in = max(_aligned(in_features, BLOCK_K), 256)
    padded_out = _aligned(out_features, BLOCK_N)
    # Triton-XDNA 3.6 cannot lower a single 256x256 output tile when K spans
    # multiple AIE reduction tiles. A second zero output tile keeps the same
    # logical result and gives AIR a legal herd mapping.
    if padded_out == BLOCK_N and padded_in > 256:
        padded_out = 2 * BLOCK_N
    prepared_input = torch.zeros(padded_rows, padded_in, dtype=torch.bfloat16)
    prepared_weight = torch.zeros(padded_in, padded_out, dtype=torch.bfloat16)
    prepared_input[:rows, :in_features] = input_tensor.to(torch.bfloat16)
    prepared_weight[:in_features, :out_features] = (
        weight_out_in.to(torch.bfloat16).T.contiguous()
    )
    return PreparedLinear(
        prepared_input.contiguous(),
        prepared_weight.contiguous(),
        rows,
        in_features,
        out_features,
    )


def cpu_linear(prepared: PreparedLinear) -> torch.Tensor:
    accumulated = (
        prepared.input_bf16.float() @ prepared.weight_k_n_bf16.float()
    )
    return accumulated[: prepared.rows, : prepared.out_features].to(torch.bfloat16)
