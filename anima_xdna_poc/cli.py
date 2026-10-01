"""Command-line entry points for probing and validating one Anima Linear."""

import argparse
import json
from pathlib import Path
import sys

import torch

from .checkpoint import discover_linear_weight
from .errors import NumericalMismatch, PrototypeError
from .linear import cpu_linear, deterministic_input, prepare_linear
from .xdna import execute, probe


def _print_failure(error: PrototypeError) -> int:
    print(f"FAIL [{error.category}]: {error}", file=sys.stderr)
    return error.exit_code


def probe_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe Triton-XDNA and XDNA 2")
    parser.parse_args(argv)
    try:
        result = probe()
    except PrototypeError as error:
        return _print_failure(error)
    print(
        json.dumps(
            {
                "status": "ready",
                "triton_xdna": result.triton_xdna_version,
                "target": result.target,
                "runtime": result.runtime,
            },
            indent=2,
        )
    )
    return 0


def linear_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare one Anima Linear projection on XDNA 2 and CPU"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--key", help="exact safetensors key instead of alias discovery")
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=1.6e-2)
    parser.add_argument("--atol", type=float)
    args = parser.parse_args(argv)

    try:
        selected = discover_linear_weight(args.checkpoint, args.key)
        inputs = deterministic_input(args.rows, selected.in_features, args.seed)
        prepared = prepare_linear(inputs, selected.tensor)
        reference = cpu_linear(prepared)
        result = execute(prepared, args.warmup, args.runs)
        atol = args.atol
        if atol is None:
            atol = 1.5e-3 * (8192 / prepared.input_bf16.shape[1]) ** 0.5
        try:
            torch.testing.assert_close(
                result.output_bf16,
                reference,
                rtol=args.rtol,
                atol=atol,
            )
        except AssertionError as error:
            max_error = (
                result.output_bf16.float() - reference.float()
            ).abs().max().item()
            raise NumericalMismatch(
                f"NPU output differs from the CPU oracle (max_abs_error={max_error:.6g}, "
                f"rtol={args.rtol}, atol={atol:.6g}): {error}"
            ) from error
    except PrototypeError as error:
        return _print_failure(error)

    print(
        json.dumps(
            {
                "status": "match",
                "weight_key": selected.key,
                "weight_shape_out_in": list(selected.tensor.shape),
                "logical_m_n_k": [
                    prepared.rows,
                    prepared.out_features,
                    prepared.in_features,
                ],
                "padded_m_n_k": list(prepared.padded_shape),
                "input_dtype": "bfloat16",
                "weight_dtype": "bfloat16",
                "accumulator_dtype": "float32",
                "output_dtype": "bfloat16",
                "compile_and_first_run_ms": result.compile_and_first_run_ms,
                "median_wall_run_ms": result.median_run_ms,
                "artifact_cache": result.artifact_cache,
                "rtol": args.rtol,
                "atol": atol,
            },
            indent=2,
        )
    )
    return 0
