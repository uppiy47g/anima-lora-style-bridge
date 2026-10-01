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


def block_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute one exact Anima/Cosmos transformer block on XDNA 2"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        help="transformer config.json (defaults beside the checkpoint)",
    )
    parser.add_argument("--image-tokens", type=int, default=1024)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--masked-context-tokens", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=3e-3)
    args = parser.parse_args(argv)
    if args.runs <= 0:
        parser.error("--runs must be positive")

    try:
        from .block import (
            compare_block_outputs,
            deterministic_block_inputs,
            run_cpu_block,
            run_xdna_block,
        )
        from .block_checkpoint import load_block_config, load_block_weights

        config_path = args.config or args.checkpoint.parent / "config.json"
        config = load_block_config(config_path)
        weights = load_block_weights(args.checkpoint, config)
        inputs = deterministic_block_inputs(
            config,
            args.image_tokens,
            args.context_tokens,
            args.seed,
            args.masked_context_tokens,
        )
        cpu_started = __import__("time").perf_counter()
        reference = run_cpu_block(config, weights, inputs)
        cpu_ms = (__import__("time").perf_counter() - cpu_started) * 1000
        npu_runs = [run_xdna_block(config, weights, inputs) for _ in range(args.runs)]
        result = npu_runs[-1]
        max_error, mean_error = compare_block_outputs(
            result.output, reference.output, args.rtol, args.atol
        )
        reference_stages = dict(reference.checkpoints)
        stage_errors = {}
        for name, value in result.checkpoints:
            difference = (value.float() - reference_stages[name].float()).abs()
            stage_errors[name] = {
                "max_abs_error": difference.max().item(),
                "mean_abs_error": difference.mean().item(),
            }
    except PrototypeError as error:
        return _print_failure(error)

    npu_flops = sum(
        metric.estimated_flops for metric in result.metrics if metric.device == "xdna2"
    )
    host_ms = sum(metric.wall_ms for metric in result.metrics if metric.device == "host")
    npu_ms = sum(metric.wall_ms for metric in result.metrics if metric.device == "xdna2")
    transfer_bytes = sum(
        metric.transfer_bytes for metric in result.metrics if metric.device == "xdna2"
    )
    payload = {
        "status": "match",
        "model": "Anima Base v1.0 Diffusers / CosmosTransformer3DModel",
        "block": weights.prefix.rstrip("."),
        "shape": {
            "batch": 1,
            "image_tokens": args.image_tokens,
            "hidden_size": config.hidden_size,
            "context_tokens": args.context_tokens,
            "context_dim": config.context_dim,
            "heads": config.num_heads,
            "head_dim": config.head_dim,
        },
        "execution": {
            "npu_operations": "all Linear, QK^T, and attention-value GEMMs",
            "host_operations": [
                "SiLU",
                "LayerNorm/modulation",
                "RMSNorm",
                "RoPE",
                "attention scaling/mask/softmax",
                "GELU",
                "gating/residual",
            ],
            "dispatches": result.dispatch_count,
            "npu_gemm_flops": npu_flops,
            "estimated_host_npu_transfer_bytes": transfer_bytes,
            "npu_stage_wall_ms": npu_ms,
            "host_stage_wall_ms": host_ms,
            "cpu_oracle_wall_ms": cpu_ms,
            "block_run_wall_ms": [
                sum(metric.wall_ms for metric in run.metrics) for run in npu_runs
            ],
            "artifact_caches": list(result.artifact_caches),
        },
        "comparison": {
            "rtol": args.rtol,
            "atol": args.atol,
            "max_abs_error": max_error,
            "mean_abs_error": mean_error,
            "stage_errors": stage_errors,
        },
        "stages": [
            {
                "name": metric.name,
                "device": metric.device,
                "wall_ms": metric.wall_ms,
                "dispatches": metric.dispatches,
                "estimated_flops": metric.estimated_flops,
                "transfer_bytes": metric.transfer_bytes,
                "h2d_bytes": metric.h2d_bytes,
                "d2h_bytes": metric.d2h_bytes,
                "allocation_count": metric.allocation_count,
                "resident_hits": metric.resident_hits,
                "host_copy_ms": metric.host_copy_ms,
                "h2d_sync_ms": metric.h2d_sync_ms,
                "kernel_sync_ms": metric.kernel_sync_ms,
                "d2h_sync_ms": metric.d2h_sync_ms,
            }
            for metric in result.metrics
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0


def chain_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Execute an Anima transformer block range with resident XDNA buffers"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--fixture", type=Path, help="captured BlockInputs .pt file")
    parser.add_argument("--image-tokens", type=int, default=1024)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-block", type=int, default=0)
    parser.add_argument("--end-block", type=int, default=28)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--oracle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.runs <= 0:
        parser.error("--warmups must be non-negative and --runs must be positive")

    try:
        from .block import BlockInputs, deterministic_block_inputs
        from .chain import AnimaXDNAChainRuntime

        with AnimaXDNAChainRuntime(
            args.checkpoint,
            args.config,
            cache_host_weights=False,
        ) as runtime:
            if args.fixture:
                captured = torch.load(args.fixture, map_location="cpu", weights_only=True)
                inputs = BlockInputs(
                    captured["hidden_states"],
                    captured["encoder_hidden_states"],
                    captured["embedded_timestep"],
                    captured["temb"],
                    (captured["cos"], captured["sin"]),
                    captured.get("attention_mask"),
                )
                input_kind = "captured model preprocessing"
            else:
                inputs = deterministic_block_inputs(
                    runtime.config,
                    args.image_tokens,
                    args.context_tokens,
                    args.seed,
                )
                input_kind = "deterministic synthetic"

            import time

            populate_started = time.perf_counter()
            validation = None
            if args.oracle:
                validation = runtime.validate_range(
                    inputs,
                    args.start_block,
                    args.end_block,
                )
                populated = validation.xdna
            else:
                populated = runtime.run_range(
                    inputs,
                    args.start_block,
                    args.end_block,
                )
            populate_ms = (time.perf_counter() - populate_started) * 1000
            for _ in range(args.warmups):
                runtime.run_range(inputs, args.start_block, args.end_block)
            repeated = []
            for _ in range(args.runs):
                started = time.perf_counter()
                result = runtime.run_range(
                    inputs,
                    args.start_block,
                    args.end_block,
                )
                repeated.append(((time.perf_counter() - started) * 1000, result))
    except PrototypeError as error:
        return _print_failure(error)

    final = repeated[-1][1]
    metrics = [
        metric
        for block in final.blocks
        for metric in block.result.metrics
    ]
    errors = []
    if validation is not None:
        errors = [
            {
                "block": error.index,
                "max_abs_error": error.max_abs_error,
                "mean_abs_error": error.mean_abs_error,
                "rms_error": error.rms_error,
                "reference_rms": error.reference_rms,
                "normalized_rms_error": error.normalized_rms_error,
            }
            for error in validation.errors
        ]
    worst = (
        max(errors, key=lambda item: item["normalized_rms_error"])
        if errors
        else None
    )
    payload = {
        "status": "match" if validation is not None else "executed_without_oracle",
        "input": input_kind,
        "block_range": [args.start_block, args.end_block],
        "shape": {
            "batch": inputs.hidden_states.shape[0],
            "image_tokens": inputs.hidden_states.shape[1],
            "hidden_size": inputs.hidden_states.shape[2],
            "context_tokens": inputs.encoder_hidden_states.shape[1],
            "context_dim": inputs.encoder_hidden_states.shape[2],
        },
        "dispatches": final.dispatch_count,
        "dispatches_per_block": final.dispatch_count / len(final.blocks),
        "populate_or_validate_ms": populate_ms,
        "cpu_oracle_ms": validation.cpu.wall_ms if validation else None,
        "cached_run_ms": [wall for wall, _ in repeated],
        "steady_median_ms": sorted(wall for wall, _ in repeated)[len(repeated) // 2],
        "profile": {
            "h2d_staging_bytes": sum(metric.h2d_bytes for metric in metrics),
            "d2h_staging_bytes": sum(metric.d2h_bytes for metric in metrics),
            "allocations": sum(metric.allocation_count for metric in metrics),
            "resident_hits": sum(metric.resident_hits for metric in metrics),
            "host_copy_ms": sum(metric.host_copy_ms for metric in metrics),
            "cache_sync_ms": sum(
                metric.h2d_sync_ms + metric.d2h_sync_ms for metric in metrics
            ),
            "kernel_and_wait_ms": sum(metric.kernel_sync_ms for metric in metrics),
            "host_operations_ms": sum(
                metric.wall_ms for metric in metrics if metric.device == "host"
            ),
            "npu_logical_flops": sum(
                metric.estimated_flops for metric in metrics if metric.device == "xdna2"
            ),
        },
        "block_wall_ms": [
            {"block": block.index, "wall_ms": block.wall_ms}
            for block in final.blocks
        ],
        "worst_block_error": worst,
        "final_error": errors[-1] if errors else None,
        "block_errors": errors,
    }
    if args.profile:
        payload["stages"] = [
            {
                "block": block.index,
                "stages": [metric.__dict__ for metric in block.result.metrics],
            }
            for block in final.blocks
        ]
    print(json.dumps(payload, indent=2))
    return 0
