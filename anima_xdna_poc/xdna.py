"""Thin, explicit Triton-XDNA runtime boundary."""

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.metadata
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Optional

import torch

from .errors import (
    CompilationFailure,
    DependencyUnavailable,
    ExecutionFailure,
    NPUUnavailable,
    UnsupportedTensor,
)
from .linear import PreparedLinear


@dataclass(frozen=True)
class ProbeResult:
    triton_xdna_version: str
    target: str
    runtime: str


@dataclass(frozen=True)
class ExecutionResult:
    output_bf16: torch.Tensor
    compile_and_first_run_ms: float
    median_run_ms: float
    artifact_cache: Optional[str]


def _imports() -> tuple[Any, Any, Any, Any]:
    try:
        import triton
        from triton.backends.amd_triton_npu.config import npu_config
        from triton.backends.amd_triton_npu.driver import (
            NPUDriver,
            detect_npu_version,
        )
    except (ImportError, ModuleNotFoundError) as error:
        raise DependencyUnavailable(
            "Triton-XDNA is not importable. Install its Windows wheel and XRT/pyxrt "
            "prerequisites described in anima_xdna_poc/README.md. "
            f"Original error: {error}"
        ) from error
    return triton, npu_config, NPUDriver, detect_npu_version


def probe() -> ProbeResult:
    _, npu_config, NPUDriver, detect_npu_version = _imports()
    try:
        target = detect_npu_version(runtime="xrt")
        NPUDriver("xrt")
    except Exception as error:
        raise NPUUnavailable(
            "XDNA 2 was not detected through the XRT runtime. Verify the AMD NPU "
            "driver, XRT SDK, pyxrt.pyd, and `xrt-smi examine`. "
            f"Original error: {error}"
        ) from error
    if target != "npu2":
        raise NPUUnavailable(
            f"detected {target!r}; this Windows-first proof of concept supports "
            "XDNA 2/AIE2P (npu2) execution only"
        )
    try:
        version = importlib.metadata.version("triton-xdna")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return ProbeResult(version, target, npu_config.runtime)


def _validate_prepared(prepared: PreparedLinear) -> tuple[int, int, int]:
    m, n, k = prepared.padded_shape
    if any(dimension % 8 for dimension in (m, n, k)):
        raise UnsupportedTensor(
            f"AIE2P BF16 matmul dimensions must be divisible by 8, got M={m}, N={n}, K={k}"
        )
    if prepared.input_bf16.dtype != torch.bfloat16:
        raise UnsupportedTensor("prepared input must be bfloat16")
    if prepared.weight_k_n_bf16.dtype != torch.bfloat16:
        raise UnsupportedTensor("prepared weight must be bfloat16")
    return m, n, k


@contextmanager
def _compiler_scope():
    from triton.backends.amd_triton_npu.config import config_context
    from triton.backends.amd_triton_npu.matmul_transform import (
        generate_matmul_transform,
    )

    build_root = Path.home() / ".cache" / "anima-xdna"
    air_project = build_root / "air_project"
    transform = build_root / "transform_aie2p_ascii.mlir"
    air_project.mkdir(parents=True, exist_ok=True)
    script = generate_matmul_transform(l1_m=64, l1_n=64, l2_k=64)
    transform.write_text(
        script.encode("ascii", errors="replace").decode("ascii"),
        encoding="ascii",
    )
    previous_cwd = Path.cwd()
    try:
        os.chdir(build_root)
        with config_context(
            air_project_path=str(air_project),
            transform_tiling_script=str(transform),
        ):
            yield
    finally:
        os.chdir(previous_cwd)


def execute(prepared: PreparedLinear, warmup: int = 1, runs: int = 5) -> ExecutionResult:
    if warmup < 0 or runs <= 0:
        raise UnsupportedTensor("warmup must be non-negative and runs must be positive")
    m, n, k = _validate_prepared(prepared)
    triton, _, NPUDriver, _ = _imports()
    probe()
    try:
        from .kernel import bf16_linear_kernel
    except Exception as error:
        raise DependencyUnavailable(f"cannot import the Triton kernel: {error}") from error

    output_fp32 = torch.empty((m, n), dtype=torch.float32)
    grid = (triton.cdiv(m, 256), triton.cdiv(n, 256))
    previous_driver = triton.runtime.driver._active
    compiled_kernel = None
    try:
        triton.runtime.driver.set_active(NPUDriver("xrt"))
        with _compiler_scope():
            started = time.perf_counter()
            try:
                compiled_kernel = bf16_linear_kernel[grid](
                    prepared.input_bf16,
                    prepared.weight_k_n_bf16,
                    output_fp32,
                    M=m,
                    N=n,
                    K=k,
                    stride_im=prepared.input_bf16.stride(0),
                    stride_ik=prepared.input_bf16.stride(1),
                    stride_wk=prepared.weight_k_n_bf16.stride(0),
                    stride_wn=prepared.weight_k_n_bf16.stride(1),
                    stride_om=output_fp32.stride(0),
                    stride_on=output_fp32.stride(1),
                    BLOCK_M=256,
                    BLOCK_N=256,
                    BLOCK_K=k,
                )
            except Exception as error:
                if isinstance(error, subprocess.CalledProcessError):
                    raise CompilationFailure(
                        f"Triton-XDNA could not compile the BF16 Linear kernel: {error}"
                    ) from error
                raise ExecutionFailure(f"XDNA 2 kernel execution failed: {error}") from error
            first_ms = (time.perf_counter() - started) * 1000.0

            timings = []
            for index in range(warmup + runs):
                started = time.perf_counter()
                try:
                    bf16_linear_kernel[grid](
                        prepared.input_bf16,
                        prepared.weight_k_n_bf16,
                        output_fp32,
                        M=m,
                        N=n,
                        K=k,
                        stride_im=prepared.input_bf16.stride(0),
                        stride_ik=prepared.input_bf16.stride(1),
                        stride_wk=prepared.weight_k_n_bf16.stride(0),
                        stride_wn=prepared.weight_k_n_bf16.stride(1),
                        stride_om=output_fp32.stride(0),
                        stride_on=output_fp32.stride(1),
                        BLOCK_M=256,
                        BLOCK_N=256,
                        BLOCK_K=k,
                    )
                except Exception as error:
                    raise ExecutionFailure(
                        f"XDNA 2 kernel execution failed: {error}"
                    ) from error
                elapsed = (time.perf_counter() - started) * 1000.0
                if index >= warmup:
                    timings.append(elapsed)
    finally:
        triton.runtime.driver.set_active(previous_driver)

    cache_dir = None
    if compiled_kernel is not None:
        try:
            from triton.backends.amd_triton_npu.driver import get_npu_cache_dir

            resolved_cache = get_npu_cache_dir(compiled_kernel)
            if resolved_cache is not None:
                cache_dir = str(resolved_cache)
        except (ImportError, TypeError, AttributeError):
            pass
    if cache_dir is None:
        try:
            from triton.backends.amd_triton_npu import driver as npu_driver

            module_path = Path(npu_driver._last_dispatched_module.__file__)
            cache_dir = str(module_path.parent)
        except (AttributeError, TypeError):
            pass
    timings.sort()
    median = timings[len(timings) // 2]
    output = output_fp32[: prepared.rows, : prepared.out_features].to(torch.bfloat16)
    return ExecutionResult(output, first_ms, median, cache_dir)
