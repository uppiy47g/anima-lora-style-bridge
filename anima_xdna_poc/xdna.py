"""Thin, explicit Triton-XDNA runtime boundary."""

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.metadata
import math
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


@dataclass(frozen=True)
class DispatchResult:
    output_bf16: torch.Tensor
    output_fp32: torch.Tensor
    wall_ms: float
    artifact_cache: Optional[str]
    h2d_bytes: int = 0
    d2h_bytes: int = 0
    allocation_count: int = 0
    resident_hit: bool = False
    host_copy_ms: float = 0.0
    h2d_sync_ms: float = 0.0
    kernel_sync_ms: float = 0.0
    d2h_sync_ms: float = 0.0
    weight_population_ms: float = 0.0
    weight_population_bytes: int = 0


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


def _artifact_cache(compiled_kernel: Any) -> Optional[str]:
    if compiled_kernel is not None:
        try:
            from triton.backends.amd_triton_npu.driver import get_npu_cache_dir

            resolved_cache = get_npu_cache_dir(compiled_kernel)
            if resolved_cache is not None:
                return str(resolved_cache)
        except (ImportError, TypeError, AttributeError):
            pass
    try:
        from triton.backends.amd_triton_npu import driver as npu_driver

        module_path = Path(npu_driver._last_dispatched_module.__file__)
        return str(module_path.parent)
    except (AttributeError, TypeError):
        return None


def _is_compilation_error(error: Exception) -> bool:
    error_type = type(error)
    return (
        isinstance(error, subprocess.CalledProcessError)
        or "compilation" in error_type.__name__.lower()
        or error_type.__module__.startswith("triton.compiler")
        or "error encountered during parsing" in str(error).lower()
    )


class XDNASession:
    """Keep one driver/compiler scope active across a multi-dispatch block."""

    def __init__(self):
        self.triton = None
        self._previous_driver = None
        self._compiler = None
        self._head_chains = {}
        self._qkv_chains = {}
        self._qkv_output_buffers = {}

    def __enter__(self):
        self.triton, _, NPUDriver, _ = _imports()
        probe()
        self._previous_driver = self.triton.runtime.driver._active
        self.triton.runtime.driver.set_active(NPUDriver("xrt"))
        self._compiler = _compiler_scope()
        self._compiler.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            for chain in self._head_chains.values():
                chain.close()
            self._head_chains.clear()
            for chain in self._qkv_chains.values():
                chain.close()
            self._qkv_chains.clear()
            for buffers in self._qkv_output_buffers.values():
                for buffer in buffers:
                    buffer.close()
            self._qkv_output_buffers.clear()
            if self._compiler is not None:
                self._compiler.__exit__(exc_type, exc_value, traceback)
        finally:
            if self.triton is not None:
                self.triton.runtime.driver.set_active(self._previous_driver)

    def dispatch(self, prepared: PreparedLinear) -> DispatchResult:
        if self.triton is None:
            raise ExecutionFailure("XDNASession must be entered before dispatch")
        m, n, k = _validate_prepared(prepared)
        try:
            from .kernel import bf16_linear_kernel
        except Exception as error:
            raise DependencyUnavailable(f"cannot import the Triton kernel: {error}") from error
        output_fp32 = torch.empty((m, n), dtype=torch.float32)
        grid = (self.triton.cdiv(m, 256), self.triton.cdiv(n, 256))
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
            if _is_compilation_error(error):
                raise CompilationFailure(
                    f"Triton-XDNA could not compile the BF16 kernel: {error}"
                ) from error
            raise ExecutionFailure(f"XDNA 2 kernel execution failed: {error}") from error
        elapsed = (time.perf_counter() - started) * 1000
        output = output_fp32[: prepared.rows, : prepared.out_features].to(torch.bfloat16)
        return DispatchResult(
            output,
            output_fp32[: prepared.rows, : prepared.out_features],
            elapsed,
            _artifact_cache(compiled_kernel),
            prepared.input_bf16.numel() * 2
            + prepared.weight_k_n_bf16.numel() * 2
            + output_fp32.numel() * 4,
            output_fp32.numel() * 4,
            3,
        )

    def dispatch_head_chain(
        self,
        input_h_m_k: torch.Tensor,
        weight_h_k_n: torch.Tensor,
        key: str,
    ) -> DispatchResult:
        if self.triton is None:
            raise ExecutionFailure("XDNASession must be entered before dispatch")
        import numpy as np
        from triton.backends.amd_triton_npu.multilaunch import NPUChain
        from .kernel import bf16_head_matmul_kernel

        heads, rows, inner = input_h_m_k.shape
        if weight_h_k_n.shape[:2] != (heads, inner):
            raise UnsupportedTensor("head-chain matmul dimensions do not match")
        columns = weight_h_k_n.shape[-1]
        padded_rows = math.ceil(rows / 256) * 256
        padded_inner = max(math.ceil(inner / 64) * 64, 256)
        padded_columns = math.ceil(columns / 256) * 256
        if padded_columns == 256 and padded_inner > 256:
            padded_columns = 512
        prepared_input = torch.zeros(
            heads, padded_rows, padded_inner, dtype=torch.bfloat16
        )
        prepared_weight = torch.zeros(
            heads, padded_inner, padded_columns, dtype=torch.bfloat16
        )
        prepared_input[:, :rows, :inner] = input_h_m_k.to(torch.bfloat16)
        prepared_weight[:, :inner, :columns] = weight_h_k_n.to(torch.bfloat16)
        signature = (heads, padded_rows, padded_columns, padded_inner)
        chain = self._head_chains.get(signature)
        if chain is None:
            chain = NPUChain(
                "anima_heads_" + "_".join(str(value) for value in signature),
                air_project_path=str(Path.home() / ".cache" / "anima-xdna" / "air_project"),
            )
            output_template = torch.empty(
                heads, padded_rows, padded_columns, dtype=torch.float32
            )
            transform = str(
                Path.home() / ".cache" / "anima-xdna" / "transform_aie2p_ascii.mlir"
            )
            grid = (
                self.triton.cdiv(padded_rows, 256),
                self.triton.cdiv(padded_columns, 256),
            )
            for head in range(heads):
                chain.add(
                    bf16_head_matmul_kernel,
                    grid,
                    {0: 0, 1: 1, 2: 2},
                    args=(prepared_input, prepared_weight, output_template),
                    constexprs={
                        "M": padded_rows,
                        "N": padded_columns,
                        "K": padded_inner,
                        "stride_ih": prepared_input.stride(0),
                        "stride_im": prepared_input.stride(1),
                        "stride_ik": prepared_input.stride(2),
                        "stride_wh": prepared_weight.stride(0),
                        "stride_wk": prepared_weight.stride(1),
                        "stride_wn": prepared_weight.stride(2),
                        "stride_oh": output_template.stride(0),
                        "stride_om": output_template.stride(1),
                        "stride_on": output_template.stride(2),
                        "BLOCK_M": 256,
                        "BLOCK_N": 256,
                        "BLOCK_K": padded_inner,
                        "HEAD": head,
                    },
                    transform_script=transform,
                )
            self._head_chains[signature] = chain
        arrays = [
            ResidentXDNASession._bf16_numpy(prepared_input),
            ResidentXDNASession._bf16_numpy(prepared_weight),
            np.empty((heads, padded_rows, padded_columns), dtype=np.float32),
        ]
        started = time.perf_counter()
        try:
            output = chain.run(
                arrays,
                bo_key="shape",
                intermediate_indices={2},
                output_indices={2},
            )[2]
            chain._anima_populated = True
        except Exception as error:
            if _is_compilation_error(error):
                raise CompilationFailure(
                    f"Triton-XDNA could not compile the head chain: {error}"
                ) from error
            raise ExecutionFailure(f"XDNA head-chain execution failed: {error}") from error
        elapsed = (time.perf_counter() - started) * 1000
        logical = torch.from_numpy(output)[:, :rows, :columns]
        return DispatchResult(
            logical.to(torch.bfloat16),
            logical,
            elapsed,
            None,
            prepared_input.numel() * 2 + prepared_weight.numel() * 2,
            output.nbytes,
            0,
            True,
            kernel_sync_ms=elapsed,
        )

    def dispatch_qkv_chain(
        self,
        prepared_qkv: tuple[PreparedLinear, PreparedLinear, PreparedLinear],
        key: str,
        shared_input: bool,
    ) -> dict[str, Any]:
        if self.triton is None:
            raise ExecutionFailure("XDNASession must be entered before dispatch")
        import numpy as np
        from triton.backends.amd_triton_npu.multilaunch import NPUChain
        from .kernel import bf16_linear_kernel

        q, k, v = prepared_qkv
        prepared = (q, k, v)
        for item in prepared:
            _validate_prepared(item)
        if shared_input and (
            q.input_bf16.shape != k.input_bf16.shape
            or q.input_bf16.shape != v.input_bf16.shape
        ):
            raise UnsupportedTensor("shared Q/K/V input layouts must match")
        signature = (
            shared_input,
            tuple(item.padded_shape for item in prepared),
        )
        chain = self._qkv_chains.get(signature)
        if chain is None:
            chain_name = "anima_qkv_" + "_".join(
                str(value)
                for shape in signature[1]
                for value in shape
            ) + ("_shared" if shared_input else "_separate")
            chain = NPUChain(
                chain_name,
                air_project_path=str(
                    Path.home() / ".cache" / "anima-xdna" / "air_project"
                ),
            )
            outputs = tuple(
                torch.empty(item.padded_shape[:2], dtype=torch.float32)
                for item in prepared
            )
            transform = str(
                Path.home() / ".cache" / "anima-xdna" / "transform_aie2p_ascii.mlir"
            )
            argument_maps = (
                ((0, 1, 2), (0, 3, 4), (0, 5, 6))
                if shared_input
                else ((0, 1, 2), (3, 4, 5), (3, 6, 7))
            )
            for index, item in enumerate(prepared):
                input_index, weight_index, output_index = argument_maps[index]
                grid = (
                    self.triton.cdiv(item.padded_shape[0], 256),
                    self.triton.cdiv(item.padded_shape[1], 256),
                )
                chain.add(
                    bf16_linear_kernel,
                    grid,
                    {0: input_index, 1: weight_index, 2: output_index},
                    args=(item.input_bf16, item.weight_k_n_bf16, outputs[index]),
                    constexprs={
                        "M": item.padded_shape[0],
                        "N": item.padded_shape[1],
                        "K": item.padded_shape[2],
                        "stride_im": item.input_bf16.stride(0),
                        "stride_ik": item.input_bf16.stride(1),
                        "stride_wk": item.weight_k_n_bf16.stride(0),
                        "stride_wn": item.weight_k_n_bf16.stride(1),
                        "stride_om": outputs[index].stride(0),
                        "stride_on": outputs[index].stride(1),
                        "BLOCK_M": 256,
                        "BLOCK_N": 256,
                        "BLOCK_K": item.padded_shape[2],
                    },
                    transform_script=transform,
                )
            self._qkv_chains[signature] = chain

        arrays = []
        for index, item in enumerate(prepared):
            if index == 0:
                arrays.append(ResidentXDNASession._bf16_numpy(item.input_bf16))
            elif not shared_input and index == 1:
                arrays.append(ResidentXDNASession._bf16_numpy(item.input_bf16))
            arrays.extend(
                (
                    ResidentXDNASession._bf16_numpy(item.weight_k_n_bf16),
                    np.empty(item.padded_shape[:2], dtype=np.float32),
                )
            )
        output_indices = {2, 4, 6} if shared_input else {2, 5, 7}
        static_indices = {1, 3, 5} if shared_input else {1, 4, 6}
        populated_keys = getattr(chain, "_anima_populated_keys", set())
        first_call = key not in populated_keys
        output_buffers = self._qkv_output_buffers.get(signature)
        output_buffers_created = output_buffers is None
        if output_buffers is None:
            from triton.backends.amd_triton_npu import shared

            created_buffers = []
            try:
                for item in prepared:
                    created_buffers.append(
                        shared.empty(
                            item.padded_shape[:2],
                            dtype=torch.float32,
                            device="xrt:0",
                        )
                    )
            except Exception:
                for buffer in created_buffers:
                    buffer.close()
                raise
            output_buffers = tuple(created_buffers)
            self._qkv_output_buffers[signature] = output_buffers
        bound_buffers = {}
        for buffer_index, output_index in enumerate(sorted(output_indices)):
            buffer = output_buffers[buffer_index]
            arrays[output_index] = buffer.numpy()
            bound_buffers[output_index] = buffer.bo
        started = time.perf_counter()
        try:
            outputs = chain.run(
                arrays,
                bo_key=key,
                static_indices=static_indices,
                intermediate_indices=output_indices,
                output_indices=output_indices,
                bound_buffers=bound_buffers,
            )
            populated_keys.add(key)
            chain._anima_populated_keys = populated_keys
        except Exception as error:
            if _is_compilation_error(error):
                raise CompilationFailure(
                    f"Triton-XDNA could not compile the Q/K/V chain: {error}"
                ) from error
            raise ExecutionFailure(f"XDNA Q/K/V chain execution failed: {error}") from error
        elapsed = (time.perf_counter() - started) * 1000
        output_indices_order = (2, 4, 6) if shared_input else (2, 5, 7)
        logical_outputs = tuple(
            torch.from_numpy(outputs[index])[
                : item.rows, : item.out_features
            ].to(torch.bfloat16)
            for item, index in zip(prepared, output_indices_order)
        )
        dynamic_indices = (0,) if shared_input else (0, 3)
        h2d_bytes = sum(arrays[index].nbytes for index in dynamic_indices)
        if first_call:
            h2d_bytes += sum(arrays[index].nbytes for index in static_indices)
        d2h_bytes = sum(arrays[index].nbytes for index in output_indices_order)
        return {
            "outputs": logical_outputs,
            "wall_ms": elapsed,
            "dispatches": 1,
            "h2d_bytes": h2d_bytes,
            "d2h_bytes": d2h_bytes,
            "allocation_count": (
                len(static_indices)
                + len(dynamic_indices)
                + (3 if output_buffers_created else 0)
            )
            if first_call
            else 0,
            "resident_hits": 0 if first_call else len(static_indices),
            "weight_population_bytes": sum(
                arrays[index].nbytes for index in static_indices
            )
            if first_call
            else 0,
        }


class _ResidentKernelRunner:
    def __init__(self, artifact_cache: str):
        import pyxrt

        cache = Path(artifact_cache)
        kernel_name = (cache / "elf_kernel_name.txt").read_text().strip()
        self.xrt = pyxrt
        self.device = pyxrt.device(0)
        self.elf = pyxrt.elf(str(cache / "aie.elf"))
        self.context = pyxrt.hw_context(self.device, self.elf)
        self.kernel = pyxrt.ext.kernel(self.context, kernel_name)
        self.dynamic_bos: Optional[tuple[list[int], Any, Any]] = None
        self.weight_bos: dict[str, tuple[int, Any]] = {}

    def run(
        self,
        arrays: list[Any],
        key: str,
        static_indices: set[int],
        scratch_indices: set[int],
        output_index: int,
    ) -> tuple[Any, dict[str, Any]]:
        import numpy as np

        sizes = [array.size * array.itemsize for array in arrays]
        first_weight = key not in self.weight_bos
        allocation_started = time.perf_counter()
        allocation_count = 0
        if self.dynamic_bos is None:
            input_bo = self.xrt.ext.bo(self.device, sizes[0])
            output_bo = self.xrt.ext.bo(self.device, sizes[2])
            self.dynamic_bos = ([sizes[0], sizes[2]], input_bo, output_bo)
            allocation_count += 2
        else:
            dynamic_sizes, input_bo, output_bo = self.dynamic_bos
            if dynamic_sizes != [sizes[0], sizes[2]]:
                raise ExecutionFailure(
                    f"resident dynamic buffers changed sizes from "
                    f"{dynamic_sizes} to {[sizes[0], sizes[2]]}"
                )
        if first_weight:
            self.weight_bos[key] = (sizes[1], self.xrt.ext.bo(self.device, sizes[1]))
            allocation_count += 1
        elif self.weight_bos[key][0] != sizes[1]:
            raise ExecutionFailure(
                f"resident weight key {key!r} changed size from "
                f"{self.weight_bos[key][0]} to {sizes[1]}"
            )
        weight_bo = self.weight_bos[key][1]
        bos = [input_bo, weight_bo, output_bo]
        allocation_ms = (time.perf_counter() - allocation_started) * 1000

        h2d_bytes = 0
        copy_ms = 0.0
        h2d_sync_ms = 0.0
        weight_population_ms = 0.0
        weight_population_bytes = 0
        for index, array in enumerate(arrays):
            if index in scratch_indices:
                continue
            if index in static_indices and not first_weight:
                continue
            started = time.perf_counter()
            source = np.frombuffer(array, dtype=np.uint8)
            target = np.frombuffer(bos[index].map(), dtype=np.uint8, count=len(source))
            np.copyto(target, source, casting="no")
            copied_ms = (time.perf_counter() - started) * 1000
            copy_ms += copied_ms
            started = time.perf_counter()
            bos[index].sync(
                self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
            )
            synced_ms = (time.perf_counter() - started) * 1000
            h2d_sync_ms += synced_ms
            h2d_bytes += sizes[index]
            if index in static_indices:
                weight_population_ms += copied_ms + synced_ms
                weight_population_bytes += sizes[index]

        started = time.perf_counter()
        run = self.xrt.run(self.kernel)
        for index, bo in enumerate(bos):
            run.set_arg(index, bo)
        run.start()
        run.wait2()
        kernel_sync_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        bos[output_index].sync(
            self.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        )
        d2h_sync_ms = (time.perf_counter() - started) * 1000
        template = arrays[output_index]
        output = np.frombuffer(
            bos[output_index].map(),
            dtype=template.dtype,
            count=template.size,
        ).reshape(template.shape)
        return output, {
            "h2d_bytes": h2d_bytes,
            "d2h_bytes": sizes[output_index],
            "allocation_count": allocation_count,
            "resident_hit": not first_weight,
            "allocation_ms": allocation_ms,
            "host_copy_ms": copy_ms,
            "h2d_sync_ms": h2d_sync_ms,
            "kernel_sync_ms": kernel_sync_ms,
            "d2h_sync_ms": d2h_sync_ms,
            "weight_population_ms": weight_population_ms,
            "weight_population_bytes": weight_population_bytes,
        }

    def close(self):
        self.dynamic_bos = None
        self.weight_bos = {}
        self.kernel = None
        self.context = None
        self.elf = None
        self.device = None


class ResidentXDNASession(XDNASession):
    """XRT BO-backed execution with weights and scratch persistent by op key."""

    def __init__(self):
        super().__init__()
        self._runners: dict[str, _ResidentKernelRunner] = {}
        self._artifacts: dict[tuple[Any, ...], str] = {}
        self._resident_specs: dict[str, tuple[str, int, int, int, int, int]] = {}

    @staticmethod
    def _bf16_numpy(tensor: torch.Tensor):
        import ml_dtypes

        return tensor.contiguous().view(torch.uint16).numpy().view(ml_dtypes.bfloat16)

    def _resident_run(
        self,
        compiled: DispatchResult,
        arrays: list[Any],
        key: str,
        logical_shape: tuple[int, ...],
    ) -> DispatchResult:
        if compiled.artifact_cache is None:
            raise ExecutionFailure("compiled kernel did not expose an artifact cache")
        runner = self._runners.get(compiled.artifact_cache)
        if runner is None:
            runner = _ResidentKernelRunner(compiled.artifact_cache)
            self._runners[compiled.artifact_cache] = runner
        started = time.perf_counter()
        output, profile = runner.run(
            arrays,
            key,
            static_indices={1},
            scratch_indices={2},
            output_index=2,
        )
        wall_ms = (time.perf_counter() - started) * 1000
        output_fp32 = torch.from_numpy(output)[
            tuple(slice(0, size) for size in logical_shape)
        ]
        return DispatchResult(
            output_fp32.to(torch.bfloat16),
            output_fp32,
            wall_ms,
            compiled.artifact_cache,
            **{name: profile[name] for name in (
                "h2d_bytes",
                "d2h_bytes",
                "allocation_count",
                "resident_hit",
                "host_copy_ms",
                "h2d_sync_ms",
                "kernel_sync_ms",
                "d2h_sync_ms",
                "weight_population_ms",
                "weight_population_bytes",
            )},
        )

    def dispatch_resident(self, prepared: PreparedLinear, key: str) -> DispatchResult:
        import numpy as np

        signature = ("linear",) + prepared.padded_shape
        artifact = self._artifacts.get(signature)
        if artifact is None:
            compiled = self.dispatch(prepared)
            if compiled.artifact_cache is None:
                raise ExecutionFailure("compiled kernel did not expose an artifact cache")
            artifact = compiled.artifact_cache
            self._artifacts[signature] = artifact
        else:
            compiled = DispatchResult(
                torch.empty(0, dtype=torch.bfloat16),
                torch.empty(0),
                0.0,
                artifact,
            )
        arrays = [
            self._bf16_numpy(prepared.input_bf16),
            self._bf16_numpy(prepared.weight_k_n_bf16),
            np.empty(
                (prepared.input_bf16.shape[0], prepared.weight_k_n_bf16.shape[1]),
                dtype=np.float32,
            ),
        ]
        result = self._resident_run(
            compiled,
            arrays,
            key,
            (prepared.rows, prepared.out_features),
        )
        self._resident_specs[key] = (
            artifact,
            prepared.input_bf16.shape[0],
            prepared.weight_k_n_bf16.shape[1],
            prepared.input_bf16.shape[1],
            prepared.rows,
            prepared.out_features,
        )
        return result

    def has_resident(self, key: str) -> bool:
        return key in self._resident_specs

    def dispatch_resident_input(
        self,
        input_tensor: torch.Tensor,
        key: str,
    ) -> DispatchResult:
        import ml_dtypes
        import numpy as np

        try:
            artifact, padded_rows, padded_columns, padded_inner, rows, columns = (
                self._resident_specs[key]
            )
        except KeyError as error:
            raise ExecutionFailure(f"resident weight is not populated: {key}") from error
        if input_tensor.ndim != 2 or input_tensor.shape[0] != rows:
            raise UnsupportedTensor("resident Linear input must be rank 2")
        if input_tensor.shape[1] > padded_inner:
            raise UnsupportedTensor(
                f"resident input width {input_tensor.shape[1]} exceeds {padded_inner}"
            )
        prepared_input = torch.zeros(
            padded_rows, padded_inner, dtype=torch.bfloat16
        )
        prepared_input[:rows, : input_tensor.shape[1]] = input_tensor.to(torch.bfloat16)
        compiled = DispatchResult(
            torch.empty(0, dtype=torch.bfloat16),
            torch.empty(0),
            0.0,
            artifact,
        )
        arrays = [
            self._bf16_numpy(prepared_input),
            np.empty((padded_inner, padded_columns), dtype=ml_dtypes.bfloat16),
            np.empty((padded_rows, padded_columns), dtype=np.float32),
        ]
        return self._resident_run(
            compiled,
            arrays,
            key,
            (rows, columns),
        )

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            for runner in self._runners.values():
                runner.close()
            self._runners.clear()
            self._artifacts.clear()
            self._resident_specs.clear()
        finally:
            super().__exit__(exc_type, exc_value, traceback)


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
                if _is_compilation_error(error):
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

    cache_dir = _artifact_cache(compiled_kernel)
    timings.sort()
    median = timings[len(timings) // 2]
    output = output_fp32[: prepared.rows, : prepared.out_features].to(torch.bfloat16)
    return ExecutionResult(output, first_ms, median, cache_dir)
