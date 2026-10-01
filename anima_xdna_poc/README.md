# Anima Linear on AMD XDNA 2

This isolated proof of concept reads one Anima DiT Linear weight directly from
its original safetensors checkpoint, runs the projection through Triton-XDNA on
XDNA 2, and compares the result with a PyTorch CPU oracle. It does **not** use
ONNX, Vitis AI EP, or a converted model format, and it does not alter the LoRA
bridge.

## Scope and data path

The default target is block 0 self-attention `q_proj`. Canonical checkpoint
aliases and an unambiguous suffix fallback are supported; `--key` accepts the
actual key when a checkpoint differs. `safetensors.safe_open` maps the file and
materializes only the selected tensor.

The checkpoint weight is `[out_features, in_features]`. The prototype converts
only that tensor to BF16, transposes it to contiguous `[K, N]`, creates a
deterministic BF16 `[M, K]` input, and pads transient M/N buffers to the
mask-free kernel's 256x256 launch tile and K to the AIE2P BF16 MAC alignment
of 8. Triton-XDNA auto-generates the matmul transform
schedule, compiles through MLIR-AIR/MLIR-AIE, and dispatches through XRT.
Accumulation is FP32; both NPU and CPU results are rounded to BF16 before
comparison. Triton's native cache stores compiled XDNA artifacts; the original
safetensors remains the only weight source.

## Windows prerequisites

- Windows 10/11 x64 and an AMD XDNA 2 / AIE2P NPU (`npu2`, including Strix,
  Strix Halo, or Krackan) with the current AMD NPU driver.
- Python 3.11-3.14. Triton-XDNA currently recommends Python 3.13 because the
  supplied `pyxrt.pyd` targets it; other versions may require building pyxrt.
- Visual Studio 2022 or newer with **Desktop development with C++**.
- The XRT Windows SDK from a
  [Xilinx/XRT release](https://github.com/Xilinx/XRT/releases). Extract its
  inner `xrt_sdk\xrt` directory to `C:\Program Files\AMD\xrt`, or set
  `XRT_DEV_DIR`. Copy the matching `python\pyxrt.pyd` into the environment's
  `site-packages`.
- Triton-XDNA 0.1 or a known source revision. The API is experimental; this
  prototype was written against
  [`amd/Triton-XDNA`](https://github.com/amd/Triton-XDNA) main at
  `51c55a0f92e77b5be24de67180e937ead475dd9c`.

Do not set `XILINX_XRT` on Windows; use `XRT_DEV_DIR` or
`AMD_TRITON_NPU_XRT_DIR`.

```powershell
py -3.13 -m venv .venv-xdna
.\.venv-xdna\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install triton-xdna `
  --find-links https://github.com/amd/Triton-XDNA/releases/expanded_assets/latest-wheels `
  --find-links https://github.com/Xilinx/mlir-aie/releases/expanded_assets/latest-wheels-no-rtti-2 `
  --find-links https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly `
  --find-links https://github.com/Xilinx/mlir-air/releases/expanded_assets/latest-air-wheels-no-rtti
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[xdna-poc]"
```

## Commands

```powershell
anima-xdna-probe
anima-xdna-linear D:\models\anima.safetensors --rows 256 --seed 0
```

A successful probe prints JSON with `"status": "ready"`, `"target": "npu2"`,
and `"runtime": "xrt"`. A successful validation prints `"status": "match"`,
the selected key and logical/padded shapes, timing, tolerance, and Triton's
artifact-cache location.

Failures are never silently sent to CPU. The CLIs label and return distinct
codes for dependency unavailable (2), NPU unavailable (3), unsupported
shape/dtype (4), compilation failure (5), execution failure (6), and numerical
mismatch (7). The reported benchmark is host wall time; Triton-XDNA's backend
does not support CUDA-oriented `triton.testing.do_bench`.

## Current limitations

- This is one bias-free Linear projection, not one transformer block.
- Only Windows XRT execution on XDNA 2/AIE2P is supported here. XDNA 1/AIE2
  Windows execution is not supported upstream.
- LoRA application, bias handling, dynamic production activations, ComfyUI
  integration, a full DiT forward pass, and image generation are not
  implemented.
- Triton-XDNA is beta software with internal Python import paths and no stable
  API compatibility guarantee.
- Large K values compile as a whole-K Triton block, matching the upstream BF16
  example. Real checkpoint validation on target hardware is required before
  expanding this approach.

The next milestone is one complete Anima transformer block, reusing the
safetensors weights and replacing its constituent operations with validated
XDNA 2 kernels before any ComfyUI backend integration.
