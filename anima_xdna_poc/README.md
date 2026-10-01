# Anima block 0 on AMD XDNA 2

This isolated proof of concept executes either one Anima Linear or the complete
first Anima/Cosmos transformer block directly from its original safetensors on
XDNA 2 through Triton-XDNA and XRT. It does **not** use ONNX, Vitis AI EP, or a
converted model format, and it does not modify the existing LoRA bridge,
ComfyUI, or model files.

## Supported model and exact block

The block adapter follows the local Diffusers `CosmosTransformer3DModel`
implementation used by Anima Base v1.0 (`diffusers 0.39.0.dev0`):

1. AdaLN-Zero: SiLU, 2048->256->6144, affine-free LayerNorm (`eps=1e-6`),
   shift/scale/gate, and the model-level `temb`.
2. Self-attention: 2048-wide Q/K/V, 16 heads of 128, learned Q/K RMSNorm
   (`eps=1e-6`), Cosmos image RoPE, non-causal attention, and output projection.
3. Cross-attention: 2048-wide Q, 1024-wide text K/V, 16x128 heads, Q/K
   RMSNorm, optional `[B,1,1,S]` mask, and output projection.
4. MLP: exact GELU (`approximate="none"`) with 2048->8192->2048 projections.
5. A gated residual follows each attention and MLP stage.

The loader validates all 20 block tensors and materializes only those tensors
through `safetensors.safe_open`; the rest of the 3.9 GB checkpoint is not
loaded. The original checkpoint remains the source of truth. Native Triton
cache entries contain only compiled kernels and layout metadata.

Positive compatibility is currently limited to the full **Anima Base v1.0
Diffusers** checkpoint. Locally discovered Aesthetic and Turbo files are LoRA
adapters over Base, not independent full transformer checkpoints. Their block
shape appears inherited from Base, but LoRA application is not implemented in
this milestone.

## NPU and host boundary

Every block Linear and every QK-transpose/attention-value GEMM runs on XDNA 2.
The current non-fused implementation uses 83 XRT dispatches: 16 model Linear
dispatches, 64 per-head attention GEMMs, and three extra dispatches because the
8192-wide MLP reduction is split into four FP32 partials. Triton-XDNA currently
rejects tensors larger than 1,048,576 elements during lowering, which prevents
a whole-K 8192 reduction. Inputs, weights, and outputs are BF16; accumulation
and split-reduction summation are FP32.

SiLU, LayerNorm/modulation, RMSNorm, RoPE, mask/scale/softmax, GELU, tensor
layout, gates, and residual additions remain explicit PyTorch CPU operations.
The CLI reports each stage's device and wall time, the total dispatch count,
logical GEMM FLOPs, source-tensor transfer estimate, and residual-stage
numerical errors. There is no silent CPU fallback.

The deterministic default fixture matches a 512x512 image path after VAE and
`[1,2,2]` patching: batch 1, 1024 image tokens, hidden size 2048, 512 prompt
tokens, and context width 1024. It uses synthetic deterministic patchified
hidden/context/timestep inputs rather than a captured generation activation;
the block semantics, shapes, real weights, mask contract, and Cosmos RoPE are
otherwise exact.

## Windows prerequisites

- Windows 10/11 x64 and AMD XDNA 2/AIE2P (`npu2`) with a current AMD NPU
  driver.
- Python 3.13 is recommended because the official Windows `pyxrt.pyd` uses that
  ABI.
- Visual Studio 2022 Desktop development with C++, CMake, and Ninja.
- An official [Xilinx/XRT release](https://github.com/Xilinx/XRT/releases)
  Windows SDK. Extract the inner `xrt` directory to a user-writable location,
  set `XRT_DEV_DIR`, and copy only its ABI-matching `pyxrt.pyd` into the
  dedicated virtual environment.
- Official AMD Triton-XDNA, MLIR-AIR, MLIR-AIE, LLVM-AIE, and PyTorch CPU
  wheels.

Validated versions are Python 3.13.9, Triton-XDNA
`3.6.0.2026093004+75679dc`, and XRT 2.21.75. The XRT
`xrt_windows_sdk.zip` SHA-256 is
`ccc244c2c423588972ade76142cdc01049477aaa39a35be97e782b97eb7c5295`.
Do not set `XILINX_XRT` on Windows; use `XRT_DEV_DIR`.

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

MLIR-AIE uses relative intermediates and upstream generated transforms contain
UTF-8 punctuation. The runtime stages compilation under the ASCII-only
`%USERPROFILE%\.cache\anima-xdna` path so Japanese repository paths work
without a global encoding change.

## Commands and output

```powershell
$env:XRT_DEV_DIR = "$env:LOCALAPPDATA\AMD\xrt-2.21.75"
anima-xdna-probe
anima-xdna-linear D:\models\Anima\transformer\diffusion_pytorch_model.safetensors
anima-xdna-block D:\models\Anima\transformer\diffusion_pytorch_model.safetensors `
  --image-tokens 1024 --context-tokens 512 --runs 2
```

`anima-xdna-probe` must report `target: npu2` and `runtime: xrt`.
`anima-xdna-block` reports `status: match`, exact shape, NPU/host operation
lists, compile/cached run timing, 83 dispatches, transfer and FLOP estimates,
artifact caches, final max/mean errors, and errors after self-attention,
cross-attention, and MLP.

On the validated Ryzen AI 9 365 / NPU Strix system, Anima Base block 0 at
1024/512 tokens produced:

- first compile/populate run: 57.002 s;
- cached process-first run: 1.570 s;
- cached steady run: 0.662 s;
- CPU oracle: 0.717 s;
- logical NPU GEMM work: 137,451,536,384 FLOPs;
- estimated source payload transfer: 0.342 GiB;
- max/mean absolute output error: `9.765625e-4` / `2.131443e-5`;
- residual-stage max errors: `4.882813e-4`, `4.882813e-4`,
  `9.765625e-4`.

The default tolerance is `rtol=0.02`, `atol=0.003`, and the observed error is
below it. The tolerance accounts for cumulative BF16 rounding over all three
residual stages; it was not widened to mask a failing result. A
small-token block was also compared with the installed authoritative Diffusers
module, with maximum CPU-oracle difference `4.882813e-4`.

Failures use distinct categories and exit codes: dependency unavailable (2),
NPU unavailable (3), unsupported shape/config/dtype (4), compilation failure
(5), execution failure (6), and numerical mismatch (7).

## Tests and limitations

CPU tests cover exact key/config discovery, lazy block loading, shape rejection,
layout/padding, deterministic inputs, RoPE, attention masks, and the complete
CPU block residual path. The real-NPU test is enabled by:

```powershell
$env:ANIMA_XDNA_CHECKPOINT = "D:\models\Anima\transformer\diffusion_pytorch_model.safetensors"
python -m unittest tests.test_xdna_poc -v
```

Without that variable or the toolchain, it skips with the precise missing
boundary.

This is not yet a ComfyUI backend or full image generator. It does not capture
live pipeline activations, apply LoRAs, execute patch/time embedding, or run all
28 blocks. Fine-grained head dispatch and repeated host/NPU transfers dominate
runtime; the CPU oracle is currently competitive after cache warmup. The next
milestone is a batched/fused attention kernel plus persistent device buffers,
then chaining all 28 blocks behind a ComfyUI MODEL wrapper while retaining the
same safetensors/LoRA source format.
