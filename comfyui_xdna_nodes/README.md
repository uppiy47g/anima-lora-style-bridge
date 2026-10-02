# Anima XDNA 2 ComfyUI MODEL wrapper

This custom-node package replaces only the 28 `transformer_blocks` in ComfyUI's
native Anima Base v1.0 or Turbo V1.1 diffusion model with the resident Triton-XDNA/XRT
runtime. It does **not** use ONNX or Vitis AI EP.

For an isolated Windows setup, sanitized launcher/model-path templates, the
full verification ladder, image evidence, and machine-readable measurements,
see the [public reproducibility report](../docs/anima-xdna-reproducibility.md).

ComfyUI still executes:

- Qwen 3 0.6B tokenization and the Anima LLM adapter
- latent patch embedding, timestep embedding, and RoPE setup
- LayerNorm/AdaLN, RMSNorm, softmax, GELU, gates, and residual operations
- final projection/unpatchify, sampler orchestration, and Qwen Image VAE

XDNA 2 executes all 644 Linear/QK/AV GEMM dispatches in the 28-block chain.
Weights, kernel objects, contexts, and BOs remain resident between denoising
steps. The wrapper never silently falls back to CPU.

## Supported contract

- ComfyUI API validated at commit
  `170594057a22673349ddf0a3d88624b7fa5865bb`
- native Base v1.0 or Turbo V1.1 BF16 checkpoint loaded by `UNETLoader`
- matching read-only XDNA source: validated Diffusers/native Base for Base,
  or the same native Turbo checkpoint for Turbo
- batch 1, latent `[1,16,1,64,64]`, 512x512 output
- adapted Qwen context `[1,512,1024]`
- BF16 model weights and block activations, FP32 sampler latent

Preview/Aesthetic checkpoints, LoRA patches, LLLite, INT8, other
batch/resolution/context sizes, and training are rejected. The normal ComfyUI
MODEL is cloned; it is not modified. Before XRT opens, the wrapper computes a
canonical full digest over all 560 block tensors. Base/Turbo or other
MODEL/source mismatches are rejected before BO population or dispatch.

## Installation

Place or clone this repository below `ComfyUI\custom_nodes`, then install it in
the same Python environment that starts ComfyUI:

```powershell
cd ComfyUI\custom_nodes\anima-lora-style-bridge
python -m pip install -e .
```

That environment also needs the Windows XRT SDK/`pyxrt`, Triton-XDNA and its
matching MLIR wheels. Set `XRT_DEV_DIR` to the XRT SDK directory before
starting ComfyUI. The packed cache defaults to
`%USERPROFILE%\.cache\anima-xdna\weights`.

## Nodes

1. Load the native Anima Base or Turbo MODEL with ComfyUI's normal `UNETLoader`.
2. Connect it to **Load/Attach Anima XDNA Model**.
3. Set `checkpoint` to the matching source. For Base, use the validated Base
   Diffusers transformer or equivalent native Base checkpoint. For Turbo,
   select the exact same native Turbo V1.1 checkpoint as `UNETLoader`. The file
   remains read-only and is the XDNA cache source of truth.
4. Connect the returned MODEL to the normal sampler.
5. Use **Anima XDNA Runtime Status** to inspect cache state, first/warm timing,
   dispatches, transfers, BO population, allocations, and resident reuse.
6. Use **Unload Anima XDNA Runtime** before switching checkpoints when an
   immediate release is required. Garbage collection also releases the
   attachment.

`rebuild_cache` explicitly recreates the packed cache. `cache_dir` may point
to another user-writable ASCII path. It must not point into the model folder.

## Verified result

On Ryzen AI 9 365 / Strix XDNA 2 (`npu2`), XRT 2.21.75 and Triton-XDNA
3.6.0.2026093004:

- native ComfyUI MODEL registration and loading succeeded
- one complete synthetic DiT comparison: normalized RMS 0.01528
- real Qwen prompt, Euler, two steps: latent normalized RMS 0.01819
- isolated cold wrapper call: 26.73 s, including 3,963,617,280 bytes of
  process-local weight BO population
- second resident wrapper call: 7.73 s total / 7.57 s chain, 644 dispatches,
  644 resident hits, zero weight population bytes and no recompile
- real Qwen prompt, CFG 3.5, one denoising step: normalized RMS 0.04068
  (the two CFG branches execute sequentially on the single-tenant NPU)
- decoded 512x512 images: PSNR 36.33 dB, SSIM 0.8458

Turbo V1.1 was additionally validated from its native 685-tensor BF16
checkpoint. A real Qwen/Euler/CFG 1 one-step sampler latent passed at
normalized RMS `0.02650`; an 8-step 512x512 run produced PSNR `19.855 dB`,
SSIM `0.82480`, and 8-bit MAE `10.884` against the CPU reference. The local
8-step setting is validated here, but is not claimed as an official V1.1
recommendation. Turbo's synthetic chain has a transient block-13 normalized
RMS peak of `0.08455` and final `0.01004`; use the documented separate
10% intermediate / 2% final gate rather than Base's uniform 5% gate.

The first call additionally verifies the 3.96 GB packed cache and populates
process-local XRT BOs, so cold and warm timings must not be compared as the
same phase.

If an applied LoRA/ModelPatcher patch, unsupported shape, incompatible
ComfyUI API, cache corruption, XRT failure, or numerical/runtime error is
encountered, execution raises an actionable error instead of calling the
original CPU block loop.
