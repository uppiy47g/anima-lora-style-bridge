# Anima XDNA 2 ComfyUI MODEL wrapper

This custom-node package replaces only the 28 `transformer_blocks` in ComfyUI's
native Anima Base v1.0 diffusion model with the resident Triton-XDNA/XRT
runtime. It does **not** use ONNX or Vitis AI EP.

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
- official native `anima-base-v1.0.safetensors` loaded by `UNETLoader`
- Diffusers Base v1.0 transformer safetensors as the read-only XDNA weight
  source
- batch 1, latent `[1,16,1,64,64]`, 512x512 output
- adapted Qwen context `[1,512,1024]`
- BF16 model weights and block activations, FP32 sampler latent

Preview/Aesthetic/Turbo full checkpoints, LoRA patches, LLLite, INT8, other
batch/resolution/context sizes, and training are rejected. The normal ComfyUI
MODEL is cloned; it is not modified.

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

1. Load the official native Anima Base MODEL with ComfyUI's normal
   `UNETLoader`.
2. Connect it to **Load/Attach Anima XDNA Model**.
3. Set `checkpoint` to the Base v1.0 **Diffusers transformer**
   `diffusion_pytorch_model.safetensors`. This file remains read-only and is
   the XDNA cache source of truth.
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

The first call additionally verifies the 3.96 GB packed cache and populates
process-local XRT BOs, so cold and warm timings must not be compared as the
same phase.

If an applied LoRA/ModelPatcher patch, unsupported shape, incompatible
ComfyUI API, cache corruption, XRT failure, or numerical/runtime error is
encountered, execution raises an actionable error instead of calling the
original CPU block loop.
