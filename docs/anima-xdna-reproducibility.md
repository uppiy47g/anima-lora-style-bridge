# Anima Base v1.0 on AMD XDNA 2: reproducibility report

## Scope and claim

This repository demonstrates Anima Base v1.0 image generation with its 28 DiT
transformer blocks accelerated on AMD XDNA 2 through Triton-XDNA and XRT.
Safetensors remain the read-only source of truth. No ONNX conversion, Vitis AI
Execution Provider, or converted distributable model is used.

The pipeline is CPU-hybrid. XDNA runs the expensive Linear and QK/AV GEMMs;
Qwen, patch/time/RoPE preparation, normalization, softmax, GELU, residuals,
final projection/unpatchify, sampling, and VAE decoding remain on the CPU.
“GPU-free primary DiT acceleration” therefore does not mean “NPU-only image
generation.” A display GPU may be active independently.

As of 2026-10-02, searches of public GitHub code and the general web for
combinations of Anima, XDNA/XDNA 2, and Triton-XDNA did not find another
Anima-specific end-to-end demo. The searches did find generic Triton-XDNA
kernel examples in [AMD's Triton-XDNA repository](https://github.com/amd/Triton-XDNA)
and non-Anima [Ryzen AI Stable Diffusion demos](https://ryzenai.docs.amd.com/en/latest/sd_demo.html).
This dated, non-exhaustive search is not an absolute novelty claim.

## Validated matrix

| Component | Validated value |
|---|---|
| Processor/NPU | AMD Ryzen AI 9 365 / Strix XDNA 2 |
| NPU target | `npu2` / AIE2P |
| Windows Python | 3.13.9 |
| PyTorch | 2.14.1+cpu |
| Triton-XDNA | 3.6.0.2026093004+75679dc |
| XRT SDK | 2.21.75 |
| ComfyUI | `170594057a22673349ddf0a3d88624b7fa5865bb` |
| Wrapper source | `737338d76bf2a7dcf9806aaba5d6d936499b95ab` |
| Model | Anima Base v1.0, BF16 |
| Shape | batch 1, latent `[1,16,1,64,64]`, 512×512, context `[1,512,1024]` |

The exact structured measurements and artifact digests are in
[`evidence/anima-xdna-validated.json`](evidence/anima-xdna-validated.json).

## Architecture and model files

The ComfyUI wrapper replaces the block loop after native patch embedding,
timestep/AdaLN setup, Qwen adaptation, and RoPE preparation, and before the
native final layer and unpatchify. The resident runtime keeps XRT device,
contexts, kernels, weights, and reusable BOs alive across denoising steps.
It performs 23 dispatches per block and 644 for all 28 blocks.

Two different files are required:

1. Native ComfyUI `anima-base-v1.0.safetensors`, selected in `UNETLoader`.
2. Diffusers Base transformer `transformer/diffusion_pytorch_model.safetensors`,
   entered in `Load/Attach Anima XDNA Model`.

The second file supplies exact block weights to the packed cache. It does not
replace the native ComfyUI checkpoint. Model weights are not included here.
Review the model licenses and obtain both files from their authorized source.

## Isolated Windows setup

Use an ASCII-only user path for ComfyUI and compiler caches. Do not modify an
existing ComfyUI Desktop installation.

```powershell
py -3.13 -m venv <XDNA_VENV>
& <XDNA_VENV>\Scripts\python.exe -m pip install --upgrade pip
# Install official AMD/Xilinx Triton-XDNA, MLIR-AIR/AIE, LLVM-AIE and
# ABI-compatible pyxrt wheels as described in anima_xdna_poc/README.md.
$env:XRT_DEV_DIR = "<XRT_SDK_DIR>"
& <XDNA_VENV>\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
& <XDNA_VENV>\Scripts\python.exe -m pip install -e "<THIS_REPOSITORY>[xdna-poc]"

git clone https://github.com/comfyanonymous/ComfyUI.git <COMFYUI_ROOT>
git -C <COMFYUI_ROOT> checkout 170594057a22673349ddf0a3d88624b7fa5865bb
# Install only missing ComfyUI requirements; do not replace the validated
# torch/Triton-XDNA stack.
```

Place a clone or directory junction of this repository under
`<COMFYUI_ROOT>\custom_nodes\anima-lora-style-bridge`. Copy
[`../examples/extra_model_paths_anima_xdna.yaml`](../examples/extra_model_paths_anima_xdna.yaml)
to a user-owned location and replace its placeholders. Never edit a
Desktop-generated model-path file for this setup.

```powershell
.\examples\run_comfyui_xdna.ps1 `
  -ComfyUIRoot "<COMFYUI_ROOT>" `
  -XDNAVenv "<XDNA_VENV>" `
  -XRTDevDir "<XRT_SDK_DIR>" `
  -ModelPathsConfig "<USER_MODEL_PATHS_YAML>" `
  -Port 8190
```

The launcher binds only `127.0.0.1`, uses ComfyUI CPU host mode, and forwards
additional arguments. The XDNA runtime remains available through
`XRT_DEV_DIR`, Triton-XDNA, and `pyxrt`.

## ComfyUI wiring

Load the native Base MODEL with `UNETLoader`, then insert
`Anima XDNA 2 / Load/Attach Anima XDNA Model` between its MODEL output and the
normal sampler MODEL input. Set `checkpoint` to the Diffusers transformer
safetensors. Leave `rebuild_cache=false`; leave `cache_dir` empty for
`%USERPROFILE%\.cache\anima-xdna\weights`, or use another user-owned ASCII
directory. Connect the wrapped MODEL to `Anima XDNA Runtime Status` when
collecting evidence. `Unload Anima XDNA Runtime` releases the attachment
before a model switch.

The API JSON in [`../examples/anima_xdna_512_api.json`](../examples/anima_xdna_512_api.json)
is a minimal insertion/status fragment, not a redistributable model workflow.
Replace `<DIFFUSERS_TRANSFORMER_SAFETENSORS>` locally and merge the fragment
into the standard native Anima Base workflow. Prompt conditioning must use
Qwen 3 0.6B and decoding must use Qwen Image VAE.

## Packed cache and cold/warm behavior

The packed cache is a regenerable local derivative, not a model. It contains
verified transposed/padded BF16 payloads and layout metadata, never the source
safetensors container or a persistent device pointer. Its key includes source
tensor content, architecture/key map, layout ABI, future adapter/quantization
namespace, tool versions, and NPU target. Manifest and payload digests are
verified; build publication is locked and atomic.

The validated 28-block entry contains 644 tensors, occupies 3,963,645,952
bytes, and has 88,080,384 bytes of padding. Cold build was 19.834–19.865 s;
a new-process verified warm hit was 4.330–4.546 s. BO upload is still required
once per process. A second wrapper call reused all 644 residents with zero
weight population and took 7.73 s total versus 26.73 s for the isolated cold
call.

## Correctness and performance evidence

| Gate | Result |
|---|---:|
| Exact block 0 max / mean absolute error | 0.0009765625 / 0.00002131443 |
| Dispatches per block, legacy → resident | 83 → 23 |
| 28-block steady time | 7.02, 7.03, 7.15 s |
| 28-block CPU oracle | 23.19 s |
| 28-block final normalized RMS | 0.007466678 |
| Real prompt, one step, CFG 3.5 normalized RMS | 0.04067577 |
| Real prompt, two steps, CFG 1.0 normalized RMS | 0.01818691 |
| Decoded image PSNR / SSIM | 36.3279 dB / 0.845809 |

Chain steady timings used one warmup followed by three measured repeats and
exclude the one-time stitched-kernel compile. The CPU oracle used the same
captured pre-block tensors. Cache cold-build and warm-hit figures came from
separate processes and report fingerprint/verification separately from BO
upload. Image comparison used identical prompt, negative prompt, seed,
sampler, step count, CFG, and initial latent. These controls are why the
20-step observation below is not included in the benchmark table.

The two-step evidence used seed 424242, Euler, 512×512, and the prompt/settings
recorded in the evidence JSON. The metadata-free outputs are:

| Reference | XDNA |
|---|---|
| ![Reference](assets/comfyui-reference-2step.png) | ![XDNA](assets/comfyui-xdna-2step.png) |

A later user-observed 20-step run reported 533.94 s total and 25.59 s/it.
It lacked enough controlled metadata and instrumentation to be a benchmark.
Likewise, a low Task Manager NPU percentage is not an AIE utilization
measurement: short dispatches, synchronization, and substantial host work can
leave gaps in that UI graph.

## Public verification ladder

Set placeholders only in the current shell; do not commit paths or captures.

```powershell
$env:XRT_DEV_DIR = "<XRT_SDK_DIR>"
$env:ANIMA_XDNA_CHECKPOINT = "<DIFFUSERS_TRANSFORMER_SAFETENSORS>"
$env:ANIMA_XDNA_CHAIN_FIXTURE = "<OPTIONAL_CAPTURED_BLOCK_INPUT_PT>"
```

1. **Dependency/device probe:** `anima-xdna-probe`
   Pass: reports `ready`, target `npu2`, runtime `xrt`.
2. **Unit tests without hardware inputs:**
   `python -m unittest discover -s tests -v`
   Pass: all unit tests pass; explicitly gated hardware tests may skip.
3. **Single Linear hardware:** `anima-xdna-linear $env:ANIMA_XDNA_CHECKPOINT`
   Pass: real XRT execution and numerical comparison pass; no CPU fallback.
4. **Block hardware:**
   `anima-xdna-block $env:ANIMA_XDNA_CHECKPOINT --image-tokens 1024 --context-tokens 512 --runs 2`
   Pass: 23-dispatch block satisfies its numerical gate.
5. **28-block chain:**
   `anima-xdna-chain $env:ANIMA_XDNA_CHECKPOINT --fixture $env:ANIMA_XDNA_CHAIN_FIXTURE --start-block 0 --end-block 28 --warmups 1 --runs 3 --profile`
   Pass: 644 dispatches and every block remains below the fixed 5% normalized-RMS gate.
6. **ComfyUI registration:** start isolated ComfyUI, then:
   `Invoke-RestMethod http://127.0.0.1:8190/object_info/LoadAttachAnimaXDNAModel`
   Pass: object info returns category `Anima XDNA 2`; repeat for
   `AnimaXDNARuntimeStatus` and `AnimaXDNAUnload`.
7. **Final image:** run the standard 512×512 Base workflow with the wrapper
   insertion. Pass: status shows 644 NPU dispatches per non-CFG chain, a second
   call shows 644 resident hits and zero weight population, and an image is
   decoded without fallback.

## Limitations and troubleshooting

Only Base v1.0 and the fixed shape above are validated. Preview/Aesthetic/Turbo
full checkpoints, LoRA/LLLite/ModelPatcher transformer patches, INT8, training,
other resolutions, other context sizes, and user batches above one are
rejected. CFG batch 2 is executed sequentially on the single-tenant NPU.

- **`dependency unavailable`:** use the same Python 3.13 environment for
  ComfyUI, `pyxrt`, Triton-XDNA, and this package.
- **`NPU unavailable`:** verify Device Manager/driver, `XRT_DEV_DIR`, and
  `anima-xdna-probe`. Do not substitute a CPU success.
- **Compiler path/CP932 failure:** move checkout/cache to an ASCII user path.
- **Checkpoint mismatch:** distinguish the native ComfyUI MODEL from the
  Diffusers transformer source.
- **Unsupported patches:** remove LoRA/LLLite; they are rejected rather than
  ignored.
- **Cache corruption:** run `anima-xdna-cache verify <checkpoint>`, then an
  explicit `rebuild` if verification reports the exact failure.
- **Low Task Manager utilization:** use CLI dispatch/kernel timings and wrapper
  diagnostics; Task Manager is not a direct AIE counter.

The next integration work is safe ModelPatcher-aware LoRA materialization and
optional INT8 layouts, followed by moving host norms/softmax/elementwise stages
onto AIE. Neither is implemented or implied by this report.
