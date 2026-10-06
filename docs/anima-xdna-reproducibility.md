# Anima Base v1.0 and Turbo V1.1 on AMD XDNA 2

## Scope and claim

This repository demonstrates Anima Base v1.0 and Turbo V1.1 image generation with their 28 DiT
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
| Models | Anima Base v1.0 and Turbo V1.1, BF16 |
| Shape | batch 1, latent `[1,16,1,64,64]`, 512×512, context `[1,512,1024]` |

The exact structured measurements and artifact digests are in
[`evidence/anima-xdna-validated.json`](evidence/anima-xdna-validated.json).

## Architecture and model files

The ComfyUI wrapper replaces the block loop after native patch embedding,
timestep/AdaLN setup, Qwen adaptation, and RoPE preparation, and before the
native final layer and unpatchify. The resident runtime keeps XRT device,
contexts, kernels, weights, and reusable BOs alive across denoising steps.
It performs 23 dispatches per block and 644 for all 28 blocks.

Base may use two different files:

1. Native ComfyUI `anima-base-v1.0.safetensors`, selected in `UNETLoader`.
2. Diffusers Base transformer `transformer/diffusion_pytorch_model.safetensors`,
   entered in `Load/Attach Anima XDNA Model`.

The second file supplies exact block weights to the packed cache. It does not
replace the native ComfyUI checkpoint. Model weights are not included here.
Review the model licenses and obtain both files from their authorized source.

Turbo uses its full native V1.1 checkpoint for both `UNETLoader` and the XDNA
`checkpoint` input. Supported schemas are Diffusers `transformer_blocks.*`,
native Base `net.*`, and native Turbo `model.diffusion_model.*`. The wrapper
computes a canonical full digest of ordered key, dtype, shape, and every byte
of all 560 consumed tensors. MODEL/source disagreement is rejected before XRT
opens or any weight BO is populated; Base and Turbo packed payloads cannot be
mixed. The validated canonical fingerprints are
`083d6f88949dec38ec35579dfd16ae96f4d453074b4fdfb6c0ed965c9cd76757`
for Base and
`066b4281037504b1b7200ecd65b4182fc765ed882ecd5ca650db7308246a8dee`
for Turbo.

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

Load the native Base or Turbo MODEL with `UNETLoader`, then insert
`Anima XDNA 2 / Load/Attach Anima XDNA Model` between its MODEL output and the
normal sampler MODEL input. Set `checkpoint` to the matching Base source or
the same native Turbo checkpoint. Leave `rebuild_cache=false`; leave `cache_dir` empty for
`%USERPROFILE%\.cache\anima-xdna\weights`, or use another user-owned ASCII
directory. Connect the wrapped MODEL to `Anima XDNA Runtime Status` when
collecting evidence. `Unload Anima XDNA Runtime` releases the attachment
before a model switch.

The API JSON in [`../examples/anima_xdna_512_api.json`](../examples/anima_xdna_512_api.json)
is a minimal insertion/status fragment, not a redistributable model workflow.
Replace `<MATCHING_BASE_OR_TURBO_SAFETENSORS>` locally and merge the fragment
into the standard native Anima workflow. Select `base` or `turbo` placeholders
without mixing their sources. Prompt conditioning must use
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

Turbo uses the same 644-record layout but a distinct payload/cache identity:
cache key `932742447a74a7168f87fb63b59f8bd0d97bc998e627dcd01731513dabb1a94c`.
Its measured cold fingerprint/pack/total were 6.833/13.831/21.271 s; a new
process warm source check/payload verification/total were 0.030/3.744/3.853 s.
Of 448 block Linear tensors, all differ from Base; the 112 RMSNorm vectors
match. Therefore kernels/layouts are reusable but payloads are not.

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

Turbo V1.1 adds these controlled gates:

| Gate | Result |
|---|---:|
| Block 0 max / mean absolute error | 0.0009765625 / 0.0000213457 |
| Block 0 XRT dispatches | 23 |
| 28-block CPU / XDNA validation chain | 26.406 / 17.228 s |
| First / resident 28-block chain | 18.263 / 7.212 s |
| Resident dispatches / hits / weight population | 644 / 644 / 0 B |
| Worst transient / final normalized RMS | 0.0845512 (block 13) / 0.01003999 |
| Real Qwen one-step latent normalized RMS | 0.02649816 |
| 8-step image PSNR / SSIM / 8-bit MAE | 19.8551 dB / 0.824799 / 10.8841 |
| First / second 8-step workflow wall | 139.75 / 136.47 s |

Turbo's sparse high-magnitude activations amplify BF16 differences around
blocks 12-19, then recover. This is not hidden by widening Base's gate. Turbo
uses a separately declared 10% worst-intermediate ceiling and 2% final ceiling,
plus the strict single-block, real one-step, and decoded-image gates. The
second workflow reused the resident attachment and cached upstream nodes; no
kernel compilation occurred. Per-step logs fell from 28.77 s on the first
step to about 11.8-12.0 s steady. Workflow wall includes Qwen/model loading,
host stages, sampler, and VAE and is not an NPU-only benchmark.

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

The Turbo evidence used the same documented prompt/seed, Euler, CFG 1,
`simple`, and eight steps. Local V1.1 metadata contained no sampler
recommendation; this is a locally validated setting initially informed by
local V1.0 guidance, not an official V1.1 recommendation.

| Turbo CPU reference | Turbo XDNA |
|---|---|
| ![Turbo reference](assets/anima-turbo-v11-reference-8step.png) | ![Turbo XDNA](assets/anima-turbo-v11-xdna-8step.png) |

A later user-observed 20-step run reported 533.94 s total and 25.59 s/it.
It lacked enough controlled metadata and instrumentation to be a benchmark.
Likewise, a low Task Manager NPU percentage is not an AIE utilization
measurement: short dispatches, synchronization, and substantial host work can
leave gaps in that UI graph.

## Optimization follow-up: measured boundary accounting

The resident profiler reports stage wall time, XRT dispatches, H2D bytes,
device-to-host-visible output bytes, BO allocations/hits, host copies,
synchronization, and kernel/wait time. Earlier evidence counted no separate
D2H staging-buffer copy and consequently reported zero D2H bytes. That was
ambiguous: every GEMM output is synchronized from its BO before CPU-side
stages consume it. The new `d2h_host_visible_bytes` counter counts those
output bytes; it does not imply a second staging allocation. The older
`d2h_staging_bytes` name remains as a compatibility alias. Ordinary Linear
D2H sync time is separate. The Triton-XDNA `NPUChain` interface does not
expose separate output-sync timing, so its output bytes are counted while its
sync remains included in `kernel_and_wait_ms`.

Warm block-0 profiles on 2026-10-06 used deterministic synthetic inputs and
the exact Base and Turbo checkpoints. Base measured 334.47 ms wall, 60.71 ms
host stages, 269.62 ms NPU stages, 156,631,040 H2D bytes, and 314,048,512
device-to-host-visible bytes. Turbo measured 342.37 ms, 68.71 ms host,
268.66 ms NPU, and the same transfers. Both reported 23 dispatches, zero
allocations, 23 resident hits, and zero weight population on the measured run.
Their one-block normalized RMS errors against the CPU oracle were 0.0025100
and 0.0025191, respectively.

For a 28-block batch-1 chain, the measured per-block traffic projects to
644 dispatches, 4,385,669,120 H2D bytes, and 8,793,358,336
device-to-host-visible bytes. The eight-step figures are projections from
that block profile (5,152 dispatches, 35,085,352,960 H2D bytes, and
70,346,866,688 output bytes), not a new ComfyUI workflow measurement. The
ComfyUI server was not running during this phase, so the validated 136.47 s
Turbo workflow baseline was not rerun and no end-to-end speedup is claimed.

The first candidate fusion, applying attention scale in the final head
matmul, was tested on the validated toolchain and rejected. Adding a scalar
epilogue caused AIR compilation to fail with `iterator_interchange` length 4
for an operation with 2 loops, both with and without an explicit BF16 cast.
The experimental path was removed. Attention scaling, normalization, RoPE,
softmax, GELU, and residual operations remain on CPU; dispatch count and
numerical behavior are unchanged. The next useful kernel experiment requires
an AIR-supported attention epilogue or a correctness-validated resident
attention pipeline, rather than forcing this unsupported lowering.

| Stage or gate | Status in this phase | Evidence / limitation |
|---|---|---|
| AdaLN/LayerNorm, RMSNorm, RoPE, mask/softmax, GELU, gates/residuals | CPU unchanged | No candidate migration passed a compiler and oracle gate in this phase. |
| Attention-scale epilogue | Not supported on current toolchain | AIR `iterator_interchange` lowering failure; trial removed, no fallback path added. |
| Block-to-block NPU activation residency | Not implemented | Host normalization/attention stages still consume visible tensors between GEMMs. |
| Dispatch reduction | No change | 23 per block / 644 per 28-block chain. |
| Turbo 8-step generation | Not rerun | Standalone ComfyUI service was unavailable; no new end-to-end timing or image comparison. |

## Public verification ladder

Set placeholders only in the current shell; do not commit paths or captures.

```powershell
$env:XRT_DEV_DIR = "<XRT_SDK_DIR>"
$env:ANIMA_XDNA_CHECKPOINT = "<DIFFUSERS_TRANSFORMER_SAFETENSORS>"
$env:ANIMA_XDNA_CHAIN_FIXTURE = "<CAPTURED_REAL_BLOCK_INPUT_PT>"
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
   Base pass: 644 dispatches and every block remains below 5% normalized RMS.
   Turbo pass: append `--max-block-nrms 0.10 --max-final-nrms 0.02`; both
   reported gates must pass using a captured real preprocessing fixture.
   Turbo's default synthetic seed-0 input is not covered and was correctly
   rejected at 9.61% final normalized RMS.
6. **ComfyUI registration:** start isolated ComfyUI, then:
   `Invoke-RestMethod http://127.0.0.1:8190/object_info/LoadAttachAnimaXDNAModel`
   Pass: object info returns category `Anima XDNA 2`; repeat for
   `AnimaXDNARuntimeStatus` and `AnimaXDNAUnload`.
7. **Final image:** run the standard 512×512 Base or Turbo workflow with the wrapper
   insertion. Pass: status shows 644 NPU dispatches per non-CFG chain, a second
   call shows 644 resident hits and zero weight population, and an image is
   decoded without fallback.

## Limitations and troubleshooting

Only Base v1.0, Turbo V1.1, and the fixed shape above are validated. Preview/Aesthetic
checkpoints, LoRA/LLLite/ModelPatcher transformer patches, INT8, training,
other resolutions, other context sizes, and user batches above one are
rejected. CFG batch 2 is executed sequentially on the single-tenant NPU.

- **`dependency unavailable`:** use the same Python 3.13 environment for
  ComfyUI, `pyxrt`, Triton-XDNA, and this package.
- **`NPU unavailable`:** verify Device Manager/driver, `XRT_DEV_DIR`, and
  `anima-xdna-probe`. Do not substitute a CPU success.
- **Compiler path/CP932 failure:** move checkout/cache to an ASCII user path.
- **Checkpoint mismatch:** choose the exact matching Base source, or use the
  same native Turbo file for MODEL and XDNA source. The full digest check is
  intentionally expensive once per attachment and cannot be disabled.
- **Unsupported patches:** remove LoRA/LLLite; they are rejected rather than
  ignored.
- **Cache corruption:** run `anima-xdna-cache verify <checkpoint>`, then an
  explicit `rebuild` if verification reports the exact failure.
- **Low Task Manager utilization:** use CLI dispatch/kernel timings and wrapper
  diagnostics; Task Manager is not a direct AIE counter.

The next integration work is safe ModelPatcher-aware LoRA materialization and
optional INT8 layouts, followed by moving host norms/softmax/elementwise stages
onto AIE. Neither is implemented or implied by this report.
