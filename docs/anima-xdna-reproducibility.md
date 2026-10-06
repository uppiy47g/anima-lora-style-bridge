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
Unless explicitly labeled as Q/K/V-chained, the historical measurements below
describe the pre-chaining resident runtime (23 dispatches/block); the current
default is 19 dispatches/block and is recorded in the final optimization
milestone.

## Architecture and model files

The ComfyUI wrapper replaces the block loop after native patch embedding,
timestep/AdaLN setup, Qwen adaptation, and RoPE preparation, and before the
native final layer and unpatchify. The resident runtime keeps XRT device,
contexts, kernels, weights, and reusable BOs alive across denoising steps.
The current default performs 19 dispatches per block and 532 for all 28;
`qkv_chaining=false` selects the earlier 23/644 comparison path.

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

Use `Anima XDNA 2 / Load Anima (BF16)` to load the native Base or Turbo model,
then insert
`Anima XDNA 2 / Load/Attach Anima XDNA Model` between its MODEL output and the
normal sampler MODEL input. Set `checkpoint` to the matching Base source or
the same native Turbo checkpoint. Leave `rebuild_cache=false`; leave `cache_dir` empty for
`%USERPROFILE%\.cache\anima-xdna\weights`, or use another user-owned ASCII
directory. Connect the wrapped MODEL to `Anima XDNA Runtime Status` when
collecting evidence. `Unload Anima XDNA Runtime` releases the attachment
before a model switch.

The stock `UNETLoader` control does not offer BF16. On the validated
CPU-only ComfyUI setup, its default Anima dtype policy selects FP32 (the
Anima config supports BF16, FP16, and FP32; `should_use_bf16(cpu)` is false).
That expands the 3,875,565,568-byte 560-block BF16 source payload to
7,751,131,136 bytes of FP32 block Parameters. `Load Anima (BF16)` calls
ComfyUI's public `load_diffusion_model` API with an explicit
`model_options={"dtype": torch.bfloat16}` and refuses the result unless every
transformer-block Parameter remains BF16. The other 125 non-block parameters
are loaded through the same ComfyUI model path and remain subject to the
checkpoint/model configuration.

The XDNA attach status reports logical CPU block Parameter bytes, unique
underlying block storage bytes, dtype/numel breakdown, and non-block Parameter
bytes after model/checkpoint fingerprint equality is verified. It also warns
if the attached block Parameters are FP32. XRT BO population is reported
separately; it is not added to CPU Parameter bytes. We intentionally retain
all block Parameters: `ModelPatcher.clone()` shares the underlying model
module, and the current wrapper does not own a reversible lazy-restore,
LoRA/patch, state-dict, or reload lifecycle for deleting them safely. The
BF16 loader is therefore the supported memory reduction; attach does not
mutate or release shared model weights.

The explicit BF16 option changes CPU pre/post-block computation dtype compared
with CPU-default FP32. The native checkpoint values are BF16 in both Base and
Turbo; the XDNA block weights, canonical fingerprint, QKV dispatch path, and
output conversion remain unchanged. A full Turbo workflow comparison is now
recorded below. The status data records measured Parameter storage; transient
loader peak commit/working set and system-wide unified-memory pressure are
separate quantities and must not be inferred from the Parameter count alone.

On the validated CPU setup, fresh standalone ComfyUI processes loaded both
native checkpoints through the same public `comfy.sd.load_diffusion_model`
API: with explicit BF16 options and with stock default options. The default
path selected FP32. Both BF16 models matched their canonical fingerprints and
retained 3,875,565,568 unique bytes across 560 transformer-block Parameters;
the 125 non-block Parameters totaled 306,572,288 bytes. The corresponding
FP32 block storage was measured at 7,751,131,136 bytes (twice the BF16
allocation).

Windows `GetProcessMemoryInfo` measured process `PrivateUsage` commit after
load at 8,193,351,680 bytes (Base BF16) and 8,192,634,880 bytes (Turbo BF16),
versus 15,317,835,776 and 15,318,614,016 bytes with the stock FP32 loader.
That is a measured 7,124,484,096-byte Base reduction and 7,125,979,136-byte
Turbo reduction, about 46.5% of the stock-FP32 process commit. Peak private
commit was lower by about 7.13 GB in each paired fresh-process measurement.
Peak working set was 8.80 GB with BF16 versus 11.34 GB (Base) / 11.49 GB
(Turbo) with FP32. These are process-level before/after/peak counters, not a
claim that the BF16 parameter delta is all private resident RAM; page-fault
counts and working set are recorded separately. Full per-run values and
methodology are in the machine-readable evidence.

The actual `Load Anima (BF16)` and `Load/Attach Anima XDNA Model` nodes were
also invoked for Base and Turbo: both reported the expected model fingerprint
and block-byte count, their ModelPatcher clones shared the underlying module,
and unload closed the attachment. The XDNA coexistence path was additionally
exercised for both variants using the BF16-loaded native ComfyUI module:
`SharedRuntime` verified the packed cache
and complete model fingerprint, opened XRT, ran the real 28-block capture
twice, then closed and garbage-collected the runtime/model. Each first chain
populated 3,963,617,280 weight-BO bytes; each resident chain populated zero
additional weight bytes and reported 644 resident hits / 532 dispatches.
Measured first/resident chain times were 20.69/10.18 s for Base and
22.35/10.54 s for Turbo. This confirms cache identity and XRT residency while
the BF16 model is loaded, but it is not a sampler or image-quality run. During
these XRT runs, peak process private commit reached about 16.57 GB; after
`close()` and GC it remained 9.22 GB (Base) / 8.25 GB (Turbo), although the
working sets fell to about 0.74 GB. Python/runtime allocator behavior did not
return all commit before process exit. XRT BO byte counts and CPU Parameter
bytes remain separately reported because unified-memory allocations must not
be double-counted.

The exact Turbo workflow was then run through the actual BF16 loader and XDNA
attach nodes, followed by a fresh stock-FP32 CPU reference with matching
inputs: Qwen 3 0.6B, Qwen Image VAE, 512x512, seed 424242, Euler/simple,
CFG 1, 8 steps, and the prompt/negative prompt above. The BF16+XDNA run took
140.316 s API wall / 135.936 s Comfy execution; the FP32 CPU reference took
115.337 s API wall / 113.005 s execution. These are single measurements from
separate fresh server launches, not a controlled performance pair, and no
speedup is claimed.

The BF16+XDNA 512x512 output is **pixel-identical** to the previous validated
XDNA image. The fresh FP32 CPU run is also pixel-identical to the previously
validated CPU reference image, confirming the reconstructed workflow inputs.
Thus BF16 retained the existing XDNA decoded-image result: against the CPU
reference the image has PSNR 19.8551 dB, SSIM 0.824799, and 8-bit MAE 10.8841,
the same previously reported quality tradeoff. The BF16+XDNA latent compared
to the stock-FP32 CPU latent has normalized RMS error 0.255463, maximum
absolute error 3.86543, and mean absolute error 0.169834. The previous XDNA
latent was not retained, so direct BF16-vs-previous-XDNA latent parity is
unverified; the established gate here is decoded-image parity to that
validated XDNA output, not FP32 CPU latent equality.

This establishes a material CPU-model memory reduction without deleting or
replacing shared ModelPatcher Parameters. It does not demonstrate a runtime
speedup, and the latent-vs-CPU difference remains a limitation despite exact
decoded-image reproduction of the validated XDNA result.

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

At the end of this profiling-only phase, the first candidate fusion, applying attention scale in the final head
matmul, was tested on the validated toolchain and rejected. Adding a scalar
epilogue caused AIR compilation to fail with `iterator_interchange` length 4
for an operation with 2 loops, both with and without an explicit BF16 cast.
The experimental path was removed. Attention scaling, normalization, RoPE,
softmax, GELU, and residual operations remain on CPU; dispatch count and
numerical behavior were unchanged in that phase. The later Q/K/V chaining
milestone below reduces projection dispatches but does not migrate those
host-side operations.

| Stage or gate | Status in this phase | Evidence / limitation |
|---|---|---|
| AdaLN/LayerNorm, RMSNorm, RoPE, mask/softmax, GELU, gates/residuals | CPU unchanged | No candidate migration passed a compiler and oracle gate in this phase. |
| Attention-scale epilogue | Not supported on current toolchain | AIR `iterator_interchange` lowering failure; trial removed, no fallback path added. |
| Block-to-block NPU activation residency | Not implemented | Host normalization/attention stages still consume visible tensors between GEMMs. |
| Dispatch reduction at that phase | No change | Historical 23 per block / 644 per 28-block chain; superseded by the next milestone. |
| Turbo 8-step generation at that phase | Not rerun | The standalone ComfyUI service was unavailable during the profiling-only phase. |

## Optimization milestone: resident QKV projection chaining

The next bounded optimization groups the three self-attention Q/K/V BF16
Linear launches into one Triton-XDNA `NPUChain` run, and does the same for
cross-attention. Self-attention stages its common input once; cross-attention
stages query and context once each. Each signature owns three float32 output
buffers created with Triton-XDNA's public `shared.empty` API and bound through
`NPUChain.run(bound_buffers=...)`. They are session-scoped, keyed by exact
shape/layout, reused across serial block calls, and closed with the session.
The code does not reach into NPUChain's private BO cache. Returned projections
are still copied to BF16 CPU tensors because subsequent normalization,
RoPE, softmax, and residual work remains host-side. Thus this is projection
dispatch/input-staging fusion and output-buffer reuse, not block-to-block
activation residency or NPU-only execution. The `qkv_chaining` ComfyUI input
defaults to true; false selects the pre-milestone dispatch path for A/B tests.

The fixture measurements below are captured from actual standalone ComfyUI
Base/Turbo prompt preprocessing (512x512, 1024 image tokens, 512 Qwen context
tokens); the local `.pt` captures are not distributed. Each “28-block chain”
is one denoising step. The H2D values are summed from per-stage XRT profile
counters, and the previously published 156,631,040-byte block profile is
historical and not used for this same-shape comparison.

| Metric | Q/K/V separate | Q/K/V chained |
|---|---:|---:|
| XRT dispatches/block | 23 | 19 |
| XRT dispatches/28-block chain | 644 | 532 |
| H2D bytes/block | 298,188,800 | 288,751,616 |
| H2D bytes/28-block chain | 8,349,286,400 | 8,085,045,248 |
| D2H-visible bytes/block | 314,048,512 | 314,048,512 |
| D2H-visible bytes/28-block chain | 8,793,358,336 | 8,793,358,336 |
| Reported BO allocation events/28-block chain | 542 | 630 |

This removes four XRT dispatches and 9,437,184 H2D bytes per block (264,241,152
bytes per denoising step, about 3.16% of measured H2D traffic); output sync
bytes do not change. The allocation counter includes per-key static weight BO
population. The three Q/K/V output BOs per signature are reused after their
initial allocation; the higher count reflects the new projection-chain BO
sets, not one fresh set of output buffers per block.

One captured-fixture validation pair measured Base at 18.397 s -> 17.313 s
and Turbo at 17.418 s -> 17.843 s. These chain-wall samples are noisy and do
not support a stable chain-speed claim. With each model's same 28-block
inputs, every old/new XDNA block output was bitwise equal. Turbo passed its published CPU-oracle
gate at 0.0599236 worst transient NRMS (block 12) and 0.0117299 final.
Base's captured real-preprocessing fixture showed 0.0782801 worst transient
NRMS (block 14) and 0.00840710 final against the CPU oracle. This exceeds
Base's strict 5% intermediate gate; the separate and chained XDNA outputs
were bitwise equal at all 28 blocks. The Base result is recorded as a
strict-gate failure, not passed by widening Base's limit. The one-block
Base/Turbo CPU-oracle tests continue to pass.

The controlled Turbo ComfyUI comparison used the same current worktree,
checkpoint, Qwen/VAE, prompt, negative prompt, seed 424242, Euler/simple,
CFG 1, eight steps, and 512x512 output. After warm-up in the same server, the
QKV-disabled workflow completed in 147.56 s and the enabled workflow in
139.18 s (8.38 s / 5.68% faster). Both generated 512x512 RGB images were
pixel-identical (8-bit MAE 0). The first post-startup run was excluded from
the paired timing because cache warm-up shifted subsequent measurements.
This is an end-to-end improvement, but not an NPU-only claim: Qwen, all
normalization/RoPE/softmax/GELU/gate/residual operations, sampler, final
projection, and VAE remain on CPU.

| Stage or gate | Current status | Evidence / limitation |
|---|---|---|
| Q/K/V dispatch fusion | Implemented | 23 -> 19 XRT dispatches/block; the 3 Linear kernels remain in one chain. |
| Q/K/V input and output buffers | Implemented, session-scoped | 9,437,184 fewer H2D bytes/block; output sync bytes unchanged; cleanup on session exit/error. |
| Block-to-block activation residency | Not implemented | Host normalization and attention stages still materialize block inputs/outputs. |
| Base real-fixture CPU chain gate | Fails strict 5% transient threshold | 7.828% at block 14; final 0.8407%; old/new XDNA paths bitwise equal. |
| Turbo real-fixture CPU chain gate | Passes 10% / 2% | 5.992% at block 12; final 1.173%. |
| Turbo 8-step workflow | Measured improvement | 147.56 s -> 139.18 s; image pixel-identical. |
| Norm/RoPE/softmax/GELU/residual migration | Not implemented | Attention-scale epilogue still has the documented AIR lowering blocker. |

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
   Pass: default 19-dispatch block satisfies its single-block numerical gate;
   `qkv_chaining=false` selects the 23-dispatch control path.
5. **28-block chain:**
   `anima-xdna-chain $env:ANIMA_XDNA_CHECKPOINT --fixture $env:ANIMA_XDNA_CHAIN_FIXTURE --start-block 0 --end-block 28 --warmups 1 --runs 3 --profile`
   The current chain uses 532 dispatches. Turbo passes with
   `--max-block-nrms 0.10 --max-final-nrms 0.02` using the captured real
   preprocessing fixture. Base retains its strict 5% gate; the real Base
   fixture currently exposes the documented 7.828% transient CPU-oracle
   mismatch, while the old/new XDNA output parity remains bitwise exact.
   Turbo's default synthetic seed-0 input is not covered and was correctly
   rejected at 9.61% final normalized RMS.
6. **ComfyUI registration:** start isolated ComfyUI, then:
   `Invoke-RestMethod http://127.0.0.1:8190/object_info/LoadAttachAnimaXDNAModel`
   Pass: object info returns category `Anima XDNA 2`; repeat for
   `AnimaXDNARuntimeStatus` and `AnimaXDNAUnload`.
7. **Final image:** run the standard 512×512 Base or Turbo workflow with the wrapper
   insertion. Pass: status shows 532 NPU dispatches per non-CFG chain, a second
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
