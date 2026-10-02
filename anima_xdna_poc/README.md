# Anima resident transformer chain on AMD XDNA 2

This isolated proof of concept executes either one Anima Linear or the complete
Anima/Cosmos transformer chain (one block, a range, or all 28 blocks) directly
from its original safetensors on XDNA 2 through Triton-XDNA and XRT. It does
**not** use ONNX, Vitis AI EP, or a converted model format, and it does not
modify the existing LoRA bridge, ComfyUI, or model files.

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

The loader validates all 20 block tensors and materializes only requested
tensors through `safetensors.safe_open`. The original checkpoint remains the
read-only source of truth. Native Triton cache entries contain compiled
kernels; the separate packed-weight cache described below is a verified,
regenerable local derivative and is never a replacement model.

Positive compatibility covers full BF16 **Anima Base v1.0** checkpoints in
Diffusers `transformer_blocks.*` or native `net.*` form and the validated full
native **Turbo V1.1** checkpoint in `model.diffusion_model.*` form. The loader
canonicalizes all 560 block tensors, while the cache manifest retains every
exact source key. LoRA adapters and quantized checkpoints remain unsupported.

## NPU, attention fusion, and residency

Every block Linear and every QK-transpose/attention-value GEMM runs on XDNA 2.
The optimized path uses **23 XRT dispatches per block**:

- 19 Linear dispatches (the 16 logical model Linear operations, with the
  8192-wide MLP output reduction split into four FP32 partials);
- four attention dispatches: self QK, self AV, cross QK, and cross AV.

Each attention dispatch uses Triton-XDNA's official `NPUChain` multi-launch API
to stitch the 16 correct head-specific Triton kernels into one ELF/XRT run.
This removes the head-count-proportional host dispatches without changing
head semantics. A single 3D Triton grid was tested and rejected because the
installed AIE2P lowering corrupted heads at realistic multi-tile shapes;
folding heads into the program M dimension was also rejected by AIR tile-size
inference. Neither unsafe path is used by the package.

The resident session owns the XRT device, ELF objects, hardware contexts,
kernels, and mapped `ext.bo` buffers. Static weight BOs are populated once and
reused across blocks/runs; dynamic input and output BOs are reused by shape.
Outputs remain mapped rather than being copied to a separate host tensor, so
the steady path has zero explicit D2H staging copies. Cache synchronization is
measured separately. After population, checkpoint tensors are replaced by
shape/dtype metadata (except the small Q/K RMSNorm vectors), avoiding a second
approximately 4 GB host copy while the resident BOs are live. The safetensors
checkpoint always remains the source of truth.

SiLU, LayerNorm/AdaLN modulation, Q/K RMSNorm, RoPE, scale/mask/softmax, exact
GELU, tensor reshapes, gates, and residual additions remain explicit PyTorch
CPU operations over shared/mapped memory. These operations and their timing
are reported; there is no silent CPU fallback. Parameterized GEMMs account for
137,451,536,384 logical FLOPs per block and are all on the NPU.

The supported 512x512 path is batch 1, 1024 image tokens, hidden size 2048,
512 context tokens, and context width 1024. The deterministic default input is
synthetic. Hardware chain validation used exact Anima model patch embedding,
timestep embedding, `temb`, and Cosmos RoPE captured read-only before block 0,
but with a random latent and synthetic text context. It is therefore
model-produced preprocessing, not a capture from a real prompt-generation
run.

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
$env:XRT_DEV_DIR = "<XRT_SDK_DIR>"
$checkpoint = "<MATCHING_BASE_OR_TURBO_SAFETENSORS>"
$fixture = "<OPTIONAL_CAPTURED_BLOCK_INPUT_PT>"
anima-xdna-probe
anima-xdna-linear $checkpoint
anima-xdna-block $checkpoint `
  --image-tokens 1024 --context-tokens 512 --runs 2
anima-xdna-chain $checkpoint `
  --fixture $fixture `
  --start-block 0 --end-block 28 --warmups 1 --runs 3 --profile
# Turbo's validated policy reports its bounded transient amplification while
# retaining a strict final-output gate:
anima-xdna-chain "<TURBO_V11_SAFETENSORS>" `
  --fixture "<CAPTURED_REAL_PREBLOCK_INPUT_PT>" `
  --start-block 0 --end-block 28 --max-block-nrms 0.10 --max-final-nrms 0.02
anima-xdna-cache build $checkpoint
anima-xdna-cache verify $checkpoint
anima-xdna-cache list
```

`anima-xdna-probe` must report `target: npu2` and `runtime: xrt`.
The block and chain CLIs report exact shape/range, NPU/host operation timing,
dispatches, H2D/D2H staging bytes, allocations/resident hits, host-copy,
cache-sync and kernel/wait time, logical NPU FLOPs, cold/populate/cached timing,
and blockwise numerical errors.

On the validated Ryzen AI 9 365 / NPU Strix system, Anima Base block 0 at
1024/512 captured inputs produced:

| Metric | Legacy per-head path | Resident stitched path |
|---|---:|---:|
| XRT dispatches/block | 83 | 23 |
| H2D staging/block | 612,237,312 B | 156,631,040 B |
| D2H staging/block | 314,048,512 B | 0 B |
| cached wall/block | about 2.086 s | 0.236-0.362 s |

One representative steady optimized block measured 252.4 ms wall, including
111.2 ms XRT kernel/wait, 35.7 ms explicit host operations, 2.5 ms mapped-BO
host copies, and 3.3 ms cache synchronization. The legacy synthetic benchmark
was 0.662 s and its CPU oracle was 0.717 s. The optimized path is both faster
than that CPU result and materially reduces dispatch and staging.

The first build of all four stitched 16-head artifacts took approximately
862 s on this toolchain; the native Triton cache reuses those shape-specific
artifacts for every block and later process. This compile cost is never
reported as steady execution.

The full 28-block resident chain measured:

- 644 dispatches (`23 x 28`);
- cached steady wall times of 7.02, 7.03, and 7.15 s;
- approximately 0.25 s median per block;
- 23.19 s for the CPU oracle (about 3.3x slower);
- 4,385,669,120 B of H2D staging and zero explicit D2H staging;
- about 3.22 s total XRT kernel/wait, 73-75 ms host copy, and 86-89 ms cache
  synchronization;
- final normalized RMS error `0.007466678`;
- worst normalized RMS error `0.041646643` at block 13;
- final max/mean absolute error 1664 / 7.2158.

Absolute errors become large at sparse, very high-magnitude activations in the
deep chain, so full-chain validation uses normalized RMS at every block with a
fixed 5% gate; all blocks pass. This does not replace strict single-block
coverage: the deterministic block integration still compares elementwise with
`rtol=0.02`, `atol=0.003`. The CPU adapter was also checked against the
installed Diffusers 0.39.0 authoritative 28-block chain and ended at about
0.7% normalized RMS difference.

Turbo's separate 10% intermediate / 2% final chain gate is defined for the
captured real preprocessing fixture used by the ComfyUI validation. The
default synthetic seed-0 input is deliberately outside that claim and was
rejected at 9.61% final normalized RMS. Always pass a real captured fixture
when reproducing the Turbo chain gate; the real one-step ComfyUI sampler test
is the end-to-end oracle.

Failures use distinct categories and exit codes: dependency unavailable (2),
NPU unavailable (3), unsupported shape/config/dtype (4), compilation failure
(5), execution failure (6), and numerical mismatch (7).

## Packed-weight cache

The default cache is
`%USERPROFILE%\.cache\anima-xdna\weights\<cache-key>`. It stores the exact BF16
K-by-N tensors consumed by the AIE2P Linear kernels after transpose, K-chunking,
zero-padding, and layout preparation, plus the four small Q/K RMSNorm vectors
per block. It does not copy or bundle the source safetensors and is not a
portable or distributable model format. The cache may be deleted and rebuilt
at any time; model privacy and license obligations still apply to the original
checkpoint.

The human-readable `manifest.json` records:

- a full SHA-256 content fingerprint of every used tensor byte range, every
  safetensors header, and a shard index when present;
- model family, complete architecture, block range, and exact weight-key map;
- source/packed dtype and shape, offsets, padding, K-chunking, and layout ABI;
- `base_model_fingerprint`, future `adapter_fingerprints`, `merge_strength`,
  `quantization_scheme`, and `layout_version` namespace fields;
- Triton-XDNA, MLIR-AIE/AIR, LLVM-AIE, XRT SDK, `npu2`, and PoC cache ABI;
- one payload SHA-256 and a SHA-256 for every packed tensor record.

A warm hit verifies Windows file identity/change-time/size, all headers, and
three distributed byte ranges in every used source tensor. The full content
hash established at build remains in the cache key. It concurrently verifies
the complete 3.96 GB payload SHA-256 before mmap. Any guard, header, range,
manifest, size, or digest mismatch is a miss or explicit integrity failure;
corrupt entries are never loaded silently. Builds use an exclusive
per-key lock, write into a unique temporary directory, fsync the payload, and
atomically rename the completed directory. Failed builds clean only their own
temporary directory.

`anima-xdna-chain` enables the cache by default. Use `--no-weight-cache`,
`--cache-dir PATH`, or `--rebuild-cache` to disable, relocate, or explicitly
regenerate it. `anima-xdna-block` has the same switches.
`anima-xdna-linear --weight-cache` reuses the full entry for exact
`transformer_blocks.N.*` Linear keys. Management commands are:

```powershell
anima-xdna-cache inspect CHECKPOINT
anima-xdna-cache verify CHECKPOINT
anima-xdna-cache rebuild CHECKPOINT
anima-xdna-cache list
anima-xdna-cache prune CHECKPOINT
anima-xdna-cache prune --entry-key <64-hex-cache-key>
```

`prune` resolves and verifies one exact manifest entry; it never recursively
deletes a wildcard or the cache root. Payload pages are memory-mapped and
copied into process-local XRT BOs. BO handles/device pointers are never
persisted. CLI timings keep fingerprint/guard validation, packing, mmap,
payload verification, and BO population separate.

For Anima Base v1.0 blocks 0-27 the final entry contains 644 unique records:
532 Linear K-chunks and 112 RMSNorm tensors. It occupies 3,963,645,952 bytes
for 3,875,565,568 logical source bytes; zero padding adds 88,080,384 bytes
(2.27%). Measurements on the validated system were:

| Phase | Time |
|---|---:|
| cache disabled, cold prepare | 10.089 s |
| cache disabled, OS-warm prepare | 5.355-5.538 s |
| clean cold fingerprint | 5.528-5.758 s |
| clean cold pack/write | 13.409-13.671 s |
| clean cold total | 19.834-19.865 s |
| new-process warm source validation | 34-42 ms |
| new-process warm payload verification | 4.316-4.529 s |
| new-process warm total | 4.330-4.546 s |

The full first-run 28-block path, including verification, BO population,
kernel execution, and host stages, fell from 23.822 s without the cache to
21.843 s with a verified warm hit. The first resident upload copied
3,963,617,280 packed Linear bytes into weight BOs and measured 516.8 ms without
the cache versus 631.0 ms from mmap-backed payload pages in the compared runs;
the cache removes source transpose/padding, not BO upload. Once resident, both
paths reuse the same BOs and have identical transfer behavior. Cached and
uncached 28-block XDNA outputs were bitwise equal, and the existing Diffusers/
CPU normalized-RMS gate continued to pass. Times vary with OS file-cache state;
cold build time is intentionally not presented as warm-hit performance.

## Tests and limitations

CPU tests cover exact key/config discovery, lazy block loading, shape rejection,
layout/padding, deterministic inputs, RoPE, attention masks, the complete CPU
block residual path, block-index mapping, and CPU range execution. Real-NPU
tests cover Linear, stitched-versus-legacy block attention, the optimized
23-dispatch block, resident reuse, and the complete 28-block numerical chain:

```powershell
$env:ANIMA_XDNA_CHECKPOINT = "<DIFFUSERS_TRANSFORMER_SAFETENSORS>"
$env:ANIMA_XDNA_CHAIN_FIXTURE = "<CAPTURED_BLOCK_INPUT_PT>"
python -m unittest discover -s tests -v
```

Without those variables or the toolchain, hardware tests skip with the precise
missing boundary.

The low-level CLIs remain independently useful, while the current
[`comfyui_xdna_nodes`](../comfyui_xdna_nodes/README.md) package now integrates
the resident chain into ComfyUI. Host normalization, softmax, GELU, gates, and
residuals still expose mapped buffers to PyTorch and require cache
synchronization; the reported H2D figure is bytes copied into mapped BO pages
on a unified-memory APU, not necessarily physical DRAM traffic. LoRA and INT8
remain intentionally unsupported.
