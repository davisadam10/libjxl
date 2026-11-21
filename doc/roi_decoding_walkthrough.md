# Walkthrough: Region-Specific, TOC Byte-Range & Sparse-Input Decoding in `libjxl`

## Summary of Changes

Implemented all four approved phases aligned with upstream `libjxl` maintainer discussions (**#1623**, **#1294**, **#3519**, **#668**, **#1133**, **#5143**, **#586**, **#4328**, **#3881**) while keeping the stable public header [decode.h](../lib/include/jxl/decode.h) **100% untouched**:

### 1. Experimental Prototype C Header (`decode_roi_experimental.h`)
- **[NEW]** [decode_roi_experimental.h](../lib/include/jxl/decode_roi_experimental.h) (registered in [jxl_lists.cmake](../lib/jxl_lists.cmake)):
  - `JxlDecoderExperimentalSetOutputRegion(dec, x, y, width, height)` — sets (or clears with `0, 0`) the requested output ROI in oriented output coordinates.
  - `JxlDecoderExperimentalGetRequiredInputRanges(dec, ranges, range_count)` — two-call sizing API returning coalesced original file byte ranges (`JxlExperimentalInputRange { offset, length }`) required to decode the ROI and its dependencies.
  - `JxlDecoderExperimentalSetInputAtOffset(dec, data, size, offset)` — bounded sparse/discontinuous input supporting arbitrary order, partial chunks, overlaps, and duplicates without requiring bytes outside the ROI.
  - `JxlDecoderExperimentalGetRegionDecodeStats(dec, stats)` — exposes `JxlExperimentalRegionFallbackReason`, group skip counts, required section counts, and planned byte totals.

### 2. Output & Render Pipeline Region Primitives
- [dec_cache.h](../lib/jxl/dec_cache.h) & [dec_cache.cc](../lib/jxl/dec_cache.cc): Added `PassesDecoderState::output_region` and `ComputeUnorientedRegion()` (inverting `transpose` → `flip_x` → `flip_y`), forwarding region coordinates to `WriteToOutputStage`, `FastXYBStage`, and `RenderPipeline::SetRenderRegion` (suppressed on referenceable frames so `frame_storage_for_referencing` stays complete).
- [stage_write.h](../lib/jxl/render_pipeline/stage_write.h), [stage_write.cc](../lib/jxl/render_pipeline/stage_write.cc), [stage_xyb.h](../lib/jxl/render_pipeline/stage_xyb.h), [stage_xyb.cc](../lib/jxl/render_pipeline/stage_xyb.cc): Clip rows to `unoriented_region` while preserving full-image post-flip coordinates for `FlipX` and 8-bit ordered dithering (`kDither`), subtracting region origin only at final buffer/callback write.
- [render_pipeline.h](../lib/jxl/render_pipeline/render_pipeline.h), [low_memory_render_pipeline.h](../lib/jxl/render_pipeline/low_memory_render_pipeline.h), [low_memory_render_pipeline.cc](../lib/jxl/render_pipeline/low_memory_render_pipeline.cc): Added `GroupBorder()`, `BaseColorShift()`, `SetRenderRegion()`, `RenderOutOfFramePadding()` (up to 4 disjoint out-of-frame strips for blended/cropped frames), and skipping of disjoint `ready_rect`s.

### 3. FrameDecoder Dependency & TOC Byte-Range Planner
- [dec_frame.h](../lib/jxl/dec_frame.h) & [dec_frame.cc](../lib/jxl/dec_frame.cc):
  - Recorded `section_codestream_offsets_` in `InitFrame()`.
  - Implemented `EstimateGroupBorderFromHeader()` and `PlanRequiredGroups(bool after_dc_global)` so required groups and coalesced file byte ranges can be planned immediately after `FrameHeader + TOC`, and refined after DC Global (with explicit `RegionFallbackReason::kModularGlobalTransforms` fallback if `ModularFrameDecoder::UsesFullImage()` is true).
  - Implemented `GetRequiredCodestreamRanges()` (handling arbitrary TOC permutations) and `GetRegionDecodeStats()`.
  - Updated `SetImageOutput`, `AddExtraChannelOutput`, `ProcessSections`, `Flush`, `HasDecodedAll`, `NumCompletePasses`, `HasEverything`, and `FinalizeFrame`.

### 4. Streaming Zero-Buffer Skip & Sparse Input Assembly
- [decode.cc](../lib/jxl/decode.cc):
  - Tracked exact `frame_data_file_offset` and `frame_data_file_offset_valid` at the end of `FrameStage::kHeader` (supporting bare codestreams, `jxlc` containers, and contiguous `jxlp` boxes).
  - Implemented bounded sparse section assembly (`SparseSectionData` + `MergeInterval`) and sparse dispatch in `JxlDecoderProcessSections` (freeing each section buffer as soon as it is decoded).
  - Upgraded sequential push streaming in `JxlDecoderProcessSections` to skip unneeded sections via `AdvanceCodestream()` with zero byte buffering and emit `JXL_DEC_FULL_IMAGE` early once all required sections complete.
  - Updated `GetMinSize` to return region-sized buffer requirements.

---

## Verification & Validation Results

### Test Suites Executed
| Suite | Command | Result |
|---|---|---|
| New ROI, Range Planning, Sparse I/O & Streaming Suite | `./build/lib/tests/decode_region_test` | **11 / 11 PASSED** (7.59 s, 0 pixel diffs) |
| On-Demand 8K ACES RGB HALF Sparse Benchmark | `./build/lib/tests/decode_region_test --gtest_also_run_disabled_tests --gtest_filter=*Float8k*` | **1 / 1 PASSED** (2.32 s, 0 pixel diffs) |
| Existing Render Pipeline Regression Suite | `./build/lib/tests/render_pipeline_test` | **1,198 / 1,198 PASSED** (37.9 s) |
| Existing Decoder Regression Suite | `./build/lib/tests/decode_test` | **2,895 / 2,895 PASSED** (30.6 s) |

### Key Coverage Verified in [decode_region_test.cc](../lib/jxl/decode_region_test.cc)
- **`SparseRandomAccessAndRangePlanning`**: Bare codestream, `jxlc` container, reversed/sub-chunked/overlapping sparse delivery, and center-first permuted TOC (`group_order = 1`) — all 0-diff bit-exact with `> 48 / 64` groups skipped and `< 50%` bytes read.
- **`Float2kRgbHalfSparseRandomAccess`**: Fast 2K (2048×1080) ACEScg linear RGB HALF (`JXL_TYPE_FLOAT16`) lossless Modular sparse random-access (bare + container); aligned 256×256 crop decodes **1 / 40 groups** with 0-diff bitwise FP16 equality.
- **`All65536HalfPatternsBitExact`** (#3881): All 65,536 IEEE-754 HALF bit patterns (all finite normals, all subnormals, `-0.0`, and Inf/NaN) across 4 groups — 100% bitwise equal to full-frame `libjxl` decode.
- **`VarDCTFeatureCombinations` & `LargeVarDCT2KRegions`**: Gaborish on/off, EPF 0/1/2/3, noise synthesis, resampling 2×/4×, progressive multi-pass, permuted TOC, unaligned 4-group crossings, 1×1 pixel, and full-image crops.
- **`ModularGroupIndependentAndSqueezeFallback`**: Verified group-skipping lossless Modular (`JXL_EXP_FALLBACK_NONE`) and global Squeeze fallback (`JXL_EXP_FALLBACK_MODULAR_GLOBAL_TRANSFORMS`) under both sequential streaming and 2-round sparse input.
- **`AllEightExifOrientations`, `OutputFormatsAndCallbacks`, `MultiFrameBlendingAndOutOfFramePadding`, `StreamingChunkSizesAndEarlyCompletion`, `ApiValidationAndRewind`**: All 8 EXIF orientations (`UINT8` dither + `FLOAT`), 1–4 channels × 4 data types × buffer/simple/MT callbacks, cropped blended frames + `RenderPadding`, 1B/64B/4KB/64KB streaming with early completion, and `Rewind` + bounds validation.

### 8K (7680×4320) ACES RGB HALF Sparse Random-Access Benchmark Result
```text
[8K ACES HALF 512x512] file=161059762 bytes, sparse_read=1274131 bytes (0.79%), ranges=3, groups=4/510, full=1365.7 ms, sparse=10.9 ms
```
- **I/O reduction**: **1.27 MB** read out of **161.1 MB** (**0.79%** of file — **126× less storage I/O**) in **3** coalesced byte ranges.
- **Compute reduction**: **4 / 510** groups decoded (**506 groups skipped**).
- **Latency**: **10.9 ms** vs. **1,365.7 ms** full decode (**125× faster**), with **100% bitwise identical** FP16 pixels.

---

# Part 2: Power-of-Two Reduced-Resolution (Downscaled) Decoding

## Summary of Changes

Implemented upstream-aligned experimental power-of-two downscaled decoding (`factor ∈ {1, 2, 4, 8}`) inside the same experimental header [decode_roi_experimental.h](../lib/include/jxl/decode_roi_experimental.h), keeping [decode.h](../lib/include/jxl/decode.h) **100% untouched**:

### 1. Experimental Downscaling API & Exactness Policies
- [decode_roi_experimental.h](../lib/include/jxl/decode_roi_experimental.h) & [decode.cc](../lib/jxl/decode.cc):
  - Added `JxlExperimentalScalePolicy` (`JXL_EXP_SCALE_FAST_APPROXIMATE`, `JXL_EXP_SCALE_REQUIRE_FAST_NATIVE`, `JXL_EXP_SCALE_REFERENCE_RESAMPLE`) and `JxlExperimentalScaleDecodeMode` (`NONE`, `NATIVE_VARDCT_DC`, `MODULAR_SQUEEZE_NATIVE`, `PROGRESSIVE_PASS_SKIP`, `STREAMING_DOWNSAMPLE_FALLBACK`).
  - Added `JxlDecoderExperimentalSetOutputDownsampling(dec, downsampling, policy)` and `JxlDecoderExperimentalGetScaledOutputInfo(dec, xsize, ysize, fast_native_available, scale_mode)`.
  - Extended `JxlExperimentalRegionDecodeStats` with `downsampling`, `scale_policy`, `scale_mode`, `total_passes`, and `required_passes`.
  - Updated `GetMinSize` and `ValidateDecodeRegionBounds` so buffer sizes reflect `DivCeil(W, factor) × DivCeil(H, factor)` (or the ROI dimensions in scaled output coordinates when combined with `SetOutputRegion`).

### 2. All Four Downscaling Decode Paths
- **Path A — Native VarDCT 1:8 DC** ([dec_frame.h](../lib/jxl/dec_frame.h), [dec_frame.cc](../lib/jxl/dec_frame.cc), [dec_cache.h](../lib/jxl/dec_cache.h), [dec_cache.cc](../lib/jxl/dec_cache.cc)):
  - Skips AC Global and all AC groups (`required_groups = 0`, `required_passes = 0`), feeds `shared->dc` directly into a native `DivCeil(W, 8) × DivCeil(H, 8)` color-conversion + output pipeline.
- **Path B — Native Modular Squeeze Partial Inverse** ([squeeze.h](../lib/jxl/modular/transform/squeeze.h), [squeeze.cc](../lib/jxl/modular/transform/squeeze.cc), [dec_modular.h](../lib/jxl/dec_modular.h), [dec_modular.cc](../lib/jxl/dec_modular.cc)):
  - Added `PartialInvSqueeze()` stopping inverse Haar steps at `target_shift = CeilLog2Nonzero(factor)` and `ModularFrameDecoder::HasSqueezeForDownsampling()` (metadata dry-run guaranteeing exact `DivCeil(W, factor) × DivCeil(H, factor)` dimensions across all channels).
  - Skips ModularAC passes with `maxShift < target_shift` (and skips AC Global + all AC passes at `1:8` since shifts `>= 3` live in `ModularDC`).
- **Path C — Progressive VarDCT Pass Skipping** ([dec_frame.cc](../lib/jxl/dec_frame.cc)):
  - Selects the smallest pass count `k` satisfying `GetDownsamplingTargetForCompletedPasses(k) <= factor`, skips finer passes in `IsSectionRequiredPlanned` / `GetRequiredCodestreamRanges`, and sets `force_draw = true` on the final required pass.
  - Composes cleanly with spatial ROI group skipping.
- **Path D — Low-Memory Streaming Float Box Downsample** ([render_pipeline_stage.h](../lib/jxl/render_pipeline/render_pipeline_stage.h), [render_pipeline.h](../lib/jxl/render_pipeline/render_pipeline.h), [stage_write.h](../lib/jxl/render_pipeline/stage_write.h), [stage_write.cc](../lib/jxl/render_pipeline/stage_write.cc)):
  - Accumulates full-res rows in unclamped `float` space (preserving negative values, HDR highlights `> 1.0`, subnormals, and `-0.0`) with row-striped mutexes and exact border-tile division `tile_w * tile_h` in `WriteToOutputStage::FinalizeOutput()`.
  - Handles all 8 EXIF orientations, extra channels, alpha unpremultiplication, and pixel callbacks.

## Verification & Validation Results (Part 2)

| Suite | Command | Result |
|---|---|---|
| New Downscaled Decoding Suite ([decode_scale_test.cc](../lib/jxl/decode_scale_test.cc)) | `./build/lib/tests/decode_scale_test` | **10 / 10 PASSED** (1.21 s) |
| On-Demand 8K→2K/1K RGB HALF Benchmark | `./build/lib/tests/decode_scale_test --gtest_also_run_disabled_tests --gtest_filter=*Float8kTo2k*` | **1 / 1 PASSED** (8.11 s) |
| Experimental ROI & Sparse I/O Suite | `./build/lib/tests/decode_region_test` | **11 / 11 PASSED** (7.50 s) |
| Existing Render Pipeline Regression Suite | `./build/lib/tests/render_pipeline_test --gtest_brief=1` | **1,198 / 1,198 PASSED** |
| Existing Decoder Regression Suite | `./build/lib/tests/decode_test --gtest_brief=1` | **2,895 / 2,895 PASSED** (31.97 s) |

### 8K (8192×4320) RGB HALF Progressive VarDCT Downscale Benchmark
```text
[8K Benchmark] total=16512334 B | full=537.7 ms | 1:4(2K)=332.4 ms (2584855 B, mode=3) | 1:8(1K)=51.3 ms (613854 B, mode=1)
```
- **8K → 2K (1:4, Progressive Pass Skip)**: **332.4 ms** vs. **537.7 ms** full decode, reading **2.58 MB / 16.51 MB (15.7%)**.
- **8K → 1K (1:8, Native VarDCT 1:8 DC)**: **51.3 ms** (**10.5× faster**), reading **613.9 KB / 16.51 MB (3.7%)**.

---

# Part 3: Unified Streaming Decoding (Full, ROI, Scaled & Scaled-ROI)

## Summary of Changes

Hardened and unified streaming decoding across all four modes (**Full**, **ROI**, **Scaled**, **Scaled-ROI**) over both **sequential chunked streaming** and **seekable/sparse byte-range streaming**, staying strictly within [decode_roi_experimental.h](../lib/include/jxl/decode_roi_experimental.h) while leaving [decode.h](../lib/include/jxl/decode.h) **100% untouched**:

### 1. Progressive `Flush()` & Non-Destructive Accumulator Lifecycle
- [render_pipeline_stage.h](../lib/jxl/render_pipeline/render_pipeline_stage.h), [render_pipeline.h](../lib/jxl/render_pipeline/render_pipeline.h), [stage_write.cc](../lib/jxl/render_pipeline/stage_write.cc):
  - Updated `FinalizeOutput(bool final_pass = true)` and added `ResetOutputAccumulator()` so progressive `JxlDecoderFlushImage()` calls do not destroy downsampling accumulators mid-frame.
- [dec_frame.cc](../lib/jxl/dec_frame.cc):
  - Updated `FrameDecoder::Flush()` to handle all four downscaling paths (native VarDCT 1:8 DC, Modular Squeeze partial inverse, progressive pass skip, and streaming float box downsample) and clear incomplete groups in `render_pipeline` after a progressive flush so group borders are never double-accumulated on subsequent chunks.

### 2. Sparse Mode Hardening, `#5143` Next-Offset Query & Telemetry
- [decode_roi_experimental.h](../lib/include/jxl/decode_roi_experimental.h) & [decode.cc](../lib/jxl/decode.cc):
  - Added `JxlDecoderExperimentalGetNextRequiredOffset(dec, &offset, &length)` aligned with upstream issue **#5143**, sharing the exact same TOC/dependency planner backend as `JxlDecoderExperimentalGetRequiredInputRanges`.
  - Enforced strict sequential/sparse handoff: `JxlDecoderExperimentalSetInputAtOffset()` requires `JxlDecoderReleaseInput()` first, and `JxlDecoderSetInput()` returns `JXL_DEC_ERROR` while sparse mode is active for the current frame.
  - Handled truncated sparse input cleanly: calling `JxlDecoderCloseInput()` with incomplete required sections returns `JXL_DEC_ERROR` instead of looping or zero-filling.
  - Extended `JxlExperimentalRegionDecodeStats` with `bytes_skipped_sequential`, `sparse_bytes_received`, `sparse_peak_bytes_buffered`, and `is_partial_fidelity` (honoring **#603** without adding unapproved public `JxlDecoderStatus` values), preserved across frame completion until `RewindDecodingState` or the next frame header.

### 3. Unified 4×2 Streaming Test Suite & Local `pread()` Harness
- **[NEW]** [decode_streaming_test.cc](../lib/jxl/decode_streaming_test.cc) (registered in [jxl_lists.cmake](../lib/jxl_lists.cmake)):
  - `FourByTwoStreamingMatrixVarDCT` & `FourByTwoStreamingMatrixModularSqueezeAndLossless`: Full, ROI, Scaled, and Scaled-ROI × Sequential (`1B/64B/4KB/64KB`) and Sparse (`GetRequiredInputRanges`, reversed, overlapping, and `#5143` `GetNextRequiredOffset`).
  - `ProgressiveFlushStreamingAllModes`: `JXL_DEC_FRAME_PROGRESSION` + `JxlDecoderFlushImage()` across Full, ROI, Scaled, and Scaled-ROI.
  - `ContainerLayoutsAndEncoderOutputModes`: Raw codestream, `jxlc`, multi-`jxlp` (with explicit `JXL_EXP_FALLBACK_MULTI_JXLP_UNKNOWN_BOUNDARIES`), and `JXL_ENC_FRAME_SETTING_OUTPUT_MODE` 0, 1, and 2 (out-of-order `jxlp` with `ftyp` v1).
  - `AdversarialSparseAndModeIsolation` & `ResetAndRewindStateHygiene` (**#5103**): Truncation, overflow, mixed-mode rejection, and single-decoder `Reset`/`Rewind` parity.
  - `Float2kPreadCoalescingStreaming`: Real POSIX `pread()` with NVMe (`0 B` gap) vs HDD NAS (`64 KB` gap) coalescing on 2048×1080 linear HDR RGB HALF.

## Verification & Validation Results (Part 3)

| Suite | Command | Result |
|---|---|---|
| New Unified 4×2 Streaming Suite ([decode_streaming_test.cc](../lib/jxl/decode_streaming_test.cc)) | `./build/lib/tests/decode_streaming_test` | **7 / 7 PASSED** (1.27 s) |
| On-Demand 8K `pread()` Streaming Matrix Benchmark | `./build/lib/tests/decode_streaming_test --gtest_also_run_disabled_tests --gtest_filter=*Float8kStreaming*` | **1 / 1 PASSED** (8.58 s) |
| Downscaled Decoding Suite | `./build/lib/tests/decode_scale_test --gtest_brief=1` | **10 / 10 PASSED** |
| ROI & Sparse I/O Suite | `./build/lib/tests/decode_region_test --gtest_brief=1` | **11 / 11 PASSED** |
| Render Pipeline Regression Suite | `./build/lib/tests/render_pipeline_test --gtest_brief=1` | **1,198 / 1,198 PASSED** |
| Decoder Regression Suite | `./build/lib/tests/decode_test --gtest_brief=1` | **2,895 / 2,895 PASSED** (31.38 s) |

### 8K (8192×4320) RGB HALF Local `pread()` Streaming Matrix
```text
[8K Streaming Matrix] total_file=3206295 B (3.06 MiB)
  Full 8K                    | 8192x4320 | pread= 3207309 B (100.03%) in   2 calls | peak_buf=3203213 B | 432.3 ms | mode=0 partial=0
  ROI 512x512 (1:1)          |  512x512  | pread=  478989 B (14.94%) in   8 calls | peak_buf= 474893 B | 144.9 ms | mode=0 partial=0
  Scaled 2K (1:4)            | 2048x1080 | pread= 1478366 B (46.11%) in   2 calls | peak_buf=1474270 B | 324.3 ms | mode=3 partial=1
  Scaled 1K (1:8 DC)         | 1024x540  | pread=  463331 B (14.45%) in   2 calls | peak_buf= 459235 B |  54.8 ms | mode=1 partial=1
  Scaled-ROI 512x512 (1:4)   |  512x512  | pread=  586619 B (18.30%) in  10 calls | peak_buf= 582523 B | 170.3 ms | mode=3 partial=1
```
