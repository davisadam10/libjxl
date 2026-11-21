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
