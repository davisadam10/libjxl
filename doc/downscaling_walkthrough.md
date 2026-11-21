# Walkthrough: Power-of-Two Reduced-Resolution (Downscaled) Decoding in `libjxl`

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

## Verification & Validation Results

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
