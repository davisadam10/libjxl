# Walkthrough: Unified Streaming Decoding (Full, ROI, Scaled & Scaled-ROI) in `libjxl`

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

## Verification & Validation Results

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
