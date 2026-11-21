# Implementation Plan: Maintainer-Aligned Unified Streaming Decoding (Full, ROI, Scaled & Scaled-ROI)

## 1. Goal & Maintainer-Alignment Principles

Ensure all four experimental decode modes—**Full-frame**, **ROI**, **Scaled (`1x/2x/4x/8x`)**, and **Scaled-ROI**—work reliably and measurably under **both** streaming transports:
1. **Sequential incremental streaming** (`JxlDecoderSetInput` / `JxlDecoderReleaseInput` / `JxlDecoderProcessInput` with arbitrary chunk sizes down to 1 byte, zero-buffer skipping of unneeded sections, and early frame completion).
2. **Seekable / sparse byte-range streaming** (`JxlDecoderExperimentalGetRequiredInputRanges`, optional `#5143`-aligned `GetNextRequiredOffset`, and `JxlDecoderExperimentalSetInputAtOffset`, driven by a transport-agnostic caller adapter such as local `pread` or HTTP 206).

### Upstream Alignment Rules (Binding)
- **`lib/include/jxl/decode.h` remains 100% untouched**: Zero unapproved public ABI symbols or new `JxlDecoderStatus` enum values (honoring **#1623**, **#3519**, **#4297**, **#603**).
- **Backward compatibility first**: Existing sequential full-frame decoding, event order, buffer ownership, and error semantics remain completely unchanged by default.
- **No conflation of partial fidelity with full fidelity (#603)**: Expose `is_partial_fidelity` in experimental telemetry rather than inventing a new public completion event or claiming a native 1:8/Squeeze/progressive preview is full-fidelity.
- **Correctness over selective I/O (#1294, #3881)**: Never decode zero-filled data in place of a missing required section; reject holes in required sections, enforce bounds/overflow checks, and keep source-vs-JXL FP16 validation distinct from ROI/streaming equality.
- **Transport stays outside `libjxl` (#5143)**: `libjxl` reports required offsets/ranges and consumes offset-tagged spans safely; `pread`, range coalescing, caching, and HTTP/S3 adapters live strictly in the caller/test harness.

---

## 2. Audit of Current Gaps in Our Codebase

Inspecting [decode.cc](../lib/jxl/decode.cc), [dec_frame.cc](../lib/jxl/dec_frame.cc), and [stage_write.cc](../lib/jxl/render_pipeline/stage_write.cc) revealed six concrete gaps to close for production-grade streaming across all 4 modes:

| # | Gap Identified | Location | Impact |
|---|---|---|---|
| **1** | **`FrameDecoder::Flush()` does not handle downscaled paths** | [dec_frame.cc](../lib/jxl/dec_frame.cc#L1030-L1095), [stage_write.cc](file:///home/antigravity/davisadam10/lib/jxl/render_pipeline/stage_write.cc#L442-L498) | Calling `JxlDecoderFlushImage()` during `JXL_DEC_FRAME_PROGRESSION` skips `WriteToOutputStage::FinalizeOutput()` (leaving Path C/D buffers empty), does not pass `output_downsampling_` to Modular Squeeze, and does not feed `shared->dc` for Path A. Also `FinalizeOutput()` unconditionally clears `accum_main_`, which breaks multi-pass progressive flushes. |
| **2** | **Truncated sparse input can loop instead of erroring** | [decode.cc](file:///home/antigravity/davisadam10/lib/jxl/decode.cc#L2057-L2063) | When `sparse_mode_active` is true and a required section is missing/incomplete, `ProcessInput` returns `JXL_DEC_NEED_MORE_INPUT` even after `JxlDecoderCloseInput()` is called. |
| **3** | **No guard against mixing sequential & sparse input mid-frame** | [decode.cc](../lib/jxl/decode.cc#L1326-L1389) | Calling `JxlDecoderSetInput` while `sparse_mode_active == true` (or vice versa without releasing/finishing the frame) is not explicitly rejected. |
| **4** | **Missing streaming telemetry & `#5143` next-offset query** | [decode_roi_experimental.h](../lib/include/jxl/decode_roi_experimental.h), [decode.cc](file:///home/antigravity/davisadam10/lib/jxl/decode.cc) | `JxlExperimentalRegionDecodeStats` does not yet report `bytes_skipped_sequential`, `sparse_bytes_received`, `sparse_peak_bytes_buffered`, or `is_partial_fidelity`. No narrow `GetNextRequiredOffset` helper (issue `#5143` style) shares the range-planner backend. |
| **5** | **Untested sequential chunked streaming for Scaled & Scaled-ROI** | [decode_scale_test.cc](../lib/jxl/decode_scale_test.cc) | Downscaled modes were only tested with full-buffer sequential input and sparse input—not 1B/64B/4KB/64KB incremental streaming, early stop before EOF, or `Reset`/`Rewind` (#5103). |
| **6** | **Untested encoder output modes 0/1/2 & real `pread` cost-model harness** | [encode.h](../lib/include/jxl/encode.h#L397-L419) | Need explicit streaming coverage for raw codestreams, `jxlc`, multi-`jxlp`, and `JXL_ENC_FRAME_SETTING_OUTPUT_MODE` 0/1/2 (including out-of-order `jxlp` with `ftyp` v1), plus a local file-backed `pread()` harness measuring coalesced requests vs seeks vs bytes read. |

---

## 3. Detailed Implementation Plan (4 Phases)

### Phase 1: Streaming Core Hardening & Progressive `Flush()` Support
1. **Non-destructive vs final `FinalizeOutput(bool clear_accum)`**:
   - [render_pipeline_stage.h](../lib/jxl/render_pipeline/render_pipeline_stage.h) & [render_pipeline.h](../lib/jxl/render_pipeline/render_pipeline.h): Update `FinalizeOutput(bool final_pass = true)` (default `true` so existing callers need no changes).
   - [stage_write.cc](../lib/jxl/render_pipeline/stage_write.cc):
     - Add `ResetAccumulator()` (or clear `accum_main_`/`accum_extra_` when `PrepareForThreads` / `ClearDone` starts a fresh progressive redraw) and only free `accum_main_`/`accum_extra_` in `FinalizeOutput(bool final_pass)` when `final_pass == true`.
     - Guard against double-accumulation when `Flush()` force-draws incomplete groups: in `FrameDecoder::Flush()`, when `output_downsampling_ > 1`, clear the accumulator before force-drawing all required groups (or when `scale_mode_ == kNativeVarDCTDC`, feed `shared->dc` directly; when `scale_mode_ == kModularSqueezeNative`, pass `output_downsampling_` to `FinalizeDecoding(..., /*inplace=*/false, output_downsampling_)`).
2. **Sparse EOF/truncation & mixed-mode guards** in [decode.cc](../lib/jxl/decode.cc):
   - In `JxlDecoderProcessCodestream`: when `!all_sections_done && dec->sparse_mode_active`, return `JXL_INPUT_ERROR("incomplete sparse input")` if `dec->input_closed` is true, else `JXL_DEC_NEED_MORE_INPUT`.
   - In `JxlDecoderSetInput`: if `dec->sparse_mode_active` is true, return `JXL_API_ERROR("Cannot mix JxlDecoderSetInput with active sparse offset input")` (caller must finish the frame, `JxlDecoderRewind`, or `JxlDecoderReset`).
   - In `JxlDecoderExperimentalSetInputAtOffset`: verify `dec->next_in == nullptr && dec->avail_in == 0` (caller called `JxlDecoderReleaseInput()` before switching from sequential header parsing to sparse payload delivery) — or cleanly document and enforce the handoff invariant.
3. **Zero-copy prefix ingestion check**:
   - Verify that when a caller parses `BASIC_INFO + FRAME + TOC` sequentially with a small chunk (e.g., 4 KB) and calls `JxlDecoderReleaseInput()`, `frame_data_file_offset` is exact and `GetRequiredInputRanges()` returns all still-needed frame sections cleanly without losing any bytes.

### Phase 2: Experimental Streaming Telemetry & `#5143` Next-Offset Query
All additions stay strictly in [decode_roi_experimental.h](../lib/include/jxl/decode_roi_experimental.h) and [decode.cc](../lib/jxl/decode.cc):
1. **Extend `JxlExperimentalRegionDecodeStats`**:
   - `uint64_t bytes_skipped_sequential` — total bytes skipped zero-copy via `AdvanceCodestream` in sequential streaming mode.
   - `uint64_t sparse_bytes_received` — total payload bytes ingested via `JxlDecoderExperimentalSetInputAtOffset`.
   - `uint64_t sparse_peak_bytes_buffered` — peak in-flight sparse section buffer bytes held simultaneously (proving bounded memory as completed sections are freed immediately).
   - `JXL_BOOL is_partial_fidelity` — `JXL_TRUE` when the frame completed via an approximate/reduced path (`NATIVE_VARDCT_DC`, `MODULAR_SQUEEZE_NATIVE`, or `PROGRESSIVE_PASS_SKIP`), `JXL_FALSE` for full-fidelity (`NONE` or `STREAMING_DOWNSAMPLE_FALLBACK`).
2. **Add `#5143`-aligned next-offset query** (sharing the exact same `GetRequiredCodestreamRanges` backend):
   ```c
   JXL_EXPORT JxlDecoderStatus JxlDecoderExperimentalGetNextRequiredOffset(
       const JxlDecoder* dec, uint64_t* offset, uint64_t* length);
   ```
   - Returns the first still-missing coalesced physical file range `{offset, length}` (or `length = 0` when all required sections are already buffered/processed).
   - Lets callers drive a minimal `#5143`-style `pread(fd, buf, length, offset)` loop without allocating a range array, while batch/async HTTP/S3 callers continue using `JxlDecoderExperimentalGetRequiredInputRanges`.
   - Explicitly documented in header comments as **EXPERIMENTAL / UNAPPROVED PROTOTYPE** pending upstream discussion on `#1623` / `#5143`.

### Phase 3: Container Layouts, Encoder Output Modes 0/1/2 & `Reset`/`Rewind` (#5103)
1. **Container & Encoder Mode Verification**:
   - Verify sequential streaming across all 4 modes (Full, ROI, Scaled, Scaled-ROI) on:
     - Raw codestream
     - Single `jxlc` container
     - Multi-`jxlp` container (e.g., 4 KB `jxlp` box splits)
     - `JXL_ENC_FRAME_SETTING_OUTPUT_MODE = 0` (buffered progressive)
     - `JXL_ENC_FRAME_SETTING_OUTPUT_MODE = 1` (seek-based streaming encoder output)
     - `JXL_ENC_FRAME_SETTING_OUTPUT_MODE = 2` (out-of-order `jxlp` with `ftyp` minor version 1)
   - Verify sparse mode cleanly reports `JXL_EXP_FALLBACK_MULTI_JXLP_UNKNOWN_BOUNDARIES` (and refuses out-of-bounds offset mapping) when a frame spans multiple `jxlp` boxes, while sequential streaming succeeds seamlessly on the exact same file.
2. **`JxlDecoderReset` & `JxlDecoderRewind` State Hygiene (#5103)**:
   - Audit `RewindDecodingState()` and `JxlDecoderReset()` in [decode.cc](../lib/jxl/decode.cc) to ensure `bytes_skipped_sequential`, `sparse_bytes_received`, `sparse_peak_bytes_buffered`, `downsampling_target`, `dc_frame_progression_done`, `frame_prog_detail`, `output_downsampling`, and `scale_policy` are cleanly reset/preserved with zero stale state across:
     - `Scaled-ROI Sparse` → `Reset` → `Full Sequential`
     - `Scaled-ROI Sparse` → `Rewind` → re-decode same target (bitwise identical)
     - `Scaled Sequential` → `Rewind` → change ROI/scale → decode new target (bitwise identical to fresh decoder)

### Phase 4: Unified 4×2 Streaming Test Suite & Local `pread()` Benchmark Harness
Create [decode_streaming_test.cc](../lib/jxl/decode_streaming_test.cc) and register it in [jxl_lists.cmake](../lib/jxl_lists.cmake):
1. **`FourByTwoStreamingMatrixVarDCT`** (Full, ROI, Scaled `1:2/1:4/1:8`, Scaled-ROI × Sequential `{1B, 64B, 4KB, 64KB}` & Sparse `{GetRequiredInputRanges, GetNextRequiredOffset, reversed, 1B-split, overlapping}`):
   - Verifies **bitwise identical** output between full-buffer, sequential chunked streaming, batch sparse streaming, and `#5143` next-offset streaming for every cell in the 4×2 matrix.
   - Verifies `bytes_skipped_sequential > 0` for ROI/Scaled/Scaled-ROI sequential streaming and `sparse_bytes_received < total_file_size` + bounded `sparse_peak_bytes_buffered` for sparse streaming.
   - Verifies `is_partial_fidelity` is `JXL_TRUE` for Path A/B/C and `JXL_FALSE` for Full/ROI/Path D.
2. **`FourByTwoStreamingMatrixModularSqueezeAndLossless`**:
   - Tests both Modular with Squeeze (`responsive=1, progressive_ac=1`) and adversarial non-Squeeze lossless Modular (`responsive=0`) across sequential chunked and sparse streaming.
3. **`ProgressiveFlushStreamingAllModes`**:
   - Subscribes to `JXL_DEC_FRAME_PROGRESSION`, feeds small sequential chunks, calls `JxlDecoderFlushImage()` at DC and intermediate pass milestones for Full, ROI, Scaled, and Scaled-ROI, and verifies valid non-corrupt intermediate rasters and exact final raster upon `JXL_DEC_FULL_IMAGE`.
4. **`ContainerLayoutsAndEncoderOutputModes`**:
   - Tests raw codestream, `jxlc`, multi-`jxlp`, and encoder `OUTPUT_MODE` 0, 1, and 2 across all 4 modes in sequential streaming, and verifies explicit `JXL_EXP_FALLBACK_MULTI_JXLP_UNKNOWN_BOUNDARIES` in sparse mode for multi-`jxlp`.
5. **`AdversarialSparseAndModeIsolation`**:
   - Truncated sparse input + `JxlDecoderCloseInput()` → `JXL_DEC_ERROR` (never hangs, never zero-fills missing required sections).
   - Mixing `JxlDecoderSetInput()` while `sparse_mode_active == true` → `JXL_DEC_ERROR`.
   - Integer overflow in `offset + size` → `JXL_DEC_ERROR`.
   - Out-of-order, duplicated, overlapping, and 1-byte-split sparse spans → bitwise identical to sequential decode.
6. **`ResetAndRewindStateHygiene`**:
   - Cross-mode `Reset` and `Rewind` transitions on a single `JxlDecoder*` compared bitwise against fresh decoders.
7. **`Float2kPreadCoalescingStreaming`** (fast, runs in main suite):
   - Writes a 2048×1080 linear HDR RGB HALF JXL to a temp file and drives all 4 modes (Full, ROI, Scaled `1:4`/`1:8`, Scaled-ROI) via real POSIX `pread()` with configurable coalescing gap (`0 B` NVMe vs `64 KB` HDD NAS), verifying HDR range preservation (`< 0.0` and `> 1.0`), exactness/PSNR, and real `pread()` byte/call counts.
8. **`DISABLED_Float8kStreamingMatrixBenchmark`** (opt-in via `--gtest_also_run_disabled_tests --gtest_filter=*Float8kStreaming*`):
   - Real local-file `pread()` + sequential streaming benchmark on an 8192×4320 RGB HALF frame across all 4 modes (Full 8K, 512×512 ROI, 2K 1:4, 1K 1:8, and 512×512 Scaled-ROI at 1:4) printing full telemetry (unique bytes read, overfetch, `pread` calls, skipped bytes, peak sparse buffer, wall latency).

---

## 4. Verification Plan
- Build and run:
  - `./build/lib/tests/decode_streaming_test`
  - `./build/lib/tests/decode_scale_test`
  - `./build/lib/tests/decode_region_test`
  - `./build/lib/tests/render_pipeline_test --gtest_brief=1`
  - `./build/lib/tests/decode_test --gtest_brief=1`
  - `./build/lib/tests/decode_streaming_test --gtest_also_run_disabled_tests --gtest_filter=*Float8kStreaming*`
- Confirm `git diff --stat lib/include/jxl/decode.h` is empty.
- Update `doc/roi_decoding_walkthrough.md` and commit `doc/streaming_plan.md` alongside the implementation if requested.

---

## 5. Finalized Decisions

1. **Sparse handoff after sequential header read (Confirmed)**: Require `JxlDecoderReleaseInput()` before switching to `JxlDecoderExperimentalSetInputAtOffset()`, and reject mixing `JxlDecoderSetInput()` while `sparse_mode_active == true` (until frame completion, `JxlDecoderRewind`, or `JxlDecoderReset`).
2. **`#5143` next-offset helper (Confirmed)**: Include `JxlDecoderExperimentalGetNextRequiredOffset(dec, &offset, &length)` alongside `JxlDecoderExperimentalGetRequiredInputRanges` in [decode_roi_experimental.h](../lib/include/jxl/decode_roi_experimental.h), sharing the same TOC/range-planner backend.
