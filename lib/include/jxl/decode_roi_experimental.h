/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @file decode_roi_experimental.h
 * @brief EXPERIMENTAL / UNSTABLE prototype C API for region-of-interest (ROI),
 * TOC byte-range planning, and sparse offset-tagged input in libjxl.
 *
 * WARNING: These declarations are fork-experimental prototypes aligned with
 * upstream design discussions (libjxl issues #1623, #1294, #3519, #668,
 * #1133, #5143). Names, structs, and semantics are NOT ratified upstream ABI
 * and may change without notice.
 */

#ifndef JXL_DECODE_ROI_EXPERIMENTAL_H_
#define JXL_DECODE_ROI_EXPERIMENTAL_H_

#include <jxl/decode.h>
#include <jxl/jxl_export.h>
#include <jxl/types.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Byte range in the original input file / codestream required to decode the
 * configured region of interest.
 */
typedef struct {
  /** Byte offset from the start of the original input file/stream. */
  uint64_t offset;
  /** Number of bytes starting at `offset`. */
  uint64_t length;
} JxlExperimentalInputRange;

/**
 * Reason why selective group skipping fell back to full-frame decoding (or
 * JXL_EXP_FALLBACK_NONE when selective skipping is active).
 */
typedef enum {
  /** Selective group skipping is active (no fallback). */
  JXL_EXP_FALLBACK_NONE = 0,
  /** No output region was configured. */
  JXL_EXP_FALLBACK_NO_REGION = 1,
  /** Frame has only a single TOC section (e.g. small image <= 1 group). */
  JXL_EXP_FALLBACK_SINGLE_SECTION = 2,
  /** Frame can be referenced by a subsequent frame (or is a DC frame). */
  JXL_EXP_FALLBACK_REFERENCEABLE_FRAME = 3,
  /** Modular frame requires global full-image transforms (e.g. Squeeze or
   * Palette). */
  JXL_EXP_FALLBACK_MODULAR_GLOBAL_TRANSFORMS = 4,
  /** Slow/simple render pipeline was selected. */
  JXL_EXP_FALLBACK_SIMPLE_RENDER_PIPELINE = 5,
  /** JPEG bitstream reconstruction mode is active. */
  JXL_EXP_FALLBACK_JPEG_RECONSTRUCTION = 6,
  /** Multi-jxlp container has unparsed later box boundaries; file-offset range
   * planning cannot map all sections yet. */
  JXL_EXP_FALLBACK_MULTI_JXLP_UNKNOWN_BOUNDARIES = 7,
} JxlExperimentalRegionFallbackReason;

/**
 * Exactness / fallback policy for reduced-resolution decoding.
 */
typedef enum {
  /** Use the fastest native codec path available (VarDCT 1:8 DC, Modular
   * Squeeze partial inverse, progressive pass skip), falling back to streaming
   * float box downsampling if the bitstream has no native low-res path. */
  JXL_EXP_SCALE_FAST_APPROXIMATE = 0,
  /** Require a fast native path that avoids full-resolution decode; return
   * JXL_DEC_ERROR if no native reduced-resolution path exists for this frame. */
  JXL_EXP_SCALE_REQUIRE_FAST_NATIVE = 1,
  /** Always decode full-resolution pixels (within any active ROI) and apply
   * the defined unclamped float box/area downsample (reference-accurate). */
  JXL_EXP_SCALE_REFERENCE_RESAMPLE = 2,
} JxlExperimentalScalePolicy;

/**
 * Reports which reduced-resolution decode strategy was selected for the frame.
 */
typedef enum {
  /** Full resolution (downsampling == 1). */
  JXL_EXP_SCALE_MODE_NONE = 0,
  /** Native VarDCT 1:8 DC-only path (skips AC Global and all AC/HF groups). */
  JXL_EXP_SCALE_MODE_NATIVE_VARDCT_DC = 1,
  /** Native Modular Squeeze partial inverse (skips fine ModularAC passes). */
  JXL_EXP_SCALE_MODE_MODULAR_SQUEEZE_NATIVE = 2,
  /** Progressive VarDCT pass skipping + streaming float box downsample. */
  JXL_EXP_SCALE_MODE_PROGRESSIVE_PASS_SKIP = 3,
  /** Full-resolution group decode + low-memory streaming float box downsample. */
  JXL_EXP_SCALE_MODE_STREAMING_DOWNSAMPLE_FALLBACK = 4,
} JxlExperimentalScaleDecodeMode;

/**
 * Diagnostic and efficiency statistics for the current/most recent frame's
 * region and/or reduced-resolution decode.
 */
typedef struct {
  /** Fallback reason (JXL_EXP_FALLBACK_NONE if selective skipping occurred). */
  JxlExperimentalRegionFallbackReason fallback_reason;
  /** Total AC/HF groups in the frame. */
  size_t total_groups;
  /** Number of AC/HF groups required (and decoded) for the ROI + halo. */
  size_t required_groups;
  /** Number of AC/HF groups skipped. */
  size_t skipped_groups;
  /** Total TOC sections in the frame. */
  size_t total_sections;
  /** Number of TOC sections required for the ROI + dependencies. */
  size_t required_sections;
  /** Number of coalesced file byte ranges planned for the ROI. */
  size_t planned_range_count;
  /** Total compressed bytes across planned ranges (excluding already-read
   * headers/TOC). */
  uint64_t planned_bytes_to_read;
  /** Total compressed payload bytes in the frame's TOC. */
  uint64_t total_frame_payload_bytes;
  /** Configured output downsampling factor (1, 2, 4, or 8). */
  uint32_t downsampling;
  /** Configured reduced-resolution exactness policy. */
  JxlExperimentalScalePolicy scale_policy;
  /** Effective reduced-resolution decode strategy selected for the frame. */
  JxlExperimentalScaleDecodeMode scale_mode;
  /** Total passes in the frame. */
  size_t total_passes;
  /** Number of passes required/decoded for the configured scale target. */
  size_t required_passes;
  /** Total compressed bytes skipped zero-copy in sequential streaming mode. */
  uint64_t bytes_skipped_sequential;
  /** Total compressed bytes ingested via sparse offset-tagged input. */
  uint64_t sparse_bytes_received;
  /** Peak simultaneous bytes buffered in sparse section assembly. */
  uint64_t sparse_peak_bytes_buffered;
  /** JXL_TRUE if the frame completed via an approximate/reduced-fidelity fast
   * path (VarDCT 1:8 DC, Modular Squeeze partial inverse, or progressive pass
   * skip); JXL_FALSE for full-fidelity or exact reference resample. */
  JXL_BOOL is_partial_fidelity;
} JxlExperimentalRegionDecodeStats;

/**
 * Sets power-of-two output downsampling factor (1, 2, 4, or 8) and exactness
 * policy for subsequent frames.
 *
 * When factor > 1:
 * - Effective output canvas dimensions become
 *   DivCeil(oriented_w, factor) x DivCeil(oriented_h, factor).
 * - @ref JxlDecoderGetBasicInfo continues to report original full-image
 *   dimensions for backward compatibility.
 * - @ref JxlDecoderImageOutBufferSize and @ref JxlDecoderExtraChannelBufferSize
 *   return buffer sizes for the effective downsampled dimensions (or active
 *   output region in downsampled coordinates).
 * - If combined with @ref JxlDecoderExperimentalSetOutputRegion, region
 *   coordinates (x, y, width, height) are specified in downsampled output
 *   coordinates [0, DivCeil(W, factor)) x [0, DivCeil(H, factor)).
 *
 * Must be called before setting the image output buffer for the current frame.
 * Passing factor == 1 restores full-resolution decoding.
 *
 * @param dec decoder object
 * @param factor power-of-two scale factor: 1, 2, 4, or 8
 * @param policy exactness / fallback policy
 * @return ::JXL_DEC_SUCCESS on success, ::JXL_DEC_ERROR on invalid parameters
 *     or state.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderExperimentalSetOutputDownsampling(
    JxlDecoder* dec, uint32_t factor, JxlExperimentalScalePolicy policy);

/**
 * Queries effective output dimensions (accounting for orientation,
 * downsampling, and any active output region) and whether a fast native
 * reduced-resolution decode path is available for the current frame.
 *
 * `width` and `height` are available after ::JXL_DEC_BASIC_INFO.
 * `fast_native_available` and `scale_mode` reflect the current frame once its
 * header and TOC have been parsed (and are refined after DC Global for Modular
 * frames). Any output pointer may be NULL if not needed.
 *
 * @param dec decoder object
 * @param width [out] effective output width in pixels (may be NULL)
 * @param height [out] effective output height in pixels (may be NULL)
 * @param fast_native_available [out] JXL_TRUE if a native fast path that avoids
 *     full-resolution decode is available (may be NULL)
 * @param scale_mode [out] selected reduced-resolution decode mode (may be NULL)
 * @return ::JXL_DEC_SUCCESS on success, ::JXL_DEC_ERROR if basic info is not
 *     yet available or dec is NULL.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderExperimentalGetScaledOutputInfo(
    const JxlDecoder* dec, uint32_t* width, uint32_t* height,
    JXL_BOOL* fast_native_available,
    JxlExperimentalScaleDecodeMode* scale_mode);

/**
 * Sets the requested output region of interest (ROI) for subsequent frames.
 *
 * Coordinates (x, y) and dimensions (width, height) are specified in the final
 * output coordinate space (after applying EXIF orientation unless
 * @ref JxlDecoderSetKeepOrientation was called with JXL_TRUE, and in
 * downsampled coordinates if @ref JxlDecoderExperimentalSetOutputDownsampling
 * was set to factor > 1).
 *
 * When set:
 * - @ref JxlDecoderImageOutBufferSize and @ref JxlDecoderExtraChannelBufferSize
 *   return the buffer size required for the requested region (width x height).
 * - Output buffers and image out callbacks receive pixels relative to the
 *   region origin (0..width-1, 0..height-1), matching full-frame decode cropped
 *   to [x, x+width) x [y, y+height).
 * - Internal dependency expansion (Gaborish/EPF/upsampling halo, LF/DC,
 *   references) is handled automatically; groups outside the dependency rect
 *   are skipped when the frame structure allows it.
 *
 * Passing width == 0 && height == 0 clears the output region and restores
 * full-image decoding.
 *
 * @param dec decoder object
 * @param x horizontal start coordinate in output pixels
 * @param y vertical start coordinate in output pixels
 * @param width region width in pixels
 * @param height region height in pixels
 * @return ::JXL_DEC_SUCCESS on success, ::JXL_DEC_ERROR on invalid parameters
 *     or state.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderExperimentalSetOutputRegion(
    JxlDecoder* dec, uint32_t x, uint32_t y, uint32_t width, uint32_t height);

/**
 * Queries the coalesced file byte ranges required to finish decoding the
 * current frame for the configured output region and downsampling target.
 *
 * Must be called after the frame header and TOC have been parsed (e.g. upon
 * receiving ::JXL_DEC_FRAME, ::JXL_DEC_NEED_IMAGE_OUT_BUFFER, or
 * ::JXL_DEC_NEED_MORE_INPUT while decoding the frame).
 *
 * Two-call sizing pattern:
 * - Pass `ranges == NULL` to query the required number of ranges in
 *   `*range_count`.
 * - Pass a buffer of at least `*range_count` entries to fill `ranges` and
 *   receive the actual count written in `*range_count`.
 *
 * Note: For Modular frames where DC Global has not yet been decoded, the
 * initial plan assumes group-independent Modular coding (the common fast
 * lossless case). If DC Global later reveals global transforms (e.g. Squeeze),
 * subsequent calls to this function return the remaining ranges needed for the
 * full-frame fallback, and `fallback_reason` reports
 * `JXL_EXP_FALLBACK_MODULAR_GLOBAL_TRANSFORMS`.
 *
 * @param dec decoder object
 * @param ranges output array of file byte ranges, or NULL to query count
 * @param range_count [in/out] capacity of `ranges` on input (when non-NULL),
 *     actual number of ranges on output
 * @return ::JXL_DEC_SUCCESS on success, ::JXL_DEC_ERROR if TOC is unavailable,
 *     the buffer is too small, or file offsets cannot be mapped.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderExperimentalGetRequiredInputRanges(
    const JxlDecoder* dec, JxlExperimentalInputRange* ranges,
    size_t* range_count);

/**
 * Queries the next single coalesced file byte range required to advance
 * decoding of the current frame (aligned with upstream issue #5143).
 *
 * Shares the exact same TOC/dependency backend as
 * @ref JxlDecoderExperimentalGetRequiredInputRanges without requiring the
 * caller to allocate a range array. When all required sections are already
 * buffered or decoded, sets `*length = 0` and returns ::JXL_DEC_SUCCESS.
 *
 * @param dec decoder object
 * @param offset [out] byte offset from the start of the original file/stream
 * @param length [out] number of bytes needed starting at `*offset` (0 if none)
 * @return ::JXL_DEC_SUCCESS on success, ::JXL_DEC_ERROR if TOC is unavailable,
 *     arguments are NULL, or file offsets cannot be mapped.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderExperimentalGetNextRequiredOffset(
    const JxlDecoder* dec, uint64_t* offset, uint64_t* length);

/**
 * Supplies a sparse / discontinuous input buffer tagged with its original file
 * byte offset.
 *
 * Must be called after @ref JxlDecoderReleaseInput has released any active
 * sequential input buffer (mixing @ref JxlDecoderSetInput with active sparse
 * offset input within the same frame is disallowed and returns
 * ::JXL_DEC_ERROR).
 *
 * Bytes intersecting still-needed TOC sections of the current frame are
 * retained (supporting arbitrary order, partial chunks, overlaps, and
 * duplicates); bytes outside required sections are ignored without buffering.
 * The caller retains ownership of `data` and may free or reuse it immediately
 * after this call returns.
 *
 * After supplying one or more ranges, call @ref JxlDecoderProcessInput to
 * advance decoding. Missing holes outside the required ranges will not block
 * frame completion. Calling @ref JxlDecoderCloseInput while required sparse
 * sections remain incomplete causes @ref JxlDecoderProcessInput to return
 * ::JXL_DEC_ERROR.
 *
 * @param dec decoder object
 * @param data pointer to compressed bytes starting at `offset` in the file
 * @param size number of bytes in `data`
 * @param offset byte offset from the start of the original file/stream
 * @return ::JXL_DEC_SUCCESS on success, ::JXL_DEC_ERROR on invalid state,
 *     overflow, or memory limit exceeded.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderExperimentalSetInputAtOffset(
    JxlDecoder* dec, const uint8_t* data, size_t size, uint64_t offset);

/**
 * Retrieves diagnostic and efficiency metrics for the current (or most
 * recently completed) frame's region and/or reduced-resolution decode.
 *
 * @param dec decoder object
 * @param stats output struct to populate
 * @return ::JXL_DEC_SUCCESS on success, ::JXL_DEC_ERROR on invalid arguments.
 */
JXL_EXPORT JxlDecoderStatus JxlDecoderExperimentalGetRegionDecodeStats(
    const JxlDecoder* dec, JxlExperimentalRegionDecodeStats* stats);

#ifdef __cplusplus
}
#endif

#endif  /* JXL_DECODE_ROI_EXPERIMENTAL_H_ */
