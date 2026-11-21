// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_DEC_FRAME_H_
#define LIB_JXL_DEC_FRAME_H_

#include <jxl/decode.h>
#include <jxl/types.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/common.h"  // JXL_HIGH_PRECISION
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_modular.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"

namespace jxl {

// Decodes a frame. Groups may be processed in parallel by `pool`.
// `metadata` is the metadata that applies to all frames of the codestream
// `decoded->metadata` must already be set and must match metadata.m.
// Used in the encoder to model decoder behaviour, and in tests.
Status DecodeFrame(PassesDecoderState* dec_state, ThreadPool* JXL_RESTRICT pool,
                   const uint8_t* next_in, size_t avail_in,
                   FrameHeader* frame_header, ImageBundle* decoded,
                   const CodecMetadata& metadata,
                   bool use_slow_rendering_pipeline = false);

// TODO(veluca): implement "forced drawing".
class FrameDecoder {
 public:
  // All parameters must outlive the FrameDecoder.
  FrameDecoder(PassesDecoderState* dec_state, const CodecMetadata& metadata,
               ThreadPool* pool, bool use_slow_rendering_pipeline)
      : dec_state_(dec_state),
        pool_(pool),
        frame_header_(&metadata),
        modular_frame_decoder_(dec_state_->memory_manager()),
        use_slow_rendering_pipeline_(use_slow_rendering_pipeline) {}

  void SetRenderSpotcolors(bool rsc) { render_spotcolors_ = rsc; }
  void SetCoalescing(bool c) { coalescing_ = c; }

  // Read FrameHeader and table of contents from the given BitReader.
  Status InitFrame(BitReader* JXL_RESTRICT br, ImageBundle* decoded,
                   bool is_preview);

  // Checks frame dimensions for their limits, and sets the output
  // image buffer.
  Status InitFrameOutput();

  struct SectionInfo {
    BitReader* JXL_RESTRICT br;
    // Logical index of the section, regardless of any permutation that may be
    // applied in the table of contents or of the physical position in the file.
    size_t id;
    // Index of the section in the order of the bytes inside the frame.
    size_t index;
  };

  struct TocEntry {
    size_t size;
    size_t id;
  };

  enum SectionStatus {
    // Processed correctly.
    kDone = 0,
    // Skipped because other required sections were not yet processed.
    kSkipped = 1,
    // Skipped because the section was already processed.
    kDuplicate = 2,
    // Only partially decoded: the section will need to be processed again.
    kPartial = 3,
  };

  // Processes `num` sections; each SectionInfo contains the index
  // of the section and a BitReader that only contains the data of the section.
  // `section_status` should point to `num` elements, and will be filled with
  // information about whether each section was processed or not.
  // A section is a part of the encoded file that is indexed by the TOC.
  Status ProcessSections(const SectionInfo* sections, size_t num,
                         SectionStatus* section_status);

  // Flushes all the data decoded so far to pixels.
  Status Flush();

  // Runs final operations once a frame data is decoded.
  // Must be called exactly once per frame, after all calls to ProcessSections.
  Status FinalizeFrame();

  // Returns dependencies of this frame on reference ids as a bit mask: bits 0-3
  // indicate reference frame 0-3 for patches and blending, bits 4-7 indicate DC
  // frames this frame depends on. Only returns a valid result after all calls
  // to ProcessSections are finished and before FinalizeFrame.
  int References() const;

  // Returns reference id of storage location where this frame is stored as a
  // bit flag, or 0 if not stored.
  // Matches the bit mask used for GetReferences: bits 0-3 indicate it is stored
  // for patching or blending, bits 4-7 indicate DC frame.
  // Unlike References, can be ran at any time as
  // soon as the frame header is known.
  static int SavedAs(const FrameHeader& header);

  uint64_t SumSectionSizes() const { return section_sizes_sum_; }
  const std::vector<TocEntry>& Toc() const { return toc_; }

  const FrameHeader& GetFrameHeader() const { return frame_header_; }

  struct ByteRange {
    uint64_t offset;
    uint64_t length;
  };

  enum class RegionFallbackReason {
    kNone = 0,
    kNoRegion = 1,
    kSingleSection = 2,
    kReferenceableFrame = 3,
    kModularGlobalTransforms = 4,
    kSimpleRenderPipeline = 5,
    kJpegReconstruction = 6,
  };

  enum class ScalePolicy {
    kFastApproximate = 0,
    kRequireFastNative = 1,
    kReferenceResample = 2,
  };

  enum class ScaleDecodeMode {
    kNone = 0,
    kNativeVarDCTDC = 1,
    kModularSqueezeNative = 2,
    kProgressivePassSkip = 3,
    kStreamingDownsampleFallback = 4,
  };

  struct RegionDecodeStats {
    RegionFallbackReason fallback_reason = RegionFallbackReason::kNoRegion;
    size_t total_groups = 0;
    size_t required_groups = 0;
    size_t skipped_groups = 0;
    size_t total_sections = 0;
    size_t required_sections = 0;
    uint64_t total_frame_payload_bytes = 0;
    size_t downsampling = 1;
    ScalePolicy scale_policy = ScalePolicy::kFastApproximate;
    ScaleDecodeMode scale_mode = ScaleDecodeMode::kNone;
    size_t total_passes = 0;
    size_t required_passes = 0;
  };

  // Returns whether a DC image has been decoded, accessible at low resolution
  // at passes.shared_storage.dc_storage
  bool HasDecodedDC() const { return finalized_dc_; }
  bool HasDecodedAll() const {
    if (toc_.size() == 1) return num_sections_done_ == 1;
    return finalized_dc_ && num_sections_done_ == num_required_sections_;
  }

  void SetDecodeRegion(const Rect& output_region, size_t oriented_xsize = 0,
                       size_t oriented_ysize = 0,
                       Orientation undo_orientation = Orientation::kIdentity) {
    output_region_ = output_region;
    if (oriented_xsize > 0 && oriented_ysize > 0) {
      region_oriented_xsize_ = oriented_xsize;
      region_oriented_ysize_ = oriented_ysize;
      region_undo_orientation_ = undo_orientation;
    }
    PlanRequiredGroups(/*after_dc_global=*/finalized_dc_);
  }

  void SetOutputDownsampling(size_t downsampling, ScalePolicy policy) {
    output_downsampling_ = (downsampling > 1) ? downsampling : 1;
    scale_policy_ = policy;
    PlanRequiredGroups(/*after_dc_global=*/finalized_dc_);
  }

  size_t OutputDownsampling() const { return output_downsampling_; }
  ScalePolicy GetScalePolicy() const { return scale_policy_; }
  ScaleDecodeMode GetScaleMode() const { return scale_mode_; }
  bool IsFastNativeScaleAvailable() const {
    return output_downsampling_ <= 1 ||
           scale_mode_ == ScaleDecodeMode::kNativeVarDCTDC ||
           scale_mode_ == ScaleDecodeMode::kModularSqueezeNative ||
           scale_mode_ == ScaleDecodeMode::kProgressivePassSkip;
  }

  // Returns true if section `section_id` is required under the current plan
  // (works even before finalized_dc_ for TOC range planning / sparse input).
  bool IsSectionRequiredPlanned(size_t section_id) const {
    if (toc_.size() <= 1) return true;
    if (scale_mode_ == ScaleDecodeMode::kNativeVarDCTDC ||
        (scale_mode_ == ScaleDecodeMode::kModularSqueezeNative &&
         required_passes_ == 0)) {
      return section_id <= frame_dim_.num_dc_groups;
    }
    const size_t ac_global_index = frame_dim_.num_dc_groups + 1;
    if (section_id <= ac_global_index) return true;
    const size_t ac_idx = section_id - ac_global_index - 1;
    const size_t acg = ac_idx % frame_dim_.num_groups;
    const size_t acp = ac_idx / frame_dim_.num_groups;
    if (acp >= required_passes_) return false;
    if (!can_skip_ac_groups_) return true;
    return acg < required_ac_groups_.size() && required_ac_groups_[acg] != 0;
  }

  // Conservative check for sequential streaming: never drops sections before
  // finalized_dc_ confirms Modular UsesFullImage().
  bool IsSectionRequired(size_t section_id) const {
    if (!finalized_dc_) return true;
    return IsSectionRequiredPlanned(section_id);
  }

  const std::vector<uint64_t>& SectionCodestreamOffsets() const {
    return section_codestream_offsets_;
  }

  // Returns sorted, optionally coalesced codestream-relative byte ranges (from
  // start of frame payload after TOC) for all still-unprocessed required
  // sections.
  void GetRequiredCodestreamRanges(bool coalesce_adjacent,
                                   std::vector<ByteRange>* ranges) const;

  RegionDecodeStats GetRegionDecodeStats() const;

  size_t NumCompletePasses() const {
    if (required_passes_ == 0) return 0;
    if (!can_skip_ac_groups_) {
      return *std::min_element(decoded_passes_per_ac_group_.begin(),
                               decoded_passes_per_ac_group_.end());
    }
    size_t min_passes = required_passes_;
    for (size_t g = 0; g < decoded_passes_per_ac_group_.size(); ++g) {
      if (required_ac_groups_[g]) {
        min_passes =
            std::min<size_t>(min_passes, decoded_passes_per_ac_group_[g]);
      }
    }
    return min_passes;
  }

  // If enabled, ProcessSections will stop and return true when the DC
  // sections have been processed, instead of starting the AC sections. This
  // will only occur if supported (that is, flushing will produce a valid
  // 1/8th*1/8th resolution image). The return value of true then does not mean
  // all sections have been processed, use HasDecodedDC and HasDecodedAll
  // to check the true finished state.
  // Returns the progressive detail that will be effective for the frame.
  JxlProgressiveDetail SetPauseAtProgressive(JxlProgressiveDetail prog_detail) {
    bool single_section =
        frame_dim_.num_groups == 1 && frame_header_.passes.num_passes == 1;
    if (frame_header_.frame_type != kSkipProgressive &&
        // If there's only one group and one pass, there is no separate section
        // for DC and the entire full resolution image is available at once.
        !single_section &&
        // If extra channels are encoded with modular without squeeze, they
        // don't support DC. If the are encoded with squeeze, DC works in theory
        // but the implementation may not yet correctly support this for Flush.
        // Therefore, can't correctly pause for a progressive step if there is
        // an extra channel (including alpha channel)
        // TODO(firsching): Check if this is still the case.
        decoded_->metadata()->extra_channel_info.empty() &&
        // DC is not guaranteed to be available in modular mode and may be a
        // black image. If squeeze is used, it may be available depending on the
        // current implementation.
        // TODO(lode): do return DC if it's known that flushing at this point
        // will produce a valid 1/8th downscaled image with modular encoding.
        frame_header_.encoding == FrameEncoding::kVarDCT) {
      progressive_detail_ = prog_detail;
    } else {
      progressive_detail_ = JxlProgressiveDetail::kFrames;
    }
    if (progressive_detail_ >= JxlProgressiveDetail::kPasses) {
      for (size_t i = 1; i < frame_header_.passes.num_passes; ++i) {
        passes_to_pause_.push_back(i);
      }
    } else if (progressive_detail_ >= JxlProgressiveDetail::kLastPasses) {
      for (size_t i = 0; i < frame_header_.passes.num_downsample; ++i) {
        passes_to_pause_.push_back(frame_header_.passes.last_pass[i] + 1);
      }
      // The format does not guarantee that these values are sorted.
      std::sort(passes_to_pause_.begin(), passes_to_pause_.end());
    }
    return progressive_detail_;
  }

  size_t NextNumPassesToPause() const {
    auto it = std::upper_bound(passes_to_pause_.begin(), passes_to_pause_.end(),
                               static_cast<int>(NumCompletePasses()));
    return (it != passes_to_pause_.end() ? *it
                                         : std::numeric_limits<size_t>::max());
  }

  // Sets the pixel callback or image buffer where the pixels will be decoded.
  //
  // @param undo_orientation: if true, indicates the frame decoder should apply
  // the exif orientation to bring the image to the intended display
  // orientation.
  Status SetImageOutput(const PixelCallback& pixel_callback, void* image_buffer,
                        size_t image_buffer_size, size_t xsize, size_t ysize,
                        JxlPixelFormat format, size_t bits_per_sample,
                        bool unpremul_alpha, bool undo_orientation) const {
    const size_t eff_ds = output_downsampling_ > 1 ? output_downsampling_ : 1;
    dec_state_->width = xsize;
    dec_state_->height = ysize;
    dec_state_->output_region = output_region_;
    dec_state_->output_downsampling = eff_ds;
    dec_state_->main_output.format = format;
    dec_state_->main_output.bits_per_sample = bits_per_sample;
    dec_state_->main_output.callback = pixel_callback;
    dec_state_->main_output.buffer = image_buffer;
    dec_state_->main_output.buffer_size = image_buffer_size;
    const size_t stride_xsize =
        (output_region_.xsize() > 0 && output_region_.ysize() > 0)
            ? output_region_.xsize()
            : DivCeil(xsize, eff_ds);
    JXL_ASSIGN_OR_RETURN(dec_state_->main_output.stride,
                         GetStride(stride_xsize, format));
    const jxl::ExtraChannelInfo* alpha =
        decoded_->metadata()->Find(jxl::ExtraChannel::kAlpha);
    if (alpha && alpha->alpha_associated && unpremul_alpha) {
      dec_state_->unpremul_alpha = true;
    }
    if (undo_orientation) {
      dec_state_->undo_orientation = decoded_->metadata()->GetOrientation();
      if (static_cast<int>(dec_state_->undo_orientation) > 4) {
        std::swap(dec_state_->width, dec_state_->height);
      }
    }
    dec_state_->extra_output.clear();
#if !JXL_HIGH_PRECISION
    if (eff_ds == 1 && dec_state_->main_output.buffer &&
        (format.data_type == JXL_TYPE_UINT8) && (format.num_channels >= 3) &&
        !dec_state_->unpremul_alpha &&
        (dec_state_->undo_orientation == Orientation::kIdentity) &&
        decoded_->metadata()->xyb_encoded &&
        dec_state_->output_encoding_info.color_encoding.IsSRGB() &&
        dec_state_->output_encoding_info.all_default_opsin &&
        (dec_state_->output_encoding_info.desired_intensity_target ==
         dec_state_->output_encoding_info.orig_intensity_target) &&
        HasFastXYBTosRGB8() && frame_header_.needs_color_transform()) {
      dec_state_->fast_xyb_srgb8_conversion = true;
    }
#endif
    return true;
  }

  Status AddExtraChannelOutput(void* buffer, size_t buffer_size, size_t xsize,
                               JxlPixelFormat format, size_t bits_per_sample) {
    const size_t eff_ds = output_downsampling_ > 1 ? output_downsampling_ : 1;
    ImageOutput out;
    out.format = format;
    out.bits_per_sample = bits_per_sample;
    out.buffer = buffer;
    out.buffer_size = buffer_size;
    const size_t stride_xsize =
        (output_region_.xsize() > 0 && output_region_.ysize() > 0)
            ? output_region_.xsize()
            : DivCeil(xsize, eff_ds);
    JXL_ASSIGN_OR_RETURN(out.stride, GetStride(stride_xsize, format));
    dec_state_->extra_output.push_back(out);
    return true;
  }

 private:
  using PassesReaders = std::array<BitReader*, kMaxNumPasses>;

  std::pair<size_t, size_t> EstimateGroupBorderFromHeader() const;
  void PlanRequiredGroups(bool after_dc_global);

  Status ProcessDCGlobal(BitReader* br);
  Status ProcessDCGroup(size_t dc_group_id, BitReader* br);
  Status FinalizeDC();
  Status AllocateOutput();
  Status ProcessACGlobal(BitReader* br);
  Status ProcessACGroup(size_t ac_group_id, PassesReaders& br,
                        size_t num_passes, size_t thread, bool force_draw,
                        bool dc_only);
  void MarkSections(const SectionInfo* sections, size_t num,
                    const SectionStatus* section_status);

  // Allocates storage for parallel decoding using up to `num_threads` threads
  // of up to `num_tasks` tasks. The value of `thread` passed to
  // `GetStorageLocation` must be smaller than the `num_threads` value passed
  // here. The value of `task` passed to `GetStorageLocation` must be smaller
  // than the value of `num_tasks` passed here.
  Status PrepareStorage(size_t num_threads, size_t num_tasks) {
    size_t storage_size = std::min(num_threads, num_tasks);
    if (storage_size > group_dec_caches_.size()) {
      group_dec_caches_.resize(storage_size);
    }
    use_task_id_ = num_threads > num_tasks;
    bool use_noise = (frame_header_.flags & FrameHeader::kNoise) != 0;
    bool use_group_ids =
        (modular_frame_decoder_.UsesFullImage() &&
         (frame_header_.encoding == FrameEncoding::kVarDCT || use_noise));
    if (dec_state_->render_pipeline) {
      JXL_RETURN_IF_ERROR(dec_state_->render_pipeline->PrepareForThreads(
          storage_size, use_group_ids));
    }
    JXL_RETURN_IF_ERROR(
        dec_state_->upsampler8x->PrepareForThreads(num_threads));
    return true;
  }

  size_t GetStorageLocation(size_t thread, size_t task) const {
    if (use_task_id_) return task;
    return thread;
  }

  static size_t BytesPerChannel(JxlDataType data_type) {
    return (data_type == JXL_TYPE_UINT8   ? 1u
            : data_type == JXL_TYPE_FLOAT ? 4u
                                          : 2u);
  }

  static StatusOr<size_t> GetStride(const size_t xsize, JxlPixelFormat format) {
    size_t x_stride;
    if (!SafeMul(BytesPerChannel(format.data_type), format.num_channels,
                 x_stride)) {
      return JXL_FAILURE("Image too large");
    }
    size_t y_stride;
    if (!SafeMul(x_stride, xsize, y_stride)) {
      return JXL_FAILURE("Image too large");
    }
    if (!SafeRoundUpTo(y_stride, format.align, y_stride)) {
      return JXL_FAILURE("Image too large");
    }
    return y_stride;
  }

  bool HasDcGroupToDecode() const {
    return std::any_of(decoded_dc_groups_.cbegin(), decoded_dc_groups_.cend(),
                       [](uint8_t ready) { return ready == 0; });
  }

  PassesDecoderState* dec_state_;
  ThreadPool* pool_;
  std::vector<TocEntry> toc_;
  uint64_t section_sizes_sum_;
  // TODO(veluca): figure out the duplication between these and dec_state_.
  FrameHeader frame_header_;
  FrameDimensions frame_dim_;
  ImageBundle* decoded_;
  ModularFrameDecoder modular_frame_decoder_;
  bool render_spotcolors_ = true;
  bool coalescing_ = true;

  std::vector<uint8_t> processed_section_;
  std::vector<uint8_t> decoded_passes_per_ac_group_;
  std::vector<uint8_t> decoded_dc_groups_;
  bool decoded_dc_global_;
  bool decoded_ac_global_;
  bool HasEverything() const;
  bool finalized_dc_ = true;
  size_t num_sections_done_ = 0;
  bool is_finalized_ = true;
  bool allocated_ = false;

  std::vector<GroupDecCache> group_dec_caches_;

  // Whether or not the task id should be used for storage indexing, instead of
  // the thread id.
  bool use_task_id_ = false;

  // Testing setting: whether or not to use the slow rendering pipeline.
  bool use_slow_rendering_pipeline_;

  JxlProgressiveDetail progressive_detail_ = kFrames;
  // Number of completed passes where section decoding should pause.
  // Used for progressive details at least kLastPasses.
  std::vector<int> passes_to_pause_;

  // Optional decode region in oriented output coordinates.
  std::vector<uint64_t> section_codestream_offsets_;
  Rect output_region_;
  size_t region_oriented_xsize_ = 0;
  size_t region_oriented_ysize_ = 0;
  Orientation region_undo_orientation_ = Orientation::kIdentity;
  bool can_skip_ac_groups_ = false;
  std::vector<uint8_t> required_ac_groups_;
  size_t num_required_groups_ = 0;
  size_t num_required_sections_ = 0;
  RegionFallbackReason fallback_reason_ = RegionFallbackReason::kNoRegion;

  size_t output_downsampling_ = 1;
  ScalePolicy scale_policy_ = ScalePolicy::kFastApproximate;
  ScaleDecodeMode scale_mode_ = ScaleDecodeMode::kNone;
  size_t required_passes_ = 0;
};

}  // namespace jxl

#endif  // LIB_JXL_DEC_FRAME_H_
