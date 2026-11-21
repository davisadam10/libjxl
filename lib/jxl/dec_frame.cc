// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/dec_frame.h"

#include <jxl/decode.h>
#include <jxl/memory_manager.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <utility>
#include <vector>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/common.h"  // kMaxNumPasses
#include "lib/jxl/compressed_dc.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_group.h"
#include "lib/jxl/dec_modular.h"
#include "lib/jxl/dec_noise.h"
#include "lib/jxl/dec_patch_dictionary.h"
#include "lib/jxl/entropy_coder.h"
#include "lib/jxl/epf.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/jpeg/jpeg_data.h"
#include "lib/jxl/loop_filter.h"
#include "lib/jxl/passes_state.h"
#include "lib/jxl/quant_weights.h"
#include "lib/jxl/quantizer.h"
#include "lib/jxl/render_pipeline/render_pipeline.h"
#include "lib/jxl/splines.h"
#include "lib/jxl/toc.h"

namespace jxl {

namespace {
Status DecodeGlobalDCInfo(BitReader* reader, bool is_jpeg,
                          PassesDecoderState* state, ThreadPool* pool) {
  JXL_RETURN_IF_ERROR(state->shared_storage.quantizer.Decode(reader));

  JXL_RETURN_IF_ERROR(DecodeBlockCtxMap(state->memory_manager(), reader,
                                        &state->shared_storage.block_ctx_map));

  JXL_RETURN_IF_ERROR(state->shared_storage.cmap.DecodeDC(reader));

  // Pre-compute info for decoding a group.
  if (is_jpeg) {
    state->shared_storage.quantizer.ClearDCMul();  // Don't dequant DC
  }

  state->shared_storage.ac_strategy.FillInvalid();
  return true;
}
}  // namespace

Status DecodeFrame(PassesDecoderState* dec_state, ThreadPool* JXL_RESTRICT pool,
                   const uint8_t* next_in, size_t avail_in,
                   FrameHeader* frame_header, ImageBundle* decoded,
                   const CodecMetadata& metadata,
                   bool use_slow_rendering_pipeline) {
  FrameDecoder frame_decoder(dec_state, metadata, pool,
                             use_slow_rendering_pipeline);

  BitReader reader(Bytes(next_in, avail_in));
  JXL_RETURN_IF_ERROR(frame_decoder.InitFrame(&reader, decoded,
                                              /*is_preview=*/false));
  JXL_RETURN_IF_ERROR(frame_decoder.InitFrameOutput());
  if (frame_header) {
    *frame_header = frame_decoder.GetFrameHeader();
  }

  JXL_RETURN_IF_ERROR(reader.AllReadsWithinBounds());
  size_t header_bytes = reader.TotalBitsConsumed() / kBitsPerByte;
  JXL_RETURN_IF_ERROR(reader.Close());

  size_t processed_bytes = header_bytes;
  Status close_ok = true;
  std::vector<std::unique_ptr<BitReader>> section_readers;
  {
    std::vector<std::unique_ptr<BitReaderScopedCloser>> section_closers;
    std::vector<FrameDecoder::SectionInfo> section_info;
    std::vector<FrameDecoder::SectionStatus> section_status;
    size_t pos = header_bytes;
    size_t index = 0;
    for (auto toc_entry : frame_decoder.Toc()) {
      JXL_RETURN_IF_ERROR(pos + toc_entry.size <= avail_in);
      auto br = make_unique<BitReader>(Bytes(next_in + pos, toc_entry.size));
      section_info.emplace_back(
          FrameDecoder::SectionInfo{br.get(), toc_entry.id, index++});
      section_closers.emplace_back(
          make_unique<BitReaderScopedCloser>(*br, close_ok));
      section_readers.emplace_back(std::move(br));
      pos += toc_entry.size;
    }
    section_status.resize(section_info.size());
    JXL_RETURN_IF_ERROR(frame_decoder.ProcessSections(
        section_info.data(), section_info.size(), section_status.data()));
    for (size_t i = 0; i < section_status.size(); i++) {
      JXL_RETURN_IF_ERROR(section_status[i] == FrameDecoder::kDone);
      processed_bytes += frame_decoder.Toc()[i].size;
    }
  }
  JXL_RETURN_IF_ERROR(close_ok);
  JXL_RETURN_IF_ERROR(frame_decoder.FinalizeFrame());
  decoded->SetDecodedBytes(processed_bytes);
  return true;
}

Status FrameDecoder::InitFrame(BitReader* JXL_RESTRICT br, ImageBundle* decoded,
                               bool is_preview) {
  decoded_ = decoded;
  JXL_ENSURE(is_finalized_);
  JxlMemoryManager* memory_manager = decoded_->memory_manager();

  // Reset the dequantization matrices to their default values.
  dec_state_->shared_storage.matrices = DequantMatrices();

  frame_header_.nonserialized_is_preview = is_preview;
  JXL_ENSURE(frame_header_.nonserialized_metadata != nullptr);
  JXL_RETURN_IF_ERROR(ReadFrameHeader(br, &frame_header_));
  frame_dim_ = frame_header_.ToFrameDimensions();
  JXL_DEBUG_V(2, "FrameHeader: %s", frame_header_.DebugString().c_str());

  const size_t num_passes = frame_header_.passes.num_passes;
  const size_t num_groups = frame_dim_.num_groups;

  // If the previous frame was not a kRegularFrame, `decoded` may have different
  // dimensions; must reset to avoid errors.
  decoded->RemoveColor();
  decoded->ClearExtraChannels();

  decoded->duration = frame_header_.animation_frame.duration;

  if (!frame_header_.nonserialized_is_preview &&
      (frame_header_.is_last || frame_header_.animation_frame.duration > 0) &&
      (frame_header_.frame_type == kRegularFrame ||
       frame_header_.frame_type == kSkipProgressive)) {
    ++dec_state_->visible_frame_index;
    dec_state_->nonvisible_frame_index = 0;
  } else {
    ++dec_state_->nonvisible_frame_index;
  }

  // Read TOC.
  const size_t toc_entries =
      NumTocEntries(num_groups, frame_dim_.num_dc_groups, num_passes);
  std::vector<uint32_t> sizes;
  std::vector<coeff_order_t> permutation;
  JXL_RETURN_IF_ERROR(
      ReadToc(memory_manager, toc_entries, br, &sizes, &permutation));
  bool have_permutation = !permutation.empty();
  toc_.resize(toc_entries);
  section_codestream_offsets_.resize(toc_entries);
  section_sizes_sum_ = 0;
  for (size_t i = 0; i < toc_entries; ++i) {
    section_codestream_offsets_[i] = section_sizes_sum_;
    toc_[i].size = sizes[i];
    size_t index = have_permutation ? permutation[i] : i;
    toc_[index].id = i;
    if (section_sizes_sum_ + toc_[i].size < section_sizes_sum_) {
      return JXL_FAILURE("group offset overflow");
    }
    section_sizes_sum_ += toc_[i].size;
  }

  if (JXL_DEBUG_V_LEVEL >= 3) {
    for (size_t i = 0; i < toc_entries; ++i) {
      JXL_DEBUG_V(3, "TOC entry %" PRIuS " size %" PRIuS " id %" PRIuS "", i,
                  toc_[i].size, toc_[i].id);
    }
  }

  JXL_ENSURE((br->TotalBitsConsumed() % kBitsPerByte) == 0);
  const size_t group_codes_begin = br->TotalBitsConsumed() / kBitsPerByte;
  JXL_ENSURE(!toc_.empty());

  // Overflow check.
  if (group_codes_begin + section_sizes_sum_ < group_codes_begin) {
    return JXL_FAILURE("Invalid group codes");
  }

  if (!frame_header_.chroma_subsampling.Is444() &&
      !(frame_header_.flags & FrameHeader::kSkipAdaptiveDCSmoothing) &&
      frame_header_.encoding == FrameEncoding::kVarDCT) {
    return JXL_FAILURE(
        "Non-444 chroma subsampling is not allowed when adaptive DC "
        "smoothing is enabled");
  }
  return true;
}

Status FrameDecoder::InitFrameOutput() {
  JXL_RETURN_IF_ERROR(
      InitializePassesSharedState(frame_header_, &dec_state_->shared_storage));
  JXL_RETURN_IF_ERROR(dec_state_->Init(frame_header_));
  modular_frame_decoder_.Init(frame_dim_);

  if (decoded_->IsJPEG()) {
    if (frame_header_.encoding == FrameEncoding::kModular) {
      return JXL_FAILURE("Cannot output JPEG from Modular");
    }
    jpeg::JPEGData* jpeg_data = decoded_->jpeg_data.get();
    size_t num_components = jpeg_data->components.size();
    if (num_components != 1 && num_components != 3) {
      return JXL_FAILURE("Invalid number of components");
    }
    if (frame_header_.nonserialized_metadata->m.xyb_encoded) {
      return JXL_FAILURE("Cannot decode to JPEG an XYB image");
    }
    auto jpeg_c_map = JpegOrder(ColorTransform::kYCbCr, num_components == 1);
    decoded_->jpeg_data->width = frame_dim_.xsize;
    decoded_->jpeg_data->height = frame_dim_.ysize;
    for (size_t c = 0; c < num_components; c++) {
      auto& component = jpeg_data->components[jpeg_c_map[c]];
      component.width_in_blocks =
          frame_dim_.xsize_blocks >> frame_header_.chroma_subsampling.HShift(c);
      component.height_in_blocks =
          frame_dim_.ysize_blocks >> frame_header_.chroma_subsampling.VShift(c);
      component.h_samp_factor =
          1 << frame_header_.chroma_subsampling.RawHShift(c);
      component.v_samp_factor =
          1 << frame_header_.chroma_subsampling.RawVShift(c);
      component.coeffs.resize(component.width_in_blocks *
                              component.height_in_blocks * jxl::kDCTBlockSize);
    }
  }

  // Clear the state.
  decoded_dc_global_ = false;
  decoded_ac_global_ = false;
  is_finalized_ = false;
  finalized_dc_ = false;
  num_sections_done_ = 0;
  decoded_dc_groups_.clear();
  decoded_dc_groups_.resize(frame_dim_.num_dc_groups);
  decoded_passes_per_ac_group_.clear();
  decoded_passes_per_ac_group_.resize(frame_dim_.num_groups, 0);
  processed_section_.clear();
  processed_section_.resize(toc_.size());
  allocated_ = false;
  can_skip_ac_groups_ = false;
  required_ac_groups_.assign(frame_dim_.num_groups, 1);
  num_required_groups_ = frame_dim_.num_groups;
  num_required_sections_ = toc_.size();
  fallback_reason_ = RegionFallbackReason::kNoRegion;
  scale_mode_ = ScaleDecodeMode::kNone;
  required_passes_ = frame_header_.passes.num_passes;
  return true;
}

Status FrameDecoder::ProcessDCGlobal(BitReader* br) {
  PassesSharedState& shared = dec_state_->shared_storage;
  JxlMemoryManager* memory_manager = shared.memory_manager;
  if (frame_header_.flags & FrameHeader::kPatches) {
    bool uses_extra_channels = false;
    JXL_RETURN_IF_ERROR(shared.image_features.patches.Decode(
        memory_manager, br, frame_dim_.xsize_padded, frame_dim_.ysize_padded,
        shared.metadata->m.num_extra_channels, &uses_extra_channels));
    if (uses_extra_channels && frame_header_.upsampling != 1) {
      for (size_t ecups : frame_header_.extra_channel_upsampling) {
        if (ecups != frame_header_.upsampling) {
          return JXL_FAILURE(
              "Cannot use extra channels in patches if color channels are "
              "subsampled differently from extra channels");
        }
      }
    }
  } else {
    shared.image_features.patches.Clear();
  }
  shared.image_features.splines.Clear();
  if (frame_header_.flags & FrameHeader::kSplines) {
    JXL_RETURN_IF_ERROR(shared.image_features.splines.Decode(
        memory_manager, br, frame_dim_.xsize * frame_dim_.ysize));
  }
  if (frame_header_.flags & FrameHeader::kNoise) {
    JXL_RETURN_IF_ERROR(DecodeNoise(br, &shared.image_features.noise_params));
  }
  JXL_RETURN_IF_ERROR(dec_state_->shared_storage.matrices.DecodeDC(br));

  if (frame_header_.encoding == FrameEncoding::kVarDCT) {
    JXL_RETURN_IF_ERROR(
        jxl::DecodeGlobalDCInfo(br, decoded_->IsJPEG(), dec_state_, pool_));
  }
  // Splines' draw cache uses the color correlation map.
  if (frame_header_.flags & FrameHeader::kSplines) {
    JXL_RETURN_IF_ERROR(shared.image_features.splines.InitializeDrawCache(
        frame_dim_.xsize_upsampled, frame_dim_.ysize_upsampled,
        dec_state_->shared->cmap.base()));
  }
  Status dec_status = modular_frame_decoder_.DecodeGlobalInfo(
      br, frame_header_, /*allow_truncated_group=*/false);
  if (dec_status.IsFatalError()) return dec_status;
  if (dec_status) {
    decoded_dc_global_ = true;
  }
  return dec_status;
}

Status FrameDecoder::ProcessDCGroup(size_t dc_group_id, BitReader* br) {
  const size_t gx = dc_group_id % frame_dim_.xsize_dc_groups;
  const size_t gy = dc_group_id / frame_dim_.xsize_dc_groups;
  const LoopFilter& lf = frame_header_.loop_filter;
  if (frame_header_.encoding == FrameEncoding::kVarDCT &&
      !(frame_header_.flags & FrameHeader::kUseDcFrame)) {
    JXL_RETURN_IF_ERROR(modular_frame_decoder_.DecodeVarDCTDC(
        frame_header_, dc_group_id, br, dec_state_));
  }
  const Rect mrect(gx * frame_dim_.dc_group_dim, gy * frame_dim_.dc_group_dim,
                   frame_dim_.dc_group_dim, frame_dim_.dc_group_dim);
  JXL_RETURN_IF_ERROR(modular_frame_decoder_.DecodeGroup(
      frame_header_, mrect, br, 3, 1000,
      ModularStreamId::ModularDC(dc_group_id),
      /*zerofill=*/false, nullptr, nullptr,
      /*allow_truncated=*/false));
  if (frame_header_.encoding == FrameEncoding::kVarDCT) {
    JXL_RETURN_IF_ERROR(modular_frame_decoder_.DecodeAcMetadata(
        frame_header_, dc_group_id, br, dec_state_));
  } else if (lf.epf_iters > 0) {
    FillImage(kInvSigmaNum / lf.epf_sigma_for_modular, &dec_state_->sigma);
  }
  decoded_dc_groups_[dc_group_id] = JXL_TRUE;
  return true;
}

Status FrameDecoder::FinalizeDC() {
  // Do Adaptive DC smoothing if enabled. This *must* happen between all the
  // ProcessDCGroup and ProcessACGroup.
  JxlMemoryManager* memory_manager = dec_state_->memory_manager();
  if (frame_header_.encoding == FrameEncoding::kVarDCT &&
      !(frame_header_.flags & FrameHeader::kSkipAdaptiveDCSmoothing) &&
      !(frame_header_.flags & FrameHeader::kUseDcFrame)) {
    JXL_RETURN_IF_ERROR(AdaptiveDCSmoothing(
        memory_manager, dec_state_->shared->quantizer.MulDC(),
        &dec_state_->shared_storage.dc_storage, pool_));
  }

  finalized_dc_ = true;
  return true;
}

Status FrameDecoder::AllocateOutput() {
  if (allocated_) return true;
  modular_frame_decoder_.MaybeDropFullImage();
  decoded_->origin = frame_header_.frame_origin;
  JXL_RETURN_IF_ERROR(
      dec_state_->InitForAC(frame_header_.passes.num_passes, nullptr));
  allocated_ = true;
  return true;
}

Status FrameDecoder::ProcessACGlobal(BitReader* br) {
  JXL_ENSURE(finalized_dc_);
  JxlMemoryManager* memory_manager = dec_state_->memory_manager();

  // Decode AC group.
  if (frame_header_.encoding == FrameEncoding::kVarDCT) {
    JXL_RETURN_IF_ERROR(dec_state_->shared_storage.matrices.Decode(
        memory_manager, br, &modular_frame_decoder_));
    JXL_RETURN_IF_ERROR(dec_state_->shared_storage.matrices.EnsureComputed(
        memory_manager, dec_state_->used_acs));

    size_t num_histo_bits =
        CeilLog2Nonzero(dec_state_->shared->frame_dim.num_groups);
    dec_state_->shared_storage.num_histograms =
        1 + br->ReadBits(num_histo_bits);

    JXL_DEBUG_V(3,
                "Processing AC global with %d passes and %" PRIuS
                " sets of histograms",
                frame_header_.passes.num_passes,
                dec_state_->shared_storage.num_histograms);

    dec_state_->code.resize(kMaxNumPasses);
    dec_state_->context_map.resize(kMaxNumPasses);
    // Read coefficient orders and histograms.
    size_t max_num_bits_ac = 0;
    for (size_t i = 0; i < frame_header_.passes.num_passes; i++) {
      uint16_t used_orders = U32Coder::Read(kOrderEnc, br);
      JXL_RETURN_IF_ERROR(DecodeCoeffOrders(
          memory_manager, used_orders, dec_state_->used_acs,
          &dec_state_->shared_storage
               .coeff_orders[i * dec_state_->shared_storage.coeff_order_size],
          br));
      size_t num_contexts =
          dec_state_->shared->num_histograms *
          dec_state_->shared_storage.block_ctx_map.NumACContexts();
      JXL_RETURN_IF_ERROR(DecodeHistograms(memory_manager, br, num_contexts,
                                           &dec_state_->code[i],
                                           &dec_state_->context_map[i]));
      // Add extra values to enable the cheat in hot loop of DecodeACVarBlock.
      dec_state_->context_map[i].resize(
          num_contexts + kZeroDensityContextLimit - kZeroDensityContextCount);
      max_num_bits_ac =
          std::max(max_num_bits_ac, dec_state_->code[i].max_num_bits);
    }
    max_num_bits_ac += CeilLog2Nonzero(frame_header_.passes.num_passes);
    // 16-bit buffer for decoding to JPEG are not implemented.
    // TODO(veluca): figure out the exact limit - 16 should still work with
    // 16-bit buffers, but we are excluding it for safety.
    bool use_16_bit = max_num_bits_ac < 16 && !decoded_->IsJPEG();
    bool store = frame_header_.passes.num_passes > 1;
    size_t xs = store ? kGroupDim * kGroupDim : 0;
    size_t ys = store ? frame_dim_.num_groups : 0;
    if (use_16_bit) {
      JXL_ASSIGN_OR_RETURN(dec_state_->coefficients,
                           ACImageT<int16_t>::Make(memory_manager, xs, ys));
    } else {
      JXL_ASSIGN_OR_RETURN(dec_state_->coefficients,
                           ACImageT<int32_t>::Make(memory_manager, xs, ys));
    }
    if (store) {
      dec_state_->coefficients->ZeroFill();
    }
  }

  // Set JPEG decoding data.
  if (decoded_->IsJPEG()) {
    decoded_->color_transform = frame_header_.color_transform;
    decoded_->chroma_subsampling = frame_header_.chroma_subsampling;
    const std::vector<QuantEncoding>& qe =
        dec_state_->shared_storage.matrices.encodings();
    if (qe.empty() || qe[0].mode != QuantEncoding::Mode::kQuantModeRAW ||
        std::abs(qe[0].qraw.qtable_den - 1.f / (8 * 255)) > 1e-8f) {
      return JXL_FAILURE(
          "Quantization table is not a JPEG quantization table.");
    }
    jpeg::JPEGData* jpeg_data = decoded_->jpeg_data.get();
    size_t num_components = jpeg_data->components.size();
    bool is_gray = (num_components == 1);
    JXL_ENSURE(frame_header_.color_transform != ColorTransform::kXYB);
    auto jpeg_c_map = JpegOrder(frame_header_.color_transform, is_gray);
    size_t qt_set = 0;
    JXL_ENSURE(num_components <= 3);
    JXL_ENSURE(qe[0].qraw.qtable->size() == 3 * 8 * 8);
    int* qtable = qe[0].qraw.qtable->data();
    for (size_t c = 0; c < num_components; c++) {
      // TODO(eustas): why 1-st quant table for gray?
      size_t quant_c = is_gray ? 1 : c;
      size_t qpos = jpeg_data->components[jpeg_c_map[c]].quant_idx;
      JXL_ENSURE(qpos != jpeg_data->quant.size());
      qt_set |= 1 << qpos;
      for (size_t x = 0; x < 8; x++) {
        for (size_t y = 0; y < 8; y++) {
          jpeg_data->quant[qpos].values[x * 8 + y] =
              qtable[quant_c * 64 + y * 8 + x];
        }
      }
    }
    for (size_t i = 0; i < jpeg_data->quant.size(); i++) {
      if (qt_set & (1 << i)) continue;
      if (i == 0) return JXL_FAILURE("First quant table unused.");
      // Unused quant table is set to copy of previous quant table
      for (size_t j = 0; j < 64; j++) {
        jpeg_data->quant[i].values[j] = jpeg_data->quant[i - 1].values[j];
      }
    }
  }
  decoded_ac_global_ = true;
  return true;
}

Status FrameDecoder::ProcessACGroup(size_t ac_group_id,
                                    BitReader* JXL_RESTRICT* br,
                                    size_t num_passes, size_t thread,
                                    bool force_draw, bool dc_only) {
  size_t group_dim = frame_dim_.group_dim;
  const size_t gx = ac_group_id % frame_dim_.xsize_groups;
  const size_t gy = ac_group_id / frame_dim_.xsize_groups;
  const size_t x = gx * group_dim;
  const size_t y = gy * group_dim;
  JxlMemoryManager* memory_manager = dec_state_->memory_manager();
  JXL_DEBUG_V(3,
              "Processing AC group %" PRIuS "(%" PRIuS ",%" PRIuS
              ") group_dim: %" PRIuS " decoded passes: %u new passes: %" PRIuS,
              ac_group_id, gx, gy, group_dim,
              decoded_passes_per_ac_group_[ac_group_id], num_passes);

  const bool skip_pipeline_input =
      (scale_mode_ == ScaleDecodeMode::kModularSqueezeNative);
  RenderPipelineInput render_pipeline_input;
  if (!skip_pipeline_input) {
    render_pipeline_input =
        dec_state_->render_pipeline->GetInputBuffers(ac_group_id, thread);
  }

  bool should_run_pipeline = true;

  if (frame_header_.encoding == FrameEncoding::kVarDCT) {
    JXL_RETURN_IF_ERROR(group_dec_caches_[thread].InitOnce(
        memory_manager, frame_header_.passes.num_passes, dec_state_->used_acs));
    JXL_RETURN_IF_ERROR(DecodeGroup(
        frame_header_, br, num_passes, ac_group_id, dec_state_,
        &group_dec_caches_[thread], thread, render_pipeline_input,
        decoded_->jpeg_data.get(), decoded_passes_per_ac_group_[ac_group_id],
        force_draw, dc_only, &should_run_pipeline));
  }

  // don't limit to image dimensions here (is done in DecodeGroup)
  const Rect mrect(x, y, group_dim, group_dim);
  bool modular_ready = false;
  size_t pass0 = decoded_passes_per_ac_group_[ac_group_id];
  size_t pass1 =
      force_draw ? frame_header_.passes.num_passes : pass0 + num_passes;
  RenderPipelineInput* rpi_ptr =
      skip_pipeline_input ? nullptr : &render_pipeline_input;
  for (size_t i = pass0; i < pass1; ++i) {
    int minShift;
    int maxShift;
    frame_header_.passes.GetDownsamplingBracket(i, minShift, maxShift);
    bool modular_pass_ready = true;
    JXL_DEBUG_V(2, "Decoding modular in group %d pass %d",
                static_cast<int>(ac_group_id), static_cast<int>(i));
    if (i < pass0 + num_passes) {
      JXL_DEBUG_V(2, "Bit reader position: %" PRIuS " / %" PRIuS,
                  br[i - pass0]->TotalBitsConsumed(),
                  br[i - pass0]->TotalBytes() * kBitsPerByte);
      JXL_RETURN_IF_ERROR(modular_frame_decoder_.DecodeGroup(
          frame_header_, mrect, br[i - pass0], minShift, maxShift,
          ModularStreamId::ModularAC(ac_group_id, i),
          /*zerofill=*/false, dec_state_, rpi_ptr,
          /*allow_truncated=*/false, &modular_pass_ready));
    } else {
      JXL_RETURN_IF_ERROR(modular_frame_decoder_.DecodeGroup(
          frame_header_, mrect, nullptr, minShift, maxShift,
          ModularStreamId::ModularAC(ac_group_id, i), /*zerofill=*/true,
          dec_state_, rpi_ptr,
          /*allow_truncated=*/false, &modular_pass_ready));
    }
    if (modular_pass_ready) modular_ready = true;
  }
  decoded_passes_per_ac_group_[ac_group_id] += num_passes;

  if ((frame_header_.flags & FrameHeader::kNoise) != 0) {
    PrepareNoiseInput(*dec_state_, frame_dim_, frame_header_, ac_group_id,
                      thread);
  }

  if (!skip_pipeline_input && !modular_frame_decoder_.UsesFullImage() &&
      !decoded_->IsJPEG()) {
    if (should_run_pipeline && modular_ready) {
      JXL_RETURN_IF_ERROR(render_pipeline_input.Done());
    } else if (force_draw) {
      return JXL_FAILURE("Modular group decoding failed.");
    }
  }
  return true;
}

void FrameDecoder::MarkSections(const SectionInfo* sections, size_t num,
                                const SectionStatus* section_status) {
  num_sections_done_ += num;
  for (size_t i = 0; i < num; i++) {
    if (section_status[i] != SectionStatus::kDone) {
      processed_section_[sections[i].id] = JXL_FALSE;
      num_sections_done_--;
    }
  }
}

Status FrameDecoder::ProcessSections(const SectionInfo* sections, size_t num,
                                     SectionStatus* section_status) {
  if (num == 0) return true;  // Nothing to process
  std::fill(section_status, section_status + num, SectionStatus::kSkipped);
  size_t dc_global_sec = num;
  size_t ac_global_sec = num;
  std::vector<size_t> dc_group_sec(frame_dim_.num_dc_groups, num);
  std::vector<std::vector<size_t>> ac_group_sec(
      frame_dim_.num_groups,
      std::vector<size_t>(frame_header_.passes.num_passes, num));
  // This keeps track of the number of ac passes we want to process during this
  // call of ProcessSections.
  std::vector<size_t> desired_num_ac_passes(frame_dim_.num_groups);
  bool single_section =
      frame_dim_.num_groups == 1 && frame_header_.passes.num_passes == 1;
  if (single_section) {
    JXL_ENSURE(num == 1);
    JXL_ENSURE(sections[0].id == 0);
    if (processed_section_[0] == JXL_FALSE) {
      processed_section_[0] = JXL_TRUE;
      ac_group_sec[0].resize(1);
      dc_global_sec = ac_global_sec = dc_group_sec[0] = ac_group_sec[0][0] = 0;
      desired_num_ac_passes[0] = 1;
    } else {
      section_status[0] = SectionStatus::kDuplicate;
    }
  } else {
    size_t ac_global_index = frame_dim_.num_dc_groups + 1;
    for (size_t i = 0; i < num; i++) {
      JXL_ENSURE(sections[i].id < processed_section_.size());
      if (processed_section_[sections[i].id]) {
        section_status[i] = SectionStatus::kDuplicate;
        continue;
      }
      if (sections[i].id == 0) {
        dc_global_sec = i;
      } else if (sections[i].id < ac_global_index) {
        dc_group_sec[sections[i].id - 1] = i;
      } else if (sections[i].id == ac_global_index) {
        ac_global_sec = i;
      } else {
        size_t ac_idx = sections[i].id - ac_global_index - 1;
        size_t acg = ac_idx % frame_dim_.num_groups;
        size_t acp = ac_idx / frame_dim_.num_groups;
        if (acp >= frame_header_.passes.num_passes) {
          return JXL_FAILURE("Invalid section ID");
        }
        ac_group_sec[acg][acp] = i;
      }
      processed_section_[sections[i].id] = JXL_TRUE;
    }
    // Count number of new passes per group.
    for (size_t g = 0; g < ac_group_sec.size(); g++) {
      size_t j = 0;
      for (; j + decoded_passes_per_ac_group_[g] <
             frame_header_.passes.num_passes;
           j++) {
        if (ac_group_sec[g][j + decoded_passes_per_ac_group_[g]] == num) {
          break;
        }
      }
      desired_num_ac_passes[g] = j;
    }
  }
  if (dc_global_sec != num) {
    Status dc_global_status = ProcessDCGlobal(sections[dc_global_sec].br);
    if (dc_global_status.IsFatalError()) return dc_global_status;
    if (dc_global_status) {
      section_status[dc_global_sec] = SectionStatus::kDone;
    } else {
      section_status[dc_global_sec] = SectionStatus::kPartial;
    }
  }

  if (decoded_dc_global_) {
    const auto process_section = [this, &dc_group_sec, &num, &sections,
                                  &section_status](size_t i,
                                                   size_t thread) -> Status {
      if (dc_group_sec[i] != num) {
        JXL_RETURN_IF_ERROR(ProcessDCGroup(i, sections[dc_group_sec[i]].br));
        section_status[dc_group_sec[i]] = SectionStatus::kDone;
      }
      return true;
    };
    JXL_RETURN_IF_ERROR(RunOnPool(pool_, 0, dc_group_sec.size(),
                                  ThreadPool::NoInit, process_section,
                                  "DecodeDCGroup"));
  }

  if (!HasDcGroupToDecode() && !finalized_dc_) {
    PlanRequiredGroups(/*after_dc_global=*/true);
    if (scale_policy_ == ScalePolicy::kRequireFastNative &&
        output_downsampling_ > 1 && !IsFastNativeScaleAvailable()) {
      return JXL_FAILURE(
          "Fast native downscaling is not available for this frame");
    }
    dec_state_->output_downsampling = output_downsampling_;
    dec_state_->native_scaled_pipeline =
        (scale_mode_ == ScaleDecodeMode::kNativeVarDCTDC ||
         scale_mode_ == ScaleDecodeMode::kModularSqueezeNative);

    PassesDecoderState::PipelineOptions pipeline_options;
    pipeline_options.use_slow_render_pipeline = use_slow_rendering_pipeline_;
    pipeline_options.coalescing = coalescing_;
    pipeline_options.render_spotcolors = render_spotcolors_;
    pipeline_options.render_noise = true;
    JXL_RETURN_IF_ERROR(dec_state_->PreparePipeline(
        frame_header_, &frame_header_.nonserialized_metadata->m, decoded_,
        pipeline_options));
    JXL_RETURN_IF_ERROR(FinalizeDC());
    JXL_RETURN_IF_ERROR(AllocateOutput());
    PlanRequiredGroups(/*after_dc_global=*/true);
    if (progressive_detail_ >= JxlProgressiveDetail::kDC) {
      MarkSections(sections, num, section_status);
      return true;
    }
  }

  const bool skip_all_ac =
      finalized_dc_ &&
      (scale_mode_ == ScaleDecodeMode::kNativeVarDCTDC ||
       (scale_mode_ == ScaleDecodeMode::kModularSqueezeNative &&
        required_passes_ == 0));

  if (!skip_all_ac && finalized_dc_ && ac_global_sec != num &&
      !decoded_ac_global_) {
    JXL_RETURN_IF_ERROR(ProcessACGlobal(sections[ac_global_sec].br));
    section_status[ac_global_sec] = SectionStatus::kDone;
  }

  if (skip_all_ac) {
    std::fill(desired_num_ac_passes.begin(), desired_num_ac_passes.end(), 0);
  } else {
    if (can_skip_ac_groups_) {
      for (size_t i = 0; i < ac_group_sec.size(); i++) {
        if (!required_ac_groups_[i]) {
          desired_num_ac_passes[i] = 0;
        }
      }
    }
    if (required_passes_ < frame_header_.passes.num_passes) {
      for (size_t i = 0; i < ac_group_sec.size(); i++) {
        size_t rem = (decoded_passes_per_ac_group_[i] < required_passes_)
                         ? (required_passes_ - decoded_passes_per_ac_group_[i])
                         : 0;
        desired_num_ac_passes[i] = std::min(desired_num_ac_passes[i], rem);
      }
    }
  }

  if (progressive_detail_ >= JxlProgressiveDetail::kLastPasses) {
    // Mark that we only want the next progression pass.
    size_t target_complete_passes = NextNumPassesToPause();
    for (size_t i = 0; i < ac_group_sec.size(); i++) {
      desired_num_ac_passes[i] =
          std::min(desired_num_ac_passes[i],
                   target_complete_passes - decoded_passes_per_ac_group_[i]);
    }
  }

  if (!skip_all_ac && decoded_ac_global_) {
    // Mark all the AC groups that we received as not complete yet.
    if (scale_mode_ != ScaleDecodeMode::kModularSqueezeNative) {
      for (size_t i = 0; i < ac_group_sec.size(); i++) {
        if (desired_num_ac_passes[i] != 0) {
          dec_state_->render_pipeline->ClearDone(i);
        }
      }
    }

    const auto prepare_storage = [this](size_t num_threads) -> Status {
      JXL_RETURN_IF_ERROR(
          PrepareStorage(num_threads, decoded_passes_per_ac_group_.size()));
      return true;
    };
    const auto process_group = [this, &ac_group_sec, &desired_num_ac_passes,
                                &num, &sections, &section_status](
                                   size_t g, size_t thread) -> Status {
      if (desired_num_ac_passes[g] == 0) {
        // no new AC pass, nothing to do
        return true;
      }
      (void)num;
      size_t first_pass = decoded_passes_per_ac_group_[g];
      BitReader* JXL_RESTRICT readers[kMaxNumPasses];
      for (size_t i = 0; i < desired_num_ac_passes[g]; i++) {
        JXL_ENSURE(ac_group_sec[g][first_pass + i] != num);
        readers[i] = sections[ac_group_sec[g][first_pass + i]].br;
      }
      const bool force_draw =
          (scale_mode_ == ScaleDecodeMode::kProgressivePassSkip &&
           required_passes_ < frame_header_.passes.num_passes &&
           first_pass + desired_num_ac_passes[g] >= required_passes_);
      JXL_RETURN_IF_ERROR(ProcessACGroup(
          g, readers, desired_num_ac_passes[g], GetStorageLocation(thread, g),
          force_draw, /*dc_only=*/false));
      for (size_t i = 0; i < desired_num_ac_passes[g]; i++) {
        section_status[ac_group_sec[g][first_pass + i]] = SectionStatus::kDone;
      }
      return true;
    };
    JXL_RETURN_IF_ERROR(RunOnPool(pool_, 0, ac_group_sec.size(),
                                  prepare_storage, process_group,
                                  "DecodeGroup"));
  }

  MarkSections(sections, num, section_status);
  return true;
}

std::pair<size_t, size_t> FrameDecoder::EstimateGroupBorderFromHeader() const {
  const size_t base_shift = CeilLog2Nonzero(frame_header_.upsampling);
  size_t pad_x = 0;
  size_t pad_y = 0;
  if (frame_header_.upsampling != 1) {
    pad_x = 2;
    pad_y = 2;
  }
  const LoopFilter& lf = frame_header_.loop_filter;
  if (lf.epf_iters >= 2) {
    pad_x += 1;
    pad_y += 1;
  }
  if (lf.epf_iters >= 1) {
    pad_x += 2;
    pad_y += 2;
  }
  if (lf.epf_iters >= 3) {
    pad_x += 3;
    pad_y += 3;
  }
  if (lf.gab) {
    pad_x += 1;
    pad_y += 1;
  }
  if (!frame_header_.chroma_subsampling.Is444()) {
    const size_t max_hs = frame_header_.chroma_subsampling.MaxHShift();
    const size_t max_vs = frame_header_.chroma_subsampling.MaxVShift();
    if (max_hs > 0) {
      size_t cx = pad_x;
      for (size_t s = 0; s < max_hs; ++s) {
        cx = DivCeil(cx, size_t{2}) + 1;
      }
      pad_x = RoundUpTo(std::max(pad_x, cx << max_hs), size_t{1} << max_hs);
    }
    if (max_vs > 0) {
      size_t cy = pad_y;
      for (size_t s = 0; s < max_vs; ++s) {
        cy = DivCeil(cy, size_t{2}) + 1;
      }
      pad_y = RoundUpTo(std::max(pad_y, cy << max_vs), size_t{1} << max_vs);
    }
  }
  if ((frame_header_.flags & FrameHeader::kNoise) != 0) {
    const size_t noise_pad = DivCeil(size_t{2}, size_t{1} << base_shift);
    pad_x = std::max(pad_x, noise_pad);
    pad_y = std::max(pad_y, noise_pad);
  }
  for (size_t ec_ups : frame_header_.extra_channel_upsampling) {
    if (ec_ups != 1) {
      const size_t ec_shift = CeilLog2Nonzero(ec_ups);
      const size_t ec_pad =
          DivCeil(size_t{2} << ec_shift, size_t{1} << base_shift);
      pad_x = std::max(pad_x, ec_pad);
      pad_y = std::max(pad_y, ec_pad);
    }
  }
  if (pad_x > 0) {
    // Match LowMemoryRenderPipeline::kGroupXAlign (16 on x86, 4 on ARM).
    pad_x = RoundUpTo(pad_x, size_t{16});
  }
  return {pad_x, pad_y};
}

void FrameDecoder::PlanRequiredGroups(bool after_dc_global) {
  can_skip_ac_groups_ = false;
  required_ac_groups_.assign(frame_dim_.num_groups, 1);
  num_required_groups_ = frame_dim_.num_groups;
  required_passes_ = frame_header_.passes.num_passes;
  scale_mode_ = ScaleDecodeMode::kNone;
  num_required_sections_ = toc_.size();

  bool has_blending = frame_header_.blending_info.mode != BlendMode::kReplace ||
                      frame_header_.custom_size_or_origin;
  for (const auto& blending_info_ec :
       frame_header_.extra_channel_blending_info) {
    if (blending_info_ec.mode != BlendMode::kReplace) has_blending = true;
  }

  const bool basic_eligible =
      !frame_header_.CanBeReferenced() && frame_header_.dc_level == 0 &&
      !use_slow_rendering_pipeline_ && !(decoded_ && decoded_->IsJPEG());

  if (output_downsampling_ > 1) {
    if (scale_policy_ == ScalePolicy::kReferenceResample || !basic_eligible) {
      scale_mode_ = ScaleDecodeMode::kStreamingDownsampleFallback;
    } else {
      // Path A: Native VarDCT 1:8 DC.
      const bool can_native_vardct_dc =
          output_downsampling_ == 8 &&
          frame_header_.encoding == FrameEncoding::kVarDCT &&
          frame_header_.upsampling == 1 &&
          frame_header_.chroma_subsampling.Is444() &&
          !(frame_header_.flags &
            (FrameHeader::kPatches | FrameHeader::kSplines |
             FrameHeader::kNoise | FrameHeader::kUseDcFrame)) &&
          !has_blending &&
          frame_header_.nonserialized_metadata->m.num_extra_channels == 0;

      // Path B: Native Modular Squeeze partial inverse.
      bool ec_compatible = true;
      for (size_t ec = 0; ec < frame_header_.extra_channel_upsampling.size();
           ++ec) {
        if (frame_header_.extra_channel_upsampling[ec] != 1) {
          ec_compatible = false;
        }
      }
      for (const auto& eci :
           frame_header_.nonserialized_metadata->m.extra_channel_info) {
        if (eci.dim_shift != 0) ec_compatible = false;
      }
      const bool can_modular_squeeze_header =
          frame_header_.encoding == FrameEncoding::kModular &&
          frame_header_.upsampling == 1 && ec_compatible &&
          !(frame_header_.flags & (FrameHeader::kPatches |
                                   FrameHeader::kSplines | FrameHeader::kNoise)) &&
          !has_blending;
      const bool can_modular_squeeze =
          can_modular_squeeze_header &&
          (after_dc_global
               ? modular_frame_decoder_.HasSqueezeForDownsampling(
                     output_downsampling_)
               : (frame_header_.passes.num_passes > 1 &&
                  frame_header_.passes.num_downsample > 0));

      // Path C: Progressive VarDCT pass skipping.
      size_t prog_required_passes = frame_header_.passes.num_passes;
      if (frame_header_.encoding == FrameEncoding::kVarDCT &&
          frame_header_.passes.num_passes > 1) {
        for (size_t k = 1; k < frame_header_.passes.num_passes; ++k) {
          if (frame_header_.passes.GetDownsamplingTargetForCompletedPasses(k) <=
              output_downsampling_) {
            prog_required_passes = k;
            break;
          }
        }
      }

      if (can_native_vardct_dc) {
        scale_mode_ = ScaleDecodeMode::kNativeVarDCTDC;
        required_passes_ = 0;
      } else if (can_modular_squeeze) {
        scale_mode_ = ScaleDecodeMode::kModularSqueezeNative;
        const int target_shift =
            static_cast<int>(CeilLog2Nonzero(output_downsampling_));
        size_t req_p = 0;
        for (size_t p = 0; p < frame_header_.passes.num_passes; ++p) {
          int minShift = 0;
          int maxShift = 0;
          frame_header_.passes.GetDownsamplingBracket(p, minShift, maxShift);
          if (maxShift >= target_shift) {
            req_p = p + 1;
          }
        }
        required_passes_ = req_p;
      } else if (prog_required_passes < frame_header_.passes.num_passes) {
        scale_mode_ = ScaleDecodeMode::kProgressivePassSkip;
        required_passes_ = prog_required_passes;
      } else {
        scale_mode_ = ScaleDecodeMode::kStreamingDownsampleFallback;
      }
    }
  }

  // Paths that skip all AC sections (Path A VarDCT 1:8 DC, Path B 1:8 Squeeze).
  if (scale_mode_ == ScaleDecodeMode::kNativeVarDCTDC ||
      (scale_mode_ == ScaleDecodeMode::kModularSqueezeNative &&
       required_passes_ == 0)) {
    can_skip_ac_groups_ = true;
    required_ac_groups_.assign(frame_dim_.num_groups, 0);
    num_required_groups_ = 0;
    num_required_sections_ =
        (toc_.size() <= 1) ? toc_.size() : (1 + frame_dim_.num_dc_groups);
    fallback_reason_ =
        (output_region_.xsize() > 0 && output_region_.ysize() > 0)
            ? RegionFallbackReason::kNone
            : RegionFallbackReason::kNoRegion;
    return;
  }

  // Path B with required_passes_ > 0: Modular Squeeze uses full_image, so all
  // spatial groups are needed for required_passes_ while fine passes are
  // skipped.
  if (scale_mode_ == ScaleDecodeMode::kModularSqueezeNative) {
    can_skip_ac_groups_ = false;
    required_ac_groups_.assign(frame_dim_.num_groups, 1);
    num_required_groups_ = frame_dim_.num_groups;
    num_required_sections_ =
        (toc_.size() <= 1)
            ? toc_.size()
            : (2 + frame_dim_.num_dc_groups +
               frame_dim_.num_groups * required_passes_);
    fallback_reason_ =
        (output_region_.xsize() > 0 && output_region_.ysize() > 0)
            ? RegionFallbackReason::kModularGlobalTransforms
            : RegionFallbackReason::kNoRegion;
    return;
  }

  const size_t full_sections_for_req_passes =
      (toc_.size() <= 1)
          ? toc_.size()
          : (2 + frame_dim_.num_dc_groups +
             frame_dim_.num_groups * required_passes_);
  num_required_sections_ = full_sections_for_req_passes;

  if (output_region_.xsize() == 0 || output_region_.ysize() == 0) {
    fallback_reason_ = RegionFallbackReason::kNoRegion;
    return;
  }
  if (toc_.size() <= 1) {
    fallback_reason_ = RegionFallbackReason::kSingleSection;
    return;
  }
  if (frame_header_.CanBeReferenced() || frame_header_.dc_level != 0) {
    fallback_reason_ = RegionFallbackReason::kReferenceableFrame;
    return;
  }
  if (use_slow_rendering_pipeline_) {
    fallback_reason_ = RegionFallbackReason::kSimpleRenderPipeline;
    return;
  }
  if (decoded_ && decoded_->IsJPEG()) {
    fallback_reason_ = RegionFallbackReason::kJpegReconstruction;
    return;
  }
  if (after_dc_global && modular_frame_decoder_.UsesFullImage()) {
    fallback_reason_ = RegionFallbackReason::kModularGlobalTransforms;
    return;
  }

  size_t unoriented_w = 0;
  size_t unoriented_h = 0;
  Orientation undo_orientation = Orientation::kIdentity;
  if (after_dc_global && dec_state_->width > 0 && dec_state_->height > 0) {
    unoriented_w = dec_state_->width;
    unoriented_h = dec_state_->height;
    undo_orientation = dec_state_->undo_orientation;
  } else if (region_oriented_xsize_ > 0 && region_oriented_ysize_ > 0) {
    unoriented_w = region_oriented_xsize_;
    unoriented_h = region_oriented_ysize_;
    undo_orientation = region_undo_orientation_;
    if (static_cast<int>(undo_orientation) > 4) {
      std::swap(unoriented_w, unoriented_h);
    }
  } else {
    fallback_reason_ = RegionFallbackReason::kNoRegion;
    return;
  }

  const size_t eff_ds = output_downsampling_ > 1 ? output_downsampling_ : 1;
  const size_t scaled_unoriented_w = DivCeil(unoriented_w, eff_ds);
  const size_t scaled_unoriented_h = DivCeil(unoriented_h, eff_ds);
  const Rect unoriented_scaled = PassesDecoderState::ComputeUnorientedRegion(
      output_region_, scaled_unoriented_w, scaled_unoriented_h,
      undo_orientation);
  if (unoriented_scaled.xsize() == 0 || unoriented_scaled.ysize() == 0) {
    fallback_reason_ = RegionFallbackReason::kNoRegion;
    return;
  }
  Rect unoriented_region = unoriented_scaled;
  if (eff_ds > 1) {
    const size_t rx0 = unoriented_scaled.x0() * eff_ds;
    const size_t ry0 = unoriented_scaled.y0() * eff_ds;
    const size_t rx1 = std::min(unoriented_w, unoriented_scaled.x1() * eff_ds);
    const size_t ry1 = std::min(unoriented_h, unoriented_scaled.y1() * eff_ds);
    unoriented_region = Rect(rx0, ry0, rx1 - rx0, ry1 - ry0);
  }

  const bool apply_origin = coalescing_ && has_blending;

  RectT<ptrdiff_t> region(unoriented_region.x0(), unoriented_region.y0(),
                          unoriented_region.xsize(), unoriented_region.ysize());
  RectT<ptrdiff_t> frame_in_image(0, 0, frame_dim_.xsize_upsampled,
                                  frame_dim_.ysize_upsampled);
  if (apply_origin) {
    frame_in_image = frame_in_image.Translate(frame_header_.frame_origin.x0,
                                              frame_header_.frame_origin.y0);
  }
  RectT<ptrdiff_t> isect = region.Intersection(frame_in_image);

  fallback_reason_ = RegionFallbackReason::kNone;
  can_skip_ac_groups_ = true;
  required_ac_groups_.assign(frame_dim_.num_groups, 0);
  size_t num_required_groups = 0;

  if (isect.xsize() > 0 && isect.ysize() > 0) {
    if (apply_origin) {
      isect = isect.Translate(-frame_header_.frame_origin.x0,
                              -frame_header_.frame_origin.y0);
    }
    const size_t base_shift =
        (after_dc_global && dec_state_->render_pipeline)
            ? dec_state_->render_pipeline->BaseColorShift()
            : CeilLog2Nonzero(frame_header_.upsampling);
    const std::pair<size_t, size_t> group_border =
        (after_dc_global && dec_state_->render_pipeline)
            ? dec_state_->render_pipeline->GroupBorder()
            : EstimateGroupBorderFromHeader();
    const size_t bx = group_border.first;
    const size_t by = group_border.second;

    const size_t fx0 = static_cast<size_t>(isect.x0());
    const size_t fy0 = static_cast<size_t>(isect.y0());
    const size_t fx1 = static_cast<size_t>(isect.x1());
    const size_t fy1 = static_cast<size_t>(isect.y1());

    const size_t cx0 = fx0 >> base_shift;
    const size_t cy0 = fy0 >> base_shift;
    const size_t cx1 =
        std::min(frame_dim_.xsize, DivCeil(fx1, size_t{1} << base_shift));
    const size_t cy1 =
        std::min(frame_dim_.ysize, DivCeil(fy1, size_t{1} << base_shift));

    const size_t ex0 = (cx0 > bx) ? (cx0 - bx) : 0;
    const size_t ey0 = (cy0 > by) ? (cy0 - by) : 0;
    const size_t ex1 = std::min(frame_dim_.xsize, cx1 + bx);
    const size_t ey1 = std::min(frame_dim_.ysize, cy1 + by);

    const size_t gx0 = ex0 / frame_dim_.group_dim;
    const size_t gy0 = ey0 / frame_dim_.group_dim;
    const size_t gx1 =
        std::min(frame_dim_.xsize_groups, DivCeil(ex1, frame_dim_.group_dim));
    const size_t gy1 =
        std::min(frame_dim_.ysize_groups, DivCeil(ey1, frame_dim_.group_dim));

    for (size_t gy = gy0; gy < gy1; ++gy) {
      for (size_t gx = gx0; gx < gx1; ++gx) {
        required_ac_groups_[gy * frame_dim_.xsize_groups + gx] = 1;
        ++num_required_groups;
      }
    }
  }

  num_required_groups_ = num_required_groups;
  num_required_sections_ =
      2 + frame_dim_.num_dc_groups + num_required_groups * required_passes_;
}

void FrameDecoder::GetRequiredCodestreamRanges(
    bool coalesce_adjacent, std::vector<ByteRange>* ranges) const {
  ranges->clear();
  for (size_t i = 0; i < toc_.size(); ++i) {
    const size_t id = toc_[i].id;
    const size_t sz = toc_[i].size;
    if (sz == 0) continue;
    if (id < processed_section_.size() && processed_section_[id]) continue;
    if (!IsSectionRequiredPlanned(id)) continue;
    const uint64_t off = section_codestream_offsets_[i];
    if (coalesce_adjacent && !ranges->empty() &&
        ranges->back().offset + ranges->back().length == off) {
      ranges->back().length += sz;
    } else {
      ranges->push_back(ByteRange{off, sz});
    }
  }
}

FrameDecoder::RegionDecodeStats FrameDecoder::GetRegionDecodeStats() const {
  RegionDecodeStats stats;
  stats.fallback_reason = fallback_reason_;
  stats.total_groups = frame_dim_.num_groups;
  stats.required_groups = num_required_groups_;
  stats.skipped_groups =
      (frame_dim_.num_groups >= num_required_groups_)
          ? (frame_dim_.num_groups - num_required_groups_)
          : 0;
  stats.total_sections = toc_.size();
  stats.required_sections = num_required_sections_;
  stats.total_frame_payload_bytes = section_sizes_sum_;
  stats.downsampling = output_downsampling_;
  stats.scale_policy = scale_policy_;
  stats.scale_mode = scale_mode_;
  stats.total_passes = frame_header_.passes.num_passes;
  stats.required_passes = required_passes_;
  return stats;
}

Status FrameDecoder::Flush() {
  bool has_blending = frame_header_.blending_info.mode != BlendMode::kReplace ||
                      frame_header_.custom_size_or_origin;
  for (const auto& blending_info_ec :
       frame_header_.extra_channel_blending_info) {
    if (blending_info_ec.mode != BlendMode::kReplace) has_blending = true;
  }
  // No early Flush() if blending is enabled.
  if (has_blending && !is_finalized_) {
    return false;
  }
  // No early Flush() - nothing to do - if the frame is a kSkipProgressive
  // frame.
  if (frame_header_.frame_type == FrameType::kSkipProgressive &&
      !is_finalized_) {
    return true;
  }
  if (decoded_->IsJPEG()) {
    // Nothing to do.
    return true;
  }
  JXL_RETURN_IF_ERROR(AllocateOutput());

  uint32_t completely_decoded_ac_pass = NumCompletePasses();
  if (completely_decoded_ac_pass < frame_header_.passes.num_passes) {
    // We don't have all AC yet: force a draw of all the missing areas.
    // Mark all sections as not complete.
    for (size_t i = 0; i < decoded_passes_per_ac_group_.size(); i++) {
      if (can_skip_ac_groups_ && !required_ac_groups_[i]) continue;
      if (decoded_passes_per_ac_group_[i] < frame_header_.passes.num_passes) {
        dec_state_->render_pipeline->ClearDone(i);
      }
    }
    const auto prepare_storage = [this](const size_t num_threads) -> Status {
      JXL_RETURN_IF_ERROR(
          PrepareStorage(num_threads, decoded_passes_per_ac_group_.size()));
      return true;
    };
    const auto process_group = [this](const uint32_t g,
                                      size_t thread) -> Status {
      if (can_skip_ac_groups_ && !required_ac_groups_[g]) {
        return true;
      }
      if (decoded_passes_per_ac_group_[g] == frame_header_.passes.num_passes) {
        // This group was drawn already, nothing to do.
        return true;
      }
      BitReader* JXL_RESTRICT readers[kMaxNumPasses] = {};
      JXL_RETURN_IF_ERROR(ProcessACGroup(
          g, readers, /*num_passes=*/0, GetStorageLocation(thread, g),
          /*force_draw=*/true, /*dc_only=*/!decoded_ac_global_));
      return true;
    };
    JXL_RETURN_IF_ERROR(RunOnPool(pool_, 0, decoded_passes_per_ac_group_.size(),
                                  prepare_storage, process_group,
                                  "ForceDrawGroup"));
  }

  // undo global modular transforms and copy int pixel buffers to float ones
  JXL_RETURN_IF_ERROR(modular_frame_decoder_.FinalizeDecoding(
      frame_header_, dec_state_, pool_, is_finalized_));

  return true;
}

int FrameDecoder::SavedAs(const FrameHeader& header) {
  if (header.frame_type == FrameType::kDCFrame) {
    // bits 16, 32, 64, 128 for DC level
    return 16 << (header.dc_level - 1);
  } else if (header.CanBeReferenced()) {
    // bits 1, 2, 4 and 8 for the references
    return 1 << header.save_as_reference;
  }

  return 0;
}

bool FrameDecoder::HasEverything() const {
  if (!decoded_dc_global_) return false;
  if (HasDcGroupToDecode()) return false;
  if (scale_mode_ == ScaleDecodeMode::kNativeVarDCTDC ||
      (scale_mode_ == ScaleDecodeMode::kModularSqueezeNative &&
       required_passes_ == 0)) {
    return true;
  }
  if (!decoded_ac_global_) return false;
  for (size_t g = 0; g < decoded_passes_per_ac_group_.size(); ++g) {
    if (can_skip_ac_groups_ && !required_ac_groups_[g]) continue;
    if (decoded_passes_per_ac_group_[g] < required_passes_) {
      return false;
    }
  }
  return true;
}

int FrameDecoder::References() const {
  if (is_finalized_) {
    return 0;
  }
  if (!HasEverything()) return 0;

  int result = 0;

  // Blending
  if (frame_header_.frame_type == FrameType::kRegularFrame ||
      frame_header_.frame_type == FrameType::kSkipProgressive) {
    bool cropped = frame_header_.custom_size_or_origin;
    if (cropped || frame_header_.blending_info.mode != BlendMode::kReplace) {
      result |= (1 << frame_header_.blending_info.source);
    }
    const auto& extra = frame_header_.extra_channel_blending_info;
    for (const auto& ecbi : extra) {
      if (cropped || ecbi.mode != BlendMode::kReplace) {
        result |= (1 << ecbi.source);
      }
    }
  }

  // Patches
  if (frame_header_.flags & FrameHeader::kPatches) {
    result |= dec_state_->shared->image_features.patches.GetReferences();
  }

  // DC Level
  if (frame_header_.flags & FrameHeader::kUseDcFrame) {
    // Reads from the next dc level
    int dc_level = frame_header_.dc_level + 1;
    // bits 16, 32, 64, 128 for DC level
    result |= (16 << (dc_level - 1));
  }

  return result;
}

Status FrameDecoder::FinalizeFrame() {
  if (is_finalized_) {
    return JXL_FAILURE("FinalizeFrame called multiple times");
  }
  is_finalized_ = true;
  if (decoded_->IsJPEG()) {
    // Nothing to do.
    return true;
  }

  if (scale_mode_ == ScaleDecodeMode::kNativeVarDCTDC) {
    const size_t dc_w = DivCeil(frame_dim_.xsize_upsampled, 8);
    const size_t dc_h = DivCeil(frame_dim_.ysize_upsampled, 8);
    FrameDimensions scaled_dim;
    scaled_dim.Set(dc_w, dc_h, frame_header_.group_size_shift, 0, 0, false, 1);
    const Image3F& dc = *dec_state_->shared->dc;
    JXL_ENSURE(dc.xsize() >= dc_w && dc.ysize() >= dc_h);
    for (size_t g = 0; g < scaled_dim.num_groups; ++g) {
      dec_state_->render_pipeline->ClearDone(g);
    }
    const auto init = [&](size_t num_threads) -> Status {
      return dec_state_->render_pipeline->PrepareForThreads(
          num_threads, /*use_group_ids=*/false);
    };
    const auto process_group = [&](const uint32_t g,
                                   size_t thread_id) -> Status {
      RenderPipelineInput input =
          dec_state_->render_pipeline->GetInputBuffers(g, thread_id);
      const Rect gr = scaled_dim.GroupRect(g);
      for (size_t c = 0; c < 3; ++c) {
        const auto& buf = input.GetBuffer(c);
        const Rect& r = buf.second;
        JXL_ENSURE(r.xsize() == gr.xsize() && r.ysize() == gr.ysize());
        for (size_t y = 0; y < r.ysize(); ++y) {
          memcpy(r.Row(buf.first, y),
                 dc.ConstPlaneRow(c, gr.y0() + y) + gr.x0(),
                 r.xsize() * sizeof(float));
        }
      }
      return input.Done();
    };
    JXL_RETURN_IF_ERROR(RunOnPool(pool_, 0, scaled_dim.num_groups, init,
                                  process_group, "VarDCTDCToPipeline"));
  } else if (scale_mode_ == ScaleDecodeMode::kModularSqueezeNative) {
    JXL_RETURN_IF_ERROR(modular_frame_decoder_.FinalizeDecoding(
        frame_header_, dec_state_, pool_, /*inplace=*/true,
        /*downsampling=*/output_downsampling_));
  } else {
    // undo global modular transforms and copy int pixel buffers to float ones
    JXL_RETURN_IF_ERROR(modular_frame_decoder_.FinalizeDecoding(
        frame_header_, dec_state_, pool_, /*inplace=*/true));
  }

  if (dec_state_->render_pipeline) {
    JXL_RETURN_IF_ERROR(
        dec_state_->render_pipeline->RenderOutOfFramePadding(0));
    JXL_RETURN_IF_ERROR(dec_state_->render_pipeline->FinalizeOutput());
  }

  if (frame_header_.CanBeReferenced()) {
    auto& info = dec_state_->shared_storage
                     .reference_frames[frame_header_.save_as_reference];
    *info.frame = std::move(dec_state_->frame_storage_for_referencing);
    info.ib_is_in_xyb = frame_header_.save_before_color_transform;
  }
  return true;
}

}  // namespace jxl
