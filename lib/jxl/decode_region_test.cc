// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/codestream_header.h>
#include <jxl/color_encoding.h>
#include <jxl/decode.h>
#include <jxl/decode_cxx.h>
#include <jxl/decode_roi_experimental.h>
#include <jxl/encode.h>
#include <jxl/encode_cxx.h>
#include <jxl/thread_parallel_runner.h>
#include <jxl/thread_parallel_runner_cxx.h>
#include <jxl/types.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "lib/jxl/testing.h"

namespace jxl {
namespace {

size_t BytesPerSample(JxlDataType data_type) {
  switch (data_type) {
    case JXL_TYPE_UINT8:
      return 1;
    case JXL_TYPE_UINT16:
    case JXL_TYPE_FLOAT16:
      return 2;
    case JXL_TYPE_FLOAT:
      return 4;
  }
  return 1;
}

// Generates a rich, non-uniform 8-bit procedural image with gradients, sharp
// edges, and high-frequency patterns so any off-by-one coordinate, dither,
// border, or orientation bug produces an immediate pixel mismatch.
std::vector<uint8_t> GenerateTestImage(size_t width, size_t height,
                                       uint32_t num_channels,
                                       uint32_t seed = 42) {
  std::vector<uint8_t> pixels(width * height * num_channels);
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      uint32_t h = static_cast<uint32_t>(x * 374761393u + y * 668265263u +
                                         seed * 1442695041u);
      h = (h ^ (h >> 13)) * 1274126177u;
      h = h ^ (h >> 16);
      size_t idx = (y * width + x) * num_channels;
      for (uint32_t c = 0; c < num_channels; ++c) {
        if (c == 3 || (num_channels == 2 && c == 1)) {
          pixels[idx + c] = static_cast<uint8_t>(
              ((x + y) % 17 == 0) ? 128 : (180 + ((h >> 8) & 75)));
        } else {
          uint32_t grad = static_cast<uint32_t>(
                              (x * 255) / std::max<size_t>(1, width - 1) +
                              (y * 255) / std::max<size_t>(1, height - 1)) /
                          2;
          uint32_t checker = (((x / 19) ^ (y / 23)) & 1) ? 40 : 0;
          uint32_t noise = (h >> (c * 8)) & 31;
          pixels[idx + c] =
              static_cast<uint8_t>((grad + checker + noise + c * 53) & 0xFF);
        }
      }
    }
  }
  return pixels;
}

// Generates a procedural RGB HALF (IEEE-754 binary16) image with finite linear
// HDR/ACES-like values (normals, small values, subnormals, and signed zero).
std::vector<uint16_t> GenerateRgbHalfImage(size_t width, size_t height,
                                           uint32_t seed = 99) {
  std::vector<uint16_t> pixels(width * height * 3);
  for (size_t y = 0; y < height; ++y) {
    for (size_t x = 0; x < width; ++x) {
      uint32_t h = static_cast<uint32_t>(x * 374761393u + y * 668265263u +
                                         seed * 1442695041u);
      h = (h ^ (h >> 13)) * 1274126177u;
      h = h ^ (h >> 16);
      size_t idx = (y * width + x) * 3;
      // Construct valid finite FP16 values (exponent 0..30, never 31=Inf/NaN):
      // mix of normal [0.0..4.0] ACES linear range and occasional subnormals.
      for (size_t c = 0; c < 3; ++c) {
        uint32_t v = (h >> (c * 10)) & 0x3FF;
        uint16_t exp = static_cast<uint16_t>(
            ((x + y + c) % 37 == 0) ? 0 : (10 + ((x / 64 + y / 64 + c) % 8)));
        uint16_t sign = ((x + y + c) % 53 == 0) ? 0x8000u : 0u;
        pixels[idx + c] =
            static_cast<uint16_t>(sign | (exp << 10) | (v & 0x3FF));
      }
    }
  }
  return pixels;
}

struct EncodeConfig {
  bool lossless = false;
  float distance = 1.0f;
  int effort = 3;
  bool modular = false;
  int responsive = -1;
  int gaborish = -1;
  int epf = -1;
  int noise = -1;
  int resampling = -1;
  int progressive_ac = -1;
  int group_order = -1;  // 0 = scanline, 1 = center-first (permuted TOC)
  JxlOrientation orientation = JXL_ORIENT_IDENTITY;
  bool use_container = false;
};

std::vector<uint8_t> EncodeJxl(const std::vector<uint8_t>& pixels, size_t width,
                               size_t height, uint32_t num_channels,
                               const EncodeConfig& cfg = EncodeConfig()) {
  auto enc = JxlEncoderMake(nullptr);
  EXPECT_NE(nullptr, enc.get());
  auto runner = JxlThreadParallelRunnerMake(nullptr, 4);
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderSetParallelRunner(enc.get(), JxlThreadParallelRunner,
                                        runner.get()));

  if (cfg.use_container) {
    EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderUseContainer(enc.get(), JXL_TRUE));
  }

  JxlBasicInfo basic_info;
  JxlEncoderInitBasicInfo(&basic_info);
  basic_info.xsize = static_cast<uint32_t>(width);
  basic_info.ysize = static_cast<uint32_t>(height);
  basic_info.bits_per_sample = 8;
  basic_info.uses_original_profile = cfg.lossless ? JXL_TRUE : JXL_FALSE;
  basic_info.orientation = cfg.orientation;
  const bool has_alpha = (num_channels == 2 || num_channels == 4);
  basic_info.num_color_channels = (num_channels < 3) ? 1 : 3;
  basic_info.num_extra_channels = has_alpha ? 1 : 0;
  basic_info.alpha_bits = has_alpha ? 8 : 0;
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetBasicInfo(enc.get(), &basic_info));

  JxlColorEncoding color_encoding = {};
  JxlColorEncodingSetToSRGB(&color_encoding,
                            /*is_gray=*/basic_info.num_color_channels == 1);
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderSetColorEncoding(enc.get(), &color_encoding));

  JxlEncoderFrameSettings* frame_settings =
      JxlEncoderFrameSettingsCreate(enc.get(), nullptr);
  EXPECT_NE(nullptr, frame_settings);
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderFrameSettingsSetOption(
                frame_settings, JXL_ENC_FRAME_SETTING_EFFORT, cfg.effort));
  if (cfg.lossless) {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderSetFrameLossless(frame_settings, JXL_TRUE));
  } else {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderSetFrameDistance(frame_settings, cfg.distance));
  }
  if (cfg.modular) {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderFrameSettingsSetOption(
                  frame_settings, JXL_ENC_FRAME_SETTING_MODULAR, 1));
  }
  if (cfg.responsive >= 0) {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderFrameSettingsSetOption(
                  frame_settings, JXL_ENC_FRAME_SETTING_RESPONSIVE,
                  cfg.responsive));
  }
  if (cfg.gaborish >= 0) {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderFrameSettingsSetOption(
                  frame_settings, JXL_ENC_FRAME_SETTING_GABORISH,
                  cfg.gaborish));
  }
  if (cfg.epf >= 0) {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderFrameSettingsSetOption(
                  frame_settings, JXL_ENC_FRAME_SETTING_EPF, cfg.epf));
  }
  if (cfg.noise >= 0) {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderFrameSettingsSetOption(
                  frame_settings, JXL_ENC_FRAME_SETTING_NOISE, cfg.noise));
  }
  if (cfg.resampling >= 0) {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderFrameSettingsSetOption(
                  frame_settings, JXL_ENC_FRAME_SETTING_RESAMPLING,
                  cfg.resampling));
  }
  if (cfg.progressive_ac >= 0) {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderFrameSettingsSetOption(
                  frame_settings, JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC,
                  cfg.progressive_ac));
  }
  if (cfg.group_order >= 0) {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderFrameSettingsSetOption(
                  frame_settings, JXL_ENC_FRAME_SETTING_GROUP_ORDER,
                  cfg.group_order));
  }

  JxlPixelFormat format = {num_channels, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderAddImageFrame(frame_settings, &format, pixels.data(),
                                    pixels.size()));
  JxlEncoderCloseInput(enc.get());

  std::vector<uint8_t> compressed(64);
  uint8_t* next_out = compressed.data();
  size_t avail_out = compressed.size();
  JxlEncoderStatus status = JXL_ENC_NEED_MORE_OUTPUT;
  while (status == JXL_ENC_NEED_MORE_OUTPUT) {
    status = JxlEncoderProcessOutput(enc.get(), &next_out, &avail_out);
    if (status == JXL_ENC_NEED_MORE_OUTPUT) {
      size_t offset = next_out - compressed.data();
      compressed.resize(compressed.size() * 2);
      next_out = compressed.data() + offset;
      avail_out = compressed.size() - offset;
    }
  }
  EXPECT_EQ(JXL_ENC_SUCCESS, status);
  compressed.resize(next_out - compressed.data());
  return compressed;
}

// Encodes an RGB HALF (FP16) lossless Modular JXL (ACES2065-1 linear primaries
// / custom primaries + linear transfer function).
std::vector<uint8_t> EncodeRgbFloat16Jxl(const std::vector<uint16_t>& pixels,
                                          size_t width, size_t height,
                                          int effort = 1,
                                          bool use_container = false,
                                          int group_order = 0) {
  auto enc = JxlEncoderMake(nullptr);
  EXPECT_NE(nullptr, enc.get());
  auto runner = JxlThreadParallelRunnerMake(nullptr, 4);
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderSetParallelRunner(enc.get(), JxlThreadParallelRunner,
                                        runner.get()));
  if (use_container) {
    EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderUseContainer(enc.get(), JXL_TRUE));
  }

  JxlBasicInfo basic_info;
  JxlEncoderInitBasicInfo(&basic_info);
  basic_info.xsize = static_cast<uint32_t>(width);
  basic_info.ysize = static_cast<uint32_t>(height);
  basic_info.bits_per_sample = 16;
  basic_info.exponent_bits_per_sample = 5;
  basic_info.uses_original_profile = JXL_TRUE;
  basic_info.num_color_channels = 3;
  basic_info.num_extra_channels = 0;
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetBasicInfo(enc.get(), &basic_info));

  // ACEScg AP1 primaries + D60 white point + linear transfer function.
  JxlColorEncoding color_encoding = {};
  color_encoding.color_space = JXL_COLOR_SPACE_RGB;
  color_encoding.white_point = JXL_WHITE_POINT_CUSTOM;
  color_encoding.white_point_xy[0] = 0.32168;
  color_encoding.white_point_xy[1] = 0.33767;
  color_encoding.primaries = JXL_PRIMARIES_CUSTOM;
  color_encoding.primaries_red_xy[0] = 0.713;
  color_encoding.primaries_red_xy[1] = 0.293;
  color_encoding.primaries_green_xy[0] = 0.165;
  color_encoding.primaries_green_xy[1] = 0.830;
  color_encoding.primaries_blue_xy[0] = 0.128;
  color_encoding.primaries_blue_xy[1] = 0.044;
  color_encoding.transfer_function = JXL_TRANSFER_FUNCTION_LINEAR;
  color_encoding.rendering_intent = JXL_RENDERING_INTENT_RELATIVE;
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderSetColorEncoding(enc.get(), &color_encoding));

  JxlEncoderFrameSettings* fs =
      JxlEncoderFrameSettingsCreate(enc.get(), nullptr);
  EXPECT_NE(nullptr, fs);
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderFrameSettingsSetOption(
                fs, JXL_ENC_FRAME_SETTING_EFFORT, effort));
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetFrameLossless(fs, JXL_TRUE));
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderFrameSettingsSetOption(
                fs, JXL_ENC_FRAME_SETTING_MODULAR, 1));
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderFrameSettingsSetOption(
                fs, JXL_ENC_FRAME_SETTING_RESPONSIVE, 0));
  if (group_order != 0) {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderFrameSettingsSetOption(
                  fs, JXL_ENC_FRAME_SETTING_GROUP_ORDER, group_order));
  }

  JxlPixelFormat format = {3, JXL_TYPE_FLOAT16, JXL_NATIVE_ENDIAN, 0};
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderAddImageFrame(fs, &format, pixels.data(),
                                    pixels.size() * sizeof(uint16_t)));
  JxlEncoderCloseInput(enc.get());

  std::vector<uint8_t> compressed(1024);
  uint8_t* next_out = compressed.data();
  size_t avail_out = compressed.size();
  JxlEncoderStatus status = JXL_ENC_NEED_MORE_OUTPUT;
  while (status == JXL_ENC_NEED_MORE_OUTPUT) {
    status = JxlEncoderProcessOutput(enc.get(), &next_out, &avail_out);
    if (status == JXL_ENC_NEED_MORE_OUTPUT) {
      size_t offset = next_out - compressed.data();
      compressed.resize(compressed.size() * 2);
      next_out = compressed.data() + offset;
      avail_out = compressed.size() - offset;
    }
  }
  EXPECT_EQ(JXL_ENC_SUCCESS, status);
  compressed.resize(next_out - compressed.data());
  return compressed;
}

struct DecodeResult {
  std::vector<uint8_t> pixels;
  size_t width = 0;
  size_t height = 0;
  size_t bytes_fed = 0;
  size_t range_requests = 0;
  std::vector<std::vector<uint8_t>> frames;
  JxlExperimentalRegionDecodeStats stats = {};
};

enum class OutputMode {
  kBuffer,
  kSimpleCallback,
  kMultithreadedCallback,
};

struct CallbackContext {
  uint8_t* buffer = nullptr;
  size_t width = 0;
  size_t height = 0;
  size_t pixel_bytes = 0;
  size_t stride = 0;
  bool out_of_bounds = false;
};

void SimpleOutCallback(void* opaque, size_t x, size_t y, size_t num_pixels,
                       const void* pixels) {
  auto* ctx = static_cast<CallbackContext*>(opaque);
  if (y >= ctx->height || x + num_pixels > ctx->width) {
    ctx->out_of_bounds = true;
    return;
  }
  memcpy(ctx->buffer + y * ctx->stride + x * ctx->pixel_bytes, pixels,
         num_pixels * ctx->pixel_bytes);
}

void* MtInitCallback(void* init_opaque, size_t /*num_threads*/,
                     size_t /*num_pixels_per_thread*/) {
  return init_opaque;
}

void MtRunCallback(void* run_opaque, size_t /*thread_id*/, size_t x, size_t y,
                   size_t num_pixels, const void* pixels) {
  SimpleOutCallback(run_opaque, x, y, num_pixels, pixels);
}

void MtDestroyCallback(void* /*run_opaque*/) {}

DecodeResult DecodeJxl(const std::vector<uint8_t>& compressed,
                       const JxlPixelFormat& format, bool has_region,
                       uint32_t x0, uint32_t y0, uint32_t region_w,
                       uint32_t region_h, size_t chunk_size = 0,
                       bool keep_orientation = false, bool use_threads = false,
                       OutputMode output_mode = OutputMode::kBuffer,
                       bool set_region_before_basic_info = false,
                       bool stop_after_first_frame = true) {
  DecodeResult res;
  auto dec = JxlDecoderMake(nullptr);
  EXPECT_NE(nullptr, dec.get());

  JxlThreadParallelRunnerPtr runner(nullptr);
  if (use_threads) {
    runner = JxlThreadParallelRunnerMake(nullptr, 4);
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderSetParallelRunner(dec.get(), JxlThreadParallelRunner,
                                          runner.get()));
  }

  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSubscribeEvents(dec.get(),
                                      JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE));
  if (keep_orientation) {
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderSetKeepOrientation(dec.get(), JXL_TRUE));
  }
  if (has_region && set_region_before_basic_info) {
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderExperimentalSetOutputRegion(dec.get(), x0, y0, region_w,
                                                    region_h));
  }

  size_t feed_pos = 0;
  const size_t step = (chunk_size == 0) ? compressed.size() : chunk_size;
  std::vector<uint8_t> current_frame;
  CallbackContext cb_ctx;

  while (true) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());
    if (status == JXL_DEC_NEED_MORE_INPUT) {
      size_t remaining = JxlDecoderReleaseInput(dec.get());
      feed_pos -= remaining;
      if (feed_pos >= compressed.size()) {
        JxlDecoderCloseInput(dec.get());
        continue;
      }
      size_t to_feed = std::min(remaining + step, compressed.size() - feed_pos);
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderSetInput(dec.get(), compressed.data() + feed_pos,
                                   to_feed));
      feed_pos += to_feed;
      res.bytes_fed = std::max(res.bytes_fed, feed_pos);
    } else if (status == JXL_DEC_BASIC_INFO) {
      JxlBasicInfo info;
      EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetBasicInfo(dec.get(), &info));
      res.width = has_region ? region_w : info.xsize;
      res.height = has_region ? region_h : info.ysize;
      if (has_region && !set_region_before_basic_info) {
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderExperimentalSetOutputRegion(dec.get(), x0, y0,
                                                        region_w, region_h));
      }
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t buffer_size = 0;
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderImageOutBufferSize(dec.get(), &format, &buffer_size));
      const size_t pixel_bytes =
          format.num_channels * BytesPerSample(format.data_type);
      EXPECT_EQ(res.width * res.height * pixel_bytes, buffer_size);
      current_frame.assign(buffer_size, 0xAA);
      if (output_mode == OutputMode::kBuffer) {
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderSetImageOutBuffer(dec.get(), &format,
                                              current_frame.data(),
                                              current_frame.size()));
      } else {
        cb_ctx.buffer = current_frame.data();
        cb_ctx.width = res.width;
        cb_ctx.height = res.height;
        cb_ctx.pixel_bytes = pixel_bytes;
        cb_ctx.stride = res.width * pixel_bytes;
        cb_ctx.out_of_bounds = false;
        if (output_mode == OutputMode::kSimpleCallback) {
          EXPECT_EQ(JXL_DEC_SUCCESS,
                    JxlDecoderSetImageOutCallback(dec.get(), &format,
                                                  SimpleOutCallback, &cb_ctx));
        } else {
          EXPECT_EQ(JXL_DEC_SUCCESS,
                    JxlDecoderSetMultithreadedImageOutCallback(
                        dec.get(), &format, MtInitCallback, MtRunCallback,
                        MtDestroyCallback, &cb_ctx));
        }
      }
    } else if (status == JXL_DEC_FULL_IMAGE) {
      if (output_mode != OutputMode::kBuffer) {
        EXPECT_FALSE(cb_ctx.out_of_bounds);
      }
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderExperimentalGetRegionDecodeStats(dec.get(),
                                                           &res.stats));
      res.frames.push_back(current_frame);
      res.pixels = current_frame;
      if (stop_after_first_frame) {
        break;
      }
    } else if (status == JXL_DEC_SUCCESS) {
      break;
    } else {
      ADD_FAILURE() << "Unexpected JxlDecoderStatus: " << status;
      break;
    }
  }
  return res;
}

// Simulates a NAS / HTTP Range client:
// 1. Streams ONLY enough initial bytes (in 512-byte increments) to parse
//    BASIC_INFO + FRAME (header + TOC).
// 2. Calls JxlDecoderExperimentalSetOutputRegion + GetRequiredInputRanges.
// 3. Fetches ONLY those exact byte ranges from `compressed` via
//    JxlDecoderExperimentalSetInputAtOffset (optionally reversed, sub-chunked,
//    or overlapped) — NEVER feeding bytes from unneeded groups!
DecodeResult DecodeJxlSparse(const std::vector<uint8_t>& compressed,
                             const JxlPixelFormat& format, uint32_t x0,
                             uint32_t y0, uint32_t region_w, uint32_t region_h,
                             bool reverse_ranges = false,
                             size_t sub_chunk = 0,
                             bool test_overlap_dup = false,
                             bool use_threads = true) {
  DecodeResult res;
  auto dec = JxlDecoderMake(nullptr);
  EXPECT_NE(nullptr, dec.get());

  JxlThreadParallelRunnerPtr runner(nullptr);
  if (use_threads) {
    runner = JxlThreadParallelRunnerMake(nullptr, 4);
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderSetParallelRunner(dec.get(), JxlThreadParallelRunner,
                                          runner.get()));
  }

  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSubscribeEvents(
                dec.get(), JXL_DEC_BASIC_INFO | JXL_DEC_FRAME |
                               JXL_DEC_FULL_IMAGE));

  size_t header_pos = 0;
  bool in_sparse_phase = false;
  res.width = region_w;
  res.height = region_h;

  auto feed_required_ranges = [&]() {
    size_t count = 0;
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderExperimentalGetRequiredInputRanges(dec.get(), nullptr,
                                                           &count));
    std::vector<JxlExperimentalInputRange> ranges(count);
    if (count > 0) {
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderExperimentalGetRequiredInputRanges(
                    dec.get(), ranges.data(), &count));
      ranges.resize(count);
    }
    if (reverse_ranges) {
      std::reverse(ranges.begin(), ranges.end());
    }
    for (const auto& r : ranges) {
      EXPECT_LE(r.offset + r.length, compressed.size());
      res.range_requests++;
      res.bytes_fed += static_cast<size_t>(r.length);
      if (sub_chunk == 0 || r.length <= sub_chunk) {
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderExperimentalSetInputAtOffset(
                      dec.get(), compressed.data() + r.offset,
                      static_cast<size_t>(r.length), r.offset));
      } else {
        // Feed in sub-chunks (with optional overlap/duplicate check).
        for (uint64_t off = 0; off < r.length; off += sub_chunk) {
          uint64_t len = std::min<uint64_t>(sub_chunk, r.length - off);
          EXPECT_EQ(JXL_DEC_SUCCESS,
                    JxlDecoderExperimentalSetInputAtOffset(
                        dec.get(), compressed.data() + r.offset + off,
                        static_cast<size_t>(len), r.offset + off));
          if (test_overlap_dup && off > 0) {
            uint64_t dup_off = off - std::min<uint64_t>(off, 17);
            uint64_t dup_len = std::min<uint64_t>(33, r.length - dup_off);
            EXPECT_EQ(JXL_DEC_SUCCESS,
                      JxlDecoderExperimentalSetInputAtOffset(
                          dec.get(), compressed.data() + r.offset + dup_off,
                          static_cast<size_t>(dup_len), r.offset + dup_off));
          }
        }
      }
    }
  };

  while (true) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());
    if (status == JXL_DEC_NEED_MORE_INPUT) {
      if (!in_sparse_phase) {
        size_t remaining = JxlDecoderReleaseInput(dec.get());
        header_pos -= remaining;
        size_t to_feed = std::min<size_t>(512, compressed.size() - header_pos);
        if (to_feed == 0) {
          ADD_FAILURE() << "Ran out of header bytes in DecodeJxlSparse";
          break;
        }
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderSetInput(dec.get(), compressed.data() + header_pos,
                                     to_feed));
        header_pos += to_feed;
        res.bytes_fed = header_pos;
      } else {
        // Second round (e.g., Modular global Squeeze fallback revealed after
        // DC Global was decoded): query and fetch remaining ranges!
        feed_required_ranges();
      }
    } else if (status == JXL_DEC_BASIC_INFO) {
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderExperimentalSetOutputRegion(dec.get(), x0, y0,
                                                      region_w, region_h));
    } else if (status == JXL_DEC_FRAME) {
      // Header + TOC parsed! Release any leftover header bytes and switch to
      // sparse random-access mode.
      size_t leftover = JxlDecoderReleaseInput(dec.get());
      res.bytes_fed = header_pos - leftover;
      in_sparse_phase = true;
      feed_required_ranges();
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t buffer_size = 0;
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderImageOutBufferSize(dec.get(), &format, &buffer_size));
      res.pixels.assign(buffer_size, 0xAA);
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderSetImageOutBuffer(dec.get(), &format,
                                            res.pixels.data(),
                                            res.pixels.size()));
    } else if (status == JXL_DEC_FULL_IMAGE) {
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderExperimentalGetRegionDecodeStats(dec.get(),
                                                           &res.stats));
      break;
    } else {
      ADD_FAILURE() << "Unexpected sparse JxlDecoderStatus: " << status;
      break;
    }
  }
  return res;
}

std::vector<uint8_t> ExtractRegion(const std::vector<uint8_t>& full_pixels,
                                   size_t full_width,
                                   const JxlPixelFormat& format, size_t x0,
                                   size_t y0, size_t w, size_t h) {
  const size_t pixel_bytes =
      format.num_channels * BytesPerSample(format.data_type);
  const size_t full_stride = full_width * pixel_bytes;
  const size_t region_stride = w * pixel_bytes;
  std::vector<uint8_t> out(h * region_stride);
  for (size_t y = 0; y < h; ++y) {
    memcpy(out.data() + y * region_stride,
           full_pixels.data() + (y0 + y) * full_stride + x0 * pixel_bytes,
           region_stride);
  }
  return out;
}

void ExpectBitExactRegion(const std::vector<uint8_t>& full_pixels,
                          size_t full_width,
                          const std::vector<uint8_t>& region_pixels,
                          const JxlPixelFormat& format, size_t x0, size_t y0,
                          size_t w, size_t h, const std::string& label = "") {
  auto expected = ExtractRegion(full_pixels, full_width, format, x0, y0, w, h);
  ASSERT_EQ(expected.size(), region_pixels.size()) << label;
  if (memcmp(expected.data(), region_pixels.data(), expected.size()) != 0) {
    const size_t pixel_bytes =
        format.num_channels * BytesPerSample(format.data_type);
    const size_t stride = w * pixel_bytes;
    for (size_t y = 0; y < h; ++y) {
      for (size_t x = 0; x < w; ++x) {
        const size_t off = y * stride + x * pixel_bytes;
        if (memcmp(expected.data() + off, region_pixels.data() + off,
                   pixel_bytes) != 0) {
          FAIL() << "First pixel mismatch at region (" << x << ", " << y
                 << ") [image (" << (x0 + x) << ", " << (y0 + y)
                 << ")] for " << label << ": expected byte[0]="
                 << static_cast<int>(expected[off])
                 << " got=" << static_cast<int>(region_pixels[off]);
        }
      }
    }
  }
}

TEST(DecodeRegionTest, SparseRandomAccessAndRangePlanning) {
  const size_t w = 2048;
  const size_t h = 2048;
  auto src = GenerateTestImage(w, h, 3, /*seed=*/111);
  const JxlPixelFormat format = {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};

  struct SparseCase {
    const char* name;
    bool use_container;
    int group_order;
    bool reverse_ranges;
    size_t sub_chunk;
    bool overlap_dup;
  };
  const SparseCase cases[] = {
      {"bare_scanline_direct", false, 0, false, 0, false},
      {"container_jxlc_reversed_chunked", true, 0, true, 1024, true},
      {"bare_permuted_toc_center_first", false, 1, true, 2048, true},
  };

  for (const auto& tc : cases) {
    EncodeConfig cfg;
    cfg.distance = 1.0f;
    cfg.effort = 1;
    cfg.epf = 2;
    cfg.use_container = tc.use_container;
    cfg.group_order = tc.group_order;
    auto jxl = EncodeJxl(src, w, h, 3, cfg);
    auto full = DecodeJxl(jxl, format, false, 0, 0, 0, 0);

    // 256x256 crop in the middle of 2048x2048 (64 groups total).
    auto sparse = DecodeJxlSparse(jxl, format, 1024, 1024, 256, 256,
                                  tc.reverse_ranges, tc.sub_chunk,
                                  tc.overlap_dup);
    ExpectBitExactRegion(full.pixels, full.width, sparse.pixels, format, 1024,
                         1024, 256, 256, tc.name);
    EXPECT_EQ(JXL_EXP_FALLBACK_NONE, sparse.stats.fallback_reason) << tc.name;
    EXPECT_EQ(64u, sparse.stats.total_groups) << tc.name;
    EXPECT_GT(sparse.stats.skipped_groups, 48u) << tc.name;
    // Must have fetched far less than half the file!
    EXPECT_LT(sparse.bytes_fed, jxl.size() / 2)
        << tc.name << ": fed " << sparse.bytes_fed << " / " << jxl.size();
  }
}

TEST(DecodeRegionTest, Float2kRgbHalfSparseRandomAccess) {
  const size_t w = 2048;
  const size_t h = 1080;
  auto src = GenerateRgbHalfImage(w, h, /*seed=*/2026);
  const JxlPixelFormat format = {3, JXL_TYPE_FLOAT16, JXL_NATIVE_ENDIAN, 0};

  for (bool use_container : {false, true}) {
    auto jxl = EncodeRgbFloat16Jxl(src, w, h, /*effort=*/1, use_container);
    auto full = DecodeJxl(jxl, format, false, 0, 0, 0, 0, 0, false, true);

    const uint32_t regions[][4] = {
        {0, 0, 256, 256},
        {512, 256, 512, 512},
        {313, 197, 512, 512},
        {w - 256, h - 256, 256, 256},
    };
    for (const auto& r : regions) {
      auto sparse =
          DecodeJxlSparse(jxl, format, r[0], r[1], r[2], r[3],
                          /*reverse_ranges=*/true, /*sub_chunk=*/4096);
      ExpectBitExactRegion(
          full.pixels, full.width, sparse.pixels, format, r[0], r[1], r[2],
          r[3],
          "Float 2K HALF container=" + std::to_string(use_container) + " x0=" +
              std::to_string(r[0]));
      EXPECT_EQ(JXL_EXP_FALLBACK_NONE, sparse.stats.fallback_reason);
      EXPECT_GT(sparse.stats.skipped_groups, 0u);
      // 2048x1080 has 8x5 = 40 groups; a 256x256 aligned crop needs only 1!
      if (r[0] == 0 && r[1] == 0 && r[2] == 256 && r[3] == 256) {
        EXPECT_EQ(1u, sparse.stats.required_groups);
        EXPECT_EQ(39u, sparse.stats.skipped_groups);
        EXPECT_LT(sparse.bytes_fed, jxl.size() / 5);
      }
    }
  }
}

// Validates all 65,536 IEEE-754 HALF (binary16) bit patterns (#3881):
// verifies 100% bitwise equality between ROI decode and full-frame libjxl
// decode for all finite normals, all subnormals, and signed zero, while
// tracking NaN/Inf patterns separately.
TEST(DecodeRegionTest, All65536HalfPatternsBitExact) {
  // 512x512 = 262,144 pixels = 4 groups (2x2), plenty of room for all 65,536
  // HALF bit patterns across all 3 channels and all 4 groups.
  const size_t w = 512;
  const size_t h = 512;
  std::vector<uint16_t> src(w * h * 3, 0);
  for (size_t i = 0; i < 65536; ++i) {
    // Spread patterns across all 4 groups: (i % 512, i / 512) covers rows 0..127
    // and we mirror into rows 256..383 so every group has every category.
    uint16_t bits = static_cast<uint16_t>(i);
    size_t x = i % w;
    size_t y0 = i / w;
    size_t y1 = y0 + 256;
    for (size_t c = 0; c < 3; ++c) {
      src[(y0 * w + x) * 3 + c] = bits;
      src[(y1 * w + x) * 3 + c] = static_cast<uint16_t>(bits ^ (c * 0x1234));
    }
  }

  auto jxl = EncodeRgbFloat16Jxl(src, w, h, /*effort=*/1);
  const JxlPixelFormat format = {3, JXL_TYPE_FLOAT16, JXL_NATIVE_ENDIAN, 0};
  auto full = DecodeJxl(jxl, format, false, 0, 0, 0, 0);

  // Test regions across all 4 groups + boundary-crossing crop.
  const uint32_t regions[][4] = {
      {0, 0, 256, 128},
      {256, 0, 256, 128},
      {128, 256, 256, 128},
  };
  size_t finite_checked = 0;
  size_t subnormal_checked = 0;
  size_t neg_zero_checked = 0;
  size_t inf_nan_checked = 0;
  size_t inf_nan_bitwise_equal = 0;

  for (const auto& r : regions) {
    auto sparse = DecodeJxlSparse(jxl, format, r[0], r[1], r[2], r[3]);
    EXPECT_EQ(JXL_EXP_FALLBACK_NONE, sparse.stats.fallback_reason);
    auto expected =
        ExtractRegion(full.pixels, full.width, format, r[0], r[1], r[2], r[3]);
    ASSERT_EQ(expected.size(), sparse.pixels.size());

    const size_t num_samples = expected.size() / sizeof(uint16_t);
    const auto* exp_u16 = reinterpret_cast<const uint16_t*>(expected.data());
    const auto* got_u16 =
        reinterpret_cast<const uint16_t*>(sparse.pixels.data());
    for (size_t s = 0; s < num_samples; ++s) {
      const uint16_t e = exp_u16[s];
      const uint16_t g = got_u16[s];
      const uint16_t exp_field = (e >> 10) & 0x1F;
      const uint16_t mant_field = e & 0x3FF;
      if (exp_field == 31) {
        ++inf_nan_checked;
        if (e == g) ++inf_nan_bitwise_equal;
      } else {
        ++finite_checked;
        if (exp_field == 0 && mant_field != 0) ++subnormal_checked;
        if (e == 0x8000u) ++neg_zero_checked;
        ASSERT_EQ(e, g) << "Finite HALF mismatch at sample " << s;
      }
    }
  }
  EXPECT_GT(finite_checked, 0u);
  EXPECT_GT(subnormal_checked, 0u);
  EXPECT_GT(neg_zero_checked, 0u);
  EXPECT_GT(inf_nan_checked, 0u);
  EXPECT_EQ(inf_nan_checked, inf_nan_bitwise_equal);
}

// Full 8K (7680x4320) linear RGB HALF sparse random-access benchmark test.
// Kept DISABLED_ by default so standard unit test runs stay fast; run on demand:
//   ./build/lib/tests/decode_region_test --gtest_also_run_disabled_tests
//   --gtest_filter=*Float8k*
TEST(DecodeRegionTest, DISABLED_Float8kRgbHalfSparseBenchmark) {
  const size_t w = 7680;
  const size_t h = 4320;
  auto src = GenerateRgbHalfImage(w, h, /*seed=*/8000);
  auto jxl = EncodeRgbFloat16Jxl(src, w, h, /*effort=*/1);
  const JxlPixelFormat format = {3, JXL_TYPE_FLOAT16, JXL_NATIVE_ENDIAN, 0};

  auto t0 = std::chrono::steady_clock::now();
  auto full = DecodeJxl(jxl, format, false, 0, 0, 0, 0, 0, false, true);
  auto t1 = std::chrono::steady_clock::now();
  auto sparse = DecodeJxlSparse(jxl, format, 3584, 2048, 512, 512, false, 0,
                                false, true);
  auto t2 = std::chrono::steady_clock::now();

  ExpectBitExactRegion(full.pixels, full.width, sparse.pixels, format, 3584,
                       2048, 512, 512, "8K Float RGB HALF 512x512");
  EXPECT_EQ(JXL_EXP_FALLBACK_NONE, sparse.stats.fallback_reason);
  EXPECT_EQ(4u, sparse.stats.required_groups);
  EXPECT_LT(sparse.bytes_fed, jxl.size() / 20);  // < 5% of file!

  const double full_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double sparse_ms =
      std::chrono::duration<double, std::milli>(t2 - t1).count();
  printf(
      "[8K Float HALF 512x512] file=%zu bytes, sparse_read=%zu bytes (%.2f%%), "
      "ranges=%zu, groups=%zu/%zu, full=%.1f ms, sparse=%.1f ms\n",
      jxl.size(), sparse.bytes_fed,
      100.0 * static_cast<double>(sparse.bytes_fed) /
          static_cast<double>(jxl.size()),
      sparse.range_requests, sparse.stats.required_groups,
      sparse.stats.total_groups, full_ms, sparse_ms);
}

TEST(DecodeRegionTest, LargeVarDCT2KRegions) {
  const size_t w = 2048;
  const size_t h = 2048;
  const JxlPixelFormat format = {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};

  auto src = GenerateTestImage(w, h, 3, /*seed=*/101);
  EncodeConfig cfg;
  cfg.distance = 1.5f;
  cfg.effort = 1;
  auto jxl = EncodeJxl(src, w, h, 3, cfg);

  auto full = DecodeJxl(jxl, format, false, 0, 0, 0, 0, 0, false, true);
  ASSERT_EQ(w, full.width);
  ASSERT_EQ(h, full.height);

  struct RectCase {
    uint32_t x0, y0, w, h;
    const char* name;
  };
  const RectCase regions[] = {
      {0, 0, 256, 256, "top-left"},
      {1024, 1024, 256, 256, "center-aligned"},
      {static_cast<uint32_t>(w - 256), static_cast<uint32_t>(h - 256), 256, 256,
       "bottom-right"},
      {243, 251, 256, 256, "cross-4-groups-unaligned"},
      {513, 777, 1, 1, "single-pixel"},
      {0, 0, static_cast<uint32_t>(w), static_cast<uint32_t>(h), "full-image"},
  };
  for (const auto& r : regions) {
    auto reg = DecodeJxl(jxl, format, true, r.x0, r.y0, r.w, r.h, 0, false,
                         /*use_threads=*/true);
    ExpectBitExactRegion(full.pixels, full.width, reg.pixels, format, r.x0,
                         r.y0, r.w, r.h, r.name);
  }
}

TEST(DecodeRegionTest, VarDCTFeatureCombinations) {
  const size_t w = 600;
  const size_t h = 600;
  auto src = GenerateTestImage(w, h, 3, /*seed=*/202);

  struct FeatureCase {
    const char* name;
    EncodeConfig cfg;
  };
  std::vector<FeatureCase> cases;

  {
    EncodeConfig c;
    c.gaborish = 0;
    c.epf = 0;
    cases.push_back({"no_gab_no_epf", c});
  }
  {
    EncodeConfig c;
    c.gaborish = 1;
    c.epf = 3;
    cases.push_back({"gab_epf3", c});
  }
  {
    EncodeConfig c;
    c.gaborish = 1;
    c.epf = 2;
    c.noise = 1;
    cases.push_back({"gab_epf2_noise", c});
  }
  {
    EncodeConfig c;
    c.resampling = 2;
    cases.push_back({"resampling_2x", c});
  }
  {
    EncodeConfig c;
    c.resampling = 4;
    cases.push_back({"resampling_4x", c});
  }
  {
    EncodeConfig c;
    c.progressive_ac = 1;
    cases.push_back({"progressive_ac", c});
  }
  {
    EncodeConfig c;
    c.group_order = 1;  // center-first permuted TOC
    c.epf = 3;
    cases.push_back({"permuted_toc_center_first", c});
  }

  const JxlPixelFormat formats[] = {
      {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0},
      {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0},
  };

  for (const auto& tc : cases) {
    auto jxl = EncodeJxl(src, w, h, 3, tc.cfg);
    for (const auto& fmt : formats) {
      auto full = DecodeJxl(jxl, fmt, false, 0, 0, 0, 0);
      const uint32_t regions[][4] = {
          {240, 244, 128, 128},
          {0, 0, 256, 256},
          {344, 344, 256, 256},
      };
      for (const auto& r : regions) {
        auto reg = DecodeJxl(jxl, fmt, true, r[0], r[1], r[2], r[3],
                             /*chunk_size=*/4096, false, /*use_threads=*/true);
        ExpectBitExactRegion(full.pixels, full.width, reg.pixels, fmt, r[0],
                             r[1], r[2], r[3], tc.name);
      }
    }
  }
}

TEST(DecodeRegionTest, ModularGroupIndependentAndSqueezeFallback) {
  const size_t w = 600;
  const size_t h = 600;
  auto src = GenerateTestImage(w, h, 4, /*seed=*/303);
  const JxlPixelFormat format = {4, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};

  // 1. Lossless modular (group-independent RCT) via both streaming & sparse!
  {
    EncodeConfig cfg;
    cfg.lossless = true;
    cfg.modular = true;
    cfg.responsive = 0;
    auto jxl = EncodeJxl(src, w, h, 4, cfg);
    auto full = DecodeJxl(jxl, format, false, 0, 0, 0, 0);
    auto reg = DecodeJxl(jxl, format, true, 200, 220, 256, 256, 4096, false,
                         /*use_threads=*/true);
    ExpectBitExactRegion(full.pixels, full.width, reg.pixels, format, 200, 220,
                         256, 256, "modular_lossless_independent");
    EXPECT_EQ(JXL_EXP_FALLBACK_NONE, reg.stats.fallback_reason);
    EXPECT_GT(reg.stats.skipped_groups, 0u);

    auto sparse = DecodeJxlSparse(jxl, format, 200, 220, 256, 256);
    ExpectBitExactRegion(full.pixels, full.width, sparse.pixels, format, 200,
                         220, 256, 256, "modular_lossless_sparse");
    EXPECT_EQ(JXL_EXP_FALLBACK_NONE, sparse.stats.fallback_reason);
  }

  // 2. Lossless modular with responsive Squeeze (use_full_image fallback) via
  //    both streaming and 2-round sparse input!
  {
    EncodeConfig cfg;
    cfg.lossless = true;
    cfg.modular = true;
    cfg.responsive = 1;
    auto jxl = EncodeJxl(src, w, h, 4, cfg);
    auto full = DecodeJxl(jxl, format, false, 0, 0, 0, 0);
    auto reg = DecodeJxl(jxl, format, true, 200, 220, 256, 256, 4096, false,
                         /*use_threads=*/true);
    ExpectBitExactRegion(full.pixels, full.width, reg.pixels, format, 200, 220,
                         256, 256, "modular_lossless_squeeze_fallback");
    EXPECT_EQ(JXL_EXP_FALLBACK_MODULAR_GLOBAL_TRANSFORMS,
              reg.stats.fallback_reason);

    auto sparse = DecodeJxlSparse(jxl, format, 200, 220, 256, 256);
    ExpectBitExactRegion(full.pixels, full.width, sparse.pixels, format, 200,
                         220, 256, 256, "modular_squeeze_sparse_fallback");
    EXPECT_EQ(JXL_EXP_FALLBACK_MODULAR_GLOBAL_TRANSFORMS,
              sparse.stats.fallback_reason);
  }

  // 3. Lossy modular
  {
    EncodeConfig cfg;
    cfg.lossless = false;
    cfg.modular = true;
    cfg.distance = 2.0f;
    auto jxl = EncodeJxl(src, w, h, 4, cfg);
    auto full = DecodeJxl(jxl, format, false, 0, 0, 0, 0);
    auto reg = DecodeJxl(jxl, format, true, 123, 277, 200, 180, 4096, false,
                         /*use_threads=*/true);
    ExpectBitExactRegion(full.pixels, full.width, reg.pixels, format, 123, 277,
                         200, 180, "modular_lossy");
  }
}

TEST(DecodeRegionTest, AllEightExifOrientations) {
  const size_t w = 570;
  const size_t h = 410;
  auto src = GenerateTestImage(w, h, 3, /*seed=*/404);
  const JxlPixelFormat fmt_u8 = {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
  const JxlPixelFormat fmt_f32 = {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};

  for (int orient = 1; orient <= 8; ++orient) {
    EncodeConfig cfg;
    cfg.orientation = static_cast<JxlOrientation>(orient);
    cfg.epf = 2;
    auto jxl = EncodeJxl(src, w, h, 3, cfg);

    for (bool keep_orient : {false, true}) {
      for (const auto& fmt : {fmt_u8, fmt_f32}) {
        auto full = DecodeJxl(jxl, fmt, false, 0, 0, 0, 0, 0, keep_orient);
        const uint32_t rx0 = 137;
        const uint32_t ry0 = 91;
        const uint32_t rw = 211;
        const uint32_t rh = 197;
        ASSERT_LE(rx0 + rw, full.width);
        ASSERT_LE(ry0 + rh, full.height);
        auto reg = DecodeJxl(jxl, fmt, true, rx0, ry0, rw, rh,
                             /*chunk_size=*/2048, keep_orient,
                             /*use_threads=*/true);
        ExpectBitExactRegion(
            full.pixels, full.width, reg.pixels, fmt, rx0, ry0, rw, rh,
            "orient=" + std::to_string(orient) +
                " keep=" + std::to_string(keep_orient) +
                " dt=" + std::to_string(fmt.data_type));
      }
    }
  }
}

TEST(DecodeRegionTest, OutputFormatsAndCallbacks) {
  const size_t w = 560;
  const size_t h = 560;
  const uint32_t rx0 = 133;
  const uint32_t ry0 = 247;
  const uint32_t rw = 256;
  const uint32_t rh = 200;

  for (uint32_t channels : {1u, 2u, 3u, 4u}) {
    auto src = GenerateTestImage(w, h, channels, /*seed=*/500 + channels);
    EncodeConfig cfg;
    cfg.epf = 2;
    auto jxl = EncodeJxl(src, w, h, channels, cfg);

    for (JxlDataType dt :
         {JXL_TYPE_UINT8, JXL_TYPE_UINT16, JXL_TYPE_FLOAT16, JXL_TYPE_FLOAT}) {
      JxlPixelFormat fmt = {channels, dt, JXL_NATIVE_ENDIAN, 0};
      auto full = DecodeJxl(jxl, fmt, false, 0, 0, 0, 0);

      for (OutputMode mode : {OutputMode::kBuffer, OutputMode::kSimpleCallback,
                              OutputMode::kMultithreadedCallback}) {
        auto reg = DecodeJxl(jxl, fmt, true, rx0, ry0, rw, rh, 0, false,
                             /*use_threads=*/true, mode);
        ExpectBitExactRegion(
            full.pixels, full.width, reg.pixels, fmt, rx0, ry0, rw, rh,
            "ch=" + std::to_string(channels) + " dt=" + std::to_string(dt) +
                " mode=" + std::to_string(static_cast<int>(mode)));
      }
    }
  }
}

TEST(DecodeRegionTest, MultiFrameBlendingAndOutOfFramePadding) {
  const size_t w = 600;
  const size_t h = 600;
  auto frame0 = GenerateTestImage(w, h, 4, /*seed=*/601);
  auto frame1 = GenerateTestImage(300, 300, 4, /*seed=*/602);

  auto enc = JxlEncoderMake(nullptr);
  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = w;
  info.ysize = h;
  info.bits_per_sample = 8;
  info.num_color_channels = 3;
  info.num_extra_channels = 1;
  info.alpha_bits = 8;
  info.have_animation = JXL_TRUE;
  info.animation.tps_numerator = 10;
  info.animation.tps_denominator = 1;
  ASSERT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetBasicInfo(enc.get(), &info));

  JxlColorEncoding color_encoding = {};
  JxlColorEncodingSetToSRGB(&color_encoding, JXL_FALSE);
  ASSERT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderSetColorEncoding(enc.get(), &color_encoding));

  JxlPixelFormat format = {4, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};

  JxlEncoderFrameSettings* fs0 =
      JxlEncoderFrameSettingsCreate(enc.get(), nullptr);
  JxlFrameHeader fh0;
  JxlEncoderInitFrameHeader(&fh0);
  fh0.duration = 1;
  ASSERT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetFrameHeader(fs0, &fh0));
  ASSERT_EQ(JXL_ENC_SUCCESS, JxlEncoderAddImageFrame(fs0, &format,
                                                     frame0.data(),
                                                     frame0.size()));

  JxlEncoderFrameSettings* fs1 =
      JxlEncoderFrameSettingsCreate(enc.get(), nullptr);
  JxlFrameHeader fh1;
  JxlEncoderInitFrameHeader(&fh1);
  fh1.duration = 1;
  fh1.layer_info.have_crop = JXL_TRUE;
  fh1.layer_info.crop_x0 = 200;
  fh1.layer_info.crop_y0 = 180;
  fh1.layer_info.xsize = 300;
  fh1.layer_info.ysize = 300;
  fh1.layer_info.blend_info.blendmode = JXL_BLEND_BLEND;
  fh1.layer_info.blend_info.source = 0;
  fh1.layer_info.blend_info.alpha = 0;
  ASSERT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetFrameHeader(fs1, &fh1));
  ASSERT_EQ(JXL_ENC_SUCCESS, JxlEncoderAddImageFrame(fs1, &format,
                                                     frame1.data(),
                                                     frame1.size()));
  JxlEncoderCloseInput(enc.get());

  std::vector<uint8_t> jxl(64);
  uint8_t* next_out = jxl.data();
  size_t avail_out = jxl.size();
  JxlEncoderStatus status = JXL_ENC_NEED_MORE_OUTPUT;
  while (status == JXL_ENC_NEED_MORE_OUTPUT) {
    status = JxlEncoderProcessOutput(enc.get(), &next_out, &avail_out);
    if (status == JXL_ENC_NEED_MORE_OUTPUT) {
      size_t offset = next_out - jxl.data();
      jxl.resize(jxl.size() * 2);
      next_out = jxl.data() + offset;
      avail_out = jxl.size() - offset;
    }
  }
  ASSERT_EQ(JXL_ENC_SUCCESS, status);
  jxl.resize(next_out - jxl.data());

  auto full = DecodeJxl(jxl, format, false, 0, 0, 0, 0, 0, false, false,
                        OutputMode::kBuffer, false,
                        /*stop_after_first_frame=*/false);
  ASSERT_EQ(2u, full.frames.size());

  const uint32_t regions[][4] = {
      {220, 200, 150, 150},  // completely inside frame 1
      {100, 100, 250, 250},  // straddling frame 1 top-left border
      {0, 0, 150, 150},      // completely outside frame 1 (RenderPadding only)
  };
  for (const auto& r : regions) {
    auto reg = DecodeJxl(jxl, format, true, r[0], r[1], r[2], r[3],
                         /*chunk_size=*/2048, false, /*use_threads=*/true,
                         OutputMode::kBuffer, false,
                         /*stop_after_first_frame=*/false);
    ASSERT_EQ(2u, reg.frames.size());
    for (size_t f = 0; f < 2; ++f) {
      ExpectBitExactRegion(full.frames[f], full.width, reg.frames[f], format,
                           r[0], r[1], r[2], r[3],
                           "frame=" + std::to_string(f) + " x0=" +
                               std::to_string(r[0]));
    }
  }
}

TEST(DecodeRegionTest, StreamingChunkSizesAndEarlyCompletion) {
  const size_t w = 2048;
  const size_t h = 2048;
  auto src = GenerateTestImage(w, h, 3, /*seed=*/707);
  const JxlPixelFormat format = {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};

  for (bool use_container : {false, true}) {
    EncodeConfig cfg;
    cfg.distance = 1.0f;
    cfg.effort = 1;
    cfg.use_container = use_container;
    auto jxl = EncodeJxl(src, w, h, 3, cfg);

    auto full = DecodeJxl(jxl, format, false, 0, 0, 0, 0);

    for (size_t chunk : {size_t{64}, size_t{4096}, size_t{65536}}) {
      auto top_left = DecodeJxl(jxl, format, true, 0, 0, 256, 256, chunk);
      ExpectBitExactRegion(full.pixels, full.width, top_left.pixels, format, 0,
                           0, 256, 256,
                           "stream top-left chunk=" + std::to_string(chunk));
      EXPECT_LT(top_left.bytes_fed, jxl.size() / 2)
          << "Expected early completion for top-left region (use_container="
          << use_container << ", chunk=" << chunk << "): fed "
          << top_left.bytes_fed << " / " << jxl.size() << " bytes";

      auto center = DecodeJxl(jxl, format, true, 1024, 1024, 256, 256, chunk);
      ExpectBitExactRegion(full.pixels, full.width, center.pixels, format, 1024,
                           1024, 256, 256,
                           "stream center chunk=" + std::to_string(chunk));
    }
  }

  // 1-byte streaming on a 300x300 (2x2 = 4 groups) image.
  auto src_small = GenerateTestImage(300, 300, 3, /*seed=*/708);
  EncodeConfig cfg_small;
  cfg_small.distance = 3.0f;
  cfg_small.effort = 1;
  auto jxl_small = EncodeJxl(src_small, 300, 300, 3, cfg_small);
  auto full_small = DecodeJxl(jxl_small, format, false, 0, 0, 0, 0);
  auto reg_1b = DecodeJxl(jxl_small, format, true, 120, 120, 128, 128,
                          /*chunk_size=*/1);
  ExpectBitExactRegion(full_small.pixels, full_small.width, reg_1b.pixels,
                       format, 120, 120, 128, 128, "1-byte streaming");
}

TEST(DecodeRegionTest, ApiValidationAndRewind) {
  const size_t w = 600;
  const size_t h = 600;
  auto src = GenerateTestImage(w, h, 3, /*seed=*/808);
  auto jxl = EncodeJxl(src, w, h, 3);
  const JxlPixelFormat format = {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};

  auto full = DecodeJxl(jxl, format, false, 0, 0, 0, 0);

  auto reg_pre = DecodeJxl(jxl, format, true, 100, 150, 256, 256, 0, false,
                           false, OutputMode::kBuffer,
                           /*set_region_before_basic_info=*/true);
  ExpectBitExactRegion(full.pixels, full.width, reg_pre.pixels, format, 100,
                       150, 256, 256, "pre-basic-info");

  {
    auto dec = JxlDecoderMake(nullptr);
    EXPECT_EQ(JXL_DEC_ERROR,
              JxlDecoderExperimentalSetOutputRegion(nullptr, 0, 0, 10, 10));
    EXPECT_EQ(JXL_DEC_ERROR,
              JxlDecoderExperimentalSetOutputRegion(dec.get(), 0, 0, 0, 10));
    EXPECT_EQ(JXL_DEC_ERROR,
              JxlDecoderExperimentalSetOutputRegion(dec.get(), 0, 0, 10, 0));
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderExperimentalSetOutputRegion(dec.get(), 0, 0, 0, 0));

    // Range query and sparse input before TOC must fail cleanly.
    size_t count = 0;
    EXPECT_EQ(JXL_DEC_ERROR,
              JxlDecoderExperimentalGetRequiredInputRanges(dec.get(), nullptr,
                                                           &count));
    EXPECT_EQ(JXL_DEC_ERROR, JxlDecoderExperimentalSetInputAtOffset(
                                 dec.get(), jxl.data(), 10, 0));

    // Out of bounds detected after basic info.
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderSubscribeEvents(
                  dec.get(), JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE));
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderSetInput(dec.get(), jxl.data(), jxl.size()));
    EXPECT_EQ(JXL_DEC_BASIC_INFO, JxlDecoderProcessInput(dec.get()));
    EXPECT_EQ(JXL_DEC_ERROR,
              JxlDecoderExperimentalSetOutputRegion(dec.get(), 500, 0, 200, 100));
    EXPECT_EQ(JXL_DEC_ERROR,
              JxlDecoderExperimentalSetOutputRegion(dec.get(), 0, 500, 100, 200));

    // Out of bounds set BEFORE basic info is rejected when basic info is read.
    JxlDecoderReset(dec.get());
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_BASIC_INFO));
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderExperimentalSetOutputRegion(dec.get(), 500, 0, 200, 100));
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderSetInput(dec.get(), jxl.data(), jxl.size()));
    EXPECT_EQ(JXL_DEC_ERROR, JxlDecoderProcessInput(dec.get()));
  }

  // Rewind and decode two different regions with the same decoder instance.
  {
    auto dec = JxlDecoderMake(nullptr);
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderSubscribeEvents(
                  dec.get(), JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE));

    const uint32_t regions[][4] = {{0, 0, 200, 200}, {350, 350, 250, 250}};
    for (const auto& r : regions) {
      JxlDecoderRewind(dec.get());
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderExperimentalSetOutputRegion(dec.get(), r[0], r[1],
                                                      r[2], r[3]));
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderSetInput(dec.get(), jxl.data(), jxl.size()));
      EXPECT_EQ(JXL_DEC_BASIC_INFO, JxlDecoderProcessInput(dec.get()));
      EXPECT_EQ(JXL_DEC_NEED_IMAGE_OUT_BUFFER,
                JxlDecoderProcessInput(dec.get()));
      size_t sz = 0;
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderImageOutBufferSize(dec.get(), &format, &sz));
      EXPECT_EQ(r[2] * r[3] * 3, sz);
      std::vector<uint8_t> buf(sz);
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderSetImageOutBuffer(dec.get(), &format, buf.data(),
                                            buf.size()));
      EXPECT_EQ(JXL_DEC_FULL_IMAGE, JxlDecoderProcessInput(dec.get()));
      ExpectBitExactRegion(full.pixels, full.width, buf, format, r[0], r[1],
                           r[2], r[3], "rewind");
    }
  }
}

}  // namespace
}  // namespace jxl
