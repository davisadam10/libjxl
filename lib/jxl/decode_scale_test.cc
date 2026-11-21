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
#include <cmath>
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

float HalfToFloat(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t mant = h & 0x3FFu;
  uint32_t bits = 0;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      // Subnormal
      float v = std::ldexp(static_cast<float>(mant), -24);
      return sign ? -v : v;
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112) << 23) | (mant << 13);
  }
  float out = 0.0f;
  memcpy(&out, &bits, sizeof(out));
  return out;
}

uint16_t FloatToHalf(float f) {
  uint32_t bits = 0;
  memcpy(&bits, &f, sizeof(bits));
  const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
  const int exp = static_cast<int>((bits >> 23) & 0xFFu) - 127;
  uint32_t mant = bits & 0x7FFFFFu;
  if (exp < -24) return sign;
  if (exp < -14) {
    mant |= 0x800000u;
    const int shift = -14 - exp + 13;
    return static_cast<uint16_t>(sign | (mant >> shift));
  }
  if (exp > 15) {
    return static_cast<uint16_t>(sign | 0x7BFFu);  // clamp to max finite half
  }
  return static_cast<uint16_t>(sign | ((exp + 15) << 10) |
                               ((mant + 0x1000u) >> 13));
}

// Smooth + textured 8-bit image so downscaling metrics are well-conditioned.
std::vector<uint8_t> GenerateSmoothTestImage(size_t width, size_t height,
                                             uint32_t num_channels,
                                             uint32_t seed = 42) {
  std::vector<uint8_t> pixels(width * height * num_channels);
  for (size_t y = 0; y < height; ++y) {
    const float fy = static_cast<float>(y) / std::max<size_t>(1, height - 1);
    for (size_t x = 0; x < width; ++x) {
      const float fx = static_cast<float>(x) / std::max<size_t>(1, width - 1);
      const size_t idx = (y * width + x) * num_channels;
      for (uint32_t c = 0; c < num_channels; ++c) {
        if (c == 3 || (num_channels == 2 && c == 1)) {
          float a = 0.3f + 0.7f * (0.5f + 0.5f * std::sin(fx * 12.0f + fy * 8.0f));
          pixels[idx + c] =
              static_cast<uint8_t>(std::clamp(a * 255.0f, 0.0f, 255.0f));
        } else {
          float wave =
              0.5f +
              0.25f * std::sin(fx * (14.0f + c * 5.0f) + seed * 0.1f) +
              0.20f * std::cos(fy * (10.0f + c * 4.0f)) +
              0.05f * std::sin((fx + fy) * 45.0f);
          pixels[idx + c] =
              static_cast<uint8_t>(std::clamp(wave * 255.0f, 0.0f, 255.0f));
        }
      }
    }
  }
  return pixels;
}

// Linear HDR float32 image with negative values (wide gamut out-of-sRGB) and
// highlights > 1.0 (e.g. [-0.25, 8.0]), plus smooth spatial variation.
std::vector<float> GenerateHdrFloatImage(size_t width, size_t height,
                                         uint32_t seed = 7) {
  std::vector<float> pixels(width * height * 3);
  for (size_t y = 0; y < height; ++y) {
    const float fy = static_cast<float>(y) / std::max<size_t>(1, height - 1);
    for (size_t x = 0; x < width; ++x) {
      const float fx = static_cast<float>(x) / std::max<size_t>(1, width - 1);
      const size_t idx = (y * width + x) * 3;
      pixels[idx + 0] = -0.25f + 4.5f * fx + 1.2f * std::sin(fy * 12.0f + seed);
      pixels[idx + 1] = 0.1f + 3.0f * fy + 0.8f * std::cos(fx * 15.0f);
      pixels[idx + 2] =
          -0.15f + 6.0f * (fx * fy) + 2.0f * std::sin((fx + fy) * 9.0f);
    }
  }
  return pixels;
}

// Linear HDR RGB HALF image with smooth gradients, highlights > 1.0,
// occasional subnormals and -0.0.
std::vector<uint16_t> GenerateHdrHalfImage(size_t width, size_t height,
                                           uint32_t seed = 13) {
  std::vector<uint16_t> pixels(width * height * 3);
  for (size_t y = 0; y < height; ++y) {
    const float fy = static_cast<float>(y) / std::max<size_t>(1, height - 1);
    for (size_t x = 0; x < width; ++x) {
      const float fx = static_cast<float>(x) / std::max<size_t>(1, width - 1);
      const size_t idx = (y * width + x) * 3;
      if ((x + y) % 97 == 0) {
        // Explicit -0.0 and tiny subnormal values.
        pixels[idx + 0] = 0x8000u;  // -0.0
        pixels[idx + 1] = 0x0005u;  // positive subnormal
        pixels[idx + 2] = 0x8003u;  // negative subnormal
      } else {
        float r = -0.25f + 3.8f * fx + 0.6f * std::sin(fy * 14.0f + seed);
        float g = 0.05f + 2.8f * fy + 0.5f * std::cos(fx * 12.0f);
        float b = -0.15f + 4.4f * (0.5f * fx + 0.5f * fy);
        pixels[idx + 0] = FloatToHalf(r);
        pixels[idx + 1] = FloatToHalf(g);
        pixels[idx + 2] = FloatToHalf(b);
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
  int progressive_ac = -1;
  int qprogressive_ac = -1;
  JxlOrientation orientation = JXL_ORIENT_IDENTITY;
};

std::vector<uint8_t> EncodeJxl(const void* pixels, size_t pixel_bytes,
                               size_t width, size_t height,
                               uint32_t num_channels, JxlDataType data_type,
                               const EncodeConfig& cfg,
                               bool linear_ap1 = false) {
  auto enc = JxlEncoderMake(nullptr);
  EXPECT_NE(nullptr, enc.get());
  auto runner = JxlThreadParallelRunnerMake(
      nullptr, JxlThreadParallelRunnerDefaultNumWorkerThreads());
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderSetParallelRunner(enc.get(), JxlThreadParallelRunner,
                                        runner.get()));

  JxlBasicInfo basic_info;
  JxlEncoderInitBasicInfo(&basic_info);
  basic_info.xsize = static_cast<uint32_t>(width);
  basic_info.ysize = static_cast<uint32_t>(height);
  basic_info.num_color_channels = (num_channels >= 3) ? 3 : 1;
  basic_info.num_extra_channels =
      (num_channels == 2 || num_channels == 4) ? 1 : 0;
  basic_info.alpha_bits =
      (num_channels == 2 || num_channels == 4)
          ? (data_type == JXL_TYPE_UINT8
                 ? 8
                 : (data_type == JXL_TYPE_FLOAT16 ? 16 : 32))
          : 0;
  basic_info.bits_per_sample =
      (data_type == JXL_TYPE_UINT8 ? 8
                                   : (data_type == JXL_TYPE_FLOAT16 ? 16 : 32));
  basic_info.exponent_bits_per_sample =
      (data_type == JXL_TYPE_FLOAT16 ? 5
                                     : (data_type == JXL_TYPE_FLOAT ? 8 : 0));
  basic_info.uses_original_profile =
      (cfg.lossless || linear_ap1 || data_type == JXL_TYPE_FLOAT ||
       data_type == JXL_TYPE_FLOAT16)
          ? JXL_TRUE
          : JXL_FALSE;
  basic_info.orientation = cfg.orientation;
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetBasicInfo(enc.get(), &basic_info));

  JxlColorEncoding color_encoding = {};
  if (linear_ap1 || data_type == JXL_TYPE_FLOAT ||
      data_type == JXL_TYPE_FLOAT16) {
    color_encoding.color_space =
        (num_channels >= 3) ? JXL_COLOR_SPACE_RGB : JXL_COLOR_SPACE_GRAY;
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
  } else {
    JxlColorEncodingSetToSRGB(&color_encoding, num_channels < 3);
  }
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderSetColorEncoding(enc.get(), &color_encoding));

  JxlEncoderFrameSettings* frame_settings =
      JxlEncoderFrameSettingsCreate(enc.get(), nullptr);
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderFrameSettingsSetOption(
                frame_settings, JXL_ENC_FRAME_SETTING_EFFORT, cfg.effort));
  if (cfg.lossless) {
    EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetFrameLossless(frame_settings, true));
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
  if (cfg.progressive_ac >= 0) {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderFrameSettingsSetOption(
                  frame_settings, JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC,
                  cfg.progressive_ac));
  }
  if (cfg.qprogressive_ac >= 0) {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderFrameSettingsSetOption(
                  frame_settings, JXL_ENC_FRAME_SETTING_QPROGRESSIVE_AC,
                  cfg.qprogressive_ac));
  }

  JxlPixelFormat format = {num_channels, data_type, JXL_NATIVE_ENDIAN, 0};
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderAddImageFrame(frame_settings, &format, pixels,
                                    pixel_bytes));
  JxlEncoderCloseInput(enc.get());

  std::vector<uint8_t> compressed;
  compressed.resize(65536);
  uint8_t* next_out = compressed.data();
  size_t avail_out = compressed.size();
  for (;;) {
    JxlEncoderStatus st =
        JxlEncoderProcessOutput(enc.get(), &next_out, &avail_out);
    if (st == JXL_ENC_NEED_MORE_OUTPUT) {
      size_t offset = next_out - compressed.data();
      compressed.resize(compressed.size() * 2);
      next_out = compressed.data() + offset;
      avail_out = compressed.size() - offset;
    } else {
      EXPECT_EQ(JXL_ENC_SUCCESS, st);
      compressed.resize(next_out - compressed.data());
      break;
    }
  }
  return compressed;
}

struct ScaleDecodeResult {
  JxlDecoderStatus final_status = JXL_DEC_ERROR;
  std::vector<uint8_t> pixels;
  uint32_t out_xsize = 0;
  uint32_t out_ysize = 0;
  JXL_BOOL fast_native_available = JXL_FALSE;
  JxlExperimentalScaleDecodeMode info_scale_mode = JXL_EXP_SCALE_MODE_NONE;
  JxlExperimentalRegionDecodeStats stats = {};
  uint64_t bytes_read = 0;
  size_t range_requests = 0;
};

ScaleDecodeResult DecodeScaled(
    const std::vector<uint8_t>& compressed, uint32_t downsampling,
    JxlExperimentalScalePolicy policy, uint32_t num_channels,
    JxlDataType data_type, bool sparse = false, bool use_region = false,
    uint32_t rx = 0, uint32_t ry = 0, uint32_t rw = 0, uint32_t rh = 0,
    bool use_callback = false, bool unpremul_alpha = false,
    bool keep_orientation = false) {
  ScaleDecodeResult res;
  auto dec = JxlDecoderMake(nullptr);
  EXPECT_NE(nullptr, dec.get());
  auto runner = JxlThreadParallelRunnerMake(
      nullptr, JxlThreadParallelRunnerDefaultNumWorkerThreads());
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSetParallelRunner(dec.get(), JxlThreadParallelRunner,
                                        runner.get()));
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSubscribeEvents(
                dec.get(),
                JXL_DEC_BASIC_INFO | JXL_DEC_FRAME | JXL_DEC_FULL_IMAGE));
  if (keep_orientation) {
    EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderSetKeepOrientation(dec.get(), true));
  }
  if (unpremul_alpha) {
    EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderSetUnpremultiplyAlpha(dec.get(), true));
  }
  if (downsampling > 1 || policy != JXL_EXP_SCALE_FAST_APPROXIMATE) {
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderExperimentalSetOutputDownsampling(dec.get(),
                                                          downsampling, policy));
  }
  if (use_region) {
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderExperimentalSetOutputRegion(dec.get(), rx, ry, rw, rh));
  }

  JxlPixelFormat format = {num_channels, data_type, JXL_NATIVE_ENDIAN, 0};

  struct CallbackCtx {
    uint8_t* dst;
    size_t row_stride;
    size_t pixel_bytes;
  } cb_ctx = {nullptr, 0, num_channels * BytesPerSample(data_type)};

  auto pixel_cb = [](void* opaque, size_t x, size_t y, size_t num_pixels,
                     const void* pixels) {
    auto* ctx = static_cast<CallbackCtx*>(opaque);
    memcpy(ctx->dst + y * ctx->row_stride + x * ctx->pixel_bytes, pixels,
           num_pixels * ctx->pixel_bytes);
  };

  size_t seq_pos = 0;
  bool sparse_started = false;
  if (!sparse) {
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderSetInput(dec.get(), compressed.data(),
                                 compressed.size()));
    JxlDecoderCloseInput(dec.get());
    res.bytes_read = compressed.size();
  } else {
    // Feed initial header prefix in small chunks until JXL_DEC_FRAME.
    size_t chunk = std::min<size_t>(4096, compressed.size());
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderSetInput(dec.get(), compressed.data(), chunk));
    seq_pos = chunk;
    res.bytes_read += chunk;
  }

  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());
    if (status == JXL_DEC_ERROR) {
      res.final_status = JXL_DEC_ERROR;
      return res;
    } else if (status == JXL_DEC_NEED_MORE_INPUT) {
      if (!sparse) {
        res.final_status = JXL_DEC_NEED_MORE_INPUT;
        return res;
      }
      if (!sparse_started) {
        size_t remaining = JxlDecoderReleaseInput(dec.get());
        size_t consumed = seq_pos - remaining;
        size_t next_chunk = std::min<size_t>(4096, compressed.size() - seq_pos);
        if (next_chunk == 0) {
          JxlDecoderCloseInput(dec.get());
          res.final_status = JXL_DEC_NEED_MORE_INPUT;
          return res;
        }
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderSetInput(dec.get(), compressed.data() + consumed,
                                     remaining + next_chunk));
        seq_pos += next_chunk;
        res.bytes_read += next_chunk;
      } else {
        size_t range_count = 0;
        JxlDecoderStatus rst = JxlDecoderExperimentalGetRequiredInputRanges(
            dec.get(), nullptr, &range_count);
        if (rst != JXL_DEC_SUCCESS || range_count == 0) {
          res.final_status = JXL_DEC_NEED_MORE_INPUT;
          return res;
        }
        std::vector<JxlExperimentalInputRange> ranges(range_count);
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderExperimentalGetRequiredInputRanges(
                      dec.get(), ranges.data(), &range_count));
        for (size_t i = 0; i < range_count; ++i) {
          EXPECT_LE(ranges[i].offset + ranges[i].length, compressed.size());
          EXPECT_EQ(JXL_DEC_SUCCESS,
                    JxlDecoderExperimentalSetInputAtOffset(
                        dec.get(), compressed.data() + ranges[i].offset,
                        ranges[i].length, ranges[i].offset));
          res.bytes_read += ranges[i].length;
          res.range_requests++;
        }
      }
    } else if (status == JXL_DEC_BASIC_INFO) {
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderExperimentalGetScaledOutputInfo(
                    dec.get(), &res.out_xsize, &res.out_ysize,
                    &res.fast_native_available, &res.info_scale_mode));
    } else if (status == JXL_DEC_FRAME) {
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderExperimentalGetScaledOutputInfo(
                    dec.get(), &res.out_xsize, &res.out_ysize,
                    &res.fast_native_available, &res.info_scale_mode));
      if (sparse) {
        size_t remaining = JxlDecoderReleaseInput(dec.get());
        res.bytes_read -= remaining;
        sparse_started = true;
        size_t range_count = 0;
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderExperimentalGetRequiredInputRanges(
                      dec.get(), nullptr, &range_count));
        if (range_count > 0) {
          std::vector<JxlExperimentalInputRange> ranges(range_count);
          EXPECT_EQ(JXL_DEC_SUCCESS,
                    JxlDecoderExperimentalGetRequiredInputRanges(
                        dec.get(), ranges.data(), &range_count));
          for (size_t i = 0; i < range_count; ++i) {
            EXPECT_LE(ranges[i].offset + ranges[i].length, compressed.size());
            EXPECT_EQ(JXL_DEC_SUCCESS,
                      JxlDecoderExperimentalSetInputAtOffset(
                          dec.get(), compressed.data() + ranges[i].offset,
                          ranges[i].length, ranges[i].offset));
            res.bytes_read += ranges[i].length;
            res.range_requests++;
          }
        }
      }
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t buf_size = 0;
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderImageOutBufferSize(dec.get(), &format, &buf_size));
      EXPECT_EQ(buf_size, static_cast<size_t>(res.out_xsize) * res.out_ysize *
                              num_channels * BytesPerSample(data_type));
      res.pixels.resize(buf_size);
      if (use_callback) {
        cb_ctx.dst = res.pixels.data();
        cb_ctx.row_stride =
            static_cast<size_t>(res.out_xsize) * cb_ctx.pixel_bytes;
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderSetImageOutCallback(dec.get(), &format, pixel_cb,
                                                &cb_ctx));
      } else {
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderSetImageOutBuffer(dec.get(), &format,
                                              res.pixels.data(),
                                              res.pixels.size()));
      }
    } else if (status == JXL_DEC_FULL_IMAGE) {
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderExperimentalGetRegionDecodeStats(dec.get(),
                                                           &res.stats));
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderExperimentalGetScaledOutputInfo(
                    dec.get(), &res.out_xsize, &res.out_ysize,
                    &res.fast_native_available, &res.info_scale_mode));
      res.final_status = JXL_DEC_SUCCESS;
      return res;
    } else if (status == JXL_DEC_SUCCESS) {
      res.final_status = JXL_DEC_SUCCESS;
      return res;
    }
  }
}

// Reference exact floating-point box downsample (matching border tile sizes).
std::vector<float> ReferenceBoxDownsampleFloat(const float* src, size_t width,
                                               size_t height, size_t channels,
                                               size_t factor) {
  const size_t sw = (width + factor - 1) / factor;
  const size_t sh = (height + factor - 1) / factor;
  std::vector<float> dst(sw * sh * channels, 0.0f);
  for (size_t sy = 0; sy < sh; ++sy) {
    const size_t y0 = sy * factor;
    const size_t y1 = std::min(height, y0 + factor);
    for (size_t sx = 0; sx < sw; ++sx) {
      const size_t x0 = sx * factor;
      const size_t x1 = std::min(width, x0 + factor);
      const float inv = 1.0f / static_cast<float>((x1 - x0) * (y1 - y0));
      for (size_t c = 0; c < channels; ++c) {
        double sum = 0.0;
        for (size_t y = y0; y < y1; ++y) {
          for (size_t x = x0; x < x1; ++x) {
            sum += src[(y * width + x) * channels + c];
          }
        }
        dst[(sy * sw + sx) * channels + c] = static_cast<float>(sum * inv);
      }
    }
  }
  return dst;
}

std::vector<float> BufferToFloats(const std::vector<uint8_t>& buf,
                                  JxlDataType data_type) {
  if (data_type == JXL_TYPE_FLOAT) {
    const size_t n = buf.size() / sizeof(float);
    std::vector<float> out(n);
    memcpy(out.data(), buf.data(), buf.size());
    return out;
  } else if (data_type == JXL_TYPE_FLOAT16) {
    const size_t n = buf.size() / sizeof(uint16_t);
    std::vector<float> out(n);
    const auto* p = reinterpret_cast<const uint16_t*>(buf.data());
    for (size_t i = 0; i < n; ++i) out[i] = HalfToFloat(p[i]);
    return out;
  } else {
    std::vector<float> out(buf.size());
    for (size_t i = 0; i < buf.size(); ++i) out[i] = buf[i] / 255.0f;
    return out;
  }
}

struct ErrorMetrics {
  double max_abs_err = 0.0;
  double mae = 0.0;
  double rmse = 0.0;
  double psnr = 0.0;
};

ErrorMetrics ComputeMetrics(const std::vector<float>& a,
                            const std::vector<float>& b,
                            double peak = 1.0) {
  EXPECT_EQ(a.size(), b.size());
  ErrorMetrics m;
  double sum_abs = 0.0;
  double sum_sq = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = std::abs(static_cast<double>(a[i]) - b[i]);
    m.max_abs_err = std::max(m.max_abs_err, d);
    sum_abs += d;
    sum_sq += d * d;
  }
  const double n = static_cast<double>(std::max<size_t>(1, a.size()));
  m.mae = sum_abs / n;
  m.rmse = std::sqrt(sum_sq / n);
  m.psnr = (m.rmse > 0.0) ? (20.0 * std::log10(peak / m.rmse)) : 120.0;
  return m;
}

TEST(DecodeScaleTest, ApiValidationAndScaledOutputInfo) {
  auto dec = JxlDecoderMake(nullptr);
  ASSERT_NE(nullptr, dec.get());
  EXPECT_EQ(JXL_DEC_ERROR,
            JxlDecoderExperimentalSetOutputDownsampling(
                nullptr, 2, JXL_EXP_SCALE_FAST_APPROXIMATE));
  EXPECT_EQ(JXL_DEC_ERROR,
            JxlDecoderExperimentalSetOutputDownsampling(
                dec.get(), 0, JXL_EXP_SCALE_FAST_APPROXIMATE));
  EXPECT_EQ(JXL_DEC_ERROR,
            JxlDecoderExperimentalSetOutputDownsampling(
                dec.get(), 3, JXL_EXP_SCALE_FAST_APPROXIMATE));
  EXPECT_EQ(JXL_DEC_ERROR,
            JxlDecoderExperimentalSetOutputDownsampling(
                dec.get(), 16, JXL_EXP_SCALE_FAST_APPROXIMATE));
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderExperimentalSetOutputDownsampling(
                dec.get(), 4, JXL_EXP_SCALE_FAST_APPROXIMATE));

  uint32_t w = 0, h = 0;
  EXPECT_EQ(JXL_DEC_NEED_MORE_INPUT,
            JxlDecoderExperimentalGetScaledOutputInfo(dec.get(), &w, &h,
                                                      nullptr, nullptr));
}

TEST(DecodeScaleTest, AdversarialNonSqueezeLosslessModularFloatHdr) {
  const size_t W = 320;
  const size_t H = 240;
  EncodeConfig cfg;
  cfg.lossless = true;
  cfg.modular = true;
  cfg.responsive = 0;  // Explicitly NO Squeeze!

  // Lossless Modular FP16 HDR with negative values, highlights > 1.0,
  // subnormals, and -0.0.
  auto src_f16 = GenerateHdrHalfImage(W, H);
  auto jxl_f16 = EncodeJxl(src_f16.data(), src_f16.size() * sizeof(uint16_t), W,
                           H, 3, JXL_TYPE_FLOAT16, cfg, /*linear_ap1=*/true);

  // REQUIRE_FAST_NATIVE must explicitly fail (never silently fall back).
  auto req_res =
      DecodeScaled(jxl_f16, 4, JXL_EXP_SCALE_REQUIRE_FAST_NATIVE, 3,
                   JXL_TYPE_FLOAT);
  EXPECT_EQ(JXL_DEC_ERROR, req_res.final_status);

  // Decode full-res to FLOAT32 as exact reference.
  auto full_res =
      DecodeScaled(jxl_f16, 1, JXL_EXP_SCALE_FAST_APPROXIMATE, 3,
                   JXL_TYPE_FLOAT);
  ASSERT_EQ(JXL_DEC_SUCCESS, full_res.final_status);
  auto full_f = BufferToFloats(full_res.pixels, JXL_TYPE_FLOAT);
  auto ref_4x = ReferenceBoxDownsampleFloat(full_f.data(), W, H, 3, 4);

  // Both FAST_APPROXIMATE and REFERENCE_RESAMPLE must fall back cleanly to
  // streaming float box downsample without clamping negative or > 1.0 values.
  for (auto policy :
       {JXL_EXP_SCALE_FAST_APPROXIMATE, JXL_EXP_SCALE_REFERENCE_RESAMPLE}) {
    auto scaled = DecodeScaled(jxl_f16, 4, policy, 3, JXL_TYPE_FLOAT);
    ASSERT_EQ(JXL_DEC_SUCCESS, scaled.final_status);
    EXPECT_EQ(80u, scaled.out_xsize);
    EXPECT_EQ(60u, scaled.out_ysize);
    EXPECT_EQ(JXL_EXP_SCALE_MODE_STREAMING_DOWNSAMPLE_FALLBACK,
              scaled.stats.scale_mode);
    EXPECT_EQ(JXL_FALSE, scaled.fast_native_available);
    auto got = BufferToFloats(scaled.pixels, JXL_TYPE_FLOAT);
    float min_v = *std::min_element(got.begin(), got.end());
    float max_v = *std::max_element(got.begin(), got.end());
    EXPECT_LT(min_v, 0.0f);  // Negative values preserved!
    EXPECT_GT(max_v, 3.0f);  // Highlights > 1.0 preserved!
    auto m = ComputeMetrics(got, ref_4x, 4.5);
    EXPECT_LT(m.max_abs_err, 1e-4);
  }

  // Also verify JXL_TYPE_FLOAT16 output preserves HDR range and matches
  // reference within FP16 precision.
  auto scaled_f16 = DecodeScaled(jxl_f16, 4, JXL_EXP_SCALE_REFERENCE_RESAMPLE,
                                 3, JXL_TYPE_FLOAT16);
  ASSERT_EQ(JXL_DEC_SUCCESS, scaled_f16.final_status);
  auto got_f16 = BufferToFloats(scaled_f16.pixels, JXL_TYPE_FLOAT16);
  EXPECT_LT(*std::min_element(got_f16.begin(), got_f16.end()), 0.0f);
  EXPECT_GT(*std::max_element(got_f16.begin(), got_f16.end()), 3.0f);
  auto m16 = ComputeMetrics(got_f16, ref_4x, 4.5);
  EXPECT_LT(m16.max_abs_err, 5e-3);
}

TEST(DecodeScaleTest, NativeVarDCT1To8DC) {
  const size_t W = 768;
  const size_t H = 512;
  auto src = GenerateSmoothTestImage(W, H, 3);
  EncodeConfig cfg;
  cfg.distance = 1.0f;
  cfg.effort = 3;
  auto jxl = EncodeJxl(src.data(), src.size(), W, H, 3, JXL_TYPE_UINT8, cfg);

  auto full = DecodeScaled(jxl, 1, JXL_EXP_SCALE_FAST_APPROXIMATE, 3,
                           JXL_TYPE_FLOAT);
  ASSERT_EQ(JXL_DEC_SUCCESS, full.final_status);
  auto full_f = BufferToFloats(full.pixels, JXL_TYPE_FLOAT);
  auto ref_8x = ReferenceBoxDownsampleFloat(full_f.data(), W, H, 3, 8);

  // Test native 1:8 DC with sparse random access and REQUIRE_FAST_NATIVE.
  auto scaled = DecodeScaled(jxl, 8, JXL_EXP_SCALE_REQUIRE_FAST_NATIVE, 3,
                             JXL_TYPE_FLOAT, /*sparse=*/true);
  ASSERT_EQ(JXL_DEC_SUCCESS, scaled.final_status);
  EXPECT_EQ(96u, scaled.out_xsize);
  EXPECT_EQ(64u, scaled.out_ysize);
  EXPECT_EQ(JXL_TRUE, scaled.fast_native_available);
  EXPECT_EQ(JXL_EXP_SCALE_MODE_NATIVE_VARDCT_DC, scaled.stats.scale_mode);
  EXPECT_EQ(0u, scaled.stats.required_groups);
  EXPECT_EQ(0u, scaled.stats.required_passes);
  // Sparse bytes read should be a tiny fraction of the full file.
  EXPECT_LT(scaled.bytes_read, jxl.size() / 2);

  auto got = BufferToFloats(scaled.pixels, JXL_TYPE_FLOAT);
  auto m = ComputeMetrics(got, ref_8x, 1.0);
  EXPECT_GT(m.psnr, 28.0);
  EXPECT_LT(m.mae, 0.04);

  // Spatial alignment check: error at 0-shift must be smaller than ±1 px shift.
  auto shift_mse = [&](int dx, int dy) {
    double s = 0.0;
    size_t cnt = 0;
    for (size_t y = 1; y + 1 < 64; ++y) {
      for (size_t x = 1; x + 1 < 96; ++x) {
        for (size_t c = 0; c < 3; ++c) {
          double d = got[(y * 96 + x) * 3 + c] -
                     ref_8x[((y + dy) * 96 + (x + dx)) * 3 + c];
          s += d * d;
          cnt++;
        }
      }
    }
    return s / cnt;
  };
  const double mse0 = shift_mse(0, 0);
  EXPECT_LT(mse0, shift_mse(1, 0));
  EXPECT_LT(mse0, shift_mse(-1, 0));
  EXPECT_LT(mse0, shift_mse(0, 1));
  EXPECT_LT(mse0, shift_mse(0, -1));
}

TEST(DecodeScaleTest, NativeModularSqueezeResponsive) {
  const size_t W = 512;
  const size_t H = 512;
  auto src = GenerateSmoothTestImage(W, H, 3);
  EncodeConfig cfg;
  cfg.lossless = true;
  cfg.modular = true;
  cfg.responsive = 1;
  cfg.progressive_ac = 1;
  auto jxl = EncodeJxl(src.data(), src.size(), W, H, 3, JXL_TYPE_UINT8, cfg);

  auto full = DecodeScaled(jxl, 1, JXL_EXP_SCALE_FAST_APPROXIMATE, 3,
                           JXL_TYPE_FLOAT);
  ASSERT_EQ(JXL_DEC_SUCCESS, full.final_status);
  auto full_f = BufferToFloats(full.pixels, JXL_TYPE_FLOAT);

  for (uint32_t factor : {2u, 4u, 8u}) {
    auto scaled = DecodeScaled(jxl, factor, JXL_EXP_SCALE_REQUIRE_FAST_NATIVE,
                               3, JXL_TYPE_FLOAT, /*sparse=*/true);
    ASSERT_EQ(JXL_DEC_SUCCESS, scaled.final_status) << "factor=" << factor;
    EXPECT_EQ(W / factor, scaled.out_xsize);
    EXPECT_EQ(H / factor, scaled.out_ysize);
    EXPECT_EQ(JXL_TRUE, scaled.fast_native_available);
    EXPECT_EQ(JXL_EXP_SCALE_MODE_MODULAR_SQUEEZE_NATIVE,
              scaled.stats.scale_mode);
    EXPECT_LT(scaled.stats.required_passes, scaled.stats.total_passes);
    if (factor == 8u) {
      EXPECT_EQ(0u, scaled.stats.required_passes);
      EXPECT_EQ(0u, scaled.stats.required_groups);
    }
    EXPECT_LT(scaled.bytes_read, jxl.size()) << "factor=" << factor;

    auto ref = ReferenceBoxDownsampleFloat(full_f.data(), W, H, 3, factor);
    auto got = BufferToFloats(scaled.pixels, JXL_TYPE_FLOAT);
    auto m = ComputeMetrics(got, ref, 1.0);
    EXPECT_GT(m.psnr, 28.0) << "factor=" << factor;
  }
}

TEST(DecodeScaleTest, ProgressiveVarDCTPassSkipping) {
  const size_t W = 768;
  const size_t H = 512;
  auto src = GenerateSmoothTestImage(W, H, 3);
  EncodeConfig cfg;
  cfg.distance = 1.0f;
  cfg.progressive_ac = 1;
  cfg.qprogressive_ac = 1;
  auto jxl = EncodeJxl(src.data(), src.size(), W, H, 3, JXL_TYPE_UINT8, cfg);

  auto full = DecodeScaled(jxl, 1, JXL_EXP_SCALE_FAST_APPROXIMATE, 3,
                           JXL_TYPE_FLOAT);
  ASSERT_EQ(JXL_DEC_SUCCESS, full.final_status);
  auto full_f = BufferToFloats(full.pixels, JXL_TYPE_FLOAT);

  for (uint32_t factor : {2u, 4u}) {
    auto scaled = DecodeScaled(jxl, factor, JXL_EXP_SCALE_FAST_APPROXIMATE, 3,
                               JXL_TYPE_FLOAT, /*sparse=*/true);
    ASSERT_EQ(JXL_DEC_SUCCESS, scaled.final_status) << "factor=" << factor;
    EXPECT_EQ(W / factor, scaled.out_xsize);
    EXPECT_EQ(H / factor, scaled.out_ysize);
    if (scaled.stats.total_passes > 1 &&
        scaled.stats.scale_mode == JXL_EXP_SCALE_MODE_PROGRESSIVE_PASS_SKIP) {
      EXPECT_LT(scaled.stats.required_passes, scaled.stats.total_passes);
      EXPECT_LT(scaled.bytes_read, jxl.size());
    }
    auto ref = ReferenceBoxDownsampleFloat(full_f.data(), W, H, 3, factor);
    auto got = BufferToFloats(scaled.pixels, JXL_TYPE_FLOAT);
    auto m = ComputeMetrics(got, ref, 1.0);
    EXPECT_GT(m.psnr, 30.0) << "factor=" << factor;
  }
}

TEST(DecodeScaleTest, ReferenceResamplePolicyOverride) {
  const size_t W = 512;
  const size_t H = 512;
  auto src = GenerateSmoothTestImage(W, H, 3);

  // VarDCT 1:8 with REFERENCE_RESAMPLE must NOT use native 1:8 DC.
  {
    EncodeConfig cfg;
    cfg.distance = 1.0f;
    auto jxl = EncodeJxl(src.data(), src.size(), W, H, 3, JXL_TYPE_UINT8, cfg);
    auto full = DecodeScaled(jxl, 1, JXL_EXP_SCALE_FAST_APPROXIMATE, 3,
                             JXL_TYPE_FLOAT);
    ASSERT_EQ(JXL_DEC_SUCCESS, full.final_status);
    auto full_f = BufferToFloats(full.pixels, JXL_TYPE_FLOAT);
    auto ref = ReferenceBoxDownsampleFloat(full_f.data(), W, H, 3, 8);

    auto scaled = DecodeScaled(jxl, 8, JXL_EXP_SCALE_REFERENCE_RESAMPLE, 3,
                               JXL_TYPE_FLOAT);
    ASSERT_EQ(JXL_DEC_SUCCESS, scaled.final_status);
    EXPECT_EQ(JXL_EXP_SCALE_MODE_STREAMING_DOWNSAMPLE_FALLBACK,
              scaled.stats.scale_mode);
    auto got = BufferToFloats(scaled.pixels, JXL_TYPE_FLOAT);
    auto m = ComputeMetrics(got, ref, 1.0);
    EXPECT_LT(m.max_abs_err, 1e-5);
  }

  // Modular Squeeze 1:4 with REFERENCE_RESAMPLE must NOT use partial Squeeze.
  {
    EncodeConfig cfg;
    cfg.lossless = true;
    cfg.modular = true;
    cfg.responsive = 1;
    auto jxl = EncodeJxl(src.data(), src.size(), W, H, 3, JXL_TYPE_UINT8, cfg);
    auto full = DecodeScaled(jxl, 1, JXL_EXP_SCALE_FAST_APPROXIMATE, 3,
                             JXL_TYPE_FLOAT);
    ASSERT_EQ(JXL_DEC_SUCCESS, full.final_status);
    auto full_f = BufferToFloats(full.pixels, JXL_TYPE_FLOAT);
    auto ref = ReferenceBoxDownsampleFloat(full_f.data(), W, H, 3, 4);

    auto scaled = DecodeScaled(jxl, 4, JXL_EXP_SCALE_REFERENCE_RESAMPLE, 3,
                               JXL_TYPE_FLOAT);
    ASSERT_EQ(JXL_DEC_SUCCESS, scaled.final_status);
    EXPECT_EQ(JXL_EXP_SCALE_MODE_STREAMING_DOWNSAMPLE_FALLBACK,
              scaled.stats.scale_mode);
    auto got = BufferToFloats(scaled.pixels, JXL_TYPE_FLOAT);
    auto m = ComputeMetrics(got, ref, 1.0);
    EXPECT_LT(m.max_abs_err, 1e-5);
  }
}

std::vector<float> ApplyOrientationFloat(const std::vector<float>& src,
                                         size_t w, size_t h, size_t channels,
                                         int orient) {
  const bool flip_x = (orient == 2 || orient == 3 || orient == 7 || orient == 8);
  const bool flip_y = (orient == 3 || orient == 4 || orient == 6 || orient == 7);
  const bool transpose = (orient >= 5);
  const size_t out_w = transpose ? h : w;
  const size_t out_h = transpose ? w : h;
  std::vector<float> dst(out_w * out_h * channels);
  for (size_t y = 0; y < h; ++y) {
    const size_t ty = flip_y ? (h - 1 - y) : y;
    for (size_t x = 0; x < w; ++x) {
      const size_t tx = flip_x ? (w - 1 - x) : x;
      const size_t ox = transpose ? ty : tx;
      const size_t oy = transpose ? tx : ty;
      for (size_t c = 0; c < channels; ++c) {
        dst[(oy * out_w + ox) * channels + c] =
            src[(y * w + x) * channels + c];
      }
    }
  }
  return dst;
}

TEST(DecodeScaleTest, OddDimensionsAndAllEightOrientations) {
  const size_t W = 517;
  const size_t H = 389;
  auto src = GenerateSmoothTestImage(W, H, 3);

  for (int orient = 1; orient <= 8; ++orient) {
    EncodeConfig cfg;
    cfg.distance = 1.0f;
    cfg.orientation = static_cast<JxlOrientation>(orient);
    auto jxl = EncodeJxl(src.data(), src.size(), W, H, 3, JXL_TYPE_UINT8, cfg);

    const size_t oriented_w = (orient > 4) ? H : W;
    const size_t oriented_h = (orient > 4) ? W : H;
    const size_t expected_sw = (oriented_w + 3) / 4;
    const size_t expected_sh = (oriented_h + 3) / 4;

    // Decode unoriented full-res, box-downsample in unoriented space, then
    // apply the EXIF orientation — exact mathematical specification of the
    // streaming box downsampler on odd dimensions.
    auto full_unoriented = DecodeScaled(
        jxl, 1, JXL_EXP_SCALE_FAST_APPROXIMATE, 3, JXL_TYPE_FLOAT,
        /*sparse=*/false, /*use_region=*/false, 0, 0, 0, 0,
        /*use_callback=*/false, /*unpremul_alpha=*/false,
        /*keep_orientation=*/true);
    ASSERT_EQ(JXL_DEC_SUCCESS, full_unoriented.final_status);
    auto full_f = BufferToFloats(full_unoriented.pixels, JXL_TYPE_FLOAT);
    auto ref_unoriented = ReferenceBoxDownsampleFloat(full_f.data(), W, H, 3, 4);
    auto ref = ApplyOrientationFloat(ref_unoriented, (W + 3) / 4, (H + 3) / 4,
                                     3, orient);
    std::vector<float> ref_clamped(ref.size());
    for (size_t i = 0; i < ref.size(); ++i) {
      ref_clamped[i] = std::clamp(ref[i], 0.0f, 1.0f);
    }

    for (JxlDataType dt : {JXL_TYPE_FLOAT, JXL_TYPE_FLOAT16, JXL_TYPE_UINT8}) {
      auto scaled = DecodeScaled(jxl, 4, JXL_EXP_SCALE_REFERENCE_RESAMPLE, 3,
                                 dt);
      ASSERT_EQ(JXL_DEC_SUCCESS, scaled.final_status)
          << "orient=" << orient << " dt=" << dt;
      EXPECT_EQ(expected_sw, scaled.out_xsize);
      EXPECT_EQ(expected_sh, scaled.out_ysize);
      auto got = BufferToFloats(scaled.pixels, dt);
      if (dt == JXL_TYPE_FLOAT) {
        auto m = ComputeMetrics(got, ref, 1.0);
        EXPECT_LT(m.max_abs_err, 1e-5) << "orient=" << orient;
      } else if (dt == JXL_TYPE_FLOAT16) {
        auto m = ComputeMetrics(got, ref, 1.0);
        EXPECT_LT(m.max_abs_err, 2e-3) << "orient=" << orient;
      } else {
        auto m = ComputeMetrics(got, ref_clamped, 1.0);
        EXPECT_LT(m.max_abs_err, 2.0 / 255.0) << "orient=" << orient;
      }
    }
  }
}

TEST(DecodeScaleTest, CallbackAndAlphaOutput) {
  const size_t W = 384;
  const size_t H = 256;
  auto src = GenerateSmoothTestImage(W, H, 4);
  EncodeConfig cfg;
  cfg.distance = 1.0f;
  auto jxl = EncodeJxl(src.data(), src.size(), W, H, 4, JXL_TYPE_UINT8, cfg);

  auto direct = DecodeScaled(jxl, 4, JXL_EXP_SCALE_REFERENCE_RESAMPLE, 4,
                             JXL_TYPE_FLOAT, /*sparse=*/false,
                             /*use_region=*/false, 0, 0, 0, 0,
                             /*use_callback=*/false);
  auto via_cb = DecodeScaled(jxl, 4, JXL_EXP_SCALE_REFERENCE_RESAMPLE, 4,
                             JXL_TYPE_FLOAT, /*sparse=*/false,
                             /*use_region=*/false, 0, 0, 0, 0,
                             /*use_callback=*/true);
  ASSERT_EQ(JXL_DEC_SUCCESS, direct.final_status);
  ASSERT_EQ(JXL_DEC_SUCCESS, via_cb.final_status);
  EXPECT_EQ(direct.pixels, via_cb.pixels);
}

TEST(DecodeScaleTest, CombinedRoiAndDownsamplingSparse) {
  const size_t W = 1024;
  const size_t H = 768;
  auto src = GenerateSmoothTestImage(W, H, 3);
  EncodeConfig cfg;
  cfg.distance = 1.0f;
  auto jxl = EncodeJxl(src.data(), src.size(), W, H, 3, JXL_TYPE_UINT8, cfg);

  // Full downscaled image at factor 2 (512 x 384).
  auto full_2x = DecodeScaled(jxl, 2, JXL_EXP_SCALE_REFERENCE_RESAMPLE, 3,
                              JXL_TYPE_FLOAT);
  ASSERT_EQ(JXL_DEC_SUCCESS, full_2x.final_status);
  ASSERT_EQ(512u, full_2x.out_xsize);
  ASSERT_EQ(384u, full_2x.out_ysize);

  // ROI in scaled coordinates: (150, 100, 128, 96) inside 512 x 384.
  const uint32_t rx = 150, ry = 100, rw = 128, rh = 96;
  auto roi_2x = DecodeScaled(jxl, 2, JXL_EXP_SCALE_REFERENCE_RESAMPLE, 3,
                             JXL_TYPE_FLOAT, /*sparse=*/true,
                             /*use_region=*/true, rx, ry, rw, rh);
  ASSERT_EQ(JXL_DEC_SUCCESS, roi_2x.final_status);
  EXPECT_EQ(rw, roi_2x.out_xsize);
  EXPECT_EQ(rh, roi_2x.out_ysize);
  EXPECT_GT(roi_2x.stats.skipped_groups, 0u);
  EXPECT_LT(roi_2x.bytes_read, jxl.size());

  // Bit-exact comparison against cropping full_2x.
  const size_t row_bytes = rw * 3 * sizeof(float);
  const size_t full_row_bytes = 512 * 3 * sizeof(float);
  for (uint32_t y = 0; y < rh; ++y) {
    const uint8_t* expected =
        full_2x.pixels.data() + (ry + y) * full_row_bytes +
        rx * 3 * sizeof(float);
    const uint8_t* actual = roi_2x.pixels.data() + y * row_bytes;
    ASSERT_EQ(0, memcmp(expected, actual, row_bytes)) << "row " << y;
  }
}

TEST(DecodeScaleTest, Float2kTo512SparseDownscale) {
  const size_t W = 2048;
  const size_t H = 1080;
  auto src = GenerateHdrHalfImage(W, H);
  EncodeConfig cfg;
  cfg.distance = 0.5f;
  cfg.effort = 3;
  auto jxl = EncodeJxl(src.data(), src.size() * sizeof(uint16_t), W, H, 3,
                       JXL_TYPE_FLOAT16, cfg, /*linear_ap1=*/true);

  auto full = DecodeScaled(jxl, 1, JXL_EXP_SCALE_FAST_APPROXIMATE, 3,
                           JXL_TYPE_FLOAT);
  ASSERT_EQ(JXL_DEC_SUCCESS, full.final_status);
  auto full_f = BufferToFloats(full.pixels, JXL_TYPE_FLOAT);

  // 2K -> 256x135 (1:8 native VarDCT DC)
  auto scaled_8x = DecodeScaled(jxl, 8, JXL_EXP_SCALE_REQUIRE_FAST_NATIVE, 3,
                                JXL_TYPE_FLOAT16, /*sparse=*/true);
  ASSERT_EQ(JXL_DEC_SUCCESS, scaled_8x.final_status);
  EXPECT_EQ(256u, scaled_8x.out_xsize);
  EXPECT_EQ(135u, scaled_8x.out_ysize);
  EXPECT_EQ(JXL_EXP_SCALE_MODE_NATIVE_VARDCT_DC, scaled_8x.stats.scale_mode);
  EXPECT_LT(scaled_8x.bytes_read, jxl.size() / 4);

  auto got_8x = BufferToFloats(scaled_8x.pixels, JXL_TYPE_FLOAT16);
  auto ref_8x = ReferenceBoxDownsampleFloat(full_f.data(), W, H, 3, 8);
  float max_val = *std::max_element(got_8x.begin(), got_8x.end());
  EXPECT_GT(max_val, 3.0f);  // Highlights > 1.0 preserved!
  auto m8 = ComputeMetrics(got_8x, ref_8x, 4.0);
  EXPECT_GT(m8.psnr, 28.0);

  // 2K -> 512x270 (1:4 reference resample)
  auto scaled_4x = DecodeScaled(jxl, 4, JXL_EXP_SCALE_REFERENCE_RESAMPLE, 3,
                                JXL_TYPE_FLOAT16);
  ASSERT_EQ(JXL_DEC_SUCCESS, scaled_4x.final_status);
  EXPECT_EQ(512u, scaled_4x.out_xsize);
  EXPECT_EQ(270u, scaled_4x.out_ysize);
  auto got_4x = BufferToFloats(scaled_4x.pixels, JXL_TYPE_FLOAT16);
  auto ref_4x = ReferenceBoxDownsampleFloat(full_f.data(), W, H, 3, 4);
  auto m4 = ComputeMetrics(got_4x, ref_4x, 4.0);
  EXPECT_LT(m4.max_abs_err, 5e-3);
}

TEST(DecodeScaleTest, DISABLED_Float8kTo2kScaleBenchmark) {
  const size_t W = 8192;
  const size_t H = 4320;
  auto src = GenerateHdrHalfImage(W, H);
  EncodeConfig cfg;
  cfg.distance = 0.5f;
  cfg.effort = 1;
  cfg.progressive_ac = 1;
  auto jxl = EncodeJxl(src.data(), src.size() * sizeof(uint16_t), W, H, 3,
                       JXL_TYPE_FLOAT16, cfg, /*linear_ap1=*/true);

  auto t0 = std::chrono::steady_clock::now();
  auto full = DecodeScaled(jxl, 1, JXL_EXP_SCALE_FAST_APPROXIMATE, 3,
                           JXL_TYPE_FLOAT16);
  auto t1 = std::chrono::steady_clock::now();
  ASSERT_EQ(JXL_DEC_SUCCESS, full.final_status);

  auto t2 = std::chrono::steady_clock::now();
  auto s4 = DecodeScaled(jxl, 4, JXL_EXP_SCALE_FAST_APPROXIMATE, 3,
                         JXL_TYPE_FLOAT16, /*sparse=*/true);
  auto t3 = std::chrono::steady_clock::now();
  ASSERT_EQ(JXL_DEC_SUCCESS, s4.final_status);

  auto t4 = std::chrono::steady_clock::now();
  auto s8 = DecodeScaled(jxl, 8, JXL_EXP_SCALE_REQUIRE_FAST_NATIVE, 3,
                         JXL_TYPE_FLOAT16, /*sparse=*/true);
  auto t5 = std::chrono::steady_clock::now();
  ASSERT_EQ(JXL_DEC_SUCCESS, s8.final_status);

  const double ms_full =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double ms_4x =
      std::chrono::duration<double, std::milli>(t3 - t2).count();
  const double ms_8x =
      std::chrono::duration<double, std::milli>(t5 - t4).count();
  std::printf(
      "[8K Benchmark] total=%zu B | full=%.1f ms | 1:4(2K)=%.1f ms (%llu B, "
      "mode=%d) | 1:8(1K)=%.1f ms (%llu B, mode=%d)\n",
      jxl.size(), ms_full, ms_4x,
      static_cast<unsigned long long>(s4.bytes_read),
      static_cast<int>(s4.stats.scale_mode), ms_8x,
      static_cast<unsigned long long>(s8.bytes_read),
      static_cast<int>(s8.stats.scale_mode));
}

}  // namespace
}  // namespace jxl
