// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <fcntl.h>
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
#include <jxl/version.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
    return static_cast<uint16_t>(sign | 0x7BFFu);
  }
  return static_cast<uint16_t>(sign | ((exp + 15) << 10) |
                               ((mant + 0x1000u) >> 13));
}

std::vector<uint8_t> GenerateSmoothRgb8(size_t width, size_t height,
                                        uint32_t seed = 42) {
  std::vector<uint8_t> pixels(width * height * 3);
  for (size_t y = 0; y < height; ++y) {
    const float fy = static_cast<float>(y) / std::max<size_t>(1, height - 1);
    for (size_t x = 0; x < width; ++x) {
      const float fx = static_cast<float>(x) / std::max<size_t>(1, width - 1);
      const size_t idx = (y * width + x) * 3;
      for (uint32_t c = 0; c < 3; ++c) {
        float wave = 0.5f +
                     0.25f * std::sin(fx * (14.0f + c * 5.0f) + seed * 0.1f) +
                     0.20f * std::cos(fy * (10.0f + c * 4.0f)) +
                     0.05f * std::sin((fx + fy) * 45.0f);
        pixels[idx + c] =
            static_cast<uint8_t>(std::clamp(wave * 255.0f, 0.0f, 255.0f));
      }
    }
  }
  return pixels;
}

std::vector<uint16_t> GenerateHdrHalfRgb(size_t width, size_t height,
                                         uint32_t seed = 17) {
  std::vector<uint16_t> pixels(width * height * 3);
  for (size_t y = 0; y < height; ++y) {
    const float fy = static_cast<float>(y) / std::max<size_t>(1, height - 1);
    for (size_t x = 0; x < width; ++x) {
      const float fx = static_cast<float>(x) / std::max<size_t>(1, width - 1);
      const size_t idx = (y * width + x) * 3;
      float r = -0.25f + 3.8f * fx + 0.6f * std::sin(fy * 14.0f + seed);
      float g = 0.05f + 2.8f * fy + 0.5f * std::cos(fx * 12.0f);
      float b = -0.15f + 4.4f * (0.5f * fx + 0.5f * fy);
      pixels[idx + 0] = FloatToHalf(r);
      pixels[idx + 1] = FloatToHalf(g);
      pixels[idx + 2] = FloatToHalf(b);
    }
  }
  return pixels;
}

struct EncodeOpts {
  bool lossless = false;
  float distance = 1.0f;
  int effort = 3;
  bool modular = false;
  int responsive = -1;
  int progressive_ac = -1;
  int qprogressive_ac = -1;
  bool use_container = false;
  int output_mode = -1;  // -1 = default, 0/1/2
};

std::vector<uint8_t> EncodeJxl(const void* pixels, size_t pixel_bytes,
                               size_t width, size_t height,
                               JxlDataType data_type, const EncodeOpts& opts,
                               bool linear_ap1 = false) {
  auto enc = JxlEncoderMake(nullptr);
  EXPECT_NE(nullptr, enc.get());
  auto runner = JxlThreadParallelRunnerMake(
      nullptr, JxlThreadParallelRunnerDefaultNumWorkerThreads());
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderSetParallelRunner(enc.get(), JxlThreadParallelRunner,
                                        runner.get()));
  if (opts.use_container || opts.output_mode == 2) {
    EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderUseContainer(enc.get(), JXL_TRUE));
  }

  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.xsize = static_cast<uint32_t>(width);
  info.ysize = static_cast<uint32_t>(height);
  info.num_color_channels = 3;
  info.bits_per_sample =
      (data_type == JXL_TYPE_UINT8 ? 8
                                   : (data_type == JXL_TYPE_FLOAT16 ? 16 : 32));
  info.exponent_bits_per_sample =
      (data_type == JXL_TYPE_FLOAT16 ? 5
                                     : (data_type == JXL_TYPE_FLOAT ? 8 : 0));
  info.uses_original_profile =
      (opts.lossless || linear_ap1 || data_type != JXL_TYPE_UINT8) ? JXL_TRUE
                                                                   : JXL_FALSE;
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetBasicInfo(enc.get(), &info));

  JxlColorEncoding color = {};
  if (linear_ap1 || data_type != JXL_TYPE_UINT8) {
    color.color_space = JXL_COLOR_SPACE_RGB;
    color.white_point = JXL_WHITE_POINT_CUSTOM;
    color.white_point_xy[0] = 0.32168;
    color.white_point_xy[1] = 0.33767;
    color.primaries = JXL_PRIMARIES_CUSTOM;
    color.primaries_red_xy[0] = 0.713;
    color.primaries_red_xy[1] = 0.293;
    color.primaries_green_xy[0] = 0.165;
    color.primaries_green_xy[1] = 0.830;
    color.primaries_blue_xy[0] = 0.128;
    color.primaries_blue_xy[1] = 0.044;
    color.transfer_function = JXL_TRANSFER_FUNCTION_LINEAR;
    color.rendering_intent = JXL_RENDERING_INTENT_RELATIVE;
  } else {
    JxlColorEncodingSetToSRGB(&color, JXL_FALSE);
  }
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetColorEncoding(enc.get(), &color));

  JxlEncoderFrameSettings* fs = JxlEncoderFrameSettingsCreate(enc.get(), nullptr);
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderFrameSettingsSetOption(
                fs, JXL_ENC_FRAME_SETTING_EFFORT, opts.effort));
  if (opts.lossless) {
    EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetFrameLossless(fs, JXL_TRUE));
  } else {
    EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetFrameDistance(fs, opts.distance));
  }
  if (opts.modular) {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderFrameSettingsSetOption(
                  fs, JXL_ENC_FRAME_SETTING_MODULAR, 1));
  }
  if (opts.responsive >= 0) {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderFrameSettingsSetOption(
                  fs, JXL_ENC_FRAME_SETTING_RESPONSIVE, opts.responsive));
  }
  if (opts.progressive_ac >= 0) {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderFrameSettingsSetOption(
                  fs, JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC,
                  opts.progressive_ac));
  }
  if (opts.qprogressive_ac >= 0) {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderFrameSettingsSetOption(
                  fs, JXL_ENC_FRAME_SETTING_QPROGRESSIVE_AC,
                  opts.qprogressive_ac));
  }
#if JPEGXL_NUMERIC_VERSION >= JPEGXL_COMPUTE_NUMERIC_VERSION(0, 12, 0)
  if (opts.output_mode >= 0) {
    EXPECT_EQ(JXL_ENC_SUCCESS,
              JxlEncoderFrameSettingsSetOption(
                  fs, JXL_ENC_FRAME_SETTING_OUTPUT_MODE, opts.output_mode));
  }
#endif

  JxlPixelFormat fmt = {3, data_type, JXL_NATIVE_ENDIAN, 0};
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderAddImageFrame(fs, &fmt, pixels, pixel_bytes));
  JxlEncoderCloseInput(enc.get());

  std::vector<uint8_t> out(65536);
  uint8_t* next_out = out.data();
  size_t avail_out = out.size();
  for (;;) {
    JxlEncoderStatus st = JxlEncoderProcessOutput(enc.get(), &next_out, &avail_out);
    if (st == JXL_ENC_NEED_MORE_OUTPUT) {
      size_t off = next_out - out.data();
      out.resize(out.size() * 2);
      next_out = out.data() + off;
      avail_out = out.size() - off;
    } else {
      EXPECT_EQ(JXL_ENC_SUCCESS, st);
      out.resize(next_out - out.data());
      break;
    }
  }
  return out;
}

// Wraps a bare codestream into a multi-jxlp container with fixed chunk size.
std::vector<uint8_t> WrapInMultiJxlp(const std::vector<uint8_t>& codestream,
                                     size_t chunk_size) {
  std::vector<uint8_t> out;
  auto push_be32 = [&](uint32_t v) {
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>(v & 0xFF));
  };
  // JXL signature box (12 bytes)
  const uint8_t kSig[12] = {0x00, 0x00, 0x00, 0x0C, 'J',  'X',
                            'L',  ' ',  0x0D, 0x0A, 0x87, 0x0A};
  out.insert(out.end(), kSig, kSig + 12);
  // ftyp box (20 bytes)
  push_be32(20);
  const char kFtyp[16] = {'f', 't', 'y', 'p', 'j', 'x', 'l', ' ',
                          0,   0,   0,   0,   'j', 'x', 'l', ' '};
  out.insert(out.end(), kFtyp, kFtyp + 16);

  uint32_t idx = 0;
  for (size_t pos = 0; pos < codestream.size(); pos += chunk_size, ++idx) {
    const size_t len = std::min(chunk_size, codestream.size() - pos);
    const bool is_last = (pos + len == codestream.size());
    push_be32(static_cast<uint32_t>(8 + 4 + len));
    out.insert(out.end(), {'j', 'x', 'l', 'p'});
    push_be32(idx | (is_last ? 0x80000000u : 0u));
    out.insert(out.end(), codestream.begin() + pos,
               codestream.begin() + pos + len);
  }
  return out;
}

enum class TransportMode {
  kFullBuffer,
  kSequentialChunked,
  kSparseBatch,
  kSparseBatchReversed,
  kSparseBatchOverlapping,
  kSparseNextOffset,  // #5143 style single-range loop
};

struct StreamTarget {
  uint32_t downsampling = 1;
  JxlExperimentalScalePolicy policy = JXL_EXP_SCALE_FAST_APPROXIMATE;
  bool use_region = false;
  uint32_t rx = 0, ry = 0, rw = 0, rh = 0;
};

struct StreamResult {
  JxlDecoderStatus status = JXL_DEC_ERROR;
  std::vector<uint8_t> pixels;
  uint32_t out_w = 0;
  uint32_t out_h = 0;
  JxlExperimentalRegionDecodeStats stats = {};
  uint64_t bytes_fed = 0;
  size_t range_calls = 0;
};

StreamResult DecodeWithTransport(const std::vector<uint8_t>& data,
                                 const StreamTarget& target,
                                 JxlDataType data_type,
                                 TransportMode transport,
                                 size_t chunk_size = 4096) {
  StreamResult res;
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
  if (target.downsampling > 1 ||
      target.policy != JXL_EXP_SCALE_FAST_APPROXIMATE) {
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderExperimentalSetOutputDownsampling(
                  dec.get(), target.downsampling, target.policy));
  }
  if (target.use_region) {
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderExperimentalSetOutputRegion(
                  dec.get(), target.rx, target.ry, target.rw, target.rh));
  }

  JxlPixelFormat fmt = {3, data_type, JXL_NATIVE_ENDIAN, 0};
  const bool is_sparse = (transport != TransportMode::kFullBuffer &&
                          transport != TransportMode::kSequentialChunked);
  size_t seq_pos = 0;
  bool sparse_started = false;

  if (transport == TransportMode::kFullBuffer) {
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderSetInput(dec.get(), data.data(), data.size()));
    JxlDecoderCloseInput(dec.get());
    res.bytes_fed = data.size();
  } else {
    size_t step = (transport == TransportMode::kSequentialChunked)
                      ? std::min(chunk_size, data.size())
                      : std::min<size_t>(4096, data.size());
    EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderSetInput(dec.get(), data.data(), step));
    seq_pos = step;
    res.bytes_fed = step;
  }

  auto feed_sparse_ranges = [&]() -> bool {
    if (transport == TransportMode::kSparseNextOffset) {
      uint64_t off = 0, len = 0;
      JxlDecoderStatus st =
          JxlDecoderExperimentalGetNextRequiredOffset(dec.get(), &off, &len);
      if (st != JXL_DEC_SUCCESS || len == 0) return false;
      EXPECT_LE(off + len, data.size());
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderExperimentalSetInputAtOffset(dec.get(),
                                                       data.data() + off, len,
                                                       off));
      res.bytes_fed += len;
      res.range_calls++;
      return true;
    }
    size_t count = 0;
    JxlDecoderStatus st = JxlDecoderExperimentalGetRequiredInputRanges(
        dec.get(), nullptr, &count);
    if (st != JXL_DEC_SUCCESS || count == 0) return false;
    std::vector<JxlExperimentalInputRange> ranges(count);
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderExperimentalGetRequiredInputRanges(
                  dec.get(), ranges.data(), &count));
    if (transport == TransportMode::kSparseBatchReversed) {
      std::reverse(ranges.begin(), ranges.end());
    }
    for (size_t i = 0; i < count; ++i) {
      const uint64_t off = ranges[i].offset;
      const uint64_t len = ranges[i].length;
      EXPECT_LE(off + len, data.size());
      if (transport == TransportMode::kSparseBatchOverlapping && len > 8) {
        // Deliver in overlapping halves + duplicate tail.
        const size_t half = static_cast<size_t>(len / 2 + 4);
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderExperimentalSetInputAtOffset(
                      dec.get(), data.data() + off, half, off));
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderExperimentalSetInputAtOffset(
                      dec.get(), data.data() + off + half - 4,
                      static_cast<size_t>(len - half + 4), off + half - 4));
        res.bytes_fed += len + 4;
        res.range_calls += 2;
      } else {
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderExperimentalSetInputAtOffset(
                      dec.get(), data.data() + off, static_cast<size_t>(len),
                      off));
        res.bytes_fed += len;
        res.range_calls++;
      }
    }
    return true;
  };

  for (;;) {
    JxlDecoderStatus st = JxlDecoderProcessInput(dec.get());
    if (st == JXL_DEC_ERROR) {
      res.status = JXL_DEC_ERROR;
      return res;
    } else if (st == JXL_DEC_NEED_MORE_INPUT) {
      if (transport == TransportMode::kFullBuffer) {
        res.status = JXL_DEC_NEED_MORE_INPUT;
        return res;
      }
      if (!sparse_started) {
        size_t rem = JxlDecoderReleaseInput(dec.get());
        size_t consumed = seq_pos - rem;
        size_t step = (transport == TransportMode::kSequentialChunked)
                          ? chunk_size
                          : 4096;
        size_t avail = data.size() - seq_pos;
        if (avail == 0) {
          JxlDecoderCloseInput(dec.get());
          res.status = JXL_DEC_NEED_MORE_INPUT;
          return res;
        }
        size_t add = std::min(step, avail);
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderSetInput(dec.get(), data.data() + consumed,
                                     rem + add));
        seq_pos += add;
        res.bytes_fed += add;
      } else {
        if (!feed_sparse_ranges()) {
          res.status = JXL_DEC_NEED_MORE_INPUT;
          return res;
        }
      }
    } else if (st == JXL_DEC_BASIC_INFO) {
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderExperimentalGetScaledOutputInfo(
                    dec.get(), &res.out_w, &res.out_h, nullptr, nullptr));
    } else if (st == JXL_DEC_FRAME) {
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderExperimentalGetScaledOutputInfo(
                    dec.get(), &res.out_w, &res.out_h, nullptr, nullptr));
      if (is_sparse) {
        size_t rem = JxlDecoderReleaseInput(dec.get());
        res.bytes_fed -= rem;
        sparse_started = true;
        feed_sparse_ranges();
      }
    } else if (st == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t buf_size = 0;
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderImageOutBufferSize(dec.get(), &fmt, &buf_size));
      res.pixels.resize(buf_size);
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderSetImageOutBuffer(dec.get(), &fmt, res.pixels.data(),
                                            res.pixels.size()));
    } else if (st == JXL_DEC_FULL_IMAGE) {
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderExperimentalGetRegionDecodeStats(dec.get(),
                                                           &res.stats));
      res.status = JXL_DEC_SUCCESS;
      return res;
    } else if (st == JXL_DEC_SUCCESS) {
      res.status = JXL_DEC_SUCCESS;
      return res;
    }
  }
}

void ExpectFloatPixelsMatch(const std::vector<uint8_t>& a,
                            const std::vector<uint8_t>& b,
                            JxlExperimentalScaleDecodeMode mode,
                            const char* ctx = "") {
  ASSERT_EQ(a.size(), b.size()) << ctx;
  const bool uses_mt_float_accum =
      (mode == JXL_EXP_SCALE_MODE_PROGRESSIVE_PASS_SKIP ||
       mode == JXL_EXP_SCALE_MODE_STREAMING_DOWNSAMPLE_FALLBACK);
  if (!uses_mt_float_accum) {
    EXPECT_EQ(a, b) << ctx;
    return;
  }
  const float* fa = reinterpret_cast<const float*>(a.data());
  const float* fb = reinterpret_cast<const float*>(b.data());
  const size_t n = a.size() / sizeof(float);
  float max_err = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    max_err = std::max(max_err, std::abs(fa[i] - fb[i]));
  }
  EXPECT_LT(max_err, 1e-6f) << ctx;
}

TEST(DecodeStreamingTest, FourByTwoStreamingMatrixVarDCT) {
  const size_t W = 768;
  const size_t H = 512;
  auto src = GenerateSmoothRgb8(W, H);
  EncodeOpts opts;
  opts.distance = 1.0f;
  opts.progressive_ac = 1;
  opts.qprogressive_ac = 1;
  auto jxl = EncodeJxl(src.data(), src.size(), W, H, JXL_TYPE_UINT8, opts);

  const std::vector<StreamTarget> targets = {
      // 1. Full
      {1, JXL_EXP_SCALE_FAST_APPROXIMATE, false, 0, 0, 0, 0},
      // 2. ROI (full-res)
      {1, JXL_EXP_SCALE_FAST_APPROXIMATE, true, 200, 120, 160, 140},
      // 3a. Scaled 1:4 (progressive pass skip)
      {4, JXL_EXP_SCALE_FAST_APPROXIMATE, false, 0, 0, 0, 0},
      // 3b. Scaled 1:8 (native VarDCT DC)
      {8, JXL_EXP_SCALE_REQUIRE_FAST_NATIVE, false, 0, 0, 0, 0},
      // 4a. Scaled-ROI 1:4 (reference resample)
      {4, JXL_EXP_SCALE_REFERENCE_RESAMPLE, true, 40, 30, 64, 48},
      // 4b. Scaled-ROI 1:8 (native VarDCT DC + ROI)
      {8, JXL_EXP_SCALE_REQUIRE_FAST_NATIVE, true, 20, 10, 40, 32},
  };

  for (size_t ti = 0; ti < targets.size(); ++ti) {
    const auto& t = targets[ti];
    auto base = DecodeWithTransport(jxl, t, JXL_TYPE_FLOAT,
                                    TransportMode::kFullBuffer);
    ASSERT_EQ(JXL_DEC_SUCCESS, base.status) << "target=" << ti;

    // Verify partial-fidelity flag matches path semantics (#603).
    const bool expect_partial =
        (base.stats.scale_mode == JXL_EXP_SCALE_MODE_NATIVE_VARDCT_DC ||
         base.stats.scale_mode == JXL_EXP_SCALE_MODE_MODULAR_SQUEEZE_NATIVE ||
         base.stats.scale_mode == JXL_EXP_SCALE_MODE_PROGRESSIVE_PASS_SKIP);
    EXPECT_EQ(expect_partial ? JXL_TRUE : JXL_FALSE,
              base.stats.is_partial_fidelity)
        << "target=" << ti;

    // Sequential chunked streaming across tiny & medium chunk sizes.
    for (size_t chunk : {size_t{1}, size_t{64}, size_t{4096}, size_t{65536}}) {
      auto seq = DecodeWithTransport(jxl, t, JXL_TYPE_FLOAT,
                                     TransportMode::kSequentialChunked, chunk);
      ASSERT_EQ(JXL_DEC_SUCCESS, seq.status)
          << "target=" << ti << " chunk=" << chunk;
      EXPECT_EQ(base.out_w, seq.out_w);
      EXPECT_EQ(base.out_h, seq.out_h);
      ExpectFloatPixelsMatch(base.pixels, seq.pixels, base.stats.scale_mode,
                             "seq");
      if (t.use_region || t.downsampling == 8) {
        EXPECT_GT(seq.stats.bytes_skipped_sequential, 0u) << "target=" << ti;
      }
    }

    // Sparse batch, reversed, overlapping, and #5143 next-offset streaming.
    for (TransportMode tm :
         {TransportMode::kSparseBatch, TransportMode::kSparseBatchReversed,
          TransportMode::kSparseBatchOverlapping,
          TransportMode::kSparseNextOffset}) {
      auto sp = DecodeWithTransport(jxl, t, JXL_TYPE_FLOAT, tm);
      ASSERT_EQ(JXL_DEC_SUCCESS, sp.status)
          << "target=" << ti << " tm=" << static_cast<int>(tm);
      ExpectFloatPixelsMatch(base.pixels, sp.pixels, base.stats.scale_mode,
                             "sparse");
      EXPECT_GT(sp.stats.sparse_bytes_received, 0u) << "target=" << ti;
      EXPECT_LE(sp.stats.sparse_peak_bytes_buffered, jxl.size());
      if (t.use_region || t.downsampling == 8) {
        EXPECT_LT(sp.bytes_fed, jxl.size()) << "target=" << ti;
      }
    }
  }
}

TEST(DecodeStreamingTest, FourByTwoStreamingMatrixModularSqueezeAndLossless) {
  const size_t W = 512;
  const size_t H = 512;
  auto src = GenerateSmoothRgb8(W, H);

  // 1) Modular with Squeeze + progressive passes.
  {
    EncodeOpts opts;
    opts.lossless = true;
    opts.modular = true;
    opts.responsive = 1;
    opts.progressive_ac = 1;
    auto jxl = EncodeJxl(src.data(), src.size(), W, H, JXL_TYPE_UINT8, opts);

    const std::vector<StreamTarget> targets = {
        {1, JXL_EXP_SCALE_FAST_APPROXIMATE, false, 0, 0, 0, 0},
        {4, JXL_EXP_SCALE_REQUIRE_FAST_NATIVE, false, 0, 0, 0, 0},
        {8, JXL_EXP_SCALE_REQUIRE_FAST_NATIVE, false, 0, 0, 0, 0},
        {4, JXL_EXP_SCALE_REQUIRE_FAST_NATIVE, true, 16, 16, 64, 64},
    };
    for (size_t ti = 0; ti < targets.size(); ++ti) {
      auto base = DecodeWithTransport(jxl, targets[ti], JXL_TYPE_FLOAT,
                                      TransportMode::kFullBuffer);
      ASSERT_EQ(JXL_DEC_SUCCESS, base.status);
      auto seq = DecodeWithTransport(jxl, targets[ti], JXL_TYPE_FLOAT,
                                     TransportMode::kSequentialChunked, 512);
      auto sp_batch = DecodeWithTransport(jxl, targets[ti], JXL_TYPE_FLOAT,
                                          TransportMode::kSparseBatch);
      auto sp_next = DecodeWithTransport(jxl, targets[ti], JXL_TYPE_FLOAT,
                                         TransportMode::kSparseNextOffset);
      ASSERT_EQ(JXL_DEC_SUCCESS, seq.status);
      ASSERT_EQ(JXL_DEC_SUCCESS, sp_batch.status);
      ASSERT_EQ(JXL_DEC_SUCCESS, sp_next.status);
      ExpectFloatPixelsMatch(base.pixels, seq.pixels, base.stats.scale_mode);
      ExpectFloatPixelsMatch(base.pixels, sp_batch.pixels,
                             base.stats.scale_mode);
      ExpectFloatPixelsMatch(base.pixels, sp_next.pixels,
                             base.stats.scale_mode);
      if (targets[ti].downsampling > 1) {
        EXPECT_EQ(JXL_TRUE, base.stats.is_partial_fidelity);
        EXPECT_LT(sp_batch.bytes_fed, jxl.size());
      }
    }
  }

  // 2) Adversarial non-Squeeze lossless Modular (responsive = 0).
  {
    EncodeOpts opts;
    opts.lossless = true;
    opts.modular = true;
    opts.responsive = 0;
    auto jxl = EncodeJxl(src.data(), src.size(), W, H, JXL_TYPE_UINT8, opts);

    // Full-res footprint [0, 192) x [0, 192) fits strictly in group 0 of 4.
    StreamTarget roi_scaled = {4, JXL_EXP_SCALE_REFERENCE_RESAMPLE, true,
                               0, 0, 48, 48};
    auto base = DecodeWithTransport(jxl, roi_scaled, JXL_TYPE_FLOAT,
                                    TransportMode::kFullBuffer);
    auto seq = DecodeWithTransport(jxl, roi_scaled, JXL_TYPE_FLOAT,
                                   TransportMode::kSequentialChunked, 512);
    auto sp = DecodeWithTransport(jxl, roi_scaled, JXL_TYPE_FLOAT,
                                  TransportMode::kSparseNextOffset);
    ASSERT_EQ(JXL_DEC_SUCCESS, base.status);
    ASSERT_EQ(JXL_DEC_SUCCESS, seq.status);
    ASSERT_EQ(JXL_DEC_SUCCESS, sp.status);
    EXPECT_EQ(JXL_FALSE, base.stats.is_partial_fidelity);
    ExpectFloatPixelsMatch(base.pixels, seq.pixels, base.stats.scale_mode);
    ExpectFloatPixelsMatch(base.pixels, sp.pixels, base.stats.scale_mode);
    EXPECT_GT(sp.stats.skipped_groups, 0u);
    EXPECT_LT(sp.bytes_fed, jxl.size());
  }
}

TEST(DecodeStreamingTest, ProgressiveFlushStreamingAllModes) {
  const size_t W = 512;
  const size_t H = 512;
  auto src = GenerateSmoothRgb8(W, H);
  EncodeOpts opts;
  opts.distance = 1.0f;
  opts.progressive_ac = 1;
  opts.qprogressive_ac = 1;
  auto jxl = EncodeJxl(src.data(), src.size(), W, H, JXL_TYPE_UINT8, opts);

  const std::vector<StreamTarget> targets = {
      {1, JXL_EXP_SCALE_FAST_APPROXIMATE, false, 0, 0, 0, 0},
      {1, JXL_EXP_SCALE_FAST_APPROXIMATE, true, 128, 128, 128, 128},
      {2, JXL_EXP_SCALE_REFERENCE_RESAMPLE, false, 0, 0, 0, 0},
      {4, JXL_EXP_SCALE_REFERENCE_RESAMPLE, true, 16, 16, 64, 64},
  };

  for (size_t ti = 0; ti < targets.size(); ++ti) {
    const auto& t = targets[ti];
    auto ref = DecodeWithTransport(jxl, t, JXL_TYPE_FLOAT,
                                   TransportMode::kFullBuffer);
    ASSERT_EQ(JXL_DEC_SUCCESS, ref.status);

    auto dec = JxlDecoderMake(nullptr);
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderSubscribeEvents(
                  dec.get(), JXL_DEC_BASIC_INFO | JXL_DEC_FRAME |
                                 JXL_DEC_FRAME_PROGRESSION |
                                 JXL_DEC_FULL_IMAGE));
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderSetProgressiveDetail(dec.get(), kLastPasses));
    if (t.downsampling > 1) {
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderExperimentalSetOutputDownsampling(
                    dec.get(), t.downsampling, t.policy));
    }
    if (t.use_region) {
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderExperimentalSetOutputRegion(dec.get(), t.rx, t.ry,
                                                      t.rw, t.rh));
    }

    JxlPixelFormat fmt = {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
    std::vector<uint8_t> out;
    size_t pos = 0;
    const size_t kStep = 1024;
    size_t first = std::min(kStep, jxl.size());
    EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderSetInput(dec.get(), jxl.data(), first));
    pos = first;
    int flushes = 0;
    bool got_full = false;

    for (;;) {
      JxlDecoderStatus st = JxlDecoderProcessInput(dec.get());
      if (st == JXL_DEC_NEED_MORE_INPUT) {
        size_t rem = JxlDecoderReleaseInput(dec.get());
        size_t consumed = pos - rem;
        size_t add = std::min(kStep, jxl.size() - pos);
        ASSERT_GT(add, 0u);
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderSetInput(dec.get(), jxl.data() + consumed,
                                     rem + add));
        pos += add;
      } else if (st == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
        size_t sz = 0;
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderImageOutBufferSize(dec.get(), &fmt, &sz));
        out.resize(sz);
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderSetImageOutBuffer(dec.get(), &fmt, out.data(),
                                              out.size()));
      } else if (st == JXL_DEC_FRAME_PROGRESSION) {
        EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderFlushImage(dec.get()));
        flushes++;
        // Verify flushed buffer contains finite values (no NaNs/Inf).
        const float* fp = reinterpret_cast<const float*>(out.data());
        for (size_t i = 0; i < out.size() / sizeof(float); ++i) {
          ASSERT_TRUE(std::isfinite(fp[i])) << "ti=" << ti << " idx=" << i;
        }
      } else if (st == JXL_DEC_FULL_IMAGE) {
        got_full = true;
        break;
      } else if (st == JXL_DEC_ERROR) {
        FAIL() << "Unexpected error during progressive flush for target " << ti;
      }
    }
    EXPECT_TRUE(got_full);
    EXPECT_GT(flushes, 0) << "ti=" << ti;
    ExpectFloatPixelsMatch(ref.pixels, out, ref.stats.scale_mode, "flush");
  }
}

TEST(DecodeStreamingTest, ContainerLayoutsAndEncoderOutputModes) {
  const size_t W = 512;
  const size_t H = 384;
  auto src = GenerateSmoothRgb8(W, H);

  // 1) Raw, jxlc, and Encoder OUTPUT_MODE 0, 1, 2.
#if JPEGXL_NUMERIC_VERSION >= JPEGXL_COMPUTE_NUMERIC_VERSION(0, 12, 0)
  for (int mode : {0, 1, 2}) {
    EncodeOpts opts;
    opts.distance = 1.0f;
    opts.output_mode = mode;
    opts.use_container = (mode == 2);
#else
  for (int mode : {0, 2}) {
    EncodeOpts opts;
    opts.distance = 1.0f;
    opts.use_container = (mode == 2);
#endif
    auto jxl = EncodeJxl(src.data(), src.size(), W, H, JXL_TYPE_UINT8, opts);

    const std::vector<StreamTarget> targets = {
        {1, JXL_EXP_SCALE_FAST_APPROXIMATE, false, 0, 0, 0, 0},
        {1, JXL_EXP_SCALE_FAST_APPROXIMATE, true, 64, 64, 128, 128},
        {8, JXL_EXP_SCALE_REQUIRE_FAST_NATIVE, false, 0, 0, 0, 0},
        {4, JXL_EXP_SCALE_REFERENCE_RESAMPLE, true, 16, 16, 48, 48},
    };
    for (const auto& t : targets) {
      auto base = DecodeWithTransport(jxl, t, JXL_TYPE_FLOAT,
                                      TransportMode::kFullBuffer);
      auto seq = DecodeWithTransport(jxl, t, JXL_TYPE_FLOAT,
                                     TransportMode::kSequentialChunked, 512);
      ASSERT_EQ(JXL_DEC_SUCCESS, base.status) << "mode=" << mode;
      ASSERT_EQ(JXL_DEC_SUCCESS, seq.status) << "mode=" << mode;
      ExpectFloatPixelsMatch(base.pixels, seq.pixels, base.stats.scale_mode,
                             "mode");
    }
  }

  // 2) Multi-jxlp container: sequential streaming works on all 4 modes;
  // sparse range mapping explicitly reports error / fallback reason.
  EncodeOpts bare_opts;
  bare_opts.distance = 1.0f;
  auto bare =
      EncodeJxl(src.data(), src.size(), W, H, JXL_TYPE_UINT8, bare_opts);
  auto multi_jxlp = WrapInMultiJxlp(bare, 4096);

  StreamTarget roi_scaled = {4, JXL_EXP_SCALE_REFERENCE_RESAMPLE, true,
                             16, 16, 48, 48};
  auto bare_ref = DecodeWithTransport(bare, roi_scaled, JXL_TYPE_FLOAT,
                                      TransportMode::kFullBuffer);
  auto jxlp_seq = DecodeWithTransport(multi_jxlp, roi_scaled, JXL_TYPE_FLOAT,
                                      TransportMode::kSequentialChunked, 256);
  ASSERT_EQ(JXL_DEC_SUCCESS, bare_ref.status);
  ASSERT_EQ(JXL_DEC_SUCCESS, jxlp_seq.status);
  ExpectFloatPixelsMatch(bare_ref.pixels, jxlp_seq.pixels,
                         bare_ref.stats.scale_mode, "multi_jxlp");
  EXPECT_EQ(JXL_EXP_FALLBACK_MULTI_JXLP_UNKNOWN_BOUNDARIES,
            jxlp_seq.stats.fallback_reason);
}

TEST(DecodeStreamingTest, AdversarialSparseAndModeIsolation) {
  const size_t W = 512;
  const size_t H = 512;
  auto src = GenerateSmoothRgb8(W, H);
  EncodeOpts opts;
  opts.distance = 1.0f;
  auto jxl = EncodeJxl(src.data(), src.size(), W, H, JXL_TYPE_UINT8, opts);

  auto dec = JxlDecoderMake(nullptr);
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSubscribeEvents(
                dec.get(),
                JXL_DEC_BASIC_INFO | JXL_DEC_FRAME | JXL_DEC_FULL_IMAGE));
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderExperimentalSetOutputRegion(dec.get(), 64, 64, 128, 128));
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSetInput(dec.get(), jxl.data(),
                               std::min<size_t>(4096, jxl.size())));

  // Advance to JXL_DEC_FRAME.
  JxlDecoderStatus st = JXL_DEC_NEED_MORE_INPUT;
  while ((st = JxlDecoderProcessInput(dec.get())) != JXL_DEC_FRAME) {
    ASSERT_NE(JXL_DEC_ERROR, st);
  }

  // 1) Calling SetInputAtOffset WITHOUT ReleaseInput must fail.
  EXPECT_EQ(JXL_DEC_ERROR,
            JxlDecoderExperimentalSetInputAtOffset(dec.get(), jxl.data(), 16,
                                                   0));

  JxlDecoderReleaseInput(dec.get());

  // 2) Overflowing offset + size must fail.
  EXPECT_EQ(JXL_DEC_ERROR,
            JxlDecoderExperimentalSetInputAtOffset(
                dec.get(), jxl.data(), 16,
                std::numeric_limits<uint64_t>::max() - 4));

  // 3) Feed only half of the first required range -> enters sparse mode.
  uint64_t off = 0, len = 0;
  ASSERT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderExperimentalGetNextRequiredOffset(dec.get(), &off, &len));
  ASSERT_GT(len, 16u);
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderExperimentalSetInputAtOffset(dec.get(), jxl.data() + off,
                                                   len / 2, off));

  // 4) Mixing JxlDecoderSetInput while sparse_mode_active == true must fail.
  EXPECT_EQ(JXL_DEC_ERROR,
            JxlDecoderSetInput(dec.get(), jxl.data(), jxl.size()));

  // 5) Advance to NEED_IMAGE_OUT_BUFFER, set buffer, then close input with
  // incomplete sparse section -> must return JXL_DEC_ERROR (never zero-fill).
  JxlPixelFormat fmt = {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  std::vector<uint8_t> out;
  st = JxlDecoderProcessInput(dec.get());
  ASSERT_EQ(JXL_DEC_NEED_IMAGE_OUT_BUFFER, st);
  size_t sz = 0;
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderImageOutBufferSize(dec.get(), &fmt, &sz));
  out.resize(sz);
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSetImageOutBuffer(dec.get(), &fmt, out.data(), out.size()));
  EXPECT_EQ(JXL_DEC_NEED_MORE_INPUT, JxlDecoderProcessInput(dec.get()));
  JxlDecoderCloseInput(dec.get());
  EXPECT_EQ(JXL_DEC_ERROR, JxlDecoderProcessInput(dec.get()));
}

TEST(DecodeStreamingTest, ResetAndRewindStateHygiene) {
  const size_t W = 512;
  const size_t H = 512;
  auto src = GenerateSmoothRgb8(W, H);
  EncodeOpts opts;
  opts.distance = 1.0f;
  auto jxl = EncodeJxl(src.data(), src.size(), W, H, JXL_TYPE_UINT8, opts);

  // Reference fresh decodes for 3 distinct targets.
  StreamTarget t_scaled_roi = {4, JXL_EXP_SCALE_REFERENCE_RESAMPLE, true,
                               16, 16, 64, 64};
  StreamTarget t_full = {1, JXL_EXP_SCALE_FAST_APPROXIMATE, false, 0, 0, 0, 0};
  StreamTarget t_dc = {8, JXL_EXP_SCALE_REQUIRE_FAST_NATIVE, false, 0, 0, 0, 0};
  auto ref_sr = DecodeWithTransport(jxl, t_scaled_roi, JXL_TYPE_FLOAT,
                                    TransportMode::kFullBuffer);
  auto ref_full = DecodeWithTransport(jxl, t_full, JXL_TYPE_FLOAT,
                                      TransportMode::kFullBuffer);
  auto ref_dc = DecodeWithTransport(jxl, t_dc, JXL_TYPE_FLOAT,
                                    TransportMode::kFullBuffer);
  ASSERT_EQ(JXL_DEC_SUCCESS, ref_sr.status);
  ASSERT_EQ(JXL_DEC_SUCCESS, ref_full.status);
  ASSERT_EQ(JXL_DEC_SUCCESS, ref_dc.status);

  // Reuse a SINGLE decoder across: Scaled-ROI -> Rewind -> same Scaled-ROI ->
  // Reset -> Full -> Reset -> 1:8 DC.
  auto dec = JxlDecoderMake(nullptr);
  JxlPixelFormat fmt = {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};

  auto run_once = [&](std::vector<uint8_t>* out) {
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderSetInput(dec.get(), jxl.data(), jxl.size()));
    JxlDecoderCloseInput(dec.get());
    for (;;) {
      JxlDecoderStatus st = JxlDecoderProcessInput(dec.get());
      if (st == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
        size_t sz = 0;
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderImageOutBufferSize(dec.get(), &fmt, &sz));
        out->resize(sz);
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderSetImageOutBuffer(dec.get(), &fmt, out->data(),
                                              out->size()));
      } else if (st == JXL_DEC_FULL_IMAGE) {
        break;
      } else {
        ASSERT_NE(JXL_DEC_ERROR, st);
        ASSERT_NE(JXL_DEC_NEED_MORE_INPUT, st);
      }
    }
  };

  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSubscribeEvents(
                dec.get(),
                JXL_DEC_BASIC_INFO | JXL_DEC_FRAME | JXL_DEC_FULL_IMAGE));
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderExperimentalSetOutputDownsampling(
                dec.get(), 4, JXL_EXP_SCALE_REFERENCE_RESAMPLE));
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderExperimentalSetOutputRegion(dec.get(), 16, 16, 64, 64));
  std::vector<uint8_t> out1, out2, out3, out4;
  run_once(&out1);
  ExpectFloatPixelsMatch(ref_sr.pixels, out1, ref_sr.stats.scale_mode,
                         "rewind_before");

  // Rewind preserves region + downsampling + subscriptions.
  JxlDecoderRewind(dec.get());
  run_once(&out2);
  EXPECT_EQ(out1, out2);
  ExpectFloatPixelsMatch(ref_sr.pixels, out2, ref_sr.stats.scale_mode,
                         "rewind_after");

  // Reset clears region + downsampling + subscriptions (#5103).
  JxlDecoderReset(dec.get());
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSubscribeEvents(
                dec.get(),
                JXL_DEC_BASIC_INFO | JXL_DEC_FRAME | JXL_DEC_FULL_IMAGE));
  run_once(&out3);
  EXPECT_EQ(ref_full.pixels, out3);

  // Reset again and switch to 1:8 native DC.
  JxlDecoderReset(dec.get());
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSubscribeEvents(
                dec.get(),
                JXL_DEC_BASIC_INFO | JXL_DEC_FRAME | JXL_DEC_FULL_IMAGE));
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderExperimentalSetOutputDownsampling(
                dec.get(), 8, JXL_EXP_SCALE_REQUIRE_FAST_NATIVE));
  run_once(&out4);
  EXPECT_EQ(ref_dc.pixels, out4);
}

// Local POSIX pread() range-source adapter with configurable coalescing gap
// (0 B = NVMe, 64 KB = HDD NAS / HTTP 206 cost model).
struct PreadMetrics {
  uint64_t bytes_read = 0;
  size_t pread_calls = 0;
  double wall_ms = 0.0;
  JxlExperimentalRegionDecodeStats stats = {};
  std::vector<uint8_t> pixels;
  uint32_t out_w = 0;
  uint32_t out_h = 0;
};

PreadMetrics DecodeViaPread(int fd, size_t file_size,
                            const StreamTarget& target, JxlDataType data_type,
                            uint64_t coalesce_gap_bytes) {
  PreadMetrics m;
  auto t0 = std::chrono::steady_clock::now();
  auto dec = JxlDecoderMake(nullptr);
  auto runner = JxlThreadParallelRunnerMake(
      nullptr, JxlThreadParallelRunnerDefaultNumWorkerThreads());
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSetParallelRunner(dec.get(), JxlThreadParallelRunner,
                                        runner.get()));
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSubscribeEvents(
                dec.get(),
                JXL_DEC_BASIC_INFO | JXL_DEC_FRAME | JXL_DEC_FULL_IMAGE));
  if (target.downsampling > 1 ||
      target.policy != JXL_EXP_SCALE_FAST_APPROXIMATE) {
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderExperimentalSetOutputDownsampling(
                  dec.get(), target.downsampling, target.policy));
  }
  if (target.use_region) {
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderExperimentalSetOutputRegion(
                  dec.get(), target.rx, target.ry, target.rw, target.rh));
  }

  auto do_pread = [&](uint64_t offset, size_t len, std::vector<uint8_t>* buf) {
    buf->resize(len);
    size_t got = 0;
    while (got < len) {
      ssize_t n = pread(fd, buf->data() + got, len - got,
                        static_cast<off_t>(offset + got));
      EXPECT_GT(n, 0);
      if (n <= 0) break;
      got += static_cast<size_t>(n);
    }
    m.bytes_read += got;
    m.pread_calls++;
  };

  std::vector<uint8_t> hdr_buf;
  size_t hdr_pos = 0;
  const size_t kHdrChunk = 4096;
  do_pread(0, std::min(kHdrChunk, file_size), &hdr_buf);
  hdr_pos = hdr_buf.size();
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSetInput(dec.get(), hdr_buf.data(), hdr_buf.size()));

  JxlPixelFormat fmt = {3, data_type, JXL_NATIVE_ENDIAN, 0};
  bool sparse_active = false;
  std::vector<uint8_t> range_buf;

  auto fetch_coalesced_ranges = [&]() {
    size_t count = 0;
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderExperimentalGetRequiredInputRanges(dec.get(), nullptr,
                                                           &count));
    if (count == 0) return;
    std::vector<JxlExperimentalInputRange> raw(count);
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderExperimentalGetRequiredInputRanges(dec.get(),
                                                           raw.data(), &count));
    // Transport-level coalescing by gap threshold (outside libjxl!).
    std::vector<JxlExperimentalInputRange> merged;
    for (const auto& r : raw) {
      if (!merged.empty() &&
          r.offset <= merged.back().offset + merged.back().length +
                          coalesce_gap_bytes) {
        uint64_t end =
            std::max(merged.back().offset + merged.back().length,
                     r.offset + r.length);
        merged.back().length = end - merged.back().offset;
      } else {
        merged.push_back(r);
      }
    }
    for (const auto& r : merged) {
      do_pread(r.offset, static_cast<size_t>(r.length), &range_buf);
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderExperimentalSetInputAtOffset(
                    dec.get(), range_buf.data(), range_buf.size(), r.offset));
    }
  };

  for (;;) {
    JxlDecoderStatus st = JxlDecoderProcessInput(dec.get());
    if (st == JXL_DEC_NEED_MORE_INPUT) {
      if (!sparse_active) {
        size_t rem = JxlDecoderReleaseInput(dec.get());
        std::vector<uint8_t> next_chunk;
        size_t to_read = std::min(kHdrChunk, file_size - hdr_pos);
        do_pread(hdr_pos, to_read, &next_chunk);
        hdr_pos += next_chunk.size();
        std::vector<uint8_t> combined(hdr_buf.end() - rem, hdr_buf.end());
        combined.insert(combined.end(), next_chunk.begin(), next_chunk.end());
        hdr_buf = std::move(combined);
        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderSetInput(dec.get(), hdr_buf.data(),
                                     hdr_buf.size()));
      } else {
        fetch_coalesced_ranges();
      }
    } else if (st == JXL_DEC_FRAME) {
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderExperimentalGetScaledOutputInfo(
                    dec.get(), &m.out_w, &m.out_h, nullptr, nullptr));
      JxlDecoderReleaseInput(dec.get());
      sparse_active = true;
      fetch_coalesced_ranges();
    } else if (st == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      size_t sz = 0;
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderImageOutBufferSize(dec.get(), &fmt, &sz));
      m.pixels.resize(sz);
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderSetImageOutBuffer(dec.get(), &fmt, m.pixels.data(),
                                            m.pixels.size()));
    } else if (st == JXL_DEC_FULL_IMAGE) {
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderExperimentalGetRegionDecodeStats(dec.get(),
                                                           &m.stats));
      break;
    } else {
      EXPECT_NE(JXL_DEC_ERROR, st);
      if (st == JXL_DEC_ERROR || st == JXL_DEC_SUCCESS) break;
    }
  }
  auto t1 = std::chrono::steady_clock::now();
  m.wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return m;
}

TEST(DecodeStreamingTest, Float2kPreadCoalescingStreaming) {
  const size_t W = 2048;
  const size_t H = 1080;
  auto src = GenerateHdrHalfRgb(W, H);
  EncodeOpts opts;
  opts.distance = 0.5f;
  opts.effort = 3;
  opts.progressive_ac = 1;
  auto jxl = EncodeJxl(src.data(), src.size() * sizeof(uint16_t), W, H,
                       JXL_TYPE_FLOAT16, opts, /*linear_ap1=*/true);

  char path[] = "/tmp/libjxl_stream_2k_XXXXXX";
  int fd = mkstemp(path);
  ASSERT_GE(fd, 0);
  unlink(path);
  ssize_t written = write(fd, jxl.data(), jxl.size());
  ASSERT_EQ(static_cast<ssize_t>(jxl.size()), written);

  const std::vector<StreamTarget> targets = {
      {1, JXL_EXP_SCALE_FAST_APPROXIMATE, true, 512, 256, 256, 256},
      {4, JXL_EXP_SCALE_FAST_APPROXIMATE, false, 0, 0, 0, 0},
      {8, JXL_EXP_SCALE_REQUIRE_FAST_NATIVE, false, 0, 0, 0, 0},
      {4, JXL_EXP_SCALE_FAST_APPROXIMATE, true, 64, 32, 128, 96},
  };

  for (size_t ti = 0; ti < targets.size(); ++ti) {
    auto nvme = DecodeViaPread(fd, jxl.size(), targets[ti], JXL_TYPE_FLOAT16,
                               /*coalesce_gap_bytes=*/0);
    auto nas = DecodeViaPread(fd, jxl.size(), targets[ti], JXL_TYPE_FLOAT16,
                              /*coalesce_gap_bytes=*/65536);
    EXPECT_EQ(nvme.pixels, nas.pixels) << "ti=" << ti;
    EXPECT_LE(nas.pread_calls, nvme.pread_calls) << "ti=" << ti;
    EXPECT_LT(nvme.bytes_read, jxl.size()) << "ti=" << ti;

    // Verify HDR range (> 1.0) is preserved without clamping.
    const auto* hptr = reinterpret_cast<const uint16_t*>(nvme.pixels.data());
    float max_v = -1e9f;
    for (size_t i = 0; i < nvme.pixels.size() / sizeof(uint16_t); ++i) {
      max_v = std::max(max_v, HalfToFloat(hptr[i]));
    }
    EXPECT_GT(max_v, 1.5f) << "ti=" << ti;
  }
  close(fd);
}

TEST(DecodeStreamingTest, DISABLED_Float8kStreamingMatrixBenchmark) {
  const size_t W = 8192;
  const size_t H = 4320;
  auto src = GenerateHdrHalfRgb(W, H);
  EncodeOpts opts;
  opts.distance = 0.5f;
  opts.effort = 1;
  opts.progressive_ac = 1;
  auto jxl = EncodeJxl(src.data(), src.size() * sizeof(uint16_t), W, H,
                       JXL_TYPE_FLOAT16, opts, /*linear_ap1=*/true);

  char path[] = "/tmp/libjxl_stream_8k_XXXXXX";
  int fd = mkstemp(path);
  ASSERT_GE(fd, 0);
  unlink(path);
  ssize_t written = write(fd, jxl.data(), jxl.size());
  ASSERT_EQ(static_cast<ssize_t>(jxl.size()), written);

  struct Case {
    const char* name;
    StreamTarget target;
  } cases[] = {
      {"Full 8K", {1, JXL_EXP_SCALE_FAST_APPROXIMATE, false, 0, 0, 0, 0}},
      {"ROI 512x512 (1:1)",
       {1, JXL_EXP_SCALE_FAST_APPROXIMATE, true, 2048, 1024, 512, 512}},
      {"Scaled 2K (1:4)",
       {4, JXL_EXP_SCALE_FAST_APPROXIMATE, false, 0, 0, 0, 0}},
      {"Scaled 1K (1:8 DC)",
       {8, JXL_EXP_SCALE_REQUIRE_FAST_NATIVE, false, 0, 0, 0, 0}},
      {"Scaled-ROI 512x512 (1:4)",
       {4, JXL_EXP_SCALE_FAST_APPROXIMATE, true, 512, 256, 512, 512}},
  };

  std::printf("[8K Streaming Matrix] total_file=%zu B (%.2f MiB)\n", jxl.size(),
              jxl.size() / (1024.0 * 1024.0));
  for (const auto& c : cases) {
    auto m = DecodeViaPread(fd, jxl.size(), c.target, JXL_TYPE_FLOAT16, 0);
    std::printf(
        "  %-26s | %4ux%-4u | pread=%8llu B (%5.2f%%) in %3zu calls | "
        "peak_buf=%7llu B | %.1f ms | mode=%d partial=%d\n",
        c.name, m.out_w, m.out_h,
        static_cast<unsigned long long>(m.bytes_read),
        100.0 * m.bytes_read / jxl.size(), m.pread_calls,
        static_cast<unsigned long long>(m.stats.sparse_peak_bytes_buffered),
        m.wall_ms, static_cast<int>(m.stats.scale_mode),
        static_cast<int>(m.stats.is_partial_fidelity));
  }
  close(fd);
}

}  // namespace
}  // namespace jxl
