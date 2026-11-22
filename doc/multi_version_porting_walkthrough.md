# Multi-Version Backporting & Upstream Forward-Porting Walkthrough

## 1. Summary

All three experimental `libjxl` features (**ROI decoding + sparse byte-range I/O**, **power-of-two downscaled decoding**, and **unified 4×2 streaming decoding**, plus all plans, walkthroughs, and unit tests) have been ported onto **4 dedicated fork release branches** anchored at exact upstream release tags on `davisadam10/libjxl`, alongside the reusable local automation CLI [`scripts/experimental_port.sh`](../scripts/experimental_port.sh).

- **Public ABI Guarantee**: [`lib/include/jxl/decode.h`](../lib/include/jxl/decode.h) is **100% untouched** (`git diff --stat <tag> -- lib/include/jxl/decode.h` empty) on every branch.
- **Verification**: All 5 GTest suites (`decode_region_test`, `decode_scale_test`, `decode_streaming_test`, `render_pipeline_test`, `decode_test`) built cleanly with Ninja and passed with **0 failures** on every release branch.

---

## 2. Verified Release Branch Matrix

| Branch | Base Upstream Tag (SHA) | Branch HEAD SHA | `decode.h` Untouched | `decode_region_test` | `decode_scale_test` | `decode_streaming_test` | `render_pipeline_test` | `decode_test` |
|---|---|---|---|---|---|---|---|---|
| **`feat/experimental-roi-decoding`** | `main` (`12e4bf40`) | `7da4f94b` | PASS | 11 / 11 | 10 / 10 | 7 / 7 | 1,198 / 1,198 | 2,895 / 2,895 |
| **`exp/v0.13.0`** | `v0.13.0` (`5a7da834`) | `a60b229a` | PASS | 11 / 11 | 10 / 10 | 7 / 7 | 1,198 / 1,198 | 2,895 / 2,895 |
| **`exp/v0.12.0`** | `v0.12.0` (`a7a9c787`) | `18d0bacd` | PASS | 11 / 11 | 10 / 10 | 7 / 7 | 174 / 174 | 2,891 / 2,891 |
| **`exp/v0.11.2`** | `v0.11.2` (`332feb17`) | `4341a9e4` | PASS | 11 / 11 | 10 / 10 | 7 / 7 | 174 / 174 | 2,888 / 2,888 |
| **`exp/v0.11.1`** *(prod one-off)* | `v0.11.1` (`794a5dcf`) | `93d73e49` | PASS | 11 / 11 | 10 / 10 | 7 / 7 | 174 / 174 | 2,888 / 2,888 |

---

## 3. Per-Version Porting & Compatibility Notes

### `exp/v0.13.0` (Base: `v0.13.0` / `5a7da834`)
- **Conflicts**: **0** — entire patch stack applied cleanly via `git am --3way`.
- **Cross-Branch Catch**: Running `decode_streaming_test` on `exp/v0.13.0` caught a cross-thread 1-ULP float assertion in `ResetAndRewindStateHygiene` (`ref_sr` decoded multi-threaded vs `out1`/`out2` single-threaded). Fixed canonically in `7da4f94b` (`ExpectFloatPixelsMatch` against `ref_sr` + exact bitwise `EXPECT_EQ(out1, out2)` for `JxlDecoderRewind` parity) and cherry-picked to all release branches.

### `exp/v0.12.0` (Base: `v0.12.0` / `a7a9c787`)
- **Conflicts (2 files, minor context alignment)**:
  - `lib/jxl/dec_cache.cc`: Preserved `v0.12.0`'s inline `has_alpha` / `alpha_c` scan alongside `unoriented_scaled_region`.
  - `lib/jxl/render_pipeline/low_memory_render_pipeline.h`: Omitted `StageBufferYSize` (introduced in `v0.13.0`).
- **API Compatibility**: `jxlp_ooo_buffer` and `JXL_ENC_FRAME_SETTING_OUTPUT_MODE` both exist in `v0.12.0`; no feature guards needed.

### `exp/v0.11.2` (Base: `v0.11.2` / `332feb17`)
- **Conflicts & `v0.11.x` Adaptations**:
  - `lib/jxl/dec_frame.h`: Preserved `v0.11` `void` return types for `SetImageOutput` / `AddExtraChannelOutput` and `size_t`-returning `GetStride`.
  - `lib/jxl/render_pipeline/stage_xyb.cc`: Preserved `v0.11` single `size_t xextra` `ProcessRow` signature.
  - `lib/jxl/render_pipeline/stage_write.cc`:
    - Used `v0.11`'s `kMaxPixelsPerCall` constant (renamed to `kChunkSize` in `v0.12`).
    - **Ordered Dithering Phase Fix (`MakeUnsigned`)**: On `v0.11`, `LoadDup128` duplicates 4 dither floats per 128-bit lane (period 4 on 256-bit SIMD, period 8 on ≤128-bit SIMD). Added an unaligned-`x0` (`x0 % Lanes(DF()) != 0`) phase-matching branch so ROI `UINT8` output matches stock `v0.11` full-image output bit-for-bit across every SIMD width and all 8 EXIF orientations, with zero change to full-image decoding.
  - `lib/jxl/decode.cc`:
    - Added `#include <map>`, dropped the `v0.12+` `dec->jxlp_ooo_buffer.empty()` clause (`v0.11` only supports in-order `jxlp`), used `v0.11`'s non-template `jxl::SafeAdd(uint64_t, uint64_t, uint64_t&)`, and replaced `v0.12+` `CanAddBuffer` with an explicit 1 GiB sparse buffer cap.
  - `lib/jxl/decode_streaming_test.cc`: Version-agnostic `#if JPEGXL_NUMERIC_VERSION >= JPEGXL_COMPUTE_NUMERIC_VERSION(0, 12, 0)` guard around `JXL_ENC_FRAME_SETTING_OUTPUT_MODE` (falls back to raw + container + multi-`jxlp` on `v0.11`).
  - `lib/jxl/decode_region_test.cc`: Set `PALETTE_COLORS=0` and `CHANNEL_COLORS_GLOBAL_PERCENT=0.0f` in `ModularGroupIndependentAndSqueezeFallback` part 1 so the `v0.11` encoder deterministically emits group-independent RCT rather than a global alpha palette.

### `exp/v0.11.1` (Production One-Off, Base: `v0.11.1` / `794a5dcf`)
- **Conflicts**: **0** — exported the verified `v0.11.2` patch series (`git format-patch v0.11.2..exp/v0.11.2`) and applied cleanly onto `v0.11.1` via `scripts/experimental_port.sh port`.

---

## 4. Forward-Porting & Maintenance Workflow (`v0.14.x`, `v0.15.x`, `upstream/main`)

Use [`scripts/experimental_port.sh`](../scripts/experimental_port.sh) from `feat/experimental-roi-decoding`:

```bash
# 1. Dry-run our patch stack against upstream/main or any new release tag (no local branch modified):
./scripts/experimental_port.sh check-upstream upstream/main
./scripts/experimental_port.sh check-upstream v0.14.0 --build

# 2. Create a new release branch (e.g., exp/v0.14.0) in a clean worktree:
./scripts/experimental_port.sh port v0.14.0 exp/v0.14.0

# 3. Verify ABI guard (decode.h untouched) + build & run all 5 GTest suites:
./scripts/experimental_port.sh verify /tmp/build-exp-v0.14.0 v0.14.0
```

Smoke test verified: `./scripts/experimental_port.sh check-upstream v0.13.0` exited `0` (`Patch apply + ABI guard PASSED on 'v0.13.0'`) and cleaned up its temporary worktree automatically.
