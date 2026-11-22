#!/usr/bin/env bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
#
# Multi-version backporting and upstream forward-porting helper for the
# experimental ROI, downscaling, and streaming decoder features.
#
# Usage:
#   scripts/experimental_port.sh export-patches [base_ref] [out_dir]
#   scripts/experimental_port.sh port <target_ref> <new_branch> [patch_dir] [worktree_dir]
#   scripts/experimental_port.sh verify [build_dir] [base_ref]
#   scripts/experimental_port.sh check-upstream [target_ref] [--build]

set -euo pipefail

SCRIPT_NAME="$(basename "$0")"
REPO_ROOT="$(git rev-parse --show-toplevel)"

usage() {
  cat <<EOF
Usage:
  $SCRIPT_NAME export-patches [base_ref] [out_dir]
      Export git format-patch series from <base_ref>..HEAD into <out_dir>.
      Defaults: base_ref = merge-base(HEAD, main|v0.13.0), out_dir = /tmp/jxl-exp-patches

  $SCRIPT_NAME port <target_ref> <new_branch> [patch_dir] [worktree_dir]
      Create <new_branch> at <target_ref> in a clean git worktree and apply the
      patch series via 'git am --3way'. Fetches <target_ref> from 'upstream' if
      not present locally.

  $SCRIPT_NAME verify [build_dir] [base_ref]
      1) Verify lib/include/jxl/decode.h is 100% untouched vs <base_ref>.
      2) Configure (Ninja) and build the 5 verification test suites.
      3) Run all 5 test suites and print a compact pass/fail summary table.

  $SCRIPT_NAME check-upstream [target_ref] [--build]
      Create a temporary detached worktree at <target_ref> (default: upstream/main),
      dry-run 'git am --3way' of the current branch's patch stack, optionally build
      and run the 3 experimental test suites (--build), and clean up automatically.
EOF
}

detect_base_ref() {
  local requested="${1:-}"
  if [[ -n "$requested" ]]; then
    echo "$requested"
    return 0
  fi
  if git rev-parse --verify -q main >/dev/null 2>&1; then
    local mb
    mb="$(git merge-base HEAD main 2>/dev/null || true)"
    if [[ -n "$mb" && "$mb" != "$(git rev-parse HEAD)" ]]; then
      echo "$mb"
      return 0
    fi
  fi
  local tag
  tag="$(git describe --tags --abbrev=0 --match 'v0.*' HEAD 2>/dev/null || true)"
  if [[ -n "$tag" ]]; then
    echo "$tag"
    return 0
  fi
  echo "ERROR: Could not auto-detect base_ref; please specify it explicitly." >&2
  return 1
}

ensure_ref_available() {
  local ref="$1"
  if git rev-parse --verify -q "${ref}^{commit}" >/dev/null 2>&1; then
    return 0
  fi
  echo "[port] Ref '$ref' not found locally; fetching from upstream..."
  if [[ "$ref" == v* ]]; then
    git fetch upstream --no-tags "refs/tags/${ref}:refs/tags/${ref}"
  elif [[ "$ref" == upstream/* ]]; then
    local branch="${ref#upstream/}"
    git fetch upstream "$branch"
  else
    git fetch upstream "$ref"
  fi
}

cmd_export_patches() {
  local base_ref
  base_ref="$(detect_base_ref "${1:-}")"
  local out_dir="${2:-/tmp/jxl-exp-patches}"

  rm -rf "$out_dir"
  mkdir -p "$out_dir"

  local count
  count="$(git rev-list --count "${base_ref}..HEAD")"
  if [[ "$count" -eq 0 ]]; then
    echo "ERROR: No commits in range ${base_ref}..HEAD" >&2
    return 1
  fi

  git format-patch --no-signature -o "$out_dir" "${base_ref}..HEAD" >/dev/null
  echo "[export-patches] Exported $count patch(es) from ${base_ref}..HEAD -> $out_dir"
  ls -1 "$out_dir"
}

cmd_port() {
  if [[ $# -lt 2 ]]; then
    echo "ERROR: port requires <target_ref> <new_branch>" >&2
    usage
    return 1
  fi
  local target_ref="$1"
  local new_branch="$2"
  local patch_dir="${3:-}"
  local worktree_dir="${4:-}"

  local tmp_patches=""
  if [[ -z "$patch_dir" ]]; then
    tmp_patches="$(mktemp -d /tmp/jxl-exp-patches.XXXXXX)"
    cmd_export_patches "" "$tmp_patches"
    patch_dir="$tmp_patches"
  fi

  if [[ -z "$worktree_dir" ]]; then
    local safe_name="${new_branch//\//-}"
    worktree_dir="/tmp/libjxl-wt-${safe_name}"
  fi

  ensure_ref_available "$target_ref"

  if [[ -d "$worktree_dir" ]]; then
    echo "ERROR: Worktree directory already exists: $worktree_dir" >&2
    echo "Remove it first with: git worktree remove --force $worktree_dir" >&2
    return 1
  fi

  echo "[port] Creating worktree '$worktree_dir' on branch '$new_branch' at '$target_ref'..."
  if git show-ref --verify --quiet "refs/heads/${new_branch}"; then
    echo "ERROR: Branch '$new_branch' already exists. Delete it first or pick another name." >&2
    return 1
  fi

  git worktree add -b "$new_branch" "$worktree_dir" "$target_ref"

  echo "[port] Applying patches from '$patch_dir' via 'git am --3way'..."
  set +e
  (
    cd "$worktree_dir"
    git am --3way "$patch_dir"/*.patch
  )
  local am_status=$?
  set -e

  if [[ -n "$tmp_patches" ]]; then
    rm -rf "$tmp_patches"
  fi

  if [[ "$am_status" -ne 0 ]]; then
    echo ""
    echo "[port] CONFLICT while applying patch series onto '$target_ref'!"
    echo "[port] Conflicting files:"
    (cd "$worktree_dir" && git diff --name-only --diff-filter=U || true)
    echo ""
    echo "[port] Resolve conflicts inside: $worktree_dir"
    echo "[port] Then run: git add <files> && git am --continue"
    return 2
  fi

  echo "[port] Clean apply succeeded on branch '$new_branch' in $worktree_dir"
}

cmd_verify() {
  local build_dir="${1:-build}"
  local base_ref
  base_ref="$(detect_base_ref "${2:-}")"

  echo "============================================================"
  echo "[verify] Branch : $(git rev-parse --abbrev-ref HEAD) ($(git rev-parse --short HEAD))"
  echo "[verify] Base   : $base_ref ($(git rev-parse --short "$base_ref"))"
  echo "[verify] Build  : $build_dir"
  echo "============================================================"

  # 1. Public ABI guard: lib/include/jxl/decode.h must be 100% untouched.
  local decode_h_diff
  decode_h_diff="$(git diff --stat "$base_ref" -- lib/include/jxl/decode.h)"
  if [[ -n "$decode_h_diff" ]]; then
    echo "ERROR: ABI GUARD FAILED — lib/include/jxl/decode.h was modified!" >&2
    echo "$decode_h_diff" >&2
    return 1
  fi
  echo "[verify] ABI guard PASS: lib/include/jxl/decode.h is 100% untouched."

  # 2. Configure with Ninja if needed.
  if [[ ! -f "${build_dir}/build.ninja" ]]; then
    echo "[verify] Configuring CMake (Ninja, RelWithDebInfo)..."
    cmake -B "$build_dir" -G Ninja \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DBUILD_TESTING=ON \
      -DJPEGXL_FORCE_SYSTEM_BROTLI=ON \
      -DJPEGXL_ENABLE_TOOLS=OFF \
      -DJPEGXL_ENABLE_MANPAGES=OFF \
      -DJPEGXL_ENABLE_BENCHMARK=OFF \
      -DJPEGXL_ENABLE_EXAMPLES=OFF \
      -DJPEGXL_ENABLE_JNI=OFF \
      -DJPEGXL_ENABLE_OPENEXR=OFF \
      -DJPEGXL_ENABLE_SJPEG=OFF
  fi

  # 3. Build the 5 verification test binaries.
  local targets=(
    decode_region_test
    decode_scale_test
    decode_streaming_test
    render_pipeline_test
    decode_test
  )
  echo "[verify] Building targets: ${targets[*]}"
  cmake --build "$build_dir" --target "${targets[@]}"

  # 4. Run all 5 test suites and collect summary.
  echo ""
  printf "%-26s | %-8s | %s\n" "Test Suite" "Status" "Details"
  printf "%-26s-+-%-8s-+-%s\n" "--------------------------" "--------" "------------------------------"

  local overall_rc=0
  for t in "${targets[@]}"; do
    local bin="${build_dir}/lib/${t}"
    if [[ ! -x "$bin" ]]; then
      bin="$(find "$build_dir" -maxdepth 4 -type f -name "$t" -executable | head -n 1 || true)"
    fi
    if [[ -z "$bin" || ! -x "$bin" ]]; then
      printf "%-26s | %-8s | %s\n" "$t" "MISSING" "binary not found in $build_dir"
      overall_rc=1
      continue
    fi

    local log_file
    log_file="$(mktemp /tmp/jxl-test-${t}.XXXXXX)"
    set +e
    "$bin" >"$log_file" 2>&1
    local rc=$?
    set -e

    local passed_line
    passed_line="$(grep -E '\[  PASSED  \]' "$log_file" | tail -n 1 || true)"
    if [[ "$rc" -eq 0 ]]; then
      printf "%-26s | %-8s | %s\n" "$t" "PASS" "${passed_line:-ok}"
    else
      local failed_line
      failed_line="$(grep -E '\[  FAILED  \]' "$log_file" | tail -n 1 || true)"
      printf "%-26s | %-8s | %s (see %s)\n" "$t" "FAIL" "${failed_line:-rc=$rc}" "$log_file"
      overall_rc=1
      continue
    fi
    rm -f "$log_file"
  done

  echo ""
  if [[ "$overall_rc" -eq 0 ]]; then
    echo "[verify] ALL 5 TEST SUITES PASSED."
  else
    echo "[verify] ONE OR MORE TEST SUITES FAILED." >&2
  fi
  return "$overall_rc"
}

cmd_check_upstream() {
  local target_ref="upstream/main"
  local do_build=0

  for arg in "$@"; do
    case "$arg" in
      --build)
        do_build=1
        ;;
      -*)
        echo "ERROR: Unknown option: $arg" >&2
        return 1
        ;;
      *)
        target_ref="$arg"
        ;;
    esac
  done

  ensure_ref_available "$target_ref"

  _CHECK_TMP_PATCHES="$(mktemp -d /tmp/jxl-check-patches.XXXXXX)"
  _CHECK_TMP_WT="$(mktemp -d /tmp/jxl-check-wt.XXXXXX)"
  rmdir "$_CHECK_TMP_WT"

  cleanup_check() {
    if [[ -n "${_CHECK_TMP_WT:-}" ]]; then
      git -C "$REPO_ROOT" worktree remove --force "$_CHECK_TMP_WT" >/dev/null 2>&1 || rm -rf "$_CHECK_TMP_WT"
    fi
    if [[ -n "${_CHECK_TMP_PATCHES:-}" ]]; then
      rm -rf "$_CHECK_TMP_PATCHES"
    fi
  }
  trap cleanup_check EXIT

  cmd_export_patches "" "$_CHECK_TMP_PATCHES"
  echo "[check-upstream] Creating temporary detached worktree at '$target_ref'..."
  git worktree add --detach "$_CHECK_TMP_WT" "$target_ref"

  echo "[check-upstream] Testing 'git am --3way' against '$target_ref'..."
  set +e
  (
    cd "$_CHECK_TMP_WT"
    git am --3way "$_CHECK_TMP_PATCHES"/*.patch
  )
  local am_rc=$?
  set -e

  if [[ "$am_rc" -ne 0 ]]; then
    echo "[check-upstream] FAIL: Patch stack does NOT apply cleanly to '$target_ref'."
    echo "[check-upstream] Conflicting files:"
    (cd "$_CHECK_TMP_WT" && git diff --name-only --diff-filter=U || true)
    return 2
  fi

  local decode_h_diff
  decode_h_diff="$(git -C "$_CHECK_TMP_WT" diff --stat "$target_ref" -- lib/include/jxl/decode.h)"
  if [[ -n "$decode_h_diff" ]]; then
    echo "[check-upstream] FAIL: lib/include/jxl/decode.h was modified!" >&2
    return 1
  fi

  echo "[check-upstream] Patch apply + ABI guard PASSED on '$target_ref'."

  if [[ "$do_build" -eq 1 ]]; then
    echo "[check-upstream] Initializing submodules and running experimental test build..."
    (
      cd "$_CHECK_TMP_WT"
      git submodule update --init --recursive
      cmake -B build-check -G Ninja \
        -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DBUILD_TESTING=ON \
        -DJPEGXL_FORCE_SYSTEM_BROTLI=ON \
        -DJPEGXL_ENABLE_TOOLS=OFF \
        -DJPEGXL_ENABLE_MANPAGES=OFF \
        -DJPEGXL_ENABLE_BENCHMARK=OFF \
        -DJPEGXL_ENABLE_EXAMPLES=OFF \
        -DJPEGXL_ENABLE_JNI=OFF \
        -DJPEGXL_ENABLE_OPENEXR=OFF \
        -DJPEGXL_ENABLE_SJPEG=OFF
      cmake --build build-check --target decode_region_test decode_scale_test decode_streaming_test
      ./build-check/lib/tests/decode_region_test
      ./build-check/lib/tests/decode_scale_test
      ./build-check/lib/tests/decode_streaming_test
    )
    echo "[check-upstream] Build & experimental tests PASSED on '$target_ref'."
  fi
}

main() {
  if [[ $# -lt 1 ]]; then
    usage
    exit 1
  fi
  local subcmd="$1"
  shift
  case "$subcmd" in
    export-patches)
      cmd_export_patches "$@"
      ;;
    port)
      cmd_port "$@"
      ;;
    verify)
      cmd_verify "$@"
      ;;
    check-upstream)
      cmd_check_upstream "$@"
      ;;
    -h|--help|help)
      usage
      ;;
    *)
      echo "ERROR: Unknown subcommand '$subcmd'" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
