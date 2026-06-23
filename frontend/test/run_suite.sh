#!/usr/bin/env bash
# Run all *.toy tests under SUITE_DIR.
# Usage: run_suite.sh LUCID_C FILECHECK SUITE_DIR EMIT_FLAG MODE
#   MODE=positive  — expect lucid-c success + FileCheck match
#   MODE=negative  — expect lucid-c failure + FileCheck match
set -uo pipefail

lucid_c="$1"
filecheck="$2"
suite_dir="$3"
emit_flag="$4"
mode="$5"

fail=0

if [ ! -f "$lucid_c" ]; then
  echo "ERROR: lucid-c not found: $lucid_c" >&2
  echo "Build it first: cmake --build <build-dir> --target lucid-c" >&2
  exit 1
fi
if [ ! -f "$filecheck" ]; then
  echo "ERROR: FileCheck not found: $filecheck" >&2
  echo "Re-configure with LLVM/MLIR on PATH, or set -DLLVM_TOOLS_BINARY_DIR=..." >&2
  exit 1
fi
if [ ! -d "$suite_dir" ]; then
  echo "ERROR: test directory not found: $suite_dir" >&2
  exit 1
fi

run_positive() {
  local test="$1"
  echo "=== RUN: $test ==="
  if ! "$lucid_c" "$test" "$emit_flag" 2>&1 | "$filecheck" "$test"; then
    echo "FAILED: $test"
    return 1
  fi
}

run_negative() {
  local test="$1"
  echo "=== RUN: $test (expect failure) ==="
  "$lucid_c" "$test" "$emit_flag" 2>&1 | "$filecheck" "$test"
  local lucid_exit=${PIPESTATUS[0]}
  local filecheck_exit=$?
  if [ "$lucid_exit" -ne 0 ] && [ "$filecheck_exit" -eq 0 ]; then
    return 0
  fi
  echo "FAILED: $test"
  return 1
}

mapfile -t tests < <(find "$suite_dir" -name '*.toy' | sort)
if [ "${#tests[@]}" -eq 0 ]; then
  echo "No *.toy tests found under $suite_dir" >&2
  exit 1
fi

for test in "${tests[@]}"; do
  case "$mode" in
    positive)
      run_positive "$test" || fail=1
      ;;
    negative)
      run_negative "$test" || fail=1
      ;;
    *)
      echo "Unknown mode: $mode" >&2
      exit 2
      ;;
  esac
done

if [ "$fail" -ne 0 ]; then
  echo "$(basename "$suite_dir") suite: $fail test(s) failed"
fi
exit "$fail"
