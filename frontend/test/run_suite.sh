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
shopt -s nullglob

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

for test in "$suite_dir"/*.toy; do
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
