#!/usr/bin/env bash
# 运行 books/examples 下全部章节的 run_examples.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LLVM_DIR="${LLVM_DIR:-/opt/homebrew/opt/llvm/lib/cmake/llvm}"
export MLIR_DIR="${MLIR_DIR:-/opt/homebrew/opt/llvm/lib/cmake/mlir}"
FAIL=0
PASS=0
SKIP=0
for ch in $(seq -w 0 24); do
  script="$ROOT/chapter${ch}/run_examples.sh"
  [[ -x "$script" ]] || { echo "MISSING $script"; FAIL=$((FAIL+1)); continue; }
  echo "======== chapter${ch} ========"
  if "$script"; then
    PASS=$((PASS+1))
  else
    code=$?
    echo "FAILED chapter${ch} (exit $code)"
    FAIL=$((FAIL+1))
  fi
done
echo "Done: pass=$PASS fail=$FAIL"
exit $(( FAIL > 0 ? 1 : 0 ))
