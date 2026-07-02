#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$DIR")/_common.sh"
LLVM_BIN="$(find_llvm_bin)"
mkdir -p "$DIR/build"
"${LLVM_BIN}/mlir-opt" "$DIR/input.mlir" \
    --convert-arith-to-llvm --convert-func-to-llvm \
    -o "$DIR/build/lowered.mlir"
grep -q 'llvm.add' "$DIR/build/lowered.mlir"
echo "全部示例通过。"
