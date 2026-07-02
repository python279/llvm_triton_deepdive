#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$DIR")/_common.sh"
LLVM_BIN="$(find_llvm_bin)"
OUT="$("${LLVM_BIN}/mlir-opt" "$DIR/test/add.mlir" --canonicalize)"
echo "$OUT" | grep -q 'arith.addi'
echo "全部示例通过。"
