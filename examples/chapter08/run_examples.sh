#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$DIR")/_common.sh"
LLVM_BIN="$(find_llvm_bin)"
[[ -n "$LLVM_BIN" ]] || { echo "error: LLVM not found"; exit 1; }
mkdir -p "$DIR/build"
cmake -G Ninja -S "$DIR" -B "$DIR/build" \
    ${LLVM_DIR:+-DLLVM_DIR="$LLVM_DIR"} \
    ${MLIR_DIR:+-DMLIR_DIR="$MLIR_DIR"}
ninja -C "$DIR/build"
OUT="$("$DIR/build/build_add" 2>&1)"
echo "$OUT" | grep -q 'func.func @add'
echo "$OUT" | grep -q 'arith.addi'
"${LLVM_BIN}/mlir-opt" "$DIR/test_scf_if.mlir" --canonicalize -o /dev/null
echo "全部示例通过。"
