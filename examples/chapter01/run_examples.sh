#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$DIR")/_common.sh"
LLVM_BIN="$(find_llvm_bin)"
CLANG="$(find_clang)"
[[ -n "$LLVM_BIN" && -n "$CLANG" ]] || { echo "error: LLVM/clang not found"; exit 1; }
mkdir -p "$DIR/build"
echo "==> hello.c"
"$CLANG" -S -emit-llvm -O0 "$DIR/hello.c" -o "$DIR/build/hello.ll"
grep -q '@add' "$DIR/build/hello.ll"
"$CLANG" "$DIR/hello.c" -o "$DIR/build/hello"
"$DIR/build/hello" | grep -q 'Result: 3'
echo "==> test.mlir"
"${LLVM_BIN}/mlir-opt" "$DIR/test.mlir" --canonicalize -o /dev/null
echo "==> matmul.c"
"$CLANG" -S -emit-llvm -O0 "$DIR/matmul.c" -o "$DIR/build/matmul.ll"
grep -E 'load|store|mul|add' "$DIR/build/matmul.ll" >/dev/null
if TRITON_OPT="$(find_triton_opt)"; [[ -n "$TRITON_OPT" ]]; then
  echo "==> test.ttir"
  "$TRITON_OPT" "$DIR/test.ttir" --canonicalize -disable-output
else
  echo "    SKIP: triton-opt not found"
fi
echo "全部示例通过。"
