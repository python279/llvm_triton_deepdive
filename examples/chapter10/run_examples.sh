#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$DIR")/_common.sh"
LLVM_BIN="$(find_llvm_bin)"
mkdir -p "$DIR/build"
"${LLVM_BIN}/mlir-opt" "$DIR/sample.mlir" --canonicalize -o "$DIR/build/sample_canon.mlir"
python3 "$DIR/count_ops.py" "$DIR/sample.mlir"
echo "全部示例通过。"
