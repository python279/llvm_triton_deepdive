#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$DIR")/_common.sh"
LLVM_BIN="$(find_llvm_bin)"
python3 "$DIR/toy_frontend.py" | grep -q 'arith.addf'
"${LLVM_BIN}/mlir-opt" "$DIR/test.toy.mlir" --canonicalize -o /dev/null
echo "全部示例通过。"
