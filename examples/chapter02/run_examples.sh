#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$DIR/build"
cmake -G Ninja -S "$DIR" -B "$DIR/build" ${LLVM_DIR:+-DLLVM_DIR="$LLVM_DIR"}
ninja -C "$DIR/build"
"$DIR/build/llvm_adt_demo" 2>&1 | grep -q 'sum=6'
echo "全部示例通过。"
