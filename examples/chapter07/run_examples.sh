#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$DIR")/_common.sh"
LLVM_BIN="$(find_llvm_bin)"
[[ -n "$LLVM_BIN" ]] || { echo "error: LLVM not found"; exit 1; }
mkdir -p "$DIR/build"
"${LLVM_BIN}/llc" "$DIR/add.ll" -o "$DIR/build/add.s"
grep -E 'add|ret' "$DIR/build/add.s" >/dev/null
echo "全部示例通过。"
