#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$DIR")/_common.sh"
LLVM_BIN="$(find_llvm_bin)"
[[ -n "$LLVM_BIN" ]] || { echo "error: LLVM not found"; exit 1; }
TBLGEN="${LLVM_BIN}/mlir-tblgen"
mkdir -p "$DIR/build"
INC="$(dirname "$LLVM_BIN")/../include"
if [[ ! -d "$INC" ]]; then
  INC="$(brew --prefix llvm 2>/dev/null)/include" || INC="/opt/homebrew/opt/llvm/include"
fi
echo "==> mlir-tblgen MiniDialect.td"
"$TBLGEN" -I"$INC" "$DIR/MiniDialect.td" --gen-op-decls -o "$DIR/build/MiniOps.h.inc"
grep -q 'class AddOp' "$DIR/build/MiniOps.h.inc"
"$TBLGEN" -I"$INC" "$DIR/MiniDialect.td" --gen-dialect-decls -o "$DIR/build/MiniDialect.h.inc"
grep -q 'class MiniDialect' "$DIR/build/MiniDialect.h.inc"
echo "全部示例通过。"
