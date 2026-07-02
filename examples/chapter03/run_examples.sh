#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$DIR")/_common.sh"
LLVM_BIN="$(find_llvm_bin)"
[[ -n "$LLVM_BIN" ]] || { echo "error: LLVM not found"; exit 1; }
"${LLVM_BIN}/llvm-as" "$DIR/example_ssa.ll" -o "$DIR/example_ssa.bc"
exit_code=0
"${LLVM_BIN}/lli" "$DIR/example_ssa.bc" || exit_code=$?
[[ "$exit_code" -eq 14 ]] || { echo "error: expected exit 14, got $exit_code"; exit 1; }
python3 "$DIR/codegen_visitor.py"
echo "全部示例通过。"
