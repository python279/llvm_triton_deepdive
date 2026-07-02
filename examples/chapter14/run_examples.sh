#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$DIR")/_common.sh"
python3 -m py_compile "$DIR/simple_kernel.py"
python3 "$DIR/simple_kernel.py"
if TRITON_OPT="$(find_triton_opt)"; [[ -n "$TRITON_OPT" ]]; then
  "$TRITON_OPT" "$DIR/fixtures/simple.ttir" --canonicalize -disable-output
else
  grep -q 'tt.get_program_id' "$DIR/fixtures/simple.ttir"
  echo "    SKIP: triton-opt not found, static fixture checked"
fi
echo "全部示例通过。"
