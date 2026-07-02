#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$DIR")/_common.sh"
grep -q '#ttg.blocked' "$DIR/vector_add.ttgir"
python3 "$DIR/encoding_calc.py"
if TRITON_OPT="$(find_triton_opt)"; [[ -n "$TRITON_OPT" ]]; then
  "$TRITON_OPT" "$DIR/vector_add.ttgir" --canonicalize -disable-output
else
  echo "    SKIP: triton-opt not found"
fi
echo "全部示例通过。"
