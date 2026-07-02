#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$DIR")/_common.sh"
grep -q 'tt.load' "$DIR/example.ttir"
grep -q 'tensor<1024xf32>' "$DIR/example.ttir"
grep -vq '#ttg' "$DIR/example.ttir"
if TRITON_OPT="$(find_triton_opt)"; [[ -n "$TRITON_OPT" ]]; then
  "$TRITON_OPT" "$DIR/example.ttir" --inline --canonicalize -disable-output
else
  echo "    SKIP: triton-opt not found"
fi
echo "全部示例通过。"
