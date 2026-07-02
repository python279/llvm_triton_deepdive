#!/usr/bin/env bash
# 第 24 章：IR dump 调试命令模板
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(dirname "$DIR")/_common.sh"
echo "export TRITON_KERNEL_DUMP=1"
echo "export TRITON_ALWAYS_COMPILE=1"
if TRITON_OPT="$(find_triton_opt)"; [[ -n "$TRITON_OPT" ]]; then
  echo "triton-opt example.ttir --inline --canonicalize"
else
  echo "SKIP: triton-opt not available"
fi
test -f "$DIR/../chapter15/example.ttir"
echo "OK: debug_dump template"
