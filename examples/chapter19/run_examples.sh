#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRITON_ROOT="$(cd "$DIR/../../.." && pwd)"
python3 "$DIR/inspect_cuda_backend.py"
grep -q '\.version' "$DIR/sample.ptx"
grep -q '\.target' "$DIR/sample.ptx"
grep -q '\.reg' "$DIR/sample.ptx"
echo "全部示例通过。"
