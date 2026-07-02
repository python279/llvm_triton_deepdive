#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
test -f "$DIR/cuda_add.cu"
python3 -m py_compile "$DIR/triton_add.py"
python3 "$DIR/triton_add.py"
echo "全部示例通过。"
