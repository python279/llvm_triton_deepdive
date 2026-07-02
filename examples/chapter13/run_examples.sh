#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 -m py_compile "$DIR/vector_add.py"
python3 "$DIR/vector_add.py"
echo "全部示例通过。"
