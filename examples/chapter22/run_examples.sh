#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
grep -q 'mygpu.matmul' "$DIR/sample_mygpu.mlir"
grep -q '#mygpu.blocked' "$DIR/sample_mygpu.mlir"
echo "全部示例通过。"
