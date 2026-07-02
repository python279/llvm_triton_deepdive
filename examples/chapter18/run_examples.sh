#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
grep -E 'ld\.global|st\.global' "$DIR/sample.ptx"
grep -q 'shfl' "$DIR/sample.ptx"
echo "全部示例通过。"
