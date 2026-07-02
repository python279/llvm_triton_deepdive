#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CH06="$(dirname "$DIR")/chapter06"
"$CH06/run_examples.sh"
test -f "$DIR/sample.mlir"
grep -q 'mini.add' "$DIR/sample.mlir"
echo "全部示例通过。"
