#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
test -f "$DIR/design_doc_template.md"
cmake -S "$DIR/scaffold" -B "$DIR/build" >/dev/null
echo "全部示例通过。"
