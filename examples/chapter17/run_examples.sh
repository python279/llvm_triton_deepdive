#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
grep -vq '#ttg' "$DIR/sample.ttir"
grep -q '#ttg.blocked' "$DIR/sample.ttgir"
echo "OK: TTIR has no encoding, TTGIR has #blocked"
echo "全部示例通过。"
