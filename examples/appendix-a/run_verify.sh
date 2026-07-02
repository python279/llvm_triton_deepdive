#!/usr/bin/env bash
# 验证附录 A 中可运行的代码参考答案
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
PASS=0
FAIL=0

run() {
    echo "==> $*"
    if "$@"; then
        PASS=$((PASS + 1))
    else
        echo "FAILED: $*"
        FAIL=$((FAIL + 1))
    fi
}

# ch02
if command -v llvm-config >/dev/null 2>&1; then
    run bash -c "c++ -std=c++17 \$(llvm-config --cppflags --ldflags --libs support demangle) \
        '$DIR/ch02_process_adt.cpp' -o /tmp/ch02_hw && /tmp/ch02_hw | grep -q 246"
else
    echo "SKIP ch02: llvm-config not found"
fi

# ch03
run llvm-as "$DIR/ch03_ssa_example.ll" -o /dev/null

# ch04 max（复用 chapter04 示例）
run bash -c "llvm-as '$ROOT/chapter04/max.ll' -o /dev/null && lli '$ROOT/chapter04/max.ll' >/dev/null 2>&1; test \$? -eq 7"

# ch04 gep
run llvm-as "$ROOT/chapter04/gep_demo.ll" -o /dev/null

# ch05（若已构建）
if [[ -x "$ROOT/chapter05/build/CountAddPass.so" ]]; then
    OPT="$(llvm-config --bindir)/opt"
    run "$OPT" -load-pass-plugin "$ROOT/chapter05/build/CountAddPass.so" \
        -passes=count-add -disable-output "$ROOT/chapter05/test.ll"
else
    echo "SKIP ch05: run books/examples/chapter05/run_examples.sh first"
fi

# ch08 mlir
if command -v mlir-opt >/dev/null 2>&1; then
    run mlir-opt "$DIR/ch08_hw1_scf_if.mlir" --canonicalize -o /dev/null
else
    echo "SKIP ch08 mlir: mlir-opt not found"
fi

# ch13 saxpy
run python3 "$DIR/ch13_saxpy.py"

echo "Done: pass=$PASS fail=$FAIL"
exit $(( FAIL > 0 ? 1 : 0 ))
