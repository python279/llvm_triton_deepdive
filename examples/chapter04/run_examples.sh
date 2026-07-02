#!/usr/bin/env bash
# 构建并运行 chapter04 全部示例。需要 LLVM 工具链（clang、llvm-as、llvm-dis、lli、opt）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p "${BUILD_DIR}"

find_llvm_bin() {
    if command -v llvm-as >/dev/null 2>&1; then
        llvm-config --bindir 2>/dev/null || dirname "$(command -v llvm-as)"
    elif [[ -x /opt/homebrew/opt/llvm/bin/llvm-as ]]; then
        echo "/opt/homebrew/opt/llvm/bin"
    elif [[ -x /usr/local/opt/llvm/bin/llvm-as ]]; then
        echo "/usr/local/opt/llvm/bin"
    else
        echo ""
    fi
}

LLVM_BIN="$(find_llvm_bin)"
if [[ -z "${LLVM_BIN}" ]]; then
    echo "error: 找不到 LLVM 工具链，请安装 LLVM（brew install llvm）" >&2
    exit 1
fi

LLVM_AS="${LLVM_BIN}/llvm-as"
LLVM_DIS="${LLVM_BIN}/llvm-dis"
LLI="${LLVM_BIN}/lli"
OPT="${LLVM_BIN}/opt"

CLANG=""
if command -v clang >/dev/null 2>&1; then
    CLANG="clang"
elif [[ -x "${LLVM_BIN}/clang" ]]; then
    CLANG="${LLVM_BIN}/clang"
else
    echo "error: 找不到 clang" >&2
    exit 1
fi

run() {
    echo
    echo "==> $*"
    "$@"
}

assemble_and_run() {
    local name="$1"
    local expected="$2"
    run "${LLVM_AS}" "${SCRIPT_DIR}/${name}.ll" -o "${BUILD_DIR}/${name}.bc"
    run "${LLVM_DIS}" "${BUILD_DIR}/${name}.bc" -o "${BUILD_DIR}/${name}_roundtrip.ll"
    local exit_code=0
    "${LLI}" "${BUILD_DIR}/${name}.bc" || exit_code=$?
    if [[ "${exit_code}" -ne "${expected}" ]]; then
        echo "error: ${name} 期望退出码 ${expected}，实际 ${exit_code}" >&2
        exit 1
    fi
    echo "    OK: ${name} 退出码 = ${expected}"
}

echo "==> 4.1 llvm-as / llvm-dis 互转"
assemble_and_run factorial 120

echo
echo "==> 4.4 sum_array"
assemble_and_run sum_array 10

echo
echo "==> 作业 1 max"
assemble_and_run max 7

echo
echo "==> 4.3 / 作业 3 GEP 演示"
run "${OPT}" -S "${SCRIPT_DIR}/gep_demo.ll" -disable-output

echo
echo "==> 4.6 函数属性"
assemble_and_run attributes_demo 3

echo
echo "==> 4.5 Clang 生成 LLVM IR"
run "${CLANG}" -S -emit-llvm -O0 "${SCRIPT_DIR}/example.c" \
    -o "${BUILD_DIR}/example_O0.ll"
run "${CLANG}" -S -emit-llvm -O2 "${SCRIPT_DIR}/example.c" \
    -o "${BUILD_DIR}/example_O2.ll"
grep -q 'define.*@add' "${BUILD_DIR}/example_O0.ll"
grep -q 'define.*@main' "${BUILD_DIR}/example_O0.ll"
echo "    OK: example.c 已生成 LLVM IR"

echo
echo "==> 作业 2 matmul IR"
run "${CLANG}" -S -emit-llvm -O0 "${SCRIPT_DIR}/matmul.c" \
    -o "${BUILD_DIR}/matmul_O0.ll"
grep -q 'define.*@matmul' "${BUILD_DIR}/matmul_O0.ll"
echo "    OK: matmul.c 已生成 LLVM IR"

echo
echo "全部示例通过。"
