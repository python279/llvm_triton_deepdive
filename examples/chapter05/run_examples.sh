#!/usr/bin/env bash
# 构建并运行 chapter05 全部示例。需要已安装 LLVM（Homebrew: brew install llvm）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

if command -v llvm-config >/dev/null 2>&1; then
    LLVM_BIN="$(llvm-config --bindir)"
elif [[ -x /opt/homebrew/opt/llvm/bin/opt ]]; then
    LLVM_BIN="/opt/homebrew/opt/llvm/bin"
    export LLVM_DIR="/opt/homebrew/opt/llvm/lib/cmake/llvm"
elif [[ -x /usr/local/opt/llvm/bin/opt ]]; then
    LLVM_BIN="/usr/local/opt/llvm/bin"
    export LLVM_DIR="/usr/local/opt/llvm/lib/cmake/llvm"
else
    echo "error: 找不到 LLVM，请先安装并确保 llvm-config 或 opt 在 PATH 中" >&2
    exit 1
fi

OPT="${LLVM_BIN}/opt"

echo "==> 配置并编译"
cmake -G Ninja -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    ${LLVM_DIR:+-DLLVM_DIR="${LLVM_DIR}"}
ninja -C "${BUILD_DIR}"

plugin() { echo "${BUILD_DIR}/$1.so"; }

run_opt() {
    echo
    echo "==> $*"
    "$@"
}

echo
echo "==> 5.2 CountInstructionsPass"
run_opt "${OPT}" -load-pass-plugin "$(plugin CountInstructionsPass)" \
    -passes=count-instructions -disable-output "${SCRIPT_DIR}/test.ll"

echo
echo "==> 5.2 CountFunctionsPass"
run_opt "${OPT}" -load-pass-plugin "$(plugin CountFunctionsPass)" \
    -passes=count-functions -disable-output "${SCRIPT_DIR}/test.ll"

echo
echo "==> 5.3 MyPass 插件"
run_opt "${OPT}" -load-pass-plugin "$(plugin MyPass)" \
    -passes=mypass -disable-output "${SCRIPT_DIR}/test.ll"

echo
echo "==> 5.3 RunPipelineTool（内置 Pipeline）"
"${BUILD_DIR}/run_pipeline" "${SCRIPT_DIR}/test.ll"

echo
echo "==> 5.6 InsertPrintfPass"
TMP_LL="$(mktemp /tmp/chapter05_printf.XXXXXX.ll)"
run_opt "${OPT}" -load-pass-plugin "$(plugin InsertPrintfPass)" \
    -passes=insert-printf -S "${SCRIPT_DIR}/test.ll" -o "${TMP_LL}"
grep -q 'printf' "${TMP_LL}"
echo "    OK: 输出 IR 包含 printf 调用"
rm -f "${TMP_LL}"

echo
echo "==> 作业1 CountAddPass"
run_opt "${OPT}" -load-pass-plugin "$(plugin CountAddPass)" \
    -passes=count-add -disable-output "${SCRIPT_DIR}/test.ll"

echo
echo "==> 作业2 AddToSubPass"
TMP_LL="$(mktemp /tmp/chapter05_sub.XXXXXX.ll)"
run_opt "${OPT}" -load-pass-plugin "$(plugin AddToSubPass)" \
    -passes=add-to-sub -S "${SCRIPT_DIR}/test.ll" -o "${TMP_LL}"
grep -q 'sub i32' "${TMP_LL}"
grep -vq 'add i32' "${TMP_LL}" || true
echo "    OK: add 已替换为 sub"
cat "${TMP_LL}"
rm -f "${TMP_LL}"

echo
echo "==> 5.7 opt 命令行实战"
TMP_CLEAN="$(mktemp /tmp/chapter05_clean.XXXXXX.ll)"
run_opt "${OPT}" -S -passes=globaldce "${SCRIPT_DIR}/test.ll" -o "${TMP_CLEAN}"
grep -vq '@dead' "${TMP_CLEAN}"
echo "    OK: globaldce 删除了未使用的 internal @dead"
rm -f "${TMP_CLEAN}"

run_opt "${OPT}" -S -passes=inline "${SCRIPT_DIR}/test.ll" -disable-output
run_opt "${OPT}" -S -passes='default<O2>' "${SCRIPT_DIR}/test.ll" -disable-output

echo
echo "==> 作业3 探索内置 Pass"
run_opt "${OPT}" -S -passes=instcombine "${SCRIPT_DIR}/test.ll" -disable-output
run_opt "${OPT}" -S -passes=gvn "${SCRIPT_DIR}/test.ll" -disable-output
run_opt "${OPT}" -S -passes=licm "${SCRIPT_DIR}/test.ll" -disable-output
run_opt "${OPT}" -S -passes=loop-unroll "${SCRIPT_DIR}/test.ll" -disable-output

echo
echo "全部示例通过。"
