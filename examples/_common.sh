#!/usr/bin/env bash
# 各章 run_examples.sh 共用的工具检测
find_llvm_bin() {
    if command -v llvm-config >/dev/null 2>&1; then
        llvm-config --bindir 2>/dev/null || dirname "$(command -v opt)"
    elif [[ -x /opt/homebrew/opt/llvm/bin/opt ]]; then
        echo "/opt/homebrew/opt/llvm/bin"
    elif [[ -x /usr/local/opt/llvm/bin/opt ]]; then
        echo "/usr/local/opt/llvm/bin"
    else
        echo ""
    fi
}

find_clang() {
    if command -v clang >/dev/null 2>&1; then
        echo "clang"
    elif [[ -n "${LLVM_BIN:-}" && -x "${LLVM_BIN}/clang" ]]; then
        echo "${LLVM_BIN}/clang"
    else
        echo ""
    fi
}

find_triton_opt() {
    if command -v triton-opt >/dev/null 2>&1; then
        command -v triton-opt
        return
    fi
    local build_dir
    build_dir="$(PYTHONPATH="${TRITON_ROOT:-}/python" python3 -c \
        'from build_helpers import get_cmake_dir; print(get_cmake_dir())' 2>/dev/null || true)"
    if [[ -n "${build_dir}" && -x "${build_dir}/bin/triton-opt" ]]; then
        echo "${build_dir}/bin/triton-opt"
    fi
}

skip_msg() { echo "    SKIP: $*"; }
