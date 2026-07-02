#!/usr/bin/env python3
"""为 books/chapterXX-*.md 插入或更新「配套示例」章节。"""
from __future__ import annotations

import re
from pathlib import Path

BOOKS = Path(__file__).resolve().parents[1]

CHAPTERS = {
    0: {
        "rows": [
            ("cuda_add.cu", "0.5", "CUDA 线程级 vector add"),
            ("triton_add.py", "0.5", "Triton 块级 vector add（需 GPU）"),
        ],
        "note": "> 导读章节；Triton 示例在无 GPU 或无 triton 时自动跳过。",
    },
    1: {
        "rows": [
            ("hello.c", "1.5", "C → LLVM IR → 可执行"),
            ("test.mlir", "1.5", "MLIR canonicalize"),
            ("test.ttir", "1.5", "Triton IR（需 triton-opt）"),
            ("matmul.c", "作业2", "clang 生成矩阵乘法 IR"),
        ],
    },
    2: {
        "rows": [
            ("llvm_adt_demo.cpp", "2.x", "LLVM SmallVector 示例"),
            ("CMakeLists.txt", "配套", "构建配置"),
        ],
        "note": "> 以阅读 Triton 源码中的 C++ 惯用法为主；示例演示 LLVM ADT。",
    },
    3: {
        "rows": [
            ("example_before_ssa.c", "作业1", "SSA 转换前 C 代码"),
            ("example_ssa.ll", "作业1", "SSA 形式 LLVM IR"),
            ("codegen_visitor.py", "3.x", "简化 AST 代码生成"),
        ],
        "note": "> 概念章节；示例用于理解 SSA 与代码生成。",
    },
    4: {
        "rows": [
            ("factorial.ll", "4.4", "递归阶乘"),
            ("sum_array.ll", "4.4", "数组求和"),
            ("example.c", "4.5", "Clang 生成 IR"),
            ("gep_demo.ll", "4.3/作业3", "GEP 演示"),
            ("attributes_demo.ll", "4.6", "函数属性"),
            ("max.ll", "作业1", "max 函数"),
            ("matmul.c", "作业2", "矩阵乘法 IR"),
        ],
    },
    5: {
        "rows": [
            ("CMakeLists.txt", "配套", "构建所有 Pass 插件"),
            ("CountInstructionsPass.cpp", "5.2", "FunctionPass"),
            ("CountFunctionsPass.cpp", "5.2", "ModulePass"),
            ("MyPass.cpp", "5.3 A", "New PM 插件"),
            ("RunPipelineTool.cpp", "5.3 B", "内置 Pipeline"),
            ("InsertPrintfPass.cpp", "5.6", "插入 printf"),
            ("CountAddPass.cpp", "作业1", "统计 add"),
            ("AddToSubPass.cpp", "作业2", "add→sub"),
            ("test.ll", "5.7", "测试 IR"),
        ],
    },
    6: {
        "rows": [
            ("MiniDialect.td", "6.3", "TableGen Dialect/Op 定义"),
            ("run_examples.sh", "作业3", "mlir-tblgen 生成验证"),
        ],
    },
    7: {
        "rows": [("add.ll", "7.x", "llc 生成汇编 smoke test")],
        "note": "> 完整后端需 LLVM 源码树；示例用 `llc` 演示 IR→汇编。",
    },
    8: {
        "rows": [
            ("build_add.cpp", "8.10", "OpBuilder 构建 @add"),
            ("test_scf_if.mlir", "作业1", "scf.if 示例"),
            ("CMakeLists.txt", "配套", "链接 MLIR"),
        ],
    },
    9: {
        "rows": [
            ("sample.mlir", "9.x", "Mini Dialect IR 文本"),
            ("../chapter06/MiniDialect.td", "9.x", "TableGen 定义（复用 ch06）"),
        ],
    },
    10: {
        "rows": [
            ("sample.mlir", "10.x", "arith Pass 测试输入"),
            ("count_ops.py", "作业", "统计 Operation 数量"),
        ],
    },
    11: {
        "rows": [
            ("input.mlir", "11.x", "方言转换输入"),
            ("run_examples.sh", "11.x", "arith→llvm 降级演示"),
        ],
    },
    12: {
        "rows": [
            ("toy_frontend.py", "12.x", "Toy 前端原型"),
            ("test.toy.mlir", "12.8", "等价 MLIR"),
        ],
    },
    13: {
        "rows": [("vector_add.py", "13.x", "完整 vector add 内核（需 GPU）")],
    },
    14: {
        "rows": [
            ("simple_kernel.py", "14.x", "最小 JIT 内核（需 GPU）"),
            ("fixtures/simple.ttir", "14.x", "静态 TTIR fixture"),
        ],
    },
    15: {
        "rows": [("example.ttir", "15.5", "完整 TTIR 示例")],
    },
    16: {
        "rows": [
            ("vector_add.ttgir", "16.4", "带 #blocked 的 TTGIR"),
            ("encoding_calc.py", "作业", "Blocked encoding 覆盖计算"),
        ],
    },
    17: {
        "rows": [
            ("sample.ttir", "17.x", "转换前 TTIR"),
            ("sample.ttgir", "17.x", "转换后 TTGIR"),
        ],
    },
    18: {
        "rows": [("sample.ptx", "18.x", "PTX 模式检查快照")],
    },
    19: {
        "rows": [
            ("inspect_cuda_backend.py", "19.x", "检查 CUDABackend 源码"),
            ("sample.ptx", "19.x", "PTX 头字段检查"),
        ],
    },
    20: {
        "rows": [
            ("minimal_backend/", "20.9", "最小可注册后端脚手架"),
            ("test_registration.py", "20.9", "验证 backends 注册"),
        ],
    },
    21: {
        "rows": [
            ("design_doc_template.md", "21.2", "设计文档模板"),
            ("scaffold/", "21.x", "目录脚手架"),
        ],
    },
    22: {
        "rows": [("sample_mygpu.mlir", "22.x", "自定义 Dialect IR 样例")],
    },
    23: {
        "rows": [("test/add.mlir", "23.x", "LIT 风格 MLIR 测试")],
    },
    24: {
        "rows": [("debug_dump.sh", "24.4", "IR dump 调试命令模板")],
    },
}


def make_section(ch: int, info: dict) -> str:
    lines = [
        "## 配套示例",
        "",
        f"本章可运行代码位于 `books/examples/chapter{ch:02d}/`：",
        "",
        "| 文件 | 章节 | 说明 |",
        "|------|------|------|",
    ]
    for f, sec, desc in info["rows"]:
        lines.append(f"| `{f}` | {sec} | {desc} |")
    lines += [
        "",
        "运行：",
        "",
        "```bash",
        f"cd books/examples/chapter{ch:02d}",
        "./run_examples.sh",
        "```",
        "",
        "一键验证全书示例：",
        "",
        "```bash",
        "cd books/examples && ./run_all.sh",
        "```",
    ]
    if note := info.get("note"):
        lines += ["", note]
    lines += ["", "---", ""]
    return "\n".join(lines)


def patch_file(path: Path, section: str) -> None:
    text = path.read_text(encoding="utf-8")
    if "## 配套示例" in text:
        text = re.sub(
            r"## 配套示例\n.*?\n---\n\n",
            section,
            text,
            count=1,
            flags=re.S,
        )
    else:
        m = re.search(r"---\n\n", text)
        if m:
            text = text[: m.end()] + section + text[m.end() :]
        else:
            text = section + text
    path.write_text(text, encoding="utf-8")


def main():
    for ch, info in CHAPTERS.items():
        files = list(BOOKS.glob(f"chapter{ch:02d}-*.md"))
        if not files:
            print(f"skip ch{ch:02d}: no markdown")
            continue
        patch_file(files[0], make_section(ch, info))
        print(f"patched {files[0].name}")


if __name__ == "__main__":
    main()
