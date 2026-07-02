# 《驯龙高手 — LLVM 与 Triton 编译器从入门到后端实战》

> **副标题：手把手教你理解 LLVM 编译框架，掌握 Triton 编译器开发，实现第三方硬件后端**

---

## 关于本书

### 目标读者

本书面向以下读者：

- **GPU 算子库开发者**（如 FlagGems 项目参与者）：需要理解 Triton 内部机制，以便适配国产 GPU
- **编译器入门者**：有 C/Python 基础，想进入 LLVM/MLIR 世界
- **AI 框架开发者**：需要定制 GPU 内核编译流程
- **硬件厂商工程师**：需要在自家硬件上实现 Triton 编译器后端

### 你将从本书学到

1. ✅ 读懂 LLVM IR，编写 LLVM Pass
2. ✅ 掌握 MLIR 框架，定义自己的 Dialect
3. ✅ 理解 Triton 编译器的完整工作流程
4. ✅ **在第三方硬件上实现 Triton 编译器后端**（最终目标）

### 前置知识

| 知识点 | 要求 | 不满足怎么办 |
|--------|------|-------------|
| C 语言基础 | 熟练 | 先学 C Primer 前 10 章 |
| Python 基础 | 熟练 | 先学 Python 基础语法 |
| GPU 编程概念 | 了解概念即可 | 第 2 章有补充 |
| 数据结构 | 了解数组/链表/树 | 随用随学 |

### 怎样使用本书

```
第零部分（第 0-1 章）→ 了解背景，搭建环境
     ↓
第一部分（第 2-3 章）→ 补 C++、补编译器基础
     ↓
第二部分（第 4-7 章）→ 深入 LLVM
     ↓
第三部分（第 8-12 章）→ 掌握 MLIR
     ↓
第四部分（第 13-19 章）→ 剖析 Triton
     ↓
第五部分（第 20-24 章）→ 🏆 实现自定义后端
```

每章包含：
- **学习目标**：本章结束后你能做什么
- **核心概念**：需要掌握的关键知识点
- **配套示例**：`books/examples/chapterXX/` 下可运行的代码与脚本（见下方）
- **实战代码**：正文中的代码片段与示例文件交叉引用
- **课后作业**：巩固练习（参考答案见[附录 A](./appendix-a-homework-answers.md)）
- **本章小结**：关键要点回顾

### 配套示例

全书 25 章（第 0–24 章）均在 `books/examples/chapterXX/` 提供可运行示例，每章目录下有 `run_examples.sh`：

```bash
# 运行某一章
cd books/examples/chapter05
./run_examples.sh

# 一键验证全书示例（chapter00–24）
cd books/examples && ./run_all.sh
```

部分 C++ 章节（第 2、5、8 章）依赖系统 LLVM/MLIR 头文件。若 Cursor / VS Code 无法跳转到 `#include "llvm/..."`，请参阅[第 1 章 §1.6](./chapter01-环境搭建与工具链.md#16-ide-代码跳转cursor--vs-code) 生成 `compile_commands.json`。

### 代号约定

```
💡    提示 / 补充说明
⚠️    常见陷阱 / 注意事项
🏆    终极目标：实现自定义后端
📝    课后作业
🔑    关键概念
```

---

## 全书目录

### 第零部分：绪论与准备

| 章 | 标题 | 内容 | 难度 |
|----|------|------|------|
| [第 0 章](./chapter00-前言与导读.md) | 前言与导读 | 编译器发展简史、LLVM 与 Triton 的定位、本书目标 | ⭐ |
| [第 1 章](./chapter01-环境搭建与工具链.md) | 环境搭建与工具链 | 安装 LLVM/MLIR/Triton、搭建开发环境、第一个编译实验 | ⭐⭐ |

### 第一部分：编译器基础

| 章 | 标题 | 内容 | 难度 |
|----|------|------|------|
| [第 2 章](./chapter02-C++编译器开发特训.md) | C++ 编译器开发特训 | 编译器场景下最常用的 C++ 特性、LLVM ADT | ⭐⭐ |
| [第 3 章](./chapter03-编译原理速通.md) | 编译原理速通 | 前端/中端/后端、AST/IR/代码生成、SSA 形式 | ⭐⭐ |

### 第二部分：LLVM 深入

| 章 | 标题 | 内容 | 难度 |
|----|------|------|------|
| [第 4 章](./chapter04-LLVM-IR详解.md) | LLVM IR 详解 | IR 语法、类型系统、指令集、SSA、控制流 | ⭐⭐⭐ |
| [第 5 章](./chapter05-LLVM-Pass框架.md) | LLVM Pass 框架 | FunctionPass/ModulePass、分析 Pass 与转换 Pass、passes 命令行 | ⭐⭐⭐ |
| [第 6 章](./chapter06-LLVM-TableGen.md) | LLVM TableGen 入门 | TableGen 语法、记录与类、目标描述文件 | ⭐⭐⭐ |
| [第 7 章](./chapter07-LLVM后端与代码生成.md) | LLVM 后端与代码生成 | SelectionDAG、寄存器分配、指令选择，**自定义后端最小示例** | ⭐⭐⭐⭐ |

### 第三部分：MLIR 框架

| 章 | 标题 | 内容 | 难度 |
|----|------|------|------|
| [第 8 章](./chapter08-MLIR核心概念.md) | MLIR 核心概念 | Dialect/Operation/Type/Attribute/Region | ⭐⭐⭐ |
| [第 9 章](./chapter09-用TableGen定义Dialect.md) | 用 TableGen 定义 Dialect | Op 定义、Type 定义、Attr 定义、mlir-tblgen | ⭐⭐⭐ |
| [第 10 章](./chapter10-MLIR-Pass编程.md) | MLIR Pass 编程 | Pattern Rewrite、Walk、Greedy Rewrite Engine | ⭐⭐⭐ |
| [第 11 章](./chapter11-方言转换框架.md) | 方言转换框架 | ConversionTarget/TypeConverter/ConversionPattern | ⭐⭐⭐⭐ |
| [第 12 章](./chapter12-实战-写一个Toy编译器.md) | 🏆 实战：写一个 Toy 编译器 | 从零实现一个迷你语言编译器 | ⭐⭐⭐⭐ |

### 第四部分：Triton 深度剖析

| 章 | 标题 | 内容 | 难度 |
|----|------|------|------|
| [第 13 章](./chapter13-Triton编程模型.md) | Triton 编程模型 | 块级编程、@triton.jit、DSL 语法 | ⭐⭐ |
| [第 14 章](./chapter14-Triton编译流水线.md) | Triton 编译流水线 | compile()、ASTSource、Backend、stages | ⭐⭐⭐ |
| [第 15 章](./chapter15-Triton-IR-TT.md) | Triton IR (TT Dialect) | 类型系统、Ops 详解、TableGen 定义 | ⭐⭐⭐⭐ |
| [第 16 章](./chapter16-TritonGPU-IR与数据布局.md) | TritonGPU IR 与数据布局 | Encoding、LinearLayout、Coalesce | ⭐⭐⭐⭐ |
| [第 17 章](./chapter17-Triton到TritonGPU转换.md) | TT → TTG 转换 | TypeConverter、布局选择策略 | ⭐⭐⭐⭐ |
| [第 18 章](./chapter18-TritonGPU到LLVM降级.md) | TTG → LLVM 降级 | 内存分配、convert_layout 展开、Dot 降级 | ⭐⭐⭐⭐⭐ |
| [第 19 章](./chapter19-NVIDIA后端案例分析.md) | NVIDIA 后端案例分析 | CUDABackend、PTX 生成、ptxas | ⭐⭐⭐⭐ |

### 第五部分：打造第三方后端 🏆

| 章 | 标题 | 内容 | 难度 |
|----|------|------|------|
| [第 20 章](./chapter20-理解Triton后端接口.md) | 理解 Triton 后端接口 | BaseBackend 抽象类、需要实现的方法 | ⭐⭐⭐⭐ |
| [第 21 章](./chapter21-设计第三方后端.md) | 设计第三方后端 | 设计文档模板、选择目标指令集、确定 IR 层级 | ⭐⭐⭐⭐⭐ |
| [第 22 章](./chapter22-实现Dialect与转换.md) | 实现 Dialect 与转换 | 自定义 TTG → 目标 IR 转换 | ⭐⭐⭐⭐⭐ |
| [第 23 章](./chapter23-代码生成与测试.md) | 代码生成与测试 | 生成目标汇编、实现编译器测试 | ⭐⭐⭐⭐⭐ |
| [第 24 章](./chapter24-集成与调试.md) | 集成与调试 | 集成到 Python 运行时、调试技巧、性能分析 | ⭐⭐⭐⭐⭐ |

### 附录

| 标题 | 内容 |
|------|------|
| [附录 A：课后作业答案](./appendix-a-homework-answers.md) | 各章课后作业参考答案；代码答案在 `examples/appendix-a/`，可运行 `./run_verify.sh` 验证 |
| [附录 B：LLVM/MLIR 速查表](./appendix-b-quick-reference.md) | 常用 API、命令行、TableGen 语法速查 |
| [附录 C：Triton 关键代码索引](../markdown_doc/12-附录-关键代码文件索引.md) | Triton 源码文件速查 |
| [附录 D：参考书目与资源](./appendix-d-references.md) | 推荐阅读清单 |

---

## 如何贡献

发现错误或有改进建议？欢迎在项目仓库提交 Issue 或 PR。

---

## 关于"驯龙"之旅

本书的每一部分对应驯龙的一个阶段：

```
第零部分 "初识龙穴"  — 了解编译器生态，搭建驯龙装备
第一部分 "磨炼刀剑"  — C++ 和编译原理的基本功
第二部分 "触摸龙骨"  — 理解 LLVM IR 的骨骼结构
第三部分 "掌握龙语"  — 学习 MLIR 的"龙之语言"（TableGen）
第四部分 "驾驭海神"  — 精通 Triton 编译器的每一环
第五部分 "征服新龙"  — 🏆 在未知硬件上实现编译器后端
```

准备好了吗？驯龙之旅开始。

---

*开始阅读：[第 0 章 — 前言与导读](./chapter00-前言与导读.md)*
