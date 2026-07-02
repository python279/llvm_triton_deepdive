# 第 1 章：环境搭建与工具链

> **本章目标**：在本地搭建完整的 LLVM/MLIR/Triton 开发环境，完成第一个编译实验。

> 驯龙手记：驯龙的第一步是认识你的工具——锻造装备、磨利刀剑。
> 本章结束后，你就有了属于自己的"驯龙套装"。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter01/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `hello.c` | 1.5 | C → LLVM IR → 可执行 |
| `test.mlir` | 1.5 | MLIR canonicalize |
| `test.ttir` | 1.5 | Triton IR（需 triton-opt） |
| `matmul.c` | 作业2 | clang 生成矩阵乘法 IR |

运行：

```bash
cd books/examples/chapter01
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 1.1 环境概览

### 我们需要什么

| 工具 | 版本要求 | 用途 |
|------|---------|------|
| C++ 编译器 | Clang 15+ / GCC 9+ | 编译 C++ 代码 |
| CMake | 3.20+ | 构建系统 |
| Ninja | 1.11+ | 快速构建 |
| LLVM/MLIR | 18.x+（含 MLIR） | 编译基础设施 |
| Python | 3.9+ | Triton 前端 |
| CUDA Toolkit | 12.x+（可选） | NVIDIA GPU 支持 |
| Git | 最新 | 源码管理 |

### 架构兼容性

```
你的开发机：
├── ✅ C++ 编译器  — 必备
├── ✅ LLVM/MLIR  — 必备（可用预编译包）
├── ✅ Python 3   — 必备
├── ✅ GPU 驱动   — 如需运行 Triton 内核
├── ✅ CUDA       — 如需 NVIDIA GPU
└── ✅ ROCm       — 如需 AMD GPU
```

> 💡 **没有 GPU 也能学！** 本书的前 20 章都可以在 CPU 上完成。只有实际运行 Triton 内核才需要 GPU。LIT 测试和 MLIR Pass 开发完全在 CPU 上。

## 1.2 安装 LLVM/MLIR

### 方案 A：使用包管理器（推荐）

```bash
# macOS (Homebrew)
brew install llvm
export LLVM_DIR=$(brew --prefix llvm)/lib/cmake/llvm
export MLIR_DIR=$(brew --prefix llvm)/lib/cmake/mlir
export PATH=$(brew --prefix llvm)/bin:$PATH

# Ubuntu/Debian（注意：apt 版本可能较旧）
# 建议从 LLVM 官方仓库安装
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 18
sudo apt install libmlir-18-dev mlir-18-tools

# Arch Linux
sudo pacman -S llvm mlir
```

### 方案 B：从源码编译（完整但耗时）

```bash
# 1. 克隆 LLVM 项目（含 MLIR）
git clone --depth 1 --branch llvmorg-18.1.0 https://github.com/llvm/llvm-project.git
cd llvm-project

# 2. 配置 CMake
cmake -G Ninja -S llvm -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
    -DLLVM_ENABLE_RTTI=ON \
    -DLLVM_INSTALL_UTILS=ON

# 3. 编译（根据机器配置可能需要 30-60 分钟）
ninja -C build -j$(nproc)

# 4. 安装
ninja -C build install
```

### 验证安装

```bash
# 验证 LLVM
llvm-config --version      # 应输出 18.x

# 验证 MLIR
mlir-opt --version         # 应输出 MLIR 版本信息

# 验证 LLVM IR 编译
echo 'int main() { return 0; }' | clang -x c - -S -emit-llvm -o -
```

## 1.3 安装 Triton

### 从源码安装（推荐）

```bash
# 克隆 Triton
git clone https://github.com/triton-lang/triton.git
cd triton

# 设置 LLVM 路径
export LLVM_DIR=/path/to/llvm/lib/cmake/llvm
export MLIR_DIR=/path/to/llvm/lib/cmake/mlir

# 安装 Python 包
pip install -e python

# 验证
python -c "import triton; print(triton.__version__)"
```

### 常见问题

```bash
# 问题 1：找不到 LLVM
# CMake Error: Could not find LLVM
# 解决：确保设置了 LLVM_DIR
export LLVM_DIR=/opt/homebrew/opt/llvm/lib/cmake/llvm

# 问题 2：MLIR 未启用
# 解决：编译 LLVM 时加上 -DLLVM_ENABLE_PROJECTS="mlir"

# 问题 3：编译内存不足
# 解决：减少并行度
pip install -e python -- -j2
```

## 1.4 核心工具介绍

### `clang` — C/C++ 编译器

```bash
# 生成 LLVM IR（文本格式）
clang -S -emit-llvm hello.c -o hello.ll

# 生成 LLVM IR（二进制位码）
clang -c -emit-llvm hello.c -o hello.bc

# 查看优化后的 IR
clang -O2 -S -emit-llvm hello.c -o hello_opt.ll
```

### `opt` — LLVM IR 优化器

```bash
# 应用指定的 Pass
opt -S -passes=instcombine input.ll -o output.ll

# 查看所有可用的 Pass
opt --print-passes

# 应用标准 O2 优化
opt -S -passes=default<O2> input.ll -o output.ll
```

### `llc` — LLVM 静态编译器

```bash
# LLVM IR → 汇编
llc input.ll -o input.s

# LLVM IR → 汇编（指定目标架构）
llc -mtriple=nvptx64-nvidia-cuda input.ll -o input.ptx
```

### `mlir-opt` — MLIR 优化器

```bash
# 运行 MLIR Pass
mlir-opt input.mlir -o output.mlir

# 查看 IR 统计
mlir-opt --print-op-stats input.mlir
```

### `mlir-translate` — MLIR 与 LLVM IR 互转

```bash
# MLIR → LLVM IR
mlir-translate --mlir-to-llvmir input.mlir -o output.ll

# LLVM IR → MLIR
mlir-translate --llvmir-to-mlir input.ll -o output.mlir
```

### `triton-opt` — Triton 专有 Pass 运行器

```bash
# 查看目前 Triton 注册了哪些 Pass
triton-opt --help | grep triton

# 运行 TritonGPU 优化 Pass
triton-opt input.ttgir --tritongpu-coalesce --tritongpu-accelerate-matmul
```

## 1.5 第一个实验：亲历编译流程

### 实验 1：C → LLVM IR → 汇编

完整源码：`books/examples/chapter01/hello.c`

```c
#include <stdio.h>
int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(1, 2);
    printf("Result: %d\n", result);
    return 0;
}
```

```bash
clang -S -emit-llvm -O0 hello.c -o hello.ll
llc hello.ll -o hello.s
clang hello.c -o hello && ./hello
```

### 实验 2：运行一个 MLIR Pass

完整源码：`books/examples/chapter01/test.mlir`

```mlir
func.func @add(%a: i32, %b: i32) -> i32 {
    %0 = arith.addi %a, %b : i32
    func.return %0 : i32
}
```

```bash
mlir-opt test.mlir --canonicalize -o test_canon.mlir
```

### 实验 3：使用 triton-opt

完整源码：`books/examples/chapter01/test.ttir`

```mlir
module {
  tt.func @add_kernel(%arg0: !tt.ptr<f32>) {
    %0 = tt.get_program_id x : i32
    tt.return
  }
}
```

```bash
triton-opt test.ttir --canonicalize -o /dev/null
```

## 1.6 IDE 代码跳转（Cursor / VS Code）

本书部分章节的 C++ 示例依赖系统安装的 LLVM/MLIR 头文件（例如 `#include "llvm/IR/Function.h"`）。**能正常编译，不代表 IDE 能跳转到这些头文件**——Cursor / VS Code 的 C++ 语言服务器（clangd 或 Microsoft C/C++ 扩展）需要知道每个源文件的 `-I`、`-D` 等编译参数，否则会把 `#include` 标红，Cmd+Click 也无法跳转。

### 原因

| 现象 | 原因 |
|------|------|
| `#include "llvm/..."` 报找不到文件 | LLVM 头文件在 Homebrew / 系统安装目录，不在本书仓库内 |
| 编译通过但无法跳转 | CMake 项目未导出 `compile_commands.json`，IDE 缺少编译数据库 |

### 已配置的章节

以下章节在 `CMakeLists.txt` 中已开启 `CMAKE_EXPORT_COMPILE_COMMANDS`，并在章节目录下提供了 `compile_commands.json` 软链接：

| 章节 | 目录 | 依赖 |
|------|------|------|
| 第 2 章 | `books/examples/chapter02/` | LLVM |
| 第 5 章 | `books/examples/chapter05/` | LLVM |
| 第 8 章 | `books/examples/chapter08/` | LLVM + MLIR |

首次克隆仓库后，或修改了 CMake 配置，需要重新配置并生成编译数据库：

```bash
# 以 chapter05 为例，chapter02 / chapter08 同理
cd books/examples/chapter05
cmake -G Ninja -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR="$(llvm-config --cmakedir)"
ln -sf build/compile_commands.json compile_commands.json
```

chapter08 还需指定 MLIR：

```bash
cd books/examples/chapter08
cmake -G Ninja -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_DIR="$(llvm-config --cmakedir)" \
    -DMLIR_DIR="$(dirname $(llvm-config --cmakedir))/../mlir/lib/cmake/mlir"
ln -sf build/compile_commands.json compile_commands.json
```

也可以直接运行各章的 `run_examples.sh`，它会自动执行 CMake 配置与编译。

### 让 IDE 生效

1. 用 Cursor / VS Code **打开仓库根目录**（`triton/`），不要只打开单个章节文件夹。
2. 执行 **Cmd+Shift+P → “clangd: Restart language server”**（若使用 Microsoft C/C++ 扩展，则选 “C/C++: Reset IntelliSense Database”）。
3. 对 `llvm/IR/Function.h` 等头文件再试 Cmd+Click 跳转。

> 💡 clangd 会从当前文件所在目录**向上**查找 `compile_commands.json`，因此各章节目录下的软链接即可被自动发现，无需在仓库根目录再建一份。

### 其他章节

第 1、3、4 章的 `.c` 文件通过 `run_examples.sh` 直接用 `clang` 编译，没有 CMake 项目，不适用上述方案。这些文件只包含标准 C 头文件，一般无需额外配置。若你自行添加了依赖 LLVM 的 C++ 代码，可在对应目录的 `CMakeLists.txt` 中加入：

```cmake
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
```

## 1.7 开发环境检查清单

```
□ clang++ 可以编译 C++17 代码
□ LLVM_DIR 环境变量已设置
□ MLIR_DIR 环境变量已设置
□ mlir-opt 可执行
□ triton-opt 可执行
□ Python 可以 import triton
□ git 可以 clone 仓库
□ cmake 版本 >= 3.20
□ ninja 已安装
□ （可选）chapter02/05/08 已生成 compile_commands.json，IDE 可跳转到 LLVM/MLIR 头文件
```

## 1.8 GPU 开发环境（可选）

如果你有 NVIDIA GPU：

```bash
# 检查 CUDA
nvidia-smi
nvcc --version

# 安装 PyTorch（用于测试 Triton 内核）
pip install torch
```

如果你有 AMD GPU：

```bash
# 检查 ROCm
rocm-smi
```

> 没有 GPU 完全不影响前 20 章的学习。第 21-24 章（自定义后端）也主要在 CPU 上验证。

---

## 📝 课后作业

### 作业 1：搭建环境

按照本章说明搭建开发环境，运行 `mlir-opt --version` 和 `triton-opt --version`，截图保存。

### 作业 2：生成 LLVM IR

编写一个 C 程序 `matmul.c`（简单的 4x4 矩阵乘法），用 clang 生成 LLVM IR，观察生成的 IR 结构，找找看 `load`、`store`、`mul`、`add` 指令在哪里。

### 作业 3：探索 triton-opt

```bash
triton-opt --help | grep "triton" | wc -l
# 输出 Triton 注册了多少个 Pass？列出前 10 个。
```

---

## 本章小结

- 开发环境包括 LLVM/MLIR（编译基础设施）和 Triton（上层编译器）
- 没有 GPU 也能学，LIT 测试和 Pass 开发都在 CPU 上运行
- C++ 示例章节（02/05/08）通过 `compile_commands.json` 配置 IDE 代码跳转
- 核心工具链：`clang`（前端）→ `opt`（优化）→ `llc`（代码生成）
- MLIR 工具链：`mlir-opt`（优化器）→ `mlir-translate`（IR 转换）
- Triton 工具链：`triton-opt`（Triton Pass 调试器）
