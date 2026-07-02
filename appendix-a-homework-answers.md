# 附录 A：课后作业参考答案

> 建议先独立完成作业后再查阅。带代码的答案位于 `books/examples/appendix-a/`，可用 `./run_verify.sh` 批量验证（需已安装 LLVM/MLIR；GPU 示例在无 CUDA 时自动 SKIP）。

---

## 第 0 章：前言与导读

### 作业 1

手绘对比图即可，核心要点：

- **传统编译器**：前端 → 单一 IR → 后端，各编译器各自为政
- **LLVM 三段式**：前端 → **统一 LLVM IR** → 优化 → 后端，IR 是共享接口

### 作业 2

MLIR 的多级 IR 对 GPU 编译特别有用的原因：

1. **矩阵乘法有多个优化层级**：Triton IR 描述块级运算，TTG IR 描述 Tensor Core 布局，LLVM IR 处理指令调度——每层关注不同优化问题
2. **GPU 特有抽象**：TritonGPU Encoding 直接表达线程-数据映射，LLVM IR 难以高效表达
3. **渐进降级**：TTG → LLVM 时，布局转换可逐步拆解为 shuffle / 共享内存操作

### 作业 3

- **LLVM**：模块化编译器基础设施，提供统一 IR 与优化/代码生成框架
- **MLIR**：在 LLVM 之上的多级 IR 框架，用 Dialect 表达不同抽象层次

---

## 第 1 章：环境搭建与工具链

### 作业 1

按 §1.7 检查清单逐项确认，保存 `mlir-opt --version` 与 `triton-opt --version` 截图。

### 作业 2

用 clang 编译 4×4 矩阵乘法，LLVM IR 中可观察到：

- 嵌套循环 → 多个 `br`、`icmp`
- 内存访问 → `load` / `store` / `getelementptr`
- 计算 → `fmul` / `fadd`

完整源码：`books/examples/chapter01/matmul.c`

### 作业 3

```bash
triton-opt --help | grep -i triton | wc -l
triton-opt --help | grep -i triton | head -10
```

输出数量取决于 Triton 版本与构建选项；若结果为 0，说明当前 `triton-opt` 未启用 Triton Pass 或路径不对，需确认 `pip install -e python` 后使用的是项目内构建产物。

---

## 第 2 章：C++ 编译器开发特训

### 作业 1：LLVM ADT 练习

完整可编译实现：`books/examples/appendix-a/ch02_process_adt.cpp`

```cpp
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

SmallVector<int> process(ArrayRef<int> values) {
    SmallVector<int> result;
    for (auto [i, v] : llvm::enumerate(values)) {
        (void)i;
        result.push_back(v * 2);
    }
    return result;
}
```

验证：

```bash
c++ -std=c++17 $(llvm-config --cppflags --ldflags --libs support demangle) \
  books/examples/appendix-a/ch02_process_adt.cpp -o /tmp/ch02_hw && /tmp/ch02_hw
# 输出：246
```

### 作业 2：阅读源码

在 `lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp` 中：

1. 继承自 `ElementwiseOpConversionBase<arith::CmpIOp, CmpIOpConversion>`
2. `matchAndRewrite` 签名：`LogicalResult matchAndRewrite(SourceOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override`
3. 通过 `createDestOps` 将 `arith.cmpi` 的谓词映射为 `LLVM::ICmpPredicate`，创建 `LLVM::ICmpOp`

### 作业 3：dyn_cast 练习

在 `lib/Dialect/TritonGPU/Transforms/Coalesce.cpp` 中搜索 `dyn_cast` / `isa`，例如：

- `dyn_cast<triton::LoadOp>(op)` — 判断是否为 load 操作
- `isa<triton::StoreOp>(op)` — 判断是否为 store 操作

行号随 Triton 版本变化，以本地源码搜索为准。

---

## 第 3 章：编译原理速通

### 作业 1：SSA 转换

完整 IR：`books/examples/appendix-a/ch03_ssa_example.ll`

```llvm
define i32 @example(i32 %x, i32 %y) {
entry:
    %z1 = add i32 %x, %y
    %cond = icmp sgt i32 %z1, 0
    br i1 %cond, label %then, label %else
then:
    %z2 = mul i32 %z1, 2
    br label %merge
else:
    %z3 = mul i32 %z1, 3
    br label %merge
merge:
    %z = phi i32 [%z2, %then], [%z3, %else]
    ret i32 %z
}
```

验证：`llvm-as books/examples/appendix-a/ch03_ssa_example.ll -o /dev/null`

### 作业 2：画 CFG

```
        [entry: z1 = x+y, cond = z1>0]
               /              \
           z1>0               z1<=0
            /                    \
      [then: z2=z1*2]        [else: z3=z1*3]
            \                    /
             \                  /
              [merge: z = phi(z2, z3)]
                      |
                   [return z]
```

### 作业 3：概念对应

| 编译原理概念 | Triton 中的对应 |
|-------------|----------------|
| 词法分析 | Python 源码解析（由 Python 解释器完成） |
| 语法分析 | Python AST 解析 |
| IR | TT IR（Triton Dialect） |
| 中间表示优化 | `make_ttir()` 中的 Pass（内联、CSE、规范化） |
| 指令选择 | TTG → LLVM 降级（含 `convert_layout` 展开） |
| 寄存器分配 | LLVM 优化阶段（`llvm.optimize_module`） |

---

## 第 4 章：LLVM IR 详解

### 作业 1：max 函数

完整可运行 IR（含 `main`）：`books/examples/chapter04/max.ll`

```llvm
define i32 @max(i32 %a, i32 %b) {
entry:
    %cond = icmp sgt i32 %a, %b
    %r = select i1 %cond, i32 %a, i32 %b
    ret i32 %r
}

define i32 @main() {
    %r = call i32 @max(i32 3, i32 7)
    ret i32 %r
}
```

验证：

```bash
llvm-as books/examples/chapter04/max.ll -o /dev/null
lli books/examples/chapter04/max.ll   # 退出码 7
```

### 作业 2：读 LLVM IR

见 `books/examples/chapter04/matmul.c` 及 `clang -S -emit-llvm -O0 matmul.c` 生成的 IR。

### 作业 3：GEP 练习

`field2` = 结构体第 2 个字段（下标 1）：

```llvm
%ptr = getelementptr %MyStruct, %MyStruct* %arr, i64 3, i32 1, i64 2
```

完整演示：`books/examples/chapter04/gep_demo.ll`（`llvm-as` 验证通过）

---

## 第 5 章：LLVM Pass 框架

### 作业 1：统计 add 指令

完整可运行插件：`books/examples/chapter05/CountAddPass.cpp`

```bash
cd books/examples/chapter05 && ./run_examples.sh
# 或：
opt -load-pass-plugin ./build/CountAddPass.so \
    -passes=count-add -disable-output test.ll
# Function add has 1 add instructions
# Function dead has 0 add instructions
```

核心逻辑（New PM）：

```cpp
void countAdds(Function &F) {
    int count = 0;
    for (BasicBlock &BB : F)
        for (Instruction &I : BB)
            if (auto *BO = dyn_cast<BinaryOperator>(&I))
                if (BO->getOpcode() == Instruction::Add)
                    ++count;
    errs() << "Function " << F.getName() << " has " << count
           << " add instructions\n";
}
```

### 作业 2：add → sub 转换

完整实现：`books/examples/chapter05/AddToSubPass.cpp`

```bash
opt -load-pass-plugin ./build/AddToSubPass.so \
    -passes=add-to-sub -S test.ll -o test_sub.ll
```

### 作业 3：使用 opt 探索

```bash
opt --print-passes                          # 列出全部 Pass
opt -S -passes=instcombine input.ll -o out.ll
opt -S -passes='default<O2>' input.ll -o out.ll
```

---

## 第 6 章：LLVM TableGen

### 作业 1：读 TritonOps.td

1. `TT_LoadOp` 的 `arguments` 共 6 项：`ptr`（TTPtrLike）、`mask`（Optional TT_BoolLike）、`other`（Optional TT_Type）、`cache`、`evict`、`isVolatile`（后三项为带默认值的 Attr）
2. `TT_DotOp` 的 `hasVerifier = 1`
3. `getSingleCombiner` 在 `TT_ReduceOp` 的 `extraClassDeclaration` 中声明

### 作业 2：ClampIOp

```tablegen
def TT_ClampIOp : TT_Op<"clampi", [Elementwise, SameOperandsAndResultType, Pure]> {
    let summary = "Clamp operation for integer types";
    let arguments = (ins TT_IntLike:$x, TT_IntLike:$min, TT_IntLike:$max);
    let results = (outs TT_IntLike:$result);
    let assemblyFormat = "$x `,` $min `,` $max attr-dict `:` type($result)";
}
```

（作业仅要求 x/min/max；浮点版 `TT_ClampFOp` 额外有 `propagateNan` 属性。）

### 作业 3：运行 mlir-tblgen

```bash
cd books/examples/chapter06 && ./run_examples.sh
```

---

## 第 7 章：LLVM 后端与代码生成

### 作业 1：概念填空

| 概念 | 说明 |
|------|------|
| 指令选择 | 将 LLVM IR 映射为目标架构机器指令 |
| 寄存器分配 | 虚拟寄存器 → 物理寄存器 |
| 指令调度 | 重排指令以隐藏延迟 |
| 代码发射 | 生成汇编或目标二进制 |

### 作业 2：策略选择

- **已有 LLVM 后端的新 GPU** → 策略 A，复用 LLVM 指令选择与寄存器分配
- **只有 C 编译器的 AI 加速器** → 策略 C，生成 C++ 模板代码最可行

### 作业 3：阅读目标描述文件

在 LLVM 源码 `lib/Target/<Arch>/` 下找到 `.td` 文件，关注：

- `def XXX : Target` — 目标定义
- `def XXXInstrInfo` — 指令描述
- `def XXXRegisterInfo` — 寄存器定义

---

## 第 8 章：MLIR 核心概念

### 作业 1：分析 MLIR 代码

IR 文件：`books/examples/appendix-a/ch08_hw1_scf_if.mlir`（`mlir-opt --canonicalize` 验证通过）

1. **5 个 Operation**（不含 ModuleOp / func.func）：`arith.cmpi`、`scf.if`、`scf.yield`×2、`func.return`
2. **含 Region 的 Op**：`scf.if`（then / else 各一个 block）
3. **`%max` 的使用者**：`func.return`

> 注意：`scf.if` 的结果类型写作 `-> (i32)`，不是 `-> i32`。

### 作业 2：用 OpBuilder 创建 IR

对应 `%result = arith.addi %a, %b : i32`：

```cpp
Value result = builder.create<arith::AddIOp>(loc, a, b);
// 或显式类型：builder.create<arith::AddIOp>(loc, builder.getI32Type(), a, b);
```

### 作业 3：探索 Triton Dialect

| Dialect | 主要 Op | 用途 |
|---------|---------|------|
| Triton (tt) | load, store, dot, reduce, addptr | 设备无关块级操作 |
| TritonGPU (ttg) | convert_layout, local_load/store | GPU 数据布局 |
| TritonNvidiaGPU (ttnv) | wgmma, tma_load | NVIDIA 特有指令 |

---

## 第 9 章：用 TableGen 定义 Dialect

### 作业 1：Mathy Dialect

```tablegen
def Mathy_Dialect : Dialect { let name = "mathy"; /* ... */ }
class Mathy_Op<string mnemonic> : Op<Mathy_Dialect, mnemonic, [Pure]> {}

def Mathy_SqrtOp : Mathy_Op<"sqrt"> {
    let arguments = (ins F32:$x);
    let results = (outs F32:$result);
}
def Mathy_NegOp : Mathy_Op<"neg"> {
    let arguments = (ins F32:$x);
    let results = (outs F32:$result);
}
def Mathy_AbsOp : Mathy_Op<"abs"> {
    let arguments = (ins F32:$x, OptionalAttr<BoolAttr>:$fast);
    let results = (outs F32:$result);
}
```

### 作业 2：带 Region 的 Op

参考 `TT_ReduceOp`：用 `let regions = (region SizedRegion<1>:$combine)` 定义组合 Region。`mathy.fold` 需接收 tensor 输入、F32 输出，Region 内实现归约组合逻辑（具体语法见 ch09 §9.3）。

### 作业 3：TT_LoadOp

1. **操作数**（最多 3 个）：`ptr`（必需）、`mask`（可选）、`other`（可选）
2. **属性**（3 个）：`cache`、`evict`、`isVolatile`（TableGen 中写在 `arguments` 里但语义上是属性）
3. **无 Region**

---

## 第 10 章：MLIR Pass 编程

### 作业 1：Add → Sub 转换

> 以下为 Pattern **片段**，需嵌入完整 MLIR Pass 工程（注册 Pass、populate patterns、驱动 mlir-opt）方可编译运行。

```cpp
struct AddToSubPattern : public OpRewritePattern<arith::AddIOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::AddIOp op,
                                  PatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<arith::SubIOp>(op, op.getLhs(), op.getRhs());
        return success();
    }
};
```

### 作业 2：CoalescePass

在 `lib/Dialect/TritonGPU/Transforms/Coalesce.cpp` 中：

1. 继承 `impl::TritonGPUCoalesceBase<CoalescePass>`
2. `runOnOperation()` 执行 `ModuleAxisInfoAnalysis`，分析 load/store
3. `pickDescriptorLoadStoreLayout` 根据张量形状和线程数选择向量化宽度

### 作业 3：triton-opt 调试

```bash
triton-opt --mlir-print-ir-after-all input.ttgir 2>&1 | less
```

---

## 第 11 章：方言转换框架

### 作业 1：概念理解

1. **ConversionTarget**：声明哪些 Op/Type 在转换过程中合法或非法
2. **TypeConverter**：源类型 → 目标类型的映射规则
3. **OpConversionPattern**：单个 Op 的转换逻辑（通过 `OpAdaptor` 获取已转换的操作数）

### 作业 2：SqrtOpConversion 伪代码

```cpp
struct SqrtOpConversion : public OpConversionPattern<mathy::SqrtOp> {
    LogicalResult matchAndRewrite(mathy::SqrtOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
        auto log = rewriter.create<math::LogOp>(op.getLoc(), adaptor.getX());
        auto half = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getF32FloatAttr(0.5f));
        auto mul = rewriter.create<arith::MulFOp>(op.getLoc(), log, half);
        auto exp = rewriter.create<math::ExpOp>(op.getLoc(), mul);
        rewriter.replaceOp(op, exp);
        return success();
    }
};
```

### 作业 3

`populateArithPatternsAndLegality` 对多种 `arith` Op 设置动态合法性；`arith::ConstantOp` 需特化（`ArithConstantPattern`），因其 `DenseElementsAttr` 必须按目标 Encoding 重塑后才能合法。

---

## 第 12 章：Toy 编译器

### 作业 3：拓张为张量语言

至少需修改：类型系统（tensor）、新增 load/store/reduce/dot Op、块级并行语义（program_id）、TTG 编码属性、Coalesce 等布局 Pass。

---

## 第 13 章：Triton 编程模型

### 作业 1：SAXPY

完整可运行实现：`books/examples/appendix-a/ch13_saxpy.py`

```python
@triton.jit
def saxpy_kernel(y_ptr, a, x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, a * x + y, mask=mask)

def saxpy(a, x, y):
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    saxpy_kernel[grid](y, a, x, n, BLOCK_SIZE=1024)
    return y
```

验证（需 CUDA + torch + triton）：`python3 books/examples/appendix-a/ch13_saxpy.py`

### 作业 2：BLOCK_SIZE 影响

- **64**：grid 更大，并行度更高，单 program 工作量小
- **4096**：grid 更小，单 program 寄存器/共享内存压力更大，可能降低 occupancy
- 最优值通常 512–2048，取决于硬件与内核

### 作业 3：core.py

1. `@builtin` — 标记 Triton 语言内置函数，参与 JIT 编译
2. `tensor` — 块级数据容器，含 dtype、shape 等属性
3. `constexpr` — 编译期常量，参与 constexpr 折叠；普通 Python 值在运行时求值

---

## 第 14 章：Triton 编译流水线

### 作业 2：compile() 流程

1. `make_backend(target)` → 选择 CUDABackend
2. `src.make_ir()` → 生成 TT IR
3. `backend.add_stages()` 注册 ttir → ttgir → llir → ptx → cubin
4. 依次执行各 stage
5. 返回 `CompiledKernel`

### 作业 3：add_stages

NVIDIA 后端注册 5 个 stage：ttir、ttgir、llir、ptx、cubin。Gluon 前端跳过 ttir，从 ttgir 开始。

---

## 第 15 章：Triton IR (TT)

### 作业 1

用 `TRITON_KERNEL_DUMP=1` 导出 SAXPY 内核的 `.ttir` / `.ttgir`，观察：

1. `tt.load` 的操作数：ptr、mask、other 及 cache/evict/isVolatile 属性
2. `.ttgir` 张量类型带 `#blocked` 等编码
3. `.ttgir` 中新增的 `ttg.convert_layout`

---

## 第 16 章：TritonGPU IR 与数据布局

### 作业 1：理解编码参数

```
sizePerThread = [1, 2]   → 每线程 2 个元素
threadsPerWarp = [16, 2] → 每 warp 32 线程
warpsPerCTA = [2, 2]     → 每 CTA 4 个 warp

每 CTA 线程数 = 32 × 4 = 128
每 CTA 元素数 = 2 × 128 = 256
tensor 64×128 = 8192 元素 → 需要 8192 / 256 = 32 个 CTA
```

### 作业 3：设计自定义编码

1. 最大向量宽度：256 bit / 16 bit = **16** 个 f16
2. 总线程数：64 × 8 = **512**
3. 总元素：256 × 512 = **131072**
4. 每线程元素：131072 / 512 = **256**
5. 最优 `sizePerThread`：在内存事务约束下尽量接近 **16**（或与 load 向量化对齐的值）

---

## 第 17 章：Triton → TritonGPU 转换

### 作业 2

1. `GenericOpPattern` — 大多数 Op 只改类型
2. 特化：`ArithConstantPattern`（Constant 的 DenseElementsAttr 需重塑）
3. `arith::ConstantOp` 需特化是因为编码信息必须写入 attribute

---

## 第 18 章：TritonGPU → LLVM 降级

### 作业 2：共享内存分配

在 `Allocation.cpp` 中遍历 `local_store` / `async_copy` 等 Op，按张量大小累加共享内存需求，分配在 `addrspace(3)`。

---

## 第 19 章：NVIDIA 后端案例分析

### 作业 2：分析 PTX

1. `.version 8.x` — PTX ISA 版本
2. `.target sm_XX` — 目标 SM 架构
3. `.reg .f32 %r<N>` — 寄存器声明（数量因内核而异，需在你导出的 PTX 中计数）
4. `.entry kernel_name` — 内核入口符号名

---

## 第 20 章：理解 Triton 后端接口

### 作业 1

阅读 `python/triton/backends/compiler.py` 中 `BaseBackend`：

1. 6 个 `@abstractmethod`：`supports_target`、`hash`、`parse_options`、`add_stages`、`load_dialects`、`get_codegen_implementation`（具体以源码为准）
2. `parse_attr` — 解析 kernel 属性
3. `hash()` 加 `@lru_cache` — 避免重复计算编译缓存键
4. `pack_metadata` — 非默认实现打包运行时元数据

### 作业 2：实现最小后端

参考 `books/examples/chapter20/minimal_backend/` 与 `test_registration.py`：

```bash
cd books/examples/chapter20 && ./run_examples.sh
# OK: minimal backend registered
```

### 作业 3：选择策略

结合第 7 章作业 2 所选硬件，说明策略 A/B/C 的选择理由，并画出 ttir → … → 二进制的流水线。

---

## 第 21 章：设计第三方后端

### 作业 2：决策分析

对「已有 LLVM 后端但优化不完善」的 GPU，建议**策略 A**（走 LLVM 后端），复用 TT/TTG 框架，重点优化降级 Pass。

---

## 第 22 章：实现 Dialect 与转换

### 作业 2 提示

TTG → 自定义 Dialect 转换步骤：

1. 定义 `ConversionTarget`
2. 定义 `TypeConverter`
3. 为各 Op 实现 `ConversionPattern`
4. `applyPartialConversion` 执行

---

## 第 23 章：代码生成与测试

### 作业 1 提示

Barrier 降级的 LIT 测试框架（示意）：

```mlir
// RUN: triton-opt %s --convert-mygpu-to-llvm | FileCheck %s
// CHECK: llvm.fence
```

---

## 第 24 章：集成与调试

### 作业 2：性能分析

核心指标：

1. **吞吐量**（FLOPs/s）
2. **带宽利用率** = 实际带宽 / 理论峰值带宽
3. **占用率** = 活跃 warp / 最大 warp
4. 瓶颈判断：计算 bound vs 内存 bound

---

## 验证脚本

```bash
cd books/examples/appendix-a && ./run_verify.sh
```

当前脚本验证：ch02 ADT、ch03 SSA IR、ch04 max/gep、ch05 CountAddPass（需先构建 chapter05）、ch08 MLIR、ch13 SAXPY（无 GPU 时 SKIP）。
