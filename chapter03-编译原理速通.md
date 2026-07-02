# 第 3 章：编译原理速通

> **本章目标**：用最快的速度掌握阅读 LLVM/Triton 源码所需的编译原理知识。
> 这不是完整的编译原理课程，而是**精准瞄准**编译器开发场景的速成。

> 驯龙手记：编译原理就是"龙的生物学"——了解 SSA 是龙的骨架结构、
> CFG 是龙的神经回路、数据流分析是龙的血液循环。
> 不学生物学也能驯龙，但懂了就能驯得更好。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter03/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `example_before_ssa.c` | 作业1 | SSA 转换前 C 代码 |
| `example_ssa.ll` | 作业1 | SSA 形式 LLVM IR |
| `codegen_visitor.py` | 3.x | 简化 AST 代码生成 |

运行：

```bash
cd books/examples/chapter03
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

> 概念章节；示例用于理解 SSA 与代码生成。

---
## 3.1 编译器三段式架构

```
源代码 → 前端（Frontend）→ 中端（Optimizer）→ 后端（Backend）→ 目标代码
```

### 每个阶段做什么

| 阶段 | 输入 | 输出 | 任务 | Triton 对应 |
|------|------|------|------|-------------|
| **前端** | 源代码 | IR | 词法分析、语法分析、语义分析 | `code_generator.py` → TT IR |
| **中端** | IR | IR（优化后） | 各种优化 Pass | `make_ttir()`, `make_ttgir()` |
| **后端** | IR | 目标代码 | 指令选择、寄存器分配 | `make_llir()` → PTX → CUBIN |

### 一个加法的旅程

```c
// C 源代码
int add(int a, int b) { return a + b; }
```

```
↓ 前端（Clang）: 词法分析 → 语法分析 → AST → LLVM IR 生成

↓ LLVM IR
define i32 @add(i32 %a, i32 %b) {
  %result = add i32 %a, %b
  ret i32 %result
}

↓ 中端（opt -O2）: 内联、常量传播、死代码消除...

↓ 后端（llc）: 指令选择、寄存器分配、指令调度

↓ x86 汇编
addl:
    movl    %edi, %eax
    addl    %esi, %eax
    retq
```

## 3.2 中间表示（IR）的重要性

### 为什么需要 IR？

没有 IR 的编译器（直接 AST → 机器码）：

```
C AST → x86
C AST → ARM
C++ AST → x86
C++ AST → ARM
= 需要 4 个独立的编译器实现
```

有 IR 的编译器（三段式）：

```
C AST → LLVM IR ─┐
C++ AST → LLVM IR ─┤
                   ▼
           LLVM IR 优化器
                   ▼
           LLVM IR → x86
           LLVM IR → ARM
= 只需要 2 个前端 + 2 个后端 = 4 个组件，可复用
```

**🔑 关键点**：IR 是前后端的共享接口。N 种语言 × M 种目标 = N+M 个组件，而不是 N×M。

### IR 的设计考量

| 特性 | 说明 |
|------|------|
| **SSA 形式** | 每个变量只赋值一次 |
| **无限虚拟寄存器** | 编译时不需要关心物理寄存器 |
| **显式控制流** | 基本块 + 分支指令 |
| **类型系统** | 强类型，支持类型推导 |

## 3.3 SSA 形式 — 静态单赋值

SSA (Static Single Assignment) 是现代编译器的核心概念。

### 什么是 SSA

```llvm
; 非 SSA 形式（x 被赋值两次）
x = 1
x = x + 1

; SSA 形式（每个 x 只赋值一次）
x1 = 1
x2 = x1 + 1
```

### SSA 的好处

1. **简化数据流分析**：每个变量的使用点可以追溯到唯一的定义点
2. **使优化更简单**：常量传播、死代码消除等 Pass 变得直接
3. **天然的 use-def 链**：编译器不需要额外维护

### Phi 指令 — 处理控制流合并

```llvm
; 控制流分支后需要合并
if (condition)
    x = 1
else
    x = 2
; 这里 x 的值来自哪里？
```

```llvm
; SSA 中的解决方案：phi 指令
entry:
    br i1 %cond, label %then, label %else
then:
    %x1 = add i32 0, 1
    br label %merge
else:
    %x2 = add i32 0, 2
    br label %merge
merge:
    %x = phi i32 [%x1, %then], [%x2, %else]
    ;         ↑从 then 块来则取 x1，从 else 块来则取 x2
```

## 3.4 基本块与控制流

```
      entry   ← 函数入口
    %a = add ; %b = mul ; br %cond
       |
   ----+----
   |        |
  then     else        ← 条件分支
  %c=x     %d=y
   |        |
   ----+----
       |
     merge              ← 控制流合并
    %e = phi ; ret %e
```

- **基本块（BasicBlock）**：一组顺序执行的指令，只有最后一个指令是分支
- **控制流图（CFG）**：基本块为节点，分支为边的有向图

## 3.5 数据流分析

数据流分析是编译器优化的基础。两种主要类型：

### 前向分析（Forward）

```
从程序入口到出口传播信息

例子：可达定义分析（Reaching Definition Analysis）
    某个变量 v 在程序点 p 上是否可见（可达）？
用途：常量传播、类型推断
```

### 后向分析（Backward）

```
从程序出口到入口传播信息

例子：活跃变量分析（Live Variable Analysis）
    变量 v 在程序点 p 之后是否还会被使用？
用途：寄存器分配、死代码消除
```

### Triton 中的分析 Pass

Triton 也有自己的分析 Pass：

| Pass | 分析内容 | 用途 |
|------|---------|------|
| `AxisInfoAnalysis` | 每个值的轴信息（位移、步长） | `CoalescePass` 推断内存访问模式 |
| `AliasAnalysis` | 指针别名分析 | 共享内存优化 |
| `AllocationAnalysis` | 共享内存分配分析 | `AllocateSharedMemoryPass` |

## 3.6 优化 Pass 分类

### 从功能分类

| 类别 | 例子 | 作用 |
|------|------|------|
| **简化** | `instcombine` | 简化指令组合 |
| **常量传播** | `sccp` | 用常量替换变量 |
| **死代码消除** | `dce`, `adce` | 删除无用的代码 |
| **循环优化** | `licm`, `loop-unroll` | 循环不变代码外提、循环展开 |
| **内联** | `inline` | 用函数体替换函数调用 |
| **向量化** | `loop-vectorize` | 标量指令转为 SIMD 指令 |
| **内存优化** | `mem2reg` | 将 stack 变量提升为 SSA 值 |

### 从处理范围分类

```
ModulePass     -- 分析整个编译单元（跨函数）
CallGraphPass -- 分析调用图
FunctionPass  -- 分析单个函数（最常用）
LoopPass      -- 分析单个循环
```

## 3.7 Triton 中的编译器概念

```
Triton 编译过程：

Python AST  --[code_generator]--> TT IR  --[make_ttir]--> 优化后 TT
                                        |
                                   TT → TTG 转换（最关键一步）
                                        |
                                   TTG IR  --[优化 Passes]--> ...
                                        |
                                   TritonGPU → LLVM 降级
                                        |
                                   LLVM IR → LLVM 优化 → PTX → CUBIN
```

## 3.8 GPU 编译器的特殊考量

### 与 CPU 编译器的区别

| 方面 | CPU 编译器 | GPU 编译器（Triton） |
|------|-----------|-------------------|
| 并行模型 | 指令级并行 | 线程级并行（SIMT） |
| 内存层级 | 缓存 → RAM | 寄存器 → Shared → Global |
| 控制流 | 分支预测 | 分支发散（warp 内） |
| 主要优化 | 指令级/循环 | 内存合并/线程映射 |
| 后端输出 | x86/ARM 汇编 | PTX/GCN 汇编 |

### GPU 特有的优化问题

**内存合并（Coalescing）**

```
不合并的访问（性能差）：
Thread 0: addr 0   Thread 1: addr 100  Thread 2: addr 200
                    ↓
需要 3 次内存事务

合并后的访问（性能好）：
Thread 0: addr 0   Thread 1: addr 4    Thread 2: addr 8
                    ↓
1 次内存事务即可
```

**Warp 分歧（Divergence）**

```
warp 内的线程走不同分支：
if (threadId % 2 == 0) {
    // 线程 0, 2, 4, ... 执行这里
} else {
    // 线程 1, 3, 5, ... 执行这里（等待前面的线程完成）
}
// 所有线程串行执行两个分支——性能损失
```

## 3.9 抽象语法树（AST）— 前端的基础

```python
# Python 代码
x = a + b * 2
```

```text
AST 表示：
        =
      /   \
     x     +
          / \
         a   *
            / \
           b   2
```

在 Triton 中，`code_generator.py` 遍历 Python AST，为每个节点生成 MLIR：

```python
# code_generator.py（简化）
class CodeGenerator(ast.NodeVisitor):
    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return self.builder.create("arith.addf", lhs, rhs)
```

---

## 📝 课后作业

### 作业 1：SSA 转换

将以下代码转换为 SSA 形式：

```c
int example(int x, int y) {
    int z = x + y;
    if (z > 0) {
        z = z * 2;
    } else {
        z = z * 3;
    }
    return z;
}
```

### 作业 2：画 CFG

对于上面的 `example` 函数，画出控制流图（基本块为节点，分支为边）。

### 作业 3：Triton 中的概念对应

| 编译原理概念 | Triton 中的对应 |
|-------------|----------------|
| 词法分析 | → |
| 语法分析 | → |
| IR | → |
| 中间表示优化 | → |
| 指令选择 | → |
| 寄存器分配 | → |

填写以上表格。

---

## 本章小结

- 编译器三段式架构：前端 → 中端（优化）→ 后端
- SSA 形式：每个变量只赋值一次，phi 指令处理控制流合并
- 基本块 + CFG 图构成函数内控制流
- 数据流分析：前向分析（如常量传播）和后向分析（如活跃变量）
- GPU 编译器特有挑战：内存合并、warp 分歧
- Triton 有自己的编译流水线，最终降级到标准 LLVM IR
