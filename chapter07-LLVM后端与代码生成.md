# 第 7 章：LLVM 后端与代码生成

> **本章目标**：理解 LLVM 后端如何从 IR 生成机器码。
>
> 驯龙手记：后端驯服是驯龙的最后一步——让龙真正"落地行动"。
> 指令选择是告诉龙用哪块肌肉，寄存器分配是协调各个肢体的协调配合，
> 代码发射是龙真正奔跑起来的那一刻。本章是第 23-24 章（自定义 Triton 后端）的重要基础。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter07/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `add.ll` | 7.x | llc 生成汇编 smoke test |

运行：

```bash
cd books/examples/chapter07
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

> 完整后端需 LLVM 源码树；示例用 `llc` 演示 IR→汇编。

---
## 7.1 LLVM 后端流水线

从 LLVM IR 到机器码的旅程：

```
LLVM后端流水线:

LLVM IR --> 指令选择(ISel: SelectionDAG/GlobalISel)
         --> 指令调度(Pre-RA Scheduling)
         --> 寄存器分配(Register Allocation: 虚拟→物理寄存器)
         --> 指令调度(Post-RA Scheduling)
         --> 代码发射(Code Emission: .s 汇编 / .o 目标文件)
```

## 7.2 指令选择 — SelectionDAG

指令选择是后端最关键的一步：将 LLVM IR 指令映射为目标架构的具体指令。

### 从 LLVM IR 到 DAG

```llvm
; LLVM IR
%sum = add i32 %a, %b
```

```text
SelectionDAG（有向无环图）：
    t0: ch = EntryNode
    t1: i32, t2: i32 = Constant<0>, Constant<1>  ; 常量
    t3: i32 = add t1, t2        ; 逻辑 add
    t4: ch = Ret t3              ; 返回

每个节点是目标指令或虚拟操作
```

### 指令选择模式

```tablegen
// 用 TableGen 描述指令选择规则
// LLVM 的 .td 文件中：

def : Pat<
    (add i32:$a, i32:$b),     // LLVM IR 模式
    (ADD32rr $a, $b)          // 目标指令
>;

// 这表示：LLVM IR 的 add → x86 的 ADD32rr（寄存器-寄存器加法）
```

### Triton 的特殊性

Triton 不直接经过 LLVM 的指令选择——它降级到**PTX 文本**（GPU 汇编），然后通过 `ptxas` 生成二进制：

```
TTG IR → LLVM IR (via MLIR) → PTX (via llvm-translate) → CUBIN (via ptxas)
                              ↑ 不经过 LLVM 后端     ↑ NVIDIA 专有汇编器
```

所以第 8-12 章（MLIR → LLVM IR）比 LLVM 后端对 Triton 开发更重要。

## 7.3 寄存器分配

寄存器分配将虚拟寄存器映射到物理寄存器。

### 为什么需要寄存器分配

```llvm
; 函数有大量 SSA 值
define i32 @large(i32 %a, i32 %b) {
    %1 = add i32 %a, %b
    %2 = mul i32 %1, %a
    %3 = sub i32 %2, %b
    ; ... 100 个中间值 ...
    %100 = add i32 %99, %1
    ret i32 %100
}
```

物理寄存器有限（如 x86-64 有 16 个通用寄存器）。寄存器分配决定：

1. 哪些值放在寄存器中（速度快）
2. 哪些值溢出到栈上（速度慢，但空间大）
3. 如何分配物理寄存器编号

### 图着色算法

```text
每个 SSA 值 = 图中的节点
两个值如果同时活跃 = 边相连
寄存器 = 不同的颜色
寄存器分配 = 用 K 种颜色给图着色

        %a ─── %b
        │       │
        │       │
        └── %c ──┘

需要 2 种颜色（因为 %a 和 %b 同时活跃）
如果只有 1 个寄存器可用 → %c 溢出到栈
```

## 7.4 目标描述文件（.td）

LLVM 后端最重要的部分是目标描述文件，用 TableGen 编写：

```tablegen
// 以 x86 为例（简化）
// X86InstrInfo.td

// 1. 定义寄存器
def RAX : X86Reg<"rax", 0>;
def RBX : X86Reg<"rbx", 3>;
// ...

// 2. 定义指令格式
class I<string mnemonic, dag outs, dag ins> {
    string Mnemonic = mnemonic;
    dag OutOperandList = outs;
    dag InOperandList = ins;
}

// 3. 定义具体指令
def ADD32rr : I<"addl", (outs GR32:$dst), (ins GR32:$src1, GR32:$src2)> {
    let Pattern = [(set GR32:$dst, (add GR32:$src1, GR32:$src2))];
}

// 4. 指令选择模式
def : Pat<(add i32:$a, i32:$b), (ADD32rr $a, $b)>;
```

## 7.5 实现一个最小后端示例

这里演示一个虚拟 CPU 的简化后端（仅用于理解概念）：

```tablegen
// MyCPU.td — 目标描述文件

// 1. 定义寄存器
def R0 : Register<"r0">;
def R1 : Register<"r1">;
def GPR : RegisterClass<"MyCPU", [i32], 0, (add R0, R1)>;

// 2. 定义指令
def ADD : Instruction {
    let Size = 4;
    let OutOperandList = (outs GPR:$dst);
    let InOperandList = (ins GPR:$src1, GPR:$src2);
    let Pattern = [(set GPR:$dst, (add GPR:$src1, GPR:$src2))];
    let AsmString = "add $dst, $src1, $src2";
}
```

```cpp
// MyCpuISelDAGToDAG.cpp — 指令选择器
class MyCpuDAGToDAGISel : public SelectionDAGISel {
    void Select(SDNode *N) override {
        switch (N->getOpcode()) {
        case ISD::ADD: {
            // 将 LLVM IR 的 ADD → MyCPU 的 ADD 指令
            SDNode *addNode = CurDAG->getMachineNode(
                MyCPU::ADD, dl,
                N->getValueType(0),
                N->getOperand(0), N->getOperand(1));
            ReplaceNode(N, addNode);
            return;
        }
        // ... 其他指令
        }
    }
};
```

## 7.6 Triton 的自定义后端路径

理解 LLVM 后端后，我们来看看 Triton 的自定义后端路径：

```
┌──────────────────────────────────────────────────┐
│  完整 LLVM 后端路径                                │
│  LLVM IR → SelectionDAG → 寄存器分配 → 汇编       │
│  → 对新 GPU 来说极其复杂，通常不推荐                 │
└──────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────┐
│  Triton 后端的推荐路径                              │
│  TT IR → TTG IR → 自定义降级 → [目标汇编/C++]    │
│  → 可以跳过 LLVM 后端，直接生成目标代码            │
└──────────────────────────────────────────────────┘
```

**实现 Triton 第三⽅后端的三种策略**：

| 策略 | 难度 | 性能 | 说明 |
|------|------|------|------|
| 生成 LLVM IR，用 LLVM 后端 | 中 | 中 | 如果目标已有 LLVM 后端 |
| 生成目标汇编（如 PTX） | 高 | 高 | 需要写完整的代码生成器 |
| 生成 C++ 模板代码 | 低 | 低 | 快速原型，性能差 |

> 第 20-24 章会深入这些选项。

## 7.7 PTX 后端 — NVIDIA 的特例

Triton 的 NVIDIA 后端代表了一种特殊路径：生成 PTX（文本汇编），调用 `ptxas` 汇编为二进制。

```mermaid
graph LR
    A[TTG IR] --> B[MLIR LLVM Dialect]
    B --> C[LLVM Native IR]
    C --> D[PTX Text]
    D --> E[ptxas]
    E --> F[CUBIN]
    
    style A fill:#bbf
    style D fill:#fbb
    style F fill:#bfb
```

**为什么 NVIDIA 可以走这条路径？**

因为 LLVM 的 NVPTX 后端已经成熟——`llvm-translate-to-asm` 可以将 LLVM IR 转为 PTX 文本。其他 GPU 如果没有 LLVM 后端，就不能直接复用这条路。

---

## 📝 课后作业

### 作业 1：概念填空

| 概念 | 说明 |
|------|------|
| 指令选择 | → |
| 寄存器分配 | → |
| 指令调度 | → |
| 代码发射 | → |

### 作业 2：选择 Triton 的后端策略

假设你需要在以下两种硬件上实现 Triton 后端，各选择哪种策略？说明理由。

1. **一个已有 LLVM 后端的新 GPU**
2. **一个只有 C 编译器的专用 AI 加速器**

### 作业 3：阅读 LLVM 目标描述文件

下载 LLVM 源码，找到 `llvm/lib/Target/X86/X86InstrInfo.td`，回答：
1. `ADD32rr` 是什么格式的指令？
2. `ADD32ri` 和 `ADD32rr` 有什么区别？

---

## 本章小结

- LLVM 后端将 IR 转为机器码：指令选择 → 调度 → 寄存器分配 → 发射
- 目标描述文件（.td）是所有后端信息的中心
- 寄存器分配用图着色算法，将虚拟寄存器映射到物理寄存器
- Triton 可通过 LLVM 后端生成 PTX，或直接生成目标汇编
- 自定义 Triton 后端的难度取决于是否复用 LLVM 后端
- 第 20-24 章将深入 Triton 后端的实现细节
