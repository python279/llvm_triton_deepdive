# 第 4 章：LLVM IR 详解

> **本章目标**：能读懂和编写 LLVM IR，理解 SSA 形式和控制流。
>
> 📂 **第二部分：LLVM 深入** — 触摸龙骨，理解 LLVM IR 的骨骼结构

> 驯龙手记：LLVM IR 是"龙之骨骼"——它定义了编译器数据流动的骨架。
> SSA 形式就是每根骨头的唯一编号，基本块就是关节，
> 而 phi 指令是连接关节的韧带。摸清骨骼结构，你就能预测龙的行动。

## 配套示例

本章可运行代码位于 `books/examples/chapter04/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `factorial.ll` | 4.4 | 递归阶乘 |
| `sum_array.ll` | 4.4 | 数组求和 |
| `example.c` | 4.5 | Clang 生成 IR |
| `gep_demo.ll` | 4.3/作业3 | GEP 演示 |
| `attributes_demo.ll` | 4.6 | 函数属性 |
| `max.ll` | 作业1 | max 函数 |
| `matmul.c` | 作业2 | 矩阵乘法 IR |

运行：

```bash
cd books/examples/chapter04
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 4.1 LLVM IR 概述

LLVM IR 有三种等价的形式：

| 形式 | 扩展名 | 特点 |
|------|--------|------|
| 可读文本 | `.ll` | 人类可调试 |
| 二进制位码 | `.bc` | 紧凑、加载快 |
| 内存表示 | — | C++ API 操作 |

三者完全等价，可以无损互转：

```bash
# 文本 → 二进制
llvm-as factorial.ll -o factorial.bc

# 二进制 → 文本
llvm-dis factorial.bc -o factorial.ll

# 解释执行（需要 IR 中有 @main）
lli factorial.bc    # factorial(5) → 退出码 120
```

## 4.2 LLVM IR 的类型系统

### 基础类型

```llvm
; 整数类型
i1          ; 1 位整数（布尔）
i8          ; 8 位整数
i32         ; 32 位整数
i64         ; 64 位整数

; 浮点类型
half        ; 16 位浮点（FP16）
float       ; 32 位浮点
double      ; 64 位浮点
fp128       ; 128 位浮点

; void
void        ; 无类型（函数无返回值）
```

### 复合类型

```llvm
; 指针类型
i32*        ; 指向 i32 的指针
float*      ; 指向 float 的指针

; 数组类型
[4 x i32]   ; 4 个 i32 的数组
[2 x [3 x float]]  ; 二维数组

; 结构体类型
{ i32, float }     ; 匿名结构体
%MyStruct = type { i32, i32, i32 }  ; 命名结构体

; 向量类型（SIMD/GPU 重要）
<4 x i32>   ; 4 个 i32 的向量
<2 x float> ; 2 个 float 的向量
```

## 4.3 LLVM 指令集入门

### 算术指令

```llvm
%r = add i32 %a, %b          ; r = a + b
%r = sub i32 %a, %b          ; r = a - b
%r = mul i32 %a, %b          ; r = a * b
%r = udiv i32 %a, %b         ; r = a / b（无符号）
%r = sdiv i32 %a, %b         ; r = a / b（有符号）
%r = fadd float %a, %b       ; r = a + b（浮点）
%r = fsub float %a, %b
%r = fmul float %a, %b
```

### 内存指令

```llvm
; 加载：从内存地址加载值
%val = load i32, i32* %ptr

; 存储：将值写入内存地址
store i32 %val, i32* %ptr

; 指针算术（GEP: GetElementPtr）
%next = getelementptr i32, i32* %base, i64 %index
; 等价于：&base[index]

; 分配栈空间
%arr = alloca [4 x i32], align 4
```

### 控制流指令

```llvm
; 无条件跳转
br label %target

; 条件跳转
br i1 %cond, label %then, label %else

; 返回
ret i32 %result       ; 有返回值
ret void              ; 无返回值

; 选择（类似三元运算符）
%r = select i1 %cond, i32 %a, i32 %b
; 等价于 cond ? a : b

; 函数调用
%r = call i32 @add(i32 %a, i32 %b)

; phi 节点（控制流合并）
%r = phi i32 [%val1, %block1], [%val2, %block2]
```

### GEP 指令详解

GEP (GetElementPtr) 是 LLVM 中最容易混淆的指令之一：

```llvm
; 定义结构体
%MyStruct = type { i32, [4 x i32], float }

; 假设我们有一个 %MyStruct*
%ptr = alloca %MyStruct

; GEP 语法：getelementptr <type>, <type>* <ptr>, <indices...>
; indices 从外层到内层逐级索引

; 获取整个结构体（%ptr 本身）
; 等同于 &ptr[0] → 得到 %MyStruct*

; 获取第一个字段（i32）
%field1 = getelementptr %MyStruct, %MyStruct* %ptr, i64 0, i32 0
;                        └─类型─┘   └──────┘  └基址┘ └索引0┘ └字段0┘
; 等同于：&ptr->field1

; 获取第二个字段的第三个元素
%elem = getelementptr %MyStruct, %MyStruct* %ptr, i64 0, i32 1, i64 2
; 等同于：&ptr->arr[2]
```

## 4.4 完整的 LLVM IR 示例

### 示例 1：简单函数

完整源码：`books/examples/chapter04/factorial.ll`

```llvm
; factorial.ll — 递归计算阶乘

define i32 @factorial(i32 %n) {
entry:
    %cond = icmp sle i32 %n, 1
    br i1 %cond, label %return, label %recurse

recurse:
    %sub = sub i32 %n, 1
    %sub_result = call i32 @factorial(i32 %sub)
    %result = mul i32 %n, %sub_result
    ret i32 %result

return:
    ret i32 1
}

define i32 @main() {
    %r = call i32 @factorial(i32 5)
    ret i32 %r
}
```

运行：

```bash
llvm-as factorial.ll -o factorial.bc
lli factorial.bc    # 退出码 120（5! = 120）
```

### 示例 2：带数组的函数

完整源码：`books/examples/chapter04/sum_array.ll`

```llvm
define i32 @sum_array(i32* %arr, i32 %len) {
entry:
    %result = alloca i32
    store i32 0, i32* %result
    %i = alloca i32
    store i32 0, i32* %i
    br label %loop_cond

loop_cond:
    %i_val = load i32, i32* %i
    %cond = icmp slt i32 %i_val, %len
    br i1 %cond, label %loop_body, label %loop_end

loop_body:
    %elem_ptr = getelementptr i32, i32* %arr, i32 %i_val
    %elem = load i32, i32* %elem_ptr
    %sum = load i32, i32* %result
    %new_sum = add i32 %sum, %elem
    store i32 %new_sum, i32* %result
    %next_i = add i32 %i_val, 1
    store i32 %next_i, i32* %i
    br label %loop_cond

loop_end:
    %final = load i32, i32* %result
    ret i32 %final
}

define i32 @main() {
    %arr = alloca [4 x i32]
    ; ... 初始化 {1,2,3,4} ...
    %sum = call i32 @sum_array(i32* %p0, i32 4)
    ret i32 %sum    ; 退出码 10
}
```

运行：

```bash
llvm-as sum_array.ll -o sum_array.bc
lli sum_array.bc    # 退出码 10
```

> **⚠️ 注意**：上面的代码展示了 LLVM IR 的"低级"用法（手动管理 alloca/load/store）。LLVM 的 `mem2reg` Pass 会自动将这种模式优化为 SSA 形式。

## 4.5 使用 Clang 生成 LLVM IR

最快的 LLVM IR 学习方式：写 C 代码，看生成的 IR。

完整源码：`books/examples/chapter04/example.c`

```c
int add(int a, int b) { return a + b; }

int mul_add(int x, int y, int z) { return x * y + z; }

int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

int main() {
    int arr[4] = {1, 2, 3, 4};
    int sum = 0;
    for (int i = 0; i < 4; i++) {
        sum += arr[i];
    }
    return sum;
}
```

```bash
# 生成带注释的 LLVM IR
clang -S -emit-llvm -O0 example.c -o example_O0.ll
cat example_O0.ll

# 生成优化后的 LLVM IR
clang -S -emit-llvm -O2 example.c -o example_O2.ll
```

> **💡 学习技巧**：写一段 C 代码，编译成 `.ll` 文件，对照 C 源码阅读 IR。这是最有效的 LLVM IR 学习方法。

## 4.6 LLVM 属性的语义描述

函数和参数可以带属性，告诉优化器更多信息：

完整源码：`books/examples/chapter04/attributes_demo.ll`

```llvm
; 函数属性
define i32 @add(i32 %a, i32 %b) nounwind readonly {
    %r = add i32 %a, %b
    ret i32 %r
}

; 参数属性
declare i32 @printf(i8* nocapture, ...) nounwind
```

## 4.7 LLVM IR 与 Triton IR 的对比

| 方面 | LLVM IR | Triton IR (tt) |
|------|---------|----------------|
| 粒度 | 标量/向量 | 块级（张量） |
| SSA | ✅ 全部 | ✅ 全部 |
| 控制流 | `br`/`switch`/`phi` | `scf.for`/`scf.if`/`tt.reduce` |
| 内存 | `load`/`store`/GEP | `tt.load`/`tt.store`/`tt.addptr` |
| 并行 | 无显式支持 | `tt.get_program_id`、块级操作 |
| 类型 | 标量/指针/结构体 | 张量/指针（带地址空间） |
| Dialect 化 | 否 | 是（MLIR Dialog） |

---

## 📝 课后作业

### 作业 1：写 LLVM IR

用 LLVM IR 写一个 `max` 函数：输入两个 i32，返回较大的那个。

参考实现：`books/examples/chapter04/max.ll`

```llvm
define i32 @max(i32 %a, i32 %b) {
entry:
    %cond = icmp sgt i32 %a, %b
    %r = select i1 %cond, i32 %a, i32 %b
    ret i32 %r
}
```

```bash
llvm-as max.ll -o max.bc && lli max.bc    # 退出码 7（max(3,7)）
```

### 作业 2：读 LLVM IR

用 `clang -S -emit-llvm -O0` 编译以下代码，找出矩阵乘法的 IR 模式：

完整源码：`books/examples/chapter04/matmul.c`

```c
#define N 4
void matmul(float A[N][N], float B[N][N], float C[N][N]) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];
}
```

```bash
clang -S -emit-llvm -O0 matmul.c -o matmul_O0.ll
```

### 作业 3：GEP 练习

给定 `%MyStruct = type { i32, [4 x float], i8* }`，写出访问 `arr[3].field2[2]`（假设 `arr` 是 `%MyStruct*`）的 GEP 指令。

参考答案（`field2` = 结构体第 2 个字段，下标 1）：

```llvm
%ptr = getelementptr %MyStruct, %MyStruct* %arr, i64 3, i32 1, i64 2
```

完整演示见 `books/examples/chapter04/gep_demo.ll`。

---

## 本章小结

- LLVM IR 有三级等价表示：文本(.ll) / 二进制(.bc) / 内存(C++ API)
- 类型系统：标量(i32/float/...)、指针(i32*)、数组([4 x i32])、向量(<4 x i32>)
- GEP = GetElementPtr = 指针算术（编译器自动计算偏移量）
- 控制流：基本块 + `br`/`ret` + `phi`
- 所有变量都是 SSA 形式——每个变量只赋值一次
- 最快的学习方法：写 C → 编译成 LLVM IR → 读 IR
