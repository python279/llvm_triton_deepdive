# 第 15 章：Triton IR (TT Dialect)

> **本章目标**：理解 TT Dialect 的类型系统、核心操作和 TableGen 定义。
> 本章是第 16-18 章（TTG、转换、降级）的基础。

> 驯龙手记：TT Dialect 是 Triton 的"龙骨"——最基础的骨架。
> 它定义了所有操作的结构，但还没有加入 GPU 的血肉（数据布局）。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter15/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `example.ttir` | 15.5 | 完整 TTIR 示例 |

运行：

```bash
cd books/examples/chapter15
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 15.1 TT Dialect 的定位

```
Python AST（块级操作）
    --> [code_generator.py]
    --> TT IR（设备无关的块级表示） [本章]
    --> [make_ttir() + passes]
    --> TTG IR（GPU 数据布局） [第16章]
    --> [TT→TTG转换] [第17章]
    --> LLVM IR [第18章]
```

TT Dialect 是 Triton 编译流程中的**第一级 MLIR IR**，也是**设备无关**的层。这里没有线程映射、没有 warp 概念、没有数据布局——只有块级的操作。

## 15.2 TT 类型系统

定义在 `include/triton/Dialect/Triton/IR/TritonTypes.td`。

```tablegen
// 标量类型
def TT_Float : AnyTypeOf<[F8E4M3FN, F8E4M3FNUZ, F8E5M2, F8E5M2FNUZ,
                           F16, BF16, F32, F64], "floating-point">;
def TT_Int   : AnyTypeOf<[I1, I4, I8, I16, I32, I64], "integer">;

// 指针类型（带 pointeeType 和 addressSpace）
def TT_PtrType : TritonTypeDef<"Pointer", "ptr"> {
    let parameters = (ins "Type":$pointeeType, "int":$addressSpace);
}

// 组合类型别名
def TT_Tensor : RankedTensorOf<[TT_Float, TT_Int, TT_Ptr]>;
def TT_FloatLike : AnyTypeOf<[TT_Float, TT_FloatTensor]>;
def TT_PtrLike : AnyTypeOf<[TT_Ptr, TT_PtrTensor]>;
def TT_Type : AnyTypeOf<[TT_FloatLike, TT_IntLike, TT_PtrLike]>;
```

### 关键类型属性

| 类型 | 例子 | 说明 |
|------|------|------|
| `!tt.ptr<f32>` | 32 位浮点指针 | `addressSpace=1`(全局) 或 `3`(共享) |
| `tensor<1024xf32>` | 1024 元素张量 | 没有编码——设备无关 |
| `tensor<128x256xf32>` | 二维张量 | 行主序 |

## 15.3 核心 Op 分类

所有 TT Op 定义在 `include/triton/Dialect/Triton/IR/TritonOps.td`（约 1379 行）。

```tablegen
class TT_Op<string mnemonic, list<Trait> traits = []> :
    Op<Triton_Dialect, mnemonic,
       !listconcat(traits, [TensorSizeTrait, VerifyTensorLayoutsTrait])> {
}
```

### 内存操作

| Op | 说明 | 关键属性 |
|----|------|---------|
| `tt.load` | 带掩码的内存加载 | `cache`, `evict`, `isVolatile` |
| `tt.store` | 带掩码的内存存储 | `cache`, `evict` |
| `tt.atomic_add` | 原子加法 | — |

`tt.load` 的定义：

```tablegen
def TT_LoadOp : TT_Op<"load", [..., Pure]> {
    let arguments = (ins
        TT_PtrLike:$ptr,             // 指针（标量或张量）
        Optional<TT_BoolLike>:$mask, // 边界掩码
        Optional<TT_Type>:$other,    // fallback 值
        TT_CacheModifierAttr:$cache, // 缓存策略
        TT_EvictionPolicyAttr:$evict, // 驱逐策略
        BoolAttr:$isVolatile         // 是否 volatile
    );
    let results = (outs TT_Type:$result);
}
```

### 算术与数学

Triton 重用了 `arith` 和 `math` Dialect 的标准操作，并定义了特有的：

```mlir
%sum = arith.addf %a, %b                 ; 加法（标准 arith）
%prod = tt.dot %a, %b, %c                ; 矩阵乘法（Triton 特有）
%sqrt = tt.precise_sqrt %x               ; 高精度开方
```

`tt.dot` 的定义：

```tablegen
def TT_DotOp : TT_Op<"dot", [Pure, ...]> {
    let arguments = (ins
        TT_FpIntTensor:$a,       // 左矩阵
        TT_FpIntTensor:$b,       // 右矩阵
        TT_FpIntTensor:$c,       // 累加器
        TT_InputPrecisionAttr:$inputPrecision,  // tf32/ieee/...
        I32Attr:$maxNumImpreciseAcc
    );
    let results = (outs TT_FpIntTensor:$d);  // d = a*b + c
}
```

### 归约与扫描

```tablegen
def TT_ReduceOp: TT_Op<"reduce", [Pure, SingleBlock, ...]> {
    let arguments = (ins Variadic<TT_Tensor>:$srcs, I32Attr:$axis);
    let results = (outs Variadic<TT_Type>:$result);
    let regions = (region SizedRegion<1>:$combineOp);
    // combineOp 是一个 Block，接收两个标量参数，返回一个标量
}

def TT_ScanOp: TT_Op<"scan", [Pure, SingleBlock, ...]> {
    let arguments = (ins Variadic<TT_Tensor>:$srcs, I32Attr:$axis, BoolAttr:$reverse);
    // 类似 reduce，但给出的是前缀扫描结果
}
```

### 控制流与张量操作

```mlir
%pid = tt.get_program_id {axis = 0 : i32} : i32
%range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32>
%ptr = tt.addptr %base, %offset : !tt.ptr<f32>, i32
%splat = tt.splat %scalar : (f32) -> tensor<1024xf32>
```

## 15.4 从 Python 到 TT IR 的映射

| Python | TT IR | 说明 |
|--------|-------|------|
| `tl.load(ptr, mask)` | `tt.load %ptr, %mask` | 加载 |
| `tl.store(ptr, val, mask)` | `tt.store %ptr, %val, %mask` | 存储 |
| `tl.dot(a, b, c)` | `tt.dot %a, %b, %c` | 矩阵乘 |
| `tl.reduce(x, 0, fn)` | `tt.reduce %x {axis=0}` | 归约 |
| `tl.program_id(0)` | `tt.get_program_id {axis=0}` | 块 ID |
| `tl.arange(0, N)` | `tt.make_range {start=0, end=N}` | 索引 |
| `a + b` | `arith.addf %a, %b` | 加法 |

## 15.5 一个完整的 TTIR 示例

运行一个简单内核并导出 TTIR：

```python
@triton.jit
def example_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)
```

生成的 TTIR：

```mlir
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @example_kernel(
      %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %c1024_i32 = arith.constant 1024 : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : (i32) -> tensor<1024xi32>
    %offsets = arith.addi %3, %2 : tensor<1024xi32>
    %4 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<1024x!tt.ptr<f32>>
    %5 = tt.addptr %4, %offsets : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %n_val = tt.splat %n : (i32) -> tensor<1024xi32>
    %mask = arith.cmpi slt, %offsets, %n_val : tensor<1024xi32>
    %x = tt.load %5, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32>
    %y = tt.load %9, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1024xf32>
    %sum = arith.addf %x, %y : tensor<1024xf32>
    tt.store %13, %sum, %mask {cache = 1 : i32, evict = 1 : i32} : tensor<1024xf32>
    tt.return
  }
}
```

> 💡 注意：TTIR 中的张量类型是 `tensor<1024xf32>`，**没有编码/布局属性**。布局将在 TTG 阶段添加。

## 15.6 TTIR 的优化 Pass

`make_ttir()` 在 NVIDIA 后端中运行的 Pass：

```python
def make_ttir(mod, metadata, opt, capability):
    pm = ir.pass_manager(mod.context)
    passes.common.add_inliner(pm)                               # 函数内联
    if capability // 10 < 9:
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm) # 描述符→指针
    passes.common.add_canonicalizer(pm)                         # 规范化
    passes.ttir.add_combine(pm)                                 # 组合优化
    passes.ttir.add_reorder_broadcast(pm)                       # 广播重排
    passes.common.add_cse(pm)                                   # 公共子表达式消除
    passes.common.add_symbol_dce(pm)                            # 符号死代码消除
    passes.ttir.add_loop_unroll(pm)                             # 循环展开
    pm.run(mod, 'make_ttir')
    return mod
```

这些 Pass 大部分是**设备无关**的——它们在任何 GPU 后端上都能运行。

## 15.7 补充：Gluon — Triton 的另一种前端

除了 `@triton.jit` 前端外，Triton 还有一个更新的前端叫 **Gluon**。它的核心思想是**语言无关的算子描述**——通过一组标准化的操作接口来描述计算，而不是绑定到 Python 语法。

```python
# Gluon 示例（python/triton/experimental/gluon/）
# 它提供了类似 Python 语义但更接近 IR 的编程接口
import triton.experimental.gluon as gluon

# Gluon 有自己的 Dialect（gluon Dialect）
# 定义在 include/triton/Dialect/Gluon/IR/GluonOps.td
# 它允许更自由的前端和后端解耦
```

Gluon 有自己的 IR、自己的编译路径（`gluon_to_ttgir()`—跳过 `ttir` 阶段），最终汇入 TTGIR 进行后续优化和代码生成。作为一个初学者，先掌握标准 `@triton.jit` 路径即可，了解 Gluon 存在即可。


---

## 📝 课后作业

### 作业 1：写一个内核并导出 TTIR

写一个 `y = a * x + y`（SAXPY）的 Triton 内核，用 `TRITON_KERNEL_DUMP=1` 导出：
1. 找到 `.ttir` 文件，用 `triton-opt` 加载
2. 找到 `tt.load` 和 `tt.store` 操作
3. 找到 `arith.mulf` 和 `arith.addf` 操作

### 作业 2：理解 TT_LoadOp

阅读 `include/triton/Dialect/Triton/IR/TritonOps.td` 中 `TT_LoadOp` 的完整定义，回答：
1. `arguments` 中包含哪些输入？
2. `assemblyFormat` 是如何定义的？
3. 它有哪些 trait？

### 作业 3：手动运行 ttir Pass

```bash
# 使用导出的 .ttir 文件
triton-opt example.ttir --inline --canonicalize -o output.ttir
# 对比前后的 IR 变化
```

---

## 本章小结

- TT Dialect 是设备无关的块级 IR，是 Triton 编译流程的第一层
- 类型系统：指针（带地址空间）、张量（无编码）、标量
- 核心 Op：`tt.load/store`（内存）、`tt.dot`（矩阵乘）、`tt.reduce/scan`（归约）
- 算术操作复用 MLIR 标准 Dialect（`arith`、`math`）
- TTIR 经过优化 Pass（内联、CSE、规范化、循环展开）后进入 TTG 阶段
- TTIR 的关键特征：**张量类型没有编码属性**——这是 TT 和 TTG 的核心区别
