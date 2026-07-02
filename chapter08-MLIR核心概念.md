# 第 8 章：MLIR 核心概念

> **本章目标**：理解 MLIR 的核心抽象——Operation、Value、Type、Attribute、Region，以及它们如何组成 Dialect。
>
> 📂 **第三部分：MLIR 框架** — 掌握龙语，学习 MLIR 的 Dialect 和 Pass

> 驯龙手记：如果说 LLVM 是龙之骨骼，MLIR 就是"龙之语言"——一种可以表达多层级思想的龙语（Dialect）。
> 以前驯龙者只能用一种方式（LLVM IR）和龙交流；现在你可以用龙语、海神语（Triton）等
> 多种语言逐层"翻译"，直到龙完全听懂。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter08/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `build_add.cpp` | 8.10 | OpBuilder 构建 @add |
| `test_scf_if.mlir` | 作业1 | scf.if 示例 |
| `CMakeLists.txt` | 配套 | 链接 MLIR |

运行：

```bash
cd books/examples/chapter08
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 8.1 MLIR 的基本单位：Operation

MLIR 中一切皆为 Operation。Module、Function、指令——都是 Operation。

```mlir
// 在 MLIR 中：
// "dialect.opname"(%operands) {attributes} regions : result_types

%result = arith.addi %a, %b : i32
%loaded = tt.load %ptr, %mask {cache = 1 : i32} : tensor<128xf32>

// 等价的数据结构视图：
Operation {
    name: "arith.addi"
    operands: [%a, %b]
    results: [%result]
    attributes: {}
    regions: []
    successors: []
    location: "test.mlir:3:1"
}
```

### Operation 的结构

```
Operation {
  Name: "arith.addi"
  Operands: [%a: i32, %b: i32]   -- 输入值
  Results:  [%result: i32]        -- 输出值
  Attributes: {}                  -- 编译时已知的属性
  Regions: []                     -- 子图（可嵌套）
  Location: "test.mlir:1:1"      -- 源码位置
}
```

## 8.2 Value — 值的流动

Value 连接生产者（定义它的 Operation）和消费者（使用它的 Operation）。

```cpp
// MLIR 中 Value 的核心 API
Value v = op->getResult(0);           // 获取 op 的第 0 个结果

// use-def 链
Operation *defOp = v.getDefiningOp();  // 谁定义了这个值（nullptr 如果是 Block 参数）
Type type = v.getType();               // 值的类型

// 从使用端看
for (Operation *user : v.getUsers()) {
    // 所有使用这个值的 Operation
}
```

## 8.3 Type — 类型系统

MLIR 类型系统是可扩展的——每个 Dialect 定义自己的类型。

### 内置类型

```mlir
i32, f32, index     ; 标准类型
none, tensor, memref  ; 内置复合类型
```

### 用户自定义类型

```mlir
// Triton 中的自定义类型
!tt.ptr<f32>                     ; 指向 f32 的指针
tensor<128x256xf16, #blocked>    ; 带编码的张量
```

### 类型约束

在 TableGen 中，类型约束用于限制 Op 的输入输出：

```tablegen
// 接受浮点或浮点张量
def TT_FloatLike : AnyTypeOf<[F32, F64, F32Tensor, F64Tensor], "floating-point">;

// 接受任何整数
def TT_IntLike : AnyTypeOf<[I1, I8, I16, I32, I64], "integer">;

// 组合约束
def TT_Type : AnyTypeOf<[TT_FloatLike, TT_IntLike, TT_PtrLike]>;
```

## 8.4 Attribute — 编译时属性

Attribute 是编译时已知的信息，不是运行时值：

```mlir
// 各种 Attribute
{axis = 0 : i32}                    ; IntegerAttr
{cache = 1 : i32, evict = "evict_first"} ; 混合
{order = [1, 0]}                    ; 数组
{encoding = #blocked}               ; Dialect 自定义 Attr
```

### 常用 Attribute 类型

| C++ 类型 | 示例 | 说明 |
|----------|------|------|
| `IntegerAttr` | `42 : i32` | 整型 |
| `FloatAttr` | `3.14 : f32` | 浮点 |
| `StringAttr` | `"hello"` | 字符串 |
| `ArrayAttr` | `[1, 2, 3]` | 数组 |
| `DictionaryAttr` | `{a = 1, b = "x"}` | 字典 |
| Dialect 自定义 | `#blocked<...>` | TTG 的编码属性 |

## 8.5 Region 与 Block

Region 是 MLIR 最强大的特性——允许操作包含嵌套的子图。

```mlir
// 最简单的 Region：函数体
func.func @add(%a: i32, %b: i32) -> i32 {
    // ↑ 这是 func.func op 的一个 Region
    // ↓ Region 包含一个或多个 Block
^bb0(%a: i32, %b: i32):      // ← Block 入口（带参数）
    %sum = arith.addi %a, %b : i32
    func.return %sum : i32
}

// 嵌套 Region：scf.for 循环
%sum = scf.for %i = %0 to %10 step %1 iterators(%c0 -> %sum_iter) {
    // ↑ 这是 scf.for 的 Region
    %new = arith.addi %sum_iter, %i : i32
    scf.yield %new : i32
} : i32
```

### Region 的 C++ API

```cpp
// 获取 Op 的 Region
Region &region = op->getRegion(0);

// 遍历 Region 中的 Block
for (Block &block : region) {
    // Block 参数（入口参数）
    for (Value arg : block.getArguments()) {
        Type argType = arg.getType();
    }
    // Block 中的操作
    for (Operation &op : block) {
        // ...
    }
}
```

## 8.6 Dialect — 方言

Dialect 是 MLIR 的核心扩展机制——一组相关的 Operation、Type、Attribute 的集合。

```mlir
// 通过名称空间区分
arith.addi      // arith Dialect 的 addi op
tt.load         // tt Dialect 的 load op
scf.for         // scf Dialect 的 for op
```

### Dialect 的 C++ 注册

```cpp
// lib/Dialect/Triton/IR/Dialect.cpp
void TritonDialect::initialize() {
    // 1. 注册自定义类型
    registerTypes();

    // 2. 注册自定义属性
    addAttributes<
        #define GET_ATTRDEF_LIST
        #include "triton/Dialect/Triton/IR/TritonAttrDefs.cpp.inc"
    >();

    // 3. 注册操作（由 mlir-tblgen 自动生成）
    addOperations<
        #define GET_OP_LIST
        #include "triton/Dialect/Triton/IR/Ops.cpp.inc"
    >();
}
```

## 8.7 MLIR 内置的常用 Dialect

| Dialect | 名称 | 说明 |
|---------|------|------|
| `func` | 函数 | `func.func`, `func.call`, `func.return` |
| `arith` | 算术 | `arith.addi`, `arith.mulf`, `arith.cmpi` |
| `math` | 数学函数 | `math.exp`, `math.sin`, `math.log` |
| `scf` | 结构化控制流 | `scf.for`, `scf.if`, `scf.yield` |
| `cf` | 非结构化控制流 | `cf.br`, `cf.cond_br` |
| `tensor` | 张量操作 | `tensor.cast`, `tensor.extract` |
| `llvm` | LLVM IR | `llvm.add`, `llvm.load`, `llvm.call` |
| `gpu` | GPU | `gpu.launch`, `gpu.thread_id` |

## 8.8 MLIR 的模块层级结构

```
module @my_module {
    // 最外层：Module（也是一个 Operation）
    
    func.func @main() {
        // 函数（func.func Operation）
        
        %result = scf.for %i = %0 to %10 step %1 {
            // 循环（scf.for Operation，内含 Region）
            
            %sum = arith.addi %i, %c1 : i32
            // ↑ 算术 Operation
            
            %load = tt.load %ptr : tensor<128xf32>
            // ↑ Triton Operation（另一个 Dialect）
            
            scf.yield %sum : i32
        } : i32
    }
}
```

## 8.9 从 LLVM 到 MLIR 的对比

| 概念 | LLVM | MLIR |
|------|------|------|
| 基本单位 | Function | Operation |
| 指令 | Instruction | Operation |
| 基本块 | BasicBlock | Block |
| 类型 | Type | Type（可扩展） |
| 值 | Value | Value |
| 属性 | 少数固定属性 | Attribute（可扩展） |
| 模块 | Module | ModuleOp（也是 Operation） |
| 扩展机制 | 目标描述 .td | Dialect |

## 8.10 实战：在 C++ 中构建 MLIR IR

理解以上概念后，让我们实际编写一小段 C++ 代码，手动构建 MLIR IR：

完整源码：`books/examples/chapter08/build_add.cpp`

```cpp
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

int main() {
    mlir::MLIRContext context;
    context.loadDialect<mlir::arith::ArithDialect>();
    context.loadDialect<mlir::func::FuncDialect>();

    mlir::OpBuilder builder(&context);
    auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());

    auto funcType = builder.getFunctionType(
        {builder.getI32Type(), builder.getI32Type()}, {builder.getI32Type()});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "add", funcType);

    auto *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    mlir::Value lhs = entryBlock->getArgument(0);
    mlir::Value rhs = entryBlock->getArgument(1);
    auto addOp =
        builder.create<mlir::arith::AddIOp>(builder.getUnknownLoc(), lhs, rhs);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                           addOp.getResult());

    module.dump();
    return 0;
}
```

运行：

```bash
cd books/examples/chapter08
cmake -G Ninja -S . -B build -DLLVM_DIR=$LLVM_DIR -DMLIR_DIR=$MLIR_DIR
ninja -C build && ./build/build_add
```

这段代码会输出：

```mlir
module {
  func.func @add(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    return %0 : i32
  }
}
```

**每个步骤的对应关系**：

| C++ 代码 | 生成的 MLIR |
|---------|------------|
| `builder.create<func::FuncOp>("add")` | `func.func @add` |
| `func.addEntryBlock()` | 创建 `^bb0` 入口块 |
| `builder.create<arith::AddFOp>(lhs, rhs)` | `%0 = arith.addf %a, %b : f32` |
| `builder.create<func::ReturnOp>(result)` | `return %0 : i32` |

### 从 C++ 到 MLIR 再到 Triton 的层次

```
C++ builder code     → 构建 MLIR Ops（本章）
TableGen 声明         → 自动生成 C++ 工具代码（第 9 章）
MLIR Ops + Types     → 组成 Dialect
Dialect + Pass       → 组成编译流水线（第 10-11 章）
Triton Dialect       → 组成 Triton 编译器（第 13-19 章）
```

### Operation 的"字典"组织方式

理解 MLIR 的关键是：所有 IR 都是 Operation。即使是最外层的 Module、函数定义、基本块——都是 Operation。这就像一个字典——每个字（Operation）都可以嵌套子字典（Region），而整个字典就是一个完整的模块。

```
字典比喻：
ModuleOp
  ├── 词条1: func.func @add         (一个 Op)
  │   └── 子词条（Region/Block）
  │       ├── 字1: arith.addi %a, %b
  │       └── 字2: func.return %result
  └── 词条2: func.func @main         (另一个 Op)
      └── 子词条（Region/Block）
          ├── 字1: arith.constant 0
          ├── 字2: func.call @add(...)
          └── 字3: func.return %result
```



---

## 📝 课后作业

### 作业 1：分析 MLIR 代码

```mlir
func.func @test(%a: i32, %b: i32) -> i32 {
    %cond = arith.cmpi sgt, %a, %b : i32
    %max = scf.if %cond -> i32 {
        scf.yield %a : i32
    } else {
        scf.yield %b : i32
    }
    func.return %max : i32
}
```

回答：
1. 这段代码中有几个 Operation？列出它们的名称。
2. 哪些 Operation 包含 Region？
3. `%max` 被哪些 Operation 使用？

### 作业 2：用 OpBuilder 创建 IR

假设你有一个 `OpBuilder builder`，写出创建以下 MLIR 代码的 C++ 代码：

```mlir
%result = arith.addi %a, %b : i32
```

### 作业 3：探索 Triton 的 Dialect

在 Triton 源码中找到 `include/triton/Dialect/` 目录，列出所有子目录（每个子目录对应一个 Dialect），填写以下表格：

| Dialect 名称 | 包含的主要 Op | 用途 |
|-------------|--------------|------|
| Triton | ? | ? |
| TritonGPU | ? | ? |
| TritonNvidiaGPU | ? | ? |

---

## 本章小结

- MLIR 的基本单位是 Operation，一切皆为 Operation（包括 Module 和 Function）
- Value 连接定义和使用，Type 可自定义扩展
- Attribute 携带编译时元数据，Region 支持嵌套子图
- Dialect 是一组相关 Op/Type/Attr 的集合，通过名称空间隔离
- MLIR 相比 LLVM IR 最大的优势：多层 IR、可扩展类型、嵌套 Region
