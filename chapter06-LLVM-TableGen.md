# 第 6 章：LLVM TableGen 入门

> **本章目标**：理解 TableGen 的作用和语法，能读懂 Triton 中的 TableGen 文件。
>
> 驯龙手记：TableGen 是"龙语词典"——你不需要从零造词，而是用已有的词汇
> （class、def、let）描述龙的每一块骨头、每一片鳞甲。这本词典是 LLVM 社区的
> 智慧结晶，学会用它就能快速定义新龙种。
> TableGen 是 LLVM 和 MLIR 的**声明式代码生成语言**。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter06/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `MiniDialect.td` | 6.3 | TableGen Dialect/Op 定义 |
| `run_examples.sh` | 作业3 | mlir-tblgen 生成验证 |

运行：

```bash
cd books/examples/chapter06
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 6.1 什么是 TableGen

TableGen 不是"通用的"——它专门用于编译器开发中的**代码生成**。

### 解决的问题

```
编译器开发中大量的重复代码：
  - Op 的 C++ 类定义
  - Op 的序列化/反序列化
  - Op 的验证逻辑
  - Op 的格式打印
  = 手写 -> 繁琐且容易出错

TableGen 方案：
  .td 文件（声明式描述）
    │
    ▼
  TableGen 工具（llvm-tblgen / mlir-tblgen）
    │
    ├── Ops.h.inc        ← Op C++ 声明
    ├── Ops.cpp.inc      ← Op C++ 实现
    ├── OpsDialect.h.inc ← Dialect 声明
    └── ...
```

### 工作流程

```
.td 文件（描述 Op 结构）
    -->  [mlir-tblgen -gen-op-defs]
    -->  .inc 文件（C++ 代码）
    -->  #include 到你的库中
```

## 6.2 TableGen 基本语法

### 注释

```tablegen
// 单行注释
/* 多行注释 */
```

### 基本类型

```tablegen
// 整型
42
0xFF

// 字符串
"hello"
"hello\nworld"   // 支持转义

// 布尔
true
false

// 列表
[1, 2, 3]
["a", "b", "c"]
```

### 记录（Record）— 核心概念

```tablegen
// 定义一个记录（Record）
def MyRecord : SomeParent {
    let field1 = 42;
    let field2 = "hello";
}

// 使用记录
def Instance : MyRecord {
    let field1 = 100;  // 覆盖父类默认值
}
```

### 类（Class）— 模板

```tablegen
// 定义类（参数化模板）
class Animal<string name, int legs> {
    string Name = name;
    int Legs = legs;
    string Sound = "?";  // 默认值
}

// 继承类并指定参数
def Dog : Animal<"dog", 4> {
    let Sound = "woof";
}

def Human : Animal<"human"> {  // 使用部分默认参数
    let Legs = 2;
}

// 使用变量
def CatWithSameLegs as Human {
    // 复制 Human 的 Legs 值
}
```

## 6.3 MLIR 中 TableGen 的使用

### 定义 Dialect

```tablegen
// include/triton/Dialect/Triton/IR/TritonDialect.td
def Triton_Dialect : Dialect {
    let name = "tt";                          // IR 中的前缀：tt.func, tt.load
    let cppNamespace = "::mlir::triton";       // C++ 命名空间
    let dependentDialects = [                 // 依赖的 Dialect
        "arith::ArithDialect",
        "math::MathDialect",
        "scf::SCFDialect",
    ];
}
```

### 定义 Op

```tablegen
// include/triton/Dialect/Triton/IR/TritonOps.td
def TT_ReduceOp: TT_Op<"reduce",
    [Pure,                        // 无副作用
     SameOperandsShape,           // 所有操作数形状相同
     SameOperandsEncoding,        // 编码相同
     SingleBlock,                 // 包含一个 Block
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {

    // 摘要
    let summary = "Reduction using generic combination algorithm";

    // 参数（IR 中的操作数 + 属性）
    let arguments = (ins
        Variadic<TT_Tensor>:$srcs,   // 操作数：多个张量
        I32Attr:$axis                // 属性：轴
    );

    // 结果
    let results = (outs
        Variadic<TT_Type>:$result    // 多个结果
    );

    // Region（包含子图）
    let regions = (region SizedRegion<1>:$combineOp);

    // 验证
    let hasVerifier = 1;
    let hasRegionVerifier = 1;

    // 额外的 C++ 方法声明
    let extraClassDeclaration = [{
        llvm::SmallVector<RankedTensorType> getInputTypes();
        ::mlir::Operation *getSingleCombiner();
    }];
}
```

### 类型定义

```tablegen
// include/triton/Dialect/Triton/IR/TritonTypes.td

// 指针类型
def TT_PtrType : TritonTypeDef<"Pointer", "ptr"> {
    let parameters = (ins
        "Type":$pointeeType,
        "int":$addressSpace
    );
    let hasCustomAssemblyFormat = 1;
}

// 类型约束（辅助宏）
def TT_Float : AnyTypeOf<[F8E4M3FN, F16, BF16, F32, F64], "floating-point">;
def TT_FloatTensor : RankedTensorOf<[TT_Float]>;
def TT_FloatLike : AnyTypeOf<[TT_Float, TT_FloatTensor]>;
```

### 属性定义

```tablegen
// TritonGPU 编码属性
def BlockedEncodingAttr : TritonGPU_Attr<"BlockedEncoding", "blocked", [...]> {
    let parameters = (ins
        ArrayRefParameter<"unsigned">:$sizePerThread,
        ArrayRefParameter<"unsigned">:$threadsPerWarp,
        ArrayRefParameter<"unsigned">:$warpsPerCTA,
        ArrayRefParameter<"unsigned">:$order,
        "CGAEncodingAttr":$CTALayout
    );
}
```

## 6.4 mlir-tblgen 生成器

```bash
# 常用的 mlir-tblgen 生成模式

# 生成 Op 定义（C++）
mlir-tblgen -gen-op-defs TritonOps.td -o /dev/stdout

# 生成 Op 声明（头文件）
mlir-tblgen -gen-op-decls TritonOps.td -o Ops.h.inc

# 生成 Dialect 声明
mlir-tblgen -gen-dialect-decls TritonDialect.td -o Dialect.h.inc

# 生成 Dialect 定义
mlir-tblgen -gen-dialect-defs TritonDialect.td -o Dialect.cpp.inc

# 生成类型定义
mlir-tblgen -gen-type-defs TritonTypes.td -o Types.cpp.inc

# 生成属性定义
mlir-tblgen -gen-attr-defs TritonAttrDefs.td -o Attrs.cpp.inc

# 查看所有生成器
mlir-tblgen --help
```

## 6.5 自定义生成器

你也可以写自己的 `mlir-tblgen` 后端——这在 MLIR 中很常见：

```cpp
// 例如：生成自动 Op 文档、Pass 注册等
// triton/include/triton/Dialect/TritonGPU/Transforms/Passes.td
def TritonGPUCoalesce : Pass<"tritongpu-coalesce"> {
    let summary = "Optimize memory access patterns";
    let dependentDialects = ["triton::gpu::TritonGPUDialect"];
    let constructor = "mlir::triton::gpu::createCoalescePass()";
}

// 生成的代码会自动处理：
// - Pass 的 C++ 基类
// - 命令行参数解析
// - 统计信息收集
```

## 6.6 Triton 中的 TableGen 文件组织

```
include/triton/Dialect/Triton/IR/
├── TritonDialect.td       → Dialect 定义
├── TritonOps.td           → 所有 Op 定义（1379 行）
├── TritonTypes.td         → 类型系统（164 行）
├── TritonAttrDefs.td      → 属性定义（154 行）
├── TritonOpInterfaces.td  → Op 接口
└── TritonTypeInterfaces.td → 类型接口

include/triton/Dialect/TritonGPU/IR/
├── TritonGPUAttrDefs.td   → 编码属性（1503 行！）
├── TritonGPUOps.td        → GPU 操作（803 行）
└── ...

third_party/nvidia/include/Dialect/NVWS/IR/
├── NVWSOps.td             → NVIDIA Warp Specialization Ops
└── NVWSAttrDefs.td        → WS 属性
```

## 6.7 从 TableGen 到 C++ 的映射

| TableGen 构造 | 生成的 C++ |
|---------------|-----------|
| `def MyOp : Op<"my_op">` | `class MyOp : public Op<...>` |
| `let arguments = (ins I32:$input)` | `Value getInput()` |
| `let results = (outs F32:$output)` | `Value getOutput()` |
| `let hasVerifier = 1` | `LogicalResult verify()` |
| `extraClassDeclaration = [{...}]` | 直接添加到最后生成的类中 |

---

## 📝 课后作业

### 作业 1：读 TritonOps.td

阅读 `include/triton/Dialect/Triton/IR/TritonOps.td`，回答：

1. `TT_LoadOp` 有几个参数（arguments）？分别是什么类型？
2. `TT_DotOp` 的 `hasVerifier` 被设置为什么？
3. `TT_ReduceOp` 的 `getSingleCombiner` 是在哪里声明的？

### 作业 2：写一个简单的 TD 定义

模仿 `TT_ClampFOp`（TritonOps.td 约第 114 行），写一个 `TT_ClampIOp`（整数版的 clamp），参数为 `x`、`min`、`max`，类型为 `TT_IntLike`。

### 作业 3：运行 mlir-tblgen

完整示例：`books/examples/chapter06/MiniDialect.td`

```bash
cd books/examples/chapter06
./run_examples.sh
# 或手动：
mlir-tblgen -I$(brew --prefix llvm)/include MiniDialect.td \
    --gen-op-decls -o build/MiniOps.h.inc
```

---

## 本章小结

- TableGen 是声明式的代码生成语言，专为编译器开发设计
- `.td` 文件定义 Dialect、Op、Type、Attr 的结构
- `mlir-tblgen` 工具自动生成 C++ 代码（.inc 文件）
- `class` 是模板，`def` 是具体实例——这是理解 TableGen 的关键
- 属性(`let arguments`)、结果(`let results`)、验证(`hasVerifier`)是最常用的三个描述
- Triton 用 TableGen 定义了完整的 TT、TTG、TTNV 三个 Dialect
