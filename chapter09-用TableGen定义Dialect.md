# 第 9 章：用 TableGen 定义 Dialect

> **本章目标**：能独立用 TableGen 定义一个完整的 Dialect。
>
> 驯龙手记：如果说 LLVM 的 TableGen 是龙语词典，那 MLIR 的 TableGen 就是
> "创造新龙语"——你可以定义全新的词汇（Op）、新的发声方式（Type）、
> 新的交流协议（Attr）。这是驯龙者从"使用者"变成"创造者"的关键一步。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter09/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `sample.mlir` | 9.x | Mini Dialect IR 文本 |
| `../chapter06/MiniDialect.td` | 9.x | TableGen 定义（复用 ch06） |

运行：

```bash
cd books/examples/chapter09
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 9.1 Dialect 定义的三要素

一个 MLIR Dialect 包含三个核心部分：

```
Triton Dialect 三要素:

  Operations (操作):
    tt.load, tt.store, tt.dot, tt.reduce, ...

  Types (类型):
    !tt.ptr<f32>, tensor<...>,

  Attributes (属性):
    CacheModifier, EvictionPolicy, ...
```

## 9.2 定义 Dialect 本身

```tablegen
// MyLangDialect.td

// 1. 引入 MLIR 基础定义
include "mlir/IR/OpBase.td"

// 2. 定义 Dialect
def MyLang_Dialect : Dialect {
    let name = "mylang";              // IR 前缀：mylang.add
    let cppNamespace = "::mlir::mylang";  // C++ 命名空间
    let summary = "My Language Dialect";
    let description = [{
        A simple dialect for educational purposes.
    }];
    let dependentDialects = [         // 依赖的 Dialect
        "arith::ArithDialect"
    ];
}
```

## 9.3 定义 Op

```tablegen
// 1. 先定义 Op 基类
class MyLang_Op<string mnemonic, list<Trait> traits = []> :
    Op<MyLang_Dialect, mnemonic, traits> {
}

// 2. 定义具体 Op

// 2a. 简单加法 Op
def MyLang_AddOp : MyLang_Op<"add", [Pure, SameOperandsAndResultType]> {
    let summary = "Addition operation";
    
    let arguments = (ins
        AnyType:$lhs,    // 任何类型
        AnyType:$rhs
    );
    let results = (outs AnyType:$result);
    
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// 2b. 带属性的 Op
def MyLang_MatMulOp : MyLang_Op<"matmul", [Pure]> {
    let summary = "Matrix multiplication";
    
    let arguments = (ins
        RankedTensorOf<[F32]>:$a,
        RankedTensorOf<[F32]>:$b,
        BoolAttr:$transpose_a    // 自定义属性
    );
    let results = (outs RankedTensorOf<[F32]>:$c);
    
    // 自定义验证
    let hasVerifier = 1;
    
    let extraClassDeclaration = [{
        static bool isSupported(int version) {
            return version >= 2;
        }
    }];
}

// 2c. 带 Region 的 Op
def MyLang_ReduceOp : MyLang_Op<"reduce", [Pure, SingleBlock]> {
    let summary = "Reduction operation";
    
    let arguments = (ins
        RankedTensorOf<[F32]>:$input,
        I32Attr:$axis
    );
    let results = (outs AnyType:$result);
    let regions = (region SizedRegion<1>:$combineOp);
}
```

## 9.4 Op 各部分详解

### `arguments` — 输入与属性

```tablegen
let arguments = (ins
    // 操作数（运行时值）
    I32:$input1,              // 标量 i32
    F32:$input2,              // 标量 f32
    RankedTensorOf<[F32]>:$tensor,  // 张量
    
    // 属性（编译时值）
    I32Attr:$int_attr,        // 整数属性
    F32Attr:$float_attr,      // 浮点属性
    BoolAttr:$flag,           // 布尔属性
    OptionalAttr<StrAttr>:$name,  // 可选属性
    DefaultValuedAttr<I32Attr, "4">:$defaulted  // 有默认值
);
```

### `results` — 输出

```tablegen
let results = (outs
    I32:$result1,             // 单个结果
    Variadic<AnyType>:$results  // 可变数量结果
);
```

### `assemblyFormat` — 打印格式

```tablegen
let assemblyFormat = [{
    $operand1 `,` $operand2 attr-dict `:` type($result)
    // 会打印为：
    // %r = mylang.op %a, %b : i32
}];

// 更复杂的格式
let assemblyFormat = [{
    $a (`transpose` $transpose^)? attr-dict `:` type($a) `->` type($result)
    // ^ 表示在条件为 true 时打印
    // ? 表示可选部分
}];
```

## 9.5 定义自定义 Type

```tablegen
// 1. 定义 Type 基类
class MyLang_TypeDef<string name, string mnemonic> :
    TypeDef<MyLang_Dialect, name> {
    let mnemonic = mnemonic;
}

// 2. 定义具体的 Type

// 2a. 简单类型：无参数
def MyLang_MyType : MyLang_TypeDef<"MyType", "mytype"> {
    let summary = "A custom type";
    let description = "A simple type without parameters";
}

// 2b. 带参数的类型（类似 TT 的指针类型）
def MyLang_PtrType : MyLang_TypeDef<"Pointer", "ptr"> {
    let summary = "Pointer type";
    let parameters = (ins
        "Type":$pointeeType,     // 指向的类型
        "int":$addressSpace      // 地址空间
    );
    let assemblyFormat = "`<` $pointeeType `,` $addressSpace `>`";
}
```

## 9.6 定义自定义 Attribute

```tablegen
// 自定义属性
def MyLang_EncodingAttr : AttrDef<MyLang_Dialect, "Encoding"> {
    let mnemonic = "encoding";
    let parameters = (ins
        ArrayRefParameter<"unsigned">:$sizePerThread,
        ArrayRefParameter<"unsigned">:$order
    );
    let assemblyFormat = "`<` `{` `sizePerThread` `=` $sizePerThread `}` `>`";
}
```

## 9.7 CMake 配置

写出 .td 文件后，需要 CMake 配置来自动生成 C++ 代码：

```cmake
# MyLang/IR/CMakeLists.txt

# 1. 声明 TableGen 目标
set(LLVM_TARGET_DEFINITIONS MyLangOps.td)

# 2. 生成 Op 定义
mlir_tablegen(MyLangOps.h.inc -gen-op-decls)
mlir_tablegen(MyLangOps.cpp.inc -gen-op-defs)

# 3. 生成 Dialect 定义
mlir_tablegen(MyLangDialect.h.inc -gen-dialect-decls)
mlir_tablegen(MyLangDialect.cpp.inc -gen-dialect-defs)

# 4. 添加构建依赖
add_public_tablegen_target(MyLangOpsIncGen)

# 5. 构建库
add_mlir_dialect_library(MLIRMyLang
    Dialect.cpp
    Ops.cpp
    Types.cpp
    DEPENDS
    MyLangOpsIncGen
    MyLangTypesIncGen
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)
```

## 9.8 C++ 中的使用

```cpp
// Dialect.cpp — Dialect 初始化
#include "MyLang/IR/MyLangDialect.h"
#include "MyLang/IR/MyLangOps.h"

void MyLangDialect::initialize() {
    addOperations<
        #define GET_OP_LIST
        #include "MyLang/IR/MyLangOps.cpp.inc"
    >();
    registerTypes();
}

// Builder 中使用
OpBuilder builder(context);
auto addOp = builder.create<MyLang::AddOp>(
    loc,
    lhs, rhs,
    resultType
);

// 使用 Op
auto result = addOp.getResult();
```

## 9.9 完整示例：一个极简 Dialect

```tablegen
// MiniDialect.td
include "mlir/IR/OpBase.td"

def Mini_Dialect : Dialect {
    let name = "mini";
    let cppNamespace = "::mlir::mini";
}

class Mini_Op<string mnemonic> :
    Op<Mini_Dialect, mnemonic, [Pure]> {
}

def Mini_AddOp : Mini_Op<"add"> {
    let arguments = (ins F32:$a, F32:$b);
    let results = (outs F32:$c);
    let assemblyFormat = "$a `,` $b attr-dict `:` type($c)";
}

def Mini_MulOp : Mini_Op<"mul"> {
    let arguments = (ins F32:$a, F32:$b);
    let results = (outs F32:$c);
    let assemblyFormat = "$a `,` $b attr-dict `:` type($c)";
}
```

```mlir
; 生成的 IR
%c = mini.add %a, %b : f32
%d = mini.mul %c, %e : f32
```

---

## 📝 课后作业

### 作业 1：定义 Op

为你的自定义 Dialect `Mathy` 定义以下 Op：

1. `mathy.sqrt` — 浮点平方根（输入 F32，输出 F32）
2. `mathy.neg` — 浮点取反（输入 F32，输出 F32）
3. `mathy.abs` — 浮点绝对值（输入 F32，输出 F32，可选属性 `fast` 表示快速近似）

### 作业 2：定义带 Region 的 Op

在 Triton 中找到 `tt.reduce` 的 TableGen 定义，理解它的 Region 结构。然后模仿它定义一个 `mathy.fold` Op，接收一个 `tensor<*xf32>` 和一个组合 Region，返回 F32。

### 作业 3：阅读 TritonOps.td

在 Triton 源码中找到 `TT_LoadOp` 的定义，回答：
1. 它有几个操作数？各自的类型是什么？
2. 它有几个属性？分别是什么？
3. 它有 Region 吗？

---

## 本章小结

- TableGen 定义 Dialect 的三要素：**Dialect 本身**、**Ops**、**Types** 和 **Attributes**
- Op 定义的核心：`arguments`（输入）、`results`（输出）、`regions`（子图）、`assemblyFormat`（打印格式）
- 类型和属性也可以自定义——`parameters` 描述它们的内部结构
- C++ 代码通过 `#include "XXX.cpp.inc"` 使用生成的代码
- Triton 完整展示了如何定义多层 Dialect（TT → TTG → TTNV）
