# 第 11 章：方言转换框架

> **本章目标**：理解 MLIR 的 Dialect Conversion 框架——这是 Triton 的 TT → TTG → LLVM 转换的核心机制。

> 驯龙手记：方言转换就是龙的"变形术"。同一头龙可以在水栖（TT Dialect）、
> 陆栖（TTG Dialect）、飞行（LLVM Dialect）三种形态间自由切换。
> ConversionTarget 决定它能变成什么形态，TypeConverter 负责鳞片颜色的转换。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter11/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `input.mlir` | 11.x | 方言转换输入 |
| `run_examples.sh` | 11.x | arith→llvm 降级演示 |

运行：

```bash
cd books/examples/chapter11
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 11.1 什么是方言转换

方言转换就是将**源 Dialect** 的操作转换为**目标 Dialect** 的操作。

```
源 Dialect (非法):                 目标 Dialect (合法):
  tt.load, tt.addptr, arith.addi    llvm.load, llvm.getelementptr, arith.addi

转换: 源 Op --> 目标 Op (通过 ConversionPattern)
```

### Triton 中的两次方言转换

```
TritonToTritonGPU → 将 tt 方言转为 ttg 方言（添加 GPU 编码）
TritonGPUToLLVM   → 将 ttg 方言转为 llvm 方言（最终代码生成）
```

## 11.2 方言转换的核心组件

```cpp
// 三个核心组件配合工作

// 1. ConversionTarget — 什么操作是"合法的"
ConversionTarget target(context);
target.addLegalDialect<LLVM::LLVMDialect>();       // LLVM Op 全部合法
target.addIllegalDialect<triton::TritonGPUDialect>(); // TTG Op 需要转换
target.addLegalOp<arith::ConstantOp>();              // 特定 Op 合法

// 2. TypeConverter — 如何转换类型
TypeConverter converter;
converter.addConversion([](Type type) -> Type {
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
        return tensorType.cloneWithEncoding(/*new encoding*/);
    }
    return type;
});

// 3. RewritePatternSet — 转换模式集合
RewritePatternSet patterns(context);
patterns.add<GenericOpPattern<tt::LoadOp>>(converter, context);
patterns.add<GenericOpPattern<tt::StoreOp>>(converter, context);
```

## 11.3 ConversionTarget

ConversionTarget 定义什么是"合法的"——即**转换目标**：

```cpp
ConversionTarget target(context);

// 完整的 Dialect 合法
target.addLegalDialect<LLVM::LLVMDialect>();
target.addLegalDialect<arith::ArithDialect>();

// 完整的 Dialect 非法（需要转换）
target.addIllegalDialect<triton::TritonDialect>();

// 特定的 Op 合法/非法
target.addLegalOp<tt::GetProgramIdOp>();
target.addIllegalOp<triton::LoadOp>();

// 动态条件——某些 Op 可能合法也可能不
target.addDynamicallyLegalOp<arith::ConstantOp>(
    [&](arith::ConstantOp op) -> bool {
        return converter.isLegal(op.getType());
    }
);
```

## 11.4 TypeConverter

TypeConverter 定义类型如何映射：

```cpp
TypeConverter converter;

// 添加类型转换规则
converter.addConversion([](Type type) -> Type {
    // 从 Triton IR 类型转为 LLVM 类型
    if (isa<triton::PointerType>(type)) {
        auto ptrType = cast<triton::PointerType>(type);
        return LLVM::LLVMPointerType::get(
            ptrType.getContext(),
            ptrType.getAddressSpace()
        );
    }
    // 不匹配此规则的，继续尝试其他规则
    return type;
});

// 添加另一种规则（顺序匹配）
converter.addConversion([](RankedTensorType type) -> Type {
    // 张量类型降级为标量类型
    return type.getElementType();
});

// 签名转换（函数参数和返回值的整体转换）
converter.addConversion([&](FunctionType type) -> Type {
    // 保持参数类型不变
    SmallVector<Type> inputs, results;
    converter.convertTypes(type.getInputs(), inputs);
    converter.convertTypes(type.getResults(), results);
    return FunctionType::get(type.getContext(), inputs, results);
});
```

## 11.5 ConversionPattern — 转换模式

```cpp
// OpConversionPattern 是方言转换专用的 Pattern

// 通用模式：保持 Op 结构，只转换类型
template <class Op>
struct GenericOpPattern : public OpConversionPattern<Op> {
    using OpConversionPattern<Op>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        // adaptor.getOperands() → 已经类型转换后的操作数
        // op.getOperands() → 原始操作数
        
        SmallVector<Type> retTypes;
        if (failed(this->getTypeConverter()->convertTypes(
                op->getResultTypes(), retTypes)))
            return failure();

        // 用转换后的类型和操作数创建新的 Op
        rewriter.replaceOpWithNewOp<Op>(op, retTypes,
                                         adaptor.getOperands(),
                                         op->getAttrs());
        return success();
    }
};

// 特化模式：对于需要改变 Op 类型的场景
struct LoadOpConversion : public OpConversionPattern<tt::LoadOp> {
    LogicalResult
    matchAndRewrite(tt::LoadOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        // 这里的 adaptor 中的操作数已经是转换后的类型
        // 可能需要将 tt.load 转换为 llvm.load
        auto newLoad = rewriter.create<LLVM::LoadOp>(
            op.getLoc(),
            /*converted type*/,
            adaptor.getPtr()  // 已经转换的指针
        );
        rewriter.replaceOp(op, newLoad.getResult());
        return success();
    }
};
```

### OpAdaptor

Adaptor 的作用：提供**类型转换后**的操作数。

```cpp
struct MyPattern : public OpConversionPattern<MyOp> {
    LogicalResult matchAndRewrite(
        MyOp op,
        typename MyOp::Adaptor adaptor,  // ← 类型已转换
        ConversionPatternRewriter &rewriter) const {

        // 原始操作数（类型转换前）
        Value origOperand = op.getOperand(0);
        Type origType = origOperand.getType();  // tensor<128xf32, #blocked>

        // adaptor 操作数（类型转换后）
        Value convertedOperand = adaptor.getOperand(0);
        Type convertedType = convertedOperand.getType();  // !llvm.struct<(f32, f32, ...)>
    }
};
```

## 11.6 执行转换

MLIR 支持两种转换模式：

### 完全转换（applyFullConversion）

```cpp
// 所有非法 Op 必须被转换
if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
```

### 部分转换（applyPartialConversion）

```cpp
// 允许保留部分非法 Op（后续其他 Pass 处理）
if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
```

## 11.7 TritonToTritonGPU 实例分析

```cpp
// lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp

// 1. TypeConverter：为张量添加 GPU 编码
class TritonGPUTypeConverter : public TypeConverter {
    Type convertType(Type type) {
        if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
            // 为张量设置默认的 blocked 编码
            return addDefaultEncoding(tensorType);
        }
        return type;
    }
};

// 2. ConversionTarget：ttg 合法，tt 非法
TritonGPUConversionTarget target(context, typeConverter);
target.addIllegalDialect<triton::TritonDialect>();
target.addLegalDialect<triton::gpu::TritonGPUDialect>();

// 3. Patterns：通用转换 + 特化模式
RewritePatternSet patterns(context);
populateArithPatternsAndLegality(typeConverter, patterns, target);
// 例如 ConstantOp 需要特化处理
patterns.add<ArithConstantPattern>(typeConverter, context);
```

## 11.8 TritonGPUToLLVM 实例分析

```cpp
// third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp

// 1. TypeConverter：TTG 类型 → LLVM 类型
//    tensor<128xf32, #blocked> → !llvm.struct<(32 x f32)>
//    编码指导了张量如何展开为多个 LLVM 值

// 2. ConversionTarget：llvm 合法，ttg 非法
target.addLegalDialect<LLVM::LLVMDialect>();
target.addIllegalDialect<triton::gpu::TritonGPUDialect>();

// 3. 注册各种降级模式
patterns.add<LoadOpConversion>(typeConverter, ...);
patterns.add<StoreOpConversion>(typeConverter, ...);
patterns.add<ReduceOpConversion>(typeConverter, ...);
// ...数百个模式
```

## 11.9 调试转换问题

```cpp
// 常用调试技巧

// 1. 打印所有失败的转换
target.addDynamicallyLegalOp<...>([](auto op) -> bool {
    if (!legal(op)) {
        op.emitError("not legal");
        return false;
    }
    return true;
});

// 2. 查看 IR 在每个 Pass 前后
// --mlir-print-ir-after-all

// 3. 在 C++ 中断言
assert(succeeded(converter.convertType(type, convertedType)));
```

---

## 📝 课后作业

### 作业 1：概念理解

解释以下三个组件各自的作用：

1. `ConversionTarget`
2. `TypeConverter`
3. `OpConversionPattern`

### 作业 2：设计一个转换

假如你要将自定义的 `Mathy` Dialect 转换为 `arith` Dialect：

```mlir
; 源
%result = mathy.sqrt %x : f32
; 目标
%result = math.exp(math.log(%x) * 0.5) : f32
```

写出 `SqrtOpConversion` 的伪代码框架。

### 作业 3：阅读 Triton 代码

在 `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp` 中找到 `populateArithPatternsAndLegality` 函数，回答：
1. 它为哪些 arith Op 设置了合法性条件？
2. 哪些 arith Op 是不合法的？为什么？

---

## 本章小结

- 方言转换是将源 Dialect 转为目标 Dialect 的核心机制
- 三个核心组件：`ConversionTarget`（合法性）、`TypeConverter`（类型映射）、`ConversionPattern`（Op 转换）
- `OpAdaptor` 提供类型转换后的操作数
- `applyFullConversion` vs `applyPartialConversion` 控制转换严格程度
- Triton 的两次方言转换（TT → TTG → LLVM）是 MLIR 方言转换的教科书级案例
- 第 22 章将教你如何在第三方后端的转换中应用这些知识
