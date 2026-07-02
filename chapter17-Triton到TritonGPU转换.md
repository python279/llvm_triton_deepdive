# 第 17 章：Triton → TritonGPU 转换

> **本章目标**：理解 TT → TTG 转换的核心机制，这是 MLIR 方言转换框架在 Triton 中的实际应用案例。

> 驯龙手记：这是给海神穿上铠甲的过程。TT（龙骨）只有最基本的骨架结构，
> TTG（铠甲）则加上了每块鳞片的位置（Encoding）。ConversionTarget 决定
> 哪块鳞片放在哪里，TypeConverter 则确保每片鳞甲都能完美贴合。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter17/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `sample.ttir` | 17.x | 转换前 TTIR |
| `sample.ttgir` | 17.x | 转换后 TTGIR |

运行：

```bash
cd books/examples/chapter17
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 17.1 转换的目的

TT → TTG 转换的核心任务：

1. **为每个张量类型添加编码属性**
2. **在布局不兼容时插入 `convert_layout`**
3. **确保所有 Op 的操作数布局一致**

```
TT IR（无编码）                TTG IR（有编码）
tensor<128xf32>     ──→    tensor<128xf32, #blocked>
tensor<128x256xf16> ──→    tensor<128x256xf16, #blocked1>

#blocked/blocked1 就是 TypeConverter 添加的布局编码
```

## 17.2 Pass 入口

`lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp`（约 768 行）。

```cpp
namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONTOTRITONGPU
#include "triton/Conversion/TritonToTritonGPU/Passes.h.inc"
}
```

这个 Pass 使用 MLIR 的标准方言转换框架：

```cpp
void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // 1. 创建 TypeConverter
    TritonGPUTypeConverter typeConverter;
    
    // 2. 创建 ConversionTarget
    TritonGPUConversionTarget target(context, typeConverter);
    target.addLegalDialect<triton::gpu::TritonGPUDialect>();
    target.addIllegalDialect<triton::TritonDialect>();
    
    // 3. 创建 Pattern
    RewritePatternSet patterns(context);
    populateArithPatternsAndLegality(typeConverter, patterns, target);
    populateTritonPatternsAndLegality(typeConverter, patterns, target);
    
    // 4. 执行
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
}
```

## 17.3 TypeConverter

`TritonGPUTypeConverter` 负责类型转换——为张量添加编码：

```cpp
class TritonGPUTypeConverter : public TypeConverter {
    Type convertType(Type type) {
        // 张量类型：添加默认编码
        if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
            Attribute encoding = getDefaultEncoding(tensorType);
            return tensorType.cloneWithEncoding(encoding);
        }
        // 非张量类型保持不变
        return type;
    }
    
    Attribute getDefaultEncoding(RankedTensorType type) {
        // 根据张量形状、目标 GPU 等选择默认编码
        // 后续 Coalesce Pass 会进一步优化
        auto shape = type.getShape();
        return BlockedEncodingAttr::get(
            type.getContext(),
            shape,
            /*sizePerThread=*/{1, ..., 1},
            /*threadsPerWarp=*/...,
            /*warpsPerCTA=*/...,
            ...
        );
    }
};
```

## 17.4 ConversionPattern

通用模式：保持 Op 结构，只转换类型：

```cpp
template <class Op>
struct GenericOpPattern : public OpConversionPattern<Op> {
    LogicalResult
    matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        SmallVector<Type> retTypes;
        if (failed(this->getTypeConverter()->convertTypes(
                op->getResultTypes(), retTypes)))
            return failure();
        // 用转换后的类型重新创建 Op
        rewriter.replaceOpWithNewOp<Op>(op, retTypes,
                                         adaptor.getOperands(),
                                         op->getAttrs());
        return success();
    }
};
```

特化模式：某些 Op 需要特殊的转换逻辑，如 `arith::ConstantOp`：

```cpp
class ArithConstantPattern : public OpConversionPattern<arith::ConstantOp> {
    LogicalResult
    matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        Type retType = getTypeConverter()->convertType(op.getType());
        auto retShapedType = cast<ShapedType>(retType);
        auto value = dyn_cast<DenseElementsAttr>(adaptor.getValue());
        value = value.reshape(retShapedType);  // 添加编码
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(
            op, retShapedType, value);
        return success();
    }
};
```

## 17.5 布局选择策略

TT → TTG 转换时使用的编码是"初步"的——后续的 Pass 会优化。

**阶段 1：初始转换**

所有张量得到一个**初始编码**（通常是 `blocked` 编码，默认参数）。

**阶段 2：Coalesce 优化**

`CoalescePass` 分析实际的内存访问模式，为每个 `load`/`store` 选择最优编码：

```cpp
// CoalescePass 分析轴信息
ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
// 根据 strides/contiguity 选择向量化宽度
int vectorSize = axisInfoAnalysis.getContiguity(loadOp);
// 重新设置 sizePerThread
```

**阶段 3：布局传播**

`RemoveLayoutConversionsPass` 传播布局信息：

```cpp
// 如果 Op A 的输出是 #blocked1，Op B 的输入是 #blocked1，
// 且 B 也是 #blocked1，则不需要 convert_layout
// 消除冗余的 convert_layout → 布局信息传递到相邻 Op
```

## 17.6 合法性检查

```cpp
class TritonGPUConversionTarget : public ConversionTarget {
    TritonGPUConversionTarget(MLIRContext &ctx, TritonGPUTypeConverter &tc) {
        addLegalDialect<TritonGPUDialect>();
        addLegalDialect<arith::ArithDialect>();
        
        addDynamicallyLegalOp<arith::ConstantOp>(
            [&](arith::ConstantOp op) {
                // ConstantOp 只有在类型合法时才合法
                return tc.isLegal(op.getType());
            });
    }
};
```

## 17.7 作用域与限制

TT → TTG 转换**只处理设备无关的部分**。例如：
- `tt.load` → 添加编码，变成带编码的 `tt.load`
- `arith.addf` → 保持 Op 不变，只改类型

它**不处理**：
- Tensor Core 的选择（`AccelerateMatmulPass` 处理）
- 共享内存分配（`AllocateSharedMemoryPass` 处理）
- 软件流水线（`PipelinePass` 处理）

---

## 📝 课后作业

### 作业 1：追踪类型转换

使用 `TRITON_KERNEL_DUMP=1` 导出 `.ttir` 和 `.ttgir`：

```bash
# 对比两个文件，观察类型变化
diff <(grep -E 'tensor<.*>|!tt\.ptr' kernel.ttir) \
     <(grep -E 'tensor<.*>|!tt\.ptr' kernel.ttgir)
```

1. `.ttir` 中的 `tensor<1024xf32>` 在 `.ttgir` 中变成了什么？
2. 指针类型有变化吗？

### 作业 2：阅读转换 Pass

在 `lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp` 中：

1. 找到 `GenericOpPattern` 的定义
2. 找出哪些 Op 有特化的转换模式（不通过 GenericOpPattern）
3. 思考：为什么 `arith::ConstantOp` 需要特化？

### 作业 3：添加自定义模式

假设你想在 TT → TTG 转换中添加一个新的 Op `tt.my_op` 的支持，需要修改哪些文件？

---

## 本章小结

- TT → TTG 转换是 MLIR 方言转换框架的典型应用
- 三个核心组件：`TritonGPUTypeConverter`（添加编码）、`ConversionTarget`（合法性）、`ConversionPattern`（Op 转换）
- 初始编码只是"占位"，后续 Coalesce 等 Pass 会优化布局
- 通用模式 `GenericOpPattern` 覆盖绝大多数 Op
- 某些 Op（如 `arith::ConstantOp`）需要特化转换模式
- 转换完成后，`.ttgir` 中的每个张量都带有一个编码属性
