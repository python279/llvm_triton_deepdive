# 第 10 章：MLIR Pass 编程

> **本章目标**：理解 MLIR Pass 框架，能编写和注册 Pass，使用 Pattern Rewrite 进行 IR 转换。

> 驯龙手记：MLIR 的 Pass 比 LLVM Pass 更灵活——你可以指挥龙的不同部位
> （OperationPass<> 可以作用于任意 Op），而不仅仅是整体动作（FunctionPass）。
> Pattern Rewrite 则是教龙学会"条件反射"——当遇到某种情况时自动执行对应动作。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter10/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `sample.mlir` | 10.x | arith Pass 测试输入 |
| `count_ops.py` | 作业 | 统计 Operation 数量 |

运行：

```bash
cd books/examples/chapter10
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 10.1 MLIR Pass 概览

```
LLVM Pass 框架:               MLIR Pass 框架:
  FunctionPass (只处理函数)      OperationPass<> (处理任意Op)

MLIR 的 Pass 是 LLVM Pass 的泛化——可以处理任意 Operation，不仅仅是函数
```

### Pass 类型

```cpp
// 1. OperationPass — 处理特定类型的 Op（最常用）
struct MyPass : public OperationPass<ModuleOp> {
    // 处理 ModuleOp
};

// 2. InterfacePass — 处理实现了某个接口的 Op
struct MyInterfacePass : public InterfacePass<FunctionOpInterface> {
    // 处理实现了 FunctionOpInterface 的 Op
};

// 3. Pass（不限制类型）
struct MyGenericPass : public Pass {
    // 运行在所有 Op 上
};
```

## 10.2 写一个 MLIR Pass

### 第一步：定义 Pass（.td 文件）

```tablegen
// include/MyLang/Transforms/Passes.td
def MyLangRemoveDeadCode : Pass<"mylang-remove-dead-code"> {
    let summary = "Remove dead operations in MyLang";
    let dependentDialects = ["mlir::mylang::MyLangDialect"];
    let constructor = "mlir::mylang::createRemoveDeadCodePass()";
}
```

### 第二步：注册 Pass

```cpp
// include/MyLang/Transforms/Passes.h
namespace mlir::mylang {
    std::unique_ptr<Pass> createRemoveDeadCodePass();
}

// 自动生成
#define GEN_PASS_DEF_MYLANGREMOVEDEADCODE
#include "MyLang/Transforms/Passes.h.inc"
```

### 第三步：实现 Pass

```cpp
// lib/MyLang/Transforms/RemoveDeadCode.cpp
#include "MyLang/Transforms/Passes.h"

namespace {
struct RemoveDeadCodePass
    : public impl::MyLangRemoveDeadCodeBase<RemoveDeadCodePass> {

    void runOnOperation() override {
        // getOperation() 返回 Pass 处理的 Op
        // 如果 Pass 是 OperationPass<ModuleOp>，返回 ModuleOp
        Operation *op = getOperation();

        // 遍历所有子 Op，删除未被使用的
        op->walk([&](Operation *innerOp) {
            if (isTriviallyDead(innerOp)) {
                // PatternRewriter 用于安全删除
                // 但在 walk 中不能直接用 rewriter.eraseOp
                // 通常使用 GreedyPatternRewriteDriver
            }
        });
    }

    bool isTriviallyDead(Operation *op) {
        return op->use_empty();  // 结果没有被任何人使用
    }
};
} // namespace
```

## 10.3 Pattern Rewrite — 重写模式

Pattern Rewrite 是 MLIR 中最重要、最强大的 IR 转换机制。

### `OpRewritePattern`

```cpp
// 匹配并重写单个 Op
struct SimplifyAddZero : public OpRewritePattern<arith::AddFOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(arith::AddFOp op,
                                  PatternRewriter &rewriter) const override {
        // match：检查是否是 x + 0
        Value rhs = op.getRhs();
        auto constOp = rhs.getDefiningOp<arith::ConstantOp>();
        if (!constOp) return failure();

        auto attr = constOp.getValue().dyn_cast<FloatAttr>();
        if (!attr || !attr.getValue().isZero())
            return failure();

        // rewrite：x + 0 → x
        rewriter.replaceOp(op, op.getLhs());
        return success();
    }
};
```

### GreedyPatternRewriteDriver

```cpp
void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SimplifyAddZero>(&getContext());
    patterns.add<SimplifyMulOne>(&getContext());
    // 添加更多 pattern...

    // 贪心重写引擎：持续应用 pattern 直到没有变化
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
        return signalPassFailure();
}
```

## 10.4 Pattern Rewriter API

```cpp
// rewriter 是修改 IR 的唯一合法方式
// 绝对不能直接修改 Operation 的操作数！

// 替换 Op
rewriter.replaceOp(op, newValue);                // 用新值替换 op 的所有使用
rewriter.replaceOpWithNewOp<NewOp>(op, args...); // 创建新 Op 并替换

// 创建新 Op
auto newOp = rewriter.create<arith::AddFOp>(loc, lhs, rhs);

// 插入点控制
rewriter.setInsertionPoint(anotherOp);           // 在 anotherOp 之前插入
rewriter.setInsertionPointToStart(block);         // 在 block 开头
rewriter.setInsertionPointToEnd(block);           // 在 block 末尾

// 删除
rewriter.eraseOp(op);                            // 删除 Op

// 内联 Region
rewriter.inlineRegionBefore(region, destBlock);
```

## 10.5 Walk — IR 遍历

```cpp
// walk 有多种形式

// 形式 1：遍历所有操作
moduleOp.walk([&](Operation *op) {
    if (isa<tt::LoadOp>(op)) {
        // 处理
    }
    return WalkResult::advance();  // 继续遍历
    // 或 WalkResult::interrupt() 提前终止
});

// 形式 2：按类型过滤（更简洁）
moduleOp.walk([&](tt::LoadOp loadOp) {
    // 只遍历 tt::LoadOp
});

// 形式 3：按类型过滤多个
moduleOp.walk([&](Operation *op) {
    return TypeSwitch<Operation *>(op)
        .Case<tt::LoadOp>([&](auto loadOp) { ... })
        .Case<tt::StoreOp>([&](auto storeOp) { ... })
        .Default([](auto) { return WalkResult::advance(); });
});
```

## 10.6 Op 操作实用方法

```cpp
// 获取 Op 的前驱和后继（基本块级别）
Operation *prev = op->getPrevNode();  // 同一个 block 的前一个 Op
Operation *next = op->getNextNode();  // 同一个 block 的后一个 Op

// Block 操作
Block *block = op->getBlock();
Block *parent = block->getParentOp()->getBlock();  // 父 Block

// Region 操作
Region *region = op->getParentRegion();
Operation *parentOp = region->getParentOp();        // 包含 Region 的 Op

// 结果和操作数
ValueRange results = op->getResults();
unsigned numResults = op->getNumResults();
Value firstResult = op->getResult(0);

ValueRange operands = op->getOperands();
Value firstOperand = op->getOperand(0);
```

## 10.7 属性操作

```cpp
// 获取属性
IntegerAttr attr = op->getAttrOfType<IntegerAttr>("num_warps");
if (attr) {
    int value = attr.getInt();
}

// 设置属性
op->setAttr("my_attr", builder.getI32IntegerAttr(42));

// 删除属性
op->removeAttr("my_attr");

// 遍历所有属性
for (auto namedAttr : op->getAttrs()) {
    StringRef name = namedAttr.getName();
    Attribute attr = namedAttr.getValue();
}
```

## 10.8 分析（Analysis）Pass

除了转换 Pass，MLIR 也有分析 Pass——只读取不修改 IR：

```cpp
struct CountInstructionsPass : public PassWrapper<CountInstructionsPass,
                                                   OperationPass<ModuleOp>> {
    // 返回统计信息
    void runOnOperation() override {
        int count = 0;
        getOperation()->walk([&](Operation *op) { count++; });
        llvm::errs() << "Total operations: " << count << "\n";
    }
};
```

分析结果可以通过 `getAnalysis<>()` 在其他 Pass 中使用。

## 10.9 调试 Pass

```bash
# IR 转储（每次 Pass 前后打印 IR）
mlir-opt input.mlir --mlir-print-ir-after-all

# 只打印特定 Pass 之后的 IR
mlir-opt input.mlir --mlir-print-ir-after=canonicalize

# 统计 Op 使用情况
mlir-opt input.mlir --print-op-stats
```

## 10.10 Triton 中的 Pass 示例

```cpp
// TritonGPU CoalescePass（简化）
// lib/Dialect/TritonGPU/Transforms/Coalesce.cpp

struct CoalescePass : public impl::TritonGPUCoalesceBase<CoalescePass> {
    void runOnOperation() override {
        ModuleOp moduleOp = getOperation();
        ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
        
        // 1. 分析每个 load/store 的轴信息
        // 2. 选择最佳布局
        // 3. 设置编码属性
        
        moduleOp.walk([&](Operation *op) {
            if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
                pickLayoutForOp(loadOp, axisInfoAnalysis);
            }
        });
    }
};
```

---

## 📝 课后作业

### 作业 1：写一个简单 Pass

写一个 MLIR Pass，将所有 `arith.addi` 替换为 `arith.subi`（注意：这只是练习，实际中不要这样破坏代码！）：

```cpp
struct AddToSubPass : public OpRewritePattern<arith::AddIOp> {
    // ...
};
```

提示：`rewriter.replaceOpWithNewOp<arith::SubIOp>(op, ...)`。

### 作业 2：读 Triton 的 Coalesce Pass

阅读 `lib/Dialect/TritonGPU/Transforms/Coalesce.cpp`，回答：
1. `CoalescePass` 继承自哪个基类？
2. `runOnOperation()` 中做了哪些分析？
3. `pickDescriptorLoadStoreLayout` 函数的作用是什么？

### 作业 3：运行 triton-opt 调试

找一个 `.ttgir` 测试文件（在 `test/TritonGPU/` 下），运行：

```bash
triton-opt --mlir-print-ir-after-all input.ttgir 2>&1 | less
```

观察每个 Pass 执行前后 IR 的变化。

---

## 本章小结

- MLIR Pass 是 LLVM Pass 的泛化——可以处理任意 Operation
- Pattern Rewrite（`OpRewritePattern`）是 IR 转换的核心机制
- `PatternRewriter` 是修改 IR 的唯一合法途径
- GreedyPatternRewriteEngine 持续应用 pattern 直到收敛
- Walk 是遍历 IR 最便捷的方式
- Triton 中的 Pass（如 CoalescePass）遵循完全相同的模式
