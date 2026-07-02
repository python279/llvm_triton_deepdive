# 第 2 章：C++ 编译器开发特训

> **本章目标**：掌握阅读和编写 LLVM/MLIR/Triton C++ 源码所需的核心 C++ 知识。
>
> 📂 **第一部分：编译器基础** — 磨炼刀剑，C++ 和编译原理的基本功
> 这不是完整的 C++ 教程，而是**编译器开发场景的 C++ 速成**。

> 驯龙手记：驯龙需要趁手的兵器。auto 是你挥剑的"惯性"，dyn_cast 是"识别不同龙种"
> 的慧眼，SmallVector 是你随身携带的"工具袋"。本章磨利你的刀剑。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter02/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `llvm_adt_demo.cpp` | 2.x | LLVM SmallVector 示例 |
| `CMakeLists.txt` | 配套 | 构建配置 |

运行：

```bash
cd books/examples/chapter02
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

> 以阅读 Triton 源码中的 C++ 惯用法为主；示例演示 LLVM ADT。

---
## 2.1 你将遇到的 C++ 特性

在 Triton 源码中，以下 C++ 特性出现的频率最高：

| 特性 | 出现频率 | Triton 示例 |
|------|---------|-------------|
| `auto` | 🔥🔥🔥🔥🔥 | `auto tensorType = dyn_cast<...>(type)` |
| 模板 | 🔥🔥🔥🔥 | `template <class Op> struct GenericOpPattern` |
| 虚函数/继承 | 🔥🔥🔥🔥 | `class BaseBackend`, `class CUDABackend` |
| Lambda | 🔥🔥🔥 | `moduleOp.walk([&](...) { ... })` |
| LLVM ADT | 🔥🔥🔥🔥🔥 | `SmallVector`, `ArrayRef`, `DenseMap` |
| `dyn_cast`/`isa` | 🔥🔥🔥🔥🔥 | MLIR 类型系统的核心 |
| `override`/`final` | 🔥🔥🔥 | Pass 重写 |
| 右值引用 | 🔥🔥 | `std::move(patterns)` |

## 2.2 `auto` 类型推导

```cpp
// 基本用法 — 自动推导类型
auto a = 42;            // int
auto b = 3.14f;         // float
const auto &ref = a;    // const int&

// Triton 中的实际使用
auto resultTensorTy = dyn_cast<RankedTensorType>(resultTy);
// 返回类型是 RankedTensorType*，但用 auto 让代码更简洁

auto b = TritonLLVMOpBuilder(loc, rewriter);
// 避免写出复杂的模板类型
```

**🔑 关键规则**：
- `auto` 去掉引用和 const（除非显式写 `const auto&`）
- 优先用 `const auto&` 遍历容器，避免拷贝
- 不要滥用 `auto`：当类型对理解代码至关重要时，显式写出类型

## 2.3 指针与引用：LLVM 的偏好

LLVM 社区的风格和标准 C++ 有些不同：

| 场景 | 推荐写法 | 说明 |
|------|---------|------|
| 函数参数（可空） | `Operation *op` | 裸指针，nullptr 表示"不存在" |
| 函数参数（必填） | `Value value` | 按值传递（Value 是句柄） |
| 函数参数（输出） | `SmallVectorImpl<...> &result` | 非 const 引用 |
| 返回值 | `Value`, `LogicalResult` | 按值返回 |
| 不属于的指针 | `std::unique_ptr<T>` | 独占所有权 |
| 共享所有权 | 很少使用 | LLVM 倾向用裸指针+明确所有权 |

```cpp
// LLVM 风格示例 — 对比标准 C++
// ✅ LLVM 风格
LogicalResult matchAndRewrite(Operation *op,  // 可空指针
                              PatternRewriter &rewriter) const {
    auto val = op->getResult(0);    // Value 是句柄，按值传递
    ...
}

// ❌ 不必要的智能指针
std::shared_ptr<Operation> op;  // 不要这样
```

## 2.4 模板基础

```cpp
// 1. 函数模板 — MLIR 转换模式中最常见
template <class Op>
struct GenericOpPattern : public OpConversionPattern<Op> {
    LogicalResult matchAndRewrite(
        Op op, typename Op::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override {
        // 这里 Op 是模板参数，会自动匹配各种操作
        // 比如 Op = tt::LoadOp, Op = tt::StoreOp, ...
    }
};

// 2. 类模板
template <typename T>
struct Box {
    T value;
    Box(T v) : value(v) {}
    T get() const { return value; }
};

// 使用
Box<int> intBox(42);
Box<float> floatBox(3.14f);

// 3. 模板特化 — 为特定类型定制
template <>
struct GenericOpPattern<arith::ConstantOp> : ... {
    // ConstantOp 需要特殊处理
};
```

## 2.5 Lambda 表达式

```cpp
// 基本语法
[capture](parameters) -> return_type { body }

// Triton 中最常见的使用场景：walk IR
moduleOp.walk([&](Operation *op) {           // [&] 按引用捕获所有外部变量
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
        // 处理 LoadOp
    }
    return WalkResult::advance();
});

// 也可以捕获特定变量
Value target;
moduleOp.walk([&target](Operation *op) {     // 只捕获 target
    if (op->getResult(0) == target) {
        // ...
    }
});
```

## 2.6 LLVM 特有数据结构 — ADT

LLVM 有自己的一套容器和算法，**不是标准 C++**，但在 Triton 中无处不在。

### `SmallVector<T, N>` — 小向量优化

```cpp
// N 个以内的元素分配在栈上，超出才堆分配
SmallVector<unsigned, 4> sizePerThread;  // 4 个以下在栈上
sizePerThread.push_back(1);
sizePerThread.push_back(2);

// 为什么用它而不是 std::vector？
// 编译器开发中大量使用小型向量（2-8 个元素），
// SmallVector 避免了大量堆分配开销

// Triton 实际代码
SmallVector<unsigned> sizePerThread(type.getRank(), 1);
for (unsigned i = 0; i < sizePerThread.size(); ++i) { ... }
```

### `ArrayRef<T>` — 只读数组视图

```cpp
// 类似 std::span — 不拥有数据，只引用
void printValues(ArrayRef<int> values) {
    for (int v : values) {
        llvm::outs() << v << "\n";
    }
}

// 任何连续容器都能隐式转换
SmallVector<int> vec = {1, 2, 3};
int arr[] = {4, 5, 6};
printValues(vec);   // OK
printValues(arr);   // OK
printValues({7, 8, 9});  // OK
```

### `StringRef` — 只读字符串

```cpp
// LLVM 的字符串类型—不拷贝、只引用
StringRef name = "hello world";    // 不分配内存
name.consume_front("hello ");      // name → "world"
name.split(':');                   // 分割

// Triton 中
StringRef mnemonic = op->getName().getStringRef();
```

### `DenseMap<K, V>` — 哈希表

```cpp
DenseMap<Value, int> valueMap;
valueMap[val] = 42;     // 存储
if (valueMap.count(val)) { ... }  // 查找
```

### `DenseSet<T>` — 哈希集合

```cpp
DenseSet<Operation *> visited;
visited.insert(op);
if (visited.contains(op)) { ... }
```

### `MapVector<K, V>` — 有序映射

```cpp
// 按插入顺序迭代的映射
MapVector<StringRef, int> orderedMap;
orderedMap["first"] = 1;
orderedMap["second"] = 2;
// 迭代顺序总是 first → second
```

### 便利算法

```cpp
#include "llvm/ADT/STLExtras.h"

// enumerate — 带索引遍历
for (auto [idx, val] : llvm::enumerate(vec)) {
    // idx 从 0 开始
    // val 是元素的引用
}

// transform — 转换
SmallVector<int> result;
llvm::transform(input, std::back_inserter(result),
                [](int x) { return x * 2; });

// map_keys / map_values
// 提取映射的键或值
```

## 2.7 `dyn_cast`、`isa`、`cast` — LLVM RTTI

这是 MLIR 最常用的模式，**必须掌握**。

```cpp
// isa — 类型检查（类似 Python 的 isinstance）
if (isa<RankedTensorType>(type)) {  // true/false
    // type 是 RankedTensorType 类型
}

// dyn_cast — 类型转换（失败返回 nullptr，类似 Python 的 as）
if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    // 安全地使用 tensorType
    auto shape = tensorType.getShape();
}  // 类型不匹配时 tensorType == nullptr

// cast — 强制类型转换（失败时 assert/崩溃）
auto tensorType = cast<RankedTensorType>(type);
// 只在你确定类型正确时使用——性能最好
```

### Triton 中的实际模式

```cpp
// 模式 1：先检查再使用
if (isa<tt::LoadOp>(op)) {
    auto loadOp = cast<tt::LoadOp>(op);
    // ...
}

// 模式 2：一步到位（推荐）
if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
    // ...
}

// 模式 3：遍历并过滤
moduleOp.walk([&](tt::LoadOp loadOp) {
    // walk 直接帮你过滤好了
});
```

## 2.8 `LogicalResult` 与 `failure()`/`success()`

```cpp
// MLIR 的"可失败返回值"模式（类似 Rust 的 Result）
LogicalResult doSomething(Operation *op) {
    if (!op) return failure();     // 失败
    if (isValid(op)) return success();  // 成功
    return failure();
}

// 在 Pattern Rewrite 中使用
LogicalResult matchAndRewrite(MyOp op, ...) override {
    if (!shouldApply(op)) return failure();   // 不匹配
    // 执行重写...
    return success();                         // 成功应用
}

// 检查调用结果
if (failed(doSomething(op))) {
    // 处理失败
}
if (succeeded(doSomething(op))) {
    // 处理成功
}
```

## 2.9 C++17 特性速览

Triton 使用 C++17，以下特性常见：

```cpp
// 1. 结构化绑定 — 分解元组/pair
SmallVector<std::pair<int, int>> pairs;
for (auto [first, second] : pairs) {
    // first 和 second 是引用
}

// 2. if 初始化
if (auto result = getOptional(); result.has_value()) {
    // result 在此作用域有效
}

// 3. 折叠表达式
template <typename... Args>
void printAll(Args... args) {
    (llvm::outs() << ... << args);
}

// 4. std::optional — 可选值
std::optional<int> getValue() {
    if (condition) return 42;
    return std::nullopt;  // 类似 Python 的 None
}
```

## 2.10 从 Python 到 C++ 思维转换表

```python
# Python 代码
```

```cpp
// 对应的 C++ / LLVM 代码
```

| Python | C++ / LLVM |
|--------|-----------|
| `x is None` | `x == nullptr` |
| `isinstance(x, Type)` | `isa<Type>(x)` |
| `x = list()` | `SmallVector<T> x` |
| `for i, v in enumerate(x):` | `for (auto [i, v] : llvm::enumerate(x))` |
| `d = {}` | `DenseMap<K, V> d` |
| `try/except` | `LogicalResult` + `failure()` |
| `@decorator` | 模板/继承 + 宏 |
| `f"hello {name}"` | `llvm::formatv("hello {0}", name)` |
| `x.y = value` | `x.setY(value)` 或 Builder |
| 鸭子类型 | 模板 + `dyn_cast` |

## 2.11 推荐练习：读一段真实代码

阅读 `lib/Dialect/Triton/IR/Ops.cpp` 中的 `ReduceOp::getSingleCombiner()` 方法（约第 661 行）：

```cpp
::mlir::Operation *ReduceOp::getSingleCombiner() {
  if (getNumOperands() != 1 || getNumResults() != 1)
    return nullptr;
  Block *block = &(*getCombineOp().begin());
  Operation *yield = block->getTerminator();
  Operation *reduceOp = yield->getOperand(0).getDefiningOp();
  if (!reduceOp || reduceOp->getNumOperands() != 2 ||
      reduceOp->getNumResults() != 1)
    return nullptr;
  Value arg0 = block->getArgument(0), arg1 = block->getArgument(1);
  Value lhs = reduceOp->getOperand(0), rhs = reduceOp->getOperand(1);
  bool reversedMapping = (lhs == arg1 && rhs == arg0) &&
                         reduceOp->hasTrait<OpTrait::IsCommutative>();
  if (!(lhs == arg0 && rhs == arg1) && !reversedMapping)
    return nullptr;
  return reduceOp;
}
```

**读代码练习**：逐行解释这段代码在做什么。你已经认识了 `nullptr`、`dyn_cast` 风格（这里用 `Operation *` 的方式）、模板 trait 检查（`hasTrait<...>()`）、以及 `Block`/`Value`/`Operation` 的 LLVM IR 数据结构。

---

## 📝 课后作业

### 作业 1：LLVM ADT 练习

用 LLVM ADT 重写以下 Python 代码：

```python
def process(values):
    result = []
    for i, v in enumerate(values):
        result.append(v * 2)
    return result
```

```cpp
// 你的答案：
SmallVector<int> process(ArrayRef<int> values) {
    SmallVector<int> result;
    // ...
    return result;
}
```

### 作业 2：阅读源码

在 `lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp` 中找到 `CmpIOpConversion` 结构体，回答：
1. 它继承自哪个基类？
2. `matchAndRewrite` 的方法签名是什么？
3. 它如何将 `arith.cmpi` 转换为 LLVM 的 `icmp`？

### 作业 3：dyn_cast 练习

阅读 `lib/Dialect/TritonGPU/Transforms/Coalesce.cpp`，找出至少 3 处使用 `dyn_cast` 或 `isa` 的代码，记录下行号和用途。

---

## 本章小结

- `auto`、`dyn_cast`/`isa`/`cast`、`LogicalResult` 是 MLIR 编程的三大基石
- LLVM ADT（`SmallVector`、`ArrayRef`、`DenseMap`）替代了大部分 STL 容器
- `SmallVector<T, N>` 是编译器开发中最重要的数据结构——栈优先分配避免堆开销
- 模板主要用于抽象 Op 类型，不用过度关注模板元编程
