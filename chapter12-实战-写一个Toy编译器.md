# 第 12 章：🏆 实战：写一个 Toy 编译器

> **本章目标**：综合运用前 11 章的知识，从零实现一个完整的迷你语言编译器。
>
> 驯龙手记：Toy 编译器是一场"驯龙模拟演习"——你将在受控环境中
> 实践学到的所有驯龙技巧：定义龙语（Dialect）、指挥龙的行动（Pass）、
> 让龙变形（Dialect Conversion）。这次演习会为你驯服真正的 Triton 做好充分准备。
> 这是第三部分的结业项目。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter12/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `toy_frontend.py` | 12.x | Toy 前端原型 |
| `test.toy.mlir` | 12.8 | 等价 MLIR |

运行：

```bash
cd books/examples/chapter12
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 12.1 Toy 语言规范

我们将在本章实现一个极简的 Toy 语言——它能做标量算术运算。

### 语法（Python 风格）

```python
# Toy 程序示例
def square(x):
    return x * x

def main():
    a = 3.0
    b = 4.0
    c = square(a) + square(b)
    print(c)
```

### 支持的语法结构

| 结构 | 示例 |
|------|------|
| 函数定义 | `def name(params):` |
| 变量赋值 | `x = expr` |
| 二元运算 | `x + y`, `x * y` |
| 函数调用 | `foo(x, y)` |
| 数字字面量 | `3.14`, `42.0` |
| 返回 | `return expr` |

仅支持 `float` 类型。

## 12.2 项目结构

```
toy-compiler/
├── CMakeLists.txt           # 顶层 CMake
├── include/
│   └── toy/
│       ├── IR/
│       │   ├── ToyDialect.td       # Dialect 定义
│       │   ├── ToyOps.td          # Op 定义
│       │   ├── ToyDialect.h
│       │   └── ToyOps.h
│       └── Transforms/
│           ├── Passes.td           # Pass 定义
│           └── Passes.h
├── lib/
│   ├── IR/
│   │   ├── CMakeLists.txt
│   │   ├── Dialect.cpp
│   │   └── Ops.cpp
│   ├── Transforms/
│   │   ├── CMakeLists.txt
│   │   └── ToyToStandard.cpp       # Toy → 标准 Dialect 转换
│   └── CMakeLists.txt
├── toy-compiler.cpp         # 主入口
└── test.toy                 # 测试文件
```

## 12.3 第一步：定义 Dialect 和 Op

### ToyDialect.td

```tablegen
include "mlir/IR/OpBase.td"

def Toy_Dialect : Dialect {
    let name = "toy";
    let cppNamespace = "::mlir::toy";
}

class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits> {
}
```

### ToyOps.td

```tablegen
include "toy/IR/ToyDialect.td"

// 常量
def Toy_ConstantOp : Toy_Op<"constant", [Pure]> {
    let arguments = (ins F64Attr:$value);
    let results = (outs F64:$result);
    let assemblyFormat = "$value attr-dict `:` type($result)";
}

// 加法
def Toy_AddOp : Toy_Op<"add", [Pure, SameOperandsAndResultType]> {
    let arguments = (ins F64:$lhs, F64:$rhs);
    let results = (outs F64:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// 乘法
def Toy_MulOp : Toy_Op<"mul", [Pure, SameOperandsAndResultType]> {
    let arguments = (ins F64:$lhs, F64:$rhs);
    let results = (outs F64:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// 函数调用
def Toy_CallOp : Toy_Op<"call", [Pure]> {
    let arguments = (ins
        FlatSymbolRefAttr:$callee,
        Variadic<F64>:$args
    );
    let results = (outs Variadic<F64>:$results);
}

// 打印
def Toy_PrintOp : Toy_Op<"print"> {
    let arguments = (ins F64:$input);
}

// 函数定义
def Toy_FuncOp : Toy_Op<"func", [
    IsolatedFromAbove,
    Symbol,
    FunctionOpInterface,
    AutomaticAllocationScope
]> {
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type
    );
    let regions = (region AnyRegion:$body);
    let assemblyFormat = [{
        $sym_name $function_type attr-dict $body
    }];
    let hasVerifier = 1;
}

// 返回
def Toy_ReturnOp : Toy_Op<"return", [Pure, Terminator, HasParent<"FuncOp">]> {
    let arguments = (ins Optional<F64>:$value);
    let assemblyFormat = "attr-dict ($value^ `:` type($value))?";
}
```

## 12.4 第二步：实现前端（AST → Toy IR）

```python
# toy_frontend.py — 简化版 Python 前端
# 实际工程中，前端应该在 C++ 中实现
# 这里用 Python 做快速原型

import ast
import sys

class ToyCodeGen(ast.NodeVisitor):
    def __init__(self):
        self.ops = []  # 收集 IR 指令
    
    def visit_FunctionDef(self, node):
        name = node.name
        params = [arg.arg for arg in node.args.args]
        self.ops.append(f"toy.func @{name}(%{', %'.join(params)}) -> f64 {{")
        for stmt in node.body:
            self.visit(stmt)
        self.ops.append("}")
    
    def visit_BinOp(self, node):
        lhs = self.visit(node.left)
        rhs = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            result = f"%{len(self.ops)} = toy.add {lhs}, {rhs} : f64"
        elif isinstance(node.op, ast.Mult):
            result = f"%{len(self.ops)} = toy.mul {lhs}, {rhs} : f64"
        self.ops.append(result)
        return result.split(" ")[0]  # 返回结果名
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            name = node.func.id
            args = [self.visit(arg) for arg in node.args]
            result = f"%{len(self.ops)} = toy.call @{name}({', '.join(args)}) : f64"
            self.ops.append(result)
            return result.split(" ")[0]
    
    def visit_Constant(self, node):
        result = f"%{len(self.ops)} = toy.constant {node.value} : f64"
        self.ops.append(result)
        return result.split(" ")[0]
    
    def visit_Return(self, node):
        if node.value:
            val = self.visit(node.value)
            self.ops.append(f"toy.return {val} : f64")
        else:
            self.ops.append("toy.return")
    
    def generate(self, source):
        tree = ast.parse(source)
        self.visit(tree)
        return "\n    ".join(self.ops)

# 测试
source = '''
def square(x):
    return x * x

def main():
    a = 3.0
    b = 4.0
    c = square(a) + square(b)
    print(c)
'''

gen = ToyCodeGen()
mlir = gen.generate(source)
print(mlir)
```

## 12.5 第三步：实现 Toy → Standard 转换

```cpp
// lib/Transforms/ToyToStandard.cpp

// Toy Op → arith Op 转换
// toy.add → arith.addf
// toy.mul → arith.mulf
// toy.constant → arith.constant

struct AddOpConversion : public OpConversionPattern<toy::AddOp> {
    using OpConversionPattern::OpConversionPattern;

    LogicalResult
    matchAndRewrite(toy::AddOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<arith::AddFOp>(
            op, adaptor.getLhs(), adaptor.getRhs());
        return success();
    }
};

struct ConstantOpConversion : public OpConversionPattern<toy::ConstantOp> {
    LogicalResult
    matchAndRewrite(toy::ConstantOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(
            op, adaptor.getValue());
        return success();
    }
};

// 内联优化：将 toy.call 替换为函数体
struct InlinePass : public OperationPass<ModuleOp> {
    void runOnOperation() override {
        ModuleOp module = getOperation();
        // 收集所有函数
        DenseMap<StringRef, FuncOp> funcs;
        module.walk([&](FuncOp func) {
            funcs[func.getName()] = func;
        });
        
        // 将 toy.call 替换为函数体
        module.walk([&](CallOp call) {
            FuncOp callee = funcs[call.getCallee()];
            // ...内联逻辑...
        });
    }
};
```

## 12.6 第四步：CMake 构建

```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.20)
project(ToyCompiler)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

add_subdirectory(lib)

add_executable(toy-compiler toy-compiler.cpp)
target_link_libraries(toy-compiler
    ToyIR
    ToyTransforms
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRConversion
)
```

## 12.7 第五步：主入口

```cpp
// toy-compiler.cpp
int main(int argc, char **argv) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::toy::ToyDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::LLVM::LLVMDialect>();

    mlir::MLIRContext context(registry);
    context.loadDialect<mlir::toy::ToyDialect>();

    // 解析输入
    auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
    if (!module) return 1;

    // 运行转换
    mlir::PassManager pm(&context);
    pm.addPass(std::make_unique<mlir::toy::InlinePass>());
    pm.addPass(mlir::createConvertToyToStandardPass());
    pm.addPass(mlir::createConvertStandardToLLVMPass());

    if (failed(pm.run(*module))) return 1;

    // 输出 LLVM IR
    module->dump();
    return 0;
}
```

## 12.8 运行测试

```bash
# 1. 编译 Toy 编译器
mkdir build && cd build
cmake .. && make -j$(nproc)

# 2. 创建测试文件
cat > test.toy.mlir << 'EOF'
module {
    toy.func @square(%x: f64) -> f64 {
        %0 = toy.mul %x, %x : f64
        toy.return %0 : f64
    }
    toy.func @main() -> f64 {
        %0 = toy.constant 3.0 : f64
        %1 = toy.constant 4.0 : f64
        %2 = toy.call @square(%0) : f64
        %3 = toy.call @square(%1) : f64
        %4 = toy.add %2, %3 : f64
        toy.return %4 : f64
    }
}
EOF

# 3. 编译
./toy-compiler test.toy.mlir

# 4. 或者用 mlir-opt 手动调试每一步
mlir-opt test.toy.mlir --convert-toy-to-standard
```

## 12.9 扩展方向

完成基础 Toy 编译器后，可以尝试扩展：

| 扩展方向 | 涉及知识 | 挑战度 |
|---------|---------|--------|
| 增加 `if/else` | Region、scf 转换 | ⭐⭐⭐ |
| 增加循环 `for` | LoopPass | ⭐⭐⭐ |
| 增加张量类型 | 类型系统扩展 | ⭐⭐⭐⭐ |
| 增加 GPU 后端 | Triton 风格后端 | ⭐⭐⭐⭐⭐ |
| 优化 Pass | 模式匹配 | ⭐⭐⭐ |

---

## 📝 课后作业

### 作业 1：完成 Toy 编译器

按照本章的步骤和代码框架，在本地完成 Toy 编译器。要求：

1. 能编译 `test.toy.mlir` 到 LLVM IR
2. 使用 `lli` 或 `clang` 运行生成的 LLVM IR

### 作业 2：增加新特性

为 Toy 语言增加 `if` 表达式的支持：

```python
def max(a, b):
    if a > b:
        return a
    else:
        return b
```

需要：
1. 在 ToyOps.td 中定义新的 Op
2. 在转换中添加对应的转换模式
3. 测试编译结果

### 作业 3：思考

将 Toy 语言扩展为**张量语言**（类似 Triton）需要哪些修改？列出至少 5 个需要改变的地方。

---

## 本章小结

- 一个完整的编译器 = 前端（AST → IR）+ 中端（IR Pass）+ 后端（IR → 目标代码）
- MLIR 让"定义自己的 Dialect"变得简单——只需要 .td 文件 + 转换 Pass
- 方言转换（Toy → Standard）是连接自定义 Dialect 和 MLIR 内置 Dialect 的桥梁
- 完成本章的 Toy 编译器是理解 Triton 编译流程的最好准备
- 从 Toy 到 Triton 的跨越：类型系统（标量→张量）、并行模型、GPU 特有编码
