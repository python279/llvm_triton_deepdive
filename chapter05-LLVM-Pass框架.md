# 第 5 章：LLVM Pass 框架

> **本章目标**：理解 LLVM Pass 的工作原理，能编写简单的分析 Pass 和转换 Pass。

> 驯龙手记：Pass 是"龙的气息流动方向"。分析 Pass 负责感知龙的状态
> （心跳、体温、呼吸频率），转换 Pass 则是向龙发出指令（"抬左腿"、"转身"）。
> 最有经验的驯龙者能同时感知多条信息并协调龙的全身动作。

## 配套示例

本章可运行代码位于 `books/examples/chapter05/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `CMakeLists.txt` | 配套 | 构建所有 Pass 插件 |
| `CountInstructionsPass.cpp` | 5.2 | FunctionPass |
| `CountFunctionsPass.cpp` | 5.2 | ModulePass |
| `MyPass.cpp` | 5.3 A | New PM 插件 |
| `RunPipelineTool.cpp` | 5.3 B | 内置 Pipeline |
| `InsertPrintfPass.cpp` | 5.6 | 插入 printf |
| `CountAddPass.cpp` | 作业1 | 统计 add |
| `AddToSubPass.cpp` | 作业2 | add→sub |
| `test.ll` | 5.7 | 测试 IR |

运行：

```bash
cd books/examples/chapter05
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 5.1 什么是 Pass

Pass = 在 IR 上运行的**分析**或**转换**单元。

```
分析 Pass：只读 IR，收集信息，不修改
  → 例如：统计函数个数、分析变量活跃性

转换 Pass：读取并修改 IR
  → 例如：常量传播、死代码消除、循环展开
```

## 5.2 Pass 类型

| Pass 类型 | 作用域 | 用途 |
|-----------|--------|------|
| `ModulePass` | 整个模块 | 跨函数分析、全局优化 |
| `CallGraphSCCPass` | 调用图 SCC | 内联分析 |
| `FunctionPass` | 单个函数 | 函数内优化（最常用） |
| `LoopPass` | 单个循环 | 循环优化 |

### FunctionPass 示例

完整源码：`books/examples/chapter05/CountInstructionsPass.cpp`

```cpp
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

void countInstructions(Function &F) {
    int count = 0;
    for (BasicBlock &BB : F)
        for (Instruction &I : BB)
            ++count;

    errs() << "Function " << F.getName() << " has " << count
           << " instructions\n";
}

// New PM（LLVM 18+ 的 opt 使用此路径）
struct CountInstructionsPass : public PassInfoMixin<CountInstructionsPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        countInstructions(F);
        return PreservedAnalyses::all();
    }
};

// Legacy PM（LLVM 17 及更早的 opt -load -count-instructions）
struct CountInstructionsLegacyPass : public FunctionPass {
    static char ID;
    CountInstructionsLegacyPass() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override {
        countInstructions(F);
        return false;
    }
};

} // namespace

char CountInstructionsLegacyPass::ID = 0;

static RegisterPass<CountInstructionsLegacyPass> X(
    "count-instructions", "Count instructions in functions");

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "count-instructions", LLVM_VERSION_STRING,
            [](PassBuilder &PB) {
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, FunctionPassManager &FPM,
                       ArrayRef<PassBuilder::PipelineElement>) {
                        if (Name == "count-instructions") {
                            FPM.addPass(CountInstructionsPass());
                            return true;
                        }
                        return false;
                    });
            }};
}
```

运行（New PM）：

```bash
opt -load-pass-plugin ./build/CountInstructionsPass.so \
    -passes=count-instructions test.ll -disable-output
# Function add has 2 instructions
# Function dead has 1 instructions
```

### ModulePass 示例

完整源码：`books/examples/chapter05/CountFunctionsPass.cpp`

```cpp
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

void countFunctions(Module &M) {
    int count = 0;
    for (Function &F : M) {
        if (!F.isDeclaration())
            ++count;
    }
    errs() << "Module has " << count << " defined functions\n";
}

struct CountFunctionsPass : public PassInfoMixin<CountFunctionsPass> {
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
        countFunctions(M);
        return PreservedAnalyses::all();
    }
};

struct CountFunctionsLegacyPass : public ModulePass {
    static char ID;
    CountFunctionsLegacyPass() : ModulePass(ID) {}

    bool runOnModule(Module &M) override {
        countFunctions(M);
        return false;
    }
};

} // namespace

char CountFunctionsLegacyPass::ID = 0;

static RegisterPass<CountFunctionsLegacyPass> X(
    "count-functions", "Count defined functions in a module");

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "count-functions", LLVM_VERSION_STRING,
            [](PassBuilder &PB) {
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, ModulePassManager &MPM,
                       ArrayRef<PassBuilder::PipelineElement>) {
                        if (Name == "count-functions") {
                            MPM.addPass(CountFunctionsPass());
                            return true;
                        }
                        return false;
                    });
            }};
}
```

运行：

```bash
opt -load-pass-plugin ./build/CountFunctionsPass.so \
    -passes=count-functions test.ll -disable-output
# Module has 2 defined functions
```

## 5.3 Pass 的注册与运行

### 传统注册方式（legacy PM）

```cpp
char CountInstructionsLegacyPass::ID = 0;
static RegisterPass<CountInstructionsLegacyPass> X(
    "count-instructions", "Count instructions in functions");
```

> Legacy PM 在 LLVM 18+ 的 `opt` 中已移除，仅作概念参考。示例中 Legacy 类命名为 `*LegacyPass`，与 New PM 的 `PassInfoMixin` 子类区分。

### 新 Pass Manager 方式（New PM）

```cpp
// 1. 定义 Pass
struct MyPass : public PassInfoMixin<MyPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        // ...处理逻辑...
        return PreservedAnalyses::all();  // 所有分析结果保持有效
    }
};

// 2. 注册
// 在 Pass 注册文件中添加
```

New PM 的注册方式有两种常见场景：

1. **作为 `opt` 可加载插件**：提供 `llvmGetPassPluginInfo()`，让 `opt -load-pass-plugin` 能发现这个 Pass。
2. **作为项目内置 Pass**：在自己的优化工具或编译器驱动里，直接把 Pass 加进 `FunctionPassManager` / `ModulePassManager`。

### 方式 A：注册成 `opt` 插件

完整源码：`books/examples/chapter05/MyPass.cpp`

```cpp
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct MyPass : public PassInfoMixin<MyPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        errs() << "Running MyPass on " << F.getName() << "\n";
        return PreservedAnalyses::all();
    }
};

} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "my-pass-plugin", LLVM_VERSION_STRING,
            [](PassBuilder &PB) {
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, FunctionPassManager &FPM,
                       ArrayRef<PassBuilder::PipelineElement>) {
                        if (Name == "mypass") {
                            FPM.addPass(MyPass());
                            return true;
                        }
                        return false;
                    });
            }};
}
```

关键点：

- `llvmGetPassPluginInfo()` 是插件入口，`opt` 会通过这个符号加载注册信息。
- `registerPipelineParsingCallback` 负责把 `-passes=mypass` 这个字符串映射到 `FPM.addPass(MyPass())`。
- 如果是 `ModulePass`，把回调参数从 `FunctionPassManager &FPM` 换成 `ModulePassManager &MPM`，并调用 `MPM.addPass(MyModulePass())`。

运行方式：

```bash
opt -load-pass-plugin ./build/MyPass.so -passes=mypass test.ll -disable-output
```

### 方式 B：注册到项目自己的 Pass Pipeline

完整源码：`books/examples/chapter05/RunPipelineTool.cpp`

```cpp
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct MyPass : public PassInfoMixin<MyPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        errs() << "Running MyPass on " << F.getName() << "\n";
        return PreservedAnalyses::all();
    }
};

void runPipeline(Module &M) {
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    PassBuilder PB;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    FunctionPassManager FPM;
    FPM.addPass(MyPass());

    ModulePassManager MPM;
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    MPM.run(M, MAM);
}

} // namespace

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input .ll file>"),
                                          cl::Required);

int main(int argc, char **argv) {
    cl::ParseCommandLineOptions(argc, argv, "chapter05 RunPipeline example\n");

    LLVMContext context;
    SMDiagnostic err;
    std::unique_ptr<Module> M = parseIRFile(InputFilename, err, context);
    if (!M) {
        err.print(argv[0], errs());
        return 1;
    }

    runPipeline(*M);
    return 0;
}
```

运行：

```bash
./build/run_pipeline test.ll
```

Triton 代码库里的 `bin/triton-llvm-opt.cpp` 就采用了类似思路：创建 `PassBuilder`、注册各级 analysis manager，然后把函数级 Pass 通过 `createModuleToFunctionPassAdaptor` 加进 `ModulePassManager`。

### 命令行运行

```bash
# New PM（LLVM 18+ 推荐）
opt -load-pass-plugin ./build/MyPass.so -passes=mypass test.ll -disable-output

# 查看内置 Pass 列表
opt --print-passes | head
```

## 5.4 IR 遍历 API

以下 API 在示例代码中均有使用，例如 `CountInstructionsPass.cpp` 遍历基本块与指令，`CountAddPass.cpp` 用 `dyn_cast<BinaryOperator>` 识别 add 指令。

### 遍历函数

```cpp
// 遍历模块中的所有函数
for (Function &F : M) {
    // F.getName()  — 函数名
    // F.arg_size() — 参数个数
    // F.isDeclaration() — 是否有函数体
}
```

### 遍历基本块

```cpp
for (BasicBlock &BB : F) {
    // BB.getName()  — 基本块名
    // BB.size()     — 指令数
    // BB.getTerminator() — 最后一条指令（分支）
}
```

### 遍历指令

```cpp
for (Instruction &I : BB) {
    // I.getOpcode()  — 指令操作码（如 Instruction::Add）
    // I.getNumOperands() — 操作数个数
    // I.getType() — 结果类型

    // 使用 dyn_cast 检查指令类型（见 CountAddPass.cpp）
    if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
        if (BO->getOpcode() == Instruction::Add) {
            // 这是一个 add 指令
        }
    }
}
```

### 操作指令的操作数

```cpp
// 遍历操作数
for (Use &U : I.operands()) {
    Value *v = U.get();
    // v->getName() — 值名
    // v->getType() — 值类型
}

// 获取特定操作数
Value *lhs = I.getOperand(0);
Value *rhs = I.getOperand(1);

// 修改操作数
I.setOperand(0, newValue);
```

### Use-Def 链

```cpp
// 从 Value 到 User：找出谁用了这个值
for (User *U : val.users()) {
    // U 是使用 val 的指令
}

// 从 User 到 Value：找出这个指令用了哪些值
for (Value *v : I.operands()) {
    // v 是 I 的输入
}
```

## 5.5 分析 Pass 的使用

```cpp
// 新 PM 中通过 AnalysisManager 获取分析结果
PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    // 获取分析结果
    auto &LI = AM.getResult<LoopAnalysis>(F);
    auto &AA = AM.getResult<AAManager>(F);

    // 使用分析结果...
    return PreservedAnalyses::all();
}
```

常用分析 Pass：

| Pass | 提供的信息 |
|------|-----------|
| `LoopAnalysis` | 循环结构 |
| `DominatorTreeAnalysis` | 支配树 |
| `PostDominatorTreeAnalysis` | 后支配树 |
| `AssumptionAnalysis` | 程序假设 |
| `ScalarEvolutionAnalysis` | 标量演化（循环分析） |
| `AAManager` | 别名分析 |

## 5.6 指令创建与 IR 构建

```cpp
// 使用 IRBuilder 创建指令
IRBuilder<> builder(context);

// 1. 设置插入点
builder.SetInsertPoint(/*BasicBlock*/, /*Instruction*/);

// 2. 创建指令
Value *v1 = builder.CreateAdd(lhs, rhs, "add_result");
Value *v2 = builder.CreateLoad(type, ptr, "loaded");
Value *v3 = builder.CreateCall(callee, args, "call_result");
builder.CreateStore(val, ptr);
builder.CreateRet(val);
builder.CreateCondBr(cond, thenBlock, elseBlock);
builder.CreateBr(targetBlock);
```

### 完整示例：在函数入口插入打印

完整源码：`books/examples/chapter05/InsertPrintfPass.cpp`

```cpp
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

bool insertPrintf(Function &F) {
    if (F.isDeclaration())
        return false;

    BasicBlock &entry = F.getEntryBlock();
    IRBuilder<> builder(&entry, entry.begin());

    Module *M = F.getParent();
    FunctionCallee printfFunc = M->getOrInsertFunction(
        "printf",
        FunctionType::get(
            IntegerType::getInt32Ty(M->getContext()),
            PointerType::get(M->getContext(), 0),
            true));

    Value *formatStr =
        builder.CreateGlobalString("Entering function: %s\n");
    builder.CreateCall(printfFunc,
                       {formatStr,
                        builder.CreateGlobalString(F.getName())});
    return true;
}

struct InsertPrintfPass : public PassInfoMixin<InsertPrintfPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        return insertPrintf(F) ? PreservedAnalyses::none()
                               : PreservedAnalyses::all();
    }
};

struct InsertPrintfLegacyPass : public FunctionPass {
    static char ID;
    InsertPrintfLegacyPass() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override { return insertPrintf(F); }
};

} // namespace

char InsertPrintfLegacyPass::ID = 0;

static RegisterPass<InsertPrintfLegacyPass> X(
    "insert-printf", "Insert printf at function entry");

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "insert-printf", LLVM_VERSION_STRING,
            [](PassBuilder &PB) {
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, FunctionPassManager &FPM,
                       ArrayRef<PassBuilder::PipelineElement>) {
                        if (Name == "insert-printf") {
                            FPM.addPass(InsertPrintfPass());
                            return true;
                        }
                        return false;
                    });
            }};
}
```

运行：

```bash
opt -load-pass-plugin ./build/InsertPrintfPass.so \
    -passes=insert-printf -S test.ll -o test_with_printf.ll
```

## 5.7 `opt` 命令行实战

测试 IR：`books/examples/chapter05/test.ll`

```bash
# 1. 测试 IR（已提供 test.ll）
cat test.ll
```

```llvm
define i32 @add(i32 %a, i32 %b) {
  %result = add i32 %a, %b
  ret i32 %result
}

define internal i32 @dead() {
  ret i32 0
}
```

```bash
# 2. 删除未使用的 internal 函数（globaldce）
opt -S -passes=globaldce test.ll -o test_cleaned.ll
# internal @dead 会被删除

# 3. 运行内联
opt -S -passes=inline test.ll -o test_inlined.ll

# 4. 运行所有 O2 优化
opt -S -passes='default<O2>' test.ll -o test_O2.ll
```

> `@dead` 需标记为 `internal`，`globaldce` 才会删除未被引用的函数定义。

## 5.8 LLVM Pass 与 MLIR Pass 的对应

| LLVM Pass | MLIR Pass |
|-----------|-----------|
| `FunctionPass` | `OperationPass<FuncOp>` |
| `ModulePass` | `OperationPass<ModuleOp>` |
| `IRBuilder` | `OpBuilder` + `PatternRewriter` |
| `dyn_cast<Instruction>` | `dyn_cast<arith::AddIOp>` |
| `PreservedAnalyses` | 类似但简化（MLIR 自动管理） |
| `AnalysisManager` | `getAnalysis<...>()` |

> **💡 理解**：MLIR 的 Pass 系统是 LLVM Pass 系统的泛化。MLIR 不是"另一个 Pass 框架"，而是 LLVM Pass 框架的多 Dialect 扩展。

---

## 📝 课后作业

### 作业 1：写一个 FunctionPass

写一个 FunctionPass，找出函数中所有 `add` 指令，统计它们的数量，并打印每个函数中的 add 数量。

参考实现：`books/examples/chapter05/CountAddPass.cpp`

```cpp
void countAdds(Function &F) {
    int count = 0;
    for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
            if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
                if (BO->getOpcode() == Instruction::Add)
                    ++count;
            }
        }
    }
    errs() << "Function " << F.getName() << " has " << count
           << " add instructions\n";
}

struct CountAddPass : public PassInfoMixin<CountAddPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        countAdds(F);
        return PreservedAnalyses::all();
    }
};
```

```bash
opt -load-pass-plugin ./build/CountAddPass.so \
    -passes=count-add test.ll -disable-output
# Function add has 1 add instructions
# Function dead has 0 add instructions
```

### 作业 2：写一个转换 Pass

写一个 FunctionPass，将函数中所有 `add` 指令替换为 `sub`（破坏性修改，但有助于理解 IR 修改流程）。

参考实现：`books/examples/chapter05/AddToSubPass.cpp`

```cpp
bool replaceAddWithSub(Function &F) {
    if (F.isDeclaration())
        return false;

    bool changed = false;
    SmallVector<BinaryOperator *, 8> toReplace;

    for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
            if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
                if (BO->getOpcode() == Instruction::Add)
                    toReplace.push_back(BO);
            }
        }
    }

    for (BinaryOperator *BO : toReplace) {
        IRBuilder<> builder(BO);
        Value *sub = builder.CreateSub(BO->getOperand(0), BO->getOperand(1),
                                       BO->getName());
        BO->replaceAllUsesWith(sub);
        BO->eraseFromParent();
        changed = true;
    }

    if (changed)
        errs() << "Replaced add with sub in function " << F.getName() << "\n";
    return changed;
}

struct AddToSubPass : public PassInfoMixin<AddToSubPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        return replaceAddWithSub(F) ? PreservedAnalyses::none()
                                    : PreservedAnalyses::all();
    }
};
```

```bash
opt -load-pass-plugin ./build/AddToSubPass.so \
    -passes=add-to-sub -S test.ll -o test_sub.ll
# %result = add i32 ...  →  %result1 = sub i32 ...
```

### 作业 3：使用 opt 探索

```bash
# 对 test.ll 依次运行以下 Pass，观察输出差异
opt -S -passes=instcombine test.ll -disable-output
opt -S -passes=gvn test.ll -disable-output
opt -S -passes=licm test.ll -disable-output
opt -S -passes=loop-unroll test.ll -disable-output
```

以上命令已包含在 `run_examples.sh` 中。

---

## 本章小结

- Pass 是 LLVM IR 的分析/转换单元，分 Module/Function/Loop 级别
- 函数内以 `Function → BasicBlock → Instruction` 三级结构组织
- `IRBuilder` 是创建和插入指令的主要工具
- Use-Def 链连接变量的使用和定义
- 分析 Pass 提供信息，转换 Pass 修改 IR
- MLIR 的 Pass 系统是 LLVM Pass 系统的泛化
