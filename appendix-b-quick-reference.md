# 附录 B：LLVM/MLIR 速查表

> 开发时快速查询的参考手册。命令需在对应工具已安装/已构建的前提下使用。

---

## B.1 常用命令行

### LLVM

| 命令 | 用途 |
|------|------|
| `clang -S -emit-llvm hello.c -o hello.ll` | C → LLVM IR（文本） |
| `opt -S -passes=instcombine input.ll -o out.ll` | 运行内置 Pass（New PM） |
| `opt -load-pass-plugin ./plugin.so -passes=my-pass input.ll -disable-output` | 加载 Pass 插件 |
| `llc input.ll -o input.s` | LLVM IR → 汇编 |
| `lli input.ll` | JIT 解释执行 LLVM IR（需含 `main`） |
| `llvm-as input.ll -o input.bc` | 文本 IR → 二进制 bitcode |
| `llvm-dis input.bc -o input.ll` | 二进制 bitcode → 文本 IR |
| `llvm-config --version` | 查看 LLVM 版本 |
| `opt --print-passes 2>&1 \| grep -i loop` | 列出名称含 loop 的 Pass |

### MLIR

| 命令 | 用途 |
|------|------|
| `mlir-opt input.mlir --canonicalize -o output.mlir` | 运行 canonicalize 等 Pass |
| `mlir-opt --print-op-stats input.mlir` | 统计 Op 使用 |
| `mlir-opt --mlir-print-ir-after-all input.mlir 2>&1 \| less` | 打印每个 Pass 后的 IR |
| `mlir-opt --mlir-print-ir-after=canonicalize input.mlir` | 指定 Pass 后打印 |
| `mlir-translate input.mlir --mlir-to-llvmir -o out.ll` | 已注册 LLVM 降级的 MLIR → LLVM IR |
| `mlir-tblgen -I$LLVM/include MyOps.td --gen-op-defs -o MyOps.cpp.inc` | TableGen → Op 定义 |
| `mlir-tblgen -I$LLVM/include MyDialect.td --gen-dialect-decls -o MyDialect.h.inc` | TableGen → Dialect 声明 |

> `mlir-tblgen` 需 `-I` 指向 MLIR/LLVM include，且输入 `.td` 文件在前、`-o` 输出在后。

### Triton

| 命令 | 用途 |
|------|------|
| `triton-opt input.ttir --tritongpu-coalesce -o output.ttgir` | 运行 Triton Pass（Pass 名即 CLI 选项） |
| `triton-opt input.ttir --canonicalize -disable-output` | 规范化 TTIR |
| `triton-opt --mlir-print-ir-after-all input.ttgir 2>&1 \| less` | 调试 Pass 流水线 |
| `TRITON_KERNEL_DUMP=1 python kernel.py` | 导出各阶段中间 IR |
| `NVPTX_ENABLE_DUMP=1 python kernel.py` | 打印 PTX（NVIDIA 后端） |
| `TRITON_ALWAYS_COMPILE=1 python kernel.py` | 禁用缓存，强制重新编译 |
| `TRITON_KERNEL_OVERRIDE=1 python kernel.py` | 启用内核覆盖调试 |

**LIT 测试**（在 Triton 构建目录下，先 `ninja triton-opt`）：

```bash
cd $BUILD_DIR
lit -v ../test/TritonGPU/coalesce.mlir
lit -v ../test/TritonGPU/    # 运行整个目录
```

**Python 测试**：

```bash
pytest python/test/unit/language/ -x -s --tb=short
```

---

## B.2 常用 CMake 代码片段

```cmake
# 查找 LLVM/MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# TableGen（与 Triton 源码风格一致）
set(LLVM_TARGET_DEFINITIONS MyOps.td)
mlir_tablegen(MyOps.h.inc -gen-op-decls)
mlir_tablegen(MyOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MyOpsIncGen)

set(LLVM_TARGET_DEFINITIONS MyDialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MyDialectIncGen)

# 添加 Dialect 库
add_mlir_dialect_library(MLIRMyDialect
    MyDialect.cpp
    MyOps.cpp
    DEPENDS
    MyOpsIncGen
    MyDialectIncGen
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)
```

---

## B.3 MLIR TableGen 语法速查

```tablegen
// Dialect 定义
def My_Dialect : Dialect {
    let name = "my";
    let cppNamespace = "::mlir::my";
}

// Op 定义模板
class My_Op<string mnemonic, list<Trait> traits = []> :
    Op<My_Dialect, mnemonic, traits> { }

// 具体 Op
def My_AddOp : My_Op<"add", [Pure, SameOperandsAndResultType]> {
    let summary = "Addition op";
    let description = [{...}];
    let arguments = (ins F32:$lhs, F32:$rhs);
    let results = (outs F32:$result);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
    let hasVerifier = 1;
    let hasFolder = 1;
    let extraClassDeclaration = [{ static bool isSupported(); }];
}

// 类型定义
def My_PtrType : TypeDef<My_Dialect, "Pointer"> {
    let mnemonic = "ptr";
    let parameters = (ins "Type":$pointeeType);
}

// 属性约束
def My_FloatLike : AnyTypeOf<[F32, F64]>;

// Pass 定义（需指定作用域，如 ModuleOp）
def MyPass : Pass<"my-pass", "ModuleOp"> {
    let summary = "Does something";
    let dependentDialects = ["my::MyDialect"];
}

// 属性定义 (AttrDef)
def My_EncodingAttr : AttrDef<My_Dialect, "Encoding"> {
    let mnemonic = "encoding";
    let parameters = (ins
        ArrayRefParameter<"unsigned">:$dims,
        "unsigned":$version
    );
    let assemblyFormat = "`<` $dims `>`";
}
```

---

## B.4 MLIR C++ API 速查

### 创建和操作

```cpp
// 创建
OpBuilder builder(context);
auto op = builder.create<arith::AddFOp>(loc, lhs, rhs);
auto op = builder.create<LLVM::LoadOp>(loc, elemTy, ptr);

// 替换
rewriter.replaceOp(oldOp, newValue);
rewriter.replaceOpWithNewOp<NewOp>(oldOp, args...);

// 删除
rewriter.eraseOp(op);

// 插入点控制
rewriter.setInsertionPoint(otherOp);
rewriter.setInsertionPointToStart(block);
rewriter.setInsertionPointToEnd(block);
```

### IR 遍历

```cpp
// Walk
moduleOp.walk([&](Operation *op) { });
moduleOp.walk([&](tt::LoadOp loadOp) { });

// Block 遍历
for (Operation &op : block) { }
for (auto &op : llvm::reverse(block)) { }  // 反向

// 结果/操作数
Value result = op->getResult(0);
Value operand = op->getOperand(0);
ValueRange results = op->getResults();
```

### 类型检查

```cpp
if (isa<RankedTensorType>(type)) { }
auto tensorType = dyn_cast<RankedTensorType>(type);
auto tensorType = cast<RankedTensorType>(type);  // 失败则 assert
```

### 属性

```cpp
// IntegerAttr
auto attr = op->getAttrOfType<IntegerAttr>("axis");
int64_t val = attr.getInt();

// FloatAttr
auto attr = op->getAttrOfType<FloatAttr>("value");
double val = attr.getValueAsDouble();

// StringAttr
auto attr = op->getAttrOfType<StringAttr>("name");
StringRef val = attr.getValue();

// 设置属性
op->setAttr("name", builder.getI32IntegerAttr(42));
op->setAttr("flag", builder.getBoolAttr(true));
```

---

## B.5 LLVM IR 指令速查

```llvm
; 算术
%r = add i32 %a, %b
%r = fadd float %a, %b

; 比较
%c = icmp sgt i32 %a, %b
%c = fcmp ogt float %a, %b

; 内存
%v = load i32, i32* %ptr
store i32 %val, i32* %ptr
%p = getelementptr i32, i32* %base, i64 %idx

; 类型转换
%t = trunc i32 %v to i8
%e = fpext float %v to double
%b = bitcast i32* %p to i8*
%pp = inttoptr i64 %addr to i32*

; 控制流
br label %block
br i1 %cond, label %t, label %f
ret i32 %val
%z = phi i32 [%v1, %b1], [%v2, %b2]
%sel = select i1 %cond, i32 %t, i32 %f

; 函数调用
%r = call i32 @func(i32 %arg)
```

---

## B.6 Git 操作速查

```bash
# 查看当前工作
git status

# 查看最近提交
git log --oneline -10

# 查看某次提交的改动
git show <commit-hash>

# 查看某个文件的修改历史
git log --oneline -p -- path/to/file.cpp

# 查看某段代码的修改历史（行号范围，最可靠）
git log -L '661,680:lib/Dialect/Triton/IR/Ops.cpp'

# 在当前 repo 搜索代码
git grep 'getSingleCombiner'
```

---

## B.7 代码风格规范

Triton 使用 `clang-format` 统一 C++ 代码风格：

```bash
# 格式化单个文件
clang-format -i path/to/file.cpp

# 在 Triton 根目录运行 pre-commit（含 clang-format）
pre-commit run --all-files

# 查看 Triton 的格式化配置
cat .clang-format   # BasedOnStyle: LLVM
```

Python 格式化：`yapf`（`pyproject.toml` 中 `column_limit = 120`）与 `ruff`（逐步替代 autopep8）。

Python 导入顺序（见 `python/triton/language/__init__.py`）：标准库 → 第三方库 → Triton 内部模块。

命名规范：

- C++ 类名：`PascalCase`（如 `CoalescePass`、`BlockedEncodingAttr`）
- C++ 函数/变量：`camelCase`（如 `getSingleCombiner`、`numWarps`）
- C++ 命名空间：`snake_case`（如 `mlir::triton::gpu`）
- Python 函数：`snake_case`（如 `program_id`、`num_programs`）
- Python 类：`PascalCase`（如 `JITFunction`、`CompiledKernel`）
- TableGen 定义：`PascalCase`（如 `TT_ReduceOp`、`TritonGPU_Dialect`）

---

## B.8 常用环境变量（Triton）

| 变量 | 作用 |
|------|------|
| `TRITON_KERNEL_DUMP` | 导出编译各阶段 IR |
| `TRITON_ALWAYS_COMPILE` | 跳过编译缓存 |
| `TRITON_KERNEL_OVERRIDE` | 内核覆盖/调试 |
| `NVPTX_ENABLE_DUMP` | 打印 PTX（NVIDIA） |
| `AMDGCN_ENABLE_DUMP` | 打印 AMDGCN（AMD） |
| `LLVM_DIR` / `MLIR_DIR` | CMake 查找 LLVM/MLIR |

定义见 `python/triton/knobs.py`。
