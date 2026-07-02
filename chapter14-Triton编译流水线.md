# 第 14 章：Triton 编译流水线

> **本章目标**：理解 `compile()` 函数的完整流程，掌握调试工具。
>
> 驯龙手记：编译流水线就是"海神 Triton 的成长轨迹"——从它还是一颗蛋
> （Python AST）开始，经历幼龙（TTIR）、少年（TTGIR）、成年（LLVM IR）、
> 直到长成翱翔于 GPU 天空的巨龙（CUBIN）。每一步成长都需要特定的"喂养"（Pass）。
> 第 15-19 章会深入每一层的详细内容。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter14/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `simple_kernel.py` | 14.x | 最小 JIT 内核（需 GPU） |
| `fixtures/simple.ttir` | 14.x | 静态 TTIR fixture |

运行：

```bash
cd books/examples/chapter14
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 14.1 编译流水线总览

```python
# 一条语句触发整个编译流程
kernel[grid](x, y, output, n, BLOCK_SIZE=1024)
```

```
用户调用 --> JITFunction.__call__()
             ├── 参数绑定和类型推断
             ├── 检查编译缓存（~/.triton/cache/<hash>/）
             │   ├── 命中 → 返回 CompiledKernel
             │   └── 未命中 → 调用 compile()
             └── 启动内核
```

### 编译管道的五个阶段

```
triton.compiler.compile():

Stage 1: ASTSource
  src.make_ir() → code_generator.py → TT IR

Stage 2: make_ttir()
  内联、规范化、CSE → 优化后的 TT IR

Stage 3: make_ttgir() [最复杂]
  TT→TTG转换(加GPU编码) → Coalesce → AccelerateMatmul → Pipeline

Stage 4: make_llir()
  TTG→LLVM Dialect降级 + LLVM优化

Stage 5: make_ptx() + make_cubin()
  LLVM→PTX + ptxas→CUBIN
```

## 14.2 compile() 源码分析

定义在 `python/triton/compiler/compiler.py` 中。

### 函数签名

```python
def compile(src, target=None, options=None, _env_vars=None):
    """
    src: ASTSource（Python AST）或 IRSource（.ttir 等文件）
    target: GPUTarget（如 backend='cuda', arch=90）
    options: 编译选项（num_warps, num_stages 等）
    """
```

### 核心流程

```python
def compile(src, target=None, options=None):
    # 1. 确定目标
    if target is None:
        target = driver.active.get_current_target()  # 当前 GPU 信息
    backend = make_backend(target)                    # 创建后端实例

    # 2. 解析源和选项
    extra_options = src.parse_options()
    options = backend.parse_options(dict(options, **extra_options))

    # 3. 检查缓存
    key = get_cache_key(src, backend, options, env_vars=env_vars)
    # 缓存命中 → 直接返回编译结果

    # 4. 注册编译阶段
    stages = dict()
    backend.add_stages(stages, options, src.language)
    # stages = {"ttir": make_ttir, "ttgir": make_ttgir,
    #            "llir": make_llir, "ptx": make_ptx, "cubin": make_cubin}

    # 5. 运行所有阶段
    for ext, compile_ir in list(stages.items())[first_stage:]:
        module = compile_ir(module, metadata)
        # 缓存每个阶段的中间 IR

    # 6. 返回 CompiledKernel
    return CompiledKernel(src, metadata_group, hash)
```

### make_backend 函数

```python
def make_backend(target: GPUTarget) -> BaseBackend:
    # 在已注册后端的 compiler 列表中查找匹配的
    actives = [x.compiler for x in backends.values()
               if x.compiler.supports_target(target)]
    # target.backend == 'cuda' → CUDA backend
    # target.backend == 'hip' → AMD HIP backend
    return actives[0](target)
```

## 14.3 ASTSource 与 IRSource

```python
# ASTSource — 从 Python AST 编译
class ASTSource:
    def __init__(self, fn, signature, constexprs=None, attrs=None):
        self.fn = fn              # JITFunction
        self.signature = signature # 参数类型

    def make_ir(self, target, options, codegen_fns, module_map, context):
        from .code_generator import ast_to_ttir
        return ast_to_ttir(self.fn, self, context=context)

# IRSource — 从 MLIR 文件编译
class IRSource:
    def __init__(self, path, context, backend):
        self.module = ir.parse_mlir_module(path, context)
        # 文件可以是 .ttir, .ttgir, .llir, .ptx
```

## 14.4 Backend 抽象基类

```python
# python/triton/backends/compiler.py
class BaseBackend:
    @abstractmethod
    def add_stages(self, stages, options, language):
        """注册编译阶段"""

    @abstractmethod
    def load_dialects(self, context):
        """加载需要使用的 MLIR Dialect"""

    @staticmethod
    @abstractmethod
    def supports_target(target):
        """判断是否支持这个 GPU 目标"""

    def hash(self):
        """后端唯一标识（用于缓存）"""
```

## 14.5 CUDABackend 的 Stage 注册

```python
# third_party/nvidia/backend/compiler.py
class CUDABackend(BaseBackend):
    def add_stages(self, stages, options, language):
        if language == Language.TRITON:
            stages["ttir"]   = lambda src, m: self.make_ttir(src, m, ...)
            stages["ttgir"]  = lambda src, m: self.make_ttgir(src, m, ...)
        stages["llir"]   = lambda src, m: self.make_llir(src, m, ...)
        stages["ptx"]    = lambda src, m: self.make_ptx(src, m, ...)
        stages["cubin"]  = lambda src, m: self.make_cubin(src, m, ...)
```

## 14.6 缓存机制

```text
# python/triton/runtime/cache.py

# 缓存位置：~/.triton/cache/<sha256_hash>/
cache/
├── <hash_1>/
│   ├── kernel.ttir         # TT IR
│   ├── kernel.ttgir        # TTG IR
│   ├── kernel.llir         # LLVM IR
│   ├── kernel.ptx          # PTX
│   ├── kernel.cubin        # 最终二进制
│   └── kernel.json         # 编译元数据
└── <hash_2>/
    └── ...

# 缓存键 = source_hash + backend_hash + options_hash + env_vars
key = get_cache_key(src, backend, options, env_vars=env_vars)
hash = hashlib.sha256(key.encode()).hexdigest()
```

## 14.7 CompiledKernel

```python
class CompiledKernel:
    def __init__(self, src, metadata_group, hash):
        # 从 JSON 解析元数据
        self.metadata = KernelMetadata(...)
        
        # 加载二进制和各阶段 IR
        self.asm = AsmDict({
            file.suffix[1:]: file.read_bytes()
            for file in asm_files
        })
        self.kernel = self.asm["cubin"]  # 二进制
        
        # 延迟初始化（只在启动前初始化）
        self.module = None
        self.function = None
        self._run = None

    def _init_handles(self):
        # 在 GPU 上加载模块
        self._run = driver.active.launcher_cls(self.src, self.metadata)
```

## 14.8 调试与中间 IR 导出

```bash
# 导出所有阶段的 IR 到缓存目录
export TRITON_KERNEL_DUMP=1
python my_kernel.py
# → 在 ~/.triton/cache/<hash>/ 中可以看到所有中间 IR

# 打印 PTX 输出
export TRITON_DUMP_NVPTX=1
python my_kernel.py

# 强制重新编译（跳过缓存）
export TRITON_ALWAYS_COMPILE=1
python my_kernel.py

# 覆盖特定阶段的 IR
export TRITON_KERNEL_OVERRIDE=1
python my_kernel.py
# 可以修改缓存中的 IR 文件来测试
```

## 14.9 triton-opt 调试

```bash
# 对保存的 .ttir 文件，手动运行 Pass
triton-opt kernel.ttir --tritongpu-coalesce --tritongpu-accelerate-matmul

# 查看每个 Pass 前后 IR 的变化
triton-opt --mlir-print-ir-after-all kernel.ttir 2>&1 | less
```

---

## 📝 课后作业

### 作业 1：编译 Pipeline 追踪

```python
# 写一个简单内核
@triton.jit
def simple_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = x * 2.0
    tl.store(y_ptr + offsets, y, mask=mask)

x = torch.randn(1024, device='cuda')
y = torch.zeros_like(x)

# 运行并导出 IR
import os
os.environ["TRITON_KERNEL_DUMP"] = "1"
simple_kernel[(1,)](x, y, 1024, BLOCK=1024)
```

1. 找到缓存目录中的 `.ttir` 文件，观察生成的 Triton IR
2. 找到 `.ttgir` 文件，对比它与 `.ttir` 的区别
3. 找到 `.llir`，理解 LLVM IR 如何表示 Triton 操作

### 作业 2：阅读 compile()

在 `python/triton/compiler/compiler.py` 中找到 `compile` 函数，画出函数调用的流程图（从获取 target 到返回 CompiledKernel）。

### 作业 3：理解 add_stages

在 `third_party/nvidia/backend/compiler.py` 中找到 `add_stages` 方法，回答：
1. 注册了几个 stage？分别是什么？
2. 对于 `Language.TRITON` 和 `Language.GLUON`，注册的阶段有什么不同？
3. 为什么 `ttir` 阶段只对 TRITON 语言注册，而对 GLUON 不注册？

---

## 本章小结

- `compile()` 是 Triton 编译器的唯一入口，接收 Python 函数或 MLIR 文件
- 编译流水线由后端注册（`add_stages`），不同后端可以有不同的阶段
- 编译结果缓存在 `~/.triton/cache/`，缓存键基于源码+目标+选项
- `CompiledKernel` 封装了编译后的内核，包含二进制、各阶段 IR 和元数据
- `TRITON_KERNEL_DUMP=1` 是最重要的调试工具——导出所有中间 IR
- 理解了 compile() 的流程，就理解了 Triton 编译器的骨架，第 15-19 章填充血肉
