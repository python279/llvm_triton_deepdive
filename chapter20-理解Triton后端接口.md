# 第 20 章：理解 Triton 后端接口

> **本章目标**：理解 `BaseBackend` 的所有抽象方法，明确实现一个后端需要哪些组件。

> 📂 **第五部分：打造第三方后端 🏆** — 征服新龙，在未知硬件上实现编译器后端

> 驯龙手记：走到这里，你已经驯服了 LLVM（巨龙）和 Triton（海神）。
> 现在是时候面对一条全新的、从未有人驯服过的龙——你自己的硬件后端。
> 本章教你"驯龙术"的通用接口——不管龙长什么样，驯服它的步骤都一样。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter20/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `minimal_backend/` | 20.9 | 最小可注册后端脚手架 |
| `test_registration.py` | 20.9 | 验证 backends 注册 |

运行：

```bash
cd books/examples/chapter20
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 20.1 后端接口总览

Triton 编译器的"后端"由两部分组成：**编译管线**（`BaseBackend`）和**运行时驱动**（`DriverBase`）。你的工作就是实现这两个接口的具体子类。

```python
# python/triton/backends/compiler.py — 编译管线
class BaseBackend(metaclass=ABCMeta):
    supports_native_tensor_specialization = True

    def __init__(self, target: GPUTarget) -> None:
        self.target = target
        assert self.supports_target(target)

    @staticmethod
    @abstractmethod
    def supports_target(target: GPUTarget) -> bool:
        """后端是否支持这个 GPU 目标"""

    @abstractmethod
    def hash(self) -> str:
        """后端唯一标识（用于缓存隔离）"""

    @abstractmethod
    def parse_options(self, opts: dict) -> object:
        """将用户选项字典解析为后端特定对象"""

    @abstractmethod
    def add_stages(self, stages: dict, options: object) -> None:
        """注册编译阶段 — 函数字典"""

    @abstractmethod
    def load_dialects(self, context) -> None:
        """加载后端需要的 MLIR Dialect"""

    @abstractmethod
    def get_codegen_implementation(self, options) -> dict:
        """提供代码生成回调函数"""

    @abstractmethod
    def get_module_map(self) -> Dict[str, ModuleType]:
        """外部库的模块映射"""

    @staticmethod
    def parse_attr(desc: str) -> list:
        """解析属性描述符"""
        ...

    @staticmethod
    def get_int_specialization(arg, **kwargs) -> str:
        """获取整数特化标记"""
        ...

    @staticmethod
    def get_tensor_specialization(arg, **kwargs) -> str:
        """获取张量特化标记"""
        ...
```

让我们逐一解释每个方法的作用和需要怎么实现。

### `supports_target(target)` — 判断是否支持这个 GPU

这是最简单的接口——它返回 `True` 当且仅当你的后端支持给定的目标。Triton 内部用一个 `GPUTarget` 数据类来描述 GPU：

```python
@dataclass(frozen=True)
class GPUTarget:
    backend: str           # 'cuda', 'hip', 'mygpu'
    arch: Union[int, str]  # 90 (SM90), gfx940, ...
    warp_size: int         # 32 (NVIDIA), 64 (AMD), ...
```

你的后端实现通常是：

```python
@staticmethod
def supports_target(target: GPUTarget) -> bool:
    return target.backend == 'mygpu'
```

### `hash()` — 后端唯一标识

因为编译缓存是基于目标+源码+后端的哈希，后端的哈希必须稳定且唯一。通常包括编译器版本和架构信息。

```python
@functools.lru_cache()
def hash(self) -> str:
    version = get_compiler_version(self.target.arch)
    return f'{version}-{self.target.arch}'
```

**为什么需要 `@lru_cache`？** 因为每次 `compile()` 调用都会多次获取哈希（缓存键、检查缓存等），但哈希值在编译过程中不变。

### `parse_options(opts)` — 解析编译选项

用户可以通过 `kernel[grid](args, num_warps=8)` 传递编译选项。这些选项以字典形式到达 `parse_options`。你的工作是验证、扩充，并返回一个对象（如 `dataclass`）。

**NVIDIA 后端的实际做法**：

```python
@dataclass(frozen=True)
class CUDAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    warp_size: int = 32
    maxnreg: Optional[int] = None
    ptx_version: int = None
    # ... 更多字段 ...

class CUDABackend(BaseBackend):
    def parse_options(self, opts) -> CUDAOptions:
        args = {'arch': knobs.runtime.override_arch or f"sm{self.target.arch}"}
        args.update({k: opts[k] for k in CUDAOptions.__dataclass_fields__.keys()
                     if k in opts and opts[k] is not None})
        return CUDAOptions(**args)
```

为你的后端设计类似的 dataclass 并实现解析逻辑。

### `add_stages(stages, options, language)` — 注册编译阶段

这是**最核心**的方法。`stages` 是一个字典，key 是文件扩展名（如 `"ttir"`, `"ttgir"`），value 是接受 `(module, metadata)` 并返回新 module 的函数。

```python
def add_stages(self, stages, options, language):
    if language == Language.TRITON:
        stages["ttir"] = lambda mod, meta: self.make_ttir(mod, meta, options)
        stages["ttgir"] = lambda mod, meta: self.make_ttgir(mod, meta, options)
    stages["llir"] = lambda mod, meta: self.make_llir(mod, meta, options)
    stages["mybin"] = lambda mod, meta: self.make_mybin(mod, meta, options)
```

每个 lambda 对应的 `make_xxx` 方法创建一个 Pass Manager，添加所需的 Pass，运行它。完成后返回新的 module。

### `load_dialects(context)` — 加载 MLIR Dialect

在 C++ 层注册你的自定义 Dialect：

```python
def load_dialects(self, ctx):
    from triton._C.libtriton import mygpu  # 自定义绑定
    mygpu.load_dialects(ctx)
```

这个调用对应 C++ 中的 `MyGPUDialect::registerInto(context)`。

### `get_codegen_implementation(options)` — 代码生成配置

提供代码生成器的配置函数（如自定义类型转换、dot 最小尺寸等）。

### `get_module_map()` — 外部库映射

提供外部数学库的 Python 模块映射：

```python
def get_module_map(self):
    from triton.language.extra.mygpu import math_lib
    return {"triton.language.extra.mygpu": math_lib}
```

## 20.2 后端注册机制

```python
# python/triton/backends/__init__.py
backends = {}  # 全局注册表

# 第三方后端通过 third_party/ 下的 __init__.py 注册
# 例如：third_party/my_backend/__init__.py
from ..backends import backends
from .backend.compiler import MyBackend
from .backend.driver import MyGPUDriver

backends["mygpu"] = {
    "compiler": MyBackend,
    "driver": MyGPUDriver(),
}
```

Triton 启动时遍历 `third_party/` 下的目录，自动导入 `__init__.py` 以完成注册。之后 `make_backend()` 通过 `supports_target()` 选择匹配的后端。

## 20.3 需要实现的接口

| 方法 | 必须 | 实现复杂度 | 说明 |
|------|------|-----------|------|
| `supports_target` | ✅ | 极低 | 一行 `return target.backend == 'xxx'` |
| `hash` | ✅ | 极低 | 返回版本+架构字符串 |
| `parse_options` | ✅ | 中 | 解析用户选项，返回 dataclass |
| `add_stages` | ✅ | 高 | 注册编译阶段（核心） |
| `load_dialects` | ✅ | 中 | 加载自定义 Dialect |
| `get_codegen_implementation` | ❌ | 中 | 提供代码生成配置 |
| `get_module_map` | ❌ | 低 | 外部库映射 |
| `pack_metadata` | ❌ | 低 | 打包元数据给运行时 |

## 20.4 compile() 如何调用后端

```python
def compile(src, target=None, options=None):
    # 步骤 1: 根据 GPUTarget 选择后端
    backend = make_backend(target)

    # 步骤 2: 解析选项 — 后端特定
    extra_options = src.parse_options()
    options = backend.parse_options(dict(options, **extra_options))

    # 步骤 3: 注册编译阶段
    stages = {}
    backend.add_stages(stages, options, src.language)

    # 步骤 4: 依次运行每个阶段
    for ext, compile_ir in list(stages.items())[first_stage:]:
        module = compile_ir(module, metadata)

    # 步骤 5: 返回 CompiledKernel
    return CompiledKernel(src, metadata_group, hash)
```

## 20.5 最小后端需要的工作

一个最小完整的后端包括：

**编译管线部分：**

1. **实现 `BaseBackend` 子类** — 一个 Python 类（如 `class MyGPUBackend`）
2. **`add_stages()` 注册编译阶段** — 至少需要将 TTGIR 降到你的目标代码
3. **每个阶段实现 IR → IR 的转换** — 通过创建 Pass Manager 并添加 Pass
4. **最后一阶段生成二进制** — 返回 `bytes` 对象

**运行时部分：**

5. **实现 `DriverBase` 子类** — GPU 设备管理
6. **提供内核启动机制** — 封装硬件加载和启动函数
7. **提供设备信息查询** — 包括目标 GPU、LLVM triple、数据布局

**注册部分：**

8. **将后端的 compiler 和 driver 注册到 `backends` 字典**

## 20.6 三种后端策略详解

### 策略 A：通过 LLVM 后端（推荐）

```
你的工作和已复用的部分：
┌───────────── 需要你实现 ─────────────┐
│ TTG IR → 自定义Dialect → LLVM IR    │
│ (你实现 TTG→自定义降级, 自定义→LLVM降级) │
└─────────────────────────────────────┘
                                      ↓
┌───────────── 复用 LLVM ──────────────┐
│ LLVM IR → LLVM优化(O3) → 目标代码   │
│ (llvm.optimize_module, llc)           │
└─────────────────────────────────────┘
```

**优势**：只需关注 TTG → LLVM 的 IR 转换，后续 LLVM 优化和代码生成全部复用。
**劣势**：依赖 LLVM 后端质量，若 LLVM 后端不完善则性能受限。
**适用**：已有成熟 LLVM 后端的目标硬件。

### 策略 B：直接生成目标汇编

```
TTG IR → 自定义降级 → 目标汇编文本 → 目标汇编器 → 目标代码
         (你实现全部)    (你实现)     (硬件厂商提供)
```

**优势**：完全控制每条指令的生成。
**劣势**：需要实现完整的代码发射器（寄存器分配、指令调度等），工程量大。
**适用**：DSP、专用加速器等无 LLVM 后端的硬件。

### 策略 C：C++ 模板生成（快速原型）

```
TTG IR → 自定义降级 → C++ 代码 → 目标C编译器 → 目标代码
```

**优势**：最快验证可行性。
**劣势**：性能极差（无效的线程映射、无寄存器分配优化）。
**适用**：可行性验证阶段，确定基本功能后再切换到策略 A 或 B。

## 20.7 目录结构模板

```
third_party/my_backend/
├── __init__.py              # 后端注册到 backends 字典
├── backend/
│   ├── __init__.py
│   ├── compiler.py          # MyGPUBackend(BaseBackend)
│   └── driver.py            # MyGPUDriver(DriverBase)
├── lib/
│   ├── CMakeLists.txt
│   ├── MyBackendToLLVM/     # TTG → LLVM 降级（策略 A）
│   │   ├── CMakeLists.txt
│   │   └── ConvertOps.cpp
│   └── MyBackendTransforms/ # 后端特有优化
│       └── MyOptimization.cpp
├── include/
│   ├── CMakeLists.txt
│   └── MyBackend/
│       └── IR/               # 自定义 Dialect TableGen 文件
├── language/
│   └── mygpu.py              # 目标特定库函数
└── tools/
    └── compile.c             # 汇编器/链接器工具封装
```

## 20.8 驱动程序接口详解

```python
# python/triton/backends/driver.py
class DriverBase:
    """运行时驱动 — 负责设备管理、内核加载与启动"""

    def get_current_target(self) -> GPUTarget:
        """返回当前 GPU 的 GPUTarget"""
        device = self.get_current_device()
        arch = get_arch_from_device(device)
        return GPUTarget('mygpu', arch, warp_size=32)

    def get_current_device(self):
        """返回当前设备句柄"""
        return _get_device_handle(0)

    @property
    def launcher_cls(self):
        """内核启动器类 — 编译后的关键"""
        return MyGPULaucnher

    @property
    def utils(self):
        """工具函数模块 — 提供内存、设备属性等"""
        return MyGPUtils()
```

### `launcher_cls` 的职责

`Launcher` 在用户调用 `kernel[grid](args)` 时被调用：

```python
class MyGPULauncher:
    def __init__(self, src, metadata):
        self.binary = src.asm['mybin']     # 二进制内核
        self.shared_mem = metadata.shared   # 共享内存需求
        self.num_warps = metadata.num_warps

    def __call__(self, *args, grid=None, **kwargs):
        # 1. 将 Python 参数复制到 GPU 设备内存
        device_args = [allocate_and_copy(arg) for arg in args]

        # 2. 启动内核
        mygpu_launch_kernel(
            self.binary,            # 二进制
            grid,                   # (gridX, gridY, gridZ)
            self.num_warps * 32,   # block 大小
            self.shared_mem,        # 共享内存
            device_args             # 参数
        )

        # 3. 同步并返回
        mygpu_sync()
```

## 20.9 从无到有：实现最小后端的步骤

假设你正在为一个"只有 C 编译器"的新硬件实现后端（策略 C——最快验证）。

**第 1 步：创建后端目录**

```bash
mkdir -p third_party/my_backend/{backend,lib,include,language,tools}
```

**第 2 步：实现 `MyGPUBackend`**

```python
# third_party/my_backend/backend/compiler.py
from triton.backends.compiler import BaseBackend, GPUTarget, Language

class MyGPUBackend(BaseBackend):
    binary_ext = "cpp"  # 生成 C++ 代码

    @staticmethod
    def supports_target(target):
        return target.backend == 'mygpu'

    def hash(self):
        return f'mygpu-v1-{self.target.arch}'

    def parse_options(self, opts):
        return MyGPUOptions(**opts)

    def add_stages(self, stages, options, language):
        if language == Language.TRITON:
            stages["ttir"] = lambda m, meta: self.make_ttir(m, meta, options)
            stages["ttgir"] = lambda m, meta: self.make_ttgir(m, meta, options)
        stages["cpp"] = lambda m, meta: self.make_cpp(m, meta, options)

    def load_dialects(self, ctx):
        pass  # 策略 C 不需要自定义 Dialect

    def get_codegen_implementation(self, options):
        return {}

    def get_module_map(self):
        return {}
```

**第 3 步：实现编译阶段**

```python
def make_ttir(self, mod, metadata, options):
    # 复用 Triton 的标准 ttir Pass
    pm = ir.pass_manager(mod.context)
    passes.common.add_inliner(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_combine(pm)
    passes.common.add_cse(pm)
    pm.run(mod, 'make_ttir')
    return mod
```

**第 4 步：注册**

```python
# third_party/my_backend/__init__.py
from triton.backends import backends
backends["mygpu"] = {"compiler": MyGPUBackend, "driver": None}  # 第 1 步先不要 driver
```

**第 5 步：测试**

```bash
pip install -e .
python -c "
import triton
# 检查后端是否被识别
from triton.backends import backends
print('mygpu' in backends)
"
```

---

## 📝 课后作业

### 作业 1：阅读 BaseBackend

阅读 `python/triton/backends/compiler.py` 中 `BaseBackend` 的完整定义，回答：

1. 哪几个方法是 `@abstractmethod`？（6 个）
2. `parse_attr` 有什么作用？
3. `hash()` 方法为什么需要 `@functools.lru_cache()`？
4. `pack_metadata` 的非默认实现是什么？

### 作业 2：实现最小后端

按照 20.9 节的步骤，为你的目标硬件创建后端框架代码。

### 作业 3：选择策略

以你在第 7 章作业 2 中选择的硬件为例：
1. 你会选择哪种后端策略？为什么？
2. 画出你的编译流水线（每个阶段的输入和输出）
3. 列出你需要实现的 Pass 列表

---

## 本章小结

- `BaseBackend` 定义了后端编译管线的 8 个接口（6 个必须、2 个可选）
- `DriverBase` 定义运行时驱动的 4 个关键组件
- 后端的核心：`add_stages()` 注册编译阶段，每个阶段是一个 `module → module` 的函数
- 三种后端策略：通过 LLVM（推荐，低工作量）、直接汇编（全控制，高工作量）、C++ 模板（最快验证）
- 后端注册在 `third_party/` 下，通过 `backends` 全局字典管理
- 最小后端只需 `supports_target` + `add_stages` + `parse_options` 三个方法即可验证可行性
- 本章是第 21-24 章（完整实现）的前置知识
