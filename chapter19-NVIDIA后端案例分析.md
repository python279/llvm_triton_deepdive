# 第 19 章：NVIDIA 后端案例分析

> **本章目标**：以 NVIDIA CUDA 后端为案例，理解完整后端的工作原理。
>
> 驯龙手记：这是观摩一位前辈驯龙高手的驯龙过程。NVIDIA 后端就像一条
> 被驯服了最多次、最成熟的龙。研究它的每一寸筋骨——CUDABackend 是缰绳，
> ptxas 是龙鞍，SM 版本分支应对的是龙的不同年龄段——你将学会驯龙的最佳实践。
> 本章是第 20-24 章（第三方后端）的基础。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter19/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `inspect_cuda_backend.py` | 19.x | 检查 CUDABackend 源码 |
| `sample.ptx` | 19.x | PTX 头字段检查 |

运行：

```bash
cd books/examples/chapter19
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 19.1 CUDABackend 的完整结构

```
third_party/nvidia/
  backend/          -- CUDABackend、driver
  lib/
    TritonNVIDIAGPUToLLVM/     -- TTG→LLVM 降级
    TritonNVIDIAGPUTransforms/ -- NVIDIA 特有优化
    NVGPUToLLVM/               -- NVGPU→LLVM
    Dialect/NVGPU/IR/          -- NVIDIA 指令 Dialect
    Dialect/NVWS/IR/           -- Warp Specialization
  language/cuda/               -- CUDA 库函数
  triton_nvidia.cc             -- Python 绑定
```

## 19.2 CUDABackend 类

```python
class CUDABackend(BaseBackend):
    binary_ext = "cubin"
    
    @staticmethod
    def supports_target(target):
        return target.backend == 'cuda'
    
    def parse_options(self, opts) -> CUDAOptions:
        # 解析编译选项，加入架构特定默认值
        # 如 FP8 支持、Tensor Core 精度等
        return CUDAOptions(**opts)
```

### CUDAOptions

```python
@dataclass(frozen=True)
class CUDAOptions:
    num_warps: int = 4          # warp 数量
    num_ctas: int = 1           # CTA 数量
    num_stages: int = 3         # 流水线阶段数
    warp_size: int = 32         # warp 大小
    maxnreg: Optional[int] = None  # 寄存器上限
    ptx_version: int = None     # PTX 版本
    enable_fp_fusion: bool = True
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4b15")
    default_dot_input_precision: str = "tf32"
```

## 19.3 编译流水线（回顾）

```python
# stages = {}  — 由 add_stages 注册
stages = {
    "ttir":   self.make_ttir,     # TTIR 优化
    "ttgir":  self.make_ttgir,    # TTGIR 优化（关键步骤）
    "llir":   self.make_llir,     # TTG → LLVM
    "ptx":    self.make_ptx,      # LLVM → PTX
    "cubin":  self.make_cubin,    # PTX → 二进制
}
```

## 19.4 SM 版本分支

`make_ttgir()` 根据 `capability` 分三条路径：

| SM 版本 | 代号 | 关键特性 |
|---------|------|---------|
| SM 80/86 | Ampere | MMA v2, TF32, Async Copy |
| SM 90 | Hopper | WGMMA, TMA, Warp Specialization |
| SM 100+ | Blackwell | MMA v5, TMEM |

```python
if capability // 10 == 8:
    # Ampere 路径
    passes.ttgpuir.add_prefetch(pm)
    nvidia.passes.hopper.add_hopper_warpspec(pm, ...)
    
elif capability // 10 == 9:
    # Hopper 路径
    passes.ttgpuir.add_fuse_nested_loops(pm)
    passes.ttgpuir.add_pipeline(pm, num_stages, dump)
    nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
    
elif capability // 10 >= 10:
    # Blackwell 路径
    passes.ttgpuir.add_assign_latencies(pm, num_stages)
    passes.ttgpuir.add_hoist_tmem_alloc(pm, False)
    passes.ttgpuir.add_warp_specialize(pm, num_stages)
    nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem(pm)
```

## 19.5 PTX 版本控制

```python
def ptx_get_version(cuda_version) -> int:
    """CUDA 版本 → PTX 版本"""
    major, minor = map(int, cuda_version.split('.'))
    if major == 12:
        return 80 + minor         # CUDA 12.0 → PTX 80
    if major == 11:
        return 70 + minor         # CUDA 11.0 → PTX 70
    ...
```

```python
def get_features(options, arch: int):
    ptx_version = get_ptx_version_from_options(options, arch)
    llvm_ptx_version = min(90, ptx_version)  # LLVM 上限
    features = f'+ptx{llvm_ptx_version}'
    return features
```

## 19.6 PTX 生成与 ptxas

```python
# LLVM → PTX（文本汇编）
def make_ptx(self, src, metadata, opt, capability):
    triple = 'nvptx64-nvidia-cuda'
    proc = sm_arch_from_capability(capability)
    features = get_features(opt, target.arch)
    ret = llvm.translate_to_asm(src, triple, proc, features, flags, ...)
    return ret  # PTX 字符串

# ptxas 编译 PTX → CUBIN
def make_cubin(self, src, metadata, opt, capability):
    ptxas = get_ptxas(target.arch).path
    subprocess.run([ptxas, '-v', f'--gpu-name={arch}', 
                    src.name, '-o', fbin], check=True)
    return cubin  # 二进制内容
```

## 19.7 NVIDIA 特有 Dialect

### NVGPU Dialect

封装 NVIDIA GPU 指令级操作：

```mlir
nvgpu.mma_sync %a, %b, %c  ; Tensor Core MMA
nvgpu.warp_group_matmul ...  ; Hopper WGMMA
nvgpu.fence ...              ; 内存栅栏
```

### NVWS Dialect (Warp Specialization)

Warp 特化——Hopper/Blackwell 的重要特性：

```mlir
nvws.assign_warp_group %num_warps  ; 分配 warp 到任务组
nvws.wait                          ; Warp 间同步
```

### TMEM (Tensor Memory)

Blackwell 特有的张量内存：

```mlir
ttnv.tma_load %desc, %coord, %smem  ; TMA 加载
ttnv.tmem_alloc ...                  ; Tensor Memory 分配
```

## 19.8 从 NVIDIA 到第三方后端的启示

| NVIDIA 后端特性 | 第三方后端的对应 |
|----------------|----------------|
| PTX 文本格式 | 目标汇编格式 |
| ptxas 汇编器 | 目标汇编器 |
| sm_80/90/100 架构分支 | 目标架构版本分支 |
| CUDA 驱动 API | 目标驱动 API |
| libdevice (CUDA 数学库) | 目标数学库 |
| NVPTX LLVM 后端 | 目标 LLVM 后端（如有） |
| Tensor Core (MMA) | 目标矩阵加速器 |
| Shared Memory (addrspace 3) | 目标片上内存 |
| Warp (32 threads) | 目标线程组织 |

---

## 📝 课后作业

### 作业 1：阅读 CUDABackend 源码

在 `third_party/nvidia/backend/compiler.py` 中：

1. `supports_target` 方法的实现是什么？
2. `parse_options` 如何处理 FP8 类型支持？
3. 如何获取 ptxas 的路径？

### 作业 2：分析 PTX 生成

运行 `TRITON_DUMP_NVPTX=1` 获取 PTX，回答：

1. 找到 `.version` 和 `.target` 指令
2. 找到 `.reg` 声明了多少个寄存器
3. `.entry` 后面是什么？这是核函数的什么名称？

### 作业 3：对比 AMD 后端

浏览 `third_party/amd/backend/compiler.py`，对比与 NVIDIA 后端的异同：

| 对比项 | NVIDIA | AMD |
|--------|--------|-----|
| backend 名称 | 'cuda' | ? |
| 二进制扩展 | 'cubin' | ? |
| 目标三元组 | nvptx64-nvidia-cuda | ? |
| 汇编格式 | PTX | ? |
| 矩阵操作 | MMA/WGMMA | ? |

---

## 本章小结

- CUDABackend 是 NVIDIA GPU 的 Triton 后端实现
- 编译流水线根据 SM 版本分支（Ampere/Hopper/Blackwell）
- NVIDIA 特有 Dialect（NVGPU/NVWS）封装了 GPU 指令级操作
- PTX 生成通过 LLVM 的 NVPTX 后端，再调用 ptxas 汇编
- 理解 NVIDIA 后端的设计模式是理解第三方后端的钥匙
- 第 20-24 章将教你在非 NVIDIA/AMD 硬件上实现类似的后端
