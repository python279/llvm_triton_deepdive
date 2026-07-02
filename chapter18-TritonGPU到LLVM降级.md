# 第 18 章：TritonGPU → LLVM 降级

> **本章目标**：理解 TTG → LLVM IR 的降级过程，聚焦 `convert_layout` 的展开、共享内存分配和代码发射。

> 驯龙手记：降级是海神从神话生物变成真实生物的过程。每一层 convert_layout
> 都是一次"形态重塑"——从图腾（TTG）变成血肉（LLVM）。共享内存分配是
> 构建它的内部器官，PTX 生成则是它的第一声呼吸。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter18/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `sample.ptx` | 18.x | PTX 模式检查快照 |

运行：

```bash
cd books/examples/chapter18
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 18.1 降级管道的定位

```
TTG IR（带编码）
  --> 共享内存分配
  --> SCF→CF（控制流扁平化）
  --> TTG→LLVM Dialect（主降级）
  --> NVGPU→LLVM（NVIDIA 特有降级）
  --> NVVM→LLVM（CUDA 内置函数）
  --> LLVM 优化（O3）
  --> LLVM IR → PTX → CUBIN
```

## 18.2 NVIDIA 后端的 make_llir

```python
# third_party/nvidia/backend/compiler.py L366-465
def make_llir(self, src, metadata, options, capability):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    
    # 1. 准备：控制流转换，内联
    passes.ttgpuir.add_combine_tensor_select_and_if(pm)
    passes.convert.add_scf_to_cf(pm)       # 结构化 → 非结构化控制流
    
    # 2. 内存分配
    nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, capability, ptx_version)
    nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm)
    
    # 3. 主降级
    nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, ptx_version, ...)
    nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm)
    nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)
    passes.convert.add_nvvm_to_llvm(pm)
    
    # 4. 后处理优化
    passes.ttgpuir.add_canonicalize_llvm_ir(pm)
    passes.common.add_cse(pm)
    passes.common.add_canonicalizer(pm)
    
    pm.run(mod, 'make_llir')
    return mod
```

## 18.3 共享内存分配

TTG 降级的**第一步**是分配共享内存，因为大多数布局转换通过共享内存完成。

```cpp
// third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.cpp

// 分析所有需要共享内存的 Op（local_store, async_copy 等）
// 计算总共享内存需求
// 在 LLVM IR 中插入 shared memory 分配

for (auto &op : moduleOps) {
    if (auto store = dyn_cast<ttg::LocalStoreOp>(op)) {
        // 根据张量类型计算共享内存大小
        sharedSize += tensorSizeInBytes(store.getType());
    }
}
// 生成：@shared_mem = internal global [total_size x i8], addrspace(3)
```

## 18.4 convert_layout 的展开

`convert_layout` 是降级中最复杂的操作。展开策略：

### 情况 1：blocked → blocked（寄存器内重排）

```
源: tensor<128xf32, #blocked_A>
目标: tensor<128xf32, #blocked_B>

策略：warp shuffle（shfl.sync.bfly）
在线程间交换寄存器数据

LLVM IR：
%shuffled = call @llvm.nvvm.shfl.sync.bfly.f32(
    %active_mask, %val, %src_lane, %width)
```

### 情况 2：blocked → swizzled_shared

```
源: tensor<128xf32, #blocked>（寄存器）
目标: tensor<128xf32, #shared>（共享内存）

策略：
1. 计算每个线程要写入的共享内存地址（按 swizzle 模式）
2. 执行 local_store（LLVM store to addrspace(3)）

LLVM IR：
%addr = getelementptr ...  ; 计算共享内存地址（含 swizzle）
store float %val, float addrspace(3)* %addr
```

### 情况 3：swizzled_shared → blocked

```
对称操作：local_load

LLVM IR：
%val = load float, float addrspace(3)* %addr
```

## 18.5 ElementwiseOp 的降级

```cpp
// ElementwiseOpToLLVM.cpp

// AddPtrOp：指针加法
struct AddPtrOpConversion : public ConvertOpToLLVMPattern<AddPtrOp> {
    LogicalResult matchAndRewrite(...) {
        // ptr + offset → GEP
        Value result = b.gep(ptrTy, elemTy, ptr, offset);
        rewriter.replaceOp(op, result);
        return success();
    }
};

// CmpIOp：比较
struct CmpIOpConversion : public ElementwiseOpConversionBase<arith::CmpIOp, ...> {
    SmallVector<LLVM::ICmpOp> createDestOps(...) {
        // arith.cmpi → LLVM icmp
        return {LLVM::ICmpOp::create(rewriter, loc, elemTy,
                                      predicate, lhs, rhs)};
    }
};
```

## 18.6 ReduceOp 的降级

```cpp
// ReduceOpToLLVM.cpp

// 如果 getSingleCombiner() 返回非空（单个组合操作）
if (auto combiner = reduceOp.getSingleCombiner()) {
    // 可以优化为 warp 级归约
    for (int step = warpSize / 2; step > 0; step /= 2) {
        // shfl.sync.bfly 交换数据
        val = shuffle_xor(val, step);
        // 应用组合操作
        val = combiner(val, shuffled_val);
    }
    // 如果硬件支持，直接用 redux.sync.add
}
```

## 18.7 DotOp 的降级 — MMA

```cpp
// third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/

// MMAv2.cpp — Ampere (SM 80/86)
// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {.satfinite}

// WGMMA.cpp — Hopper (SM 90)
// wgmma.fence, wgmma.commit_group

// MMAv5.cpp — Blackwell (SM 100+)
```

## 18.8 LLVM 优化与 PTX 生成

```python
def make_llir(self, src, metadata, options, capability):
    # ... 前面 Pass 运行完，得到 LLVM Dialect IR ...
    
    # LLVM Dialect → LLVM Native IR
    llvm_mod = llvm.to_module(mod, context)
    
    # 设置目标
    triple = 'nvptx64-nvidia-cuda'
    proc = sm_arch_from_capability(capability)  # sm_90
    features = get_features(options, target.arch)  # +ptx86
    llvm.attach_datalayout(llvm_mod, triple, proc, features)
    
    # LLVM O3 优化
    llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3,
                         disable_slp_vectorizer=capability == 80)
    return str(llvm_mod)
```

---

## 📝 课后作业

### 作业 1：PTX 观察

运行一个简单内核，用 `TRITON_DUMP_NVPTX=1` 查看 PTX 输出，回答：

1. 找到 `ld.global` 指令（全局内存加载）
2. 找到 `st.global` 指令（全局内存存储）
3. 找到 `shfl.sync.bfly` 或 `redux.sync`（如果有归约）
4. 找到 `mma.sync` 或 `wgmma`（如果有矩阵乘法）

### 作业 2：阅读 AllocateSharedMemory

在 `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Allocation.cpp` 中：

1. 共享内存分配策略是什么？
2. 不同 Op 的共享内存需求如何计算？
3. 分配后共享内存的地址空间是多少？

### 作业 3：DotOp 降级路径

研究 `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/DotOpToLLVM/` 中的文件，回答：

1. MMAv2、WGMMA、MMAv5 各对应什么 GPU 架构？
2. 三者注册的 PTX 指令分别是什么？
3. FMA.cpp 在什么条件下使用（回退方案）？

---

## 本章小结

- TTG → LLVM 降级是整个编译流程中代码量最大、最复杂的部分
- 共享内存分配是降级的第一步——大多数布局转换依赖共享内存
- `convert_layout` 有多种展开策略：warp shuffle、shared memory R/W
- 硬件操作（MMA、`redux.sync`）在满足条件时被使用，否则回退到通用模式
- 降级后生成 LLVM Dialect，再转为 LLVM Native IR，最后通过 LLVM 优化生成 PTX
- NVIDIA 特有操作（WGMMA、TMA）通过 NVGPU → LLVM 降级处理
