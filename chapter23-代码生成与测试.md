# 第 23 章：代码生成与测试

> **本章目标**：实现 MyGPU Dialect → 目标代码的生成，并建立完整的测试体系。
>
> 驯龙手记："新龙驯服了吗？让它跑两步看看！"代码生成是让新龙真正跑起来，
> 测试则是检验它跑得稳不稳（正确性）、快不快（性能）。
> LIT 测试是慢走训练，E2E 测试是跨越障碍物，CI/CD 是每天的例行训练。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter23/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `test/add.mlir` | 23.x | LIT 风格 MLIR 测试 |

运行：

```bash
cd books/examples/chapter23
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 23.1 从 MyGPU IR 到目标代码

对于策略 A（通过 LLVM 后端）：

```
MyGPU IR (mygpu Dialect)
  --> mygpu.matmul → LLVM IR 的 call 指令（调用外部库）
  --> mygpu.barrier → LLVM IR 的 fence + call
  --> mygpu.load → LLVM IR 的 load
  --> LLVM IR（标准）
  --> llc -mtriple=mygpu-unknown-elf
  --> 目标汇编/目标文件
```

### 实现 MyGPU → LLVM 降级

```cpp
// third_party/my_backend/lib/MyGPUToLLVM/MyGPUToLLVM.cpp

struct BarrierOpLowering : public OpRewritePattern<mygpu::BarrierOp> {
    using OpRewritePattern::OpRewritePattern;
    
    LogicalResult matchAndRewrite(mygpu::BarrierOp op,
                                  PatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        // LLVM IR 的内存屏障
        // fence acq_rel 确保内存可见性
        rewriter.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::acq_rel);
        
        // 如果目标平台需要特殊指令
        // 可以生成内联汇编
        // rewriter.create<LLVM::InlineAsmOp>(...)
        
        rewriter.eraseOp(op);
        return success();
    }
};

struct MatrixMulLowering : public OpRewritePattern<mygpu::MatrixMulOp> {
    LogicalResult matchAndRewrite(mygpu::MatrixMulOp op,
                                  PatternRewriter &rewriter) const override {
        Location loc = op.getLoc();
        // 方式 1：调用外部 BLAS 库
        // auto call = rewriter.create<LLVM::CallOp>(loc, ...);
        
        // 方式 2：展开为循环（性能差）
        // for (i...) for (j...) for (k...) result[i][j] += a[i][k] * b[k][j];
        
        // 方式 3：生成内联汇编（性能好）
        auto asmStr = "mygpu.matmul $0, $1, $2, $3";
        auto asmOp = rewriter.create<LLVM::InlineAsmOp>(
            loc, resultTypes, operands, asmStr, constraints,
            /*has_side_effects=*/true);
        
        rewriter.replaceOp(op, asmOp.getResult(0));
        return success();
    }
};
```

## 23.2 二进制生成

```cpp
// 调用 LLVM 后端生成目标文件
void makeBinary(ModuleOp module, const std::string &outputPath) {
    // 用 llc 生成目标代码
    llvm::LLVMContext context;
    auto llvmModule = llvm::translateModuleToLLVMIR(module, context);
    
    // 设置目标
    llvmModule->setTargetTriple("mygpu-unknown-elf");
    llvmModule->setDataLayout("e-m:e-p:32:32-i64:64-f32:32");
    
    // 运行 LLVM 优化
    llvm::legacy::PassManager pm;
    pm.add(llvm::createVerifierPass());
    // 添加 O2 优化
    pm.run(*llvmModule);
    
    // 输出目标文件
    std::error_code EC;
    llvm::raw_fd_ostream out(outputPath, EC);
    llvm::legacy::PassManager emitPM;
    if (auto *TM = llvm::EngineBuilder().selectTarget()) {
        TM->addPassesToEmitFile(emitPM, out, 
                                 llvm::CodeGenFileType::ObjectFile);
        emitPM.run(*llvmModule);
    }
}
```

## 23.3 运行时支持

### Python 驱动层

```python
# third_party/my_backend/backend/driver.py

class MyGPUDriver(DriverBase):
    def __init__(self):
        # 加载运行时库
        self._lib = ctypes.CDLL("libmygpu_runtime.so")
    
    def get_current_device(self):
        return ctypes.c_int(0)  # 假设单设备
    
    def get_current_target(self):
        return GPUTarget('mygpu', 2, warp_size=16)  # 架构 v2
    
    @property
    def launcher_cls(self):
        return MyGPULaucnher
    
    @property
    def utils(self):
        return MyGPUUtils()

class MyGPULauncher:
    def __init__(self, src, metadata):
        self.binary = src.asm['mybin']
    
    def __call__(self, *args, grid, **kwargs):
        # 1. 分配 GPU 内存
        # 2. 拷贝数据
        # 3. 启动内核
        # 4. 同步
        # 5. 拷贝结果回
        pass
```

### C/C++ 运行时层

```c
// third_party/my_backend/tools/mygpu/compile.h

// 设备管理
typedef struct { int dev_id; } MyGPUDevice;
MyGPUDevice* mygpu_get_device(int id);
void mygpu_free_device(MyGPUDevice* dev);

// 内存管理
void* mygpu_alloc(MyGPUDevice* dev, size_t size, int kind);  // kind: 0=global, 1=shared
void mygpu_free(MyGPUDevice* dev, void* ptr);
void mygpu_memcpy(MyGPUDevice* dev, void* dst, const void* src, size_t size, int dir); // dir: 0=H2D, 1=D2H

// 内核管理
typedef struct { ... } MyGPUModule;
MyGPUModule* mygpu_load_module(const void* data, size_t size);
void mygpu_launch_kernel(MyGPUModule* mod, const char* name,
                          int gridX, int gridY, int gridZ,
                          void** args, size_t sharedMem);
void mygpu_sync(MyGPUDevice* dev);
```

## 23.4 测试体系

### 层级 1：LIT 测试

```mlir
// test/MyGPU/basic_ops.mlir
// RUN: triton-opt %s --convert-mygpu-to-llvm | FileCheck %s

module {
  mygpu.func @test_load_store(%ptr: !mygpu.ptr<f32>) {
    %0 = mygpu.load %ptr : f32
    mygpu.store %0, %ptr : f32
    mygpu.return
  }
}
// CHECK: llvm.load
// CHECK: llvm.store
```

### 层级 2：Python 单元测试

```python
# python/test/unit/my_backend/test_basic.py

def test_single_element_add():
    @triton.jit
    def add_kernel(x_ptr, y_ptr, output_ptr):
        pid = tl.program_id(axis=0)
        x = tl.load(x_ptr + pid)
        y = tl.load(y_ptr + pid)
        tl.store(output_ptr + pid, x + y)
    
    # 准备数据
    x = np.array([3.0], dtype=np.float32)
    y = np.array([4.0], dtype=np.float32)
    out = np.zeros([1], dtype=np.float32)
    
    # 启动内核
    add_kernel[(1,)](x, y, out)
    
    # 验证
    assert np.allclose(out, [7.0])
```

### 层级 3：端到端测试

```python
def test_matmul():
    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, **meta):
        pass  # ... 完整的 matmul 实现

    # 与 PyTorch 参考结果对比
    a = torch.randn(128, 256, device='mygpu')
    b = torch.randn(256, 64, device='mygpu')
    c = torch.zeros(128, 64, device='mygpu')
    
    matmul_kernel[(grid,)](a, b, c, ...)
    expected = torch.mm(a, b)
    
    assert torch.allclose(c, expected, atol=1e-4)
```

## 23.5 CICD 集成

```yaml
# .github/workflows/backend-test-mygpu.yml
name: MyGPU Backend Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v3
      - name: Build Triton with MyGPU
        run: |
          export LLVM_DIR=/opt/llvm/lib/cmake/llvm
          pip install -e python
      - name: Run LIT tests
        run: lit -v test/MyGPU/
      - name: Run Python tests
        run: pytest python/test/unit/my_backend/ -v
```

## 23.6 测试覆盖率目标

| 测试类型 | 覆盖率目标 | 优先级 |
|---------|-----------|--------|
| Op 定义 | 所有 Op 的创建和打印 | P0 |
| Op 验证 | 验证逻辑覆盖所有错误条件 | P0 |
| 转换 Pass | 所有 TT Op → 目标 Op | P0 |
| 降级 Pass | 目标 Op → LLVM + 代码生成 | P0 |
| 单元素内核 | Vector add, mul | P0 |
| 典型内核 | Matmul, reduction, softmax | P1 |
| 边界条件 | 大 Block, 0-size, 边缘 | P1 |
| 性能回归 | 核心算子的性能基准 | P2 |

---

## 📝 课后作业

### 作业 1：实现 Barrier 降级

参照 23.1 节，实现 `mygpu.barrier` → LLVM IR 的降级模式。要求：
1. 生成 `LLVM::FenceOp`（`acq_rel` 语义）
2. 生成平台相关的 barrier 调用
3. 用 LIT 测试验证

### 作业 2：实现测试

为你的后端实现以下 LIT 测试：
1. `test_load_store.mlir` — 加载和存储的基本测试
2. `test_matmul.mlir` — 矩阵乘法的测试
3. `test_barrier.mlir` — 同步 barrier 的测试

### 作业 3：编写端到端测试

用 Python 编写一个端到端的 matmul 测试（参考 23.4 节），与参考实现对比结果。

---

## 23.7 性能分析工具

完成基本功能后，下一步是优化性能。以下是 Triton 后端开发中最常用的性能分析工具：

### 时间测量

```python
import time
start = time.time()
kernel[grid](*args)
torch.cuda.synchronize()
elapsed = time.time() - start
print(f"Kernel time: {elapsed*1000:.2f} ms")

# 2. Triton 内置基准框架
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=[128 * i for i in range(1, 5)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
    )
)
def benchmark(M, N, K, provider):
    ...  # 返回 ms
```

### 硬件性能计数器

NVIDIA：

```bash
# Nsight Compute (ncu) — 内核级性能分析
ncu --set full python my_kernel.py

# 关键指标解释
# - Achieved Occupancy: 实际占用率 vs 理论最大
# - Memory Throughput: 内存带宽利用率（%）
# - Compute Throughput: 计算单元利用率（SM、Tensor Core）
# - L1/L2 Hit Rate: 缓存命中率
# - Branch Efficiency: 分支效率

# Nsight Systems (nsys) — 系统级时间线
nsys profile -t cuda,nvtx python my_kernel.py
```

AMD：

```bash
# ROCprofiler
rocprof --stats python my_kernel.py

# Omnitrace / Omniperf
```

通用：

```bash
# NVTX / ROCTX 范围标记
# 在 Triton 调试输出中查看
TRITON_DUMP_NVPTX=1 python my_kernel.py
```

### 内核占用率计算

```python
# 根据编译元数据计算理论占用率
def calc_occupancy(num_warps, shared_mem, max_warps_per_sm=64, max_shared=100*1024):
    warp_limit = max_warps_per_sm // num_warps
    shared_limit = max_shared // shared_mem if shared_mem > 0 else warp_limit
    return min(warp_limit, shared_limit) * num_warps / max_warps_per_sm * 100
```

### 性能基线对比

```python
# 将你的后端性能与参考实现对比
baseline = benchmark_torch_mm(M, N, K)       # PyTorch cuBLAS 性能
triton_perf = benchmark_triton_mm(M, N, K)    # Triton + 你的后端
print(f"Performance: {triton_perf/baseline*100:.1f}% of cuBLAS")
# 目标：达到 70%+ 为良好，50%+ 为可行
```



## 本章小结

- 代码生成 = MyGPU Dialect → LLVM IR + LLVM 后端处理
- 矩阵乘降级的三种方式：外部库调用、循环展开、内联汇编
- 运行时 = 驱动层（Python）+ 运行时库（C/C++）
- 测试体系三层次：LIT（Op/Pass 测试）→ 单元测试（Python）→ 端到端（完整内核）
- CI/CD 自动运行测试，覆盖从 Op 定义到性能基准的各个层次
