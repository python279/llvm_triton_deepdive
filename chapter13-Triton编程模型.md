# 第 13 章：Triton 编程模型

> **本章目标**：理解 Triton 的块级编程模型，掌握 `@triton.jit`、语言核心类型、内置函数的用法。
>
> 📂 **第四部分：Triton 深度剖析** — 驾驭海神，精通 Triton 编译器的每一环

> 驯龙手记：Triton 即海神（海马/人鱼），是你在 GPU 水域中最好的坐骑。
> 你不需要在奔腾的浪花（线程）中挣扎——只需指挥 Triton 冲向正确的方向（块级操作），
> 它自己会处理好水下的暗流（线程映射、内存合并、同步）。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter13/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `vector_add.py` | 13.x | 完整 vector add 内核（需 GPU） |

运行：

```bash
cd books/examples/chapter13
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 13.1 从 CUDA 到 Triton

```cuda
// CUDA — 手动管理线程
__global__ void saxpy(float *y, float a, float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] = a * x[i] + y[i];
}
dim3 grid(128);
dim3 block(256);
saxpy<<<grid, block>>>(y, a, x, n);
```

```python
# Triton — 块级描述
@triton.jit
def saxpy_kernel(y_ptr, a, x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, a * x + y, mask=mask)
```

**关键区别**：
- CUDA 中你需要指定 `gridDim` 和 `blockDim`
- Triton 中你只需要指定 `BLOCK` 大小和计算逻辑，编译器自动处理线程映射

## 13.2 `@triton.jit` 装饰器

```python
# python/triton/runtime/jit.py
# @triton.jit 将 Python 函数标记为 JIT 内核

@triton.jit
def kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    """BLOCK 是编译时常量（tl.constexpr）"""
    ...
```

**执行流程**：

```
1. 用户调用 kernel(x, y, BLOCK=1024)
2. JITFunction.__call__() 被调用
3. 参数绑定和类型推断
4. 检查编译缓存（`~/.triton/cache/<hash>/`）
5. 缓存命中 → 返回 CompiledKernel
6. 缓存未命中 → 编译 → 缓存 → 返回
```

## 13.3 Triton 语言核心类型

### `tensor` — 张量

```python
# Triton 中的 tensor 代表线程内的工作"块"
# 可以是标量（单元素）或多元素块

x = tl.load(ptr)            # 返回 tensor（块或标量，取决于 ptr 类型）
y = tl.arange(0, 1024)      # 创建 tensor<1024xi32>
z = x + y                   # 逐元素加法
```

### `dtype` — 数据类型

```python
# 内置数据类型
tl.float32    # 32 位浮点
tl.float16    # 16 位浮点
tl.bfloat16   # BF16
tl.int32      # 32 位整数
tl.int64      # 64 位整数
```

### `constexpr` — 编译时常量

```python
@triton.jit
def kernel(BLOCK: tl.constexpr):  # BLOCK 编译时确定
    offsets = tl.arange(0, BLOCK)  # BLOCK 用于确定 tensor 形状
```

## 13.4 核心内置函数

### 内存操作

```python
# 加载 — mask 实现边界检查
x = tl.load(ptr, mask=mask, other=0.0)

# 存储
tl.store(ptr, value, mask=mask)

# 原子操作
old = tl.atomic_add(ptr, val)
old = tl.atomic_max(ptr, val)
```

### 并行原语

```python
# 获取 program ID（块索引）
pid = tl.program_id(axis=0)   # 相当于 CUDA 的 blockIdx.x

# 获取 program 数量（网格维度）
num = tl.num_programs(axis=0) # 相当于 CUDA 的 gridDim.x
```

### 算术与数学

```python
# 逐元素运算
z = x + y     # 加法
z = x * y     # 乘法
z = x / y     # 除法

# 数学函数
z = tl.exp(x)    # 指数
z = tl.log(x)    # 对数
z = tl.sin(x)    # 正弦
z = tl.sqrt(x)   # 平方根
```

### 归约

```python
# 块内归约
result = tl.sum(x, axis=0)      # 求和
result = tl.max(x, axis=0)      # 最大值
result = tl.min(x, axis=0)      # 最小值

# 自定义归约
result = tl.reduce(x, axis=0, combine_fn=my_fn)

# 前缀扫描
result = tl.associative_scan(x, axis=0, combine_fn=my_fn)
```

### 矩阵乘法

```python
# Tensor Core 矩阵乘法
c = tl.dot(a, b, c)  # c = a @ b + c
```

## 13.5 完整示例：Vector Add

```python
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. 计算这个 program 处理的元素范围
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 2. 边界检查
    mask = offsets < n_elements
    
    # 3. 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 4. 计算
    output = x + y
    
    # 5. 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)

# 启动内核
def vector_add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)  # 计算需要的 program 数量
    vector_add_kernel[grid](x, y, output, n, BLOCK_SIZE=BLOCK_SIZE)
    return output

# 测试
x = torch.randn(10000, device='cuda')
y = torch.randn(10000, device='cuda')
result = vector_add(x, y)
assert torch.allclose(result, x + y)
```

## 13.6 完整示例：矩阵乘法

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 计算当前块的位置
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    
    # A 块：[BLOCK_M, BLOCK_K]，B 块：[BLOCK_K, BLOCK_N]
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        # 加载 A 和 B 的子块
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + 
                     (offs_k + k)[None, :] * stride_ak)
        b = tl.load(b_ptr + (offs_k + k)[:, None] * stride_bk + 
                     offs_bn[None, :] * stride_bn)
        # 矩阵乘法累加
        accumulator = tl.dot(a, b, accumulator)
    
    # 存储结果
    c = accumulator.to(tl.float16)
    tl.store(c_ptr + offs_am[:, None] * stride_cm + 
              offs_bn[None, :] * stride_cn, c)
```

## 13.7 与 Python 运行时交互

```python
# 自动调优装饰器
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def kernel(x_ptr, y_ptr, n, BLOCK_SIZE: tl.constexpr):
    ...

# 手动启动
kernel[grid](x_ptr, y_ptr, n, BLOCK_SIZE=1024)
```

---

## 📝 课后作业

### 作业 1：写一个向量乘加内核

用 Triton 实现 `y = a * x + y`（SAXPY），其中 `a` 是标量，`x` 和 `y` 是向量。用 PyTorch 验证结果的正确性。

### 作业 2：修改 Block 大小

在上面的 Vector Add 示例中，将 `BLOCK_SIZE` 从 1024 改为 64 和 4096，分别观察性能变化。思考为什么？

### 作业 3：阅读 core.py

快速浏览 `python/triton/language/core.py` 的前 300 行，回答：
1. `builtin` 装饰器的作用是什么？
2. `tensor` 类有哪些属性？
3. `constexpr` 类和普通 Python 值有什么区别？

---

## 本章小结

- Triton 的编程模型是**块级编程**——操作整个张量块，不关心单个线程
- `@triton.jit` 将 Python 函数标记为 JIT 编译的 GPU 内核
- 核心类型：`tensor`（块数据）、`dtype`（数据类型）、`constexpr`（编译时常量）
- `tl.load/store` 进行内存操作，`mask` 参数处理边界
- `tl.dot` 自动利用 Tensor Core 加速矩阵乘法
- Triton 代码可以直接在 GPU 上运行，编译产生的 PTX 通过 ptxas 转为 CUBIN
