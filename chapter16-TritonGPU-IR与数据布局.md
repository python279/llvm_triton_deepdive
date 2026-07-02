# 第 16 章：TritonGPU IR 与数据布局

> **本章目标**：理解 TTG Dialect 的数据布局编码，掌握编码如何控制线程-元素映射。

> 📂 **第四部分：Triton 深度剖析**

> 驯龙手记：编码（Encoding）就是海神 Triton 身上的"鳞甲排列方式"。同一头海神，
> 鳞甲可以排成不同的图案——有的图案适合冲刺（计算密集型），有的适合潜行（内存密集型）。
> 编译器的工作是选择最适合当前任务的鳞甲排列。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter16/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `vector_add.ttgir` | 16.4 | 带 #blocked 的 TTGIR |
| `encoding_calc.py` | 作业 | Blocked encoding 覆盖计算 |

运行：

```bash
cd books/examples/chapter16
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 16.1 TTG 与 TT 的区别

TTG Dialect 是 Triton 编译流程中**最关键的一层**——它解答了"数据的物理归属"问题。

```mlir
; TT IR（设备无关）
%x = tt.load %ptr : tensor<128xf32>

; TTG IR（带 GPU 编码）
#blocked = #ttg.blocked<{sizePerThread=[1], threadsPerWarp=[32], warpsPerCTA=[4], order=[0]}>
%x = tt.load %ptr : tensor<128xf32, #blocked>
           ↑ 张量类型上多了编码属性
```

**关键区别**：
- TT 中的张量是 `tensor<128xf32>`（无编码）——只描述了"128 个连续元素"这一逻辑事实
- TTG 中的张量是 `tensor<128xf32, #blocked>`（有编码）——精确描述了"128 个元素如何分配到 128 个线程"这一物理事实

编码回答了一个核心问题：**张量的每个元素由哪个 GPU 线程持有？**

### 为什么需要编码？

考虑一个简单的 `tensor<128xf32>` — 128 个浮点数。NVIDIA GPU 通常有 128 个线程（4 warps × 32 threads）。你如何决定哪个线程持有哪个元素？

```
方案 A: Thread 0 → element 0, Thread 1 → element 1, ...（顺序分配）
方案 B: Thread 0 → elements 0,1, Thread 1 → elements 2,3, ...（每线程 2 元素）
方案 C: Warp 0 → elements 0-31, Warp 1 → elements 64-95, ...（交错分配）
```

每种方安对内存合并（coalescing）的影响不同。如果 threads 连续访问连续地址，GPU 能将它们合并为一次内存事务（16x 性能提升）。编码的选择直接影响内核运行时性能。

## 16.2 编码（Encoding）体系

TTG 定义了一系列编码，全部在 `include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td`（1503 行）。

### `BlockedEncodingAttr` — 寄存器布局

最基本的编码——元素分布在线程的寄存器中。这是你在 TTG IR 中最常见的编码类型。

```tablegen
def BlockedEncodingAttr : TritonGPU_Attr<"BlockedEncoding", "blocked", [...]> {
    let parameters = (ins
        ArrayRefParameter<"unsigned">:$sizePerThread,  // 每个线程持几个元素
        ArrayRefParameter<"unsigned">:$threadsPerWarp, // 每 warp 线程数
        ArrayRefParameter<"unsigned">:$warpsPerCTA,    // 每 CTA warp 数
        ArrayRefParameter<"unsigned">:$order,          // 维度排序
        "CGAEncodingAttr":$CTALayout                   // 线程组布局
    );
}
```

**五个参数逐一详解**：

1. **`sizePerThread`** — 每个线程持有张量的几个元素
   - `[1]` = 每个线程 1 个元素（1D 张量）
   - `[4, 1]` = 每个线程 4 个元素在 dim0，1 个在 dim1（2D 张量）
   - 更大的 `sizePerThread` 可以减少指令数，但可能增加寄存器压力

2. **`threadsPerWarp`** — 一个 warp 内的 32 个线程如何排到 2D/3D 网格
   - `[32]` = 1D 排列（32×1）
   - `[16, 2]` = 2D 排列（16 行 × 2 列）
   - 乘积必须等于 warp 大小（NVIDIA=32, AMD=64）

3. **`warpsPerCTA`** — CTA（Cooperative Thread Array）内 warp 如何排列
   - `[4]` = 4 个 warp 线性排列
   - `[2, 2]` = 4 个 warp 排列为 2×2

4. **`order`** — 维度优先级（哪个维度是"最连续的"）
   - `[0]` = dim0 是连续维度
   - `[1, 0]` = dim1 比 dim0 更连续（即列优先/row-major）
   - 直接影响内存合并效果！`order=[1,0]` 通常用于 N 方向连续的张量

5. **`CTALayout`** — CTA 的全局排列（跨多个 CTA 时使用）

**参数含义**（以 `tensor<128x256xf16>` 为例）：

```
#blocked = #ttg.blocked<{
    sizePerThread = [1, 2],     // 每个线程持有 [1个M维度, 2个N维度] = 2个元素
    threadsPerWarp = [16, 2],   // warp 内 32 线程排为 [16行, 2列]
    warpsPerCTA = [2, 4],       // 每个 CTA 8 个 warp 排为 [2行, 4列]
    order = [1, 0]              // N 维度是连续（最快变化）的
}>
```

**覆盖范围计算**：

```
沿 M 维度：sizePerThread[0] × threadsPerWarp[0] × warpsPerCTA[0]
         = 1 × 16 × 2 = 32 个元素
沿 N 维度：sizePerThread[1] × threadsPerWarp[1] × warpsPerCTA[1]
         = 2 × 2 × 4 = 16 个元素

一个 CTA 覆盖 32×16 = 512 个元素
tensor 有 128×256 = 32768 个元素
需要 32768/512 = 64 个 CTA（4×16 网格）
```

### 编码如何影响性能：一个具体例子

```python
# 场景：加载一个 128×256 的矩阵 A
# A 在内存中行优先存储：A[0,0], A[0,1], ..., A[0,255], A[1,0], ...
```

**方案 A：`order = [1, 0]`（列优先，N 顺时针）**

```
Thread 0 持有: A[0,0], A[0,1]
Thread 1 持有: A[0,2], A[0,3]
...
Adjacent threads access adjacent addresses → 1 memory transaction ✓
```

**方案 B：`order = [0, 1]`（行优先，M 顺时针）**

```
Thread 0 持有: A[0,0], A[1,0]
Thread 1 持有: A[2,0], A[3,0]
...
Adjacent threads access addresses 256 apart → 32 memory transactions ✗
```

**性能差距可达 16-32x**。

### `SwizzledSharedEncodingAttr` — 共享内存布局

当数据需要从寄存器经过共享内存（用于 warp 间数据交换）时，使用这个编码。Swizzle 通过 XOR 操作重新排列共享内存地址，避免 bank conflict。

```tablegen
def SwizzledSharedEncodingAttr : TritonGPU_Attr<"SwizzledSharedEncoding", "swizzled_shared", [...]> {
    let parameters = (ins
        "unsigned":$vec,         // 向量化宽度
        "unsigned":$perPhase,    // 每个 phase 的行数
        "unsigned":$maxPhase,    // phase 数量
        ArrayRefParameter<"unsigned">:$order,
        "CGAEncodingAttr":$CGALayout
    );
}
```

**什么是 Bank Conflict？**

NVIDIA GPU 的共享内存有 32 个 bank。如果 32 个线程在同一个时钟周期访问不同的 bank，可以在一次时钟内完成。但如果多个线程访问同一个 bank（而不同地址），需要多次时钟。

```
无 bank conflict：
  Thread[0] → bank[0], Thread[1] → bank[1], ..., Thread[31] → bank[31]
  ✓ 一次时钟完成

有 bank conflict（2-way）：
  Thread[0] → bank[0], Thread[1] → bank[0], ..., Thread[31] → bank[31]
  ✗ 两次时钟（前两个线程排队等 bank[0]）
```

**Swizzle 如何解决这个问题？**

```
原始排列（顺序访问）：
  Row 0: [0, 1, 2, 3]      → bank[0..3]
  Row 1: [4, 5, 6, 7]      → bank[4..7]
  Row 2: [8, 9, 10, 11]    → bank[8..11]

如果每行 4 个元素都用 8 个线程读取，row 0 和 row 2 都冲突在 bank[0]
→ Swizzle 后：

#ttg.swizzled_shared<{vec=1, perPhase=1, maxPhase=4, order=[1,0]}>
  Row 0: [0, 1, 2, 3]       → bank[0..3]   XOR 0 → [0,1,2,3]
  Row 1: [5, 4, 7, 6]       → bank[1,0,3,2] XOR 1 → [5,4,7,6]
  Row 2: [10, 11, 8, 9]     → bank[2,3,0,1] XOR 2 → [10,11,8,9]
  Row 3: [15, 14, 13, 12]   → bank[3,2,1,0] XOR 3 → [15,14,13,12]

现在 row 0 和 row 2 不冲突！
```

### `MmaEncodingAttr` — Tensor Core 布局

矩阵乘法操作 `tt.dot` 要求特定的输入/输出布局，与 Tensor Core 的硬件实现绑定。NVIDIA GPU 的不同代数使用不同的 MMA 指令：

| GPU世代 | MMA版本 | Tile形状(A) | Tile形状(B) | PTX指令 |
|---------|---------|------------|------------|---------|
| Ampere | MMAv2 | 16×8×16 | 8×16×16 | `mma.sync.aligned.m16n8k16` |
| Hopper | WGMMA | 64×8×16 | 8×64×16 | `wgmma.mma_async.sync.aligned` |
| Blackwell | MMAv5 | 可变 | 可变 | 新一代PTX指令 |

`MmaEncodingAttr` 描述输入矩阵 A 和 B 如何分布在 warp 内的线程上以匹配硬件 Matrix Multiply-Accumulate 单元的期望格式。这是一个高度优化的、硬件特定的布局——你永远不会手动创建它（Coalesce + Accelerate Matmul Pass 自动选择它）。

## 16.3 convert_layout — 最重要的 TTG Op

`convert_layout` 是 TTG IR 中**出现频率最高、最重要**的操作。每次数据在两种编码之间移动时，必须插入它。

```mlir
#blocked = #ttg.blocked<{sizePerThread=[1], threadsPerWarp=[32], warpsPerCTA=[4], order=[0]}>
#shared = #ttg.swizzled_shared<{vec=8, perPhase=4, maxPhase=4, order=[1, 0]}>
#mma = #ttg.mma<{version=2, warpsPerCTA=[2, 2]}>

; 情形 1: 寄存器 → 共享内存
%in_shared = ttg.convert_layout %in_reg : tensor<128x256xf16, #blocked> → tensor<128x256xf16, #shared>

; 情形 2: 共享内存 → Tensor Core 输入
%a_mma = ttg.convert_layout %a_shared : tensor<128x32xf16, #shared> → tensor<128x32xf16, #mma>

; 情形 3: Tensor Core 输出 → 寄存器（写回用）
%c_reg = ttg.convert_layout %c_mma : tensor<128x256xf32, #mma> → tensor<128x256xf32, #blocked>
```

**`convert_layout` 底层实现**：

当编译到 LLVM IR 时，`convert_layout` 被展开为实际的 GPU 指令。展开方式取决于源编码和目标编码：

1. **blocked → blocked（寄存器内重塑）**：warp shuffle 指令（`shfl.sync.bfly`）在线程间交换数据
2. **blocked → swizzled_shared（写共享内存）**：每个线程计算 XOR-swizzled 地址，执行 `local_store`
3. **swizzled_shared → blocked（读共享内存）**：执行 `local_load` 并根据 swizzle 模式解 XOR
4. **shared → mma（Tensor Core 准备）**：重新排列线程持有的数据以匹配 MMA 硬件的期望格式

## 16.4 一个完整的 TTGIR 示例

```mlir
#blocked = #ttg.blocked<{sizePerThread=[1], threadsPerWarp=[32], warpsPerCTA=[4], order=[0]}>
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @vector_add(%ptr_a: !tt.ptr, %ptr_b: !tt.ptr, %ptr_out: !tt.ptr) {
    ; 1. 计算程序ID和偏移
    %pid = tt.get_program_id {axis = 0 : i32} : i32
    %off = arith.muli %pid, %cst : i32
    %off_t = tt.splat %off : (i32) -> tensor<1024xi32, #blocked>
    %range = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %idx = arith.addi %off_t, %range : tensor<1024xi32, #blocked>

    ; 2. 指针计算
    %ptr_a_t = tt.splat %ptr_a : (!tt.ptr) -> tensor<1024x!tt.ptr, #blocked>
    %a_addr = tt.addptr %ptr_a_t, %idx : tensor<1024x!tt.ptr, #blocked>, tensor<1024xi32, #blocked>
    %b_addr = tt.addptr %ptr_a_t, %idx : tensor<1024x!tt.ptr, #blocked>, tensor<1024xi32, #blocked>

    ; 3. 全局内存加载 — 注意每个张量类型都带 #blocked 编码
    %a = tt.load %a_addr : tensor<1024xf32, #blocked>
    %b = tt.load %b_addr : tensor<1024xf32, #blocked>

    ; 4. 计算 — 编码保持一致，不需要 convert_layout
    %sum = arith.addf %a, %b : tensor<1024xf32, #blocked>

    ; 5. 写回全局内存
    tt.store %ptr_out_t, %sum : tensor<1024xf32, #blocked>
    tt.return
  }
}
```

## 16.5 TTG 的关键 Op

| Op | 说明 | 底层实现 |
|----|------|---------|
| `ttg.convert_layout` | 数据布局转换 | warp shuffle / shared mem R/W |
| `ttg.local_load` | 共享内存 → 寄存器 | addrspace(3) load |
| `ttg.local_store` | 寄存器 → 共享内存 | addrspace(3) store（含 swizzle） |
| `ttg.alloc_shared` | 分配共享内存 | LLVM `alloca` in addrspace(3) |
| `ttg.async_copy_global_to_local` | 异步拷贝 | CUDA `cp.async` (SM80+) |
| `ttg.async_wait` | 等待异步拷贝完成 | 硬件同步指令 |
| `ttg.async_commit_group` | 提交异步拷贝组 | 硬件同步指令 |

## 16.6 TTG 优化 Pass

定义在 `include/triton/Dialect/TritonGPU/Transforms/Passes.td`。

| Pass | 作用 | 关键文件 |
|------|------|---------|
| **Coalesce** | 分析内存访问，选择最优布局 | `Coalesce.cpp` |
| AccelerateMatmul | 识别 matmul 并选择 Tensor Core 编码 | `AccelerateMatmul.cpp` |
| RemoveLayoutConversions | 消除冗余的 convert_layout | `RemoveLayoutConversions.cpp` |
| OptimizeDotOperands | 优化 dot 的操作数布局 | `OptimizeDotOperands.cpp` |
| Pipeline | 软件流水线 | `Pipeliner/` |
| CoalesceAsyncCopy | 异步拷贝合并 | `CoalesceAsyncCopy.cpp` |
| ReorderInstructions | 指令重排 | `ReorderInstructions.cpp` |
| ReduceDataDuplication | 减少数据重复 | `ReduceDataDuplication.cpp` |

### Coalesce Pass 的核心逻辑（逐行解析）

`lib/Dialect/TritonGPU/Transforms/Coalesce.cpp`：

```cpp
// 核心函数：为 load/store 选择最优数据布局
static Attribute pickDescriptorLoadStoreLayout(
    int numWarps, int threadsPerWarp, RankedTensorType type) {
    
    // step 1: 计算这个 CTA 处理的 tiling
    auto shapePerCTA = triton::gpu::getShapePerCTA(type);
    int numElems = product<int64_t>(shapePerCTA);

    // step 2: 确定每个线程要处理的元素数
    int numThreads = numWarps * threadsPerWarp;
    int numElemsPerThread = std::max(numElems / numThreads, 1);

    // step 3: 确定最大向量化宽度（受限于内存位宽）
    // 例如，f16: 128/16 = 8 (最多一次加载 8 个 f16)
    int maxVectorSize = 128 / type.getElementTypeBitWidth();
    int vectorSize = std::min(numElemsPerThread, maxVectorSize);

    // step 4: 构造编码——最后一个维度向量化
    SmallVector<unsigned> sizePerThread(type.getRank(), 1);
    sizePerThread.back() = vectorSize;  // e.g., [1, vectorSize]

    SmallVector<unsigned> order = getMatrixOrder(type.getRank(), /*rowMajor*/ true);
    auto cgaLayout = triton::gpu::getCGALayout(type.getEncoding());

    Attribute layout = triton::gpu::BlockedEncodingAttr::get(
        type.getContext(), type.getShape(), sizePerThread, order,
        numWarps, threadsPerWarp, cgaLayout);
    return layout;
}
```

**这个函数的核心洞察**：向量化宽度 = min(每线程元素数, 硬件允许的最大向量宽度)。选择 `sizePerThread.back()` = vectorSize 使得连续的线程访问连续的内存地址。

## 16.7 数据流示例：Matmul 的完整布局变化

一个矩阵乘法内核（`A @ B + C`）经历了 5 次数据布局转换：

```
1. 加载 A/B 到寄存器（Blocked 编码，连续访问最大化）
   %A_reg = tt.load %A_ptr : tensor<128x256xf16, #blocked>
   %B_reg = tt.load %B_ptr : tensor<256x128xf16, #blocked>

2. 转换到共享内存（Swizzled 编码，避免 bank conflict）
   %A_shared = ttg.convert_layout %A_reg
       : tensor<128x256xf16, #blocked> -> tensor<128x256xf16, #swizzled_shared>
   %B_shared = ttg.convert_layout %B_reg
       : tensor<256x128xf16, #blocked> -> tensor<256x128xf16, #swizzled_shared>

3. 从共享内存加载到 Tensor Core 输入布局
   %A_mma = ttg.convert_layout %A_shared
       : tensor<128x32xf16, #swizzled_shared> -> tensor<128x32xf16, #mma>
   %B_mma = ttg.convert_layout %B_shared
       : tensor<32x128xf16, #swizzled_shared> -> tensor<32x128xf16, #mma>

4. Tensor Core 计算
   %C_mma = tt.dot %A_mma, %B_mma, %C_mma
       : tensor<128x32xf16, #mma> * tensor<32x128xf16, #mma>
       -> tensor<128x128xf32, #mma>

5. 转回寄存器布局，写回全局内存
   %C_reg = ttg.convert_layout %C_mma
       : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
   tt.store %C_ptr, %C_reg : tensor<128x128xf32, #blocked>
```

每一步的数据布局（编码）都经过精心选择：Coalesce Pass 处理步骤 1，AccelerateMatmul Pass 选择步骤 3 的 MMA 编码，Pipeline Pass 可能将步骤 2 分拆为多个异步拷贝步骤。

---

## 📝 课后作业

### 作业 1：理解编码参数

给定一个 `tensor<64x128xf32>`，计算以下编码需要的线程数和 warp 数：

```
#blocked = #ttg.blocked<{
    sizePerThread = [1, 2],
    threadsPerWarp = [16, 2],
    warpsPerCTA = [2, 2],
    order = [1, 0]
}>
```

提示：`需要总线程数 = threadsPerWarp[0] × threadsPerWarp[1] × warpsPerCTA[0] × warpsPerCTA[1]`。然后计算每线程元素数，得出总覆盖范围。

### 作业 2：对比 TTIR 和 TTGIR

运行一个内核，用 `TRITON_KERNEL_DUMP=1` 导出 `.ttir` 和 `.ttgir`：

1. 对比两个文件中张量类型的区别
2. 找出 `.ttgir` 中出现了哪些 `.ttir` 中没有的新 Op
3. 找出 `convert_layout` 出现的位置
4. 对每一个 `convert_layout`，解释为什么需要这个转换

### 作业 3：设计自定义编码

假设你正在实现一个新 GPU 的 Triton 后端。该 GPU 有 64 个线程/warp 和 8 warps/CTA，支持 256-bit 内存事务。为一个 `tensor<256x512xf16>` 的 `tt.load` 设计最优的 `sizePerThread`：

1. 最大向量宽度是多少？（f16 = 16 bits，256/16 = 16）
2. 总线程数是多少？（64 × 8 = 512）
3. 总共多少元素？（256 × 512 = 131072）
4. 每线程多少元素？（131072 / 512 = 256）
5. 最优 `sizePerThread` 是多少？

### 作业 4：Swizzle 模式计算

对于 `#ttg.swizzled_shared<{vec=4, perPhase=2, maxPhase=2, order=[1, 0]}>`：

1. row 0 的 4 个元素 [0, 1, 2, 3] 在共享内存中如何排列？
2. row 1 的 4 个元素 [4, 5, 6, 7] 在共享内存中如何排列？（提示：XOR 1）
3. row 2 的 4 个元素 [8, 9, 10, 11] 在共享内存中如何排列？

---

## 本章小结

- TTG Dialect = TT Dialect + GPU 数据布局编码
- 编码（Encoding）回答核心问题：每个张量元素由哪个线程持有
- `BlockedEncodingAttr`：寄存器布局（5 个参数：sizePerThread, threadsPerWarp, warpsPerCTA, order, CTALayout）
- `SwizzledSharedEncodingAttr`：共享内存布局（XOR swizzle 避免 bank conflict）
- `MmaEncodingAttr`：Tensor Core 布局（硬件特定，自动选择）
- `convert_layout` 是最关键的 TTG Op——在编码之间转换，底层展开为 shuffle 或 共享内存操作
- TTG 优化 Pass（Coalesce、AccelerateMatmul 等）自动选择最优编码
- 内存合并（coalescing）是编码选择的首要优化目标
