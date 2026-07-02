# 第 24 章：集成与调试

> **本章目标**：掌握后端调试技巧，理解如何将后端集成到 Triton 中。
>
> 驯龙手记：驯龙的最后一课——当龙不听话时怎么办？调试就是在龙的耳边低声细语，
> 倾听它的反馈（错误信息）；性能分析是测量它的奔跑速度、跳跃高度；
> 集成则是让它和龙群中的其他龙（Triton 框架）和谐共处。恭喜，你已经成为了一名驯龙高手。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter24/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `debug_dump.sh` | 24.4 | IR dump 调试命令模板 |

运行：

```bash
cd books/examples/chapter24
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 24.1 集成后的完整调用链

```python
# 用户代码
result = my_kernel[grid](x, y, BLOCK=1024)
```

```text
用户调用 → JITFunction → compile() → make_backend
    → MyBackend → add_stages → ttir → ttgir
    → mygpu → llir → mybin → CompiledKernel
    → MyGPULauncher → GPU 执行
```

## 24.2 后端注册入口

```python
# third_party/my_backend/__init__.py
from triton.backends import backends
from .backend.compiler import MyBackend
from .backend.driver import MyGPUDriver

# 注册后端
backends["my_backend"] = {
    "compiler": MyBackend,
    "driver": MyGPUDriver(),
}
```

Triton 启动时自动遍历 `third_party/` 目录（通过 `setup.py` 中的 entry_points 或 `importlib`），加载注册的后端。

## 24.3 调试层级

```
调试四层级:
  L1: Python运行时 -- 参数绑定、网格维度、内存分配
  L2: IR正确性    -- TRITON_KERNEL_DUMP、triton-opt、FileCheck
  L3: 代码生成     -- LLVM IR、目标汇编、寄存器分配
  L4: 硬件执行     -- 仿真器、printf、性能计数器
```

## 24.4 IR 调试技巧

```bash
# 1. 导出所有中间 IR
export TRITON_KERNEL_DUMP=1
python my_kernel.py

# 2. 直接加载并运行 triton-opt
# 从缓存中的 .mygpu 文件手动运行 Pass
triton-opt kernel.mygpu --convert-mygpu-to-llvm -o output.ll

# 3. 打印 Pass 执行前后的 IR
triton-opt --mlir-print-ir-before-all --mlir-print-ir-after-all input.mlir
```

## 24.5 常见问题及排查

### 问题 1：编译错误 — "Unsupported operation"

```
triton-opt: error: 'mygpu.load' op trying to legalize 
operation that was not marked as legal or illegal
```

**原因**：`ConversionTarget` 中未标记 `mygpu.load` 的合法性。
**修复**：在 ConversionTarget 中添加：

```cpp
target.addLegalOp<mygpu::LoadOp>();
```

### 问题 2：运行时崩溃 — 段错误

**可能原因**：
1. 共享内存不足 → 检查 `alloc_shared` 的大小
2. 指针地址空间错误 → 检查 TypeConverter 中的地址空间映射
3. warp 索引越界 → 检查 `get_local_id` 的语义

### 问题 3：计算结果错误

```
输入: [1, 2, 3]  + [4, 5, 6]
预期: [5, 7, 9]
实际: [5, 7, 0]  ← 最后元素丢失
```

**排查步骤**：
1. 检查 mask 是否正确
2. 检查网格维度是否覆盖所有元素
3. 检查共享内存的 bank conflict 或 swizzle 模式
4. 用极简输入逐步验证

## 24.6 性能调试

```python
import time
start = time.time()
kernel[grid](*args)
end = time.time()
print(f"Kernel time: {end - start:.3f}ms")

# 2. triton 内置基准
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],
        x_vals=[128 * i for i in range(1, 5)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
    )
)
def benchmark(M, N, K, provider):
    return ...

# 3. 对比参考实现
# CUDA 参考时间 vs MyGPU 时间
```

## 24.7 开发 checklist

### 编译管线

```
□ BaseBackend 子类实现
□ add_stages 注册所有阶段
□ 各阶段 Pass 正确运行
□ typeConverter 正确映射所有类型
□ conversionTarget 标记正确的 Dialect
□ 最后一阶段生成正确格式的二进制
□ 缓存机制正常工作
```

### 运行时

```
□ DriverBase 子类实现
□ 设备信息正确返回
□ 内核加载
□ 内核启动
□ 内存分配/释放
□ 数据拷贝 H2D/D2H
□ 同步操作
```

### 测试

```
□ LIT 测试覆盖每个 Pass
□ Python 单元测试覆盖基本运算
□ 端到端测试验证 matmul
□ 回归测试覆盖边界条件
□ 无 GPU 时的测试隔离
```

## 24.8 完成一个后端的时间线（参考）

```
第 1-2 月：学习阶段
├── 学习 LLVM/MLIR 基础
├── 理解 Triton 架构
└── 完成本书前 20 章

第 3-4 月：原型阶段
├── 定义 Dialect（1 周）
├── 实现 TTG → 自定义 Dialect 转换（3 周）
├── 实现自定义 Dialect → LLVM 降级（2 周）
└── 运行时集成（1 周）

第 5-6 月：稳定阶段
├── 修复已知 bug（2 周）
├── 优化性能（2 周）
├── 完善测试覆盖（2 周）
└── 编写文档（1 周）
```

## 24.9 🏆 最终项目

祝贺你走到这里！你的最终项目是：

**为一个现有的非 NVIDIA/AMD 硬件（或模拟器）实现完整的 Triton 编译器后端。**

项目要求：
1. 完整的 MMD（设计文档）
2. 至少通过 LIT 测试覆盖 10 个核心 Op
3. 至少通过 3 个 Python 端到端测试（vector add, matmul, reduction）
4. 性能达到手写参考实现的 50%+
5. 包含完整的 CI 测试配置

---

## 📝 课后作业

### 作业 1：调试练习

人为引入一个 bug（如 TypeConverter 中不转换某个类型），观察 triton-opt 的输出错误信息，写出排查过程。

### 作业 2：性能分析

对你后端的 matmul 实现进行性能分析：
1. 测量不同大小矩阵的执行时间
2. 计算 FLOPs 利用率
3. 找到性能瓶颈（内存带宽限制还是计算限制）

### 作业 3：完成 Checklist

根据 24.7 节的 checklist，逐项检查你的后端实现，记录每项完成状态和备注。

---

## 本章小结

- 后端集成通过 `third_party/` 目录注册，Triton 自动发现
- 调试四层级：Python 运行时 → IR 正确性 → 代码生成 → 硬件执行
- `TRITON_KERNEL_DUMP=1` + `triton-opt` 是核心调试工具
- 常见问题：ConversionTarget 配置错误、地址空间错误、共享内存不足
- 后端开发时间线：3 个月原型 → 6 个月稳定
- 🏆 最终项目：为真实硬件实现完整的 Triton 后端
- 本书到此结束，但你的编译器开发之旅才刚刚开始！
