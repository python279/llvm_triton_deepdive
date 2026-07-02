# 附录 D：参考书目与资源

> 按阶段推荐的深度学习资源。链接于 2026 年 7 月核对；若个别站点临时不可用，可尝试镜像或官方文档首页。

---

## D.1 C++ 学习资源

### 入门

| 资源 | 作者 | 说明 |
|------|------|------|
| **《C++ Primer》**（第 5 版） | Stanley B. Lippman、Josée Lajoie、Barbara E. Moo | C++11 经典入门，读前 12 章足够 |
| **[cppreference.com](https://en.cppreference.com)** | — | 在线 C++ 参考手册（必收藏） |

### 进阶

| 资源 | 作者 | 说明 |
|------|------|------|
| **《Effective Modern C++》** | Scott Meyers | C++11/14 最佳实践 |
| **《C++ Templates: The Complete Guide》**（第 2 版） | David Vandevoorde、Nicolai M. Josuttis | 模板深度教程（用到时查阅） |

## D.2 LLVM 学习资源

### 官方资源

| 资源 | 链接 | 说明 |
|------|------|------|
| **LLVM 文档** | [llvm.org/docs](https://llvm.org/docs/) | 官方文档入口 |
| **LLVM IR 语言参考** | [LangRef.html](https://llvm.org/docs/LangRef.html) | IR 指令速查 |
| **Kaleidoscope 教程** | [tutorial/](https://llvm.org/docs/tutorial/) | 手写编译器经典入门 |
| **TableGen 文档** | [TableGen/](https://llvm.org/docs/TableGen/) | 官方 TableGen 教程 |
| **New Pass Manager** | [NewPassManager.html](https://llvm.org/docs/NewPassManager.html) | **本书 Pass 示例使用此路径（推荐）** |
| **Writing an LLVM Pass（Legacy）** | [WritingAnLLVMPass.html](https://llvm.org/docs/WritingAnLLVMPass.html) | 旧版 Pass Manager，仅供对照 Legacy 插件 |

### 视频

| 资源 | 链接 | 说明 |
|------|------|------|
| **LLVM Developers' Meeting** | [YouTube 搜索](https://www.youtube.com/results?search_query=LLVM+Developers+Meeting) | 年度会议演讲录像 |
| **EuroLLVM** | [YouTube 搜索](https://www.youtube.com/results?search_query=EuroLLVM) | 欧洲 LLVM 会议 |
| **"LLVM IR Tutorial" by Mike Shah** | [YouTube 搜索](https://www.youtube.com/results?search_query=Mike+Shah+LLVM+IR+Tutorial) | IR 入门系列 |

## D.3 MLIR 学习资源

### 官方资源

| 资源 | 链接 | 说明 |
|------|------|------|
| **MLIR 文档** | [mlir.llvm.org](https://mlir.llvm.org/) | 所有 MLIR 文档入口 |
| **MLIR Toy 教程** | [Tutorials/Toy/](https://mlir.llvm.org/docs/Tutorials/Toy/) | **最佳 MLIR 入门（必做！）** |
| **MLIR Language Reference** | [LangRef/](https://mlir.llvm.org/docs/LangRef/) | MLIR 语言参考 |
| **Dialect Conversion** | [DialectConversion/](https://mlir.llvm.org/docs/DialectConversion/) | 方言转换框架详解 |

### 论文与扩展

| 资源 | 链接 | 说明 |
|------|------|------|
| **MLIR 论文** | [arXiv:2002.11054](https://arxiv.org/abs/2002.11054) | *MLIR: A Compiler Infrastructure for the End of Moore's Law* |
| **MLIR 会议演讲** | CGO、EuroLLVM 的 MLIR Track | 前沿实践 |
| **CIRCT 项目** | [circt.llvm.org](https://circt.llvm.org/) | MLIR 在硬件设计中的应用 |

## D.4 Triton 学习资源

### 官方资源

| 资源 | 链接 | 说明 |
|------|------|------|
| Triton 文档 | [triton-lang.org](https://triton-lang.org/main/index.html) | 官方文档首页 |
| Triton GitHub | [github.com/triton-lang/triton](https://github.com/triton-lang/triton) | 源码仓库 |
| Triton 教程 | [python/tutorials/](https://github.com/triton-lang/triton/tree/main/python/tutorials) | 官方 Jupyter 教程 |
| Triton Python API | [python-api/triton.html](https://triton-lang.org/main/python-api/triton.html) | `triton`、`triton.jit` 等 API |

### 社区资源

| 资源 | 链接 | 说明 |
|------|------|------|
| **Triton 论文（MAPL 2019）** | [PLDI 2019 页面](https://pldi19.sigplan.org/details/mapl-2019-papers/1/Triton-An-Intermediate-Language-and-Compiler-for-Tiled-Neural-Network-Computations) | Philippe Tillet、H. T. Kung、David Cox；*Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* |
| **Triton 会议演讲** | 各 AI 会议 Triton 相关议题 | 可在 YouTube / 会议站点检索 |
| **Triton 博客** | [OpenAI 介绍文章](https://openai.com/index/triton/) | 2021 年官方发布说明 |

## D.5 GPU 编程资源

| 资源 | 作者 / 来源 | 链接 | 说明 |
|------|-------------|------|------|
| **CUDA C++ Programming Guide** | NVIDIA | [docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | 官方 CUDA 编程指南 |
| **《Programming Massively Parallel Processors》**（第 4 版） | David B. Kirk、Wen-mei W. Hwu | — | 经典 GPU 体系结构教材（"PPoPP 书"） |
| **《Professional CUDA C Programming》** | John Cheng、Max Grossman、Ty McKercher | — | CUDA 实战 |
| **GPU Gems 系列** | NVIDIA | [developer.nvidia.com](https://developer.nvidia.com/gpugems/gpugems/contributors) | GPU 算法与优化案例集 |

## D.6 编译器通用资源

### 经典教材

| 书名 | 作者 | 说明 |
|------|------|------|
| **《Compilers: Principles, Techniques, and Tools》**（第 2 版，龙书） | Alfred V. Aho、Monica S. Lam、Ravi Sethi、Jeffrey D. Ullman | 编译原理经典 |
| **《Engineering a Compiler》**（第 3 版） | Keith D. Cooper、Linda Torczon | 实用编译器设计 |
| **《Modern Compiler Implementation in C》**（虎书 C 版） | Andrew W. Appel | 现代化编译器实现（另有 ML/Java 版） |

### 工程实践

| 资源 | 链接 | 说明 |
|------|------|------|
| **Compiler Explorer** | [godbolt.org](https://godbolt.org) | 在线查看编译结果 |
| **LLVM Weekly** | [llvmweekly.org](https://llvmweekly.org/) | Alex Bradbury 主编的 LLVM 周报 |
| **LLVM Discourse** | [discourse.llvm.org](https://discourse.llvm.org/) | 官方论坛（已取代原 llvm-dev 邮件列表主讨论区） |

## D.7 推荐学习路径（按优先级排列）

### 入门阶段（1-3 月）

```
优先级 1: 完成 MLIR Toy 教程
优先级 2: 读本书第 1-10 章
优先级 3: 读 C++ Primer 相关章节
优先级 4: 浏览 LLVM IR 语言参考
```

### 进阶阶段（3-6 月）

```
优先级 1: 读本书第 11-19 章
优先级 2: 阅读 Triton 核心 Pass 源码
优先级 3: 编写第一个 Triton Pass
优先级 4: 读 DialectConversion 文档
```

### 实战阶段（6 月+）

```
优先级 1: 读本书第 20-24 章
优先级 2: 实现第三方后端原型
优先级 3: 阅读 NVIDIA/AMD 后端完整代码
优先级 4: 参与 Triton 社区贡献
```

## D.8 常用开发社区

| 社区 | 链接 | 用途 |
|------|------|------|
| LLVM Discourse | [discourse.llvm.org](https://discourse.llvm.org/) | LLVM/MLIR 官方论坛 |
| LLVM Discord | [discord.gg/xS7Z362](https://discord.gg/xS7Z362) | 官方 Discord 邀请链接 |
| Triton GitHub Issues | [github.com/triton-lang/triton/issues](https://github.com/triton-lang/triton/issues) | 问题讨论与贡献 |
| Stack Overflow | [stackoverflow.com](https://stackoverflow.com/) | 通用编程问题 |
| CIRCT / MLIR 讨论 | [CIRCT Discourse](https://discourse.llvm.org/c/subprojects/circt/31) | CIRCT 子项目频道 |

---

**建议**：把 [godbolt.org](https://godbolt.org) 设为浏览器首页——它是最快上手 LLVM IR 的工具。
