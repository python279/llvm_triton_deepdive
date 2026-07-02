# 第 22 章：实现 Dialect 与转换

> **本章目标**：实战实现第三方后端的 Dialect 定义和 TTG → 目标 Dialect 转换。
>
> 驯龙手记：现在是打造"龙鞍"的时候了。Dialect 定义是鞍具的设计图，
> TTG → 目标 Dialect 转换是将鞍具安装到龙身上的过程。这套鞍具（你的后端）
> 将让 Triton 这头海神可以骑到你的新龙背上，让它俩并肩作战。

---

## 配套示例

本章可运行代码位于 `books/examples/chapter22/`：

| 文件 | 章节 | 说明 |
|------|------|------|
| `sample_mygpu.mlir` | 22.x | 自定义 Dialect IR 样例 |

运行：

```bash
cd books/examples/chapter22
./run_examples.sh
```

一键验证全书示例：

```bash
cd books/examples && ./run_all.sh
```

---
## 22.1 场景假设

我们假设在 MyGPU 上实现策略 A（通过 LLVM 后端）。MyGPU 规格：

| 参数 | 值 |
|------|-----|
| warp 大小 | 16 |
| 每 CTA 最大 warp | 16 |
| 片上内存 (LDS) | 64KB |
| 矩阵乘 | 16×16×16 F32 单元 |
| LLVM 后端 | `mygpu` target 已存在 |

## 22.2 定义 MyGPU Dialect

```tablegen
// third_party/my_backend/include/Dialect/MyGPU/IR/MyGPUDialect.td
include "mlir/IR/OpBase.td"

def MyGPU_Dialect : Dialect {
    let name = "mygpu";
    let cppNamespace = "::mlir::mygpu";
    let dependentDialects = [
        "LLVM::LLVMDialect"
    ];
}

class MyGPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<MyGPU_Dialect, mnemonic, traits> {
}
```

```tablegen
// third_party/my_backend/include/Dialect/MyGPU/IR/MyGPUOps.td

def MyGPU_MatrixMulOp : MyGPU_Op<"matmul", [Pure, SameOperandsAndResultType]> {
    let summary = "MyGPU 16x16 matrix multiply";
    
    let arguments = (ins
        RankedTensorOf<[F32]>:$a,
        RankedTensorOf<[F32]>:$b,
        RankedTensorOf<[F32]>:$c
    );
    let results = (outs RankedTensorOf<[F32]>:$d);
    let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($a) `*` type($b) `->` type($d)";
    let hasVerifier = 1;
}

def MyGPU_BarrierOp : MyGPU_Op<"barrier", []> {
    let summary = "MyGPU thread barrier";
    let assemblyFormat = "attr-dict";
}
```

## 22.3 自定义 MyGPU 编码

```tablegen
// third_party/my_backend/include/Dialect/MyGPU/IR/MyGPUAttrDefs.td
def MyGPU_BlockedEncoding : AttrDef<MyGPU_Dialect, "Blocked"> {
    let mnemonic = "blocked";
    let parameters = (ins
        ArrayRefParameter<"unsigned">:$sizePerThread,
        ArrayRefParameter<"unsigned">:$threadsPerWarp,
        ArrayRefParameter<"unsigned">:$warpsPerCTA,
        ArrayRefParameter<"unsigned">:$order
    );
}
```

## 22.4 实现 TTG → MyGPU 转换

```cpp
// third_party/my_backend/lib/MyGPUToLLVM/ConvertTritonGPUToMyGPU.cpp

namespace {
using namespace mlir;
using namespace mlir::triton;

class TritonGPUToMyGPUPass 
    : public PassWrapper<TritonGPUToMyGPUPass, OperationPass<ModuleOp>> {
    
    void runOnOperation() override {
        ModuleOp module = getOperation();
        
        // 1. TypeConverter：TTG 编码 → MyGPU 编码
        TypeConverter typeConverter;
        typeConverter.addConversion([&](RankedTensorType type) -> Type {
            // 将 TTG 编码转为 MyGPU 编码
            auto encoding = type.getEncoding();
            auto blocked = dyn_cast<triton::gpu::BlockedEncodingAttr>(encoding);
            if (!blocked) return type;
            
            auto myEncoding = mygpu::BlockedEncodingAttr::get(
                type.getContext(),
                blocked.getSizePerThread(),
                blocked.getThreadsPerWarp(),
                blocked.getWarpsPerCTA(),
                blocked.getOrder()
            );
            return type.cloneWithEncoding(myEncoding);
        });
        
        // 2. ConversionTarget
        ConversionTarget target(getContext());
        target.addLegalDialect<mygpu::MyGPUDialect>();
        target.addIllegalDialect<triton::gpu::TritonGPUDialect>();
        
        // 3. Patterns
        RewritePatternSet patterns(&getContext());
        patterns.add<GenericOpPattern<ttg::LoadOp>>(typeConverter, &getContext());
        patterns.add<GenericOpPattern<ttg::StoreOp>>(typeConverter, &getContext());
        // ... 其他 pattern
        
        // 4. 执行
        if (failed(applyPartialConversion(module, target, std::move(patterns))))
            signalPassFailure();
    }
};
}
```

## 22.5 实现 DotOp → MyGPU 矩阵乘

```cpp
// 关键模式：将 tt.dot 映射为 mygpu.matmul

struct DotOpConversion : public OpConversionPattern<tt::DotOp> {
    LogicalResult
    matchAndRewrite(tt::DotOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        
        // 检查操作数类型
        auto aType = op.getA().getType().dyn_cast<RankedTensorType>();
        auto bType = op.getB().getType().dyn_cast<RankedTensorType>();
        
        // 获取形状
        auto aShape = aType.getShape();  // [M, K]
        auto bShape = bType.getShape();  // [K, N]
        
        if (aShape[0] == 16 && aShape[1] == 16 && bShape[1] == 16) {
            // 使用硬件矩阵乘指令
            rewriter.replaceOpWithNewOp<mygpu::MatrixMulOp>(
                op, op.getType(), adaptor.getA(), adaptor.getB(), adaptor.getC());
            return success();
        }
        
        // 兜底：展开为标量乘加
        // ...实现通用矩阵乘循环...
        return success();
    }
};
```

## 22.6 TTG → LLVM 降级的复用

如果目标有 LLVM 后端，MyGPU Dialect 可以降级到 LLVM Dialect：

```cpp
// third_party/my_backend/lib/MyGPUToLLVM/ConvertMyGPUToLLVM.cpp

struct MatrixMulOpConversion : public OpConversionPattern<mygpu::MatrixMulOp> {
    LogicalResult
    matchAndRewrite(mygpu::MatrixMulOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
        // 将 mygpu.matmul 降级为 LLVM IR
        // 可以生成 inline asm 或调用外部函数
        // ...
        rewriter.replaceOp(op, result);
        return success();
    }
};
```

## 22.7 编译管线的集成

```python
# third_party/my_backend/backend/compiler.py

class MyBackend(BaseBackend):
    def __init__(self, target):
        super().__init__(target)
        self.binary_ext = "mybin"
    
    @staticmethod
    def supports_target(target):
        return target.backend == 'mygpu'
    
    def load_dialects(self, ctx):
        # 加载自定义 Dialect
        mygpu.load_dialects(ctx)
    
    def add_stages(self, stages, options, language):
        if language == Language.TRITON:
            stages["ttir"] = lambda src, m: self.make_ttir(src, m, ...)
            stages["ttgir"] = lambda src, m: self.make_ttgir(src, m, ...)
        
        # 自定义阶段：TTG → MyGPU Dialect
        stages["mygpu"] = lambda src, m: self.make_mygpu_ir(src, m, ...)
        
        # 复用 LLVM 后端生成目标代码
        stages["llir"] = lambda src, m: self.make_llir(src, m, ...)
        stages["mybin"] = lambda src, m: self.make_mybin(src, m, ...)
    
    def make_mygpu_ir(self, mod, metadata, options):
        pm = ir.pass_manager(mod.context)
        # 运行 TTG → MyGPU 转换
        mygpu.passes.add_convert_ttgpuir_to_mygpu(pm)
        pm.run(mod, 'make_mygpu_ir')
        return mod
```

## 22.8 遇到的典型问题

| 问题 | 解决方案 |
|------|---------|
| TTG 编码不兼容 | 在 TypeConverter 中自定义编码映射 |
| dot 硬件不支持大矩阵 | 拆解为多个硬件操作（tiling） |
| 控制流不支持 Structurize | 确保 LLVM 后端支持所有控制流结构 |
| 内存地址空间映射 | 在 TypeConverter 中处理 `addrspace` 转换 |
| 缺少 PTX 等价物 | 用内联 ASM 或外部函数调用实现 |

---

## 📝 课后作业

### 作业 1：实现 Dialect 定义

按照 22.2 节的模板，为你的目标硬件定义一个完整的 Dialect（至少包含矩阵乘、加载/存储、同步三个 Op）。

### 作业 2：实现转换 Pass

参照 22.4-22.5 节的代码，实现 TTG → 你的 Dialect 的转换 Pass。至少覆盖：
1. `tt.load` → 你的 `load` Op
2. `tt.dot` → 你的 `matmul` Op
3. 类型转换（TTG 编码 → 你的编码）

### 作业 3：编译管线集成

参照 22.7 节，写出后端完整的 `add_stages` 实现，包括所有必要的 Pass 调用。

---

## 本章小结

- 实现第三方后端需要：自定义 Dialect（Op + Type + Attr）+ 转换 Pass
- TTG → 目标 Dialect 的转换是核心工作量
- TypeConverter 处理编码映射，ConversionPattern 处理 Op 映射
- `tt.dot` 的降级是关键——尽量映射到硬件矩阵乘指令
- MyGPU Dialect 再降级到 LLVM Dialect 以复用 LLVM 后端
- 第 23 章继续讨论代码生成和测试
