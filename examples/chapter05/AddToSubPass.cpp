#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

bool replaceAddWithSub(Function &F) {
    if (F.isDeclaration())
        return false;

    bool changed = false;
    SmallVector<BinaryOperator *, 8> toReplace;

    for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
            if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
                if (BO->getOpcode() == Instruction::Add)
                    toReplace.push_back(BO);
            }
        }
    }

    for (BinaryOperator *BO : toReplace) {
        IRBuilder<> builder(BO);
        Value *sub = builder.CreateSub(BO->getOperand(0), BO->getOperand(1),
                                       BO->getName());
        BO->replaceAllUsesWith(sub);
        BO->eraseFromParent();
        changed = true;
    }

    if (changed)
        errs() << "Replaced add with sub in function " << F.getName() << "\n";
    return changed;
}

struct AddToSubPass : public PassInfoMixin<AddToSubPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        return replaceAddWithSub(F) ? PreservedAnalyses::none()
                                    : PreservedAnalyses::all();
    }
};

struct AddToSubLegacyPass : public FunctionPass {
    static char ID;
    AddToSubLegacyPass() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override {
        return replaceAddWithSub(F);
    }
};

} // namespace

char AddToSubLegacyPass::ID = 0;

static RegisterPass<AddToSubLegacyPass> X("add-to-sub",
                                          "Replace add with sub");

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "add-to-sub", LLVM_VERSION_STRING,
            [](PassBuilder &PB) {
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, FunctionPassManager &FPM,
                       ArrayRef<PassBuilder::PipelineElement>) {
                        if (Name == "add-to-sub") {
                            FPM.addPass(AddToSubPass());
                            return true;
                        }
                        return false;
                    });
            }};
}
