#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

void countInstructions(Function &F) {
    int count = 0;
    for (BasicBlock &BB : F)
        for (Instruction &I : BB)
            ++count;

    errs() << "Function " << F.getName() << " has " << count
           << " instructions\n";
}

// New PM（LLVM 18+ 的 opt 使用此路径）
struct CountInstructionsPass : public PassInfoMixin<CountInstructionsPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        countInstructions(F);
        return PreservedAnalyses::all();
    }
};

// Legacy PM（LLVM 17 及更早的 opt -load -count-instructions）
struct CountInstructionsLegacyPass : public FunctionPass {
    static char ID;
    CountInstructionsLegacyPass() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override {
        countInstructions(F);
        return false;
    }
};

} // namespace

char CountInstructionsLegacyPass::ID = 0;

static RegisterPass<CountInstructionsLegacyPass> X(
    "count-instructions", "Count instructions in functions");

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "count-instructions", LLVM_VERSION_STRING,
            [](PassBuilder &PB) {
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, FunctionPassManager &FPM,
                       ArrayRef<PassBuilder::PipelineElement>) {
                        if (Name == "count-instructions") {
                            FPM.addPass(CountInstructionsPass());
                            return true;
                        }
                        return false;
                    });
            }};
}
