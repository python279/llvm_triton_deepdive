#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

void countAdds(Function &F) {
    int count = 0;
    for (BasicBlock &BB : F) {
        for (Instruction &I : BB) {
            if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
                if (BO->getOpcode() == Instruction::Add)
                    ++count;
            }
        }
    }
    errs() << "Function " << F.getName() << " has " << count
           << " add instructions\n";
}

struct CountAddPass : public PassInfoMixin<CountAddPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        countAdds(F);
        return PreservedAnalyses::all();
    }
};

struct CountAddLegacyPass : public FunctionPass {
    static char ID;
    CountAddLegacyPass() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override {
        countAdds(F);
        return false;
    }
};

} // namespace

char CountAddLegacyPass::ID = 0;

static RegisterPass<CountAddLegacyPass> X("count-add",
                                          "Count add instructions");

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "count-add", LLVM_VERSION_STRING,
            [](PassBuilder &PB) {
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, FunctionPassManager &FPM,
                       ArrayRef<PassBuilder::PipelineElement>) {
                        if (Name == "count-add") {
                            FPM.addPass(CountAddPass());
                            return true;
                        }
                        return false;
                    });
            }};
}
