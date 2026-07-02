#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

void countFunctions(Module &M) {
    int count = 0;
    for (Function &F : M) {
        if (!F.isDeclaration())
            ++count;
    }
    errs() << "Module has " << count << " defined functions\n";
}

struct CountFunctionsPass : public PassInfoMixin<CountFunctionsPass> {
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
        countFunctions(M);
        return PreservedAnalyses::all();
    }
};

struct CountFunctionsLegacyPass : public ModulePass {
    static char ID;
    CountFunctionsLegacyPass() : ModulePass(ID) {}

    bool runOnModule(Module &M) override {
        countFunctions(M);
        return false;
    }
};

} // namespace

char CountFunctionsLegacyPass::ID = 0;

static RegisterPass<CountFunctionsLegacyPass> X(
    "count-functions", "Count defined functions in a module");

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "count-functions", LLVM_VERSION_STRING,
            [](PassBuilder &PB) {
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, ModulePassManager &MPM,
                       ArrayRef<PassBuilder::PipelineElement>) {
                        if (Name == "count-functions") {
                            MPM.addPass(CountFunctionsPass());
                            return true;
                        }
                        return false;
                    });
            }};
}
