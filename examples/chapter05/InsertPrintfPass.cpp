#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

bool insertPrintf(Function &F) {
    if (F.isDeclaration())
        return false;

    BasicBlock &entry = F.getEntryBlock();
    IRBuilder<> builder(&entry, entry.begin());

    Module *M = F.getParent();
    FunctionCallee printfFunc = M->getOrInsertFunction(
        "printf",
        FunctionType::get(
            IntegerType::getInt32Ty(M->getContext()),
            PointerType::get(M->getContext(), 0),
            true));

    Value *formatStr =
        builder.CreateGlobalString("Entering function: %s\n");
    builder.CreateCall(printfFunc,
                       {formatStr,
                        builder.CreateGlobalString(F.getName())});
    return true;
}

struct InsertPrintfPass : public PassInfoMixin<InsertPrintfPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        return insertPrintf(F) ? PreservedAnalyses::none()
                               : PreservedAnalyses::all();
    }
};

struct InsertPrintfLegacyPass : public FunctionPass {
    static char ID;
    InsertPrintfLegacyPass() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override { return insertPrintf(F); }
};

} // namespace

char InsertPrintfLegacyPass::ID = 0;

static RegisterPass<InsertPrintfLegacyPass> X(
    "insert-printf", "Insert printf at function entry");

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "insert-printf", LLVM_VERSION_STRING,
            [](PassBuilder &PB) {
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, FunctionPassManager &FPM,
                       ArrayRef<PassBuilder::PipelineElement>) {
                        if (Name == "insert-printf") {
                            FPM.addPass(InsertPrintfPass());
                            return true;
                        }
                        return false;
                    });
            }};
}
