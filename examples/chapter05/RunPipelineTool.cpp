#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

struct MyPass : public PassInfoMixin<MyPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        errs() << "Running MyPass on " << F.getName() << "\n";
        return PreservedAnalyses::all();
    }
};

void runPipeline(Module &M) {
    LoopAnalysisManager LAM;
    FunctionAnalysisManager FAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    PassBuilder PB;
    PB.registerModuleAnalyses(MAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    FunctionPassManager FPM;
    FPM.addPass(MyPass());

    ModulePassManager MPM;
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
    MPM.run(M, MAM);
}

} // namespace

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input .ll file>"),
                                          cl::Required);

int main(int argc, char **argv) {
    cl::ParseCommandLineOptions(argc, argv, "chapter05 RunPipeline example\n");

    LLVMContext context;
    SMDiagnostic err;
    std::unique_ptr<Module> M = parseIRFile(InputFilename, err, context);
    if (!M) {
        err.print(argv[0], errs());
        return 1;
    }

    runPipeline(*M);
    return 0;
}
