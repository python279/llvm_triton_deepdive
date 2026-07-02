#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

int main() {
    mlir::MLIRContext context;
    context.loadDialect<mlir::arith::ArithDialect>();
    context.loadDialect<mlir::func::FuncDialect>();

    mlir::OpBuilder builder(&context);
    auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    builder.setInsertionPointToStart(module.getBody());

    auto funcType = builder.getFunctionType(
        {builder.getI32Type(), builder.getI32Type()}, {builder.getI32Type()});
    auto func = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(), "add", funcType);

    auto *entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    mlir::Value lhs = entryBlock->getArgument(0);
    mlir::Value rhs = entryBlock->getArgument(1);
    auto addOp =
        builder.create<mlir::arith::AddIOp>(builder.getUnknownLoc(), lhs, rhs);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc(),
                                           addOp.getResult());

    module.dump();
    return 0;
}
