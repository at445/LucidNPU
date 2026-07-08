#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/Pass.h"
#include <memory>
#include "Dialect.h"
#include "Ops.h"
#include "Passes.h"

#define DEBUG_TYPE "shape-inference"
using namespace mlir;
using namespace lucid_frontend;
#include "ShapeInfrenceOpInterface.cpp.inc"

namespace {

class ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, OperationPass<::lucid_frontend::FuncOp>> {

protected:
  void runOnOperation() override {
    auto f = getOperation();
    llvm::SmallPtrSet<Operation *, 16> worklist;
    f.walk([&](Operation * op){
        if(hasDynamicShapeReturn(op)) {
            worklist.insert(op);
        }
    });

    while (!worklist.empty()) {
        auto iter = llvm::find_if(worklist, isAllOperandsInfered);
        if (iter == worklist.end()) break;

        Operation * op = *iter;

        LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");
        if (auto inferOp = llvm::dyn_cast<ShapeInference>(*op)) {
            inferOp.inferShapes();
        } else {
            op->emitError("unable to infer shape of operation without shape "
                      "inference interface");
            return signalPassFailure();
        }
        worklist.erase(op);
    }

    if (!worklist.empty()) {
        f.emitError("Shape inference failed, ")
            << worklist.size() << " operations couldn't be inferred\n";
        signalPassFailure();
    }
    return;
  }

  static bool hasDynamicShapeReturn(Operation * op) {
    return llvm::any_of(op->getResultTypes(), [](Type typ){ 
        return llvm::isa<UnrankedTensorType>(typ);
    });
  }

  static bool isAllOperandsInfered(Operation * op) {
    return llvm::all_of(op->getOperandTypes(), [](Type typ){
        return llvm::isa<RankedTensorType>(typ);
    });
  }
};
}

std::unique_ptr<mlir::Pass> lucid_frontend::createShapeInferencePass() {
    return std::make_unique<ShapeInferencePass>();
}
