#ifndef LUCID_FRONT_END_PASSES_H
#define LUCID_FRONT_END_PASSES_H

#include <memory>
#include "mlir/Pass/Pass.h"


namespace lucid_frontend {
    std::unique_ptr<mlir::Pass> createShapeInferencePass();
    std::unique_ptr<mlir::Pass> createAffineLoweringPass();
    std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
} // namespace lucid_frontend

#endif // LUCID_FRONT_END_PASSES_H
