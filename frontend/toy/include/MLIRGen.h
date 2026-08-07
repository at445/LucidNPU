#ifndef LUCID_FRONTEND_MLIR_GEN_H
#define LUCID_FRONTEND_MLIR_GEN_H

namespace mlir {
    class MLIRContext;
    template <typename OpTy> class OwningOpRef;
    class ModuleOp;
} // namespace mlir

namespace lucid_frontend { 
namespace toy {
class ModuleAST;
/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
    ModuleAST &moduleAST);
}} // namespace lucid_frontend::toy
#endif
