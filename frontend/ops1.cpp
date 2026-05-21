/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Ops.td                                                               *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::lucid_frontend::ConstantOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace lucid_frontend {
} // namespace lucid_frontend
namespace lucid_frontend {

//===----------------------------------------------------------------------===//
// ::lucid_frontend::ConstantOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
} // namespace detail
ConstantOpAdaptor::ConstantOpAdaptor(ConstantOp op) : ConstantOpGenericAdaptor(op->getOperands(), op) {}

::llvm::LogicalResult ConstantOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes) {
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::llvm::LogicalResult ConstantOp::verifyInvariantsImpl() {
  return ::mlir::success();
}

::llvm::LogicalResult ConstantOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace lucid_frontend
MLIR_DEFINE_EXPLICIT_TYPE_ID(::lucid_frontend::ConstantOp)


#endif  // GET_OP_CLASSES

