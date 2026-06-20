#include "Dialect.h"
#include "Ops.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <string>
using namespace mlir;
using namespace toy;

#include "Dialect.cpp.inc"

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
}

void FuncOp::build(OpBuilder &builder, OperationState &state, StringRef name, FunctionType type) {
  OpBuilder::InsertionGuard guard(builder);
  state.addAttribute("sym_name", builder.getStringAttr(name));
  Region *body = state.addRegion();
  Block *entry = builder.createBlock(body);
  for (Type argType : type.getInputs())
    entry->addArgument(argType, state.location);
}

#define GET_OP_CLASSES
#include "Ops.cpp.inc"
