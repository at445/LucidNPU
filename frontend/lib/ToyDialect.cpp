#include "Dialect.h"
#include "Ops.h"

#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace toy;

#include "Dialect.cpp.inc"

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "Ops.cpp.inc"
