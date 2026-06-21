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
void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, double value) {
    auto dataType = RankedTensorType::get({}, odsBuilder.getF64Type());
    auto elemAttr = ::mlir::DenseElementsAttr::get(dataType, value);
    ConstantOp::build(odsBuilder, odsState, dataType, elemAttr);
}

/// The 'OpAsmParser' class provides a collection of methods for parsing
/// various punctuation, as well as attributes, operands, types, etc. Each of
/// these methods returns a `ParseResult`. This class is a wrapper around
/// `LogicalResult` that can be converted to a boolean `true` value on failure,
/// or `false` on success. This allows for easily chaining together a set of
/// parser rules. These rules are used to populate an `mlir::OperationState`
/// similarly to the `build` methods described above.
mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
    mlir::DenseElementsAttr value;
    if (parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseAttribute(value, "value", result.attributes))
        return failure();

    result.addTypes(value.getType());
    return success();
}

/// The 'OpAsmPrinter' class is a stream that allows for formatting
/// strings, attributes, operands, types, etc.
void ConstantOp::print(mlir::OpAsmPrinter &printer) {
    printer << " ";
    printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
    printer << getValue();
}

::llvm::LogicalResult ConstantOp::verify() {
    // If the return type of the constant is not an unranked tensor, the shape
    // must match the shape of the attribute holding the data.
    auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
    if (!resultType)
      return success();
  
    // Check that the rank of the attribute type matches the rank of the constant
    // result type.
    auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
    auto attrRank = attrType.getRank();
    if (attrRank != resultType.getRank()) {
      return emitOpError("return type must match the one of the attached value "
                         "attribute: ")
             << attrType.getRank() << " != " << resultType.getRank();
    }

    auto attrShape = attrType.getShape();
    auto resultShape = resultType.getShape();

    if (attrShape != resultShape) {
        return emitOpError(
            "return type shape mismatches its attribute at dimension ")
            << attrShape << " != " << resultShape;
    }

    return mlir::success();
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
