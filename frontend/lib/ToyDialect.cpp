#include "Dialect.h"
#include "Ops.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>
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

std::optional<RankedTensorType> inferMatmulResultType(RankedTensorType lhs, 
    RankedTensorType rhs,Type elementType) 
{
    auto lShape = lhs.getShape();
    auto rShape = rhs.getShape();
    auto lRank = lhs.getRank();
    auto rRank = rhs.getRank();
    if ((lRank < 1 || rRank < 1) || (lRank < 2 && rRank < 2))
        return std::nullopt;
    
    if ((lRank > rRank ? lRank - rRank : rRank - lRank) > 1)
        return std::nullopt;

    auto compatibleJudge = [] (int64_t a, int64_t b) -> bool {
        return a == ShapedType::kDynamic || b == ShapedType::kDynamic || a == b;
    };

    ArrayRef<int64_t> lBatch, rBatch;
    llvm::SmallVector<int64_t, 2> concat;
    if (lRank > rRank) {                 // (*, M x K) @ (*, K)     -> (*, M)
        lBatch = lShape.drop_back(2);
        rBatch = rShape.drop_back(1);
        if (!compatibleJudge(lShape[lRank - 1], rShape[rRank -1]))
            return std::nullopt;
        concat.push_back(lShape[lRank-2]);
    } else if (lRank == rRank) {         // (*, M x K) @ (*, K x N) -> (*, M x N)
        lBatch = lShape.drop_back(2);
        rBatch = rShape.drop_back(2);
        if (!compatibleJudge(lShape[lRank - 1], rShape[rRank -2])) 
            return std::nullopt;
        concat.push_back(lShape[lRank-2]);
        concat.push_back(rShape[rRank-1]);
    } else {                             // (*, K)     @ (*, K x N) -> (*, N)
        lBatch = lShape.drop_back(1);
        rBatch = rShape.drop_back(2);
        if (!compatibleJudge(lShape[lRank - 1], rShape[rRank -2])) 
            return std::nullopt;
        concat.push_back(rShape[rRank-1]);
    }

    for (int i = 0 ; i<lBatch.size();  i++) {
        if (!compatibleJudge(lShape[i], rShape[i])) 
            return std::nullopt;
    }

    SmallVector<int64_t, 4> outShape(lBatch.begin(), lBatch.end());
    outShape.append(concat.begin(), concat.end());
    return RankedTensorType::get(outShape, elementType);
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

void MatrixMulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                      mlir::Value lhs, mlir::Value rhs) {
    auto lhsType = llvm::dyn_cast<mlir::RankedTensorType>(lhs.getType());
    auto rhsType = llvm::dyn_cast<mlir::RankedTensorType>(rhs.getType());

    if (!lhsType || !rhsType) {
        state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
        
    }
    auto resultType = inferMatmulResultType(lhsType, rhsType, rhsType.getElementType());
    if (!resultType) {
        state.addTypes(RankedTensorType::get({}, builder.getF64Type()));
    } else {
        state.addTypes(*resultType);
    }
    state.addOperands({lhs, rhs});
}



::llvm::LogicalResult MatrixMulOp::verify() {
    auto resultType = dyn_cast<RankedTensorType>(getResult().getType());
    auto lhsType = llvm::dyn_cast<mlir::RankedTensorType>(getOperands()[0].getType());
    auto rhsType = llvm::dyn_cast<mlir::RankedTensorType>(getOperands()[1].getType());

    if (!resultType || !lhsType || !rhsType)
        return success();

    auto inferred = inferMatmulResultType(lhsType, rhsType, lhsType.getElementType());

    if (!inferred)
        return emitOpError("incompatible shapes for matrix multiplication: ")
            << "\n" << lhsType << " x " << rhsType;
    if (*inferred != resultType)
        return emitOpError("result type ") << resultType
            << " does not match inferred type " << *inferred;
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
