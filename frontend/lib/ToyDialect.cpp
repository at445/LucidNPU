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
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/PatternMatch.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
using namespace mlir;
using namespace lucid_frontend;
#include "Dialect.cpp.inc"
#include "ToyInlineInterface.hpp"
void ToyDialect::initialize() {
    addOperations<
        #define GET_OP_LIST
        #include "Ops.cpp.inc"
    >();
    addInterface<lucid_frontend::ToyInlinerInterface>();
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
//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//
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

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//
::llvm::LogicalResult ReturnOp::verify() {
    if (ReturnSize() > 1) {
        return emitOpError("The result type of return statement \
            can have at most one return value or none.");
    }

    auto funcType = getParentOp().getFunctionType();
    if (funcType.getResults().size() != ReturnSize())
    {
        return emitOpError(
            llvm::formatv("return arity {0} on '{1}' does not match function result arity {2}.",
            ReturnSize(), getParentOp().getSymName(), funcType.getResults().size()));
    }
    
    if ((ReturnSize() == 1) && (getOperandTypes().front() != funcType.getResults().front())) {
        return emitOpError("The result type of return statement ") << getLoc() 
            << "\n is not matched with the type of function " << getParentOp().getSymName() 
            << " on " << getParentOp()->getLoc()
            << "\n Type of return statement" << getOperandTypes().front()
            << "\n Type of function return" << funcType.getResults().front();
    }

    return mlir::success();
}


//===----------------------------------------------------------------------===//
// MatrixMulOp
//===----------------------------------------------------------------------===//
void MatrixMulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                      mlir::Value lhs, mlir::Value rhs) {

    auto lhsType = llvm::dyn_cast<mlir::RankedTensorType>(lhs.getType());
    auto rhsType = llvm::dyn_cast<mlir::RankedTensorType>(rhs.getType());

    if (!lhsType || !rhsType) {
        state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
    } else {
        auto inferred = inferMatmulResultType(lhsType, rhsType, lhsType.getElementType());
        if (!inferred) {
            state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
        }
        state.addTypes(*inferred);
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

void MatrixMulOp::inferShapes() {
    auto lhsType = llvm::dyn_cast<mlir::RankedTensorType>(getOperands()[0].getType());
    auto rhsType = llvm::dyn_cast<mlir::RankedTensorType>(getOperands()[1].getType());

    if (!lhsType || !rhsType) return;

    auto inferred = inferMatmulResultType(lhsType, rhsType, lhsType.getElementType());
    if (!inferred) return;

    getResult().setType(*inferred);
}

::llvm::LogicalResult ReshapeOp::verify() {
    auto argTyp = mlir::dyn_cast<RankedTensorType>(getInput().getType());
    auto retType = mlir::dyn_cast<RankedTensorType>(getResult().getType());
    
    if (argTyp && retType) { // Neither of them is unranked type 
        if (argTyp.hasStaticShape() && retType.hasStaticShape()) { // Both are fully static types. 
            auto argShape = argTyp.getShape();
            auto retShape = retType.getShape();
            auto argSize = std::accumulate(argShape.begin(), argShape.end(), 
                1, std::multiplies<int64_t>());
            auto retSize = std::accumulate(retShape.begin(), retShape.end(), 
                1, std::multiplies<int64_t>());
            if (argSize != retSize) {
                return emitOpError("incompatible reshape: ")
                    << "\n" << argTyp << " reshape to " << retType;
            }
        }
    } 

    return mlir::success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
    llvm::StringRef name, mlir::FunctionType type,
    llvm::ArrayRef<mlir::NamedAttribute> attrs) {
    // FunctionOpInterface provides a convenient `build` method that will populate
    // the state of our FuncOp, and create an entry block.
    buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
                 mlir::OperationState &result) {
    // Dispatch to the FunctionOpInterface provided utility method that parses the
    // function operation.
    auto buildFuncType =
    [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
    llvm::ArrayRef<mlir::Type> results,
    mlir::function_interface_impl::VariadicFlag,
    std::string &) { return builder.getFunctionType(argTypes, results); };

    return mlir::function_interface_impl::parseFunctionOp(
    parser, result, /*allowVariadic=*/false,
    getFunctionTypeAttrName(result.name), buildFuncType,
    getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::print(mlir::OpAsmPrinter &p) {
    // Dispatch to the FunctionOpInterface provided utility method that prints the
    // function operation.
    mlir::function_interface_impl::printFunctionOp(
    p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
    getArgAttrsAttrName(), getResAttrsAttrName());
}

void TransposeOp::inferShapes() {
    auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
    if (!inputType || (inputType.getRank() < 2)) return;

    llvm::SmallVector<int64_t, 4> transposed(inputType.getShape());
    std::swap(transposed[transposed.size()-1], transposed[transposed.size()-2]);

    
    getResult().setType(mlir::RankedTensorType::get(transposed, inputType.getElementType()));
}

void CastOp::inferShapes() {
    auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
    if (!inputType) return;
    getResult().setType(inputType);
}

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Set the callee for the generic call operation, this is required by the call
/// interface.
void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range GenericCallOp::getArgOperands() { 
    return getInputs(); 
}

/// Get the argument operands to the called function as a mutable range, this is
/// required by the call interface.
MutableOperandRange GenericCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}
namespace {
    /// Include the patterns defined in the Declarative Rewrite framework.
    #include "CustomizedCanonicalize.inc"
} // namespace

void TransposeOp::getCanonicalizationPatterns(mlir::RewritePatternSet&results, mlir::MLIRContext* context) {
    results.add<TransposeTransposeOptPattern>(context);
}

void ReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet&results, mlir::MLIRContext* context) {
    results.add<ReshapeReshapeOptPattern>(context);
    results.add<RedundantReshapeOptPattern>(context);
    results.add<FoldReshapeConstantOptPattern>(context);
}

bool CastOp::areCastCompatible(::mlir::TypeRange inputs, ::mlir::TypeRange outputs) {
    if (inputs.size() != outputs.size()) return false;
    return llvm::all_of(llvm::zip(inputs, outputs), [](auto pair){
        auto [input, output] = pair;
        TensorType intyp = llvm::dyn_cast<TensorType>(input);
        TensorType outtyp = llvm::dyn_cast<TensorType>(output);
        if (!intyp || !outtyp || intyp.getElementType() != outtyp.getElementType())
            return false;
        if (!intyp.hasRank() || !outtyp.hasRank()) 
            return true;
        return intyp == outtyp;
    });
}

// Implementations of all Ops on toy dialect
#define GET_OP_CLASSES
#include "Ops.cpp.inc"

