#include "Dialect.h"
#include "Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Casting.h"
using namespace mlir;
using namespace lucid_frontend;
std::optional<RankedTensorType> inferMatmulResultType(RankedTensorType lhs, 
    RankedTensorType rhs,Type elementType);
namespace {
    /// Include the patterns defined in the Declarative Rewrite framework.
    #include "CustomizedCanonicalize.inc"
} // namespace
// replace the operation of transposeOp on a vector to reshapeOp
// for example:
//      %1 = toy.transpose %0 : tensor<6xf64> to tensor<6xf64>
// to:
//      %1 = toy.reshape %0 : tensor<6xf64> to tensor<6x1xf64>
class TansposeVector2Reshape: public OpRewritePattern<lucid_frontend::TransposeOp> {
    using OpRewritePattern<lucid_frontend::TransposeOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(lucid_frontend::TransposeOp op,
                                        PatternRewriter &rewriter) const final {
        auto operandTyp = llvm::dyn_cast<RankedTensorType>(op->getOperandTypes().front());
        if (!operandTyp) return mlir::failure();

        if (operandTyp.getRank() ==1) {
            auto newTyp = mlir::RankedTensorType::get({operandTyp.getShape().front(), 1}, operandTyp.getElementType());
            rewriter.replaceOpWithNewOp<lucid_frontend::ReshapeOp>(op, newTyp, op->getOperands().front());
        }
        return mlir::success();
    }
};

// Implicit promotion in vector and matrix multiplication to [1, n] matrix multiplication
// for example:
//      %2 = toy.matrix_mul %0, %1 : tensor<6xf64>, tensor<..xf64> -> tensor<..xf64>
// to:
//      %3 = toy.reshape %0 : tensor<6xf64> to tensor<1x6xf64>
//      %2 = toy.matrix_mul %3, %1 : tensor<1x6xf64>, tensor<..xf64> -> tensor<..xf64>
class PromoteVectorMaxtrix2MaxtrixMaxtrix : public OpRewritePattern<lucid_frontend::MatrixMulOp> {
    using OpRewritePattern<lucid_frontend::MatrixMulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(lucid_frontend::MatrixMulOp op,
                                        PatternRewriter &rewriter) const final {
        auto lhsOperandTyp = llvm::dyn_cast<RankedTensorType>(op.getLhs().getType());
        if (!lhsOperandTyp) return mlir::failure();

        if (lhsOperandTyp.getRank() ==1) {
            auto newTyp = mlir::RankedTensorType::get({1, lhsOperandTyp.getShape().front()}, lhsOperandTyp.getElementType());
            auto reshapeOp = rewriter.create<lucid_frontend::ReshapeOp>(op->getLoc(), newTyp, op.getLhs());
            
            Type resultTyp = op.getType();
            if (auto rhsOperandTyp = llvm::dyn_cast<RankedTensorType>(op.getRhs().getType())) {
                auto inferred = inferMatmulResultType(newTyp, rhsOperandTyp, lhsOperandTyp.getElementType());
                if (inferred) {
                    resultTyp = *inferred;
                }
            }
            
            auto newOp = rewriter.create<lucid_frontend::MatrixMulOp>(
                op.getLoc(), resultTyp, reshapeOp, op.getRhs());
            rewriter.replaceOp(op, newOp);
            
            return mlir::success();
        }
        return mlir::failure();
    }
};

// Implicit promotion in vector and matrix multiplication to [1, n] matrix multiplication
// for example:
//      %2 = toy.matrix_mul %0, %1 : tensor<..xf64>, tensor<6xf64> -> tensor<..xf64>
// to:
//      %3 = toy.reshape %0 : tensor<6xf64> to tensor<6x1xf64>
//      %2 = toy.matrix_mul %3, %1 : tensor<..xf64>, tensor<6x1xf64> -> tensor<..xf64>
class PromoteMaxtrixVector2MaxtrixMaxtrix : public OpRewritePattern<lucid_frontend::MatrixMulOp> {
    using OpRewritePattern<lucid_frontend::MatrixMulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(lucid_frontend::MatrixMulOp op,
                                        PatternRewriter &rewriter) const final {
        auto rhsOperandTyp = llvm::dyn_cast<RankedTensorType>(op.getRhs().getType());
        if (!rhsOperandTyp) return mlir::failure();

        if (rhsOperandTyp.getRank() ==1) {
            auto newTyp = mlir::RankedTensorType::get({rhsOperandTyp.getShape().front(), 1}, rhsOperandTyp.getElementType());
            auto reshapeOp = rewriter.create<lucid_frontend::ReshapeOp>(op->getLoc(), newTyp, op.getRhs());

            Type resultTyp = op.getType();
            if (auto lhsOperandTyp = llvm::dyn_cast<RankedTensorType>(op.getLhs().getType())) {
                auto inferred = inferMatmulResultType(lhsOperandTyp, newTyp, newTyp.getElementType());
                if (inferred) {
                    resultTyp = *inferred;
                }
            } 
            
            auto newOp = rewriter.create<lucid_frontend::MatrixMulOp>(
                op.getLoc(), resultTyp, op.getLhs(), reshapeOp);
            rewriter.replaceOp(op, newOp);
            
            return mlir::success();
        }
        return mlir::failure();
    }
};


void TransposeOp::getCanonicalizationPatterns(mlir::RewritePatternSet&results, mlir::MLIRContext* context) {
    results.add<TransposeTransposeOptPattern>(context);
    results.add<TansposeVector2Reshape>(context);
}

void ReshapeOp::getCanonicalizationPatterns(mlir::RewritePatternSet&results, mlir::MLIRContext* context) {
    results.add<ReshapeReshapeOptPattern>(context);
    results.add<RedundantReshapeOptPattern>(context);
    results.add<FoldReshapeConstantOptPattern>(context);
}
void MatrixMulOp::getCanonicalizationPatterns(mlir::RewritePatternSet&results, mlir::MLIRContext* context) {
    results.add<PromoteVectorMaxtrix2MaxtrixMaxtrix>(context);
    results.add<PromoteMaxtrixVector2MaxtrixMaxtrix>(context);
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
