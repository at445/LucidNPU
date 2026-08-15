
#include "Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "Dialect.h"
#include "Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
using namespace ::mlir;
using namespace lucid_frontend::toy;
namespace {

struct ReturnOpLowering: public OpRewritePattern<ReturnOp> {
    using OpRewritePattern<ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ReturnOp op,
                                        PatternRewriter &rewriter) const final {
        if (op->getOperands().size() > 0) {
            return mlir::failure();
        }

        rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
        return mlir::success();
    }
};

class PrintOpLowering : public OpConversionPattern<PrintOp> {
    using OpConversionPattern<PrintOp>::OpConversionPattern;
public:
    LogicalResult matchAndRewrite(PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final 
    {
        auto inputType = llvm::dyn_cast<RankedTensorType>(op.getInput().getType());
        if (!inputType) {
            return rewriter.notifyMatchFailure(op, "expected ranked tensor input");
        }
        
        ModuleOp parentModule = op->getParentOfType<ModuleOp>();
        auto loc = op->getLoc();

        // 1. Get-or-insert the runtime print prototype for this rank, e.g.
        //    func.func @lucid_print_2d(tensor<?x?xf64>) -> ()
        auto printRef = getOrInsertPrintFuncOp(rewriter, inputType.getRank(),
                                               parentModule);

        // 2. Cast the static-shape input to the prototype's dynamic shape and
        //    call it. The call produces no results; the toy.print is erased.
        Value input = adaptor.getInput();
        auto dynType = getDynamicTensorType(rewriter.getContext(),
                                            inputType.getRank());
        if (input.getType() != dynType) {
            // create an tensor.cast op if not match
            input = tensor::CastOp::create(rewriter, loc, dynType, input);
        }
        func::CallOp::create(rewriter, loc, printRef, TypeRange{}, input);
        rewriter.eraseOp(op);
        return mlir::success();
    }

private:
    static RankedTensorType getDynamicTensorType(MLIRContext *context,
                                                 int64_t rank) {
        return RankedTensorType::get(
            SmallVector<int64_t>(rank, ShapedType::kDynamic),
            Float64Type::get(context));
    }

    /// Get-or-insert the runtime print declaration `@lucid_print_<rank>d`
    /// on the parent module. func::FuncOp carries the Symbol trait, so the
    /// module's symbol table sees the new symbol as soon as it is created.
    FlatSymbolRefAttr getOrInsertPrintFuncOp(PatternRewriter &rewriter,
                                             int64_t rank, ModuleOp module) const {
        std::string printName = "lucid_print_" + std::to_string(rank) + "d";
        auto context = module.getContext();
        if (module.lookupSymbol<func::FuncOp>(printName)) {
            return SymbolRefAttr::get(context, printName);
        }

        // Insert the print declaration into the body of the parent module.
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        auto funcType = FunctionType::get(context,
                                          {getDynamicTensorType(context, rank)},
                                          /*results=*/{});
        auto funcOp = func::FuncOp::create(rewriter, module->getLoc(),
                                           printName, funcType);
        // A bodiless func.func is a declaration and must not be public.
        funcOp.setVisibility(SymbolTable::Visibility::Private);
        return SymbolRefAttr::get(context, printName);
    }
};

class TransposeOpLowering : public OpConversionPattern<TransposeOp> {
    using OpConversionPattern<TransposeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final 
    {
        auto resultType = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
        if (!resultType) {
            return rewriter.notifyMatchFailure(op, "expected ranked tensor result");
        }

        auto empty = tensor::EmptyOp::create(rewriter, op->getLoc(),
                                        resultType.getShape(),
                                        resultType.getElementType());


        int64_t rank = resultType.getRank();
        SmallVector<int64_t> permutation;
        llvm::append_range(permutation, llvm::seq<int64_t>(0, rank));
        if (rank > 1) {
            std::swap(permutation[rank - 1], permutation[rank - 2]);
        }

        auto transpose = linalg::TransposeOp::create(
            rewriter, op.getLoc(), adaptor.getInput(), empty, permutation);
        rewriter.replaceOp(op, transpose);
        return mlir::success();
    }
};


class MatrixMulOpLowering : public OpConversionPattern<MatrixMulOp> {
    using OpConversionPattern<MatrixMulOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(MatrixMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final 
    {
        auto resultType = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
        if (!resultType) {
            return rewriter.notifyMatchFailure(op, "expected ranked tensor result");
        }
 
        auto attr = rewriter.getZeroAttr(resultType.getElementType());
        Value fillValue = arith::ConstantOp::create(
            rewriter, op->getLoc(), resultType.getElementType(), attr);
        auto empty = tensor::EmptyOp::create(rewriter, op->getLoc(),
                                        resultType.getShape(),
                                        resultType.getElementType());
        auto filled =  linalg::FillOp::create(rewriter, op->getLoc(), fillValue, empty->getResults());

        auto matmul = linalg::MatmulOp::create(rewriter, op->getLoc(), adaptor.getOperands(), filled->getResults());
            
        rewriter.replaceOp(op, matmul);
        return mlir::success();
    }
};


struct FuncOpLowering : public OpConversionPattern<FuncOp> {
  using OpConversionPattern<FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We only lower the main function as we expect that all other functions
    // have been inlined.
    if (op.getName() != "main")
      return failure();

    // Verify that the given main has no inputs and results.
    if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "expected 'main' to have 0 inputs and 0 results";
      });
    }

    // Create a new non-toy function, with the same region.
    auto func = mlir::func::FuncOp::create(rewriter, op.getLoc(), op.getName(),
                                                    op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};


struct ConstantOpLowering : public OpRewritePattern<ConstantOp> {
    using OpRewritePattern<ConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(ConstantOp op,
                                    PatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, 
            op->getOpResult(0).getType(), op.getValue());
        return success();
    }
};

struct ReshapeOpLowering : public OpConversionPattern<ReshapeOp> {
    using OpConversionPattern<ReshapeOp>::OpConversionPattern;
    LogicalResult
    matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
        auto inTyp = llvm::cast<TensorType>(op->getOperand(0).getType());
        auto outTyp = op.getResult().getType();
        if(!inTyp) {
            return rewriter.notifyMatchFailure(op, [op](Diagnostic &diag) {
                diag << "type inllegl" << op->getOperand(0).getType();
            });
        }
        
        if (inTyp == outTyp) {
            rewriter.replaceOp(op, adaptor.getInput());
            return mlir::success();
        }

        auto reassIndices = getReassociationIndicesForReshape(inTyp, outTyp);
        // can be converted to an expand or a collapse
        if (reassIndices.has_value()) {
            if (inTyp.getRank() > outTyp.getRank()) {
                rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
                    op, outTyp, adaptor.getInput(), reassIndices.value());
                
            } else {
                rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
                    op, outTyp, adaptor.getInput(), reassIndices.value());
            }
            return mlir::success();
        }

        auto expandIndices = getExpandUnionPoint(inTyp.getShape(), outTyp.getShape());
        // can be converted to an expand and then a collapse
        if (expandIndices.has_value()) {
            auto boundingType = mlir::RankedTensorType::get(expandIndices->shape, rewriter.getF64Type());
            auto expandOp = tensor::ExpandShapeOp::create(rewriter, op->getLoc(), 
            boundingType, adaptor.getInput(), expandIndices->srcReassociation.value());
            auto collapseOp = tensor::CollapseShapeOp::create(rewriter, op->getLoc(),
            outTyp, expandOp->getResult(0), expandIndices->targetReassociation.value());
            rewriter.replaceOp(op, collapseOp);
            return mlir::success();
        }

        auto unionPoint = getCollapseUnionPoint(inTyp.getShape(), outTyp.getShape());
        if (!unionPoint.has_value()) {
            return rewriter.notifyMatchFailure(op, 
                "The total number of output and input elements does not match. ");
        } else {
            auto boundingType = mlir::RankedTensorType::get(unionPoint->shape, rewriter.getF64Type());
            auto collapseOp = tensor::CollapseShapeOp::create(rewriter, op->getLoc(), 
            boundingType, adaptor.getInput(), unionPoint->srcReassociation.value());
            auto expandOp = tensor::ExpandShapeOp::create(rewriter, op.getLoc(),
            outTyp, collapseOp.getResult(), unionPoint->targetReassociation.value());
            rewriter.replaceOp(op, expandOp);
            return mlir::success();
        }     
    }
private:
    struct expandResult {
        std::optional<SmallVector<ReassociationIndices>> srcReassociation;
        llvm::SmallVector<int64_t, 4> shape;
        std::optional<SmallVector<ReassociationIndices>> targetReassociation;
        expandResult() : srcReassociation({}), targetReassociation({}) {};
    };

    std::optional<struct expandResult>
    getExpandUnionPoint(llvm::ArrayRef<int64_t> iShape, llvm::ArrayRef<int64_t> jShape) const {
        llvm::SmallVector<int64_t, 4> iTemp(iShape);
        llvm::SmallVector<int64_t, 4> jTemp(jShape);
        struct expandResult ret;
        int64_t i = 0, j = 0;
        while (i < iTemp.size() && j < jTemp.size()) {
            if (iTemp[i] == jTemp[j]) {
                ret.shape.push_back(iTemp[i]);
                i++;
                j++;
            } else if (iTemp[i] > jTemp[j] && iTemp[i] % jTemp[j] == 0) {
                ret.shape.push_back(jTemp[j]);
                iTemp[i] = iTemp[i] / jTemp[j];
                j++;
                
            } else if (iTemp[i] < jTemp[j] && jTemp[j] % iTemp[i] == 0) {
                ret.shape.push_back(iTemp[i]);
                jTemp[j] =  jTemp[j] / iTemp[i];
                i++;
            } else {
                return {};
            }
        }

        while (i < iTemp.size() && iTemp[i] == 1) {
            ret.shape.push_back(iTemp[i]);
            i++;
        }
        while (j < jTemp.size() && jTemp[j] == 1) {
            ret.shape.push_back(jTemp[j]);
            j++;
        }
        if (i != iTemp.size() || j != jTemp.size()) {
            return {};
        }

        ret.srcReassociation = computeReassociation(iShape, ret.shape);
        ret.targetReassociation = computeReassociation(jShape, ret.shape);
        
        return ret;
    }

    static SmallVector<ReassociationIndices>
    computeReassociation(ArrayRef<int64_t> lowDim, ArrayRef<int64_t> highDim) {
        int64_t i = 0, j = 0;
        SmallVector<ReassociationIndices> ret;
        while (i < lowDim.size() && j < highDim.size()) {
            ReassociationIndices group;
            if (lowDim[i] == highDim[j]) {
                group.push_back(j);
                i++;
                j++;
            } else if (lowDim[i] > highDim[j]) {
                int product = 1;
                while (j < highDim.size() && (product < lowDim[i])) {
                    group.push_back(j);
                    product = product * highDim[j];
                    j++;
                }
                i++;
            } else { // cannot be here
                llvm::llvm_unreachable_internal("The total size of the dim corresponding to the low-dim array \
                    should be greater than or equal to that of the high-dim array.");
            }
            ret.push_back(group);
        }

        while (j < highDim.size() && highDim[j] == 1) {
            ret.back().push_back(j);
            j++;
        }
        return ret;
    }

    std::optional<struct expandResult>
    getCollapseUnionPoint(llvm::ArrayRef<int64_t> iShape, llvm::ArrayRef<int64_t> jShape) const
    {
        int i = 0, j = 0;
        int64_t iMux = 1, jMux = 1;
        struct expandResult ret;
        while (i < iShape.size() && j < jShape.size()) {
            if (iMux >= jMux) {
                jMux = jMux * jShape[j];
                j++;
            } else { 
                iMux = iMux * iShape[i];
                i++;
            } 
            if (iMux == jMux && iMux != 1) {
                ret.shape.push_back(iMux);
                // comsume the following element 1 greedly
                while (i < iShape.size() && iShape[i] == 1 ) i++;
                while (j < jShape.size() && jShape[j] == 1 ) j++;
                // reset the accumlate multiply result
                iMux = 1;
                jMux = 1;
            }
        }

        // the canditate element is not blanced from now on
        if (j == jShape.size()) {
            while(i < iShape.size()) {
                iMux = iMux * iShape[i++];
            }
        }
        if (i == iShape.size()) {
             while(j < jShape.size()) {
                jMux = jMux * jShape[j++];
            }
        }

        if (iMux != jMux) { // if not match, that means an error
            return {};
        }
        // a leftover balanced 1 is not a real segment.
        // size-1 dims are absorbed into the adjacent reassociation groups instead.
        if (iMux != 1) {
            ret.shape.push_back(iMux);
        }

        ret.srcReassociation = computeReassociation(ret.shape, iShape);
        ret.targetReassociation = computeReassociation(ret.shape, jShape);
        return ret;
    }
};


// linalg named elementwise ops are destination-passing style: the operands
// are split into `inputs` and `outputs` segments (AttrSizedOperandSegments)
// and the result is written into an `outputs` init tensor. They must be built
// with the (resultTypes, inputs, outputs) builder so that both
// `operandSegmentSizes` and the region body are populated; the generic
// (resultTypes, operands) builder does neither and fails verification with
// "operand count does not match the total size in operandSegmentSizes".
template <typename LinalgOpTy>
static LogicalResult lowerElementwiseToLinalg(Operation *op,
                                              ValueRange operands,
                                              PatternRewriter &rewriter) {
    auto resultType = llvm::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType) {
        return rewriter.notifyMatchFailure(op, "expected ranked tensor result");
    }
    auto init = tensor::EmptyOp::create(rewriter, op->getLoc(),
                                        resultType.getShape(),
                                        resultType.getElementType());
    rewriter.replaceOpWithNewOp<LinalgOpTy>(op, resultType, operands,
                                            ValueRange{init.getResult()});
    return success();
}

struct AddOpLowering : public OpConversionPattern<AddOp> {
    using OpConversionPattern<AddOp>::OpConversionPattern;
    LogicalResult
    matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
        return lowerElementwiseToLinalg<linalg::AddOp>(op, adaptor.getOperands(),
                                                       rewriter);
    }
};
struct SubOpLowering : public OpConversionPattern<SubOp> {
    using OpConversionPattern<SubOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(SubOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const final {
        return lowerElementwiseToLinalg<linalg::SubOp>(op, adaptor.getOperands(),
                                                       rewriter);
    }
};
struct MulOpLowering : public OpConversionPattern<MulOp> {
    using OpConversionPattern<MulOp>::OpConversionPattern;

    LogicalResult 
    matchAndRewrite(MulOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const final {
        return lowerElementwiseToLinalg<linalg::MulOp>(op, adaptor.getOperands(),
                                                       rewriter);
    }
};
struct DivOpLowering : public OpConversionPattern<DivOp> {
    using OpConversionPattern<DivOp>::OpConversionPattern;

    LogicalResult 
    matchAndRewrite(DivOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const final {
        return lowerElementwiseToLinalg<linalg::DivOp>(op, adaptor.getOperands(),
                                                       rewriter);
    }
};

}


namespace {
class LinAlgLoweringPass 
    : public mlir::PassWrapper<LinAlgLoweringPass, ::mlir::OperationPass<::mlir::ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinAlgLoweringPass)

    // tell the pass manager, the following dialect will be loaded
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<linalg::LinalgDialect, func::FuncDialect,
        arith::ArithDialect>();
    }

    void runOnOperation() override {
        ConversionTarget target(getContext());
        // tell the convert pass, the following dialect won't be needed to lowering
        target.addLegalDialect<linalg::LinalgDialect, func::FuncDialect,
        arith::ArithDialect, BuiltinDialect, tensor::TensorDialect>();

        target.addIllegalDialect<ToyDialect>();
        auto context = &getContext();
        RewritePatternSet patterns(context);

        patterns.add<ConstantOpLowering, ReturnOpLowering, FuncOpLowering,
                    TransposeOpLowering, ReshapeOpLowering,
                    MatrixMulOpLowering, AddOpLowering, SubOpLowering,
                    MulOpLowering, DivOpLowering, PrintOpLowering>(context);

        if (llvm::failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
}


std::unique_ptr<mlir::Pass> lucid_frontend::toy::createLinalgLoweringPass() {
    return std::make_unique<LinAlgLoweringPass>();
}