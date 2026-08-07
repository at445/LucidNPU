
#include "Ops.h"
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
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <memory>
#include <numeric>
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

class TransposeOpLowering : public OpConversionPattern<TransposeOp> {
    using OpConversionPattern<TransposeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final 
    {
        return mlir::success();
    }
};


class MatrixMulOpLowering : public OpConversionPattern<MatrixMulOp> {
    using OpConversionPattern<MatrixMulOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(MatrixMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final 
    {
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
        auto inputShape = llvm::cast<TensorType>(op->getOperand(0).getType());
        auto outputShape = op.getResult().getType();
        if(!inputShape) {
            return rewriter.notifyMatchFailure(op, [op](Diagnostic &diag) {
                diag << "type inllegl" << op->getOperand(0).getType();
            });
        }
        if (inputShape == outputShape) {
            rewriter.replaceOp(op, adaptor.getInput());
            return mlir::success();
        }

        bool aligment = false;
        int i = 0, j = 0;
        int64_t iMux = 1, jMux = 1;
        // 2 x 3 x 3 x 3
        auto iShape = inputShape.getShape();
        // 6 x 9
        auto jShape = outputShape.getShape();
        std::deque<std::pair<int, int>> ret;
        // iMux = 6 jMux = 54 i = 2 j = 2 iShape.size() = 4 jShape.size() = 2
        // ret = {2,1} 
        while (i < iShape.size() && j < jShape.size()) {
            if (iMux >= jMux) {
                jMux = jMux * jShape[j];
                j++; 
            } else { 
                iMux = iMux * iShape[i];
                i++; 
            } 
            if (iMux == jMux) {
                ret.push_back({i, j}); 
            }
        }
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
        if (iMux == jMux) {
            ret.push_back({i, j}); 
        } else {
            return rewriter.notifyMatchFailure(op, [op](Diagnostic &diag) {
                diag << "type inlegel, rhs: " << op->getOperand(0).getType() << "   result: " <<  op->getResult().getType();
            });
        }


        return mlir::success();
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
        arith::ArithDialect, BuiltinDialect>();

        target.addIllegalDialect<ToyDialect>();
        target.addLegalOp<PrintOp>();
        auto context = &getContext();
        RewritePatternSet patterns(context);

        patterns.add<ConstantOpLowering, ReturnOpLowering, FuncOpLowering,
                    TransposeOpLowering, ReshapeOpLowering,
                    MatrixMulOpLowering>(context);

        if (llvm::failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
}


std::unique_ptr<mlir::Pass> lucid_frontend::toy::createLinalgLoweringPass() {
    return std::make_unique<LinAlgLoweringPass>();
}