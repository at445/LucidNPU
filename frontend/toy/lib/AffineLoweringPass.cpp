
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

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#include <deque>
#include <functional>
#include <memory>
#include <numeric>
#include <utility>
using namespace ::mlir;
using namespace lucid_frontend::toy;
namespace {
static MemRefType ConvertTensor2MemRef(RankedTensorType type) {
    return MemRefType::get(type.getShape(), type.getElementType());
}

static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = memref::AllocOp::create(rewriter, loc, type);

  // Make sure to allocate at the beginning of the block.
  auto parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = memref::DeallocOp::create(rewriter, loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

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
template <typename BinaryOp, typename LowerBinaryOp>
class BinaryOpLowering : public ConversionPattern {
public:
    BinaryOpLowering(MLIRContext *ctx)
        : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final 
    {
        auto loc = op->getLoc();
        auto tensorType = llvm::cast<RankedTensorType>(op->getResultTypes().front());

        auto memRefType = ConvertTensor2MemRef(tensorType);
        auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

        // all the low bound of loop in affine loop are 0 
        llvm::SmallVector<int64_t, 4> lowBounds(tensorType.getRank(), 0);
        // all the upper bound of loop in affine loop comes from the tensorType 
        auto shape = tensorType.getShape();
        // all the step of loop in affine loop are 1
        llvm::SmallVector<int64_t, 4> steps(tensorType.getRank(), 1);

        // lowering pattern of calling buildAffineLoopNest
        // affine.for ... {
        //   affine.for %i = 0 to m step 1 {
        //     affine.for %j = 0 to n step 1 {
        //       %op1 = affine.load %input1[..., %i, %j] : memref<...xf32>
        //       %op2 = affine.load %input2[..., %i, %j] : memref<...xf32>
        //       %result = arith.binaryOp %op1, %op2 : f32
        //       affine.store %result, %alloc[..., %i, %j] : memref<...xf32>
        //     }
        //   }
        //} 
        affine::buildAffineLoopNest(rewriter, loc, lowBounds, shape, steps, 
            [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs){
                typename BinaryOp::Adaptor  adaptor(operands);
                auto lhsValue = affine::AffineLoadOp::create(
                    nestedBuilder, loc, adaptor.getLhs(), ivs);
                auto rhsValue = affine::AffineLoadOp::create(
                    nestedBuilder, loc, adaptor.getRhs(), ivs);
                auto binaryOpRsult = LowerBinaryOp::create(nestedBuilder, loc, lhsValue, rhsValue);
                affine::AffineStoreOp::create(nestedBuilder, loc, binaryOpRsult, alloc, ivs);
        });

        rewriter.replaceOp(op, alloc);
        return llvm::success();
    }
};
using AddOpLowering = BinaryOpLowering<AddOp, arith::AddFOp>;
using SubOpLowering = BinaryOpLowering<SubOp, arith::SubFOp>;
using MulOpLowering = BinaryOpLowering<MulOp, arith::MulFOp>;
using DivOpLowering = BinaryOpLowering<DivOp, arith::DivFOp>;

class TransposeOpLowering : public OpConversionPattern<TransposeOp> {
    using OpConversionPattern<TransposeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final 
    {

        auto loc = op->getLoc();
        auto tensorType = llvm::cast<RankedTensorType>(op.getType());
        if (tensorType.getRank() < 2)  return llvm::success();

        auto memRefType = ConvertTensor2MemRef(tensorType);
        auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

        // all the low bound of loop in affine loop are 0 
        llvm::SmallVector<int64_t, 4> lowBounds(tensorType.getRank(), 0);
        // all the upper bound of loop in affine loop comes from the tensorType 
        auto shape = tensorType.getShape();
        // all the step of loop in affine loop are 1
        llvm::SmallVector<int64_t, 4> steps(tensorType.getRank(), 1);
        
        // lowering pattern of calling buildAffineLoopNest
        // affine.for .. {
        //   affine.for %i = 0 to m step 1 {
        //     affine.for %j = 0 to n step 1 {
        //        ....
        //       %loadedVal = affine.load %input[..., %j, %i] : ...
        //        ....  
        //       affine.store %loadedVal, %alloc[..., %i, %j] : ...
        //     }
        //   }
        //}
        affine::buildAffineLoopNest(rewriter, loc, lowBounds, shape, steps, 
            [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs){
                llvm::SmallVector<Value, 4> values(ivs);
                std::swap(values[values.size()-1], values[values.size()-2]);
                auto loadVal = affine::AffineLoadOp::create(
                    nestedBuilder, loc, adaptor.getInput(), values);
                affine::AffineStoreOp::create(nestedBuilder, loc, loadVal, alloc, ivs);
        });

        rewriter.replaceOp(op, alloc);
        return mlir::success();
    }
};


class MatrixMulOpLowering : public OpConversionPattern<MatrixMulOp> {
    using OpConversionPattern<MatrixMulOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(MatrixMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final 
    {
        auto loc = op->getLoc();
        auto tensorType = llvm::cast<RankedTensorType>(op->getResultTypes().front());

        auto memRefType = ConvertTensor2MemRef(tensorType);
        auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

        // all the low bound of loop in affine loop are 0 
        llvm::SmallVector<int64_t, 4> lowBounds(tensorType.getRank(), 0);
        // all the upper bound of loop in affine loop comes from the tensorType 
        auto shape = tensorType.getShape();
        // all the step of loop in affine loop are 1
        llvm::SmallVector<int64_t, 4> steps(tensorType.getRank(), 1);

        // lowering pattern of calling buildAffineLoopNest
        // affine.for .. {
        //   affine.for %i = 0 to m step 1 {
        //     affine.for %j = 0 to n step 1 {
        //       %zero = arith.constant 0.000000e+00 : f64
        //       affine.store %zero, %acc[..., %i, %j] : memref<...xf32>
        //       affine.for %k = 0 to K step 1 
        //       {
        //         %op1 = affine.load %input1[..., %i, %k] : memref<...xf32>
        //         %op2 = affine.load %input2[..., %k, %j] : memref<...xf32>
        //         %mul = arith.mulf %op1, %op2 : f32
        //         %partial = affine.load %result, %acc[..., %i, %j] : memref<1xf32>
        //         %sum = arith.muladd %mul, %partial : f64
        //         affine.store %sum, %acc[..., %i, %j] : memref<1xf32>
        //       }
        //     }
        //   }
        //}

        affine::buildAffineLoopNest(rewriter, loc, lowBounds, shape, steps, 
            [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs_outer)
        {
            auto zero = arith::ConstantOp::create(nestedBuilder, loc, 
                nestedBuilder.getF64Type(), nestedBuilder.getF64FloatAttr(0.0));
            affine::AffineStoreOp::create(nestedBuilder, loc, zero, alloc, ivs_outer);

            auto tensorType = llvm::cast<RankedTensorType>(*op.operand_type_begin());
            auto K = tensorType.getShape().back();

            affine::buildAffineLoopNest(rewriter, loc, {0}, {K}, {1}, 
                [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs_inner)
            {
                llvm::SmallVector<Value, 4> values{ivs_outer};
                auto cache = values.back(); //%j
                values.pop_back(); //[..., %i]
                values.push_back(ivs_inner.front());//[..., %i, %k]
                auto loadedLhs = affine::AffineLoadOp::create(
                    nestedBuilder, loc, adaptor.getLhs(), values);

                values.pop_back(); //[..., %i]
                values.pop_back(); //[...]
                values.push_back(ivs_inner.front()); //[..., %k]
                values.push_back(cache); //[..., %k, %j]
                auto loadedRhs = affine::AffineLoadOp::create(
                    nestedBuilder, loc, adaptor.getRhs(), values);

                auto mulVal =  arith::MulFOp::create(
                    nestedBuilder, loc, loadedLhs, loadedRhs);

                auto partial = affine::AffineLoadOp::create(
                    nestedBuilder, loc, alloc, ivs_outer);
                
                auto sumVal =  arith::AddFOp::create(
                    nestedBuilder, loc, partial, mulVal);
                
                affine::AffineStoreOp::create(nestedBuilder, loc, sumVal, alloc, ivs_outer);
            });
        });

        rewriter.replaceOp(op, alloc);
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
        DenseElementsAttr constantValue = op.getValue();
        Location loc = op.getLoc();

        // When lowering the constant operation, we allocate and assign the constant
        // values to a corresponding memref allocation.
        auto tensorType = llvm::cast<RankedTensorType>(op.getType());
        auto memRefType = ConvertTensor2MemRef(tensorType);
        auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

        int64_t maxIdx = -1;
        llvm::for_each(memRefType.getShape(), [&](int64_t dimNum){
            if (maxIdx < dimNum) maxIdx = dimNum;
        });

        llvm::SmallVector<mlir::Value> indexList;
        for (int64_t i = 0; i < maxIdx; ++i) {
            indexList.push_back(arith::ConstantIndexOp::create(rewriter, loc, i));
        }

        auto valueIt = constantValue.value_begin<FloatAttr>();

        llvm::SmallVector<mlir::Value> values;
        int attrIdx = 0;
        recuriveCreatScala(memRefType.getShape(), indexList, values, [&](ValueRange vals) {
            Value val = arith::ConstantOp::create(rewriter, loc, memRefType.getElementType(), *valueIt++);
            attrIdx++;
            affine::AffineStoreOp::create(rewriter, loc, val, alloc, vals);
        });

        rewriter.replaceOp(op, alloc);
        return success();
    }
private:
    void recuriveCreatScala(llvm::ArrayRef<int64_t> shape,
                            llvm::ArrayRef<mlir::Value> referValues,
                            llvm::SmallVector<mlir::Value>& values,
                            const std::function<void(mlir::ValueRange)>& func,
                            int layer = 0) const {
        if (layer == shape.size()) {
            //generate IR here
            func(ValueRange(values));
            return;
        }

        for (int64_t i = 0; i < shape[layer]; ++i) {
            values.push_back(referValues[i]);
            recuriveCreatScala(shape, referValues, values,func, layer + 1);
            values.pop_back();
        }

    }
};

struct PrintOpLowering : public OpConversionPattern<PrintOp> {
    using OpConversionPattern<PrintOp>::OpConversionPattern;
    LogicalResult
    matchAndRewrite(PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
        rewriter.replaceOpWithNewOp<PrintOp>(op, adaptor.getInput());
        return mlir::success();
    }
};

struct ReshapeOpLowering : public OpConversionPattern<ReshapeOp> {
    using OpConversionPattern<ReshapeOp>::OpConversionPattern;
    LogicalResult
    matchAndRewrite(ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
        auto tensorType = llvm::cast<RankedTensorType>(op.getType());
        auto memRefType = ConvertTensor2MemRef(tensorType);
        auto castOp = memref::CastOp::create(
            rewriter, op.getLoc(), memRefType, adaptor.getInput());
        rewriter.replaceOp(op, castOp);
        return mlir::success();
    }
};


}


namespace {
class AffineLoweringPass 
    : public mlir::PassWrapper<AffineLoweringPass, ::mlir::OperationPass<::mlir::ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AffineLoweringPass)

    // tell the pass manager, the following dialect will be loaded
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<affine::AffineDialect, func::FuncDialect,
        arith::ArithDialect, memref::MemRefDialect>();
    }

    void runOnOperation() override {
        ConversionTarget target(getContext());
        // tell the convert pass, the following dialect won't be needed to lowering
        target.addLegalDialect<affine::AffineDialect, func::FuncDialect,
        arith::ArithDialect, memref::MemRefDialect, BuiltinDialect>();

        target.addIllegalDialect<ToyDialect>();
        target.addDynamicallyLegalOp<PrintOp>([](auto op){
            return llvm::none_of(op->getOperandTypes(), [](auto typ){
                return llvm::isa<TensorType>(typ);
            });
        });
        auto context = &getContext();
        RewritePatternSet patterns(context);

        patterns.add<ConstantOpLowering, ReturnOpLowering, FuncOpLowering,
                    PrintOpLowering, TransposeOpLowering, ReshapeOpLowering,
                    AddOpLowering, SubOpLowering, MulOpLowering, DivOpLowering,
                    MatrixMulOpLowering>(context);

        if (llvm::failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
}


std::unique_ptr<mlir::Pass> lucid_frontend::toy::createAffineLoweringPass() {
    return std::make_unique<AffineLoweringPass>();
}