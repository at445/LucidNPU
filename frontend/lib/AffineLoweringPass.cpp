
#include "Ops.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
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
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
using namespace ::mlir;
namespace {
static MemRefType ConvertTensor2MemRef(RankedTensorType type) {
    return MemRefType::get(type.getShape(), type.getElementType());
}

static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto &parentBlock = alloc->getParentRegion()->front();
  alloc->moveBefore(&parentBlock.front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock.back());
  return alloc;
}

struct ReturnOpLowering: public OpRewritePattern<lucid_frontend::ReturnOp> {
    using OpRewritePattern<lucid_frontend::ReturnOp>::OpRewritePattern;

    virtual LogicalResult matchAndRewrite(lucid_frontend::ReturnOp op,
                                        PatternRewriter &rewriter) const final {
        if (op->getOperands().size() > 0) {
            return mlir::failure();
        }

        rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
        return mlir::success();
    }
};

struct FuncOpLowering : public OpConversionPattern<lucid_frontend::FuncOp> {
  using OpConversionPattern<lucid_frontend::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(lucid_frontend::FuncOp op, OpAdaptor adaptor,
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
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                    op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConstantOpLowering : public OpRewritePattern<lucid_frontend::ConstantOp> {
    using OpRewritePattern<lucid_frontend::ConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(lucid_frontend::ConstantOp op,
                                    PatternRewriter &rewriter) const final {
        DenseElementsAttr constantValue = op.getValue();
        Location loc = op.getLoc();

        // When lowering the constant operation, we allocate and assign the constant
        // values to a corresponding memref allocation.
        auto tensorType = llvm::cast<RankedTensorType>(op.getType());
        auto memRefType = ConvertTensor2MemRef(tensorType);
        auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

        // We will be generating constant indices up-to the largest dimension.
        // Create these constants up-front to avoid large amounts of redundant
        // operations.
        auto valueShape = memRefType.getShape();
        SmallVector<Value, 8> constantIndices;

        if (!valueShape.empty()) {
        for (auto i : llvm::seq<int64_t>(0, *llvm::max_element(valueShape)))
            constantIndices.push_back(
                rewriter.create<arith::ConstantIndexOp>(loc, i));
        } else {
        // This is the case of a tensor of rank 0.
        constantIndices.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, 0));
        }
        rewriter.replaceOp(op, alloc);
        return success();
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

        target.addIllegalDialect<lucid_frontend::ToyDialect>();
        target.addDynamicallyLegalOp<lucid_frontend::ReturnOp>([](auto op){
            return llvm::none_of(op->getOperandTypes(), [](auto typ){
                return llvm::isa<TensorType>(typ);
            });
        });
        auto context = &getContext();
        RewritePatternSet patterns(context);
        patterns.add<ConstantOpLowering, ReturnOpLowering, FuncOpLowering>(context);

        if (llvm::failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
}


std::unique_ptr<mlir::Pass> lucid_frontend::createAffineLoweringPass() {
    return std::make_unique<AffineLoweringPass>();
}