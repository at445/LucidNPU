//===----------------------------------------------------------------------===//
//
// This file implements full lowering of Toy operations to LLVM MLIR dialect.
// 'toy.print' is lowered to a loop nest that calls `printf` on each element of
// the input array. The file also sets up the ToyToLLVMLoweringPass. This pass
// lowers the combination of Arithmetic + Affine + SCF + Func dialects to the
// LLVM one:
//
//                         Affine --
//                                  |
//                                  v
//                       Arithmetic + Func --> LLVM (Dialect)
//                                  ^
//                                  |
//     'toy.print' --> Loop (SCF) --
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

#include "Ops.h"
#include "Dialect.h"
#include "Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <memory>
#include <utility>
using namespace mlir;
namespace  {
/// Lowers `toy.print` to a loop nest calling `printf` on each of the individual
/// elements of the array.

class PrintOpLowering: public mlir::OpRewritePattern<lucid_frontend::PrintOp> {
    using OpRewritePattern<lucid_frontend::PrintOp>::OpRewritePattern;
protected:
    mlir::LogicalResult matchAndRewrite(lucid_frontend::PrintOp op,
                                        mlir::PatternRewriter &rewriter) const final {
        auto *context = rewriter.getContext();
        auto memRefType = llvm::cast<MemRefType>((*op->operand_type_begin()));
        auto memRefShape = memRefType.getShape();
        auto loc = op->getLoc();

        ModuleOp parentModule = op->getParentOfType<ModuleOp>();

        // 1. Create the global declare of printf function(LLVM::LLVMFuncOp) on ModuleOp.
        auto printfRef = getOrInsertPrintfOp(rewriter, parentModule);

        // 2. Generate a global string related to format limitations in the printf function.
        Value formatCst = getOrCreateStringGlobalOp(parentModule, rewriter, 
            "frmt_spec",StringRef("%f \0", 4), loc);
        
        // 3. Generate a global string of new line for printf usage
        Value newLineCst = getOrCreateStringGlobalOp(parentModule, rewriter,
            "nl", StringRef("\n\0", 2), loc);
        
        // 4. Create SCF for Op, the content like:
        // %lower0 = arith.constant 0 : index
        // %upper0 = arith.constant 2 : index
        // %step0  = arith.constant 1 : index
        // scf.for %iv0 = %lower0 to %upper0 step %step0 {
        //      ...
        //   llvm.call @printf(%newLineCst) : (!llvm.ptr<i8>) -> i32
        //   scf.yield
        // }
        llvm::SmallVector<Value, 4> loopIvs;
        for (unsigned i = 0, sum = memRefShape.size(); i < sum; ++i) {
            auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
            auto upperBound = rewriter.create<arith::ConstantIndexOp>(loc, memRefShape[i]);
            auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

            auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
            // scf::ForOp contains an additional YieldOp. manually clear it first.
            for (Operation &op : *loop.getBody())
                rewriter.eraseOp(&op);
            loopIvs.push_back(loop.getInductionVar());

            rewriter.setInsertionPointToEnd(loop.getBody());
            if (i != sum - 1) {
                rewriter.create<LLVM::CallOp>(loc, getPrintfType(context), printfRef, newLineCst);
            }
            rewriter.create<scf::YieldOp>(loc);
            rewriter.setInsertionPointToStart(loop.getBody());
        }

        // Generate a call to printf for the current element of the loop.
        //  %val = memref.load %a[%iv0, %iv1] : memref<2x2xf64>
        //  llvm.call @printf(%frmt_spec, %val) : (!llvm.ptr<i8>, f64) -> i32
        auto loadOp = rewriter.create<memref::LoadOp>(loc, op.getInput(), loopIvs);
        rewriter.create<LLVM::CallOp>(loc, getPrintfType(context),
         printfRef, ArrayRef<Value>({formatCst, loadOp}));

        rewriter.eraseOp(op);
        return success();
    }
private:
    /// Create a function declaration for printf, the signature is:
    ///   * `i32 (i8*, ...)`
    LLVM::LLVMFunctionType getPrintfType(MLIRContext *context) const {
        auto llvmFnTyp = LLVM::LLVMFunctionType::get(
            IntegerType::get(context, 32), 
            LLVM::LLVMPointerType::get(context), /*isVarArg=*/true);
        return llvmFnTyp;
    } 
    /// This function has two functions:
    /// 1. insert a new LLVMFuncOp into moduleOp if not exist
    /// 2. get the symbol of printf on moduleOp
    FlatSymbolRefAttr getOrInsertPrintfOp(PatternRewriter &rewriter, ModuleOp module) const {
        const auto printfName = "printf";
        auto context = module->getContext();
        if (module.lookupSymbol<LLVM::LLVMFuncOp>(printfName)) {
            return SymbolRefAttr::get(context, printfName);
        }

        // Insert the printf function into the body of the parent module.
        PatternRewriter::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        rewriter.create<LLVM::LLVMFuncOp>(module->getLoc(), printfName, getPrintfType(context));
        // because of LLVMFuncOp has Symbol trait, after created the symbolTable on moduleOp 
        // will be changed atuomatically.
        return SymbolRefAttr::get(context, printfName);
    }

    Value getOrCreateStringGlobalOp(ModuleOp module, OpBuilder &builder,
                                    StringRef strName, StringRef strValue, Location loc) const {
        LLVM::GlobalOp globalOp;
        if (!(globalOp = module.lookupSymbol<LLVM::GlobalOp>(strName))) {
            // Insert the printf function into the body of the parent module.
            PatternRewriter::InsertionGuard guard(builder);
            builder.setInsertionPointToStart(module.getBody());

            auto int8Typ = IntegerType::get(builder.getContext(), 8);
            auto arrayTyp = LLVM::LLVMArrayType::get(int8Typ, strValue.size());
            globalOp = builder.create<LLVM::GlobalOp>(loc, arrayTyp, true, 
                LLVM::Linkage::Internal, strName, 
                // notes: The builder.getxxxAttr function is used to 
                //        convert the object of the view class into an object on the IR tree.
                builder.getStrArrayAttr(strValue), 
                0);
        }

        // create on GPEOp pointer to the first character of global string.
        Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, globalOp);

        auto int64Typ = IntegerType::get(builder.getContext(), 64);
        auto idxAttr = builder.getIndexAttr(0);
        Value cst0 = builder.create<LLVM::ConstantOp>(loc, int64Typ, idxAttr);

        auto llvmPtrTyp = LLVM::LLVMPointerType::get(builder.getContext());
        auto GEPOp = builder.create<LLVM::GEPOp>(loc,
            llvmPtrTyp, // the result type of GEPOp is an LLVMPointerType
            globalOp.getType(), //the element type of GEPOp is LLVMArrayType(int8)
            globalPtr,  // point to the global string pointer 
            ArrayRef<Value>{cst0, cst0});
        return GEPOp;
    }
};
}

namespace {
class LLVMLoweringPass
    : public mlir::PassWrapper<LLVMLoweringPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMLoweringPass);
    // tell the pass manager, the following dialect will be loaded
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<mlir::LLVM::LLVMDialect, mlir::scf::SCFDialect>();
    }

    void runOnOperation() final {
        mlir::LLVMConversionTarget target(getContext());
        target.addLegalOp<mlir::ModuleOp>();

        mlir::LLVMTypeConverter typeConverter(&getContext());

        RewritePatternSet patterns(&getContext());
        populateAffineToStdConversionPatterns(patterns);
        populateSCFToControlFlowConversionPatterns(patterns);
        mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
        populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
        cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
        populateFuncToLLVMConversionPatterns(typeConverter, patterns);

        // The only remaining operation to lower from the `toy` dialect, is the
        // PrintOp.
        patterns.add<PrintOpLowering>(&getContext());

        if (llvm::failed(applyFullConversion(getOperation(), target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
}

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> lucid_frontend::createLowerToLLVMPass() {
  return std::make_unique<LLVMLoweringPass>();
}
