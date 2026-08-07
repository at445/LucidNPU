#include "MLIRGen.h"
#include "AST.h"
#include "Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include <cstddef>
#include <functional>
#include <numeric>
#include <optional>
#include <vector>
#include <iostream>
using namespace lucid_frontend;
using namespace lucid_frontend::toy;
namespace {
class MLIRGenImpl {
public:
    MLIRGenImpl(mlir::MLIRContext &context) : m_builder(&context){ }
    mlir::ModuleOp mlirGen(const ModuleAST &moduleAST) {
        m_module = mlir::ModuleOp::create(m_builder.getUnknownLoc());
        m_builder.setInsertionPointToEnd(m_module.getBody());
        for (const auto &func : moduleAST) {
            mlirGen(*func);
        }
        if (llvm::failed(mlir::verify(m_module))) {
            m_module->emitError("module verification error");
            return nullptr;
        }
        return m_module;
    }

private:
    llvm::StringMap<FuncOp> FunctionDecl;

    void mlirGen(const FunctionAST& functionAST) {
        // create a symbolTable for function scope
        llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> funScope(m_symbolTable);
        auto proto = functionAST.getProto();

        m_builder.setInsertionPointToEnd(m_module.getBody());
        // Create an MLIR function for the given prototype.
        FuncOp function = mlirGen(*proto);
        if (!function) 
            return;
        
        // Set all non-main functions to private so that the subsequent inline operation can be performed.
        if (proto->getName() != "main") 
            function.setPrivate();

        FunctionDecl.insert({function.getSymName(), function});

        // Insert the formal parameters of the current function into the symbol table.
        mlir::Block &entryBlock = function.front();
        for (const auto nameAndValue: llvm::zip(proto->getArgs(), entryBlock.getArguments())) {
            if (failed(declare(std::get<0>(nameAndValue)->getName(),
                         std::get<1>(nameAndValue))))
            return;
        }

        // Emit the body of the function.
        m_builder.setInsertionPointToStart(&entryBlock);
        for (auto *stmt : functionAST.getBody()) {
            if (mlir::failed(mlirGen(*stmt)))
                return;
        }
        
        ReturnOp returnOp = nullptr;
        if (!entryBlock.empty())
            returnOp = llvm::dyn_cast<ReturnOp>(entryBlock.back());
        
        // Implicit void return if the function body didn't end with return.
        if (!returnOp) {
            returnOp = ReturnOp::create(m_builder, locConvert(proto->loc()));
            return;
        }

        // change the function result type base on return op type
        auto oldFuncType = function.getFunctionType();
        function.setFunctionType(oldFuncType.clone(oldFuncType.getInputs(), returnOp.getOperandTypes()));
    }

    mlir::LogicalResult mlirGen(const ReturnStmtAST &returnAST) {
        auto loc = locConvert(returnAST.loc());
        if (returnAST.getExpr().has_value()) {
            mlir::Value expr = mlirGen(**returnAST.getExpr());
            if (!expr)
                return mlir::failure();
            ReturnOp::create(m_builder, loc, expr);
        } else {
            ReturnOp::create(m_builder, loc, mlir::ValueRange{});
        }
        return mlir::success();
    }

    /// Dispatch codegen for statement subclass using RTTI.
    mlir::LogicalResult mlirGen(const StmtAST &stmt) {
        switch (stmt.getKind()) {
        case StmtAST::Stmt_VarDecl:
            return mlirGen(llvm::cast<VarDeclStmtAST>(stmt));
        case StmtAST::Stmt_Return:
            return mlirGen(llvm::cast<ReturnStmtAST>(stmt));
        case StmtAST::Stmt_Expr:
            return mlirGen(llvm::cast<ExprStmtAST>(stmt));
        default:
            mlir::emitError(locConvert(stmt.loc()))
                << "MLIR codegen encountered an unhandled statement kind '"
                << llvm::Twine(stmt.getKind()) << "'";
            return mlir::failure();
        }
    }

    mlir::LogicalResult mlirGen(const ExprStmtAST &exprStmtAst) {
        switch (exprStmtAst.getExpr()->getKind()) {
            case ExprAST::Expr_Call:
            case ExprAST::Expr_Print:
                mlirGen(*exprStmtAst.getExpr());
                return mlir::success();
            default:
                mlir::emitError(locConvert(exprStmtAst.loc())) <<
                    "Function calls without side effects are not permitted.";
                return mlir::failure();
        }
    }

    mlir::LogicalResult mlirGen(const VarDeclStmtAST &varDeclAst) {
        auto *init = varDeclAst.getInitVal();
        if (!init) {
            mlir::emitError(locConvert(varDeclAst.loc()),
                            "missing initializer in variable declaration");
            return mlir::failure();
        }
        auto value = mlirGen(*init);
        if (!value) {
            mlir::emitError(locConvert(varDeclAst.loc()),
                            "convert MLIR value failed");
            return mlir::failure();
        }

        auto typ = varDeclAst.getType();
        if(typ && *typ) {
            value = ReshapeOp::create(
                m_builder, locConvert(varDeclAst.loc()),  typeConvert(**typ), value);
        }

        if (failed(declare(varDeclAst.getName(), value))) {
            mlir::emitError(locConvert(varDeclAst.loc()),
                            "duplicated name used");
            return mlir::failure();
        }

        return mlir::success();
    }

    /// Dispatch codegen for the right expression subclass using RTTI.
    mlir::Value mlirGen(ExprAST &expr) {
        switch (expr.getKind()) {
        case ExprAST::Expr_Num: 
            return mlirGen(llvm::cast<NumberExprAST>(expr));
        case ExprAST::Expr_Literal:
            return mlirGen(llvm::cast<LiteralExprAST>(expr));
        case ExprAST::Expr_Var:
            return mlirGen(llvm::cast<VariableExprAST>(expr));
        case ExprAST::Expr_BinOp:
            return mlirGen(llvm::cast<BinaryExprAST>(expr));
        case ExprAST::Expr_Call:
            return mlirGen(llvm::cast<CallExprAST>(expr));
        case ExprAST::Expr_Print:
            return mlirGen(llvm::cast<PrintExprAST>(expr));
        case ExprAST::Expr_Transpose:
            return mlirGen(llvm::cast<TransposeExprAST>(expr));
        default:
            mlir::emitError(locConvert(expr.loc()))
                << "MLIR codegen encountered an unhandled expr kind '"
                << llvm::Twine(expr.getKind()) << "'";
            return nullptr;
        }
    }

    mlir::Value mlirGen(const PrintExprAST &printExpr) {
        auto value = mlirGen(*printExpr.getArg());
        if (!value) return nullptr;
        PrintOp::create(m_builder, locConvert(printExpr.loc()), value);
        return nullptr;
    } 

    mlir::Value mlirGen(const TransposeExprAST &transExpr) {
        auto value = mlirGen(*transExpr.getArg());
        if (!value) return nullptr;
        
        return TransposeOp::create(
            m_builder,
            locConvert(transExpr.loc()), 
            getType({}),
            value);
    } 

    mlir::Value mlirGen(const CallExprAST &callExpr) {
        auto it = FunctionDecl.find(callExpr.getCallee());
        if (it == FunctionDecl.end()) {
            mlir::emitError(locConvert(callExpr.loc()))
                << "no defined function found for '" << callExpr.getCallee() << "'";
        }
        llvm::SmallVector<mlir::Value, 4> values;
        for (auto & arg: callExpr.getArgs()) {
            auto val = mlirGen(*arg);
            if (!val) return nullptr;
            values.push_back(val);
        }
        return GenericCallOp::create(
            m_builder,
            locConvert(callExpr.loc()), 
            it->second.getFunctionType().getResult(0),
            callExpr.getCallee(), 
            values,
            /*arg_attrs=*/nullptr,
            /*res_attrs=*/nullptr);
    } 


    mlir::Value mlirGen(const BinaryExprAST &binaryExpr) {
        auto lhs = mlirGen(*binaryExpr.getLHS());
        auto rhs = mlirGen(*binaryExpr.getRHS());
        auto op = binaryExpr.getOp();
        auto loc = locConvert(binaryExpr.loc());
        switch (op) {
            case '+':
                return AddOp::create(m_builder, loc, lhs, rhs);
            case '-':
                return SubOp::create(m_builder, loc, lhs, rhs);
            case '*':
                return MulOp::create(m_builder, loc, lhs, rhs);
            case '/':
                return DivOp::create(m_builder, loc, lhs, rhs);
            case '@':
                return MatrixMulOp::create(m_builder, loc, lhs, rhs);
            default:
                llvm_unreachable("unknown binary op");
        }
        return nullptr;
    }

    mlir::Value mlirGen(const VariableExprAST &var) {
        if (auto value = m_symbolTable.lookup(var.getName()))
            return value;
        mlir::emitError(locConvert(var.loc()), "unknown variable '")
            << var.getName() << "'";
        return nullptr;
    }

    mlir::Value mlirGen(const NumberExprAST &number) {
        auto loc = locConvert(number.loc());
        return ConstantOp::create(m_builder, loc,  number.getValue());
    }

    mlir::Value mlirGen(const LiteralExprAST &literal) {
        auto type = getType(literal.getDims());

        std::vector<double> data;
        data.reserve(std::accumulate(literal.getDims().begin(), literal.getDims().end(),
                                         1, std::multiplies<int>()));

        collectData(literal, data);

        auto mlirType = mlir::RankedTensorType::get(literal.getDims(), m_builder.getF64Type());

        auto dataAttribute = mlir::DenseElementsAttr::get(mlirType, llvm::ArrayRef(data));

        return ConstantOp::create(m_builder, locConvert(literal.loc()), type, dataAttribute);
    }

    void collectData(const ExprAST &expr, std::vector<double> &data) {
        if (const auto lit = llvm::dyn_cast<const LiteralExprAST>(&expr)) {
            for (auto &value : lit->getValues()) {
                collectData(*value, data);
            }
        } else {
            assert(llvm::isa<NumberExprAST>(expr) && "expected literal or number expr");
            data.push_back(llvm::cast<NumberExprAST>(expr).getValue());
        }
    }

    llvm::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
        if (m_symbolTable.count(var))
          return mlir::failure();
        m_symbolTable.insert(var, value);
        return mlir::success();
    }
    
    FuncOp mlirGen(const PrototypeAST & prototypeAST) {
        auto loc = locConvert(prototypeAST.loc());
        // This is a generic function, the return type will be inferred later.
        // Arguments type are uniformly unranked tensors.
        llvm::SmallVector<mlir::Type, 4> argTypes(prototypeAST.getArgs().size(),
                                                  getType({}));
        auto funcType = m_builder.getFunctionType(argTypes, mlir::TypeRange());

        return FuncOp::create(m_builder, loc, prototypeAST.getName(), funcType);
    }

    mlir::Type getType(llvm::ArrayRef<int64_t> shape) {
        if (shape.empty())
            return mlir::UnrankedTensorType::get(m_builder.getF64Type());
        return mlir::RankedTensorType::get(shape, m_builder.getF64Type());
    }

    /// Helper conversion for a Toy VarType to an MLIR Type.
    mlir::Type typeConvert(const VarType &type) { return getType(type.shape); }


    /// Helper conversion for a Toy AST location to an MLIR location.
    mlir::Location locConvert(const Location &loc) {
        return mlir::FileLineColLoc::get(m_builder.getStringAttr(loc.file), loc.line, loc.col);
    }

private:
    mlir::ModuleOp m_module;
    mlir::OpBuilder m_builder;
    llvm::ScopedHashTable<llvm::StringRef, mlir::Value> m_symbolTable;
};
}

namespace lucid_frontend { namespace toy {
    // The public API for codegen.
    mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, ModuleAST &moduleAST) {
        return MLIRGenImpl(context).mlirGen(moduleAST);
    }
}} // namespace lucid_frontend::toy