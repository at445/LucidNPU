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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include <functional>
#include <numeric>
#include <optional>
#include <vector>
using namespace lucid_frontend;
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
    void mlirGen(const FunctionAST& functionAST) {
        // create a symbolTable for function scope
        llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> funScope(m_symbolTable);

        // Create an MLIR function for the given prototype.
        mlir::toy::FuncOp function = mlirGen(*functionAST.getProto());
        if (!function)
            return;

        // Insert the formal parameters of the current function into the symbol table.
        mlir::Block &entryBlock = function.front();
        auto protoArgs = functionAST.getProto()->getArgs();
        for (const auto nameAndValue: llvm::zip(protoArgs, entryBlock.getArguments())) {
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

        // Implicit void return if the function body didn't end with return.
        mlir::toy::ReturnOp returnOp;
        if (!entryBlock.empty())
            returnOp = llvm::dyn_cast<mlir::toy::ReturnOp>(entryBlock.back());
        if (!returnOp)
            m_builder.create<mlir::toy::ReturnOp>(
                locConvert(functionAST.getProto()->loc()), mlir::ValueRange{});
    }

    mlir::LogicalResult mlirGen(const ReturnStmtAST &returnAST) {
        auto loc = locConvert(returnAST.loc());
        if (returnAST.getExpr().has_value()) {
            mlir::Value expr = mlirGen(**returnAST.getExpr());
            if (!expr)
                return mlir::failure();
            m_builder.create<mlir::toy::ReturnOp>(loc, expr);
        } else {
            m_builder.create<mlir::toy::ReturnOp>(loc, mlir::ValueRange{});
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
        default:
            mlir::emitError(locConvert(stmt.loc()))
                << "MLIR codegen encountered an unhandled statement kind '"
                << llvm::Twine(stmt.getKind()) << "'";
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
        default:
            mlir::emitError(locConvert(expr.loc()))
                << "MLIR codegen encountered an unhandled expr kind '"
                << llvm::Twine(expr.getKind()) << "'";
            return nullptr;
        }
    }

    mlir::Value mlirGen(const BinaryExprAST &binaryExpr) {
        auto lhs = mlirGen(*binaryExpr.getLHS());
        auto rhs = mlirGen(*binaryExpr.getRHS());
        auto type = lhs.getType();
        auto op = binaryExpr.getOp();
        auto loc = locConvert(binaryExpr.loc());
        switch (op) {
            case '+':
                return m_builder.create<mlir::toy::AddOp>(loc, lhs, rhs);
            case '-':
                return m_builder.create<mlir::toy::SubOp>(loc, lhs, rhs);
            case '*':
                return m_builder.create<mlir::toy::MulOp>(loc, lhs, rhs);
            case '/':
                return m_builder.create<mlir::toy::DivOp>(loc, lhs, rhs);
            case '@':
                return m_builder.create<mlir::toy::MatrixMulOp>(loc, lhs, rhs);
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
        auto loc = number.loc();
        return m_builder.create<mlir::toy::ConstantOp>(locConvert(number.loc()),  number.getValue());
    }

    mlir::Value mlirGen(const LiteralExprAST &literal) {
        auto type = getType(literal.getDims());

        std::vector<double> data;
        data.reserve(std::accumulate(literal.getDims().begin(), literal.getDims().end(),
                                         1, std::multiplies<int>()));

        collectData(literal, data);

        auto mlirType = mlir::RankedTensorType::get(literal.getDims(), m_builder.getF64Type());

        auto dataAttribute = mlir::DenseElementsAttr::get(mlirType, llvm::ArrayRef(data));

        return m_builder.create<mlir::toy::ConstantOp>(locConvert(literal.loc()), type, dataAttribute);
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
    
    mlir::toy::FuncOp mlirGen(const PrototypeAST & prototypeAST) {
        auto loc = locConvert(prototypeAST.loc());
        // This is a generic function, the return type will be inferred later.
        // Arguments type are uniformly unranked tensors.
        llvm::SmallVector<mlir::Type, 4> argTypes(prototypeAST.getArgs().size(),
                                                  getType({}));
        auto funcType = m_builder.getFunctionType(argTypes, std::nullopt);

        return m_builder.create<mlir::toy::FuncOp>(loc, prototypeAST.getName(), funcType);
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

namespace lucid_frontend {
    // The public API for codegen.
    mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, ModuleAST &moduleAST) {
        return MLIRGenImpl(context).mlirGen(moduleAST);
    }
} // namespace lucid_frontend