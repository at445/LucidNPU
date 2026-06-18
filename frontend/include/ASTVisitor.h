//===- ASTVisitor.h - CRTP Visitor for the Toy AST ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a base visitor for the Toy AST using the Curiously 
// Recurring Template Pattern (CRTP). This allows for efficient static 
// dispatch without virtual function overhead, providing a clean separation 
// between data nodes and their behaviors.
//
//===----------------------------------------------------------------------===//
#ifndef LUCID_FRONTEND_AST_VISITOR_H
#define LUCID_FRONTEND_AST_VISITOR_H
#include "AST.h"
#include "llvm/Support/Casting.h"

namespace lucid_frontend {
template <typename Derived, typename RetTy = void>
class ASTVisitor {
protected:
    RetTy visitor(ExprAST * ptr) {
        if (!ptr) return RetTy();
        switch (ptr->getKind())
        {
        case ExprAST::ExprASTKind::Expr_VarDecl:
            return static_cast<Derived *>(this)->visitVarDeclare(llvm::cast<VarDeclExprAST>(ptr));
        case ExprAST::ExprASTKind::Expr_Return:
            return static_cast<Derived *>(this)->visitReturn(llvm::cast<ReturnExprAST>(ptr));
        case ExprAST::ExprASTKind::Expr_Num:
            return static_cast<Derived *>(this)->visitNumber(llvm::cast<NumberExprAST>(ptr));
        case ExprAST::ExprASTKind::Expr_Literal:
            return static_cast<Derived *>(this)->visitLiteral(llvm::cast<LiteralExprAST>(ptr));
        case ExprAST::ExprASTKind::Expr_Var:
            return static_cast<Derived *>(this)->visitVariable(llvm::cast<VariableExprAST>(ptr));
        case ExprAST::ExprASTKind::Expr_BinOp:
            return static_cast<Derived *>(this)->visitBinary(llvm::cast<BinaryExprAST>(ptr));
        case ExprAST::ExprASTKind::Expr_Call:
            return static_cast<Derived *>(this)->visitFunctionCall(llvm::cast<CallExprAST>(ptr));
        case ExprAST::ExprASTKind::Expr_Print:
            return static_cast<Derived *>(this)->visitPrint(llvm::cast<PrintExprAST>(ptr));
        case ExprAST::ExprASTKind::Expr_Transpose:
            return static_cast<Derived *>(this)->visitTranspose(llvm::cast<TransposeExprAST>(ptr));
        default:
            llvm_unreachable("Unknown ExprASTKind");
        }
    } 
    RetTy visitor(PrototypeAST * ptr) {
        return static_cast<Derived *>(this)->visitPrototype(ptr);
    }
    RetTy visitor(FunctionAST * ptr) {
        return static_cast<Derived *>(this)->visitFunction(ptr);
    }
    RetTy visitor(ModuleAST * ptr) {
        return static_cast<Derived *>(this)->visitModule(ptr);
    }
};
}
#endif
