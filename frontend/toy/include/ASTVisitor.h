//===- ASTVisitor.h - CRTP Visitor for the Toy AST ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a base visitor for the Toy AST using the Curiously
// Recurring Template Pattern (CRTP). Expression and statement nodes are
// dispatched through separate visitor overloads.
//
//===----------------------------------------------------------------------===//
#ifndef LUCID_FRONTEND_AST_VISITOR_H
#define LUCID_FRONTEND_AST_VISITOR_H
#include "AST.h"
#include "llvm/Support/Casting.h"

namespace lucid_frontend { namespace toy {
template <typename Derived, typename RetTy = void>
class ASTVisitor {
protected:
  RetTy visitor(ExprAST *ptr) {
    if (!ptr)
      return RetTy();
    switch (ptr->getKind()) {
    case ExprAST::ExprASTKind::Expr_Num:
      return static_cast<Derived *>(this)->visitNumber(
          llvm::cast<NumberExprAST>(ptr));
    case ExprAST::ExprASTKind::Expr_Literal:
      return static_cast<Derived *>(this)->visitLiteral(
          llvm::cast<LiteralExprAST>(ptr));
    case ExprAST::ExprASTKind::Expr_Var:
      return static_cast<Derived *>(this)->visitVariable(
          llvm::cast<VariableExprAST>(ptr));
    case ExprAST::ExprASTKind::Expr_BinOp:
      return static_cast<Derived *>(this)->visitBinary(
          llvm::cast<BinaryExprAST>(ptr));
    case ExprAST::ExprASTKind::Expr_Call:
      return static_cast<Derived *>(this)->visitFunctionCall(
          llvm::cast<CallExprAST>(ptr));
    case ExprAST::ExprASTKind::Expr_Print:
      return static_cast<Derived *>(this)->visitPrint(
          llvm::cast<PrintExprAST>(ptr));
    case ExprAST::ExprASTKind::Expr_Transpose:
      return static_cast<Derived *>(this)->visitTranspose(
          llvm::cast<TransposeExprAST>(ptr));
    default:
      llvm_unreachable("Unknown ExprASTKind");
    }
  }

  RetTy visitor(StmtAST *ptr) {
    if (!ptr)
      return RetTy();
    switch (ptr->getKind()) {
    case StmtAST::StmtASTKind::Stmt_VarDecl:
      return static_cast<Derived *>(this)->visitVarDecl(
          llvm::cast<VarDeclStmtAST>(ptr));
    case StmtAST::StmtASTKind::Stmt_Return:
      return static_cast<Derived *>(this)->visitReturn(
          llvm::cast<ReturnStmtAST>(ptr));
    case StmtAST::StmtASTKind::Stmt_Expr:
      return static_cast<Derived *>(this)->visitExprStmt(
          llvm::cast<ExprStmtAST>(ptr));
    default:
      llvm_unreachable("Unknown StmtASTKind");
    }
  }

  RetTy visitor(PrototypeAST *ptr) {
    return static_cast<Derived *>(this)->visitPrototype(ptr);
  }
  RetTy visitor(FunctionAST *ptr) {
    return static_cast<Derived *>(this)->visitFunction(ptr);
  }
  RetTy visitor(ModuleAST *ptr) {
    return static_cast<Derived *>(this)->visitModule(ptr);
  }
};
}} // namespace lucid_frontend::toy
#endif
