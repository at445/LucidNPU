//===- AST.h - Node definition for the Toy AST ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST for the Toy language. It is optimized for
// simplicity, not efficiency. The AST forms a tree structure allocated in
// a BumpPtrAllocator, where nodes hold unowned pointers to their children.
// Array and string properties are represented via lightweight ArrayRef and
// StringRef views to eliminate copying overhead and fit into MLIR semantics.
//
//===----------------------------------------------------------------------===//
#ifndef LUCID_FRONTEND_AST_H
#define LUCID_FRONTEND_AST_H

#include "Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <optional>
#include <utility>
#include <vector>

namespace lucid_frontend {

/// A variable type with shape information.
struct VarType {
  llvm::ArrayRef<int64_t> shape;
  VarType(llvm::ArrayRef<int64_t> shape): shape(shape){}
};

/// Base class for all expression nodes.
class ExprAST {
public:
  enum ExprASTKind {
    Expr_VarDecl,
    Expr_Return,
    Expr_Num,
    Expr_Literal,
    Expr_Var,
    Expr_BinOp,
    Expr_Call,
    Expr_Print,
    Expr_Transpose
  };

  ExprAST(ExprASTKind kind, Location location)
      : kind(kind), location(location) {}
  virtual ~ExprAST() = default;

  ExprASTKind getKind() const { return kind; }

  const Location &loc() const { return location; }

private:
  const ExprASTKind kind;
  Location location;
};

/// A block-list of expressions.
using ExprASTList = llvm::ArrayRef<ExprAST *>;

/// Expression class for numeric literals like "1.0".
class NumberExprAST : public ExprAST {
  double val;

public:
  NumberExprAST(Location loc, double val)
      : ExprAST(Expr_Num, loc), val(val) {}

  double getValue() const { return val; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Num; }
};

/// Expression class for a literal value.
class LiteralExprAST : public ExprAST {
  ExprASTList values;
  llvm::ArrayRef<int64_t> dims;

public:
  LiteralExprAST(Location loc, ExprASTList values,
                 llvm::ArrayRef<int64_t> dims)
      : ExprAST(Expr_Literal, loc), values(values), dims(dims) {}

  ExprASTList getValues() const { return values; }
  llvm::ArrayRef<int64_t> getDims() const { return dims; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Literal; }
};

/// Expression class for referencing a variable, like "a".
class VariableExprAST : public ExprAST {
  llvm::StringRef name;

public:
  VariableExprAST(Location loc, llvm::StringRef name)
      : ExprAST(Expr_Var, loc), name(name) {}

  llvm::StringRef getName() const { return name; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Var; }
};

/// Expression class for defining a variable.
class VarDeclExprAST : public ExprAST {
  llvm::StringRef name;
  std::optional<VarType*> type; // Type is optional, it can be inferred
  ExprAST* initVal;

public:
  VarDeclExprAST(Location loc, llvm::StringRef name, 
                  std::optional<VarType*> type, ExprAST* initVal)
      : ExprAST(Expr_VarDecl, loc), name(name),
        type(type), initVal(initVal) {}

  llvm::StringRef getName() const { return name; }
  ExprAST* getInitVal() const { return initVal; }
  const std::optional<VarType*> getType() const { return type; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_VarDecl; }
};

/// Expression class for a return operator.
class ReturnExprAST : public ExprAST {
  std::optional<ExprAST*> expr;

public:
  ReturnExprAST(Location loc, std::optional<ExprAST*> expr)
      : ExprAST(Expr_Return, loc), expr(expr) {}

  std::optional<ExprAST*> getExpr() const {
    return expr;
  }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Return; }
};

/// Expression class for a binary operator.
class BinaryExprAST : public ExprAST {
  char op;
  ExprAST *lhs, *rhs;

public:
  char getOp() const { return op; }
  ExprAST *getLHS() const { return lhs; }
  ExprAST *getRHS() const { return rhs; }

  BinaryExprAST(Location loc, char op, ExprAST* lhs,
                ExprAST* rhs)
      : ExprAST(Expr_BinOp, loc), op(op), lhs(lhs), rhs(rhs) {}

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_BinOp; }
};

/// Expression class for function calls.
class CallExprAST : public ExprAST {
  llvm::StringRef callee;
  ExprASTList args;

public:
  CallExprAST(Location loc, const llvm::StringRef &callee, ExprASTList args)
      : ExprAST(Expr_Call, loc), callee(callee),
        args(args) {}

  llvm::StringRef getCallee() const { return callee; }
  ExprASTList getArgs() const { return args; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Call; }
};

/// Expression class for builtin print calls.
class PrintExprAST : public ExprAST {
  ExprAST * arg;

public:
  PrintExprAST(Location loc, ExprAST* arg)
      : ExprAST(Expr_Print, loc), arg(arg) {}

  ExprAST *getArg() const { return arg; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Print; }
};

/// Expression class for builtin print calls.
class TransposeExprAST : public ExprAST {
  ExprAST * arg;

public:
  TransposeExprAST(Location loc, ExprAST* arg)
      : ExprAST(Expr_Transpose, loc), arg(arg) {}

  ExprAST *getArg() const { return arg; }

  /// LLVM style RTTI
  static bool classof(const ExprAST *c) { return c->getKind() == Expr_Transpose; }
};

/// This class represents the "prototype" for a function, which captures its
/// name, and its argument names (thus implicitly the number of arguments the
/// function takes).
class PrototypeAST {
  Location location;
  llvm::StringRef name;
  llvm::ArrayRef<VariableExprAST*> args;

public:
  PrototypeAST(Location location, const llvm::StringRef &name,
               llvm::ArrayRef<VariableExprAST*> args)
      : location(location), name(name), args(args) {}

  const Location &loc() const { return location; }
  llvm::StringRef getName() const { return name; }
  llvm::ArrayRef<VariableExprAST*> getArgs() const { return args; }
};

/// This class represents a function definition itself.
class FunctionAST {
  PrototypeAST* proto;
  ExprASTList body;

public:
  FunctionAST(PrototypeAST* proto,
              ExprASTList body)
      : proto(proto), body(body) {}
  PrototypeAST *getProto() const { return proto; }
  ExprASTList getBody() const { return body; }
};

/// This class represents a list of functions to be processed together
class ModuleAST {
  llvm::ArrayRef<FunctionAST*> functions;

public:
  ModuleAST(llvm::ArrayRef<FunctionAST*>  functions)
      : functions(functions) {}

  auto begin() { return functions.begin(); }
  auto end() { return functions.end(); }
};

} // namespace lucid_frontend

#endif // LUCID_FRONTEND_AST_H