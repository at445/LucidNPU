#include "ASTDumper.h"

namespace lucid_frontend {

void ASTDumper::Dump(ModuleAST *module) {
  curIndent = 0;
  visitor(module);
}

void ASTDumper::visitPrototype(PrototypeAST *proto) {
  INDENT();
  llvm::errs() << "Proto '" << proto->getName() << "' " << loc(proto) << "\n";
  for (int i = 0; i < curIndent; i++)
    llvm::errs() << "  ";
  llvm::errs() << "Params: [";
  llvm::interleaveComma(proto->getArgs(), llvm::errs(),
                        [](auto &arg) { llvm::errs() << arg->getName(); });
  llvm::errs() << "]\n";
}

void ASTDumper::visitFunction(FunctionAST *func) {
  INDENT();
  llvm::errs() << "Function \n";
  visitor(func->getProto());

  INDENT();
  llvm::errs() << "Block {\n";
  for (auto *stmt : func->getBody())
    visitor(stmt);
  for (int i = 0; i < curIndent; i++)
    llvm::errs() << "  ";
  llvm::errs() << "} // Block\n";
}

void ASTDumper::visitModule(ModuleAST *module) {
  INDENT();
  llvm::errs() << "Module:\n";
  for (auto *f : *module)
    visitor(f);
}

void ASTDumper::visitVarDecl(VarDeclStmtAST *stmt) {
  INDENT();
  llvm::errs() << "VarDecl " << stmt->getName();
  llvm::errs() << " ";
  printType(stmt->getType());
  llvm::errs() << " " << loc(stmt) << "\n";

  Indent level_(curIndent);
  visitor(stmt->getInitVal());
}

void ASTDumper::visitReturn(ReturnStmtAST *stmt) {
  INDENT();
  llvm::errs() << "Return\n";
  {
    Indent level_(curIndent);
    if (stmt->getExpr().has_value()) {
      visitor(*stmt->getExpr());
    } else {
      INDENT();
      llvm::errs() << "(void)\n";
    }
  }
}

void ASTDumper::visitExprStmt(ExprStmtAST *stmt) {
  INDENT();
  llvm::errs() << "ExprStmt " << loc(stmt) << "\n";
  {
    Indent level_(curIndent);
    visitor(stmt->getExpr());
  }
}

void ASTDumper::visitNumber(NumberExprAST *expr) {
  INDENT();
  llvm::errs() << expr->getValue() << " " << loc(expr) << "\n";
}

void ASTDumper::printLit(ExprAST *litOrNum) {
  if (auto *num = llvm::dyn_cast<NumberExprAST>(litOrNum)) {
    llvm::errs() << num->getValue();
    return;
  }
  auto *literal = llvm::cast<LiteralExprAST>(litOrNum);

  llvm::errs() << "<";
  llvm::interleaveComma(literal->getDims(), llvm::errs());
  llvm::errs() << ">";

  llvm::errs() << "[ ";
  llvm::interleaveComma(literal->getValues(), llvm::errs(),
                        [&](auto &elt) { printLit(elt); });
  llvm::errs() << "]";
}

void ASTDumper::visitLiteral(LiteralExprAST *expr) {
  INDENT();
  llvm::errs() << "Literal: ";
  printLit(expr);
  llvm::errs() << " " << loc(expr) << "\n";
}

void ASTDumper::visitVariable(VariableExprAST *expr) {
  INDENT();
  llvm::errs() << "var: " << expr->getName() << " " << loc(expr) << "\n";
}

void ASTDumper::visitBinary(BinaryExprAST *expr) {
  INDENT();
  llvm::errs() << "BinOp: " << expr->getOp() << " " << loc(expr) << "\n";
  {
    Indent level_(curIndent);
    visitor(expr->getLHS());
    visitor(expr->getRHS());
  }
}

void ASTDumper::visitFunctionCall(CallExprAST *expr) {
  INDENT();
  llvm::errs() << "Call '" << expr->getCallee() << "' [ " << loc(expr) << "\n";
  {
    Indent level_(curIndent);
    for (auto *arg : expr->getArgs()) {
      visitor(arg);
    }
  }
  INDENT();
  llvm::errs() << "]\n";
}

void ASTDumper::visitPrint(PrintExprAST *expr) {
  INDENT();
  llvm::errs() << "Print [ " << loc(expr) << "\n";
  {
    Indent level_(curIndent);
    visitor(expr->getArg());
  }
  INDENT();
  llvm::errs() << "]\n";
}

void ASTDumper::visitTranspose(TransposeExprAST *expr) {
  INDENT();
  llvm::errs() << "Transpose [ " << loc(expr) << "\n";
  {
    Indent level_(curIndent);
    visitor(expr->getArg());
  }
  INDENT();
  llvm::errs() << "]\n";
}

void ASTDumper::printType(std::optional<VarType *> type) {
  llvm::errs() << "<";
  if (type && *type)
    llvm::interleaveComma((*type)->shape, llvm::errs());
  llvm::errs() << ">";
}

} // namespace lucid_frontend
