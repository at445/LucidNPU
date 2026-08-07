#ifndef LUCID_FRONTEND_AST_DUMPER_H
#define LUCID_FRONTEND_AST_DUMPER_H

#include "ASTVisitor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/STLExtras.h"
#include <string>

namespace lucid_frontend { 
namespace toy {

/// Return a formatted string for the location of any node
template <typename T>
static std::string loc(T *node) {
  const auto &loc = node->loc();
  return (llvm::Twine("@") + loc.file + ":" + llvm::Twine(loc.line) + ":" +
          llvm::Twine(loc.col))
      .str();
}

class ASTDumper : public ASTVisitor<ASTDumper> {
  friend class ASTVisitor<ASTDumper>;

public:
  ASTDumper(const ASTDumper &) = delete;
  ASTDumper &operator=(const ASTDumper &) = delete;

  static ASTDumper &getInstance() {
    static ASTDumper instance;
    return instance;
  }

  void Dump(ModuleAST *module);

private:
  ASTDumper() = default;
  void visitVarDecl(VarDeclStmtAST *stmt);
  void visitReturn(ReturnStmtAST *stmt);
  void visitExprStmt(ExprStmtAST *stmt);
  void visitNumber(NumberExprAST *expr);
  void visitLiteral(LiteralExprAST *expr);
  void visitVariable(VariableExprAST *expr);
  void visitBinary(BinaryExprAST *expr);
  void visitFunctionCall(CallExprAST *expr);
  void visitPrint(PrintExprAST *expr);
  void visitTranspose(TransposeExprAST *expr);
  void visitPrototype(PrototypeAST *proto);
  void visitFunction(FunctionAST *func);
  void visitModule(ModuleAST *module);
  void printType(std::optional<VarType *> type);
  void printLit(ExprAST *litOrNum);

  struct Indent {
    Indent(int &level) : level(level) { ++level; }
    ~Indent() { --level; }
    int &level;
  };

  void INDENT() {
    for (int i = 0; i < curIndent; i++) {
      llvm::errs() << "  ";
    }
  }

  int curIndent = 0;
};
}} // namespace lucid_frontend::toy
#endif
