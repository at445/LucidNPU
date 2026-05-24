#include "ASTDumper.h"

namespace lucid_frontend {

void ASTDumper::Dump(ModuleAST *module) {
    curIndent = 0; 
    visitor(module);
}

void ASTDumper::visitPrototype(PrototypeAST *expr) {
    INDENT();
    llvm::errs() << "Proto '" << expr->getName() << "' " << loc(expr) << "\n";
    for (int i = 0; i < curIndent; i++)
      llvm::errs() << "  ";
    llvm::errs() << "Params: [";
    llvm::interleaveComma(expr->getArgs(), llvm::errs(),
                            [](auto &arg) { llvm::errs() << arg->getName(); });
    llvm::errs() << "]\n";
}
void ASTDumper::visitFunction(FunctionAST *expr) {
    INDENT();
    llvm::errs() << "Function \n";
    visitor(expr->getProto());

    INDENT();
    llvm::errs() << "Block {\n";
    for (auto &itr : expr->getBody())
        visitor(itr);
    for (int i = 0; i < curIndent; i++)
      llvm::errs() << "  ";
    llvm::errs() << "} // Block\n";
}
void ASTDumper::visitModule(ModuleAST *expr) {
    INDENT();
    llvm::errs() << "Module:\n";
    for (auto &f : *expr)
        visitor(f);
}

void ASTDumper::visitVarDeclare(VarDeclExprAST *expr) {
    INDENT();
    llvm::errs() << "VarDecl " << expr->getName();
    llvm::errs() << " ";
    printType(expr->getType());
    llvm::errs() << " " << loc(expr) << "\n";
    
    Indent level_(curIndent);
    visitor(expr->getInitVal());
}

void ASTDumper::visitReturn(ReturnExprAST *expr) {
    INDENT();
    llvm::errs() << "Return\n";
    {
        Indent level_(curIndent);
        if (expr->getExpr().has_value()) {
            visitor(*expr->getExpr());
        } else {
            INDENT();
            llvm::errs() << "(void)\n";
        }
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
        for (auto arg : expr->getArgs()) {
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

} // namespace lucid_frontend
