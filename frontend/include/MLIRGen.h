#ifndef LUCID_FRONTEND_MLIR_GEN_H
#define LUCID_FRONTEND_MLIR_GEN_H
#include "ASTVisitor.h"
namespace lucid_frontend {
class MLIRGen : public ASTVisitor<MLIRGen> {
    friend class ASTVisitor<MLIRGen>;

public:
    MLIRGen(const MLIRGen&) = delete;
    MLIRGen& operator=(const MLIRGen&) = delete;

    static MLIRGen& getInstance() {
        static MLIRGen instance;
        return instance;
    }

    void Generate(ModuleAST *expr);

private:
    MLIRGen() = default;

    void visitVarDeclare(VarDeclExprAST *expr);
    void visitReturn(ReturnExprAST *expr);
    void visitNumber(NumberExprAST *expr);
    void visitLiteral(LiteralExprAST *expr);
    void visitVariable(VariableExprAST *expr);
    void visitBinary(BinaryExprAST *expr);
    void visitFunctionCall(CallExprAST *expr);
    void visitPrint(PrintExprAST *expr);
    void visitTranspose(TransposeExprAST *expr);
    void visitPrototype(PrototypeAST *expr);
    void visitFunction(FunctionAST *expr);
    void visitModule(ModuleAST *expr);
    void printType(std::optional<VarType*> type);
};
}// namespace lucid_frontend
#endif