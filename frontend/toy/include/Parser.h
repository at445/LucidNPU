//===- Parser.h - Toy Language Parser -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the parser for the Toy language. It processes the Token
// provided by the Lexer and returns an AST allocated via BumpPtrAllocator.
//
//===----------------------------------------------------------------------===//

#ifndef LUCID_FRONTEND_PARSER_H
#define LUCID_FRONTEND_PARSER_H

#include "AST.h"
#include "Lexer.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/StringSaver.h"

#include <optional>
#include <utility>
#include <vector>

namespace lucid_frontend {

/// This is a simple recursive parser for the Toy language. It produces a well
/// formed AST from a stream of Token supplied by the Lexer. Memory is managed
/// via an externally-provided llvm::BumpPtrAllocator so that the AST outlives 
/// the parsing phase without excessive smart pointer overhead. No semantic checks
/// or symbol resolution is performed. For example, variables are referenced by
/// string and the code could reference an undeclared variable and the parsing
/// succeeds.
class Parser {
public:
  /// Create a Parser for the supplied lexer and a memory arena allocator.
  Parser(llvm::BumpPtrAllocator &allocator, Lexer &lexer) 
    : allocator(allocator), lexer(lexer) {}

  /// Parse a full Module. A module is a list of function definitions.
  ModuleAST * parseModule() {
    lexer.getNextToken(); // prime the lexer

    // Parse functions one at a time and accumulate in a temporary std::vector
    // before copying them securely into the memory arena.
    std::vector<FunctionAST *> functions;
    while (auto f = parseDefinition()) {
      functions.push_back(f);
      if (lexer.getCurToken() == tok_eof)
        break;
    }
    // If we didn't reach EOF, there was an error during parsing
    if (lexer.getCurToken() != tok_eof)
      return parseError<ModuleAST>("nothing", "at end of module");

    return new (allocator) ModuleAST(AllocaInArena(functions));
  }

private:
  
  llvm::BumpPtrAllocator &allocator;
  llvm::StringSaver saver{allocator};
  Lexer &lexer;

  /// Parse a return statement.
  /// return :== return ; | return expr ;
  ReturnStmtAST *parseReturn() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_return);

    // return takes an optional argument
    std::optional<ExprAST *> expr;
    if (lexer.getCurToken() != ';') {
      expr = parseExpression();
      if (!expr)
        return nullptr;
    }
    return new (allocator) ReturnStmtAST(loc, expr);
  }

  /// Parse a literal number.
  /// numberexpr ::= number
  ExprAST* parseNumberExpr() {
    auto loc = lexer.getLastLocation();
    auto result = new (allocator) NumberExprAST(loc, lexer.getValue());
    lexer.consume(tok_number);
    return result;
  }

  /// Parse a literal array expression.
  /// tensorLiteral ::= [ literalList ] | number
  /// literalList ::= tensorLiteral | tensorLiteral, literalList
  ExprAST* parseTensorLiteralExpr() {
    auto loc = lexer.getLastLocation();
    lexer.consume(Token('['));

    // Hold the list of values at this nesting level.
    std::vector<ExprAST *> values;
    // Hold the dimensions for all the nesting inside this level.
    std::vector<int64_t> dims;
    do {
      // We can have either another nested array or a number literal.
      if (lexer.getCurToken() == '[') {
        values.push_back(parseTensorLiteralExpr());
        if (!values.back())
          return nullptr; // parse error in the nested array.
      } else {
        if (lexer.getCurToken() != tok_number)
          return parseError<ExprAST>("<num> or [", "in literal expression");
        values.push_back(parseNumberExpr());
      }

      // End of this list on ']'
      if (lexer.getCurToken() == ']')
        break;

      // Elements are separated by a comma.
      if (lexer.getCurToken() != ',')
        return parseError<ExprAST>("] or ,", "in literal expression");

      lexer.getNextToken(); // eat ,
    } while (true);
    if (values.empty())
      return parseError<ExprAST>("<something>", "to fill literal expression");
    lexer.getNextToken(); // eat ]

    /// Fill in the dimensions now. First the current nesting level:
    dims.push_back(values.size());

    /// If there is any nested array, process all of them and ensure that
    /// dimensions are uniform.
    if (llvm::any_of(values, [](ExprAST* expr) {
          return llvm::isa<LiteralExprAST>(expr);
        })) {
      auto *firstLiteral = llvm::dyn_cast<LiteralExprAST>(values.front());
      if (!firstLiteral)
        return parseError<ExprAST>("uniform well-nested dimensions",
                                   "inside literal expression");

      // Append the nested dimensions to the current level
      const llvm::ArrayRef<int64_t> firstDims = firstLiteral->getDims();
      for (int64_t dim : firstDims)
        dims.push_back(dim);

      // Sanity check that shape is uniform across all elements of the list.
      for (auto &expr : values) {
        auto *exprLiteral = llvm::cast<LiteralExprAST>(expr);
        if (!exprLiteral)
          return parseError<ExprAST>("uniform well-nested dimensions",
                                     "inside literal expression");
        if (exprLiteral->getDims() != firstDims)
          return parseError<ExprAST>("uniform well-nested dimensions",
                                     "inside literal expression");
      }
    }

    return new (allocator) LiteralExprAST(loc, AllocaInArena(values),AllocaInArena(dims));
  }

  /// parenexpr ::= '(' expression ')'
  ExprAST* parseParenExpr() {
    lexer.getNextToken(); // eat (.
    auto v = parseExpression();
    if (!v)
      return nullptr;

    if (lexer.getCurToken() != ')')
      return parseError<ExprAST>(")", "to close expression with parentheses");
    lexer.consume(Token(')'));
    return v;
  }

  /// identifierexpr
  ///   ::= identifier
  ///   ::= identifier '(' expression ')'
  ExprAST* parseIdentifierExpr() {

    auto name = saver.save(lexer.getId());
    auto loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat identifier.

    if (lexer.getCurToken() != '(') // Simple variable ref.
      return new (allocator) VariableExprAST(loc, name);

    // This is a function call.
    lexer.consume(Token('('));
    std::vector<ExprAST*> args;
    if (lexer.getCurToken() != ')') {
      while (true) {
        if (auto arg = parseExpression())
          args.push_back(arg);
        else
          return nullptr;

        if (lexer.getCurToken() == ')')
          break;

        if (lexer.getCurToken() != ',')
          return parseError<ExprAST>(", or )", "in argument list");
        lexer.getNextToken();
      }
    }
    lexer.consume(Token(')'));

    // It can be a builtin call to print
    if (name == "print") {
      if (args.size() != 1)
        return parseError<ExprAST>("<single arg>", "as argument to print()");

      return new (allocator) PrintExprAST(loc, args.front());
    } else if (name == "transpose") {
      if (args.size() != 1)
        return parseError<ExprAST>("<single arg>", "as argument to print()");

      return new (allocator) TransposeExprAST(loc, args.front());
    }

    // Call to a user-defined function
    return new (allocator)CallExprAST(loc, name, AllocaInArena(args));
  }

  /// primary
  ///   ::= identifierexpr
  ///   ::= numberexpr
  ///   ::= parenexpr
  ///   ::= tensorliteral
  ExprAST* parsePrimary() {
    switch (lexer.getCurToken()) {
    default:
      llvm::errs() << "unknown token '" << lexer.getCurToken()
                   << "' when expecting an expression\n";
      return nullptr;
    case tok_identifier:
      return parseIdentifierExpr();
    case tok_number:
      return parseNumberExpr();
    case '(':
      return parseParenExpr();
    case '[':
      return parseTensorLiteralExpr();
    case ';':
      return nullptr;
    case '}':
      return nullptr;
    }
  }

  /// Recursively parse the right hand side of a binary expression, the ExprPrec
  /// argument indicates the precedence of the current binary operator.
  ///
  /// binoprhs ::= ('+' primary)*
  ExprAST* parseBinOpRHS(int exprPrec, ExprAST* lhs) {
    // If this is a binop, find its precedence.
    while (true) {
      int tokPrec = getTokPrecedence();

      // If this is a binop that binds at least as tightly as the current binop,
      // consume it, otherwise we are done.
      if (tokPrec < exprPrec)
        return lhs;

      // Okay, we know this is a binop.
      int binOp = lexer.getCurToken();
      lexer.consume(Token(binOp));
      auto loc = lexer.getLastLocation();

      // Parse the primary expression after the binary operator.
      auto rhs = parsePrimary();
      if (!rhs)
        return parseError<ExprAST>("expression", "to complete binary operator");

      // If BinOp binds less tightly with rhs than the operator after rhs, let
      // the pending operator take rhs as its lhs.
      int nextPrec = getTokPrecedence();
      if (tokPrec < nextPrec) {
        rhs = parseBinOpRHS(tokPrec + 1, std::move(rhs));
        if (!rhs)
          return nullptr;
      }

      // Merge lhs/RHS.
      lhs = new (allocator) BinaryExprAST(loc, binOp, lhs, rhs);
    }
  }

  /// expression::= primary binop rhs
  ExprAST* parseExpression() {
    auto lhs = parsePrimary();
    if (!lhs)
      return nullptr;

    return parseBinOpRHS(0, lhs);
  }

  /// type ::= < shape_list >
  /// shape_list ::= num | num , shape_list
  VarType* parseType() {
    if (lexer.getCurToken() != '<')
      return parseError<VarType>("<", "to begin type");
    lexer.getNextToken(); // eat <


    std::vector<int64_t> shape;

    while (lexer.getCurToken() == tok_number) {
      shape.push_back(lexer.getValue());
      lexer.getNextToken();
      if (lexer.getCurToken() == ',')
        lexer.getNextToken();
    }

    if (lexer.getCurToken() != '>')
      return parseError<VarType>(">", "to end type");
    lexer.getNextToken(); // eat >
    return new (allocator) VarType(AllocaInArena(shape));
  }

  /// Parse a variable declaration, it starts with a `var` keyword followed by
  /// and identifier and an optional type (shape specification) before the
  /// initializer.
  /// decl ::= var identifier [ type ] = expr
  VarDeclStmtAST *parseDeclaration() {
    if (lexer.getCurToken() != tok_var)
      return parseError<VarDeclStmtAST>("var", "to begin declaration");
    auto loc = lexer.getLastLocation();
    lexer.getNextToken(); // eat var

    if (lexer.getCurToken() != tok_identifier)
      return parseError<VarDeclStmtAST>("identified",
                                        "after 'var' declaration");
    auto id = saver.save(lexer.getId());
    lexer.getNextToken(); // eat id

    std::optional<VarType *> type; // Type is optional, it can be inferred
    if (lexer.getCurToken() == '<') {
      type = parseType();
      if (!type)
        return nullptr;
    }

    lexer.consume(Token('='));
    auto expr = parseExpression();
    return new (allocator) VarDeclStmtAST(loc, id, type, expr);
  }

  /// Parse a block: a list of statements separated by semicolons and wrapped in
  /// curly braces.
  ///
  /// block ::= { statement_list }
  /// statement_list ::= block_stmt ; statement_list
  /// block_stmt ::= decl | "return" | expr
  StmtASTList *parseBlock() {
    if (lexer.getCurToken() != '{')
      return parseError<StmtASTList>("{", "to begin block");
    lexer.consume(Token('{'));

    std::vector<StmtAST *> stmts;

    // Ignore empty statements: swallow sequences of semicolons.
    while (lexer.getCurToken() == ';')
      lexer.consume(Token(';'));

    while (lexer.getCurToken() != '}' && lexer.getCurToken() != tok_eof) {
      if (lexer.getCurToken() == tok_var) {
        auto varDecl = parseDeclaration();
        if (!varDecl)
          return nullptr;
        stmts.push_back(varDecl);
      } else if (lexer.getCurToken() == tok_return) {
        auto ret = parseReturn();
        if (!ret)
          return nullptr;
        stmts.push_back(ret);
      } else {
        auto loc = lexer.getLastLocation();
        auto expr = parseExpression();
        if (!expr)
          return nullptr;
        stmts.push_back(new (allocator) ExprStmtAST(loc, expr));
      }
      // Ensure that elements are separated by a semicolon.
      if (lexer.getCurToken() != ';')
        return parseError<StmtASTList>(";", "after statement");

      // Ignore empty statements: swallow sequences of semicolons.
      while (lexer.getCurToken() == ';')
        lexer.consume(Token(';'));
    }

    if (lexer.getCurToken() != '}')
      return parseError<StmtASTList>("}", "to close block");

    lexer.consume(Token('}'));
    return new (allocator) StmtASTList(AllocaInArena(stmts));
  }

  /// prototype ::= def id '(' decl_list ')'
  /// decl_list ::= identifier | identifier, decl_list
  PrototypeAST * parsePrototype() {
    auto loc = lexer.getLastLocation();

    if (lexer.getCurToken() != tok_def)
      return parseError<PrototypeAST>("def", "in prototype");
    lexer.consume(tok_def);

    if (lexer.getCurToken() != tok_identifier)
      return parseError<PrototypeAST>("function name", "in prototype");

    auto fnName = saver.save(lexer.getId());
    lexer.consume(tok_identifier);

    if (lexer.getCurToken() != '(')
      return parseError<PrototypeAST>("(", "in prototype");
    lexer.consume(Token('('));

    std::vector<VariableExprAST*> args;
    if (lexer.getCurToken() != ')') {
      do {
        auto name = saver.save(lexer.getId());
        auto loc = lexer.getLastLocation();
        lexer.consume(tok_identifier);
        args.push_back(new (allocator)VariableExprAST(loc, name));
        if (lexer.getCurToken() != ',')
          break;
        lexer.consume(Token(','));
        if (lexer.getCurToken() != tok_identifier)
          return parseError<PrototypeAST>(
              "identifier", "after ',' in function parameter list");
      } while (true);
    }
    if (lexer.getCurToken() != ')')
      return parseError<PrototypeAST>(")", "to end function prototype");

    // success.
    lexer.consume(Token(')'));
    return new (allocator)PrototypeAST(loc, fnName, AllocaInArena(args));
  }

  /// Parse a function definition, we expect a prototype initiated with the
  /// `def` keyword, followed by a block containing a list of expressions.
  ///
  /// definition ::= prototype block
  FunctionAST * parseDefinition() {
    auto proto = parsePrototype();
    if (!proto)
      return nullptr;

    if (auto block = parseBlock())
      return new (allocator)FunctionAST(proto, *block);
    return nullptr;
  }

  /// Get the precedence of the pending binary operator token.
  int getTokPrecedence() {
    if (!isascii(lexer.getCurToken()))
      return -1;

    // 1 is lowest precedence.
    switch (static_cast<char>(lexer.getCurToken())) {
    case '-':
    case '+':
      return 20;
    case '*':
    case '/':
      return 40; // element-wise mul / div
    case '@':
      return 50; // matrix multiplication
    default:
      return -1;
    }
  }

  /// Helper function to signal errors while parsing, it takes an argument
  /// indicating the expected token and another argument giving more context.
  /// Location is retrieved from the lexer to enrich the error message.
  template <typename R, typename T, typename U = const char *>
  R* parseError(T &&expected, U &&context = "") {
    auto curToken = lexer.getCurToken();
    llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", "
                 << lexer.getLastLocation().col << "): expected '" << expected
                 << "' " << context << " but has Token " << curToken;
    if (isprint(curToken))
      llvm::errs() << " '" << (char)curToken << "'";
    llvm::errs() << "\n";
    return nullptr;
  }

  /// Helper function to allocate elements from a temporary std::vector into
  /// the BumpPtrAllocator and return a lightweight llvm::ArrayRef view.
  /// This ensures dynamically-sized list nodes outlive the parsing phase.
  template <typename T>
  llvm::ArrayRef<T> AllocaInArena(const std::vector<T>& vec) {
    T * buffer = allocator.Allocate<T>(vec.size());
    std::uninitialized_copy(vec.begin(), vec.end(), buffer);
    return llvm::ArrayRef<T>(buffer, vec.size());
  }

};

} // namespace lucid_frontend

#endif // LUCID_FRONTEND_PARSER_H
