#include "AST.h"
#include "Lexer.h"
#include "Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

using namespace lucid_frontend;
namespace cl = llvm::cl;
namespace {
    enum Action {None, DumpAST};
}

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<enum Action> emitAction( "emit", 
                                        cl::desc("Select the kind of output desired"),
                                        cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")));



int main(int argc, char **argv) {
    cl::ParseCommandLineOptions(argc, argv, "this is a tool for LucidNPU, it will convert the Toy Language to .mlir file");
    return 0;
}