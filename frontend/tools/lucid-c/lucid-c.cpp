#include "AST.h"
#include "Lexer.h"
#include "Parser.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

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

/// Returns a lucid_frontend AST resulting from parsing the file or a nullptr on error.
static lucid_frontend::ModuleAST * parseInputFile(llvm::StringRef filename) {
  static llvm::BumpPtrAllocator allocator;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  auto buffer = fileOrErr.get()->getBuffer();
  LexerBuffer lexer(buffer.begin(), buffer.end(), filename);
  Parser parser(allocator, lexer);
  return parser.parseModule();
}

int main(int argc, char **argv) {
    cl::ParseCommandLineOptions(argc, argv, "this is a tool for LucidNPU, it will convert the Toy Language to .mlir file");
    
    auto moduleAST = parseInputFile(inputFileName);
    if (!moduleAST)
        return 1;

    switch (emitAction) {
    case Action::DumpAST:
        //dump(*moduleAST);
        return 0;
    default:
        llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    }

    return 0;
}