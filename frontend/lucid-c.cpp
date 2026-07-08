#include "AST.h"
#include "Lexer.h"
#include "Parser.h"
#include "Ops.h"
#include "Dialect.h"
#include "ASTDumper.h"
#include "MLIRGen.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace lucid_frontend;
namespace cl = llvm::cl;
namespace {
    enum Action {None, DumpAST, DumpMLIR};
}

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<enum Action> emitAction( "emit", 
                                        cl::desc("Select the kind of output desired"),
                                        cl::values(clEnumValN(DumpAST, "ast", "dump AST")),
                                        cl::values(clEnumValN(DumpMLIR, "mlir", "dump MLIR"))
                                    );

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));

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
    mlir::registerPassManagerCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "this is a tool for LucidNPU, it will convert the Toy Language to Toy dialect format");
    
    auto moduleAST = parseInputFile(inputFileName);
    
    if (!moduleAST)
        return 1;

    switch (emitAction) {
    case Action::DumpAST:
        ASTDumper::getInstance().Dump(moduleAST);
        return 0;

    case Action::DumpMLIR:
        {
            mlir::MLIRContext context;
            // Load our Dialect in this MLIR Context.
            context.getOrLoadDialect<lucid_frontend::ToyDialect>();
            mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen(context, *moduleAST);
            if (!module) {
                llvm::errs() << "MLIR generation failed\n";
                return 1;
            }
            if (enableOpt) {
                mlir::PassManager pm(module.get()->getName());
                // Apply any generic pass manager command line options and run the pipeline.
                if (mlir::failed(mlir::applyPassManagerCLOptions(pm))) return 4;

                pm.addPass(mlir::createInlinerPass());

                // Now that there is only one function, we can infer the shapes of each of
                // the operations.
                mlir::OpPassManager &optPM = pm.nest<lucid_frontend::FuncOp>();
                // Add a run of the canonicalizer to optimize the mlir module.
                optPM.addPass(mlir::createCanonicalizerPass());

                if (mlir::failed(pm.run(*module))) return 4;
            }
            module->print(llvm::outs());
            llvm::outs() << "\n";
        }
        return 0;

    default:
        llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    }

    return 0;
}