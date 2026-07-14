#include "AST.h"
#include "Lexer.h"
#include "Parser.h"
#include "Ops.h"
#include "Dialect.h"
#include "ASTDumper.h"
#include "MLIRGen.h"
#include "Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

namespace cl = llvm::cl;
namespace {
    enum Action {
        None, 
        DumpAST, 
        DumpToyMLIR,
        DumpMLIRAffine,
        DumpMLIRLLVM,
        DumpLLVM,
        RunJIT,
    };
}

static cl::opt<std::string> inputFileName(cl::Positional,
                                          cl::desc("<input toy file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<enum Action> emitAction( "emit", 
                                        cl::desc("Select the kind of output desired"),
                                        cl::values(clEnumValN(DumpAST, "ast", "dump AST")),
                                        cl::values(clEnumValN(DumpToyMLIR, "toy", "dump toy MLIR")),
                                        cl::values(clEnumValN(DumpMLIRAffine, "affine", "dump affine Dialect"))
                                    );

static cl::opt<bool> enableOpt("opt",
                               cl::desc("Enable optimizations (canonicalizer, CSE, shape inference)"),
                               cl::init(false));

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
  lucid_frontend::LexerBuffer lexer(buffer.begin(), buffer.end(), filename);
  lucid_frontend::Parser parser(allocator, lexer);
  return parser.parseModule();
}

int main(int argc, char **argv) {
    mlir::registerPassManagerCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "this is a tool for LucidNPU, it will convert the Toy Language to Toy dialect format");
    if (emitAction == Action::None) {
        llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
    }

    mlir::MLIRContext context;
    // Register the func dialect inliner extension
    mlir::DialectRegistry registry;
    mlir::func::registerInlinerExtension(registry);
    context.appendDialectRegistry(registry);
    // Load our Dialect in this MLIR Context.
    context.getOrLoadDialect<lucid_frontend::ToyDialect>();

    auto moduleAST = parseInputFile(inputFileName);
    if (!moduleAST) {
        llvm::errs() << "AST generation failed\n";
        return 1;
    }
    if (emitAction == Action::DumpAST) {
        lucid_frontend::ASTDumper::getInstance().Dump(moduleAST);
        return 0;
    }

    mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen(context, *moduleAST);
    if (!module) {
        llvm::errs() << "MLIR generation failed\n";
        return 2;
    }
    mlir::PassManager pm(module.get()->getName());
    // Apply any generic pass manager command line options and run the pipeline.
    if (mlir::failed(mlir::applyPassManagerCLOptions(pm))) return 3;
    pm.addPass(mlir::createInlinerPass());

    // Shape inference is always required for -emit=mlir and -emit=affine.
    // -opt controls additional canonicalizer + CSE passes for mlir.
    // For affine, opt is always forced on.
    bool runOpt = (emitAction == Action::DumpMLIRAffine) || enableOpt;

    if (emitAction == Action::DumpToyMLIR || emitAction == Action::DumpMLIRAffine) {
        mlir::OpPassManager &shapePM = pm.nest<lucid_frontend::FuncOp>();
        shapePM.addPass(lucid_frontend::createShapeInferencePass());
        if (runOpt) {
            shapePM.addPass(mlir::createCanonicalizerPass());
            shapePM.addPass(mlir::createCSEPass());
        }
    }

    if (emitAction == Action::DumpMLIRAffine) {
        pm.addPass(lucid_frontend::createAffineLoweringPass());
        mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
        // Add a few cleanups post lowering.
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::createCSEPass());
        if (enableOpt) {
            // Add optimizations 
            optPM.addPass(mlir::affine::createLoopFusionPass());
            optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
        }
    }
    if (mlir::failed(pm.run(*module))) return 4;
    module->print(llvm::outs());
    llvm::outs() << "\n";
    return 0;
}