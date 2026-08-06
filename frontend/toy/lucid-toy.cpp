#include "AST.h"
#include "Lexer.h"
#include "Parser.h"
#include "Ops.h"
#include "Dialect.h"
#include "ASTDumper.h"
#include "MLIRGen.h"
#include "Passes.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include <cstddef>
#include <optional>

namespace cl = llvm::cl;
namespace {
    enum Action {
        None, 
        DumpAST, 
        DumpToyMLIR,
        DumpMLIRAffine,
        DumpMLIRLLVM,
        DumpLLVM,
    };

    enum RunMode {
        Native,
        JIT,
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
                                        cl::values(clEnumValN(DumpMLIRAffine, "affine", "dump affine Dialect")),
                                        cl::values(clEnumValN(DumpMLIRLLVM, "LLVM", "dump LLVM Dialect")),
                                        cl::values(clEnumValN(DumpLLVM, "llvm-ir", "dump llvm IR"))
                                    );

static cl::opt<enum RunMode> runMode("run",
                                     cl::desc("Select how to run the program"),
                                     cl::values(clEnumValN(Native, "native", "generate native executable (default)")),
                                     cl::values(clEnumValN(JIT, "jit", "JIT compile and run")),
                                     cl::init(Native));

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

static mlir::OwningOpRef<mlir::ModuleOp> MLIRLowerProcess(mlir::MLIRContext &context, lucid_frontend::ModuleAST * moduleAST, bool needLLVM) 
{
    // Register the func dialect inliner extension
    mlir::DialectRegistry registry;
    mlir::func::registerInlinerExtension(registry);
    context.appendDialectRegistry(registry);
    // Load our Dialect in this MLIR Context.
    context.getOrLoadDialect<lucid_frontend::ToyDialect>();
    mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen(context, *moduleAST);
    if (!module) {
        llvm::errs() << "MLIR generation failed\n";
        return nullptr;
    }
    mlir::PassManager pm(module.get()->getName());
    // Apply any generic pass manager command line options and run the pipeline.
    if (mlir::failed(mlir::applyPassManagerCLOptions(pm))) return nullptr;
    pm.addPass(mlir::createInlinerPass());

    auto needCaonicalizer = (needLLVM || emitAction >= Action::DumpMLIRAffine);
    auto lowering2Affine = (needLLVM || emitAction >= Action::DumpMLIRAffine);
    auto lowering2LLVM = (needLLVM || emitAction >= Action::DumpMLIRLLVM);
    mlir::OpPassManager &shapePM = pm.nest<lucid_frontend::FuncOp>();
    shapePM.addPass(lucid_frontend::createShapeInferencePass());
    if (enableOpt || needCaonicalizer) {
        shapePM.addPass(mlir::createCanonicalizerPass());
        shapePM.addPass(mlir::createCSEPass());
    }

    if (lowering2Affine) {
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

    if (lowering2LLVM) {
        pm.addPass(lucid_frontend::createLowerToLLVMPass());
        pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());
    }

    if (mlir::failed(pm.run(*module))) {
        return nullptr;
    }
    return module;
}

std::unique_ptr<llvm::Module> lowerTollvm(llvm::LLVMContext& llvmContext, mlir::ModuleOp module) {
    // register the buildin dialect interface on mlir
    // this will tell the lowering process how to lower some basic mlir op to llvm ir.
    // like: mlir::ModuleOp -> llvm::Module
    mlir::registerBuiltinDialectTranslation(*module->getContext());

    // register the LLVM dialect interface 
    // this will tell the lowering process how to lower some LLVM mlir op to llvm ir.
    // like: llvm.addop -> llvm::Instruction 
    mlir::registerLLVMDialectTranslation(*module->getContext());

    
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return nullptr;
    }
    
    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Configure the LLVM Module
    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrError) {
        llvm::errs() << "Could not create JITTargetMachineBuilder\n";
        return nullptr;
    }

    auto tmOrError = tmBuilderOrError->createTargetMachine();
    if (!tmOrError) {
        llvm::errs() << "Could not create TargetMachine\n";
        return nullptr;
    }
    mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                            tmOrError.get().get());
    /// Optionally run an optimization pipeline over the llvm module.
    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);
    if (auto err = optPipeline(llvmModule.get())) {
        llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
        return nullptr;
    }
    return llvmModule;
}

int runJit(std::unique_ptr<llvm::LLVMContext> ctx, std::unique_ptr<llvm::Module> llvmModule) {
    // 1. Initialize the target and the assembly printer
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // 2. create the instance of LLJIT
    auto jitOrErr = llvm::orc::LLJITBuilder().create();
    if (!jitOrErr) {
        llvm::errs() << "Failed to create JIT: "
                     << llvm::toString(jitOrErr.takeError()) << "\n";
        return -1;
    }
    auto jit = std::move(*jitOrErr);

    // 3. Wrap llvm::Module and its Context into a thread-safe module and incorporate JIT functionality.
    llvm::orc::ThreadSafeModule tsm(std::move(llvmModule), std::move(ctx));
    if (auto err = jit->addIRModule(std::move(tsm))) {
        llvm::errs() << "Failed to add IR module to JIT: "
                     << llvm::toString(std::move(err)) << "\n";
        return -1;
    }

    // 4. Search for the entry function "main"
    auto symOrErr = jit->lookup("main");
    if (!symOrErr) {
        llvm::errs() << "Failed to lookup 'main' function: "
                     << llvm::toString(symOrErr.takeError()) << "\n";
        return -1;
    }

    // 5. Modify the function signature and execute
    // The main function signature in the toy language is void()
    auto *mainFunc = symOrErr->toPtr<void()>();
    mainFunc();
    return 0;
}
int compileNative(std::unique_ptr<llvm::Module> llvmModule, llvm::StringRef outputFilename) {
    // lowerTollvm already called InitializeNativeTarget / InitializeNativeTargetAsmPrinter,
    // but calling them again is harmless.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Triple and data layout are already set by lowerTollvm
    auto targetTriple = llvmModule->getTargetTriple();

    std::string error;
    auto *target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
    if (!target) {
        llvm::errs() << "Failed to lookup target: " << error << "\n";
        return -1;
    }

    auto cpu = "generic";
    auto features = "";
    llvm::TargetOptions opt;
    auto targetMachine = target->createTargetMachine(targetTriple, cpu, features, opt, llvm::Reloc::PIC_);
    llvmModule->setDataLayout(targetMachine->createDataLayout());

    // Emit object file
    std::string objFilename = (outputFilename + ".o").str();
    std::error_code ec;
    llvm::raw_fd_ostream dest(objFilename, ec, llvm::sys::fs::OF_None);
    if (ec) {
        llvm::errs() << "Could not open file: " << ec.message() << "\n";
        return -1;
    }

    llvm::legacy::PassManager pass;
    if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, llvm::CodeGenFileType::ObjectFile)) {
        llvm::errs() << "TargetMachine can't emit an object file\n";
        return -1;
    }

    pass.run(*llvmModule);
    dest.flush();

    // Link object file into executable using g++
    std::string cmd = "g++ " + objFilename + " -o " + outputFilename.str();
    int ret = system(cmd.c_str());
    if (ret != 0) {
        llvm::errs() << "Linking failed\n";
        return -1;
    }

    llvm::outs() << "Generated executable: " << outputFilename << "\n";
    return 0;
}

int main(int argc, char **argv) {
    mlir::registerPassManagerCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "this is a tool for LucidNPU, it will convert the Toy Language to Toy dialect format");

    auto moduleAST = parseInputFile(inputFileName);
    if (!moduleAST) {
        llvm::errs() << "AST generation failed\n";
        return 1;
    }

    if (emitAction == Action::DumpAST) {
        lucid_frontend::ASTDumper::getInstance().Dump(moduleAST);
        return 0;
    }

    // Only lower all the way to the LLVM dialect when we actually need LLVM IR
    // (either to dump llvm-ir, or to JIT/native-compile when no -emit level is
    // requested). Otherwise stop the pipeline at the requested dialect level so
    // that -emit=toy / -emit=affine / -emit=LLVM dump the expected IR.
    bool needLLVM = (emitAction == Action::None) || (emitAction == Action::DumpLLVM);
    mlir::MLIRContext context;
    mlir::OwningOpRef<mlir::ModuleOp> module = MLIRLowerProcess(context, moduleAST, needLLVM);
    if (!module) {
        llvm::errs() << "mlir generation or lowering failed\n";
        return 1;
    }

    // Dump MLIR at the requested level
    if (emitAction >= Action::DumpToyMLIR && emitAction <= Action::DumpMLIRLLVM) {
        module->dump();
        return 0;
    }

    auto llvmContext = std::make_unique<llvm::LLVMContext>();
    auto llvmModule = lowerTollvm(*llvmContext, *module);
    if (!llvmModule) {
        llvm::errs() << "llvm lowering failed\n";
        return 1;
    }

    if (emitAction == Action::DumpLLVM) {
        llvmModule->dump();
        return 0;
    }

    if (runMode == JIT) {
        return runJit(std::move(llvmContext), std::move(llvmModule));
    }

    // Native: generate executable
    return compileNative(std::move(llvmModule), "a.out");
}