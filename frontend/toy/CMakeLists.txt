add_subdirectory(td)

set(TARGET_NAME lucid-toy)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)


set(${TARGET_NAME}_LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
    MLIRAnalysis
    MLIRBuiltinToLLVMIRTranslation
    MLIRCallInterfaces
    MLIRCastInterfaces
    MLIRExecutionEngine
    MLIRFunctionInterfaces
    MLIRIR
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRLLVMToLLVMIRTranslation
    MLIRMemRefDialect
    MLIRParser
    MLIRPass
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRTransforms
    LLVMX86CodeGen
    LLVMX86Desc
    LLVMX86Info
    LLVMOrcJIT
)

add_llvm_executable(${TARGET_NAME}
    lib/ASTDumper.cpp
    lib/MLIRGen.cpp
    lib/ToyDialect.cpp
    lib/ShapeInferencePass.cpp
    lib/AffineLoweringPass.cpp
    lib/CustomizedCanonicalize.cpp
    lib/LLVMLoweringPass.cpp
    lucid-toy.cpp
)

add_dependencies(${TARGET_NAME}  ToyDialectIncGen)
llvm_update_compile_flags(${TARGET_NAME})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(${TARGET_NAME} PRIVATE -O0 -ggdb3 -fno-inline)
endif()

target_include_directories(${TARGET_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/td
)
target_link_libraries(${TARGET_NAME} PRIVATE ${${TARGET_NAME}_LIBS})

add_subdirectory(test)
