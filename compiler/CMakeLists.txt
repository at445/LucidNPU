set(LUCID_OPT_LIBS
    MLIROptLib
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
)

add_llvm_executable(lucid-opt
    tools/lucid-opt/lucid-opt.cpp
)

llvm_update_compile_flags(lucid-opt)

target_include_directories(lucid-opt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(lucid-opt PRIVATE ${LUCID_OPT_LIBS})
