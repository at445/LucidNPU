set(TARGET_NAME lucid-c)

set(${TARGET_NAME}_LIBS
    MLIRSupport
)

add_llvm_executable(${TARGET_NAME}
    lib/ASTDumper.cpp
    tools/lucid-c/lucid-c.cpp
)

llvm_update_compile_flags(${TARGET_NAME})

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${TARGET_NAME} PRIVATE ${${TARGET_NAME}_LIBS})


# Find FileCheck for validation
find_program(FILECHECK_EXECUTABLE FileCheck PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
if(NOT FILECHECK_EXECUTABLE)
  # Fallback to system path if not found in LLVM binary dir
  find_program(FILECHECK_EXECUTABLE FileCheck)
endif()

if(FILECHECK_EXECUTABLE)
  add_test(NAME ASTTest1 COMMAND bash -c "$<TARGET_FILE:lucid-c> ${CMAKE_CURRENT_SOURCE_DIR}/test/ast_test1.toy -emit=ast 2>&1 | ${FILECHECK_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/ast_test1.toy")
  add_test(NAME ASTTest2 COMMAND bash -c "$<TARGET_FILE:lucid-c> ${CMAKE_CURRENT_SOURCE_DIR}/test/ast_test2.toy -emit=ast 2>&1 | ${FILECHECK_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test/ast_test2.toy")
else()
  message(WARNING "FileCheck not found. AST Integration tests will not be validated, only executed.")
  add_test(NAME ASTTest1 COMMAND lucid-c ${CMAKE_CURRENT_SOURCE_DIR}/test/ast_test1.toy -emit=ast)
  add_test(NAME ASTTest2 COMMAND lucid-c ${CMAKE_CURRENT_SOURCE_DIR}/test/ast_test2.toy -emit=ast)
endif()
