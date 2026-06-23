add_subdirectory(include)

set(TARGET_NAME lucid-c)

set(${TARGET_NAME}_LIBS
    MLIRAnalysis
    MLIRFunctionInterfaces
    MLIRInferTypeOpInterface
    MLIRIR
    MLIRParser
    MLIRSideEffectInterfaces
    MLIRSupport
    MLIRTransforms
)

add_llvm_executable(${TARGET_NAME}
    lib/ASTDumper.cpp
    lib/MLIRGen.cpp
    lib/ToyDialect.cpp
    lucid-c.cpp
)

add_dependencies(${TARGET_NAME} ToyOpsIncGen)

llvm_update_compile_flags(${TARGET_NAME})

# llvm_update_compile_flags inherits LLVM flags; force -O0 for readable single-stepping.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(${TARGET_NAME} PRIVATE -O0 -ggdb3 -fno-inline)
endif()

target_include_directories(${TARGET_NAME} PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/include
)
target_link_libraries(${TARGET_NAME} PRIVATE ${${TARGET_NAME}_LIBS})


# Find FileCheck for validation
find_program(FILECHECK_EXECUTABLE FileCheck PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
if(NOT FILECHECK_EXECUTABLE)
  find_program(FILECHECK_EXECUTABLE FileCheck)
endif()

if(FILECHECK_EXECUTABLE)
  set(LUCID_C_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)
  set(RUN_SUITE ${LUCID_C_TEST_DIR}/run_suite.sh)

  function(add_lucid_c_suite name suite_dir emit_flag mode label)
    add_test(NAME ${name}
      COMMAND ${RUN_SUITE}
        $<TARGET_FILE:lucid-c>
        ${FILECHECK_EXECUTABLE}
        ${LUCID_C_TEST_DIR}/${suite_dir}
        ${emit_flag}
        ${mode})
    set_tests_properties(${name} PROPERTIES
      LABELS "${label};frontend"
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    add_custom_target(${name}-run
      COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR}
              ${CMAKE_CTEST_COMMAND} --output-on-failure -R ${name}
      DEPENDS lucid-c
      USES_TERMINAL
      COMMENT "Build lucid-c and run ${name}")
  endfunction()

  add_lucid_c_suite(ASTDumpSuite ast -emit=ast positive ast)
  add_lucid_c_suite(MLIRDumpSuite mlir -emit=mlir positive mlir)
  add_lucid_c_suite(ErrorSuite error -emit=mlir negative error)

  add_custom_target(check-frontend
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR}
            ${CMAKE_CTEST_COMMAND} --output-on-failure -L frontend
    DEPENDS lucid-c
    USES_TERMINAL
    COMMENT "Build lucid-c and run all frontend FileCheck suites")
else()
  message(WARNING "FileCheck not found. Integration tests are disabled.")
endif()
