add_subdirectory(td)

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
    MLIRCastInterfaces
)

add_llvm_executable(${TARGET_NAME}
    lib/ASTDumper.cpp
    lib/MLIRGen.cpp
    lib/ToyDialect.cpp
    lib/ShapeInferencePass.cpp
    lucid-c.cpp
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


find_program(FILECHECK_EXECUTABLE FileCheck PATHS ${LLVM_TOOLS_BINARY_DIR} NO_DEFAULT_PATH)
if(NOT FILECHECK_EXECUTABLE)
  find_program(FILECHECK_EXECUTABLE FileCheck)
endif()

if(FILECHECK_EXECUTABLE)
  set(LUCID_C_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)

  # add_lucid_c_suite(SUITE_NAME EMIT_FLAG TEST_DIR)
  #   SUITE_NAME — CTest name prefix, e.g. MLIRDumpSuite
  #   EMIT_FLAG  — passed to lucid-c, e.g. -emit=mlir
  #   TEST_DIR   — directory under frontend/test/ scanned recursively for *.toy
  function(add_lucid_c_suite name emit_flag test_dir)
    set(suite_root ${LUCID_C_TEST_DIR}/${test_dir})
    file(GLOB_RECURSE suite_tests CONFIGURE_DEPENDS "${suite_root}/*.toy")
    if(test_dir STREQUAL "mlir")
      list(FILTER suite_tests EXCLUDE REGEX "/opt/")
    endif()
    list(SORT suite_tests)

    if(NOT suite_tests)
      message(FATAL_ERROR "No *.toy tests under ${suite_root}")
    endif()

    foreach(test_file IN LISTS suite_tests)
      file(RELATIVE_PATH test_rel ${suite_root} ${test_file})
      string(REPLACE "/" "_" test_id ${test_rel})
      string(REPLACE ".toy" "" test_id ${test_id})

      file(READ "${test_file}" test_header LIMIT 128)
      if(test_header MATCHES "# RUN: not")
        add_test(NAME ${name}.${test_id}
          COMMAND bash -c
            "out=\"$($<TARGET_FILE:lucid-c> \"${test_file}\" ${emit_flag} 2>&1)\"; \
             status=$?; \
             printf '%s\\n' \"$out\" | \"${FILECHECK_EXECUTABLE}\" \"${test_file}\"; \
             test $status -ne 0")
      else()
        add_test(NAME ${name}.${test_id}
          COMMAND bash -c
            "set -o pipefail; \
             \"$<TARGET_FILE:lucid-c>\" \"${test_file}\" ${emit_flag} 2>&1 \
             | \"${FILECHECK_EXECUTABLE}\" \"${test_file}\"")
      endif()

      set_tests_properties(${name}.${test_id} PROPERTIES
        LABELS "frontend;${name}"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    endforeach()

    add_custom_target(${name}-run
      COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR}
              ${CMAKE_CTEST_COMMAND} --output-on-failure -R "^${name}\\."
      DEPENDS lucid-c
      USES_TERMINAL
      COMMENT "Run ${name}")
  endfunction()

  add_lucid_c_suite(ASTDumpSuite -emit=ast ast)
  add_lucid_c_suite(MLIRDumpSuite -emit=mlir mlir)
  add_lucid_c_suite(MLIROptSuite "-emit=mlir -opt" mlir/opt)
  add_lucid_c_suite(ErrorSuite -emit=mlir error)

  add_custom_target(check-frontend
    COMMAND ${CMAKE_COMMAND} -E chdir ${CMAKE_BINARY_DIR}
            ${CMAKE_CTEST_COMMAND} --output-on-failure -L frontend
    DEPENDS lucid-c
    USES_TERMINAL
    COMMENT "Run all frontend FileCheck tests")
else()
  message(WARNING "FileCheck not found. Integration tests are disabled.")
endif()
