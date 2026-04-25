set(TARGET_NAME lucid-c)

set(${TARGET_NAME}_LIBS
    MLIRSupport
)

add_llvm_executable(${TARGET_NAME}
    tools/lucid-c/lucid-c.cpp
)

llvm_update_compile_flags(${TARGET_NAME})

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${TARGET_NAME} PRIVATE ${${TARGET_NAME}_LIBS})
