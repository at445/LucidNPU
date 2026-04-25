set(RUNTIME_SOURCES
    lib/runtime.cpp
)
add_library(lucid_runtime SHARED ${RUNTIME_SOURCES})
target_include_directories(lucid_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)