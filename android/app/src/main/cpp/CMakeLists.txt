cmake_minimum_required(VERSION 3.18)
project(asrllm C CXX)
set(CMAKE_CXX_STANDARD 17)
set(WHISPER_DIR ${CMAKE_SOURCE_DIR}/../../../../whisper.cpp)

# ggml (CPU backend only)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${WHISPER_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml)

add_library(asrllm SHARED
    jni.c
    ${WHISPER_DIR}/src/whisper.cpp
    ${WHISPER_DIR}/src/aneforge/whisper-aneforge.cpp)
target_include_directories(asrllm PRIVATE ${WHISPER_DIR}/include ${WHISPER_DIR}/src ${WHISPER_DIR}/ggml/include)
target_compile_definitions(asrllm PRIVATE WHISPER_VERSION="android")
target_compile_options(asrllm PRIVATE -O3)
# Baseline ARMv8 NEON only (no fp16/dotprod flags) so it runs on every 64-bit phone.
find_library(LOG_LIB log)
target_link_libraries(asrllm ggml ${LOG_LIB})
target_link_options(asrllm PRIVATE "-Wl,-z,max-page-size=16384")
