cmake_minimum_required(VERSION 3.18)
project(asrllm C CXX)
set(CMAKE_CXX_STANDARD 17)
set(WHISPER_DIR ${CMAKE_SOURCE_DIR}/../../../../whisper.cpp)

# ggml: CPU backend built in several variants (armv8.0 baseline ... armv8.2 dotprod/fp16,
# armv8.6 i8mm, armv9 SVE/SME). At run time ggml scores each libggml-cpu-*.so against the
# phone's CPU features and loads the best one, so old phones still work.
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL ON CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS ON CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${WHISPER_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml)

add_library(asrllm SHARED
    jni.c
    ${WHISPER_DIR}/src/whisper.cpp
    ${WHISPER_DIR}/src/aneforge/whisper-aneforge.cpp)
target_include_directories(asrllm PRIVATE ${WHISPER_DIR}/include ${WHISPER_DIR}/src ${WHISPER_DIR}/ggml/include)
target_compile_definitions(asrllm PRIVATE WHISPER_VERSION="android" WHISPER_SHARED WHISPER_BUILD)
target_compile_options(asrllm PRIVATE -O3)
find_library(LOG_LIB log)
target_link_libraries(asrllm ggml ggml-base ${LOG_LIB})
target_link_options(asrllm PRIVATE "-Wl,-z,max-page-size=16384")
