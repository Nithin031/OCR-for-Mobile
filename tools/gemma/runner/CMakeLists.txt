# Builds the pinned llama.cpp tools (llama-quantize, llama-perplexity) and
# gemma_runner in one CPU-only, static build, so all three come from the same
# llama.cpp commit and the same compiler flags.
#
#   cmake -S tools/gemma/runner -B <build> -DLLAMA_CPP_DIR=<llama.cpp checkout> -DCMAKE_BUILD_TYPE=Release
#   cmake --build <build> --config Release --target llama-quantize llama-perplexity gemma_runner
cmake_minimum_required(VERSION 3.18)
project(gemma_g0 C CXX)

set(LLAMA_CPP_DIR "" CACHE PATH "llama.cpp checkout at the tag in tools/gemma/llama_cpp_pin.json")
if (NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "LLAMA_CPP_DIR must point at a llama.cpp checkout (got '${LLAMA_CPP_DIR}')")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# CPU only, static, no network code, only what G0 needs.
set(BUILD_SHARED_LIBS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS     ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP       OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL         OFF CACHE BOOL "" FORCE)

add_subdirectory(${LLAMA_CPP_DIR} llama.cpp)

add_executable(gemma_runner gemma_runner.cpp)
target_link_libraries(gemma_runner PRIVATE llama ggml)
target_include_directories(gemma_runner PRIVATE ${LLAMA_CPP_DIR}/vendor)
target_compile_features(gemma_runner PRIVATE cxx_std_17)
