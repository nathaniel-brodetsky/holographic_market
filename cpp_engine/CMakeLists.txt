cmake_minimum_required(VERSION 3.26)

project(holographic_engine VERSION 0.3.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost 1.74 REQUIRED COMPONENTS system)

find_path(SIMDJSON_INCLUDE_DIR simdjson.h
        HINTS /usr/include /usr/local/include REQUIRED)
find_library(SIMDJSON_LIB simdjson
        HINTS /usr/lib /usr/local/lib /usr/lib64 /usr/lib/x86_64-linux-gnu REQUIRED)

set(CUML_ROOT "" CACHE PATH "Path to cuML install (conda env)")
find_path(CUML_INCLUDE_DIR cuml/cluster/dbscan.hpp
        HINTS ${CUML_ROOT}/include /opt/miniforge3/envs/rapids-holographic/include
        REQUIRED)
find_library(CUML_LIB cuml
        HINTS ${CUML_ROOT}/lib /opt/miniforge3/envs/rapids-holographic/lib
        REQUIRED)

include(CheckCXXCompilerFlag)
include(CheckCXXSourceRuns)

add_library(engine_compile_options INTERFACE)
target_compile_options(engine_compile_options INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:
        -O3 -mtune=native -funroll-loops
        -fno-exceptions -fno-rtti -fno-omit-frame-pointer
        -fvisibility=hidden
        -Wall -Wextra -Wshadow -Wconversion -Wsign-conversion
        -Wundef -Wunreachable-code -Wnull-dereference
        -Werror -pedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -Xcompiler=-mtune=native,-funroll-loops,-fno-omit-frame-pointer
        -Xcompiler=-fvisibility=hidden,-Wall,-Wextra>)

check_cxx_compiler_flag(-flto HAS_LTO)
if (HAS_LTO)
    target_compile_options(engine_compile_options INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-flto>)
    target_link_options(engine_compile_options INTERFACE -flto)
endif ()

set(CMAKE_REQUIRED_FLAGS "-mavx512f")
check_cxx_source_runs("
#include <immintrin.h>
int main(){__m512 a=_mm512_setzero_ps();__m512 b=_mm512_add_ps(a,a);(void)b;return 0;}
" AVX512_RUNS)
unset(CMAKE_REQUIRED_FLAGS)

if (AVX512_RUNS)
    target_compile_options(engine_compile_options INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-mavx512f -mavx512bw -mavx512dq -mavx512vl>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mavx512f,-mavx512bw,-mavx512dq,-mavx512vl>)
    target_compile_definitions(engine_compile_options INTERFACE ARCH_HAS_AVX512=1)
    message(STATUS "AVX-512 : verified at runtime")
endif ()

set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA arch")

add_library(cuda_compile_options INTERFACE)
target_compile_options(cuda_compile_options INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -O3 --use_fast_math
        --expt-relaxed-constexpr --expt-extended-lambda
        -lineinfo --generate-line-info
        -Xcompiler=-fno-exceptions,-fno-rtti,-fvisibility=hidden
        -Xptxas=-v -Xptxas=--allow-expensive-optimizations=true>)

add_executable(holographic_bench
        src/main.cpp
        src/gpu_lob_mirror.cu
        src/lobpcg_solver.cu
        src/hodge_kernel.cu
        src/cuda_pipeline.cu
        src/cuml_clustering.cu
)

target_include_directories(holographic_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${SIMDJSON_INCLUDE_DIR}
        ${CUML_INCLUDE_DIR}
)

target_link_libraries(holographic_bench PRIVATE
        engine_compile_options
        cuda_compile_options
        Threads::Threads
        CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::cuda_driver
        Boost::system
        ${SIMDJSON_LIB}
        ${CUML_LIB}
        ssl crypto
)

set_target_properties(holographic_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        INTERPROCEDURAL_OPTIMIZATION ${HAS_LTO}
)

message(STATUS "Build type        : ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler          : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CUDA toolkit      : ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Boost             : ${Boost_VERSION}")
message(STATUS "simdjson include  : ${SIMDJSON_INCLUDE_DIR}")
message(STATUS "cuML include      : ${CUML_INCLUDE_DIR}")
