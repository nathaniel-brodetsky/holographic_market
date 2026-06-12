cmake_minimum_required(VERSION 3.26)

project(holographic_engine VERSION 2.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
find_package(Threads     REQUIRED)
find_package(OpenSSL     REQUIRED)
find_package(Boost 1.81  REQUIRED COMPONENTS system json)

find_path(SIMDJSON_INCLUDE_DIR simdjson.h
    HINTS /usr/include /usr/local/include)
find_library(SIMDJSON_LIB simdjson
    HINTS /usr/lib /usr/local/lib /usr/lib64 /usr/lib/x86_64-linux-gnu)

set(CUML_ROOT "" CACHE PATH "cuML install prefix")
find_path(CUML_INCLUDE_DIR cuml/cluster/dbscan.hpp
    HINTS ${CUML_ROOT}/include /opt/miniforge3/envs/rapids-holographic/include)
find_library(CUML_LIB cuml
    HINTS ${CUML_ROOT}/lib /opt/miniforge3/envs/rapids-holographic/lib)

# ── Compile option targets ────────────────────────────────────────────────────
include(CheckCXXCompilerFlag)
include(CheckCXXSourceRuns)

add_library(holo_compile_options INTERFACE)

# C++20 coroutines require exceptions — do NOT add -fno-exceptions here
target_compile_options(holo_compile_options INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3 -mtune=native -funroll-loops
        -fno-omit-frame-pointer -fvisibility=hidden
        -fcoroutines
        -Wall -Wextra -Wshadow -Wconversion -Wsign-conversion
        -Wundef -Wunreachable-code -Wnull-dereference
        -Werror -pedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -Xcompiler=-mtune=native,-funroll-loops,-fno-omit-frame-pointer
        -Xcompiler=-fvisibility=hidden,-Wall,-Wextra,-fcoroutines>)

check_cxx_compiler_flag(-flto HAS_LTO)
if(HAS_LTO)
    target_compile_options(holo_compile_options INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-flto>)
    target_link_options(holo_compile_options INTERFACE -flto)
endif()

set(CMAKE_REQUIRED_FLAGS "-mavx512f")
check_cxx_source_runs("
#include <immintrin.h>
int main(){__m512 a=_mm512_setzero_ps();__m512 b=_mm512_add_ps(a,a);(void)b;return 0;}
" AVX512_RUNS)
unset(CMAKE_REQUIRED_FLAGS)
if(AVX512_RUNS)
    target_compile_options(holo_compile_options INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-mavx512f -mavx512bw -mavx512dq -mavx512vl>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mavx512f,-mavx512bw,-mavx512dq,-mavx512vl>)
    target_compile_definitions(holo_compile_options INTERFACE ARCH_HAS_AVX512=1)
    message(STATUS "AVX-512 : verified at runtime")
else()
    check_cxx_compiler_flag(-mavx2 HAS_AVX2)
    if(HAS_AVX2)
        target_compile_options(holo_compile_options INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mavx2,-mfma>)
        target_compile_definitions(holo_compile_options INTERFACE ARCH_HAS_AVX2=1)
        message(STATUS "AVX-512 : unavailable, AVX2 fallback")
    endif()
endif()

set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA arch (L4/4070=89, A100=80)")

add_library(holo_cuda_options INTERFACE)
target_compile_options(holo_cuda_options INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3 --use_fast_math
        --expt-relaxed-constexpr --expt-extended-lambda
        -lineinfo --generate-line-info
        -Xcompiler=-fno-omit-frame-pointer,-fvisibility=hidden
        -Xptxas=-v -Xptxas=--allow-expensive-optimizations=true>)

# ── Shared include directories ────────────────────────────────────────────────
# Engine root is the include root so #include <core/...> <math/...> etc. resolve.
set(HOLO_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
)
if(SIMDJSON_INCLUDE_DIR)
    list(APPEND HOLO_INCLUDE_DIRS ${SIMDJSON_INCLUDE_DIR})
endif()
if(CUML_INCLUDE_DIR)
    list(APPEND HOLO_INCLUDE_DIRS ${CUML_INCLUDE_DIR})
endif()

# ── Module: holo_math (GPU topology engine) ───────────────────────────────────
add_library(holo_math STATIC
    math/gpu_lob_mirror.cu
    math/lobpcg_solver.cu
    math/hodge_kernel.cu
    math/cuda_pipeline.cu
)
target_include_directories(holo_math PUBLIC ${HOLO_INCLUDE_DIRS})
target_link_libraries(holo_math PUBLIC
    holo_compile_options holo_cuda_options
    CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::cuda_driver)
set_target_properties(holo_math PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# ── Module: holo_ai (RAPIDS cuML clustering) ─────────────────────────────────
add_library(holo_ai STATIC
    ai/cuml_clustering.cu
)
target_include_directories(holo_ai PUBLIC ${HOLO_INCLUDE_DIRS})
target_link_libraries(holo_ai PUBLIC
    holo_compile_options holo_cuda_options
    CUDA::cudart)
if(CUML_LIB)
    target_link_libraries(holo_ai PUBLIC ${CUML_LIB})
    target_compile_definitions(holo_ai PUBLIC HOLO_HAS_CUML=1)
endif()
set_target_properties(holo_ai PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# ── Shared link deps for net targets ─────────────────────────────────────────
set(HOLO_NET_LINK_LIBS
    Boost::system Boost::json
    OpenSSL::SSL OpenSSL::Crypto
    Threads::Threads
)
if(SIMDJSON_LIB)
    list(APPEND HOLO_NET_LINK_LIBS ${SIMDJSON_LIB})
    target_compile_definitions(holo_math PUBLIC HOLO_HAS_SIMDJSON=1)
endif()

# ── Executable: holographic_live ─────────────────────────────────────────────
add_executable(holographic_live app/main_live.cpp)
target_include_directories(holographic_live PRIVATE ${HOLO_INCLUDE_DIRS})
target_link_libraries(holographic_live PRIVATE
    holo_compile_options
    holo_math
    holo_ai
    ${HOLO_NET_LINK_LIBS}
)
target_compile_definitions(holographic_live PRIVATE
    BOOST_ASIO_HAS_CO_AWAIT=1
    BOOST_ASIO_USE_TS_EXECUTOR_AS_DEFAULT=0
)
set_target_properties(holographic_live PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    INTERPROCEDURAL_OPTIMIZATION ${HAS_LTO}
)

# ── Executable: holographic_backtest ─────────────────────────────────────────
add_executable(holographic_backtest app/main_backtest.cpp)
target_include_directories(holographic_backtest PRIVATE ${HOLO_INCLUDE_DIRS})
target_link_libraries(holographic_backtest PRIVATE
    holo_compile_options
    holo_math
    holo_ai
    ${HOLO_NET_LINK_LIBS}
)
target_compile_definitions(holographic_backtest PRIVATE
    BOOST_ASIO_HAS_CO_AWAIT=1
    BOOST_ASIO_USE_TS_EXECUTOR_AS_DEFAULT=0
)
set_target_properties(holographic_backtest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    INTERPROCEDURAL_OPTIMIZATION ${HAS_LTO}
)

# ── Status ────────────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "━━━━ HOLOGRAPHIC ENGINE v2.0 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
message(STATUS "Build type        : ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler          : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CUDA toolkit      : ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "LTO               : ${HAS_LTO}")
message(STATUS "Boost             : ${Boost_VERSION}")
message(STATUS "OpenSSL           : ${OPENSSL_VERSION}")
message(STATUS "simdjson          : ${SIMDJSON_INCLUDE_DIR}")
message(STATUS "cuML              : ${CUML_INCLUDE_DIR}")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  holo_math        — GPU topology engine (STATIC)")
message(STATUS "  holo_ai          — RAPIDS cuML clustering (STATIC)")
message(STATUS "  holographic_live — live trading binary")
message(STATUS "  holographic_backtest — CSV replay binary (Phase VI stub)")
message(STATUS "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

option(ENABLE_ASAN "AddressSanitizer" OFF)
if(ENABLE_ASAN)
    foreach(tgt holographic_live holographic_backtest)
        target_compile_options(${tgt} PRIVATE -fsanitize=address,undefined)
        target_link_options(${tgt} PRIVATE -fsanitize=address,undefined)
    endforeach()
endif()
