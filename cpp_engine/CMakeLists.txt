cmake_minimum_required(VERSION 3.26)

project(
        holographic_engine
        VERSION 0.4.0
        LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenSSL REQUIRED)

find_package(Boost 1.81 REQUIRED
        COMPONENTS
        system
        url
        json
)

include(CheckCXXCompilerFlag)
include(CheckCXXSourceRuns)

add_library(engine_compile_options INTERFACE)

target_compile_options(engine_compile_options INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -mtune=native
        -funroll-loops
        -fno-omit-frame-pointer
        -fvisibility=hidden
        -Wall
        -Wextra
        -Wshadow
        -Wconversion
        -Wsign-conversion
        -Wundef
        -Wunreachable-code
        -Wnull-dereference
        -Werror
        -pedantic
        >
        $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -Xcompiler=-mtune=native
        -Xcompiler=-funroll-loops
        -Xcompiler=-fno-omit-frame-pointer
        -Xcompiler=-fvisibility=hidden
        -Xcompiler=-Wall
        -Xcompiler=-Wextra
        >
)

check_cxx_compiler_flag(-flto HAS_LTO)
if (HAS_LTO)
    target_compile_options(engine_compile_options INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-flto>)
    target_link_options(engine_compile_options INTERFACE -flto)
endif ()

set(CMAKE_REQUIRED_FLAGS "-mavx512f")
check_cxx_source_runs("
#include <immintrin.h>
int main() {
    __m512 a = _mm512_setzero_ps();
    __m512 b = _mm512_add_ps(a, a);
    (void)b;
    return 0;
}
" AVX512_RUNS)
unset(CMAKE_REQUIRED_FLAGS)

if (AVX512_RUNS)
    target_compile_options(engine_compile_options INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-mavx512f -mavx512bw -mavx512dq -mavx512vl>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mavx512f,-mavx512bw,-mavx512dq,-mavx512vl>)
    target_compile_definitions(engine_compile_options INTERFACE ARCH_HAS_AVX512=1)
    message(STATUS "AVX-512 : verified at runtime")
else ()
    check_cxx_compiler_flag(-mavx2 HAS_AVX2)
    if (HAS_AVX2)
        target_compile_options(engine_compile_options INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-mavx2,-mfma>)
        target_compile_definitions(engine_compile_options INTERFACE ARCH_HAS_AVX2=1)
        message(STATUS "AVX-512 : unavailable, AVX2 fallback")
    endif ()
endif ()

set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING
        "CUDA architectures to compile for (e.g. 86, 89, 90, native)")

add_library(cuda_compile_options INTERFACE)
target_compile_options(cuda_compile_options INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -lineinfo
        --generate-line-info
        -Xcompiler=-fno-omit-frame-pointer
        -Xcompiler=-fvisibility=hidden
        -Xptxas=-v
        -Xptxas=--allow-expensive-optimizations=true
        >
)

# ── Shared CUDA objects ────────────────────────────────────────────────────────

set(HOLO_CUDA_SOURCES
        src/gpu_lob_mirror.cu
        src/lobpcg_solver.cu
        src/hodge_kernel.cu
        src/cuda_pipeline.cu
)

set(HOLO_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${OPENSSL_INCLUDE_DIR}
)

set(HOLO_LINK_LIBS
        engine_compile_options
        cuda_compile_options
        Threads::Threads
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        CUDA::cusolver
        CUDA::cuda_driver
        Boost::system
        Boost::url
        Boost::json
        OpenSSL::SSL
        OpenSSL::Crypto
)

set(HOLO_COMPILE_DEFS
        BOOST_ASIO_HAS_CO_AWAIT=1
        BOOST_ASIO_USE_TS_EXECUTOR_AS_DEFAULT=0
)

# ── holographic_bench ─────────────────────────────────────────────────────────

add_executable(holographic_bench
        src/main.cpp
        ${HOLO_CUDA_SOURCES}
)

target_include_directories(holographic_bench PRIVATE ${HOLO_INCLUDE_DIRS})
target_link_libraries(holographic_bench PRIVATE ${HOLO_LINK_LIBS})
target_compile_definitions(holographic_bench PRIVATE ${HOLO_COMPILE_DEFS})

set_target_properties(holographic_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        INTERPROCEDURAL_OPTIMIZATION ${HAS_LTO}
)

# ── holographic_backtest ──────────────────────────────────────────────────────

add_executable(holographic_backtest
        src/backtest_main.cpp
        ${HOLO_CUDA_SOURCES}
)

target_include_directories(holographic_backtest PRIVATE ${HOLO_INCLUDE_DIRS})
target_link_libraries(holographic_backtest PRIVATE ${HOLO_LINK_LIBS})
target_compile_definitions(holographic_backtest PRIVATE ${HOLO_COMPILE_DEFS})

set_target_properties(holographic_backtest PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        INTERPROCEDURAL_OPTIMIZATION ${HAS_LTO}
)

# ── Sanitizer options (shared) ────────────────────────────────────────────────

option(ENABLE_ASAN "AddressSanitizer" OFF)
if (ENABLE_ASAN)
    foreach (tgt holographic_bench holographic_backtest)
        target_compile_options(${tgt} PRIVATE -fsanitize=address,undefined)
        target_link_options(${tgt} PRIVATE -fsanitize=address,undefined)
    endforeach ()
endif ()

option(ENABLE_TSAN "ThreadSanitizer" OFF)
if (ENABLE_TSAN)
    foreach (tgt holographic_bench holographic_backtest)
        target_compile_options(${tgt} PRIVATE -fsanitize=thread)
        target_link_options(${tgt} PRIVATE -fsanitize=thread)
    endforeach ()
endif ()

message(STATUS "Build type        : ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler          : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CUDA toolkit      : ${CUDAToolkit_VERSION}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "LTO               : ${HAS_LTO}")
message(STATUS "Boost             : ${Boost_VERSION}")
message(STATUS "OpenSSL           : ${OPENSSL_VERSION}")