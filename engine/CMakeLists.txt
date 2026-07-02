cmake_minimum_required(VERSION 3.22)

project(holographic_engine VERSION 2.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
    HINTS ${CUML_ROOT}/include $ENV{CONDA_PREFIX}/include)
find_library(CUML_LIB cuml
    HINTS ${CUML_ROOT}/lib $ENV{CONDA_PREFIX}/lib)

include(CheckCXXCompilerFlag)
include(CheckCXXSourceRuns)

add_library(holo_compile_options INTERFACE)

target_compile_options(holo_compile_options INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3 -mtune=native -funroll-loops
        -fno-omit-frame-pointer -fvisibility=hidden
        -fcoroutines
        -Wall -Wextra -Wshadow -Wconversion 
        -Wundef -Wunreachable-code -Wnull-dereference
         -pedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -Xcompiler=-mtune=native,-funroll-loops,-fno-omit-frame-pointer
        -Xcompiler=-fvisibility=hidden,-Wall,-Wextra,-fcoroutines>)

set(HAS_LTO OFF)
if(HAS_LTO)
    target_compile_options(holo_compile_options INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-flto>)
    target_link_options(holo_compile_options INTERFACE -flto)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "CUDA arch" FORCE)
endif()

add_library(holo_cuda_options INTERFACE)
target_compile_options(holo_cuda_options INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -std=c++20
        -O3 --use_fast_math
        --expt-relaxed-constexpr --expt-extended-lambda
        --generate-line-info
        -Xcompiler=-fno-omit-frame-pointer,-fvisibility=hidden
        -Xptxas=--allow-expensive-optimizations=true>)

# Opt-in only: prints per-kernel register/spill stats on every compile.
# Re-enable locally with: cmake -DHOLO_VERBOSE_PTXAS=ON ..
option(HOLO_VERBOSE_PTXAS "Print ptxas register usage (-Xptxas=-v) during build" OFF)
if(HOLO_VERBOSE_PTXAS)
    target_compile_options(holo_cuda_options INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
endif()

set(HOLO_PROJECT_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(HOLO_SYSTEM_INCLUDE_DIRS
    ${CUDAToolkit_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${OPENSSL_INCLUDE_DIR}
)
if(SIMDJSON_INCLUDE_DIR)
    list(APPEND HOLO_SYSTEM_INCLUDE_DIRS ${SIMDJSON_INCLUDE_DIR})
endif()
if(CUML_INCLUDE_DIR)
    list(APPEND HOLO_SYSTEM_INCLUDE_DIRS ${CUML_INCLUDE_DIR})
endif()

# Helper: attach our own headers as normal (warning-checked) includes and
# third-party headers as SYSTEM includes, so -Wall/-Wextra/-Wconversion/
# -Wpedantic only ever fire on code we actually own.
function(holo_target_includes target scope)
    target_include_directories(${target} ${scope} ${HOLO_PROJECT_INCLUDE_DIRS})
    target_include_directories(${target} SYSTEM ${scope} ${HOLO_SYSTEM_INCLUDE_DIRS})
endfunction()

# ── holo_math ─────────────────────────────────────────────────────────────────
add_library(holo_math STATIC
    math/gpu_lob_mirror.cu
    math/lobpcg_solver.cu
    math/hodge_kernel.cu
    math/floer_homology.cu
    math/cuda_pipeline.cu
)
holo_target_includes(holo_math PUBLIC)
target_link_libraries(holo_math PUBLIC
    holo_compile_options holo_cuda_options
    CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::cusolver)
set_target_properties(holo_math PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# ── holo_ai ───────────────────────────────────────────────────────────────────
add_library(holo_ai STATIC
    ai/cuml_clustering.cu
)
holo_target_includes(holo_ai PUBLIC)
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

set(HOLO_NET_LINK_LIBS
    Boost::system Boost::json
    OpenSSL::SSL OpenSSL::Crypto
    Threads::Threads
)
if(SIMDJSON_LIB)
    list(APPEND HOLO_NET_LINK_LIBS ${SIMDJSON_LIB})
    target_compile_definitions(holo_math PUBLIC HOLO_HAS_SIMDJSON=1)
endif()

# ── holographic_live ──────────────────────────────────────────────────────────
add_executable(holographic_live app/main_live.cpp)
holo_target_includes(holographic_live PRIVATE)
target_link_libraries(holographic_live PRIVATE
    holo_compile_options holo_math holo_ai ${HOLO_NET_LINK_LIBS})
target_compile_definitions(holographic_live PRIVATE
    BOOST_ASIO_HAS_CO_AWAIT=1
    BOOST_ASIO_USE_TS_EXECUTOR_AS_DEFAULT=0)
set_target_properties(holographic_live PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    INTERPROCEDURAL_OPTIMIZATION ${HAS_LTO})

# ── holographic_backtest ──────────────────────────────────────────────────────
add_executable(holographic_backtest app/main_backtest.cpp)
holo_target_includes(holographic_backtest PRIVATE)
target_link_libraries(holographic_backtest PRIVATE
    holo_compile_options holo_math holo_ai ${HOLO_NET_LINK_LIBS})
target_compile_definitions(holographic_backtest PRIVATE
    BOOST_ASIO_HAS_CO_AWAIT=1
    BOOST_ASIO_USE_TS_EXECUTOR_AS_DEFAULT=0)
set_target_properties(holographic_backtest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    INTERPROCEDURAL_OPTIMIZATION ${HAS_LTO})
