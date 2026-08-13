cmake_minimum_required(VERSION 3.22)

project(holographic_engine VERSION 2.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
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
if(CUML_ROOT)
    list(APPEND CMAKE_PREFIX_PATH ${CUML_ROOT})
endif()
# Deliberately NOT hand-rolling find_path()/find_library() + manual
# -isystem paths for cuML/rmm/raft/CCCL anymore (see git history / prior
# discussion for the full story): RAPIDS vendors several parallel,
# independently-versioned copies of thrust/cub under this one prefix
# (targets/x86_64-linux/include/cub from the cuda-cccl package,
# include/rapids/cub from librmm/libraft/libcuml's own isolated overlay,
# include/libcudf/rapids/cub from libcudf's own separate overlay, plus
# cupy's vendored copy) -- deliberately, using inline-namespace version
# tagging so different RAPIDS components can each pin a different CCCL
# version without ABI collisions. Only cuml's own generated CMake config
# (which itself calls find_package(rmm CONFIG)/find_package(raft CONFIG)
# and inherits *their* correct INTERFACE_INCLUDE_DIRECTORIES, in the
# right order, for the specific CCCL copy each of them was built against)
# can get this right. Any include path we add by hand risks mixing an
# incompatible pair of these trees in one translation unit, which doesn't
# fail cleanly -- it produces a wall of unrelated-looking template errors
# out of rmm/mr/device/per_device_resource.hpp and a
# THRUST_VERSION/CUB_VERSION static_assert failure in thrust/version.h.
find_package(cuml CONFIG QUIET)

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
    # Was hardcoded to 89 (Ada Lovelace: L4/L40/RTX 40-series). PTX/SASS
    # built for a higher virtual architecture does NOT JIT down to older
    # hardware -- only forward (same-or-newer). Running arch=89 binaries
    # on an Ampere A100 (compute capability 8.0) produces exactly the
    # kind of deferred "named symbol not found (500)" cudaGetLastError()
    # surfaces here, not a clean "unsupported architecture" error at
    # kernel launch time.
    #
    # 80 = A100/A30 (Ampere, datacenter). If you build for a different
    # GPU, override explicitly: cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89
    # (Ada/L4/L40/RTX 40xx), =90 (H100), =86 (RTX 30xx), etc. -- or, on
    # CMake >=3.24 with CUDA >=11.6, use -DCMAKE_CUDA_ARCHITECTURES=native
    # to auto-detect whatever GPU is actually present at configure time
    # instead of hardcoding any single target.
    set(CMAKE_CUDA_ARCHITECTURES "80" CACHE STRING "CUDA arch" FORCE)
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
# Deliberately nothing cuML/rmm/raft/CCCL-related appended here anymore --
# that all comes from the cuml::cuml++ imported target (see the
# find_package(cuml CONFIG) comment above) via target_link_libraries on
# holo_ai below, which is the only thing that gets the include order
# right for RAPIDS's several parallel-versioned CCCL trees.

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
if(cuml_FOUND)
    target_link_libraries(holo_ai PUBLIC cuml::cuml++)
    target_compile_definitions(holo_ai PUBLIC HOLO_HAS_CUML=1)
else()
    message(STATUS
        "cuml not found via find_package(cuml CONFIG) -- holo_ai will "
        "build without HOLO_HAS_CUML (TopologyClusterer::fit() stays a "
        "no-op, see ai/cuml_clustering.cu). Pass -DCUML_ROOT=<rapids env "
        "prefix> (e.g. the conda/mamba env prefix, not just its lib/ or "
        "include/) so find_package can locate "
        "<prefix>/lib/cmake/cuml/cuml-config.cmake.")
endif()
set_target_properties(holo_ai PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

# cuda_pipeline.cu (in holo_math) now instantiates and drives
# holo::cuda::TopologyClusterer directly (stage VI wiring), so holo_math
# has a real link-time dependency on holo_ai's symbols -- previously this
# only worked "by accident" because both static libs happened to get
# linked together into holographic_live/holographic_backtest. Make it
# explicit so `cmake --build build --target holo_math` (e.g. for the
# smoke test) and any future target that links holo_math without also
# separately listing holo_ai still resolve correctly. holo_ai does not
# link back to holo_math (it only uses header-only bits of
# math/cuda_utils.cuh), so this stays a one-directional, acyclic edge.
target_link_libraries(holo_math PUBLIC holo_ai)

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
# NOTE: do NOT add BOOST_ASIO_USE_TS_EXECUTOR_AS_DEFAULT here (even as
# =0). Boost.Asio config macros like this are #ifdef-gated, not #if-gated
# -- merely *defining* it (any value) forces the legacy TS-style
# boost::asio::executor as the default executor instead of any_io_executor,
# which is broken together with experimental::awaitable_operators /
# parallel_group in Boost 1.84.0: https://github.com/chriskohlhoff/asio/issues/1397
# ("field has incomplete type 'executor_type' (aka 'void')" in
# parallel_group.hpp). The code already uses asio::any_io_executor
# explicitly throughout net/*.hpp; this macro was silently fighting that.
add_executable(holographic_live app/main_live.cpp)
holo_target_includes(holographic_live PRIVATE)
target_link_libraries(holographic_live PRIVATE
    holo_compile_options holo_math holo_ai ${HOLO_NET_LINK_LIBS})
target_compile_definitions(holographic_live PRIVATE
    BOOST_ASIO_HAS_CO_AWAIT=1)
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
    BOOST_ASIO_HAS_CO_AWAIT=1)
set_target_properties(holographic_backtest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    INTERPROCEDURAL_OPTIMIZATION ${HAS_LTO})