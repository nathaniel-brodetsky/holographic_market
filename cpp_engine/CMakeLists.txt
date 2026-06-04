cmake_minimum_required(VERSION 3.26)

project(
        holographic_engine
        VERSION 0.1.0
        LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckCXXCompilerFlag)

add_library(engine_compile_options INTERFACE)

target_compile_options(engine_compile_options INTERFACE
        -O3
        -march=native
        -mtune=native
        -funroll-loops
        -fno-exceptions
        -fno-rtti
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
)

check_cxx_compiler_flag(-flto=thin HAS_THIN_LTO)
if (HAS_THIN_LTO)
    target_compile_options(engine_compile_options INTERFACE -flto=thin)
    target_link_options(engine_compile_options INTERFACE -flto=thin)
else ()
    target_compile_options(engine_compile_options INTERFACE -flto)
    target_link_options(engine_compile_options INTERFACE -flto)
endif ()

check_cxx_compiler_flag(-mavx512f HAS_AVX512)
if (HAS_AVX512)
    target_compile_options(engine_compile_options INTERFACE
            -mavx512f
            -mavx512bw
            -mavx512dq
            -mavx512vl
            -mavx512cd
    )
    target_compile_definitions(engine_compile_options INTERFACE ARCH_HAS_AVX512=1)
else ()
    check_cxx_compiler_flag(-mavx2 HAS_AVX2)
    if (HAS_AVX2)
        target_compile_options(engine_compile_options INTERFACE -mavx2 -mfma)
        target_compile_definitions(engine_compile_options INTERFACE ARCH_HAS_AVX2=1)
    endif ()
endif ()

find_package(Threads REQUIRED)

add_executable(holographic_bench
        src/main.cpp
)

target_include_directories(holographic_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(holographic_bench PRIVATE
        engine_compile_options
        Threads::Threads
)

set_target_properties(holographic_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        INTERPROCEDURAL_OPTIMIZATION TRUE
)

option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if (ENABLE_ASAN)
    target_compile_options(holographic_bench PRIVATE -fsanitize=address,undefined)
    target_link_options(holographic_bench PRIVATE -fsanitize=address,undefined)
endif ()

option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
if (ENABLE_TSAN)
    target_compile_options(holographic_bench PRIVATE -fsanitize=thread)
    target_link_options(holographic_bench PRIVATE -fsanitize=thread)
endif ()

message(STATUS "Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler        : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "AVX-512 support : ${HAS_AVX512}")
message(STATUS "Thin LTO        : ${HAS_THIN_LTO}")