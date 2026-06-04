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

add_library(engine_compile_options INTERFACE)

# Убрали жесткие AVX-512 флаги. -march=native сам выжмет максимум (AVX2)
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
        -Werror
        -pedantic
)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-flto HAS_LTO)
if (HAS_LTO)
    target_compile_options(engine_compile_options INTERFACE -flto)
    target_link_options(engine_compile_options INTERFACE -flto)
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
        INTERPROCEDURAL_OPTIMIZATION ${HAS_LTO}
)

message(STATUS "Build type      : ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler        : ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "LTO Enabled     : ${HAS_LTO}")