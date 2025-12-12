cmake_minimum_required(VERSION 3.18) # Needed for new Python find
project(ffi_module LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Find Python (creates Python::Module target for Nanobind)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# --- Boost interval (header-only)
include_directories("D:/Work/jax-interval-arithmetic/boost_1_88_0")

# --- Add Nanobind from source
add_subdirectory("D:/Work/jax-interval-arithmetic/nanobind" nanobind_build)

# --- Build Python extension module
nanobind_add_module(ffi_module ffi.cpp)

# Save compiler intermediate files when using GCC
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # produce .ii/.s files (or use -save-temps=obj to place them next to .o files)
  target_compile_options(ffi_module PRIVATE -save-temps)
  # target_compile_options(ffi_module PRIVATE -save-temps=obj)
endif()

# --- Link Python module target
target_link_libraries(ffi_module PRIVATE Python::Module)

# --- Optional: static linking
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
