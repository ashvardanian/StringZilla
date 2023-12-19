cmake_minimum_required(VERSION 3.1)
project(
  stringzilla
  VERSION 0.1.0
  LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 20)

# Set a default build type to "Release" if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Determine if StringZilla is built as a subproject (using `add_subdirectory`)
# or if it is the main project
set(STRINGZILLA_IS_MAIN_PROJECT OFF)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(STRINGZILLA_IS_MAIN_PROJECT ON)
endif()

# Options
option(STRINGZILLA_INSTALL "Install CMake targets" OFF)
option(STRINGZILLA_BUILD_TEST "Compile a native unit test in C++"
  ${STRINGZILLA_IS_MAIN_PROJECT})
option(STRINGZILLA_BUILD_BENCHMARK "Compile a native benchmark in C++"
  ${STRINGZILLA_IS_MAIN_PROJECT})
option(STRINGZILLA_BUILD_WOLFRAM "Compile Wolfram Language bindings" OFF)

# Includes
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
include(ExternalProject)
include(CheckCSourceCompiles)

# Allow CMake 3.13+ to override options when using FetchContent /
# add_subdirectory
if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

# Configuration
include(GNUInstallDirs)
set(STRINGZILLA_TARGET_NAME ${PROJECT_NAME})
set(STRINGZILLA_INCLUDE_BUILD_DIR "${PROJECT_SOURCE_DIR}/include/")

# Define our library
add_library(${STRINGZILLA_TARGET_NAME} INTERFACE)

target_include_directories(
  ${STRINGZILLA_TARGET_NAME}
  INTERFACE $<BUILD_INTERFACE:${STRINGZILLA_INCLUDE_BUILD_DIR}>
  $<INSTALL_INTERFACE:include>)

# Conditional Compilation for Specialized Implementations
# check_c_source_compiles(" #include <immintrin.h> int main() { __m256i v =
# _mm256_set1_epi32(0); return 0; }" STRINGZILLA_HAS_AVX2)
# if(STRINGZILLA_HAS_AVX2) target_sources(${STRINGZILLA_TARGET_NAME} PRIVATE
# "src/avx2.c") endif()
if(STRINGZILLA_INSTALL)
  install(
    TARGETS ${STRINGZILLA_TARGET_NAME}
    EXPORT ${STRINGZILLA_TARGETS_EXPORT_NAME}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES
    DESTINATION ${STRINGZILLA_INCLUDE_INSTALL_DIR})
  install(DIRECTORY ${STRINGZILLA_INCLUDE_BUILD_DIR}
    DESTINATION ${STRINGZILLA_INCLUDE_INSTALL_DIR})
endif()

if(${STRINGZILLA_BUILD_TEST} OR ${STRINGZILLA_BUILD_BENCHMARK})
  add_executable(stringzilla_bench scripts/bench_substring.cpp)
  target_link_libraries(stringzilla_bench PRIVATE ${STRINGZILLA_TARGET_NAME})
  set_target_properties(stringzilla_bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY
    ${CMAKE_BINARY_DIR})
  target_link_options(stringzilla_bench PRIVATE
    "-Wl,--unresolved-symbols=ignore-all")

  # Check for compiler and set flags for stringzilla_bench
  if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
    # Set -march=native and -fmax-errors=1 for all build types
    target_compile_options(stringzilla_bench PRIVATE "-march=native")
    target_compile_options(stringzilla_bench PRIVATE "-fmax-errors=1")

    # Set -O3 for Release build, and -g for Debug and RelWithDebInfo
    target_compile_options(stringzilla_bench PRIVATE
      "$<$<CONFIG:Release>:-O3>"
      "$<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:-g>")
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    # Intel specific flags
    target_compile_options(stringzilla_bench PRIVATE "-xHost")
  elseif(${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
    # MSVC specific flags or other settings
  endif()

  if(${CMAKE_VERSION} VERSION_EQUAL 3.13 OR ${CMAKE_VERSION} VERSION_GREATER
    3.13)
    include(CTest)
    enable_testing()
    add_test(NAME stringzilla_bench COMMAND stringzilla_bench)
  endif()
endif()
