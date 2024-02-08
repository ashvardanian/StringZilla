cmake_minimum_required(VERSION 3.12)

project(
  stringzilla
  VERSION 3.0.0
  LANGUAGES C CXX
  DESCRIPTION "SIMD-accelerated string search, sort, hashes, fingerprints, & edit distances"
  HOMEPAGE_URL "https://github.com/ashvardanian/stringzilla")

set(PROJECT_IS_TOP_LEVEL 0)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(PROJECT_IS_TOP_LEVEL 1)
endif()

include(cmake/warning-guard.cmake)

add_library(stringzilla INTERFACE)
add_library(stringzilla::stringzilla ALIAS stringzilla)
target_include_directories(
  stringzilla ${warning_guard}
  INTERFACE "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")
target_compile_features(stringzilla INTERFACE cxx_std_11)

add_library(stringzilla_c SHARED c/lib.c)
add_library(stringzilla::stringzilla_c ALIAS stringzilla_c)
set_property(TARGET stringzilla_c PROPERTY OUTPUT_NAME stringzilla)
target_include_directories(
  stringzilla_c ${warning_guard}
  PUBLIC "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")
target_compile_features(stringzilla_c PUBLIC c_std_99)

add_library(stringzillite SHARED c/lib.c)
add_library(stringzilla::stringzillite ALIAS stringzillite)
target_compile_definitions(stringzillite PRIVATE SZ_AVOID_LIBC=1)
target_include_directories(
  stringzillite ${warning_guard}
  PUBLIC "\$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>")
target_compile_features(stringzillite PUBLIC c_std_99)

set_target_properties(
  stringzillite stringzilla_c PROPERTIES
  C_VISIBILITY_PRESET hidden
  VERSION "${PROJECT_VERSION}"
  SOVERSION 1)

if(MSVC)
  target_compile_definitions(stringzillite PRIVATE "__SIZE_TYPE__=unsigned long long")
endif()

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

if(NOT stringzilla_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
    AUTHOR_WARNING
    "Developer mode is intended for developers of stringzilla")
endif()

include(cmake/dev.cmake)
