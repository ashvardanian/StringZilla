# This CMake file is heavily inspired by following `stringzilla` CMake:
# https://github.com/nlohmann/json/blob/develop/CMakeLists.txt
cmake_minimum_required(VERSION 3.1)
project(stringzilla VERSION 0.1.0 LANGUAGES C CXX)

set (CMAKE_C_STANDARD 11)
set (CMAKE_CXX_STANDARD 11)

# Determine if USearch is built as a subproject (using `add_subdirectory`) or if it is the main project
set(STRINGZILLA_IS_MAIN_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(STRINGZILLA_IS_MAIN_PROJECT ON)
endif()

# Options
option(STRINGZILLA_INSTALL "Install CMake targets" OFF)
option(STRINGZILLA_BUILD_TEST "Compile a native unit test in C++" ${STRINGZILLA_IS_MAIN_PROJECT})
option(STRINGZILLA_BUILD_BENCHMARK "Compile a native benchmark in C++" ${STRINGZILLA_IS_MAIN_PROJECT})
option(STRINGZILLA_BUILD_WOLFRAM "Compile Wolfram Language bindings" OFF)

# Includes
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
include(ExternalProject)

# Allow CMake 3.13+ to override options when using FetchContent / add_subdirectory
if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif ()

# Configuration
include(GNUInstallDirs)
set(STRINGZILLA_TARGET_NAME               ${PROJECT_NAME})
set(STRINGZILLA_CONFIG_INSTALL_DIR        "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}" CACHE INTERNAL "")
set(STRINGZILLA_INCLUDE_INSTALL_DIR       "${CMAKE_INSTALL_INCLUDEDIR}")
set(STRINGZILLA_TARGETS_EXPORT_NAME       "${PROJECT_NAME}Targets")
set(STRINGZILLA_CMAKE_CONFIG_TEMPLATE     "cmake/config.cmake.in")
set(STRINGZILLA_CMAKE_CONFIG_DIR          "${CMAKE_CURRENT_BINARY_DIR}")
set(STRINGZILLA_CMAKE_VERSION_CONFIG_FILE "${STRINGZILLA_CMAKE_CONFIG_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
set(STRINGZILLA_CMAKE_PROJECT_CONFIG_FILE "${STRINGZILLA_CMAKE_CONFIG_DIR}/${PROJECT_NAME}Config.cmake")
set(STRINGZILLA_CMAKE_PROJECT_TARGETS_FILE "${STRINGZILLA_CMAKE_CONFIG_DIR}/${PROJECT_NAME}Targets.cmake")
set(STRINGZILLA_PKGCONFIG_INSTALL_DIR      "${CMAKE_INSTALL_DATADIR}/pkgconfig")


# Define our header-only library
add_library(${STRINGZILLA_TARGET_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${STRINGZILLA_TARGET_NAME} ALIAS ${STRINGZILLA_TARGET_NAME})
set(STRINGZILLA_INCLUDE_BUILD_DIR "${PROJECT_SOURCE_DIR}/include/")

target_compile_definitions(
    ${STRINGZILLA_TARGET_NAME}
    INTERFACE
    $<$<NOT:$<BOOL:${JSON_GlobalUDLs}>>:STRINGZILLA_USE_OPENMP=0>
)
target_include_directories(
    ${STRINGZILLA_TARGET_NAME}
    ${STRINGZILLA_SYSTEM_INCLUDE} INTERFACE
    $<BUILD_INTERFACE:${STRINGZILLA_INCLUDE_BUILD_DIR}>
    $<INSTALL_INTERFACE:include>
)

if(STRINGZILLA_INSTALL)
    install(
        DIRECTORY ${STRINGZILLA_INCLUDE_BUILD_DIR}
        DESTINATION ${STRINGZILLA_INCLUDE_INSTALL_DIR}
    )
    install(
        FILES ${STRINGZILLA_CMAKE_PROJECT_CONFIG_FILE} ${STRINGZILLA_CMAKE_VERSION_CONFIG_FILE}
        DESTINATION ${STRINGZILLA_CONFIG_INSTALL_DIR}
    )
    export(
        TARGETS ${STRINGZILLA_TARGET_NAME}
        NAMESPACE ${PROJECT_NAME}::
        FILE ${STRINGZILLA_CMAKE_PROJECT_TARGETS_FILE}
    )
    install(
        TARGETS ${STRINGZILLA_TARGET_NAME}
        EXPORT ${STRINGZILLA_TARGETS_EXPORT_NAME}
        INCLUDES DESTINATION ${STRINGZILLA_INCLUDE_INSTALL_DIR}
    )
    install(
        EXPORT ${STRINGZILLA_TARGETS_EXPORT_NAME}
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${STRINGZILLA_CONFIG_INSTALL_DIR}
    )
    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
        DESTINATION ${STRINGZILLA_PKGCONFIG_INSTALL_DIR}
    )
endif()

if(${STRINGZILLA_BUILD_TEST} OR ${STRINGZILLA_BUILD_BENCHMARK})
  add_executable(stringzilla_test scripts/test.cpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

  target_include_directories(stringzilla_test PRIVATE stringzilla)
  set_target_properties(stringzilla_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

  if(${CMAKE_VERSION} VERSION_EQUAL 3.13 OR ${CMAKE_VERSION} VERSION_GREATER 3.13)
      include(CTest)
      enable_testing()
      add_test(NAME stringzilla_test COMMAND stringzilla_test)
  endif()
endif()

