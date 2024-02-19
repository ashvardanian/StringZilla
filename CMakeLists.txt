cmake_minimum_required(VERSION 3.1)
project(
  stringzilla
  VERSION 3.1.2
  LANGUAGES C CXX
  DESCRIPTION "SIMD-accelerated string search, sort, hashes, fingerprints, & edit distances"
  HOMEPAGE_URL "https://github.com/ashvardanian/stringzilla")

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17) # This gives many issues for msvc and clang-cl, especially if later on you set it to std-c++11 later on in the tests...

set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_COMPILE_WARNING_AS_ERROR)
set(DEV_USER_NAME $ENV{USER})

message(STATUS "C Compiler ID: ${CMAKE_C_COMPILER_ID}")
message(STATUS "C Compiler Version: ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Set a default build type to "Release" if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE
    Release
    CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# Determine if StringZilla is built as a subproject (using `add_subdirectory`)
# or if it is the main project
set(STRINGZILLA_IS_MAIN_PROJECT OFF)

if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(STRINGZILLA_IS_MAIN_PROJECT ON)
endif()

# Installation options
option(STRINGZILLA_INSTALL "Install CMake targets" OFF)
option(STRINGZILLA_BUILD_TEST "Compile a native unit test in C++"
  ${STRINGZILLA_IS_MAIN_PROJECT})
option(STRINGZILLA_BUILD_BENCHMARK "Compile a native benchmark in C++"
  ${STRINGZILLA_IS_MAIN_PROJECT})
option(STRINGZILLA_BUILD_SHARED "Compile a dynamic library" ${STRINGZILLA_IS_MAIN_PROJECT})
set(STRINGZILLA_TARGET_ARCH
  ""
  CACHE STRING "Architecture to tell the compiler to optimize for (-march)")

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
add_library(${PROJECT_NAME}::${STRINGZILLA_TARGET_NAME} ALIAS ${STRINGZILLA_TARGET_NAME})

target_include_directories(
  ${STRINGZILLA_TARGET_NAME}
  INTERFACE $<BUILD_INTERFACE:${STRINGZILLA_INCLUDE_BUILD_DIR}>
  $<INSTALL_INTERFACE:include>)

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

if(${CMAKE_VERSION} VERSION_EQUAL 3.13 OR ${CMAKE_VERSION} VERSION_GREATER 3.13)
  include(CTest)
  enable_testing()
endif()

# Function to set compiler-specific flags
function(set_compiler_flags target cpp_standard target_arch)
  target_include_directories(${target} PRIVATE scripts)
  target_link_libraries(${target} PRIVATE ${STRINGZILLA_TARGET_NAME})

  # Set output directory for single-configuration generators (like Make)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<0:>
  )

  # Set output directory for multi-configuration generators (like Visual Studio)
  foreach(config IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER ${config} config_upper)
    set_target_properties(${target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_BINARY_DIR}/$<0:>
    )
  endforeach()

  # Set the C++ standard
  if(NOT ${cpp_standard} STREQUAL "")
    # Use the /Zc:__cplusplus flag to correctly define the __cplusplus macro in MSVC
    set(CXX_STANDARD_MSVC "/std:c++${cpp_standard};/Zc:__cplusplus")
    set(CXX_STANDARD_GNU "-std=c++${cpp_standard}")
    target_compile_options(${target} PRIVATE
      "$<$<CXX_COMPILER_ID:MSVC>:${CXX_STANDARD_MSVC}>"
      "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>:${CXX_STANDARD_GNU}>"
    )
  endif()

  # Maximum warnings level & warnings as error.
  # MVC uses numeric values:
  # > 4068 for "unknown pragmas".
  # > 4146 for "unary minus operator applied to unsigned type, result still unsigned".
  target_compile_options(
    ${target}
    PRIVATE
    "$<$<CXX_COMPILER_ID:MSVC>:/STOP;/wd4068;/wd4146>" # For MSVC, /WX would have been sufficient
    "$<$<CXX_COMPILER_ID:GNU>:-Wall;-Wextra;-pedantic;-Werror;-Wfatal-errors;-Wno-unknown-pragmas;-Wno-cast-function-type;-Wno-unused-function>"
    "$<$<CXX_COMPILER_ID:Clang>:-Wall;-Wextra;-pedantic;-Werror;-Wfatal-errors;-Wno-unknown-pragmas>"
    "$<$<CXX_COMPILER_ID:AppleClang>:-Wall;-Wextra;-pedantic;-Werror;-Wfatal-errors;-Wno-unknown-pragmas>"
  )

  # Set optimization options for different compilers differently
  target_compile_options(
    ${target}
    PRIVATE
    "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-O3>"
    "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:-g>"
    "$<$<AND:$<CXX_COMPILER_ID:Clang>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:-O3>"
    "$<$<AND:$<CXX_COMPILER_ID:Clang>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:-g>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<OR:$<CONFIG:Release>,$<CONFIG:RelWithDebInfo>>>:/O2>"
    "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>>:/Zi>"
  )

  # If available, enable Position Independent Code
  if(CMAKE_POSITION_INDEPENDENT_CODE)
    target_compile_options(${target} PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fPIC>")
    target_link_options(${target} PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fPIC>")
  endif()

  # Ask GCC to avoid `__builtin_memcpy` where we know what we are doing.
  # https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html
  target_compile_options(${target} PRIVATE "$<$<CXX_COMPILER_ID:GNU>:-fno-builtin-memcmp>")
  target_compile_options(${target} PRIVATE "$<$<CXX_COMPILER_ID:GNU>:-fno-builtin-memchr>")
  target_compile_options(${target} PRIVATE "$<$<CXX_COMPILER_ID:GNU>:-fno-builtin-memcpy>")
  target_compile_options(${target} PRIVATE "$<$<CXX_COMPILER_ID:GNU>:-fno-builtin-memset>")

  # Check for ${target_arch} and set it or use "march=native" if not defined
  if("${target_arch}" STREQUAL "")
    # MSVC does not have a direct equivalent to -march=native
    target_compile_options(
      ${target} PRIVATE
      "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<NOT:$<CXX_COMPILER_ID:AppleClang>>>:-march=native>"
      "$<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>")
  else()
    target_compile_options(
      ${target}
      PRIVATE
      "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<NOT:$<CXX_COMPILER_ID:AppleClang>>>:-march=${target_arch}>"
      "$<$<CXX_COMPILER_ID:MSVC>:/arch:${target_arch}>")
  endif()

  # Define SZ_DETECT_BIG_ENDIAN macro based on system byte order
  if(CMAKE_C_BYTE_ORDER STREQUAL "BIG_ENDIAN")
    set(SZ_DETECT_BIG_ENDIAN 1)
  else()
    set(SZ_DETECT_BIG_ENDIAN 0)
  endif()

  target_compile_definitions(
    ${target}
    PRIVATE
    "SZ_DETECT_BIG_ENDIAN=${SZ_DETECT_BIG_ENDIAN}"
  )

  # Sanitizer options for Debug mode
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(
      ${target}
      PRIVATE
      "$<$<CXX_COMPILER_ID:GNU,Clang>:-fsanitize=address;-fsanitize=leak>"
      "$<$<CXX_COMPILER_ID:MSVC>:/fsanitize=address>")

    target_link_options(
      ${target}
      PRIVATE
      "$<$<CXX_COMPILER_ID:GNU,Clang>:-fsanitize=address;-fsanitize=leak>"
      "$<$<CXX_COMPILER_ID:MSVC>:/fsanitize=address>")

    # Define SZ_DEBUG macro based on build configuration
    target_compile_definitions(
      ${target}
      PRIVATE
      "$<$<CONFIG:Debug>:SZ_DEBUG=1>"
      "$<$<NOT:$<CONFIG:Debug>>:SZ_DEBUG=0>"
    )
  endif()
endfunction()

function(define_launcher exec_name source cpp_standard target_arch)
  add_executable(${exec_name} ${source})
  set_compiler_flags(${exec_name} ${cpp_standard} "${target_arch}")
  add_test(NAME ${exec_name} COMMAND ${exec_name})
endfunction()

if(${STRINGZILLA_BUILD_BENCHMARK})
  define_launcher(stringzilla_bench_search scripts/bench_search.cpp 17 "${STRINGZILLA_TARGET_ARCH}")
  define_launcher(stringzilla_bench_similarity scripts/bench_similarity.cpp 17 "${STRINGZILLA_TARGET_ARCH}")
  define_launcher(stringzilla_bench_sort scripts/bench_sort.cpp 17 "${STRINGZILLA_TARGET_ARCH}")
  define_launcher(stringzilla_bench_token scripts/bench_token.cpp 17 "${STRINGZILLA_TARGET_ARCH}")
  define_launcher(stringzilla_bench_container scripts/bench_container.cpp 17 "${STRINGZILLA_TARGET_ARCH}")
endif()

if(${STRINGZILLA_BUILD_TEST})
  # Make sure that the compilation passes for different C++ standards
  # ! Keep in mind, MSVC only supports C++11 and newer.
  define_launcher(stringzilla_test_cpp11 scripts/test.cpp 11 "${STRINGZILLA_TARGET_ARCH}")
  define_launcher(stringzilla_test_cpp14 scripts/test.cpp 14 "${STRINGZILLA_TARGET_ARCH}")
  define_launcher(stringzilla_test_cpp17 scripts/test.cpp 17 "${STRINGZILLA_TARGET_ARCH}")
  define_launcher(stringzilla_test_cpp20 scripts/test.cpp 20 "${STRINGZILLA_TARGET_ARCH}")

  # Check system architecture to avoid complex cross-compilation workflows, but
  # compile multiple backends: disabling all SIMD, enabling only AVX2, only AVX-512, only Arm Neon.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    # x86 specific backends
    define_launcher(stringzilla_test_cpp20_x86_serial scripts/test.cpp 20 "ivybridge")
    define_launcher(stringzilla_test_cpp20_x86_avx2 scripts/test.cpp 20 "haswell")
    define_launcher(stringzilla_test_cpp20_x86_avx512 scripts/test.cpp 20 "sapphirerapids")
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64")
    # ARM specific backends
    define_launcher(stringzilla_test_cpp20_arm_serial scripts/test.cpp 20 "armv8-a")
    define_launcher(stringzilla_test_cpp20_arm_neon scripts/test.cpp 20 "armv8-a+simd")
  endif()
endif()

if(${STRINGZILLA_BUILD_SHARED})
  add_library(stringzilla_shared SHARED c/lib.c)
  set_compiler_flags(stringzilla_shared "" "${STRINGZILLA_TARGET_ARCH}")
  set_target_properties(stringzilla_shared PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
    PUBLIC_HEADER include/stringzilla/stringzilla.h)

  # Try compiling a version without linking the LibC
  add_library(stringzillite SHARED c/lib.c)
  set_compiler_flags(stringzillite "" "${STRINGZILLA_TARGET_ARCH}")
  target_compile_definitions(stringzillite PRIVATE "SZ_AVOID_LIBC=1")
  set_target_properties(stringzillite PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    POSITION_INDEPENDENT_CODE ON
    PUBLIC_HEADER include/stringzilla/stringzilla.h)
endif()