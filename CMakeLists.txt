cmake_minimum_required(VERSION 3.1)
project(
  stringzilla
  VERSION 3.8.4
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

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(STATUS "Pointer size: 64-bit")
else()
  message(STATUS "Pointer size: 32-bit")
endif()

# Set a default build type to "Release" if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE
    Release
    CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  SET(SZ_PLATFORM_X86 TRUE)
  message(STATUS "Platform: x86")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64")
  SET(SZ_PLATFORM_ARM TRUE)
  message(STATUS "Platform: ARM")
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

if (MSVC)
  # Remove /RTC* from MSVC debug flags by default (it will be added back in the set_compiler_flags function)
  # Beacuse /RTC* cannot be used without the crt so it needs to be disabled for that specifc target
  string(REGEX REPLACE "/RTC[^ ]*" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
  string(REGEX REPLACE "/RTC[^ ]*" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
endif()

# Function to set compiler-specific flags
function(set_compiler_flags target cpp_standard target_arch)
  get_target_property(target_type ${target} TYPE)

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
    set_target_properties(${target} PROPERTIES CXX_STANDARD ${cpp_standard})
  endif()

  # Use the /Zc:__cplusplus flag to correctly define the __cplusplus macro in MSVC
  target_compile_options(${target} PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/Zc:__cplusplus>")

  # Maximum warnings level & warnings as error.
  # MVC uses numeric values:
  # > 4068 for "unknown pragmas".
  # > 4146 for "unary minus operator applied to unsigned type, result still unsigned".
  # We also specify /utf-8 to properly UTF-8 symbols in tests.
  target_compile_options(
    ${target}
    PRIVATE
    "$<$<CXX_COMPILER_ID:MSVC>:/Bt;/wd4068;/wd4146;/utf-8;/WX>"
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

  if(NOT target_type STREQUAL "SHARED_LIBRARY")
    if(MSVC)
      target_compile_options(${target} PRIVATE "$<$<CONFIG:Debug>:/RTC1>")
    endif()
  endif()

  # If available, enable Position Independent Code
  get_target_property(target_pic ${target} POSITION_INDEPENDENT_CODE)
  if(target_pic)
    target_compile_options(${target} PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fPIC>")
    target_link_options(${target} PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fPIC>")
    target_compile_definitions(${target} PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:SZ_PIC>")
  endif()

  # Avoid builtin functions where we know what we are doing.
  target_compile_options(${target} PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fno-builtin-memcmp>")
  target_compile_options(${target} PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fno-builtin-memchr>")
  target_compile_options(${target} PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fno-builtin-memcpy>")
  target_compile_options(${target} PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-fno-builtin-memset>")
  target_compile_options(${target} PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/Oi->")

  # Check for ${target_arch} and set it or use the current system if not defined
  if("${target_arch}" STREQUAL "")
    # Only use the current system if we are not cross compiling
    if((NOT CMAKE_CROSSCOMPILING) OR (CMAKE_SYSTEM_PROCESSOR MATCHES CMAKE_HOST_SYSTEM_PROCESSOR))
      if (NOT MSVC)
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-march=native" supports_march_native)
        if (supports_march_native)
          target_compile_options(${target} PRIVATE "-march=native")
        endif()
      else()
        # MSVC does not have a direct equivalent to -march=native
        target_compile_options(${target} PRIVATE "/arch:AVX2")
      endif()
    endif()
  else()
    target_compile_options(
      ${target}
      PRIVATE
      "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-march=${target_arch}>"
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
    if(NOT target_type STREQUAL "SHARED_LIBRARY")
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
    endif()

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
  if(SZ_PLATFORM_X86)
    # x86 specific backends
    if (MSVC)
      define_launcher(stringzilla_test_cpp20_x86_serial scripts/test.cpp 20 "AVX")
      define_launcher(stringzilla_test_cpp20_x86_avx2 scripts/test.cpp 20 "AVX2")
      define_launcher(stringzilla_test_cpp20_x86_avx512 scripts/test.cpp 20 "AVX512")
    else()
      define_launcher(stringzilla_test_cpp20_x86_serial scripts/test.cpp 20 "ivybridge")
      define_launcher(stringzilla_test_cpp20_x86_avx2 scripts/test.cpp 20 "haswell")
      define_launcher(stringzilla_test_cpp20_x86_avx512 scripts/test.cpp 20 "sapphirerapids")
    endif()
  elseif(SZ_PLATFORM_ARM)
    # ARM specific backends
    define_launcher(stringzilla_test_cpp20_arm_serial scripts/test.cpp 20 "armv8-a")
    define_launcher(stringzilla_test_cpp20_arm_neon scripts/test.cpp 20 "armv8-a+simd")
  endif()
endif()

if(${STRINGZILLA_BUILD_SHARED})

  function(define_shared target)
    add_library(${target} SHARED c/lib.c)

    set_target_properties(${target} PROPERTIES
      VERSION ${PROJECT_VERSION}
      SOVERSION 1
      POSITION_INDEPENDENT_CODE ON
      PUBLIC_HEADER include/stringzilla/stringzilla.h)

    if (SZ_PLATFORM_X86)
      if (MSVC)
        set_compiler_flags(${target} "" "SSE2")
      else()
        set_compiler_flags(${target} "" "ivybridge")
      endif()

      target_compile_definitions(${target} PRIVATE
        "SZ_USE_X86_AVX512=1"
        "SZ_USE_X86_AVX2=1"
        "SZ_USE_ARM_NEON=0"
        "SZ_USE_ARM_SVE=0")
    elseif(SZ_PLATFORM_ARM)
      set_compiler_flags(${target} "" "armv8-a")

      target_compile_definitions(${target} PRIVATE
        "SZ_USE_X86_AVX512=0"
        "SZ_USE_X86_AVX2=0"
        "SZ_USE_ARM_NEON=1"
        "SZ_USE_ARM_SVE=1")
    endif()
  endfunction()

  define_shared(stringzilla_shared)
  target_compile_definitions(stringzilla_shared PRIVATE "SZ_AVOID_LIBC=0")
  target_compile_definitions(stringzilla_shared PRIVATE "SZ_OVERRIDE_LIBC=1")

  # Try compiling a version without linking the LibC
  define_shared(stringzillite)
  target_compile_definitions(stringzillite PRIVATE "SZ_AVOID_LIBC=1")
  target_compile_definitions(stringzillite PRIVATE "SZ_OVERRIDE_LIBC=1")

  # Avoid built-ins on MSVC and other compilers, as that will cause compileration errors
  target_compile_options(stringzillite PRIVATE
    "$<$<CXX_COMPILER_ID:GNU,Clang>:-fno-builtin;-nostdlib>"
    "$<$<CXX_COMPILER_ID:MSVC>:/Oi-;/GS->")
  target_link_options(stringzillite PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang>:-nostdlib>")
  target_link_options(stringzillite PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/NODEFAULTLIB>")
endif()
