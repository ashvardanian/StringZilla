# build_current.cmake
# Cross-platform script to build the current file's target
#
# Usage: cmake -DFILE=<file> -DBUILD_TYPE=<Debug|Release> -P build_current.cmake

cmake_minimum_required(VERSION 3.14)

if (NOT DEFINED FILE)
    message(FATAL_ERROR "FILE not specified. Usage: cmake -DFILE=path/to/file.cpp -P build_current.cmake")
endif ()

if (NOT DEFINED BUILD_TYPE)
    set(BUILD_TYPE "Debug")
    message(STATUS "BUILD_TYPE not specified, defaulting to Debug")
endif ()

# Extract basename without extension
get_filename_component(BASENAME "${FILE}" NAME_WE)

# Map filename patterns to CMake targets
if (BASENAME MATCHES "^bench_(.+)$")
    # Benchmark files: bench_find.cpp -> stringzilla_bench_find_cpp20
    set(TARGET "stringzilla_${BASENAME}_cpp20")
else ()
    message(FATAL_ERROR "Unknown file pattern: ${BASENAME}\nSupported patterns:\n  - bench_*.cpp\n  - test_stringzilla.cpp\n  - test_stringzillas.cpp")
endif ()

# Determine build directory
string(TOLOWER "${BUILD_TYPE}" build_type_lower)
set(BUILD_DIR "${CMAKE_CURRENT_LIST_DIR}/../build_${build_type_lower}")

# Verify build directory exists
if (NOT EXISTS "${BUILD_DIR}")
    message(FATAL_ERROR "Build directory not found: ${BUILD_DIR}\nRun: cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -B ${BUILD_DIR}")
endif ()

message(STATUS "Building target: ${TARGET}")
message(STATUS "Build directory: ${BUILD_DIR}")
message(STATUS "Build type: ${BUILD_TYPE}")

# Execute the build
execute_process(
    COMMAND cmake --build ${BUILD_DIR} --config ${BUILD_TYPE} --target ${TARGET}
    RESULT_VARIABLE result
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
)

if (result EQUAL 0)
    message(STATUS "Build succeeded: ${TARGET}")
else ()
    message(FATAL_ERROR "Build failed with exit code: ${result}")
endif ()
