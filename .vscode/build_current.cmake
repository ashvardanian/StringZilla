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

# Every CPU benchmark and test source compiles into one executable per directory
get_filename_component(parent_path_ "${FILE}" DIRECTORY)
get_filename_component(directory_ "${parent_path_}" NAME)
if (directory_ STREQUAL "bench")
    set(target_ "stringzilla_bench")
elseif (directory_ STREQUAL "test")
    set(target_ "stringzilla_test")
else ()
    message(FATAL_ERROR "Unknown file location: ${FILE}\nSupported locations:\n  - bench/*.cpp\n  - test/*.cpp")
endif ()

# Determine build directory
string(TOLOWER "${BUILD_TYPE}" build_type_lowercase_)
set(build_directory_ "${CMAKE_CURRENT_LIST_DIR}/../build_${build_type_lowercase_}")

# Verify build directory exists
if (NOT EXISTS "${build_directory_}")
    message(
        FATAL_ERROR
            "Build directory not found: ${build_directory_}\nRun: cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -B ${build_directory_}"
    )
endif ()

message(STATUS "Building target: ${target_}")
message(STATUS "Build directory: ${build_directory_}")
message(STATUS "Build type: ${BUILD_TYPE}")

# Execute the build
execute_process(
    COMMAND cmake --build ${build_directory_} --config ${BUILD_TYPE} --target ${target_}
    RESULT_VARIABLE result_
    WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/..
)

if (result_ EQUAL 0)
    message(STATUS "Build succeeded: ${target_}")
else ()
    message(FATAL_ERROR "Build failed with exit code: ${result_}")
endif ()
