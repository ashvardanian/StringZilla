#!/bin/bash

# Assign arguments to variables
BUILD_TYPE=$1    # Debug or Release
COMPILER=$2      # GCC, LLVM, or MSVC

# Set common flags
COMMON_FLAGS="-DSTRINGZILLA_BUILD_TEST=1 -DSTRINGZILLA_BUILD_BENCHMARK=1 -DSTRINGZILLA_BUILD_SHARED=0"

# Compiler specific settings
case "$COMPILER" in
    "GCC")
        COMPILER_FLAGS="-DCMAKE_CXX_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12"
        ;;
    "LLVM")
        COMPILER_FLAGS="-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++"
        ;;
    "MSVC")
        COMPILER_FLAGS="" 
        ;;
    *)
        echo "Unknown compiler: $COMPILER"
        exit 1
        ;;
esac

# Set build type
case "$BUILD_TYPE" in
    "Debug")
        BUILD_DIR="build_debug"
        BUILD_FLAGS="-DCMAKE_BUILD_TYPE=Debug"
        ;;
    "Release")
        BUILD_DIR="build_release"
        BUILD_FLAGS="-DCMAKE_BUILD_TYPE=RelWithDebInfo"
        ;;
    *)
        echo "Unknown build type: $BUILD_TYPE"
        exit 1
        ;;
esac

# Execute commands
cmake $COMMON_FLAGS $COMPILER_FLAGS $BUILD_FLAGS -B $BUILD_DIR && cmake --build $BUILD_DIR --config $BUILD_TYPE
