#!/bin/bash
# This Bash script compiles the CMake-based project with different compilers for different verrsions of C++
# This is what should happen if only GCC 12 is installed and we are running on Sapphire Rapids.
#
#       cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
#           -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_C_COMPILER=gcc-12 \
#           -DSTRINGZILLA_TARGET_ARCH="sandybridge" -B build_release/gcc-12-sandybridge && \
#           cmake --build build_release/gcc-12-sandybridge --config Release
#       cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
#           -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_C_COMPILER=gcc-12 \
#           -DSTRINGZILLA_TARGET_ARCH="haswell" -B build_release/gcc-12-haswell && \
#           cmake --build build_release/gcc-12-haswell --config Release
#       cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
#           -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_C_COMPILER=gcc-12 \
#           -DSTRINGZILLA_TARGET_ARCH="sapphirerapids" -B build_release/gcc-12-sapphirerapids && \
#           cmake --build build_release/gcc-12-sapphirerapids --config Release

# Array of target architectures
declare -a architectures=("sandybridge" "haswell" "sapphirerapids")

# Function to get installed versions of a compiler
get_versions() {
    local compiler_prefix=$1
    local versions=()

    echo "Checking for compilers in /usr/bin with prefix: $compiler_prefix"

    # Check if the directory /usr/bin exists and is a directory
    if [ -d "/usr/bin" ]; then
        for version in /usr/bin/${compiler_prefix}-*; do
            echo "Checking: $version"
            if [[ -x "$version" ]]; then
                local ver=${version##*-}
                echo "Found compiler version: $ver"
                versions+=("$ver")
            fi
        done
    else
        echo "/usr/bin does not exist or is not a directory"
    fi
    
    echo ${versions[@]}
}

# Get installed versions of GCC and Clang
gcc_versions=$(get_versions gcc)
clang_versions=$(get_versions clang)

# Compile for each combination of compiler and architecture
for arch in "${ARCHS[@]}"; do
    for gcc_version in $gcc_versions; do
        cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
              -DCMAKE_CXX_COMPILER=g++-$gcc_version -DCMAKE_C_COMPILER=gcc-$gcc_version \
              -DSTRINGZILLA_TARGET_ARCH="$arch" -B "build_release/gcc-$gcc_version-$arch" && \
              cmake --build "build_release/gcc-$gcc_version-$arch" --config Release
    done

    for clang_version in $clang_versions; do
        cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
              -DCMAKE_CXX_COMPILER=clang++-$clang_version -DCMAKE_C_COMPILER=clang-$clang_version \
              -DSTRINGZILLA_TARGET_ARCH="$arch" -B "build_release/clang-$clang_version-$arch" && \
              cmake --build "build_release/clang-$clang_version-$arch" --config Release
    done
done

