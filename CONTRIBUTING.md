# Contributing to StringZilla

Thank you for coming here! It's always nice to have third-party contributors 🤗
Depending on the type of contribution, you may need to follow different steps.

---

## Project Structure

The project is split into the following parts:

- `include/stringzilla/stringzilla.h` - single-header C implementation.
- `include/stringzilla/stringzilla.hpp` - single-header C++ wrapper.
- `python/*` - Python bindings.
- `javascript/*` - JavaScript bindings.
- `scripts/*` - Scripts for benchmarking and testing.

For minimal test coverage, check the following scripts:

- `test.cpp` - tests C++ API (not underlying C) against STL.
- `test.py` - tests Python API against native strings.
- `test.js`.

At the C++ level all benchmarks also validate the results against the STL baseline, serving as tests on real-world data.
They have the broadest coverage of the library, and are the most important to keep up-to-date:

- `bench_token.cpp` - token-level ops, like hashing, ordering, equality checks.
- `bench_search.cpp` - bidirectional substring search, both exact and fuzzy.
- `bench_similarity.cpp` - benchmark all edit distance backends.
- `bench_sort.cpp` - sorting, partitioning, merging.
- `bench_container.cpp` - STL containers with different string keys.

The role of Python benchmarks is less to provide absolute number, but to compare against popular tools in the Python ecosystem.

- `bench_search.(py|ipynb)` - compares against native Python `str`.
- `bench_sort.(py|ipynb)` - compares against `pandas`.
- `bench_similarity.(ipynb)` - compares against `jellyfish`, `editdistance`, etc.

## Benchmarking Datasets

It's not always easy to find good datasets for benchmarking strings workloads.
I use several ASCII and UTF8 international datasets.
You can download them using the following commands:

```sh
# English Leipzig Corpora Collection
# 124 MB, 1'000'000 lines of ASCII, 8'388'608 tokens of mean length 5
wget --no-clobber -O leipzig1M.txt https://introcs.cs.princeton.edu/python/42sort/leipzig1m.txt 

# Hutter Prize "enwik9" dataset for compression
# 1 GB (0.3 GB compressed), 13'147'025 lines of ASCII, 67'108'864 tokens of mean length 6
wget --no-clobber -O enwik9.zip http://mattmahoney.net/dc/enwik9.zip
unzip enwik9.zip && rm enwik9.zip && mv enwik9 enwik9.txt

# XL Sum dataset for multilingual extractive summarization
# 4.7 GB (1.7 GB compressed), 1'004'598 lines of UTF8, 268'435'456 tokens of mean length 8
wget --no-clobber -O xlsum.csv.gz https://github.com/ashvardanian/xl-sum/releases/download/v1.0.0/xlsum.csv.gz
gzip -d xlsum.csv.gz
```

## IDE Integrations

The project was originally developed in VS Code, and contains a set of configuration files for that IDE under `.vscode/`.

- `tasks.json` - build tasks for CMake.
- `launch.json` - debugger launchers for CMake.
- `extensions.json` - recommended extensions for VS Code, including:
    - `ms-vscode.cpptools-themes` - C++ language support.
    - `ms-vscode.cmake-tools`, `cheshirekow.cmake-format` - CMake integration.
    - `ms-python.python`, `ms-python.black-formatter` - Python language support.
    - `yzhang.markdown-all-in-one` - formatting Markdown.
    - `aaron-bond.better-comments` - color-coded comments.

## Code Styling

The project uses `.clang-format` to enforce a consistent code style.
Modern IDEs, like VS Code, can be configured to automatically format the code on save.

- East const over const West. Write `char const*` instead of `const char*`.
- For color-coded comments start the line with `!` for warnings or `?` for questions.
- Sort the includes: standard libraries, third-party libraries, and only then internal project headers.

For C++ code:

- Explicitly use `std::` or `sz::` namespaces over global `memcpy`, `uint64_t`, etc.
- Explicitly mark `noexcept` or `noexcept(false)` for all library interfaces.
- Document all possible exceptions of an interface using `@throw` in Doxygen.
- Avoid C-style variadic arguments in favor of templates.
- Avoid C-style casts in favor of `static_cast`, `reinterpret_cast`, and `const_cast`, except for places where a C function is called.
- Use lower-case names for everything, except settings/conditions macros. Function-like macros, that take arguments, should be lowercase as well.
- In templates prefer `typename` over `class`.
- Prepend "private" symbols with `_` underscore.

For Python code:

- Use lower-case names for functions and variables.

## Contributing in C++ and C

The primary C implementation and the C++ wrapper are built with CMake.
Assuming the extensive use of new SIMD intrinsics and recent C++ language features, using a recent compiler is recommended.
We prefer GCC 12, which is available from default Ubuntu repositories with Ubuntu 22.04 LTS onwards.
If this is your first experience with CMake, use the following commands to get started:

```bash
sudo apt-get update && sudo apt-get install cmake build-essential libjemalloc-dev g++-12 gcc-12 # Ubuntu
brew install libomp llvm # MacOS
```

### Testing

Using modern syntax, this is how you build and run the test suite:

```bash
cmake -DSTRINGZILLA_BUILD_TEST=1 -DCMAKE_BUILD_TYPE=Debug -B build_debug
cmake --build build_debug --config Debug          # Which will produce the following targets:
build_debug/stringzilla_test_cpp20                # Unit test for the entire library compiled for current hardware
build_debug/stringzilla_test_cpp20_x86_serial     # x86 variant compiled for IvyBridge - last arch. before AVX2
build_debug/stringzilla_test_cpp20_arm_serial     # Arm variant compiled without Neon
```

To use CppCheck for static analysis make sure to export the compilation commands.
Overall, CppCheck and Clang-Tidy are extremely noisy and not suitable for CI, but may be useful for local development.

```bash
sudo apt install cppcheck clang-tidy-11

cmake -B build_artifacts \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
  -DSTRINGZILLA_BUILD_BENCHMARK=1 \
  -DSTRINGZILLA_BUILD_TEST=1

cppcheck --project=build_artifacts/compile_commands.json --enable=all

clang-tidy-11 -p build_artifacts
```

I'd recommend putting the following breakpoints:

- `__asan::ReportGenericError` - to detect illegal memory accesses.
- `__GI_exit` - to stop at exit points - the end of running any executable.
- `__builtin_unreachable` - to catch all the places where the code is expected to be unreachable.

### Benchmarking

For benchmarks, you can use the following commands:

```bash
cmake -DSTRINGZILLA_BUILD_BENCHMARK=1 -B build_release
cmake --build build_release --config Release      # Which will produce the following targets:
build_release/stringzilla_bench_search <path>     # for substring search
build_release/stringzilla_bench_token <path>      # for hashing, equality comparisons, etc.
build_release/stringzilla_bench_similarity <path> # for edit distances and alignment scores
build_release/stringzilla_bench_sort <path>       # for sorting arrays of strings
build_release/stringzilla_bench_container <path>  # for STL containers with string keys
```

### Benchmarking Hardware-Specific Optimizations

Running on modern hardware, you may want to compile the code for older generations to compare the relative performance.
The assumption would be that newer ISA extensions would provide better performance.
On x86_64, you can use the following commands to compile for Sandy Bridge, Haswell, and Sapphire Rapids:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
    -DSTRINGZILLA_TARGET_ARCH="ivybridge" -B build_release/ivybridge && \
    cmake --build build_release/ivybridge --config Release
cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
    -DSTRINGZILLA_TARGET_ARCH="haswell" -B build_release/haswell && \
    cmake --build build_release/haswell --config Release
cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
    -DSTRINGZILLA_TARGET_ARCH="sapphirerapids" -B build_release/sapphirerapids && \
    cmake --build build_release/sapphirerapids --config Release
```

### Benchmarking Compiler-Specific Optimizations

Alternatively, you may want to compare the performance of the code compiled with different compilers.
On x86_64, you may want to compare GCC, Clang, and ICX.

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 -DSTRINGZILLA_BUILD_SHARED=1 \
    -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_C_COMPILER=gcc-12 \
    -B build_release/gcc && cmake --build build_release/gcc --config Release
cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 -DSTRINGZILLA_BUILD_SHARED=1 \
    -DCMAKE_CXX_COMPILER=clang++-14 -DCMAKE_C_COMPILER=clang-14 \
    -B build_release/clang && cmake --build build_release/clang --config Release
```

### Profiling

To simplify tracing and profiling, build with symbols using the `RelWithDebInfo` configuration.
Here is an example for profiling one target - `stringzilla_bench_token`.

```bash
cmake -DSTRINGZILLA_BUILD_BENCHMARK=1 \
    -DSTRINGZILLA_BUILD_TEST=1 \
    -DSTRINGZILLA_BUILD_SHARED=1 \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -B build_profile
cmake --build build_profile --config Release --target stringzilla_bench_token

# Check that the debugging symbols are there with your favorite tool
readelf --sections build_profile/stringzilla_bench_token | grep debug
objdump -h build_profile/stringzilla_bench_token | grep debug

# Profile
sudo perf record -g build_profile/stringzilla_bench_token ./leipzig1M.txt
sudo perf report
```

### Testing in Docker

It might be a good idea to check the compatibility against the most popular Linux distributions.
Docker is the goto-choice for that.

#### Alpine

Alpine is one of the most popular Linux distributions for containers, due to it's size.
The base image is only ~3 MB, and it's based on musl libc, which is different from glibc.

```bash
sudo docker run -it --rm -v "$(pwd)":/workspace/StringZilla alpine:latest /bin/ash
cd /workspace/StringZilla
apk add --update make cmake g++ gcc
cmake -DSTRINGZILLA_BUILD_TEST=1 -DCMAKE_BUILD_TYPE=Debug -B build_debug
cmake --build build_debug --config Debug
build_debug/stringzilla_test_cpp20
```

#### Intel Clear Linux

Clear Linux is a distribution optimized for Intel hardware, and is known for its performance.
It has rolling releases, and is based on glibc.
It might be a good choice for compiling with Intel oneAPI compilers.

```bash
sudo docker run -it --rm -v "$(pwd)":/workspace/StringZilla clearlinux:latest /bin/bash
cd /workspace/StringZilla
swupd update
swupd bundle-add c-basic dev-utils
cmake -DSTRINGZILLA_BUILD_TEST=1 -DCMAKE_BUILD_TYPE=Debug -B build_debug
cmake --build build_debug --config Debug
build_debug/stringzilla_test_cpp20
```

For benchmarks:

```bash
cmake -DSTRINGZILLA_BUILD_TEST=1 -DSTRINGZILLA_BUILD_BENCHMARK=1 -B build_release
cmake --build build_release --config Release
```

#### Amazon Linux

For CentOS-based __Amazon Linux 2023__:

```bash
sudo docker run -it --rm -v "$(pwd)":/workspace/StringZilla amazonlinux:2023 bash
cd /workspace/StringZilla
yum install -y make cmake3 gcc g++
cmake3 -DSTRINGZILLA_BUILD_TEST=1 -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DSTRINGZILLA_TARGET_ARCH="ivybridge" \
    -B build_debug
cmake3 --build build_debug --config Debug --target stringzilla_test_cpp11
build_debug/stringzilla_test_cpp11
```

The CentOS-based __Amazon Linux 2__ is still used in older AWS Lambda functions.
Sadly, the newest GCC version it supports is 10, and it can't handle AVX-512 instructions.

```bash
sudo docker run -it --rm -v "$(pwd)":/workspace/StringZilla amazonlinux:2 bash
cd /workspace/StringZilla
yum install -y make cmake3 gcc10 gcc10-c++
cmake3 -DSTRINGZILLA_BUILD_TEST=1 -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc -DSTRINGZILLA_TARGET_ARCH="ivybridge" \
    -B build_debug
cmake3 --build build_debug --config Debug --target stringzilla_test_cpp11
build_debug/stringzilla_test_cpp11
```

> [!CAUTION]
> 
> Even with GCC 10 the tests compilation will fail, as the STL implementation of the `insert` function doesn't conform to standard.
> The `s.insert(s.begin() + 1, {'a', 'b', 'c'}) == (s.begin() + 1)` expression is illformed, as the `std::string::insert` return `void`.

---

Don't forget to clean up Docker afterwards.

```bash
docker system prune -a --volumes
```

### Cross Compilation

Unlike GCC, LLVM handles cross compilation very easily.
You just need to pass the right `TARGET_ARCH` and `BUILD_ARCH` to CMake.
The [list includes](https://packages.ubuntu.com/search?keywords=crossbuild-essential&searchon=names):

- `crossbuild-essential-amd64` for 64-bit x86
- `crossbuild-essential-arm64` for 64-bit Arm
- `crossbuild-essential-armhf` for 32-bit ARM hard-float
- `crossbuild-essential-armel` for 32-bit ARM soft-float (emulates `float`)
- `crossbuild-essential-riscv64` for RISC-V
- `crossbuild-essential-powerpc` for PowerPC
- `crossbuild-essential-s390x` for IBM Z
- `crossbuild-essential-mips` for MIPS
- `crossbuild-essential-ppc64el` for PowerPC 64-bit little-endian

Here is an example for cross-compiling for Arm64 on an x86_64 machine:

```sh
sudo apt-get update
sudo apt-get install -y clang lld make crossbuild-essential-arm64 crossbuild-essential-armhf
export CC="clang"
export CXX="clang++"
export AR="llvm-ar"
export NM="llvm-nm"
export RANLIB="llvm-ranlib"
export TARGET_ARCH="aarch64-linux-gnu" # Or "x86_64-linux-gnu"
export BUILD_ARCH="arm64" # Or "amd64"

cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER_TARGET=${TARGET_ARCH} \
    -DCMAKE_CXX_COMPILER_TARGET=${TARGET_ARCH} \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=${BUILD_ARCH} \
    -B build_artifacts
cmake --build build_artifacts --config Release
```

## Contributing in Python

Python bindings are implemented using pure CPython, so you wouldn't need to install SWIG, PyBind11, or any other third-party library.

```bash
pip install -e . # To build locally from source
```

### Testing

For testing we use PyTest, which may not be installed on your system.

```bash
pip install pytest              # To install PyTest
pytest scripts/test.py -s -x    # Runs tests printing logs and stops on the first failure
```

On a related note, StringZilla for Python seems to cover more OS and hardware combinations, than NumPy.
That's why NumPy isn't a required dependency.
Still, many tests may use NumPy, so consider installing it on mainstream platforms.

```bash
pip install numpy
```

Before you ship, please make sure the `cibuilwheel` packaging works and tests pass on other platforms.
Don't forget to use the right [CLI arguments][cibuildwheel-cli] to avoid overloading your Docker runtime.

```bash
cibuildwheel
cibuildwheel --platform linux                   # works on any OS and builds all Linux backends
cibuildwheel --platform linux --archs x86_64    # 64-bit x86, the most common on desktop and servers
cibuildwheel --platform linux --archs aarch64   # 64-bit Arm for mobile devices, Apple M-series, and AWS Graviton
cibuildwheel --platform linux --archs i686      # 32-bit Linux
cibuildwheel --platform linux --archs s390x     # emulating big-endian IBM Z
cibuildwheel --platform macos                   # works only on MacOS
cibuildwheel --platform windows                 # works only on Windows
```

You may need root previligies for multi-architecture builds:

```bash
sudo $(which cibuildwheel) --platform linux
```

On Windows and MacOS, to avoid frequent path resolution issues, you may want to use:

```bash
python -m cibuildwheel --platform windows
```

[cibuildwheel-cli]: https://cibuildwheel.readthedocs.io/en/stable/options/#command-line

### Benchmarking

For high-performance low-latency benchmarking, stick to C/C++ native benchmarks, as the CPython is likely to cause bottlenecks.
For benchmarking, the following scripts are provided.

```sh
python scripts/bench_search.py --haystack_path "your file" --needle "your pattern" # real data
python scripts/bench_search.py --haystack_pattern "abcd" --haystack_length 1e9 --needle "abce" # synthetic data
python scripts/similarity_bench.py --text_path "your file" # edit ditance computations
```

Alternatively, you can explore the Jupyter notebooks in `scripts/` directory.

## Contributing in JavaScript

```bash
npm ci && npm test
```

## Contributing in Swift

```bash
swift build && swift test
```

Running Swift on Linux requires a couple of extra steps, as the Swift compiler is not available in the default repositories.
Please get the most recent Swift tarball from the [official website](https://www.swift.org/install/).
At the time of writing, for 64-bit Arm CPU running Ubuntu 22.04, the following commands would work:

```bash
wget https://download.swift.org/swift-5.9.2-release/ubuntu2204-aarch64/swift-5.9.2-RELEASE/swift-5.9.2-RELEASE-ubuntu22.04-aarch64.tar.gz
tar xzf swift-5.9.2-RELEASE-ubuntu22.04-aarch64.tar.gz
sudo mv swift-5.9.2-RELEASE-ubuntu22.04-aarch64 /usr/share/swift
echo "export PATH=/usr/share/swift/usr/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
```

You can check the available images on [`swift.org/download` page](https://www.swift.org/download/#releases).
For x86 CPUs, the following commands would work:

```bash
wget https://download.swift.org/swift-5.9.2-release/ubuntu2204/swift-5.9.2-RELEASE/swift-5.9.2-RELEASE-ubuntu22.04.tar.gz
tar xzf swift-5.9.2-RELEASE-ubuntu22.04.tar.gz
sudo mv swift-5.9.2-RELEASE-ubuntu22.04 /usr/share/swift
echo "export PATH=/usr/share/swift/usr/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
```

Alternatively, on Linux, the official Swift Docker image can be used for builds and tests:

```bash
sudo docker run --rm -v "$PWD:/workspace" -w /workspace swift:5.9 /bin/bash -cl "swift build -c release --static-swift-stdlib && swift test -c release --enable-test-discovery"
```

## Contributing in Rust

```bash
cargo test
```

If you are updating the package contents, you can validate the list of included files using the following command:

```bash
cargo package --list --allow-dirty
```

If you want to run benchmarks against third-party implementations, check out the [`ashvardanian/memchr_vs_stringzilla`](https://github.com/ashvardanian/memchr_vs_stringzilla/) repository.

## General Performance Observations

### Unaligned Loads

One common surface of attack for performance optimizations is minimizing unaligned loads.
Such solutions are beautiful from the algorithmic perspective, but often lead to worse performance.
It's often cheaper to issue two interleaving wide-register loads, than try minimizing those loads at the cost of juggling registers.

### Register Pressure

Byte-level comparisons are simpler and often faster, than n-gram comparisons with subsequent interleaving.
In the following example we search for 4-byte needles in a haystack, loading at different offsets, and comparing then as arrays of 32-bit integers.

```c
h0_vec.zmm = _mm512_loadu_epi8(h);
h1_vec.zmm = _mm512_loadu_epi8(h + 1);
h2_vec.zmm = _mm512_loadu_epi8(h + 2);
h3_vec.zmm = _mm512_loadu_epi8(h + 3);
matches0 = _mm512_cmpeq_epi32_mask(h0_vec.zmm, n_vec.zmm);
matches1 = _mm512_cmpeq_epi32_mask(h1_vec.zmm, n_vec.zmm);
matches2 = _mm512_cmpeq_epi32_mask(h2_vec.zmm, n_vec.zmm);
matches3 = _mm512_cmpeq_epi32_mask(h3_vec.zmm, n_vec.zmm);
if (matches0 | matches1 | matches2 | matches3)
    return h + sz_u64_ctz(_pdep_u64(matches0, 0x1111111111111111) | //
                          _pdep_u64(matches1, 0x2222222222222222) | //
                          _pdep_u64(matches2, 0x4444444444444444) | //
                          _pdep_u64(matches3, 0x8888888888888888));
```

A simpler solution would be to compare byte-by-byte, but in that case we would need to populate multiple registers, broadcasting different letters of the needle into them.
That may not be noticeable on a micro-benchmark, but it would be noticeable on real-world workloads, where the CPU will speculatively interleave those search operations with something else happening in that context.

## Working on Alternative Hardware Backends

## Working on Faster Edit Distances

## Working on Random String Generators

## Working on Sequence Processing and Sorting


