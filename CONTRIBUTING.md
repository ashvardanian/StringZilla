# Contributing to StringZilla

Thank you for coming here! It's always nice to have third-party contributors 🤗
Depending on the type of contribution, you may need to follow different steps.

---

## Project Structure

The project is split into the following parts:

- `include/stringzilla/stringzilla.h` - single-header C implementation.
- `include/stringzilla/stringzilla.hpp` - single-header C++ wrapper.
- `python/**` - Python bindings.
- `javascript/**` - JavaScript bindings.
- `scripts/**` - Scripts for benchmarking and testing.

The scripts name convention is as follows: `<workload>_<nature>.<language>`.
An example would be, `search_bench.cpp` or `similarity_fuzz.py`.
The nature of the script can be:

- `bench` - bounded in time benchmarking, generally on user-provided data.
- `fuzz` - unbounded in time fuzzing, generally on randomly generated data.
- `test` - unit tests.

## Contributing in C++ and C

The primary C implementation and the C++ wrapper are built with CMake.
Assuming the extensive use of new SIMD intrinsics and recent C++ language features, using a recent compiler is recommended.
We prefer GCC 12, which is available from default Ubuntu repositories with Ubuntu 22.04 LTS onwards.
If this is your first experience with CMake, use the following commands to get started:

```bash
sudo apt-get update && sudo apt-get install cmake build-essential libjemalloc-dev g++-12 gcc-12 # Ubuntu
brew install libomp llvm # MacOS
```

Using modern syntax, this is how you build and run the test suite:

```bash
cmake -DSTRINGZILLA_BUILD_TEST=1 -B ./build_debug
cmake --build ./build_debug --config Debug  # Which will produce the following targets:
./build_debug/search_test                   # Unit test for substring search
```

For benchmarks, you can use the following commands:

```bash
cmake -DSTRINGZILLA_BUILD_BENCHMARK=1 -B ./build_release
cmake --build ./build_release --config Release  # Which will produce the following targets:
./build_release/search_bench                    # Benchmark for substring search
./build_release/sort_bench                      # Benchmark for sorting arrays of strings
```

Running on modern hardware, you may want to compile the code for older generations to compare the relative performance.
The assumption would be that newer ISA extensions would provide better performance.
On x86_64, you can use the following commands to compile for Sandy Bridge, Haswell, and Sapphire Rapids:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
    -DCMAKE_CXX_FLAGS="-march=sandybridge" -DCMAKE_C_FLAGS="-march=sandybridge" \
    -B ./build_release/sandybridge && cmake --build build_release/sandybridge --config Release
cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
    -DCMAKE_CXX_FLAGS="-march=haswell" -DCMAKE_C_FLAGS="-march=haswell" \
    -B ./build_release/haswell && cmake --build build_release/haswell --config Release
cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
    -DCMAKE_CXX_FLAGS="-march=sapphirerapids" -DCMAKE_C_FLAGS="-march=sapphirerapids" \
    -B ./build_release/sapphirerapids && cmake --build build_release/sapphirerapids --config Release

./build_release/sandybridge/stringzilla_search_bench
./build_release/haswell/stringzilla_search_bench
./build_release/sapphirerapids/stringzilla_search_bench
```

Alternatively, you may want to compare the performance of the code compiled with different compilers.
On x86_64, you may want to compare GCC, Clang, and ICX.

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
    -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_C_COMPILER=gcc-12 \
    -B ./build_release/gcc && cmake --build build_release/gcc --config Release
cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
    -DCMAKE_CXX_COMPILER=clang++-14 -DCMAKE_C_COMPILER=clang-14 \
    -B ./build_release/clang && cmake --build build_release/clang --config Release
```

## Contibuting in Python

Python bindings are implemented using pure CPython, so you wouldn't need to install SWIG, PyBind11, or any other third-party library.

```bash
pip install -e . # To build locally from source
```

For testing we use PyTest, which may not be installed on your system.

```bash
pip install pytest      # To install PyTest
pytest scripts/ -s -x   # To run the test suite
```

For fuzzing we love the ability to call the native C implementation from Python bypassing the binding layer.
For that we use Cppyy, derived from Cling, a Clang-based C++ interpreter.

```bash
pip install cppyy                       # To install Cppyy
python scripts/similarity_fuzz.py       # To run the fuzzing script
```

For benchmarking, the following scripts are provided.

```sh
python scripts/search_bench.py --haystack_path "your file" --needle "your pattern" # real data
python scripts/search_bench.py --haystack_pattern "abcd" --haystack_length 1e9 --needle "abce" # synthetic data
python scripts/similarity_bench.py --text_path "your file" # edit ditance computations
```

Before you ship, please make sure the packaging works.

```bash
cibuildwheel --platform linux
```

## Roadmap

The project is in its early stages of development.
So outside of basic bug-fixes, several features are still missing, and can be implemented by you.
Future development plans include:

- [x] [Replace PyBind11 with CPython](https://github.com/ashvardanian/StringZilla/issues/35), [blog](https://ashvardanian.com/posts/pybind11-cpython-tutorial/.
- [x] [Bindings for JavaScript](https://github.com/ashvardanian/StringZilla/issues/25).
- [x] [Reverse-order operations](https://github.com/ashvardanian/StringZilla/issues/12).
- [ ] [Faster string sorting algorithm](https://github.com/ashvardanian/StringZilla/issues/45).
- [ ] [Splitting with multiple separators at once](https://github.com/ashvardanian/StringZilla/issues/29).
- [ ] Arm NEON backend.
- [ ] Bindings for Rust.
- [ ] Arm SVE backend.
- [ ] Stateful automata-based search.

## Working on Alternative Hardware Backends

## Working on Faster Edit Distances

## Working on Random String Generators

## Working on Sequence Processing and Sorting


