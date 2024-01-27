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

Using modern syntax, this is how you build and run the test suite:

```bash
cmake -DSTRINGZILLA_BUILD_TEST=1 -B build_debug
cmake --build ./build_debug --config Debug          # Which will produce the following targets:
./build_debug/stringzilla_test_cpp20                # Unit test for the entire library
```

For benchmarks, you can use the following commands:

```bash
cmake -DSTRINGZILLA_BUILD_BENCHMARK=1 -B build_release
cmake --build ./build_release --config Release      # Which will produce the following targets:
./build_release/stringzilla_bench_search <path>     # for substring search
./build_release/stringzilla_bench_token <path>      # for hashing, equality comparisons, etc.
./build_release/stringzilla_bench_similarity <path> # for edit distances and alignment scores
./build_release/stringzilla_bench_sort <path>       # for sorting arrays of strings
./build_release/stringzilla_bench_container <path>  # for STL containers with string keys
```

You may want to download some datasets for benchmarks, like these:

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

Running on modern hardware, you may want to compile the code for older generations to compare the relative performance.
The assumption would be that newer ISA extensions would provide better performance.
On x86_64, you can use the following commands to compile for Sandy Bridge, Haswell, and Sapphire Rapids:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
    -DSTRINGZILLA_TARGET_ARCH="sandybridge" -B build_release/sandybridge && \
    cmake --build build_release/sandybridge --config Release
cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
    -DSTRINGZILLA_TARGET_ARCH="haswell" -B build_release/haswell && \
    cmake --build build_release/haswell --config Release
cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
    -DSTRINGZILLA_TARGET_ARCH="sapphirerapids" -B build_release/sapphirerapids && \
    cmake --build build_release/sapphirerapids --config Release
```

Alternatively, you may want to compare the performance of the code compiled with different compilers.
On x86_64, you may want to compare GCC, Clang, and ICX.

```bash
cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
    -DCMAKE_CXX_COMPILER=g++-12 -DCMAKE_C_COMPILER=gcc-12 \
    -B build_release/gcc && cmake --build build_release/gcc --config Release
cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
    -DCMAKE_CXX_COMPILER=clang++-14 -DCMAKE_C_COMPILER=clang-14 \
    -B build_release/clang && cmake --build build_release/clang --config Release
```

## Contributing in Python

Python bindings are implemented using pure CPython, so you wouldn't need to install SWIG, PyBind11, or any other third-party library.

```bash
pip install -e . # To build locally from source
```

For testing we use PyTest, which may not be installed on your system.

```bash
pip install pytest              # To install PyTest
pytest scripts/test.py -s -x    # To run the test suite
```

For fuzzing we love the ability to call the native C implementation from Python bypassing the binding layer.
For that we use Cppyy, derived from Cling, a Clang-based C++ interpreter.

```bash
pip install cppyy                       # To install Cppyy
python scripts/similarity_fuzz.py       # To run the fuzzing script
```

For benchmarking, the following scripts are provided.

```sh
python scripts/bench_search.py --haystack_path "your file" --needle "your pattern" # real data
python scripts/bench_search.py --haystack_pattern "abcd" --haystack_length 1e9 --needle "abce" # synthetic data
python scripts/similarity_bench.py --text_path "your file" # edit ditance computations
```

Before you ship, please make sure the packaging works.

```bash
cibuildwheel --platform linux
```

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

## Roadmap

The project is in its early stages of development.
So outside of basic bug-fixes, several features are still missing, and can be implemented by you.
Future development plans include:

- [x] [Replace PyBind11 with CPython](https://github.com/ashvardanian/StringZilla/issues/35), [blog](https://ashvardanian.com/posts/pybind11-cpython-tutorial/.
- [x] [Bindings for JavaScript](https://github.com/ashvardanian/StringZilla/issues/25).
- [x] [Reverse-order operations](https://github.com/ashvardanian/StringZilla/issues/12).
- [ ] [Faster string sorting algorithm](https://github.com/ashvardanian/StringZilla/issues/45).
- [ ] [Splitting with multiple separators at once](https://github.com/ashvardanian/StringZilla/issues/29).
- [ ] Universal hashing solution.
- [ ] Add `.pyi` interface for Python.
- [ ] Arm NEON backend.
- [ ] Bindings for Rust.
- [ ] Arm SVE backend.
- [ ] Stateful automata-based search.

## General Performance Observations

### Unaligned Loads

One common surface of attach for performance optimizations is minimizing unaligned loads.
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


