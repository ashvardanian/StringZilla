# StringZilla 🦖

StringZilla is the GodZilla of string libraries, using [SIMD][faq-simd] and [SWAR][faq-swar] to accelerate string operations for modern CPUs.
It is significantly faster than the default string libraries in Python and C++, and offers a more powerful API.
Aside from exact search, the library also accelerates fuzzy search, edit distance computation, and sorting.

[faq-simd]: https://en.wikipedia.org/wiki/Single_instruction,_multiple_data
[faq-swar]: https://en.wikipedia.org/wiki/SWAR

- Code in C? Replace LibC's `<string.h>` with C 99 `<stringzilla.h>`  - [_more_](#quick-start-c-🛠️)
- Code in C++? Replace STL's `<string>` with C++ 11 `<stringzilla.hpp>` - [_more_](#quick-start-cpp-🛠️)
- Code in Python? Upgrade your `str` to faster `Str` - [_more_](#quick-start-python-🐍)

__Features:__

| Feature \ Library              | C++ STL |    LibC |      StringZilla |
| :----------------------------- | ------: | ------: | ---------------: |
| Substring Search               |  1 GB/s | 12 GB/s |          12 GB/s |
| Reverse Order Substring Search |  1 GB/s |       ❌ |          12 GB/s |
| Fuzzy Search                   |       ❌ |       ❌ |                ? |
| Levenshtein Edit Distance      |       ❌ |       ❌ |                ✅ |
| Hashing                        |       ✅ |       ❌ |                ✅ |
| Interface                      |     C++ |       C | C , C++ , Python |

> Benchmarks were conducted on a 1 GB English text corpus, with an average word length of 5 characters.
> The hardware used is an AVX-512 capable Intel Sapphire Rapids CPU.
> The code was compiled with GCC 12, using `glibc` v2.35.

__Who is this for?__

- For data-engineers often memory-mapping and parsing large datasets, like the [CommonCrawl](https://commoncrawl.org/).
- For Python, C, or C++ software engineers looking for faster strings for their apps.
- For Bioinformaticians and Search Engineers measuring edit distances and fuzzy-matching.
- For students learning practical applications of SIMD and SWAR and how libraries like LibC are implemented.
- For hardware designers, needing a SWAR baseline for strings-processing functionality.

__Limitations:__

- Assumes little-endian architecture (most CPUs, including x86, Arm, RISC-V).
- Assumes ASCII or UTF-8 encoding (most content and systems).
- Assumes 64-bit address space (most modern CPUs).

__Technical insghts:__

- Uses SWAR and SIMD to accelerate exact search for very short needles under 4 bytes.
- Uses the Shift-Or Bitap algorithm for mid-length needles under 64 bytes.
- Uses the Boyer-Moore-Horpool algorithm with Raita heuristic for longer needles.
- Uses the Manber-Wu improvement of the Shift-Or algorithm for bounded fuzzy search.
- Uses the two-row Wagner-Fisher algorithm for edit distance computation.
- Uses the Needleman-Wunsh improvement for parameterized edit distance computation.
- Uses the Karp-Rabin rolling hashes to produce binary fingerprints.
- Uses Radix Sort to accelerate sorting of strings.

The choice of the optimal algorithm is predicated on the length of the needle and the alphabet cardinality.
If the amount of compute per byte is low and the needles are beyond longer than the cache-line (64 bytes), skip-table-based approaches are preferred.
In other cases, brute force approaches can be more efficient.
On the engineering side, the library:

- Implement the Small String Optimization for strings shorter than 23 bytes.
- Avoids PyBind11, SWIG, `ParseTuple` and other CPython sugar to minimize call latency. [_details_](https://ashvardanian.com/posts/pybind11-cpython-tutorial/) 

## Quick Start: Python 🐍

1. Install via pip: `pip install stringzilla`  
2. Import the classes you need: `from stringzilla import Str, Strs, File`  

### Basic Usage

StringZilla offers two mostly interchangeable core classes:

```python
from stringzilla import Str, File

text_from_str = Str('some-string')
text_from_file = Str(File('some-file.txt'))
```

The `Str` is designed to replace long Python `str` strings and wrap our C-level API.
On the other hand, the `File` memory-maps a file from persistent memory without loading its copy into RAM.
The contents of that file would remain immutable, and the mapping can be shared by multiple Python processes simultaneously.
A standard dataset pre-processing use case would be to map a sizeable textual dataset like Common Crawl into memory, spawn child processes, and split the job between them.

### Basic Operations

- Length: `len(text) -> int`
- Indexing: `text[42] -> str`
- Slicing: `text[42:46] -> Str`
- String conversion: `str(text) -> str`
- Substring check: `'substring' in text -> bool`
- Hashing: `hash(text) -> int`

### Advanced Operations

- `text.contains('substring', start=0, end=9223372036854775807) -> bool`
- `text.find('substring', start=0, end=9223372036854775807) -> int`
- `text.count('substring', start=0, end=9223372036854775807, allowoverlap=False) -> int`
- `text.splitlines(keeplinebreaks=False, separator='\n') -> Strs`
- `text.split(separator=' ', maxsplit=9223372036854775807, keepseparator=False) -> Strs`

### Collection-Level Operations

Once split into a `Strs` object, you can sort, shuffle, and reorganize the slices.

```python
lines: Strs = text.split(separator='\n')
lines.sort()
lines.shuffle(seed=42)
```

Need copies?

```python
sorted_copy: Strs = lines.sorted()
shuffled_copy: Strs = lines.shuffled(seed=42)
```

Basic `list`-like operations are also supported:

```python
lines.append('Pythonic string')
lines.extend(shuffled_copy)
```

### Low-Level Python API

The StringZilla CPython bindings implement vector-call conventions for faster calls.

```py
import stringzilla as sz

contains: bool = sz.contains("haystack", "needle", start=0, end=9223372036854775807)
offset: int = sz.find("haystack", "needle", start=0, end=9223372036854775807)
count: int = sz.count("haystack", "needle", start=0, end=9223372036854775807, allowoverlap=False)
levenshtein: int = sz.levenshtein("needle", "nidl")
```

## Quick Start: C/C++ 🛠️

The library is header-only, so you can just copy the `stringzilla.h` header into your project.
Alternatively, add it as a submodule, and include it in your build system.

```sh
git submodule add https://github.com/ashvardanian/stringzilla.git
```

Or using a pure CMake approach:

```cmake
FetchContent_Declare(stringzilla GIT_REPOSITORY https://github.com/ashvardanian/stringzilla.git)
FetchContent_MakeAvailable(stringzilla)
```

### Basic Usage with C 99 and Newer

There is a stable C 99 interface, where all function names are prefixed with `sz_`.
Most interfaces are well documented, and come with self-explanatory names and examples.
In some cases, hardware specific overloads are available, like `sz_find_avx512` or `sz_find_neon`.
Both are companions of the `sz_find`, first for x86 CPUs with AVX-512 support, and second for Arm NEON-capable CPUs.

```c
#include <stringzilla/stringzilla.h>

// Initialize your haystack and needle
sz_string_view_t haystack = {your_text, your_text_length};
sz_string_view_t needle = {your_subtext, your_subtext_length};

// Perform string-level operations
sz_size_t substring_position = sz_find(haystack.start, haystack.length, needle.start, needle.length);
sz_size_t substring_position = sz_find_avx512(haystack.start, haystack.length, needle.start, needle.length);
sz_size_t substring_position = sz_find_neon(haystack.start, haystack.length, needle.start, needle.length);

// Hash strings
sz_u64_t hash = sz_hash(haystack.start, haystack.length);

// Perform collection level operations
sz_sequence_t array = {your_order, your_count, your_get_start, your_get_length, your_handle};
sz_sort(&array, &your_config);
```

### Basic Usage with C++ 11 and Newer

There is a stable C++ 11 interface available in ther `ashvardanian::stringzilla` namespace.
It comes with two STL-like classes: `string_view` and `string`.
The first is a non-owning view of a string, and the second is a mutable string with a [Small String Optimization][faq-sso].

```cpp
#include <stringzilla/stringzilla.hpp>

namespace sz = ashvardanian::stringzilla;

sz::string haystack = "some string";
sz::string_view needle = sz::string_view(haystack).substr(0, 4);

auto substring_position = haystack.find(needle); // Or `rfind`
auto hash = std::hash<sz::string_view>(haystack); // Compatible with STL's `std::hash`

haystack.end() - haystack.begin() == haystack.size(); // Or `rbegin`, `rend`
haystack.find_first_of(" \w\t") == 4; // Or `find_last_of`, `find_first_not_of`, `find_last_not_of`
haystack.starts_with(needle) == true; // Or `ends_with`
haystack.remove_prefix(needle.size()); // Why is this operation inplace?!
haystack.contains(needle) == true; // STL has this only from C++ 23 onwards
haystack.compare(needle) == 1; // Or `haystack <=> needle` in C++ 20 and beyond
```

### Beyond Standard Templates Library

Aside from conventional `std::string` interfaces, non-STL extensions are available.

```cpp
haystack.count(needle) == 1; // Why is this not in STL?!
haystack.edit_distance(needle) == 7;
haystack.find_edited(needle, bound);
haystack.rfind_edited(needle, bound);
```

### Ranges

One of the most common use cases is to split a string into a collection of substrings.
Which would often result in snippets like the one below.

```cpp
std::vector<std::string> lines = your_split(haystack, '\n');
std::vector<std::string> words = your_split(lines, ' ');
```

Those allocate memory for each string and the temporary vectors.
Each of those can be orders of magnitude more expensive, than even serial for-loop over character.
To avoid those, StringZilla provides lazily-evaluated ranges.

```cpp
for (auto line : split_substrings(haystack, '\r\n'))
    for (auto word : split_chars(line, ' \w\t.,;:!?'))
        std::cout << word << std::endl;
```

Each of those is available in reverse order as well.
It also allows interleaving matches, and controlling the inclusion/exclusion of the separator itself into the result.
Debugging pointer offsets is not a pleasant excersise, so keep the following functions in mind.

- `split_substrings`.
- `split_chars`.
- `split_not_chars`.
- `reverse_split_substrings`.
- `reverse_split_chars`.
- `reverse_split_not_chars`.
- `search_substrings`.
- `reverse_search_substrings`.
- `search_chars`.
- `reverse_search_chars`.
- `search_other_chars`.
- `reverse_search_other_chars`.

### Debugging

For maximal performance, the library does not perform any bounds checking in Release builds.
That behaviour is controllable for both C and C++ interfaces via the `STRINGZILLA_DEBUG` macro.

[faq-sso]: https://cpp-optimizations.netlify.app/small_strings/

## Contributing 👾

Please check out the [contributing guide](CONTRIBUTING.md) for more details on how to setup the development environment and contribute to this project.
If you like this project, you may also enjoy [USearch][usearch], [UCall][ucall], [UForm][uform], and [SimSIMD][simsimd]. 🤗

[usearch]: https://github.com/unum-cloud/usearch
[ucall]: https://github.com/unum-cloud/ucall
[uform]: https://github.com/unum-cloud/uform
[simsimd]: https://github.com/ashvardanian/simsimd

### Development

CPython:

```sh
# Clean up, install, and test!
rm -rf build && pip install -e . && pytest scripts/ -s -x

# Install without dependencies
pip install -e . --no-index --no-deps
```

NodeJS:

```sh
npm install && npm test
```

### Benchmarking

To benchmark on some custom file and pattern combinations:

```sh
python scripts/bench_substring.py --haystack_path "your file" --needle "your pattern"
```

To benchmark on synthetic data:

```sh
python scripts/bench_substring.py --haystack_pattern "abcd" --haystack_length 1e9 --needle "abce"
```

### Packaging

To validate packaging:

```sh
cibuildwheel --platform linux
```

### Compiling C++ Tests

Running benchmarks:

```sh
cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 -B ./build_release
cmake --build build_release --config Release
./build_release/stringzilla_bench_substring
```

Comparing different hardware setups:

```sh
cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
    -DCMAKE_CXX_FLAGS="-march=sandybridge" -DCMAKE_C_FLAGS="-march=sandybridge" \
    -B ./build_release/sandybridge && cmake --build build_release/sandybridge --config Release
cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
    -DCMAKE_CXX_FLAGS="-march=haswell" -DCMAKE_C_FLAGS="-march=haswell" \
    -B ./build_release/haswell && cmake --build build_release/haswell --config Release
cmake -DCMAKE_BUILD_TYPE=Release -DSTRINGZILLA_BUILD_BENCHMARK=1 \
    -DCMAKE_CXX_FLAGS="-march=sapphirerapids" -DCMAKE_C_FLAGS="-march=sapphirerapids" \
    -B ./build_release/sapphirerapids && cmake --build build_release/sapphirerapids --config Release

./build_release/sandybridge/stringzilla_bench_substring
./build_release/haswell/stringzilla_bench_substring
./build_release/sapphirerapids/stringzilla_bench_substring
```

Running tests:

```sh
cmake -DCMAKE_BUILD_TYPE=Debug -DSTRINGZILLA_BUILD_TEST=1 -B ./build_debug
cmake --build build_debug --config Debug
./build_debug/stringzilla_test_substring
```

On MacOS it's recommended to use non-default toolchain:

```sh
# Install dependencies
brew install libomp llvm

# Compile and run tests
cmake -B ./build_release \
    -DCMAKE_C_COMPILER="gcc-12" \
    -DCMAKE_CXX_COMPILER="g++-12" \
    -DSTRINGZILLA_USE_OPENMP=1 \
    -DSTRINGZILLA_BUILD_TEST=1 \
    -DSTRINGZILLA_BUILD_BENCHMARK=1 \
    && \
    make -C ./build_release -j && ./build_release/stringzilla_bench_substring
```

## License 📜

Feel free to use the project under Apache 2.0 or the Three-clause BSD license at your preference.

---




# The weirdest interfaces of C++23 strings:

## Third `std::basic_string_view<CharT,Traits>::find`

constexpr size_type find( basic_string_view v, size_type pos = 0 ) const noexcept;
(1)	(since C++17)
constexpr size_type find( CharT ch, size_type pos = 0 ) const noexcept;
(2)	(since C++17)
constexpr size_type find( const CharT* s, size_type pos, size_type count ) const;
(3)	(since C++17)
constexpr size_type find( const CharT* s, size_type pos = 0 ) const;
(4)	(since C++17)


## HTML Parsing

```txt
<tag>       Isolated tag start
<tag\w      Tag start with attributes
<tag/>      Self-closing tag
</tag>      Tag end
```

In any case, the tag name is always followed by whitespace, `/` or `>`.
And is always preceded by whitespace. `/` or `<`.

Important distinctions between XML and HTML:

- XML does not truncate multiple white-spaces, while HTML does.