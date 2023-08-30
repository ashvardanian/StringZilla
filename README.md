# StringZilla 🦖

StringZilla is the Godzilla of string libraries, splitting, sorting, and shuffling large textual datasets faster than you can say "Tokyo Tower" 😅

- [x] [Python docs](#quick-start-python-🐍)
- [x] [C docs](#quick-start-c-🛠️)
- [ ] JavaScript docs.
- [ ] Rust docs.

## Performance

StringZilla uses a heuristic so simple it's almost stupid... but it works.
It matches the first few letters of words with hyper-scalar code to achieve `memcpy` speeds.
__The implementation fits into a single C 99 header file__ and uses different SIMD flavors and SWAR on older platforms.
So if you're haunted by `open(...).readlines()` and `str().splitlines()` taking forever, this should help 😊

### Substring Search

| Backend \ Device         |                    IoT |                   Laptop |                    Server |
| :----------------------- | ---------------------: | -----------------------: | ------------------------: |
| **Speed Comparison** 🐇  |                        |                          |                           |
| Python `for` loop        |                 4 MB/s |                  14 MB/s |                   11 MB/s |
| C++ `for` loop           |               520 MB/s |                 1.0 GB/s |                  900 MB/s |
| C++ `string.find`        |               560 MB/s |                 1.2 GB/s |                  1.3 GB/s |
| Scalar StringZilla       |                 2 GB/s |                 3.3 GB/s |                  3.5 GB/s |
| Hyper-Scalar StringZilla |           **4.3 GB/s** |              **12 GB/s** |             **12.1 GB/s** |
| **Efficiency Metrics** 📊 |                        |                          |                           |
| CPU Specs                | 8-core ARM, 0.5 W/core | 8-core Intel, 5.6 W/core | 22-core Intel, 6.3 W/core |
| Performance/Core         |         2.1 - 3.3 GB/s |              **11 GB/s** |                 10.5 GB/s |
| Bytes/Joule              |           **4.2 GB/J** |                   2 GB/J |                  1.6 GB/J |

### Partition & Sort

Coming soon.

## Quick Start: Python 🐍

1️. Install via pip: `pip install stringzilla`  
2. Import classes: `from stringzilla import Str, File, Strs`  

### Basic Usage

StringZilla offers two mostly interchangeable core classes:

```python
from stringzilla import Str, File

text1 = Str('some-string')
text2 = File('some-file.txt')
```

The `Str` is designed to replace long Python `str` strings and wrap our C-level API.
On the other hand, the `File` memory-maps a file from persistent memory without loading its copy into RAM.
The contents of that file would remain immutable, and the mapping can be shared by multiple Python processes simultaneously.
A standard dataset pre-processing use case would be to map a sizeable textual dataset like Common Crawl into memory, spawn child processes, and split the job between them.

### Basic Operations

- Length: `len(text) -> int`
- Indexing: `text[42] -> str`
- Slicing: `text[42:46] -> str`

### Advanced Operations

- `'substring' in text -> bool`
- `text.contains('substring', start=0, end=9223372036854775807) -> bool`
- `text.find('substring', start=0, end=9223372036854775807) -> int`
- `text.count('substring', start=0, end=9223372036854775807, allowoverlap=False) -> int`

### Splitting and Line Operations

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

## Quick Start: C 🛠️

There is an ABI-stable C 99 interface, in case you have a database, an operating system, or a runtime you want to integrate with StringZilla.

```c
#include "stringzilla.h"

// Initialize your haystack and needle
strzl_haystack_t haystack = {your_text, your_text_length};
strzl_needle_t needle = {your_subtext, your_subtext_length, your_anomaly_offset};

// Perform string-level operations
size_t character_count = strzl_naive_count_char(haystack, 'a');
size_t character_position = strzl_naive_find_char(haystack, 'a');
size_t substring_position = strzl_naive_find_substr(haystack, needle);

// Perform collection level operations
strzl_array_t array = {your_order, your_count, your_get_begin, your_get_length, your_handle};
strzl_sort(&array, &your_config);
```

## Contributing 👾

Here's how to set up your dev environment and run some tests.
Future development plans include:

- Faster string sorting algorithm.
- Bindings for JavaScript, Java, and Rust.
- Support for reverse-order operations in Python.
- Splitting CSV rows into columns.
- Arm SVE backend.

### Development

```sh
# Clean up and install
rm -rf build && pip install -e . && pytest scripts/test.py -s -x

# Install without dependencies
pip install -e . --no-index --no-deps
```

### Benchmarking

To benchmark on some custom file and pattern combinations:

```sh
python scripts/bench.py --haystack_path "your file" --needle "your pattern"
```

To benchmark on synthetic data:

```sh
python scripts/bench.py --haystack_pattern "abcd" --haystack_length 1e9 --needle "abce"
```

### Packaging

To validate packaging:

```sh
cibuildwheel --platform linux
```

### Compiling C++ Tests

```sh
# Install dependencies
brew install libomp llvm

# Compile and run tests
cmake -B ./build_release \
    -DCMAKE_C_COMPILER="/opt/homebrew/opt/llvm/bin/clang" \
    -DCMAKE_CXX_COMPILER="/opt/homebrew/opt/llvm/bin/clang++" \
    -DSTRINGZILLA_USE_OPENMP=1 \
    -DSTRINGZILLA_BUILD_TEST=1 \
    && \
    make -C ./build_release -j && ./build_release/stringzilla_test
```

## License 📜

Feel free to use the project under Apache 2.0 or the Three-clause BSD license at your preference.

---

If you like this project, you may also enjoy [USearch][usearch], [UCall][ucall], [UForm][uform], [UStore][ustore], [SimSIMD][simsimd], and [TenPack][tenpack] 🤗

[usearch]: https://github.com/unum-cloud/usearch
[ucall]: https://github.com/unum-cloud/ucall
[uform]: https://github.com/unum-cloud/uform
[ustore]: https://github.com/unum-cloud/ustore
[simsimd]: https://github.com/ashvardanian/simsimd
[tenpack]: https://github.com/ashvardanian/tenpack
