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
| __Speed Comparison__ 🐇   |                        |                          |                           |
| Python `for` loop        |                 4 MB/s |                  14 MB/s |                   11 MB/s |
| C++ `for` loop           |               520 MB/s |                 1.0 GB/s |                  900 MB/s |
| C++ `string.find`        |               560 MB/s |                 1.2 GB/s |                  1.3 GB/s |
| Scalar StringZilla       |                 2 GB/s |                 3.3 GB/s |                  3.5 GB/s |
| Hyper-Scalar StringZilla |           __4.3 GB/s__ |              __12 GB/s__ |             __12.1 GB/s__ |
| __Efficiency Metrics__ 📊 |                        |                          |                           |
| CPU Specs                | 8-core ARM, 0.5 W/core | 8-core Intel, 5.6 W/core | 22-core Intel, 6.3 W/core |
| Performance/Core         |         2.1 - 3.3 GB/s |              __11 GB/s__ |                 10.5 GB/s |
| Bytes/Joule              |           __4.2 GB/J__ |                   2 GB/J |                  1.6 GB/J |

### Partition & Sort

Coming soon.

## Quick Start: Python 🐍

1️. Install via pip: `pip install stringzilla`  
1. Import the classes you need: `from stringzilla import Str, Strs, File`  

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
levenstein: int = sz.levenstein("needle", "nidl")
```

## Quick Start: C 🛠️

There is an ABI-stable C 99 interface, in case you have a database, an operating system, or a runtime you want to integrate with StringZilla.

```c
#include "stringzilla.h"

// Initialize your haystack and needle
sz_haystack_t haystack = {your_text, your_text_length};
sz_needle_t needle = {your_subtext, your_subtext_length, your_anomaly_offset};

// Perform string-level operations
size_t character_count = sz_count_char_swar(haystack, 'a');
size_t character_position = sz_find_char_swar(haystack, 'a');
size_t substring_position = sz_find_substr_swar(haystack, needle);

// Perform collection level operations
sz_sequence_t array = {your_order, your_count, your_get_start, your_get_length, your_handle};
sz_sort(&array, &your_config);
```

## Contributing 👾

Future development plans include:

- [x] Replace PyBind11 with CPython.
- Reverse-order operations in Python #12.
- Bindings for JavaScript #25, Java, and Rust.
- Faster string sorting algorithm.
- Splitting CSV rows into columns.
- Splitting with multiple separators at once #29.
- UTF-8 validation.
- Arm SVE backend.

Here's how to set up your dev environment and run some tests.

### Development

CPython:

```sh
# Clean up and install
rm -rf build && pip install -e . && pytest scripts/test.py -s -x

# Install without dependencies
pip install -e . --no-index --no-deps
```

NodeJS:

```sh
npm install && node javascript/test.js
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
