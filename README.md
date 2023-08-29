# StringZilla: The Godzilla of String Libraries 🦖

Welcome to StringZilla, where we don't just handle strings, we *devour* them! 🍽️ If you've been on the hunt for a string library that's not just fast but *freakishly fast*, you've hit the jackpot. 🎰 StringZilla is the Godzilla of string libraries, stomping through your text faster than you can say "Tokyo Tower"! 🗼

## Unleash the Beast: Performance 🚀

StringZilla uses a heuristic so simple, it's almost stupid. But don't be fooled! This bad boy matches the first few letters of words with hyper-scalar code to achieve ludicrous speed. 🏎️💨 It's practical, easy to implement with different flavors of SIMD, and even SWAR for those less fortunate platforms. If you're haunted by `open(...).readlines()` and `str().splitlines()` taking forever, then StringZilla is your dream come true. 🌈

### The Speed Showdown 🏁

| Algorithm / Metric        |    IoT (8-core ARM, 0.5 W/core)    |  Laptop (8-core Intel, 5.6 W/core)  |  Server (22-core Intel, 6.3 W/core)   |
| :------------------------ | :--------------------------------: | :--------------------------------: | :-----------------------------------: |
| **Speed Comparison** 🐢🐇  |                                   |                                    |                                       |
| Python `for`-loop 🐌           |  4 MB/s                           | 14 MB/s                            |  11 MB/s                              |
| C++ `for`-loop 🏍️              | 520 MB/s                          | 1.0 GB/s                           | 900 MB/s                              |
| C++ `std::string::find` 🚗    | 560 MB/s                          | 1.2 GB/s                           | 1.3 GB/s                              |
| Scalar Stringzilla 🚀  |  2 GB/s                           | 3.3 GB/s                           | 3.5 GB/s                              |
| Hyper-Scalar Stringzilla 🛸 | **4.3 GB/s**                | **12 GB/s**                        | **12.1 GB/s**                         |
| **Performance Metrics** 📊|                                   |                                    |                                       |
| Performance/Core 💪       | 2.1 - 3.3 GB/s                    | **11 GB/s**                         | 10.5 GB/s                             |
| Bytes/Joule ⚡            | **4.2 GB/J**                      | 2 GB/J                              | 1.6 GB/J                              |

## Quick Start: Python 🐍

1️⃣ Install via pip: `pip install stringzilla`  
2️⃣ Import classes: `from stringzilla import Str, File, Strs`  
3️⃣ Unleash the beast with built-in methods for string operations. 🎉

### Basic Usage 🛠️

Stringzilla offers two interchangeable classes for your string and file munching needs:

```python
from stringzilla import Str, File

text1 = Str('some-string')
text2 = File('some-file.txt')
```

### Basic Operations 📏

- Length: `len(text) -> int`
- Substring check: `'substring' in text -> bool`
- Indexing: `text[42] -> str`
- Slicing: `text[42:46] -> str`

### Advanced Operations 🧠

- `text.contains('substring', start=0, end=9223372036854775807) -> bool`
- `text.find('substring', start=0, end=9223372036854775807) -> int`
- `text.count('substring', start=0, end=9223372036854775807, allowoverlap=False) -> int`

### Splitting and Line Operations 🍕

- `text.splitlines(keeplinebreaks=False, separator='\n') -> Strs`
- `text.split(separator=' ', maxsplit=9223372036854775807, keepseparator=False) -> Strs`

### Collection-Level Operations 🎲

Once split into a `Strs` object, you can sort, shuffle, and more:

```python
lines = text.split(separator='\n')
lines.sort()
lines.shuffle(seed=42)
```

Sorted or shuffled copies? No problemo!

```python
sorted_copy = lines.sorted()
shuffled_copy = lines.shuffled(seed=42)
```

Appending and extending? Easy peasy!

```python
lines.append('Pythonic string')
lines.extend(shuffled_copy)
```

So what are you waiting for? Unleash the Godzilla of string libraries on your code today! 🦖🔥

## Quick Start: C 🛠️🔥

Building a database, an operating system, or a runtime for your new fancy programming language? Why settle for LibC when you can unleash the Godzilla of string libraries? 🦖

```c
#include "stringzilla.h"

// Initialize your haystack and needle
strzl_haystack_t haystack = {your_text, your_text_length};
strzl_needle_t needle = {your_subtext, your_subtext_length, your_anomaly_offset};

// Count occurrences of a character like a boss 😎
size_t count = strzl_naive_count_char(haystack, 'a');

// Find a character like you're searching for treasure 🏴‍☠️
size_t position = strzl_naive_find_char(haystack, 'a');

// Find a substring like it's Waldo 🕵️‍♂️
size_t substring_position = strzl_naive_find_substr(haystack, needle);

// Sort an array of strings like you're Marie Kondo 🗂️
strzl_array_t array = {your_order, your_count, your_get_begin, your_get_length, your_handle};
strzl_sort(&array, &your_config);
```

## Contributing: Be a Part of the Monster Squad! 👾

Ready to contribute? Here's how you can set up your dev environment and run some tests.

### Development Scripts 📜

```sh
# Clean up and install
rm -rf build && pip install -e . && pytest scripts/test.py -s -x

# Install without dependencies
pip install -e . --no-index --no-deps
```

### Benchmarking 🏋️‍♂️

To benchmark on some custom file and pattern combinations:

```sh
python scripts/bench.py --haystack_path "your file" --needle "your pattern"
```

To benchmark on synthetic data:

```sh
python scripts/bench.py --haystack_pattern "abcd" --haystack_length 1e9 --needle "abce"
```

### Packaging 📦

To validate packaging:

```sh
cibuildwheel --platform linux
```

### Compiling C++ Tests 🧪

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

So, are you ready to join the Monster Squad and make StringZilla even more epic? Let's do this! 🦖🚀
