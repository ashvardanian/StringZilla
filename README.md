# StringZilla 🦖

![StringZilla banner](https://github.com/ashvardanian/ashvardanian/blob/master/repositories/StringZilla.png?raw=true)

The world wastes a minimum of $100M annually due to inefficient string operations.
A typical codebase processes strings character by character, resulting in too many branches and data-dependencies, neglecting 90% of modern CPU's potential.
LibC is different.
It attempts to leverage SIMD instructions to boost some operations, and is often used by higher-level languages, runtimes, and databases.
But it isn't perfect.
1️⃣ First, even on common hardware, including over a billion 64-bit ARM CPUs, common functions like `strstr` and `memmem` only achieve 1/3 of the CPU's throughput.
2️⃣ Second, SIMD coverage is inconsistent: acceleration in forward scans does not guarantee speed in the reverse-order search.
3️⃣ At last, most high-level languages can't always use LibC, as the strings are often not NULL-terminated or may contain the Unicode "Zero" character in the middle of the string.
That's why StringZilla was created.
To provide predictably high performance, portable to any modern platform, operating system, and programming language.

[![StringZilla Python installs](https://static.pepy.tech/personalized-badge/stringzilla?period=total&units=abbreviation&left_color=black&right_color=blue&left_text=StringZilla%20Python%20installs)](https://github.com/ashvardanian/stringzilla)
[![StringZilla Rust installs](https://img.shields.io/crates/d/stringzilla?logo=rust&label=Rust%20installs)](https://crates.io/crates/stringzilla)
[![Ubuntu status](https://img.shields.io/github/checks-status/ashvardanian/StringZilla/main?checkName=Linux%20CI&label=Ubuntu)](https://github.com/ashvardanian/StringZilla/actions/workflows/release.yml)
[![Windows status](https://img.shields.io/github/checks-status/ashvardanian/StringZilla/main?checkName=Windows%20CI&label=Windows)](https://github.com/ashvardanian/StringZilla/actions/workflows/release.yml)
[![macOS status](https://img.shields.io/github/checks-status/ashvardanian/StringZilla/main?checkName=macOS%20CI&label=macOS)](https://github.com/ashvardanian/StringZilla/actions/workflows/release.yml)
![StringZilla code size](https://img.shields.io/github/languages/code-size/ashvardanian/stringzilla)

StringZilla is the GodZilla of string libraries, using [SIMD][faq-simd] and [SWAR][faq-swar] to accelerate string operations on modern CPUs.
It is up to __10x faster than the default and even other SIMD-accelerated string libraries__ in C, C++, Python, and other languages, while covering broad functionality.
It __accelerates exact and fuzzy string matching, edit distance computations, sorting, lazily-evaluated ranges to avoid memory allocations, and even random-string generators__.

[faq-simd]: https://en.wikipedia.org/wiki/Single_instruction,_multiple_data
[faq-swar]: https://en.wikipedia.org/wiki/SWAR

- 🐂 __[C](#basic-usage-with-c-99-and-newer) :__ Upgrade LibC's `<string.h>` to `<stringzilla/stringzilla.h>`  in C 99
- 🐉 __[C++](#basic-usage-with-c-11-and-newer):__ Upgrade STL's `<string>` to `<stringzilla/stringzilla.hpp>` in C++ 11
- 🧮 __[CUDA](#cuda):__ Process in-bulk with `<stringzillas/stringzillas.cuh>` in CUDA C++ 17
- 🐍 __[Python](#quick-start-python-🐍):__ Upgrade your `str` to faster `Str`
- 🦀 __[Rust](#quick-start-rust-🦀):__ Use the `StringZilla` traits crate
- 🦫 __[Go](#quick-start-golang-🦫):__ Use the `StringZilla` cGo module
- 🍎 __[Swift](#quick-start-swift-🍏):__ Use the `String+StringZilla` extension
- 🟨 __[JavaScript](#quick-start-javascript-🟨):__ Use the `StringZilla` library
- 🐚 __[Shell][faq-shell]__: Accelerate common CLI tools with `sz_` prefix
- 📚 Researcher? Jump to [Algorithms & Design Decisions](#algorithms--design-decisions-📚)
- 💡 Thinking to contribute? Look for ["good first issues"][first-issues]
- 🤝 And check the [guide](https://github.com/ashvardanian/StringZilla/blob/main/CONTRIBUTING.md) to set up the environment
- Want more bindings or features? Let [me](https://github.com/ashvardanian) know!

[faq-shell]: https://github.com/ashvardanian/StringZilla/blob/main/cli/README.md
[first-issues]: https://github.com/ashvardanian/StringZilla/issues

__Who is this for?__

- For data-engineers parsing large datasets, like the [CommonCrawl](https://commoncrawl.org/), [RedPajama](https://github.com/togethercomputer/RedPajama-Data), or [LAION](https://laion.ai/blog/laion-5b/).
- For software engineers optimizing strings in their apps and services.
- For bioinformaticians and search engineers looking for edit-distances for [USearch](https://github.com/unum-cloud/usearch).
- For [DBMS][faq-dbms] devs, optimizing `LIKE`, `ORDER BY`, and `GROUP BY` operations.
- For hardware designers, needing a SWAR baseline for string-processing functionality.
- For students studying SIMD/SWAR applications to non-data-parallel operations.

[faq-dbms]: https://en.wikipedia.org/wiki/Database

## Performance

<table style="width: 100%; text-align: center; table-layout: fixed;">
  <colgroup>
    <col style="width: 25%;">
    <col style="width: 25%;">
    <col style="width: 25%;">
    <col style="width: 25%;">
  </colgroup>
  <tr>
    <th align="center">C</th>
    <th align="center">C++</th>
    <th align="center">Python</th>
    <th align="center">StringZilla</th>
  </tr>
  <!-- Substrings, normal order -->
  <tr>
    <td colspan="4" align="center">find the first occurrence of a random word from text, ≅ 5 bytes long</td>
  </tr>
  <tr>
    <td align="center">
      <code>strstr</code> <sup>1</sup><br/>
      <span style="color:#ABABAB;">x86:</span> <b>7.4</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>2.0</b> GB/s
    </td>
    <td align="center">
      <code>.find</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>2.9</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>1.6</b> GB/s
    </td>
    <td align="center">
      <code>.find</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>1.1</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.6</b> GB/s
    </td>
    <td align="center">
      <code>sz_find</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>10.6</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>7.1</b> GB/s
    </td>
  </tr>
  <!-- Substrings, reverse order -->
  <tr>
    <td colspan="4" align="center">find the last occurrence of a random word from text, ≅ 5 bytes long</td>
  </tr>
  <tr>
    <td align="center">⚪</td>
    <td align="center">
      <code>.rfind</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.5</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.4</b> GB/s
    </td>
    <td align="center">
      <code>.rfind</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.9</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.5</b> GB/s
    </td>
    <td align="center">
      <code>sz_rfind</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>10.8</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>6.7</b> GB/s
    </td>
  </tr>
  <!-- Characters, normal order -->
  <tr>
    <td colspan="4" align="center">split lines separated by <code>\n</code> or <code>\r</code> <sup>2</sup></td>
  </tr>
  <tr>
    <td align="center">
      <code>strcspn</code> <sup>1</sup><br/>
      <span style="color:#ABABAB;">x86:</span> <b>5.42</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>2.19</b> GB/s
    </td>
    <td align="center">
      <code>.find_first_of</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.59</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.46</b> GB/s
    </td>
    <td align="center">
      <code>re.finditer</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.06</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.02</b> GB/s
    </td>
    <td align="center">
      <code>sz_find_byteset</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>4.08</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>3.22</b> GB/s
    </td>
  </tr>
  <!-- Characters, reverse order -->
  <tr>
    <td colspan="4" align="center">find the last occurrence of any of 6 whitespaces <sup>2</sup></td>
  </tr>
  <tr>
    <td align="center">⚪</td>
    <td align="center">
      <code>.find_last_of</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.25</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.25</b> GB/s
    </td>
    <td align="center">⚪</td>
    <td align="center">
      <code>sz_rfind_byteset</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.43</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.23</b> GB/s
    </td>
  </tr>
  <!-- Random Generation -->
  <tr>
    <td colspan="4" align="center">Random string from a given alphabet, 20 bytes long <sup>5</sup></td>
  </tr>
  <tr>
    <td align="center">
      <code>rand() % n</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>18.0</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>9.4</b> MB/s
    </td>
    <td align="center">
      <code>std::uniform_int_distribution</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>47.2</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>20.4</b> MB/s
    </td>
    <td align="center">
      <code>join(random.choices(...))</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>13.3</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>5.9</b> MB/s
    </td>
    <td align="center">
      <code>sz_fill_random</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>56.2</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>25.8</b> MB/s
    </td>
  </tr>
  <!-- Mapping characters with lookup table transforms -->
  <tr>
    <td colspan="4" align="center">Mapping characters with lookup table transforms</td>
  </tr>
  <tr>
    <td align="center">⚪</td>
    <td align="center">
      <code>std::transform</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>3.81</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>2.65</b> GB/s
    </td>
    <td align="center">
      <code>str.translate</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>260.0</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>140.0</b> MB/s
    </td>
    <td align="center">
      <code>sz_lookup</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>21.2</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>8.5</b> GB/s
    </td>
  </tr>
  <!-- Sorting -->
  <tr>
    <td colspan="4" align="center">Get sorted order, ≅ 8 million English words <sup>6</sup></td>
  </tr>
  <tr>
    <td align="center">
      <code>qsort_r</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>3.55</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>5.77</b> s
    </td>
    <td align="center">
      <code>std::sort</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>2.79</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>4.02</b> s
    </td>
    <td align="center">
      <code>numpy.argsort</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>7.58</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>13.00</b> s
    </td>
    <td align="center">
      <code>sz_sequence_argsort</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>1.91</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>2.37</b> s
    </td>
  </tr>
  <!-- Edit Distance -->
  <tr>
    <td colspan="4" align="center">Levenshtein edit distance, text lines ≅ 100 bytes long</td>
  </tr>
  <tr>
    <td align="center">⚪</td>
    <td align="center">⚪</td>
    <td align="center">
      via <code>NLTK</code> <sup>3</sup><br/>
      <span style="color:#ABABAB;">x86:</span> <b>2,490,161</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>2,081,543</b> CUPS
    </td>
    <td align="center">
      <code>szs_levenshtein_distances_t</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>78,851,644</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>36,857,367</b> &centerdot;
      <span style="color:#ABABAB;">cuda:</span> <b>3,369,569,512</b> CUPS
    </td>
  </tr>
  <!-- Alignment Score -->
  <tr>
    <td colspan="4" align="center">Needleman-Wunsch alignment scores, proteins ≅ 1 K amino acids long</td>
  </tr>
  <tr>
    <td align="center">⚪</td>
    <td align="center">⚪</td>
    <td align="center">
      via <code>biopython</code> <sup>4</sup><br/>
      <span style="color:#ABABAB;">x86:</span> <b>575,981,513</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>436,350,732</b> CUPS
    </td>
    <td align="center">
      <code>szs_needleman_wunsch_scores_t</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>452,629,942</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>520,170,239</b> &centerdot;
      <span style="color:#ABABAB;">cuda:</span> <b>9,017,327,818</b> CUPS
    </td>
  </tr>
</table>

StringZilla has a lot of functionality, most of which is covered by benchmarks across C, C++, Python and other languages.
You can find those in the `./scripts` directory, with usage notes listed in the [`CONTRIBUTING.md`](CONTRIBUTING.md) file.
Notably, if the CPU supports misaligned loads, even the 64-bit SWAR backends are faster than either standard library.

> Most benchmarks were conducted on a 1 GB English text corpus, with an average word length of 6 characters.
> The code was compiled with GCC 12, using `glibc` v2.35.
> The benchmarks were performed on Arm-based Graviton3 AWS `c7g` instances and `r7iz` Intel Sapphire Rapids.
> Most modern Arm-based 64-bit CPUs will have similar relative speedups.
> Variance within x86 CPUs will be larger.
> For CUDA benchmarks, the Nvidia H100 GPUs were used.
> <sup>1</sup> Unlike other libraries, LibC requires strings to be NULL-terminated.
> <sup>2</sup> Six whitespaces in the ASCII set are: ` \t\n\v\f\r`. Python's and other standard libraries have specialized functions for those.
> <sup>3</sup> Most Python libraries for strings are also implemented in C.
> <sup>4</sup> Unlike the rest of BioPython, the alignment score computation is [implemented in C](https://github.com/biopython/biopython/blob/master/Bio/Align/_pairwisealigner.c).
> <sup>5</sup> All modulo operations were conducted with `uint8_t` to allow compilers more optimization opportunities.
> The C++ STL and StringZilla benchmarks used a 64-bit [Mersenne Twister][faq-mersenne-twister] as the generator.
> For C, C++, and StringZilla, an in-place update of the string was used.
> In Python every string had to be allocated as a new object, which makes it less fair.
> <sup>6</sup> Contrary to the popular opinion, Python's default `sorted` function works faster than the C and C++ standard libraries.
> That holds for large lists or tuples of strings, but fails as soon as you need more complex logic, like sorting dictionaries by a string key, or producing the "sorted order" permutation.
> The latter is very common in database engines and is most similar to `numpy.argsort`.
> The current StringZilla solution can be at least 4x faster without loss of generality.

[faq-mersenne-twister]: https://en.wikipedia.org/wiki/Mersenne_Twister

## Functionality

StringZilla is compatible with most modern CPUs, and provides a broad range of functionality.
It's split into 2 layers:

1. StringZilla: single-header C library and C++ wrapper for high-performance string operations.
2. StringZillas: parallel CPU/GPU backends used for large-batch operations and accelerators.

Having a second C++/CUDA layer greatly simplifies the implementation of similarity scoring and fingerprinting functions, which would otherwise require too much error-prone boilerplate code in pure C.
Both layers are designed to be extremely portable:

- [x] across both little-endian and big-endian architectures.
- [x] across 32-bit and 64-bit hardware architectures.
- [x] across operating systems and compilers.
- [x] across ASCII and UTF-8 encoded inputs.

Not all features are available across all bindings.
Consider contributing if you need a feature that's not yet implemented.

|                                | Maturity |   C   |  C++  | Python | Rust  |  JS   | Swift |  Go   |
| :----------------------------- | :------: | :---: | :---: | :----: | :---: | :---: | :---: | :---: |
| Substring Search               |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |
| Character Set Search           |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |
| Sorting & Sequence Operations  |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ⚪   |   ⚪   |   ⚪   |
| Streaming Hashes               |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ✅   |   ✅   |   ✅   |
| Small String Class             |    🧐     |   ✅   |   ✅   |   ❌    |   ⚪   |   ❌   |   ❌   |   ❌   |
| Lazy Ranges, Compressed Arrays |    🌳     |   ❌   |   ✅   |   ✅    |   ✅   |   ❌   |   ⚪   |   ⚪   |
|                                |          |       |       |        |       |       |       |       |
| Parallel Similarity Scoring    |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ⚪   |   ⚪   |   ⚪   |
| Parallel Rolling Fingerprints  |    🌳     |   ✅   |   ✅   |   ✅    |   ✅   |   ⚪   |   ⚪   |   ⚪   |

> 🌳 parts are used in production.
> 🧐 parts are in beta.
> 🚧 parts are under active development, and are likely to break in subsequent releases.
> ✅ are implemented.
> ⚪ are considered.
> ❌ are not intended.

## Quick Start: Python 🐍

Python bindings are available on PyPI for Python 3.8+, and can be installed with `pip`.

```bash
pip install stringzilla         # for serial algorithms
pip install stringzillas-cpus   # for parallel multi-CPU backends
pip install stringzillas-cuda   # for parallel Nvidia GPU backend
```

You can immediately check the installed version and the used hardware capabilities with following commands:

```bash
python -c "import stringzilla; print(stringzilla.__version__)"
python -c "import stringzillas; print(stringzillas.__version__)"
python -c "import stringzilla; print(stringzilla.__capabilities__)"     # for serial algorithms
python -c "import stringzillas; print(stringzillas.__capabilities__)"   # for parallel algorithms
```

### Basic Usage

If you've ever used the Python `str`, `bytes`, `bytearray`, or `memoryview` classes, you'll know what to expect.
StringZilla's `Str` class is a hybrid of the above, providing a `str`-like interface to byte arrays.

```python
from stringzilla import Str, File

text_from_str = Str('some-string') # no copies, just a view
text_from_bytes = Str(b'some-array') # no copies, just a view
text_from_file = Str(File('some-file.txt')) # memory-mapped file

import numpy as np
alphabet_array = np.arange(ord("a"), ord("z"), dtype=np.uint8)
text_from_array = Str(memoryview(alphabet_array))
```

The `File` class memory-maps a file from persistent storage without loading its copy into RAM.
The contents of that file would remain immutable, and the mapping can be shared by multiple Python processes simultaneously.
A standard dataset pre-processing use case would be to map a sizable textual dataset like Common Crawl into memory, spawn child processes, and split the job between them.

### Basic Operations

- Length: `len(text) -> int`
- Indexing: `text[42] -> str`
- Slicing: `text[42:46] -> Str`
- Substring check: `'substring' in text -> bool`
- Hashing: `hash(text) -> int`
- String conversion: `str(text) -> str`

### Advanced Operations

```py
import sys

x: bool = text.contains('substring', start=0, end=sys.maxsize)
x: int = text.find('substring', start=0, end=sys.maxsize)
x: int = text.count('substring', start=0, end=sys.maxsize, allowoverlap=False)
x: str = text.decode(encoding='utf-8', errors='strict')
x: Strs = text.split(separator=' ', maxsplit=sys.maxsize, keepseparator=False)
x: Strs = text.rsplit(separator=' ', maxsplit=sys.maxsize, keepseparator=False)
x: Strs = text.splitlines(keeplinebreaks=False, maxsplit=sys.maxsize)
```

It's important to note that the last function's behavior is slightly different from Python's `str.splitlines`.
The [native version][faq-splitlines] matches `\n`, `\r`, `\v` or `\x0b`, `\f` or `\x0c`, `\x1c`, `\x1d`, `\x1e`, `\x85`, `\r\n`, `\u2028`, `\u2029`, including 3x two-byte-long runes.
The StringZilla version matches only `\n`, `\v`, `\f`, `\r`, `\x1c`, `\x1d`, `\x1e`, `\x85`, avoiding two-byte-long runes.

[faq-splitlines]: https://docs.python.org/3/library/stdtypes.html#str.splitlines

### Character Set Operations

Python strings don't natively support character set operations.
This forces people to use regular expressions, which are slow and hard to read.
To avoid the need for `re.finditer`, StringZilla provides the following interfaces:

```py
x: int = text.find_first_of('chars', start=0, end=sys.maxsize)
x: int = text.find_last_of('chars', start=0, end=sys.maxsize)
x: int = text.find_first_not_of('chars', start=0, end=sys.maxsize)
x: int = text.find_last_not_of('chars', start=0, end=sys.maxsize)
x: Strs = text.split_byteset(separator='chars', maxsplit=sys.maxsize, keepseparator=False)
x: Strs = text.rsplit_byteset(separator='chars', maxsplit=sys.maxsize, keepseparator=False)
```

You can also transform the string using Look-Up Tables (LUTs), mapping it to a different character set.
This would result in a copy - `str` for `str` inputs and `bytes` for other types.

```py
x: str = text.translate('chars', {}, start=0, end=sys.maxsize, inplace=False)
x: bytes = text.translate(b'chars', {}, start=0, end=sys.maxsize, inplace=False)
```

For efficiency reasons, pass the LUT as a string or bytes object, not as a dictionary.
This can be useful in high-throughput applications dealing with binary data, including bioinformatics and image processing.
Here is an example:

```py
import stringzilla as sz
look_up_table = bytes(range(256)) # Identity LUT
image = open("/image/path.jpeg", "rb").read()
sz.translate(image, look_up_table, inplace=True)
```

### Hash

Single-shot and incremental hashing are both supported:

```py
import stringzilla as sz

# One-shot - stable 64-bit output across all platforms!
one = sz.hash(b"Hello, world!", seed=42)

# Incremental updates return itself; digest does not consume state
hasher = sz.Hasher(seed=42)
hasher.update(b"Hello, ").update(b"world!")
streamed = hasher.digest() # or `hexdigest()` for a string
assert one == streamed
```

### Collection-Level Operations

Once split into a `Strs` object, you can sort, shuffle, and reorganize the slices with minimal memory footprint.
If all the chunks are located in consecutive memory regions, the memory overhead can be as low as 4 bytes per chunk.

```python
lines: Strs = text.split(separator='\n') # 4 bytes per line overhead for under 4 GB of text
batch: Strs = lines.sample(seed=42) # 10x faster than `random.choices`
lines.shuffle(seed=42) # or shuffle all lines in place and shard with slices
lines_sorted: Strs = lines.sorted() # returns a new Strs in sorted order
order: tuple = lines.argsort() # similar to `numpy.argsort`
```

Working on [RedPajama][redpajama], addressing 20 billion annotated English documents, one will need only 160 GB of RAM instead of terabytes.
Once loaded, the data will be memory-mapped, and can be reused between multiple Python processes without copies.
And of course, you can use slices to navigate the dataset and shard it between multiple workers.

```python
lines[::3] # every third line
lines[1::1] # every odd line
lines[:-100:-1] # last 100 lines in reverse order
```

[redpajama]: https://github.com/togethercomputer/RedPajama-Data

### Iterators and Memory Efficiency

Python's operations like `split()` and `readlines()` immediately materialize a `list` of copied parts.
This can be very memory-inefficient for large datasets.
StringZilla saves a lot of memory by viewing existing memory regions as substrings, but even more memory can be saved by using lazily evaluated iterators.

```py
x: SplitIterator[Str] = text.split_iter(separator=' ', keepseparator=False)
x: SplitIterator[Str] = text.rsplit_iter(separator=' ', keepseparator=False)
x: SplitIterator[Str] = text.split_byteset_iter(separator='chars', keepseparator=False)
x: SplitIterator[Str] = text.rsplit_byteset_iter(separator='chars', keepseparator=False)
```

StringZilla can easily be 10x more memory efficient than native Python classes for tokenization.
With lazy operations, it practically becomes free.

```py
import stringzilla as sz
%load_ext memory_profiler

text = open("enwik9.txt", "r").read() # 1 GB, mean word length 7.73 bytes
%memit text.split() # increment: 8670.12 MiB (152 ms)
%memit sz.split(text) # increment: 530.75 MiB (25 ms)
%memit sum(1 for _ in sz.split_iter(text)) # increment: 0.00 MiB
```

### Low-Level Python API

Aside from calling the methods on the `Str` and `Strs` classes, you can also call the global functions directly on `str` and `bytes` instances.
Assuming StringZilla CPython bindings are implemented [without any intermediate tools like SWIG or PyBind](https://ashvardanian.com/posts/pybind11-cpython-tutorial/), the call latency should be similar to native classes.

```py
import stringzilla as sz

contains: bool = sz.contains("haystack", "needle", start=0, end=sys.maxsize)
offset: int = sz.find("haystack", "needle", start=0, end=sys.maxsize)
count: int = sz.count("haystack", "needle", start=0, end=sys.maxsize, allowoverlap=False)
```

### Similarity Scores

StringZilla exposes high-performance, batch-oriented similarity via the `stringzillas` module. 
Use `DeviceScope` to pick hardware and optionally limit capabilities per engine.

```py
import stringzilla as sz
import stringzillas as szs

cpu_scope = szs.DeviceScope(cpu_cores=4)    # force CPU-only
gpu_scope = szs.DeviceScope(gpu_device=0)   # pick GPU 0 if available

strings_a = sz.Strs(["kitten", "flaw"])
strings_b = sz.Strs(["sitting", "lawn"])

engine = szs.LevenshteinDistances(
    match=0, mismatch=2,        # costs don't have to be 1
    open=3, extend=1,           # may be different in Bio
    capabilities=("serial",)    # avoid SIMD 🤭
)
distances = engine(strings_a, strings_b, device=cpu_scope)
assert int(distances[0]) == 3 and int(distances[1]) == 2
```

Note, that this computes byte-level distances.
For UTF-8 codepoints, use a different engine class:

```py
strings_a = sz.Strs(["café", "αβγδ"])
strings_b = sz.Strs(["cafe", "αγδ"])
engine = szs.LevenshteinDistancesUTF8(capabilities=("serial",))
distances = engine(strings_a, strings_b, device=cpu_scope)
assert int(distances[0]) == 1 and int(distances[1]) == 1
```

For alignment scoring provide a 256×256 substitution matrix using NumPy:

```py
import numpy as np
import stringzilla as sz
import stringzillas as szs

substitution_matrix = np.zeros((256, 256), dtype=np.int8)
substitution_matrix.fill(-1)                # mismatch score
np.fill_diagonal(substitution_matrix, 0)    # match score

engine = szs.NeedlemanWunsch(substitution_matrix=substitution_matrix, open=1, extend=1)
scores = engine(strings_a, strings_b, device=cpu_scope)
```

Several Python libraries provide edit distance computation.
Most are implemented in C but may be slower than StringZilla on large inputs.
For proteins ~10k chars, 100 pairs:

- [JellyFish](https://github.com/jamesturk/jellyfish): 62.3s
- [EditDistance](https://github.com/roy-ht/editdistance): 32.9s
- StringZilla: __0.8s__

Using the same proteins for Needleman-Wunsch alignment scores:

- [BioPython](https://github.com/biopython/biopython): 25.8s
- StringZilla: __7.8s__

<details>
  <summary><b>§ Example converting from BioPython to StringZilla.</b></summary>

```py
import numpy as np
from Bio import Align
from Bio.Align import substitution_matrices

aligner = Align.PairwiseAligner()
aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
aligner.open_gap_score = 1
aligner.extend_gap_score = 1

# Convert the matrix to NumPy
subs_packed = np.array(aligner.substitution_matrix).astype(np.int8)
subs_reconstructed = np.zeros((256, 256), dtype=np.int8)

# Initialize all banned characters to a the largest possible penalty
subs_reconstructed.fill(127)
for packed_row, packed_row_aminoacid in enumerate(aligner.substitution_matrix.alphabet):
    for packed_column, packed_column_aminoacid in enumerate(aligner.substitution_matrix.alphabet):
        reconstructed_row = ord(packed_row_aminoacid)
        reconstructed_column = ord(packed_column_aminoacid)
        subs_reconstructed[reconstructed_row, reconstructed_column] = subs_packed[packed_row, packed_column]

# Let's pick two examples of tripeptides (made of 3 amino acids)
glutathione = "ECG" # Need to rebuild human tissue?
thyrotropin_releasing_hormone = "QHP" # Or to regulate your metabolism?

import stringzillas as szs
engine = szs.NeedlemanWunsch(substitution_matrix=subs_reconstructed, open=1, extend=1)
score = int(engine(sz.Strs([glutathione]), sz.Strs([thyrotropin_releasing_hormone]))[0])
assert score == aligner.score(glutathione, thyrotropin_releasing_hormone) # Equal to 6
```

</details>

### Rolling Fingerprints

MinHashing is a common technique for Information Retrieval, producing compact representations of large documents.
For $D$ hash-functions and a text of length $L$, in the worst case it involves computing $O(D \cdot L)$ hashes.

```py
import numpy as np
import stringzilla as sz
import stringzillas as szs

texts = sz.Strs([
    "quick brown fox jumps over the lazy dog",
    "quick brown fox jumped over a very lazy dog",
])

cpu = szs.DeviceScope(cpu_cores=4)
ndim = 1024
window_widths = np.array([4, 6, 8, 10], dtype=np.uint64)
engine = szs.Fingerprints(
    ndim=ndim,
    window_widths=window_widths,    # optional
    alphabet_size=256,              # default for byte strings
    capabilities=("serial",),       # defaults to all, can also pass a `DeviceScope`
)

hashes, counts = engine(texts, device=cpu)
assert hashes.shape == (len(texts), ndim)
assert counts.shape == (len(texts), ndim)
assert hashes.dtype == np.uint32 and counts.dtype == np.uint32
```

### Serialization

#### Filesystem

Similar to how `File` can be used to read a large file, other interfaces can be used to dump strings to disk faster.
The `Str` class has `write_to` to write the string to a file, and `offset_within` to obtain integer offsets of substring view in larger string for navigation.

```py
web_archive = Str("<html>...</html><html>...</html>")
_, end_tag, next_doc = web_archive.partition("</html>") # or use `find`
next_doc_offset = next_doc.offset_within(web_archive)
web_archive.write_to("next_doc.html") # no GIL, no copies, just a view
```

#### PyArrow

A `Str` is easy to cast to [PyArrow](https://arrow.apache.org/docs/python/arrays.html#string-and-binary-types) buffers.

```py
from pyarrow import foreign_buffer
from stringzilla import Strs

strs = Strs(["alpha", "beta", "gamma"])
arrow = foreign_buffer(strs.address, strs.nbytes, strs)
```

And only slightly harder to convert in reverse direction:

```py
arr = pa.Array.from_buffers(
    pa.large_string() if strs.offsets_are_large else pa.string(),
    len(strs),
    [None,
     pa.foreign_buffer(strs.offsets_address, strs.offsets_nbytes, strs),
     pa.foreign_buffer(strs.tape_address, strs.tape_nbytes, strs)],
)
```

That means you can convert `Str` to `pyarrow.Buffer` and `Strs` to `pyarrow.Array` without extra copies.

## Quick Start: C/C++ 🛠️

The C library is header-only, so you can just copy the `stringzilla.h` header into your project.
Same applies to C++, where you would copy the `stringzilla.hpp` header.
Alternatively, add it as a submodule, and include it in your build system.

```sh
git submodule add https://github.com/ashvardanian/StringZilla.git external/stringzilla
git submodule update --init --recursive
```

Or using a pure CMake approach:

```cmake
FetchContent_Declare(
    stringzilla
    GIT_REPOSITORY https://github.com/ashvardanian/StringZilla.git
    GIT_TAG main  # or specify a version tag
)
FetchContent_MakeAvailable(stringzilla)
```

Last, but not the least, you can also install it as a library, and link against it.
This approach is worse for inlining, but brings [dynamic runtime dispatch](#dynamic-dispatch) for the most advanced CPU features.

### Basic Usage with C 99 and Newer

There is a stable C 99 interface, where all function names are prefixed with `sz_`.
Most interfaces are well documented, and come with self-explanatory names and examples.
In some cases, hardware specific overloads are available, like `sz_find_skylake` or `sz_find_neon`.
Both are companions of the `sz_find`, first for x86 CPUs with AVX-512 support, and second for Arm NEON-capable CPUs.

```c
#include <stringzilla/stringzilla.h>

// Initialize your haystack and needle
sz_string_view_t haystack = {your_text, your_text_length};
sz_string_view_t needle = {your_subtext, your_subtext_length};

// Perform string-level operations auto-picking the backend or dispatching manually
sz_cptr_t ptr = sz_find(haystack.start, haystack.length, needle.start, needle.length);
sz_size_t substring_position = ptr ? (sz_size_t)(ptr - haystack.start) : SZ_SIZE_MAX; // SZ_SIZE_MAX if not found

// Backend-specific variants return pointers as well
sz_cptr_t ptr = sz_find_skylake(haystack.start, haystack.length, needle.start, needle.length);
sz_cptr_t ptr = sz_find_haswell(haystack.start, haystack.length, needle.start, needle.length);
sz_cptr_t ptr = sz_find_neon(haystack.start, haystack.length, needle.start, needle.length);

// Hash strings at once
sz_u64_t hash = sz_hash(haystack.start, haystack.length, 42);    // 42 is the seed
sz_u64_t checksum = sz_bytesum(haystack.start, haystack.length); // or accumulate byte values

// Hash strings incrementally with "init", "update", and "digest":
sz_hash_state_t state; 
sz_hash_state_init(&state, 42);
sz_hash_state_update(&state, haystack.start, 1);                        // first char
sz_hash_state_update(&state, haystack.start + 1, haystack.length - 1);  // rest of the string
sz_u64_t streamed_hash = sz_hash_state_digest(&state);

// Perform collection level operations
sz_sequence_t array = {your_handle, your_count, your_get_start, your_get_length};
sz_sequence_argsort(&array, &your_config);
```

<details>
  <summary><b>§ Mapping from LibC to StringZilla.</b></summary>

By design, StringZilla has a couple of notable differences from LibC:

1. all strings are expected to have a length, and are not necessarily null-terminated.
2. every operations has a reverse order counterpart.

That way `sz_find` and `sz_rfind` are similar to `strstr` and `strrstr` in LibC.
Similarly, `sz_find_byte` and `sz_rfind_byte` replace `memchr` and `memrchr`.
The `sz_find_byteset` maps to `strspn` and `strcspn`, while `sz_rfind_byteset` has no sibling in LibC.

<table>
    <tr>
        <th>LibC Functionality</th>
        <th>StringZilla Equivalents</th>
    </tr>
    <tr>
        <td><code>memchr(haystack, needle, haystack_length)</code>, <code>strchr</code></td>
        <td><code>sz_find_byte(haystack, haystack_length, needle)</code></td>
    </tr>
    <tr>
        <td><code>memrchr(haystack, needle, haystack_length)</code></td>
        <td><code>sz_rfind_byte(haystack, haystack_length, needle)</code></td>
    </tr>
    <tr>
        <td><code>memcmp</code>, <code>strcmp</code></td>
        <td><code>sz_order</code>, <code>sz_equal</code></td>
    </tr>
    <tr>
        <td><code>strlen(haystack)</code></td>
        <td><code>sz_find_byte(haystack, haystack_length, needle)</code></td>
    </tr>
    <tr>
        <td><code>strcspn(haystack, needles)</code></td>
        <td><code>sz_rfind_byteset(haystack, haystack_length, needles_bitset)</code></td>
    </tr>
    <tr>
        <td><code>strspn(haystack, needles)</code></td>
        <td><code>sz_find_byteset(haystack, haystack_length, needles_bitset)</code></td>
    </tr>
    <tr>
        <td><code>memmem(haystack, haystack_length, needle, needle_length)</code>, <code>strstr</code></td>
        <td><code>sz_find(haystack, haystack_length, needle, needle_length)</code></td>
    </tr>
    <tr>
        <td><code>memcpy(destination, source, destination_length)</code></td>
        <td><code>sz_copy(destination, source, destination_length)</code></td>
    </tr>
    <tr>
        <td><code>memmove(destination, source, destination_length)</code></td>
        <td><code>sz_move(destination, source, destination_length)</code></td>
    </tr>
    <tr>
        <td><code>memset(destination, value, destination_length)</code></td>
        <td><code>sz_fill(destination, destination_length, value)</code></td>
    </tr>
</table>

</details>

### Basic Usage with C++ 11 and Newer

There is a stable C++ 11 interface available in the `ashvardanian::stringzilla` namespace.
It comes with two STL-like classes: `string_view` and `string`.
The first is a non-owning view of a string, and the second is a mutable string with a [Small String Optimization][faq-sso].

```cpp
#include <stringzilla/stringzilla.hpp>

namespace sz = ashvardanian::stringzilla;

sz::string haystack = "some string";
sz::string_view needle = sz::string_view(haystack).substr(0, 4);

auto substring_position = haystack.find(needle); // Or `rfind`
auto hash = std::hash<sz::string_view>{}(haystack); // Compatible with STL's `std::hash`

haystack.end() - haystack.begin() == haystack.size(); // Or `rbegin`, `rend`
haystack.find_first_of(" \v\t") == 4; // Or `find_last_of`, `find_first_not_of`, `find_last_not_of`
haystack.starts_with(needle) == true; // Or `ends_with`
haystack.remove_prefix(needle.size()); // Why is this operation in-place?!
haystack.contains(needle) == true; // STL has this only from C++ 23 onwards
haystack.compare(needle) == 1; // Or `haystack <=> needle` in C++ 20 and beyond
```

StringZilla also provides string literals for automatic type resolution, [similar to STL][stl-literal]:

```cpp
using sz::literals::operator""_sv;
using std::literals::operator""sv;

auto a = "some string"; // char const *
auto b = "some string"sv; // std::string_view
auto b = "some string"_sv; // sz::string_view
```

[stl-literal]: https://en.cppreference.com/w/cpp/string/basic_string_view/operator%22%22sv

### Similarity Scores

StringZilla exposes high-performance, batch-oriented similarity via the `stringzillas/stringzillas.h` header. 
Use `szs_device_scope_t` to pick hardware and optionally limit capabilities per engine.

```cpp
#include <stringzillas/stringzillas.h>

szs_device_scope_t device = NULL;
szs_device_scope_init_default(&device);

szs_levenshtein_distances_t engine = NULL;
szs_levenshtein_distances_init(0, 1, 1, 1, /*alloc*/ NULL, /*caps*/ sz_cap_serial_k, &engine);

sz_sequence_u32tape_t strings_a {data_a, offsets_a, count}; // or `sz_sequence_u64tape_t` for large inputs
sz_sequence_u32tape_t strings_b {data_b, offsets_b, count}; // or `sz_sequence_t` to pass generic containers

sz_size_t distances[count];
szs_levenshtein_distances_u32tape(engine, device, &strings_a, &strings_b, distances, sizeof(distances[0]));

szs_levenshtein_distances_free(engine);
szs_device_scope_free(device);
```

To target a different device, use the appropriate `szs_device_scope_init_{cpu_cores,gpu_device}` function.
When dealing with GPU backends, make sure to use the "unified memory" allocators exposed as `szs_unified_{alloc,free}`.
Similar stable C ABIs are exposed for other workloads as well.

- UTF-8: `szs_levenshtein_distances_utf8_{sequence,u32tape,u64tape}`
- Needleman-Wunsch: `szs_needleman_wunsch_scores_{sequence,u32tape,u64tape}`
- Smith-Waterman: `szs_smith_waterman_scores_{sequence,u32tape,u64tape}`

Moreover, in C++ codebases one can tap into the raw templates implementing that functionality, customizing them with custom executors, SIMD plugins, etc.
For that include `stringzillas/similarities.hpp` for C++ and `stringzillas/similarities.cuh` for CUDA.

```cpp
#include <stringzillas/similarities.hpp>
#include <stringzilla/types.hpp>       // tape of strings
#include <fork_union.hpp>              // optional thread pool

namespace sz = ashvardanian::stringzilla;
namespace szs = ashvardanian::stringzillas;

// Pack strings into an Arrow-like tape
std::vector<std::string> left = {"kitten", "flaw"};
std::vector<std::string> right = {"sitting", "lawn"};
sz::arrow_strings_tape<char, sz::size_t, std::allocator<char>> tape_a, tape_b;
auto _ = tape_a.try_assign(left.begin(), left.end());
auto _ = tape_b.try_assign(right.begin(), right.end());

// Run on the current thread
using levenshtein_t = szs::levenshtein_distances<char, szs::linear_gap_costs_t, std::allocator<char>, sz_cap_serial_k>;
levenshtein_t engine {szs::uniform_substitution_costs_t{0,1}, szs::linear_gap_costs_t{1}};
std::size_t distances[2];
auto _ = engine(tape_a, tape_b, distances);

// Or run in parallel with a pool
fork_union::basic_pool_t pool;
auto _ = pool.try_spawn(std::thread::hardware_concurrency());
auto _ = engine(tape_a, tape_b, distances, pool);
```

All of the potentially failing StringZillas' interfaces return error codes, and none raise C++ exceptions.
Parallelism is enabled at both collection-level and within individual pairs of large inputs.

### Rolling Fingerprints

StringZilla exposes parallel fingerprinting (Min-Hashes or Count-Min-Sketches) via the `stringzillas/stringzillas.h` header. 
Use `szs_device_scope_t` to pick hardware and optionally limit capabilities per engine.

```c
#include <stringzillas/stringzillas.h>

szs_device_scope_t device = NULL;
szs_device_scope_init_default(&device);

szs_fingerprints_t engine = NULL;
sz_size_t const dims = 1024; sz_size_t const window_widths[] = {4, 6, 8, 10};
szs_fingerprints_init(dims, /*alphabet*/ 256, window_widths, 4, /*alloc*/ NULL, /*caps*/ sz_cap_serial_k, &engine);

sz_sequence_u32tape_t texts = {data, offsets, count};
sz_u32_t *min_hashes = (sz_u32_t*)szs_unified_alloc(count * dims * sizeof(*min_hashes));
sz_u32_t *min_counts = (sz_u32_t*)szs_unified_alloc(count * dims * sizeof(*min_counts));
szs_fingerprints_u32tape(engine, device, &texts,
    min_hashes, dims * sizeof(*min_hashes),     // support strided matrices
    min_counts, dims * sizeof(*min_counts));    // for both output arguments

szs_fingerprints_free(engine);
szs_device_scope_free(device);
```

Moreover, in C++ codebases one can tap into the raw templates implementing that functionality, customizing them with custom executors, SIMD plugins, etc.
For that include `stringzillas/fingerprints.hpp` for C++ and `stringzillas/fingerprints.cuh` for CUDA.

```cpp
#include <stringzillas/fingerprints.hpp>
#include <stringzilla/types.hpp>       // tape of strings
#include <fork_union.hpp>              // optional thread pool

namespace sz = ashvardanian::stringzilla;
namespace szs = ashvardanian::stringzillas;

// Pack strings into an Arrow-like tape
std::vector<std::string> docs = {"alpha beta", "alpha betta"};
sz::arrow_strings_tape<char, sz::size_t, std::allocator<char>> tape;
auto _ = tape.try_assign(docs.begin(), docs.end());

// Run on the current thread with a Rabin-Karp family hasher
constexpr std::size_t dimensions_k = 256;
constexpr std::size_t window_width_k = 7;
using row_t = std::array<sz_u32_t, 256>;
using fingerprinter_t = szs::floating_rolling_hashers<sz_cap_serial_k, dimensions_k>;
fingerprinter_t engine;
auto _ = engine.try_extend(window_width_k, dimensions_k);
std::vector<row_t> hashes(docs.size()), counts(docs.size());
auto _ = engine(tape, hashes, counts);

// Or run in parallel with a pool
fork_union::basic_pool_t pool;
auto _ = pool.try_spawn(std::thread::hardware_concurrency());
auto _ = engine(tape, hashes, counts, pool);
```

### CUDA

StringZilla provides CUDA C++ templates for composable string batch-processing operations.
Different GPUs have varying warp sizes, shared memory capacities, and register counts, affecting algorithm selection, so it's important to query the `gpu_specs_t` via `gpu_specs_fetch`.
For memory management, ensure that you use GPU-visible' unified memory` exposed in an STL-compatible manner as a `unified_alloc` template class.
For error handling, `cuda_status_t` extends the traditional `status_t` with GPU-specific information.
It's implicitly convertible to `status_t`, so you can use it in places expecting a `status_t`.

Most algorithms can load-balance both a large number of small strings and a small number of large strings.
Still, with large H100-scale GPUs, it's best to submit thousands of inputs at once.

### Memory Ownership and Small String Optimization

Most operations in StringZilla don't assume any memory ownership.
But in addition to the read-only search-like operations StringZilla provides a minimalistic C and C++ implementations for a memory owning string "class".
Like other efficient string implementations, it uses the [Small String Optimization][faq-sso] (SSO) to avoid heap allocations for short strings.

[faq-sso]: https://cpp-optimizations.netlify.app/small_strings/

```c
typedef union sz_string_t {
    struct internal {
        sz_ptr_t start;
        sz_u8_t length;
        char chars[SZ_STRING_INTERNAL_SPACE]; /// Ends with a null-terminator.
    } internal;

    struct external {
        sz_ptr_t start;
        sz_size_t length;        
        sz_size_t space; /// The length of the heap-allocated buffer.
        sz_size_t padding;
    } external;

} sz_string_t;
```

As one can see, a short string can be kept on the stack, if it fits within `internal.chars` array.
Before 2015 GCC string implementation was just 8 bytes, and could only fit 7 characters.
Different STL implementations today have different thresholds for the Small String Optimization.
Similar to GCC, StringZilla is 32 bytes in size, and similar to Clang it can fit 22 characters on stack.
Our layout might be preferential, if you want to avoid branches.
If you use a different compiler, you may want to check its SSO buffer size with a [simple Gist](https://gist.github.com/ashvardanian/c197f15732d9855c4e070797adf17b21).

|                       | `libstdc++` in  GCC 13 | `libc++` in Clang 17 | StringZilla |
| :-------------------- | ---------------------: | -------------------: | ----------: |
| `sizeof(std::string)` |                     32 |                   24 |          32 |
| Small String Capacity |                     15 |               __22__ |      __22__ |

This design has been since ported to many high-level programming languages.
Swift, for example, [can store 15 bytes](https://developer.apple.com/documentation/swift/substring/withutf8(_:)#discussion) in the `String` instance itself.
StringZilla implements SSO at the C level, providing the `sz_string_t` union and a simple API for primary operations.

```c
sz_memory_allocator_t allocator;
sz_string_t string;

// Init and make sure we are on stack
sz_string_init(&string);
sz_string_is_on_stack(&string); // == sz_true_k

// Optionally pre-allocate space on the heap for future insertions.
sz_string_grow(&string, 100, &allocator); // == sz_true_k

// Append, erase, insert into the string.
sz_string_expand(&string, 0, "_Hello_", 7, &allocator); // == sz_true_k
sz_string_expand(&string, SZ_SIZE_MAX, "world", 5, &allocator); // == sz_true_k
sz_string_erase(&string, 0, 1);

// Unpacking & introspection.
sz_ptr_t string_start;
sz_size_t string_length;
sz_size_t string_space;
sz_bool_t string_is_external;
sz_string_unpack(string, &string_start, &string_length, &string_space, &string_is_external);
sz_equal(string_start, "Hello_world", 11); // == sz_true_k

// Reclaim some memory.
sz_string_shrink_to_fit(&string, &allocator); // == sz_true_k
sz_string_free(&string, &allocator);
```

Unlike the conventional C strings, the `sz_string_t` is allowed to contain null characters.
To safely print those, pass the `string_length` to `printf` as well.

```c
printf("%.*s\n", (int)string_length, string_start);
```

### What's Wrong with the C Standard Library?

StringZilla is not a drop-in replacement for the C Standard Library.
It's designed to be a safer and more modern alternative.
Conceptually:

1. LibC strings are expected to be null-terminated, so to use the efficient LibC implementations on slices of larger strings, you'd have to copy them, which is more expensive than the original string operation.
2. LibC functionality is asymmetric - you can find the first and the last occurrence of a character within a string, but you can't find the last occurrence of a substring.
3. LibC function names are typically very short and cryptic.
4. LibC lacks crucial functionality like hashing and doesn't provide primitives for less critical but relevant operations like fuzzy matching.

Something has to be said about its support for UTF-8.
Aside from a single-byte `char` type, LibC provides `wchar_t`:

- The size of `wchar_t` is not consistent across platforms. On Windows, it's typically 16 bits (suitable for UTF-16), while on Unix-like systems, it's usually 32 bits (suitable for UTF-32). This inconsistency can lead to portability issues when writing cross-platform code.
- `wchar_t` is designed to represent wide characters in a fixed-width format (UTF-16 or UTF-32). In contrast, UTF-8 is a variable-length encoding, where each character can take from 1 to 4 bytes. This fundamental difference means that `wchar_t` and UTF-8 are incompatible.

StringZilla [partially addresses those issues](#unicode-utf-8-and-wide-characters).

### What's Wrong with the C++ Standard Library?

| C++ Code                             | Evaluation Result | Invoked Signature              |
| :----------------------------------- | :---------------- | :----------------------------- |
| `"Loose"s.replace(2, 2, "vath"s, 1)` | `"Loathe"` 🤢      | `(pos1, count1, str2, pos2)`   |
| `"Loose"s.replace(2, 2, "vath", 1)`  | `"Love"` 🥰        | `(pos1, count1, str2, count2)` |

StringZilla is designed to be a drop-in replacement for the C++ Standard Templates Library.
That said, some of the design decisions of STL strings are highly controversial, error-prone, and expensive.
Most notably:

1. Argument order for `replace`, `insert`, `erase` and similar functions is impossible to guess.
2. Bounds-checking exceptions for `substr`-like functions are only thrown for one side of the range.
3. Returning string copies in `substr`-like functions results in absurd volume of allocations.
4. Incremental construction via `push_back`-like functions goes through too many branches.
5. Inconsistency between `string` and `string_view` methods, like the lack of `remove_prefix` and `remove_suffix`.

Check the following set of asserts validating the `std::string` specification.
It's not realistic to expect the average developer to remember the [14 overloads of `std::string::replace`][stl-replace].

[stl-replace]: https://en.cppreference.com/w/cpp/string/basic_string/replace

```cpp
using str = std::string;

assert(str("hello world").substr(6) == "world");
assert(str("hello world").substr(6, 100) == "world"); // 106 is beyond the length of the string, but its OK
assert_throws(str("hello world").substr(100), std::out_of_range);   // 100 is beyond the length of the string
assert_throws(str("hello world").substr(20, 5), std::out_of_range); // 20 is beyond the length of the string
assert_throws(str("hello world").substr(-1, 5), std::out_of_range); // -1 casts to unsigned without any warnings...
assert(str("hello world").substr(0, -1) == "hello world");          // -1 casts to unsigned without any warnings...

assert(str("hello").replace(1, 2, "123") == "h123lo");
assert(str("hello").replace(1, 2, str("123"), 1) == "h23lo");
assert(str("hello").replace(1, 2, "123", 1) == "h1lo");
assert(str("hello").replace(1, 2, "123", 1, 1) == "h2lo");
assert(str("hello").replace(1, 2, str("123"), 1, 1) == "h2lo");
assert(str("hello").replace(1, 2, 3, 'a') == "haaalo");
assert(str("hello").replace(1, 2, {'a', 'b'}) == "hablo");
```

To avoid those issues, StringZilla provides an alternative consistent interface.
It supports signed arguments, and doesn't have more than 3 arguments per function or
The standard API and our alternative can be conditionally disabled with `SZ_SAFETY_OVER_COMPATIBILITY=1`.
When it's enabled, the _~~subjectively~~_ risky overloads from the Standard will be disabled.

```cpp
using str = sz::string;

str("a:b").front(1) == "a"; // no checks, unlike `substr`
str("a:b").front(2) == "2"; // take first 2 characters
str("a:b").back(-1) == "b"; // accepting negative indices
str("a:b").back(-2) == ":b"; // similar to Python's `"a:b"[-2:]`
str("a:b").sub(1, -1) == ":"; // similar to Python's `"a:b"[1:-1]`
str("a:b").sub(-2, -1) == ":"; // similar to Python's `"a:b"[-2:-1]`
str("a:b").sub(-2, 1) == ""; // similar to Python's `"a:b"[-2:1]`
"a:b"_sv[{-2, -1}] == ":"; // works on views and overloads `operator[]`
```

Assuming StringZilla is a header-only library you can use the full API in some translation units and gradually transition to safer restricted API in others.
Bonus - all the bound checking is branchless, so it has a constant cost and won't hurt your branch predictor.

### Beyond the C++ Standard Library - Learning from Python

Python is arguably the most popular programming language for data science.
In part, that's due to the simplicity of its standard interfaces.
StringZilla brings some of that functionality to C++.

- Content checks: `isalnum`, `isalpha`, `isascii`, `isdigit`, `islower`, `isspace`, `isupper`.
- Trimming character sets: `lstrip`, `rstrip`, `strip`.
- Trimming string matches: `remove_prefix`, `remove_suffix`.
- Ranges of search results: `splitlines`, `split`, `rsplit`.
- Number of non-overlapping substring matches: `count`.
- Partitioning: `partition`, `rpartition`.

For example, when parsing documents, it is often useful to split it into substrings.
Most often, after that, you would compute the length of the skipped part, the offset and the length of the remaining part.
This results in a lot of pointer arithmetic and is error-prone.
StringZilla provides a convenient `partition` function, which returns a tuple of three string views, making the code cleaner.

```cpp
auto parts = haystack.partition(':'); // Matching a character
auto [before, match, after] = haystack.partition(':'); // Structure unpacking
auto [before, match, after] = haystack.partition(sz::byteset(":;")); // Character-set argument
auto [before, match, after] = haystack.partition(" : "); // String argument
auto [before, match, after] = haystack.rpartition(sz::whitespaces_set()); // Split around the last whitespace
```

Combining those with the `split` function, one can easily parse a CSV file or HTTP headers.

```cpp
for (auto line : haystack.split("\r\n")) {
    auto [key, _, value] = line.partition(':');
    headers[key.strip()] = value.strip();
}
```

Some other extensions are not present in the Python standard library either.
Let's go through the C++ functionality category by category.

- [Splits and Ranges](#splits-and-ranges).
- [Concatenating Strings without Allocations](#concatenating-strings-without-allocations).
- [Random Generation](#random-generation).
- [Edit Distances and Fuzzy Search](#levenshtein-edit-distance-and-alignment-scores).

Some of the StringZilla interfaces are not available even Python's native `str` class.
Here is a sneak peek of the most useful ones.

```cpp
text.hash(); // -> 64 bit unsigned integer 
text.ssize(); // -> 64 bit signed length to avoid `static_cast<std::ssize_t>(text.size())`
text.contains_only(" \w\t"); // == text.find_first_not_of(sz::byteset(" \w\t")) == npos;
text.contains(sz::whitespaces_set()); // == text.find(sz::byteset(sz::whitespaces_set())) != npos;

// Simpler slicing than `substr`
text.front(10); // -> sz::string_view
text.back(10); // -> sz::string_view

// Safe variants, which clamp the range into the string bounds
using sz::string::cap;
text.front(10, cap) == text.front(std::min(10, text.size()));
text.back(10, cap) == text.back(std::min(10, text.size()));

// Character set filtering
text.lstrip(sz::whitespaces_set()).rstrip(sz::newlines_set()); // like Python
text.front(sz::whitespaces_set()); // all leading whitespaces
text.back(sz::digits_set()); // all numerical symbols forming the suffix

// Incremental construction
using sz::string::unchecked;
text.push_back('x'); // no surprises here
text.push_back('x', unchecked); // no bounds checking, Rust style
text.try_push_back('x'); // returns `false` if the string is full and the allocation failed

sz::concatenate(text, "@", domain, ".", tld); // No allocations
```

### Splits and Ranges

One of the most common use cases is to split a string into a collection of substrings.
Which would often result in [StackOverflow lookups][so-split] and snippets like the one below.

[so-split]: https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c

```cpp
std::vector<std::string> lines = split(haystack, "\r\n"); // string delimiter
std::vector<std::string> words = split(lines, ' '); // character delimiter
```

Those allocate memory for each string and the temporary vectors.
Each allocation can be orders of magnitude more expensive, than even serial `for`-loop over characters.
To avoid those, StringZilla provides lazily-evaluated ranges, compatible with the [Range-v3][range-v3] library.

[range-v3]: https://github.com/ericniebler/range-v3

```cpp
for (auto line : haystack.split("\r\n"))
    for (auto word : line.split(sz::byteset(" \w\t.,;:!?")))
        std::cout << word << std::endl;
```

Each of those is available in reverse order as well.
It also allows interleaving matches, if you want both inclusions of `xx` in `xxx`.
Debugging pointer offsets is not a pleasant exercise, so keep the following functions in mind.

- `haystack.[r]find_all(needle, interleaving)`
- `haystack.[r]find_all(sz::byteset(""))`
- `haystack.[r]split(needle)`
- `haystack.[r]split(sz::byteset(""))`

For $N$ matches the split functions will report $N+1$ matches, potentially including empty strings.
Ranges have a few convenience methods as well:

```cpp
range.size(); // -> std::size_t
range.empty(); // -> bool
range.template to<std::set<std::sting>>(); 
range.template to<std::vector<std::sting_view>>(); 
```

### Concatenating Strings without Allocations

Another common string operation is concatenation.
The STL provides `std::string::operator+` and `std::string::append`, but those are not very efficient, if multiple invocations are performed.

```cpp
std::string name, domain, tld;
auto email = name + "@" + domain + "." + tld; // 4 allocations
```

The efficient approach would be to pre-allocate the memory and copy the strings into it.

```cpp
std::string email;
email.reserve(name.size() + domain.size() + tld.size() + 2);
email.append(name), email.append("@"), email.append(domain), email.append("."), email.append(tld);
```

That's mouthful and error-prone.
StringZilla provides a more convenient `concatenate` function, which takes a variadic number of arguments.
It also overrides the `operator|` to concatenate strings lazily, without any allocations.

```cpp
auto email = sz::concatenate(name, "@", domain, ".", tld);   // 0 allocations
auto email = name | "@" | domain | "." | tld;                // 0 allocations
sz::string email = name | "@" | domain | "." | tld;          // 1 allocations
```

### Random Generation

Software developers often need to generate random strings for testing purposes.
The STL provides `std::generate` and `std::random_device`, that can be used with StringZilla.

```cpp
sz::string random_string(std::size_t length, char const *alphabet, std::size_t cardinality) {
    sz::string result(length, '\0');
    static std::random_device seed_source; // Expensive to construct - due to system calls
    static std::mt19937 generator(seed_source()); // Also expensive - due to the state size
    std::uniform_int_distribution<std::size_t> distribution(0, cardinality);
    std::generate(result.begin(), result.end(), [&]() { return alphabet[distribution(generator)]; });
    return result;
}
```

Mouthful and slow.
StringZilla provides a C native method - `sz_fill_random` and a convenient C++ wrapper - `sz::generate`.
Similar to Python it also defines the commonly used character sets.

```cpp
auto protein = sz::string::random(300, "ARNDCQEGHILKMFPSTWYV"); // static method
auto dna = sz::basic_string<custom_allocator>::random(3_000_000_000, "ACGT");

dna.fill_random("ACGT"); // `noexcept` pre-allocated version
dna.fill_random(&std::rand, "ACGT"); // pass any generator, like `std::mt19937`

char uuid[36];
sz::fill_random(sz::string_span(uuid, 36), "0123456789abcdef-"); // Overwrite any buffer
```

### Bulk Replacements

In text processing, it's often necessary to replace all occurrences of a specific substring or set of characters within a string.
Standard library functions may not offer the most efficient or convenient methods for performing bulk replacements, especially when dealing with large strings or performance-critical applications.

- `haystack.replace_all(needle_string, replacement_string)`
- `haystack.replace_all(sz::byteset(""), replacement_string)`
- `haystack.try_replace_all(needle_string, replacement_string)`
- `haystack.try_replace_all(sz::byteset(""), replacement_string)`
- `haystack.lookup(sz::look_up_table::identity())`
- `haystack.lookup(sz::look_up_table::identity(), haystack.data())`


### Sorting in C and C++

LibC provides `qsort` and STL provides `std::sort`.
Both have their quirks.
The LibC standard has no way to pass a context to the comparison function, that's only possible with platform-specific extensions.
Those have [different arguments order](https://stackoverflow.com/a/39561369) on every OS.

```c
// Linux: https://linux.die.net/man/3/qsort_r
void qsort_r(void *elements, size_t count, size_t element_width, 
    int (*compare)(void const *left, void const *right, void *context),
    void *context);
// macOS and FreeBSD: https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man3/qsort_r.3.html
void qsort_r(void *elements, size_t count, size_t element_width, 
    void *context,
    int (*compare)(void *context, void const *left, void const *right));
// Windows conflicts with ISO `qsort_s`: https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/qsort-s?view=msvc-170
void qsort_s(id *elements, size_t count, size_t element_width, 
    int (*compare)(void *context, void const *left, void const *right),
    void *context);
```

C++ generic algorithm is not perfect either.
There is no guarantee in the standard that `std::sort` won't allocate any memory.
If you are running on embedded, in real-time or on 100+ CPU cores per node, you may want to avoid that.
StringZilla doesn't solve the general case, but hopes to improve the performance for strings.
Use `sz_sequence_argsort`, or the high-level `sz::argsort`, which can be used sort any collection of elements convertible to `sz::string_view`.

```cpp
std::vector<std::string> data({"c", "b", "a"});
std::vector<std::size_t> order = sz::argsort(data); //< Simple shortcut

// Or, taking care of memory allocation:
sz::argsort(data.begin(), data.end(), order.data(), [](auto const &x) -> sz::string_view { return x; });
```

### Standard C++ Containers with String Keys

The C++ Standard Templates Library provides several associative containers, often used with string keys.

```cpp
std::map<std::string, int, std::less<std::string>> sorted_words;
std::unordered_map<std::string, int, std::hash<std::string>, std::equal_to<std::string>> words;
```

The performance of those containers is often limited by the performance of the string keys, especially on reads.
StringZilla can be used to accelerate containers with `std::string` keys, by overriding the default comparator and hash functions.

```cpp
std::map<std::string, int, sz::less> sorted_words;
std::unordered_map<std::string, int, sz::hash, sz::equal_to> words;
```

Alternatively, a better approach would be to use the `sz::string` class as a key.
The right hash function and comparator would be automatically selected and the performance gains would be more noticeable if the keys are short.

```cpp
std::map<sz::string, int> sorted_words;
std::unordered_map<sz::string, int> words;
```

### Compilation Settings and Debugging

__`SZ_DEBUG`__:

> For maximal performance, the C library does not perform any bounds checking in Release builds.
> In C++, bounds checking happens only in places where the STL `std::string` would do it.
> If you want to enable more aggressive bounds-checking, define `SZ_DEBUG` before including the header.
> If not explicitly set, it will be inferred from the build type.

__`SZ_USE_HASWELL`, `SZ_USE_SKYLAKE`, `SZ_USE_ICE`, `SZ_USE_NEON`, `SZ_USE_NEON_AES`, `SZ_USE_SVE`, `SZ_USE_SVE2`, `SZ_USE_SVE2_AES`__:

> One can explicitly disable certain families of SIMD instructions for compatibility purposes.
> Default values are inferred at compile time depending on compiler support (for dynamic dispatch) and the target architecture (for static dispatch).

__`SZ_USE_CUDA`, `SZ_USE_KEPLER`, `SZ_USE_HOPPER`__:

> One can explicitly disable certain families of PTX instructions for compatibility purposes.
> Default values are inferred at compile time depending on compiler support (for dynamic dispatch) and the target architecture (for static dispatch).

__`SZ_DYNAMIC_DISPATCH`__:

> By default, StringZilla is a header-only library.
> But if you are running on different generations of devices, it makes sense to pre-compile the library for all supported generations at once, and dispatch at runtime.
> This flag does just that and is used to produce the `stringzilla.so` shared library, as well as the Python bindings.

__`SZ_USE_MISALIGNED_LOADS`__:

> Default is platform-dependent: enabled on x86 (where unaligned accesses are fast), disabled on others by default.
> When enabled, many byte-level operations use word-sized loads, which can significantly accelerate the serial (SWAR) backend.
> Consider enabling it explicitly if you are targeting platforms that support fast unaligned loads.

__`SZ_AVOID_LIBC`__ and __`SZ_OVERRIDE_LIBC`__:

> When using the C header-only library one can disable the use of LibC.
> This may affect the type resolution system on obscure hardware platforms. 
> Moreover, one may let `stringzilla` override the common symbols like the `memcpy` and `memset` with its own implementations.
> In that case you can use the [`LD_PRELOAD` trick][ld-preload-trick] to prioritize its symbols over the ones from the LibC and accelerate existing string-heavy applications without recompiling them.
> It also adds a layer of security, as the `stringzilla` isn't [undefined for NULL inputs][redhat-memcpy-ub] like `memcpy(NULL, NULL, 0)`.

[ld-preload-trick]: https://ashvardanian.com/posts/ld-preload-libsee
[redhat-memcpy-ub]: https://developers.redhat.com/articles/2024/12/11/making-memcpynull-null-0-well-defined

__`SZ_AVOID_STL`__ and __`SZ_SAFETY_OVER_COMPATIBILITY`__:

> When using the C++ interface one can disable implicit conversions from `std::string` to `sz::string` and back.
> If not needed, the `<string>` and `<string_view>` headers will be excluded, reducing compilation time.
> Moreover, if STL compatibility is a low priority, one can make the API safer by disabling the overloads, which are subjectively error prone.

__`STRINGZILLA_BUILD_SHARED`, `STRINGZILLA_BUILD_TEST`, `STRINGZILLA_BUILD_BENCHMARK`, `STRINGZILLA_TARGET_ARCH`__ for CMake users:

> When compiling the tests and benchmarks, you can explicitly set the target hardware architecture.
> It's synonymous to GCC's `-march` flag and is used to enable/disable the appropriate instruction sets.
> You can also disable the shared library build, if you don't need it.

## Quick Start: Rust 🦀

StringZilla is available as a Rust crate, with documentation available on [docs.rs/stringzilla](https://docs.rs/stringzilla).
You can immediately check the installed version and the used hardware capabilities with following commands:

```bash
cargo add stringzilla
cargo run --example version
```

To use the latest crate release in your project, add the following to your `Cargo.toml`:

```toml
[dependencies]
stringzilla = ">=3"                                     # for serial algorithms
stringzilla = { version = ">=3", features = ["cpus"] }  # for parallel multi-CPU backends
stringzilla = { version = ">=3", features = ["cuda"] }  # for parallel Nvidia GPU backend
```

Or if you want to use the latest pre-release version from the repository:

```toml
[dependencies]
stringzilla = { git = "https://github.com/ashvardanian/stringzilla", branch = "main-dev" }
```

Once installed, all of the functionality is available through the `stringzilla` namespace.
Many interfaces will look familiar to the users of the `memchr` crate.

```rust
use stringzilla::sz;

// Identical to `memchr::memmem::find` and `memchr::memmem::rfind` functions
sz::find("Hello, world!", "world") // 7
sz::rfind("Hello, world!", "world") // 7

// Generalizations of `memchr::memrchr[123]`
sz::find_byte_from("Hello, world!", "world") // 2
sz::rfind_byte_from("Hello, world!", "world") // 11
```

Unlike `memchr`, the throughput of `stringzilla` is [high in both normal and reverse-order searches][memchr-benchmarks].
It also provides no constraints on the size of the character set, while `memchr` allows only 1, 2, or 3 characters.
In addition to global functions, `stringzilla` provides a `StringZilla` extension trait:

```rust
use stringzilla::StringZilla;

let my_string: String = String::from("Hello, world!");
let my_str = my_string.as_str();
let my_cow_str = Cow::from(&my_string);

// Use the generic function with a String
assert_eq!(my_string.sz_find("world"), Some(7));
assert_eq!(my_string.sz_rfind("world"), Some(7));
assert_eq!(my_string.sz_find_byte_from("world"), Some(2));
assert_eq!(my_string.sz_rfind_byte_from("world"), Some(11));
assert_eq!(my_string.sz_find_byte_not_from("world"), Some(0));
assert_eq!(my_string.sz_rfind_byte_not_from("world"), Some(12));

// Same works for &str and Cow<'_, str>
assert_eq!(my_str.sz_find("world"), Some(7));
assert_eq!(my_cow_str.as_ref().sz_find("world"), Some(7));
```

The library also exposes Levenshtein and Hamming edit distances for byte arrays and UTF-8 strings, as well as Needleman-Wunsch alignment scores.

```rust
use stringzilla::sz;

// Handling arbitrary byte arrays:
sz::levenshtein_distance("Hello, world!", "Hello, world?"); // 1
sz::hamming_distance("Hello, world!", "Hello, world?"); // 1
sz::alignment_score("Hello, world!", "Hello, world?", sz::unary_substitution_costs(), -1); // -1

// Handling UTF-8 strings:
sz::hamming_distance_utf8("αβγδ", "αγγδ") // 1
sz::levenshtein_distance_utf8("façade", "facade") // 1
```

[memchr-benchmarks]: https://github.com/ashvardanian/StringWa.rs

### Hash

Single-shot and incremental hashing are both supported:

```rs
let mut hasher = sz::Hasher::new(42);
hasher.write(b"Hello, ");
hasher.write(b"world!");
let streamed = hasher.finish();

let mut hasher = sz::Hasher::new(42);
hasher.write(b"Hello, world!");
assert_eq!(streamed, hasher.finish());
```

To use StringZilla with `std::collections`:

```rs
use std::collections::HashMap;
let mut map: HashMap<&str, i32, sz::BuildSzHasher> =
    HashMap::with_hasher(sz::BuildSzHasher::with_seed(42));
map.insert("a", 1);
assert_eq!(map.get("a"), Some(&1));
```

### Similarity Scores

StringZilla exposes high-performance, batch-oriented similarity via the `szs` module.
Use `DeviceScope` to pick hardware and optionally limit capabilities per engine.

```rust
use stringzilla::szs; // re-exported as `szs`

let cpu_scope = szs::DeviceScope::cpu_cores(4).unwrap();    // force CPU-only
let gpu_scope = szs::DeviceScope::gpu_device(0).unwrap();   // pick GPU 0 if available
let strings_a = vec!["kitten", "flaw"];
let strings_b = vec!["sitting", "lawn"];

let engine = szs::LevenshteinDistances::new(
    &cpu_scope,
    0,  // match cost
    2,  // mismatch cost - costs don't have to be 1
    3,  // open cost - may be different in Bio
    1,  // extend cost
).unwrap();
let distances = engine.compute(&cpu_scope, &strings_a, &strings_b).unwrap();
assert_eq!(distances[0], 3);
assert_eq!(distances[1], 2);
```

Note, that this computes byte-level distances.
For UTF-8 codepoints, use a different engine class:

```rust
let strings_a = vec!["café", "αβγδ"];
let strings_b = vec!["cafe", "αγδ"];
let engine = szs::LevenshteinDistancesUtf8::new(&cpu_scope, 0, 1, 1, 1).unwrap();
let distances = engine.compute(&cpu_scope, &strings_a, &strings_b).unwrap();
assert_eq!(distances, vec![1, 1]);
```

Similarly, for variable substitution costs, also pass in a a weights matrix:

```rust
let mut substitution_matrix = [-1i8; 256 * 256];
for i in 0..256 { substitution_matrix[i * 256 + i] = 0; }
let engine = szs::NeedlemanWunschScores::new(&cpu_scope, &substitution_matrix, -3, -1).unwrap();
let scores = engine.compute(&cpu_scope, &strings_a, &strings_b).unwrap();
```

Or for local alignment scores:

```rust
let engine = szs::SmithWatermanScores::new(&cpu_scope, &substitution_matrix, -3, -1).unwrap();
let local_scores = engine.compute(&cpu_scope, &strings_a, &strings_b).unwrap();
```

### Rolling Fingerprints

MinHashing is a common technique for Information Retrieval, producing compact representations of large documents.
For $D$ hash-functions and a text of length $L$, in the worst case it involves computing $O(D \cdot L)$ hashes.

```rust
use stringzilla::szs;

let texts = vec![
    "quick brown fox jumps over the lazy dog",
    "quick brown fox jumped over a very lazy dog",
];
let cpu = szs::DeviceScope::cpu_cores(4).unwrap();
let ndim = 1024;
let window_widths = vec![4u64, 6, 8, 10];

let engine = szs::Fingerprints::new(
    ndim,           // number of hash functions & dimensions
    &window_widths, // optional predefined window widths
    256,            // default alphabet size for byte strings
    &cpu            // device scope
).unwrap();

let (hashes, counts) = engine.compute(&cpu, &texts).unwrap();
assert_eq!(hashes.len(), texts.len() * ndim);
assert_eq!(counts.len(), texts.len() * ndim);
```

## Quick Start: JavaScript 🟨

Install the Node.js package and use zero-copy `Buffer` APIs.

```bash
npm install stringzilla
node -p "require('stringzilla').capabilities" # for CommonJS
node -e "import('stringzilla').then(m=>console.log(m.default.capabilities)).catch(console.error)" # for ESM
```

```js
import sz from 'stringzilla';

const haystack = Buffer.from('Hello, world!');
const needle = Buffer.from('world');

// Substring search (BigInt offsets)
const firstIndex = sz.find(haystack, needle);      // 7n
const lastIndex = sz.findLast(haystack, needle);   // 7n

// Character / charset search
const firstOIndex = sz.findByte(haystack, 'o'.charCodeAt(0));                 // 4n
const firstVowelIndex = sz.findByteFrom(haystack, Buffer.from('aeiou'));      // 1n
const lastVowelIndex = sz.findLastByteFrom(haystack, Buffer.from('aeiou'));   // 8n

// Counting (optionally overlapping)
const lCount = sz.count(haystack, Buffer.from('l'));                // 3n
const llOverlapCount = sz.count(haystack, Buffer.from('ll'), true); // 1n

// Equality/ordering utilities
const isEqual = sz.equal(Buffer.from('a'), Buffer.from('a'));
const order = sz.compare(Buffer.from('a'), Buffer.from('b')); // -1, 0, or 1

// Other helpers
const byteSum = sz.byteSum(haystack); // sum of bytes as BigInt
```

### Hash

Single-shot and incremental hashing are both supported:

```js
import sz from 'stringzilla';

// One-shot - stable 64-bit output across all platforms!
const hash = sz.hash(Buffer.from('Hello, world!'), 42); // returns BigInt

// Incremental updates - hasher maintains state
const hasher = new sz.Hasher(42); // seed: 42
hasher.update(Buffer.from('Hello, '));
hasher.update(Buffer.from('world!'));
const streamedHash = hasher.digest(); // returns BigInt
console.assert(hash === streamedHash);
```

## Quick Start: Swift 🍏

StringZilla can be added as a dependency in the Swift Package Manager.
In your `Package.swift` file, add the following:

```swift
dependencies: [
    .package(url: "https://github.com/ashvardanian/stringzilla")
]
```

The package currently covers only the most basic functionality, but is planned to be extended to cover the full C++ API.

```swift
var s = "Hello, world! Welcome to StringZilla. 👋"
s[s.findFirst(substring: "world")!...] // "world! Welcome to StringZilla. 👋"
s[s.findLast(substring: "o")!...] // "o StringZilla. 👋"
s[s.findFirst(characterFrom: "aeiou")!...] // "ello, world! Welcome to StringZilla. 👋"
s[s.findLast(characterFrom: "aeiou")!...] // "a. 👋")
s[s.findFirst(characterNotFrom: "aeiou")!...] // "Hello, world! Welcome to StringZilla. 👋"
```

### Hash

StringZilla provides high-performance hashing for Swift strings:

```swift
import StringZilla

// One-shot hashing - stable 64-bit output across all platforms!
let hash = "Hello, world!".hash(seed: 42)

// Incremental hashing for streaming data
var hasher = SZHasher(seed: 42)
hasher.update("Hello, ")
hasher.update("world!")
let streamedHash = hasher.digest()
assert(hash == streamedHash)
```

## Quick Start: GoLang 🦫

Add the Go binding as a module dependency:

```bash
go get github.com/ashvardanian/stringzilla/golang@latest
```

Build the shared C library once, then ensure your runtime can locate it (Linux shown):

```bash
cmake -B build_shared -D STRINGZILLA_BUILD_SHARED=1 -D CMAKE_BUILD_TYPE=Release
cmake --build build_shared --target stringzilla_shared --config Release
export LD_LIBRARY_PATH="$PWD/build_shared:$LD_LIBRARY_PATH"
```

Use finders (substring, bytes, and sets):

```go
package main

import (
    "fmt"
    sz "github.com/ashvardanian/stringzilla/golang"
)

func main() {
    s := "the quick brown fox jumps over the lazy dog"

    // Substrings
    fmt.Println(sz.Contains(s, "brown"))        // true
    fmt.Println(sz.Index(s, "the"))             // 0
    fmt.Println(sz.LastIndex(s, "the"))         // 35

    // Single bytes
    fmt.Println(sz.IndexByte(s, 'o'))            // 12
    fmt.Println(sz.LastIndexByte(s, 'o'))        // 41

    // Byte sets
    fmt.Println(sz.IndexAny(s, "aeiou"))        // 2  (first vowel)
    fmt.Println(sz.LastIndexAny(s, "aeiou"))    // 43 (last vowel)

    // Counting with/without overlaps
    fmt.Println(sz.Count("aaaaa", "aa", false)) // 2
    fmt.Println(sz.Count("aaaaa", "aa", true))  // 4
    fmt.Println(sz.Count("abc", "", false))     // 4
    fmt.Println(sz.Bytesum("ABC"), sz.Bytesum("ABCD"))
}
```

### Hash

Single-shot and incremental hashing are both supported:

```go
one := sz.Hash("Hello, world!", 42)

hasher := sz.NewHasher(42)
hasher.Write([]byte("Hello, "))
hasher.Write([]byte("world!"))
streamed := hasher.Digest()
fmt.Println(one == streamed) // true
```

## Algorithms & Design Decisions 📚

StringZilla aims to optimize some of the slowest string operations.
Some popular operations, however, like equality comparisons and relative order checking, almost always complete on some of the very first bytes in either string.
In such operations vectorization is almost useless, unless huge and very similar strings are considered.
StringZilla implements those operations as well, but won't result in substantial speedups.

### Exact Substring Search

Substring search algorithms are generally divided into: comparison-based, automaton-based, and bit-parallel.
Different families are effective for different alphabet sizes and needle lengths.
The more operations are needed per-character - the more effective SIMD would be.
The longer the needle - the more effective the skip-tables are.
StringZilla uses different exact substring search algorithms for different needle lengths and backends:

- When no SIMD is available - SWAR (SIMD Within A Register) algorithms are used on 64-bit words.
- Boyer-Moore-Horspool (BMH) algorithm with Raita heuristic variation for longer needles.
- SIMD algorithms are randomized to look at different parts of the needle.

On very short needles, especially 1-4 characters long, brute force with SIMD is the fastest solution.
On mid-length needles, bit-parallel algorithms are effective, as the character masks fit into 32-bit or 64-bit words.
Either way, if the needle is under 64-bytes long, on haystack traversal we will still fetch every CPU cache line.
So the only way to improve performance is to reduce the number of comparisons.
The snippet below shows how StringZilla accomplishes that for needles of length two.

https://github.com/ashvardanian/StringZilla/blob/266c01710dddf71fc44800f36c2f992ca9735f87/include/stringzilla/stringzilla.h#L1585-L1637

Going beyond that, to long needles, Boyer-Moore (BM) and its variants are often the best choice.
It has two tables: the good-suffix shift and the bad-character shift.
Common choice is to use the simplified BMH algorithm, which only uses the bad-character shift table, reducing the pre-processing time.
We do the same for mid-length needles up to 256 bytes long.
That way the stack-allocated shift table remains small.

https://github.com/ashvardanian/StringZilla/blob/46e957cd4f9ecd4945318dd3c48783dd11323f37/include/stringzilla/stringzilla.h#L1774-L1825

In the C++ Standards Library, the `std::string::find` function uses the BMH algorithm with Raita's heuristic.
Before comparing the entire string, it matches the first, last, and the middle character.
Very practical, but can be slow for repetitive characters.
Both SWAR and SIMD backends of StringZilla have a cheap pre-processing step, where we locate unique characters.
This makes the library a lot more practical when dealing with non-English corpora.

https://github.com/ashvardanian/StringZilla/blob/46e957cd4f9ecd4945318dd3c48783dd11323f37/include/stringzilla/stringzilla.h#L1398-L1431

All those, still, have $O(hn)$ worst case complexity.
To guarantee $O(h)$ worst case time complexity, the Apostolico-Giancarlo (AG) algorithm adds an additional skip-table.
Preprocessing phase is $O(n+sigma)$ in time and space.
On traversal, performs from $(h/n)$ to $(3h/2)$ comparisons.
It however, isn't practical on modern CPUs.
A simpler idea, the Galil-rule might be a more relevant optimizations, if many matches must be found.

Other algorithms previously considered and deprecated:

- Apostolico-Giancarlo algorithm for longer needles. _Control-flow is too complex for efficient vectorization._
- Shift-Or-based Bitap algorithm for short needles. _Slower than SWAR._
- Horspool-style bad-character check in SIMD backends. _Effective only for very long needles, and very uneven character distributions between the needle and the haystack. Faster "character-in-set" check needed to generalize._

> § Reading materials.
> [Exact String Matching Algorithms in Java](https://www-igm.univ-mlv.fr/~lecroq/string).
> [SIMD-friendly algorithms for substring searching](http://0x80.pl/articles/simd-strfind.html).

### Exact Multiple Substring Search

Few algorithms for multiple substring search are known.
Most are based on the Aho-Corasick automaton, which is a generalization of the KMP algorithm.
The naive implementation, however:

- Allocates disjoint memory for each Trie node and Automaton state.
- Requires a lot of pointer chasing, limiting speculative execution.
- Has a lot of branches and conditional moves, which are hard to predict.
- Matches text a character at a time, which is slow on modern CPUs.

There are several ways to improve the original algorithm.
One is to use sparse DFA representation, which is more cache-friendly, but would require extra processing to navigate state transitions.

### Levenshtein Edit Distance

Levenshtein distance is the best known edit-distance for strings, that checks, how many insertions, deletions, and substitutions are needed to transform one string to another.
It's extensively used in approximate string-matching, spell-checking, and bioinformatics.

The computational cost of the Levenshtein distance is $O(n * m)$, where $n$ and $m$ are the lengths of the string arguments.
To compute that, the naive approach requires $O(n * m)$ space to store the "Levenshtein matrix", the bottom-right corner of which will contain the Levenshtein distance.
The algorithm producing the matrix has been simultaneously studied/discovered by the Soviet mathematicians Vladimir Levenshtein in 1965, Taras Vintsyuk in 1968, and American computer scientists - Robert Wagner, David Sankoff, Michael J. Fischer in the following years.
Several optimizations are known:

1. __Space Optimization__: The matrix can be computed in $O(min(n,m))$ space, by only storing the last two rows of the matrix.
2. __Divide and Conquer__: Hirschberg's algorithm can be applied to decompose the computation into subtasks.
3. __Automata__: Levenshtein automata can be effective, if one of the strings doesn't change, and is a subject to many comparisons.
4. __Shift-Or__: Bit-parallel algorithms transpose the matrix into a bit-matrix, and perform bitwise operations on it.

The last approach is quite powerful and performant, and is used by the great [RapidFuzz][rapidfuzz] library.
It's less known, than the others, derived from the Baeza-Yates-Gonnet algorithm, extended to bounded edit-distance search by Manber and Wu in 1990s, and further extended by Gene Myers in 1999 and Heikki Hyyro between 2002 and 2004.

StringZilla introduces a different approach, extensively used in Unum's internal combinatorial optimization libraries.
The approach doesn't change the number of trivial operations, but performs them in a different order, removing the data dependency, that occurs when computing the insertion costs.
This results in much better vectorization for intra-core parallelism and potentially multi-core evaluation of a single request.
Moreover, it's easy to generalize to weighted edit-distances, where the cost of a substitution between two characters may not be the same for all pairs, often used in bioinformatics.

Next design goals:

- [x] Generalize fast traversals to non-square matrices.
- [ ] Port x86 AVX-512 solution to Arm NEON.

> § Reading materials.
> [Faster Levenshtein Distances with a SIMD-friendly Traversal Order](https://ashvardanian.com/posts/levenshtein-diagonal).

[rapidfuzz]: https://github.com/rapidfuzz/RapidFuzz

### Needleman-Wunsch Alignment Score for Bioinformatics

The field of bioinformatics studies various representations of biological structures.
The "primary" representations are generally strings over sparse alphabets:

- [DNA][faq-dna] sequences, where the alphabet is {A, C, G, T}, ranging from ~100 characters for short reads to 3 billion for the human genome.
- [RNA][faq-rna] sequences, where the alphabet is {A, C, G, U}, ranging from ~50 characters for tRNA to thousands for mRNA.
- [Proteins][faq-protein], where the alphabet is made of 22 amino acids, ranging from 2 characters for [dipeptide][faq-dipeptide] to 35,000 for [Titin][faq-titin], the longest protein.

The shorter the representation, the more often researchers may want to use custom substitution matrices.
Meaning that the cost of a substitution between two characters may not be the same for all pairs.

StringZilla adapts the fairly efficient two-row Wagner-Fisher algorithm as a baseline serial implementation of the Needleman-Wunsch score.
It supports arbitrary alphabets up to 256 characters, and can be used with either [BLOSUM][faq-blosum], [PAM][faq-pam], or other substitution matrices.
It also uses SIMD for hardware acceleration of the substitution lookups.
This however, does not __yet__ break the data-dependency for insertion costs, where 80% of the time is wasted.
With that solved, the SIMD implementation will become 5x faster than the serial one.

[faq-dna]: https://en.wikipedia.org/wiki/DNA
[faq-rna]: https://en.wikipedia.org/wiki/RNA
[faq-protein]: https://en.wikipedia.org/wiki/Protein
[faq-blosum]: https://en.wikipedia.org/wiki/BLOSUM
[faq-pam]: https://en.wikipedia.org/wiki/Point_accepted_mutation
[faq-dipeptide]: https://en.wikipedia.org/wiki/Dipeptide
[faq-titin]: https://en.wikipedia.org/wiki/Titin

Next design goals:

- [ ] Needleman-Wunsch Automata

### Memory Copying, Fills, and Moves

A lot has been written about the time computers spend copying memory and how that operation is implemented in LibC.
Interestingly, the operation can still be improved, as most Assembly implementations use outdated instructions.
Even performance-oriented STL replacements, like Meta's [Folly v2024.09.23 focus on AVX2](https://github.com/facebook/folly/blob/main/folly/memset.S), and don't take advantage of the new masked instructions in AVX-512 or SVE.

In AVX-512, StringZilla uses non-temporal stores to avoid cache pollution, when dealing with very large strings.
Moreover, it handles the unaligned head and the tails of the `target` buffer separately, ensuring that writes in big copies are always aligned to cache-line boundaries.
That's true for both AVX2 and AVX-512 backends.

StringZilla also contains "drafts" of smarter, but less efficient algorithms, that minimize the number of unaligned loads, performing shuffles and permutations.
That's a topic for future research, as the performance gains are not yet satisfactory.

> § Reading materials.
> [`memset` benchmarks](https://github.com/nadavrot/memset_benchmark?tab=readme-ov-file) by Nadav Rotem.
> [Cache Associativity](https://en.algorithmica.org/hpc/cpu-cache/associativity/) by Sergey Slotin.

### Random Generation

Generating random strings from different alphabets is a very common operation.
StringZilla accepts an arbitrary [Pseudorandom Number Generator][faq-prng] to produce noise, and an array of characters to sample from.
Sampling is optimized to avoid integer division, a costly operation on modern CPUs.
For that a 768-byte long lookup table is used to perform 2 lookups, 1 multiplication, 2 shifts, and 2 accumulations.

https://github.com/ashvardanian/StringZilla/blob/266c01710dddf71fc44800f36c2f992ca9735f87/include/stringzilla/stringzilla.h#L2490-L2533

[faq-prng]: https://en.wikipedia.org/wiki/Pseudorandom_number_generator

### Sorting

For lexicographic sorting of strings, StringZilla uses a "hybrid-hybrid" approach with $O(n * log(n))$ and.

1. Radix sort for first bytes exported into a continuous buffer for locality.
2. IntroSort on partially ordered chunks to balance efficiency and worst-case performance.
   1. IntroSort begins with a QuickSort.
   2. If the recursion depth exceeds a certain threshold, it switches to a HeapSort.

A better algorithm is in development.
Check #173 for design goals and progress updates.

### Hashing

> [!WARNING]
> Hash functions are not cryptographically safe and are currently under active development.
> They may change in future __minor__ releases.

Choosing the right hashing algorithm for your application can be crucial from both performance and security standpoint.
In StringZilla a 64-bit rolling hash function is reused for both string hashes and substring hashes, Rabin-style fingerprints.
Rolling hashes take the same amount of time to compute hashes with different window sizes, and are fast to update.
Those are not however perfect hashes, and collisions are frequent.
StringZilla attempts to use SIMD, but the performance is not __yet__ satisfactory.
On Intel Sapphire Rapids, the following numbers can be expected for N-way parallel variants.

- 4-way AVX2 throughput with 64-bit integer multiplication (no native support): 0.28 GB/s.
- 4-way AVX2 throughput with 32-bit integer multiplication: 0.54 GB/s.
- 4-way AVX-512DQ throughput with 64-bit integer multiplication: 0.46 GB/s.
- 4-way AVX-512 throughput with 32-bit integer multiplication: 0.58 GB/s.
- 8-way AVX-512 throughput with 32-bit integer multiplication: 0.11 GB/s.

Next design goals:

- [ ] Try gear-hash and other rolling approaches.

#### Why not CRC32?

Cyclic Redundancy Check 32 is one of the most commonly used hash functions in Computer Science.
It has in-hardware support on both x86 and Arm, for both 8-bit, 16-bit, 32-bit, and 64-bit words.
The `0x1EDC6F41` polynomial is used in iSCSI, Btrfs, ext4, and the `0x04C11DB7` in SATA, Ethernet, Zlib, PNG.
In case of Arm more than one polynomial is supported.
It is, however, somewhat limiting for Big Data usecases, which often have to deal with more than 4 Billion strings, making collisions unavoidable.
Moreover, the existing SIMD approaches are tricky, combining general purpose computations with specialized instructions, to utilize more silicon in every cycle.

> § Reading materials.
> [Comprehensive derivation of approaches](https://github.com/komrad36/CRC)
> [Faster computation for 4 KB buffers on x86](https://www.corsix.org/content/fast-crc32c-4k)
> [Comparing different lookup tables](https://create.stephan-brumme.com/crc32)
> Great open-source implementations.
> [By Peter Cawley](https://github.com/corsix/fast-crc32)
> [By Stephan Brumme](https://github.com/stbrumme/crc32)

#### Other Modern Alternatives

[MurmurHash](https://github.com/aappleby/smhasher/blob/master/README.md) from 2008 by Austin Appleby is one of the best known non-cryptographic hashes.
It has a very short implementation and is capable of producing 32-bit and 128-bit hashes.
The [CityHash](https://opensource.googleblog.com/2011/04/introducing-cityhash) from 2011 by Google and the [xxHash](https://github.com/Cyan4973/xxHash) improve on that, better leveraging the super-scalar nature of modern CPUs and producing 64-bit and 128-bit hashes.

Neither of those functions are cryptographic, unlike MD5, SHA, and BLAKE algorithms.
Most of cryptographic hashes are based on the Merkle-Damgård construction, and aren't resistant to the length-extension attacks.
Current state of the Art, might be the [BLAKE3](https://github.com/BLAKE3-team/BLAKE3) algorithm.
It's resistant to a broad range of attacks, can process 2 bytes per CPU cycle, and comes with a very optimized official implementation for C and Rust.
It has the same 128-bit security level as the BLAKE2, and achieves its performance gains by reducing the number of mixing rounds, and processing data in 1 KiB chunks, which is great for longer strings, but may result in poor performance on short ones.

All mentioned libraries have undergone extensive testing and are considered production-ready.
They can definitely accelerate your application, but so may the downstream mixer.
For instance, when a hash-table is constructed, the hashes are further shrunk to address table buckets.
If the mixer loses entropy, the performance gains from the hash function may be lost.
An example would be power-of-two modulo, which is a common mixer, but is known to be weak.
One alternative would be the [fastrange](https://github.com/lemire/fastrange) by Daniel Lemire.
Another one is the [Fibonacci hash trick](https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/) using the Golden Ratio, also used in StringZilla.

### Unicode, UTF-8, and Wide Characters

Most StringZilla operations are byte-level, so they work well with ASCII and UTF-8 content out of the box.
In some cases, like edit-distance computation, the result of byte-level evaluation and character-level evaluation may differ.
So StringZilla provides following functions to work with Unicode:

- `szs_levenshtein_distance_utf8` - computes the Levenshtein distance between two UTF-8 strings.
- `sz_hamming_distance_utf8` - computes the Hamming distance between two UTF-8 strings.

Java, JavaScript, Python 2, C#, and Objective-C, however, use wide characters (`wchar`) - two byte long codes, instead of the more reasonable fixed-length UTF-32 or variable-length UTF-8.
This leads [to all kinds of offset-counting issues][wide-char-offsets] when facing four-byte long Unicode characters.
So consider transcoding with [simdutf](https://github.com/simdutf/simdutf), if you are coming from such environments.

[wide-char-offsets]: https://josephg.com/blog/string-length-lies/

## Dynamic Dispatch

Due to the high-level of fragmentation of SIMD support in different CPUs, StringZilla uses the names of select Intel and ARM CPU generations for its backends.
You can query supported backends and use them manually.
Use it to guarantee constant performance, or to explore how different algorithms scale on your hardware.

```c
sz_find(text, length, pattern, 3);          // Auto-dispatch
sz_find_haswell(text, length, pattern, 3);  // Intel Haswell+ AVX2
sz_find_skylake(text, length, pattern, 3);  // Intel Skylake+ AVX-512
sz_find_neon(text, length, pattern, 3);     // Arm NEON 128-bit
sz_find_sve(text, length, pattern, 3);      // Arm SVE 128/256/512/1024/2048-bit
```

StringZilla automatically picks the most advanced backend for the given CPU.
Similarly, in Python, you can log the auto-detected capabilities:

```python
python -c "import stringzilla; print(stringzilla.__capabilities__)"         # ('serial', 'haswell', 'skylake', 'ice', 'neon', 'sve', 'sve2+aes')
python -c "import stringzilla; print(stringzilla.__capabilities_str__)"     # "haswell, skylake, ice, neon, sve, sve2+aes"
```

You can also explicitly set the backend to use, or scope the backend to a specific function.

```python
import stringzilla as sz
sz.reset_capabilities(('serial',))          # Force SWAR backend
sz.reset_capabilities(('haswell',))         # Force AVX2 backend
sz.reset_capabilities(('neon',))            # Force NEON backend
sz.reset_capabilities(sz.__capabilities__)  # Reset to auto-dispatch
```

## Contributing 👾

Please check out the [contributing guide](https://github.com/ashvardanian/StringZilla/blob/main/CONTRIBUTING.md) for more details on how to set up the development environment and contribute to this project.
If you like this project, you may also enjoy [USearch][usearch], [UCall][ucall], [UForm][uform], and [SimSIMD][simsimd]. 🤗

[usearch]: https://github.com/unum-cloud/usearch
[ucall]: https://github.com/unum-cloud/ucall
[uform]: https://github.com/unum-cloud/uform
[simsimd]: https://github.com/ashvardanian/simsimd

If you like strings and value efficiency, you may also enjoy the following projects:

- [simdutf](https://github.com/simdutf/simdutf) - transcoding UTF-8, UTF-16, and UTF-32 LE and BE.
- [hyperscan](https://github.com/intel/hyperscan) - regular expressions with SIMD acceleration.
- [pyahocorasick](https://github.com/WojciechMula/pyahocorasick) - Aho-Corasick algorithm in Python.
- [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) - fast string matching in C++ and Python.

If you are looking for more reading materials on this topic, consider the following:

- [5x faster strings with SIMD & SWAR](https://ashvardanian.com/posts/stringzilla/).
- [The Painful Pitfalls of C++ STL Strings](https://ashvardanian.com/posts/painful-strings/).

## License 📜

Feel free to use the project under Apache 2.0 or the Three-clause BSD license at your preference.
