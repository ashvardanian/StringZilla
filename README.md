# StringZilla 🦖

[![StringZilla Python installs](https://static.pepy.tech/personalized-badge/stringzilla?period=total&units=abbreviation&left_color=black&right_color=blue&left_text=StringZilla%20Python%20installs)](https://github.com/ashvardanian/stringzilla)
[![StringZilla Rust installs](https://img.shields.io/crates/d/stringzilla?logo=rust")](https://crates.io/crates/stringzilla)
![StringZilla code size](https://img.shields.io/github/languages/code-size/ashvardanian/stringzilla)

StringZilla is the GodZilla of string libraries, using [SIMD][faq-simd] and [SWAR][faq-swar] to accelerate string operations for modern CPUs.
It is significantly faster than the default string libraries in Python and C++, and offers a more powerful API.
Aside from exact search, the library also accelerates fuzzy search, edit distance computation, and sorting.

[faq-simd]: https://en.wikipedia.org/wiki/Single_instruction,_multiple_data
[faq-swar]: https://en.wikipedia.org/wiki/SWAR

- __[C](#quick-start-c-🛠️):__ Upgrade LibC's `<string.h>` to `<stringzilla.h>`  in C 99
- __[C++](#quick-start-cpp-🛠️):__ Upgrade STL's `<string>` to `<stringzilla.hpp>` in C++ 11
- __[Python](#quick-start-python-🐍):__ Upgrade your `str` to faster `Str`
- __[Swift](#quick-start-swift-🍎):__ Use the `String+StringZilla` extension
- __[Rust](#quick-start-rust-🦀):__ Use the `StringZilla` crate
- Code in other languages? Let us know!

![](StringZilla-rounded.png)

## Throughput Benchmarks

StringZilla has a lot of functionality, most of which is covered by benchmarks across C, C++, Python and other languages.
You can find those in the `./scripts` directory, with usage notes listed in the `CONTRIBUTING.md` file.
The following table summarizes the most important benchmarks performed on Arm-based Graviton3 AWS `c7g` instances and `r7iz` Intel Sapphire Rapids.

<table style="width: 100%; text-align: center; table-layout: fixed;">
  <colgroup>
    <col style="width: 25%;">
    <col style="width: 25%;">
    <col style="width: 25%;">
    <col style="width: 25%;">
  </colgroup>
  <tr>
    <th align="center">LibC</th>
    <th align="center">C++ Standard</th>
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
    <td align="center">❌</td>
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
    <td colspan="4" align="center">find the first occurrence of any of 6 whitespaces <sup>2</sup></td>
  </tr>
  <tr>
    <td align="center">
      <code>strcspn</code> <sup>1</sup><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.74</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.29</b> GB/s
    </td>
    <td align="center">
      <code>.find_first_of</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.25</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.23</b> GB/s
    </td>
    <td align="center">
      <code>re.finditer</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.06</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.02</b> GB/s
    </td>
    <td align="center">
      <code>sz_find_charset</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.43</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.23</b> GB/s
    </td>
  </tr>
  <!-- Characters, reverse order -->
  <tr>
    <td colspan="4" align="center">find the last occurrence of any of 6 whitespaces <sup>2</sup></td>
  </tr>
  <tr>
    <td align="center">❌</td>
    <td align="center">
      <code>.find_last_of</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.25</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.25</b> GB/s
    </td>
    <td align="center">❌</td>
    <td align="center">
      <code>sz_rfind_charset</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>0.43</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>0.23</b> GB/s
    </td>
  </tr>
  <!-- Edit Distance -->
  <tr>
    <td colspan="4" align="center">Levenshtein edit distance, ≅ 5 bytes long</td>
  </tr>
  <tr>
    <td align="center">❌</td>
    <td align="center">❌</td>
    <td align="center">
      via <code>jellyfish</code> <sup>3</sup><br/>
      <span style="color:#ABABAB;">x86:</span> <b>?</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>2,220</b> ns
    </td>
    <td align="center">
      <code>sz_edit_distance</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>99</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>180</b> ns
    </td>
  </tr>
  <!-- Alignment Score -->
  <tr>
    <td colspan="4" align="center">Needleman-Wunsh alignment scores, ≅ 300 aminoacids long</td>
  </tr>
  <tr>
    <td align="center">❌</td>
    <td align="center">❌</td>
    <td align="center">
      via <code>biopython</code> <sup>4</sup><br/>
      <span style="color:#ABABAB;">x86:</span> <b>?</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>254</b> ms
    </td>
    <td align="center">
      <code>sz_alignment_score</code><br/>
      <span style="color:#ABABAB;">x86:</span> <b>73</b> &centerdot;
      <span style="color:#ABABAB;">arm:</span> <b>177</b> ms
    </td>
  </tr>
</table>

> Benchmarks were conducted on a 1 GB English text corpus, with an average word length of 5 characters.
> The hardware used is an AVX-512 capable Intel Sapphire Rapids CPU.
> The code was compiled with GCC 12, using `glibc` v2.35.

__Who is this for?__

- For data-engineers often memory-mapping and parsing large datasets, like the [CommonCrawl](https://commoncrawl.org/).
- For Python, C, or C++ software engineers looking for faster strings for their apps.
- For Bioinformaticians and Search Engineers measuring edit distances and fuzzy-matching.
- For hardware designers, needing a SWAR baseline for strings-processing functionality.
- For students studying SIMD/SWAR applications to non-data-parallel operations.

__Technical insights:__

- Uses SWAR and SIMD to accelerate exact search for very short needles under 4 bytes.
- Uses the Shift-Or Bitap algorithm for mid-length needles under 64 bytes.
- Uses the Boyer-Moore-Horspool algorithm with Raita heuristic for longer needles.
- Uses the Manber-Wu improvement of the Shift-Or algorithm for bounded fuzzy search.
- Uses the two-row Wagner-Fisher algorithm for Levenshtein edit distance computation.
- Uses the Needleman-Wunsch improvement for parameterized edit distance computation.
- Uses the Karp-Rabin rolling hashes to produce binary fingerprints.
- Uses Radix Sort to accelerate sorting of strings.

The choice of the optimal algorithm is predicated on the length of the needle and the alphabet cardinality.
If the amount of compute per byte is low and the needles are beyond longer than the cache-line (64 bytes), skip-table-based approaches are preferred.
In other cases, brute force approaches can be more efficient.
On the engineering side, the library:

- Implement the Small String Optimization for strings shorter than 23 bytes.
- Avoids PyBind11, SWIG, `ParseTuple` and other CPython sugar to minimize call latency. [_details_](https://ashvardanian.com/posts/pybind11-cpython-tutorial/) 

> [!NOTE]
> Current StringZilla design assumes little-endian architecture, ASCII or UTF-8 encoding, and 64-bit address space.
> This covers most modern CPUs, including x86, Arm, RISC-V.
> Feel free to open an issue if you need support for other architectures.


## Supported Functionality

| Functionality        | C 99 | C++ 11 | Python | Swift | Rust |
| :------------------- | :--- | :----- | :----- | :---- | :--- |
| Substring Search     | ✅    | ✅      | ✅      | ✅     | ✅    |
| Character Set Search | ✅    | ✅      | ✅      | ✅     | ✅    |
| Edit Distance        | ✅    | ✅      | ✅      | ✅     | ❌    |
| Small String Class   | ✅    | ✅      | ❌      | ❌     | ❌    |
| Sequence Operation   | ✅    | ❌      | ✅      | ❌     | ❌    |
| Lazy Ranges          | ❌    | ✅      | ❌      | ❌     | ❌    |
| Fingerprints         | ✅    | ✅      | ❌      | ❌     | ❌    |

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

Assuming superior search speed splitting should also work 3x faster than with native Python strings.
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

Those collections of `Strs` are designed to keep the memory consumption low.
If all the chunks are located in consecutive memory regions, the memory overhead can be as low as 4 bytes per chunk.
That's designed to handle very large datasets, like [RedPajama][redpajama].
To address all 20 Billion annotated english documents in it, one will need only 160 GB of RAM instead of Terabytes.

[redpajama]: https://github.com/togethercomputer/RedPajama-Data

### Low-Level Python API

The StringZilla CPython bindings implement vector-call conventions for faster calls.

```py
import stringzilla as sz

contains: bool = sz.contains("haystack", "needle", start=0, end=9223372036854775807)
offset: int = sz.find("haystack", "needle", start=0, end=9223372036854775807)
count: int = sz.count("haystack", "needle", start=0, end=9223372036854775807, allowoverlap=False)
edit_distance: int = sz.edit_distance("needle", "nidl")
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

Unlike LibC:

- all strings are expected to have a length, and are not necesserily null-terminated.
- every operations has a reverse order counterpart.

That way `sz_find` and `sz_rfind` are similar to `strstr` and `strrstr` in LibC.
Similarly, `sz_find_byte` and `sz_rfind_byte` replace `memchr` and `memrchr`.
The `sz_find_charset` maps to `strspn` and `strcspn`, while `sz_rfind_charset` has no sibling in LibC.

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
        <td><code>sz_rfind_charset(haystack, haystack_length, needles_bitset)</code></td>
    </tr>
    <tr>
        <td><code>strspn(haystack, needles)</code></td>
        <td><code>sz_find_charset(haystack, haystack_length, needles_bitset)</code></td>
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
auto hash = std::hash<sz::string_view>(haystack); // Compatible with STL's `std::hash`

haystack.end() - haystack.begin() == haystack.size(); // Or `rbegin`, `rend`
haystack.find_first_of(" \w\t") == 4; // Or `find_last_of`, `find_first_not_of`, `find_last_not_of`
haystack.starts_with(needle) == true; // Or `ends_with`
haystack.remove_prefix(needle.size()); // Why is this operation in-place?!
haystack.contains(needle) == true; // STL has this only from C++ 23 onwards
haystack.compare(needle) == 1; // Or `haystack <=> needle` in C++ 20 and beyond
```

StringZilla also provides string literals for automatic type resolution, [similar to STL][stl-literal]:

```cpp
using sz::literals::operator""_sz;
using std::literals::operator""sv;

auto a = "some string"; // char const *
auto b = "some string"sv; // std::string_view
auto b = "some string"_sz; // sz::string_view
```

[stl-literal]: https://en.cppreference.com/w/cpp/string/basic_string_view/operator%22%22sv

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

|                       | `libstdc++` in  GCC 13 | `libc++` in Clang 17 | StringZilla |
| :-------------------- | ---------------------: | -------------------: | ----------: |
| `sizeof(std::string)` |                     32 |                   24 |          32 |
| Small String Capacity |                     15 |               __22__ |      __22__ |

> Use the following gist to check on your compiler: https://gist.github.com/ashvardanian/c197f15732d9855c4e070797adf17b21

Other langauges, also freuqnetly rely on such optimizations.

- Swift can store 15 bytes in the `String` struct. [docs](https://developer.apple.com/documentation/swift/substring/withutf8(_:)#discussion)

For C++ users, the `sz::string` class hides those implementation details under the hood.
For C users, less familiar with C++ classes, the `sz_string_t` union is available with following API.

```c
sz_memory_allocator_t allocator;
sz_string_t string;

// Init and make sure we are on stack
sz_string_init(&string);
sz_string_is_on_stack(&string); // == sz_true_k

// Optionally pre-allocate space on the heap for future insertions.
sz_string_grow(&string, 100, &allocator); // == sz_true_k

// Append, erase, insert into the string.
sz_string_append(&string, "_Hello_", 7, &allocator); // == sz_true_k
sz_string_append(&string, "world", 5, &allocator); // == sz_true_k
sz_string_erase(&string, 0, 1);

// Upacking & introspection.
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

### Against the Standard Library

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
str("a:b").back(-1) == "b"; // accepting negative indices
str("a:b").sub(1, -1) == ":"; // similar to Python's `"a:b"[1:-1]`
str("a:b").sub(-2, -1) == ":"; // similar to Python's `"a:b"[-2:-1]`
str("a:b").sub(-2, 1) == ""; // similar to Python's `"a:b"[-2:1]`
"a:b"_sz[{-2, -1}] == ":"; // works on views and overloads `operator[]`
```

Assuming StringZilla is a header-only library you can use the full API in some translation units and gradually transition to safer restricted API in others.
Bonus - all the bound checking is branchless, so it has a constant cost and won't hurt your branch predictor.


### Beyond the Standard Templates Library - Learning from Python

Python is arguably the most popular programming language for data science.
In part, that's due to the simplicity of its standard interfaces.
StringZilla brings some of thet functionality to C++.

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
auto [before, match, after] = haystack.partition(char_set(":;")); // Character-set argument
auto [before, match, after] = haystack.partition(" : "); // String argument
auto [before, match, after] = haystack.rpartition(sz::whitespaces); // Split around the last whitespace
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
text.contains_only(" \w\t"); // == text.find_first_not_of(char_set(" \w\t")) == npos;
text.contains(sz::whitespaces); // == text.find(char_set(sz::whitespaces)) != npos;

// Simpler slicing than `substr`
text.front(10); // -> sz::string_view
text.back(10); // -> sz::string_view

// Safe variants, which clamp the range into the string bounds
using sz::string::cap;
text.front(10, cap) == text.front(std::min(10, text.size()));
text.back(10, cap) == text.back(std::min(10, text.size()));

// Character set filtering
text.lstrip(sz::whitespaces).rstrip(sz::newlines); // like Python
text.front(sz::whitespaces); // all leading whitespaces
text.back(sz::digits); // all numerical symbols forming the suffix

// Incremental construction
using sz::string::unchecked;
text.push_back('x'); // no surprises here
text.push_back('x', unchecked); // no bounds checking, Rust style
text.try_push_back('x'); // returns `false` if the string is full and the allocation failed

sz::concatenate(text, "@", domain, ".", tld); // No allocations
text + "@" + domain + "." + tld; // No allocations, if `SZ_LAZY_CONCAT` is defined
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
    for (auto word : line.split(char_set(" \w\t.,;:!?")))
        std::cout << word << std::endl;
```

Each of those is available in reverse order as well.
It also allows interleaving matches, if you want both inclusions of `xx` in `xxx`.
Debugging pointer offsets is not a pleasant exercise, so keep the following functions in mind.

- `haystack.[r]find_all(needle, interleaving)`
- `haystack.[r]find_all(char_set(""))`
- `haystack.[r]split(needle)`
- `haystack.[r]split(char_set(""))`

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
    static std::random_device seed_source; // Too expensive to construct every time
    std::mt19937 generator(seed_source());
    std::uniform_int_distribution<std::size_t> distribution(1, cardinality);
    std::generate(result.begin(), result.end(), [&]() { return alphabet[distribution(generator)]; });
    return result;
}
```

Mouthful and slow.
StringZilla provides a C native method - `sz_generate` and a convenient C++ wrapper - `sz::generate`.
Similar to Python it also defines the commonly used character sets.

```cpp
sz::string word = sz::generate(5, sz::ascii_letters);
sz::string packet = sz::generate(length, sz::base64);

auto protein = sz::string::random(300, "ARNDCQEGHILKMFPSTWYV"); // static method
auto dna = sz::basic_string<custom_allocator>::random(3_000_000_000, "ACGT");

dna.randomize("ACGT"); // `noexcept` pre-allocated version
dna.randomize(&std::rand, "ACGT"); // custom distribution
```

Recent benchmarks suggest the following numbers for strings of different lengths.

| Length | `std::generate` → `std::string` | `sz::generate` → `sz::string` |
| -----: | ------------------------------: | ----------------------------: |
|      5 |                        0.5 GB/s |                      1.5 GB/s |
|     20 |                        0.3 GB/s |                      1.5 GB/s |
|    100 |                        0.2 GB/s |                      1.5 GB/s |

### Levenshtein Edit Distance and Alignment Scores

### Fuzzy Search with Bounded Levenshtein Distance

```cpp
// For Levenshtein distance, the following are available:
text.edit_distance(other[, upper_bound]) == 7; // May perform a memory allocation
text.find_similar(other[, upper_bound]);
text.rfind_similar(other[, upper_bound]);
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
std::map<std::string, int, sz::string_view_less> sorted_words;
std::unordered_map<std::string, int, sz::string_view_hash, sz::string_view_equal_to> words;
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

__`SZ_USE_X86_AVX512`, `SZ_USE_X86_AVX2`, `SZ_USE_ARM_NEON`__:

> One can explicitly disable certain families of SIMD instructions for compatibility purposes.
> Default values are inferred at compile time.

__`SZ_DYNAMIC_DISPATCH`__:

> By default, StringZilla is a header-only library.
> But if you are running on different generations of devices, it makes sense to pre-compile the library for all supported generations at once, and dispatch at runtime.
> This flag does just that and is used to produce the `stringzilla.so` shared library, as well as the Python bindings.

__`SZ_USE_MISALIGNED_LOADS`__:

> By default, StringZilla avoids misaligned loads.
> If supported, it replaces many byte-level operations with word-level ones.
> Going from `char`-like types to `uint64_t`-like ones can significantly accelerate the serial (SWAR) backend.
> So consider enabling it if you are building for some embedded device.

__`SZ_AVOID_LIBC`__:

> When using the C header-only library one can disable the use of LibC.
> This may affect the type resolution system on obscure hardware platforms. 

__`SZ_AVOID_STL`__:

> When using the C++ interface one can disable conversions from `std::string` to `sz::string` and back.
> If not needed, the `<string>` and `<string_view>` headers will be excluded, reducing compilation time.

__`STRINGZILLA_BUILD_SHARED`, `STRINGZILLA_BUILD_TEST`, `STRINGZILLA_BUILD_BENCHMARK`, `STRINGZILLA_TARGET_ARCH`__ for CMake users:

> When compiling the tests and benchmarks, you can explicitly set the target hardware architecture.
> It's synonymous to GCC's `-march` flag and is used to enable/disable the appropriate instruction sets.
> You can also disable the shared library build, if you don't need it.

## Algorithms & Design Decisions 📚

StringZilla aims to optimize some of the slowest string operations.
Some popular operations, however, like equality comparisons and relative order checking, almost always complete on some of the very first bytes in either string.
In such operations vectorization is almost useless, unless huge and very similar strings are considered.
StringZilla implements those operations as well, but won't result in substantial speedups.

### Exact Substring Search

StringZilla uses different exact substring search algorithms for different needle lengths and backends:

- When no SIMD is available - SWAR (SIMD Within A Register) algorithms are used on 64-bit words.
- Boyer-Moore-Horspool (BMH) algorithm with Raita heuristic variation for longer needles.
- SIMD algorithms are randomized to look at different parts of the needle.
- Apostolico-Giancarlo algorithm is _considered_ for longer needles, if preprocessing time isn't an issue.

Substring search algorithms are generally divided into: comparison-based, automaton-based, and bit-parallel.
Different families are effective for different alphabet sizes and needle lengths.
The more operations are needed per-character - the more effective SIMD would be.
The longer the needle - the more effective the skip-tables are.

On very short needles, especially 1-4 characters long, brute force with SIMD is the fastest solution.
On mid-length needles, bit-parallel algorithms are effective, as the character masks fit into 32-bit or 64-bit words.
Either way, if the needle is under 64-bytes long, on haystack traversal we will still fetch every CPU cache line.
So the only way to improve performance is to reduce the number of comparisons.

Going beyond that, to long needles, Boyer-Moore (BM) and its variants are often the best choice.
It has two tables: the good-suffix shift and the bad-character shift.
Common choice is to use the simplified BMH algorithm, which only uses the bad-character shift table, reducing the pre-processing time.
In the C++ Standards Library, the `std::string::find` function uses the BMH algorithm with Raita's heuristic.
We do something similar longer needles.

All those, still, have $O(hn)$ worst case complexity, and struggle with repetitive needle patterns.
To guarantee $O(h)$ worst case time complexity, the Apostolico-Giancarlo (AG) algorithm adds an additional skip-table.
Preprocessing phase is $O(n+sigma)$ in time and space.
On traversal, performs from $(h/n)$ to $(3h/2)$ comparisons.
We should consider implementing it if we can:

- accelerate the preprocessing phase of the needle.
- simplify the control-flow of the main loop.
- replace the array of shift values with a circular buffer.

Reading materials:

- Exact String Matching Algorithms in Java: https://www-igm.univ-mlv.fr/~lecroq/string
- SIMD-friendly algorithms for substring searching: http://0x80.pl/articles/simd-strfind.html


### Hashing

Hashing is a very deeply studies subject with countless implementations.
Choosing the right hashing algorithm for your application can be crucial from both performance and security standpoint.
In StringZilla a 64-bit rolling hash function is reused for both string hashes and substring hashes, Rabin-style fingerprints.
Rolling hashes take the same amount of time to compute hashes with different window sizes, and are fast to update.
Those are not however perfect hashes, and collisions are frequent.
To reduce those.


They are not, however, optimal for cryptographic purposes, and require integer multiplication, which is not always fast.
Using SIMD, we can process N interleaving slices of the input in parallel.
On Intel Sapphire Rapids, the following numbers can be expected for N-way parallel variants.

- 4-way AVX2 throughput with 64-bit integer multiplication (no native support): 0.28 GB/s.
- 4-way AVX2 throughput with 32-bit integer multiplication: 0.54 GB/s.
- 4-way AVX-512DQ throughput with 64-bit integer multiplication: 0.46 GB/s.
- 4-way AVX-512 throughput with 32-bit integer multiplication: 0.58 GB/s.
- 8-way AVX-512 throughput with 32-bit integer multiplication: 0.11 GB/s.



#### Why not CRC32?

Cyclic Redundancy Check 32 is one of the most commonly used hash functions in Computer Science.
It has in-hardware support on both x86 and Arm, for both 8-bit, 16-bit, 32-bit, and 64-bit words.
The `0x1EDC6F41` polynomial is used in iSCSI, Btrfs, ext4, and the `0x04C11DB7` in SATA, Ethernet, Zlib, PNG.
In case of Arm more than one polynomial is supported.
It is, however, somewhat limiting for Big Data usecases, which often have to deal with more than 4 Billion strings, making collisions unavoidable.
Moreover, the existing SIMD approaches are tricky, combining general purpose computations with specialized instructions, to utilize more silicon in every cycle.

Some of the best articles on CRC32:

- [Comprehensive derivation of approaches](https://github.com/komrad36/CRC)
- [Faster computation for 4 KB buffers on x86](https://www.corsix.org/content/fast-crc32c-4k)
- [Comparing different lookup tables](https://create.stephan-brumme.com/crc32)

Some of the best open-source implementations:

- [By Peter Cawley](https://github.com/corsix/fast-crc32)
- [By Stephan Brumme](https://github.com/stbrumme/crc32)

#### Other Modern Alternatives

[MurmurHash](https://github.com/aappleby/smhasher/blob/master/README.md) from 2008 by Austin Appleby is one of the best known non-cryptographic hashes.
It has a very short implementation and is capable of producing 32-bit and 128-bit hashes.
The [CityHash](https://opensource.googleblog.com/2011/04/introducing-cityhash) from 2011 by Google and the [xxHash](https://github.com/Cyan4973/xxHash) improve on that, better leveraging the super-scalar nature of modern CPUs and producing 64-bit and 128-bit hashes.

Neither of those functions are cryptographic, unlike MD5, SHA, and BLAKE algorithms.
Most of cryptographic hashes are based on the Merkle-Damgård construction, and aren't resistant to the length-extension attacks.
Current state of the Art, might be the [BLAKE3](https://github.com/BLAKE3-team/BLAKE3) algorithm.
It's resistant to a broad range of attacks, can process 2 bytes per CPU cycle, and comes with a very optimized official implementation for C and Rust.
It has the same 128-bit security level as the BLAKE2, and achieves its performance gains by reducing the number of mixing rounds, and processing data in 1 KiB chunks, which is great for longer strings, but may result in poor performance on short ones.

> [!TIP]
> All mentioned libraries have undergone extensive testing and are considered production-ready.
> They can definitely accelerate your application, but so may the downstream mixer.
> For instance, when a hash-table is constructed, the hashes are further shrunk to address table buckets.
> If the mixer looses entropy, the performance gains from the hash function may be lost.
> An example would be power-of-two modulo, which is a common mixer, but is known to be weak.
> One alternative would be the [fastrange](https://github.com/lemire/fastrange) by Daniel Lemire.
> Another one is the [Fibonacci hash trick](https://probablydance.com/2018/06/16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/) using the Golden Ratio, also used in StringZilla.

### Levenshtein Edit Distance

StringZilla can compute the Levenshtein edit distance between two strings.
For that the two-row Wagner-Fisher algorithm is used, which is a space-efficient variant of the Needleman-Wunsch algorithm.
The algorithm is implemented in C and C++ and is available in the `stringzilla.h` and `stringzilla.hpp` headers respectively.
It's also available in Python via the `Str.edit_distance` method and as a global function in the `stringzilla` module.

```py
import stringzilla as sz

words = open('leipzig1M').read().split(' ')

for word in words:
    sz.edit_distance(word, "rebel")
    sz.edit_distance(word, "statement")
    sz.edit_distance(word, "sent")
```

Even without SIMD optimizations, one can expect the following evaluation time for the main `for`-loop on short word-like tokens on a modern CPU core.

- [EditDistance](https://github.com/roy-ht/editdistance): 28.7s
- [JellyFish](https://github.com/jamesturk/jellyfish/): 26.8s
- [Levenshtein](https://github.com/maxbachmann/Levenshtein): 8.6s
- StringZilla: __4.2s__

### Needleman-Wunsch Alignment Score for Bioinformatics

Similar to the conventional Levenshtein edit distance, StringZilla can compute the Needleman-Wunsch alignment score.
It's practically the same, but parameterized with a scoring matrix for different substitutions and tunable penalties for insertions and deletions.

### Unicode, UTF-8, and Wide Characters

UTF-8 is the most common encoding for Unicode characters.
Yet, some programming languages use wide characters (`wchar`) - two byte long codes.
These include Java, JavaScript, Python 2, C#, and Objective-C, to name a few.
This leads [to all kinds of offset-counting issues][wide-char-offsets] when facing four-byte long Unicode characters.

[wide-char-offsets]: https://josephg.com/blog/string-length-lies/

## Contributing 👾

Please check out the [contributing guide](CONTRIBUTING.md) for more details on how to setup the development environment and contribute to this project.
If you like this project, you may also enjoy [USearch][usearch], [UCall][ucall], [UForm][uform], and [SimSIMD][simsimd]. 🤗

[usearch]: https://github.com/unum-cloud/usearch
[ucall]: https://github.com/unum-cloud/ucall
[uform]: https://github.com/unum-cloud/uform
[simsimd]: https://github.com/ashvardanian/simsimd

## License 📜

Feel free to use the project under Apache 2.0 or the Three-clause BSD license at your preference.
