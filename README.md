# StringZilla 🦖

StringZilla is the GodZilla of string libraries, using [SIMD][faq-simd] and [SWAR][faq-swar] to accelerate string operations for modern CPUs.
It is significantly faster than the default string libraries in Python and C++, and offers a more powerful API.
Aside from exact search, the library also accelerates fuzzy search, edit distance computation, and sorting.

[faq-simd]: https://en.wikipedia.org/wiki/Single_instruction,_multiple_data
[faq-swar]: https://en.wikipedia.org/wiki/SWAR

- Code in C? Replace LibC's `<string.h>` with C 99 `<stringzilla.h>`  - [_more_](#quick-start-c-🛠️)
- Code in C++? Replace STL's `<string>` with C++ 11 `<stringzilla.hpp>` - [_more_](#quick-start-cpp-🛠️)
- Code in Python? Upgrade your `str` to faster `Str` - [_more_](#quick-start-python-🐍)
- Code in other languages? Let us know!

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
- For hardware designers, needing a SWAR baseline for strings-processing functionality.
- For students studying SIMD/SWAR applications to non-data-parallel operations.

__Limitations:__

- Assumes little-endian architecture (most CPUs, including x86, Arm, RISC-V).
- Assumes ASCII or UTF-8 encoding (most content and systems).
- Assumes 64-bit address space (most modern CPUs).

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
Like other efficient string implementations, it uses the [Small String Optimization][faq-sso] to avoid heap allocations for short strings.

```c
typedef union sz_string_t {
    struct on_stack {
        sz_ptr_t start;
        sz_u8_t length;
        char chars[sz_string_stack_space]; /// Ends with a null-terminator.
    } on_stack;

    struct on_heap {
        sz_ptr_t start;
        sz_size_t length;        
        sz_size_t space; /// The length of the heap-allocated buffer.
        sz_size_t padding;
    } on_heap;

} sz_string_t;
```

As one can see, a short string can be kept on the stack, if it fits within `on_stack.chars` array.
Before 2015 GCC string implementation was just 8 bytes.
Today, practically all variants are at least 32 bytes, so two of them fit in a cache line.
Practically all of them can only store 15 bytes of the "Small String" on the stack.
StringZilla can store strings up to 22 bytes long on the stack, while avoiding any branches on pointer and length lookups.

|                       | GCC 13 | Clang 17 | ICX 2024 |     StringZilla |
| :-------------------- | -----: | -------: | -------: | --------------: |
| `sizeof(std::string)` |     32 |       32 |       32 |              32 |
| Small String Capacity |     15 |       15 |       15 | __22__ (+ 47 %) |

> Use the following gist to check on your compiler: https://gist.github.com/ashvardanian/c197f15732d9855c4e070797adf17b21

For C++ users, the `sz::string` class hides those implementation details under the hood.
For C users, less familiar with C++ classes, the `sz_string_t` union is available with following API.

```c
sz_memory_allocator_t allocator;
sz_string_t string;

// Init and make sure we are on stack
sz_string_init(&string);
assert(sz_string_is_on_stack(&string) == sz_true_k);

// Optionally pre-allocate space on the heap for future insertions.
assert(sz_string_grow(&string, 100, &allocator) == sz_true_k);

// Append, erase, insert into the string.
assert(sz_string_append(&string, "_Hello_", 7, &allocator) == sz_true_k);
assert(sz_string_append(&string, "world", 5, &allocator) == sz_true_k);
sz_string_erase(&string, 0, 1);

// Upacking & introspection.
sz_ptr_t string_start;
sz_size_t string_length;
sz_size_t string_space;
sz_bool_t string_is_on_heap;
sz_string_unpack(string, &string_start, &string_length, &string_space, &string_is_on_heap);
assert(sz_equal(string_start, "Hello_world", 11) == sz_true_k);

// Reclaim some memory.
assert(sz_string_shrink_to_fit(&string, &allocator) == sz_true_k);
sz_string_free(&string, &allocator);
```

Unlike the conventional C strings, the `sz_string_t` is allowed to contain null characters.
To safely print those, pass the `string_length` to `printf` as well.

```c
printf("%.*s\n", (int)string_length, string_start);
```

### Beyond the Standard Templates Library

Aside from conventional `std::string` interfaces, non-STL extensions are available.

```cpp
haystack.count(needle) == 1; // Why is this not in STL?!

haystack.edit_distance(needle) == 7;
haystack.find_similar(needle, bound);
haystack.rfind_similar(needle, bound);
```

When parsing documents, it is often useful to split it into substrings.
Most often, after that, you would compute the length of the skipped part, the offset and the length of the remaining part.
StringZilla provides a convenient `split` function, which returns a tuple of three string views, making the code cleaner.

```cpp
auto [before, match, after] = haystack.split(':');
auto [before, match, after] = haystack.split(character_set(":;"));
auto [before, match, after] = haystack.split(" : ");
```

### Ranges

One of the most common use cases is to split a string into a collection of substrings.
Which would often result in [StackOverflow lookups][so-split] and snippets like the one below.

[so-split]: https://stackoverflow.com/questions/14265581/parse-split-a-string-in-c-using-string-delimiter-standard-c

```cpp
std::vector<std::string> lines = your_split_by_substrings(haystack, "\r\n");
std::vector<std::string> words = your_split_by_character(lines, ' ');
```

Those allocate memory for each string and the temporary vectors.
Each allocation can be orders of magnitude more expensive, than even serial `for`-loop over characters.
To avoid those, StringZilla provides lazily-evaluated ranges, compatible with the [Range-v3][range-v3] library.

[range-v3]: https://github.com/ericniebler/range-v3

```cpp
for (auto line : haystack.split_all("\r\n"))
    for (auto word : line.split_all(character_set(" \w\t.,;:!?")))
        std::cout << word << std::endl;
```

Each of those is available in reverse order as well.
It also allows interleaving matches, if you want both inclusions of `xx` in `xxx`.
Debugging pointer offsets is not a pleasant exercise, so keep the following functions in mind.

- `haystack.[r]find_all(needle, interleaving)`
- `haystack.[r]find_all(character_set(""))`
- `haystack.[r]split_all(needle)`
- `haystack.[r]split_all(character_set(""))`

For $N$ matches the split functions will report $N+1$ matches, potentially including empty strings.
Ranges have a few convinience methods as well:

```cpp
range.size(); // -> std::size_t
range.empty(); // -> bool
range.template to<std::set<std::sting>>(); 
range.template to<std::vector<std::sting_view>>(); 
```

### TODO: STL Containers with String Keys

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

### TODO: Concatenating Strings

Ansother common string operation is concatenation.
The STL provides `std::string::operator+` and `std::string::append`, but those are not the most efficient, if multiple invocations are performed.

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
StringZilla provides a more convenient `concat` function, which takes a variadic number of arguments.

```cpp
auto email = sz::concat(name, "@", domain, ".", tld);
```

Moreover, if the first or second argument of the expression is a StringZilla string, the concatenation can be poerformed lazily using the same `operator+` syntax.
That behavior is disabled for compatibility by default, but can be enabled by defining `SZ_LAZY_CONCAT` macro.

```cpp
sz::string name, domain, tld;
auto email_expression = name + "@" + domain + "." + tld;    // 0 allocations
sz::string email = name + "@" + domain + "." + tld;         // 1 allocations
```

### Debugging

For maximal performance, the library does not perform any bounds checking in Release builds.
That behavior is controllable for both C and C++ interfaces via the `STRINGZILLA_DEBUG` macro.

[faq-sso]: https://cpp-optimizations.netlify.app/small_strings/

## Algorithms 📚

### Hashing

### Substring Search

## Contributing 👾

Please check out the [contributing guide](CONTRIBUTING.md) for more details on how to setup the development environment and contribute to this project.
If you like this project, you may also enjoy [USearch][usearch], [UCall][ucall], [UForm][uform], and [SimSIMD][simsimd]. 🤗

[usearch]: https://github.com/unum-cloud/usearch
[ucall]: https://github.com/unum-cloud/ucall
[uform]: https://github.com/unum-cloud/uform
[simsimd]: https://github.com/ashvardanian/simsimd

## License 📜

Feel free to use the project under Apache 2.0 or the Three-clause BSD license at your preference.
