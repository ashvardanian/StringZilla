# StringZilla for C and C++

StringZilla is implemented as __C-level string kernels__ with a thin __C++ binding__ for efficient byte-string processing, exposing __lazy, range-compatible iterators__ over an all-`noexcept` surface.
It bundles:

- a header-only C library importable from `<stringzilla/stringzilla.h>`,
- a header-only C++ library importable from `<stringzilla/stringzilla.hpp>`,
- a precompiled __shared__ library enabling dynamic dispatch for maximum portability.

The plain C ABI exposes each kernel family as a stable C 99 surface: substring and byte-set search, non-cryptographic hashing and checksums, lexicographic comparison, sorting and intersection of string collections, and `memcpy`/`memmove`/`memset`/lookup-table memory transforms.
The thin C++ binding rebuilds the STL `<string>` and `<string_view>` surface on top of those kernels, adding an owning Small-String-Optimized container, allocation-free splitting and partitioning views, and free functions for hashing, sorting, and translation.

The fastest SIMD backend is picked per CPU — at compile time in header-only mode, or at runtime when the library is built with dynamic dispatch.
There is no hidden global allocation and no thread pool: every function that may allocate takes an explicit `sz_memory_allocator_t *`, and parallel or GPU engines live in a separate distribution, not in these headers.

The headers compile as freestanding C 99 — set `SZ_AVOID_LIBC=1` to drop the libc dependency — and as C++11 or newer for the `sz::` layer.
Per-family hubs `find.h`, `hash.h`, `sort.h`, `compare.h`, `intersect.h`, `memory.h`, and `small_string.h` forward to the per-ISA kernels under the matching subdirectories — `find/haswell.h`, `hash/icelake.h`, `memory/neon.h`, and so on — each guarded by an `SZ_USE_*` macro.

## Installation

The core is header-only, so the simplest integration is to put `include/` on your search path and include the umbrella header.
No build step is required, and the best SIMD backend for your compiler flags is selected at compile time.

```c
#include <stringzilla/stringzilla.h> // C
```

```cpp
#include <stringzilla/stringzilla.hpp> // C++
namespace sz = ashvardanian::stringzilla;
```

Each family can also be included on its own when you only need a slice of the API and want to keep compile times down:

```c
#include <stringzilla/find.h> // `sz_find`, `sz_rfind`, byte-set scans
#include <stringzilla/hash.h> // `sz_hash`, `sz_bytesum`, incremental state
#include <stringzilla/sort.h> // `sz_sequence_argsort`
#include <stringzilla/compare.h> // `sz_equal`, `sz_order`
#include <stringzilla/intersect.h> // `sz_sequence_intersect`
#include <stringzilla/memory.h> // `sz_copy`, `sz_move`, `sz_fill`, `sz_lookup`
#include <stringzilla/small_string.h> // `sz_string_t` SSO container
```

### CMake, Header Only

With CMake 3.14 or newer, pull the project in with `FetchContent` and link the interface target `stringzilla::stringzilla_header`, which only adds `include/` to your search path.

```cmake
include(FetchContent)
FetchContent_Declare(
    stringzilla
    GIT_REPOSITORY https://github.com/ashvardanian/stringzilla.git
    GIT_TAG main)
FetchContent_MakeAvailable(stringzilla)

target_link_libraries(your_app PRIVATE stringzilla::stringzilla_header)
```

`add_subdirectory(stringzilla)` works the same way.
Pulled in as a subproject, `STRINGZILLA_BUILD_SHARED` defaults to off, so only the header target is configured and nothing is compiled.

### CMake, Precompiled Library

For a single binary that resolves the best backend at __runtime__, turn on `STRINGZILLA_BUILD_SHARED` and link the shared target `stringzilla::stringzilla_shared`.

```cmake
set(STRINGZILLA_BUILD_SHARED ON)
FetchContent_MakeAvailable(stringzilla)

target_link_libraries(your_app PRIVATE stringzilla::stringzilla_shared)
```

`stringzilla_shared` compiles every backend and initializes the dispatch table once at load, so one artifact runs optimally on any CPU.
On Linux a libc-free variant `stringzilla_bare` is built alongside it with `SZ_AVOID_LIBC=1`.
Configure with `-DSTRINGZILLA_INSTALL=ON` to install the headers and libraries system-wide.

### Build Knobs

The behavior of the core is controlled entirely by preprocessor macros, so you can tune it without touching the sources.
The most important ones, mirrored from `stringzilla.h`:

| Macro                     | Default   | Effect                                   |
| ------------------------- | --------- | ---------------------------------------- |
| `SZ_DYNAMIC_DISPATCH`     | `0`       | Compile-time vs. runtime backend choice  |
| `SZ_DEBUG`                | `0`       | Debug assertions and logging             |
| `SZ_AVOID_LIBC`           | `0`       | Freestanding build, no libc              |
| `SZ_USE_MISALIGNED_LOADS` | platform  | Unaligned word loads in SWAR fallbacks   |
| `SZ_SWAR_THRESHOLD`       | `24`      | Length below which scalar loops are used |
| `SZ_CACHE_LINE_WIDTH`     | `64`      | Cache-line width for heuristics          |
| `SZ_CACHE_SIZE`           | `1048576` | L1d+L2 size for non-temporal stores      |

`SZ_DYNAMIC_DISPATCH` controls dispatch: with `0` the best backend is chosen at compile time and every public function is `static`/inline, while `1` compiles all backends and selects one at runtime through a dispatch table — this is how the shared library is built.
`SZ_AVOID_LIBC` builds freestanding without the C standard library, which disables the default `malloc`-based allocator and the `offsetof` static checks.
`SZ_USE_MISALIGNED_LOADS` allows unaligned word loads in the SWAR fallbacks where the platform permits.

Per-ISA backends are toggled with their own macros; if left undefined they are auto-detected from the compiler's target flags:
`SZ_USE_WESTMERE` for SSE4.2 + AES-NI, `SZ_USE_GOLDMONT` for SHA-NI, `SZ_USE_HASWELL` for AVX2, `SZ_USE_SKYLAKE` for AVX-512 F/BW/VL, `SZ_USE_ICELAKE` for AVX-512 VBMI + VAES, then `SZ_USE_NEON`, `SZ_USE_NEONAES`, `SZ_USE_NEONSHA`, `SZ_USE_SVE`, `SZ_USE_SVE2`, and `SZ_USE_SVE2AES` on ARM, `SZ_USE_V128` and `SZ_USE_V128RELAXED` for WebAssembly SIMD128, `SZ_USE_RVV` and `SZ_USE_RVVCRYPTO` for the RISC-V Vector extension, `SZ_USE_LASX` for LoongArch, and `SZ_USE_POWERVSX` for IBM Power.

The umbrella header also exposes the version triple as `STRINGZILLA_H_VERSION_MAJOR`/`_MINOR`/`_PATCH`, with matching `sz_version_major()`, `sz_version_minor()`, and `sz_version_patch()` accessors.

### Compilers and Platforms

The headers require only C 99 and C++ 11, so any reasonably modern compiler builds the serial and SWAR baseline.
The per-ISA SIMD kernels and the project's own tests are CI-validated with these toolchains:

| Toolchain | Recommended Versions |
| --------- | -------------------- |
| GCC       | 12 or newer          |
| Clang     | 16 or newer          |
| MSVC      | Visual Studio 2022   |
| NVCC      | CUDA 12+             |

GCC 10 and older miss a conforming STL `insert` and fail to build the tests.
On macOS, prefer Homebrew Clang over Apple Clang; on Windows, MinGW with GCC works alongside MSVC.
NVCC with CUDA 12 builds the GPU engines, which live in the separate `stringzillas` distribution.

StringZilla also __compiles to WebAssembly__: the `wasm32` toolchain targets `wasm32-wasip1` with `-msimd128 -mrelaxed-simd`, which enables the `SZ_USE_V128` and `SZ_USE_V128RELAXED` kernels listed above.

## Types

### C types

The C API is built around length-bounded raw pointers, so most functions take a `sz_cptr_t`, an alias of `char const *`, or a `sz_ptr_t`, an alias of `char *`, together with a `sz_size_t` length, and return either a `sz_cptr_t` into the input or one of the small enums below.

Scalar and result types from `types.h`:

- `sz_cptr_t`, `sz_ptr_t` — immutable and mutable byte pointers.
- `sz_size_t`, `sz_ssize_t` — unsigned and signed size types.
- `sz_u8_t`, `sz_u32_t`, `sz_u64_t` — fixed-width integers; `sz_rune_t` is a 32-bit Unicode scalar value.
- `sz_bool_t` — `{ sz_false_k = 0, sz_true_k = 1 }`.
- `sz_ordering_t` — `{ sz_less_k = -1, sz_equal_k = 0, sz_greater_k = 1 }`, the C analog of `std::strong_ordering`.
- `sz_status_t` — `sz_success_k = 0`, `sz_bad_alloc_k`, `sz_invalid_utf8_k`, `sz_contains_duplicates_k`, `sz_overflow_risk_k`, and friends.
- `sz_sorted_idx_t`, `sz_pgram_t` — a sorted-permutation index and a pointer-sized N-gram, both aliases of `sz_size_t`.

`sz_byteset_t` is a 256-bit set of byte values used by the byte-set search functions:

```c
sz_byteset_t set;
sz_byteset_init(&set);                 // all bits cleared
sz_byteset_add(&set, '\n');            // include a byte
int has = sz_byteset_contains(&set, '\n');
sz_byteset_invert(&set);               // complement the set
```

`sz_memory_allocator_t` is the explicit allocator, a `{ allocate, free, handle }` triple, handed to any function that may allocate:

```c
sz_memory_allocator_t alloc;
sz_memory_allocator_init_default(&alloc);          // libc malloc/free
// or a fixed arena with no dynamic allocation:
char arena[4096];
sz_memory_allocator_init_fixed(&alloc, arena, sizeof(arena));
```

`sz_sequence_t` is the read-only adapter over an arbitrary collection of strings used by the sort and intersect families.
It exposes a `count` plus `get_start`/`get_length` callbacks over an opaque `handle`, and the helper below wires it to a plain array of C strings:

```c
char const *words[] = {"banana", "apple", "cherry"};
sz_sequence_t seq;
sz_sequence_from_null_terminated_strings(words, 3, &seq);
```

There is no dedicated `sz_string_view_t` struct in the C ABI — views are passed as the `(pointer, length)` pairs described above.
The one owning C string type is `sz_string_t`, documented under [Memory Operations](#memory-operations) below.

### C++ types

The C++ layer lives in `namespace ashvardanian::stringzilla`, conventionally aliased as `sz`, and is built on two class templates plus a 256-bit `byteset`.

- `sz::basic_string_slice<char_type>` — a non-owning, length-bounded view. Two aliases exist: the read-only `sz::string_view = basic_string_slice<char const>` and the mutable `sz::string_span = basic_string_slice<char>`.
- `sz::basic_string<char_type, allocator = std::allocator<char_type>>` — an owning, Small-String-Optimized container; the alias `sz::string` uses `char` and the default `std::allocator`. The allocator must be stateless, i.e. empty.
- `sz::byteset = sz::basic_byteset<char>` — the C++ wrapper around `sz_byteset_t`, constructible from an initializer list, a `(pointer, count)` pair, or a `std::array`, and composable with `operator|`.
- `sz::status_t` — a scoped enum mirroring `sz_status_t` with `success_k`, `bad_alloc_k`, `invalid_utf8_k`, `contains_duplicates_k`, and the rest, returned by the fallible `try_*` free functions.

`sz::string_view` is implicitly constructible from a `char const *`, a `(pointer, length)` pair, `std::string`, and `std::string_view`, and converts back to both; `sz::string` adds construction from repeats, sub-ranges, initializer lists, and lazy `sz::concatenate(...)` expressions.
Both satisfy the `std::string`/`std::string_view` member surface — `data`, `size`, `length`, `empty`, `at`, `front`, `back`, iterators, `substr`, `compare`, the relational operators, and on C++20 `operator<=>` — so they are drop-in for most code.
Both also stream through `operator<<`, and `std::hash<sz::string>` / `std::hash<sz::string_view>` specializations are provided so they work as unordered-container keys out of the box.

User-defined literals live in `sz::literals`:

```cpp
using namespace sz::literals;
sz::string_view v = "needle"_sv;   // a view, no allocation
sz::byteset s = " \t\n\r"_bs;      // a byte-set from a literal
```

Convenience byte-sets are available as free functions: `sz::whitespaces_set()`, `sz::newlines_set()`, `sz::digits_set()`, `sz::hexdigits_set()`, `sz::octdigits_set()`, and `sz::ascii_letters_set()`.

## Searching and Counting

### C

The forward and reverse substring scans mirror libc's `memchr`/`memrchr` and `memmem`, but always take explicit lengths and return a pointer into the haystack, or `NULL` when there is no match:

```c
sz_cptr_t sz_find_byte(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
sz_cptr_t sz_rfind_byte(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
sz_cptr_t sz_find(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length);
sz_cptr_t sz_rfind(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length);
```

Byte-set scans replace `strspn`/`strcspn` and take a prebuilt `sz_byteset_t`:

```c
sz_cptr_t sz_find_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set);
sz_cptr_t sz_rfind_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set);
```

Four header-only shortcuts build the set for you from a needle string, optionally inverting it:

```c
sz_cptr_t sz_find_byte_from(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length);
sz_cptr_t sz_find_byte_not_from(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length);
sz_cptr_t sz_rfind_byte_from(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length);
sz_cptr_t sz_rfind_byte_not_from(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length);
```

Example: find a substring, then count occurrences by re-scanning the tail.

```c
#include <stringzilla/stringzilla.h>

sz_size_t count_occurrences(sz_cptr_t text, sz_size_t length, sz_cptr_t needle, sz_size_t needle_length) {
    sz_size_t total = 0;
    sz_cptr_t cursor = text, end = text + length;
    while ((cursor = sz_find(cursor, (sz_size_t)(end - cursor), needle, needle_length))) {
        ++total;
        cursor += needle_length;             // disjoint matches; use `+1` for overlapping
    }
    return total;
}

int main(void) {
    sz_byteset_t whitespace;
    sz_byteset_init(&whitespace);
    char const *spaces = " \t\n\r\v\f";
    for (char const *p = spaces; *p; ++p) sz_byteset_add(&whitespace, *p);

    char const *line = "  hello world";
    sz_cptr_t first_word = sz_find_byteset(line, 13, /*inverted?*/ &whitespace); // first whitespace
    (void)first_word;
    return 0;
}
```

### C++

The view and the owning string expose `find`/`rfind` for sub-strings, single characters, and byte-sets, plus `contains`, `starts_with`, `ends_with`, and the full `find_first_of` / `find_last_of` / `find_first_not_of` / `find_last_not_of` family taking either a `byteset` or a view.
For repeated matching the view offers lazy, allocation-free ranges: `find_all(needle)` and `rfind_all(needle)`, each with `sz::include_overlaps` / `sz::exclude_overlaps` tag overloads, alongside `find_all(byteset)` and `rfind_all(byteset)`.

```cpp
#include <stringzilla/stringzilla.hpp>
namespace sz = ashvardanian::stringzilla;

int main() {
    sz::string_view haystack = "the quick brown fox";

    auto pos = haystack.find("quick");           // index, or sz::string_view::npos
    assert(pos == 4);
    assert(haystack.contains("brown"));
    assert(haystack.starts_with("the"));
    assert(haystack.ends_with("fox"));

    std::size_t words = 0;
    for (auto word : haystack.find_all(sz::whitespaces_set())) (void)word, ++words;

    auto vowels = sz::byteset {"aeiou", 5};
    auto first_vowel = haystack.find_first_of(vowels);
    assert(first_vowel == 2); // index of 'e'
    return 0;
}
```

The view also offers character-class predicates — `is_alpha`, `is_alnum`, `is_ascii`, `is_digit`, `is_lower`, `is_upper`, `is_space`, `is_printable`, and the general `contains_only(byteset)`.

## Splitting and Partitioning

This is a C++-only convenience layer; the C ABI provides the underlying scans `sz_find` and `sz_find_byteset` that you compose by hand.

`partition` and `rpartition` return a three-way `sz::string_partition_result` struct with `before`, `match`, and `after` members around the first or last occurrence of a pattern.
The pattern can be a view, a single character, or a byte-set.

`split`, `rsplit`, and `splitlines` return lazy ranges of `string_view`s.
`split(view)` yields a `sz::find_splits_view`, `rsplit(view)` yields a `sz::rfind_splits_view`, and the byte-set overloads `split(byteset)` and `rsplit(byteset)` yield the matching character-splitter views.
`split(byteset)` and `rsplit(byteset)` default to whitespace, and `splitlines()` defaults to the newline set.

The key property is that these ranges are _lazy_ — they borrow from the source buffer and walk it one `string_view` at a time, so the next boundary is computed only when you ask for it.
Splitting a large document into millions of tokens therefore costs __no per-token allocation__: each iteration step advances a cursor and hands back a borrowed view.
This is the decisive contrast with the idiomatic STL approach, where splitting builds a `std::vector<std::string>` and heap-allocates a fresh string for every element, copying its bytes out of the source.

```cpp
#include <stringzilla/stringzilla.hpp>
namespace sz = ashvardanian::stringzilla;

int main() {
    sz::string_view csv = "alice,30,engineer";

    sz::string_partition_result<sz::string_view> first = csv.partition(',');
    sz::string_view name = first.before;
    assert(name == "alice");
    assert(first.match == ",");
    assert(first.after == "30,engineer");

    std::size_t fields = 0;
    for (sz::string_view field : csv.split(",")) // "alice", "30", "engineer"
        ++fields;                                // no string is allocated per field
    assert(fields == 3);

    sz::string_view doc = "line one\nline two\nline three";
    for (sz::string_view line : doc.splitlines()) (void)line;

    return 0;
}
```

Each range exposes `begin`/`end`, dereferences to a `sz::string_view`, and can be consumed in a single pass.
When you genuinely need the tokens to outlive the source you can still materialize them into a `std::vector<sz::string_view>`, but only then do you pay for storage — the default path stays allocation-free.

## UTF-8 Segmentation

Splitting Unicode text into the units a human actually perceives is deceptively hard, and getting it wrong corrupts data.
A single emoji can be dozens of bytes and several codepoints yet one perceived character, CJK runs carry no spaces, and a period is not always the end of a sentence.
StringZilla ships the real algorithms — UAX-29 graphemes, words, and sentences, plus UAX-14 line-break opportunities — as lazy ranges that borrow from the source and yield one `sz::string_view` per segment with no per-segment allocation.

A string's length is really three different numbers, and only the grapheme count matches what a user sees:

```cpp
#include <stringzilla/stringzilla.hpp>
#include <cassert>
namespace sz = ashvardanian::stringzilla;

int main() {
    sz::string_view family = "👨‍👩‍👧‍👦"; // one emoji, one perceived character

    std::size_t codepoints = 0;
    for (sz_rune_t r : family.utf8_runes()) (void)r, ++codepoints;
    std::size_t graphemes = 0;
    for (sz::string_view g : family.utf8_graphemes()) (void)g, ++graphemes;

    assert(family.size() == 25); // 25 bytes on the wire
    assert(codepoints == 7); // four people joined by three zero-width joiners
    assert(graphemes == 1); // one grapheme cluster
    return 0;
}
```

This is exactly why truncating by bytes or codepoints tears emoji apart — it can slice a skin-tone modifier off its base, or split a flag in half.
Iterating graphemes lets you cut on a real boundary:

```cpp
sz::string_view msg = "🙏🏽👍🏿✨ thanks";
std::size_t cut = 0, seen = 0;
for (sz::string_view g : msg.utf8_graphemes()) { if (seen++ == 3) break; cut += g.size(); }
assert(msg.sub(0, cut) == "🙏🏽👍🏿✨"); // the three emoji keep their skin-tone modifiers
```

Word and sentence boundaries are just as treacherous: contractions, CJK, decimals, and URLs all defeat naive splitting.
UAX-29 keeps `can't` whole while separating CJK ideographs that no space divides, and refuses to break a sentence inside `$9.99`:

```cpp
sz::string_view phrase = "can't stop 北京!";
assert(*phrase.utf8_words().begin() == "can't"); // apostrophe is interior; 北 and 京 split apart

sz::string_view prose = "She paid $9.99. Cheap.";
std::size_t sentences = 0;
for (sz::string_view s : prose.utf8_sentences()) (void)s, ++sentences;
assert(sentences == 2); // the dot inside $9.99 is not a sentence break
```

The full family of ranges, each borrowing from the source and yielding `sz::string_view` segments, or `sz_rune_t` for runes:

- `utf8_runes()` — every codepoint as a decoded `sz_rune_t` UTF-32 scalar.
- `utf8_graphemes()` — UAX-29 grapheme clusters, the user-perceived characters.
- `utf8_words()` — UAX-29 word boundaries.
- `utf8_sentences()` — UAX-29 sentence boundaries.
- `utf8_lines()` — all seven Unicode newline characters plus the CRLF pair.
- `utf8_tokens()` — the Unicode "White_Space" set, skipping empty segments.
- `utf8_linewraps()` — UAX-14 line-break opportunities, both hard breaks and soft wrap points.

The `utf8_words`, `utf8_graphemes`, `utf8_sentences`, and `utf8_linewraps` ranges _tile_ the input — every byte belongs to exactly one segment, with no gaps and no empty slices.
Because every range borrows and walks the buffer once, segmenting a multi-megabyte document into graphemes or words allocates nothing.

## Case Insensitive Search and Folding

Case-insensitive matching across full Unicode is more than ASCII `tolower` — `ß` matches `ss`, accents fold, and the matched byte-length can differ from the needle's.
`utf8_uncased_find` searches case-insensitively _without pre-folding the haystack_: it folds on the fly, so you never allocate a folded copy of a large document just to search it.

```cpp
#include <stringzilla/stringzilla.hpp>
#include <cassert>
namespace sz = ashvardanian::stringzilla;

int main() {
    sz::string_view hay = "Take the STRAßE downtown";
    auto m = hay.utf8_uncased_find("strasse"); // no folded copy of `hay` is made
    assert(m.offset == 9);
    assert(hay.sub(m.offset, m.offset + m.length) == "STRAßE"); // matched the mixed-case ß form
    return 0;
}
```

The result carries both the `offset` and the matched `length`, since folding can make the matched span longer or shorter than the needle.
When you need the folded text itself, fold a `sz::string` in place:

```cpp
sz::string greeting = "Grüße";
greeting.try_utf8_uncased_fold(); // in place; `ß` expands to `ss`
assert(greeting == "grüsse");
```

In C, `sz_utf8_uncased_find` takes a `sz_utf8_uncased_needle_metadata_t *` that it fills on the first call and reuses on later ones, so repeated searches for the same needle skip re-analysis:

```c
#include <stringzilla/stringzilla.h>

sz_utf8_uncased_needle_metadata_t needle = {0}; // cached across searches of the same pattern
sz_size_t matched_length = 0;
sz_cptr_t hit = sz_utf8_uncased_find(haystack, haystack_length, "café", 5, &needle, &matched_length);
// `hit` points at the first case-insensitive match, or NULL; reuse `needle` for the next haystack.
```

## Trimming and Translating

Trimming is exposed in C++ on the view: `strip(byteset)`, `lstrip(byteset)`, and `rstrip(byteset)` return a narrowed view with the matching leading/trailing bytes removed — no copy is made.

```cpp
#include <stringzilla/stringzilla.hpp>
namespace sz = ashvardanian::stringzilla;

int main() {
    sz::string_view padded = "  \t spaced out \n";
    sz::string_view tight = padded.strip(sz::whitespaces_set());
    assert(tight == "spaced out");
    return 0;
}
```

Translation — mapping every byte through a 256-entry lookup table — is offered both at the C ABI level through `sz_lookup`, documented under [Memory Operations](#memory-operations), and in C++ through `sz::lookup`:

```cpp
#include <stringzilla/stringzilla.hpp>
namespace sz = ashvardanian::stringzilla;

int main() {
    sz::basic_look_up_table<char> to_upper;        // identity by default; fill as needed
    for (int i = 'a'; i <= 'z'; ++i) to_upper[(char)i] = (char)(i - 32);

    sz::string text = "hello";
    text.lookup(to_upper); // in place
    assert(text == "HELLO");
    return 0;
}
```

The C side ships ready-made table initializers — `sz_lookup_init_lower`, `sz_lookup_init_upper`, and `sz_lookup_init_ascii` — plus a fast `sz_isascii(text, length)` predicate for selecting ASCII-only fast paths.


## Hashing and Checksums

### C

Hashing comes in three flavors: a byte checksum, a seeded single-shot hash, and an incremental streaming state.
All produce identical output across every backend and platform, in both single-shot and incremental modes.

```c
sz_u64_t sz_bytesum(sz_cptr_t text, sz_size_t length);
sz_u64_t sz_hash(sz_cptr_t text, sz_size_t length, sz_u64_t seed);
void sz_hash_multiseed(sz_cptr_t text, sz_size_t length,
                       sz_u64_t const *seeds, sz_size_t seeds_count, sz_u64_t *hashes);
```

`sz_hash` is non-cryptographic, fast for both short and long inputs, passes the SMHasher `--extra` suite, and uses the AES extensions where present.
`sz_hash_multiseed` hashes one input under many seeds at once — for feature hashing, Count-Min sketches, Bloom filters, and MinHash/LSH — by normalizing the input into AES blocks once and replaying cheap per-seed rounds.

Incremental hashing uses an opaque `sz_hash_state_t`:

```c
void sz_hash_state_init(sz_hash_state_t *state, sz_u64_t seed);
void sz_hash_state_update(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length);
sz_u64_t sz_hash_state_digest(sz_hash_state_t const *state);
sz_bool_t sz_hash_state_equal(sz_hash_state_t const *lhs, sz_hash_state_t const *rhs);
```

A streaming SHA-256 is also provided, producing a 32-byte digest, with hardware backends on x86 SHA-NI, ARM NEON-SHA, and others:

```c
void sz_sha256_state_init(sz_sha256_state_t *state);
void sz_sha256_state_update(sz_sha256_state_t *state, sz_cptr_t data, sz_size_t length);
void sz_sha256_state_digest(sz_sha256_state_t const *state, sz_u8_t digest[32]);
```

The same AES primitives back a reproducible pseudo-random fill, useful with `sz_lookup` for generating random strings over a chosen alphabet:

```c
void sz_fill_random(sz_ptr_t text, sz_size_t length, sz_u64_t nonce);
```

Example combining a single-shot hash, a streamed hash, and a checksum:

```c
#include <stringzilla/stringzilla.h>

int main(void) {
    sz_u64_t single = sz_hash("hello world", 11, 42);

    sz_hash_state_t state;
    sz_hash_state_init(&state, 42);
    sz_hash_state_update(&state, "hello ", 6);
    sz_hash_state_update(&state, "world", 5);
    sz_u64_t streamed = sz_hash_state_digest(&state);
    assert(streamed == single);

    sz_u64_t sum = sz_bytesum("hi", 2);
    assert(sum == 209);
    return 0;
}
```

### C++

In C++, hashing is exposed as methods — `view.hash(seed = 0)`, `view.bytesum()`, and `view.hash_multiseed(seeds, hashes)` — and as the function object `sz::hash{}` together with `sz::equal_to{}`, which are convenient as template arguments for `std::unordered_map`/`std::unordered_set`.
`sz::fill_random(slice, nonce)` fills a mutable span with reproducible noise.

```cpp
#include <stringzilla/stringzilla.hpp>
#include <unordered_map>
#include <unordered_set>
#include <cassert>
namespace sz = ashvardanian::stringzilla;

int main() {
    // Drop the SIMD hash and equality into the STL containers as template arguments.
    std::unordered_map<std::string, int, sz::hash, sz::equal_to> counts;
    for (char const *word : {"apple", "banana", "apple"}) counts[word] += 1;
    assert(counts["apple"] == 2);
    assert(counts.size() == 2);

    // `sz::string` keys also work in the plain containers, via the bundled `std::hash` specialization.
    std::unordered_set<sz::string> uniq = {"apple", "banana"};
    assert(uniq.count("apple") == 1);

    sz::string noise(16, '\0');
    sz::fill_random(noise.span(), /*nonce*/ 7); // reproducible noise
    assert(noise.size() == 16);
    return 0;
}
```

## Sorting and Sequences

These operate on a read-only `sz_sequence_t` and never reorder the underlying data — they fill an `order` permutation array instead.
Sorting is __stable__, so equal elements keep their input order, supports descending order via a `reverse` flag, and supports partial / top-K sorting via `top_count` — pass `0` to sort everything.
Every allocation these kernels need is routed through the `sz_memory_allocator_t` you supply — pass `NULL` for the default `malloc`-based one — so, unlike `std::sort` or `thrust::sort`, no scratch memory is ever allocated behind your back.

### C

The C API fills a caller-owned `order` array and reports success through a `sz_status_t`:

```c
sz_status_t sz_sequence_argsort(
    sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
    sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse);

sz_status_t sz_sequence_argsort_utf8_uncased(
    sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
    sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse);
```

`sz_sequence_argsort` orders byte-lexicographically; `sz_sequence_argsort_utf8_uncased` orders under Unicode case-folding, folding small chunks on the fly, and malformed UTF-8 sorts by raw byte value so the order stays total and deterministic.
The `alloc` argument may be `NULL` to use the default allocator.
The integer-sort core that backs these — `sz_pgrams_sort_serial` and its per-ISA variants — is exposed for benchmarking but is not part of the stable, runtime-dispatched contract.

```c
#include <stringzilla/stringzilla.h>

int main(void) {
    char const *strings[] = {"banana", "apple", "cherry"};
    sz_sequence_t sequence;
    sz_sequence_from_null_terminated_strings(strings, 3, &sequence);

    sz_sorted_idx_t order[3];
    sz_status_t status = sz_sequence_argsort(&sequence, NULL, order, /*top_count*/ 0, sz_false_k);
    assert(status == sz_success_k);
    assert(order[0] == 1 && order[1] == 0 && order[2] == 2); // apple, banana, cherry
    return 0;
}
```

### Set Intersection

`sz_sequence_intersect` computes the intersection of two __deduplicated__ binary string sequences using a hash table, writing matched positions from each side and leaving unmatched slots as `SZ_SIZE_MAX`:

```c
sz_status_t sz_sequence_intersect(
    sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence,
    sz_memory_allocator_t *alloc, sz_u64_t seed, sz_size_t *intersection_size,
    sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions);
```

The `seed` randomizes the hash table to resist adversarial inputs; the position arrays must each fit at least `min(first->count, second->count)` entries.
The companion `sz_sequence_join_semantics_t` enum documents the SQL-style JOIN modes the family is designed around, namely `sz_join_inner_strict_k`, `sz_join_inner_k`, `sz_join_left_outer_k`, `sz_join_right_outer_k`, `sz_join_full_outer_k`, and `sz_join_cross_k`.

```c
char const *first[]  = {"banana", "apple", "cherry"};
char const *second[] = {"cherry", "orange", "pineapple", "banana"};
sz_sequence_t a, b;
sz_sequence_from_null_terminated_strings(first, 3, &a);
sz_sequence_from_null_terminated_strings(second, 4, &b);

sz_size_t intersection_size;
sz_sorted_idx_t a_pos[3], b_pos[3];  // sized to the smaller sequence
sz_status_t status = sz_sequence_intersect(&a, &b, NULL, /*seed*/ 0,
                                           &intersection_size, a_pos, b_pos);
assert(intersection_size == 2); // "banana" and "cherry"
```

### C++

Matching the all-`noexcept` surface, the C++ sort takes any container plus a string-extractor callable and writes the permutation into a caller-owned span, returning a `status_t`:

```cpp
status_t sz::try_argsort(container, extractor, span<sorted_idx_t> order, top_count = 0, reverse = false);
status_t sz::try_argsort_utf8_uncased(container, extractor, span<sorted_idx_t> order, top_count = 0, reverse = false);
expected<size_t, status_t> sz::try_intersect(first, extractor, second, extractor, seed,
                                             span<sorted_idx_t> first_positions, span<sorted_idx_t> second_positions);
```

The outputs are sized `sz::span<sorted_idx_t>` views over caller-owned storage. Size `order` to `container.size()` — the whole span receives a permutation, and a non-zero `top_count` only requests that the first `top_count` entries be fully sorted. Size each position span to `min(first.size(), second.size())`. `try_intersect` returns the number of matched pairs as an `expected<size_t, status_t>`, so there is no separate count out-parameter.

```cpp
#include <stringzilla/stringzilla.hpp>
#include <vector>
#include <cassert>
namespace sz = ashvardanian::stringzilla;

int main() {
    std::vector<sz::string> names = {"banana", "apple", "cherry"};

    sz::sorted_idx_t order[3];
    sz::status_t status = sz::try_argsort(
        names, [](sz::string const &s) -> sz::string_view { return s; }, {order, 3});
    assert(status == sz::status_t::success_k);
    assert(order[0] == 1 && order[1] == 0 && order[2] == 2); // apple, banana, cherry
    return 0;
}
```

For code that accepts exceptions, throwing `sz::argsort` and `sz::argsort_utf8_uncased` convenience overloads return the `std::vector<sorted_idx_t>` directly.

## Memory Operations

### C

The four memory kernels mirror `memcpy`, `memmove`, `memset`, and a lookup-table transform, all writing into the first argument and minimizing unaligned stores:

```c
void sz_copy(sz_ptr_t target, sz_cptr_t source, sz_size_t length); // like memcpy (no overlap)
void sz_move(sz_ptr_t target, sz_cptr_t source, sz_size_t length); // like memmove (overlap ok)
void sz_fill(sz_ptr_t target, sz_size_t length, sz_u8_t value); // like memset
void sz_lookup(sz_ptr_t target, sz_size_t length, sz_cptr_t source, char const lut[256]);
```

`sz_lookup` applies `target[i] = lut[source[i]]`; `target` and `source` may alias but must not partially overlap, and the table must be exactly 256 bytes.

```c
#include <ctype.h>
#include <stringzilla/stringzilla.h>

int main(void) {
    char to_lower[256];
    sz_lookup_init_lower(to_lower);
    char buffer[3] = {'A', 'B', 'C'};
    sz_lookup(buffer, 3, buffer, to_lower);
    assert(buffer[0] == 'a' && buffer[1] == 'b' && buffer[2] == 'c'); // "abc"
    return 0;
}
```

### The `sz_string_t` SSO Container

`sz_string_t` is a tiny memory-owning string with Small-String-Optimization: short strings, up to 22 bytes plus the terminator on 64-bit, live inline with no heap allocation, while longer strings spill to the heap.
Its length can be read branchlessly, and many length changes avoid branches entirely.
Every mutating call takes an explicit `sz_memory_allocator_t *`.

Lifecycle and read-only access:

```c
void sz_string_init(sz_string_t *string);
sz_ptr_t sz_string_init_length(sz_string_t *string, sz_size_t length, sz_memory_allocator_t *allocator);
sz_bool_t sz_string_is_on_stack(sz_string_t const *string);
void sz_string_unpack(sz_string_t const *string, sz_ptr_t *start, sz_size_t *length, sz_size_t *space, sz_bool_t *is_external);
void sz_string_range(sz_string_t const *string, sz_ptr_t *start, sz_size_t *length);
sz_size_t sz_string_length(sz_string_t const *string);
sz_bool_t sz_string_equal(sz_string_t const *a, sz_string_t const *b);
sz_ordering_t sz_string_order(sz_string_t const *a, sz_string_t const *b);
```

Growth and mutation, where all but `sz_string_erase` may allocate and return `SZ_NULL_CHAR` on failure:

```c
sz_ptr_t sz_string_reserve(sz_string_t *string, sz_size_t new_capacity, sz_memory_allocator_t *allocator);
sz_ptr_t sz_string_expand(sz_string_t *string, sz_size_t offset, sz_size_t added_length, sz_memory_allocator_t *allocator);
sz_size_t sz_string_erase(sz_string_t *string, sz_size_t offset, sz_size_t length);
sz_ptr_t sz_string_shrink_to_fit(sz_string_t *string, sz_memory_allocator_t *allocator);
void sz_string_free(sz_string_t *string, sz_memory_allocator_t *allocator);
```

`sz_string_expand` opens an uninitialized gap of `added_length` at `offset` that you then populate, often via `sz_copy`; `sz_string_erase` removes a range without ever allocating and cannot fail.

```c
#include <stringzilla/stringzilla.h>

int main(void) {
    sz_memory_allocator_t alloc;
    sz_memory_allocator_init_default(&alloc);

    sz_string_t s;
    sz_string_init(&s);

    sz_ptr_t room = sz_string_expand(&s, 0, 5, &alloc);  // 5-byte gap at offset 0
    if (!room) return 1;
    sz_copy(room, "hello", 5);

    sz_ptr_t start; sz_size_t length;
    sz_string_range(&s, &start, &length);
    assert(length == 5);

    sz_string_free(&s, &alloc);
    return 0;
}
```

### C++

In C++, `sz::string` is the owning, SSO container and `sz::string_span` is its mutable view; both rebuild the `std::string` surface.
Mutating members (`append`, `push_back`, `insert`, `erase`, `replace`, `resize`, `reserve`, `shrink_to_fit`, `clear`, `swap`, `operator+=`, `operator+`, `assign`) are present, alongside `data`/`c_str`/`size`/`capacity` and the relational operators.
Every potentially-allocating mutation also has a non-throwing `try_*` twin — `try_append`, `try_reserve`, `try_resize`, `try_assign`, `try_insert`, `try_replace`, `try_replace_all`, and so on — that returns a `bool`/`size_type` instead of throwing, for allocation-failure-aware code.

```cpp
#include <stringzilla/stringzilla.hpp>
namespace sz = ashvardanian::stringzilla;

int main() {
    sz::string greeting = "hello";
    greeting.append(", ");
    greeting += "world";
    assert(greeting == "hello, world");
    greeting.push_back('!');

    if (!greeting.try_reserve(1024)) return 1;   // non-throwing growth
    greeting.replace_all("l", "L");              // via try_replace_all under the hood

    assert(greeting.starts_with("heLLo"));
    return 0;
}
```

## Runtime Dispatch and Capabilities

Every kernel exists in a serial form plus one or more per-ISA forms, named with a backend suffix (`_serial`, `_westmere`, `_haswell`, `_skylake`, `_icelake`, `_neon`, `_neonaes`, `_neonsha`, `_sve`, `_sve2`, `_sve2aes`, `_v128`, `_v128relaxed`, `_rvv`, `_lasx`, `_powervsx`).
The public, suffix-free name such as `sz_find` or `sz_hash` resolves to the best available backend.

- __Header-only mode, `SZ_DYNAMIC_DISPATCH=0`, the default.__ The dispatch is resolved at compile time: the umbrella header picks the most advanced backend enabled by your compiler flags, and the suffix-free functions inline straight to it. No runtime indirection.
- __Dynamic-dispatch mode, `SZ_DYNAMIC_DISPATCH=1`.__ All backends are compiled and a dispatch table is initialized once, then the suffix-free functions jump through it. This is how the prebuilt shared library is shipped, so a single binary runs optimally on any CPU. `sz_dynamic_dispatch()` returns non-zero in this mode.

Capabilities are introspectable at both compile and run time. `sz_capability_t` is a bitmask whose flags include `sz_cap_serial_k` and `sz_cap_parallel_k`, the x86 tiers `sz_cap_westmere_k`, `sz_cap_goldmont_k`, `sz_cap_haswell_k`, `sz_cap_skylake_k`, and `sz_cap_icelake_k`, the ARM tiers `sz_cap_neon_k`, `sz_cap_neonaes_k`, `sz_cap_neonsha_k`, `sz_cap_sve_k`, `sz_cap_sve2_k`, and `sz_cap_sve2aes_k`, the portable-SIMD tiers `sz_cap_v128_k`, `sz_cap_v128relaxed_k`, `sz_cap_rvv_k`, `sz_cap_rvvcrypto_k`, `sz_cap_lasx_k`, and `sz_cap_powervsx_k`, and the GPU tiers `sz_cap_cuda_k`, `sz_cap_kepler_k`, and `sz_cap_hopper_k`.

```c
sz_capability_t sz_capabilities(void); // intersection of compile-time and runtime
sz_capability_t sz_capabilities_comptime(void); // what this build was compiled for
sz_capability_t sz_capabilities_runtime(void); // what this CPU supports
sz_cptr_t sz_capabilities_to_string(sz_capability_t caps); // e.g. "serial,haswell,skylake"
int sz_dynamic_dispatch(void);
```

The runtime probe inspects CPUID on x86, the AArch64 ID registers on ARM with a `SIGILL`-guarded `mrs` fallback to NEON-only, and `getauxval`/`riscv_hwprobe` on RISC-V; WebAssembly, LoongArch, and Power report their compile-time capabilities.

```c
#include <stdio.h>
#include <stringzilla/stringzilla.h>

int main(void) {
    sz_capability_t caps = sz_capabilities();
    printf("StringZilla %d.%d.%d, backends: %s, dynamic=%d\n",
           sz_version_major(), sz_version_minor(), sz_version_patch(),
           sz_capabilities_to_string(caps), sz_dynamic_dispatch());
    return 0;
}
```

You can also call any backend directly when you have already established the target supports it — for example `sz_find_haswell(...)` or `sz_hash_icelake(...)` — which is handy for benchmarking and for pinning a code path in a controlled environment.
