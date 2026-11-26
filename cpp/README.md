# StringZilla for C++

The C++ SDK rebuilds the STL `<string>` and `<string_view>` surface on top of the [C library](../c/README.md), adding an owning Small-String-Optimized container, __lazy, range-compatible__ splitting and partitioning views, and free functions for hashing, sorting, and translation, over an all-`noexcept` surface.
Its headers live in `cpp/stringzilla/`, `<stringzilla/stringzilla.hpp>` includes them all, and everything lives in `namespace ashvardanian::stringzilla`, conventionally aliased as `sz`.
Every fallible call returns an `sz::status_t` or an `sz::expected`, and never throws.

## Installation

C++ code links `stringzilla::cpp`, which adds `cpp/` as an include root beside the `c/` of `stringzilla::abi`, requires C++20, and links `stringzilla::static`:

```cmake
include(FetchContent)
FetchContent_Declare(
    stringzilla
    GIT_REPOSITORY https://github.com/ashvardanian/StringZilla.git
    GIT_TAG main)
FetchContent_MakeAvailable(stringzilla)

target_link_libraries(your_app PRIVATE stringzilla::cpp)
```

The [C guide](../c/README.md#installation) covers the shared, header-only and installed variants.
In header-only builds the dispatch points report `sz_missing_library_k`, so the C++ layer runs the serial kernels over its default mask, which is empty there.
`STRINGZILLA_WITH_STL=0` drops the conversions from and to `std::string`, as the [compilation settings](../include/stringzilla/README.md#compilation-settings-and-debugging) describe.

## Quickstart

```cpp
#include <stringzilla/stringzilla.hpp>
namespace sz = ashvardanian::stringzilla;

int main() {
    sz::string_view_t text = "the quick brown fox";
    return text.find("brown") == 10 ? 0 : 1;
}
```

## Types

The C++ layer is built on the types below plus a 256-bit `byteset`.

- `sz::string_slice<char_type>` — a non-owning, length-bounded view over `char` or `char const`.
  Two aliases exist: the read-only `sz::string_view_t = string_slice<char const>` and the mutable `sz::string_span_t = string_slice<char>`.
- `sz::string<allocator = std::allocator<char>>` — an owning, Small-String-Optimized `char` container; the alias `sz::string_t` uses the default `std::allocator`.
  A stateful allocator travels with the string under the usual `std::allocator_traits` propagation rules, and an empty one costs no bytes.
- `sz::strings_view_t` — a non-owning column of strings, wrapping a `sz_strings_view_t` in any of its layouts.
- `sz::strings<allocator = sz::heap_alloc<char>>` — an owned, move-only column, a `sz_strings_span_t` tape and the allocator of its blocks; the alias `sz::strings_t` uses the default.
- `sz::byteset_t` — the C++ wrapper around `sz_byteset_t`, constructible from an initializer list, a `(pointer, count)` pair, or a `std::array`, and composable with `operator|`.
- `sz::status_t` — a scoped enum mirroring `sz_status_t` with `success_k`, `bad_alloc_k`, `invalid_utf8_k`, `contains_duplicates_k`, and the rest, returned by the fallible free functions and the string's allocating verbs.

`sz::string_view_t` is implicitly constructible from a `char const *`, a `(pointer, length)` pair, `std::string`, and `std::string_view`, and converts back to both; `sz::string_t` adds construction from repeats, sub-ranges, initializer lists, and lazy `sz::concatenate(...)` expressions.
Both satisfy the `std::string`/`std::string_view` member surface — `data`, `size`, `length`, `empty`, `at`, `front`, `back`, iterators, `substr`, `compare`, the relational operators, and on C++20 `operator<=>` — so they are drop-in for most code.
Both also stream through `operator<<`, and `std::hash<sz::string_t>` / `std::hash<sz::string_view_t>` specializations are provided so they work as unordered-container keys out of the box.

User-defined literals live in `sz::literals`:

```cpp
using namespace sz::literals;
sz::string_view_t v = "needle"_sv;   // a view, no allocation
sz::byteset_t s = " \t\n\r"_bs;      // a byte-set from a literal
```

Convenience byte-sets are available as free functions: `sz::whitespaces_set()`, `sz::newlines_set()`, `sz::digits_set()`, `sz::hexdigits_set()`, `sz::octdigits_set()`, and `sz::ascii_letters_set()`.

## Searching and Counting

The view and the owning string expose `find`/`rfind` for sub-strings, single characters, and byte-sets, plus `contains`, `starts_with`, `ends_with`, and the full `find_first_of` / `find_last_of` / `find_first_not_of` / `find_last_not_of` family taking either a `byteset` or a view.
For repeated matching the view offers lazy, allocation-free ranges: `find_all(needle)` and `rfind_all(needle)`, each with `sz::include_overlaps` / `sz::exclude_overlaps` tag overloads, alongside `find_all(byteset)` and `rfind_all(byteset)`.
An empty needle matches at every offset from 0 to `size()` inclusive, so it yields `size() + 1` zero-width matches.

```cpp
#include <stringzilla/stringzilla.hpp>
namespace sz = ashvardanian::stringzilla;

int main() {
    sz::string_view_t haystack = "the quick brown fox";

    auto pos = haystack.find("quick");           // index, or sz::string_view_t::npos
    assert(pos == 4);
    assert(haystack.contains("brown"));
    assert(haystack.starts_with("the"));
    assert(haystack.ends_with("fox"));

    std::size_t words = 0;
    for (auto word : haystack.find_all(sz::whitespaces_set())) (void)word, ++words;

    auto vowels = sz::byteset_t {"aeiou", 5};
    auto first_vowel = haystack.find_first_of(vowels);
    assert(first_vowel == 2); // index of 'e'
    return 0;
}
```

The view also offers character-class predicates — `is_alpha`, `is_alnum`, `is_ascii`, `is_digit`, `is_lower`, `is_upper`, `is_space`, `is_printable`, and the general `contains_only(byteset)`.

## Splitting and Partitioning

This is a C++-only convenience layer; the C ABI provides the underlying scans `sz_find_best` and `sz_find_byteset_best` that you compose by hand.

`partition` and `rpartition` return a three-way `sz::string_partition_result` struct with `before`, `match`, and `after` members around the first or last occurrence of a pattern.
The pattern can be a view, a single character, or a byte-set.

`split`, `rsplit`, and `splitlines` return lazy ranges of `string_view`s.
`split(view)` yields a `sz::find_splits_view`, `rsplit(view)` yields a `sz::rfind_splits_view`, and the byte-set overloads `split(byteset)` and `rsplit(byteset)` yield the matching character-splitter views.
`split(byteset)` and `rsplit(byteset)` default to whitespace, and `splitlines()` defaults to the newline set.
A trailing separator yields a final empty segment, and an empty separator never splits, yielding the whole text as the only segment.

The key property is that these ranges are _lazy_ — they view the source buffer and walk it one `string_view` at a time, so the next boundary is computed only when you ask for it.
Splitting a large document into millions of tokens therefore costs __no per-token allocation__: each iteration step advances a cursor and returns a view into the source.
This is the decisive contrast with the idiomatic STL approach, where splitting builds a `std::vector<std::string>` and heap-allocates a fresh string for every element, copying its bytes out of the source.

```cpp
#include <stringzilla/stringzilla.hpp>
namespace sz = ashvardanian::stringzilla;

int main() {
    sz::string_view_t csv = "alice,30,engineer";

    sz::string_partition_result<sz::string_view_t> first = csv.partition(',');
    sz::string_view_t name = first.before;
    assert(name == "alice");
    assert(first.match == ",");
    assert(first.after == "30,engineer");

    std::size_t fields = 0;
    for (sz::string_view_t field : csv.split(",")) // "alice", "30", "engineer"
        ++fields;                                // no string is allocated per field
    assert(fields == 3);

    sz::string_view_t doc = "line one\nline two\nline three";
    for (sz::string_view_t line : doc.splitlines()) (void)line;

    return 0;
}
```

Each range exposes `begin`/`end`, dereferences to a `sz::string_view_t`, and can be consumed in a single pass.
When you genuinely need the tokens to outlive the source you can still materialize them into a `std::vector<sz::string_view_t>`, but only then do you pay for storage — the default path stays allocation-free.

## Unicode Segmentation

Splitting Unicode text into the units a human actually perceives is deceptively hard, and getting it wrong corrupts data.
A single emoji can be dozens of bytes and several codepoints yet one perceived character, CJK runs carry no spaces, and a period is not always the end of a sentence.
StringZilla ships the real algorithms — UAX-29 graphemes, words, and sentences, plus UAX-14 line-break opportunities — as lazy ranges that view the source and yield one `sz::string_view_t` per segment with no per-segment allocation.

A string's length is really three different numbers, and only the grapheme count matches what a user sees:

```cpp
#include <stringzilla/stringzilla.hpp>
#include <cassert>
namespace sz = ashvardanian::stringzilla;

int main() {
    sz::string_view_t family = "👨‍👩‍👧‍👦"; // one emoji, one perceived character

    std::size_t codepoints = 0;
    for (sz_rune_t r : family.utf8_runes()) (void)r, ++codepoints;
    std::size_t graphemes = 0;
    for (sz::string_view_t g : family.utf8_graphemes()) (void)g, ++graphemes;

    assert(family.size() == 25); // 25 bytes on the wire
    assert(codepoints == 7); // four people joined by three zero-width joiners
    assert(graphemes == 1); // one grapheme cluster
    return 0;
}
```

This is exactly why truncating by bytes or codepoints tears emoji apart — it can slice a skin-tone modifier off its base, or split a flag in half.
Iterating graphemes lets you cut on a real boundary:

```cpp
sz::string_view_t msg = "🙏🏽👍🏿✨ thanks";
std::size_t cut = 0, seen = 0;
for (sz::string_view_t g : msg.utf8_graphemes()) { if (seen++ == 3) break; cut += g.size(); }
assert(msg.sub(0, cut) == "🙏🏽👍🏿✨"); // the three emoji keep their skin-tone modifiers
```

Word and sentence boundaries are just as treacherous: contractions, CJK, decimals, and URLs all defeat naive splitting.
UAX-29 keeps `can't` whole while separating CJK ideographs that no space divides, and refuses to break a sentence inside `$9.99`:

```cpp
sz::string_view_t phrase = "can't stop 北京!";
assert(*phrase.utf8_wordbreaks().begin() == "can't"); // apostrophe is interior; 北 and 京 split apart

sz::string_view_t prose = "She paid $9.99. Cheap.";
std::size_t sentences = 0;
for (sz::string_view_t s : prose.utf8_sentences()) (void)s, ++sentences;
assert(sentences == 2); // the dot inside $9.99 is not a sentence break
```

The full family of ranges, each viewing the source and yielding `sz::string_view_t` segments, or `sz_rune_t` for runes.
Naming follows one rule: the bare name (`newlines`/`whitespaces`/`delimiters`) yields the __separators__ the kernel finds, while `split_*` yields the content __between__ them:

- `utf8_runes()` — every codepoint as a decoded `sz_rune_t` UTF-32 scalar, with U+FFFD for ill-formed bytes and a truncated tail.
- `utf8_graphemes()` — UAX-29 grapheme clusters, the user-perceived characters.
- `utf8_wordbreaks()` — all UAX-29 word segments (words and the separators between them).
- `utf8_sentences()` — UAX-29 sentence boundaries.
- `utf8_linebreaks()` — UAX-14 line-break opportunities, both hard breaks and soft wrap points.
- `utf8_split_newlines()` / `utf8_newlines()` — content between the seven Unicode newlines (CRLF as one), or the newline runs themselves.
- `utf8_split_whitespaces()` / `utf8_whitespaces()` — content between Unicode "White_Space" runs (tokens), or the whitespace runs.
- `utf8_split_delimiters()` / `utf8_delimiters()` — content between any Unicode punctuation/symbol/separator, or the delimiter runs.

On any `split_*` range, `.skip_empty()` drops empty segments and `.with_separators()` interleaves segments and separators losslessly (concatenation reconstructs the input).
The `utf8_wordbreaks`, `utf8_graphemes`, `utf8_sentences`, and `utf8_linebreaks` ranges _tile_ the input — every byte belongs to exactly one segment, with no gaps and no empty slices.
Because every range views the buffer and walks it once, segmenting a multi-megabyte document into graphemes or words allocates nothing.
The [C guide](../c/README.md#unicode-segmentation) describes the batched kernels underneath.

## Case Folding and Normalization

Case-insensitive matching across full Unicode is more than ASCII `tolower` — `ß` matches `ss`, accents fold, and the matched byte-length can differ from the needle's.
`utf8_uncased_search` searches case-insensitively _without pre-folding the haystack_: it folds on the fly, so you never allocate a folded copy of a large document just to search it.

```cpp
#include <stringzilla/stringzilla.hpp>
#include <cassert>
namespace sz = ashvardanian::stringzilla;

int main() {
    sz::string_view_t hay = "Take the STRAßE downtown";
    auto m = hay.utf8_uncased_search("strasse"); // no folded copy of `hay` is made
    assert(m.offset == 9);
    assert(hay.sub(m.offset, m.offset + m.length) == "STRAßE"); // matched the mixed-case ß form
    return 0;
}
```

The result carries both the `offset` and the matched `length`, since folding can make the matched span longer or shorter than the needle.
To search many haystacks for one needle, prepare it once as a `sz::utf8_uncased_needle_t`, which points at the needle's bytes, so they must outlive it, and pass it to `utf8_uncased_search` or to `utf8_uncased_matches`, the lazy range of every non-overlapping match.
Matches sit on codepoint boundaries, so an empty needle matches at every boundary, the end included.
When you need the folded text itself, fold a `sz::string_t` in place:

```cpp
sz::string_t greeting;
if (sz::failed(greeting.assign("Grüße"))) return 1;
if (sz::failed(greeting.utf8_uncased_fold())) return 1; // in place; `ß` expands to `ss`
assert(greeting == "grüsse");
```

## Trimming and Translating

Trimming is exposed in C++ on the view: `strip(byteset)`, `lstrip(byteset)`, and `rstrip(byteset)` return a narrowed view with the matching leading/trailing bytes removed — no copy is made.

```cpp
#include <stringzilla/stringzilla.hpp>
namespace sz = ashvardanian::stringzilla;

int main() {
    sz::string_view_t padded = "  \t spaced out \n";
    sz::string_view_t tight = padded.strip(sz::whitespaces_set());
    assert(tight == "spaced out");
    return 0;
}
```

Translation — mapping every byte through a 256-entry lookup table — is offered both at the C ABI level through `sz_strings_lookup_best`, documented in the [C guide](../c/README.md#string-collections-and-containers), and in C++ through `sz::lookup`:

```cpp
#include <stringzilla/stringzilla.hpp>
namespace sz = ashvardanian::stringzilla;

int main() {
    sz::look_up_table_t to_upper = sz::look_up_table_t::identity();
    for (int i = 'a'; i <= 'z'; ++i) to_upper[(char)i] = (char)(i - 32);

    sz::string_t text;
    if (sz::failed(text.assign("hello"))) return 1;
    text.lookup(to_upper); // in place
    assert(text == "HELLO");
    return 0;
}
```

## Hashing and Checksums

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

    // `sz::string_t` keys also work in the plain containers, via the bundled `std::hash` specialization.
    std::unordered_set<sz::string_t> uniq = {"apple", "banana"};
    assert(uniq.count("apple") == 1);

    sz::string_t noise(16, '\0');
    sz::fill_random(noise.span(), /*nonce*/ 7); // reproducible noise
    assert(noise.size() == 16);
    return 0;
}
```

## Sorting and Sets

Sorting is stable and every scratch allocation comes from the allocator you pass, as the [C guide](../c/README.md#sorting-and-sets) describes.
Matching the all-`noexcept` surface, the C++ sort takes any container plus a string-extractor callable and writes the permutation into a caller-owned span, returning a `status_t`:

```cpp
status_t sz::argsort(container, extractor, span<sz_size_t> order, top_count = 0, reverse = false,
                     capabilities = sz::default_capabilities(), stream = nullptr, allocator = {});
status_t sz::argsort_utf8_uncased(container, extractor, span<sz_size_t> order, top_count = 0, reverse = false,
                                  capabilities = sz::default_capabilities(), stream = nullptr, allocator = {});
expected<size_t> sz::intersect(first, extractor, second, extractor, seed,
                               span<sz_size_t> first_positions, span<sz_size_t> second_positions,
                               capabilities = sz::default_capabilities(), stream = nullptr, allocator = {});
```

The outputs are sized `std::span<sz_size_t>` views over caller-owned storage, and the scratch memory comes from `allocator`.
Size `order` to `container.size()` — the whole span receives a permutation, and a non-zero `top_count` only requests that the first `top_count` entries be fully sorted.
Size each position span to `min(first.size(), second.size())`; a shorter span returns `unexpected_dimensions_k`.
`intersect` returns the number of matched pairs as an `expected<size_t>`, so there is no separate count out-parameter.

```cpp
#include <stringzilla/stringzilla.hpp>
#include <vector>
#include <cassert>
namespace sz = ashvardanian::stringzilla;

int main() {
    std::vector<sz::string_t> names = {"banana", "apple", "cherry"};

    sz_size_t order[3];
    sz::status_t status = sz::argsort(
        names, [](sz::string_t const &s) -> sz::string_view_t { return s; }, {order, 3});
    assert(status == sz::status_t::success_k);
    assert(order[0] == 1 && order[1] == 0 && order[2] == 2); // apple, banana, cherry

    auto [sorted, sort_status] = sz::argsort(names);
    assert(sz::succeeded(sort_status) && sorted[0] == 1);
    return 0;
}
```

Without output spans, `sz::argsort`, `sz::argsort_utf8_uncased` and `sz::intersect` allocate their results from the allocator passed last and return them in an `expected`, never throwing.

## String Collections and Containers

In C++, `sz::string_t` is the owning, SSO container and `sz::string_span_t` is its mutable view; both rebuild the `std::string` surface, and nothing in them throws.
Mutating members (`append`, `push_back`, `insert`, `erase`, `replace`, `resize`, `reserve`, `shrink_to_fit`, `clear`, `swap`, `assign`) are present, alongside `data`/`c_str`/`size`/`capacity` and the relational operators.
Every allocating verb returns a `sz::status_t`, or an `sz::expected` beside an iterator, and `insert_at`, `erase_slice` and `replace_slice` take signed offsets counted from the end, like Python's slices.
Copies and conversions come from `copy` and `make`, which return an `sz::expected`, and positions past the end clamp to it.

```cpp
#include <stringzilla/stringzilla.hpp>
namespace sz = ashvardanian::stringzilla;

int main() {
    auto [greeting, status] = sz::string_t::make("hello");
    if (sz::failed(status)) return 1;
    if (sz::failed(greeting.append(", world"))) return 1;
    assert(greeting == "hello, world");

    if (sz::failed(greeting.push_back('!'))) return 1; // allocating verbs return a status
    if (sz::failed(greeting.reserve(1024))) return 1;
    if (sz::failed(greeting.replace_all("l", "L"))) return 1;

    assert(greeting.starts_with("heLLo"));
    return 0;
}
```

Collections pair the same way: `sz::strings_view_t` views an Arrow `String` or `LargeString` array from `(data, offsets, count)`, a span of `sz::string_view_t`, or an owned column, and `sz::strings_t` owns one.
Its conversions, `gather`, `escape`, `unescape`, `lookup` and `normalize`, return a fresh `sz::strings` inside an `sz::expected` with the status, and leave nothing allocated on failure.
`find`, `hashes` and `to_rows` write into spans the caller owns.
Every one takes a capability mask, `sz::default_capabilities()` unless given, and a stream, like the C dispatch points.

```cpp
#include <stringzilla/stringzilla.hpp>
namespace sz = ashvardanian::stringzilla;

int main() {
    char const payload[] = "alphabetagamma";
    sz_u32_t const offsets[] = {0, 5, 9, 14};
    sz::strings_view_t const column(payload, offsets, 3); // an Arrow `String` array, no copies

    auto const [escaped, escape_status] = column.escape(sz_escape_json_k);
    if (sz::failed(escape_status)) return 1;

    std::size_t malformed_index = 0;
    auto const [unescaped, unescape_status] = escaped.view().unescape(sz_escape_json_k, malformed_index);
    assert(sz::succeeded(unescape_status) && unescaped[1] == "beta");

    std::size_t positions[3];
    if (sz::failed(column.find("a", positions))) return 1; // the first match in every string
    return 0;
}
```

## Capabilities & Devices

Every wrapper that takes a mask, like `sz::lookup` or `sz::argsort`, defaults to `sz::default_capabilities()`, the enabled CPU mask, while `sz::device_t::make(kind, ordinal)` opens one GPU to ask its `capabilities_enabled()`, and `sz::device_t::cpu().configure_thread(mask)` wraps `sz_configure_thread_cpu`.
The [C guide](../c/README.md#runtime-dispatch) describes the masks and the capabilities they hold.

## What's Wrong with the C++ Standard Library?

| C++ Code                             | Evaluation Result | Invoked Signature              |
| :----------------------------------- | :---------------- | :----------------------------- |
| `"Loose"s.replace(2, 2, "vath"s, 1)` | `"Loathe"` 🤢     | `(pos1, count1, str2, pos2)`   |
| `"Loose"s.replace(2, 2, "vath", 1)`  | `"Love"` 🥰       | `(pos1, count1, str2, count2)` |

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

[stl-replace]: https://en.cppreference.com/w/cpp/string/string/replace

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

```cpp
using str = sz::string_view_t;

str("a:b").front(1) == "a"; // no checks, unlike `substr`
str("a:b").front(2) == "a:"; // take first 2 characters
str("a:b").back(-1) == "b"; // accepting negative indices
str("a:b").back(-2) == ":b"; // similar to Python's `"a:b"[-2:]`
str("a:b").sub(1, -1) == ":"; // similar to Python's `"a:b"[1:-1]`
str("a:b").sub(-2, -1) == ":"; // similar to Python's `"a:b"[-2:-1]`
str("a:b").sub(-2, 1) == ""; // similar to Python's `"a:b"[-2:1]`
"a:b"_sv[{-2, -1}] == ":"; // works on views and overloads `operator[]`
```

The C++ layer is a set of headers, so you can use the full API in some translation units and gradually transition to safer restricted API in others.
Bonus - all the bound checking is branchless, so it has a constant cost and won't hurt your branch predictor.

## Beyond the C++ Standard Library, Learning from Python

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
auto [before, match, after] = haystack.partition(sz::byteset_t(":;", 2)); // Character-set argument
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

- [Splits and Ranges](#splitting-and-partitioning).
- Concatenating strings without allocations.
- Random generation.
- Edit distances and fuzzy search.

Some of the StringZilla interfaces are not available even Python's native `str` class.
Here is a sneak peek of the most useful ones.

```cpp
text.hash(); // -> 64 bit unsigned integer 
text.ssize(); // -> 64 bit signed length to avoid `static_cast<std::ssize_t>(text.size())`
text.contains_only(sz::byteset_t(" \t", 2)); // == text.find_first_not_of(sz::byteset_t(" \t", 2)) == npos;
text.contains(' '); // == text.find(' ') != npos;

// Simpler slicing than `substr`
text.front(10); // -> sz::string_view_t
text.back(10); // -> sz::string_view_t

// Safe variants, which clamp the range into the string bounds
using sz::string_t::cap;
text.front(10, cap) == text.front(std::min(10, text.size()));
text.back(10, cap) == text.back(std::min(10, text.size()));

// Character set filtering
text.lstrip(sz::whitespaces_set()).rstrip(sz::newlines_set()); // like Python
text.front(sz::whitespaces_set()); // all leading whitespaces
text.back(sz::digits_set()); // all numerical symbols forming the suffix

// Incremental construction
using sz::string_t::unchecked;
text.push_back('x'); // returns `bad_alloc_k` instead of throwing if growing fails
text.push_back('x', unchecked); // no bounds checking, Rust style

sz::concatenate(text, "@", domain, ".", tld); // No allocations
```
