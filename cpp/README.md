# StringZilla for C++

The C++ SDK rebuilds the STL `<string>` and `<string_view>` surface on top of the [C library](../c/README.md), adding an owning Small-String-Optimized container, __lazy, range-compatible__ splitting and partitioning views, and free functions for search, hashing, sorting, and translation, over an all-`noexcept` surface.
The containers live in `include/stringzilla/` and call no kernel, so any code may use them header-only.
The verbs live in `cpp/stringzilla/`, one header per C family, and link the library.
`<stringzilla/stringzilla.hpp>` includes them all, and everything lives in `namespace ashvardanian::stringzilla`, conventionally aliased as `sz`.
Every fallible call returns an `sz::status_t` or an `sz::expected`, and never throws.
Batch calls and engine verbs forward to their C twins, and the references that check the kernels live in [`test/`](../test/README.md), never here.

## Installation

C++ code links `stringzilla::cpp`, which adds `cpp/` as an include root beside the `c/` and `include/` of `stringzilla::abi`, requires C++20, and links `stringzilla::static`:

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
A header-only build may include the containers of `include/stringzilla/`, while every header of `cpp/stringzilla/` stops it with an `#error`, as the verbs link the library.

## Quickstart

```cpp
#include <stringzilla/stringzilla.hpp>
namespace sz = ashvardanian::stringzilla;

int main() {
    sz::string_view_t text = "the quick brown fox";
    return sz::find(text, "brown") == 10 ? 0 : 1;
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

`sz::string_view_t` is implicitly constructible from a `char const *`, a `(pointer, length)` pair, `std::string`, and `std::string_view`, and converts back to both; `sz::string_t::make` builds an owning string from a view or a lazy `sz::concatenate(...)` expression, and `copy()` duplicates one, both returning an `sz::expected`.
Both keep the `std::string`/`std::string_view` members no kernel backs — `data`, `size`, `length`, `empty`, `at`, `front`, `back`, iterators, `substr`, `compare`, the relational operators, and `operator<=>`, which order bytes through `std::char_traits`.
Both also stream through `operator<<`, and `<stringzilla/hash.hpp>` specializes `std::hash<sz::string_t>` / `std::hash<sz::string_view_t>`, so they work as unordered-container keys.

Every operation a kernel of the library backs is a free function taking any string, from StringZilla and STL strings to string literals, and its member spelling is deleted, with a comment naming the function.
These members are gone, each replaced by the free function of the same name with the string first, like `sz::find(text, "x")` for `text.find("x")`:

- `find`, `rfind`, `contains`, `count`, `find_first_of`, `find_first_not_of`, `find_last_of`, `find_last_not_of`, `contains_only`, the `is_alpha` family, `partition`, `rpartition`, `lstrip`, `rstrip`, `strip`, `find_all`, `rfind_all`, `split` and `rsplit` in `<stringzilla/find.hpp>`, where `split(byteset)` is `sz::split_characters` and `splitlines()` is `sz::split_characters(text, sz::newlines_set())`;
- `starts_with`, `ends_with`, and the matching `remove_prefix` and `remove_suffix` in `<stringzilla/compare.hpp>`;
- `hash`, `hash_multiseed`, `bytesum` and `fill_random` in `<stringzilla/hash.hpp>`, with `sz::hash_t` and the `std::hash` specializations;
- `utf8_count`, `utf8_seek`, `utf8_runes` and the other `utf8_*` ranges in `<stringzilla/utf8_runes.hpp>` and its siblings;
- `utf8_find_denormalized`, `is_normalized`, `utf8_normalize`, `utf8_uncased_*` and `utf8_uncased_fold` in `<stringzilla/utf8_norm.hpp>` and `<stringzilla/utf8_uncased.hpp>`;
- `replace_all` in `<stringzilla/find.hpp>`, and `lookup` in `<stringzilla/strings_verbs.hpp>`, both taking the string to change in place;
- the column members `find`, `hashes`, `utf8_count`, `to_rows`, `gather`, `escape`, `unescape`, `lookup`, `normalize` and `join` in `<stringzilla/strings_verbs.hpp>`, where `strings::split` is `sz::split_into` and `strings::split_lines` is `sz::split_lines`.

A verb over the default mask returns its plain result, and the overload taking an explicit capability mask and stream returns an `sz::expected`.
The lazy ranges keep only views, so they take lvalues and borrowed views, and refuse an owning temporary like `std::string("a b")` at compile time.

User-defined literals live in `sz::literals`:

```cpp
using namespace sz::literals;
sz::string_view_t v = "needle"_sv;   // a view, no allocation
sz::byteset_t s = " \t\n\r"_bs;      // a byte-set from a literal
```

Convenience byte-sets are available as free functions: `sz::whitespaces_set()`, `sz::newlines_set()`, `sz::digits_set()`, `sz::hexdigits_set()`, `sz::octdigits_set()`, and `sz::ascii_letters_set()`.

## Searching and Counting

`sz::find` and `sz::rfind` search any string for sub-strings, single characters, and byte-sets, beside `sz::contains`, `sz::starts_with`, `sz::ends_with`, and the full `find_first_of` / `find_last_of` / `find_first_not_of` / `find_last_not_of` family taking either a `byteset` or a string of bytes.
For repeated matching there are lazy, allocation-free ranges: `sz::find_all(text, needle)` and `sz::rfind_all(text, needle)`, each with `sz::include_overlaps` / `sz::exclude_overlaps` tag overloads, alongside `sz::find_all_characters` and `sz::rfind_all_characters` over a byte-set.
An empty needle matches at every offset from 0 to `size()` inclusive, so it yields `size() + 1` zero-width matches.

```cpp
#include <stringzilla/stringzilla.hpp>
namespace sz = ashvardanian::stringzilla;

int main() {
    sz::string_view_t haystack = "the quick brown fox";

    auto pos = sz::find(haystack, "quick"); // index, or sz::string_view_t::npos
    assert(pos == 4);
    assert(sz::contains(haystack, "brown"));
    assert(sz::starts_with(haystack, "the"));
    assert(sz::ends_with(haystack, "fox"));

    std::size_t words = 0;
    for (auto word : sz::find_all_characters(haystack, sz::whitespaces_set())) (void)word, ++words;

    auto vowels = sz::byteset_t {"aeiou", 5};
    auto first_vowel = sz::find_first_of(haystack, vowels);
    assert(first_vowel == 2); // index of 'e'
    return 0;
}
```

Character-class predicates take any string too — `sz::is_alpha`, `is_alnum`, `is_ascii`, `is_digit`, `is_lower`, `is_upper`, `is_space`, `is_printable`, and the general `sz::contains_only(text, byteset)`.

## Splitting and Partitioning

This is a C++-only convenience layer; the C ABI provides the underlying scans `sz_find_best` and `sz_find_byteset_best` that you compose by hand.

`sz::partition` and `sz::rpartition` return a three-way `sz::string_partition_result` struct with `before`, `match`, and `after` members around the first or last occurrence of a pattern.
The pattern can be a view, a single character, or a byte-set.

`sz::split` and `sz::rsplit` return lazy ranges of `string_view`s.
`sz::split(text, needle)` yields a `sz::find_splits_view`, `sz::rsplit(text, needle)` yields a `sz::rfind_splits_view`, and `sz::split_characters(text, set)` and `sz::rsplit_characters(text, set)` yield the matching character-splitter views, like `sz::split_characters(text, sz::newlines_set())` for lines.
A trailing separator yields a final empty segment, and an empty separator never splits, yielding the whole text as the only segment.

The key property is that these ranges are _lazy_ — they view the source buffer and walk it one `string_view` at a time, so the next boundary is computed only when you ask for it.
Splitting a large document into millions of tokens therefore costs __no per-token allocation__: each iteration step advances a cursor and returns a view into the source.
This is the decisive contrast with the idiomatic STL approach, where splitting builds a `std::vector<std::string>` and heap-allocates a fresh string for every element, copying its bytes out of the source.

```cpp
#include <stringzilla/stringzilla.hpp>
namespace sz = ashvardanian::stringzilla;

int main() {
    sz::string_view_t csv = "alice,30,engineer";

    sz::string_partition_result<sz::string_view_t> first = sz::partition(csv, ',');
    sz::string_view_t name = first.before;
    assert(name == "alice");
    assert(first.match == ",");
    assert(first.after == "30,engineer");

    std::size_t fields = 0;
    for (sz::string_view_t field : sz::split(csv, ",")) // "alice", "30", "engineer"
        ++fields;                                     // no string is allocated per field
    assert(fields == 3);

    sz::string_view_t doc = "line one\nline two\nline three";
    for (sz::string_view_t line : sz::split_characters(doc, sz::newlines_set())) (void)line;

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
    for (sz_rune_t r : sz::utf8_runes(family)) (void)r, ++codepoints;
    std::size_t graphemes = 0;
    for (sz::string_view_t g : sz::utf8_graphemes(family)) (void)g, ++graphemes;

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
for (sz::string_view_t g : sz::utf8_graphemes(msg)) { if (seen++ == 3) break; cut += g.size(); }
assert(msg.sub(0, cut) == "🙏🏽👍🏿✨"); // the three emoji keep their skin-tone modifiers
```

Word and sentence boundaries are just as treacherous: contractions, CJK, decimals, and URLs all defeat naive splitting.
UAX-29 keeps `can't` whole while separating CJK ideographs that no space divides, and refuses to break a sentence inside `$9.99`:

```cpp
sz::string_view_t phrase = "can't stop 北京!";
assert(*sz::utf8_wordbreaks(phrase).begin() == "can't"); // apostrophe is interior; 北 and 京 split apart

sz::string_view_t prose = "She paid $9.99. Cheap.";
std::size_t sentences = 0;
for (sz::string_view_t s : sz::utf8_sentences(prose)) (void)s, ++sentences;
assert(sentences == 2); // the dot inside $9.99 is not a sentence break
```

The full family of ranges, free functions taking the text first, each viewing the source and yielding `sz::string_view_t` segments, or `sz_rune_t` for runes.
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
    auto m = sz::utf8_uncased_search(hay, "strasse"); // no folded copy of `hay` is made
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
if (sz::failed(sz::utf8_uncased_fold(greeting))) return 1; // in place; `ß` expands to `ss`
assert(greeting == "grüsse");
```

## Trimming and Translating

Trimming takes any string: `sz::strip(text, byteset)`, `sz::lstrip(text, byteset)`, and `sz::rstrip(text, byteset)` return a narrowed view with the matching leading/trailing bytes removed — no copy is made.

```cpp
#include <stringzilla/stringzilla.hpp>
namespace sz = ashvardanian::stringzilla;

int main() {
    sz::string_view_t padded = "  \t spaced out \n";
    sz::string_view_t tight = sz::strip(padded, sz::whitespaces_set());
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
    if (sz::failed(sz::lookup(text.span(), to_upper))) return 1; // in place
    assert(text == "HELLO");
    return 0;
}
```

## Hashing and Checksums

In C++, hashing is exposed as free functions over any string — `sz::hash(text, seed = 0)`, `sz::bytesum(text)`, and `sz::hash_multiseed(text, seeds, hashes)`, which returns `unexpected_dimensions_k` unless both spans have one entry per seed — and as the function object `sz::hash_t{}` together with `sz::equal_to_t{}`, which are convenient as template arguments for `std::unordered_map`/`std::unordered_set`.
`sz::fill_random(slice, nonce)` fills a mutable span with reproducible noise.

```cpp
#include <stringzilla/stringzilla.hpp>
#include <unordered_map>
#include <unordered_set>
#include <cassert>
namespace sz = ashvardanian::stringzilla;

int main() {
    // Drop the SIMD hash and equality into the STL containers as template arguments.
    std::unordered_map<std::string, int, sz::hash_t, sz::equal_to_t> counts;
    for (char const *word : {"apple", "banana", "apple"}) counts[word] += 1;
    assert(counts["apple"] == 2);
    assert(counts.size() == 2);

    // `sz::string_t` keys also work in the plain containers, via the bundled `std::hash` specialization.
    std::unordered_set<sz::string_t> uniq;
    for (char const *word : {"apple", "banana", "apple"}) {
        auto [key, status] = sz::string_t::make(word);
        if (sz::failed(status)) return 1;
        uniq.insert(std::move(key));
    }
    assert(uniq.size() == 2);

    sz::string_t noise;
    if (sz::failed(noise.resize(16))) return 1;
    if (sz::failed(sz::fill_random(noise.span(), /*nonce*/ 7))) return 1; // reproducible noise
    assert(noise.size() == 16);
    return 0;
}
```

## Sorting and Sets

Sorting is stable and every scratch allocation comes from the allocator you pass, as the [C guide](../c/README.md#sorting-and-sets) describes.
Matching the all-`noexcept` surface, the C++ sort takes any container plus a string-extractor callable and writes the permutation into a caller-owned span, returning a `status_t`:

```cpp
status_t sz::argsort(container, extractor, span<sz_size_t> order, top_count = 0,
                     direction = sz::sort_direction_t::ascending_k, allocator = {},
                     capabilities = sz::default_capabilities(), stream = nullptr);
status_t sz::argsort_utf8_uncased(container, extractor, span<sz_size_t> order, top_count = 0,
                                  direction = sz::sort_direction_t::ascending_k, allocator = {},
                                  capabilities = sz::default_capabilities(), stream = nullptr);
expected<size_t> sz::intersect(first, extractor, second, extractor, seed,
                               span<sz_size_t> first_positions, span<sz_size_t> second_positions,
                               allocator = {}, capabilities = sz::default_capabilities(), stream = nullptr);
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
    std::vector<sz::string_view_t> names = {"banana", "apple", "cherry"};

    sz_size_t order[3];
    sz::status_t status = sz::argsort(
        names, [](sz::string_view_t name) { return name; }, {order, 3});
    assert(status == sz::status_t::success_k);
    assert(order[0] == 1 && order[1] == 0 && order[2] == 2); // apple, banana, cherry

    auto [sorted, sort_status] = sz::argsort(names, 0, sz::sort_direction_t::descending_k);
    assert(sz::succeeded(sort_status) && sorted[0] == 2); // cherry first
    return 0;
}
```

Without output spans, `sz::argsort`, `sz::argsort_utf8_uncased` and `sz::intersect` allocate their results from the allocator passed before the capabilities and the stream, and return them in an `expected`, never throwing.

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
    if (sz::failed(sz::replace_all(greeting, "l", "L"))) return 1;

    assert(sz::starts_with(greeting, "heLLo"));
    return 0;
}
```

Collections pair the same way: `sz::strings_view_t` views an Arrow `String` or `LargeString` array from `(data, offsets, count)`, a span of `sz::string_view_t`, or an owned column, and `sz::strings_t` owns one.
Its conversions, `sz::gather`, `sz::escape`, `sz::unescape`, `sz::lookup` and `sz::normalize`, take the column first and return a fresh `sz::strings` inside an `sz::expected` with the status, leaving nothing allocated on failure.
`sz::find`, `sz::hashes` and `sz::to_rows` write into spans the caller owns, and `sz::split_into` and `sz::split_lines` cut one text into an owned column.
Every one takes a capability mask, `sz::default_capabilities()` unless given, and a stream, like the C dispatch points.

```cpp
#include <stringzilla/stringzilla.hpp>
namespace sz = ashvardanian::stringzilla;

int main() {
    char const payload[] = "alphabetagamma";
    sz_u32_t const offsets[] = {0, 5, 9, 14};
    sz::strings_view_t const column(payload, offsets, 3); // an Arrow `String` array, no copies

    auto const [escaped, escape_status] = sz::escape(column, sz_escape_json_k);
    if (sz::failed(escape_status)) return 1;

    std::size_t malformed_index = 0;
    auto const [unescaped, unescape_status] = sz::unescape(escaped.view(), sz_escape_json_k, malformed_index);
    assert(sz::succeeded(unescape_status) && unescaped[1] == "beta");

    std::size_t positions[3];
    if (sz::failed(sz::find(column, "a", positions))) return 1; // the first match in every string
    return 0;
}
```

## Batch Engines

Five engines prepare one side of a batch once and then serve any number of batches on the other side: `sz::levenshtein_engine_t`, `sz::ngrams_engine_t`, `sz::substrings_engine_t`, `sz::subdocuments_engine_t` and `sz::regex_engine_t`.
Each holds only its C struct, is move-only, and comes from `make(...)`, which takes the C initializer's inputs in order, then an optional `sz_allocator_t *`, a capability mask and a stream, and returns an `sz::expected`.
Every verb forwards to its C twin and writes into spans the caller owns, with output strides in bytes, refusing with `unexpected_dimensions_k` a span shorter than what the call would touch.

Edit distances from every query to every candidate land in a `[queries, candidates]` block:

```cpp
sz::string_view_t const queries[] {"kitten", "flaw"}, candidates[] {"sitting", "lawn"};
auto [engine, status] = sz::levenshtein_engine_t::make(sz::strings_view_t(queries), sz_levenshtein_bytes_k);
std::size_t distances[4];
status = engine.distances(sz::strings_view_t(candidates), SIZE_MAX, distances, 2 * sizeof(std::size_t));
assert(distances[0] == 3 && distances[3] == 2);
```

N-gram overlaps are the shares of each query's distinct N-grams a candidate holds, per width, and sketches are MinHash signatures at those widths:

```cpp
sz::string_view_t const queries[] {"abcd"}, candidates[] {"xabcdx", "zzz"};
std::size_t const widths[] {2, 3};
auto [engine, status] = sz::ngrams_engine_t::make(sz::strings_view_t(queries), widths, 0);
float shares[4]; // [queries, candidates, widths]
status = engine.overlaps(sz::strings_view_t(candidates), 1.0f, shares, 4 * sizeof(float), 2 * sizeof(float));
std::uint8_t sketches[2][64];
status = engine.sketches(sz::strings_view_t(candidates), 64, {&sketches[0][0], 128}, 64);
```

A substring vocabulary marks, counts, finds and rewrites matches in many haystacks, and scores them as one BM25 query:

```cpp
sz::string_view_t const needles[] {"he", "she"}, haystacks[] {"ushers", "none"};
auto [engine, status] = sz::substrings_engine_t::make(sz::strings_view_t(needles), sz_cased_k,
                                                      sz_substrings_overlapping_k,
                                                      STRINGZILLA_SUBSTRINGS_HOT_STATES_AUTO, 0, 0);
std::size_t counts[2];
status = engine.counts(sz::strings_view_t(haystacks), counts, sizeof(std::size_t)); // {2, 0}
sz_match_t matches[2];
std::size_t offsets[3]; // one boundary per haystack, plus the total
status = engine.find(sz::strings_view_t(haystacks), matches, offsets);
```

JSON or XML paths select nodes in many documents, naming the first malformed path or document on failure:

```cpp
sz::string_view_t const paths[] {"$.a"}, documents[] {R"({"a": 1})", R"({"b": 2})"};
std::size_t malformed_index = 0;
auto [engine, status] = sz::subdocuments_engine_t::make(sz::strings_view_t(paths), sz_subdocuments_json_k,
                                                        sz_subdocuments_options_none_k, malformed_index);
std::size_t counts[2];
status = engine.counts(sz::strings_view_t(documents), counts, sizeof(std::size_t), malformed_index); // {1, 0}
```

Regular expressions count, find, rewrite and extract groups, with `$N` references in replacements:

```cpp
sz::string_view_t const patterns[] {"[0-9]+"}, haystacks[] {"a1b22", "none"}, replacements[] {"#"};
std::size_t malformed_index = 0;
auto [engine, status] = sz::regex_engine_t::make(sz::strings_view_t(patterns), sz_cased_k, malformed_index);
char target[16];
std::size_t offsets[3];
status = engine.replace(sz::strings_view_t(haystacks), sz::strings_view_t(replacements), target, offsets);
assert(sz::string_view_t(target, offsets[2]) == "a#b#none");
```

## Capabilities & Devices

Every verb that takes a mask, like `sz::lookup` or `sz::argsort`, defaults to `sz::default_capabilities()` from `<stringzilla/devices.hpp>`, the enabled CPU mask, while `sz::device_t::make(kind, ordinal)` opens one GPU to ask its `capabilities_enabled()`, and `sz::device_t::cpu().configure_thread(mask)` wraps `sz_configure_thread_cpu`.
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

The containers are headers alone, so you can use them in some translation units and gradually transition to the safer restricted API in others.
Bonus - all the bound checking is branchless, so it has a constant cost and won't hurt your branch predictor.

## Beyond the C++ Standard Library, Learning from Python

Python is arguably the most popular programming language for data science.
In part, that's due to the simplicity of its standard interfaces.
StringZilla brings some of that functionality to C++.

- Content checks: `isalnum`, `isalpha`, `isascii`, `isdigit`, `islower`, `isspace`, `isupper`.
- Trimming character sets: `lstrip`, `rstrip`, `strip`.
- Trimming string matches: `remove_prefix`, `remove_suffix`.
- Ranges of search results: `split`, `rsplit`, `split_characters`.
- Number of non-overlapping substring matches: `count`.
- Partitioning: `partition`, `rpartition`.

For example, when parsing documents, it is often useful to split it into substrings.
Most often, after that, you would compute the length of the skipped part, the offset and the length of the remaining part.
This results in a lot of pointer arithmetic and is error-prone.
StringZilla provides a convenient `sz::partition` function, which returns a tuple of three string views, making the code cleaner.

```cpp
auto parts = sz::partition(haystack, ':'); // Matching a character
auto [before, match, after] = sz::partition(haystack, ':'); // Structure unpacking
auto [before, match, after] = sz::partition(haystack, sz::byteset_t(":;", 2)); // Character-set argument
auto [before, match, after] = sz::partition(haystack, " : "); // String argument
auto [before, match, after] = sz::rpartition(haystack, sz::whitespaces_set()); // Split around the last whitespace
```

Combining those with the `split` function, one can easily parse a CSV file or HTTP headers.

```cpp
for (auto line : sz::split(haystack, "\r\n")) {
    auto [key, _, value] = sz::partition(line, ':');
    headers[sz::strip(key, sz::whitespaces_set())] = sz::strip(value, sz::whitespaces_set());
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
sz::string_view_t text = "  user 42\n", domain = "example", tld = "com";
sz::hash(text); // -> 64 bit unsigned integer
text.ssize(); // -> signed length, sparing a `static_cast<std::ptrdiff_t>(text.size())`
sz::contains_only(text, sz::byteset_t(" \t", 2)); // == sz::find_first_not_of(text, sz::byteset_t(" \t", 2)) == npos;
sz::contains(text, ' '); // == sz::find(text, ' ') != npos;

// Simpler slicing than `substr`, with signed offsets like Python's
text.front(5); // -> sz::string_view_t, like `text[:5]`
text.back(-3); // -> sz::string_view_t, like `text[-3:]`

// Safe variants, which clamp the range into the string bounds
text.sub(2, 100) == text.back(2); // like `text[2:100]`

// Character set filtering
sz::rstrip(sz::lstrip(text, sz::whitespaces_set()), sz::newlines_set()); // like Python

// Incremental construction
sz::string_t owned;
if (sz::failed(owned.push_back('x'))) return 1; // `bad_alloc_k` instead of throwing if growing fails

auto [joined, status] = sz::string_t::make(sz::concatenate(text, "@", domain, ".", tld)); // One allocation
```
