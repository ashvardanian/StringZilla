# StringZilla for C

The C ABI exposes each kernel family as a stable C 99 surface: substring and byte-set search, non-cryptographic hashing and checksums, lexicographic comparison, sorting and intersection of string collections, collection conversions from escaping and normalization to byte translation, and the stateful cross-product engines for edit distances, N-gram overlaps and MinHash sketches, and multi-pattern search.
It ships as a precompiled __static__ or __shared__ library that compiles every capability once and picks one per call, for maximum portability, declared by the C headers `<stringzilla/stringzilla.h>`, which can also compile the kernels inline with `STRINGZILLA_HEADER_ONLY=1`.
The [C++ SDK](../cpp/README.md) is built on this library, and the [kernels](../include/stringzilla/README.md) it compiles are described beside their sources.

Every verb has a dispatch point, like `sz_find_best`, which runs the best capability a mask shares with the verb's list, and the mask to pass on the CPU is the one `sz_capabilities_enabled_cpu` reports.
There is no hidden global allocation and no thread pool: every function that may allocate takes an explicit `sz_allocator_t *`, and an engine holds the blocks its allocator handed it until you free it.
An engine is prepared by `sz_levenshtein_engine_init`, and its twins for N-grams and substrings, for one device's capabilities and a stream, which names the device, so choosing a device is choosing the mask and the stream an engine is built with rather than setting a global.
Its rounds, such as `sz_levenshtein_distances`, take a trailing stream: null on the CPU, while on a GPU they enqueue there and return, leaving the join to the caller.
A GPU engine's init uses its stream for its own work alone and sizes a round's device memory from the budgets it takes, so no compute verb allocates or joins, and a round carrying more haystacks than its budget, or on Metal more candidates, is refused with `sz_unexpected_dimensions_k`.
N-gram and Levenshtein engines keep no per-round state on a CUDA device, so several streams may score one engine at once, while the rounds of one substrings engine share its arena and its report, so the caller orders them.

The headers compile as freestanding C 99; set `STRINGZILLA_WITH_LIBC=0` to drop the libc dependency.

## Installation

The headers declare the StringZilla library, so the simplest integration links it and includes the umbrella header.
Its dispatch points pick the best capability at runtime, so one binary runs optimally on any CPU.

```c
#include <stringzilla/stringzilla.h>
```

Each family can also be included on its own when you only need a slice of the API and want to keep compile times down:

```c
#include <stringzilla/find.h> // `sz_find_best`, `sz_rfind_best`, byte-set scans
#include <stringzilla/hash.h> // `sz_hash_best`, `sz_bytesum_best`, incremental state
#include <stringzilla/sort.h> // `sz_sort_order_best`
#include <stringzilla/compare.h> // `sz_equal_best`, `sz_order_best`
#include <stringzilla/intersect.h> // `sz_intersect_best`
#include <stringzilla/strings.h> // `sz_strings_to_tape_best`, `sz_strings_lookup_best`
#include <stringzilla/small_string.h> // `sz_string_t` SSO container
#include <stringzilla/levenshtein.h> // `sz_levenshtein_engine_init`, `sz_levenshtein_distances`
#include <stringzilla/ngrams.h> // `sz_ngrams_engine_init`, `sz_ngrams_overlaps`, `sz_ngrams_sketches`
#include <stringzilla/substrings.h> // `sz_substrings_engine_init`, `sz_substrings_counts`
```

### CMake, Precompiled Library

With CMake 3.14 or newer, pull the project in with `FetchContent` and link the static target `stringzilla::static`, which compiles every capability the toolchain can build.

```cmake
include(FetchContent)
FetchContent_Declare(
    stringzilla
    GIT_REPOSITORY https://github.com/ashvardanian/StringZilla.git
    GIT_TAG main)
FetchContent_MakeAvailable(stringzilla)

target_link_libraries(your_app PRIVATE stringzilla::static)
```

`add_subdirectory(stringzilla)` works the same way.
Pulled in as a subproject, the static library is compiled only once something links it, and `STRINGZILLA_BUILD_LIBRARIES` defaults to off.
Turn that on for the shared target `stringzilla::shared`:

```cmake
set(STRINGZILLA_BUILD_LIBRARIES ON)
FetchContent_MakeAvailable(stringzilla)

target_link_libraries(your_app PRIVATE stringzilla::shared)
```

Either library's dispatch points pick a capability by the mask each call passes, so one artifact runs optimally on any CPU.
`stringzilla::abi` carries the C headers alone, with nothing to link, for a project that brings its own build of the library.

### CMake, Header Only

C code that wants no compiled dependency links the interface target `stringzilla::header`, which adds `include/` to your search path and defines `STRINGZILLA_HEADER_ONLY=1`:

```cmake
target_link_libraries(your_app PRIVATE stringzilla::header)
```

The kernels your compiler flags enable then compile into your translation units, and you call them by name, like `sz_find_haswell`.
Every dispatch point and finder reports `sz_missing_library_k` there.

### CMake, Installed Package

Configure with `-DSTRINGZILLA_INSTALL=ON` to install the headers and libraries system-wide, then find them from an unrelated project:

```cmake
find_package(stringzilla REQUIRED)

target_link_libraries(your_app PRIVATE stringzilla::shared)
```

`stringzilla::shared` joins the package whenever `STRINGZILLA_BUILD_LIBRARIES` was on for the install.
`stringzilla::header` always does, as it needs no submodules and compiles nothing, so it works straight out of a release tarball.

A shared-library install also writes `lib/pkgconfig/stringzilla.pc`, for build systems that read pkg-config rather than CMake:

```sh
cc main.c $(pkg-config --cflags --libs stringzilla) -o main
```

The build knobs and the supported toolchains are listed beside the [kernels](../include/stringzilla/README.md#build-knobs).

## Quickstart

```c
#include <stdio.h>
#include <stringzilla/stringzilla.h>

int main(void) {
    sz_capability_t capabilities = sz_cap_serial_k;
    sz_capabilities_enabled_cpu(&capabilities); // what this CPU runs and this binary holds
    char const haystack[] = "the quick brown fox";
    sz_cptr_t match = NULL;
    sz_status_t status = sz_find_best(haystack, sizeof(haystack) - 1, "brown", 5, &match, capabilities, NULL);
    if (status != sz_success_k || !match) return 1; // a pointer to the match, or NULL
    printf("found at %d\n", (int)(match - haystack)); // 10
    return 0;
}
```

## Types

The C API is built around length-bounded raw pointers, so most functions take a `sz_cptr_t`, an alias of `char const *`, or a `sz_ptr_t`, an alias of `char *`, together with a `sz_size_t` length, and return either a `sz_cptr_t` into the input or one of the small enums below.

Scalar and result types from `types.h`:

- `sz_cptr_t`, `sz_ptr_t` — immutable and mutable byte pointers.
- `sz_size_t`, `sz_ssize_t` — unsigned and signed size types.
- `sz_u8_t`, `sz_u32_t`, `sz_u64_t` — fixed-width integers; `sz_rune_t` is a 32-bit Unicode scalar value.
- `sz_bool_t` — `{ sz_false_k = 0, sz_true_k = 1 }`.
- `sz_ordering_t` — `{ sz_less_k = -1, sz_equal_k = 0, sz_greater_k = 1 }`, the C analog of `std::strong_ordering`.
- `sz_status_t` — `sz_success_k = 0`, `sz_bad_alloc_k`, `sz_invalid_utf8_k`, `sz_contains_duplicates_k`, `sz_overflow_risk_k`, and friends.
- `sz_pgram_t` — a pointer-sized N-gram, an alias of `sz_size_t`, which also indexes every permutation.

`sz_byteset_t` is a 256-bit set of byte values used by the byte-set search functions:

```c
sz_byteset_t set;
sz_byteset_init(&set);                 // all bits cleared
sz_byteset_add(&set, '\n');            // include a byte
int has = sz_byteset_contains(&set, '\n');
sz_byteset_invert(&set);               // complement the set
```

`sz_allocator_t` is the explicit allocator, a `{ allocate, free, handle }` triple, handed to any function that may allocate:

```c
sz_allocator_t allocator;
sz_status_t status = sz_allocator_init_heap(&allocator); // libc malloc/free
if (status != sz_success_k) return status;
// or a fixed arena with no dynamic allocation:
char arena[4096];
status = sz_allocator_init_arena(&allocator, arena, sizeof(arena), 64);
if (status != sz_success_k) return status;
```

`sz_strings_view_t` is the read-only collection of strings every collection family takes, in one of three layouts: an Apache Arrow tape of 32-bit or 64-bit offsets, which GPUs read directly, or an accessor writing up to 64 views per call over any other shape.
Helpers wire it to plain arrays of C strings, of views, of records, and of NUL-padded rows:

```c
char const *words[] = {"banana", "apple", "cherry"};
sz_strings_view_t strings;
sz_strings_from_null_terminated_strings(words, 3, &strings);
```

There is no dedicated `sz_string_view_t` struct in the C ABI — views are passed as the `(pointer, length)` pairs described above.
The one owning C string type is `sz_string_t`, documented under [String Collections and Containers](#string-collections-and-containers) below.

## Searching and Counting

The forward and reverse substring scans mirror libc's `memchr`/`memrchr` and `memmem`, but always take explicit lengths and report a pointer into the haystack, or `NULL` when there is no match.
Like every dispatch point, each takes the mask to pick a capability from and a stream, null on the CPU, and returns `sz_success_k`, or `sz_missing_kernel_k` when no capability in the mask has the kernel:

```c
sz_status_t sz_find_byte_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, //
                              sz_cptr_t *match, sz_capability_t capabilities, sz_stream_t stream);
sz_status_t sz_rfind_byte_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, //
                               sz_cptr_t *match, sz_capability_t capabilities, sz_stream_t stream);
sz_status_t sz_find_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length,
                         sz_cptr_t *match, sz_capability_t capabilities, sz_stream_t stream);
sz_status_t sz_rfind_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length,
                          sz_cptr_t *match, sz_capability_t capabilities, sz_stream_t stream);
```

Byte-set scans replace `strspn`/`strcspn` and take a prebuilt `sz_byteset_t`:

```c
sz_status_t sz_find_byteset_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *set, //
                                 sz_cptr_t *match, sz_capability_t capabilities, sz_stream_t stream);
sz_status_t sz_rfind_byteset_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *set, //
                                  sz_cptr_t *match, sz_capability_t capabilities, sz_stream_t stream);
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

sz_size_t count_occurrences(sz_cptr_t text, sz_size_t length, sz_cptr_t needle, sz_size_t needle_length,
                            sz_capability_t capabilities) {
    sz_size_t total = 0;
    sz_cptr_t cursor = text, end = text + length, match = NULL;
    while (sz_find_best(cursor, (sz_size_t)(end - cursor), needle, needle_length, &match, capabilities, NULL) ==
               sz_success_k && match) {
        ++total;
        cursor = match + needle_length;      // disjoint matches; use `+1` for overlapping
    }
    return total;
}

int main(void) {
    sz_capability_t capabilities = sz_cap_serial_k;
    sz_capabilities_enabled_cpu(&capabilities);

    sz_byteset_t whitespace;
    sz_byteset_init(&whitespace);
    char const *spaces = " \t\n\r\v\f";
    for (char const *p = spaces; *p; ++p) sz_byteset_add(&whitespace, *p);

    char const *line = "  hello world";
    sz_cptr_t first_space = NULL;
    sz_find_byteset_best(line, 13, &whitespace, &first_space, capabilities, NULL);
    (void)first_space;
    return 0;
}
```

## Unicode Segmentation

UAX-29 graphemes, words, and sentences, plus UAX-14 line-break opportunities, are batched kernels in C and lazy ranges in the [C++ SDK](../cpp/README.md#unicode-segmentation).
`sz_utf8_graphemes_best` and its three tiling siblings write one length per segment, each segment starting where the previous one ended, so after a full batch the caller resumes at the sum of its lengths.
The newline, whitespace and delimiter finders write offsets and lengths of their matches, and after a full batch `bytes_consumed` ends exactly at the last match; otherwise it is the whole input.
`sz_utf8_decode_best` treats the end of the text as the end of the input, decoding a truncated final sequence to one U+FFFD, so a caller feeding it chunks splits them at a rune boundary.

## Case Folding

Case-insensitive matching across full Unicode folds `ß` to `ss`, so the matched length can differ from the needle's.
`sz_utf8_uncased_needle_init_best` analyses the needle once into a `sz_utf8_uncased_needle_t`, and `sz_utf8_uncased_search_best` only reads it, so any number of searches, under any capabilities, share one analysis.
The prepared needle points at the needle's bytes rather than copying them, so they must outlive it:

```c
#include <stringzilla/stringzilla.h>

sz_utf8_uncased_needle_t needle;
sz_utf8_uncased_needle_init_best("café", 5, &needle, capabilities, NULL);
sz_cptr_t hit = NULL;
sz_size_t match_length = 0;
sz_utf8_uncased_search_best(haystack, haystack_length, &needle, &hit, &match_length, capabilities, NULL);
// `hit` points at the first case-insensitive match, or NULL; reuse `needle` for the next haystack.
```

An empty needle matches at the start of every haystack, with a zero `match_length`.

## Translating

Translation maps every byte through a 256-entry lookup table with `sz_strings_lookup_best`, documented under [String Collections and Containers](#string-collections-and-containers).
The library ships ready-made table initializers — `sz_strings_lookup_init_lower`, `sz_strings_lookup_init_upper`, and `sz_strings_lookup_init_ascii`.
To select an ASCII-only fast path, scan for the complement of `sz_byteset_init_ascii` with `sz_find_byteset_best`, which reaches the SIMD byteset kernels instead of a scalar loop.

## Hashing and Checksums

Hashing comes in three flavors: a byte checksum, a seeded single-shot hash, and an incremental streaming state.
All produce identical output across every backend and platform, in both single-shot and incremental modes.

```c
sz_status_t sz_bytesum_best(sz_cptr_t text, sz_size_t length, sz_u64_t *checksum, //
                            sz_capability_t capabilities, sz_stream_t stream);
sz_status_t sz_hash_best(sz_cptr_t text, sz_size_t length, sz_u64_t seed, sz_u64_t *hash, //
                         sz_capability_t capabilities, sz_stream_t stream);
sz_status_t sz_hash_multiseed_best(sz_cptr_t text, sz_size_t length, sz_u64_t const *seeds, sz_size_t seeds_count,
                                   sz_u64_t *hashes, sz_capability_t capabilities, sz_stream_t stream);
```

`sz_hash_best` is non-cryptographic, fast for both short and long inputs, passes the SMHasher `--extra` suite, and uses the AES extensions where present.
`sz_hash_multiseed_best` hashes one input under many seeds at once — for feature hashing, Count-Min sketches, Bloom filters, and MinHash/LSH — by normalizing the input into AES blocks once and replaying cheap per-seed rounds.

Incremental hashing uses an opaque `sz_hash_state_t`, whose layout every capability shares, so each call may pass its own mask:

```c
sz_status_t sz_hash_state_init_best(sz_hash_state_t *state, sz_u64_t seed, sz_capability_t capabilities, sz_stream_t stream);
sz_status_t sz_hash_state_update_best(sz_hash_state_t *state, sz_cptr_t text, sz_size_t length, //
                                      sz_capability_t capabilities, sz_stream_t stream);
sz_status_t sz_hash_state_digest_best(sz_hash_state_t const *state, sz_u64_t *hash, //
                                      sz_capability_t capabilities, sz_stream_t stream);
sz_bool_t sz_hash_state_equal(sz_hash_state_t const *lhs, sz_hash_state_t const *rhs);
```

A streaming SHA-256 is also provided, producing a 32-byte digest, with hardware backends on x86 SHA-NI, ARM NEON-SHA, and others:

```c
sz_status_t sz_sha256_state_init_best(sz_sha256_state_t *state, sz_capability_t capabilities, sz_stream_t stream);
sz_status_t sz_sha256_state_update_best(sz_sha256_state_t *state, sz_cptr_t text, sz_size_t length, //
                                        sz_capability_t capabilities, sz_stream_t stream);
sz_status_t sz_sha256_state_digest_best(sz_sha256_state_t const *state, sz_u8_t digest[32], //
                                        sz_capability_t capabilities, sz_stream_t stream);
```

The same AES primitives back a reproducible pseudo-random fill, useful with `sz_strings_lookup_best` for generating random strings over a chosen alphabet:

```c
sz_status_t sz_fill_random_best(sz_ptr_t target, sz_size_t length, sz_u64_t nonce, //
                                sz_capability_t capabilities, sz_stream_t stream);
```

Example combining a single-shot hash, a streamed hash, and a checksum:

```c
#include <stringzilla/stringzilla.h>

int main(void) {
    sz_capability_t capabilities = sz_cap_serial_k;
    sz_capabilities_enabled_cpu(&capabilities);

    sz_u64_t single = 0;
    sz_hash_best("hello world", 11, 42, &single, capabilities, NULL);

    sz_hash_state_t state;
    sz_u64_t streamed = 0;
    sz_hash_state_init_best(&state, 42, capabilities, NULL);
    sz_hash_state_update_best(&state, "hello ", 6, capabilities, NULL);
    sz_hash_state_update_best(&state, "world", 5, capabilities, NULL);
    sz_hash_state_digest_best(&state, &streamed, capabilities, NULL);
    assert(streamed == single);

    sz_u64_t sum = 0;
    sz_bytesum_best("hi", 2, &sum, capabilities, NULL);
    assert(sum == 209);
    return 0;
}
```

## Sorting and Sets

These operate on a read-only `sz_strings_view_t` in any layout and never reorder the underlying data — they fill an `order` permutation array instead.
Sorting is __stable__, so equal elements keep their input order, supports descending order via a `reverse` flag, and supports partial / top-K sorting via `top_count` — pass `0` to sort everything.
Every allocation these kernels need is routed through the `sz_allocator_t` you supply — pass `NULL` for the default `malloc`-based one — so, unlike `std::sort` or `thrust::sort`, no scratch memory is ever allocated behind your back.

The C API fills a caller-owned `order` array and reports success through a `sz_status_t`:

```c
sz_status_t sz_sort_order_best(
    sz_strings_view_t const *strings, sz_size_t top_count, sz_bool_t reverse,
    sz_allocator_t *allocator, sz_size_t *order,
    sz_capability_t capabilities, sz_stream_t stream);

sz_status_t sz_sort_uncased_order_best(
    sz_strings_view_t const *strings, sz_size_t top_count, sz_bool_t reverse,
    sz_allocator_t *allocator, sz_size_t *order,
    sz_capability_t capabilities, sz_stream_t stream);
```

`sz_sort_order_best` orders byte-lexicographically; `sz_sort_uncased_order_best` orders under Unicode case-folding, folding small chunks on the fly, and malformed UTF-8 sorts by raw byte value so the order stays total and deterministic.
The `allocator` argument may be `NULL` to use the default allocator.

```c
#include <stringzilla/stringzilla.h>

int main(void) {
    sz_capability_t capabilities = sz_cap_serial_k;
    sz_capabilities_enabled_cpu(&capabilities);

    char const *fruits[] = {"banana", "apple", "cherry"};
    sz_strings_view_t strings;
    sz_strings_from_null_terminated_strings(fruits, 3, &strings);

    sz_size_t order[3];
    sz_status_t status = sz_sort_order_best(&strings, /*top_count*/ 0, sz_false_k, NULL, order, capabilities, NULL);
    assert(status == sz_success_k);
    assert(order[0] == 1 && order[1] == 0 && order[2] == 2); // apple, banana, cherry
    return 0;
}
```

### Set Intersection

`sz_intersect_best` computes the intersection of two __deduplicated__ binary string collections using a hash table, writing matched positions from each side and leaving unmatched slots as `STRINGZILLA_SIZE_MAX`:

```c
sz_status_t sz_intersect_best(
    sz_strings_view_t const *first_strings, sz_strings_view_t const *second_strings,
    sz_allocator_t *allocator, sz_u64_t seed, sz_size_t *intersection_count,
    sz_size_t *first_positions, sz_size_t *second_positions,
    sz_capability_t capabilities, sz_stream_t stream);
```

The `seed` randomizes the hash table to resist adversarial inputs; the position arrays must each fit at least `min(first->count, second->count)` entries.
The companion `sz_join_semantics_t` enum documents the SQL-style JOIN modes the family is designed around, namely `sz_join_inner_strict_k`, `sz_join_inner_k`, `sz_join_left_outer_k`, `sz_join_right_outer_k`, `sz_join_full_outer_k`, and `sz_join_cross_k`.

```c
char const *first[]  = {"banana", "apple", "cherry"};
char const *second[] = {"cherry", "orange", "pineapple", "banana"};
sz_strings_view_t a, b;
sz_strings_from_null_terminated_strings(first, 3, &a);
sz_strings_from_null_terminated_strings(second, 4, &b);

sz_size_t intersection_count;
sz_size_t a_positions[3], b_positions[3];  // sized to the smaller collection
sz_status_t status = sz_intersect_best(&a, &b, NULL, /*seed*/ 0, &intersection_count, a_positions, b_positions,
                                       capabilities, NULL);
assert(intersection_count == 2); // "banana" and "cherry"
```

## String Collections and Containers

Collections are read through `sz_strings_view_t` in one of three layouts: an Apache Arrow `String` tape with 32-bit offsets, a `LargeString` tape with 64-bit offsets, or an accessor that writes the views of up to 64 strings per call.
Accessors describe arrays of views, NUL-terminated pointers, struct members, record pointers and NUL-padded rows without copying a byte, while tapes are what Arrow and the GPU kernels read directly.
Every conversion takes a view, an optional array of indices that reorders, samples or repeats it, and a count, and writes a writable `sz_strings_span_t` tape:

```c
sz_status_t sz_strings_to_tape_best(sz_strings_view_t const *strings, sz_size_t const *order, sz_size_t count,
                                    sz_allocator_t *allocator, sz_strings_span_t *tape,
                                    sz_capability_t capabilities, sz_stream_t stream);
sz_status_t sz_strings_escape_best(sz_strings_view_t const *strings, sz_size_t const *order, sz_size_t count,
                                   sz_escape_kind_t kind, sz_allocator_t *allocator, sz_strings_span_t *tape,
                                   sz_capability_t capabilities, sz_stream_t stream);
sz_status_t sz_strings_unescape_best(sz_strings_view_t const *strings, sz_size_t const *order, sz_size_t count,
                                     sz_escape_kind_t kind, sz_allocator_t *allocator, sz_strings_span_t *tape,
                                     sz_size_t *malformed_index, sz_capability_t capabilities, sz_stream_t stream);
sz_status_t sz_strings_normalize_best(sz_strings_view_t const *strings, sz_size_t const *order, sz_size_t count,
                                      sz_normal_form_t form, sz_allocator_t *allocator, sz_strings_span_t *tape,
                                      sz_capability_t capabilities, sz_stream_t stream);
sz_status_t sz_strings_lookup_best(sz_strings_view_t const *strings, sz_size_t const *order, sz_size_t count,
                                   char const lut[256], sz_allocator_t *allocator, sz_strings_span_t *tape,
                                   sz_capability_t capabilities, sz_stream_t stream);
sz_status_t sz_strings_to_rows_best(sz_strings_view_t const *strings, sz_size_t const *order, sz_size_t count,
                                    sz_strings_rows_t const *rows, sz_size_t *longest,
                                    sz_capability_t capabilities, sz_stream_t stream);
```

Every conversion is atomic: it succeeds with the whole result in the tape, or fails and leaves the tape, its blocks and their bytes exactly as they were, whether an allocation was refused or a string was malformed.
A zeroed span receives fresh blocks from the allocator, blocks that already have the result's sizes are written in place, and any other span's old blocks go back only once the result is written.
The allocator's `reallocate` is an optional fast path; without it, a block that must change size moves through `allocate`, `memcpy` and `free`.
Escaping covers JSON, JSON in ASCII the way Python's `json.dumps` writes it, RFC 4180 CSV, hexadecimal, HTML, URL percent-encoding, and terminal-safe logs, and unescaping reports the first malformed string.
Normalization brings every string into NFC, NFD, NFKC, NFKD or NFKC_Casefold, and NUL-padded rows serve NumPy `S` arrays and fixed-width record fields.
A single buffer is a tape of one string, so lowercasing in place looks like this:

```c
#include <stringzilla/stringzilla.h>

int main(void) {
    sz_capability_t capabilities = sz_cap_serial_k;
    sz_capabilities_enabled_cpu(&capabilities);

    char to_lower[256];
    sz_strings_lookup_init_lower(to_lower);
    char buffer[3] = {'A', 'B', 'C'};
    sz_u32_t offsets[2] = {0, 3};
    sz_strings_span_t tape;
    tape.layout = sz_strings_u32_tape_k, tape.count = 1;
    tape.data.u32_tape.data = buffer, tape.data.u32_tape.offsets = offsets;
    sz_strings_view_t strings;
    sz_strings_from_span(&tape, &strings);
    sz_strings_lookup_best(&strings, NULL, 1, to_lower, NULL, &tape, capabilities, NULL);
    assert(buffer[0] == 'a' && buffer[1] == 'b' && buffer[2] == 'c'); // "abc"
    return 0;
}
```

### The `sz_string_t` SSO Container

`sz_string_t` is a tiny memory-owning string with Small-String-Optimization: short strings, up to 22 bytes plus the terminator on 64-bit, live inline with no heap allocation, while longer strings spill to the heap.
Its length can be read branchlessly, and many length changes avoid branches entirely.
Every mutating call takes an explicit `sz_allocator_t *`.

Lifecycle and read-only access:

```c
void sz_string_init(sz_string_t *string);
sz_ptr_t sz_string_init_length(sz_string_t *string, sz_size_t length, sz_allocator_t *allocator);
sz_bool_t sz_string_is_on_stack(sz_string_t const *string);
void sz_string_unpack(sz_string_t const *string, sz_ptr_t *start, sz_size_t *length, sz_size_t *space, sz_bool_t *is_external);
void sz_string_range(sz_string_t const *string, sz_ptr_t *start, sz_size_t *length);
sz_size_t sz_string_length(sz_string_t const *string);
sz_bool_t sz_string_equal(sz_string_t const *a, sz_string_t const *b);
sz_ordering_t sz_string_order(sz_string_t const *a, sz_string_t const *b);
```

Growth and mutation, where all but `sz_string_erase` may allocate and return `STRINGZILLA_NULL_CHAR` on failure:

```c
sz_ptr_t sz_string_reserve(sz_string_t *string, sz_size_t new_capacity, sz_allocator_t *allocator);
sz_ptr_t sz_string_expand(sz_string_t *string, sz_size_t offset, sz_size_t added_length, sz_allocator_t *allocator);
sz_size_t sz_string_erase(sz_string_t *string, sz_size_t offset, sz_size_t length);
sz_ptr_t sz_string_shrink_to_fit(sz_string_t *string, sz_allocator_t *allocator);
void sz_string_free(sz_string_t *string, sz_allocator_t *allocator);
```

`sz_string_expand` opens an uninitialized gap of `added_length` at `offset` that you then populate, often via `memcpy`; `sz_string_erase` removes a range without ever allocating and cannot fail.

```c
#include <string.h>
#include <stringzilla/stringzilla.h>

int main(void) {
    sz_allocator_t allocator;
    if (sz_allocator_init_heap(&allocator) != sz_success_k) return 1;

    sz_string_t s;
    sz_string_init(&s);

    sz_ptr_t room = sz_string_expand(&s, 0, 5, &allocator);  // 5-byte gap at offset 0
    if (!room) return 1;
    memcpy(room, "hello", 5);

    sz_ptr_t start; sz_size_t length;
    sz_string_range(&s, &start, &length);
    assert(length == 5);

    sz_string_free(&s, &allocator);
    return 0;
}
```

### Small String Optimization

Most operations in StringZilla don't assume any memory ownership, and `sz_string_t` is the one memory-owning string "class", which `sz::string_t` wraps in C++.
Like other efficient string implementations, it uses the [Small String Optimization][faq-sso] (SSO) to avoid heap allocations for short strings.

[faq-sso]: https://cpp-optimizations.netlify.app/small_strings/

A short string can be kept on the stack, if it fits within the inline character array.
Before 2015 GCC string implementation was just 8 bytes, and could only fit 7 characters.
Different STL implementations today have different thresholds for the Small String Optimization.
Similar to GCC, StringZilla is 32 bytes in size, and similar to Clang it can fit 22 characters on stack.
Our layout might be preferential, if you want to avoid branches.
If you use a different compiler, you may want to check its SSO buffer size with a [simple Gist](https://gist.github.com/ashvardanian/c197f15732d9855c4e070797adf17b21).

|                 | `libstdc++` in GCC 13 | `libc++` in Clang 17 | StringZilla |
| :-------------- | --------------------: | -------------------: | ----------: |
| String `sizeof` |                    32 |                   24 |          32 |
| Inner Capacity  |                    15 |               __22__ |      __22__ |

This design has been since ported to many high-level programming languages.
Swift, for example, [can store 15 bytes](https://developer.apple.com/documentation/swift/substring/withutf8(_:)#discussion) in the `String` instance itself.
StringZilla implements SSO at the C level, providing the `sz_string_t` union and the simple API above.

Unlike the conventional C strings, the `sz_string_t` is allowed to contain null characters.
To safely print those, pass the unpacked length to `printf` as well.

```c
sz_ptr_t string_start;
sz_size_t string_length;
sz_string_range(&string, &string_start, &string_length);
printf("%.*s\n", (int)string_length, string_start);
```

## Engines & Devices

The three engine families prepare a batch of queries once and score it against many batches of candidates, writing into a strided block the caller supplies rather than allocating one per round.
Every output stride counts bytes, as NumKong's do, and a stride that splits an entry answers `sz_unexpected_dimensions_k`.
Each init takes the mask, the allocator and a stream last, and each round is a single verb with no `_best` twin, as the engine already holds the capability its init picked:

```c
sz_status_t sz_levenshtein_engine_init(sz_levenshtein_engine_t *engine, sz_strings_view_t const *queries,
                                       sz_levenshtein_symbol_t symbol, sz_capability_t capabilities,
                                       sz_allocator_t *allocator, sz_stream_t stream);
sz_status_t sz_levenshtein_distances(sz_levenshtein_engine_t *engine, sz_strings_view_t const *candidates,
                                     sz_size_t const *symbol_counts, sz_size_t const *order, sz_size_t bound,
                                     sz_size_t *distances, sz_size_t distances_stride, sz_stream_t stream);
void sz_levenshtein_engine_free(sz_levenshtein_engine_t *engine, sz_stream_t stream);
```

Their per-tier building blocks and their design notes live beside the kernels, in [`levenshtein/README.md`](../include/stringzilla/levenshtein/README.md), [`ngrams/README.md`](../include/stringzilla/ngrams/README.md), and [`substrings/README.md`](../include/stringzilla/substrings/README.md).

A device round reads memory the device reaches, and strings only as a tape, never through an accessor, which is host code. `strings.h` declares allocation and gathering into tapes; `capabilities.h` declares stream creation, synchronization and release. Their `_best` dispatch points take a mask, with `_serial`, `_cuda`, `_rocm` and `_metal` twins, the GPU ones exported only where built:

- `sz_allocator_init_unified_best` fills a stateless `sz_allocator_t` whose blocks the host and the device share: the heap on the CPU, managed memory on CUDA and ROCm, shared buffers on Metal, each made on the device of the stream it is called with.
- `sz_strings_to_tape_best` gathers strings of any layout into a `sz_strings_span_t` tape of payload and offsets, two blocks from the allocator it is given, and `sz_strings_span_free` returns both.
- `sz_stream_synchronize_best` waits for one stream, after which what a round wrote is readable from the host.

```c
sz_capabilities_enabled_cuda(0, &caps);
sz_capabilities_enabled_cpu(&cpu_caps);
sz_allocator_init_unified_best(&unified, caps);
sz_strings_span_t tape = {0};
sz_strings_to_tape_best(&host_candidates, NULL, host_candidates.count, &unified, &tape, cpu_caps, NULL);
sz_strings_view_t candidates;
sz_strings_from_span(&tape, &candidates);
sz_levenshtein_engine_init(&engine, &queries, sz_levenshtein_bytes_k, caps, NULL, stream);
sz_levenshtein_distances(&engine, &candidates, NULL, NULL, STRINGZILLA_SIZE_MAX, distances,
                         candidates.count * sizeof(sz_size_t), stream);
sz_stream_synchronize_best(caps, stream);
sz_levenshtein_engine_free(&engine, stream);
sz_strings_span_free(&tape, &unified, stream);
```

A null stream is the default stream of the default device, and a vendor's devices are only numbered by its producers, like `sz_capabilities_enabled_cuda(ordinal, &caps)`.

## Runtime Dispatch

Every kernel exists in a serial form plus one or more per-ISA forms, named with a backend suffix (`_serial`, `_westmere`, `_goldmont`, `_haswell`, `_skylake`, `_icelake`, `_neon`, `_neonaes`, `_neonsha`, `_sve`, `_sve2`, `_sve2aes`, `_v128`, `_v128relaxed`, `_rvv`, `_rvvcrypto`, `_loongsonasx`, `_powervsx`), and the engines add one per GPU vendor, `_cuda`, `_rocm` and `_metal`.
Each verb also has a dispatch point, like `sz_find_best` or `sz_hash_best`, taking the kernels' arguments plus a capability mask before the trailing stream, and running the best capability the mask shares with the verb's list, `serial` first on the CPU.
It returns `sz_missing_kernel_k` when no capability in the mask has the kernel, as a GPU-only mask does for a CPU-only verb.

- __Library builds, `STRINGZILLA_HEADER_ONLY=0`, the default.__
  Every capability the toolchain builds is compiled into one binary, and the dispatch points pick among them by the mask each call passes.
  This is how the prebuilt libraries ship, so a single binary runs optimally on any CPU.
- __Header-only builds, `STRINGZILLA_HEADER_ONLY=1`.__
  The kernels your compiler flags enable inline into your translation unit, so calling one of them by name, like `sz_find_haswell`, costs no indirection at all.
  The dispatch points and finders compile to stubs returning `sz_missing_library_k`, while the C++ layer runs the serial kernels over its default mask, which is empty here.

Capabilities are introspectable at both compile and run time.
`sz_capability_t` is a 64-bit mask from `capabilities.h`, one bit per capability.
Within each architecture the bits ascend by preference, so the highest bit a mask shares with a verb's kernels names the kernel to run.

- __Serial__: `sz_cap_serial_k`, bit 0, which every CPU runs.
- __x86__: `sz_cap_westmere_k`, `sz_cap_goldmont_k`, `sz_cap_haswell_k`, `sz_cap_skylake_k`, `sz_cap_icelake_k`.
- __Arm__: `sz_cap_neon_k`, `sz_cap_neonaes_k`, `sz_cap_neonsha_k`, `sz_cap_sve_k`, `sz_cap_sve2_k`, `sz_cap_sve2aes_k`.
- __RISC-V__: `sz_cap_rvv_k`, `sz_cap_rvvcrypto_k`.
- __WebAssembly__: `sz_cap_v128_k`, `sz_cap_v128relaxed_k`.
- __LoongArch and Power__: `sz_cap_loongsonasx_k`, `sz_cap_powervsx_k`.
- __GPUs__: one baseline per vendor above every CPU bit, `sz_cap_cuda_k` at bit 48, `sz_cap_rocm_k` at 56, and `sz_cap_metal_k` at 60.
  CUDA generations follow its baseline: `sz_cap_ampere_k` at 49 for warp reductions and asynchronous copies from compute capability 8.0, `sz_cap_hopper_k` at 51 for clusters from 9.0, and `sz_cap_blackwell_k` at 52 for cluster launch control from 10.0.
  ROCm generations follow its baseline: `sz_cap_cdna3_k` at 57 for DPP wavefront shifts on gfx942, `sz_cap_cdna4_k` at 58 for the same shifts on gfx950, and `sz_cap_cdna5_k` at 59 for DPP row shifts bridged by `v_permlanex16` on gfx1250 and gfx1251.

`sz_cap_cpus_k` and `sz_cap_gpus_k` group the CPU and the GPU bits, and `sz_cap_any_k` sets every bit.

```c
sz_status_t sz_capabilities_detected_cpu(sz_capability_t *capabilities); // what this CPU runs
sz_status_t sz_capabilities_compiled_cpu(sz_capability_t *capabilities); // what this build holds kernels for
sz_capability_t sz_capabilities_compiled_cpu_mask(void);                 // the same, constant in the calling unit
sz_status_t sz_capabilities_enabled_cpu(sz_capability_t *capabilities);  // both at once, always with serial
sz_status_t sz_configure_thread_cpu(sz_capability_t capabilities);       // once per dispatching thread
sz_status_t sz_count_devices_cuda(sz_size_t *count);
sz_status_t sz_capabilities_detected_cuda(sz_size_t ordinal, sz_capability_t *capabilities);
sz_status_t sz_capabilities_compiled_cuda(sz_size_t ordinal, sz_capability_t *capabilities);
sz_status_t sz_capabilities_enabled_cuda(sz_size_t ordinal, sz_capability_t *capabilities);
sz_size_t sz_capabilities_name(sz_capability_t capabilities, char *buffer, sz_size_t capacity); // "serial,haswell"
char const *sz_status_name(sz_status_t status);
```

ROCm and Metal have the same four queries, ending in `_rocm` and `_metal`.
The library probes the CPU once per process and caches the answer, while header-only builds probe on every call.
The GPU queries ask the vendor's runtime every time, by that runtime's own device ordinal, and report no devices where the vendor isn't built.
A GPU binary carries code per architecture, so `compiled` is per device: each GPU unit holds one probe kernel, and `sz_capabilities_compiled_cuda` or `sz_capabilities_compiled_rocm` asks the runtime whether it loads that kernel on the device, as `cudaFuncGetAttributes` and `hipFuncGetAttributes` answer.
Projects compiling StringZilla's building blocks into their own GPU units ask the same of their own kernels.
The CPU probe inspects CPUID on x86, the AArch64 ID registers on Arm once the kernel's `HWCAP_CPUID` says it emulates `mrs`, falling back to what the compiler's flags imply, `getauxval`/`riscv_hwprobe` on RISC-V, and the auxiliary-vector HWCAPs on LoongArch and Power.
Detection always reports the full hardware truth, independent of which tiers a build compiled in; `sz_capabilities_enabled_cpu()` intersects it with the compile-time mask.
Where the OS cannot be asked, detection reports only the capabilities the compiler's own flags guarantee, never the compiled ones, which the CPU may lack.
WebAssembly is the exception with no runtime probe at all — a module carrying unsupported SIMD opcodes fails validation at instantiation, so its capabilities are fixed at compile time.
Nothing is process-wide: narrowing the mask a call passes narrows the choice, so `capabilities & ~sz_cap_sve_k` skips SVE and `sz_cap_serial_k` alone runs the reference kernel.
A host-only process never starts a GPU driver, as only the GPU queries and the engines built for a GPU reach one.
`sz_configure_thread_cpu` prepares the calling thread for the capabilities it is given, and only those; every current capability needs nothing, so it is a no-op today, kept so bindings call it where NumKong's call theirs.

On Arm, having SVE in the capability mask doesn't mean SVE kernels always win: at the common 128-bit vector length the scalable kernels for length-sensitive operations — comparisons, memory transforms, substring search, UTF-8 token scanning — are often slower than their NEON twins, while crypto-heavy operations like hashing prefer SVE2 at any width.
So each such SVE kernel measures the register width on the running CPU (`svcntb`) and hands the 128-bit case to its NEON twin, and the dispatch point needs no special case.
To force a specific backend regardless, narrow the mask or call its kernel directly.

CMake, which every binding builds through, probes which capabilities the toolchain can __compile__: each capability's probe, `probes/<capability>.c`, calls one of its kernels, compiled header-only at the baseline flags as the library compiles it.
Each capability's headers scope its kernels with a target pragma, on every platform; LASX and POWER9 need GCC 15 or newer for that, as Clang's `lasxintrin.h` and `altivec.h` stay closed without `-mlasx` and `-mcpu=power9`.
Libraries compile every capability the toolchain builds and leave the rest to the mask each call passes; header-only builds enable the capabilities their own flags name.
`-D STRINGZILLA_TARGET_<CAPABILITY>=0` drops a capability from the libraries, and `=1` keeps one only where its probe compiles.
Six words name a capability's states, from configure to dispatch:

| Word | Question | When | Scope |
| --- | --- | --- | --- |
| __probe__ | Can the toolchain build this capability? | configure: `try_compile` of `probes/<capability>.c` | build |
| __target__ | Is this capability's unit in the build? | compile: `STRINGZILLA_TARGET_<CAPABILITY>` | build |
| __implied__ | What do the baseline flags already guarantee? | compile, from predefined macros, and at runtime where the OS cannot be asked | build |
| __compiled__ | Does the binary carry this capability's code for this processor? | runtime: `sz_capabilities_compiled_<group>` | CPU: the build; GPU: each device |
| __detected__ | Does this processor offer the capability? | runtime: `sz_capabilities_detected_<group>` | processor |
| __enabled__ | What may dispatch pick? | runtime: `sz_capabilities_enabled_<group>`, detected and compiled | processor |

A monoversioned build names one machine: one CPU and at most one GPU architecture per vendor, shared by every GPU in it.
It compiles the best capability for that machine and folds every dispatch row to it, so link-time optimization drops every other kernel.
On any other machine a call answers `sz_missing_kernel_k`, never a crash and never a slower kernel in silence.
Sibling GPU capabilities form no chain, so with `STRINGZILLA_MULTIVERSIONED=OFF` a configure listing two CUDA or two ROCm architectures fails.

```c
#include <stdio.h>
#include <stringzilla/stringzilla.h>

int main(void) {
    sz_capability_t caps;
    sz_capabilities_enabled_cpu(&caps);
    char names[STRINGZILLA_CAPABILITIES_NAME_CAPACITY];
    sz_capabilities_name(caps, names, sizeof(names));
    printf("StringZilla %d.%d.%d, backends: %s\n", sz_version_major(), sz_version_minor(), sz_version_patch(), names);
    return 0;
}
```

You can also call any capability's kernel directly, with the same arguments short of the mask, when you have already established the target supports it — for example `sz_find_haswell(...)` or `sz_hash_icelake(...)` — which is handy for benchmarking and for pinning a code path in a controlled environment.

Picking a capability costs about a nanosecond per call: noise for long inputs, but not for hot short-string paths, like a sort's comparator or a per-token byte search.
There, call a capability's kernel directly, like `sz_order_neon` or `sz_find_byte_haswell`, or resolve one once through `sz_find_kernel_punned`, and call it many times:

```c
sz_kernel_order_t order = NULL;
sz_capability_t capability = 0;
sz_find_kernel_punned(sz_kernel_order_k, caps, (sz_kernel_punned_t *)&order, &capability);

sz_ordering_t ordering;
order("apple", 5, "banana", 6, &ordering, NULL);
```

It picks exactly what the dispatch point picks for the same mask, and returns `sz_missing_kernel_k` when no capability in the mask has the kernel.
Each family exports its own finder over its kinds, like `sz_compare_find_kernel` in `compare.h`, and `sz_find_kernel_punned` covers every family through one entry point.
Every verb has a kind, `sz_kernel_<verb>_k`, and a pointer type to cast the result to, `sz_kernel_<verb>_t`, taking the dispatch point's arguments short of the mask.
`sz_kernel_name` spells a kind without its `sz_kernel_` prefix and `_k` suffix, like `"find_byte"`, and `sz_kernel_named` maps such a name back, or to `sz_kernel_unknown_k`.
Verbs of one shape share a type, like `sz_kernel_find_t` for both `sz_kernel_find_k` and `sz_kernel_rfind_k`, `sz_kernel_utf8_segmenter_t` for the four tiling UTF-8 segmenters, and `sz_kernel_utf8_tokenizer_t` for the newline, whitespace and delimiter finders.
In C++, the [SDK wrappers](../cpp/README.md#capabilities--devices) default to the enabled CPU mask.

## What's Wrong with the C Standard Library?

StringZilla is not a drop-in replacement for the C Standard Library.
It's designed to be a safer and more modern alternative.
Conceptually:

1. LibC strings are expected to be null-terminated, so to use the efficient LibC implementations on slices of larger strings, you'd have to copy them, which is more expensive than the original string operation.
2. LibC functionality is asymmetric — you can find the first and the last occurrence of a character within a string, but you can't find the last occurrence of a substring.
3. LibC function names are typically very short and cryptic.
4. LibC lacks crucial functionality like hashing and doesn't provide primitives for less critical but relevant operations like fuzzy matching.

Something has to be said about its support for UTF-8.
Aside from a single-byte `char` type, LibC provides `wchar_t`:

- The size of `wchar_t` is not consistent across platforms.
  On Windows, it's typically 16 bits (suitable for UTF-16), while on Unix-like systems, it's usually 32 bits (suitable for UTF-32).
  This inconsistency can lead to portability issues when writing cross-platform code.
- `wchar_t` is designed to represent wide characters in a fixed-width format (UTF-16 or UTF-32).
  In contrast, UTF-8 is a variable-length encoding, where each character can take from 1 to 4 bytes.
  This fundamental difference means that `wchar_t` and UTF-8 are incompatible.

StringZilla [partially addresses those issues](#unicode-segmentation).
