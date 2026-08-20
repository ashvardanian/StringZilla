# StringZillas for C and C++

StringZillas is a __bulk-processing framework__ for __fingerprinting and fuzzy matching__ of __web-scale text corpora and bioinformatics sequences__.
Unlike the single-string `stringzilla` library, every operation here is a __stateful engine__ that you initialize once and reuse across many calls.
Each engine consumes __large collections__ of strings at once, scoring them as a cross-product matrix or sketching them into compact fingerprints.
And each call runs across a chosen slice of __CPU cores or a CUDA GPU__, selected through a lightweight device handle you pass on every call.

## Installation

StringZillas is __compiled__, not header-only: the heavy template instantiations for every engine are baked into precompiled libraries so you link them rather than recompile them.
The CMake project is named `stringzilla`, and it exposes four StringZillas library targets gated by two options.

The CPU libraries `stringzillas::cpus_shared` and `stringzillas::cpus_static` are built when `STRINGZILLAS_BUILD_SHARED` is on.
The CUDA libraries `stringzillas::cuda_shared` and `stringzillas::cuda_static` are additionally built when `STRINGZILLA_BUILD_CUDA` is on.
The longer `stringzilla::stringzillas_cpus_shared` spelling still resolves, and likewise for the rest.

Pull the project in with `FetchContent` or `add_subdirectory`, then link a target:

```cmake
include(FetchContent)
FetchContent_Declare(
    stringzilla
    GIT_REPOSITORY https://github.com/ashvardanian/stringzilla
    GIT_TAG main)
set(STRINGZILLAS_BUILD_SHARED ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(stringzilla)

add_executable(my_app main.c)
target_link_libraries(my_app PRIVATE stringzillas::cpus_shared)
```

To also build and link the GPU engines, enable CUDA and link the matching target:

```cmake
set(STRINGZILLA_BUILD_CUDA ON CACHE BOOL "" FORCE)
# ... after FetchContent_MakeAvailable ...
target_link_libraries(my_app PRIVATE stringzillas::cuda_shared)
```

An installed build exposes the same engines as their own CMake package, separate from the single-string `stringzilla` one:

```cmake
find_package(stringzillas REQUIRED)

target_link_libraries(my_app PRIVATE stringzillas::cpus_shared)
```

Only the shared variants are exported: the static archives are assembled from OBJECT libraries that an installed package cannot carry, so `stringzillas::cpus_static` and `stringzillas::cuda_static` are reachable in-tree only, and ship the same engines for local testing and benchmarking.
All four propagate the headers and the `SZ_DYNAMIC_DISPATCH=1` interface definition, so a consumer externs the precompiled engines instead of re-instantiating them.

The public C surface is one header:

```c
#include <stringzillas/stringzillas.h>
```

## The Engine Model

Every StringZillas operation follows the same lifecycle: an opaque engine handle, created with `*_init`, called many times, and released with `*_free`.
The engine owns its scratch arenas and pre-configured costs, so reusing one handle across batches amortizes setup over the whole corpus.

The C handles are opaque pointers:

```c
typedef void *szs_levenshtein_distances_t;
typedef void *szs_levenshtein_distances_utf8_t;
typedef void *szs_needleman_wunsch_scores_t;
typedef void *szs_smith_waterman_scores_t;
typedef void *szs_fingerprints_t;
```

Every call also takes a __device scope__, an opaque handle describing where the work runs:

```c
typedef void *szs_device_scope_t;

sz_status_t szs_device_scope_init_default(szs_device_scope_t *scope, char const **error_message);
sz_status_t szs_device_scope_init_cpu_cores(sz_size_t cpu_cores, szs_device_scope_t *scope, char const **error_message);
sz_status_t szs_device_scope_init_gpu_device(sz_size_t gpu_device, szs_device_scope_t *scope, char const **error_message);
sz_status_t szs_device_scope_get_cpu_cores(szs_device_scope_t scope, sz_size_t *cpu_cores, char const **error_message);
sz_status_t szs_device_scope_get_gpu_device(szs_device_scope_t scope, sz_size_t *gpu_device, char const **error_message);
sz_status_t szs_device_scope_get_capabilities(szs_device_scope_t scope, sz_capability_t *capabilities, char const **error_message);
void szs_device_scope_free(szs_device_scope_t scope);
```

`szs_device_scope_init_default` picks system defaults.
`szs_device_scope_init_cpu_cores` targets a fraction of CPU cores: pass `0` for all cores, `1` to use only the calling thread.
`szs_device_scope_init_gpu_device` targets a specific GPU by device index.
One device scope can be shared across engines and calls, then freed with `szs_device_scope_free`.

### Inputs: Sequences and Apache Arrow Tapes

Engines accept three input layouts, all describing a collection of strings.
The flexible `sz_sequence_t` (from the `stringzilla` headers) is a `std::vector<std::string_view>`-like structure indexed by a getter callback.
The two tape types are Apache Arrow-like contiguous layouts: one data blob plus `count + 1` offsets, where `lengths[i] = offsets[i + 1] - offsets[i]`.

```c
typedef struct sz_sequence_u32tape_t {
    sz_cptr_t data;
    sz_u32_t const *offsets;
    sz_size_t count;
} sz_sequence_u32tape_t;

typedef struct sz_sequence_u64tape_t {
    sz_cptr_t data;
    sz_u64_t const *offsets;
    sz_size_t count;
} sz_sequence_u64tape_t;
```

Use `sz_sequence_u32tape_t` for space-efficient collections under 4 GB and `sz_sequence_u64tape_t` for larger ones.
Each engine exposes a call on each layout, for example `szs_levenshtein_distances`, `szs_levenshtein_distances_u32tape`, and `szs_levenshtein_distances_u64tape`.

### Unified Memory

A GPU scope addresses its work from the device, so every buffer you hand an engine must be __device-accessible__.
That means unified memory, or plain device memory you allocated yourself.

```c
sz_status_t sz_memory_allocator_init_unified(sz_memory_allocator_t *alloc, char const **error_message);
void *szs_unified_alloc(sz_size_t size_bytes);
void szs_unified_free(void *ptr, sz_size_t size_bytes);
```

The rule covers __outputs as well as inputs__, and the small per-collection arrays with them.
Match counts, tape offsets, BM25 weights and lengths, and score vectors are no different from a result matrix.
Host memory is refused with `sz_device_memory_mismatch_k` and nothing is written, so a copy across the bus is yours to make rather than one the engine makes unasked.
Page-locked host memory counts as host memory here, since the driver reports it as such.

The single-value out-parameters are the exception and take ordinary host addresses: `matches_total`, `matches_found`, `output_bytes_written`, and every `error_message`.
A CPU scope has no such requirement and reads and writes host memory throughout.

Pass the resulting `sz_memory_allocator_t *` as the `alloc` argument of any engine's `*_init`, or pass `NULL` to use the default allocator.

### The Cross Product Convention

Every similarity engine scores each `queries[query_index]` against each `candidates[candidate_index]` and writes cell `(query_index, candidate_index)` to `results[query_index * results_row_stride + candidate_index]`.
The `results_row_stride` is the number of elements between consecutive query rows, and must be at least the candidate count, so the matrix can live inside a wider allocation.
Passing `candidates == NULL` requests symmetric self-similarity of `queries`: the lower triangle is computed and mirrored, with `rows == columns`.

## Edit Distances

The Levenshtein engines compute the minimum-cost edit distance between strings as an unsigned `sz_size_t` matrix.
`szs_levenshtein_distances` compares __byte by byte__, while `szs_levenshtein_distances_utf8` compares __codepoint by codepoint__ for correct results on multibyte UTF-8 text.

Both `*_init` calls take per-edit costs and a gap model.
With `open == extend` the gaps are __linear__ (every inserted or deleted run costs the same per position); with `extend < open` the gaps are __affine__ (opening a gap is dearer than extending it).

```c
sz_status_t szs_levenshtein_distances_init(
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend,
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,
    szs_levenshtein_distances_t *engine, char const **error_message);

sz_status_t szs_levenshtein_distances(
    szs_levenshtein_distances_t engine, szs_device_scope_t device,
    sz_sequence_t const *queries, sz_sequence_t const *candidates,
    sz_size_t *results, sz_size_t results_row_stride,
    char const **error_message);

void szs_levenshtein_distances_free(szs_levenshtein_distances_t engine);
```

The UTF-8 variant mirrors this exactly with `szs_levenshtein_distances_utf8_init`, `szs_levenshtein_distances_utf8`, and `szs_levenshtein_distances_utf8_free`, plus `_u32tape` and `_u64tape` call variants.

A self-contained byte-distance example over a tiny corpus:

```c
#include <assert.h>
#include <stringzillas/stringzillas.h>

void run(void) {
    char const *strings[] = {"listen", "silent", "kitten"};
    sz_sequence_t queries;
    sz_sequence_from_null_terminated_strings(strings, 3, &queries);

    szs_device_scope_t device = NULL;
    char const *error = NULL;
    sz_status_t status = szs_device_scope_init_default(&device, &error);
    assert(status == sz_success_k);

    szs_levenshtein_distances_t engine = NULL;
    status = szs_levenshtein_distances_init(
        0, 1, 1, 1, // ... match, mismatch, open, extend (linear gaps)
        NULL, szs_capabilities(), &engine, &error);
    assert(status == sz_success_k);

    sz_size_t results[9];
    status = szs_levenshtein_distances(engine, device, &queries, NULL, results, 3, &error);
    assert(status == sz_success_k);
    assert(results[0] == 0); // ... distance of "listen" to itself

    szs_levenshtein_distances_free(engine);
    szs_device_scope_free(device);
}
```

## Alignment Scores

For sequence alignment the engines maximize a __signed__ similarity score, written into an `sz_ssize_t` matrix.
`szs_needleman_wunsch_scores` performs __global__ alignment (Needleman-Wunsch, end-to-end), while `szs_smith_waterman_scores` performs __local__ alignment (Smith-Waterman, best-scoring subsequence).
These are the workhorses of __bioinformatics__: aligning DNA, RNA, and protein sequences against reference databases at scale.

Both engines take a __compact, class-based substitution matrix__ instead of a full 256x256 table.
A 256-entry `byte_to_class` array maps each input byte to one of 32 character classes, and a row-major 32x32 `class_substitution_costs` matrix gives the signed cost between any two classes.
This folds the common biological alphabets (and ASCII case-folding tables) into a SIMD-friendly 32-class form while still expressing matrices like BLOSUM62 or NUC44.
Gap costs use the same `open` and `extend` pair as the edit-distance engines.

```c
sz_status_t szs_needleman_wunsch_scores_init(
    sz_u8_t const *byte_to_class, sz_error_cost_t const *class_substitution_costs,
    sz_error_cost_t open, sz_error_cost_t extend,
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,
    szs_needleman_wunsch_scores_t *engine, char const **error_message);

sz_status_t szs_needleman_wunsch_scores(
    szs_needleman_wunsch_scores_t engine, szs_device_scope_t device,
    sz_sequence_t const *queries, sz_sequence_t const *candidates,
    sz_ssize_t *results, sz_size_t results_row_stride,
    char const **error_message);

void szs_needleman_wunsch_scores_free(szs_needleman_wunsch_scores_t engine);
```

Smith-Waterman exposes the identical shape through `szs_smith_waterman_scores_init`, `szs_smith_waterman_scores`, and `szs_smith_waterman_scores_free`, and both engines provide `_u32tape` and `_u64tape` call variants.

A self-contained protein-style alignment setup:

```c
#include <assert.h>
#include <stringzillas/stringzillas.h>

void run(void) {
    sz_u8_t byte_to_class[256] = {0};        // ... map each byte to one of 32 classes
    sz_error_cost_t class_costs[32 * 32] = {0}; // ... row-major 32x32 substitution costs

    szs_device_scope_t device = NULL;
    char const *error = NULL;
    sz_status_t status = szs_device_scope_init_default(&device, &error);
    assert(status == sz_success_k);

    szs_smith_waterman_scores_t engine = NULL;
    status = szs_smith_waterman_scores_init(
        byte_to_class, class_costs,
        4, 1, // ... gap open and extend (affine: extend < open)
        NULL, szs_capabilities(), &engine, &error);
    assert(status == sz_success_k);

    char const *proteins[] = {"MKExample", "MKExampla"};
    sz_size_t lengths[] = {9, 9};
    sz_sequence_t queries;
    queries.count = 2;
    queries.handle = proteins;
    queries.get_start = NULL;
    queries.get_length = NULL;

    sz_ssize_t results[4];
    status = szs_smith_waterman_scores(engine, device, &queries, NULL, results, 2, &error);
    assert(status == sz_success_k);

    szs_smith_waterman_scores_free(engine);
    szs_device_scope_free(device);
    (void)lengths;
}
```

## Multi-Pattern Search

The substrings engine compiles a whole needle set into one Aho-Corasick automaton and walks every haystack against all of them at once, so a dictionary of thousands of terms costs one pass rather than thousands.
Unlike the other engines it is constructed in two steps: `szs_substrings_init` picks the backend and allocates the handle, while `szs_substrings_index` compiles a needle set into it and sizes the automaton's hot and cold tiers against the cache that will walk it.
Re-indexing replaces the needle set, so one engine serves a new vocabulary or another device without being rebuilt.

```c
sz_status_t szs_substrings_init(
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,
    szs_substrings_t *engine, char const **error_message);

sz_status_t szs_substrings_index(
    szs_substrings_t engine, sz_sequence_t const *needles,
    szs_substrings_case_sensitivity_t case_sensitivity, szs_device_scope_t device,
    char const **error_message);

sz_status_t szs_substrings_count(
    szs_substrings_t engine, szs_device_scope_t device,
    sz_sequence_t const *haystacks, szs_substrings_overlap_policy_t overlap_policy,
    sz_size_t *counts, sz_size_t *matches_total,
    char const **error_message);

sz_status_t szs_substrings_find(
    szs_substrings_t engine, szs_device_scope_t device,
    sz_sequence_t const *haystacks, szs_substrings_overlap_policy_t overlap_policy,
    szs_substrings_match_t *matches, sz_size_t matches_capacity, sz_size_t *matches_found,
    char const **error_message);

void szs_substrings_free(szs_substrings_t engine);
```

`szs_substrings_cased_k` matches raw bytes and accepts any needle, including malformed UTF-8, while `szs_substrings_uncased_k` applies full Unicode case folding and requires valid UTF-8 on both sides.
Folding does not preserve byte length — needle `k` matches both the 1-byte `k` and the 3-byte Kelvin sign `U+212A` — which is why every `szs_substrings_match_t` carries its own `byte_length` rather than borrowing the needle's.

The overlap policy decides how matches that collide are resolved: `szs_substrings_overlapping_k` reports all of them, while `szs_substrings_leftmost_longest_k` and `szs_substrings_leftmost_first_k` each keep one non-overlapping cover.
`szs_substrings_find` follows the size-query convention — passing `matches_capacity == 0` writes nothing and returns `sz_unexpected_dimensions_k` with `matches_found` holding the count a full call would need.

Two more operations share the same automaton.
`szs_substrings_score_bm25` treats the dictionary itself as the query, taking one `needle_weights` entry per needle and returning one score per haystack; its term frequencies are raw overlapping counts, which is classic BM25, so no overlap policy applies.
`szs_substrings_replace_u32tape` and its 64-bit sibling rewrite every haystack, substituting each match with its needle's replacement, and accept only the tape layouts, since a rewrite's product is itself a tape.
Size their `output_data` with `szs_substrings_replace_bound`, which answers from the needle set alone and needs no haystacks, no walk, and no device.

```c
#include <assert.h>
#include <stringzillas/stringzillas.h>

void multi_pattern_example(void) {
    char const *error = NULL;
    szs_device_scope_t device = NULL;
    sz_status_t status = szs_device_scope_init_default(&device, &error);
    assert(status == sz_success_k);

    char const *needle_texts[] = {"he", "she", "his", "hers"};
    sz_size_t const needle_count = 4;
    sz_sequence_t needles;
    sz_sequence_from_null_terminated_strings(needle_texts, needle_count, &needles);

    szs_substrings_t engine = NULL;
    status = szs_substrings_init(NULL, sz_cap_serial_k, &engine, &error);
    assert(status == sz_success_k);
    status = szs_substrings_index(engine, &needles, szs_substrings_cased_k, device, &error);
    assert(status == sz_success_k);

    char const *haystack_texts[] = {"ushers", "hershey"};
    sz_sequence_t haystacks;
    sz_sequence_from_null_terminated_strings(haystack_texts, 2, &haystacks);

    sz_size_t counts[2], matches_total = 0;
    status = szs_substrings_count(engine, device, &haystacks, szs_substrings_overlapping_k, //
                                  counts, &matches_total, &error);
    assert(status == sz_success_k);
    assert(matches_total == 7);

    szs_substrings_free(engine);
    szs_device_scope_free(device);
}
```

On a GPU scope every buffer above is device-accessible, `counts` and the BM25 weights included, as [Unified Memory](#unified-memory) spells out; `matches_total`, `matches_found`, and `output_bytes_written` stay ordinary host addresses.

## Fingerprints

The fingerprints engine sketches each string into a fixed-width MinHash plus a Count-Min-Sketch, so two near-duplicate documents land on overlapping hash dimensions even after small edits.
This powers __near-duplicate detection__ and __multi-pattern search__ across __web-scale__ corpora: a single pass over a tape yields a compact signature per document that you can index, cluster, or compare cheaply.

The engine rolls several hash windows of different widths over each string and keeps the minimum hash per dimension.
`dimensions` is the total fingerprint width (ideally a multiple of `64 * window_widths_count`); `alphabet_size` tunes the rolling base (256 for binary, 128 for ASCII, 4 for DNA, 22 for protein); `window_widths` lists the n-gram window sizes, or `NULL` for the defaults `[3, 4, 5, 7, 9, 11, 15, 31]`.
A `seed` makes every per-dimension multiplier reproducible.

```c
sz_status_t szs_fingerprints_init(
    sz_size_t dimensions, sz_size_t alphabet_size,
    sz_size_t const *window_widths, sz_size_t window_widths_count, sz_u64_t seed,
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,
    szs_fingerprints_t *engine, char const **error_message);

sz_status_t szs_fingerprints_sequence(
    szs_fingerprints_t engine, szs_device_scope_t device,
    sz_sequence_t const *texts,
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,
    sz_u32_t *min_counts, sz_size_t min_counts_stride,
    char const **error_message);

void szs_fingerprints_free(szs_fingerprints_t engine);
```

Each call writes the MinHash sketch into `min_hashes` and the Count-Min-Sketch into `min_counts`, with the per-text strides giving the byte distance between consecutive results.
Tape inputs use `szs_fingerprints_u32tape` and `szs_fingerprints_u64tape`.

Aim for at least 64 dimensions per window width: that saturates AVX-512 register utilization and activates a full 32-thread CUDA warp (64 for an AMD wave), and for Tweet-sized strings `64` dimensions of each of `[3, 5, 7, 9]` is a solid default.

```c
#include <assert.h>
#include <stringzillas/stringzillas.h>

void run(void) {
    szs_device_scope_t device = NULL;
    char const *error = NULL;
    sz_status_t status = szs_device_scope_init_default(&device, &error);
    assert(status == sz_success_k);

    szs_fingerprints_t engine = NULL;
    status = szs_fingerprints_init(
        256, 256, // ... 256 dimensions, byte alphabet
        NULL, 0,  // ... default window widths
        42,       // ... reproducibility seed
        NULL, szs_capabilities(), &engine, &error);
    assert(status == sz_success_k);

    // One tape of two documents, and both output arrays, in memory either backend can reach.
    char *data = (char *)szs_unified_alloc(38);
    sz_u32_t *offsets = (sz_u32_t *)szs_unified_alloc(3 * sizeof(sz_u32_t));
    sz_u32_t *hashes = (sz_u32_t *)szs_unified_alloc(2 * 256 * sizeof(sz_u32_t));
    sz_u32_t *counts = (sz_u32_t *)szs_unified_alloc(2 * 256 * sizeof(sz_u32_t));
    memcpy(data, "the quick brown foxthe quick brown dog", 38);
    offsets[0] = 0, offsets[1] = 19, offsets[2] = 38;

    sz_sequence_u32tape_t texts;
    texts.data = data;
    texts.offsets = offsets;
    texts.count = 2;

    status = szs_fingerprints_u32tape(
        engine, device, &texts,
        hashes, 256 * sizeof(sz_u32_t),
        counts, 256 * sizeof(sz_u32_t),
        &error);
    assert(status == sz_success_k);

    szs_unified_free(data, 38);
    szs_unified_free(offsets, 3 * sizeof(sz_u32_t));
    szs_unified_free(hashes, 2 * 256 * sizeof(sz_u32_t));
    szs_unified_free(counts, 2 * 256 * sizeof(sz_u32_t));
    szs_fingerprints_free(engine);
    szs_device_scope_free(device);
}
```

## Devices and Parallelism

The device scope is the single knob for __where__ and __how widely__ an engine runs, and the same engine handle can be driven by any scope.

| Scope      | Initializer                        | Targets                 |
| ---------- | ---------------------------------- | ----------------------- |
| Default    | `szs_device_scope_init_default`    | System defaults         |
| CPU slice  | `szs_device_scope_init_cpu_cores`  | A fraction of CPU cores |
| GPU device | `szs_device_scope_init_gpu_device` | One CUDA GPU            |

A CPU slice spreads the cross-product (or the corpus of texts) across a thread pool, so you can reserve cores for other work by asking for fewer than all of them.
A GPU device offloads the whole batch to one CUDA device, where each engine routes string pairs into size-tiered kernels.
Every buffer a GPU scope touches must be device-accessible, which the __Unified Memory__ section above states in full.

The underlying C++ engines are templated on an __executor__: `dummy_executor_t` runs serially, and `forkunion_executor_t` is the preferred library-grade thread pool, wrapping a [ForkUnion](https://github.com/ashvardanian/ForkUnion) pool through its C API so the compiled runtime handles NUMA-aware placement.
The C ABI hides this choice behind the device scope, picking the right executor for the cores or GPU you requested.

## Runtime Dispatch and Capabilities

Each `*_init` takes a `sz_capability_t` mask that pins which instruction set the engine uses.
Query the current machine's capabilities and pass them straight through to let StringZillas pick the best available backend:

```c
sz_capability_t szs_capabilities(void);
```

On x86 this dispatches across Westmere, Goldmont, Haswell (AVX2), Skylake, and Ice Lake (AVX-512) backends; on Arm across NEON, SVE, and SVE2; and on CUDA across the base SIMT, Kepler video-SIMD, and Hopper DPX tiers.
The precompiled libraries carry every backend for their platform and select per call at runtime, so one binary runs optimally across a fleet of mixed hardware.
You can read back what a device scope resolved to with `szs_device_scope_get_capabilities`.

Version helpers round out the surface:

```c
int szs_version_major(void);
int szs_version_minor(void);
int szs_version_patch(void);
```
