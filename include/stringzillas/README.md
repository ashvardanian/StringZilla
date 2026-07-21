# StringZillas for C and C++

StringZillas is a __bulk-processing framework__ for __fingerprinting and fuzzy matching__ of __web-scale text corpora and bioinformatics sequences__.
Unlike the single-string `stringzilla` library, every operation here is a __stateful engine__ that you initialize once and reuse across many calls.
Each engine consumes __large collections__ of strings at once, scoring them as a cross-product matrix or sketching them into compact fingerprints.
And each call runs across a chosen slice of __CPU cores or a CUDA GPU__, selected through a lightweight device handle you pass on every call.

## Installation

StringZillas is __compiled__, not header-only: the heavy template instantiations for every engine are baked into precompiled libraries so you link them rather than recompile them.
The CMake project is named `stringzilla`, and it exposes four StringZillas library targets gated by two options.

The CPU libraries `stringzillas_cpus_shared` and `stringzillas_cpus_static` are built when `STRINGZILLAS_BUILD_SHARED` is on.
The CUDA libraries `stringzillas_cuda_shared` and `stringzillas_cuda_static` are additionally built when `STRINGZILLA_BUILD_CUDA` is on.
Every target also gets a namespaced alias under `stringzilla::`, for example `stringzilla::stringzillas_cpus_shared`.

There is no `find_package` config export, so pull the project in with `FetchContent` (or `add_subdirectory`) and link a target:

```cmake
include(FetchContent)
FetchContent_Declare(
    stringzilla
    GIT_REPOSITORY https://github.com/ashvardanian/stringzilla
    GIT_TAG main)
set(STRINGZILLAS_BUILD_SHARED ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(stringzilla)

add_executable(my_app main.c)
target_link_libraries(my_app PRIVATE stringzilla::stringzillas_cpus_shared)
```

To also build and link the GPU engines, enable CUDA and link the matching target:

```cmake
set(STRINGZILLA_BUILD_CUDA ON CACHE BOOL "" FORCE)
# ... after FetchContent_MakeAvailable ...
target_link_libraries(my_app PRIVATE stringzilla::stringzillas_cuda_shared)
```

The static variants `stringzillas_cpus_static` and `stringzillas_cuda_static` ship the same engines for local testing and benchmarking.
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

For zero-copy sharing between the CPU and the GPU, initialize the allocator with unified memory.
On CUDA-capable systems it uses `cudaMallocManaged`, so the same buffers are reachable from both host and device.

```c
sz_status_t sz_memory_allocator_init_unified(sz_memory_allocator_t *alloc, char const **error_message);
void *szs_unified_alloc(sz_size_t size_bytes);
void szs_unified_free(void *ptr, sz_size_t size_bytes);
```

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
    sz_size_t lengths[] = {6, 6, 6};
    sz_sequence_t queries;
    queries.count = 3;
    queries.handle = strings;
    queries.get_start = NULL; // ... wire your own getters in real code
    queries.get_length = NULL;

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
    (void)lengths;
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

    char const *docs[] = {"the quick brown fox", "the quick brown dog"};
    sz_size_t lengths[] = {19, 19};
    sz_sequence_t texts;
    texts.count = 2;
    texts.handle = docs;
    texts.get_start = NULL;
    texts.get_length = NULL;

    sz_u32_t hashes[2 * 256];
    sz_u32_t counts[2 * 256];
    status = szs_fingerprints_sequence(
        engine, device, &texts,
        hashes, 256 * sizeof(sz_u32_t),
        counts, 256 * sizeof(sz_u32_t),
        &error);
    assert(status == sz_success_k);

    szs_fingerprints_free(engine);
    szs_device_scope_free(device);
    (void)lengths;
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
Unified memory is what makes the GPU path seamless: allocate inputs and outputs through the unified allocator, and the same pointers are valid on host and device with no explicit copies.

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
