/**
 *  @brief  StringZillas is a collection of advanced string algorithms, designed to be used in Big Data applications.
 *          It is generally faster than LibC, and has a broader & cleaner interface for safer @b length-bounded strings.
 *          On modern CPUs it uses AVX2, AVX-512, NEON, SVE, & SVE2 @b SIMD instructions & provides SWAR for older CPUs.
 *          On @b CUDA-capable GPUs it also provides C++ kernels for bulk processing.
 *
 *  Unlike traditional StringZilla interfaces, all of the functions:
 *  - operators are stateful, and should be reused between calls;
 *  - receive large collections of string inputs instead of just one or two strings;
 *  - receive executors or thread pools to parallelize the work, targeting a fraction of CPU cores or a GPU;
 *  - support overriding a default memory allocator with a custom one, wrapped into `sz_memory_allocator_t`.
 *
 *  Under the hood, a ton of C++ templates are instantiated to handle different types of inputs, like:
 *  - `sz_sequence_t` - for a C-style `std::vector<std::string_view>`-like structure.
 *  - `sz_arrow_u32tape_t`, `sz_arrow_u64tape_t` - for Apache Arrow-compatible tapes with 32-bit and 64-bit offsets.
 *
 *  Those templates also reuse the same pre-configured operators for different thread-pool & executor types,
 *  hardware capability levels.
 *
 *  @file   stringzillas.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLAS_H_
#define STRINGZILLAS_H_

#include <stringzilla/stringzilla.h> // `sz_sequence_t` and other types

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Apache Arrow-compatible tape for non-NULL strings with 32-bit offsets.
 *  @sa `sz_arrow_u64tape_t` for larger collections.
 */
struct sz_arrow_u32tape_t {
    sz_cptr_t data;
    sz_u32_t const *offsets;
    sz_size_t count;
};

/**
 *  @brief Apache Arrow-compatible tape for non-NULL strings with 64-bit offsets.
 *  @sa `sz_arrow_u32tape_t` for smaller space-efficient collections.
 */
struct sz_arrow_u64tape_t {
    sz_cptr_t data;
    sz_u64_t const *offsets;
    sz_size_t count;
};

/**
 *  @brief Prepares the default allocator for unified memory management.
 *  @note When compiled on CUDA-capable systems, this function will use `cudaMallocManaged`.
 */
SZ_DYNAMIC sz_status_t sz_memory_allocator_init_unified(sz_memory_allocator_t *alloc);

/**
 *  Doesn't aim to provide the same level of granularity as the C++ API.
 *  It expects that the C functions will be called in bulk, generally,
 *  by just a single caller, either targeting:
 *
 *  - a single CPU core,
 *  - a fraction of CPU cores through some global thread pool,
 *  - a single GPU device.
 *
 *  Set `cpu_cores` to 0 to target all available CPU cores, to -1 to avoid CPUs, to 1 to use only calling thread.
 *  Set `gpu_device` to -1 to avoid GPUs, or to a positive device ID to target a specific GPU.
 */
typedef void *sz_device_scope_t;

SZ_DYNAMIC sz_status_t sz_device_scope_init_default(sz_device_scope_t *scope);
SZ_DYNAMIC sz_status_t sz_device_scope_init_cpu_cores(sz_size_t cpu_cores, sz_device_scope_t *scope);
SZ_DYNAMIC sz_status_t sz_device_scope_init_gpu_device(sz_size_t gpu_device, sz_device_scope_t *scope);
SZ_DYNAMIC sz_status_t sz_device_scope_free(sz_device_scope_t scope);

/*  APIs for computing edit-distances between binary and UTF-8 strings.
 *  Supports `sz_sequence_t`, `sz_arrow_u32tape_t`, and `sz_arrow_u64tape_t` inputs.
 */
typedef void *sz_levenshtein_distances_t;
typedef void *sz_levenshtein_distances_utf8_t;

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_init(                                              //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,                              //
    sz_levenshtein_distances_t *engine);

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_sequence(        //
    sz_levenshtein_distances_t engine, sz_device_scope_t device, //
    sz_sequence_t const *a, sz_sequence_t const *b,              //
    sz_size_t *results, sz_size_t results_stride);

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_u32tape(         //
    sz_levenshtein_distances_t engine, sz_device_scope_t device, //
    sz_arrow_u32tape_t const *a, sz_arrow_u32tape_t const *b,    //
    sz_size_t *results, sz_size_t results_stride);

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_u64tape(         //
    sz_levenshtein_distances_t engine, sz_device_scope_t device, //
    sz_arrow_u64tape_t const *a, sz_arrow_u64tape_t const *b,    //
    sz_size_t *results, sz_size_t results_stride);

SZ_DYNAMIC void sz_levenshtein_distances_free(sz_levenshtein_distances_t engine);

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_utf8_init(                                         //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,                              //
    sz_levenshtein_distances_utf8_t *engine);

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_utf8_sequence(        //
    sz_levenshtein_distances_utf8_t engine, sz_device_scope_t device, //
    sz_sequence_t const *a, sz_sequence_t const *b,                   //
    sz_size_t *results, sz_size_t results_stride);

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_utf8_u32tape(         //
    sz_levenshtein_distances_utf8_t engine, sz_device_scope_t device, //
    sz_arrow_u32tape_t const *a, sz_arrow_u32tape_t const *b,         //
    sz_size_t *results, sz_size_t results_stride);

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_utf8_u64tape(         //
    sz_levenshtein_distances_utf8_t engine, sz_device_scope_t device, //
    sz_arrow_u64tape_t const *a, sz_arrow_u64tape_t const *b,         //
    sz_size_t *results, sz_size_t results_stride);

SZ_DYNAMIC void sz_levenshtein_distances_utf8_free(sz_levenshtein_distances_utf8_t engine);

/*  APIs for computing similarity scores between pairs of strings.
 *  Supports `sz_sequence_t`, `sz_arrow_u32tape_t`, and `sz_arrow_u64tape_t` inputs.
 */

typedef void *sz_needleman_wunsch_scores_t;
typedef void *sz_smith_waterman_scores_t;

SZ_DYNAMIC sz_status_t sz_needleman_wunsch_scores_init(                        //
    sz_error_cost_t const *subs, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,          //
    sz_needleman_wunsch_scores_t *engine);

SZ_DYNAMIC sz_status_t sz_needleman_wunsch_scores_sequence(        //
    sz_needleman_wunsch_scores_t engine, sz_device_scope_t device, //
    sz_sequence_t const *a, sz_sequence_t const *b,                //
    sz_ssize_t *results, sz_size_t results_stride);

SZ_DYNAMIC sz_status_t sz_needleman_wunsch_scores_u32tape(         //
    sz_needleman_wunsch_scores_t engine, sz_device_scope_t device, //
    sz_arrow_u32tape_t const *a, sz_arrow_u32tape_t const *b,      //
    sz_ssize_t *results, sz_size_t results_stride);

SZ_DYNAMIC sz_status_t sz_needleman_wunsch_scores_u64tape(         //
    sz_needleman_wunsch_scores_t engine, sz_device_scope_t device, //
    sz_arrow_u64tape_t const *a, sz_arrow_u64tape_t const *b,      //
    sz_ssize_t *results, sz_size_t results_stride);

SZ_DYNAMIC void sz_needleman_wunsch_scores_free(sz_needleman_wunsch_scores_t engine);

SZ_DYNAMIC sz_status_t sz_smith_waterman_scores_init(                          //
    sz_error_cost_t const *subs, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,          //
    sz_smith_waterman_scores_t *engine);

SZ_DYNAMIC sz_status_t sz_smith_waterman_scores_sequence(        //
    sz_smith_waterman_scores_t engine, sz_device_scope_t device, //
    sz_sequence_t const *a, sz_sequence_t const *b,              //
    sz_ssize_t *results, sz_size_t results_stride);

SZ_DYNAMIC sz_status_t sz_smith_waterman_scores_u32tape(         //
    sz_smith_waterman_scores_t engine, sz_device_scope_t device, //
    sz_arrow_u32tape_t const *a, sz_arrow_u32tape_t const *b,    //
    sz_ssize_t *results, sz_size_t results_stride);

SZ_DYNAMIC sz_status_t sz_smith_waterman_scores_u64tape(         //
    sz_smith_waterman_scores_t engine, sz_device_scope_t device, //
    sz_arrow_u64tape_t const *a, sz_arrow_u64tape_t const *b,    //
    sz_ssize_t *results, sz_size_t results_stride);

SZ_DYNAMIC void sz_smith_waterman_scores_free(sz_smith_waterman_scores_t engine);

/**
 *  APIs for computing fingerprints, Min-Hashes, and Count-Min-Sketches of binary and UTF-8 strings.
 *  Supports `sz_sequence_t`, `sz_arrow_u32tape_t`, and `sz_arrow_u64tape_t` inputs.
 *
 *  @section Speed Considerations
 *
 *  For each window width you should aim for a multiple of 64 dimensions. Rolling hashes with identical window widths
 *  will share the same memory access pattern and can be effectively parallelized. For each platform, different minimum
 *  dimensions are recommended:
 *
 *  - on AVX-512 capable CPUs, take at least 8 hash-functions of each width,
 *  - on AVX-512 capable CPUs with a physical 512-bit path, take 16 or more, to increase register utilization,
 *  - on Nvidia GPUs, take at least 32 hash-functions of each width, to activate all 32 threads in a warp.
 *  - on AMD GPUs, take at least 64 hash-functions of each width, to activate all 64 threads in a wave.
 *
 *  Assuming 64 is the smallest size saturating all platforms - its a great default.
 *
 *  Having too many dimensions is also a problem, as we'll end up with a ton of redundant compute.
 *  For short Tweet-sized strings of 256 characters, 64 dimensions of each of [3, 5, 7, 9] seem like a good default.
 *  For web packets, around 1 KB memory pages, 64 dimensions of each of [3, 4, 5, 7, 9, 11, 15, 31] are a good default.
 *  For longer strings, like 4 KB memory pages, we can aim for 128 dimensions of the same widths.
 */
typedef void *sz_fingerprints_t;
typedef void *sz_fingerprints_utf8_t;

/**
 *  @brief Initializes a fingerprinting engine.
 *  @param alphabet_size The size of the alphabet (256 for binary, 128 for ASCII, 4 for DNA, 22 for protein).
 *  @param window_widths An optional array of window widths for the fingerprints, like [3, 4, 5, 7, 9, 11, 15, 31].
 *  @param window_widths_count The number of window widths in the @p window_widths array.
 *  @param dimensions_per_window_width The number of dimensions for each window width, ideally 64 or its multiple.
 *  @param alloc A memory allocator to use for allocating memory.
 *  @param device A device scope to use for parallel execution.
 *  @param engine Pointer to the initialized fingerprinting engine.
 */
SZ_DYNAMIC sz_status_t sz_fingerprints_init(                              //
    sz_size_t alphabet_size, sz_size_t const *window_widths,              //
    sz_size_t window_widths_count, sz_size_t dimensions_per_window_width, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,     //
    sz_fingerprints_t *engine);

SZ_DYNAMIC sz_status_t sz_fingerprints_sequence(        //
    sz_fingerprints_t engine, sz_device_scope_t device, //
    sz_sequence_t const *texts,                         //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,  //
    sz_u32_t *min_counts, sz_size_t min_counts_stride);

SZ_DYNAMIC sz_status_t sz_fingerprints_u64tape(         //
    sz_fingerprints_t engine, sz_device_scope_t device, //
    sz_arrow_u64tape_t const *texts,                    //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,  //
    sz_u32_t *min_counts, sz_size_t min_counts_stride);

SZ_DYNAMIC sz_status_t sz_fingerprints_u32tape(         //
    sz_fingerprints_t engine, sz_device_scope_t device, //
    sz_arrow_u32tape_t const *texts,                    //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,  //
    sz_u32_t *min_counts, sz_size_t min_counts_stride);

SZ_DYNAMIC void sz_fingerprints_free(sz_fingerprints_t engine);

SZ_DYNAMIC sz_status_t sz_fingerprints_utf8_init(                         //
    sz_size_t alphabet_size, sz_size_t const *window_widths,              //
    sz_size_t window_widths_count, sz_size_t dimensions_per_window_width, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,     //
    sz_fingerprints_utf8_t *engine);

SZ_DYNAMIC sz_status_t sz_fingerprints_utf8_sequence(        //
    sz_fingerprints_utf8_t engine, sz_device_scope_t device, //
    sz_sequence_t const *texts,                              //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,       //
    sz_u32_t *min_counts, sz_size_t min_counts_stride);

SZ_DYNAMIC sz_status_t sz_fingerprints_utf8_u64tape(         //
    sz_fingerprints_utf8_t engine, sz_device_scope_t device, //
    sz_arrow_u64tape_t const *texts,                         //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,       //
    sz_u32_t *min_counts, sz_size_t min_counts_stride);

SZ_DYNAMIC sz_status_t sz_fingerprints_utf8_u32tape(         //
    sz_fingerprints_utf8_t engine, sz_device_scope_t device, //
    sz_arrow_u32tape_t const *texts,                         //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,       //
    sz_u32_t *min_counts, sz_size_t min_counts_stride);

SZ_DYNAMIC void sz_fingerprints_utf8_free(sz_fingerprints_utf8_t engine);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // STRINGZILLAS_H_
