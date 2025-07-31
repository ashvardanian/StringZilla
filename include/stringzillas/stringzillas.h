/**
 *  @brief  StringZillas is a collection of advanced string algorithms, designed to be used in Big Data applications.
 *          It is generally faster than LibC, and has a broader & cleaner interface for safer @b length-bounded strings.
 *          On modern CPUs it uses AVX2, AVX-512, NEON, SVE, & SVE2 @b SIMD instructions & provides SWAR for older CPUs.
 *          On @b CUDA-capable GPUs it also provides C++ kernels for bulk processing.
 *
 *  Unlike traditional StringZilla interfaces, all of the functions:
 *  - receive large collections of string inputs instead of just one or two strings;
 *  - support overriding a default memory allocator with a custom one, wrapped into `sz_memory_allocator_t`;
 *  - support a custom device scope, wrapped into `sz_device_scope_t`, targeting a fraction of CPU cores or a GPU.
 *
 *  @file   stringzillas.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLAS_H_
#define STRINGZILLAS_H_

#include "stringzilla.h" // `sz_sequence_t` and other types

#ifdef __cplusplus
extern "C" {
#endif

struct sz_arrow_u32tape_t {
    sz_cptr_t data;
    sz_u32_t const *lengths;
    sz_size_t count;
};

struct sz_arrow_u64tape_t {
    sz_cptr_t data;
    sz_u64_t const *lengths;
    sz_size_t count;
};

/**
 *  Doesn't aim to provide the same level of granularity as the C++ API.
 *  It expects that the C functions will be called in bulk, generally,
 *  by just a single caller, either targeting:
 *
 *  - a single CPU core,
 *  - a fraction of CPU cores through some global thread pool,
 *  - a single GPU device.
 */
struct sz_device_scope_t {
    sz_ssize_t cpu_cores;
    sz_ssize_t gpu_device;
};

SZ_DYNAMIC void sz_allocate_at_least(                      //
    sz_size_t size_bytes, sz_device_scope_t const *device, //
    sz_ptr_t *result_address, sz_size_t *result_size);

SZ_DYNAMIC void sz_free(sz_ptr_t ptr, sz_size_t size_bytes, sz_device_scope_t const *device);

/*  APIs for computing edit-distances between binary and UTF-8 strings.
 *  Supports `sz_sequence_t`, `sz_arrow_u32tape_t`, and `sz_arrow_u64tape_t` inputs.
 */

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_sequence(                                          //
    sz_sequence_t const *a, sz_sequence_t const *b,                                                //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_size_t *results);

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_utf8_sequence(                                     //
    sz_sequence_t const *a, sz_sequence_t const *b,                                                //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_size_t *results);

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_u32tape(                                           //
    sz_arrow_u32tape_t const *a, sz_arrow_u32tape_t const *b,                                      //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_size_t *results);

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_utf8_u32tape(                                      //
    sz_arrow_u32tape_t const *a, sz_arrow_u32tape_t const *b,                                      //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_size_t *results);

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_u64tape(                                           //
    sz_arrow_u64tape_t const *a, sz_arrow_u64tape_t const *b,                                      //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_size_t *results);

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_utf8_u64tape(                                      //
    sz_arrow_u64tape_t const *a, sz_arrow_u64tape_t const *b,                                      //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_size_t *results);

/*  APIs for computing similarity scores between pairs of strings.
 *  Supports `sz_sequence_t`, `sz_arrow_u32tape_t`, and `sz_arrow_u64tape_t` inputs.
 */

SZ_DYNAMIC sz_status_t sz_needleman_wunsch_scores_sequence(                    //
    sz_sequence_t const *a, sz_sequence_t const *b,                            //
    sz_error_cost_t const *subs, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_ssize_t *results);

SZ_DYNAMIC sz_status_t sz_smith_waterman_scores_sequence(                      //
    sz_sequence_t const *a, sz_sequence_t const *b,                            //
    sz_error_cost_t const *subs, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_ssize_t *results);

SZ_DYNAMIC sz_status_t sz_needleman_wunsch_scores_u32tape(                     //
    sz_arrow_u32tape_t const *a, sz_arrow_u32tape_t const *b,                  //
    sz_error_cost_t const *subs, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_ssize_t *results);

SZ_DYNAMIC sz_status_t sz_smith_waterman_scores_u32tape(                       //
    sz_arrow_u32tape_t const *a, sz_arrow_u32tape_t const *b,                  //
    sz_error_cost_t const *subs, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_ssize_t *results);

SZ_DYNAMIC sz_status_t sz_needleman_wunsch_scores_u64tape(                     //
    sz_arrow_u64tape_t const *a, sz_arrow_u64tape_t const *b,                  //
    sz_error_cost_t const *subs, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_ssize_t *results);

SZ_DYNAMIC sz_status_t sz_smith_waterman_scores_u64tape(                       //
    sz_arrow_u64tape_t const *a, sz_arrow_u64tape_t const *b,                  //
    sz_error_cost_t const *subs, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_ssize_t *results);

/*  APIs for computing fingerprints, Min-Hashes, and Count-Min-Sketches of binary and UTF-8 strings.
 *  Supports `sz_sequence_t`, `sz_arrow_u32tape_t`, and `sz_arrow_u64tape_t` inputs.
 */

SZ_DYNAMIC sz_status_t sz_fingerprints_u64tape(                                     //
    sz_arrow_u64tape_t const *texts, sz_size_t alphabet_size, sz_size_t dimensions, //
    sz_size_t const *window_widths, sz_size_t window_widths_count,                  //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device,                  //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                              //
    sz_u32_t *min_counts, sz_size_t min_counts_stride);

SZ_DYNAMIC sz_status_t sz_fingerprints_utf8_u64tape(                                //
    sz_arrow_u64tape_t const *texts, sz_size_t alphabet_size, sz_size_t dimensions, //
    sz_size_t const *window_widths, sz_size_t window_widths_count,                  //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device,                  //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                              //
    sz_u32_t *min_counts, sz_size_t min_counts_stride);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // STRINGZILLAS_H_
