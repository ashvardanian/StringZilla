/**
 *  @brief  StringZillas is a collection of advanced string algorithms, designed to be used in Big Data applications.
 *          It is generally faster than LibC, and has a broader & cleaner interface for safer @b length-bounded strings.
 *          On modern CPUs it uses AVX2, AVX-512, NEON, SVE, & SVE2 @b SIMD instructions & provides SWAR for older CPUs.
 *          On @b CUDA-capable GPUs it also provides C++ kernels for bulk processing.
 *
 *  @file   stringzillas.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLAS_H_
#define STRINGZILLAS_H_

#include "stringzilla.h"

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

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_sequence(                                          //
    sz_sequence_t const *a, sz_sequence_t const *b,                                                //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_size_t *results);

SZ_DYNAMIC sz_status_t sz_needleman_wunsch_scores_sequence(                    //
    sz_sequence_t const *a, sz_sequence_t const *b,                            //
    sz_error_cost_t const *subs, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_ssize_t *results);

SZ_DYNAMIC sz_status_t sz_smith_waterman_scores_sequence(                      //
    sz_sequence_t const *a, sz_sequence_t const *b,                            //
    sz_error_cost_t const *subs, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_ssize_t *results);

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_u32tape(                                           //
    sz_arrow_u32tape_t const *a, sz_arrow_u32tape_t const *b,                                      //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_size_t *results);

SZ_DYNAMIC sz_status_t sz_needleman_wunsch_scores_u32tape(                     //
    sz_arrow_u32tape_t const *a, sz_arrow_u32tape_t const *b,                  //
    sz_error_cost_t const *subs, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_ssize_t *results);

SZ_DYNAMIC sz_status_t sz_smith_waterman_scores_u32tape(                       //
    sz_arrow_u32tape_t const *a, sz_arrow_u32tape_t const *b,                  //
    sz_error_cost_t const *subs, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_ssize_t *results);

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_u64tape(                                           //
    sz_arrow_u64tape_t const *a, sz_arrow_u64tape_t const *b,                                      //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_size_t *results);

SZ_DYNAMIC sz_status_t sz_needleman_wunsch_scores_u64tape(                     //
    sz_arrow_u64tape_t const *a, sz_arrow_u64tape_t const *b,                  //
    sz_error_cost_t const *subs, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_ssize_t *results);

SZ_DYNAMIC sz_status_t sz_smith_waterman_scores_u64tape(                       //
    sz_arrow_u64tape_t const *a, sz_arrow_u64tape_t const *b,                  //
    sz_error_cost_t const *subs, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t *alloc, sz_device_scope_t const *device, sz_ssize_t *results);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // STRINGZILLAS_H_
