/**
 *  @brief  StringZilla is a collection of advanced string algorithms, designed to be used in Big Data applications.
 *          It is generally faster than LibC, and has a broader & cleaner interface for safer @b length-bounded strings.
 *          On modern CPUs it uses AVX2, AVX-512, NEON, SVE, & SVE2 @b SIMD instructions & provides SWAR for older CPUs.
 *          On @b CUDA-capable GPUs it also provides C++ kernels for bulk processing.
 *
 *  @file   stringcuzilla.h
 *  @author Ash Vardanian
 */
#ifndef STRINGCUZILLA_H_
#define STRINGCUZILLA_H_

#include "stringzilla.h"

#ifdef __cplusplus
extern "C" {
#endif

SZ_DYNAMIC sz_status_t sz_levenshtein_distances_u32tape( //
    sz_cptr_t a_data, sz_u32_t const *a_lengths,         //
    sz_cptr_t b_data, sz_u32_t const *b_lengths,         //
    sz_size_t count,                                     //
    sz_size_t bound,                                     //
    sz_memory_allocator_t *alloc, sz_size_t *results);

SZ_DYNAMIC sz_status_t sz_needleman_wunsch_scores_u32tape( //
    sz_cptr_t a_data, sz_u32_t const *a_lengths,           //
    sz_cptr_t b_data, sz_u32_t const *b_lengths,           //
    sz_size_t count,                                       //
    sz_error_cost_t const *subs, sz_error_cost_t gap,      //
    sz_memory_allocator_t *alloc, sz_ssize_t *results);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // STRINGCUZILLA_H_
