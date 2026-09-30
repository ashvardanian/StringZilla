/**
 *  @file c/dispatch/sort.c
 *  @author Ash Vardanian
 *  @date February 15, 2025
 *  @brief The cased and uncased arg-sort capability lists, dispatch points and finder.
 */
#include <stringzilla/sort.h>

#include "dispatch.h"

static sz_capability_kernels_t const *sz_sequence_argsort_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_sequence_argsort_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_sequence_argsort_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_sequence_argsort_skylake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_sequence_argsort_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        (sz_kernel_punned_t)&sz_sequence_argsort_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_sequence_argsort_rvv,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_sequence_argsort_uncased_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_sequence_argsort_uncased_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_sequence_argsort_uncased_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_sequence_argsort_uncased_skylake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_sequence_argsort_uncased_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        (sz_kernel_punned_t)&sz_sequence_argsort_uncased_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_sequence_argsort_uncased_rvv,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

STRINGZILLA_API sz_status_t sz_sequence_argsort_best(sz_sequence_t const *sequence, sz_size_t top_count,
                                                     sz_bool_t reverse, sz_memory_allocator_t *allocator,
                                                     sz_size_t *order, sz_capability_t capabilities, void *stream) {
    sz_kernel_sequence_argsort_t const kernel = (sz_kernel_sequence_argsort_t)sz_kernel_pick_(
        capabilities, sz_sequence_argsort_capabilities());
    return kernel ? kernel(sequence, top_count, reverse, allocator, order, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_sequence_argsort_uncased_best(              //
    sz_sequence_t const *sequence, sz_size_t top_count, sz_bool_t reverse, //
    sz_memory_allocator_t *allocator, sz_size_t *order,                    //
    sz_capability_t capabilities, void *stream) {
    sz_kernel_sequence_argsort_t const kernel = (sz_kernel_sequence_argsort_t)sz_kernel_pick_(
        capabilities, sz_sequence_argsort_uncased_capabilities());
    return kernel ? kernel(sequence, top_count, reverse, allocator, order, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_sort_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_sequence_argsort_k: lists = sz_sequence_argsort_capabilities(); break;
    case sz_kernel_sequence_argsort_uncased_k: lists = sz_sequence_argsort_uncased_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
