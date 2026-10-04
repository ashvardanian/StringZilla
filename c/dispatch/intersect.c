/**
 *  @file c/dispatch/intersect.c
 *  @author Ash Vardanian
 *  @date March 7, 2025
 *  @brief The unordered string-set intersection capability list, dispatch point and finder.
 */
#include <stringzilla/intersect.h>

#include "dispatch.h"

static sz_capability_kernels_t const *sz_sequence_intersect_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_sequence_intersect_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_sequence_intersect_westmere,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_sequence_intersect_icelake,
#endif
#if STRINGZILLA_TARGET_NEONAES
        (sz_kernel_punned_t)&sz_sequence_intersect_neonaes,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neonaes_k * STRINGZILLA_TARGET_NEONAES,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

STRINGZILLA_API sz_status_t sz_sequence_intersect_best(                        //
    sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence, //
    sz_allocator_t *allocator, sz_u64_t seed, sz_size_t *intersection_count,   //
    sz_size_t *first_positions, sz_size_t *second_positions,                   //
    sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_sequence_intersect_t const kernel = (sz_kernel_sequence_intersect_t)sz_kernel_pick_(
        capabilities, sz_sequence_intersect_capabilities());
    return kernel ? kernel(first_sequence, second_sequence, allocator, seed, intersection_count, first_positions,
                           second_positions, stream)
                  : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_intersect_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                     sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_sequence_intersect_k: lists = sz_sequence_intersect_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
