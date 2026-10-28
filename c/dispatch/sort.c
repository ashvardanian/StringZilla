/**
 *  @file c/dispatch/sort.c
 *  @author Ash Vardanian
 *  @date February 15, 2025
 *  @brief The cased and uncased arg-sort capability lists, dispatch points and finder.
 */
#include <stringzilla/sort.h>

#include "dispatch.h"

static sz_capability_kernels_t const *sz_sort_order_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
                   sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
                   sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV) &
                  STRINGZILLA_PROFILE_CAPABILITIES
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_sort_order_serial,
#if STRINGZILLA_TARGET_HASWELL
        above_k & ~(sz_cap_haswell_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_sort_order_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        above_k & ~(sz_cap_skylake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_sort_order_skylake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_sort_order_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        above_k & ~(sz_cap_sve_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_sort_order_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        above_k & ~(sz_cap_rvv_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_sort_order_rvv,
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

static sz_capability_kernels_t const *sz_sort_uncased_order_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
                   sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
                   sz_cap_sve_k * STRINGZILLA_TARGET_SVE | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV) &
                  STRINGZILLA_PROFILE_CAPABILITIES
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_sort_uncased_order_serial,
#if STRINGZILLA_TARGET_HASWELL
        above_k & ~(sz_cap_haswell_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_sort_uncased_order_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        above_k & ~(sz_cap_skylake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_sort_uncased_order_skylake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_sort_uncased_order_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        above_k & ~(sz_cap_sve_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_sort_uncased_order_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        above_k & ~(sz_cap_rvv_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_sort_uncased_order_rvv,
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

STRINGZILLA_API sz_status_t sz_sort_order_best(sz_strings_view_t const *strings, sz_size_t top_count, sz_bool_t reverse,
                                               sz_allocator_t *allocator, sz_size_t *order,
                                               sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_sort_order_t const kernel = (sz_kernel_sort_order_t)sz_kernel_pick_(capabilities,
                                                                                  sz_sort_order_capabilities());
    return kernel ? kernel(strings, top_count, reverse, allocator, order, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_sort_uncased_order_best(sz_strings_view_t const *strings, sz_size_t top_count,
                                                       sz_bool_t reverse, sz_allocator_t *allocator, sz_size_t *order,
                                                       sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_sort_order_t const kernel = (sz_kernel_sort_order_t)sz_kernel_pick_(capabilities,
                                                                                  sz_sort_uncased_order_capabilities());
    return kernel ? kernel(strings, top_count, reverse, allocator, order, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_sort_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_sort_order_k: lists = sz_sort_order_capabilities(); break;
    case sz_kernel_sort_uncased_order_k: lists = sz_sort_uncased_order_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
