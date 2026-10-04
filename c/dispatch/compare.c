/**
 *  @file c/dispatch/compare.c
 *  @author Ash Vardanian
 *  @date January 16, 2024
 *  @brief Comparison dispatch points @c sz_equal_best and @c sz_order_best, their lists and finder.
 */
#include <stringzilla/compare.h>

#include "dispatch.h"

/*  One list per verb: a null slot, then a kernel per capability bit, ascending by bit. */

static sz_capability_kernels_t const *sz_equal_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_equal_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_equal_westmere,
#endif
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_equal_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_equal_skylake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_equal_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        (sz_kernel_punned_t)&sz_equal_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_equal_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_equal_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_equal_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_equal_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_equal_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL | sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE |
             sz_cap_neon_k * STRINGZILLA_TARGET_NEON | sz_cap_sve_k * STRINGZILLA_TARGET_SVE |
             sz_cap_rvv_k * STRINGZILLA_TARGET_RVV | sz_cap_v128_k * STRINGZILLA_TARGET_V128 |
             sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_order_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_order_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_order_westmere,
#endif
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_order_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_order_skylake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_order_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        (sz_kernel_punned_t)&sz_order_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_order_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_order_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_order_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_order_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_order_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_westmere_k * STRINGZILLA_TARGET_WESTMERE |
             sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL | sz_cap_skylake_k * STRINGZILLA_TARGET_SKYLAKE |
             sz_cap_neon_k * STRINGZILLA_TARGET_NEON | sz_cap_sve_k * STRINGZILLA_TARGET_SVE |
             sz_cap_rvv_k * STRINGZILLA_TARGET_RVV | sz_cap_v128_k * STRINGZILLA_TARGET_V128 |
             sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

STRINGZILLA_API sz_status_t sz_equal_best(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal,
                                          sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_equal_t const kernel = (sz_kernel_equal_t)sz_kernel_pick_(capabilities, sz_equal_capabilities());
    return kernel ? kernel(a, b, length, equal, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_order_best(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                          sz_ordering_t *ordering, sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_order_t const kernel = (sz_kernel_order_t)sz_kernel_pick_(capabilities, sz_order_capabilities());
    return kernel ? kernel(a, a_length, b, b_length, ordering, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_compare_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                   sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_equal_k: lists = sz_equal_capabilities(); break;
    case sz_kernel_order_k: lists = sz_order_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
