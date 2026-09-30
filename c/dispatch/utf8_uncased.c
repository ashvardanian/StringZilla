/**
 *  @file c/dispatch/utf8_uncased.c
 *  @author Ash Vardanian
 *  @date November 23, 2025
 *  @brief The uncased search preparation, search, order and cased-finder capability lists, dispatch
 *      points and finder.
 *
 *  Isolated from the folding unit, `utf8_uncased_fold.c`, because the AVX-512 per-script find
 *  kernels in `utf8_uncased/icelake.h` are by far the heaviest single compilation in the core;
 *  their own translation unit lets the rest of the UTF-8 case domain build in parallel.
 */
#include <stringzilla/utf8_uncased.h>

#include "dispatch.h"

static sz_capability_kernels_t const *sz_utf8_uncased_needle_init_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_utf8_uncased_needle_init_serial,
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k, cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_utf8_uncased_search_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_utf8_uncased_search_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_utf8_uncased_search_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_utf8_uncased_search_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_utf8_uncased_search_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_utf8_uncased_search_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_utf8_uncased_search_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_utf8_uncased_search_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_utf8_uncased_search_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_utf8_uncased_search_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2 | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_utf8_uncased_order_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_utf8_uncased_order_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_utf8_uncased_order_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_utf8_uncased_order_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_utf8_uncased_order_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_utf8_uncased_order_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_utf8_uncased_order_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_utf8_uncased_order_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_utf8_uncased_order_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_utf8_uncased_order_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2 | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_utf8_find_cased_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_utf8_find_cased_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_utf8_find_cased_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_utf8_find_cased_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_utf8_find_cased_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_utf8_find_cased_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_utf8_find_cased_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_utf8_find_cased_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_utf8_find_cased_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_utf8_find_cased_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2 | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX |
             sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

STRINGZILLA_API sz_status_t sz_utf8_uncased_needle_init_best(sz_cptr_t needle, sz_size_t needle_length,
                                                             sz_utf8_uncased_needle_t *prepared,
                                                             sz_capability_t capabilities, void *stream) {
    sz_kernel_utf8_uncased_needle_init_t const kernel = (sz_kernel_utf8_uncased_needle_init_t)sz_kernel_pick_(
        capabilities, sz_utf8_uncased_needle_init_capabilities());
    return kernel ? kernel(needle, needle_length, prepared, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_utf8_uncased_search_best(sz_cptr_t haystack, sz_size_t haystack_length,
                                                        sz_utf8_uncased_needle_t const *needle, sz_cptr_t *match,
                                                        sz_size_t *match_length, sz_capability_t capabilities,
                                                        void *stream) {
    sz_kernel_utf8_uncased_search_t const kernel = (sz_kernel_utf8_uncased_search_t)sz_kernel_pick_(
        capabilities, sz_utf8_uncased_search_capabilities());
    return kernel ? kernel(haystack, haystack_length, needle, match, match_length, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_utf8_uncased_order_best( //
    sz_cptr_t a, sz_size_t a_length,                    //
    sz_cptr_t b, sz_size_t b_length,                    //
    sz_ordering_t *ordering, sz_capability_t capabilities, void *stream) {
    sz_kernel_utf8_uncased_order_t const kernel = (sz_kernel_utf8_uncased_order_t)sz_kernel_pick_(
        capabilities, sz_utf8_uncased_order_capabilities());
    return kernel ? kernel(a, a_length, b, b_length, ordering, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_utf8_find_cased_best(sz_cptr_t text, sz_size_t length, sz_cptr_t *match,
                                                    sz_capability_t capabilities, void *stream) {
    sz_kernel_utf8_find_cased_t const kernel = (sz_kernel_utf8_find_cased_t)sz_kernel_pick_(
        capabilities, sz_utf8_find_cased_capabilities());
    return kernel ? kernel(text, length, match, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_utf8_uncased_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                        sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_utf8_uncased_needle_init_k: lists = sz_utf8_uncased_needle_init_capabilities(); break;
    case sz_kernel_utf8_uncased_search_k: lists = sz_utf8_uncased_search_capabilities(); break;
    case sz_kernel_utf8_uncased_order_k: lists = sz_utf8_uncased_order_capabilities(); break;
    case sz_kernel_utf8_find_cased_k: lists = sz_utf8_find_cased_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
