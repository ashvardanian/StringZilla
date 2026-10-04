/**
 *  @file c/dispatch/find.c
 *  @author Ash Vardanian
 *  @date January 16, 2024
 *  @brief Search dispatch points, `sz_find*_best` and `sz_rfind*_best`, and their finder.
 */
#include <stringzilla/find.h>

#include "dispatch.h"

/*  One list per verb: a null slot, then a kernel per capability bit, ascending by bit. */

static sz_capability_kernels_t const *sz_find_byte_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_find_byte_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_find_byte_westmere,
#endif
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_find_byte_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_find_byte_skylake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_find_byte_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        (sz_kernel_punned_t)&sz_find_byte_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_find_byte_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_find_byte_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_find_byte_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_find_byte_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_find_byte_powervsx,
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

static sz_capability_kernels_t const *sz_rfind_byte_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_rfind_byte_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_rfind_byte_westmere,
#endif
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_rfind_byte_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_rfind_byte_skylake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_rfind_byte_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        (sz_kernel_punned_t)&sz_rfind_byte_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_rfind_byte_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_rfind_byte_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_rfind_byte_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_rfind_byte_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_rfind_byte_powervsx,
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

static sz_capability_kernels_t const *sz_find_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_find_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_find_westmere,
#endif
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_find_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_find_skylake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_find_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        (sz_kernel_punned_t)&sz_find_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_find_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_find_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_find_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_find_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_find_powervsx,
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

static sz_capability_kernels_t const *sz_rfind_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_rfind_serial,
#if STRINGZILLA_TARGET_WESTMERE
        (sz_kernel_punned_t)&sz_rfind_westmere,
#endif
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_rfind_haswell,
#endif
#if STRINGZILLA_TARGET_SKYLAKE
        (sz_kernel_punned_t)&sz_rfind_skylake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_rfind_neon,
#endif
#if STRINGZILLA_TARGET_SVE
        (sz_kernel_punned_t)&sz_rfind_sve,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_rfind_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_rfind_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_rfind_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_rfind_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_rfind_powervsx,
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

static sz_capability_kernels_t const *sz_find_byteset_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_find_byteset_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_find_byteset_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_find_byteset_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_find_byteset_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_find_byteset_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_find_byteset_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_find_byteset_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_find_byteset_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_find_byteset_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_find_byteset_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2 | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_rfind_byteset_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_rfind_byteset_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_rfind_byteset_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_rfind_byteset_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_rfind_byteset_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_rfind_byteset_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_rfind_byteset_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_rfind_byteset_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_rfind_byteset_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_rfind_byteset_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_rfind_byteset_powervsx,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
             sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2 | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
             sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
             sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX | sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

STRINGZILLA_API sz_status_t sz_find_byte_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                              sz_cptr_t *match, sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_find_byte_t const kernel = (sz_kernel_find_byte_t)sz_kernel_pick_(capabilities,
                                                                                sz_find_byte_capabilities());
    return kernel ? kernel(haystack, haystack_length, needle, match, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_rfind_byte_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                               sz_cptr_t *match, sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_find_byte_t const kernel = (sz_kernel_find_byte_t)sz_kernel_pick_(capabilities,
                                                                                sz_rfind_byte_capabilities());
    return kernel ? kernel(haystack, haystack_length, needle, match, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_find_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                         sz_size_t needle_length, sz_cptr_t *match, sz_capability_t capabilities,
                                         sz_stream_t stream) {
    sz_kernel_find_t const kernel = (sz_kernel_find_t)sz_kernel_pick_(capabilities, sz_find_capabilities());
    return kernel ? kernel(haystack, haystack_length, needle, needle_length, match, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_rfind_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                          sz_size_t needle_length, sz_cptr_t *match, sz_capability_t capabilities,
                                          sz_stream_t stream) {
    sz_kernel_find_t const kernel = (sz_kernel_find_t)sz_kernel_pick_(capabilities, sz_rfind_capabilities());
    return kernel ? kernel(haystack, haystack_length, needle, needle_length, match, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_find_byteset_best(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *set,
                                                 sz_cptr_t *match, sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_find_byteset_t const kernel = (sz_kernel_find_byteset_t)sz_kernel_pick_(capabilities,
                                                                                      sz_find_byteset_capabilities());
    return kernel ? kernel(haystack, haystack_length, set, match, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_rfind_byteset_best(sz_cptr_t haystack, sz_size_t haystack_length,
                                                  sz_byteset_t const *set, sz_cptr_t *match,
                                                  sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_find_byteset_t const kernel = (sz_kernel_find_byteset_t)sz_kernel_pick_(capabilities,
                                                                                      sz_rfind_byteset_capabilities());
    return kernel ? kernel(haystack, haystack_length, set, match, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_find_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_find_k: lists = sz_find_capabilities(); break;
    case sz_kernel_rfind_k: lists = sz_rfind_capabilities(); break;
    case sz_kernel_find_byte_k: lists = sz_find_byte_capabilities(); break;
    case sz_kernel_rfind_byte_k: lists = sz_rfind_byte_capabilities(); break;
    case sz_kernel_find_byteset_k: lists = sz_find_byteset_capabilities(); break;
    case sz_kernel_rfind_byteset_k: lists = sz_rfind_byteset_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
