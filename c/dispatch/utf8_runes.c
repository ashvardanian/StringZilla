/**
 *  @file c/dispatch/utf8_runes.c
 *  @author Ash Vardanian
 *  @date November 19, 2025
 *  @brief The UTF-8 codepoint capability lists, dispatch points for counting, find-nth, and rune
 *      unpacking, and their finder.
 */
#include <stringzilla/utf8_runes.h>

#include "dispatch.h"

static sz_capability_kernels_t const *sz_utf8_count_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_utf8_count_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_utf8_count_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_utf8_count_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_utf8_count_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_utf8_count_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_utf8_count_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_utf8_count_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_utf8_count_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_utf8_count_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_utf8_count_powervsx,
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

static sz_capability_kernels_t const *sz_utf8_seek_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_utf8_seek_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_utf8_seek_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_utf8_seek_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_utf8_seek_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_utf8_seek_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_utf8_seek_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_utf8_seek_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        (sz_kernel_punned_t)&sz_utf8_seek_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_utf8_seek_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_utf8_seek_powervsx,
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

static sz_capability_kernels_t const *sz_utf8_decode_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_utf8_decode_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_utf8_decode_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_utf8_decode_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_utf8_decode_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_utf8_decode_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_utf8_decode_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_utf8_decode_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_utf8_decode_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_utf8_decode_powervsx,
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

STRINGZILLA_API sz_status_t sz_utf8_count_best(sz_cptr_t text, sz_size_t length, sz_size_t *count,
                                               sz_capability_t capabilities, void *stream) {
    sz_kernel_utf8_count_t const kernel = (sz_kernel_utf8_count_t)sz_kernel_pick_(capabilities,
                                                                                  sz_utf8_count_capabilities());
    return kernel ? kernel(text, length, count, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_utf8_seek_best(sz_cptr_t text, sz_size_t length, sz_size_t n, sz_cptr_t *position,
                                              sz_capability_t capabilities, void *stream) {
    sz_kernel_utf8_seek_t const kernel = (sz_kernel_utf8_seek_t)sz_kernel_pick_(capabilities,
                                                                                sz_utf8_seek_capabilities());
    return kernel ? kernel(text, length, n, position, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_utf8_decode_best(       //
    sz_cptr_t text, sz_size_t length,                  //
    sz_rune_t *runes, sz_size_t runes_capacity,        //
    sz_size_t *runes_count, sz_size_t *bytes_consumed, //
    sz_capability_t capabilities, void *stream) {
    sz_kernel_utf8_decode_t const kernel = (sz_kernel_utf8_decode_t)sz_kernel_pick_(capabilities,
                                                                                    sz_utf8_decode_capabilities());
    return kernel ? kernel(text, length, runes, runes_capacity, runes_count, bytes_consumed, stream)
                  : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_utf8_runes_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                      sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_utf8_count_k: lists = sz_utf8_count_capabilities(); break;
    case sz_kernel_utf8_seek_k: lists = sz_utf8_seek_capabilities(); break;
    case sz_kernel_utf8_decode_k: lists = sz_utf8_decode_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
