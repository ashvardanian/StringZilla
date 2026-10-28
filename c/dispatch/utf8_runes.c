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
    enum {
        above_k = (sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
                   sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
                   sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2 | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
                   sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
                   sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX |
                   sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX) &
                  STRINGZILLA_PROFILE_CAPABILITIES
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_count_serial,
#if STRINGZILLA_TARGET_HASWELL
        above_k & ~(sz_cap_haswell_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_count_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        above_k & ~(sz_cap_icelake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_count_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_count_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        above_k & ~(sz_cap_sve2_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_count_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        above_k & ~(sz_cap_rvv_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_count_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        above_k & ~(sz_cap_v128_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_count_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        above_k & ~(sz_cap_v128relaxed_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_count_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        above_k & ~(sz_cap_loongsonasx_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_count_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        above_k & ~(sz_cap_powervsx_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_count_powervsx,
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

static sz_capability_kernels_t const *sz_utf8_count_strings_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE |
                   sz_cap_neon_k * STRINGZILLA_TARGET_NEON) &
                  STRINGZILLA_PROFILE_CAPABILITIES
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_count_strings_serial,
#if STRINGZILLA_TARGET_ICELAKE
        above_k & ~(sz_cap_icelake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_count_strings_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_count_strings_neon,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

static sz_capability_kernels_t const *sz_utf8_seek_capabilities(void) {
    enum {
        above_k = (sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
                   sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
                   sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2 | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
                   sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_v128relaxed_k * STRINGZILLA_TARGET_V128RELAXED |
                   sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX |
                   sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX) &
                  STRINGZILLA_PROFILE_CAPABILITIES
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_seek_serial,
#if STRINGZILLA_TARGET_HASWELL
        above_k & ~(sz_cap_haswell_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_seek_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        above_k & ~(sz_cap_icelake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_seek_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_seek_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        above_k & ~(sz_cap_sve2_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_seek_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        above_k & ~(sz_cap_rvv_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_seek_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        above_k & ~(sz_cap_v128_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_seek_v128,
#endif
#if STRINGZILLA_TARGET_V128RELAXED
        above_k & ~(sz_cap_v128relaxed_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_seek_v128relaxed,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        above_k & ~(sz_cap_loongsonasx_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_seek_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        above_k & ~(sz_cap_powervsx_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_seek_powervsx,
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
    enum {
        above_k = (sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
                   sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_neon_k * STRINGZILLA_TARGET_NEON |
                   sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2 | sz_cap_rvv_k * STRINGZILLA_TARGET_RVV |
                   sz_cap_v128_k * STRINGZILLA_TARGET_V128 | sz_cap_loongsonasx_k * STRINGZILLA_TARGET_LOONGSONASX |
                   sz_cap_powervsx_k * STRINGZILLA_TARGET_POWERVSX) &
                  STRINGZILLA_PROFILE_CAPABILITIES
    };
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        above_k & ~(sz_cap_serial_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_decode_serial,
#if STRINGZILLA_TARGET_HASWELL
        above_k & ~(sz_cap_haswell_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_decode_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        above_k & ~(sz_cap_icelake_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_decode_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        above_k & ~(sz_cap_neon_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_decode_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        above_k & ~(sz_cap_sve2_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_decode_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        above_k & ~(sz_cap_rvv_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_decode_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        above_k & ~(sz_cap_v128_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_decode_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        above_k & ~(sz_cap_loongsonasx_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_decode_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        above_k & ~(sz_cap_powervsx_k * 2 - 1) ? STRINGZILLA_NULL : (sz_kernel_punned_t)&sz_utf8_decode_powervsx,
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
                                               sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_utf8_count_t const kernel = (sz_kernel_utf8_count_t)sz_kernel_pick_(capabilities,
                                                                                  sz_utf8_count_capabilities());
    return kernel ? kernel(text, length, count, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_utf8_count_strings_best(sz_strings_view_t const *strings, sz_size_t const *order,
                                                       sz_size_t count, sz_size_t *runes, sz_size_t *malformed_index,
                                                       sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_utf8_count_strings_t const kernel = (sz_kernel_utf8_count_strings_t)sz_kernel_pick_(
        capabilities, sz_utf8_count_strings_capabilities());
    return kernel ? kernel(strings, order, count, runes, malformed_index, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_utf8_seek_best(sz_cptr_t text, sz_size_t length, sz_size_t n, sz_cptr_t *position,
                                              sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_utf8_seek_t const kernel = (sz_kernel_utf8_seek_t)sz_kernel_pick_(capabilities,
                                                                                sz_utf8_seek_capabilities());
    return kernel ? kernel(text, length, n, position, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_utf8_decode_best(       //
    sz_cptr_t text, sz_size_t length,                  //
    sz_rune_t *runes, sz_size_t runes_capacity,        //
    sz_size_t *runes_count, sz_size_t *bytes_consumed, //
    sz_capability_t capabilities, sz_stream_t stream) {
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
    case sz_kernel_utf8_count_strings_k: lists = sz_utf8_count_strings_capabilities(); break;
    case sz_kernel_utf8_seek_k: lists = sz_utf8_seek_capabilities(); break;
    case sz_kernel_utf8_decode_k: lists = sz_utf8_decode_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
