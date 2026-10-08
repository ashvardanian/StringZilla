/**
 *  @file c/dispatch/utf8_tokens.c
 *  @author Ash Vardanian
 *  @date November 18, 2025
 *  @brief UTF-8 newline, whitespace and delimiter capability lists, dispatch points and finder.
 */
#include <stringzilla/utf8_tokens.h>

#include "dispatch.h"

static sz_capability_kernels_t const *sz_utf8_newlines_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_utf8_newlines_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_utf8_newlines_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_utf8_newlines_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_utf8_newlines_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_utf8_newlines_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_utf8_newlines_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_utf8_newlines_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_utf8_newlines_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_utf8_newlines_powervsx,
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

static sz_capability_kernels_t const *sz_utf8_whitespaces_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_utf8_whitespaces_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_utf8_whitespaces_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_utf8_whitespaces_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_utf8_whitespaces_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_utf8_whitespaces_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_utf8_whitespaces_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_utf8_whitespaces_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_utf8_whitespaces_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_utf8_whitespaces_powervsx,
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

static sz_capability_kernels_t const *sz_utf8_delimiters_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_utf8_delimiters_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_utf8_delimiters_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_utf8_delimiters_icelake,
#endif
#if STRINGZILLA_TARGET_NEON
        (sz_kernel_punned_t)&sz_utf8_delimiters_neon,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_utf8_delimiters_sve2,
#endif
#if STRINGZILLA_TARGET_RVV
        (sz_kernel_punned_t)&sz_utf8_delimiters_rvv,
#endif
#if STRINGZILLA_TARGET_V128
        (sz_kernel_punned_t)&sz_utf8_delimiters_v128,
#endif
#if STRINGZILLA_TARGET_LOONGSONASX
        (sz_kernel_punned_t)&sz_utf8_delimiters_loongsonasx,
#endif
#if STRINGZILLA_TARGET_POWERVSX
        (sz_kernel_punned_t)&sz_utf8_delimiters_powervsx,
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

STRINGZILLA_API sz_status_t sz_utf8_newlines_best(                                  //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed,                            //
    sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_utf8_tokenizer_t const kernel = (sz_kernel_utf8_tokenizer_t)sz_kernel_pick_(
        capabilities, sz_utf8_newlines_capabilities());
    return kernel ? kernel(text, length, match_offsets, match_lengths, matches_capacity, matches_count, bytes_consumed,
                           stream)
                  : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_utf8_whitespaces_best(                               //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed,                            //
    sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_utf8_tokenizer_t const kernel = (sz_kernel_utf8_tokenizer_t)sz_kernel_pick_(
        capabilities, sz_utf8_whitespaces_capabilities());
    return kernel ? kernel(text, length, match_offsets, match_lengths, matches_capacity, matches_count, bytes_consumed,
                           stream)
                  : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_utf8_delimiters_best(                                //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed,                            //
    sz_capability_t capabilities, sz_stream_t stream) {
    sz_kernel_utf8_tokenizer_t const kernel = (sz_kernel_utf8_tokenizer_t)sz_kernel_pick_(
        capabilities, sz_utf8_delimiters_capabilities());
    return kernel ? kernel(text, length, match_offsets, match_lengths, matches_capacity, matches_count, bytes_consumed,
                           stream)
                  : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_utf8_tokens_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                       sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_utf8_newlines_k: lists = sz_utf8_newlines_capabilities(); break;
    case sz_kernel_utf8_whitespaces_k: lists = sz_utf8_whitespaces_capabilities(); break;
    case sz_kernel_utf8_delimiters_k: lists = sz_utf8_delimiters_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
