/**
 *  @file c/dispatch/utf8_graphemes.c
 *  @author Ash Vardanian
 *  @date June 20, 2026
 *  @brief The UAX-29 grapheme cluster capability list, dispatch point and finder.
 */
#include <stringzilla/utf8_graphemes.h>

#include "dispatch.h"

static sz_capability_kernels_t const *sz_utf8_graphemes_capabilities(void) {
    static sz_kernel_punned_t const cpu[] = {
        STRINGZILLA_NULL,
        (sz_kernel_punned_t)&sz_utf8_graphemes_serial,
#if STRINGZILLA_TARGET_HASWELL
        (sz_kernel_punned_t)&sz_utf8_graphemes_haswell,
#endif
#if STRINGZILLA_TARGET_ICELAKE
        (sz_kernel_punned_t)&sz_utf8_graphemes_icelake,
#endif
#if STRINGZILLA_TARGET_SVE2
        (sz_kernel_punned_t)&sz_utf8_graphemes_sve2,
#endif
    };
    static sz_capability_kernels_t const lists[sz_capability_groups_k] = {
        {sz_cap_serial_k | sz_cap_haswell_k * STRINGZILLA_TARGET_HASWELL |
             sz_cap_icelake_k * STRINGZILLA_TARGET_ICELAKE | sz_cap_sve2_k * STRINGZILLA_TARGET_SVE2,
         cpu},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
        {0, sz_no_kernels_},
    };
    return lists;
}

STRINGZILLA_API sz_status_t sz_utf8_graphemes_best(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                   sz_size_t capacity, sz_size_t *count, sz_capability_t capabilities,
                                                   sz_stream_t stream) {
    sz_kernel_utf8_segmenter_t const kernel = (sz_kernel_utf8_segmenter_t)sz_kernel_pick_(
        capabilities, sz_utf8_graphemes_capabilities());
    return kernel ? kernel(text, length, lengths, capacity, count, stream) : sz_missing_kernel_k;
}

STRINGZILLA_API sz_status_t sz_utf8_graphemes_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                          sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_capability_kernels_t const *lists = STRINGZILLA_NULL;
    switch (kind) {
    case sz_kernel_utf8_graphemes_k: lists = sz_utf8_graphemes_capabilities(); break;
    default: break;
    }
    *kernel = lists ? sz_kernel_pick_(capabilities, lists) : (sz_kernel_punned_t)STRINGZILLA_NULL;
    *capability = *kernel ? sz_capability_pick_(capabilities, lists) : 0;
    return *kernel ? sz_success_k : sz_missing_kernel_k;
}
