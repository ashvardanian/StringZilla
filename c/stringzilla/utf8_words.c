/**
 *  @file c/stringzilla/utf8_words.c
 *  @brief Per-domain dispatch shim for UAX-29 word boundary segmentation.
 *  @author Ash Vardanian
 */
#include "dispatch.h"
#include <stringzilla/utf8_words.h>

SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_words_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps);

    impl->utf8_word_find_boundaries = sz_utf8_word_find_boundaries_serial;
    impl->utf8_word_rfind_boundaries = sz_utf8_word_rfind_boundaries_serial;

#if SZ_USE_ICELAKE
    if (caps & sz_cap_icelake_k) {
        impl->utf8_word_find_boundaries = sz_utf8_word_find_boundaries_icelake;
        impl->utf8_word_rfind_boundaries = sz_utf8_word_rfind_boundaries_icelake;
    }
#endif

#if SZ_USE_V128
    if (caps & sz_cap_v128_k) {
        impl->utf8_word_find_boundaries = sz_utf8_word_find_boundaries_v128;
        impl->utf8_word_rfind_boundaries = sz_utf8_word_rfind_boundaries_v128;
    }
#endif

#if SZ_USE_RVV
    if (caps & sz_cap_rvv_k) {
        impl->utf8_word_find_boundaries = sz_utf8_word_find_boundaries_rvv;
        impl->utf8_word_rfind_boundaries = sz_utf8_word_rfind_boundaries_rvv;
    }
#endif

#if SZ_USE_LASX
    if (caps & sz_cap_lasx_k) {
        impl->utf8_word_find_boundaries = sz_utf8_word_find_boundaries_lasx;
        impl->utf8_word_rfind_boundaries = sz_utf8_word_rfind_boundaries_lasx;
    }
#endif

#if SZ_USE_POWERVSX
    if (caps & sz_cap_powervsx_k) {
        impl->utf8_word_find_boundaries = sz_utf8_word_find_boundaries_powervsx;
        impl->utf8_word_rfind_boundaries = sz_utf8_word_rfind_boundaries_powervsx;
    }
#endif
}

SZ_DYNAMIC sz_size_t sz_utf8_word_find_boundaries(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                  sz_size_t *word_lengths, sz_size_t words_capacity,
                                                  sz_size_t *bytes_consumed) {
    return sz_dispatch_table.utf8_word_find_boundaries(text, length, word_starts, word_lengths, words_capacity,
                                                       bytes_consumed);
}

SZ_DYNAMIC sz_size_t sz_utf8_word_rfind_boundaries(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                   sz_size_t *word_lengths, sz_size_t words_capacity,
                                                   sz_size_t *bytes_consumed) {
    return sz_dispatch_table.utf8_word_rfind_boundaries(text, length, word_starts, word_lengths, words_capacity,
                                                        bytes_consumed);
}
