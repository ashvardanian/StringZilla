/**
 *  @file c/stringzilla/utf8_sentences.c
 *  @brief Per-domain dispatch shim for UAX-29 sentence segmentation.
 *  @author Ash Vardanian
 */
#include "dispatch.h"
#include <stringzilla/utf8_sentences.h>

SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_sentences_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps);

    impl->utf8_sentence_find_boundaries = sz_utf8_sentence_find_boundaries_serial;

#if SZ_USE_ICELAKE
    if (caps & sz_cap_icelake_k) { impl->utf8_sentence_find_boundaries = sz_utf8_sentence_find_boundaries_icelake; }
#endif
}

SZ_DYNAMIC sz_size_t sz_utf8_sentence_find_boundaries(sz_cptr_t text, sz_size_t length, sz_size_t *sentence_starts,
                                                      sz_size_t *sentence_lengths, sz_size_t sentences_capacity,
                                                      sz_size_t *bytes_consumed) {
    return sz_dispatch_table.utf8_sentence_find_boundaries(text, length, sentence_starts, sentence_lengths,
                                                           sentences_capacity, bytes_consumed);
}
