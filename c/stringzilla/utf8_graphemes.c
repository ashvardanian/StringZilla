/**
 *  @file c/stringzilla/utf8_graphemes.c
 *  @brief Per-domain dispatch shim for UAX-29 grapheme cluster segmentation.
 *  @author Ash Vardanian
 */
#include "dispatch.h"
#include <stringzilla/utf8_graphemes.h>

SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_graphemes_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps);

    impl->utf8_grapheme_find_boundaries = sz_utf8_grapheme_find_boundaries_serial;
    impl->utf8_grapheme_rfind_boundaries = sz_utf8_grapheme_rfind_boundaries_serial;

#if SZ_USE_ICELAKE
    if (caps & sz_cap_icelake_k) {
        impl->utf8_grapheme_find_boundaries = sz_utf8_grapheme_find_boundaries_icelake;
        impl->utf8_grapheme_rfind_boundaries = sz_utf8_grapheme_rfind_boundaries_icelake;
    }
#endif
}

SZ_DYNAMIC sz_size_t sz_utf8_grapheme_find_boundaries(sz_cptr_t text, sz_size_t length, sz_size_t *cluster_starts,
                                                      sz_size_t *cluster_lengths, sz_size_t clusters_capacity,
                                                      sz_size_t *bytes_consumed) {
    return sz_dispatch_table.utf8_grapheme_find_boundaries(text, length, cluster_starts, cluster_lengths,
                                                           clusters_capacity, bytes_consumed);
}

SZ_DYNAMIC sz_size_t sz_utf8_grapheme_rfind_boundaries(sz_cptr_t text, sz_size_t length, sz_size_t *cluster_starts,
                                                       sz_size_t *cluster_lengths, sz_size_t clusters_capacity,
                                                       sz_size_t *bytes_consumed) {
    return sz_dispatch_table.utf8_grapheme_rfind_boundaries(text, length, cluster_starts, cluster_lengths,
                                                            clusters_capacity, bytes_consumed);
}
