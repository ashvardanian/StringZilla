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

    impl->utf8_graphemes = sz_utf8_graphemes_serial;

#if SZ_USE_HASWELL
    if (caps & sz_cap_haswell_k) { impl->utf8_graphemes = sz_utf8_graphemes_haswell; }
#endif
    // The 128-bit Arm fronts are intentionally NOT installed: grapheme clusters are so dense that the
    // windowed engines' fixed costs exceed the scalar walk at ANY output capacity - on Graviton 5 the serial
    // kernel leads every corpus (mixed 185 vs 64 NEON / 101 SVE2 MiB/s, Chinese 298 vs 93 / 106) - so NEON
    // stays a tested reserve and the scalable front waits for wider registers to flip the economics.
#if SZ_USE_ICELAKE
    if (caps & sz_cap_icelake_k) { impl->utf8_graphemes = sz_utf8_graphemes_icelake; }
#endif
#if SZ_USE_SVE2
    if ((caps & sz_cap_sve2_k) && sz_sve_wider_than_neon_()) { impl->utf8_graphemes = sz_utf8_graphemes_sve2; }
#endif
}

SZ_API_RUNTIME sz_size_t sz_utf8_graphemes(sz_cptr_t text, sz_size_t length, sz_size_t *cluster_starts,
                                           sz_size_t *cluster_lengths, sz_size_t clusters_capacity,
                                           sz_size_t *bytes_consumed) {
    return sz_dispatch_table.utf8_graphemes(text, length, cluster_starts, cluster_lengths, clusters_capacity,
                                            bytes_consumed);
}
