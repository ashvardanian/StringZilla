/**
 *  @file c/stringzilla/utf8_linebreaks.c
 *  @author Ash Vardanian
 *  @date June 20, 2026
 *  @brief Per-domain dispatch shim for UAX-14 line break segmentation.
 */
#include <stringzilla/utf8_linebreaks.h>

#include "dispatch.h"

STRINGZILLA_DISPATCH_INTERNAL void sz_dispatch_utf8_linebreaks_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_cpu_table;
    sz_unused_(caps);

    impl->utf8_linebreaks = sz_utf8_linebreaks_serial;

#if STRINGZILLA_TARGET_HASWELL
    if (caps & sz_cap_haswell_k) { impl->utf8_linebreaks = sz_utf8_linebreaks_haswell; }
#endif
#if STRINGZILLA_TARGET_NEON
    if (caps & sz_cap_neon_k) { impl->utf8_linebreaks = sz_utf8_linebreaks_neon; }
#endif
#if STRINGZILLA_TARGET_ICELAKE
    if (caps & sz_cap_icelake_k) { impl->utf8_linebreaks = sz_utf8_linebreaks_icelake; }
#endif
#if STRINGZILLA_TARGET_SVE2
    // Wider-than-NEON registers are where the scalable front wins; measure before un-gating at 128 bits.
    if ((caps & sz_cap_sve2_k) && sz_sve_wider_than_neon_()) { impl->utf8_linebreaks = sz_utf8_linebreaks_sve2; }
#endif
}

STRINGZILLA_API_RUNTIME sz_size_t sz_utf8_linebreaks(sz_cptr_t text, sz_size_t length, sz_size_t *line_starts,
                                                     sz_size_t *line_lengths, sz_size_t lines_capacity,
                                                     sz_size_t *bytes_consumed) {
    return sz_dispatch_cpu_table.utf8_linebreaks(text, length, line_starts, line_lengths, lines_capacity, bytes_consumed);
}
