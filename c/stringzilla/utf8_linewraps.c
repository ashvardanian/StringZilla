/**
 *  @file c/stringzilla/utf8_linewraps.c
 *  @brief Per-domain dispatch shim for UAX-14 line break segmentation.
 *  @author Ash Vardanian
 */
#include "dispatch.h"
#include <stringzilla/utf8_linewraps.h>

SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_lines_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps);

    impl->utf8_linewraps = sz_utf8_linewraps_serial;

#if SZ_USE_ICELAKE
    if (caps & sz_cap_icelake_k) { impl->utf8_linewraps = sz_utf8_linewraps_icelake; }
#endif
}

SZ_DYNAMIC sz_size_t sz_utf8_linewraps(sz_cptr_t text, sz_size_t length, sz_size_t *line_starts,
                                       sz_size_t *line_lengths, sz_size_t lines_capacity, sz_size_t *bytes_consumed) {
    return sz_dispatch_table.utf8_linewraps(text, length, line_starts, line_lengths, lines_capacity, bytes_consumed);
}
