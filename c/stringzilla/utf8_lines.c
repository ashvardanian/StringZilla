/**
 *  @file c/stringzilla/utf8_lines.c
 *  @brief Per-domain dispatch shim for UAX-14 line break segmentation.
 *  @author Ash Vardanian
 */
#include "dispatch.h"
#include <stringzilla/utf8_lines.h>

SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_lines_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps);

    impl->utf8_line_find_boundaries = sz_utf8_line_find_boundaries_serial;

#if SZ_USE_ICELAKE
    if (caps & sz_cap_icelake_k) { impl->utf8_line_find_boundaries = sz_utf8_line_find_boundaries_icelake; }
#endif
}

SZ_DYNAMIC sz_size_t sz_utf8_line_find_boundaries(sz_cptr_t text, sz_size_t length, sz_size_t *line_starts,
                                                  sz_size_t *line_lengths, sz_u8_t *mandatory, sz_size_t lines_capacity,
                                                  sz_size_t *bytes_consumed) {
    return sz_dispatch_table.utf8_line_find_boundaries(text, length, line_starts, line_lengths, mandatory,
                                                       lines_capacity, bytes_consumed);
}
