/**
 *  @file c/stringzilla/utf8_tokens.c
 *  @brief Per-domain dispatch shim for UTF-8 newline, whitespace, and general delimiter scanning.
 *  @author Ash Vardanian
 */
#include "dispatch.h"
#include <stringzilla/utf8_tokens.h>

SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_delimiters_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps);

    impl->utf8_newlines = sz_utf8_newlines_serial;
    impl->utf8_whitespaces = sz_utf8_whitespaces_serial;
    impl->utf8_delimiters = sz_utf8_delimiters_serial;

#if SZ_USE_HASWELL
    if (caps & sz_cap_haswell_k) {
        impl->utf8_newlines = sz_utf8_newlines_haswell;
        impl->utf8_whitespaces = sz_utf8_whitespaces_haswell;
        impl->utf8_delimiters = sz_utf8_delimiters_haswell;
    }
#endif

#if SZ_USE_ICELAKE
    if (caps & sz_cap_icelake_k) {
        impl->utf8_newlines = sz_utf8_newlines_icelake;
        impl->utf8_whitespaces = sz_utf8_whitespaces_icelake;
        impl->utf8_delimiters = sz_utf8_delimiters_icelake;
    }
#endif

#if SZ_USE_NEON
    if (caps & sz_cap_neon_k) {
        impl->utf8_newlines = sz_utf8_newlines_neon;
        impl->utf8_whitespaces = sz_utf8_whitespaces_neon;
        impl->utf8_delimiters = sz_utf8_delimiters_neon;
    }
#endif

#if SZ_USE_SVE2
    if (caps & sz_cap_sve2_k) {
        if (SZ_ENFORCE_SVE_OVER_NEON) {
            impl->utf8_newlines = sz_utf8_newlines_sve2;
            impl->utf8_whitespaces = sz_utf8_whitespaces_sve2;
        }
    }
#endif

#if SZ_USE_V128
    if (caps & sz_cap_v128_k) {
        impl->utf8_newlines = sz_utf8_newlines_v128;
        impl->utf8_whitespaces = sz_utf8_whitespaces_v128;
    }
#endif

#if SZ_USE_RVV
    if (caps & sz_cap_rvv_k) {
        impl->utf8_newlines = sz_utf8_newlines_rvv;
        impl->utf8_whitespaces = sz_utf8_whitespaces_rvv;
    }
#endif

#if SZ_USE_LASX
    if (caps & sz_cap_lasx_k) {
        impl->utf8_newlines = sz_utf8_newlines_lasx;
        impl->utf8_whitespaces = sz_utf8_whitespaces_lasx;
    }
#endif

#if SZ_USE_POWERVSX
    if (caps & sz_cap_powervsx_k) {
        impl->utf8_newlines = sz_utf8_newlines_powervsx;
        impl->utf8_whitespaces = sz_utf8_whitespaces_powervsx;
    }
#endif
}

SZ_DYNAMIC sz_size_t sz_utf8_newlines(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                      sz_size_t *match_lengths, sz_size_t matches_capacity, sz_size_t *bytes_consumed) {
    return sz_dispatch_table.utf8_newlines(text, length, match_offsets, match_lengths, matches_capacity,
                                           bytes_consumed);
}

SZ_DYNAMIC sz_size_t sz_utf8_whitespaces(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                         sz_size_t *match_lengths, sz_size_t matches_capacity,
                                         sz_size_t *bytes_consumed) {
    return sz_dispatch_table.utf8_whitespaces(text, length, match_offsets, match_lengths, matches_capacity,
                                              bytes_consumed);
}

SZ_DYNAMIC sz_size_t sz_utf8_delimiters(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                        sz_size_t *match_lengths, sz_size_t matches_capacity,
                                        sz_size_t *bytes_consumed) {
    return sz_dispatch_table.utf8_delimiters(text, length, match_offsets, match_lengths, matches_capacity,
                                             bytes_consumed);
}
