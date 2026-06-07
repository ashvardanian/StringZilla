/**
 *  @file c/stringzilla/utf8_iterate.c
 *  @brief Per-domain dispatch shim for UTF-8 iteration: count, find-nth, newline/whitespace,
 *         rune unpacking, and UAX-29 word boundaries.
 *  @author Ash Vardanian
 *  @date January 16, 2024
 */
#include "dispatch.h"
#include <stringzilla/utf8_iterate.h>

SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_iterate_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps);

    impl->utf8_count = sz_utf8_count_serial;
    impl->utf8_find_nth = sz_utf8_find_nth_serial;
    impl->utf8_find_newline = sz_utf8_find_newline_serial;
    impl->utf8_find_whitespace = sz_utf8_find_whitespace_serial;
    impl->utf8_unpack_chunk = sz_utf8_unpack_chunk_serial;
    impl->utf8_word_find_boundary = sz_utf8_word_find_boundary_serial;
    impl->utf8_word_rfind_boundary = sz_utf8_word_rfind_boundary_serial;

#if SZ_USE_HASWELL
    if (caps & sz_cap_haswell_k) {
        impl->utf8_count = sz_utf8_count_haswell;
        impl->utf8_find_nth = sz_utf8_find_nth_haswell;
        impl->utf8_find_newline = sz_utf8_find_newline_haswell;
        impl->utf8_find_whitespace = sz_utf8_find_whitespace_haswell;
    }
#endif

#if SZ_USE_ICELAKE
    if (caps & sz_cap_icelake_k) {
        impl->utf8_count = sz_utf8_count_icelake;
        impl->utf8_find_nth = sz_utf8_find_nth_icelake;
        impl->utf8_find_newline = sz_utf8_find_newline_icelake;
        impl->utf8_find_whitespace = sz_utf8_find_whitespace_icelake;
        impl->utf8_unpack_chunk = sz_utf8_unpack_chunk_icelake;
    }
#endif

#if SZ_USE_NEON
    if (caps & sz_cap_neon_k) {
        impl->utf8_count = sz_utf8_count_neon;
        impl->utf8_find_nth = sz_utf8_find_nth_neon;
        impl->utf8_find_newline = sz_utf8_find_newline_neon;
        impl->utf8_find_whitespace = sz_utf8_find_whitespace_neon;
        impl->utf8_unpack_chunk = sz_utf8_unpack_chunk_neon;
    }
#endif

#if SZ_USE_SVE2
    if (caps & sz_cap_sve2_k) {
        impl->utf8_count = sz_utf8_count_sve2;
        impl->utf8_find_nth = sz_utf8_find_nth_sve2;

        if (SZ_ENFORCE_SVE_OVER_NEON) {
            impl->utf8_find_newline = sz_utf8_find_newline_sve2;
            impl->utf8_find_whitespace = sz_utf8_find_whitespace_sve2;
        }
    }
#endif

#if SZ_USE_V128
    if (caps & sz_cap_v128_k) {
        impl->utf8_count = sz_utf8_count_v128;
        impl->utf8_find_nth = sz_utf8_find_nth_v128;
        impl->utf8_find_newline = sz_utf8_find_newline_v128;
        impl->utf8_find_whitespace = sz_utf8_find_whitespace_v128;
        impl->utf8_unpack_chunk = sz_utf8_unpack_chunk_v128;
        impl->utf8_word_find_boundary = sz_utf8_word_find_boundary_v128;
        impl->utf8_word_rfind_boundary = sz_utf8_word_rfind_boundary_v128;
    }
#endif

#if SZ_USE_V128RELAXED
    if (caps & sz_cap_v128relaxed_k) {
        impl->utf8_count = sz_utf8_count_v128relaxed;
        impl->utf8_find_nth = sz_utf8_find_nth_v128relaxed;
        impl->utf8_find_newline = sz_utf8_find_newline_v128relaxed;
        impl->utf8_find_whitespace = sz_utf8_find_whitespace_v128relaxed;
    }
#endif

#if SZ_USE_RVV
    if (caps & sz_cap_rvv_k) {
        impl->utf8_count = sz_utf8_count_rvv;
        impl->utf8_find_nth = sz_utf8_find_nth_rvv;
        impl->utf8_find_newline = sz_utf8_find_newline_rvv;
        impl->utf8_find_whitespace = sz_utf8_find_whitespace_rvv;
        impl->utf8_unpack_chunk = sz_utf8_unpack_chunk_rvv;
        impl->utf8_word_find_boundary = sz_utf8_word_find_boundary_rvv;
        impl->utf8_word_rfind_boundary = sz_utf8_word_rfind_boundary_rvv;
    }
#endif

#if SZ_USE_LASX
    if (caps & sz_cap_lasx_k) {
        impl->utf8_count = sz_utf8_count_lasx;
        impl->utf8_find_nth = sz_utf8_find_nth_lasx;
        impl->utf8_find_newline = sz_utf8_find_newline_lasx;
        impl->utf8_find_whitespace = sz_utf8_find_whitespace_lasx;
        impl->utf8_unpack_chunk = sz_utf8_unpack_chunk_lasx;
        impl->utf8_word_find_boundary = sz_utf8_word_find_boundary_lasx;
        impl->utf8_word_rfind_boundary = sz_utf8_word_rfind_boundary_lasx;
    }
#endif

#if SZ_USE_POWERVSX
    if (caps & sz_cap_powervsx_k) {
        impl->utf8_count = sz_utf8_count_powervsx;
        impl->utf8_find_nth = sz_utf8_find_nth_powervsx;
        impl->utf8_find_newline = sz_utf8_find_newline_powervsx;
        impl->utf8_find_whitespace = sz_utf8_find_whitespace_powervsx;
        impl->utf8_unpack_chunk = sz_utf8_unpack_chunk_powervsx;
        impl->utf8_word_find_boundary = sz_utf8_word_find_boundary_powervsx;
        impl->utf8_word_rfind_boundary = sz_utf8_word_rfind_boundary_powervsx;
    }
#endif
}

SZ_DYNAMIC sz_size_t sz_utf8_count(sz_cptr_t text, sz_size_t length) {
    return sz_dispatch_table.utf8_count(text, length);
}

SZ_DYNAMIC sz_cptr_t sz_utf8_find_nth(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    return sz_dispatch_table.utf8_find_nth(text, length, n);
}

SZ_DYNAMIC sz_cptr_t sz_utf8_unpack_chunk(sz_cptr_t text, sz_size_t length, sz_rune_t *runes, sz_size_t runes_capacity,
                                          sz_size_t *runes_unpacked) {
    return sz_dispatch_table.utf8_unpack_chunk(text, length, runes, runes_capacity, runes_unpacked);
}

SZ_DYNAMIC sz_cptr_t sz_utf8_find_newline(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    return sz_dispatch_table.utf8_find_newline(text, length, matched_length);
}

SZ_DYNAMIC sz_cptr_t sz_utf8_find_whitespace(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    return sz_dispatch_table.utf8_find_whitespace(text, length, matched_length);
}

SZ_DYNAMIC sz_cptr_t sz_utf8_word_find_boundary(sz_cptr_t text, sz_size_t length, sz_size_t *boundary_width) {
    return sz_dispatch_table.utf8_word_find_boundary(text, length, boundary_width);
}

SZ_DYNAMIC sz_cptr_t sz_utf8_word_rfind_boundary(sz_cptr_t text, sz_size_t length, sz_size_t *boundary_width) {
    return sz_dispatch_table.utf8_word_rfind_boundary(text, length, boundary_width);
}
