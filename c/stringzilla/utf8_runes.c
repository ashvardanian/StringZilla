/**
 *  @file c/stringzilla/utf8_runes.c
 *  @brief Per-domain dispatch shim for UTF-8 codepoint mechanics: count, find-nth, and rune unpacking.
 *  @author Ash Vardanian
 */
#include "dispatch.h"
#include <stringzilla/utf8_runes.h>

SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_runes_update_(sz_capability_t caps) {
    sz_implementations_t *impl = &sz_dispatch_table;
    sz_unused_(caps);

    impl->utf8_count = sz_utf8_count_serial;
    impl->utf8_seek = sz_utf8_seek_serial;
    impl->utf8_decode = sz_utf8_decode_serial;

#if SZ_USE_HASWELL
    if (caps & sz_cap_haswell_k) {
        impl->utf8_count = sz_utf8_count_haswell;
        impl->utf8_seek = sz_utf8_seek_haswell;
        impl->utf8_decode = sz_utf8_decode_haswell;
    }
#endif

#if SZ_USE_ICELAKE
    if (caps & sz_cap_icelake_k) {
        impl->utf8_count = sz_utf8_count_icelake;
        impl->utf8_seek = sz_utf8_seek_icelake;
        impl->utf8_decode = sz_utf8_decode_icelake;
    }
#endif

#if SZ_USE_NEON
    if (caps & sz_cap_neon_k) {
        impl->utf8_count = sz_utf8_count_neon;
        impl->utf8_seek = sz_utf8_seek_neon;
        impl->utf8_decode = sz_utf8_decode_neon;
    }
#endif

#if SZ_USE_SVE2
    if (caps & sz_cap_sve2_k) {
        impl->utf8_count = sz_utf8_count_sve2;
        impl->utf8_seek = sz_utf8_seek_sve2;
        impl->utf8_decode = sz_utf8_decode_sve2;
    }
#endif

#if SZ_USE_V128
    if (caps & sz_cap_v128_k) {
        impl->utf8_count = sz_utf8_count_v128;
        impl->utf8_seek = sz_utf8_seek_v128;
        impl->utf8_decode = sz_utf8_decode_v128;
    }
#endif

#if SZ_USE_V128RELAXED
    if (caps & sz_cap_v128relaxed_k) {
        impl->utf8_count = sz_utf8_count_v128relaxed;
        impl->utf8_seek = sz_utf8_seek_v128relaxed;
    }
#endif

#if SZ_USE_RVV
    if (caps & sz_cap_rvv_k) {
        impl->utf8_count = sz_utf8_count_rvv;
        impl->utf8_seek = sz_utf8_seek_rvv;
        impl->utf8_decode = sz_utf8_decode_rvv;
    }
#endif

#if SZ_USE_LASX
    if (caps & sz_cap_lasx_k) {
        impl->utf8_count = sz_utf8_count_lasx;
        impl->utf8_seek = sz_utf8_seek_lasx;
        impl->utf8_decode = sz_utf8_decode_lasx;
    }
#endif

#if SZ_USE_POWERVSX
    if (caps & sz_cap_powervsx_k) {
        impl->utf8_count = sz_utf8_count_powervsx;
        impl->utf8_seek = sz_utf8_seek_powervsx;
        impl->utf8_decode = sz_utf8_decode_powervsx;
    }
#endif
}

SZ_DYNAMIC sz_size_t sz_utf8_count(sz_cptr_t text, sz_size_t length) {
    return sz_dispatch_table.utf8_count(text, length);
}

SZ_DYNAMIC sz_cptr_t sz_utf8_seek(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    return sz_dispatch_table.utf8_seek(text, length, n);
}

SZ_DYNAMIC sz_cptr_t sz_utf8_decode(sz_cptr_t text, sz_size_t length, sz_rune_t *runes, sz_size_t runes_capacity,
                                    sz_size_t *runes_unpacked) {
    return sz_dispatch_table.utf8_decode(text, length, runes, runes_capacity, runes_unpacked);
}
