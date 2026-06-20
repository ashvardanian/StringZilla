/**
 *  @brief SVE2 backend for UTF-8 codepoint mechanics.
 *  @file include/stringzilla/utf8_codepoints/sve2.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_CODEPOINTS_SVE2_H_
#define STRINGZILLA_UTF8_CODEPOINTS_SVE2_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_codepoints/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_SVE2
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve+sve2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve+sve2")
#endif

SZ_PUBLIC sz_size_t sz_utf8_count_sve2(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const step = svcntb();
    sz_size_t char_count = 0;

    // Count bytes that are NOT continuation bytes: (byte & 0xC0) != 0x80
    for (sz_size_t offset = 0; offset < length; offset += step) {
        svbool_t pg = svwhilelt_b8((sz_u64_t)offset, (sz_u64_t)length);
        svuint8_t text_vec = svld1_u8(pg, text_u8 + offset);
        svbool_t is_start = svcmpne_n_u8(pg, svand_n_u8_x(pg, text_vec, 0xC0), 0x80);
        char_count += svcntp_b8(pg, is_start);
    }
    return char_count;
}

/** @brief  Return a pointer to the start byte of the `n`-th UTF-8 codepoint, or `SZ_NULL_CHAR` if absent.
 *
 *  Single-window O(1) locate mirroring the RVV backend: each window zero-extends `svcntw()` bytes into 32-bit
 *  lanes (SVE2 has no `svcompact_u8`, so the index domain is 32-bit), skips whole windows by lead popcount, then
 *  `svcompact_u32` packs the lead-lane iota and `svlastb` reads the `n`-th packed lane. Byte-exact to serial. */
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_sve2(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const window_bytes = svcntw(); // one byte per 32-bit lane
    svuint32_t const lane_iota = svindex_u32(0, 1);
    while (length) {
        svbool_t const pg = svwhilelt_b32_u64(0, (sz_u64_t)length);
        svuint32_t const bytes_u32 = svld1ub_u32(pg, text_u8);
        // Leading byte iff `(byte & 0xC0) != 0x80`.
        svbool_t const lead = svcmpne_n_u32(pg, svand_n_u32_x(pg, bytes_u32, 0xC0), 0x80);
        sz_size_t const lead_count = svcntp_b32(pg, lead);
        if (n >= lead_count) {
            n -= lead_count;
            text_u8 += window_bytes, length -= sz_min_of_two(window_bytes, length);
            continue;
        }
        svuint32_t const packed = svcompact_u32(lead, lane_iota);
        sz_size_t const lane = svlastb_u32(svwhilelt_b32_u64(0, (sz_u64_t)(n + 1)), packed);
        return (sz_cptr_t)(text_u8 + lane);
    }
    return SZ_NULL_CHAR;
}

/*  Multistep newline / whitespace iteration (SVE2 / AArch64 scalable). Each tile classifies delimiter starts
 *  branchlessly into a per-lane start predicate + byte-length vector; the masked tail reads zero so a truncated
 *  multi-byte delimiter at EOF never matches (no serial tail). Starts are trusted in `[0, tile - 3]` on a full
 *  tile (2-/3-byte views fully loaded) and in every valid lane on the final tile. Output is capacity-cut to
 *  `matches_capacity`. For newlines a CRLF is one 2-byte match: a `text[pos-1] == '\r'` carry suppresses the
 *  straddling LF and a post-loop skip mirrors it; whitespace does no CRLF merging. */

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE2

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_CODEPOINTS_SVE2_H_
