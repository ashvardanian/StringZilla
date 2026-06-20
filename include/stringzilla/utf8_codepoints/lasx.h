/**
 *  @brief LoongArch LASX backend for UTF-8 codepoint mechanics.
 *  @file include/stringzilla/utf8_codepoints/lasx.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_CODEPOINTS_LASX_H_
#define STRINGZILLA_UTF8_CODEPOINTS_LASX_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_codepoints/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_LASX
/** @brief  UTF-8 to UTF-32 transcoding for one chunk; forwards to serial.
 *  The data-dependent codepoint stride, capacity bound, and incomplete-trailing rule have no clean
 *  value-identical 256-bit LASX form (same reason the NEON backend delegates). */
SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_lasx(  //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {
    return sz_utf8_unpack_chunk_serial(text, length, runes, runes_capacity, runes_unpacked);
}

/** @brief  Recombine the two per-128-bit-lane 16-bit `__lasx_xvmskltz_b` sign-bit masks into one 32-bit mask,
 *          matching AVX2's `_mm256_movemask_epi8` (word 0 = low lane, word 4 = high lane). */
SZ_INTERNAL sz_u32_t sz_xvmovemask_b_utf8_lasx_(__m256i sign_extended) {
    __m256i collected = __lasx_xvmskltz_b(sign_extended);
    sz_u32_t low = (sz_u32_t)__lasx_xvpickve2gr_wu(collected, 0);
    sz_u32_t high = (sz_u32_t)__lasx_xvpickve2gr_wu(collected, 4);
    return (low & 0xFFFFu) | ((high & 0xFFFFu) << 16);
}

#pragma region Multistep Newline & Whitespace Iteration

/*  Multistep newline / whitespace iteration (LoongArch LASX).
 *
 *  Each 32-byte tile is classified branchlessly into a `start_bits` mask of every delimiter start plus disjoint
 *  2-byte / 3-byte start masks (built from `sz_xvmovemask_b_utf8_lasx_` and bitmask shifts). The peel left-packs
 *  with the same dword-index table the AVX2 backend uses, in eight 4-lane sub-blocks. Starts are trusted in lanes
 *  [0,29] and we step 30; a `t[pos-1] == '\r'` carry suppresses an LF completing a tile-straddling CRLF. */

/** @brief  Store the low `group` (1..4) of four 64-bit lanes of `values_u64x4` to `destination`. */
SZ_INTERNAL void sz_utf8_iterate_store_group_lasx_(__m256i values_u64x4, sz_size_t group, sz_size_t *destination) {
    if (group == 4) { __lasx_xvst(values_u64x4, destination, 0); }
    else {
        __lasx_xvstelm_d(values_u64x4, destination, 0, 0);
        if (group >= 2) __lasx_xvstelm_d(values_u64x4, destination, 8, 1);
        if (group >= 3) __lasx_xvstelm_d(values_u64x4, destination, 16, 2);
    }
}

SZ_PUBLIC sz_size_t sz_utf8_count_lasx(sz_cptr_t text, sz_size_t length) {
    __m256i continuation_mask_vec = __lasx_xvreplgr2vr_b((char)0xC0);
    __m256i continuation_pattern_vec = __lasx_xvreplgr2vr_b((char)0x80);
    __m256i const one_vec = __lasx_xvreplgr2vr_b(1);
    __m256i const zero_vec = __lasx_xvreplgr2vr_b(0);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t char_count = 0;

    // Keep a per-lane vector accumulator across the loop and reduce ONCE at the end, instead of doing a
    // movemask + scalar `popcount` per block. A start byte is any byte where `(b & 0xC0) != 0x80`; we map
    // that to a 0/1 byte and fold adjacent pairs into a 16x u16 accumulator via `xvhaddw_hu_bu`. Each u16
    // lane grows by at most 2 per block, so 65535/2 = 32767 blocks fit before overflow; we flush the u16
    // lanes into a 4x u64 accumulator every 32000 blocks to stay safe (rare for realistic inputs).
    __m256i sums_d = zero_vec;
    __m256i sums_h = zero_vec;
    sz_size_t blocks_since_flush = 0;
    while (length >= 32) {
        __m256i text_vec = __lasx_xvld(text_u8, 0);
        // Extract top 2 bits of each byte; continuation bytes equal 0x80. `is_start` is 0xFF for starts.
        __m256i headers = __lasx_xvand_v(text_vec, continuation_mask_vec);
        __m256i is_cont = __lasx_xvseq_b(headers, continuation_pattern_vec); // 0xFF where continuation
        // `andn` style: 1 where start, 0 where continuation. (~is_cont) & 1.
        __m256i is_start = __lasx_xvandn_v(is_cont, one_vec); // (~is_cont) & one
        sums_h = __lasx_xvadd_h(sums_h, __lasx_xvhaddw_hu_bu(is_start, is_start));
        if (++blocks_since_flush == 32000) {
            __m256i widened_w = __lasx_xvhaddw_wu_hu(sums_h, sums_h);
            sums_d = __lasx_xvadd_d(sums_d, __lasx_xvhaddw_du_wu(widened_w, widened_w));
            sums_h = zero_vec;
            blocks_since_flush = 0;
        }
        text_u8 += 32, length -= 32;
    }
    {
        __m256i widened_w = __lasx_xvhaddw_wu_hu(sums_h, sums_h);
        sums_d = __lasx_xvadd_d(sums_d, __lasx_xvhaddw_du_wu(widened_w, widened_w));
    }
    char_count += (sz_size_t)__lasx_xvpickve2gr_du(sums_d, 0) + (sz_size_t)__lasx_xvpickve2gr_du(sums_d, 1) +
                  (sz_size_t)__lasx_xvpickve2gr_du(sums_d, 2) + (sz_size_t)__lasx_xvpickve2gr_du(sums_d, 3);

    char_count += sz_utf8_count_serial((sz_cptr_t)text_u8, length);
    return char_count;
}

/** @brief  Same block logic as `sz_utf8_count_lasx`, but locating the Nth start byte. LASX has no `PDEP`, so the
 *          start-byte bitmask is fed to `sz_u32_nth_set_bit` to land on the Nth set bit. */
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_lasx(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    __m256i continuation_mask_vec = __lasx_xvreplgr2vr_b((char)0xC0);
    __m256i continuation_pattern_vec = __lasx_xvreplgr2vr_b((char)0x80);
    __m256i const one_vec = __lasx_xvreplgr2vr_b(1);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;

    sz_u256_vec_t text_vec, headers_vec;
    while (length >= 32) {
        text_vec.lasx = __lasx_xvld(text_u8, 0);
        headers_vec.lasx = __lasx_xvand_v(text_vec.lasx, continuation_mask_vec);
        __m256i is_continuation = __lasx_xvseq_b(headers_vec.lasx, continuation_pattern_vec);
        sz_u32_t start_byte_mask = ~sz_xvmovemask_b_utf8_lasx_(is_continuation);

        // Count start bytes in-vector: each start lane carries a 0x01, so `xvpcnt_d` sums one bit per such lane.
        __m256i is_start = __lasx_xvandn_v(is_continuation, one_vec);
        __m256i start_byte_counts = __lasx_xvpcnt_d(is_start);
        sz_size_t start_byte_count =
            (sz_size_t)(__lasx_xvpickve2gr_du(start_byte_counts, 0) + __lasx_xvpickve2gr_du(start_byte_counts, 1) +
                        __lasx_xvpickve2gr_du(start_byte_counts, 2) + __lasx_xvpickve2gr_du(start_byte_counts, 3));

        if (n >= start_byte_count) {
            n -= start_byte_count, text_u8 += 32, length -= 32;
            continue;
        }
        return (sz_cptr_t)(text_u8 + sz_u32_nth_set_bit(start_byte_mask, n));
    }

    return sz_utf8_find_nth_serial((sz_cptr_t)text_u8, length, n);
}

/*  UAX-29 word boundary detection (forward & reverse).
 *
 *  The stateful look-around sub-rules stay in the serial reference; LASX accelerates the common case of long
 *  ASCII-letter or ASCII-digit runs whose interior positions can never be boundaries (WB5 / WB8). Skipping only
 *  proven non-boundaries keeps the returned pointer and `boundary_width` byte-for-byte identical to `_serial`. */
#endif // SZ_USE_LASX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_CODEPOINTS_LASX_H_
