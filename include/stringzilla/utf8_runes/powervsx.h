/**
 *  @brief POWER VSX backend for UTF-8 codepoint mechanics.
 *  @file include/stringzilla/utf8_runes/powervsx.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_RUNES_POWERVSX_H_
#define STRINGZILLA_UTF8_RUNES_POWERVSX_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

SZ_API_COMPTIME sz_size_t sz_utf8_count_powervsx(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t char_count = 0;

    __vector unsigned char const header_mask_u8x16 = vec_splats((unsigned char)0xC0);
    __vector unsigned char const continuation_pattern_u8x16 = vec_splats((unsigned char)0x80);
    __vector unsigned char const one_byte_u8x16 = vec_splats((unsigned char)1);

    // `vec_msum(starts, ones, acc)` is the in-lane accumulator: each `starts` byte is 0 or 1, so each
    // step adds at most 4 to any 32-bit lane. A lane can absorb `UINT_MAX / 4 ≈ 1.07B` iterations
    // before overflowing. We process the input in blocks of that many 16-byte windows so the only
    // horizontal reduction happens once per block — out of the hot inner loop — and the inner loop
    // carries no per-iteration counter or branch. Results stay identical to `sz_utf8_count_serial`.
    sz_size_t const block_windows = (sz_size_t)1000000000; // 1B * 4 < UINT_MAX.
    while (length >= 16) {
        sz_size_t windows = length / 16;
        if (windows > block_windows) windows = block_windows;
        __vector unsigned int start_count_u32x4 = vec_splats((unsigned int)0);
        for (sz_size_t window_index = 0; window_index < windows; ++window_index, text_u8 += 16) {
            __vector unsigned char text_window_u8x16 = vec_xl(0, text_u8);
            __vector unsigned char header_result_u8x16 = vec_and(text_window_u8x16, header_mask_u8x16);
            // Continuation bytes → 0xFF, starts → 0x00; invert so each start contributes 0x01.
            __vector unsigned char continuation_mask_u8x16 = (__vector unsigned char)vec_cmpeq(
                header_result_u8x16, continuation_pattern_u8x16);
            __vector unsigned char start_mask_u8x16 = vec_andc(one_byte_u8x16, continuation_mask_u8x16);
            start_count_u32x4 = vec_msum(start_mask_u8x16, one_byte_u8x16,
                                         start_count_u32x4); // Sum the start flags into the 32-bit lanes.
        }
        char_count += (sz_size_t)start_count_u32x4[0] + start_count_u32x4[1] + start_count_u32x4[2] +
                      start_count_u32x4[3];
        length -= windows * 16;
    }

    if (length) char_count += sz_utf8_count_serial((sz_cptr_t)text_u8, length);
    return char_count;
}

/*  x86-`movemask`-equivalent for VSX: gathers the MSB of each of the 16 bytes into bit `i` (lowest-addressed
 *  byte → bit 0) via `vec_vbpermq`, identical to `sz_utf8_movemask_powervsx_` below but reachable before its
 *  first user (`sz_utf8_seek_powervsx` and the multistep iterators); distinctly named so both coexist in
 *  one translation unit. */
SZ_HELPER_INLINE sz_u32_t sz_utf8_iterate_movemask_powervsx_(__vector unsigned char compared) {
    __vector unsigned char const gather_indices_u8x16 = {120, 112, 104, 96, 88, 80, 72, 64,
                                                         56,  48,  40,  32, 24, 16, 8,  0};
    __vector unsigned long long const gathered_mask_u64x2 = vec_vbpermq(compared, gather_indices_u8x16);
#if SZ_IS_BIG_ENDIAN_
    return (sz_u32_t)(gathered_mask_u64x2[0] & 0xFFFFull);
#else
    return (sz_u32_t)(gathered_mask_u64x2[1] & 0xFFFFull);
#endif
}

/**
 *  @brief Locate the start byte of the n-th code-point (0-indexed), mirroring `sz_utf8_seek_serial`.
 *         Per tile a code-point-start bitmask `~continuation_mask` is popcounted to skip whole tiles; the
 *         n-th start's lane comes from `sz_u32_nth_set_bit`. The `< 16` tail defers to the serial reference.
 */
SZ_API_COMPTIME sz_cptr_t sz_utf8_seek_powervsx(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;

    __vector unsigned char const header_mask_u8x16 = vec_splats((unsigned char)0xC0);
    __vector unsigned char const continuation_pattern_u8x16 = vec_splats((unsigned char)0x80);

    while (length >= 16) {
        __vector unsigned char text_window_u8x16 = vec_xl(0, text_u8);
        __vector unsigned char header_result_u8x16 = vec_and(text_window_u8x16, header_mask_u8x16);
        __vector unsigned char continuation_mask_u8x16 = (__vector unsigned char)vec_cmpeq(header_result_u8x16,
                                                                                           continuation_pattern_u8x16);
        sz_u32_t start_bits = (~sz_utf8_iterate_movemask_powervsx_(continuation_mask_u8x16)) & 0xFFFFu;
        sz_size_t const start_count = (sz_size_t)sz_u32_popcount(start_bits);

        if (n >= start_count) {
            n -= start_count, text_u8 += 16, length -= 16;
            continue;
        }
        return (sz_cptr_t)(text_u8 + sz_u32_nth_set_bit(start_bits, n));
    }

    return sz_utf8_seek_serial((sz_cptr_t)text_u8, length, n);
}

#if !SZ_IS_BIG_ENDIAN_

#pragma region Shared SIMD leaf substrate

/*  TOTAL in-vector UTF-8 decode for the little-endian POWER VSX backend, the algorithmic twin of
 *  @ref sz_utf8_decode_once_icelake_. Only the SIMD primitives differ: `__mmask64` lane masks become
 *  `sz_u64_t` masks assembled from per-register `vec_vbpermq` movemasks, `vpcompressb` becomes a shuffle-LUT
 *  left-pack via `vec_perm`, and `vpermb` gathers become `vec_perm` over the 32-byte register pairs. The gate
 *  algebra (well-formed, maximal-subpart length, orphan promotion, U+FFFD substitution) is byte-for-byte the
 *  Ice Lake / serial contract: one U+FFFD per Unicode 17.0 §3.9 maximal ill-formed subpart. */

/** @brief  Left-pack shuffle table: row `mask` holds the ascending byte indices (in [0,8)) of the set bits of an
 *          8-bit mask, padded with 0xFF. The VSX `vec_perm`-based stand-in for `vpcompressb` / BMI2 `pext`; two rows
 *          (low byte, high byte of a 16-bit lane mask) left-pack one 16-lane register's start offsets in two permutes
 *          stitched by `popcount(low8)`. 2 KB of `.rodata`, read straight (no per-call materialization). */
static sz_u8_t const sz_utf8_compress8_powervsx_[256][8] = {{255, 255, 255, 255, 255, 255, 255, 255},
                                                            {0, 255, 255, 255, 255, 255, 255, 255},
                                                            {1, 255, 255, 255, 255, 255, 255, 255},
                                                            {0, 1, 255, 255, 255, 255, 255, 255},
                                                            {2, 255, 255, 255, 255, 255, 255, 255},
                                                            {0, 2, 255, 255, 255, 255, 255, 255},
                                                            {1, 2, 255, 255, 255, 255, 255, 255},
                                                            {0, 1, 2, 255, 255, 255, 255, 255},
                                                            {3, 255, 255, 255, 255, 255, 255, 255},
                                                            {0, 3, 255, 255, 255, 255, 255, 255},
                                                            {1, 3, 255, 255, 255, 255, 255, 255},
                                                            {0, 1, 3, 255, 255, 255, 255, 255},
                                                            {2, 3, 255, 255, 255, 255, 255, 255},
                                                            {0, 2, 3, 255, 255, 255, 255, 255},
                                                            {1, 2, 3, 255, 255, 255, 255, 255},
                                                            {0, 1, 2, 3, 255, 255, 255, 255},
                                                            {4, 255, 255, 255, 255, 255, 255, 255},
                                                            {0, 4, 255, 255, 255, 255, 255, 255},
                                                            {1, 4, 255, 255, 255, 255, 255, 255},
                                                            {0, 1, 4, 255, 255, 255, 255, 255},
                                                            {2, 4, 255, 255, 255, 255, 255, 255},
                                                            {0, 2, 4, 255, 255, 255, 255, 255},
                                                            {1, 2, 4, 255, 255, 255, 255, 255},
                                                            {0, 1, 2, 4, 255, 255, 255, 255},
                                                            {3, 4, 255, 255, 255, 255, 255, 255},
                                                            {0, 3, 4, 255, 255, 255, 255, 255},
                                                            {1, 3, 4, 255, 255, 255, 255, 255},
                                                            {0, 1, 3, 4, 255, 255, 255, 255},
                                                            {2, 3, 4, 255, 255, 255, 255, 255},
                                                            {0, 2, 3, 4, 255, 255, 255, 255},
                                                            {1, 2, 3, 4, 255, 255, 255, 255},
                                                            {0, 1, 2, 3, 4, 255, 255, 255},
                                                            {5, 255, 255, 255, 255, 255, 255, 255},
                                                            {0, 5, 255, 255, 255, 255, 255, 255},
                                                            {1, 5, 255, 255, 255, 255, 255, 255},
                                                            {0, 1, 5, 255, 255, 255, 255, 255},
                                                            {2, 5, 255, 255, 255, 255, 255, 255},
                                                            {0, 2, 5, 255, 255, 255, 255, 255},
                                                            {1, 2, 5, 255, 255, 255, 255, 255},
                                                            {0, 1, 2, 5, 255, 255, 255, 255},
                                                            {3, 5, 255, 255, 255, 255, 255, 255},
                                                            {0, 3, 5, 255, 255, 255, 255, 255},
                                                            {1, 3, 5, 255, 255, 255, 255, 255},
                                                            {0, 1, 3, 5, 255, 255, 255, 255},
                                                            {2, 3, 5, 255, 255, 255, 255, 255},
                                                            {0, 2, 3, 5, 255, 255, 255, 255},
                                                            {1, 2, 3, 5, 255, 255, 255, 255},
                                                            {0, 1, 2, 3, 5, 255, 255, 255},
                                                            {4, 5, 255, 255, 255, 255, 255, 255},
                                                            {0, 4, 5, 255, 255, 255, 255, 255},
                                                            {1, 4, 5, 255, 255, 255, 255, 255},
                                                            {0, 1, 4, 5, 255, 255, 255, 255},
                                                            {2, 4, 5, 255, 255, 255, 255, 255},
                                                            {0, 2, 4, 5, 255, 255, 255, 255},
                                                            {1, 2, 4, 5, 255, 255, 255, 255},
                                                            {0, 1, 2, 4, 5, 255, 255, 255},
                                                            {3, 4, 5, 255, 255, 255, 255, 255},
                                                            {0, 3, 4, 5, 255, 255, 255, 255},
                                                            {1, 3, 4, 5, 255, 255, 255, 255},
                                                            {0, 1, 3, 4, 5, 255, 255, 255},
                                                            {2, 3, 4, 5, 255, 255, 255, 255},
                                                            {0, 2, 3, 4, 5, 255, 255, 255},
                                                            {1, 2, 3, 4, 5, 255, 255, 255},
                                                            {0, 1, 2, 3, 4, 5, 255, 255},
                                                            {6, 255, 255, 255, 255, 255, 255, 255},
                                                            {0, 6, 255, 255, 255, 255, 255, 255},
                                                            {1, 6, 255, 255, 255, 255, 255, 255},
                                                            {0, 1, 6, 255, 255, 255, 255, 255},
                                                            {2, 6, 255, 255, 255, 255, 255, 255},
                                                            {0, 2, 6, 255, 255, 255, 255, 255},
                                                            {1, 2, 6, 255, 255, 255, 255, 255},
                                                            {0, 1, 2, 6, 255, 255, 255, 255},
                                                            {3, 6, 255, 255, 255, 255, 255, 255},
                                                            {0, 3, 6, 255, 255, 255, 255, 255},
                                                            {1, 3, 6, 255, 255, 255, 255, 255},
                                                            {0, 1, 3, 6, 255, 255, 255, 255},
                                                            {2, 3, 6, 255, 255, 255, 255, 255},
                                                            {0, 2, 3, 6, 255, 255, 255, 255},
                                                            {1, 2, 3, 6, 255, 255, 255, 255},
                                                            {0, 1, 2, 3, 6, 255, 255, 255},
                                                            {4, 6, 255, 255, 255, 255, 255, 255},
                                                            {0, 4, 6, 255, 255, 255, 255, 255},
                                                            {1, 4, 6, 255, 255, 255, 255, 255},
                                                            {0, 1, 4, 6, 255, 255, 255, 255},
                                                            {2, 4, 6, 255, 255, 255, 255, 255},
                                                            {0, 2, 4, 6, 255, 255, 255, 255},
                                                            {1, 2, 4, 6, 255, 255, 255, 255},
                                                            {0, 1, 2, 4, 6, 255, 255, 255},
                                                            {3, 4, 6, 255, 255, 255, 255, 255},
                                                            {0, 3, 4, 6, 255, 255, 255, 255},
                                                            {1, 3, 4, 6, 255, 255, 255, 255},
                                                            {0, 1, 3, 4, 6, 255, 255, 255},
                                                            {2, 3, 4, 6, 255, 255, 255, 255},
                                                            {0, 2, 3, 4, 6, 255, 255, 255},
                                                            {1, 2, 3, 4, 6, 255, 255, 255},
                                                            {0, 1, 2, 3, 4, 6, 255, 255},
                                                            {5, 6, 255, 255, 255, 255, 255, 255},
                                                            {0, 5, 6, 255, 255, 255, 255, 255},
                                                            {1, 5, 6, 255, 255, 255, 255, 255},
                                                            {0, 1, 5, 6, 255, 255, 255, 255},
                                                            {2, 5, 6, 255, 255, 255, 255, 255},
                                                            {0, 2, 5, 6, 255, 255, 255, 255},
                                                            {1, 2, 5, 6, 255, 255, 255, 255},
                                                            {0, 1, 2, 5, 6, 255, 255, 255},
                                                            {3, 5, 6, 255, 255, 255, 255, 255},
                                                            {0, 3, 5, 6, 255, 255, 255, 255},
                                                            {1, 3, 5, 6, 255, 255, 255, 255},
                                                            {0, 1, 3, 5, 6, 255, 255, 255},
                                                            {2, 3, 5, 6, 255, 255, 255, 255},
                                                            {0, 2, 3, 5, 6, 255, 255, 255},
                                                            {1, 2, 3, 5, 6, 255, 255, 255},
                                                            {0, 1, 2, 3, 5, 6, 255, 255},
                                                            {4, 5, 6, 255, 255, 255, 255, 255},
                                                            {0, 4, 5, 6, 255, 255, 255, 255},
                                                            {1, 4, 5, 6, 255, 255, 255, 255},
                                                            {0, 1, 4, 5, 6, 255, 255, 255},
                                                            {2, 4, 5, 6, 255, 255, 255, 255},
                                                            {0, 2, 4, 5, 6, 255, 255, 255},
                                                            {1, 2, 4, 5, 6, 255, 255, 255},
                                                            {0, 1, 2, 4, 5, 6, 255, 255},
                                                            {3, 4, 5, 6, 255, 255, 255, 255},
                                                            {0, 3, 4, 5, 6, 255, 255, 255},
                                                            {1, 3, 4, 5, 6, 255, 255, 255},
                                                            {0, 1, 3, 4, 5, 6, 255, 255},
                                                            {2, 3, 4, 5, 6, 255, 255, 255},
                                                            {0, 2, 3, 4, 5, 6, 255, 255},
                                                            {1, 2, 3, 4, 5, 6, 255, 255},
                                                            {0, 1, 2, 3, 4, 5, 6, 255},
                                                            {7, 255, 255, 255, 255, 255, 255, 255},
                                                            {0, 7, 255, 255, 255, 255, 255, 255},
                                                            {1, 7, 255, 255, 255, 255, 255, 255},
                                                            {0, 1, 7, 255, 255, 255, 255, 255},
                                                            {2, 7, 255, 255, 255, 255, 255, 255},
                                                            {0, 2, 7, 255, 255, 255, 255, 255},
                                                            {1, 2, 7, 255, 255, 255, 255, 255},
                                                            {0, 1, 2, 7, 255, 255, 255, 255},
                                                            {3, 7, 255, 255, 255, 255, 255, 255},
                                                            {0, 3, 7, 255, 255, 255, 255, 255},
                                                            {1, 3, 7, 255, 255, 255, 255, 255},
                                                            {0, 1, 3, 7, 255, 255, 255, 255},
                                                            {2, 3, 7, 255, 255, 255, 255, 255},
                                                            {0, 2, 3, 7, 255, 255, 255, 255},
                                                            {1, 2, 3, 7, 255, 255, 255, 255},
                                                            {0, 1, 2, 3, 7, 255, 255, 255},
                                                            {4, 7, 255, 255, 255, 255, 255, 255},
                                                            {0, 4, 7, 255, 255, 255, 255, 255},
                                                            {1, 4, 7, 255, 255, 255, 255, 255},
                                                            {0, 1, 4, 7, 255, 255, 255, 255},
                                                            {2, 4, 7, 255, 255, 255, 255, 255},
                                                            {0, 2, 4, 7, 255, 255, 255, 255},
                                                            {1, 2, 4, 7, 255, 255, 255, 255},
                                                            {0, 1, 2, 4, 7, 255, 255, 255},
                                                            {3, 4, 7, 255, 255, 255, 255, 255},
                                                            {0, 3, 4, 7, 255, 255, 255, 255},
                                                            {1, 3, 4, 7, 255, 255, 255, 255},
                                                            {0, 1, 3, 4, 7, 255, 255, 255},
                                                            {2, 3, 4, 7, 255, 255, 255, 255},
                                                            {0, 2, 3, 4, 7, 255, 255, 255},
                                                            {1, 2, 3, 4, 7, 255, 255, 255},
                                                            {0, 1, 2, 3, 4, 7, 255, 255},
                                                            {5, 7, 255, 255, 255, 255, 255, 255},
                                                            {0, 5, 7, 255, 255, 255, 255, 255},
                                                            {1, 5, 7, 255, 255, 255, 255, 255},
                                                            {0, 1, 5, 7, 255, 255, 255, 255},
                                                            {2, 5, 7, 255, 255, 255, 255, 255},
                                                            {0, 2, 5, 7, 255, 255, 255, 255},
                                                            {1, 2, 5, 7, 255, 255, 255, 255},
                                                            {0, 1, 2, 5, 7, 255, 255, 255},
                                                            {3, 5, 7, 255, 255, 255, 255, 255},
                                                            {0, 3, 5, 7, 255, 255, 255, 255},
                                                            {1, 3, 5, 7, 255, 255, 255, 255},
                                                            {0, 1, 3, 5, 7, 255, 255, 255},
                                                            {2, 3, 5, 7, 255, 255, 255, 255},
                                                            {0, 2, 3, 5, 7, 255, 255, 255},
                                                            {1, 2, 3, 5, 7, 255, 255, 255},
                                                            {0, 1, 2, 3, 5, 7, 255, 255},
                                                            {4, 5, 7, 255, 255, 255, 255, 255},
                                                            {0, 4, 5, 7, 255, 255, 255, 255},
                                                            {1, 4, 5, 7, 255, 255, 255, 255},
                                                            {0, 1, 4, 5, 7, 255, 255, 255},
                                                            {2, 4, 5, 7, 255, 255, 255, 255},
                                                            {0, 2, 4, 5, 7, 255, 255, 255},
                                                            {1, 2, 4, 5, 7, 255, 255, 255},
                                                            {0, 1, 2, 4, 5, 7, 255, 255},
                                                            {3, 4, 5, 7, 255, 255, 255, 255},
                                                            {0, 3, 4, 5, 7, 255, 255, 255},
                                                            {1, 3, 4, 5, 7, 255, 255, 255},
                                                            {0, 1, 3, 4, 5, 7, 255, 255},
                                                            {2, 3, 4, 5, 7, 255, 255, 255},
                                                            {0, 2, 3, 4, 5, 7, 255, 255},
                                                            {1, 2, 3, 4, 5, 7, 255, 255},
                                                            {0, 1, 2, 3, 4, 5, 7, 255},
                                                            {6, 7, 255, 255, 255, 255, 255, 255},
                                                            {0, 6, 7, 255, 255, 255, 255, 255},
                                                            {1, 6, 7, 255, 255, 255, 255, 255},
                                                            {0, 1, 6, 7, 255, 255, 255, 255},
                                                            {2, 6, 7, 255, 255, 255, 255, 255},
                                                            {0, 2, 6, 7, 255, 255, 255, 255},
                                                            {1, 2, 6, 7, 255, 255, 255, 255},
                                                            {0, 1, 2, 6, 7, 255, 255, 255},
                                                            {3, 6, 7, 255, 255, 255, 255, 255},
                                                            {0, 3, 6, 7, 255, 255, 255, 255},
                                                            {1, 3, 6, 7, 255, 255, 255, 255},
                                                            {0, 1, 3, 6, 7, 255, 255, 255},
                                                            {2, 3, 6, 7, 255, 255, 255, 255},
                                                            {0, 2, 3, 6, 7, 255, 255, 255},
                                                            {1, 2, 3, 6, 7, 255, 255, 255},
                                                            {0, 1, 2, 3, 6, 7, 255, 255},
                                                            {4, 6, 7, 255, 255, 255, 255, 255},
                                                            {0, 4, 6, 7, 255, 255, 255, 255},
                                                            {1, 4, 6, 7, 255, 255, 255, 255},
                                                            {0, 1, 4, 6, 7, 255, 255, 255},
                                                            {2, 4, 6, 7, 255, 255, 255, 255},
                                                            {0, 2, 4, 6, 7, 255, 255, 255},
                                                            {1, 2, 4, 6, 7, 255, 255, 255},
                                                            {0, 1, 2, 4, 6, 7, 255, 255},
                                                            {3, 4, 6, 7, 255, 255, 255, 255},
                                                            {0, 3, 4, 6, 7, 255, 255, 255},
                                                            {1, 3, 4, 6, 7, 255, 255, 255},
                                                            {0, 1, 3, 4, 6, 7, 255, 255},
                                                            {2, 3, 4, 6, 7, 255, 255, 255},
                                                            {0, 2, 3, 4, 6, 7, 255, 255},
                                                            {1, 2, 3, 4, 6, 7, 255, 255},
                                                            {0, 1, 2, 3, 4, 6, 7, 255},
                                                            {5, 6, 7, 255, 255, 255, 255, 255},
                                                            {0, 5, 6, 7, 255, 255, 255, 255},
                                                            {1, 5, 6, 7, 255, 255, 255, 255},
                                                            {0, 1, 5, 6, 7, 255, 255, 255},
                                                            {2, 5, 6, 7, 255, 255, 255, 255},
                                                            {0, 2, 5, 6, 7, 255, 255, 255},
                                                            {1, 2, 5, 6, 7, 255, 255, 255},
                                                            {0, 1, 2, 5, 6, 7, 255, 255},
                                                            {3, 5, 6, 7, 255, 255, 255, 255},
                                                            {0, 3, 5, 6, 7, 255, 255, 255},
                                                            {1, 3, 5, 6, 7, 255, 255, 255},
                                                            {0, 1, 3, 5, 6, 7, 255, 255},
                                                            {2, 3, 5, 6, 7, 255, 255, 255},
                                                            {0, 2, 3, 5, 6, 7, 255, 255},
                                                            {1, 2, 3, 5, 6, 7, 255, 255},
                                                            {0, 1, 2, 3, 5, 6, 7, 255},
                                                            {4, 5, 6, 7, 255, 255, 255, 255},
                                                            {0, 4, 5, 6, 7, 255, 255, 255},
                                                            {1, 4, 5, 6, 7, 255, 255, 255},
                                                            {0, 1, 4, 5, 6, 7, 255, 255},
                                                            {2, 4, 5, 6, 7, 255, 255, 255},
                                                            {0, 2, 4, 5, 6, 7, 255, 255},
                                                            {1, 2, 4, 5, 6, 7, 255, 255},
                                                            {0, 1, 2, 4, 5, 6, 7, 255},
                                                            {3, 4, 5, 6, 7, 255, 255, 255},
                                                            {0, 3, 4, 5, 6, 7, 255, 255},
                                                            {1, 3, 4, 5, 6, 7, 255, 255},
                                                            {0, 1, 3, 4, 5, 6, 7, 255},
                                                            {2, 3, 4, 5, 6, 7, 255, 255},
                                                            {0, 2, 3, 4, 5, 6, 7, 255},
                                                            {1, 2, 3, 4, 5, 6, 7, 255},
                                                            {0, 1, 2, 3, 4, 5, 6, 7}};

/** @brief  Reduce the 16 byte-lane booleans (0x00/0xFF) of @p compared into a 16-bit lane mask (bit `i` <=> lane `i`,
 *          lowest-addressed byte → bit 0), the VSX `vpmovmskb` via `vec_vbpermq`. Mirror of
 *          @ref sz_utf8_iterate_movemask_powervsx_ but kept local to the decode path for clarity at the call sites. */
SZ_HELPER_INLINE sz_u32_t sz_utf8_movemask16_powervsx_(__vector unsigned char compared) {
    __vector unsigned char const gather_indices_u8x16 = {120, 112, 104, 96, 88, 80, 72, 64,
                                                         56,  48,  40,  32, 24, 16, 8,  0};
    __vector unsigned long long const gathered_mask_u64x2 = vec_vbpermq(compared, gather_indices_u8x16);
    return (sz_u32_t)(gathered_mask_u64x2[1] & 0xFFFFull);
}

/** @brief  Combine the four per-register VSX movemasks of @p booleans into one 64-bit lane mask: register `r` → bits
 *          [16*r, 16*r+16). The VSX twin of the NEON `mask_combine` and the Ice Lake native `__mmask64`. */
SZ_HELPER_INLINE sz_u64_t sz_utf8_mask_combine_powervsx_( //
    __vector unsigned char const *booleans) {
    sz_u64_t mask = (sz_u64_t)sz_utf8_movemask16_powervsx_(booleans[0]);
    mask |= (sz_u64_t)sz_utf8_movemask16_powervsx_(booleans[1]) << 16;
    mask |= (sz_u64_t)sz_utf8_movemask16_powervsx_(booleans[2]) << 32;
    mask |= (sz_u64_t)sz_utf8_movemask16_powervsx_(booleans[3]) << 48;
    return mask;
}

/** @brief  Gather 16 bytes from the 64-byte window held in @p regs at the byte offsets @p index (lanes in [0,64)),
 *          gather-free. VSX `vec_perm` addresses only a 32-byte register pair, so the low half `(regs[0],regs[1])`
 *          and high half `(regs[2],regs[3])` are each permuted and the per-lane `index >= 32` predicate selects the
 *          half. Lanes whose offset runs past the window read the zero padding (the value is discarded downstream). */
SZ_HELPER_INLINE __vector unsigned char sz_utf8_gather16_powervsx_( //
    __vector unsigned char const *regs, __vector unsigned char index) {
    __vector unsigned char const half_select_threshold_u8x16 = vec_splats((unsigned char)32);
    __vector unsigned char const gathered_low_u8x16 = vec_perm(regs[0], regs[1], index);
    __vector unsigned char const gathered_high_u8x16 = vec_perm(regs[2], regs[3],
                                                                vec_sub(index, half_select_threshold_u8x16));
    __vector bool char const select_high_mask = vec_cmpge(index, half_select_threshold_u8x16);
    return vec_sel(gathered_low_u8x16, gathered_high_u8x16, select_high_mask);
}

/**
 *  @brief  Left-pack the emitted start byte-offsets of one 16-lane register @p emit16 into @p packed (ascending),
 *          appending after @p produced already-packed offsets and returning the new count. The VSX `vpcompressb`
 *          analogue: split @p emit16 into low8 / high8, `vec_perm` the byte-index identity vector by the matching
 *          2 KB shuffle-LUT rows, add the register base @p base, and stitch the two halves by `popcount(low8)`.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_compress_starts_powervsx_( //
    sz_u32_t emit16, int base, sz_u8_t *packed, sz_size_t produced) {
    sz_u32_t const low8 = emit16 & 0xFFu;
    sz_u32_t const high8 = (emit16 >> 8) & 0xFFu;
    __vector unsigned char const byte_iota_u8x16 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    __vector unsigned char const register_base_u8x16 = vec_splats((unsigned char)base);
    __vector unsigned char const eight_u8x16 = vec_splats((unsigned char)8);

    // Low byte: indices 0..7 select packed local offsets; the LUT's 0xFF rows select garbage lanes we never store.
    __vector unsigned char const low_byte_shuffle_lut_u8x16 = vec_xl(0, sz_utf8_compress8_powervsx_[low8]);
    __vector unsigned char const low_byte_packed_u8x16 = vec_add(
        vec_perm(byte_iota_u8x16, byte_iota_u8x16, low_byte_shuffle_lut_u8x16), register_base_u8x16);
    // High byte: indices 0..7 are local offsets in [8,16); add 8 before the base.
    __vector unsigned char const high_byte_shuffle_lut_u8x16 = vec_xl(0, sz_utf8_compress8_powervsx_[high8]);
    __vector unsigned char const high_byte_packed_u8x16 = vec_add(
        vec_add(vec_perm(byte_iota_u8x16, byte_iota_u8x16, high_byte_shuffle_lut_u8x16), eight_u8x16),
        register_base_u8x16);

    sz_u128_vec_t low_bytes_vec, high_bytes_vec;
    low_bytes_vec.vsx_u8 = low_byte_packed_u8x16;
    high_bytes_vec.vsx_u8 = high_byte_packed_u8x16;
    int const low_count = sz_u32_popcount(low8);
    int const high_count = sz_u32_popcount(high8);
    for (int i = 0; i < low_count; ++i) packed[produced + (sz_size_t)i] = low_bytes_vec.u8s[i];
    for (int i = 0; i < high_count; ++i) packed[produced + (sz_size_t)low_count + (sz_size_t)i] = high_bytes_vec.u8s[i];
    return produced + (sz_size_t)low_count + (sz_size_t)high_count;
}

/**
 *  @brief  Decode the emitted start lanes @p emit_starts of a classified 64-byte window (held in the four @p regs
 *          16-byte registers) into sequential UTF-32 runes, the VSX sibling of
 *          @ref sz_utf8_rune_drain_icelake_. The start byte-offsets are
 *          left-packed by the shuffle-LUT, then a single 16-lane-block loop gathers the lead + up to three trailing
 *          bytes via @ref sz_utf8_gather16_powervsx_ and width-blends each codepoint; the wider 3rd/4th trailing
 *          bytes are gathered and blended CONDITIONALLY (@p has_three / @p has_four sibling `if`s). Every
 *          @p ill_formed start (its maximal ill-formed subpart) is overwritten with U+FFFD after the blend, and the
 *          resume cursor reads the last emitted start's offset + its compacted @p consumed_length (the maximal-subpart
 *          length), so an ill-formed trailing lane never skips bytes owed their own next U+FFFD.
 *  @return Number of runes emitted; sets @p consumed_bytes to the byte span they cover (the resume cursor delta).
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_rune_drain_powervsx_( //
    __vector unsigned char const *regs, sz_u64_t emit_starts, sz_u64_t ill_formed, sz_u8_t const *consumed_length,
    int has_three, int has_four, sz_size_t emit_count, sz_rune_t *runes, sz_size_t capacity,
    sz_size_t *consumed_bytes) {

    // Left-pack the emitted start offsets, register by register, via the shuffle-LUT (no scalar `ctz` walk).
    sz_u512_vec_t packed_offsets_vec;
    sz_size_t packed = 0;
    for (int reg = 0; reg < 4; ++reg)
        packed = sz_utf8_compress_starts_powervsx_((sz_u32_t)((emit_starts >> (reg * 16)) & 0xFFFFu), reg * 16,
                                                   packed_offsets_vec.u8s, packed);

    sz_size_t const want = emit_count < capacity ? emit_count : capacity;
    sz_size_t produced = 0;
    __vector unsigned char const one_byte_u8x16 = vec_splats((unsigned char)1);
    __vector unsigned char const two_byte_u8x16 = vec_splats((unsigned char)2);
    __vector unsigned char const three_byte_u8x16 = vec_splats((unsigned char)3);
    for (sz_size_t block_start = 0; block_start < want; block_start += 16) {
        // Load up to 16 packed start offsets; trailing lanes index >= 64 read the zero pad (value discarded below).
        sz_u128_vec_t index_lanes_vec;
        for (int lane = 0; lane < 16; ++lane) {
            sz_size_t const source = block_start + (sz_size_t)lane;
            index_lanes_vec.u8s[lane] = source < want ? packed_offsets_vec.u8s[source] : (sz_u8_t)0xFF;
        }
        __vector unsigned char const gather_index_u8x16 = index_lanes_vec.vsx_u8;
        __vector unsigned char const lead_byte_u8x16 = sz_utf8_gather16_powervsx_(regs, gather_index_u8x16);
        __vector unsigned char const continuation_byte_1_u8x16 = sz_utf8_gather16_powervsx_(
            regs, vec_add(gather_index_u8x16, one_byte_u8x16));
        __vector unsigned char const continuation_byte_2_u8x16 =
            has_three ? sz_utf8_gather16_powervsx_(regs, vec_add(gather_index_u8x16, two_byte_u8x16))
                      : vec_splats((unsigned char)0);
        __vector unsigned char const continuation_byte_3_u8x16 =
            has_four ? sz_utf8_gather16_powervsx_(regs, vec_add(gather_index_u8x16, three_byte_u8x16))
                     : vec_splats((unsigned char)0);

        // Widen each of the 16 byte lanes to a 32-bit codepoint lane and width-blend (1/2/3/4-byte) branchlessly.
        __vector unsigned char const selector_quarter_u8x16[4] = {
            {0x00, 0x10, 0x10, 0x10, 0x01, 0x10, 0x10, 0x10, 0x02, 0x10, 0x10, 0x10, 0x03, 0x10, 0x10, 0x10},
            {0x04, 0x10, 0x10, 0x10, 0x05, 0x10, 0x10, 0x10, 0x06, 0x10, 0x10, 0x10, 0x07, 0x10, 0x10, 0x10},
            {0x08, 0x10, 0x10, 0x10, 0x09, 0x10, 0x10, 0x10, 0x0a, 0x10, 0x10, 0x10, 0x0b, 0x10, 0x10, 0x10},
            {0x0c, 0x10, 0x10, 0x10, 0x0d, 0x10, 0x10, 0x10, 0x0e, 0x10, 0x10, 0x10, 0x0f, 0x10, 0x10, 0x10}};
        __vector unsigned char const zero_u8x16 = vec_splats((unsigned char)0);
        for (int quarter = 0; quarter < 4; ++quarter) {
            sz_size_t const out_base = block_start + (sz_size_t)quarter * 4;
            if (out_base >= want) break;
            __vector unsigned char const selector_u8x16 = selector_quarter_u8x16[quarter];
            __vector unsigned int const lead_u32x4 = (__vector unsigned int)vec_perm(lead_byte_u8x16, zero_u8x16,
                                                                                     selector_u8x16);
            __vector unsigned int const continuation_byte_1_u32x4 = (__vector unsigned int)vec_perm(
                continuation_byte_1_u8x16, zero_u8x16, selector_u8x16);
            __vector unsigned int const continuation_byte_2_u32x4 = (__vector unsigned int)vec_perm(
                continuation_byte_2_u8x16, zero_u8x16, selector_u8x16);
            __vector unsigned int const continuation_byte_3_u32x4 = (__vector unsigned int)vec_perm(
                continuation_byte_3_u8x16, zero_u8x16, selector_u8x16);

            __vector unsigned int const mask_word_3f_u32x4 = vec_splats((unsigned int)0x3F);
            __vector unsigned int const mask_word_c0_u32x4 = vec_splats((unsigned int)0xC0);
            __vector unsigned int const mask_word_e0_u32x4 = vec_splats((unsigned int)0xE0);
            __vector unsigned int const mask_word_f0_u32x4 = vec_splats((unsigned int)0xF0);
            __vector bool int const two_byte_mask_u32x4 = vec_and(vec_cmpge(lead_u32x4, mask_word_c0_u32x4),
                                                                  vec_cmplt(lead_u32x4, mask_word_e0_u32x4));
            __vector unsigned int const two_byte_codepoint_u32x4 = vec_or(
                vec_sl(vec_and(lead_u32x4, vec_splats((unsigned int)0x1F)), vec_splats((unsigned int)6)),
                vec_and(continuation_byte_1_u32x4, mask_word_3f_u32x4));
            __vector unsigned int decoded_codepoint_u32x4 = vec_sel(lead_u32x4, two_byte_codepoint_u32x4,
                                                                    two_byte_mask_u32x4);
            // The 3-/4-byte width blends run CONDITIONALLY via two sibling (depth-1, NOT nested) `if`s gated by
            // `has_three` / `has_four`, mirroring the AVX2 sibling-if drain: the block-level gather of the 3rd byte
            // is skipped for ASCII/2-byte windows and the 4th-byte gather + 4-byte assembly is skipped for CJK
            // (3-byte-only) windows. `has_four ⟹ has_three`, so the 4-byte sibling always sees a populated 3rd byte.
            // Bit-exact with the gated form: a window with no 3-/4-byte lead carries an all-false width mask.
            if (has_three) {
                __vector bool int const three_byte_mask_u32x4 = vec_and(vec_cmpge(lead_u32x4, mask_word_e0_u32x4),
                                                                        vec_cmplt(lead_u32x4, mask_word_f0_u32x4));
                __vector unsigned int const three_byte_codepoint_u32x4 = vec_or(
                    vec_or(vec_sl(vec_and(lead_u32x4, vec_splats((unsigned int)0x0F)), vec_splats((unsigned int)12)),
                           vec_sl(vec_and(continuation_byte_1_u32x4, mask_word_3f_u32x4), vec_splats((unsigned int)6))),
                    vec_and(continuation_byte_2_u32x4, mask_word_3f_u32x4));
                decoded_codepoint_u32x4 = vec_sel(decoded_codepoint_u32x4, three_byte_codepoint_u32x4,
                                                  three_byte_mask_u32x4);
            }
            if (has_four) { // SIBLING, not nested; has_four ⟹ has_three so the 3rd byte is already gathered.
                __vector bool int const four_byte_mask_u32x4 = vec_cmpge(lead_u32x4, mask_word_f0_u32x4);
                __vector unsigned int const four_byte_codepoint_u32x4 = vec_or(
                    vec_or(
                        vec_sl(vec_and(lead_u32x4, vec_splats((unsigned int)0x07)), vec_splats((unsigned int)18)),
                        vec_sl(vec_and(continuation_byte_1_u32x4, mask_word_3f_u32x4), vec_splats((unsigned int)12))),
                    vec_or(vec_sl(vec_and(continuation_byte_2_u32x4, mask_word_3f_u32x4), vec_splats((unsigned int)6)),
                           vec_and(continuation_byte_3_u32x4, mask_word_3f_u32x4)));
                decoded_codepoint_u32x4 = vec_sel(decoded_codepoint_u32x4, four_byte_codepoint_u32x4,
                                                  four_byte_mask_u32x4);
            }

            sz_u128_vec_t lane_values_vec;
            lane_values_vec.vsx_u32 = decoded_codepoint_u32x4;
            sz_size_t lanes = want - out_base;
            if (lanes > 4) lanes = 4;
            for (sz_size_t lane = 0; lane < lanes; ++lane) {
                sz_size_t const flat = out_base + lane;
                int const offset = packed_offsets_vec.u8s[flat];
                sz_rune_t value = (sz_rune_t)lane_values_vec.u32s[lane];
                if ((ill_formed >> offset) & 1) value = (sz_rune_t)sz_rune_replacement_k;
                runes[flat] = value;
            }
            produced += lanes;
        }
    }

    int const last_offset = packed_offsets_vec.u8s[produced - 1];
    *consumed_bytes = (sz_size_t)last_offset + (sz_size_t)consumed_length[last_offset];
    return produced;
}

#pragma endregion Shared SIMD leaf substrate

/**
 *  @brief  Decode one window of @p text into dense UTF-32 @p runes by the uniform "ASCII fast lane → classify →
 *          per-lane well-formed + orphan promotion → compress emitted starts → gather → width-blend → blend
 *          U+FFFD" path, the VSX sibling of @ref sz_utf8_decode_once_icelake_. The decode is TOTAL: clean and
 *          dirty bytes are handled in-vector, one U+FFFD per maximal ill-formed subpart (Unicode 17.0 §3.9 / W3C),
 *          bit-exact with @ref sz_utf8_decode_serial. The step declines (`*runes_unpacked == 0`, cursor
 *          unchanged) ONLY when the first lead's declared sequence crosses the window edge (a boundary truncation),
 *          which the public entry finalizes without a serial re-decode.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_decode_once_powervsx_( //
    sz_cptr_t text, sz_size_t length,                   //
    sz_rune_t *runes, sz_size_t runes_capacity,         //
    sz_size_t *runes_unpacked) {

    sz_size_t const chunk = length < 64 ? length : 64;

    // Stage the window into a zero-padded 64-byte buffer so tail lanes read as 0x00 (a 1-byte start of value 0 that
    // never overruns and gates clean); decodable starts never read past `chunk` by construction.
    sz_u512_vec_t staging_vec;
    for (sz_size_t index = 0; index < 64; ++index) staging_vec.u8s[index] = index < chunk ? (sz_u8_t)text[index] : 0;
    __vector unsigned char input_register_u8x16[4];
    input_register_u8x16[0] = vec_xl(0, staging_vec.u8s + 0);
    input_register_u8x16[1] = vec_xl(0, staging_vec.u8s + 16);
    input_register_u8x16[2] = vec_xl(0, staging_vec.u8s + 32);
    input_register_u8x16[3] = vec_xl(0, staging_vec.u8s + 48);

    // ASCII fast lane: a whole window of 1-byte runes widens directly with `vec_perm` zero-extend, no classification.
    __vector unsigned char const high_bit_mask_u8x16 = vec_splats((unsigned char)0x80);
    __vector unsigned char nonascii_mask_u8x16[4];
    for (int reg = 0; reg < 4; ++reg)
        nonascii_mask_u8x16[reg] = (__vector unsigned char)vec_cmpge(input_register_u8x16[reg], high_bit_mask_u8x16);
    sz_u64_t const loaded_mask = sz_u64_mask_until_serial_(chunk);
    sz_u64_t const nonascii = sz_utf8_mask_combine_powervsx_(nonascii_mask_u8x16) & loaded_mask;
    if (nonascii == 0) {
        sz_size_t const want = chunk < runes_capacity ? chunk : runes_capacity;
        for (sz_size_t index = 0; index < want; ++index) runes[index] = (sz_rune_t)staging_vec.u8s[index];
        *runes_unpacked = want;
        return text + want;
    }

    // Single-source classification via the high-nibble length LUT (the SAME table the serial reference uses), so a
    // lead and its declared length can never disagree. The per-register movemasks assemble the 64-bit lane masks.
    __vector unsigned char const length_lookup_table_u8x16 = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    __vector unsigned char const continuation_mask_u8x16 = vec_splats((unsigned char)0xC0);
    __vector unsigned char const continuation_pattern_u8x16 = vec_splats((unsigned char)0x80);
    __vector unsigned char const low_nibble_mask_u8x16 = vec_splats((unsigned char)0x0F);
    __vector unsigned char continuation_mask_per_register_u8x16[4], length_2_mask_u8x16[4], length_3_mask_u8x16[4],
        length_4_mask_u8x16[4];
    __vector unsigned char sequence_length_per_register_u8x16[4];
    for (int reg = 0; reg < 4; ++reg) {
        __vector unsigned char const register_value_u8x16 = input_register_u8x16[reg];
        continuation_mask_per_register_u8x16[reg] = (__vector unsigned char)vec_cmpeq(
            vec_and(register_value_u8x16, continuation_mask_u8x16), continuation_pattern_u8x16);
        __vector unsigned char const high_nibble_u8x16 = vec_and(
            vec_sr(register_value_u8x16, vec_splats((unsigned char)4)), low_nibble_mask_u8x16);
        __vector unsigned char const sequence_length_u8x16 = vec_perm(length_lookup_table_u8x16,
                                                                      length_lookup_table_u8x16, high_nibble_u8x16);
        sequence_length_per_register_u8x16[reg] = sequence_length_u8x16;
        length_2_mask_u8x16[reg] = (__vector unsigned char)vec_cmpeq(sequence_length_u8x16,
                                                                     vec_splats((unsigned char)2));
        length_3_mask_u8x16[reg] = (__vector unsigned char)vec_cmpeq(sequence_length_u8x16,
                                                                     vec_splats((unsigned char)3));
        length_4_mask_u8x16[reg] = (__vector unsigned char)vec_cmpeq(sequence_length_u8x16,
                                                                     vec_splats((unsigned char)4));
    }
    sz_u64_t const continuation_bits = sz_utf8_mask_combine_powervsx_(continuation_mask_per_register_u8x16) &
                                       loaded_mask;
    sz_u64_t const starts_bits = loaded_mask & ~continuation_bits;
    sz_u64_t const length_ge_two = sz_utf8_mask_combine_powervsx_(length_2_mask_u8x16) |
                                   sz_utf8_mask_combine_powervsx_(length_3_mask_u8x16) |
                                   sz_utf8_mask_combine_powervsx_(length_4_mask_u8x16);
    sz_u64_t const length_ge_three = sz_utf8_mask_combine_powervsx_(length_3_mask_u8x16) |
                                     sz_utf8_mask_combine_powervsx_(length_4_mask_u8x16);
    sz_u64_t const length_ge_four = sz_utf8_mask_combine_powervsx_(length_4_mask_u8x16);
    sz_u64_t const length_ge_two_starts = length_ge_two & starts_bits;
    sz_u64_t const length_ge_three_starts = length_ge_three & starts_bits;
    sz_u64_t const length_ge_four_starts = length_ge_four & starts_bits;

    // Branchless overrun defer: per lane `index + length`, compared against `chunk`; the FIRST overrunning start
    // bounds the decodable prefix (well-formed text overruns only at the trailing truncation; a malformed `E0 C0`
    // overruns earlier). A `vec_vbpermq` movemask + one `ctz` finds that first lane (no scalar per-lane loop).
    __vector unsigned char const byte_iota_u8x16 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    __vector unsigned char const chunk_size_u8x16 = vec_splats((unsigned char)chunk);
    __vector unsigned char overrun_mask_u8x16[4];
    for (int reg = 0; reg < 4; ++reg) {
        __vector unsigned char const global_lane_offset_u8x16 = vec_add(byte_iota_u8x16,
                                                                        vec_splats((unsigned char)(reg * 16)));
        __vector unsigned char const sequence_end_offset_u8x16 = vec_add(global_lane_offset_u8x16,
                                                                         sequence_length_per_register_u8x16[reg]);
        overrun_mask_u8x16[reg] = (__vector unsigned char)vec_cmpgt(sequence_end_offset_u8x16, chunk_size_u8x16);
    }
    sz_u64_t const overruns = sz_utf8_mask_combine_powervsx_(overrun_mask_u8x16) & starts_bits;
    sz_size_t const decodable_end = overruns ? (sz_size_t)sz_u64_ctz(overruns) : chunk;
    sz_u64_t const decodable_mask = sz_u64_mask_until_serial_(decodable_end);

    // First-continuation availability + range checks for E0/ED/F0/F4 (overlong / surrogate / out-of-range), and the
    // bad-lead test (C0/C1 length-2 by the LUT, F5..FF length-4 by the LUT), all as `sz_u64_t` mask algebra. The first
    // continuation of a lead in register `reg` is the next byte, gathered by a +1 `vec_perm` over `(reg, reg+1)`.
    __vector unsigned char const successor_iota_u8x16 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    sz_u64_t bad_lead_bits = 0, overlong_or_surrogate_or_range_bits = 0;
    int const has_three = (length_ge_three_starts) != 0;
    int const has_four = (length_ge_four_starts) != 0;
    {
        __vector unsigned char bad_lead_mask_u8x16[4];
        __vector unsigned char overlong_surrogate_range_mask_u8x16[4];
        __vector unsigned char const byte_c2_u8x16 = vec_splats((unsigned char)0xC2);
        __vector unsigned char const byte_f4_u8x16 = vec_splats((unsigned char)0xF4);
        __vector unsigned char const byte_a0_u8x16 = vec_splats((unsigned char)0xA0);
        __vector unsigned char const byte_90_u8x16 = vec_splats((unsigned char)0x90);
        __vector unsigned char const byte_e0_u8x16 = vec_splats((unsigned char)0xE0);
        __vector unsigned char const byte_ed_u8x16 = vec_splats((unsigned char)0xED);
        __vector unsigned char const byte_f0_u8x16 = vec_splats((unsigned char)0xF0);
        for (int reg = 0; reg < 4; ++reg) {
            __vector unsigned char const register_value_u8x16 = input_register_u8x16[reg];
            __vector unsigned char const successor_register_u8x16 = (reg < 3) ? input_register_u8x16[reg + 1]
                                                                              : input_register_u8x16[3];
            __vector unsigned char const next_byte_u8x16 = vec_perm(register_value_u8x16, successor_register_u8x16,
                                                                    successor_iota_u8x16);
            __vector bool char const length_2_mask = vec_cmpeq(sequence_length_per_register_u8x16[reg],
                                                               vec_splats((unsigned char)2));
            __vector bool char const length_4_mask = vec_cmpeq(sequence_length_per_register_u8x16[reg],
                                                               vec_splats((unsigned char)4));
            __vector unsigned char const lead_lt_c2_mask_u8x16 = (__vector unsigned char)vec_cmplt(register_value_u8x16,
                                                                                                   byte_c2_u8x16);
            __vector unsigned char const lead_gt_f4_mask_u8x16 = (__vector unsigned char)vec_cmpgt(register_value_u8x16,
                                                                                                   byte_f4_u8x16);
            bad_lead_mask_u8x16[reg] = vec_or(vec_and((__vector unsigned char)length_2_mask, lead_lt_c2_mask_u8x16),
                                              vec_and((__vector unsigned char)length_4_mask, lead_gt_f4_mask_u8x16));
            __vector unsigned char const e0_bad_mask_u8x16 = vec_and(
                (__vector unsigned char)vec_cmpeq(register_value_u8x16, byte_e0_u8x16),
                (__vector unsigned char)vec_cmplt(next_byte_u8x16, byte_a0_u8x16));
            __vector unsigned char const ed_bad_mask_u8x16 = vec_and(
                (__vector unsigned char)vec_cmpeq(register_value_u8x16, byte_ed_u8x16),
                (__vector unsigned char)vec_cmpge(next_byte_u8x16, byte_a0_u8x16));
            __vector unsigned char const f0_bad_mask_u8x16 = vec_and(
                (__vector unsigned char)vec_cmpeq(register_value_u8x16, byte_f0_u8x16),
                (__vector unsigned char)vec_cmplt(next_byte_u8x16, byte_90_u8x16));
            __vector unsigned char const f4_bad_mask_u8x16 = vec_and(
                (__vector unsigned char)vec_cmpeq(register_value_u8x16, byte_f4_u8x16),
                (__vector unsigned char)vec_cmpge(next_byte_u8x16, byte_90_u8x16));
            overlong_surrogate_range_mask_u8x16[reg] = vec_or(vec_or(e0_bad_mask_u8x16, ed_bad_mask_u8x16),
                                                              vec_or(f0_bad_mask_u8x16, f4_bad_mask_u8x16));
        }
        bad_lead_bits = sz_utf8_mask_combine_powervsx_(bad_lead_mask_u8x16) & starts_bits;
        overlong_or_surrogate_or_range_bits = sz_utf8_mask_combine_powervsx_(overlong_surrogate_range_mask_u8x16) &
                                              starts_bits;
    }

    // Per-lane continuation availability at the declared trailing slots, evaluated at the lead lane (mirror icelake).
    sz_u64_t const cont1 = starts_bits & (continuation_bits >> 1);
    sz_u64_t const cont2 = starts_bits & (continuation_bits >> 2);
    sz_u64_t const cont3 = starts_bits & (continuation_bits >> 3);
    sz_u64_t const b1_range_ok = starts_bits & ~overlong_or_surrogate_or_range_bits;
    sz_u64_t const first_ok = cont1 & b1_range_ok & ~bad_lead_bits;

    sz_u64_t const length_one_bits = starts_bits & ~length_ge_two_starts;
    sz_u64_t const length_two_bits = length_ge_two_starts & ~length_ge_three_starts;
    sz_u64_t const wf1 = length_one_bits;
    sz_u64_t const wf2 = length_two_bits & ~bad_lead_bits & cont1;
    sz_u64_t const wf3 = (length_ge_three_starts & ~length_ge_four_starts) & b1_range_ok & cont1 & cont2;
    sz_u64_t const wf4 = length_ge_four_starts & ~bad_lead_bits & b1_range_ok & cont1 & cont2 & cont3;
    sz_u64_t const well_formed = (wf1 | wf2 | wf3 | wf4) & decodable_mask;

    // Per-lane maximal-subpart length: extend across each continuation slot a well-formed sequence would accept.
    sz_u64_t const step2 = length_ge_two_starts & first_ok;
    sz_u64_t const step3 = step2 & length_ge_three_starts & cont2;
    sz_u64_t const step4 = step3 & length_ge_four_starts & cont3;

    // Orphan promotion: a continuation byte covered by NO lead's maximal-subpart span becomes its own 1-byte U+FFFD.
    sz_u64_t const covered = ((step2 & decodable_mask) << 1) | ((step3 & decodable_mask) << 2) |
                             ((step4 & decodable_mask) << 3);
    sz_u64_t const orphan = continuation_bits & decodable_mask & ~covered;
    sz_u64_t const emit_starts = (starts_bits | orphan) & decodable_mask;
    sz_size_t const emit_count = (sz_size_t)sz_u64_popcount(emit_starts);
    if (emit_count == 0) { return *runes_unpacked = 0, text; } // Window-edge truncation → driver finalize.
    sz_u64_t const ill_formed = emit_starts & ~well_formed;

    // Per-lane (window-order) maximal-subpart length: 1 + the continuation slots the steps reached.
    sz_u512_vec_t consumed_length_vec;
    for (int lane = 0; lane < 64; ++lane) {
        sz_u8_t length_here = 1;
        length_here = (sz_u8_t)(length_here + (sz_u8_t)((step2 >> lane) & 1));
        length_here = (sz_u8_t)(length_here + (sz_u8_t)((step3 >> lane) & 1));
        length_here = (sz_u8_t)(length_here + (sz_u8_t)((step4 >> lane) & 1));
        consumed_length_vec.u8s[lane] = length_here;
    }

    sz_size_t consumed = 0;
    sz_size_t const produced = sz_utf8_rune_drain_powervsx_(input_register_u8x16, emit_starts, ill_formed,
                                                            consumed_length_vec.u8s, has_three, has_four, emit_count,
                                                            runes, runes_capacity, &consumed);
    *runes_unpacked = produced;
    return text + consumed;
}

#endif // !SZ_IS_BIG_ENDIAN_

SZ_API_COMPTIME sz_cptr_t sz_utf8_decode_powervsx( //
    sz_cptr_t text, sz_size_t length,              //
    sz_rune_t *runes, sz_size_t runes_capacity,    //
    sz_size_t *runes_unpacked) {
#if !SZ_IS_BIG_ENDIAN_
    sz_cptr_t cursor = text;
    sz_cptr_t const end = text + length;
    sz_size_t runes_written = 0;
    while (runes_written < runes_capacity && cursor < end) {
        sz_size_t step_unpacked = 0;
        sz_cptr_t next = sz_utf8_decode_once_powervsx_(cursor, (sz_size_t)(end - cursor), runes + runes_written,
                                                       runes_capacity - runes_written, &step_unpacked);
        if (step_unpacked) {
            runes_written += step_unpacked;
            cursor = next;
            continue;
        }
        // The in-vector step decodes its whole decodable span; `step_unpacked == 0` only when the very first lead
        // declares a sequence crossing the window edge (a boundary truncation). A resumable truncation breaks and
        // awaits more bytes; a bad/overlong truncated lead at the edge finalizes to one U+FFFD over its maximal
        // ill-formed subpart — a bounded <=3-byte finalize, never a serial window re-decode.
        if (sz_utf8_incomplete_tail_(cursor, end)) break;
        runes[runes_written++] = (sz_rune_t)sz_rune_replacement_k;
        cursor += sz_utf8_maximal_subpart_(cursor, end);
    }
    *runes_unpacked = runes_written;
    return cursor;
#else
    return sz_utf8_decode_serial(text, length, runes, runes_capacity, runes_unpacked);
#endif
}

#pragma region Multistep newline / whitespace iteration

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_POWERVSX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_RUNES_POWERVSX_H_
