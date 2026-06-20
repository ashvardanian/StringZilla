/**
 *  @brief POWER VSX backend for UTF-8 codepoint mechanics.
 *  @file include/stringzilla/utf8_codepoints/powervsx.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_CODEPOINTS_POWERVSX_H_
#define STRINGZILLA_UTF8_CODEPOINTS_POWERVSX_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_codepoints/serial.h"

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

SZ_PUBLIC sz_size_t sz_utf8_count_powervsx(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t char_count = 0;

    __vector unsigned char const mask_c0_vec = vec_splats((unsigned char)0xC0);
    __vector unsigned char const pat_80_vec = vec_splats((unsigned char)0x80);
    __vector unsigned char const ones_vec = vec_splats((unsigned char)1);

    // `vec_msum(starts, ones, acc)` is the in-lane accumulator: each `starts` byte is 0 or 1, so each
    // step adds at most 4 to any 32-bit lane. A lane can absorb `UINT_MAX / 4 ≈ 1.07B` iterations
    // before overflowing. We process the input in blocks of that many 16-byte windows so the only
    // horizontal reduction happens once per block — out of the hot inner loop — and the inner loop
    // carries no per-iteration counter or branch. Results stay identical to `sz_utf8_count_serial`.
    sz_size_t const block_windows = (sz_size_t)1000000000; // 1B * 4 < UINT_MAX.
    while (length >= 16) {
        sz_size_t windows = length / 16;
        if (windows > block_windows) windows = block_windows;
        __vector unsigned int accumulator_vec = vec_splats((unsigned int)0);
        for (sz_size_t window_index = 0; window_index < windows; ++window_index, text_u8 += 16) {
            __vector unsigned char bytes_vec = vec_xl(0, text_u8);
            __vector unsigned char headers_vec = vec_and(bytes_vec, mask_c0_vec);
            // Continuation bytes → 0xFF, starts → 0x00; invert so each start contributes 0x01.
            __vector unsigned char is_continuation_vec = (__vector unsigned char)vec_cmpeq(headers_vec, pat_80_vec);
            __vector unsigned char starts_vec = vec_andc(ones_vec, is_continuation_vec);
            accumulator_vec = vec_msum(starts_vec, ones_vec,
                                       accumulator_vec); // Sum the start flags into the 32-bit lanes.
        }
        char_count += (sz_size_t)accumulator_vec[0] + accumulator_vec[1] + accumulator_vec[2] + accumulator_vec[3];
        length -= windows * 16;
    }

    if (length) char_count += sz_utf8_count_serial((sz_cptr_t)text_u8, length);
    return char_count;
}

/*  x86-`movemask`-equivalent for VSX: gathers the MSB of each of the 16 bytes into bit `i` (lowest-addressed
 *  byte -> bit 0) via `vec_vbpermq`, identical to `sz_utf8_movemask_powervsx_` below but reachable before its
 *  first user (`sz_utf8_find_nth_powervsx` and the multistep iterators); distinctly named so both coexist in
 *  one translation unit. */
SZ_INTERNAL sz_u32_t sz_utf8_iterate_movemask_powervsx_(__vector unsigned char compared) {
    __vector unsigned char const indices = {120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0};
    __vector unsigned long long const gathered = vec_vbpermq(compared, indices);
#if SZ_IS_BIG_ENDIAN_
    return (sz_u32_t)(gathered[0] & 0xFFFFull);
#else
    return (sz_u32_t)(gathered[1] & 0xFFFFull);
#endif
}

/**
 *  @brief Locate the start byte of the n-th code-point (0-indexed), mirroring `sz_utf8_find_nth_serial`.
 *         Per tile a code-point-start bitmask `~continuation_mask` is popcounted to skip whole tiles; the
 *         n-th start's lane comes from `sz_u32_nth_set_bit`. The `< 16` tail defers to the serial reference.
 */
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_powervsx(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;

    __vector unsigned char const mask_c0_vec = vec_splats((unsigned char)0xC0);
    __vector unsigned char const pat_80_vec = vec_splats((unsigned char)0x80);

    while (length >= 16) {
        __vector unsigned char bytes_vec = vec_xl(0, text_u8);
        __vector unsigned char headers_vec = vec_and(bytes_vec, mask_c0_vec);
        __vector unsigned char is_continuation_vec = (__vector unsigned char)vec_cmpeq(headers_vec, pat_80_vec);
        sz_u32_t start_bits = (~sz_utf8_iterate_movemask_powervsx_(is_continuation_vec)) & 0xFFFFu;
        sz_size_t const start_count = (sz_size_t)sz_u32_popcount(start_bits);

        if (n >= start_count) {
            n -= start_count, text_u8 += 16, length -= 16;
            continue;
        }
        return (sz_cptr_t)(text_u8 + sz_u32_nth_set_bit(start_bits, n));
    }

    return sz_utf8_find_nth_serial((sz_cptr_t)text_u8, length, n);
}

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_powervsx( //
    sz_cptr_t text, sz_size_t length,              //
    sz_rune_t *runes, sz_size_t runes_capacity,    //
    sz_size_t *runes_unpacked) {
#if !SZ_IS_BIG_ENDIAN_
    sz_u8_t const *text_cursor = (sz_u8_t const *)text;
    sz_size_t runes_written = 0;

    // `vec_perm` selectors that zero-extend bytes [0..3], [4..7], [8..11], [12..15] of a 16-byte
    // vector into four `u32` lanes each. Index `0x10` selects a byte from the (zero) second operand,
    // placing 0x00 in the upper three bytes of every little-endian 32-bit lane.
    __vector unsigned char const selector0_vec = {0x00, 0x10, 0x10, 0x10, 0x01, 0x10, 0x10, 0x10,
                                                  0x02, 0x10, 0x10, 0x10, 0x03, 0x10, 0x10, 0x10};
    __vector unsigned char const selector1_vec = {0x04, 0x10, 0x10, 0x10, 0x05, 0x10, 0x10, 0x10,
                                                  0x06, 0x10, 0x10, 0x10, 0x07, 0x10, 0x10, 0x10};
    __vector unsigned char const selector2_vec = {0x08, 0x10, 0x10, 0x10, 0x09, 0x10, 0x10, 0x10,
                                                  0x0a, 0x10, 0x10, 0x10, 0x0b, 0x10, 0x10, 0x10};
    __vector unsigned char const selector3_vec = {0x0c, 0x10, 0x10, 0x10, 0x0d, 0x10, 0x10, 0x10,
                                                  0x0e, 0x10, 0x10, 0x10, 0x0f, 0x10, 0x10, 0x10};
    __vector unsigned char const zero_vec = vec_splats((unsigned char)0);

    while (length >= 16 && runes_written + 16 <= runes_capacity) {
        __vector unsigned char bytes_vec = vec_xl(0, text_cursor);
        // A window is pure ASCII when no byte has its high bit set.
        if (vec_any_ge(bytes_vec, vec_splats((unsigned char)0x80))) break;
        vec_xst((__vector unsigned int)vec_perm(bytes_vec, zero_vec, selector0_vec), 0,
                (unsigned int *)(runes + runes_written + 0));
        vec_xst((__vector unsigned int)vec_perm(bytes_vec, zero_vec, selector1_vec), 0,
                (unsigned int *)(runes + runes_written + 4));
        vec_xst((__vector unsigned int)vec_perm(bytes_vec, zero_vec, selector2_vec), 0,
                (unsigned int *)(runes + runes_written + 8));
        vec_xst((__vector unsigned int)vec_perm(bytes_vec, zero_vec, selector3_vec), 0,
                (unsigned int *)(runes + runes_written + 12));
        runes_written += 16;
        text_cursor += 16;
        length -= 16;
    }

    // Finish the remainder (non-ASCII or sub-window) with the serial decoder so the output and the
    // returned cursor stay bit-exact with `sz_utf8_unpack_chunk_serial`.
    sz_size_t tail_unpacked = 0;
    sz_cptr_t cursor = sz_utf8_unpack_chunk_serial((sz_cptr_t)text_cursor, length, runes + runes_written,
                                                   runes_capacity - runes_written, &tail_unpacked);
    *runes_unpacked = runes_written + tail_unpacked;
    return cursor;
#else
    return sz_utf8_unpack_chunk_serial(text, length, runes, runes_capacity, runes_unpacked);
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

#endif // STRINGZILLA_UTF8_CODEPOINTS_POWERVSX_H_
