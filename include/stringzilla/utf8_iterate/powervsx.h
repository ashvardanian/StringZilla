/**
 *  @brief IBM Power VSX backend for utf8.
 *  @file include/stringzilla/utf8_iterate/powervsx.h
 *  @author Ash Vardanian
 *  @sa utf8.h
 */
#ifndef STRINGZILLA_UTF8_ITERATE_POWERVSX_H_
#define STRINGZILLA_UTF8_ITERATE_POWERVSX_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_iterate/serial.h"

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

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_powervsx(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    //! Extracting the n-th code-point start requires a per-bit population scan; not worth
    //! vectorizing without a PDEP-equivalent — defer to serial, mirroring the NEON backend.
    return sz_utf8_find_nth_serial(text, length, n);
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

/*  Multistep newline / whitespace iteration (IBM Power VSX / Power9).
 *
 *  Each 16-byte window is classified branchlessly into a `start_bits` bitmask (every delimiter start) plus
 *  disjoint 2-/3-byte start bitmasks, all extracted with the single-instruction `sz_utf8_movemask_powervsx_`
 *  (bit `i` carries the MSB of lane `i`, i.e. byte `position+i`). VSX has no `vpcompressb`, so the peel
 *  left-packs the 16 lanes in eight 2-lane sub-blocks with a `vec_perm` table (a VSX vector holds two `u64`,
 *  so two is the natural compaction width), building the absolute `(position+lane, length)` u64 pairs from
 *  `start_bits` and the disjoint 2-/3-byte masks - no `ctz` index-find and no group/else branch. VSX has no
 *  masked store, so each compacted sub-block is full-stored to a fixed-width stack scratch advancing by its
 *  surviving count, then the low `emit_count` entries are copied to the output (mirroring the NEON peel). We
 *  trust starts in lanes [0,13] and step 14 so any <=3-byte delimiter from a trusted lane is fully loaded;
 *  a 1-byte `t[pos-1] == '\r'` carry suppresses an LF that completes a CRLF straddling the window edge (the
 *  only delimiter whose tail is itself a match). The caller computes the capacity cut like the other backends:
 *  it peels only `emit_count` matches, then resumes past the last emitted match if the buffer filled. */

/*  x86-`movemask`-equivalent for VSX: gathers the MSB of each of the 16 bytes into bit `i` (lowest-addressed
 *  byte -> bit 0) via `vec_vbpermq`, identical to `sz_utf8_movemask_powervsx_` below but reachable before it
 *  for the multistep iterators (and distinctly named so both can coexist in one translation unit). */
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
 *  @brief  Peel the window's first `emit_count` matches by SIMD left-pack (no `ctz`, no per-match branch).
 *
 *  Walks the 16-lane `start_bits` mask in eight ascending 2-lane sub-blocks. Each sub-block builds the two
 *  candidate `(position+lane, length)` `u64` pairs, gathers the set lanes to the front of a `vector unsigned
 *  long long` with one `vec_perm` (a byte-granular permute driven by `compact2_lut`, the same left-pack idea as
 *  the NEON / Haswell peels), and full-stores both survivors to a fixed-width stack scratch advancing by the
 *  sub-block's surviving count. VSX has no masked store, so the scratch is 18-wide to absorb the trailing
 *  2-lane spill of the last compacted sub-block (at most 14 trusted matches per window); the low `emit_count`
 *  entries are then copied to the caller's output, preserving ascending lane order and the original
 *  `emit_count` truncation byte-for-byte. */
SZ_INTERNAL void sz_utf8_iterate_peel_window_powervsx_(                        //
    sz_u32_t start_bits, sz_u32_t two_byte_starts, sz_u32_t three_byte_starts, //
    sz_size_t emit_count, sz_size_t position,                                  //
    sz_size_t *match_offsets, sz_size_t *match_lengths) {

    // Byte-permutation rows for the four 2-bit sub-block masks (lane 0 = bytes [0,8), lane 1 = bytes [8,16)):
    // row `[m]` gathers the `m`-selected `u64` lanes to the front of a `vector unsigned long long` via `vec_perm`.
    // Rows in mask order: none (unused identity), lane 0 only, lane 1 only, both lanes.
    static unsigned char const compact2_lut[4][16] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    };
    static sz_size_t const popcount2_lut[4] = {0, 1, 1, 2};

    sz_size_t scratch_offsets[18], scratch_lengths[18];
    sz_size_t filled = 0;
    for (sz_size_t sub_block = 0; sub_block < 8; ++sub_block) {
        sz_size_t const base_lane = sub_block * 2;
        sz_u32_t const submask = (start_bits >> base_lane) & 0x3u;
        if (!submask) continue;

        // Per-lane length: 1, plus 1 on a 2-byte start, plus 2 on a 3-byte start (the masks are disjoint).
        sz_u32_t const two_byte_sub = (two_byte_starts >> base_lane) & 0x3u;
        sz_u32_t const three_byte_sub = (three_byte_starts >> base_lane) & 0x3u;
        __vector unsigned long long const candidate_offsets = {(unsigned long long)(position + base_lane),
                                                               (unsigned long long)(position + base_lane + 1)};
        __vector unsigned long long const candidate_lengths = {
            1u + (two_byte_sub & 1u) + 2u * (three_byte_sub & 1u),
            1u + ((two_byte_sub >> 1) & 1u) + 2u * ((three_byte_sub >> 1) & 1u)};

        __vector unsigned char const permutation = vec_xl(0, (unsigned char const *)compact2_lut[submask]);
        __vector unsigned long long const packed_offsets = (__vector unsigned long long)vec_perm(
            (__vector unsigned char)candidate_offsets, (__vector unsigned char)candidate_offsets, permutation);
        __vector unsigned long long const packed_lengths = (__vector unsigned long long)vec_perm(
            (__vector unsigned char)candidate_lengths, (__vector unsigned char)candidate_lengths, permutation);
        vec_xst(packed_offsets, 0, (unsigned long long *)(scratch_offsets + filled));
        vec_xst(packed_lengths, 0, (unsigned long long *)(scratch_lengths + filled));
        filled += popcount2_lut[submask];
    }

    for (sz_size_t emitted = 0; emitted < emit_count; ++emitted)
        match_offsets[emitted] = scratch_offsets[emitted], match_lengths[emitted] = scratch_lengths[emitted];
}

SZ_PUBLIC sz_size_t sz_utf8_find_newlines_powervsx(     //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;

    __vector unsigned char const newline_vec = vec_splats((unsigned char)'\n');
    __vector unsigned char const vertical_tab_vec = vec_splats((unsigned char)'\v');
    __vector unsigned char const form_feed_vec = vec_splats((unsigned char)'\f');
    __vector unsigned char const carriage_return_vec = vec_splats((unsigned char)'\r');
    __vector unsigned char const lead_c2_vec = vec_splats((unsigned char)0xC2);
    __vector unsigned char const x_85_vec = vec_splats((unsigned char)0x85);
    __vector unsigned char const lead_e2_vec = vec_splats((unsigned char)0xE2);
    __vector unsigned char const byte_80_vec = vec_splats((unsigned char)0x80);
    __vector unsigned char const x_a8_vec = vec_splats((unsigned char)0xA8);
    __vector unsigned char const x_a9_vec = vec_splats((unsigned char)0xA9);

    while (position + 16 <= length && count < matches_capacity) {
        __vector unsigned char window = vec_xl(0, text_u8 + position);
        sz_u32_t newline_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window, newline_vec));
        sz_u32_t carriage_return_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window, carriage_return_vec));
        sz_u32_t one_byte_bits =
            newline_bits | carriage_return_bits |
            sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, vertical_tab_vec)) |
            sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, form_feed_vec));

        // 2-byte NEL (C2 85); 3-byte LS/PS (E2 80 A8/A9) - bit `i+1` is the next lane, so suffixes shift right.
        sz_u32_t lead_c2_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window, lead_c2_vec));
        sz_u32_t x_85_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_85_vec));
        sz_u32_t nel_bits = lead_c2_bits & (x_85_bits >> 1);
        sz_u32_t lead_e2_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window, lead_e2_vec));
        sz_u32_t byte_80_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window, byte_80_vec));
        sz_u32_t lead_e280_bits = lead_e2_bits & (byte_80_bits >> 1);
        sz_u32_t x_a8_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_a8_vec));
        sz_u32_t x_a9_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_a9_vec));
        sz_u32_t line_para_bits = lead_e280_bits & ((x_a8_bits | x_a9_bits) >> 2);

        // CRLF: a CR whose next lane is LF is a single 2-byte match; its trailing LF must not also be emitted.
        sz_u32_t crlf_bits = carriage_return_bits & (newline_bits >> 1);
        sz_u32_t lf_of_crlf_bits = newline_bits & (carriage_return_bits << 1);

        sz_u32_t two_byte_starts = crlf_bits | nel_bits;
        sz_u32_t three_byte_starts = line_para_bits;
        sz_u32_t start_bits = (one_byte_bits | nel_bits | line_para_bits) & ~lf_of_crlf_bits;
        start_bits &= (sz_u32_t)0x3FFF; // trust lanes [0,13]; step 14

        // Suppress a leading LF already consumed by a CRLF that straddled the previous window edge.
        if (position != 0 && text_u8[position - 1] == '\r') start_bits &= ~(newline_bits & (sz_u32_t)1);

        sz_size_t const window_matches = (sz_size_t)sz_u32_popcount(start_bits);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_window_powervsx_(start_bits, two_byte_starts, three_byte_starts, emit_count, position,
                                                  match_offsets + count, match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) { // output buffer full: resume past the last emitted match.
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += 14;
    }

    // Skip a CRLF's trailing LF if it straddles into the serial tail (the CR was emitted as a 2-byte match).
    if (position != 0 && position < length && text_u8[position - 1] == '\r' && text_u8[position] == '\n') ++position;
    count += sz_utf8_find_newlines_serial_((sz_cptr_t)(text_u8 + position), length - position, position,
                                           match_offsets + count, match_lengths + count, matches_capacity - count,
                                           bytes_consumed);
    return count;
}

SZ_PUBLIC sz_size_t sz_utf8_find_whitespaces_powervsx(  //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;

    __vector unsigned char const tab_vec = vec_splats((unsigned char)'\t');
    __vector unsigned char const carriage_return_vec = vec_splats((unsigned char)'\r');
    __vector unsigned char const x_20_vec = vec_splats((unsigned char)' ');
    __vector unsigned char const lead_c2_vec = vec_splats((unsigned char)0xC2);
    __vector unsigned char const x_85_vec = vec_splats((unsigned char)0x85);
    __vector unsigned char const x_a0_vec = vec_splats((unsigned char)0xA0);
    __vector unsigned char const x_e1_vec = vec_splats((unsigned char)0xE1);
    __vector unsigned char const lead_e2_vec = vec_splats((unsigned char)0xE2);
    __vector unsigned char const x_e3_vec = vec_splats((unsigned char)0xE3);
    __vector unsigned char const x_9a_vec = vec_splats((unsigned char)0x9A);
    __vector unsigned char const byte_80_vec = vec_splats((unsigned char)0x80);
    __vector unsigned char const x_81_vec = vec_splats((unsigned char)0x81);
    __vector unsigned char const x_8d_vec = vec_splats((unsigned char)0x8D);
    __vector unsigned char const x_a8_vec = vec_splats((unsigned char)0xA8);
    __vector unsigned char const x_a9_vec = vec_splats((unsigned char)0xA9);
    __vector unsigned char const x_af_vec = vec_splats((unsigned char)0xAF);
    __vector unsigned char const x_9f_vec = vec_splats((unsigned char)0x9F);

    while (position + 16 <= length && count < matches_capacity) {
        __vector unsigned char window = vec_xl(0, text_u8 + position);
        // 1-byte: space, plus the contiguous range [\t, \r] == [9, 13].
        sz_u32_t one_byte_bits =
            sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_20_vec)) |
            sz_utf8_iterate_movemask_powervsx_(vec_and((__vector unsigned char)vec_cmpge(window, tab_vec),
                                                       (__vector unsigned char)vec_cmple(window, carriage_return_vec)));

        // 2-byte: C2 85 (NEL), C2 A0 (NBSP).
        sz_u32_t lead_c2_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window, lead_c2_vec));
        sz_u32_t x_85_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_85_vec));
        sz_u32_t x_a0_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_a0_vec));
        sz_u32_t two_byte_starts = lead_c2_bits & ((x_85_bits >> 1) | (x_a0_bits >> 1));

        // 3-byte: E1 9A 80 (ogham); E2 80 [80-8D]; E2 80 AF; E2 81 9F; E2 80 A8/A9; E3 80 80.
        sz_u32_t byte_80_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window, byte_80_vec));
        sz_u32_t lead_e2_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window, lead_e2_vec));
        sz_u32_t lead_e280_bits = lead_e2_bits & (byte_80_bits >> 1);
        sz_u32_t x_e1_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_e1_vec));
        sz_u32_t x_9a_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_9a_vec));
        sz_u32_t ogham_bits = x_e1_bits & (x_9a_bits >> 1) & (byte_80_bits >> 2);
        sz_u32_t x_80_ge_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpge(window, byte_80_vec));
        sz_u32_t x_8d_le_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmple(window, x_8d_vec));
        sz_u32_t range_e280_bits = lead_e280_bits & (x_80_ge_bits >> 2) & (x_8d_le_bits >> 2);
        sz_u32_t x_af_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_af_vec));
        sz_u32_t nnbsp_bits = lead_e280_bits & (x_af_bits >> 2);
        sz_u32_t x_81_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_81_vec));
        sz_u32_t x_9f_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_9f_vec));
        sz_u32_t mmsp_bits = lead_e2_bits & (x_81_bits >> 1) & (x_9f_bits >> 2);
        sz_u32_t x_a8_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_a8_vec));
        sz_u32_t x_a9_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_a9_vec));
        sz_u32_t line_bits = lead_e280_bits & (x_a8_bits >> 2);
        sz_u32_t para_bits = lead_e280_bits & (x_a9_bits >> 2);
        sz_u32_t x_e3_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_e3_vec));
        sz_u32_t ideographic_bits = x_e3_bits & (byte_80_bits >> 1) & (byte_80_bits >> 2);
        sz_u32_t three_byte_starts = ogham_bits | range_e280_bits | nnbsp_bits | mmsp_bits | line_bits | para_bits |
                                     ideographic_bits;

        sz_u32_t start_bits = (one_byte_bits | two_byte_starts | three_byte_starts) & (sz_u32_t)0x3FFF;

        sz_size_t const window_matches = (sz_size_t)sz_u32_popcount(start_bits);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_window_powervsx_(start_bits, two_byte_starts, three_byte_starts, emit_count, position,
                                                  match_offsets + count, match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) { // output buffer full: resume past the last emitted match.
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += 14;
    }

    count += sz_utf8_find_whitespaces_serial_((sz_cptr_t)(text_u8 + position), length - position, position,
                                              match_offsets + count, match_lengths + count, matches_capacity - count,
                                              bytes_consumed);
    return count;
}

#pragma endregion Multistep newline / whitespace iteration

/**
 *  @brief UAX-29 word boundary detection using IBM Power VSX (forward & reverse).
 *
 *  The full UAX-29 segmenter is a stateful walk whose decision at each candidate position depends on the
 *  Word_Break properties of the surrounding runes, including multi-code-point look-around (WB6/WB7 mid-letter,
 *  WB11/WB12 numeric, WB15/WB16 Regional_Indicator parity) and WB4 Extend/Format/ZWJ skipping. We keep those
 *  stateful sub-rules in the serial reference, but accelerate the dominant common case with VSX: long runs of
 *  ASCII letters (`[A-Za-z]`) or ASCII digits (`[0-9]`).
 *
 *  Interior positions of a maximal ASCII-letter run are governed @b unconditionally by WB5 (AHLetter ×
 *  AHLetter ⇒ no break, no look-around needed), and interior positions of an ASCII-digit run by WB8 (Numeric ×
 *  Numeric ⇒ no break). Such positions can never be a boundary, so we skip them at vector speed. The first
 *  position whose byte class differs from its predecessor (or is not letter/digit) is the earliest position
 *  that could be a boundary; from there the serial reference re-tests that exact position. Because we only ever
 *  skip @b proven non-boundaries, the result is byte-for-byte identical to `_serial` (same returned pointer and
 *  `boundary_width`).
 *
 *  Per-byte classification (0 = ASCII letter, 1 = ASCII digit, 2 = other ASCII, 3 = non-ASCII) is computed
 *  in-register with VSX range compares and `vec_sel`. We store the 16-byte class vector and fold it against
 *  the previous byte's class to form a "safe interior" bitmask, then scan that mask with a trailing/leading
 *  zero count to locate the first/last candidate boundary.
 */

/*  x86-`movemask`-equivalent for VSX: gathers the MSB of each of the 16 bytes into bit `i` (lowest-addressed
 *  byte -> bit 0) via `vec_vbpermq`. Named distinctly from `find/powervsx.h`'s identical helper so both can
 *  coexist in one translation unit. */
SZ_INTERNAL sz_u64_t sz_utf8_movemask_powervsx_(__vector unsigned char compared) {
    __vector unsigned char const indices = {120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0};
    __vector unsigned long long const gathered = vec_vbpermq(compared, indices);
#if SZ_IS_BIG_ENDIAN_
    return (sz_u64_t)gathered[0] & 0xFFFFull;
#else
    return (sz_u64_t)gathered[1] & 0xFFFFull;
#endif
}

/*  Boundary mask for the trusted lanes [2,14] of an all-ASCII 16-byte window. Each Word_Break class is a few
 *  VSX range/equality compares; the eight per-class lane bitmasks are extracted with the single-instruction
 *  `vec_vbpermq` movemask and fed to the shared portable join routine - matching the Haswell/v128/LASX path
 *  (extracting to the wide integer ALU measured faster than evaluating the rules in-vector). */
SZ_INTERNAL sz_u32_t sz_utf8_word_break_boundary_mask_powervsx_(__vector unsigned char bytes_vec) {
    __vector unsigned char lowered = vec_or(bytes_vec, vec_splats((unsigned char)0x20));
    __vector unsigned char is_aletter = vec_and(
        (__vector unsigned char)vec_cmpge(lowered, vec_splats((unsigned char)0x61)),
        (__vector unsigned char)vec_cmple(lowered, vec_splats((unsigned char)0x7A)));
    __vector unsigned char is_numeric = vec_and(
        (__vector unsigned char)vec_cmpge(bytes_vec, vec_splats((unsigned char)0x30)),
        (__vector unsigned char)vec_cmple(bytes_vec, vec_splats((unsigned char)0x39)));
    __vector unsigned char is_extendnumlet = (__vector unsigned char)vec_cmpeq(bytes_vec,
                                                                               vec_splats((unsigned char)0x5F));
    __vector unsigned char is_midletter = (__vector unsigned char)vec_cmpeq(bytes_vec, vec_splats((unsigned char)0x3A));
    __vector unsigned char is_midnum = vec_or(
        (__vector unsigned char)vec_cmpeq(bytes_vec, vec_splats((unsigned char)0x2C)),
        (__vector unsigned char)vec_cmpeq(bytes_vec, vec_splats((unsigned char)0x3B)));
    __vector unsigned char is_mid_quotes = vec_or(
        vec_or((__vector unsigned char)vec_cmpeq(bytes_vec, vec_splats((unsigned char)0x22)),
               (__vector unsigned char)vec_cmpeq(bytes_vec, vec_splats((unsigned char)0x27))),
        (__vector unsigned char)vec_cmpeq(bytes_vec, vec_splats((unsigned char)0x2E)));
    __vector unsigned char is_cr = (__vector unsigned char)vec_cmpeq(bytes_vec, vec_splats((unsigned char)0x0D));
    __vector unsigned char is_lf = (__vector unsigned char)vec_cmpeq(bytes_vec, vec_splats((unsigned char)0x0A));
    sz_u64_t join = sz_utf8_word_break_join_from_class_masks_(
        sz_utf8_movemask_powervsx_(is_aletter), sz_utf8_movemask_powervsx_(is_numeric),
        sz_utf8_movemask_powervsx_(is_extendnumlet), sz_utf8_movemask_powervsx_(is_midletter),
        sz_utf8_movemask_powervsx_(is_midnum), sz_utf8_movemask_powervsx_(is_mid_quotes),
        sz_utf8_movemask_powervsx_(is_cr), sz_utf8_movemask_powervsx_(is_lf));
    return (sz_u32_t)((~join) & 0x7FFCu); // trusted lanes [2,14]
}

SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_powervsx( //
    sz_cptr_t text, sz_size_t length,                      //
    sz_size_t *word_starts, sz_size_t *word_lengths,       //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t word_start = 0; // Start of the word currently being accumulated (always a boundary).
    // Skip the first codepoint (position 0 is always a boundary, WB1).
    sz_size_t position = sz_utf8_codepoint_length_(text_u8[0]);

    // Byte-permutation rows compacting a 2-lane sub-block's set `u64` boundary positions to the front of a
    // `vector unsigned long long` via `vec_perm` (a VSX vector holds two `u64`): row `[m]` in ascending lane
    // order, indexed by the dense 2-bit submask. Mirrors the newline peel's `compact2_lut`.
    static unsigned char const compact2_lut[4][16] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    };

    // Oracle-free fast path: an all-ASCII window [position-2, position+14) resolves boundaries at positions
    // [position, position+12]; one fixed sub-block loop compacts each 2-lane group and emits it as a
    // shifted-difference, carrying the open `word_start` into lane 0 and the group's first boundary into lane 1.
    while (position < length) {
        int ascii_window = position >= 2 && position + 14 <= length;
        __vector unsigned char window = vec_splats((unsigned char)0);
        if (ascii_window) {
            window = vec_xl(0, text_u8 + position - 2); // lane j = byte position-2+j
            ascii_window = !vec_any_ge(window, vec_splats((unsigned char)0x80));
        }
        if (!ascii_window) { // Non-ASCII window or near the edges: one scalar codepoint step.
            if (sz_utf8_is_word_boundary_serial(text, length, position)) {
                if (words == words_capacity) {
                    if (bytes_consumed) *bytes_consumed = word_start;
                    return words;
                }
                word_starts[words] = word_start, word_lengths[words] = position - word_start, ++words;
                word_start = position;
            }
            position += sz_utf8_codepoint_length_(text_u8[position]);
            continue;
        }

        sz_u32_t boundary = sz_utf8_word_break_boundary_mask_powervsx_(window); // trusted lanes [2,14]
        for (sz_size_t sub_block = 0; sub_block < 8; ++sub_block) {
            sz_u32_t const submask = (boundary >> (sub_block * 2)) & 0x3u;
            if (!submask) continue;
            sz_size_t const taken = (sz_size_t)sz_u32_popcount(submask);
            sz_size_t const stored = sz_min_of_two(taken, words_capacity - words);

            sz_size_t const base = position - 2 + sub_block * 2; // lane k of this sub-block = byte base+k
            __vector unsigned long long const positions = {(unsigned long long)base, (unsigned long long)(base + 1)};
            __vector unsigned char const permutation = vec_xl(0, compact2_lut[submask]);
            __vector unsigned long long const boundaries = (__vector unsigned long long)vec_perm(
                (__vector unsigned char)positions, (__vector unsigned char)positions, permutation);
            __vector unsigned long long const starts = {(unsigned long long)word_start, boundaries[0]};
            __vector unsigned long long const lengths = vec_sub(boundaries, starts);

            sz_size_t scratch_starts[2], scratch_lengths[2];
            vec_xst(starts, 0, (unsigned long long *)scratch_starts);
            vec_xst(lengths, 0, (unsigned long long *)scratch_lengths);
            for (sz_size_t i = 0; i < stored; ++i)
                word_starts[words + i] = scratch_starts[i], word_lengths[words + i] = scratch_lengths[i];
            words += stored;
            if (stored) word_start = word_starts[words - 1] + word_lengths[words - 1];
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_start;
                return words;
            }
        }
        position += 13; // Resolved [position, position+12]; next unresolved boundary is at position+13.
    }

    if (words == words_capacity) {
        if (bytes_consumed) *bytes_consumed = word_start;
        return words;
    }
    word_starts[words] = word_start;
    word_lengths[words] = length - word_start;
    ++words;
    if (bytes_consumed) *bytes_consumed = length;
    return words;
}

SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_powervsx( //
    sz_cptr_t text, sz_size_t length,                       //
    sz_size_t *word_starts, sz_size_t *word_lengths,        //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = length;
        return 0;
    }
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t word_end = length; // End of the word currently being accumulated (always a boundary).
    // Move back one codepoint from the end (position length is always a boundary, WB2).
    sz_size_t position = length - 1;
    while (position > 0 && (text_u8[position] & 0xC0) == 0x80) position--;

    // Descending counterpart of the forward `compact2_lut`: row `[m]` gathers a 2-lane sub-block's set `u64`
    // boundary positions to the front in HIGH-to-LOW lane order. For submask `0b11` lane 1 is emitted first
    // (its bytes [8,16) lead), then lane 0; single-lane submasks coincide with the ascending table.
    static unsigned char const compact2_lut_descending[4][16] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15},
        {8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7},
    };

    // Oracle-free fast path: an all-ASCII window [position-14, position+2) resolves boundaries at positions
    // [position-12, position]; one fixed sub-block loop walks high-to-low, compacting each 2-lane group in
    // descending lane order and emitting it as a shifted-difference (lane 0 carries the open `word_end`).
    while (position > 0) {
        sz_size_t base = position - 14; // lane j = byte base+j; lane 14 = byte position, trusted lanes [2,14]
        int ascii_window = position >= 14 && position + 2 <= length;
        __vector unsigned char window = vec_splats((unsigned char)0);
        if (ascii_window) {
            window = vec_xl(0, text_u8 + base);
            ascii_window = !vec_any_ge(window, vec_splats((unsigned char)0x80));
        }
        if (!ascii_window) { // Non-ASCII window or near the edges: one scalar codepoint step.
            if (sz_utf8_is_word_boundary_serial(text, length, position)) {
                if (words == words_capacity) {
                    if (bytes_consumed) *bytes_consumed = word_end;
                    return words;
                }
                word_starts[words] = position, word_lengths[words] = word_end - position, ++words;
                word_end = position;
            }
            position--;
            while (position > 0 && (text_u8[position] & 0xC0) == 0x80) position--;
            continue;
        }

        sz_u32_t boundary = sz_utf8_word_break_boundary_mask_powervsx_(window); // trusted lanes [2,14]
        for (sz_size_t sub_block = 8; sub_block-- > 0;) {                       // high-to-low for descending emission
            sz_u32_t const submask = (boundary >> (sub_block * 2)) & 0x3u;
            if (!submask) continue;
            sz_size_t const taken = (sz_size_t)sz_u32_popcount(submask);
            sz_size_t const stored = sz_min_of_two(taken, words_capacity - words);

            sz_size_t const group_base = base + sub_block * 2; // lane k of this sub-block = byte group_base+k
            __vector unsigned long long const positions = {(unsigned long long)group_base,
                                                           (unsigned long long)(group_base + 1)};
            __vector unsigned char const permutation = vec_xl(0, compact2_lut_descending[submask]);
            __vector unsigned long long const boundaries = (__vector unsigned long long)vec_perm(
                (__vector unsigned char)positions, (__vector unsigned char)positions, permutation);
            __vector unsigned long long const previous = {(unsigned long long)word_end, boundaries[0]};
            __vector unsigned long long const lengths = vec_sub(previous, boundaries);

            sz_size_t scratch_starts[2], scratch_lengths[2];
            vec_xst(boundaries, 0, (unsigned long long *)scratch_starts);
            vec_xst(lengths, 0, (unsigned long long *)scratch_lengths);
            for (sz_size_t i = 0; i < stored; ++i)
                word_starts[words + i] = scratch_starts[i], word_lengths[words + i] = scratch_lengths[i];
            words += stored;
            if (stored) word_end = word_starts[words - 1];
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_end;
                return words;
            }
        }
        position = base + 1; // Resolved down to position-12; next unresolved boundary is at position-13.
    }

    if (words == words_capacity) {
        if (bytes_consumed) *bytes_consumed = word_end;
        return words;
    }
    word_starts[words] = 0;
    word_lengths[words] = word_end;
    ++words;
    if (bytes_consumed) *bytes_consumed = 0;
    return words;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_POWERVSX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_ITERATE_POWERVSX_H_
