/**
 *  @brief NEON (AArch64) backend for UTF-8 delimiter scanning.
 *  @file include/stringzilla/utf8_delimiters/neon.h
 *  @author Ash Vardanian
 *
 *  Fully-vectorized, gather-free twin of @ref sz_find_delimiters_utf8_serial, the NEON port of the Ice Lake reference.
 *  Each 64-byte window is decoded once through the shared codepoint substrate (`sz_utf8_codepoints_decode_window_neon_`),
 *  and EVERY codepoint-start lane's delimiter membership is resolved IN-REGISTER over the decoded `high`/`low` halves:
 *  the BMP block id via a `vqtbl4q_u8` 256-entry lookup, the 32-byte bitmap byte via a transposed-column `vqtbl1q_u8`
 *  cascade (NEON has no `vpermi2b` page network, so the bitmaps are read column-major: `column_{low>>3}[block_id]`),
 *  and the per-lane bit test by a `vqtbl1q_u8` bit-mask. The 4-byte (astral) lanes reconstruct the full 21-bit
 *  codepoint and walk the small astral L1/L2 network the same gather-free way; the whole astral block is gated on a
 *  4-byte lead being present. Per-lane validity (overlong / surrogate / out-of-range / bad-continuation, mirroring
 *  `sz_rune_parse`) is computed as a mask and intersected with membership, so a malformed lead is never reported and
 *  the result is bit-identical to the serial oracle for the returned pointer and matched length. There is NO scalar
 *  per-codepoint membership and NO ASCII fast path with a scalar tail; no vector-indexed gather is ever issued.
 */
#ifndef STRINGZILLA_UTF8_DELIMITERS_NEON_H_
#define STRINGZILLA_UTF8_DELIMITERS_NEON_H_

#include "stringzilla/types.h"

#include "stringzilla/utf8_codepoints/neon.h"
#include "stringzilla/utf8_delimiters/serial.h"
#include "stringzilla/utf8_delimiters/tables.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

#pragma region Gather free membership

/** @brief  Per-lane single-bit test `(bitmap_byte >> (low & 7)) & 1` for one quarter, returned as 0x00/0xFF lanes. */
SZ_INTERNAL uint8x16_t sz_delimiter_test_bit_neon_(uint8x16_t bitmap_byte, uint8x16_t low) {
    static sz_u8_t const bit_for_low3[16] = {1, 2, 4, 8, 16, 32, 64, 128, 0, 0, 0, 0, 0, 0, 0, 0};
    uint8x16_t const bit_table = vld1q_u8(bit_for_low3);
    uint8x16_t const bit_mask = vqtbl1q_u8(bit_table, vandq_u8(low, vdupq_n_u8(0x07)));
    return vtstq_u8(bitmap_byte, bit_mask);
}

/**
 *  @brief  Read one bitmap byte per lane from a transposed-column table: `bitmap_byte = column_{low>>3}[block_id]`.
 *
 *  NEON has no `vpermi2b` page network, so the 32-byte bitmaps are stored column-major in @p columns (column `c`, 64
 *  ids of `bitmaps[id*32 + c]`, at `columns + c*64`). For each of the 32 columns the 64-entry column is read at
 *  @p block_id via a 4-row `vqtbl1q_u8` cascade, then the lane keeps the candidate whose column index matches its
 *  `(low >> 3)`. Gather-free — only `vld1q`/`vqtbl1q`/`vceqq`/`vbslq`. @p block_id / @p low address one quarter.
 */
SZ_INTERNAL uint8x16_t sz_delimiter_bitmap_byte_neon_(sz_u8_t const *columns, uint8x16_t block_id, uint8x16_t low) {
    uint8x16_t const within = vandq_u8(block_id, vdupq_n_u8(0x0F));
    uint8x16_t const selector = vshrq_n_u8(block_id, 4);
    uint8x16_t const nibble = vshrq_n_u8(low, 3); // (low >> 3) in [0, 32): the column index
    uint8x16_t bitmap_byte = vdupq_n_u8(0);
    for (int column = 0; column < 32; ++column) {
        uint8x16_t const candidate = sz_utf8_codepoints_cascade_stage_neon_(columns + column * 64, 4, selector, within);
        uint8x16_t const here = vceqq_u8(nibble, vdupq_n_u8((sz_u8_t)column));
        bitmap_byte = vbslq_u8(here, candidate, bitmap_byte);
    }
    return bitmap_byte;
}

/**
 *  @brief  BMP (codepoint < 0x10000) delimiter membership for one quarter, gather-free; returns 0x00/0xFF lanes.
 *
 *  The decode window only reconstructs `high`/`low` for 2-/3-byte leads; ASCII lanes (top bit clear) carry their
 *  codepoint in the raw byte itself, so override them with (high=0, low=byte) before addressing the BMP tables.
 *  `high` (cp >> 8, in [0,256)) selects a 32-byte bitmap row id through the aligned 256-entry `bmp_block` table via
 *  `vqtbl4q_u8`; the bitmap byte at `row_id*32 + (low >> 3)` is read column-major; the bit `(low & 7)` is tested.
 */
SZ_INTERNAL uint8x16_t sz_delimiter_bmp_membership_neon_(uint8x16_t window, uint8x16_t high_in, uint8x16_t low_in) {
    uint8x16_t const is_ascii = vcltq_u8(window, vdupq_n_u8(0x80));
    uint8x16_t const high = vbicq_u8(high_in, is_ascii);       // high = is_ascii ? 0 : high_in
    uint8x16_t const low = vbslq_u8(is_ascii, window, low_in); // low  = is_ascii ? byte : low_in
    uint8x16_t const block_id = sz_utf8_codepoints_lut256_neon_(sz_utf8_delimiter_bmp_block_, high);
    uint8x16_t const bitmap_byte = sz_delimiter_bitmap_byte_neon_(sz_utf8_delimiter_bmp_bitmaps_columns_, block_id,
                                                                  low);
    return sz_delimiter_test_bit_neon_(bitmap_byte, low);
}

/**
 *  @brief  Astral (codepoint >= 0x10000) delimiter membership for one quarter, gather-free; returns 0x00/0xFF lanes.
 *
 *  Reconstructs the byte-domain components of `offset = cp - 0x10000` directly per lane (no 32-bit-lane widening):
 *  with `cp = b0<<18 | b1<<12 | b2<<6 | b3` (b0 = lead & 7, b1..b3 = continuation & 0x3F) and `offset = cp - 0x10000`,
 *  the parts are `super = (cp >> 16) - 1` (no borrow: subtracting exactly 1<<16), `sub = (cp >> 8) & 0xFF`,
 *  `low8 = cp & 0xFF`. The `super` (0..15) selects an L1 group; `group*256 + sub` selects a bitmap row id (group < 2,
 *  so the two 256-entry halves of the L2 table are read and blended by the group bit); the bit `(low8 & 7)` is tested.
 */
SZ_INTERNAL uint8x16_t sz_delimiter_astral_membership_neon_(uint8x16_t window, uint8x16_t next1, uint8x16_t next2,
                                                            uint8x16_t next3) {
    uint8x16_t const b0 = vandq_u8(window, vdupq_n_u8(0x07));
    uint8x16_t const b1 = vandq_u8(next1, vdupq_n_u8(0x3F));
    uint8x16_t const b2 = vandq_u8(next2, vdupq_n_u8(0x3F));
    uint8x16_t const b3 = vandq_u8(next3, vdupq_n_u8(0x3F));

    // cp >> 16 = (b0 << 2) | (b1 >> 4); super = (cp >> 16) - 1.
    uint8x16_t const cp_hi = vorrq_u8(vshlq_n_u8(b0, 2), vshrq_n_u8(b1, 4));
    uint8x16_t const super = vsubq_u8(cp_hi, vdupq_n_u8(1));
    // (cp >> 8) & 0xFF = (b1 << 4) | (b2 >> 2).
    uint8x16_t const sub = vorrq_u8(vshlq_n_u8(b1, 4), vshrq_n_u8(b2, 2));
    // cp & 0xFF = (b2 << 6) | b3.
    uint8x16_t const low8 = vorrq_u8(vshlq_n_u8(b2, 6), b3);

    // group = astral_l1[super]; super in [0, 16), so a single 16-entry cascade row addresses it.
    uint8x16_t const group = sz_utf8_codepoints_cascade_stage_neon_(sz_utf8_delimiter_astral_l1_, 1, vdupq_n_u8(0),
                                                                    super);

    // row_id = astral_l2[group*256 + sub]: read both 256-entry halves and blend by the group bit (group < 2).
    uint8x16_t const row_group0 = sz_utf8_codepoints_lut256_neon_(sz_utf8_delimiter_astral_l2_ + 0, sub);
    uint8x16_t const row_group1 = sz_utf8_codepoints_lut256_neon_(sz_utf8_delimiter_astral_l2_ + 256, sub);
    uint8x16_t const group_is_one = vceqq_u8(group, vdupq_n_u8(1));
    uint8x16_t const row_id = vbslq_u8(group_is_one, row_group1, row_group0);

    uint8x16_t const bitmap_byte = sz_delimiter_bitmap_byte_neon_(sz_utf8_delimiter_astral_bitmaps_columns_, row_id,
                                                                  low8);
    return sz_delimiter_test_bit_neon_(bitmap_byte, low8);
}

/**
 *  @brief  Per-lane UTF-8 validity for codepoint-start lanes, mirroring `sz_rune_parse` exactly: a 2/3/4-byte lead is
 *          valid only when its continuation bytes are well-formed and it is not overlong, a surrogate, or beyond
 *          U+10FFFF. Returned as one 64-bit lane mask. Span-clamping (so a lead near the loaded edge whose
 *          continuation bytes wrap is rejected) is applied by the caller via `byte_span`; here the substrate masks are
 *          already loaded-clamped. An invalid lead is never reported (serial advances one byte and re-syncs).
 */
SZ_INTERNAL sz_u64_t sz_delimiter_valid_starts_neon_(sz_utf8_codepoints_window_neon_t const *decoded,
                                                     uint8x16_t const *next1, uint8x16_t const *next2,
                                                     uint8x16_t const *next3) {
    uint8x16_t const continuation_mask = vdupq_n_u8(0xC0), continuation_pattern = vdupq_n_u8(0x80);
    uint8x16_t c1_ok_bool[4], c2_ok_bool[4], c3_ok_bool[4];
    uint8x16_t lead_ge_c2_bool[4], three_well_bool[4], four_well_bool[4], ascii_bool[4];
    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const here = decoded->window[quarter];
        uint8x16_t const n1 = next1[quarter];
        c1_ok_bool[quarter] = vceqq_u8(vandq_u8(n1, continuation_mask), continuation_pattern);
        c2_ok_bool[quarter] = vceqq_u8(vandq_u8(next2[quarter], continuation_mask), continuation_pattern);
        c3_ok_bool[quarter] = vceqq_u8(vandq_u8(next3[quarter], continuation_mask), continuation_pattern);
        ascii_bool[quarter] = vcltq_u8(here, vdupq_n_u8(0x80));

        // 2-byte: lead >= 0xC2 (reject C0/C1 overlong).
        lead_ge_c2_bool[quarter] = vcgeq_u8(here, vdupq_n_u8(0xC2));

        // 3-byte: not overlong (E0 with next1 < 0xA0); not surrogate (ED with next1 >= 0xA0).
        uint8x16_t const lead_e0 = vceqq_u8(here, vdupq_n_u8(0xE0));
        uint8x16_t const lead_ed = vceqq_u8(here, vdupq_n_u8(0xED));
        uint8x16_t const n1_lt_a0 = vcltq_u8(n1, vdupq_n_u8(0xA0));
        uint8x16_t const bad_e0 = vandq_u8(lead_e0, n1_lt_a0);
        uint8x16_t const bad_ed = vandq_u8(lead_ed, vmvnq_u8(n1_lt_a0));
        three_well_bool[quarter] = vmvnq_u8(vorrq_u8(bad_e0, bad_ed));

        // 4-byte: lead <= 0xF4; not overlong (F0 with next1 < 0x90); not > U+10FFFF (F4 with next1 >= 0x90).
        uint8x16_t const lead_le_f4 = vcleq_u8(here, vdupq_n_u8(0xF4));
        uint8x16_t const lead_f0 = vceqq_u8(here, vdupq_n_u8(0xF0));
        uint8x16_t const lead_f4 = vceqq_u8(here, vdupq_n_u8(0xF4));
        uint8x16_t const n1_lt_90 = vcltq_u8(n1, vdupq_n_u8(0x90));
        uint8x16_t const bad_f0 = vandq_u8(lead_f0, n1_lt_90);
        uint8x16_t const bad_f4 = vandq_u8(lead_f4, vmvnq_u8(n1_lt_90));
        four_well_bool[quarter] = vandq_u8(lead_le_f4, vmvnq_u8(vorrq_u8(bad_f0, bad_f4)));
    }

    sz_u64_t const c1_ok = sz_utf8_codepoints_mask_combine_neon_(c1_ok_bool[0], c1_ok_bool[1], c1_ok_bool[2],
                                                                 c1_ok_bool[3]);
    sz_u64_t const c2_ok = sz_utf8_codepoints_mask_combine_neon_(c2_ok_bool[0], c2_ok_bool[1], c2_ok_bool[2],
                                                                 c2_ok_bool[3]);
    sz_u64_t const c3_ok = sz_utf8_codepoints_mask_combine_neon_(c3_ok_bool[0], c3_ok_bool[1], c3_ok_bool[2],
                                                                 c3_ok_bool[3]);
    sz_u64_t const ascii = sz_utf8_codepoints_mask_combine_neon_(ascii_bool[0], ascii_bool[1], ascii_bool[2],
                                                                 ascii_bool[3]);
    sz_u64_t const lead_ge_c2 = sz_utf8_codepoints_mask_combine_neon_(lead_ge_c2_bool[0], lead_ge_c2_bool[1],
                                                                      lead_ge_c2_bool[2], lead_ge_c2_bool[3]);
    sz_u64_t const three_well = sz_utf8_codepoints_mask_combine_neon_(three_well_bool[0], three_well_bool[1],
                                                                      three_well_bool[2], three_well_bool[3]);
    sz_u64_t const four_well = sz_utf8_codepoints_mask_combine_neon_(four_well_bool[0], four_well_bool[1],
                                                                     four_well_bool[2], four_well_bool[3]);

    sz_u64_t const loaded = sz_utf8_codepoints_mask_until_(decoded->loaded);
    sz_u64_t const span2 = sz_utf8_codepoints_mask_until_(decoded->loaded >= 1 ? decoded->loaded - 1 : 0);
    sz_u64_t const span3 = sz_utf8_codepoints_mask_until_(decoded->loaded >= 2 ? decoded->loaded - 2 : 0);
    sz_u64_t const span4 = sz_utf8_codepoints_mask_until_(decoded->loaded >= 3 ? decoded->loaded - 3 : 0);

    sz_u64_t const two_ok = decoded->two_byte_starts & span2 & c1_ok & lead_ge_c2;
    sz_u64_t const three_ok = decoded->three_byte_starts & span3 & c1_ok & c2_ok & three_well;
    sz_u64_t const four_ok = decoded->four_byte_starts & span4 & c1_ok & c2_ok & c3_ok & four_well;
    return (ascii & loaded) | two_ok | three_ok | four_ok;
}

#pragma endregion Gather free membership

#pragma region Forward driver

/** @copydoc sz_find_delimiter_utf8 */
SZ_PUBLIC sz_cptr_t sz_find_delimiters_utf8_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    sz_u8_t const *const text_u8 = (sz_u8_t const *)text;

    sz_size_t base = 0;
    while (base < length) {
        sz_utf8_codepoints_window_neon_t const decoded = sz_utf8_codepoints_decode_window_neon_(text_u8 + base,
                                                                                                length - base);
        sz_size_t const loaded = decoded.loaded;

        uint8x16_t next1[4], next2[4], next3[4];
        sz_utf8_codepoints_forward_neighbours_neon_(decoded.window, next1, next2, next3);

        // Effective-window trim: a multi-byte lead near the 64-byte edge whose span runs past `loaded` would decode
        // against a wrapped neighbour; defer it to the next window.
        sz_size_t byte_span = loaded;
        if (loaded >= 64) {
            sz_u64_t const overrun = (decoded.two_byte_starts & ~sz_utf8_codepoints_mask_until_(loaded - 1)) |
                                     (decoded.three_byte_starts & ~sz_utf8_codepoints_mask_until_(loaded - 2)) |
                                     (decoded.four_byte_starts & ~sz_utf8_codepoints_mask_until_(loaded - 3));
            byte_span = overrun ? (sz_size_t)sz_u64_ctz(overrun) : loaded;
        }
        sz_u64_t const span_mask = sz_utf8_codepoints_mask_until_(byte_span);

        sz_u64_t const valid_starts = sz_delimiter_valid_starts_neon_(&decoded, next1, next2, next3) &
                                      decoded.codepoint_starts & span_mask;

        // BMP membership for every lane; astral membership blended onto the four-byte lanes (gated on presence).
        sz_u64_t const four_byte = decoded.four_byte_starts & span_mask;
        sz_u64_t member = 0;
        for (int quarter = 0; quarter < 4; ++quarter) {
            uint8x16_t const bmp = sz_delimiter_bmp_membership_neon_(decoded.window[quarter], decoded.high[quarter],
                                                                     decoded.low[quarter]);
            member |= sz_utf8_codepoints_movemask16_neon_(bmp) << (16 * quarter);
        }
        // Blend astral over the four-byte lanes; the per-lane BMP/astral decision stays exact on the bit masks.
        if (four_byte) {
            sz_u64_t astral_member = 0;
            for (int quarter = 0; quarter < 4; ++quarter) {
                uint8x16_t const astral = sz_delimiter_astral_membership_neon_(decoded.window[quarter], next1[quarter],
                                                                               next2[quarter], next3[quarter]);
                astral_member |= sz_utf8_codepoints_movemask16_neon_(astral) << (16 * quarter);
            }
            member = (member & ~four_byte) | (astral_member & four_byte);
        }

        sz_u64_t const hits = member & valid_starts;
        if (hits) {
            sz_size_t const lane = (sz_size_t)sz_u64_ctz(hits);
            if (matched_length) {
                sz_size_t length_at_lane = 1;
                length_at_lane += (decoded.two_byte_starts >> lane) & 1;
                length_at_lane += ((decoded.three_byte_starts >> lane) & 1) * 2;
                length_at_lane += ((decoded.four_byte_starts >> lane) & 1) * 3;
                *matched_length = length_at_lane;
            }
            return (sz_cptr_t)(text_u8 + base + lane);
        }
        base += byte_span ? byte_span : 1;
    }

    if (matched_length) *matched_length = 0;
    return SZ_NULL_CHAR;
}

#pragma endregion Forward driver

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEON

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_DELIMITERS_NEON_H_
