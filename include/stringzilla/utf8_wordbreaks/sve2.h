/**
 *  @brief SVE2 backend for UAX-29 word boundaries.
 *  @file include/stringzilla/utf8_wordbreaks/sve2.h
 *  @author Ash Vardanian
 *
 *  Full-width SVE2 UAX-29 word-break kernel `sz_utf8_wordbreaks_sve2`, assembled from three validated leaves
 *  (in-register classifier + decode/classify + rule engine) plus the `build_frame` lowering leaf, the
 *  single-`svcompact_u32` drain, and the advancing driver. The SVE2 twin of the AVX2 (Haswell) / Ice Lake /
 *  NEON kernels: no path scalar-walks codepoints, spills a vector to call the serial oracle, or issues a gather.
 *  Validated bit-exact vs `sz_utf8_wordbreaks_serial` (streaming-cursor: clamped capacity + `bytes_consumed`
 *  resume) over the regression cases, a capacity x alignment sweep, and >=2,000,000 random valid/malformed
 *  inputs at `svcntb()==64` (so one window == one 64-bit lane-mask domain).
 *
 *  Implementation note: the engine's `<<k` / `>>1` lane algebra is realized with `svext`-based shift macros
 *  (`sz_utf8_word_break_lane_up_sve2_(v, k)` / `sz_utf8_word_break_lane_dn_sve2_(v, k)`) so the shift amount
 *  stays an integer-constant expression and the whole file compiles as both C99 and C++, matching the
 *  sibling kernels.
 */
#ifndef STRINGZILLA_UTF8_WORDBREAKS_SVE2_H_
#define STRINGZILLA_UTF8_WORDBREAKS_SVE2_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_wordbreaks/tables.h"
#include "stringzilla/utf8_wordbreaks/serial.h"
#include "stringzilla/utf8_runes/sve2.h"

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

/** @brief In-register SVE2 Word_Break classifier core (nibble-cascade trie + predicate/lane-mask bridge). */

/** @brief  Read up to 256 LUT entries by per-lane u8 index via overlapping `svtbl_u8` chunks (gather-free). */
SZ_HELPER_AUTO svuint8_t sz_utf8_rune_lut_sve2_(sz_u8_t const *table, int count, svuint8_t index_u8x) {
    svbool_t const pg_b8x = svptrue_b8();
    int const vl = (int)svcntb();
    svuint8_t result_u8x = svdup_n_u8(0);
    for (int base = 0; base < count; base += vl) {
        int const here = (count - base) < vl ? (count - base) : vl;
        svbool_t const load_b8x = svwhilelt_b8_u64(0, (sz_u64_t)here);
        svuint8_t const chunk_u8x = svld1_u8(load_b8x, table + base);
        svuint8_t const idx_u8x = svsub_n_u8_x(pg_b8x, index_u8x, (sz_u8_t)base);
        result_u8x = svorr_u8_x(pg_b8x, result_u8x, svtbl_u8(chunk_u8x, idx_u8x));
    }
    return result_u8x;
}

/** @brief  Select one of `tile_count` 16-entry rows by `selector` and index it by `within` (nibble cascade tile). */
SZ_HELPER_AUTO svuint8_t sz_utf8_rune_cascade_sve2_(sz_u8_t const *table, int tile_count, svuint8_t selector_u8x,
                                                    svuint8_t within_u8x) {
    svbool_t const pg_b8x = svptrue_b8();
    svbool_t const row_b8x = svwhilelt_b8_u64(0, 16);
    svuint8_t result_u8x = svdup_n_u8(0);
    for (int tile = 0; tile < tile_count; ++tile) {
        svuint8_t const picked_u8x = svtbl_u8(svld1_u8(row_b8x, table + tile * 16), within_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(pg_b8x, selector_u8x, (sz_u8_t)tile), picked_u8x, result_u8x);
    }
    return result_u8x;
}

/** @brief  Word_Break class byte for sixteen-bit-wide BMP codepoints (per-lane high = cp>>8, low = cp&0xFF). */
SZ_HELPER_AUTO svuint8_t sz_utf8_word_break_bmp_class_sve2_(svuint8_t high_u8x, svuint8_t low_u8x) {
    svbool_t const pg_b8x = svptrue_b8();
    svuint8_t const page_u8x = sz_utf8_rune_lut_sve2_(sz_utf8_word_break_haswell_stage1_, 256, high_u8x);
    svuint8_t const low_high_u8x = svand_n_u8_x(pg_b8x, svlsr_n_u8_x(pg_b8x, low_u8x, 4), 0x0F);
    svuint8_t const leaf_lo_u8x = sz_utf8_rune_cascade_sve2_(sz_utf8_word_break_haswell_stage2_lo_,
                                                             sz_utf8_word_break_haswell_stage2_lo_count_k / 16,
                                                             page_u8x, low_high_u8x);
    svuint8_t const leaf_hi_u8x = sz_utf8_rune_cascade_sve2_(sz_utf8_word_break_haswell_stage2_hi_,
                                                             sz_utf8_word_break_haswell_stage2_hi_count_k / 16,
                                                             page_u8x, low_high_u8x);
    svuint8_t const leaf_group_u8x = svorr_u8_x(
        pg_b8x, svand_n_u8_x(pg_b8x, svlsr_n_u8_x(pg_b8x, leaf_lo_u8x, 4), 0x0F), svlsl_n_u8_x(pg_b8x, leaf_hi_u8x, 4));
    svuint8_t const leaf_low_nibble_u8x = svand_n_u8_x(pg_b8x, leaf_lo_u8x, 0x0F);
    svuint8_t const low_low_u8x = svand_n_u8_x(pg_b8x, low_u8x, 0x0F);
    svuint8_t const lut_index_u8x = svorr_u8_x(pg_b8x, svlsl_n_u8_x(pg_b8x, leaf_low_nibble_u8x, 4), low_low_u8x);
    svuint8_t result_u8x = svdup_n_u8(0);
    for (int group = 0; group < (int)sz_utf8_word_break_haswell_leaf_groups_k; ++group) {
        svuint8_t const value_u8x = sz_utf8_rune_lut_sve2_(sz_utf8_word_break_haswell_stage3_groups_ + group * 256, 256,
                                                           lut_index_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(pg_b8x, leaf_group_u8x, (sz_u8_t)group), value_u8x, result_u8x);
    }
    return result_u8x;
}

/** @brief  Word_Break class byte for sixteen ASTRAL codepoints over the 20-bit offset = cp - 0x10000. */
SZ_HELPER_AUTO svuint8_t sz_utf8_word_break_astral_class_sve2_(svuint8_t plane_off_u8x, svuint8_t high_u8x,
                                                               svuint8_t low_u8x) {
    svbool_t const pg_b8x = svptrue_b8();
    svuint8_t const n4_u8x = svand_n_u8_x(pg_b8x, plane_off_u8x, 0x0F);
    svuint8_t const n3_u8x = svand_n_u8_x(pg_b8x, svlsr_n_u8_x(pg_b8x, high_u8x, 4), 0x0F);
    svuint8_t const stage1_index_u8x = svorr_u8_x(pg_b8x, svlsl_n_u8_x(pg_b8x, n4_u8x, 4), n3_u8x);
    svuint8_t const page_u8x = sz_utf8_rune_lut_sve2_(sz_utf8_word_break_haswell_astral_stage1_, 256, stage1_index_u8x);
    svuint8_t const n2_u8x = svand_n_u8_x(pg_b8x, high_u8x, 0x0F);
    svuint8_t const leaf2_u8x = sz_utf8_rune_cascade_sve2_(sz_utf8_word_break_haswell_astral_stage2_lo_,
                                                           sz_utf8_word_break_haswell_astral_stage2_lo_count_k / 16,
                                                           page_u8x, n2_u8x);
    svuint8_t const n1_u8x = svand_n_u8_x(pg_b8x, svlsr_n_u8_x(pg_b8x, low_u8x, 4), 0x0F);
    svuint8_t const leaf_lo_u8x = sz_utf8_rune_cascade_sve2_(sz_utf8_word_break_haswell_astral_stage3_lo_,
                                                             sz_utf8_word_break_haswell_astral_stage3_lo_count_k / 16,
                                                             leaf2_u8x, n1_u8x);
    svuint8_t const leaf_hi_u8x = sz_utf8_rune_cascade_sve2_(sz_utf8_word_break_haswell_astral_stage3_hi_,
                                                             sz_utf8_word_break_haswell_astral_stage3_hi_count_k / 16,
                                                             leaf2_u8x, n1_u8x);
    svuint8_t const n0_u8x = svand_n_u8_x(pg_b8x, low_u8x, 0x0F);
    svuint8_t const leaf_group_u8x = svorr_u8_x(
        pg_b8x, svand_n_u8_x(pg_b8x, svlsr_n_u8_x(pg_b8x, leaf_lo_u8x, 4), 0x0F), svlsl_n_u8_x(pg_b8x, leaf_hi_u8x, 4));
    svuint8_t const leaf_low_nibble_u8x = svand_n_u8_x(pg_b8x, leaf_lo_u8x, 0x0F);
    svuint8_t const stage4_index_u8x = svorr_u8_x(pg_b8x, svlsl_n_u8_x(pg_b8x, leaf_low_nibble_u8x, 4), n0_u8x);
    svuint8_t result_u8x = svdup_n_u8(0);
    for (int group = 0; group < (int)sz_utf8_word_break_haswell_astral_leaf_groups_k; ++group) {
        svuint8_t const value_u8x = sz_utf8_rune_lut_sve2_(
            sz_utf8_word_break_haswell_astral_stage4_groups_ + group * 256, 256, stage4_index_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(pg_b8x, leaf_group_u8x, (sz_u8_t)group), value_u8x, result_u8x);
    }
    return result_u8x;
}

/** @brief  Predicate -> 64-bit lane mask bridge (`svcntb()==64` => one window == one 64-bit mask domain). */
SZ_HELPER_AUTO sz_u64_t sz_utf8_pred_to_u64_sve2_(svbool_t p_b8x, sz_size_t loaded) {
    sz_u8_t buf[256];
    svbool_t const loaded_b8x = svwhilelt_b8_u64(0, (sz_u64_t)loaded);
    svst1_u8(loaded_b8x, buf, svdup_u8_z(p_b8x, 1));
    sz_u64_t mask = 0;
    for (sz_size_t i = 0; i < loaded; ++i) mask |= (sz_u64_t)buf[i] << i;
    return mask;
}

/** @brief  64-bit lane mask -> predicate bridge, the inverse of @ref sz_utf8_pred_to_u64_sve2_. */
SZ_HELPER_AUTO svbool_t sz_utf8_u64_to_pred_sve2_(sz_u64_t mask, sz_size_t loaded) {
    sz_u8_t buf[256];
    for (sz_size_t i = 0; i < loaded; ++i) buf[i] = (sz_u8_t)((mask >> i) & 1u);
    svbool_t const loaded_b8x = svwhilelt_b8_u64(0, (sz_u64_t)loaded);
    return svcmpne_n_u8(loaded_b8x, svld1_u8(loaded_b8x, buf), 0);
}

/** @brief Decode + classify one window. */
typedef struct sz_utf8_word_window_sve2_t {
    sz_utf8_word_break_partition_t partition;
    sz_u64_t four_byte_starts;
    sz_size_t loaded;
} sz_utf8_word_window_sve2_t;

/** @brief  Decode one window into per-lane (high, low, plane_off) substrate bytes + the codepoint partition. */
SZ_HELPER_AUTO sz_utf8_word_window_sve2_t sz_utf8_word_decode_window_sve2_( //
    sz_u8_t const *text, sz_size_t available, int at_end_of_text,           //
    sz_u8_t *high_out, sz_u8_t *low_out, sz_u8_t *plane_off_out) {

    svbool_t const pg_b8x = svptrue_b8();
    sz_size_t const tile = svcntb() < 64 ? (sz_size_t)svcntb() : 64;
    sz_size_t const loaded = available < tile ? available : tile;
    sz_u64_t const valid = sz_u64_mask_until_serial_(loaded);
    svbool_t const loaded_b8x = svwhilelt_b8_u64(0, (sz_u64_t)loaded);

    svuint8_t const zeros_u8x = svdup_n_u8(0);
    svuint8_t const bytes_u8x = svld1_u8(loaded_b8x, text);
    svuint8_t const next1_u8x = svext_u8(bytes_u8x, zeros_u8x, 1);
    svuint8_t const next2_u8x = svext_u8(bytes_u8x, zeros_u8x, 2);
    svuint8_t const next3_u8x = svext_u8(bytes_u8x, zeros_u8x, 3);

    svbool_t const continuation_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xC0), 0x80);
    svbool_t const two_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xE0), 0xC0);
    svbool_t const three_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xF0), 0xE0);
    svbool_t const four_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xF8), 0xF0);

    sz_u64_t const continuation = sz_utf8_pred_to_u64_sve2_(continuation_b8x, loaded) & valid;
    sz_u64_t const length_two = sz_utf8_pred_to_u64_sve2_(two_b8x, loaded) & valid;
    sz_u64_t const length_three = sz_utf8_pred_to_u64_sve2_(three_b8x, loaded) & valid;
    sz_u64_t const four_strict = sz_utf8_pred_to_u64_sve2_(four_b8x, loaded) & valid;
    sz_u64_t const lead_ge_f8 = sz_utf8_pred_to_u64_sve2_(svcmpge_n_u8(loaded_b8x, bytes_u8x, 0xF8), loaded) & valid;
    sz_u64_t const length_four = (four_strict | lead_ge_f8) & valid;

    sz_u64_t const next1_ge_a0 = sz_utf8_pred_to_u64_sve2_(svcmpge_n_u8(loaded_b8x, next1_u8x, 0xA0), loaded);
    sz_u64_t const next1_ge_90 = sz_utf8_pred_to_u64_sve2_(svcmpge_n_u8(loaded_b8x, next1_u8x, 0x90), loaded);
    sz_u64_t const lead_c0 = sz_utf8_pred_to_u64_sve2_(svcmpeq_n_u8(loaded_b8x, bytes_u8x, 0xC0), loaded) & valid;
    sz_u64_t const lead_c1 = sz_utf8_pred_to_u64_sve2_(svcmpeq_n_u8(loaded_b8x, bytes_u8x, 0xC1), loaded) & valid;
    sz_u64_t const lead_e0 = sz_utf8_pred_to_u64_sve2_(svcmpeq_n_u8(loaded_b8x, bytes_u8x, 0xE0), loaded) & valid;
    sz_u64_t const lead_ed = sz_utf8_pred_to_u64_sve2_(svcmpeq_n_u8(loaded_b8x, bytes_u8x, 0xED), loaded) & valid;
    sz_u64_t const lead_f0 = sz_utf8_pred_to_u64_sve2_(svcmpeq_n_u8(loaded_b8x, bytes_u8x, 0xF0), loaded) & valid;
    sz_u64_t const lead_f4 = sz_utf8_pred_to_u64_sve2_(svcmpeq_n_u8(loaded_b8x, bytes_u8x, 0xF4), loaded) & valid;
    sz_u64_t const lead_f5_or_more = sz_utf8_pred_to_u64_sve2_(svcmpge_n_u8(loaded_b8x, bytes_u8x, 0xF5), loaded) &
                                     valid;
    sz_u64_t const bad_second_byte = (lead_c0 | lead_c1) | (lead_e0 & ~next1_ge_a0) | (lead_ed & next1_ge_a0) |
                                     (lead_f0 & ~next1_ge_90) | (lead_f4 & next1_ge_90) | lead_f5_or_more;

    svuint8_t const high_two_u8x = svand_n_u8_x(pg_b8x, svlsr_n_u8_x(pg_b8x, svand_n_u8_x(pg_b8x, bytes_u8x, 0x1F), 2),
                                                0x07);
    svuint8_t const low_two_u8x = svorr_u8_x(pg_b8x, svlsl_n_u8_x(pg_b8x, svand_n_u8_x(pg_b8x, bytes_u8x, 0x03), 6),
                                             svand_n_u8_x(pg_b8x, next1_u8x, 0x3F));
    svuint8_t const high_three_u8x = svorr_u8_x(pg_b8x, svlsl_n_u8_x(pg_b8x, svand_n_u8_x(pg_b8x, bytes_u8x, 0x0F), 4),
                                                svand_n_u8_x(pg_b8x, svlsr_n_u8_x(pg_b8x, next1_u8x, 2), 0x0F));
    svuint8_t const low_three_u8x = svorr_u8_x(pg_b8x, svlsl_n_u8_x(pg_b8x, svand_n_u8_x(pg_b8x, next1_u8x, 0x03), 6),
                                               svand_n_u8_x(pg_b8x, next2_u8x, 0x3F));
    svuint8_t const plane_u8x = svorr_u8_x(
        pg_b8x, svand_n_u8_x(pg_b8x, svlsl_n_u8_x(pg_b8x, svand_n_u8_x(pg_b8x, bytes_u8x, 0x07), 2), 0x1C),
        svand_n_u8_x(pg_b8x, svlsr_n_u8_x(pg_b8x, next1_u8x, 4), 0x03));
    svuint8_t const high_four_u8x = svorr_u8_x(
        pg_b8x, svand_n_u8_x(pg_b8x, svlsl_n_u8_x(pg_b8x, svand_n_u8_x(pg_b8x, next1_u8x, 0x0F), 4), 0xF0),
        svand_n_u8_x(pg_b8x, svlsr_n_u8_x(pg_b8x, next2_u8x, 2), 0x0F));
    svuint8_t const low_four_u8x = svorr_u8_x(pg_b8x, svlsl_n_u8_x(pg_b8x, svand_n_u8_x(pg_b8x, next2_u8x, 0x03), 6),
                                              svand_n_u8_x(pg_b8x, next3_u8x, 0x3F));
    svuint8_t const plane_off_u8x = svsub_n_u8_x(pg_b8x, plane_u8x, 1);

    svuint8_t high_u8x = svdup_n_u8(0);
    svuint8_t low_u8x = bytes_u8x;
    high_u8x = svsel_u8(two_b8x, high_two_u8x, high_u8x);
    low_u8x = svsel_u8(two_b8x, low_two_u8x, low_u8x);
    high_u8x = svsel_u8(three_b8x, high_three_u8x, high_u8x);
    low_u8x = svsel_u8(three_b8x, low_three_u8x, low_u8x);
    high_u8x = svsel_u8(four_b8x, high_four_u8x, high_u8x);
    low_u8x = svsel_u8(four_b8x, low_four_u8x, low_u8x);

    svst1_u8(loaded_b8x, high_out, high_u8x);
    svst1_u8(loaded_b8x, low_out, low_u8x);
    svst1_u8(loaded_b8x, plane_off_out, plane_off_u8x);

    sz_utf8_word_window_sve2_t result;
    result.partition = sz_utf8_word_break_partition_from_masks_(continuation, length_two, length_three, length_four,
                                                                bad_second_byte, valid, at_end_of_text);
    result.four_byte_starts = four_strict;
    result.loaded = loaded;
    return result;
}

/** @brief  Classify each lane of one window into its Word_Break class byte (ASCII/BMP/astral, forced-Other). */
SZ_HELPER_AUTO void sz_utf8_word_classify_window_sve2_(                //
    sz_u8_t const *high, sz_u8_t const *low, sz_u8_t const *plane_off, //
    sz_u64_t four_byte_starts, sz_u64_t forced_other, sz_size_t loaded, sz_u8_t *out) {
    svbool_t const loaded_b8x = svwhilelt_b8_u64(0, (sz_u64_t)loaded);
    svuint8_t const high_u8x = svld1_u8(loaded_b8x, high);
    svuint8_t const low_u8x = svld1_u8(loaded_b8x, low);
    svuint8_t classes_u8x = sz_utf8_word_break_bmp_class_sve2_(high_u8x, low_u8x);
    if (four_byte_starts) {
        svuint8_t const plane_off_u8x = svld1_u8(loaded_b8x, plane_off);
        svbool_t const four_b8x = sz_utf8_u64_to_pred_sve2_(four_byte_starts, loaded);
        classes_u8x = svsel_u8(four_b8x, sz_utf8_word_break_astral_class_sve2_(plane_off_u8x, high_u8x, low_u8x),
                               classes_u8x);
    }
    if (forced_other) {
        svbool_t const other_b8x = sz_utf8_u64_to_pred_sve2_(forced_other, loaded);
        classes_u8x = svsel_u8(other_b8x, svdup_n_u8((sz_u8_t)sz_utf8_word_break_other_k), classes_u8x);
    }
    svst1_u8(loaded_b8x, out, classes_u8x);
}

/** @brief Full-width SVE2 rule engine; the portable `sz_u64_t` engine's `<<k` / `>>1` lane algebra is realized on
 *         per-lane 0/1 vectors via `svext`-based lane shifts (so it scales beyond a 64-bit movemask). */
SZ_HELPER_INLINE svuint8_t sz_utf8_word_break_lane_up1_sve2_(svuint8_t a) { return svinsr_n_u8(a, 0); }
SZ_HELPER_INLINE svuint8_t sz_utf8_word_break_lane_dn1_sve2_(svuint8_t a) { return svext_u8(a, svdup_n_u8(0), 1); }
// The `svext` lane-shift amount must be an integer-constant expression, so the variable-`k` helpers are
// function-like macros rather than functions: the literal `k` at each call site reaches the intrinsic
// unchanged. Macros also keep this header a single code path for C99 and C++ — this file is compiled into
// the C99 library (`c/stringzilla/utf8_wordbreaks.c`), where the earlier `template <int K>` helpers with
// their `extern "C++"` island could not parse.
#define sz_utf8_word_break_lane_up_sve2_(v, k) svrev_u8(svext_u8(svrev_u8((v)), svdup_n_u8(0), (k)))
#define sz_utf8_word_break_lane_dn_sve2_(v, k) svext_u8((v), svdup_n_u8(0), (k))

SZ_HELPER_AUTO svuint8_t sz_utf8_word_break_fill_right_sve2_(svuint8_t seed, svuint8_t gate) {
    svbool_t const pg_b8x = svptrue_b8();
    svuint8_t bits_u8x = seed, reach_u8x = gate;
    bits_u8x = svorr_u8_x(pg_b8x, bits_u8x,
                          svand_u8_x(pg_b8x, sz_utf8_word_break_lane_up_sve2_(bits_u8x, 1), reach_u8x));
    reach_u8x = svand_u8_x(pg_b8x, reach_u8x, sz_utf8_word_break_lane_up_sve2_(reach_u8x, 1));
    bits_u8x = svorr_u8_x(pg_b8x, bits_u8x,
                          svand_u8_x(pg_b8x, sz_utf8_word_break_lane_up_sve2_(bits_u8x, 2), reach_u8x));
    reach_u8x = svand_u8_x(pg_b8x, reach_u8x, sz_utf8_word_break_lane_up_sve2_(reach_u8x, 2));
    bits_u8x = svorr_u8_x(pg_b8x, bits_u8x,
                          svand_u8_x(pg_b8x, sz_utf8_word_break_lane_up_sve2_(bits_u8x, 4), reach_u8x));
    reach_u8x = svand_u8_x(pg_b8x, reach_u8x, sz_utf8_word_break_lane_up_sve2_(reach_u8x, 4));
    bits_u8x = svorr_u8_x(pg_b8x, bits_u8x,
                          svand_u8_x(pg_b8x, sz_utf8_word_break_lane_up_sve2_(bits_u8x, 8), reach_u8x));
    reach_u8x = svand_u8_x(pg_b8x, reach_u8x, sz_utf8_word_break_lane_up_sve2_(reach_u8x, 8));
    bits_u8x = svorr_u8_x(pg_b8x, bits_u8x,
                          svand_u8_x(pg_b8x, sz_utf8_word_break_lane_up_sve2_(bits_u8x, 16), reach_u8x));
    reach_u8x = svand_u8_x(pg_b8x, reach_u8x, sz_utf8_word_break_lane_up_sve2_(reach_u8x, 16));
    bits_u8x = svorr_u8_x(pg_b8x, bits_u8x,
                          svand_u8_x(pg_b8x, sz_utf8_word_break_lane_up_sve2_(bits_u8x, 32), reach_u8x));
    return bits_u8x;
}
SZ_HELPER_AUTO svuint8_t sz_utf8_word_break_fill_left_sve2_(svuint8_t seed, svuint8_t gate) {
    svbool_t const pg_b8x = svptrue_b8();
    svuint8_t bits_u8x = seed, reach_u8x = gate;
    bits_u8x = svorr_u8_x(pg_b8x, bits_u8x,
                          svand_u8_x(pg_b8x, sz_utf8_word_break_lane_dn_sve2_(bits_u8x, 1), reach_u8x));
    reach_u8x = svand_u8_x(pg_b8x, reach_u8x, sz_utf8_word_break_lane_dn_sve2_(reach_u8x, 1));
    bits_u8x = svorr_u8_x(pg_b8x, bits_u8x,
                          svand_u8_x(pg_b8x, sz_utf8_word_break_lane_dn_sve2_(bits_u8x, 2), reach_u8x));
    reach_u8x = svand_u8_x(pg_b8x, reach_u8x, sz_utf8_word_break_lane_dn_sve2_(reach_u8x, 2));
    bits_u8x = svorr_u8_x(pg_b8x, bits_u8x,
                          svand_u8_x(pg_b8x, sz_utf8_word_break_lane_dn_sve2_(bits_u8x, 4), reach_u8x));
    reach_u8x = svand_u8_x(pg_b8x, reach_u8x, sz_utf8_word_break_lane_dn_sve2_(reach_u8x, 4));
    bits_u8x = svorr_u8_x(pg_b8x, bits_u8x,
                          svand_u8_x(pg_b8x, sz_utf8_word_break_lane_dn_sve2_(bits_u8x, 8), reach_u8x));
    reach_u8x = svand_u8_x(pg_b8x, reach_u8x, sz_utf8_word_break_lane_dn_sve2_(reach_u8x, 8));
    bits_u8x = svorr_u8_x(pg_b8x, bits_u8x,
                          svand_u8_x(pg_b8x, sz_utf8_word_break_lane_dn_sve2_(bits_u8x, 16), reach_u8x));
    reach_u8x = svand_u8_x(pg_b8x, reach_u8x, sz_utf8_word_break_lane_dn_sve2_(reach_u8x, 16));
    bits_u8x = svorr_u8_x(pg_b8x, bits_u8x,
                          svand_u8_x(pg_b8x, sz_utf8_word_break_lane_dn_sve2_(bits_u8x, 32), reach_u8x));
    return bits_u8x;
}
SZ_HELPER_AUTO svuint8_t sz_utf8_word_break_smear_right_sve2_(svuint8_t bits, svuint8_t reach) {
    svbool_t const pg_b8x = svptrue_b8();
    for (int s = 0; s < sz_utf8_word_break_smear_steps_k; ++s)
        bits = svorr_u8_x(pg_b8x, bits, svand_u8_x(pg_b8x, sz_utf8_word_break_lane_up1_sve2_(bits), reach));
    return bits;
}
SZ_HELPER_AUTO svuint8_t sz_utf8_word_break_ri_join_sve2_(svuint8_t ri, svuint8_t run_gate, int inbound_parity,
                                                          svuint8_t *inclusive_out) {
    svbool_t const pg_b8x = svptrue_b8();
    svuint8_t bits_u8x = ri, reach_u8x = run_gate;
    bits_u8x = sveor_u8_x(pg_b8x, bits_u8x,
                          svand_u8_x(pg_b8x, sz_utf8_word_break_lane_up_sve2_(bits_u8x, 1), reach_u8x));
    reach_u8x = svand_u8_x(pg_b8x, reach_u8x, sz_utf8_word_break_lane_up_sve2_(reach_u8x, 1));
    bits_u8x = sveor_u8_x(pg_b8x, bits_u8x,
                          svand_u8_x(pg_b8x, sz_utf8_word_break_lane_up_sve2_(bits_u8x, 2), reach_u8x));
    reach_u8x = svand_u8_x(pg_b8x, reach_u8x, sz_utf8_word_break_lane_up_sve2_(reach_u8x, 2));
    bits_u8x = sveor_u8_x(pg_b8x, bits_u8x,
                          svand_u8_x(pg_b8x, sz_utf8_word_break_lane_up_sve2_(bits_u8x, 4), reach_u8x));
    reach_u8x = svand_u8_x(pg_b8x, reach_u8x, sz_utf8_word_break_lane_up_sve2_(reach_u8x, 4));
    bits_u8x = sveor_u8_x(pg_b8x, bits_u8x,
                          svand_u8_x(pg_b8x, sz_utf8_word_break_lane_up_sve2_(bits_u8x, 8), reach_u8x));
    reach_u8x = svand_u8_x(pg_b8x, reach_u8x, sz_utf8_word_break_lane_up_sve2_(reach_u8x, 8));
    bits_u8x = sveor_u8_x(pg_b8x, bits_u8x,
                          svand_u8_x(pg_b8x, sz_utf8_word_break_lane_up_sve2_(bits_u8x, 16), reach_u8x));
    reach_u8x = svand_u8_x(pg_b8x, reach_u8x, sz_utf8_word_break_lane_up_sve2_(reach_u8x, 16));
    bits_u8x = sveor_u8_x(pg_b8x, bits_u8x,
                          svand_u8_x(pg_b8x, sz_utf8_word_break_lane_up_sve2_(bits_u8x, 32), reach_u8x));
    if (inbound_parity) {
        svuint8_t const lane0_u8x = svdup_u8_z(svwhilelt_b8_u64(0, 1), 1);
        bits_u8x = sveor_u8_x(pg_b8x, bits_u8x,
                              sz_utf8_word_break_fill_right_sve2_(svand_u8_x(pg_b8x, run_gate, lane0_u8x), run_gate));
    }
    *inclusive_out = bits_u8x;
    return svand_u8_x(pg_b8x, ri, sveor_u8_x(pg_b8x, bits_u8x, ri));
}
SZ_HELPER_INLINE sz_u8_t sz_utf8_word_break_at_first_sve2_(svuint8_t m, svuint8_t v) {
    svbool_t const pg_b8x = svptrue_b8();
    svbool_t const p_b8x = svcmpne_n_u8(pg_b8x, m, 0);
    svbool_t const first_b8x = svand_b_z(svptrue_b8(), p_b8x, svbrka_b_z(svptrue_b8(), p_b8x));
    return svlastb_u8(first_b8x, v);
}
SZ_HELPER_INLINE sz_u8_t sz_utf8_word_break_at_last_sve2_(svuint8_t m, svuint8_t v) {
    svbool_t const pg_b8x = svptrue_b8();
    return svlastb_u8(svcmpne_n_u8(pg_b8x, m, 0), v);
}
SZ_HELPER_INLINE sz_u8_t sz_utf8_word_break_lane0_sve2_(svuint8_t v) { return svlasta_u8(svpfalse_b(), v); }

SZ_HELPER_AUTO sz_size_t sz_utf8_word_break_decide_window_sve2_(                                                    //
    svuint8_t class_aletter_in, svuint8_t class_hebrew_in, svuint8_t class_numeric_in, svuint8_t class_katakana_in, //
    svuint8_t class_extendnumlet_in, svuint8_t class_extend_in, svuint8_t class_zwj_in, svuint8_t class_format_in,  //
    svuint8_t class_midletter_in, svuint8_t class_midnum_in, svuint8_t class_mid_quotes_in, svuint8_t class_cr_in,  //
    svuint8_t class_lf_in, svuint8_t class_newline_in, svuint8_t class_regional_in,                                 //
    svuint8_t wseg_in, svuint8_t pictographic_in, svuint8_t double_quote_byte_in, svuint8_t single_quote_byte_in,   //
    svuint8_t start_bytes_all_u8x, svuint8_t continuation_all_u8x, svuint8_t forced_other_u8x,                      //
    svuint8_t length_two_u8x, svuint8_t length_three_u8x, svuint8_t length_four_u8x, svuint8_t classes_u8x,         //
    sz_size_t loaded, sz_utf8_word_break_carry_t *carry, sz_bool_t more_text, svbool_t *breaks_out) {
    svbool_t const pg_b8x = svptrue_b8();

    svuint8_t const zeros_u8x = svdup_n_u8(0);
    svuint8_t const ones_u8x = svdup_n_u8(1);
    svuint8_t const iota_u8x = svindex_u8(0, 1);
    svuint8_t const lane0mask_u8x = svdup_u8_z(svwhilelt_b8_u64(0, 1), 1);
    sz_size_t const high_lane = loaded - 1;
    int const at_tail = !more_text;

    svuint8_t const valid_u8x = svdup_u8_z(svwhilelt_b8_u64(0, (sz_u64_t)loaded), 1);
    svuint8_t const start_bytes_u8x = svand_u8_x(pg_b8x, start_bytes_all_u8x, valid_u8x);
    svuint8_t const continuation_u8x = svand_u8_x(pg_b8x, continuation_all_u8x, valid_u8x);

    svuint8_t const lead_two_u8x = svand_u8_x(pg_b8x, length_two_u8x, start_bytes_u8x);
    svuint8_t const lead_three_u8x = svand_u8_x(pg_b8x, length_three_u8x, start_bytes_u8x);
    svuint8_t const lead_four_u8x = svand_u8_x(pg_b8x, length_four_u8x, start_bytes_u8x);
    svuint8_t const nm1_u8x = sveor_n_u8_x(
        pg_b8x, svdup_u8_z(svwhilelt_b8_u64(0, (sz_u64_t)(loaded > 1 ? loaded - 1 : 0)), 1), 1);
    svuint8_t const nm2_u8x = sveor_n_u8_x(
        pg_b8x, svdup_u8_z(svwhilelt_b8_u64(0, (sz_u64_t)(loaded > 2 ? loaded - 2 : 0)), 1), 1);
    svuint8_t const nm3_u8x = sveor_n_u8_x(
        pg_b8x, svdup_u8_z(svwhilelt_b8_u64(0, (sz_u64_t)(loaded > 3 ? loaded - 3 : 0)), 1), 1);
    svuint8_t const truncated_raw_u8x = svand_u8_x(
        pg_b8x,
        svorr_u8_x(
            pg_b8x,
            svorr_u8_x(pg_b8x, svand_u8_x(pg_b8x, lead_two_u8x, nm1_u8x), svand_u8_x(pg_b8x, lead_three_u8x, nm2_u8x)),
            svand_u8_x(pg_b8x, lead_four_u8x, nm3_u8x)),
        valid_u8x);
    svuint8_t const truncated_u8x = svorr_u8_x(pg_b8x, truncated_raw_u8x,
                                               svand_u8_x(pg_b8x, forced_other_u8x, valid_u8x));

    sz_u8_t const left_property = carry->left_property;

    svuint8_t const class_aletter_u8x = svand_u8_x(pg_b8x, svorr_u8_x(pg_b8x, class_aletter_in, class_hebrew_in),
                                                   start_bytes_u8x);
    svuint8_t const class_hebrew_u8x = svand_u8_x(pg_b8x, class_hebrew_in, start_bytes_u8x);
    svuint8_t const class_numeric_u8x = svand_u8_x(pg_b8x, class_numeric_in, start_bytes_u8x);
    svuint8_t const class_katakana_u8x = svand_u8_x(pg_b8x, class_katakana_in, start_bytes_u8x);
    svuint8_t const class_extendnumlet_u8x = svand_u8_x(pg_b8x, class_extendnumlet_in, start_bytes_u8x);
    svuint8_t const lead_ignorable_u8x = svand_u8_x(
        pg_b8x, svorr_u8_x(pg_b8x, svorr_u8_x(pg_b8x, class_extend_in, class_zwj_in), class_format_in),
        start_bytes_u8x);

    svuint8_t const mid_letter_or_quotes_u8x = svand_u8_x(
        pg_b8x, svorr_u8_x(pg_b8x, class_midletter_in, class_mid_quotes_in), start_bytes_u8x);
    svuint8_t const mid_num_or_quotes_u8x = svand_u8_x(pg_b8x, svorr_u8_x(pg_b8x, class_midnum_in, class_mid_quotes_in),
                                                       start_bytes_u8x);
    svuint8_t const mid_quotes_u8x = svand_u8_x(pg_b8x, class_mid_quotes_in, start_bytes_u8x);
    svuint8_t const mid_any_u8x = svand_u8_x(
        pg_b8x, svorr_u8_x(pg_b8x, svorr_u8_x(pg_b8x, class_midletter_in, class_midnum_in), class_mid_quotes_in),
        start_bytes_u8x);
    svuint8_t const class_cr_u8x = svand_u8_x(pg_b8x, class_cr_in, start_bytes_u8x);
    svuint8_t const class_lf_u8x = svand_u8_x(pg_b8x, class_lf_in, start_bytes_u8x);
    svuint8_t const class_newline_u8x = svorr_u8_x(pg_b8x, svorr_u8_x(pg_b8x, class_cr_u8x, class_lf_u8x),
                                                   svand_u8_x(pg_b8x, class_newline_in, start_bytes_u8x));
    svuint8_t const class_zwj_u8x = svand_u8_x(pg_b8x, class_zwj_in, start_bytes_u8x);
    svuint8_t const class_regional_u8x = svand_u8_x(pg_b8x, class_regional_in, start_bytes_u8x);
    svuint8_t const wseg_u8x = svand_u8_x(pg_b8x, svand_u8_x(pg_b8x, wseg_in, start_bytes_u8x),
                                          sveor_n_u8_x(pg_b8x, truncated_u8x, 1));

    svuint8_t const flow_ignorable_u8x = svorr_u8_x(pg_b8x, continuation_u8x, lead_ignorable_u8x);
    svuint8_t const aletter_right_base_u8x = sz_utf8_word_break_fill_right_sve2_(class_aletter_u8x, flow_ignorable_u8x);
    svuint8_t const numeric_right_base_u8x = sz_utf8_word_break_fill_right_sve2_(class_numeric_u8x, flow_ignorable_u8x);
    svuint8_t const hebrew_right_base_u8x = sz_utf8_word_break_fill_right_sve2_(class_hebrew_u8x, flow_ignorable_u8x);
    svuint8_t const aletter_left_base_u8x = sz_utf8_word_break_fill_left_sve2_(class_aletter_u8x, flow_ignorable_u8x);
    svuint8_t const numeric_left_base_u8x = sz_utf8_word_break_fill_left_sve2_(class_numeric_u8x, flow_ignorable_u8x);
    svuint8_t const hebrew_left_base_u8x = sz_utf8_word_break_fill_left_sve2_(class_hebrew_u8x, flow_ignorable_u8x);
    int const left_is_aletter = left_property == sz_utf8_word_break_aletter_k;
    int const left_is_hebrew = left_property == sz_utf8_word_break_hebrew_letter_k;
    int const left_is_numeric = left_property == sz_utf8_word_break_numeric_k;

    svuint8_t const significant_leads_u8x = svand_u8_x(pg_b8x, start_bytes_u8x,
                                                       sveor_n_u8_x(pg_b8x, lead_ignorable_u8x, 1));
    svuint8_t const edge_region_u8x = svand_u8_x(
        pg_b8x, svdup_u8_z(svbrka_b_z(svptrue_b8(), svcmpne_n_u8(pg_b8x, significant_leads_u8x, 0)), 1), valid_u8x);

    int const carry_bridge_aletter = carry->bridge_open && carry->bridge_kind == sz_utf8_word_break_bridge_aletter_k;
    int const carry_bridge_numeric = carry->bridge_open && carry->bridge_kind == sz_utf8_word_break_bridge_numeric_k;
    int const carry_bridge_hebrew = carry->bridge_open && carry->bridge_kind == sz_utf8_word_break_bridge_hebrew_k;
    svuint8_t const seed_aletter_pre_u8x = (left_is_aletter || left_is_hebrew || carry_bridge_aletter) ? edge_region_u8x
                                                                                                       : zeros_u8x;
    svuint8_t const seed_numeric_pre_u8x = (left_is_numeric || carry_bridge_numeric) ? edge_region_u8x : zeros_u8x;
    svuint8_t const seed_hebrew_pre_u8x = (left_is_hebrew || carry_bridge_hebrew) ? edge_region_u8x : zeros_u8x;

    svuint8_t const double_quote_u8x = svand_u8_x(pg_b8x, mid_quotes_u8x, double_quote_byte_in);
    svuint8_t const mid_letter_quotes_no_double_u8x = svand_u8_x(pg_b8x, mid_letter_or_quotes_u8x,
                                                                 sveor_n_u8_x(pg_b8x, double_quote_u8x, 1));
    svuint8_t const mid_num_quotes_no_double_u8x = svand_u8_x(pg_b8x, mid_num_or_quotes_u8x,
                                                              sveor_n_u8_x(pg_b8x, double_quote_u8x, 1));
    svuint8_t const mid_open_aletter_u8x = svand_u8_x(
        pg_b8x, mid_letter_quotes_no_double_u8x,
        svorr_u8_x(pg_b8x, sz_utf8_word_break_lane_up1_sve2_(aletter_right_base_u8x), seed_aletter_pre_u8x));
    svuint8_t const mid_open_numeric_u8x = svand_u8_x(
        pg_b8x, mid_num_quotes_no_double_u8x,
        svorr_u8_x(pg_b8x, sz_utf8_word_break_lane_up1_sve2_(numeric_right_base_u8x), seed_numeric_pre_u8x));
    svuint8_t const mid_open_hebrew_u8x = svand_u8_x(
        pg_b8x, double_quote_u8x,
        svorr_u8_x(pg_b8x, sz_utf8_word_break_lane_up1_sve2_(hebrew_right_base_u8x), seed_hebrew_pre_u8x));
    svuint8_t const mid_open_u8x = svorr_u8_x(pg_b8x, svorr_u8_x(pg_b8x, mid_open_aletter_u8x, mid_open_numeric_u8x),
                                              mid_open_hebrew_u8x);
    svuint8_t const bridge_u8x = svorr_u8_x(
        pg_b8x,
        svorr_u8_x(pg_b8x,
                   svand_u8_x(pg_b8x, mid_open_aletter_u8x, sz_utf8_word_break_lane_dn1_sve2_(aletter_left_base_u8x)),
                   svand_u8_x(pg_b8x, mid_open_numeric_u8x, sz_utf8_word_break_lane_dn1_sve2_(numeric_left_base_u8x))),
        svand_u8_x(pg_b8x, mid_open_hebrew_u8x, sz_utf8_word_break_lane_dn1_sve2_(hebrew_left_base_u8x)));
    int const bridge_any = svptest_any(svptrue_b8(), svcmpne_n_u8(pg_b8x, bridge_u8x, 0));

    svuint8_t const flow_u8x = svorr_u8_x(pg_b8x, flow_ignorable_u8x, bridge_u8x);
    svuint8_t aletter_right_u8x, numeric_right_u8x, aletter_left_u8x, numeric_left_u8x;
    if (!bridge_any) {
        aletter_right_u8x = aletter_right_base_u8x;
        numeric_right_u8x = numeric_right_base_u8x;
        aletter_left_u8x = aletter_left_base_u8x;
        numeric_left_u8x = numeric_left_base_u8x;
    }
    else {
        aletter_right_u8x = sz_utf8_word_break_fill_right_sve2_(class_aletter_u8x, flow_u8x);
        numeric_right_u8x = sz_utf8_word_break_fill_right_sve2_(class_numeric_u8x, flow_u8x);
        aletter_left_u8x = sz_utf8_word_break_fill_left_sve2_(class_aletter_u8x, flow_u8x);
        numeric_left_u8x = sz_utf8_word_break_fill_left_sve2_(class_numeric_u8x, flow_u8x);
    }
    svuint8_t const katakana_right_u8x = sz_utf8_word_break_fill_right_sve2_(class_katakana_u8x, flow_u8x);
    svuint8_t const extendnumlet_right_u8x = sz_utf8_word_break_fill_right_sve2_(class_extendnumlet_u8x, flow_u8x);
    svuint8_t const hebrew_right_u8x = sz_utf8_word_break_fill_right_sve2_(class_hebrew_u8x, flow_u8x);
    svuint8_t const katakana_left_u8x = sz_utf8_word_break_fill_left_sve2_(class_katakana_u8x, flow_u8x);
    svuint8_t const extendnumlet_left_u8x = sz_utf8_word_break_fill_left_sve2_(class_extendnumlet_u8x, flow_u8x);

    svuint8_t previous_newline_u8x = sz_utf8_word_break_lane_up1_sve2_(
        sz_utf8_word_break_smear_right_sve2_(class_newline_u8x, continuation_u8x));
    svuint8_t previous_cr_u8x = sz_utf8_word_break_lane_up1_sve2_(
        sz_utf8_word_break_smear_right_sve2_(class_cr_u8x, continuation_u8x));
    svuint8_t const previous_zwj_u8x = svorr_u8_x(
        pg_b8x,
        sz_utf8_word_break_lane_up1_sve2_(sz_utf8_word_break_smear_right_sve2_(class_zwj_u8x, continuation_u8x)),
        carry->prev_ends_in_zwj ? lane0mask_u8x : zeros_u8x);
    int const left_is_cr = left_property == sz_utf8_word_break_cr_k;
    int const left_is_newline = left_is_cr || left_property == sz_utf8_word_break_lf_k ||
                                left_property == sz_utf8_word_break_newline_k;
    if (left_is_cr) previous_cr_u8x = svorr_u8_x(pg_b8x, previous_cr_u8x, lane0mask_u8x);
    if (left_is_newline) previous_newline_u8x = svorr_u8_x(pg_b8x, previous_newline_u8x, lane0mask_u8x);

    int const leading_mid_bridged = svptest_any(
        svptrue_b8(), svcmpne_n_u8(pg_b8x, svand_u8_x(pg_b8x, edge_region_u8x, bridge_u8x), 0));
    svuint8_t const carry_seed_extension_u8x =
        leading_mid_bridged
            ? svand_u8_x(
                  pg_b8x,
                  sz_utf8_word_break_lane_up1_sve2_(sz_utf8_word_break_fill_right_sve2_(edge_region_u8x, flow_u8x)),
                  start_bytes_u8x)
            : zeros_u8x;
    svuint8_t const carry_seed_u8x = svorr_u8_x(pg_b8x, edge_region_u8x, carry_seed_extension_u8x);
    svuint8_t const seed_aletter_u8x = (left_is_aletter || left_is_hebrew) ? carry_seed_u8x : zeros_u8x;
    svuint8_t const seed_numeric_u8x = left_is_numeric ? carry_seed_u8x : zeros_u8x;
    svuint8_t const seed_hebrew_u8x = left_is_hebrew ? carry_seed_u8x : zeros_u8x;
    svuint8_t const seed_katakana_u8x = (left_property == sz_utf8_word_break_katakana_k) ? carry_seed_u8x : zeros_u8x;
    svuint8_t const seed_extendnumlet_u8x = (left_property == sz_utf8_word_break_extendnumlet_k) ? carry_seed_u8x
                                                                                                 : zeros_u8x;
    svuint8_t const previous_hebrew_u8x = svorr_u8_x(pg_b8x, sz_utf8_word_break_lane_up1_sve2_(hebrew_right_u8x),
                                                     seed_hebrew_u8x);
    svuint8_t const previous_aletter_u8x = svorr_u8_x(pg_b8x, sz_utf8_word_break_lane_up1_sve2_(aletter_right_u8x),
                                                      seed_aletter_u8x);
    svuint8_t const previous_numeric_u8x = svorr_u8_x(pg_b8x, sz_utf8_word_break_lane_up1_sve2_(numeric_right_u8x),
                                                      seed_numeric_u8x);
    svuint8_t const previous_katakana_u8x = svorr_u8_x(pg_b8x, sz_utf8_word_break_lane_up1_sve2_(katakana_right_u8x),
                                                       seed_katakana_u8x);
    svuint8_t const previous_extendnumlet_u8x = svorr_u8_x(
        pg_b8x, sz_utf8_word_break_lane_up1_sve2_(extendnumlet_right_u8x), seed_extendnumlet_u8x);
    svuint8_t const next_aletter_u8x = aletter_left_u8x;
    svuint8_t const next_numeric_u8x = numeric_left_u8x;
    svuint8_t const next_katakana_u8x = katakana_left_u8x;
    svuint8_t const next_extendnumlet_u8x = extendnumlet_left_u8x;

    svuint8_t join_u8x = svorr_u8_x(pg_b8x,
                                    svorr_u8_x(pg_b8x, svand_u8_x(pg_b8x, previous_aletter_u8x, next_aletter_u8x),
                                               svand_u8_x(pg_b8x, previous_numeric_u8x, next_numeric_u8x)),
                                    svorr_u8_x(pg_b8x, svand_u8_x(pg_b8x, previous_aletter_u8x, next_numeric_u8x),
                                               svand_u8_x(pg_b8x, previous_numeric_u8x, next_aletter_u8x)));
    join_u8x = svorr_u8_x(pg_b8x, join_u8x, svand_u8_x(pg_b8x, previous_katakana_u8x, next_katakana_u8x));
    join_u8x = svorr_u8_x(
        pg_b8x, join_u8x,
        svand_u8_x(pg_b8x,
                   svorr_u8_x(pg_b8x,
                              svorr_u8_x(pg_b8x, svorr_u8_x(pg_b8x, previous_aletter_u8x, previous_numeric_u8x),
                                         previous_katakana_u8x),
                              previous_extendnumlet_u8x),
                   next_extendnumlet_u8x));
    join_u8x = svorr_u8_x(
        pg_b8x, join_u8x,
        svand_u8_x(pg_b8x, previous_extendnumlet_u8x,
                   svorr_u8_x(pg_b8x, svorr_u8_x(pg_b8x, next_aletter_u8x, next_numeric_u8x), next_katakana_u8x)));
    join_u8x = svorr_u8_x(pg_b8x, join_u8x, svand_u8_x(pg_b8x, previous_cr_u8x, class_lf_u8x));

    if (svptest_any(svptrue_b8(), svcmpne_n_u8(pg_b8x, class_zwj_u8x, 0)) || carry->prev_ends_in_zwj) {
        svuint8_t const pictographic_u8x = svand_u8_x(pg_b8x, svand_u8_x(pg_b8x, pictographic_in, start_bytes_u8x),
                                                      sveor_n_u8_x(pg_b8x, truncated_u8x, 1));
        join_u8x = svorr_u8_x(pg_b8x, join_u8x, svand_u8_x(pg_b8x, previous_zwj_u8x, pictographic_u8x));
    }

    svuint8_t const previous_wseg_u8x = svorr_u8_x(
        pg_b8x, sz_utf8_word_break_lane_up1_sve2_(sz_utf8_word_break_smear_right_sve2_(wseg_u8x, continuation_u8x)),
        carry->prev_is_wseg ? lane0mask_u8x : zeros_u8x);
    join_u8x = svorr_u8_x(pg_b8x, join_u8x, svand_u8_x(pg_b8x, previous_wseg_u8x, wseg_u8x));

    svuint8_t ri_inclusive_u8x = zeros_u8x;
    int const lane0_continues_ri =
        carry->left_property == sz_utf8_word_break_regional_ind_k &&
        (sz_utf8_word_break_lane0_sve2_(svorr_u8_x(pg_b8x, class_regional_u8x, flow_ignorable_u8x)) != 0);
    svuint8_t ri_run_gate_u8x = svorr_u8_x(
        pg_b8x,
        svorr_u8_x(pg_b8x, sz_utf8_word_break_fill_right_sve2_(class_regional_u8x, flow_ignorable_u8x),
                   sz_utf8_word_break_fill_left_sve2_(class_regional_u8x, flow_ignorable_u8x)),
        class_regional_u8x);
    if (lane0_continues_ri) ri_run_gate_u8x = svorr_u8_x(pg_b8x, ri_run_gate_u8x, lane0mask_u8x);
    svuint8_t const ri_join_u8x = sz_utf8_word_break_ri_join_sve2_(
        class_regional_u8x, ri_run_gate_u8x, lane0_continues_ri ? carry->ri_parity : 0, &ri_inclusive_u8x);
    join_u8x = svorr_u8_x(pg_b8x, join_u8x, ri_join_u8x);

    svuint8_t const force_break_u8x = svand_u8_x(
        pg_b8x,
        svand_u8_x(pg_b8x, svorr_u8_x(pg_b8x, previous_newline_u8x, class_newline_u8x),
                   sveor_n_u8_x(pg_b8x, svand_u8_x(pg_b8x, previous_cr_u8x, class_lf_u8x), 1)),
        start_bytes_u8x);

    svuint8_t const start_bytes_no_bridge_u8x = svand_u8_x(pg_b8x, start_bytes_u8x,
                                                           sveor_n_u8_x(pg_b8x, bridge_u8x, 1));
    svuint8_t boundary_u8x = svand_u8_x(pg_b8x, sveor_n_u8_x(pg_b8x, join_u8x, 1), start_bytes_no_bridge_u8x);
    svuint8_t const single_quote_u8x = svand_u8_x(pg_b8x, mid_quotes_u8x, single_quote_byte_in);
    boundary_u8x = svand_u8_x(pg_b8x, boundary_u8x,
                              sveor_n_u8_x(pg_b8x, svand_u8_x(pg_b8x, single_quote_u8x, previous_hebrew_u8x), 1));
    boundary_u8x = svand_u8_x(pg_b8x, boundary_u8x, sveor_n_u8_x(pg_b8x, lead_ignorable_u8x, 1));
    boundary_u8x = svorr_u8_x(pg_b8x, boundary_u8x, svand_u8_x(pg_b8x, force_break_u8x, start_bytes_no_bridge_u8x));

    svuint8_t const reach_gate_u8x = flow_ignorable_u8x;
    svuint8_t const back_gate_u8x = svorr_u8_x(pg_b8x, flow_ignorable_u8x, mid_any_u8x);
    svuint8_t const top_bit_u8x = at_tail ? zeros_u8x
                                          : svdup_u8_z(svcmpeq_n_u8(svptrue_b8(), iota_u8x, (sz_u8_t)high_lane), 1);
    svuint8_t const open_to_edge_u8x = svand_u8_x(
        pg_b8x, sz_utf8_word_break_fill_right_sve2_(mid_open_u8x, reach_gate_u8x), top_bit_u8x);
    int const open_to_edge_any = !at_tail && svptest_any(svptrue_b8(), svcmpne_n_u8(pg_b8x, open_to_edge_u8x, 0));
    svuint8_t const undecided_u8x =
        open_to_edge_any
            ? svand_u8_x(pg_b8x, mid_open_u8x, sz_utf8_word_break_fill_left_sve2_(open_to_edge_u8x, back_gate_u8x))
            : zeros_u8x;

    svuint8_t const produced_u8x = svand_u8_x(
        pg_b8x, start_bytes_u8x,
        svorr_u8_x(pg_b8x, sveor_n_u8_x(pg_b8x, lane0mask_u8x, 1), carry->have_prev ? lane0mask_u8x : zeros_u8x));
    boundary_u8x = svand_u8_x(pg_b8x, boundary_u8x, produced_u8x);
    sz_size_t resolved = loaded;
    if (svptest_any(svptrue_b8(), svcmpne_n_u8(pg_b8x, undecided_u8x, 0))) {
        svbool_t const und_p_b8x = svcmpne_n_u8(pg_b8x, undecided_u8x, 0);
        svbool_t const brkb_b8x = svbrkb_b_z(svptrue_b8(), und_p_b8x);
        resolved = svcntp_b8(svptrue_b8(), brkb_b8x);
        boundary_u8x = svand_u8_x(pg_b8x, boundary_u8x, svdup_u8_z(brkb_b8x, 1));
    }

    svuint8_t const resolved_valid_u8x = svdup_u8_z(svwhilelt_b8_u64(0, (sz_u64_t)resolved), 1);
    svuint8_t const resolved_starts_u8x = svand_u8_x(pg_b8x, start_bytes_u8x, resolved_valid_u8x);
    svuint8_t const resolved_significant_u8x = svand_u8_x(pg_b8x, resolved_starts_u8x,
                                                          sveor_n_u8_x(pg_b8x, lead_ignorable_u8x, 1));
    int const resolved_sig_any = svptest_any(svptrue_b8(), svcmpne_n_u8(pg_b8x, resolved_significant_u8x, 0));

    sz_u8_t left_out;
    if (!resolved_sig_any) {
        int const carry_is_newline = left_property == sz_utf8_word_break_cr_k ||
                                     left_property == sz_utf8_word_break_lf_k ||
                                     left_property == sz_utf8_word_break_newline_k;
        svuint8_t const ignorable_here_u8x = svand_u8_x(pg_b8x, lead_ignorable_u8x, resolved_valid_u8x);
        int const ign_any = svptest_any(svptrue_b8(), svcmpne_n_u8(pg_b8x, ignorable_here_u8x, 0));
        if ((carry_is_newline || carry->have_prev == 0) && ign_any)
            left_out = sz_utf8_word_break_at_first_sve2_(ignorable_here_u8x, classes_u8x);
        else left_out = left_property;
    }
    else {
        sz_u8_t base_class = sz_utf8_word_break_at_last_sve2_(resolved_significant_u8x, classes_u8x);
        int const base_is_newline = base_class == sz_utf8_word_break_cr_k || base_class == sz_utf8_word_break_lf_k ||
                                    base_class == sz_utf8_word_break_newline_k;
        svuint8_t const incl_to_last_u8x = sz_utf8_word_break_fill_left_sve2_(resolved_significant_u8x, ones_u8x);
        svuint8_t const ignorable_after_u8x = svand_u8_x(pg_b8x,
                                                         svand_u8_x(pg_b8x, lead_ignorable_u8x, resolved_valid_u8x),
                                                         sveor_n_u8_x(pg_b8x, incl_to_last_u8x, 1));
        if (base_is_newline && svptest_any(svptrue_b8(), svcmpne_n_u8(pg_b8x, ignorable_after_u8x, 0)))
            base_class = sz_utf8_word_break_at_first_sve2_(ignorable_after_u8x, classes_u8x);
        left_out = base_class;
    }
    carry->left_property = left_out;
    carry->have_prev = 1;

    if (resolved_sig_any) {
        sz_u8_t const top_is_regional = sz_utf8_word_break_at_last_sve2_(resolved_significant_u8x, class_regional_u8x);
        sz_u8_t const edge_parity = sz_utf8_word_break_at_last_sve2_(resolved_significant_u8x, ri_inclusive_u8x);
        carry->ri_parity = top_is_regional ? edge_parity : 0;
    }

    if (svptest_any(svptrue_b8(), svcmpne_n_u8(pg_b8x, resolved_starts_u8x, 0))) {
        carry->prev_is_wseg = sz_utf8_word_break_at_last_sve2_(resolved_starts_u8x, wseg_u8x);
        carry->prev_ends_in_zwj = sz_utf8_word_break_at_last_sve2_(resolved_starts_u8x, class_zwj_u8x);
    }
    else {
        carry->prev_is_wseg = 0;
        carry->prev_ends_in_zwj = 0;
    }

    svuint8_t const open_at_edge_u8x = svand_u8_x(
        pg_b8x, sz_utf8_word_break_fill_right_sve2_(mid_open_u8x, reach_gate_u8x), resolved_valid_u8x);
    sz_size_t const k = resolved > 0 ? resolved - 1 : 0;
    svuint8_t const open_run_u8x = svand_u8_x(pg_b8x, open_at_edge_u8x,
                                              sveor_n_u8_x(pg_b8x, svdup_u8_z(svwhilelt_b8_u64(0, (sz_u64_t)k), 1), 1));
    if (svptest_any(svptrue_b8(), svcmpne_n_u8(pg_b8x, open_run_u8x, 0))) {
        svuint8_t const reaching_u8x = svand_u8_x(pg_b8x, mid_open_u8x,
                                                  sz_utf8_word_break_fill_left_sve2_(open_run_u8x, back_gate_u8x));
        if (svptest_any(svptrue_b8(), svcmpne_n_u8(pg_b8x, reaching_u8x, 0))) {
            sz_u8_t const is_hebrew = sz_utf8_word_break_at_last_sve2_(reaching_u8x, mid_open_hebrew_u8x);
            sz_u8_t const is_numeric = sz_utf8_word_break_at_last_sve2_(reaching_u8x, mid_open_numeric_u8x);
            carry->bridge_open = 1;
            carry->bridge_kind = is_hebrew ? sz_utf8_word_break_bridge_hebrew_k
                                           : (is_numeric ? sz_utf8_word_break_bridge_numeric_k
                                                         : sz_utf8_word_break_bridge_aletter_k);
        }
        else {
            carry->bridge_open = 0;
            carry->bridge_kind = sz_utf8_word_break_bridge_none_k;
        }
    }
    else if (resolved_sig_any) {
        carry->bridge_open = 0;
        carry->bridge_kind = sz_utf8_word_break_bridge_none_k;
    }

    *breaks_out = svcmpne_n_u8(pg_b8x, boundary_u8x, 0);
    return resolved;
}

/** @brief  Per-lane u16 (=(high<<8)|low) membership in any sorted [lo,hi] range -> 0/1 byte mask; mirrors
 *          `sz_utf8_word_break_range16_one_neon_` (not-below: high greater, or equal-high with low at least lo;
 *          symmetric for not-above). Used by the WSegSpace / Extended_Pictographic scans inside @ref
 *          sz_utf8_word_break_resolve_window_sve2_. */
SZ_HELPER_AUTO svuint8_t sz_utf8_word_break_range16_sve2_(svuint8_t high, svuint8_t low, sz_u16_t const *lo_t,
                                                          sz_u16_t const *hi_t, int count) {
    svbool_t const pg_b8x = svptrue_b8();
    svbool_t acc_b8x = svpfalse_b();
    for (int i = 0; i < count; ++i) {
        sz_u8_t const lh = (sz_u8_t)(lo_t[i] >> 8), ll = (sz_u8_t)(lo_t[i] & 0xFF);
        sz_u8_t const hh = (sz_u8_t)(hi_t[i] >> 8), hl = (sz_u8_t)(hi_t[i] & 0xFF);
        svbool_t const ge_b8x = svorr_b_z(
            pg_b8x, svcmpgt_n_u8(pg_b8x, high, lh),
            svand_b_z(pg_b8x, svcmpeq_n_u8(pg_b8x, high, lh), svcmpge_n_u8(pg_b8x, low, ll)));
        svbool_t const le_b8x = svorr_b_z(
            pg_b8x, svcmplt_n_u8(pg_b8x, high, hh),
            svand_b_z(pg_b8x, svcmpeq_n_u8(pg_b8x, high, hh), svcmple_n_u8(pg_b8x, low, hl)));
        acc_b8x = svorr_b_z(pg_b8x, acc_b8x, svand_b_z(pg_b8x, ge_b8x, le_b8x));
    }
    return svdup_u8_z(acc_b8x, 1);
}

/** @brief  Lower one window to the engine's lane masks (held as `svuint8_t` locals, no spill) and run the rule
 *          engine. Builds the 15 Word_Break class masks + WSegSpace / Extended_Pictographic / quote bytes +
 *          partition masks (the old `build_frame`), then decides for @p complete_limit and re-resolves the carry
 *          to @p adv when an open bridge is still undecided at the edge -- the two-edge carry logic that mirrors
 *          the NEON driver. Replaces the deleted spill-array frame round-trip with in-register masks. */
SZ_HELPER_AUTO sz_size_t sz_utf8_word_break_resolve_window_sve2_(                                                  //
    sz_u8_t const *raw, sz_u8_t const *high_a, sz_u8_t const *low_a, sz_u8_t const *plane_a, sz_u8_t const *cls_a, //
    sz_u64_t start_bytes_all, sz_u64_t length_two, sz_u64_t length_three, sz_u64_t length_four,                    //
    sz_u64_t continuation_all, sz_u64_t forced_other, sz_u64_t four_byte_starts,                                   //
    sz_size_t loaded, sz_size_t complete_limit, sz_bool_t more_text, int want_pictographic,                        //
    sz_utf8_word_break_carry_t *carry, svbool_t *breaks_out) {

    svbool_t const pg_b8x = svptrue_b8();
    svbool_t const loaded_b8x = svwhilelt_b8_u64(0, (sz_u64_t)loaded);
    sz_u64_t const valid = sz_u64_mask_until_serial_(loaded);
    sz_u64_t const start_bytes = start_bytes_all & valid;

    svuint8_t classes_u8x = svld1_u8(loaded_b8x, cls_a);

    // Truncated-edge U+FFFD reclassify: a multi-byte lead whose declared span runs past `loaded` -> Other.
    sz_u64_t const lead_two = length_two & start_bytes;
    sz_u64_t const lead_three = length_three & start_bytes;
    sz_u64_t const lead_four = length_four & start_bytes;
    sz_u64_t const truncated_raw = ((lead_two & ~sz_u64_mask_until_serial_(loaded > 1 ? loaded - 1 : 0)) |
                                    (lead_three & ~sz_u64_mask_until_serial_(loaded > 2 ? loaded - 2 : 0)) |
                                    (lead_four & ~sz_u64_mask_until_serial_(loaded > 3 ? loaded - 3 : 0))) &
                                   valid;
    if (truncated_raw)
        classes_u8x = svsel_u8(sz_utf8_u64_to_pred_sve2_(truncated_raw, loaded),
                               svdup_n_u8((sz_u8_t)sz_utf8_word_break_other_k), classes_u8x);

    // 15 Word_Break class masks as in-register 0/1 byte vectors (the old per-class store macro, inlined per class).
    svuint8_t const class_aletter_u8x = svdup_u8_z(
        svcmpeq_n_u8(loaded_b8x, classes_u8x, (sz_u8_t)sz_utf8_word_break_aletter_k), 1);
    svuint8_t const class_hebrew_u8x = svdup_u8_z(
        svcmpeq_n_u8(loaded_b8x, classes_u8x, (sz_u8_t)sz_utf8_word_break_hebrew_letter_k), 1);
    svuint8_t const class_numeric_u8x = svdup_u8_z(
        svcmpeq_n_u8(loaded_b8x, classes_u8x, (sz_u8_t)sz_utf8_word_break_numeric_k), 1);
    svuint8_t const class_katakana_u8x = svdup_u8_z(
        svcmpeq_n_u8(loaded_b8x, classes_u8x, (sz_u8_t)sz_utf8_word_break_katakana_k), 1);
    svuint8_t const class_extendnumlet_u8x = svdup_u8_z(
        svcmpeq_n_u8(loaded_b8x, classes_u8x, (sz_u8_t)sz_utf8_word_break_extendnumlet_k), 1);
    svuint8_t const class_extend_u8x = svdup_u8_z(
        svcmpeq_n_u8(loaded_b8x, classes_u8x, (sz_u8_t)sz_utf8_word_break_extend_k), 1);
    svuint8_t const class_zwj_u8x = svdup_u8_z(svcmpeq_n_u8(loaded_b8x, classes_u8x, (sz_u8_t)sz_utf8_word_break_zwj_k),
                                               1);
    svuint8_t const class_format_u8x = svdup_u8_z(
        svcmpeq_n_u8(loaded_b8x, classes_u8x, (sz_u8_t)sz_utf8_word_break_format_k), 1);
    svuint8_t const class_midletter_u8x = svdup_u8_z(
        svcmpeq_n_u8(loaded_b8x, classes_u8x, (sz_u8_t)sz_utf8_word_break_midletter_k), 1);
    svuint8_t const class_midnum_u8x = svdup_u8_z(
        svcmpeq_n_u8(loaded_b8x, classes_u8x, (sz_u8_t)sz_utf8_word_break_midnum_k), 1);
    svuint8_t const class_mid_quotes_u8x = svdup_u8_z(
        svcmpeq_n_u8(loaded_b8x, classes_u8x, (sz_u8_t)sz_utf8_word_break_mid_quotes_k), 1);
    svuint8_t const class_cr_u8x = svdup_u8_z(svcmpeq_n_u8(loaded_b8x, classes_u8x, (sz_u8_t)sz_utf8_word_break_cr_k),
                                              1);
    svuint8_t const class_lf_u8x = svdup_u8_z(svcmpeq_n_u8(loaded_b8x, classes_u8x, (sz_u8_t)sz_utf8_word_break_lf_k),
                                              1);
    svuint8_t const class_newline_u8x = svdup_u8_z(
        svcmpeq_n_u8(loaded_b8x, classes_u8x, (sz_u8_t)sz_utf8_word_break_newline_k), 1);
    svuint8_t const class_regional_u8x = svdup_u8_z(
        svcmpeq_n_u8(loaded_b8x, classes_u8x, (sz_u8_t)sz_utf8_word_break_regional_ind_k), 1);

    svuint8_t const validv_u8x = svdup_u8_z(loaded_b8x, 1);
    svuint8_t const raw_u8x = svld1_u8(loaded_b8x, raw);
    svuint8_t const high_u8x = svld1_u8(loaded_b8x, high_a);
    svuint8_t const low_u8x = svld1_u8(loaded_b8x, low_a);

    svuint8_t const non_ascii_u8x = svand_u8_x(pg_b8x, svdup_u8_z(svcmpge_n_u8(pg_b8x, raw_u8x, 0x80), 1), validv_u8x);
    // Astral (4-byte) lanes carry the SMP low-16 in high/low (the astral classifier needs it). All WSegSpace and
    // all BMP-table pictographic ranges live in the BMP, so the BMP (high,low) range scans must EXCLUDE 4-byte
    // lanes or an astral codepoint whose low-16 lands in a BMP range (e.g. U+42009 -> 0x2009) false-matches.
    svuint8_t const four_u8x = svdup_u8_z(sz_utf8_u64_to_pred_sve2_(four_byte_starts & valid, loaded), 1);
    svuint8_t const bmp_lane_u8x = svand_u8_x(pg_b8x, non_ascii_u8x, sveor_n_u8_x(pg_b8x, four_u8x, 1));
    svuint8_t const double_quote_byte_u8x = svand_u8_x(pg_b8x, svdup_u8_z(svcmpeq_n_u8(pg_b8x, raw_u8x, 0x22), 1),
                                                       validv_u8x);
    svuint8_t const single_quote_byte_u8x = svand_u8_x(pg_b8x, svdup_u8_z(svcmpeq_n_u8(pg_b8x, raw_u8x, 0x27), 1),
                                                       validv_u8x);

    // WB3d WSegSpace: ASCII U+0020 byte compare, else multibyte (high,low) range membership (BMP non-ASCII lanes only).
    svuint8_t const wseg_ascii_u8x = svand_u8_x(pg_b8x, svdup_u8_z(svcmpeq_n_u8(pg_b8x, raw_u8x, 0x20), 1), validv_u8x);
    svuint8_t wseg_multibyte_u8x = svdup_n_u8(0);
    if (svptest_any(loaded_b8x, svcmpne_n_u8(pg_b8x, bmp_lane_u8x, 0))) {
        wseg_multibyte_u8x = svand_u8_x(
            pg_b8x,
            sz_utf8_word_break_range16_sve2_(high_u8x, low_u8x, sz_utf8_word_break_wseg_lo_,
                                             sz_utf8_word_break_wseg_hi_, sz_utf8_word_break_wseg_count_k),
            bmp_lane_u8x);
    }
    svuint8_t const wseg_u8x = svorr_u8_x(pg_b8x, wseg_multibyte_u8x, wseg_ascii_u8x);

    // WB3c Extended_Pictographic: BMP range scan on non-4-byte non-ASCII lanes, SMP range scan on plane-1 4-byte
    // lanes (decode already stores SMP low-16 in high/low there). Rare-class gated on `want_pictographic`.
    svuint8_t pictographic_u8x = svdup_n_u8(0);
    if (want_pictographic) {
        svuint8_t const plane_off_u8x = svld1_u8(loaded_b8x, plane_a);
        svuint8_t const plane_one_u8x = svdup_u8_z(svcmpeq_n_u8(pg_b8x, plane_off_u8x, 0), 1);
        svuint8_t const pict_bmp_u8x = svand_u8_x(
            pg_b8x,
            svand_u8_x(
                pg_b8x,
                sz_utf8_word_break_range16_sve2_(high_u8x, low_u8x, sz_utf8_word_break_pict_bmp_lo_,
                                                 sz_utf8_word_break_pict_bmp_hi_, sz_utf8_word_break_pict_bmp_count_k),
                non_ascii_u8x),
            sveor_n_u8_x(pg_b8x, four_u8x, 1));
        svuint8_t const pict_smp_u8x = svand_u8_x(
            pg_b8x,
            svand_u8_x(
                pg_b8x,
                sz_utf8_word_break_range16_sve2_(high_u8x, low_u8x, sz_utf8_word_break_pict_smp_lo_,
                                                 sz_utf8_word_break_pict_smp_hi_, sz_utf8_word_break_pict_smp_count_k),
                four_u8x),
            plane_one_u8x);
        pictographic_u8x = svorr_u8_x(pg_b8x, pict_bmp_u8x, pict_smp_u8x);
    }

    svuint8_t const start_bytes_u8x = svdup_u8_z(sz_utf8_u64_to_pred_sve2_(start_bytes_all, loaded), 1);
    svuint8_t const continuation_u8x = svdup_u8_z(sz_utf8_u64_to_pred_sve2_(continuation_all, loaded), 1);
    svuint8_t const forced_other_u8x = svdup_u8_z(sz_utf8_u64_to_pred_sve2_(forced_other, loaded), 1);
    svuint8_t const length_two_u8x = svdup_u8_z(sz_utf8_u64_to_pred_sve2_(length_two, loaded), 1);
    svuint8_t const length_three_u8x = svdup_u8_z(sz_utf8_u64_to_pred_sve2_(length_three, loaded), 1);
    svuint8_t const length_four_u8x = svdup_u8_z(sz_utf8_u64_to_pred_sve2_(length_four, loaded), 1);

    // Decide for the per-iteration `complete_limit` edge; re-resolve the carry to `adv` if the bridge is still open.
    sz_utf8_word_break_carry_t carry_full = *carry;
    sz_size_t const adv = sz_utf8_word_break_decide_window_sve2_(
        class_aletter_u8x, class_hebrew_u8x, class_numeric_u8x, class_katakana_u8x, class_extendnumlet_u8x,
        class_extend_u8x, class_zwj_u8x, class_format_u8x, class_midletter_u8x, class_midnum_u8x, class_mid_quotes_u8x,
        class_cr_u8x, class_lf_u8x, class_newline_u8x, class_regional_u8x, wseg_u8x, pictographic_u8x,
        double_quote_byte_u8x, single_quote_byte_u8x, start_bytes_u8x, continuation_u8x, forced_other_u8x,
        length_two_u8x, length_three_u8x, length_four_u8x, classes_u8x, complete_limit, &carry_full, more_text,
        breaks_out);
    if (adv > 0 && adv < complete_limit) {
        sz_utf8_word_break_carry_t carry_to_edge = *carry;
        svbool_t tmp_b8x;
        sz_utf8_word_break_decide_window_sve2_(
            class_aletter_u8x, class_hebrew_u8x, class_numeric_u8x, class_katakana_u8x, class_extendnumlet_u8x,
            class_extend_u8x, class_zwj_u8x, class_format_u8x, class_midletter_u8x, class_midnum_u8x,
            class_mid_quotes_u8x, class_cr_u8x, class_lf_u8x, class_newline_u8x, class_regional_u8x, wseg_u8x,
            pictographic_u8x, double_quote_byte_u8x, single_quote_byte_u8x, start_bytes_u8x, continuation_u8x,
            forced_other_u8x, length_two_u8x, length_three_u8x, length_four_u8x, classes_u8x, adv, &carry_to_edge,
            sz_true_k, &tmp_b8x);
        *carry = carry_to_edge;
    }
    else { *carry = carry_full; }
    return adv;
}

/** @brief Single-`svcompact_u32` boundary drain (the SVE2 word twin of `sz_utf8_rune_drain_sve2_`'s shape). Boundary
 *         lanes (all in [0, svcntw())) are compacted once; each consecutive pair forms one (start,length) span chained
 *         through the carried open `word_start`. No svcntw sub-block peel loop. */
SZ_HELPER_AUTO sz_size_t sz_utf8_word_drain_sve2_(svbool_t boundary_b8, sz_size_t base, sz_size_t *starts,
                                                  sz_size_t *lengths, sz_size_t produced, sz_size_t capacity,
                                                  sz_size_t *word_start_io) {
    svbool_t const pg_b32x = svptrue_b32();
    svuint8_t const flags_u8x = svdup_u8_z(boundary_b8, 1);
    svbool_t const emit_b32x = svcmpne_n_u32(pg_b32x, svunpklo_u32(svunpklo_u16(flags_u8x)), 0);
    sz_size_t const k = svcntp_b32(pg_b32x, emit_b32x);
    if (k == 0 || produced >= capacity) return produced;
    svuint32_t const packed_u32x = svcompact_u32(emit_b32x, svindex_u32(0, 1));
    sz_u32_t idx[64];
    svst1_u32(pg_b32x, idx, packed_u32x);
    sz_size_t word_start = *word_start_io;
    sz_size_t emitted = 0;
    while (emitted < k && produced < capacity) {
        sz_size_t const position = base + (sz_size_t)idx[emitted];
        starts[produced] = word_start;
        lengths[produced] = position - word_start;
        word_start = position;
        ++produced, ++emitted;
    }
    *word_start_io = word_start;
    return produced;
}

/** @brief Forward UAX-29 word segmentation (SVE2), mirroring `sz_utf8_wordbreaks_neon` step for step. To keep the
 *         drain a single `svcompact_u32`, each iteration commits at most `svcntw()` bytes (the 32-bit compaction
 *         width), exactly as the SVE2 rune driver caps its emit span; the carry mechanism re-decodes the deferred
 *         tail, so the windowed result is bit-exact with serial regardless of the per-iteration ceiling. */
SZ_API_COMPTIME sz_size_t sz_utf8_wordbreaks_sve2(   //
    sz_cptr_t text, sz_size_t length,                //
    sz_size_t *word_starts, sz_size_t *word_lengths, //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const cap = (sz_size_t)svcntw(); // single-`svcompact_u32` drain domain (16 at VL=512)

    sz_size_t words = 0;      // words written to the output
    sz_size_t word_start = 0; // start byte of the currently open (unfinished) word
    sz_size_t position = 0;   // codepoint-aligned anchor of the next window

    sz_utf8_word_break_carry_t carry = sz_utf8_word_break_carry_sot_();

    while (position < length) {
        sz_size_t const available = length - position;
        sz_size_t const loaded_guess = available < 64 ? available : 64;
        int const decode_at_end = (position + loaded_guess >= length);

        sz_u8_t high[64], low[64], plane[64], cls[64];
        sz_utf8_word_window_sve2_t const w = sz_utf8_word_decode_window_sve2_(text_u8 + position, available,
                                                                              decode_at_end, high, low, plane);
        sz_size_t const loaded = w.loaded;
        sz_u64_t const start_bytes_all = w.partition.start_bytes;
        sz_u64_t const continuation_all = w.partition.continuation;
        sz_u64_t const forced_other = w.partition.forced_other;
        sz_u64_t const length_two = w.partition.length_two;
        sz_u64_t const length_three = w.partition.length_three;
        sz_u64_t const length_four = w.partition.length_four;
        sz_utf8_word_classify_window_sve2_(high, low, plane, w.four_byte_starts, forced_other, loaded, cls);

        // Effective per-iteration commit edge: at most `cap` bytes (the single-`svcompact` ceiling).
        sz_size_t const effective = loaded < cap ? loaded : cap;
        sz_bool_t const more_text = (position + effective < length) ? sz_true_k : sz_false_k;

        // complete_limit relative to the `effective` edge: the largest codepoint-aligned point with no straddle.
        sz_size_t complete_limit = effective;
        if (more_text) {
            sz_u64_t const evalid = sz_u64_mask_until_serial_(effective);
            sz_u64_t const two = length_two & start_bytes_all;
            sz_u64_t const three = length_three & start_bytes_all;
            sz_u64_t const four = length_four & start_bytes_all;
            sz_u64_t const straddle = ((two & ~sz_u64_mask_until_serial_(effective > 1 ? effective - 1 : 0)) |
                                       (three & ~sz_u64_mask_until_serial_(effective > 2 ? effective - 2 : 0)) |
                                       (four & ~sz_u64_mask_until_serial_(effective > 3 ? effective - 3 : 0))) &
                                      evalid;
            sz_size_t limit = straddle ? (sz_size_t)sz_u64_ctz(straddle) : effective;
            if ((text_u8[position + effective] & 0xC0) == 0x80) {
                sz_u64_t const starts_in_eff = start_bytes_all & evalid;
                if (starts_in_eff) {
                    sz_size_t const last_lead = (sz_size_t)(63 - sz_u64_clz(starts_in_eff));
                    sz_size_t const last_lead_length = sz_utf8_lead_length_(text_u8[position + last_lead]);
                    if (last_lead + last_lead_length > effective && last_lead < limit) limit = last_lead;
                }
            }
            if (limit > 0) complete_limit = limit;
        }

        int want_pictographic = carry.prev_ends_in_zwj;
        if (!want_pictographic) {
            svbool_t const loaded_b8x = svwhilelt_b8_u64(0, (sz_u64_t)loaded);
            want_pictographic = svptest_any(
                loaded_b8x, svcmpeq_n_u8(loaded_b8x, svld1_u8(loaded_b8x, cls), (sz_u8_t)sz_utf8_word_break_zwj_k));
        }

        svbool_t breaks_b8x;
        sz_size_t const adv = sz_utf8_word_break_resolve_window_sve2_(
            text_u8 + position, high, low, plane, cls, start_bytes_all, length_two, length_three, length_four,
            continuation_all, forced_other, w.four_byte_starts, loaded, complete_limit, more_text, want_pictographic,
            &carry, &breaks_b8x);

        svbool_t const boundary_b8x = svand_b_z(svptrue_b8(), breaks_b8x, svwhilelt_b8_u64(0, (sz_u64_t)adv));
        words = sz_utf8_word_drain_sve2_(boundary_b8x, position, word_starts, word_lengths, words, words_capacity,
                                         &word_start);
        if (words == words_capacity) {
            if (bytes_consumed) *bytes_consumed = word_start;
            return words;
        }

        position += (adv > 0 && adv < complete_limit) ? adv : (complete_limit ? complete_limit : loaded);
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

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE2

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDBREAKS_SVE2_H_
