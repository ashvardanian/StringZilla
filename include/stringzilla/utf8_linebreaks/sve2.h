/**
 *  @brief SVE2 (AArch64 scalable) backend for UAX-14 line-break opportunities.
 *  @file include/stringzilla/utf8_linebreaks/sve2.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_LINEBREAKS_SVE2_H_
#define STRINGZILLA_UTF8_LINEBREAKS_SVE2_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_linebreaks/tables.h"
#include "stringzilla/utf8_linebreaks/serial.h"
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

#pragma region UAX 14 Line Boundaries forward kernel

/** @brief  Flat-palette index for one chunk of ASTRAL codepoints over the 20-bit offset = cp - 0x10000 (5-nibble
 *          cascade), the SVE2 twin of @ref sz_line_break_classify_astral_neon_. Returns 62-entry palette indices. */
SZ_HELPER_AUTO svuint8_t sz_line_break_classify_astral_sve2_(svuint8_t plane_u8x, svuint8_t high_u8x,
                                                             svuint8_t low_u8x) {
    svbool_t const all_b8x = svptrue_b8();
    svuint8_t const n4_u8x = svand_n_u8_x(all_b8x, plane_u8x, 0x0F);
    svuint8_t const n3_u8x = svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, high_u8x, 4), 0x0F);
    svuint8_t const stage1_index_u8x = svorr_u8_x(all_b8x, svlsl_n_u8_x(all_b8x, n4_u8x, 4), n3_u8x);
    svuint8_t const page_u8x = sz_utf8_rune_lut_sve2_(sz_utf8_line_break_haswell_astral_stage1_, 256, stage1_index_u8x);
    svuint8_t const n2_u8x = svand_n_u8_x(all_b8x, high_u8x, 0x0F);
    svuint8_t const leaf2_lo_u8x = sz_utf8_rune_cascade_sve2_(
        sz_utf8_line_break_haswell_astral_stage2_lo_, (int)sz_utf8_line_break_haswell_astral_stage2_lo_count_k / 16,
        page_u8x, n2_u8x);
    svuint8_t const n1_u8x = svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, low_u8x, 4), 0x0F);
    svuint8_t const leaf_lo_u8x = sz_utf8_rune_cascade_sve2_(
        sz_utf8_line_break_haswell_astral_stage3_lo_, (int)sz_utf8_line_break_haswell_astral_stage3_lo_count_k / 16,
        leaf2_lo_u8x, n1_u8x);
    svuint8_t const leaf_hi_u8x = sz_utf8_rune_cascade_sve2_(
        sz_utf8_line_break_haswell_astral_stage3_hi_, (int)sz_utf8_line_break_haswell_astral_stage3_hi_count_k / 16,
        leaf2_lo_u8x, n1_u8x);
    svuint8_t const n0_u8x = svand_n_u8_x(all_b8x, low_u8x, 0x0F);
    svuint8_t const leaf_group_u8x = svorr_u8_x(all_b8x,
                                                svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, leaf_lo_u8x, 4), 0x0F),
                                                svlsl_n_u8_x(all_b8x, leaf_hi_u8x, 4));
    svuint8_t const stage4_index_u8x = svorr_u8_x(
        all_b8x, svlsl_n_u8_x(all_b8x, svand_n_u8_x(all_b8x, leaf_lo_u8x, 0x0F), 4), n0_u8x);
    svuint8_t result_u8x = svdup_n_u8(0);
    for (int group = 0; group < (int)sz_utf8_line_break_haswell_astral_leaf_groups_k; ++group) {
        svuint8_t const value_u8x = sz_utf8_rune_lut_sve2_(
            sz_utf8_line_break_haswell_astral_stage4_groups_ + group * 256, 256, stage4_index_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(all_b8x, leaf_group_u8x, (sz_u8_t)group), value_u8x, result_u8x);
    }
    return result_u8x;
}

/** @brief  Split one chunk of flat-palette indices into the low and high bytes of their 16-bit Line_Break
 *          descriptors, gathered straight from the 64-word palette by one `svld1uh_gather` per 32-bit quarter -
 *          the SVE2 stand-in for the NEON resident `vqtbl4q` pair and the AVX2 `vpgatherdd`. */
SZ_HELPER_AUTO void sz_line_break_flat_descriptors_sve2_(svuint8_t palette_indices_u8x,
                                                         svuint8_t *descriptor_low_u8x_out,
                                                         svuint8_t *descriptor_high_u8x_out) {
    svbool_t const all_b32x = svptrue_b32();
    sz_u16_t const *palette = sz_utf8_line_break_flat_palette_;
    svuint16_t const indices_lo_u16x = svunpklo_u16(palette_indices_u8x),
                     indices_hi_u16x = svunpkhi_u16(palette_indices_u8x);

    svuint32_t const first_u32x = svld1uh_gather_u32offset_u32(
        all_b32x, palette, svlsl_n_u32_x(all_b32x, svunpklo_u32(indices_lo_u16x), 1));
    svuint32_t const second_u32x = svld1uh_gather_u32offset_u32(
        all_b32x, palette, svlsl_n_u32_x(all_b32x, svunpkhi_u32(indices_lo_u16x), 1));
    svuint32_t const third_u32x = svld1uh_gather_u32offset_u32(
        all_b32x, palette, svlsl_n_u32_x(all_b32x, svunpklo_u32(indices_hi_u16x), 1));
    svuint32_t const fourth_u32x = svld1uh_gather_u32offset_u32(
        all_b32x, palette, svlsl_n_u32_x(all_b32x, svunpkhi_u32(indices_hi_u16x), 1));

    // Narrow the four word-lane quarters back into byte-lane order, once for each descriptor byte.
    svuint16_t const low_first_u16x = svuzp1_u16(svreinterpret_u16_u32(svand_n_u32_x(all_b32x, first_u32x, 0xFF)),
                                                 svreinterpret_u16_u32(svand_n_u32_x(all_b32x, second_u32x, 0xFF)));
    svuint16_t const low_second_u16x = svuzp1_u16(svreinterpret_u16_u32(svand_n_u32_x(all_b32x, third_u32x, 0xFF)),
                                                  svreinterpret_u16_u32(svand_n_u32_x(all_b32x, fourth_u32x, 0xFF)));
    *descriptor_low_u8x_out = svuzp1_u8(svreinterpret_u8_u16(low_first_u16x), svreinterpret_u8_u16(low_second_u16x));
    svuint16_t const high_first_u16x = svuzp1_u16(svreinterpret_u16_u32(svlsr_n_u32_x(all_b32x, first_u32x, 8)),
                                                  svreinterpret_u16_u32(svlsr_n_u32_x(all_b32x, second_u32x, 8)));
    svuint16_t const high_second_u16x = svuzp1_u16(svreinterpret_u16_u32(svlsr_n_u32_x(all_b32x, third_u32x, 8)),
                                                   svreinterpret_u16_u32(svlsr_n_u32_x(all_b32x, fourth_u32x, 8)));
    *descriptor_high_u8x_out = svuzp1_u8(svreinterpret_u8_u16(high_first_u16x), svreinterpret_u8_u16(high_second_u16x));
}

/** @brief  Expand one chunk of flat-palette indices to the LB1-resolved class byte, the engine side byte and the
 *          DottedCircle predicate - the SVE2 twin of @ref sz_line_break_flat_palette_unpack_neon_. Applies the
 *          serial resolution aliasing (SA -> AL/CM, AI/SG/XX -> AL, CJ -> NS); RI/ZWJ side bits come from the RAW
 *          class, the mark side bit from the resolved class. */
SZ_HELPER_AUTO void sz_line_break_flat_palette_unpack_sve2_(svuint8_t palette_indices_u8x, svuint8_t *classes_u8x_out,
                                                            svuint8_t *side_u8x_out, svbool_t *dotted_b8x_out) {
    svbool_t const all_b8x = svptrue_b8();
    svuint8_t descriptor_low_u8x, descriptor_high_u8x;
    sz_line_break_flat_descriptors_sve2_(palette_indices_u8x, &descriptor_low_u8x, &descriptor_high_u8x);

    svuint8_t const raw_classes_u8x = svand_n_u8_x(all_b8x, descriptor_low_u8x, 0x3F);
    svbool_t const is_sa_b8x = svcmpeq_n_u8(all_b8x, raw_classes_u8x, (sz_u8_t)sz_line_break_sa_k);
    svbool_t const sa_is_mark_b8x = svcmpne_n_u8(all_b8x, svand_n_u8_x(all_b8x, descriptor_high_u8x, 1 << 4), 0);
    svuint8_t classes_u8x = svsel_u8(is_sa_b8x, svdup_n_u8((sz_u8_t)sz_line_break_al_k), raw_classes_u8x);
    classes_u8x = svsel_u8(svand_b_z(all_b8x, is_sa_b8x, sa_is_mark_b8x), svdup_n_u8((sz_u8_t)sz_line_break_cm_k),
                           classes_u8x);
    svbool_t const is_alias_b8x = svorr_b_z(
        all_b8x,
        svorr_b_z(all_b8x, svcmpeq_n_u8(all_b8x, classes_u8x, (sz_u8_t)sz_line_break_ai_k),
                  svcmpeq_n_u8(all_b8x, classes_u8x, (sz_u8_t)sz_line_break_sg_k)),
        svcmpeq_n_u8(all_b8x, classes_u8x, (sz_u8_t)sz_line_break_xx_k));
    classes_u8x = svsel_u8(is_alias_b8x, svdup_n_u8((sz_u8_t)sz_line_break_al_k), classes_u8x);
    classes_u8x = svsel_u8(svcmpeq_n_u8(all_b8x, classes_u8x, (sz_u8_t)sz_line_break_cj_k),
                           svdup_n_u8((sz_u8_t)sz_line_break_ns_k), classes_u8x);

    svuint8_t side_u8x = svdup_u8_z(svcmpne_n_u8(all_b8x, svand_n_u8_x(all_b8x, descriptor_low_u8x, 1 << 6), 0),
                                    (sz_u8_t)sz_line_break_side_pi_k);
    side_u8x = svorr_u8_m(svcmpne_n_u8(all_b8x, svand_n_u8_x(all_b8x, descriptor_low_u8x, 1 << 7), 0), side_u8x,
                          svdup_n_u8((sz_u8_t)sz_line_break_side_pf_k));
    side_u8x = svorr_u8_m(svcmpne_n_u8(all_b8x, svand_n_u8_x(all_b8x, descriptor_high_u8x, 1 << 0), 0), side_u8x,
                          svdup_n_u8((sz_u8_t)sz_line_break_side_eaw_k));
    side_u8x = svorr_u8_m(svcmpne_n_u8(all_b8x, svand_n_u8_x(all_b8x, descriptor_high_u8x, 1 << 1), 0), side_u8x,
                          svdup_n_u8((sz_u8_t)(sz_line_break_side_cn_k | sz_line_break_side_ext_k)));
    side_u8x = svorr_u8_m(svcmpeq_n_u8(all_b8x, raw_classes_u8x, (sz_u8_t)sz_line_break_ri_k), side_u8x,
                          svdup_n_u8((sz_u8_t)sz_line_break_side_ri_k));
    side_u8x = svorr_u8_m(svcmpeq_n_u8(all_b8x, raw_classes_u8x, (sz_u8_t)sz_line_break_zwj_k), side_u8x,
                          svdup_n_u8((sz_u8_t)sz_line_break_side_zwj_k));
    svbool_t const class_is_mark_b8x = svorr_b_z(all_b8x,
                                                 svcmpeq_n_u8(all_b8x, classes_u8x, (sz_u8_t)sz_line_break_cm_k),
                                                 svcmpeq_n_u8(all_b8x, classes_u8x, (sz_u8_t)sz_line_break_zwj_k));
    side_u8x = svorr_u8_m(class_is_mark_b8x, side_u8x, svdup_n_u8((sz_u8_t)sz_line_break_side_mark_k));

    *classes_u8x_out = classes_u8x;
    *side_u8x_out = side_u8x;
    *dotted_b8x_out = svcmpne_n_u8(all_b8x, svand_n_u8_x(all_b8x, descriptor_high_u8x, 1 << 5), 0);
}

/** @brief  Membership mask of class @p cls over the six class bit-planes (class ids are < 64). */
SZ_HELPER_INLINE sz_u64_t sz_line_break_plane_class_sve2_(sz_u64_t const *planes, sz_u8_t cls) {
    sz_u64_t members = ~0ull;
    for (int bit = 0; bit < 6; ++bit) members &= ((cls >> bit) & 1) ? planes[bit] : ~planes[bit];
    return members;
}

/**
 *  @brief  Forward UAX-14 line-break-opportunity kernel (SVE2, vector-length agnostic). Bit-exact with
 *          `sz_utf8_linebreaks_serial` and the other ISA fronts: a chunked-window classify + mask-lowering
 *          front-end feeds the shared portable rule engine @ref sz_line_break_decide_window_.
 *
 *  Two streaming passes per 64-byte window. The first lowers the byte-shape predicates (continuation, lead
 *  lengths, the dangerous-lead validity refinements) to `sz_u64_t` masks, so the serial "consume-1 U+FFFD"
 *  malformed policy resolves as plain window-wide mask algebra. The second classifies every byte lane in-register
 *  - the BMP flat leaf through gathered byte loads, descriptors through gathered 16-bit loads, astral leads
 *  through the nibble cascade - stores the per-lane class/side bytes the engine's carry extraction reads, and
 *  lowers the class bit-planes and side masks the frame needs; fifteen-plus per-class masks then assemble from
 *  six bit-planes with scalar mask algebra instead of one compare per class.
 */
SZ_API_COMPTIME sz_size_t sz_utf8_linebreaks_sve2( //
    sz_cptr_t text, sz_size_t length,              //
    sz_size_t *starts, sz_size_t *lengths,         //
    sz_size_t capacity, sz_size_t *bytes_consumed) {

    if (length == 0 || capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const vector_bytes = svcntb();
    sz_size_t const chunk_bytes = vector_bytes < 64 ? vector_bytes : 64;
    sz_size_t const window_capacity = (64 / chunk_bytes) * chunk_bytes;
    svbool_t const all_b8x = svptrue_b8();
    svuint8_t const lane_iota_u8x = svindex_u8(0, 1);

    sz_size_t produced = 0;
    sz_size_t line_start = 0;
    sz_size_t position = 0;
    sz_line_break_carry_t carry = sz_line_break_carry_sot_();

    while (position < length) {
        sz_size_t const available = length - position;
        sz_size_t const loaded = available < window_capacity ? available : window_capacity;
        sz_u64_t const loaded_mask = sz_u64_mask_until_serial_(loaded);
        sz_bool_t const more_text = (sz_bool_t)(position + loaded < length);

        // First pass: lower the byte-shape predicates of every chunk to window-wide masks.
        sz_u64_t continuation = 0, two_byte = 0, three_byte = 0, four_byte = 0, true_ascii = 0;
        sz_u64_t overlong2 = 0, overlong3 = 0, surrogate3 = 0, overlong4 = 0, above4 = 0;
        for (sz_size_t chunk_base = 0; chunk_base < loaded; chunk_base += chunk_bytes) {
            sz_size_t const ahead = loaded - chunk_base;
            sz_size_t const chunk_span = ahead < chunk_bytes ? ahead : chunk_bytes;
            sz_size_t const peek_span = ahead > chunk_bytes
                                            ? (ahead - chunk_bytes < chunk_bytes ? ahead - chunk_bytes : chunk_bytes)
                                            : 0;
            svbool_t const loaded_b8x = svwhilelt_b8_u64(0, (sz_u64_t)chunk_span);
            svbool_t const peek_b8x = svwhilelt_b8_u64(0, (sz_u64_t)peek_span);
            svuint8_t const bytes_u8x = svld1_u8(loaded_b8x, text_u8 + position + chunk_base);
            svuint8_t const peek_u8x = svld1_u8(peek_b8x,
                                                text_u8 + position + chunk_base + (peek_span ? chunk_bytes : 0));

            svbool_t const continuation_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xC0), 0x80);
            svbool_t const two_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xE0), 0xC0);
            svbool_t const three_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xF0), 0xE0);
            svbool_t const four_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xF8), 0xF0);
            svbool_t const ascii_b8x = svcmplt_n_u8(loaded_b8x, bytes_u8x, 0x80);
            continuation |= sz_utf8_rune_pred_to_u64_sve2_(continuation_b8x) << chunk_base;
            two_byte |= sz_utf8_rune_pred_to_u64_sve2_(two_b8x) << chunk_base;
            three_byte |= sz_utf8_rune_pred_to_u64_sve2_(three_b8x) << chunk_base;
            four_byte |= sz_utf8_rune_pred_to_u64_sve2_(four_b8x) << chunk_base;
            true_ascii |= sz_utf8_rune_pred_to_u64_sve2_(ascii_b8x) << chunk_base;

            // Only C0/C1, E0, ED, F0, F4/>=F5 leads can be overlong / surrogate / out-of-range; refine their
            // validity with the peeked first continuation only when such a lead is present in the chunk.
            svbool_t const danger_b8x = svorr_b_z(
                loaded_b8x,
                svorr_b_z(loaded_b8x,
                          svand_b_z(loaded_b8x, svcmpge_n_u8(loaded_b8x, bytes_u8x, 0xC0),
                                    svcmplt_n_u8(loaded_b8x, bytes_u8x, 0xC2)),
                          svcmpeq_n_u8(loaded_b8x, bytes_u8x, 0xE0)),
                svorr_b_z(loaded_b8x, svcmpeq_n_u8(loaded_b8x, bytes_u8x, 0xED),
                          svcmpge_n_u8(loaded_b8x, bytes_u8x, 0xF0)));
            if (svptest_any(loaded_b8x, danger_b8x)) {
                svuint8_t const next1_u8x = svext_u8(bytes_u8x, peek_u8x, 1);
                svbool_t const b1_lt_a0_b8x = svcmplt_n_u8(all_b8x, next1_u8x, 0xA0);
                svbool_t const b1_lt_90_b8x = svcmplt_n_u8(all_b8x, next1_u8x, 0x90);
                overlong2 |= sz_utf8_rune_pred_to_u64_sve2_(
                                 svand_b_z(loaded_b8x, two_b8x,
                                           svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0x1E), 0)))
                             << chunk_base;
                overlong3 |= sz_utf8_rune_pred_to_u64_sve2_(
                                 svand_b_z(loaded_b8x, svcmpeq_n_u8(loaded_b8x, bytes_u8x, 0xE0), b1_lt_a0_b8x))
                             << chunk_base;
                surrogate3 |= sz_utf8_rune_pred_to_u64_sve2_(svand_b_z(loaded_b8x,
                                                                       svcmpeq_n_u8(loaded_b8x, bytes_u8x, 0xED),
                                                                       svnot_b_z(loaded_b8x, b1_lt_a0_b8x)))
                              << chunk_base;
                overlong4 |= sz_utf8_rune_pred_to_u64_sve2_(
                                 svand_b_z(loaded_b8x, svcmpeq_n_u8(loaded_b8x, bytes_u8x, 0xF0), b1_lt_90_b8x))
                             << chunk_base;
                above4 |= sz_utf8_rune_pred_to_u64_sve2_(
                              svorr_b_z(loaded_b8x,
                                        svand_b_z(loaded_b8x, svcmpeq_n_u8(loaded_b8x, bytes_u8x, 0xF4),
                                                  svnot_b_z(loaded_b8x, b1_lt_90_b8x)),
                                        svcmpge_n_u8(loaded_b8x, bytes_u8x, 0xF5)))
                          << chunk_base;
            }
        }

        // The serial consume-1 U+FFFD policy as window-wide mask algebra, bit-identical to the other fronts.
        sz_u64_t const next1_continuation = continuation >> 1, next2_continuation = continuation >> 2,
                       next3_continuation = continuation >> 3;
        sz_u64_t const valid2 = two_byte & next1_continuation & ~overlong2;
        sz_u64_t const valid3 = three_byte & next1_continuation & next2_continuation & ~(overlong3 | surrogate3);
        sz_u64_t const valid4 = four_byte & next1_continuation & next2_continuation & next3_continuation &
                                ~(overlong4 | above4);
        sz_u64_t const is_astral = valid4 & loaded_mask;
        sz_u64_t const valid_start = true_ascii | valid2 | valid3 | valid4;
        sz_u64_t const consumed = (((valid2 | valid3 | valid4) << 1) | ((valid3 | valid4) << 2) | (valid4 << 3)) &
                                  continuation & loaded_mask;
        sz_u64_t const starts_mask = loaded_mask & ~consumed;
        sz_u64_t const replacement = starts_mask & ~valid_start;
        sz_u64_t const non_start = loaded_mask & ~starts_mask;
        sz_u64_t const raw_starts = loaded_mask & ~continuation;
        sz_size_t const complete_limit = sz_line_break_complete_limit_masks_(
            loaded, raw_starts, two_byte & raw_starts, three_byte & raw_starts, four_byte & raw_starts, more_text);

        // Second pass: classify every byte lane, store the class/side bytes the engine's carry extraction reads,
        // and lower the class bit-planes, side masks and DottedCircle lanes.
        sz_u8_t effective_class_byte[64], side_byte[64];
        sz_u64_t class_planes[6] = {0, 0, 0, 0, 0, 0};
        sz_u64_t side_pi = 0, side_pf = 0, side_eaw = 0, side_cn = 0, side_ext = 0, dotted = 0;
        for (sz_size_t chunk_base = 0; chunk_base < loaded; chunk_base += chunk_bytes) {
            sz_size_t const ahead = loaded - chunk_base;
            sz_size_t const chunk_span = ahead < chunk_bytes ? ahead : chunk_bytes;
            sz_size_t const peek_span = ahead > chunk_bytes
                                            ? (ahead - chunk_bytes < chunk_bytes ? ahead - chunk_bytes : chunk_bytes)
                                            : 0;
            svbool_t const loaded_b8x = svwhilelt_b8_u64(0, (sz_u64_t)chunk_span);
            svbool_t const peek_b8x = svwhilelt_b8_u64(0, (sz_u64_t)peek_span);
            svuint8_t const bytes_u8x = svld1_u8(loaded_b8x, text_u8 + position + chunk_base);
            svuint8_t const peek_u8x = svld1_u8(peek_b8x,
                                                text_u8 + position + chunk_base + (peek_span ? chunk_bytes : 0));
            svuint8_t const next1_u8x = svext_u8(bytes_u8x, peek_u8x, 1);
            svuint8_t const next2_u8x = svext_u8(bytes_u8x, peek_u8x, 2);
            svuint8_t const next3_u8x = svext_u8(bytes_u8x, peek_u8x, 3);

            svbool_t const two_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xE0), 0xC0);
            svbool_t const three_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xF0), 0xE0);
            svbool_t const four_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xF8), 0xF0);

            // Codepoint (high, low) bytes per lead length; raw lanes (ASCII, continuations, >= 0xF8) keep
            // `low = raw, high = 0` and replacement lanes are overridden to U+FFFD's palette index downstream.
            svuint8_t const two_high_u8x = svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, bytes_u8x, 2), 0x07);
            svuint8_t const two_low_u8x = svorr_u8_x(all_b8x,
                                                     svand_n_u8_x(all_b8x, svlsl_n_u8_x(all_b8x, bytes_u8x, 6), 0xC0),
                                                     svand_n_u8_x(all_b8x, next1_u8x, 0x3F));
            svuint8_t const three_high_u8x = svorr_u8_x(
                all_b8x, svlsl_n_u8_x(all_b8x, svand_n_u8_x(all_b8x, bytes_u8x, 0x0F), 4),
                svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, next1_u8x, 2), 0x0F));
            svuint8_t const three_low_u8x = svorr_u8_x(all_b8x,
                                                       svand_n_u8_x(all_b8x, svlsl_n_u8_x(all_b8x, next1_u8x, 6), 0xC0),
                                                       svand_n_u8_x(all_b8x, next2_u8x, 0x3F));
            svuint8_t const four_low_u8x = svorr_u8_x(all_b8x,
                                                      svand_n_u8_x(all_b8x, svlsl_n_u8_x(all_b8x, next2_u8x, 6), 0xC0),
                                                      svand_n_u8_x(all_b8x, next3_u8x, 0x3F));
            svuint8_t const four_high_u8x = svorr_u8_x(
                all_b8x, svand_n_u8_x(all_b8x, svlsl_n_u8_x(all_b8x, next1_u8x, 4), 0xF0),
                svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, next2_u8x, 2), 0x0F));

            svuint8_t high_u8x = svdup_u8_z(two_b8x, 0);
            high_u8x = svsel_u8(two_b8x, two_high_u8x, high_u8x);
            high_u8x = svsel_u8(three_b8x, three_high_u8x, high_u8x);
            high_u8x = svsel_u8(four_b8x, four_high_u8x, high_u8x);
            svuint8_t low_u8x = svsel_u8(two_b8x, two_low_u8x, bytes_u8x);
            low_u8x = svsel_u8(three_b8x, three_low_u8x, low_u8x);
            low_u8x = svsel_u8(four_b8x, four_low_u8x, low_u8x);

            sz_u64_t const chunk_lanes = sz_u64_mask_until_serial_(chunk_span);
            svuint8_t bmp_index_u8x = sz_utf8_rune_flat_lookup_ascii_gated_sve2_(
                sz_utf8_line_break_bmp_page_lut_, sz_utf8_line_break_flat_bmp_, bytes_u8x, high_u8x, low_u8x);
            if ((replacement >> chunk_base) & chunk_lanes) {
                svbool_t const replacement_b8x = sz_utf8_rune_u64_to_pred_sve2_(replacement >> chunk_base,
                                                                                lane_iota_u8x);
                svuint8_t const fffd_index_u8x = sz_utf8_rune_flat_lookup_sve2_(
                    sz_utf8_line_break_bmp_page_lut_, sz_utf8_line_break_flat_bmp_, svdup_n_u8(0xFF), svdup_n_u8(0xFD));
                bmp_index_u8x = svsel_u8(replacement_b8x, fffd_index_u8x, bmp_index_u8x);
            }

            svuint8_t classes_u8x, side_u8x;
            svbool_t dotted_b8x;
            sz_line_break_flat_palette_unpack_sve2_(bmp_index_u8x, &classes_u8x, &side_u8x, &dotted_b8x);

            if ((is_astral >> chunk_base) & chunk_lanes) {
                // The astral cascade speaks the 62-entry palette, whose class/side/dotted byte tables carry the
                // same LB1 resolution as the flat descriptor unpack, so blending resolved bytes is bit-identical.
                svbool_t const astral_b8x = sz_utf8_rune_u64_to_pred_sve2_(is_astral >> chunk_base, lane_iota_u8x);
                svuint8_t const plane_u8x = svorr_u8_x(
                    all_b8x, svand_n_u8_x(all_b8x, svlsl_n_u8_x(all_b8x, bytes_u8x, 2), 0x1C),
                    svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, next1_u8x, 4), 0x03));
                svuint8_t const plane_off_u8x = svsub_n_u8_x(all_b8x, svsel_u8(astral_b8x, plane_u8x, svdup_n_u8(1)),
                                                             1);
                svuint8_t const astral_index_u8x = sz_line_break_classify_astral_sve2_(plane_off_u8x, high_u8x,
                                                                                       low_u8x);
                svuint8_t const astral_classes_u8x = sz_utf8_rune_lut_sve2_(sz_utf8_line_break_palette_class_, 256,
                                                                            astral_index_u8x);
                svuint8_t const astral_side_u8x = sz_utf8_rune_lut_sve2_(sz_utf8_line_break_palette_side_, 256,
                                                                         astral_index_u8x);
                svuint8_t const astral_dotted_u8x = sz_utf8_rune_lut_sve2_(sz_utf8_line_break_palette_dotted_, 256,
                                                                           astral_index_u8x);
                classes_u8x = svsel_u8(astral_b8x, astral_classes_u8x, classes_u8x);
                side_u8x = svsel_u8(astral_b8x, astral_side_u8x, side_u8x);
                dotted_b8x = svsel_b(astral_b8x, svcmpne_n_u8(all_b8x, astral_dotted_u8x, 0), dotted_b8x);
            }

            svst1_u8(loaded_b8x, effective_class_byte + chunk_base, classes_u8x);
            svst1_u8(loaded_b8x, side_byte + chunk_base, side_u8x);

            for (int bit = 0; bit < 6; ++bit)
                class_planes[bit] |= sz_utf8_rune_pred_to_u64_sve2_(svcmpne_n_u8(
                                         loaded_b8x, svand_n_u8_x(loaded_b8x, classes_u8x, (sz_u8_t)(1 << bit)), 0))
                                     << chunk_base;
            side_pi |= sz_utf8_rune_pred_to_u64_sve2_(svcmpne_n_u8(
                           loaded_b8x, svand_n_u8_x(loaded_b8x, side_u8x, (sz_u8_t)sz_line_break_side_pi_k), 0))
                       << chunk_base;
            side_pf |= sz_utf8_rune_pred_to_u64_sve2_(svcmpne_n_u8(
                           loaded_b8x, svand_n_u8_x(loaded_b8x, side_u8x, (sz_u8_t)sz_line_break_side_pf_k), 0))
                       << chunk_base;
            side_eaw |= sz_utf8_rune_pred_to_u64_sve2_(svcmpne_n_u8(
                            loaded_b8x, svand_n_u8_x(loaded_b8x, side_u8x, (sz_u8_t)sz_line_break_side_eaw_k), 0))
                        << chunk_base;
            side_cn |= sz_utf8_rune_pred_to_u64_sve2_(svcmpne_n_u8(
                           loaded_b8x, svand_n_u8_x(loaded_b8x, side_u8x, (sz_u8_t)sz_line_break_side_cn_k), 0))
                       << chunk_base;
            side_ext |= sz_utf8_rune_pred_to_u64_sve2_(svcmpne_n_u8(
                            loaded_b8x, svand_n_u8_x(loaded_b8x, side_u8x, (sz_u8_t)sz_line_break_side_ext_k), 0))
                        << chunk_base;
            dotted |= sz_utf8_rune_pred_to_u64_sve2_(svand_b_z(loaded_b8x, dotted_b8x, loaded_b8x)) << chunk_base;
        }

        // Byte-level cluster frame (LB9/LB10) as scalar mask algebra over the lowered planes.
        sz_u64_t const cm_mask = sz_line_break_plane_class_sve2_(class_planes, (sz_u8_t)sz_line_break_cm_k) &
                                 loaded_mask;
        sz_u64_t const zwj_mask = sz_line_break_plane_class_sve2_(class_planes, (sz_u8_t)sz_line_break_zwj_k) &
                                  loaded_mask;
        sz_u64_t const mark_start = (cm_mask | zwj_mask) & starts_mask;
        sz_u64_t const excluded = (sz_line_break_plane_class_sve2_(class_planes, (sz_u8_t)sz_line_break_bk_k) |
                                   sz_line_break_plane_class_sve2_(class_planes, (sz_u8_t)sz_line_break_cr_k) |
                                   sz_line_break_plane_class_sve2_(class_planes, (sz_u8_t)sz_line_break_lf_k) |
                                   sz_line_break_plane_class_sve2_(class_planes, (sz_u8_t)sz_line_break_nl_k) |
                                   sz_line_break_plane_class_sve2_(class_planes, (sz_u8_t)sz_line_break_sp_k) |
                                   sz_line_break_plane_class_sve2_(class_planes, (sz_u8_t)sz_line_break_zw_k)) &
                                  starts_mask;
        sz_u64_t const good_base = starts_mask & ~excluded & ~mark_start;
        sz_u64_t const mark_bytes = sz_u64_fill_right_(mark_start, non_start) | mark_start;
        sz_u64_t const flood = sz_u64_fill_right_(good_base, non_start | mark_bytes);
        sz_u64_t const attached = flood & mark_start;
        sz_u64_t const lone_mark = mark_start & ~attached;

        // LB10: lone marks reclassify to AL, with their side bits cleared - in the byte arrays the engine reads
        // and in the lowered masks alike.
        if (lone_mark)
            for (sz_size_t chunk_base = 0; chunk_base < loaded; chunk_base += chunk_bytes) {
                sz_size_t const chunk_span = loaded - chunk_base < chunk_bytes ? loaded - chunk_base : chunk_bytes;
                if (!((lone_mark >> chunk_base) & sz_u64_mask_until_serial_(chunk_span))) continue;
                svbool_t const loaded_b8x = svwhilelt_b8_u64(0, (sz_u64_t)chunk_span);
                svbool_t const lone_b8x = sz_utf8_rune_u64_to_pred_sve2_(lone_mark >> chunk_base, lane_iota_u8x);
                svuint8_t const classes_u8x = svld1_u8(loaded_b8x, effective_class_byte + chunk_base);
                svuint8_t const side_u8x = svld1_u8(loaded_b8x, side_byte + chunk_base);
                svst1_u8(loaded_b8x, effective_class_byte + chunk_base,
                         svsel_u8(lone_b8x, svdup_n_u8((sz_u8_t)sz_line_break_al_k), classes_u8x));
                svst1_u8(loaded_b8x, side_byte + chunk_base, svsel_u8(lone_b8x, svdup_n_u8(0), side_u8x));
            }

        sz_line_break_frame_t frame;
        frame.base = starts_mask & ~attached;
        frame.gate = non_start | attached;
        frame.attached = attached;
        frame.lone_mark = lone_mark;
        frame.non_start = non_start;
        frame.dotted = dotted & starts_mask;
        frame.starts = starts_mask;
        frame.replacement = replacement;
        //  Full unroll is load-bearing (mirrors the other fronts): the per-class combos are pure scalar algebra,
        //  so the ones the engine never reads fold away.
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 64
#endif
        for (sz_size_t cls = 0; cls < sz_line_break_class_count_k; ++cls)
            frame.effective_class[cls] = sz_line_break_plane_class_sve2_(class_planes, (sz_u8_t)cls) & loaded_mask;
        frame.effective_class[sz_line_break_cm_k] &= ~lone_mark;
        frame.effective_class[sz_line_break_zwj_k] &= ~lone_mark;
        frame.effective_class[sz_line_break_al_k] |= lone_mark;
        frame.raw_zwj = zwj_mask;
        frame.side_pi = side_pi & ~lone_mark;
        frame.side_pf = side_pf & ~lone_mark;
        frame.side_eaw = side_eaw & ~lone_mark;
        frame.side_cn = side_cn & ~lone_mark;
        frame.side_ext = side_ext & ~lone_mark;

        sz_line_break_carry_t carry_next = carry;
        sz_line_break_window_t const win = sz_line_break_decide_window_(&frame, effective_class_byte, side_byte, carry,
                                                                        &carry_next, complete_limit, more_text);
        sz_u64_t const commit = win.breaks & sz_u64_mask_until_serial_(win.resolved);

        produced = sz_utf8_rune_drain_forward_serial_(commit, position, starts, lengths, produced, capacity,
                                                      &line_start);
        if (produced >= capacity) {
            if (bytes_consumed) *bytes_consumed = line_start;
            return produced;
        }

        sz_size_t const advance = win.resolved ? win.resolved : complete_limit;
        carry = carry_next;
        position += advance ? advance : loaded;
    }

    if (produced < capacity) starts[produced] = line_start, lengths[produced] = length - line_start, ++produced;
    if (bytes_consumed) *bytes_consumed = length;
    return produced;
}

#pragma endregion UAX 14 Line Boundaries forward kernel

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE2

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_LINEBREAKS_SVE2_H_
