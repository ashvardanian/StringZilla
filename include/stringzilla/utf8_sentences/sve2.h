/**
 *  @brief SVE2 (AArch64 scalable) backend for UAX-29 sentence boundaries.
 *  @file include/stringzilla/utf8_sentences/sve2.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_SENTENCES_SVE2_H_
#define STRINGZILLA_UTF8_SENTENCES_SVE2_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_sentences/tables.h"
#include "stringzilla/utf8_sentences/serial.h"
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

#pragma region UAX 29 Sentence Boundaries forward kernel

/** @brief  Sentence_Break class byte for one chunk of ASTRAL codepoints over the 20-bit offset = cp - 0x10000
 *          (5-nibble cascade), the SVE2 twin of @ref sz_utf8_sentence_break_astral_class_neon_. Per-lane bytes:
 *          @p plane = (offset>>16)&0xFF (low nibble meaningful), @p high = (offset>>8)&0xFF, @p low = offset&0xFF.
 *          Bit-exact with `sz_rune_sentence_break_property` over all astral planes. */
SZ_HELPER_AUTO svuint8_t sz_utf8_sentence_break_astral_class_sve2_(svuint8_t plane_u8x, svuint8_t high_u8x,
                                                                   svuint8_t low_u8x) {
    svbool_t const all_b8x = svptrue_b8();
    svuint8_t const n4_u8x = svand_n_u8_x(all_b8x, plane_u8x, 0x0F);
    svuint8_t const n3_u8x = svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, high_u8x, 4), 0x0F);
    svuint8_t const stage1_index_u8x = svorr_u8_x(all_b8x, svlsl_n_u8_x(all_b8x, n4_u8x, 4), n3_u8x);
    svuint8_t const page_u8x = sz_utf8_rune_lut_sve2_(sz_utf8_sentence_break_haswell_astral_stage1_, 256,
                                                      stage1_index_u8x);
    svuint8_t const n2_u8x = svand_n_u8_x(all_b8x, high_u8x, 0x0F);
    svuint8_t const leaf2_lo_u8x = sz_utf8_rune_cascade_sve2_(
        sz_utf8_sentence_break_haswell_astral_stage2_lo_,
        (int)sz_utf8_sentence_break_haswell_astral_stage2_lo_count_k / 16, page_u8x, n2_u8x);
    svuint8_t const n1_u8x = svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, low_u8x, 4), 0x0F);
    svuint8_t const leaf_lo_u8x = sz_utf8_rune_cascade_sve2_(
        sz_utf8_sentence_break_haswell_astral_stage3_lo_,
        (int)sz_utf8_sentence_break_haswell_astral_stage3_lo_count_k / 16, leaf2_lo_u8x, n1_u8x);
    svuint8_t const leaf_hi_u8x = sz_utf8_rune_cascade_sve2_(
        sz_utf8_sentence_break_haswell_astral_stage3_hi_,
        (int)sz_utf8_sentence_break_haswell_astral_stage3_hi_count_k / 16, leaf2_lo_u8x, n1_u8x);
    svuint8_t const n0_u8x = svand_n_u8_x(all_b8x, low_u8x, 0x0F);
    svuint8_t const leaf_group_u8x = svorr_u8_x(all_b8x,
                                                svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, leaf_lo_u8x, 4), 0x0F),
                                                svlsl_n_u8_x(all_b8x, leaf_hi_u8x, 4));
    svuint8_t const stage4_index_u8x = svorr_u8_x(
        all_b8x, svlsl_n_u8_x(all_b8x, svand_n_u8_x(all_b8x, leaf_lo_u8x, 0x0F), 4), n0_u8x);
    svuint8_t result_u8x = svdup_n_u8(0);
    for (int group = 0; group < (int)sz_utf8_sentence_break_haswell_astral_leaf_groups_k; ++group) {
        svuint8_t const value_u8x = sz_utf8_rune_lut_sve2_(
            sz_utf8_sentence_break_haswell_astral_stage4_groups_ + group * 256, 256, stage4_index_u8x);
        result_u8x = svsel_u8(svcmpeq_n_u8(all_b8x, leaf_group_u8x, (sz_u8_t)group), value_u8x, result_u8x);
    }
    return result_u8x;
}

/** @brief  Per-byte-lane Sentence_Break class for ONE window chunk, fully in-register - the SVE2 twin of
 *          @ref sz_utf8_sentence_break_classify_quarter_neon_. Raw lanes (ASCII, continuation bytes, `>= 0xF8`)
 *          keep `low = raw, high = 0` exactly like serial's blind decode; 2-/3-byte leads rebuild the BMP
 *          (high, low) pair from the peeked neighbours; 4-byte leads split by their blind plane between the BMP
 *          cascade (plane 0, overlong encodings), the astral cascade (planes 1..16), and class Other (planes over
 *          16, e.g. `F5..F7` leads). The class on non-start lanes is irrelevant - only start lanes compact. */
SZ_HELPER_AUTO svuint8_t sz_utf8_sentence_break_classify_chunk_sve2_(                   //
    svuint8_t bytes_u8x, svuint8_t next1_u8x, svuint8_t next2_u8x, svuint8_t next3_u8x, //
    svbool_t two_b8x, svbool_t three_b8x, svbool_t four_b8x) {
    svbool_t const all_b8x = svptrue_b8();

    // 2-byte: high = (b0 & 0x1F) >> 2, low = ((b0 & 0x03) << 6) | (b1 & 0x3F).
    svuint8_t const two_high_u8x = svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, bytes_u8x, 2), 0x07);
    svuint8_t const two_low_u8x = svorr_u8_x(all_b8x, svand_n_u8_x(all_b8x, svlsl_n_u8_x(all_b8x, bytes_u8x, 6), 0xC0),
                                             svand_n_u8_x(all_b8x, next1_u8x, 0x3F));
    // 3-byte: high = ((b0 & 0x0F) << 4) | ((b1 >> 2) & 0x0F), low = ((b1 & 0x03) << 6) | (b2 & 0x3F).
    svuint8_t const three_high_u8x = svorr_u8_x(all_b8x,
                                                svlsl_n_u8_x(all_b8x, svand_n_u8_x(all_b8x, bytes_u8x, 0x0F), 4),
                                                svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, next1_u8x, 2), 0x0F));
    svuint8_t const three_low_u8x = svorr_u8_x(all_b8x,
                                               svand_n_u8_x(all_b8x, svlsl_n_u8_x(all_b8x, next1_u8x, 6), 0xC0),
                                               svand_n_u8_x(all_b8x, next2_u8x, 0x3F));

    svuint8_t high_u8x = svdup_u8_z(two_b8x, 0); // raw lanes keep high = 0
    high_u8x = svsel_u8(two_b8x, two_high_u8x, high_u8x);
    high_u8x = svsel_u8(three_b8x, three_high_u8x, high_u8x);
    svuint8_t low_u8x = svsel_u8(two_b8x, two_low_u8x, bytes_u8x); // raw lanes keep low = raw
    low_u8x = svsel_u8(three_b8x, three_low_u8x, low_u8x);

    if (svptest_any(all_b8x, four_b8x)) {
        // 4-byte: cp = ((b0&7)<<18)|((b1&0x3F)<<12)|((b2&0x3F)<<6)|(b3&0x3F); the blind plane is cp bits [16,21).
        svuint8_t const four_low_u8x = svorr_u8_x(all_b8x,
                                                  svand_n_u8_x(all_b8x, svlsl_n_u8_x(all_b8x, next2_u8x, 6), 0xC0),
                                                  svand_n_u8_x(all_b8x, next3_u8x, 0x3F));
        svuint8_t const four_high_u8x = svorr_u8_x(all_b8x,
                                                   svand_n_u8_x(all_b8x, svlsl_n_u8_x(all_b8x, next1_u8x, 4), 0xF0),
                                                   svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, next2_u8x, 2), 0x0F));
        low_u8x = svsel_u8(four_b8x, four_low_u8x, low_u8x);
        high_u8x = svsel_u8(four_b8x, four_high_u8x, high_u8x);

        svuint8_t const plane_u8x = svorr_u8_x(all_b8x,
                                               svand_n_u8_x(all_b8x, svlsl_n_u8_x(all_b8x, bytes_u8x, 2), 0x1C),
                                               svand_n_u8_x(all_b8x, svlsr_n_u8_x(all_b8x, next1_u8x, 4), 0x03));
        svbool_t const plane_nonzero_b8x = svand_b_z(all_b8x, four_b8x, svcmpgt_n_u8(all_b8x, plane_u8x, 0));
        svbool_t const plane_le16_b8x = svcmple_n_u8(all_b8x, plane_u8x, 0x10);
        svbool_t const is_astral_b8x = svand_b_z(all_b8x, plane_nonzero_b8x, plane_le16_b8x);
        svbool_t const is_overrange_b8x = svbic_b_z(all_b8x, plane_nonzero_b8x, plane_le16_b8x);
        svuint8_t const plane_off_u8x = svsub_n_u8_x(all_b8x, svsel_u8(is_astral_b8x, plane_u8x, svdup_n_u8(1)), 1);
        svuint8_t const bmp_u8x = sz_utf8_rune_flat_lookup_ascii_gated_sve2_(
            sz_utf8_sentence_break_bmp_page_lut_, sz_utf8_sentence_break_flat_bmp_, bytes_u8x, high_u8x, low_u8x);
        svuint8_t const astral_u8x = sz_utf8_sentence_break_astral_class_sve2_(plane_off_u8x, high_u8x, low_u8x);
        svuint8_t const classed_u8x = svsel_u8(is_astral_b8x, astral_u8x, bmp_u8x);
        return svsel_u8(is_overrange_b8x, svdup_n_u8(0), classed_u8x); // cp >= 0x110000 lanes force class Other
    }
    return sz_utf8_rune_flat_lookup_ascii_gated_sve2_(sz_utf8_sentence_break_bmp_page_lut_,
                                                      sz_utf8_sentence_break_flat_bmp_, bytes_u8x, high_u8x, low_u8x);
}

/** @brief  Build the per-class membership frame from the dense class byte stream via its four bit-planes: each
 *          plane lowers to a u64 with the shared predicate bridge, and the fifteen class masks assemble from the
 *          planes with scalar mask algebra - 4 predicate compares per dense vector instead of 15. */
SZ_HELPER_AUTO sz_utf8_sentence_break_frame_t sz_utf8_sentence_break_frame_sve2_(sz_u8_t const *dense_classes,
                                                                                 sz_size_t count) {
    sz_size_t const vector_bytes = svcntb() < 64 ? svcntb() : 64;
    sz_u64_t planes[4] = {0, 0, 0, 0};
    for (sz_size_t base = 0; base < count; base += vector_bytes) {
        svbool_t const loaded_b8x = svwhilelt_b8_u64((sz_u64_t)base, (sz_u64_t)count);
        svuint8_t const classes_u8x = svld1_u8(loaded_b8x, dense_classes + base);
        planes[0] |=
            sz_utf8_rune_pred_to_u64_sve2_(svcmpne_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, classes_u8x, 1), 0))
            << base;
        planes[1] |=
            sz_utf8_rune_pred_to_u64_sve2_(svcmpne_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, classes_u8x, 2), 0))
            << base;
        planes[2] |=
            sz_utf8_rune_pred_to_u64_sve2_(svcmpne_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, classes_u8x, 4), 0))
            << base;
        planes[3] |=
            sz_utf8_rune_pred_to_u64_sve2_(svcmpne_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, classes_u8x, 8), 0))
            << base;
    }
    sz_u64_t const valid = (count >= 64) ? ~0ull : ((1ull << count) - 1);
    sz_utf8_sentence_break_frame_t frame;
    for (int cls = 0; cls < 15; ++cls)
        frame.by_class[cls] = ((cls & 1) ? planes[0] : ~planes[0]) & ((cls & 2) ? planes[1] : ~planes[1]) &
                              ((cls & 4) ? planes[2] : ~planes[2]) & ((cls & 8) ? planes[3] : ~planes[3]) & valid;
    return frame;
}

/**
 *  @brief  Forward UAX-29 sentence segmentation kernel (SVE2, vector-length agnostic). Bit-exact with
 *          `sz_utf8_sentences_serial` and the other ISA fronts: a chunked-window classify + dense-compaction
 *          front-end feeds the shared portable rule engine @ref sz_utf8_sentence_break_decide_block_, whose dense
 *          break bits map back to byte positions in one walk over the codepoint-start lanes.
 *
 *  Each 64-byte window streams as `64 / svcntb()` register chunks with one peeked vector ahead: `svext` supplies
 *  the classifier's forward neighbours across chunk edges, and the per-chunk start-lane classes compact straight
 *  into the dense class stream the engine consumes (one `svcompact_u32` + truncating store per 32-bit quarter),
 *  so the only memory the window touches is that stream - the shared engine's own input format.
 */
SZ_API_COMPTIME sz_size_t sz_utf8_sentences_sve2(            //
    sz_cptr_t text, sz_size_t length,                        //
    sz_size_t *sentence_starts, sz_size_t *sentence_lengths, //
    sz_size_t sentences_capacity, sz_size_t *bytes_consumed) {

    sz_size_t sentences = 0;
    if (length == 0 || sentences_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const vector_bytes = svcntb();
    sz_size_t const chunk_bytes = vector_bytes < 64 ? vector_bytes : 64;
    sz_size_t const window_capacity = (64 / chunk_bytes) * chunk_bytes;

    sz_size_t sentence_start = 0;
    sz_size_t position = 0;

    sz_utf8_sentence_break_carry_t carry;
    carry.have_prev = 0;
    carry.prev_raw = carry.prev_eff = carry.prev_prev_eff = (sz_u8_t)sz_sentence_break_other_k;
    carry.in_shadow = carry.shadow_aterm = carry.shadow_saw_sp = 0;
    carry.sb8_pending = 0;

    sz_size_t sb8_pending_position = 0;
    int sb8_pending_active = 0;

    while (position < length) {
        sz_size_t const available = length - position;
        sz_size_t const loaded = available < window_capacity ? available : window_capacity;
        sz_bool_t const more_text = (position + loaded < length) ? sz_true_k : sz_false_k;
        sz_u8_t const byte_after = more_text ? text_u8[position + loaded] : 0;

        // Stream the window chunk by chunk: classify every byte lane, lower the start masks, and compact the
        // start-lane classes into the dense stream. The trailing-lead trim only ever falls in the last chunk, so
        // its dense lanes are the only ones the `complete_limit` cut can exclude.
        sz_u64_t start_bytes = (position == 0) ? 1ull : 0ull; // SB1: the first byte opens a codepoint
        sz_u64_t compacted_bytes = 0;
        sz_u8_t dense_classes[64];
        sz_size_t dense_count = 0;
        sz_size_t complete_limit = loaded;
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

            svbool_t const continuation_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xC0), 0x80);
            svbool_t starts_b8x = svbic_b_z(loaded_b8x, loaded_b8x, continuation_b8x);
            svbool_t const two_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xE0), 0xC0);
            svbool_t const three_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xF0), 0xE0);
            svbool_t const four_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(loaded_b8x, bytes_u8x, 0xF8), 0xF0);
            start_bytes |= sz_utf8_rune_pred_to_u64_sve2_(starts_b8x) << chunk_base;

            svbool_t compact_b8x = starts_b8x;
            if (position == 0 && chunk_base == 0)
                compact_b8x = svorr_b_z(loaded_b8x, compact_b8x,
                                        svcmpeq_n_u8(loaded_b8x, svindex_u8(0, 1), 0)); // forced lead continuation
            if (ahead <= chunk_bytes) { // the window edge lives in this chunk: apply the complete-limit trim
                sz_u64_t const chunk_two = sz_utf8_rune_pred_to_u64_sve2_(two_b8x) << chunk_base;
                sz_u64_t const chunk_three = sz_utf8_rune_pred_to_u64_sve2_(three_b8x) << chunk_base;
                sz_u64_t const chunk_four = sz_utf8_rune_pred_to_u64_sve2_(four_b8x) << chunk_base;
                complete_limit = sz_utf8_sentence_break_complete_limit_masks_(
                    loaded, start_bytes, chunk_two, chunk_three, chunk_four, byte_after, more_text);
                if (complete_limit < loaded && complete_limit > chunk_base)
                    compact_b8x = svand_b_z(loaded_b8x, compact_b8x,
                                            svwhilelt_b8_u64(0, (sz_u64_t)(complete_limit - chunk_base)));
                else if (complete_limit <= chunk_base) compact_b8x = svpfalse_b();
            }
            compacted_bytes |= sz_utf8_rune_pred_to_u64_sve2_(compact_b8x) << chunk_base;

            svuint8_t const classes_u8x = sz_utf8_sentence_break_classify_chunk_sve2_(
                bytes_u8x, next1_u8x, next2_u8x, next3_u8x, two_b8x, three_b8x, four_b8x);

            // Dense compaction: one `svcompact_u32` per quarter packs the start-lane classes, and the truncating
            // store appends them to the dense stream the shared engine consumes.
            svuint8_t const compact_flags_u8x = svdup_u8_z(compact_b8x, 1);
            svuint16_t const flags_lo_u16x = svunpklo_u16(compact_flags_u8x),
                             flags_hi_u16x = svunpkhi_u16(compact_flags_u8x);
            svuint16_t const class_lo_u16x = svunpklo_u16(classes_u8x), class_hi_u16x = svunpkhi_u16(classes_u8x);
            for (int quarter = 0; quarter < 4; ++quarter) {
                int const high_half = quarter & 2, high_quarter = quarter & 1;
                svuint32_t const flags_q_u32x = sz_utf8_rune_quarter_words_sve2_(flags_lo_u16x, flags_hi_u16x,
                                                                                 high_half, high_quarter);
                svbool_t const emit_b32x = svcmpne_n_u32(svptrue_b32(), flags_q_u32x, 0);
                sz_size_t const found = svcntp_b32(svptrue_b32(), emit_b32x);
                if (found == 0) continue;
                svuint32_t const class_q_u32x = sz_utf8_rune_quarter_words_sve2_(class_lo_u16x, class_hi_u16x,
                                                                                 high_half, high_quarter);
                svst1b_u32(svwhilelt_b32_u64(0, (sz_u64_t)found), dense_classes + dense_count,
                           svcompact_u32(emit_b32x, class_q_u32x));
                dense_count += found;
            }
        }
        sz_u64_t const complete_mask = sz_u64_mask_until_serial_(complete_limit);
        sz_u64_t const dense_start_lanes = start_bytes & complete_mask;
        // A deep trim (byte-after-continuation rule reaching a last lead in an earlier chunk) can leave exactly
        // one already-compacted trailing entry past the limit - no start can follow the last lead - so drop it.
        if (compacted_bytes & ~complete_mask) --dense_count;

        sz_utf8_sentence_break_carry_t carry_full = carry;
        sz_utf8_sentence_break_frame_t const frame = sz_utf8_sentence_break_frame_sve2_(dense_classes, dense_count);
        sz_utf8_sentence_break_window_t const win = sz_utf8_sentence_break_decide_block_(
            &frame, dense_classes, dense_count, &carry_full, more_text);

        // Resolve a previously deferred SB8 boundary before any of this window's boundaries.
        if (sb8_pending_active && win.sb8_resolution != 0) {
            if (win.sb8_resolution == 1) {
                if (sentences == sentences_capacity) {
                    if (bytes_consumed) *bytes_consumed = sentence_start;
                    return sentences;
                }
                sentence_starts[sentences] = sentence_start;
                sentence_lengths[sentences] = sb8_pending_position - sentence_start;
                ++sentences;
                sentence_start = sb8_pending_position;
            }
            sb8_pending_active = 0;
        }

        // Map the trusted dense breaks back to byte positions in one walk over the start lanes, emitting each
        // closed segment; the same walk reports the resume byte of a partially resolved window.
        sz_size_t const dense_adv = win.resolved < dense_count ? win.resolved : dense_count;
        sz_u64_t const dense_breaks = win.breaks & sz_u64_mask_until_serial_(dense_adv);
        sz_size_t advance_lane = 0;
        sentences = sz_utf8_sentence_break_emit_dense_serial_(
            dense_start_lanes, dense_breaks, dense_adv, position, !carry.have_prev,
            complete_limit ? complete_limit : loaded, sentence_starts, sentence_lengths, sentences, sentences_capacity,
            &sentence_start, &advance_lane);
        if (sentences == sentences_capacity) {
            if (bytes_consumed) *bytes_consumed = sentence_start;
            return sentences;
        }

        if (dense_adv >= dense_count) {
            carry = carry_full;
            position += complete_limit ? complete_limit : loaded;
        }
        else if (dense_adv > 0) {
            sz_utf8_sentence_break_carry_t carry_to_edge = carry;
            sz_utf8_sentence_break_decide_dense_(dense_classes, dense_adv, &carry_to_edge, sz_true_k);
            carry = carry_to_edge;
            position += advance_lane;
        }
        else {
            if (!sb8_pending_active) {
                sb8_pending_active = 1;
                sb8_pending_position = position;
            }
            carry = carry_full;
            carry.sb8_pending = 1;
            position += complete_limit ? complete_limit : loaded;
        }
    }

    if (sb8_pending_active) {
        if (sentences == sentences_capacity) {
            if (bytes_consumed) *bytes_consumed = sentence_start;
            return sentences;
        }
        sentence_starts[sentences] = sentence_start;
        sentence_lengths[sentences] = sb8_pending_position - sentence_start;
        ++sentences;
        sentence_start = sb8_pending_position;
    }

    if (sentences == sentences_capacity) {
        if (bytes_consumed) *bytes_consumed = sentence_start;
        return sentences;
    }
    sentence_starts[sentences] = sentence_start;
    sentence_lengths[sentences] = length - sentence_start;
    ++sentences;
    if (bytes_consumed) *bytes_consumed = length;
    return sentences;
}

#pragma endregion UAX 29 Sentence Boundaries forward kernel

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE2

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_SENTENCES_SVE2_H_
