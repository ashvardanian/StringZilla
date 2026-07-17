/**
 *  @brief SVE2 backend for UTF-8 newline and whitespace delimiter scanning.
 *  @file include/stringzilla/utf8_tokens/sve2.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_TOKENS_SVE2_H_
#define STRINGZILLA_UTF8_TOKENS_SVE2_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_tokens/serial.h"
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

/** @brief  Drain the first @p emit_count match starts within the first @p span byte lanes into absolute
 *          `(offset, length)` pairs, one 32-bit quarter at a time - the `(offset, length)` twin of
 *          @ref sz_utf8_rune_drain_sve2_.
 *
 *  SVE2 has no `svcompact_u8`, so each quarter of the byte-granular start lanes widens to 32-bit (the lane index
 *  rides a biased iota) and one `svcompact_u32` packs the start indices while a lockstep `svcompact_u32` packs
 *  their byte lengths; the offsets turn absolute only after the 64-bit widening, so multi-gigabyte inputs never
 *  truncate. The caller's `while` loop resumes past the last emitted match when the capacity cuts the tile. */
SZ_HELPER_AUTO void sz_utf8_token_drain_sve2_(                                  //
    svbool_t starts, svuint8_t lengths_u8x, sz_size_t position, sz_size_t span, //
    sz_size_t emit_count, sz_size_t *match_offsets, sz_size_t *match_lengths) {

    svbool_t const all_b32x = svptrue_b32();
    svbool_t const all_b64x = svptrue_b64();
    sz_size_t const quarter_lanes = svcntw();
    sz_size_t const half_lanes = svcntd();
    svbool_t const span_b8x = svwhilelt_b8_u64(0, (sz_u64_t)span);
    svuint8_t const start_flags_u8x = svdup_u8_z(svand_b_z(span_b8x, starts, span_b8x), 1);
    svuint16_t const flags_low_u16x = svunpklo_u16(start_flags_u8x), flags_high_u16x = svunpkhi_u16(start_flags_u8x);
    svuint16_t const lengths_low_u16x = svunpklo_u16(lengths_u8x), lengths_high_u16x = svunpkhi_u16(lengths_u8x);

    sz_size_t produced = 0;
    for (int quarter = 0; quarter < 4 && produced < emit_count; ++quarter) {
        int const high_half = quarter & 2, high_quarter = quarter & 1;
        svuint32_t const flags_u32x = sz_utf8_rune_quarter_words_sve2_(flags_low_u16x, flags_high_u16x, high_half,
                                                                       high_quarter);
        svbool_t const emit_b32x = svcmpne_n_u32(all_b32x, flags_u32x, 0);
        sz_size_t const found = svcntp_b32(all_b32x, emit_b32x);
        if (found == 0) continue;
        sz_size_t const want = sz_min_of_two(found, emit_count - produced);
        svuint32_t const packed_lanes_u32x = svcompact_u32(emit_b32x,
                                                           svindex_u32((sz_u32_t)(quarter * quarter_lanes), 1));
        svuint32_t const packed_lengths_u32x = svcompact_u32(
            emit_b32x, sz_utf8_rune_quarter_words_sve2_(lengths_low_u16x, lengths_high_u16x, high_half, high_quarter));
        svbool_t const store_low_b64x = svwhilelt_b64_u64(0, (sz_u64_t)want);
        svst1_u64(store_low_b64x, (sz_u64_t *)(match_offsets + produced),
                  svadd_n_u64_x(all_b64x, svunpklo_u64(packed_lanes_u32x), (sz_u64_t)position));
        svst1_u64(store_low_b64x, (sz_u64_t *)(match_lengths + produced), svunpklo_u64(packed_lengths_u32x));
        if (want > half_lanes) {
            svbool_t const store_high_b64x = svwhilelt_b64_u64((sz_u64_t)half_lanes, (sz_u64_t)want);
            svst1_u64(store_high_b64x, (sz_u64_t *)(match_offsets + produced + half_lanes),
                      svadd_n_u64_x(all_b64x, svunpkhi_u64(packed_lanes_u32x), (sz_u64_t)position));
            svst1_u64(store_high_b64x, (sz_u64_t *)(match_lengths + produced + half_lanes),
                      svunpkhi_u64(packed_lengths_u32x));
        }
        produced += want;
    }
}

SZ_API_COMPTIME sz_size_t sz_utf8_newlines_sve2(        //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const step = svcntb();
    // Too narrow to host a safe 3-byte straddle and a 32-bit compaction sub-block? Delegate wholesale to serial.
    if (step < 16)
        return sz_utf8_newlines_serial(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);

    // Fixed logical tile (64 bytes when the register is wide enough), capped by the register width so the
    // shifted views for 2-/3-byte delimiters are always fully loaded on a full tile.
    sz_size_t const tile = step < 64 ? step : 64;
    sz_size_t count = 0, position = 0;

    // Same delimiter set & branchless mask shape as the single-tile scan, but every multi-byte predicate is
    // computed unconditionally and accumulated into a per-lane start predicate + length vector.
    svuint8_t const one_byte_set_u8x = svdupq_n_u8('\n', '\v', '\f', '\r', '\n', '\n', '\n', '\n', '\n', '\n', '\n',
                                                   '\n', '\n', '\n', '\n', '\n');
    while (position < length && count < matches_capacity) {
        // Full-tile scan: the shifted 2nd/3rd-byte views splice one peeked vector across the tile edge, and a final
        // partial tile reads zero past the end, so a truncated multi-byte delimiter at EOF can never match - and a
        // CRLF can never straddle unseen. At vector lengths wider than the tile the peek is empty; the shifted
        // views stay inside the loaded register.
        sz_size_t const remaining = length - position;
        sz_size_t const usable = sz_min_of_two(tile, remaining);
        sz_size_t const load_span = sz_min_of_two(step, remaining);
        sz_size_t const beyond_tile = remaining > tile ? remaining - tile : 0;
        sz_size_t const peek_span = step == tile ? sz_min_of_two(tile, beyond_tile) : 0;
        svbool_t const pg_b8x = svwhilelt_b8_u64(0, (sz_u64_t)usable);
        svuint8_t const text0_u8x = svld1_u8(svwhilelt_b8_u64(0, (sz_u64_t)load_span), text_u8 + position);
        svuint8_t const peek_u8x = svld1_u8(svwhilelt_b8_u64(0, (sz_u64_t)peek_span),
                                            text_u8 + position + (peek_span ? tile : 0));
        svuint8_t const text1_u8x = svext_u8(text0_u8x, peek_u8x, 1);
        svuint8_t const text2_u8x = svext_u8(text0_u8x, peek_u8x, 2);

        // 1-byte: \n \v \f \r
        svbool_t one_byte_mask_b8x = svmatch_u8(pg_b8x, text0_u8x, one_byte_set_u8x);
        // 2-byte: C2 85 (NEL); CRLF is a single 2-byte match.
        svbool_t cr_mask_b8x = svcmpeq_n_u8(pg_b8x, text0_u8x, '\r');
        svbool_t lf_mask_b8x = svcmpeq_n_u8(pg_b8x, text0_u8x, '\n');
        svbool_t crlf_mask_b8x = svand_b_z(pg_b8x, cr_mask_b8x, svcmpeq_n_u8(pg_b8x, text1_u8x, '\n'));
        svbool_t nel_mask_b8x = svand_b_z(pg_b8x, svcmpeq_n_u8(pg_b8x, text0_u8x, 0xC2),
                                          svcmpeq_n_u8(pg_b8x, text1_u8x, 0x85));
        svbool_t two_byte_mask_b8x = svorr_b_z(pg_b8x, crlf_mask_b8x, nel_mask_b8x);
        // 3-byte: E2 80 A8 (LS), E2 80 A9 (PS)
        svbool_t lead_e280_mask_b8x = svand_b_z(pg_b8x, svcmpeq_n_u8(pg_b8x, text0_u8x, 0xE2),
                                                svcmpeq_n_u8(pg_b8x, text1_u8x, 0x80));
        svbool_t three_byte_mask_b8x = svand_b_z(
            pg_b8x, lead_e280_mask_b8x,
            svorr_b_z(pg_b8x, svcmpeq_n_u8(pg_b8x, text2_u8x, 0xA8), svcmpeq_n_u8(pg_b8x, text2_u8x, 0xA9)));

        // A CR that completes a CRLF must be emitted once (len 2); the trailing LF must NOT also be a match.
        // `crlf_mask` marks the CR lane; the LF to suppress is the next lane. Up-shift a 0/1 CRLF flag by one
        // lane (svinsr inserts 0 at lane 0 and slides everything toward higher indices) to land on that LF.
        svuint8_t crlf_flag_u8x = svdup_u8_z(crlf_mask_b8x, 1);
        svbool_t lf_of_crlf_mask_b8x = svand_b_z(pg_b8x, lf_mask_b8x,
                                                 svcmpne_n_u8(pg_b8x, svinsr_n_u8(crlf_flag_u8x, 0), 0));
        // The peeked `text1` view sees every CRLF whole, so no LF can arrive already consumed by a previous tile.
        svbool_t starts_b8x = svbic_b_z(
            pg_b8x, svorr_b_z(pg_b8x, one_byte_mask_b8x, svorr_b_z(pg_b8x, two_byte_mask_b8x, three_byte_mask_b8x)),
            lf_of_crlf_mask_b8x);

        // Per-lane length: 1 + (2-byte) + 2*(3-byte); the masks are disjoint.
        svuint8_t lengths_u8x = svdup_n_u8(1);
        lengths_u8x = svadd_n_u8_m(two_byte_mask_b8x, lengths_u8x, 1);
        lengths_u8x = svadd_n_u8_m(three_byte_mask_b8x, lengths_u8x, 2);

        sz_size_t const span_matches = svcntp_b8(pg_b8x, starts_b8x);
        sz_size_t const emit_count = sz_min_of_two(span_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_token_drain_sve2_(starts_b8x, lengths_u8x, position, usable, emit_count, match_offsets + count,
                                      match_lengths + count);
        count += emit_count;
        // The whole span is scanned, so skip it wholesale unless the capacity cut the tile short - then resume
        // past the last emitted match; a trailing match may extend beyond the span edge and wins the `max`.
        sz_size_t const span_end = position + usable;
        if (emit_count) {
            sz_size_t const last_end = match_offsets[count - 1] + match_lengths[count - 1];
            // On a capacity-limited return `bytes_consumed` must be the last emitted match's end - callers treat
            // it as the resume boundary - so the full-span skip applies only while the scan continues.
            int const skip_whole_span = emit_count == span_matches && count < matches_capacity && span_end > last_end;
            position = skip_whole_span ? span_end : last_end;
        }
        else position = span_end;
        if (count == matches_capacity) break; // output buffer full: `position` already past the last match
    }

    if (bytes_consumed) *bytes_consumed = position;
    return count;
}

SZ_API_COMPTIME sz_size_t sz_utf8_whitespaces_sve2(     //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const step = svcntb();
    if (step < 16)
        return sz_utf8_whitespaces_serial(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);

    sz_size_t const tile = step < 64 ? step : 64;
    sz_size_t count = 0, position = 0;

    svuint8_t const one_byte_set_u8x = svdupq_n_u8(' ', '\t', '\n', '\v', '\f', '\r', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                                                   ' ', ' ', ' ');
    // Every multi-byte whitespace begins with one of these leads; one `svmatch` gates the whole multi-byte block.
    svuint8_t const multi_byte_leads_u8x = svdupq_n_u8(0xC2, 0xE1, 0xE2, 0xE3, 0xC2, 0xC2, 0xC2, 0xC2, 0xC2, 0xC2, 0xC2,
                                                       0xC2, 0xC2, 0xC2, 0xC2, 0xC2);
    // Valid third bytes for E2 80 XX: U+2000-U+200A (0x80-0x8A), plus U+2028 (0xA8), U+2029 (0xA9). U+200B/200C/200D
    // are Format chars, not White_Space; the 16-lane set test repeats 0x80 (already a member) in their freed slots.
    svuint8_t const e280_third_bytes_u8x = svdupq_n_u8(0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A,
                                                       0x80, 0x80, 0x80, 0xA8, 0xA9);

    while (position < length && count < matches_capacity) {
        // Full-tile scan: the shifted 2nd/3rd-byte views splice one peeked vector across the tile edge, and a final
        // partial tile reads zero past the end, so a truncated multi-byte delimiter at EOF can never match. At
        // vector lengths wider than the tile the peek is empty - the shifted views stay inside the loaded register.
        sz_size_t const remaining = length - position;
        sz_size_t const usable = sz_min_of_two(tile, remaining);
        sz_size_t const load_span = sz_min_of_two(step, remaining);
        sz_size_t const beyond_tile = remaining > tile ? remaining - tile : 0;
        sz_size_t const peek_span = step == tile ? sz_min_of_two(tile, beyond_tile) : 0;
        svbool_t const pg_b8x = svwhilelt_b8_u64(0, (sz_u64_t)usable);
        svuint8_t const text0_u8x = svld1_u8(svwhilelt_b8_u64(0, (sz_u64_t)load_span), text_u8 + position);
        svuint8_t const peek_u8x = svld1_u8(svwhilelt_b8_u64(0, (sz_u64_t)peek_span),
                                            text_u8 + position + (peek_span ? tile : 0));
        svuint8_t const text1_u8x = svext_u8(text0_u8x, peek_u8x, 1);
        svuint8_t const text2_u8x = svext_u8(text0_u8x, peek_u8x, 2);

        // 1-byte: ' ' \t \n \v \f \r
        svbool_t starts_b8x = svmatch_u8(pg_b8x, text0_u8x, one_byte_set_u8x);
        svuint8_t lengths_u8x = svdup_n_u8(1);
        if (svptest_any(pg_b8x, svmatch_u8(pg_b8x, text0_u8x, multi_byte_leads_u8x))) {
            // 2-byte: C2 {85, A0} (NEL, NBSP). NO CRLF merging for whitespace.
            svbool_t const lead_c2_mask_b8x = svcmpeq_n_u8(pg_b8x, text0_u8x, 0xC2);
            svbool_t const two_byte_mask_b8x = svand_b_z(
                pg_b8x, lead_c2_mask_b8x,
                svorr_b_z(pg_b8x, svcmpeq_n_u8(pg_b8x, text1_u8x, 0x85), svcmpeq_n_u8(pg_b8x, text1_u8x, 0xA0)));
            // 3-byte: E1 9A 80 (Ogham)
            svbool_t const ogham_mask_b8x = svand_b_z(
                pg_b8x, svand_b_z(pg_b8x, svcmpeq_n_u8(pg_b8x, text0_u8x, 0xE1), svcmpeq_n_u8(pg_b8x, text1_u8x, 0x9A)),
                svcmpeq_n_u8(pg_b8x, text2_u8x, 0x80));
            // 3-byte: E2 80 [80-8A] | A8 | A9 (set match); E2 80 AF (NNBSP); E2 81 9F (MMSP)
            svbool_t const lead_e2_mask_b8x = svcmpeq_n_u8(pg_b8x, text0_u8x, 0xE2);
            svbool_t const lead_e280_mask_b8x = svand_b_z(pg_b8x, lead_e2_mask_b8x,
                                                          svcmpeq_n_u8(pg_b8x, text1_u8x, 0x80));
            svbool_t const e280xx_mask_b8x = svand_b_z(pg_b8x, lead_e280_mask_b8x,
                                                       svmatch_u8(pg_b8x, text2_u8x, e280_third_bytes_u8x));
            svbool_t const nnbsp_mask_b8x = svand_b_z(pg_b8x, lead_e280_mask_b8x,
                                                      svcmpeq_n_u8(pg_b8x, text2_u8x, 0xAF));
            svbool_t const mmsp_mask_b8x = svand_b_z(
                pg_b8x, svand_b_z(pg_b8x, lead_e2_mask_b8x, svcmpeq_n_u8(pg_b8x, text1_u8x, 0x81)),
                svcmpeq_n_u8(pg_b8x, text2_u8x, 0x9F));
            // 3-byte: E3 80 80 (Ideographic)
            svbool_t const ideographic_mask_b8x = svand_b_z(
                pg_b8x, svand_b_z(pg_b8x, svcmpeq_n_u8(pg_b8x, text0_u8x, 0xE3), svcmpeq_n_u8(pg_b8x, text1_u8x, 0x80)),
                svcmpeq_n_u8(pg_b8x, text2_u8x, 0x80));

            svbool_t const three_byte_mask_b8x = svorr_b_z(
                pg_b8x, svorr_b_z(pg_b8x, ogham_mask_b8x, svorr_b_z(pg_b8x, e280xx_mask_b8x, nnbsp_mask_b8x)),
                svorr_b_z(pg_b8x, mmsp_mask_b8x, ideographic_mask_b8x));
            starts_b8x = svorr_b_z(pg_b8x, starts_b8x, svorr_b_z(pg_b8x, two_byte_mask_b8x, three_byte_mask_b8x));

            // Per-lane length: 1 + (2-byte) + 2*(3-byte); the masks are disjoint.
            lengths_u8x = svadd_n_u8_m(two_byte_mask_b8x, lengths_u8x, 1);
            lengths_u8x = svadd_n_u8_m(three_byte_mask_b8x, lengths_u8x, 2);
        }

        sz_size_t const span_matches = svcntp_b8(pg_b8x, starts_b8x);
        sz_size_t const emit_count = sz_min_of_two(span_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_token_drain_sve2_(starts_b8x, lengths_u8x, position, usable, emit_count, match_offsets + count,
                                      match_lengths + count);
        count += emit_count;
        // The whole span is scanned, so skip it wholesale unless the capacity cut the tile short - then resume
        // past the last emitted match; a trailing match may extend beyond the span edge and wins the `max`.
        sz_size_t const span_end = position + usable;
        if (emit_count) {
            sz_size_t const last_end = match_offsets[count - 1] + match_lengths[count - 1];
            // On a capacity-limited return `bytes_consumed` must be the last emitted match's end - callers treat
            // it as the resume boundary - so the full-span skip applies only while the scan continues.
            int const skip_whole_span = emit_count == span_matches && count < matches_capacity && span_end > last_end;
            position = skip_whole_span ? span_end : last_end;
        }
        else position = span_end;
        if (count == matches_capacity) break; // output buffer full: `position` already past the last match
    }

    if (bytes_consumed) *bytes_consumed = position;
    return count;
}

/** @brief  One widened quarter of the BMP delimiter membership: `bmp_block_[high]` selects a 32-byte bitmap row,
 *          then `bmp_bitmaps_[block * 32 + (low >> 3)]` is the bitmap byte, both stages by `LD1B` byte gathers -
 *          the delimiter twin of @ref sz_utf8_rune_flat_lookup_quarter_sve2_. Every offset is in-bounds by
 *          construction: block ids index real rows and `(low >> 3) < 32`. */
SZ_HELPER_INLINE svuint32_t sz_utf8_delimiter_bmp_quarter_sve2_(svuint32_t high_u32x, svuint32_t low_u32x) {
    svbool_t const all_b32x = svptrue_b32();
    svuint32_t const block_u32x = svld1ub_gather_u32offset_u32(all_b32x, sz_utf8_delimiter_bmp_block_, high_u32x);
    svuint32_t const bitmap_offsets_u32x = svadd_u32_x(all_b32x, svlsl_n_u32_x(all_b32x, block_u32x, 5),
                                                       svlsr_n_u32_x(all_b32x, low_u32x, 3));
    return svld1ub_gather_u32offset_u32(all_b32x, sz_utf8_delimiter_bmp_bitmaps_, bitmap_offsets_u32x);
}

/** @brief  One widened quarter of the ASTRAL delimiter membership over `offset = codepoint - 0x10000`: the L1
 *          group by @p super (a 16-entry `svtbl` row), the bitmap row id by an `astral_l2_` gather at
 *          `group * 256 + sub`, and the bitmap byte by an `astral_bitmaps_` gather. */
SZ_HELPER_INLINE svuint32_t sz_utf8_delimiter_astral_quarter_sve2_(svuint32_t super_u32x, svuint32_t sub_u32x,
                                                                   svuint32_t low8_u32x) {
    svbool_t const all_b32x = svptrue_b32();
    svuint32_t const group_u32x = svld1ub_gather_u32offset_u32(all_b32x, sz_utf8_delimiter_astral_l1_, super_u32x);
    svuint32_t const row_offsets_u32x = svadd_u32_x(all_b32x, svlsl_n_u32_x(all_b32x, group_u32x, 8), sub_u32x);
    svuint32_t const row_u32x = svld1ub_gather_u32offset_u32(all_b32x, sz_utf8_delimiter_astral_l2_, row_offsets_u32x);
    svuint32_t const bitmap_offsets_u32x = svadd_u32_x(all_b32x, svlsl_n_u32_x(all_b32x, row_u32x, 5),
                                                       svlsr_n_u32_x(all_b32x, low8_u32x, 3));
    return svld1ub_gather_u32offset_u32(all_b32x, sz_utf8_delimiter_astral_bitmaps_, bitmap_offsets_u32x);
}

/** @brief  Bitmap byte per lane for a two-input byte-domain quarter walk: widens @p first_u8x / @p second_u8x
 *          (and @p third_u8x for the astral form when non-NULL semantics apply) and narrows the gathered bytes
 *          back into one byte-lane vector, the delimiter twin of @ref sz_utf8_rune_flat_lookup_sve2_. */
SZ_HELPER_AUTO svuint8_t sz_utf8_delimiter_bmp_bitmap_sve2_(svuint8_t high_u8x, svuint8_t low_u8x) {
    svuint16_t const high_low_half_u16x = svunpklo_u16(high_u8x), high_high_half_u16x = svunpkhi_u16(high_u8x);
    svuint16_t const low_low_half_u16x = svunpklo_u16(low_u8x), low_high_half_u16x = svunpkhi_u16(low_u8x);
    svuint32_t const first_u32x = sz_utf8_delimiter_bmp_quarter_sve2_(svunpklo_u32(high_low_half_u16x),
                                                                      svunpklo_u32(low_low_half_u16x));
    svuint32_t const second_u32x = sz_utf8_delimiter_bmp_quarter_sve2_(svunpkhi_u32(high_low_half_u16x),
                                                                       svunpkhi_u32(low_low_half_u16x));
    svuint32_t const third_u32x = sz_utf8_delimiter_bmp_quarter_sve2_(svunpklo_u32(high_high_half_u16x),
                                                                      svunpklo_u32(low_high_half_u16x));
    svuint32_t const fourth_u32x = sz_utf8_delimiter_bmp_quarter_sve2_(svunpkhi_u32(high_high_half_u16x),
                                                                       svunpkhi_u32(low_high_half_u16x));
    svuint16_t const packed_low_u16x = svuzp1_u16(svreinterpret_u16_u32(first_u32x),
                                                  svreinterpret_u16_u32(second_u32x));
    svuint16_t const packed_high_u16x = svuzp1_u16(svreinterpret_u16_u32(third_u32x),
                                                   svreinterpret_u16_u32(fourth_u32x));
    return svuzp1_u8(svreinterpret_u8_u16(packed_low_u16x), svreinterpret_u8_u16(packed_high_u16x));
}

/** @copydoc sz_utf8_delimiter_bmp_bitmap_sve2_ */
SZ_HELPER_AUTO svuint8_t sz_utf8_delimiter_astral_bitmap_sve2_(svuint8_t super_u8x, svuint8_t sub_u8x,
                                                               svuint8_t low8_u8x) {
    svuint16_t const super_low_u16x = svunpklo_u16(super_u8x), super_high_u16x = svunpkhi_u16(super_u8x);
    svuint16_t const sub_low_u16x = svunpklo_u16(sub_u8x), sub_high_u16x = svunpkhi_u16(sub_u8x);
    svuint16_t const low8_low_u16x = svunpklo_u16(low8_u8x), low8_high_u16x = svunpkhi_u16(low8_u8x);
    svuint32_t const first_u32x = sz_utf8_delimiter_astral_quarter_sve2_(
        svunpklo_u32(super_low_u16x), svunpklo_u32(sub_low_u16x), svunpklo_u32(low8_low_u16x));
    svuint32_t const second_u32x = sz_utf8_delimiter_astral_quarter_sve2_(
        svunpkhi_u32(super_low_u16x), svunpkhi_u32(sub_low_u16x), svunpkhi_u32(low8_low_u16x));
    svuint32_t const third_u32x = sz_utf8_delimiter_astral_quarter_sve2_(
        svunpklo_u32(super_high_u16x), svunpklo_u32(sub_high_u16x), svunpklo_u32(low8_high_u16x));
    svuint32_t const fourth_u32x = sz_utf8_delimiter_astral_quarter_sve2_(
        svunpkhi_u32(super_high_u16x), svunpkhi_u32(sub_high_u16x), svunpkhi_u32(low8_high_u16x));
    svuint16_t const packed_low_u16x = svuzp1_u16(svreinterpret_u16_u32(first_u32x),
                                                  svreinterpret_u16_u32(second_u32x));
    svuint16_t const packed_high_u16x = svuzp1_u16(svreinterpret_u16_u32(third_u32x),
                                                   svreinterpret_u16_u32(fourth_u32x));
    return svuzp1_u8(svreinterpret_u8_u16(packed_low_u16x), svreinterpret_u8_u16(packed_high_u16x));
}

/** @brief  Per-lane single-bit test `(bitmap_byte >> (low & 7)) & 1` as a byte predicate. */
SZ_HELPER_INLINE svbool_t sz_utf8_delimiter_test_bit_sve2_(svbool_t pg_b8x, svuint8_t bitmap_u8x, svuint8_t low_u8x) {
    svuint8_t const bit_mask_u8x = svlsl_u8_x(pg_b8x, svdup_n_u8(1), svand_n_u8_x(pg_b8x, low_u8x, 7));
    return svcmpne_n_u8(pg_b8x, svand_u8_x(pg_b8x, bitmap_u8x, bit_mask_u8x), 0);
}

/** @copydoc sz_utf8_delimiters */
SZ_API_COMPTIME sz_size_t sz_utf8_delimiters_sve2(      //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const step = svcntb();
    if (step < 16)
        return sz_utf8_delimiters_serial(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);

    svbool_t const all_b8x = svptrue_b8();
    svuint8_t const zeros_u8x = svdup_n_u8(0);
    // Per-lane sequence length from the lead's high nibble: {1 x12, 2, 2, 3, 4}, so a lead and its length can
    // never disagree and 0xF8..0xFF map to length 4.
    svuint8_t const length_lut_u8x = svdupq_n_u8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4);
    sz_size_t count = 0, position = 0;

    while (position < length && count < matches_capacity) {
        sz_size_t const valid_lanes = sz_min_of_two(step, length - position);
        sz_bool_t const is_final_tile = (sz_bool_t)(valid_lanes < step);
        // Trust starts up to `step - 4` on a full tile so a 4-byte delimiter is always fully loaded; the final
        // tile trusts every valid lane (truncated tails read zero and never validate).
        sz_size_t const usable = is_final_tile ? valid_lanes : step - 3;
        svbool_t const load_b8x = svwhilelt_b8_u64(0, (sz_u64_t)valid_lanes);
        svbool_t const pg_b8x = svwhilelt_b8_u64(0, (sz_u64_t)usable);
        svuint8_t const text0_u8x = svld1_u8(load_b8x, text_u8 + position);

        svbool_t starts_b8x;
        svuint8_t lengths_u8x;

        // All-ASCII tile: membership is one 16-byte TBL over the first half of bitmap row 0 (`bmp_block_[0] == 0`,
        // and ASCII columns stop at `0x7F >> 3 == 15`); every valid lane is a one-byte start.
        svbool_t const non_ascii_b8x = svcmplt_n_s8(load_b8x, svreinterpret_s8_u8(text0_u8x), 0);
        if (!svptest_any(load_b8x, non_ascii_b8x)) {
            svuint8_t const row0_u8x = svld1_u8(svwhilelt_b8_u64(0, 16), sz_utf8_delimiter_bmp_bitmaps_);
            svuint8_t const bitmap_u8x = svtbl_u8(row0_u8x, svlsr_n_u8_x(load_b8x, text0_u8x, 3));
            starts_b8x = svand_b_z(load_b8x, sz_utf8_delimiter_test_bit_sve2_(load_b8x, bitmap_u8x, text0_u8x),
                                   load_b8x);
            lengths_u8x = svdup_n_u8(1);
            sz_size_t const span = valid_lanes;
            sz_size_t const span_matches = svcntp_b8(load_b8x, starts_b8x);
            sz_size_t const emit_count = sz_min_of_two(span_matches, matches_capacity - count);
            if (emit_count)
                sz_utf8_token_drain_sve2_(starts_b8x, lengths_u8x, position, span, emit_count, match_offsets + count,
                                          match_lengths + count);
            count += emit_count;
            position = emit_count ? match_offsets[count - 1] + match_lengths[count - 1] : position + span;
            if (count == matches_capacity) break;
            continue;
        }

        svuint8_t const text1_u8x = svext_u8(text0_u8x, zeros_u8x, 1);
        svuint8_t const text2_u8x = svext_u8(text0_u8x, zeros_u8x, 2);
        svuint8_t const text3_u8x = svext_u8(text0_u8x, zeros_u8x, 3);

        // Structural classes and UTF-8 validity, mirroring the serial decode: continuation chains, C0/C1 and
        // F5..FF bad leads, E0/ED/F0/F4 first-continuation ranges. An invalid lead is never reported.
        svbool_t const ascii_b8x = svcmplt_n_u8(pg_b8x, text0_u8x, 0x80);
        svbool_t const is_two_b8x = svand_b_z(pg_b8x, svcmpge_n_u8(pg_b8x, text0_u8x, 0xC0),
                                              svcmplt_n_u8(pg_b8x, text0_u8x, 0xE0));
        svbool_t const is_three_b8x = svand_b_z(pg_b8x, svcmpge_n_u8(pg_b8x, text0_u8x, 0xE0),
                                                svcmplt_n_u8(pg_b8x, text0_u8x, 0xF0));
        svbool_t const is_four_b8x = svcmpge_n_u8(pg_b8x, text0_u8x, 0xF0);
        svbool_t const continuation_1_b8x = svcmpeq_n_u8(pg_b8x, svand_n_u8_x(pg_b8x, text1_u8x, 0xC0), 0x80);
        svbool_t const continuation_2_b8x = svcmpeq_n_u8(pg_b8x, svand_n_u8_x(pg_b8x, text2_u8x, 0xC0), 0x80);
        svbool_t const continuation_3_b8x = svcmpeq_n_u8(pg_b8x, svand_n_u8_x(pg_b8x, text3_u8x, 0xC0), 0x80);

        svbool_t const two_ok_b8x = svand_b_z(pg_b8x, continuation_1_b8x, svcmpge_n_u8(pg_b8x, text0_u8x, 0xC2));
        svbool_t const lead_e0_b8x = svcmpeq_n_u8(pg_b8x, text0_u8x, 0xE0);
        svbool_t const lead_ed_b8x = svcmpeq_n_u8(pg_b8x, text0_u8x, 0xED);
        svbool_t const next_lt_a0_b8x = svcmplt_n_u8(pg_b8x, text1_u8x, 0xA0);
        svbool_t const bad_three_b8x = svorr_b_z(pg_b8x, svand_b_z(pg_b8x, lead_e0_b8x, next_lt_a0_b8x),
                                                 svbic_b_z(pg_b8x, lead_ed_b8x, next_lt_a0_b8x));
        svbool_t const three_ok_b8x = svbic_b_z(pg_b8x, svand_b_z(pg_b8x, continuation_1_b8x, continuation_2_b8x),
                                                bad_three_b8x);
        svbool_t const lead_f0_b8x = svcmpeq_n_u8(pg_b8x, text0_u8x, 0xF0);
        svbool_t const lead_f4_b8x = svcmpeq_n_u8(pg_b8x, text0_u8x, 0xF4);
        svbool_t const next_lt_90_b8x = svcmplt_n_u8(pg_b8x, text1_u8x, 0x90);
        svbool_t const bad_four_b8x = svorr_b_z(pg_b8x, svcmpgt_n_u8(pg_b8x, text0_u8x, 0xF4),
                                                svorr_b_z(pg_b8x, svand_b_z(pg_b8x, lead_f0_b8x, next_lt_90_b8x),
                                                          svbic_b_z(pg_b8x, lead_f4_b8x, next_lt_90_b8x)));
        svbool_t const four_ok_b8x = svbic_b_z(
            pg_b8x, svand_b_z(pg_b8x, svand_b_z(pg_b8x, continuation_1_b8x, continuation_2_b8x), continuation_3_b8x),
            bad_four_b8x);

        svbool_t const valid_b8x = svorr_b_z(pg_b8x,
                                             svorr_b_z(pg_b8x, ascii_b8x, svand_b_z(pg_b8x, is_two_b8x, two_ok_b8x)),
                                             svorr_b_z(pg_b8x, svand_b_z(pg_b8x, is_three_b8x, three_ok_b8x),
                                                       svand_b_z(pg_b8x, is_four_b8x, four_ok_b8x)));

        // BMP (high, low) reconstruction: ASCII lanes keep (0, byte); 2-/3-byte lanes fold their continuations.
        svuint8_t const two_high_u8x = svlsr_n_u8_x(pg_b8x, svand_n_u8_x(pg_b8x, text0_u8x, 0x1F), 2);
        svuint8_t const two_low_u8x = svorr_u8_x(pg_b8x, svlsl_n_u8_x(pg_b8x, svand_n_u8_x(pg_b8x, text0_u8x, 0x03), 6),
                                                 svand_n_u8_x(pg_b8x, text1_u8x, 0x3F));
        svuint8_t const three_high_u8x = svorr_u8_x(pg_b8x,
                                                    svlsl_n_u8_x(pg_b8x, svand_n_u8_x(pg_b8x, text0_u8x, 0x0F), 4),
                                                    svand_n_u8_x(pg_b8x, svlsr_n_u8_x(pg_b8x, text1_u8x, 2), 0x0F));
        svuint8_t const three_low_u8x = svorr_u8_x(pg_b8x,
                                                   svlsl_n_u8_x(pg_b8x, svand_n_u8_x(pg_b8x, text1_u8x, 0x03), 6),
                                                   svand_n_u8_x(pg_b8x, text2_u8x, 0x3F));
        svuint8_t high_u8x = svsel_u8(is_three_b8x, three_high_u8x, two_high_u8x);
        svuint8_t low_u8x = svsel_u8(is_three_b8x, three_low_u8x, two_low_u8x);
        high_u8x = svsel_u8(ascii_b8x, zeros_u8x, high_u8x);
        low_u8x = svsel_u8(ascii_b8x, text0_u8x, low_u8x);

        svuint8_t const bmp_bitmap_u8x = sz_utf8_delimiter_bmp_bitmap_sve2_(high_u8x, low_u8x);
        svbool_t member_b8x = sz_utf8_delimiter_test_bit_sve2_(pg_b8x, bmp_bitmap_u8x, low_u8x);

        // Astral membership overrides the (meaningless) BMP fold on valid 4-byte lanes; a valid 4-byte sequence
        // is always >= U+10000, so `super = (codepoint >> 16) - 1` never borrows.
        svbool_t const four_valid_b8x = svand_b_z(pg_b8x, is_four_b8x, four_ok_b8x);
        if (svptest_any(pg_b8x, four_valid_b8x)) {
            svuint8_t const b1_u8x = svand_n_u8_x(pg_b8x, text1_u8x, 0x3F);
            svuint8_t const b2_u8x = svand_n_u8_x(pg_b8x, text2_u8x, 0x3F);
            svuint8_t const super_u8x = svsub_n_u8_x(
                pg_b8x,
                svorr_u8_x(pg_b8x, svlsl_n_u8_x(pg_b8x, svand_n_u8_x(pg_b8x, text0_u8x, 0x07), 2),
                           svlsr_n_u8_x(pg_b8x, b1_u8x, 4)),
                1);
            svuint8_t const sub_u8x = svorr_u8_x(pg_b8x, svlsl_n_u8_x(pg_b8x, b1_u8x, 4),
                                                 svlsr_n_u8_x(pg_b8x, b2_u8x, 2));
            svuint8_t const low8_u8x = svorr_u8_x(pg_b8x, svlsl_n_u8_x(pg_b8x, b2_u8x, 6),
                                                  svand_n_u8_x(pg_b8x, text3_u8x, 0x3F));
            svuint8_t const astral_bitmap_u8x = sz_utf8_delimiter_astral_bitmap_sve2_(super_u8x, sub_u8x, low8_u8x);
            svbool_t const astral_member_b8x = sz_utf8_delimiter_test_bit_sve2_(pg_b8x, astral_bitmap_u8x, low8_u8x);
            member_b8x = svsel_b(four_valid_b8x, astral_member_b8x, member_b8x);
        }

        svbool_t const continuation_here_b8x = svcmpeq_n_u8(pg_b8x, svand_n_u8_x(pg_b8x, text0_u8x, 0xC0), 0x80);
        starts_b8x = svbic_b_z(pg_b8x, svand_b_z(pg_b8x, member_b8x, valid_b8x), continuation_here_b8x);
        lengths_u8x = svtbl_u8(length_lut_u8x, svlsr_n_u8_x(all_b8x, text0_u8x, 4));

        sz_size_t const span = usable;
        sz_size_t const span_matches = svcntp_b8(pg_b8x, starts_b8x);
        sz_size_t const emit_count = sz_min_of_two(span_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_token_drain_sve2_(starts_b8x, lengths_u8x, position, span, emit_count, match_offsets + count,
                                      match_lengths + count);
        count += emit_count;
        position = emit_count ? match_offsets[count - 1] + match_lengths[count - 1] : position + span;
        if (count == matches_capacity) break;
    }

    if (bytes_consumed) *bytes_consumed = sz_min_of_two(position, length);
    return count;
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

#endif // STRINGZILLA_UTF8_TOKENS_SVE2_H_
