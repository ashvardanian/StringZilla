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

/** @brief  Membership of each byte-lane @p value_u8x in a 32-byte (256-bit) bitmap held in two zero-padded
 *          16-byte table vectors: the byte at `value >> 3` rides two `svtbl` tables - the second addressed at
 *          `index - 16`, where the wrap past the zero-padded table reads zero at any vector length - and the
 *          bit `value & 7` decides. The in-register twin of the gathered two-level walk this file used to do. */
SZ_HELPER_INLINE svbool_t sz_utf8_delimiter_bitmap32_sve2_(svbool_t pg_b8x, svuint8_t table_low_u8x,
                                                           svuint8_t table_high_u8x, svuint8_t value_u8x) {
    svuint8_t const byte_index_u8x = svlsr_n_u8_x(pg_b8x, value_u8x, 3);
    svuint8_t const bitmap_u8x = svorr_u8_x(pg_b8x, svtbl_u8(table_low_u8x, byte_index_u8x),
                                            svtbl_u8(table_high_u8x, svsub_n_u8_x(pg_b8x, byte_index_u8x, 16)));
    svuint8_t const bit_mask_u8x = svlsl_u8_x(pg_b8x, svdup_n_u8(1), svand_n_u8_x(pg_b8x, value_u8x, 7));
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

    // Two 32-byte bitmaps as zero-padded 16-byte table pairs: BMP row 0 (exact membership of every codepoint
    // below U+0100) and the block-level pre-filter over `cp >> 8`.
    svbool_t const table_b8x = svwhilelt_b8_u64(0, 16);
    svuint8_t const row0_low_u8x = svld1_u8(table_b8x, sz_utf8_delimiter_bmp_bitmaps_);
    svuint8_t const row0_high_u8x = svld1_u8(table_b8x, sz_utf8_delimiter_bmp_bitmaps_ + 16);
    svuint8_t const suspicious_low_u8x = svld1_u8(table_b8x, sz_utf8_delimiter_bmp_suspicious_highs_);
    svuint8_t const suspicious_high_u8x = svld1_u8(table_b8x, sz_utf8_delimiter_bmp_suspicious_highs_ + 16);
    sz_size_t count = 0, position = 0;

    // 64-byte windows streamed as register chunks, each peeking one vector ahead so `svext` views cover every
    // lane's continuations - no trusted-margin re-scan, no cross-window straddle. Per-chunk verdicts lower to
    // one window-wide lane mask and drain once per window; the independent per-chunk resolution chains overlap
    // in the out-of-order core.
    while (position < length && count < matches_capacity) {
        sz_size_t const window = sz_min_of_two((sz_size_t)64, length - position);
        sz_u64_t hits_mask = 0;
        sz_size_t chunk_offset = 0;
        svuint8_t chunk_u8x = svld1_u8(svwhilelt_b8_u64(0, (sz_u64_t)sz_min_of_two(step, length - position)),
                                       text_u8 + position);
        while (chunk_offset < window) {
            sz_size_t const beyond = length - position - chunk_offset;
            sz_size_t const peek_span = beyond > step ? sz_min_of_two(step, beyond - step) : 0;
            svuint8_t const peek_u8x = svld1_u8(svwhilelt_b8_u64(0, (sz_u64_t)peek_span),
                                                text_u8 + position + chunk_offset + step);
            svbool_t const pg_b8x = svwhilelt_b8_u64(0, (sz_u64_t)sz_min_of_two(step, window - chunk_offset));
            svuint8_t const text0_u8x = chunk_u8x;

            // All-ASCII chunk: membership is the row-0 bitmap directly (`bmp_block_[0] == 0`, and ASCII columns
            // stop in the low table half); every lane is a one-byte start.
            svbool_t const non_ascii_b8x = svcmplt_n_s8(pg_b8x, svreinterpret_s8_u8(text0_u8x), 0);
            if (!svptest_any(pg_b8x, non_ascii_b8x)) {
                svbool_t const starts_b8x = sz_utf8_delimiter_bitmap32_sve2_(pg_b8x, row0_low_u8x, row0_high_u8x,
                                                                             text0_u8x);
                if (svptest_any(pg_b8x, starts_b8x))
                    hits_mask |= sz_utf8_rune_pred_to_u64_sve2_(starts_b8x) << chunk_offset;
                chunk_u8x = peek_u8x;
                chunk_offset += step;
                continue;
            }

            svuint8_t const text1_u8x = svext_u8(text0_u8x, peek_u8x, 1);
            svuint8_t const text2_u8x = svext_u8(text0_u8x, peek_u8x, 2);

            // Structural classes and UTF-8 validity, mirroring the serial decode: continuation chains, C0/C1 and
            // F5..FF bad leads, E0/ED/F0/F4 first-continuation ranges. An invalid lead is never reported. The
            // 3-byte and 4-byte refinements run only when such leads are present, so 2-byte scripts skip them.
            svbool_t const ascii_b8x = svcmplt_n_u8(pg_b8x, text0_u8x, 0x80);
            svbool_t const is_two_b8x = svand_b_z(pg_b8x, svcmpge_n_u8(pg_b8x, text0_u8x, 0xC0),
                                                  svcmplt_n_u8(pg_b8x, text0_u8x, 0xE0));
            svbool_t const is_three_b8x = svand_b_z(pg_b8x, svcmpge_n_u8(pg_b8x, text0_u8x, 0xE0),
                                                    svcmplt_n_u8(pg_b8x, text0_u8x, 0xF0));
            svbool_t const is_four_b8x = svcmpge_n_u8(pg_b8x, text0_u8x, 0xF0);
            svbool_t const continuation_1_b8x = svcmpeq_n_u8(pg_b8x, svand_n_u8_x(pg_b8x, text1_u8x, 0xC0), 0x80);
            svbool_t const two_ok_b8x = svand_b_z(pg_b8x, continuation_1_b8x, svcmpge_n_u8(pg_b8x, text0_u8x, 0xC2));
            svbool_t valid_b8x = svorr_b_z(pg_b8x, ascii_b8x, svand_b_z(pg_b8x, is_two_b8x, two_ok_b8x));
            svbool_t four_valid_b8x = svpfalse_b();

            int const any_three = svptest_any(pg_b8x, is_three_b8x);
            int const any_four = svptest_any(pg_b8x, is_four_b8x);
            if (any_three | any_four) {
                svbool_t const continuation_2_b8x = svcmpeq_n_u8(pg_b8x, svand_n_u8_x(pg_b8x, text2_u8x, 0xC0), 0x80);
                if (any_three) {
                    svbool_t const lead_e0_b8x = svcmpeq_n_u8(pg_b8x, text0_u8x, 0xE0);
                    svbool_t const lead_ed_b8x = svcmpeq_n_u8(pg_b8x, text0_u8x, 0xED);
                    svbool_t const next_lt_a0_b8x = svcmplt_n_u8(pg_b8x, text1_u8x, 0xA0);
                    svbool_t const bad_three_b8x = svorr_b_z(pg_b8x, svand_b_z(pg_b8x, lead_e0_b8x, next_lt_a0_b8x),
                                                             svbic_b_z(pg_b8x, lead_ed_b8x, next_lt_a0_b8x));
                    svbool_t const three_ok_b8x = svbic_b_z(
                        pg_b8x, svand_b_z(pg_b8x, continuation_1_b8x, continuation_2_b8x), bad_three_b8x);
                    valid_b8x = svorr_b_z(pg_b8x, valid_b8x, svand_b_z(pg_b8x, is_three_b8x, three_ok_b8x));
                }
                if (any_four) {
                    svbool_t const continuation_3_b8x = svcmpeq_n_u8(
                        pg_b8x, svand_n_u8_x(pg_b8x, svext_u8(text0_u8x, peek_u8x, 3), 0xC0), 0x80);
                    svbool_t const lead_f0_b8x = svcmpeq_n_u8(pg_b8x, text0_u8x, 0xF0);
                    svbool_t const lead_f4_b8x = svcmpeq_n_u8(pg_b8x, text0_u8x, 0xF4);
                    svbool_t const next_lt_90_b8x = svcmplt_n_u8(pg_b8x, text1_u8x, 0x90);
                    svbool_t const bad_four_b8x = svorr_b_z(
                        pg_b8x, svcmpgt_n_u8(pg_b8x, text0_u8x, 0xF4),
                        svorr_b_z(pg_b8x, svand_b_z(pg_b8x, lead_f0_b8x, next_lt_90_b8x),
                                  svbic_b_z(pg_b8x, lead_f4_b8x, next_lt_90_b8x)));
                    svbool_t const four_ok_b8x = svbic_b_z(
                        pg_b8x,
                        svand_b_z(pg_b8x, svand_b_z(pg_b8x, continuation_1_b8x, continuation_2_b8x),
                                  continuation_3_b8x),
                        bad_four_b8x);
                    four_valid_b8x = svand_b_z(pg_b8x, is_four_b8x, four_ok_b8x);
                    valid_b8x = svorr_b_z(pg_b8x, valid_b8x, four_valid_b8x);
                }
            }

            // BMP (high, low) reconstruction: ASCII lanes keep (0, byte); 2-/3-byte lanes fold continuations.
            svuint8_t high_u8x = svlsr_n_u8_x(pg_b8x, svand_n_u8_x(pg_b8x, text0_u8x, 0x1F), 2);
            svuint8_t low_u8x = svorr_u8_x(pg_b8x, svlsl_n_u8_x(pg_b8x, svand_n_u8_x(pg_b8x, text0_u8x, 0x03), 6),
                                           svand_n_u8_x(pg_b8x, text1_u8x, 0x3F));
            if (any_three) {
                svuint8_t const three_high_u8x = svorr_u8_x(
                    pg_b8x, svlsl_n_u8_x(pg_b8x, svand_n_u8_x(pg_b8x, text0_u8x, 0x0F), 4),
                    svand_n_u8_x(pg_b8x, svlsr_n_u8_x(pg_b8x, text1_u8x, 2), 0x0F));
                svuint8_t const three_low_u8x = svorr_u8_x(
                    pg_b8x, svlsl_n_u8_x(pg_b8x, svand_n_u8_x(pg_b8x, text1_u8x, 0x03), 6),
                    svand_n_u8_x(pg_b8x, text2_u8x, 0x3F));
                high_u8x = svsel_u8(is_three_b8x, three_high_u8x, high_u8x);
                low_u8x = svsel_u8(is_three_b8x, three_low_u8x, low_u8x);
            }
            high_u8x = svsel_u8(ascii_b8x, svdup_n_u8(0), high_u8x);
            low_u8x = svsel_u8(ascii_b8x, text0_u8x, low_u8x);

            // Exact membership, fully in-register. Lanes below U+0100 read bitmap row 0 directly; the rest
            // survive the block-level pre-filter only if their 256-codepoint block holds ANY delimiter (row 1 -
            // most of the plane, including all CJK letters - is the unique empty row), and the survivors resolve
            // one DISTINCT high byte at a time: running text shares one to three highs per chunk, and each round
            // loads that block's 32-byte row once and settles every lane carrying it.
            svbool_t const noncont_b8x = svcmpne_n_u8(pg_b8x, svand_n_u8_x(pg_b8x, text0_u8x, 0xC0), 0x80);
            svbool_t const startable_b8x = svand_b_z(pg_b8x, valid_b8x, noncont_b8x);
            svbool_t const high0_b8x = svcmpeq_n_u8(pg_b8x, high_u8x, 0);
            svbool_t const row0_member_b8x = sz_utf8_delimiter_bitmap32_sve2_(pg_b8x, row0_low_u8x, row0_high_u8x,
                                                                              low_u8x);
            svbool_t const block_suspicious_b8x = sz_utf8_delimiter_bitmap32_sve2_(pg_b8x, suspicious_low_u8x,
                                                                                   suspicious_high_u8x, high_u8x);
            svbool_t hits_b8x = svand_b_z(pg_b8x, startable_b8x, svand_b_z(pg_b8x, high0_b8x, row0_member_b8x));
            svbool_t unresolved_b8x = svand_b_z(
                pg_b8x, startable_b8x,
                svbic_b_z(pg_b8x, svbic_b_z(pg_b8x, block_suspicious_b8x, high0_b8x), is_four_b8x));
            while (svptest_any(pg_b8x, unresolved_b8x)) {
                sz_u8_t const shared_high = svlastb_u8(unresolved_b8x, high_u8x);
                sz_u8_t const *row = sz_utf8_delimiter_bmp_bitmaps_ +
                                     (sz_size_t)sz_utf8_delimiter_bmp_block_[shared_high] * 32;
                svuint8_t const row_low_u8x = svld1_u8(table_b8x, row);
                svuint8_t const row_high_u8x = svld1_u8(table_b8x, row + 16);
                svbool_t const same_b8x = svand_b_z(pg_b8x, unresolved_b8x,
                                                    svcmpeq_n_u8(pg_b8x, high_u8x, shared_high));
                svbool_t const member_b8x = sz_utf8_delimiter_bitmap32_sve2_(same_b8x, row_low_u8x, row_high_u8x,
                                                                             low_u8x);
                hits_b8x = svorr_b_z(pg_b8x, hits_b8x, svand_b_z(same_b8x, member_b8x, same_b8x));
                unresolved_b8x = svbic_b_z(pg_b8x, unresolved_b8x, same_b8x);
            }

            // Astral membership: the same distinct-value resolution keyed on the (super, sub) pair, where
            // `super = (codepoint >> 16) - 1` never borrows for a valid 4-byte sequence. Rare lanes, still
            // fully in-register.
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
                                                      svand_n_u8_x(pg_b8x, svext_u8(text0_u8x, peek_u8x, 3), 0x3F));
                svbool_t remaining_b8x = four_valid_b8x;
                while (svptest_any(pg_b8x, remaining_b8x)) {
                    sz_u8_t const shared_super = svlastb_u8(remaining_b8x, super_u8x);
                    svbool_t const super_same_b8x = svand_b_z(pg_b8x, remaining_b8x,
                                                              svcmpeq_n_u8(pg_b8x, super_u8x, shared_super));
                    sz_u8_t const shared_sub = svlastb_u8(super_same_b8x, sub_u8x);
                    sz_u8_t const *row =
                        sz_utf8_delimiter_astral_bitmaps_ +
                        (sz_size_t)sz_utf8_delimiter_astral_l2_[(sz_size_t)sz_utf8_delimiter_astral_l1_[shared_super] *
                                                                    256 +
                                                                shared_sub] *
                            32;
                    svuint8_t const row_low_u8x = svld1_u8(table_b8x, row);
                    svuint8_t const row_high_u8x = svld1_u8(table_b8x, row + 16);
                    svbool_t const same_b8x = svand_b_z(pg_b8x, super_same_b8x,
                                                        svcmpeq_n_u8(pg_b8x, sub_u8x, shared_sub));
                    svbool_t const member_b8x = sz_utf8_delimiter_bitmap32_sve2_(same_b8x, row_low_u8x, row_high_u8x,
                                                                                 low8_u8x);
                    hits_b8x = svorr_b_z(pg_b8x, hits_b8x, svand_b_z(same_b8x, member_b8x, same_b8x));
                    remaining_b8x = svbic_b_z(pg_b8x, remaining_b8x, same_b8x);
                }
            }

            if (svptest_any(pg_b8x, hits_b8x)) hits_mask |= sz_utf8_rune_pred_to_u64_sve2_(hits_b8x) << chunk_offset;
            chunk_u8x = peek_u8x;
            chunk_offset += step;
        }

        sz_size_t const emit_count = hits_mask ? sz_utf8_delimiter_emit_matches_(
                                                     text_u8, position, hits_mask, match_offsets + count,
                                                     match_lengths + count, matches_capacity - count)
                                               : 0;
        count += emit_count;
        // On a capacity-limited return `bytes_consumed` must be the last emitted match's end - callers treat it
        // as the resume boundary - and an emitted delimiter may end past the window, so the full-window skip
        // honors the furthest match end.
        sz_size_t const last_end = emit_count ? match_offsets[count - 1] + match_lengths[count - 1] : 0;
        position = count == matches_capacity ? last_end : sz_max_of_two(position + window, last_end);
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
