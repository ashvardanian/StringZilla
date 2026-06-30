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

/** @brief  Drain the first `emit_count` matches in the first `span` (<= `svcntw()`) byte lanes into absolute
 *          `(offset, length)` pairs with ONE `svcompact_u32` - the `(offset, length)` twin of
 *          @ref sz_utf8_rune_drain_sve2_.
 *
 *  SVE2 has no `svcompact_u8`, so the byte-granular start lanes are widened to 32-bit (their lane index IS the
 *  iota) and a single `svcompact_u32` packs the start indices while a lockstep `svcompact_u32` packs their byte
 *  lengths. There is no sub-block store loop: the caller's `while` loop resumes past the last emitted match to
 *  cover the rest of the window, exactly as the rune drain lets its driver take more windows. */
SZ_HELPER_AUTO void sz_utf8_token_drain_sve2_(                                  //
    svbool_t starts, svuint8_t lengths_u8x, sz_size_t position, sz_size_t span, //
    sz_size_t emit_count, sz_size_t *match_offsets, sz_size_t *match_lengths) {

    // Restrict to the first `span` (<= `svcntw()`) byte lanes, then widen those to 32-bit so ONE `svcompact_u32`
    // packs the start lane-indices; lane `j` of every compacted vector belongs to the `j`-th emitted start.
    svbool_t const span_b8x = svwhilelt_b8_u64(0, (sz_u64_t)span);
    svuint8_t const start_flags_u8x = svdup_u8_z(svand_b_z(span_b8x, starts, span_b8x), 1);
    svbool_t const emit_starts_b32x = svcmpne_n_u32(svptrue_b32(), svunpklo_u32(svunpklo_u16(start_flags_u8x)), 0);

    svuint32_t const lane_iota_u32x = svindex_u32(0, 1);
    svuint32_t const lengths_word_u32x = svunpklo_u32(svunpklo_u16(lengths_u8x));
    svuint32_t const offsets_packed_u32x = svcompact_u32(emit_starts_b32x, lane_iota_u32x);
    svuint32_t const lengths_packed_u32x = svcompact_u32(emit_starts_b32x, lengths_word_u32x);
    svuint32_t const absolute_offsets_u32x = svadd_n_u32_x(svptrue_b32(), offsets_packed_u32x, (sz_u32_t)position);

    // Widen the first `emit_count` packed 32-bit results to 64-bit `sz_size_t` lanes: low half via `svunpklo_u64`,
    // remainder via the high half. `emit_count <= span <= svcntw()`, so the two halves always suffice.
    svbool_t const store_lo_b64x = svwhilelt_b64_u64(0, (sz_u64_t)emit_count);
    svst1_u64(store_lo_b64x, (sz_u64_t *)match_offsets, svunpklo_u64(absolute_offsets_u32x));
    svst1_u64(store_lo_b64x, (sz_u64_t *)match_lengths, svunpklo_u64(lengths_packed_u32x));
    sz_size_t const lo_count = svcntp_b64(svptrue_b64(), store_lo_b64x);
    if (emit_count > lo_count) {
        svbool_t const store_hi_b64x = svwhilelt_b64_u64(0, (sz_u64_t)(emit_count - lo_count));
        svst1_u64(store_hi_b64x, (sz_u64_t *)(match_offsets + lo_count), svunpkhi_u64(absolute_offsets_u32x));
        svst1_u64(store_hi_b64x, (sz_u64_t *)(match_lengths + lo_count), svunpkhi_u64(lengths_packed_u32x));
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
    svuint8_t const zeros_u8x = svdup_n_u8(0);
    sz_size_t count = 0, position = 0;

    // Same delimiter set & branchless mask shape as the single-tile scan, but every multi-byte predicate is
    // computed unconditionally and accumulated into a per-lane start predicate + length vector.
    svuint8_t const one_byte_set_u8x = svdupq_n_u8('\n', '\v', '\f', '\r', '\n', '\n', '\n', '\n', '\n', '\n', '\n',
                                                   '\n', '\n', '\n', '\n', '\n');
    while (position < length && count < matches_capacity) {
        // Predicate the load over the valid lanes so a final partial tile reads zero past the end; a truncated
        // multi-byte delimiter at EOF can therefore never match.
        sz_size_t const valid_lanes = sz_min_of_two(tile, length - position);
        sz_bool_t const is_final_tile = (sz_bool_t)(valid_lanes < tile);
        // Trust starts in lanes `[0, tile - 3]` on a full tile (so a 2-/3-byte delimiter is fully loaded); on the
        // final tile trust every valid lane (truncated tails read zero) and consume the whole remainder.
        sz_size_t const usable = is_final_tile ? valid_lanes : tile - 2;
        svbool_t const load_pg_b8x = svwhilelt_b8_u64(0, (sz_u64_t)valid_lanes);
        svbool_t const pg_b8x = svwhilelt_b8_u64(0, (sz_u64_t)usable); // trusted-start lanes
        svuint8_t text0_u8x = svld1_u8(load_pg_b8x, text_u8 + position);
        svuint8_t text1_u8x = svext_u8(text0_u8x, zeros_u8x, 1);
        svuint8_t text2_u8x = svext_u8(text0_u8x, zeros_u8x, 2);

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
        svbool_t starts_b8x = svbic_b_z(
            pg_b8x, svorr_b_z(pg_b8x, one_byte_mask_b8x, svorr_b_z(pg_b8x, two_byte_mask_b8x, three_byte_mask_b8x)),
            lf_of_crlf_mask_b8x);
        // Suppress a leading LF already consumed by a CRLF that straddled the previous tile edge.
        if (position != 0 && text_u8[position - 1] == '\r') {
            svbool_t lane0_b8x = svcmpeq_n_u8(svptrue_b8(), svindex_u8(0, 1), 0);
            starts_b8x = svbic_b_z(pg_b8x, starts_b8x, svand_b_z(pg_b8x, lane0_b8x, lf_mask_b8x));
        }

        // Per-lane length: 1 + (2-byte) + 2*(3-byte); the masks are disjoint.
        svuint8_t lengths_u8x = svdup_n_u8(1);
        lengths_u8x = svadd_n_u8_m(two_byte_mask_b8x, lengths_u8x, 1);
        lengths_u8x = svadd_n_u8_m(three_byte_mask_b8x, lengths_u8x, 2);

        // Emit the matches in the first `svcntw()` byte lanes with ONE `svcompact_u32`; the `while` loop resumes
        // past the last emitted match to cover the rest of the window (no sub-block store loop).
        sz_size_t const span = sz_min_of_two((sz_size_t)svcntw(), usable);
        svbool_t const span_pg_b8x = svwhilelt_b8_u64(0, (sz_u64_t)span);
        sz_size_t const span_matches = svcntp_b8(span_pg_b8x, svand_b_z(span_pg_b8x, starts_b8x, span_pg_b8x));
        sz_size_t const emit_count = sz_min_of_two(span_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_token_drain_sve2_(starts_b8x, lengths_u8x, position, span, emit_count, match_offsets + count,
                                      match_lengths + count);
        count += emit_count;
        // Resume past the last emitted match, or skip the whole (match-free) span.
        position = emit_count ? match_offsets[count - 1] + match_lengths[count - 1] : position + span;
        if (count == matches_capacity) break; // output buffer full: `position` already past the last match
    }

    // Skip a CRLF's trailing LF if it straddles the resume boundary (the CR was emitted as a 2-byte match).
    if (position != 0 && position < length && text_u8[position - 1] == '\r' && text_u8[position] == '\n') ++position;
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
    svuint8_t const zeros_u8x = svdup_n_u8(0);
    sz_size_t count = 0, position = 0;

    svuint8_t const one_byte_set_u8x = svdupq_n_u8(' ', '\t', '\n', '\v', '\f', '\r', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                                                   ' ', ' ', ' ');
    // Valid third bytes for E2 80 XX: U+2000-U+200A (0x80-0x8A), plus U+2028 (0xA8), U+2029 (0xA9). U+200B/200C/200D
    // are Format chars, not White_Space; the 16-lane set test repeats 0x80 (already a member) in their freed slots.
    svuint8_t const e280_third_bytes_u8x = svdupq_n_u8(0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A,
                                                       0x80, 0x80, 0x80, 0xA8, 0xA9);

    while (position < length && count < matches_capacity) {
        // Predicate the load over the valid lanes so a final partial tile reads zero past the end; a truncated
        // multi-byte delimiter at EOF can therefore never match.
        sz_size_t const valid_lanes = sz_min_of_two(tile, length - position);
        sz_bool_t const is_final_tile = (sz_bool_t)(valid_lanes < tile);
        sz_size_t const usable = is_final_tile ? valid_lanes : tile - 2;
        svbool_t const load_pg_b8x = svwhilelt_b8_u64(0, (sz_u64_t)valid_lanes);
        svbool_t const pg_b8x = svwhilelt_b8_u64(0, (sz_u64_t)usable); // trusted-start lanes
        svuint8_t text0_u8x = svld1_u8(load_pg_b8x, text_u8 + position);
        svuint8_t text1_u8x = svext_u8(text0_u8x, zeros_u8x, 1);
        svuint8_t text2_u8x = svext_u8(text0_u8x, zeros_u8x, 2);

        // 1-byte: ' ' \t \n \v \f \r
        svbool_t one_byte_mask_b8x = svmatch_u8(pg_b8x, text0_u8x, one_byte_set_u8x);
        // 2-byte: C2 {85, A0} (NEL, NBSP). NO CRLF merging for whitespace.
        svbool_t lead_c2_mask_b8x = svcmpeq_n_u8(pg_b8x, text0_u8x, 0xC2);
        svbool_t two_byte_mask_b8x = svand_b_z(
            pg_b8x, lead_c2_mask_b8x,
            svorr_b_z(pg_b8x, svcmpeq_n_u8(pg_b8x, text1_u8x, 0x85), svcmpeq_n_u8(pg_b8x, text1_u8x, 0xA0)));
        // 3-byte: E1 9A 80 (Ogham)
        svbool_t ogham_mask_b8x = svand_b_z(
            pg_b8x, svand_b_z(pg_b8x, svcmpeq_n_u8(pg_b8x, text0_u8x, 0xE1), svcmpeq_n_u8(pg_b8x, text1_u8x, 0x9A)),
            svcmpeq_n_u8(pg_b8x, text2_u8x, 0x80));
        // 3-byte: E2 80 [80-8A] | A8 | A9 (set match); E2 80 AF (NNBSP); E2 81 9F (MMSP)
        svbool_t lead_e2_mask_b8x = svcmpeq_n_u8(pg_b8x, text0_u8x, 0xE2);
        svbool_t lead_e280_mask_b8x = svand_b_z(pg_b8x, lead_e2_mask_b8x, svcmpeq_n_u8(pg_b8x, text1_u8x, 0x80));
        svbool_t e280xx_mask_b8x = svand_b_z(pg_b8x, lead_e280_mask_b8x,
                                             svmatch_u8(pg_b8x, text2_u8x, e280_third_bytes_u8x));
        svbool_t nnbsp_mask_b8x = svand_b_z(pg_b8x, lead_e280_mask_b8x, svcmpeq_n_u8(pg_b8x, text2_u8x, 0xAF));
        svbool_t mmsp_mask_b8x = svand_b_z(pg_b8x,
                                           svand_b_z(pg_b8x, lead_e2_mask_b8x, svcmpeq_n_u8(pg_b8x, text1_u8x, 0x81)),
                                           svcmpeq_n_u8(pg_b8x, text2_u8x, 0x9F));
        // 3-byte: E3 80 80 (Ideographic)
        svbool_t ideographic_mask_b8x = svand_b_z(
            pg_b8x, svand_b_z(pg_b8x, svcmpeq_n_u8(pg_b8x, text0_u8x, 0xE3), svcmpeq_n_u8(pg_b8x, text1_u8x, 0x80)),
            svcmpeq_n_u8(pg_b8x, text2_u8x, 0x80));

        svbool_t three_byte_mask_b8x = svorr_b_z(
            pg_b8x, svorr_b_z(pg_b8x, ogham_mask_b8x, svorr_b_z(pg_b8x, e280xx_mask_b8x, nnbsp_mask_b8x)),
            svorr_b_z(pg_b8x, mmsp_mask_b8x, ideographic_mask_b8x));
        svbool_t starts_b8x = svorr_b_z(pg_b8x, one_byte_mask_b8x,
                                        svorr_b_z(pg_b8x, two_byte_mask_b8x, three_byte_mask_b8x));

        // Per-lane length: 1 + (2-byte) + 2*(3-byte); the masks are disjoint.
        svuint8_t lengths_u8x = svdup_n_u8(1);
        lengths_u8x = svadd_n_u8_m(two_byte_mask_b8x, lengths_u8x, 1);
        lengths_u8x = svadd_n_u8_m(three_byte_mask_b8x, lengths_u8x, 2);

        // Emit the matches in the first `svcntw()` byte lanes with ONE `svcompact_u32`; the `while` loop resumes
        // past the last emitted match to cover the rest of the window (no sub-block store loop).
        sz_size_t const span = sz_min_of_two((sz_size_t)svcntw(), usable);
        svbool_t const span_pg_b8x = svwhilelt_b8_u64(0, (sz_u64_t)span);
        sz_size_t const span_matches = svcntp_b8(span_pg_b8x, svand_b_z(span_pg_b8x, starts_b8x, span_pg_b8x));
        sz_size_t const emit_count = sz_min_of_two(span_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_token_drain_sve2_(starts_b8x, lengths_u8x, position, span, emit_count, match_offsets + count,
                                      match_lengths + count);
        count += emit_count;
        // Resume past the last emitted match, or skip the whole (match-free) span.
        position = emit_count ? match_offsets[count - 1] + match_lengths[count - 1] : position + span;
        if (count == matches_capacity) break; // output buffer full: `position` already past the last match
    }

    if (bytes_consumed) *bytes_consumed = position;
    return count;
}

/*  UAX-29 word segmentation: vectorize the dominant all-ASCII case and defer every stateful / non-ASCII or
 *  window-edge position to one scalar `sz_utf8_is_word_boundary_serial` step, so the result is byte-exact to
 *  serial. An all-ASCII window (lane `j` = byte `base + j`, `base = position - 2`) is classified to its
 *  `sz_utf8_word_break_t` property, the no-break join is replayed on per-lane vectors (below), and boundaries
 *  (the join complement over trusted lanes `[2, svcntb() - 2]`) are compacted with `svcompact_u32`. */

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
