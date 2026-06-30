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

/** @brief  Peel one tile's first `emit_count` matches into absolute `(offset, length)` pairs.
 *
 *  SVE2 has no `svcompact_u8`, so the byte-granular start/length lanes are walked in `svcntw()`-byte sub-blocks:
 *  each gathers its lanes with `svtbl_u8`, widens to 32-bit, `svcompact_u32` packs the matches, and the store is
 *  bounded by `emit_count` so the output never exceeds the remaining capacity. */
SZ_HELPER_AUTO void sz_utf8_iterate_peel_tile_sve2_(                               //
    svbool_t starts, svuint8_t lengths_u8, sz_size_t usable, sz_size_t emit_count, //
    sz_size_t position, sz_size_t *match_offsets, sz_size_t *match_lengths) {

    svuint8_t const start_flags_vec = svdup_u8_z(starts, 1); // 1 in start lanes, 0 elsewhere
    svuint8_t const lengths_vec = lengths_u8;
    svuint8_t const lane_iota = svindex_u8(0, 1);

    sz_size_t const words = svcntw(); // 32-bit lanes per register
    sz_size_t matches_emitted = 0;
    for (sz_size_t base = 0; base < usable && matches_emitted < emit_count; base += words) {
        svbool_t const pw = svwhilelt_b32_u64((sz_u64_t)base, (sz_u64_t)usable);
        svuint8_t const gather_indices = svadd_n_u8_x(svptrue_b8(), lane_iota, (sz_u8_t)base);
        svuint8_t const flags_sub = svtbl_u8(start_flags_vec, gather_indices);
        svuint32_t const flags_w = svunpklo_u32(svunpklo_u16(flags_sub));
        svbool_t const sub_starts = svcmpne_n_u32(pw, flags_w, 0);
        if (!svptest_any(pw, sub_starts)) continue;

        svuint8_t const lengths_sub = svtbl_u8(lengths_vec, gather_indices);
        svuint32_t const length_w = svunpklo_u32(svunpklo_u16(lengths_sub));
        svuint32_t const offset_w = svindex_u32((sz_u32_t)(position + base), 1);
        svuint32_t const offset_packed = svcompact_u32(sub_starts, offset_w);
        svuint32_t const length_packed = svcompact_u32(sub_starts, length_w);
        sz_size_t const here = svcntp_b32(pw, sub_starts);
        // Cap this sub-block's emission at the remaining output budget so the total never exceeds `emit_count`.
        sz_size_t const emit_here = sz_min_of_two(here, emit_count - matches_emitted);

        // Widen the packed 32-bit results to 64-bit lanes and store: low half via `svunpklo_u64`, remainder via hi.
        svbool_t const pd_lo = svwhilelt_b64_u64(0, (sz_u64_t)emit_here);
        svst1_u64(pd_lo, (sz_u64_t *)(match_offsets + matches_emitted), svunpklo_u64(offset_packed));
        svst1_u64(pd_lo, (sz_u64_t *)(match_lengths + matches_emitted), svunpklo_u64(length_packed));
        sz_size_t const lo_count = svcntp_b64(svptrue_b64(), pd_lo);
        if (emit_here > lo_count) {
            svbool_t const pd_hi = svwhilelt_b64_u64(0, (sz_u64_t)(emit_here - lo_count));
            svst1_u64(pd_hi, (sz_u64_t *)(match_offsets + matches_emitted + lo_count), svunpkhi_u64(offset_packed));
            svst1_u64(pd_hi, (sz_u64_t *)(match_lengths + matches_emitted + lo_count), svunpkhi_u64(length_packed));
        }
        matches_emitted += emit_here;
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
    svuint8_t const zeros = svdup_n_u8(0);
    sz_size_t count = 0, position = 0;

    // Same delimiter set & branchless mask shape as the single-tile scan, but every multi-byte predicate is
    // computed unconditionally and accumulated into a per-lane start predicate + length vector.
    svuint8_t const one_byte_set = svdupq_n_u8('\n', '\v', '\f', '\r', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n',
                                               '\n', '\n', '\n', '\n');
    while (position < length && count < matches_capacity) {
        // Predicate the load over the valid lanes so a final partial tile reads zero past the end; a truncated
        // multi-byte delimiter at EOF can therefore never match.
        sz_size_t const valid_lanes = sz_min_of_two(tile, length - position);
        sz_bool_t const is_final_tile = (sz_bool_t)(valid_lanes < tile);
        // Trust starts in lanes `[0, tile - 3]` on a full tile (so a 2-/3-byte delimiter is fully loaded); on the
        // final tile trust every valid lane (truncated tails read zero) and consume the whole remainder.
        sz_size_t const usable = is_final_tile ? valid_lanes : tile - 2;
        svbool_t const load_pg = svwhilelt_b8_u64(0, (sz_u64_t)valid_lanes);
        svbool_t const pg = svwhilelt_b8_u64(0, (sz_u64_t)usable); // trusted-start lanes
        svuint8_t text0 = svld1_u8(load_pg, text_u8 + position);
        svuint8_t text1 = svext_u8(text0, zeros, 1);
        svuint8_t text2 = svext_u8(text0, zeros, 2);

        // 1-byte: \n \v \f \r
        svbool_t one_byte_mask = svmatch_u8(pg, text0, one_byte_set);
        // 2-byte: C2 85 (NEL); CRLF is a single 2-byte match.
        svbool_t cr_mask = svcmpeq_n_u8(pg, text0, '\r');
        svbool_t lf_mask = svcmpeq_n_u8(pg, text0, '\n');
        svbool_t crlf_mask = svand_b_z(pg, cr_mask, svcmpeq_n_u8(pg, text1, '\n'));
        svbool_t nel_mask = svand_b_z(pg, svcmpeq_n_u8(pg, text0, 0xC2), svcmpeq_n_u8(pg, text1, 0x85));
        svbool_t two_byte_mask = svorr_b_z(pg, crlf_mask, nel_mask);
        // 3-byte: E2 80 A8 (LS), E2 80 A9 (PS)
        svbool_t lead_e280_mask = svand_b_z(pg, svcmpeq_n_u8(pg, text0, 0xE2), svcmpeq_n_u8(pg, text1, 0x80));
        svbool_t three_byte_mask = svand_b_z(
            pg, lead_e280_mask, svorr_b_z(pg, svcmpeq_n_u8(pg, text2, 0xA8), svcmpeq_n_u8(pg, text2, 0xA9)));

        // A CR that completes a CRLF must be emitted once (len 2); the trailing LF must NOT also be a match.
        // `crlf_mask` marks the CR lane; the LF to suppress is the next lane. Up-shift a 0/1 CRLF flag by one
        // lane (svinsr inserts 0 at lane 0 and slides everything toward higher indices) to land on that LF.
        svuint8_t crlf_flag = svdup_u8_z(crlf_mask, 1);
        svbool_t lf_of_crlf_mask = svand_b_z(pg, lf_mask, svcmpne_n_u8(pg, svinsr_n_u8(crlf_flag, 0), 0));
        svbool_t starts = svbic_b_z(pg, svorr_b_z(pg, one_byte_mask, svorr_b_z(pg, two_byte_mask, three_byte_mask)),
                                    lf_of_crlf_mask);
        // Suppress a leading LF already consumed by a CRLF that straddled the previous tile edge.
        if (position != 0 && text_u8[position - 1] == '\r') {
            svbool_t lane0 = svcmpeq_n_u8(svptrue_b8(), svindex_u8(0, 1), 0);
            starts = svbic_b_z(pg, starts, svand_b_z(pg, lane0, lf_mask));
        }

        // Per-lane length: 1 + (2-byte) + 2*(3-byte); the masks are disjoint.
        svuint8_t lengths_u8 = svdup_n_u8(1);
        lengths_u8 = svadd_n_u8_m(two_byte_mask, lengths_u8, 1);
        lengths_u8 = svadd_n_u8_m(three_byte_mask, lengths_u8, 2);

        // Capacity cut: emit only as many matches as the output buffer can still hold.
        sz_size_t const window_matches = svcntp_b8(pg, starts);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_tile_sve2_(starts, lengths_u8, usable, emit_count, position, match_offsets + count,
                                            match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) { // output buffer full: resume past the last emitted match
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += usable;
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
    svuint8_t const zeros = svdup_n_u8(0);
    sz_size_t count = 0, position = 0;

    svuint8_t const one_byte_set = svdupq_n_u8(' ', '\t', '\n', '\v', '\f', '\r', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                                               ' ', ' ', ' ');
    // Valid third bytes for E2 80 XX: U+2000-U+200A (0x80-0x8A), plus U+2028 (0xA8), U+2029 (0xA9). U+200B/200C/200D
    // are Format chars, not White_Space; the 16-lane set test repeats 0x80 (already a member) in their freed slots.
    svuint8_t const e280_third_bytes = svdupq_n_u8(0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A,
                                                   0x80, 0x80, 0x80, 0xA8, 0xA9);

    while (position < length && count < matches_capacity) {
        // Predicate the load over the valid lanes so a final partial tile reads zero past the end; a truncated
        // multi-byte delimiter at EOF can therefore never match.
        sz_size_t const valid_lanes = sz_min_of_two(tile, length - position);
        sz_bool_t const is_final_tile = (sz_bool_t)(valid_lanes < tile);
        sz_size_t const usable = is_final_tile ? valid_lanes : tile - 2;
        svbool_t const load_pg = svwhilelt_b8_u64(0, (sz_u64_t)valid_lanes);
        svbool_t const pg = svwhilelt_b8_u64(0, (sz_u64_t)usable); // trusted-start lanes
        svuint8_t text0 = svld1_u8(load_pg, text_u8 + position);
        svuint8_t text1 = svext_u8(text0, zeros, 1);
        svuint8_t text2 = svext_u8(text0, zeros, 2);

        // 1-byte: ' ' \t \n \v \f \r
        svbool_t one_byte_mask = svmatch_u8(pg, text0, one_byte_set);
        // 2-byte: C2 {85, A0} (NEL, NBSP). NO CRLF merging for whitespace.
        svbool_t lead_c2_mask = svcmpeq_n_u8(pg, text0, 0xC2);
        svbool_t two_byte_mask = svand_b_z(pg, lead_c2_mask,
                                           svorr_b_z(pg, svcmpeq_n_u8(pg, text1, 0x85), svcmpeq_n_u8(pg, text1, 0xA0)));
        // 3-byte: E1 9A 80 (Ogham)
        svbool_t ogham_mask = svand_b_z(pg, svand_b_z(pg, svcmpeq_n_u8(pg, text0, 0xE1), svcmpeq_n_u8(pg, text1, 0x9A)),
                                        svcmpeq_n_u8(pg, text2, 0x80));
        // 3-byte: E2 80 [80-8A] | A8 | A9 (set match); E2 80 AF (NNBSP); E2 81 9F (MMSP)
        svbool_t lead_e2_mask = svcmpeq_n_u8(pg, text0, 0xE2);
        svbool_t lead_e280_mask = svand_b_z(pg, lead_e2_mask, svcmpeq_n_u8(pg, text1, 0x80));
        svbool_t e280xx_mask = svand_b_z(pg, lead_e280_mask, svmatch_u8(pg, text2, e280_third_bytes));
        svbool_t nnbsp_mask = svand_b_z(pg, lead_e280_mask, svcmpeq_n_u8(pg, text2, 0xAF));
        svbool_t mmsp_mask = svand_b_z(pg, svand_b_z(pg, lead_e2_mask, svcmpeq_n_u8(pg, text1, 0x81)),
                                       svcmpeq_n_u8(pg, text2, 0x9F));
        // 3-byte: E3 80 80 (Ideographic)
        svbool_t ideographic_mask = svand_b_z(
            pg, svand_b_z(pg, svcmpeq_n_u8(pg, text0, 0xE3), svcmpeq_n_u8(pg, text1, 0x80)),
            svcmpeq_n_u8(pg, text2, 0x80));

        svbool_t three_byte_mask = svorr_b_z(pg, svorr_b_z(pg, ogham_mask, svorr_b_z(pg, e280xx_mask, nnbsp_mask)),
                                             svorr_b_z(pg, mmsp_mask, ideographic_mask));
        svbool_t starts = svorr_b_z(pg, one_byte_mask, svorr_b_z(pg, two_byte_mask, three_byte_mask));

        // Per-lane length: 1 + (2-byte) + 2*(3-byte); the masks are disjoint.
        svuint8_t lengths_u8 = svdup_n_u8(1);
        lengths_u8 = svadd_n_u8_m(two_byte_mask, lengths_u8, 1);
        lengths_u8 = svadd_n_u8_m(three_byte_mask, lengths_u8, 2);

        // Capacity cut: emit only as many matches as the output buffer can still hold.
        sz_size_t const window_matches = svcntp_b8(pg, starts);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_tile_sve2_(starts, lengths_u8, usable, emit_count, position, match_offsets + count,
                                            match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) { // output buffer full: resume past the last emitted match
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += usable;
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
