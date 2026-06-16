/**
 *  @brief SVE2 backend for UTF-8 traversal.
 *  @file include/stringzilla/utf8_iterate/sve2.h
 *  @author Ash Vardanian
 *  @sa utf8.h
 */
#ifndef STRINGZILLA_UTF8_ITERATE_SVE2_H_
#define STRINGZILLA_UTF8_ITERATE_SVE2_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_iterate/serial.h"

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

SZ_PUBLIC sz_size_t sz_utf8_count_sve2(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const step = svcntb();
    sz_size_t char_count = 0;

    // Count bytes that are NOT continuation bytes: (byte & 0xC0) != 0x80
    for (sz_size_t offset = 0; offset < length; offset += step) {
        svbool_t pg = svwhilelt_b8((sz_u64_t)offset, (sz_u64_t)length);
        svuint8_t text_vec = svld1_u8(pg, text_u8 + offset);
        svbool_t is_start = svcmpne_n_u8(pg, svand_n_u8_x(pg, text_vec, 0xC0), 0x80);
        char_count += svcntp_b8(pg, is_start);
    }
    return char_count;
}

/** @brief  Return a pointer to the start byte of the `n`-th UTF-8 codepoint, or `SZ_NULL_CHAR` if absent.
 *
 *  Single-window O(1) locate mirroring the RVV backend: each window zero-extends `svcntw()` bytes into 32-bit
 *  lanes (SVE2 has no `svcompact_u8`, so the index domain is 32-bit), skips whole windows by lead popcount, then
 *  `svcompact_u32` packs the lead-lane iota and `svlastb` reads the `n`-th packed lane. Byte-exact to serial. */
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_sve2(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const window_bytes = svcntw(); // one byte per 32-bit lane
    svuint32_t const lane_iota = svindex_u32(0, 1);
    while (length) {
        svbool_t const pg = svwhilelt_b32_u64(0, (sz_u64_t)length);
        svuint32_t const bytes_u32 = svld1ub_u32(pg, text_u8);
        // Leading byte iff `(byte & 0xC0) != 0x80`.
        svbool_t const lead = svcmpne_n_u32(pg, svand_n_u32_x(pg, bytes_u32, 0xC0), 0x80);
        sz_size_t const lead_count = svcntp_b32(pg, lead);
        if (n >= lead_count) {
            n -= lead_count;
            text_u8 += window_bytes, length -= sz_min_of_two(window_bytes, length);
            continue;
        }
        svuint32_t const packed = svcompact_u32(lead, lane_iota);
        sz_size_t const lane = svlastb_u32(svwhilelt_b32_u64(0, (sz_u64_t)(n + 1)), packed);
        return (sz_cptr_t)(text_u8 + lane);
    }
    return SZ_NULL_CHAR;
}

/*  Multistep newline / whitespace iteration (SVE2 / AArch64 scalable). Each tile classifies delimiter starts
 *  branchlessly into a per-lane start predicate + byte-length vector; the masked tail reads zero so a truncated
 *  multi-byte delimiter at EOF never matches (no serial tail). Starts are trusted in `[0, tile - 3]` on a full
 *  tile (2-/3-byte views fully loaded) and in every valid lane on the final tile. Output is capacity-cut to
 *  `matches_capacity`. For newlines a CRLF is one 2-byte match: a `text[pos-1] == '\r'` carry suppresses the
 *  straddling LF and a post-loop skip mirrors it; whitespace does no CRLF merging. */

/** @brief  Peel one tile's first `emit_count` matches into absolute `(offset, length)` pairs.
 *
 *  SVE2 has no `svcompact_u8`, so the byte-granular start/length lanes are walked in `svcntw()`-byte sub-blocks:
 *  each gathers its lanes with `svtbl_u8`, widens to 32-bit, `svcompact_u32` packs the matches, and the store is
 *  bounded by `emit_count` so the output never exceeds the remaining capacity. */
SZ_INTERNAL void sz_utf8_iterate_peel_tile_sve2_(                  //
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

SZ_PUBLIC sz_size_t sz_utf8_find_newlines_sve2(             //
    sz_cptr_t text, sz_size_t length,                       //
    sz_size_t *match_offsets, sz_size_t *match_lengths,     //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const step = svcntb();
    // Too narrow to host a safe 3-byte straddle and a 32-bit compaction sub-block? Delegate wholesale to serial.
    if (step < 16)
        return sz_utf8_find_newlines_serial(text, length, match_offsets, match_lengths, matches_capacity,
                                            bytes_consumed);

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

SZ_PUBLIC sz_size_t sz_utf8_find_whitespaces_sve2(          //
    sz_cptr_t text, sz_size_t length,                       //
    sz_size_t *match_offsets, sz_size_t *match_lengths,     //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const step = svcntb();
    if (step < 16)
        return sz_utf8_find_whitespaces_serial(text, length, match_offsets, match_lengths, matches_capacity,
                                               bytes_consumed);

    sz_size_t const tile = step < 64 ? step : 64;
    svuint8_t const zeros = svdup_n_u8(0);
    sz_size_t count = 0, position = 0;

    svuint8_t const one_byte_set = svdupq_n_u8(' ', '\t', '\n', '\v', '\f', '\r', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                                               ' ', ' ');
    // Valid third bytes for E2 80 XX: U+2000-U+200D (0x80-0x8D), plus U+2028 (0xA8), U+2029 (0xA9).
    svuint8_t const e280_third_bytes = svdupq_n_u8(0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A,
                                                   0x8B, 0x8C, 0x8D, 0xA8, 0xA9);

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
        svbool_t two_byte_mask = svand_b_z(
            pg, lead_c2_mask, svorr_b_z(pg, svcmpeq_n_u8(pg, text1, 0x85), svcmpeq_n_u8(pg, text1, 0xA0)));
        // 3-byte: E1 9A 80 (Ogham)
        svbool_t ogham_mask = svand_b_z(pg, svand_b_z(pg, svcmpeq_n_u8(pg, text0, 0xE1), svcmpeq_n_u8(pg, text1, 0x9A)),
                                        svcmpeq_n_u8(pg, text2, 0x80));
        // 3-byte: E2 80 [80-8D] | A8 | A9 (set match); E2 80 AF (NNBSP); E2 81 9F (MMSP)
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
 *  `sz_tr29_word_break_t` property, the no-break join is replayed on per-lane vectors (below), and boundaries
 *  (the join complement over trusted lanes `[2, svcntb() - 2]`) are compacted with `svcompact_u32`. */

/** @brief  Classify an all-ASCII window to `sz_tr29_word_break_t` properties via eight 16-entry `svtbl_u8` rows. */
SZ_INTERNAL svuint8_t sz_utf8_word_break_classify_ascii_sve2_(svbool_t pg, svuint8_t bytes) {
    // Eight rows of the ASCII Word_Break property table (high nibble → 16 low-nibble entries); a single `svtbl_u8`
    // reads 16 entries from lanes [0,16), so each row is broadcast with `svdupq_n_u8` and indexed by the low nibble.
    svuint8_t const row0 = svdupq_n_u8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0x03, 0x03, 0x01, 0, 0);
    svuint8_t const row1 = svdup_n_u8(0);
    svuint8_t const row2 = svdupq_n_u8(0, 0, 0x0F, 0, 0, 0, 0, 0x0F, 0, 0, 0, 0, 0x0E, 0, 0x0F, 0);
    svuint8_t const row3 =
        svdupq_n_u8(0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0D, 0x0E, 0, 0, 0, 0);
    svuint8_t const row4 =
        svdupq_n_u8(0, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08);
    svuint8_t const row5 =
        svdupq_n_u8(0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0, 0, 0, 0, 0x0C);
    svuint8_t const row6 =
        svdupq_n_u8(0, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08);
    svuint8_t const row7 =
        svdupq_n_u8(0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0, 0, 0, 0, 0);

    svuint8_t const low_nibble = svand_n_u8_x(pg, bytes, 0x0F);
    svuint8_t const high_nibble = svlsr_n_u8_x(pg, bytes, 4);
    svuint8_t classes = svtbl_u8(row0, low_nibble);
    classes = svsel_u8(svcmpeq_n_u8(pg, high_nibble, 1), svtbl_u8(row1, low_nibble), classes);
    classes = svsel_u8(svcmpeq_n_u8(pg, high_nibble, 2), svtbl_u8(row2, low_nibble), classes);
    classes = svsel_u8(svcmpeq_n_u8(pg, high_nibble, 3), svtbl_u8(row3, low_nibble), classes);
    classes = svsel_u8(svcmpeq_n_u8(pg, high_nibble, 4), svtbl_u8(row4, low_nibble), classes);
    classes = svsel_u8(svcmpeq_n_u8(pg, high_nibble, 5), svtbl_u8(row5, low_nibble), classes);
    classes = svsel_u8(svcmpeq_n_u8(pg, high_nibble, 6), svtbl_u8(row6, low_nibble), classes);
    classes = svsel_u8(svcmpeq_n_u8(pg, high_nibble, 7), svtbl_u8(row7, low_nibble), classes);
    return classes;
}

/** @brief  Vector replay of `sz_utf8_word_break_join_from_class_masks_`: lane `i` is 1 where the boundary before
 *          it is suppressed by a UAX-29 no-break rule. `<< k` becomes a `k`-lane up-shift (`svinsr`), `>> 1` a
 *          one-lane down-shift (`svext`). Intentional divergence: a scalable SVE register exceeds 64 bits at
 *          VLEN >= 1024, so there is no `sz_u64_t` movemask to shift — the cascade runs on per-lane 0/1 vectors. */
SZ_INTERNAL svuint8_t sz_utf8_word_break_join_lanes_sve2_(svbool_t pg, svuint8_t classes) {
    svuint8_t const aletter = svdup_u8_z(svcmpeq_n_u8(pg, classes, sz_tr29_word_break_aletter_k), 1);
    svuint8_t const numeric = svdup_u8_z(svcmpeq_n_u8(pg, classes, sz_tr29_word_break_numeric_k), 1);
    svuint8_t const extendnumlet = svdup_u8_z(svcmpeq_n_u8(pg, classes, sz_tr29_word_break_extendnumlet_k), 1);
    svuint8_t const midletter = svdup_u8_z(svcmpeq_n_u8(pg, classes, sz_tr29_word_break_midletter_k), 1);
    svuint8_t const midnum = svdup_u8_z(svcmpeq_n_u8(pg, classes, sz_tr29_word_break_midnum_k), 1);
    svuint8_t const mid_quotes = svdup_u8_z(svcmpeq_n_u8(pg, classes, sz_tr29_word_break_mid_quotes_k), 1);
    svuint8_t const carriage_return = svdup_u8_z(svcmpeq_n_u8(pg, classes, sz_tr29_word_break_cr_k), 1);
    svuint8_t const line_feed = svdup_u8_z(svcmpeq_n_u8(pg, classes, sz_tr29_word_break_lf_k), 1);
    svuint8_t const zeros = svdup_n_u8(0);

    svuint8_t const mid_letter_or_quotes = svorr_u8_x(pg, midletter, mid_quotes);
    svuint8_t const mid_num_or_quotes = svorr_u8_x(pg, midnum, mid_quotes);
    svuint8_t const aletter_or_numeric = svorr_u8_x(pg, aletter, numeric);

    // `prev_1` lane i holds lane i-1 (bitmask `<< 1`); `prev_2` holds i-2 (`<< 2`); `next_1` holds i+1 (`>> 1`).
    svuint8_t const aletter_prev_1 = svinsr_n_u8(aletter, 0);
    svuint8_t const aletter_prev_2 = svinsr_n_u8(aletter_prev_1, 0);
    svuint8_t const aletter_next_1 = svext_u8(aletter, zeros, 1);
    svuint8_t const numeric_prev_1 = svinsr_n_u8(numeric, 0);
    svuint8_t const numeric_prev_2 = svinsr_n_u8(numeric_prev_1, 0);
    svuint8_t const numeric_next_1 = svext_u8(numeric, zeros, 1);
    svuint8_t const mid_letter_or_quotes_prev_1 = svinsr_n_u8(mid_letter_or_quotes, 0);
    svuint8_t const mid_num_or_quotes_prev_1 = svinsr_n_u8(mid_num_or_quotes, 0);
    svuint8_t const extendnumlet_prev_1 = svinsr_n_u8(extendnumlet, 0);
    svuint8_t const aletter_or_numeric_or_extendnumlet_prev_1 =
        svinsr_n_u8(svorr_u8_x(pg, aletter_or_numeric, extendnumlet), 0);
    svuint8_t const carriage_return_prev_1 = svinsr_n_u8(carriage_return, 0);

    svuint8_t join = svand_u8_x(pg, carriage_return_prev_1, line_feed);                                   // WB3
    join = svorr_u8_x(pg, join, svand_u8_x(pg, aletter_prev_1, aletter));                                 // WB5
    join = svorr_u8_x(pg, join,
                      svand_u8_x(pg, svand_u8_x(pg, aletter_prev_1, mid_letter_or_quotes), aletter_next_1)); // WB6
    join = svorr_u8_x(pg, join,
                      svand_u8_x(pg, svand_u8_x(pg, aletter_prev_2, mid_letter_or_quotes_prev_1), aletter)); // WB7
    join = svorr_u8_x(pg, join, svand_u8_x(pg, numeric_prev_1, numeric));                                 // WB8
    join = svorr_u8_x(pg, join, svand_u8_x(pg, aletter_prev_1, numeric));                                 // WB9
    join = svorr_u8_x(pg, join, svand_u8_x(pg, numeric_prev_1, aletter));                                 // WB10
    join = svorr_u8_x(pg, join,
                      svand_u8_x(pg, svand_u8_x(pg, numeric_prev_1, mid_num_or_quotes), numeric_next_1)); // WB11
    join = svorr_u8_x(pg, join,
                      svand_u8_x(pg, svand_u8_x(pg, numeric_prev_2, mid_num_or_quotes_prev_1), numeric)); // WB12
    join = svorr_u8_x(pg, join, svand_u8_x(pg, aletter_or_numeric_or_extendnumlet_prev_1, extendnumlet)); // WB13a
    join = svorr_u8_x(pg, join, svand_u8_x(pg, extendnumlet_prev_1, aletter_or_numeric));                 // WB13b
    return join;
}

/** @brief  Compact the absolute byte positions of the set lanes of `boundary` (over lanes `[0, usable)`) into
 *          `out`, walking `svcntw()`-lane sub-blocks with `svcompact_u32`. Returns the number of positions. */
SZ_INTERNAL sz_size_t sz_utf8_word_compact_boundaries_sve2_(svbool_t boundary, sz_size_t window_base, sz_size_t usable,
                                                            sz_u32_t *out) {
    svuint8_t const boundary_flags = svdup_u8_z(boundary, 1);
    svuint8_t const lane_iota = svindex_u8(0, 1);
    sz_size_t const words = svcntw();
    sz_size_t produced = 0;
    for (sz_size_t base = 0; base < usable; base += words) {
        svbool_t const pw = svwhilelt_b32_u64((sz_u64_t)base, (sz_u64_t)usable);
        svuint8_t const gather_indices = svadd_n_u8_x(svptrue_b8(), lane_iota, (sz_u8_t)base);
        svuint32_t const flags_w = svunpklo_u32(svunpklo_u16(svtbl_u8(boundary_flags, gather_indices)));
        svbool_t const sub_boundary = svcmpne_n_u32(pw, flags_w, 0);
        if (!svptest_any(pw, sub_boundary)) continue;
        svuint32_t const positions = svindex_u32((sz_u32_t)(window_base + base), 1);
        svuint32_t const packed = svcompact_u32(sub_boundary, positions);
        sz_size_t const here = svcntp_b32(pw, sub_boundary);
        svst1_u32(svwhilelt_b32_u64(0, (sz_u64_t)here), out + produced, packed);
        produced += here;
    }
    return produced;
}

SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_sve2( //
    sz_cptr_t text, sz_size_t length,                  //
    sz_size_t *word_starts, sz_size_t *word_lengths,   //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_size_t const window_bytes = svcntb();
    if (window_bytes < 16) // Too narrow to host the [2, W-2] trusted window: defer wholesale to serial.
        return sz_utf8_word_find_boundaries_serial(text, length, word_starts, word_lengths, words_capacity,
                                                   bytes_consumed);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t word_start = 0; // Start of the word currently being accumulated (always a boundary).
    // Position 0 is always a boundary; the first reportable boundary is after the first codepoint.
    sz_size_t position = sz_utf8_codepoint_length_(text_u8[0]);
    // Sized to the 2048-bit SVE architectural maximum: one boundary per window byte, and `svcntb() <= 256`.
    sz_u32_t boundaries[256];

    // Oracle-free fast path: an all-ASCII window `[position-2, position-2+window_bytes)` resolves boundaries at
    // positions `[position, position+window_bytes-4]`, carrying the open `word_start` into the first emitted word.
    while (position < length) {
        sz_size_t const base = position - 2; // lane j = byte base+j; trusted lanes [2, window_bytes-2]
        sz_bool_t ascii_window = (sz_bool_t)(position >= 2 && position - 2 + window_bytes <= length);
        svbool_t const window_pg = svptrue_b8();
        svuint8_t window = svdup_n_u8(0);
        if (ascii_window) {
            window = svld1_u8(window_pg, text_u8 + base);
            ascii_window = (sz_bool_t)!svptest_any(window_pg, svcmpge_n_u8(window_pg, window, 0x80));
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

        svuint8_t const classes = sz_utf8_word_break_classify_ascii_sve2_(window_pg, window);
        svuint8_t const join = sz_utf8_word_break_join_lanes_sve2_(window_pg, classes);
        // Boundaries are the complement of `join` over the trusted lanes [2, window_bytes-2].
        svbool_t const trusted = svand_b_z(window_pg, svcmpge_n_u8(window_pg, svindex_u8(0, 1), 2),
                                           svcmple_n_u8(window_pg, svindex_u8(0, 1), (sz_u8_t)(window_bytes - 2)));
        svbool_t const boundary = svand_b_z(window_pg, trusted, svcmpeq_n_u8(window_pg, join, 0));

        sz_size_t const found = sz_utf8_word_compact_boundaries_sve2_(boundary, base, window_bytes, boundaries);
        sz_size_t const emit = sz_min_of_two(found, words_capacity - words);
        for (sz_size_t i = 0; i < emit; ++i) {
            sz_size_t const next_boundary = boundaries[i];
            word_starts[words] = word_start, word_lengths[words] = next_boundary - word_start, ++words;
            word_start = next_boundary;
        }
        if (words == words_capacity) {
            if (bytes_consumed) *bytes_consumed = word_start;
            return words;
        }
        // Resolved [position, position+window_bytes-4]; if all boundaries fit, the next unresolved boundary is at
        // position+window_bytes-3, otherwise resume from the open word start to re-classify the dropped tail.
        position = emit < found ? word_start : position + window_bytes - 3;
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

SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_sve2( //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *word_starts, sz_size_t *word_lengths,    //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = length;
        return 0;
    }

    sz_size_t const window_bytes = svcntb();
    if (window_bytes < 16)
        return sz_utf8_word_rfind_boundaries_serial(text, length, word_starts, word_lengths, words_capacity,
                                                    bytes_consumed);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t word_end = length; // End of the word currently being accumulated (always a boundary).
    sz_size_t position = length - 1;
    while (position > 0 && (text_u8[position] & 0xC0) == 0x80) position--;
    // Sized to the 2048-bit SVE architectural maximum: one boundary per window byte, and `svcntb() <= 256`.
    sz_u32_t boundaries[256];

    // Oracle-free fast path: an all-ASCII window `[position-(window_bytes-2), position+2)` resolves boundaries at
    // positions `[position-(window_bytes-4), position]`, walked high-to-low and carrying the open `word_end`.
    while (position > 0) {
        sz_size_t const base = position - (window_bytes - 2); // lane j = byte base+j; trusted [2, window_bytes-2]
        sz_bool_t ascii_window = (sz_bool_t)(position >= window_bytes - 2 && position + 2 <= length);
        svbool_t const window_pg = svptrue_b8();
        svuint8_t window = svdup_n_u8(0);
        if (ascii_window) {
            window = svld1_u8(window_pg, text_u8 + base);
            ascii_window = (sz_bool_t)!svptest_any(window_pg, svcmpge_n_u8(window_pg, window, 0x80));
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

        svuint8_t const classes = sz_utf8_word_break_classify_ascii_sve2_(window_pg, window);
        svuint8_t const join = sz_utf8_word_break_join_lanes_sve2_(window_pg, classes);
        svbool_t const trusted = svand_b_z(window_pg, svcmpge_n_u8(window_pg, svindex_u8(0, 1), 2),
                                           svcmple_n_u8(window_pg, svindex_u8(0, 1), (sz_u8_t)(window_bytes - 2)));
        svbool_t const boundary = svand_b_z(window_pg, trusted, svcmpeq_n_u8(window_pg, join, 0));

        sz_size_t const found = sz_utf8_word_compact_boundaries_sve2_(boundary, base, window_bytes, boundaries);
        sz_size_t const emit = sz_min_of_two(found, words_capacity - words);
        // Emit high-to-low: the compacted positions are ascending, so walk them from the tail.
        for (sz_size_t i = 0; i < emit; ++i) {
            sz_size_t const this_boundary = boundaries[found - 1 - i];
            word_starts[words] = this_boundary, word_lengths[words] = word_end - this_boundary, ++words;
            word_end = this_boundary;
        }
        if (words == words_capacity) {
            if (bytes_consumed) *bytes_consumed = word_end;
            return words;
        }
        // Resolved down to position-(window_bytes-4); if all fit, the next unresolved boundary is at
        // position-(window_bytes-3), otherwise resume from the open word end to re-classify the dropped head.
        position = emit < found ? word_end : base + 1;
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
#endif // SZ_USE_SVE2

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_ITERATE_SVE2_H_
