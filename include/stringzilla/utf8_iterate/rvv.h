/**
 *  @brief RISC-V Vector (RVV 1.0) backend for utf8.
 *  @file include/stringzilla/utf8_iterate/rvv.h
 *  @author Ash Vardanian
 *  @sa utf8.h
 */
#ifndef STRINGZILLA_UTF8_ITERATE_RVV_H_
#define STRINGZILLA_UTF8_ITERATE_RVV_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_iterate/serial.h"
#include "stringzilla/utf8_runes.h" // `sz_rune_parse_unchecked`

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_RVV
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

/**
 *  @brief Decode UTF-8 to UTF-32 codepoints, byte-for-byte equivalent to `sz_utf8_unpack_chunk_serial`.
 *
 *  Vector-classifies the maximal ASCII prefix of each strip (`vmsgtu` high bit, `vfirst`) and widens it
 *  straight into the rune output; any non-ASCII byte takes a single serial `sz_rune_parse` step.
 */
/** @brief Widen `count` ASCII bytes (u8 -> u16 -> u32) and store them as runes. */
SZ_INTERNAL void sz_utf8_rvv_store_ascii_run_(sz_rune_t *runes_out, sz_u8_t const *src, sz_size_t count) {
    sz_size_t done = 0;
    while (done < count) {
        sz_size_t widened_vector_length = __riscv_vsetvl_e8m2(count - done);
        vuint8m2_t ascii_u8m2 = __riscv_vle8_v_u8m2(src + done, widened_vector_length);
        vuint16m4_t ascii_u16m4 = __riscv_vwcvtu_x_x_v_u16m4(ascii_u8m2, widened_vector_length);
        vuint32m8_t ascii_u32m8 = __riscv_vwcvtu_x_x_v_u32m8(ascii_u16m4, widened_vector_length);
        __riscv_vse32_v_u32m8(runes_out + done, ascii_u32m8, widened_vector_length);
        done += widened_vector_length;
    }
}

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_rvv(   //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {

    sz_u8_t const *text_cursor = (sz_u8_t const *)text;
    sz_u8_t const *text_end = text_cursor + length;
    sz_size_t runes_written = 0;

    while (text_cursor < text_end && runes_written < runes_capacity) {
        // Consume a maximal ASCII run, bounded by remaining input and remaining capacity.
        sz_size_t bytes_left = (sz_size_t)(text_end - text_cursor);
        sz_size_t cap_left = runes_capacity - runes_written;
        sz_size_t want = bytes_left < cap_left ? bytes_left : cap_left;

        sz_size_t ascii_run = 0;
        while (ascii_run < want) {
            sz_size_t vector_length = __riscv_vsetvl_e8m8(want - ascii_run);
            vuint8m8_t bytes_u8m8 = __riscv_vle8_v_u8m8(text_cursor + ascii_run, vector_length);
            vbool1_t non_ascii = __riscv_vmsgtu_vx_u8m8_b1(bytes_u8m8, 0x7F, vector_length);
            long first_match_index = __riscv_vfirst_m_b1(non_ascii, vector_length);
            sz_size_t take = first_match_index < 0 ? vector_length : (sz_size_t)first_match_index;
            if (take) sz_utf8_rvv_store_ascii_run_(runes + runes_written + ascii_run, text_cursor + ascii_run, take);
            ascii_run += take;
            if (first_match_index >= 0) break; // Hit a non-ASCII byte.
        }
        runes_written += ascii_run;
        text_cursor += ascii_run;
        if (runes_written >= runes_capacity || text_cursor >= text_end) break;

        // Now `*text_cursor` is a multi-byte lead (or a stray continuation). Decode exactly like serial.
        sz_rune_t rune;
        sz_rune_length_t const rune_length = sz_rune_parse_unchecked((sz_cptr_t)text_cursor, &rune);
        if (text_cursor + rune_length > text_end) break; // Incomplete sequence at buffer boundary.
        runes[runes_written++] = rune;
        text_cursor += rune_length;
    }

    *runes_unpacked = runes_written;
    return (sz_cptr_t)text_cursor;
}

/** @brief Count UTF-8 codepoints: `vcpop` the leading (non-continuation) bytes per strip. */
SZ_PUBLIC sz_size_t sz_utf8_count_rvv(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0;
    while (length) {
        sz_size_t vector_length = __riscv_vsetvl_e8m8(length);
        vuint8m8_t bytes_u8m8 = __riscv_vle8_v_u8m8(text_u8, vector_length);
        // Leading byte iff `(c & 0xC0) != 0x80`.
        vuint8m8_t masked = __riscv_vand_vx_u8m8(bytes_u8m8, 0xC0, vector_length);
        vbool1_t lead_mask = __riscv_vmsne_vx_u8m8_b1(masked, 0x80, vector_length);
        count += (sz_size_t)__riscv_vcpop_m_b1(lead_mask, vector_length);
        text_u8 += vector_length, length -= vector_length;
    }
    return count;
}

/**
 *  @brief Locate the start of the n-th codepoint, byte-for-byte equivalent to the serial baseline.
 *
 *  Skips whole strips by `vcpop` of leading bytes, then locates the wanted lead in-register with `viota`
 *  (the lane that is a lead and whose prefix count equals the target). Runs at `e8m4` so the `vbool2`
 *  lead mask pairs with a `u16m8` iota whose lane count never overflows the prefix counts.
 */
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_rvv(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t seen = 0;
    while (length) {
        sz_size_t vector_length = __riscv_vsetvl_e8m4(length);
        vuint8m4_t bytes_u8m4 = __riscv_vle8_v_u8m4(text_u8, vector_length);
        vbool2_t lead_mask = __riscv_vmsne_vx_u8m4_b2(__riscv_vand_vx_u8m4(bytes_u8m4, 0xC0, vector_length), 0x80,
                                                      vector_length);
        sz_size_t strip_leads = (sz_size_t)__riscv_vcpop_m_b2(lead_mask, vector_length);
        if (seen + strip_leads > n) {
            sz_u16_t target = (sz_u16_t)(n - seen); // 0-based index of the wanted lead byte within this strip
            vuint16m8_t prefix = __riscv_viota_m_u16m8(lead_mask, vector_length);
            vbool2_t hit = __riscv_vmand_mm_b2(__riscv_vmseq_vx_u16m8_b2(prefix, target, vector_length), lead_mask,
                                               vector_length);
            return (sz_cptr_t)(text_u8 + __riscv_vfirst_m_b2(hit, vector_length));
        }
        seen += strip_leads;
        text_u8 += vector_length, length -= vector_length;
    }
    return SZ_NULL_CHAR;
}

/*  Multistep newline / whitespace iteration (RVV 1.0).
 *
 *  Fully masked / streaming: each iteration loads `vl = vsetvl_e8m4(length - position)` lanes, so the final
 *  iteration is the masked tail (carry bytes past `length` read as 0, so a truncated delimiter at EOF never
 *  matches) — no serial tail call. A small per-set classifier builds the per-lane byte-length vector; one
 *  shared driver (`sz_utf8_iterate_multistep_rvv_`) handles the window/carry/trusted-lane/peel scaffolding. */

/*  Cap the logical tile so a `u16` iota addresses every lane. RVV 1.0 allows `VLEN` up to 64 Kib, i.e.
 *  `e8m4` `VLMAX` up to 32768 — well within `u16`. */
SZ_INTERNAL sz_size_t sz_utf8_iterate_tile_bytes_rvv_(void) {
    sz_size_t vlmax = __riscv_vsetvlmax_e8m4();
    return vlmax < 4 ? 4 : vlmax; // need at least 4 lanes so the trusted region [0, tile-3] is non-empty
}

/** @brief Peel a window's first `emit_count` matches: compress lane offsets + lengths, widen-store absolute pairs. */
SZ_INTERNAL void sz_utf8_iterate_peel_tile_rvv_(                          //
    vuint8m4_t length_u8m4, vbool2_t start_mask, sz_size_t tile_position, //
    sz_size_t vector_length, sz_size_t emit_count, sz_size_t *match_offsets, sz_size_t *match_lengths) {

    if (!emit_count) return;

    // `u16m8` and `u8m4` share the SEW/LMUL ratio (-> `vbool2`), so one mask drives both compactions.
    vuint16m8_t lane_iota_u16m8 = __riscv_vid_v_u16m8(vector_length);
    vuint16m8_t compact_indices_u16m8 = __riscv_vcompress_vm_u16m8(lane_iota_u16m8, start_mask, vector_length);
    vuint8m4_t compact_lengths_u8m4 = __riscv_vcompress_vm_u8m4(length_u8m4, start_mask, vector_length);

    // Widen only the low `emit_count` compacted indices to 64-bit, add the absolute tile base, store both arrays.
    sz_size_t store_offset = 0;
    while (store_offset < emit_count) {
        sz_size_t store_length = __riscv_vsetvl_e64m8(emit_count - store_offset);
        vuint16m2_t indices_u16m2 = __riscv_vget_v_u16m8_u16m2(
            __riscv_vslidedown_vx_u16m8(compact_indices_u16m8, store_offset, vector_length), 0);
        vuint8m1_t lengths_u8m1 = __riscv_vget_v_u8m4_u8m1(
            __riscv_vslidedown_vx_u8m4(compact_lengths_u8m4, store_offset, vector_length), 0);

        vuint64m8_t offsets_u64m8 = __riscv_vadd_vx_u64m8(
            __riscv_vwcvtu_x_x_v_u64m8(__riscv_vwcvtu_x_x_v_u32m4(indices_u16m2, store_length), store_length),
            (sz_u64_t)tile_position, store_length);
        __riscv_vse64_v_u64m8((sz_u64_t *)match_offsets + store_offset, offsets_u64m8, store_length);

        vuint64m8_t lengths_u64m8 = __riscv_vwcvtu_x_x_v_u64m8(
            __riscv_vwcvtu_x_x_v_u32m4(__riscv_vwcvtu_x_x_v_u16m2(lengths_u8m1, store_length), store_length),
            store_length);
        __riscv_vse64_v_u64m8((sz_u64_t *)match_lengths + store_offset, lengths_u64m8, store_length);
        store_offset += store_length;
    }
}

/*  Classify a tile into a per-lane byte-length vector (0 = no delimiter starts here). The 2nd/3rd bytes are
 *  carried from the buffer via `next`/`after_next`; multi-byte masks are computed unconditionally. */
SZ_INTERNAL vuint8m4_t sz_utf8_classify_newlines_rvv_(sz_u8_t const *text_u8, sz_size_t position, vuint8m4_t bytes_u8m4,
                                                      vuint8m4_t next_u8m4, vuint8m4_t after_next_u8m4,
                                                      sz_size_t vector_length) {
    vuint8m4_t length_u8m4 = __riscv_vmv_v_x_u8m4(0, vector_length);
    // '\n' '\v' '\f' (0x0A-0x0C) and a lone '\r' (0x0D): length 1.
    vbool2_t is_lf_vt_ff = __riscv_vmsltu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(bytes_u8m4, 0x0A, vector_length), 3,
                                                     vector_length);
    vbool2_t is_cr = __riscv_vmseq_vx_u8m4_b2(bytes_u8m4, 0x0D, vector_length);
    length_u8m4 = __riscv_vmerge_vxm_u8m4(length_u8m4, 1, __riscv_vmor_mm_b2(is_lf_vt_ff, is_cr, vector_length),
                                          vector_length);
    // '\r\n': length 2 (overrides the lone-'\r').
    vbool2_t is_cr_lf = __riscv_vmand_mm_b2(is_cr, __riscv_vmseq_vx_u8m4_b2(next_u8m4, 0x0A, vector_length),
                                            vector_length);
    length_u8m4 = __riscv_vmerge_vxm_u8m4(length_u8m4, 2, is_cr_lf, vector_length);
    // U+0085 NEL (C2 85): length 2.
    vbool2_t is_nel = __riscv_vmand_mm_b2(__riscv_vmseq_vx_u8m4_b2(bytes_u8m4, 0xC2, vector_length),
                                          __riscv_vmseq_vx_u8m4_b2(next_u8m4, 0x85, vector_length), vector_length);
    length_u8m4 = __riscv_vmerge_vxm_u8m4(length_u8m4, 2, is_nel, vector_length);
    // U+2028 / U+2029 (E2 80 A8 / E2 80 A9): length 3.
    vbool2_t is_e2_80 = __riscv_vmand_mm_b2(__riscv_vmseq_vx_u8m4_b2(bytes_u8m4, 0xE2, vector_length),
                                            __riscv_vmseq_vx_u8m4_b2(next_u8m4, 0x80, vector_length), vector_length);
    vbool2_t is_a8_a9 = __riscv_vmor_mm_b2(__riscv_vmseq_vx_u8m4_b2(after_next_u8m4, 0xA8, vector_length),
                                           __riscv_vmseq_vx_u8m4_b2(after_next_u8m4, 0xA9, vector_length),
                                           vector_length);
    length_u8m4 = __riscv_vmerge_vxm_u8m4(length_u8m4, 3, __riscv_vmand_mm_b2(is_e2_80, is_a8_a9, vector_length),
                                          vector_length);
    // An LF that is the 2nd byte of a CRLF is owned by the CR's match: the previous byte (carried via
    // `vslide1up`) being '\r' identifies it, also covering a CRLF that straddled the previous tile edge.
    sz_u8_t carry_prev = (position != 0) ? text_u8[position - 1] : 0;
    vuint8m4_t prev_u8m4 = __riscv_vslide1up_vx_u8m4(bytes_u8m4, carry_prev, vector_length);
    vbool2_t is_lf_of_crlf = __riscv_vmand_mm_b2(__riscv_vmseq_vx_u8m4_b2(bytes_u8m4, 0x0A, vector_length),
                                                 __riscv_vmseq_vx_u8m4_b2(prev_u8m4, 0x0D, vector_length),
                                                 vector_length);
    return __riscv_vmerge_vxm_u8m4(length_u8m4, 0, is_lf_of_crlf, vector_length);
}

SZ_INTERNAL vuint8m4_t sz_utf8_classify_whitespaces_rvv_(vuint8m4_t bytes_u8m4, vuint8m4_t next_u8m4,
                                                         vuint8m4_t after_next_u8m4, sz_size_t vector_length) {
    vuint8m4_t length_u8m4 = __riscv_vmv_v_x_u8m4(0, vector_length);
    // ASCII whitespace: '\t'-'\r' (0x09-0x0D) and ' ' (0x20): length 1.
    vbool2_t is_ascii_ws = __riscv_vmor_mm_b2(
        __riscv_vmsltu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(bytes_u8m4, 0x09, vector_length), 5, vector_length),
        __riscv_vmseq_vx_u8m4_b2(bytes_u8m4, 0x20, vector_length), vector_length);
    length_u8m4 = __riscv_vmerge_vxm_u8m4(length_u8m4, 1, is_ascii_ws, vector_length);
    // U+0085 NEL (C2 85) and U+00A0 NBSP (C2 A0): length 2.
    vbool2_t is_c2 = __riscv_vmseq_vx_u8m4_b2(bytes_u8m4, 0xC2, vector_length);
    vbool2_t next_85_a0 = __riscv_vmor_mm_b2(__riscv_vmseq_vx_u8m4_b2(next_u8m4, 0x85, vector_length),
                                             __riscv_vmseq_vx_u8m4_b2(next_u8m4, 0xA0, vector_length), vector_length);
    length_u8m4 = __riscv_vmerge_vxm_u8m4(length_u8m4, 2, __riscv_vmand_mm_b2(is_c2, next_85_a0, vector_length),
                                          vector_length);
    // U+1680 Ogham space mark (E1 9A 80): length 3.
    vbool2_t is_ogham = __riscv_vmand_mm_b2(
        __riscv_vmand_mm_b2(__riscv_vmseq_vx_u8m4_b2(bytes_u8m4, 0xE1, vector_length),
                            __riscv_vmseq_vx_u8m4_b2(next_u8m4, 0x9A, vector_length), vector_length),
        __riscv_vmseq_vx_u8m4_b2(after_next_u8m4, 0x80, vector_length), vector_length);
    length_u8m4 = __riscv_vmerge_vxm_u8m4(length_u8m4, 3, is_ogham, vector_length);
    // U+2000-U+200D, U+2028, U+2029, U+202F (E2 80 {80-8D, A8, A9, AF}) and U+205F (E2 81 9F): length 3.
    vbool2_t is_e2 = __riscv_vmseq_vx_u8m4_b2(bytes_u8m4, 0xE2, vector_length);
    vbool2_t is_e2_80 = __riscv_vmand_mm_b2(is_e2, __riscv_vmseq_vx_u8m4_b2(next_u8m4, 0x80, vector_length),
                                            vector_length);
    vbool2_t third_2000_200d = __riscv_vmsltu_vx_u8m4_b2(__riscv_vsub_vx_u8m4(after_next_u8m4, 0x80, vector_length),
                                                         0x0E, vector_length);
    vbool2_t third_a8_a9_af = __riscv_vmor_mm_b2(
        __riscv_vmor_mm_b2(__riscv_vmseq_vx_u8m4_b2(after_next_u8m4, 0xA8, vector_length),
                           __riscv_vmseq_vx_u8m4_b2(after_next_u8m4, 0xA9, vector_length), vector_length),
        __riscv_vmseq_vx_u8m4_b2(after_next_u8m4, 0xAF, vector_length), vector_length);
    vbool2_t is_e2_80_ws = __riscv_vmand_mm_b2(
        is_e2_80, __riscv_vmor_mm_b2(third_2000_200d, third_a8_a9_af, vector_length), vector_length);
    vbool2_t is_e2_81_9f = __riscv_vmand_mm_b2(
        __riscv_vmand_mm_b2(is_e2, __riscv_vmseq_vx_u8m4_b2(next_u8m4, 0x81, vector_length), vector_length),
        __riscv_vmseq_vx_u8m4_b2(after_next_u8m4, 0x9F, vector_length), vector_length);
    length_u8m4 = __riscv_vmerge_vxm_u8m4(length_u8m4, 3, __riscv_vmor_mm_b2(is_e2_80_ws, is_e2_81_9f, vector_length),
                                          vector_length);
    // U+3000 ideographic space (E3 80 80): length 3.
    vbool2_t is_e3 = __riscv_vmand_mm_b2(
        __riscv_vmand_mm_b2(__riscv_vmseq_vx_u8m4_b2(bytes_u8m4, 0xE3, vector_length),
                            __riscv_vmseq_vx_u8m4_b2(next_u8m4, 0x80, vector_length), vector_length),
        __riscv_vmseq_vx_u8m4_b2(after_next_u8m4, 0x80, vector_length), vector_length);
    return __riscv_vmerge_vxm_u8m4(length_u8m4, 3, is_e3, vector_length);
}

/*  Shared window/carry/trusted-lane/peel scaffolding for both delimiter sets. `classify_newlines` selects the
 *  per-lane classifier and the post-loop CRLF straddle fixup; otherwise the two paths are identical. */
SZ_INTERNAL sz_size_t sz_utf8_iterate_multistep_rvv_(   //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed, int classify_newlines) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;
    sz_size_t tile = sz_utf8_iterate_tile_bytes_rvv_();

    while (position < length && count < matches_capacity) {
        sz_size_t vector_length = __riscv_vsetvl_e8m4(length - position);
        vuint8m4_t bytes_u8m4 = __riscv_vle8_v_u8m4(text_u8 + position, vector_length);
        sz_u8_t carry_first = (position + vector_length < length) ? text_u8[position + vector_length] : 0;
        sz_u8_t carry_second = (position + vector_length + 1 < length) ? text_u8[position + vector_length + 1] : 0;
        vuint8m4_t next_u8m4 = __riscv_vslide1down_vx_u8m4(bytes_u8m4, carry_first, vector_length);
        vuint8m4_t after_next_u8m4 = __riscv_vslide1down_vx_u8m4(next_u8m4, carry_second, vector_length);

        vuint8m4_t length_u8m4 = classify_newlines
                                     ? sz_utf8_classify_newlines_rvv_(text_u8, position, bytes_u8m4, next_u8m4,
                                                                      after_next_u8m4, vector_length)
                                     : sz_utf8_classify_whitespaces_rvv_(bytes_u8m4, next_u8m4, after_next_u8m4,
                                                                         vector_length);

        // Trust starts in lanes [0, vl-3] on a full tile so =<3-byte tails are fully loaded; on the final
        // (masked) tile (`vl < tile`) trust every valid lane (carried tails are real bytes or the at-EOF 0).
        sz_size_t trusted_lanes = (vector_length >= tile) ? vector_length - 2 : vector_length;
        vuint16m8_t lane_iota_u16m8 = __riscv_vid_v_u16m8(vector_length);
        vbool2_t trusted_mask = __riscv_vmsltu_vx_u16m8_b2(lane_iota_u16m8, (sz_u16_t)trusted_lanes, vector_length);
        vbool2_t start_mask = __riscv_vmand_mm_b2(__riscv_vmsne_vx_u8m4_b2(length_u8m4, 0, vector_length), trusted_mask,
                                                  vector_length);

        // Capacity cut: emit only as many of this tile's matches as the output can still hold.
        sz_size_t window_matches = (sz_size_t)__riscv_vcpop_m_b2(start_mask, vector_length);
        sz_size_t emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        sz_utf8_iterate_peel_tile_rvv_(length_u8m4, start_mask, position, vector_length, emit_count,
                                       match_offsets + count, match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) { // output buffer full: resume past the last emitted match
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += (vector_length >= tile) ? tile - 2 : vector_length;
    }

    // For newlines: if the loop stopped on the LF of an already-emitted CRLF, resume past it.
    if (classify_newlines && position != 0 && position < length && text_u8[position - 1] == '\r' &&
        text_u8[position] == '\n')
        ++position;
    if (bytes_consumed) *bytes_consumed = position;
    return count;
}

SZ_PUBLIC sz_size_t sz_utf8_find_newlines_rvv(          //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {
    return sz_utf8_iterate_multistep_rvv_(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed,
                                          1);
}

SZ_PUBLIC sz_size_t sz_utf8_find_whitespaces_rvv(       //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {
    return sz_utf8_iterate_multistep_rvv_(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed,
                                          0);
}

/*  UAX-29 word-boundary detection (TR29 Word_Break), byte-identical to the serial reference.
 *
 *  An outer window driver (`_rvv_` / `_rfind_`) slides an `e8m4` window anchored two bytes before the open
 *  boundary, so trusted lanes `[2, vl-2]` carry the full `i-2..i+1` neighbour context. An all-ASCII window is
 *  classified from its raw bytes and reduced to a per-lane boundary mask, compacted, biased to absolute
 *  offsets, and emitted as a shifted-difference `(start, length)` stream. Any non-ASCII / edge window takes
 *  one scalar codepoint step; malformed UTF-8 (which a single window scan cannot reproduce) is routed
 *  wholesale to serial by the well-formedness gate below. */

/*  Vectorized UTF-8 well-formedness gate. The serial code mixes lead-length enumeration with backward decode,
 *  which disagree on malformed input, so only well-formed buffers may take the table path. Pure ASCII is
 *  trivially well-formed and detected with a single `vfirst` "any byte >= 0x80" scan (folding in the
 *  always-malformed 0xF8..0xFF reject); a non-ASCII buffer then takes one linear framing walk. */
SZ_INTERNAL int sz_utf8_word_break_is_well_formed_rvv_(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t offset = 0;
    int has_non_ascii = 0;
    while (offset < length) {
        sz_size_t vector_length = __riscv_vsetvl_e8m8(length - offset);
        vuint8m8_t bytes_u8m8 = __riscv_vle8_v_u8m8(text_u8 + offset, vector_length);
        // 0xF8..0xFF is a subset of >= 0x80, so the always-malformed reject only runs on high-byte windows.
        if (__riscv_vfirst_m_b1(__riscv_vmsgtu_vx_u8m8_b1(bytes_u8m8, 0x7F, vector_length), vector_length) >= 0) {
            has_non_ascii = 1;
            if (__riscv_vfirst_m_b1(__riscv_vmsgtu_vx_u8m8_b1(bytes_u8m8, 0xF7, vector_length), vector_length) >= 0)
                return 0;
        }
        offset += vector_length;
    }
    // Pure ASCII is trivially well-formed: skip the per-byte framing walk entirely.
    if (!has_non_ascii) return 1;
    // Confirm sequence framing for the non-ASCII buffer with a single linear walk (decode vs char-length).
    offset = 0;
    while (offset < length) {
        sz_u8_t lead = text_u8[offset];
        sz_size_t codepoint_length = sz_utf8_codepoint_length_(lead);
        if ((lead & 0xC0) == 0x80) return 0;              // stray continuation byte at a lead position
        if (offset + codepoint_length > length) return 0; // truncated trailing sequence
        for (sz_size_t byte_index = 1; byte_index < codepoint_length; ++byte_index)
            if ((text_u8[offset + byte_index] & 0xC0) != 0x80) return 0; // missing continuation byte
        offset += codepoint_length;
    }
    return 1;
}

/** @brief Local word-break decision: 1 (break), 0 (no break), or -1 (stateful rule — consult serial). */
SZ_INTERNAL int sz_utf8_word_break_local_decision_rvv_(sz_u8_t prev_prop, sz_u8_t after_raw, sz_u8_t after_prop) {
    // WB3: CR x LF
    if (prev_prop == sz_tr29_word_break_cr_k && after_raw == sz_tr29_word_break_lf_k) return 0;
    // WB3a: break after Newline/CR/LF
    if (prev_prop == sz_tr29_word_break_newline_k || prev_prop == sz_tr29_word_break_cr_k ||
        prev_prop == sz_tr29_word_break_lf_k)
        return 1;
    // WB3b: break before Newline/CR/LF (raw after, as serial)
    if (after_raw == sz_tr29_word_break_newline_k || after_raw == sz_tr29_word_break_cr_k ||
        after_raw == sz_tr29_word_break_lf_k)
        return 1;
    // WB3c: ZWJ x (anything)
    if (prev_prop == sz_tr29_word_break_zwj_k) return 0;

    // From here serial uses the WB4-effective `after_prop`.
    int prev_ah = (prev_prop == sz_tr29_word_break_aletter_k || prev_prop == sz_tr29_word_break_hebrew_letter_k);
    int after_ah = (after_prop == sz_tr29_word_break_aletter_k || after_prop == sz_tr29_word_break_hebrew_letter_k);
    int prev_num = (prev_prop == sz_tr29_word_break_numeric_k);
    int after_num = (after_prop == sz_tr29_word_break_numeric_k);
    int after_q = (after_prop == sz_tr29_word_break_mid_quotes_k);

    // WB5: AHLetter x AHLetter
    if (prev_ah && after_ah) return 0;

    // WB6 trigger: AHLetter x (MidLetter|MidNumLetQ) -> needs lookahead.
    if (prev_ah && (after_prop == sz_tr29_word_break_midletter_k || after_q)) return -1;

    // WB7 trigger: (MidLetter|MidNumLetQ) x AHLetter -> needs lookback.
    if ((prev_prop == sz_tr29_word_break_midletter_k || prev_prop == sz_tr29_word_break_mid_quotes_k) && after_ah)
        return -1;

    // WB7a: Hebrew_Letter x Single_Quote (mid-quotes)
    if (prev_prop == sz_tr29_word_break_hebrew_letter_k && after_prop == sz_tr29_word_break_mid_quotes_k) return 0;

    // WB8: Numeric x Numeric
    if (prev_num && after_num) return 0;
    // WB9: AHLetter x Numeric
    if (prev_ah && after_num) return 0;
    // WB10: Numeric x AHLetter
    if (prev_num && after_ah) return 0;

    // WB11 trigger: Numeric x (MidNum|MidNumLetQ) -> needs lookahead.
    if (prev_num && (after_prop == sz_tr29_word_break_midnum_k || after_q)) return -1;
    // WB12 trigger: (MidNum|MidNumLetQ) x Numeric -> needs lookback.
    if ((prev_prop == sz_tr29_word_break_midnum_k || prev_prop == sz_tr29_word_break_mid_quotes_k) && after_num)
        return -1;

    // WB13: Katakana x Katakana
    if (prev_prop == sz_tr29_word_break_katakana_k && after_prop == sz_tr29_word_break_katakana_k) return 0;
    // WB13a: (AHLetter|Numeric|Katakana|ExtendNumLet) x ExtendNumLet
    if ((prev_ah || prev_num || prev_prop == sz_tr29_word_break_katakana_k ||
         prev_prop == sz_tr29_word_break_extendnumlet_k) &&
        after_prop == sz_tr29_word_break_extendnumlet_k)
        return 0;
    // WB13b: ExtendNumLet x (AHLetter|Numeric|Katakana)
    if (prev_prop == sz_tr29_word_break_extendnumlet_k &&
        (after_ah || after_num || after_prop == sz_tr29_word_break_katakana_k))
        return 0;

    // WB15/16 trigger: RI x RI -> parity dependent.
    if (prev_prop == sz_tr29_word_break_regional_ind_k && after_prop == sz_tr29_word_break_regional_ind_k) return -1;

    // WB999: break.
    return 1;
}

/** @brief Per-codepoint non-ASCII word-break step, reproducing `sz_utf8_is_word_boundary_serial` exactly. */
SZ_INTERNAL sz_bool_t sz_utf8_word_break_decision_at_rvv_(sz_cptr_t text, sz_size_t length, sz_size_t position) {
    sz_u8_t prev_prop = sz_utf8_word_break_previous_property_(text, position);
    sz_size_t after_position = position;
    sz_u8_t after_raw = sz_rune_word_break_property(sz_utf8_decode_(text, length, &after_position));
    sz_u8_t after_prop = sz_utf8_word_break_effective_property_(text, length, position, (sz_size_t *)0);
    int decision = sz_utf8_word_break_local_decision_rvv_(prev_prop, after_raw, after_prop);
    return (decision < 0) ? sz_utf8_is_word_boundary_serial(text, length, position) : (sz_bool_t)decision;
}

/*  All-ASCII window word-break boundary mask. ASCII has no Extend/Format/ZWJ/RI/Hebrew/Katakana, so the
 *  look-around rules reduce to neighbour slides (`vslide1up` for i-1/i-2, `vslide1down` for i+1). Bit `i` is
 *  set when the boundary BEFORE lane `i` is not suppressed, restricted to trusted lanes [2, vl-2]. */
SZ_INTERNAL vbool2_t sz_utf8_word_break_ascii_boundary_mask_rvv_(vuint8m4_t props_u8m4, sz_size_t vector_length) {
    // Effective neighbour properties via slides (the fill values land outside the trusted lanes).
    vuint8m4_t prev1_u8m4 = __riscv_vslide1up_vx_u8m4(props_u8m4, 0, vector_length);   // lane i-1
    vuint8m4_t prev2_u8m4 = __riscv_vslide1up_vx_u8m4(prev1_u8m4, 0, vector_length);   // lane i-2
    vuint8m4_t next1_u8m4 = __riscv_vslide1down_vx_u8m4(props_u8m4, 0, vector_length); // lane i+1

    // Per-class predicates for this lane and its slid neighbours. ASCII has no Hebrew letter, so AHLetter == ALetter.
    vbool2_t cur_aletter = __riscv_vmseq_vx_u8m4_b2(props_u8m4, sz_tr29_word_break_aletter_k, vector_length);
    vbool2_t cur_numeric = __riscv_vmseq_vx_u8m4_b2(props_u8m4, sz_tr29_word_break_numeric_k, vector_length);
    vbool2_t cur_extendnumlet = __riscv_vmseq_vx_u8m4_b2(props_u8m4, sz_tr29_word_break_extendnumlet_k, vector_length);
    vbool2_t cur_lf = __riscv_vmseq_vx_u8m4_b2(props_u8m4, sz_tr29_word_break_lf_k, vector_length);

    vbool2_t prev1_aletter = __riscv_vmseq_vx_u8m4_b2(prev1_u8m4, sz_tr29_word_break_aletter_k, vector_length);
    vbool2_t prev1_numeric = __riscv_vmseq_vx_u8m4_b2(prev1_u8m4, sz_tr29_word_break_numeric_k, vector_length);
    vbool2_t prev1_extendnumlet = __riscv_vmseq_vx_u8m4_b2(prev1_u8m4, sz_tr29_word_break_extendnumlet_k,
                                                           vector_length);
    vbool2_t prev1_cr = __riscv_vmseq_vx_u8m4_b2(prev1_u8m4, sz_tr29_word_break_cr_k, vector_length);
    vbool2_t prev1_midletter = __riscv_vmseq_vx_u8m4_b2(prev1_u8m4, sz_tr29_word_break_midletter_k, vector_length);
    vbool2_t prev1_midnum = __riscv_vmseq_vx_u8m4_b2(prev1_u8m4, sz_tr29_word_break_midnum_k, vector_length);
    vbool2_t prev1_quotes = __riscv_vmseq_vx_u8m4_b2(prev1_u8m4, sz_tr29_word_break_mid_quotes_k, vector_length);

    vbool2_t prev2_aletter = __riscv_vmseq_vx_u8m4_b2(prev2_u8m4, sz_tr29_word_break_aletter_k, vector_length);
    vbool2_t prev2_numeric = __riscv_vmseq_vx_u8m4_b2(prev2_u8m4, sz_tr29_word_break_numeric_k, vector_length);

    vbool2_t next1_aletter = __riscv_vmseq_vx_u8m4_b2(next1_u8m4, sz_tr29_word_break_aletter_k, vector_length);
    vbool2_t next1_numeric = __riscv_vmseq_vx_u8m4_b2(next1_u8m4, sz_tr29_word_break_numeric_k, vector_length);

    // Helper combinations (MidLetter|Quotes), (MidNum|Quotes), (ALetter|Numeric).
    vbool2_t prev1_midletter_or_q = __riscv_vmor_mm_b2(prev1_midletter, prev1_quotes, vector_length);
    vbool2_t prev1_midnum_or_q = __riscv_vmor_mm_b2(prev1_midnum, prev1_quotes, vector_length);
    vbool2_t cur_midletter = __riscv_vmseq_vx_u8m4_b2(props_u8m4, sz_tr29_word_break_midletter_k, vector_length);
    vbool2_t cur_midnum = __riscv_vmseq_vx_u8m4_b2(props_u8m4, sz_tr29_word_break_midnum_k, vector_length);
    vbool2_t cur_quotes = __riscv_vmseq_vx_u8m4_b2(props_u8m4, sz_tr29_word_break_mid_quotes_k, vector_length);
    vbool2_t cur_midletter_or_q = __riscv_vmor_mm_b2(cur_midletter, cur_quotes, vector_length);
    vbool2_t cur_midnum_or_q = __riscv_vmor_mm_b2(cur_midnum, cur_quotes, vector_length);
    vbool2_t prev1_aletter_num_enl = __riscv_vmor_mm_b2(__riscv_vmor_mm_b2(prev1_aletter, prev1_numeric, vector_length),
                                                        prev1_extendnumlet, vector_length);
    vbool2_t cur_aletter_or_num = __riscv_vmor_mm_b2(cur_aletter, cur_numeric, vector_length);

    // WB3  CR x LF: prev1==CR && cur==LF.
    vbool2_t join = __riscv_vmand_mm_b2(prev1_cr, cur_lf, vector_length);
    // WB5  prev1==AH && cur==AH.
    join = __riscv_vmor_mm_b2(join, __riscv_vmand_mm_b2(prev1_aletter, cur_aletter, vector_length), vector_length);
    // WB6  prev1==AH && cur∈{ML,Q} && next1==AH.
    join = __riscv_vmor_mm_b2(join,
                              __riscv_vmand_mm_b2(__riscv_vmand_mm_b2(prev1_aletter, cur_midletter_or_q, vector_length),
                                                  next1_aletter, vector_length),
                              vector_length);
    // WB7  prev2==AH && prev1∈{ML,Q} && cur==AH.
    join = __riscv_vmor_mm_b2(
        join,
        __riscv_vmand_mm_b2(__riscv_vmand_mm_b2(prev2_aletter, prev1_midletter_or_q, vector_length), cur_aletter,
                            vector_length),
        vector_length);
    // WB8  prev1==Num && cur==Num.
    join = __riscv_vmor_mm_b2(join, __riscv_vmand_mm_b2(prev1_numeric, cur_numeric, vector_length), vector_length);
    // WB9  prev1==AH && cur==Num.
    join = __riscv_vmor_mm_b2(join, __riscv_vmand_mm_b2(prev1_aletter, cur_numeric, vector_length), vector_length);
    // WB10 prev1==Num && cur==AH.
    join = __riscv_vmor_mm_b2(join, __riscv_vmand_mm_b2(prev1_numeric, cur_aletter, vector_length), vector_length);
    // WB11 prev1==Num && cur∈{MN,Q} && next1==Num.
    join = __riscv_vmor_mm_b2(join,
                              __riscv_vmand_mm_b2(__riscv_vmand_mm_b2(prev1_numeric, cur_midnum_or_q, vector_length),
                                                  next1_numeric, vector_length),
                              vector_length);
    // WB12 prev2==Num && prev1∈{MN,Q} && cur==Num.
    join = __riscv_vmor_mm_b2(join,
                              __riscv_vmand_mm_b2(__riscv_vmand_mm_b2(prev2_numeric, prev1_midnum_or_q, vector_length),
                                                  cur_numeric, vector_length),
                              vector_length);
    // WB13a prev1∈{AH,Num,ENL} && cur==ENL.
    join = __riscv_vmor_mm_b2(join, __riscv_vmand_mm_b2(prev1_aletter_num_enl, cur_extendnumlet, vector_length),
                              vector_length);
    // WB13b prev1==ENL && cur∈{AH,Num}.
    join = __riscv_vmor_mm_b2(join, __riscv_vmand_mm_b2(prev1_extendnumlet, cur_aletter_or_num, vector_length),
                              vector_length);

    // Boundary BEFORE a lane is the complement of the join mask, restricted to trusted lanes [2, vl-2].
    vuint16m8_t lane_iota_u16m8 = __riscv_vid_v_u16m8(vector_length);
    vbool2_t below_high = __riscv_vmsltu_vx_u16m8_b2(lane_iota_u16m8, (sz_u16_t)(vector_length - 1), vector_length);
    vbool2_t below_low = __riscv_vmsltu_vx_u16m8_b2(lane_iota_u16m8, 2, vector_length);
    vbool2_t trusted = __riscv_vmandn_mm_b2(below_high, below_low, vector_length); // (lane < vl-1) && !(lane < 2)
    return __riscv_vmandn_mm_b2(trusted, join, vector_length);                     // trusted & ~join
}

/** @brief Classify an all-ASCII strip's Word_Break properties via one `vluxei8` over the 128-entry table. */
SZ_INTERNAL vuint8m4_t sz_utf8_word_break_classify_ascii_bytes_rvv_(vuint8m4_t bytes_u8m4, sz_size_t vector_length) {
    vuint8m4_t indices_u8m4 = __riscv_vand_vx_u8m4(bytes_u8m4, 0x7F, vector_length);
    return __riscv_vluxei8_v_u8m4(&sz_utf8_word_break_property_ascii_[0], indices_u8m4, vector_length);
}

/*  Emit one all-ASCII window's boundaries as a shifted-difference `(start, length)` stream, parameterized by
 *  direction. Forward: lane 0 of `starts` carries the open `word_start`, `lengths = boundary - start`. Reverse:
 *  the compacted indices are reversed (`vrgatherei16`) so emission walks high-to-low, `starts = boundary`,
 *  `lengths = previous - boundary`. In both directions the carried boundary is the last emitted boundary.
 *  Returns the number of boundaries emitted, writes whether the output buffer filled into `*filled`. */
SZ_INTERNAL sz_size_t sz_utf8_word_break_emit_ascii_rvv_(                                       //
    vuint8m4_t bytes_u8m4, sz_size_t base, sz_size_t vector_length,                             //
    sz_size_t *word_starts, sz_size_t *word_lengths, sz_size_t words, sz_size_t words_capacity, //
    sz_size_t *carry, int reverse, int *filled) {

    vuint8m4_t props_u8m4 = sz_utf8_word_break_classify_ascii_bytes_rvv_(bytes_u8m4, vector_length);
    vbool2_t boundary_mask = sz_utf8_word_break_ascii_boundary_mask_rvv_(props_u8m4, vector_length);

    // Compact boundary lane indices; bias to absolute byte offsets; emit as a shifted-difference.
    vuint16m8_t lane_iota_u16m8 = __riscv_vid_v_u16m8(vector_length);
    vuint16m8_t boundary_indices_u16m8 = __riscv_vcompress_vm_u16m8(lane_iota_u16m8, boundary_mask, vector_length);
    sz_size_t window_boundaries = (sz_size_t)__riscv_vcpop_m_b2(boundary_mask, vector_length);
    sz_size_t emit_count = sz_min_of_two(window_boundaries, words_capacity - words);

    // Reverse the compacted indices over their low `window_boundaries` lanes so emission runs high-to-low.
    if (reverse) {
        vuint16m8_t reverse_index_u16m8 = __riscv_vrsub_vx_u16m8(
            lane_iota_u16m8, (sz_u16_t)(window_boundaries ? window_boundaries - 1 : 0), vector_length);
        boundary_indices_u16m8 = __riscv_vrgatherei16_vv_u16m8(boundary_indices_u16m8, reverse_index_u16m8,
                                                               vector_length);
    }

    sz_size_t emitted = 0;
    sz_u64_t previous_boundary = (sz_u64_t)*carry; // carried into lane 0 of the first chunk
    while (emitted < emit_count) {
        sz_size_t chunk = __riscv_vsetvl_e64m8(emit_count - emitted);
        vuint16m2_t chunk_indices_u16m2 = __riscv_vget_v_u16m8_u16m2(
            __riscv_vslidedown_vx_u16m8(boundary_indices_u16m8, emitted, vector_length), 0);
        vuint64m8_t boundaries_u64m8 = __riscv_vadd_vx_u64m8(
            __riscv_vwcvtu_x_x_v_u64m8(__riscv_vwcvtu_x_x_v_u32m4(chunk_indices_u16m2, chunk), chunk), (sz_u64_t)base,
            chunk);
        vuint64m8_t slid_u64m8 = __riscv_vslide1up_vx_u64m8(boundaries_u64m8, previous_boundary, chunk);
        if (reverse) {
            __riscv_vse64_v_u64m8((sz_u64_t *)word_starts + words + emitted, boundaries_u64m8, chunk);
            __riscv_vse64_v_u64m8((sz_u64_t *)word_lengths + words + emitted,
                                  __riscv_vsub_vv_u64m8(slid_u64m8, boundaries_u64m8, chunk), chunk);
            previous_boundary = word_starts[words + emitted + chunk - 1];
        }
        else {
            __riscv_vse64_v_u64m8((sz_u64_t *)word_starts + words + emitted, slid_u64m8, chunk);
            __riscv_vse64_v_u64m8((sz_u64_t *)word_lengths + words + emitted,
                                  __riscv_vsub_vv_u64m8(boundaries_u64m8, slid_u64m8, chunk), chunk);
            previous_boundary = word_starts[words + emitted + chunk - 1] + word_lengths[words + emitted + chunk - 1];
        }
        emitted += chunk;
    }
    if (emit_count)
        *carry = reverse ? word_starts[words + emit_count - 1]
                         : word_starts[words + emit_count - 1] + word_lengths[words + emit_count - 1];
    *filled = emit_count < window_boundaries; // output buffer full: caller resumes past the last emitted boundary
    return emit_count;
}

/*  Vectorized UAX-29 word-boundary scan, byte-exact to the serial reference. An outer flat window loop loads a
 *  strip anchored two bytes before the open boundary (`base = position-2`); all-ASCII strips emit in-register
 *  via `sz_utf8_word_break_emit_ascii_rvv_`, any other window takes one scalar codepoint step. The capacity cut
 *  sets `*bytes_consumed = word_start` (a true boundary) so a fresh next call restarts cleanly. */
SZ_INTERNAL sz_size_t sz_utf8_word_find_boundaries_rvv_(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                        sz_size_t *word_lengths, sz_size_t words_capacity,
                                                        sz_size_t *bytes_consumed) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t words = 0;
    sz_size_t word_start = 0; // Start of the in-progress word (always a true boundary).
    sz_size_t position = sz_utf8_codepoint_length_(text_u8[0]); // position 0 is a boundary; start past it

    while (position < length && words < words_capacity) {
        sz_size_t base = position - 2; // lane j = byte base+j; trusted lanes [2, vl-2] → boundaries [position, ...]
        sz_size_t vector_length = (position >= 2) ? __riscv_vsetvl_e8m4(length - base) : 0;
        int ascii_window = position >= 2 && vector_length >= 5;
        vuint8m4_t bytes_u8m4;
        if (ascii_window) {
            bytes_u8m4 = __riscv_vle8_v_u8m4(text_u8 + base, vector_length);
            ascii_window = __riscv_vfirst_m_b2(__riscv_vmsgtu_vx_u8m4_b2(bytes_u8m4, 0x7F, vector_length),
                                               vector_length) < 0;
        }
        if (ascii_window) {
            int filled;
            words += sz_utf8_word_break_emit_ascii_rvv_(bytes_u8m4, base, vector_length, word_starts, word_lengths,
                                                        words, words_capacity, &word_start, 0, &filled);
            if (filled) {
                if (bytes_consumed) *bytes_consumed = word_start;
                return words;
            }
            position += vector_length - 3; // Resolved [position, position+vl-4]; next boundary at position+vl-3.
            continue;
        }

        // Non-ASCII window or near the edges: one scalar codepoint step.
        if (sz_utf8_word_break_decision_at_rvv_(text, length, position)) {
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_start;
                return words;
            }
            word_starts[words] = word_start, word_lengths[words] = position - word_start, ++words;
            word_start = position;
        }
        position += sz_utf8_codepoint_length_(text_u8[position]);
    }

    // The trailing span [word_start, length) is the last word (end of text is always a boundary).
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

/*  Reverse counterpart of `sz_utf8_word_find_boundaries_rvv_`. The strip is anchored so its top lane sits one
 *  byte past the open boundary (`base = position+2-vl`), giving trusted lanes [2, vl-2] → boundaries
 *  [position-vl+4, position]; emission runs high-to-low via the reversed-index path of the shared emitter. */
SZ_INTERNAL sz_size_t sz_utf8_word_rfind_boundaries_rvv_(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                         sz_size_t *word_lengths, sz_size_t words_capacity,
                                                         sz_size_t *bytes_consumed) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t words = 0;
    sz_size_t word_end = length;     // End of the in-progress word (always a true boundary).
    sz_size_t position = length - 1; // position `length` is a boundary; step back to the first reportable position
    while (position > 0 && (text_u8[position] & 0xC0) == 0x80) position--;

    while (position > 0 && words < words_capacity) {
        sz_size_t vector_length = (position + 2 <= length) ? __riscv_vsetvl_e8m4(position + 2) : 0;
        sz_size_t base = position + 2 - vector_length; // lane j = byte base+j
        int ascii_window = position + 2 <= length && vector_length >= 5;
        vuint8m4_t bytes_u8m4;
        if (ascii_window) {
            bytes_u8m4 = __riscv_vle8_v_u8m4(text_u8 + base, vector_length);
            ascii_window = __riscv_vfirst_m_b2(__riscv_vmsgtu_vx_u8m4_b2(bytes_u8m4, 0x7F, vector_length),
                                               vector_length) < 0;
        }
        if (ascii_window) {
            int filled;
            words += sz_utf8_word_break_emit_ascii_rvv_(bytes_u8m4, base, vector_length, word_starts, word_lengths,
                                                        words, words_capacity, &word_end, 1, &filled);
            if (filled) {
                if (bytes_consumed) *bytes_consumed = word_end;
                return words;
            }
            position = base + 1; // Resolved down to position-vl+4; next unresolved boundary at position-vl+3.
            continue;
        }

        // Non-ASCII window or near the edges: one scalar codepoint step.
        if (sz_utf8_word_break_decision_at_rvv_(text, length, position)) {
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_end;
                return words;
            }
            word_starts[words] = position, word_lengths[words] = word_end - position, ++words;
            word_end = position;
        }
        position--;
        while (position > 0 && (text_u8[position] & 0xC0) == 0x80) position--;
    }

    // The leading span [0, word_end) is the last word (start of text is always a boundary).
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

/** @brief Common driver: route malformed UTF-8 to serial, otherwise run the vectorized window scan. */
SZ_INTERNAL sz_size_t sz_utf8_word_break_scan_rvv_(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                   sz_size_t *word_lengths, sz_size_t words_capacity,
                                                   sz_size_t *bytes_consumed, int reverse) {
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = (length == 0) ? 0 : (reverse ? length : 0);
        return 0;
    }
    if (!sz_utf8_word_break_is_well_formed_rvv_(text, length))
        return reverse ? sz_utf8_word_rfind_boundaries_serial(text, length, word_starts, word_lengths, words_capacity,
                                                              bytes_consumed)
                       : sz_utf8_word_find_boundaries_serial(text, length, word_starts, word_lengths, words_capacity,
                                                             bytes_consumed);
    return reverse ? sz_utf8_word_rfind_boundaries_rvv_(text, length, word_starts, word_lengths, words_capacity,
                                                        bytes_consumed)
                   : sz_utf8_word_find_boundaries_rvv_(text, length, word_starts, word_lengths, words_capacity,
                                                       bytes_consumed);
}

SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_rvv(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                     sz_size_t *word_lengths, sz_size_t words_capacity,
                                                     sz_size_t *bytes_consumed) {
    return sz_utf8_word_break_scan_rvv_(text, length, word_starts, word_lengths, words_capacity, bytes_consumed, 0);
}

SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_rvv(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                      sz_size_t *word_lengths, sz_size_t words_capacity,
                                                      sz_size_t *bytes_consumed) {
    return sz_utf8_word_break_scan_rvv_(text, length, word_starts, word_lengths, words_capacity, bytes_consumed, 1);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_RVV

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_ITERATE_RVV_H_
