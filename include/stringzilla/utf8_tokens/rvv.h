/**
 *  @brief RISC-V Vector backend for UTF-8 newline and whitespace delimiter scanning.
 *  @file include/stringzilla/utf8_tokens/rvv.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_TOKENS_RVV_H_
#define STRINGZILLA_UTF8_TOKENS_RVV_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h" // `sz_rune_decode_unchecked`
#include "stringzilla/utf8_tokens/serial.h"
#include "stringzilla/utf8_runes/rvv.h"

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

SZ_PUBLIC sz_size_t sz_utf8_newlines_rvv(               //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {
    return sz_utf8_iterate_multistep_rvv_(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed,
                                          1);
}

SZ_PUBLIC sz_size_t sz_utf8_whitespaces_rvv(            //
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

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_RVV

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_TOKENS_RVV_H_
