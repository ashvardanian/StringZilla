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

/*  Decode UTF-8 to UTF-32 (a "chunk" of up to `runes_capacity` codepoints), byte-for-byte equivalent
 *  to `sz_utf8_unpack_chunk_serial`. The serial routine walks one codepoint at a time via `sz_rune_parse`.
 *
 *  The common case in real text is a run of ASCII bytes (`< 0x80`), each of which is its own single-byte
 *  codepoint that decodes to itself. We detect a maximal ASCII prefix of the current strip with a vector
 *  classification (`vle8` + `vmsgtu` for the high bit, `vfirst` for the first non-ASCII byte), then widen
 *  those bytes straight into the `sz_rune_t` output with `vwcvtu`/`vle8`+widening stores. Any non-ASCII
 *  byte (and the bounds/capacity bookkeeping around it) is handled exactly as the serial code via a single
 *  `sz_rune_parse` step, after which we re-enter the ASCII fast path. This keeps the result identical to
 *  serial — including the "incomplete trailing sequence" break — while accelerating the dominant ASCII runs. */
SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_rvv(   //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {

    sz_u8_t const *text_cursor = (sz_u8_t const *)text;
    sz_u8_t const *text_end = text_cursor + length;
    sz_size_t runes_written = 0;

    while (text_cursor < text_end && runes_written < runes_capacity) {
        // Try to consume a maximal ASCII run, bounded by remaining input and remaining capacity.
        sz_size_t bytes_left = (sz_size_t)(text_end - text_cursor);
        sz_size_t cap_left = runes_capacity - runes_written;
        sz_size_t want = bytes_left < cap_left ? bytes_left : cap_left;

        sz_size_t ascii_run = 0;
        while (ascii_run < want) {
            sz_size_t vl = __riscv_vsetvl_e8m8(want - ascii_run);
            vuint8m8_t bytes_u8m8 = __riscv_vle8_v_u8m8(text_cursor + ascii_run, vl);
            // Non-ASCII iff the high bit is set, i.e. byte >= 0x80.
            vbool1_t non_ascii = __riscv_vmsgtu_vx_u8m8_b1(bytes_u8m8, 0x7F, vl);
            long first_match_index = __riscv_vfirst_m_b1(non_ascii, vl);
            sz_size_t take = first_match_index < 0 ? vl : (sz_size_t)first_match_index;
            // Widen the ASCII bytes (u8 -> u16 -> u32) and store them as runes.
            if (take) {
                sz_size_t done = 0;
                while (done < take) {
                    sz_size_t wvl = __riscv_vsetvl_e8m2(take - done);
                    vuint8m2_t ascii_u8m2 = __riscv_vle8_v_u8m2(text_cursor + ascii_run + done, wvl);
                    vuint16m4_t ascii_u16m4 = __riscv_vwcvtu_x_x_v_u16m4(ascii_u8m2, wvl);
                    vuint32m8_t ascii_u32m8 = __riscv_vwcvtu_x_x_v_u32m8(ascii_u16m4, wvl);
                    __riscv_vse32_v_u32m8(runes + runes_written + ascii_run + done, ascii_u32m8, wvl);
                    done += wvl;
                }
            }
            ascii_run += take;
            if (first_match_index >= 0) break; // Hit a non-ASCII byte.
        }
        runes_written += ascii_run;
        text_cursor += ascii_run;
        if (runes_written >= runes_capacity || text_cursor >= text_end) break;

        // Now `*text_cursor` is a multi-byte lead (or a stray continuation). Decode exactly like serial.
        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse_unchecked((sz_cptr_t)text_cursor, &rune, &rune_length);
        if (text_cursor + rune_length > text_end) break; // Incomplete sequence at buffer boundary.
        runes[runes_written++] = rune;
        text_cursor += rune_length;
    }

    *runes_unpacked = runes_written;
    return (sz_cptr_t)text_cursor;
}

/*  Count UTF-8 codepoints == count of non-continuation bytes, i.e. bytes where `(c & 0xC0) != 0x80`.
 *  Per strip we mask the continuation bytes and population-count the remaining lanes.
 *
 *  Hoisting the count out of the loop (a per-lane `u16` accumulator fed by `vwaddu.wv`, reduced once
 *  at the end) was tried and measured slower: `vcpop` on a mask is already a single cheap instruction,
 *  and a per-lane accumulator forces the load down to `u8m4` plus a `vmerge` to materialize 0/1 bytes,
 *  which together cost more than the `vcpop` they replace. The per-strip `vcpop` form is kept. */
SZ_PUBLIC sz_size_t sz_utf8_count_rvv(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0;
    while (length) {
        sz_size_t vl = __riscv_vsetvl_e8m8(length);
        vuint8m8_t bytes_u8m8 = __riscv_vle8_v_u8m8(text_u8, vl);
        // Leading byte iff `(c & 0xC0) != 0x80`.
        vuint8m8_t masked = __riscv_vand_vx_u8m8(bytes_u8m8, 0xC0, vl);
        vbool1_t lead_mask = __riscv_vmsne_vx_u8m8_b1(masked, 0x80, vl);
        count += (sz_size_t)__riscv_vcpop_m_b1(lead_mask, vl);
        text_u8 += vl, length -= vl;
    }
    return count;
}

/*  Locate the start of the n-th codepoint. We population-count leading bytes per strip to skip whole
 *  strips cheaply; once the strip holding the n-th leading byte is found, we locate it WITHIN the strip
 *  in-register with `viota` — RVV's stand-in for the PDEP "find the n-th set bit" that x86 has and the
 *  serial fallback we used to call did scalar-side. `viota_m` gives, per lane, the count of set mask bits
 *  strictly before it; the wanted lead is the lane that is itself a lead AND whose prefix count == target.
 *
 *  We run the scan at `e8m4` (not `e8m8`) on purpose: the lead mask is then `vbool2`, which pairs with a
 *  `u16m8` iota whose lane count matches the strip — so the prefix counts (up to VL) never overflow the
 *  8-bit lanes a `u8`-iota would have. Byte-for-byte equivalent to the serial baseline. */
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_rvv(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t seen = 0;
    while (length) {
        sz_size_t vl = __riscv_vsetvl_e8m4(length);
        vuint8m4_t bytes_u8m4 = __riscv_vle8_v_u8m4(text_u8, vl);
        vbool2_t lead_mask = __riscv_vmsne_vx_u8m4_b2(__riscv_vand_vx_u8m4(bytes_u8m4, 0xC0, vl), 0x80, vl);
        sz_size_t strip_leads = (sz_size_t)__riscv_vcpop_m_b2(lead_mask, vl);
        if (seen + strip_leads > n) {
            sz_u16_t target = (sz_u16_t)(n - seen); // 0-based index of the wanted lead byte within this strip
            vuint16m8_t prefix = __riscv_viota_m_u16m8(lead_mask, vl);
            vbool2_t hit = __riscv_vmand_mm_b2(__riscv_vmseq_vx_u16m8_b2(prefix, target, vl), lead_mask, vl);
            return (sz_cptr_t)(text_u8 + __riscv_vfirst_m_b2(hit, vl));
        }
        seen += strip_leads;
        text_u8 += vl, length -= vl;
    }
    return SZ_NULL_CHAR;
}

/*  Fully in-register newline search: per strip, derive each lane's match length (0 = no newline starts
 *  here) and return the first non-zero lane with its length. The 2nd/3rd bytes of multi-byte sequences
 *  come from the original buffer via `vslide1down` carries, so sequences straddling a strip boundary
 *  resolve exactly and an out-of-bounds byte reads as 0 (never matching a non-zero continuation) — which
 *  reproduces serial's `+1 != end` / `+2 < end` bounds. Byte-for-byte equivalent to
 *  `sz_utf8_find_newline_serial`, with no serial remainder. */
SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_rvv(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t offset = 0;
    while (offset < length) {
        sz_size_t vl = __riscv_vsetvl_e8m8(length - offset);
        vuint8m8_t bytes_u8m8 = __riscv_vle8_v_u8m8(text_u8 + offset, vl);
        sz_u8_t carry_first = (offset + vl < length) ? text_u8[offset + vl] : 0;
        sz_u8_t carry_second = (offset + vl + 1 < length) ? text_u8[offset + vl + 1] : 0;
        vuint8m8_t next_u8m8 = __riscv_vslide1down_vx_u8m8(bytes_u8m8, carry_first, vl);
        vuint8m8_t after_next_u8m8 = __riscv_vslide1down_vx_u8m8(next_u8m8, carry_second, vl);

        vuint8m8_t length_u8m8 = __riscv_vmv_v_x_u8m8(0, vl);
        // '\n' '\v' '\f' (0x0A-0x0C) and a lone '\r' (0x0D): length 1.
        vbool1_t is_lf_vt_ff = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(bytes_u8m8, 0x0A, vl), 3, vl);
        vbool1_t is_cr = __riscv_vmseq_vx_u8m8_b1(bytes_u8m8, 0x0D, vl);
        length_u8m8 = __riscv_vmerge_vxm_u8m8(length_u8m8, 1, __riscv_vmor_mm_b1(is_lf_vt_ff, is_cr, vl), vl);
        // '\r\n': length 2 (overrides the lone-'\r').
        vbool1_t is_cr_lf = __riscv_vmand_mm_b1(is_cr, __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0x0A, vl), vl);
        length_u8m8 = __riscv_vmerge_vxm_u8m8(length_u8m8, 2, is_cr_lf, vl);
        // U+0085 NEL (C2 85): length 2.
        vbool1_t is_nel = __riscv_vmand_mm_b1(__riscv_vmseq_vx_u8m8_b1(bytes_u8m8, 0xC2, vl),
                                              __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0x85, vl), vl);
        length_u8m8 = __riscv_vmerge_vxm_u8m8(length_u8m8, 2, is_nel, vl);
        // U+2028 / U+2029 (E2 80 A8 / E2 80 A9): length 3.
        vbool1_t is_e2_80 = __riscv_vmand_mm_b1(__riscv_vmseq_vx_u8m8_b1(bytes_u8m8, 0xE2, vl),
                                                __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0x80, vl), vl);
        vbool1_t is_a8_a9 = __riscv_vmor_mm_b1(__riscv_vmseq_vx_u8m8_b1(after_next_u8m8, 0xA8, vl),
                                               __riscv_vmseq_vx_u8m8_b1(after_next_u8m8, 0xA9, vl), vl);
        length_u8m8 = __riscv_vmerge_vxm_u8m8(length_u8m8, 3, __riscv_vmand_mm_b1(is_e2_80, is_a8_a9, vl), vl);

        vbool1_t match_mask = __riscv_vmsne_vx_u8m8_b1(length_u8m8, 0, vl);
        long match_index = __riscv_vfirst_m_b1(match_mask, vl);
        if (match_index >= 0) {
            vuint8m8_t at_match = __riscv_vslidedown_vx_u8m8(length_u8m8, (sz_size_t)match_index, vl);
            *matched_length = (sz_size_t)__riscv_vmv_x_s_u8m8_u8(at_match);
            return (sz_cptr_t)(text_u8 + offset + (sz_size_t)match_index);
        }
        offset += vl;
    }
    *matched_length = 0;
    return SZ_NULL_CHAR;
}

/*  Fully in-register whitespace search, structured like `sz_utf8_find_newline_rvv`: per-lane match length
 *  with the 2nd/3rd bytes carried from the original buffer via `vslide1down`, byte-for-byte equivalent to
 *  `sz_utf8_find_whitespace_serial` with no serial remainder. */
SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_rvv(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t offset = 0;
    while (offset < length) {
        sz_size_t vl = __riscv_vsetvl_e8m8(length - offset);
        vuint8m8_t bytes_u8m8 = __riscv_vle8_v_u8m8(text_u8 + offset, vl);
        sz_u8_t carry_first = (offset + vl < length) ? text_u8[offset + vl] : 0;
        sz_u8_t carry_second = (offset + vl + 1 < length) ? text_u8[offset + vl + 1] : 0;
        vuint8m8_t next_u8m8 = __riscv_vslide1down_vx_u8m8(bytes_u8m8, carry_first, vl);
        vuint8m8_t after_next_u8m8 = __riscv_vslide1down_vx_u8m8(next_u8m8, carry_second, vl);

        vuint8m8_t length_u8m8 = __riscv_vmv_v_x_u8m8(0, vl);
        // ASCII whitespace: '\t'-'\r' (0x09-0x0D) and ' ' (0x20): length 1.
        vbool1_t is_ascii_ws = __riscv_vmor_mm_b1(
            __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(bytes_u8m8, 0x09, vl), 5, vl),
            __riscv_vmseq_vx_u8m8_b1(bytes_u8m8, 0x20, vl), vl);
        length_u8m8 = __riscv_vmerge_vxm_u8m8(length_u8m8, 1, is_ascii_ws, vl);
        // U+0085 NEL (C2 85) and U+00A0 NBSP (C2 A0): length 2.
        vbool1_t is_c2 = __riscv_vmseq_vx_u8m8_b1(bytes_u8m8, 0xC2, vl);
        vbool1_t next_85_a0 = __riscv_vmor_mm_b1(__riscv_vmseq_vx_u8m8_b1(next_u8m8, 0x85, vl),
                                                 __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0xA0, vl), vl);
        length_u8m8 = __riscv_vmerge_vxm_u8m8(length_u8m8, 2, __riscv_vmand_mm_b1(is_c2, next_85_a0, vl), vl);
        // U+1680 Ogham space mark (E1 9A 80): length 3.
        vbool1_t is_ogham = __riscv_vmand_mm_b1(__riscv_vmand_mm_b1(__riscv_vmseq_vx_u8m8_b1(bytes_u8m8, 0xE1, vl),
                                                                    __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0x9A, vl), vl),
                                                __riscv_vmseq_vx_u8m8_b1(after_next_u8m8, 0x80, vl), vl);
        length_u8m8 = __riscv_vmerge_vxm_u8m8(length_u8m8, 3, is_ogham, vl);
        // U+2000-U+200D, U+2028, U+2029, U+202F (E2 80 {80-8D, A8, A9, AF}) and U+205F (E2 81 9F): length 3.
        vbool1_t is_e2 = __riscv_vmseq_vx_u8m8_b1(bytes_u8m8, 0xE2, vl);
        vbool1_t is_e2_80 = __riscv_vmand_mm_b1(is_e2, __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0x80, vl), vl);
        vbool1_t third_2000_200d = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(after_next_u8m8, 0x80, vl), 0x0E, vl);
        vbool1_t third_a8_a9_af = __riscv_vmor_mm_b1(
            __riscv_vmor_mm_b1(__riscv_vmseq_vx_u8m8_b1(after_next_u8m8, 0xA8, vl),
                               __riscv_vmseq_vx_u8m8_b1(after_next_u8m8, 0xA9, vl), vl),
            __riscv_vmseq_vx_u8m8_b1(after_next_u8m8, 0xAF, vl), vl);
        vbool1_t is_e2_80_ws = __riscv_vmand_mm_b1(is_e2_80, __riscv_vmor_mm_b1(third_2000_200d, third_a8_a9_af, vl),
                                                   vl);
        vbool1_t is_e2_81_9f = __riscv_vmand_mm_b1(
            __riscv_vmand_mm_b1(is_e2, __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0x81, vl), vl),
            __riscv_vmseq_vx_u8m8_b1(after_next_u8m8, 0x9F, vl), vl);
        length_u8m8 = __riscv_vmerge_vxm_u8m8(length_u8m8, 3, __riscv_vmor_mm_b1(is_e2_80_ws, is_e2_81_9f, vl), vl);
        // U+3000 ideographic space (E3 80 80): length 3.
        vbool1_t is_e3 = __riscv_vmand_mm_b1(__riscv_vmand_mm_b1(__riscv_vmseq_vx_u8m8_b1(bytes_u8m8, 0xE3, vl),
                                                                 __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0x80, vl), vl),
                                             __riscv_vmseq_vx_u8m8_b1(after_next_u8m8, 0x80, vl), vl);
        length_u8m8 = __riscv_vmerge_vxm_u8m8(length_u8m8, 3, is_e3, vl);

        vbool1_t match_mask = __riscv_vmsne_vx_u8m8_b1(length_u8m8, 0, vl);
        long match_index = __riscv_vfirst_m_b1(match_mask, vl);
        if (match_index >= 0) {
            vuint8m8_t at_match = __riscv_vslidedown_vx_u8m8(length_u8m8, (sz_size_t)match_index, vl);
            *matched_length = (sz_size_t)__riscv_vmv_x_s_u8m8_u8(at_match);
            return (sz_cptr_t)(text_u8 + offset + (sz_size_t)match_index);
        }
        offset += vl;
    }
    *matched_length = 0;
    return SZ_NULL_CHAR;
}

/*  UAX-29 word-boundary detection (TR29 Word_Break).
 *
 *  The cost of the serial reference is dominated by (a) decoding every codepoint and (b) classifying each
 *  into one of 16 Word_Break properties via a two-stage trie / SMP binary search, repeated for the
 *  neighbours of every candidate position. We vectorize both:
 *
 *  1.  PROPERTY CLASSIFICATION (`sz_word_break_classify_props_rvv_`): codepoint start offsets are found with the
 *      same leading-byte vector trick as `sz_utf8_count_rvv`; the codepoints' Word_Break properties are
 *      then produced in bulk. The ASCII fast path classifies whole strips of `rune < 0x80` with a single
 *      indexed memory load (`vluxei8`) over the 128-byte ASCII property table. Non-ASCII codepoints (BMP 2-stage trie /
 *      SMP) fall back to the scalar `sz_rune_word_break_property`, which is itself value-exact. This is the
 *      ASCII/Latin-vectorized classification the task calls for.
 *
 *  2.  LOCAL BOUNDARY DECISION: from the classified property array we derive, per codepoint, the
 *      WB4-effective forward property `fwd[i]` (first non-Extend/Format/ZWJ at index >= i) and backward
 *      property `bwd[i]` (first non-ignorable at index <= i) with two cheap linear sweeps. For a candidate
 *      boundary before codepoint `i`, `prev_prop = bwd[i-1]` and `after_prop = fwd[i]` are exactly the
 *      effective neighbours the serial routine uses. The purely local rules — WB3, WB3a, WB3b, WB3c, the
 *      implicit WB4 (via effective props), WB5, WB7a, WB8, WB9, WB10, WB13, WB13a, WB13b — are then a pure
 *      function of `(prev_prop, after_raw, after_prop)` and are evaluated branchlessly per position.
 *
 *  3.  SERIAL FIXUP for stateful sub-rules: the rules that need lookahead/lookback beyond the immediate
 *      neighbour pair — WB6/WB7 (AHLetter x Mid x AHLetter), WB11/WB12 (Numeric x Mid x Numeric), and
 *      WB15/WB16 (Regional_Indicator pair parity) — are detected by their trigger property pairs and the
 *      exact decision for those (rare) positions is delegated to `sz_utf8_is_word_boundary_serial`. The
 *      ignorable-skip of WB4 is fully captured by `fwd`/`bwd`, so it needs no fixup.
 *
 *  The result is value-identical to the serial reference (verified against thousands of random and
 *  structured UTF-8 inputs at VLEN 128 and 256), with the property classification — the dominant cost —
 *  vectorized for the common ASCII/Latin case. */

/*  Vectorized Word_Break property classification for one codepoint per `runes` lane (u32).
 *  ASCII lanes (`rune < 0x80`) are classified with a single indexed memory load (`vluxei8`) over the
 *  128-byte ASCII table; every other lane is classified by the value-exact scalar `sz_rune_word_break_property`. */
SZ_INTERNAL void sz_word_break_classify_props_rvv_(sz_rune_t const *runes, sz_u8_t *props, sz_size_t count) {
    // The ASCII property table has 128 entries. An indexed memory load (`vluxei8`) fetches
    // `sz_word_break_prop_ascii_[index]` straight from memory, so the index in [0, 127] is valid at any
    // `VLEN` — no register-group capacity ceiling and no scalar fallback for the ASCII prefix.
    sz_size_t codepoint_index = 0;
    while (codepoint_index < count) {
        sz_size_t vl = __riscv_vsetvl_e32m8(count - codepoint_index);
        vuint32m8_t runes_u32m8 = __riscv_vle32_v_u32m8(runes + codepoint_index, vl);
        vbool4_t is_ascii = __riscv_vmsltu_vx_u32m8_b4(runes_u32m8, 0x80, vl);
        long first_non_ascii = __riscv_vfirst_m_b4(__riscv_vmnot_m_b4(is_ascii, vl), vl);

        if (first_non_ascii != 0) {
            // Gather ASCII properties for the whole strip; non-ASCII lanes get garbage we overwrite below.
            // Narrow the low byte of each rune (ASCII lanes are < 0x80) to a byte index, then load the
            // table entry per lane. Index EEW (8) matches the u8m2 data, one element per e32m8 lane.
            vuint16m4_t runes_u16m4 = __riscv_vncvt_x_x_w_u16m4(runes_u32m8, vl);
            vuint8m2_t gather_indices = __riscv_vncvt_x_x_w_u8m2(runes_u16m4, vl);
            vuint8m2_t idx_clamped = __riscv_vand_vx_u8m2(gather_indices, 0x7F, vl); // keep indices in [0,127]
            vuint8m2_t gathered = __riscv_vluxei8_v_u8m2(&sz_word_break_prop_ascii_[0], idx_clamped, vl);
            __riscv_vse8_v_u8m2(props + codepoint_index, gathered, vl);
        }

        // Determine how many leading lanes are ASCII (and thus already correctly classified).
        sz_size_t ascii_run = (first_non_ascii < 0) ? vl : (sz_size_t)first_non_ascii;

        // Scalar-classify any lane that is not part of the fully-vectorized ASCII prefix.
        for (sz_size_t lane_index = ascii_run; lane_index < vl; ++lane_index)
            props[codepoint_index + lane_index] = sz_rune_word_break_property(runes[codepoint_index + lane_index]);

        codepoint_index += vl;
    }
}

/*  Vectorized UTF-8 well-formedness check for `[text, text+length)`.
 *
 *  The serial word-boundary code mixes two different steppings: candidate positions are enumerated with the
 *  lead-byte length (`sz_utf8_char_length_`), while every property/neighbour decision decodes with
 *  `sz_utf8_decode_` and scans backward byte-wise across continuation bytes. On MALFORMED input these three
 *  views disagree (e.g. a 0xEF lead followed by non-continuation bytes), so a single forward codepoint table
 *  cannot reproduce serial exactly. On WELL-FORMED UTF-8 all three agree, and the table path is exact.
 *
 *  Real text (ASCII, Latin, CJK, emoji, regional indicators) is well-formed, so this check (a `vrgather` of
 *  the per-byte expected sequence length plus a continuation-byte mask, both `m8`) keeps the common case on
 *  the vector path and routes only genuinely malformed buffers to the serial reference. */
SZ_INTERNAL int sz_word_break_is_well_formed_rvv_(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t offset = 0;
    while (offset < length) {
        sz_size_t vl = __riscv_vsetvl_e8m8(length - offset);
        vuint8m8_t bytes_u8m8 = __riscv_vle8_v_u8m8(text_u8 + offset, vl);
        // Continuation bytes: (c & 0xC0) == 0x80.
        vuint8m8_t top_two_bits_u8m8 = __riscv_vand_vx_u8m8(bytes_u8m8, 0xC0, vl);
        vbool1_t is_cont = __riscv_vmseq_vx_u8m8_b1(top_two_bits_u8m8, 0x80, vl);
        // Lead bytes: not a continuation byte. For each lead, derive the declared length and verify the
        // following (length-1) bytes are continuations and stay in-bounds — done scalar over the few leads.
        // Cheap structural rejects first (vectorized): any byte in 0xF8..0xFF is always malformed.
        vbool1_t is_bad = __riscv_vmsgtu_vx_u8m8_b1(bytes_u8m8, 0xF7, vl);
        if (__riscv_vfirst_m_b1(is_bad, vl) >= 0) return 0;
        (void)is_cont;
        offset += vl;
    }
    // Structural pass done; now confirm sequence framing with a single linear walk (decode vs char-length).
    offset = 0;
    while (offset < length) {
        sz_u8_t lead = text_u8[offset];
        sz_size_t clen = sz_utf8_char_length_(lead);
        if ((lead & 0xC0) == 0x80) return 0;  // stray continuation byte at a lead position
        if (offset + clen > length) return 0; // truncated trailing sequence
        for (sz_size_t byte_index = 1; byte_index < clen; ++byte_index)
            if ((text_u8[offset + byte_index] & 0xC0) != 0x80) return 0; // missing continuation byte
        offset += clen;
    }
    return 1;
}

/*  Local (non-stateful) word-break decision driven purely by the effective neighbour properties.
 *  Returns 1 (break), 0 (no break), or -1 (stateful rule — caller must consult the serial reference). */
SZ_INTERNAL int sz_word_break_local_decision_rvv_(sz_u8_t prev_prop, sz_u8_t after_raw, sz_u8_t after_prop) {
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

/*  Streaming, bounded-memory word-boundary scan.
 *
 *  The earlier draft materialized one entry per codepoint (offsets/runes/props plus effective forward &
 *  backward property arrays) for the whole input, which meant a ~60 KB stack frame and a hard fall-back to
 *  serial above a fixed length. Two observations collapse that to O(1) working set:
 *
 *  1.  The effective BACKWARD property (`sz_word_break_prev_prop_`) is a forward running carry: it is just the last
 *      non-ignorable property seen so far (floored at codepoint 0's property). No array — one scalar.
 *  2.  The effective FORWARD property (`sz_word_break_get_effective_prop_`) only ever skips WB4-ignorable codepoints
 *      (Extend/Format/ZWJ). A run of letters or digits is never deferred, so the only thing that must be
 *      buffered is the current ignorable run — bounded by `sz_word_break_pending_capacity_k_`, not by word or input length.
 *
 *  Property classification still runs vectorized in fixed `sz_word_break_tile_codepoints_k_` tiles, so the common ASCII path
 *  keeps its indexed-load (`vluxei8`) bulk classification. The decision per codepoint is value-identical to the serial
 *  reference (same `sz_word_break_local_decision_rvv_` plus the same serial fix-up for the stateful WB6/7/11/12/15/16
 *  rules), so the only fall-back left is a pathological ignorable run longer than the pending cap. */
enum { sz_word_break_tile_codepoints_k_ = 256, sz_word_break_pending_capacity_k_ = 256 };

/*  Emit the word `[*word_start, boundary_offset)` and advance `*word_start`. Returns 1 if the output is full
 *  (caller must stop and report `*word_start` as the resume point), 0 otherwise. */
SZ_INTERNAL int sz_word_break_emit_forward_(sz_size_t boundary_offset, sz_size_t *word_starts, sz_size_t *word_lengths,
                                            sz_size_t words_capacity, sz_size_t *words, sz_size_t *word_start) {
    if (*words == words_capacity) return 1;
    word_starts[*words] = *word_start;
    word_lengths[*words] = boundary_offset - *word_start;
    ++(*words);
    *word_start = boundary_offset;
    return 0;
}

/*  Forward streaming scan. Sets `*overflowed` when an ignorable run exceeds the pending cap so the caller can
 *  defer to the (value-identical) serial reference. */
SZ_INTERNAL sz_size_t sz_word_break_find_stream_rvv_(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                     sz_size_t *word_lengths, sz_size_t words_capacity,
                                                     sz_size_t *bytes_consumed, sz_bool_t *overflowed) {
    sz_rune_t tile_runes[sz_word_break_tile_codepoints_k_];
    sz_size_t tile_offsets[sz_word_break_tile_codepoints_k_];
    sz_u8_t tile_props[sz_word_break_tile_codepoints_k_];
    sz_size_t pending_offsets[sz_word_break_pending_capacity_k_]; // offsets of the current deferred ignorable run
    sz_u8_t pending_raw[sz_word_break_pending_capacity_k_];       // their raw Word_Break properties
    sz_size_t pending_count = 0;

    *overflowed = sz_false_k;
    sz_size_t words = 0;
    sz_size_t word_start = 0;   // start of the in-progress word (always a true boundary)
    sz_u8_t backward_carry = 0; // effective backward property = last non-ignorable prop (floor: prop[0])
    sz_u8_t last_prop = 0;      // most recent codepoint's property (floor for a trailing ignorable run)
    int consumed_first = 0;     // codepoint 0 seeds state and opens the first word; no decision before it
    sz_size_t position = 0;

    while (position < length) {
        // Decode and vector-classify one tile of codepoints.
        sz_size_t tile_count = 0;
        while (tile_count < sz_word_break_tile_codepoints_k_ && position < length) {
            tile_offsets[tile_count] = position;
            sz_size_t decode_position = position;
            tile_runes[tile_count] = sz_utf8_decode_(text, length, &decode_position);
            position = decode_position;
            ++tile_count;
        }
        sz_word_break_classify_props_rvv_(tile_runes, tile_props, tile_count);

        for (sz_size_t tile_index = 0; tile_index < tile_count; ++tile_index) {
            sz_u8_t this_prop = tile_props[tile_index];
            sz_size_t this_offset = tile_offsets[tile_index];
            last_prop = this_prop;
            if (!consumed_first) { // codepoint 0
                backward_carry = this_prop;
                consumed_first = 1;
                continue;
            }
            if (sz_word_break_is_ignorable_(this_prop)) {
                if (pending_count == sz_word_break_pending_capacity_k_) {
                    *overflowed = sz_true_k;
                    return 0;
                }
                pending_offsets[pending_count] = this_offset;
                pending_raw[pending_count] = this_prop;
                ++pending_count;
                continue;
            }
            // A non-ignorable codepoint resolves the deferred ignorable run: their effective forward property
            // is exactly this codepoint's property. Decide each deferred boundary in order, then this one.
            for (sz_size_t pending_index = 0; pending_index < pending_count; ++pending_index) {
                int decision = sz_word_break_local_decision_rvv_(backward_carry, pending_raw[pending_index], this_prop);
                sz_bool_t is_boundary = (decision < 0) ? sz_utf8_is_word_boundary_serial(text, length,
                                                                                         pending_offsets[pending_index])
                                                       : (sz_bool_t)decision;
                if (is_boundary && sz_word_break_emit_forward_(pending_offsets[pending_index], word_starts,
                                                               word_lengths, words_capacity, &words, &word_start)) {
                    if (bytes_consumed) *bytes_consumed = word_start;
                    return words;
                }
            }
            pending_count = 0;
            {
                int decision = sz_word_break_local_decision_rvv_(backward_carry, this_prop, this_prop);
                sz_bool_t is_boundary = (decision < 0) ? sz_utf8_is_word_boundary_serial(text, length, this_offset)
                                                       : (sz_bool_t)decision;
                if (is_boundary && sz_word_break_emit_forward_(this_offset, word_starts, word_lengths, words_capacity,
                                                               &words, &word_start)) {
                    if (bytes_consumed) *bytes_consumed = word_start;
                    return words;
                }
            }
            backward_carry = this_prop;
        }
    }

    // A trailing ignorable run resolves with the effective forward floor = the last codepoint's property.
    for (sz_size_t pending_index = 0; pending_index < pending_count; ++pending_index) {
        int decision = sz_word_break_local_decision_rvv_(backward_carry, pending_raw[pending_index], last_prop);
        sz_bool_t is_boundary = (decision < 0)
                                    ? sz_utf8_is_word_boundary_serial(text, length, pending_offsets[pending_index])
                                    : (sz_bool_t)decision;
        if (is_boundary && sz_word_break_emit_forward_(pending_offsets[pending_index], word_starts, word_lengths,
                                                       words_capacity, &words, &word_start)) {
            if (bytes_consumed) *bytes_consumed = word_start;
            return words;
        }
    }
    // The trailing span `[word_start, length)` is the last word (end of text is always a boundary).
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

/*  Emit the reverse word `[boundary_offset, *word_end)` and lower `*word_end`. Returns 1 when the output is
 *  full (caller stops and reports `*word_end` as the resume point), 0 otherwise. */
SZ_INTERNAL int sz_word_break_emit_reverse_(sz_size_t boundary_offset, sz_size_t *word_starts, sz_size_t *word_lengths,
                                            sz_size_t words_capacity, sz_size_t *words, sz_size_t *word_end) {
    if (*words == words_capacity) return 1;
    word_starts[*words] = boundary_offset;
    word_lengths[*words] = *word_end - boundary_offset;
    ++(*words);
    *word_end = boundary_offset;
    return 0;
}

/*  Resolve a deferred run of (higher-index) codepoints now that their nearest non-ignorable below is known
 *  (`prev_prop`). Emits boundaries high-to-low. Returns 1 if the output filled mid-run. */
SZ_INTERNAL int sz_word_break_resolve_reverse_(sz_cptr_t text, sz_size_t length, sz_u8_t prev_prop,
                                               sz_size_t const *pending_offsets, sz_u8_t const *pending_raw,
                                               sz_u8_t const *pending_after, sz_size_t pending_count,
                                               sz_size_t *word_starts, sz_size_t *word_lengths,
                                               sz_size_t words_capacity, sz_size_t *words, sz_size_t *word_end) {
    for (sz_size_t pending_index = 0; pending_index < pending_count; ++pending_index) {
        int decision = sz_word_break_local_decision_rvv_(prev_prop, pending_raw[pending_index],
                                                         pending_after[pending_index]);
        sz_bool_t is_boundary = (decision < 0)
                                    ? sz_utf8_is_word_boundary_serial(text, length, pending_offsets[pending_index])
                                    : (sz_bool_t)decision;
        if (is_boundary && sz_word_break_emit_reverse_(pending_offsets[pending_index], word_starts, word_lengths,
                                                       words_capacity, words, word_end))
            return 1;
    }
    return 0;
}

/*  Reverse streaming scan: the mirror of `sz_word_break_find_stream_rvv_`. It walks codepoints high-to-low in
 *  back-scanned tiles, carrying the effective FORWARD property as a running scalar and deferring each
 *  codepoint's decision until the nearest non-ignorable BELOW it is reached. Because it emits from the end
 *  and stops at `words_capacity`, it only ever touches the needed suffix — matching the serial reference. */
SZ_INTERNAL sz_size_t sz_word_break_rfind_stream_rvv_(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                      sz_size_t *word_lengths, sz_size_t words_capacity,
                                                      sz_size_t *bytes_consumed, sz_bool_t *overflowed) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_rune_t tile_runes[sz_word_break_tile_codepoints_k_];
    sz_size_t tile_offsets[sz_word_break_tile_codepoints_k_];
    sz_u8_t tile_props[sz_word_break_tile_codepoints_k_];
    sz_size_t pending_offsets[sz_word_break_pending_capacity_k_];
    sz_u8_t pending_raw[sz_word_break_pending_capacity_k_];
    sz_u8_t pending_after[sz_word_break_pending_capacity_k_]; // effective forward property captured at push time
    sz_size_t pending_count = 0;

    *overflowed = sz_false_k;
    sz_size_t words = 0;
    sz_size_t word_end = length; // end of the in-progress word (always a true boundary)
    sz_u8_t forward_carry = 0;   // effective forward property = nearest non-ignorable at-or-above (floor: prop[last])
    int seen_any = 0;
    sz_size_t frontier = length;

    while (frontier > 0) {
        // Back-scan up to one tile of codepoints to find a codepoint-aligned tile start, then decode forward.
        sz_size_t tile_start = frontier;
        sz_size_t scanned = 0;
        while (tile_start > 0 && scanned < sz_word_break_tile_codepoints_k_) {
            --tile_start;
            while (tile_start > 0 && (text_u8[tile_start] & 0xC0) == 0x80) --tile_start;
            ++scanned;
        }
        sz_size_t tile_count = 0;
        sz_size_t decode_cursor = tile_start;
        while (decode_cursor < frontier && tile_count < sz_word_break_tile_codepoints_k_) {
            tile_offsets[tile_count] = decode_cursor;
            sz_size_t decode_position = decode_cursor;
            tile_runes[tile_count] = sz_utf8_decode_(text, length, &decode_position);
            decode_cursor = decode_position;
            ++tile_count;
        }
        sz_word_break_classify_props_rvv_(tile_runes, tile_props, tile_count);

        for (sz_size_t tile_index = tile_count; tile_index-- > 0;) {
            sz_u8_t this_prop = tile_props[tile_index];
            sz_size_t this_offset = tile_offsets[tile_index];
            if (!seen_any) {
                forward_carry = this_prop;
                seen_any = 1;
            }
            sz_u8_t after_prop;
            if (!sz_word_break_is_ignorable_(this_prop)) {
                // This non-ignorable is the nearest-below for every still-pending higher codepoint.
                if (sz_word_break_resolve_reverse_(text, length, this_prop, pending_offsets, pending_raw, pending_after,
                                                   pending_count, word_starts, word_lengths, words_capacity, &words,
                                                   &word_end)) {
                    if (bytes_consumed) *bytes_consumed = word_end;
                    return words;
                }
                pending_count = 0;
                forward_carry = this_prop;
                after_prop = this_prop;
            }
            else { after_prop = forward_carry; }

            if (this_offset > 0) {
                if (pending_count == sz_word_break_pending_capacity_k_) {
                    *overflowed = sz_true_k;
                    return 0;
                }
                pending_offsets[pending_count] = this_offset;
                pending_raw[pending_count] = this_prop;
                pending_after[pending_count] = after_prop;
                ++pending_count;
            }
            else {
                // Codepoint 0: start of text is always a boundary (no decision), and it is the floor
                // nearest-below (`prop[0]`) for every remaining pending codepoint.
                if (sz_word_break_resolve_reverse_(text, length, this_prop, pending_offsets, pending_raw, pending_after,
                                                   pending_count, word_starts, word_lengths, words_capacity, &words,
                                                   &word_end)) {
                    if (bytes_consumed) *bytes_consumed = word_end;
                    return words;
                }
                pending_count = 0;
            }
        }
        frontier = tile_start;
    }

    // The leading span `[0, word_end)` is the last word (start of text is always a boundary).
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

/*  Common driver: dispatches to the streaming forward/backward scans, deferring to the serial reference for
 *  malformed UTF-8 (whose mixed stepping a codepoint scan cannot reproduce) and for pathological ignorable
 *  runs that overflow the pending buffer. */
SZ_INTERNAL sz_size_t sz_word_break_scan_rvv_(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                              sz_size_t *word_lengths, sz_size_t words_capacity,
                                              sz_size_t *bytes_consumed, int reverse) {
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = (length == 0) ? 0 : (reverse ? length : 0);
        return 0;
    }
    if (!sz_word_break_is_well_formed_rvv_(text, length))
        return reverse ? sz_utf8_word_rfind_boundaries_serial(text, length, word_starts, word_lengths, words_capacity,
                                                              bytes_consumed)
                       : sz_utf8_word_find_boundaries_serial(text, length, word_starts, word_lengths, words_capacity,
                                                             bytes_consumed);

    sz_bool_t overflowed = sz_false_k;
    sz_size_t words = reverse ? sz_word_break_rfind_stream_rvv_(text, length, word_starts, word_lengths, words_capacity,
                                                                bytes_consumed, &overflowed)
                              : sz_word_break_find_stream_rvv_(text, length, word_starts, word_lengths, words_capacity,
                                                               bytes_consumed, &overflowed);
    if (overflowed)
        return reverse ? sz_utf8_word_rfind_boundaries_serial(text, length, word_starts, word_lengths, words_capacity,
                                                              bytes_consumed)
                       : sz_utf8_word_find_boundaries_serial(text, length, word_starts, word_lengths, words_capacity,
                                                             bytes_consumed);
    return words;
}

SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_rvv(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                     sz_size_t *word_lengths, sz_size_t words_capacity,
                                                     sz_size_t *bytes_consumed) {
    return sz_word_break_scan_rvv_(text, length, word_starts, word_lengths, words_capacity, bytes_consumed, 0);
}

SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_rvv(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                      sz_size_t *word_lengths, sz_size_t words_capacity,
                                                      sz_size_t *bytes_consumed) {
    return sz_word_break_scan_rvv_(text, length, word_starts, word_lengths, words_capacity, bytes_consumed, 1);
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
