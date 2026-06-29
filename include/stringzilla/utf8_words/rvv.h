/**
 *  @brief RISC-V Vector backend for UAX-29 word boundaries.
 *  @file include/stringzilla/utf8_words/rvv.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_WORDS_RVV_H_
#define STRINGZILLA_UTF8_WORDS_RVV_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h" // `sz_rune_decode_unchecked`
#include "stringzilla/utf8_words/tables.h"
#include "stringzilla/utf8_words/serial.h"
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

/*  Vectorized UTF-8 well-formedness gate. The serial code mixes lead-length enumeration with backward decode,
 *  which disagree on malformed input, so only well-formed buffers may take the table path. Pure ASCII is
 *  trivially well-formed and detected with a single `vfirst` "any byte >= 0x80" scan (folding in the
 *  always-malformed 0xF8..0xFF reject); a non-ASCII buffer then takes one linear framing walk. */
SZ_HELPER_AUTO int sz_utf8_word_break_is_well_formed_rvv_(sz_cptr_t text, sz_size_t length) {
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
        sz_size_t codepoint_length = sz_utf8_lead_length_(lead);
        if ((lead & 0xC0) == 0x80) return 0;              // stray continuation byte at a lead position
        if (offset + codepoint_length > length) return 0; // truncated trailing sequence
        for (sz_size_t byte_index = 1; byte_index < codepoint_length; ++byte_index)
            if ((text_u8[offset + byte_index] & 0xC0) != 0x80) return 0; // missing continuation byte
        offset += codepoint_length;
    }
    return 1;
}

/** @brief Local word-break decision: 1 (break), 0 (no break), or -1 (stateful rule — consult serial). */
SZ_HELPER_AUTO int sz_utf8_word_break_local_decision_rvv_(sz_u8_t prev_prop, sz_u8_t after_raw, sz_u8_t after_prop) {
    // WB3: CR x LF
    if (prev_prop == sz_utf8_word_break_cr_k && after_raw == sz_utf8_word_break_lf_k) return 0;
    // WB3a: break after Newline/CR/LF
    if (prev_prop == sz_utf8_word_break_newline_k || prev_prop == sz_utf8_word_break_cr_k ||
        prev_prop == sz_utf8_word_break_lf_k)
        return 1;
    // WB3b: break before Newline/CR/LF (raw after, as serial)
    if (after_raw == sz_utf8_word_break_newline_k || after_raw == sz_utf8_word_break_cr_k ||
        after_raw == sz_utf8_word_break_lf_k)
        return 1;
    // WB3c: ZWJ x (anything)
    if (prev_prop == sz_utf8_word_break_zwj_k) return 0;

    // From here serial uses the WB4-effective `after_prop`.
    int prev_ah = (prev_prop == sz_utf8_word_break_aletter_k || prev_prop == sz_utf8_word_break_hebrew_letter_k);
    int after_ah = (after_prop == sz_utf8_word_break_aletter_k || after_prop == sz_utf8_word_break_hebrew_letter_k);
    int prev_num = (prev_prop == sz_utf8_word_break_numeric_k);
    int after_num = (after_prop == sz_utf8_word_break_numeric_k);
    int after_q = (after_prop == sz_utf8_word_break_mid_quotes_k);

    // WB5: AHLetter x AHLetter
    if (prev_ah && after_ah) return 0;

    // WB6 trigger: AHLetter x (MidLetter|MidNumLetQ) -> needs lookahead.
    if (prev_ah && (after_prop == sz_utf8_word_break_midletter_k || after_q)) return -1;

    // WB7 trigger: (MidLetter|MidNumLetQ) x AHLetter -> needs lookback.
    if ((prev_prop == sz_utf8_word_break_midletter_k || prev_prop == sz_utf8_word_break_mid_quotes_k) && after_ah)
        return -1;

    // WB7a: Hebrew_Letter x Single_Quote (mid-quotes)
    if (prev_prop == sz_utf8_word_break_hebrew_letter_k && after_prop == sz_utf8_word_break_mid_quotes_k) return 0;

    // WB8: Numeric x Numeric
    if (prev_num && after_num) return 0;
    // WB9: AHLetter x Numeric
    if (prev_ah && after_num) return 0;
    // WB10: Numeric x AHLetter
    if (prev_num && after_ah) return 0;

    // WB11 trigger: Numeric x (MidNum|MidNumLetQ) -> needs lookahead.
    if (prev_num && (after_prop == sz_utf8_word_break_midnum_k || after_q)) return -1;
    // WB12 trigger: (MidNum|MidNumLetQ) x Numeric -> needs lookback.
    if ((prev_prop == sz_utf8_word_break_midnum_k || prev_prop == sz_utf8_word_break_mid_quotes_k) && after_num)
        return -1;

    // WB13: Katakana x Katakana
    if (prev_prop == sz_utf8_word_break_katakana_k && after_prop == sz_utf8_word_break_katakana_k) return 0;
    // WB13a: (AHLetter|Numeric|Katakana|ExtendNumLet) x ExtendNumLet
    if ((prev_ah || prev_num || prev_prop == sz_utf8_word_break_katakana_k ||
         prev_prop == sz_utf8_word_break_extendnumlet_k) &&
        after_prop == sz_utf8_word_break_extendnumlet_k)
        return 0;
    // WB13b: ExtendNumLet x (AHLetter|Numeric|Katakana)
    if (prev_prop == sz_utf8_word_break_extendnumlet_k &&
        (after_ah || after_num || after_prop == sz_utf8_word_break_katakana_k))
        return 0;

    // WB15/16 trigger: RI x RI -> parity dependent.
    if (prev_prop == sz_utf8_word_break_regional_ind_k && after_prop == sz_utf8_word_break_regional_ind_k) return -1;

    // WB999: break.
    return 1;
}

/** @brief Per-codepoint non-ASCII word-break step, reproducing `sz_utf8_is_word_boundary_serial` exactly. */
SZ_HELPER_AUTO sz_bool_t sz_utf8_word_break_decision_at_rvv_(sz_cptr_t text, sz_size_t length, sz_size_t position) {
    sz_u8_t prev_prop = sz_utf8_word_break_previous_property_(text, position);
    sz_size_t after_position = position;
    sz_u8_t after_raw = sz_rune_word_break_property(sz_utf8_next_rune_(text, length, &after_position));
    sz_u8_t after_prop = sz_utf8_word_break_effective_property_(text, length, position, (sz_size_t *)0);
    int decision = sz_utf8_word_break_local_decision_rvv_(prev_prop, after_raw, after_prop);
    return (decision < 0) ? sz_utf8_is_word_boundary_serial(text, length, position) : (sz_bool_t)decision;
}

/*  All-ASCII window word-break boundary mask. ASCII has no Extend/Format/ZWJ/RI/Hebrew/Katakana, so the
 *  look-around rules reduce to neighbour slides (`vslide1up` for i-1/i-2, `vslide1down` for i+1). Bit `i` is
 *  set when the boundary BEFORE lane `i` is not suppressed, restricted to trusted lanes [2, vl-2]. */
SZ_HELPER_AUTO vbool2_t sz_utf8_word_break_ascii_boundary_mask_rvv_(vuint8m4_t props_u8m4, sz_size_t vector_length) {
    // Effective neighbour properties via slides (the fill values land outside the trusted lanes).
    vuint8m4_t prev1_u8m4 = __riscv_vslide1up_vx_u8m4(props_u8m4, 0, vector_length);   // lane i-1
    vuint8m4_t prev2_u8m4 = __riscv_vslide1up_vx_u8m4(prev1_u8m4, 0, vector_length);   // lane i-2
    vuint8m4_t next1_u8m4 = __riscv_vslide1down_vx_u8m4(props_u8m4, 0, vector_length); // lane i+1

    // Per-class predicates for this lane and its slid neighbours. ASCII has no Hebrew letter, so AHLetter == ALetter.
    vbool2_t cur_aletter = __riscv_vmseq_vx_u8m4_b2(props_u8m4, sz_utf8_word_break_aletter_k, vector_length);
    vbool2_t cur_numeric = __riscv_vmseq_vx_u8m4_b2(props_u8m4, sz_utf8_word_break_numeric_k, vector_length);
    vbool2_t cur_extendnumlet = __riscv_vmseq_vx_u8m4_b2(props_u8m4, sz_utf8_word_break_extendnumlet_k, vector_length);
    vbool2_t cur_lf = __riscv_vmseq_vx_u8m4_b2(props_u8m4, sz_utf8_word_break_lf_k, vector_length);

    vbool2_t prev1_aletter = __riscv_vmseq_vx_u8m4_b2(prev1_u8m4, sz_utf8_word_break_aletter_k, vector_length);
    vbool2_t prev1_numeric = __riscv_vmseq_vx_u8m4_b2(prev1_u8m4, sz_utf8_word_break_numeric_k, vector_length);
    vbool2_t prev1_extendnumlet = __riscv_vmseq_vx_u8m4_b2(prev1_u8m4, sz_utf8_word_break_extendnumlet_k,
                                                           vector_length);
    vbool2_t prev1_cr = __riscv_vmseq_vx_u8m4_b2(prev1_u8m4, sz_utf8_word_break_cr_k, vector_length);
    vbool2_t prev1_midletter = __riscv_vmseq_vx_u8m4_b2(prev1_u8m4, sz_utf8_word_break_midletter_k, vector_length);
    vbool2_t prev1_midnum = __riscv_vmseq_vx_u8m4_b2(prev1_u8m4, sz_utf8_word_break_midnum_k, vector_length);
    vbool2_t prev1_quotes = __riscv_vmseq_vx_u8m4_b2(prev1_u8m4, sz_utf8_word_break_mid_quotes_k, vector_length);

    vbool2_t prev2_aletter = __riscv_vmseq_vx_u8m4_b2(prev2_u8m4, sz_utf8_word_break_aletter_k, vector_length);
    vbool2_t prev2_numeric = __riscv_vmseq_vx_u8m4_b2(prev2_u8m4, sz_utf8_word_break_numeric_k, vector_length);

    vbool2_t next1_aletter = __riscv_vmseq_vx_u8m4_b2(next1_u8m4, sz_utf8_word_break_aletter_k, vector_length);
    vbool2_t next1_numeric = __riscv_vmseq_vx_u8m4_b2(next1_u8m4, sz_utf8_word_break_numeric_k, vector_length);

    // Helper combinations (MidLetter|Quotes), (MidNum|Quotes), (ALetter|Numeric).
    vbool2_t prev1_midletter_or_q = __riscv_vmor_mm_b2(prev1_midletter, prev1_quotes, vector_length);
    vbool2_t prev1_midnum_or_q = __riscv_vmor_mm_b2(prev1_midnum, prev1_quotes, vector_length);
    vbool2_t cur_midletter = __riscv_vmseq_vx_u8m4_b2(props_u8m4, sz_utf8_word_break_midletter_k, vector_length);
    vbool2_t cur_midnum = __riscv_vmseq_vx_u8m4_b2(props_u8m4, sz_utf8_word_break_midnum_k, vector_length);
    vbool2_t cur_quotes = __riscv_vmseq_vx_u8m4_b2(props_u8m4, sz_utf8_word_break_mid_quotes_k, vector_length);
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
SZ_HELPER_INLINE vuint8m4_t sz_utf8_word_break_classify_ascii_bytes_rvv_(vuint8m4_t bytes_u8m4,
                                                                         sz_size_t vector_length) {
    vuint8m4_t indices_u8m4 = __riscv_vand_vx_u8m4(bytes_u8m4, 0x7F, vector_length);
    return __riscv_vluxei8_v_u8m4(&sz_utf8_word_break_property_ascii_[0], indices_u8m4, vector_length);
}

/*  Emit one all-ASCII window's boundaries as a shifted-difference `(start, length)` stream, parameterized by
 *  direction. Forward: lane 0 of `starts` carries the open `word_start`, `lengths = boundary - start`. Reverse:
 *  the compacted indices are reversed (`vrgatherei16`) so emission walks high-to-low, `starts = boundary`,
 *  `lengths = previous - boundary`. In both directions the carried boundary is the last emitted boundary.
 *  Returns the number of boundaries emitted, writes whether the output buffer filled into `*filled`. */
SZ_HELPER_AUTO sz_size_t sz_utf8_word_break_emit_ascii_rvv_(                                    //
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
SZ_HELPER_AUTO sz_size_t sz_utf8_words_rvv_(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                            sz_size_t *word_lengths, sz_size_t words_capacity,
                                            sz_size_t *bytes_consumed) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t words = 0;
    sz_size_t word_start = 0;                              // Start of the in-progress word (always a true boundary).
    sz_size_t position = sz_utf8_lead_length_(text_u8[0]); // position 0 is a boundary; start past it

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
        position += sz_utf8_lead_length_(text_u8[position]);
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

/** @brief Common driver: route malformed UTF-8 to serial, otherwise run the vectorized window scan. */
SZ_HELPER_AUTO sz_size_t sz_utf8_word_break_scan_rvv_(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                      sz_size_t *word_lengths, sz_size_t words_capacity,
                                                      sz_size_t *bytes_consumed) {
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    if (!sz_utf8_word_break_is_well_formed_rvv_(text, length))
        return sz_utf8_words_serial(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
    return sz_utf8_words_rvv_(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
}

SZ_API_COMPTIME sz_size_t sz_utf8_words_rvv(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                            sz_size_t *word_lengths, sz_size_t words_capacity,
                                            sz_size_t *bytes_consumed) {
    return sz_utf8_word_break_scan_rvv_(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
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

#endif // STRINGZILLA_UTF8_WORDS_RVV_H_
