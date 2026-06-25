/**
 *  @brief RISC-V Vector backend for UTF-8 codepoint mechanics.
 *  @file include/stringzilla/utf8_runes/rvv.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_RUNES_RVV_H_
#define STRINGZILLA_UTF8_RUNES_RVV_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h" // `sz_rune_decode`, `sz_utf8_incomplete_tail_`
#include "stringzilla/utf8_runes/serial.h"

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
 *  @brief Decode UTF-8 to UTF-32 codepoints, byte-for-byte equivalent to `sz_utf8_decode_serial`.
 *
 *  Vector-classifies the maximal ASCII prefix of each strip (`vmsgtu` high bit, `vfirst`) and widens it
 *  straight into the rune output; any non-ASCII byte takes a single serial `sz_rune_decode` step.
 */
/** @brief Widen `count` ASCII bytes (u8 -> u16 -> u32) and store them as runes. */
SZ_INTERNAL void sz_utf8_decode_ascii_run_rvv_(sz_rune_t *runes_out, sz_u8_t const *src, sz_size_t count) {
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

/**
 *  @brief  Decode a maximal run of well-formed 2-byte sequences (Latin-1/Cyrillic/Greek/Hebrew/Arabic ranges) with
 *          ZERO gather, the 2-byte sibling of @ref sz_utf8_decode_ascii_run_rvv_.
 *
 *  `vlseg2e8` deinterleaves the byte stream into a lead lane and a continuation lane in two registers; the run is
 *  validated entirely in-register (every lead in `[0xC2, 0xDF]` rejecting the `C0/C1` overlongs, every continuation in
 *  `[0x80, 0xBF]`) and a `vfirst` over the fault mask bounds the maximal good prefix. The kept prefix decodes by
 *  `((lead & 0x1F) << 6) | (cont & 0x3F)`, widens to `u32`, and stores densely; whatever follows (a non-2-byte lead, a
 *  bad pair, or a half pair at the edge) is handed back to the general window path. Reads only whole pairs, so it never
 *  over-reads the input.
 *
 *  @param  text            Cursor positioned at a 2-byte lead.
 *  @param  length          Bytes remaining from @p text.
 *  @param  runes           Output buffer.
 *  @param  capacity        Remaining rune capacity.
 *  @param  consumed_bytes  Set to `2 * runes_emitted` (the byte span of the decoded prefix).
 *  @return Number of runes emitted (0 if the very first pair is not a well-formed 2-byte sequence).
 */
SZ_INTERNAL sz_size_t sz_utf8_decode_two_byte_run_rvv_( //
    sz_cptr_t text, sz_size_t length, sz_rune_t *runes, sz_size_t capacity, sz_size_t *consumed_bytes) {

    sz_u8_t const *bytes = (sz_u8_t const *)text;
    sz_size_t const pairs_in_input = length / 2; // Whole [lead, cont] pairs available.
    sz_size_t const pairs_wanted = pairs_in_input < capacity ? pairs_in_input : capacity;
    sz_size_t emitted = 0;

    while (emitted < pairs_wanted) {
        sz_size_t const vector_length = __riscv_vsetvl_e8m2(pairs_wanted - emitted);
        vuint8m2x2_t const deinterleaved = __riscv_vlseg2e8_v_u8m2x2(bytes + emitted * 2, vector_length);
        vuint8m2_t const lead = __riscv_vget_v_u8m2x2_u8m2(deinterleaved, 0);
        vuint8m2_t const cont = __riscv_vget_v_u8m2x2_u8m2(deinterleaved, 1);

        // A lane is a well-formed 2-byte sequence iff its lead is in `[0xC2, 0xDF]` (rejecting the C0/C1 overlongs and
        // any 3/4-byte or stray lead) and its continuation is in `[0x80, 0xBF]`; the first fault bounds the run.
        vbool4_t const lead_bad = __riscv_vmor_mm_b4(__riscv_vmsltu_vx_u8m2_b4(lead, 0xC2, vector_length),
                                                     __riscv_vmsgtu_vx_u8m2_b4(lead, 0xDF, vector_length),
                                                     vector_length);
        vbool4_t const cont_bad = __riscv_vmsne_vx_u8m2_b4(__riscv_vand_vx_u8m2(cont, 0xC0, vector_length), 0x80,
                                                           vector_length);
        vbool4_t const pair_bad = __riscv_vmor_mm_b4(lead_bad, cont_bad, vector_length);
        long const first_bad = __riscv_vfirst_m_b4(pair_bad, vector_length);
        sz_size_t const take = first_bad < 0 ? vector_length : (sz_size_t)first_bad;

        if (take) {
            vuint16m4_t const lead_wide = __riscv_vwcvtu_x_x_v_u16m4(lead, take);
            vuint16m4_t const cont_wide = __riscv_vwcvtu_x_x_v_u16m4(cont, take);
            vuint16m4_t const codepoints = __riscv_vor_vv_u16m4(
                __riscv_vsll_vx_u16m4(__riscv_vand_vx_u16m4(lead_wide, 0x1F, take), 6, take),
                __riscv_vand_vx_u16m4(cont_wide, 0x3F, take), take);
            __riscv_vse32_v_u32m8(runes + emitted, __riscv_vwcvtu_x_x_v_u32m8(codepoints, take), take);
        }
        emitted += take;
        if (first_bad >= 0) break; // The run ended at a non-2-byte / malformed pair.
    }

    *consumed_bytes = emitted * 2;
    return emitted;
}

/**
 *  @brief  Decode the dense set of EMITTED-start lanes of one classified window into sequential UTF-32 runes,
 *          the RVV sibling of @ref sz_utf8_rune_drain_icelake_.
 *
 *  Decodes EVERY lane in place: three constant-stride `vslidedown` passes (stride 1/2/3) bring each lane its three
 *  forward neighbours with fixed latency, a branchless width-blend (1/2/3/4-byte) assembles a value at full `u32`
 *  width on every lane, and ONE `vcompress` over @p emit_mask packs the emitted-start lanes to the low lanes for a
 *  dense store. This replaces the four per-codepoint `vrgather` passes (the throughput bottleneck on every real RVV
 *  core) with cheap fixed-stride slides, mirroring the slide-not-gather decode of the Ice Lake window kernel.
 *
 *  TOTAL decode: @p emit_mask also covers promoted orphan continuation bytes, and @p ill_mask marks every emitted
 *  lane whose maximal ill-formed subpart must collapse to a single U+FFFD (Unicode 17.0 §3.9 / W3C). The width-blend
 *  assembles the (for ill-formed lanes, garbage) values, then those lanes are overwritten with U+FFFD before the
 *  `vcompress`. The resume cursor reads the per-lane @p consumed_length (the maximal-subpart length, in window order)
 *  at the last emitted lane, so an ill-formed trailing lane never skips bytes that owe their own next U+FFFD.
 *
 *  @param  window_bytes      The raw window bytes loaded at `e8m1` (lanes `[0, window)` valid).
 *  @param  emit_mask         Mask of decodable emitted-start lanes (true leads + promoted orphan continuations).
 *  @param  ill_mask          Mask of emitted lanes that must collapse to one U+FFFD (subset of @p emit_mask).
 *  @param  consumed_length   Per-lane maximal-subpart byte length (1-4) for each emitted lane (window order).
 *  @param  decodable         Population count of @p emit_mask (number of runes to emit, before capping).
 *  @param  vector_length     The active `e8m1` vector length of this window.
 *  @param  runes             Output buffer; at most @p capacity runes are written.
 *  @param  capacity          Remaining rune capacity.
 *  @param  consumed_bytes    Set to the byte span the emitted runes cover (the resume-cursor delta).
 *  @return Number of runes emitted.
 */
SZ_INTERNAL sz_size_t sz_utf8_rune_drain_rvv_( //
    vuint8m1_t window_bytes, vbool8_t emit_mask, vbool8_t ill_mask, vuint8m1_t consumed_length, sz_size_t decodable,
    sz_size_t vector_length, sz_rune_t *runes, sz_size_t capacity, sz_size_t *consumed_bytes) {

    sz_size_t const want = decodable < capacity ? decodable : capacity;

    // Bring each lane its three forward neighbours by constant-stride slides (lanes past the end read 0); no gather.
    vuint8m1_t const second_bytes = __riscv_vslidedown_vx_u8m1(window_bytes, 1, vector_length);
    vuint8m1_t const third_bytes = __riscv_vslidedown_vx_u8m1(window_bytes, 2, vector_length);
    vuint8m1_t const fourth_bytes = __riscv_vslidedown_vx_u8m1(window_bytes, 3, vector_length);

    // Widen the four byte streams to u32 over the whole window so the codepoint arithmetic never overflows a SMP value.
    vuint32m4_t const lead = __riscv_vzext_vf4_u32m4(window_bytes, vector_length);
    vuint32m4_t const second = __riscv_vzext_vf4_u32m4(second_bytes, vector_length);
    vuint32m4_t const third = __riscv_vzext_vf4_u32m4(third_bytes, vector_length);
    vuint32m4_t const fourth = __riscv_vzext_vf4_u32m4(fourth_bytes, vector_length);

    // Width classes from the lead byte, mirroring `sz_rune_decode` reconstruction exactly.
    vbool8_t const is_two = __riscv_vmand_mm_b8(__riscv_vmsgeu_vx_u32m4_b8(lead, 0xC0, vector_length),
                                                __riscv_vmsltu_vx_u32m4_b8(lead, 0xE0, vector_length), vector_length);
    vbool8_t const is_three = __riscv_vmand_mm_b8(__riscv_vmsgeu_vx_u32m4_b8(lead, 0xE0, vector_length),
                                                  __riscv_vmsltu_vx_u32m4_b8(lead, 0xF0, vector_length), vector_length);
    vbool8_t const is_four = __riscv_vmsgeu_vx_u32m4_b8(lead, 0xF0, vector_length);

    // ASCII keeps the lead value; each multi-byte class is blended on top in-register.
    vuint32m4_t codepoints = lead;

    // 2-byte: ((lead & 0x1F) << 6) | (second & 0x3F).
    vuint32m4_t const two_byte = __riscv_vor_vv_u32m4(
        __riscv_vsll_vx_u32m4(__riscv_vand_vx_u32m4(lead, 0x1F, vector_length), 6, vector_length),
        __riscv_vand_vx_u32m4(second, 0x3F, vector_length), vector_length);
    codepoints = __riscv_vmerge_vvm_u32m4(codepoints, two_byte, is_two, vector_length);

    // 3-byte: ((lead & 0x0F) << 12) | ((second & 0x3F) << 6) | (third & 0x3F).
    vuint32m4_t const three_byte = __riscv_vor_vv_u32m4(
        __riscv_vor_vv_u32m4(
            __riscv_vsll_vx_u32m4(__riscv_vand_vx_u32m4(lead, 0x0F, vector_length), 12, vector_length),
            __riscv_vsll_vx_u32m4(__riscv_vand_vx_u32m4(second, 0x3F, vector_length), 6, vector_length), vector_length),
        __riscv_vand_vx_u32m4(third, 0x3F, vector_length), vector_length);
    codepoints = __riscv_vmerge_vvm_u32m4(codepoints, three_byte, is_three, vector_length);

    // 4-byte: ((lead & 0x07) << 18) | ((second & 0x3F) << 12) | ((third & 0x3F) << 6) | (fourth & 0x3F).
    vuint32m4_t const four_byte = __riscv_vor_vv_u32m4(
        __riscv_vor_vv_u32m4(
            __riscv_vsll_vx_u32m4(__riscv_vand_vx_u32m4(lead, 0x07, vector_length), 18, vector_length),
            __riscv_vsll_vx_u32m4(__riscv_vand_vx_u32m4(second, 0x3F, vector_length), 12, vector_length),
            vector_length),
        __riscv_vor_vv_u32m4(__riscv_vsll_vx_u32m4(__riscv_vand_vx_u32m4(third, 0x3F, vector_length), 6, vector_length),
                             __riscv_vand_vx_u32m4(fourth, 0x3F, vector_length), vector_length),
        vector_length);
    codepoints = __riscv_vmerge_vvm_u32m4(codepoints, four_byte, is_four, vector_length);

    // Overwrite every ill-formed emitted lane with U+FFFD; the garbage decode at those lanes is discarded.
    codepoints = __riscv_vmerge_vxm_u32m4(codepoints, (sz_rune_t)sz_rune_replacement_k, ill_mask, vector_length);

    // ONE compress packs the values at the emitted-start lanes down to a dense run; store the kept prefix.
    vuint32m4_t const packed = __riscv_vcompress_vm_u32m4(codepoints, emit_mask, vector_length);
    __riscv_vse32_v_u32m4(runes, packed, want);

    // Resume delta = last emitted start offset + its maximal-subpart length. Compress the start byte-offsets and the
    // per-lane consumed lengths the same way (no gather) so the `want`-th start and its own subpart length drive the
    // exact serial resume point (for a well-formed lane the subpart length equals the declared rune length; for an
    // ill-formed lane it is the 1-3 byte subpart, so the cursor never skips bytes owing their own next U+FFFD).
    vuint8m1_t const lane_identity = __riscv_vid_v_u8m1(vector_length);
    vuint8m1_t const start_offsets = __riscv_vcompress_vm_u8m1(lane_identity, emit_mask, vector_length);
    vuint8m1_t const consumed_compressed = __riscv_vcompress_vm_u8m1(consumed_length, emit_mask, vector_length);
    sz_size_t const last_start = (sz_size_t)__riscv_vmv_x_s_u8m1_u8(
        __riscv_vslidedown_vx_u8m1(start_offsets, want - 1, vector_length));
    sz_size_t const last_length = (sz_size_t)__riscv_vmv_x_s_u8m1_u8(
        __riscv_vslidedown_vx_u8m1(consumed_compressed, want - 1, vector_length));
    *consumed_bytes = last_start + last_length;
    return want;
}

/**
 *  @brief  Decode one `e8m1` window of @p text into dense UTF-32 @p runes by the uniform TOTAL "classify -> per-lane
 *          well-formed + orphan promotion -> compress emitted starts -> slide-gather -> width-blend -> merge U+FFFD"
 *          path, the RVV sibling of @ref sz_utf8_decode_once_icelake_. The decode is TOTAL: well-formed leads,
 *          ill-formed leads and orphan continuation bytes are all handled in-vector, one U+FFFD per maximal ill-formed
 *          subpart (Unicode 17.0 §3.9 / W3C), bit-exact with the serial reference. The step declines
 *          (`*runes_unpacked == 0`, cursor unchanged) ONLY when the first lead's declared sequence crosses the window
 *          edge (a boundary truncation), which the public entry finalizes without a serial re-decode.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_decode_once_rvv_( //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {

    // Cap the window at 192 bytes so every lane index, length, and `lane + length` stays exact in the `u8` domain
    // used for the gather offsets and the overrun test (`e8m1` reaches 256 lanes only at VLEN 2048). The cap costs
    // nothing on today's 128/256/512-bit parts, where `e8m1` already yields 16/32/64 lanes.
    sz_size_t const request = length < 192 ? length : 192;
    sz_size_t const vector_length = __riscv_vsetvl_e8m1(request);
    sz_size_t const window = vector_length; // Bytes loaded into lanes [0, window).
    vuint8m1_t const window_bytes = __riscv_vle8_v_u8m1((sz_u8_t const *)text, vector_length);

    // Single-source classification: the lead's high nibble selects its declared length {1x12,2,2,3,4} - both the
    // soundness gate and the width-blend derive from it, so a lead and its length can never disagree, and the LUT
    // forces 0xF8..0xFF (high nibble 0xF) to length 4 where the bad-lead test then rejects them.
    vuint8m1_t const length_lut = __riscv_vle8_v_u8m1(
        (sz_u8_t const *)(sz_u8_t[16]) {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4}, 16);
    vuint8m1_t const high_nibble = __riscv_vsrl_vx_u8m1(window_bytes, 4, vector_length);
    vuint8m1_t const lengths = __riscv_vrgather_vv_u8m1(length_lut, high_nibble, vector_length);

    // Start bytes: every loaded lane that is not a continuation byte `0x80..0xBF` (i.e. `(byte & 0xC0) != 0x80`).
    vuint8m1_t const top_two = __riscv_vand_vx_u8m1(window_bytes, 0xC0, vector_length);
    vbool8_t const continuations = __riscv_vmseq_vx_u8m1_b8(top_two, 0x80, vector_length);
    vbool8_t const starts = __riscv_vmnot_m_b8(continuations, vector_length);

    // Defer every start whose declared sequence would reach past the window: well-formed text has only the trailing
    // one (a resumable truncation), but a malformed lead-in-lead (e.g. `E0 C0`) can overrun earlier - the FIRST
    // overrunning start bounds the decodable prefix, and its bytes resume next window or via serial.
    vuint8m1_t const sequence_end = __riscv_vadd_vv_u8m1(__riscv_vid_v_u8m1(vector_length), lengths, vector_length);
    vbool8_t const overruns = __riscv_vmand_mm_b8(
        __riscv_vmsgtu_vx_u8m1_b8(sequence_end, (sz_u8_t)window, vector_length), starts, vector_length);
    long const first_overrun = __riscv_vfirst_m_b8(overruns, vector_length);
    sz_size_t const decodable_end = first_overrun < 0 ? window : (sz_size_t)first_overrun;

    // The decodable span `[0, decodable_end)` bounds every emitted start. A decodable 3/4-byte lead near the edge may
    // read trailing bytes at lanes up to `decodable_end + 2` (still inside the window, since only a WINDOW overrun
    // bounds `decodable_end`), and those trailing slots are read in-lane below by `vslidedown` of the byte stream.
    vuint8m1_t const lane_identity = __riscv_vid_v_u8m1(vector_length);
    vbool8_t const within_decodable = __riscv_vmsltu_vx_u8m1_b8(lane_identity, (sz_u8_t)decodable_end, vector_length);

    // TOTAL per-lane classification (no decline). Mirrors `sz_utf8_decode_once_icelake_`: every start lane is a
    // candidate, well-formed leads decode to their value, ill-formed leads and orphan continuation bytes each collapse
    // to one U+FFFD over their maximal ill-formed subpart (Unicode 17.0 §3.9 / W3C). The bit algebra is carried in the
    // lane domain: each lead-lane predicate is a 0/1 byte vector, "shift toward continuations" is `vslideup`, and the
    // lead lane reads its trailing continuation slots via `vslidedown` of the continuation flags.
    vbool8_t const len_eq_two = __riscv_vmseq_vx_u8m1_b8(lengths, 2, vector_length);
    vbool8_t const len_ge_two = __riscv_vmsgeu_vx_u8m1_b8(lengths, 2, vector_length);
    vbool8_t const len_ge_three = __riscv_vmsgeu_vx_u8m1_b8(lengths, 3, vector_length);
    vbool8_t const len_eq_four = __riscv_vmseq_vx_u8m1_b8(lengths, 4, vector_length);

    // Length classes restricted to start lanes (a continuation byte's "length" from the LUT is meaningless).
    vbool8_t const start_len_one = __riscv_vmandn_mm_b8(starts, len_ge_two, vector_length);
    vbool8_t const start_len_two = __riscv_vmand_mm_b8(
        starts, __riscv_vmandn_mm_b8(len_ge_two, len_ge_three, vector_length), vector_length);
    vbool8_t const start_len_three = __riscv_vmand_mm_b8(
        starts, __riscv_vmandn_mm_b8(len_ge_three, len_eq_four, vector_length), vector_length);
    vbool8_t const start_len_four = __riscv_vmand_mm_b8(starts, len_eq_four, vector_length);
    vbool8_t const start_len_ge_two = __riscv_vmand_mm_b8(starts, len_ge_two, vector_length);
    vbool8_t const start_len_ge_three = __riscv_vmand_mm_b8(starts, len_ge_three, vector_length);
    vbool8_t const start_len_ge_four = start_len_four;

    // 0/1 flag for whether each lane is a continuation byte, so the lead lane can read its trailing slots by slide-down
    // (lane i reads lane i+1/i+2/i+3). An off-window source lane reads 0, i.e. "no continuation present".
    vuint8m1_t const zero_bytes = __riscv_vmv_v_x_u8m1(0, vector_length);
    vuint8m1_t const cont_flag = __riscv_vmerge_vxm_u8m1(zero_bytes, 1, continuations, vector_length);
    vbool8_t const cont1 = __riscv_vmand_mm_b8(
        starts, __riscv_vmsne_vx_u8m1_b8(__riscv_vslidedown_vx_u8m1(cont_flag, 1, vector_length), 0, vector_length),
        vector_length);
    vbool8_t const cont2 = __riscv_vmand_mm_b8(
        starts, __riscv_vmsne_vx_u8m1_b8(__riscv_vslidedown_vx_u8m1(cont_flag, 2, vector_length), 0, vector_length),
        vector_length);
    vbool8_t const cont3 = __riscv_vmand_mm_b8(
        starts, __riscv_vmsne_vx_u8m1_b8(__riscv_vslidedown_vx_u8m1(cont_flag, 3, vector_length), 0, vector_length),
        vector_length);

    // Bad leads: a 2-byte lead < 0xC2 (C0/C1 overlong), or a 4-byte lead > 0xF4 (F5..FF out of range, length-4 by LUT).
    vbool8_t const bad_two = __riscv_vmand_mm_b8(
        start_len_two, __riscv_vmsltu_vx_u8m1_b8(window_bytes, 0xC2, vector_length), vector_length);
    vbool8_t const bad_four = __riscv_vmand_mm_b8(
        start_len_four, __riscv_vmsgtu_vx_u8m1_b8(window_bytes, 0xF4, vector_length), vector_length);
    vbool8_t const bad_lead = __riscv_vmor_mm_b8(bad_two, bad_four, vector_length);

    // First-continuation range violations for E0/ED/F0/F4 (overlong 3/4-byte, surrogate, > U+10FFFF), read at the lead
    // lane via slide-down. For any other lead these are empty, so `b1_range_ok` keeps the lane.
    vuint8m1_t const next_byte = __riscv_vslidedown_vx_u8m1(window_bytes, 1, vector_length);
    vbool8_t const e0_overlong = __riscv_vmand_mm_b8(__riscv_vmseq_vx_u8m1_b8(window_bytes, 0xE0, vector_length),
                                                     __riscv_vmsltu_vx_u8m1_b8(next_byte, 0xA0, vector_length),
                                                     vector_length);
    vbool8_t const ed_surrogate = __riscv_vmand_mm_b8(__riscv_vmseq_vx_u8m1_b8(window_bytes, 0xED, vector_length),
                                                      __riscv_vmsgeu_vx_u8m1_b8(next_byte, 0xA0, vector_length),
                                                      vector_length);
    vbool8_t const f0_overlong = __riscv_vmand_mm_b8(__riscv_vmseq_vx_u8m1_b8(window_bytes, 0xF0, vector_length),
                                                     __riscv_vmsltu_vx_u8m1_b8(next_byte, 0x90, vector_length),
                                                     vector_length);
    vbool8_t const f4_range = __riscv_vmand_mm_b8(__riscv_vmseq_vx_u8m1_b8(window_bytes, 0xF4, vector_length),
                                                  __riscv_vmsgeu_vx_u8m1_b8(next_byte, 0x90, vector_length),
                                                  vector_length);
    vbool8_t const range_bad = __riscv_vmand_mm_b8(
        starts,
        __riscv_vmor_mm_b8(__riscv_vmor_mm_b8(e0_overlong, ed_surrogate, vector_length),
                           __riscv_vmor_mm_b8(f0_overlong, f4_range, vector_length), vector_length),
        vector_length);
    vbool8_t const b1_range_ok = __riscv_vmandn_mm_b8(starts, range_bad, vector_length);
    // A valid first continuation: present, range-valid, and the lead is not itself bad (C0/C1, F5..FF).
    vbool8_t const first_ok = __riscv_vmandn_mm_b8(__riscv_vmand_mm_b8(cont1, b1_range_ok, vector_length), bad_lead,
                                                   vector_length);

    // Well-formed leads (vectorized `sz_rune_decode` success).
    vbool8_t const wf1 = start_len_one;
    vbool8_t const wf2 = __riscv_vmand_mm_b8(__riscv_vmandn_mm_b8(start_len_two, bad_lead, vector_length), cont1,
                                             vector_length);
    vbool8_t const wf3 = __riscv_vmand_mm_b8(__riscv_vmand_mm_b8(start_len_three, b1_range_ok, vector_length),
                                             __riscv_vmand_mm_b8(cont1, cont2, vector_length), vector_length);
    vbool8_t const wf4 = __riscv_vmand_mm_b8(
        __riscv_vmand_mm_b8(__riscv_vmandn_mm_b8(start_len_four, bad_lead, vector_length), b1_range_ok, vector_length),
        __riscv_vmand_mm_b8(__riscv_vmand_mm_b8(cont1, cont2, vector_length), cont3, vector_length), vector_length);
    vbool8_t const well_formed_all = __riscv_vmor_mm_b8(__riscv_vmor_mm_b8(wf1, wf2, vector_length),
                                                        __riscv_vmor_mm_b8(wf3, wf4, vector_length), vector_length);
    vbool8_t const well_formed = __riscv_vmand_mm_b8(well_formed_all, within_decodable, vector_length);

    // Per-lane maximal-subpart steps (mirror of `sz_utf8_maximal_subpart_`): a lead extends across each continuation
    // slot a well-formed sequence would still accept. step2/3/4 are LEAD-lane flags marking that the 2nd/3rd/4th byte
    // is consumed by this lead's maximal subpart (well-formed OR the bytes an ill-formed lead's single U+FFFD covers).
    vbool8_t const step2 = __riscv_vmand_mm_b8(start_len_ge_two, first_ok, vector_length);
    vbool8_t const step3 = __riscv_vmand_mm_b8(__riscv_vmand_mm_b8(step2, start_len_ge_three, vector_length), cont2,
                                               vector_length);
    vbool8_t const step4 = __riscv_vmand_mm_b8(__riscv_vmand_mm_b8(step3, start_len_ge_four, vector_length), cont3,
                                               vector_length);

    // Orphan promotion: a continuation byte not covered by ANY lead's maximal-subpart span becomes its own 1-byte
    // U+FFFD. The covered slots are the step2/3/4 lead flags smeared forward by 1/2/3 lanes (toward the continuation
    // they consume), restricted to the decodable span. `vslideup` shifts toward higher lanes (off-window reads 0).
    vbool8_t const step2_dec = __riscv_vmand_mm_b8(step2, within_decodable, vector_length);
    vbool8_t const step3_dec = __riscv_vmand_mm_b8(step3, within_decodable, vector_length);
    vbool8_t const step4_dec = __riscv_vmand_mm_b8(step4, within_decodable, vector_length);
    vuint8m1_t const step2_flag = __riscv_vmerge_vxm_u8m1(zero_bytes, 1, step2_dec, vector_length);
    vuint8m1_t const step3_flag = __riscv_vmerge_vxm_u8m1(zero_bytes, 1, step3_dec, vector_length);
    vuint8m1_t const step4_flag = __riscv_vmerge_vxm_u8m1(zero_bytes, 1, step4_dec, vector_length);
    vuint8m1_t const covered_bytes = __riscv_vor_vv_u8m1(
        __riscv_vor_vv_u8m1(__riscv_vslideup_vx_u8m1(zero_bytes, step2_flag, 1, vector_length),
                            __riscv_vslideup_vx_u8m1(zero_bytes, step3_flag, 2, vector_length), vector_length),
        __riscv_vslideup_vx_u8m1(zero_bytes, step4_flag, 3, vector_length), vector_length);
    vbool8_t const covered = __riscv_vmsne_vx_u8m1_b8(covered_bytes, 0, vector_length);
    vbool8_t const orphan = __riscv_vmand_mm_b8(
        __riscv_vmandn_mm_b8(__riscv_vmand_mm_b8(continuations, within_decodable, vector_length), covered,
                             vector_length),
        within_decodable, vector_length);

    // Emitted starts: every decodable true start plus every promoted orphan. Ill-formed = emitted minus well-formed
    // (orphans are continuations, never well-formed).
    vbool8_t const emit_starts = __riscv_vmand_mm_b8(
        __riscv_vmor_mm_b8(__riscv_vmand_mm_b8(starts, within_decodable, vector_length), orphan, vector_length),
        within_decodable, vector_length);
    sz_size_t const emit_count = (sz_size_t)__riscv_vcpop_m_b8(emit_starts, vector_length);
    if (emit_count == 0) { return *runes_unpacked = 0, text; } // Nothing decodable -> window-edge finalize in driver.
    vbool8_t const ill_formed = __riscv_vmandn_mm_b8(emit_starts, well_formed, vector_length);

    // Per-lane maximal-subpart length = 1 + step2 + step3 + step4 (added at the LEAD lane). For an orphan/bad lead this
    // stays 1; for a partial 3/4-byte subpart it is 2 or 3; for a well-formed lane it equals the declared rune length.
    vuint8m1_t consumed_length = __riscv_vmv_v_x_u8m1(1, vector_length);
    consumed_length = __riscv_vadd_vx_u8m1_mu(step2, consumed_length, consumed_length, 1, vector_length);
    consumed_length = __riscv_vadd_vx_u8m1_mu(step3, consumed_length, consumed_length, 1, vector_length);
    consumed_length = __riscv_vadd_vx_u8m1_mu(step4, consumed_length, consumed_length, 1, vector_length);

    sz_size_t consumed = 0;
    sz_size_t const produced = sz_utf8_rune_drain_rvv_(window_bytes, emit_starts, ill_formed, consumed_length,
                                                       emit_count, vector_length, runes, runes_capacity, &consumed);
    *runes_unpacked = produced;
    return text + consumed;
}

/**
 *  @brief  Decode UTF-8 to UTF-32 codepoints, byte-for-byte equivalent to `sz_utf8_decode_serial`.
 *
 *  Vector-classifies the maximal ASCII prefix of each strip (`vmsgtu` high bit, `vfirst`) and widens it straight
 *  into the rune output; non-ASCII regions are decoded a whole `e8m1` window at a time via the uniform TOTAL
 *  classify/gate/compress/gather/width-blend path (@ref sz_utf8_decode_once_rvv_) - clean and dirty bytes alike
 *  handled in-vector, one U+FFFD per maximal ill-formed subpart. The step declines (`step_unpacked == 0`) only when the
 *  very first lead declares a sequence crossing the window edge (a boundary truncation), which the public entry
 *  finalizes with a single bounded `sz_utf8_maximal_subpart_` step - never a per-codepoint serial re-decode.
 */
SZ_PUBLIC sz_cptr_t sz_utf8_decode_rvv(         //
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
            if (take) sz_utf8_decode_ascii_run_rvv_(runes + runes_written + ascii_run, text_cursor + ascii_run, take);
            ascii_run += take;
            if (first_match_index >= 0) break; // Hit a non-ASCII byte.
        }
        runes_written += ascii_run;
        text_cursor += ascii_run;
        if (runes_written >= runes_capacity || text_cursor >= text_end) break;

        // 2-byte fast lane: when the next lead opens a well-formed 2-byte sequence (`0xC2..0xDF`), `vlseg2e8`
        // deinterleaves the lead/continuation streams and decodes the maximal such run with zero gather. Whatever ends
        // the run (a non-2-byte lead, a malformed pair, or an odd trailing byte) is left to the general window path.
        if (*text_cursor >= 0xC2 && *text_cursor <= 0xDF) {
            sz_size_t two_byte_consumed = 0;
            sz_size_t two_byte_runes = sz_utf8_decode_two_byte_run_rvv_(
                (sz_cptr_t)text_cursor, (sz_size_t)(text_end - text_cursor), runes + runes_written,
                runes_capacity - runes_written, &two_byte_consumed);
            if (two_byte_runes) {
                runes_written += two_byte_runes;
                text_cursor += two_byte_consumed;
                continue;
            }
        }

        // Now `*text_cursor` is a multi-byte lead (or a stray continuation). Decode a whole window in-register; the
        // step declines an ill-formed or truncated-only window, in which case the serial reference guarantees progress.
        sz_size_t step_unpacked = 0;
        sz_cptr_t next = sz_utf8_decode_once_rvv_((sz_cptr_t)text_cursor, (sz_size_t)(text_end - text_cursor),
                                                  runes + runes_written, runes_capacity - runes_written,
                                                  &step_unpacked);
        if (step_unpacked) {
            runes_written += step_unpacked;
            text_cursor = (sz_u8_t const *)next;
            continue;
        }

        // The in-vector step decodes its whole decodable span; `step_unpacked == 0` only when the very first lead
        // declares a sequence crossing the window edge (a boundary truncation). A resumable truncation breaks and awaits
        // more bytes; a bad/overlong truncated lead at the edge finalizes to one U+FFFD over its maximal ill-formed
        // subpart - a bounded <=3-byte finalize, never a per-codepoint serial re-decode.
        if (sz_utf8_incomplete_tail_((sz_cptr_t)text_cursor, (sz_cptr_t)text_end)) break;
        runes[runes_written++] = (sz_rune_t)sz_rune_replacement_k;
        text_cursor += sz_utf8_maximal_subpart_((sz_cptr_t)text_cursor, (sz_cptr_t)text_end);
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

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_RVV

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_RUNES_RVV_H_
