/**
 *  @brief RISC-V Vector backend for UTF-8 case folding (delegates to serial).
 *  @file include/stringzilla/utf8_uncased_fold/rvv.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased_fold.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_FOLD_RVV_H_
#define STRINGZILLA_UTF8_UNCASED_FOLD_RVV_H_

#include "stringzilla/utf8_uncased_fold/serial.h"

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

/*  Per-codepoint deltas for 2-byte Latin Extended sequences, indexed by the continuation byte's low 6 bits:
 *  0x00 = identity, 0x01 = fold by +1, 0x80 = irregular (route to serial). Identical values to the NEON /
 *  Ice Lake `cN_deltas_lut`, in ascending index order. Generated from Unicode full case folding; verified
 *  byte-exact against the serial reference.
 *
 *  The C4 / C5 / C6 sub-tables are laid out contiguously in one 192-byte table so a SINGLE indexed memory
 *  load (`vluxei8`) keyed by `family_base + low6` (family_base = 0 / 64 / 128 for C4 / C5 / C6) covers all
 *  three families in one gather — the fold handlers below exploit this to run a strip at `e8m8`. The three
 *  `sz_utf8_fold_latin_c{4,5,6}_deltas_rvv_` names alias the matching 64-byte windows so the
 *  `utf8_uncased` strips can keep loading a single family with one `vle8`. */
static sz_u8_t const sz_utf8_fold_latin_c456_deltas_rvv_[192] = {
    // C4 80-BF
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,       //
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,       //
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,       //
    0x80, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0x80, //
    // C5 80-BF
    0, 1, 0, 1, 0, 1, 0, 1, 0, 0x80, 1, 0, 1, 0, 1, 0,    //
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,       //
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,       //
    1, 0, 1, 0, 1, 0, 1, 0, 0x80, 1, 0, 1, 0, 1, 0, 0x80, //
    // C6 80-BF
    0, 0x80, 1, 0, 1, 0, 0x80, 1, 0, 0x80, 0x80, 1, 0, 0, 0x80, 0x80,       //
    0x80, 1, 0, 0x80, 0x80, 0, 0x80, 0x80, 1, 0, 0, 0, 0x80, 0x80, 0, 0x80, //
    1, 0, 1, 0, 1, 0, 0x80, 1, 0, 0x80, 0, 0, 1, 0, 0x80, 1,                //
    0, 0x80, 0x80, 1, 0, 1, 0, 0x80, 1, 0, 0, 0, 1, 0, 0, 0};
static sz_u8_t const *const sz_utf8_fold_latin_c4_deltas_rvv_ = sz_utf8_fold_latin_c456_deltas_rvv_ + 0;
static sz_u8_t const *const sz_utf8_fold_latin_c5_deltas_rvv_ = sz_utf8_fold_latin_c456_deltas_rvv_ + 64;
static sz_u8_t const *const sz_utf8_fold_latin_c6_deltas_rvv_ = sz_utf8_fold_latin_c456_deltas_rvv_ + 128;

/*  Case-fold a strip of ASCII bytes: `c + ((c - 'A' <= 25) * 0x20)`, the vector form of `sz_ascii_fold_`.
 *  Only `[0x41, 0x5A]` shifts by `+0x20`; every other lane passes through unchanged. */
SZ_HELPER_INLINE vuint8m8_t sz_utf8_fold_ascii_rvv_(vuint8m8_t source_u8m8, sz_size_t vector_length) {
    vbool1_t is_upper_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 'A', vector_length), 25,
                                                     vector_length);
    vuint8m8_t lowered_u8m8 = __riscv_vadd_vx_u8m8(source_u8m8, 0x20, vector_length);
    return __riscv_vmerge_vvm_u8m8(source_u8m8, lowered_u8m8, is_upper_b1, vector_length);
}

/*  Largest strip length that does not split a trailing multi-byte sequence across strips. On the final strip
 *  (`vector_length == remaining`) the whole input ends here, so nothing is trimmed; otherwise a last codepoint whose
 *  declared length runs past `vector_length` is excluded and reprocessed in the next strip. */
SZ_HELPER_INLINE sz_size_t sz_utf8_fold_trim_incomplete_(sz_u8_t const *source_ptr, sz_size_t vector_length,
                                                         sz_size_t remaining) {
    if (vector_length >= remaining) return vector_length;
    sz_size_t boundary = vector_length;
    while (boundary && (source_ptr[boundary - 1] & 0xC0) == 0x80) --boundary; // back up to the last lead
    if (!boundary) return vector_length;
    sz_u8_t lead = source_ptr[boundary - 1];
    sz_size_t needed = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
    return ((boundary - 1) + needed > vector_length) ? (boundary - 1) : vector_length;
}

/*  Vector-fold one strip of Latin text (ASCII + Latin-1 Supplement C2/C3 + Latin Extended-A/B C4-C6) in
 *  place — the length-preserving working set of most Latin-script languages. Mirrors the NEON latin chunk:
 *  Latin-1 uppercase is `+0x20`, Latin Extended is a `+0` / `+1` parity delta from the `cN` LUTs, and the
 *  length-preserving special cases ß→"ss" and µ→μ are byte replacements. Continuation deltas flagged `0x80`
 *  (expand / shrink / cross-block) and any lead outside C2-C6 are "stops": the consumed prefix ends on the
 *  character boundary before the first stop, and the caller routes that one codepoint to serial.
 *
 *  Runs at `e8m8`: the per-lane delta is fetched from the contiguous C4/C5/C6 table in memory with a single
 *  `vluxei8` keyed by `family_base + low6`, so the LUT no longer has to share a register group with the
 *  indices (which is what pinned the gather-based form to `e8m4`). Sets `*needs_serial` when it stopped on a
 *  non-handled codepoint (vs. merely trimming a trailing incomplete sequence). Returns the number of bytes
 *  folded and written. */
SZ_HELPER_INLINE sz_size_t sz_utf8_fold_latin_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t remaining,
                                                         sz_u8_t *destination_ptr, int *needs_serial) {
    sz_size_t vector_length = __riscv_vsetvl_e8m8(remaining);
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);
    sz_u8_t next_carry = (vector_length < remaining) ? source_ptr[vector_length] : 0;
    vuint8m8_t next_u8m8 = __riscv_vslide1down_vx_u8m8(source_u8m8, next_carry, vector_length);

    vbool1_t is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(source_u8m8, 0xC0, vector_length), 0x80,
                                                           vector_length);
    vbool1_t after_c2_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xC2, vector_length);
    vbool1_t after_c3_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xC3, vector_length);
    // After a C4/C5/C6 lead: the previous byte is in [0xC4, 0xC6]; the family base is `(prev - 0xC4) * 64`.
    vbool1_t after_c456_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(previous_u8m8, 0xC4, vector_length), 2,
                                                       vector_length);

    // Latin Extended-A/B continuation deltas: one indexed load over the combined C4/C5/C6 table, keyed by
    // `family_base + low6`, applied only on continuation bytes that follow a C4/C5/C6 lead.
    vbool1_t delta_lanes_b1 = __riscv_vmand_mm_b1(after_c456_b1, is_continuation_b1, vector_length);
    vuint8m8_t low6_u8m8 = __riscv_vand_vx_u8m8(source_u8m8, 0x3F, vector_length);
    vuint8m8_t family_base_u8m8 = __riscv_vmul_vx_u8m8(__riscv_vsub_vx_u8m8(previous_u8m8, 0xC4, vector_length), 64,
                                                       vector_length);
    vuint8m8_t lut_index_u8m8 = __riscv_vadd_vv_u8m8(family_base_u8m8, low6_u8m8, vector_length);
    vuint8m8_t delta_u8m8 = __riscv_vmv_v_x_u8m8(0, vector_length);
    delta_u8m8 = __riscv_vluxei8_v_u8m8_mu(delta_lanes_b1, delta_u8m8, sz_utf8_fold_latin_c456_deltas_rvv_,
                                           lut_index_u8m8, vector_length);
    vbool1_t is_irregular_b1 = __riscv_vmsne_vx_u8m8_b1(__riscv_vand_vx_u8m8(delta_u8m8, 0x80, vector_length), 0,
                                                        vector_length);

    // Build the folded strip: ASCII A-Z, Latin-1 upper +0x20, ß/µ replacements, then the Extended delta.
    vbool1_t is_upper_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 'A', vector_length), 25,
                                                     vector_length);
    vuint8m8_t folded_u8m8 = __riscv_vmerge_vvm_u8m8(
        source_u8m8, __riscv_vadd_vx_u8m8(source_u8m8, 0x20, vector_length), is_upper_b1, vector_length);
    // Latin-1 Supplement 'À'-'Þ' (C3 80-9E, excluding '×' at 0x97) get +0x20.
    vbool1_t latin1_range_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 0x80, vector_length), 0x1F,
                                                         vector_length);
    vbool1_t is_latin1_upper_b1 = __riscv_vmandn_mm_b1(__riscv_vmand_mm_b1(after_c3_b1, latin1_range_b1, vector_length),
                                                       __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0x97, vector_length),
                                                       vector_length);
    folded_u8m8 = __riscv_vmerge_vvm_u8m8(folded_u8m8, __riscv_vadd_vx_u8m8(folded_u8m8, 0x20, vector_length),
                                          is_latin1_upper_b1, vector_length);
    // ß (C3 9F) -> "ss": both bytes become 's'.
    vbool1_t eszett_lead_b1 = __riscv_vmand_mm_b1(__riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xC3, vector_length),
                                                  __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0x9F, vector_length),
                                                  vector_length);
    vbool1_t eszett_second_b1 = __riscv_vmand_mm_b1(
        after_c3_b1, __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0x9F, vector_length), vector_length);
    folded_u8m8 = __riscv_vmerge_vxm_u8m8(
        folded_u8m8, 's', __riscv_vmor_mm_b1(eszett_lead_b1, eszett_second_b1, vector_length), vector_length);
    // µ (C2 B5) -> μ (CE BC): lead byte becomes 0xCE, continuation 0xBC.
    folded_u8m8 = __riscv_vmerge_vxm_u8m8(
        folded_u8m8, 0xCE,
        __riscv_vmand_mm_b1(__riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xC2, vector_length),
                            __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0xB5, vector_length), vector_length),
        vector_length);
    folded_u8m8 = __riscv_vmerge_vxm_u8m8(
        folded_u8m8, 0xBC,
        __riscv_vmand_mm_b1(after_c2_b1, __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xB5, vector_length), vector_length),
        vector_length);
    // Latin Extended +1 parity delta (0x80 irregulars corrupt their lane but are trimmed off below).
    folded_u8m8 = __riscv_vadd_vv_u8m8(folded_u8m8, delta_u8m8, vector_length);

    // Stops: any lead outside the C2-C6 family, a malformed family lead, or an irregular continuation.
    vbool1_t is_non_ascii_b1 = __riscv_vmsgtu_vx_u8m8_b1(source_u8m8, 0x7F, vector_length);
    vbool1_t is_lead_b1 = __riscv_vmandn_mm_b1(is_non_ascii_b1, is_continuation_b1, vector_length);
    vbool1_t is_family_lead_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 0xC2, vector_length), 4,
                                                           vector_length);
    vbool1_t is_foreign_lead_b1 = __riscv_vmandn_mm_b1(is_lead_b1, is_family_lead_b1, vector_length);
    // Well-formed C2-C6 lead (mirrors `sz_rune_decode`): the next byte must be a continuation. C2-C6 are
    // all 2-byte leads with no overlong/surrogate special case, so the continuation test is the whole gate; a
    // malformed family lead is treated as foreign so the strip stops before it and serial copies one byte.
    vbool1_t next_is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(next_u8m8, 0xC0, vector_length),
                                                                0x80, vector_length);
    vbool1_t malformed_family_lead_b1 = __riscv_vmandn_mm_b1(is_family_lead_b1, next_is_continuation_b1, vector_length);
    vbool1_t stop_b1 = __riscv_vmor_mm_b1(
        __riscv_vmor_mm_b1(is_foreign_lead_b1, malformed_family_lead_b1, vector_length), is_irregular_b1,
        vector_length);

    long first_stop = __riscv_vfirst_m_b1(stop_b1, vector_length);
    sz_size_t consumed;
    if (first_stop >= 0) {
        consumed = (sz_size_t)first_stop;
        while (consumed && (source_ptr[consumed] & 0xC0) == 0x80) --consumed; // back up to the lead
        *needs_serial = 1;
    }
    else {
        consumed = vector_length;
        // Don't split a trailing incomplete sequence: a lead in the last byte belongs to the next strip.
        if (vector_length < remaining && source_ptr[vector_length - 1] >= 0xC0) --consumed;
        *needs_serial = 0;
    }
    if (consumed) __riscv_vse8_v_u8m8(destination_ptr, folded_u8m8, consumed);
    return consumed;
}

/*  Vector-fold one strip of basic Cyrillic (D0/D1 leads) in place. Uppercase sub-ranges are keyed by the
 *  second byte's high nibble — one indexed memory load (`vluxei8`) over a 16-entry table yields the offset
 *  (8→+0x10, 9→+0x20, A→−0x20) — and the D0→D1 lead rewrite is a masked `+1` where the lowercase lives in
 *  the next block. Cyrillic Extended-A (D1 A0+) and any non-D0/D1 lead are stops routed to serial.
 *
 *  Runs at `e8m8`: the offset table is read from memory, so it no longer has to share a register group with
 *  the indices the way a `vrgather` would, lifting the old `e8m4` ceiling. Same stop-and-serial contract as
 *  `sz_utf8_fold_latin_strip_rvv_`. */
SZ_HELPER_INLINE sz_size_t sz_utf8_fold_cyrillic_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t remaining,
                                                            sz_u8_t *destination_ptr, int *needs_serial) {
    static sz_u8_t const second_byte_offsets[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x20, 0xE0, 0, 0, 0, 0, 0};
    sz_size_t vector_length = __riscv_vsetvl_e8m8(remaining);
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    sz_u8_t next_carry = (vector_length < remaining) ? source_ptr[vector_length] : 0;
    vuint8m8_t next_u8m8 = __riscv_vslide1down_vx_u8m8(source_u8m8, next_carry, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);

    vbool1_t is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(source_u8m8, 0xC0, vector_length), 0x80,
                                                           vector_length);
    vbool1_t is_d0_b1 = __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xD0, vector_length);
    vbool1_t is_d1_b1 = __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xD1, vector_length);
    vbool1_t after_d0_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xD0, vector_length);
    vbool1_t is_lead_b1 = __riscv_vmandn_mm_b1(__riscv_vmsgtu_vx_u8m8_b1(source_u8m8, 0x7F, vector_length),
                                               is_continuation_b1, vector_length);
    vbool1_t is_family_lead_b1 = __riscv_vmor_mm_b1(is_d0_b1, is_d1_b1, vector_length);
    vbool1_t is_foreign_lead_b1 = __riscv_vmandn_mm_b1(is_lead_b1, is_family_lead_b1, vector_length);
    // Well-formed D0/D1 lead (mirrors `sz_rune_decode`): the next byte must be a continuation. D0/D1 are
    // 2-byte leads with no overlong/surrogate special case, so a malformed family lead is foreign and the strip
    // stops before it for serial to copy one byte.
    vbool1_t next_is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(next_u8m8, 0xC0, vector_length),
                                                                0x80, vector_length);
    vbool1_t malformed_family_lead_b1 = __riscv_vmandn_mm_b1(is_family_lead_b1, next_is_continuation_b1, vector_length);
    // Cyrillic Extended-A (D1 A0+) folds by parity across blocks — leave it to serial.
    vbool1_t is_extended_b1 = __riscv_vmand_mm_b1(is_d1_b1, __riscv_vmsgeu_vx_u8m8_b1(next_u8m8, 0xA0, vector_length),
                                                  vector_length);
    vbool1_t stop_b1 = __riscv_vmor_mm_b1(
        __riscv_vmor_mm_b1(is_foreign_lead_b1, malformed_family_lead_b1, vector_length), is_extended_b1, vector_length);

    // Second-byte offset by high nibble, applied only after a D0 lead.
    vuint8m8_t offset_u8m8 = __riscv_vmv_v_x_u8m8(0, vector_length);
    offset_u8m8 = __riscv_vluxei8_v_u8m8_mu(after_d0_b1, offset_u8m8, second_byte_offsets,
                                            __riscv_vsrl_vx_u8m8(source_u8m8, 4, vector_length), vector_length);
    vbool1_t is_upper_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 'A', vector_length), 25,
                                                     vector_length);
    vuint8m8_t folded_u8m8 = __riscv_vmerge_vvm_u8m8(
        source_u8m8, __riscv_vadd_vx_u8m8(source_u8m8, 0x20, vector_length), is_upper_b1, vector_length);
    folded_u8m8 = __riscv_vadd_vv_u8m8(folded_u8m8, offset_u8m8, vector_length);
    // Lead rewrite D0 -> D1 (+1) where the next byte is 80-8F or A0-AF.
    vbool1_t next_80_8f_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(next_u8m8, 0x80, vector_length), 0x10,
                                                       vector_length);
    vbool1_t next_a0_af_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(next_u8m8, 0xA0, vector_length), 0x10,
                                                       vector_length);
    vbool1_t needs_d1_b1 = __riscv_vmand_mm_b1(
        is_d0_b1, __riscv_vmor_mm_b1(next_80_8f_b1, next_a0_af_b1, vector_length), vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(needs_d1_b1, folded_u8m8, folded_u8m8, 1, vector_length);

    long first_stop = __riscv_vfirst_m_b1(stop_b1, vector_length);
    sz_size_t consumed;
    if (first_stop >= 0) {
        consumed = (sz_size_t)first_stop;
        while (consumed && (source_ptr[consumed] & 0xC0) == 0x80) --consumed;
        *needs_serial = 1;
    }
    else {
        consumed = sz_utf8_fold_trim_incomplete_(source_ptr, vector_length, remaining);
        *needs_serial = 0;
    }
    if (consumed) __riscv_vse8_v_u8m8(destination_ptr, folded_u8m8, consumed);
    return consumed;
}

/*  Vector-fold one strip of basic Greek (CE/CF leads) in place. 'Α'-'Ο' (CE 91-9F) fold `+0x20`; 'Π'-'Ρ'
 *  and 'Σ'-'Ϋ' fold `-0x20` with a CE->CF lead promotion (+1); final sigma 'ς' (CF 82) folds to 'σ' (+1).
 *  Accented uppercase (CE 84-90), the expanding 'ΰ' (CE B0), the CF 8F+ symbols, and any non-CE/CF lead are
 *  stops routed to serial. Same `e8m8` strip / stop-and-serial contract as the other handlers. */
SZ_HELPER_INLINE sz_size_t sz_utf8_fold_greek_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t remaining,
                                                         sz_u8_t *destination_ptr, int *needs_serial) {
    sz_size_t vector_length = __riscv_vsetvl_e8m8(remaining);
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    sz_u8_t next_carry = (vector_length < remaining) ? source_ptr[vector_length] : 0;
    vuint8m8_t next_u8m8 = __riscv_vslide1down_vx_u8m8(source_u8m8, next_carry, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);

    vbool1_t is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(source_u8m8, 0xC0, vector_length), 0x80,
                                                           vector_length);
    vbool1_t is_ce_b1 = __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xCE, vector_length);
    vbool1_t is_cf_b1 = __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xCF, vector_length);
    vbool1_t after_ce_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xCE, vector_length);
    vbool1_t after_cf_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xCF, vector_length);
    vbool1_t is_lead_b1 = __riscv_vmandn_mm_b1(__riscv_vmsgtu_vx_u8m8_b1(source_u8m8, 0x7F, vector_length),
                                               is_continuation_b1, vector_length);
    vbool1_t is_family_lead_b1 = __riscv_vmor_mm_b1(is_ce_b1, is_cf_b1, vector_length);
    vbool1_t is_foreign_lead_b1 = __riscv_vmandn_mm_b1(is_lead_b1, is_family_lead_b1, vector_length);
    // Well-formed CE/CF lead (mirrors `sz_rune_decode`): the next byte must be a continuation. CE/CF are
    // 2-byte leads with no overlong/surrogate special case, so a malformed family lead is foreign and the strip
    // stops before it for serial to copy one byte.
    vbool1_t next_is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(next_u8m8, 0xC0, vector_length),
                                                                0x80, vector_length);
    vbool1_t malformed_family_lead_b1 = __riscv_vmandn_mm_b1(is_family_lead_b1, next_is_continuation_b1, vector_length);
    // CE excluded: next < 0x91 (accented uppercase) or next == 0xB0 ('ΰ' expands). CF excluded: next >= 0x8F.
    vbool1_t ce_excluded_b1 = __riscv_vmand_mm_b1(
        is_ce_b1,
        __riscv_vmor_mm_b1(__riscv_vmsltu_vx_u8m8_b1(next_u8m8, 0x91, vector_length),
                           __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0xB0, vector_length), vector_length),
        vector_length);
    vbool1_t cf_excluded_b1 = __riscv_vmand_mm_b1(is_cf_b1, __riscv_vmsgeu_vx_u8m8_b1(next_u8m8, 0x8F, vector_length),
                                                  vector_length);
    vbool1_t stop_b1 = __riscv_vmor_mm_b1(
        __riscv_vmor_mm_b1(is_foreign_lead_b1, malformed_family_lead_b1, vector_length),
        __riscv_vmor_mm_b1(ce_excluded_b1, cf_excluded_b1, vector_length), vector_length);

    // Promoting ranges (second byte A0-A1 or A3-AB), used both for the second-byte -0x20 and the lead +1.
    vbool1_t in_promote_a0_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 0xA0, vector_length), 0x02,
                                                          vector_length);
    vbool1_t in_promote_a3_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 0xA3, vector_length), 0x09,
                                                          vector_length);
    vbool1_t next_promote_a0_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(next_u8m8, 0xA0, vector_length), 0x02,
                                                            vector_length);
    vbool1_t next_promote_a3_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(next_u8m8, 0xA3, vector_length), 0x09,
                                                            vector_length);

    vbool1_t is_upper_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 'A', vector_length), 25,
                                                     vector_length);
    vuint8m8_t folded_u8m8 = __riscv_vmerge_vvm_u8m8(
        source_u8m8, __riscv_vadd_vx_u8m8(source_u8m8, 0x20, vector_length), is_upper_b1, vector_length);
    vbool1_t basic_upper_b1 = __riscv_vmand_mm_b1(
        after_ce_b1,
        __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 0x91, vector_length), 0x0F, vector_length),
        vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(basic_upper_b1, folded_u8m8, folded_u8m8, 0x20, vector_length);
    vbool1_t promoting_upper_b1 = __riscv_vmand_mm_b1(
        after_ce_b1, __riscv_vmor_mm_b1(in_promote_a0_b1, in_promote_a3_b1, vector_length), vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(promoting_upper_b1, folded_u8m8, folded_u8m8, 0xE0, vector_length); // -0x20
    vbool1_t final_sigma_b1 = __riscv_vmand_mm_b1(
        after_cf_b1, __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0x82, vector_length), vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(final_sigma_b1, folded_u8m8, folded_u8m8, 0x01, vector_length);
    vbool1_t promotes_lead_b1 = __riscv_vmand_mm_b1(
        is_ce_b1, __riscv_vmor_mm_b1(next_promote_a0_b1, next_promote_a3_b1, vector_length), vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(promotes_lead_b1, folded_u8m8, folded_u8m8, 0x01, vector_length); // CE -> CF

    long first_stop = __riscv_vfirst_m_b1(stop_b1, vector_length);
    sz_size_t consumed;
    if (first_stop >= 0) {
        consumed = (sz_size_t)first_stop;
        while (consumed && (source_ptr[consumed] & 0xC0) == 0x80) --consumed;
        *needs_serial = 1;
    }
    else {
        consumed = sz_utf8_fold_trim_incomplete_(source_ptr, vector_length, remaining);
        *needs_serial = 0;
    }
    if (consumed) __riscv_vse8_v_u8m8(destination_ptr, folded_u8m8, consumed);
    return consumed;
}

/*  Vector-fold one strip of Armenian (D4/D5/D6 leads) in place. Disjoint second-byte offsets — D4 B1-BF and
 *  D5 90-96 fold `-0x10`, D5 80-8F folds `+0x30` — plus the lead `+1` rewrites D4->D5 (next B1-BF) and
 *  D5->D6 (next 90-96). The 'և' ligature (D6 87, expands), the D4 Cyrillic-Supplement range (next < B1), and
 *  any non-D4/D5/D6 lead are stops routed to serial. Same `e8m8` strip / stop-and-serial contract. */
SZ_HELPER_INLINE sz_size_t sz_utf8_fold_armenian_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t remaining,
                                                            sz_u8_t *destination_ptr, int *needs_serial) {
    sz_size_t vector_length = __riscv_vsetvl_e8m8(remaining);
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    sz_u8_t next_carry = (vector_length < remaining) ? source_ptr[vector_length] : 0;
    vuint8m8_t next_u8m8 = __riscv_vslide1down_vx_u8m8(source_u8m8, next_carry, vector_length);
    vuint8m8_t previous_u8m8 = __riscv_vslide1up_vx_u8m8(source_u8m8, 0, vector_length);

    vbool1_t is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(source_u8m8, 0xC0, vector_length), 0x80,
                                                           vector_length);
    vbool1_t is_d4_b1 = __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xD4, vector_length);
    vbool1_t is_d5_b1 = __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xD5, vector_length);
    vbool1_t is_d6_b1 = __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xD6, vector_length);
    vbool1_t after_d4_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xD4, vector_length);
    vbool1_t after_d5_b1 = __riscv_vmseq_vx_u8m8_b1(previous_u8m8, 0xD5, vector_length);
    vbool1_t is_lead_b1 = __riscv_vmandn_mm_b1(__riscv_vmsgtu_vx_u8m8_b1(source_u8m8, 0x7F, vector_length),
                                               is_continuation_b1, vector_length);
    vbool1_t is_family_b1 = __riscv_vmor_mm_b1(__riscv_vmor_mm_b1(is_d4_b1, is_d5_b1, vector_length), is_d6_b1,
                                               vector_length);
    vbool1_t is_foreign_lead_b1 = __riscv_vmandn_mm_b1(is_lead_b1, is_family_b1, vector_length);
    // Well-formed D4/D5/D6 lead (mirrors `sz_rune_decode`): the next byte must be a continuation. D4-D6 are
    // 2-byte leads with no overlong/surrogate special case, so a malformed family lead is foreign and the strip
    // stops before it for serial to copy one byte.
    vbool1_t next_is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(next_u8m8, 0xC0, vector_length),
                                                                0x80, vector_length);
    vbool1_t malformed_family_lead_b1 = __riscv_vmandn_mm_b1(is_family_b1, next_is_continuation_b1, vector_length);
    vbool1_t d4_stop_b1 = __riscv_vmand_mm_b1(is_d4_b1, __riscv_vmsltu_vx_u8m8_b1(next_u8m8, 0xB1, vector_length),
                                              vector_length);
    vbool1_t ligature_stop_b1 = __riscv_vmand_mm_b1(is_d6_b1, __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0x87, vector_length),
                                                    vector_length);
    vbool1_t stop_b1 = __riscv_vmor_mm_b1(
        __riscv_vmor_mm_b1(is_foreign_lead_b1, malformed_family_lead_b1, vector_length),
        __riscv_vmor_mm_b1(d4_stop_b1, ligature_stop_b1, vector_length), vector_length);

    vbool1_t is_upper_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 'A', vector_length), 25,
                                                     vector_length);
    vuint8m8_t folded_u8m8 = __riscv_vmerge_vvm_u8m8(
        source_u8m8, __riscv_vadd_vx_u8m8(source_u8m8, 0x20, vector_length), is_upper_b1, vector_length);
    // Second-byte offsets (disjoint lanes).
    vbool1_t d4_second_b1 = __riscv_vmand_mm_b1(
        after_d4_b1, __riscv_vmsgeu_vx_u8m8_b1(source_u8m8, 0xB1, vector_length), vector_length);
    vbool1_t d5_plus30_b1 = __riscv_vmand_mm_b1(
        after_d5_b1,
        __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 0x80, vector_length), 0x10, vector_length),
        vector_length);
    vbool1_t d5_minus10_b1 = __riscv_vmand_mm_b1(
        after_d5_b1,
        __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 0x90, vector_length), 0x07, vector_length),
        vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(__riscv_vmor_mm_b1(d4_second_b1, d5_minus10_b1, vector_length), folded_u8m8,
                                          folded_u8m8, 0xF0, vector_length); // -0x10
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(d5_plus30_b1, folded_u8m8, folded_u8m8, 0x30, vector_length);
    // Lead +1 rewrites D4->D5 and D5->D6.
    vbool1_t promotes_d4_b1 = __riscv_vmand_mm_b1(is_d4_b1, __riscv_vmsgeu_vx_u8m8_b1(next_u8m8, 0xB1, vector_length),
                                                  vector_length);
    vbool1_t promotes_d5_b1 = __riscv_vmand_mm_b1(
        is_d5_b1, __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(next_u8m8, 0x90, vector_length), 0x07, vector_length),
        vector_length);
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(__riscv_vmor_mm_b1(promotes_d4_b1, promotes_d5_b1, vector_length),
                                          folded_u8m8, folded_u8m8, 0x01, vector_length);

    long first_stop = __riscv_vfirst_m_b1(stop_b1, vector_length);
    sz_size_t consumed;
    if (first_stop >= 0) {
        consumed = (sz_size_t)first_stop;
        while (consumed && (source_ptr[consumed] & 0xC0) == 0x80) --consumed;
        *needs_serial = 1;
    }
    else {
        consumed = sz_utf8_fold_trim_incomplete_(source_ptr, vector_length, remaining);
        *needs_serial = 0;
    }
    if (consumed) __riscv_vse8_v_u8m8(destination_ptr, folded_u8m8, consumed);
    return consumed;
}

/*  Vector-fold one strip of Georgian (3-byte E1 82/83 sequences) in place — a uniform 3-byte->3-byte fold
 *  whose uppercase subset is decided by the THIRD byte, so the lead reads two bytes forward and the
 *  uppercase flag is carried one lane to the second-byte rewrite and two lanes to the third-byte offset:
 *  lead E1->E2, second 82/83->B4, third -0x20 (E1 82) or +0x20 (E1 83). Non-Georgian E1 second bytes and
 *  any non-E1 lead are stops routed to serial. Same `e8m8` strip / stop-and-serial contract. */
SZ_HELPER_INLINE sz_size_t sz_utf8_fold_georgian_strip_rvv_(sz_u8_t const *source_ptr, sz_size_t remaining,
                                                            sz_u8_t *destination_ptr, int *needs_serial) {
    sz_size_t vector_length = __riscv_vsetvl_e8m8(remaining);
    vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
    sz_u8_t carry1 = (vector_length < remaining) ? source_ptr[vector_length] : 0;
    sz_u8_t carry2 = (vector_length + 1 < remaining) ? source_ptr[vector_length + 1] : 0;
    vuint8m8_t next_u8m8 = __riscv_vslide1down_vx_u8m8(source_u8m8, carry1, vector_length);
    vuint8m8_t next_next_u8m8 = __riscv_vslide1down_vx_u8m8(next_u8m8, carry2, vector_length);

    vbool1_t is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(__riscv_vand_vx_u8m8(source_u8m8, 0xC0, vector_length), 0x80,
                                                           vector_length);
    vbool1_t is_e1_b1 = __riscv_vmseq_vx_u8m8_b1(source_u8m8, 0xE1, vector_length);
    vbool1_t is_lead_b1 = __riscv_vmandn_mm_b1(__riscv_vmsgtu_vx_u8m8_b1(source_u8m8, 0x7F, vector_length),
                                               is_continuation_b1, vector_length);
    vbool1_t is_foreign_lead_b1 = __riscv_vmandn_mm_b1(is_lead_b1, is_e1_b1, vector_length);
    vbool1_t is_82_lead_b1 = __riscv_vmand_mm_b1(is_e1_b1, __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0x82, vector_length),
                                                 vector_length);
    vbool1_t is_83_lead_b1 = __riscv_vmand_mm_b1(is_e1_b1, __riscv_vmseq_vx_u8m8_b1(next_u8m8, 0x83, vector_length),
                                                 vector_length);
    vbool1_t is_foreign_e1_b1 = __riscv_vmandn_mm_b1(
        is_e1_b1, __riscv_vmor_mm_b1(is_82_lead_b1, is_83_lead_b1, vector_length), vector_length);
    // Well-formed E1 82/83 lead (mirrors `sz_rune_decode`): the second byte (82/83) is already a
    // continuation, so the gate is that the THIRD byte is a continuation too. E1 has no overlong/surrogate
    // special case. A malformed E1 family lead is treated as foreign so the strip stops before it and serial
    // copies one byte.
    vbool1_t third_is_continuation_b1 = __riscv_vmseq_vx_u8m8_b1(
        __riscv_vand_vx_u8m8(next_next_u8m8, 0xC0, vector_length), 0x80, vector_length);
    vbool1_t malformed_family_lead_b1 = __riscv_vmandn_mm_b1(
        __riscv_vmor_mm_b1(is_82_lead_b1, is_83_lead_b1, vector_length), third_is_continuation_b1, vector_length);
    vbool1_t stop_b1 = __riscv_vmor_mm_b1(__riscv_vmor_mm_b1(is_foreign_lead_b1, is_foreign_e1_b1, vector_length),
                                          malformed_family_lead_b1, vector_length);

    // Uppercase keyed by the third byte: E1 82 third >= A0; E1 83 third in 80-85, or 87, or 8D.
    vbool1_t is_82_upper_lead_b1 = __riscv_vmand_mm_b1(
        is_82_lead_b1, __riscv_vmsgeu_vx_u8m8_b1(next_next_u8m8, 0xA0, vector_length), vector_length);
    vbool1_t third_83_range_b1 = __riscv_vmsltu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(next_next_u8m8, 0x80, vector_length),
                                                           0x06, vector_length);
    vbool1_t third_83_extra_b1 = __riscv_vmor_mm_b1(__riscv_vmseq_vx_u8m8_b1(next_next_u8m8, 0x87, vector_length),
                                                    __riscv_vmseq_vx_u8m8_b1(next_next_u8m8, 0x8D, vector_length),
                                                    vector_length);
    vbool1_t is_83_upper_lead_b1 = __riscv_vmand_mm_b1(
        is_83_lead_b1, __riscv_vmor_mm_b1(third_83_range_b1, third_83_extra_b1, vector_length), vector_length);

    // Carry the per-block uppercase flag forward as a 0/1 byte: lane +1 (second) and lane +2 (third).
    vuint8m8_t flag82_lead_u8m8 = __riscv_vmerge_vxm_u8m8(__riscv_vmv_v_x_u8m8(0, vector_length), 1,
                                                          is_82_upper_lead_b1, vector_length);
    vuint8m8_t flag83_lead_u8m8 = __riscv_vmerge_vxm_u8m8(__riscv_vmv_v_x_u8m8(0, vector_length), 1,
                                                          is_83_upper_lead_b1, vector_length);
    vuint8m8_t flag82_second_u8m8 = __riscv_vslide1up_vx_u8m8(flag82_lead_u8m8, 0, vector_length);
    vuint8m8_t flag83_second_u8m8 = __riscv_vslide1up_vx_u8m8(flag83_lead_u8m8, 0, vector_length);
    vuint8m8_t flag82_third_u8m8 = __riscv_vslide1up_vx_u8m8(flag82_second_u8m8, 0, vector_length);
    vuint8m8_t flag83_third_u8m8 = __riscv_vslide1up_vx_u8m8(flag83_second_u8m8, 0, vector_length);

    vbool1_t is_upper_lead_b1 = __riscv_vmor_mm_b1(is_82_upper_lead_b1, is_83_upper_lead_b1, vector_length);
    vbool1_t is_upper_second_b1 = __riscv_vmsne_vx_u8m8_b1(
        __riscv_vor_vv_u8m8(flag82_second_u8m8, flag83_second_u8m8, vector_length), 0, vector_length);
    vbool1_t is_82_upper_third_b1 = __riscv_vmsne_vx_u8m8_b1(flag82_third_u8m8, 0, vector_length);
    vbool1_t is_83_upper_third_b1 = __riscv_vmsne_vx_u8m8_b1(flag83_third_u8m8, 0, vector_length);

    vbool1_t is_upper_b1 = __riscv_vmsleu_vx_u8m8_b1(__riscv_vsub_vx_u8m8(source_u8m8, 'A', vector_length), 25,
                                                     vector_length);
    vuint8m8_t folded_u8m8 = __riscv_vmerge_vvm_u8m8(
        source_u8m8, __riscv_vadd_vx_u8m8(source_u8m8, 0x20, vector_length), is_upper_b1, vector_length);
    folded_u8m8 = __riscv_vmerge_vxm_u8m8(folded_u8m8, 0xE2, is_upper_lead_b1, vector_length); // lead E1 -> E2
    folded_u8m8 = __riscv_vmerge_vxm_u8m8(folded_u8m8, 0xB4, is_upper_second_b1,
                                          vector_length); // second 82/83 -> B4
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(is_82_upper_third_b1, folded_u8m8, folded_u8m8, 0xE0,
                                          vector_length); // third -0x20
    folded_u8m8 = __riscv_vadd_vx_u8m8_mu(is_83_upper_third_b1, folded_u8m8, folded_u8m8, 0x20,
                                          vector_length); // third +0x20

    long first_stop = __riscv_vfirst_m_b1(stop_b1, vector_length);
    sz_size_t consumed;
    if (first_stop >= 0) {
        consumed = (sz_size_t)first_stop;
        while (consumed && (source_ptr[consumed] & 0xC0) == 0x80) --consumed;
        *needs_serial = 1;
    }
    else {
        consumed = sz_utf8_fold_trim_incomplete_(source_ptr, vector_length, remaining);
        *needs_serial = 0;
    }
    if (consumed) __riscv_vse8_v_u8m8(destination_ptr, folded_u8m8, consumed);
    return consumed;
}

/*  Byte-for-byte equivalent to `sz_utf8_uncased_fold_serial`. Maximal ASCII runs are folded at `e8m8`, Latin /
 *  Cyrillic / Greek / Armenian / Georgian text are folded in place by their `_strip_rvv_` handlers
 *  (dispatched on the lead byte), and any other script / length-changing fold is handled by the value-exact
 *  serial decode/fold/encode for that one codepoint before re-entering the vector path. These folds are 1:1
 *  in length, so the destination tracks the source for those runs. */
SZ_API_COMPTIME sz_size_t sz_utf8_uncased_fold_rvv(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {
    sz_u8_t const *source_ptr = (sz_u8_t const *)source;
    sz_u8_t const *source_end = source_ptr + source_length;
    sz_u8_t *destination_ptr = (sz_u8_t *)destination;

    while (source_ptr < source_end) {
        // Fold a maximal ASCII run in vector strips, stopping at the first non-ASCII byte.
        sz_size_t ascii_bytes = (sz_size_t)(source_end - source_ptr);
        while (ascii_bytes) {
            sz_size_t vector_length = __riscv_vsetvl_e8m8(ascii_bytes);
            vuint8m8_t source_u8m8 = __riscv_vle8_v_u8m8(source_ptr, vector_length);
            long first_non_ascii = __riscv_vfirst_m_b1(__riscv_vmsgtu_vx_u8m8_b1(source_u8m8, 0x7F, vector_length),
                                                       vector_length);
            sz_size_t ascii_run = first_non_ascii < 0 ? vector_length : (sz_size_t)first_non_ascii;
            if (ascii_run)
                __riscv_vse8_v_u8m8(destination_ptr, sz_utf8_fold_ascii_rvv_(source_u8m8, vector_length), ascii_run);
            source_ptr += ascii_run;
            destination_ptr += ascii_run;
            ascii_bytes -= ascii_run;
            if (first_non_ascii >= 0) break; // hit a non-ASCII lead
        }
        if (source_ptr >= source_end) break;

        // Dispatch the non-ASCII region to the matching vectorized script handler by its lead byte.
        int needs_serial = 1;
        sz_size_t consumed = 0;
        sz_size_t remaining = (sz_size_t)(source_end - source_ptr);
        sz_u8_t lead = source_ptr[0];
        if (lead >= 0xC2 && lead <= 0xC6)
            consumed = sz_utf8_fold_latin_strip_rvv_(source_ptr, remaining, destination_ptr, &needs_serial);
        else if (lead == 0xD0 || lead == 0xD1)
            consumed = sz_utf8_fold_cyrillic_strip_rvv_(source_ptr, remaining, destination_ptr, &needs_serial);
        else if (lead == 0xCE || lead == 0xCF)
            consumed = sz_utf8_fold_greek_strip_rvv_(source_ptr, remaining, destination_ptr, &needs_serial);
        else if (lead == 0xD4 || lead == 0xD5 || lead == 0xD6)
            consumed = sz_utf8_fold_armenian_strip_rvv_(source_ptr, remaining, destination_ptr, &needs_serial);
        else if (lead == 0xE1)
            consumed = sz_utf8_fold_georgian_strip_rvv_(source_ptr, remaining, destination_ptr, &needs_serial);
        source_ptr += consumed;
        destination_ptr += consumed;
        if (!needs_serial && consumed) continue; // a clean vectorized strip (or trailing-incomplete trim)

        // One codepoint that the vector path can't fold in place: fold only well-formed runes; copy malformed
        // bytes through unchanged, resyncing one byte at a time, exactly as the strict serial reference does.
        sz_rune_t rune;
        sz_rune_length_t const rune_length = sz_rune_decode((sz_cptr_t)source_ptr, (sz_cptr_t)source_end, &rune);
        if (rune_length == sz_rune_invalid_k) {
            *destination_ptr++ = *source_ptr++;
            continue;
        }
        source_ptr += rune_length;
        sz_rune_t folded_runes[3]; // Unicode case folding produces at most 3 runes
        sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
        for (sz_size_t rune_index = 0; rune_index != folded_count; ++rune_index)
            destination_ptr += sz_rune_encode(folded_runes[rune_index], destination_ptr);
    }
    return (sz_size_t)(destination_ptr - (sz_u8_t *)destination);
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

#endif // STRINGZILLA_UTF8_UNCASED_FOLD_RVV_H_
