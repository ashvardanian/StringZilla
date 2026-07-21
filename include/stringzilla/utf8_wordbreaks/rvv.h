/**
 *  @file   include/stringzilla/utf8_wordbreaks/rvv.h
 *  @author Ash Vardanian
 *  @brief  Fully-vectorized UAX-29 Word_Break segmentation for RVV 1.0 (RISC-V). The RVV twin of the AVX2 (Haswell),
 *          Ice Lake, and NEON kernels: no path scalar-walks codepoints or calls the serial oracle.
 *
 *  Each 64-byte window lives at `e8m4` — the one SEW/LMUL pair whose VLMAX is exactly 64 lanes at VLEN=128 (wider
 *  machines clamp `vl` to 64) AND the maximum data LMUL admitting the 16-bit indexed `vluxei16` byte gather. Every
 *  per-codepoint BMP Word_Break property resolves through the shared page-compressed flat table via
 *  @ref sz_utf8_rune_flat_lookup_rvv_ (a real masked vector gather, unlike NEON's bounded scalar walk), and the
 *  Supplementary Plane through the same 5-nibble cascade as chained indexed loads, bit-identical to
 *  `sz_rune_word_break_property` over the whole code space.
 *
 *  The classified window is lowered to the portable @ref sz_utf8_word_break_frame_t and handed to the SHARED
 *  `sz_utf8_word_break_decide_window_` rule engine, so WB1-WB16 (including the cross-window bridge shadow / RI
 *  parity / left-context carry / WB3c neighbour coupling) run once in portable `sz_u64_t` bit algebra. Malformed
 *  input is handled in-vector: ill-formed leads and strays become `forced_other` in the portable partition
 *  resolver, truncated-edge leads are reclassified in the frame builder — no well-formedness prepass, no serial
 *  routing.
 *
 *  Mask-domain discipline: the classify leaves stay entirely in `vbool2_t` mask registers, recomputing the lead
 *  classes from the threaded raw window instead of round-tripping the window struct's `sz_u64_t` fields back through
 *  memory. The only `sz_u64_t` -> mask raises (`vlm.v`) that survive are the genuine engine/portable boundaries: the
 *  truncated-edge reclassify (mixes the scalar `loaded` clamp), the driver's `forced_other` merge (partition-resolver
 *  origin), and the drain's boundary compaction (engine `breaks` origin).
 */
#ifndef STRINGZILLA_UTF8_WORDBREAKS_RVV_H_
#define STRINGZILLA_UTF8_WORDBREAKS_RVV_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_wordbreaks/tables.h"
#include "stringzilla/utf8_wordbreaks/serial.h"
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

#pragma region UAX 29 Word Boundaries forward kernel

#pragma region In register vectorized classifier

/** @brief  Word_Break class byte for the ASCII lanes (cp < 0x80) via the 128-entry property table, one masked
 *          `vluxei8` gather (index `byte & 0x7F` is always in-bounds); masked-off lanes keep @p inactive_u8m4.
 *          The window byte equals the codepoint on ASCII lanes. */
SZ_HELPER_INLINE vuint8m4_t sz_utf8_word_break_ascii_class_rvv_(vuint8m4_t raw_u8m4, vbool2_t active_b2,
                                                                vuint8m4_t inactive_u8m4) {
    vuint8m4_t const indices_u8m4 = __riscv_vand_vx_u8m4(raw_u8m4, 0x7F, 64);
    return __riscv_vluxei8_v_u8m4_mu(active_b2, inactive_u8m4, &sz_utf8_word_break_property_ascii_[0], indices_u8m4,
                                     64);
}

/** @brief  Word_Break class byte for the ASTRAL lanes over the 20-bit offset = cp - 0x10000 (5-nibble cascade), the
 *          RVV twin of @ref sz_utf8_word_break_astral_class_neon_ with the four table stages resolved by chained
 *          indexed loads instead of `vqtbl` scans. Garbage lanes stay in-bounds by stage-table totality (stage-1
 *          values < 14, stage-2 values below the stage-3 row count) except at stage 4, where `leaf_group` is
 *          clamped for the address and the result zeroed on out-of-range groups, matching the NEON blend loop.
 *          Bit-exact with `sz_rune_word_break_property` over the Supplementary Planes. */
SZ_HELPER_AUTO vuint8m4_t sz_utf8_word_break_astral_class_rvv_( //
    vuint8m4_t plane_off_u8m4, vuint8m4_t high_u8m4, vuint8m4_t low_u8m4) {
    vuint8m4_t const n4_u8m4 = __riscv_vand_vx_u8m4(plane_off_u8m4, 0x0F, 64);
    vuint8m4_t const n3_u8m4 = __riscv_vsrl_vx_u8m4(high_u8m4, 4, 64);
    vuint8m4_t const stage1_index_u8m4 = __riscv_vor_vv_u8m4(__riscv_vsll_vx_u8m4(n4_u8m4, 4, 64), n3_u8m4, 64);
    vuint8m4_t const page_u8m4 = __riscv_vluxei8_v_u8m4(&sz_utf8_word_break_haswell_astral_stage1_[0],
                                                        stage1_index_u8m4, 64);

    vuint8m4_t const n2_u8m4 = __riscv_vand_vx_u8m4(high_u8m4, 0x0F, 64);
    vuint8m4_t const stage2_index_u8m4 = __riscv_vor_vv_u8m4(__riscv_vsll_vx_u8m4(page_u8m4, 4, 64), n2_u8m4, 64);
    vuint8m4_t const leaf2_u8m4 = __riscv_vluxei8_v_u8m4(&sz_utf8_word_break_haswell_astral_stage2_lo_[0],
                                                         stage2_index_u8m4, 64);

    vuint8m4_t const n1_u8m4 = __riscv_vand_vx_u8m4(__riscv_vsrl_vx_u8m4(low_u8m4, 4, 64), 0x0F, 64);
    vuint16m8_t const stage3_index_u16m8 = __riscv_vwaddu_wv_u16m8(
        __riscv_vsll_vx_u16m8(__riscv_vzext_vf2_u16m8(leaf2_u8m4, 64), 4, 64), n1_u8m4, 64);
    vuint8m4_t const leaf_lo_u8m4 = __riscv_vluxei16_v_u8m4(&sz_utf8_word_break_haswell_astral_stage3_lo_[0],
                                                            stage3_index_u16m8, 64);
    vuint8m4_t const leaf_hi_u8m4 = __riscv_vluxei16_v_u8m4(&sz_utf8_word_break_haswell_astral_stage3_hi_[0],
                                                            stage3_index_u16m8, 64);

    vuint8m4_t const n0_u8m4 = __riscv_vand_vx_u8m4(low_u8m4, 0x0F, 64);
    vuint8m4_t const leaf_group_u8m4 = __riscv_vor_vv_u8m4(
        __riscv_vand_vx_u8m4(__riscv_vsrl_vx_u8m4(leaf_lo_u8m4, 4, 64), 0x0F, 64),
        __riscv_vsll_vx_u8m4(leaf_hi_u8m4, 4, 64), 64);
    vuint8m4_t const stage4_lut_index_u8m4 = __riscv_vor_vv_u8m4(
        __riscv_vsll_vx_u8m4(__riscv_vand_vx_u8m4(leaf_lo_u8m4, 0x0F, 64), 4, 64), n0_u8m4, 64);
    vbool2_t const group_bad_b2 = __riscv_vmsgtu_vx_u8m4_b2(leaf_group_u8m4,
                                                            sz_utf8_word_break_haswell_astral_leaf_groups_k - 1, 64);
    vuint16m8_t const stage4_index_u16m8 = __riscv_vwaddu_wv_u16m8(
        __riscv_vsll_vx_u16m8(
            __riscv_vzext_vf2_u16m8(
                __riscv_vminu_vx_u8m4(leaf_group_u8m4, sz_utf8_word_break_haswell_astral_leaf_groups_k - 1, 64), 64),
            8, 64),
        stage4_lut_index_u8m4, 64);
    vuint8m4_t const gathered_u8m4 = __riscv_vluxei16_v_u8m4(&sz_utf8_word_break_haswell_astral_stage4_groups_[0],
                                                             stage4_index_u16m8, 64);
    return __riscv_vmerge_vxm_u8m4(gathered_u8m4, 0, group_bad_b2, 64);
}

/** @brief  Per-window byte-lane classification (RVV): the Word_Break class byte per lane, valid only on
 *          codepoint-start lanes (the engine reads classes only at starts). The RVV twin of
 *          @ref sz_utf8_word_break_classify_window_neon_, bit-identical on every start lane.
 *
 *  The lead-class lane masks are recomputed here in `vbool2_t` registers straight from the threaded @p raw_u8m4
 *  (four `vand`/`vmseq` compares), rather than raising the window struct's `sz_u64_t` fields back through memory —
 *  `vbool2_t` is sizeless and cannot ride the struct, and the compares are cheaper than a `vlm.v` round-trip. The
 *  `loaded` clamp is an in-register `vid < loaded` compare (lane indices <= 63 fit `u8`). ASCII and BMP resolve
 *  through UNCONDITIONAL masked gathers — inactive lanes perform no memory access, and index safety comes from table
 *  totality, never from the mask — so only the rare 4-gather astral cascade keeps a branch. */
SZ_HELPER_AUTO vuint8m4_t sz_utf8_word_break_classify_window_rvv_(vuint8m4_t const raw_u8m4,
                                                                  sz_utf8_rune_window_rvv_t const *window) {
    // In-register lead classes and the `loaded` clamp, mirroring the struct fields `decode_window` lowered for the
    // engine but kept as masks so no `sz_u64_t` is raised back to a `vbool2_t` here.
    vbool2_t const within_loaded_b2 = __riscv_vmsltu_vx_u8m4_b2(__riscv_vid_v_u8m4(64), (sz_u8_t)window->loaded, 64);
    vbool2_t const two_byte_b2 = __riscv_vmseq_vx_u8m4_b2(__riscv_vand_vx_u8m4(raw_u8m4, 0xE0, 64), 0xC0, 64);
    vbool2_t const three_byte_b2 = __riscv_vmseq_vx_u8m4_b2(__riscv_vand_vx_u8m4(raw_u8m4, 0xF0, 64), 0xE0, 64);
    vbool2_t const ascii_start_b2 = __riscv_vmand_mm_b2(within_loaded_b2, __riscv_vmsltu_vx_u8m4_b2(raw_u8m4, 0x80, 64),
                                                        64);
    vbool2_t const bmp_start_b2 = __riscv_vmand_mm_b2(within_loaded_b2,
                                                      __riscv_vmor_mm_b2(two_byte_b2, three_byte_b2, 64), 64);

    // The three start masks are mutually exclusive by construction, so the blend order is immaterial: ASCII fills
    // over the zero splat, the BMP flat gather rides the ASCII lanes through its inactive operand, astral merges.
    vuint8m4_t classes_u8m4 = sz_utf8_word_break_ascii_class_rvv_(raw_u8m4, ascii_start_b2,
                                                                  __riscv_vmv_v_x_u8m4(0, 64));

    vuint8m4x2_t const halves_u8m4x2 = sz_utf8_rune_bmp_halves_rvv_(raw_u8m4);
    classes_u8m4 = sz_utf8_rune_flat_lookup_rvv_(&sz_utf8_word_break_bmp_page_lut_[0], &sz_utf8_word_break_flat_bmp_[0],
                                                 __riscv_vget_v_u8m4x2_u8m4(halves_u8m4x2, 0),
                                                 __riscv_vget_v_u8m4x2_u8m4(halves_u8m4x2, 1), bmp_start_b2,
                                                 classes_u8m4);

    // 4-byte (astral) lanes: reconstruct the codepoint from the lead + three forward neighbours, then the astral
    // cascade addressed by offset = codepoint - 0x10000 (the offset's plane nibble is `plane - 1`).
    if (window->four_byte_starts) {
        vuint8m4_t const next1_u8m4 = sz_utf8_rune_forward_neighbour_rvv_(raw_u8m4, 1);
        vuint8m4_t const next2_u8m4 = sz_utf8_rune_forward_neighbour_rvv_(raw_u8m4, 2);
        vuint8m4_t const next3_u8m4 = sz_utf8_rune_forward_neighbour_rvv_(raw_u8m4, 3);
        vuint8m4_t const plane_u8m4 = __riscv_vor_vv_u8m4(
            __riscv_vsll_vx_u8m4(__riscv_vand_vx_u8m4(raw_u8m4, 0x07, 64), 2, 64),
            __riscv_vand_vx_u8m4(__riscv_vsrl_vx_u8m4(next1_u8m4, 4, 64), 0x03, 64), 64);
        vuint8m4_t const high_four_u8m4 = __riscv_vor_vv_u8m4(
            __riscv_vsll_vx_u8m4(__riscv_vand_vx_u8m4(next1_u8m4, 0x0F, 64), 4, 64),
            __riscv_vand_vx_u8m4(__riscv_vsrl_vx_u8m4(next2_u8m4, 2, 64), 0x0F, 64), 64);
        vuint8m4_t const low_four_u8m4 = __riscv_vor_vv_u8m4(
            __riscv_vsll_vx_u8m4(__riscv_vand_vx_u8m4(next2_u8m4, 0x03, 64), 6, 64),
            __riscv_vand_vx_u8m4(next3_u8m4, 0x3F, 64), 64);
        vbool2_t const four_start_b2 = __riscv_vmand_mm_b2(
            within_loaded_b2, __riscv_vmseq_vx_u8m4_b2(__riscv_vand_vx_u8m4(raw_u8m4, 0xF8, 64), 0xF0, 64), 64);
        vuint8m4_t const plane_off_u8m4 = __riscv_vsub_vx_u8m4(plane_u8m4, 1, 64);
        vuint8m4_t const astral_u8m4 = sz_utf8_word_break_astral_class_rvv_(plane_off_u8m4, high_four_u8m4,
                                                                            low_four_u8m4);
        classes_u8m4 = __riscv_vmerge_vvm_u8m4(classes_u8m4, astral_u8m4, four_start_b2, 64);
    }
    return classes_u8m4;
}

#pragma endregion In register vectorized classifier

#pragma region Mask algebra extractor

/** @brief  A 64-bit "class byte == @p value" lane mask over the classified window (`vmseq` -> mask bits). */
SZ_HELPER_INLINE sz_u64_t sz_utf8_word_break_class_mask_rvv_(vuint8m4_t classes_u8m4, sz_u8_t value) {
    return sz_utf8_rune_mask_to_bits_rvv_(__riscv_vmseq_vx_u8m4_b2(classes_u8m4, value, 64));
}

/** @brief  A 64-bit "raw window byte == @p value" lane mask. */
SZ_HELPER_INLINE sz_u64_t sz_utf8_word_break_byte_equal_rvv_(vuint8m4_t raw_u8m4, sz_u8_t value) {
    return sz_utf8_rune_mask_to_bits_rvv_(__riscv_vmseq_vx_u8m4_b2(raw_u8m4, value, 64));
}

/** @brief  A 64-bit "raw window byte >= @p bound" (unsigned) lane mask. */
SZ_HELPER_INLINE sz_u64_t sz_utf8_word_break_byte_ge_rvv_(vuint8m4_t raw_u8m4, sz_u8_t bound) {
    return sz_utf8_rune_mask_to_bits_rvv_(__riscv_vmsgeu_vx_u8m4_b2(raw_u8m4, bound, 64));
}

/** @brief  Assemble the per-lane BMP codepoint `(high << 8) | low` at `e16m8` for the 16-bit range scans. */
SZ_HELPER_INLINE vuint16m8_t sz_utf8_word_break_codepoint16_rvv_(vuint8m4_t high_u8m4, vuint8m4_t low_u8m4) {
    return __riscv_vwaddu_wv_u16m8(__riscv_vsll_vx_u16m8(__riscv_vzext_vf2_u16m8(high_u8m4, 64), 8, 64), low_u8m4, 64);
}

/** @brief  A 64-bit "codepoint16 in any sorted `[lo, hi]` range" lane mask, the RVV twin of
 *          @ref sz_utf8_word_break_range16_mask_neon_ over native 16-bit lanes (WSegSpace / Extended_Pictographic). */
SZ_HELPER_AUTO sz_u64_t sz_utf8_word_break_range16_mask_rvv_( //
    vuint16m8_t values_u16m8, sz_u16_t const *lo_table, sz_u16_t const *hi_table, int count) {
    vbool2_t hit_b2 = __riscv_vmclr_m_b2(64);
    for (int range = 0; range < count; ++range) {
        vbool2_t const not_below_b2 = __riscv_vmsgeu_vx_u16m8_b2(values_u16m8, lo_table[range], 64);
        vbool2_t const not_above_b2 = __riscv_vmsleu_vx_u16m8_b2(values_u16m8, hi_table[range], 64);
        hit_b2 = __riscv_vmor_mm_b2(hit_b2, __riscv_vmand_mm_b2(not_below_b2, not_above_b2, 64), 64);
    }
    return sz_utf8_rune_mask_to_bits_rvv_(hit_b2);
}

#pragma endregion Mask algebra extractor

#pragma region Codepoint partition

/** @brief  Resolve one window into the maximal-subpart partition — the RVV twin of
 *          @ref sz_utf8_word_break_partition_neon_: compute the per-ISA `sz_u64_t` masks and delegate to the
 *          portable @ref sz_utf8_word_break_partition_from_masks_. */
SZ_HELPER_AUTO sz_utf8_word_break_partition_t sz_utf8_word_break_partition_rvv_( //
    vuint8m4_t const raw_u8m4, sz_utf8_rune_window_rvv_t const *window, sz_u64_t valid, int at_end_of_text) {
    sz_u64_t const real_continuation = window->continuation & valid;
    // Declared length follows the serial high-nibble rule: 0xC/0xD -> 2, 0xE -> 3, 0xF -> 4. The strict
    // `two`/`three_byte_starts` masks already match 0xC0-0xDF and 0xE0-0xEF; only `length_four` needs widening to
    // fold the ill-formed leads 0xF8-0xFF so they collapse to U+FFFD like serial/Haswell instead of leaking.
    sz_u64_t const length_two = window->two_byte_starts & valid;
    sz_u64_t const length_three = window->three_byte_starts & valid;
    sz_u64_t const length_four = (window->four_byte_starts | sz_utf8_word_break_byte_ge_rvv_(raw_u8m4, 0xF8)) & valid;
    sz_u64_t const length_ge_two = length_two | length_three | length_four;
    sz_u64_t bad_second_byte = 0ull;
    if (length_ge_two) {
        vuint8m4_t const next1_u8m4 = sz_utf8_rune_forward_neighbour_rvv_(raw_u8m4, 1);
        sz_u64_t const next1_at_least_a0 = sz_utf8_word_break_byte_ge_rvv_(next1_u8m4, 0xA0);
        sz_u64_t const next1_at_least_90 = sz_utf8_word_break_byte_ge_rvv_(next1_u8m4, 0x90);
        sz_u64_t const lead_c0_c1 = (sz_utf8_word_break_byte_equal_rvv_(raw_u8m4, 0xC0) |
                                     sz_utf8_word_break_byte_equal_rvv_(raw_u8m4, 0xC1)) &
                                    valid;
        sz_u64_t const lead_e0 = sz_utf8_word_break_byte_equal_rvv_(raw_u8m4, 0xE0) & valid;
        sz_u64_t const lead_ed = sz_utf8_word_break_byte_equal_rvv_(raw_u8m4, 0xED) & valid;
        sz_u64_t const lead_f0 = sz_utf8_word_break_byte_equal_rvv_(raw_u8m4, 0xF0) & valid;
        sz_u64_t const lead_f4 = sz_utf8_word_break_byte_equal_rvv_(raw_u8m4, 0xF4) & valid;
        sz_u64_t const lead_f5_or_more = sz_utf8_word_break_byte_ge_rvv_(raw_u8m4, 0xF5) & valid;
        bad_second_byte = lead_c0_c1 | (lead_e0 & ~next1_at_least_a0) | (lead_ed & next1_at_least_a0) |
                          (lead_f0 & ~next1_at_least_90) | (lead_f4 & next1_at_least_90) | lead_f5_or_more;
    }
    return sz_utf8_word_break_partition_from_masks_(real_continuation, length_two, length_three, length_four,
                                                    bad_second_byte, valid, at_end_of_text);
}

#pragma endregion Codepoint partition

#pragma region Mask algebra frame builder

/** @brief  Per-ISA extractor: lower one classified 64-byte window to the portable
 *          @ref sz_utf8_word_break_frame_t — the RVV twin of @ref sz_utf8_word_break_build_frame_neon_. Applies
 *          the truncated-edge U+FFFD reclassify to the class lanes, materializes every per-class lane mask + the
 *          raw-byte membership masks, the Extended_Pictographic mask (BMP + SMP range scan), and the per-lane
 *          class byte array. */
SZ_HELPER_AUTO sz_utf8_word_break_frame_t sz_utf8_word_break_build_frame_rvv_( //
    vuint8m4_t const raw_u8m4, sz_utf8_rune_window_rvv_t const *window, vuint8m4_t classes_u8m4,
    sz_u64_t start_bytes_all, sz_u64_t length_two, sz_u64_t length_three, sz_u64_t length_four, int want_pictographic) {

    sz_size_t const loaded = window->loaded;
    sz_u64_t const valid = sz_u64_mask_until_serial_(loaded);
    sz_u64_t const start_bytes = start_bytes_all & valid;

    // Truncated-edge U+FFFD reclassify (force the class to Other on a lead whose declared span runs past `loaded`).
    sz_u64_t const lead_two = length_two & start_bytes;
    sz_u64_t const lead_three = length_three & start_bytes;
    sz_u64_t const lead_four = length_four & start_bytes;
    sz_u64_t const truncated_raw = ((lead_two & ~sz_u64_mask_until_serial_(loaded > 1 ? loaded - 1 : 0)) |
                                    (lead_three & ~sz_u64_mask_until_serial_(loaded > 2 ? loaded - 2 : 0)) |
                                    (lead_four & ~sz_u64_mask_until_serial_(loaded > 3 ? loaded - 3 : 0))) &
                                   valid;
    // `vlm.v` #1 of 3: `truncated_raw` mixes the partition's `sz_u64_t` masks with the scalar `loaded` clamp, so it
    // is genuinely engine-domain; a single edge-gated raise merges Other over the affected leads.
    if (truncated_raw)
        classes_u8m4 = __riscv_vmerge_vxm_u8m4(classes_u8m4, (sz_u8_t)sz_utf8_word_break_other_k,
                                               sz_utf8_rune_bits_to_mask_rvv_(truncated_raw), 64);

    sz_utf8_word_break_frame_t frame;
    frame.class_aletter = sz_utf8_word_break_class_mask_rvv_(classes_u8m4, sz_utf8_word_break_aletter_k);
    frame.class_hebrew = sz_utf8_word_break_class_mask_rvv_(classes_u8m4, sz_utf8_word_break_hebrew_letter_k);
    frame.class_numeric = sz_utf8_word_break_class_mask_rvv_(classes_u8m4, sz_utf8_word_break_numeric_k);
    frame.class_katakana = sz_utf8_word_break_class_mask_rvv_(classes_u8m4, sz_utf8_word_break_katakana_k);
    frame.class_extendnumlet = sz_utf8_word_break_class_mask_rvv_(classes_u8m4, sz_utf8_word_break_extendnumlet_k);
    frame.class_extend = sz_utf8_word_break_class_mask_rvv_(classes_u8m4, sz_utf8_word_break_extend_k);
    frame.class_zwj = sz_utf8_word_break_class_mask_rvv_(classes_u8m4, sz_utf8_word_break_zwj_k);
    frame.class_format = sz_utf8_word_break_class_mask_rvv_(classes_u8m4, sz_utf8_word_break_format_k);
    frame.class_midletter = sz_utf8_word_break_class_mask_rvv_(classes_u8m4, sz_utf8_word_break_midletter_k);
    frame.class_midnum = sz_utf8_word_break_class_mask_rvv_(classes_u8m4, sz_utf8_word_break_midnum_k);
    frame.class_mid_quotes = sz_utf8_word_break_class_mask_rvv_(classes_u8m4, sz_utf8_word_break_mid_quotes_k);
    frame.class_cr = sz_utf8_word_break_class_mask_rvv_(classes_u8m4, sz_utf8_word_break_cr_k);
    frame.class_lf = sz_utf8_word_break_class_mask_rvv_(classes_u8m4, sz_utf8_word_break_lf_k);
    frame.class_newline = sz_utf8_word_break_class_mask_rvv_(classes_u8m4, sz_utf8_word_break_newline_k);
    frame.class_regional = sz_utf8_word_break_class_mask_rvv_(classes_u8m4, sz_utf8_word_break_regional_ind_k);

    sz_u64_t const non_ascii_lanes = sz_utf8_word_break_byte_ge_rvv_(raw_u8m4, 0x80) & valid;
    frame.non_ascii_lanes = non_ascii_lanes;
    frame.double_quote_byte = sz_utf8_word_break_byte_equal_rvv_(raw_u8m4, 0x22) & valid;
    frame.single_quote_byte = sz_utf8_word_break_byte_equal_rvv_(raw_u8m4, 0x27) & valid;

    // One shared BMP codepoint derivation for both 16-bit range scans: WB3d WSegSpace and the WB3c
    // Extended_Pictographic BMP scan read the SAME `codepoint16`, differing only in the range table, so the
    // `bmp_halves` -> `codepoint16` chain is built once. `pictographic_bmp` only survives on non-ASCII, non-4-byte
    // lanes, so gating the whole block on `non_ascii_lanes` loses nothing. The single live `codepoint16` group dies
    // before the SMP path forms its own, so the two never coexist and the compiler spills only the raw window.
    sz_u64_t wseg_multibyte = 0ull;
    sz_u64_t pictographic_bmp = 0ull;
    if (non_ascii_lanes) {
        vuint8m4x2_t const halves_u8m4x2 = sz_utf8_rune_bmp_halves_rvv_(raw_u8m4);
        vuint16m8_t const codepoint16_u16m8 = sz_utf8_word_break_codepoint16_rvv_(
            __riscv_vget_v_u8m4x2_u8m4(halves_u8m4x2, 0), __riscv_vget_v_u8m4x2_u8m4(halves_u8m4x2, 1));
        wseg_multibyte = sz_utf8_word_break_range16_mask_rvv_(codepoint16_u16m8, sz_utf8_word_break_wseg_lo_,
                                                              sz_utf8_word_break_wseg_hi_,
                                                              sz_utf8_word_break_wseg_count_k) &
                         non_ascii_lanes;
        if (want_pictographic)
            pictographic_bmp = sz_utf8_word_break_range16_mask_rvv_(codepoint16_u16m8, sz_utf8_word_break_pict_bmp_lo_,
                                                                    sz_utf8_word_break_pict_bmp_hi_,
                                                                    sz_utf8_word_break_pict_bmp_count_k);
    }
    frame.wseg = (wseg_multibyte | (sz_utf8_word_break_byte_equal_rvv_(raw_u8m4, 0x20) & valid));

    // WB3c Extended_Pictographic SMP membership (range scan on plane-one 4-byte lanes). Rare-class gated on
    // `want_pictographic` (an in-window ZWJ or the carried `prev_ends_in_zwj`), so the ~156-range scan is skipped on
    // the common no-ZWJ window. The engine applies the final gating.
    frame.pictographic = 0ull;
    sz_u64_t const four_byte = window->four_byte_starts & valid;
    if (want_pictographic) {
        vuint8m4_t const next1_u8m4 = sz_utf8_rune_forward_neighbour_rvv_(raw_u8m4, 1);
        vuint8m4_t const next2_u8m4 = sz_utf8_rune_forward_neighbour_rvv_(raw_u8m4, 2);
        vuint8m4_t const next3_u8m4 = sz_utf8_rune_forward_neighbour_rvv_(raw_u8m4, 3);
        vuint8m4_t const plane_u8m4 = __riscv_vor_vv_u8m4(
            __riscv_vsll_vx_u8m4(__riscv_vand_vx_u8m4(raw_u8m4, 0x07, 64), 2, 64),
            __riscv_vand_vx_u8m4(__riscv_vsrl_vx_u8m4(next1_u8m4, 4, 64), 0x03, 64), 64);
        vuint8m4_t const smp_high_u8m4 = __riscv_vor_vv_u8m4(
            __riscv_vsll_vx_u8m4(__riscv_vand_vx_u8m4(next1_u8m4, 0x0F, 64), 4, 64),
            __riscv_vand_vx_u8m4(__riscv_vsrl_vx_u8m4(next2_u8m4, 2, 64), 0x0F, 64), 64);
        vuint8m4_t const smp_low_u8m4 = __riscv_vor_vv_u8m4(
            __riscv_vsll_vx_u8m4(__riscv_vand_vx_u8m4(next2_u8m4, 0x03, 64), 6, 64),
            __riscv_vand_vx_u8m4(next3_u8m4, 0x3F, 64), 64);
        sz_u64_t const plane_one = sz_utf8_rune_mask_to_bits_rvv_(__riscv_vmseq_vx_u8m4_b2(plane_u8m4, 1, 64));
        sz_u64_t const pictographic_smp = sz_utf8_word_break_range16_mask_rvv_(
            sz_utf8_word_break_codepoint16_rvv_(smp_high_u8m4, smp_low_u8m4), sz_utf8_word_break_pict_smp_lo_,
            sz_utf8_word_break_pict_smp_hi_, sz_utf8_word_break_pict_smp_count_k);
        frame.pictographic = (pictographic_bmp & non_ascii_lanes & ~four_byte) |
                             (pictographic_smp & four_byte & plane_one);
    }

    __riscv_vse8_v_u8m4(frame.classes_byte, classes_u8m4, 64);
    return frame;
}

#pragma endregion Mask algebra frame builder

#pragma region Forward driver

/**
 *  @brief  Forward UAX-29 word segmentation over `[0, length)` (RVV 1.0): the overlap-free advancing driver,
 *          mirroring @ref sz_utf8_wordbreaks_neon over the RVV window/classify/partition/decide/drain leaves.
 *          Bit-exact with `sz_utf8_wordbreaks_serial` and every other windowed backend.
 */
SZ_API_COMPTIME sz_size_t sz_utf8_wordbreaks_rvv(    //
    sz_cptr_t text, sz_size_t length,                //
    sz_size_t *word_starts, sz_size_t *word_lengths, //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;

    sz_size_t words = 0;         // words written to the output
    sz_size_t word_start = 0;    // start byte of the currently open (unfinished) word
    sz_size_t bridge_anchor = 0; // byte offset of the consumed, still-unresolved Mid* (valid while bridge_open)
    sz_size_t position = 0;      // codepoint-aligned anchor of the next window (advances cleanly)

    sz_utf8_word_break_carry_t carry = sz_utf8_word_break_carry_sot_();

    while (position < length) {
        // The ONE raw-window materialization: the `loaded` clamp stays adjacent to the load so the zero-tail
        // invariant every downstream `vslidedown` neighbour depends on flows from this single point.
        sz_size_t const loaded = (length - position) < 64 ? (length - position) : 64;
        vuint8m4_t const raw_u8m4 = sz_utf8_rune_load64_rvv_(text_u8 + position, loaded);
        sz_utf8_rune_window_rvv_t const window = sz_utf8_rune_decode_window_rvv_(raw_u8m4, loaded);
        sz_u64_t const valid = sz_u64_mask_until_serial_(loaded);

        vuint8m4_t classes_u8m4 = sz_utf8_word_break_classify_window_rvv_(raw_u8m4, &window);

        sz_utf8_word_break_partition_t const partition = sz_utf8_word_break_partition_rvv_(raw_u8m4, &window, valid,
                                                                                           position + loaded >= length);
        sz_u64_t const start_bytes_all = partition.start_bytes;
        sz_u64_t const continuation_all = partition.continuation;
        sz_u64_t const forced_other = partition.forced_other;
        sz_u64_t const length_two = partition.length_two;
        sz_u64_t const length_three = partition.length_three;
        sz_u64_t const length_four = partition.length_four;
        // `vlm.v` #2 of 3: `forced_other` originates in the portable partition resolver, so its merge of Other over
        // the strays/short leads is a genuine engine-domain raise.
        if (forced_other)
            classes_u8m4 = __riscv_vmerge_vxm_u8m4(classes_u8m4, (sz_u8_t)sz_utf8_word_break_other_k,
                                                   sz_utf8_rune_bits_to_mask_rvv_(forced_other), 64);

        sz_bool_t const more_text = (position + loaded < length) ? sz_true_k : sz_false_k;

        // Effective-window<64: when more text follows, the final codepoint's blind byte span may spill past the
        // 64-byte edge. Trust and classify only up to the last fully decoded codepoint (`complete_limit`).
        sz_size_t complete_limit = loaded;
        if (more_text) {
            sz_u64_t const two = length_two & start_bytes_all;
            sz_u64_t const three = length_three & start_bytes_all;
            sz_u64_t const four = length_four & start_bytes_all;
            sz_u64_t const straddle = ((two & ~sz_u64_mask_until_serial_(loaded > 1 ? loaded - 1 : 0)) |
                                       (three & ~sz_u64_mask_until_serial_(loaded > 2 ? loaded - 2 : 0)) |
                                       (four & ~sz_u64_mask_until_serial_(loaded > 3 ? loaded - 3 : 0))) &
                                      valid;
            // First straddling lead = lowest set bit of `straddle`. Base rv64gcv has no scalar Zbb `ctz`, but the
            // exempt `sz_u64_clz` is already on this path, so `63 - clz(lowest_bit)` yields the index without raising
            // the scalar `straddle` into a mask register for a `vfirst`.
            sz_size_t limit = straddle ? (sz_size_t)(63 - sz_u64_clz(straddle & (0ull - straddle))) : loaded;
            if ((text_u8[position + loaded] & 0xC0) == 0x80) {
                sz_size_t const last_lead = (sz_size_t)(63 - sz_u64_clz(start_bytes_all));
                sz_size_t const last_lead_length = sz_utf8_lead_length_(text_u8[position + last_lead]);
                if (last_lead + last_lead_length > loaded && last_lead < limit) limit = last_lead;
            }
            if (limit > 0) complete_limit = limit;
        }

        int const want_pictographic = sz_utf8_word_break_class_mask_rvv_(classes_u8m4, sz_utf8_word_break_zwj_k) != 0 ||
                                      carry.prev_ends_in_zwj;
        sz_utf8_word_break_frame_t const frame = sz_utf8_word_break_build_frame_rvv_(
            raw_u8m4, &window, classes_u8m4, start_bytes_all, length_two, length_three, length_four, want_pictographic);

        sz_utf8_word_break_carry_t carry_full = carry;
        sz_utf8_word_break_window_t const win = sz_utf8_word_break_decide_window_(
            &frame, start_bytes_all, continuation_all, forced_other, length_two, length_three, length_four,
            complete_limit, &carry_full, more_text);

        sz_size_t const adv = win.resolved;
        sz_u64_t const boundary_lanes = win.breaks & sz_u64_mask_until_serial_(adv);
        if (win.deferred_break) {
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_start;
                return words;
            }
            word_starts[words] = word_start;
            word_lengths[words] = bridge_anchor - word_start;
            ++words;
            word_start = bridge_anchor;
        }

        words = sz_utf8_rune_drain_forward_rvv_(boundary_lanes, position, word_starts, word_lengths, words,
                                                words_capacity, &word_start);
        if (words == words_capacity) {
            if (bytes_consumed) *bytes_consumed = word_start;
            return words;
        }

        if (adv > 0 && adv < complete_limit) {
            sz_utf8_word_break_carry_t carry_to_edge = carry;
            sz_utf8_word_break_decide_window_(&frame, start_bytes_all, continuation_all, forced_other, length_two,
                                              length_three, length_four, adv, &carry_to_edge, sz_true_k);
            carry = carry_to_edge;
            position += adv;
        }
        else {
            int const bridge_opened = carry_full.bridge_open && !carry.bridge_open;
            carry = carry_full;
            if (bridge_opened) bridge_anchor = position;
            position += complete_limit ? complete_limit : loaded;
        }
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

#pragma endregion Forward driver

#pragma endregion UAX 29 Word Boundaries forward kernel
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_RVV

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDBREAKS_RVV_H_
