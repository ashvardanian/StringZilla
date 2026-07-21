/**
 *  @brief Arm SVE2 backend for UTF-8 case folding.
 *  @file include/stringzilla/utf8_uncased_fold/sve2.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased_fold.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_FOLD_SVE2_H_
#define STRINGZILLA_UTF8_UNCASED_FOLD_SVE2_H_

#include "stringzilla/utf8_uncased_fold/serial.h"
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

/*  The scalable twin of the NEON fold: the same 64-byte logical superchunk and family handlers, walked as
 *  `64 / svcntb()` register chunks with one peeked vector, so chunk decisions stay comparable across the serial,
 *  NEON, Ice Lake, and SVE2 back-ends on the same input. After-lead positions ride the shared value-domain
 *  up-shift with a cross-chunk carry; stop lanes lower once per chunk through the predicate bridge and resolve
 *  through the shared serial boundary walk-back. */

/** @brief Folds ASCII A-Z down to a-z in one register, leaving every other byte unchanged. */
SZ_HELPER_INLINE svuint8_t sz_utf8_fold_sve2_ascii_(svuint8_t source_u8x) {
    svbool_t const all_b8x = svptrue_b8();
    svbool_t const is_upper_b8x = svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, source_u8x, 'A'), 26);
    return svadd_n_u8_m(is_upper_b8x, source_u8x, 0x20);
}

/** @brief  Per-lead well-formedness mirror of `sz_rune_decode` - the SVE2 twin of
 *          @ref sz_utf8_fold_neon_malformed_lead_ over one (chunk, peek) pair.
 *  @return Predicate set on every lead byte that does NOT begin a well-formed rune. */
SZ_HELPER_AUTO svbool_t sz_utf8_fold_sve2_malformed_lead_(svuint8_t source_u8x, svuint8_t peek_u8x) {
    svbool_t const all_b8x = svptrue_b8();
    svuint8_t const next1_u8x = svext_u8(source_u8x, peek_u8x, 1);
    svuint8_t const next2_u8x = svext_u8(source_u8x, peek_u8x, 2);
    svuint8_t const next3_u8x = svext_u8(source_u8x, peek_u8x, 3);

    svbool_t const is_continuation_b8x = svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0x80), 0x40);
    svbool_t const is_lead_b8x = svbic_b_z(all_b8x, svcmpge_n_u8(all_b8x, source_u8x, 0x80), is_continuation_b8x);
    svbool_t const next1_continuation_b8x = svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, next1_u8x, 0x80), 0x40);
    svbool_t const next2_continuation_b8x = svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, next2_u8x, 0x80), 0x40);
    svbool_t const next3_continuation_b8x = svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, next3_u8x, 0x80), 0x40);

    svbool_t const is_two_byte_lead_b8x = svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0xC2), 0x1E);
    svbool_t const is_three_byte_lead_b8x = svcmpeq_n_u8(all_b8x, svand_n_u8_x(all_b8x, source_u8x, 0xF0), 0xE0);
    svbool_t const is_four_byte_lead_b8x = svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0xF0), 0x05);

    svbool_t const two_complete_b8x = svand_b_z(all_b8x, is_two_byte_lead_b8x, next1_continuation_b8x);
    svbool_t const three_complete_b8x = svand_b_z(all_b8x, is_three_byte_lead_b8x,
                                                  svand_b_z(all_b8x, next1_continuation_b8x, next2_continuation_b8x));
    svbool_t const four_complete_b8x = svand_b_z(
        all_b8x, is_four_byte_lead_b8x,
        svand_b_z(all_b8x, next1_continuation_b8x, svand_b_z(all_b8x, next2_continuation_b8x, next3_continuation_b8x)));

    svbool_t const e0_bad_b8x = svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, source_u8x, 0xE0),
                                          svcmplt_n_u8(all_b8x, next1_u8x, 0xA0));
    svbool_t const ed_bad_b8x = svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, source_u8x, 0xED),
                                          svcmpge_n_u8(all_b8x, next1_u8x, 0xA0));
    svbool_t const f0_bad_b8x = svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, source_u8x, 0xF0),
                                          svcmplt_n_u8(all_b8x, next1_u8x, 0x90));
    svbool_t const f4_bad_b8x = svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, source_u8x, 0xF4),
                                          svcmpge_n_u8(all_b8x, next1_u8x, 0x90));
    svbool_t const bad_special_b8x = svorr_b_z(all_b8x, svorr_b_z(all_b8x, e0_bad_b8x, ed_bad_b8x),
                                               svorr_b_z(all_b8x, f0_bad_b8x, f4_bad_b8x));

    svbool_t const well_formed_b8x = svbic_b_z(
        all_b8x, svorr_b_z(all_b8x, two_complete_b8x, svorr_b_z(all_b8x, three_complete_b8x, four_complete_b8x)),
        bad_special_b8x);
    return svbic_b_z(all_b8x, is_lead_b8x, well_formed_b8x);
}

/** @brief  Folds a 64-byte superchunk containing only caseless multi-byte scripts mixed with ASCII - the SVE2
 *          twin of @ref sz_utf8_uncased_fold_neon_caseless_chunk_.
 *  @return Bytes consumed; always 62..64, never zero. */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_sve2_caseless_chunk_(sz_cptr_t source, sz_ptr_t target) {
    sz_size_t const chunk_bytes = svcntb() < 64 ? svcntb() : 64;
    for (sz_size_t chunk_base = 0; chunk_base < 64; chunk_base += chunk_bytes) {
        svbool_t const loaded_b8x = svwhilelt_b8_u64((sz_u64_t)chunk_base, 64);
        svuint8_t const source_u8x = svld1_u8(loaded_b8x, (sz_u8_t const *)source + chunk_base);
        svst1_u8(loaded_b8x, (sz_u8_t *)target + chunk_base, sz_utf8_fold_sve2_ascii_(source_u8x));
    }
    // A 2-byte lead is incomplete only in the last byte; a 3-byte lead in the last two.
    sz_u8_t const second_to_last_byte = (sz_u8_t)source[62], last_byte = (sz_u8_t)source[63];
    if ((second_to_last_byte & 0xF0) == 0xE0) return 62;
    if ((last_byte & 0xC0) == 0xC0) return 63;
    return 64;
}

/** @brief  Folds a 64-byte superchunk of Latin text in place - the SVE2 twin of
 *          @ref sz_utf8_uncased_fold_neon_latin_chunk_, with the same C4/C5/C6 delta tables read through
 *          chunked `svtbl` walks and the same stop policy.
 *  @return Bytes consumed and written, or zero if the first character needs the serial path. */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_sve2_latin_chunk_(sz_cptr_t source, sz_ptr_t target) {
    svbool_t const all_b8x = svptrue_b8();
    sz_size_t const chunk_bytes = svcntb() < 64 ? svcntb() : 64;
    svuint8_t const lane_iota_u8x = svindex_u8(0, 1);
    svuint8_t const zeros_u8x = svdup_n_u8(0);

    svuint8_t previous_c2_u8x = zeros_u8x, previous_c3_u8x = zeros_u8x, previous_c4_u8x = zeros_u8x;
    svuint8_t previous_c5_u8x = zeros_u8x, previous_c6_u8x = zeros_u8x, previous_e1_u8x = zeros_u8x;
    svuint8_t previous_additional_second_u8x = zeros_u8x, previous_ba_second_u8x = zeros_u8x;
    sz_u64_t stop_lanes = 0;

    for (sz_size_t chunk_base = 0; chunk_base < 64; chunk_base += chunk_bytes) {
        sz_size_t const chunk_span = 64 - chunk_base < chunk_bytes ? 64 - chunk_base : chunk_bytes;
        svbool_t const loaded_b8x = svwhilelt_b8_u64(0, (sz_u64_t)chunk_span);
        svuint8_t const source_u8x = svld1_u8(loaded_b8x, (sz_u8_t const *)source + chunk_base);
        sz_size_t const peek_span = 64 - chunk_base > chunk_bytes ? chunk_bytes : 0;
        svuint8_t const peek_u8x = svld1_u8(svwhilelt_b8_u64(0, (sz_u64_t)peek_span),
                                            (sz_u8_t const *)source + chunk_base + (peek_span ? chunk_bytes : 0));
        svuint8_t const next_byte_u8x = svext_u8(source_u8x, peek_u8x, 1);

        svbool_t const is_c2_b8x = svcmpeq_n_u8(loaded_b8x, source_u8x, 0xC2);
        svbool_t const is_c3_b8x = svcmpeq_n_u8(loaded_b8x, source_u8x, 0xC3);
        svbool_t const is_c4_b8x = svcmpeq_n_u8(loaded_b8x, source_u8x, 0xC4);
        svbool_t const is_c5_b8x = svcmpeq_n_u8(loaded_b8x, source_u8x, 0xC5);
        svbool_t const is_c6_b8x = svcmpeq_n_u8(loaded_b8x, source_u8x, 0xC6);
        svbool_t const is_e1_b8x = svcmpeq_n_u8(loaded_b8x, source_u8x, 0xE1);
        svbool_t const is_continuation_b8x = svcmplt_n_u8(loaded_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0x80), 0x40);

        svbool_t const is_lead_b8x = svbic_b_z(loaded_b8x, svcmpge_n_u8(loaded_b8x, source_u8x, 0x80),
                                               is_continuation_b8x);
        svbool_t const is_family_lead_b8x = svorr_b_z(
            loaded_b8x, svcmplt_n_u8(loaded_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0xC2), 0x05), is_e1_b8x);
        svbool_t const is_foreign_lead_b8x = svbic_b_z(loaded_b8x, is_lead_b8x, is_family_lead_b8x);

        svuint8_t const after_c2_u8x = sz_utf8_shift_value_up_pair_sve2_(previous_c2_u8x, svdup_u8_z(is_c2_b8x, 1), 1,
                                                                         lane_iota_u8x, (sz_u8_t)chunk_bytes);
        svuint8_t const after_c3_u8x = sz_utf8_shift_value_up_pair_sve2_(previous_c3_u8x, svdup_u8_z(is_c3_b8x, 1), 1,
                                                                         lane_iota_u8x, (sz_u8_t)chunk_bytes);
        svuint8_t const after_c4_u8x = sz_utf8_shift_value_up_pair_sve2_(previous_c4_u8x, svdup_u8_z(is_c4_b8x, 1), 1,
                                                                         lane_iota_u8x, (sz_u8_t)chunk_bytes);
        svuint8_t const after_c5_u8x = sz_utf8_shift_value_up_pair_sve2_(previous_c5_u8x, svdup_u8_z(is_c5_b8x, 1), 1,
                                                                         lane_iota_u8x, (sz_u8_t)chunk_bytes);
        svuint8_t const after_c6_u8x = sz_utf8_shift_value_up_pair_sve2_(previous_c6_u8x, svdup_u8_z(is_c6_b8x, 1), 1,
                                                                         lane_iota_u8x, (sz_u8_t)chunk_bytes);
        svuint8_t const after_e1_u8x = sz_utf8_shift_value_up_pair_sve2_(previous_e1_u8x, svdup_u8_z(is_e1_b8x, 1), 1,
                                                                         lane_iota_u8x, (sz_u8_t)chunk_bytes);
        svbool_t const after_c2_b8x = svcmpne_n_u8(all_b8x, after_c2_u8x, 0);
        svbool_t const after_c3_b8x = svcmpne_n_u8(all_b8x, after_c3_u8x, 0);
        svbool_t const after_e1_b8x = svcmpne_n_u8(all_b8x, after_e1_u8x, 0);

        // E1 sequences qualify only when the second byte is B8-BB (Latin Extended Additional).
        svbool_t const is_additional_second_b8x = svand_b_z(
            all_b8x, after_e1_b8x, svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0xB8), 0x04));
        svbool_t const foreign_e1_second_b8x = svbic_b_z(all_b8x, after_e1_b8x, is_additional_second_b8x);
        svbool_t const is_ba_second_b8x = svand_b_z(all_b8x, is_additional_second_b8x,
                                                    svcmpeq_n_u8(all_b8x, source_u8x, 0xBA));
        svuint8_t const additional_third_u8x = sz_utf8_shift_value_up_pair_sve2_(
            previous_additional_second_u8x, svdup_u8_z(is_additional_second_b8x, 1), 1, lane_iota_u8x,
            (sz_u8_t)chunk_bytes);
        svuint8_t const ba_third_u8x = sz_utf8_shift_value_up_pair_sve2_(
            previous_ba_second_u8x, svdup_u8_z(is_ba_second_b8x, 1), 1, lane_iota_u8x, (sz_u8_t)chunk_bytes);

        // Latin Extended Additional irregulars: E1 BA 96-9E expand or shrink when folded.
        svbool_t const irregular_additional_b8x = svand_b_z(
            all_b8x, svcmpne_n_u8(all_b8x, ba_third_u8x, 0),
            svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0x96), 0x09));

        // Latin Extended 2-byte deltas: the shared C4/C5/C6 tables through chunked `svtbl` walks, each gated by
        // its after-lead predicate; the three contributions land on disjoint lanes.
        svbool_t const after_c4_b8x = svcmpne_n_u8(all_b8x, after_c4_u8x, 0);
        svbool_t const after_c5_b8x = svcmpne_n_u8(all_b8x, after_c5_u8x, 0);
        svbool_t const after_c6_b8x = svcmpne_n_u8(all_b8x, after_c6_u8x, 0);
        svuint8_t const delta_indices_u8x = svand_n_u8_x(all_b8x, source_u8x, 0x3F);
        svuint8_t extended_deltas_u8x = svdup_u8_z(after_c4_b8x,
                                                   0); // placeholder lanes; each select below seats its family's delta
        extended_deltas_u8x = svsel_u8(after_c4_b8x,
                                       sz_utf8_rune_lut_sve2_(sz_utf8_fold_c4_deltas_lut_, 64, delta_indices_u8x),
                                       extended_deltas_u8x);
        extended_deltas_u8x = svsel_u8(after_c5_b8x,
                                       sz_utf8_rune_lut_sve2_(sz_utf8_fold_c5_deltas_lut_, 64, delta_indices_u8x),
                                       extended_deltas_u8x);
        extended_deltas_u8x = svsel_u8(after_c6_b8x,
                                       sz_utf8_rune_lut_sve2_(sz_utf8_fold_c6_deltas_lut_, 64, delta_indices_u8x),
                                       extended_deltas_u8x);
        extended_deltas_u8x = svand_u8_z(is_continuation_b8x, extended_deltas_u8x, extended_deltas_u8x);
        svbool_t const irregular_extended_b8x = svcmpne_n_u8(all_b8x, svand_n_u8_x(all_b8x, extended_deltas_u8x, 0x80),
                                                             0);

        svbool_t const malformed_lead_b8x = sz_utf8_fold_sve2_malformed_lead_(source_u8x, peek_u8x);
        svbool_t const stop_b8x = svorr_b_z(
            loaded_b8x, svorr_b_z(loaded_b8x, irregular_extended_b8x, foreign_e1_second_b8x),
            svorr_b_z(loaded_b8x, svorr_b_z(loaded_b8x, irregular_additional_b8x, is_foreign_lead_b8x),
                      malformed_lead_b8x));
        stop_lanes |= sz_utf8_rune_pred_to_u64_sve2_(svand_b_z(loaded_b8x, stop_b8x, loaded_b8x)) << chunk_base;

        // 1. ASCII A-Z; 2. Latin-1 Supplement 'À'-'Þ' (C3 80-9E minus '×' 0x97) +0x20.
        svuint8_t folded_u8x = sz_utf8_fold_sve2_ascii_(source_u8x);
        svbool_t is_latin1_upper_b8x = svand_b_z(all_b8x, after_c3_b8x,
                                                 svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0x80), 0x1F));
        is_latin1_upper_b8x = svbic_b_z(all_b8x, is_latin1_upper_b8x, svcmpeq_n_u8(all_b8x, source_u8x, 0x97));
        folded_u8x = svadd_n_u8_m(is_latin1_upper_b8x, folded_u8x, 0x20);

        // 3. 'ß' (C3 9F) -> "ss": replace both bytes in place.
        svbool_t const is_eszett_b8x = svorr_b_z(
            all_b8x, svand_b_z(all_b8x, after_c3_b8x, svcmpeq_n_u8(all_b8x, source_u8x, 0x9F)),
            svand_b_z(all_b8x, is_c3_b8x, svcmpeq_n_u8(all_b8x, next_byte_u8x, 0x9F)));
        folded_u8x = svsel_u8(is_eszett_b8x, svdup_n_u8('s'), folded_u8x);

        // 4. 'µ' (C2 B5) -> 'μ' (CE BC): both bytes replaced in place.
        svbool_t const is_micro_lead_b8x = svand_b_z(all_b8x, is_c2_b8x, svcmpeq_n_u8(all_b8x, next_byte_u8x, 0xB5));
        svbool_t const is_micro_second_b8x = svand_b_z(all_b8x, after_c2_b8x, svcmpeq_n_u8(all_b8x, source_u8x, 0xB5));
        folded_u8x = svsel_u8(is_micro_lead_b8x, svdup_n_u8(0xCE), folded_u8x);
        folded_u8x = svsel_u8(is_micro_second_b8x, svdup_n_u8(0xBC), folded_u8x);

        // 5. Latin Extended-A/B: masked +1 on the continuation byte (irregular lanes sit past the stop cut).
        folded_u8x = svadd_u8_x(all_b8x, folded_u8x, extended_deltas_u8x);

        // 6. Latin Extended Additional: +1 on even third bytes of E1 B8-BB sequences.
        svbool_t const is_even_third_b8x = svand_b_z(all_b8x, svcmpne_n_u8(all_b8x, additional_third_u8x, 0),
                                                     svcmpeq_n_u8(all_b8x, svand_n_u8_x(all_b8x, source_u8x, 0x01), 0));
        folded_u8x = svadd_n_u8_m(is_even_third_b8x, folded_u8x, 0x01);

        svst1_u8(loaded_b8x, (sz_u8_t *)target + chunk_base, folded_u8x);

        previous_c2_u8x = svdup_u8_z(is_c2_b8x, 1), previous_c3_u8x = svdup_u8_z(is_c3_b8x, 1);
        previous_c4_u8x = svdup_u8_z(is_c4_b8x, 1), previous_c5_u8x = svdup_u8_z(is_c5_b8x, 1);
        previous_c6_u8x = svdup_u8_z(is_c6_b8x, 1), previous_e1_u8x = svdup_u8_z(is_e1_b8x, 1);
        previous_additional_second_u8x = svdup_u8_z(is_additional_second_b8x, 1);
        previous_ba_second_u8x = svdup_u8_z(is_ba_second_b8x, 1);
    }

    if (stop_lanes) return sz_utf8_fold_stop_boundary_serial_(stop_lanes, source);
    sz_u8_t const second_to_last_byte = (sz_u8_t)source[62], last_byte = (sz_u8_t)source[63];
    if (second_to_last_byte == 0xE1) return 62;
    if ((sz_u8_t)(last_byte - 0xC2) < 0x05 || last_byte == 0xE1) return 63;
    return 64;
}

/** @brief  Folds a 64-byte superchunk of basic Cyrillic mixed with ASCII - the SVE2 twin of
 *          @ref sz_utf8_uncased_fold_neon_cyrillic_chunk_.
 *  @return Bytes consumed and written, or zero if the first character needs another path. */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_sve2_cyrillic_chunk_(sz_cptr_t source, sz_ptr_t target) {
    static sz_u8_t const second_byte_offsets_lut_[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x20, 0xE0, 0, 0, 0, 0, 0};
    svbool_t const all_b8x = svptrue_b8();
    sz_size_t const chunk_bytes = svcntb() < 64 ? svcntb() : 64;
    svuint8_t const lane_iota_u8x = svindex_u8(0, 1);
    svuint8_t const offsets_lut_u8x = svld1_u8(svwhilelt_b8_u64(0, 16), second_byte_offsets_lut_);
    svuint8_t previous_d0_u8x = svdup_n_u8(0);
    sz_u64_t stop_lanes = 0;

    for (sz_size_t chunk_base = 0; chunk_base < 64; chunk_base += chunk_bytes) {
        sz_size_t const chunk_span = 64 - chunk_base < chunk_bytes ? 64 - chunk_base : chunk_bytes;
        svbool_t const loaded_b8x = svwhilelt_b8_u64(0, (sz_u64_t)chunk_span);
        svuint8_t const source_u8x = svld1_u8(loaded_b8x, (sz_u8_t const *)source + chunk_base);
        sz_size_t const peek_span = 64 - chunk_base > chunk_bytes ? chunk_bytes : 0;
        svuint8_t const peek_u8x = svld1_u8(svwhilelt_b8_u64(0, (sz_u64_t)peek_span),
                                            (sz_u8_t const *)source + chunk_base + (peek_span ? chunk_bytes : 0));
        svuint8_t const next_byte_u8x = svext_u8(source_u8x, peek_u8x, 1);

        svbool_t const is_d0_b8x = svcmpeq_n_u8(loaded_b8x, source_u8x, 0xD0);
        svbool_t const is_d1_b8x = svcmpeq_n_u8(loaded_b8x, source_u8x, 0xD1);
        svbool_t const is_continuation_b8x = svcmplt_n_u8(loaded_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0x80), 0x40);
        svbool_t const is_lead_b8x = svbic_b_z(loaded_b8x, svcmpge_n_u8(loaded_b8x, source_u8x, 0x80),
                                               is_continuation_b8x);
        svbool_t const is_foreign_lead_b8x = svbic_b_z(loaded_b8x, is_lead_b8x,
                                                       svorr_b_z(loaded_b8x, is_d0_b8x, is_d1_b8x));

        // Cyrillic Extended-A (D1 A0+) folds by parity and stays on the serial path.
        svbool_t const is_extended_b8x = svand_b_z(loaded_b8x, is_d1_b8x, svcmpge_n_u8(all_b8x, next_byte_u8x, 0xA0));
        svbool_t const malformed_lead_b8x = sz_utf8_fold_sve2_malformed_lead_(source_u8x, peek_u8x);
        svbool_t const stop_b8x = svorr_b_z(loaded_b8x, svorr_b_z(loaded_b8x, is_foreign_lead_b8x, is_extended_b8x),
                                            malformed_lead_b8x);
        stop_lanes |= sz_utf8_rune_pred_to_u64_sve2_(svand_b_z(loaded_b8x, stop_b8x, loaded_b8x)) << chunk_base;

        // Second-byte offsets keyed by the high nibble: 80-8F -> +0x10, 90-9F -> +0x20, A0-AF -> -0x20.
        svuint8_t const after_d0_u8x = sz_utf8_shift_value_up_pair_sve2_(previous_d0_u8x, svdup_u8_z(is_d0_b8x, 1), 1,
                                                                         lane_iota_u8x, (sz_u8_t)chunk_bytes);
        svbool_t const after_d0_b8x = svcmpne_n_u8(all_b8x, after_d0_u8x, 0);
        svuint8_t const offsets_u8x = svand_u8_z(after_d0_b8x,
                                                 svtbl_u8(offsets_lut_u8x, svlsr_n_u8_x(all_b8x, source_u8x, 4)),
                                                 svtbl_u8(offsets_lut_u8x, svlsr_n_u8_x(all_b8x, source_u8x, 4)));
        svuint8_t folded_u8x = svadd_u8_x(all_b8x, sz_utf8_fold_sve2_ascii_(source_u8x), offsets_u8x);

        // Lead rewrite D0 -> D1 (+1) where the lowercase lives in the next block: seconds 80-8F and A0-AF.
        svbool_t const needs_d1_b8x = svand_b_z(
            all_b8x, is_d0_b8x,
            svorr_b_z(all_b8x, svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, next_byte_u8x, 0x80), 0x10),
                      svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, next_byte_u8x, 0xA0), 0x10)));
        folded_u8x = svadd_n_u8_m(needs_d1_b8x, folded_u8x, 0x01);

        svst1_u8(loaded_b8x, (sz_u8_t *)target + chunk_base, folded_u8x);
        previous_d0_u8x = svdup_u8_z(is_d0_b8x, 1);
    }

    if (stop_lanes) return sz_utf8_fold_stop_boundary_serial_(stop_lanes, source);
    sz_u8_t const last_byte = (sz_u8_t)source[63];
    return (last_byte == 0xD0 || last_byte == 0xD1) ? 63 : 64;
}

/** @brief  Folds a 64-byte superchunk of basic Greek mixed with ASCII - the SVE2 twin of
 *          @ref sz_utf8_uncased_fold_neon_greek_chunk_.
 *  @return Bytes consumed and written, or zero if the first character needs another path. */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_sve2_greek_chunk_(sz_cptr_t source, sz_ptr_t target) {
    svbool_t const all_b8x = svptrue_b8();
    sz_size_t const chunk_bytes = svcntb() < 64 ? svcntb() : 64;
    svuint8_t const lane_iota_u8x = svindex_u8(0, 1);
    svuint8_t previous_ce_u8x = svdup_n_u8(0), previous_cf_u8x = svdup_n_u8(0);
    sz_u64_t stop_lanes = 0;

    for (sz_size_t chunk_base = 0; chunk_base < 64; chunk_base += chunk_bytes) {
        sz_size_t const chunk_span = 64 - chunk_base < chunk_bytes ? 64 - chunk_base : chunk_bytes;
        svbool_t const loaded_b8x = svwhilelt_b8_u64(0, (sz_u64_t)chunk_span);
        svuint8_t const source_u8x = svld1_u8(loaded_b8x, (sz_u8_t const *)source + chunk_base);
        sz_size_t const peek_span = 64 - chunk_base > chunk_bytes ? chunk_bytes : 0;
        svuint8_t const peek_u8x = svld1_u8(svwhilelt_b8_u64(0, (sz_u64_t)peek_span),
                                            (sz_u8_t const *)source + chunk_base + (peek_span ? chunk_bytes : 0));
        svuint8_t const next_byte_u8x = svext_u8(source_u8x, peek_u8x, 1);

        svbool_t const is_ce_b8x = svcmpeq_n_u8(loaded_b8x, source_u8x, 0xCE);
        svbool_t const is_cf_b8x = svcmpeq_n_u8(loaded_b8x, source_u8x, 0xCF);
        svbool_t const is_continuation_b8x = svcmplt_n_u8(loaded_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0x80), 0x40);
        svbool_t const is_lead_b8x = svbic_b_z(loaded_b8x, svcmpge_n_u8(loaded_b8x, source_u8x, 0x80),
                                               is_continuation_b8x);
        svbool_t const is_foreign_lead_b8x = svbic_b_z(loaded_b8x, is_lead_b8x,
                                                       svorr_b_z(loaded_b8x, is_ce_b8x, is_cf_b8x));

        // Exclusions at the LEAD position: accented uppercase and 'ΰ' (CE < 0x91 or == 0xB0), symbols (CF >= 8F).
        svbool_t const ce_excluded_b8x = svand_b_z(
            loaded_b8x, is_ce_b8x,
            svorr_b_z(all_b8x, svcmplt_n_u8(all_b8x, next_byte_u8x, 0x91), svcmpeq_n_u8(all_b8x, next_byte_u8x, 0xB0)));
        svbool_t const cf_excluded_b8x = svand_b_z(loaded_b8x, is_cf_b8x, svcmpge_n_u8(all_b8x, next_byte_u8x, 0x8F));
        svbool_t const malformed_lead_b8x = sz_utf8_fold_sve2_malformed_lead_(source_u8x, peek_u8x);
        svbool_t const stop_b8x = svorr_b_z(
            loaded_b8x,
            svorr_b_z(loaded_b8x, is_foreign_lead_b8x, svorr_b_z(loaded_b8x, ce_excluded_b8x, cf_excluded_b8x)),
            malformed_lead_b8x);
        stop_lanes |= sz_utf8_rune_pred_to_u64_sve2_(svand_b_z(loaded_b8x, stop_b8x, loaded_b8x)) << chunk_base;

        svuint8_t const after_ce_u8x = sz_utf8_shift_value_up_pair_sve2_(previous_ce_u8x, svdup_u8_z(is_ce_b8x, 1), 1,
                                                                         lane_iota_u8x, (sz_u8_t)chunk_bytes);
        svuint8_t const after_cf_u8x = sz_utf8_shift_value_up_pair_sve2_(previous_cf_u8x, svdup_u8_z(is_cf_b8x, 1), 1,
                                                                         lane_iota_u8x, (sz_u8_t)chunk_bytes);
        svbool_t const after_ce_b8x = svcmpne_n_u8(all_b8x, after_ce_u8x, 0);
        svbool_t const after_cf_b8x = svcmpne_n_u8(all_b8x, after_cf_u8x, 0);

        // 'Α'-'Ο': +0x20 in place; 'Π'-'Ρ' and 'Σ'-'Ϋ': -0x20 (+0xE0 wrap) with a CE -> CF lead promotion;
        // final sigma 'ς' (CF 82) -> 'σ': +1.
        svbool_t const is_basic_upper_b8x = svand_b_z(
            all_b8x, after_ce_b8x, svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0x91), 0x0F));
        svbool_t const is_promoting_upper_b8x = svand_b_z(
            all_b8x, after_ce_b8x,
            svorr_b_z(all_b8x, svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0xA0), 0x02),
                      svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0xA3), 0x09)));
        svbool_t const is_final_sigma_b8x = svand_b_z(all_b8x, after_cf_b8x, svcmpeq_n_u8(all_b8x, source_u8x, 0x82));

        svuint8_t folded_u8x = sz_utf8_fold_sve2_ascii_(source_u8x);
        folded_u8x = svadd_n_u8_m(is_basic_upper_b8x, folded_u8x, 0x20);
        folded_u8x = svadd_n_u8_m(is_promoting_upper_b8x, folded_u8x, 0xE0);
        folded_u8x = svadd_n_u8_m(is_final_sigma_b8x, folded_u8x, 0x01);

        svbool_t const promotes_lead_b8x = svand_b_z(
            all_b8x, is_ce_b8x,
            svorr_b_z(all_b8x, svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, next_byte_u8x, 0xA0), 0x02),
                      svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, next_byte_u8x, 0xA3), 0x09)));
        folded_u8x = svadd_n_u8_m(promotes_lead_b8x, folded_u8x, 0x01);

        svst1_u8(loaded_b8x, (sz_u8_t *)target + chunk_base, folded_u8x);
        previous_ce_u8x = svdup_u8_z(is_ce_b8x, 1), previous_cf_u8x = svdup_u8_z(is_cf_b8x, 1);
    }

    if (stop_lanes) return sz_utf8_fold_stop_boundary_serial_(stop_lanes, source);
    sz_u8_t const last_byte = (sz_u8_t)source[63];
    return (last_byte == 0xCE || last_byte == 0xCF) ? 63 : 64;
}

/** @brief  Folds a 64-byte superchunk of Armenian (D4-D6 leads) mixed with ASCII - the SVE2 twin of
 *          @ref sz_utf8_uncased_fold_neon_armenian_chunk_.
 *  @return Bytes consumed and written, or zero if the first character needs another path. */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_sve2_armenian_chunk_(sz_cptr_t source, sz_ptr_t target) {
    svbool_t const all_b8x = svptrue_b8();
    sz_size_t const chunk_bytes = svcntb() < 64 ? svcntb() : 64;
    svuint8_t const lane_iota_u8x = svindex_u8(0, 1);
    svuint8_t previous_d4_u8x = svdup_n_u8(0), previous_d5_u8x = svdup_n_u8(0);
    sz_u64_t stop_lanes = 0;

    for (sz_size_t chunk_base = 0; chunk_base < 64; chunk_base += chunk_bytes) {
        sz_size_t const chunk_span = 64 - chunk_base < chunk_bytes ? 64 - chunk_base : chunk_bytes;
        svbool_t const loaded_b8x = svwhilelt_b8_u64(0, (sz_u64_t)chunk_span);
        svuint8_t const source_u8x = svld1_u8(loaded_b8x, (sz_u8_t const *)source + chunk_base);
        sz_size_t const peek_span = 64 - chunk_base > chunk_bytes ? chunk_bytes : 0;
        svuint8_t const peek_u8x = svld1_u8(svwhilelt_b8_u64(0, (sz_u64_t)peek_span),
                                            (sz_u8_t const *)source + chunk_base + (peek_span ? chunk_bytes : 0));
        svuint8_t const next_byte_u8x = svext_u8(source_u8x, peek_u8x, 1);

        svbool_t const is_d4_b8x = svcmpeq_n_u8(loaded_b8x, source_u8x, 0xD4);
        svbool_t const is_d5_b8x = svcmpeq_n_u8(loaded_b8x, source_u8x, 0xD5);
        svbool_t const is_d6_b8x = svcmpeq_n_u8(loaded_b8x, source_u8x, 0xD6);
        svbool_t const is_continuation_b8x = svcmplt_n_u8(loaded_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0x80), 0x40);
        svbool_t const is_lead_b8x = svbic_b_z(loaded_b8x, svcmpge_n_u8(loaded_b8x, source_u8x, 0x80),
                                               is_continuation_b8x);
        svbool_t const is_armenian_lead_b8x = svorr_b_z(loaded_b8x, svorr_b_z(loaded_b8x, is_d4_b8x, is_d5_b8x),
                                                        is_d6_b8x);
        svbool_t const is_foreign_lead_b8x = svbic_b_z(loaded_b8x, is_lead_b8x, is_armenian_lead_b8x);

        // Stops at the LEAD: D4 with a Cyrillic-Supplement second (<= B0) and the ligature (D6 87).
        svbool_t const is_d4_stop_b8x = svand_b_z(loaded_b8x, is_d4_b8x, svcmplt_n_u8(all_b8x, next_byte_u8x, 0xB1));
        svbool_t const is_ligature_stop_b8x = svand_b_z(loaded_b8x, is_d6_b8x,
                                                        svcmpeq_n_u8(all_b8x, next_byte_u8x, 0x87));
        svbool_t const malformed_lead_b8x = sz_utf8_fold_sve2_malformed_lead_(source_u8x, peek_u8x);
        svbool_t const stop_b8x = svorr_b_z(
            loaded_b8x,
            svorr_b_z(loaded_b8x, is_foreign_lead_b8x, svorr_b_z(loaded_b8x, is_d4_stop_b8x, is_ligature_stop_b8x)),
            malformed_lead_b8x);
        stop_lanes |= sz_utf8_rune_pred_to_u64_sve2_(svand_b_z(loaded_b8x, stop_b8x, loaded_b8x)) << chunk_base;

        svuint8_t const after_d4_u8x = sz_utf8_shift_value_up_pair_sve2_(previous_d4_u8x, svdup_u8_z(is_d4_b8x, 1), 1,
                                                                         lane_iota_u8x, (sz_u8_t)chunk_bytes);
        svuint8_t const after_d5_u8x = sz_utf8_shift_value_up_pair_sve2_(previous_d5_u8x, svdup_u8_z(is_d5_b8x, 1), 1,
                                                                         lane_iota_u8x, (sz_u8_t)chunk_bytes);
        svbool_t const after_d4_b8x = svcmpne_n_u8(all_b8x, after_d4_u8x, 0);
        svbool_t const after_d5_b8x = svcmpne_n_u8(all_b8x, after_d5_u8x, 0);

        // Second-byte offsets on disjoint lanes: D4 B1-BF -> -0x10 (+0xF0), D5 80-8F -> +0x30, D5 90-96 -> -0x10.
        svbool_t const is_d4_second_b8x = svand_b_z(all_b8x, after_d4_b8x, svcmpge_n_u8(all_b8x, source_u8x, 0xB1));
        svbool_t const is_d5_plus30_b8x = svand_b_z(
            all_b8x, after_d5_b8x, svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0x80), 0x10));
        svbool_t const is_d5_minus10_b8x = svand_b_z(
            all_b8x, after_d5_b8x, svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0x90), 0x07));

        svuint8_t folded_u8x = sz_utf8_fold_sve2_ascii_(source_u8x);
        folded_u8x = svadd_n_u8_m(svorr_b_z(all_b8x, is_d4_second_b8x, is_d5_minus10_b8x), folded_u8x, 0xF0);
        folded_u8x = svadd_n_u8_m(is_d5_plus30_b8x, folded_u8x, 0x30);

        // Lead +1 rewrites: D4 -> D5 where next is B1-BF, D5 -> D6 where next is 90-96.
        svbool_t const promotes_d4_b8x = svand_b_z(all_b8x, is_d4_b8x, svcmpge_n_u8(all_b8x, next_byte_u8x, 0xB1));
        svbool_t const promotes_d5_b8x = svand_b_z(
            all_b8x, is_d5_b8x, svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, next_byte_u8x, 0x90), 0x07));
        folded_u8x = svadd_n_u8_m(svorr_b_z(all_b8x, promotes_d4_b8x, promotes_d5_b8x), folded_u8x, 0x01);

        svst1_u8(loaded_b8x, (sz_u8_t *)target + chunk_base, folded_u8x);
        previous_d4_u8x = svdup_u8_z(is_d4_b8x, 1), previous_d5_u8x = svdup_u8_z(is_d5_b8x, 1);
    }

    if (stop_lanes) return sz_utf8_fold_stop_boundary_serial_(stop_lanes, source);
    sz_u8_t const last_byte = (sz_u8_t)source[63];
    return (last_byte == 0xD4 || last_byte == 0xD5 || last_byte == 0xD6) ? 63 : 64;
}

/** @brief  Folds a 64-byte superchunk of Georgian (E1 82/83 content) mixed with ASCII - the SVE2 twin of
 *          @ref sz_utf8_uncased_fold_neon_georgian_chunk_.
 *  @return Bytes consumed and written, or zero if the first character needs another path. */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_sve2_georgian_chunk_(sz_cptr_t source, sz_ptr_t target) {
    svbool_t const all_b8x = svptrue_b8();
    sz_size_t const chunk_bytes = svcntb() < 64 ? svcntb() : 64;
    svuint8_t const lane_iota_u8x = svindex_u8(0, 1);
    svuint8_t previous_82_upper_lead_u8x = svdup_n_u8(0), previous_83_upper_lead_u8x = svdup_n_u8(0);
    svuint8_t previous_82_upper_second_u8x = svdup_n_u8(0), previous_83_upper_second_u8x = svdup_n_u8(0);
    sz_u64_t stop_lanes = 0;

    for (sz_size_t chunk_base = 0; chunk_base < 64; chunk_base += chunk_bytes) {
        sz_size_t const chunk_span = 64 - chunk_base < chunk_bytes ? 64 - chunk_base : chunk_bytes;
        svbool_t const loaded_b8x = svwhilelt_b8_u64(0, (sz_u64_t)chunk_span);
        svuint8_t const source_u8x = svld1_u8(loaded_b8x, (sz_u8_t const *)source + chunk_base);
        sz_size_t const peek_span = 64 - chunk_base > chunk_bytes ? chunk_bytes : 0;
        svuint8_t const peek_u8x = svld1_u8(svwhilelt_b8_u64(0, (sz_u64_t)peek_span),
                                            (sz_u8_t const *)source + chunk_base + (peek_span ? chunk_bytes : 0));
        svuint8_t const next_byte_u8x = svext_u8(source_u8x, peek_u8x, 1);
        svuint8_t const next_next_byte_u8x = svext_u8(source_u8x, peek_u8x, 2);

        svbool_t const is_e1_b8x = svcmpeq_n_u8(loaded_b8x, source_u8x, 0xE1);
        svbool_t const is_continuation_b8x = svcmplt_n_u8(loaded_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0x80), 0x40);
        svbool_t const is_lead_b8x = svbic_b_z(loaded_b8x, svcmpge_n_u8(loaded_b8x, source_u8x, 0x80),
                                               is_continuation_b8x);
        svbool_t const is_foreign_lead_b8x = svbic_b_z(loaded_b8x, is_lead_b8x, is_e1_b8x);

        svbool_t const is_82_lead_b8x = svand_b_z(loaded_b8x, is_e1_b8x, svcmpeq_n_u8(all_b8x, next_byte_u8x, 0x82));
        svbool_t const is_83_lead_b8x = svand_b_z(loaded_b8x, is_e1_b8x, svcmpeq_n_u8(all_b8x, next_byte_u8x, 0x83));
        svbool_t const is_foreign_e1_b8x = svbic_b_z(loaded_b8x, is_e1_b8x,
                                                     svorr_b_z(loaded_b8x, is_82_lead_b8x, is_83_lead_b8x));
        svbool_t const malformed_lead_b8x = sz_utf8_fold_sve2_malformed_lead_(source_u8x, peek_u8x);
        svbool_t const stop_b8x = svorr_b_z(loaded_b8x, svorr_b_z(loaded_b8x, is_foreign_lead_b8x, is_foreign_e1_b8x),
                                            malformed_lead_b8x);
        stop_lanes |= sz_utf8_rune_pred_to_u64_sve2_(svand_b_z(loaded_b8x, stop_b8x, loaded_b8x)) << chunk_base;

        // Uppercase Georgian is keyed by the THIRD byte: E1 82 third >= A0, E1 83 third in 80-85 / 87 / 8D.
        svbool_t const is_82_upper_lead_b8x = svand_b_z(loaded_b8x, is_82_lead_b8x,
                                                        svcmpge_n_u8(all_b8x, next_next_byte_u8x, 0xA0));
        svbool_t const is_83_third_b8x = svorr_b_z(
            all_b8x, svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, next_next_byte_u8x, 0x80), 0x06),
            svorr_b_z(all_b8x, svcmpeq_n_u8(all_b8x, next_next_byte_u8x, 0x87),
                      svcmpeq_n_u8(all_b8x, next_next_byte_u8x, 0x8D)));
        svbool_t const is_83_upper_lead_b8x = svand_b_z(loaded_b8x, is_83_lead_b8x, is_83_third_b8x);

        // Carry the uppercase flag forward: lane +1 for the second-byte rewrite, +2 for the third.
        svuint8_t const is_82_upper_second_u8x = sz_utf8_shift_value_up_pair_sve2_(
            previous_82_upper_lead_u8x, svdup_u8_z(is_82_upper_lead_b8x, 1), 1, lane_iota_u8x, (sz_u8_t)chunk_bytes);
        svuint8_t const is_83_upper_second_u8x = sz_utf8_shift_value_up_pair_sve2_(
            previous_83_upper_lead_u8x, svdup_u8_z(is_83_upper_lead_b8x, 1), 1, lane_iota_u8x, (sz_u8_t)chunk_bytes);
        svuint8_t const is_82_upper_third_u8x = sz_utf8_shift_value_up_pair_sve2_(
            previous_82_upper_second_u8x, is_82_upper_second_u8x, 1, lane_iota_u8x, (sz_u8_t)chunk_bytes);
        svuint8_t const is_83_upper_third_u8x = sz_utf8_shift_value_up_pair_sve2_(
            previous_83_upper_second_u8x, is_83_upper_second_u8x, 1, lane_iota_u8x, (sz_u8_t)chunk_bytes);

        svbool_t const is_upper_lead_b8x = svorr_b_z(all_b8x, is_82_upper_lead_b8x, is_83_upper_lead_b8x);
        svbool_t const is_upper_second_b8x = svorr_b_z(all_b8x, svcmpne_n_u8(all_b8x, is_82_upper_second_u8x, 0),
                                                       svcmpne_n_u8(all_b8x, is_83_upper_second_u8x, 0));

        // Rewrites: lead E1 -> E2, second 82/83 -> B4, third -0x20 (82, +0xE0 wrap) or +0x20 (83).
        svuint8_t folded_u8x = sz_utf8_fold_sve2_ascii_(source_u8x);
        folded_u8x = svsel_u8(is_upper_lead_b8x, svdup_n_u8(0xE2), folded_u8x);
        folded_u8x = svsel_u8(is_upper_second_b8x, svdup_n_u8(0xB4), folded_u8x);
        folded_u8x = svadd_n_u8_m(svcmpne_n_u8(all_b8x, is_82_upper_third_u8x, 0), folded_u8x, 0xE0);
        folded_u8x = svadd_n_u8_m(svcmpne_n_u8(all_b8x, is_83_upper_third_u8x, 0), folded_u8x, 0x20);

        svst1_u8(loaded_b8x, (sz_u8_t *)target + chunk_base, folded_u8x);
        previous_82_upper_lead_u8x = svdup_u8_z(is_82_upper_lead_b8x, 1);
        previous_83_upper_lead_u8x = svdup_u8_z(is_83_upper_lead_b8x, 1);
        previous_82_upper_second_u8x = is_82_upper_second_u8x;
        previous_83_upper_second_u8x = is_83_upper_second_u8x;
    }

    if (stop_lanes) return sz_utf8_fold_stop_boundary_serial_(stop_lanes, source);
    sz_u8_t const second_to_last_byte = (sz_u8_t)source[62], last_byte = (sz_u8_t)source[63];
    if (second_to_last_byte == 0xE1) return 62;
    if (last_byte == 0xE1) return 63;
    return 64;
}

/** @brief  Folds a 64-byte superchunk of caseless scripts mixed with SAFE guarded punctuation - the SVE2 twin
 *          of @ref sz_utf8_uncased_fold_neon_guarded_chunk_.
 *  @return Bytes consumed and written, or zero if the first character needs another path. */
SZ_HELPER_AUTO sz_size_t sz_utf8_uncased_fold_sve2_guarded_chunk_(sz_cptr_t source, sz_ptr_t target) {
    svbool_t const all_b8x = svptrue_b8();
    sz_size_t const chunk_bytes = svcntb() < 64 ? svcntb() : 64;
    sz_u64_t stop_lanes = 0;

    for (sz_size_t chunk_base = 0; chunk_base < 64; chunk_base += chunk_bytes) {
        sz_size_t const chunk_span = 64 - chunk_base < chunk_bytes ? 64 - chunk_base : chunk_bytes;
        svbool_t const loaded_b8x = svwhilelt_b8_u64(0, (sz_u64_t)chunk_span);
        svuint8_t const source_u8x = svld1_u8(loaded_b8x, (sz_u8_t const *)source + chunk_base);
        sz_size_t const peek_span = 64 - chunk_base > chunk_bytes ? chunk_bytes : 0;
        svuint8_t const peek_u8x = svld1_u8(svwhilelt_b8_u64(0, (sz_u64_t)peek_span),
                                            (sz_u8_t const *)source + chunk_base + (peek_span ? chunk_bytes : 0));
        svuint8_t const next_byte_u8x = svext_u8(source_u8x, peek_u8x, 1);

        svbool_t const is_caseless_two_b8x = svcmplt_n_u8(loaded_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0xD7), 0x09);
        svbool_t const is_three_b8x = svcmpeq_n_u8(loaded_b8x, svand_n_u8_x(all_b8x, source_u8x, 0xF0), 0xE0);
        svbool_t const is_continuation_b8x = svcmplt_n_u8(loaded_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0x80), 0x40);
        svbool_t const is_lead_b8x = svbic_b_z(loaded_b8x, svcmpge_n_u8(loaded_b8x, source_u8x, 0x80),
                                               is_continuation_b8x);
        svbool_t const is_foreign_lead_b8x = svbic_b_z(loaded_b8x, is_lead_b8x,
                                                       svorr_b_z(loaded_b8x, is_caseless_two_b8x, is_three_b8x));

        // E1 and EF are always case-aware; E2 only 80-83 is safe, EA all but 99-9F and AD-AE.
        svbool_t const is_e2_b8x = svcmpeq_n_u8(loaded_b8x, source_u8x, 0xE2);
        svbool_t const is_ea_b8x = svcmpeq_n_u8(loaded_b8x, source_u8x, 0xEA);
        svbool_t stop_b8x = svorr_b_z(loaded_b8x, is_foreign_lead_b8x,
                                      svorr_b_z(loaded_b8x, svcmpeq_n_u8(loaded_b8x, source_u8x, 0xE1),
                                                svcmpeq_n_u8(loaded_b8x, source_u8x, 0xEF)));
        stop_b8x = svorr_b_z(
            loaded_b8x, stop_b8x,
            svbic_b_z(loaded_b8x, is_e2_b8x, svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, next_byte_u8x, 0x80), 0x04)));
        stop_b8x = svorr_b_z(
            loaded_b8x, stop_b8x,
            svand_b_z(loaded_b8x, is_ea_b8x,
                      svorr_b_z(all_b8x, svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, next_byte_u8x, 0x99), 0x07),
                                svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, next_byte_u8x, 0xAD), 0x02))));
        stop_b8x = svorr_b_z(loaded_b8x, stop_b8x, sz_utf8_fold_sve2_malformed_lead_(source_u8x, peek_u8x));
        stop_lanes |= sz_utf8_rune_pred_to_u64_sve2_(svand_b_z(loaded_b8x, stop_b8x, loaded_b8x)) << chunk_base;

        svst1_u8(loaded_b8x, (sz_u8_t *)target + chunk_base, sz_utf8_fold_sve2_ascii_(source_u8x));
    }

    if (stop_lanes) return sz_utf8_fold_stop_boundary_serial_(stop_lanes, source);
    sz_u8_t const second_to_last_byte = (sz_u8_t)source[62], last_byte = (sz_u8_t)source[63];
    if ((second_to_last_byte & 0xF0) == 0xE0) return 62;
    if ((last_byte & 0xF0) == 0xE0 || (sz_u8_t)(last_byte - 0xD7) < 0x09) return 63;
    return 64;
}

SZ_API_COMPTIME sz_size_t sz_utf8_uncased_fold_sve2(sz_cptr_t source, sz_size_t source_length, sz_ptr_t target) {
    // The same 64-byte logical superchunk and family dispatch as the serial / NEON / Ice Lake kernels, so chunk
    // decisions stay comparable for differential debugging. The output buffer holds at least 3x the source, so
    // full-register stores with partial advance are safe anywhere inside the superchunk path.
    sz_ptr_t const target_start = target;
    svbool_t const all_b8x = svptrue_b8();
    sz_size_t const chunk_bytes = svcntb() < 64 ? svcntb() : 64;

    while (source_length >= 64) {
        // FAST PATH: pure ASCII decided per chunk before any classification.
        int any_non_ascii = 0;
        sz_u8_t lead_families = 0;
        for (sz_size_t chunk_base = 0; chunk_base < 64 && !any_non_ascii; chunk_base += chunk_bytes) {
            svbool_t const loaded_b8x = svwhilelt_b8_u64((sz_u64_t)chunk_base, 64);
            svuint8_t const source_u8x = svld1_u8(loaded_b8x, (sz_u8_t const *)source + chunk_base);
            any_non_ascii = svptest_any(loaded_b8x, svcmpge_n_u8(loaded_b8x, source_u8x, 0x80));
        }
        if (!any_non_ascii) {
            for (sz_size_t chunk_base = 0; chunk_base < 64; chunk_base += chunk_bytes) {
                svbool_t const loaded_b8x = svwhilelt_b8_u64((sz_u64_t)chunk_base, 64);
                svuint8_t const source_u8x = svld1_u8(loaded_b8x, (sz_u8_t const *)source + chunk_base);
                svst1_u8(loaded_b8x, (sz_u8_t *)target + chunk_base, sz_utf8_fold_sve2_ascii_(source_u8x));
            }
            target += 64, source += 64, source_length -= 64;
            continue;
        }

        // Classify lead bytes: the shared 64-entry family LUT through a chunked `svtbl` walk, OR-reduced with
        // one `svorv` per chunk - the scalable stand-in for NEON's `vqtbl4q` + byte reduction.
        for (sz_size_t chunk_base = 0; chunk_base < 64; chunk_base += chunk_bytes) {
            svbool_t const loaded_b8x = svwhilelt_b8_u64((sz_u64_t)chunk_base, 64);
            svuint8_t const source_u8x = svld1_u8(loaded_b8x, (sz_u8_t const *)source + chunk_base);
            svbool_t const is_continuation_b8x = svcmplt_n_u8(loaded_b8x, svsub_n_u8_x(all_b8x, source_u8x, 0x80),
                                                              0x40);
            svbool_t const is_lead_b8x = svbic_b_z(loaded_b8x, svcmpge_n_u8(loaded_b8x, source_u8x, 0x80),
                                                   is_continuation_b8x);
            svuint8_t const raw_families_u8x = sz_utf8_rune_lut_sve2_(sz_utf8_fold_lead_families_lut_, 64,
                                                                      svand_n_u8_x(all_b8x, source_u8x, 0x3F));
            svuint8_t const families_u8x = svand_u8_z(is_lead_b8x, raw_families_u8x, raw_families_u8x);
            lead_families |= (sz_u8_t)svorv_u8(loaded_b8x, families_u8x);
        }

        if (!(lead_families & ~sz_utf8_fold_lead_caseless_flag_k)) {
            sz_size_t const handled = sz_utf8_uncased_fold_sve2_caseless_chunk_(source, target);
            target += handled, source += handled, source_length -= handled;
            continue;
        }
        if (lead_families &
            (sz_utf8_fold_lead_latin_flag_k | sz_utf8_fold_lead_latin_extended_flag_k | sz_utf8_fold_lead_e1_flag_k)) {
            sz_size_t const handled = sz_utf8_uncased_fold_sve2_latin_chunk_(source, target);
            if (handled) {
                target += handled, source += handled, source_length -= handled;
                continue;
            }
        }
        if (lead_families & sz_utf8_fold_lead_e1_flag_k) {
            sz_size_t const handled = sz_utf8_uncased_fold_sve2_georgian_chunk_(source, target);
            if (handled) {
                target += handled, source += handled, source_length -= handled;
                continue;
            }
        }
        if (lead_families & sz_utf8_fold_lead_cyrillic_flag_k) {
            sz_size_t const handled = sz_utf8_uncased_fold_sve2_cyrillic_chunk_(source, target);
            if (handled) {
                target += handled, source += handled, source_length -= handled;
                continue;
            }
        }
        if (lead_families & sz_utf8_fold_lead_greek_flag_k) {
            sz_size_t const handled = sz_utf8_uncased_fold_sve2_greek_chunk_(source, target);
            if (handled) {
                target += handled, source += handled, source_length -= handled;
                continue;
            }
        }
        if (lead_families & sz_utf8_fold_lead_guarded_flag_k) {
            sz_size_t const handled = sz_utf8_uncased_fold_sve2_guarded_chunk_(source, target);
            if (handled) {
                target += handled, source += handled, source_length -= handled;
                continue;
            }
        }
        if (lead_families & sz_utf8_fold_lead_complex_flag_k) {
            sz_size_t const handled = sz_utf8_uncased_fold_sve2_armenian_chunk_(source, target);
            if (handled) {
                target += handled, source += handled, source_length -= handled;
                continue;
            }
        }
        // One rune through the serial logic: byte-for-byte the serial reference, just slower.
        sz_rune_t rune;
        sz_rune_length_t const rune_length = sz_rune_decode(source, source + source_length, &rune);
        if (rune_length == sz_rune_invalid_k) {
            *(sz_u8_t *)target = *(sz_u8_t const *)source;
            target += 1, source += 1, source_length -= 1;
            continue;
        }
        sz_rune_t folded_runes[3];
        sz_size_t const folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
        for (sz_size_t rune_index = 0; rune_index != folded_count; ++rune_index)
            target += sz_rune_encode(folded_runes[rune_index], (sz_u8_t *)target);
        source += rune_length, source_length -= rune_length;
    }

    if (source_length) target += sz_utf8_uncased_fold_serial(source, source_length, target);
    return (sz_size_t)(target - target_start);
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

#endif // STRINGZILLA_UTF8_UNCASED_FOLD_SVE2_H_
