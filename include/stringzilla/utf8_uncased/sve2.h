/**
 *  @brief Arm SVE2 backend for UTF-8 case-insensitive substring search.
 *  @file include/stringzilla/utf8_uncased/sve2.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_SVE2_H_
#define STRINGZILLA_UTF8_UNCASED_SVE2_H_

#include "stringzilla/utf8_uncased/serial.h"
#include "stringzilla/utf8_runes/sve2.h"
#include "stringzilla/find/sve.h" // `sz_find_sve` for caseless needles

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

/*  The scalable twin of the NEON uncased search. One Z register is the scan chunk, so the NEON two-register
 *  boundary dance disappears: the previous-byte view is a single `svtbl` down-shift (lane 0 reads zero, the
 *  chunk-start convention every backend shares), the next-byte view one `svext`, and tail chunks load through
 *  predicated `svld1` - no zero-padded stack buffers at all. Candidate masks lower once per chunk through the
 *  predicate bridge and iterate through the shared serial candidate pop. */

/** @brief Bytes in the unsigned range [start, start + span): one wrap-around subtract + compare. */
SZ_HELPER_INLINE svbool_t sz_utf8_uncased_sve2_in_range_(svuint8_t values_u8x, sz_u8_t start, sz_u8_t span) {
    svbool_t const all_b8x = svptrue_b8();
    return svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, values_u8x, start), span);
}

/** @brief The previous byte per lane (lane 0 reads zero - a match never needs its lead's predecessor). */
SZ_HELPER_INLINE svuint8_t sz_utf8_uncased_sve2_previous_(svuint8_t values_u8x) {
    svbool_t const all_b8x = svptrue_b8();
    return svtbl_u8(values_u8x, svsub_n_u8_x(all_b8x, svindex_u8(0, 1), 1));
}

/** @brief The next byte per lane (the last lane reads zero; the overlapping scan re-derives it next chunk). */
SZ_HELPER_INLINE svuint8_t sz_utf8_uncased_sve2_next_(svuint8_t values_u8x) {
    return svext_u8(values_u8x, svdup_n_u8(0), 1);
}

/** @brief Folds ASCII A-Z down to a-z, leaving every other byte unchanged. */
SZ_HELPER_INLINE svuint8_t sz_utf8_uncased_search_sve2_ascii_fold_(svuint8_t text_u8x) {
    svbool_t const all_b8x = svptrue_b8();
    return svadd_n_u8_m(svcmplt_n_u8(all_b8x, svsub_n_u8_x(all_b8x, text_u8x, 'A'), 26), text_u8x, 0x20);
}

/** @brief View-signature ASCII fold: only the raw bytes matter. */
SZ_HELPER_NOINLINE svuint8_t sz_utf8_uncased_search_sve2_ascii_fold_views_(svuint8_t text_u8x, svuint8_t previous_u8x,
                                                                           svuint8_t previous2_u8x,
                                                                           svuint8_t next_u8x) {
    sz_unused_(previous_u8x);
    sz_unused_(previous2_u8x);
    sz_unused_(next_u8x);
    return sz_utf8_uncased_search_sve2_ascii_fold_(text_u8x);
}

#pragma region Scripted Uncased Find

/** @brief Folds one sub-vector of haystack text using script-specific rules; the driver supplies the
 *         previous-byte, previous-previous-byte, and next-byte views with cross-sub-vector carries. */
typedef svuint8_t (*sz_utf8_uncased_fold_sve2_t_)(svuint8_t text_u8x, svuint8_t previous_u8x, svuint8_t previous2_u8x,
                                                  svuint8_t next_u8x);

/** @brief Non-zero when the sub-vector holds "danger" characters that fold to a different byte width. */
typedef int (*sz_utf8_uncased_alarm_sve2_t_)(svuint8_t text_u8x, svuint8_t previous_u8x, svuint8_t next_u8x,
                                             svbool_t loaded_b8x);

/**
 *  @brief  Shared scan loop behind all script-specific uncased searches - the SVE2 twin of
 *          @ref sz_utf8_uncased_search_neon_scripted_ over one-vector chunks. The driver is force-inlined into
 *          each thin per-script wrapper, so the callbacks resolve to direct calls.
 */
SZ_HELPER_INLINE sz_cptr_t sz_utf8_uncased_search_sve2_scripted_( //
    sz_utf8_uncased_fold_sve2_t_ fold,                            //
    sz_utf8_uncased_alarm_sve2_t_ alarm,                          //
    sz_cptr_t haystack, sz_size_t haystack_length,                //
    sz_cptr_t needle, sz_size_t needle_length,                    //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,     //
    sz_size_t *matched_length) {

    sz_assert_(needle_metadata && "needle_metadata must be provided");
    sz_assert_(needle_metadata->folded_slice_length > 0 && "folded window must be non-empty");

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_size_t const vector_bytes = svcntb();
    // Two sub-vectors per chunk (like NEON's 32-byte pair) so long folded windows keep many valid starts;
    // the candidate masks live in a u64, capping the chunk at 64 bytes on very wide registers.
    sz_size_t const chunk_capacity = vector_bytes * 2 < 64 ? vector_bytes * 2 : 64;
    sz_size_t const sub_vectors = chunk_capacity > vector_bytes ? 2 : 1;
    svuint8_t const lane_iota_u8x = svindex_u8(0, 1);
    sz_assert_(folded_window_length <= 16 && "expect folded needle part to fit in 16 bytes");

    svbool_t const all_b8x = svptrue_b8();
    svbool_t const window_b8x = svwhilelt_b8_u64(0, (sz_u64_t)folded_window_length);
    svuint8_t const needle_window_u8x = svld1_u8(svwhilelt_b8_u64(0, 16),
                                                 (sz_u8_t const *)needle_metadata->folded_slice);

    sz_size_t const offset_second = needle_metadata->probe_second;
    sz_size_t const offset_third = needle_metadata->probe_third;
    sz_size_t const offset_last = folded_window_length - 1;
    sz_u8_t const probe_first = needle_metadata->folded_slice[0];
    sz_u8_t const probe_second = needle_metadata->folded_slice[offset_second];
    sz_u8_t const probe_third = needle_metadata->folded_slice[offset_third];
    sz_u8_t const probe_last = needle_metadata->folded_slice[offset_last];

    sz_rune_t needle_first_safe_folded_rune = 0;
    if (alarm) sz_rune_decode_unchecked((sz_cptr_t)(needle_metadata->folded_slice), &needle_first_safe_folded_rune);

    sz_cptr_t haystack_ptr = haystack;
    while (haystack_ptr < haystack_end) {
        sz_size_t const available = (sz_size_t)(haystack_end - haystack_ptr);
        if (available < folded_window_length) break;

        sz_size_t const chunk_size = available < chunk_capacity ? available : chunk_capacity;
        sz_size_t const valid_starts = chunk_size - folded_window_length + 1;
        sz_size_t const step = !alarm ? valid_starts : valid_starts > 2 ? valid_starts - 2 : 1;
        sz_size_t const low_size = chunk_size < vector_bytes ? chunk_size : vector_bytes;
        sz_size_t const high_size = chunk_size > vector_bytes ? chunk_size - vector_bytes : 0;
        svbool_t const low_b8x = svwhilelt_b8_u64(0, (sz_u64_t)low_size);
        svbool_t const high_b8x = svwhilelt_b8_u64(0, (sz_u64_t)high_size);
        svuint8_t const low_u8x = svld1_u8(low_b8x, (sz_u8_t const *)haystack_ptr);
        svuint8_t const high_u8x = svld1_u8(high_b8x, (sz_u8_t const *)haystack_ptr + (high_size ? vector_bytes : 0));

        // Driver-built neighbour views with the cross-sub-vector carries the fold semantics require.
        svuint8_t const low_previous_u8x = sz_utf8_uncased_sve2_previous_(low_u8x);
        svuint8_t const low_previous2_u8x = sz_utf8_uncased_sve2_previous_(low_previous_u8x);
        svuint8_t const low_next_u8x = svext_u8(low_u8x, high_u8x, 1);
        svuint8_t const high_previous_u8x = sz_utf8_shift_value_up_pair_sve2_(low_u8x, high_u8x, 1, lane_iota_u8x,
                                                                              (sz_u8_t)vector_bytes);
        svuint8_t const high_previous2_u8x = sz_utf8_shift_value_up_pair_sve2_(low_u8x, high_u8x, 2, lane_iota_u8x,
                                                                               (sz_u8_t)vector_bytes);
        svuint8_t const high_next_u8x = svext_u8(high_u8x, svdup_n_u8(0), 1);

        if (alarm && (alarm(low_u8x, low_previous_u8x, low_next_u8x, low_b8x) ||
                      (sub_vectors > 1 && alarm(high_u8x, high_previous_u8x, high_next_u8x, high_b8x)))) {
            sz_cptr_t match = sz_utf8_uncased_search_in_danger_zone_( //
                haystack, haystack_length, needle, needle_length, haystack_ptr, chunk_size,
                needle_first_safe_folded_rune, needle_metadata->offset_in_unfolded, matched_length);
            if (match) return match;
            haystack_ptr += step;
            continue;
        }

        // Fold once, then filter candidates with 4 probe positions: scalar shifts over the lowered
        // candidate mask substitute the Ice Lake k-mask shifts bit-for-bit.
        svuint8_t const folded_low_u8x = fold(low_u8x, low_previous_u8x, low_previous2_u8x, low_next_u8x);
        svuint8_t const folded_high_u8x = fold(high_u8x, high_previous_u8x, high_previous2_u8x, high_next_u8x);
        sz_u64_t matches = sz_utf8_rune_pred_to_u64_sve2_(svcmpeq_n_u8(all_b8x, folded_low_u8x, probe_first)) |
                           (sz_utf8_rune_pred_to_u64_sve2_(svcmpeq_n_u8(all_b8x, folded_high_u8x, probe_first))
                            << vector_bytes);
        matches &= (sz_utf8_rune_pred_to_u64_sve2_(svcmpeq_n_u8(all_b8x, folded_low_u8x, probe_second)) |
                    (sz_utf8_rune_pred_to_u64_sve2_(svcmpeq_n_u8(all_b8x, folded_high_u8x, probe_second))
                     << vector_bytes)) >>
                   offset_second;
        matches &= (sz_utf8_rune_pred_to_u64_sve2_(svcmpeq_n_u8(all_b8x, folded_low_u8x, probe_third)) |
                    (sz_utf8_rune_pred_to_u64_sve2_(svcmpeq_n_u8(all_b8x, folded_high_u8x, probe_third))
                     << vector_bytes)) >>
                   offset_third;
        matches &= (sz_utf8_rune_pred_to_u64_sve2_(svcmpeq_n_u8(all_b8x, folded_low_u8x, probe_last)) |
                    (sz_utf8_rune_pred_to_u64_sve2_(svcmpeq_n_u8(all_b8x, folded_high_u8x, probe_last))
                     << vector_bytes)) >>
                   offset_last;
        matches &= sz_u64_mask_until_serial_(valid_starts);

        while (matches) {
            sz_size_t const candidate_offset = sz_utf8_uncased_pop_candidate_(&matches);
            sz_cptr_t const haystack_candidate_ptr = haystack_ptr + candidate_offset;

            // Re-fold the candidate window through the SAME per-chunk semantics: a predicated load zeroes
            // every byte past the window, so the fold sees a zero predecessor and successor at the edges.
            sz_size_t const candidate_available = (sz_size_t)(haystack_end - haystack_candidate_ptr);
            svbool_t const window_load_b8x = svwhilelt_b8_u64(
                0, (sz_u64_t)(candidate_available < folded_window_length ? candidate_available : folded_window_length));
            svuint8_t const window_u8x = svld1_u8(window_load_b8x, (sz_u8_t const *)haystack_candidate_ptr);
            svuint8_t const window_previous_u8x = sz_utf8_uncased_sve2_previous_(window_u8x);
            svuint8_t const folded_window_u8x = fold(window_u8x, window_previous_u8x,
                                                     sz_utf8_uncased_sve2_previous_(window_previous_u8x),
                                                     sz_utf8_uncased_sve2_next_(window_u8x));
            svbool_t const equal_b8x = svcmpeq_u8(window_b8x, folded_window_u8x, needle_window_u8x);
            if (svcntp_b8(window_b8x, equal_b8x) != folded_window_length) continue;

            sz_cptr_t match = sz_utf8_uncased_verify_match_(                                               //
                haystack, haystack_length, needle, needle_length,                                          //
                haystack_candidate_ptr - haystack, needle_metadata->folded_slice_length,                   //
                needle_metadata->offset_in_unfolded,                                                       //
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, //
                matched_length);
            if (match) return match;
        }
        haystack_ptr += step;
    }

    if (alarm && haystack_ptr < haystack_end) {
        sz_cptr_t match = sz_utf8_uncased_search_in_danger_zone_( //
            haystack, haystack_length, needle, needle_length, haystack_ptr, (sz_size_t)(haystack_end - haystack_ptr),
            needle_first_safe_folded_rune, needle_metadata->offset_in_unfolded, matched_length);
        if (match) return match;
    }

    return SZ_NULL_CHAR;
}

/** @brief 3-probe ASCII uncased search: probes at 0, mid, last cover ALL bytes of windows up to 3 bytes,
 *         so candidates skip window verification and go straight to head/tail validation. */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_sve2_ascii_3probe_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                  //
    sz_cptr_t needle, sz_size_t needle_length,                      //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,       //
    sz_size_t *matched_length) {

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_size_t const vector_bytes = svcntb();
    sz_size_t const chunk_capacity = vector_bytes * 2 < 64 ? vector_bytes * 2 : 64;
    svbool_t const all_b8x = svptrue_b8();

    sz_size_t const offset_second = folded_window_length / 2;
    sz_size_t const offset_last = folded_window_length - 1;
    sz_u8_t const probe_first = needle_metadata->folded_slice[0];
    sz_u8_t const probe_second = needle_metadata->folded_slice[offset_second];
    sz_u8_t const probe_last = needle_metadata->folded_slice[offset_last];

    sz_cptr_t haystack_ptr = haystack;
    while (haystack_ptr < haystack_end) {
        sz_size_t const available = (sz_size_t)(haystack_end - haystack_ptr);
        if (available < folded_window_length) break;
        sz_size_t const chunk_size = available < chunk_capacity ? available : chunk_capacity;
        sz_size_t const valid_starts = chunk_size - folded_window_length + 1;

        sz_size_t const low_size = chunk_size < vector_bytes ? chunk_size : vector_bytes;
        sz_size_t const high_size = chunk_size > vector_bytes ? chunk_size - vector_bytes : 0;
        svuint8_t const folded_low_u8x = sz_utf8_uncased_search_sve2_ascii_fold_(
            svld1_u8(svwhilelt_b8_u64(0, (sz_u64_t)low_size), (sz_u8_t const *)haystack_ptr));
        svuint8_t const folded_high_u8x = sz_utf8_uncased_search_sve2_ascii_fold_(svld1_u8(
            svwhilelt_b8_u64(0, (sz_u64_t)high_size), (sz_u8_t const *)haystack_ptr + (high_size ? vector_bytes : 0)));

        sz_u64_t matches = sz_utf8_rune_pred_to_u64_sve2_(svcmpeq_n_u8(all_b8x, folded_low_u8x, probe_first)) |
                           (sz_utf8_rune_pred_to_u64_sve2_(svcmpeq_n_u8(all_b8x, folded_high_u8x, probe_first))
                            << vector_bytes);
        matches &= (sz_utf8_rune_pred_to_u64_sve2_(svcmpeq_n_u8(all_b8x, folded_low_u8x, probe_second)) |
                    (sz_utf8_rune_pred_to_u64_sve2_(svcmpeq_n_u8(all_b8x, folded_high_u8x, probe_second))
                     << vector_bytes)) >>
                   offset_second;
        matches &= (sz_utf8_rune_pred_to_u64_sve2_(svcmpeq_n_u8(all_b8x, folded_low_u8x, probe_last)) |
                    (sz_utf8_rune_pred_to_u64_sve2_(svcmpeq_n_u8(all_b8x, folded_high_u8x, probe_last))
                     << vector_bytes)) >>
                   offset_last;
        matches &= sz_u64_mask_until_serial_(valid_starts);

        while (matches) {
            sz_size_t const candidate_offset = sz_utf8_uncased_pop_candidate_(&matches);
            sz_cptr_t match = sz_utf8_uncased_verify_match_(                                               //
                haystack, haystack_length, needle, needle_length,                                          //
                (haystack_ptr + candidate_offset) - haystack, folded_window_length,                        //
                needle_metadata->offset_in_unfolded,                                                       //
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, //
                matched_length);
            if (match) return match;
        }
        haystack_ptr += valid_starts;
    }
    return SZ_NULL_CHAR;
}

#pragma endregion Scripted Uncased Find

#pragma region Script Folds and Alarms

/** @brief Western European fold: ASCII, Latin-1 uppercase +0x20 (minus multiply sign), Eszett to "ss". */
SZ_HELPER_NOINLINE svuint8_t sz_utf8_uncased_search_sve2_western_europe_fold_(svuint8_t text_u8x,
                                                                              svuint8_t previous_u8x,
                                                                              svuint8_t previous2_u8x,
                                                                              svuint8_t next_u8x) {
    sz_unused_(previous2_u8x);
    svbool_t const all_b8x = svptrue_b8();
    svuint8_t result_u8x = sz_utf8_uncased_search_sve2_ascii_fold_(text_u8x);
    svbool_t const after_c3_b8x = svcmpeq_n_u8(all_b8x, previous_u8x, 0xC3);

    svbool_t const eszett_second_b8x = svand_b_z(all_b8x, after_c3_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0x9F));
    svbool_t const eszett_lead_b8x = svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xC3),
                                               svcmpeq_n_u8(all_b8x, next_u8x, 0x9F));
    result_u8x = svsel_u8(svorr_b_z(all_b8x, eszett_second_b8x, eszett_lead_b8x), svdup_n_u8('s'), result_u8x);

    svbool_t const latin1_upper_b8x = svbic_b_z(
        all_b8x, svand_b_z(all_b8x, after_c3_b8x, sz_utf8_uncased_sve2_in_range_(text_u8x, 0x80, 0x1F)),
        svorr_b_z(all_b8x, eszett_second_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0x97)));
    return svadd_n_u8_m(latin1_upper_b8x, result_u8x, 0x20);
}

/** @brief Western European alarm: width-changing folds route to the serial danger-zone scanner. */
SZ_HELPER_NOINLINE int sz_utf8_uncased_search_sve2_western_europe_alarm_(svuint8_t text_u8x, svuint8_t previous_u8x,
                                                                         svuint8_t next_u8x, svbool_t loaded_b8x) {
    svbool_t const all_b8x = svptrue_b8();
    svbool_t const after_c5_b8x = svcmpeq_n_u8(all_b8x, previous_u8x, 0xC5);

    svbool_t danger_b8x = svand_b_z( // Capital Sharp S & co (E1 BA 96-9E)
        all_b8x, svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xBA), svcmpeq_n_u8(all_b8x, previous_u8x, 0xE1)),
        sz_utf8_uncased_sve2_in_range_(next_u8x, 0x96, 0x09));
    danger_b8x = svorr_b_z( // Kelvin / Angstrom (E2 84 AA/AB)
        all_b8x, danger_b8x,
        svand_b_z(all_b8x,
                  svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0x84), svcmpeq_n_u8(all_b8x, previous_u8x, 0xE2)),
                  svorr_b_z(all_b8x, svcmpeq_n_u8(all_b8x, next_u8x, 0xAA), svcmpeq_n_u8(all_b8x, next_u8x, 0xAB))));
    danger_b8x = svorr_b_z( // Ligatures (EF AC xx)
        all_b8x, danger_b8x,
        svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xAC), svcmpeq_n_u8(all_b8x, previous_u8x, 0xEF)));
    danger_b8x = svorr_b_z( // Long S (C5 BF) and 'Ÿ' (C5 B8)
        all_b8x, danger_b8x,
        svand_b_z(all_b8x, after_c5_b8x,
                  svorr_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xBF), svcmpeq_n_u8(all_b8x, text_u8x, 0xB8))));
    danger_b8x = svorr_b_z( // Sharp S (C3 9F)
        all_b8x, danger_b8x,
        svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0x9F), svcmpeq_n_u8(all_b8x, previous_u8x, 0xC3)));
    return svptest_any(loaded_b8x, svand_b_z(loaded_b8x, danger_b8x, loaded_b8x));
}

/** @brief Central European fold: Latin-1 +0x20 and the Latin Extended-A parity via the shared delta LUTs. */
SZ_HELPER_NOINLINE svuint8_t sz_utf8_uncased_search_sve2_central_europe_fold_(svuint8_t text_u8x,
                                                                              svuint8_t previous_u8x,
                                                                              svuint8_t previous2_u8x,
                                                                              svuint8_t next_u8x) {
    sz_unused_(previous2_u8x);
    sz_unused_(next_u8x);
    svbool_t const all_b8x = svptrue_b8();
    svuint8_t result_u8x = sz_utf8_uncased_search_sve2_ascii_fold_(text_u8x);
    svbool_t const after_c3_b8x = svcmpeq_n_u8(all_b8x, previous_u8x, 0xC3);
    svbool_t const after_c4_b8x = svcmpeq_n_u8(all_b8x, previous_u8x, 0xC4);
    svbool_t const after_c5_b8x = svcmpeq_n_u8(all_b8x, previous_u8x, 0xC5);
    svbool_t const continuation_b8x = sz_utf8_uncased_sve2_in_range_(text_u8x, 0x80, 0x40);

    svbool_t const latin1_upper_b8x = svbic_b_z(
        all_b8x, svand_b_z(all_b8x, after_c3_b8x, sz_utf8_uncased_sve2_in_range_(text_u8x, 0x80, 0x1F)),
        svcmpeq_n_u8(all_b8x, text_u8x, 0x97));
    result_u8x = svadd_n_u8_m(latin1_upper_b8x, result_u8x, 0x20);

    svuint8_t const delta_indices_u8x = svand_n_u8_x(all_b8x, text_u8x, 0x3F);
    svuint8_t deltas_u8x = svdup_n_u8(0);
    deltas_u8x = svsel_u8(after_c4_b8x,
                          sz_utf8_rune_lut_sve2_(sz_utf8_uncased_central_c4_deltas_lut_, 64, delta_indices_u8x),
                          deltas_u8x);
    deltas_u8x = svsel_u8(after_c5_b8x,
                          sz_utf8_rune_lut_sve2_(sz_utf8_uncased_central_c5_deltas_lut_, 64, delta_indices_u8x),
                          deltas_u8x);
    deltas_u8x = svand_u8_z(continuation_b8x, deltas_u8x, deltas_u8x);
    return svadd_u8_x(all_b8x, result_u8x, deltas_u8x);
}

/** @brief Central European alarm: cross-block and width-changing folds. */
SZ_HELPER_NOINLINE int sz_utf8_uncased_search_sve2_central_europe_alarm_(svuint8_t text_u8x, svuint8_t previous_u8x,
                                                                         svuint8_t next_u8x, svbool_t loaded_b8x) {
    sz_unused_(next_u8x);
    svbool_t const all_b8x = svptrue_b8();
    svbool_t const after_c4_b8x = svcmpeq_n_u8(all_b8x, previous_u8x, 0xC4);
    svbool_t const after_c5_b8x = svcmpeq_n_u8(all_b8x, previous_u8x, 0xC5);

    svbool_t danger_b8x = svand_b_z( // Kelvin (E2 84 xx)
        all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0x84), svcmpeq_n_u8(all_b8x, previous_u8x, 0xE2));
    danger_b8x = svorr_b_z( // Sharp S (C3 9F)
        all_b8x, danger_b8x,
        svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0x9F), svcmpeq_n_u8(all_b8x, previous_u8x, 0xC3)));
    danger_b8x = svorr_b_z( // Dotted I (C4 B0) and 'Ŀ' (C4 BF)
        all_b8x, danger_b8x,
        svand_b_z(all_b8x, after_c4_b8x,
                  svorr_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xB0), svcmpeq_n_u8(all_b8x, text_u8x, 0xBF))));
    danger_b8x = svorr_b_z( // Long S (C5 BF) and 'Ÿ' (C5 B8)
        all_b8x, danger_b8x,
        svand_b_z(all_b8x, after_c5_b8x,
                  svorr_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xBF), svcmpeq_n_u8(all_b8x, text_u8x, 0xB8))));
    danger_b8x = svorr_b_z( // Ligatures (EF AC xx)
        all_b8x, danger_b8x,
        svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xAC), svcmpeq_n_u8(all_b8x, previous_u8x, 0xEF)));
    return svptest_any(loaded_b8x, svand_b_z(loaded_b8x, danger_b8x, loaded_b8x));
}

/** @brief Cyrillic fold: second-byte high-nibble offsets after D0 plus the D0 -> D1 lead promotion. */
SZ_HELPER_NOINLINE svuint8_t sz_utf8_uncased_search_sve2_cyrillic_fold_(svuint8_t text_u8x, svuint8_t previous_u8x,
                                                                        svuint8_t previous2_u8x, svuint8_t next_u8x) {
    sz_unused_(previous2_u8x);
    static sz_u8_t const cyrillic_offset_lut[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x20, 0xE0, 0, 0, 0, 0, 0};
    svbool_t const all_b8x = svptrue_b8();
    svuint8_t const offsets_lut_u8x = svld1_u8(svwhilelt_b8_u64(0, 16), cyrillic_offset_lut);
    svuint8_t result_u8x = sz_utf8_uncased_search_sve2_ascii_fold_(text_u8x);

    svbool_t const after_d0_b8x = svcmpeq_n_u8(all_b8x, previous_u8x, 0xD0);
    svuint8_t const offsets_u8x = svtbl_u8(offsets_lut_u8x, svlsr_n_u8_x(all_b8x, text_u8x, 4));
    result_u8x = svadd_u8_x(all_b8x, result_u8x, svand_u8_z(after_d0_b8x, offsets_u8x, offsets_u8x));

    svbool_t const needs_d1_b8x = svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xD0),
                                            svorr_b_z(all_b8x, sz_utf8_uncased_sve2_in_range_(next_u8x, 0x80, 0x10),
                                                      sz_utf8_uncased_sve2_in_range_(next_u8x, 0xA0, 0x10)));
    return svadd_n_u8_m(needs_d1_b8x, result_u8x, 0x01);
}

/** @brief Cyrillic alarm: Cyrillic Extended-C (E1 B2 80-88) folds into basic 2-byte letters. */
SZ_HELPER_NOINLINE int sz_utf8_uncased_search_sve2_cyrillic_alarm_(svuint8_t text_u8x, svuint8_t previous_u8x,
                                                                   svuint8_t next_u8x, svbool_t loaded_b8x) {
    svbool_t const all_b8x = svptrue_b8();
    svbool_t const danger_b8x = svand_b_z(
        all_b8x, svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xB2), svcmpeq_n_u8(all_b8x, previous_u8x, 0xE1)),
        sz_utf8_uncased_sve2_in_range_(next_u8x, 0x80, 0x09));
    return svptest_any(loaded_b8x, svand_b_z(loaded_b8x, danger_b8x, loaded_b8x));
}

/** @brief Greek fold: the shared CE delta / promotion LUTs, final sigma, and the micro sign join. */
SZ_HELPER_NOINLINE svuint8_t sz_utf8_uncased_search_sve2_greek_fold_(svuint8_t text_u8x, svuint8_t previous_u8x,
                                                                     svuint8_t previous2_u8x, svuint8_t next_u8x) {
    sz_unused_(previous2_u8x);
    svbool_t const all_b8x = svptrue_b8();
    svuint8_t result_u8x = sz_utf8_uncased_search_sve2_ascii_fold_(text_u8x);
    svbool_t const after_ce_b8x = svcmpeq_n_u8(all_b8x, previous_u8x, 0xCE);
    svbool_t const after_cf_b8x = svcmpeq_n_u8(all_b8x, previous_u8x, 0xCF);
    svbool_t const after_c2_b8x = svcmpeq_n_u8(all_b8x, previous_u8x, 0xC2);
    svbool_t const continuation_b8x = sz_utf8_uncased_sve2_in_range_(text_u8x, 0x80, 0x40);
    svbool_t const after_ce_cont_b8x = svand_b_z(all_b8x, after_ce_b8x, continuation_b8x);
    svuint8_t const delta_indices_u8x = svand_n_u8_x(all_b8x, text_u8x, 0x3F);

    svuint8_t const ce_delta_raw_u8x = sz_utf8_rune_lut_sve2_(sz_utf8_uncased_greek_ce_deltas_lut_, 64,
                                                              delta_indices_u8x);
    svuint8_t const ce_delta_u8x = svand_u8_z(after_ce_cont_b8x, ce_delta_raw_u8x, ce_delta_raw_u8x);

    svbool_t const final_sigma_b8x = svand_b_z(all_b8x, after_cf_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0x82));
    svbool_t const micro_second_b8x = svand_b_z(all_b8x, after_c2_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xB5));

    svuint8_t offsets_u8x = ce_delta_u8x;
    offsets_u8x = svadd_n_u8_m(final_sigma_b8x, offsets_u8x, 0x01);
    offsets_u8x = svadd_n_u8_m(micro_second_b8x, offsets_u8x, 0x07);
    // Lead rewrites anchor on the lead's own carried next byte, so they survive sub-vector boundaries:
    // CE -> CF (+1) where the next byte's promotion entry is set, C2 -> CE (+0x0C) before B5.
    svbool_t const next_continuation_b8x = sz_utf8_uncased_sve2_in_range_(next_u8x, 0x80, 0x40);
    svuint8_t const next_promote_u8x = sz_utf8_rune_lut_sve2_(sz_utf8_uncased_greek_ce_promotes_lut_, 64,
                                                              svand_n_u8_x(all_b8x, next_u8x, 0x3F));
    svbool_t const promotes_lead_b8x = svand_b_z(
        all_b8x, svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xCE), next_continuation_b8x),
        svcmpne_n_u8(all_b8x, next_promote_u8x, 0));
    offsets_u8x = svadd_n_u8_m(promotes_lead_b8x, offsets_u8x, 0x01);
    svbool_t const micro_lead_b8x = svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xC2),
                                              svcmpeq_n_u8(all_b8x, next_u8x, 0xB5));
    offsets_u8x = svadd_n_u8_m(micro_lead_b8x, offsets_u8x, 0x0C);
    return svadd_u8_x(all_b8x, result_u8x, offsets_u8x);
}

/** @brief Greek alarm: expanding diaeresis vowels, Greek symbols, the Ohm sign, polytonic/archaic leads. */
SZ_HELPER_NOINLINE int sz_utf8_uncased_search_sve2_greek_alarm_(svuint8_t text_u8x, svuint8_t previous_u8x,
                                                                svuint8_t next_u8x, svbool_t loaded_b8x) {
    sz_unused_(next_u8x);
    svbool_t const all_b8x = svptrue_b8();

    svbool_t danger_b8x = svand_b_z( // 'ΐ', 'ΰ' (CE 90 / CE B0)
        all_b8x, svcmpeq_n_u8(all_b8x, previous_u8x, 0xCE),
        svorr_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0x90), svcmpeq_n_u8(all_b8x, text_u8x, 0xB0)));
    svbool_t const cf_seconds_b8x = svorr_b_z(
        all_b8x, svorr_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0x90), svcmpeq_n_u8(all_b8x, text_u8x, 0x91)),
        svorr_b_z(
            all_b8x, svorr_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0x95), svcmpeq_n_u8(all_b8x, text_u8x, 0x96)),
            svorr_b_z(
                all_b8x,
                svorr_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xB0), svcmpeq_n_u8(all_b8x, text_u8x, 0xB1)),
                svorr_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xB4), svcmpeq_n_u8(all_b8x, text_u8x, 0xB5)))));
    danger_b8x = svorr_b_z( // Greek symbols (CF 9x / CF Bx)
        all_b8x, danger_b8x, svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, previous_u8x, 0xCF), cf_seconds_b8x));
    danger_b8x = svorr_b_z( // Ohm sign (E2 84 xx)
        all_b8x, danger_b8x,
        svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0x84), svcmpeq_n_u8(all_b8x, previous_u8x, 0xE2)));
    danger_b8x = svorr_b_z( // Blanket polytonic & archaic leads
        all_b8x, danger_b8x,
        svorr_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xE1), svcmpeq_n_u8(all_b8x, text_u8x, 0xCD)));
    return svptest_any(loaded_b8x, svand_b_z(loaded_b8x, danger_b8x, loaded_b8x));
}

/** @brief Armenian fold: three second-byte offsets on disjoint lanes plus the two +1 lead promotions. */
SZ_HELPER_NOINLINE svuint8_t sz_utf8_uncased_search_sve2_armenian_fold_(svuint8_t text_u8x, svuint8_t previous_u8x,
                                                                        svuint8_t previous2_u8x, svuint8_t next_u8x) {
    sz_unused_(previous2_u8x);
    svbool_t const all_b8x = svptrue_b8();
    svuint8_t result_u8x = sz_utf8_uncased_search_sve2_ascii_fold_(text_u8x);
    svbool_t const after_d4_b8x = svcmpeq_n_u8(all_b8x, previous_u8x, 0xD4);
    svbool_t const after_d5_b8x = svcmpeq_n_u8(all_b8x, previous_u8x, 0xD5);

    svbool_t const d4_upper_b8x = svand_b_z(all_b8x, after_d4_b8x,
                                            sz_utf8_uncased_sve2_in_range_(text_u8x, 0xB1, 0x4F));
    svbool_t const d5_low_b8x = svand_b_z(all_b8x, after_d5_b8x, sz_utf8_uncased_sve2_in_range_(text_u8x, 0x80, 0x10));
    svbool_t const d5_high_b8x = svand_b_z(all_b8x, after_d5_b8x, sz_utf8_uncased_sve2_in_range_(text_u8x, 0x90, 0x07));
    svbool_t const minus_10_b8x = svorr_b_z(all_b8x, d4_upper_b8x, d5_high_b8x);

    result_u8x = svadd_n_u8_m(minus_10_b8x, result_u8x, 0xF0);
    result_u8x = svadd_n_u8_m(d5_low_b8x, result_u8x, 0x30);
    // Lead +1 rewrites anchor on the lead's own carried next byte, so they survive sub-vector boundaries:
    // D4 -> D5 where next is B1-BF, D5 -> D6 where next is 90-96.
    svbool_t const promotes_d4_b8x = svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xD4),
                                               sz_utf8_uncased_sve2_in_range_(next_u8x, 0xB1, 0x4F));
    svbool_t const promotes_d5_b8x = svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xD5),
                                               sz_utf8_uncased_sve2_in_range_(next_u8x, 0x90, 0x07));
    return svadd_n_u8_m(svorr_b_z(all_b8x, promotes_d4_b8x, promotes_d5_b8x), result_u8x, 0x01);
}

/** @brief Armenian alarm: the Ech-Yiwn ligature (D6 87) and the presentation-form ligatures (EF AC xx). */
SZ_HELPER_NOINLINE int sz_utf8_uncased_search_sve2_armenian_alarm_(svuint8_t text_u8x, svuint8_t previous_u8x,
                                                                   svuint8_t next_u8x, svbool_t loaded_b8x) {
    sz_unused_(next_u8x);
    svbool_t const all_b8x = svptrue_b8();
    svbool_t danger_b8x = svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0x87),
                                    svcmpeq_n_u8(all_b8x, previous_u8x, 0xD6));
    danger_b8x = svorr_b_z(
        all_b8x, danger_b8x,
        svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xAC), svcmpeq_n_u8(all_b8x, previous_u8x, 0xEF)));
    return svptest_any(loaded_b8x, svand_b_z(loaded_b8x, danger_b8x, loaded_b8x));
}

/** @brief Vietnamese fold: four Latin blocks folding in place, incl. the E1 B8-BB third-byte parity. */
SZ_HELPER_NOINLINE svuint8_t sz_utf8_uncased_search_sve2_vietnamese_fold_(svuint8_t text_u8x, svuint8_t previous_u8x,
                                                                          svuint8_t previous2_u8x, svuint8_t next_u8x) {
    sz_unused_(next_u8x);
    svbool_t const all_b8x = svptrue_b8();
    svuint8_t result_u8x = sz_utf8_uncased_search_sve2_ascii_fold_(text_u8x);
    svbool_t const after_c3_b8x = svcmpeq_n_u8(all_b8x, previous_u8x, 0xC3);
    svbool_t const after_c4_b8x = svcmpeq_n_u8(all_b8x, previous_u8x, 0xC4);
    svbool_t const after_c5_b8x = svcmpeq_n_u8(all_b8x, previous_u8x, 0xC5);
    svbool_t const after_c6_b8x = svcmpeq_n_u8(all_b8x, previous_u8x, 0xC6);

    svbool_t const c3_target_b8x = svbic_b_z(
        all_b8x, svand_b_z(all_b8x, after_c3_b8x, sz_utf8_uncased_sve2_in_range_(text_u8x, 0x80, 0x1F)),
        svcmpeq_n_u8(all_b8x, text_u8x, 0x97));

    svbool_t const odd_b8x = svcmpeq_n_u8(all_b8x, svand_n_u8_x(all_b8x, text_u8x, 0x01), 0x01);
    svbool_t const inverted_b8x = svorr_b_z(
        all_b8x, svand_b_z(all_b8x, after_c4_b8x, sz_utf8_uncased_sve2_in_range_(text_u8x, 0xB9, 0x06)),
        svand_b_z(all_b8x, after_c5_b8x, sz_utf8_uncased_sve2_in_range_(text_u8x, 0x00, 0x89)));
    svbool_t const extended_even_b8x = svbic_b_z(
        all_b8x, svbic_b_z(all_b8x, svorr_b_z(all_b8x, after_c4_b8x, after_c5_b8x), inverted_b8x), odd_b8x);
    svbool_t const fold_extended_b8x = svorr_b_z(all_b8x, extended_even_b8x, svand_b_z(all_b8x, inverted_b8x, odd_b8x));

    svbool_t const c6_target_b8x = svand_b_z(
        all_b8x, after_c6_b8x,
        svorr_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xA0), svcmpeq_n_u8(all_b8x, text_u8x, 0xAF)));

    svbool_t const after_e1_pair_b8x = svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, previous2_u8x, 0xE1),
                                                 sz_utf8_uncased_sve2_in_range_(previous_u8x, 0xB8, 0x04));
    svbool_t const excluded_third_b8x = svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, previous_u8x, 0xBA),
                                                  sz_utf8_uncased_sve2_in_range_(text_u8x, 0x96, 0x0A));
    svbool_t const fold_e1_b8x = svbic_b_z(all_b8x, svbic_b_z(all_b8x, after_e1_pair_b8x, excluded_third_b8x), odd_b8x);

    result_u8x = svadd_n_u8_m(c3_target_b8x, result_u8x, 0x20);
    return svadd_n_u8_m(svorr_b_z(all_b8x, fold_extended_b8x, svorr_b_z(all_b8x, c6_target_b8x, fold_e1_b8x)),
                        result_u8x, 0x01);
}

/** @brief Vietnamese alarm: the expanding E1 BA 96-9F block and the shared Latin width-changers. */
SZ_HELPER_NOINLINE int sz_utf8_uncased_search_sve2_vietnamese_alarm_(svuint8_t text_u8x, svuint8_t previous_u8x,
                                                                     svuint8_t next_u8x, svbool_t loaded_b8x) {
    svbool_t const all_b8x = svptrue_b8();

    svbool_t danger_b8x = svand_b_z( // E1 BA 96-9F (expanding third byte)
        all_b8x, svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xBA), svcmpeq_n_u8(all_b8x, previous_u8x, 0xE1)),
        sz_utf8_uncased_sve2_in_range_(next_u8x, 0x96, 0x0A));
    danger_b8x = svorr_b_z( // Sharp S (C3 9F)
        all_b8x, danger_b8x,
        svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0x9F), svcmpeq_n_u8(all_b8x, previous_u8x, 0xC3)));
    danger_b8x = svorr_b_z( // Long S (C5 BF)
        all_b8x, danger_b8x,
        svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xBF), svcmpeq_n_u8(all_b8x, previous_u8x, 0xC5)));
    danger_b8x = svorr_b_z( // Ligatures (EF AC xx)
        all_b8x, danger_b8x,
        svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xAC), svcmpeq_n_u8(all_b8x, previous_u8x, 0xEF)));
    danger_b8x = svorr_b_z( // Kelvin (E2 84 xx)
        all_b8x, danger_b8x,
        svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0x84), svcmpeq_n_u8(all_b8x, previous_u8x, 0xE2)));
    return svptest_any(loaded_b8x, svand_b_z(loaded_b8x, danger_b8x, loaded_b8x));
}

/** @brief Georgian alarm: the historical scripts (Mtavruli, Asomtavruli, Nuskhuri) fold across blocks. */
SZ_HELPER_NOINLINE int sz_utf8_uncased_search_sve2_georgian_alarm_(svuint8_t text_u8x, svuint8_t previous_u8x,
                                                                   svuint8_t next_u8x, svbool_t loaded_b8x) {
    svbool_t const all_b8x = svptrue_b8();
    svbool_t const after_e1_b8x = svcmpeq_n_u8(all_b8x, previous_u8x, 0xE1);

    svbool_t danger_b8x = svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xB2), after_e1_b8x);
    danger_b8x = svorr_b_z( // Asomtavruli (E1 82 A0-E5)
        all_b8x, danger_b8x,
        svand_b_z(all_b8x, svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0x82), after_e1_b8x),
                  sz_utf8_uncased_sve2_in_range_(next_u8x, 0xA0, 0x46)));
    danger_b8x = svorr_b_z( // Nuskhuri (E2 B4)
        all_b8x, danger_b8x,
        svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xB4), svcmpeq_n_u8(all_b8x, previous_u8x, 0xE2)));
    return svptest_any(loaded_b8x, svand_b_z(loaded_b8x, danger_b8x, loaded_b8x));
}

#pragma endregion Script Folds and Alarms

SZ_API_COMPTIME sz_cptr_t sz_utf8_find_cased_sve2(sz_cptr_t str, sz_size_t length) {
    svbool_t const all_b8x = svptrue_b8();
    sz_size_t const chunk_capacity = svcntb() < 64 ? svcntb() : 64;
    sz_cptr_t text_cursor = str;

    // Advance by chunk minus a 3-byte slack so every checked lead's continuations stay in the same load.
    while (length) {
        sz_size_t const block_length = length < chunk_capacity - 3 ? length : chunk_capacity - 3;
        sz_u64_t const lead_mask = sz_u64_mask_until_serial_(block_length);
        svbool_t const loaded_b8x = svwhilelt_b8_u64(0, (sz_u64_t)(length < chunk_capacity ? length : chunk_capacity));
        svuint8_t const text_u8x = svld1_u8(loaded_b8x, (sz_u8_t const *)text_cursor);
        svuint8_t const next_u8x = sz_utf8_uncased_sve2_next_(text_u8x);

        // 1. Any ASCII letter resolves the question immediately.
        svbool_t const ascii_letter_b8x = svorr_b_z(loaded_b8x, sz_utf8_uncased_sve2_in_range_(text_u8x, 'A', 26),
                                                    sz_utf8_uncased_sve2_in_range_(text_u8x, 'a', 26));
        if (svptest_any(loaded_b8x, ascii_letter_b8x)) return sz_utf8_find_cased_serial(text_cursor, length);

        // 2. Non-ASCII leads split by declared length; each bicameral family bails to the serial scan.
        sz_u64_t const non_ascii = sz_utf8_rune_pred_to_u64_sve2_(
                                       svand_b_z(loaded_b8x, svcmpge_n_u8(loaded_b8x, text_u8x, 0x80), loaded_b8x)) &
                                   lead_mask;
        if (non_ascii) {
            sz_u64_t const two_mask = sz_utf8_rune_pred_to_u64_sve2_(
                                          svcmpeq_n_u8(all_b8x, svand_n_u8_x(all_b8x, text_u8x, 0xE0), 0xC0)) &
                                      lead_mask;
            sz_u64_t const three_mask = sz_utf8_rune_pred_to_u64_sve2_(
                                            svcmpeq_n_u8(all_b8x, svand_n_u8_x(all_b8x, text_u8x, 0xF0), 0xE0)) &
                                        lead_mask;
            sz_u64_t const four_mask = sz_utf8_rune_pred_to_u64_sve2_(
                                           svcmpeq_n_u8(all_b8x, svand_n_u8_x(all_b8x, text_u8x, 0xF8), 0xF0)) &
                                       lead_mask;

            // 4-byte bicameral scripts (SMP): F0 with second byte 90/91/96/9D/9E.
            if (four_mask) {
                svbool_t const smp_second_b8x = svorr_b_z(
                    all_b8x,
                    svorr_b_z(all_b8x, svcmpeq_n_u8(all_b8x, next_u8x, 0x90), svcmpeq_n_u8(all_b8x, next_u8x, 0x91)),
                    svorr_b_z(all_b8x, svcmpeq_n_u8(all_b8x, next_u8x, 0x96),
                              svorr_b_z(all_b8x, svcmpeq_n_u8(all_b8x, next_u8x, 0x9D),
                                        svcmpeq_n_u8(all_b8x, next_u8x, 0x9E))));
                sz_u64_t const smp_mask = sz_utf8_rune_pred_to_u64_sve2_(
                    svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xF0), smp_second_b8x));
                if (smp_mask & four_mask) return sz_utf8_find_cased_serial(text_cursor, length);
            }

            // 2-byte bicameral leads C3-D6 and the micro sign C2 B5.
            if (two_mask) {
                sz_u64_t const bicameral_mask = sz_utf8_rune_pred_to_u64_sve2_(
                    sz_utf8_uncased_sve2_in_range_(text_u8x, 0xC3, 0x14));
                if (bicameral_mask & two_mask) return sz_utf8_find_cased_serial(text_cursor, length);
                sz_u64_t const micro_mask = sz_utf8_rune_pred_to_u64_sve2_(
                    svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xC2), svcmpeq_n_u8(all_b8x, next_u8x, 0xB5)));
                if (micro_mask & two_mask) return sz_utf8_find_cased_serial(text_cursor, length);
            }

            // 3-byte bicameral sequences: E1/EF blanket, E2 outside 80-83, EA inside 99-9F / AC-AE.
            if (three_mask) {
                sz_u64_t const e1_ef_mask = sz_utf8_rune_pred_to_u64_sve2_(
                    svorr_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xE1), svcmpeq_n_u8(all_b8x, text_u8x, 0xEF)));
                if (e1_ef_mask & three_mask) return sz_utf8_find_cased_serial(text_cursor, length);
                sz_u64_t const e2_unsafe_mask = sz_utf8_rune_pred_to_u64_sve2_(
                    svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xE2),
                              svnot_b_z(all_b8x, sz_utf8_uncased_sve2_in_range_(next_u8x, 0x80, 0x04))));
                if (e2_unsafe_mask & three_mask) return sz_utf8_find_cased_serial(text_cursor, length);
                sz_u64_t const ea_unsafe_mask = sz_utf8_rune_pred_to_u64_sve2_(
                    svand_b_z(all_b8x, svcmpeq_n_u8(all_b8x, text_u8x, 0xEA),
                              svorr_b_z(all_b8x, sz_utf8_uncased_sve2_in_range_(next_u8x, 0x99, 0x07),
                                        sz_utf8_uncased_sve2_in_range_(next_u8x, 0xAC, 0x03))));
                if (ea_unsafe_mask & three_mask) return sz_utf8_find_cased_serial(text_cursor, length);
            }
        }

        text_cursor += block_length;
        length -= block_length;
    }
    return SZ_NULL_CHAR;
}

SZ_API_COMPTIME sz_cptr_t sz_utf8_uncased_search_sve2( //
    sz_cptr_t haystack, sz_size_t haystack_length,     //
    sz_cptr_t needle, sz_size_t needle_length,         //
    sz_utf8_uncased_needle_metadata_t *needle_metadata, sz_size_t *matched_length) {

    if (needle_length == 0) {
        *matched_length = 0;
        return haystack;
    }

    int const is_unknown = needle_metadata->kernel_id == sz_utf8_uncased_rune_unknown_k;
    int const known_agnostic = needle_metadata->kernel_id == sz_utf8_uncased_rune_invariant_k;
    if (known_agnostic || (is_unknown && sz_utf8_find_cased_sve2(needle, needle_length) == SZ_NULL_CHAR)) {
        sz_cptr_t result = sz_find_sve(haystack, haystack_length, needle, needle_length);
        *matched_length = result ? needle_length : 0;
        return result;
    }

    if (is_unknown) {
        sz_utf8_uncased_needle_metadata_(needle, needle_length, needle_metadata);
        if (needle_metadata->kernel_id == sz_utf8_uncased_rune_fallback_serial_k)
            return sz_utf8_uncased_search_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                                 matched_length);
    }

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_ascii_invariant_k) {
        if (needle_metadata->folded_slice_length <= 3)
            return sz_utf8_uncased_search_sve2_ascii_3probe_( //
                haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
        return sz_utf8_uncased_search_sve2_scripted_( //
            sz_utf8_uncased_search_sve2_ascii_fold_views_, (sz_utf8_uncased_alarm_sve2_t_)SZ_NULL, haystack,
            haystack_length, needle, needle_length, needle_metadata, matched_length);
    }
    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_western_europe_k)
        return sz_utf8_uncased_search_sve2_scripted_( //
            sz_utf8_uncased_search_sve2_western_europe_fold_, sz_utf8_uncased_search_sve2_western_europe_alarm_,
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_central_europe_k)
        return sz_utf8_uncased_search_sve2_scripted_( //
            sz_utf8_uncased_search_sve2_central_europe_fold_, sz_utf8_uncased_search_sve2_central_europe_alarm_,
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_cyrillic_k)
        return sz_utf8_uncased_search_sve2_scripted_( //
            sz_utf8_uncased_search_sve2_cyrillic_fold_, sz_utf8_uncased_search_sve2_cyrillic_alarm_, haystack,
            haystack_length, needle, needle_length, needle_metadata, matched_length);
    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_greek_k)
        return sz_utf8_uncased_search_sve2_scripted_( //
            sz_utf8_uncased_search_sve2_greek_fold_, sz_utf8_uncased_search_sve2_greek_alarm_, haystack,
            haystack_length, needle, needle_length, needle_metadata, matched_length);
    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_armenian_k)
        return sz_utf8_uncased_search_sve2_scripted_( //
            sz_utf8_uncased_search_sve2_armenian_fold_, sz_utf8_uncased_search_sve2_armenian_alarm_, haystack,
            haystack_length, needle, needle_length, needle_metadata, matched_length);
    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_vietnamese_k)
        return sz_utf8_uncased_search_sve2_scripted_( //
            sz_utf8_uncased_search_sve2_vietnamese_fold_, sz_utf8_uncased_search_sve2_vietnamese_alarm_, haystack,
            haystack_length, needle, needle_length, needle_metadata, matched_length);
    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_georgian_k)
        return sz_utf8_uncased_search_sve2_scripted_( //
            sz_utf8_uncased_search_sve2_ascii_fold_views_, sz_utf8_uncased_search_sve2_georgian_alarm_, haystack,
            haystack_length, needle, needle_length, needle_metadata, matched_length);

    needle_metadata->kernel_id = sz_utf8_uncased_rune_fallback_serial_k;
    return sz_utf8_uncased_search_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                         matched_length);
}

SZ_API_COMPTIME sz_ordering_t sz_utf8_uncased_order_sve2(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                         sz_size_t b_length) {
    return sz_utf8_uncased_order_serial(a, a_length, b, b_length);
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

#endif // STRINGZILLA_UTF8_UNCASED_SVE2_H_
