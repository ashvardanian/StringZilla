/**
 *  @brief Ice Lake (AVX-512) uncased UTF-8 search, comparison & invariance kernels.
 *  @file include/stringzilla/utf8_uncased/icelake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_ICELAKE_H_
#define STRINGZILLA_UTF8_UNCASED_ICELAKE_H_

#include "stringzilla/find/skylake.h" // `sz_find_skylake`
#include "stringzilla/utf8_uncased/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_ICELAKE
#if defined(__clang__)
#pragma clang attribute push(                                                                          \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2")
#endif

#pragma region ASCII Uncased Find

/**
 *  @brief Fold a ZMM register using ASCII case folding rules.
 *  @sa sz_utf8_uncased_rune_ascii_invariant_k
 *
 *  @param text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_uncased_find_icelake_ascii_fold_zmm_(__m512i text_zmm) {
    __m512i const a_upper_zmm = _mm512_set1_epi8('A');
    __m512i const range26_zmm = _mm512_set1_epi8(26);
    __m512i const x_20_zmm = _mm512_set1_epi8(0x20);

    // Only fold bytes in range A-Z: (byte - 'A') < 26
    __mmask64 upper_mask = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_zmm, a_upper_zmm), range26_zmm);
    return _mm512_mask_add_epi8(text_zmm, upper_mask, text_zmm, x_20_zmm);
}

/**
 *  @brief 3-probe ASCII uncased search using XOR + VPTERNLOG + VPTESTNMB.
 *
 *  For needles with folded_slice_length ≤ 3, probes at positions 0, mid, last cover ALL bytes.
 *  Uses parallel loads from different offsets, XOR for difference detection,
 *  VPTERNLOG to combine all 3, and VPTESTNMB to find matches.
 *  No window verification needed since probes cover the entire window.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_uncased_find_icelake_ascii_3probe_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                //
    sz_cptr_t needle, sz_size_t needle_length,                    //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,     //
    sz_size_t *matched_length) {

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_size_t const step = 64 - folded_window_length + 1;

    // For ≤3 bytes: positions 0, mid, last cover ALL positions
    // 1-byte: 0=last, 2-byte: 0,1, 3-byte: 0,1,2
    sz_size_t const offset_second = folded_window_length / 2;
    sz_size_t const offset_last = folded_window_length - 1;

    // Broadcast probe bytes
    sz_u512_vec_t probe_first_vec, probe_second_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[0]);
    probe_second_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_second]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_last]);

    sz_u512_vec_t haystack_first_vec, haystack_second_vec, haystack_last_vec;
    sz_cptr_t haystack_ptr = haystack;

    // Main loop
    while (haystack_ptr + 64 + offset_last <= haystack_end) {
        // 3 parallel loads from different offsets - all hit L1 cache
        haystack_first_vec.zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(_mm512_loadu_si512(haystack_ptr));
        haystack_second_vec.zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(
            _mm512_loadu_si512(haystack_ptr + offset_second));
        haystack_last_vec.zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(
            _mm512_loadu_si512(haystack_ptr + offset_last));

        // XOR for difference detection - 0 where probe matches
        __m512i diff_first_zmm = _mm512_xor_si512(haystack_first_vec.zmm, probe_first_vec.zmm);
        __m512i diff_second_zmm = _mm512_xor_si512(haystack_second_vec.zmm, probe_second_vec.zmm);
        __m512i diff_last_zmm = _mm512_xor_si512(haystack_last_vec.zmm, probe_last_vec.zmm);

        // VPTERNLOG: 0xFE = A|B|C - combines all 3 in a single instruction
        __m512i combined_zmm = _mm512_ternarylogic_epi64(diff_first_zmm, diff_second_zmm, diff_last_zmm, 0xFE);

        // VPTESTNMB: find zero bytes (all 3 probes matched)
        __mmask64 matches_mask = _mm512_testn_epi8_mask(combined_zmm, combined_zmm);
        matches_mask &= sz_u64_mask_until_(step);

        for (; matches_mask; matches_mask &= matches_mask - 1) {
            sz_size_t candidate_offset = sz_u64_ctz(matches_mask);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;

            // No window verification needed - probes cover all positions
            // Go directly to head/tail validation
            sz_cptr_t match = sz_utf8_uncased_verify_match_(                                               //
                haystack, haystack_length,                                                                 //
                needle, needle_length,                                                                     //
                haystack_candidate_ptr - haystack, folded_window_length,                                   //
                needle_metadata->offset_in_unfolded,                                                       //
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, //
                matched_length);
            if (match) { return match; }
        }
        haystack_ptr += step;
    }

    // Tail processing with masked loads
    sz_size_t remaining = (sz_size_t)(haystack_end - haystack_ptr);
    if (remaining >= folded_window_length) {
        sz_size_t valid_starts = remaining - folded_window_length + 1;
        __mmask64 valid_mask = sz_u64_mask_until_(valid_starts);

        __mmask64 load_first_mask = sz_u64_mask_until_(remaining);
        __mmask64 load_second_mask = (remaining > offset_second) ? sz_u64_mask_until_(remaining - offset_second) : 0;
        __mmask64 load_last_mask = (remaining > offset_last) ? sz_u64_mask_until_(remaining - offset_last) : 0;

        haystack_first_vec.zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(
            _mm512_maskz_loadu_epi8(load_first_mask, haystack_ptr));
        haystack_second_vec.zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(
            _mm512_maskz_loadu_epi8(load_second_mask, haystack_ptr + offset_second));
        haystack_last_vec.zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(
            _mm512_maskz_loadu_epi8(load_last_mask, haystack_ptr + offset_last));

        __m512i diff_first_zmm = _mm512_xor_si512(haystack_first_vec.zmm, probe_first_vec.zmm);
        __m512i diff_second_zmm = _mm512_xor_si512(haystack_second_vec.zmm, probe_second_vec.zmm);
        __m512i diff_last_zmm = _mm512_xor_si512(haystack_last_vec.zmm, probe_last_vec.zmm);

        __m512i combined_zmm = _mm512_ternarylogic_epi64(diff_first_zmm, diff_second_zmm, diff_last_zmm, 0xFE);
        __mmask64 matches_mask = _mm512_testn_epi8_mask(combined_zmm, combined_zmm);
        matches_mask &= valid_mask;

        for (; matches_mask; matches_mask &= matches_mask - 1) {
            sz_size_t candidate_offset = sz_u64_ctz(matches_mask);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;

            sz_cptr_t match = sz_utf8_uncased_verify_match_(                                               //
                haystack, haystack_length,                                                                 //
                needle, needle_length,                                                                     //
                haystack_candidate_ptr - haystack, folded_window_length,                                   //
                needle_metadata->offset_in_unfolded,                                                       //
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, //
                matched_length);
            if (match) { return match; }
        }
    }

    return SZ_NULL_CHAR;
}

/**
 *  @brief 4-probe ASCII uncased search using XOR + VPTERNLOG + VPTESTNMB.
 *
 *  For needles with folded_slice_length ≥ 4, probes at 4 positions filter candidates quickly.
 *  Uses parallel loads, XOR for difference detection, VPTERNLOG + VPOR to combine,
 *  and VPTESTNMB to find matches. Window verification IS required since probes
 *  don't cover all positions in the folded window.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_uncased_find_icelake_ascii_4probe_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                //
    sz_cptr_t needle, sz_size_t needle_length,                    //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,     //
    sz_size_t *matched_length) {

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_size_t const step = 64 - folded_window_length + 1;

    // 4 probe positions from metadata
    sz_size_t const offset_second = needle_metadata->probe_second;
    sz_size_t const offset_third = needle_metadata->probe_third;
    sz_size_t const offset_last = folded_window_length - 1;

    // Pre-load folded window for full verification (probes only check 4 positions)
    __mmask16 const folded_window_mask = sz_u16_mask_until_(folded_window_length);
    sz_u128_vec_t needle_window_vec, haystack_candidate_vec;
    needle_window_vec.xmm = _mm_loadu_si128((__m128i const *)needle_metadata->folded_slice);

    // Broadcast probe bytes
    sz_u512_vec_t probe_first_vec, probe_second_vec, probe_third_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[0]);
    probe_second_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_second]);
    probe_third_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_third]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_last]);

    sz_u512_vec_t haystack_first_vec, haystack_second_vec, haystack_third_vec, haystack_last_vec;
    sz_cptr_t haystack_ptr = haystack;

    // Main loop
    while (haystack_ptr + 64 + offset_last <= haystack_end) {
        // 4 parallel loads from different offsets - all hit L1 cache
        haystack_first_vec.zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(_mm512_loadu_si512(haystack_ptr));
        haystack_second_vec.zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(
            _mm512_loadu_si512(haystack_ptr + offset_second));
        haystack_third_vec.zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(
            _mm512_loadu_si512(haystack_ptr + offset_third));
        haystack_last_vec.zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(
            _mm512_loadu_si512(haystack_ptr + offset_last));

        // XOR for difference detection - 0 where probe matches
        __m512i diff_first_zmm = _mm512_xor_si512(haystack_first_vec.zmm, probe_first_vec.zmm);
        __m512i diff_second_zmm = _mm512_xor_si512(haystack_second_vec.zmm, probe_second_vec.zmm);
        __m512i diff_third_zmm = _mm512_xor_si512(haystack_third_vec.zmm, probe_third_vec.zmm);
        __m512i diff_last_zmm = _mm512_xor_si512(haystack_last_vec.zmm, probe_last_vec.zmm);

        // VPTERNLOG: 0xFE = A|B|C for first 3, then OR the 4th
        __m512i combined_zmm = _mm512_ternarylogic_epi64(diff_first_zmm, diff_second_zmm, diff_third_zmm, 0xFE);
        combined_zmm = _mm512_or_si512(combined_zmm, diff_last_zmm);

        // VPTESTNMB: find zero bytes (all 4 probes matched)
        __mmask64 matches_mask = _mm512_testn_epi8_mask(combined_zmm, combined_zmm);
        matches_mask &= sz_u64_mask_until_(step);

        for (; matches_mask; matches_mask &= matches_mask - 1) {
            sz_size_t candidate_offset = sz_u64_ctz(matches_mask);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;

            // Verify full window - probes don't cover all positions
            haystack_candidate_vec.xmm = _mm_maskz_loadu_epi8(folded_window_mask, haystack_candidate_ptr);
            haystack_candidate_vec.xmm = _mm512_castsi512_si128(
                sz_utf8_uncased_find_icelake_ascii_fold_zmm_(_mm512_castsi128_si512(haystack_candidate_vec.xmm)));

            __mmask16 window_mismatch_mask = _mm_mask_cmpneq_epi8_mask(folded_window_mask, haystack_candidate_vec.xmm,
                                                                       needle_window_vec.xmm);
            if (window_mismatch_mask) continue;

            // Window matched - validate head/tail
            sz_cptr_t match = sz_utf8_uncased_verify_match_(                                               //
                haystack, haystack_length,                                                                 //
                needle, needle_length,                                                                     //
                haystack_candidate_ptr - haystack, folded_window_length,                                   //
                needle_metadata->offset_in_unfolded,                                                       //
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, //
                matched_length);
            if (match) { return match; }
        }
        haystack_ptr += step;
    }

    // Tail processing with masked loads
    sz_size_t remaining = (sz_size_t)(haystack_end - haystack_ptr);
    if (remaining >= folded_window_length) {
        sz_size_t valid_starts = remaining - folded_window_length + 1;
        __mmask64 valid_mask = sz_u64_mask_until_(valid_starts);

        __mmask64 load_first_mask = sz_u64_mask_until_(remaining);
        __mmask64 load_second_mask = (remaining > offset_second) ? sz_u64_mask_until_(remaining - offset_second) : 0;
        __mmask64 load_third_mask = (remaining > offset_third) ? sz_u64_mask_until_(remaining - offset_third) : 0;
        __mmask64 load_last_mask = (remaining > offset_last) ? sz_u64_mask_until_(remaining - offset_last) : 0;

        haystack_first_vec.zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(
            _mm512_maskz_loadu_epi8(load_first_mask, haystack_ptr));
        haystack_second_vec.zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(
            _mm512_maskz_loadu_epi8(load_second_mask, haystack_ptr + offset_second));
        haystack_third_vec.zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(
            _mm512_maskz_loadu_epi8(load_third_mask, haystack_ptr + offset_third));
        haystack_last_vec.zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(
            _mm512_maskz_loadu_epi8(load_last_mask, haystack_ptr + offset_last));

        __m512i diff_first_zmm = _mm512_xor_si512(haystack_first_vec.zmm, probe_first_vec.zmm);
        __m512i diff_second_zmm = _mm512_xor_si512(haystack_second_vec.zmm, probe_second_vec.zmm);
        __m512i diff_third_zmm = _mm512_xor_si512(haystack_third_vec.zmm, probe_third_vec.zmm);
        __m512i diff_last_zmm = _mm512_xor_si512(haystack_last_vec.zmm, probe_last_vec.zmm);

        __m512i combined_zmm = _mm512_ternarylogic_epi64(diff_first_zmm, diff_second_zmm, diff_third_zmm, 0xFE);
        combined_zmm = _mm512_or_si512(combined_zmm, diff_last_zmm);
        __mmask64 matches_mask = _mm512_testn_epi8_mask(combined_zmm, combined_zmm);
        matches_mask &= valid_mask;

        for (; matches_mask; matches_mask &= matches_mask - 1) {
            sz_size_t candidate_offset = sz_u64_ctz(matches_mask);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;

            // Verify full window
            haystack_candidate_vec.xmm = _mm_maskz_loadu_epi8(folded_window_mask, haystack_candidate_ptr);
            haystack_candidate_vec.xmm = _mm512_castsi512_si128(
                sz_utf8_uncased_find_icelake_ascii_fold_zmm_(_mm512_castsi128_si512(haystack_candidate_vec.xmm)));

            __mmask16 window_mismatch_mask = _mm_mask_cmpneq_epi8_mask(folded_window_mask, haystack_candidate_vec.xmm,
                                                                       needle_window_vec.xmm);
            if (window_mismatch_mask) continue;

            sz_cptr_t match = sz_utf8_uncased_verify_match_(                                               //
                haystack, haystack_length,                                                                 //
                needle, needle_length,                                                                     //
                haystack_candidate_ptr - haystack, folded_window_length,                                   //
                needle_metadata->offset_in_unfolded,                                                       //
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, //
                matched_length);
            if (match) { return match; }
        }
    }

    return SZ_NULL_CHAR;
}

#pragma endregion // ASCII Uncased Find

#pragma region Scripted Uncased Find

/** @brief Folds one ZMM register of haystack text using script-specific rules. */
typedef __m512i (*sz_utf8_uncased_fold_zmm_t_)(__m512i text_zmm);

/**
 *  @brief Flags positions of "danger" characters that fold to a different byte width.
 *  @param load_mask Bitmask of the bytes actually loaded from the haystack, for tail-safe range checks.
 */
typedef __mmask64 (*sz_utf8_uncased_alarm_zmm_t_)(__m512i text_zmm, __mmask64 load_mask);

/**
 *  @brief Shared scan loop behind all script-specific uncased searches.
 *
 *  Scans the entire haystack from byte 0, looking for the folded window pattern.
 *  When found, verifies the head (backwards) and tail (forwards) using codepoint-by-codepoint
 *  comparison to handle variable-width folding correctly.
 *
 *  Every per-script kernel is a thin wrapper passing its own @p fold and @p alarm callbacks.
 *  The driver is force-inlined into each wrapper, so the callbacks resolve to direct calls
 *  with no indirect branches in the emitted code.
 *
 *  @param fold Script-specific ZMM case-folding callback.
 *  @param alarm Script-specific danger detection callback, or NULL if the script has no
 *      danger characters: the danger branch disappears and the full step is used.
 *  @param haystack Pointer to the haystack string.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the full needle string.
 *  @param needle_length Length of the full needle in bytes.
 *  @param needle_metadata Pre-folded window content with probe positions.
 *  @param matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_FORCE_INLINE sz_cptr_t sz_utf8_uncased_find_icelake_scripted_( //
    sz_utf8_uncased_fold_zmm_t_ fold,                             //
    sz_utf8_uncased_alarm_zmm_t_ alarm,                           //
    sz_cptr_t haystack, sz_size_t haystack_length,                //
    sz_cptr_t needle, sz_size_t needle_length,                    //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,     //
    sz_size_t *matched_length) {

    // Validate inputs
    sz_assert_(needle_metadata && "needle_metadata must be provided");
    sz_assert_(needle_metadata->folded_slice_length > 0 && "folded window must be non-empty");

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_assert_(folded_window_length <= 16 && "expect folded needle part to fit in XMM registers");

    // Pre-load folded window into XMM
    __mmask16 const folded_window_mask = sz_u16_mask_until_(folded_window_length);
    sz_u128_vec_t needle_window_vec, haystack_candidate_vec;
    needle_window_vec.xmm = _mm_loadu_si128((__m128i const *)needle_metadata->folded_slice);

    // 4 probe positions
    sz_size_t const offset_second = needle_metadata->probe_second;
    sz_size_t const offset_third = needle_metadata->probe_third;
    sz_size_t const offset_last = folded_window_length - 1;

    sz_u512_vec_t probe_first_vec, probe_second_vec, probe_third_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[0]);
    probe_second_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_second]);
    probe_third_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_third]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_last]);

    // Pre-load the first folded rune for danger zone matching
    sz_rune_t needle_first_safe_folded_rune = 0;
    if (alarm) sz_rune_parse_unchecked((sz_cptr_t)(needle_metadata->folded_slice), &needle_first_safe_folded_rune);

    sz_u512_vec_t haystack_vec;
    sz_cptr_t haystack_ptr = haystack;
    while (haystack_ptr < haystack_end) {
        sz_size_t const available = (sz_size_t)(haystack_end - haystack_ptr);
        if (available < folded_window_length) break;

        sz_size_t const chunk_size = available < 64 ? available : 64;
        sz_size_t const valid_starts = chunk_size - folded_window_length + 1;
        // For danger detection across chunk boundaries, reduce step size to ensure
        // 3-byte patterns at chunk end are fully visible in the next chunk.
        // For tail chunks (valid_starts <= 2), step = 1 ensures progress.
        // Scripts without danger characters advance by the full window count.
        sz_size_t const step = !alarm ? valid_starts : valid_starts > 2 ? valid_starts - 2 : 1;
        __mmask64 const load_mask = sz_u64_mask_until_(chunk_size);
        __mmask64 const valid_mask = sz_u64_mask_until_(valid_starts);

        haystack_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, haystack_ptr);

        // Check for anomalies (characters that fold to different byte widths)
        if (alarm) {
            __mmask64 danger_mask = alarm(haystack_vec.zmm, load_mask);
            if (danger_mask) {
                // The danger zone handler scans for the needle's first safe rune (at offset_in_unfolded).
                // The whole chunk is scanned, not just `valid_starts` positions: an expanding danger
                // character makes the haystack span SHORTER than the folded window, so a real match
                // can start within the window's length of the chunk end.
                sz_cptr_t match = sz_utf8_uncased_find_in_danger_zone_( //
                    haystack, haystack_length,                          //
                    needle, needle_length,                              //
                    haystack_ptr, chunk_size,                           // extended danger zone
                    needle_first_safe_folded_rune,                      // pivot point
                    needle_metadata->offset_in_unfolded,                // its location in the needle
                    matched_length);
                if (match) return match;
                haystack_ptr += step;
                continue;
            }
        }

        // Fold and 4-way probe filter
        haystack_vec.zmm = fold(haystack_vec.zmm);

        sz_u64_t matches = _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_first_vec.zmm);
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_second_vec.zmm) >> offset_second;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_third_vec.zmm) >> offset_third;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_last_vec.zmm) >> offset_last;
        matches &= valid_mask;

        // Candidate Verification
        for (; matches; matches &= matches - 1) {
            sz_size_t candidate_offset = sz_u64_ctz(matches);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;

            haystack_candidate_vec.xmm = _mm_maskz_loadu_epi8(folded_window_mask, haystack_candidate_ptr);
            haystack_candidate_vec.xmm = _mm512_castsi512_si128(
                fold(_mm512_castsi128_si512(haystack_candidate_vec.xmm)));

            __mmask16 window_mismatch_mask = _mm_mask_cmpneq_epi8_mask(folded_window_mask, haystack_candidate_vec.xmm,
                                                                       needle_window_vec.xmm);
            if (window_mismatch_mask) continue;

            sz_cptr_t match = sz_utf8_uncased_verify_match_(                             //
                haystack, haystack_length,                                               //
                needle, needle_length,                                                   //
                haystack_candidate_ptr - haystack, needle_metadata->folded_slice_length, // matched offset & length
                needle_metadata->offset_in_unfolded,                                     // head
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, // tail
                matched_length);
            if (match) { return match; }
        }
        haystack_ptr += step;
    }

    // Expanding danger characters ('ᾳ' folding to "αι") make the haystack span SHORTER than the
    // folded needle window, so a match can still start in the sub-window tail the loop never
    // probes. The tail is shorter than the 16-byte window, so the serial scan costs nothing.
    if (alarm && haystack_ptr < haystack_end) {
        sz_cptr_t match = sz_utf8_uncased_find_in_danger_zone_(     //
            haystack, haystack_length,                              //
            needle, needle_length,                                  //
            haystack_ptr, (sz_size_t)(haystack_end - haystack_ptr), // the unprobed tail
            needle_first_safe_folded_rune,                          // pivot point
            needle_metadata->offset_in_unfolded,                    // its location in the needle
            matched_length);
        if (match) { return match; }
    }

    return SZ_NULL_CHAR;
}

#pragma endregion // Scripted Uncased Find

#pragma region Western European Uncased Find

/**
 *  @brief Fold a ZMM register using Western European case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_western_europe_k
 *
 *  @param text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_uncased_find_icelake_western_europe_fold_naively_zmm_(__m512i text_zmm) {
    // Start with ASCII folded
    __m512i result_zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(text_zmm);

    // Constants for Latin folding
    __m512i const x_20_zmm = _mm512_set1_epi8(0x20);
    __m512i const x_73_zmm = _mm512_set1_epi8('s');

    // Constants for Latin-1 Supplement (C3 lead byte)
    // Note: µ (Micro Sign, C2 B5) is BANNED - needles with µ use serial fallback
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3); // Latin-1 Supplement (upper half)
    __m512i const x_9f_zmm = _mm512_set1_epi8((char)0x9F); // 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)

    // Range Logic Constants for Uppercase Detection (C3 80..9E):
    __m512i const x_80_zmm = _mm512_set1_epi8((char)0x80); // Lower bound of the range (offset)
    __m512i const x_1f_zmm = _mm512_set1_epi8((char)0x1F); // Length of the relevant range (0x9F - 0x80 = 0x1F)

    // Explicit Exclusions from Range Folding:
    // Note: '÷' Division Sign (C3 B7) is outside the uppercase range (0x80..0x9E), so it's safe.
    __m512i const x_97_zmm = _mm512_set1_epi8((char)0x97); // '×' Multiplication Sign (C3 97) - has no case

    // 1. Handle Eszett: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
    __mmask64 is_c3_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c3_zmm);
    __mmask64 is_after_c3_mask = is_c3_mask << 1;
    __mmask64 is_eszett_second_mask = _mm512_mask_cmpeq_epi8_mask(is_after_c3_mask, text_zmm, x_9f_zmm);
    __mmask64 is_eszett_mask = is_eszett_second_mask | (is_eszett_second_mask >> 1);
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_eszett_mask, x_73_zmm);

    // 2. Handle Latin-1 supplement uppercase letters (C3 80-9E) → add 0x20
    //    We need to map:
    //    - 'À' (C3 80) ... 'Þ' (C3 9E) to 'à' (C3 A0) ... 'þ' (C3 BE)
    //    Exceptions:
    //    - 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73) is already handled above
    //    - '×' (C3 97) is the Multiplication Sign, no case variant (so exclude it)
    __mmask64 is_97_mask = _mm512_mask_cmpeq_epi8_mask(is_after_c3_mask, text_zmm, x_97_zmm);
    __mmask64 is_latin1_upper_mask = _mm512_mask_cmplt_epu8_mask(
        is_after_c3_mask & ~is_eszett_second_mask & ~is_97_mask, _mm512_sub_epi8(text_zmm, x_80_zmm), x_1f_zmm);

    // Apply all folding transforms
    // +0x20 for Latin-1 (ASCII already handled in the base call)
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_latin1_upper_mask, result_zmm, x_20_zmm);
    return result_zmm;
}

/**
 *  @brief Fold a ZMM register using Western European case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_western_europe_k
 *
 *  @param text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_uncased_find_icelake_western_europe_fold_efficiently_zmm_(__m512i text_zmm) {
    // No cheaper instruction selection has been found for this fold yet: the naive
    // version already folds Latin-1 with a single masked range check and a blend.
    return sz_utf8_uncased_find_icelake_western_europe_fold_naively_zmm_(text_zmm);
}

/**
 *  @brief Naive alarm function for Western Europe danger zone detection.
 *
 *  Detects positions where danger characters occur that require special handling:
 *  - E1 BA 9E: 'ẞ' (U+1E9E, E1 BA 9E) → "ss" (U+0073 U+0073, 73 73) (3 bytes → 2 bytes)
 *  - E2 84 AA: 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B) (3 bytes → 1 byte)
 *  - E2 84 AB: 'Å' (U+212B, E2 84 AB) → 'å' (U+00E5, C3 A5) (3 bytes → 2 bytes)
 *  - EF AC 80-86: Ligatures:
 *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
 *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
 *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
 *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
 *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
 *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
 *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
 *  - C5 BF: 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73) (2 bytes → 1 byte)
 *  - C3 9F: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73) (2 bytes → 2 bytes, 1 rune → 2 runes)
 *
 *  Uses 12 CMPEQ operations (5 lead + 7 second byte checks).
 *
 *  @param text_zmm The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_INTERNAL __mmask64 sz_utf8_uncased_find_icelake_western_europe_alarm_naively_zmm_(__m512i text_zmm) {
    // Lead byte constants
    __m512i const x_e1_zmm = _mm512_set1_epi8((char)0xE1);
    __m512i const x_e2_zmm = _mm512_set1_epi8((char)0xE2);
    __m512i const x_ef_zmm = _mm512_set1_epi8((char)0xEF);
    __m512i const x_c5_zmm = _mm512_set1_epi8((char)0xC5);
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3);

    // Second/third byte constants
    __m512i const x_ba_zmm = _mm512_set1_epi8((char)0xBA);
    __m512i const x_84_zmm = _mm512_set1_epi8((char)0x84);
    __m512i const x_ac_zmm = _mm512_set1_epi8((char)0xAC);
    __m512i const x_bf_zmm = _mm512_set1_epi8((char)0xBF);
    __m512i const x_9f_zmm = _mm512_set1_epi8((char)0x9F);
    __m512i const x_aa_zmm = _mm512_set1_epi8((char)0xAA);
    __m512i const x_ab_zmm = _mm512_set1_epi8((char)0xAB);

    // Lead bytes (5 CMPEQ)
    __mmask64 is_e1_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_e1_zmm);
    __mmask64 is_e2_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_e2_zmm);
    __mmask64 is_ef_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ef_zmm);
    __mmask64 is_c5_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c5_zmm);
    __mmask64 is_c3_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c3_zmm);

    // Second/third bytes (7 CMPEQ)
    __mmask64 is_ba_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ba_zmm);
    __mmask64 is_84_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_84_zmm);
    __mmask64 is_ac_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ac_zmm);
    __mmask64 is_bf_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_bf_zmm);
    __mmask64 is_9f_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_9f_zmm);
    __mmask64 is_aa_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_aa_zmm);
    __mmask64 is_ab_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ab_zmm);
    __mmask64 is_b8_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xB8));

    // E1 BA is dangerous only when the third byte is 96-9E ('ẖ'-'ẞ': all expand to ASCII-led
    // sequences when folded). The rest of E1 BA covers Vietnamese 'Ạ'-'ỿ'-class letters that
    // fold in place - flagging them blanket-style sent dense Vietnamese text into the serial
    // danger-zone scanner on every chunk. The third-byte refinement is branched-over since
    // most non-Vietnamese text has no E1 BA pairs at all.
    __mmask64 is_e1_ba_danger_mask = (is_e1_mask << 1) & is_ba_mask;
    if (is_e1_ba_danger_mask) {
        __mmask64 is_expanding_third_mask = _mm512_cmplt_epu8_mask(
            _mm512_sub_epi8(text_zmm, _mm512_set1_epi8((char)0x96)), _mm512_set1_epi8(0x09));
        is_e1_ba_danger_mask &= is_expanding_third_mask >> 1;
    }

    // Danger mask construction (checking third bytes for E1 BA and E2 84 to avoid false positives)
    return is_e1_ba_danger_mask |                                                // Capital Sharp S & co (E1 BA 96-9E)
           ((is_e2_mask << 1) & is_84_mask & ((is_aa_mask | is_ab_mask) >> 1)) | // Kelvin/Angstrom (E2 84 AA/AB)
           ((is_ef_mask << 1) & is_ac_mask) |                                    // Ligatures (EF AC xx)
           ((is_c5_mask << 1) & is_bf_mask) |                                    // Long S (C5 BF)
           ((is_c5_mask << 1) & is_b8_mask) |                                    // 'Ÿ' (C5 B8) → 'ÿ' (C3 BF)
           ((is_c3_mask << 1) & is_9f_mask);                                     // Sharp S (C3 9F)
}

/**
 *  @brief Optimized alarm function for Western Europe danger zone detection.
 *
 *  Every Western European danger sequence starts with a 2-byte (lead, second) pair with a UNIQUE
 *  second byte, so the pair test inverts: one VPERMB maps each second byte to the lead that would
 *  make it dangerous (sentinel 0x00 otherwise), another VPERMB materializes the previous byte
 *  in-register, and a single VPCMPEQB matches them up. Two qualifiers make it exact: the second
 *  byte must be a continuation (its low-6-bit VPERMB index aliases ASCII and lead bytes), and the
 *  previous byte must be >= C0 (kills the sentinel matching a 0x00 byte and the zeroed lane 0).
 *  The third-byte refinements (E1 BA only expands for thirds 96-9E, E2 84 only for AA/AB) sit
 *  behind a `pair` branch, so the hot path is 5 port-5 ops instead of the naive 12 CMPEQs.
 *
 *  @param text_zmm The text ZMM register to scan for danger bytes.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_INTERNAL __mmask64 sz_utf8_uncased_find_icelake_western_europe_alarm_efficiently_zmm_(__m512i text_zmm,
                                                                                         __mmask64 load_mask) {
    sz_unused_(load_mask); // Present for the shared `sz_utf8_uncased_alarm_zmm_t_` signature

    // Index vector for materializing the previous byte: lane i takes byte i-1, lane 0 is zeroed
    __m512i const previous_byte_indices = _mm512_set_epi8( //
        62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35,
        34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6,
        5, 4, 3, 2, 1, 0, 0);
    // Expected lead per danger second byte: 84 → E2 (Kelvin/Angstrom), 9F → C3 ('ß'),
    // AC → EF (ligatures), BA → E1 ('ẞ' & co), BF → C5 ('ſ'), B8 → C5 ('Ÿ' → 'ÿ'); 0x00 elsewhere
    __m512i const expected_leads_lut = _mm512_set_epi8(
        // Indices 0x3F-0x30 (second bytes BF-B0)
        (char)0xC5, 0, 0, 0, 0, (char)0xE1, 0, (char)0xC5, 0, 0, 0, 0, 0, 0, 0, 0,
        // Indices 0x2F-0x20 (second bytes AF-A0)
        0, 0, 0, (char)0xEF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Indices 0x1F-0x10 (second bytes 9F-90)
        (char)0xC3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Indices 0x0F-0x00 (second bytes 8F-80)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (char)0xE2, 0, 0, 0, 0);
    __m512i const x_80_zmm = _mm512_set1_epi8((char)0x80);
    __m512i const x_40_zmm = _mm512_set1_epi8((char)0x40);
    __m512i const x_c0_zmm = _mm512_set1_epi8((char)0xC0);

    __m512i previous_zmm = _mm512_maskz_permutexvar_epi8(0xFFFFFFFFFFFFFFFEull, previous_byte_indices, text_zmm);
    __m512i expected_leads_zmm = _mm512_permutexvar_epi8(text_zmm, expected_leads_lut);
    __mmask64 pair_mask = _mm512_cmpeq_epi8_mask(previous_zmm, expected_leads_zmm);
    pair_mask &= _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_zmm, x_80_zmm), x_40_zmm);
    pair_mask &= _mm512_cmpge_epu8_mask(previous_zmm, x_c0_zmm);

    __mmask64 danger_mask = 0;
    if (pair_mask) {
        // Rare path: some pair matched, refine the third-byte conditions
        __mmask64 is_ba_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xBA));
        __mmask64 is_84_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0x84));
        // E1 BA is dangerous only for thirds 96-9E (the expanding 'ẖ'-'ẞ' class)
        __mmask64 is_expanding_third_mask = _mm512_cmplt_epu8_mask(
            _mm512_sub_epi8(text_zmm, _mm512_set1_epi8((char)0x96)), _mm512_set1_epi8(0x09));
        __mmask64 ba_pair_mask = pair_mask & is_ba_mask & (is_expanding_third_mask >> 1);
        // E2 84 is dangerous only for thirds AA/AB (Kelvin/Angstrom)
        __mmask64 is_aa_or_ab_mask = _mm512_cmpeq_epi8_mask(_mm512_and_si512(text_zmm, _mm512_set1_epi8((char)0xFE)),
                                                            _mm512_set1_epi8((char)0xAA));
        __mmask64 kelvin_pair_mask = pair_mask & is_84_mask & (is_aa_or_ab_mask >> 1);
        danger_mask = (pair_mask & ~(is_ba_mask | is_84_mask)) | ba_pair_mask | kelvin_pair_mask;
    }

    sz_assert_(danger_mask == sz_utf8_uncased_find_icelake_western_europe_alarm_naively_zmm_(text_zmm) &&
               "Efficient Western European alarm must match naive implementation");
    return danger_mask;
}

/**
 *  @brief Western European uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_western_europe_k
 *
 *  Scans the entire haystack from byte 0, looking for the folded window pattern.
 *  When found, verifies the head (backwards) and tail (forwards) using codepoint-by-codepoint
 *  comparison to handle variable-width folding correctly (e.g., 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)).
 *
 *  @param haystack Pointer to the haystack string.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the full needle string.
 *  @param needle_length Length of the full needle in bytes.
 *  @param needle_metadata Pre-folded window content with probe positions.
 *  @param matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_uncased_find_icelake_western_europe_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                  //
    sz_cptr_t needle, sz_size_t needle_length,                      //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,       //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_find_icelake_scripted_( //
        sz_utf8_uncased_find_icelake_western_europe_fold_efficiently_zmm_,
        sz_utf8_uncased_find_icelake_western_europe_alarm_efficiently_zmm_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Western European Uncased Find

#pragma region Central European Uncased Find

/**
 *  @brief Fold a ZMM register using Central European case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_central_europe_k
 *
 *  @param text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_uncased_find_icelake_central_europe_fold_naively_zmm_(__m512i text_zmm) {
    // Start with ASCII folded
    __m512i result_zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(text_zmm);

    // Constants for Latin folding
    __m512i const x_20_zmm = _mm512_set1_epi8((char)0x20); // 32
    __m512i const x_01_zmm = _mm512_set1_epi8((char)0x01); // 1

    // Constants for Latin-1 Supplement (C3 lead byte)
    // Range C3 80-9E (Uppercases), excluding C3 97 (×) and C3 9F (ß - not folded here)
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3);
    __m512i const x_80_zmm = _mm512_set1_epi8((char)0x80);
    __m512i const x_9e_zmm = _mm512_set1_epi8((char)0x9E);
    __m512i const x_97_zmm = _mm512_set1_epi8((char)0x97);

    // Constants for Latin Extended-A (C4, C5 lead bytes)
    __m512i const x_c4_zmm = _mm512_set1_epi8((char)0xC4);
    __m512i const x_c5_zmm = _mm512_set1_epi8((char)0xC5);

    // 1. Latin-1 Supplement: C3 80-9E → +0x20
    //    We check for C3 lead, then check if 2nd byte is in [80, 9E] and != 97
    __mmask64 is_c3_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c3_zmm);
    __mmask64 is_after_c3_mask = is_c3_mask << 1;

    __mmask64 is_latin1_range = _mm512_mask_cmpge_epu8_mask(is_after_c3_mask, result_zmm, x_80_zmm);
    is_latin1_range &= _mm512_mask_cmple_epu8_mask(is_after_c3_mask, result_zmm, x_9e_zmm);
    __mmask64 is_97 = _mm512_mask_cmpeq_epi8_mask(is_after_c3_mask, result_zmm, x_97_zmm);

    __mmask64 fold_latin1 = is_latin1_range & ~is_97;
    result_zmm = _mm512_mask_add_epi8(result_zmm, fold_latin1, result_zmm, x_20_zmm);

    // 2. Latin Extended-A: C4xx / C5xx case folding
    //    The uppercase/lowercase parity pattern varies within Latin Extended-A:
    //    - C4 80-B7 (U+0100-U+0137): uppercase = EVEN second bytes
    //    - C4 B9-BD (U+0139-U+013D): uppercase = ODD (Ĺ,Ļ,Ľ → +1); 'ĸ' (C4 B8) is caseless,
    //      'ľ' (C4 BE) is lowercase, and 'Ŀ' (C4 BF) folds across leads to 'ŀ' (C5 80),
    //      so it is routed through the alarm instead
    //    - C5 81-87 (U+0141-U+0147): uppercase = ODD (Ł,Ń,Ņ,Ň → +1)
    //    - C5 8A-B6 (U+014A-U+0176): uppercase = EVEN (Ŋ-Ŷ → +1)
    //    - C5 B9-BD (U+0179-U+017D): uppercase = ODD (Ź,Ż,Ž → +1)
    //    NOT folded (handled elsewhere or excluded):
    //    - 'ŀ' (U+0140, C5 80)
    //    - 'ň' (U+0148, C5 88)
    //    - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
    //    - 'ŷ' (U+0177, C5 B7)
    //    - 'Ÿ' (U+0178, C5 B8) → 'ÿ' (U+00FF, C3 BF)
    //    - 'ž' (U+017E, C5 BE)
    //    - 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
    __mmask64 is_c4_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c4_zmm);
    __mmask64 is_c5_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c5_zmm);
    __mmask64 is_after_c4_mask = is_c4_mask << 1;
    __mmask64 is_after_c5_mask = is_c5_mask << 1;

    __mmask64 is_even_mask = _mm512_testn_epi8_mask(result_zmm, x_01_zmm); // (val & 1) == 0
    __mmask64 is_odd_mask = ~is_even_mask;

    // C5 sub-range detection for second bytes
    __m512i const x_81_zmm = _mm512_set1_epi8((char)0x81);
    __m512i const x_87_zmm = _mm512_set1_epi8((char)0x87);
    __m512i const x_8a_zmm = _mm512_set1_epi8((char)0x8A);
    __m512i const x_b6_zmm = _mm512_set1_epi8((char)0xB6);
    __m512i const x_b9_zmm = _mm512_set1_epi8((char)0xB9);
    __m512i const x_bd_zmm = _mm512_set1_epi8((char)0xBD);
    __m512i const x_b7_zmm = _mm512_set1_epi8((char)0xB7);

    // C4 80-B7: even = uppercase
    __mmask64 is_c4_80_b7 = _mm512_mask_cmpge_epu8_mask(is_after_c4_mask, result_zmm, x_80_zmm);
    is_c4_80_b7 &= _mm512_mask_cmple_epu8_mask(is_after_c4_mask, result_zmm, x_b7_zmm);

    // C4 B9-BD: odd = uppercase (Ĺ,Ļ,Ľ)
    __mmask64 is_c4_b9_bd = _mm512_mask_cmpge_epu8_mask(is_after_c4_mask, result_zmm, x_b9_zmm);
    is_c4_b9_bd &= _mm512_mask_cmple_epu8_mask(is_after_c4_mask, result_zmm, x_bd_zmm);

    // C5 81-87: odd = uppercase (Ł,Ń,Ņ,Ň)
    __mmask64 is_c5_81_87 = _mm512_mask_cmpge_epu8_mask(is_after_c5_mask, result_zmm, x_81_zmm);
    is_c5_81_87 &= _mm512_mask_cmple_epu8_mask(is_after_c5_mask, result_zmm, x_87_zmm);

    // C5 8A-B6: even = uppercase (Ŋ-Ŷ)
    __mmask64 is_c5_8a_b6 = _mm512_mask_cmpge_epu8_mask(is_after_c5_mask, result_zmm, x_8a_zmm);
    is_c5_8a_b6 &= _mm512_mask_cmple_epu8_mask(is_after_c5_mask, result_zmm, x_b6_zmm);

    // C5 B9-BD: odd = uppercase (Ź,Ż,Ž)
    __mmask64 is_c5_b9_bd = _mm512_mask_cmpge_epu8_mask(is_after_c5_mask, result_zmm, x_b9_zmm);
    is_c5_b9_bd &= _mm512_mask_cmple_epu8_mask(is_after_c5_mask, result_zmm, x_bd_zmm);

    // Fold: C4 80-B7 even, C4 B9-BD odd, C5 81-87 odd, C5 8A-B6 even, C5 B9-BD odd
    __mmask64 fold_latext = (is_c4_80_b7 & is_even_mask) | (is_c4_b9_bd & is_odd_mask) | (is_c5_81_87 & is_odd_mask) |
                            (is_c5_8a_b6 & is_even_mask) | (is_c5_b9_bd & is_odd_mask);

    result_zmm = _mm512_mask_add_epi8(result_zmm, fold_latext, result_zmm, x_01_zmm);

    return result_zmm;
}

/**
 *  @brief Fold a ZMM register using Central European case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_central_europe_k
 *
 *  @param text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_uncased_find_icelake_central_europe_fold_efficiently_zmm_(__m512i text_zmm) {
    // Inline ASCII fold (avoiding function call overhead)
    __m512i const a_upper_zmm = _mm512_set1_epi8('A');
    __m512i const range26_zmm = _mm512_set1_epi8(26);
    __m512i const x_20_zmm = _mm512_set1_epi8(0x20);
    __mmask64 is_upper_mask = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_zmm, a_upper_zmm), range26_zmm);
    __m512i result_zmm = _mm512_mask_add_epi8(text_zmm, is_upper_mask, text_zmm, x_20_zmm);

    // Lead bytes & continuation detection
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3);
    __m512i const x_c4_zmm = _mm512_set1_epi8((char)0xC4);
    __m512i const x_c5_zmm = _mm512_set1_epi8((char)0xC5);
    __m512i const x_80_zmm = _mm512_set1_epi8((char)0x80);
    __m512i const x_40_zmm = _mm512_set1_epi8((char)0x40);

    // Like the Greek fold, the Central European transforms are fully determined by the second
    // byte of the C3/C4/C5-led sequence, so three VPERMB lookups (indexed by the continuation
    // byte's low 6 bits, qualified by `is_continuation_mask` since low-6-bit indices alias ASCII
    // and lead bytes) replace the range comparisons and their subtracts. This also keeps the
    // live-constant count low, which matters under GCC: it re-materializes broadcast constants
    // inside loops once their number outgrows the register budget.
    //
    // Latin-1 second-byte delta (C3 lead): 80-9E fold with +0x20, except 97 ('×').
    __m512i const latin1_deltas_lut = _mm512_set_epi8(
        // Indices 0x3F-0x30 (second bytes B0-BF): lowercase, no change
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Indices 0x2F-0x20 (second bytes A0-AF): lowercase, no change
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Indices 0x1F-0x10 (second bytes 90-9F): 90-9E fold, except 97
        0, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        // Indices 0x0F-0x00 (second bytes 80-8F): all fold
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20);
    // Latin Extended-A second-byte delta (C4 lead): +1 on 80-B7 even and B9-BD odd;
    // 'ĸ' (B8) is caseless, 'ľ' (BE) lowercase, and 'Ŀ' (BF) folds cross-lead via the alarm.
    __m512i const c4_deltas_lut = _mm512_set_epi8(
        // Indices 0x3F-0x30 (second bytes B0-BF): B0-B6 even, B9/BB/BD odd
        0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1,
        // Indices 0x2F-0x20 (second bytes A0-AF): even
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        // Indices 0x1F-0x10 (second bytes 90-9F): even
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        // Indices 0x0F-0x00 (second bytes 80-8F): even
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1);
    // Latin Extended-A second-byte delta (C5 lead): +1 on 81-87 odd, 8A-B6 even, B9-BD odd.
    __m512i const extended_deltas_lut = _mm512_set_epi8(
        // Indices 0x3F-0x30 (second bytes B0-BF): B0-B6 even, B9/BB/BD odd
        0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1,
        // Indices 0x2F-0x20 (second bytes A0-AF): even
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        // Indices 0x1F-0x10 (second bytes 90-9F): even
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        // Indices 0x0F-0x00 (second bytes 80-8F): 8A/8C/8E even, 81-87 odd
        0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0);

    // Lead byte detection; only continuation bytes (0x80-0xBF) qualify for the lookups.
    __mmask64 is_c3_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c3_zmm);
    __mmask64 is_c4_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c4_zmm);
    __mmask64 is_c5_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c5_zmm);
    __mmask64 is_continuation_mask = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_zmm, x_80_zmm), x_40_zmm);

    // All three lookups run in parallel off the same index vector
    __m512i latin1_deltas = _mm512_permutexvar_epi8(text_zmm, latin1_deltas_lut);
    __m512i c4_deltas = _mm512_permutexvar_epi8(text_zmm, c4_deltas_lut);
    __m512i extended_deltas = _mm512_permutexvar_epi8(text_zmm, extended_deltas_lut);

    __mmask64 fold_latin1_mask = (is_c3_mask << 1) & is_continuation_mask;
    __mmask64 fold_c4_mask = (is_c4_mask << 1) & is_continuation_mask;
    __mmask64 fold_extended_mask = (is_c5_mask << 1) & is_continuation_mask;

    // Build 3 offset vectors in parallel (positions are disjoint), combine with VPTERNLOG OR
    __m512i latin1_offsets_zmm = _mm512_maskz_mov_epi8(fold_latin1_mask, latin1_deltas);
    __m512i extended_offsets_zmm = _mm512_maskz_mov_epi8(fold_extended_mask, extended_deltas);
    __m512i parity_offsets_zmm = _mm512_maskz_mov_epi8(fold_c4_mask, c4_deltas);
    __m512i offset_zmm = _mm512_ternarylogic_epi64(latin1_offsets_zmm, extended_offsets_zmm, parity_offsets_zmm, 0xFE);
    result_zmm = _mm512_add_epi8(result_zmm, offset_zmm);

    sz_assert_(_mm512_cmpeq_epi8_mask(sz_utf8_uncased_find_icelake_central_europe_fold_naively_zmm_(text_zmm),
                                      result_zmm) == (__mmask64)-1 &&
               "Efficient Central European fold does not match naive implementation");
    return result_zmm;
}

/**
 *  @brief Naive alarm function for Central Europe danger zone detection.
 *
 *  Detects positions where danger characters occur that require special handling:
 *  - E2 84 AA: 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B) (3 bytes → 1 byte)
 *  - C3 9F: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73) (2 bytes → 2 bytes, 1 rune → 2 runes)
 *  - C4 B0: 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87) (2 bytes → 3 bytes)
 *  - C5 BF: 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73) (2 bytes → 1 byte)
 *  - EF AC 80-86: Ligatures:
 *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
 *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
 *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
 *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
 *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
 *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
 *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
 *
 *  Uses 10 CMPEQ operations (5 lead + 5 second byte checks).
 *
 *  @param text_zmm The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_INTERNAL __mmask64 sz_utf8_uncased_find_icelake_central_europe_alarm_naively_zmm_(__m512i text_zmm) {
    // Lead byte constants
    __m512i const x_e2_zmm = _mm512_set1_epi8((char)0xE2); // for Kelvin sign
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3); // for Sharp S
    __m512i const x_c4_zmm = _mm512_set1_epi8((char)0xC4); // for Dotted I
    __m512i const x_c5_zmm = _mm512_set1_epi8((char)0xC5); // for Long S
    __m512i const x_ef_zmm = _mm512_set1_epi8((char)0xEF); // for ligatures

    // Second byte constants
    __m512i const x_84_zmm = _mm512_set1_epi8((char)0x84); // 2nd byte of Kelvin
    __m512i const x_9f_zmm = _mm512_set1_epi8((char)0x9F); // 2nd byte of Sharp S
    __m512i const x_b0_zmm = _mm512_set1_epi8((char)0xB0); // 2nd byte of Dotted I
    __m512i const x_bf_zmm = _mm512_set1_epi8((char)0xBF); // 2nd byte of Long S
    __m512i const x_ac_zmm = _mm512_set1_epi8((char)0xAC); // 2nd byte of ligatures

    // Lead bytes (5 CMPEQ)
    __mmask64 is_e2_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_e2_zmm);
    __mmask64 is_c3_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c3_zmm);
    __mmask64 is_c4_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c4_zmm);
    __mmask64 is_c5_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c5_zmm);
    __mmask64 is_ef_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ef_zmm);

    // Second bytes (5 CMPEQ)
    __mmask64 is_84_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_84_zmm);
    __mmask64 is_9f_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_9f_zmm);
    __mmask64 is_b0_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_b0_zmm);
    __mmask64 is_bf_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_bf_zmm);
    __mmask64 is_ac_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ac_zmm);
    __mmask64 is_b8_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xB8));

    // Danger mask construction
    return ((is_e2_mask << 1) & is_84_mask) | // Kelvin (E2 84 AA)
           ((is_c3_mask << 1) & is_9f_mask) | // Sharp S (C3 9F)
           ((is_c4_mask << 1) & is_b0_mask) | // Dotted I (C4 B0)
           ((is_c4_mask << 1) & is_bf_mask) | // 'Ŀ' (C4 BF) → 'ŀ' (C5 80), crosses lead bytes
           ((is_c5_mask << 1) & is_bf_mask) | // Long S (C5 BF)
           ((is_c5_mask << 1) & is_b8_mask) | // 'Ÿ' (C5 B8) → 'ÿ' (C3 BF)
           ((is_ef_mask << 1) & is_ac_mask);  // Ligatures (EF AC xx)
}

/**
 *  @brief Optimized alarm function for Central Europe danger zone detection.
 *
 *  Every Central European danger character is a 2-byte (lead, second) pair with a UNIQUE second
 *  byte, so the pair test inverts naturally: one VPERMB maps each second byte to the lead that
 *  would make it dangerous (sentinel 0x00 otherwise), another VPERMB materializes the previous
 *  byte in-register, and a single VPCMPEQB matches them up. Two qualifiers make it exact:
 *  the second byte must be a continuation (its low-6-bit VPERMB index aliases ASCII and lead
 *  bytes), and the previous byte must be >= C0 (kills the sentinel matching a 0x00 byte and the
 *  zeroed lane 0). That is 5 port-5 ops instead of the naive 10 CMPEQs - and far fewer
 *  mask-register ops, which Clang would otherwise pile onto the same ports.
 *
 *  @param text_zmm The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_INTERNAL __mmask64 sz_utf8_uncased_find_icelake_central_europe_alarm_efficiently_zmm_(__m512i text_zmm,
                                                                                         __mmask64 load_mask) {
    sz_unused_(load_mask); // Present for the shared `sz_utf8_uncased_alarm_zmm_t_` signature

    // Index vector for materializing the previous byte: lane i takes byte i-1, lane 0 is zeroed
    __m512i const previous_byte_indices = _mm512_set_epi8( //
        62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35,
        34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6,
        5, 4, 3, 2, 1, 0, 0);
    // Expected lead per danger second byte: 84 → E2 (Kelvin), 9F → C3 ('ß'), B0 → C4 ('İ'),
    // BF → C5 ('ſ'), B8 → C5 ('Ÿ' → 'ÿ'), AC → EF (ligatures); 0x00 elsewhere.
    // 'Ŀ' (C4 BF) also folds across leads, but BF's slot is taken by C5 - it gets an exact pair.
    __m512i const expected_leads_lut = _mm512_set_epi8(
        // Indices 0x3F-0x30 (second bytes BF-B0)
        (char)0xC5, 0, 0, 0, 0, 0, 0, (char)0xC5, 0, 0, 0, 0, 0, 0, 0, (char)0xC4,
        // Indices 0x2F-0x20 (second bytes AF-A0)
        0, 0, 0, (char)0xEF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Indices 0x1F-0x10 (second bytes 9F-90)
        (char)0xC3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Indices 0x0F-0x00 (second bytes 8F-80)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (char)0xE2, 0, 0, 0, 0);
    __m512i const x_80_zmm = _mm512_set1_epi8((char)0x80);
    __m512i const x_40_zmm = _mm512_set1_epi8((char)0x40);
    __m512i const x_c0_zmm = _mm512_set1_epi8((char)0xC0);

    __m512i previous_zmm = _mm512_maskz_permutexvar_epi8(0xFFFFFFFFFFFFFFFEull, previous_byte_indices, text_zmm);
    __m512i expected_leads_zmm = _mm512_permutexvar_epi8(text_zmm, expected_leads_lut);
    __mmask64 danger_mask = _mm512_cmpeq_epi8_mask(previous_zmm, expected_leads_zmm);
    danger_mask &= _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_zmm, x_80_zmm), x_40_zmm);
    danger_mask &= _mm512_cmpge_epu8_mask(previous_zmm, x_c0_zmm);

    // 'Ŀ' (C4 BF): the second exact pair on the BF column
    __mmask64 is_c4_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xC4));
    __mmask64 is_bf_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xBF));
    danger_mask |= (is_c4_mask << 1) & is_bf_mask;

    sz_assert_(danger_mask == sz_utf8_uncased_find_icelake_central_europe_alarm_naively_zmm_(text_zmm) &&
               "Efficient Central European alarm must match naive implementation");
    return danger_mask;
}

/**
 *  @brief Central European uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_central_europe_k
 *
 *  @param haystack Pointer to the haystack string.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the full needle string.
 *  @param needle_length Length of the full needle in bytes.
 *  @param needle_metadata Safe window metadata.
 *  @param matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_uncased_find_icelake_central_europe_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                  //
    sz_cptr_t needle, sz_size_t needle_length,                      //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,       //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_find_icelake_scripted_( //
        sz_utf8_uncased_find_icelake_central_europe_fold_efficiently_zmm_,
        sz_utf8_uncased_find_icelake_central_europe_alarm_efficiently_zmm_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Central European Uncased Find

#pragma region Cyrillic Uncased Find

/**
 *  @brief Fold a ZMM register using Cyrillic case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_cyrillic_k
 *
 *  Handles Basic Cyrillic (D0/D1) and Extended Cyrillic (D2/D3) ranges.
 *
 *  @param text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_uncased_find_icelake_cyrillic_fold_naively_zmm_(__m512i text_zmm) {
    // Start with ASCII folded
    __m512i result_zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(text_zmm);

    // Constants for Basic Cyrillic Folding (D0/D1 only)
    // Note: Extended Cyrillic (D2/D3) is BANNED - needles with D2/D3 use serial fallback
    __m512i const x_10_zmm = _mm512_set1_epi8((char)0x10); // +16 for Extensions (D0 80-8F → D1 90-9F)
    __m512i const x_20_zmm = _mm512_set1_epi8((char)0x20); // +/-32 for basic block shifts

    // Lead Bytes:
    __m512i const x_d0_zmm = _mm512_set1_epi8((char)0xD0); // Basic Cyrillic upper (U+0400-U+043F)
    __m512i const x_d1_zmm = _mm512_set1_epi8((char)0xD1); // Basic Cyrillic lower (U+0440-U+047F)

    // Range Boundary Constants:
    __m512i const x_80_zmm = _mm512_set1_epi8((char)0x80);    // Base offset (0x80)
    __m512i const x_90_zmm = _mm512_set1_epi8((char)0x90);    // D0: 'А' start
    __m512i const x_a0_zmm = _mm512_set1_epi8((char)0xA0);    // D0: 'Р' start
    __m512i const x_len16_zmm = _mm512_set1_epi8((char)0x10); // 16: D0 block length

    // Basic Cyrillic (D0/D1 lead bytes) - U+0400 to U+047F
    // Three sub-ranges need folding:
    //   - Extensions:  D0 80-8F ('Ѐ'-'Џ') → D1 90-9F ('ѐ'-'џ') : lead D0→D1, second +0x10
    //   - Basic A-Pe:  D0 90-9F ('А'-'П') → D0 B0-BF ('а'-'п') : lead unchanged, second +0x20
    //   - Basic Er-Ya: D0 A0-AF ('Р'-'Я') → D1 80-8F ('р'-'я') : lead D0→D1, second -0x20
    __mmask64 is_lead_d0_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_d0_zmm);
    __mmask64 is_after_d0_mask = is_lead_d0_mask << 1;

    // Detect each uppercase sub-range (second bytes following D0)
    __mmask64 is_ext_range = _mm512_mask_cmplt_epu8_mask( // D0 80-8F: Extensions 'Ѐ'-'Џ'
        is_after_d0_mask, _mm512_sub_epi8(text_zmm, x_80_zmm), x_len16_zmm);
    __mmask64 is_basic1_range = _mm512_mask_cmplt_epu8_mask( // D0 90-9F: Basic 'А'-'П'
        is_after_d0_mask, _mm512_sub_epi8(text_zmm, x_90_zmm), x_len16_zmm);
    __mmask64 is_basic2_range = _mm512_mask_cmplt_epu8_mask( // D0 A0-AF: Basic 'Р'-'Я'
        is_after_d0_mask, _mm512_sub_epi8(text_zmm, x_a0_zmm), x_len16_zmm);

    // Change lead byte D0 → D1 for Extensions and Er-Ya (their lowercase lives in D1)
    __mmask64 change_lead_mask = (is_ext_range >> 1) | (is_basic2_range >> 1);
    result_zmm = _mm512_mask_mov_epi8(result_zmm, change_lead_mask, x_d1_zmm);

    // Apply second-byte transformations
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_ext_range, result_zmm, x_10_zmm);    // +0x10
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_basic1_range, result_zmm, x_20_zmm); // +0x20
    result_zmm = _mm512_mask_sub_epi8(result_zmm, is_basic2_range, result_zmm, x_20_zmm); // -0x20

    return result_zmm;
}

/**
 *  @brief Fold a ZMM register using Cyrillic case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_cyrillic_k
 *
 *  This function uses VPSHUFB optimization for Cyrillic case-folding. Why?
 *  Cyrillic has a clean high-nibble pattern: second bytes 8x/9x/Ax map directly
 *  to distinct offsets (+0x10, +0x20, -0x20). A single VPSHUFB lookup replaces
 *  3 range comparisons + 3 masked moves.
 *
 *  @param text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_uncased_find_icelake_cyrillic_fold_efficiently_zmm_(__m512i text_zmm) {
    // Start with ASCII folded
    __m512i result_zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(text_zmm);

    // Constants for Basic Cyrillic Folding (D0/D1 only)
    // Note: Extended Cyrillic (D2/D3) is BANNED - needles with D2/D3 use serial fallback
    __m512i const x_d0_zmm = _mm512_set1_epi8((char)0xD0);
    __m512i const x_d1_zmm = _mm512_set1_epi8((char)0xD1);
    __m512i const x_0f_zmm = _mm512_set1_epi8(0x0F);

    // Detect D0 lead bytes and positions following them
    __mmask64 is_d0_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_d0_zmm);
    __mmask64 is_after_d0_mask = is_d0_mask << 1;

    // VPSHUFB Lookup Table: High nibble of second byte → offset
    // Second bytes after D0:
    //   80-8F (high nibble 8): Extensions 'Ѐ'-'Џ' → +0x10
    //   90-9F (high nibble 9): Basic 'А'-'П'      → +0x20
    //   A0-AF (high nibble A): Basic 'Р'-'Я'      → -0x20 (0xE0)
    //   B0-BF (high nibble B): lowercase, no change → 0x00
    //
    // LUT layout: index by high nibble (0-F), repeated across 4 lanes
    __m512i const offset_lut = _mm512_set_epi8(
        // Lane 3 (bytes 48-63): indices F E D C B A 9 8 7 6 5 4 3 2 1 0
        0, 0, 0, 0, 0, (char)0xE0, (char)0x20, (char)0x10, 0, 0, 0, 0, 0, 0, 0, 0,
        // Lane 2 (bytes 32-47)
        0, 0, 0, 0, 0, (char)0xE0, (char)0x20, (char)0x10, 0, 0, 0, 0, 0, 0, 0, 0,
        // Lane 1 (bytes 16-31)
        0, 0, 0, 0, 0, (char)0xE0, (char)0x20, (char)0x10, 0, 0, 0, 0, 0, 0, 0, 0,
        // Lane 0 (bytes 0-15)
        0, 0, 0, 0, 0, (char)0xE0, (char)0x20, (char)0x10, 0, 0, 0, 0, 0, 0, 0, 0);

    // Extract high nibble of each byte: (byte >> 4) & 0x0F
    __m512i high_nibbles = _mm512_and_si512(_mm512_srli_epi16(text_zmm, 4), x_0f_zmm);

    // Single VPSHUFB lookup: high_nibble → offset value
    __m512i offsets = _mm512_shuffle_epi8(offset_lut, high_nibbles);

    // Zero out offsets for non-continuation positions (not after D0)
    offsets = _mm512_maskz_mov_epi8(is_after_d0_mask, offsets);

    // Apply offsets to continuation bytes
    result_zmm = _mm512_add_epi8(result_zmm, offsets);

    // Lead byte changes: D0 → D1 for high nibble 8 (ext) or A (basic2)
    // These ranges have lowercase in the D1 block
    __m512i const x_08_zmm = _mm512_set1_epi8(0x08);
    __m512i const x_0a_zmm = _mm512_set1_epi8(0x0A);
    __mmask64 is_8x = _mm512_mask_cmpeq_epi8_mask(is_after_d0_mask, high_nibbles, x_08_zmm);
    __mmask64 is_ax = _mm512_mask_cmpeq_epi8_mask(is_after_d0_mask, high_nibbles, x_0a_zmm);
    __mmask64 change_lead_mask = ((is_8x | is_ax) >> 1) & is_d0_mask;
    result_zmm = _mm512_mask_mov_epi8(result_zmm, change_lead_mask, x_d1_zmm);

    sz_assert_(_mm512_cmpeq_epi8_mask(sz_utf8_uncased_find_icelake_cyrillic_fold_naively_zmm_(text_zmm), result_zmm) ==
                   (__mmask64)-1 &&
               "Efficient Cyrillic fold does not match naive implementation");
    return result_zmm;
}

/**
 *  @brief Naive alarm function for Cyrillic danger zone detection.
 *
 *  Basic Cyrillic itself never changes byte width when folded, and Extended Cyrillic needles
 *  (D2/D3 leads) are banned at needle-analysis time. The one haystack-side hazard is Cyrillic
 *  Extended-C: 'ᲀ'-'ᲈ' (U+1C80-1C88, E1 B2 80-88) fold INTO basic 2-byte Cyrillic letters
 *  ('в', 'д', 'о', 'с', 'т', 'ъ', 'ѣ'), so a 3-byte haystack character can match a 2-byte
 *  needle character and must go through the serial danger-zone scanner.
 *
 *  @param text_zmm The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_INTERNAL __mmask64 sz_utf8_uncased_find_icelake_cyrillic_alarm_naively_zmm_(__m512i text_zmm) {
    __mmask64 is_e1_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xE1));
    __mmask64 is_b2_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xB2));
    __mmask64 is_folding_third_mask = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_zmm, _mm512_set1_epi8((char)0x80)),
                                                             _mm512_set1_epi8(0x09));
    return (is_e1_mask << 1) & is_b2_mask & (is_folding_third_mask >> 1);
}

/**
 *  @brief Optimized alarm function for Cyrillic danger zone detection.
 *  @sa sz_utf8_uncased_find_icelake_cyrillic_alarm_naively_zmm_
 *
 *  The E1 B2 pair is absent from virtually all real Cyrillic text, so the third-byte
 *  refinement hides behind a branch and the hot path is two compares.
 *
 *  @param text_zmm The haystack ZMM register.
 *  @param load_mask Present for the shared `sz_utf8_uncased_alarm_zmm_t_` signature.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_INTERNAL __mmask64 sz_utf8_uncased_find_icelake_cyrillic_alarm_efficiently_zmm_(__m512i text_zmm,
                                                                                   __mmask64 load_mask) {
    sz_unused_(load_mask);
    __mmask64 is_e1_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xE1));
    __mmask64 is_b2_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xB2));
    __mmask64 danger_mask = (is_e1_mask << 1) & is_b2_mask;
    if (danger_mask) {
        __mmask64 is_folding_third_mask = _mm512_cmplt_epu8_mask(
            _mm512_sub_epi8(text_zmm, _mm512_set1_epi8((char)0x80)), _mm512_set1_epi8(0x09));
        danger_mask &= is_folding_third_mask >> 1;
    }
    sz_assert_(danger_mask == sz_utf8_uncased_find_icelake_cyrillic_alarm_naively_zmm_(text_zmm) &&
               "Efficient Cyrillic alarm must match naive implementation");
    return danger_mask;
}

/**
 *  @brief Cyrillic uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_cyrillic_k
 *
 *  @param haystack Pointer to the haystack string.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the full needle string.
 *  @param needle_length Length of the full needle in bytes.
 *  @param needle_metadata Safe window metadata.
 *  @param matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_uncased_find_icelake_cyrillic_( //
    sz_cptr_t haystack, sz_size_t haystack_length,            //
    sz_cptr_t needle, sz_size_t needle_length,                //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata, //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_find_icelake_scripted_( //
        sz_utf8_uncased_find_icelake_cyrillic_fold_efficiently_zmm_,
        sz_utf8_uncased_find_icelake_cyrillic_alarm_efficiently_zmm_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Cyrillic Uncased Find

#pragma region Armenian Uncased Find

/**
 *  @brief Fold a ZMM register using Armenian case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_armenian_k
 *
 *  @param text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_uncased_find_icelake_armenian_fold_naively_zmm_(__m512i text_zmm) {
    // Start with ASCII folded
    __m512i result_zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(text_zmm);

    // Constants for Armenian folding
    // Lead bytes:
    __m512i const x_d4_zmm = _mm512_set1_epi8((char)0xD4);
    __m512i const x_d5_zmm = _mm512_set1_epi8((char)0xD5);
    __m512i const x_d6_zmm = _mm512_set1_epi8((char)0xD6);

    // Range offsets:
    __m512i const x_b1_zmm = _mm512_set1_epi8((char)0xB1);
    __m512i const x_96_zmm = _mm512_set1_epi8((char)0x96);

    // Transformation deltas:
    __m512i const x_30_zmm = _mm512_set1_epi8((char)0x30); // 48
    __m512i const x_10_zmm = _mm512_set1_epi8((char)0x10); // 16

    // Identify Lead Bytes
    __mmask64 is_d4_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_d4_zmm);
    __mmask64 is_d5_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_d5_zmm);
    __mmask64 is_after_d4_mask = is_d4_mask << 1;
    __mmask64 is_after_d5_mask = is_d5_mask << 1;

    // 1. D4 ranges (Uppercase D4 B1-BF → Lowercase D5 A1-AF)
    // 'Ա' (D4 B1) ... 'Ձ' (D4 BF) → 'ա' (D5 A1) ... 'ձ' (D5 AF)
    // Lead D4 → D5, Second B1-BF → A1-AF (-0x10)
    __mmask64 is_d4_upper = _mm512_mask_cmpge_epu8_mask(is_after_d4_mask, result_zmm, x_b1_zmm);
    // Upper bound implied by 0xFF (mask covers all B1-BF if we just check >= B1)

    // Apply D4 transformations
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_d4_upper >> 1, x_d5_zmm);        // Lead D4 → D5
    result_zmm = _mm512_mask_sub_epi8(result_zmm, is_d4_upper, result_zmm, x_10_zmm); // Second -0x10

    // 2. D5 ranges (Uppercase D5 80-96 → Lowercase D5 B0-BF / D6 80-86)
    // Group 2a: D5 80-8F ('Ղ'-'Տ') → D5 B0-BF ('ղ'-'տ')
    // Offset +0x30
    __mmask64 is_d5_upper_subset1 = _mm512_mask_cmple_epu8_mask(is_after_d5_mask, result_zmm,
                                                                _mm512_set1_epi8((char)0x8F));
    is_d5_upper_subset1 &= _mm512_mask_cmpge_epu8_mask(is_after_d5_mask, result_zmm, _mm512_set1_epi8((char)0x80));

    result_zmm = _mm512_mask_add_epi8(result_zmm, is_d5_upper_subset1, result_zmm, x_30_zmm); // +0x30

    // Group 2b: D5 90-96 ('Ր'-'Ֆ') → D6 80-86 ('ր'-'ֆ')
    // Lead D5 → D6, Second 90-96 → 80-86 (-0x10)
    __mmask64 is_d5_upper_subset2 = _mm512_mask_cmpge_epu8_mask(is_after_d5_mask, result_zmm,
                                                                _mm512_set1_epi8((char)0x90));
    is_d5_upper_subset2 &= _mm512_mask_cmple_epu8_mask(is_after_d5_mask, result_zmm, x_96_zmm);

    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_d5_upper_subset2 >> 1, x_d6_zmm);        // Lead D5 → D6
    result_zmm = _mm512_mask_sub_epi8(result_zmm, is_d5_upper_subset2, result_zmm, x_10_zmm); // Second -0x10

    return result_zmm;
}

/**
 *  @brief Fold a ZMM register using Armenian case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_armenian_k
 *
 *  This function uses VPTERNLOG optimization for Armenian case-folding. Why?
 *  Armenian has 3 distinct lead bytes (D4/D5/D6) with different sub-ranges.
 *  VPTERNLOG allows building 3 offset vectors in parallel (no dependencies),
 *  then combining them with a single OR operation (imm8=0xFE: A | B | C).
 *
 *  @param text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_uncased_find_icelake_armenian_fold_efficiently_zmm_(__m512i text_zmm) {
    // Inline ASCII fold (avoiding function call overhead)
    __m512i const a_upper_zmm = _mm512_set1_epi8('A');
    __m512i const range26_zmm = _mm512_set1_epi8(26);
    __m512i const x_20_zmm = _mm512_set1_epi8(0x20);
    __mmask64 is_upper_mask = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_zmm, a_upper_zmm), range26_zmm);
    __m512i result_zmm = _mm512_mask_add_epi8(text_zmm, is_upper_mask, text_zmm, x_20_zmm);

    // Lead bytes
    __m512i const x_d4_zmm = _mm512_set1_epi8((char)0xD4);
    __m512i const x_d5_zmm = _mm512_set1_epi8((char)0xD5);
    __m512i const x_d6_zmm = _mm512_set1_epi8((char)0xD6);

    // Range compression constants
    __m512i const x_b1_zmm = _mm512_set1_epi8((char)0xB1);
    __m512i const x_80_zmm = _mm512_set1_epi8((char)0x80);
    __m512i const x_90_zmm = _mm512_set1_epi8((char)0x90);
    __m512i const x_0f_zmm = _mm512_set1_epi8((char)0x0F); // 15: D4 B1-BF range
    __m512i const x_10_zmm = _mm512_set1_epi8((char)0x10); // 16: D5 80-8F range
    __m512i const x_07_zmm = _mm512_set1_epi8((char)0x07); // 7: D5 90-96 range
    __m512i const x_30_zmm = _mm512_set1_epi8((char)0x30); // +0x30 offset
    __m512i const x_f0_zmm = _mm512_set1_epi8((char)0xF0); // -0x10 offset

    // Lead byte detection (all parallel, no dependencies)
    __mmask64 is_d4_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_d4_zmm);
    __mmask64 is_d5_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_d5_zmm);
    __mmask64 is_after_d4_mask = is_d4_mask << 1;
    __mmask64 is_after_d5_mask = is_d5_mask << 1;

    // Range compression: All 3 ranges computed in parallel
    // D4 B1-BF: (byte - B1) < 15
    __mmask64 is_d4_upper = _mm512_mask_cmplt_epu8_mask(is_after_d4_mask, _mm512_sub_epi8(result_zmm, x_b1_zmm),
                                                        x_0f_zmm);
    // D5 80-8F: (byte - 80) < 16
    __mmask64 is_d5_subset1 = _mm512_mask_cmplt_epu8_mask(is_after_d5_mask, _mm512_sub_epi8(result_zmm, x_80_zmm),
                                                          x_10_zmm);
    // D5 90-96: (byte - 90) < 7
    __mmask64 is_d5_subset2 = _mm512_mask_cmplt_epu8_mask(is_after_d5_mask, _mm512_sub_epi8(result_zmm, x_90_zmm),
                                                          x_07_zmm);

    // Lead byte changes (applied before offset to preserve original lead byte positions)
    // D4 B1-BF: lead D4 → D5
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_d4_upper >> 1, x_d5_zmm);
    // D5 90-96: lead D5 → D6
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_d5_subset2 >> 1, x_d6_zmm);

    // Build 3 offset vectors in parallel, combine with OR
    // Each maskz_mov is independent (no data dependencies between them)
    __m512i off_d4_zmm = _mm512_maskz_mov_epi8(is_d4_upper, x_f0_zmm);      // -0x10 for D4 B1-BF
    __m512i off_d5_s1_zmm = _mm512_maskz_mov_epi8(is_d5_subset1, x_30_zmm); // +0x30 for D5 80-8F
    __m512i off_d5_s2_zmm = _mm512_maskz_mov_epi8(is_d5_subset2, x_f0_zmm); // -0x10 for D5 90-96

    // VPTERNLOG: A | B | C (imm8 = 0xFE)
    __m512i offset_zmm = _mm512_ternarylogic_epi64(off_d4_zmm, off_d5_s1_zmm, off_d5_s2_zmm, 0xFE);

    // Single add applies all offsets simultaneously
    result_zmm = _mm512_add_epi8(result_zmm, offset_zmm);

    sz_assert_(_mm512_cmpeq_epi8_mask(sz_utf8_uncased_find_icelake_armenian_fold_naively_zmm_(text_zmm), result_zmm) ==
                   (__mmask64)-1 &&
               "Efficient Armenian fold must match naive implementation");
    return result_zmm;
}

/**
 *  @brief Naive alarm function for Armenian danger zone detection.
 *
 *  Detects positions where danger characters occur that require special handling:
 *  - D6 87: 'և' (U+0587, D6 87) → "եւ" (U+0565 U+0582, D5 A5 D6 82) (2 bytes → 4 bytes)
 *  - EF AC 93-97: Armenian ligatures 'ﬓ'..'ﬗ' (U+FB13-U+FB17, EF AC 93-97) → 2 codepoints (4 bytes)
 *
 *  Uses 4 CMPEQ operations (2 lead + 2 second byte checks).
 *
 *  @param text_zmm The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_INTERNAL __mmask64 sz_utf8_uncased_find_icelake_armenian_alarm_naively_zmm_(__m512i text_zmm) {
    // Lead byte constants
    __m512i const x_d6_zmm = _mm512_set1_epi8((char)0xD6); // for Ech-Yiwn
    __m512i const x_ef_zmm = _mm512_set1_epi8((char)0xEF); // for ligatures

    // Second byte constants
    __m512i const x_87_zmm = _mm512_set1_epi8((char)0x87); // 2nd byte of Ech-Yiwn
    __m512i const x_ac_zmm = _mm512_set1_epi8((char)0xAC); // 2nd byte of ligatures

    // Lead bytes (2 CMPEQ)
    __mmask64 is_d6_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_d6_zmm);
    __mmask64 is_ef_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ef_zmm);

    // Second bytes (2 CMPEQ)
    __mmask64 is_87_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_87_zmm);
    __mmask64 is_ac_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ac_zmm);

    // Danger mask construction
    __mmask64 ech_yiwn_mask = (is_d6_mask << 1) & is_87_mask; // Ech-Yiwn (D6 87)
    __mmask64 ef_ac_mask = (is_ef_mask << 1) & is_ac_mask;    // Ligatures (EF AC)
    return ech_yiwn_mask | ef_ac_mask;
}

/**
 *  @brief Optimized alarm function for Armenian danger zone detection.
 *
 *  Currently delegates to the naive implementation.
 *  Armenian has only 4 CMPEQ ops, so optimization overhead may not be worthwhile.
 *
 *  @param text_zmm The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_INTERNAL __mmask64 sz_utf8_uncased_find_icelake_armenian_alarm_efficiently_zmm_(__m512i text_zmm,
                                                                                   __mmask64 load_mask) {
    sz_unused_(load_mask); // Present for the shared `sz_utf8_uncased_alarm_zmm_t_` signature

    // Only 4 CMPEQ operations - optimization overhead likely not worth it
    return sz_utf8_uncased_find_icelake_armenian_alarm_naively_zmm_(text_zmm);
}

/**
 *  @brief Armenian uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_armenian_k
 *
 *  @param haystack Pointer to the haystack string.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the full needle string.
 *  @param needle_length Length of the full needle in bytes.
 *  @param needle_metadata Pre-folded window content with probe positions.
 *  @param matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_uncased_find_icelake_armenian_( //
    sz_cptr_t haystack, sz_size_t haystack_length,            //
    sz_cptr_t needle, sz_size_t needle_length,                //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata, //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_find_icelake_scripted_( //
        sz_utf8_uncased_find_icelake_armenian_fold_efficiently_zmm_,
        sz_utf8_uncased_find_icelake_armenian_alarm_efficiently_zmm_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Armenian Uncased Find

#pragma region Greek Uncased Find

/**
 *  @brief Fold a ZMM register using Greek case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_greek_k
 *
 *  @param text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_uncased_find_icelake_greek_fold_naively_zmm_(__m512i text_zmm) {
    // Start with ASCII folded
    __m512i result_zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(text_zmm);

    // Constants for Greek folding
    // Lead bytes:
    __m512i const x_ce_zmm = _mm512_set1_epi8((char)0xCE);
    __m512i const x_cf_zmm = _mm512_set1_epi8((char)0xCF);
    // 'µ' (U+00B5, C2 B5) → 'μ' (U+03BC, CE BC) in-place.
    __m512i const x_c2_zmm = _mm512_set1_epi8((char)0xC2);
    __m512i const x_b5_zmm = _mm512_set1_epi8((char)0xB5);
    __m512i const x_bc_zmm = _mm512_set1_epi8((char)0xBC);

    // Ranges for CE lead byte:
    // CE 86-8F (Accented Upper): 'Ά' (86) ... 'Ώ' (8F)
    // CE 91-9F (Basic Upper 1):  'Α' (91) ... 'Ο' (9F)
    // CE A0-A9 (Basic Upper 2):  'Π' (A0) ... 'Ω' (A9)
    // CE AA-AB (Dialytika Upper): 'Ϊ' (AA) ... 'Ϋ' (AB)
    __m512i const x_86_zmm = _mm512_set1_epi8((char)0x86);
    __m512i const x_8f_zmm = _mm512_set1_epi8((char)0x8F);
    __m512i const x_91_zmm = _mm512_set1_epi8((char)0x91);
    __m512i const x_9f_zmm = _mm512_set1_epi8((char)0x9F);
    __m512i const x_a0_zmm = _mm512_set1_epi8((char)0xA0);
    __m512i const x_a9_zmm = _mm512_set1_epi8((char)0xA9);
    __m512i const x_aa_zmm = _mm512_set1_epi8((char)0xAA);
    __m512i const x_ab_zmm = _mm512_set1_epi8((char)0xAB);

    // Specific characters:
    // 'ς' (U+03C2, CF 82) → 'σ' (U+03C3, CF 83)
    __m512i const x_82_zmm = _mm512_set1_epi8((char)0x82);
    __m512i const x_83_zmm = _mm512_set1_epi8((char)0x83);

    // Offsets:
    __m512i const x_20_zmm = _mm512_set1_epi8((char)0x20); // +32 (0x20)
    __m512i const x_e0_zmm = _mm512_set1_epi8((char)0xE0); // -32 (0xE0)
    __m512i const x_01_zmm = _mm512_set1_epi8((char)0x01); // +1 (0x01)

    // Identify Lead Bytes
    __mmask64 is_ce_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_ce_zmm);
    __mmask64 is_cf_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_cf_zmm);
    __mmask64 is_after_ce_mask = is_ce_mask << 1;
    __mmask64 is_after_cf_mask = is_cf_mask << 1;

    // 1. CE ranges (Uppercase → Lowercase)
    // Basic Greek Upper (Range 1): CE 91-9F ('Α'-'Ο') → CE B1-BF ('α'-'ο') (Add 0x20)
    __mmask64 is_basic1 = _mm512_mask_cmpge_epu8_mask(is_after_ce_mask, result_zmm, x_91_zmm);
    is_basic1 &= _mm512_mask_cmple_epu8_mask(is_after_ce_mask, result_zmm, x_9f_zmm);

    // Basic Greek Upper (Range 2): CE A0-A9 ('Π'-'Ω') → CF 80-89 ('π'-'ω') (Change lead CE->CF, subtract 0x20 from
    // 2nd)
    __mmask64 is_basic2 = _mm512_mask_cmpge_epu8_mask(is_after_ce_mask, result_zmm, x_a0_zmm);
    is_basic2 &= _mm512_mask_cmple_epu8_mask(is_after_ce_mask, result_zmm, x_a9_zmm);

    // Accented Greek Upper: CE 86-8F ('Ά'-'Ώ') → Lowercase
    // Most map nicely: CE 8x → CE Ax (+20) or CE 8x → CF 8x (Lead change, 2nd same)
    __mmask64 is_accented = _mm512_mask_cmpge_epu8_mask(is_after_ce_mask, result_zmm, x_86_zmm);
    is_accented &= _mm512_mask_cmple_epu8_mask(is_after_ce_mask, result_zmm, x_8f_zmm);

    // Sub-masks for specific transformations within the accented block
    __mmask64 is_86 = is_accented & _mm512_cmpeq_epi8_mask(result_zmm, x_86_zmm);

    __mmask64 is_88_8a = is_accented & _mm512_cmpge_epu8_mask(result_zmm, _mm512_set1_epi8((char)0x88));
    is_88_8a &= _mm512_cmple_epu8_mask(result_zmm, _mm512_set1_epi8((char)0x8A));

    __mmask64 is_8c = is_accented & _mm512_cmpeq_epi8_mask(result_zmm, _mm512_set1_epi8((char)0x8C));

    __mmask64 is_8e_8f = is_accented & _mm512_cmpge_epu8_mask(result_zmm, _mm512_set1_epi8((char)0x8E));

    // Dialytika Greek Upper: CE AA-AB ('Ϊ', 'Ϋ') → CF 8A-8B (Lead CE->CF, 2nd -0x20)
    __mmask64 is_dialytika = _mm512_mask_cmpge_epu8_mask(is_after_ce_mask, result_zmm, x_aa_zmm);
    is_dialytika &= _mm512_mask_cmple_epu8_mask(is_after_ce_mask, result_zmm, x_ab_zmm);

    // 2. CF ranges (Final Sigma)
    // 'ς' (U+03C2, CF 82) → 'σ' (U+03C3, CF 83)
    __mmask64 is_final_sigma = _mm512_mask_cmpeq_epi8_mask(is_after_cf_mask, result_zmm, x_82_zmm);

    // Apply transformations using masked operations
    // Apply Basic Greek Upper (Range 1)
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_basic1, result_zmm, x_20_zmm);

    // Apply Basic Greek Upper (Range 2)
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_basic2 >> 1, x_cf_zmm);        // Lead CE → CF
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_basic2, result_zmm, x_e0_zmm); // 2nd -0x20 (using add E0)

    // Apply Accented Greek Upper
    // 1. Additions for Same-Block Mappings (CE → CE)
    // 'Ά' → +0x26
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_86, result_zmm, _mm512_set1_epi8(0x26));
    // 'Έ', 'Ή', 'Ί' → +0x25
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_88_8a, result_zmm, _mm512_set1_epi8(0x25));

    // 2. Lead Byte Changes (CE → CF)
    // 'Ό' (8C), 'Ύ' (8E), 'Ώ' (8F)
    __mmask64 change_lead = (is_8c >> 1) | (is_8e_8f >> 1);
    result_zmm = _mm512_mask_mov_epi8(result_zmm, change_lead, x_cf_zmm);

    // 3. Second Byte Changes for CF targets
    // 'Ύ', 'Ώ' → -1
    result_zmm = _mm512_mask_sub_epi8(result_zmm, is_8e_8f, result_zmm, x_01_zmm);

    // Apply Dialytika Greek Upper
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_dialytika >> 1, x_cf_zmm);
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_dialytika, result_zmm, x_e0_zmm); // -0x20

    // Apply Final Sigma
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_final_sigma, x_83_zmm);

    // Apply Micro Sign folding: 'µ' (U+00B5, C2 B5) → 'μ' (U+03BC, CE BC)
    __mmask64 is_c2_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c2_zmm);
    __mmask64 is_micro_second = (is_c2_mask << 1) & _mm512_cmpeq_epi8_mask(result_zmm, x_b5_zmm);
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_micro_second >> 1, x_ce_zmm); // Lead C2 → CE
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_micro_second, x_bc_zmm);      // Second B5 → BC

    return result_zmm;
}

/**
 *  @brief Fold a ZMM register using Greek case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_greek_k
 *
 *  This function uses VPERMB lookup tables for Greek case-folding. Why?
 *  Every Greek transform is fully determined by the second byte of the CE-led sequence:
 *  five distinct offsets (+0x20, -0x20, +0x26, +0x25, -1) over six sub-ranges, plus a
 *  CE->CF lead promotion for four of them. A continuation byte 0x80-0xBF maps to a unique
 *  VPERMB index through its low 6 bits, so one lookup yields the second-byte delta and one
 *  more yields the lead-promotion flag - replacing ~10 range comparisons and the offset
 *  merging. This also keeps the live-constant count low: GCC re-materializes broadcast
 *  constants inside loops once their number outgrows the register budget.
 *
 *  @param text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_uncased_find_icelake_greek_fold_efficiently_zmm_(__m512i text_zmm) {
    // Inline ASCII fold (avoiding function call overhead)
    __m512i const a_upper_zmm = _mm512_set1_epi8('A');
    __m512i const range26_zmm = _mm512_set1_epi8(26);
    __m512i const x_20_zmm = _mm512_set1_epi8(0x20);
    __mmask64 is_upper_mask = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_zmm, a_upper_zmm), range26_zmm);
    __m512i result_zmm = _mm512_mask_add_epi8(text_zmm, is_upper_mask, text_zmm, x_20_zmm);

    // Constants for Greek folding
    __m512i const x_ce_zmm = _mm512_set1_epi8((char)0xCE);
    __m512i const x_cf_zmm = _mm512_set1_epi8((char)0xCF);
    __m512i const x_c2_zmm = _mm512_set1_epi8((char)0xC2);
    __m512i const x_80_zmm = _mm512_set1_epi8((char)0x80);
    __m512i const x_40_zmm = _mm512_set1_epi8((char)0x40);
    __m512i const x_01_zmm = _mm512_set1_epi8(0x01);

    // Second-byte delta, indexed by the low 6 bits of the continuation byte (0x80-0xBF → 0x00-0x3F):
    //   CE 86       'Ά' → 'ά' (CE AC):    +0x26
    //   CE 88-8A    'Έ'-'Ί' → CE AD-AF:   +0x25
    //   CE 8E-8F    'Ύ','Ώ' → CF 8D-8E:   -1 and lead CE->CF
    //   CE 8C       'Ό' → CF 8C:           0 and lead CE->CF
    //   CE 91-9F    'Α'-'Ο' → CE B1-BF:   +0x20
    //   CE A0-A9    'Π'-'Ω' → CF 80-89:   -0x20 and lead CE->CF
    //   CE AA-AB    'Ϊ','Ϋ' → CF 8A-8B:   -0x20 and lead CE->CF
    __m512i const second_byte_deltas_lut = _mm512_set_epi8(
        // Indices 0x3F-0x30 (second bytes B0-BF): lowercase, no change
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Indices 0x2F-0x20 (second bytes A0-AF): A0-AB get -0x20
        0, 0, 0, 0, (char)0xE0, (char)0xE0, (char)0xE0, (char)0xE0, (char)0xE0, (char)0xE0, (char)0xE0, (char)0xE0,
        (char)0xE0, (char)0xE0, (char)0xE0, (char)0xE0,
        // Indices 0x1F-0x10 (second bytes 90-9F): 91-9F get +0x20
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0,
        // Indices 0x0F-0x00 (second bytes 80-8F): accented uppercase
        (char)0xFF, (char)0xFF, 0, 0, 0, 0x25, 0x25, 0x25, 0, 0x26, 0, 0, 0, 0, 0, 0);
    // Lead promotion flag (CE → CF means +1 on the lead byte), same indexing:
    __m512i const lead_promotions_lut = _mm512_set_epi8(
        // Indices 0x3F-0x30 (second bytes B0-BF)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Indices 0x2F-0x20 (second bytes A0-AF): A0-AB promote
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        // Indices 0x1F-0x10 (second bytes 90-9F)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Indices 0x0F-0x00 (second bytes 80-8F): 8C, 8E, 8F promote
        1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    // Lead byte detection; only continuation bytes (0x80-0xBF) qualify for the lookups,
    // since the low-6-bit index aliases ASCII and lead bytes onto the same table slots.
    __mmask64 is_ce_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ce_zmm);
    __mmask64 is_cf_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_cf_zmm);
    __mmask64 is_continuation_mask = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_zmm, x_80_zmm), x_40_zmm);
    __mmask64 fold_positions_mask = (is_ce_mask << 1) & is_continuation_mask;

    // Both lookups run in parallel off the same index vector
    __m512i second_byte_deltas = _mm512_permutexvar_epi8(text_zmm, second_byte_deltas_lut);
    __m512i lead_promotions = _mm512_permutexvar_epi8(text_zmm, lead_promotions_lut);

    // Apply second-byte deltas and promote flagged leads from CE to CF (+1)
    result_zmm = _mm512_mask_add_epi8(result_zmm, fold_positions_mask, result_zmm, second_byte_deltas);
    __mmask64 promote_lead_mask = _mm512_mask_test_epi8_mask(fold_positions_mask, lead_promotions, lead_promotions);
    result_zmm = _mm512_mask_add_epi8(result_zmm, promote_lead_mask >> 1, result_zmm, x_01_zmm);

    // Final sigma: 'ς' (U+03C2, CF 82) → 'σ' (U+03C3, CF 83) (+1 on the second byte)
    __mmask64 is_final_sigma_mask = _mm512_mask_cmpeq_epi8_mask(is_cf_mask << 1, text_zmm,
                                                                _mm512_set1_epi8((char)0x82));
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_final_sigma_mask, result_zmm, x_01_zmm);

    // Micro sign: 'µ' (U+00B5, C2 B5) → 'μ' (U+03BC, CE BC)
    __mmask64 is_c2_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c2_zmm);
    __mmask64 is_micro_second_mask = (is_c2_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xB5));
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_micro_second_mask >> 1, x_ce_zmm);
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_micro_second_mask, _mm512_set1_epi8((char)0xBC));

    sz_assert_(_mm512_cmpeq_epi8_mask(sz_utf8_uncased_find_icelake_greek_fold_naively_zmm_(text_zmm), result_zmm) ==
                   (__mmask64)-1 &&
               "Efficient Greek fold must match naive implementation");
    return result_zmm;
}

/**
 *  @brief Detect danger zone positions in a ZMM register for Greek text.
 *
 *  Danger characters are those that expand during case folding or have alternative representations:
 *  - CE 90: 'ΐ' (U+0390) expands to 3 codepoints
 *  - CE B0: 'ΰ' (U+03B0) expands to 3 codepoints
 *  - CF 90,91,95,96: Greek symbols 'ϐ', 'ϑ', 'ϕ', 'ϖ'
 *  - CF B0,B1,B5: Greek symbols 'ϰ', 'ϱ', 'ϵ'
 *  - E2 84: Ohm sign 'Ω' prefix
 *  - E1: Polytonic Greek / Extensions
 *  - CD: Combining Diacritical Marks
 *
 *  @param text_zmm The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_INTERNAL __mmask64 sz_utf8_uncased_find_icelake_greek_alarm_naively_zmm_(__m512i text_zmm) {
    // All constants local to function
    __m512i const x_ce_zmm = _mm512_set1_epi8((char)0xCE);
    __m512i const x_cf_zmm = _mm512_set1_epi8((char)0xCF);
    __m512i const x_e2_zmm = _mm512_set1_epi8((char)0xE2);
    __m512i const x_e1_zmm = _mm512_set1_epi8((char)0xE1);
    __m512i const x_cd_zmm = _mm512_set1_epi8((char)0xCD);
    __m512i const x_90_zmm = _mm512_set1_epi8((char)0x90);
    __m512i const x_b0_zmm = _mm512_set1_epi8((char)0xB0);
    __m512i const x_91_zmm = _mm512_set1_epi8((char)0x91);
    __m512i const x_95_zmm = _mm512_set1_epi8((char)0x95);
    __m512i const x_96_zmm = _mm512_set1_epi8((char)0x96);
    __m512i const x_b1_zmm = _mm512_set1_epi8((char)0xB1);
    __m512i const x_b5_zmm = _mm512_set1_epi8((char)0xB5);
    __m512i const x_84_zmm = _mm512_set1_epi8((char)0x84);

    // Lead bytes (5 CMPEQ)
    __mmask64 is_ce_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ce_zmm);
    __mmask64 is_cf_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_cf_zmm);
    __mmask64 is_e2_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_e2_zmm);
    __mmask64 is_e1_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_e1_zmm);
    __mmask64 is_cd_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_cd_zmm);

    // Second bytes (8 CMPEQ)
    __mmask64 is_90_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_90_zmm);
    __mmask64 is_b0_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_b0_zmm);
    __mmask64 is_9x_mask = is_90_mask | _mm512_cmpeq_epi8_mask(text_zmm, x_91_zmm) | //
                           _mm512_cmpeq_epi8_mask(text_zmm, x_95_zmm) | _mm512_cmpeq_epi8_mask(text_zmm, x_96_zmm);
    __mmask64 is_bx_mask = is_b0_mask | _mm512_cmpeq_epi8_mask(text_zmm, x_b1_zmm) |
                           _mm512_cmpeq_epi8_mask(text_zmm, x_b5_zmm) |
                           _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xB4)); // 'ϴ' → 'θ' (CE B8)
    __mmask64 is_84_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_84_zmm);

    // Danger mask construction
    __mmask64 ce_danger_mask = (is_ce_mask << 1) & (is_90_mask | is_b0_mask);
    __mmask64 cf_danger_mask = (is_cf_mask << 1) & (is_9x_mask | is_bx_mask);
    __mmask64 e2_danger_mask = (is_e2_mask << 1) & is_84_mask;
    return ce_danger_mask | cf_danger_mask | e2_danger_mask | is_e1_mask | is_cd_mask;
}

/**
 *  @brief Optimized danger zone detection for Greek text using bit-folded second-byte compares.
 *  @sa sz_utf8_uncased_find_icelake_greek_alarm_naively_zmm_
 *
 *  @param text_zmm The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_INTERNAL __mmask64 sz_utf8_uncased_find_icelake_greek_alarm_efficiently_zmm_(__m512i text_zmm, __mmask64 load_mask) {
    sz_unused_(load_mask); // Present for the shared `sz_utf8_uncased_alarm_zmm_t_` signature

    // The naive reference burns 13 CMPEQs - all bound to port 5. The danger second bytes pair up
    // across the 9x and Bx columns (90/B0, 91/B1, 95/B5), so clearing bit 5 with one VPANDD (a
    // port-0-capable op) lets a single compare cover each pair; only 96 stays an exact match,
    // since B6 ('϶') is safe and must not alias onto it. The CE/CF leads differ only in bit 0,
    // so one masked compare finds both. That trims the compare count to 9. A VPERMB flag-LUT
    // variant went lower still (4 port-5 ops) but lost under Clang, which lowers the masked
    // VPTESTMB extractions and mask-register logic into longer k-op chains on the same ports.
    __m512i const x_df_zmm = _mm512_set1_epi8((char)0xDF);
    __m512i const x_fe_zmm = _mm512_set1_epi8((char)0xFE);
    __m512i const x_90_zmm = _mm512_set1_epi8((char)0x90);
    __m512i const x_91_zmm = _mm512_set1_epi8((char)0x91);
    __m512i const x_95_zmm = _mm512_set1_epi8((char)0x95);
    __m512i const x_96_zmm = _mm512_set1_epi8((char)0x96);
    __m512i const x_ce_zmm = _mm512_set1_epi8((char)0xCE);
    __m512i const x_cf_zmm = _mm512_set1_epi8((char)0xCF);
    __m512i const x_cd_zmm = _mm512_set1_epi8((char)0xCD);
    __m512i const x_e1_zmm = _mm512_set1_epi8((char)0xE1);
    __m512i const x_e2_zmm = _mm512_set1_epi8((char)0xE2);
    __m512i const x_84_zmm = _mm512_set1_epi8((char)0x84);

    // Second bytes, folded across the 9x/Bx columns (4 compares instead of 7)
    __m512i folded_zmm = _mm512_and_si512(text_zmm, x_df_zmm);
    __mmask64 is_90_or_b0_mask = _mm512_cmpeq_epi8_mask(folded_zmm, x_90_zmm);
    __mmask64 is_91_or_b1_mask = _mm512_cmpeq_epi8_mask(folded_zmm, x_91_zmm);
    __mmask64 is_95_or_b5_mask = _mm512_cmpeq_epi8_mask(folded_zmm, x_95_zmm);
    __mmask64 is_96_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_96_zmm);
    // B4 ('ϴ' → 'θ') stays exact: its bit-5 column partner is 94, and CE 94 is 'Δ' - folding
    // them would alarm on every capital delta in ordinary Greek text.
    __mmask64 is_b4_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xB4));
    __mmask64 is_84_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_84_zmm);

    // Lead bytes: CE/CF share all but bit 0 (1 compare for the union + 1 exact for CF)
    __mmask64 is_ce_or_cf_mask = _mm512_cmpeq_epi8_mask(_mm512_and_si512(text_zmm, x_fe_zmm), x_ce_zmm);
    __mmask64 is_cf_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_cf_zmm);
    __mmask64 is_cd_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_cd_zmm);
    __mmask64 is_e1_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_e1_zmm);
    __mmask64 is_e2_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_e2_zmm);

    // CE and CF both alarm on 90/B0; only CF alarms on the extra Greek-symbol seconds
    __mmask64 danger_mask = ((is_ce_or_cf_mask << 1) & is_90_or_b0_mask) |
                            ((is_cf_mask << 1) & (is_91_or_b1_mask | is_95_or_b5_mask | is_96_mask | is_b4_mask)) |
                            ((is_e2_mask << 1) & is_84_mask) | is_cd_mask | is_e1_mask;

    sz_assert_(danger_mask == sz_utf8_uncased_find_icelake_greek_alarm_naively_zmm_(text_zmm) &&
               "Efficient Greek alarm must match naive implementation");
    return danger_mask;
}

/**
 *  @brief Greek uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_greek_k
 *
 *  @param haystack Pointer to the haystack string.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the full needle string.
 *  @param needle_length Length of the full needle in bytes.
 *  @param needle_metadata Pre-folded window content with probe positions.
 *  @param matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_uncased_find_icelake_greek_(    //
    sz_cptr_t haystack, sz_size_t haystack_length,            //
    sz_cptr_t needle, sz_size_t needle_length,                //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata, //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_find_icelake_scripted_( //
        sz_utf8_uncased_find_icelake_greek_fold_efficiently_zmm_,
        sz_utf8_uncased_find_icelake_greek_alarm_efficiently_zmm_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Greek Uncased Find

#pragma region Vietnamese Uncased Find

/**
 *  @brief Fold a ZMM register using Vietnamese case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_vietnamese_k
 *
 *  @param text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_uncased_find_icelake_vietnamese_fold_naively_zmm_(__m512i text_zmm) {
    // Start with ASCII folded
    __m512i result_zmm = sz_utf8_uncased_find_icelake_ascii_fold_zmm_(text_zmm);

    // Constants
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3);
    __m512i const x_c4_zmm = _mm512_set1_epi8((char)0xC4);
    __m512i const x_c5_zmm = _mm512_set1_epi8((char)0xC5);
    __m512i const x_c6_zmm = _mm512_set1_epi8((char)0xC6);
    __m512i const x_e1_zmm = _mm512_set1_epi8((char)0xE1);

    __m512i const x_b8_zmm = _mm512_set1_epi8((char)0xB8);
    __m512i const x_bb_zmm = _mm512_set1_epi8((char)0xBB);
    __m512i const x_ba_zmm = _mm512_set1_epi8((char)0xBA);
    __m512i const x_96_zmm = _mm512_set1_epi8((char)0x96);
    __m512i const x_9f_zmm = _mm512_set1_epi8((char)0x9F);

    __m512i const x_01_zmm = _mm512_set1_epi8((char)0x01);
    __m512i const x_20_zmm = _mm512_set1_epi8((char)0x20);

    // Masks for lead bytes
    __mmask64 is_c3_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c3_zmm);
    __mmask64 is_c4_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c4_zmm);
    __mmask64 is_c5_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c5_zmm);
    __mmask64 is_c6_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c6_zmm);
    __mmask64 is_e1_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_e1_zmm);

    __mmask64 is_after_c3_mask = is_c3_mask << 1;
    __mmask64 is_after_c4_mask = is_c4_mask << 1;
    __mmask64 is_after_c5_mask = is_c5_mask << 1;
    __mmask64 is_after_c6_mask = is_c6_mask << 1;

    // 1. Latin-1 Supplement (C3): Upper 80-96, 98-9E → +0x20
    // Same as Western/Central
    __mmask64 is_c3_target_mask = _mm512_mask_cmple_epu8_mask(is_after_c3_mask, result_zmm,
                                                              _mm512_set1_epi8((char)0x9E)); // <= 9E
    // Exclude multiplication sign 0xD7/0x97 if needed?
    // Western kernel excludes 0x97 (x) and 0xDF (ss).
    // Here we check <= 9E. 0x97 is usually Multiply, but standard folding keeps it?
    // Central kernel: excludes 97.
    is_c3_target_mask &= ~_mm512_cmpeq_epi8_mask(result_zmm, _mm512_set1_epi8((char)0x97)); // Exclude ×
    is_c3_target_mask &= _mm512_mask_cmpge_epu8_mask(is_after_c3_mask, result_zmm, _mm512_set1_epi8((char)0x80));

    result_zmm = _mm512_mask_add_epi8(result_zmm, is_c3_target_mask, result_zmm, x_20_zmm);

    // 2. Latin Extended-A (C4/C5): Even → Odd (+1) for MOST characters
    // Standard pattern (U+0100-U+0138, U+014A-U+017F): Even=uppercase, Odd=lowercase
    // INVERTED pattern (U+0139-U+0148): Odd=uppercase, Even=lowercase
    //   - After C4: B9,BB,BD,BF are uppercase (odd), BA,BC,BE are lowercase (even)
    //   - After C5: 81,83,85,87 are uppercase (odd), 80,82,84,86,88 are lowercase (even)
    // Note: 'Ŀ' (U+013F, C4 BF) → 'ŀ' (U+0140, C5 80) crosses lead bytes, handled specially by safety profile
    __mmask64 is_c4_c5_target_mask = is_after_c4_mask | is_after_c5_mask;
    __mmask64 is_even_mask = _mm512_cmpeq_epi8_mask(_mm512_and_si512(result_zmm, x_01_zmm), _mm512_setzero_si512());
    __mmask64 is_odd_mask = ~is_even_mask;

    // Identify the inverted range where Even=lowercase (should NOT be transformed +1)
    // After C4: B9-BE (U+0139-U+013E: Ĺ-ľ inverted pattern)
    // Note: BF (Ŀ U+013F) excluded - its lowercase ŀ (U+0140) is C5 80 (different lead byte)
    __mmask64 is_c4_inverted_range_mask = is_after_c4_mask &
                                          _mm512_cmpge_epu8_mask(result_zmm, _mm512_set1_epi8((char)0xB9)) &
                                          _mm512_cmple_epu8_mask(result_zmm, _mm512_set1_epi8((char)0xBE));
    // After C5: 80-88 (even bytes 80, 82, 84, 86, 88 are lowercase)
    __mmask64 is_c5_inverted_range_mask = is_after_c5_mask &
                                          _mm512_cmple_epu8_mask(result_zmm, _mm512_set1_epi8((char)0x88));
    __mmask64 is_inverted_range_mask = is_c4_inverted_range_mask | is_c5_inverted_range_mask;

    // Standard range: apply +1 to even bytes (uppercase → lowercase)
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_c4_c5_target_mask & is_even_mask & ~is_inverted_range_mask,
                                      result_zmm, x_01_zmm);
    // Inverted range: apply +1 to odd bytes (uppercase → lowercase)
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_inverted_range_mask & is_odd_mask, result_zmm, x_01_zmm);

    // 3. Latin Extended-B (C6): Specific Vietnamese chars
    // 'Ơ' (U+01A0, C6 A0) → 'ơ' (U+01A1, C6 A1) (even → odd)
    // 'Ư' (U+01AF, C6 AF) → 'ư' (U+01B0, C6 B0) (odd → even)
    __mmask64 is_c6_A0_mask = is_after_c6_mask & _mm512_cmpeq_epi8_mask(result_zmm, _mm512_set1_epi8((char)0xA0));
    __mmask64 is_c6_AF_mask = is_after_c6_mask & _mm512_cmpeq_epi8_mask(result_zmm, _mm512_set1_epi8((char)0xAF));

    result_zmm = _mm512_mask_add_epi8(result_zmm, is_c6_A0_mask | is_c6_AF_mask, result_zmm, x_01_zmm);

    // 4. Latin Extended Additional (E1 B8-BB): Even → Odd (+1) for Uppercase
    // E1 lead → Second is B8..BB → Third is Even
    // Exception: 1E 96-9F (Second=BA, Third=96-9F)
    __mmask64 is_e1_second_mask = is_e1_mask << 1;
    __mmask64 is_valid_second_mask = _mm512_mask_cmpge_epu8_mask(is_e1_second_mask, result_zmm, x_b8_zmm);
    is_valid_second_mask &= _mm512_mask_cmple_epu8_mask(is_e1_second_mask, result_zmm, x_bb_zmm);

    __mmask64 is_e1_third_mask = is_valid_second_mask << 1;

    // Check exclusion: Second == BA && Third in 96-9F
    __mmask64 is_ba_second_mask = is_e1_second_mask & _mm512_cmpeq_epi8_mask(result_zmm, x_ba_zmm);
    __mmask64 is_excluded_third_mask = (is_ba_second_mask << 1) &
                                       _mm512_mask_cmpge_epu8_mask(is_e1_third_mask, result_zmm, x_96_zmm) &
                                       _mm512_mask_cmple_epu8_mask(is_e1_third_mask, result_zmm, x_9f_zmm);

    // Apply +1 to Even bytes in valid third positions, excluding the danger range
    __mmask64 is_e1_target_mask = is_e1_third_mask & is_even_mask & ~is_excluded_third_mask;

    result_zmm = _mm512_mask_add_epi8(result_zmm, is_e1_target_mask, result_zmm, x_01_zmm);

    return result_zmm;
}

/**
 *  @brief Fold a ZMM register using Vietnamese case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_vietnamese_k
 *
 *  @param text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_uncased_find_icelake_vietnamese_fold_efficiently_zmm_(__m512i text_zmm) {
    // Inline ASCII fold (avoiding function call overhead)
    __m512i const a_upper_zmm = _mm512_set1_epi8('A');
    __m512i const range26_zmm = _mm512_set1_epi8(26);
    __m512i const x_20_zmm = _mm512_set1_epi8(0x20);
    __m512i const x_01_zmm = _mm512_set1_epi8(0x01);
    __mmask64 is_upper_mask = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_zmm, a_upper_zmm), range26_zmm);
    __m512i result_zmm = _mm512_mask_add_epi8(text_zmm, is_upper_mask, text_zmm, x_20_zmm);

    // Vietnamese folds across five lead contexts (C3/C4/C5/C6/E1), and every context's rule is a
    // function of the continuation byte alone. The naive version spends ~13 compare-to-mask ops
    // (all port 5) plus their subtracts on those rules; here a single VPERMB lookup, indexed by
    // the byte's low 6 bits and qualified by `is_continuation_mask` (low-6-bit indices alias
    // ASCII and lead bytes onto the same table slots), packs one flag bit per context rule:
    //   bit 0 (0x01)  C4-context +1: bytes B9-BE fold odd parity, the rest fold even
    //   bit 1 (0x02)  C5-context +1: bytes <= 88 fold odd parity, the rest fold even
    //   bit 2 (0x04)  valid E1 second byte (B8-BB, Latin Extended Additional)
    //   bit 3 (0x08)  E1 second byte BA, whose 96-9F third bytes are excluded
    //   bit 4 (0x10)  excluded third byte (96-9F)
    //   bit 5 (0x20)  C3-context +0x20: bytes 80-9E except 97 ('×')
    //   bit 6 (0x40)  C6-context +1: bytes A0 ('Ơ') and AF ('Ư')
    // The flags are extracted with masked VPTESTMB ops (which, unlike compares, are not bound to
    // port 5), and the broadcast bit constants 0x01/0x04/0x20/0x40 are shared with other uses, so
    // the live-constant count stays low enough that GCC stops re-materializing broadcasts inside
    // the scan loop - the main reason the naive version measures ~2.5x slower under GCC.
    __m512i const fold_flags_lut = _mm512_set_epi8(
        // Indices 0x3F-0x30 (second bytes BF-B0)
        0x00, 0x02, 0x01, 0x02, 0x05, 0x0E, 0x05, 0x07, 0x00, 0x03, 0x00, 0x03, 0x00, 0x03, 0x00, 0x03,
        // Indices 0x2F-0x20 (second bytes AF-A0)
        0x40, 0x03, 0x00, 0x03, 0x00, 0x03, 0x00, 0x03, 0x00, 0x03, 0x00, 0x03, 0x00, 0x03, 0x00, 0x43,
        // Indices 0x1F-0x10 (second bytes 9F-90)
        0x10, 0x33, 0x30, 0x33, 0x30, 0x33, 0x30, 0x33, 0x10, 0x33, 0x20, 0x23, 0x20, 0x23, 0x20, 0x23,
        // Indices 0x0F-0x00 (second bytes 8F-80)
        0x20, 0x23, 0x20, 0x23, 0x20, 0x23, 0x20, 0x21, 0x22, 0x21, 0x22, 0x21, 0x22, 0x21, 0x22, 0x21);

    // Lead & continuation detection constants
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3);
    __m512i const x_e1_zmm = _mm512_set1_epi8((char)0xE1);
    __m512i const x_80_zmm = _mm512_set1_epi8((char)0x80);
    __m512i const x_40_zmm = _mm512_set1_epi8((char)0x40);
    __m512i const x_02_zmm = _mm512_set1_epi8(0x02);
    __m512i const x_04_zmm = _mm512_set1_epi8(0x04);
    __m512i const x_08_zmm = _mm512_set1_epi8(0x08);
    __m512i const x_10_zmm = _mm512_set1_epi8(0x10);

    // The four Latin leads are contiguous (C3-C6), so one range compare finds them all and the
    // lead's own low bits (via the parity test below and one VPTESTMB) tell the contexts apart:
    // C3 = ..11, C4 = ..00, C5 = ..01, C6 = ..10.
    __mmask64 is_latin_lead_mask = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_zmm, x_c3_zmm), x_04_zmm);
    __mmask64 is_e1_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_e1_zmm);
    __mmask64 is_continuation_mask = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_zmm, x_80_zmm), x_40_zmm);
    // Parity is taken from the raw text: ASCII and Latin-1 folds both add 0x20, preserving bit 0.
    __mmask64 is_even_mask = _mm512_testn_epi8_mask(text_zmm, x_01_zmm);
    __mmask64 has_bit1_mask = _mm512_test_epi8_mask(text_zmm, x_02_zmm);

    __mmask64 is_after_c3_mask = (is_latin_lead_mask & ~is_even_mask & has_bit1_mask) << 1;
    __mmask64 is_after_c4_mask = (is_latin_lead_mask & is_even_mask & ~has_bit1_mask) << 1;
    __mmask64 is_after_c5_mask = (is_latin_lead_mask & ~is_even_mask & ~has_bit1_mask) << 1;
    __mmask64 is_after_c6_mask = (is_latin_lead_mask & is_even_mask & has_bit1_mask) << 1;
    __mmask64 is_e1_second_mask = (is_e1_mask << 1) & is_continuation_mask;

    __m512i fold_flags_zmm = _mm512_permutexvar_epi8(text_zmm, fold_flags_lut);

    // 1. Latin-1 Supplement (C3): 80-9E except 97 → +0x20
    __mmask64 fold_c3_mask = _mm512_mask_test_epi8_mask(is_after_c3_mask & is_continuation_mask, //
                                                        fold_flags_zmm, x_20_zmm);

    // 2. Latin Extended-A (C4/C5): +1 with context-specific parity, straight from the LUT.
    //    The naive reference also reaches non-continuation bytes after malformed C4/C5 leads
    //    (its parity rule has no range bound, and the C5 inverted range is `byte <= 88`), so
    //    those positions replicate its behavior with plain mask logic: C4 folds even bytes,
    //    C5 folds odd ASCII (<= 88 inverted rule) and even non-ASCII bytes.
    __mmask64 fold_c4_mask = _mm512_mask_test_epi8_mask(is_after_c4_mask & is_continuation_mask, //
                                                        fold_flags_zmm, x_01_zmm);
    __mmask64 fold_c5_mask = _mm512_mask_test_epi8_mask(is_after_c5_mask & is_continuation_mask, //
                                                        fold_flags_zmm, x_02_zmm);
    __mmask64 is_ascii_mask = _mm512_testn_epi8_mask(text_zmm, x_80_zmm);
    __mmask64 fold_malformed_c4_mask = is_after_c4_mask & ~is_continuation_mask & is_even_mask;
    __mmask64 fold_malformed_c5_mask = is_after_c5_mask & ~is_continuation_mask & (is_ascii_mask ^ is_even_mask);

    // 3. Latin Extended-B (C6): 'Ơ' (U+01A0, C6 A0) and 'Ư' (U+01AF, C6 AF) → +1
    __mmask64 fold_c6_mask = _mm512_mask_test_epi8_mask(is_after_c6_mask & is_continuation_mask, //
                                                        fold_flags_zmm, x_40_zmm);

    // 4. Latin Extended Additional (E1 B8-BB): third byte even → +1,
    //    except the expanding 'ẖ'-'ẟ' block (E1 BA 96-9F)
    __mmask64 is_valid_second_mask = _mm512_mask_test_epi8_mask(is_e1_second_mask, fold_flags_zmm, x_04_zmm);
    __mmask64 is_ba_second_mask = _mm512_mask_test_epi8_mask(is_e1_second_mask, fold_flags_zmm, x_08_zmm);
    __mmask64 is_excluded_third_mask = _mm512_mask_test_epi8_mask((is_ba_second_mask << 1) & is_continuation_mask,
                                                                  fold_flags_zmm, x_10_zmm);
    __mmask64 fold_e1_mask = (is_valid_second_mask << 1) & is_even_mask & ~is_excluded_third_mask;

    // Merge the disjoint +0x20 and +1 positions into one offset vector and apply with one add
    __mmask64 fold_plus_one_mask = fold_c4_mask | fold_c5_mask | fold_malformed_c4_mask | fold_malformed_c5_mask |
                                   fold_c6_mask | fold_e1_mask;
    __m512i offset_zmm = _mm512_maskz_mov_epi8(fold_c3_mask, x_20_zmm);
    offset_zmm = _mm512_mask_mov_epi8(offset_zmm, fold_plus_one_mask, x_01_zmm);
    result_zmm = _mm512_add_epi8(result_zmm, offset_zmm);

    sz_assert_(_mm512_cmpeq_epi8_mask(sz_utf8_uncased_find_icelake_vietnamese_fold_naively_zmm_(text_zmm),
                                      result_zmm) == (__mmask64)-1 &&
               "Efficient Vietnamese fold must match naive implementation");
    return result_zmm;
}

/**
 *  @brief Naive alarm function for Vietnamese danger zone detection.
 *
 *  Detects positions where danger characters occur that require special handling:
 *  - E1 BA 96-9F: Latin Extensions like 'ẞ' (U+1E9E, E1 BA 9E) → "ss" (U+0073 U+0073, 73 73)
 *  - C3 9F: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
 *  - C5 BF: 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
 *  - EF AC 80-86: Ligatures:
 *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
 *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
 *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
 *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
 *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
 *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
 *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
 *  - E2 84 AA: 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B)
 *
 *  Uses ~11 CMPEQ + 2 range operations.
 *  Note: Returns danger positions shifted to the START of multi-byte sequences.
 *
 *  @param text_zmm The haystack ZMM register.
 *  @param load_mask Mask of valid bytes in the ZMM register.
 *  @return Bitmask of positions where danger characters are detected (at sequence start).
 */
SZ_INTERNAL __mmask64 sz_utf8_uncased_find_icelake_vietnamese_alarm_naively_zmm_(__m512i text_zmm,
                                                                                 __mmask64 load_mask) {
    // Lead byte constants
    __m512i const x_e1_zmm = _mm512_set1_epi8((char)0xE1);
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3);
    __m512i const x_c5_zmm = _mm512_set1_epi8((char)0xC5);
    __m512i const x_ef_zmm = _mm512_set1_epi8((char)0xEF);
    __m512i const x_e2_zmm = _mm512_set1_epi8((char)0xE2);

    // Second byte constants
    __m512i const x_ba_zmm = _mm512_set1_epi8((char)0xBA);
    __m512i const x_9f_zmm = _mm512_set1_epi8((char)0x9F);
    __m512i const x_bf_zmm = _mm512_set1_epi8((char)0xBF);
    __m512i const x_ac_zmm = _mm512_set1_epi8((char)0xAC);
    __m512i const x_84_zmm = _mm512_set1_epi8((char)0x84);
    __m512i const x_96_zmm = _mm512_set1_epi8((char)0x96);

    // Lead bytes (5 CMPEQ)
    __mmask64 is_e1_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_e1_zmm);
    __mmask64 is_c3_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c3_zmm);
    __mmask64 is_c5_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c5_zmm);
    __mmask64 is_ef_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ef_zmm);
    __mmask64 is_e2_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_e2_zmm);

    // Early exit if no lead bytes
    if (!(is_e1_mask | is_c3_mask | is_c5_mask | is_ef_mask | is_e2_mask)) return 0;

    // Check for Latin Extended Additional exclusions (E1 BA 96-9F)
    __mmask64 ba_second_mask = (is_e1_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_ba_zmm);
    __mmask64 bad_third_mask = (ba_second_mask << 1) & _mm512_mask_cmpge_epu8_mask(load_mask, text_zmm, x_96_zmm) &
                               _mm512_mask_cmple_epu8_mask(load_mask, text_zmm, x_9f_zmm);

    // Check for Sharp S (C3 9F)
    __mmask64 sharp_s_mask = (is_c3_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_9f_zmm);

    // Check for Long S (C5 BF)
    __mmask64 long_s_mask = (is_c5_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_bf_zmm);

    // Check for Ligatures (EF AC)
    __mmask64 ligature_mask = (is_ef_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_ac_zmm);

    // Check for Kelvin (E2 84)
    __mmask64 kelvin_mask = (is_e2_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_84_zmm);

    // Shift back to sequence start positions
    return (bad_third_mask >> 2) | (sharp_s_mask >> 1) | (long_s_mask >> 1) | (ligature_mask >> 1) | (kelvin_mask >> 1);
}

/**
 *  @brief Optimized alarm function for Vietnamese danger zone detection.
 *
 *  Reduces port 5 pressure from ~12 to ~11 operations by:
 *  - E1/E2 consecutive: 2 CMPEQ → 1 CMPLT + 1 VPTESTNMB (p0)
 *
 *  Port summary: 11 p5 ops + 1 p0 op (vs 12 p5 originally)
 *
 *  @param text_zmm The haystack ZMM register.
 *  @param load_mask Mask of valid bytes in the ZMM register.
 *  @return Bitmask of positions where danger characters are detected (at sequence start).
 */
SZ_INTERNAL __mmask64 sz_utf8_uncased_find_icelake_vietnamese_alarm_efficiently_zmm_(__m512i text_zmm,
                                                                                     __mmask64 load_mask) {
    // Range constants
    __m512i const x_e1_zmm = _mm512_set1_epi8((char)0xE1);
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3);
    __m512i const x_c5_zmm = _mm512_set1_epi8((char)0xC5);
    __m512i const x_ef_zmm = _mm512_set1_epi8((char)0xEF);
    __m512i const x_ba_zmm = _mm512_set1_epi8((char)0xBA);
    __m512i const x_9f_zmm = _mm512_set1_epi8((char)0x9F);
    __m512i const x_bf_zmm = _mm512_set1_epi8((char)0xBF);
    __m512i const x_ac_zmm = _mm512_set1_epi8((char)0xAC);
    __m512i const x_84_zmm = _mm512_set1_epi8((char)0x84);
    __m512i const x_96_zmm = _mm512_set1_epi8((char)0x96);
    __m512i const x_02_zmm = _mm512_set1_epi8(0x02);

    // Check for E1/E2 range: (byte - 0xE1) < 2  [1 CMPLT on p5]
    __m512i off_e1_zmm = _mm512_sub_epi8(text_zmm, x_e1_zmm);
    __mmask64 is_e1_or_e2_mask = _mm512_cmplt_epu8_mask(off_e1_zmm, x_02_zmm);
    __mmask64 is_e1_mask = is_e1_or_e2_mask & _mm512_testn_epi8_mask(off_e1_zmm, off_e1_zmm); // offset==0 [p0]
    __mmask64 is_e2_mask = is_e1_or_e2_mask & ~is_e1_mask;

    // Other lead bytes (3 CMPEQ on p5)
    __mmask64 is_c3_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c3_zmm);
    __mmask64 is_c5_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c5_zmm);
    __mmask64 is_ef_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ef_zmm);

    // Early exit if no lead bytes
    if (!(is_e1_or_e2_mask | is_c3_mask | is_c5_mask | is_ef_mask)) return 0;

    // Check for Latin Extended Additional exclusions (E1 BA 96-9F)
    __mmask64 ba_second_mask = (is_e1_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_ba_zmm);
    __mmask64 bad_third_mask = (ba_second_mask << 1) & _mm512_mask_cmpge_epu8_mask(load_mask, text_zmm, x_96_zmm) &
                               _mm512_mask_cmple_epu8_mask(load_mask, text_zmm, x_9f_zmm);

    // Check for Sharp S (C3 9F)
    __mmask64 sharp_s_mask = (is_c3_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_9f_zmm);

    // Check for Long S (C5 BF)
    __mmask64 long_s_mask = (is_c5_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_bf_zmm);

    // Check for Ligatures (EF AC)
    __mmask64 ligature_mask = (is_ef_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_ac_zmm);

    // Check for Kelvin (E2 84)
    __mmask64 kelvin_mask = (is_e2_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_84_zmm);

    // Shift back to sequence start positions
    __mmask64 danger_mask = (bad_third_mask >> 2) | (sharp_s_mask >> 1) | (long_s_mask >> 1) | (ligature_mask >> 1) |
                            (kelvin_mask >> 1);

    sz_assert_(danger_mask == sz_utf8_uncased_find_icelake_vietnamese_alarm_naively_zmm_(text_zmm, load_mask) &&
               "Efficient Vietnamese alarm must match naive implementation");
    return danger_mask;
}

/**
 *  @brief Vietnamese uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_vietnamese_k
 *
 *  @param haystack Pointer to the haystack string.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the full needle string.
 *  @param needle_length Length of the full needle in bytes.
 *  @param needle_metadata Pre-folded window content with probe positions.
 *  @param matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_uncased_find_icelake_vietnamese_( //
    sz_cptr_t haystack, sz_size_t haystack_length,              //
    sz_cptr_t needle, sz_size_t needle_length,                  //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,   //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_find_icelake_scripted_( //
        sz_utf8_uncased_find_icelake_vietnamese_fold_efficiently_zmm_,
        sz_utf8_uncased_find_icelake_vietnamese_alarm_efficiently_zmm_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Vietnamese Uncased Find

#pragma region Georgian Uncased Find

/**
 *  @brief Detect danger zones in Georgian text (Mtavruli, Asomtavruli, Nuskhuri).
 *  @sa sz_utf8_uncased_rune_safe_georgian_k
 *
 *  Georgian Mkhedruli is caseless, so we only need to detect non-Mkhedruli Georgian scripts
 *  that require special handling via serial fallback:
 *  - Mtavruli (E1 B2 xx): Modern uppercase, folds to Mkhedruli
 *  - Asomtavruli (E1 82 A0-E5): Historical uppercase, folds to Nuskhuri
 *  - Nuskhuri (E2 B4 xx): Ecclesiastical script
 *
 *  All Georgian scripts use 3-byte UTF-8, so no length changes during folding.
 */
SZ_INTERNAL __mmask64 sz_utf8_uncased_find_icelake_georgian_alarm_zmm_(__m512i text_zmm, __mmask64 load_mask) {
    sz_unused_(load_mask); // Present for the shared `sz_utf8_uncased_alarm_zmm_t_` signature

    // Lead byte detection
    __mmask64 is_e1_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xE1));
    __mmask64 is_e2_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xE2));

    // Second byte detection
    __mmask64 is_b2_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xB2)); // Mtavruli
    __mmask64 is_82_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0x82)); // Asomtavruli
    __mmask64 is_b4_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xB4)); // Nuskhuri

    // E1 B2 xx = Mtavruli (folds to Mkhedruli)
    __mmask64 mtavruli_mask = (is_e1_mask << 1) & is_b2_mask;

    // E1 82 [A0-E5] = Asomtavruli (folds to Nuskhuri)
    // Third byte range check: (byte - 0xA0) < 0x46 covers A0-E5
    __mmask64 after_e1_82_mask = (is_e1_mask << 1) & is_82_mask;
    __m512i offset_a0_zmm = _mm512_add_epi8(text_zmm, _mm512_set1_epi8(0x60)); // -0xA0 = +0x60
    __mmask64 in_a0_e5_mask = _mm512_cmplt_epu8_mask(offset_a0_zmm, _mm512_set1_epi8(0x46));
    __mmask64 asomtavruli_mask = (after_e1_82_mask << 1) & in_a0_e5_mask;

    // E2 B4 xx = Nuskhuri
    __mmask64 nuskhuri_mask = (is_e2_mask << 1) & is_b4_mask;

    // Shift back to sequence start positions
    return (mtavruli_mask >> 1) | (asomtavruli_mask >> 2) | (nuskhuri_mask >> 1);
}

/**
 *  @brief Optimized alarm function for Georgian danger zone detection.
 *  @sa sz_utf8_uncased_find_icelake_georgian_alarm_zmm_
 *
 *  Every Georgian danger sequence starts with a 2-byte (lead, second) pair with a UNIQUE second
 *  byte, so the pair test inverts: one VPERMB maps each second byte to the lead that would make
 *  it dangerous (sentinel 0x00 otherwise), another VPERMB materializes the previous byte
 *  in-register, and a single VPCMPEQB matches them up. Two qualifiers make it exact: the second
 *  byte must be a continuation (its low-6-bit VPERMB index aliases ASCII and lead bytes), and the
 *  previous byte must be >= C0 (kills the sentinel matching a 0x00 byte and the zeroed lane 0).
 *  The Asomtavruli third-byte range check sits behind a `pair` branch, keeping the hot path at
 *  5 port-5 ops. The reference `sz_utf8_uncased_find_icelake_georgian_alarm_zmm_` keeps
 *  its name unsuffixed because the probe harness references it directly.
 */
SZ_INTERNAL __mmask64 sz_utf8_uncased_find_icelake_georgian_alarm_efficiently_zmm_(__m512i text_zmm,
                                                                                   __mmask64 load_mask) {
    sz_unused_(load_mask); // Present for the shared `sz_utf8_uncased_alarm_zmm_t_` signature

    // Index vector for materializing the previous byte: lane i takes byte i-1, lane 0 is zeroed
    __m512i const previous_byte_indices = _mm512_set_epi8( //
        62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35,
        34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6,
        5, 4, 3, 2, 1, 0, 0);
    // Expected lead per danger second byte: B2 → E1 (Mtavruli), 82 → E1 (Asomtavruli),
    // B4 → E2 (Nuskhuri); 0x00 elsewhere
    __m512i const expected_leads_lut = _mm512_set_epi8(
        // Indices 0x3F-0x30 (second bytes BF-B0)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (char)0xE2, 0, (char)0xE1, 0, 0,
        // Indices 0x2F-0x20 (second bytes AF-A0)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Indices 0x1F-0x10 (second bytes 9F-90)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        // Indices 0x0F-0x00 (second bytes 8F-80)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (char)0xE1, 0, 0);
    __m512i const x_80_zmm = _mm512_set1_epi8((char)0x80);
    __m512i const x_40_zmm = _mm512_set1_epi8((char)0x40);
    __m512i const x_c0_zmm = _mm512_set1_epi8((char)0xC0);

    __m512i previous_zmm = _mm512_maskz_permutexvar_epi8(0xFFFFFFFFFFFFFFFEull, previous_byte_indices, text_zmm);
    __m512i expected_leads_zmm = _mm512_permutexvar_epi8(text_zmm, expected_leads_lut);
    __mmask64 pair_mask = _mm512_cmpeq_epi8_mask(previous_zmm, expected_leads_zmm);
    pair_mask &= _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_zmm, x_80_zmm), x_40_zmm);
    pair_mask &= _mm512_cmpge_epu8_mask(previous_zmm, x_c0_zmm);

    __mmask64 danger_mask = 0;
    if (pair_mask) {
        // Rare path: some pair matched, refine the Asomtavruli third-byte range (E1 82 A0-E5)
        __mmask64 is_82_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0x82));
        __mmask64 in_a0_e5_mask = _mm512_cmplt_epu8_mask(_mm512_add_epi8(text_zmm, _mm512_set1_epi8(0x60)),
                                                         _mm512_set1_epi8(0x46));
        __mmask64 asomtavruli_pair_mask = pair_mask & is_82_mask & (in_a0_e5_mask >> 1);
        // Shift back to sequence start positions
        danger_mask = ((pair_mask & ~is_82_mask) >> 1) | (asomtavruli_pair_mask >> 1);
    }

    sz_assert_(danger_mask == sz_utf8_uncased_find_icelake_georgian_alarm_zmm_(text_zmm, load_mask) &&
               "Efficient Georgian alarm must match the reference implementation");
    return danger_mask;
}

/**
 *  @brief Fold Georgian text - only ASCII A-Z needs folding.
 *  @sa sz_utf8_uncased_rune_safe_georgian_k
 *
 *  Georgian Mkhedruli is caseless, so Georgian characters pass through unchanged.
 *  Only ASCII uppercase letters need folding for mixed Latin text.
 */
SZ_INTERNAL __m512i sz_utf8_uncased_find_icelake_georgian_fold_zmm_(__m512i text_zmm) {
    // ASCII A-Z range check: (byte - 'A') <= 25
    __m512i offset_a_zmm = _mm512_sub_epi8(text_zmm, _mm512_set1_epi8('A'));
    __mmask64 is_upper_mask = _mm512_cmple_epu8_mask(offset_a_zmm, _mm512_set1_epi8(25));
    return _mm512_mask_add_epi8(text_zmm, is_upper_mask, text_zmm, _mm512_set1_epi8(0x20));
}

/**
 *  @brief Georgian uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_georgian_k
 *
 *  This is the fastest non-ASCII kernel because Mkhedruli is caseless - no Georgian
 *  folding needed in the hot path. Only ASCII A-Z folding for mixed text.
 *
 *  @param haystack Pointer to the haystack string.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the full needle string.
 *  @param needle_length Length of the full needle in bytes.
 *  @param needle_metadata Pre-folded window content with probe positions.
 *  @param matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_uncased_find_icelake_georgian_( //
    sz_cptr_t haystack, sz_size_t haystack_length,            //
    sz_cptr_t needle, sz_size_t needle_length,                //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata, //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_find_icelake_scripted_( //
        sz_utf8_uncased_find_icelake_georgian_fold_zmm_,
        sz_utf8_uncased_find_icelake_georgian_alarm_efficiently_zmm_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Georgian Uncased Find

SZ_PUBLIC sz_cptr_t sz_utf8_uncased_find_icelake(  //
    sz_cptr_t haystack, sz_size_t haystack_length, //
    sz_cptr_t needle, sz_size_t needle_length,     //
    sz_utf8_uncased_needle_metadata_t *needle_metadata, sz_size_t *matched_length) {

    // Handle the obvious edge cases first
    if (needle_length == 0) {
        *matched_length = 0;
        return haystack;
    }

    // If the needle is entirely made of case-less characters - perform direct substring search
    int const is_unknown = needle_metadata->kernel_id == sz_utf8_uncased_rune_unknown_k;
    int const known_agnostic = needle_metadata->kernel_id == sz_utf8_uncased_rune_invariant_k;
    if (known_agnostic || (is_unknown && sz_utf8_uncased_violation_icelake(needle, needle_length) == SZ_NULL_CHAR)) {
        sz_cptr_t result = sz_find_skylake(haystack, haystack_length, needle, needle_length);
        *matched_length = result ? needle_length : 0;
        return result;
    }

    // Analyze needle to find the best safe window for each script
    if (is_unknown) {
        sz_utf8_uncased_needle_metadata_(needle, needle_length, needle_metadata);
        // If no SIMD-safe window found, fall back to serial immediately
        if (needle_metadata->kernel_id == sz_utf8_uncased_rune_fallback_serial_k)
            return sz_utf8_uncased_find_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                               matched_length);
    }

    // Dispatch to appropriate kernel
    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_ascii_invariant_k) {
        // Use XOR+VPTERNLOG+VPTESTNMB variants for better port distribution
        if (needle_metadata->folded_slice_length <= 3)
            return sz_utf8_uncased_find_icelake_ascii_3probe_( //
                haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
        else
            return sz_utf8_uncased_find_icelake_ascii_4probe_( //
                haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    }

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_western_europe_k)
        return sz_utf8_uncased_find_icelake_western_europe_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_central_europe_k)
        return sz_utf8_uncased_find_icelake_central_europe_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_greek_k)
        return sz_utf8_uncased_find_icelake_greek_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_armenian_k)
        return sz_utf8_uncased_find_icelake_armenian_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_vietnamese_k)
        return sz_utf8_uncased_find_icelake_vietnamese_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_cyrillic_k)
        return sz_utf8_uncased_find_icelake_cyrillic_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_georgian_k)
        return sz_utf8_uncased_find_icelake_georgian_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    // No suitable SIMD path found (needle has complex Unicode), fall back to serial
    needle_metadata->kernel_id = sz_utf8_uncased_rune_fallback_serial_k;
    return sz_utf8_uncased_find_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                       matched_length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_uncased_violation_icelake(sz_cptr_t str, sz_size_t length) {
    sz_u8_t const *text_cursor = (sz_u8_t const *)str;

    // Pre-computed constants
    __m512i const a_upper_u8x64 = _mm512_set1_epi8('A');
    __m512i const a_lower_u8x64 = _mm512_set1_epi8('a');
    __m512i const range26_u8x64 = _mm512_set1_epi8(26);
    __m512i const x80_u8x64 = _mm512_set1_epi8((char)0x80);
    __m512i const xc0_u8x64 = _mm512_set1_epi8((char)0xC0);
    __m512i const xc3_u8x64 = _mm512_set1_epi8((char)0xC3);
    __m512i const xe0_u8x64 = _mm512_set1_epi8((char)0xE0);
    __m512i const xf0_u8x64 = _mm512_set1_epi8((char)0xF0);
    __m512i const xf8_u8x64 = _mm512_set1_epi8((char)0xF8);

    // Single loop: advance by min(length, 61), check leads in first `block_length` positions
    while (length) {
        sz_size_t block_length = sz_min_of_two(length, 61);
        __mmask64 lead_mask = sz_u64_mask_until_(block_length);
        __mmask64 load_mask = sz_u64_clamp_mask_until_(length); // min(length, 64) bits
        __m512i text_u8x64 = _mm512_maskz_loadu_epi8(load_mask, text_cursor);

        // 1. ASCII letter check (zeros beyond string are fine - not letters)
        __mmask64 is_upper_mask = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_u8x64, a_upper_u8x64), range26_u8x64);
        __mmask64 is_lower_mask = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_u8x64, a_lower_u8x64), range26_u8x64);
        if (is_upper_mask | is_lower_mask) return sz_utf8_uncased_violation_serial(str, length);

        // 2. Check for non-ASCII in lead positions
        __mmask64 is_non_ascii_mask = _mm512_movepi8_mask(text_u8x64) & lead_mask;
        if (is_non_ascii_mask) {
            // 3. Identify UTF-8 lead bytes
            __mmask64 is_two_mask = _mm512_cmpeq_epi8_mask(_mm512_and_si512(text_u8x64, xe0_u8x64), xc0_u8x64) &
                                    lead_mask;
            __mmask64 is_three_mask = _mm512_cmpeq_epi8_mask(_mm512_and_si512(text_u8x64, xf0_u8x64), xe0_u8x64) &
                                      lead_mask;
            __mmask64 is_four_mask = _mm512_cmpeq_epi8_mask(_mm512_and_si512(text_u8x64, xf8_u8x64), xf0_u8x64) &
                                     lead_mask;

            // 4. Check 4-byte bicameral scripts (SMP): F0 with second byte 90/91/96/9D/9E
            if (is_four_mask) {
                __mmask64 after_f0_mask = is_four_mask << 1;
                __mmask64 is_90_mask = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0x90));
                __mmask64 is_91_mask = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0x91));
                __mmask64 is_96_mask = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0x96));
                __mmask64 is_9d_mask = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0x9D));
                __mmask64 is_9e_mask = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0x9E));
                if (after_f0_mask & (is_90_mask | is_91_mask | is_96_mask | is_9d_mask | is_9e_mask))
                    return sz_utf8_uncased_violation_serial(str, length);
            }

            // 5. Check 2-byte bicameral leads: C3-D6
            // C3-CF: Latin Extended (umlauts, accents, Eszett)
            // D0-D1: Cyrillic, D4-D6: Armenian (D6 needed for small letters U+0580+)
            if (is_two_mask) {
                __mmask64 is_bicameral_mask = _mm512_cmplt_epu8_mask( //
                    _mm512_sub_epi8(text_u8x64, xc3_u8x64), _mm512_set1_epi8(0x14));

                // Special case: C2 B5 = U+00B5 MICRO SIGN folds to Greek mu (U+03BC)
                __mmask64 is_c2_mask = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xC2)) & is_two_mask;
                if (is_c2_mask) {
                    __mmask64 after_c2_mask = is_c2_mask << 1;
                    __mmask64 is_b5_mask = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xB5));
                    if (after_c2_mask & is_b5_mask) return sz_utf8_uncased_violation_serial(str, length);
                }

                // Note: CA 80-BF includes both IPA Extensions (U+0280-02AF) and Spacing Modifier Letters
                // (U+02B0-02BF). Spacing Modifier Letters CAN appear in case fold expansions:
                // e.g., ẚ (U+1E9A) folds to [a, ʾ] where ʾ = U+02BE is a Spacing Modifier Letter.
                // So we must NOT exclude this range from bicameral check.

                if (is_bicameral_mask & is_two_mask) return sz_utf8_uncased_violation_serial(str, length);
            }

            // 6. Check 3-byte bicameral sequences
            if (is_three_mask) {
                // E1: Georgian, Greek Extended, Latin Extended Additional
                __mmask64 is_e1_mask = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xE1));
                if (is_e1_mask & is_three_mask) return sz_utf8_uncased_violation_serial(str, length);

                // EF: Fullwidth Latin
                __mmask64 is_ef_mask = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xEF));
                if (is_ef_mask & is_three_mask) return sz_utf8_uncased_violation_serial(str, length);

                // E2: Safe only for second byte 80-83
                __mmask64 is_e2_mask = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xE2)) & is_three_mask;
                if (is_e2_mask) {
                    __mmask64 after_e2_mask = is_e2_mask << 1;
                    __mmask64 e2_second_safe_mask = _mm512_cmplt_epu8_mask( //
                        _mm512_sub_epi8(text_u8x64, x80_u8x64), _mm512_set1_epi8(0x04));
                    if (after_e2_mask & ~e2_second_safe_mask) return sz_utf8_uncased_violation_serial(str, length);
                }

                // EA: Bicameral second bytes 99-9F, AD-AE
                __mmask64 is_ea_mask = _mm512_cmpeq_epi8_mask(text_u8x64, _mm512_set1_epi8((char)0xEA)) & is_three_mask;
                if (is_ea_mask) {
                    __mmask64 after_ea_mask = is_ea_mask << 1;
                    __mmask64 is_99_range_mask = _mm512_cmplt_epu8_mask( //
                        _mm512_sub_epi8(text_u8x64, _mm512_set1_epi8((char)0x99)), _mm512_set1_epi8(0x07));
                    __mmask64 is_ad_range_mask = _mm512_cmplt_epu8_mask( //
                        _mm512_sub_epi8(text_u8x64, _mm512_set1_epi8((char)0xAC)), _mm512_set1_epi8(0x03));
                    if (after_ea_mask & (is_99_range_mask | is_ad_range_mask))
                        return sz_utf8_uncased_violation_serial(str, length);
                }
            }
        }

        text_cursor += block_length;
        length -= block_length;
    }

    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_ordering_t sz_utf8_uncased_order_icelake(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                      sz_size_t b_length) {
    return sz_utf8_uncased_order_serial(a, a_length, b, b_length);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // SZ_USE_ICELAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_UNCASED_ICELAKE_H_
