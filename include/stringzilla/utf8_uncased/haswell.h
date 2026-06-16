/**
 *  @brief Haswell (AVX2) uncased UTF-8 search, comparison & invariance backend.
 *  @file include/stringzilla/utf8_uncased/haswell.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased.h
 *
 *  Ports the Ice Lake architecture to 32-byte YMM chunks: the needle classification, match
 *  verification, and danger-zone scanning stay ISA-independent in `serial.h`; per-script
 *  `fold`/`alarm` helpers become YMM functions; and the shared force-inlined `scripted_` driver
 *  walks the haystack in 32-byte steps. AVX-512 k-mask algebra maps onto two AVX2 domains:
 *  byte-mask vectors (for folds, which must feed masked adds) and 32-bit `VPMOVMSKB` integers
 *  (for alarms and probe filters, where scalar shifts replace k-mask shifts bit-for-bit).
 */
#ifndef STRINGZILLA_UTF8_UNCASED_HASWELL_H_
#define STRINGZILLA_UTF8_UNCASED_HASWELL_H_

#include "stringzilla/find/haswell.h" // `sz_find_haswell`
#include "stringzilla/utf8_uncased/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "bmi", "bmi2")
#endif

#pragma region Shared AVX2 Helpers

/**
 *  @brief Detects bytes in the unsigned range [range_start, range_start + range_length).
 *      AVX2 has no unsigned byte compares, so `(x − start) ≤ limit` is realized as
 *      `min_epu8(x − start, limit) == x − start`: the wrap-around subtraction maps the range
 *      onto [0, limit] and `VPMINUB` + `VPCMPEQB` realize the unsigned `≤` in two single-uop
 *      instructions - cheaper and clearer than the sign-flip `VPXOR` + `VPCMPGTB` alternative.
 */
SZ_INTERNAL __m256i sz_utf8_uncased_haswell_in_byte_range_(__m256i values_ymm, sz_u8_t range_start, sz_u8_t range_length) {
    __m256i offsets_ymm = _mm256_sub_epi8(values_ymm, _mm256_set1_epi8((char)range_start));
    return _mm256_cmpeq_epi8(_mm256_min_epu8(offsets_ymm, _mm256_set1_epi8((char)(range_length - 1))), offsets_ymm);
}

/**
 *  @brief Shifts the 32 source bytes right by one lane, so lane `i` holds byte `i − 1`; lane 0
 *      receives zero. This is the vector-domain equivalent of Ice Lake's `k-mask << 1` idiom:
 *      a continuation byte at lane 0 whose lead sits in the previous chunk stays unfolded in
 *      BOTH ports, and such positions can never start a match anyway (the needle's folded
 *      window always begins with a full rune, never a continuation byte).
 *      AVX2 `VPALIGNR` works per 128-bit lane, so a `VPERM2I128` first materializes the carry.
 */
SZ_INTERNAL __m256i sz_utf8_uncased_haswell_previous_bytes_(__m256i source_ymm) {
    __m256i carry_ymm = _mm256_permute2x128_si256(source_ymm, source_ymm, 0x08); // [zero, source.low]
    return _mm256_alignr_epi8(source_ymm, carry_ymm, 15);
}

/**
 *  @brief Shifts the 32 source bytes left by one lane, so lane `i` holds byte `i + 1`; lane 31
 *      receives zero. Vector-domain equivalent of Ice Lake's `k-mask >> 1`.
 */
SZ_INTERNAL __m256i sz_utf8_uncased_haswell_next_bytes_(__m256i source_ymm) {
    __m256i carry_ymm = _mm256_permute2x128_si256(source_ymm, source_ymm, 0x81); // [source.high, zero]
    return _mm256_alignr_epi8(carry_ymm, source_ymm, 1);
}

/** @brief First N bits set; BZHI keeps `n == 32` defined, unlike the `(1 << n) − 1` idiom. */
SZ_INTERNAL sz_u32_t sz_utf8_uncased_haswell_mask_until_(sz_size_t n) {
    return (sz_u32_t)_bzhi_u32(0xFFFFFFFFu, (unsigned)n);
}

/**
 *  @brief Loads up to 32 bytes through a zeroed stack buffer, never touching memory past
 *      `source + length`. The zero padding mirrors Ice Lake's `maskz` loads: zero bytes match
 *      no probe inside a valid window and trip no alarm, so tail chunks reuse the main-loop
 *      logic unchanged instead of branching into a separate epilogue.
 */
SZ_INTERNAL __m256i sz_utf8_uncased_haswell_load_padded_ymm_(sz_cptr_t source, sz_size_t length) {
    sz_u8_t buffer[32] = {0};
    for (sz_size_t byte_index = 0; byte_index < length; ++byte_index) buffer[byte_index] = (sz_u8_t)source[byte_index];
    return _mm256_lddqu_si256((__m256i const *)buffer);
}

/**
 *  @brief Loads up to 16 bytes for candidate-window verification without over-reading the
 *      haystack: the fast full load is taken whenever 16 bytes remain, and only the last few
 *      candidates near the haystack end pay for the zero-padded stack copy.
 */
SZ_INTERNAL __m128i sz_utf8_uncased_haswell_load_window_xmm_(sz_cptr_t source, sz_size_t available) {
    if (available >= 16) return _mm_lddqu_si128((__m128i const *)source);
    sz_u8_t buffer[16] = {0};
    for (sz_size_t byte_index = 0; byte_index < available; ++byte_index)
        buffer[byte_index] = (sz_u8_t)source[byte_index];
    return _mm_lddqu_si128((__m128i const *)buffer);
}

#pragma endregion // Shared AVX2 Helpers

#pragma region ASCII Uncased Find

/**
 *  @brief Fold a YMM register using ASCII case folding rules.
 *  @sa sz_utf8_uncased_rune_ascii_invariant_k
 */
SZ_INTERNAL __m256i sz_utf8_uncased_find_haswell_ascii_fold_ymm_(__m256i text_ymm) {
    // Only fold bytes in range A-Z; the masked add avoids `VPBLENDVB` (2 uops on Haswell)
    __m256i is_ascii_upper_ymm = sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 'A', 26);
    return _mm256_add_epi8(text_ymm, _mm256_and_si256(is_ascii_upper_ymm, _mm256_set1_epi8(0x20)));
}

/**
 *  @brief 3-probe ASCII uncased search over 32-byte chunks.
 *
 *  For needles with folded_slice_length ≤ 3, probes at positions 0, mid, last cover ALL bytes
 *  of the window, so no candidate-window verification is needed - candidates go straight to
 *  head/tail validation. Unlike the Ice Lake shifted-loads variant, the chunk is folded ONCE
 *  and the probe equality masks are shifted as 32-bit `VPMOVMSKB` integers: with windows ≤ 16
 *  bytes every chunk still exposes ≥ 17 valid start positions per iteration.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_uncased_find_haswell_ascii_3probe_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                         //
    sz_cptr_t needle, sz_size_t needle_length,                             //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,     //
    sz_size_t *matched_length) {

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;

    // For ≤3 bytes: positions 0, mid, last cover ALL positions
    // 1-byte: 0=last, 2-byte: 0,1, 3-byte: 0,1,2
    sz_size_t const offset_second = folded_window_length / 2;
    sz_size_t const offset_last = folded_window_length - 1;

    __m256i const probe_first_ymm = _mm256_set1_epi8((char)needle_metadata->folded_slice[0]);
    __m256i const probe_second_ymm = _mm256_set1_epi8((char)needle_metadata->folded_slice[offset_second]);
    __m256i const probe_last_ymm = _mm256_set1_epi8((char)needle_metadata->folded_slice[offset_last]);

    sz_cptr_t haystack_ptr = haystack;
    while (haystack_ptr < haystack_end) {
        sz_size_t const available = (sz_size_t)(haystack_end - haystack_ptr);
        if (available < folded_window_length) break;
        sz_size_t const chunk_size = available < 32 ? available : 32;
        sz_size_t const valid_starts = chunk_size - folded_window_length + 1;

        __m256i text_ymm = available >= 32 ? _mm256_lddqu_si256((__m256i const *)haystack_ptr)
                                           : sz_utf8_uncased_haswell_load_padded_ymm_(haystack_ptr, chunk_size);
        __m256i folded_ymm = sz_utf8_uncased_find_haswell_ascii_fold_ymm_(text_ymm);

        sz_u32_t matches = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(folded_ymm, probe_first_ymm));
        matches &= (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(folded_ymm, probe_second_ymm)) >> offset_second;
        matches &= (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(folded_ymm, probe_last_ymm)) >> offset_last;
        matches &= sz_utf8_uncased_haswell_mask_until_(valid_starts);

        for (; matches; matches &= matches - 1) {
            sz_size_t const candidate_offset = (sz_size_t)sz_u32_ctz(matches);
            sz_cptr_t const haystack_candidate_ptr = haystack_ptr + candidate_offset;

            // No window verification needed - probes cover all positions,
            // go directly to head/tail validation
            sz_cptr_t match = sz_utf8_uncased_verify_match_(                                      //
                haystack, haystack_length,                                                                 //
                needle, needle_length,                                                                     //
                haystack_candidate_ptr - haystack, folded_window_length,                                   //
                needle_metadata->offset_in_unfolded,                                                       //
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, //
                matched_length);
            if (match) { return match; }
        }
        haystack_ptr += valid_starts;
    }

    return SZ_NULL_CHAR;
}

#pragma endregion // ASCII Uncased Find

#pragma region Scripted Uncased Find

/** @brief Folds one YMM register of haystack text using script-specific rules. */
typedef __m256i (*sz_utf8_uncased_fold_ymm_t_)(__m256i text_ymm);

/**
 *  @brief Flags positions of "danger" characters that fold to a different byte width.
 *  @param load_mask Bitmask of the bytes actually loaded from the haystack, for tail-safe range checks.
 */
typedef sz_u32_t (*sz_utf8_uncased_alarm_ymm_t_)(__m256i text_ymm, sz_u32_t load_mask);

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
 *  Tail chunks shorter than 32 bytes are zero-padded through a stack buffer, so they take the
 *  exact main-loop path; and like on Ice Lake, two danger-scan rules preserve correctness for
 *  expanding folds ('ẞ' → "ss"): an alarmed chunk is danger-scanned in FULL (not just its valid
 *  start positions), and once fewer than `folded_window_length` bytes remain, the leftover tail
 *  gets a final danger scan - a haystack span SHORTER than the folded window can still hide a
 *  real match there.
 *
 *  @param fold Script-specific YMM case-folding callback.
 *  @param alarm Script-specific danger detection callback, or NULL if the script has no
 *      danger characters: the danger branch disappears and the full step is used.
 */
SZ_FORCE_INLINE sz_cptr_t sz_utf8_uncased_find_haswell_scripted_( //
    sz_utf8_uncased_fold_ymm_t_ fold,                             //
    sz_utf8_uncased_alarm_ymm_t_ alarm,                           //
    sz_cptr_t haystack, sz_size_t haystack_length,                         //
    sz_cptr_t needle, sz_size_t needle_length,                             //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,     //
    sz_size_t *matched_length) {

    sz_assert_(needle_metadata && "needle_metadata must be provided");
    sz_assert_(needle_metadata->folded_slice_length > 0 && "folded window must be non-empty");

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_assert_(folded_window_length <= 16 && "expect folded needle part to fit in XMM registers");

    // Pre-load folded window into XMM; the byte-mask vector replicates Ice Lake's `maskz` window
    // load: bytes past the window are zeroed BEFORE folding, so a lead byte at the window edge
    // never borrows fold context from haystack bytes outside the window
    sz_u32_t const folded_window_mask = sz_utf8_uncased_haswell_mask_until_(folded_window_length);
    __m128i const needle_window_xmm = _mm_lddqu_si128((__m128i const *)needle_metadata->folded_slice);
    __m128i const window_keep_xmm = _mm_cmpgt_epi8(_mm_set1_epi8((char)folded_window_length),
                                                   _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));

    // 4 probe positions
    sz_size_t const offset_second = needle_metadata->probe_second;
    sz_size_t const offset_third = needle_metadata->probe_third;
    sz_size_t const offset_last = folded_window_length - 1;

    __m256i const probe_first_ymm = _mm256_set1_epi8((char)needle_metadata->folded_slice[0]);
    __m256i const probe_second_ymm = _mm256_set1_epi8((char)needle_metadata->folded_slice[offset_second]);
    __m256i const probe_third_ymm = _mm256_set1_epi8((char)needle_metadata->folded_slice[offset_third]);
    __m256i const probe_last_ymm = _mm256_set1_epi8((char)needle_metadata->folded_slice[offset_last]);

    // Pre-load the first folded rune for danger zone matching
    sz_rune_t needle_first_safe_folded_rune = 0;
    if (alarm) {
        sz_rune_length_t rune_byte_length;
        sz_rune_parse_unchecked((sz_cptr_t)(needle_metadata->folded_slice), &needle_first_safe_folded_rune,
                                &rune_byte_length);
    }

    sz_cptr_t haystack_ptr = haystack;
    while (haystack_ptr < haystack_end) {
        sz_size_t const available = (sz_size_t)(haystack_end - haystack_ptr);
        if (available < folded_window_length) break;

        sz_size_t const chunk_size = available < 32 ? available : 32;
        sz_size_t const valid_starts = chunk_size - folded_window_length + 1;
        // For danger detection across chunk boundaries, reduce step size to ensure
        // 3-byte patterns at chunk end are fully visible in the next chunk.
        // For tail chunks (valid_starts <= 2), step = 1 ensures progress.
        // Scripts without danger characters advance by the full window count.
        sz_size_t const step = !alarm ? valid_starts : valid_starts > 2 ? valid_starts - 2 : 1;
        sz_u32_t const load_mask = sz_utf8_uncased_haswell_mask_until_(chunk_size);
        sz_u32_t const valid_mask = sz_utf8_uncased_haswell_mask_until_(valid_starts);

        __m256i text_ymm = available >= 32 ? _mm256_lddqu_si256((__m256i const *)haystack_ptr)
                                           : sz_utf8_uncased_haswell_load_padded_ymm_(haystack_ptr, chunk_size);

        // Check for anomalies (characters that fold to different byte widths)
        if (alarm) {
            sz_u32_t danger_mask = alarm(text_ymm, load_mask);
            if (danger_mask) {
                // The danger zone handler scans for the needle's first safe rune (at offset_in_unfolded).
                // The whole chunk is scanned, not just `valid_starts` positions: an expanding danger
                // character makes the haystack span SHORTER than the folded window, so a real match
                // can start within the window's length of the chunk end.
                sz_cptr_t match = sz_utf8_uncased_find_in_danger_zone_( //
                    haystack, haystack_length,                                   //
                    needle, needle_length,                                       //
                    haystack_ptr, chunk_size,                                    // extended danger zone
                    needle_first_safe_folded_rune,                               // pivot point
                    needle_metadata->offset_in_unfolded,                         // its location in the needle
                    matched_length);
                if (match) return match;
                haystack_ptr += step;
                continue;
            }
        }

        // Fold once, then filter candidates with 4 probe positions: scalar shifts over the
        // 32-bit movemasks substitute Ice Lake's k-mask shifts bit-for-bit
        __m256i folded_ymm = fold(text_ymm);
        sz_u32_t matches = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(folded_ymm, probe_first_ymm));
        matches &= (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(folded_ymm, probe_second_ymm)) >> offset_second;
        matches &= (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(folded_ymm, probe_third_ymm)) >> offset_third;
        matches &= (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(folded_ymm, probe_last_ymm)) >> offset_last;
        matches &= valid_mask;

        // Candidate Verification
        for (; matches; matches &= matches - 1) {
            sz_size_t const candidate_offset = (sz_size_t)sz_u32_ctz(matches);
            sz_cptr_t const haystack_candidate_ptr = haystack_ptr + candidate_offset;

            // Re-fold the candidate window: zero-extending the 16-byte view into a YMM register
            // keeps the script fold's per-register semantics identical to the main chunk fold
            __m128i window_xmm = sz_utf8_uncased_haswell_load_window_xmm_(
                haystack_candidate_ptr, (sz_size_t)(haystack_end - haystack_candidate_ptr));
            window_xmm = _mm_and_si128(window_xmm, window_keep_xmm);
            __m256i folded_window_ymm = fold(_mm256_inserti128_si256(_mm256_setzero_si256(), window_xmm, 0));
            sz_u32_t window_equal_mask = (sz_u32_t)_mm_movemask_epi8(
                _mm_cmpeq_epi8(_mm256_castsi256_si128(folded_window_ymm), needle_window_xmm));
            if ((window_equal_mask & folded_window_mask) != folded_window_mask) continue;

            sz_cptr_t match = sz_utf8_uncased_verify_match_(                    //
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
        sz_cptr_t match = sz_utf8_uncased_find_in_danger_zone_( //
            haystack, haystack_length,                                   //
            needle, needle_length,                                       //
            haystack_ptr, (sz_size_t)(haystack_end - haystack_ptr),      // the unprobed tail
            needle_first_safe_folded_rune,                               // pivot point
            needle_metadata->offset_in_unfolded,                         // its location in the needle
            matched_length);
        if (match) { return match; }
    }

    return SZ_NULL_CHAR;
}

/**
 *  @brief 4-probe ASCII uncased search: the shared scripted driver with the ASCII fold
 *      and no alarm - ASCII never changes byte width when folded, so the danger machinery
 *      compiles away entirely and the step covers every valid start position.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_uncased_find_haswell_ascii_4probe_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                         //
    sz_cptr_t needle, sz_size_t needle_length,                             //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,     //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_find_haswell_scripted_( //
        sz_utf8_uncased_find_haswell_ascii_fold_ymm_,
        (sz_utf8_uncased_alarm_ymm_t_)SZ_NULL, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Scripted Uncased Find

#pragma region Western European Uncased Find

/**
 *  @brief Fold a YMM register using Western European case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_western_europe_k
 *
 *  Handles ASCII A-Z, the Latin-1 Supplement uppercase range 'À'-'Þ' (C3 80-9E → +0x20,
 *  excluding the caseless '×' C3 97), and 'ß' (U+00DF, C3 9F) → "ss" where BOTH bytes of the
 *  pair become 's' so the folded image matches the needle's "ss".
 */
SZ_INTERNAL __m256i sz_utf8_uncased_find_haswell_western_europe_fold_ymm_(__m256i text_ymm) {
    __m256i result_ymm = sz_utf8_uncased_find_haswell_ascii_fold_ymm_(text_ymm);
    __m256i previous_bytes_ymm = sz_utf8_uncased_haswell_previous_bytes_(text_ymm);
    __m256i is_after_c3_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xC3));

    // 1. Handle Eszett: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73): the second-byte flag
    //    propagates one lane back to also rewrite the C3 lead
    __m256i is_eszett_second_ymm = _mm256_and_si256(is_after_c3_ymm,
                                                    _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x9F)));
    __m256i is_eszett_ymm = _mm256_or_si256(is_eszett_second_ymm, sz_utf8_uncased_haswell_next_bytes_(is_eszett_second_ymm));
    result_ymm = _mm256_or_si256(_mm256_andnot_si256(is_eszett_ymm, result_ymm),
                                 _mm256_and_si256(is_eszett_ymm, _mm256_set1_epi8('s')));

    // 2. Handle Latin-1 supplement uppercase letters (C3 80-9E) → add 0x20,
    //    excluding '×' (C3 97, no case variant) and 'ß' (C3 9F, already handled above)
    __m256i is_97_ymm = _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x97));
    __m256i is_latin1_upper_ymm = _mm256_and_si256(
        is_after_c3_ymm, _mm256_andnot_si256(_mm256_or_si256(is_eszett_second_ymm, is_97_ymm),
                                             sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0x80, 0x1F)));
    result_ymm = _mm256_add_epi8(result_ymm, _mm256_and_si256(is_latin1_upper_ymm, _mm256_set1_epi8(0x20)));
    return result_ymm;
}

/**
 *  @brief Alarm function for Western Europe danger zone detection.
 *
 *  Detects positions where danger characters occur that require special handling:
 *  - E1 BA 96-9E: 'ẖ'-'ẞ' all expand to ASCII-led sequences when folded; the third-byte
 *    qualification matters because the rest of E1 BA covers Vietnamese letters that fold
 *    in place - flagging them blanket-style would send dense Vietnamese text into the
 *    serial danger-zone scanner on every chunk
 *  - E2 84 AA/AB: 'K' (U+212A) → 'k' and 'Å' (U+212B) → 'å' (3 bytes → 1-2 bytes)
 *  - EF AC 80-86: Latin ligatures 'ﬀ'-'ﬆ' → ASCII pairs/triples
 *  - C5 BF: 'ſ' (U+017F) → 's' (2 bytes → 1 byte)
 *  - C5 B8: 'Ÿ' (U+0178) → 'ÿ' (C3 BF), crosses lead bytes
 *  - C3 9F: 'ß' (U+00DF) → "ss" (1 rune → 2 runes)
 *
 *  All pair tests run as scalar shift+AND over the compare movemasks - the same bit algebra
 *  as Ice Lake's k-masks, including the boundary behavior where a lead at lane 31 defers to
 *  the next (overlapping) chunk.
 */
SZ_INTERNAL sz_u32_t sz_utf8_uncased_find_haswell_western_europe_alarm_ymm_(__m256i text_ymm,
                                                                                     sz_u32_t load_mask) {
    sz_unused_(load_mask); // Present for the shared `sz_utf8_uncased_alarm_ymm_t_` signature

    // Lead bytes (5 CMPEQ + movemask)
    sz_u32_t is_e1_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xE1)));
    sz_u32_t is_e2_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xE2)));
    sz_u32_t is_ef_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xEF)));
    sz_u32_t is_c5_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xC5)));
    sz_u32_t is_c3_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xC3)));

    // Second/third bytes (8 CMPEQ + movemask)
    sz_u32_t is_ba_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xBA)));
    sz_u32_t is_84_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x84)));
    sz_u32_t is_ac_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xAC)));
    sz_u32_t is_bf_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xBF)));
    sz_u32_t is_9f_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x9F)));
    sz_u32_t is_aa_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xAA)));
    sz_u32_t is_ab_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xAB)));
    sz_u32_t is_b8_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xB8)));

    // E1 BA is dangerous only when the third byte is 96-9E; the refinement is branched-over
    // since most non-Vietnamese text has no E1 BA pairs at all
    sz_u32_t is_e1_ba_danger_mask = (is_e1_mask << 1) & is_ba_mask;
    if (is_e1_ba_danger_mask) {
        sz_u32_t is_expanding_third_mask = (sz_u32_t)_mm256_movemask_epi8(
            sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0x96, 0x09));
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
 *  @brief Western European uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_western_europe_k
 */
SZ_INTERNAL sz_cptr_t sz_utf8_uncased_find_haswell_western_europe_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                           //
    sz_cptr_t needle, sz_size_t needle_length,                               //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,       //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_find_haswell_scripted_( //
        sz_utf8_uncased_find_haswell_western_europe_fold_ymm_,
        sz_utf8_uncased_find_haswell_western_europe_alarm_ymm_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Western European Uncased Find

#pragma region Central European Uncased Find

/**
 *  @brief Fold a YMM register using Central European case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_central_europe_k
 *
 *  Latin-1 Supplement folds with +0x20 (C3 80-9E, except '×' C3 97); Latin Extended-A folds
 *  with +1 on a parity pattern that flips across sub-ranges:
 *  - C4 80-B7 (U+0100-U+0137): uppercase = EVEN second bytes
 *  - C4 B9-BD (U+0139-U+013D): uppercase = ODD ('Ĺ','Ļ','Ľ'); 'ĸ' (C4 B8) is caseless and
 *    'Ŀ' (C4 BF) folds across leads to 'ŀ' (C5 80), so it is routed through the alarm instead
 *  - C5 81-87 (U+0141-U+0147): uppercase = ODD ('Ł','Ń','Ņ','Ň')
 *  - C5 8A-B6 (U+014A-U+0176): uppercase = EVEN ('Ŋ'-'Ŷ')
 *  - C5 B9-BD (U+0179-U+017D): uppercase = ODD ('Ź','Ż','Ž')
 */
SZ_INTERNAL __m256i sz_utf8_uncased_find_haswell_central_europe_fold_ymm_(__m256i text_ymm) {
    __m256i result_ymm = sz_utf8_uncased_find_haswell_ascii_fold_ymm_(text_ymm);
    __m256i previous_bytes_ymm = sz_utf8_uncased_haswell_previous_bytes_(text_ymm);
    __m256i is_after_c3_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xC3));
    __m256i is_after_c4_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xC4));
    __m256i is_after_c5_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xC5));

    // 1. Latin-1 Supplement: C3 80-9E → +0x20, except '×' (C3 97)
    __m256i is_latin1_upper_ymm = _mm256_and_si256(
        is_after_c3_ymm, _mm256_andnot_si256(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x97)),
                                             sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0x80, 0x1F)));
    result_ymm = _mm256_add_epi8(result_ymm, _mm256_and_si256(is_latin1_upper_ymm, _mm256_set1_epi8(0x20)));

    // 2. Latin Extended-A: +1 on the parity sub-ranges; the codepoint's low bit equals the
    //    continuation byte's low bit, so byte parity stands in for codepoint parity
    __m256i is_odd_ymm = _mm256_cmpeq_epi8(_mm256_and_si256(text_ymm, _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01));
    __m256i c4_ranges_ymm = _mm256_or_si256(
        _mm256_andnot_si256(is_odd_ymm, sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0x80, 0x38)), // C4 80-B7 even
        _mm256_and_si256(is_odd_ymm, sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0xB9, 0x05)));   // C4 B9-BD odd
    __m256i c5_ranges_ymm = _mm256_or_si256(
        _mm256_and_si256(is_odd_ymm, _mm256_or_si256(sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0x81, 0x07),   // 81-87
                                                     sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0xB9, 0x05))), // B9-BD
        _mm256_andnot_si256(is_odd_ymm, sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0x8A, 0x2D))); // C5 8A-B6 even
    __m256i fold_extended_ymm = _mm256_or_si256(_mm256_and_si256(is_after_c4_ymm, c4_ranges_ymm),
                                                _mm256_and_si256(is_after_c5_ymm, c5_ranges_ymm));
    result_ymm = _mm256_add_epi8(result_ymm, _mm256_and_si256(fold_extended_ymm, _mm256_set1_epi8(0x01)));
    return result_ymm;
}

/**
 *  @brief Alarm function for Central Europe danger zone detection.
 *
 *  Detects positions where danger characters occur that require special handling:
 *  - E2 84: 'K' Kelvin sign (E2 84 AA, 3 bytes → 1 byte)
 *  - C3 9F: 'ß' (U+00DF) → "ss" (1 rune → 2 runes)
 *  - C4 B0: 'İ' (U+0130) → "i̇" (2 bytes → 3 bytes)
 *  - C4 BF: 'Ŀ' (U+013F) → 'ŀ' (C5 80), crosses lead bytes
 *  - C5 BF: 'ſ' (U+017F) → 's' (2 bytes → 1 byte)
 *  - C5 B8: 'Ÿ' (U+0178) → 'ÿ' (C3 BF), crosses lead bytes
 *  - EF AC 80-86: Latin ligatures 'ﬀ'-'ﬆ' → ASCII pairs/triples
 */
SZ_INTERNAL sz_u32_t sz_utf8_uncased_find_haswell_central_europe_alarm_ymm_(__m256i text_ymm,
                                                                                     sz_u32_t load_mask) {
    sz_unused_(load_mask); // Present for the shared `sz_utf8_uncased_alarm_ymm_t_` signature

    // Lead bytes (5 CMPEQ + movemask)
    sz_u32_t is_e2_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xE2)));
    sz_u32_t is_c3_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xC3)));
    sz_u32_t is_c4_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xC4)));
    sz_u32_t is_c5_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xC5)));
    sz_u32_t is_ef_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xEF)));

    // Second bytes (6 CMPEQ + movemask)
    sz_u32_t is_84_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x84)));
    sz_u32_t is_9f_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x9F)));
    sz_u32_t is_b0_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xB0)));
    sz_u32_t is_bf_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xBF)));
    sz_u32_t is_ac_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xAC)));
    sz_u32_t is_b8_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xB8)));

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
 *  @brief Central European uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_central_europe_k
 */
SZ_INTERNAL sz_cptr_t sz_utf8_uncased_find_haswell_central_europe_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                           //
    sz_cptr_t needle, sz_size_t needle_length,                               //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,       //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_find_haswell_scripted_( //
        sz_utf8_uncased_find_haswell_central_europe_fold_ymm_,
        sz_utf8_uncased_find_haswell_central_europe_alarm_ymm_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Central European Uncased Find

#pragma region Cyrillic Uncased Find

/**
 *  @brief Fold a YMM register using Cyrillic case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_cyrillic_k
 *
 *  Basic Cyrillic has a clean high-nibble pattern on the second byte after a D0 lead:
 *  8x → +0x10 ('Ѐ'-'Џ' land in the D1 block), 9x → +0x20 ('А'-'П' stay under D0),
 *  Ax → −0x20 ('Р'-'Я' land in the D1 block), Bx → 0 (already lowercase). One `VPSHUFB`
 *  lookup replaces 3 range comparisons + 3 masked adds; the table is mirrored into both
 *  128-bit lanes since `VPSHUFB` works per lane. Extended Cyrillic (D2/D3) needles are
 *  BANNED at classification time, so only D0 continuations need folding.
 */
SZ_INTERNAL __m256i sz_utf8_uncased_find_haswell_cyrillic_fold_ymm_(__m256i text_ymm) {
    __m256i result_ymm = sz_utf8_uncased_find_haswell_ascii_fold_ymm_(text_ymm);
    __m256i previous_bytes_ymm = sz_utf8_uncased_haswell_previous_bytes_(text_ymm);
    __m256i is_after_d0_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xD0));

    // Second-byte offsets keyed by the high nibble: 8 → +0x10, 9 → +0x20, A → −0x20 (0xE0)
    __m256i const cyrillic_offset_lut = _mm256_setr_epi8(              //
        0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x20, (char)0xE0, 0, 0, 0, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x20, (char)0xE0, 0, 0, 0, 0, 0);
    __m256i high_nibbles_ymm = _mm256_and_si256(_mm256_srli_epi16(text_ymm, 4), _mm256_set1_epi8(0x0F));
    __m256i offsets_ymm = _mm256_shuffle_epi8(cyrillic_offset_lut, high_nibbles_ymm);
    result_ymm = _mm256_add_epi8(result_ymm, _mm256_and_si256(is_after_d0_ymm, offsets_ymm));

    // Lead fixup: 'Ѐ'-'Џ' (seconds 80-8F) and 'Р'-'Я' (seconds A0-AF) have lowercase in the D1
    // block, so their D0 lead takes a masked +1; 'А'-'П' (seconds 90-9F) stay under D0
    __m256i next_bytes_ymm = sz_utf8_uncased_haswell_next_bytes_(text_ymm);
    __m256i is_d0_ymm = _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xD0));
    __m256i needs_d1_ymm = _mm256_and_si256(
        is_d0_ymm, _mm256_or_si256(sz_utf8_uncased_haswell_in_byte_range_(next_bytes_ymm, 0x80, 0x10),
                                   sz_utf8_uncased_haswell_in_byte_range_(next_bytes_ymm, 0xA0, 0x10)));
    result_ymm = _mm256_add_epi8(result_ymm, _mm256_and_si256(needs_d1_ymm, _mm256_set1_epi8(0x01)));
    return result_ymm;
}

/**
 *  @brief Alarm function for Cyrillic danger zone detection.
 *
 *  Basic Cyrillic itself never changes byte width when folded, and Extended Cyrillic needles
 *  (D2/D3 leads) are banned at needle-analysis time. The one haystack-side hazard is Cyrillic
 *  Extended-C: 'ᲀ'-'ᲈ' (U+1C80-1C88, E1 B2 80-88) fold INTO basic 2-byte Cyrillic letters
 *  ('в', 'д', 'о', 'с', 'т', 'ъ', 'ѣ'), so a 3-byte haystack character can match a 2-byte
 *  needle character and must go through the serial danger-zone scanner. The E1 B2 pair is
 *  absent from virtually all real Cyrillic text, so the third-byte refinement hides behind
 *  a branch and the hot path is two compares.
 */
SZ_INTERNAL sz_u32_t sz_utf8_uncased_find_haswell_cyrillic_alarm_ymm_(__m256i text_ymm, sz_u32_t load_mask) {
    sz_unused_(load_mask); // Present for the shared `sz_utf8_uncased_alarm_ymm_t_` signature
    sz_u32_t is_e1_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xE1)));
    sz_u32_t is_b2_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xB2)));
    sz_u32_t danger_mask = (is_e1_mask << 1) & is_b2_mask;
    if (danger_mask) {
        sz_u32_t is_folding_third_mask = (sz_u32_t)_mm256_movemask_epi8(
            sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0x80, 0x09));
        danger_mask &= is_folding_third_mask >> 1;
    }
    return danger_mask;
}

/**
 *  @brief Cyrillic uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_cyrillic_k
 */
SZ_INTERNAL sz_cptr_t sz_utf8_uncased_find_haswell_cyrillic_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                     //
    sz_cptr_t needle, sz_size_t needle_length,                         //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata, //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_find_haswell_scripted_( //
        sz_utf8_uncased_find_haswell_cyrillic_fold_ymm_,
        sz_utf8_uncased_find_haswell_cyrillic_alarm_ymm_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Cyrillic Uncased Find

#pragma region Armenian Uncased Find

/**
 *  @brief Fold a YMM register using Armenian case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_armenian_k
 *
 *  Armenian uppercase spans two lead bytes and folds into three target blocks:
 *  - D4 B1-BF: 'Ա'-'Ձ' → D5 A1-AF 'ա'-'ձ' (second −0x10, lead D4 → D5)
 *  - D5 80-8F: 'Ղ'-'Տ' → D5 B0-BF 'ղ'-'տ' (second +0x30, lead unchanged)
 *  - D5 90-96: 'Ր'-'Ֆ' → D6 80-86 'ր'-'ֆ' (second −0x10, lead D5 → D6)
 *
 *  Both lead rewrites are a +1 increment (D4 → D5, D5 → D6), so the second-byte flags
 *  propagate one lane back through `next_bytes` and join the single merged offset add -
 *  all rule masks flag disjoint byte positions. The D4 range checks only the lower bound,
 *  mirroring the Ice Lake reference: valid continuation bytes never exceed BF.
 */
SZ_INTERNAL __m256i sz_utf8_uncased_find_haswell_armenian_fold_ymm_(__m256i text_ymm) {
    __m256i result_ymm = sz_utf8_uncased_find_haswell_ascii_fold_ymm_(text_ymm);
    __m256i previous_bytes_ymm = sz_utf8_uncased_haswell_previous_bytes_(text_ymm);
    __m256i is_after_d4_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xD4));
    __m256i is_after_d5_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xD5));

    // Second-byte classes; the [B1, FF] range realizes the unbounded `≥ B1` check
    __m256i is_d4_upper_ymm = _mm256_and_si256(is_after_d4_ymm, //
                                               sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0xB1, 0x4F));
    __m256i is_d5_low_ymm = _mm256_and_si256(is_after_d5_ymm, //
                                             sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0x80, 0x10));
    __m256i is_d5_high_ymm = _mm256_and_si256(is_after_d5_ymm, //
                                              sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0x90, 0x07));

    // Both −0x10 classes also bump their lead by one block (D4 → D5, D5 → D6)
    __m256i is_minus_10_ymm = _mm256_or_si256(is_d4_upper_ymm, is_d5_high_ymm);
    __m256i lead_plus_one_ymm = sz_utf8_uncased_haswell_next_bytes_(is_minus_10_ymm);

    // Disjoint positions merge into ONE offset vector and a single add
    __m256i offsets_ymm = _mm256_and_si256(is_minus_10_ymm, _mm256_set1_epi8((char)0xF0));
    offsets_ymm = _mm256_or_si256(offsets_ymm, _mm256_and_si256(is_d5_low_ymm, _mm256_set1_epi8(0x30)));
    offsets_ymm = _mm256_or_si256(offsets_ymm, _mm256_and_si256(lead_plus_one_ymm, _mm256_set1_epi8(0x01)));
    return _mm256_add_epi8(result_ymm, offsets_ymm);
}

/**
 *  @brief Alarm function for Armenian danger zone detection.
 *
 *  Detects positions where danger characters occur that require special handling:
 *  - D6 87: 'և' (U+0587) → "եւ" (D5 A5 D6 82), the Ech-Yiwn ligature (2 bytes → 4 bytes)
 *  - EF AC 93-97: presentation-form ligatures 'ﬓ'-'ﬗ' (U+FB13-U+FB17) → 2 codepoints each
 *
 *  The EF AC pair alarms without a third-byte refinement, exactly like the Ice Lake
 *  reference: the only EF AC neighbors are the Latin/Hebrew presentation forms, which
 *  never appear inside Armenian haystacks, so the coarser test costs nothing in practice.
 */
SZ_INTERNAL sz_u32_t sz_utf8_uncased_find_haswell_armenian_alarm_ymm_(__m256i text_ymm, sz_u32_t load_mask) {
    sz_unused_(load_mask); // Present for the shared `sz_utf8_uncased_alarm_ymm_t_` signature

    // Lead bytes (2 CMPEQ + movemask)
    sz_u32_t is_d6_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xD6)));
    sz_u32_t is_ef_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xEF)));

    // Second bytes (2 CMPEQ + movemask)
    sz_u32_t is_87_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x87)));
    sz_u32_t is_ac_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xAC)));

    // Danger mask construction
    return ((is_d6_mask << 1) & is_87_mask) | // Ech-Yiwn (D6 87)
           ((is_ef_mask << 1) & is_ac_mask);  // Ligatures (EF AC xx)
}

/**
 *  @brief Armenian uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_armenian_k
 */
SZ_INTERNAL sz_cptr_t sz_utf8_uncased_find_haswell_armenian_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                     //
    sz_cptr_t needle, sz_size_t needle_length,                         //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata, //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_find_haswell_scripted_( //
        sz_utf8_uncased_find_haswell_armenian_fold_ymm_,
        sz_utf8_uncased_find_haswell_armenian_alarm_ymm_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Armenian Uncased Find

#pragma region Greek Uncased Find

/**
 *  @brief Fold a YMM register using Greek case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_greek_k
 *
 *  Monotonic Greek folds entirely on the byte after a CE/CF/C2 lead:
 *  - CE 91-9F: 'Α'-'Ο' → CE B1-BF 'α'-'ο' (second +0x20)
 *  - CE A0-A9: 'Π'-'Ω' → CF 80-89 'π'-'ω' (second −0x20, lead CE → CF)
 *  - CE 86: 'Ά' → CE AC 'ά' (second +0x26)
 *  - CE 88-8A: 'Έ'-'Ί' → CE AD-AF 'έ'-'ί' (second +0x25)
 *  - CE 8C: 'Ό' → CF 8C 'ό' (lead change only)
 *  - CE 8E-8F: 'Ύ','Ώ' → CF 8D-8E 'ύ','ώ' (second −1, lead CE → CF)
 *  - CE AA-AB: 'Ϊ','Ϋ' → CF 8A-8B 'ϊ','ϋ' (second −0x20, lead CE → CF)
 *  - CF 82: 'ς' → CF 83 'σ' (final sigma, +1)
 *  - C2 B5: 'µ' → CE BC 'μ' (micro sign joins Greek mu: lead +0x0C, second +0x07)
 *
 *  Every rule hits DISJOINT byte positions, so the per-rule deltas merge into one offset
 *  vector with AND/OR ops and a single add applies them all. All CE → CF lead rewrites are
 *  a +1 increment, propagated back from the second-byte flags through `next_bytes` like
 *  the Eszett rewrite in the Western European fold.
 */
SZ_INTERNAL __m256i sz_utf8_uncased_find_haswell_greek_fold_ymm_(__m256i text_ymm) {
    __m256i result_ymm = sz_utf8_uncased_find_haswell_ascii_fold_ymm_(text_ymm);
    __m256i previous_bytes_ymm = sz_utf8_uncased_haswell_previous_bytes_(text_ymm);
    __m256i is_after_ce_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xCE));
    __m256i is_after_cf_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xCF));
    __m256i is_after_c2_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xC2));

    // Second-byte classes after a CE lead
    __m256i is_basic1_ymm = _mm256_and_si256(is_after_ce_ymm, //
                                             sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0x91, 0x0F));
    __m256i is_basic2_ymm = _mm256_and_si256(is_after_ce_ymm, //
                                             sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0xA0, 0x0A));
    __m256i is_86_ymm = _mm256_and_si256(is_after_ce_ymm, //
                                         _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x86)));
    __m256i is_88_8a_ymm = _mm256_and_si256(is_after_ce_ymm, //
                                            sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0x88, 0x03));
    __m256i is_8c_ymm = _mm256_and_si256(is_after_ce_ymm, //
                                         _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x8C)));
    __m256i is_8e_8f_ymm = _mm256_and_si256(is_after_ce_ymm, //
                                            sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0x8E, 0x02));
    __m256i is_dialytika_ymm = _mm256_and_si256(is_after_ce_ymm, //
                                                sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0xAA, 0x02));

    // Final sigma 'ς' (CF 82) and the micro sign's second byte (C2 B5)
    __m256i is_final_sigma_ymm = _mm256_and_si256(is_after_cf_ymm, //
                                                  _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x82)));
    __m256i is_micro_second_ymm = _mm256_and_si256(is_after_c2_ymm, //
                                                   _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xB5)));

    // Propagate the lead rewrites back from the second-byte flags: CE → CF is +1 for four
    // classes; the micro sign's C2 → CE is +0x0C and stays separate
    __m256i promote_lead_ymm = sz_utf8_uncased_haswell_next_bytes_(
        _mm256_or_si256(_mm256_or_si256(is_basic2_ymm, is_dialytika_ymm), _mm256_or_si256(is_8c_ymm, is_8e_8f_ymm)));
    __m256i micro_lead_ymm = sz_utf8_uncased_haswell_next_bytes_(is_micro_second_ymm);

    // Disjoint positions merge into ONE offset vector and a single add
    __m256i is_minus_20_ymm = _mm256_or_si256(is_basic2_ymm, is_dialytika_ymm);
    __m256i is_plus_one_ymm = _mm256_or_si256(is_final_sigma_ymm, promote_lead_ymm);
    __m256i offsets_ymm = _mm256_and_si256(is_basic1_ymm, _mm256_set1_epi8(0x20));
    offsets_ymm = _mm256_or_si256(offsets_ymm, _mm256_and_si256(is_minus_20_ymm, _mm256_set1_epi8((char)0xE0)));
    offsets_ymm = _mm256_or_si256(offsets_ymm, _mm256_and_si256(is_86_ymm, _mm256_set1_epi8(0x26)));
    offsets_ymm = _mm256_or_si256(offsets_ymm, _mm256_and_si256(is_88_8a_ymm, _mm256_set1_epi8(0x25)));
    offsets_ymm = _mm256_or_si256(offsets_ymm, _mm256_and_si256(is_8e_8f_ymm, _mm256_set1_epi8((char)0xFF)));
    offsets_ymm = _mm256_or_si256(offsets_ymm, _mm256_and_si256(is_plus_one_ymm, _mm256_set1_epi8(0x01)));
    offsets_ymm = _mm256_or_si256(offsets_ymm, _mm256_and_si256(is_micro_second_ymm, _mm256_set1_epi8(0x07)));
    offsets_ymm = _mm256_or_si256(offsets_ymm, _mm256_and_si256(micro_lead_ymm, _mm256_set1_epi8(0x0C)));
    return _mm256_add_epi8(result_ymm, offsets_ymm);
}

/**
 *  @brief Alarm function for Greek danger zone detection.
 *
 *  Detects positions where danger characters occur that require special handling:
 *  - CE 90 / CE B0: 'ΐ', 'ΰ' expand to 3 codepoints when folded
 *  - CF 90, 91, 95, 96: Greek symbols 'ϐ', 'ϑ', 'ϕ', 'ϖ' fold to basic letters
 *  - CF B0, B1, B4, B5: 'ϰ', 'ϱ', 'ϴ', 'ϵ' fold to basic letters ('ϴ' → CE B8 'θ')
 *  - E2 84: Ohm sign 'Ω' (U+2126) prefix (3 bytes → 2 bytes)
 *  - E1 (blanket): polytonic Greek Extended, with single-, double-, and triple-expanding folds
 *  - CD (blanket): archaic letters and combining marks adjacent to the Greek block
 *
 *  Modern Greek text is pure CE/CF sequences, so the blanket E1/CD lead alarms almost
 *  never fire - but when they do, the driver's step−2 retreat keeps a 3-byte danger
 *  sequence straddling the chunk edge fully visible in the next chunk.
 */
SZ_INTERNAL sz_u32_t sz_utf8_uncased_find_haswell_greek_alarm_ymm_(__m256i text_ymm, sz_u32_t load_mask) {
    sz_unused_(load_mask); // Present for the shared `sz_utf8_uncased_alarm_ymm_t_` signature

    // Lead bytes (5 CMPEQ + movemask)
    sz_u32_t is_ce_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xCE)));
    sz_u32_t is_cf_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xCF)));
    sz_u32_t is_e2_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xE2)));
    sz_u32_t is_e1_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xE1)));
    sz_u32_t is_cd_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xCD)));

    // Second bytes (9 CMPEQ + movemask)
    sz_u32_t is_90_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x90)));
    sz_u32_t is_b0_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xB0)));
    sz_u32_t is_9x_mask = is_90_mask |
                          (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x91))) |
                          (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x95))) |
                          (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x96)));
    sz_u32_t is_bx_mask = is_b0_mask |
                          (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xB1))) |
                          (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xB4))) |
                          (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xB5)));
    sz_u32_t is_84_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x84)));

    // Danger mask construction
    return ((is_ce_mask << 1) & (is_90_mask | is_b0_mask)) | // 'ΐ', 'ΰ' (CE 90 / CE B0)
           ((is_cf_mask << 1) & (is_9x_mask | is_bx_mask)) | // Greek symbols (CF 9x / CF Bx)
           ((is_e2_mask << 1) & is_84_mask) |                // Ohm sign (E2 84 A6)
           is_e1_mask | is_cd_mask;                          // Blanket polytonic & archaic leads
}

/**
 *  @brief Greek uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_greek_k
 */
SZ_INTERNAL sz_cptr_t sz_utf8_uncased_find_haswell_greek_(    //
    sz_cptr_t haystack, sz_size_t haystack_length,                     //
    sz_cptr_t needle, sz_size_t needle_length,                         //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata, //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_find_haswell_scripted_( //
        sz_utf8_uncased_find_haswell_greek_fold_ymm_,
        sz_utf8_uncased_find_haswell_greek_alarm_ymm_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Greek Uncased Find

#pragma region Vietnamese Uncased Find

/**
 *  @brief Fold a YMM register using Vietnamese case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_vietnamese_k
 *
 *  Vietnamese letters spread across four Latin blocks, all folding in place:
 *  - C3 80-9E: Latin-1 Supplement uppercase → +0x20, except the caseless '×' (C3 97)
 *  - C4/C5: Latin Extended-A folds with +1 keyed on continuation parity - the codepoint's
 *    low bit equals the byte's low bit. Most of the block folds EVEN seconds; the
 *    sub-ranges C4 B9-BE ('Ĺ'-'ľ') and C5 80-88 ('ŀ'-'ň') invert and fold ODD seconds
 *  - C6 A0 / C6 AF: 'Ơ' → 'ơ' and 'Ư' → 'ư' (+1)
 *  - E1 B8-BB: Latin Extended Additional folds EVEN third bytes with +1, except the
 *    expanding E1 BA 96-9F block ('ẖ'-'ẟ'), which the alarm routes to the serial scanner
 *
 *  The third-byte rule needs the byte TWO lanes back, so a second `previous_bytes` pass
 *  materializes it; all rule masks flag disjoint positions and merge into one offset add.
 */
SZ_INTERNAL __m256i sz_utf8_uncased_find_haswell_vietnamese_fold_ymm_(__m256i text_ymm) {
    __m256i result_ymm = sz_utf8_uncased_find_haswell_ascii_fold_ymm_(text_ymm);
    __m256i previous_bytes_ymm = sz_utf8_uncased_haswell_previous_bytes_(text_ymm);
    __m256i previous2_bytes_ymm = sz_utf8_uncased_haswell_previous_bytes_(previous_bytes_ymm);
    __m256i is_after_c3_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xC3));
    __m256i is_after_c4_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xC4));
    __m256i is_after_c5_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xC5));
    __m256i is_after_c6_ymm = _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xC6));

    // 1. Latin-1 Supplement: C3 80-9E → +0x20, except '×' (C3 97)
    __m256i is_c3_target_ymm = _mm256_and_si256(
        is_after_c3_ymm, _mm256_andnot_si256(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x97)),
                                             sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0x80, 0x1F)));

    // 2. Latin Extended-A: +1 on EVEN seconds, except the inverted sub-ranges C4 B9-BE and
    //    C5 00-88 (the unsigned `≤ 88` bound mirrors the Ice Lake reference) which fold ODD
    __m256i is_odd_ymm = _mm256_cmpeq_epi8(_mm256_and_si256(text_ymm, _mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x01));
    __m256i is_inverted_ymm = _mm256_or_si256(
        _mm256_and_si256(is_after_c4_ymm, sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0xB9, 0x06)),
        _mm256_and_si256(is_after_c5_ymm, sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0x00, 0x89)));
    __m256i is_extended_even_ymm = _mm256_andnot_si256(
        is_odd_ymm, _mm256_andnot_si256(is_inverted_ymm, _mm256_or_si256(is_after_c4_ymm, is_after_c5_ymm)));
    __m256i fold_extended_ymm = _mm256_or_si256(is_extended_even_ymm, _mm256_and_si256(is_inverted_ymm, is_odd_ymm));

    // 3. Latin Extended-B: 'Ơ' (C6 A0) and 'Ư' (C6 AF) → +1
    __m256i is_c6_target_ymm = _mm256_and_si256(
        is_after_c6_ymm, _mm256_or_si256(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xA0)),
                                         _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xAF))));

    // 4. Latin Extended Additional: EVEN third bytes after an E1 B8-BB pair → +1,
    //    except the expanding E1 BA 96-9F block
    __m256i is_after_e1_pair_ymm = _mm256_and_si256(
        _mm256_cmpeq_epi8(previous2_bytes_ymm, _mm256_set1_epi8((char)0xE1)),
        sz_utf8_uncased_haswell_in_byte_range_(previous_bytes_ymm, 0xB8, 0x04));
    __m256i is_excluded_third_ymm = _mm256_and_si256(
        _mm256_cmpeq_epi8(previous_bytes_ymm, _mm256_set1_epi8((char)0xBA)),
        sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0x96, 0x0A));
    __m256i fold_e1_ymm = _mm256_andnot_si256(is_odd_ymm,
                                              _mm256_andnot_si256(is_excluded_third_ymm, is_after_e1_pair_ymm));

    // Disjoint positions merge into ONE offset vector and a single add
    __m256i is_plus_one_ymm = _mm256_or_si256(fold_extended_ymm, _mm256_or_si256(is_c6_target_ymm, fold_e1_ymm));
    __m256i offsets_ymm = _mm256_or_si256(_mm256_and_si256(is_c3_target_ymm, _mm256_set1_epi8(0x20)),
                                          _mm256_and_si256(is_plus_one_ymm, _mm256_set1_epi8(0x01)));
    return _mm256_add_epi8(result_ymm, offsets_ymm);
}

/**
 *  @brief Alarm function for Vietnamese danger zone detection.
 *
 *  Detects positions where danger characters occur that require special handling:
 *  - E1 BA 96-9F: 'ẖ'-'ẟ' expand to ASCII-led sequences when folded ('ẞ' → "ss"); the
 *    third-byte qualification matters because the rest of E1 BA covers Vietnamese letters
 *    that fold in place - flagging them blanket-style would send dense Vietnamese text
 *    into the serial danger-zone scanner on every chunk
 *  - C3 9F: 'ß' (U+00DF) → "ss" (1 rune → 2 runes)
 *  - C5 BF: 'ſ' (U+017F) → 's' (2 bytes → 1 byte)
 *  - EF AC 80-86: Latin ligatures 'ﬀ'-'ﬆ' → ASCII pairs/triples
 *  - E2 84 AA: 'K' Kelvin sign (3 bytes → 1 byte)
 *
 *  Ice Lake qualifies the third-byte range compare with the load mask; here the driver's
 *  padded loads already zero every absent byte, and zero never lands inside [96, 9F], so
 *  the unqualified compare is exactly as safe-negative on tail chunks. Unlike the other
 *  alarms, the result is shifted back to the SEQUENCE-START positions, mirroring the
 *  Ice Lake reference bit-for-bit.
 */
SZ_INTERNAL sz_u32_t sz_utf8_uncased_find_haswell_vietnamese_alarm_ymm_(__m256i text_ymm, sz_u32_t load_mask) {
    sz_unused_(load_mask); // Padded loads zero absent bytes, so range compares are safe-negative

    // Lead bytes (5 CMPEQ + movemask)
    sz_u32_t is_e1_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xE1)));
    sz_u32_t is_c3_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xC3)));
    sz_u32_t is_c5_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xC5)));
    sz_u32_t is_ef_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xEF)));
    sz_u32_t is_e2_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xE2)));

    // Vietnamese text is dense in E1 (and the safe E1 BA letters), but plain Latin chunks
    // skip all the second-byte work behind this early exit
    if (!(is_e1_mask | is_c3_mask | is_c5_mask | is_ef_mask | is_e2_mask)) return 0;

    // E1 BA pairs refine on the expanding 96-9F third byte
    sz_u32_t is_ba_second_mask = (is_e1_mask << 1) & (sz_u32_t)_mm256_movemask_epi8(
                                                         _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xBA)));
    sz_u32_t is_bad_third_mask = (is_ba_second_mask << 1) &
                                 (sz_u32_t)_mm256_movemask_epi8(
                                     sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0x96, 0x0A));

    // Two-byte pair alarms (4 CMPEQ + movemask)
    sz_u32_t sharp_s_mask = (is_c3_mask << 1) &
                            (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x9F)));
    sz_u32_t long_s_mask = (is_c5_mask << 1) &
                           (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xBF)));
    sz_u32_t ligature_mask = (is_ef_mask << 1) &
                             (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xAC)));
    sz_u32_t kelvin_mask = (is_e2_mask << 1) &
                           (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x84)));

    // Shift back to sequence-start positions
    return (is_bad_third_mask >> 2) | ((sharp_s_mask | long_s_mask | ligature_mask | kelvin_mask) >> 1);
}

/**
 *  @brief Vietnamese uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_vietnamese_k
 */
SZ_INTERNAL sz_cptr_t sz_utf8_uncased_find_haswell_vietnamese_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                       //
    sz_cptr_t needle, sz_size_t needle_length,                           //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,   //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_find_haswell_scripted_( //
        sz_utf8_uncased_find_haswell_vietnamese_fold_ymm_,
        sz_utf8_uncased_find_haswell_vietnamese_alarm_ymm_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Vietnamese Uncased Find

#pragma region Georgian Uncased Find

/**
 *  @brief Alarm function for Georgian danger zone detection.
 *
 *  Georgian Mkhedruli (E1 83 xx and the tail of E1 82) is caseless, so the haystack-side
 *  hazards are the OTHER Georgian scripts, which all fold across blocks:
 *  - E1 B2 xx: Mtavruli uppercase, folds to Mkhedruli
 *  - E1 82 A0-E5: Asomtavruli historical uppercase, folds to Nuskhuri
 *  - E2 B4 xx: Nuskhuri, target of Asomtavruli folds
 *
 *  Modern Georgian is E1 83 leads, so neither second-byte pair matches and the kernel
 *  almost never alarms. The Asomtavruli third-byte range compare is unqualified by the
 *  load mask: the driver's padded loads zero absent bytes, and zero never lands inside
 *  [A0, E5], so tail chunks stay safe-negative. The result is shifted back to the
 *  SEQUENCE-START positions, mirroring the Ice Lake reference bit-for-bit.
 */
SZ_INTERNAL sz_u32_t sz_utf8_uncased_find_haswell_georgian_alarm_ymm_(__m256i text_ymm, sz_u32_t load_mask) {
    sz_unused_(load_mask); // Padded loads zero absent bytes, so range compares are safe-negative

    // Lead bytes (2 CMPEQ + movemask)
    sz_u32_t is_e1_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xE1)));
    sz_u32_t is_e2_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xE2)));

    // Second bytes (3 CMPEQ + movemask)
    sz_u32_t is_b2_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xB2)));
    sz_u32_t is_82_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x82)));
    sz_u32_t is_b4_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xB4)));

    // E1 B2 = Mtavruli; E1 82 refines on the A0-E5 third byte for Asomtavruli; E2 B4 = Nuskhuri
    sz_u32_t mtavruli_mask = (is_e1_mask << 1) & is_b2_mask;
    sz_u32_t asomtavruli_mask = (((is_e1_mask << 1) & is_82_mask) << 1) &
                                (sz_u32_t)_mm256_movemask_epi8(sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0xA0, 0x46));
    sz_u32_t nuskhuri_mask = (is_e2_mask << 1) & is_b4_mask;

    // Shift back to sequence-start positions
    return (mtavruli_mask >> 1) | (asomtavruli_mask >> 2) | (nuskhuri_mask >> 1);
}

/**
 *  @brief Georgian uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_georgian_k
 *
 *  The fastest non-ASCII kernel: Mkhedruli is caseless, so the fold callback is just the
 *  ASCII fold for mixed Latin text and the alarm only watches for the historical scripts.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_uncased_find_haswell_georgian_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                     //
    sz_cptr_t needle, sz_size_t needle_length,                         //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata, //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_find_haswell_scripted_( //
        sz_utf8_uncased_find_haswell_ascii_fold_ymm_,
        sz_utf8_uncased_find_haswell_georgian_alarm_ymm_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Georgian Uncased Find

SZ_PUBLIC sz_cptr_t sz_utf8_uncased_find_haswell( //
    sz_cptr_t haystack, sz_size_t haystack_length,         //
    sz_cptr_t needle, sz_size_t needle_length,             //
    sz_utf8_uncased_needle_metadata_t *needle_metadata, sz_size_t *matched_length) {

    // Handle the obvious edge cases first
    if (needle_length == 0) {
        *matched_length = 0;
        return haystack;
    }

    // If the needle is entirely made of case-less characters - perform direct substring search
    int const is_unknown = needle_metadata->kernel_id == sz_utf8_uncased_rune_unknown_k;
    int const known_agnostic = needle_metadata->kernel_id == sz_utf8_uncased_rune_invariant_k;
    if (known_agnostic || (is_unknown && sz_utf8_uncased_violation_haswell(needle, needle_length) == SZ_NULL_CHAR)) {
        sz_cptr_t result = sz_find_haswell(haystack, haystack_length, needle, needle_length);
        *matched_length = result ? needle_length : 0;
        return result;
    }

    // Analyze needle to find the best safe window for each script
    if (is_unknown) {
        sz_utf8_uncased_needle_metadata_(needle, needle_length, needle_metadata);
        // If no SIMD-safe window found, fall back to serial immediately
        if (needle_metadata->kernel_id == sz_utf8_uncased_rune_fallback_serial_k)
            return sz_utf8_uncased_find_serial(haystack, haystack_length, needle, needle_length,
                                                        needle_metadata, matched_length);
    }

    // Dispatch to appropriate kernel
    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_ascii_invariant_k) {
        if (needle_metadata->folded_slice_length <= 3)
            return sz_utf8_uncased_find_haswell_ascii_3probe_( //
                haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
        else
            return sz_utf8_uncased_find_haswell_ascii_4probe_( //
                haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    }

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_western_europe_k)
        return sz_utf8_uncased_find_haswell_western_europe_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_central_europe_k)
        return sz_utf8_uncased_find_haswell_central_europe_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_cyrillic_k)
        return sz_utf8_uncased_find_haswell_cyrillic_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_greek_k)
        return sz_utf8_uncased_find_haswell_greek_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_armenian_k)
        return sz_utf8_uncased_find_haswell_armenian_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_vietnamese_k)
        return sz_utf8_uncased_find_haswell_vietnamese_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_georgian_k)
        return sz_utf8_uncased_find_haswell_georgian_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    // No suitable SIMD path found (needle has complex Unicode), fall back to serial
    needle_metadata->kernel_id = sz_utf8_uncased_rune_fallback_serial_k;
    return sz_utf8_uncased_find_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                                matched_length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_uncased_violation_haswell(sz_cptr_t str, sz_size_t length) {
    sz_cptr_t text_cursor = str;

    // Single loop: advance by min(length, 29), check leads in the first `block_length` positions;
    // the 3-byte slack keeps every checked lead's continuation bytes inside the same 32-byte load
    while (length) {
        sz_size_t block_length = sz_min_of_two(length, 29);
        sz_u32_t lead_mask = sz_utf8_uncased_haswell_mask_until_(block_length);
        __m256i text_ymm = length >= 32 ? _mm256_lddqu_si256((__m256i const *)text_cursor)
                                        : sz_utf8_uncased_haswell_load_padded_ymm_(text_cursor, length);

        // 1. ASCII letter check (zeros beyond the string are fine - not letters)
        sz_u32_t is_upper_mask = (sz_u32_t)_mm256_movemask_epi8(sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 'A', 26));
        sz_u32_t is_lower_mask = (sz_u32_t)_mm256_movemask_epi8(sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 'a', 26));
        if (is_upper_mask | is_lower_mask) return sz_utf8_uncased_violation_serial(str, length);

        // 2. Check for non-ASCII in lead positions
        sz_u32_t is_non_ascii_mask = (sz_u32_t)_mm256_movemask_epi8(text_ymm) & lead_mask;
        if (is_non_ascii_mask) {
            // 3. Identify UTF-8 lead bytes
            __m256i const xe0_ymm = _mm256_set1_epi8((char)0xE0);
            __m256i const xf0_ymm = _mm256_set1_epi8((char)0xF0);
            sz_u32_t is_two_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_and_si256(text_ymm, xe0_ymm),
                                                                                    _mm256_set1_epi8((char)0xC0))) &
                                   lead_mask;
            sz_u32_t is_three_mask = (sz_u32_t)_mm256_movemask_epi8(
                                         _mm256_cmpeq_epi8(_mm256_and_si256(text_ymm, xf0_ymm), xe0_ymm)) &
                                     lead_mask;
            sz_u32_t is_four_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(
                                        _mm256_and_si256(text_ymm, _mm256_set1_epi8((char)0xF8)), xf0_ymm)) &
                                    lead_mask;

            // 4. Check 4-byte bicameral scripts (SMP): F0 with second byte 90/91/96/9D/9E
            if (is_four_mask) {
                sz_u32_t after_f0_mask = is_four_mask << 1;
                sz_u32_t is_90_mask = (sz_u32_t)_mm256_movemask_epi8(
                    _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x90)));
                sz_u32_t is_91_mask = (sz_u32_t)_mm256_movemask_epi8(
                    _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x91)));
                sz_u32_t is_96_mask = (sz_u32_t)_mm256_movemask_epi8(
                    _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x96)));
                sz_u32_t is_9d_mask = (sz_u32_t)_mm256_movemask_epi8(
                    _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x9D)));
                sz_u32_t is_9e_mask = (sz_u32_t)_mm256_movemask_epi8(
                    _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0x9E)));
                if (after_f0_mask & (is_90_mask | is_91_mask | is_96_mask | is_9d_mask | is_9e_mask)) return sz_utf8_uncased_violation_serial(str, length);
            }

            // 5. Check 2-byte bicameral leads: C3-D6
            // C3-CF: Latin Extended (umlauts, accents, Eszett)
            // D0-D1: Cyrillic, D4-D6: Armenian (D6 needed for small letters U+0580+)
            if (is_two_mask) {
                sz_u32_t is_bicameral_mask = (sz_u32_t)_mm256_movemask_epi8(
                    sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0xC3, 0x14));

                // Special case: C2 B5 = U+00B5 MICRO SIGN folds to Greek mu (U+03BC)
                sz_u32_t is_c2_mask = (sz_u32_t)_mm256_movemask_epi8(
                                          _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xC2))) &
                                      is_two_mask;
                if (is_c2_mask) {
                    sz_u32_t is_b5_mask = (sz_u32_t)_mm256_movemask_epi8(
                        _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xB5)));
                    if ((is_c2_mask << 1) & is_b5_mask) return sz_utf8_uncased_violation_serial(str, length);
                }

                // Note: CA 80-BF includes both IPA Extensions (U+0280-02AF) and Spacing Modifier Letters
                // (U+02B0-02BF). Spacing Modifier Letters CAN appear in case fold expansions:
                // e.g., ẚ (U+1E9A) folds to [a, ʾ] where ʾ = U+02BE is a Spacing Modifier Letter.
                // So we must NOT exclude this range from the bicameral check.
                if (is_bicameral_mask & is_two_mask) return sz_utf8_uncased_violation_serial(str, length);
            }

            // 6. Check 3-byte bicameral sequences
            if (is_three_mask) {
                // E1: Georgian, Greek Extended, Latin Extended Additional
                sz_u32_t is_e1_mask = (sz_u32_t)_mm256_movemask_epi8(
                    _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xE1)));
                if (is_e1_mask & is_three_mask) return sz_utf8_uncased_violation_serial(str, length);

                // EF: Fullwidth Latin
                sz_u32_t is_ef_mask = (sz_u32_t)_mm256_movemask_epi8(
                    _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xEF)));
                if (is_ef_mask & is_three_mask) return sz_utf8_uncased_violation_serial(str, length);

                // E2: Safe only for second byte 80-83
                sz_u32_t is_e2_mask = (sz_u32_t)_mm256_movemask_epi8(
                                          _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xE2))) &
                                      is_three_mask;
                if (is_e2_mask) {
                    sz_u32_t e2_second_safe_mask = (sz_u32_t)_mm256_movemask_epi8(
                        sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0x80, 0x04));
                    if ((is_e2_mask << 1) & ~e2_second_safe_mask) return sz_utf8_uncased_violation_serial(str, length);
                }

                // EA: Bicameral second bytes 99-9F, AC-AE
                sz_u32_t is_ea_mask = (sz_u32_t)_mm256_movemask_epi8(
                                          _mm256_cmpeq_epi8(text_ymm, _mm256_set1_epi8((char)0xEA))) &
                                      is_three_mask;
                if (is_ea_mask) {
                    sz_u32_t is_99_range_mask = (sz_u32_t)_mm256_movemask_epi8(
                        sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0x99, 0x07));
                    sz_u32_t is_ac_range_mask = (sz_u32_t)_mm256_movemask_epi8(
                        sz_utf8_uncased_haswell_in_byte_range_(text_ymm, 0xAC, 0x03));
                    if ((is_ea_mask << 1) & (is_99_range_mask | is_ac_range_mask)) return sz_utf8_uncased_violation_serial(str, length);
                }
            }
        }

        text_cursor += block_length;
        length -= block_length;
    }

    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_ordering_t sz_utf8_uncased_order_haswell(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                               sz_size_t b_length) {
    return sz_utf8_uncased_order_serial(a, a_length, b, b_length);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_UNCASED_HASWELL_H_
