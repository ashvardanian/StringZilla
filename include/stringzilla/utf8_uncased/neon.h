/**
 *  @brief NEON (Arm) uncased UTF-8 search, comparison & invariance backend.
 *  @file include/stringzilla/utf8_uncased/neon.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased.h
 *
 *  Ports the validated AVX2 architecture to 32-byte NEON chunks held as `uint8x16x2_t` (two
 *  16-byte registers, mirroring a 32-byte YMM). The needle classification, match verification,
 *  and danger-zone scanning stay ISA-independent in `serial.h`; per-script `fold`/`alarm`
 *  helpers become `uint8x16x2_t` functions; and the shared force-inlined `scripted_` driver walks
 *  the haystack in 32-byte steps. AVX2's two compare domains map directly: byte-mask vectors (for
 *  folds, which feed masked adds) and a 32-bit movemask integer (for alarms and probe filters,
 *  where scalar shifts replace `VPMOVMSKB`-shift algebra bit-for-bit). The 256-bit register has no
 *  NEON equivalent, so the fold callbacks splice the internal 16-byte boundary themselves with
 *  `vextq_u8`: the predecessor of register 1's lane 0 is register 0's lane 15, exactly as AVX2's
 *  `VPERM2I128` carries the low lane across the 128-bit edge.
 */
#ifndef STRINGZILLA_UTF8_UNCASED_NEON_H_
#define STRINGZILLA_UTF8_UNCASED_NEON_H_

#include "stringzilla/find/neon.h" // `sz_find_neon`
#include "stringzilla/utf8_uncased/serial.h"
#include "stringzilla/utf8_uncased_fold/neon.h" // `sz_utf8_fold_neon_ascii_`

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

#pragma region Shared NEON Helpers

/**
 *  @brief Collapses two 16-byte comparison results into a 32-bit movemask, one bit per byte.
 *      Lane `i` of the low register sets bit `i`, lane `j` of the high register sets bit `16 + j`.
 *
 *  NEON has no `VPMOVMSKB`, so each register is AND-ed with the per-lane bit weights
 *  {1, 2, 4, ..., 128} repeated across the two 8-byte halves, then `vaddv_u8` horizontally sums
 *  each half into its 8 packed bits. Unlike the `vshrn` nibble trick (4 bits per byte), this
 *  yields exactly ONE bit per byte, so the probe filter's scalar shifts by runtime probe offsets
 *  port from the AVX2 driver unchanged.
 */
SZ_HELPER_INLINE sz_u32_t sz_utf8_uncased_neon_movemask_u8x16x2_(uint8x16_t low_cmp_u8x16, uint8x16_t high_cmp_u8x16) {
    // MSVC's ARM64 `uint8x16_t` is a `__n128` struct that rejects scalar brace-init, so load from `.rodata`.
    static sz_u8_t const bit_weights[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint8x16_t const bit_weights_u8x16 = vld1q_u8(bit_weights);
    uint8x16_t low_bits_u8x16 = vandq_u8(low_cmp_u8x16, bit_weights_u8x16);
    uint8x16_t high_bits_u8x16 = vandq_u8(high_cmp_u8x16, bit_weights_u8x16);
    sz_u32_t low_mask = (sz_u32_t)vaddv_u8(vget_low_u8(low_bits_u8x16)) |
                        ((sz_u32_t)vaddv_u8(vget_high_u8(low_bits_u8x16)) << 8);
    sz_u32_t high_mask = (sz_u32_t)vaddv_u8(vget_low_u8(high_bits_u8x16)) |
                         ((sz_u32_t)vaddv_u8(vget_high_u8(high_bits_u8x16)) << 8);
    return low_mask | (high_mask << 16);
}

/**
 *  @brief Detects bytes in the unsigned range [range_start, range_start + range_length).
 *      NEON has unsigned byte compares, so `(x − start) < length` is one wrap-around subtraction
 *      plus one `VCLT`: bytes below `start` wrap above `length` and drop out.
 */
SZ_HELPER_INLINE uint8x16_t sz_utf8_uncased_neon_in_byte_range_u8x16_(uint8x16_t values_u8x16, sz_u8_t range_start,
                                                                      sz_u8_t range_length) {
    return vcltq_u8(vsubq_u8(values_u8x16, vdupq_n_u8(range_start)), vdupq_n_u8(range_length));
}

/**
 *  @brief Shifts the 32 chunk bytes right by one lane, so lane `i` holds byte `i − 1`. Lane 0 of
 *      the low register receives zero; lane 0 of the high register receives the low register's
 *      lane 15 - the carry across the internal 16-byte boundary. Vector-domain equivalent of
 *      Ice Lake's `k-mask << 1` and AVX2's `VPERM2I128` + `VPALIGNR`.
 *
 *      The zero fill at lane 0 is correct because a real match starts on a character boundary
 *      where the lead byte's fold never needs its predecessor; the candidate-window re-fold uses
 *      the same zero-predecessor convention, so both agree at every real match position.
 */
SZ_HELPER_INLINE uint8x16x2_t sz_utf8_uncased_neon_previous_bytes_u8x16x2_(uint8x16x2_t source_u8x16x2) {
    uint8x16_t const zero_u8x16 = vdupq_n_u8(0x00);
    uint8x16x2_t result_u8x16x2;
    result_u8x16x2.val[0] = vextq_u8(zero_u8x16, source_u8x16x2.val[0], 15);
    result_u8x16x2.val[1] = vextq_u8(source_u8x16x2.val[0], source_u8x16x2.val[1], 15);
    return result_u8x16x2;
}

/**
 *  @brief Shifts the 32 chunk bytes left by one lane, so lane `i` holds byte `i + 1`. Lane 15 of
 *      the high register receives zero; lane 15 of the low register receives the high register's
 *      lane 0. Vector-domain equivalent of Ice Lake's `k-mask >> 1`.
 */
SZ_HELPER_INLINE uint8x16x2_t sz_utf8_uncased_neon_next_bytes_u8x16x2_(uint8x16x2_t source_u8x16x2) {
    uint8x16_t const zero_u8x16 = vdupq_n_u8(0x00);
    uint8x16x2_t result_u8x16x2;
    result_u8x16x2.val[0] = vextq_u8(source_u8x16x2.val[0], source_u8x16x2.val[1], 1);
    result_u8x16x2.val[1] = vextq_u8(source_u8x16x2.val[1], zero_u8x16, 1);
    return result_u8x16x2;
}

/** @brief First N bits set, defined for `n == 32` (where `(1u << n) − 1` is undefined). */
SZ_HELPER_INLINE sz_u32_t sz_utf8_uncased_neon_mask_until_(sz_size_t n) {
    return n >= 32 ? 0xFFFFFFFFu : ((sz_u32_t)1 << n) - 1;
}

/**
 *  @brief Loads up to 32 bytes through a zeroed stack buffer, never touching memory past
 *      `source + length`. The zero padding mirrors Ice Lake's `maskz` loads: zero bytes match no
 *      probe inside a valid window and trip no alarm, so tail chunks reuse the main-loop logic
 *      unchanged instead of branching into a separate epilogue.
 */
SZ_HELPER_AUTO uint8x16x2_t sz_utf8_uncased_neon_load_padded_u8x16x2_(sz_cptr_t source, sz_size_t length) {
    sz_u8_t buffer[32] = {0};
    for (sz_size_t byte_index = 0; byte_index < length; ++byte_index) buffer[byte_index] = (sz_u8_t)source[byte_index];
    return vld1q_u8_x2(buffer);
}

/**
 *  @brief Loads up to 16 bytes for candidate-window verification without over-reading the
 *      haystack: the fast full load is taken whenever 16 bytes remain, and only the last few
 *      candidates near the haystack end pay for the zero-padded stack copy.
 */
SZ_HELPER_AUTO uint8x16_t sz_utf8_uncased_neon_load_window_u8x16_(sz_cptr_t source, sz_size_t available) {
    if (available >= 16) return vld1q_u8((sz_u8_t const *)source);
    sz_u8_t buffer[16] = {0};
    for (sz_size_t byte_index = 0; byte_index < available; ++byte_index)
        buffer[byte_index] = (sz_u8_t)source[byte_index];
    return vld1q_u8(buffer);
}

#pragma endregion // Shared NEON Helpers

#pragma region ASCII Uncased Find

/**
 *  @brief Fold a 32-byte chunk using ASCII case folding rules.
 *  @sa sz_utf8_uncased_rune_ascii_invariant_k
 */
SZ_HELPER_AUTO uint8x16x2_t sz_utf8_uncased_search_neon_ascii_fold_u8x16x2_(uint8x16x2_t text_u8x16x2) {
    uint8x16x2_t result_u8x16x2;
    // Only fold bytes in range A-Z; the masked add stays branch-free across both registers
    result_u8x16x2.val[0] = sz_utf8_fold_neon_ascii_(text_u8x16x2.val[0]);
    result_u8x16x2.val[1] = sz_utf8_fold_neon_ascii_(text_u8x16x2.val[1]);
    return result_u8x16x2;
}

/**
 *  @brief 3-probe ASCII uncased search over 32-byte chunks.
 *
 *  For needles with folded_slice_length ≤ 3, probes at positions 0, mid, last cover ALL bytes
 *  of the window, so no candidate-window verification is needed - candidates go straight to
 *  head/tail validation. The chunk is folded ONCE and the probe equality masks are shifted as
 *  32-bit movemask integers: with windows ≤ 16 bytes every chunk still exposes ≥ 17 valid start
 *  positions per iteration.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_neon_ascii_3probe_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                  //
    sz_cptr_t needle, sz_size_t needle_length,                      //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,       //
    sz_size_t *matched_length) {

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;

    // For ≤3 bytes: positions 0, mid, last cover ALL positions
    // 1-byte: 0=last, 2-byte: 0,1, 3-byte: 0,1,2
    sz_size_t const offset_second = folded_window_length / 2;
    sz_size_t const offset_last = folded_window_length - 1;

    uint8x16_t const probe_first_u8x16 = vdupq_n_u8(needle_metadata->folded_slice[0]);
    uint8x16_t const probe_second_u8x16 = vdupq_n_u8(needle_metadata->folded_slice[offset_second]);
    uint8x16_t const probe_last_u8x16 = vdupq_n_u8(needle_metadata->folded_slice[offset_last]);

    sz_cptr_t haystack_ptr = haystack;
    while (haystack_ptr < haystack_end) {
        sz_size_t const available = (sz_size_t)(haystack_end - haystack_ptr);
        if (available < folded_window_length) break;
        sz_size_t const chunk_size = available < 32 ? available : 32;
        sz_size_t const valid_starts = chunk_size - folded_window_length + 1;

        uint8x16x2_t text_u8x16x2 = available >= 32
                                        ? vld1q_u8_x2((sz_u8_t const *)haystack_ptr)
                                        : sz_utf8_uncased_neon_load_padded_u8x16x2_(haystack_ptr, chunk_size);
        uint8x16x2_t folded_u8x16x2 = sz_utf8_uncased_search_neon_ascii_fold_u8x16x2_(text_u8x16x2);

        sz_u32_t matches = sz_utf8_uncased_neon_movemask_u8x16x2_(vceqq_u8(folded_u8x16x2.val[0], probe_first_u8x16),
                                                                  vceqq_u8(folded_u8x16x2.val[1], probe_first_u8x16));
        matches &= sz_utf8_uncased_neon_movemask_u8x16x2_(vceqq_u8(folded_u8x16x2.val[0], probe_second_u8x16),
                                                          vceqq_u8(folded_u8x16x2.val[1], probe_second_u8x16)) >>
                   offset_second;
        matches &= sz_utf8_uncased_neon_movemask_u8x16x2_(vceqq_u8(folded_u8x16x2.val[0], probe_last_u8x16),
                                                          vceqq_u8(folded_u8x16x2.val[1], probe_last_u8x16)) >>
                   offset_last;
        matches &= sz_utf8_uncased_neon_mask_until_(valid_starts);

        for (; matches; matches &= matches - 1) {
            sz_size_t const candidate_offset = (sz_size_t)sz_u32_ctz(matches);
            sz_cptr_t const haystack_candidate_ptr = haystack_ptr + candidate_offset;

            // No window verification needed - probes cover all positions,
            // go directly to head/tail validation
            sz_cptr_t match = sz_utf8_uncased_verify_match_(                                               //
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

/** @brief Folds one 32-byte chunk of haystack text using script-specific rules. */
typedef uint8x16x2_t (*sz_utf8_uncased_fold_u8x16x2_t_)(uint8x16x2_t text_u8x16x2);

/**
 *  @brief Flags positions of "danger" characters that fold to a different byte width.
 *  @param load_mask Bitmask of the bytes actually loaded from the haystack, for tail-safe range checks.
 */
typedef sz_u32_t (*sz_utf8_uncased_alarm_u8x16x2_t_)(uint8x16x2_t text_u8x16x2, sz_u32_t load_mask);

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
 *  @param fold Script-specific 32-byte case-folding callback.
 *  @param alarm Script-specific danger detection callback, or NULL if the script has no
 *      danger characters: the danger branch disappears and the full step is used.
 */
SZ_HELPER_INLINE sz_cptr_t sz_utf8_uncased_search_neon_scripted_( //
    sz_utf8_uncased_fold_u8x16x2_t_ fold,                         //
    sz_utf8_uncased_alarm_u8x16x2_t_ alarm,                       //
    sz_cptr_t haystack, sz_size_t haystack_length,                //
    sz_cptr_t needle, sz_size_t needle_length,                    //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,     //
    sz_size_t *matched_length) {

    sz_assert_(needle_metadata && "needle_metadata must be provided");
    sz_assert_(needle_metadata->folded_slice_length > 0 && "folded window must be non-empty");

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_assert_(folded_window_length <= 16 && "expect folded needle part to fit in one register");

    // Pre-load folded window into one register; the byte-mask `window_keep` replicates Ice Lake's
    // `maskz` window load: bytes past the window are zeroed BEFORE folding, so a lead byte at the
    // window edge never borrows fold context from haystack bytes outside the window
    sz_u32_t const folded_window_mask = sz_utf8_uncased_neon_mask_until_(folded_window_length);
    uint8x16_t const needle_window_u8x16 = vld1q_u8((sz_u8_t const *)needle_metadata->folded_slice);
    static sz_u8_t const lane_indices[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint8x16_t const lane_indices_u8x16 = vld1q_u8(lane_indices);
    uint8x16_t const window_keep_u8x16 = vcltq_u8(lane_indices_u8x16, vdupq_n_u8((sz_u8_t)folded_window_length));

    // 4 probe positions
    sz_size_t const offset_second = needle_metadata->probe_second;
    sz_size_t const offset_third = needle_metadata->probe_third;
    sz_size_t const offset_last = folded_window_length - 1;

    uint8x16_t const probe_first_u8x16 = vdupq_n_u8(needle_metadata->folded_slice[0]);
    uint8x16_t const probe_second_u8x16 = vdupq_n_u8(needle_metadata->folded_slice[offset_second]);
    uint8x16_t const probe_third_u8x16 = vdupq_n_u8(needle_metadata->folded_slice[offset_third]);
    uint8x16_t const probe_last_u8x16 = vdupq_n_u8(needle_metadata->folded_slice[offset_last]);

    // Pre-load the first folded rune for danger zone matching
    sz_rune_t needle_first_safe_folded_rune = 0;
    if (alarm) sz_rune_decode_unchecked((sz_cptr_t)(needle_metadata->folded_slice), &needle_first_safe_folded_rune);

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
        sz_u32_t const load_mask = sz_utf8_uncased_neon_mask_until_(chunk_size);
        sz_u32_t const valid_mask = sz_utf8_uncased_neon_mask_until_(valid_starts);

        uint8x16x2_t text_u8x16x2 = available >= 32
                                        ? vld1q_u8_x2((sz_u8_t const *)haystack_ptr)
                                        : sz_utf8_uncased_neon_load_padded_u8x16x2_(haystack_ptr, chunk_size);

        // Check for anomalies (characters that fold to different byte widths)
        if (alarm) {
            sz_u32_t danger_mask = alarm(text_u8x16x2, load_mask);
            if (danger_mask) {
                // The danger zone handler scans for the needle's first safe rune (at offset_in_unfolded).
                // The whole chunk is scanned, not just `valid_starts` positions: an expanding danger
                // character makes the haystack span SHORTER than the folded window, so a real match
                // can start within the window's length of the chunk end.
                sz_cptr_t match = sz_utf8_uncased_search_in_danger_zone_( //
                    haystack, haystack_length,                            //
                    needle, needle_length,                                //
                    haystack_ptr, chunk_size,                             // extended danger zone
                    needle_first_safe_folded_rune,                        // pivot point
                    needle_metadata->offset_in_unfolded,                  // its location in the needle
                    matched_length);
                if (match) return match;
                haystack_ptr += step;
                continue;
            }
        }

        // Fold once, then filter candidates with 4 probe positions: scalar shifts over the
        // 32-bit movemasks substitute Ice Lake's k-mask shifts bit-for-bit
        uint8x16x2_t folded_u8x16x2 = fold(text_u8x16x2);
        sz_u32_t matches = sz_utf8_uncased_neon_movemask_u8x16x2_(vceqq_u8(folded_u8x16x2.val[0], probe_first_u8x16),
                                                                  vceqq_u8(folded_u8x16x2.val[1], probe_first_u8x16));
        matches &= sz_utf8_uncased_neon_movemask_u8x16x2_(vceqq_u8(folded_u8x16x2.val[0], probe_second_u8x16),
                                                          vceqq_u8(folded_u8x16x2.val[1], probe_second_u8x16)) >>
                   offset_second;
        matches &= sz_utf8_uncased_neon_movemask_u8x16x2_(vceqq_u8(folded_u8x16x2.val[0], probe_third_u8x16),
                                                          vceqq_u8(folded_u8x16x2.val[1], probe_third_u8x16)) >>
                   offset_third;
        matches &= sz_utf8_uncased_neon_movemask_u8x16x2_(vceqq_u8(folded_u8x16x2.val[0], probe_last_u8x16),
                                                          vceqq_u8(folded_u8x16x2.val[1], probe_last_u8x16)) >>
                   offset_last;
        matches &= valid_mask;

        // Candidate Verification
        for (; matches; matches &= matches - 1) {
            sz_size_t const candidate_offset = (sz_size_t)sz_u32_ctz(matches);
            sz_cptr_t const haystack_candidate_ptr = haystack_ptr + candidate_offset;

            // Re-fold the candidate window: loading the ≤16-byte view into the low register with a
            // zeroed high register keeps the script fold's per-register semantics identical to the
            // main chunk fold (zero predecessor for byte 0, zero successor past the window)
            uint8x16_t window_u8x16 = sz_utf8_uncased_neon_load_window_u8x16_(
                haystack_candidate_ptr, (sz_size_t)(haystack_end - haystack_candidate_ptr));
            window_u8x16 = vandq_u8(window_u8x16, window_keep_u8x16);
            uint8x16x2_t window_chunk_u8x16x2;
            window_chunk_u8x16x2.val[0] = window_u8x16;
            window_chunk_u8x16x2.val[1] = vdupq_n_u8(0x00);
            uint8x16x2_t folded_window_u8x16x2 = fold(window_chunk_u8x16x2);
            sz_u32_t window_equal_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(
                vceqq_u8(folded_window_u8x16x2.val[0], needle_window_u8x16), vdupq_n_u8(0x00));
            if ((window_equal_mask & folded_window_mask) != folded_window_mask) continue;

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
        sz_cptr_t match = sz_utf8_uncased_search_in_danger_zone_(   //
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

/**
 *  @brief 4-probe ASCII uncased search: the shared scripted driver with the ASCII fold
 *      and no alarm - ASCII never changes byte width when folded, so the danger machinery
 *      compiles away entirely and the step covers every valid start position.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_neon_ascii_4probe_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                  //
    sz_cptr_t needle, sz_size_t needle_length,                      //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,       //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_search_neon_scripted_( //
        sz_utf8_uncased_search_neon_ascii_fold_u8x16x2_,
        (sz_utf8_uncased_alarm_u8x16x2_t_)SZ_NULL, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Scripted Uncased Find

#pragma region Western European Uncased Find

/**
 *  @brief Fold a 32-byte chunk using Western European case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_western_europe_k
 *
 *  Handles ASCII A-Z, the Latin-1 Supplement uppercase range 'À'-'Þ' (C3 80-9E → +0x20,
 *  excluding the caseless '×' C3 97), and 'ß' (U+00DF, C3 9F) → "ss" where BOTH bytes of the
 *  pair become 's' so the folded image matches the needle's "ss".
 */
SZ_HELPER_NOINLINE uint8x16x2_t sz_utf8_uncased_search_neon_western_europe_fold_u8x16x2_(uint8x16x2_t text_u8x16x2) {
    uint8x16x2_t result_u8x16x2 = sz_utf8_uncased_search_neon_ascii_fold_u8x16x2_(text_u8x16x2);
    uint8x16x2_t previous_bytes_u8x16x2 = sz_utf8_uncased_neon_previous_bytes_u8x16x2_(text_u8x16x2);

    for (sz_size_t register_index = 0; register_index != 2; ++register_index) {
        uint8x16_t text_u8x16 = text_u8x16x2.val[register_index];
        uint8x16_t previous_bytes_u8x16 = previous_bytes_u8x16x2.val[register_index];
        uint8x16_t result_u8x16 = result_u8x16x2.val[register_index];
        uint8x16_t is_after_c3_u8x16 = vceqq_u8(previous_bytes_u8x16, vdupq_n_u8(0xC3));

        // 1. Handle Eszett: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73): the second-byte
        //    flag propagates one lane back to also rewrite the C3 lead. The back-propagation must
        //    cross the internal boundary, so the per-register next-shift carries the high
        //    register's lane 0 flag onto the low register's lane 15.
        uint8x16_t is_eszett_second_u8x16 = vandq_u8(is_after_c3_u8x16, vceqq_u8(text_u8x16, vdupq_n_u8(0x9F)));
        uint8x16x2_t is_eszett_second_u8x16x2;
        is_eszett_second_u8x16x2.val[register_index] = is_eszett_second_u8x16;
        is_eszett_second_u8x16x2.val[register_index ^ 1] = vdupq_n_u8(0x00);
        uint8x16_t next_eszett_second_u8x16 =
            sz_utf8_uncased_neon_next_bytes_u8x16x2_(is_eszett_second_u8x16x2).val[register_index];
        uint8x16_t is_eszett_u8x16 = vorrq_u8(is_eszett_second_u8x16, next_eszett_second_u8x16);
        result_u8x16 = vbslq_u8(is_eszett_u8x16, vdupq_n_u8('s'), result_u8x16);

        // 2. Handle Latin-1 supplement uppercase letters (C3 80-9E) → add 0x20,
        //    excluding '×' (C3 97, no case variant) and 'ß' (C3 9F, already handled above)
        uint8x16_t is_97_u8x16 = vceqq_u8(text_u8x16, vdupq_n_u8(0x97));
        uint8x16_t is_latin1_upper_u8x16 = vandq_u8(
            is_after_c3_u8x16, vbicq_u8(sz_utf8_uncased_neon_in_byte_range_u8x16_(text_u8x16, 0x80, 0x1F),
                                        vorrq_u8(is_eszett_second_u8x16, is_97_u8x16)));
        result_u8x16 = vaddq_u8(result_u8x16, vandq_u8(is_latin1_upper_u8x16, vdupq_n_u8(0x20)));
        result_u8x16x2.val[register_index] = result_u8x16;
    }
    return result_u8x16x2;
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
SZ_HELPER_NOINLINE sz_u32_t sz_utf8_uncased_search_neon_western_europe_alarm_u8x16x2_(uint8x16x2_t text_u8x16x2,
                                                                                      sz_u32_t load_mask) {
    sz_unused_(load_mask); // Present for the shared `sz_utf8_uncased_alarm_u8x16x2_t_` signature

    // The driver only tests the danger mask for non-emptiness, so the whole alarm stays in the
    // byte-mask domain: every pattern is anchored at its SECOND byte, with the lead read from the
    // `previous` view and the third byte from the `next` view. AVX2's `movemask <</>> 1` predecessor
    // algebra becomes plain `vceqq` against those shifted views, and a single `vmaxvq_u8` over the
    // accumulated danger lanes replaces the dozen per-value horizontal reductions.
    uint8x16x2_t previous_u8x16x2 = sz_utf8_uncased_neon_previous_bytes_u8x16x2_(text_u8x16x2);
    uint8x16x2_t next_u8x16x2 = sz_utf8_uncased_neon_next_bytes_u8x16x2_(text_u8x16x2);

    uint8x16_t any_danger_u8x16 = vdupq_n_u8(0);
    for (sz_size_t register_index = 0; register_index != 2; ++register_index) {
        uint8x16_t text_u8x16 = text_u8x16x2.val[register_index];
        uint8x16_t previous_u8x16 = previous_u8x16x2.val[register_index];
        uint8x16_t next_u8x16 = next_u8x16x2.val[register_index];
        uint8x16_t is_after_c5_u8x16 = vceqq_u8(previous_u8x16, vdupq_n_u8(0xC5));

        uint8x16_t danger_u8x16 = vandq_u8( // Capital Sharp S & co (E1 BA 96-9E)
            vandq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0xBA)), vceqq_u8(previous_u8x16, vdupq_n_u8(0xE1))),
            sz_utf8_uncased_neon_in_byte_range_u8x16_(next_u8x16, 0x96, 0x09));
        danger_u8x16 = vorrq_u8( // Kelvin/Angstrom (E2 84 AA/AB)
            danger_u8x16,
            vandq_u8(vandq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0x84)), vceqq_u8(previous_u8x16, vdupq_n_u8(0xE2))),
                     vorrq_u8(vceqq_u8(next_u8x16, vdupq_n_u8(0xAA)), vceqq_u8(next_u8x16, vdupq_n_u8(0xAB)))));
        danger_u8x16 = vorrq_u8( // Ligatures (EF AC xx)
            danger_u8x16, vandq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0xAC)), vceqq_u8(previous_u8x16, vdupq_n_u8(0xEF))));
        danger_u8x16 = vorrq_u8( // Long S (C5 BF)
            danger_u8x16, vandq_u8(is_after_c5_u8x16, vceqq_u8(text_u8x16, vdupq_n_u8(0xBF))));
        danger_u8x16 = vorrq_u8( // 'Ÿ' (C5 B8) → 'ÿ' (C3 BF)
            danger_u8x16, vandq_u8(is_after_c5_u8x16, vceqq_u8(text_u8x16, vdupq_n_u8(0xB8))));
        danger_u8x16 = vorrq_u8( // Sharp S (C3 9F)
            danger_u8x16, vandq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0x9F)), vceqq_u8(previous_u8x16, vdupq_n_u8(0xC3))));
        any_danger_u8x16 = vorrq_u8(any_danger_u8x16, danger_u8x16);
    }
    return vmaxvq_u8(any_danger_u8x16);
}

/**
 *  @brief Western European uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_western_europe_k
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_neon_western_europe_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                    //
    sz_cptr_t needle, sz_size_t needle_length,                        //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,         //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_search_neon_scripted_( //
        sz_utf8_uncased_search_neon_western_europe_fold_u8x16x2_,
        sz_utf8_uncased_search_neon_western_europe_alarm_u8x16x2_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Western European Uncased Find

#pragma region Central European Uncased Find

/**
 *  @brief Fold a 32-byte chunk using Central European case-folding rules.
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
SZ_HELPER_NOINLINE uint8x16x2_t sz_utf8_uncased_search_neon_central_europe_fold_u8x16x2_(uint8x16x2_t text_u8x16x2) {
    uint8x16x2_t result_u8x16x2 = sz_utf8_uncased_search_neon_ascii_fold_u8x16x2_(text_u8x16x2);
    uint8x16x2_t previous_bytes_u8x16x2 = sz_utf8_uncased_neon_previous_bytes_u8x16x2_(text_u8x16x2);
    uint8x16x4_t const c4_deltas_lut_u8x16x4 = vld1q_u8_x4(sz_utf8_uncased_central_c4_deltas_lut_);
    uint8x16x4_t const c5_deltas_lut_u8x16x4 = vld1q_u8_x4(sz_utf8_uncased_central_c5_deltas_lut_);

    for (sz_size_t register_index = 0; register_index != 2; ++register_index) {
        uint8x16_t text_u8x16 = text_u8x16x2.val[register_index];
        uint8x16_t previous_bytes_u8x16 = previous_bytes_u8x16x2.val[register_index];
        uint8x16_t result_u8x16 = result_u8x16x2.val[register_index];
        uint8x16_t is_after_c3_u8x16 = vceqq_u8(previous_bytes_u8x16, vdupq_n_u8(0xC3));
        uint8x16_t is_after_c4_u8x16 = vceqq_u8(previous_bytes_u8x16, vdupq_n_u8(0xC4));
        uint8x16_t is_after_c5_u8x16 = vceqq_u8(previous_bytes_u8x16, vdupq_n_u8(0xC5));
        uint8x16_t is_continuation_u8x16 = vcltq_u8(vsubq_u8(text_u8x16, vdupq_n_u8(0x80)), vdupq_n_u8(0x40));

        // 1. Latin-1 Supplement: C3 80-9E → +0x20, except '×' (C3 97)
        uint8x16_t is_latin1_upper_u8x16 = vandq_u8(
            is_after_c3_u8x16, vbicq_u8(sz_utf8_uncased_neon_in_byte_range_u8x16_(text_u8x16, 0x80, 0x1F),
                                        vceqq_u8(text_u8x16, vdupq_n_u8(0x97))));
        result_u8x16 = vaddq_u8(result_u8x16, vandq_u8(is_latin1_upper_u8x16, vdupq_n_u8(0x20)));

        // 2. Latin Extended-A: one `vqtbl4q_u8` per lead family resolves the +1 parity that the
        //    range/odd-parity checks used to assemble. The `is_continuation` mask drops bytes
        //    outside [0x80, 0xBF] that `text & 0x3F` would otherwise alias onto a folding index.
        uint8x16_t delta_indices_u8x16 = vandq_u8(text_u8x16, vdupq_n_u8(0x3F));
        uint8x16_t fold_extended_u8x16 = vorrq_u8(
            vandq_u8(vqtbl4q_u8(c4_deltas_lut_u8x16x4, delta_indices_u8x16), is_after_c4_u8x16),
            vandq_u8(vqtbl4q_u8(c5_deltas_lut_u8x16x4, delta_indices_u8x16), is_after_c5_u8x16));
        fold_extended_u8x16 = vandq_u8(fold_extended_u8x16, is_continuation_u8x16);
        result_u8x16 = vaddq_u8(result_u8x16, fold_extended_u8x16);
        result_u8x16x2.val[register_index] = result_u8x16;
    }
    return result_u8x16x2;
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
SZ_HELPER_NOINLINE sz_u32_t sz_utf8_uncased_search_neon_central_europe_alarm_u8x16x2_(uint8x16x2_t text_u8x16x2,
                                                                                      sz_u32_t load_mask) {
    sz_unused_(load_mask); // Present for the shared `sz_utf8_uncased_alarm_u8x16x2_t_` signature

    // Byte-mask danger detection anchored at the second byte; the lead comes from the `previous`
    // view. All pairs are two-byte, so no `next` lookup is needed, and one `vmaxvq_u8` gates the
    // driver's danger branch in place of seven per-value movemasks.
    uint8x16x2_t previous_u8x16x2 = sz_utf8_uncased_neon_previous_bytes_u8x16x2_(text_u8x16x2);

    uint8x16_t any_danger_u8x16 = vdupq_n_u8(0);
    for (sz_size_t register_index = 0; register_index != 2; ++register_index) {
        uint8x16_t text_u8x16 = text_u8x16x2.val[register_index];
        uint8x16_t previous_u8x16 = previous_u8x16x2.val[register_index];
        uint8x16_t is_after_c4_u8x16 = vceqq_u8(previous_u8x16, vdupq_n_u8(0xC4));
        uint8x16_t is_after_c5_u8x16 = vceqq_u8(previous_u8x16, vdupq_n_u8(0xC5));

        uint8x16_t danger_u8x16 = vandq_u8( // Kelvin (E2 84 AA)
            vceqq_u8(text_u8x16, vdupq_n_u8(0x84)), vceqq_u8(previous_u8x16, vdupq_n_u8(0xE2)));
        danger_u8x16 = vorrq_u8( // Sharp S (C3 9F)
            danger_u8x16, vandq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0x9F)), vceqq_u8(previous_u8x16, vdupq_n_u8(0xC3))));
        danger_u8x16 = vorrq_u8( // Dotted I (C4 B0)
            danger_u8x16, vandq_u8(is_after_c4_u8x16, vceqq_u8(text_u8x16, vdupq_n_u8(0xB0))));
        danger_u8x16 = vorrq_u8( // 'Ŀ' (C4 BF) → 'ŀ' (C5 80), crosses lead bytes
            danger_u8x16, vandq_u8(is_after_c4_u8x16, vceqq_u8(text_u8x16, vdupq_n_u8(0xBF))));
        danger_u8x16 = vorrq_u8( // Long S (C5 BF)
            danger_u8x16, vandq_u8(is_after_c5_u8x16, vceqq_u8(text_u8x16, vdupq_n_u8(0xBF))));
        danger_u8x16 = vorrq_u8( // 'Ÿ' (C5 B8) → 'ÿ' (C3 BF)
            danger_u8x16, vandq_u8(is_after_c5_u8x16, vceqq_u8(text_u8x16, vdupq_n_u8(0xB8))));
        danger_u8x16 = vorrq_u8( // Ligatures (EF AC xx)
            danger_u8x16, vandq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0xAC)), vceqq_u8(previous_u8x16, vdupq_n_u8(0xEF))));
        any_danger_u8x16 = vorrq_u8(any_danger_u8x16, danger_u8x16);
    }
    return vmaxvq_u8(any_danger_u8x16);
}

/**
 *  @brief Central European uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_central_europe_k
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_neon_central_europe_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                    //
    sz_cptr_t needle, sz_size_t needle_length,                        //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,         //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_search_neon_scripted_( //
        sz_utf8_uncased_search_neon_central_europe_fold_u8x16x2_,
        sz_utf8_uncased_search_neon_central_europe_alarm_u8x16x2_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Central European Uncased Find

#pragma region Cyrillic Uncased Find

/**
 *  @brief Fold a 32-byte chunk using Cyrillic case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_cyrillic_k
 *
 *  Basic Cyrillic has a clean high-nibble pattern on the second byte after a D0 lead:
 *  8x → +0x10 ('Ѐ'-'Џ' land in the D1 block), 9x → +0x20 ('А'-'П' stay under D0),
 *  Ax → −0x20 ('Р'-'Я' land in the D1 block), Bx → 0 (already lowercase). One `vqtbl1q_u8`
 *  lookup over a 16-entry table replaces 3 range comparisons + 3 masked adds. Extended Cyrillic
 *  (D2/D3) needles are BANNED at classification time, so only D0 continuations need folding.
 */
SZ_HELPER_NOINLINE uint8x16x2_t sz_utf8_uncased_search_neon_cyrillic_fold_u8x16x2_(uint8x16x2_t text_u8x16x2) {
    uint8x16x2_t result_u8x16x2 = sz_utf8_uncased_search_neon_ascii_fold_u8x16x2_(text_u8x16x2);
    uint8x16x2_t previous_bytes_u8x16x2 = sz_utf8_uncased_neon_previous_bytes_u8x16x2_(text_u8x16x2);
    uint8x16x2_t next_bytes_u8x16x2 = sz_utf8_uncased_neon_next_bytes_u8x16x2_(text_u8x16x2);

    // Second-byte offsets keyed by the high nibble: 8 → +0x10, 9 → +0x20, A → −0x20 (0xE0)
    static sz_u8_t const cyrillic_offset_lut[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x20, 0xE0, 0, 0, 0, 0, 0};
    uint8x16_t const cyrillic_offset_lut_u8x16 = vld1q_u8(cyrillic_offset_lut);

    for (sz_size_t register_index = 0; register_index != 2; ++register_index) {
        uint8x16_t text_u8x16 = text_u8x16x2.val[register_index];
        uint8x16_t previous_bytes_u8x16 = previous_bytes_u8x16x2.val[register_index];
        uint8x16_t next_bytes_u8x16 = next_bytes_u8x16x2.val[register_index];
        uint8x16_t result_u8x16 = result_u8x16x2.val[register_index];
        uint8x16_t is_after_d0_u8x16 = vceqq_u8(previous_bytes_u8x16, vdupq_n_u8(0xD0));

        uint8x16_t high_nibbles_u8x16 = vshrq_n_u8(text_u8x16, 4);
        uint8x16_t offsets_u8x16 = vqtbl1q_u8(cyrillic_offset_lut_u8x16, high_nibbles_u8x16);
        result_u8x16 = vaddq_u8(result_u8x16, vandq_u8(is_after_d0_u8x16, offsets_u8x16));

        // Lead fixup: 'Ѐ'-'Џ' (seconds 80-8F) and 'Р'-'Я' (seconds A0-AF) have lowercase in the D1
        // block, so their D0 lead takes a masked +1; 'А'-'П' (seconds 90-9F) stay under D0
        uint8x16_t is_d0_u8x16 = vceqq_u8(text_u8x16, vdupq_n_u8(0xD0));
        uint8x16_t needs_d1_u8x16 = vandq_u8(
            is_d0_u8x16, vorrq_u8(sz_utf8_uncased_neon_in_byte_range_u8x16_(next_bytes_u8x16, 0x80, 0x10),
                                  sz_utf8_uncased_neon_in_byte_range_u8x16_(next_bytes_u8x16, 0xA0, 0x10)));
        result_u8x16 = vaddq_u8(result_u8x16, vandq_u8(needs_d1_u8x16, vdupq_n_u8(0x01)));
        result_u8x16x2.val[register_index] = result_u8x16;
    }
    return result_u8x16x2;
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
SZ_HELPER_NOINLINE sz_u32_t sz_utf8_uncased_search_neon_cyrillic_alarm_u8x16x2_(uint8x16x2_t text_u8x16x2,
                                                                                sz_u32_t load_mask) {
    sz_unused_(load_mask); // Present for the shared `sz_utf8_uncased_alarm_u8x16x2_t_` signature

    // E1 B2 is dangerous only when the third (next) byte folds, i.e. lands in 80-88. Anchored at the
    // B2 second byte: lead from `previous`, third from `next`, gated by one `vmaxvq_u8`.
    uint8x16x2_t previous_u8x16x2 = sz_utf8_uncased_neon_previous_bytes_u8x16x2_(text_u8x16x2);
    uint8x16x2_t next_u8x16x2 = sz_utf8_uncased_neon_next_bytes_u8x16x2_(text_u8x16x2);

    uint8x16_t any_danger_u8x16 = vdupq_n_u8(0);
    for (sz_size_t register_index = 0; register_index != 2; ++register_index) {
        uint8x16_t danger_u8x16 = vandq_u8(
            vandq_u8(vceqq_u8(text_u8x16x2.val[register_index], vdupq_n_u8(0xB2)),
                     vceqq_u8(previous_u8x16x2.val[register_index], vdupq_n_u8(0xE1))),
            sz_utf8_uncased_neon_in_byte_range_u8x16_(next_u8x16x2.val[register_index], 0x80, 0x09));
        any_danger_u8x16 = vorrq_u8(any_danger_u8x16, danger_u8x16);
    }
    return vmaxvq_u8(any_danger_u8x16);
}

/**
 *  @brief Cyrillic uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_cyrillic_k
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_neon_cyrillic_( //
    sz_cptr_t haystack, sz_size_t haystack_length,              //
    sz_cptr_t needle, sz_size_t needle_length,                  //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,   //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_search_neon_scripted_( //
        sz_utf8_uncased_search_neon_cyrillic_fold_u8x16x2_,
        sz_utf8_uncased_search_neon_cyrillic_alarm_u8x16x2_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Cyrillic Uncased Find

#pragma region Armenian Uncased Find

/**
 *  @brief Fold a 32-byte chunk using Armenian case-folding rules.
 *  @sa sz_utf8_uncased_rune_safe_armenian_k
 *
 *  Armenian uppercase spans two lead bytes and folds into three target blocks:
 *  - D4 B1-BF: 'Ա'-'Ձ' → D5 A1-AF 'ա'-'ձ' (second −0x10, lead D4 → D5)
 *  - D5 80-8F: 'Ղ'-'Տ' → D5 B0-BF 'ղ'-'տ' (second +0x30, lead unchanged)
 *  - D5 90-96: 'Ր'-'Ֆ' → D6 80-86 'ր'-'ֆ' (second −0x10, lead D5 → D6)
 *
 *  Both lead rewrites are a +1 increment (D4 → D5, D5 → D6), so the second-byte flags propagate
 *  one lane back through `next_bytes` and join the single merged offset add - all rule masks flag
 *  disjoint byte positions. The D4 range checks only the lower bound, mirroring the reference:
 *  valid continuation bytes never exceed BF.
 */
SZ_HELPER_NOINLINE uint8x16x2_t sz_utf8_uncased_search_neon_armenian_fold_u8x16x2_(uint8x16x2_t text_u8x16x2) {
    uint8x16x2_t result_u8x16x2 = sz_utf8_uncased_search_neon_ascii_fold_u8x16x2_(text_u8x16x2);
    uint8x16x2_t previous_bytes_u8x16x2 = sz_utf8_uncased_neon_previous_bytes_u8x16x2_(text_u8x16x2);

    // The lead +1 bump comes from the SECOND byte's class shifted one lane forward, which must
    // cross the internal boundary - so the two per-register `is_minus_10` masks are assembled
    // first, then a single 32-byte `next_bytes` carries the high register's lane 0 to lane 15.
    uint8x16x2_t is_minus_10_u8x16x2;
    uint8x16x2_t is_d5_low_u8x16x2;
    for (sz_size_t register_index = 0; register_index != 2; ++register_index) {
        uint8x16_t text_u8x16 = text_u8x16x2.val[register_index];
        uint8x16_t previous_bytes_u8x16 = previous_bytes_u8x16x2.val[register_index];
        uint8x16_t is_after_d4_u8x16 = vceqq_u8(previous_bytes_u8x16, vdupq_n_u8(0xD4));
        uint8x16_t is_after_d5_u8x16 = vceqq_u8(previous_bytes_u8x16, vdupq_n_u8(0xD5));

        // Second-byte classes; the [B1, FF] range realizes the unbounded `≥ B1` check
        uint8x16_t is_d4_upper_u8x16 = vandq_u8(is_after_d4_u8x16, //
                                                sz_utf8_uncased_neon_in_byte_range_u8x16_(text_u8x16, 0xB1, 0x4F));
        uint8x16_t is_d5_low_u8x16 = vandq_u8(is_after_d5_u8x16, //
                                              sz_utf8_uncased_neon_in_byte_range_u8x16_(text_u8x16, 0x80, 0x10));
        uint8x16_t is_d5_high_u8x16 = vandq_u8(is_after_d5_u8x16, //
                                               sz_utf8_uncased_neon_in_byte_range_u8x16_(text_u8x16, 0x90, 0x07));
        is_minus_10_u8x16x2.val[register_index] = vorrq_u8(is_d4_upper_u8x16, is_d5_high_u8x16);
        is_d5_low_u8x16x2.val[register_index] = is_d5_low_u8x16;
    }

    uint8x16x2_t lead_plus_one_u8x16x2 = sz_utf8_uncased_neon_next_bytes_u8x16x2_(is_minus_10_u8x16x2);
    for (sz_size_t register_index = 0; register_index != 2; ++register_index) {
        // Disjoint positions merge into ONE offset vector and a single add
        uint8x16_t offsets_u8x16 = vandq_u8(is_minus_10_u8x16x2.val[register_index], vdupq_n_u8(0xF0));
        offsets_u8x16 = vorrq_u8(offsets_u8x16, vandq_u8(is_d5_low_u8x16x2.val[register_index], vdupq_n_u8(0x30)));
        offsets_u8x16 = vorrq_u8(offsets_u8x16, vandq_u8(lead_plus_one_u8x16x2.val[register_index], vdupq_n_u8(0x01)));
        result_u8x16x2.val[register_index] = vaddq_u8(result_u8x16x2.val[register_index], offsets_u8x16);
    }
    return result_u8x16x2;
}

/**
 *  @brief Alarm function for Armenian danger zone detection.
 *
 *  Detects positions where danger characters occur that require special handling:
 *  - D6 87: 'և' (U+0587) → "եւ" (D5 A5 D6 82), the Ech-Yiwn ligature (2 bytes → 4 bytes)
 *  - EF AC 93-97: presentation-form ligatures 'ﬓ'-'ﬗ' (U+FB13-U+FB17) → 2 codepoints each
 *
 *  The EF AC pair alarms without a third-byte refinement, exactly like the reference: the only
 *  EF AC neighbors are the Latin/Hebrew presentation forms, which never appear inside Armenian
 *  haystacks, so the coarser test costs nothing in practice.
 */
SZ_HELPER_NOINLINE sz_u32_t sz_utf8_uncased_search_neon_armenian_alarm_u8x16x2_(uint8x16x2_t text_u8x16x2,
                                                                                sz_u32_t load_mask) {
    sz_unused_(load_mask); // Present for the shared `sz_utf8_uncased_alarm_u8x16x2_t_` signature

    // Two two-byte pairs anchored at their second byte; lead from `previous`, gated by `vmaxvq_u8`.
    uint8x16x2_t previous_u8x16x2 = sz_utf8_uncased_neon_previous_bytes_u8x16x2_(text_u8x16x2);

    uint8x16_t any_danger_u8x16 = vdupq_n_u8(0);
    for (sz_size_t register_index = 0; register_index != 2; ++register_index) {
        uint8x16_t text_u8x16 = text_u8x16x2.val[register_index];
        uint8x16_t previous_u8x16 = previous_u8x16x2.val[register_index];
        uint8x16_t danger_u8x16 = vandq_u8( // Ech-Yiwn (D6 87)
            vceqq_u8(text_u8x16, vdupq_n_u8(0x87)), vceqq_u8(previous_u8x16, vdupq_n_u8(0xD6)));
        danger_u8x16 = vorrq_u8( // Ligatures (EF AC xx)
            danger_u8x16, vandq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0xAC)), vceqq_u8(previous_u8x16, vdupq_n_u8(0xEF))));
        any_danger_u8x16 = vorrq_u8(any_danger_u8x16, danger_u8x16);
    }
    return vmaxvq_u8(any_danger_u8x16);
}

/**
 *  @brief Armenian uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_armenian_k
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_neon_armenian_( //
    sz_cptr_t haystack, sz_size_t haystack_length,              //
    sz_cptr_t needle, sz_size_t needle_length,                  //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,   //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_search_neon_scripted_( //
        sz_utf8_uncased_search_neon_armenian_fold_u8x16x2_,
        sz_utf8_uncased_search_neon_armenian_alarm_u8x16x2_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Armenian Uncased Find

#pragma region Greek Uncased Find

/**
 *  @brief Fold a 32-byte chunk using Greek case-folding rules.
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
 *  the Eszett rewrite in the Western European fold. The lead rewrites must cross the internal
 *  boundary, so the second-byte class masks are assembled per register first, then a single
 *  32-byte `next_bytes` carries the high register's lane 0 to the low register's lane 15.
 */
SZ_HELPER_NOINLINE uint8x16x2_t sz_utf8_uncased_search_neon_greek_fold_u8x16x2_(uint8x16x2_t text_u8x16x2) {
    uint8x16x2_t result_u8x16x2 = sz_utf8_uncased_search_neon_ascii_fold_u8x16x2_(text_u8x16x2);
    uint8x16x2_t previous_bytes_u8x16x2 = sz_utf8_uncased_neon_previous_bytes_u8x16x2_(text_u8x16x2);

    uint8x16x4_t const ce_deltas_lut_u8x16x4 = vld1q_u8_x4(sz_utf8_uncased_greek_ce_deltas_lut_);
    uint8x16x4_t const ce_promotes_lut_u8x16x4 = vld1q_u8_x4(sz_utf8_uncased_greek_ce_promotes_lut_);

    uint8x16x2_t promote_seconds_u8x16x2; // CE → CF (+1) second-byte flags, pre-shift
    uint8x16x2_t micro_seconds_u8x16x2;   // C2 → CE (+0x0C) micro-sign second flags, pre-shift
    uint8x16x2_t partial_offsets_u8x16x2; // every delta that does NOT need a cross-boundary shift

    for (sz_size_t register_index = 0; register_index != 2; ++register_index) {
        uint8x16_t text_u8x16 = text_u8x16x2.val[register_index];
        uint8x16_t previous_bytes_u8x16 = previous_bytes_u8x16x2.val[register_index];
        uint8x16_t is_after_ce_u8x16 = vceqq_u8(previous_bytes_u8x16, vdupq_n_u8(0xCE));
        uint8x16_t is_after_cf_u8x16 = vceqq_u8(previous_bytes_u8x16, vdupq_n_u8(0xCF));
        uint8x16_t is_after_c2_u8x16 = vceqq_u8(previous_bytes_u8x16, vdupq_n_u8(0xC2));
        uint8x16_t is_continuation_u8x16 = vcltq_u8(vsubq_u8(text_u8x16, vdupq_n_u8(0x80)), vdupq_n_u8(0x40));
        uint8x16_t after_ce_cont_u8x16 = vandq_u8(is_after_ce_u8x16, is_continuation_u8x16);
        uint8x16_t delta_indices_u8x16 = vandq_u8(text_u8x16, vdupq_n_u8(0x3F));

        // Second-byte deltas after CE and the matching CE → CF promotion flags come from two table
        // lookups; the `is_continuation` mask keeps `text & 0x3F` from aliasing onto a folding index
        uint8x16_t ce_delta_u8x16 = vandq_u8(vqtbl4q_u8(ce_deltas_lut_u8x16x4, delta_indices_u8x16),
                                             after_ce_cont_u8x16);
        promote_seconds_u8x16x2.val[register_index] = vandq_u8(vqtbl4q_u8(ce_promotes_lut_u8x16x4, delta_indices_u8x16),
                                                               after_ce_cont_u8x16);

        // Final sigma 'ς' (CF 82) and the micro sign's second byte (C2 B5)
        uint8x16_t is_final_sigma_u8x16 = vandq_u8(is_after_cf_u8x16, vceqq_u8(text_u8x16, vdupq_n_u8(0x82)));
        uint8x16_t is_micro_second_u8x16 = vandq_u8(is_after_c2_u8x16, vceqq_u8(text_u8x16, vdupq_n_u8(0xB5)));
        micro_seconds_u8x16x2.val[register_index] = is_micro_second_u8x16;

        uint8x16_t offsets_u8x16 = ce_delta_u8x16;
        offsets_u8x16 = vorrq_u8(offsets_u8x16, vandq_u8(is_final_sigma_u8x16, vdupq_n_u8(0x01)));
        offsets_u8x16 = vorrq_u8(offsets_u8x16, vandq_u8(is_micro_second_u8x16, vdupq_n_u8(0x07)));
        partial_offsets_u8x16x2.val[register_index] = offsets_u8x16;
    }

    // Propagate the lead rewrites back from the second-byte flags across the internal boundary
    uint8x16x2_t promote_lead_u8x16x2 = sz_utf8_uncased_neon_next_bytes_u8x16x2_(promote_seconds_u8x16x2);
    uint8x16x2_t micro_lead_u8x16x2 = sz_utf8_uncased_neon_next_bytes_u8x16x2_(micro_seconds_u8x16x2);
    for (sz_size_t register_index = 0; register_index != 2; ++register_index) {
        uint8x16_t offsets_u8x16 = partial_offsets_u8x16x2.val[register_index];
        offsets_u8x16 = vorrq_u8(offsets_u8x16, vandq_u8(promote_lead_u8x16x2.val[register_index], vdupq_n_u8(0x01)));
        offsets_u8x16 = vorrq_u8(offsets_u8x16, vandq_u8(micro_lead_u8x16x2.val[register_index], vdupq_n_u8(0x0C)));
        result_u8x16x2.val[register_index] = vaddq_u8(result_u8x16x2.val[register_index], offsets_u8x16);
    }
    return result_u8x16x2;
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
SZ_HELPER_NOINLINE sz_u32_t sz_utf8_uncased_search_neon_greek_alarm_u8x16x2_(uint8x16x2_t text_u8x16x2,
                                                                             sz_u32_t load_mask) {
    sz_unused_(load_mask); // Present for the shared `sz_utf8_uncased_alarm_u8x16x2_t_` signature

    // Pair danger anchored at the second byte (lead from `previous`); the polytonic & archaic leads
    // E1/CD are blanket hazards flagged at their own lane. One `vmaxvq_u8` gates the danger branch.
    uint8x16x2_t previous_u8x16x2 = sz_utf8_uncased_neon_previous_bytes_u8x16x2_(text_u8x16x2);

    uint8x16_t any_danger_u8x16 = vdupq_n_u8(0);
    for (sz_size_t register_index = 0; register_index != 2; ++register_index) {
        uint8x16_t text_u8x16 = text_u8x16x2.val[register_index];
        uint8x16_t previous_u8x16 = previous_u8x16x2.val[register_index];

        // 'ΐ', 'ΰ' (CE 90 / CE B0)
        uint8x16_t second_90_or_b0_u8x16 = vorrq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0x90)),
                                                    vceqq_u8(text_u8x16, vdupq_n_u8(0xB0)));
        uint8x16_t danger_u8x16 = vandq_u8(vceqq_u8(previous_u8x16, vdupq_n_u8(0xCE)), second_90_or_b0_u8x16);

        // Greek symbols (CF 9x / CF Bx)
        uint8x16_t second_9x_u8x16 = vorrq_u8(
            vorrq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0x90)), vceqq_u8(text_u8x16, vdupq_n_u8(0x91))),
            vorrq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0x95)), vceqq_u8(text_u8x16, vdupq_n_u8(0x96))));
        uint8x16_t second_bx_u8x16 = vorrq_u8(
            vorrq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0xB0)), vceqq_u8(text_u8x16, vdupq_n_u8(0xB1))),
            vorrq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0xB4)), vceqq_u8(text_u8x16, vdupq_n_u8(0xB5))));
        danger_u8x16 = vorrq_u8(danger_u8x16, vandq_u8(vceqq_u8(previous_u8x16, vdupq_n_u8(0xCF)),
                                                       vorrq_u8(second_9x_u8x16, second_bx_u8x16)));

        // Ohm sign (E2 84 A6)
        danger_u8x16 = vorrq_u8(
            danger_u8x16, vandq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0x84)), vceqq_u8(previous_u8x16, vdupq_n_u8(0xE2))));

        // Blanket polytonic & archaic leads
        danger_u8x16 = vorrq_u8(
            danger_u8x16, vorrq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0xE1)), vceqq_u8(text_u8x16, vdupq_n_u8(0xCD))));
        any_danger_u8x16 = vorrq_u8(any_danger_u8x16, danger_u8x16);
    }
    return vmaxvq_u8(any_danger_u8x16);
}

/**
 *  @brief Greek uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_greek_k
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_neon_greek_(  //
    sz_cptr_t haystack, sz_size_t haystack_length,            //
    sz_cptr_t needle, sz_size_t needle_length,                //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata, //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_search_neon_scripted_( //
        sz_utf8_uncased_search_neon_greek_fold_u8x16x2_,
        sz_utf8_uncased_search_neon_greek_alarm_u8x16x2_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Greek Uncased Find

#pragma region Vietnamese Uncased Find

/**
 *  @brief Fold a 32-byte chunk using Vietnamese case-folding rules.
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
SZ_HELPER_NOINLINE uint8x16x2_t sz_utf8_uncased_search_neon_vietnamese_fold_u8x16x2_(uint8x16x2_t text_u8x16x2) {
    uint8x16x2_t result_u8x16x2 = sz_utf8_uncased_search_neon_ascii_fold_u8x16x2_(text_u8x16x2);
    uint8x16x2_t previous_bytes_u8x16x2 = sz_utf8_uncased_neon_previous_bytes_u8x16x2_(text_u8x16x2);
    uint8x16x2_t previous2_bytes_u8x16x2 = sz_utf8_uncased_neon_previous_bytes_u8x16x2_(previous_bytes_u8x16x2);

    for (sz_size_t register_index = 0; register_index != 2; ++register_index) {
        uint8x16_t text_u8x16 = text_u8x16x2.val[register_index];
        uint8x16_t previous_bytes_u8x16 = previous_bytes_u8x16x2.val[register_index];
        uint8x16_t previous2_bytes_u8x16 = previous2_bytes_u8x16x2.val[register_index];
        uint8x16_t result_u8x16 = result_u8x16x2.val[register_index];
        uint8x16_t is_after_c3_u8x16 = vceqq_u8(previous_bytes_u8x16, vdupq_n_u8(0xC3));
        uint8x16_t is_after_c4_u8x16 = vceqq_u8(previous_bytes_u8x16, vdupq_n_u8(0xC4));
        uint8x16_t is_after_c5_u8x16 = vceqq_u8(previous_bytes_u8x16, vdupq_n_u8(0xC5));
        uint8x16_t is_after_c6_u8x16 = vceqq_u8(previous_bytes_u8x16, vdupq_n_u8(0xC6));

        // 1. Latin-1 Supplement: C3 80-9E → +0x20, except '×' (C3 97)
        uint8x16_t is_c3_target_u8x16 = vandq_u8(
            is_after_c3_u8x16, vbicq_u8(sz_utf8_uncased_neon_in_byte_range_u8x16_(text_u8x16, 0x80, 0x1F),
                                        vceqq_u8(text_u8x16, vdupq_n_u8(0x97))));

        // 2. Latin Extended-A: +1 on EVEN seconds, except the inverted sub-ranges C4 B9-BE and
        //    C5 00-88 (the unsigned `≤ 88` bound mirrors the reference) which fold ODD
        uint8x16_t is_odd_u8x16 = vceqq_u8(vandq_u8(text_u8x16, vdupq_n_u8(0x01)), vdupq_n_u8(0x01));
        uint8x16_t is_inverted_u8x16 = vorrq_u8(
            vandq_u8(is_after_c4_u8x16, sz_utf8_uncased_neon_in_byte_range_u8x16_(text_u8x16, 0xB9, 0x06)),
            vandq_u8(is_after_c5_u8x16, sz_utf8_uncased_neon_in_byte_range_u8x16_(text_u8x16, 0x00, 0x89)));
        uint8x16_t is_extended_even_u8x16 = vbicq_u8(
            vbicq_u8(vorrq_u8(is_after_c4_u8x16, is_after_c5_u8x16), is_inverted_u8x16), is_odd_u8x16);
        uint8x16_t fold_extended_u8x16 = vorrq_u8(is_extended_even_u8x16, vandq_u8(is_inverted_u8x16, is_odd_u8x16));

        // 3. Latin Extended-B: 'Ơ' (C6 A0) and 'Ư' (C6 AF) → +1
        uint8x16_t is_c6_target_u8x16 = vandq_u8(is_after_c6_u8x16, vorrq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0xA0)),
                                                                             vceqq_u8(text_u8x16, vdupq_n_u8(0xAF))));

        // 4. Latin Extended Additional: EVEN third bytes after an E1 B8-BB pair → +1,
        //    except the expanding E1 BA 96-9F block
        uint8x16_t is_after_e1_pair_u8x16 = vandq_u8(
            vceqq_u8(previous2_bytes_u8x16, vdupq_n_u8(0xE1)),
            sz_utf8_uncased_neon_in_byte_range_u8x16_(previous_bytes_u8x16, 0xB8, 0x04));
        uint8x16_t is_excluded_third_u8x16 = vandq_u8(
            vceqq_u8(previous_bytes_u8x16, vdupq_n_u8(0xBA)),
            sz_utf8_uncased_neon_in_byte_range_u8x16_(text_u8x16, 0x96, 0x0A));
        uint8x16_t fold_e1_u8x16 = vbicq_u8(vbicq_u8(is_after_e1_pair_u8x16, is_excluded_third_u8x16), is_odd_u8x16);

        // Disjoint positions merge into ONE offset vector and a single add
        uint8x16_t is_plus_one_u8x16 = vorrq_u8(fold_extended_u8x16, vorrq_u8(is_c6_target_u8x16, fold_e1_u8x16));
        uint8x16_t offsets_u8x16 = vorrq_u8(vandq_u8(is_c3_target_u8x16, vdupq_n_u8(0x20)),
                                            vandq_u8(is_plus_one_u8x16, vdupq_n_u8(0x01)));
        result_u8x16x2.val[register_index] = vaddq_u8(result_u8x16, offsets_u8x16);
    }
    return result_u8x16x2;
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
 *  reference bit-for-bit.
 */
SZ_HELPER_NOINLINE sz_u32_t sz_utf8_uncased_search_neon_vietnamese_alarm_u8x16x2_(uint8x16x2_t text_u8x16x2,
                                                                                  sz_u32_t load_mask) {
    sz_unused_(load_mask); // Padded loads zero absent bytes, so range compares are safe-negative

    // All hazards anchored at their second byte: lead from `previous`, the E1 BA expanding third
    // byte from `next`. The dense-E1 early exit is unnecessary now that the per-value movemasks are
    // gone - the whole alarm is a handful of `vceqq`/`vand` ops gated by one `vmaxvq_u8`.
    uint8x16x2_t previous_u8x16x2 = sz_utf8_uncased_neon_previous_bytes_u8x16x2_(text_u8x16x2);
    uint8x16x2_t next_u8x16x2 = sz_utf8_uncased_neon_next_bytes_u8x16x2_(text_u8x16x2);

    uint8x16_t any_danger_u8x16 = vdupq_n_u8(0);
    for (sz_size_t register_index = 0; register_index != 2; ++register_index) {
        uint8x16_t text_u8x16 = text_u8x16x2.val[register_index];
        uint8x16_t previous_u8x16 = previous_u8x16x2.val[register_index];
        uint8x16_t next_u8x16 = next_u8x16x2.val[register_index];

        uint8x16_t danger_u8x16 = vandq_u8( // E1 BA 96-9F (expanding third byte)
            vandq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0xBA)), vceqq_u8(previous_u8x16, vdupq_n_u8(0xE1))),
            sz_utf8_uncased_neon_in_byte_range_u8x16_(next_u8x16, 0x96, 0x0A));
        danger_u8x16 = vorrq_u8( // Sharp S (C3 9F)
            danger_u8x16, vandq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0x9F)), vceqq_u8(previous_u8x16, vdupq_n_u8(0xC3))));
        danger_u8x16 = vorrq_u8( // Long S (C5 BF)
            danger_u8x16, vandq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0xBF)), vceqq_u8(previous_u8x16, vdupq_n_u8(0xC5))));
        danger_u8x16 = vorrq_u8( // Ligatures (EF AC xx)
            danger_u8x16, vandq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0xAC)), vceqq_u8(previous_u8x16, vdupq_n_u8(0xEF))));
        danger_u8x16 = vorrq_u8( // Kelvin (E2 84 xx)
            danger_u8x16, vandq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0x84)), vceqq_u8(previous_u8x16, vdupq_n_u8(0xE2))));
        any_danger_u8x16 = vorrq_u8(any_danger_u8x16, danger_u8x16);
    }
    return vmaxvq_u8(any_danger_u8x16);
}

/**
 *  @brief Vietnamese uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_vietnamese_k
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_neon_vietnamese_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                //
    sz_cptr_t needle, sz_size_t needle_length,                    //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,     //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_search_neon_scripted_( //
        sz_utf8_uncased_search_neon_vietnamese_fold_u8x16x2_,
        sz_utf8_uncased_search_neon_vietnamese_alarm_u8x16x2_, //
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
 *  SEQUENCE-START positions, mirroring the reference bit-for-bit.
 */
SZ_HELPER_NOINLINE sz_u32_t sz_utf8_uncased_search_neon_georgian_alarm_u8x16x2_(uint8x16x2_t text_u8x16x2,
                                                                                sz_u32_t load_mask) {
    sz_unused_(load_mask); // Padded loads zero absent bytes, so range compares are safe-negative

    // E1 B2 = Mtavruli; E1 82 refines on the A0-E5 third (next) byte for Asomtavruli; E2 B4 =
    // Nuskhuri. Anchored at the second byte (lead from `previous`), gated by one `vmaxvq_u8`.
    uint8x16x2_t previous_u8x16x2 = sz_utf8_uncased_neon_previous_bytes_u8x16x2_(text_u8x16x2);
    uint8x16x2_t next_u8x16x2 = sz_utf8_uncased_neon_next_bytes_u8x16x2_(text_u8x16x2);

    uint8x16_t any_danger_u8x16 = vdupq_n_u8(0);
    for (sz_size_t register_index = 0; register_index != 2; ++register_index) {
        uint8x16_t text_u8x16 = text_u8x16x2.val[register_index];
        uint8x16_t previous_u8x16 = previous_u8x16x2.val[register_index];
        uint8x16_t next_u8x16 = next_u8x16x2.val[register_index];
        uint8x16_t is_after_e1_u8x16 = vceqq_u8(previous_u8x16, vdupq_n_u8(0xE1));

        uint8x16_t danger_u8x16 = vandq_u8( // Mtavruli (E1 B2)
            vceqq_u8(text_u8x16, vdupq_n_u8(0xB2)), is_after_e1_u8x16);
        danger_u8x16 = vorrq_u8( // Asomtavruli (E1 82 A0-E5)
            danger_u8x16, vandq_u8(vandq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0x82)), is_after_e1_u8x16),
                                   sz_utf8_uncased_neon_in_byte_range_u8x16_(next_u8x16, 0xA0, 0x46)));
        danger_u8x16 = vorrq_u8( // Nuskhuri (E2 B4)
            danger_u8x16, vandq_u8(vceqq_u8(text_u8x16, vdupq_n_u8(0xB4)), vceqq_u8(previous_u8x16, vdupq_n_u8(0xE2))));
        any_danger_u8x16 = vorrq_u8(any_danger_u8x16, danger_u8x16);
    }
    return vmaxvq_u8(any_danger_u8x16);
}

/**
 *  @brief Georgian uncased search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_uncased_rune_safe_georgian_k
 *
 *  The fastest non-ASCII kernel: Mkhedruli is caseless, so the fold callback is just the
 *  ASCII fold for mixed Latin text and the alarm only watches for the historical scripts.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_uncased_search_neon_georgian_( //
    sz_cptr_t haystack, sz_size_t haystack_length,              //
    sz_cptr_t needle, sz_size_t needle_length,                  //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata,   //
    sz_size_t *matched_length) {
    return sz_utf8_uncased_search_neon_scripted_( //
        sz_utf8_uncased_search_neon_ascii_fold_u8x16x2_,
        sz_utf8_uncased_search_neon_georgian_alarm_u8x16x2_, //
        haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
}

#pragma endregion // Georgian Uncased Find

SZ_API_COMPTIME sz_cptr_t sz_utf8_uncased_search_neon( //
    sz_cptr_t haystack, sz_size_t haystack_length,     //
    sz_cptr_t needle, sz_size_t needle_length,         //
    sz_utf8_uncased_needle_metadata_t *needle_metadata, sz_size_t *matched_length) {

    // Handle the obvious edge cases first
    if (needle_length == 0) {
        *matched_length = 0;
        return haystack;
    }

    // If the needle is entirely made of case-less characters - perform direct substring search
    int const is_unknown = needle_metadata->kernel_id == sz_utf8_uncased_rune_unknown_k;
    int const known_agnostic = needle_metadata->kernel_id == sz_utf8_uncased_rune_invariant_k;
    if (known_agnostic || (is_unknown && sz_utf8_find_cased_neon(needle, needle_length) == SZ_NULL_CHAR)) {
        sz_cptr_t result = sz_find_neon(haystack, haystack_length, needle, needle_length);
        *matched_length = result ? needle_length : 0;
        return result;
    }

    // Analyze needle to find the best safe window for each script
    if (is_unknown) {
        sz_utf8_uncased_needle_metadata_(needle, needle_length, needle_metadata);
        // If no SIMD-safe window found, fall back to serial immediately
        if (needle_metadata->kernel_id == sz_utf8_uncased_rune_fallback_serial_k)
            return sz_utf8_uncased_search_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                                 matched_length);
    }

    // Dispatch to appropriate kernel
    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_ascii_invariant_k) {
        if (needle_metadata->folded_slice_length <= 3)
            return sz_utf8_uncased_search_neon_ascii_3probe_( //
                haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
        else
            return sz_utf8_uncased_search_neon_ascii_4probe_( //
                haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    }

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_western_europe_k)
        return sz_utf8_uncased_search_neon_western_europe_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_central_europe_k)
        return sz_utf8_uncased_search_neon_central_europe_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_cyrillic_k)
        return sz_utf8_uncased_search_neon_cyrillic_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_greek_k)
        return sz_utf8_uncased_search_neon_greek_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_armenian_k)
        return sz_utf8_uncased_search_neon_armenian_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_vietnamese_k)
        return sz_utf8_uncased_search_neon_vietnamese_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_uncased_rune_safe_georgian_k)
        return sz_utf8_uncased_search_neon_georgian_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    // No suitable SIMD path found (needle has complex Unicode), fall back to serial
    needle_metadata->kernel_id = sz_utf8_uncased_rune_fallback_serial_k;
    return sz_utf8_uncased_search_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                         matched_length);
}

SZ_API_COMPTIME sz_cptr_t sz_utf8_find_cased_neon(sz_cptr_t str, sz_size_t length) {
    sz_cptr_t text_cursor = str;

    // Single loop: advance by min(length, 29), check leads in the first `block_length` positions;
    // the 3-byte slack keeps every checked lead's continuation bytes inside the same 32-byte load
    while (length) {
        sz_size_t block_length = sz_min_of_two(length, 29);
        sz_u32_t lead_mask = sz_utf8_uncased_neon_mask_until_(block_length);
        uint8x16x2_t text_u8x16x2 = length >= 32 ? vld1q_u8_x2((sz_u8_t const *)text_cursor)
                                                 : sz_utf8_uncased_neon_load_padded_u8x16x2_(text_cursor, length);
        uint8x16_t low_u8x16 = text_u8x16x2.val[0], high_u8x16 = text_u8x16x2.val[1];

        // 1. ASCII letter check (zeros beyond the string are fine - not letters)
        sz_u32_t is_upper_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(
            sz_utf8_uncased_neon_in_byte_range_u8x16_(low_u8x16, 'A', 26),
            sz_utf8_uncased_neon_in_byte_range_u8x16_(high_u8x16, 'A', 26));
        sz_u32_t is_lower_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(
            sz_utf8_uncased_neon_in_byte_range_u8x16_(low_u8x16, 'a', 26),
            sz_utf8_uncased_neon_in_byte_range_u8x16_(high_u8x16, 'a', 26));
        if (is_upper_mask | is_lower_mask) return sz_utf8_find_cased_serial(text_cursor, length);

        // 2. Check for non-ASCII in lead positions
        sz_u32_t is_non_ascii_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(vcgeq_u8(low_u8x16, vdupq_n_u8(0x80)),
                                                                            vcgeq_u8(high_u8x16, vdupq_n_u8(0x80))) &
                                     lead_mask;
        if (is_non_ascii_mask) {
            // 3. Identify UTF-8 lead bytes
            uint8x16_t const xe0_u8x16 = vdupq_n_u8(0xE0);
            uint8x16_t const xf0_u8x16 = vdupq_n_u8(0xF0);
            sz_u32_t is_two_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(
                                       vceqq_u8(vandq_u8(low_u8x16, xe0_u8x16), vdupq_n_u8(0xC0)),
                                       vceqq_u8(vandq_u8(high_u8x16, xe0_u8x16), vdupq_n_u8(0xC0))) &
                                   lead_mask;
            sz_u32_t is_three_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(
                                         vceqq_u8(vandq_u8(low_u8x16, xf0_u8x16), xe0_u8x16),
                                         vceqq_u8(vandq_u8(high_u8x16, xf0_u8x16), xe0_u8x16)) &
                                     lead_mask;
            sz_u32_t is_four_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(
                                        vceqq_u8(vandq_u8(low_u8x16, vdupq_n_u8(0xF8)), xf0_u8x16),
                                        vceqq_u8(vandq_u8(high_u8x16, vdupq_n_u8(0xF8)), xf0_u8x16)) &
                                    lead_mask;

            // 4. Check 4-byte bicameral scripts (SMP): F0 with second byte 90/91/96/9D/9E
            if (is_four_mask) {
                sz_u32_t after_f0_mask = is_four_mask << 1;
                sz_u32_t is_90_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(vceqq_u8(low_u8x16, vdupq_n_u8(0x90)),
                                                                             vceqq_u8(high_u8x16, vdupq_n_u8(0x90)));
                sz_u32_t is_91_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(vceqq_u8(low_u8x16, vdupq_n_u8(0x91)),
                                                                             vceqq_u8(high_u8x16, vdupq_n_u8(0x91)));
                sz_u32_t is_96_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(vceqq_u8(low_u8x16, vdupq_n_u8(0x96)),
                                                                             vceqq_u8(high_u8x16, vdupq_n_u8(0x96)));
                sz_u32_t is_9d_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(vceqq_u8(low_u8x16, vdupq_n_u8(0x9D)),
                                                                             vceqq_u8(high_u8x16, vdupq_n_u8(0x9D)));
                sz_u32_t is_9e_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(vceqq_u8(low_u8x16, vdupq_n_u8(0x9E)),
                                                                             vceqq_u8(high_u8x16, vdupq_n_u8(0x9E)));
                if (after_f0_mask & (is_90_mask | is_91_mask | is_96_mask | is_9d_mask | is_9e_mask))
                    return sz_utf8_find_cased_serial(text_cursor, length);
            }

            // 5. Check 2-byte bicameral leads: C3-D6
            // C3-CF: Latin Extended (umlauts, accents, Eszett)
            // D0-D1: Cyrillic, D4-D6: Armenian (D6 needed for small letters U+0580+)
            if (is_two_mask) {
                sz_u32_t is_bicameral_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(
                    sz_utf8_uncased_neon_in_byte_range_u8x16_(low_u8x16, 0xC3, 0x14),
                    sz_utf8_uncased_neon_in_byte_range_u8x16_(high_u8x16, 0xC3, 0x14));

                // Special case: C2 B5 = U+00B5 MICRO SIGN folds to Greek mu (U+03BC)
                sz_u32_t is_c2_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(vceqq_u8(low_u8x16, vdupq_n_u8(0xC2)),
                                                                             vceqq_u8(high_u8x16, vdupq_n_u8(0xC2))) &
                                      is_two_mask;
                if (is_c2_mask) {
                    sz_u32_t is_b5_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(
                        vceqq_u8(low_u8x16, vdupq_n_u8(0xB5)), vceqq_u8(high_u8x16, vdupq_n_u8(0xB5)));
                    if ((is_c2_mask << 1) & is_b5_mask) return sz_utf8_find_cased_serial(text_cursor, length);
                }

                // Note: CA 80-BF includes both IPA Extensions (U+0280-02AF) and Spacing Modifier Letters
                // (U+02B0-02BF). Spacing Modifier Letters CAN appear in case fold expansions:
                // e.g., ẚ (U+1E9A) folds to [a, ʾ] where ʾ = U+02BE is a Spacing Modifier Letter.
                // So we must NOT exclude this range from the bicameral check.
                if (is_bicameral_mask & is_two_mask) return sz_utf8_find_cased_serial(text_cursor, length);
            }

            // 6. Check 3-byte bicameral sequences
            if (is_three_mask) {
                // E1: Georgian, Greek Extended, Latin Extended Additional
                sz_u32_t is_e1_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(vceqq_u8(low_u8x16, vdupq_n_u8(0xE1)),
                                                                             vceqq_u8(high_u8x16, vdupq_n_u8(0xE1)));
                if (is_e1_mask & is_three_mask) return sz_utf8_find_cased_serial(text_cursor, length);

                // EF: Fullwidth Latin
                sz_u32_t is_ef_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(vceqq_u8(low_u8x16, vdupq_n_u8(0xEF)),
                                                                             vceqq_u8(high_u8x16, vdupq_n_u8(0xEF)));
                if (is_ef_mask & is_three_mask) return sz_utf8_find_cased_serial(text_cursor, length);

                // E2: Safe only for second byte 80-83
                sz_u32_t is_e2_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(vceqq_u8(low_u8x16, vdupq_n_u8(0xE2)),
                                                                             vceqq_u8(high_u8x16, vdupq_n_u8(0xE2))) &
                                      is_three_mask;
                if (is_e2_mask) {
                    sz_u32_t e2_second_safe_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(
                        sz_utf8_uncased_neon_in_byte_range_u8x16_(low_u8x16, 0x80, 0x04),
                        sz_utf8_uncased_neon_in_byte_range_u8x16_(high_u8x16, 0x80, 0x04));
                    if ((is_e2_mask << 1) & ~e2_second_safe_mask) return sz_utf8_find_cased_serial(text_cursor, length);
                }

                // EA: Bicameral second bytes 99-9F, AC-AE
                sz_u32_t is_ea_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(vceqq_u8(low_u8x16, vdupq_n_u8(0xEA)),
                                                                             vceqq_u8(high_u8x16, vdupq_n_u8(0xEA))) &
                                      is_three_mask;
                if (is_ea_mask) {
                    sz_u32_t is_99_range_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(
                        sz_utf8_uncased_neon_in_byte_range_u8x16_(low_u8x16, 0x99, 0x07),
                        sz_utf8_uncased_neon_in_byte_range_u8x16_(high_u8x16, 0x99, 0x07));
                    sz_u32_t is_ac_range_mask = sz_utf8_uncased_neon_movemask_u8x16x2_(
                        sz_utf8_uncased_neon_in_byte_range_u8x16_(low_u8x16, 0xAC, 0x03),
                        sz_utf8_uncased_neon_in_byte_range_u8x16_(high_u8x16, 0xAC, 0x03));
                    if ((is_ea_mask << 1) & (is_99_range_mask | is_ac_range_mask))
                        return sz_utf8_find_cased_serial(text_cursor, length);
                }
            }
        }

        text_cursor += block_length;
        length -= block_length;
    }

    return SZ_NULL_CHAR;
}

SZ_API_COMPTIME sz_ordering_t sz_utf8_uncased_order_neon(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                         sz_size_t b_length) {
    return sz_utf8_uncased_order_serial(a, a_length, b, b_length);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEON

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_UNCASED_NEON_H_
