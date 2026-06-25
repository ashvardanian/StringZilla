/**
 *  @brief Serial backend for UTF-8 codepoint mechanics (count, find-nth, unpack) and shared decode helpers.
 *  @file include/stringzilla/utf8_runes/serial.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_RUNES_SERIAL_H_
#define STRINGZILLA_UTF8_RUNES_SERIAL_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma region Rune Codec

/** @brief Whether @p byte is a UTF-8 continuation byte (`0x80..0xBF`). The single low-level predicate every decode
 *         path shares, so `sz_rune_decode` and `sz_utf8_maximal_subpart_` can never disagree on validity. */
SZ_INTERNAL sz_bool_t sz_utf8_is_continuation_(sz_u8_t byte) { return (sz_bool_t)((byte & 0xC0) == 0x80); }

/** @brief Whether @p second is a valid @b first continuation for lead byte @p lead: a continuation byte that also
 *         satisfies the E0/ED/F0/F4 overlong/surrogate/range constraint (Unicode Table 3-7). For C2..DF and the
 *         unconstrained 3/4-byte leads it is just the continuation test. Shared by `sz_rune_decode` and
 *         `sz_utf8_maximal_subpart_` so the "is byte 2 in range" rule lives in exactly one place. */
SZ_INTERNAL sz_bool_t sz_utf8_first_continuation_ok_(sz_u8_t lead, sz_u8_t second) {
    if ((second & 0xC0) != 0x80) return sz_false_k;
    if (lead == 0xE0) return (sz_bool_t)(second >= 0xA0); // overlong 3-byte
    if (lead == 0xED) return (sz_bool_t)(second < 0xA0);  // surrogate U+D800..U+DFFF
    if (lead == 0xF0) return (sz_bool_t)(second >= 0x90); // overlong 4-byte
    if (lead == 0xF4) return (sz_bool_t)(second < 0x90);  // > U+10FFFF
    return sz_true_k;
}

/** @brief Bounds-checked, validating UTF-8 decode. Returns the codepoint's 1-4 byte length in @p rune, or
 *         `sz_rune_invalid_k` (0) when the bytes at @p utf8 do not begin a well-formed, shortest-form,
 *         non-surrogate, in-range codepoint (a truncated trailing sequence before @p utf8_end included). The
 *         single authority for "is this a foldable/normalizable rune"; the decode-side mirror of `sz_rune_encode`.
 *         On failure use `sz_utf8_maximal_subpart_` for how many bytes the resulting U+FFFD consumes. */
SZ_INTERNAL sz_rune_length_t sz_rune_decode(sz_cptr_t utf8, sz_cptr_t utf8_end, sz_rune_t *rune) {
    sz_u8_t const *u = (sz_u8_t const *)utf8;
    sz_size_t const available = (sz_size_t)((sz_u8_t const *)utf8_end - u);
    sz_u8_t const lead = u[0];
    if (lead < 0x80) {
        *rune = lead;
        return sz_rune_1byte_k;
    }
    if (lead < 0xC2) return sz_rune_invalid_k; // continuation byte, or C0/C1 (overlong 2-byte)
    if (lead < 0xE0) {                         // C2..DF
        if (available < 2 || !sz_utf8_first_continuation_ok_(lead, u[1])) return sz_rune_invalid_k;
        *rune = (sz_rune_t)(lead & 0x1F) << 6 | (u[1] & 0x3F);
        return sz_rune_2bytes_k;
    }
    if (lead < 0xF0) { // E0..EF
        if (available < 3 || !sz_utf8_first_continuation_ok_(lead, u[1]) || !sz_utf8_is_continuation_(u[2]))
            return sz_rune_invalid_k;
        *rune = (sz_rune_t)(lead & 0x0F) << 12 | (sz_rune_t)(u[1] & 0x3F) << 6 | (u[2] & 0x3F);
        return sz_rune_3bytes_k;
    }
    if (lead <= 0xF4) { // F0..F4
        if (available < 4 || !sz_utf8_first_continuation_ok_(lead, u[1]) || !sz_utf8_is_continuation_(u[2]) ||
            !sz_utf8_is_continuation_(u[3]))
            return sz_rune_invalid_k;
        *rune = (sz_rune_t)(lead & 0x07) << 18 | (sz_rune_t)(u[1] & 0x3F) << 12 | (sz_rune_t)(u[2] & 0x3F) << 6 |
                (u[3] & 0x3F);
        return sz_rune_4bytes_k;
    }
    return sz_rune_invalid_k; // F5..FF
}

/** @brief Byte length (1..3) of the maximal ill-formed subpart starting at @p utf8, per Unicode 17.0 §3.9 and the
 *         W3C Encoding Standard: the longest prefix that matches the start of @b some well-formed sequence, so a
 *         single U+FFFD replaces it and decoding resyncs immediately after. A bad lead (stray continuation, C0/C1,
 *         F5..FF) is length 1; a good 3/4-byte lead whose continuation chain breaks at byte 2 or 3 yields 2 or 3.
 *         @pre `[utf8, utf8_end)` does not begin a well-formed rune (`sz_rune_decode` returned `sz_rune_invalid_k`).
 *         Shares `sz_utf8_first_continuation_ok_`/`sz_utf8_is_continuation_` with `sz_rune_decode` so the two never
 *         disagree on where a sequence stops being well-formed. */
SZ_INTERNAL sz_size_t sz_utf8_maximal_subpart_(sz_cptr_t utf8, sz_cptr_t utf8_end) {
    sz_u8_t const *u = (sz_u8_t const *)utf8;
    sz_size_t const available = (sz_size_t)((sz_u8_t const *)utf8_end - u);
    sz_u8_t const lead = u[0];
    // A bad lead is its own 1-byte subpart: stray continuation (< 0xC2), C0/C1 overlong, F5..FF out of range.
    if (lead < 0xC2 || lead > 0xF4) return 1;
    // 2/3/4-byte leads: count leading bytes consistent with a well-formed sequence, stopping at the first break.
    if (available < 2 || !sz_utf8_first_continuation_ok_(lead, u[1])) return 1; // byte 1 breaks it (incl. C2..DF)
    if (lead < 0xE0) return 1;                                      // C2..DF + good b1 would be well-formed (defensive)
    if (available < 3 || !sz_utf8_is_continuation_(u[2])) return 2; // byte 2 breaks it
    return 3;                                                       // E0..F4: break is at byte 3 (b1, b2 are good)
}

/** @brief Decode the 1-4 byte sequence the lead byte declares, with NO bounds check and NO validation; returns
 *         the byte length and stores the codepoint in @p rune.
 *  @warning Assumes valid, complete UTF-8 (a truncated trailing sequence over-reads). Use `sz_rune_decode` for the
 *           bounds-checked + validating variant, or `sz_utf8_valid()` first. */
SZ_INTERNAL sz_rune_length_t sz_rune_decode_unchecked(sz_cptr_t utf8, sz_rune_t *rune) {
    sz_u8_t const *u8s = (sz_u8_t const *)utf8;
    sz_u8_t lead = *u8s++;
    sz_rune_length_t length = (sz_rune_length_t)(1 + (lead >= 0xC0U) + (lead >= 0xE0U) + (lead >= 0xF0U));
    switch (length) {
    case 1: *rune = lead; break;
    case 2: *rune = (lead & 0x1FU) << 6 | (u8s[0] & 0x3FU); break;
    case 3: *rune = (lead & 0x0FU) << 12 | (u8s[0] & 0x3FU) << 6 | (u8s[1] & 0x3FU); break;
    default:
        *rune = (sz_rune_t)(lead & 0x07U) << 18 | (u8s[0] & 0x3FU) << 12 | (u8s[1] & 0x3FU) << 6 | (u8s[2] & 0x3FU);
        break;
    }
    return length;
}

/** @brief Encode a UTF-32 codepoint to UTF-8 (1-4 bytes). @return byte count, or `sz_rune_invalid_k` if invalid. */
SZ_INTERNAL sz_rune_length_t sz_rune_encode(sz_rune_t rune, sz_u8_t *utf8s) {
    if (rune <= 0x7F) {
        utf8s[0] = (sz_u8_t)rune;
        return sz_rune_1byte_k;
    }
    else if (rune <= 0x7FF) {
        utf8s[0] = (sz_u8_t)(0xC0 | (rune >> 6));
        utf8s[1] = (sz_u8_t)(0x80 | (rune & 0x3F));
        return sz_rune_2bytes_k;
    }
    else if (rune <= 0xFFFF) {
        if (rune >= 0xD800 && rune <= 0xDFFF) return sz_rune_invalid_k; // reject surrogates
        utf8s[0] = (sz_u8_t)(0xE0 | (rune >> 12));
        utf8s[1] = (sz_u8_t)(0x80 | ((rune >> 6) & 0x3F));
        utf8s[2] = (sz_u8_t)(0x80 | (rune & 0x3F));
        return sz_rune_3bytes_k;
    }
    else if (rune <= 0x10FFFF) {
        utf8s[0] = (sz_u8_t)(0xF0 | (rune >> 18));
        utf8s[1] = (sz_u8_t)(0x80 | ((rune >> 12) & 0x3F));
        utf8s[2] = (sz_u8_t)(0x80 | ((rune >> 6) & 0x3F));
        utf8s[3] = (sz_u8_t)(0x80 | (rune & 0x3F));
        return sz_rune_4bytes_k;
    }
    return sz_rune_invalid_k;
}

/** @brief Whether `[text, text+length)` is entirely well-formed UTF-8. */
SZ_PUBLIC sz_bool_t sz_utf8_valid(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u8_t const *end_u8 = text_u8 + length;
    while (text_u8 < end_u8) {
        sz_rune_t rune;
        sz_rune_length_t const consumed = sz_rune_decode((sz_cptr_t)text_u8, (sz_cptr_t)end_u8, &rune);
        if (consumed == sz_rune_invalid_k) return sz_false_k;
        text_u8 += consumed;
    }
    return sz_true_k;
}

/** @brief Whether `[text, end)` is a well-formed but @b truncated multi-byte prefix: a valid lead followed only by
 *         valid (so far) continuation bytes, with fewer bytes present than the lead declares. Such a tail is not
 *         ill-formed - a streaming decoder stops on it and resumes once more bytes arrive, rather than substituting
 *         U+FFFD. Genuinely ill-formed bytes (a bad lead, a malformed present continuation, or an overlong/surrogate/
 *         out-of-range prefix) return false so the caller emits the replacement character. */
SZ_INTERNAL sz_bool_t sz_utf8_incomplete_tail_(sz_cptr_t text, sz_cptr_t end) {
    sz_u8_t const *u = (sz_u8_t const *)text;
    sz_size_t const available = (sz_size_t)((sz_u8_t const *)end - u);
    if (!available) return sz_false_k;
    sz_u8_t const lead = u[0];
    sz_rune_length_t declared;
    if (lead < 0x80) return sz_false_k;
    else if (lead >= 0xC2 && lead < 0xE0) declared = sz_rune_2bytes_k;
    else if (lead >= 0xE0 && lead < 0xF0) declared = sz_rune_3bytes_k;
    else if (lead >= 0xF0 && lead <= 0xF4) declared = sz_rune_4bytes_k;
    else return sz_false_k; // C0/C1, F5..FF, or a lone continuation - ill-formed, not merely truncated
    if (available >= (sz_size_t)declared) return sz_false_k; // all bytes present; `sz_rune_decode` judges validity
    for (sz_size_t index = 1; index < available; ++index)
        if ((u[index] & 0xC0) != 0x80) return sz_false_k;    // a present continuation is malformed - ill-formed now
    if (available >= 2) {                                    // first-continuation range constraints, where present
        if (lead == 0xE0 && u[1] < 0xA0) return sz_false_k;  // overlong
        if (lead == 0xED && u[1] >= 0xA0) return sz_false_k; // surrogate
        if (lead == 0xF0 && u[1] < 0x90) return sz_false_k;  // overlong
        if (lead == 0xF4 && u[1] >= 0x90) return sz_false_k; // > U+10FFFF
    }
    return sz_true_k;
}

#pragma endregion

SZ_PUBLIC sz_size_t sz_utf8_count_serial(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u8_t const *end_u8 = text_u8 + length;
    sz_size_t char_count = 0;

    while (text_u8 < end_u8) {
        // Count this byte if it's NOT a continuation byte
        if ((*text_u8 & 0xC0) != 0x80) char_count++;
        text_u8++;
    }

    return char_count;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_serial(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u8_t const *end_u8 = text_u8 + length;
    sz_size_t char_count = 0;

    while (text_u8 < end_u8) {
        // Check if this is NOT a continuation byte
        if ((*text_u8 & 0xC0) != 0x80) {
            if (char_count == n) return (sz_cptr_t)text_u8;
            char_count++;
        }
        text_u8++;
    }

    // If we reached the end without finding the nth character, return NULL
    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_utf8_decode_serial(      //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {

    sz_cptr_t text_cursor = text;
    sz_cptr_t text_end = text + length;
    sz_size_t runes_written = 0;

    // The reference for the unified contract: decode each codepoint, substitute one U+FFFD per maximal ill-formed
    // subpart (resyncing by the subpart's 1-3 byte length, Unicode 17.0 §3.9 / W3C), and stop on a well-formed but
    // truncated trailing prefix so a streaming caller resumes once more bytes arrive. Every emitted rune is valid.
    while (text_cursor < text_end && runes_written < runes_capacity) {
        if (sz_utf8_incomplete_tail_(text_cursor, text_end)) break; // Resumable truncation - hand back to caller.
        sz_rune_t rune;
        sz_rune_length_t rune_length = sz_rune_decode(text_cursor, text_end, &rune);
        if (rune_length == sz_rune_invalid_k) {
            // One U+FFFD per maximal ill-formed subpart (Unicode 17.0 §3.9 / W3C), consuming 1-3 bytes.
            rune = (sz_rune_t)sz_rune_replacement_k;
            rune_length = (sz_rune_length_t)sz_utf8_maximal_subpart_(text_cursor, text_end);
        }
        runes[runes_written++] = rune;
        text_cursor += rune_length;
    }

    *runes_unpacked = runes_written;
    return text_cursor;
}

/** @brief Decode the UTF-8 codepoint at `*position`, advancing it. Ill-formed bytes (lone continuation, overlong,
 *         surrogate, out-of-range, truncated) decode to U+FFFD consuming one byte — the segmentation-side
 *         substitution the SIMD `sz_utf8_rune_decode_window_` classifiers mirror byte-for-byte. (The
 *         rune-unpack path uses the ICU maximal-subpart resync instead; segmentation keeps the 1-byte contract.) */
SZ_INTERNAL sz_rune_t sz_utf8_next_rune_(sz_cptr_t text, sz_size_t length, sz_size_t *position) {
    if (*position >= length) return 0;
    sz_rune_t rune;
    sz_rune_length_t const consumed = sz_rune_decode(text + *position, text + length, &rune);
    if (consumed == sz_rune_invalid_k) {
        (*position)++;
        return 0xFFFD;
    }
    *position += consumed;
    return rune;
}

/**
 *  @brief Get the UTF-8 sequence length from a lead byte, branchlessly.
 *
 *  The length is fully determined by the lead byte's high nibble: 0x0-0xB map to 1 (ASCII and, for robustness,
 *  stray continuation bytes treated as single bytes), 0xC-0xD to 2, 0xE to 3, 0xF to 4. A single 16-entry
 *  table replaces the four-way `if`-ladder that ran on every codepoint advance.
 */
SZ_INTERNAL sz_size_t sz_utf8_lead_length_(sz_u8_t lead_byte) {
    static sz_u8_t const length_by_high_nibble[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    return length_by_high_nibble[lead_byte >> 4];
}

/** @brief Returns the start offset of the codepoint preceding `position` (a codepoint start), or `position` if none. */
SZ_INTERNAL sz_size_t sz_utf8_previous_rune_start_(sz_cptr_t text, sz_size_t position) {
    if (position == 0) return 0;
    sz_size_t previous = position - 1;
    while (previous > 0 && ((sz_u8_t)text[previous] & 0xC0) == 0x80) previous--;
    return previous;
}

#pragma region Shared bitmask boundary algebra

/*  ISA-independent `sz_u64_t` boundary-mask algebra shared by every backend (serial / haswell / icelake / neon).
 *  It lives here so each `utf8_runes/<isa>.h` inherits it through its `#include "utf8_runes/serial.h"` and
 *  the per-family rule code is written once and reused across ISAs (CONTRIBUTING-KERNELS.md §6.8). */

/** @brief  Smear set bits rightward (toward higher lanes) for @p steps, gated by the @p reach mask each step. */
SZ_INTERNAL sz_u64_t sz_u64_smear_right_(sz_u64_t bits, sz_u64_t reach, int steps) {
    for (int step = 0; step < steps; ++step) bits |= (bits << 1) & reach;
    return bits;
}

/**
 *  @brief  Unbounded segmented flood of @p seed bits rightward (toward higher lanes) through every lane where
 *          @p gate is set, in log-depth Kogge-Stone doubling steps. Reaches the full 64-lane span without a
 *          per-lane loop, so runs of arbitrary length resolve in six iterations (sentence SB8 / line SP-runs).
 *  @note   A bit at lane `i` ends up set when `seed[i]` or there is a contiguous `gate`-true run from some
 *          seeded lane up to and including `i`. The gate is contracted alongside the flood so each doubling
 *          step still only crosses lanes that are themselves gated.
 */
SZ_INTERNAL sz_u64_t sz_u64_fill_right_(sz_u64_t seed, sz_u64_t gate) {
    sz_u64_t bits = seed;
    sz_u64_t reach = gate;
    for (int shift = 1; shift < 64; shift <<= 1) {
        bits |= (bits << shift) & reach;
        reach &= reach << shift;
    }
    return bits;
}

/**
 *  @brief  Unbounded segmented flood of @p seed bits leftward (toward lower lanes) through every lane where
 *          @p gate is set, in log-depth Kogge-Stone doubling steps. Mirror of @ref sz_u64_fill_right_.
 */
SZ_INTERNAL sz_u64_t sz_u64_fill_left_(sz_u64_t seed, sz_u64_t gate) {
    sz_u64_t bits = seed;
    sz_u64_t reach = gate;
    for (int shift = 1; shift < 64; shift <<= 1) {
        bits |= (bits >> shift) & reach;
        reach &= reach >> shift;
    }
    return bits;
}

/**
 *  @brief  Segmented exclusive prefix-XOR parity over each maximal @p gate run: lane i carries the XOR of @p seed
 *          across the contiguous run of @p gate ending at i. The dual of @ref sz_u64_fill_right_ (OR ->
 *          XOR), in log-depth Kogge-Stone doubling. Used by the grapheme GB12/13 Regional_Indicator parity scan.
 */
SZ_INTERNAL sz_u64_t sz_u64_segmented_parity_(sz_u64_t seed, sz_u64_t gate) {
    sz_u64_t parity = seed;
    sz_u64_t reach = gate;
    for (int shift = 1; shift < 64; shift <<= 1) {
        parity ^= (parity << shift) & reach;
        reach &= reach << shift;
    }
    return parity;
}

/** @brief  Low @p count bits set (`[0, count)`), 0 for `count==0`, all-ones for `count>=64`. Portable, branch-light
 *          replacement for the BMI2 `sz_u64_mask_until_` so the shared boundary algebra compiles on every backend. */
SZ_INTERNAL sz_u64_t sz_u64_mask_until_serial_(sz_size_t count) {
    return count >= 64 ? (sz_u64_t) ~(sz_u64_t)0 : (((sz_u64_t)1 << count) - 1);
}

#pragma endregion Shared bitmask boundary algebra

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_RUNES_SERIAL_H_
