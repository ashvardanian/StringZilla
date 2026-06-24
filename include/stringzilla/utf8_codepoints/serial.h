/**
 *  @brief Serial backend for UTF-8 codepoint mechanics (count, find-nth, unpack) and shared decode helpers.
 *  @file include/stringzilla/utf8_codepoints/serial.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_CODEPOINTS_SERIAL_H_
#define STRINGZILLA_UTF8_CODEPOINTS_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes.h" // `sz_rune_parse_unchecked`

#ifdef __cplusplus
extern "C" {
#endif

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

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_serial( //
    sz_cptr_t text, sz_size_t length,            //
    sz_rune_t *runes, sz_size_t runes_capacity,  //
    sz_size_t *runes_unpacked) {

    sz_cptr_t text_cursor = text;
    sz_cptr_t text_end = text + length;
    sz_size_t runes_written = 0;

    // Process up to runes_capacity codepoints or end of input.
    while (text_cursor < text_end && runes_written < runes_capacity) {
        sz_rune_t rune;
        sz_rune_length_t const rune_length = sz_rune_parse_unchecked(text_cursor, &rune);
        if (text_cursor + rune_length > text_end) break; // Incomplete sequence at buffer boundary
        runes[runes_written++] = rune;
        text_cursor += rune_length;
    }

    *runes_unpacked = runes_written;
    return text_cursor;
}

/** @brief Decode the UTF-8 codepoint at `*position`, advancing it. Ill-formed bytes (lone continuation, overlong,
 *         surrogate, out-of-range, truncated) decode to U+FFFD consuming one byte — the canonical replacement-char
 *         substitution that the SIMD `sz_utf8_codepoints_decode_window_` mirrors byte-for-byte. */
SZ_INTERNAL sz_rune_t sz_utf8_decode_(sz_cptr_t text, sz_size_t length, sz_size_t *position) {
    if (*position >= length) return 0;
    sz_rune_t rune;
    sz_rune_length_t const consumed = sz_rune_parse(text + *position, text + length, &rune);
    if (consumed == sz_utf8_invalid_k) {
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
SZ_INTERNAL sz_size_t sz_utf8_codepoint_length_(sz_u8_t lead_byte) {
    static sz_u8_t const length_by_high_nibble[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    return length_by_high_nibble[lead_byte >> 4];
}

/** @brief Returns the start offset of the codepoint preceding `position` (a codepoint start), or `position` if none. */
SZ_INTERNAL sz_size_t sz_utf8_previous_codepoint_start_(sz_cptr_t text, sz_size_t position) {
    if (position == 0) return 0;
    sz_size_t previous = position - 1;
    while (previous > 0 && ((sz_u8_t)text[previous] & 0xC0) == 0x80) previous--;
    return previous;
}

#pragma region Shared bitmask boundary algebra

/*  ISA-independent `sz_u64_t` boundary-mask algebra shared by every backend (serial / haswell / icelake / neon).
 *  It lives here so each `utf8_codepoints/<isa>.h` inherits it through its `#include "utf8_codepoints/serial.h"` and
 *  the per-family rule code is written once and reused across ISAs (CONTRIBUTING-KERNELS.md §6.8). */

/** @brief  Largest index `<= index` at which @p text begins a codepoint (skips trailing continuation bytes
 *          `0x80..0xBF`). Used by the reverse driver to snap an anchor to a codepoint start. */
SZ_INTERNAL sz_size_t sz_utf8_codepoints_start_at_or_before_(sz_u8_t const *text, sz_size_t index) {
    while (index > 0 && (text[index] & 0xC0) == 0x80) --index;
    return index;
}

/** @brief  Smear set bits rightward (toward higher lanes) for @p steps, gated by the @p reach mask each step. */
SZ_INTERNAL sz_u64_t sz_utf8_codepoints_smear_right_(sz_u64_t bits, sz_u64_t reach, int steps) {
    for (int step = 0; step < steps; ++step) bits |= (bits << 1) & reach;
    return bits;
}

/** @brief  Smear set bits leftward (toward lower lanes) for @p steps, gated by the @p reach mask each step. */
SZ_INTERNAL sz_u64_t sz_utf8_codepoints_smear_left_(sz_u64_t bits, sz_u64_t reach, int steps) {
    for (int step = 0; step < steps; ++step) bits |= (bits >> 1) & reach;
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
SZ_INTERNAL sz_u64_t sz_utf8_codepoints_fill_right_(sz_u64_t seed, sz_u64_t gate) {
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
 *          @p gate is set, in log-depth Kogge-Stone doubling steps. Mirror of @ref sz_utf8_codepoints_fill_right_.
 */
SZ_INTERNAL sz_u64_t sz_utf8_codepoints_fill_left_(sz_u64_t seed, sz_u64_t gate) {
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
 *          across the contiguous run of @p gate ending at i. The dual of @ref sz_utf8_codepoints_fill_right_ (OR ->
 *          XOR), in log-depth Kogge-Stone doubling. Used by the grapheme GB12/13 Regional_Indicator parity scan.
 */
SZ_INTERNAL sz_u64_t sz_utf8_codepoints_segmented_parity_(sz_u64_t seed, sz_u64_t gate) {
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
SZ_INTERNAL sz_u64_t sz_utf8_codepoints_mask_until_(sz_size_t count) {
    return count >= 64 ? (sz_u64_t) ~(sz_u64_t)0 : (((sz_u64_t)1 << count) - 1);
}

#pragma endregion Shared bitmask boundary algebra

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_CODEPOINTS_SERIAL_H_
