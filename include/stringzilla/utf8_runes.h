/**
 *  @brief ISA-agnostic Unicode rune codec (decode/encode/validate) and case folding (no SIMD intrinsics).
 *  @file include/stringzilla/utf8_runes.h
 *  @author Ash Vardanian
 *  @sa utf8_uncased.h
 */
#ifndef STRINGZILLA_UTF8_RUNES_H_
#define STRINGZILLA_UTF8_RUNES_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core Types

/**
 *  @brief Lightweight metadata for a safe window within a script path.
 *
 *  This struct only contains location and length information needed for kernel selection.
 *  The actual case-folding and probe computation is deferred until after the best kernel
 *  is chosen, using `sz_utf8_uncased_needle_metadata_t`.
 */
typedef struct sz_utf8_string_slice_t {
    sz_size_t offset;       // Start offset in original needle (bytes)
    sz_size_t length;       // Byte length in original needle
    sz_size_t runes_within; // Codepoints within this window
} sz_utf8_string_slice_t;

/**
 *  @brief Tiny wrapper for substring search queries with pre-located probing positions.
 *
 *  Reuse this structure to avoid re-computing the probe positions for the same needle multiple times.
 *  It's created internally in a multi-step process of:
 *  1. locating the longest "safe" slice of the needle with respect to different SIMD folding kernels,
 *  2. shrinking it further to find the most diverse slice that fits into a `folded_slice` when case-folded.
 *
 *  Unlike the exact substring search kernels, it uses 4 probe positions instead of 3:
 *    - first: implicit at `folded_slice[0]`
 *    - second: `probe_second`
 *    - third: `probe_third`
 *    - last: implicit at `folded_slice[folded_slice_length - 1]`
 */
typedef struct sz_utf8_uncased_needle_metadata_t {
    sz_size_t offset_in_unfolded; // Number of bytes in the "unsafe LONG NeedLe" before the safe & folded part
    sz_size_t length_in_unfolded; // Number of bytes in the safe part of the actual "NeedLe" before folding
    sz_u8_t folded_slice[16];
    sz_u8_t folded_slice_length;
    sz_u8_t probe_second; // Position of the second relevant character in the folded slice
    sz_u8_t probe_third;  // Position of the third relevant character in the folded slice
    sz_u8_t kernel_id;    // The unique identifier of the kernel best suited for searching this needle
} sz_utf8_uncased_needle_metadata_t;

#pragma endregion // Core Types

#pragma region Unicode Core

/**  Helper macro for readable assertions - use for SIMD implementation reference */
#define sz_is_in_range_(x, low, high) ((x) >= (low) && (x) <= (high))

#pragma region Rune Codec

/** @brief Bounds-checked, validating UTF-8 decode. Returns the codepoint's 1-4 byte length in @p rune, or
 *         `sz_utf8_invalid_k` (0) when the bytes at @p utf8 do not begin a well-formed, shortest-form,
 *         non-surrogate, in-range codepoint (a truncated trailing sequence before @p utf8_end included). The
 *         single authority for "is this a foldable/normalizable rune"; the decode-side mirror of `sz_rune_export`.
 *         A byte that does not begin a well-formed rune is its own 1-byte maximal subpart - callers copy it
 *         through unchanged and resync at the next byte. */
SZ_INTERNAL sz_rune_length_t sz_rune_parse(sz_cptr_t utf8, sz_cptr_t utf8_end, sz_rune_t *rune) {
    sz_u8_t const *u = (sz_u8_t const *)utf8;
    sz_size_t const available = (sz_size_t)((sz_u8_t const *)utf8_end - u);
    sz_u8_t const lead = u[0];
    if (lead < 0x80) { *rune = lead; return sz_utf8_rune_1byte_k; }
    if (lead < 0xC2) return sz_utf8_invalid_k; // continuation byte, or C0/C1 (overlong 2-byte)
    if (lead < 0xE0) {                         // C2..DF
        if (available < 2 || (u[1] & 0xC0) != 0x80) return sz_utf8_invalid_k;
        *rune = (sz_rune_t)(lead & 0x1F) << 6 | (u[1] & 0x3F);
        return sz_utf8_rune_2bytes_k;
    }
    if (lead < 0xF0) { // E0..EF
        if (available < 3 || (u[1] & 0xC0) != 0x80 || (u[2] & 0xC0) != 0x80) return sz_utf8_invalid_k;
        if (lead == 0xE0 && u[1] < 0xA0) return sz_utf8_invalid_k;  // overlong
        if (lead == 0xED && u[1] >= 0xA0) return sz_utf8_invalid_k; // surrogate (U+D800-U+DFFF)
        *rune = (sz_rune_t)(lead & 0x0F) << 12 | (sz_rune_t)(u[1] & 0x3F) << 6 | (u[2] & 0x3F);
        return sz_utf8_rune_3bytes_k;
    }
    if (lead <= 0xF4) { // F0..F4
        if (available < 4 || (u[1] & 0xC0) != 0x80 || (u[2] & 0xC0) != 0x80 || (u[3] & 0xC0) != 0x80)
            return sz_utf8_invalid_k;
        if (lead == 0xF0 && u[1] < 0x90) return sz_utf8_invalid_k;  // overlong
        if (lead == 0xF4 && u[1] >= 0x90) return sz_utf8_invalid_k; // > U+10FFFF
        *rune = (sz_rune_t)(lead & 0x07) << 18 | (sz_rune_t)(u[1] & 0x3F) << 12 | (sz_rune_t)(u[2] & 0x3F) << 6 |
                (u[3] & 0x3F);
        return sz_utf8_rune_4bytes_k;
    }
    return sz_utf8_invalid_k; // F5..FF
}

/** @brief Decode the 1-4 byte sequence the lead byte declares, with NO bounds check and NO validation; returns
 *         the byte length and stores the codepoint in @p rune.
 *  @warning Assumes valid, complete UTF-8 (a truncated trailing sequence over-reads). Use `sz_rune_parse` for the
 *           bounds-checked + validating variant, or `sz_utf8_valid()` first. */
SZ_INTERNAL sz_rune_length_t sz_rune_parse_unchecked(sz_cptr_t utf8, sz_rune_t *rune) {
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

/** @brief Encode a UTF-32 codepoint to UTF-8 (1-4 bytes). @return byte count, or `sz_utf8_invalid_k` if invalid. */
SZ_INTERNAL sz_rune_length_t sz_rune_export(sz_rune_t rune, sz_u8_t *utf8s) {
    if (rune <= 0x7F) {
        utf8s[0] = (sz_u8_t)rune;
        return sz_utf8_rune_1byte_k;
    }
    else if (rune <= 0x7FF) {
        utf8s[0] = (sz_u8_t)(0xC0 | (rune >> 6));
        utf8s[1] = (sz_u8_t)(0x80 | (rune & 0x3F));
        return sz_utf8_rune_2bytes_k;
    }
    else if (rune <= 0xFFFF) {
        if (rune >= 0xD800 && rune <= 0xDFFF) return sz_utf8_invalid_k; // reject surrogates
        utf8s[0] = (sz_u8_t)(0xE0 | (rune >> 12));
        utf8s[1] = (sz_u8_t)(0x80 | ((rune >> 6) & 0x3F));
        utf8s[2] = (sz_u8_t)(0x80 | (rune & 0x3F));
        return sz_utf8_rune_3bytes_k;
    }
    else if (rune <= 0x10FFFF) {
        utf8s[0] = (sz_u8_t)(0xF0 | (rune >> 18));
        utf8s[1] = (sz_u8_t)(0x80 | ((rune >> 12) & 0x3F));
        utf8s[2] = (sz_u8_t)(0x80 | ((rune >> 6) & 0x3F));
        utf8s[3] = (sz_u8_t)(0x80 | (rune & 0x3F));
        return sz_utf8_rune_4bytes_k;
    }
    return sz_utf8_invalid_k;
}

/** @brief Whether `[text, text+length)` is entirely well-formed UTF-8. */
SZ_PUBLIC sz_bool_t sz_utf8_valid(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u8_t const *end_u8 = text_u8 + length;
    while (text_u8 < end_u8) {
        sz_rune_t rune;
        sz_rune_length_t const consumed = sz_rune_parse((sz_cptr_t)text_u8, (sz_cptr_t)end_u8, &rune);
        if (consumed == sz_utf8_invalid_k) return sz_false_k;
        text_u8 += consumed;
    }
    return sz_true_k;
}

#pragma endregion

/**
 *  @brief Fold a Unicode codepoint to its case-folded form (Unicode 17.0).
 *
 *  Optimization strategy:
 *  - Single-comparison range checks: `(sz_u32_t)(rune - base) <= size` instead of two comparisons
 *  - Combined upper+lower ranges: check both cases, apply offset only for uppercase (branchless)
 *  - Combined even/odd ranges: check full range, apply +1 only for uppercase parity
 *  - Hierarchical by UTF-8 byte width for early exit on common cases
 *  - Per-section switches for irregular mappings (better compiler optimization)
 *
 *  Each range check includes an assertion with traditional bounds for SIMD implementation reference.
 */
SZ_INTERNAL sz_size_t sz_unicode_fold_codepoint_(sz_rune_t rune, sz_rune_t *folded) {
    // clang-format off

    // 1-byte UTF-8 (U+0000-007F): ASCII - only A-Z needs folding
    if (rune <= 0x7F) {
        if ((sz_u32_t)(rune - 0x41) <= 25) { // A-Z: 0x41-0x5A (26 chars)
            sz_assert_(sz_is_in_range_(rune, 0x0041, 0x005A));
            folded[0] = rune + 0x20; return 1; }
        folded[0] = rune; return 1;  // digits, punctuation, control chars unchanged
    }

    // 2-byte UTF-8 (U+0080-07FF): Latin, Greek, Cyrillic, Armenian
    if (rune <= 0x7FF) {
        // Cyrillic А-я: 0x0410-0x044F (upper 0x0410-0x042F, lower 0x0430-0x044F)
        if ((sz_u32_t)(rune - 0x0410) <= 0x3F) {
            sz_assert_(sz_is_in_range_(rune, 0x0410, 0x044F));
            folded[0] = rune + ((rune <= 0x042F) * 0x20); return 1; } // +32 if upper, +0 if lower

        // Latin-1 À-þ: 0x00C0-0x00FE (upper 0x00C0-0x00DE, lower 0x00E0-0x00FE)
        if ((sz_u32_t)(rune - 0x00C0) <= 0x3E) {
            sz_assert_(sz_is_in_range_(rune, 0x00C0, 0x00FE));
            if ((rune | 0x20) == 0xF7) { folded[0] = rune; return 1; } // × (D7) and ÷ (F7) unchanged
            // 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
            if (rune == 0x00DF) { folded[0] = 0x0073; folded[1] = 0x0073; return 2; }
            folded[0] = rune + ((rune <= 0x00DE) * 0x20); return 1; }

        // Greek Α-Ρ: 0x0391-0x03A1 → α-ρ (+32)
        if ((sz_u32_t)(rune - 0x0391) <= 0x10) {
            sz_assert_(sz_is_in_range_(rune, 0x0391, 0x03A1));
            folded[0] = rune + 0x20; return 1; }

        // Greek Σ-Ϋ: 0x03A3-0x03AB → σ-ϋ (+32)
        if ((sz_u32_t)(rune - 0x03A3) <= 0x08) {
            sz_assert_(sz_is_in_range_(rune, 0x03A3, 0x03AB));
            folded[0] = rune + 0x20; return 1; }

        // Cyrillic Ѐ-Џ: 0x0400-0x040F → ѐ-џ (+80)
        if ((sz_u32_t)(rune - 0x0400) <= 0x0F) {
            sz_assert_(sz_is_in_range_(rune, 0x0400, 0x040F));
            folded[0] = rune + 0x50; return 1; }

        // Armenian Ա-Ֆ: 0x0531-0x0556 → ա-ֆ (+48)
        if ((sz_u32_t)(rune - 0x0531) <= 0x25) {
            sz_assert_(sz_is_in_range_(rune, 0x0531, 0x0556));
            folded[0] = rune + 0x30; return 1; }

        // Greek Έ-Ί: 0x0388-0x038A (+37)
        if ((sz_u32_t)(rune - 0x0388) <= 0x02) {
            sz_assert_(sz_is_in_range_(rune, 0x0388, 0x038A));
            folded[0] = rune + 0x25; return 1; }

        // Greek Ͻ-Ͽ: 0x03FD-0x03FF → ͻ-Ϳ (-130)
        if ((sz_u32_t)(rune - 0x03FD) <= 0x02) {
            sz_assert_(sz_is_in_range_(rune, 0x03FD, 0x03FF));
            folded[0] = rune - 130; return 1; }

        // Next let's handle the even/odd parity-based ranges
        sz_u32_t is_even = ((rune & 1) == 0);

        // Latin Extended-A: Ā-Į (0x0100-0x012E, even → +1)
        if ((sz_u32_t)(rune - 0x0100) <= 0x2E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0100, 0x012E));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-A: Ĳ-Ķ (0x0132-0x0136, even → +1)
        if ((sz_u32_t)(rune - 0x0132) <= 0x04 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0132, 0x0136));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-A: Ĺ-Ň (0x0139-0x0147, odd → +1)
        if ((sz_u32_t)(rune - 0x0139) <= 0x0E && !is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0139, 0x0147));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-A: Ŋ-Ŷ (0x014A-0x0176, even → +1)
        if ((sz_u32_t)(rune - 0x014A) <= 0x2C && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x014A, 0x0176));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-A: Ź-Ž (0x0179-0x017D, odd → +1)
        if ((sz_u32_t)(rune - 0x0179) <= 0x04 && !is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0179, 0x017D));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-B: Ǎ-Ǜ (0x01CD-0x01DB, odd → +1)
        if ((sz_u32_t)(rune - 0x01CD) <= 0x0E && !is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x01CD, 0x01DB));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-B: Ǟ-Ǯ (0x01DE-0x01EE, even → +1)
        if ((sz_u32_t)(rune - 0x01DE) <= 0x10 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x01DE, 0x01EE));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-B: Ǹ-Ǿ (0x01F8-0x01FE, even → +1)
        if ((sz_u32_t)(rune - 0x01F8) <= 0x06 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x01F8, 0x01FE));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-B: Ȁ-Ȟ (0x0200-0x021E, even → +1)
        if ((sz_u32_t)(rune - 0x0200) <= 0x1E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0200, 0x021E));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-B: Ȣ-Ȳ (0x0222-0x0232, even → +1)
        if ((sz_u32_t)(rune - 0x0222) <= 0x10 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0222, 0x0232));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-B: Ɇ-Ɏ (0x0246-0x024E, even → +1)
        if ((sz_u32_t)(rune - 0x0246) <= 0x08 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0246, 0x024E));
            folded[0] = rune + 1; return 1; }

        // Greek archaic: Ͱ-Ͳ (0x0370-0x0372, even → +1)
        if ((sz_u32_t)(rune - 0x0370) <= 0x02 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0370, 0x0372));
            folded[0] = rune + 1; return 1; }

        // Greek archaic: Ϙ-Ϯ (0x03D8-0x03EE, even → +1)
        if ((sz_u32_t)(rune - 0x03D8) <= 0x16 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x03D8, 0x03EE));
            folded[0] = rune + 1; return 1; }

        // Cyrillic extended: Ѡ-Ҁ (0x0460-0x0480, even → +1)
        if ((sz_u32_t)(rune - 0x0460) <= 0x20 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0460, 0x0480));
            folded[0] = rune + 1; return 1; }

        // Cyrillic extended: Ҋ-Ҿ (0x048A-0x04BE, even → +1)
        if ((sz_u32_t)(rune - 0x048A) <= 0x34 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x048A, 0x04BE));
            folded[0] = rune + 1; return 1; }

        // Cyrillic extended: Ӂ-Ӎ (0x04C1-0x04CD, odd → +1)
        if ((sz_u32_t)(rune - 0x04C1) <= 0x0C && !is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x04C1, 0x04CD));
            folded[0] = rune + 1; return 1; }

        // Cyrillic extended: Ӑ-Ӿ (0x04D0-0x04FE, even → +1)
        if ((sz_u32_t)(rune - 0x04D0) <= 0x2E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x04D0, 0x04FE));
            folded[0] = rune + 1; return 1; }

        // Cyrillic extended: Ԁ-Ԯ (0x0500-0x052E, even → +1)
        if ((sz_u32_t)(rune - 0x0500) <= 0x2E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0500, 0x052E));
            folded[0] = rune + 1; return 1; }

        // Next let's handle the 2-byte irregular one-to-one mappings
        switch (rune) {
        // Latin-1 Supplement specials
        case 0x00B5: folded[0] = 0x03BC; return 1; // 'µ' (U+00B5, C2 B5) → 'μ' (U+03BC, CE BC)
        case 0x0178: folded[0] = 0x00FF; return 1; // 'Ÿ' (U+0178, C5 B8) → 'ÿ' (U+00FF, C3 BF)
        case 0x017F: folded[0] = 0x0073; return 1; // 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
        // Latin Extended-B: African/IPA letters (0x0181-0x01BF)
        case 0x0181: folded[0] = 0x0253; return 1; // 'Ɓ' (U+0181, C6 81) → 'ɓ' (U+0253, C9 93)
        case 0x0182: folded[0] = 0x0183; return 1; // 'Ƃ' (U+0182, C6 82) → 'ƃ' (U+0183, C6 83)
        case 0x0184: folded[0] = 0x0185; return 1; // 'Ƅ' (U+0184, C6 84) → 'ƅ' (U+0185, C6 85)
        case 0x0186: folded[0] = 0x0254; return 1; // 'Ɔ' (U+0186, C6 86) → 'ɔ' (U+0254, C9 94)
        case 0x0187: folded[0] = 0x0188; return 1; // 'Ƈ' (U+0187, C6 87) → 'ƈ' (U+0188, C6 88)
        case 0x0189: folded[0] = 0x0256; return 1; // 'Ɖ' (U+0189, C6 89) → 'ɖ' (U+0256, C9 96)
        case 0x018A: folded[0] = 0x0257; return 1; // 'Ɗ' (U+018A, C6 8A) → 'ɗ' (U+0257, C9 97)
        case 0x018B: folded[0] = 0x018C; return 1; // 'Ƌ' (U+018B, C6 8B) → 'ƌ' (U+018C, C6 8C)
        case 0x018E: folded[0] = 0x01DD; return 1; // 'Ǝ' (U+018E, C6 8E) → 'ǝ' (U+01DD, C7 9D)
        case 0x018F: folded[0] = 0x0259; return 1; // 'Ə' (U+018F, C6 8F) → 'ə' (U+0259, C9 99)
        case 0x0190: folded[0] = 0x025B; return 1; // 'Ɛ' (U+0190, C6 90) → 'ɛ' (U+025B, C9 9B)
        case 0x0191: folded[0] = 0x0192; return 1; // 'Ƒ' (U+0191, C6 91) → 'ƒ' (U+0192, C6 92)
        case 0x0193: folded[0] = 0x0260; return 1; // 'Ɠ' (U+0193, C6 93) → 'ɠ' (U+0260, C9 A0)
        case 0x0194: folded[0] = 0x0263; return 1; // 'Ɣ' (U+0194, C6 94) → 'ɣ' (U+0263, C9 A3)
        case 0x0196: folded[0] = 0x0269; return 1; // 'Ɩ' (U+0196, C6 96) → 'ɩ' (U+0269, C9 A9)
        case 0x0197: folded[0] = 0x0268; return 1; // 'Ɨ' (U+0197, C6 97) → 'ɨ' (U+0268, C9 A8)
        case 0x0198: folded[0] = 0x0199; return 1; // 'Ƙ' (U+0198, C6 98) → 'ƙ' (U+0199, C6 99)
        case 0x019C: folded[0] = 0x026F; return 1; // 'Ɯ' (U+019C, C6 9C) → 'ɯ' (U+026F, C9 AF)
        case 0x019D: folded[0] = 0x0272; return 1; // 'Ɲ' (U+019D, C6 9D) → 'ɲ' (U+0272, C9 B2)
        case 0x019F: folded[0] = 0x0275; return 1; // 'Ɵ' (U+019F, C6 9F) → 'ɵ' (U+0275, C9 B5)
        case 0x01A0: folded[0] = 0x01A1; return 1; // 'Ơ' (U+01A0, C6 A0) → 'ơ' (U+01A1, C6 A1)
        case 0x01A2: folded[0] = 0x01A3; return 1; // 'Ƣ' (U+01A2, C6 A2) → 'ƣ' (U+01A3, C6 A3)
        case 0x01A4: folded[0] = 0x01A5; return 1; // 'Ƥ' (U+01A4, C6 A4) → 'ƥ' (U+01A5, C6 A5)
        case 0x01A6: folded[0] = 0x0280; return 1; // 'Ʀ' (U+01A6, C6 A6) → 'ʀ' (U+0280, CA 80)
        case 0x01A7: folded[0] = 0x01A8; return 1; // 'Ƨ' (U+01A7, C6 A7) → 'ƨ' (U+01A8, C6 A8)
        case 0x01A9: folded[0] = 0x0283; return 1; // 'Ʃ' (U+01A9, C6 A9) → 'ʃ' (U+0283, CA 83)
        case 0x01AC: folded[0] = 0x01AD; return 1; // 'Ƭ' (U+01AC, C6 AC) → 'ƭ' (U+01AD, C6 AD)
        case 0x01AE: folded[0] = 0x0288; return 1; // 'Ʈ' (U+01AE, C6 AE) → 'ʈ' (U+0288, CA 88)
        case 0x01AF: folded[0] = 0x01B0; return 1; // 'Ư' (U+01AF, C6 AF) → 'ư' (U+01B0, C6 B0)
        case 0x01B1: folded[0] = 0x028A; return 1; // 'Ʊ' (U+01B1, C6 B1) → 'ʊ' (U+028A, CA 8A)
        case 0x01B2: folded[0] = 0x028B; return 1; // 'Ʋ' (U+01B2, C6 B2) → 'ʋ' (U+028B, CA 8B)
        case 0x01B3: folded[0] = 0x01B4; return 1; // 'Ƴ' (U+01B3, C6 B3) → 'ƴ' (U+01B4, C6 B4)
        case 0x01B5: folded[0] = 0x01B6; return 1; // 'Ƶ' (U+01B5, C6 B5) → 'ƶ' (U+01B6, C6 B6)
        case 0x01B7: folded[0] = 0x0292; return 1; // 'Ʒ' (U+01B7, C6 B7) → 'ʒ' (U+0292, CA 92)
        case 0x01B8: folded[0] = 0x01B9; return 1; // 'Ƹ' (U+01B8, C6 B8) → 'ƹ' (U+01B9, C6 B9)
        case 0x01BC: folded[0] = 0x01BD; return 1; // 'Ƽ' (U+01BC, C6 BC) → 'ƽ' (U+01BD, C6 BD)
        // Digraphs: Serbian/Croatian DŽ, LJ, NJ and DZ
        case 0x01C4: folded[0] = 0x01C6; return 1; // 'Ǆ' (U+01C4, C7 84) → 'ǆ' (U+01C6, C7 86)
        case 0x01C5: folded[0] = 0x01C6; return 1; // 'ǅ' (U+01C5, C7 85) → 'ǆ' (U+01C6, C7 86)
        case 0x01C7: folded[0] = 0x01C9; return 1; // 'Ǉ' (U+01C7, C7 87) → 'ǉ' (U+01C9, C7 89)
        case 0x01C8: folded[0] = 0x01C9; return 1; // 'ǈ' (U+01C8, C7 88) → 'ǉ' (U+01C9, C7 89)
        case 0x01CA: folded[0] = 0x01CC; return 1; // 'Ǌ' (U+01CA, C7 8A) → 'ǌ' (U+01CC, C7 8C)
        case 0x01CB: folded[0] = 0x01CC; return 1; // 'ǋ' (U+01CB, C7 8B) → 'ǌ' (U+01CC, C7 8C)
        case 0x01F1: folded[0] = 0x01F3; return 1; // 'Ǳ' (U+01F1, C7 B1) → 'ǳ' (U+01F3, C7 B3)
        case 0x01F2: folded[0] = 0x01F3; return 1; // 'ǲ' (U+01F2, C7 B2) → 'ǳ' (U+01F3, C7 B3)
        // Latin Extended-B: isolated irregulars
        case 0x01F4: folded[0] = 0x01F5; return 1; // 'Ǵ' (U+01F4, C7 B4) → 'ǵ' (U+01F5, C7 B5)
        case 0x01F6: folded[0] = 0x0195; return 1; // 'Ƕ' (U+01F6, C7 B6) → 'ƕ' (U+0195, C6 95)
        case 0x01F7: folded[0] = 0x01BF; return 1; // 'Ƿ' (U+01F7, C7 B7) → 'ƿ' (U+01BF, C6 BF)
        case 0x0220: folded[0] = 0x019E; return 1; // 'Ƞ' (U+0220, C8 A0) → 'ƞ' (U+019E, C6 9E)
        case 0x023A: folded[0] = 0x2C65; return 1; // 'Ⱥ' (U+023A, C8 BA) → 'ⱥ' (U+2C65, E2 B1 A5)
        case 0x023B: folded[0] = 0x023C; return 1; // 'Ȼ' (U+023B, C8 BB) → 'ȼ' (U+023C, C8 BC)
        case 0x023D: folded[0] = 0x019A; return 1; // 'Ƚ' (U+023D, C8 BD) → 'ƚ' (U+019A, C6 9A)
        case 0x023E: folded[0] = 0x2C66; return 1; // 'Ⱦ' (U+023E, C8 BE) → 'ⱦ' (U+2C66, E2 B1 A6)
        case 0x0241: folded[0] = 0x0242; return 1; // 'Ɂ' (U+0241, C9 81) → 'ɂ' (U+0242, C9 82)
        case 0x0243: folded[0] = 0x0180; return 1; // 'Ƀ' (U+0243, C9 83) → 'ƀ' (U+0180, C6 80)
        case 0x0244: folded[0] = 0x0289; return 1; // 'Ʉ' (U+0244, C9 84) → 'ʉ' (U+0289, CA 89)
        case 0x0245: folded[0] = 0x028C; return 1; // 'Ʌ' (U+0245, C9 85) → 'ʌ' (U+028C, CA 8C)
        // Greek: combining iota, accented vowels, variant forms
        case 0x0345: folded[0] = 0x03B9; return 1; // 'ͅ' (U+0345, CD 85) → 'ι' (U+03B9, CE B9)
        case 0x0376: folded[0] = 0x0377; return 1; // 'Ͷ' (U+0376, CD B6) → 'ͷ' (U+0377, CD B7)
        case 0x037F: folded[0] = 0x03F3; return 1; // 'Ϳ' (U+037F, CD BF) → 'ϳ' (U+03F3, CF B3)
        case 0x0386: folded[0] = 0x03AC; return 1; // 'Ά' (U+0386, CE 86) → 'ά' (U+03AC, CE AC)
        case 0x038C: folded[0] = 0x03CC; return 1; // 'Ό' (U+038C, CE 8C) → 'ό' (U+03CC, CF 8C)
        case 0x038E: folded[0] = 0x03CD; return 1; // 'Ύ' (U+038E, CE 8E) → 'ύ' (U+03CD, CF 8D)
        case 0x038F: folded[0] = 0x03CE; return 1; // 'Ώ' (U+038F, CE 8F) → 'ώ' (U+03CE, CF 8E)
        case 0x03C2: folded[0] = 0x03C3; return 1; // 'ς' (U+03C2, CF 82) → 'σ' (U+03C3, CF 83)
        case 0x03CF: folded[0] = 0x03D7; return 1; // 'Ϗ' (U+03CF, CF 8F) → 'ϗ' (U+03D7, CF 97)
        case 0x03D0: folded[0] = 0x03B2; return 1; // 'ϐ' (U+03D0, CF 90) → 'β' (U+03B2, CE B2)
        case 0x03D1: folded[0] = 0x03B8; return 1; // 'ϑ' (U+03D1, CF 91) → 'θ' (U+03B8, CE B8)
        case 0x03D5: folded[0] = 0x03C6; return 1; // 'ϕ' (U+03D5, CF 95) → 'φ' (U+03C6, CF 86)
        case 0x03D6: folded[0] = 0x03C0; return 1; // 'ϖ' (U+03D6, CF 96) → 'π' (U+03C0, CF 80)
        case 0x03F0: folded[0] = 0x03BA; return 1; // 'ϰ' (U+03F0, CF B0) → 'κ' (U+03BA, CE BA)
        case 0x03F1: folded[0] = 0x03C1; return 1; // 'ϱ' (U+03F1, CF B1) → 'ρ' (U+03C1, CF 81)
        case 0x03F4: folded[0] = 0x03B8; return 1; // 'ϴ' (U+03F4, CF B4) → 'θ' (U+03B8, CE B8)
        case 0x03F5: folded[0] = 0x03B5; return 1; // 'ϵ' (U+03F5, CF B5) → 'ε' (U+03B5, CE B5)
        case 0x03F7: folded[0] = 0x03F8; return 1; // 'Ϸ' (U+03F7, CF B7) → 'ϸ' (U+03F8, CF B8)
        case 0x03F9: folded[0] = 0x03F2; return 1; // 'Ϲ' (U+03F9, CF B9) → 'ϲ' (U+03F2, CF B2)
        case 0x03FA: folded[0] = 0x03FB; return 1; // 'Ϻ' (U+03FA, CF BA) → 'ϻ' (U+03FB, CF BB)
        // Cyrillic: palochka
        case 0x04C0: folded[0] = 0x04CF; return 1; // 'Ӏ' (U+04C0, D3 80) → 'ӏ' (U+04CF, D3 8F)
        }

        // 2-byte one-to-many expansions
        switch (rune) {
        // ß handled inline in Latin-1 range above... interestingly the capital Eszett is in the 3-byte range!
        // case 0x00DF: folded[0] = 0x0073; folded[1] = 0x0073; return 2;
        
        // 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
        case 0x0130: folded[0] = 0x0069; folded[1] = 0x0307; return 2;
        // 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
        case 0x0149: folded[0] = 0x02BC; folded[1] = 0x006E; return 2;
        // 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C)
        case 0x01F0: folded[0] = 0x006A; folded[1] = 0x030C; return 2;
        // 'ΐ' (U+0390, CE 90) → "ΐ" (U+03B9 U+0308 U+0301, CE B9 CC 88 CC 81)
        case 0x0390: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0301; return 3;
        // 'ΰ' (U+03B0, CE B0) → "ΰ" (U+03C5 U+0308 U+0301, CF 85 CC 88 CC 81)
        case 0x03B0: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0301; return 3;
        // 'և' (U+0587, D6 87) → "եւ" (U+0565 U+0582, D5 A5 D6 82)
        case 0x0587: folded[0] = 0x0565; folded[1] = 0x0582; return 2;
        }

        folded[0] = rune; return 1;  // 2-byte: no folding needed
    }

    // 3-byte UTF-8 (U+0800-FFFF): Georgian, Cherokee, Greek Extended, etc.
    if (rune <= 0xFFFF) {
        // Georgian Ⴀ-Ⴥ: 0x10A0-0x10C5 (+7264)
        if ((sz_u32_t)(rune - 0x10A0) <= 0x25) {
            sz_assert_(sz_is_in_range_(rune, 0x10A0, 0x10C5));
            folded[0] = rune + 0x1C60; return 1; }

        // Georgian Mtavruli Ა-Ჺ: 0x1C90-0x1CBA (-3008)
        if ((sz_u32_t)(rune - 0x1C90) <= 0x2A) {
            sz_assert_(sz_is_in_range_(rune, 0x1C90, 0x1CBA));
            folded[0] = rune - 0xBC0; return 1; }

        // Georgian Mtavruli Ჽ-Ჿ: 0x1CBD-0x1CBF (-3008)
        if ((sz_u32_t)(rune - 0x1CBD) <= 0x02) {
            sz_assert_(sz_is_in_range_(rune, 0x1CBD, 0x1CBF));
            folded[0] = rune - 0xBC0; return 1; }

        // Cherokee Ᏸ-Ᏽ: 0x13F8-0x13FD (-8)
        if ((sz_u32_t)(rune - 0x13F8) <= 0x05) {
            sz_assert_(sz_is_in_range_(rune, 0x13F8, 0x13FD));
            folded[0] = rune - 8; return 1; }

        // Cherokee Ꭰ-Ᏼ: 0xAB70-0xABBF → Ꭰ-Ᏼ: 0x13A0-0x13EF (-38864)
        if ((sz_u32_t)(rune - 0xAB70) <= 0x4F) {
            sz_assert_(sz_is_in_range_(rune, 0xAB70, 0xABBF));
            folded[0] = rune - 0x97D0; return 1; }

        // Greek Extended: multiple -8 offset ranges
        if ((sz_u32_t)(rune - 0x1F08) <= 0x07) { // Ἀ-Ἇ
            sz_assert_(sz_is_in_range_(rune, 0x1F08, 0x1F0F));
            folded[0] = rune - 8; return 1; }
        if ((sz_u32_t)(rune - 0x1F18) <= 0x05) { // Ἐ-Ἕ
            sz_assert_(sz_is_in_range_(rune, 0x1F18, 0x1F1D));
            folded[0] = rune - 8; return 1; }
        if ((sz_u32_t)(rune - 0x1F28) <= 0x07) { // Ἠ-Ἧ
            sz_assert_(sz_is_in_range_(rune, 0x1F28, 0x1F2F));
            folded[0] = rune - 8; return 1; }
        if ((sz_u32_t)(rune - 0x1F38) <= 0x07) { // Ἰ-Ἷ
            sz_assert_(sz_is_in_range_(rune, 0x1F38, 0x1F3F));
            folded[0] = rune - 8; return 1; }
        if ((sz_u32_t)(rune - 0x1F48) <= 0x05) { // Ὀ-Ὅ
            sz_assert_(sz_is_in_range_(rune, 0x1F48, 0x1F4D));
            folded[0] = rune - 8; return 1; }
        if ((sz_u32_t)(rune - 0x1F68) <= 0x07) { // Ὠ-Ὧ
            sz_assert_(sz_is_in_range_(rune, 0x1F68, 0x1F6F));
            folded[0] = rune - 8; return 1; }

        // Greek Extended Ὲ-Ή: 0x1FC8-0x1FCB (-86)
        if ((sz_u32_t)(rune - 0x1FC8) <= 0x03) {
            sz_assert_(sz_is_in_range_(rune, 0x1FC8, 0x1FCB));
            folded[0] = rune - 86; return 1; }

        // Roman numerals Ⅰ-Ⅿ: 0x2160-0x216F (+16)
        if ((sz_u32_t)(rune - 0x2160) <= 0x0F) {
            sz_assert_(sz_is_in_range_(rune, 0x2160, 0x216F));
            folded[0] = rune + 0x10; return 1; }

        // Circled letters Ⓐ-Ⓩ: 0x24B6-0x24CF (+26)
        if ((sz_u32_t)(rune - 0x24B6) <= 0x19) {
            sz_assert_(sz_is_in_range_(rune, 0x24B6, 0x24CF));
            folded[0] = rune + 0x1A; return 1; }

        // Glagolitic Ⰰ-Ⱟ: 0x2C00-0x2C2F (+48)
        if ((sz_u32_t)(rune - 0x2C00) <= 0x2F) {
            sz_assert_(sz_is_in_range_(rune, 0x2C00, 0x2C2F));
            folded[0] = rune + 0x30; return 1; }

        // Fullwidth Ａ-Ｚ: 0xFF21-0xFF3A (+32)
        if ((sz_u32_t)(rune - 0xFF21) <= 0x19) {
            sz_assert_(sz_is_in_range_(rune, 0xFF21, 0xFF3A));
            folded[0] = rune + 0x20; return 1; }

        // Next let's handle the even/odd parity-based ranges
        sz_u32_t is_even = ((rune & 1) == 0);

        // Latin Extended Additional Ḁ-Ẕ: 0x1E00-0x1E94
        if ((sz_u32_t)(rune - 0x1E00) <= 0x94 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x1E00, 0x1E94));
            folded[0] = rune + 1; return 1; }

        // Latin Extended Additional (Vietnamese) Ạ-Ỿ: 0x1EA0-0x1EFE
        if ((sz_u32_t)(rune - 0x1EA0) <= 0x5E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x1EA0, 0x1EFE));
            folded[0] = rune + 1; return 1; }

        // Coptic Ⲁ-Ⳣ: 0x2C80-0x2CE2
        if ((sz_u32_t)(rune - 0x2C80) <= 0x62 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x2C80, 0x2CE2));
            folded[0] = rune + 1; return 1; }

        // Cyrillic Extended-B Ꙁ-Ꙭ: 0xA640-0xA66C
        if ((sz_u32_t)(rune - 0xA640) <= 0x2C && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0xA640, 0xA66C));
            folded[0] = rune + 1; return 1; }

        // Cyrillic Extended-B Ꚁ-Ꚛ: 0xA680-0xA69A
        if ((sz_u32_t)(rune - 0xA680) <= 0x1A && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0xA680, 0xA69A));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-D ranges
        if ((sz_u32_t)(rune - 0xA722) <= 0x0C && is_even) { // Ꜣ-Ꜯ
            sz_assert_(sz_is_in_range_(rune, 0xA722, 0xA72E));
            folded[0] = rune + 1; return 1; }
        if ((sz_u32_t)(rune - 0xA732) <= 0x3C && is_even) { // Ꜳ-Ꝯ
            sz_assert_(sz_is_in_range_(rune, 0xA732, 0xA76E));
            folded[0] = rune + 1; return 1; }
        if ((sz_u32_t)(rune - 0xA77E) <= 0x08 && is_even) { // Ꝿ-Ꞇ
            sz_assert_(sz_is_in_range_(rune, 0xA77E, 0xA786));
            folded[0] = rune + 1; return 1; }
        if ((sz_u32_t)(rune - 0xA790) <= 0x02 && is_even) { // Ꞑ-Ꞓ
            sz_assert_(sz_is_in_range_(rune, 0xA790, 0xA792));
            folded[0] = rune + 1; return 1; }
        if ((sz_u32_t)(rune - 0xA796) <= 0x12 && is_even) { // Ꞗ-Ꞩ
            sz_assert_(sz_is_in_range_(rune, 0xA796, 0xA7A8));
            folded[0] = rune + 1; return 1; }
        if ((sz_u32_t)(rune - 0xA7B4) <= 0x0E && is_even) { // Ꞵ-Ꟃ
            sz_assert_(sz_is_in_range_(rune, 0xA7B4, 0xA7C2));
            folded[0] = rune + 1; return 1; }

        // Next let's handle the 3-byte irregular one-to-one mappings
        switch (rune) {
        // Georgian irregular
        case 0x10C7: folded[0] = 0x2D27; return 1; // 'Ⴧ' (U+10C7, E1 83 87) → 'ⴧ' (U+2D27, E2 B4 A7)
        case 0x10CD: folded[0] = 0x2D2D; return 1; // 'Ⴭ' (U+10CD, E1 83 8D) → 'ⴭ' (U+2D2D, E2 B4 AD)
        // Cyrillic Extended-C: Old Slavonic variant forms
        case 0x1C80: folded[0] = 0x0432; return 1; // 'ᲀ' (U+1C80, E1 B2 80) → 'в' (U+0432, D0 B2)
        case 0x1C81: folded[0] = 0x0434; return 1; // 'ᲁ' (U+1C81, E1 B2 81) → 'д' (U+0434, D0 B4)
        case 0x1C82: folded[0] = 0x043E; return 1; // 'ᲂ' (U+1C82, E1 B2 82) → 'о' (U+043E, D0 BE)
        case 0x1C83: folded[0] = 0x0441; return 1; // 'ᲃ' (U+1C83, E1 B2 83) → 'с' (U+0441, D1 81)
        case 0x1C84: folded[0] = 0x0442; return 1; // 'ᲄ' (U+1C84, E1 B2 84) → 'т' (U+0442, D1 82)
        case 0x1C85: folded[0] = 0x0442; return 1; // 'ᲅ' (U+1C85, E1 B2 85) → 'т' (U+0442, D1 82)
        case 0x1C86: folded[0] = 0x044A; return 1; // 'ᲆ' (U+1C86, E1 B2 86) → 'ъ' (U+044A, D1 8A)
        case 0x1C87: folded[0] = 0x0463; return 1; // 'ᲇ' (U+1C87, E1 B2 87) → 'ѣ' (U+0463, D1 A3)
        case 0x1C88: folded[0] = 0xA64B; return 1; // 'ᲈ' (U+1C88, E1 B2 88) → 'ꙋ' (U+A64B, EA 99 8B)
        case 0x1C89: folded[0] = 0x1C8A; return 1; // 'Ᲊ' (U+1C89, E1 B2 89) → 'ᲊ' (U+1C8A, E1 B2 8A)
        // Latin Extended Additional: long s with dot
        case 0x1E9B: folded[0] = 0x1E61; return 1; // 'ẛ' (U+1E9B, E1 BA 9B) → 'ṡ' (U+1E61, E1 B9 A1)
        // Greek Extended: vowels with breathing marks (irregular offsets)
        case 0x1F59: folded[0] = 0x1F51; return 1; // 'Ὑ' (U+1F59, E1 BD 99) → 'ὑ' (U+1F51, E1 BD 91)
        case 0x1F5B: folded[0] = 0x1F53; return 1; // 'Ὓ' (U+1F5B, E1 BD 9B) → 'ὓ' (U+1F53, E1 BD 93)
        case 0x1F5D: folded[0] = 0x1F55; return 1; // 'Ὕ' (U+1F5D, E1 BD 9D) → 'ὕ' (U+1F55, E1 BD 95)
        case 0x1F5F: folded[0] = 0x1F57; return 1; // 'Ὗ' (U+1F5F, E1 BD 9F) → 'ὗ' (U+1F57, E1 BD 97)
        case 0x1FB8: folded[0] = 0x1FB0; return 1; // 'Ᾰ' (U+1FB8, E1 BE B8) → 'ᾰ' (U+1FB0, E1 BE B0)
        case 0x1FB9: folded[0] = 0x1FB1; return 1; // 'Ᾱ' (U+1FB9, E1 BE B9) → 'ᾱ' (U+1FB1, E1 BE B1)
        case 0x1FBA: folded[0] = 0x1F70; return 1; // 'Ὰ' (U+1FBA, E1 BE BA) → 'ὰ' (U+1F70, E1 BD B0)
        case 0x1FBB: folded[0] = 0x1F71; return 1; // 'Ά' (U+1FBB, E1 BE BB) → 'ά' (U+1F71, E1 BD B1)
        case 0x1FBE: folded[0] = 0x03B9; return 1; // 'ι' (U+1FBE, E1 BE BE) → 'ι' (U+03B9, CE B9)
        case 0x1FD8: folded[0] = 0x1FD0; return 1; // 'Ῐ' (U+1FD8, E1 BF 98) → 'ῐ' (U+1FD0, E1 BF 90)
        case 0x1FD9: folded[0] = 0x1FD1; return 1; // 'Ῑ' (U+1FD9, E1 BF 99) → 'ῑ' (U+1FD1, E1 BF 91)
        case 0x1FDA: folded[0] = 0x1F76; return 1; // 'Ὶ' (U+1FDA, E1 BF 9A) → 'ὶ' (U+1F76, E1 BD B6)
        case 0x1FDB: folded[0] = 0x1F77; return 1; // 'Ί' (U+1FDB, E1 BF 9B) → 'ί' (U+1F77, E1 BD B7)
        case 0x1FE8: folded[0] = 0x1FE0; return 1; // 'Ῠ' (U+1FE8, E1 BF A8) → 'ῠ' (U+1FE0, E1 BF A0)
        case 0x1FE9: folded[0] = 0x1FE1; return 1; // 'Ῡ' (U+1FE9, E1 BF A9) → 'ῡ' (U+1FE1, E1 BF A1)
        case 0x1FEA: folded[0] = 0x1F7A; return 1; // 'Ὺ' (U+1FEA, E1 BF AA) → 'ὺ' (U+1F7A, E1 BD BA)
        case 0x1FEB: folded[0] = 0x1F7B; return 1; // 'Ύ' (U+1FEB, E1 BF AB) → 'ύ' (U+1F7B, E1 BD BB)
        case 0x1FEC: folded[0] = 0x1FE5; return 1; // 'Ῥ' (U+1FEC, E1 BF AC) → 'ῥ' (U+1FE5, E1 BF A5)
        case 0x1FF8: folded[0] = 0x1F78; return 1; // 'Ὸ' (U+1FF8, E1 BF B8) → 'ὸ' (U+1F78, E1 BD B8)
        case 0x1FF9: folded[0] = 0x1F79; return 1; // 'Ό' (U+1FF9, E1 BF B9) → 'ό' (U+1F79, E1 BD B9)
        case 0x1FFA: folded[0] = 0x1F7C; return 1; // 'Ὼ' (U+1FFA, E1 BF BA) → 'ὼ' (U+1F7C, E1 BD BC)
        case 0x1FFB: folded[0] = 0x1F7D; return 1; // 'Ώ' (U+1FFB, E1 BF BB) → 'ώ' (U+1F7D, E1 BD BD)
        // Letterlike Symbols: compatibility mappings
        case 0x2126: folded[0] = 0x03C9; return 1; // 'Ω' (U+2126, E2 84 A6) → 'ω' (U+03C9, CF 89)
        case 0x212A: folded[0] = 0x006B; return 1; // 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B)
        case 0x212B: folded[0] = 0x00E5; return 1; // 'Å' (U+212B, E2 84 AB) → 'å' (U+00E5, C3 A5)
        case 0x2132: folded[0] = 0x214E; return 1; // 'Ⅎ' (U+2132, E2 84 B2) → 'ⅎ' (U+214E, E2 85 8E)
        case 0x2183: folded[0] = 0x2184; return 1; // 'Ↄ' (U+2183, E2 86 83) → 'ↄ' (U+2184, E2 86 84)
        // Latin Extended-C: irregular mappings to IPA/other blocks
        case 0x2C60: folded[0] = 0x2C61; return 1; // 'Ⱡ' (U+2C60, E2 B1 A0) → 'ⱡ' (U+2C61, E2 B1 A1)
        case 0x2C62: folded[0] = 0x026B; return 1; // 'Ɫ' (U+2C62, E2 B1 A2) → 'ɫ' (U+026B, C9 AB)
        case 0x2C63: folded[0] = 0x1D7D; return 1; // 'Ᵽ' (U+2C63, E2 B1 A3) → 'ᵽ' (U+1D7D, E1 B5 BD)
        case 0x2C64: folded[0] = 0x027D; return 1; // 'Ɽ' (U+2C64, E2 B1 A4) → 'ɽ' (U+027D, C9 BD)
        case 0x2C67: folded[0] = 0x2C68; return 1; // 'Ⱨ' (U+2C67, E2 B1 A7) → 'ⱨ' (U+2C68, E2 B1 A8)
        case 0x2C69: folded[0] = 0x2C6A; return 1; // 'Ⱪ' (U+2C69, E2 B1 A9) → 'ⱪ' (U+2C6A, E2 B1 AA)
        case 0x2C6B: folded[0] = 0x2C6C; return 1; // 'Ⱬ' (U+2C6B, E2 B1 AB) → 'ⱬ' (U+2C6C, E2 B1 AC)
        case 0x2C6D: folded[0] = 0x0251; return 1; // 'Ɑ' (U+2C6D, E2 B1 AD) → 'ɑ' (U+0251, C9 91)
        case 0x2C6E: folded[0] = 0x0271; return 1; // 'Ɱ' (U+2C6E, E2 B1 AE) → 'ɱ' (U+0271, C9 B1)
        case 0x2C6F: folded[0] = 0x0250; return 1; // 'Ɐ' (U+2C6F, E2 B1 AF) → 'ɐ' (U+0250, C9 90)
        case 0x2C70: folded[0] = 0x0252; return 1; // 'Ɒ' (U+2C70, E2 B1 B0) → 'ɒ' (U+0252, C9 92)
        case 0x2C72: folded[0] = 0x2C73; return 1; // 'Ⱳ' (U+2C72, E2 B1 B2) → 'ⱳ' (U+2C73, E2 B1 B3)
        case 0x2C75: folded[0] = 0x2C76; return 1; // 'Ⱶ' (U+2C75, E2 B1 B5) → 'ⱶ' (U+2C76, E2 B1 B6)
        case 0x2C7E: folded[0] = 0x023F; return 1; // 'Ȿ' (U+2C7E, E2 B1 BE) → 'ȿ' (U+023F, C8 BF)
        case 0x2C7F: folded[0] = 0x0240; return 1; // 'Ɀ' (U+2C7F, E2 B1 BF) → 'ɀ' (U+0240, C9 80)
        // Coptic: irregular cases outside even/odd range
        case 0x2CEB: folded[0] = 0x2CEC; return 1; // 'Ⳬ' (U+2CEB, E2 B3 AB) → 'ⳬ' (U+2CEC, E2 B3 AC)
        case 0x2CED: folded[0] = 0x2CEE; return 1; // 'Ⳮ' (U+2CED, E2 B3 AD) → 'ⳮ' (U+2CEE, E2 B3 AE)
        case 0x2CF2: folded[0] = 0x2CF3; return 1; // 'Ⳳ' (U+2CF2, E2 B3 B2) → 'ⳳ' (U+2CF3, E2 B3 B3)
        // Latin Extended-D: isolated irregulars
        case 0xA779: folded[0] = 0xA77A; return 1; // 'Ꝺ' (U+A779, EA 9D B9) → 'ꝺ' (U+A77A, EA 9D BA)
        case 0xA77B: folded[0] = 0xA77C; return 1; // 'Ꝼ' (U+A77B, EA 9D BB) → 'ꝼ' (U+A77C, EA 9D BC)
        case 0xA77D: folded[0] = 0x1D79; return 1; // 'Ᵹ' (U+A77D, EA 9D BD) → 'ᵹ' (U+1D79, E1 B5 B9)
        case 0xA78B: folded[0] = 0xA78C; return 1; // 'Ꞌ' (U+A78B, EA 9E 8B) → 'ꞌ' (U+A78C, EA 9E 8C)
        case 0xA78D: folded[0] = 0x0265; return 1; // 'Ɥ' (U+A78D, EA 9E 8D) → 'ɥ' (U+0265, C9 A5)
        case 0xA7AA: folded[0] = 0x0266; return 1; // 'Ɦ' (U+A7AA, EA 9E AA) → 'ɦ' (U+0266, C9 A6)
        case 0xA7AB: folded[0] = 0x025C; return 1; // 'Ɜ' (U+A7AB, EA 9E AB) → 'ɜ' (U+025C, C9 9C)
        case 0xA7AC: folded[0] = 0x0261; return 1; // 'Ɡ' (U+A7AC, EA 9E AC) → 'ɡ' (U+0261, C9 A1)
        case 0xA7AD: folded[0] = 0x026C; return 1; // 'Ɬ' (U+A7AD, EA 9E AD) → 'ɬ' (U+026C, C9 AC)
        case 0xA7AE: folded[0] = 0x026A; return 1; // 'Ɪ' (U+A7AE, EA 9E AE) → 'ɪ' (U+026A, C9 AA)
        case 0xA7B0: folded[0] = 0x029E; return 1; // 'Ʞ' (U+A7B0, EA 9E B0) → 'ʞ' (U+029E, CA 9E)
        case 0xA7B1: folded[0] = 0x0287; return 1; // 'Ʇ' (U+A7B1, EA 9E B1) → 'ʇ' (U+0287, CA 87)
        case 0xA7B2: folded[0] = 0x029D; return 1; // 'Ʝ' (U+A7B2, EA 9E B2) → 'ʝ' (U+029D, CA 9D)
        case 0xA7B3: folded[0] = 0xAB53; return 1; // 'Ꭓ' (U+A7B3, EA 9E B3) → 'ꭓ' (U+AB53, EA AD 93)
        case 0xA7C4: folded[0] = 0xA794; return 1; // 'Ꞔ' (U+A7C4, EA 9F 84) → 'ꞔ' (U+A794, EA 9E 94)
        case 0xA7C5: folded[0] = 0x0282; return 1; // 'Ʂ' (U+A7C5, EA 9F 85) → 'ʂ' (U+0282, CA 82)
        case 0xA7C6: folded[0] = 0x1D8E; return 1; // 'Ᶎ' (U+A7C6, EA 9F 86) → 'ᶎ' (U+1D8E, E1 B6 8E)
        case 0xA7C7: folded[0] = 0xA7C8; return 1; // 'Ꟈ' (U+A7C7, EA 9F 87) → 'ꟈ' (U+A7C8, EA 9F 88)
        case 0xA7C9: folded[0] = 0xA7CA; return 1; // 'Ꟊ' (U+A7C9, EA 9F 89) → 'ꟊ' (U+A7CA, EA 9F 8A)
        case 0xA7CB: folded[0] = 0x0264; return 1; // 'Ɤ' (U+A7CB, EA 9F 8B) → 'ɤ' (U+0264, C9 A4)
        case 0xA7CC: folded[0] = 0xA7CD; return 1; // 'Ꟍ' (U+A7CC, EA 9F 8C) → 'ꟍ' (U+A7CD, EA 9F 8D)
        case 0xA7CE: folded[0] = 0xA7CF; return 1; // '꟎' (U+A7CE, EA 9F 8E) → '꟏' (U+A7CF, EA 9F 8F)
        case 0xA7D0: folded[0] = 0xA7D1; return 1; // 'Ꟑ' (U+A7D0, EA 9F 90) → 'ꟑ' (U+A7D1, EA 9F 91)
        case 0xA7D2: folded[0] = 0xA7D3; return 1; // '꟒' (U+A7D2, EA 9F 92) → 'ꟓ' (U+A7D3, EA 9F 93)
        case 0xA7D4: folded[0] = 0xA7D5; return 1; // '꟔' (U+A7D4, EA 9F 94) → 'ꟕ' (U+A7D5, EA 9F 95)
        case 0xA7D6: folded[0] = 0xA7D7; return 1; // 'Ꟗ' (U+A7D6, EA 9F 96) → 'ꟗ' (U+A7D7, EA 9F 97)
        case 0xA7D8: folded[0] = 0xA7D9; return 1; // 'Ꟙ' (U+A7D8, EA 9F 98) → 'ꟙ' (U+A7D9, EA 9F 99)
        case 0xA7DA: folded[0] = 0xA7DB; return 1; // 'Ꟛ' (U+A7DA, EA 9F 9A) → 'ꟛ' (U+A7DB, EA 9F 9B)
        case 0xA7DC: folded[0] = 0x019B; return 1; // 'Ƛ' (U+A7DC, EA 9F 9C) → 'ƛ' (U+019B, C6 9B)
        case 0xA7F5: folded[0] = 0xA7F6; return 1; // 'Ꟶ' (U+A7F5, EA 9F B5) → 'ꟶ' (U+A7F6, EA 9F B6)
        }

        // Next let's handle the 3-byte one-to-many expansions
        switch (rune) {
        // Latin Extended Additional
        // 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
        case 0x1E96: folded[0] = 0x0068; folded[1] = 0x0331; return 2;
        // 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
        case 0x1E97: folded[0] = 0x0074; folded[1] = 0x0308; return 2;
        // 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
        case 0x1E98: folded[0] = 0x0077; folded[1] = 0x030A; return 2;
        // 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
        case 0x1E99: folded[0] = 0x0079; folded[1] = 0x030A; return 2;
        // 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
        case 0x1E9A: folded[0] = 0x0061; folded[1] = 0x02BE; return 2;
        // 'ẞ' (U+1E9E, E1 BA 9E) → "ss" (U+0073 U+0073, 73 73)
        case 0x1E9E: folded[0] = 0x0073; folded[1] = 0x0073; return 2;
        // Greek Extended: breathing marks
        // 'ὐ' (U+1F50, E1 BD 90) → "ὐ" (U+03C5 U+0313, CF 85 CC 93)
        case 0x1F50: folded[0] = 0x03C5; folded[1] = 0x0313; return 2;
        // 'ὒ' (U+1F52, E1 BD 92) → "ὒ" (U+03C5 U+0313 U+0300, CF 85 CC 93 CC 80)
        case 0x1F52: folded[0] = 0x03C5; folded[1] = 0x0313; folded[2] = 0x0300; return 3;
        // 'ὔ' (U+1F54, E1 BD 94) → "ὔ" (U+03C5 U+0313 U+0301, CF 85 CC 93 CC 81)
        case 0x1F54: folded[0] = 0x03C5; folded[1] = 0x0313; folded[2] = 0x0301; return 3;
        // 'ὖ' (U+1F56, E1 BD 96) → "ὖ" (U+03C5 U+0313 U+0342, CF 85 CC 93 CD 82)
        case 0x1F56: folded[0] = 0x03C5; folded[1] = 0x0313; folded[2] = 0x0342; return 3;
        // Greek Extended: iota subscript combinations (0x1F80-0x1FAF)
        // 'ᾀ' (U+1F80, E1 BE 80) → "ἀι" (U+1F00 U+03B9, E1 BC 80 CE B9)
        case 0x1F80: folded[0] = 0x1F00; folded[1] = 0x03B9; return 2;
        // 'ᾁ' (U+1F81, E1 BE 81) → "ἁι" (U+1F01 U+03B9, E1 BC 81 CE B9)
        case 0x1F81: folded[0] = 0x1F01; folded[1] = 0x03B9; return 2;
        // 'ᾂ' (U+1F82, E1 BE 82) → "ἂι" (U+1F02 U+03B9, E1 BC 82 CE B9)
        case 0x1F82: folded[0] = 0x1F02; folded[1] = 0x03B9; return 2;
        // 'ᾃ' (U+1F83, E1 BE 83) → "ἃι" (U+1F03 U+03B9, E1 BC 83 CE B9)
        case 0x1F83: folded[0] = 0x1F03; folded[1] = 0x03B9; return 2;
        // 'ᾄ' (U+1F84, E1 BE 84) → "ἄι" (U+1F04 U+03B9, E1 BC 84 CE B9)
        case 0x1F84: folded[0] = 0x1F04; folded[1] = 0x03B9; return 2;
        // 'ᾅ' (U+1F85, E1 BE 85) → "ἅι" (U+1F05 U+03B9, E1 BC 85 CE B9)
        case 0x1F85: folded[0] = 0x1F05; folded[1] = 0x03B9; return 2;
        // 'ᾆ' (U+1F86, E1 BE 86) → "ἆι" (U+1F06 U+03B9, E1 BC 86 CE B9)
        case 0x1F86: folded[0] = 0x1F06; folded[1] = 0x03B9; return 2;
        // 'ᾇ' (U+1F87, E1 BE 87) → "ἇι" (U+1F07 U+03B9, E1 BC 87 CE B9)
        case 0x1F87: folded[0] = 0x1F07; folded[1] = 0x03B9; return 2;
        // 'ᾈ' (U+1F88, E1 BE 88) → "ἀι" (U+1F00 U+03B9, E1 BC 80 CE B9)
        case 0x1F88: folded[0] = 0x1F00; folded[1] = 0x03B9; return 2;
        // 'ᾉ' (U+1F89, E1 BE 89) → "ἁι" (U+1F01 U+03B9, E1 BC 81 CE B9)
        case 0x1F89: folded[0] = 0x1F01; folded[1] = 0x03B9; return 2;
        // 'ᾊ' (U+1F8A, E1 BE 8A) → "ἂι" (U+1F02 U+03B9, E1 BC 82 CE B9)
        case 0x1F8A: folded[0] = 0x1F02; folded[1] = 0x03B9; return 2;
        // 'ᾋ' (U+1F8B, E1 BE 8B) → "ἃι" (U+1F03 U+03B9, E1 BC 83 CE B9)
        case 0x1F8B: folded[0] = 0x1F03; folded[1] = 0x03B9; return 2;
        // 'ᾌ' (U+1F8C, E1 BE 8C) → "ἄι" (U+1F04 U+03B9, E1 BC 84 CE B9)
        case 0x1F8C: folded[0] = 0x1F04; folded[1] = 0x03B9; return 2;
        // 'ᾍ' (U+1F8D, E1 BE 8D) → "ἅι" (U+1F05 U+03B9, E1 BC 85 CE B9)
        case 0x1F8D: folded[0] = 0x1F05; folded[1] = 0x03B9; return 2;
        // 'ᾎ' (U+1F8E, E1 BE 8E) → "ἆι" (U+1F06 U+03B9, E1 BC 86 CE B9)
        case 0x1F8E: folded[0] = 0x1F06; folded[1] = 0x03B9; return 2;
        // 'ᾏ' (U+1F8F, E1 BE 8F) → "ἇι" (U+1F07 U+03B9, E1 BC 87 CE B9)
        case 0x1F8F: folded[0] = 0x1F07; folded[1] = 0x03B9; return 2;
        // 'ᾐ' (U+1F90, E1 BE 90) → "ἠι" (U+1F20 U+03B9, E1 BC A0 CE B9)
        case 0x1F90: folded[0] = 0x1F20; folded[1] = 0x03B9; return 2;
        // 'ᾑ' (U+1F91, E1 BE 91) → "ἡι" (U+1F21 U+03B9, E1 BC A1 CE B9)
        case 0x1F91: folded[0] = 0x1F21; folded[1] = 0x03B9; return 2;
        // 'ᾒ' (U+1F92, E1 BE 92) → "ἢι" (U+1F22 U+03B9, E1 BC A2 CE B9)
        case 0x1F92: folded[0] = 0x1F22; folded[1] = 0x03B9; return 2;
        // 'ᾓ' (U+1F93, E1 BE 93) → "ἣι" (U+1F23 U+03B9, E1 BC A3 CE B9)
        case 0x1F93: folded[0] = 0x1F23; folded[1] = 0x03B9; return 2;
        // 'ᾔ' (U+1F94, E1 BE 94) → "ἤι" (U+1F24 U+03B9, E1 BC A4 CE B9)
        case 0x1F94: folded[0] = 0x1F24; folded[1] = 0x03B9; return 2;
        // 'ᾕ' (U+1F95, E1 BE 95) → "ἥι" (U+1F25 U+03B9, E1 BC A5 CE B9)
        case 0x1F95: folded[0] = 0x1F25; folded[1] = 0x03B9; return 2;
        // 'ᾖ' (U+1F96, E1 BE 96) → "ἦι" (U+1F26 U+03B9, E1 BC A6 CE B9)
        case 0x1F96: folded[0] = 0x1F26; folded[1] = 0x03B9; return 2;
        // 'ᾗ' (U+1F97, E1 BE 97) → "ἧι" (U+1F27 U+03B9, E1 BC A7 CE B9)
        case 0x1F97: folded[0] = 0x1F27; folded[1] = 0x03B9; return 2;
        // 'ᾘ' (U+1F98, E1 BE 98) → "ἠι" (U+1F20 U+03B9, E1 BC A0 CE B9)
        case 0x1F98: folded[0] = 0x1F20; folded[1] = 0x03B9; return 2;
        // 'ᾙ' (U+1F99, E1 BE 99) → "ἡι" (U+1F21 U+03B9, E1 BC A1 CE B9)
        case 0x1F99: folded[0] = 0x1F21; folded[1] = 0x03B9; return 2;
        // 'ᾚ' (U+1F9A, E1 BE 9A) → "ἢι" (U+1F22 U+03B9, E1 BC A2 CE B9)
        case 0x1F9A: folded[0] = 0x1F22; folded[1] = 0x03B9; return 2;
        // 'ᾛ' (U+1F9B, E1 BE 9B) → "ἣι" (U+1F23 U+03B9, E1 BC A3 CE B9)
        case 0x1F9B: folded[0] = 0x1F23; folded[1] = 0x03B9; return 2;
        // 'ᾜ' (U+1F9C, E1 BE 9C) → "ἤι" (U+1F24 U+03B9, E1 BC A4 CE B9)
        case 0x1F9C: folded[0] = 0x1F24; folded[1] = 0x03B9; return 2;
        // 'ᾝ' (U+1F9D, E1 BE 9D) → "ἥι" (U+1F25 U+03B9, E1 BC A5 CE B9)
        case 0x1F9D: folded[0] = 0x1F25; folded[1] = 0x03B9; return 2;
        // 'ᾞ' (U+1F9E, E1 BE 9E) → "ἦι" (U+1F26 U+03B9, E1 BC A6 CE B9)
        case 0x1F9E: folded[0] = 0x1F26; folded[1] = 0x03B9; return 2;
        // 'ᾟ' (U+1F9F, E1 BE 9F) → "ἧι" (U+1F27 U+03B9, E1 BC A7 CE B9)
        case 0x1F9F: folded[0] = 0x1F27; folded[1] = 0x03B9; return 2;
        // 'ᾠ' (U+1FA0, E1 BE A0) → "ὠι" (U+1F60 U+03B9, E1 BD A0 CE B9)
        case 0x1FA0: folded[0] = 0x1F60; folded[1] = 0x03B9; return 2;
        // 'ᾡ' (U+1FA1, E1 BE A1) → "ὡι" (U+1F61 U+03B9, E1 BD A1 CE B9)
        case 0x1FA1: folded[0] = 0x1F61; folded[1] = 0x03B9; return 2;
        // 'ᾢ' (U+1FA2, E1 BE A2) → "ὢι" (U+1F62 U+03B9, E1 BD A2 CE B9)
        case 0x1FA2: folded[0] = 0x1F62; folded[1] = 0x03B9; return 2;
        // 'ᾣ' (U+1FA3, E1 BE A3) → "ὣι" (U+1F63 U+03B9, E1 BD A3 CE B9)
        case 0x1FA3: folded[0] = 0x1F63; folded[1] = 0x03B9; return 2;
        // 'ᾤ' (U+1FA4, E1 BE A4) → "ὤι" (U+1F64 U+03B9, E1 BD A4 CE B9)
        case 0x1FA4: folded[0] = 0x1F64; folded[1] = 0x03B9; return 2;
        // 'ᾥ' (U+1FA5, E1 BE A5) → "ὥι" (U+1F65 U+03B9, E1 BD A5 CE B9)
        case 0x1FA5: folded[0] = 0x1F65; folded[1] = 0x03B9; return 2;
        // 'ᾦ' (U+1FA6, E1 BE A6) → "ὦι" (U+1F66 U+03B9, E1 BD A6 CE B9)
        case 0x1FA6: folded[0] = 0x1F66; folded[1] = 0x03B9; return 2;
        // 'ᾧ' (U+1FA7, E1 BE A7) → "ὧι" (U+1F67 U+03B9, E1 BD A7 CE B9)
        case 0x1FA7: folded[0] = 0x1F67; folded[1] = 0x03B9; return 2;
        // 'ᾨ' (U+1FA8, E1 BE A8) → "ὠι" (U+1F60 U+03B9, E1 BD A0 CE B9)
        case 0x1FA8: folded[0] = 0x1F60; folded[1] = 0x03B9; return 2;
        // 'ᾩ' (U+1FA9, E1 BE A9) → "ὡι" (U+1F61 U+03B9, E1 BD A1 CE B9)
        case 0x1FA9: folded[0] = 0x1F61; folded[1] = 0x03B9; return 2;
        // 'ᾪ' (U+1FAA, E1 BE AA) → "ὢι" (U+1F62 U+03B9, E1 BD A2 CE B9)
        case 0x1FAA: folded[0] = 0x1F62; folded[1] = 0x03B9; return 2;
        // 'ᾫ' (U+1FAB, E1 BE AB) → "ὣι" (U+1F63 U+03B9, E1 BD A3 CE B9)
        case 0x1FAB: folded[0] = 0x1F63; folded[1] = 0x03B9; return 2;
        // 'ᾬ' (U+1FAC, E1 BE AC) → "ὤι" (U+1F64 U+03B9, E1 BD A4 CE B9)
        case 0x1FAC: folded[0] = 0x1F64; folded[1] = 0x03B9; return 2;
        // 'ᾭ' (U+1FAD, E1 BE AD) → "ὥι" (U+1F65 U+03B9, E1 BD A5 CE B9)
        case 0x1FAD: folded[0] = 0x1F65; folded[1] = 0x03B9; return 2;
        // 'ᾮ' (U+1FAE, E1 BE AE) → "ὦι" (U+1F66 U+03B9, E1 BD A6 CE B9)
        case 0x1FAE: folded[0] = 0x1F66; folded[1] = 0x03B9; return 2;
        // 'ᾯ' (U+1FAF, E1 BE AF) → "ὧι" (U+1F67 U+03B9, E1 BD A7 CE B9)
        case 0x1FAF: folded[0] = 0x1F67; folded[1] = 0x03B9; return 2;
        // Greek Extended: vowel + iota subscript (0x1FB2-0x1FFC)
        // 'ᾲ' (U+1FB2, E1 BE B2) → "ὰι" (U+1F70 U+03B9, E1 BD B0 CE B9)
        case 0x1FB2: folded[0] = 0x1F70; folded[1] = 0x03B9; return 2;
        // 'ᾳ' (U+1FB3, E1 BE B3) → "αι" (U+03B1 U+03B9, CE B1 CE B9)
        case 0x1FB3: folded[0] = 0x03B1; folded[1] = 0x03B9; return 2;
        // 'ᾴ' (U+1FB4, E1 BE B4) → "άι" (U+03AC U+03B9, CE AC CE B9)
        case 0x1FB4: folded[0] = 0x03AC; folded[1] = 0x03B9; return 2;
        // 'ᾶ' (U+1FB6, E1 BE B6) → "ᾶ" (U+03B1 U+0342, CE B1 CD 82)
        case 0x1FB6: folded[0] = 0x03B1; folded[1] = 0x0342; return 2;
        // 'ᾷ' (U+1FB7, E1 BE B7) → "ᾶι" (U+03B1 U+0342 U+03B9, CE B1 CD 82 CE B9)
        case 0x1FB7: folded[0] = 0x03B1; folded[1] = 0x0342; folded[2] = 0x03B9; return 3;
        // 'ᾼ' (U+1FBC, E1 BE BC) → "αι" (U+03B1 U+03B9, CE B1 CE B9)
        case 0x1FBC: folded[0] = 0x03B1; folded[1] = 0x03B9; return 2;
        // 'ῂ' (U+1FC2, E1 BF 82) → "ὴι" (U+1F74 U+03B9, E1 BD B4 CE B9)
        case 0x1FC2: folded[0] = 0x1F74; folded[1] = 0x03B9; return 2;
        // 'ῃ' (U+1FC3, E1 BF 83) → "ηι" (U+03B7 U+03B9, CE B7 CE B9)
        case 0x1FC3: folded[0] = 0x03B7; folded[1] = 0x03B9; return 2;
        // 'ῄ' (U+1FC4, E1 BF 84) → "ήι" (U+03AE U+03B9, CE AE CE B9)
        case 0x1FC4: folded[0] = 0x03AE; folded[1] = 0x03B9; return 2;
        // 'ῆ' (U+1FC6, E1 BF 86) → "ῆ" (U+03B7 U+0342, CE B7 CD 82)
        case 0x1FC6: folded[0] = 0x03B7; folded[1] = 0x0342; return 2;
        // 'ῇ' (U+1FC7, E1 BF 87) → "ῆι" (U+03B7 U+0342 U+03B9, CE B7 CD 82 CE B9)
        case 0x1FC7: folded[0] = 0x03B7; folded[1] = 0x0342; folded[2] = 0x03B9; return 3;
        // 'ῌ' (U+1FCC, E1 BF 8C) → "ηι" (U+03B7 U+03B9, CE B7 CE B9)
        case 0x1FCC: folded[0] = 0x03B7; folded[1] = 0x03B9; return 2;
        // 'ῒ' (U+1FD2, E1 BF 92) → "ῒ" (U+03B9 U+0308 U+0300, CE B9 CC 88 CC 80)
        case 0x1FD2: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0300; return 3;
        // 'ΐ' (U+1FD3, E1 BF 93) → "ΐ" (U+03B9 U+0308 U+0301, CE B9 CC 88 CC 81)
        case 0x1FD3: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0301; return 3;
        // 'ῖ' (U+1FD6, E1 BF 96) → "ῖ" (U+03B9 U+0342, CE B9 CD 82)
        case 0x1FD6: folded[0] = 0x03B9; folded[1] = 0x0342; return 2;
        // 'ῗ' (U+1FD7, E1 BF 97) → "ῗ" (U+03B9 U+0308 U+0342, CE B9 CC 88 CD 82)
        case 0x1FD7: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0342; return 3;
        // 'ῢ' (U+1FE2, E1 BF A2) → "ῢ" (U+03C5 U+0308 U+0300, CF 85 CC 88 CC 80)
        case 0x1FE2: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0300; return 3;
        // 'ΰ' (U+1FE3, E1 BF A3) → "ΰ" (U+03C5 U+0308 U+0301, CF 85 CC 88 CC 81)
        case 0x1FE3: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0301; return 3;
        // 'ῤ' (U+1FE4, E1 BF A4) → "ῤ" (U+03C1 U+0313, CF 81 CC 93)
        case 0x1FE4: folded[0] = 0x03C1; folded[1] = 0x0313; return 2;
        // 'ῦ' (U+1FE6, E1 BF A6) → "ῦ" (U+03C5 U+0342, CF 85 CD 82)
        case 0x1FE6: folded[0] = 0x03C5; folded[1] = 0x0342; return 2;
        // 'ῧ' (U+1FE7, E1 BF A7) → "ῧ" (U+03C5 U+0308 U+0342, CF 85 CC 88 CD 82)
        case 0x1FE7: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0342; return 3;
        // 'ῲ' (U+1FF2, E1 BF B2) → "ὼι" (U+1F7C U+03B9, E1 BD BC CE B9)
        case 0x1FF2: folded[0] = 0x1F7C; folded[1] = 0x03B9; return 2;
        // 'ῳ' (U+1FF3, E1 BF B3) → "ωι" (U+03C9 U+03B9, CF 89 CE B9)
        case 0x1FF3: folded[0] = 0x03C9; folded[1] = 0x03B9; return 2;
        // 'ῴ' (U+1FF4, E1 BF B4) → "ώι" (U+03CE U+03B9, CF 8E CE B9)
        case 0x1FF4: folded[0] = 0x03CE; folded[1] = 0x03B9; return 2;
        // 'ῶ' (U+1FF6, E1 BF B6) → "ῶ" (U+03C9 U+0342, CF 89 CD 82)
        case 0x1FF6: folded[0] = 0x03C9; folded[1] = 0x0342; return 2;
        // 'ῷ' (U+1FF7, E1 BF B7) → "ῶι" (U+03C9 U+0342 U+03B9, CF 89 CD 82 CE B9)
        case 0x1FF7: folded[0] = 0x03C9; folded[1] = 0x0342; folded[2] = 0x03B9; return 3;
        // 'ῼ' (U+1FFC, E1 BF BC) → "ωι" (U+03C9 U+03B9, CF 89 CE B9)
        case 0x1FFC: folded[0] = 0x03C9; folded[1] = 0x03B9; return 2;
        // Alphabetic Presentation Forms: ligatures
        // 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
        case 0xFB00: folded[0] = 0x0066; folded[1] = 0x0066; return 2;
        // 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
        case 0xFB01: folded[0] = 0x0066; folded[1] = 0x0069; return 2;
        // 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
        case 0xFB02: folded[0] = 0x0066; folded[1] = 0x006C; return 2;
        // 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
        case 0xFB03: folded[0] = 0x0066; folded[1] = 0x0066; folded[2] = 0x0069; return 3;
        // 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
        case 0xFB04: folded[0] = 0x0066; folded[1] = 0x0066; folded[2] = 0x006C; return 3;
        // 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
        case 0xFB05: folded[0] = 0x0073; folded[1] = 0x0074; return 2;
        // 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
        case 0xFB06: folded[0] = 0x0073; folded[1] = 0x0074; return 2;
        // Armenian ligatures
        // 'ﬓ' (U+FB13, EF AC 93) → "մն" (U+0574 U+0576, D5 B4 D5 B6)
        case 0xFB13: folded[0] = 0x0574; folded[1] = 0x0576; return 2;
        // 'ﬔ' (U+FB14, EF AC 94) → "մե" (U+0574 U+0565, D5 B4 D5 A5)
        case 0xFB14: folded[0] = 0x0574; folded[1] = 0x0565; return 2;
        // 'ﬕ' (U+FB15, EF AC 95) → "մի" (U+0574 U+056B, D5 B4 D5 AB)
        case 0xFB15: folded[0] = 0x0574; folded[1] = 0x056B; return 2;
        // 'ﬖ' (U+FB16, EF AC 96) → "վն" (U+057E U+0576, D5 BE D5 B6)
        case 0xFB16: folded[0] = 0x057E; folded[1] = 0x0576; return 2;
        // 'ﬗ' (U+FB17, EF AC 97) → "մխ" (U+0574 U+056D, D5 B4 D5 AD)
        case 0xFB17: folded[0] = 0x0574; folded[1] = 0x056D; return 2;
        }

        folded[0] = rune; return 1;  // 3-byte: no folding needed
    }

    // 4-byte UTF-8 (U+10000-10FFFF): Deseret, Osage, Vithkuqi, etc.

    // Deseret 𐐀-𐐧: 0x10400-0x10427 (+40)
    if ((sz_u32_t)(rune - 0x10400) <= 0x27) {
        sz_assert_(sz_is_in_range_(rune, 0x10400, 0x10427));
        folded[0] = rune + 0x28; return 1; }

    // Osage 𐒰-𐓓: 0x104B0-0x104D3 (+40)
    if ((sz_u32_t)(rune - 0x104B0) <= 0x23) {
        sz_assert_(sz_is_in_range_(rune, 0x104B0, 0x104D3));
        folded[0] = rune + 0x28; return 1; }

    // Vithkuqi: 3 ranges with gaps, all +39
    if ((sz_u32_t)(rune - 0x10570) <= 0x0A) { // 0x10570-0x1057A
        sz_assert_(sz_is_in_range_(rune, 0x10570, 0x1057A));
        folded[0] = rune + 0x27; return 1; }
    if ((sz_u32_t)(rune - 0x1057C) <= 0x0E) { // 0x1057C-0x1058A
        sz_assert_(sz_is_in_range_(rune, 0x1057C, 0x1058A));
        folded[0] = rune + 0x27; return 1; }
    if ((sz_u32_t)(rune - 0x1058C) <= 0x06) { // 0x1058C-0x10592
        sz_assert_(sz_is_in_range_(rune, 0x1058C, 0x10592));
        folded[0] = rune + 0x27; return 1; }

    // Old Hungarian: 0x10C80-0x10CB2 (+64)
    if ((sz_u32_t)(rune - 0x10C80) <= 0x32) {
        sz_assert_(sz_is_in_range_(rune, 0x10C80, 0x10CB2));
        folded[0] = rune + 0x40; return 1; }

    // Garay: 0x10D50-0x10D65 (+32)
    if ((sz_u32_t)(rune - 0x10D50) <= 0x15) {
        sz_assert_(sz_is_in_range_(rune, 0x10D50, 0x10D65));
        folded[0] = rune + 0x20; return 1; }

    // Warang Citi: 0x118A0-0x118BF (+32)
    if ((sz_u32_t)(rune - 0x118A0) <= 0x1F) {
        sz_assert_(sz_is_in_range_(rune, 0x118A0, 0x118BF));
        folded[0] = rune + 0x20; return 1; }

    // Medefaidrin: 0x16E40-0x16E5F (+32)
    if ((sz_u32_t)(rune - 0x16E40) <= 0x1F) {
        sz_assert_(sz_is_in_range_(rune, 0x16E40, 0x16E5F));
        folded[0] = rune + 0x20; return 1; }

    // Beria Erfe: 0x16EA0-0x16EB8 (+27)
    if ((sz_u32_t)(rune - 0x16EA0) <= 0x18) {
        sz_assert_(sz_is_in_range_(rune, 0x16EA0, 0x16EB8));
        folded[0] = rune + 0x1B; return 1; }

    // Adlam: 0x1E900-0x1E921 (+34)
    if ((sz_u32_t)(rune - 0x1E900) <= 0x21) {
        sz_assert_(sz_is_in_range_(rune, 0x1E900, 0x1E921));
        folded[0] = rune + 0x22; return 1; }

    // Next let's handle the 4-byte irregular mappings
    switch (rune) {
    // Vithkuqi: Albanian historical script
    case 0x10594: folded[0] = 0x105BB; return 1; // '𐖔' (U+010594, F0 90 96 94) → '𐖻' (U+0105BB, F0 90 96 BB)
    case 0x10595: folded[0] = 0x105BC; return 1; // '𐖕' (U+010595, F0 90 96 95) → '𐖼' (U+0105BC, F0 90 96 BC)
    }

    folded[0] = rune; return 1;  // No folding needed
    // clang-format on
}

/**
 *  @brief Branchless ASCII case fold - converts A-Z to a-z.
 *  Uses unsigned subtraction trick: (c - 'A') <= 25 is true only for uppercase letters.
 */
SZ_INTERNAL sz_u8_t sz_ascii_fold_(sz_u8_t c) { return c + (((sz_u8_t)(c - 'A') <= 25u) * 0x20); }

/**
 *  @brief Folded-rune representation of a byte that does not begin a well-formed codepoint.
 *
 *  A malformed byte folds to itself and is matched/compared byte-for-byte, never as a Unicode codepoint.
 *  Tagging it above the valid Unicode range (0x10FFFF) keeps it distinct from every real folded rune, so a
 *  lone malformed byte 0xFC can only match another malformed 0xFC - never the valid rune U+00FC ('ü'). Two
 *  equal malformed bytes still produce equal tagged runes, preserving byte-for-byte matching.
 */
SZ_INTERNAL sz_rune_t sz_rune_malformed_byte_(sz_u8_t byte) { return 0x80000000u | (sz_rune_t)byte; }

/**
 *  @brief Iterator state for streaming through folded UTF-8 runes.
 *  Handles one-to-many case folding expansions (e.g., 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)) transparently.
 */
typedef struct {
    sz_cptr_t ptr;           // Current position in UTF-8 string
    sz_cptr_t end;           // End of string
    sz_rune_t pending[4];    // Buffered folded runes from one-to-many expansions
    sz_size_t pending_count; // Number of pending folded runes
    sz_size_t pending_idx;   // Current index into pending buffer
} sz_utf8_folded_iter_t_;

/** @brief Initialize a folded rune iterator. */
SZ_INTERNAL void sz_utf8_folded_iter_init_(sz_utf8_folded_iter_t_ *iterator, sz_cptr_t string, sz_size_t length) {
    iterator->ptr = string;
    iterator->end = string + length;
    iterator->pending_count = 0;
    iterator->pending_idx = 0;
}

/**
 *  @brief Get next folded rune. Returns `sz_false_k` when exhausted.
 *  Malformed UTF-8 is handled losslessly: a byte that does not begin a well-formed codepoint is emitted as a
 *  single literal byte (tagged so it compares byte-for-byte and never collides with a real folded codepoint) and
 *  the iterator resyncs by one byte, never reading past `end`.
 */
SZ_INTERNAL sz_bool_t sz_utf8_folded_iter_next_(sz_utf8_folded_iter_t_ *it, sz_rune_t *out_rune) {
    // Refill pending buffer if exhausted
    if (it->pending_idx >= it->pending_count) {
        if (it->ptr >= it->end) return sz_false_k;

        // ASCII fast-path: fold inline without buffering
        sz_u8_t lead = *(sz_u8_t const *)it->ptr;
        if (lead < 0x80) {
            *out_rune = sz_ascii_fold_(lead);
            it->ptr++;
            it->pending_count = 0; // Clear pending buffer
            it->pending_idx = 0;   // Signal first rune of new codepoint for source tracking
            return sz_true_k;
        }

        // Multi-byte UTF-8: decode (bounds-checked), fold, and buffer. A byte that does not begin a
        // well-formed codepoint folds to itself (>= 0x80 bytes are unchanged by `sz_ascii_fold_`) and resyncs
        // by one byte, never over-reading past `end`.
        sz_rune_t rune;
        sz_rune_length_t const rune_length = sz_rune_parse(it->ptr, it->end, &rune);
        if (rune_length == sz_utf8_invalid_k) {
            *out_rune = sz_rune_malformed_byte_(lead);
            it->ptr++;
            it->pending_count = 0;
            it->pending_idx = 0;
            return sz_true_k;
        }

        it->ptr += rune_length;
        // Pre-fill pending buffer with sentinel values to prevent stale data from causing false matches.
        // The fold function will overwrite positions it uses; unused positions keep the sentinel.
        // This follows the same pattern as sz_utf8_uncased_find_2folded_serial_ and
        // sz_utf8_uncased_find_3folded_serial_.
        it->pending[0] = 0xFFFFFFFFu;
        it->pending[1] = 0xFFFFFFFEu;
        it->pending[2] = 0xFFFFFFFDu;
        it->pending[3] = 0xFFFFFFFCu;
        it->pending_count = sz_unicode_fold_codepoint_(rune, it->pending);
        it->pending_idx = 0;
    }

    *out_rune = it->pending[it->pending_idx++];
    return sz_true_k;
}

/**
 *  @brief Reverse iterator state for streaming through folded UTF-8 runes backwards.
 * Handles one-to-many case folding expansions (e.g., 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)) transparently
 * in reverse order.
 */
typedef struct {
    sz_cptr_t ptr;           // Current position (points to byte AFTER current sequence)
    sz_cptr_t start;         // Start of string (stop when ptr reaches this)
    sz_rune_t pending[4];    // Buffered folded runes from one-to-many expansions (in reverse order)
    sz_size_t pending_count; // Number of pending folded runes
    sz_size_t pending_idx;   // Current index into pending buffer
} sz_utf8_folded_reverse_iter_t_;

/** @brief Initialize a reverse folded rune iterator. Iterates from end towards start. */
SZ_INTERNAL void sz_utf8_folded_reverse_iter_init_(sz_utf8_folded_reverse_iter_t_ *it, sz_cptr_t start, sz_cptr_t end) {
    it->ptr = end;
    it->start = start;
    it->pending_count = 0;
    it->pending_idx = 0;
}

/**
 *  @brief Get previous folded rune (walking backwards). Returns `sz_false_k` when exhausted.
 * When a codepoint folds to multiple runes (like 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)), returns them in
 * reverse order ('s', then 's'). Malformed UTF-8 is handled losslessly and byte-identically to the forward
 * iterator: a byte that does not begin/end a well-formed codepoint is emitted as a single tagged literal byte and
 * the iterator resyncs by one byte, so the backward rune stream is exactly the reverse of the forward stream.
 */
SZ_INTERNAL sz_bool_t sz_utf8_folded_reverse_iter_prev_(sz_utf8_folded_reverse_iter_t_ *it, sz_rune_t *out_rune) {
    // Return pending runes if any (stored in reverse order, consumed in reverse)
    if (it->pending_idx < it->pending_count) {
        *out_rune = it->pending[it->pending_count - 1 - it->pending_idx];
        it->pending_idx++;
        return sz_true_k;
    }

    // Refill: find previous codepoint
    if (it->ptr <= it->start) return sz_false_k;

    // Remember one-past-the-end of the sequence we are about to decode, so the strict decode is bounded
    // and a malformed run resyncs one byte at a time - mirroring the forward iterator byte-for-byte.
    sz_cptr_t const sequence_end = it->ptr;

    // The byte immediately before `sequence_end` is the last byte of whatever codepoint ends here.
    sz_u8_t const last_byte = *(sz_u8_t const *)(sequence_end - 1);

    // ASCII fast-path: a byte < 0x80 is always its own complete 1-byte codepoint.
    if (last_byte < 0x80) {
        it->ptr = sequence_end - 1;
        *out_rune = sz_ascii_fold_(last_byte);
        it->pending_count = 0;
        it->pending_idx = 0;
        return sz_true_k;
    }

    // Otherwise walk backwards over up to 3 continuation bytes (0x80-0xBF) to locate a candidate lead.
    // A well-formed multi-byte rune is at most 4 bytes, so stop after considering 4 positions.
    sz_cptr_t candidate = sequence_end - 1;
    for (sz_size_t back = 0; back < 3 && candidate > it->start && (*(sz_u8_t const *)candidate & 0xC0) == 0x80;
         ++back)
        candidate--;

    // Multi-byte UTF-8: decode (bounded) and fold only if the bytes from the candidate lead form a well-formed
    // codepoint that ends EXACTLY at `sequence_end`. Otherwise the last byte does not begin/end a valid rune, so
    // treat it as a literal folded-to-itself byte and resync by one - matching the forward iterator byte-for-byte.
    sz_rune_t rune;
    sz_rune_length_t const rune_length = sz_rune_parse(candidate, sequence_end, &rune);
    if (rune_length == sz_utf8_invalid_k || candidate + rune_length != sequence_end) {
        it->ptr = sequence_end - 1;
        *out_rune = sz_rune_malformed_byte_(last_byte);
        it->pending_count = 0;
        it->pending_idx = 0;
        return sz_true_k;
    }
    it->ptr = candidate;

    // Store folded runes in pending buffer
    it->pending[0] = 0xFFFFFFFFu;
    it->pending[1] = 0xFFFFFFFEu;
    it->pending[2] = 0xFFFFFFFDu;
    it->pending[3] = 0xFFFFFFFCu;
    it->pending_count = sz_unicode_fold_codepoint_(rune, it->pending);
    it->pending_idx = 1; // We'll return the last one now, then the rest in subsequent calls

    // Return the LAST folded rune first (since we're going backwards)
    *out_rune = it->pending[it->pending_count - 1];
    return sz_true_k;
}

/**
 *  @brief Safety profile for a single character across all script paths.
 *
 *  A safety profile for a "needle" is a set of conditions that allow simpler haystack on-the-fly folding
 *  than the proper `sz_utf8_uncased_fold`, but without losing any possible matches. That's typically achieved
 *  finding parts of the needle, that never appear in any multi-byte expansions of complex characters, so
 *  we don't need to shuffle data within a CPU register - just swap some byte sequences with others.
 *
 *  Assuming the complexity of Unicode, the number of such rules to take care of is quite significant, so
 *  it's hard to achieve matching speeds beyond 500 MB/s for arbitrary needles. However, if separate them
 *  by language groups and Unicode subranges, the 5 GB/s target becomes approachable.
 */
typedef enum {
    sz_utf8_uncased_rune_unknown_k = 0,
    /**
     *  @brief Describes a safety-class profile for contextually-safe ASCII characters, mostly for English text,
     *         exclusive to single-byte characters without case-folding "collisions" and ambiguities.
     *
     *  If all of the following @b needle-constraints are satisfied, our uncased UTF-8 substring search
     *  becomes no more than a trivial uncased ASCII substring search, where the only @b haystack-folding
     *  operation to be applied is mapping A-Z to a-z:
     *
     *  - 'a' (U+0061, 61) - can't be last; can't precede 'ʾ' (U+02BE, CA BE) to avoid:
     *    - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
     *  - 'f' (U+0066, 66) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede 'f' (U+0066, 66), 'i' (U+0069, 69), 'l' (U+006C, 6C) to avoid:
     *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'h' (U+0068, 68) - can't be last; can't precede '̱' (U+0331, CC B1) to avoid:
     *    - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede '̇' (U+0307, CC 87) to avoid:
     *    - 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *  - 'j' (U+006A, 6A) - can't be last; can't precede '̌' (U+030C, CC 8C) to avoid:
     *    - 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C)
     *  - 'k' (U+006B, 6B) - can't be present at all, because it's a folding target of the Kelvin sign:
     *    - 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B)
     *  - 'l' (U+006C, 6C) - can't be first; can't follow 'f' (U+0066, 66) to avoid:
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'n' (U+006E, 6E) - can't be first; can't follow 'ʼ' (U+02BC, CA BC) to avoid:
     *    - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
     *  - 's' (U+0073, 73) - can't be present at all, because it's a folding target of the old S sign:
     *    - 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73);
     *    can't precede '̈' (U+0308, CC 88) to avoid:
     *    - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 'w' (U+0077, 77) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
     *  - 'y' (U+0079, 79) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
     *
     *  This means, that all ASCII characters beyond the rules above are considered "safe" for this profile.
     *  This includes English alphabet letters like: b, c, d, e, g, m, o, p, q, r, u, v, x, z,
     *  as well as digits, punctuation, symbols, and control characters.
     */
    sz_utf8_uncased_rune_ascii_invariant_k = 1,

    /**
     *  @brief Describes a safety-class profile for contextually-safe ASCII + Latin-1 Supplements designed mostly
     *         for Western European languages (like French, German, Spanish, & Portuguese) with a mixture of
     *         single-byte and double-byte UTF-8 character sequences.
     *
     *  @sa sz_utf8_uncased_rune_ascii_invariant_k for a simpler variant.
     *
     *  Unlike the ASCII fast path, these kernels fold a wider range of characters:
     *  - 26x original ASCII uppercase letters: 'A' (U+0041, 41) → 'a' (U+0061, 61), 'Z' (U+005A, 5A) → 'z' (U+007A, 7A)
     *  - 30x Latin-1 supplement uppercase letters for French, German, Spanish, & Portuguese, like:
     *    - 'À' (U+00C0, C3 80) → 'à' (U+00E0, C3 A0),
     *    - 'Ñ' (U+00D1, C3 91) → 'ñ' (U+00F1, C3 B1),
     *    - 'Ü' (U+00DC, C3 9C) → 'ü' (U+00FC, C3 BC)
     *  - 1x special case of folding from Latin-1 to ASCII pair, preserving byte-width:
     *    - 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
     *
     *  This doesn't cover Latin-A and Latin-B extensions (like Polish, Czech, Hungarian, & Turkish letters).
     *  This also inherits some of the contextual limitations from `sz_utf8_uncased_rune_ascii_invariant_k`, but not all!
     *
     *  The lowercase 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73) is folded in-place (2 bytes → 2 bytes).
     *  This creates a mid-expansion matching issue: if a needle starts or ends with 's', the SIMD kernel might find
     *  a match at the second byte of 'ß' (the UTF-8 continuation byte 0x9F) instead of at a codepoint boundary.
     *  Example: haystack "ßStra" folds to "ssstra", needle "sstra" matches at position 1 (the 0x9F byte of 'ß').
     *  To avoid this, 's' is only safe when NOT at the start or end of the needle (contextual restriction).
     *
     *  The uppercase 'ẞ' (U+1E9E, E1 BA 9E) also folds into "ss" (U+0073 U+0073, 73 73), but is outside of Latin-1.
     *  In UTF-8 it is a 3-byte sequence, so it resizes into a 2-byte sequence when folded. Luckily for us,
     *  it's almost never used in practice: introduced to Unicode in 2008 and officially adopted into German
     *  orthography in 2017. When processing the haystack, we check if 'ẞ' appears, and if so, we revert to
     *  serial processing for that tiny block of text.
     *
     *  Another place where 's' (U+0073, 73) appears are ligatures 'ﬅ' (U+FB05, EF AC 85) and 'ﬆ' (U+FB06, EF AC 86)
     *  that both fold into "st" (U+0073 U+0074, 73 74). They also result in serial fallback when detected in the
     *  haystack. If we detect all of those ligatures from 'ﬀ' (U+FB00, EF AC 80) to 'ﬆ' (U+FB06, EF AC 86), we can
     *  safely allow both 'f' (U+0066, 66) and 'l' (U+006C, 6C).
     *
     *  There is one more 3-byte problematic range to consider - from (E1 BA 96) to (E1 BA 9A), which includes:
     *  'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1), 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88),
     *  'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A), 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A),
     *  'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE). If we correctly detect that range in the haystack, we
     *  can safely allow 'h' (U+0068, 68), 't' (U+0074, 74), 'w' (U+0077, 77), 'y' (U+0079, 79), and 'a' (U+0061, 61) in
     *  needles!
     *
     *  There is also a Unicode rule for folding the Kelvin 'K' (U+212A, E2 84 AA) into 'k' (U+006B, 6B).
     *  That sign is extremely rare in Western European languages, while the lowercase 'k' is obviously common
     *  in German and English. In French, Spanish, and Portuguese - less so. So we add one more check
     *  for 'K' (U+212A, E2 84 AA) in the haystack, and if detected, again - revert to serial.
     *  Similarly, we check for 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73).
     *  It's archaic in modern languages but theoretically possible in historical texts.
     *
     *  So we allow 'k' unconditionally and inherit/extend the following limitations from
     *  `sz_utf8_uncased_rune_ascii_invariant_k`:
     *
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66); can't precede '̇' (U+0307, CC 87)
     *    to avoid 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87).
     *    It's the Turkish dotted capital I that expands into a 3-byte sequence when folded. It typically appears
     *    at the start of words, like: İstanbul (the city), İngilizce (English language).
     *
     *  - 'j' (U+006A, 6A) - can't be last; can't precede '̌' (U+030C, CC 8C)
     *    to avoid: 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C).
     *    It's the "J with Caron", used in phonetic transcripts and romanization of Iranian, Armenian, Georgian.
     *
     *  - 'n' (U+006E, 6E) - can't be first; can't follow 'ʼ' (U+02BC, CA BC)
     *    to avoid: 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E).
     *    It's mostly used in Afrikaans (South Africa/Namibia), contracted from Dutch "een" (one/a), in phrases
     *    like "Dit is 'n boom" (It is a tree), "Dit is 'n appel" (This is an apple).
     *
     *  - 's' (U+0073, 73) - can't be first or last, or part of the folded "ss" (U+0073 U+0073, 73 73) prefix or suffix,
     *    to avoid mid-ß-expansion matches: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73) is folded in-place,
     *    so a needle starting/ending with 's' could match at position 1 (the 0x9F continuation byte).
     *    Example: "ßStra" → "ssstra", needle "sstra" would match at the second byte of 'ß'.
     *    Needles with 's' in the middle are safe.
     *
     *  We also add one more limitation for a special 2-byte character that is an irregular folding target of
     *  codepoints of different length:
     *
     *  - 'å' (U+00E5, C3 A5) - is the folding target of both 'Å' (U+00C5, C3 85) in Latin-1 and
     *    the Angstrom Sign 'Å' (U+212B, E2 84 AB) → 'å' (U+00E5, C3 A5), so needle cannot contain 'å' (U+00E5, C3 A5)
     *    to avoid ambiguity.
     *
     *  There is also a Latin-1 character that doesn't change the width, but we still ban it from the safe strings:
     *
     *  - 'µ' (U+00B5, C2 B5) - the mathematical Micro sign folds to the Greek lowercase 'μ' (U+03BC, CE BC), which
     *    is also a folding target of the uppercase Greek letter 'Μ' (U+039C, CE 9C). To avoid having to filter/check
     *    for Greek symbols in the haystacks, we ban the Micro sign from the needles.
     *
     *  This means, that all of the ASCII and Latin-1 characters beyond the rules above are considered "safe"
     *  for this profile. This includes English alphabet letters like: b, c, d, e, g, k, m, o, p, q, r, u, v, x, z,
     *  as well as digits, punctuation, symbols, and control characters.
     */
    sz_utf8_uncased_rune_safe_western_europe_k = 2,

    /**
     *  @brief Describes a safety-class profile for contextually-safe ASCII + Latin-1 + Latin-A Supplements designed
     *         mostly for Central European languages (like Polish, Czech, & Hungarian) and Turkish with a mixture of
     *         single-byte, double-byte, and rare triple-byte UTF-8 character sequences.
     *
     *  @sa sz_utf8_uncased_rune_ascii_invariant_k for a simpler variant.
     *
     *  Unlike the ASCII fast path, these kernels fold a wider range of characters:
     *  - 26x original ASCII uppercase letters: 'A' (U+0041, 41) → 'a' (U+0061, 61), 'Z' (U+005A, 5A) → 'z' (U+007A, 7A)
     *  - 30x Latin-1 supplement uppercase letters for French, German, Spanish, & Portuguese, like:
     *    - 'À' (U+00C0, C3 80) → 'à' (U+00E0, C3 A0),
     *    - 'Ñ' (U+00D1, C3 91) → 'ñ' (U+00F1, C3 B1),
     *    - 'Ü' (U+00DC, C3 9C) → 'ü' (U+00FC, C3 BC)
     *  - 63x Latin-A extension uppercase letters for Polish, Czech, Hungarian, & Turkish, like:
     *    - 'Ą' (U+0104, C4 84) → 'ą' (U+0105, C4 85),
     *    - 'Ł' (U+0141, C5 81) → 'ł' (U+0142, C5 82),
     *    - 'Č' (U+010C, C4 8C) → 'č' (U+010D, C4 8D)
     *
     *  This doesn't cover Latin-B extensions (like Baltic, Romanian, & Vietnamese letters), and is not optimal
     *  for Western European languages, assuming the lack of "ss" handling for German Eszett 'ß' (U+00DF, C3 9F).
     *  There is, however, a huge overlap between the Central European, Western European, and Turkic scripts:
     *
     *  - Czech has the highest overlap - nearly half of Czech words with Latin-A characters (like Č, Ř, Š, Ž)
     *    also contain Latin-1 characters (Á, É, Í, Ó, Ú, Ý). Examples: sčítání, dalšími, řízení, systémů.
     *  - Polish has minimal word-level overlap because Polish only uses Ó/ó from Latin-1, and most Polish-specific
     *    letters (Ą, Ę, Ł, Ń, Ś, Ź, Ż) are in Latin-A. Example: mieszkańców (has both ń and ó).
     *  - Turkish has moderate overlap from Ç, Ö, Ü (Latin-1) mixing with Ğ, İ, Ş (Latin-A).
     *    Examples: içeriği, öğrencilerden, dönüşüm.
     *
     *  All those languages are not always related linguistically:
     *
     *  - Czech and Polish are Slavic languages, that use Latin script with háčeks since 15th century.
     *  - Hungarian is a Uralic language, that adopted Latin script in 11th century.
     *  - Turkish is a Turkic (Altaic) language, that switched from Arabic to Latin script in 1928.
     *    Atatürk's 1928 alphabet reform:
     *    - borrowed Ç, Ö, Ü from French and German subsets of Latin-1 Supplement (C3 lead byte).
     *    - introduced Ğ, İ, Ş, which ended up in the Latin Extended-A (C4/C5 lead byte).
     *
     *  But due to overlapping character sets, they can all benefit from the same fast path.
     *
     *  There is also a Unicode rule for folding the Kelvin 'K' (U+212A, E2 84 AA) into 'k' (U+006B, 6B).
     *  That sign is extremely rare in Western European languages, while the lowercase 'k' is very common in Turkish,
     *  Czech, Polish. So we add one more check for 'K' (U+212A, E2 84 AA) in the haystack, and if detected,
     *  again - revert to serial. Same logic applies to 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73).
     *
     *  The Turkish dotted 'İ' (U+0130, C4 B0) expands into a 3-byte sequence. We detect it when scanning through the
     *  haystack and fall back to the serial algorithm. That's pretty much the only triple-byte sequence we will
     *  frequently encounter in Turkish text.
     *
     *  We inherit most contextual limitations for some of the ASCII characters from
     *  `sz_utf8_uncased_rune_ascii_invariant_k`:
     *
     *  - 'a' (U+0061, 61) - can't be last; can't precede 'ʾ' (U+02BE, CA BE) to avoid:
     *    - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
     *  - 'f' (U+0066, 66) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede 'f' (U+0066, 66), 'i' (U+0069, 69), 'l' (U+006C, 6C) to avoid:
     *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'h' (U+0068, 68) - can't be last; can't precede '̱' (U+0331, CC B1) to avoid:
     *    - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede '̇' (U+0307, CC 87) to avoid:
     *    - 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *  - 'j' (U+006A, 6A) - can't be last; can't precede '̌' (U+030C, CC 8C) to avoid:
     *    - 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C)
     *  - 'l' (U+006C, 6C) - can't be first; can't follow 'f' (U+0066, 66) to avoid:
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'n' (U+006E, 6E) - can't be first; can't follow 'ʼ' (U+02BC, CA BC) to avoid:
     *    - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
     *  - 's' (U+0073, 73) - can't be first or last; can't follow 's' (U+0073, 73);
     *    can't precede 's' (U+0073, 73), 't' (U+0074, 74) to avoid:
     *    - 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
     *    - 'ẞ' (U+1E9E, E1 BA 9E) → "ss" (U+0073 U+0073, 73 73)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73);
     *    can't precede '̈' (U+0308, CC 88) to avoid:
     *    - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 'w' (U+0077, 77) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
     *  - 'y' (U+0079, 79) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
     *
     *  We also inherit one more limitation from the Latin-1 profile, same as `sz_utf8_uncased_rune_safe_western_europe_k`:
     *
     *  - 'å' (U+00E5, C3 A5) - is the folding target of both 'Å' (U+00C5, C3 85) in Latin-1 and
     *    the Angstrom Sign 'Å' (U+212B, E2 84 AB) → 'å' (U+00E5, C3 A5), so needle cannot contain 'å' (U+00E5, C3 A5)
     *    to avoid ambiguity.
     *
     *  This means, that all of the ASCII and Latin-1 characters beyond the rules above are considered "safe"
     *  for this profile. This includes English alphabet letters like: b, c, d, e, g, k, m, o, p, q, r, u, v, x, z,
     *  as well as digits, punctuation, symbols, and control characters.
     */
    sz_utf8_uncased_rune_safe_central_europe_k = 3,

    /**
     *  @brief Describes a safety-class profile for contextually-safe ASCII + Basic Cyrillic designed mostly
     *         for East Slavic languages (like Russian, Ukrainian, & Belarusian) and South Slavic languages
     *         (like Serbian, Bulgarian, & Macedonian), but excluding Cyrillic Extensions.
     *
     *  @sa sz_utf8_uncased_rune_ascii_invariant_k for the inherited ASCII rules.
     *
     *  Unlike the ASCII fast path, these kernels fold a wider range of characters:
     *  - 26x original ASCII uppercase letters: 'A' (U+0041, 41) → 'a' (U+0061, 61), 'Z' (U+005A, 5A) → 'z' (U+007A, 7A)
     *  - 32x Basic Cyrillic uppercase letters:
     *    - 'А' (U+0410, D0 90) → 'а' (U+0430, D0 B0) through 'П' (U+041F, D0 9F) → 'п' (U+043F, D0 BF)
     *    - 'Р' (U+0420, D0 A0) → 'р' (U+0440, D1 80) through 'Я' (U+042F, D0 AF) → 'я' (U+044F, D1 8F)
     *  - 16x Cyrillic extensions for non-Russian Slavic languages:
     *    - 'Ѐ' (U+0400, D0 80) → 'ѐ' (U+0450, D1 90) - Cyrillic E with grave (Macedonian, Serbian)
     *    - 'Ё' (U+0401, D0 81) → 'ё' (U+0451, D1 91) - Cyrillic IO (Russian, Belarusian)
     *    - 'Ђ' (U+0402, D0 82) → 'ђ' (U+0452, D1 92) - Cyrillic DJE (Serbian)
     *    - 'Ѓ' (U+0403, D0 83) → 'ѓ' (U+0453, D1 93) - Cyrillic GJE (Macedonian)
     *    - 'Є' (U+0404, D0 84) → 'є' (U+0454, D1 94) - Cyrillic Ukrainian IE (Ukrainian)
     *    - 'Ѕ' (U+0405, D0 85) → 'ѕ' (U+0455, D1 95) - Cyrillic DZE (Macedonian)
     *    - 'І' (U+0406, D0 86) → 'і' (U+0456, D1 96) - Cyrillic Byelorussian-Ukrainian I (Ukrainian, Belarusian)
     *    - 'Ї' (U+0407, D0 87) → 'ї' (U+0457, D1 97) - Cyrillic YI (Ukrainian)
     *    - 'Ј' (U+0408, D0 88) → 'ј' (U+0458, D1 98) - Cyrillic JE (Serbian, Macedonian)
     *    - 'Љ' (U+0409, D0 89) → 'љ' (U+0459, D1 99) - Cyrillic LJE (Serbian, Macedonian)
     *    - 'Њ' (U+040A, D0 8A) → 'њ' (U+045A, D1 9A) - Cyrillic NJE (Serbian, Macedonian)
     *    - 'Ћ' (U+040B, D0 8B) → 'ћ' (U+045B, D1 9B) - Cyrillic TSHE (Serbian)
     *    - 'Ќ' (U+040C, D0 8C) → 'ќ' (U+045C, D1 9C) - Cyrillic KJE (Macedonian)
     *    - 'Ѝ' (U+040D, D0 8D) → 'ѝ' (U+045D, D1 9D) - Cyrillic I with grave (Bulgarian, Macedonian)
     *    - 'Ў' (U+040E, D0 8E) → 'ў' (U+045E, D1 9E) - Cyrillic short U (Belarusian)
     *    - 'Џ' (U+040F, D0 8F) → 'џ' (U+045F, D1 9F) - Cyrillic DZHE (Serbian, Macedonian)
     *
     *  UTF-8 byte patterns for Basic Cyrillic (D0/D1 lead bytes):
     *  - D0 80-8F: Extensions uppercase 'Ѐ'-'Џ' (U+0400-U+040F) → fold to D1 90-9F
     *  - D0 90-9F: Basic uppercase 'А'-'П' (U+0410-U+041F) → fold to D0 B0-BF (same lead byte)
     *  - D0 A0-AF: Basic uppercase 'Р'-'Я' (U+0420-U+042F) → fold to D1 80-8F (cross lead byte)
     *  - D0 B0-BF: Basic lowercase 'а'-'п' (U+0430-U+043F)
     *  - D1 80-8F: Basic lowercase 'р'-'я' (U+0440-U+044F)
     *  - D1 90-9F: Extensions lowercase 'ѐ'-'џ' (U+0450-U+045F)
     *
     *  We entirely ban all of the Extended Cyrillic (D2/D3 lead bytes), sometimes used in Ukranian,
     *  Kazakh, and Uzbek languages, like the 'Ґ' (U+0490, D2 90) → 'ґ' (U+0491, D2 91) folding with even/odd ordering
     *  of uppercase and lowercase. Similar rules apply to some Chechen, and various Turkic languages.
     *  But there are also exceptions, like the Palochka 'Ӏ' (U+04C0, D3 80) → 'ӏ' (U+04CF, D3 8F).
     *  By omitting those extensions we can make our folding kernel much lighter.
     *
     *  We inherit ALL contextual ASCII limitations from `sz_utf8_uncased_rune_ascii_invariant_k`:
     *
     *  - 'a' (U+0061, 61) - can't be last; can't precede 'ʾ' (U+02BE, CA BE) to avoid:
     *     - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
     *  - 'f' (U+0066, 66) - can't be first or last; can't follow 'f' (U+0066, 66);
     *     can't precede 'f' (U+0066, 66), 'i' (U+0069, 69), 'l' (U+006C, 6C) to avoid:
     *     - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
     *     - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *     - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *     - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *     - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'h' (U+0068, 68) - can't be last; can't precede '̱' (U+0331, CC B1) to avoid:
     *     - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66);
     *     can't precede '̇' (U+0307, CC 87) to avoid:
     *     - 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
     *     - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *     - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *  - 'j' (U+006A, 6A) - can't be last; can't precede '̌' (U+030C, CC 8C) to avoid:
     *     - 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C)
     *  - 'k' (U+006B, 6B) - can't be present at all, because it's a folding target of the Kelvin sign:
     *     - 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B)
     *  - 'l' (U+006C, 6C) - can't be first; can't follow 'f' (U+0066, 66) to avoid:
     *     - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *     - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'n' (U+006E, 6E) - can't be first; can't follow 'ʼ' (U+02BC, CA BC) to avoid:
     *     - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
     *  - 's' (U+0073, 73) - can't be present at all, because it's a folding target of the old S sign:
     *    - 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73);
     *     can't precede '̈' (U+0308, CC 88) to avoid:
     *     - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *     - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *     - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73);
     *     can't precede '̈' (U+0308, CC 88) to avoid:
     *     - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *     - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *     - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 'w' (U+0077, 77) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *     - 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
     *  - 'y' (U+0079, 79) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *     - 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
     *
     *  These ASCII constraints are necessary because mixed-script documents (Cyrillic + Latin) may contain
     *  Latin ligatures, German Eszett, or Turkish İ that the Cyrillic fold function doesn't handle.
     *
     *  This means, that all ASCII characters beyond the rules above are considered "safe" for this profile.
     *  This includes English alphabet letters like: b, c, d, e, g, m, o, p, q, r, u, v, x, z,
     *  as well as digits, punctuation, symbols, and control characters.
     */
    sz_utf8_uncased_rune_safe_cyrillic_k = 4,

    /**
     *  @brief Describes a safety-class profile for contextually-safe ASCII + Basic Greek designed mostly
     *         for Modern Greek (Demotic) text with a mixture of single-byte and double-byte UTF-8
     *         character sequences.
     *
     *  @sa sz_utf8_uncased_rune_ascii_invariant_k for the inherited ASCII rules.
     *
     *  Unlike the ASCII fast path, these kernels fold a wider range of characters:
     *  - 26x original ASCII uppercase letters: 'A' (U+0041, 41) → 'a' (U+0061, 61), 'Z' (U+005A, 5A) → 'z' (U+007A, 7A)
     *  - 24x Basic Greek uppercase letters (monotonic, without diacritics):
     *    - 'Α' (U+0391, CE 91) → 'α' (U+03B1, CE B1) through 'Ο' (U+039F, CE 9F) → 'ο' (U+03BF, CE BF)
     *    - 'Π' (U+03A0, CE A0) → 'π' (U+03C0, CF 80) through 'Ω' (U+03A9, CE A9) → 'ω' (U+03C9, CF 89)
     *  - 1x Final sigma to regular sigma:
     *    - 'ς' (U+03C2, CF 82) → 'σ' (U+03C3, CF 83)
     *  - 7x Greek accented uppercase letters (tonos only, modern orthography):
     *    - 'Ά' (U+0386, CE 86) → 'ά' (U+03AC, CE AC)
     *    - 'Έ' (U+0388, CE 88) → 'έ' (U+03AD, CE AD)
     *    - 'Ή' (U+0389, CE 89) → 'ή' (U+03AE, CE AE)
     *    - 'Ί' (U+038A, CE 8A) → 'ί' (U+03AF, CE AF)
     *    - 'Ό' (U+038C, CE 8C) → 'ό' (U+03CC, CF 8C)
     *    - 'Ύ' (U+038E, CE 8E) → 'ύ' (U+03CD, CF 8D)
     *    - 'Ώ' (U+038F, CE 8F) → 'ώ' (U+03CE, CF 8E)
     *  - 2x Greek uppercase letters with dialytika:
     *    - 'Ϊ' (U+03AA, CE AA) → 'ϊ' (U+03CA, CF 8A)
     *    - 'Ϋ' (U+03AB, CE AB) → 'ϋ' (U+03CB, CF 8B)
     *
     *  UTF-8 byte patterns for Basic Greek (CE/CF lead bytes):
     *  - CE 86-8F: Accented uppercase 'Ά'-'Ώ' (with gaps) → CE AC-AF or CF 8C-8E
     *  - CE 91-9F: Basic uppercase 'Α'-'Ο' (U+0391-U+039F) → CE B1-BF (same lead byte)
     *  - CE A0-A9: Basic uppercase 'Π'-'Ω' (U+03A0-U+03A9) → CF 80-89 (cross lead byte)
     *  - CE AA-AB: Dialytika uppercase 'Ϊ'-'Ϋ' (U+03AA-U+03AB) → CF 8A-8B (cross lead byte)
     *  - CE AC-AF: Accented lowercase 'ά'-'ί' (U+03AC-U+03AF)
     *  - CE B1-BF: Basic lowercase 'α'-'ο' (U+03B1-U+03BF)
     *  - CF 80-89: Basic lowercase 'π'-'ω' (U+03C0-U+03C9), includes 'ς' (U+03C2, CF 82) and 'σ' (U+03C3, CF 83)
     *  - CF 8A-8E: Accented/dialytika lowercase 'ϊ'-'ώ' (U+03CA-U+03CE)
     *
     *  Greek symbol variants that fold to basic letters (detected in haystack, serial fallback):
     *  - 'ϐ' (U+03D0, CF 90) → 'β' (U+03B2, CE B2) - Greek Beta Symbol
     *  - 'ϑ' (U+03D1, CF 91) → 'θ' (U+03B8, CE B8) - Greek Theta Symbol
     *  - 'ϕ' (U+03D5, CF 95) → 'φ' (U+03C6, CF 86) - Greek Phi Symbol
     *  - 'ϖ' (U+03D6, CF 96) → 'π' (U+03C0, CF 80) - Greek Pi Symbol
     *  - 'ϰ' (U+03F0, CF B0) → 'κ' (U+03BA, CE BA) - Greek Kappa Symbol
     *  - 'ϱ' (U+03F1, CF B1) → 'ρ' (U+03C1, CF 81) - Greek Rho Symbol
     *  - 'ϵ' (U+03F5, CF B5) → 'ε' (U+03B5, CE B5) - Greek Lunate Epsilon Symbol
     *
     *  Excluded from the needle (require serial fallback when detected in haystack):
     *
     *  - 'ΐ' (U+0390, CE 90) → "ΐ" (U+03B9 U+0308 U+0301, CE B9 CC 88 CC 81) - iota with dialytika and tonos
     *    EXPANDS to "ΐ" (U+03B9 U+0308 U+0301) - 3 codepoints!
     *  - 'ΰ' (U+03B0, CE B0) → "ΰ" (U+03C5 U+0308 U+0301, CF 85 CC 88 CC 81) - upsilon with dialytika and tonos
     *    EXPANDS to "ΰ" (U+03C5 U+0308 U+0301) - 3 codepoints!
     *  - Greek Extended / Polytonic (U+1F00-U+1FFF, E1 BC-BF lead bytes):
     *    Ancient Greek with breathing marks, accents, and iota subscript. Many expand to multiple
     *    codepoints, e.g., 'ᾈ' (U+1F88) → "ἀι" (U+1F00 U+03B9, E1 BC 80 CE B9), 'ᾳ' (U+1FB3) → "αι" (U+03B1 U+03B9, CE
     * B1 CE B9). Polytonic Greek is used primarily in academic, religious, and historical texts.
     *
     *  Note on the Micro Sign 'µ' (U+00B5, C2 B5):
     *  The Latin-1 micro sign folds TO Greek mu 'μ' (U+03BC, CE BC). This is handled by the Latin-1
     *  kernel path (sz_utf8_uncased_rune_safe_western_europe_k), not the Greek path. The Greek kernel
     *  only handles characters that originate in the Greek block.
     *
     *  We inherit @b all contextual ASCII limitations from `sz_utf8_uncased_rune_ascii_invariant_k`:
     *
     *  - 'a' (U+0061, 61) - can't be last; can't precede 'ʾ' (U+02BE, CA BE) to avoid:
     *    - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
     *  - 'f' (U+0066, 66) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede 'f' (U+0066, 66), 'i' (U+0069, 69), 'l' (U+006C, 6C) to avoid:
     *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'h' (U+0068, 68) - can't be last; can't precede '̱' (U+0331, CC B1) to avoid:
     *    - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede '̇' (U+0307, CC 87) to avoid:
     *    - 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *  - 'j' (U+006A, 6A) - can't be last; can't precede '̌' (U+030C, CC 8C) to avoid:
     *    - 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C)
     *  - 'k' (U+006B, 6B) - can't be present at all, because it's a folding target of the Kelvin sign:
     *    - 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B)
     *  - 'l' (U+006C, 6C) - can't be first; can't follow 'f' (U+0066, 66) to avoid:
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'n' (U+006E, 6E) - can't be first; can't follow 'ʼ' (U+02BC, CA BC) to avoid:
     *    - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
     *  - 's' (U+0073, 73) - can't be present at all, because it's a folding target of the old S sign:
     *    - 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73);
     *    can't precede '̈' (U+0308, CC 88) to avoid:
     *    - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73);
     *    can't precede '̈' (U+0308, CC 88) to avoid:
     *    - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 'w' (U+0077, 77) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
     *  - 'y' (U+0079, 79) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
     *
     *  These ASCII constraints are necessary because mixed-script documents (Greek + Latin) are common
     *  in scientific notation, brand names, and modern Greek text with English loanwords.
     *
     *  This means, that all ASCII characters beyond the rules above are considered "safe" for this profile.
     *  This includes English alphabet letters like: b, c, d, e, g, m, o, p, q, r, u, v, x, z,
     *  as well as digits, punctuation, symbols, and control characters.
     */
    sz_utf8_uncased_rune_safe_greek_k = 5,

    /**
     *  @brief Describes a safety-class profile for contextually-safe ASCII + Basic Armenian.
     *  @sa sz_utf8_uncased_rune_ascii_invariant_k for the inherited ASCII rules.
     *
     *  These kernels fold:
     *  - 26x ASCII uppercase letters: 'A' (U+0041, 41) → 'a' (U+0061, 61), 'Z' (U+005A, 5A) → 'z' (U+007A, 7A)
     *  - 38x Armenian uppercase letters: '
     *    - 'Ա' (U+0531, D4 B1) → 'ա' (U+0561, D5 A1)
     *    - 'Ֆ' (U+0556, D5 96) → 'ֆ' (U+0586, D6 86)
     *
     *  UTF-8 byte ranges handled:
     *  - D4 B1-BF: uppercase 'Ա' (U+0531) through 'Ձ' (U+053F)
     *  - D5 80-96: uppercase 'Ղ' (U+0540) through 'Ֆ' (U+0556)
     *  - D5 A1-BF: lowercase 'ա' (U+0561) through 'ի' (U+057F)
     *  - D6 80-86: lowercase 'լ' (U+0580) through 'ֆ' (U+0586)
     *
     *  We inherit @b all contextual ASCII limitations from `sz_utf8_uncased_rune_ascii_invariant_k`:
     *
     *  - 'a' (U+0061, 61) - can't be last; can't precede 'ʾ' (U+02BE, CA BE) to avoid:
     *    - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
     *  - 'f' (U+0066, 66) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede 'f' (U+0066, 66), 'i' (U+0069, 69), 'l' (U+006C, 6C) to avoid:
     *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'h' (U+0068, 68) - can't be last; can't precede '̱' (U+0331, CC B1) to avoid:
     *    - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede '̇' (U+0307, CC 87) to avoid:
     *    - 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *  - 'j' (U+006A, 6A) - can't be last; can't precede '̌' (U+030C, CC 8C) to avoid:
     *    - 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C)
     *  - 'k' (U+006B, 6B) - can't be present at all, because it's a folding target of the Kelvin sign:
     *    - 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B)
     *  - 'l' (U+006C, 6C) - can't be first; can't follow 'f' (U+0066, 66) to avoid:
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'n' (U+006E, 6E) - can't be first; can't follow 'ʼ' (U+02BC, CA BC) to avoid:
     *    - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
     *  - 's' (U+0073, 73) - can't be present at all, because it's a folding target of the old S sign:
     *    - 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73);
     *    can't precede '̈' (U+0308, CC 88) to avoid:
     *    - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73);
     *    can't precede '̈' (U+0308, CC 88) to avoid:
     *    - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 'w' (U+0077, 77) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
     *  - 'y' (U+0079, 79) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
     *
     *  We also add rules specific to Armenian ligatures:
     *
     *  - 'և' (U+0587, D6 87) → "եւ" (U+0565 U+0582, D5 A5 D6 82) - very common
     *  - 'ﬓ' (U+FB13, EF AC 93) → "մն" (U+0574 U+0576, D5 B4 D5 B6) - quite rare
     *  - 'ﬔ' (U+FB14, EF AC 94) → "մե" (U+0574 U+0565, D5 B4 D5 A5) - quite rare
     *  - 'ﬕ' (U+FB15, EF AC 95) → "մի" (U+0574 U+056B, D5 B4 D5 AB) - quite rare
     *  - 'ﬖ' (U+FB16, EF AC 96) → "վն" (U+057E U+0576, D5 BE D5 B6) - quite rare
     *  - 'ﬗ' (U+FB17, EF AC 97) → "մխ" (U+0574 U+056D, D5 B4 D5 AD) - quite rare
     *
     *  Specific constraints by character:
     *
     *  - 'ե' (U+0565, D5 A5) - can't be first; can't follow 'մ' (U+0574, D5 B4);
     *     can't precede 'ւ' (U+0582, D6 82) to avoid:
     *     - 'և' (U+0587, D6 87) → "եւ" (U+0565 U+0582, D5 A5 D6 82)
     *     - 'ﬔ' (U+FB14, EF AC 94) → "մե" (U+0574 U+0565, D5 B4 D5 A5)
     *  - 'ւ' (U+0582, D6 82) - can't be last; can't follow 'ե' (U+0565, D5 A5) to avoid:
     *     - 'և' (U+0587, D6 87) → "եւ" (U+0565 U+0582, D5 A5 D6 82)
     *  - 'մ' (U+0574, D5 B4) - can't be last; can't precede 'ն' (U+0576, D5 B6), 'ե' (U+0565, D5 A5),
     *     'ի' (U+056B, D5 AB), 'խ' (U+056D, D5 AD) to avoid:
     *     - 'ﬓ' (U+FB13, EF AC 93) → "մն" (U+0574 U+0576, D5 B4 D5 B6)
     *     - 'ﬔ' (U+FB14, EF AC 94) → "մե" (U+0574 U+0565, D5 B4 D5 A5)
     *     - 'ﬕ' (U+FB15, EF AC 95) → "մի" (U+0574 U+056B, D5 B4 D5 AB)
     *     - 'ﬗ' (U+FB17, EF AC 97) → "մխ" (U+0574 U+056D, D5 B4 D5 AD)
     *  - 'ն' (U+0576, D5 B6) - can't be first; can't follow 'մ' (U+0574, D5 B4), 'վ' (U+057E, D5 BE) to avoid:
     *     - 'ﬓ' (U+FB13, EF AC 93) → "մն" (U+0574 U+0576, D5 B4 D5 B6)
     *     - 'ﬖ' (U+FB16, EF AC 96) → "վն" (U+057E U+0576, D5 BE D5 B6)
     *  - 'ի' (U+056B, D5 AB) - can't be first; can't follow 'մ' (U+0574, D5 B4) to avoid:
     *     - 'ﬕ' (U+FB15, EF AC 95) → "մի" (U+0574 U+056B, D5 B4 D5 AB)
     *  - 'վ' (U+057E, D5 BE) - can't be first; can't precede 'ն' (U+0576, D5 B6) to avoid:
     *     - 'ﬖ' (U+FB16, EF AC 96) → "վն" (U+057E U+0576, D5 BE D5 B6)
     *  - 'խ' (U+056D, D5 AD) - can't be first; can't follow 'մ' (U+0574, D5 B4) to avoid:
     *     - 'ﬗ' (U+FB17, EF AC 97) → "մխ" (U+0574 U+056D, D5 B4 D5 AD)
     *
     *  This means that Armenian needles containing these specific bigrams (եւ, մն, մե, մի, վն, մխ)
     *  cannot use the fast path because finding them separately might miss the precomposed ligatures
     *  present in the haystack.
     */
    sz_utf8_uncased_rune_safe_armenian_k = 6,

    /**
     *  @brief Describes a safety-class profile for contextually-safe ASCII + Latin-1 + Latin Extended Additional.
     *  @sa sz_utf8_uncased_rune_safe_central_europe_k for the inherited Latin rules.
     *
     *  These kernels extend Latin-1/A/B with Vietnamese characters:
     *  - Everything from `sz_utf8_uncased_rune_safe_central_europe_k` (ASCII + Latin-1/A)
     *  - 166x Latin Extended Additional letters (U+1E00-U+1E95, U+1EA0-U+1EFF) for Vietnamese.
     *    Include precomposed Latin letters with additional diacritics (e.g. Ạ/ạ, Ả/ả, Ấ/ấ).
     *
     *  UTF-8 byte ranges handled:
     *  - 00-7F: ASCII, e.g. 'a' (U+0061, 61)
     *  - C2/C3: Latin-1 Supplement, e.g. 'â' (U+00E2, C3 A2)
     *  - C4-C5: Latin Extended-A, e.g. 'đ' (U+0111, C4 91)
     *  - C6: Latin Extended-B (for ơ, ư), e.g. 'ơ' (U+01A1, C6 A1)
     *  - E1 B8 80 - E1 BA 95: Latin Extended Additional (U+1E00-U+1E95), e.g. 'Ḁ' (U+1E00, E1 B8 80)
     *  - E1 BA A0 - E1 BB BF: Latin Extended Additional (U+1EA0-U+1EFF), e.g. 'ạ' (U+1EA1, E1 BA A1)
     *
     *  There is also a Unicode rule for folding the Kelvin 'K' (U+212A, E2 84 AA) into 'k' (U+006B, 6B).
     *  That sign is extremely rare, while the lowercase 'k' is common in Vietnamese (e.g. "kem", "kéo").
     *  So we add one more check for 'K' (U+212A, E2 84 AA) in the haystack, and if detected, again - revert to serial.
     *
     *  We inherit most contextual limitations for some of the ASCII characters from
     * `sz_utf8_uncased_rune_ascii_invariant_k`:
     *
     *  - 'a' (U+0061, 61) - can't be last; can't precede 'ʾ' (U+02BE, CA BE) to avoid:
     *    - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
     *  - 'f' (U+0066, 66) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede 'f' (U+0066, 66), 'i' (U+0069, 69), 'l' (U+006C, 6C) to avoid:
     *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'h' (U+0068, 68) - can't be last; can't precede '̱' (U+0331, CC B1) to avoid:
     *    - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede '̇' (U+0307, CC 87) to avoid:
     *    - 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *  - 'j' (U+006A, 6A) - can't be last; can't precede '̌' (U+030C, CC 8C) to avoid:
     *    - 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C)
     *  - 'l' (U+006C, 6C) - can't be first; can't follow 'f' (U+0066, 66) to avoid:
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'n' (U+006E, 6E) - can't be first; can't follow 'ʼ' (U+02BC, CA BC) to avoid:
     *    - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
     *  - 's' (U+0073, 73) - can't be first or last; can't follow 's' (U+0073, 73);
     *    can't precede 's' (U+0073, 73), 't' (U+0074, 74) to avoid:
     *    - 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
     *    - 'ẞ' (U+1E9E, E1 BA 9E) → "ss" (U+0073 U+0073, 73 73)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *    - 'ẛ' (U+1E9B, E1 BA 9B) → 'ṡ' (U+1E61, E1 B9 A1) [Latin Extended Additional]
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73);
     *    can't precede '̈' (U+0308, CC 88) to avoid:
     *    - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 'w' (U+0077, 77) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
     *  - 'y' (U+0079, 79) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
     *
     *  We also inherit one more limitation from the Latin-1 profile:
     *
     *  - 'å' (U+00E5, C3 A5) - is the folding target of both 'Å' (U+00C5, C3 85) in Latin-1 and
     *    the Angstrom Sign 'Å' (U+212B, E2 84 AB) → 'å' (U+00E5, C3 A5), so needle cannot contain 'å' (U+00E5, C3 A5)
     *    to avoid ambiguity.
     *
     *  This means, that all other ASCII and Latin-1/A/Ext-Add characters are "safe" to use with this kernel.
     */
    sz_utf8_uncased_rune_safe_vietnamese_k = 7,

    /**
     *  @brief Describes a safety-class profile for Georgian Mkhedruli script.
     *  @sa sz_utf8_uncased_rune_ascii_invariant_k for inherited ASCII rules.
     *
     *  Georgian Mkhedruli (U+10D0-U+10FF) is caseless - no folding needed for Georgian characters.
     *  Only ASCII A-Z folding for mixed text. Mtavruli (U+1C90-U+1CBF), Asomtavruli (U+10A0-U+10C5),
     *  and Nuskhuri (U+2D00-U+2D25) trigger alarm for serial fallback (rare in modern text).
     *
     *  All Georgian scripts use 3-byte UTF-8 sequences and fold to 3-byte sequences, so there
     *  are no length changes during case folding - making this the simplest non-ASCII kernel.
     */
    sz_utf8_uncased_rune_safe_georgian_k = 8,

    sz_utf8_uncased_rune_invariant_k = 9,
    sz_utf8_uncased_rune_fallback_serial_k = 255,
} sz_utf8_uncased_rune_safety_profile_t_;

#pragma endregion // Unicode Core

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_RUNES_H_
