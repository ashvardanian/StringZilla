/**
 *  @brief  Hardware-accelerated UTF-8 text processing utilities, that require unpacking into UTF-32 runes.
 *  @file   utf8_unpack.h
 *  @author Ash Vardanian
 *
 *  Work in progress:
 *
 *  - `sz_utf8_case_fold` - Unicode case folding for codepoints
 *  - `sz_utf8_find_case_insensitive` - case-insensitive substring search in UTF-8 strings
 *  - `sz_utf8_unpack_chunk` - convert UTF-8 to UTF-32 in a streaming manner
 */
#ifndef STRINGZILLA_UTF8_UNPACK_H_
#define STRINGZILLA_UTF8_UNPACK_H_

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief  Unpack a UTF-8 string into UTF-32 codepoints.
 *
 *  This function is designed for streaming-like decoding with smart iterators built on top of it.
 *  The iterator would unpack a continuous slice of UTF-8 text into UTF-32 codepoints in chunks,
 *  yielding them upstream - only one at a time. This avoids allocating large buffers for the entire
 *  UTF-32 string, which can be 4x the size of the UTF-8 input.
 *
 *  This functionality is similar to the `simdutf` library's UTF-8 to UTF-32 conversion routines,
 *  but unlike most of them - performs no validity checks, and leverages an assumption that absolute
 *  majority of written text doesn't mix codepoints of every length in each register-sized chunk.
 *
 *  - English text and source code is predominantly 1-byte ASCII characters.
 *  - Broader European languages with diacritics mostly use 2-byte characters with 1-byte punctuation.
 *  - Chinese & Jamapanese mostly use 3-byte characters with rare punctuation, which can be 1- or 3-byte.
 *  - Korean uses 3-byte characters with 1-byte spaces; word are 2-6 syllables or 6-16 bytes.
 *
 *  It's a different story for emoji-heavy texts, which can mix 4-byte characters more frequently.
 *
 *  @param[in] text UTF-8 string to unpack.
 *  @param[in] length Number of bytes in the string (up to 64).
 *  @param[out] runes Output buffer for UTF-32 codepoints (recommended to be at least @b 64 entries wide).
 *  @param[in] runes_capacity Capacity of the @p runes buffer (number of sz_rune_t entries).
 *  @param[out] runes_unpacked Number of runes unpacked.
 *  @return Pointer to the byte after the last unpacked byte in @p text.
 */
SZ_DYNAMIC sz_cptr_t sz_utf8_unpack_chunk(      //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked);

/**
 *  @brief  Apply Unicode case folding to a UTF-8 string.
 *
 *  Case folding normalizes text for case-insensitive comparisons by mapping uppercase letters
 *  to their lowercase equivalents and handling special expansions defined in Unicode CaseFolding.txt.
 *
 *  @section Buffer Sizing
 *
 *  The destination buffer must be at least `source_length * 3` bytes to guarantee sufficient space
 *  for worst-case expansion. The maximum expansion ratio is 3:1 (3x), which occurs with Greek
 *  characters that expand to three codepoints under case folding.
 *
 *  Worst-case example: U+0390 (2 bytes: CE 90) expands to U+03B9 + U+0308 + U+0301 (6 bytes total).
 *  A string of N such characters would expand from 2N to 6N bytes (3x expansion).
 *
 *  @param[in] source UTF-8 string to be case-folded.
 *  @param[in] source_length Number of bytes in the source buffer.
 *  @param[out] destination Buffer to write the case-folded UTF-8 string.
 *  @return Number of bytes written to the destination buffer.
 *
 *  @warning The caller must ensure the destination buffer is large enough. No bounds checking
 *           is performed. Use `source_length * 3` for safety.
 *  @warning The source must contain valid UTF-8. Behavior is undefined for invalid input.
 *
 *  @example Basic usage:
 *  @code
 *      char const *source = "HELLO";
 *      sz_size_t capacity = 5 * 3; // Safe overestimate
 *      char destination[15];
 *      sz_size_t result_length = sz_utf8_case_fold(source, 5, destination);
 *      // destination now contains "hello", result_length = 5
 *  @endcode
 */
SZ_DYNAMIC sz_size_t sz_utf8_case_fold(        //
    sz_cptr_t source, sz_size_t source_length, //
    sz_ptr_t destination);

/**
 *  @brief  Case-insensitive substring search in UTF-8 strings.
 *
 *  In applications where the haystack remains largely static and memory/storage is cheap, it is recommended
 *  to pre-process the haystack into a case-folded version using Unicode Case Folding (e.g., via the ICU
 *  library) and subsequently use the simpler `sz_find()` function for repeated searches. This avoids the cost
 *  of performing full folding logic during every search operation.
 *
 *  This function applies full Unicode Case Folding as defined in the Unicode Standard (UAX #21 and
 *  CaseFolding.txt), covering all bicameral scripts, all offset-based one-to-one folds, all table-based
 *  one-to-one folds, and all normative one-to-many expansions.
 *
 *  The following character mappings are supported:
 *
 *  - ASCII Latin letters A–Z (U+0041–U+005A) are folded to a–z (U+0061–U+007A) using a trivial +32 offset.
 *  - Fullwidth Latin letters Ａ–Ｚ (U+FF21–U+FF3A) are folded to ａ–ｚ (U+FF41–U+FF5A) with the same +32 offset.
 *  - Cyrillic uppercase А–Я (U+0410–U+042F) are folded to а–я (U+0430–U+044F) using a +32 offset.
 *  - Armenian uppercase Ա–Ֆ (U+0531–U+0556) are folded to ա–ֆ (U+0561–U+0586) using a +48 offset.
 *  - Georgian Mtavruli letters Ა-Ჿ (U+1C90–U+1CBF, excluding 2) are folded to their Mkhedruli equivalents
 *    (U+10D0–U+10FF) using a fixed linear translation defined by the Unicode Standard.
 *  - Greek uppercase Α–Ω (U+0391–U+03A9) are folded to α–ω (U+03B1–U+03C9) via a +32 offset, with a normative
 *    context-sensitive rule for sigma: Σ (U+03A3) folds to σ (U+03C3) or ς (U+03C2) depending on word-final
 *    position, as required by SpecialCasing.txt.
 *  - Latin Extended characters include numerous one-to-one folds and several one-to-many expansions, including:
 *      ß  (U+00DF) → "ss" (U+0073 U+0073)
 *      ẞ  (U+1E9E) → "ss"
 *    as well as mixed-case digraphs and trigraphs normalized to lowercase sequences.
 *  - Turkish and Azerbaijani dotted/dotless-I rules follow SpecialCasing.txt, including:
 *      İ (U+0130)  → "i̇" (U+0069 U+0307)
 *      I (U+0049)  →  i   (U+0069)
 *      ı (U+0131)  →  ı   (already lowercase)
 *    with full locale-correct behavior.
 *  - Lithuanian accented I/J mappings that require combining-dot additions or removals are processed
 *    as multi-codepoint expansions exactly as specified in SpecialCasing.txt.
 *  - Additional bicameral scripts—Cherokee, Deseret, Osage, Warang Citi, Adlam—use their normative
 *    one-to-one uppercase-to-lowercase mappings defined in CaseFolding.txt.
 *
 *  Folding is applied during matching without rewriting the entire haystack. Multi-codepoint expansions,
 *  contextual folds, and combining-mark adjustments are handled at comparison time.
 *
 *  @param[in] haystack UTF-8 string to be searched.
 *  @param[in] haystack_length Number of bytes in the haystack buffer.
 *  @param[in] needle UTF-8 substring to search for.
 *  @param[in] needle_length Number of bytes in the needle substring.
 *  @param[out] matched_length Number of bytes in the matched region.
 *  @return Pointer to the first matching substring from @p haystack, or @c SZ_NULL_CHAR if not found.
 */
SZ_DYNAMIC sz_cptr_t sz_utf8_find_case_insensitive( //
    sz_cptr_t haystack, sz_size_t haystack_length,  //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length);

#pragma endregion

#pragma region Platform-Specific Backends

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_serial( //
    sz_cptr_t text, sz_size_t length,            //
    sz_rune_t *runes, sz_size_t runes_capacity,  //
    sz_size_t *runes_unpacked);
SZ_PUBLIC sz_size_t sz_utf8_case_fold_serial(  //
    sz_cptr_t source, sz_size_t source_length, //
    sz_ptr_t destination);
SZ_PUBLIC sz_cptr_t sz_utf8_find_case_insensitive_serial( //
    sz_cptr_t haystack, sz_size_t haystack_length,        //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length);

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_haswell( //
    sz_cptr_t text, sz_size_t length,             //
    sz_rune_t *runes, sz_size_t runes_capacity,   //
    sz_size_t *runes_unpacked);
SZ_PUBLIC sz_size_t sz_utf8_case_fold_haswell( //
    sz_cptr_t source, sz_size_t source_length, //
    sz_ptr_t destination);
SZ_PUBLIC sz_cptr_t sz_utf8_find_case_insensitive_haswell( //
    sz_cptr_t haystack, sz_size_t haystack_length,         //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length);

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_ice( //
    sz_cptr_t text, sz_size_t length,         //
    sz_rune_t *runes, sz_size_t runes_capacity, sz_size_t *runes_unpacked);
SZ_PUBLIC sz_size_t sz_utf8_case_fold_ice( //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination);
SZ_PUBLIC sz_cptr_t sz_utf8_find_case_insensitive_ice( //
    sz_cptr_t haystack, sz_size_t haystack_length,     //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length);

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_neon( //
    sz_cptr_t text, sz_size_t length,          //
    sz_rune_t *runes, sz_size_t runes_capacity, sz_size_t *runes_unpacked);
SZ_PUBLIC sz_size_t sz_utf8_case_fold_neon( //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination);
SZ_PUBLIC sz_cptr_t sz_utf8_find_case_insensitive_neon( //
    sz_cptr_t haystack, sz_size_t haystack_length,      //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length);

#pragma endregion

/*  Implementation Section
 *  Following the same pattern as find.h with implementations in the header file.
 */

#pragma region Serial Implementation

SZ_PUBLIC sz_bool_t sz_utf8_valid_serial(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u8_t const *end_u8 = text_u8 + length;

    while (text_u8 < end_u8) {
        sz_u8_t byte1 = *text_u8;

        // 1-byte sequence (0x00-0x7F)
        if (byte1 <= 0x7F) { text_u8 += 1; }

        // 2-byte sequence (0xC2-0xDF)
        else if (byte1 >= 0xC2 && byte1 <= 0xDF) {
            if (text_u8 + 1 >= end_u8) return sz_false_k;
            sz_u8_t byte2 = text_u8[1];
            if ((byte2 & 0xC0) != 0x80) return sz_false_k; // Invalid continuation
            text_u8 += 2;
        }

        // 3-byte sequence (0xE0-0xEF)
        else if (byte1 >= 0xE0 && byte1 <= 0xEF) {
            if (text_u8 + 2 >= end_u8) return sz_false_k;
            sz_u8_t byte2 = text_u8[1];
            sz_u8_t byte3 = text_u8[2];
            if ((byte2 & 0xC0) != 0x80 || (byte3 & 0xC0) != 0x80) return sz_false_k;

            // Check for overlong encodings and surrogates
            if (byte1 == 0xE0 && byte2 < 0xA0) return sz_false_k;  // Overlong
            if (byte1 == 0xED && byte2 >= 0xA0) return sz_false_k; // Surrogate (U+D800-U+DFFF)

            text_u8 += 3;
        }

        // 4-byte sequence (0xF0-0xF4)
        else if (byte1 >= 0xF0 && byte1 <= 0xF4) {
            if (text_u8 + 3 >= end_u8) return sz_false_k;
            sz_u8_t byte2 = text_u8[1];
            sz_u8_t byte3 = text_u8[2];
            sz_u8_t byte4 = text_u8[3];
            if ((byte2 & 0xC0) != 0x80 || (byte3 & 0xC0) != 0x80 || (byte4 & 0xC0) != 0x80) return sz_false_k;

            // Check for overlong and out-of-range
            if (byte1 == 0xF0 && byte2 < 0x90) return sz_false_k;  // Overlong
            if (byte1 == 0xF4 && byte2 >= 0x90) return sz_false_k; // > U+10FFFF

            text_u8 += 4;
        }

        // Invalid lead byte
        else
            return sz_false_k;
    }

    return sz_true_k;
}

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_serial( //
    sz_cptr_t text, sz_size_t length,            //
    sz_rune_t *runes, sz_size_t runes_capacity,  //
    sz_size_t *runes_unpacked) {

    sz_cptr_t src = text;
    sz_cptr_t src_end = text + length;
    sz_size_t runes_written = 0;

    // Process up to runes_capacity codepoints or end of input
    while (src < src_end && runes_written < runes_capacity) {
        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse(src, &rune, &rune_length);
        if (rune_length == sz_utf8_invalid_k) break;
        if (src + rune_length > src_end) break; // Incomplete sequence
        runes[runes_written++] = rune;
        src += rune_length;
    }

    *runes_unpacked = runes_written;
    return src;
}

// clang-format off
SZ_INTERNAL sz_size_t sz_unicode_fold_codepoint_(sz_rune_t rune, sz_rune_t *folded) {
    // 2-byte character ranges with +32 translations
    if (rune >= 0x0041 && rune <= 0x005A) { folded[0] = rune + 0x20; return 1; } // ASCII A-Z → a-z
    if (rune >= 0x00C0 && rune <= 0x00D6) { folded[0] = rune + 0x20; return 1; } // Latin-1 À-Ö → à-ö
    if (rune >= 0x00D8 && rune <= 0x00DE) { folded[0] = rune + 0x20; return 1; } // Latin-1 Ø-Þ → ø-þ
    if (rune >= 0x0391 && rune <= 0x03A1) { folded[0] = rune + 0x20; return 1; } // Greek Α-Ρ → α-ρ
    if (rune >= 0x03A3 && rune <= 0x03AB) { folded[0] = rune + 0x20; return 1; } // Greek Σ-Ϋ → σ-ϋ
    if (rune >= 0x0410 && rune <= 0x042F) { folded[0] = rune + 0x20; return 1; } // Cyrillic А-Я → а-я
    if (rune >= 0xFF21 && rune <= 0xFF3A) { folded[0] = rune + 0x20; return 1; } // Fullwidth Ａ-Ｚ → ａ-ｚ
    if (rune >= 0x10D50 && rune <= 0x10D65) { folded[0] = rune + 0x20; return 1; } // Garay
    if (rune >= 0x118A0 && rune <= 0x118BF) { folded[0] = rune + 0x20; return 1; } // Warang Citi
    if (rune >= 0x16E40 && rune <= 0x16E5F) { folded[0] = rune + 0x20; return 1; } // Medefaidrin
    // 2-byte character ranges with +48 translations
    if (rune >= 0x0531 && rune <= 0x0556) { folded[0] = rune + 0x30; return 1; } // Armenian Ա-Ֆ → ա-ֆ
    if (rune >= 0x2C00 && rune <= 0x2C2F) { folded[0] = rune + 0x30; return 1; } // Glagolitic Ⰰ-Ⱟ → ⰰ-ⱟ
    // 2-byte character ranges with -8 translations
    if (rune >= 0x13F8 && rune <= 0x13FD) { folded[0] = rune + 0xFFFFFFF8; return 1; } // Cherokee Ᏸ-Ᏽ (-8)
    if (rune >= 0x1F08 && rune <= 0x1F0F) { folded[0] = rune + 0xFFFFFFF8; return 1; } // Greek Ἀ-Ἇ (-8)
    if (rune >= 0x1F18 && rune <= 0x1F1D) { folded[0] = rune + 0xFFFFFFF8; return 1; } // Greek Ἐ-Ἕ (-8)
    if (rune >= 0x1F28 && rune <= 0x1F2F) { folded[0] = rune + 0xFFFFFFF8; return 1; } // Greek Ἠ-Ἧ (-8)
    if (rune >= 0x1F38 && rune <= 0x1F3F) { folded[0] = rune + 0xFFFFFFF8; return 1; } // Greek Ἰ-Ἷ (-8)
    if (rune >= 0x1F48 && rune <= 0x1F4D) { folded[0] = rune + 0xFFFFFFF8; return 1; } // Greek Ὀ-Ὅ (-8)
    if (rune >= 0x1F68 && rune <= 0x1F6F) { folded[0] = rune + 0xFFFFFFF8; return 1; } // Greek Ὠ-Ὧ (-8)
    // 2-byte character ranges with various other translations
    if (rune >= 0x1C90 && rune <= 0x1CBA) { folded[0] = rune + 0xFFFFF440; return 1; } // Georgian Mtavruli Ა-Ჺ (-3008)
    if (rune >= 0x1CBD && rune <= 0x1CBF) { folded[0] = rune + 0xFFFFF440; return 1; } // Georgian Mtavruli Ჽ-Ჿ (-3008)
    if (rune >= 0xAB70 && rune <= 0xABBF) { folded[0] = rune + 0xFFFF6830; return 1; } // Cherokee Ꭰ-Ᏼ (offset -38864)
    if (rune >= 0x10A0 && rune <= 0x10C5) { folded[0] = rune + 0x1C60; return 1; } // Georgian Ⴀ-Ⴥ (+7264)
    if (rune >= 0x24B6 && rune <= 0x24CF) { folded[0] = rune + 0x1A; return 1; } // Circled Ⓐ-Ⓩ → ⓐ-ⓩ (+26)
    if (rune >= 0x2160 && rune <= 0x216F) { folded[0] = rune + 0x10; return 1; } // Roman numerals Ⅰ-Ⅿ → ⅰ-ⅿ (+16)
    if (rune >= 0x0400 && rune <= 0x040F) { folded[0] = rune + 0x50; return 1; } // Cyrillic Ѐ-Џ → ѐ-џ (+80)
    if (rune >= 0x03FD && rune <= 0x03FF) { folded[0] = rune + 0xFFFFFF7E; return 1; } // Greek Ͻ-Ͽ (-130)
    if (rune >= 0x1FC8 && rune <= 0x1FCB) { folded[0] = rune + 0xFFFFFFAA; return 1; } // Greek Ὲ-Ή (-86)
    if (rune >= 0x0388 && rune <= 0x038A) { folded[0] = rune + 0x25; return 1; } // Greek Έ-Ί (+37)
    // 3-byte character ranges with various translations
    if (rune >= 0x10400 && rune <= 0x10427) { folded[0] = rune + 0x28; return 1; } // Deseret 𐐀-𐐧 → 𐐨-𐑏 (+40)
    if (rune >= 0x104B0 && rune <= 0x104D3) { folded[0] = rune + 0x28; return 1; } // Osage 𐒰-𐓓 → 𐓘-𐓻 (+40)
    if (rune >= 0x10C80 && rune <= 0x10CB2) { folded[0] = rune + 0x40; return 1; } // Old Hungarian (+64)
    if (rune >= 0x10570 && rune <= 0x1057A) { folded[0] = rune + 0x27; return 1; } // Vithkuqi (+39)
    if (rune >= 0x1057C && rune <= 0x1058A) { folded[0] = rune + 0x27; return 1; } // Vithkuqi (+39)
    if (rune >= 0x1058C && rune <= 0x10592) { folded[0] = rune + 0x27; return 1; } // Vithkuqi (+39)
    if (rune >= 0x1E900 && rune <= 0x1E921) { folded[0] = rune + 0x22; return 1; } // Adlam 𞤀-𞤡 → 𞤢-𞥃 (+34)
    if (rune >= 0x16EA0 && rune <= 0x16EB8) { folded[0] = rune + 0x1B; return 1; } // Kawi (+27)

    // Even/odd +1 mappings: uppercase at even codepoint, lowercase at odd (or vice versa)
    sz_u32_t is_even = ((rune & 1) == 0), is_odd = !is_even;
    // Latin Extended-A: Ą Ć Ę Ł Ń Ś Ź Ż, Č Ď Ě Ň Ř Š Ť Ž, Ő Ű, Ş Ğ
    if (rune >= 0x0100 && rune <= 0x012E && is_even) { folded[0] = rune + 1; return 1; } // Ā-Į
    if (rune >= 0x0132 && rune <= 0x0136 && is_even) { folded[0] = rune + 1; return 1; } // Ĳ-Ķ
    if (rune >= 0x0139 && rune <= 0x0147 && is_odd)  { folded[0] = rune + 1; return 1; } // Ĺ-Ň
    if (rune >= 0x014A && rune <= 0x0176 && is_even) { folded[0] = rune + 1; return 1; } // Ŋ-Ŷ
    if (rune >= 0x0179 && rune <= 0x017D && is_odd)  { folded[0] = rune + 1; return 1; } // Ź-Ž
    // Latin Extended-B: Pinyin, Romanian, Serbian/Croatian
    if (rune >= 0x01CD && rune <= 0x01DB && is_odd)  { folded[0] = rune + 1; return 1; } // Ǎ-Ǜ
    if (rune >= 0x01DE && rune <= 0x01EE && is_even) { folded[0] = rune + 1; return 1; } // Ǟ-Ǯ
    if (rune >= 0x01F8 && rune <= 0x01FE && is_even) { folded[0] = rune + 1; return 1; } // Ǹ-Ǿ
    if (rune >= 0x0200 && rune <= 0x021E && is_even) { folded[0] = rune + 1; return 1; } // Ȁ-Ȟ
    if (rune >= 0x0222 && rune <= 0x0232 && is_even) { folded[0] = rune + 1; return 1; } // Ȣ-Ȳ
    if (rune >= 0x0246 && rune <= 0x024E && is_even) { folded[0] = rune + 1; return 1; } // Ɇ-Ɏ
    // Greek archaic
    if (rune >= 0x0370 && rune <= 0x0372 && is_even) { folded[0] = rune + 1; return 1; } // Ͱ-Ͳ
    if (rune == 0x0376) { folded[0] = 0x0377; return 1; } // Ͷ → ͷ
    if (rune >= 0x03D8 && rune <= 0x03EE && is_even) { folded[0] = rune + 1; return 1; } // Ϙ-Ϯ
    // Cyrillic extended
    if (rune >= 0x0460 && rune <= 0x0480 && is_even) { folded[0] = rune + 1; return 1; } // Ѡ-Ҁ
    if (rune >= 0x048A && rune <= 0x04BE && is_even) { folded[0] = rune + 1; return 1; } // Ҋ-Ҿ
    if (rune >= 0x04C1 && rune <= 0x04CD && is_odd)  { folded[0] = rune + 1; return 1; } // Ӂ-Ӎ
    if (rune >= 0x04D0 && rune <= 0x04FE && is_even) { folded[0] = rune + 1; return 1; } // Ӑ-Ӿ
    if (rune >= 0x0500 && rune <= 0x052E && is_even) { folded[0] = rune + 1; return 1; } // Ԁ-Ԯ
    // Latin Extended Additional: Vietnamese Ạ Ả Ấ Ầ...
    if (rune >= 0x1E00 && rune <= 0x1E94 && is_even) { folded[0] = rune + 1; return 1; } // Ḁ-Ẕ
    if (rune >= 0x1EA0 && rune <= 0x1EFE && is_even) { folded[0] = rune + 1; return 1; } // Ạ-Ỿ
    // Coptic
    if (rune >= 0x2C80 && rune <= 0x2CE2 && is_even) { folded[0] = rune + 1; return 1; } // Ⲁ-Ⳣ
    // Cyrillic Extended-B
    if (rune >= 0xA640 && rune <= 0xA66C && is_even) { folded[0] = rune + 1; return 1; } // Ꙁ-Ꙭ
    if (rune >= 0xA680 && rune <= 0xA69A && is_even) { folded[0] = rune + 1; return 1; } // Ꚁ-Ꚛ
    // Latin Extended-D
    if (rune >= 0xA722 && rune <= 0xA72E && is_even) { folded[0] = rune + 1; return 1; } // Ꜣ-Ꜯ
    if (rune >= 0xA732 && rune <= 0xA76E && is_even) { folded[0] = rune + 1; return 1; } // Ꜳ-Ꝯ
    if (rune >= 0xA77E && rune <= 0xA786 && is_even) { folded[0] = rune + 1; return 1; } // Ꝿ-Ꞇ
    if (rune >= 0xA790 && rune <= 0xA792 && is_even) { folded[0] = rune + 1; return 1; } // Ꞑ-Ꞓ
    if (rune >= 0xA796 && rune <= 0xA7A8 && is_even) { folded[0] = rune + 1; return 1; } // Ꞗ-Ꞩ
    if (rune >= 0xA7B4 && rune <= 0xA7C2 && is_even) { folded[0] = rune + 1; return 1; } // Ꞵ-Ꟃ
    if (rune == 0xA7C7 || rune == 0xA7C9) { folded[0] = rune + 1; return 1; } // Ꟈ, Ꟊ
    if (rune >= 0xA7CC && rune <= 0xA7D8 && is_even) { folded[0] = rune + 1; return 1; } // Ꟍ-Ꟙ
    if (rune == 0xA7DA) { folded[0] = 0xA7DB; return 1; } // Ꟛ → ꟛ
    if (rune == 0xA7F5) { folded[0] = 0xA7F6; return 1; } // Ꟶ → ꟶ

    // Irregular one-to-one mappings: ~90 cases that don't follow even/odd patterns
    switch (rune) {
    // Latin-1 Supplement & specials
    case 0x00B5: folded[0] = 0x03BC; return 1; // µ → μ (micro sign to Greek mu)
    case 0x0178: folded[0] = 0x00FF; return 1; // Ÿ → ÿ
    case 0x017F: folded[0] = 0x0073; return 1; // ſ → s (long s)
    // Latin Extended-B: African/IPA letters with irregular mappings (0x0181-0x01BF)
    case 0x0181: folded[0] = 0x0253; return 1; // Ɓ → ɓ
    case 0x0182: folded[0] = 0x0183; return 1; // Ƃ → ƃ
    case 0x0184: folded[0] = 0x0185; return 1; // Ƅ → ƅ
    case 0x0186: folded[0] = 0x0254; return 1; // Ɔ → ɔ
    case 0x0187: folded[0] = 0x0188; return 1; // Ƈ → ƈ
    case 0x0189: folded[0] = 0x0256; return 1; // Ɖ → ɖ
    case 0x018A: folded[0] = 0x0257; return 1; // Ɗ → ɗ
    case 0x018B: folded[0] = 0x018C; return 1; // Ƌ → ƌ
    case 0x018E: folded[0] = 0x01DD; return 1; // Ǝ → ǝ
    case 0x018F: folded[0] = 0x0259; return 1; // Ə → ə (schwa, Azerbaijani)
    case 0x0190: folded[0] = 0x025B; return 1; // Ɛ → ɛ
    case 0x0191: folded[0] = 0x0192; return 1; // Ƒ → ƒ
    case 0x0193: folded[0] = 0x0260; return 1; // Ɠ → ɠ
    case 0x0194: folded[0] = 0x0263; return 1; // Ɣ → ɣ
    case 0x0196: folded[0] = 0x0269; return 1; // Ɩ → ɩ
    case 0x0197: folded[0] = 0x0268; return 1; // Ɨ → ɨ
    case 0x0198: folded[0] = 0x0199; return 1; // Ƙ → ƙ
    case 0x019C: folded[0] = 0x026F; return 1; // Ɯ → ɯ
    case 0x019D: folded[0] = 0x0272; return 1; // Ɲ → ɲ
    case 0x019F: folded[0] = 0x0275; return 1; // Ɵ → ɵ
    case 0x01A0: folded[0] = 0x01A1; return 1; // Ơ → ơ (Vietnamese)
    case 0x01A2: folded[0] = 0x01A3; return 1; // Ƣ → ƣ
    case 0x01A4: folded[0] = 0x01A5; return 1; // Ƥ → ƥ
    case 0x01A6: folded[0] = 0x0280; return 1; // Ʀ → ʀ
    case 0x01A7: folded[0] = 0x01A8; return 1; // Ƨ → ƨ
    case 0x01A9: folded[0] = 0x0283; return 1; // Ʃ → ʃ
    case 0x01AC: folded[0] = 0x01AD; return 1; // Ƭ → ƭ
    case 0x01AE: folded[0] = 0x0288; return 1; // Ʈ → ʈ
    case 0x01AF: folded[0] = 0x01B0; return 1; // Ư → ư (Vietnamese)
    case 0x01B1: folded[0] = 0x028A; return 1; // Ʊ → ʊ
    case 0x01B2: folded[0] = 0x028B; return 1; // Ʋ → ʋ
    case 0x01B3: folded[0] = 0x01B4; return 1; // Ƴ → ƴ
    case 0x01B5: folded[0] = 0x01B6; return 1; // Ƶ → ƶ
    case 0x01B7: folded[0] = 0x0292; return 1; // Ʒ → ʒ
    case 0x01B8: folded[0] = 0x01B9; return 1; // Ƹ → ƹ
    case 0x01BC: folded[0] = 0x01BD; return 1; // Ƽ → ƽ

    // Digraphs: Serbian/Croatian DŽ, LJ, NJ and DZ
    case 0x01C4: folded[0] = 0x01C6; return 1; // Ǆ → ǆ
    case 0x01C5: folded[0] = 0x01C6; return 1; // ǅ → ǆ (titlecase)
    case 0x01C7: folded[0] = 0x01C9; return 1; // Ǉ → ǉ
    case 0x01C8: folded[0] = 0x01C9; return 1; // ǈ → ǉ (titlecase)
    case 0x01CA: folded[0] = 0x01CC; return 1; // Ǌ → ǌ
    case 0x01CB: folded[0] = 0x01CC; return 1; // ǋ → ǌ (titlecase)
    case 0x01F1: folded[0] = 0x01F3; return 1; // Ǳ → ǳ
    case 0x01F2: folded[0] = 0x01F3; return 1; // ǲ → ǳ (titlecase)
    // Latin Extended-B: isolated irregulars
    case 0x01F4: folded[0] = 0x01F5; return 1; // Ǵ → ǵ (between ranges)
    case 0x01F6: folded[0] = 0x0195; return 1; // Ƕ → ƕ (hwair)
    case 0x01F7: folded[0] = 0x01BF; return 1; // Ƿ → ƿ (wynn)
    case 0x0220: folded[0] = 0x019E; return 1; // Ƞ → ƞ
    case 0x023A: folded[0] = 0x2C65; return 1; // Ⱥ → ⱥ
    case 0x023B: folded[0] = 0x023C; return 1; // Ȼ → ȼ
    case 0x023D: folded[0] = 0x019A; return 1; // Ƚ → ƚ
    case 0x023E: folded[0] = 0x2C66; return 1; // Ⱦ → ⱦ
    case 0x0241: folded[0] = 0x0242; return 1; // Ɂ → ɂ
    case 0x0243: folded[0] = 0x0180; return 1; // Ƀ → ƀ
    case 0x0244: folded[0] = 0x0289; return 1; // Ʉ → ʉ
    case 0x0245: folded[0] = 0x028C; return 1; // Ʌ → ʌ

    // Greek: combining iota, accented vowels, variant forms
    case 0x0345: folded[0] = 0x03B9; return 1; // ͅ → ι (combining iota subscript)
    case 0x037F: folded[0] = 0x03F3; return 1; // Ϳ → ϳ
    case 0x0386: folded[0] = 0x03AC; return 1; // Ά → ά
    case 0x038C: folded[0] = 0x03CC; return 1; // Ό → ό
    case 0x038E: folded[0] = 0x03CD; return 1; // Ύ → ύ
    case 0x038F: folded[0] = 0x03CE; return 1; // Ώ → ώ
    case 0x03C2: folded[0] = 0x03C3; return 1; // ς → σ (final sigma)
    case 0x03CF: folded[0] = 0x03D7; return 1; // Ϗ → ϗ
    case 0x03D0: folded[0] = 0x03B2; return 1; // ϐ → β (beta symbol)
    case 0x03D1: folded[0] = 0x03B8; return 1; // ϑ → θ (theta symbol)
    case 0x03D5: folded[0] = 0x03C6; return 1; // ϕ → φ (phi symbol)
    case 0x03D6: folded[0] = 0x03C0; return 1; // ϖ → π (pi symbol)
    case 0x03F0: folded[0] = 0x03BA; return 1; // ϰ → κ (kappa symbol)
    case 0x03F1: folded[0] = 0x03C1; return 1; // ϱ → ρ (rho symbol)
    case 0x03F4: folded[0] = 0x03B8; return 1; // ϴ → θ
    case 0x03F5: folded[0] = 0x03B5; return 1; // ϵ → ε (lunate epsilon)
    case 0x03F7: folded[0] = 0x03F8; return 1; // Ϸ → ϸ
    case 0x03F9: folded[0] = 0x03F2; return 1; // Ϲ → ϲ
    case 0x03FA: folded[0] = 0x03FB; return 1; // Ϻ → ϻ
    // Cyrillic: palochka (irregular +15 offset)
    case 0x04C0: folded[0] = 0x04CF; return 1; // Ӏ → ӏ
    // Georgian: large offsets to lowercase block
    case 0x10C7: folded[0] = 0x2D27; return 1; // Ⴧ → ⴧ
    case 0x10CD: folded[0] = 0x2D2D; return 1; // Ⴭ → ⴭ
    // Cyrillic Extended-C: Old Slavonic variant forms (map to basic Cyrillic)
    case 0x1C80: folded[0] = 0x0432; return 1; // ᲀ → в
    case 0x1C81: folded[0] = 0x0434; return 1; // ᲁ → д
    case 0x1C82: folded[0] = 0x043E; return 1; // ᲂ → о
    case 0x1C83: folded[0] = 0x0441; return 1; // ᲃ → с
    case 0x1C84: folded[0] = 0x0442; return 1; // ᲄ → т
    case 0x1C85: folded[0] = 0x0442; return 1; // ᲅ → т
    case 0x1C86: folded[0] = 0x044A; return 1; // ᲆ → ъ
    case 0x1C87: folded[0] = 0x0463; return 1; // ᲇ → ѣ
    case 0x1C88: folded[0] = 0xA64B; return 1; // ᲈ → ꙋ
    case 0x1C89: folded[0] = 0x1C8A; return 1; // Ᲊ → ᲊ
    // Latin Extended Additional: long s with dot above (irregular target)
    case 0x1E9B: folded[0] = 0x1E61; return 1; // ẛ → ṡ

    // Greek Extended: vowels with breathing marks (irregular offsets)
    case 0x1F59: folded[0] = 0x1F51; return 1; // 'Ὑ' → 'ὑ'
    case 0x1F5B: folded[0] = 0x1F53; return 1; // 'Ὓ' → 'ὓ'
    case 0x1F5D: folded[0] = 0x1F55; return 1; // 'Ὕ' → 'ὕ'
    case 0x1F5F: folded[0] = 0x1F57; return 1; // 'Ὗ' → 'ὗ'
    case 0x1FB8: folded[0] = 0x1FB0; return 1; // 'Ᾰ' → 'ᾰ'
    case 0x1FB9: folded[0] = 0x1FB1; return 1; // 'Ᾱ' → 'ᾱ'
    case 0x1FBA: folded[0] = 0x1F70; return 1; // 'Ὰ' → 'ὰ'
    case 0x1FBB: folded[0] = 0x1F71; return 1; // 'Ά' → 'ά'
    case 0x1FBE: folded[0] = 0x03B9; return 1; // 'ι' → 'ι'
    case 0x1FD8: folded[0] = 0x1FD0; return 1; // 'Ῐ' → 'ῐ'
    case 0x1FD9: folded[0] = 0x1FD1; return 1; // 'Ῑ' → 'ῑ'
    case 0x1FDA: folded[0] = 0x1F76; return 1; // 'Ὶ' → 'ὶ'
    case 0x1FDB: folded[0] = 0x1F77; return 1; // 'Ί' → 'ί'
    case 0x1FE8: folded[0] = 0x1FE0; return 1; // 'Ῠ' → 'ῠ'
    case 0x1FE9: folded[0] = 0x1FE1; return 1; // 'Ῡ' → 'ῡ'
    case 0x1FEA: folded[0] = 0x1F7A; return 1; // 'Ὺ' → 'ὺ'
    case 0x1FEB: folded[0] = 0x1F7B; return 1; // 'Ύ' → 'ύ'
    case 0x1FEC: folded[0] = 0x1FE5; return 1; // 'Ῥ' → 'ῥ'
    case 0x1FF8: folded[0] = 0x1F78; return 1; // 'Ὸ' → 'ὸ'
    case 0x1FF9: folded[0] = 0x1F79; return 1; // 'Ό' → 'ό'
    case 0x1FFA: folded[0] = 0x1F7C; return 1; // 'Ὼ' → 'ὼ'
    case 0x1FFB:
        folded[0] = 0x1F7D; return 1; // 'Ώ' → 'ώ'
    // Letterlike Symbols: compatibility mappings
    case 0x2126: folded[0] = 0x03C9; return 1; // 'Ω' → 'ω'
    case 0x212A: folded[0] = 0x006B; return 1; // 'K' → 'k'
    case 0x212B: folded[0] = 0x00E5; return 1; // 'Å' → 'å'
    case 0x2132: folded[0] = 0x214E; return 1; // 'Ⅎ' → 'ⅎ'
    case 0x2183:
        folded[0] = 0x2184; return 1; // 'Ↄ' → 'ↄ'

    // Latin Extended-C: irregular mappings to IPA/other blocks
    case 0x2C60: folded[0] = 0x2C61; return 1; // 'Ⱡ' → 'ⱡ'
    case 0x2C62: folded[0] = 0x026B; return 1; // 'Ɫ' → 'ɫ'
    case 0x2C63: folded[0] = 0x1D7D; return 1; // 'Ᵽ' → 'ᵽ'
    case 0x2C64: folded[0] = 0x027D; return 1; // 'Ɽ' → 'ɽ'
    case 0x2C67: folded[0] = 0x2C68; return 1; // 'Ⱨ' → 'ⱨ'
    case 0x2C69: folded[0] = 0x2C6A; return 1; // 'Ⱪ' → 'ⱪ'
    case 0x2C6B: folded[0] = 0x2C6C; return 1; // 'Ⱬ' → 'ⱬ'
    case 0x2C6D: folded[0] = 0x0251; return 1; // 'Ɑ' → 'ɑ'
    case 0x2C6E: folded[0] = 0x0271; return 1; // 'Ɱ' → 'ɱ'
    case 0x2C6F: folded[0] = 0x0250; return 1; // 'Ɐ' → 'ɐ'
    case 0x2C70: folded[0] = 0x0252; return 1; // 'Ɒ' → 'ɒ'
    case 0x2C72: folded[0] = 0x2C73; return 1; // 'Ⱳ' → 'ⱳ'
    case 0x2C75: folded[0] = 0x2C76; return 1; // 'Ⱶ' → 'ⱶ'
    case 0x2C7E: folded[0] = 0x023F; return 1; // 'Ȿ' → 'ȿ'
    case 0x2C7F:
        folded[0] = 0x0240; return 1; // 'Ɀ' → 'ɀ'

    // Coptic: irregular cases outside the even/odd range
    case 0x2CEB: folded[0] = 0x2CEC; return 1; // 'Ⳬ' → 'ⳬ'
    case 0x2CED: folded[0] = 0x2CEE; return 1; // 'Ⳮ' → 'ⳮ'
    case 0x2CF2:
        folded[0] = 0x2CF3; return 1; // 'Ⳳ' → 'ⳳ'

    // Latin Extended-D: isolated irregulars with non-standard offsets
    case 0xA779: folded[0] = 0xA77A; return 1; // 'Ꝺ' → 'ꝺ'
    case 0xA77B: folded[0] = 0xA77C; return 1; // 'Ꝼ' → 'ꝼ'
    case 0xA77D: folded[0] = 0x1D79; return 1; // 'Ᵹ' → 'ᵹ'
    case 0xA78B: folded[0] = 0xA78C; return 1; // 'Ꞌ' → 'ꞌ'
    case 0xA78D: folded[0] = 0x0265; return 1; // 'Ɥ' → 'ɥ'
    case 0xA7AA: folded[0] = 0x0266; return 1; // 'Ɦ' → 'ɦ'
    case 0xA7AB: folded[0] = 0x025C; return 1; // 'Ɜ' → 'ɜ'
    case 0xA7AC: folded[0] = 0x0261; return 1; // 'Ɡ' → 'ɡ'
    case 0xA7AD: folded[0] = 0x026C; return 1; // 'Ɬ' → 'ɬ'
    case 0xA7AE: folded[0] = 0x026A; return 1; // 'Ɪ' → 'ɪ'
    case 0xA7B0: folded[0] = 0x029E; return 1; // 'Ʞ' → 'ʞ'
    case 0xA7B1: folded[0] = 0x0287; return 1; // 'Ʇ' → 'ʇ'
    case 0xA7B2: folded[0] = 0x029D; return 1; // 'Ʝ' → 'ʝ'
    case 0xA7B3: folded[0] = 0xAB53; return 1; // 'Ꭓ' → 'ꭓ'
    case 0xA7C4: folded[0] = 0xA794; return 1; // 'Ꞔ' → 'ꞔ'
    case 0xA7C5: folded[0] = 0x0282; return 1; // 'Ʂ' → 'ʂ'
    case 0xA7C6: folded[0] = 0x1D8E; return 1; // 'Ᶎ' → 'ᶎ'
    case 0xA7CB: folded[0] = 0x0264; return 1; // 'Ɤ' → 'ɤ'
    case 0xA7DC:
        folded[0] = 0x019B; return 1; // 'Ƛ' → 'ƛ'

    // Vithkuqi: Albanian historical script
    case 0x10594: folded[0] = 0x105BB; return 1; // '𐖔' → '𐖻'
    case 0x10595: folded[0] = 0x105BC; return 1; // '𐖕' → '𐖼'
    }

    // One-to-many expansions
    switch (rune) {
    case 0x00DF: folded[0] = 0x0073; folded[1] = 0x0073; return 2; // ß → ss (German)
    case 0x0130: folded[0] = 0x0069; folded[1] = 0x0307; return 2; // İ → i + combining (Turkish)
    case 0x0149: folded[0] = 0x02BC; folded[1] = 0x006E; return 2; // ŉ → ʼn (Afrikaans)
    case 0x01F0: folded[0] = 0x006A; folded[1] = 0x030C; return 2; // ǰ → j + combining
    case 0x0390: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0301; return 3; // ΐ → ι + 2 combining (Greek)
    case 0x03B0: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0301; return 3; // ΰ → υ + 2 combining (Greek)
    case 0x0587: folded[0] = 0x0565; folded[1] = 0x0582; return 2; // և → ե (Armenian)
    case 0x1E96: folded[0] = 0x0068; folded[1] = 0x0331; return 2; // ẖ → h + combining
    case 0x1E97: folded[0] = 0x0074; folded[1] = 0x0308; return 2; // ẗ → t + combining
    case 0x1E98: folded[0] = 0x0077; folded[1] = 0x030A; return 2; // ẘ → w + combining
    case 0x1E99: folded[0] = 0x0079; folded[1] = 0x030A; return 2; // ẙ → y + combining
    case 0x1E9A: folded[0] = 0x0061; folded[1] = 0x02BE; return 2; // ẚ → aʾ
    case 0x1E9E: folded[0] = 0x0073; folded[1] = 0x0073; return 2; // ẞ → ss (German capital Eszett)
    case 0x1F50: folded[0] = 0x03C5; folded[1] = 0x0313; return 2; // ὐ → υ + combining (Greek)
    case 0x1F52: folded[0] = 0x03C5; folded[1] = 0x0313; folded[2] = 0x0300; return 3; // ὒ → υ + 2 combining
    case 0x1F54: folded[0] = 0x03C5; folded[1] = 0x0313; folded[2] = 0x0301; return 3; // ὔ → υ + 2 combining
    case 0x1F56: folded[0] = 0x03C5; folded[1] = 0x0313; folded[2] = 0x0342; return 3; // ὖ → υ + 2 combining
    case 0x1F80: folded[0] = 0x1F00; folded[1] = 0x03B9; return 2; // ᾀ → ἀι (Greek iota subscript)
    case 0x1F81: folded[0] = 0x1F01; folded[1] = 0x03B9; return 2; // ᾁ → ἁι
    case 0x1F82: folded[0] = 0x1F02; folded[1] = 0x03B9; return 2; // ᾂ → ἂι
    case 0x1F83: folded[0] = 0x1F03; folded[1] = 0x03B9; return 2; // ᾃ → ἃι
    case 0x1F84: folded[0] = 0x1F04; folded[1] = 0x03B9; return 2; // ᾄ → ἄι
    case 0x1F85: folded[0] = 0x1F05; folded[1] = 0x03B9; return 2; // ᾅ → ἅι
    case 0x1F86: folded[0] = 0x1F06; folded[1] = 0x03B9; return 2; // ᾆ → ἆι
    case 0x1F87: folded[0] = 0x1F07; folded[1] = 0x03B9; return 2; // ᾇ → ἇι
    case 0x1F88: folded[0] = 0x1F00; folded[1] = 0x03B9; return 2; // ᾈ → ἀι
    case 0x1F89: folded[0] = 0x1F01; folded[1] = 0x03B9; return 2; // ᾉ → ἁι
    case 0x1F8A: folded[0] = 0x1F02; folded[1] = 0x03B9; return 2; // ᾊ → ἂι
    case 0x1F8B: folded[0] = 0x1F03; folded[1] = 0x03B9; return 2; // ᾋ → ἃι
    case 0x1F8C: folded[0] = 0x1F04; folded[1] = 0x03B9; return 2; // ᾌ → ἄι
    case 0x1F8D: folded[0] = 0x1F05; folded[1] = 0x03B9; return 2; // ᾍ → ἅι
    case 0x1F8E: folded[0] = 0x1F06; folded[1] = 0x03B9; return 2; // ᾎ → ἆι
    case 0x1F8F: folded[0] = 0x1F07; folded[1] = 0x03B9; return 2; // ᾏ → ἇι
    case 0x1F90: folded[0] = 0x1F20; folded[1] = 0x03B9; return 2; // ᾐ → ἠι
    case 0x1F91: folded[0] = 0x1F21; folded[1] = 0x03B9; return 2; // ᾑ → ἡι
    case 0x1F92: folded[0] = 0x1F22; folded[1] = 0x03B9; return 2; // ᾒ → ἢι
    case 0x1F93: folded[0] = 0x1F23; folded[1] = 0x03B9; return 2; // ᾓ → ἣι
    case 0x1F94: folded[0] = 0x1F24; folded[1] = 0x03B9; return 2; // ᾔ → ἤι
    case 0x1F95: folded[0] = 0x1F25; folded[1] = 0x03B9; return 2; // ᾕ → ἥι
    case 0x1F96: folded[0] = 0x1F26; folded[1] = 0x03B9; return 2; // ᾖ → ἦι
    case 0x1F97: folded[0] = 0x1F27; folded[1] = 0x03B9; return 2; // ᾗ → ἧι
    case 0x1F98: folded[0] = 0x1F20; folded[1] = 0x03B9; return 2; // ᾘ → ἠι
    case 0x1F99: folded[0] = 0x1F21; folded[1] = 0x03B9; return 2; // ᾙ → ἡι
    case 0x1F9A: folded[0] = 0x1F22; folded[1] = 0x03B9; return 2; // ᾚ → ἢι
    case 0x1F9B: folded[0] = 0x1F23; folded[1] = 0x03B9; return 2; // ᾛ → ἣι
    case 0x1F9C: folded[0] = 0x1F24; folded[1] = 0x03B9; return 2; // ᾜ → ἤι
    case 0x1F9D: folded[0] = 0x1F25; folded[1] = 0x03B9; return 2; // ᾝ → ἥι
    case 0x1F9E: folded[0] = 0x1F26; folded[1] = 0x03B9; return 2; // ᾞ → ἦι
    case 0x1F9F: folded[0] = 0x1F27; folded[1] = 0x03B9; return 2; // ᾟ → ἧι
    case 0x1FA0: folded[0] = 0x1F60; folded[1] = 0x03B9; return 2; // ᾠ → ὠι
    case 0x1FA1: folded[0] = 0x1F61; folded[1] = 0x03B9; return 2; // ᾡ → ὡι
    case 0x1FA2: folded[0] = 0x1F62; folded[1] = 0x03B9; return 2; // ᾢ → ὢι
    case 0x1FA3: folded[0] = 0x1F63; folded[1] = 0x03B9; return 2; // ᾣ → ὣι
    case 0x1FA4: folded[0] = 0x1F64; folded[1] = 0x03B9; return 2; // ᾤ → ὤι
    case 0x1FA5: folded[0] = 0x1F65; folded[1] = 0x03B9; return 2; // ᾥ → ὥι
    case 0x1FA6: folded[0] = 0x1F66; folded[1] = 0x03B9; return 2; // ᾦ → ὦι
    case 0x1FA7: folded[0] = 0x1F67; folded[1] = 0x03B9; return 2; // ᾧ → ὧι
    case 0x1FA8: folded[0] = 0x1F60; folded[1] = 0x03B9; return 2; // ᾨ → ὠι
    case 0x1FA9: folded[0] = 0x1F61; folded[1] = 0x03B9; return 2; // ᾩ → ὡι
    case 0x1FAA: folded[0] = 0x1F62; folded[1] = 0x03B9; return 2; // ᾪ → ὢι
    case 0x1FAB: folded[0] = 0x1F63; folded[1] = 0x03B9; return 2; // ᾫ → ὣι
    case 0x1FAC: folded[0] = 0x1F64; folded[1] = 0x03B9; return 2; // ᾬ → ὤι
    case 0x1FAD: folded[0] = 0x1F65; folded[1] = 0x03B9; return 2; // ᾭ → ὥι
    case 0x1FAE: folded[0] = 0x1F66; folded[1] = 0x03B9; return 2; // ᾮ → ὦι
    case 0x1FAF: folded[0] = 0x1F67; folded[1] = 0x03B9; return 2; // ᾯ → ὧι
    case 0x1FB2: folded[0] = 0x1F70; folded[1] = 0x03B9; return 2; // ᾲ → ὰι
    case 0x1FB3: folded[0] = 0x03B1; folded[1] = 0x03B9; return 2; // ᾳ → αι
    case 0x1FB4: folded[0] = 0x03AC; folded[1] = 0x03B9; return 2; // ᾴ → άι
    case 0x1FB6: folded[0] = 0x03B1; folded[1] = 0x0342; return 2; // ᾶ → α + combining
    case 0x1FB7: folded[0] = 0x03B1; folded[1] = 0x0342; folded[2] = 0x03B9; return 3; // ᾷ → α + 2 combining
    case 0x1FBC: folded[0] = 0x03B1; folded[1] = 0x03B9; return 2; // ᾼ → αι
    case 0x1FC2: folded[0] = 0x1F74; folded[1] = 0x03B9; return 2; // ῂ → ὴι
    case 0x1FC3: folded[0] = 0x03B7; folded[1] = 0x03B9; return 2; // ῃ → ηι
    case 0x1FC4: folded[0] = 0x03AE; folded[1] = 0x03B9; return 2; // ῄ → ήι
    case 0x1FC6: folded[0] = 0x03B7; folded[1] = 0x0342; return 2; // ῆ → η + combining
    case 0x1FC7: folded[0] = 0x03B7; folded[1] = 0x0342; folded[2] = 0x03B9; return 3; // ῇ → η + 2 combining
    case 0x1FCC: folded[0] = 0x03B7; folded[1] = 0x03B9; return 2; // ῌ → ηι
    case 0x1FD2: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0300; return 3; // ῒ → ι + 2 combining
    case 0x1FD3: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0301; return 3; // ΐ → ι + 2 combining
    case 0x1FD6: folded[0] = 0x03B9; folded[1] = 0x0342; return 2; // ῖ → ι + combining
    case 0x1FD7: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0342; return 3; // ῗ → ι + 2 combining
    case 0x1FE2: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0300; return 3; // ῢ → υ + 2 combining
    case 0x1FE3: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0301; return 3; // ΰ → υ + 2 combining
    case 0x1FE4: folded[0] = 0x03C1; folded[1] = 0x0313; return 2; // ῤ → ρ + combining
    case 0x1FE6: folded[0] = 0x03C5; folded[1] = 0x0342; return 2; // ῦ → υ + combining
    case 0x1FE7: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0342; return 3; // ῧ → υ + 2 combining
    case 0x1FF2: folded[0] = 0x1F7C; folded[1] = 0x03B9; return 2; // ῲ → ὼι
    case 0x1FF3: folded[0] = 0x03C9; folded[1] = 0x03B9; return 2; // ῳ → ωι
    case 0x1FF4: folded[0] = 0x03CE; folded[1] = 0x03B9; return 2; // ῴ → ώι
    case 0x1FF6: folded[0] = 0x03C9; folded[1] = 0x0342; return 2; // ῶ → ω + combining
    case 0x1FF7: folded[0] = 0x03C9; folded[1] = 0x0342; folded[2] = 0x03B9; return 3; // ῷ → ω + 2 combining
    case 0x1FFC: folded[0] = 0x03C9; folded[1] = 0x03B9; return 2; // ῼ → ωι
    case 0xFB00: folded[0] = 0x0066; folded[1] = 0x0066; return 2; // ﬀ → ff
    case 0xFB01: folded[0] = 0x0066; folded[1] = 0x0069; return 2; // ﬁ → fi
    case 0xFB02: folded[0] = 0x0066; folded[1] = 0x006C; return 2; // ﬂ → fl
    case 0xFB03: folded[0] = 0x0066; folded[1] = 0x0066; folded[2] = 0x0069; return 3; // ﬃ → ffi
    case 0xFB04: folded[0] = 0x0066; folded[1] = 0x0066; folded[2] = 0x006C; return 3; // ﬄ → ffl
    case 0xFB05: folded[0] = 0x0073; folded[1] = 0x0074; return 2; // ﬅ → st
    case 0xFB06: folded[0] = 0x0073; folded[1] = 0x0074; return 2; // ﬆ → st
    case 0xFB13: folded[0] = 0x0574; folded[1] = 0x0576; return 2; // ﬓ → մն
    case 0xFB14: folded[0] = 0x0574; folded[1] = 0x0565; return 2; // ﬔ → մե
    case 0xFB15: folded[0] = 0x0574; folded[1] = 0x056B; return 2; // ﬕ → մի
    case 0xFB16: folded[0] = 0x057E; folded[1] = 0x0576; return 2; // ﬖ → վն
    case 0xFB17: folded[0] = 0x0574; folded[1] = 0x056D; return 2; // ﬗ → մխ
    }

    folded[0] = rune; return 1; // No folding
    // clang-format on
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_case_insensitive_serial( //
    sz_cptr_t haystack, sz_size_t haystack_length,        //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length) {

    if (needle_length == 0) {
        *matched_length = 0;
        return haystack;
    }

    // Pre-fold the needle into a buffer of codepoints
    sz_rune_t folded_needle[1024];
    sz_size_t folded_needle_count = 0;
    sz_cptr_t needle_ptr = needle;
    sz_cptr_t needle_end = needle + needle_length;

    while (needle_ptr < needle_end && folded_needle_count < 1024) {
        sz_rune_t cp;
        sz_rune_length_t rune_length;
        sz_rune_parse(needle_ptr, &cp, &rune_length);
        if (rune_length == 0) break;

        // Apply case folding
        sz_rune_t folded[4];
        sz_size_t folded_count = sz_unicode_fold_codepoint_(cp, folded);
        for (sz_size_t i = 0; i < folded_count && folded_needle_count < 1024; ++i)
            folded_needle[folded_needle_count++] = folded[i];

        needle_ptr += rune_length;
    }

    if (folded_needle_count == 0) {
        *matched_length = 0;
        return SZ_NULL_CHAR;
    }

    // Search through the haystack
    sz_cptr_t haystack_ptr = haystack;
    sz_cptr_t haystack_end = haystack + haystack_length;

    while (haystack_ptr < haystack_end) {
        sz_cptr_t match_start = haystack_ptr;
        sz_cptr_t match_ptr = haystack_ptr;
        sz_size_t needle_idx = 0;
        sz_bool_t mismatch = sz_false_k;

        // Try to match the folded needle at this position
        while (needle_idx < folded_needle_count && match_ptr < haystack_end && !mismatch) {
            sz_rune_t cp;
            sz_rune_length_t rune_length;
            sz_rune_parse(match_ptr, &cp, &rune_length);
            if (rune_length == 0) break;

            // Apply case folding
            sz_rune_t folded[4];
            sz_size_t folded_count = sz_unicode_fold_codepoint_(cp, folded);

            // Compare all folded codepoints against the needle
            for (sz_size_t i = 0; i < folded_count && needle_idx < folded_needle_count; ++i) {
                if (folded[i] != folded_needle[needle_idx]) {
                    mismatch = sz_true_k;
                    break;
                }
                needle_idx++;
            }

            match_ptr += rune_length;
        }

        // Did we match the entire needle?
        if (needle_idx == folded_needle_count && !mismatch) {
            *matched_length = (sz_size_t)(match_ptr - match_start);
            return match_start;
        }

        // Move to next codepoint in haystack
        sz_rune_t cp;
        sz_rune_length_t rune_length;
        sz_rune_parse(haystack_ptr, &cp, &rune_length);
        if (rune_length == 0) break;
        haystack_ptr += rune_length;
    }

    *matched_length = 0;
    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_size_t sz_utf8_case_fold_serial(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {

    sz_u8_t const *src = (sz_u8_t const *)source;
    sz_u8_t const *src_end = src + source_length;
    sz_u8_t *dst = (sz_u8_t *)destination;
    sz_u8_t *dst_start = dst;

    while (src < src_end) {
        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse((sz_cptr_t)src, &rune, &rune_length);
        sz_assert_(rune_length != sz_utf8_invalid_k && "Input text is not valid UTF-8");
        src += rune_length;

        sz_rune_t folded[4];
        sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded);
        for (sz_size_t i = 0; i != folded_count; ++i) dst += sz_rune_export(folded[i], dst);
    }

    return (sz_size_t)(dst - dst_start);
}

#pragma endregion // Serial Implementation

#pragma region Ice Lake Implementation
#if SZ_USE_ICE
#if defined(__clang__)
#pragma clang attribute push(                                                                          \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2")
#endif

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_ice(   //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {

    // Process up to the minimum of: available bytes, output capacity * 4, or optimal chunk size (64)
    sz_size_t chunk_size = sz_min_of_three(length, runes_capacity * 4, 64);
    sz_u512_vec_t text_vec, runes_vec;
    __mmask64 load_mask = sz_u64_mask_until_(chunk_size);
    text_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, (sz_u8_t const *)text);

    // Check, how many of the next characters are single byte (ASCII) codepoints
    // ASCII bytes have bit 7 clear (0x00-0x7F), non-ASCII have bit 7 set (0x80-0xFF)
    __mmask64 non_ascii_mask = _mm512_movepi8_mask(text_vec.zmm);
    // Find first non-ASCII byte or end of loaded data
    sz_size_t ascii_prefix_length = sz_u64_ctz(non_ascii_mask | ~load_mask);

    if (ascii_prefix_length) {
        // Unpack the last 16 bytes of text into the next 16 runes.
        // Even if we have more than 16 ASCII characters, we don't want to overcomplicate control flow here.
        sz_size_t runes_to_place = sz_min_of_three(ascii_prefix_length, 16, runes_capacity);
        runes_vec.zmm = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(text_vec.zmm));
        _mm512_mask_storeu_epi32(runes, sz_u16_mask_until_(runes_to_place), runes_vec.zmm);
        *runes_unpacked = runes_to_place;
        return text + runes_to_place;
    }

    // Check for the number of 2-byte characters
    // 2-byte UTF-8: [lead, cont] where lead=110xxxxx (0xC0-0xDF), cont=10xxxxxx (0x80-0xBF)
    // In 16-bit little-endian: 0xCCLL where LL=lead, CC=cont
    // Mask: 0xC0E0 (cont & 0xC0, lead & 0xE0), Pattern: 0x80C0 (cont=0x80, lead=0xC0)
    __mmask32 non_two_byte_mask = _mm512_cmpneq_epi16_mask(
        _mm512_and_si512(text_vec.zmm, _mm512_set1_epi16((short)0xC0E0)), _mm512_set1_epi16((short)0x80C0));
    sz_size_t two_byte_prefix_length = sz_u64_ctz(non_two_byte_mask);
    if (two_byte_prefix_length) {
        // Unpack the last 32 bytes of text into the next 32 runes.
        // Even if we have more than 32 two-byte characters, we don't want to overcomplicate control flow here.
        sz_size_t runes_to_place = sz_min_of_three(two_byte_prefix_length, 32, runes_capacity);
        runes_vec.zmm = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(text_vec.zmm));
        // Decode 2-byte UTF-8: ((lead & 0x1F) << 6) | (cont & 0x3F)
        // After cvtepu16_epi32: value = 0x0000CCLL where LL=lead (bits 7-0), CC=cont (bits 15-8)
        runes_vec.zmm = _mm512_or_si512(                                                      //
            _mm512_slli_epi32(_mm512_and_si512(runes_vec.zmm, _mm512_set1_epi32(0x1FU)), 6),  // (lead & 0x1F) << 6
            _mm512_and_si512(_mm512_srli_epi32(runes_vec.zmm, 8), _mm512_set1_epi32(0x3FU))); // (cont & 0x3F)
        _mm512_mask_storeu_epi32(runes, sz_u32_mask_until_(runes_to_place), runes_vec.zmm);
        *runes_unpacked = runes_to_place;
        return text + runes_to_place * 2;
    }

    // Check for the number of 3-byte characters - in this case we can't easily cast to 16-bit integers
    // and check for equality, but we can pre-define the masks and values we expect at each byte position.
    // For 3-byte UTF-8 sequences, we check if bytes match the pattern: 1110xxxx 10xxxxxx 10xxxxxx
    // We need to check every 3rd byte starting from position 0.
    sz_u512_vec_t three_byte_mask_vec, three_byte_pattern_vec;
    three_byte_mask_vec.zmm = _mm512_set1_epi32(0x00C0C0F0);    // Mask: [F0, C0, C0, 00] per 4-byte slot
    three_byte_pattern_vec.zmm = _mm512_set1_epi32(0x008080E0); // Pattern: [E0, 80, 80, 00] per 4-byte slot

    // Create permutation indices to gather 3-byte sequences into 4-byte slots
    // Input:  [b0 b1 b2]    [b3 b4 b5]    [b6 b7 b8]    ... (up to 16 triplets from 48 bytes)
    // Output: [b0 b1 b2 XX] [b3 b4 b5 XX] [b6 b7 b8 XX] ... (16 slots, 4th byte zeroed)
    sz_u512_vec_t permute_indices;
    permute_indices.zmm = _mm512_setr_epi32(
        // Triplets 0-3:  [0,1,2,_] [3,4,5,_] [6,7,8,_] [9,10,11,_]
        0x40020100, 0x40050403, 0x40080706, 0x400B0A09,
        // Triplets 4-7:  [12,13,14,_] [15,16,17,_] [18,19,20,_] [21,22,23,_]
        0x400E0D0C, 0x40111010, 0x40141312, 0x40171615,
        // Triplets 8-11: [24,25,26,_] [27,28,29,_] [30,31,32,_] [33,34,35,_]
        0x401A1918, 0x401D1C1B, 0x40201F1E, 0x40232221,
        // Triplets 12-15: [36,37,38,_] [39,40,41,_] [42,43,44,_] [45,46,47,_]
        0x40262524, 0x40292827, 0x402C2B2A, 0x402F2E2D);

    // Permute to gather triplets into slots
    sz_u512_vec_t gathered_triplets;
    gathered_triplets.zmm = _mm512_permutexvar_epi8(permute_indices.zmm, text_vec.zmm);

    // Check if gathered bytes match 3-byte UTF-8 pattern
    sz_u512_vec_t masked_triplets;
    masked_triplets.zmm = _mm512_and_si512(gathered_triplets.zmm, three_byte_mask_vec.zmm);
    __mmask16 three_byte_match_mask = _mm512_cmpeq_epi32_mask(masked_triplets.zmm, three_byte_pattern_vec.zmm);
    sz_size_t three_byte_prefix_length = sz_u64_ctz(~three_byte_match_mask);

    if (three_byte_prefix_length) {
        // Unpack up to 16 three-byte characters (48 bytes of input).
        sz_size_t runes_to_place = sz_min_of_three(three_byte_prefix_length, 16, runes_capacity);
        // Decode: ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F)
        // gathered_triplets has: [b0, b1, b2, XX] in each 32-bit slot (little-endian: 0xXXb2b1b0)
        // Extract: b0 from bits 7-0, b1 from bits 15-8, b2 from bits 23-16
        runes_vec.zmm = _mm512_or_si512(
            _mm512_or_si512(
                // (b0 & 0x0F) << 12
                _mm512_slli_epi32(_mm512_and_si512(gathered_triplets.zmm, _mm512_set1_epi32(0x0FU)), 12),
                // (b1 & 0x3F) << 6
                _mm512_slli_epi32(
                    _mm512_and_si512(_mm512_srli_epi32(gathered_triplets.zmm, 8), _mm512_set1_epi32(0x3FU)), 6)),
            _mm512_and_si512(_mm512_srli_epi32(gathered_triplets.zmm, 16), _mm512_set1_epi32(0x3FU))); // (b2 & 0x3F)
        _mm512_mask_storeu_epi32(runes, sz_u16_mask_until_(runes_to_place), runes_vec.zmm);
        *runes_unpacked = runes_to_place;
        return text + runes_to_place * 3;
    }

    // Check for the number of 4-byte characters
    // For 4-byte UTF-8 sequences: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
    // With a homogeneous 4-byte prefix, we have perfect 4-byte alignment (up to 16 sequences in 64 bytes)
    sz_u512_vec_t four_byte_mask_vec, four_byte_pattern_vec;
    four_byte_mask_vec.zmm = _mm512_set1_epi32((int)0xC0C0C0F8);    // Mask: [F8, C0, C0, C0] per 4-byte slot
    four_byte_pattern_vec.zmm = _mm512_set1_epi32((int)0x808080F0); // Pattern: [F0, 80, 80, 80] per 4-byte slot

    // Mask and check for 4-byte pattern in each 32-bit slot
    sz_u512_vec_t masked_quads;
    masked_quads.zmm = _mm512_and_si512(text_vec.zmm, four_byte_mask_vec.zmm);
    __mmask16 four_byte_match_mask = _mm512_cmpeq_epi32_mask(masked_quads.zmm, four_byte_pattern_vec.zmm);
    sz_size_t four_byte_prefix_length = sz_u64_ctz(~four_byte_match_mask);

    if (four_byte_prefix_length) {
        // Unpack up to 16 four-byte characters (64 bytes of input).
        sz_size_t runes_to_place = sz_min_of_three(four_byte_prefix_length, 16, runes_capacity);
        // Decode: ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F)
        runes_vec.zmm = _mm512_or_si512(
            _mm512_or_si512(
                // (b0 & 0x07) << 18
                _mm512_slli_epi32(_mm512_and_si512(text_vec.zmm, _mm512_set1_epi32(0x07U)), 18),
                // (b1 & 0x3F) << 12
                _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(text_vec.zmm, 8), _mm512_set1_epi32(0x3FU)), 12)),
            _mm512_or_si512(
                // (b2 & 0x3F) << 6
                _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(text_vec.zmm, 16), _mm512_set1_epi32(0x3FU)), 6),
                // (b3 & 0x3F)
                _mm512_and_si512(_mm512_srli_epi32(text_vec.zmm, 24), _mm512_set1_epi32(0x3FU))));
        _mm512_mask_storeu_epi32(runes, sz_u16_mask_until_(runes_to_place), runes_vec.zmm);
        *runes_unpacked = runes_to_place;
        return text + runes_to_place * 4;
    }

    // Seems like broken unicoode?
    *runes_unpacked = 0;
    return text;
}

SZ_PUBLIC sz_size_t sz_utf8_case_fold_ice(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {
    return sz_utf8_case_fold_serial(source, source_length, destination);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_case_insensitive_ice( //
    sz_cptr_t haystack, sz_size_t haystack_length,     //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length) {
    return sz_utf8_find_case_insensitive_serial(haystack, haystack_length, needle, needle_length, matched_length);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_ICE
#pragma endregion // Ice Lake Implementation

#pragma region Haswell Implementation
#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2,bmi,bmi2,popcnt")
#endif

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_haswell( //
    sz_cptr_t text, sz_size_t length,             //
    sz_rune_t *runes, sz_size_t runes_capacity,   //
    sz_size_t *runes_unpacked) {
    return sz_utf8_unpack_chunk_serial(text, length, runes, runes_capacity, runes_unpacked);
}

SZ_PUBLIC sz_size_t sz_utf8_case_fold_haswell(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {
    return sz_utf8_case_fold_serial(source, source_length, destination);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_case_insensitive_haswell( //
    sz_cptr_t haystack, sz_size_t haystack_length,         //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length) {
    return sz_utf8_find_case_insensitive_serial(haystack, haystack_length, needle, needle_length, matched_length);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_HASWELL
#pragma endregion // Haswell Implementation

#pragma region NEON Implementation

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_neon(  //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {
    return sz_utf8_unpack_chunk_serial(text, length, runes, runes_capacity, runes_unpacked);
}

SZ_PUBLIC sz_size_t sz_utf8_case_fold_neon(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {
    return sz_utf8_case_fold_serial(source, source_length, destination);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_case_insensitive_neon( //
    sz_cptr_t haystack, sz_size_t haystack_length,      //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length) {
    return sz_utf8_find_case_insensitive_serial(haystack, haystack_length, needle, needle_length, matched_length);
}

#pragma endregion // NEON Implementation

#pragma region Dynamic Dispatch

#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_cptr_t sz_utf8_unpack_chunk(sz_cptr_t text, sz_size_t length, sz_rune_t *runes, sz_size_t runes_capacity,
                                          sz_size_t *runes_unpacked) {
#if SZ_USE_ICE
    return sz_utf8_unpack_chunk_ice(text, length, runes, runes_capacity, runes_unpacked);
#elif SZ_USE_HASWELL
    return sz_utf8_unpack_chunk_haswell(text, length, runes, runes_capacity, runes_unpacked);
#else
    return sz_utf8_unpack_chunk_serial(text, length, runes, runes_capacity, runes_unpacked);
#endif
}

SZ_DYNAMIC sz_size_t sz_utf8_case_fold(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {
#if SZ_USE_ICE
    return sz_utf8_case_fold_ice(source, source_length, destination);
#elif SZ_USE_HASWELL
    return sz_utf8_case_fold_haswell(source, source_length, destination);
#else
    return sz_utf8_case_fold_serial(source, source_length, destination);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_utf8_find_case_insensitive(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                   sz_size_t needle_length, sz_size_t *matched_length) {
#if SZ_USE_ICE
    return sz_utf8_find_case_insensitive_ice(haystack, haystack_length, needle, needle_length, matched_length);
#elif SZ_USE_HASWELL
    return sz_utf8_find_case_insensitive_haswell(haystack, haystack_length, needle, needle_length, matched_length);
#else
    return sz_utf8_find_case_insensitive_serial(haystack, haystack_length, needle, needle_length, matched_length);
#endif
}

#endif // !SZ_DYNAMIC_DISPATCH

#pragma endregion // Dynamic Dispatch

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_UNPACK_H_
