/**
 *  @brief Serial backend for UTF-8 case folding.
 *  @file include/stringzilla/utf8_uncased_fold/serial.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased_fold.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_FOLD_SERIAL_H_
#define STRINGZILLA_UTF8_UNCASED_FOLD_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h"

#ifdef __cplusplus
extern "C" {
#endif
/**
 *  @brief Branchless ASCII case fold - converts A-Z to a-z.
 *  Uses unsigned subtraction trick: (c - 'A') <= 25 is true only for uppercase letters.
 */
SZ_HELPER_AUTO sz_u8_t sz_ascii_fold_(sz_u8_t c) { return c + (((sz_u8_t)(c - 'A') <= 25u) * 0x20); }

/**
 *  @brief Folded-rune representation of a byte that does not begin a well-formed codepoint.
 *
 *  A malformed byte folds to itself and is matched/compared byte-for-byte, never as a Unicode codepoint.
 *  Tagging it above the valid Unicode range (0x10FFFF) keeps it distinct from every real folded rune, so a
 *  lone malformed byte 0xFC can only match another malformed 0xFC - never the valid rune U+00FC ('ü'). Two
 *  equal malformed bytes still produce equal tagged runes, preserving byte-for-byte matching.
 */
SZ_HELPER_INLINE sz_rune_t sz_rune_malformed_byte_(sz_u8_t byte) { return 0x80000000u | (sz_rune_t)byte; }
/**  Helper macro for readable assertions - use for SIMD implementation reference */
#define sz_is_in_range_(x, low, high) ((x) >= (low) && (x) <= (high))

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
SZ_HELPER_AUTO sz_size_t sz_unicode_fold_codepoint_(sz_rune_t rune, sz_rune_t *folded) {

    // 1-byte UTF-8 (U+0000-007F): ASCII - only A-Z needs folding
    if (rune <= 0x7F) {
        if ((sz_u32_t)(rune - 0x41) <= 25) { // A-Z: 0x41-0x5A (26 chars)
            sz_assert_(sz_is_in_range_(rune, 0x0041, 0x005A));
            folded[0] = rune + 0x20;
            return 1;
        }
        folded[0] = rune;
        return 1; // digits, punctuation, control chars unchanged
    }

    // 2-byte UTF-8 (U+0080-07FF): Latin, Greek, Cyrillic, Armenian
    if (rune <= 0x7FF) {
        // Cyrillic А-я: 0x0410-0x044F (upper 0x0410-0x042F, lower 0x0430-0x044F)
        if ((sz_u32_t)(rune - 0x0410) <= 0x3F) {
            sz_assert_(sz_is_in_range_(rune, 0x0410, 0x044F));
            folded[0] = rune + ((rune <= 0x042F) * 0x20);
            return 1;
        } // +32 if upper, +0 if lower

        // Latin-1 À-þ: 0x00C0-0x00FE (upper 0x00C0-0x00DE, lower 0x00E0-0x00FE)
        if ((sz_u32_t)(rune - 0x00C0) <= 0x3E) {
            sz_assert_(sz_is_in_range_(rune, 0x00C0, 0x00FE));
            if ((rune | 0x20) == 0xF7) {
                folded[0] = rune;
                return 1;
            } // × (D7) and ÷ (F7) unchanged
            // 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
            if (rune == 0x00DF) {
                folded[0] = 0x0073;
                folded[1] = 0x0073;
                return 2;
            }
            folded[0] = rune + ((rune <= 0x00DE) * 0x20);
            return 1;
        }

        // Greek Α-Ρ: 0x0391-0x03A1 → α-ρ (+32)
        if ((sz_u32_t)(rune - 0x0391) <= 0x10) {
            sz_assert_(sz_is_in_range_(rune, 0x0391, 0x03A1));
            folded[0] = rune + 0x20;
            return 1;
        }

        // Greek Σ-Ϋ: 0x03A3-0x03AB → σ-ϋ (+32)
        if ((sz_u32_t)(rune - 0x03A3) <= 0x08) {
            sz_assert_(sz_is_in_range_(rune, 0x03A3, 0x03AB));
            folded[0] = rune + 0x20;
            return 1;
        }

        // Cyrillic Ѐ-Џ: 0x0400-0x040F → ѐ-џ (+80)
        if ((sz_u32_t)(rune - 0x0400) <= 0x0F) {
            sz_assert_(sz_is_in_range_(rune, 0x0400, 0x040F));
            folded[0] = rune + 0x50;
            return 1;
        }

        // Armenian Ա-Ֆ: 0x0531-0x0556 → ա-ֆ (+48)
        if ((sz_u32_t)(rune - 0x0531) <= 0x25) {
            sz_assert_(sz_is_in_range_(rune, 0x0531, 0x0556));
            folded[0] = rune + 0x30;
            return 1;
        }

        // Greek Έ-Ί: 0x0388-0x038A (+37)
        if ((sz_u32_t)(rune - 0x0388) <= 0x02) {
            sz_assert_(sz_is_in_range_(rune, 0x0388, 0x038A));
            folded[0] = rune + 0x25;
            return 1;
        }

        // Greek Ͻ-Ͽ: 0x03FD-0x03FF → ͻ-Ϳ (-130)
        if ((sz_u32_t)(rune - 0x03FD) <= 0x02) {
            sz_assert_(sz_is_in_range_(rune, 0x03FD, 0x03FF));
            folded[0] = rune - 130;
            return 1;
        }

        // Next let's handle the even/odd parity-based ranges
        sz_u32_t is_even = ((rune & 1) == 0);

        // Latin Extended-A: Ā-Į (0x0100-0x012E, even → +1)
        if ((sz_u32_t)(rune - 0x0100) <= 0x2E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0100, 0x012E));
            folded[0] = rune + 1;
            return 1;
        }

        // Latin Extended-A: Ĳ-Ķ (0x0132-0x0136, even → +1)
        if ((sz_u32_t)(rune - 0x0132) <= 0x04 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0132, 0x0136));
            folded[0] = rune + 1;
            return 1;
        }

        // Latin Extended-A: Ĺ-Ň (0x0139-0x0147, odd → +1)
        if ((sz_u32_t)(rune - 0x0139) <= 0x0E && !is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0139, 0x0147));
            folded[0] = rune + 1;
            return 1;
        }

        // Latin Extended-A: Ŋ-Ŷ (0x014A-0x0176, even → +1)
        if ((sz_u32_t)(rune - 0x014A) <= 0x2C && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x014A, 0x0176));
            folded[0] = rune + 1;
            return 1;
        }

        // Latin Extended-A: Ź-Ž (0x0179-0x017D, odd → +1)
        if ((sz_u32_t)(rune - 0x0179) <= 0x04 && !is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0179, 0x017D));
            folded[0] = rune + 1;
            return 1;
        }

        // Latin Extended-B: Ǎ-Ǜ (0x01CD-0x01DB, odd → +1)
        if ((sz_u32_t)(rune - 0x01CD) <= 0x0E && !is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x01CD, 0x01DB));
            folded[0] = rune + 1;
            return 1;
        }

        // Latin Extended-B: Ǟ-Ǯ (0x01DE-0x01EE, even → +1)
        if ((sz_u32_t)(rune - 0x01DE) <= 0x10 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x01DE, 0x01EE));
            folded[0] = rune + 1;
            return 1;
        }

        // Latin Extended-B: Ǹ-Ǿ (0x01F8-0x01FE, even → +1)
        if ((sz_u32_t)(rune - 0x01F8) <= 0x06 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x01F8, 0x01FE));
            folded[0] = rune + 1;
            return 1;
        }

        // Latin Extended-B: Ȁ-Ȟ (0x0200-0x021E, even → +1)
        if ((sz_u32_t)(rune - 0x0200) <= 0x1E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0200, 0x021E));
            folded[0] = rune + 1;
            return 1;
        }

        // Latin Extended-B: Ȣ-Ȳ (0x0222-0x0232, even → +1)
        if ((sz_u32_t)(rune - 0x0222) <= 0x10 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0222, 0x0232));
            folded[0] = rune + 1;
            return 1;
        }

        // Latin Extended-B: Ɇ-Ɏ (0x0246-0x024E, even → +1)
        if ((sz_u32_t)(rune - 0x0246) <= 0x08 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0246, 0x024E));
            folded[0] = rune + 1;
            return 1;
        }

        // Greek archaic: Ͱ-Ͳ (0x0370-0x0372, even → +1)
        if ((sz_u32_t)(rune - 0x0370) <= 0x02 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0370, 0x0372));
            folded[0] = rune + 1;
            return 1;
        }

        // Greek archaic: Ϙ-Ϯ (0x03D8-0x03EE, even → +1)
        if ((sz_u32_t)(rune - 0x03D8) <= 0x16 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x03D8, 0x03EE));
            folded[0] = rune + 1;
            return 1;
        }

        // Cyrillic extended: Ѡ-Ҁ (0x0460-0x0480, even → +1)
        if ((sz_u32_t)(rune - 0x0460) <= 0x20 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0460, 0x0480));
            folded[0] = rune + 1;
            return 1;
        }

        // Cyrillic extended: Ҋ-Ҿ (0x048A-0x04BE, even → +1)
        if ((sz_u32_t)(rune - 0x048A) <= 0x34 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x048A, 0x04BE));
            folded[0] = rune + 1;
            return 1;
        }

        // Cyrillic extended: Ӂ-Ӎ (0x04C1-0x04CD, odd → +1)
        if ((sz_u32_t)(rune - 0x04C1) <= 0x0C && !is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x04C1, 0x04CD));
            folded[0] = rune + 1;
            return 1;
        }

        // Cyrillic extended: Ӑ-Ӿ (0x04D0-0x04FE, even → +1)
        if ((sz_u32_t)(rune - 0x04D0) <= 0x2E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x04D0, 0x04FE));
            folded[0] = rune + 1;
            return 1;
        }

        // Cyrillic extended: Ԁ-Ԯ (0x0500-0x052E, even → +1)
        if ((sz_u32_t)(rune - 0x0500) <= 0x2E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0500, 0x052E));
            folded[0] = rune + 1;
            return 1;
        }

        // Next let's handle the 2-byte irregular one-to-one mappings
        switch (rune) {
        // Latin-1 Supplement specials
        case 0x00B5: folded[0] = 0x03BC; return 1; // 'µ' (U+00B5, C2 B5) → 'μ' (U+03BC, CE BC)
        case 0x0178: folded[0] = 0x00FF; return 1; // 'Ÿ' (U+0178, C5 B8) → 'ÿ' (U+00FF, C3 BF)
        case 0x017F:
            folded[0] = 0x0073;
            return 1; // 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
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
        case 0x01BC:
            folded[0] = 0x01BD;
            return 1; // 'Ƽ' (U+01BC, C6 BC) → 'ƽ' (U+01BD, C6 BD)
        // Digraphs: Serbian/Croatian DŽ, LJ, NJ and DZ
        case 0x01C4: folded[0] = 0x01C6; return 1; // 'Ǆ' (U+01C4, C7 84) → 'ǆ' (U+01C6, C7 86)
        case 0x01C5: folded[0] = 0x01C6; return 1; // 'ǅ' (U+01C5, C7 85) → 'ǆ' (U+01C6, C7 86)
        case 0x01C7: folded[0] = 0x01C9; return 1; // 'Ǉ' (U+01C7, C7 87) → 'ǉ' (U+01C9, C7 89)
        case 0x01C8: folded[0] = 0x01C9; return 1; // 'ǈ' (U+01C8, C7 88) → 'ǉ' (U+01C9, C7 89)
        case 0x01CA: folded[0] = 0x01CC; return 1; // 'Ǌ' (U+01CA, C7 8A) → 'ǌ' (U+01CC, C7 8C)
        case 0x01CB: folded[0] = 0x01CC; return 1; // 'ǋ' (U+01CB, C7 8B) → 'ǌ' (U+01CC, C7 8C)
        case 0x01F1: folded[0] = 0x01F3; return 1; // 'Ǳ' (U+01F1, C7 B1) → 'ǳ' (U+01F3, C7 B3)
        case 0x01F2:
            folded[0] = 0x01F3;
            return 1; // 'ǲ' (U+01F2, C7 B2) → 'ǳ' (U+01F3, C7 B3)
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
        case 0x0245:
            folded[0] = 0x028C;
            return 1; // 'Ʌ' (U+0245, C9 85) → 'ʌ' (U+028C, CA 8C)
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
        case 0x03FA:
            folded[0] = 0x03FB;
            return 1; // 'Ϻ' (U+03FA, CF BA) → 'ϻ' (U+03FB, CF BB)
        // Cyrillic: palochka
        case 0x04C0: folded[0] = 0x04CF; return 1; // 'Ӏ' (U+04C0, D3 80) → 'ӏ' (U+04CF, D3 8F)
        }

        // 2-byte one-to-many expansions
        switch (rune) {
        // ß handled inline in Latin-1 range above... interestingly the capital Eszett is in the 3-byte range!
        // case 0x00DF: folded[0] = 0x0073; folded[1] = 0x0073; return 2;

        // 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
        case 0x0130:
            folded[0] = 0x0069;
            folded[1] = 0x0307;
            return 2;
        // 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
        case 0x0149:
            folded[0] = 0x02BC;
            folded[1] = 0x006E;
            return 2;
        // 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C)
        case 0x01F0:
            folded[0] = 0x006A;
            folded[1] = 0x030C;
            return 2;
        // 'ΐ' (U+0390, CE 90) → "ΐ" (U+03B9 U+0308 U+0301, CE B9 CC 88 CC 81)
        case 0x0390:
            folded[0] = 0x03B9;
            folded[1] = 0x0308;
            folded[2] = 0x0301;
            return 3;
        // 'ΰ' (U+03B0, CE B0) → "ΰ" (U+03C5 U+0308 U+0301, CF 85 CC 88 CC 81)
        case 0x03B0:
            folded[0] = 0x03C5;
            folded[1] = 0x0308;
            folded[2] = 0x0301;
            return 3;
        // 'և' (U+0587, D6 87) → "եւ" (U+0565 U+0582, D5 A5 D6 82)
        case 0x0587:
            folded[0] = 0x0565;
            folded[1] = 0x0582;
            return 2;
        }

        folded[0] = rune;
        return 1; // 2-byte: no folding needed
    }

    // 3-byte UTF-8 (U+0800-FFFF): Georgian, Cherokee, Greek Extended, etc.
    if (rune <= 0xFFFF) {
        // Georgian Ⴀ-Ⴥ: 0x10A0-0x10C5 (+7264)
        if ((sz_u32_t)(rune - 0x10A0) <= 0x25) {
            sz_assert_(sz_is_in_range_(rune, 0x10A0, 0x10C5));
            folded[0] = rune + 0x1C60;
            return 1;
        }

        // Georgian Mtavruli Ა-Ჺ: 0x1C90-0x1CBA (-3008)
        if ((sz_u32_t)(rune - 0x1C90) <= 0x2A) {
            sz_assert_(sz_is_in_range_(rune, 0x1C90, 0x1CBA));
            folded[0] = rune - 0xBC0;
            return 1;
        }

        // Georgian Mtavruli Ჽ-Ჿ: 0x1CBD-0x1CBF (-3008)
        if ((sz_u32_t)(rune - 0x1CBD) <= 0x02) {
            sz_assert_(sz_is_in_range_(rune, 0x1CBD, 0x1CBF));
            folded[0] = rune - 0xBC0;
            return 1;
        }

        // Cherokee Ᏸ-Ᏽ: 0x13F8-0x13FD (-8)
        if ((sz_u32_t)(rune - 0x13F8) <= 0x05) {
            sz_assert_(sz_is_in_range_(rune, 0x13F8, 0x13FD));
            folded[0] = rune - 8;
            return 1;
        }

        // Cherokee Ꭰ-Ᏼ: 0xAB70-0xABBF → Ꭰ-Ᏼ: 0x13A0-0x13EF (-38864)
        if ((sz_u32_t)(rune - 0xAB70) <= 0x4F) {
            sz_assert_(sz_is_in_range_(rune, 0xAB70, 0xABBF));
            folded[0] = rune - 0x97D0;
            return 1;
        }

        // Greek Extended: multiple -8 offset ranges
        if ((sz_u32_t)(rune - 0x1F08) <= 0x07) { // Ἀ-Ἇ
            sz_assert_(sz_is_in_range_(rune, 0x1F08, 0x1F0F));
            folded[0] = rune - 8;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x1F18) <= 0x05) { // Ἐ-Ἕ
            sz_assert_(sz_is_in_range_(rune, 0x1F18, 0x1F1D));
            folded[0] = rune - 8;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x1F28) <= 0x07) { // Ἠ-Ἧ
            sz_assert_(sz_is_in_range_(rune, 0x1F28, 0x1F2F));
            folded[0] = rune - 8;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x1F38) <= 0x07) { // Ἰ-Ἷ
            sz_assert_(sz_is_in_range_(rune, 0x1F38, 0x1F3F));
            folded[0] = rune - 8;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x1F48) <= 0x05) { // Ὀ-Ὅ
            sz_assert_(sz_is_in_range_(rune, 0x1F48, 0x1F4D));
            folded[0] = rune - 8;
            return 1;
        }
        if ((sz_u32_t)(rune - 0x1F68) <= 0x07) { // Ὠ-Ὧ
            sz_assert_(sz_is_in_range_(rune, 0x1F68, 0x1F6F));
            folded[0] = rune - 8;
            return 1;
        }

        // Greek Extended Ὲ-Ή: 0x1FC8-0x1FCB (-86)
        if ((sz_u32_t)(rune - 0x1FC8) <= 0x03) {
            sz_assert_(sz_is_in_range_(rune, 0x1FC8, 0x1FCB));
            folded[0] = rune - 86;
            return 1;
        }

        // Roman numerals Ⅰ-Ⅿ: 0x2160-0x216F (+16)
        if ((sz_u32_t)(rune - 0x2160) <= 0x0F) {
            sz_assert_(sz_is_in_range_(rune, 0x2160, 0x216F));
            folded[0] = rune + 0x10;
            return 1;
        }

        // Circled letters Ⓐ-Ⓩ: 0x24B6-0x24CF (+26)
        if ((sz_u32_t)(rune - 0x24B6) <= 0x19) {
            sz_assert_(sz_is_in_range_(rune, 0x24B6, 0x24CF));
            folded[0] = rune + 0x1A;
            return 1;
        }

        // Glagolitic Ⰰ-Ⱟ: 0x2C00-0x2C2F (+48)
        if ((sz_u32_t)(rune - 0x2C00) <= 0x2F) {
            sz_assert_(sz_is_in_range_(rune, 0x2C00, 0x2C2F));
            folded[0] = rune + 0x30;
            return 1;
        }

        // Fullwidth Ａ-Ｚ: 0xFF21-0xFF3A (+32)
        if ((sz_u32_t)(rune - 0xFF21) <= 0x19) {
            sz_assert_(sz_is_in_range_(rune, 0xFF21, 0xFF3A));
            folded[0] = rune + 0x20;
            return 1;
        }

        // Next let's handle the even/odd parity-based ranges
        sz_u32_t is_even = ((rune & 1) == 0);

        // Latin Extended Additional Ḁ-Ẕ: 0x1E00-0x1E94
        if ((sz_u32_t)(rune - 0x1E00) <= 0x94 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x1E00, 0x1E94));
            folded[0] = rune + 1;
            return 1;
        }

        // Latin Extended Additional (Vietnamese) Ạ-Ỿ: 0x1EA0-0x1EFE
        if ((sz_u32_t)(rune - 0x1EA0) <= 0x5E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x1EA0, 0x1EFE));
            folded[0] = rune + 1;
            return 1;
        }

        // Coptic Ⲁ-Ⳣ: 0x2C80-0x2CE2
        if ((sz_u32_t)(rune - 0x2C80) <= 0x62 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x2C80, 0x2CE2));
            folded[0] = rune + 1;
            return 1;
        }

        // Cyrillic Extended-B Ꙁ-Ꙭ: 0xA640-0xA66C
        if ((sz_u32_t)(rune - 0xA640) <= 0x2C && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0xA640, 0xA66C));
            folded[0] = rune + 1;
            return 1;
        }

        // Cyrillic Extended-B Ꚁ-Ꚛ: 0xA680-0xA69A
        if ((sz_u32_t)(rune - 0xA680) <= 0x1A && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0xA680, 0xA69A));
            folded[0] = rune + 1;
            return 1;
        }

        // Latin Extended-D ranges
        if ((sz_u32_t)(rune - 0xA722) <= 0x0C && is_even) { // Ꜣ-Ꜯ
            sz_assert_(sz_is_in_range_(rune, 0xA722, 0xA72E));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0xA732) <= 0x3C && is_even) { // Ꜳ-Ꝯ
            sz_assert_(sz_is_in_range_(rune, 0xA732, 0xA76E));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0xA77E) <= 0x08 && is_even) { // Ꝿ-Ꞇ
            sz_assert_(sz_is_in_range_(rune, 0xA77E, 0xA786));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0xA790) <= 0x02 && is_even) { // Ꞑ-Ꞓ
            sz_assert_(sz_is_in_range_(rune, 0xA790, 0xA792));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0xA796) <= 0x12 && is_even) { // Ꞗ-Ꞩ
            sz_assert_(sz_is_in_range_(rune, 0xA796, 0xA7A8));
            folded[0] = rune + 1;
            return 1;
        }
        if ((sz_u32_t)(rune - 0xA7B4) <= 0x0E && is_even) { // Ꞵ-Ꟃ
            sz_assert_(sz_is_in_range_(rune, 0xA7B4, 0xA7C2));
            folded[0] = rune + 1;
            return 1;
        }

        // Next let's handle the 3-byte irregular one-to-one mappings
        switch (rune) {
        // Georgian irregular
        case 0x10C7: folded[0] = 0x2D27; return 1; // 'Ⴧ' (U+10C7, E1 83 87) → 'ⴧ' (U+2D27, E2 B4 A7)
        case 0x10CD:
            folded[0] = 0x2D2D;
            return 1; // 'Ⴭ' (U+10CD, E1 83 8D) → 'ⴭ' (U+2D2D, E2 B4 AD)
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
        case 0x1C89:
            folded[0] = 0x1C8A;
            return 1; // 'Ᲊ' (U+1C89, E1 B2 89) → 'ᲊ' (U+1C8A, E1 B2 8A)
        // Latin Extended Additional: long s with dot
        case 0x1E9B:
            folded[0] = 0x1E61;
            return 1; // 'ẛ' (U+1E9B, E1 BA 9B) → 'ṡ' (U+1E61, E1 B9 A1)
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
        case 0x1FFB:
            folded[0] = 0x1F7D;
            return 1; // 'Ώ' (U+1FFB, E1 BF BB) → 'ώ' (U+1F7D, E1 BD BD)
        // Letterlike Symbols: compatibility mappings
        case 0x2126: folded[0] = 0x03C9; return 1; // 'Ω' (U+2126, E2 84 A6) → 'ω' (U+03C9, CF 89)
        case 0x212A: folded[0] = 0x006B; return 1; // 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B)
        case 0x212B: folded[0] = 0x00E5; return 1; // 'Å' (U+212B, E2 84 AB) → 'å' (U+00E5, C3 A5)
        case 0x2132: folded[0] = 0x214E; return 1; // 'Ⅎ' (U+2132, E2 84 B2) → 'ⅎ' (U+214E, E2 85 8E)
        case 0x2183:
            folded[0] = 0x2184;
            return 1; // 'Ↄ' (U+2183, E2 86 83) → 'ↄ' (U+2184, E2 86 84)
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
        case 0x2C7F:
            folded[0] = 0x0240;
            return 1; // 'Ɀ' (U+2C7F, E2 B1 BF) → 'ɀ' (U+0240, C9 80)
        // Coptic: irregular cases outside even/odd range
        case 0x2CEB: folded[0] = 0x2CEC; return 1; // 'Ⳬ' (U+2CEB, E2 B3 AB) → 'ⳬ' (U+2CEC, E2 B3 AC)
        case 0x2CED: folded[0] = 0x2CEE; return 1; // 'Ⳮ' (U+2CED, E2 B3 AD) → 'ⳮ' (U+2CEE, E2 B3 AE)
        case 0x2CF2:
            folded[0] = 0x2CF3;
            return 1; // 'Ⳳ' (U+2CF2, E2 B3 B2) → 'ⳳ' (U+2CF3, E2 B3 B3)
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
        case 0x1E96:
            folded[0] = 0x0068;
            folded[1] = 0x0331;
            return 2;
        // 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
        case 0x1E97:
            folded[0] = 0x0074;
            folded[1] = 0x0308;
            return 2;
        // 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
        case 0x1E98:
            folded[0] = 0x0077;
            folded[1] = 0x030A;
            return 2;
        // 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
        case 0x1E99:
            folded[0] = 0x0079;
            folded[1] = 0x030A;
            return 2;
        // 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
        case 0x1E9A:
            folded[0] = 0x0061;
            folded[1] = 0x02BE;
            return 2;
        // 'ẞ' (U+1E9E, E1 BA 9E) → "ss" (U+0073 U+0073, 73 73)
        case 0x1E9E:
            folded[0] = 0x0073;
            folded[1] = 0x0073;
            return 2;
        // Greek Extended: breathing marks
        // 'ὐ' (U+1F50, E1 BD 90) → "ὐ" (U+03C5 U+0313, CF 85 CC 93)
        case 0x1F50:
            folded[0] = 0x03C5;
            folded[1] = 0x0313;
            return 2;
        // 'ὒ' (U+1F52, E1 BD 92) → "ὒ" (U+03C5 U+0313 U+0300, CF 85 CC 93 CC 80)
        case 0x1F52:
            folded[0] = 0x03C5;
            folded[1] = 0x0313;
            folded[2] = 0x0300;
            return 3;
        // 'ὔ' (U+1F54, E1 BD 94) → "ὔ" (U+03C5 U+0313 U+0301, CF 85 CC 93 CC 81)
        case 0x1F54:
            folded[0] = 0x03C5;
            folded[1] = 0x0313;
            folded[2] = 0x0301;
            return 3;
        // 'ὖ' (U+1F56, E1 BD 96) → "ὖ" (U+03C5 U+0313 U+0342, CF 85 CC 93 CD 82)
        case 0x1F56:
            folded[0] = 0x03C5;
            folded[1] = 0x0313;
            folded[2] = 0x0342;
            return 3;
        // Greek Extended: iota subscript combinations (0x1F80-0x1FAF)
        // 'ᾀ' (U+1F80, E1 BE 80) → "ἀι" (U+1F00 U+03B9, E1 BC 80 CE B9)
        case 0x1F80:
            folded[0] = 0x1F00;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾁ' (U+1F81, E1 BE 81) → "ἁι" (U+1F01 U+03B9, E1 BC 81 CE B9)
        case 0x1F81:
            folded[0] = 0x1F01;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾂ' (U+1F82, E1 BE 82) → "ἂι" (U+1F02 U+03B9, E1 BC 82 CE B9)
        case 0x1F82:
            folded[0] = 0x1F02;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾃ' (U+1F83, E1 BE 83) → "ἃι" (U+1F03 U+03B9, E1 BC 83 CE B9)
        case 0x1F83:
            folded[0] = 0x1F03;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾄ' (U+1F84, E1 BE 84) → "ἄι" (U+1F04 U+03B9, E1 BC 84 CE B9)
        case 0x1F84:
            folded[0] = 0x1F04;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾅ' (U+1F85, E1 BE 85) → "ἅι" (U+1F05 U+03B9, E1 BC 85 CE B9)
        case 0x1F85:
            folded[0] = 0x1F05;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾆ' (U+1F86, E1 BE 86) → "ἆι" (U+1F06 U+03B9, E1 BC 86 CE B9)
        case 0x1F86:
            folded[0] = 0x1F06;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾇ' (U+1F87, E1 BE 87) → "ἇι" (U+1F07 U+03B9, E1 BC 87 CE B9)
        case 0x1F87:
            folded[0] = 0x1F07;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾈ' (U+1F88, E1 BE 88) → "ἀι" (U+1F00 U+03B9, E1 BC 80 CE B9)
        case 0x1F88:
            folded[0] = 0x1F00;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾉ' (U+1F89, E1 BE 89) → "ἁι" (U+1F01 U+03B9, E1 BC 81 CE B9)
        case 0x1F89:
            folded[0] = 0x1F01;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾊ' (U+1F8A, E1 BE 8A) → "ἂι" (U+1F02 U+03B9, E1 BC 82 CE B9)
        case 0x1F8A:
            folded[0] = 0x1F02;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾋ' (U+1F8B, E1 BE 8B) → "ἃι" (U+1F03 U+03B9, E1 BC 83 CE B9)
        case 0x1F8B:
            folded[0] = 0x1F03;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾌ' (U+1F8C, E1 BE 8C) → "ἄι" (U+1F04 U+03B9, E1 BC 84 CE B9)
        case 0x1F8C:
            folded[0] = 0x1F04;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾍ' (U+1F8D, E1 BE 8D) → "ἅι" (U+1F05 U+03B9, E1 BC 85 CE B9)
        case 0x1F8D:
            folded[0] = 0x1F05;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾎ' (U+1F8E, E1 BE 8E) → "ἆι" (U+1F06 U+03B9, E1 BC 86 CE B9)
        case 0x1F8E:
            folded[0] = 0x1F06;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾏ' (U+1F8F, E1 BE 8F) → "ἇι" (U+1F07 U+03B9, E1 BC 87 CE B9)
        case 0x1F8F:
            folded[0] = 0x1F07;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾐ' (U+1F90, E1 BE 90) → "ἠι" (U+1F20 U+03B9, E1 BC A0 CE B9)
        case 0x1F90:
            folded[0] = 0x1F20;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾑ' (U+1F91, E1 BE 91) → "ἡι" (U+1F21 U+03B9, E1 BC A1 CE B9)
        case 0x1F91:
            folded[0] = 0x1F21;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾒ' (U+1F92, E1 BE 92) → "ἢι" (U+1F22 U+03B9, E1 BC A2 CE B9)
        case 0x1F92:
            folded[0] = 0x1F22;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾓ' (U+1F93, E1 BE 93) → "ἣι" (U+1F23 U+03B9, E1 BC A3 CE B9)
        case 0x1F93:
            folded[0] = 0x1F23;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾔ' (U+1F94, E1 BE 94) → "ἤι" (U+1F24 U+03B9, E1 BC A4 CE B9)
        case 0x1F94:
            folded[0] = 0x1F24;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾕ' (U+1F95, E1 BE 95) → "ἥι" (U+1F25 U+03B9, E1 BC A5 CE B9)
        case 0x1F95:
            folded[0] = 0x1F25;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾖ' (U+1F96, E1 BE 96) → "ἦι" (U+1F26 U+03B9, E1 BC A6 CE B9)
        case 0x1F96:
            folded[0] = 0x1F26;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾗ' (U+1F97, E1 BE 97) → "ἧι" (U+1F27 U+03B9, E1 BC A7 CE B9)
        case 0x1F97:
            folded[0] = 0x1F27;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾘ' (U+1F98, E1 BE 98) → "ἠι" (U+1F20 U+03B9, E1 BC A0 CE B9)
        case 0x1F98:
            folded[0] = 0x1F20;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾙ' (U+1F99, E1 BE 99) → "ἡι" (U+1F21 U+03B9, E1 BC A1 CE B9)
        case 0x1F99:
            folded[0] = 0x1F21;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾚ' (U+1F9A, E1 BE 9A) → "ἢι" (U+1F22 U+03B9, E1 BC A2 CE B9)
        case 0x1F9A:
            folded[0] = 0x1F22;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾛ' (U+1F9B, E1 BE 9B) → "ἣι" (U+1F23 U+03B9, E1 BC A3 CE B9)
        case 0x1F9B:
            folded[0] = 0x1F23;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾜ' (U+1F9C, E1 BE 9C) → "ἤι" (U+1F24 U+03B9, E1 BC A4 CE B9)
        case 0x1F9C:
            folded[0] = 0x1F24;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾝ' (U+1F9D, E1 BE 9D) → "ἥι" (U+1F25 U+03B9, E1 BC A5 CE B9)
        case 0x1F9D:
            folded[0] = 0x1F25;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾞ' (U+1F9E, E1 BE 9E) → "ἦι" (U+1F26 U+03B9, E1 BC A6 CE B9)
        case 0x1F9E:
            folded[0] = 0x1F26;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾟ' (U+1F9F, E1 BE 9F) → "ἧι" (U+1F27 U+03B9, E1 BC A7 CE B9)
        case 0x1F9F:
            folded[0] = 0x1F27;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾠ' (U+1FA0, E1 BE A0) → "ὠι" (U+1F60 U+03B9, E1 BD A0 CE B9)
        case 0x1FA0:
            folded[0] = 0x1F60;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾡ' (U+1FA1, E1 BE A1) → "ὡι" (U+1F61 U+03B9, E1 BD A1 CE B9)
        case 0x1FA1:
            folded[0] = 0x1F61;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾢ' (U+1FA2, E1 BE A2) → "ὢι" (U+1F62 U+03B9, E1 BD A2 CE B9)
        case 0x1FA2:
            folded[0] = 0x1F62;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾣ' (U+1FA3, E1 BE A3) → "ὣι" (U+1F63 U+03B9, E1 BD A3 CE B9)
        case 0x1FA3:
            folded[0] = 0x1F63;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾤ' (U+1FA4, E1 BE A4) → "ὤι" (U+1F64 U+03B9, E1 BD A4 CE B9)
        case 0x1FA4:
            folded[0] = 0x1F64;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾥ' (U+1FA5, E1 BE A5) → "ὥι" (U+1F65 U+03B9, E1 BD A5 CE B9)
        case 0x1FA5:
            folded[0] = 0x1F65;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾦ' (U+1FA6, E1 BE A6) → "ὦι" (U+1F66 U+03B9, E1 BD A6 CE B9)
        case 0x1FA6:
            folded[0] = 0x1F66;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾧ' (U+1FA7, E1 BE A7) → "ὧι" (U+1F67 U+03B9, E1 BD A7 CE B9)
        case 0x1FA7:
            folded[0] = 0x1F67;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾨ' (U+1FA8, E1 BE A8) → "ὠι" (U+1F60 U+03B9, E1 BD A0 CE B9)
        case 0x1FA8:
            folded[0] = 0x1F60;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾩ' (U+1FA9, E1 BE A9) → "ὡι" (U+1F61 U+03B9, E1 BD A1 CE B9)
        case 0x1FA9:
            folded[0] = 0x1F61;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾪ' (U+1FAA, E1 BE AA) → "ὢι" (U+1F62 U+03B9, E1 BD A2 CE B9)
        case 0x1FAA:
            folded[0] = 0x1F62;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾫ' (U+1FAB, E1 BE AB) → "ὣι" (U+1F63 U+03B9, E1 BD A3 CE B9)
        case 0x1FAB:
            folded[0] = 0x1F63;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾬ' (U+1FAC, E1 BE AC) → "ὤι" (U+1F64 U+03B9, E1 BD A4 CE B9)
        case 0x1FAC:
            folded[0] = 0x1F64;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾭ' (U+1FAD, E1 BE AD) → "ὥι" (U+1F65 U+03B9, E1 BD A5 CE B9)
        case 0x1FAD:
            folded[0] = 0x1F65;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾮ' (U+1FAE, E1 BE AE) → "ὦι" (U+1F66 U+03B9, E1 BD A6 CE B9)
        case 0x1FAE:
            folded[0] = 0x1F66;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾯ' (U+1FAF, E1 BE AF) → "ὧι" (U+1F67 U+03B9, E1 BD A7 CE B9)
        case 0x1FAF:
            folded[0] = 0x1F67;
            folded[1] = 0x03B9;
            return 2;
        // Greek Extended: vowel + iota subscript (0x1FB2-0x1FFC)
        // 'ᾲ' (U+1FB2, E1 BE B2) → "ὰι" (U+1F70 U+03B9, E1 BD B0 CE B9)
        case 0x1FB2:
            folded[0] = 0x1F70;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾳ' (U+1FB3, E1 BE B3) → "αι" (U+03B1 U+03B9, CE B1 CE B9)
        case 0x1FB3:
            folded[0] = 0x03B1;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾴ' (U+1FB4, E1 BE B4) → "άι" (U+03AC U+03B9, CE AC CE B9)
        case 0x1FB4:
            folded[0] = 0x03AC;
            folded[1] = 0x03B9;
            return 2;
        // 'ᾶ' (U+1FB6, E1 BE B6) → "ᾶ" (U+03B1 U+0342, CE B1 CD 82)
        case 0x1FB6:
            folded[0] = 0x03B1;
            folded[1] = 0x0342;
            return 2;
        // 'ᾷ' (U+1FB7, E1 BE B7) → "ᾶι" (U+03B1 U+0342 U+03B9, CE B1 CD 82 CE B9)
        case 0x1FB7:
            folded[0] = 0x03B1;
            folded[1] = 0x0342;
            folded[2] = 0x03B9;
            return 3;
        // 'ᾼ' (U+1FBC, E1 BE BC) → "αι" (U+03B1 U+03B9, CE B1 CE B9)
        case 0x1FBC:
            folded[0] = 0x03B1;
            folded[1] = 0x03B9;
            return 2;
        // 'ῂ' (U+1FC2, E1 BF 82) → "ὴι" (U+1F74 U+03B9, E1 BD B4 CE B9)
        case 0x1FC2:
            folded[0] = 0x1F74;
            folded[1] = 0x03B9;
            return 2;
        // 'ῃ' (U+1FC3, E1 BF 83) → "ηι" (U+03B7 U+03B9, CE B7 CE B9)
        case 0x1FC3:
            folded[0] = 0x03B7;
            folded[1] = 0x03B9;
            return 2;
        // 'ῄ' (U+1FC4, E1 BF 84) → "ήι" (U+03AE U+03B9, CE AE CE B9)
        case 0x1FC4:
            folded[0] = 0x03AE;
            folded[1] = 0x03B9;
            return 2;
        // 'ῆ' (U+1FC6, E1 BF 86) → "ῆ" (U+03B7 U+0342, CE B7 CD 82)
        case 0x1FC6:
            folded[0] = 0x03B7;
            folded[1] = 0x0342;
            return 2;
        // 'ῇ' (U+1FC7, E1 BF 87) → "ῆι" (U+03B7 U+0342 U+03B9, CE B7 CD 82 CE B9)
        case 0x1FC7:
            folded[0] = 0x03B7;
            folded[1] = 0x0342;
            folded[2] = 0x03B9;
            return 3;
        // 'ῌ' (U+1FCC, E1 BF 8C) → "ηι" (U+03B7 U+03B9, CE B7 CE B9)
        case 0x1FCC:
            folded[0] = 0x03B7;
            folded[1] = 0x03B9;
            return 2;
        // 'ῒ' (U+1FD2, E1 BF 92) → "ῒ" (U+03B9 U+0308 U+0300, CE B9 CC 88 CC 80)
        case 0x1FD2:
            folded[0] = 0x03B9;
            folded[1] = 0x0308;
            folded[2] = 0x0300;
            return 3;
        // 'ΐ' (U+1FD3, E1 BF 93) → "ΐ" (U+03B9 U+0308 U+0301, CE B9 CC 88 CC 81)
        case 0x1FD3:
            folded[0] = 0x03B9;
            folded[1] = 0x0308;
            folded[2] = 0x0301;
            return 3;
        // 'ῖ' (U+1FD6, E1 BF 96) → "ῖ" (U+03B9 U+0342, CE B9 CD 82)
        case 0x1FD6:
            folded[0] = 0x03B9;
            folded[1] = 0x0342;
            return 2;
        // 'ῗ' (U+1FD7, E1 BF 97) → "ῗ" (U+03B9 U+0308 U+0342, CE B9 CC 88 CD 82)
        case 0x1FD7:
            folded[0] = 0x03B9;
            folded[1] = 0x0308;
            folded[2] = 0x0342;
            return 3;
        // 'ῢ' (U+1FE2, E1 BF A2) → "ῢ" (U+03C5 U+0308 U+0300, CF 85 CC 88 CC 80)
        case 0x1FE2:
            folded[0] = 0x03C5;
            folded[1] = 0x0308;
            folded[2] = 0x0300;
            return 3;
        // 'ΰ' (U+1FE3, E1 BF A3) → "ΰ" (U+03C5 U+0308 U+0301, CF 85 CC 88 CC 81)
        case 0x1FE3:
            folded[0] = 0x03C5;
            folded[1] = 0x0308;
            folded[2] = 0x0301;
            return 3;
        // 'ῤ' (U+1FE4, E1 BF A4) → "ῤ" (U+03C1 U+0313, CF 81 CC 93)
        case 0x1FE4:
            folded[0] = 0x03C1;
            folded[1] = 0x0313;
            return 2;
        // 'ῦ' (U+1FE6, E1 BF A6) → "ῦ" (U+03C5 U+0342, CF 85 CD 82)
        case 0x1FE6:
            folded[0] = 0x03C5;
            folded[1] = 0x0342;
            return 2;
        // 'ῧ' (U+1FE7, E1 BF A7) → "ῧ" (U+03C5 U+0308 U+0342, CF 85 CC 88 CD 82)
        case 0x1FE7:
            folded[0] = 0x03C5;
            folded[1] = 0x0308;
            folded[2] = 0x0342;
            return 3;
        // 'ῲ' (U+1FF2, E1 BF B2) → "ὼι" (U+1F7C U+03B9, E1 BD BC CE B9)
        case 0x1FF2:
            folded[0] = 0x1F7C;
            folded[1] = 0x03B9;
            return 2;
        // 'ῳ' (U+1FF3, E1 BF B3) → "ωι" (U+03C9 U+03B9, CF 89 CE B9)
        case 0x1FF3:
            folded[0] = 0x03C9;
            folded[1] = 0x03B9;
            return 2;
        // 'ῴ' (U+1FF4, E1 BF B4) → "ώι" (U+03CE U+03B9, CF 8E CE B9)
        case 0x1FF4:
            folded[0] = 0x03CE;
            folded[1] = 0x03B9;
            return 2;
        // 'ῶ' (U+1FF6, E1 BF B6) → "ῶ" (U+03C9 U+0342, CF 89 CD 82)
        case 0x1FF6:
            folded[0] = 0x03C9;
            folded[1] = 0x0342;
            return 2;
        // 'ῷ' (U+1FF7, E1 BF B7) → "ῶι" (U+03C9 U+0342 U+03B9, CF 89 CD 82 CE B9)
        case 0x1FF7:
            folded[0] = 0x03C9;
            folded[1] = 0x0342;
            folded[2] = 0x03B9;
            return 3;
        // 'ῼ' (U+1FFC, E1 BF BC) → "ωι" (U+03C9 U+03B9, CF 89 CE B9)
        case 0x1FFC:
            folded[0] = 0x03C9;
            folded[1] = 0x03B9;
            return 2;
        // Alphabetic Presentation Forms: ligatures
        // 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
        case 0xFB00:
            folded[0] = 0x0066;
            folded[1] = 0x0066;
            return 2;
        // 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
        case 0xFB01:
            folded[0] = 0x0066;
            folded[1] = 0x0069;
            return 2;
        // 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
        case 0xFB02:
            folded[0] = 0x0066;
            folded[1] = 0x006C;
            return 2;
        // 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
        case 0xFB03:
            folded[0] = 0x0066;
            folded[1] = 0x0066;
            folded[2] = 0x0069;
            return 3;
        // 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
        case 0xFB04:
            folded[0] = 0x0066;
            folded[1] = 0x0066;
            folded[2] = 0x006C;
            return 3;
        // 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
        case 0xFB05:
            folded[0] = 0x0073;
            folded[1] = 0x0074;
            return 2;
        // 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
        case 0xFB06:
            folded[0] = 0x0073;
            folded[1] = 0x0074;
            return 2;
        // Armenian ligatures
        // 'ﬓ' (U+FB13, EF AC 93) → "մն" (U+0574 U+0576, D5 B4 D5 B6)
        case 0xFB13:
            folded[0] = 0x0574;
            folded[1] = 0x0576;
            return 2;
        // 'ﬔ' (U+FB14, EF AC 94) → "մե" (U+0574 U+0565, D5 B4 D5 A5)
        case 0xFB14:
            folded[0] = 0x0574;
            folded[1] = 0x0565;
            return 2;
        // 'ﬕ' (U+FB15, EF AC 95) → "մի" (U+0574 U+056B, D5 B4 D5 AB)
        case 0xFB15:
            folded[0] = 0x0574;
            folded[1] = 0x056B;
            return 2;
        // 'ﬖ' (U+FB16, EF AC 96) → "վն" (U+057E U+0576, D5 BE D5 B6)
        case 0xFB16:
            folded[0] = 0x057E;
            folded[1] = 0x0576;
            return 2;
        // 'ﬗ' (U+FB17, EF AC 97) → "մխ" (U+0574 U+056D, D5 B4 D5 AD)
        case 0xFB17:
            folded[0] = 0x0574;
            folded[1] = 0x056D;
            return 2;
        }

        folded[0] = rune;
        return 1; // 3-byte: no folding needed
    }

    // 4-byte UTF-8 (U+10000-10FFFF): Deseret, Osage, Vithkuqi, etc.

    // Deseret 𐐀-𐐧: 0x10400-0x10427 (+40)
    if ((sz_u32_t)(rune - 0x10400) <= 0x27) {
        sz_assert_(sz_is_in_range_(rune, 0x10400, 0x10427));
        folded[0] = rune + 0x28;
        return 1;
    }

    // Osage 𐒰-𐓓: 0x104B0-0x104D3 (+40)
    if ((sz_u32_t)(rune - 0x104B0) <= 0x23) {
        sz_assert_(sz_is_in_range_(rune, 0x104B0, 0x104D3));
        folded[0] = rune + 0x28;
        return 1;
    }

    // Vithkuqi: 3 ranges with gaps, all +39
    if ((sz_u32_t)(rune - 0x10570) <= 0x0A) { // 0x10570-0x1057A
        sz_assert_(sz_is_in_range_(rune, 0x10570, 0x1057A));
        folded[0] = rune + 0x27;
        return 1;
    }
    if ((sz_u32_t)(rune - 0x1057C) <= 0x0E) { // 0x1057C-0x1058A
        sz_assert_(sz_is_in_range_(rune, 0x1057C, 0x1058A));
        folded[0] = rune + 0x27;
        return 1;
    }
    if ((sz_u32_t)(rune - 0x1058C) <= 0x06) { // 0x1058C-0x10592
        sz_assert_(sz_is_in_range_(rune, 0x1058C, 0x10592));
        folded[0] = rune + 0x27;
        return 1;
    }

    // Old Hungarian: 0x10C80-0x10CB2 (+64)
    if ((sz_u32_t)(rune - 0x10C80) <= 0x32) {
        sz_assert_(sz_is_in_range_(rune, 0x10C80, 0x10CB2));
        folded[0] = rune + 0x40;
        return 1;
    }

    // Garay: 0x10D50-0x10D65 (+32)
    if ((sz_u32_t)(rune - 0x10D50) <= 0x15) {
        sz_assert_(sz_is_in_range_(rune, 0x10D50, 0x10D65));
        folded[0] = rune + 0x20;
        return 1;
    }

    // Warang Citi: 0x118A0-0x118BF (+32)
    if ((sz_u32_t)(rune - 0x118A0) <= 0x1F) {
        sz_assert_(sz_is_in_range_(rune, 0x118A0, 0x118BF));
        folded[0] = rune + 0x20;
        return 1;
    }

    // Medefaidrin: 0x16E40-0x16E5F (+32)
    if ((sz_u32_t)(rune - 0x16E40) <= 0x1F) {
        sz_assert_(sz_is_in_range_(rune, 0x16E40, 0x16E5F));
        folded[0] = rune + 0x20;
        return 1;
    }

    // Beria Erfe: 0x16EA0-0x16EB8 (+27)
    if ((sz_u32_t)(rune - 0x16EA0) <= 0x18) {
        sz_assert_(sz_is_in_range_(rune, 0x16EA0, 0x16EB8));
        folded[0] = rune + 0x1B;
        return 1;
    }

    // Adlam: 0x1E900-0x1E921 (+34)
    if ((sz_u32_t)(rune - 0x1E900) <= 0x21) {
        sz_assert_(sz_is_in_range_(rune, 0x1E900, 0x1E921));
        folded[0] = rune + 0x22;
        return 1;
    }

    // Next let's handle the 4-byte irregular mappings
    switch (rune) {
    // Vithkuqi: Albanian historical script
    case 0x10594: folded[0] = 0x105BB; return 1; // '𐖔' (U+010594, F0 90 96 94) → '𐖻' (U+0105BB, F0 90 96 BB)
    case 0x10595: folded[0] = 0x105BC; return 1; // '𐖕' (U+010595, F0 90 96 95) → '𐖼' (U+0105BC, F0 90 96 BC)
    }

    folded[0] = rune;
    return 1; // No folding needed
}

/**
 *  @brief Helper function performing case-folding under the constraint, that no output may be incomplete.
 *
 *  @param source Pointer to the source UTF-8 data, must be valid UTF-8.
 *  @param source_length Length of the source data in bytes.
 *  @param destination Pointer to the destination buffer.
 *  @param destination_length Length of the destination buffer in bytes.
 *  @param codepoints_consumed Number of codepoints read from source.
 *  @param codepoints_exported Number of codepoints written to destination.
 *  @param bytes_consumed Number of bytes read from source.
 *  @param bytes_exported Number of bytes written to destination.
 */
SZ_HELPER_AUTO void sz_utf8_uncased_fold_upto_(                     //
    sz_cptr_t source, sz_size_t source_length,                      //
    sz_ptr_t destination, sz_size_t destination_length,             //
    sz_size_t *codepoints_consumed, sz_size_t *codepoints_exported, //
    sz_size_t *bytes_consumed, sz_size_t *bytes_exported) {

    sz_u8_t const *const source_start = (sz_u8_t const *)source;
    sz_u8_t const *const source_limit = source_start + source_length;
    sz_u8_t *const destination_start = (sz_u8_t *)destination;
    sz_u8_t *const destination_limit = destination_start + destination_length;

    sz_u8_t const *source_ptr = source_start;
    sz_u8_t *destination_ptr = destination_start;
    sz_size_t codepoints_read = 0;
    sz_size_t codepoints_written = 0;

    while (source_ptr < source_limit && destination_ptr < destination_limit) {

        // Fast path for ASCII optimization
        while (source_ptr < source_limit && destination_ptr < destination_limit && *source_ptr < 0x80) {
            *destination_ptr++ = sz_ascii_fold_(*source_ptr++);
            codepoints_read++;
            codepoints_written++;
        }

        // Check if we hit boundaries
        if (source_ptr >= source_limit || destination_ptr >= destination_limit) break;

        // Fold only well-formed runes; copy malformed bytes through unchanged, resyncing one byte at a time.
        sz_rune_t source_rune;
        sz_rune_length_t const source_rune_length = sz_rune_decode((sz_cptr_t)source_ptr, (sz_cptr_t)source_limit,
                                                                   &source_rune);
        if (source_rune_length == sz_rune_invalid_k) {
            *destination_ptr++ = *source_ptr++; // destination has room (checked at the loop top)
            codepoints_read++, codepoints_written++;
            continue;
        }

        // Perform Unicode folding
        sz_rune_t target_runes[3];
        sz_size_t target_runes_count = sz_unicode_fold_codepoint_(source_rune, target_runes);

        // In the worst case scenario, when folded, the text becomes 3x longer.
        // That's the story of 'ΐ' (U+0390, CE 90) becoming three codepoints (U+03B9 U+0308 U+0301, CE B9 CC 88 CC 81).
        sz_u8_t target_bytes[12]; // 3 runes max, each up to 4 bytes
        sz_size_t target_bytes_count = 0;
        for (sz_size_t rune_index = 0; rune_index < target_runes_count; ++rune_index)
            target_bytes_count += sz_rune_encode(target_runes[rune_index], target_bytes + target_bytes_count);

        if (destination_ptr + target_bytes_count > destination_limit) break;
        for (sz_size_t byte_index = 0; byte_index < target_bytes_count; ++byte_index)
            *destination_ptr++ = target_bytes[byte_index];

        source_ptr += source_rune_length;
        codepoints_read++;
        codepoints_written += target_runes_count;
    }

    if (codepoints_consumed) *codepoints_consumed = codepoints_read;
    if (codepoints_exported) *codepoints_exported = codepoints_written;
    if (bytes_consumed) *bytes_consumed = (sz_size_t)(source_ptr - source_start);
    if (bytes_exported) *bytes_exported = (sz_size_t)(destination_ptr - destination_start);
}

SZ_API_COMPTIME sz_size_t sz_utf8_uncased_fold_serial(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {

    sz_u8_t const *source_ptr = (sz_u8_t const *)source;
    sz_u8_t const *source_end = source_ptr + source_length;
    sz_u8_t *destination_ptr = (sz_u8_t *)destination;

    // Well-formed runes fold; any byte not starting a well-formed codepoint is copied through (sz_rune_decode).
    while (source_ptr < source_end) {
        // ASCII fast-path: process consecutive ASCII bytes without UTF-8 decode/encode overhead.
        // This handles ~95% of typical text content with minimal branching.
        while (source_ptr < source_end && *source_ptr < 0x80) *destination_ptr++ = sz_ascii_fold_(*source_ptr++);
        if (source_ptr >= source_end) break;

        // Fold only well-formed runes; copy malformed bytes through unchanged, resyncing one byte at a time.
        sz_rune_t rune;
        sz_rune_length_t const rune_length = sz_rune_decode((sz_cptr_t)source_ptr, (sz_cptr_t)source_end, &rune);
        if (rune_length == sz_rune_invalid_k) {
            *destination_ptr++ = *source_ptr++;
            continue;
        }
        source_ptr += rune_length;

        sz_rune_t folded_runes[3]; // Unicode case folding produces at most 3 runes
        sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
        for (sz_size_t rune_index = 0; rune_index != folded_count; ++rune_index)
            destination_ptr += sz_rune_encode(folded_runes[rune_index], destination_ptr);
    }

    return (sz_size_t)(destination_ptr - (sz_u8_t *)destination);
}

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_UNCASED_FOLD_SERIAL_H_
