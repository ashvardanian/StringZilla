/**
 *  @brief Serial backend for UTF-8 traversal.
 *  @file include/stringzilla/utf8_iterate/serial.h
 *  @author Ash Vardanian
 *  @sa utf8.h
 */
#ifndef STRINGZILLA_UTF8_ITERATE_SERIAL_H_
#define STRINGZILLA_UTF8_ITERATE_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_iterate/tables.h" // generated UAX-29 Word_Break property tables

#ifdef __cplusplus
extern "C" {
#endif

SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_serial(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    // TODO: Optimize with a SWAR variant
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u8_t const *end_u8 = text_u8 + length;
    while (text_u8 != end_u8) {
        sz_u8_t const c = *text_u8;
        switch (c) {
        case '\n':
        case '\v':
        case '\f': *matched_length = 1; return (sz_cptr_t)text_u8;
        // Differentiate between "\r" and "\r\n"
        case '\r':
            if (text_u8 + 1 != end_u8 && text_u8[1] == '\n') {
                *matched_length = 2;
                return (sz_cptr_t)text_u8;
            }
            else {
                *matched_length = 1;
                return (sz_cptr_t)text_u8;
            }
        // Matching the 0xC285 character
        case 0xC2:
            if (text_u8 + 1 != end_u8 && text_u8[1] == 0x85) {
                *matched_length = 2;
                return (sz_cptr_t)text_u8;
            }
            else {
                ++text_u8;
                continue;
            }
        // Matching 3-byte newline characters
        case 0xE2:
            if (text_u8 + 2 < end_u8 && text_u8[1] == 0x80 && text_u8[2] == 0xA8) {
                *matched_length = 3;
                return (sz_cptr_t)text_u8;
            }
            else if (text_u8 + 2 < end_u8 && text_u8[1] == 0x80 && text_u8[2] == 0xA9) {
                *matched_length = 3;
                return (sz_cptr_t)text_u8;
            }
            else {
                ++text_u8;
                continue;
            }
        default: ++text_u8; continue;
        }
    }
    *matched_length = 0;
    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_serial(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    // TODO: Optimize with a SWAR variant
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u8_t const *end_u8 = text_u8 + length;
    while (text_u8 != end_u8) {
        sz_u8_t const c = *text_u8;
        switch (c) {
        case ' ':
        case '\t':
        case '\n':
        case '\v':
        case '\f':
        case '\r': *matched_length = 1; return (sz_cptr_t)text_u8;
        // Matching 2-byte whitespace characters
        case 0xC2:
            if (text_u8 + 1 != end_u8) {
                // U+0085: NEXT LINE (NEL)
                if (text_u8[1] == 0x85) {
                    *matched_length = 2;
                    return (sz_cptr_t)text_u8;
                }
                // U+00A0: NO-BREAK SPACE
                else if (text_u8[1] == 0xA0) {
                    *matched_length = 2;
                    return (sz_cptr_t)text_u8;
                }
            }
            ++text_u8;
            continue;
        // Matching the 0xE19A80 ogham space mark
        case 0xE1:
            if (text_u8 + 2 < end_u8 && text_u8[1] == 0x9A && text_u8[2] == 0x80) {
                *matched_length = 3;
                return (sz_cptr_t)text_u8;
            }
            else {
                ++text_u8;
                continue;
            }
        // Match the 3-byte whitespace characters starting with 0xE2
        case 0xE2:
            if (text_u8 + 2 < end_u8) {
                // U+2000 to U+200D: 0xE28080 to 0xE2808D
                if (text_u8[1] == 0x80 && text_u8[2] >= 0x80 && text_u8[2] <= 0x8D) {
                    *matched_length = 3;
                    return (sz_cptr_t)text_u8;
                }
                // U+2028: LINE SEPARATOR (0xE280A8)
                else if (text_u8[1] == 0x80 && text_u8[2] == 0xA8) {
                    *matched_length = 3;
                    return (sz_cptr_t)text_u8;
                }
                // U+2029: PARAGRAPH SEPARATOR (0xE280A9)
                else if (text_u8[1] == 0x80 && text_u8[2] == 0xA9) {
                    *matched_length = 3;
                    return (sz_cptr_t)text_u8;
                }
                // U+202F: NARROW NO-BREAK SPACE (0xE280AF)
                else if (text_u8[1] == 0x80 && text_u8[2] == 0xAF) {
                    *matched_length = 3;
                    return (sz_cptr_t)text_u8;
                }
                // U+205F: MEDIUM MATHEMATICAL SPACE (0xE2819F)
                else if (text_u8[1] == 0x81 && text_u8[2] == 0x9F) {
                    *matched_length = 3;
                    return (sz_cptr_t)text_u8;
                }
            }
            ++text_u8;
            continue;
        // Match the 3-byte ideographic space
        case 0xE3:
            if (text_u8 + 2 < end_u8 && text_u8[1] == 0x80 && text_u8[2] == 0x80) {
                *matched_length = 3;
                return (sz_cptr_t)text_u8;
            }
            else {
                ++text_u8;
                continue;
            }
        }
        ++text_u8;
    }
    *matched_length = 0;
    return SZ_NULL_CHAR;
}

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
        sz_rune_length_t rune_length;
        sz_rune_parse_unchecked(text_cursor, &rune, &rune_length);
        if (text_cursor + rune_length > text_end) break; // Incomplete sequence at buffer boundary
        runes[runes_written++] = rune;
        text_cursor += rune_length;
    }

    *runes_unpacked = runes_written;
    return text_cursor;
}

#pragma region UAX-29 Word Boundaries

/**
 *  @brief Get the Unicode TR29 Word_Break property for a codepoint.
 *
 *  Returns one of the 16 Word_Break property values (0-15):
 *  - sz_tr29_word_break_other_k (0): Default - creates word boundary
 *  - sz_tr29_word_break_cr_k (1): Carriage Return
 *  - sz_tr29_word_break_lf_k (2): Line Feed
 *  - sz_tr29_word_break_newline_k (3): Other newlines
 *  - sz_tr29_word_break_extend_k (4): Combining marks
 *  - sz_tr29_word_break_zwj_k (5): Zero Width Joiner
 *  - sz_tr29_word_break_format_k (6): Format characters
 *  - sz_tr29_word_break_regional_ind_k (7): Regional Indicators
 *  - sz_tr29_word_break_aletter_k (8): Alphabetic letters
 *  - sz_tr29_word_break_hebrew_letter_k (9): Hebrew letters
 *  - sz_tr29_word_break_numeric_k (10): Digits
 *  - sz_tr29_word_break_katakana_k (11): Japanese Katakana
 *  - sz_tr29_word_break_extendnumlet_k (12): Underscore/connector
 *  - sz_tr29_word_break_midletter_k (13): Mid-letter punctuation
 *  - sz_tr29_word_break_midnum_k (14): Mid-number punctuation
 *  - sz_tr29_word_break_mid_quotes_k (15): MidNumLet + quotes
 */
SZ_PUBLIC sz_u8_t sz_rune_word_break_property(sz_rune_t rune) {
    // Fast path 1: ASCII - direct table lookup
    if (rune < 0x80) { return sz_utf8_word_break_property_ascii_[rune]; }

    // Fast path 2: Latin Extended-A (U+0100-U+017F) - all ALetter
    if (rune >= 0x0100 && rune <= 0x017F) return sz_tr29_word_break_aletter_k;

    // Fast path 3: Latin Extended-B (U+0180-U+024F) - all ALetter
    if (rune >= 0x0180 && rune <= 0x024F) return sz_tr29_word_break_aletter_k;

    // Fast path 4: Hangul Syllables (U+AC00-U+D7A3) - all ALetter
    if (rune >= 0xAC00 && rune <= 0xD7A3) return sz_tr29_word_break_aletter_k;

    // Fast path 5: CJK Extension A (U+3400-U+4DBF) - all Other/boundary
    if (rune >= 0x3400 && rune <= 0x4DBF) return sz_tr29_word_break_other_k;

    // Fast path 6: CJK Unified (U+4E00-U+9FFF) - all Other/boundary
    if (rune >= 0x4E00 && rune <= 0x9FFF) return sz_tr29_word_break_other_k;

    // BMP two-stage lookup with nibble-packed stage2
    if (rune < 0x10000) {
        sz_u8_t block_index = sz_utf8_word_break_property_stage1_[rune >> 8];

        // Special indices
        if (block_index == 0) return sz_tr29_word_break_other_k;   // All boundary
        if (block_index == 1) return sz_tr29_word_break_aletter_k; // All ALetter
        if (block_index == 255) return sz_tr29_word_break_other_k; // Fast path handled above

        // Nibble-packed lookup: each byte holds 2 properties
        sz_size_t stage2_offset = ((sz_size_t)(block_index - 0x80)) * 128;
        sz_u8_t low_byte = (sz_u8_t)(rune & 0xFF);
        sz_u8_t packed = sz_utf8_word_break_property_stage2_[stage2_offset + (low_byte >> 1)];
        return (low_byte & 1) ? (packed >> 4) : (packed & 0x0F);
    }

    // SMP: Binary search over transition points
    // Each entry is packed as (codepoint << 4) | property
    // Property changes at each transition point
    sz_size_t search_low = 0, search_high = SZ_TR29_WB_SMP_TRANSITION_COUNT_;
    while (search_low < search_high) {
        sz_size_t mid = (search_low + search_high) / 2;
        sz_rune_t trans_cp = sz_utf8_word_break_property_smp_transitions_[mid] >> 4;
        if (rune < trans_cp) { search_high = mid; }
        else { search_low = mid + 1; }
    }
    // search_low is now the index of first transition > rune, so search_low-1 is our transition
    if (search_low == 0) return sz_tr29_word_break_other_k;
    return (sz_u8_t)(sz_utf8_word_break_property_smp_transitions_[search_low - 1] & 0x0F);
}

/**
 *  @brief Check if a codepoint is a "word character" (has word-forming property).
 *
 *  This is a convenience function that returns true if the codepoint has a
 *  Word_Break property that typically forms words (ALetter, Hebrew_Letter,
 *  Numeric, Katakana, ExtendNumLet, or mid-word punctuation).
 *
 *  @param[in] rune The Unicode codepoint to check.
 *  @return sz_true_k if the codepoint is a word character, sz_false_k otherwise.
 */
SZ_PUBLIC sz_bool_t sz_rune_is_word_char(sz_rune_t rune) {
    sz_u8_t property = sz_rune_word_break_property(rune);
    // Word characters: ALetter(8), Hebrew_Letter(9), Numeric(10), Katakana(11),
    // ExtendNumLet(12), MidLetter(13), MidNum(14), MidNumLet/Quotes(15)
    return (sz_bool_t)(property >= sz_tr29_word_break_aletter_k);
}

/**
 *  @brief Decode a UTF-8 codepoint starting at the given position.
 *  @return The decoded codepoint, or 0xFFFD on error. Updates position to next codepoint.
 */
SZ_INTERNAL sz_rune_t sz_utf8_decode_(sz_cptr_t text, sz_size_t length, sz_size_t *position) {
    if (*position >= length) return 0;
    sz_u8_t lead_byte = (sz_u8_t)text[*position];
    if (lead_byte < 0x80) {
        (*position)++;
        return lead_byte;
    }
    if ((lead_byte & 0xE0) == 0xC0 && *position + 1 < length) {
        sz_rune_t rune = ((lead_byte & 0x1F) << 6) | (text[*position + 1] & 0x3F);
        *position += 2;
        return rune;
    }
    if ((lead_byte & 0xF0) == 0xE0 && *position + 2 < length) {
        sz_rune_t rune = ((lead_byte & 0x0F) << 12) | ((text[*position + 1] & 0x3F) << 6) |
                         (text[*position + 2] & 0x3F);
        *position += 3;
        return rune;
    }
    if ((lead_byte & 0xF8) == 0xF0 && *position + 3 < length) {
        sz_rune_t rune = ((lead_byte & 0x07) << 18) | ((text[*position + 1] & 0x3F) << 12) |
                         ((text[*position + 2] & 0x3F) << 6) | (text[*position + 3] & 0x3F);
        *position += 4;
        return rune;
    }
    (*position)++;
    return 0xFFFD;
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

/*  Each Word_Break property fits in 4 bits, so a 16-bit constant is a membership set: bit `p` marks property
 *  `p` as a member. Testing membership is then a shift-and-mask, replacing chains of equality comparisons in
 *  the hot WB4 skip loops. */
enum {
    sz_utf8_word_break_ignorable_set_k = (1u << sz_tr29_word_break_extend_k) | (1u << sz_tr29_word_break_zwj_k) |
                                         (1u << sz_tr29_word_break_format_k),
    sz_utf8_word_break_aletter_or_hebrew_set_k = (1u << sz_tr29_word_break_aletter_k) |
                                                 (1u << sz_tr29_word_break_hebrew_letter_k),
    sz_utf8_word_break_mid_quotes_set_k = (1u << sz_tr29_word_break_mid_quotes_k),
};

/** @brief Check if a property is WB4-ignorable (Extend, Format, ZWJ). */
SZ_INTERNAL sz_bool_t sz_utf8_word_break_is_ignorable_(sz_u8_t property) {
    return (sz_bool_t)((sz_utf8_word_break_ignorable_set_k >> property) & 1u);
}

/** @brief Check if a property is AHLetter (ALetter or Hebrew_Letter). */
SZ_INTERNAL sz_bool_t sz_utf8_word_break_is_aletter_or_hebrew_(sz_u8_t property) {
    return (sz_bool_t)((sz_utf8_word_break_aletter_or_hebrew_set_k >> property) & 1u);
}

/**
 *  @brief Check if a property is MidNumLetQ (MidNumLet or Single_Quote).
 *         In our encoding, MID_QUOTES (15) covers MidNumLet + quotes.
 */
SZ_INTERNAL sz_bool_t sz_utf8_word_break_is_mid_quotes_(sz_u8_t property) {
    return (sz_bool_t)((sz_utf8_word_break_mid_quotes_set_k >> property) & 1u);
}

/**
 *  @brief Portable UAX-29 "join" (guaranteed non-boundary) mask for an all-ASCII window.
 *
 *  Given per-lane class-membership masks (bit `i` set => lane `i` has that class), returns a mask whose bit `i`
 *  is set when the boundary before lane `i` is suppressed by a no-break rule. ASCII contains no
 *  Extend/Format/ZWJ/Regional_Indicator/Hebrew/Katakana, so WB4 and WB15/16 never apply and the look-around
 *  rules reduce to neighbour bit-shifts: `<< 1` reads the previous lane, `>> 1` the next, `<< 2` two back. The
 *  result is exact for lanes whose i-2 and i+1 neighbours are in-window. Every fixed-width SIMD backend reduces
 *  its classified window to these masks and calls this one shared routine, so the rule logic lives in a single
 *  portable, testable place rather than being re-derived per ISA.
 */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_join_from_class_masks_(                                        //
    sz_u64_t aletter_mask, sz_u64_t numeric_mask, sz_u64_t extendnumlet_mask, sz_u64_t midletter_mask, //
    sz_u64_t midnum_mask, sz_u64_t mid_quotes_mask, sz_u64_t carriage_return_mask, sz_u64_t line_feed_mask) {
    sz_u64_t mid_letter_or_quotes_mask = midletter_mask | mid_quotes_mask;
    sz_u64_t mid_num_or_quotes_mask = midnum_mask | mid_quotes_mask;
    sz_u64_t join = (carriage_return_mask << 1) & line_feed_mask;                         // WB3  CR x LF
    join |= (aletter_mask << 1) & aletter_mask;                                           // WB5
    join |= (aletter_mask << 1) & mid_letter_or_quotes_mask & (aletter_mask >> 1);        // WB6
    join |= (aletter_mask << 2) & (mid_letter_or_quotes_mask << 1) & aletter_mask;        // WB7
    join |= (numeric_mask << 1) & numeric_mask;                                           // WB8
    join |= (aletter_mask << 1) & numeric_mask;                                           // WB9
    join |= (numeric_mask << 1) & aletter_mask;                                           // WB10
    join |= (numeric_mask << 1) & mid_num_or_quotes_mask & (numeric_mask >> 1);           // WB11
    join |= (numeric_mask << 2) & (mid_num_or_quotes_mask << 1) & numeric_mask;           // WB12
    join |= ((aletter_mask | numeric_mask | extendnumlet_mask) << 1) & extendnumlet_mask; // WB13a
    join |= (extendnumlet_mask << 1) & (aletter_mask | numeric_mask);                     // WB13b
    return join;
}

/**
 *  @brief Skip forward past WB4-ignorable characters (Extend, Format, ZWJ).
 *  @return Position after ignorables, or original position if none.
 */
SZ_INTERNAL sz_size_t sz_utf8_skip_ignorables_forward_(sz_cptr_t text, sz_size_t length, sz_size_t position) {
    while (position < length) {
        sz_size_t next_position = position;
        sz_rune_t rune = sz_utf8_decode_(text, length, &next_position);
        sz_u8_t property = sz_rune_word_break_property(rune);
        if (!sz_utf8_word_break_is_ignorable_(property)) break;
        position = next_position;
    }
    return position;
}

/**
 *  @brief Get the effective property at position, skipping WB4 ignorables.
 *         Returns the property and updates next_pos to position after ignorables.
 */
SZ_INTERNAL sz_u8_t sz_utf8_word_break_effective_property_(sz_cptr_t text, sz_size_t length, sz_size_t position,
                                                           sz_size_t *next_position) {
    sz_size_t current = position;
    sz_rune_t rune = sz_utf8_decode_(text, length, &current);
    sz_u8_t property = sz_rune_word_break_property(rune);

    // Skip ignorables to find effective next
    while (current < length && sz_utf8_word_break_is_ignorable_(property)) {
        rune = sz_utf8_decode_(text, length, &current);
        property = sz_rune_word_break_property(rune);
    }

    if (next_position) *next_position = current;
    return property;
}

/**
 *  @brief Look back to find the previous non-ignorable property.
 *  @return The property of the previous significant character, or sz_tr29_word_break_other_k if none.
 */
SZ_INTERNAL sz_u8_t sz_utf8_word_break_previous_property_(sz_cptr_t text, sz_size_t position) {
    if (position == 0) return sz_tr29_word_break_other_k;

    // Scan backward to find start of previous codepoint
    sz_size_t previous = position - 1;
    while (previous > 0 && ((sz_u8_t)text[previous] & 0xC0) == 0x80) previous--;

    // Decode and get property
    sz_size_t decode_position = previous;
    sz_rune_t rune = sz_utf8_decode_(text, position, &decode_position);
    sz_u8_t property = sz_rune_word_break_property(rune);

    // Skip back over ignorables
    while (sz_utf8_word_break_is_ignorable_(property) && previous > 0) {
        previous--;
        while (previous > 0 && ((sz_u8_t)text[previous] & 0xC0) == 0x80) previous--;
        decode_position = previous;
        rune = sz_utf8_decode_(text, position, &decode_position);
        property = sz_rune_word_break_property(rune);
    }

    return property;
}

/**
 *  @brief Count Regional Indicators before position (for WB15/16).
 */
SZ_INTERNAL sz_size_t sz_utf8_word_break_count_regional_indicators_before_(sz_cptr_t text, sz_size_t position) {
    sz_size_t count = 0;
    sz_size_t current = position;

    while (current > 0) {
        // Find previous codepoint
        sz_size_t previous = current - 1;
        while (previous > 0 && ((sz_u8_t)text[previous] & 0xC0) == 0x80) previous--;

        sz_size_t decode_position = previous;
        sz_rune_t rune = sz_utf8_decode_(text, current, &decode_position);
        sz_u8_t property = sz_rune_word_break_property(rune);

        if (property == sz_tr29_word_break_regional_ind_k) {
            count++;
            current = previous;
        }
        else if (sz_utf8_word_break_is_ignorable_(property)) {
            current = previous; // Skip ignorables
        }
        else { break; }
    }
    return count;
}

/**
 *  @brief Check if position is a word boundary per Unicode TR29.
 *
 *  Implements the TR29 word boundary algorithm. Position 0 and position == length
 *  are always boundaries (WB1/WB2).
 */
SZ_PUBLIC sz_bool_t sz_utf8_is_word_boundary_serial(sz_cptr_t text, sz_size_t length, sz_size_t position) {
    // WB1: Break at start of text
    if (position == 0) return sz_true_k;
    // WB2: Break at end of text
    if (position >= length) return sz_true_k;

    // Never break at UTF-8 continuation bytes (0x80-0xBF)
    if (((sz_u8_t)text[position] & 0xC0) == 0x80) return sz_false_k;

    // Get properties of characters before and after the boundary
    sz_u8_t previous_property = sz_utf8_word_break_previous_property_(text, position);

    sz_size_t after_position = position;
    sz_rune_t after_rune = sz_utf8_decode_(text, length, &after_position);
    sz_u8_t after_prop = sz_rune_word_break_property(after_rune);

    // WB3: Do not break between CR and LF
    if (previous_property == sz_tr29_word_break_cr_k && after_prop == sz_tr29_word_break_lf_k) return sz_false_k;

    // WB3a: Break after Newline, CR, LF
    if (previous_property == sz_tr29_word_break_newline_k || previous_property == sz_tr29_word_break_cr_k ||
        previous_property == sz_tr29_word_break_lf_k)
        return sz_true_k;

    // WB3b: Break before Newline, CR, LF
    if (after_prop == sz_tr29_word_break_newline_k || after_prop == sz_tr29_word_break_cr_k ||
        after_prop == sz_tr29_word_break_lf_k)
        return sz_true_k;

    // WB3c: Do not break within emoji ZWJ sequences
    // (Simplified: don't break ZWJ × anything)
    if (previous_property == sz_tr29_word_break_zwj_k) return sz_false_k;

    // WB4: Ignore Format and Extend characters - get effective properties
    // Skip ignorables after position to get effective "after" property
    if (sz_utf8_word_break_is_ignorable_(after_prop)) {
        sz_size_t skip_position = position;
        after_prop = sz_utf8_word_break_effective_property_(text, length, skip_position, (sz_size_t *)0);
    }

    // WB5: Do not break between AHLetter
    if (sz_utf8_word_break_is_aletter_or_hebrew_(previous_property) &&
        sz_utf8_word_break_is_aletter_or_hebrew_(after_prop))
        return sz_false_k;

    // WB6: Do not break AHLetter × (MidLetter|MidNumLetQ) × AHLetter
    if (sz_utf8_word_break_is_aletter_or_hebrew_(previous_property) &&
        (after_prop == sz_tr29_word_break_midletter_k || sz_utf8_word_break_is_mid_quotes_(after_prop))) {
        // Look ahead to see if followed by AHLetter
        sz_size_t lookahead = after_position;
        lookahead = sz_utf8_skip_ignorables_forward_(text, length, lookahead);
        if (lookahead < length) {
            sz_size_t lookahead_position = lookahead;
            sz_rune_t la_rune = sz_utf8_decode_(text, length, &lookahead_position);
            sz_u8_t lookahead_property = sz_rune_word_break_property(la_rune);
            if (sz_utf8_word_break_is_aletter_or_hebrew_(lookahead_property)) return sz_false_k;
        }
    }

    // WB7: Do not break AHLetter (MidLetter|MidNumLetQ) × AHLetter
    if ((previous_property == sz_tr29_word_break_midletter_k || sz_utf8_word_break_is_mid_quotes_(previous_property)) &&
        sz_utf8_word_break_is_aletter_or_hebrew_(after_prop)) {
        // Look back to see if preceded by AHLetter
        // This requires looking at the character before the previous codepoint's position
        sz_size_t previous_cp_start = position - 1;
        while (previous_cp_start > 0 && ((sz_u8_t)text[previous_cp_start] & 0xC0) == 0x80) previous_cp_start--;
        if (previous_cp_start > 0) {
            sz_size_t pre_previous_codepoint_start = previous_cp_start - 1;
            while (pre_previous_codepoint_start > 0 && ((sz_u8_t)text[pre_previous_codepoint_start] & 0xC0) == 0x80)
                pre_previous_codepoint_start--;
            sz_size_t pre_previous_position = pre_previous_codepoint_start;
            sz_rune_t pp_rune = sz_utf8_decode_(text, previous_cp_start, &pre_previous_position);
            sz_u8_t pre_previous_property = sz_rune_word_break_property(pp_rune);
            if (sz_utf8_word_break_is_aletter_or_hebrew_(pre_previous_property)) return sz_false_k;
        }
    }

    // WB7a: Do not break Hebrew_Letter × Single_Quote
    if (previous_property == sz_tr29_word_break_hebrew_letter_k && after_prop == sz_tr29_word_break_mid_quotes_k)
        return sz_false_k;

    // WB8: Do not break Numeric × Numeric
    if (previous_property == sz_tr29_word_break_numeric_k && after_prop == sz_tr29_word_break_numeric_k)
        return sz_false_k;

    // WB9: Do not break AHLetter × Numeric
    if (sz_utf8_word_break_is_aletter_or_hebrew_(previous_property) && after_prop == sz_tr29_word_break_numeric_k)
        return sz_false_k;

    // WB10: Do not break Numeric × AHLetter
    if (previous_property == sz_tr29_word_break_numeric_k && sz_utf8_word_break_is_aletter_or_hebrew_(after_prop))
        return sz_false_k;

    // WB11: Do not break Numeric × (MidNum|MidNumLetQ) × Numeric
    if (previous_property == sz_tr29_word_break_numeric_k &&
        (after_prop == sz_tr29_word_break_midnum_k || sz_utf8_word_break_is_mid_quotes_(after_prop))) {
        sz_size_t lookahead = after_position;
        lookahead = sz_utf8_skip_ignorables_forward_(text, length, lookahead);
        if (lookahead < length) {
            sz_size_t lookahead_position = lookahead;
            sz_rune_t la_rune = sz_utf8_decode_(text, length, &lookahead_position);
            sz_u8_t lookahead_property = sz_rune_word_break_property(la_rune);
            if (lookahead_property == sz_tr29_word_break_numeric_k) return sz_false_k;
        }
    }

    // WB12: Do not break Numeric (MidNum|MidNumLetQ) × Numeric (reverse of WB11)
    if ((previous_property == sz_tr29_word_break_midnum_k || sz_utf8_word_break_is_mid_quotes_(previous_property)) &&
        after_prop == sz_tr29_word_break_numeric_k) {
        // Check if preceded by Numeric
        sz_size_t previous_cp_start = position - 1;
        while (previous_cp_start > 0 && ((sz_u8_t)text[previous_cp_start] & 0xC0) == 0x80) previous_cp_start--;
        if (previous_cp_start > 0) {
            sz_size_t pre_previous_codepoint_start = previous_cp_start - 1;
            while (pre_previous_codepoint_start > 0 && ((sz_u8_t)text[pre_previous_codepoint_start] & 0xC0) == 0x80)
                pre_previous_codepoint_start--;
            sz_size_t pre_previous_position = pre_previous_codepoint_start;
            sz_rune_t pp_rune = sz_utf8_decode_(text, previous_cp_start, &pre_previous_position);
            sz_u8_t pre_previous_property = sz_rune_word_break_property(pp_rune);
            if (pre_previous_property == sz_tr29_word_break_numeric_k) return sz_false_k;
        }
    }

    // WB13: Do not break Katakana × Katakana
    if (previous_property == sz_tr29_word_break_katakana_k && after_prop == sz_tr29_word_break_katakana_k)
        return sz_false_k;

    // WB13a: Do not break (AHLetter|Numeric|Katakana|ExtendNumLet) × ExtendNumLet
    if ((sz_utf8_word_break_is_aletter_or_hebrew_(previous_property) ||
         previous_property == sz_tr29_word_break_numeric_k || previous_property == sz_tr29_word_break_katakana_k ||
         previous_property == sz_tr29_word_break_extendnumlet_k) &&
        after_prop == sz_tr29_word_break_extendnumlet_k)
        return sz_false_k;

    // WB13b: Do not break ExtendNumLet × (AHLetter|Numeric|Katakana)
    if (previous_property == sz_tr29_word_break_extendnumlet_k &&
        (sz_utf8_word_break_is_aletter_or_hebrew_(after_prop) || after_prop == sz_tr29_word_break_numeric_k ||
         after_prop == sz_tr29_word_break_katakana_k))
        return sz_false_k;

    // WB15/16: Do not break between Regional Indicators (keep pairs together)
    if (previous_property == sz_tr29_word_break_regional_ind_k && after_prop == sz_tr29_word_break_regional_ind_k) {
        // Count RI before - if odd, don't break (we're in the middle of a pair)
        sz_size_t regional_indicator_count = sz_utf8_word_break_count_regional_indicators_before_(text, position);
        if (regional_indicator_count % 2 == 1) return sz_false_k;
    }

    // WB999: Otherwise, break everywhere
    return sz_true_k;
}

/*  Plural UAX-29 word segmentation.
 *
 *  A single left-to-right sweep emits every word into two parallel arrays - `word_starts[i]` is the byte
 *  offset of the i-th word and `word_lengths[i]` its byte length - so one call segments the whole input
 *  without the caller looping. The boundary set is exactly the set of positions for which
 *  `sz_utf8_is_word_boundary_serial` is true.
 *
 *  When the output arrays fill before the input is exhausted, `*bytes_consumed` is set to the start of the
 *  first word that did not fit - always a true TR29 boundary - so a caller can resume from `text +
 *  *bytes_consumed` with a fresh buffer and obtain the identical remainder (a boundary resets all WB context,
 *  so resuming with start-of-text semantics at a boundary is exact). Consecutive boundaries are always at
 *  least one codepoint apart, so no zero-length word is ever emitted; `length == 0` emits zero words.
 */
SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_serial( //
    sz_cptr_t text, sz_size_t length,                    //
    sz_size_t *word_starts, sz_size_t *word_lengths,     //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_size_t word_start = 0; // Start of the word currently being accumulated (always a boundary).
    // Position 0 is always a boundary, so the first reportable interior boundary is after the first codepoint.
    sz_size_t position = sz_utf8_codepoint_length_((sz_u8_t)text[0]);

    while (position < length) {
        if (sz_utf8_is_word_boundary_serial(text, length, position)) {
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_start;
                return words;
            }
            word_starts[words] = word_start;
            word_lengths[words] = position - word_start;
            ++words;
            word_start = position;
        }
        position += sz_utf8_codepoint_length_((sz_u8_t)text[position]);
    }

    // The trailing span [word_start, length) is the last word (end of text is always a boundary).
    if (words == words_capacity) {
        if (bytes_consumed) *bytes_consumed = word_start;
        return words;
    }
    word_starts[words] = word_start;
    word_lengths[words] = length - word_start;
    ++words;
    if (bytes_consumed) *bytes_consumed = length;
    return words;
}

/*  Reverse counterpart of `sz_utf8_word_find_boundaries_serial`: words are emitted from the end of the text
 *  backward (`word_starts[0]` is the last word, `word_starts[1]` the one before it, and so on). The boundary
 *  decision still uses the forward predicate `sz_utf8_is_word_boundary_serial`, which always sees full left
 *  context, so the segmentation matches the forward pass.
 *  On a full buffer `*bytes_consumed` is set to the end offset of the earliest (leftmost) emitted word - a true
 *  boundary - so a caller resumes by calling again with `length == *bytes_consumed`.
 */
SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_serial( //
    sz_cptr_t text, sz_size_t length,                     //
    sz_size_t *word_starts, sz_size_t *word_lengths,      //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = length;
        return 0;
    }

    sz_size_t word_end = length; // End of the word currently being accumulated (always a boundary).
    // Position `length` is always a boundary, so step back one codepoint to the first reportable position.
    sz_size_t position = length - 1;
    while (position > 0 && ((sz_u8_t)text[position] & 0xC0) == 0x80) position--;

    while (position > 0) {
        if (sz_utf8_is_word_boundary_serial(text, length, position)) {
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_end;
                return words;
            }
            word_starts[words] = position;
            word_lengths[words] = word_end - position;
            ++words;
            word_end = position;
        }
        position--;
        while (position > 0 && ((sz_u8_t)text[position] & 0xC0) == 0x80) position--;
    }

    // The leading span [0, word_end) is the last word emitted (start of text is always a boundary).
    if (words == words_capacity) {
        if (bytes_consumed) *bytes_consumed = word_end;
        return words;
    }
    word_starts[words] = 0;
    word_lengths[words] = word_end;
    ++words;
    if (bytes_consumed) *bytes_consumed = 0;
    return words;
}

#pragma endregion // UAX-29 Word Boundaries

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_ITERATE_SERIAL_H_
