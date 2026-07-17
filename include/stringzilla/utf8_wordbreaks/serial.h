/**
 *  @brief Serial backend for UAX-29 word boundaries.
 *  @file include/stringzilla/utf8_wordbreaks/serial.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_WORDBREAKS_SERIAL_H_
#define STRINGZILLA_UTF8_WORDBREAKS_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_wordbreaks/tables.h"
#include "stringzilla/utf8_runes/serial.h" // shared decode helpers

#ifdef __cplusplus
extern "C" {
#endif

#pragma region UAX 29 Word Boundaries

/**
 *  @brief Returns the UAX-29 Word_Break property (0-15) for a codepoint.
 *
 *  The oracle every SIMD Word_Break classifier is bit-exact against: arithmetic big ranges first, then a flat
 *  low-plane LUT for codepoint < 0x800, then a B=8 / SB=16 trie over the rest of the BMP, then a sorted astral
 *  range list, defaulting to Other for everything else.
 */
SZ_API_COMPTIME sz_u8_t sz_rune_word_break_property(sz_rune_t rune) {
    for (sz_size_t range = 0; range < sz_utf8_word_break_big_count_k; ++range)
        if (rune >= sz_utf8_word_break_big_lo_[range] && rune <= sz_utf8_word_break_big_hi_[range])
            return sz_utf8_word_break_big_cls_[range];

    if (rune < 0x800u) return sz_utf8_word_break_flat_lut_0800_[rune];

    if (rune < 0x10000u) {
        sz_u32_t const offset = (sz_u32_t)(rune - 0x800u);
        sz_u32_t const block = offset / sz_utf8_word_break_trie_block_k;
        sz_u32_t const within = offset % sz_utf8_word_break_trie_block_k;
        sz_u32_t const super = block / sz_utf8_word_break_trie_subblock_k;
        sz_u32_t const super_offset = block % sz_utf8_word_break_trie_subblock_k;
        sz_u8_t const level1 = sz_utf8_word_break_trie_l1_[super];
        sz_u16_t const leaf = sz_utf8_word_break_trie_l2_[level1 * sz_utf8_word_break_trie_subblock_k + super_offset];
        return sz_utf8_word_break_trie_leaf_[leaf * sz_utf8_word_break_trie_block_k + within];
    }

    for (sz_size_t range = 0; range < sz_utf8_word_break_astral_count_k; ++range)
        if (rune >= sz_utf8_word_break_astral_lo_[range] && rune <= sz_utf8_word_break_astral_hi_[range])
            return sz_utf8_word_break_astral_cls_[range];

    return sz_utf8_word_break_other_k;
}

/**
 *  @brief Check if a codepoint is a "word character" (has a word-forming Word_Break property).
 */
SZ_API_COMPTIME sz_bool_t sz_rune_is_word_char(sz_rune_t rune) {
    sz_u8_t property = sz_rune_word_break_property(rune);
    // Word characters: ALetter(8), Hebrew_Letter(9), Numeric(10), Katakana(11),
    // ExtendNumLet(12), MidLetter(13), MidNum(14), MidNumLet/Quotes(15)
    return (sz_bool_t)(property >= sz_utf8_word_break_aletter_k);
}

/**
 *  @brief Check if @p rune is Extended_Pictographic (UAX-29 WB3c). Not part of the 4-bit Word_Break model, so it is
 *         resolved by binary search over the sorted Extended_Pictographic range table.
 */
SZ_API_COMPTIME sz_bool_t sz_rune_is_extended_pictographic(sz_rune_t rune) {
    int low = 0, high = (int)sz_utf8_word_break_pict_u32_count_k - 1;
    while (low <= high) {
        int const mid = low + (high - low) / 2;
        if ((sz_u32_t)rune < sz_utf8_word_break_pict_u32_lo_[mid]) { high = mid - 1; }
        else if ((sz_u32_t)rune > sz_utf8_word_break_pict_u32_hi_[mid]) { low = mid + 1; }
        else { return sz_true_k; }
    }
    return sz_false_k;
}

/** @brief Check if @p rune is WSegSpace (UAX-29 WB3d); resolved by range membership (six ranges). */
SZ_API_COMPTIME sz_bool_t sz_rune_is_wsegspace(sz_rune_t rune) {
    for (int range = 0; range < (int)sz_utf8_word_break_wseg_u32_count_k; ++range)
        if ((sz_u32_t)rune >= sz_utf8_word_break_wseg_u32_lo_[range] &&
            (sz_u32_t)rune <= sz_utf8_word_break_wseg_u32_hi_[range])
            return sz_true_k;
    return sz_false_k;
}

/*  Each Word_Break property fits in 4 bits, so a 16-bit constant is a membership set: bit `p` marks property
 *  `p` as a member. Testing membership is then a shift-and-mask, replacing chains of equality comparisons in
 *  the hot WB4 skip loops. */
enum {
    sz_utf8_word_break_ignorable_set_k = (1u << sz_utf8_word_break_extend_k) | (1u << sz_utf8_word_break_zwj_k) |
                                         (1u << sz_utf8_word_break_format_k),
    sz_utf8_word_break_aletter_or_hebrew_set_k = (1u << sz_utf8_word_break_aletter_k) |
                                                 (1u << sz_utf8_word_break_hebrew_letter_k),
    sz_utf8_word_break_mid_quotes_set_k = (1u << sz_utf8_word_break_mid_quotes_k),
};

/** @brief Check if a property is WB4-ignorable (Extend, Format, ZWJ). */
SZ_HELPER_INLINE sz_bool_t sz_utf8_word_break_is_ignorable_(sz_u8_t property) {
    return (sz_bool_t)((sz_utf8_word_break_ignorable_set_k >> property) & 1u);
}

/** @brief Check if a property is AHLetter (ALetter or Hebrew_Letter). */
SZ_HELPER_INLINE sz_bool_t sz_utf8_word_break_is_aletter_or_hebrew_(sz_u8_t property) {
    return (sz_bool_t)((sz_utf8_word_break_aletter_or_hebrew_set_k >> property) & 1u);
}

/**
 *  @brief Check if a property is MidNumLetQ (MidNumLet or Single_Quote).
 *         In our encoding, MID_QUOTES (15) covers MidNumLet + quotes.
 */
SZ_HELPER_INLINE sz_bool_t sz_utf8_word_break_is_mid_quotes_(sz_u8_t property) {
    return (sz_bool_t)((sz_utf8_word_break_mid_quotes_set_k >> property) & 1u);
}

/**
 *  @brief Portable UAX-29 "join" (guaranteed non-boundary) mask for an all-ASCII window.
 *
 *  From per-lane class masks, returns a mask whose bit `i` is set when the boundary before lane `i` is
 *  suppressed by a no-break rule. ASCII never triggers WB4 or WB15/16, so the rules reduce to neighbour shifts;
 *  the result is exact only for lanes whose i-2 and i+1 neighbours are in-window.
 */
SZ_HELPER_INLINE sz_u64_t sz_utf8_word_break_join_from_class_masks_(                                   //
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
SZ_HELPER_AUTO sz_size_t sz_utf8_skip_ignorables_forward_(sz_cptr_t text, sz_size_t length, sz_size_t position) {
    while (position < length) {
        sz_size_t next_position = position;
        sz_rune_t rune = sz_utf8_next_rune_(text, length, &next_position);
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
SZ_HELPER_AUTO sz_u8_t sz_utf8_word_break_effective_property_(sz_cptr_t text, sz_size_t length, sz_size_t position,
                                                              sz_size_t *next_position) {
    sz_size_t current = position;
    sz_rune_t rune = sz_utf8_next_rune_(text, length, &current);
    sz_u8_t property = sz_rune_word_break_property(rune);

    // Skip ignorables to find effective next
    while (current < length && sz_utf8_word_break_is_ignorable_(property)) {
        rune = sz_utf8_next_rune_(text, length, &current);
        property = sz_rune_word_break_property(rune);
    }

    if (next_position) *next_position = current;
    return property;
}

/**
 *  @brief Look back to find the previous non-ignorable property.
 *  @return The property of the previous significant character, or sz_utf8_word_break_other_k if none.
 */
SZ_HELPER_AUTO sz_u8_t sz_utf8_word_break_previous_property_(sz_cptr_t text, sz_size_t position) {
    if (position == 0) return sz_utf8_word_break_other_k;

    // Scan backward to find start of previous codepoint
    sz_size_t previous = position - 1;
    while (previous > 0 && ((sz_u8_t)text[previous] & 0xC0) == 0x80) previous--;

    // Decode and get property
    sz_size_t decode_position = previous;
    sz_rune_t rune = sz_utf8_next_rune_(text, position, &decode_position);
    sz_u8_t property = sz_rune_word_break_property(rune);

    // Skip back over ignorables (WB4). EXCEPTION: an ignorable immediately after a Newline/CR/LF — or at sot — is
    // "de-ignored" and acts as its own base, so stop and return ITS property rather than the newline behind it
    // (otherwise WB3/WB3a would wrongly fire on the codepoint after the de-ignored ignorable).
    while (sz_utf8_word_break_is_ignorable_(property)) {
        if (previous == 0) break;
        sz_size_t predecessor = previous - 1;
        while (predecessor > 0 && ((sz_u8_t)text[predecessor] & 0xC0) == 0x80) predecessor--;
        sz_size_t predecessor_decode = predecessor;
        sz_rune_t predecessor_rune = sz_utf8_next_rune_(text, previous, &predecessor_decode);
        sz_u8_t predecessor_property = sz_rune_word_break_property(predecessor_rune);
        if (predecessor_property == sz_utf8_word_break_newline_k || predecessor_property == sz_utf8_word_break_cr_k ||
            predecessor_property == sz_utf8_word_break_lf_k)
            break;
        previous = predecessor;
        property = predecessor_property;
    }

    return property;
}

/**
 *  @brief Count Regional Indicators before position (for WB15/16).
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_word_break_count_regional_indicators_before_(sz_cptr_t text, sz_size_t position) {
    sz_size_t count = 0;
    sz_size_t current = position;

    while (current > 0) {
        // Find previous codepoint
        sz_size_t previous = current - 1;
        while (previous > 0 && ((sz_u8_t)text[previous] & 0xC0) == 0x80) previous--;

        sz_size_t decode_position = previous;
        sz_rune_t rune = sz_utf8_next_rune_(text, current, &decode_position);
        sz_u8_t property = sz_rune_word_break_property(rune);

        if (property == sz_utf8_word_break_regional_ind_k) {
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
 *  @brief One UAX-29 "element": a base codepoint together with the run of Extend/Format/ZWJ it absorbs (WB4).
 *
 *  The Word_Break rules WB5-WB16 operate on these elements, not raw codepoints: WB4 folds every Extend, Format,
 *  and ZWJ into the preceding base, EXCEPT when that base is sot, CR, LF, or a Newline (then the ignorable is
 *  "de-ignored" and starts its own element). `ends_in_zwj` records whether the element's LAST codepoint is a ZWJ,
 *  which is all WB3c (`ZWJ x Extended_Pictographic`) needs. `valid` is false for the sot/eot sentinels.
 */
typedef struct sz_word_element_t {
    /** Word_Break property of the base codepoint. */
    sz_u8_t property;
    /** Base codepoint (disambiguates Single_Quote / Double_Quote and Extended_Pictographic). */
    sz_rune_t codepoint;
    /** Byte offset of the base codepoint. */
    sz_size_t start;
    /** The element's final codepoint is U+200D ZERO WIDTH JOINER. */
    sz_bool_t ends_in_zwj;
    /** False for the sot/eot sentinel. */
    sz_bool_t valid;
} sz_word_element_t;

/** @brief True for the newline family CR/LF/Newline, which neither absorb (WB4) nor are absorbed. */
SZ_HELPER_INLINE sz_bool_t sz_word_is_newline_(sz_u8_t property) {
    return (sz_bool_t)(property == sz_utf8_word_break_cr_k || property == sz_utf8_word_break_lf_k ||
                       property == sz_utf8_word_break_newline_k);
}

/*  The 4-bit model lumps Single_Quote (U+0027), Double_Quote (U+0022), and MidNumLet into MID_QUOTES, so the
 *  WB6/WB7/WB7a-c/WB11/WB12 distinctions are recovered from the codepoint. MidNumLetQ = MidNumLet + Single_Quote,
 *  i.e. every MID_QUOTES codepoint that is NOT the Double_Quote. */
SZ_HELPER_INLINE sz_bool_t sz_word_is_single_quote_(sz_u8_t property, sz_rune_t codepoint) {
    return (sz_bool_t)(property == sz_utf8_word_break_mid_quotes_k && codepoint == 0x0027u);
}
SZ_HELPER_INLINE sz_bool_t sz_word_is_double_quote_(sz_u8_t property, sz_rune_t codepoint) {
    return (sz_bool_t)(property == sz_utf8_word_break_mid_quotes_k && codepoint == 0x0022u);
}
SZ_HELPER_INLINE sz_bool_t sz_word_is_mid_num_let_q_(sz_u8_t property, sz_rune_t codepoint) {
    return (sz_bool_t)(property == sz_utf8_word_break_mid_quotes_k && codepoint != 0x0022u);
}

/** @brief Byte offset of the codepoint start immediately before @p position (which must be > 0), using the canonical
 *         maximal-subpart partition so it agrees with the forward `sz_utf8_next_rune_` on malformed input: a run of
 *         continuation bytes is absorbed into the nearest preceding lead ONLY if that lead's sequence actually reaches
 *         @p position; otherwise the byte just before @p position is a stray continuation (its own U+FFFD). A naive
 *         continuation-skip would absorb a stray into the previous codepoint and SKIP it during the backward look-back,
 *         crossing a streamed resume point and forging a WB5/6/7 join the forward pass never sees - that asymmetry is
 *         the capacity-dependence. Treating every stray as its own non-skippable U+FFFD makes resume reproduce. */
SZ_HELPER_AUTO sz_size_t sz_word_previous_start_(sz_cptr_t text, sz_size_t position) {
    sz_size_t lead = position - 1;
    while (lead > 0 && ((sz_u8_t)text[lead] & 0xC0) == 0x80) lead--;
    sz_size_t reach = lead;
    sz_utf8_next_rune_(text, position, &reach);
    return reach == position ? lead : position - 1;
}

/** @brief The element whose final codepoint ends just before @p position (sot sentinel when @p position == 0). */
SZ_HELPER_AUTO sz_word_element_t sz_word_previous_element_(sz_cptr_t text, sz_size_t position) {
    sz_word_element_t element;
    element.valid = sz_false_k;
    if (position == 0) return element;

    sz_size_t last_start = sz_word_previous_start_(text, position);
    sz_size_t decode_at = last_start;
    sz_rune_t last_codepoint = sz_utf8_next_rune_(text, position, &decode_at);
    sz_u8_t last_property = sz_rune_word_break_property(last_codepoint);
    element.ends_in_zwj = (sz_bool_t)(last_property == sz_utf8_word_break_zwj_k);

    sz_size_t base_start = last_start;
    sz_rune_t base_codepoint = last_codepoint;
    sz_u8_t base_property = last_property;
    while (sz_utf8_word_break_is_ignorable_(base_property) && base_start > 0) {
        sz_size_t predecessor_start = sz_word_previous_start_(text, base_start);
        sz_size_t predecessor_at = predecessor_start;
        sz_rune_t predecessor_codepoint = sz_utf8_next_rune_(text, base_start, &predecessor_at);
        //  WB4 exception: an ignorable directly after a Newline/CR/LF is de-ignored and is its own base.
        if (sz_word_is_newline_(sz_rune_word_break_property(predecessor_codepoint))) break;
        base_start = predecessor_start;
        base_codepoint = predecessor_codepoint;
        base_property = sz_rune_word_break_property(predecessor_codepoint);
    }
    element.property = base_property;
    element.codepoint = base_codepoint;
    element.start = base_start;
    element.valid = sz_true_k;
    return element;
}

/** @brief The element following the one whose base is at @p position (eot sentinel at end of text). */
SZ_HELPER_AUTO sz_word_element_t sz_word_next_element_(sz_cptr_t text, sz_size_t length, sz_size_t position) {
    sz_word_element_t element;
    element.valid = sz_false_k;
    sz_size_t cursor = position;
    sz_rune_t base_codepoint = sz_utf8_next_rune_(text, length, &cursor);
    if (!sz_word_is_newline_(sz_rune_word_break_property(base_codepoint))) {
        while (cursor < length) {
            sz_size_t probe = cursor;
            sz_rune_t codepoint = sz_utf8_next_rune_(text, length, &probe);
            if (!sz_utf8_word_break_is_ignorable_(sz_rune_word_break_property(codepoint))) break;
            cursor = probe;
        }
    }
    if (cursor >= length) return element;
    sz_size_t decode_at = cursor;
    sz_rune_t next_codepoint = sz_utf8_next_rune_(text, length, &decode_at);
    element.property = sz_rune_word_break_property(next_codepoint);
    element.codepoint = next_codepoint;
    element.start = cursor;
    element.ends_in_zwj = sz_false_k;
    element.valid = sz_true_k;
    return element;
}

/** @brief Count of contiguous Regional_Indicator elements immediately before @p position (for WB15/WB16 parity). */
SZ_HELPER_AUTO sz_size_t sz_word_regional_run_before_(sz_cptr_t text, sz_size_t position) {
    sz_size_t count = 0;
    sz_size_t cursor = position;
    for (;;) {
        sz_word_element_t element = sz_word_previous_element_(text, cursor);
        if (!element.valid || element.property != sz_utf8_word_break_regional_ind_k) break;
        ++count;
        cursor = element.start;
    }
    return count;
}

/**
 *  @brief Whether @p position is a UAX-29 word boundary. Direct, branch-per-rule transcription of WB1-WB16 over
 *         the WB4 element model; used unchanged by the forward and reverse drivers (and every backend that
 *         delegates here), so segmentation is identical in either direction.
 */
SZ_API_COMPTIME sz_bool_t sz_utf8_is_word_boundary_serial(sz_cptr_t text, sz_size_t length, sz_size_t position) {
    if (position == 0) return sz_true_k;      // WB1
    if (position >= length) return sz_true_k; // WB2
    // Never split INSIDE a codepoint - but only a continuation byte genuinely covered by a preceding lead's
    // maximal-subpart sequence is interior. A stray continuation (the lead stops short of it) is its OWN U+FFFD
    // codepoint, so a boundary may fall before it; matching the canonical forward decode keeps streamed segmentation
    // capacity-independent on malformed input.
    if (((sz_u8_t)text[position] & 0xC0) == 0x80) {
        sz_size_t lead = position;
        while (lead > 0 && ((sz_u8_t)text[lead] & 0xC0) == 0x80) lead--;
        sz_size_t reach = lead;
        sz_utf8_next_rune_(text, length, &reach);
        if (reach > position) return sz_false_k; // interior byte of a valid multi-byte codepoint
    }

    sz_size_t after_at = position;
    sz_rune_t next_codepoint = sz_utf8_next_rune_(text, length, &after_at);
    sz_u8_t next_property = sz_rune_word_break_property(next_codepoint);

    sz_size_t immediate_start = sz_word_previous_start_(text, position);
    sz_size_t immediate_at = immediate_start;
    sz_rune_t immediate_codepoint = sz_utf8_next_rune_(text, position, &immediate_at);
    sz_u8_t immediate_property = sz_rune_word_break_property(immediate_codepoint);

    if (immediate_property == sz_utf8_word_break_cr_k && next_property == sz_utf8_word_break_lf_k)
        return sz_false_k;                                                  // WB3  CR x LF
    if (sz_word_is_newline_(immediate_property)) return sz_true_k;          // WB3a break after Newline/CR/LF
    if (sz_word_is_newline_(next_property)) return sz_true_k;               // WB3b break before Newline/CR/LF
    if (sz_utf8_word_break_is_ignorable_(next_property)) return sz_false_k; // WB4 absorb Extend/Format/ZWJ

    sz_word_element_t previous = sz_word_previous_element_(text, position);
    sz_u8_t previous_property = previous.property;
    sz_rune_t previous_codepoint = previous.codepoint;

    if (previous.ends_in_zwj && sz_rune_is_extended_pictographic(next_codepoint))
        return sz_false_k; // WB3c ZWJ x Extended_Pictographic
    // WB3c/WB3d precede WB4, so they test RAW adjacency: WB3d needs the immediate predecessor codepoint (an
    // intervening Extend, absorbed by WB4, must still force the break) rather than the folded element base.
    if (sz_rune_is_wsegspace(immediate_codepoint) && sz_rune_is_wsegspace(next_codepoint))
        return sz_false_k; // WB3d WSegSpace x WSegSpace

    sz_bool_t previous_ah = sz_utf8_word_break_is_aletter_or_hebrew_(previous_property);
    sz_bool_t next_ah = sz_utf8_word_break_is_aletter_or_hebrew_(next_property);

    if (previous_ah && next_ah) return sz_false_k; // WB5
    if (previous_ah &&
        (next_property == sz_utf8_word_break_midletter_k || sz_word_is_mid_num_let_q_(next_property, next_codepoint))) {
        sz_word_element_t after = sz_word_next_element_(text, length, position);
        if (after.valid && sz_utf8_word_break_is_aletter_or_hebrew_(after.property)) return sz_false_k; // WB6
    }
    if ((previous_property == sz_utf8_word_break_midletter_k ||
         sz_word_is_mid_num_let_q_(previous_property, previous_codepoint)) &&
        next_ah) {
        sz_word_element_t before = sz_word_previous_element_(text, previous.start);
        if (before.valid && sz_utf8_word_break_is_aletter_or_hebrew_(before.property)) return sz_false_k; // WB7
    }
    if (previous_property == sz_utf8_word_break_hebrew_letter_k &&
        sz_word_is_single_quote_(next_property, next_codepoint))
        return sz_false_k; // WB7a Hebrew x Single_Quote
    if (previous_property == sz_utf8_word_break_hebrew_letter_k &&
        sz_word_is_double_quote_(next_property, next_codepoint)) {
        sz_word_element_t after = sz_word_next_element_(text, length, position);
        if (after.valid && after.property == sz_utf8_word_break_hebrew_letter_k) return sz_false_k; // WB7b
    }
    if (sz_word_is_double_quote_(previous_property, previous_codepoint) &&
        next_property == sz_utf8_word_break_hebrew_letter_k) {
        sz_word_element_t before = sz_word_previous_element_(text, previous.start);
        if (before.valid && before.property == sz_utf8_word_break_hebrew_letter_k) return sz_false_k; // WB7c
    }
    if (previous_property == sz_utf8_word_break_numeric_k && next_property == sz_utf8_word_break_numeric_k)
        return sz_false_k;                                                               // WB8
    if (previous_ah && next_property == sz_utf8_word_break_numeric_k) return sz_false_k; // WB9
    if (previous_property == sz_utf8_word_break_numeric_k && next_ah) return sz_false_k; // WB10
    if ((previous_property == sz_utf8_word_break_midnum_k ||
         sz_word_is_mid_num_let_q_(previous_property, previous_codepoint)) &&
        next_property == sz_utf8_word_break_numeric_k) {
        sz_word_element_t before = sz_word_previous_element_(text, previous.start);
        if (before.valid && before.property == sz_utf8_word_break_numeric_k) return sz_false_k; // WB11
    }
    if (previous_property == sz_utf8_word_break_numeric_k &&
        (next_property == sz_utf8_word_break_midnum_k || sz_word_is_mid_num_let_q_(next_property, next_codepoint))) {
        sz_word_element_t after = sz_word_next_element_(text, length, position);
        if (after.valid && after.property == sz_utf8_word_break_numeric_k) return sz_false_k; // WB12
    }
    if (previous_property == sz_utf8_word_break_katakana_k && next_property == sz_utf8_word_break_katakana_k)
        return sz_false_k; // WB13
    if ((previous_ah || previous_property == sz_utf8_word_break_numeric_k ||
         previous_property == sz_utf8_word_break_katakana_k ||
         previous_property == sz_utf8_word_break_extendnumlet_k) &&
        next_property == sz_utf8_word_break_extendnumlet_k)
        return sz_false_k; // WB13a
    if (previous_property == sz_utf8_word_break_extendnumlet_k &&
        (next_ah || next_property == sz_utf8_word_break_numeric_k || next_property == sz_utf8_word_break_katakana_k))
        return sz_false_k; // WB13b
    if (previous_property == sz_utf8_word_break_regional_ind_k && next_property == sz_utf8_word_break_regional_ind_k) {
        if (sz_word_regional_run_before_(text, previous.start) % 2 == 0) return sz_false_k; // WB15/WB16
    }
    return sz_true_k; // WB999
}

/**
 *  @brief Forward run-state carried across codepoints by the bulk segmenter so WB3..WB16 resolve without the
 *         per-position backward re-walks the `sz_utf8_is_word_boundary_serial` oracle performs. The oracle re-derives
 *         the WB4 effective-previous element, the element two bases back, and the Regional_Indicator run length
 *         BACKWARD on every position (the run re-count is O(n²) on long Regional_Indicator runs); here they are tracked
 *         forward. The forward WB6/WB7b/WB12 lookahead keeps reusing `sz_word_next_element_` (the same bounded forward
 *         fold the oracle uses), so only the backward rescans are removed. Fields mirror `sz_utf8_word_break_carry_t`.
 */
typedef struct sz_word_serial_state_t {
    /** Word_Break class of the WB4 effective-previous element's base codepoint. */
    sz_u8_t previous_property;
    /** That base codepoint (disambiguates Single_Quote / Double_Quote and Extended_Pictographic). */
    sz_rune_t previous_codepoint;
    /** Class of the element one base further back (the WB7 / WB11 / WB7c left context). */
    sz_u8_t before_property;
    /** That base codepoint. */
    sz_rune_t before_codepoint;
    /** The previous element's last codepoint is U+200D ZERO WIDTH JOINER (WB3c). */
    sz_bool_t previous_ends_in_zwj;
    /** Class of the raw immediately-previous codepoint (WB3 / WB3a / WB3d). */
    sz_u8_t previous_raw_property;
    /** That raw codepoint (WB3d WSegSpace x WSegSpace). */
    sz_rune_t previous_raw_codepoint;
    /** The Regional_Indicator run ending at the previous element has odd length (WB15 / WB16). */
    sz_bool_t regional_indicator_run_odd;
    /** A codepoint has been processed (clears the WB1 start-of-text state). */
    sz_bool_t has_previous;
} sz_word_serial_state_t;

/** @brief Advance @p state by one codepoint: fold it into the previous element (WB4) or open a new element base,
 *         maintaining the two-back base chain, the Regional_Indicator parity, and the raw-previous fields. */
SZ_HELPER_AUTO void sz_word_serial_advance_(sz_word_serial_state_t *state, sz_u8_t property, sz_rune_t codepoint) {
    sz_bool_t const after_newline = (sz_bool_t)(state->has_previous &&
                                                sz_word_is_newline_(state->previous_raw_property));
    sz_bool_t const is_ignorable = sz_utf8_word_break_is_ignorable_(property);
    // A non-ignorable codepoint, the very first codepoint, or an ignorable directly after a Newline (WB4 de-ignore)
    // opens a fresh element base; every other ignorable is absorbed into the standing element (its base is unchanged).
    if (!is_ignorable || after_newline || !state->has_previous) {
        state->before_property = state->previous_property;
        state->before_codepoint = state->previous_codepoint;
        state->previous_property = property;
        state->previous_codepoint = codepoint;
        state->regional_indicator_run_odd = (property == sz_utf8_word_break_regional_ind_k)
                                                ? (sz_bool_t)(!state->regional_indicator_run_odd)
                                                : sz_false_k;
    }
    state->previous_ends_in_zwj = (sz_bool_t)(property == sz_utf8_word_break_zwj_k);
    state->previous_raw_property = property;
    state->previous_raw_codepoint = codepoint;
    state->has_previous = sz_true_k;
}

/** @brief Boundary decision (WB3..WB999) between @p state's previous codepoint and the @p next codepoint. The only
 *         right-context rules (WB6 / WB7b / WB12) reuse the bounded `sz_word_next_element_` forward fold; all left
 *         context comes from @p state. Byte-identical to `sz_utf8_is_word_boundary_serial`. */
SZ_HELPER_AUTO sz_bool_t sz_word_serial_boundary_(sz_word_serial_state_t const *state, sz_u8_t next_property,
                                                  sz_rune_t next_codepoint, sz_cptr_t text, sz_size_t length,
                                                  sz_size_t position) {
    sz_u8_t const immediate_property = state->previous_raw_property;
    sz_rune_t const immediate_codepoint = state->previous_raw_codepoint;
    if (immediate_property == sz_utf8_word_break_cr_k && next_property == sz_utf8_word_break_lf_k)
        return sz_false_k;                                                  // WB3  CR x LF
    if (sz_word_is_newline_(immediate_property)) return sz_true_k;          // WB3a break after Newline/CR/LF
    if (sz_word_is_newline_(next_property)) return sz_true_k;               // WB3b break before Newline/CR/LF
    if (sz_utf8_word_break_is_ignorable_(next_property)) return sz_false_k; // WB4 absorb Extend/Format/ZWJ

    sz_u8_t const previous_property = state->previous_property;
    sz_rune_t const previous_codepoint = state->previous_codepoint;
    if (state->previous_ends_in_zwj && sz_rune_is_extended_pictographic(next_codepoint))
        return sz_false_k; // WB3c ZWJ x Extended_Pictographic
    if (sz_rune_is_wsegspace(immediate_codepoint) && sz_rune_is_wsegspace(next_codepoint))
        return sz_false_k; // WB3d WSegSpace x WSegSpace

    sz_bool_t const previous_is_alphabetic_or_hebrew = sz_utf8_word_break_is_aletter_or_hebrew_(previous_property);
    sz_bool_t const next_is_alphabetic_or_hebrew = sz_utf8_word_break_is_aletter_or_hebrew_(next_property);

    if (previous_is_alphabetic_or_hebrew && next_is_alphabetic_or_hebrew) return sz_false_k; // WB5
    if (previous_is_alphabetic_or_hebrew &&
        (next_property == sz_utf8_word_break_midletter_k || sz_word_is_mid_num_let_q_(next_property, next_codepoint))) {
        sz_word_element_t const after = sz_word_next_element_(text, length, position);
        if (after.valid && sz_utf8_word_break_is_aletter_or_hebrew_(after.property)) return sz_false_k; // WB6
    }
    if ((previous_property == sz_utf8_word_break_midletter_k ||
         sz_word_is_mid_num_let_q_(previous_property, previous_codepoint)) &&
        next_is_alphabetic_or_hebrew) {
        if (sz_utf8_word_break_is_aletter_or_hebrew_(state->before_property)) return sz_false_k; // WB7
    }
    if (previous_property == sz_utf8_word_break_hebrew_letter_k &&
        sz_word_is_single_quote_(next_property, next_codepoint))
        return sz_false_k; // WB7a Hebrew x Single_Quote
    if (previous_property == sz_utf8_word_break_hebrew_letter_k &&
        sz_word_is_double_quote_(next_property, next_codepoint)) {
        sz_word_element_t const after = sz_word_next_element_(text, length, position);
        if (after.valid && after.property == sz_utf8_word_break_hebrew_letter_k) return sz_false_k; // WB7b
    }
    if (sz_word_is_double_quote_(previous_property, previous_codepoint) &&
        next_property == sz_utf8_word_break_hebrew_letter_k) {
        if (state->before_property == sz_utf8_word_break_hebrew_letter_k) return sz_false_k; // WB7c
    }
    if (previous_property == sz_utf8_word_break_numeric_k && next_property == sz_utf8_word_break_numeric_k)
        return sz_false_k;                                                                                    // WB8
    if (previous_is_alphabetic_or_hebrew && next_property == sz_utf8_word_break_numeric_k) return sz_false_k; // WB9
    if (previous_property == sz_utf8_word_break_numeric_k && next_is_alphabetic_or_hebrew) return sz_false_k; // WB10
    if ((previous_property == sz_utf8_word_break_midnum_k ||
         sz_word_is_mid_num_let_q_(previous_property, previous_codepoint)) &&
        next_property == sz_utf8_word_break_numeric_k) {
        if (state->before_property == sz_utf8_word_break_numeric_k) return sz_false_k; // WB11
    }
    if (previous_property == sz_utf8_word_break_numeric_k &&
        (next_property == sz_utf8_word_break_midnum_k || sz_word_is_mid_num_let_q_(next_property, next_codepoint))) {
        sz_word_element_t const after = sz_word_next_element_(text, length, position);
        if (after.valid && after.property == sz_utf8_word_break_numeric_k) return sz_false_k; // WB12
    }
    if (previous_property == sz_utf8_word_break_katakana_k && next_property == sz_utf8_word_break_katakana_k)
        return sz_false_k; // WB13
    if ((previous_is_alphabetic_or_hebrew || previous_property == sz_utf8_word_break_numeric_k ||
         previous_property == sz_utf8_word_break_katakana_k ||
         previous_property == sz_utf8_word_break_extendnumlet_k) &&
        next_property == sz_utf8_word_break_extendnumlet_k)
        return sz_false_k; // WB13a
    if (previous_property == sz_utf8_word_break_extendnumlet_k &&
        (next_is_alphabetic_or_hebrew || next_property == sz_utf8_word_break_numeric_k ||
         next_property == sz_utf8_word_break_katakana_k))
        return sz_false_k; // WB13b
    if (previous_property == sz_utf8_word_break_regional_ind_k && next_property == sz_utf8_word_break_regional_ind_k) {
        if (state->regional_indicator_run_odd) return sz_false_k; // WB15/WB16
    }
    return sz_true_k; // WB999
}

/*  Plural UAX-29 word segmentation: ONE left-to-right sweep emits every word into parallel `word_starts` /
 *  `word_lengths`, carrying the WB run-state so each codepoint is decoded once and no boundary re-walks its left
 *  context (O(n), no per-position backward rescans). Byte-identical to driving `sz_utf8_is_word_boundary_serial` per
 *  position. On a full buffer `*bytes_consumed` is the start of the first word that did not fit - always a true TR29
 *  boundary - so a caller resumes from `text + *bytes_consumed` and obtains the identical remainder.
 */
SZ_API_COMPTIME sz_size_t sz_utf8_wordbreaks_serial( //
    sz_cptr_t text, sz_size_t length,                //
    sz_size_t *word_starts, sz_size_t *word_lengths, //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_word_serial_state_t state;
    state.previous_property = (sz_u8_t)sz_utf8_word_break_other_k;
    state.previous_codepoint = 0;
    state.before_property = (sz_u8_t)sz_utf8_word_break_other_k;
    state.before_codepoint = 0;
    state.previous_ends_in_zwj = sz_false_k;
    state.previous_raw_property = (sz_u8_t)sz_utf8_word_break_other_k;
    state.previous_raw_codepoint = 0;
    state.regional_indicator_run_odd = sz_false_k;
    state.has_previous = sz_false_k;

    // Position 0 is always a boundary (WB1); seed the state from the first codepoint and report interior boundaries
    // from the second onward. Step by the maximal-subpart partition (the same `sz_utf8_next_rune_` grid the predicate and
    // its forward look-ahead use) so a streamed resume reproduces the identical segmentation on malformed input.
    sz_size_t word_start = 0;
    sz_size_t position = 0;
    sz_rune_t const first_codepoint = sz_utf8_next_rune_(text, length, &position);
    sz_word_serial_advance_(&state, sz_rune_word_break_property(first_codepoint), first_codepoint);

    while (position < length) {
        sz_size_t next_position = position;
        sz_rune_t const next_codepoint = sz_utf8_next_rune_(text, length, &next_position);
        sz_u8_t const next_property = sz_rune_word_break_property(next_codepoint);
        if (sz_word_serial_boundary_(&state, next_property, next_codepoint, text, length, position)) {
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_start;
                return words;
            }
            word_starts[words] = word_start;
            word_lengths[words] = position - word_start;
            ++words;
            word_start = position;
        }
        sz_word_serial_advance_(&state, next_property, next_codepoint);
        position = next_position;
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

#pragma region Portable Word_Break Codepoint Partition

/**
 *  @brief  Canonical maximal-subpart codepoint partition (Unicode U+FFFD substitution, the serial reference's exact
 *          model after its malformed-UTF-8 fix). Implemented as a reachability fixpoint over the declared-length masks
 *          gated by @p claims_full; portable (intrinsic-free), shared by every backend.
 *
 *  @param claims_full  Lead lanes whose full declared multi-byte sequence is well-formed; a length>=2 lead NOT in this
 *                      set collapses to a 1-byte U+FFFD.
 */
SZ_HELPER_AUTO sz_u64_t sz_utf8_word_break_subpart_starts_(sz_u64_t length_one, sz_u64_t length_two,
                                                           sz_u64_t length_three, sz_u64_t length_four,
                                                           sz_u64_t claims_full, sz_u64_t valid) {
    sz_u64_t const length_ge_two = length_two | length_three | length_four;
    sz_u64_t const sub1 = length_one | (length_ge_two & ~claims_full);
    sz_u64_t const sub2 = length_two & claims_full;
    sz_u64_t const sub3 = length_three & claims_full;
    sz_u64_t const sub4 = length_four & claims_full;
    sz_u64_t reach = 1ull; // lane 0 is always a start in a codepoint-aligned window
    for (int iteration = 0; iteration < 64; ++iteration) {
        sz_u64_t const next = ((reach & sub1) << 1) | ((reach & sub2) << 2) | ((reach & sub3) << 3) |
                              ((reach & sub4) << 4);
        sz_u64_t const grown = reach | (next & valid);
        if (grown == reach) break;
        reach = grown;
    }
    return reach & valid;
}

/** @brief  The codepoint partition of one window plus the lanes that must be reclassified to U+FFFD (Other). */
typedef struct sz_utf8_word_break_partition_t {
    /** Codepoint-start lanes under the canonical maximal-subpart partition. */
    sz_u64_t start_bytes;
    /** Claimed continuation bytes (the interior of valid multi-byte codepoints). */
    sz_u64_t continuation;
    /** Lanes whose class must be forced to U+FFFD/Other: strays + short/ill-formed leads. */
    sz_u64_t forced_other;
    /** High-nibble declared-length lead masks (two/three/four bytes), reused by the block resolver's truncation. */
    sz_u64_t length_two;
    sz_u64_t length_three;
    sz_u64_t length_four;
} sz_utf8_word_break_partition_t;

/**
 *  @brief  Portable maximal-subpart partition resolver over precomputed `sz_u64_t` masks (the per-ISA extractor
 *          supplies @p real_continuation, the high-nibble declared-length masks, and @p bad_second_byte). A well-formed
 *          window collapses to the O(1) continuation-bit partition; only a stray continuation, a short lead, or an
 *          overlong/surrogate/range lead takes the data-dependent reachability fixpoint. @p at_end_of_text
 *          distinguishes a benign interior straddle from a true end-of-text truncation.
 */
SZ_HELPER_AUTO sz_utf8_word_break_partition_t sz_utf8_word_break_partition_from_masks_( //
    sz_u64_t real_continuation, sz_u64_t length_two, sz_u64_t length_three, sz_u64_t length_four,
    sz_u64_t bad_second_byte, sz_u64_t valid, int at_end_of_text) {
    sz_u64_t const length_one = valid & ~length_two & ~length_three & ~length_four;
    sz_u64_t const length_ge_two = length_two | length_three | length_four;
    sz_u64_t const claimed_unmasked = (length_ge_two << 1) | ((length_three | length_four) << 2) | (length_four << 3);
    sz_u64_t const claimed_optimistic = claimed_unmasked & valid;
    sz_u64_t const top_one_lane = valid & ~(valid >> 1);
    sz_u64_t const top_two_lanes = valid & ~(valid >> 2);
    sz_u64_t const top_three_lanes = valid & ~(valid >> 3);
    sz_u64_t const declared_past_window = (length_two & top_one_lane) | (length_three & top_two_lanes) |
                                          (length_four & top_three_lanes);
    int const end_of_text_truncated_lead = at_end_of_text && declared_past_window != 0;

    sz_utf8_word_break_partition_t result;
    result.length_two = length_two;
    result.length_three = length_three;
    result.length_four = length_four;
    if (real_continuation == claimed_optimistic && bad_second_byte == 0 && !end_of_text_truncated_lead) {
        result.start_bytes = valid & ~real_continuation;
        result.continuation = real_continuation;
        result.forced_other = 0ull;
    }
    else {
        sz_u64_t const continuation_at_1 = real_continuation >> 1, continuation_at_2 = real_continuation >> 2,
                       continuation_at_3 = real_continuation >> 3;
        sz_u64_t const claims_full = ((length_two & continuation_at_1) |
                                      (length_three & continuation_at_1 & continuation_at_2) |
                                      (length_four & continuation_at_1 & continuation_at_2 & continuation_at_3)) &
                                     ~bad_second_byte;
        sz_u64_t const start_bytes = sz_utf8_word_break_subpart_starts_(length_one, length_two, length_three,
                                                                        length_four, claims_full, valid);
        sz_u64_t const stray_continuation = start_bytes & real_continuation;
        sz_u64_t const short_lead = length_ge_two & ~claims_full & start_bytes;
        result.start_bytes = start_bytes;
        result.continuation = real_continuation & ~stray_continuation;
        result.forced_other = short_lead | stray_continuation;
    }
    return result;
}

#pragma endregion Portable Word_Break Codepoint Partition

#pragma region Portable Word_Break Block Rule Engine

/** @brief  Smear reach (steps): covers the widest 4-byte codepoint's continuation bytes for the WB3c/WB3d/RI
 *          adjacency smears (the letter/numeric reach uses unbounded `fill` instead). */
enum { sz_utf8_word_break_smear_steps_k = 12 };

/** @brief  Bridge-shadow left-letter kind carried across the window edge for the deferred WB6/7/11/12 bridge. */
enum {
    sz_utf8_word_break_bridge_none_k = 0,    /**< No open bridge shadow. */
    sz_utf8_word_break_bridge_aletter_k = 1, /**< Letter Mid* awaiting an AHLetter (WB6/WB7). */
    sz_utf8_word_break_bridge_numeric_k = 2, /**< Numeric Mid* awaiting a Numeric (WB11/WB12). */
    sz_utf8_word_break_bridge_hebrew_k = 3,  /**< Hebrew " awaiting a Hebrew_Letter (WB7b/WB7c). */
};

/**
 *  @brief  Cross-window left-context register carry: the effective `previous_property` at the next window's anchor,
 *          the open bridge-shadow run-state (`bridge_open` / `bridge_kind`), the open Regional_Indicator run parity
 *          (WB15/WB16), and the WB3d/WB3c raw-adjacency bits. Threaded forward so the vectorized path never re-walks
 *          a straddle, re-derives a carry, or calls the serial oracle.
 */
typedef struct sz_utf8_word_break_carry_t {
    /** 1 if a WB6/7/11/12 bridge shadow is open at the previous block's edge. */
    sz_u8_t bridge_open;
    /** Which left letter opened the shadow (@ref sz_utf8_word_break_bridge_none_k ...). */
    sz_u8_t bridge_kind;
    /** Word_Break class of the consumed, still-unresolved Mid* (the left context if the bridge fails). */
    sz_u8_t bridge_mid_class;
    /** 0 when WB7a (Hebrew_Letter x Single_Quote) forbids the deferred break at the mid. */
    sz_u8_t bridge_unprotected;
    /** Effective `previous_property` at the next window's first emitted lane. */
    sz_u8_t left_property;
    /** 0 only at the very start of the text (WB1 sot). */
    sz_u8_t have_prev;
    /** Parity of the contiguous Regional_Indicator run open at that lane (0 or 1). */
    sz_u8_t ri_parity;
    /** The codepoint immediately below lane 0 is a WSegSpace (WB3d across the edge). */
    sz_u8_t prev_is_wseg;
    /** The previous element's LAST codepoint is a bare ZWJ (WB3c across the edge). */
    sz_u8_t prev_ends_in_zwj;
} sz_utf8_word_break_carry_t;

/** @brief  One classified block resolved into per-lane boundary bits plus the byte the driver advances to. */
typedef struct sz_utf8_word_break_window_t {
    /** Bit `i` set => a UAX-29 word boundary begins at codepoint-lead lane `i`. */
    sz_u64_t breaks;
    /** Exclusive upper bound, in bytes, on lanes whose break bit is fully trusted. */
    sz_size_t resolved;
    /** 1 when a carried bridge failed: the driver emits one boundary at its anchored mid byte. */
    sz_u8_t deferred_break;
} sz_utf8_word_break_window_t;

/** @brief  Start-of-text carry: no previous element, all runs closed. */
SZ_HELPER_AUTO sz_utf8_word_break_carry_t sz_utf8_word_break_carry_sot_(void) {
    sz_utf8_word_break_carry_t carry;
    carry.bridge_open = 0;
    carry.bridge_kind = sz_utf8_word_break_bridge_none_k;
    carry.bridge_mid_class = (sz_u8_t)sz_utf8_word_break_other_k;
    carry.bridge_unprotected = 0;
    carry.left_property = (sz_u8_t)sz_utf8_word_break_other_k; // WB1: sot has no left context
    carry.have_prev = 0;
    carry.ri_parity = 0;
    carry.prev_is_wseg = 0;
    carry.prev_ends_in_zwj = 0;
    return carry;
}

/**
 *  @brief  WB15/WB16 join mask: within every maximal Regional_Indicator run, suppress the boundary before each RI
 *          whose index (counting from the run start, plus the inbound run parity) is odd. The per-lane parity is a
 *          segmented exclusive prefix-XOR of @p ri over its own runs, seeded by @p inbound_parity at the lowest
 *          lane, computed in log-depth Kogge-Stone doubling (no per-lane loop). @p ri lanes are RI codepoint
 *          starts; ignorables/continuations between two RIs are part of the same run via @p run_gate.
 */
SZ_HELPER_AUTO sz_u64_t sz_utf8_word_break_ri_join_( //
    sz_u64_t ri, sz_u64_t run_gate, sz_u8_t inbound_parity, sz_u64_t *inclusive_parity_out) {
    sz_u64_t bits = ri;
    sz_u64_t reach = run_gate;
    for (int shift = 1; shift < 64; shift <<= 1) {
        bits ^= (bits << shift) & reach;
        reach &= reach << shift;
    }
    if (inbound_parity) bits ^= sz_u64_fill_right_(run_gate & 1ull, run_gate);
    *inclusive_parity_out = bits;
    return ri & (bits ^ ri);
}

/** @brief  Per-lane class membership of one decoded 64-byte window, precomputed by a per-ISA extractor so the portable
 *          rule engine sources every mask from `sz_u64_t` words without touching the codepoint vectors. Every class
 *          mask is raw (NOT yet `& start_bytes`); the truncated-edge U+FFFD reclassify has ALREADY been applied by the
 *          builder, so the masks (and @ref classes_byte) reflect post-truncation classes. The WB3c (Extended_Pictographic),
 *          WB3d (WSegSpace), and Single_Quote / Double_Quote distinctions live outside the 4-bit class model and arrive
 *          as their own raw-membership masks. */
typedef struct sz_utf8_word_break_frame_t {
    sz_u64_t class_aletter;      /**< ALetter lanes (NOT including Hebrew_Letter). */
    sz_u64_t class_hebrew;       /**< Hebrew_Letter lanes. */
    sz_u64_t class_numeric;      /**< Numeric lanes. */
    sz_u64_t class_katakana;     /**< Katakana lanes. */
    sz_u64_t class_extendnumlet; /**< ExtendNumLet lanes. */
    sz_u64_t class_extend;       /**< Extend lanes. */
    sz_u64_t class_zwj;          /**< ZWJ lanes. */
    sz_u64_t class_format;       /**< Format lanes. */
    sz_u64_t class_midletter;    /**< MidLetter lanes. */
    sz_u64_t class_midnum;       /**< MidNum lanes. */
    sz_u64_t class_mid_quotes;   /**< MidNumLet/Quotes lanes (covers MidNumLet + ' + "). */
    sz_u64_t class_cr;           /**< CR lanes. */
    sz_u64_t class_lf;           /**< LF lanes. */
    sz_u64_t class_newline;      /**< Newline lanes. */
    sz_u64_t class_regional;     /**< Regional_Indicator lanes. */
    sz_u64_t non_ascii_lanes;    /**< Bytes with the high bit set (within `loaded`). */
    sz_u64_t wseg;               /**< WSegSpace raw membership (NOT yet `& start_bytes & ~truncated`). */
    sz_u64_t double_quote_byte;  /**< Raw window byte == U+0022 (within `loaded`). */
    sz_u64_t single_quote_byte;  /**< Raw window byte == U+0027 (within `loaded`). */
    sz_u64_t pictographic;       /**< Extended_Pictographic raw membership (NOT yet `& start_bytes & ~truncated`). */
    sz_u8_t classes_byte[64];    /**< Post-truncation Word_Break class byte per lane (for the carry edge reads). */
} sz_utf8_word_break_frame_t;

/**
 *  @brief  Portable byte-level UAX-29 rule engine: decide a block of @p loaded classified bytes (per-lane masks in
 *          @p frame, the open bridge shadow / RI parity / left context arriving in @p carry) into per-lane word-break
 *          bits, mirroring the serial WB1-WB16 over the WB4 element model. Every WB3-WB16 rule is pure `sz_u64_t` bit
 *          algebra over the precomputed lane masks; the cross-window left context arrives as register-carry seeds at
 *          lane 0, so nothing is scalar re-walked.
 *
 *  The bridge's right context is unbounded; the in-window `bridge` is exact unless an open shadow reaches the block
 *  edge undecided, in which case `resolved` is clamped before that lane so the next, fully-contextual window
 *  re-resolves it. @p carry is updated from the trailing run-state read at the block edge by plain shifts.
 */
SZ_HELPER_INLINE sz_utf8_word_break_window_t sz_utf8_word_break_decide_window_( //
    sz_utf8_word_break_frame_t const *frame, sz_u64_t start_bytes_all, sz_u64_t continuation_all, sz_u64_t forced_other,
    sz_u64_t length_two, sz_u64_t length_three, sz_u64_t length_four, sz_size_t loaded,
    sz_utf8_word_break_carry_t *carry, sz_bool_t more_text) {

    sz_u64_t const valid = sz_u64_mask_until_serial_(loaded);
    sz_u64_t const start_bytes = start_bytes_all & valid;
    sz_u64_t const continuation = continuation_all & valid;
    sz_size_t const high_lane = loaded - 1;
    int const at_tail = !more_text; // no text past this block: a trailing undecided shadow is decided (force-break)

    // Strict U+FFFD substitution for an ill-formed multi-byte lead truncated by the buffer end (the builder has already
    // reclassified these lanes to Other in the class masks; recompute the mask here for the raw-byte WB3c/WB3d gating).
    sz_u64_t const lead_two = length_two & start_bytes;
    sz_u64_t const lead_three = length_three & start_bytes;
    sz_u64_t const lead_four = length_four & start_bytes;
    sz_u64_t const truncated_raw = ((lead_two & ~sz_u64_mask_until_serial_(loaded > 1 ? loaded - 1 : 0)) |
                                    (lead_three & ~sz_u64_mask_until_serial_(loaded > 2 ? loaded - 2 : 0)) |
                                    (lead_four & ~sz_u64_mask_until_serial_(loaded > 3 ? loaded - 3 : 0))) &
                                   valid;
    // Every U+FFFD lane — truncated at the edge OR ill-formed (already forced to Other by the driver) — must be excluded
    // from the raw-byte WB3c (pictograph) and WB3d (WSegSpace) checks, which read the raw membership masks directly.
    sz_u64_t const truncated = truncated_raw | (forced_other & valid);

    sz_u8_t left_property = carry->left_property;

    sz_u64_t const class_aletter = (frame->class_aletter | frame->class_hebrew) & start_bytes;
    sz_u64_t const class_hebrew = frame->class_hebrew & start_bytes;
    sz_u64_t const class_numeric = frame->class_numeric & start_bytes;
    sz_u64_t const class_katakana = frame->class_katakana & start_bytes;
    sz_u64_t const class_extendnumlet = frame->class_extendnumlet & start_bytes;
    sz_u64_t const lead_ignorable = (frame->class_extend | frame->class_zwj | frame->class_format) & start_bytes;

    sz_u64_t const mid_letter_or_quotes = (frame->class_midletter | frame->class_mid_quotes) & start_bytes;
    sz_u64_t const mid_num_or_quotes = (frame->class_midnum | frame->class_mid_quotes) & start_bytes;
    sz_u64_t const mid_quotes = frame->class_mid_quotes & start_bytes;
    sz_u64_t const mid_any = (frame->class_midletter | frame->class_midnum | frame->class_mid_quotes) & start_bytes;
    sz_u64_t const class_cr = frame->class_cr & start_bytes;
    sz_u64_t const class_lf = frame->class_lf & start_bytes;
    sz_u64_t const class_newline = (class_cr | class_lf | (frame->class_newline & start_bytes));
    sz_u64_t const class_zwj = frame->class_zwj & start_bytes;
    sz_u64_t const class_regional = frame->class_regional & start_bytes;

    // WSegSpace lanes (WB3d): raw (high,low) membership, excluding any truncated U+FFFD lead.
    sz_u64_t const wseg = frame->wseg & start_bytes & ~truncated;

    // WB4: Extend/Format/ZWJ are transparent. A "flow" mask lets a class run reach across ignorables (and the CLAIMED
    // continuation bytes of a multi-byte codepoint) so neighbour shifts read the effective adjacent class.
    sz_u64_t const flow_ignorable = continuation | lead_ignorable;
    sz_u64_t const aletter_right_base = sz_u64_fill_right_(class_aletter, flow_ignorable);
    sz_u64_t const numeric_right_base = sz_u64_fill_right_(class_numeric, flow_ignorable);
    sz_u64_t const hebrew_right_base = sz_u64_fill_right_(class_hebrew, flow_ignorable);
    sz_u64_t const aletter_left_base = sz_u64_fill_left_(class_aletter, flow_ignorable);
    sz_u64_t const numeric_left_base = sz_u64_fill_left_(class_numeric, flow_ignorable);
    sz_u64_t const hebrew_left_base = sz_u64_fill_left_(class_hebrew, flow_ignorable);
    // The leading window-edge region runs from lane 0 up to and including the first significant (non-ignorable) lead.
    sz_u64_t const significant_leads = start_bytes & ~lead_ignorable;
    sz_u64_t const edge_region = significant_leads
                                     ? sz_u64_mask_until_serial_((sz_size_t)sz_u64_ctz(significant_leads) + 1)
                                     : valid;

    // Carried bridge resolution: the first significant lead either completes the deferred WB6/7/11/12 bridge (and
    // joins through the plain left-property seeds below), or fails it — the driver then emits one boundary at its
    // anchored mid byte, and the mid becomes the effective left context.
    sz_u8_t deferred_break = 0;
    if (carry->bridge_open && (significant_leads || at_tail)) {
        sz_u8_t const first_class = significant_leads ? frame->classes_byte[(sz_size_t)sz_u64_ctz(significant_leads)]
                                                      : (sz_u8_t)sz_utf8_word_break_other_k;
        int const completes = (carry->bridge_kind == sz_utf8_word_break_bridge_numeric_k &&
                               first_class == sz_utf8_word_break_numeric_k) ||
                              (carry->bridge_kind == sz_utf8_word_break_bridge_aletter_k &&
                               sz_utf8_word_break_is_aletter_or_hebrew_(first_class)) ||
                              (carry->bridge_kind == sz_utf8_word_break_bridge_hebrew_k &&
                               first_class == sz_utf8_word_break_hebrew_letter_k);
        deferred_break = (sz_u8_t)(!completes && carry->bridge_unprotected);
        if (!completes) left_property = carry->bridge_mid_class;
        carry->bridge_open = 0;
        carry->bridge_kind = sz_utf8_word_break_bridge_none_k;
    }
    int const left_is_aletter = left_property == sz_utf8_word_break_aletter_k;
    int const left_is_hebrew = left_property == sz_utf8_word_break_hebrew_letter_k;
    int const left_is_numeric = left_property == sz_utf8_word_break_numeric_k;

    sz_u64_t const seed_aletter_pre = sz_u64_or_if_(0ull, edge_region, left_is_aletter || left_is_hebrew);
    sz_u64_t const seed_numeric_pre = sz_u64_or_if_(0ull, edge_region, left_is_numeric);
    sz_u64_t const seed_hebrew_pre = sz_u64_or_if_(0ull, edge_region, left_is_hebrew);

    // WB6/WB7 and WB11/WB12: a Mid* lane between two letters (or numerics) bridges the run. MidNumLetQ excludes
    // Double_Quote, which bridges ONLY Hebrew x " x Hebrew (WB7b/WB7c) through a Hebrew-only bridge.
    sz_u64_t const double_quote = mid_quotes & frame->double_quote_byte;
    sz_u64_t const mid_letter_quotes_no_double = mid_letter_or_quotes & ~double_quote;
    sz_u64_t const mid_num_quotes_no_double = mid_num_or_quotes & ~double_quote;
    sz_u64_t const mid_open_aletter = mid_letter_quotes_no_double & ((aletter_right_base << 1) | seed_aletter_pre);
    sz_u64_t const mid_open_numeric = mid_num_quotes_no_double & ((numeric_right_base << 1) | seed_numeric_pre);
    sz_u64_t const mid_open_hebrew = double_quote & ((hebrew_right_base << 1) | seed_hebrew_pre);
    sz_u64_t const mid_open = mid_open_aletter | mid_open_numeric | mid_open_hebrew;
    sz_u64_t bridge = (mid_open_aletter & (aletter_left_base >> 1)) | (mid_open_numeric & (numeric_left_base >> 1)) |
                      (mid_open_hebrew & (hebrew_left_base >> 1));

    sz_u64_t const flow = flow_ignorable | bridge;

    sz_u64_t aletter_right, numeric_right, aletter_left, numeric_left;
    if (bridge == 0) {
        aletter_right = aletter_right_base;
        numeric_right = numeric_right_base;
        aletter_left = aletter_left_base;
        numeric_left = numeric_left_base;
    }
    else {
        aletter_right = sz_u64_fill_right_(class_aletter, flow);
        numeric_right = sz_u64_fill_right_(class_numeric, flow);
        aletter_left = sz_u64_fill_left_(class_aletter, flow);
        numeric_left = sz_u64_fill_left_(class_numeric, flow);
    }
    sz_u64_t const katakana_right = sz_u64_fill_right_(class_katakana, flow);
    sz_u64_t const extendnumlet_right = sz_u64_fill_right_(class_extendnumlet, flow);
    sz_u64_t const hebrew_right = sz_u64_fill_right_(class_hebrew, flow);
    sz_u64_t const katakana_left = sz_u64_fill_left_(class_katakana, flow);
    sz_u64_t const extendnumlet_left = sz_u64_fill_left_(class_extendnumlet, flow);

    // WB3/WB3a/WB3b/WB3c: Newline (CR/LF/Newline) forces a break on both sides except CR x LF; ZWJ x suppresses the
    // break after a ZWJ. Multi-byte newlines smear across their OWN continuation bytes (NOT the wider `flow`).
    sz_u64_t previous_newline = (sz_u64_smear_right_(class_newline, continuation, sz_utf8_word_break_smear_steps_k)
                                 << 1);
    sz_u64_t previous_cr = (sz_u64_smear_right_(class_cr, continuation, sz_utf8_word_break_smear_steps_k) << 1);
    sz_u64_t previous_zwj = (sz_u64_smear_right_(class_zwj, continuation, sz_utf8_word_break_smear_steps_k) << 1) |
                            sz_u64_or_if_(0ull, 1ull, carry->prev_ends_in_zwj != 0);
    int const left_is_cr = left_property == sz_utf8_word_break_cr_k;
    int const left_is_newline = left_is_cr || left_property == sz_utf8_word_break_lf_k ||
                                left_property == sz_utf8_word_break_newline_k;
    previous_cr = sz_u64_or_if_(previous_cr, 1ull, left_is_cr);
    previous_newline = sz_u64_or_if_(previous_newline, 1ull, left_is_newline);

    int const leading_mid_bridged = (edge_region & bridge) != 0;
    sz_u64_t const carry_seed_extension = leading_mid_bridged
                                              ? ((sz_u64_fill_right_(edge_region, flow) << 1) & start_bytes)
                                              : 0ull;
    sz_u64_t const carry_seed = edge_region | carry_seed_extension;
    sz_u64_t const seed_aletter = sz_u64_or_if_(0ull, carry_seed, left_is_aletter || left_is_hebrew);
    sz_u64_t const seed_numeric = sz_u64_or_if_(0ull, carry_seed, left_is_numeric);
    sz_u64_t const seed_hebrew = sz_u64_or_if_(0ull, carry_seed, left_is_hebrew);
    sz_u64_t const seed_katakana = sz_u64_or_if_(0ull, carry_seed, left_property == sz_utf8_word_break_katakana_k);
    sz_u64_t const seed_extendnumlet = sz_u64_or_if_(0ull, carry_seed,
                                                     left_property == sz_utf8_word_break_extendnumlet_k);
    sz_u64_t const previous_hebrew = (hebrew_right << 1) | seed_hebrew;
    sz_u64_t const previous_aletter = (aletter_right << 1) | seed_aletter;
    sz_u64_t const previous_numeric = (numeric_right << 1) | seed_numeric;
    sz_u64_t const previous_katakana = (katakana_right << 1) | seed_katakana;
    sz_u64_t const previous_extendnumlet = (extendnumlet_right << 1) | seed_extendnumlet;
    sz_u64_t const next_aletter = aletter_left;
    sz_u64_t const next_numeric = numeric_left;
    sz_u64_t const next_katakana = katakana_left;
    sz_u64_t const next_extendnumlet = extendnumlet_left;

    sz_u64_t join = (previous_aletter & next_aletter) | (previous_numeric & next_numeric) | // WB5, WB8
                    (previous_aletter & next_numeric) | (previous_numeric & next_aletter);  // WB9, WB10
    join |= previous_katakana & next_katakana;                                              // WB13
    join |= (previous_aletter | previous_numeric | previous_katakana | previous_extendnumlet) &
            next_extendnumlet;                                                     // WB13a
    join |= previous_extendnumlet & (next_aletter | next_numeric | next_katakana); // WB13b
    join |= previous_cr & class_lf;                                                // WB3  CR x LF

    // WB3c: ZWJ x Extended_Pictographic, gated behind an in-window ZWJ (or a carried ZWJ).
    if (class_zwj || carry->prev_ends_in_zwj) {
        sz_u64_t const pictographic = frame->pictographic & start_bytes & ~truncated;
        join |= previous_zwj & pictographic; // WB3c
    }

    // WB3d: WSegSpace x WSegSpace, raw adjacency over the space's OWN continuation bytes only.
    sz_u64_t const previous_wseg = (sz_u64_smear_right_(wseg, continuation, sz_utf8_word_break_smear_steps_k) << 1) |
                                   sz_u64_or_if_(0ull, 1ull, carry->prev_is_wseg != 0);
    join |= previous_wseg & wseg;

    // WB15/WB16: Regional_Indicator pairs, resolved by the segmented parity scan.
    sz_u64_t ri_inclusive = 0;
    int const lane0_continues_ri = carry->left_property == sz_utf8_word_break_regional_ind_k &&
                                   ((class_regional | flow_ignorable) & 1ull) != 0;
    sz_u64_t const ri_run_gate = sz_u64_or_if_(sz_u64_fill_right_(class_regional, flow_ignorable) |
                                                   sz_u64_fill_left_(class_regional, flow_ignorable) | class_regional,
                                               1ull, lane0_continues_ri);
    sz_u64_t const ri_join = sz_utf8_word_break_ri_join_(
        class_regional, ri_run_gate, (sz_u8_t)(lane0_continues_ri ? carry->ri_parity : 0), &ri_inclusive);
    join |= ri_join;

    // WB3a/WB3b force a break before and after every Newline/CR/LF lane, overriding any join (CR x LF kept).
    sz_u64_t const force_break = ((previous_newline | class_newline) & ~(previous_cr & class_lf)) & start_bytes;

    sz_u64_t const start_bytes_no_bridge = start_bytes & ~bridge;
    sz_u64_t boundary = (~join) & start_bytes_no_bridge;
    // WB7a is Hebrew_Letter x Single_Quote (U+0027) only.
    sz_u64_t const single_quote = mid_quotes & frame->single_quote_byte;
    boundary &= ~(single_quote & previous_hebrew);   // WB7a: Hebrew_Letter x Single_Quote
    boundary &= ~lead_ignorable;                     // WB4: never break before an Extend/Format/ZWJ lane
    boundary |= force_break & start_bytes_no_bridge; // WB3a/WB3b break around Newline/CR/LF

    // Bridge shadow (the deferred WB6/7/11/12 right context).
    sz_u64_t const reach_gate = flow_ignorable;
    sz_u64_t const back_gate = flow_ignorable | mid_any;
    sz_u64_t const top_bit = at_tail ? 0ull : (1ull << high_lane);
    sz_u64_t const open_to_edge = sz_u64_fill_right_(mid_open, reach_gate) & top_bit;
    sz_u64_t const undecided = (!at_tail && open_to_edge) ? (mid_open & sz_u64_fill_left_(open_to_edge, back_gate))
                                                          : 0ull;

    sz_u64_t const lowbit = carry->have_prev ? 1ull : 0ull;
    sz_u64_t const produced = start_bytes & (~1ull | lowbit);
    boundary &= produced;
    sz_size_t resolved = loaded;
    if (undecided) {
        sz_size_t const lane = (sz_size_t)sz_u64_ctz(undecided);
        resolved = lane;
        boundary &= sz_u64_mask_until_serial_(lane);
    }

    // Update the carry from the block's high edge (resolved lane and below).
    sz_u64_t const resolved_valid = sz_u64_mask_until_serial_(resolved);
    sz_u64_t const resolved_starts = start_bytes & resolved_valid;
    sz_u64_t const resolved_significant = resolved_starts & ~lead_ignorable;
    sz_u8_t left_out;
    if (resolved_significant == 0) {
        int const carry_is_newline = left_property == sz_utf8_word_break_cr_k ||
                                     left_property == sz_utf8_word_break_lf_k ||
                                     left_property == sz_utf8_word_break_newline_k;
        sz_u64_t const ignorable_here = lead_ignorable & resolved_valid;
        if ((carry_is_newline || carry->have_prev == 0) && ignorable_here)
            left_out = frame->classes_byte[(sz_size_t)sz_u64_ctz(ignorable_here)];
        else left_out = left_property;
    }
    else {
        sz_size_t const last_lane = (sz_size_t)(63 - sz_u64_clz(resolved_significant));
        sz_u8_t base_class = frame->classes_byte[last_lane];
        int const base_is_newline = base_class == sz_utf8_word_break_cr_k || base_class == sz_utf8_word_break_lf_k ||
                                    base_class == sz_utf8_word_break_newline_k;
        sz_u64_t const ignorable_after = lead_ignorable & resolved_valid & ~sz_u64_mask_until_serial_(last_lane + 1);
        if (base_is_newline && ignorable_after)
            base_class = frame->classes_byte[(sz_size_t)sz_u64_ctz(ignorable_after)];
        left_out = base_class;
    }
    carry->left_property = left_out;
    carry->have_prev = 1;

    // Outbound RI parity.
    if (resolved_significant) {
        sz_size_t const edge_lane = (sz_size_t)(63 - sz_u64_clz(resolved_significant));
        sz_u8_t const top_is_regional = (sz_u8_t)((class_regional >> edge_lane) & 1ull);
        sz_u8_t const edge_parity = (sz_u8_t)((ri_inclusive >> edge_lane) & 1ull);
        carry->ri_parity = top_is_regional ? edge_parity : 0;
    }

    // Outbound WB3d / WB3c raw-adjacency.
    if (resolved_starts) {
        sz_size_t const edge_start = (sz_size_t)(63 - sz_u64_clz(resolved_starts));
        carry->prev_is_wseg = (sz_u8_t)((wseg >> edge_start) & 1ull);
        carry->prev_ends_in_zwj = (sz_u8_t)((class_zwj >> edge_start) & 1ull);
    }
    else {
        carry->prev_is_wseg = 0;
        carry->prev_ends_in_zwj = 0;
    }

    // Outbound bridge shadow: an unresolved Mid* at lane 0 (`resolved == 0`) is about to be consumed by the
    // driver's forced advance; carry its kind, class, and WB7a protection until a significant lead resolves it.
    if (undecided & 1ull) {
        carry->bridge_open = 1;
        carry->bridge_kind = (sz_u8_t)((mid_open_hebrew & 1ull)    ? sz_utf8_word_break_bridge_hebrew_k
                                       : (mid_open_numeric & 1ull) ? sz_utf8_word_break_bridge_numeric_k
                                                                   : sz_utf8_word_break_bridge_aletter_k);
        carry->bridge_mid_class = frame->classes_byte[0];
        carry->bridge_unprotected = (sz_u8_t)(~(single_quote & previous_hebrew) & 1ull);
    }

    sz_utf8_word_break_window_t result;
    result.breaks = boundary;
    result.resolved = resolved;
    result.deferred_break = deferred_break;
    return result;
}

#pragma endregion Portable Word_Break Block Rule Engine

#pragma endregion // UAX 29 Word Boundaries

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDBREAKS_SERIAL_H_
