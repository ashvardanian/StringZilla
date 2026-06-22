/**
 *  @brief Serial backend for UAX-29 word boundaries.
 *  @file include/stringzilla/utf8_words/serial.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_WORDS_SERIAL_H_
#define STRINGZILLA_UTF8_WORDS_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_words/tables.h"
#include "stringzilla/utf8_codepoints/serial.h" // shared decode helpers
#include "stringzilla/utf8_runes.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region UAX-29 Word Boundaries

/**
 *  @brief Returns the UAX-29 Word_Break property (0-15) for a codepoint.
 *
 *  Mirrors the gather-free Word_Break classifier: arithmetic big ranges first, then a flat low-plane LUT
 *  for codepoint < 0x800, then a B=8 / SB=16 trie over the rest of the BMP, then a sorted astral range list,
 *  defaulting to Other for everything else.
 */
SZ_PUBLIC sz_u8_t sz_rune_word_break_property(sz_rune_t rune) {
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
SZ_PUBLIC sz_bool_t sz_rune_is_word_char(sz_rune_t rune) {
    sz_u8_t property = sz_rune_word_break_property(rune);
    // Word characters: ALetter(8), Hebrew_Letter(9), Numeric(10), Katakana(11),
    // ExtendNumLet(12), MidLetter(13), MidNum(14), MidNumLet/Quotes(15)
    return (sz_bool_t)(property >= sz_utf8_word_break_aletter_k);
}

/**
 *  @brief Check if @p rune is Extended_Pictographic (UAX-29 WB3c). Not part of the 4-bit Word_Break model, so it is
 *         resolved by binary search over the sorted Extended_Pictographic range table.
 */
SZ_PUBLIC sz_bool_t sz_rune_is_extended_pictographic(sz_rune_t rune) {
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
SZ_PUBLIC sz_bool_t sz_rune_is_wsegspace(sz_rune_t rune) {
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
 *  From per-lane class masks, returns a mask whose bit `i` is set when the boundary before lane `i` is
 *  suppressed by a no-break rule. ASCII never triggers WB4 or WB15/16, so the rules reduce to neighbour shifts;
 *  the result is exact only for lanes whose i-2 and i+1 neighbours are in-window.
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
 *  @return The property of the previous significant character, or sz_utf8_word_break_other_k if none.
 */
SZ_INTERNAL sz_u8_t sz_utf8_word_break_previous_property_(sz_cptr_t text, sz_size_t position) {
    if (position == 0) return sz_utf8_word_break_other_k;

    // Scan backward to find start of previous codepoint
    sz_size_t previous = position - 1;
    while (previous > 0 && ((sz_u8_t)text[previous] & 0xC0) == 0x80) previous--;

    // Decode and get property
    sz_size_t decode_position = previous;
    sz_rune_t rune = sz_utf8_decode_(text, position, &decode_position);
    sz_u8_t property = sz_rune_word_break_property(rune);

    // Skip back over ignorables (WB4). EXCEPTION: an ignorable immediately after a Newline/CR/LF — or at sot — is
    // "de-ignored" and acts as its own base, so stop and return ITS property rather than the newline behind it
    // (otherwise WB3/WB3a would wrongly fire on the codepoint after the de-ignored ignorable).
    while (sz_utf8_word_break_is_ignorable_(property)) {
        if (previous == 0) break;
        sz_size_t predecessor = previous - 1;
        while (predecessor > 0 && ((sz_u8_t)text[predecessor] & 0xC0) == 0x80) predecessor--;
        sz_size_t predecessor_decode = predecessor;
        sz_rune_t predecessor_rune = sz_utf8_decode_(text, previous, &predecessor_decode);
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
    sz_u8_t property;      /**< Word_Break property of the base codepoint. */
    sz_rune_t codepoint;   /**< Base codepoint (disambiguates Single/Double_Quote, Extended_Pictographic). */
    sz_size_t start;       /**< Byte offset of the base codepoint. */
    sz_bool_t ends_in_zwj; /**< The element's final codepoint is U+200D ZERO WIDTH JOINER. */
    sz_bool_t valid;       /**< False for the sot/eot sentinel. */
} sz_word_element_t;

/** @brief True for the newline family CR/LF/Newline, which neither absorb (WB4) nor are absorbed. */
SZ_INTERNAL sz_bool_t sz_word_is_newline_(sz_u8_t property) {
    return (sz_bool_t)(property == sz_utf8_word_break_cr_k || property == sz_utf8_word_break_lf_k ||
                       property == sz_utf8_word_break_newline_k);
}

/*  The 4-bit model lumps Single_Quote (U+0027), Double_Quote (U+0022), and MidNumLet into MID_QUOTES, so the
 *  WB6/WB7/WB7a-c/WB11/WB12 distinctions are recovered from the codepoint. MidNumLetQ = MidNumLet + Single_Quote,
 *  i.e. every MID_QUOTES codepoint that is NOT the Double_Quote. */
SZ_INTERNAL sz_bool_t sz_word_is_single_quote_(sz_u8_t property, sz_rune_t codepoint) {
    return (sz_bool_t)(property == sz_utf8_word_break_mid_quotes_k && codepoint == 0x0027u);
}
SZ_INTERNAL sz_bool_t sz_word_is_double_quote_(sz_u8_t property, sz_rune_t codepoint) {
    return (sz_bool_t)(property == sz_utf8_word_break_mid_quotes_k && codepoint == 0x0022u);
}
SZ_INTERNAL sz_bool_t sz_word_is_mid_num_let_q_(sz_u8_t property, sz_rune_t codepoint) {
    return (sz_bool_t)(property == sz_utf8_word_break_mid_quotes_k && codepoint != 0x0022u);
}

/** @brief Byte offset of the codepoint start immediately before @p position (which must be > 0), using the canonical
 *         maximal-subpart partition so it agrees with the forward `sz_utf8_decode_` on malformed input: a run of
 *         continuation bytes is absorbed into the nearest preceding lead ONLY if that lead's sequence actually reaches
 *         @p position; otherwise the byte just before @p position is a stray continuation (its own U+FFFD). A naive
 *         continuation-skip would absorb a stray into the previous codepoint and SKIP it during the backward look-back,
 *         crossing a streamed resume point and forging a WB5/6/7 join the forward pass never sees - that asymmetry is
 *         the capacity-dependence. Treating every stray as its own non-skippable U+FFFD makes resume reproduce. */
SZ_INTERNAL sz_size_t sz_word_previous_start_(sz_cptr_t text, sz_size_t position) {
    sz_size_t lead = position - 1;
    while (lead > 0 && ((sz_u8_t)text[lead] & 0xC0) == 0x80) lead--;
    sz_size_t reach = lead;
    sz_utf8_decode_(text, position, &reach);
    return reach == position ? lead : position - 1;
}

/** @brief The element whose final codepoint ends just before @p position (sot sentinel when @p position == 0). */
SZ_INTERNAL sz_word_element_t sz_word_previous_element_(sz_cptr_t text, sz_size_t position) {
    sz_word_element_t element;
    element.valid = sz_false_k;
    if (position == 0) return element;

    sz_size_t last_start = sz_word_previous_start_(text, position);
    sz_size_t decode_at = last_start;
    sz_rune_t last_codepoint = sz_utf8_decode_(text, position, &decode_at);
    sz_u8_t last_property = sz_rune_word_break_property(last_codepoint);
    element.ends_in_zwj = (sz_bool_t)(last_property == sz_utf8_word_break_zwj_k);

    sz_size_t base_start = last_start;
    sz_rune_t base_codepoint = last_codepoint;
    sz_u8_t base_property = last_property;
    while (sz_utf8_word_break_is_ignorable_(base_property) && base_start > 0) {
        sz_size_t predecessor_start = sz_word_previous_start_(text, base_start);
        sz_size_t predecessor_at = predecessor_start;
        sz_rune_t predecessor_codepoint = sz_utf8_decode_(text, base_start, &predecessor_at);
        /*  WB4 exception: an ignorable directly after a Newline/CR/LF is de-ignored and is its own base. */
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
SZ_INTERNAL sz_word_element_t sz_word_next_element_(sz_cptr_t text, sz_size_t length, sz_size_t position) {
    sz_word_element_t element;
    element.valid = sz_false_k;
    sz_size_t cursor = position;
    sz_rune_t base_codepoint = sz_utf8_decode_(text, length, &cursor);
    if (!sz_word_is_newline_(sz_rune_word_break_property(base_codepoint))) {
        while (cursor < length) {
            sz_size_t probe = cursor;
            sz_rune_t codepoint = sz_utf8_decode_(text, length, &probe);
            if (!sz_utf8_word_break_is_ignorable_(sz_rune_word_break_property(codepoint))) break;
            cursor = probe;
        }
    }
    if (cursor >= length) return element;
    sz_size_t decode_at = cursor;
    sz_rune_t next_codepoint = sz_utf8_decode_(text, length, &decode_at);
    element.property = sz_rune_word_break_property(next_codepoint);
    element.codepoint = next_codepoint;
    element.start = cursor;
    element.ends_in_zwj = sz_false_k;
    element.valid = sz_true_k;
    return element;
}

/** @brief Count of contiguous Regional_Indicator elements immediately before @p position (for WB15/WB16 parity). */
SZ_INTERNAL sz_size_t sz_word_regional_run_before_(sz_cptr_t text, sz_size_t position) {
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
SZ_PUBLIC sz_bool_t sz_utf8_is_word_boundary_serial(sz_cptr_t text, sz_size_t length, sz_size_t position) {
    if (position == 0) return sz_true_k;      /* WB1 */
    if (position >= length) return sz_true_k; /* WB2 */
    /*  Never split INSIDE a codepoint - but only a continuation byte genuinely covered by a preceding lead's
     *  maximal-subpart sequence is interior. A stray continuation (the lead stops short of it) is its OWN U+FFFD
     *  codepoint, so a boundary may fall before it; matching the canonical forward decode keeps streamed segmentation
     *  capacity-independent on malformed input. */
    if (((sz_u8_t)text[position] & 0xC0) == 0x80) {
        sz_size_t lead = position;
        while (lead > 0 && ((sz_u8_t)text[lead] & 0xC0) == 0x80) lead--;
        sz_size_t reach = lead;
        sz_utf8_decode_(text, length, &reach);
        if (reach > position) return sz_false_k; /* interior byte of a valid multi-byte codepoint */
    }

    sz_size_t after_at = position;
    sz_rune_t next_codepoint = sz_utf8_decode_(text, length, &after_at);
    sz_u8_t next_property = sz_rune_word_break_property(next_codepoint);

    sz_size_t immediate_start = sz_word_previous_start_(text, position);
    sz_size_t immediate_at = immediate_start;
    sz_rune_t immediate_codepoint = sz_utf8_decode_(text, position, &immediate_at);
    sz_u8_t immediate_property = sz_rune_word_break_property(immediate_codepoint);

    if (immediate_property == sz_utf8_word_break_cr_k && next_property == sz_utf8_word_break_lf_k)
        return sz_false_k;                                                  /* WB3  CR x LF */
    if (sz_word_is_newline_(immediate_property)) return sz_true_k;          /* WB3a break after Newline/CR/LF */
    if (sz_word_is_newline_(next_property)) return sz_true_k;               /* WB3b break before Newline/CR/LF */
    if (sz_utf8_word_break_is_ignorable_(next_property)) return sz_false_k; /* WB4 absorb Extend/Format/ZWJ */

    sz_word_element_t previous = sz_word_previous_element_(text, position);
    sz_u8_t previous_property = previous.property;
    sz_rune_t previous_codepoint = previous.codepoint;

    if (previous.ends_in_zwj && sz_rune_is_extended_pictographic(next_codepoint))
        return sz_false_k; /* WB3c ZWJ x Extended_Pictographic */
    /*  WB3c/WB3d precede WB4, so they test RAW adjacency: WB3d needs the immediate predecessor codepoint (an
     *  intervening Extend, absorbed by WB4, must still force the break) rather than the folded element base. */
    if (sz_rune_is_wsegspace(immediate_codepoint) && sz_rune_is_wsegspace(next_codepoint))
        return sz_false_k; /* WB3d WSegSpace x WSegSpace */

    sz_bool_t previous_ah = sz_utf8_word_break_is_aletter_or_hebrew_(previous_property);
    sz_bool_t next_ah = sz_utf8_word_break_is_aletter_or_hebrew_(next_property);

    if (previous_ah && next_ah) return sz_false_k; /* WB5 */
    if (previous_ah &&
        (next_property == sz_utf8_word_break_midletter_k || sz_word_is_mid_num_let_q_(next_property, next_codepoint))) {
        sz_word_element_t after = sz_word_next_element_(text, length, position);
        if (after.valid && sz_utf8_word_break_is_aletter_or_hebrew_(after.property)) return sz_false_k; /* WB6 */
    }
    if ((previous_property == sz_utf8_word_break_midletter_k ||
         sz_word_is_mid_num_let_q_(previous_property, previous_codepoint)) &&
        next_ah) {
        sz_word_element_t before = sz_word_previous_element_(text, previous.start);
        if (before.valid && sz_utf8_word_break_is_aletter_or_hebrew_(before.property)) return sz_false_k; /* WB7 */
    }
    if (previous_property == sz_utf8_word_break_hebrew_letter_k &&
        sz_word_is_single_quote_(next_property, next_codepoint))
        return sz_false_k; /* WB7a Hebrew x Single_Quote */
    if (previous_property == sz_utf8_word_break_hebrew_letter_k &&
        sz_word_is_double_quote_(next_property, next_codepoint)) {
        sz_word_element_t after = sz_word_next_element_(text, length, position);
        if (after.valid && after.property == sz_utf8_word_break_hebrew_letter_k) return sz_false_k; /* WB7b */
    }
    if (sz_word_is_double_quote_(previous_property, previous_codepoint) &&
        next_property == sz_utf8_word_break_hebrew_letter_k) {
        sz_word_element_t before = sz_word_previous_element_(text, previous.start);
        if (before.valid && before.property == sz_utf8_word_break_hebrew_letter_k) return sz_false_k; /* WB7c */
    }
    if (previous_property == sz_utf8_word_break_numeric_k && next_property == sz_utf8_word_break_numeric_k)
        return sz_false_k;                                                               /* WB8 */
    if (previous_ah && next_property == sz_utf8_word_break_numeric_k) return sz_false_k; /* WB9 */
    if (previous_property == sz_utf8_word_break_numeric_k && next_ah) return sz_false_k; /* WB10 */
    if ((previous_property == sz_utf8_word_break_midnum_k ||
         sz_word_is_mid_num_let_q_(previous_property, previous_codepoint)) &&
        next_property == sz_utf8_word_break_numeric_k) {
        sz_word_element_t before = sz_word_previous_element_(text, previous.start);
        if (before.valid && before.property == sz_utf8_word_break_numeric_k) return sz_false_k; /* WB11 */
    }
    if (previous_property == sz_utf8_word_break_numeric_k &&
        (next_property == sz_utf8_word_break_midnum_k || sz_word_is_mid_num_let_q_(next_property, next_codepoint))) {
        sz_word_element_t after = sz_word_next_element_(text, length, position);
        if (after.valid && after.property == sz_utf8_word_break_numeric_k) return sz_false_k; /* WB12 */
    }
    if (previous_property == sz_utf8_word_break_katakana_k && next_property == sz_utf8_word_break_katakana_k)
        return sz_false_k; /* WB13 */
    if ((previous_ah || previous_property == sz_utf8_word_break_numeric_k ||
         previous_property == sz_utf8_word_break_katakana_k ||
         previous_property == sz_utf8_word_break_extendnumlet_k) &&
        next_property == sz_utf8_word_break_extendnumlet_k)
        return sz_false_k; /* WB13a */
    if (previous_property == sz_utf8_word_break_extendnumlet_k &&
        (next_ah || next_property == sz_utf8_word_break_numeric_k || next_property == sz_utf8_word_break_katakana_k))
        return sz_false_k; /* WB13b */
    if (previous_property == sz_utf8_word_break_regional_ind_k && next_property == sz_utf8_word_break_regional_ind_k) {
        if (sz_word_regional_run_before_(text, previous.start) % 2 == 0) return sz_false_k; /* WB15/WB16 */
    }
    return sz_true_k; /* WB999 */
}

/*  Plural UAX-29 word segmentation: one left-to-right sweep emits every word into parallel
 *  `word_starts` / `word_lengths` arrays, at exactly the positions where `sz_utf8_is_word_boundary_serial` holds.
 *  On a full buffer `*bytes_consumed` is the start of the first word that did not fit - always a true TR29
 *  boundary - so a caller resumes from `text + *bytes_consumed` and obtains the identical remainder.
 */
SZ_PUBLIC sz_size_t sz_utf8_words_serial(            //
    sz_cptr_t text, sz_size_t length,                //
    sz_size_t *word_starts, sz_size_t *word_lengths, //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_size_t word_start = 0; // Start of the word currently being accumulated (always a boundary).
    // Position 0 is always a boundary, so the first reportable interior boundary is after the first codepoint. Step by
    // the continuation-bit partition (next non-continuation byte) - the SAME partition the boundary predicate and its
    // backward look-back use - so segmentation is identical regardless of where a streamed call resumes. A
    // declared-length step would land off that grid on a truncated/stray sequence and make the resume point
    // capacity-dependent on malformed input.
    sz_size_t position = 0;
    sz_utf8_decode_(text, length, &position);

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
        sz_utf8_decode_(text, length, &position);
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

#pragma endregion // UAX-29 Word Boundaries

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDS_SERIAL_H_
