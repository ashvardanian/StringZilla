/**
 *  @brief Serial backend for UAX-29 sentence boundaries.
 *  @file include/stringzilla/utf8_sentences/serial.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_SENTENCES_SERIAL_H_
#define STRINGZILLA_UTF8_SENTENCES_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_sentences/tables.h"
#include "stringzilla/utf8_codepoints/serial.h" // shared decode helpers
#include "stringzilla/utf8_runes.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region UAX-29 Sentence Boundaries

/** @brief Returns the UAX-29 Sentence_Break property (0-14) for a codepoint. */
SZ_PUBLIC sz_u8_t sz_rune_sentence_break_property(sz_rune_t rune) {
    for (sz_size_t range = 0; range < sz_utf8_sentence_break_big_oletter_count_k; ++range)
        if (rune >= sz_utf8_sentence_break_big_oletter_lo_[range] &&
            rune <= sz_utf8_sentence_break_big_oletter_hi_[range])
            return sz_sentence_break_oletter_k;

    if (rune < 0x800u) return sz_utf8_sentence_break_flat_lut_0800_[rune];

    if (rune < 0x10000u) {
        sz_u32_t const offset = (sz_u32_t)(rune - 0x800u);
        sz_u32_t const block = offset / sz_utf8_sentence_break_trie_block_k;
        sz_u32_t const within = offset % sz_utf8_sentence_break_trie_block_k;
        sz_u32_t const super = block / sz_utf8_sentence_break_trie_subblock_k;
        sz_u32_t const super_offset = block % sz_utf8_sentence_break_trie_subblock_k;
        sz_u8_t const level1 = sz_utf8_sentence_break_trie_l1_[super];
        sz_u16_t const leaf =
            sz_utf8_sentence_break_trie_l2_[level1 * sz_utf8_sentence_break_trie_subblock_k + super_offset];
        return sz_utf8_sentence_break_trie_leaf_[leaf * sz_utf8_sentence_break_trie_block_k + within];
    }

    for (sz_size_t range = 0; range < sz_utf8_sentence_break_astral_count_k; ++range)
        if (rune >= sz_utf8_sentence_break_astral_lo_[range] && rune <= sz_utf8_sentence_break_astral_hi_[range])
            return sz_utf8_sentence_break_astral_cls_[range];

    return sz_sentence_break_other_k;
}

/** @brief True for a Sentence_Break ParaSep (Sep, CR, or LF). */
SZ_INTERNAL sz_bool_t sz_sentence_break_is_parasep_(sz_u8_t property) {
    return (sz_bool_t)(property == sz_sentence_break_sep_k || property == sz_sentence_break_cr_k ||
                       property == sz_sentence_break_lf_k);
}
/** @brief True for a Sentence_Break SATerm (STerm or ATerm). */
SZ_INTERNAL sz_bool_t sz_sentence_break_is_saterm_(sz_u8_t property) {
    return (sz_bool_t)(property == sz_sentence_break_sterm_k || property == sz_sentence_break_aterm_k);
}
/** @brief True for an SB5-transparent character (Extend or Format). */
SZ_INTERNAL sz_bool_t sz_sentence_break_is_transparent_(sz_u8_t property) {
    return (sz_bool_t)(property == sz_sentence_break_extend_k || property == sz_sentence_break_format_k);
}

/**
 *  @brief Start offset of the codepoint after @p position: the next non-continuation byte, or @p length.
 *         Mirrors the SIMD `sz_utf8_codepoints_decode_window_` codepoint-start convention so the serial and
 *         Ice Lake backends segment malformed input identically (UAX-29 leaves ill-formed bytes undefined).
 */
SZ_INTERNAL sz_size_t sz_sentence_break_next_start_(sz_cptr_t text, sz_size_t length, sz_size_t position) {
    sz_size_t next = position + 1;
    while (next < length && ((sz_u8_t)text[next] & 0xC0) == 0x80) ++next;
    return next;
}

/**
 *  @brief Sentence_Break property of the codepoint starting at `start`, decoded BLINDLY to mirror the SIMD
 *         `sz_utf8_codepoints_decode_window_`: the lead's strict length class (2-byte `110xxxxx`, 3-byte
 *         `1110xxxx`, 4-byte `11110xxx`; everything else single-byte) selects how many following bytes are
 *         folded in with no continuation/overlong/surrogate validation, missing trailing bytes read as zero.
 *         Valid UTF-8 decodes identically to the checked path; only ill-formed input differs, by design.
 */
SZ_INTERNAL sz_u8_t sz_sentence_break_property_at_(sz_cptr_t text, sz_size_t length, sz_size_t start) {
    sz_u8_t const lead = (sz_u8_t)text[start];
    int const lead_length = ((lead & 0xE0u) == 0xC0u)   ? 2
                            : ((lead & 0xF0u) == 0xE0u) ? 3
                            : ((lead & 0xF8u) == 0xF0u) ? 4
                                                        : 1;
    sz_u8_t const byte1 = (start + 1 < length) ? (sz_u8_t)text[start + 1] : 0;
    sz_u8_t const byte2 = (start + 2 < length) ? (sz_u8_t)text[start + 2] : 0;
    sz_u8_t const byte3 = (start + 3 < length) ? (sz_u8_t)text[start + 3] : 0;
    sz_rune_t rune;
    switch (lead_length) {
    case 2: rune = ((sz_rune_t)(lead & 0x1Fu) << 6) | (byte1 & 0x3Fu); break;
    case 3: rune = ((sz_rune_t)(lead & 0x0Fu) << 12) | ((sz_rune_t)(byte1 & 0x3Fu) << 6) | (byte2 & 0x3Fu); break;
    case 4:
        rune = ((sz_rune_t)(lead & 0x07u) << 18) | ((sz_rune_t)(byte1 & 0x3Fu) << 12) |
               ((sz_rune_t)(byte2 & 0x3Fu) << 6) | (byte3 & 0x3Fu);
        break;
    default: rune = lead; break;
    }
    return sz_rune_sentence_break_property(rune);
}

/**
 *  @brief Walk to the start of the nearest non-(Extend|Format) codepoint at or before `start`, or return the
 *         leftmost codepoint start when the whole prefix is transparent. `*found` reports whether a
 *         significant codepoint was reached.
 */
SZ_INTERNAL sz_size_t sz_sentence_break_prev_significant_(sz_cptr_t text, sz_size_t length, sz_size_t start,
                                                          sz_bool_t *found) {
    sz_size_t position = start;
    for (;;) {
        if (!sz_sentence_break_is_transparent_(sz_sentence_break_property_at_(text, length, position))) {
            *found = sz_true_k;
            return position;
        }
        if (position == 0) {
            *found = sz_false_k;
            return position;
        }
        position = sz_utf8_previous_codepoint_start_(text, position);
    }
}

/**
 *  @brief Check if `position` is a sentence boundary per Unicode TR29 (SB1-SB998). Forward-only.
 */
SZ_PUBLIC sz_bool_t sz_utf8_is_sentence_boundary_serial(sz_cptr_t text, sz_size_t length, sz_size_t position) {
    if (position == 0) return sz_true_k;      // SB1
    if (position >= length) return sz_true_k; // SB2
    if (((sz_u8_t)text[position] & 0xC0) == 0x80) return sz_false_k;

    sz_size_t before_start = sz_utf8_previous_codepoint_start_(text, position);
    sz_u8_t before = sz_sentence_break_property_at_(text, length, before_start);
    sz_u8_t after = sz_sentence_break_property_at_(text, length, position);

    if (before == sz_sentence_break_cr_k && after == sz_sentence_break_lf_k) return sz_false_k;        // SB3
    if (sz_sentence_break_is_parasep_(before)) return sz_true_k;                                       // SB4
    if (after == sz_sentence_break_extend_k || after == sz_sentence_break_format_k) return sz_false_k; // SB5

    /* Effective preceding class with Extend/Format ignored. */
    sz_bool_t has_significant = sz_false_k;
    sz_size_t eff_before_start = sz_sentence_break_prev_significant_(text, length, before_start, &has_significant);
    sz_u8_t eff_before = has_significant ? sz_sentence_break_property_at_(text, length, eff_before_start)
                                         : (sz_u8_t)sz_sentence_break_other_k;

    if (eff_before == sz_sentence_break_aterm_k && after == sz_sentence_break_numeric_k) return sz_false_k; // SB6

    if (eff_before == sz_sentence_break_aterm_k && after == sz_sentence_break_upper_k && has_significant &&
        eff_before_start > 0) { // SB7
        sz_bool_t found2 = sz_false_k;
        sz_size_t before2 = sz_sentence_break_prev_significant_(
            text, length, sz_utf8_previous_codepoint_start_(text, eff_before_start), &found2);
        sz_u8_t pp = found2 ? sz_sentence_break_property_at_(text, length, before2)
                            : (sz_u8_t)sz_sentence_break_other_k;
        if (pp == sz_sentence_break_upper_k || pp == sz_sentence_break_lower_k) return sz_false_k;
    }

    /* Trailing SATerm Close* Sp* context ending at `before_start`, Extend/Format transparent. */
    {
        sz_bool_t walk_found = sz_false_k;
        sz_size_t walk = sz_sentence_break_prev_significant_(text, length, before_start, &walk_found);
        sz_bool_t saw_sp = sz_false_k;
        while (walk_found && sz_sentence_break_property_at_(text, length, walk) == sz_sentence_break_sp_k) {
            saw_sp = sz_true_k;
            if (walk == 0) {
                walk_found = sz_false_k;
                break;
            }
            walk = sz_sentence_break_prev_significant_(text, length, sz_utf8_previous_codepoint_start_(text, walk),
                                                       &walk_found);
        }
        while (walk_found && sz_sentence_break_property_at_(text, length, walk) == sz_sentence_break_close_k) {
            if (walk == 0) {
                walk_found = sz_false_k;
                break;
            }
            walk = sz_sentence_break_prev_significant_(text, length, sz_utf8_previous_codepoint_start_(text, walk),
                                                       &walk_found);
        }
        sz_u8_t term = walk_found ? sz_sentence_break_property_at_(text, length, walk)
                                  : (sz_u8_t)sz_sentence_break_other_k;
        sz_bool_t has_saterm = sz_sentence_break_is_saterm_(term);
        sz_bool_t has_aterm = (sz_bool_t)(term == sz_sentence_break_aterm_k);

        if (has_saterm || has_aterm) {
            if (has_aterm) { // SB8: ATerm Close* Sp* x (not listed)* Lower
                sz_size_t scan = position;
                sz_bool_t found_lower = sz_false_k;
                while (scan < length) {
                    sz_u8_t scan_prop = sz_sentence_break_property_at_(text, length, scan);
                    if (sz_sentence_break_is_transparent_(scan_prop)) {
                        scan = sz_sentence_break_next_start_(text, length, scan);
                        continue;
                    }
                    if (scan_prop == sz_sentence_break_lower_k) {
                        found_lower = sz_true_k;
                        break;
                    }
                    if (scan_prop == sz_sentence_break_oletter_k || scan_prop == sz_sentence_break_upper_k ||
                        sz_sentence_break_is_parasep_(scan_prop) || sz_sentence_break_is_saterm_(scan_prop))
                        break;
                    scan = sz_sentence_break_next_start_(text, length, scan);
                }
                if (found_lower) return sz_false_k;
            }
            if (after == sz_sentence_break_scontinue_k || sz_sentence_break_is_saterm_(after))
                return sz_false_k; // SB8a
            if (!saw_sp && (after == sz_sentence_break_close_k || after == sz_sentence_break_sp_k ||
                            sz_sentence_break_is_parasep_(after)))
                return sz_false_k;                                                                          // SB9
            if (after == sz_sentence_break_sp_k || sz_sentence_break_is_parasep_(after)) return sz_false_k; // SB10
            return sz_true_k;                                                                               // SB11
        }
    }

    return sz_false_k; // SB998
}

/**
 *  @brief Plural UAX-29 sentence segmentation: one forward sweep emits every sentence into parallel
 *         `sentence_starts` / `sentence_lengths` arrays. Forward-only (no reverse counterpart).
 */
SZ_PUBLIC sz_size_t sz_utf8_sentences_serial(                //
    sz_cptr_t text, sz_size_t length,                        //
    sz_size_t *sentence_starts, sz_size_t *sentence_lengths, //
    sz_size_t sentences_capacity, sz_size_t *bytes_consumed) {

    sz_size_t sentences = 0;
    if (length == 0 || sentences_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_size_t sentence_start = 0;
    sz_size_t position = sz_sentence_break_next_start_(text, length, 0);
    while (position < length) {
        if (sz_utf8_is_sentence_boundary_serial(text, length, position)) {
            if (sentences == sentences_capacity) {
                if (bytes_consumed) *bytes_consumed = sentence_start;
                return sentences;
            }
            sentence_starts[sentences] = sentence_start;
            sentence_lengths[sentences] = position - sentence_start;
            ++sentences;
            sentence_start = position;
        }
        position = sz_sentence_break_next_start_(text, length, position);
    }

    if (sentences == sentences_capacity) {
        if (bytes_consumed) *bytes_consumed = sentence_start;
        return sentences;
    }
    sentence_starts[sentences] = sentence_start;
    sentence_lengths[sentences] = length - sentence_start;
    ++sentences;
    if (bytes_consumed) *bytes_consumed = length;
    return sentences;
}

#pragma endregion // UAX-29 Sentence Boundaries

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_SENTENCES_SERIAL_H_
