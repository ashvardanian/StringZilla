/**
 *  @brief Serial backend for UAX-29 grapheme cluster boundaries.
 *  @file include/stringzilla/utf8_graphemes/serial.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_GRAPHEMES_SERIAL_H_
#define STRINGZILLA_UTF8_GRAPHEMES_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_graphemes/tables.h"
#include "stringzilla/utf8_codepoints/serial.h" // shared decode helpers
#include "stringzilla/utf8_runes.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region UAX-29 Grapheme Cluster Boundaries

/** @brief Returns the packed Grapheme_Cluster_Break descriptor (gcb | incb << 4 | extpict << 6) for a codepoint. */
SZ_PUBLIC sz_u8_t sz_rune_grapheme_break_property(sz_rune_t rune) {
    if (rune >= 0xAC00u && rune <= 0xD7A3u)
        return ((rune - 0xAC00u) % 28u == 0u) ? (sz_u8_t)(sz_grapheme_break_hangul_lv_k)
                                              : (sz_u8_t)(sz_grapheme_break_hangul_lvt_k);
    if (rune < 0x10000u) {
        sz_u8_t const mid = sz_utf8_grapheme_break_stage_hi_[rune >> 8];
        sz_u8_t const sub = sz_utf8_grapheme_break_stage_mid_[mid * 16u + ((rune >> 4) & 0xFu)];
        sz_u8_t const index = sz_utf8_grapheme_break_stage_sub_[sub * 16u + (rune & 0xFu)];
        return sz_utf8_grapheme_break_id_to_desc_[index];
    }
    for (sz_size_t range = 0; range < sz_utf8_grapheme_break_astral_count_k; ++range)
        if (rune >= sz_utf8_grapheme_break_astral_lo_[range] && rune <= sz_utf8_grapheme_break_astral_hi_[range])
            return sz_utf8_grapheme_break_id_to_desc_[sz_utf8_grapheme_break_astral_id_[range]];
    return 0;
}

/** @brief Extracts the Grapheme_Cluster_Break class (bits 0-3) from a packed descriptor. */
SZ_INTERNAL sz_u8_t sz_grapheme_break_descriptor_gcb_(sz_u8_t descriptor) { return (sz_u8_t)(descriptor & 0x0Fu); }
/** @brief Extracts the Indic_Conjunct_Break value (bits 4-5) from a packed descriptor. */
SZ_INTERNAL sz_u8_t sz_grapheme_break_descriptor_incb_(sz_u8_t descriptor) {
    return (sz_u8_t)((descriptor >> 4) & 0x03u);
}
/** @brief Extracts the Extended_Pictographic flag (bit 6) from a packed descriptor. */
SZ_INTERNAL sz_bool_t sz_grapheme_break_descriptor_extpict_(sz_u8_t descriptor) {
    return (sz_bool_t)((descriptor >> 6) & 1u);
}

/**
 *  @brief Check if `position` is a grapheme cluster boundary per Unicode TR29 (GB1-GB999, incl. GB9c and GB11).
 */
SZ_PUBLIC sz_bool_t sz_utf8_is_grapheme_boundary_serial(sz_cptr_t text, sz_size_t length, sz_size_t position) {
    if (position == 0) return sz_true_k;      // GB1
    if (position >= length) return sz_true_k; // GB2
    if (((sz_u8_t)text[position] & 0xC0) == 0x80) return sz_false_k;

    sz_size_t before_start = sz_utf8_previous_codepoint_start_(text, position);
    sz_size_t decode_before = before_start;
    sz_rune_t before_rune = sz_utf8_decode_(text, length, &decode_before);
    sz_size_t decode_after = position;
    sz_rune_t after_rune = sz_utf8_decode_(text, length, &decode_after);

    sz_u8_t before_descriptor = sz_rune_grapheme_break_property(before_rune);
    sz_u8_t after_descriptor = sz_rune_grapheme_break_property(after_rune);
    sz_u8_t before_class = sz_grapheme_break_descriptor_gcb_(before_descriptor);
    sz_u8_t after_class = sz_grapheme_break_descriptor_gcb_(after_descriptor);

    if (before_class == sz_grapheme_break_cr_k && after_class == sz_grapheme_break_lf_k) return sz_false_k; // GB3
    if (before_class == sz_grapheme_break_control_k || before_class == sz_grapheme_break_cr_k ||
        before_class == sz_grapheme_break_lf_k)
        return sz_true_k; // GB4
    if (after_class == sz_grapheme_break_control_k || after_class == sz_grapheme_break_cr_k ||
        after_class == sz_grapheme_break_lf_k)
        return sz_true_k; // GB5
    if (before_class == sz_grapheme_break_hangul_l_k &&
        (after_class == sz_grapheme_break_hangul_l_k || after_class == sz_grapheme_break_hangul_v_k ||
         after_class == sz_grapheme_break_hangul_lv_k || after_class == sz_grapheme_break_hangul_lvt_k))
        return sz_false_k; // GB6
    if ((before_class == sz_grapheme_break_hangul_lv_k || before_class == sz_grapheme_break_hangul_v_k) &&
        (after_class == sz_grapheme_break_hangul_v_k || after_class == sz_grapheme_break_hangul_t_k))
        return sz_false_k; // GB7
    if ((before_class == sz_grapheme_break_hangul_lvt_k || before_class == sz_grapheme_break_hangul_t_k) &&
        after_class == sz_grapheme_break_hangul_t_k)
        return sz_false_k;                                                                                      // GB8
    if (after_class == sz_grapheme_break_extend_k || after_class == sz_grapheme_break_zwj_k) return sz_false_k; // GB9
    if (after_class == sz_grapheme_break_spacingmark_k) return sz_false_k;                                      // GB9a
    if (before_class == sz_grapheme_break_prepend_k) return sz_false_k;                                         // GB9b

    /* GB9c: Indic conjunct Consonant [Extend Linker]* Linker [Extend Linker]* x Consonant. */
    if (sz_grapheme_break_descriptor_incb_(after_descriptor) == sz_grapheme_incb_consonant_k) {
        sz_bool_t seen_linker = sz_false_k;
        sz_size_t scan = position;
        while (scan > 0) {
            sz_size_t scan_start = sz_utf8_previous_codepoint_start_(text, scan);
            sz_size_t decode_scan = scan_start;
            sz_rune_t scan_rune = sz_utf8_decode_(text, length, &decode_scan);
            sz_u8_t scan_incb = sz_grapheme_break_descriptor_incb_(sz_rune_grapheme_break_property(scan_rune));
            if (scan_incb == sz_grapheme_incb_linker_k) {
                seen_linker = sz_true_k;
                scan = scan_start;
                continue;
            }
            if (scan_incb == sz_grapheme_incb_extend_k) {
                scan = scan_start;
                continue;
            }
            if (scan_incb == sz_grapheme_incb_consonant_k) {
                if (seen_linker) return sz_false_k;
                break;
            }
            break;
        }
    }

    /* GB11: Extended_Pictographic Extend* ZWJ x Extended_Pictographic. Walk strictly before the ZWJ over
     *  Extend* and require an Extended_Pictographic base. */
    if (before_class == sz_grapheme_break_zwj_k && sz_grapheme_break_descriptor_extpict_(after_descriptor) &&
        before_start > 0) {
        sz_size_t walk = before_start; // we always inspect the codepoint that starts before `walk`
        sz_bool_t found = sz_false_k;
        for (;;) {
            sz_size_t walk_start = sz_utf8_previous_codepoint_start_(text, walk);
            sz_size_t decode_walk = walk_start;
            sz_rune_t walk_rune = sz_utf8_decode_(text, length, &decode_walk);
            sz_u8_t walk_descriptor = sz_rune_grapheme_break_property(walk_rune);
            if (sz_grapheme_break_descriptor_gcb_(walk_descriptor) == sz_grapheme_break_extend_k && walk_start > 0) {
                walk = walk_start;
                continue;
            }
            if (sz_grapheme_break_descriptor_gcb_(walk_descriptor) != sz_grapheme_break_extend_k)
                found = sz_grapheme_break_descriptor_extpict_(walk_descriptor);
            break;
        }
        if (found) return sz_false_k;
    }

    /* GB12/GB13: Regional_Indicator x Regional_Indicator only across even-count runs (pairs). */
    if (before_class == sz_grapheme_break_regional_indicator_k &&
        after_class == sz_grapheme_break_regional_indicator_k) {
        sz_size_t ri_count = 0;
        sz_size_t scan = position;
        while (scan > 0) {
            sz_size_t scan_start = sz_utf8_previous_codepoint_start_(text, scan);
            sz_size_t decode_scan = scan_start;
            sz_rune_t scan_rune = sz_utf8_decode_(text, length, &decode_scan);
            if (sz_grapheme_break_descriptor_gcb_(sz_rune_grapheme_break_property(scan_rune)) ==
                sz_grapheme_break_regional_indicator_k) {
                ++ri_count;
                scan = scan_start;
            }
            else break;
        }
        if (ri_count % 2 == 1) return sz_false_k;
    }

    return sz_true_k; // GB999
}

/**
 *  @brief Plural UAX-29 grapheme cluster segmentation: one forward sweep emits every cluster into parallel
 *         `cluster_starts` / `cluster_lengths` arrays at each `sz_utf8_is_grapheme_boundary_serial` position.
 */
SZ_PUBLIC sz_size_t sz_utf8_grapheme_find_boundaries_serial( //
    sz_cptr_t text, sz_size_t length,                        //
    sz_size_t *cluster_starts, sz_size_t *cluster_lengths,   //
    sz_size_t clusters_capacity, sz_size_t *bytes_consumed) {

    sz_size_t clusters = 0;
    if (length == 0 || clusters_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_size_t cluster_start = 0;
    sz_size_t position = sz_utf8_codepoint_length_((sz_u8_t)text[0]);
    while (position < length) {
        if (sz_utf8_is_grapheme_boundary_serial(text, length, position)) {
            if (clusters == clusters_capacity) {
                if (bytes_consumed) *bytes_consumed = cluster_start;
                return clusters;
            }
            cluster_starts[clusters] = cluster_start;
            cluster_lengths[clusters] = position - cluster_start;
            ++clusters;
            cluster_start = position;
        }
        position += sz_utf8_codepoint_length_((sz_u8_t)text[position]);
    }

    if (clusters == clusters_capacity) {
        if (bytes_consumed) *bytes_consumed = cluster_start;
        return clusters;
    }
    cluster_starts[clusters] = cluster_start;
    cluster_lengths[clusters] = length - cluster_start;
    ++clusters;
    if (bytes_consumed) *bytes_consumed = length;
    return clusters;
}

/** @brief Reverse counterpart of `sz_utf8_grapheme_find_boundaries_serial`: clusters emitted from the end backward. */
SZ_PUBLIC sz_size_t sz_utf8_grapheme_rfind_boundaries_serial( //
    sz_cptr_t text, sz_size_t length,                         //
    sz_size_t *cluster_starts, sz_size_t *cluster_lengths,    //
    sz_size_t clusters_capacity, sz_size_t *bytes_consumed) {

    sz_size_t clusters = 0;
    if (length == 0 || clusters_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = length;
        return 0;
    }

    sz_size_t cluster_end = length;
    sz_size_t position = length - 1;
    while (position > 0 && ((sz_u8_t)text[position] & 0xC0) == 0x80) position--;

    while (position > 0) {
        if (sz_utf8_is_grapheme_boundary_serial(text, length, position)) {
            if (clusters == clusters_capacity) {
                if (bytes_consumed) *bytes_consumed = cluster_end;
                return clusters;
            }
            cluster_starts[clusters] = position;
            cluster_lengths[clusters] = cluster_end - position;
            ++clusters;
            cluster_end = position;
        }
        position--;
        while (position > 0 && ((sz_u8_t)text[position] & 0xC0) == 0x80) position--;
    }

    if (clusters == clusters_capacity) {
        if (bytes_consumed) *bytes_consumed = cluster_end;
        return clusters;
    }
    cluster_starts[clusters] = 0;
    cluster_lengths[clusters] = cluster_end;
    ++clusters;
    if (bytes_consumed) *bytes_consumed = 0;
    return clusters;
}

#pragma endregion // UAX-29 Grapheme Cluster Boundaries

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_GRAPHEMES_SERIAL_H_
