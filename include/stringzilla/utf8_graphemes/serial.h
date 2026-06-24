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

#pragma region UAX 29 Grapheme Cluster Boundaries

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

/**
 *  @brief Start offset of the codepoint after @p position: the next non-continuation byte, or @p length.
 *         Mirrors the SIMD `sz_utf8_codepoints_decode_window_` codepoint-start convention (every loaded
 *         non-continuation byte begins a codepoint) so the serial and Ice Lake backends step over malformed
 *         input identically. Stepping by the lead's declared byte length would skip past trailing
 *         non-continuation bytes of a truncated lead, diverging from the window decoder.
 */
SZ_INTERNAL sz_size_t sz_grapheme_break_next_start_(sz_cptr_t text, sz_size_t length, sz_size_t position) {
    sz_size_t next = position + 1;
    while (next < length && ((sz_u8_t)text[next] & 0xC0) == 0x80) ++next;
    return next;
}

/**
 *  @brief Grapheme_Cluster_Break descriptor of the codepoint starting at @p start, decoded BLINDLY to mirror the
 *         Ice Lake `sz_grapheme_classify_window_` value reconstruction byte-for-byte (§6.1 value-based dispatch).
 *
 *         The lead's strict length class selects the fold width — ASCII `< 0x80` stays the raw byte, 3-byte
 *         `1110xxxx` and 4-byte `11110xxx` use their reconstruction, and EVERY other non-ASCII lead (2-byte
 *         `110xxxxx`, stray `0xF8..0xFF`, isolated `0xC0/0xC1`) folds through the 2-byte formula exactly as the
 *         SIMD window does. No continuation / overlong / surrogate validation and no U+FFFD substitution: the
 *         class is dispatched on the decoded VALUE so the serial and Ice Lake backends agree bit-for-bit on
 *         ill-formed input (UAX-29 leaves such bytes undefined). Valid UTF-8 decodes identically to the checked
 *         path; only malformed input differs, by design.
 */
SZ_INTERNAL sz_u8_t sz_grapheme_break_property_at_(sz_cptr_t text, sz_size_t length, sz_size_t start) {
    sz_u8_t const lead = (sz_u8_t)text[start];
    sz_u8_t const byte1 = (start + 1 < length) ? (sz_u8_t)text[start + 1] : 0;
    sz_u8_t const byte2 = (start + 2 < length) ? (sz_u8_t)text[start + 2] : 0;
    sz_u8_t const byte3 = (start + 3 < length) ? (sz_u8_t)text[start + 3] : 0;
    sz_rune_t rune;
    if (lead < 0x80u) { rune = lead; }
    else if (lead >= 0xF8u) {
        // `0xF8..0xFF` begin no valid UTF-8 sequence of any length and are not 2-/3-/4-byte lead patterns, so the
        // SIMD window leaves them out of every lead-length mask. Classify as U+FFFD / Other (§6.2) with no neighbour
        // fold, matching the Ice Lake backend exactly even when such a byte sits on a 64-byte window edge.
        rune = 0xFFFDu;
    }
    else if ((lead & 0xF8u) == 0xF0u) {
        sz_rune_t const plane = (sz_rune_t)((lead & 0x07u) << 2) | ((byte1 >> 4) & 0x03u);
        sz_rune_t const mid = (sz_rune_t)((byte1 & 0x0Fu) << 4) | ((byte2 >> 2) & 0x0Fu);
        sz_rune_t const lo = (sz_rune_t)((byte2 & 0x03u) << 6) | (byte3 & 0x3Fu);
        rune = (plane << 16) | (mid << 8) | lo;
    }
    else if ((lead & 0xF0u) == 0xE0u) {
        sz_u8_t const high = (sz_u8_t)(((lead & 0x0Fu) << 4) | ((byte1 >> 2) & 0x0Fu));
        sz_u8_t const low = (sz_u8_t)(((byte1 & 0x03u) << 6) | (byte2 & 0x3Fu));
        rune = ((sz_rune_t)high << 8) | low;
    }
    else {
        sz_u8_t const high = (sz_u8_t)(((lead & 0x1Fu) >> 2) & 0x07u);
        sz_u8_t const low = (sz_u8_t)(((lead & 0x03u) << 6) | (byte1 & 0x3Fu));
        rune = ((sz_rune_t)high << 8) | low;
    }
    return sz_rune_grapheme_break_property(rune);
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
    // When `prev_start` lands on an orphan continuation byte there is no real codepoint before `position`: the SIMD
    // window excludes continuation bytes from `codepoint_starts`, so the codepoint at `position` is the FIRST real
    // codepoint with no left context and the leading orphan bytes form their own cluster. Force a boundary to match.
    if (((sz_u8_t)text[before_start] & 0xC0u) == 0x80u) return sz_true_k;
    sz_u8_t before_descriptor = sz_grapheme_break_property_at_(text, length, before_start);
    sz_u8_t after_descriptor = sz_grapheme_break_property_at_(text, length, position);
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

    // GB9c: Indic conjunct Consonant [Extend Linker]* Linker [Extend Linker]* x Consonant.
    if (sz_grapheme_break_descriptor_incb_(after_descriptor) == sz_grapheme_incb_consonant_k) {
        sz_bool_t seen_linker = sz_false_k;
        sz_size_t scan = position;
        while (scan > 0) {
            sz_size_t scan_start = sz_utf8_previous_codepoint_start_(text, scan);
            sz_u8_t scan_incb = sz_grapheme_break_descriptor_incb_(
                sz_grapheme_break_property_at_(text, length, scan_start));
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

    // GB11: Extended_Pictographic Extend* ZWJ x Extended_Pictographic. Walk strictly before the ZWJ over
    // Extend* and require an Extended_Pictographic base.
    if (before_class == sz_grapheme_break_zwj_k && sz_grapheme_break_descriptor_extpict_(after_descriptor) &&
        before_start > 0) {
        sz_size_t walk = before_start; // we always inspect the codepoint that starts before `walk`
        sz_bool_t found = sz_false_k;
        for (;;) {
            sz_size_t walk_start = sz_utf8_previous_codepoint_start_(text, walk);
            sz_u8_t walk_descriptor = sz_grapheme_break_property_at_(text, length, walk_start);
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

    // GB12/GB13: Regional_Indicator x Regional_Indicator only across even-count runs (pairs).
    if (before_class == sz_grapheme_break_regional_indicator_k &&
        after_class == sz_grapheme_break_regional_indicator_k) {
        sz_size_t ri_count = 0;
        sz_size_t scan = position;
        while (scan > 0) {
            sz_size_t scan_start = sz_utf8_previous_codepoint_start_(text, scan);
            if (sz_grapheme_break_descriptor_gcb_(sz_grapheme_break_property_at_(text, length, scan_start)) ==
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
 *  @brief Forward run-state carried across codepoints by the bulk segmenter, so the GB9c / GB11 / GB12-13
 *         unbounded runs resolve in O(1) per codepoint instead of a backward re-walk (the scalar twin of the
 *         Ice Lake register carry).
 */
typedef struct sz_grapheme_serial_state_t {
    sz_u8_t previous_descriptor;           // packed descriptor of the previous codepoint
    sz_bool_t regional_indicator_run_odd;  // RI run ending at the previous codepoint has odd length
    sz_bool_t extended_pictographic_run;   // previous codepoint continues an (ExtPict Extend*) run
    sz_bool_t zero_width_joiner_connector; // previous codepoint is a ZWJ closing an ExtPict run
    sz_bool_t indic_conjunct_open;         // a Consonant-rooted InCB run is open at the previous codepoint
    sz_bool_t indic_conjunct_seen_linker;  // ... and a Linker has appeared in that open run
} sz_grapheme_serial_state_t;

/** @brief Boundary decision between @p state's previous codepoint and the @p after codepoint, GB3..GB13 in O(1). */
SZ_INTERNAL sz_bool_t sz_grapheme_serial_boundary_(sz_grapheme_serial_state_t const *state, sz_u8_t after) {
    sz_u8_t const before_class = sz_grapheme_break_descriptor_gcb_(state->previous_descriptor);
    sz_u8_t const after_class = sz_grapheme_break_descriptor_gcb_(after);
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
    // GB9c: Consonant [Extend|Linker]* Linker [Extend|Linker]* x Consonant -- the open run carries the state.
    if (sz_grapheme_break_descriptor_incb_(after) == sz_grapheme_incb_consonant_k && state->indic_conjunct_open &&
        state->indic_conjunct_seen_linker)
        return sz_false_k;
    // GB11: ...ExtPict Extend* ZWJ x ExtPict -- the previous ZWJ-connector carries the run.
    if (state->zero_width_joiner_connector && sz_grapheme_break_descriptor_extpict_(after)) return sz_false_k;
    // GB12/GB13: RI x RI when the run ending at the previous RI is odd (so this RI completes a pair).
    if (before_class == sz_grapheme_break_regional_indicator_k &&
        after_class == sz_grapheme_break_regional_indicator_k && state->regional_indicator_run_odd)
        return sz_false_k;
    return sz_true_k; // GB999
}

/** @brief Advance @p state by the @p after codepoint: toggle/close the RI, ExtPict-ZWJ and InCB runs. */
SZ_INTERNAL void sz_grapheme_serial_advance_(sz_grapheme_serial_state_t *state, sz_u8_t after) {
    sz_u8_t const after_class = sz_grapheme_break_descriptor_gcb_(after);
    state->regional_indicator_run_odd = (after_class == sz_grapheme_break_regional_indicator_k)
                                            ? (sz_bool_t)(!state->regional_indicator_run_odd)
                                            : sz_false_k;
    if (sz_grapheme_break_descriptor_extpict_(after)) {
        state->extended_pictographic_run = sz_true_k;
        state->zero_width_joiner_connector = sz_false_k;
    }
    else if (after_class == sz_grapheme_break_extend_k && state->extended_pictographic_run) {
        state->zero_width_joiner_connector = sz_false_k; // Extend continues the ExtPict run
    }
    else if (after_class == sz_grapheme_break_zwj_k && state->extended_pictographic_run) {
        state->zero_width_joiner_connector = sz_true_k; // ZWJ closes the run, opens the connector
        state->extended_pictographic_run = sz_false_k;
    }
    else {
        state->extended_pictographic_run = sz_false_k;
        state->zero_width_joiner_connector = sz_false_k;
    }
    sz_u8_t const after_incb = sz_grapheme_break_descriptor_incb_(after);
    if (after_incb == sz_grapheme_incb_consonant_k) {
        state->indic_conjunct_open = sz_true_k;
        state->indic_conjunct_seen_linker = sz_false_k;
    }
    else if (after_incb == sz_grapheme_incb_linker_k && state->indic_conjunct_open) {
        state->indic_conjunct_seen_linker = sz_true_k;
    }
    else if (after_incb == sz_grapheme_incb_extend_k && state->indic_conjunct_open) {
        // Extend continues the open conjunct run unchanged.
    }
    else {
        state->indic_conjunct_open = sz_false_k;
        state->indic_conjunct_seen_linker = sz_false_k;
    }
    state->previous_descriptor = after;
}

/**
 *  @brief Plural UAX-29 grapheme cluster segmentation: ONE forward sweep emits every cluster into parallel
 *         `cluster_starts` / `cluster_lengths`, carrying the GB9c/GB11/GB12-13 runs in a register state (O(n), no
 *         backward re-walks). Byte-identical to the per-position `sz_utf8_is_grapheme_boundary_serial`.
 */
SZ_PUBLIC sz_size_t sz_utf8_graphemes_serial(              //
    sz_cptr_t text, sz_size_t length,                      //
    sz_size_t *cluster_starts, sz_size_t *cluster_lengths, //
    sz_size_t clusters_capacity, sz_size_t *bytes_consumed) {

    sz_size_t clusters = 0;
    if (length == 0 || clusters_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_size_t cluster_start = 0;
    sz_grapheme_serial_state_t state;
    state.regional_indicator_run_odd = sz_false_k;
    state.extended_pictographic_run = sz_false_k;
    state.zero_width_joiner_connector = sz_false_k;
    state.indic_conjunct_open = sz_false_k;
    state.indic_conjunct_seen_linker = sz_false_k;
    sz_grapheme_serial_advance_(&state, sz_grapheme_break_property_at_(text, length, 0)); // seed from codepoint 0

    sz_size_t position = sz_grapheme_break_next_start_(text, length, 0);
    while (position < length) {
        sz_u8_t const after_descriptor = sz_grapheme_break_property_at_(text, length, position);
        // A leading orphan continuation before `position` has no real left codepoint: force a boundary (matches the
        // window decoder, which excludes continuation bytes from codepoint starts). Bounded back-scan, amortized O(n).
        sz_size_t const before_start = sz_utf8_previous_codepoint_start_(text, position);
        sz_bool_t const boundary = (((sz_u8_t)text[before_start] & 0xC0u) == 0x80u)
                                       ? sz_true_k
                                       : sz_grapheme_serial_boundary_(&state, after_descriptor);
        if (boundary) {
            if (clusters == clusters_capacity) {
                if (bytes_consumed) *bytes_consumed = cluster_start;
                return clusters;
            }
            cluster_starts[clusters] = cluster_start;
            cluster_lengths[clusters] = position - cluster_start;
            ++clusters;
            cluster_start = position;
        }
        sz_grapheme_serial_advance_(&state, after_descriptor);
        position = sz_grapheme_break_next_start_(text, length, position);
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

#pragma endregion // UAX 29 Grapheme Cluster Boundaries

#pragma region Portable grapheme boundary algebra

/**
 *  @brief  Cross-window left-context carry: the situation just AFTER the previous window's last codepoint, so this
 *          window's codepoint 0 sees an exact (i-1) context for all of GB3..GB13. ISA-independent.
 */
typedef struct sz_grapheme_carry_t {
    int has_previous;                /**< Stream has emitted at least one codepoint already. */
    sz_u64_t previous_class_bit[14]; /**< Bit 0 set => the previous codepoint had that gcb class. */
    int regional_indicator_run_odd;  /**< RI run ending at the previous codepoint has odd inclusive length. */
    int extended_pictographic_run;   /**< Previous codepoint tails an (ExtPict Extend*) chain. */
    int zero_width_joiner_connector; /**< Previous codepoint is a ZWJ closing an (ExtPict Extend*) run. */
    int indic_conjunct_open;         /**< An InCB run rooted at a Consonant is open at the previous cp. */
    int indic_conjunct_seen_linker;  /**< ... and a Linker has appeared in that open run. */
} sz_grapheme_carry_t;

/** @brief  A zero-initialized cross-window carry (stream start: no previous codepoint, no open runs). */
SZ_INTERNAL sz_grapheme_carry_t sz_grapheme_carry_empty_(void) {
    sz_grapheme_carry_t carry;
    carry.has_previous = 0;
    for (int class_index = 0; class_index < 14; ++class_index) carry.previous_class_bit[class_index] = 0;
    carry.regional_indicator_run_odd = 0;
    carry.extended_pictographic_run = 0;
    carry.zero_width_joiner_connector = 0;
    carry.indic_conjunct_open = 0;
    carry.indic_conjunct_seen_linker = 0;
    return carry;
}

/** @brief  Per-window class membership masks derived once from the codepoint-dense packed descriptor bytes. */
typedef struct sz_grapheme_window_masks_t {
    sz_u64_t class_bit[14]; /**< `class_bit[c]` bit i set => dense codepoint i is grapheme class c. */
    sz_u64_t extended_pictographic;
    sz_u64_t indic_consonant, indic_extend, indic_linker;
} sz_grapheme_window_masks_t;

/**
 *  @brief  The three unbounded run-carries (GB12/13 RI parity, GB11 ExtPict-ZWJ, GB9c InCB), computed once from this
 *          window's masks and the inbound carry. Returns their join-mask contribution and updates @p next for the
 *          following window. Bit i of the result set → a no-break suppression sits before dense codepoint i.
 *          ISA-independent: pure `sz_u64_t` bit-mask algebra over the substrate scan primitives.
 */
SZ_INTERNAL sz_u64_t sz_grapheme_stateful_joins_(sz_grapheme_window_masks_t const *window, int codepoint_count,
                                                 sz_grapheme_carry_t const *previous, sz_grapheme_carry_t *next) {
    sz_u64_t const regional = window->class_bit[sz_grapheme_break_regional_indicator_k];
    sz_u64_t const extend = window->class_bit[sz_grapheme_break_extend_k];
    sz_u64_t const zero_width_joiner = window->class_bit[sz_grapheme_break_zwj_k];
    int const last = codepoint_count - 1;
    sz_u64_t const last_bit = (codepoint_count > 0) ? (1ull << last) : 0;
    sz_u64_t join = 0;

    // GB12/13: no-break before a Regional_Indicator that is an EVEN member of its maximal run. A segmented XOR parity
    // scan over the run computes membership; the run is seeded from the inbound carry's RI parity. Skipped wholesale
    // when the window holds no Regional_Indicator (the overwhelmingly common case).
    if (regional) {
        sz_u64_t const previous_ri = (regional << 1) |
                                     (previous->previous_class_bit[sz_grapheme_break_regional_indicator_k] & 1ull);
        sz_u64_t const starts = regional & ~previous_ri;
        // Seed lane 0 from the inbound run parity, branchless: a lane-0 RI that is NOT a fresh run start continues a
        // previous-window run, so XOR in the carried odd-parity bit. Then a segmented XOR-parity scan over the run.
        sz_u64_t const seed = regional ^
                              (regional & ~starts & 1ull & (sz_u64_t)(previous->regional_indicator_run_odd & 1));
        sz_u64_t const parity = sz_utf8_codepoints_segmented_parity_(seed, regional & ~starts);
        join |= regional & previous_ri & ~parity; // parity[i]==0 -> even member -> pairs with i-1 -> no break
        next->regional_indicator_run_odd = (regional & last_bit) ? ((parity & last_bit) != 0) : 0;
    }
    else { next->regional_indicator_run_odd = 0; }

    // GB11: Extended_Pictographic Extend* ZWJ x Extended_Pictographic, via a right-smear of the ExtPict run through
    // Extend, a ZWJ whose left lane sits inside that run, and a no-break before the following ExtPict. Skipped when no
    // ExtPict / ZWJ lane is present and no chain is open from the previous window.
    if (window->extended_pictographic | zero_width_joiner | (previous->extended_pictographic_run ? 1ull : 0ull)) {
        sz_u64_t run = window->extended_pictographic;
        run |= (extend & 1ull) & (sz_u64_t)(previous->extended_pictographic_run); // branchless lane-0 chain seed
        run = sz_utf8_codepoints_fill_right_(run, extend);
        sz_u64_t const previous_run = (run << 1) | (previous->extended_pictographic_run ? 1ull : 0ull);
        sz_u64_t const connector = zero_width_joiner & previous_run;
        sz_u64_t const previous_connector = (connector << 1) | (previous->zero_width_joiner_connector ? 1ull : 0ull);
        join |= previous_connector & window->extended_pictographic;
        next->extended_pictographic_run = (run & last_bit) != 0;
        next->zero_width_joiner_connector = (connector & last_bit) != 0;
    }
    else {
        next->extended_pictographic_run = 0;
        next->zero_width_joiner_connector = 0;
    }

    // GB9c: Consonant [Extend|Linker]* Linker [Extend|Linker]* x Consonant. The open run and the seen-linker run both
    // flood rightward through the continuation lanes (Extend|Linker), seeded from the inbound carry. Skipped when no
    // Indic_Conjunct_Break lane is present and no conjunct is open from the previous window.
    if (window->indic_consonant | window->indic_extend | window->indic_linker |
        (previous->indic_conjunct_open ? 1ull : 0ull)) {
        sz_u64_t const continuation = window->indic_extend | window->indic_linker;
        sz_u64_t open = window->indic_consonant;
        open |= (continuation & 1ull) & (sz_u64_t)(previous->indic_conjunct_open); // branchless lane-0 open seed
        open = sz_utf8_codepoints_fill_right_(open, continuation);
        sz_u64_t seen = window->indic_linker & open;
        seen |= (continuation & 1ull) &
                (sz_u64_t)(previous->indic_conjunct_open & previous->indic_conjunct_seen_linker);
        seen = sz_utf8_codepoints_fill_right_(seen, continuation & open);
        sz_u64_t previous_open_seen = ((open & seen) << 1);
        previous_open_seen |= (sz_u64_t)(previous->indic_conjunct_open & previous->indic_conjunct_seen_linker);
        join |= previous_open_seen & window->indic_consonant;
        int const last_in_run = ((window->indic_consonant | continuation) & last_bit) != 0 && (open & last_bit) != 0;
        next->indic_conjunct_open = last_in_run;
        next->indic_conjunct_seen_linker = last_in_run && ((seen & last_bit) != 0);
    }
    else {
        next->indic_conjunct_open = 0;
        next->indic_conjunct_seen_linker = 0;
    }
    return join;
}

/**
 *  @brief  Per-class shift of @p current into the (i-1) position, seeding lane 0 from the inbound carry's last class.
 */
SZ_INTERNAL sz_u64_t sz_grapheme_previous_(sz_u64_t current, sz_grapheme_carry_t const *previous, int class_index) {
    return (current << 1) | (previous->previous_class_bit[class_index] & 1ull);
}

/**
 *  @brief  Portable boundary-algebra engine: from per-class window masks + the inbound carry, compute the
 *          codepoint-dense boundary bitmask (bit i set → a grapheme break sits before dense codepoint i) and thread
 *          the carry forward. The ISA-independent twin of the icelake fused window-boundaries; a per-ISA extractor
 *          builds @p window from descriptors, then calls this. Pure `sz_u64_t` algebra, no intrinsics.
 */
SZ_INTERNAL sz_u64_t sz_grapheme_window_boundaries_(sz_grapheme_window_masks_t const *window, int codepoint_count,
                                                    sz_u64_t valid, sz_grapheme_carry_t *carry) {
    sz_grapheme_carry_t const previous = *carry;

    sz_u64_t const carriage_return = window->class_bit[sz_grapheme_break_cr_k];
    sz_u64_t const line_feed = window->class_bit[sz_grapheme_break_lf_k];
    sz_u64_t const control = window->class_bit[sz_grapheme_break_control_k];
    sz_u64_t const extend = window->class_bit[sz_grapheme_break_extend_k];
    sz_u64_t const zero_width_joiner = window->class_bit[sz_grapheme_break_zwj_k];
    sz_u64_t const prepend = window->class_bit[sz_grapheme_break_prepend_k];
    sz_u64_t const spacing_mark = window->class_bit[sz_grapheme_break_spacingmark_k];
    sz_u64_t const hangul_lead = window->class_bit[sz_grapheme_break_hangul_l_k];
    sz_u64_t const hangul_vowel = window->class_bit[sz_grapheme_break_hangul_v_k];
    sz_u64_t const hangul_trailing = window->class_bit[sz_grapheme_break_hangul_t_k];
    sz_u64_t const hangul_lead_vowel = window->class_bit[sz_grapheme_break_hangul_lv_k];
    sz_u64_t const hangul_lead_vowel_trailing = window->class_bit[sz_grapheme_break_hangul_lvt_k];

    sz_u64_t const previous_cr = sz_grapheme_previous_(carriage_return, &previous, sz_grapheme_break_cr_k);
    sz_u64_t const previous_lf = sz_grapheme_previous_(line_feed, &previous, sz_grapheme_break_lf_k);
    sz_u64_t const previous_control = sz_grapheme_previous_(control, &previous, sz_grapheme_break_control_k);
    sz_u64_t const previous_prepend = sz_grapheme_previous_(prepend, &previous, sz_grapheme_break_prepend_k);
    sz_u64_t const previous_lead = sz_grapheme_previous_(hangul_lead, &previous, sz_grapheme_break_hangul_l_k);
    sz_u64_t const previous_vowel = sz_grapheme_previous_(hangul_vowel, &previous, sz_grapheme_break_hangul_v_k);
    sz_u64_t const previous_trailing = sz_grapheme_previous_(hangul_trailing, &previous, sz_grapheme_break_hangul_t_k);
    sz_u64_t const previous_lead_vowel = sz_grapheme_previous_(hangul_lead_vowel, &previous,
                                                               sz_grapheme_break_hangul_lv_k);
    sz_u64_t const previous_lead_vowel_trailing = sz_grapheme_previous_(hangul_lead_vowel_trailing, &previous,
                                                                        sz_grapheme_break_hangul_lvt_k);

    sz_u64_t join = 0;
    join |= previous_cr & line_feed; // GB3  CR x LF
    join |= previous_lead & (hangul_lead | hangul_vowel | hangul_lead_vowel | hangul_lead_vowel_trailing); // GB6
    join |= (previous_lead_vowel | previous_vowel) & (hangul_vowel | hangul_trailing);                     // GB7
    join |= (previous_lead_vowel_trailing | previous_trailing) & hangul_trailing;                          // GB8
    join |= (extend | zero_width_joiner); // GB9  x (Extend|ZWJ)
    join |= spacing_mark;                 // GB9a x SpacingMark
    join |= previous_prepend;             // GB9b Prepend x

    join |= sz_grapheme_stateful_joins_(window, codepoint_count, &previous, carry); // GB9c, GB11, GB12/13

    // GB4/GB5 force a break around Control/CR/LF (except CR x LF kept by GB3).
    sz_u64_t const force = ((previous_control | previous_cr | previous_lf) | (control | carriage_return | line_feed)) &
                           ~(previous_cr & line_feed);

    sz_u64_t boundary = (~join | force) & valid;
    if (!previous.has_previous) boundary |= (valid & 1ull); // GB1 at stream start

    if (codepoint_count > 0) {
        sz_u64_t const last_bit = 1ull << (codepoint_count - 1);
        for (int class_index = 0; class_index < 14; ++class_index)
            carry->previous_class_bit[class_index] = (window->class_bit[class_index] & last_bit) ? 1ull : 0ull;
        carry->has_previous = 1;
    }
    return boundary;
}

#pragma endregion Portable grapheme boundary algebra

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_GRAPHEMES_SERIAL_H_
