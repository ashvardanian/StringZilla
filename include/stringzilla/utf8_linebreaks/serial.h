/**
 *  @brief Serial backend for UAX-14 line break boundaries.
 *  @file include/stringzilla/utf8_linebreaks/serial.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_LINEBREAKS_SERIAL_H_
#define STRINGZILLA_UTF8_LINEBREAKS_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_linebreaks/tables.h"
#include "stringzilla/utf8_runes/serial.h" // shared decode helpers

#ifdef __cplusplus
extern "C" {
#endif

#pragma region UAX 14 Line Boundaries

/** @brief Returns the UAX-14 palette descriptor (class in bits 0-5, side flags in bits 6-13) for a codepoint. */
SZ_API_COMPTIME sz_u16_t sz_rune_line_break_property(sz_rune_t rune) {
    for (sz_size_t range = 0; range < sz_utf8_line_break_big_count_k; ++range)
        if (rune >= sz_utf8_line_break_big_lo_[range] && rune <= sz_utf8_line_break_big_hi_[range])
            return sz_utf8_line_break_palette_[sz_utf8_line_break_big_idx_[range]];
    if (rune < 0x800u) return sz_utf8_line_break_palette_[sz_utf8_line_break_page_lut_[rune]];
    if (rune < 0x10000u) {
        sz_u32_t const offset = (sz_u32_t)(rune - 0x800u);
        sz_u32_t const block = offset / sz_utf8_line_break_trie_block_k;
        sz_u32_t const within = offset % sz_utf8_line_break_trie_block_k;
        sz_u32_t const super = block / sz_utf8_line_break_trie_subblock_k;
        sz_u32_t const super_offset = block % sz_utf8_line_break_trie_subblock_k;
        sz_u8_t const level1 = sz_utf8_line_break_trie_l1_[super];
        sz_u16_t const leaf = sz_utf8_line_break_trie_l2_[level1 * sz_utf8_line_break_trie_subblock_k + super_offset];
        sz_u8_t const index = sz_utf8_line_break_trie_leaf_[leaf * sz_utf8_line_break_trie_block_k + within];
        return sz_utf8_line_break_palette_[index];
    }
    for (sz_size_t range = 0; range < sz_utf8_line_break_astral_count_k; ++range)
        if (rune >= sz_utf8_line_break_astral_lo_[range] && rune <= sz_utf8_line_break_astral_hi_[range])
            return sz_utf8_line_break_palette_[sz_utf8_line_break_astral_idx_[range]];
    return sz_utf8_line_break_palette_[0];
}

/** @brief Resolved Line_Break class (palette bits 0-5) of a descriptor. */
SZ_HELPER_INLINE sz_u8_t sz_line_break_descriptor_class_(sz_u16_t descriptor) { return (sz_u8_t)(descriptor & 0x3Fu); }
/** @brief True if a descriptor carries the general-category Pi (initial quote) flag. */
SZ_HELPER_INLINE sz_bool_t sz_line_break_descriptor_is_pi_(sz_u16_t descriptor) {
    return (sz_bool_t)((descriptor >> 6) & 1u);
}
/** @brief True if a descriptor carries the general-category Pf (final quote) flag. */
SZ_HELPER_INLINE sz_bool_t sz_line_break_descriptor_is_pf_(sz_u16_t descriptor) {
    return (sz_bool_t)((descriptor >> 7) & 1u);
}
/** @brief True if a descriptor carries the East-Asian Width F/W/H flag. */
SZ_HELPER_INLINE sz_bool_t sz_line_break_descriptor_is_eaw_(sz_u16_t descriptor) {
    return (sz_bool_t)((descriptor >> 8) & 1u);
}
/** @brief True if a descriptor is an unassigned Extended_Pictographic (LB30b second clause). */
SZ_HELPER_INLINE sz_bool_t sz_line_break_descriptor_is_extpict_cn_(sz_u16_t descriptor) {
    return (sz_bool_t)((descriptor >> 9) & 1u);
}
/** @brief True if a descriptor is Dotted_Circle U+25CC (an aksara base in LB28a). */
SZ_HELPER_INLINE sz_bool_t sz_line_break_descriptor_is_dotted_circle_(sz_u16_t descriptor) {
    return (sz_bool_t)((descriptor >> 13) & 1u);
}
/** @brief True for a Line_Break CM or ZWJ class (LB9 attachment candidates). */
SZ_HELPER_INLINE sz_bool_t sz_line_break_is_cm_or_zwj_(sz_u8_t line_break_class) {
    return (sz_bool_t)(line_break_class == sz_line_break_cm_k || line_break_class == sz_line_break_zwj_k);
}

/** @brief One decoded codepoint's LB1-resolved Line_Break class; advances @p position and returns the descriptor. */
SZ_HELPER_AUTO sz_u8_t sz_line_break_decode_one_(sz_cptr_t text, sz_size_t length, sz_size_t *position,
                                                 sz_u16_t *descriptor_out) {
    sz_size_t decode = *position;
    sz_rune_t const rune = sz_utf8_next_rune_(text, length, &decode);
    sz_u16_t const descriptor = sz_rune_line_break_property(rune);
    sz_u8_t line_break_class = sz_line_break_descriptor_class_(descriptor);
    // LB1: SA → CM if combining (descriptor mark bit) else AL; AI/SG/XX → AL; CJ → NS.
    if (line_break_class == sz_line_break_sa_k)
        line_break_class = (descriptor & (1u << 12)) ? (sz_u8_t)sz_line_break_cm_k : (sz_u8_t)sz_line_break_al_k;
    else if (line_break_class == sz_line_break_ai_k || line_break_class == sz_line_break_sg_k ||
             line_break_class == sz_line_break_xx_k)
        line_break_class = (sz_u8_t)sz_line_break_al_k;
    else if (line_break_class == sz_line_break_cj_k) line_break_class = (sz_u8_t)sz_line_break_ns_k;
    *descriptor_out = descriptor;
    *position = decode > *position ? decode : *position + 1;
    return line_break_class;
}

/** @brief One LB9/LB10-collapsed cluster: a base codepoint with its trailing combining marks already folded in. */
typedef struct sz_line_break_cluster_t {
    sz_size_t byte_start;      /**< absolute byte offset of the cluster's base codepoint */
    sz_u16_t descriptor;       /**< base descriptor (0 for a lone mark reclassified to AL) */
    sz_u8_t line_break_class;  /**< effective Line_Break class (≡ raw class, post LB9/LB10) */
    sz_bool_t preceded_by_zwj; /**< the codepoint immediately before the base was a ZWJ (LB8a) */
    sz_bool_t valid;           /**< sz_false_k once the text is exhausted */
} sz_line_break_cluster_t;

/** @brief A fully-zeroed invalid cluster standing in for start/end of text — every field is set, so no slot is ever
 *         read uninitialized even though `valid == sz_false_k` gates its use. */
SZ_HELPER_INLINE sz_line_break_cluster_t sz_line_break_cluster_invalid_(void) {
    sz_line_break_cluster_t cluster;
    cluster.byte_start = 0;
    cluster.descriptor = 0;
    cluster.line_break_class = 0;
    cluster.preceded_by_zwj = sz_false_k;
    cluster.valid = sz_false_k;
    return cluster;
}

/**
 *  @brief Decodes the next LB9/LB10 cluster — the base codepoint plus every trailing CM/ZWJ that attaches to it.
 *         A combining mark with no attachable base (start of text, or after BK/CR/LF/NL/SP/ZW) is LB10: kept as a
 *         lone AL cluster. @p last_codepoint_was_zwj carries the LB8a "preceded by ZWJ" bit across calls.
 */
SZ_HELPER_AUTO sz_line_break_cluster_t sz_line_break_next_cluster_(sz_cptr_t text, sz_size_t length,
                                                                   sz_size_t *position,
                                                                   sz_bool_t *last_codepoint_was_zwj) {
    sz_line_break_cluster_t cluster;
    if (*position >= length) return sz_line_break_cluster_invalid_();
    cluster.valid = sz_true_k;
    cluster.preceded_by_zwj = *last_codepoint_was_zwj;
    cluster.byte_start = *position;

    sz_u16_t descriptor;
    sz_u8_t const base_class = sz_line_break_decode_one_(text, length, position, &descriptor);
    *last_codepoint_was_zwj = (sz_bool_t)(base_class == sz_line_break_zwj_k);

    // LB10: a lone CM/ZWJ acts as AL; LB9: a base accepts trailing marks unless it is itself a hard break or space.
    sz_bool_t attachable;
    if (sz_line_break_is_cm_or_zwj_(base_class)) {
        cluster.line_break_class = (sz_u8_t)sz_line_break_al_k;
        cluster.descriptor = 0;
        attachable = sz_true_k;
    }
    else {
        cluster.line_break_class = base_class;
        cluster.descriptor = descriptor;
        attachable = (sz_bool_t)(base_class != sz_line_break_bk_k && base_class != sz_line_break_cr_k &&
                                 base_class != sz_line_break_lf_k && base_class != sz_line_break_nl_k &&
                                 base_class != sz_line_break_sp_k && base_class != sz_line_break_zw_k);
    }

    // Fold trailing combining marks (LB9) into the base; the base stays attachable across a run of marks.
    while (attachable && *position < length) {
        sz_size_t peek = *position;
        sz_u16_t mark_descriptor;
        sz_u8_t const mark_class = sz_line_break_decode_one_(text, length, &peek, &mark_descriptor);
        if (!sz_line_break_is_cm_or_zwj_(mark_class)) break;
        *last_codepoint_was_zwj = (sz_bool_t)(mark_class == sz_line_break_zwj_k);
        *position = peek;
    }
    return cluster;
}

/**
 *  @brief Forward-carried UAX-14 run state — the streaming analogue of the per-cluster context the window engine
 *         rebuilt each window: the nearest non-space cluster (LB8/14-17), the "NU (SY|IS)*" numeric run (LB25), and
 *         the Regional_Indicator parity (LB30a). Advanced by `right` each step via `sz_line_break_serial_advance_`.
 */
typedef struct sz_line_break_serial_state_t {
    sz_u8_t last_non_space_class;            /**< nearest non-SP class at or left of `left` (LB8/14-17) */
    sz_u16_t last_non_space_descriptor;      /**< its base descriptor (LB15a gc=Pi test) */
    sz_u8_t last_non_space_left_class;       /**< class of the cluster just left of it (LB15a left context) */
    sz_bool_t last_non_space_is_first;       /**< it is cluster 0 (LB15a quote_base_index == 0) */
    sz_bool_t numeric_run_open;              /**< a "NU (SY|IS)*" run is open at `left` (LB25) */
    sz_bool_t numeric_run_open_before;       /**< ... was open at `previous2` (LB25 close before CL/CP) */
    sz_bool_t regional_indicator_parity_odd; /**< the RI run ending at `left` has odd length (LB30a) */
    sz_bool_t last_codepoint_was_zwj; /**< the last codepoint decoded was a ZWJ (LB8a; carried into the decoder) */
    sz_size_t cluster_index;          /**< index of `right`; the LB8 `had_space` / left-context `≥ 2` guards */
} sz_line_break_serial_state_t;

/** @brief Advance @p state by the `right` cluster (the one about to become `left`): refresh the nearest non-space
 *         context, the numeric-run flags, the Regional_Indicator parity, and the cluster counter. */
SZ_HELPER_AUTO void sz_line_break_serial_advance_(sz_line_break_serial_state_t *state,
                                                  sz_line_break_cluster_t const *right, sz_u8_t left_class) {
    sz_u8_t const right_class = right->line_break_class;
    if (right_class != sz_line_break_sp_k) {
        state->last_non_space_class = right_class;
        state->last_non_space_descriptor = right->descriptor;
        state->last_non_space_left_class = left_class;
        state->last_non_space_is_first = sz_false_k;
    }
    state->numeric_run_open_before = state->numeric_run_open;
    if (right_class == sz_line_break_nu_k) state->numeric_run_open = sz_true_k;
    else if (!(state->numeric_run_open && (right_class == sz_line_break_sy_k || right_class == sz_line_break_is_k)))
        state->numeric_run_open = sz_false_k;
    if (right_class == sz_line_break_ri_k)
        state->regional_indicator_parity_odd = (sz_bool_t)!state->regional_indicator_parity_odd;
    else state->regional_indicator_parity_odd = sz_false_k;
    ++state->cluster_index;
}

/**
 *  @brief Plural UAX-14 line-break segmentation: one streaming left-to-right sweep emits every break opportunity
 *         (LB1-LB31) into parallel `line_starts` / `line_lengths`. Forward-only (no reverse counterpart).
 *
 *  Like the word/grapheme/sentence serial kernels, this decodes each codepoint once and carries the bounded UAX-14
 *  context forward — no fixed window, O(1) stack — so a line of any length segments correctly. The carried context is
 *  a tiny sliding run of five LB9/LB10-collapsed clusters (two back for LB19/LB20a/LB21a/LB28a, two ahead for
 *  LB15/LB19/LB25/LB28a) plus the scalar run-state the rules summarise: the nearest non-space cluster (LB8/14-17), the
 *  "NU (SY|IS)*" numeric run (LB25), and the Regional_Indicator parity (LB30a). On a full output buffer
 *  `*bytes_consumed` is the start of the open line that did not fit — always a true LB boundary — so a caller resumes
 *  from `text + *bytes_consumed` and obtains the identical remainder.
 */
SZ_API_COMPTIME sz_size_t sz_utf8_linebreaks_serial( //
    sz_cptr_t text, sz_size_t length,                //
    sz_size_t *line_starts, sz_size_t *line_lengths, //
    sz_size_t lines_capacity, sz_size_t *bytes_consumed) {

    sz_size_t lines = 0;
    if (length == 0 || lines_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_size_t position = 0;
    sz_line_break_serial_state_t state;
    state.last_codepoint_was_zwj = sz_false_k; // seeds the decoder; the rest of `state` is seeded from `left` below

    // Five-cluster sliding context centred on the (left | right) boundary under test: `previous2` (two left, for
    // LB19/LB20a/LB21a/LB28a), `left`, `right`, `ahead1`, `ahead2` (LB15/LB19/LB25/LB28a lookahead). Invalid slots
    // stand in for start/end of text, where the rules' index ± k guards already do the right thing.
    sz_line_break_cluster_t previous2 = sz_line_break_cluster_invalid_(); // stands in for "before start of text"
    sz_line_break_cluster_t left = sz_line_break_next_cluster_(text, length, &position, &state.last_codepoint_was_zwj);
    sz_line_break_cluster_t right = sz_line_break_next_cluster_(text, length, &position, &state.last_codepoint_was_zwj);
    sz_line_break_cluster_t ahead1 = sz_line_break_next_cluster_(text, length, &position,
                                                                 &state.last_codepoint_was_zwj);
    sz_line_break_cluster_t ahead2 = sz_line_break_next_cluster_(text, length, &position,
                                                                 &state.last_codepoint_was_zwj);

    // The open line begins at `left` (the first cluster; LB2 has no break before start of text).
    sz_size_t open_line_start = left.byte_start; // = 0

    // Seed the forward-carried run state from the first cluster; `sz_line_break_serial_advance_` advances it by `right`
    // each step — the continuous analogue of the window engine's per-cluster advance, never re-seeded.
    state.last_non_space_class = left.line_break_class;
    state.last_non_space_descriptor = left.descriptor;
    state.last_non_space_left_class = (sz_u8_t)sz_line_break_xx_k;
    state.last_non_space_is_first = sz_true_k;
    state.numeric_run_open = (sz_bool_t)(left.line_break_class == sz_line_break_nu_k);
    state.numeric_run_open_before = sz_false_k;
    state.regional_indicator_parity_odd = (sz_bool_t)(left.line_break_class == sz_line_break_ri_k);
    state.cluster_index = 1; // index of `right`; the LB8 `had_space` / left-context `≥ 2` guards

    while (right.valid) {
        sz_u8_t const left_class = left.line_break_class;
        sz_u8_t const right_class = right.line_break_class;
        sz_bool_t is_break = sz_false_k;
        sz_bool_t resolved = sz_false_k;

        sz_u8_t const leftmost_non_space_class = state.last_non_space_class;
        sz_bool_t const had_space = (sz_bool_t)(left_class == sz_line_break_sp_k && state.cluster_index >= 2);
        sz_bool_t const left_has_predecessor = (sz_bool_t)(state.cluster_index >= 2); // left_base_index > 0

        // First-match-wins precedence over LB4-LB31: the first rule whose condition holds latches `resolved` so no
        // later rule fires, with `is_break` carrying the join/break verdict. Order is UAX-14 verbatim. Neighbour
        // lookups read the sliding clusters (`previous2`/`left`/`right`/`ahead1`/`ahead2`) and the carried run-state.
        if (!resolved && left_class == sz_line_break_bk_k) {
            is_break = sz_true_k;
            resolved = sz_true_k;
        } // LB4
        if (!resolved && left_class == sz_line_break_cr_k && right_class == sz_line_break_lf_k)
            resolved = sz_true_k; // LB5
        if (!resolved && (left_class == sz_line_break_cr_k || left_class == sz_line_break_lf_k ||
                          left_class == sz_line_break_nl_k)) {
            is_break = sz_true_k;
            resolved = sz_true_k;
        }
        if (!resolved && (right_class == sz_line_break_bk_k || right_class == sz_line_break_cr_k ||
                          right_class == sz_line_break_lf_k || right_class == sz_line_break_nl_k))
            resolved = sz_true_k; // LB6
        if (!resolved && (right_class == sz_line_break_sp_k || right_class == sz_line_break_zw_k))
            resolved = sz_true_k; // LB7
        if (!resolved && leftmost_non_space_class == sz_line_break_zw_k && had_space) {
            is_break = sz_true_k;
            resolved = sz_true_k;
        } // LB8
        if (!resolved && left_class == sz_line_break_zw_k) {
            is_break = sz_true_k;
            resolved = sz_true_k;
        }
        if (!resolved && right.preceded_by_zwj)
            resolved = sz_true_k; // LB8a: no break after ZWJ (preserved across the collapse)
        if (!resolved && sz_line_break_is_cm_or_zwj_(right_class) && left_class != sz_line_break_bk_k &&
            left_class != sz_line_break_cr_k && left_class != sz_line_break_lf_k && left_class != sz_line_break_nl_k &&
            left_class != sz_line_break_sp_k && left_class != sz_line_break_zw_k)
            resolved = sz_true_k; // LB9
        if (!resolved && (right_class == sz_line_break_wj_k || left_class == sz_line_break_wj_k))
            resolved = sz_true_k;                                                // LB11
        if (!resolved && left_class == sz_line_break_gl_k) resolved = sz_true_k; // LB12
        if (!resolved && right_class == sz_line_break_gl_k &&
            !(left_class == sz_line_break_sp_k || left_class == sz_line_break_ba_k ||
              left_class == sz_line_break_hy_k || left_class == sz_line_break_hh_k))
            resolved = sz_true_k; // LB12a
        if (!resolved && (right_class == sz_line_break_cl_k || right_class == sz_line_break_cp_k ||
                          right_class == sz_line_break_ex_k || right_class == sz_line_break_sy_k))
            resolved = sz_true_k;                                                              // LB13
        if (!resolved && leftmost_non_space_class == sz_line_break_op_k) resolved = sz_true_k; // LB14
        // LB15a: opening Pi quote after an allowed left context, across spaces.
        if (!resolved && leftmost_non_space_class == sz_line_break_qu_k &&
            sz_line_break_descriptor_is_pi_(state.last_non_space_descriptor)) {
            sz_bool_t left_ok;
            if (state.last_non_space_is_first) left_ok = sz_true_k;
            else
                left_ok = (sz_bool_t)(state.last_non_space_left_class == sz_line_break_bk_k ||
                                      state.last_non_space_left_class == sz_line_break_cr_k ||
                                      state.last_non_space_left_class == sz_line_break_lf_k ||
                                      state.last_non_space_left_class == sz_line_break_nl_k ||
                                      state.last_non_space_left_class == sz_line_break_op_k ||
                                      state.last_non_space_left_class == sz_line_break_qu_k ||
                                      state.last_non_space_left_class == sz_line_break_gl_k ||
                                      state.last_non_space_left_class == sz_line_break_sp_k ||
                                      state.last_non_space_left_class == sz_line_break_zw_k);
            if (left_ok) resolved = sz_true_k;
        }
        // LB15b: closing Pf quote followed by an allowed class or end-of-text.
        if (!resolved && right_class == sz_line_break_qu_k && sz_line_break_descriptor_is_pf_(right.descriptor)) {
            sz_bool_t right_ok;
            if (!ahead1.valid) right_ok = sz_true_k;
            else
                right_ok = (sz_bool_t)(ahead1.line_break_class == sz_line_break_sp_k ||
                                       ahead1.line_break_class == sz_line_break_gl_k ||
                                       ahead1.line_break_class == sz_line_break_wj_k ||
                                       ahead1.line_break_class == sz_line_break_cl_k ||
                                       ahead1.line_break_class == sz_line_break_qu_k ||
                                       ahead1.line_break_class == sz_line_break_cp_k ||
                                       ahead1.line_break_class == sz_line_break_ex_k ||
                                       ahead1.line_break_class == sz_line_break_is_k ||
                                       ahead1.line_break_class == sz_line_break_sy_k ||
                                       ahead1.line_break_class == sz_line_break_bk_k ||
                                       ahead1.line_break_class == sz_line_break_cr_k ||
                                       ahead1.line_break_class == sz_line_break_lf_k ||
                                       ahead1.line_break_class == sz_line_break_nl_k ||
                                       ahead1.line_break_class == sz_line_break_zw_k);
            if (right_ok) resolved = sz_true_k;
        }
        if (!resolved && left_class == sz_line_break_sp_k && right_class == sz_line_break_is_k && ahead1.valid &&
            ahead1.line_break_class == sz_line_break_nu_k) {
            is_break = sz_true_k;
            resolved = sz_true_k;
        } // LB15.3
        if (!resolved && right_class == sz_line_break_is_k) resolved = sz_true_k; // LB15.4
        if (!resolved &&
            (leftmost_non_space_class == sz_line_break_cl_k || leftmost_non_space_class == sz_line_break_cp_k) &&
            right_class == sz_line_break_ns_k)
            resolved = sz_true_k; // LB16
        if (!resolved && leftmost_non_space_class == sz_line_break_b2_k && right_class == sz_line_break_b2_k)
            resolved = sz_true_k; // LB17
        if (!resolved && left_class == sz_line_break_sp_k) {
            is_break = sz_true_k;
            resolved = sz_true_k;
        } // LB18
        // LB19 (East-Asian-aware quotation).
        if (!resolved && right_class == sz_line_break_qu_k && !sz_line_break_descriptor_is_pi_(right.descriptor))
            resolved = sz_true_k;
        if (!resolved && left_class == sz_line_break_qu_k && !sz_line_break_descriptor_is_pf_(left.descriptor))
            resolved = sz_true_k;
        if (!resolved && right_class == sz_line_break_qu_k && !sz_line_break_descriptor_is_eaw_(left.descriptor))
            resolved = sz_true_k;
        if (!resolved && right_class == sz_line_break_qu_k &&
            (!ahead1.valid || !sz_line_break_descriptor_is_eaw_(ahead1.descriptor)))
            resolved = sz_true_k;
        if (!resolved && left_class == sz_line_break_qu_k && !sz_line_break_descriptor_is_eaw_(right.descriptor))
            resolved = sz_true_k;
        if (!resolved && left_class == sz_line_break_qu_k &&
            (!left_has_predecessor || !sz_line_break_descriptor_is_eaw_(previous2.descriptor)))
            resolved = sz_true_k;
        if (!resolved && (right_class == sz_line_break_cb_k || left_class == sz_line_break_cb_k)) {
            is_break = sz_true_k;
            resolved = sz_true_k;
        } // LB20
        // LB20a: (sot|allowed) (HY|HH) × (AL|HL).
        if (!resolved && (left_class == sz_line_break_hy_k || left_class == sz_line_break_hh_k) &&
            (right_class == sz_line_break_al_k || right_class == sz_line_break_hl_k)) {
            sz_bool_t left_context_ok;
            if (!left_has_predecessor) left_context_ok = sz_true_k;
            else
                left_context_ok = (sz_bool_t)(previous2.line_break_class == sz_line_break_bk_k ||
                                              previous2.line_break_class == sz_line_break_cr_k ||
                                              previous2.line_break_class == sz_line_break_lf_k ||
                                              previous2.line_break_class == sz_line_break_nl_k ||
                                              previous2.line_break_class == sz_line_break_sp_k ||
                                              previous2.line_break_class == sz_line_break_zw_k ||
                                              previous2.line_break_class == sz_line_break_cb_k ||
                                              previous2.line_break_class == sz_line_break_gl_k);
            if (left_context_ok) resolved = sz_true_k;
        }
        if (!resolved && (right_class == sz_line_break_ba_k || right_class == sz_line_break_hy_k ||
                          right_class == sz_line_break_hh_k || right_class == sz_line_break_ns_k ||
                          left_class == sz_line_break_bb_k))
            resolved = sz_true_k; // LB21
        // LB21a: HL (HY|HH) × [^HL] — the HL precedes the (HY|HH) cluster.
        if (!resolved && left_has_predecessor && previous2.line_break_class == sz_line_break_hl_k &&
            (left_class == sz_line_break_hy_k || left_class == sz_line_break_hh_k) && right_class != sz_line_break_hl_k)
            resolved = sz_true_k;
        if (!resolved && left_class == sz_line_break_sy_k && right_class == sz_line_break_hl_k)
            resolved = sz_true_k;                                                 // LB21b
        if (!resolved && right_class == sz_line_break_in_k) resolved = sz_true_k; // LB22
        if (!resolved && (left_class == sz_line_break_al_k || left_class == sz_line_break_hl_k) &&
            right_class == sz_line_break_nu_k)
            resolved = sz_true_k; // LB23
        if (!resolved && left_class == sz_line_break_nu_k &&
            (right_class == sz_line_break_al_k || right_class == sz_line_break_hl_k))
            resolved = sz_true_k;
        if (!resolved && left_class == sz_line_break_pr_k &&
            (right_class == sz_line_break_id_k || right_class == sz_line_break_eb_k ||
             right_class == sz_line_break_em_k))
            resolved = sz_true_k; // LB23a
        if (!resolved &&
            (left_class == sz_line_break_id_k || left_class == sz_line_break_eb_k ||
             left_class == sz_line_break_em_k) &&
            right_class == sz_line_break_po_k)
            resolved = sz_true_k;
        if (!resolved && (left_class == sz_line_break_pr_k || left_class == sz_line_break_po_k) &&
            (right_class == sz_line_break_al_k || right_class == sz_line_break_hl_k))
            resolved = sz_true_k; // LB24
        if (!resolved && (left_class == sz_line_break_al_k || left_class == sz_line_break_hl_k) &&
            (right_class == sz_line_break_pr_k || right_class == sz_line_break_po_k))
            resolved = sz_true_k;
        // LB25: numeric clusters, with the preceding "NU (SY|IS)*" run carried in `numeric_run_open` (run open at
        // `left`) and `numeric_run_open_before` (run open at `previous2`, the state before a CL/CP left context).
        if (!resolved) {
            sz_bool_t const has_preceding_numeric_run = state.numeric_run_open;
            sz_bool_t has_preceding_numeric_run_then_close = sz_false_k;
            if (left_class == sz_line_break_cl_k || left_class == sz_line_break_cp_k)
                has_preceding_numeric_run_then_close = state.numeric_run_open_before;
            if (!resolved && has_preceding_numeric_run_then_close &&
                (right_class == sz_line_break_po_k || right_class == sz_line_break_pr_k))
                resolved = sz_true_k;
            if (!resolved && has_preceding_numeric_run &&
                (right_class == sz_line_break_po_k || right_class == sz_line_break_pr_k))
                resolved = sz_true_k;
            if (!resolved && left_class == sz_line_break_po_k && right_class == sz_line_break_op_k && ahead1.valid &&
                ahead1.line_break_class == sz_line_break_nu_k)
                resolved = sz_true_k;
            if (!resolved && left_class == sz_line_break_po_k && right_class == sz_line_break_op_k && ahead2.valid &&
                ahead1.line_break_class == sz_line_break_is_k && ahead2.line_break_class == sz_line_break_nu_k)
                resolved = sz_true_k;
            if (!resolved && left_class == sz_line_break_po_k && right_class == sz_line_break_nu_k)
                resolved = sz_true_k;
            if (!resolved && left_class == sz_line_break_pr_k && right_class == sz_line_break_op_k && ahead1.valid &&
                ahead1.line_break_class == sz_line_break_nu_k)
                resolved = sz_true_k;
            if (!resolved && left_class == sz_line_break_pr_k && right_class == sz_line_break_op_k && ahead2.valid &&
                ahead1.line_break_class == sz_line_break_is_k && ahead2.line_break_class == sz_line_break_nu_k)
                resolved = sz_true_k;
            if (!resolved && left_class == sz_line_break_pr_k && right_class == sz_line_break_nu_k)
                resolved = sz_true_k;
            if (!resolved && left_class == sz_line_break_hy_k && right_class == sz_line_break_nu_k)
                resolved = sz_true_k;
            if (!resolved && left_class == sz_line_break_is_k && right_class == sz_line_break_nu_k)
                resolved = sz_true_k;
            if (!resolved && has_preceding_numeric_run && right_class == sz_line_break_nu_k) resolved = sz_true_k;
        }
        // LB26 / LB27 Hangul.
        if (!resolved && left_class == sz_line_break_jl_k &&
            (right_class == sz_line_break_jl_k || right_class == sz_line_break_jv_k ||
             right_class == sz_line_break_h2_k || right_class == sz_line_break_h3_k))
            resolved = sz_true_k;
        if (!resolved && (left_class == sz_line_break_jv_k || left_class == sz_line_break_h2_k) &&
            (right_class == sz_line_break_jv_k || right_class == sz_line_break_jt_k))
            resolved = sz_true_k;
        if (!resolved && (left_class == sz_line_break_jt_k || left_class == sz_line_break_h3_k) &&
            right_class == sz_line_break_jt_k)
            resolved = sz_true_k;
        if (!resolved &&
            (left_class == sz_line_break_jl_k || left_class == sz_line_break_jv_k || left_class == sz_line_break_jt_k ||
             left_class == sz_line_break_h2_k || left_class == sz_line_break_h3_k) &&
            right_class == sz_line_break_po_k)
            resolved = sz_true_k;
        if (!resolved && left_class == sz_line_break_pr_k &&
            (right_class == sz_line_break_jl_k || right_class == sz_line_break_jv_k ||
             right_class == sz_line_break_jt_k || right_class == sz_line_break_h2_k ||
             right_class == sz_line_break_h3_k))
            resolved = sz_true_k;
        if (!resolved && (left_class == sz_line_break_al_k || left_class == sz_line_break_hl_k) &&
            (right_class == sz_line_break_al_k || right_class == sz_line_break_hl_k))
            resolved = sz_true_k; // LB28
        // LB28a: aksara clusters; Dotted_Circle acts as an aksara base.
        if (!resolved) {
            sz_bool_t const left_is_dotted_circle = sz_line_break_descriptor_is_dotted_circle_(left.descriptor);
            sz_bool_t const right_is_dotted_circle = sz_line_break_descriptor_is_dotted_circle_(right.descriptor);
            sz_bool_t const left_is_base = (sz_bool_t)(left_class == sz_line_break_ak_k ||
                                                       left_class == sz_line_break_as_k || left_is_dotted_circle);
            sz_bool_t const right_is_base = (sz_bool_t)(right_class == sz_line_break_ak_k ||
                                                        right_class == sz_line_break_as_k || right_is_dotted_circle);
            sz_bool_t const right_base_or_dotted_circle = (sz_bool_t)(right_class == sz_line_break_ak_k ||
                                                                      right_is_dotted_circle);
            if (!resolved && left_class == sz_line_break_ap_k && right_is_base) resolved = sz_true_k; // 28.11
            if (!resolved && left_is_base && (right_class == sz_line_break_vf_k || right_class == sz_line_break_vi_k))
                resolved = sz_true_k; // 28.12
            if (!resolved && left_class == sz_line_break_vi_k && right_base_or_dotted_circle &&
                left_has_predecessor) { // 28.13
                if (previous2.line_break_class == sz_line_break_ak_k ||
                    previous2.line_break_class == sz_line_break_as_k ||
                    sz_line_break_descriptor_is_dotted_circle_(previous2.descriptor))
                    resolved = sz_true_k;
            }
            if (!resolved && left_is_base && right_is_base && ahead1.valid &&
                ahead1.line_break_class == sz_line_break_vf_k)
                resolved = sz_true_k; // 28.14
        }
        if (!resolved && left_class == sz_line_break_is_k &&
            (right_class == sz_line_break_al_k || right_class == sz_line_break_hl_k))
            resolved = sz_true_k; // LB29
        // LB30.
        if (!resolved &&
            (left_class == sz_line_break_al_k || left_class == sz_line_break_hl_k ||
             left_class == sz_line_break_nu_k) &&
            right_class == sz_line_break_op_k && !sz_line_break_descriptor_is_eaw_(right.descriptor))
            resolved = sz_true_k;
        if (!resolved && left_class == sz_line_break_cp_k &&
            (right_class == sz_line_break_al_k || right_class == sz_line_break_hl_k ||
             right_class == sz_line_break_nu_k) &&
            !sz_line_break_descriptor_is_eaw_(left.descriptor))
            resolved = sz_true_k;
        // LB30a: RI RI keep pairs — an odd Regional_Indicator run length keeps the pair.
        if (!resolved && left_class == sz_line_break_ri_k && right_class == sz_line_break_ri_k &&
            state.regional_indicator_parity_odd)
            resolved = sz_true_k;
        // LB30b.
        if (!resolved && left_class == sz_line_break_eb_k && right_class == sz_line_break_em_k) resolved = sz_true_k;
        if (!resolved && right_class == sz_line_break_em_k && sz_line_break_descriptor_is_extpict_cn_(left.descriptor))
            resolved = sz_true_k;

        if (!resolved) is_break = sz_true_k; // LB31

        if (is_break) {
            if (lines == lines_capacity) {
                if (bytes_consumed) *bytes_consumed = open_line_start;
                return lines;
            }
            line_starts[lines] = open_line_start;
            line_lengths[lines] = right.byte_start - open_line_start;
            ++lines;
            open_line_start = right.byte_start;
        }

        // Advance the forward-carried run state by `right`, then slide the five-cluster context forward by one and
        // decode the new look-ahead tail.
        sz_line_break_serial_advance_(&state, &right, left_class);
        previous2 = left;
        left = right;
        right = ahead1;
        ahead1 = ahead2;
        ahead2 = sz_line_break_next_cluster_(text, length, &position, &state.last_codepoint_was_zwj);
    }

    // The trailing span [open_line_start, length) is the final line; end of text is a boundary (LB3).
    if (lines == lines_capacity) {
        if (bytes_consumed) *bytes_consumed = open_line_start;
        return lines;
    }
    line_starts[lines] = open_line_start;
    line_lengths[lines] = length - open_line_start;
    ++lines;
    if (bytes_consumed) *bytes_consumed = length;
    return lines;
}

#pragma endregion // UAX 14 Line Boundaries

#pragma region UAX 14 Line Boundaries window engine

/** @brief Number of distinct UAX-14 Line_Break classes (`xx_k` .. `hh_k`); array stride for per-class membership. */
enum { sz_line_break_class_count_k = 49 };

/** @brief Engine side-word bits, one per descriptor flag the LB rules read. */
enum {
    sz_line_break_side_pi_k = 0x01,     /**< gc=Pi (LB15a/LB19) */
    sz_line_break_side_pf_k = 0x02,     /**< gc=Pf (LB15b/LB19) */
    sz_line_break_side_eaw_k = 0x04,    /**< East-Asian Width F/W/H (LB19/LB30) */
    sz_line_break_side_cn_k = 0x08,     /**< unassigned (LB30b, only in conjunction with ExtPict) */
    sz_line_break_side_ext_k = 0x10,    /**< Extended_Pictographic (LB30b) */
    sz_line_break_side_ri_k = 0x20,     /**< raw RI class (LB30a parity) */
    sz_line_break_side_zwj_k = 0x40,    /**< raw ZWJ class (LB8a) */
    sz_line_break_side_mark_k = 0x80,   /**< resolved CM|ZWJ (LB9/LB10 attachment) */
    sz_line_break_side_dotted_k = 0x100 /**< DottedCircle U+25CC (LB28a); above the byte side-word */
};

/** @brief Window state carried across the 64-lane block window_edge: the left context + open runs straddling into
 *         lane 0, so the next window decides its first cluster's break-before with NO byte re-read (no overlap). */
typedef struct sz_line_break_carry_t {
    sz_u8_t have_prev; /**< 0 only at start-of-text (LB2); 1 once a cluster precedes lane 0 */
    sz_u64_t
        previous_class_bit; /**< one-hot effective LB class of the cluster ending just before lane 0 (post LB9/10) */
    sz_u64_t
        previous2_class_bit; /**< one-hot LB class of the cluster two before lane 0 (2-left rules LB19/LB20a/LB21a) */
    sz_u8_t left_eaw;        /**< the cluster before lane 0 is East-Asian F/W/H (LB19 side bit) */
    sz_u8_t left2_eaw;       /**< the cluster two before lane 0 is East-Asian F/W/H (LB19 ~prev2EAW) */
    sz_u8_t left_pf;         /**< the cluster before lane 0 carries gc=Pf (LB19 `prev_(QU & ~PF)`) */
    sz_u8_t left_aksara;     /**< the cluster before lane 0 is an aksara base AK|AS|DottedCircle (LB28a) */
    sz_u8_t left2_aksara;    /**< the cluster two before lane 0 is an aksara base (LB28a `prev_(base_vi)`) */
    sz_u8_t left_extpict_cn; /**< the cluster before lane 0 is Extended_Pictographic & unassigned (LB30b) */
    sz_u8_t prev_is_zwj;     /**< the cluster before lane 0 is a bare ZWJ (LB8a no-break-after) */
    sz_u8_t open_sp_opener;  /**< class anchoring an open "X SP*" run (LB14/16/17/8); 0xFF = none */
    sz_u8_t in_nu_run;       /**< lane 0 continues a "NU (SY|IS)*" run (LB25) */
    sz_u8_t in_nu_close;     /**< the cluster before lane 0 is a CL/CP closing a NU run (LB25 nu_run_close) */
    sz_u8_t ri_open;         /**< lane 0 continues an RI sequence (parity carries) */
    sz_u8_t ri_parity_odd;   /**< odd count of RI characters precede lane 0 within the open sequence */
    sz_u8_t qupi_sp_open;    /**< lane 0 continues an open "[QU&Pi] SP*" governed run (LB15a) */
} sz_line_break_carry_t;

/** @brief Is the carried one-hot class word @p class_bits set for class `cls`? `class_bits` is hoisted once per
 *         window into a register (0 at start-of-text), so every call is a pure register shift/and the compiler CSEs. */
SZ_HELPER_INLINE sz_bool_t sz_line_break_class_is_(sz_u64_t class_bits, sz_u8_t cls) {
    return (sz_bool_t)((class_bits >> cls) & 1ull);
}

/** @brief Is the carried one-hot class word @p class_bits a member of @p class_set? */
SZ_HELPER_INLINE sz_bool_t sz_line_break_class_in_(sz_u64_t class_bits, sz_u64_t class_set) {
    return (sz_bool_t)((class_bits & class_set) != 0);
}

/** @brief Result of one window decision: break-before bits + the trust horizon (lanes [0,resolved) are committable). */
typedef struct sz_line_break_window_t {
    sz_u64_t breaks;    /**< break-before bits at cluster-base lanes */
    sz_size_t resolved; /**< lowest lane whose verdict needs context past the window edge; commit below this */
} sz_line_break_window_t;

/** @brief Start-of-text carry: no previous cluster, all runs closed. */
SZ_HELPER_INLINE sz_line_break_carry_t sz_line_break_carry_sot_(void) {
    sz_line_break_carry_t carry;
    carry.have_prev = 0, carry.previous_class_bit = 1ull << sz_line_break_xx_k,
    carry.previous2_class_bit = 1ull << sz_line_break_xx_k;
    carry.left_eaw = 0, carry.left2_eaw = 0, carry.left_pf = 0, carry.left_aksara = 0, carry.left2_aksara = 0,
    carry.left_extpict_cn = 0;
    carry.prev_is_zwj = 0, carry.open_sp_opener = 0xFF, carry.in_nu_run = 0, carry.in_nu_close = 0, carry.ri_open = 0,
    carry.ri_parity_odd = 0, carry.qupi_sp_open = 0;
    return carry;
}

/** @brief  Largest byte prefix of a decode window whose codepoints are all fully loaded, from the plain u64 lane
 *          masks - the mask-domain twin of the per-ISA `complete_limit` helpers, for back-ends that carry their
 *          window state as scalars. Never below 1 when the window is non-empty. */
SZ_HELPER_AUTO sz_size_t sz_line_break_complete_limit_masks_(sz_size_t loaded, sz_u64_t start_bytes,
                                                             sz_u64_t two_byte_starts, sz_u64_t three_byte_starts,
                                                             sz_u64_t four_byte_starts, sz_bool_t more_text) {
    if (!more_text) return loaded;
    sz_u64_t const straddle = ((two_byte_starts & ~sz_u64_mask_until_serial_(loaded > 1 ? loaded - 1 : 0)) |
                               (three_byte_starts & ~sz_u64_mask_until_serial_(loaded > 2 ? loaded - 2 : 0)) |
                               (four_byte_starts & ~sz_u64_mask_until_serial_(loaded > 3 ? loaded - 3 : 0))) &
                              start_bytes;
    sz_size_t const limit = straddle ? (sz_size_t)sz_u64_ctz(straddle) : loaded;
    return limit > 0 ? limit : loaded;
}

/** @brief Per-lane class/side membership of one decoded 64-byte window, precomputed by a per-ISA extractor so the
 *         portable rule engine sources every mask from `sz_u64_t` words without touching the codepoint vectors. */
typedef struct sz_line_break_frame_t {
    sz_u64_t base, gate, attached, lone_mark;               /**< from the byte-level cluster frame */
    sz_u64_t non_start, dotted, starts, replacement;        /**< from the classifier */
    sz_u64_t effective_class[sz_line_break_class_count_k];  /**< membership per class, AFTER LB10 lone->AL; NOT &base */
    sz_u64_t raw_zwj;                                       /**< class_mask(classified.classes, zwj_k), pre-effective */
    sz_u64_t side_pi, side_pf, side_eaw, side_cn, side_ext; /**< side-bit membership masks (NOT yet &base) */
} sz_line_break_frame_t;

/** @brief OR-reduction of the per-class membership masks over the inclusive class range [@p lo, @p hi]; the portable
 *         twin of the icelake `class_range_mask` byte-range compare used to cheaply gate the script blocks. */
SZ_HELPER_AUTO sz_u64_t sz_line_break_effective_range_(sz_line_break_frame_t const *frame, sz_u8_t lo, sz_u8_t hi) {
    sz_u64_t accumulated = 0;
    for (sz_u8_t cls = lo; cls <= hi; ++cls) accumulated |= frame->effective_class[cls];
    return accumulated;
}

/** @brief Previous-cluster mask: bit k set => the cluster ending just before lane k has a base in @p class_base. */
SZ_HELPER_INLINE sz_u64_t sz_line_break_prev_(sz_u64_t class_base, sz_u64_t gate) {
    return sz_u64_fill_right_(class_base, gate) << 1;
}

/** @brief Next-cluster mask: bit k set => the cluster starting just after lane k has a base in @p class_base. */
SZ_HELPER_INLINE sz_u64_t sz_line_break_next_(sz_u64_t class_base, sz_u64_t gate) {
    return sz_u64_fill_left_(class_base, gate) >> 1;
}

/** @brief Carry-aware previous-cluster mask: like @ref sz_line_break_prev_, but additionally marks @p edge (lane 0)
 *         when the carried left cluster matches (@p left_in_set), so cross-window left context needs no byte re-read. */
SZ_HELPER_INLINE sz_u64_t sz_line_break_prevc_(sz_u64_t class_base, sz_u64_t gate, sz_bool_t left_in_set,
                                               sz_u64_t edge) {
    return (sz_u64_fill_right_(class_base, gate) << 1) | (left_in_set ? edge : 0);
}

/** @brief Commit a forced break at the undecided lanes of @p where (sets break + settled). */
SZ_HELPER_INLINE void sz_line_break_force_break_(sz_u64_t where, sz_u64_t all, sz_u64_t *settled, sz_u64_t *breaks) {
    sz_u64_t const w = where & ~*settled & all;
    *breaks |= w, *settled |= w;
}

/** @brief Commit a forced join (no break) at the undecided lanes of @p where (sets settled only). */
SZ_HELPER_INLINE void sz_line_break_force_join_(sz_u64_t where, sz_u64_t all, sz_u64_t *settled) {
    *settled |= where & ~*settled & all;
}

/** @brief Opener-governed "X SP*" run over byte-start lanes (LB8/14/16/17): flood the opener rightward across the
 *         transparent gate and the space bases; the result marks the opener + the governed space base lanes. */
SZ_HELPER_INLINE sz_u64_t sz_line_break_run_byte_(sz_u64_t opener, sz_u64_t spaces, sz_u64_t gate) {
    return sz_u64_fill_right_(opener, gate | spaces) & (opener | spaces);
}

/** @brief Inclusive segmented prefix-XOR of @p members over each contiguous @p run_gate run (LB30a RI parity); when
 *         @p inbound_parity, an odd virtual member below lane 0 toggles the whole leading run. Returns the per-lane
 *         inclusive parity. The prefix-XOR scan is the shared substrate primitive; this wrapper adds the RI-pairing
 *         inbound-run seed (mirrors the word kernel's `ri_join_`). */
SZ_HELPER_INLINE sz_u64_t sz_line_break_segmented_parity_(sz_u64_t members, sz_u64_t run_gate,
                                                          sz_bool_t inbound_parity) {
    sz_u64_t bits = sz_u64_segmented_parity_(members, run_gate);
    if (inbound_parity) bits ^= sz_u64_fill_right_(run_gate & 1ull, run_gate);
    return bits;
}

/**
 *  @brief  Portable byte-level UAX-14 rule engine: decide break-before bits at the cluster-base lanes of one decoded
 *          window. Every LB1-LB31 rule is expressed over byte-start lanes as `sz_u64_t` bit-mask algebra, reading
 *          effective neighbours across `gate` via `sz_line_break_prev_`/`_next_` (no scalar collapse, no intrinsics).
 *          All per-lane class/side membership is supplied precomputed in @p frame by a per-ISA extractor. The returned
 *          mask has bits only at base lanes (a break opportunity precedes that cluster).
 */
SZ_HELPER_INLINE sz_line_break_window_t sz_line_break_decide_window_(
    sz_line_break_frame_t const *frame, sz_u8_t const *effective_class_byte, sz_u8_t const *side_byte,
    sz_line_break_carry_t carry, sz_line_break_carry_t *carry_out, sz_size_t complete_limit, sz_bool_t more_text) {
    sz_u64_t const base = frame->base, gate = frame->gate, non_start = frame->non_start;
    sz_line_break_window_t empty;
    empty.breaks = 0, empty.resolved = complete_limit;
    if (!base) {
        //  All-ignorable window: nothing to decide, thread the inbound carry forward unchanged.
        if (carry_out) *carry_out = carry;
        return empty;
    }
    sz_bool_t const is_sot = (sz_bool_t)(!carry.have_prev);
    sz_bool_t const is_eot = (sz_bool_t)(!more_text);
    //  Hoist the carried one-hot class words into registers once. At start-of-text (`!have_prev`) they read as 0, so
    //  every `sz_line_break_class_is_/in_` test folds `have_prev` in and the repeated shifts/ands CSE in-register.
    sz_u64_t const left_bit = carry.have_prev ? carry.previous_class_bit : 0ull;
    sz_u64_t const left2_bit = carry.have_prev ? carry.previous2_class_bit : 0ull;
    sz_u64_t const first_base = base & (0ull - base);
    sz_u64_t const last_base = 1ull << (63 - sz_u64_clz(base));
    //  When a cluster precedes lane 0 (carried left context), lane 0's break-before is decided by the rules just like
    //  any interior lane; only at true start-of-text is lane 0 exempt (LB2).
    sz_u64_t const interior_lanes = carry.have_prev ? base : (base & ~first_base);
    sz_u64_t const edge = carry.have_prev ? first_base : 0;
    sz_u64_t const start_of_text = is_sot ? first_base : 0;
    sz_u64_t const end_of_text = is_eot ? last_base : 0;
    sz_u64_t const dotted_circle = frame->dotted & base & ~frame->lone_mark;

    //  The carried-left class is now read on demand through `sz_line_break_left_is_` / `sz_line_break_left_in_` (and the
    //  `left2_` variants) at each rule's call site, so the per-class boolean wall is gone. Only the East-Asian-width
    //  side bits, which live outside the class field, keep dedicated booleans here.
    sz_bool_t const left_is_east_asian_width = (sz_bool_t)(carry.have_prev && carry.left_eaw != 0);
    sz_bool_t const left2_is_east_asian_width = (sz_bool_t)(carry.have_prev && carry.left2_eaw != 0);
    //  Shared class sets for the two-left LB20a allow-list and (below) the one-left LB20a allow-list. Listing the SAME
    //  classes the original boolean tested: BK, CR, LF, NL, SP, ZW, CB, GL.
    sz_u64_t const lb20a_allowed_mask = (1ull << sz_line_break_bk_k) | (1ull << sz_line_break_cr_k) |
                                        (1ull << sz_line_break_lf_k) | (1ull << sz_line_break_nl_k) |
                                        (1ull << sz_line_break_sp_k) | (1ull << sz_line_break_zw_k) |
                                        (1ull << sz_line_break_cb_k) | (1ull << sz_line_break_gl_k);
    sz_bool_t const left2_is_lb20a_allowed = sz_line_break_class_in_(left2_bit, lb20a_allowed_mask);

    sz_u64_t const class_break_mandatory = frame->effective_class[sz_line_break_bk_k] & base;
    sz_u64_t const class_carriage_return = frame->effective_class[sz_line_break_cr_k] & base;
    sz_u64_t const class_line_feed = frame->effective_class[sz_line_break_lf_k] & base;
    sz_u64_t const class_next_line = frame->effective_class[sz_line_break_nl_k] & base;
    sz_u64_t const class_space = frame->effective_class[sz_line_break_sp_k] & base;
    sz_u64_t const class_zero_width_space = frame->effective_class[sz_line_break_zw_k] & base;
    sz_u64_t const class_word_joiner = frame->effective_class[sz_line_break_wj_k] & base;
    sz_u64_t const class_glue = frame->effective_class[sz_line_break_gl_k] & base;
    sz_u64_t const class_open_punctuation = frame->effective_class[sz_line_break_op_k] & base;
    sz_u64_t const class_close_punctuation = frame->effective_class[sz_line_break_cl_k] & base;
    sz_u64_t const class_close_parenthesis = frame->effective_class[sz_line_break_cp_k] & base;
    sz_u64_t const class_quotation = frame->effective_class[sz_line_break_qu_k] & base;
    sz_u64_t const class_nonstarter = frame->effective_class[sz_line_break_ns_k] & base;
    sz_u64_t const class_break_both = frame->effective_class[sz_line_break_b2_k] & base;
    sz_u64_t const class_hyphen = frame->effective_class[sz_line_break_hy_k] & base;
    sz_u64_t const class_unambiguous_hyphen = frame->effective_class[sz_line_break_hh_k] & base;
    sz_u64_t const class_break_after = frame->effective_class[sz_line_break_ba_k] & base;
    sz_u64_t const class_break_before = frame->effective_class[sz_line_break_bb_k] & base;
    sz_u64_t const class_inseparable = frame->effective_class[sz_line_break_in_k] & base;
    sz_u64_t const class_contingent_break = frame->effective_class[sz_line_break_cb_k] & base;
    sz_u64_t const class_infix_separator = frame->effective_class[sz_line_break_is_k] & base;
    sz_u64_t const class_numeric = frame->effective_class[sz_line_break_nu_k] & base;
    sz_u64_t const class_symbol = frame->effective_class[sz_line_break_sy_k] & base;
    sz_u64_t const class_exclamation = frame->effective_class[sz_line_break_ex_k] & base;
    sz_u64_t const class_postfix_numeric = frame->effective_class[sz_line_break_po_k] & base;
    sz_u64_t const class_prefix_numeric = frame->effective_class[sz_line_break_pr_k] & base;
    sz_u64_t const class_alphabetic = frame->effective_class[sz_line_break_al_k] & base;
    sz_u64_t const class_hebrew_letter = frame->effective_class[sz_line_break_hl_k] & base;
    sz_u64_t const class_ideographic = frame->effective_class[sz_line_break_id_k] & base;
    sz_u64_t const class_emoji_base = frame->effective_class[sz_line_break_eb_k] & base;
    sz_u64_t const class_emoji_modifier = frame->effective_class[sz_line_break_em_k] & base;
    sz_u64_t const class_regional_indicator = frame->effective_class[sz_line_break_ri_k] & base;
    //  class_aksara and class_aksara_start are also read by the cross-window carry-out below, so they stay at the top
    //  even though their rules live in the gated LB28a block. The remaining Brahmic masks (AP/VF/VI) and all five
    //  Hangul masks (JL/JV/JT/H2/H3) are used only inside their respective gated blocks and are extracted there.
    sz_u64_t const class_aksara = frame->effective_class[sz_line_break_ak_k] & base;
    sz_u64_t const class_aksara_start = frame->effective_class[sz_line_break_as_k] & base;
    sz_u64_t const zwj_starts = frame->raw_zwj & frame->starts;

    sz_u64_t const side_quote_initial = frame->side_pi & base;
    sz_u64_t const side_quote_final = frame->side_pf & base;
    sz_u64_t const side_east_asian_width = frame->side_eaw & base;

    sz_u64_t settled = 0, breaks = 0;
    sz_bool_t const carry_op = (sz_bool_t)(carry.open_sp_opener == sz_line_break_op_k);
    sz_bool_t const carry_clcp = (sz_bool_t)(carry.open_sp_opener == sz_line_break_cl_k ||
                                             carry.open_sp_opener == sz_line_break_cp_k);
    sz_bool_t const carry_b2 = (sz_bool_t)(carry.open_sp_opener == sz_line_break_b2_k);
    sz_bool_t const carry_zw = (sz_bool_t)(carry.open_sp_opener == sz_line_break_zw_k);
    //  "X SP*" run governance (LB8/14/16/17). The fill_right floods are skipped whenever no opener is present in the
    //  window AND none is carried open across the edge -- the dominant CJK/plain-text case -- leaving every gov empty
    //  (bit-identical, since prev_(0) contributes nothing and carry-out then finds no opener).
    sz_u64_t opener_governance = 0, close_governance = 0, break_both_governance = 0, zero_width_governance = 0;
    if ((class_open_punctuation | class_close_punctuation | class_close_parenthesis | class_break_both |
         class_zero_width_space) ||
        carry_op || carry_clcp || carry_b2 || carry_zw) {
        sz_u64_t const op_seed = class_open_punctuation | (carry_op && (class_space & first_base) ? first_base : 0);
        sz_u64_t const clcp_seed = (class_close_punctuation | class_close_parenthesis) |
                                   (carry_clcp && (class_space & first_base) ? first_base : 0);
        sz_u64_t const b2_seed = class_break_both | (carry_b2 && (class_space & first_base) ? first_base : 0);
        sz_u64_t const zw_seed = class_zero_width_space | (carry_zw && (class_space & first_base) ? first_base : 0);
        //  The four "X SP*" floods (LB8/14/16/17) share one transparent reach `gate | spaces`: hoist the Kogge-Stone
        //  reach recurrence once and step the four seeds together. Equivalent to four `sz_line_break_run_byte_` calls.
        sz_u64_t const flood_gate = gate | class_space;
        sz_u64_t opener_bits = op_seed, close_bits = clcp_seed, break_both_bits = b2_seed, zero_width_bits = zw_seed,
                 reach = flood_gate;
        for (int shift = 1; shift < 64; shift <<= 1) {
            opener_bits |= (opener_bits << shift) & reach;
            close_bits |= (close_bits << shift) & reach;
            break_both_bits |= (break_both_bits << shift) & reach;
            zero_width_bits |= (zero_width_bits << shift) & reach;
            reach &= reach << shift;
        }
        opener_governance = opener_bits & (op_seed | class_space);
        close_governance = close_bits & (clcp_seed | class_space);
        break_both_governance = break_both_bits & (b2_seed | class_space);
        zero_width_governance = zero_width_bits & (zw_seed | class_space);
    }
    //  LB15a "(sot|allowed) [QU&Pi] SP* x" governance, seeded by a carried open QU·Pi run when lane 0 continues it.
    sz_u64_t const quote_initial = class_quotation & side_quote_initial;
    sz_u64_t quote_initial_governance = 0;
    if (quote_initial || carry.qupi_sp_open) {
        sz_u64_t const qupi_allowed_left = class_break_mandatory | class_carriage_return | class_line_feed |
                                           class_next_line | class_open_punctuation | class_quotation | class_glue |
                                           class_space | class_zero_width_space;
        //  A QU·Pi opening at lane 0 may have its allowed-left cluster carried across the edge (LB15a left context).
        sz_bool_t const carry_qupi_left = (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_bk_k) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_cr_k) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_lf_k) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_nl_k) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_op_k) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_qu_k) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_gl_k) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_sp_k) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_zw_k));
        sz_u64_t qupi_seed = quote_initial &
                             (sz_line_break_prevc_(qupi_allowed_left, gate, carry_qupi_left, edge) | start_of_text);
        if (carry.qupi_sp_open && (class_space & first_base)) qupi_seed |= first_base;
        quote_initial_governance = sz_line_break_run_byte_(qupi_seed, class_space, gate);
    }

    // LB4: BK !
    sz_line_break_force_break_(
        sz_line_break_prevc_(class_break_mandatory, gate, sz_line_break_class_is_(left_bit, sz_line_break_bk_k), edge) &
            interior_lanes,
        base, &settled, &breaks);
    // LB5: CR x LF ; CR ! ; LF ! ; NL !
    sz_line_break_force_join_((sz_line_break_prevc_(class_carriage_return, gate,
                                                    sz_line_break_class_is_(left_bit, sz_line_break_cr_k), edge) &
                               class_line_feed) &
                                  interior_lanes,
                              base, &settled);
    sz_line_break_force_break_(sz_line_break_prevc_(class_carriage_return | class_line_feed | class_next_line, gate,
                                                    (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_cr_k) ||
                                                                sz_line_break_class_is_(left_bit, sz_line_break_lf_k) ||
                                                                sz_line_break_class_is_(left_bit, sz_line_break_nl_k)),
                                                    edge) &
                                   interior_lanes,
                               base, &settled, &breaks);
    // LB6: x (BK|CR|LF|NL)
    sz_line_break_force_join_(
        (class_break_mandatory | class_carriage_return | class_line_feed | class_next_line) & interior_lanes, base,
        &settled);
    // LB7: x SP ; x ZW
    sz_line_break_force_join_((class_space | class_zero_width_space) & interior_lanes, base, &settled);
    // LB8: ZW SP* /
    sz_line_break_force_break_(
        (sz_line_break_prevc_(zero_width_governance, gate, carry_zw, edge) & ~zero_width_governance) & interior_lanes,
        base, &settled, &breaks);
    // LB8a: ZWJ x (codepoint-level: previous codepoint is the original ZWJ class). A carried bare ZWJ joins lane 0.
    {
        sz_u64_t zwj_join_targets = (sz_u64_fill_right_(zwj_starts, non_start) << 1) & base;
        if (carry.prev_is_zwj) zwj_join_targets |= edge;
        sz_line_break_force_join_(zwj_join_targets & interior_lanes, base, &settled);
    }
    // LB9: x (attached CM|ZWJ) -- attached marks are never base lanes, so no break-before exists there (implicit).
    // LB11: x WJ ; WJ x
    sz_line_break_force_join_(
        (class_word_joiner |
         sz_line_break_prevc_(class_word_joiner, gate, sz_line_break_class_is_(left_bit, sz_line_break_wj_k), edge)) &
            interior_lanes,
        base, &settled);
    // LB12: GL x
    sz_line_break_force_join_(
        sz_line_break_prevc_(class_glue, gate, sz_line_break_class_is_(left_bit, sz_line_break_gl_k), edge) &
            interior_lanes,
        base, &settled);
    // LB12a: [^SP BA HY HH] x GL
    sz_u64_t const space_break_hyphen_prev = sz_line_break_prevc_(
        class_space | class_break_after | class_hyphen | class_unambiguous_hyphen, gate,
        (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_sp_k) ||
                    sz_line_break_class_is_(left_bit, sz_line_break_ba_k) ||
                    sz_line_break_class_is_(left_bit, sz_line_break_hy_k) ||
                    sz_line_break_class_is_(left_bit, sz_line_break_hh_k)),
        edge);
    sz_line_break_force_join_((class_glue & ~space_break_hyphen_prev) & interior_lanes, base, &settled);
    // LB13: x CL ; x CP ; x EX ; x SY
    sz_line_break_force_join_(
        (class_close_punctuation | class_close_parenthesis | class_exclamation | class_symbol) & interior_lanes, base,
        &settled);
    // LB14: OP SP* x. A carried open "OP SP*" run governs lane 0 (the opener/space may lie before the window edge).
    sz_line_break_force_join_(sz_line_break_prevc_(opener_governance, gate, carry_op, edge) & interior_lanes, base,
                              &settled);
    // LB15a: (sot|allowed) [QU&Pi] SP* x  (governance hoisted above with the other SP-run openers)
    sz_line_break_force_join_(
        sz_line_break_prevc_(quote_initial_governance, gate, (sz_bool_t)carry.qupi_sp_open, edge) & interior_lanes,
        base, &settled);
    // LB15b: x [QU&Pf] (followed by allowed | eot)
    {
        sz_u64_t const quote_final = class_quotation & side_quote_final;
        sz_u64_t const right_context_set = class_space | class_glue | class_word_joiner | class_close_punctuation |
                                           class_quotation | class_close_parenthesis | class_exclamation |
                                           class_infix_separator | class_symbol | class_break_mandatory |
                                           class_carriage_return | class_line_feed | class_next_line |
                                           class_zero_width_space;
        sz_u64_t const right_ok = sz_line_break_next_(right_context_set, gate) | (quote_final & end_of_text);
        sz_line_break_force_join_((quote_final & right_ok) & interior_lanes, base, &settled);
    }
    // LB15.3: SP / IS NU
    sz_u64_t const space_prev = sz_line_break_prevc_(class_space, gate,
                                                     sz_line_break_class_is_(left_bit, sz_line_break_sp_k), edge);
    sz_line_break_force_break_(
        (space_prev & class_infix_separator & sz_line_break_next_(class_numeric, gate)) & interior_lanes, base,
        &settled, &breaks);
    // LB15.4: x IS
    sz_line_break_force_join_(class_infix_separator & interior_lanes, base, &settled);
    // LB16: (CL|CP) SP* NS
    sz_line_break_force_join_(
        (class_nonstarter & sz_line_break_prevc_(close_governance, gate, carry_clcp, edge)) & interior_lanes, base,
        &settled);
    // LB17: B2 SP* B2
    sz_line_break_force_join_(
        (class_break_both & sz_line_break_prevc_(break_both_governance, gate, carry_b2, edge)) & interior_lanes, base,
        &settled);
    // LB18: SP /
    sz_line_break_force_break_(space_prev & interior_lanes, base, &settled, &breaks);
    // LB19 group (East-Asian-aware quotation); every term reads a QU cluster, so skip when none is present in the
    // window AND the carried left is neither a QU nor an East-Asian cluster that LB19 reads across the edge.
    if (class_quotation || sz_line_break_class_is_(left_bit, sz_line_break_qu_k) || left_is_east_asian_width ||
        left2_is_east_asian_width) {
        sz_u64_t const previous_east_asian_width = sz_line_break_prevc_(side_east_asian_width, gate,
                                                                        left_is_east_asian_width, edge);
        sz_u64_t const next_east_asian_width = sz_line_break_next_(side_east_asian_width, gate);
        sz_u64_t previous2_east_asian_width = sz_line_break_prev_(previous_east_asian_width & base, gate);
        if (left2_is_east_asian_width)
            previous2_east_asian_width |= edge; // lane 0's two-left cluster is the carried left2 (East-Asian)
        sz_u64_t const sot_next = is_sot ? (sz_line_break_prev_(first_base, gate) & base) : 0;
        sz_u64_t const quotation_prev = sz_line_break_prevc_(
            class_quotation, gate, sz_line_break_class_is_(left_bit, sz_line_break_qu_k), edge);
        sz_line_break_force_join_((class_quotation & ~side_quote_initial) & interior_lanes, base, &settled);
        sz_line_break_force_join_(
            sz_line_break_prevc_(class_quotation & ~side_quote_final, gate,
                                 (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_qu_k) && !carry.left_pf),
                                 edge) &
                interior_lanes,
            base, &settled);
        sz_line_break_force_join_((class_quotation & ~previous_east_asian_width) & interior_lanes, base, &settled);
        sz_line_break_force_join_((class_quotation & (~next_east_asian_width | end_of_text)) & interior_lanes, base,
                                  &settled);
        sz_line_break_force_join_((quotation_prev & ~side_east_asian_width) & interior_lanes, base, &settled);
        sz_line_break_force_join_((quotation_prev & (~previous2_east_asian_width | sot_next)) & interior_lanes, base,
                                  &settled);
    }
    // LB20: / CB ; CB /
    sz_line_break_force_break_(
        (class_contingent_break | sz_line_break_prevc_(class_contingent_break, gate,
                                                       sz_line_break_class_is_(left_bit, sz_line_break_cb_k), edge)) &
            interior_lanes,
        base, &settled, &breaks);
    // LB20a: (sot|allowed) (HY|HH) x (AL|HL); fires when a HY/HH cluster precedes an AL|HL, in-window or carried.
    if ((class_hyphen | class_unambiguous_hyphen) || sz_line_break_class_is_(left_bit, sz_line_break_hy_k) ||
        sz_line_break_class_is_(left_bit, sz_line_break_hh_k)) {
        sz_u64_t const leftset = class_break_mandatory | class_carriage_return | class_line_feed | class_next_line |
                                 class_space | class_zero_width_space | class_contingent_break | class_glue;
        //  The HY/HH cluster qualifies when preceded by an allowed-left cluster (or sot). At lane 0 the HY/HH is the
        //  carried left, so the allowed-left is the carried left2 (or sot if no left2).
        sz_bool_t const carry_hy_ok = (sz_bool_t)((sz_line_break_class_is_(left_bit, sz_line_break_hy_k) ||
                                                   sz_line_break_class_is_(left_bit, sz_line_break_hh_k)) &&
                                                  (left2_is_lb20a_allowed ||
                                                   sz_line_break_class_is_(left2_bit, sz_line_break_xx_k)));
        //  When the HY/HH itself is at lane 0, its allowed-left predecessor is the CARRIED left cluster -- the in-window
        //  prev_ cannot see it, so qualify the lane-0 HY/HH via the carried left class in the allowed set (same classes
        //  as `lb20a_allowed_mask`: BK, CR, LF, NL, SP, ZW, CB, GL).
        sz_bool_t const left_is_lb20a_allowed = sz_line_break_class_in_(left_bit, lb20a_allowed_mask);
        sz_u64_t const hy_ok = (class_hyphen | class_unambiguous_hyphen) &
                               (sz_line_break_prevc_(leftset, gate, left_is_lb20a_allowed, edge) | start_of_text);
        sz_line_break_force_join_(
            (sz_line_break_prevc_(hy_ok, gate, carry_hy_ok, edge) & (class_alphabetic | class_hebrew_letter)) &
                interior_lanes,
            base, &settled);
    }
    // LB21: x BA ; x HY ; x HH ; x NS ; BB x
    sz_line_break_force_join_(
        (class_break_after | class_hyphen | class_unambiguous_hyphen | class_nonstarter |
         sz_line_break_prevc_(class_break_before, gate, sz_line_break_class_is_(left_bit, sz_line_break_bb_k), edge)) &
            interior_lanes,
        base, &settled);
    // LB21a: HL (HY|HH) x [^HL]; fires when an HL-preceded HY/HH cluster precedes a non-HL, in-window or carried.
    // The HL anchor may itself be the carried left cluster while the HY/HH sits at in-window lane 0, so the gate
    // must also fire on a carried-left HL (otherwise the block is skipped and LB31 spuriously breaks before the x).
    if ((class_hebrew_letter && (class_hyphen | class_unambiguous_hyphen)) ||
        sz_line_break_class_is_(left_bit, sz_line_break_hl_k) ||
        ((sz_line_break_class_is_(left_bit, sz_line_break_hy_k) ||
          sz_line_break_class_is_(left_bit, sz_line_break_hh_k)) &&
         sz_line_break_class_is_(left2_bit, sz_line_break_hl_k))) {
        sz_bool_t const carry_hl_hy = (sz_bool_t)((sz_line_break_class_is_(left_bit, sz_line_break_hy_k) ||
                                                   sz_line_break_class_is_(left_bit, sz_line_break_hh_k)) &&
                                                  sz_line_break_class_is_(left2_bit, sz_line_break_hl_k));
        //  A lane-0 HY/HH whose HL predecessor is the carried left cluster: prev_ misses it, so use prevc_ with Lc_HL.
        sz_u64_t const hl_hy = sz_line_break_prevc_(class_hebrew_letter, gate,
                                                    sz_line_break_class_is_(left_bit, sz_line_break_hl_k), edge) &
                               (class_hyphen | class_unambiguous_hyphen);
        sz_line_break_force_join_(
            (sz_line_break_prevc_(hl_hy, gate, carry_hl_hy, edge) & ~class_hebrew_letter) & interior_lanes, base,
            &settled);
    }
    // LB21b: SY x HL
    sz_line_break_force_join_(
        (sz_line_break_prevc_(class_symbol, gate, sz_line_break_class_is_(left_bit, sz_line_break_sy_k), edge) &
         class_hebrew_letter) &
            interior_lanes,
        base, &settled);
    // LB22: x IN
    sz_line_break_force_join_(class_inseparable & interior_lanes, base, &settled);
    // LB23 + LB24: alphabetic/numeric adjacency; fires when an AL/HL cluster is present, in-window or carried left.
    sz_bool_t const alpha_hebrew_active = (sz_bool_t)((class_alphabetic | class_hebrew_letter) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_al_k) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_hl_k));
    sz_u64_t const alpha_hebrew_prev = alpha_hebrew_active
                                           ? sz_line_break_prevc_(
                                                 class_alphabetic | class_hebrew_letter, gate,
                                                 (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_al_k) ||
                                                             sz_line_break_class_is_(left_bit, sz_line_break_hl_k)),
                                                 edge)
                                           : 0;
    if (alpha_hebrew_active) {
        sz_u64_t const prefix_postfix_prev = sz_line_break_prevc_(
            class_prefix_numeric | class_postfix_numeric, gate,
            (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_pr_k) ||
                        sz_line_break_class_is_(left_bit, sz_line_break_po_k)),
            edge);
        // LB23: (AL|HL) x NU ; NU x (AL|HL)
        sz_line_break_force_join_((alpha_hebrew_prev & class_numeric) & interior_lanes, base, &settled);
        sz_line_break_force_join_(
            (sz_line_break_prevc_(class_numeric, gate, sz_line_break_class_is_(left_bit, sz_line_break_nu_k), edge) &
             (class_alphabetic | class_hebrew_letter)) &
                interior_lanes,
            base, &settled);
        // LB24: (PR|PO) x (AL|HL) ; (AL|HL) x (PR|PO)
        sz_line_break_force_join_((prefix_postfix_prev & (class_alphabetic | class_hebrew_letter)) & interior_lanes,
                                  base, &settled);
        sz_line_break_force_join_((alpha_hebrew_prev & (class_prefix_numeric | class_postfix_numeric)) & interior_lanes,
                                  base, &settled);
    }
    // LB23a: PR x (ID|EB|EM) ; (ID|EB|EM) x PO  (fires for CJK ID, kept ungated)
    sz_u64_t const prefix_numeric_prev = sz_line_break_prevc_(
        class_prefix_numeric, gate, sz_line_break_class_is_(left_bit, sz_line_break_pr_k), edge);
    sz_line_break_force_join_(
        (prefix_numeric_prev & (class_ideographic | class_emoji_base | class_emoji_modifier)) & interior_lanes, base,
        &settled);
    sz_line_break_force_join_((sz_line_break_prevc_(class_ideographic | class_emoji_base | class_emoji_modifier, gate,
                                                    (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_id_k) ||
                                                                sz_line_break_class_is_(left_bit, sz_line_break_eb_k) ||
                                                                sz_line_break_class_is_(left_bit, sz_line_break_em_k)),
                                                    edge) &
                               class_postfix_numeric) &
                                  interior_lanes,
                              base, &settled);
    // LB25: numeric clusters
    if (class_numeric || carry.in_nu_run || carry.in_nu_close) {
        sz_u64_t const symbol_or_infix = class_symbol | class_infix_separator;
        sz_u64_t const nu_seed = class_numeric |
                                 (carry.in_nu_run && ((class_numeric | symbol_or_infix) & first_base) ? first_base : 0);
        sz_u64_t const numeric_run = sz_u64_fill_right_(nu_seed, gate | symbol_or_infix) & (nu_seed | symbol_or_infix);
        sz_u64_t const numeric_run_prev = sz_line_break_prevc_(numeric_run, gate, (sz_bool_t)carry.in_nu_run, edge);
        //  A CL/CP closing a NU run; the run may have closed at the carried left (carry.in_nu_close).
        sz_u64_t const numeric_run_close = numeric_run_prev & (class_close_punctuation | class_close_parenthesis);
        //  Hoisted lookahead predicates for the PO/PR x OP IS? NU forms (LB25): the cluster one ahead is NU, one
        //  ahead is IS, and (for the OP IS NU form) the cluster two ahead is NU.
        sz_u64_t const next_numeric = sz_line_break_next_(class_numeric, gate);
        sz_u64_t const next_infix = sz_line_break_next_(class_infix_separator, gate);
        sz_u64_t const numeric_two_ahead = sz_line_break_next_(next_numeric & base, gate);
        sz_u64_t const postfix_prev = sz_line_break_prevc_(class_postfix_numeric, gate,
                                                           sz_line_break_class_is_(left_bit, sz_line_break_po_k), edge);
        sz_u64_t const prefix_prev = prefix_numeric_prev;
        sz_line_break_force_join_((sz_line_break_prevc_(numeric_run_close, gate, (sz_bool_t)carry.in_nu_close, edge) &
                                   (class_postfix_numeric | class_prefix_numeric)) &
                                      interior_lanes,
                                  base, &settled);
        sz_line_break_force_join_((numeric_run_prev & (class_postfix_numeric | class_prefix_numeric)) & interior_lanes,
                                  base, &settled);
        sz_line_break_force_join_((postfix_prev & class_open_punctuation & next_numeric) & interior_lanes, base,
                                  &settled);
        sz_line_break_force_join_(
            (postfix_prev & class_open_punctuation & next_infix & numeric_two_ahead) & interior_lanes, base, &settled);
        sz_line_break_force_join_((postfix_prev & class_numeric) & interior_lanes, base, &settled);
        sz_line_break_force_join_((prefix_prev & class_open_punctuation & next_numeric) & interior_lanes, base,
                                  &settled);
        sz_line_break_force_join_(
            (prefix_prev & class_open_punctuation & next_infix & numeric_two_ahead) & interior_lanes, base, &settled);
        sz_line_break_force_join_((prefix_prev & class_numeric) & interior_lanes, base, &settled);
        sz_line_break_force_join_(
            (sz_line_break_prevc_(class_hyphen, gate, sz_line_break_class_is_(left_bit, sz_line_break_hy_k), edge) &
             class_numeric) &
                interior_lanes,
            base, &settled);
        sz_line_break_force_join_((sz_line_break_prevc_(class_infix_separator, gate,
                                                        sz_line_break_class_is_(left_bit, sz_line_break_is_k), edge) &
                                   class_numeric) &
                                      interior_lanes,
                                  base, &settled);
        sz_line_break_force_join_((numeric_run_prev & class_numeric) & interior_lanes, base, &settled);
    }
    // LB26 + LB27: Hangul. The window-presence half of the gate is a cheap combined range test over the Hangul class
    // bytes -- H2/H3 (33,34) and JL/JV/JT (37,38,39), with HL(35)/ID(36) excluded by the gap between the two ranges --
    // so the five individual JL/JV/JT/H2/H3 masks are extracted only when the block actually fires.
    sz_u64_t const class_hangul_present =
        (sz_line_break_effective_range_(frame, sz_line_break_h2_k, sz_line_break_h3_k) |
         sz_line_break_effective_range_(frame, sz_line_break_jl_k, sz_line_break_jt_k)) &
        base;
    if (class_hangul_present || sz_line_break_class_is_(left_bit, sz_line_break_jl_k) ||
        sz_line_break_class_is_(left_bit, sz_line_break_jv_k) ||
        sz_line_break_class_is_(left_bit, sz_line_break_jt_k) ||
        sz_line_break_class_is_(left_bit, sz_line_break_h2_k) ||
        sz_line_break_class_is_(left_bit, sz_line_break_h3_k) ||
        sz_line_break_class_is_(left_bit, sz_line_break_pr_k)) {
        sz_u64_t const class_hangul_l_jamo = frame->effective_class[sz_line_break_jl_k] & base;
        sz_u64_t const class_hangul_v_jamo = frame->effective_class[sz_line_break_jv_k] & base;
        sz_u64_t const class_hangul_t_jamo = frame->effective_class[sz_line_break_jt_k] & base;
        sz_u64_t const class_hangul_lv_syllable = frame->effective_class[sz_line_break_h2_k] & base;
        sz_u64_t const class_hangul_lvt_syllable = frame->effective_class[sz_line_break_h3_k] & base;
        sz_line_break_force_join_(
            (sz_line_break_prevc_(class_hangul_l_jamo, gate, sz_line_break_class_is_(left_bit, sz_line_break_jl_k),
                                  edge) &
             (class_hangul_l_jamo | class_hangul_v_jamo | class_hangul_lv_syllable | class_hangul_lvt_syllable)) &
                interior_lanes,
            base, &settled);
        sz_line_break_force_join_(
            (sz_line_break_prevc_(class_hangul_v_jamo | class_hangul_lv_syllable, gate,
                                  (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_jv_k) ||
                                              sz_line_break_class_is_(left_bit, sz_line_break_h2_k)),
                                  edge) &
             (class_hangul_v_jamo | class_hangul_t_jamo)) &
                interior_lanes,
            base, &settled);
        sz_line_break_force_join_(
            (sz_line_break_prevc_(class_hangul_t_jamo | class_hangul_lvt_syllable, gate,
                                  (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_jt_k) ||
                                              sz_line_break_class_is_(left_bit, sz_line_break_h3_k)),
                                  edge) &
             class_hangul_t_jamo) &
                interior_lanes,
            base, &settled);
        sz_line_break_force_join_(
            (sz_line_break_prevc_(class_hangul_l_jamo | class_hangul_v_jamo | class_hangul_t_jamo |
                                      class_hangul_lv_syllable | class_hangul_lvt_syllable,
                                  gate,
                                  (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_jl_k) ||
                                              sz_line_break_class_is_(left_bit, sz_line_break_jv_k) ||
                                              sz_line_break_class_is_(left_bit, sz_line_break_jt_k) ||
                                              sz_line_break_class_is_(left_bit, sz_line_break_h2_k) ||
                                              sz_line_break_class_is_(left_bit, sz_line_break_h3_k)),
                                  edge) &
             class_postfix_numeric) &
                interior_lanes,
            base, &settled);
        sz_line_break_force_join_(
            (prefix_numeric_prev & (class_hangul_l_jamo | class_hangul_v_jamo | class_hangul_t_jamo |
                                    class_hangul_lv_syllable | class_hangul_lvt_syllable)) &
                interior_lanes,
            base, &settled);
    }
    // LB28: (AL|HL) x (AL|HL)
    if (alpha_hebrew_active)
        sz_line_break_force_join_((alpha_hebrew_prev & (class_alphabetic | class_hebrew_letter)) & interior_lanes, base,
                                  &settled);
    // LB28a: aksara (Brahmic); DottedCircle acts as an aksara base
    {
        sz_bool_t const left_is_aksara = (sz_bool_t)(carry.have_prev && carry.left_aksara != 0);
        sz_bool_t const left2_is_aksara = (sz_bool_t)(carry.have_prev && carry.left2_aksara != 0);
        //  The window-presence half of the gate is a cheap combined range test over the contiguous Brahmic class bytes
        //  AK..VI (43..47), which covers AK/AP/AS/VF/VI exactly, ORed with the separately-held DottedCircle. The AP/VF/VI
        //  masks live only inside this block, so they are extracted here rather than at the top.
        sz_u64_t const class_brahmic_present =
            (sz_line_break_effective_range_(frame, sz_line_break_ak_k, sz_line_break_vi_k) & base) | dotted_circle;
        if (class_brahmic_present || sz_line_break_class_is_(left_bit, sz_line_break_ap_k) || left_is_aksara ||
            (sz_line_break_class_is_(left_bit, sz_line_break_vi_k) && left2_is_aksara)) {
            sz_u64_t const class_aksara_prebase = frame->effective_class[sz_line_break_ap_k] & base;
            sz_u64_t const class_virama_final = frame->effective_class[sz_line_break_vf_k] & base;
            sz_u64_t const class_virama = frame->effective_class[sz_line_break_vi_k] & base;
            sz_u64_t const aksara = class_aksara | class_aksara_start | dotted_circle;
            sz_u64_t const aksara_dc = class_aksara | dotted_circle;
            sz_u64_t const aksara_prev = sz_line_break_prevc_(aksara, gate, left_is_aksara, edge);
            sz_line_break_force_join_(
                (sz_line_break_prevc_(class_aksara_prebase, gate, sz_line_break_class_is_(left_bit, sz_line_break_ap_k),
                                      edge) &
                 aksara) &
                    interior_lanes,
                base, &settled);
            sz_line_break_force_join_((aksara_prev & (class_virama_final | class_virama)) & interior_lanes, base,
                                      &settled);
            {
                sz_u64_t base_vi = aksara_prev & class_virama;
                sz_line_break_force_join_(
                    (sz_line_break_prevc_(
                         base_vi, gate,
                         (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_vi_k) && left2_is_aksara), edge) &
                     aksara_dc) &
                        interior_lanes,
                    base, &settled);
            }
            sz_line_break_force_join_(
                (aksara_prev & aksara & sz_line_break_next_(class_virama_final, gate)) & interior_lanes, base,
                &settled);
        }
    }
    // LB29: IS x (AL|HL)
    if ((class_alphabetic | class_hebrew_letter) || sz_line_break_class_is_(left_bit, sz_line_break_is_k))
        sz_line_break_force_join_((sz_line_break_prevc_(class_infix_separator, gate,
                                                        sz_line_break_class_is_(left_bit, sz_line_break_is_k), edge) &
                                   (class_alphabetic | class_hebrew_letter)) &
                                      interior_lanes,
                                  base, &settled);
    // LB30: (AL|HL|NU) x OP[^EAW] ; CP[^EAW] x (AL|HL|NU); requires an OP or CP cluster, in-window or carried CP.
    if ((class_open_punctuation | class_close_parenthesis) || sz_line_break_class_is_(left_bit, sz_line_break_cp_k)) {
        sz_line_break_force_join_(
            (sz_line_break_prevc_(class_alphabetic | class_hebrew_letter | class_numeric, gate,
                                  (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_al_k) ||
                                              sz_line_break_class_is_(left_bit, sz_line_break_hl_k) ||
                                              sz_line_break_class_is_(left_bit, sz_line_break_nu_k)),
                                  edge) &
             class_open_punctuation & ~side_east_asian_width) &
                interior_lanes,
            base, &settled);
        //  CP[^EAW]: a carried left CP qualifies only when it is NOT East-Asian-wide.
        sz_line_break_force_join_(
            (sz_line_break_prevc_(class_close_parenthesis & ~side_east_asian_width, gate,
                                  (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_cp_k) && !carry.left_eaw),
                                  edge) &
             (class_alphabetic | class_hebrew_letter | class_numeric)) &
                interior_lanes,
            base, &settled);
    }
    // LB30a: RI RI even-parity pairing. The inclusive segmented prefix-XOR over each maximal RI run gives, at each RI,
    // the parity of indicators at-or-before it; the 2nd of each pair (even inclusive parity) is joined. A run
    // straddling in from the previous window toggles the leading run via the inbound-parity seed.
    if (class_regional_indicator) {
        sz_u64_t const ri_run_gate = sz_u64_fill_right_(class_regional_indicator, gate) | class_regional_indicator;
        sz_bool_t const inbound = (sz_bool_t)(carry.ri_open && carry.ri_parity_odd);
        sz_u64_t const inclusive = sz_line_break_segmented_parity_(class_regional_indicator, ri_run_gate, inbound);
        sz_line_break_force_join_((class_regional_indicator & ~inclusive) & interior_lanes, base, &settled);
    }
    // LB30b: EB x EM ; [ExtPict & Cn] x EM. The ExtPict/Cn side masks feed only this block, so extract them here.
    if (class_emoji_modifier) {
        sz_u64_t const side_unassigned = frame->side_cn & base;
        sz_u64_t const side_extended_pictographic = frame->side_ext & base;
        sz_line_break_force_join_(
            (sz_line_break_prevc_(class_emoji_base, gate, sz_line_break_class_is_(left_bit, sz_line_break_eb_k), edge) &
             class_emoji_modifier) &
                interior_lanes,
            base, &settled);
        sz_line_break_force_join_((sz_line_break_prevc_(side_extended_pictographic & side_unassigned, gate,
                                                        (sz_bool_t)carry.left_extpict_cn, edge) &
                                   class_emoji_modifier) &
                                      interior_lanes,
                                  base, &settled);
    }

    // LB31: break everywhere else.
    breaks |= interior_lanes & ~settled;

    //  Trust horizon: the lowest lane whose break-before verdict needs context past the window edge. Open unbounded
    //  lookahead runs (NU runs, "X SP*" opener runs, "[QU&Pi] SP*" runs) that flood to the top valid lane are
    //  undecided, and the bounded +2-codepoint lookahead (LB19 East-Asian, LB25 PO/PR OP IS? NU) leaves the top two
    //  codepoints of the complete region undecided. At the buffer tail there is no next window, so nothing clamps.
    sz_u64_t const symbol_or_infix_all = class_symbol | class_infix_separator;
    sz_u64_t const nu_seed_all =
        class_numeric | (carry.in_nu_run && ((class_numeric | symbol_or_infix_all) & first_base) ? first_base : 0);
    sz_u64_t const numeric_run_all = sz_u64_fill_right_(nu_seed_all, gate | symbol_or_infix_all) &
                                     (nu_seed_all | symbol_or_infix_all);
    sz_size_t resolved = complete_limit;
    if (more_text) {
        sz_u64_t const complete_mask = sz_u64_mask_until_serial_(complete_limit);
        sz_u64_t const complete_base = base & complete_mask;
        sz_size_t const top_base = complete_base ? (sz_size_t)(63 - sz_u64_clz(complete_base)) : 0;
        sz_u64_t const top_bit = complete_base ? (1ull << top_base) : 0ull;
        //  (a) NU run open at the top of the complete region: its closer / PO|PR suffix may lie past the edge.
        sz_u64_t undecided = (numeric_run_all & top_bit) ? top_bit : 0ull;
        //  (b) "X SP*" opener runs open at the top: the governed `x` lies past the edge.
        undecided |= ((opener_governance | close_governance | break_both_governance | zero_width_governance |
                       quote_initial_governance) &
                      top_bit);
        if (undecided) {
            //  Flood the open construct back left across its gate to the run's lowest lane: the next window re-derives
            //  the verdict there with the carried run-state, so trust ends at that base.
            sz_u64_t const run_gate = gate | symbol_or_infix_all | class_space;
            sz_u64_t const reaching = base & sz_u64_fill_left_(undecided, run_gate);
            sz_size_t const low_lane = reaching ? (sz_size_t)sz_u64_ctz(reaching) : top_base;
            if (low_lane < resolved) resolved = low_lane;
        }
        //  (d) Bounded +2-codepoint lookahead: hold back the top two complete codepoints (their right context may be
        //  out of window) unless they were already settled by an unconditional rule (which never reads past the edge).
        sz_size_t const lookahead = 2;
        sz_size_t count = 0, lane = top_base;
        for (;;) {
            sz_u64_t const lane_mask = sz_u64_mask_until_serial_(lane);
            sz_u64_t const lower = complete_base & lane_mask;
            if (!lower || count + 1 >= lookahead) {
                if (lane < resolved) resolved = lane;
                break;
            }
            lane = (sz_size_t)(63 - sz_u64_clz(lower));
            ++count;
        }
        if (resolved > complete_limit) resolved = complete_limit;
    }
    breaks &= sz_u64_mask_until_serial_(resolved);

    //  The driver advances by `advance` bytes after this call: the clamp horizon `resolved` when it bit before the
    //  complete edge, else the whole complete span. The carry anchors at THAT byte so a single decision rebuilds the
    //  next window's exact left context -- no second pass. (`resolved == 0` is an unbounded run longer than one window;
    //  the driver steps the full complete span for progress, so the carry anchors at the complete edge too.)
    sz_size_t const advance = resolved ? resolved : complete_limit;

    //  Cross-window carry: run state of the cluster ending at the highest base lane below `advance` (the next window's
    //  left context). An all-ignorable region below `advance` threads the inbound carry unchanged.
    if (carry_out) {
        sz_line_break_carry_t out = carry;
        sz_u64_t const below = base & sz_u64_mask_until_serial_(advance);
        if (below) {
            //  At least one cluster resolved: rebuild the full left context + run-state at the complete edge.
            out.open_sp_opener = 0xFF, out.in_nu_run = 0, out.in_nu_close = 0, out.ri_open = 0, out.ri_parity_odd = 0,
            out.qupi_sp_open = 0;
            sz_size_t const previous_cluster_lane = (sz_size_t)(63 - sz_u64_clz(below));
            sz_u64_t const previous_cluster_bit = 1ull << previous_cluster_lane; // prev cluster's base lane
            sz_u64_t const below2 = below & ~previous_cluster_bit;
            sz_size_t const previous2_cluster_lane = below2 ? (sz_size_t)(63 - sz_u64_clz(below2)) : 0;
            sz_u64_t const previous2_cluster_bit = below2 ? (1ull << previous2_cluster_lane) : 0ull;

            out.have_prev = 1;
            out.previous_class_bit = 1ull << effective_class_byte[previous_cluster_lane];
            sz_u8_t const side_pbit = side_byte[previous_cluster_lane];
            out.left_eaw = (sz_u8_t)((side_pbit & sz_line_break_side_eaw_k) != 0);
            out.left_pf = (sz_u8_t)((side_pbit & sz_line_break_side_pf_k) != 0);
            out.left_extpict_cn = (sz_u8_t)(((side_pbit & sz_line_break_side_ext_k) != 0) &&
                                            ((side_pbit & sz_line_break_side_cn_k) != 0));
            out.left_aksara = (sz_u8_t)(((class_aksara | class_aksara_start) & previous_cluster_bit) != 0 ||
                                        (dotted_circle & previous_cluster_bit) != 0);
            //  LB8a is codepoint-level: the previous CLUSTER joins lane 0 when its LAST codepoint is a bare ZWJ -- that
            //  ZWJ is an attached mark (LB9), so it sits ABOVE the base `pbit`, not on it. Test the highest codepoint
            //  START below the edge, not the highest base.
            sz_u64_t const starts_below = frame->starts & sz_u64_mask_until_serial_(advance);
            sz_u64_t const last_start = starts_below ? (1ull << (63 - sz_u64_clz(starts_below))) : 0ull;
            out.prev_is_zwj = (sz_u8_t)((zwj_starts & last_start) != 0);
            //  The two-left context is the prior resolved base, or -- when only one cluster resolved here -- the inbound
            //  carry's left cluster (which lay just before that single resolved base).
            if (below2) {
                out.previous2_class_bit = 1ull << effective_class_byte[previous2_cluster_lane];
                sz_u8_t const side_p2bit = side_byte[previous2_cluster_lane];
                out.left2_eaw = (sz_u8_t)((side_p2bit & sz_line_break_side_eaw_k) != 0);
                out.left2_aksara = (sz_u8_t)(((class_aksara | class_aksara_start) & previous2_cluster_bit) != 0 ||
                                             (dotted_circle & previous2_cluster_bit) != 0);
            }
            else {
                out.previous2_class_bit = carry.have_prev ? carry.previous_class_bit : (1ull << sz_line_break_xx_k);
                out.left2_eaw = carry.have_prev ? carry.left_eaw : (sz_u8_t)0;
                out.left2_aksara = carry.have_prev ? carry.left_aksara : (sz_u8_t)0;
            }

            out.qupi_sp_open = (sz_u8_t)((quote_initial_governance & previous_cluster_bit) &&
                                         ((class_space | quote_initial) & previous_cluster_bit));
            if ((opener_governance & previous_cluster_bit) &&
                ((class_space | class_open_punctuation) & previous_cluster_bit))
                out.open_sp_opener = sz_line_break_op_k;
            else if ((close_governance & previous_cluster_bit) &&
                     ((class_space | class_close_punctuation | class_close_parenthesis) & previous_cluster_bit))
                out.open_sp_opener = (class_close_parenthesis & previous_cluster_bit) ? (sz_u8_t)sz_line_break_cp_k
                                                                                      : (sz_u8_t)sz_line_break_cl_k;
            else if ((break_both_governance & previous_cluster_bit) &&
                     ((class_space | class_break_both) & previous_cluster_bit))
                out.open_sp_opener = sz_line_break_b2_k;
            else if ((zero_width_governance & previous_cluster_bit) &&
                     ((class_space | class_zero_width_space) & previous_cluster_bit))
                out.open_sp_opener = sz_line_break_zw_k;
            out.in_nu_run = (sz_u8_t)((numeric_run_all & previous_cluster_bit) != 0);
            //  A CL/CP closing a NU run (its predecessor cluster is in the run): seeds LB25 nu_run_close next window.
            sz_u64_t const nu_run_close_all = sz_line_break_prevc_(numeric_run_all, gate, (sz_bool_t)carry.in_nu_run,
                                                                   edge) &
                                              (class_close_punctuation | class_close_parenthesis);
            out.in_nu_close = (sz_u8_t)((nu_run_close_all & previous_cluster_bit) != 0);
            sz_u64_t const ri_run_gate = sz_u64_fill_right_(class_regional_indicator, gate) | class_regional_indicator;
            sz_bool_t const inbound = (sz_bool_t)(carry.ri_open && carry.ri_parity_odd);
            sz_u64_t const inclusive = sz_line_break_segmented_parity_(class_regional_indicator, ri_run_gate, inbound);
            out.ri_open = (sz_u8_t)((class_regional_indicator & previous_cluster_bit) != 0);
            out.ri_parity_odd = (sz_u8_t)((inclusive & previous_cluster_bit) != 0);
        }
        *carry_out = out;
    }
    sz_line_break_window_t result;
    result.breaks = breaks;
    result.resolved = resolved;
    return result;
}

#pragma endregion // UAX 14 Line Boundaries window engine

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_LINEBREAKS_SERIAL_H_
