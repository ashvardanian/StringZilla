/**
 *  @brief Serial backend for UAX-14 line break boundaries.
 *  @file include/stringzilla/utf8_linewraps/serial.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_LINEWRAPS_SERIAL_H_
#define STRINGZILLA_UTF8_LINEWRAPS_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_linewraps/tables.h"
#include "stringzilla/utf8_codepoints/serial.h" // shared decode helpers
#include "stringzilla/utf8_runes.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region UAX 14 Line Boundaries

/** @brief Returns the UAX-14 palette descriptor (class in bits 0-5, side flags in bits 6-13) for a codepoint. */
SZ_PUBLIC sz_u16_t sz_rune_line_break_property(sz_rune_t rune) {
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
SZ_INTERNAL sz_u8_t sz_line_break_descriptor_class_(sz_u16_t descriptor) { return (sz_u8_t)(descriptor & 0x3Fu); }
/** @brief True if a descriptor carries the general-category Pi (initial quote) flag. */
SZ_INTERNAL sz_bool_t sz_line_break_descriptor_is_pi_(sz_u16_t descriptor) {
    return (sz_bool_t)((descriptor >> 6) & 1u);
}
/** @brief True if a descriptor carries the general-category Pf (final quote) flag. */
SZ_INTERNAL sz_bool_t sz_line_break_descriptor_is_pf_(sz_u16_t descriptor) {
    return (sz_bool_t)((descriptor >> 7) & 1u);
}
/** @brief True if a descriptor carries the East-Asian Width F/W/H flag. */
SZ_INTERNAL sz_bool_t sz_line_break_descriptor_is_eaw_(sz_u16_t descriptor) {
    return (sz_bool_t)((descriptor >> 8) & 1u);
}
/** @brief True if a descriptor is an unassigned Extended_Pictographic (LB30b second clause). */
SZ_INTERNAL sz_bool_t sz_line_break_descriptor_is_extpict_cn_(sz_u16_t descriptor) {
    return (sz_bool_t)((descriptor >> 9) & 1u);
}
/** @brief True if a descriptor is Dotted_Circle U+25CC (an aksara base in LB28a). */
SZ_INTERNAL sz_bool_t sz_line_break_descriptor_is_dotted_circle_(sz_u16_t descriptor) {
    return (sz_bool_t)((descriptor >> 13) & 1u);
}
/** @brief True for a Line_Break CM or ZWJ class (LB9 attachment candidates). */
SZ_INTERNAL sz_bool_t sz_line_break_is_cm_or_zwj_(sz_u8_t line_break_class) {
    return (sz_bool_t)(line_break_class == sz_line_break_cm_k || line_break_class == sz_line_break_zwj_k);
}

/**
 *  @brief Cluster base index of codepoint @p index: walks left over attached CM/ZWJ marks (LB9) to the base.
 */
SZ_INTERNAL sz_size_t sz_line_break_cluster_base_(sz_u8_t const *raw_classes, sz_u8_t const *effective_classes,
                                                  sz_size_t index) {
    while (index > 0 && sz_line_break_is_cm_or_zwj_(raw_classes[index]) &&
           !(effective_classes[index - 1] == sz_line_break_bk_k || effective_classes[index - 1] == sz_line_break_cr_k ||
             effective_classes[index - 1] == sz_line_break_lf_k || effective_classes[index - 1] == sz_line_break_nl_k ||
             effective_classes[index - 1] == sz_line_break_sp_k || effective_classes[index - 1] == sz_line_break_zw_k))
        --index;
    return index;
}

/** @brief Per-call internal window: codepoints buffered on the stack before the LB1-LB31 sweep runs. */
enum { sz_utf8_line_window_k = 1024 };

/**
 *  @brief Plural UAX-14 line-break segmentation: one forward sweep emits every UAX-14 break opportunity
 *         (LB1-LB31) into parallel `line_starts` / `line_lengths` arrays. Forward-only (no reverse counterpart).
 *
 *  Implements LB1-LB31 with no caller scratch. The text is processed in fixed internal windows of at most
 *  `sz_utf8_line_window_k` codepoints decoded into small stack arrays (byte start, descriptor, LB1-resolved
 *  raw class, post-LB9/LB10 effective class). Each window is re-anchored at the current open line's start: an
 *  emitted opportunity is a hard reset point past which no LB rule reads, so re-decoding the open line from
 *  `line_start` is bit-identical to the whole-text sweep. When the window fills before the input ends,
 *  `bytes_consumed` carries the resume offset (the open line's start) for the next call.
 */
SZ_PUBLIC sz_size_t sz_utf8_linewraps_serial(        //
    sz_cptr_t text, sz_size_t length,                //
    sz_size_t *line_starts, sz_size_t *line_lengths, //
    sz_size_t lines_capacity, sz_size_t *bytes_consumed) {

    sz_size_t lines = 0;
    if (length == 0 || lines_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_size_t codepoint_byte_starts_buffer[sz_utf8_line_window_k];
    sz_u16_t codepoint_descriptors_buffer[sz_utf8_line_window_k];
    sz_u8_t raw_classes_buffer[sz_utf8_line_window_k];
    sz_u8_t effective_classes_buffer[sz_utf8_line_window_k];
    sz_u8_t prev_zwj_buffer[sz_utf8_line_window_k];
    sz_size_t *codepoint_byte_starts = codepoint_byte_starts_buffer;
    sz_u16_t *codepoint_descriptors = codepoint_descriptors_buffer;
    sz_u8_t *raw_classes = raw_classes_buffer;
    sz_u8_t *effective_classes = effective_classes_buffer;
    sz_u8_t *prev_zwj = prev_zwj_buffer;

    // `line_start` (absolute byte offset) anchors the current open line; each window decodes from here.
    sz_size_t line_start = 0;
    while (line_start < length && lines < lines_capacity) {

        // Decode this window; resolve LB1: SA -> CM if combining (descriptor mark bit) else AL; AI/SG/XX -> AL;
        // CJ -> NS. `window_end` is the absolute byte offset one past the last decoded codepoint.
        sz_size_t count = 0;
        sz_size_t position = line_start;
        sz_bool_t window_full = sz_false_k;
        while (position < length) {
            if (count == sz_utf8_line_window_k) {
                window_full = sz_true_k;
                break;
            }
            sz_size_t decode = position;
            sz_rune_t rune = sz_utf8_decode_(text, length, &decode);
            sz_u16_t cp_descriptor = sz_rune_line_break_property(rune);
            sz_u8_t line_break_class = sz_line_break_descriptor_class_(cp_descriptor);
            if (line_break_class == sz_line_break_sa_k)
                line_break_class = (cp_descriptor & (1u << 12)) ? (sz_u8_t)sz_line_break_cm_k
                                                                : (sz_u8_t)sz_line_break_al_k;
            else if (line_break_class == sz_line_break_ai_k || line_break_class == sz_line_break_sg_k ||
                     line_break_class == sz_line_break_xx_k)
                line_break_class = sz_line_break_al_k;
            else if (line_break_class == sz_line_break_cj_k) line_break_class = sz_line_break_ns_k;
            codepoint_byte_starts[count] = position; // absolute byte offset
            codepoint_descriptors[count] = cp_descriptor;
            raw_classes[count] = line_break_class;
            effective_classes[count] = line_break_class;
            ++count;
            position = decode > position ? decode : position + 1;
        }
        sz_size_t const window_end = position;
        // The window is a true text tail (rather than a forced cut) when it reached `length`. Only a true tail
        // may emit the trailing span and stop; a forced cut resumes from the open line in the next call.
        sz_bool_t const window_is_tail = (sz_bool_t)(!window_full && window_end == length);

        // LB9/LB10 cluster collapse (matches the SIMD path): drop a combining mark that attaches to a preceding
        // base so the rule sweep below sees exactly one entry per cluster and no rule's fixed-offset neighbour
        // lookup (left or right) can land on a mark. A lone mark (LB10) is kept but reclassified to AL. `prev_zwj`
        // records whether the codepoint immediately preceding each survivor was a ZWJ, preserving LB8a ("no break
        // after ZWJ") which the collapse would otherwise lose. `codepoint_byte_starts` is compacted to the
        // survivors' byte offsets, so the emit naturally folds each cluster's marks into the segment of its base.
        sz_size_t kept_count = 0;
        sz_bool_t attachable = sz_false_k, pending_zwj = sz_false_k;
        for (sz_size_t codepoint_index = 0; codepoint_index < count; ++codepoint_index) {
            sz_u8_t line_break_class = raw_classes[codepoint_index];
            sz_bool_t const is_mark = (sz_bool_t)(line_break_class == sz_line_break_cm_k ||
                                                  line_break_class == sz_line_break_zwj_k);
            if (is_mark && attachable) {
                pending_zwj = (sz_bool_t)(line_break_class == sz_line_break_zwj_k);
                continue;
            }
            prev_zwj[kept_count] = (sz_u8_t)pending_zwj;
            if (is_mark) {
                raw_classes[kept_count] = sz_line_break_al_k, effective_classes[kept_count] = sz_line_break_al_k,
                codepoint_descriptors[kept_count] = 0;
                attachable = sz_true_k;
            }
            else {
                raw_classes[kept_count] = line_break_class, effective_classes[kept_count] = line_break_class,
                codepoint_descriptors[kept_count] = codepoint_descriptors[codepoint_index];
                attachable =
                    (sz_bool_t)(line_break_class != sz_line_break_bk_k && line_break_class != sz_line_break_cr_k &&
                                line_break_class != sz_line_break_lf_k && line_break_class != sz_line_break_nl_k &&
                                line_break_class != sz_line_break_sp_k && line_break_class != sz_line_break_zw_k);
            }
            codepoint_byte_starts[kept_count] = codepoint_byte_starts[codepoint_index];
            pending_zwj = (sz_bool_t)(line_break_class == sz_line_break_zwj_k);
            ++kept_count;
        }
        count = kept_count;

        // LB9/LB10 effective class: CM/ZWJ attaches to the prior cluster base, else (LB10) acts as AL.
        for (sz_size_t codepoint_index = 0; codepoint_index < count; ++codepoint_index) {
            if (!sz_line_break_is_cm_or_zwj_(raw_classes[codepoint_index])) continue;
            // LB10 (start of text / open line)
            if (codepoint_index == 0) {
                effective_classes[codepoint_index] = sz_line_break_al_k;
                continue;
            }
            sz_u8_t base = effective_classes[codepoint_index - 1];
            if (base != sz_line_break_bk_k && base != sz_line_break_cr_k && base != sz_line_break_lf_k &&
                base != sz_line_break_nl_k && base != sz_line_break_sp_k && base != sz_line_break_zw_k)
                effective_classes[codepoint_index] = base;                // LB9 attach
            else effective_classes[codepoint_index] = sz_line_break_al_k; // LB10
        }

        // `local_line_start` is the open line's start expressed as a byte offset; it begins at the window origin
        // and advances with each emitted opportunity. Spans are emitted with absolute byte offsets from
        // `codepoint_byte_starts`.
        sz_size_t local_line_start = line_start;
        sz_bool_t hit_capacity = sz_false_k;

        // Forward-carried run state replacing the per-position backward scans (LB8 nearest-non-space, LB25 numeric
        // run, LB30a Regional_Indicator parity) so each window is O(count), not O(count^2). All three are window-local:
        // the window re-anchors at an emitted opportunity, a hard reset past which no rule reads, so seeding from the
        // window's first cluster reproduces the within-window backward scans exactly. Each is advanced at the loop
        // tail by the right cluster, so at iteration `codepoint_index` it reflects clusters [0, codepoint_index-1].
        sz_size_t last_non_space_index = 0; // nearest non-SP cluster at or left of the previous cluster, else 0
        sz_bool_t numeric_run_open = (sz_bool_t)(effective_classes[0] == sz_line_break_nu_k);
        sz_bool_t numeric_run_open_before = sz_false_k; // `numeric_run_open` as of two clusters back (LB25 close case)
        sz_bool_t regional_indicator_parity_odd = (sz_bool_t)(effective_classes[0] == sz_line_break_ri_k &&
                                                              raw_classes[0] == sz_line_break_ri_k);

        for (sz_size_t codepoint_index = 1; codepoint_index < count; ++codepoint_index) {
            sz_u8_t left_class = effective_classes[codepoint_index - 1];
            sz_u8_t right_class = effective_classes[codepoint_index];
            sz_bool_t is_break = sz_false_k;
            sz_bool_t resolved = sz_false_k;

            // last non-space context (LB8/14/15a/16/17): the nearest non-SP effective class at or left of
            // codepoint_index-1, and whether an SP run intervenes. `had_space` matches the old scan: an SP at
            // codepoint_index-1 with a further-left cluster (the lone leading SP at the window origin skipped none).
            sz_u8_t leftmost_non_space_class = effective_classes[last_non_space_index];
            sz_bool_t had_space = (sz_bool_t)(left_class == sz_line_break_sp_k && codepoint_index >= 2);

            // cluster base index of the codepoint at codepoint_index-1 (skip attached marks).
            sz_size_t left_base_index = sz_line_break_cluster_base_(raw_classes, effective_classes,
                                                                    codepoint_index - 1);

            // The dispatch below is a first-match-wins precedence chain over LB4-LB31: the first rule whose
            // condition holds sets the join/break verdict (`is_break`) and latches `resolved` so no later rule
            // fires. Rule order encodes UAX-14 precedence verbatim, so the latch preserves the exact rule that
            // fires for each position.
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
            if (!resolved && prev_zwj[codepoint_index])
                resolved = sz_true_k; // LB8a: no break after ZWJ (preserved across the collapse)
            if (!resolved && sz_line_break_is_cm_or_zwj_(raw_classes[codepoint_index]) &&
                left_class != sz_line_break_bk_k && left_class != sz_line_break_cr_k &&
                left_class != sz_line_break_lf_k && left_class != sz_line_break_nl_k &&
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
                sz_line_break_descriptor_is_pi_(codepoint_descriptors[sz_line_break_cluster_base_(
                    raw_classes, effective_classes, last_non_space_index)])) {
                sz_size_t quote_base_index = sz_line_break_cluster_base_(raw_classes, effective_classes,
                                                                         last_non_space_index);
                sz_bool_t left_ok;
                if (quote_base_index == 0) left_ok = sz_true_k;
                else {
                    sz_u8_t left_context_class = effective_classes[quote_base_index - 1];
                    left_ok = (sz_bool_t)(left_context_class == sz_line_break_bk_k ||
                                          left_context_class == sz_line_break_cr_k ||
                                          left_context_class == sz_line_break_lf_k ||
                                          left_context_class == sz_line_break_nl_k ||
                                          left_context_class == sz_line_break_op_k ||
                                          left_context_class == sz_line_break_qu_k ||
                                          left_context_class == sz_line_break_gl_k ||
                                          left_context_class == sz_line_break_sp_k ||
                                          left_context_class == sz_line_break_zw_k);
                }
                if (left_ok) resolved = sz_true_k;
            }
            // LB15b: closing Pf quote followed by an allowed class or end-of-text.
            if (!resolved && right_class == sz_line_break_qu_k &&
                sz_line_break_descriptor_is_pf_(codepoint_descriptors[codepoint_index])) {
                sz_bool_t right_ok;
                if (codepoint_index + 1 >= count) right_ok = sz_true_k;
                else {
                    sz_u8_t right_context_class = effective_classes[codepoint_index + 1];
                    right_ok = (sz_bool_t)(right_context_class == sz_line_break_sp_k ||
                                           right_context_class == sz_line_break_gl_k ||
                                           right_context_class == sz_line_break_wj_k ||
                                           right_context_class == sz_line_break_cl_k ||
                                           right_context_class == sz_line_break_qu_k ||
                                           right_context_class == sz_line_break_cp_k ||
                                           right_context_class == sz_line_break_ex_k ||
                                           right_context_class == sz_line_break_is_k ||
                                           right_context_class == sz_line_break_sy_k ||
                                           right_context_class == sz_line_break_bk_k ||
                                           right_context_class == sz_line_break_cr_k ||
                                           right_context_class == sz_line_break_lf_k ||
                                           right_context_class == sz_line_break_nl_k ||
                                           right_context_class == sz_line_break_zw_k);
                }
                if (right_ok) resolved = sz_true_k;
            }
            if (!resolved && left_class == sz_line_break_sp_k && right_class == sz_line_break_is_k &&
                codepoint_index + 1 < count && effective_classes[codepoint_index + 1] == sz_line_break_nu_k) {
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
            if (!resolved && right_class == sz_line_break_qu_k &&
                !sz_line_break_descriptor_is_pi_(codepoint_descriptors[codepoint_index]))
                resolved = sz_true_k;
            if (!resolved && left_class == sz_line_break_qu_k &&
                !sz_line_break_descriptor_is_pf_(codepoint_descriptors[left_base_index]))
                resolved = sz_true_k;
            if (!resolved && right_class == sz_line_break_qu_k &&
                !sz_line_break_descriptor_is_eaw_(codepoint_descriptors[left_base_index]))
                resolved = sz_true_k;
            if (!resolved && right_class == sz_line_break_qu_k &&
                (codepoint_index + 1 >= count ||
                 !sz_line_break_descriptor_is_eaw_(codepoint_descriptors[codepoint_index + 1])))
                resolved = sz_true_k;
            if (!resolved && left_class == sz_line_break_qu_k &&
                !sz_line_break_descriptor_is_eaw_(codepoint_descriptors[codepoint_index]))
                resolved = sz_true_k;
            if (!resolved && left_class == sz_line_break_qu_k &&
                (left_base_index == 0 ||
                 !sz_line_break_descriptor_is_eaw_(codepoint_descriptors[sz_line_break_cluster_base_(
                     raw_classes, effective_classes, left_base_index - 1)])))
                resolved = sz_true_k;
            if (!resolved && (right_class == sz_line_break_cb_k || left_class == sz_line_break_cb_k)) {
                is_break = sz_true_k;
                resolved = sz_true_k;
            } // LB20
            // LB20a: (sot|allowed) (HY|HH) x (AL|HL).
            if (!resolved && (left_class == sz_line_break_hy_k || left_class == sz_line_break_hh_k) &&
                (right_class == sz_line_break_al_k || right_class == sz_line_break_hl_k)) {
                sz_bool_t left_context_ok;
                if (left_base_index == 0) left_context_ok = sz_true_k;
                else {
                    sz_u8_t left_context_class = effective_classes[left_base_index - 1];
                    left_context_ok = (sz_bool_t)(left_context_class == sz_line_break_bk_k ||
                                                  left_context_class == sz_line_break_cr_k ||
                                                  left_context_class == sz_line_break_lf_k ||
                                                  left_context_class == sz_line_break_nl_k ||
                                                  left_context_class == sz_line_break_sp_k ||
                                                  left_context_class == sz_line_break_zw_k ||
                                                  left_context_class == sz_line_break_cb_k ||
                                                  left_context_class == sz_line_break_gl_k);
                }
                if (left_context_ok) resolved = sz_true_k;
            }
            if (!resolved && (right_class == sz_line_break_ba_k || right_class == sz_line_break_hy_k ||
                              right_class == sz_line_break_hh_k || right_class == sz_line_break_ns_k ||
                              left_class == sz_line_break_bb_k))
                resolved = sz_true_k; // LB21
            // LB21a: HL (HY|HH) x [^HL] -- the HL precedes the (HY|HH) cluster (skip its marks)
            if (!resolved && left_base_index > 0 && effective_classes[left_base_index - 1] == sz_line_break_hl_k &&
                (left_class == sz_line_break_hy_k || left_class == sz_line_break_hh_k) &&
                right_class != sz_line_break_hl_k)
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
            // LB25: numeric clusters. The preceding "NU (SY|IS)*" run (optionally followed by a CL/CP close) is
            // carried forward in `numeric_run_open` (run open at codepoint_index-1) and `numeric_run_open_before`
            // (run open at codepoint_index-2, the state just before a CL/CP left context).
            if (!resolved) {
                sz_bool_t has_preceding_numeric_run = numeric_run_open;
                sz_bool_t has_preceding_numeric_run_then_close = sz_false_k;
                if (left_class == sz_line_break_cl_k || left_class == sz_line_break_cp_k)
                    has_preceding_numeric_run_then_close = numeric_run_open_before;
                if (!resolved && has_preceding_numeric_run_then_close &&
                    (right_class == sz_line_break_po_k || right_class == sz_line_break_pr_k))
                    resolved = sz_true_k;
                if (!resolved && has_preceding_numeric_run &&
                    (right_class == sz_line_break_po_k || right_class == sz_line_break_pr_k))
                    resolved = sz_true_k;
                if (!resolved && left_class == sz_line_break_po_k && right_class == sz_line_break_op_k &&
                    codepoint_index + 1 < count && effective_classes[codepoint_index + 1] == sz_line_break_nu_k)
                    resolved = sz_true_k;
                if (!resolved && left_class == sz_line_break_po_k && right_class == sz_line_break_op_k &&
                    codepoint_index + 2 < count && effective_classes[codepoint_index + 1] == sz_line_break_is_k &&
                    effective_classes[codepoint_index + 2] == sz_line_break_nu_k)
                    resolved = sz_true_k;
                if (!resolved && left_class == sz_line_break_po_k && right_class == sz_line_break_nu_k)
                    resolved = sz_true_k;
                if (!resolved && left_class == sz_line_break_pr_k && right_class == sz_line_break_op_k &&
                    codepoint_index + 1 < count && effective_classes[codepoint_index + 1] == sz_line_break_nu_k)
                    resolved = sz_true_k;
                if (!resolved && left_class == sz_line_break_pr_k && right_class == sz_line_break_op_k &&
                    codepoint_index + 2 < count && effective_classes[codepoint_index + 1] == sz_line_break_is_k &&
                    effective_classes[codepoint_index + 2] == sz_line_break_nu_k)
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
                (left_class == sz_line_break_jl_k || left_class == sz_line_break_jv_k ||
                 left_class == sz_line_break_jt_k || left_class == sz_line_break_h2_k ||
                 left_class == sz_line_break_h3_k) &&
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
                sz_bool_t left_is_dotted_circle = sz_line_break_descriptor_is_dotted_circle_(
                    codepoint_descriptors[left_base_index]);
                sz_bool_t right_is_dotted_circle = sz_line_break_descriptor_is_dotted_circle_(
                    codepoint_descriptors[codepoint_index]);
                sz_bool_t left_is_base = (sz_bool_t)(left_class == sz_line_break_ak_k ||
                                                     left_class == sz_line_break_as_k || left_is_dotted_circle);
                sz_bool_t right_is_base = (sz_bool_t)(right_class == sz_line_break_ak_k ||
                                                      right_class == sz_line_break_as_k || right_is_dotted_circle);
                sz_bool_t right_base_or_dotted_circle = (sz_bool_t)(right_class == sz_line_break_ak_k ||
                                                                    right_is_dotted_circle);
                if (!resolved && left_class == sz_line_break_ap_k && right_is_base) resolved = sz_true_k; // 28.11
                if (!resolved && left_is_base &&
                    (right_class == sz_line_break_vf_k || right_class == sz_line_break_vi_k))
                    resolved = sz_true_k; // 28.12
                if (!resolved && left_class == sz_line_break_vi_k && right_base_or_dotted_circle &&
                    left_base_index > 0) { // 28.13
                    sz_size_t predecessor_index = sz_line_break_cluster_base_(raw_classes, effective_classes,
                                                                              left_base_index - 1);
                    sz_u8_t predecessor_class = effective_classes[predecessor_index];
                    if (predecessor_class == sz_line_break_ak_k || predecessor_class == sz_line_break_as_k ||
                        sz_line_break_descriptor_is_dotted_circle_(codepoint_descriptors[predecessor_index]))
                        resolved = sz_true_k;
                }
                if (!resolved && left_is_base && right_is_base && codepoint_index + 1 < count &&
                    effective_classes[codepoint_index + 1] == sz_line_break_vf_k)
                    resolved = sz_true_k; // 28.14
            }
            if (!resolved && left_class == sz_line_break_is_k &&
                (right_class == sz_line_break_al_k || right_class == sz_line_break_hl_k))
                resolved = sz_true_k; // LB29
            // LB30.
            if (!resolved &&
                (left_class == sz_line_break_al_k || left_class == sz_line_break_hl_k ||
                 left_class == sz_line_break_nu_k) &&
                right_class == sz_line_break_op_k &&
                !sz_line_break_descriptor_is_eaw_(codepoint_descriptors[codepoint_index]))
                resolved = sz_true_k;
            if (!resolved && left_class == sz_line_break_cp_k &&
                (right_class == sz_line_break_al_k || right_class == sz_line_break_hl_k ||
                 right_class == sz_line_break_nu_k) &&
                !sz_line_break_descriptor_is_eaw_(codepoint_descriptors[left_base_index]))
                resolved = sz_true_k;
            // LB30a: RI RI keep pairs. The parity of the Regional_Indicator run ending at codepoint_index-1 is
            // carried in `regional_indicator_parity_odd`; an odd count keeps the pair (LB30a applies).
            if (!resolved && left_class == sz_line_break_ri_k && right_class == sz_line_break_ri_k &&
                regional_indicator_parity_odd)
                resolved = sz_true_k;
            // LB30b.
            if (!resolved && left_class == sz_line_break_eb_k && right_class == sz_line_break_em_k)
                resolved = sz_true_k;
            if (!resolved && right_class == sz_line_break_em_k &&
                sz_line_break_descriptor_is_extpict_cn_(codepoint_descriptors[left_base_index]))
                resolved = sz_true_k;

            if (!resolved) is_break = sz_true_k; // LB31

            // A forced cut leaves the last two codepoints without real right context (the LB lookahead reaches at
            // most codepoint_index+2), so on such a window only opportunities with two further in-window codepoints
            // are committed; the open line resumes in the next call where full context is available. A true tail
            // commits every break.
            if (is_break && (window_is_tail || codepoint_index + 2 < count)) {
                if (lines == lines_capacity) {
                    hit_capacity = sz_true_k;
                    break;
                }
                line_starts[lines] = local_line_start;
                line_lengths[lines] = codepoint_byte_starts[codepoint_index] - local_line_start;
                ++lines;
                local_line_start = codepoint_byte_starts[codepoint_index];
            }

            // Advance the forward-carried run state by the right cluster (codepoint_index) for the next iteration,
            // reproducing the window-local backward scans these fields replace.
            if (right_class != sz_line_break_sp_k) last_non_space_index = codepoint_index;
            numeric_run_open_before = numeric_run_open;
            if (right_class == sz_line_break_nu_k) numeric_run_open = sz_true_k;
            else if (!(numeric_run_open && (right_class == sz_line_break_sy_k || right_class == sz_line_break_is_k)))
                numeric_run_open = sz_false_k;
            if (right_class == sz_line_break_ri_k && raw_classes[codepoint_index] == sz_line_break_ri_k)
                regional_indicator_parity_odd = (sz_bool_t)!regional_indicator_parity_odd;
            else regional_indicator_parity_odd = sz_false_k;
        }

        if (hit_capacity) {
            if (bytes_consumed) *bytes_consumed = local_line_start;
            return lines;
        }
        if (window_is_tail) {
            // The trailing span [local_line_start, length) is the final line; end of text is a boundary (LB3).
            if (lines == lines_capacity) {
                if (bytes_consumed) *bytes_consumed = local_line_start;
                return lines;
            }
            line_starts[lines] = local_line_start;
            line_lengths[lines] = length - local_line_start;
            ++lines;
            if (bytes_consumed) *bytes_consumed = length;
            return lines;
        }
        // Forced cut: the open line is unresolved. Re-anchor the next window at it (guaranteed progress as long as a
        // single line fits the window). If no break committed in this window, advance past the window to avoid a
        // stall on a pathologically long line, mirroring the resume-offset contract.
        line_start = (local_line_start > line_start) ? local_line_start : window_end;
    }

    if (bytes_consumed) *bytes_consumed = line_start;
    return lines;
}

#pragma endregion // UAX 14 Line Boundaries

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_LINEWRAPS_SERIAL_H_
