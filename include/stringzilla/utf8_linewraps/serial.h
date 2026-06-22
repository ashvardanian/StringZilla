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

#pragma region UAX-14 Line Boundaries

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
SZ_INTERNAL sz_bool_t sz_line_break_is_cm_or_zwj_(sz_u8_t cls) {
    return (sz_bool_t)(cls == sz_line_break_cm_k || cls == sz_line_break_zwj_k);
}

/**
 *  @brief Cluster base index of codepoint @p index: walks left over attached CM/ZWJ marks (LB9) to the base.
 */
SZ_INTERNAL sz_size_t sz_line_break_cluster_base_(sz_u8_t const *raw, sz_u8_t const *eff, sz_size_t index) {
    while (index > 0 && sz_line_break_is_cm_or_zwj_(raw[index]) &&
           !(eff[index - 1] == sz_line_break_bk_k || eff[index - 1] == sz_line_break_cr_k ||
             eff[index - 1] == sz_line_break_lf_k || eff[index - 1] == sz_line_break_nl_k ||
             eff[index - 1] == sz_line_break_sp_k || eff[index - 1] == sz_line_break_zw_k))
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

    sz_size_t start_buffer[sz_utf8_line_window_k];
    sz_u16_t descriptor_buffer[sz_utf8_line_window_k];
    sz_u8_t raw_buffer[sz_utf8_line_window_k];
    sz_u8_t eff_buffer[sz_utf8_line_window_k];
    sz_u8_t prev_zwj_buffer[sz_utf8_line_window_k];
    sz_size_t *start = start_buffer;
    sz_u16_t *descriptor = descriptor_buffer;
    sz_u8_t *raw = raw_buffer;
    sz_u8_t *eff = eff_buffer;
    sz_u8_t *prev_zwj = prev_zwj_buffer;

    /* `line_start` (absolute byte offset) anchors the current open line; each window decodes from here. */
    sz_size_t line_start = 0;
    while (line_start < length && lines < lines_capacity) {

        /* Decode this window; resolve LB1: SA -> CM if combining (descriptor mark bit) else AL; AI/SG/XX -> AL;
         *  CJ -> NS. `window_end` is the absolute byte offset one past the last decoded codepoint. */
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
            sz_u8_t cls = sz_line_break_descriptor_class_(cp_descriptor);
            if (cls == sz_line_break_sa_k)
                cls = (cp_descriptor & (1u << 12)) ? (sz_u8_t)sz_line_break_cm_k : (sz_u8_t)sz_line_break_al_k;
            else if (cls == sz_line_break_ai_k || cls == sz_line_break_sg_k || cls == sz_line_break_xx_k)
                cls = sz_line_break_al_k;
            else if (cls == sz_line_break_cj_k) cls = sz_line_break_ns_k;
            start[count] = position; // absolute byte offset
            descriptor[count] = cp_descriptor;
            raw[count] = cls;
            eff[count] = cls;
            ++count;
            position = decode > position ? decode : position + 1;
        }
        sz_size_t const window_end = position;
        /* The window is a true text tail (rather than a forced cut) when it reached `length`. Only a true tail
         *  may emit the trailing span and stop; a forced cut resumes from the open line in the next call. */
        sz_bool_t const window_is_tail = (sz_bool_t)(!window_full && window_end == length);

        /* LB9/LB10 cluster collapse (matches the SIMD path): drop a combining mark that attaches to a preceding
         *  base so the rule sweep below sees exactly one entry per cluster and no rule's fixed-offset neighbour
         *  lookup (left or right) can land on a mark. A lone mark (LB10) is kept but reclassified to AL. `prev_zwj`
         *  records whether the codepoint immediately preceding each survivor was a ZWJ, preserving LB8a ("no break
         *  after ZWJ") which the collapse would otherwise lose. `start` is compacted to the survivors' byte offsets,
         *  so the emit naturally folds each cluster's marks into the segment of its base. */
        {
            sz_size_t kept = 0;
            sz_bool_t attachable = sz_false_k, pending_zwj = sz_false_k;
            for (sz_size_t i = 0; i < count; ++i) {
                sz_u8_t cls = raw[i];
                sz_bool_t const is_mark = (sz_bool_t)(cls == sz_line_break_cm_k || cls == sz_line_break_zwj_k);
                if (is_mark && attachable) {
                    pending_zwj = (sz_bool_t)(cls == sz_line_break_zwj_k);
                    continue;
                }
                prev_zwj[kept] = (sz_u8_t)pending_zwj;
                if (is_mark) {
                    raw[kept] = sz_line_break_al_k, eff[kept] = sz_line_break_al_k, descriptor[kept] = 0;
                    attachable = sz_true_k;
                }
                else {
                    raw[kept] = cls, eff[kept] = cls, descriptor[kept] = descriptor[i];
                    attachable = (sz_bool_t)(cls != sz_line_break_bk_k && cls != sz_line_break_cr_k &&
                                             cls != sz_line_break_lf_k && cls != sz_line_break_nl_k &&
                                             cls != sz_line_break_sp_k && cls != sz_line_break_zw_k);
                }
                start[kept] = start[i];
                pending_zwj = (sz_bool_t)(cls == sz_line_break_zwj_k);
                ++kept;
            }
            count = kept;
        }

        /* LB9/LB10 effective class: CM/ZWJ attaches to the prior cluster base, else (LB10) acts as AL. */
        for (sz_size_t k = 0; k < count; ++k) {
            if (!sz_line_break_is_cm_or_zwj_(raw[k])) continue;
            if (k == 0) {
                eff[k] = sz_line_break_al_k;
                continue;
            } // LB10 (start of text / open line)
            sz_u8_t base = eff[k - 1];
            if (base != sz_line_break_bk_k && base != sz_line_break_cr_k && base != sz_line_break_lf_k &&
                base != sz_line_break_nl_k && base != sz_line_break_sp_k && base != sz_line_break_zw_k)
                eff[k] = base;                // LB9 attach
            else eff[k] = sz_line_break_al_k; // LB10
        }

        /* `local_line_start` is the open line's start expressed as a byte offset; it begins at the window origin
         *  and advances with each emitted opportunity. Spans are emitted with absolute byte offsets from `start`. */
        sz_size_t local_line_start = line_start;
        sz_bool_t hit_capacity = sz_false_k;
        for (sz_size_t k = 1; k < count; ++k) {
            sz_u8_t b = eff[k - 1];
            sz_u8_t a = eff[k];
            sz_bool_t is_break = sz_false_k;

            /* last non-space context: nearest j<=k-1 with eff[j] != SP, plus whether spaces were skipped. */
            sz_size_t j = k - 1;
            sz_bool_t had_space = sz_false_k;
            while (j > 0 && eff[j] == sz_line_break_sp_k) {
                had_space = sz_true_k;
                --j;
            }
            sz_u8_t before_sp = eff[j];

            /* cluster base index of the codepoint at k-1 (skip attached marks). */
            sz_size_t b_base = sz_line_break_cluster_base_(raw, eff, k - 1);

            if (b == sz_line_break_bk_k) {
                is_break = sz_true_k;
                goto decided;
            } // LB4
            if (b == sz_line_break_cr_k && a == sz_line_break_lf_k) { goto decided; } // LB5
            if (b == sz_line_break_cr_k || b == sz_line_break_lf_k || b == sz_line_break_nl_k) {
                is_break = sz_true_k;
                goto decided;
            }
            if (a == sz_line_break_bk_k || a == sz_line_break_cr_k || a == sz_line_break_lf_k ||
                a == sz_line_break_nl_k) {
                goto decided;
            } // LB6
            if (a == sz_line_break_sp_k || a == sz_line_break_zw_k) { goto decided; } // LB7
            if (before_sp == sz_line_break_zw_k && had_space) {
                is_break = sz_true_k;
                goto decided;
            } // LB8
            if (b == sz_line_break_zw_k) {
                is_break = sz_true_k;
                goto decided;
            }
            if (prev_zwj[k]) { goto decided; } // LB8a: no break after ZWJ (preserved across the cluster collapse)
            if (sz_line_break_is_cm_or_zwj_(raw[k]) && b != sz_line_break_bk_k && b != sz_line_break_cr_k &&
                b != sz_line_break_lf_k && b != sz_line_break_nl_k && b != sz_line_break_sp_k &&
                b != sz_line_break_zw_k) {
                goto decided;
            } // LB9
            if (a == sz_line_break_wj_k || b == sz_line_break_wj_k) { goto decided; } // LB11
            if (b == sz_line_break_gl_k) { goto decided; }                            // LB12
            if (a == sz_line_break_gl_k && !(b == sz_line_break_sp_k || b == sz_line_break_ba_k ||
                                             b == sz_line_break_hy_k || b == sz_line_break_hh_k)) {
                goto decided;
            } // LB12a
            if (a == sz_line_break_cl_k || a == sz_line_break_cp_k || a == sz_line_break_ex_k ||
                a == sz_line_break_sy_k) {
                goto decided;
            } // LB13
            if (before_sp == sz_line_break_op_k) { goto decided; } // LB14
            /* LB15a: opening Pi quote after an allowed left context, across spaces. */
            if (before_sp == sz_line_break_qu_k &&
                sz_line_break_descriptor_is_pi_(descriptor[sz_line_break_cluster_base_(raw, eff, j)])) {
                sz_size_t q_base = sz_line_break_cluster_base_(raw, eff, j);
                sz_bool_t left_ok;
                if (q_base == 0) left_ok = sz_true_k;
                else {
                    sz_u8_t lj = eff[q_base - 1];
                    left_ok =
                        (sz_bool_t)(lj == sz_line_break_bk_k || lj == sz_line_break_cr_k || lj == sz_line_break_lf_k ||
                                    lj == sz_line_break_nl_k || lj == sz_line_break_op_k || lj == sz_line_break_qu_k ||
                                    lj == sz_line_break_gl_k || lj == sz_line_break_sp_k || lj == sz_line_break_zw_k);
                }
                if (left_ok) { goto decided; }
            }
            /* LB15b: closing Pf quote followed by an allowed class or end-of-text. */
            if (a == sz_line_break_qu_k && sz_line_break_descriptor_is_pf_(descriptor[k])) {
                sz_bool_t right_ok;
                if (k + 1 >= count) right_ok = sz_true_k;
                else {
                    sz_u8_t rj = eff[k + 1];
                    right_ok =
                        (sz_bool_t)(rj == sz_line_break_sp_k || rj == sz_line_break_gl_k || rj == sz_line_break_wj_k ||
                                    rj == sz_line_break_cl_k || rj == sz_line_break_qu_k || rj == sz_line_break_cp_k ||
                                    rj == sz_line_break_ex_k || rj == sz_line_break_is_k || rj == sz_line_break_sy_k ||
                                    rj == sz_line_break_bk_k || rj == sz_line_break_cr_k || rj == sz_line_break_lf_k ||
                                    rj == sz_line_break_nl_k || rj == sz_line_break_zw_k);
                }
                if (right_ok) { goto decided; }
            }
            if (b == sz_line_break_sp_k && a == sz_line_break_is_k && k + 1 < count &&
                eff[k + 1] == sz_line_break_nu_k) {
                is_break = sz_true_k;
                goto decided;
            } // LB15.3
            if (a == sz_line_break_is_k) { goto decided; } // LB15.4
            if ((before_sp == sz_line_break_cl_k || before_sp == sz_line_break_cp_k) && a == sz_line_break_ns_k) {
                goto decided; // LB16
            }
            if (before_sp == sz_line_break_b2_k && a == sz_line_break_b2_k) { goto decided; } // LB17
            if (b == sz_line_break_sp_k) {
                is_break = sz_true_k;
                goto decided;
            } // LB18
            /* LB19 (East-Asian-aware quotation). */
            if (a == sz_line_break_qu_k && !sz_line_break_descriptor_is_pi_(descriptor[k])) { goto decided; }
            if (b == sz_line_break_qu_k && !sz_line_break_descriptor_is_pf_(descriptor[b_base])) { goto decided; }
            if (a == sz_line_break_qu_k && !sz_line_break_descriptor_is_eaw_(descriptor[b_base])) { goto decided; }
            if (a == sz_line_break_qu_k && (k + 1 >= count || !sz_line_break_descriptor_is_eaw_(descriptor[k + 1]))) {
                goto decided;
            }
            if (b == sz_line_break_qu_k && !sz_line_break_descriptor_is_eaw_(descriptor[k])) { goto decided; }
            if (b == sz_line_break_qu_k &&
                (b_base == 0 ||
                 !sz_line_break_descriptor_is_eaw_(descriptor[sz_line_break_cluster_base_(raw, eff, b_base - 1)]))) {
                goto decided;
            }
            if (a == sz_line_break_cb_k || b == sz_line_break_cb_k) {
                is_break = sz_true_k;
                goto decided;
            } // LB20
            /* LB20a: (sot|allowed) (HY|HH) x (AL|HL). */
            if ((b == sz_line_break_hy_k || b == sz_line_break_hh_k) &&
                (a == sz_line_break_al_k || a == sz_line_break_hl_k)) {
                sz_bool_t left_ctx;
                if (b_base == 0) left_ctx = sz_true_k;
                else {
                    sz_u8_t lj = eff[b_base - 1];
                    left_ctx = (sz_bool_t)(lj == sz_line_break_bk_k || lj == sz_line_break_cr_k ||
                                           lj == sz_line_break_lf_k || lj == sz_line_break_nl_k ||
                                           lj == sz_line_break_sp_k || lj == sz_line_break_zw_k ||
                                           lj == sz_line_break_cb_k || lj == sz_line_break_gl_k);
                }
                if (left_ctx) { goto decided; }
            }
            if (a == sz_line_break_ba_k || a == sz_line_break_hy_k || a == sz_line_break_hh_k ||
                a == sz_line_break_ns_k || b == sz_line_break_bb_k) {
                goto decided;
            } // LB21
            if (b_base > 0) { // LB21a: HL (HY|HH) x [^HL] -- the HL precedes the (HY|HH) cluster (skip its marks)
                if (eff[b_base - 1] == sz_line_break_hl_k && (b == sz_line_break_hy_k || b == sz_line_break_hh_k) &&
                    a != sz_line_break_hl_k) {
                    goto decided;
                }
            }
            if (b == sz_line_break_sy_k && a == sz_line_break_hl_k) { goto decided; } // LB21b
            if (a == sz_line_break_in_k) { goto decided; }                            // LB22
            if ((b == sz_line_break_al_k || b == sz_line_break_hl_k) && a == sz_line_break_nu_k) {
                goto decided;
            } // LB23
            if (b == sz_line_break_nu_k && (a == sz_line_break_al_k || a == sz_line_break_hl_k)) { goto decided; }
            if (b == sz_line_break_pr_k &&
                (a == sz_line_break_id_k || a == sz_line_break_eb_k || a == sz_line_break_em_k)) {
                goto decided;
            } // LB23a
            if ((b == sz_line_break_id_k || b == sz_line_break_eb_k || b == sz_line_break_em_k) &&
                a == sz_line_break_po_k) {
                goto decided;
            }
            if ((b == sz_line_break_pr_k || b == sz_line_break_po_k) &&
                (a == sz_line_break_al_k || a == sz_line_break_hl_k)) {
                goto decided;
            } // LB24
            if ((b == sz_line_break_al_k || b == sz_line_break_hl_k) &&
                (a == sz_line_break_pr_k || a == sz_line_break_po_k)) {
                goto decided;
            }
            /* LB25: numeric clusters. */
            {
                sz_bool_t left_nu_run = sz_false_k;
                sz_size_t m = k;
                while (m > 0 && (eff[m - 1] == sz_line_break_sy_k || eff[m - 1] == sz_line_break_is_k)) --m;
                if (m > 0 && eff[m - 1] == sz_line_break_nu_k) left_nu_run = sz_true_k;
                sz_bool_t left_nu_run_then_close = sz_false_k;
                if (b == sz_line_break_cl_k || b == sz_line_break_cp_k) {
                    sz_size_t mm = k - 1;
                    while (mm > 0 && (eff[mm - 1] == sz_line_break_sy_k || eff[mm - 1] == sz_line_break_is_k)) --mm;
                    if (mm > 0 && eff[mm - 1] == sz_line_break_nu_k) left_nu_run_then_close = sz_true_k;
                }
                if (left_nu_run_then_close && (a == sz_line_break_po_k || a == sz_line_break_pr_k)) { goto decided; }
                if (left_nu_run && (a == sz_line_break_po_k || a == sz_line_break_pr_k)) { goto decided; }
                if (b == sz_line_break_po_k && a == sz_line_break_op_k && k + 1 < count &&
                    eff[k + 1] == sz_line_break_nu_k) {
                    goto decided;
                }
                if (b == sz_line_break_po_k && a == sz_line_break_op_k && k + 2 < count &&
                    eff[k + 1] == sz_line_break_is_k && eff[k + 2] == sz_line_break_nu_k) {
                    goto decided;
                }
                if (b == sz_line_break_po_k && a == sz_line_break_nu_k) { goto decided; }
                if (b == sz_line_break_pr_k && a == sz_line_break_op_k && k + 1 < count &&
                    eff[k + 1] == sz_line_break_nu_k) {
                    goto decided;
                }
                if (b == sz_line_break_pr_k && a == sz_line_break_op_k && k + 2 < count &&
                    eff[k + 1] == sz_line_break_is_k && eff[k + 2] == sz_line_break_nu_k) {
                    goto decided;
                }
                if (b == sz_line_break_pr_k && a == sz_line_break_nu_k) { goto decided; }
                if (b == sz_line_break_hy_k && a == sz_line_break_nu_k) { goto decided; }
                if (b == sz_line_break_is_k && a == sz_line_break_nu_k) { goto decided; }
                if (left_nu_run && a == sz_line_break_nu_k) { goto decided; }
            }
            /* LB26 / LB27 Hangul. */
            if (b == sz_line_break_jl_k && (a == sz_line_break_jl_k || a == sz_line_break_jv_k ||
                                            a == sz_line_break_h2_k || a == sz_line_break_h3_k)) {
                goto decided;
            }
            if ((b == sz_line_break_jv_k || b == sz_line_break_h2_k) &&
                (a == sz_line_break_jv_k || a == sz_line_break_jt_k)) {
                goto decided;
            }
            if ((b == sz_line_break_jt_k || b == sz_line_break_h3_k) && a == sz_line_break_jt_k) { goto decided; }
            if ((b == sz_line_break_jl_k || b == sz_line_break_jv_k || b == sz_line_break_jt_k ||
                 b == sz_line_break_h2_k || b == sz_line_break_h3_k) &&
                a == sz_line_break_po_k) {
                goto decided;
            }
            if (b == sz_line_break_pr_k &&
                (a == sz_line_break_jl_k || a == sz_line_break_jv_k || a == sz_line_break_jt_k ||
                 a == sz_line_break_h2_k || a == sz_line_break_h3_k)) {
                goto decided;
            }
            if ((b == sz_line_break_al_k || b == sz_line_break_hl_k) &&
                (a == sz_line_break_al_k || a == sz_line_break_hl_k)) {
                goto decided;
            } // LB28
            /* LB28a: aksara clusters; Dotted_Circle acts as an aksara base. */
            {
                sz_bool_t b_dotted = sz_line_break_descriptor_is_dotted_circle_(descriptor[b_base]);
                sz_bool_t a_dotted = sz_line_break_descriptor_is_dotted_circle_(descriptor[k]);
                sz_bool_t b_is_base = (sz_bool_t)(b == sz_line_break_ak_k || b == sz_line_break_as_k || b_dotted);
                sz_bool_t a_is_base = (sz_bool_t)(a == sz_line_break_ak_k || a == sz_line_break_as_k || a_dotted);
                sz_bool_t a_base_or_dc = (sz_bool_t)(a == sz_line_break_ak_k || a_dotted);
                if (b == sz_line_break_ap_k && a_is_base) { goto decided; }                              // 28.11
                if (b_is_base && (a == sz_line_break_vf_k || a == sz_line_break_vi_k)) { goto decided; } // 28.12
                if (b == sz_line_break_vi_k && a_base_or_dc) {                                           // 28.13
                    sz_size_t vi_pos = b_base;
                    if (vi_pos > 0) {
                        sz_size_t pre = sz_line_break_cluster_base_(raw, eff, vi_pos - 1);
                        sz_u8_t pc = eff[pre];
                        if (pc == sz_line_break_ak_k || pc == sz_line_break_as_k ||
                            sz_line_break_descriptor_is_dotted_circle_(descriptor[pre])) {
                            goto decided;
                        }
                    }
                }
                if (b_is_base && a_is_base && k + 1 < count && eff[k + 1] == sz_line_break_vf_k) {
                    goto decided;
                } // 28.14
            }
            if (b == sz_line_break_is_k && (a == sz_line_break_al_k || a == sz_line_break_hl_k)) {
                goto decided;
            } // LB29
            /* LB30. */
            if ((b == sz_line_break_al_k || b == sz_line_break_hl_k || b == sz_line_break_nu_k) &&
                a == sz_line_break_op_k && !sz_line_break_descriptor_is_eaw_(descriptor[k])) {
                goto decided;
            }
            if (b == sz_line_break_cp_k &&
                (a == sz_line_break_al_k || a == sz_line_break_hl_k || a == sz_line_break_nu_k) &&
                !sz_line_break_descriptor_is_eaw_(descriptor[b_base])) {
                goto decided;
            }
            /* LB30a: RI RI keep pairs. */
            if (b == sz_line_break_ri_k && a == sz_line_break_ri_k) {
                sz_size_t ri = 0;
                sz_size_t scan = b_base;
                sz_bool_t scanning = sz_true_k;
                while (scanning && eff[scan] == sz_line_break_ri_k && raw[scan] == sz_line_break_ri_k) {
                    ++ri;
                    if (scan == 0) break;
                    scan = sz_line_break_cluster_base_(raw, eff, scan - 1);
                }
                if (ri % 2 == 1) { goto decided; }
            }
            /* LB30b. */
            if (b == sz_line_break_eb_k && a == sz_line_break_em_k) { goto decided; }
            if (a == sz_line_break_em_k && sz_line_break_descriptor_is_extpict_cn_(descriptor[b_base])) {
                goto decided;
            }

            is_break = sz_true_k; // LB31

        decided:
            /* A forced cut leaves the last two codepoints without real right context (the LB lookahead reaches at most
         *  k+2), so on such a window only opportunities with two further in-window codepoints are committed; the
         *  open line resumes in the next call where full context is available. A true tail commits every break. */
            if (is_break && (window_is_tail || k + 2 < count)) {
                if (lines == lines_capacity) {
                    hit_capacity = sz_true_k;
                    break;
                }
                line_starts[lines] = local_line_start;
                line_lengths[lines] = start[k] - local_line_start;
                ++lines;
                local_line_start = start[k];
            }
        }

        if (hit_capacity) {
            if (bytes_consumed) *bytes_consumed = local_line_start;
            return lines;
        }
        if (window_is_tail) {
            /* The trailing span [local_line_start, length) is the final line; end of text is a boundary (LB3). */
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
        /* Forced cut: the open line is unresolved. Re-anchor the next window at it (guaranteed progress as long as a
     *  single line fits the window). If no break committed in this window, advance past the window to avoid a
     *  stall on a pathologically long line, mirroring the resume-offset contract. */
        line_start = (local_line_start > line_start) ? local_line_start : window_end;
    }

    if (bytes_consumed) *bytes_consumed = line_start;
    return lines;
}

#pragma endregion // UAX-14 Line Boundaries

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_LINEWRAPS_SERIAL_H_
