/**
 *  @brief Serial backend for UAX-29 sentence boundaries.
 *  @file include/stringzilla/utf8_sentences/serial.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_SENTENCES_SERIAL_H_
#define STRINGZILLA_UTF8_SENTENCES_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_sentences/tables.h"
#include "stringzilla/utf8_runes/serial.h" // shared decode helpers

#ifdef __cplusplus
extern "C" {
#endif

#pragma region UAX 29 Sentence Boundaries

/** @brief Returns the UAX-29 Sentence_Break property (0-14) for a codepoint. */
SZ_API_COMPTIME sz_u8_t sz_rune_sentence_break_property(sz_rune_t rune) {
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
SZ_HELPER_INLINE sz_bool_t sz_sentence_break_is_parasep_(sz_u8_t property) {
    return (sz_bool_t)(property == sz_sentence_break_sep_k || property == sz_sentence_break_cr_k ||
                       property == sz_sentence_break_lf_k);
}
/** @brief True for a Sentence_Break SATerm (STerm or ATerm). */
SZ_HELPER_INLINE sz_bool_t sz_sentence_break_is_saterm_(sz_u8_t property) {
    return (sz_bool_t)(property == sz_sentence_break_sterm_k || property == sz_sentence_break_aterm_k);
}
/** @brief True for an SB5-transparent character (Extend or Format). */
SZ_HELPER_INLINE sz_bool_t sz_sentence_break_is_transparent_(sz_u8_t property) {
    return (sz_bool_t)(property == sz_sentence_break_extend_k || property == sz_sentence_break_format_k);
}
/** @brief SB8 stop set excluding Lower: OLetter, Upper, ParaSep, or SATerm — a significant class that ends the
 *         ATerm neutral run and confirms the deferred SB11 break (a Lower in the run suppresses it instead). */
SZ_HELPER_INLINE sz_bool_t sz_sentence_break_sb8_stops_(sz_u8_t property) {
    return (sz_bool_t)(property == sz_sentence_break_oletter_k || property == sz_sentence_break_upper_k ||
                       sz_sentence_break_is_parasep_(property) || sz_sentence_break_is_saterm_(property));
}

/**
 *  @brief Start offset of the codepoint after @p position: the next non-continuation byte, or @p length.
 *         Mirrors the SIMD `sz_utf8_rune_decode_window_` codepoint-start convention so the serial and
 *         Ice Lake backends segment malformed input identically (UAX-29 leaves ill-formed bytes undefined).
 */
SZ_HELPER_AUTO sz_size_t sz_sentence_break_next_start_(sz_cptr_t text, sz_size_t length, sz_size_t position) {
    sz_size_t next = position + 1;
    while (next < length && ((sz_u8_t)text[next] & 0xC0) == 0x80) ++next;
    return next;
}

/**
 *  @brief Sentence_Break property of the codepoint starting at `start`, decoded BLINDLY to mirror the SIMD
 *         `sz_utf8_rune_decode_window_`: the lead's strict length class (2-byte `110xxxxx`, 3-byte
 *         `1110xxxx`, 4-byte `11110xxx`; everything else single-byte) selects how many following bytes are
 *         folded in with no continuation/overlong/surrogate validation, missing trailing bytes read as zero.
 *         Valid UTF-8 decodes identically to the checked path; only ill-formed input differs, by design.
 */
SZ_HELPER_AUTO sz_u8_t sz_sentence_break_property_at_(sz_cptr_t text, sz_size_t length, sz_size_t start) {
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
 *  @brief Forward run-state carried across codepoints by the bulk segmenter so the SB3..SB11 rules resolve in O(1)
 *         per codepoint — the scalar twin of the Ice Lake register carry, mirroring `sz_grapheme_serial_state_t`.
 *         The trailing `SATerm Close* Sp*` context and the Extend/Format-transparent significant chain are tracked
 *         forward here, so the bulk driver decodes each codepoint once with no backward re-scan.
 */
typedef struct sz_sentence_serial_state_t {
    /** Raw SB property of the immediately previous codepoint (SB3 CR x LF, SB4). */
    sz_u8_t previous_property;
    /** SB property of the last non-Extend/Format codepoint (the SB6 / SB7 `eff_before`). */
    sz_u8_t previous_significant;
    /** The significant codepoint before `previous_significant` (SB7). */
    sz_u8_t before_significant;
    /** Open `SATerm Close* Sp*` context terminator (ATerm / STerm), else Other = none. */
    sz_u8_t terminator;
    /** A Close followed the terminator (before any Sp). */
    sz_bool_t terminator_saw_close;
    /** An Sp followed the terminator (and the Close* run). */
    sz_bool_t terminator_saw_space;
    /** A codepoint has been processed. */
    sz_bool_t has_previous;
} sz_sentence_serial_state_t;

/** @brief Advance @p state by the @p current codepoint property: update the significant chain and the
 *         `SATerm Close* Sp*` terminator context (Extend / Format are transparent and leave both unchanged). */
SZ_HELPER_AUTO void sz_sentence_serial_advance_(sz_sentence_serial_state_t *state, sz_u8_t current) {
    if (!sz_sentence_break_is_transparent_(current)) {
        state->before_significant = state->previous_significant;
        state->previous_significant = current;
        if (sz_sentence_break_is_saterm_(current)) { // ATerm / STerm opens a fresh terminator context
            state->terminator = current;
            state->terminator_saw_close = sz_false_k;
            state->terminator_saw_space = sz_false_k;
        }
        else if (state->terminator != (sz_u8_t)sz_sentence_break_other_k) {
            // The context is `terminator Close* Sp*`: a Close extends it only before any Sp, an Sp extends it, and
            // any other significant codepoint (including a ParaSep) closes it.
            if (current == sz_sentence_break_close_k && !state->terminator_saw_space)
                state->terminator_saw_close = sz_true_k;
            else if (current == sz_sentence_break_sp_k) state->terminator_saw_space = sz_true_k;
            else state->terminator = (sz_u8_t)sz_sentence_break_other_k;
        }
    }
    state->previous_property = current;
    state->has_previous = sz_true_k;
}

/** @brief A boundary verdict between two codepoints. SB8's Lower-lookahead cannot be resolved against `after` alone, so
 *         the ATerm-neutral case yields `pending`: the driver carries it forward and resolves it at the next
 *         significant codepoint (a Lower suppresses, an SB8 stop confirms), settling as a break at end-of-text. */
typedef enum sz_sentence_decision_t {
    sz_sentence_decision_no_break_k = 0,
    sz_sentence_decision_break_k = 1,
    sz_sentence_decision_pending_k = 2,
} sz_sentence_decision_t;

/** @brief Boundary decision (SB3..SB998) between @p state's previous codepoint and the @p after codepoint, in O(1) with
 *         no forward re-scan: SB8's Lower-lookahead is deferred as `pending` and resolved forward by the driver. */
SZ_HELPER_AUTO sz_sentence_decision_t sz_sentence_serial_boundary_(sz_sentence_serial_state_t const *state,
                                                                   sz_u8_t after) {
    sz_u8_t const before = state->previous_property;
    if (before == sz_sentence_break_cr_k && after == sz_sentence_break_lf_k)
        return sz_sentence_decision_no_break_k;                                     // SB3
    if (sz_sentence_break_is_parasep_(before)) return sz_sentence_decision_break_k; // SB4
    if (after == sz_sentence_break_extend_k || after == sz_sentence_break_format_k)
        return sz_sentence_decision_no_break_k; // SB5

    sz_u8_t const eff_before = state->previous_significant; // Extend/Format-transparent effective preceding class
    if (eff_before == sz_sentence_break_aterm_k && after == sz_sentence_break_numeric_k)
        return sz_sentence_decision_no_break_k; // SB6
    if (eff_before == sz_sentence_break_aterm_k && after == sz_sentence_break_upper_k &&
        (state->before_significant == sz_sentence_break_upper_k ||
         state->before_significant == sz_sentence_break_lower_k))
        return sz_sentence_decision_no_break_k; // SB7

    if (sz_sentence_break_is_saterm_(
            state->terminator)) { // an `SATerm Close* Sp*` context ends at the previous codepoint
        if (state->terminator == sz_sentence_break_aterm_k && after == sz_sentence_break_lower_k)
            return sz_sentence_decision_no_break_k; // SB8, Lower immediately after the context: never a boundary
        if (after == sz_sentence_break_scontinue_k || sz_sentence_break_is_saterm_(after))
            return sz_sentence_decision_no_break_k; // SB8a
        if (!state->terminator_saw_space && (after == sz_sentence_break_close_k || after == sz_sentence_break_sp_k ||
                                             sz_sentence_break_is_parasep_(after)))
            return sz_sentence_decision_no_break_k; // SB9
        if (after == sz_sentence_break_sp_k || sz_sentence_break_is_parasep_(after))
            return sz_sentence_decision_no_break_k; // SB10
        if (state->terminator == sz_sentence_break_aterm_k && !sz_sentence_break_sb8_stops_(after))
            return sz_sentence_decision_pending_k; // SB8 undecided: a Lower may still follow through the neutral run
        return sz_sentence_decision_break_k;       // SB11
    }
    return sz_sentence_decision_no_break_k; // SB998
}

/** @brief Append the sentence ending at @p boundary to the output arrays and re-anchor the running start. Returns
 *         sz_false_k when the capacity is exhausted (no room): the caller stops and reports the emitted prefix. */
SZ_HELPER_AUTO sz_bool_t sz_sentence_serial_emit_(sz_size_t boundary, sz_size_t *sentence_starts,
                                                  sz_size_t *sentence_lengths, sz_size_t sentences_capacity,
                                                  sz_size_t *sentences, sz_size_t *sentence_start) {
    if (*sentences == sentences_capacity) return sz_false_k;
    sentence_starts[*sentences] = *sentence_start;
    sentence_lengths[*sentences] = boundary - *sentence_start;
    ++(*sentences);
    *sentence_start = boundary;
    return sz_true_k;
}

/**
 *  @brief Plural UAX-29 sentence segmentation: ONE forward sweep emits every sentence into parallel
 *         `sentence_starts` / `sentence_lengths`, carrying the SB run-state so each codepoint is decoded once (O(n),
 *         no backward re-walks).
 */
SZ_API_COMPTIME sz_size_t sz_utf8_sentences_serial(          //
    sz_cptr_t text, sz_size_t length,                        //
    sz_size_t *sentence_starts, sz_size_t *sentence_lengths, //
    sz_size_t sentences_capacity, sz_size_t *bytes_consumed) {

    sz_size_t sentences = 0;
    if (length == 0 || sentences_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_sentence_serial_state_t state;
    state.previous_property = (sz_u8_t)sz_sentence_break_other_k;
    state.previous_significant = (sz_u8_t)sz_sentence_break_other_k;
    state.before_significant = (sz_u8_t)sz_sentence_break_other_k;
    state.terminator = (sz_u8_t)sz_sentence_break_other_k;
    state.terminator_saw_close = sz_false_k;
    state.terminator_saw_space = sz_false_k;
    state.has_previous = sz_false_k;
    sz_sentence_serial_advance_(&state, sz_sentence_break_property_at_(text, length, 0)); // seed from codepoint 0

    sz_size_t sentence_start = 0;
    sz_size_t position = sz_sentence_break_next_start_(text, length, 0);
    sz_bool_t boundary_pending = sz_false_k; // a deferred SB8 verdict awaits the next significant codepoint
    sz_size_t boundary_pending_position = 0; // the byte offset that boundary would carry if confirmed
    while (position < length) {
        sz_u8_t const after = sz_sentence_break_property_at_(text, length, position);
        if (boundary_pending) {
            if (after == sz_sentence_break_lower_k) { boundary_pending = sz_false_k; } // SB8: Lower → suppress
            else if (sz_sentence_break_sb8_stops_(after)) { // stop → confirm the deferred break
                if (!sz_sentence_serial_emit_(boundary_pending_position, sentence_starts, sentence_lengths,
                                              sentences_capacity, &sentences, &sentence_start)) {
                    if (bytes_consumed) *bytes_consumed = sentence_start;
                    return sentences;
                }
                boundary_pending = sz_false_k;
            }
            else { // a neutral codepoint extends the SB8 run: keep the deferred verdict, no boundary here
                sz_sentence_serial_advance_(&state, after);
                position = sz_sentence_break_next_start_(text, length, position);
                continue;
            }
        }
        sz_sentence_decision_t const decision = sz_sentence_serial_boundary_(&state, after);
        if (decision == sz_sentence_decision_break_k) {
            if (!sz_sentence_serial_emit_(position, sentence_starts, sentence_lengths, sentences_capacity, &sentences,
                                          &sentence_start)) {
                if (bytes_consumed) *bytes_consumed = sentence_start;
                return sentences;
            }
        }
        else if (decision == sz_sentence_decision_pending_k) {
            boundary_pending = sz_true_k;
            boundary_pending_position = position;
        }
        sz_sentence_serial_advance_(&state, after);
        position = sz_sentence_break_next_start_(text, length, position);
    }

    // End of text: no Lower can follow, so any deferred SB8 verdict settles as a break before the final sentence.
    if (boundary_pending && !sz_sentence_serial_emit_(boundary_pending_position, sentence_starts, sentence_lengths,
                                                      sentences_capacity, &sentences, &sentence_start)) {
        if (bytes_consumed) *bytes_consumed = sentence_start;
        return sentences;
    }
    if (!sz_sentence_serial_emit_(length, sentence_starts, sentence_lengths, sentences_capacity, &sentences,
                                  &sentence_start)) {
        if (bytes_consumed) *bytes_consumed = sentence_start;
        return sentences;
    }
    if (bytes_consumed) *bytes_consumed = length;
    return sentences;
}

#pragma endregion // UAX 29 Sentence Boundaries

#pragma region Portable dense rule engine

/**
 *  @brief  Cross-window register carry: the open `SATerm Close* Sp*` shadow run-state, the trailing significant /
 *          raw classes the bounded-lookback rules (SB3/SB4/SB6/SB7) need at the next block head, plus the SB8
 *          "neutral chain awaiting a Lower verdict" pending state that threads an unbounded right context across
 *          windows without any serial re-walk or oracle call. The portable engine carry, shared verbatim by every
 *          ISA backend (the icelake and haswell extractors only build the dense class stream that feeds it).
 */
typedef struct sz_utf8_sentence_break_carry_t {
    /** A SATerm has opened a shadow that is still open at the previous block's edge. */
    sz_u8_t in_shadow;
    /** The opening terminator was ATerm (matters for SB8). */
    sz_u8_t shadow_aterm;
    /** At least one Sp seen since the terminator (SB9 vs SB10). */
    sz_u8_t shadow_saw_sp;
    /** False only at the very start of the text (SB1). */
    sz_u8_t have_prev;
    /** Class of the last significant codepoint of the previous block. */
    sz_u8_t prev_eff;
    /** Class of the second-last significant codepoint. */
    sz_u8_t prev_prev_eff;
    /** Class of the last raw codepoint of the previous block (SB3/SB4 raw-before). */
    sz_u8_t prev_raw;
    /** An ATerm SB8 boundary deferred because its neutral right-context (no Close/Sp shadow) ran past the block edge;
     *  the awaited Lower-vs-stop verdict threads here across blocks. */
    sz_u8_t sb8_pending;
} sz_utf8_sentence_break_carry_t;

/** @brief  One classified block resolved into per-dense-codepoint break bits plus the metadata the driver stitches. */
typedef struct sz_utf8_sentence_break_window_t {
    /** Bit `i` set => a boundary begins before dense codepoint `i` (scattered to byte lanes by the driver). */
    sz_u64_t breaks;
    /** Exclusive upper bound, in dense codepoints, on lanes whose break bit is fully trusted (SB8 edge). */
    sz_size_t resolved;
    /** Verdict for an entering `carry->sb8_pending`: 0 still pending, 1 break, 2 no break. */
    sz_u8_t sb8_resolution;
} sz_utf8_sentence_break_window_t;

/** @brief  Per-class membership masks of the dense codepoint stream: bit `i` of `by_class[c]` set when dense lane `i`
 *          (in @p valid) carries Sentence_Break class `c`. The ISA-independent dense frame the rule engine consumes;
 *          built in ONE pass over @p dense_classes (a backend may instead fill it with vector compares - the engine
 *          only reads `by_class[...]`). 15 Sentence_Break classes (0..14). */
typedef struct sz_utf8_sentence_break_frame_t {
    sz_u64_t by_class[15]; /**< by_class[c] = lanes whose class == c, restricted to valid. */
} sz_utf8_sentence_break_frame_t;

/** @brief  Build @ref sz_utf8_sentence_break_frame_t from the dense class byte stream in one forward pass. */
SZ_HELPER_AUTO sz_utf8_sentence_break_frame_t sz_utf8_sentence_break_frame_from_dense_(sz_u8_t const *dense_classes,
                                                                                       sz_u64_t valid) {
    sz_utf8_sentence_break_frame_t frame;
    for (int cls = 0; cls < 15; ++cls) frame.by_class[cls] = 0;
    sz_u64_t remaining = valid;
    while (remaining) {
        int const lane = sz_u64_ctz(remaining);
        remaining &= remaining - 1;
        frame.by_class[dense_classes[lane]] |= (sz_u64_t)1 << lane;
    }
    return frame;
}

/**
 *  @brief  Resolve a block of @p count dense classified codepoints (lanes `0..count-1` in @p dense_classes, the open
 *          shadow and left context arriving in @p carry) into dense per-codepoint sentence-break bits. The ISA driver
 *          compacts the byte-lane classes to this dense stream and scatters the result back to byte lanes.
 *
 *  All of SB3-SB998 are pure 64-bit bit algebra over the per-class lane masks: the two-phase `SATerm Close* Sp*`
 *  shadow (`close_phase` then `shadow`), the SB8 in-window lower-ahead (`fill_left` of Lower through neutral via
 *  the shared substrate), the SB6/SB7 effective-previous chains across Extend/Format (the `flow` gate hops only the
 *  remaining ignorable codepoints, no continuation bytes), and the SB3/SB4 raw-before (a single `<< 1` on the dense
 *  stream). @p carry is updated with the trailing run-state. `breaks` bit 0 is the inter-block boundary (only
 *  meaningful when `have_prev`).
 *
 *  SB8's right context is unbounded; the in-window `fill_left` is exact unless a trailing neutral run reaches the
 *  block edge (the Lower may lie in the next block). `resolved` is clamped before such an undecided lane so the
 *  driver re-anchors the next window with full forward context - a register carry only, never a scalar re-walk or
 *  oracle call. Intrinsic-free: the only ISA-specific work (decode, classify, dense compaction, byte-lane scatter)
 *  lives in the backend extractor; this engine is shared by serial / icelake / haswell verbatim.
 */
SZ_HELPER_AUTO sz_utf8_sentence_break_window_t sz_utf8_sentence_break_decide_block_( //
    sz_utf8_sentence_break_frame_t const *frame, sz_u8_t const *dense_classes, sz_size_t count,
    sz_utf8_sentence_break_carry_t *carry, sz_bool_t more_text) {

    sz_u64_t const valid = (count >= 64) ? ~0ull : ((1ull << count) - 1);
    sz_u64_t const m_extend = frame->by_class[sz_sentence_break_extend_k];
    sz_u64_t const m_format = frame->by_class[sz_sentence_break_format_k];
    sz_u64_t const m_ignorable = m_extend | m_format;
    sz_u64_t const m_cr = frame->by_class[sz_sentence_break_cr_k];
    sz_u64_t const m_lf = frame->by_class[sz_sentence_break_lf_k];
    sz_u64_t const m_sep = frame->by_class[sz_sentence_break_sep_k];
    sz_u64_t const m_sp = frame->by_class[sz_sentence_break_sp_k];
    sz_u64_t const m_lower = frame->by_class[sz_sentence_break_lower_k];
    sz_u64_t const m_upper = frame->by_class[sz_sentence_break_upper_k];
    sz_u64_t const m_oletter = frame->by_class[sz_sentence_break_oletter_k];
    sz_u64_t const m_numeric = frame->by_class[sz_sentence_break_numeric_k];
    sz_u64_t const m_aterm = frame->by_class[sz_sentence_break_aterm_k];
    sz_u64_t const m_scont = frame->by_class[sz_sentence_break_scontinue_k];
    sz_u64_t const m_sterm = frame->by_class[sz_sentence_break_sterm_k];
    sz_u64_t const m_close = frame->by_class[sz_sentence_break_close_k];
    sz_u64_t const m_parasep = m_sep | m_cr | m_lf;
    sz_u64_t const m_saterm = m_aterm | m_sterm;
    sz_u64_t const significant = valid & ~m_ignorable;

    // Dense adjacency rides a `flow` gate that hops only over Extend/Format codepoints (SB5 transparency); the
    // continuation-byte half of the byte-lane `flow` is gone because the stream is already codepoint-dense.
    sz_u64_t const flow = m_ignorable;
    int const have_prev = carry->have_prev;
    sz_u8_t const prev_eff = carry->prev_eff;
    sz_u8_t const prev_prev_eff = carry->prev_prev_eff;
    sz_u8_t const prev_raw = carry->prev_raw;
    int const was_sb8_pending = carry->sb8_pending;

    // Leading edge region: lane 0 up to and including the first significant lead, where the cross-window carried
    // class is injected so lane-0 left context arrives as a register carry (no scalar re-walk).
    int const first_sig = significant ? sz_u64_ctz(significant) : -1;
    sz_u64_t const edge_region = (first_sig < 0) ? valid : sz_u64_mask_until_serial_((sz_size_t)first_sig + 1);

    // Effective-previous significant codepoint (SB6 ATerm, SB7 Upper/Lower two-back seeds).
    sz_u64_t const eb_aterm = (sz_u64_fill_right_(m_aterm & significant, flow) << 1) |
                              sz_u64_or_if_(0ull, edge_region, have_prev && prev_eff == sz_sentence_break_aterm_k);
    sz_u64_t const eb_upper = (sz_u64_fill_right_(m_upper & significant, flow) << 1) |
                              sz_u64_or_if_(0ull, edge_region, have_prev && prev_eff == sz_sentence_break_upper_k);
    sz_u64_t const eb_lower = (sz_u64_fill_right_(m_lower & significant, flow) << 1) |
                              sz_u64_or_if_(0ull, edge_region, have_prev && prev_eff == sz_sentence_break_lower_k);

    sz_u64_t const eb2_upper = (sz_u64_fill_right_(eb_upper & significant, flow) << 1) |
                               sz_u64_or_if_(0ull, edge_region,
                                             have_prev && prev_prev_eff == sz_sentence_break_upper_k);
    sz_u64_t const eb2_lower = (sz_u64_fill_right_(eb_lower & significant, flow) << 1) |
                               sz_u64_or_if_(0ull, edge_region,
                                             have_prev && prev_prev_eff == sz_sentence_break_lower_k);

    // Two-phase monotone `SATerm Close* Sp*` shadow. `flow` spans ignorables; only Close / Sp lead classes widen
    // each phase's gate. The carried open-run state seeds lane 0.
    sz_u64_t const gate_close = flow | m_close;
    sz_u64_t const gate_sp = flow | m_sp;
    sz_u64_t const in_shadow_carry = (sz_u64_t)(carry->in_shadow != 0);
    sz_u64_t const saw_sp_carry = (sz_u64_t)(carry->shadow_saw_sp != 0);
    sz_u64_t const aterm_carry = (sz_u64_t)(carry->shadow_aterm != 0);
    sz_u64_t const lane0_close = (m_ignorable | m_close) & 1ull;
    sz_u64_t const lane0_sp = (m_ignorable | m_sp) & 1ull;
    sz_u64_t const lane0_sp_only = m_sp & 1ull;
    sz_u64_t const open_no_sp = in_shadow_carry & ~saw_sp_carry;
    sz_u64_t const open_with_sp = in_shadow_carry & saw_sp_carry;
    sz_u64_t const carry_close_seed = sz_u64_or_if_(0ull, lane0_close, (int)open_no_sp);
    sz_u64_t const carry_sp_seed = sz_u64_or_if_(sz_u64_or_if_(0ull, lane0_sp_only, (int)open_no_sp), lane0_sp,
                                                 (int)open_with_sp);
    sz_u64_t const carry_aterm_close = sz_u64_or_if_(0ull, lane0_close, (int)(open_no_sp & aterm_carry));
    sz_u64_t const carry_aterm_sp = sz_u64_or_if_(sz_u64_or_if_(0ull, lane0_sp_only, (int)(open_no_sp & aterm_carry)),
                                                  lane0_sp, (int)(open_with_sp & aterm_carry));

    sz_u64_t const close_phase = sz_u64_fill_right_(m_saterm | carry_close_seed, gate_close) | carry_sp_seed;
    sz_u64_t const shadow = sz_u64_fill_right_(close_phase | carry_sp_seed, gate_sp);
    sz_u64_t const sp_in_shadow = (m_sp & shadow) | carry_sp_seed;
    sz_u64_t const saw_sp_upto = sz_u64_fill_right_(sp_in_shadow, gate_sp) & shadow;
    sz_u64_t const aterm_close_phase = sz_u64_fill_right_(m_aterm | carry_aterm_close, gate_close) | carry_aterm_sp;
    sz_u64_t const aterm_shadow = sz_u64_fill_right_(aterm_close_phase | carry_aterm_sp, gate_sp);

    // SB8 in-window lower-ahead: `fill_left` of Lower through the SB8 neutral set. In dense space a stop is one
    // lane, so no continuation-byte exclusion is needed; the neutral gate is just every non-stop codepoint.
    sz_u64_t const sb8_stop = m_lower | m_upper | m_oletter | m_parasep | m_saterm;
    sz_u64_t const neutral = valid & ~sb8_stop;
    sz_u64_t const lower_ahead = sz_u64_fill_left_(m_lower, neutral);
    sz_u64_t const top_bit = (count >= 64) ? (1ull << 63) : (1ull << (count - 1));
    sz_u64_t const neutral_to_edge = sz_u64_fill_left_(top_bit & neutral, neutral) & ~lower_ahead;

    // Raw immediately-preceding codepoint for SB3/SB4: dense `<<1` reads the previous codepoint regardless of
    // Extend/Format (raw, not SB5-skipped). Lane 0 seeds from the carried raw class.
    sz_u64_t raw_before_cr = (m_cr << 1);
    sz_u64_t raw_before_parasep = (m_parasep << 1);
    raw_before_cr = sz_u64_or_if_(raw_before_cr, 1ull, have_prev && prev_raw == sz_sentence_break_cr_k);
    raw_before_parasep = sz_u64_or_if_(
        raw_before_parasep, 1ull,
        have_prev && (prev_raw == sz_sentence_break_sep_k || prev_raw == sz_sentence_break_cr_k ||
                      prev_raw == sz_sentence_break_lf_k));

    sz_u64_t const r_sb3 = raw_before_cr & m_lf;
    sz_u64_t const r_sb4 = raw_before_parasep & valid;
    sz_u64_t const r_sb5 = m_ignorable;
    sz_u64_t const r_sb6 = eb_aterm & m_numeric;
    sz_u64_t const r_sb7 = (eb2_upper | eb2_lower) & eb_aterm & m_upper;

    sz_u64_t in_shadow_before = shadow << 1;
    sz_u64_t aterm_shadow_before = aterm_shadow << 1;
    sz_u64_t saw_sp_before = saw_sp_upto << 1;
    in_shadow_before = sz_u64_or_if_(in_shadow_before, 1ull, (int)in_shadow_carry);
    aterm_shadow_before = sz_u64_or_if_(aterm_shadow_before, 1ull, (int)(in_shadow_carry & aterm_carry));
    saw_sp_before = sz_u64_or_if_(saw_sp_before, 1ull, (int)(in_shadow_carry & saw_sp_carry));

    sz_u64_t const r_sb8 = aterm_shadow_before & in_shadow_before & lower_ahead;
    sz_u64_t const r_sb8a = in_shadow_before & (m_scont | m_saterm);
    sz_u64_t const r_sb9 = in_shadow_before & ~saw_sp_before & (m_close | m_sp | m_parasep);
    sz_u64_t const r_sb10 = in_shadow_before & (m_sp | m_parasep);
    sz_u64_t const r_sb11 = in_shadow_before & valid;

    sz_u64_t decided = 0, brk = 0;
    decided |= r_sb3 & ~decided;
    brk |= r_sb4 & ~decided, decided |= r_sb4 & ~decided;
    decided |= r_sb5 & ~decided;
    decided |= r_sb6 & ~decided;
    decided |= r_sb7 & ~decided;
    decided |= r_sb8 & ~decided;
    decided |= r_sb8a & ~decided;
    decided |= r_sb9 & ~decided;
    decided |= r_sb10 & ~decided;
    sz_u64_t const decided_pre_sb11 = decided;
    brk |= r_sb11 & ~decided, decided |= r_sb11 & ~decided;

    sz_u64_t const lowbit = have_prev ? 1ull : 0ull;
    sz_u64_t const produced = valid & (~1ull | lowbit);
    sz_u64_t breaks = brk & produced;

    sz_u64_t const undecided =
        more_text ? (aterm_shadow_before & in_shadow_before & neutral_to_edge & produced & ~decided_pre_sb11) : 0ull;
    sz_size_t resolved = count;
    if (undecided) {
        sz_size_t const lane = sz_u64_ctz(undecided);
        resolved = lane;
        breaks &= sz_u64_mask_until_serial_(lane);
    }

    // Update the carry from this window's high edge: trailing shadow run-state and the trailing class context.
    {
        int const edge = (int)count - 1;
        if (valid) carry->prev_raw = dense_classes[edge];
        if (significant) {
            int const last = 63 - sz_u64_clz(significant);
            carry->prev_eff = dense_classes[last];
            sz_u64_t const sig2 = significant & ~(1ull << last);
            if (sig2) carry->prev_prev_eff = dense_classes[63 - sz_u64_clz(sig2)];
        }
        carry->have_prev = 1;
        carry->in_shadow = (sz_u8_t)((shadow >> edge) & 1ull);
        carry->shadow_aterm = (sz_u8_t)((aterm_shadow >> edge) & 1ull);
        carry->shadow_saw_sp = (sz_u8_t)((saw_sp_upto >> edge) & 1ull);
    }

    sz_u8_t sb8_resolution = 0;
    if (was_sb8_pending) {
        if (((lower_ahead | m_lower) & 1ull) != 0) sb8_resolution = 2;
        else if (more_text && (neutral_to_edge & 1ull) != 0) sb8_resolution = 0;
        else sb8_resolution = 1;
    }
    carry->sb8_pending = (sz_u8_t)(was_sb8_pending && sb8_resolution == 0);

    sz_utf8_sentence_break_window_t result;
    result.breaks = breaks;
    result.resolved = resolved;
    result.sb8_resolution = sb8_resolution;
    return result;
}

/** @brief  Largest byte prefix of a decode window whose codepoints are all fully loaded, from the plain u64 lane
 *          masks - the mask-domain twin of the per-ISA `complete_limit` helpers, shared by back-ends that carry
 *          their window state as scalars. Never below 1 when the window is non-empty. */
SZ_HELPER_AUTO sz_size_t sz_utf8_sentence_break_complete_limit_masks_( //
    sz_size_t loaded, sz_u64_t start_bytes, sz_u64_t two_byte_starts, sz_u64_t three_byte_starts,
    sz_u64_t four_byte_starts, sz_u8_t byte_after, sz_bool_t more_text) {
    if (!more_text || !start_bytes) return loaded;
    sz_u64_t const straddle = (two_byte_starts & ~sz_u64_mask_until_serial_(loaded > 1 ? loaded - 1 : 0)) |
                              (three_byte_starts & ~sz_u64_mask_until_serial_(loaded > 2 ? loaded - 2 : 0)) |
                              (four_byte_starts & ~sz_u64_mask_until_serial_(loaded > 3 ? loaded - 3 : 0));
    sz_size_t limit = straddle ? (sz_size_t)sz_u64_ctz(straddle) : loaded;
    if ((byte_after & 0xC0) == 0x80) {
        sz_size_t const last_lead = (sz_size_t)(63 - sz_u64_clz(start_bytes));
        if (last_lead < limit) limit = last_lead;
    }
    return limit > 0 ? limit : loaded;
}

/** @brief  Emit the resolved dense sentence boundaries as `(start, length)` segments, walking the codepoint-start
 *          lanes once: dense index `j` maps to the `j`-th set bit of @p start_lanes, a set bit of @p dense_breaks
 *          below @p dense_limit closes the running segment at that byte, and the walk also reports the byte lane
 *          of the `dense_limit`-th start (the resume position of a partially resolved window). Honors @p capacity
 *          and the carried previous boundary exactly like @ref sz_utf8_rune_drain_forward_.
 *  @return Number of segments produced; sets @p advance_lane_out to the byte lane of start @p dense_limit, or
 *          @p loaded when every start resolved. */
SZ_HELPER_AUTO sz_size_t sz_utf8_sentence_break_emit_dense_serial_(                                     //
    sz_u64_t start_lanes, sz_u64_t dense_breaks, sz_size_t dense_limit, sz_size_t base, int skip_lane0, //
    sz_size_t loaded, sz_size_t *starts, sz_size_t *lengths, sz_size_t produced, sz_size_t capacity,
    sz_size_t *segment_start_io, sz_size_t *advance_lane_out) {
    sz_size_t segment_start = *segment_start_io;
    sz_size_t dense_index = 0;
    sz_size_t advance_lane = loaded;
    for (sz_u64_t remaining = start_lanes; remaining; remaining &= remaining - 1, ++dense_index) {
        sz_size_t const lane = (sz_size_t)sz_u64_ctz(remaining);
        if (dense_index == dense_limit) {
            advance_lane = lane;
            break;
        }
        if (((dense_breaks >> dense_index) & 1ull) == 0) continue;
        if (skip_lane0 && lane == 0) continue;
        if (produced == capacity) break;
        sz_size_t const boundary = base + lane;
        starts[produced] = segment_start;
        lengths[produced] = boundary - segment_start;
        segment_start = boundary;
        ++produced;
    }
    *segment_start_io = segment_start;
    *advance_lane_out = advance_lane;
    return produced;
}

/** @brief  Convenience entry for backends without vector class compares: build the dense frame in one pass, then run
 *          @ref sz_utf8_sentence_break_decide_block_. The vector backends build the frame with `vpcmpeqb` instead. */
SZ_HELPER_AUTO sz_utf8_sentence_break_window_t sz_utf8_sentence_break_decide_dense_( //
    sz_u8_t const *dense_classes, sz_size_t count, sz_utf8_sentence_break_carry_t *carry, sz_bool_t more_text) {
    sz_u64_t const valid = (count >= 64) ? ~0ull : ((1ull << count) - 1);
    sz_utf8_sentence_break_frame_t const frame = sz_utf8_sentence_break_frame_from_dense_(dense_classes, valid);
    return sz_utf8_sentence_break_decide_block_(&frame, dense_classes, count, carry, more_text);
}

#pragma endregion Portable dense rule engine

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_SENTENCES_SERIAL_H_
