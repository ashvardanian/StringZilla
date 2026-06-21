/**
 *  @brief Ice Lake backend for UAX-29 word boundaries: fully vectorized, zero-scalar on every path.
 *  @file include/stringzilla/utf8_words/icelake.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_WORDS_ICELAKE_H_
#define STRINGZILLA_UTF8_WORDS_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_words/tables.h"
#include "stringzilla/utf8_words/serial.h"
#include "stringzilla/utf8_codepoints/icelake.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_ICELAKE
#if defined(__clang__)
#pragma clang attribute push(                                                                          \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2")
#endif

/*  UAX-29 word boundary detection (Ice Lake / AVX-512), fully vectorized end-to-end.
 *
 *  Each 64-byte window is decoded by the shared codepoint substrate, classified to per-codepoint Word_Break
 *  properties entirely in-register (ASCII permute, big arithmetic ranges, a cp < 0x800 page LUT, the shared
 *  two-stage trie for the scattered BMP residue, and arithmetic ranges for every Supplementary-Plane codepoint),
 *  then the UAX-29 no-break algebra WB3-WB16 runs as branchless 64-lane bit operations. Two pieces of cross-window
 *  state thread through a register carry: the effective `previous_property` (`carry_left`) and the parity of the
 *  open Regional_Indicator run (`carry_ri_parity`, for WB15/WB16). No path scalar-walks codepoints, spills a ZMM
 *  to the stack, calls the serial oracle, or issues a gather; only the trusted sub-window with full in-register
 *  left and right context is emitted, and the driver advances by the last fully-resolved codepoint start. */

/** @brief  Left-context budget (bytes): WB6/WB7/WB11/WB12 look two codepoints back across WB4 ignorables, so each
 *          window carries this many real left lanes for the bridge (a Mid* between two letters) to be fully
 *          in-register; 18 bytes covers two 4-byte codepoints plus interleaved Extend/Format. Below this (the
 *          buffer head) the algebra falls back to the `left_property` seam injection. */
enum { sz_utf8_word_break_left_context_k = 18 };

/** @brief  Smear reach (steps): covers the widest 4-byte codepoint's continuation bytes for the WB3c/WB3d/RI
 *          adjacency smears (the letter/numeric reach uses unbounded `fill` instead). */
enum { sz_utf8_word_break_smear_steps_k = 12 };

/** @brief  Significant (non-ignorable) codepoints reserved at the window top, whose forward WB6/WB7 two-codepoint
 *          context spills past the edge; the cross-window carry re-resolves them. */
enum { sz_utf8_word_break_reserve_codepoints_k = 2 };

/** @brief  Reverse driver: lane the scan anchor `position` sits at within its window, kept well below the top so
 *          the data-dependent forward window resolves it on ignorable-dense input. */
enum { sz_utf8_word_break_reverse_position_lane_k = 33 };

#pragma region Gather Free Word_Break Classifier

/** @brief  AVX-512 classification of an all-ASCII 64-byte vector to WB properties via table lookup. */
SZ_INTERNAL __m512i sz_utf8_word_break_classify_ascii_icelake_(__m512i ascii_bytes) {
    // The 128-entry ASCII table fits in two ZMM registers; bit 6 of the 7-bit index selects the half.
    __m512i const low_table = _mm512_loadu_epi8(sz_utf8_word_break_property_ascii_);
    __m512i const high_table = _mm512_loadu_epi8(sz_utf8_word_break_property_ascii_ + 64);
    __mmask64 const high_half = _mm512_test_epi8_mask(ascii_bytes, _mm512_set1_epi8(0x40));
    __m512i const low_result = _mm512_permutexvar_epi8(ascii_bytes, low_table);
    __m512i const high_result = _mm512_permutexvar_epi8(ascii_bytes, high_table);
    return _mm512_mask_blend_epi8(high_half, low_result, high_result);
}

/** @brief  64-lane mask of bytes whose class equals @p value. */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_class_mask_(__m512i classes, sz_u8_t value) {
    return _cvtmask64_u64(_mm512_cmpeq_epi8_mask(classes, _mm512_set1_epi8((char)value)));
}

/**
 *  @brief  64-lane mask of lanes whose `(high, low)` 16-bit value lies inside any sorted `[lo, hi]` range. Used for
 *          the WSegSpace (WB3d) and Extended_Pictographic (WB3c) sets, which are NOT part of the 4-bit Word_Break
 *          model: each is a branchless per-lane lexicographic byte-pair compare, looped over the range list (the
 *          loop is over ranges, not lanes — every iteration resolves all 64 lanes, like the shared range scans).
 */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_range16_mask_( //
    __m512i high, __m512i low, sz_u16_t const *lo_table, sz_u16_t const *hi_table, int count) {
    __mmask64 hit = _cvtu64_mask64(0);
    for (int range = 0; range < count; ++range) {
        __m512i const lo_high = _mm512_set1_epi8((char)(lo_table[range] >> 8));
        __m512i const lo_low = _mm512_set1_epi8((char)(lo_table[range] & 0xFF));
        __m512i const hi_high = _mm512_set1_epi8((char)(hi_table[range] >> 8));
        __m512i const hi_low = _mm512_set1_epi8((char)(hi_table[range] & 0xFF));
        __mmask64 const at_or_above_low = _kor_mask64( //
            _mm512_cmp_epu8_mask(high, lo_high, _MM_CMPINT_NLE),
            _kand_mask64(_mm512_cmpeq_epi8_mask(high, lo_high), _mm512_cmp_epu8_mask(low, lo_low, _MM_CMPINT_NLT)));
        __mmask64 const at_or_below_high = _kor_mask64( //
            _mm512_cmp_epu8_mask(high, hi_high, _MM_CMPINT_LT),
            _kand_mask64(_mm512_cmpeq_epi8_mask(high, hi_high), _mm512_cmp_epu8_mask(low, hi_low, _MM_CMPINT_LE)));
        hit = _kor_mask64(hit, _kand_mask64(at_or_above_low, at_or_below_high));
    }
    return _cvtmask64_u64(hit);
}

/** @brief  Per-position "joined" mask for an all-ASCII chunk; exact for in-window i-2 and i+1 neighbours. */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_join_mask_ascii_(__m512i classes) {
    // ASCII has no Hebrew, so AHLetter == ALetter; reduce each class to a 64-lane bitmask and defer the rule
    // logic to the shared portable routine.
    return sz_utf8_word_break_join_from_class_masks_(
        sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_aletter_k),
        sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_numeric_k),
        sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_extendnumlet_k),
        sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_midletter_k),
        sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_midnum_k),
        sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_mid_quotes_k),
        sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_cr_k),
        sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_lf_k));
}

/**
 *  @brief  Look up one cp < 0x800 page over @ref sz_utf8_word_break_flat_lut_0800_ held in 32 ZMM registers and
 *          addressed by an in-register `vpermi2b` page network. `in_seven` selects within a 128-byte segment-pair;
 *          `page` selects the pair. Returns the per-lane class for every lane (caller masks to cp < 0x800 lanes).
 */
SZ_INTERNAL __m512i sz_utf8_word_break_small_page_(__m512i high, __m512i low) {
    __m512i const in_seven = _mm512_and_si512(low, _mm512_set1_epi8(0x7F));
    __m512i const low_high_bit = sz_utf8_codepoints_srl8_(low, 7, 0x01);
    __m512i const page = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(high, _mm512_set1_epi8(0x07)), 1),
                                         low_high_bit);
    __m512i segment[32];
    for (int segment_index = 0; segment_index < 32; ++segment_index)
        segment[segment_index] = _mm512_loadu_si512(sz_utf8_word_break_flat_lut_0800_ + segment_index * 64);
    // The 16 segment-pairs each resolve their own 128-byte `vpermi2b` window (irreducible: a 2KB table over
    // 128-byte permute windows requires exactly 16 permutes). Selecting `candidate[page]` was a 16-deep SERIAL
    // chain of `cmpeq(page,p)` + `mask_mov` — a 16-long dependency on the latency-critical path of the classifier.
    // Preferred: a 4-deep balanced binary mux keyed on the four bits of `page` (one `vptest` per bit). Same lanes
    // selected (page in [0,15] always), but the critical chain shrinks 16 -> 4, measured ~3x latency on this path.
    __m512i candidate[16];
    for (int pair = 0; pair < 16; ++pair)
        candidate[pair] = _mm512_permutex2var_epi8(segment[2 * pair], in_seven, segment[2 * pair + 1]);
    __mmask64 const page_bit0 = _mm512_test_epi8_mask(page, _mm512_set1_epi8(0x01));
    for (int pair = 0; pair < 8; ++pair)
        candidate[pair] = _mm512_mask_blend_epi8(page_bit0, candidate[2 * pair], candidate[2 * pair + 1]);
    __mmask64 const page_bit1 = _mm512_test_epi8_mask(page, _mm512_set1_epi8(0x02));
    for (int pair = 0; pair < 4; ++pair)
        candidate[pair] = _mm512_mask_blend_epi8(page_bit1, candidate[2 * pair], candidate[2 * pair + 1]);
    __mmask64 const page_bit2 = _mm512_test_epi8_mask(page, _mm512_set1_epi8(0x04));
    for (int pair = 0; pair < 2; ++pair)
        candidate[pair] = _mm512_mask_blend_epi8(page_bit2, candidate[2 * pair], candidate[2 * pair + 1]);
    __mmask64 const page_bit3 = _mm512_test_epi8_mask(page, _mm512_set1_epi8(0x08));
    return _mm512_mask_blend_epi8(page_bit3, candidate[0], candidate[1]);
}

/**
 *  @brief  Classify the four Supplementary-Plane bytes of every 4-byte lead lane to Word_Break properties, fully
 *          in-register via arithmetic ranges. Covers the classes that actually occur in SMP text:
 *          Regional_Indicator (U+1F1E6..U+1F1FF), Extend (skin-tone modifiers U+1F3FB..U+1F3FF, variation
 *          selectors U+E0100..U+E01EF, tag spaces U+E0020..U+E007F), Format (the tag begin U+E0001), and Other
 *          (every other SMP codepoint, including Extended_Pictographic whose Word_Break is Other). No SMP lane
 *          ever leaves the vectorized path.
 */
SZ_INTERNAL __m512i sz_utf8_word_break_classify_smp_(__m512i window, __m512i next1, __m512i next2, __m512i next3) {
    __m512i const plane = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(window, _mm512_set1_epi8(0x07)), 2),
                                          sz_utf8_codepoints_srl8_(next1, 4, 0x03));
    __m512i const mid = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(next1, _mm512_set1_epi8(0x0F)), 4),
                                        sz_utf8_codepoints_srl8_(next2, 2, 0x0F));
    __m512i const low = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(next2, _mm512_set1_epi8(0x03)), 6),
                                        _mm512_and_si512(next3, _mm512_set1_epi8(0x3F)));

    __m512i smp_class = _mm512_set1_epi8((char)sz_utf8_word_break_other_k);

    // Plane 0x01: Regional_Indicator U+1F1E6..U+1F1FF and skin-tone Extend U+1F3FB..U+1F3FF.
    __mmask64 const plane01 = _mm512_cmpeq_epi8_mask(plane, _mm512_set1_epi8(0x01));
    __mmask64 const ri = plane01 & _mm512_cmpeq_epi8_mask(mid, _mm512_set1_epi8((char)0xF1)) &
                         _mm512_cmp_epu8_mask(low, _mm512_set1_epi8((char)0xE6), _MM_CMPINT_NLT);
    __mmask64 const skin = plane01 & _mm512_cmpeq_epi8_mask(mid, _mm512_set1_epi8((char)0xF3)) &
                           _mm512_cmp_epu8_mask(low, _mm512_set1_epi8((char)0xFB), _MM_CMPINT_NLT);

    // Plane 0x0E: tag begin U+E0001 (Format); tag spaces U+E0020..U+E007F and variation selectors
    // U+E0100..U+E01EF (Extend). `mid` is the third-from-low 8 bits; for U+E00xx mid == 0x00, for U+E01xx mid
    // == 0x01. The plane byte already isolates U+0Exxxx, so the (mid, low) pair pins the sub-ranges.
    __mmask64 const plane0e = _mm512_cmpeq_epi8_mask(plane, _mm512_set1_epi8(0x0E));
    __mmask64 const tag_begin = plane0e & _mm512_cmpeq_epi8_mask(mid, _mm512_setzero_si512()) &
                                _mm512_cmpeq_epi8_mask(low, _mm512_set1_epi8(0x01));
    __mmask64 const tag_space = plane0e & _mm512_cmpeq_epi8_mask(mid, _mm512_setzero_si512()) &
                                _mm512_cmp_epu8_mask(low, _mm512_set1_epi8(0x20), _MM_CMPINT_NLT) &
                                _mm512_cmp_epu8_mask(low, _mm512_set1_epi8(0x7F), _MM_CMPINT_LE);
    __mmask64 const variation = plane0e & _mm512_cmpeq_epi8_mask(mid, _mm512_set1_epi8(0x01)) &
                                _mm512_cmp_epu8_mask(low, _mm512_set1_epi8((char)0xEF), _MM_CMPINT_LE);

    smp_class = _mm512_mask_mov_epi8(smp_class, ri, _mm512_set1_epi8((char)sz_utf8_word_break_regional_ind_k));
    smp_class = _mm512_mask_mov_epi8(smp_class, skin | tag_space | variation,
                                     _mm512_set1_epi8((char)sz_utf8_word_break_extend_k));
    smp_class = _mm512_mask_mov_epi8(smp_class, tag_begin, _mm512_set1_epi8((char)sz_utf8_word_break_format_k));
    return smp_class;
}

/**
 *  @brief  Classify a 64-byte window to per-lane Word_Break properties (one class per codepoint-start lane),
 *          entirely in-register and gather-free. ASCII lanes go through the ASCII permute. BMP codepoints are
 *          reconstructed into per-lane `high`/`low` halves and resolved by arithmetic big ranges (Latin / Hangul /
 *          CJK), a cp < 0x800 page LUT, and the shared two-stage trie for the scattered BMP residue. SMP leads are
 *          classified by @ref sz_utf8_word_break_classify_smp_. There is no scalar cold loop and no SMP deferral.
 */
SZ_INTERNAL __m512i sz_utf8_word_break_classify_window_( //
    __m512i window, __m512i high, __m512i low, __mmask64 is_four_byte, __m512i next1, __m512i next2, __m512i next3) {
    __mmask64 const is_ascii = ~_mm512_movepi8_mask(window);

    __mmask64 const high_eq_01 = _mm512_cmpeq_epi8_mask(high, _mm512_set1_epi8(0x01));
    __mmask64 const high_eq_02 = _mm512_cmpeq_epi8_mask(high, _mm512_set1_epi8(0x02));
    __mmask64 const low_le_4f = _mm512_cmp_epu8_mask(low, _mm512_set1_epi8(0x4F), _MM_CMPINT_LE);
    __mmask64 const latin = high_eq_01 | (high_eq_02 & low_le_4f);
    // Range collapse: each big range was written as two pieces with a redundant interior `high` boundary compare.
    // Hangul U+AC00..U+D7A3 is the single contiguous high-range [0xAC, 0xD7] minus the low-hole `high==0xD7 &&
    // low>0xA3`; CJK U+3400..U+9FFF (Ext-A + main) is [0x34, 0x9F] minus the low-hole `high==0x4D && low>0xBF`.
    // Preferred over the two-clause union: drops the interior 0xD6/0x4C/0x4E compares, ~10% fewer port-5 ops on
    // the range region, bit-identical over all 65,536 (high,low) pairs. `latin` is deliberately NOT collapsed:
    // folding it to `high<=0x02` would admit `high==0x00` lanes into `fast_aletter` and corrupt the `cold` mask.
    __mmask64 const hangul = (_mm512_cmp_epu8_mask(high, _mm512_set1_epi8((char)0xAC), _MM_CMPINT_NLT) &
                              _mm512_cmp_epu8_mask(high, _mm512_set1_epi8((char)0xD7), _MM_CMPINT_LE)) &
                             ~(_mm512_cmpeq_epi8_mask(high, _mm512_set1_epi8((char)0xD7)) &
                               _mm512_cmp_epu8_mask(low, _mm512_set1_epi8((char)0xA3), _MM_CMPINT_NLE));
    __mmask64 const cjk_combined = (_mm512_cmp_epu8_mask(high, _mm512_set1_epi8(0x34), _MM_CMPINT_NLT) &
                                    _mm512_cmp_epu8_mask(high, _mm512_set1_epi8((char)0x9F), _MM_CMPINT_LE)) &
                                   ~(_mm512_cmpeq_epi8_mask(high, _mm512_set1_epi8(0x4D)) &
                                     _mm512_cmp_epu8_mask(low, _mm512_set1_epi8((char)0xBF), _MM_CMPINT_NLE));
    __mmask64 const fast_aletter = latin | hangul;

    // Compose the cheap resolutions first so each lane lands on exactly one class: ASCII permute, the big
    // arithmetic ranges (Latin/Hangul -> ALetter, CJK/CJK-Ext-A -> Other), and the in-register SMP classifier.
    __m512i const smp_class = sz_utf8_word_break_classify_smp_(window, next1, next2, next3);
    __m512i classes = _mm512_setzero_si512();
    classes = _mm512_mask_blend_epi8(cjk_combined, classes, _mm512_set1_epi8((char)sz_utf8_word_break_other_k));
    classes = _mm512_mask_blend_epi8(fast_aletter, classes, _mm512_set1_epi8((char)sz_utf8_word_break_aletter_k));
    classes = _mm512_mask_mov_epi8(classes, is_four_byte, smp_class);
    classes = _mm512_mask_blend_epi8(is_ascii, classes, sz_utf8_word_break_classify_ascii_icelake_(window));

    // cp < 0x800 page LUT for the 2-byte residue (Latin-1 supplement, Greek/Cyrillic, ...), gated to lanes with
    // `high < 0x08` so a window without any 2-byte residue skips the whole `vpermi2b` page network.
    __mmask64 const is_small = _mm512_cmp_epu8_mask(high, _mm512_set1_epi8(0x08), _MM_CMPINT_LT) & ~is_ascii &
                               ~is_four_byte;
    if (is_small) classes = _mm512_mask_mov_epi8(classes, is_small, sz_utf8_word_break_small_page_(high, low));

    // Scattered 3-byte BMP residue (everything outside the big arithmetic ranges) via the shared two-stage trie,
    // replacing the old scalar cold loop and the family-local tier-2 page network: all such lanes resolved at once,
    // gather-free, no spill. Gated to the residue mask so pure-CJK/Latin/ASCII windows skip the trie network.
    __mmask64 const is_bmp_three = _mm512_cmp_epu8_mask(high, _mm512_set1_epi8(0x08), _MM_CMPINT_NLT);
    __mmask64 const cold = is_bmp_three & ~(cjk_combined | fast_aletter | is_four_byte);
    if (cold)
        classes = _mm512_mask_mov_epi8(
            classes, cold,
            sz_utf8_codepoints_trie_walk_icelake_(
                high, low, sz_utf8_word_break_trie_l1_, sz_utf8_word_break_trie_l2_, sz_utf8_word_break_trie_leaf_,
                sz_utf8_word_break_trie_block_k, sz_utf8_word_break_trie_subblock_k,
                (int)(sizeof(sz_utf8_word_break_trie_l1_) / sizeof(sz_utf8_word_break_trie_l1_[0])),
                (int)(sizeof(sz_utf8_word_break_trie_l2_) / sizeof(sz_utf8_word_break_trie_l2_[0])),
                (int)(sizeof(sz_utf8_word_break_trie_leaf_) / sizeof(sz_utf8_word_break_trie_leaf_[0]))));
    return classes;
}

#pragma endregion Gather Free Word_Break Classifier

#pragma region Word_Break Boundary Algebra

/**
 *  @brief  Cross-window left-context carry: the effective `previous_property` at the next window's anchor and the
 *          parity of the open Regional_Indicator run (1 => an odd number of RIs are already open, so the next RI
 *          joins; for WB15/WB16). Threaded forward so the vectorized path never re-walks the straddle or re-counts.
 */
typedef struct sz_utf8_word_break_carry_t {
    sz_u8_t left_property; /**< Effective `previous_property` at the next window's first emitted lane. */
    sz_u8_t ri_parity;     /**< Parity of the contiguous RI run open at that lane (0 or 1). */
} sz_utf8_word_break_carry_t;

/** @brief  One window classified into boundary bits plus the metadata the driver needs to advance. */
typedef struct sz_utf8_word_break_window_t {
    sz_u64_t boundary;                /**< Bit `i` set => a true UAX-29 word boundary begins at lane `i`. */
    sz_u64_t start_bytes;             /**< Bit `i` set => lane `i` is a codepoint-start byte. */
    sz_u64_t lead_ignorable;          /**< Bit `i` set => lane `i` is a WB4 ignorable (Extend/Format/ZWJ) lead. */
    sz_size_t resolved;               /**< Exclusive upper bound on codepoint-start lanes fully resolved. */
    sz_utf8_word_break_carry_t carry; /**< Cross-window register carry at lane `resolved`. */
} sz_utf8_word_break_window_t;

/**
 *  @brief  In-register equivalent of `sz_utf8_word_break_previous_property_` at the next window's anchor: the class
 *          of the last significant (non-ignorable) codepoint strictly below lane @p resolved, read from this
 *          window's own classified lanes via a `vpcompressb` of the class vector (no spill, no scalar walk).
 */
SZ_INTERNAL sz_u8_t sz_utf8_word_break_seam_left_from_classes_( //
    __m512i classes, sz_u64_t start_bytes, sz_u64_t lead_ignorable, sz_size_t resolved, sz_u8_t fallback) {
    sz_u64_t const significant = start_bytes & ~lead_ignorable & sz_u64_mask_until_(resolved);
    if (significant == 0) return fallback; // no significant lead resolved this window: keep the inbound carry
    sz_size_t const lane = (sz_size_t)(63 - sz_u64_clz(significant));
    __m512i const isolated = _mm512_maskz_compress_epi8(_cvtu64_mask64((sz_u64_t)1 << lane), classes);
    return (sz_u8_t)_mm_cvtsi128_si32(_mm512_castsi512_si128(isolated));
}

/**
 *  @brief  WB15/WB16 join mask: within every maximal Regional_Indicator run, suppress the boundary before each RI
 *          whose index (counting from the run start, plus the inbound run parity) is odd. The per-lane parity is a
 *          segmented exclusive prefix-XOR of @p ri over its own runs, seeded by @p inbound_parity at the lowest
 *          lane, computed in log-depth Kogge-Stone doubling (no per-lane loop). @p ri lanes are RI codepoint
 *          starts; ignorables/continuations between two RIs are part of the same run via @p run_gate.
 */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_ri_join_( //
    sz_u64_t ri, sz_u64_t run_gate, sz_u8_t inbound_parity, sz_u64_t *inclusive_parity_out) {
    // Inclusive segmented prefix-XOR of `ri` over each contiguous `run_gate` run.
    sz_u64_t bits = ri;
    sz_u64_t reach = run_gate;
    for (int shift = 1; shift < 64; shift <<= 1) {
        bits ^= (bits << shift) & reach;
        reach &= reach << shift;
    }
    // Inbound seed: a virtual RI strictly below lane 0 belonging to the same run as lane 0. In the inclusive parity
    // it contributes a CONSTANT toggle to every lane of that run, so it is XORed in AFTER the prefix-XOR — XORing the
    // flood into `ri` BEFORE the scan is wrong, since the prefix-XOR of a solid run-length block alternates per lane
    // instead of staying constant, corrupting the parity of every multi-lane run (the long-RI-run seam bug). The
    // lane-0 run is isolated by flooding lane 0 rightward across `run_gate`.
    if (inbound_parity) bits ^= sz_utf8_codepoints_fill_right_(run_gate & 1ull, run_gate);
    // `bits[lane]` is now the parity of RIs (plus the inbound seed) at and below `lane` within its run. The
    // exclusive parity strictly below an RI lane is `bits ^ ri`; an odd exclusive count means we sit mid-pair, so
    // the boundary before this RI is suppressed (WB15/WB16).
    *inclusive_parity_out = bits;
    return ri & (bits ^ ri);
}

/**
 *  @brief  Classify the 64-byte window anchored at @p base into UAX-29 boundary bits, fully vectorized.
 *
 *  Two lanes of left context (or fewer at the buffer head) let WB6/WB7/WB12 read their left neighbour; that
 *  neighbour's class and the open RI parity arrive in @p carry. All of WB3-WB16 are branchless bit algebra over
 *  the per-lane class masks: RI parity (WB15/WB16) is resolved in-register by @ref sz_utf8_word_break_ri_join_,
 *  and there is no serial-oracle deferral. Lanes lacking full in-register right context (the top edge) are not
 *  emitted; `resolved` reports the exclusive upper bound the window fully resolved.
 */
SZ_INTERNAL sz_utf8_word_break_window_t sz_utf8_word_break_window_boundaries_( //
    sz_u8_t const *text, sz_size_t length, sz_size_t base, sz_size_t anchor_lane, __m512i lane_identity,
    sz_utf8_word_break_carry_t carry) {
    sz_u8_t const left_property = carry.left_property;
    sz_size_t valid_lanes = length - base;
    if (valid_lanes > 64) valid_lanes = 64;
    sz_size_t const high_lane = valid_lanes - 1;
    int const at_tail = valid_lanes < 64; // the window reaches the end of the buffer

    sz_utf8_codepoints_window_t const decoded = sz_utf8_codepoints_decode_window_(text + base, valid_lanes,
                                                                                  lane_identity);
    __m512i const window = decoded.window;
    sz_u64_t const continuation = _cvtmask64_u64(decoded.continuation);
    sz_u64_t const start_bytes_all = _cvtmask64_u64(decoded.codepoint_starts);

    sz_utf8_word_break_window_t result;

    if (_mm512_movepi8_mask(window) == 0) { // All-ASCII window: one codepoint per lane.
        __m512i const ascii_classes = sz_utf8_word_break_classify_ascii_icelake_(window);
        // Lanes [0, anchor_lane) are pure left context; the top lane lacks a right neighbour unless it is the
        // buffer end. Trust [anchor_lane, high_lane] (inclusive of high_lane only at the tail).
        sz_size_t const trusted_hi = at_tail ? high_lane : (high_lane > 0 ? high_lane - 1 : 0);
        sz_u64_t const trusted = sz_u64_mask_until_(trusted_hi + 1) & ~sz_u64_mask_until_(anchor_lane);
        // An all-ASCII window has one codepoint per lane, so the genuine left lanes [0, anchor_lane) already carry
        // the previous codepoints' classes; the join mask reads them directly and `left_property` is only needed
        // for the seam at the very buffer head (anchor_lane == 0), handled by the BMP path's `seam`.
        sz_u64_t boundary = (~sz_utf8_word_break_join_mask_ascii_(ascii_classes)) & start_bytes_all & trusted;
        // WB3d: WSegSpace x WSegSpace. The only ASCII WSegSpace is U+0020, one codepoint per lane here, so suppress
        // the break before a space that directly follows a space.
        sz_u64_t const ascii_space = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8(0x20)));
        boundary &= ~(ascii_space & (ascii_space << 1));
        // Double_Quote (U+0022) bridges only Hebrew x " x Hebrew (WB7b/WB7c); ASCII has no Hebrew, so the shared
        // join treats it as a MidNumLetQ bridge it must not be. Force the break before every " and the codepoint
        // after it.
        sz_u64_t const ascii_double_quote = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8(0x22)));
        boundary |= (ascii_double_quote | (ascii_double_quote << 1)) & start_bytes_all & trusted;
        result.boundary = boundary;
        result.start_bytes = start_bytes_all;
        result.lead_ignorable = 0; // pure-ASCII window: Extend/Format/ZWJ are all non-ASCII
        result.resolved = trusted_hi + 1;
        result.carry.left_property = sz_utf8_word_break_seam_left_from_classes_(ascii_classes, start_bytes_all, 0,
                                                                                result.resolved, left_property);
        result.carry.ri_parity = 0; // ASCII never carries Regional_Indicator
        return result;
    }

    __m512i const high = decoded.high;
    __m512i const low = decoded.low;
    __m512i const next1 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(1)), window);
    __m512i const next2 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(2)), window);
    __m512i const next3 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(3)), window);
    __m512i const classes = sz_utf8_word_break_classify_window_(window, high, low, decoded.four_byte_starts, next1,
                                                                next2, next3);

    sz_u64_t const start_bytes = start_bytes_all;
    sz_u64_t const anchor_floor = ~sz_u64_mask_until_(anchor_lane);

    sz_u64_t const class_aletter = (sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_aletter_k) |
                                    sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_hebrew_letter_k)) &
                                   start_bytes;
    sz_u64_t const class_hebrew = sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_hebrew_letter_k) &
                                  start_bytes;
    sz_u64_t const class_numeric = sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_numeric_k) & start_bytes;
    sz_u64_t const class_katakana = sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_katakana_k) &
                                    start_bytes;
    sz_u64_t const class_extendnumlet = sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_extendnumlet_k) &
                                        start_bytes;
    sz_u64_t const lead_ignorable = (sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_extend_k) |
                                     sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_zwj_k) |
                                     sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_format_k)) &
                                    start_bytes;

    // Forward-reach trust: every emitted boundary needs its full WB6/WB7/WB11/WB12 two-codepoint-ahead context
    // in-window. The unbounded `fill` smears reach the next significant codepoint across ANY ignorable-run length,
    // so the trust horizon is data-dependent: reserve the top `reserve_codepoints` SIGNIFICANT (non-ignorable)
    // codepoints, whose forward context spills past the window edge; the carry re-resolves them next window. For
    // dense text this reserves far fewer lanes than a fixed bound, and it never under-reserves on long ignorables.
    sz_size_t right_high = high_lane;
    if (!at_tail) {
        sz_u64_t const significant = _andn_u64(lead_ignorable, start_bytes) & sz_u64_mask_until_(valid_lanes);
        sz_size_t const significant_count = (sz_size_t)sz_u64_popcount(significant);
        // `right_high` is the highest significant lane that is NOT among the reserved top `reserve_codepoints_k`,
        // i.e. the `(significant_count - reserve_codepoints_k)`-th significant lane counted from the bottom. A
        // single `vpdep` isolates that set bit (no loop, no per-iteration `clz`); below the reserve count the
        // window resolves only the anchor.
        if (significant_count > sz_utf8_word_break_reserve_codepoints_k) {
            sz_size_t const select_index = significant_count - sz_utf8_word_break_reserve_codepoints_k - 1;
            right_high = (sz_size_t)sz_u64_ctz(_pdep_u64((sz_u64_t)1 << select_index, significant));
            if (right_high < anchor_lane) right_high = anchor_lane;
        }
        else right_high = anchor_lane;
        // Edge truncation: the decoder loads exactly `valid_lanes` bytes, so a multi-byte lead near the window top
        // can have its continuation bytes spill past the loaded window and be MISCLASSIFIED (its `next1..3` lanes
        // read zero). Such a lane is a `start_bytes` bit but not a resolvable codepoint. Exclude every trusted lane
        // whose codepoint length runs past `valid_lanes`: cap `right_high` below the lowest spilling start. The
        // driver then re-bases the next window so the truncated codepoint is fully in-register. (At the buffer tail
        // there is nothing past the window, so this never fires there.) This even excludes a spilling ANCHOR; the
        // driver finalizes the pending word in that case, which is exact when the spill is a trailing ignorable.
        sz_size_t const top_start = (sz_size_t)(63 - sz_u64_clz(start_bytes | 1ull));
        sz_size_t const top_length = sz_utf8_codepoint_length_(text[base + top_start]);
        if (top_start + top_length > valid_lanes && right_high >= top_start)
            right_high = top_start > 0 ? top_start - 1 : 0;
    }
    sz_u64_t const mid_letter_or_quotes = (sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_midletter_k) |
                                           sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_mid_quotes_k)) &
                                          start_bytes;
    sz_u64_t const mid_num_or_quotes = (sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_midnum_k) |
                                        sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_mid_quotes_k)) &
                                       start_bytes;
    sz_u64_t const mid_quotes = sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_mid_quotes_k) & start_bytes;
    sz_u64_t const mid_any = (sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_midletter_k) |
                              sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_midnum_k) |
                              sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_mid_quotes_k)) &
                             start_bytes;
    sz_u64_t const class_cr = sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_cr_k) & start_bytes;
    sz_u64_t const class_lf = sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_lf_k) & start_bytes;
    sz_u64_t const class_newline =
        (class_cr | class_lf | sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_newline_k)) & start_bytes;
    sz_u64_t const class_zwj = sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_zwj_k) & start_bytes;
    sz_u64_t const class_regional = sz_utf8_word_break_class_mask_(classes, sz_utf8_word_break_regional_ind_k) &
                                    start_bytes;

    // Seam injection lane: the anchor's left neighbour is the previous codepoint, whose effective class is the
    // register carry `left_property`. The two lanes below the anchor are that neighbour's trailing bytes (possibly
    // continuation bytes of a multi-byte codepoint), so they do NOT carry its class; instead we inject
    // `left_property` at lane `anchor_lane - 1` and let the `<< 1` neighbour shift read it at the anchor. At the
    // buffer head (anchor_lane == 0) there is no lane below; `left_property` is Other (WB1) and the anchor breaks.
    sz_u64_t const seam = anchor_lane > 0 ? (1ull << (anchor_lane - 1)) : 0ull;

    // WSegSpace lanes (WB3d) are outside the 4-bit Word_Break model; detect them by branchless (high,low) range
    // membership (6 ranges). Needed before the fast path so a run of non-ASCII spaces (e.g. U+3000) is not mistaken
    // for break-everywhere "Other" leads.
    // The decoder fills `high`/`low` for >=2-byte leads only; an ASCII lane's `low` is derived from the next byte
    // (garbage), so detect the sole ASCII WSegSpace (U+0020) from the raw window byte and gate the multibyte range
    // check to non-ASCII lanes.
    sz_u64_t const non_ascii_lanes = _cvtmask64_u64(_mm512_movepi8_mask(window));
    sz_u64_t const wseg = ((sz_utf8_word_break_range16_mask_(high, low, sz_utf8_word_break_wseg_lo_,
                                                             sz_utf8_word_break_wseg_hi_,
                                                             sz_utf8_word_break_wseg_count_k) &
                            non_ascii_lanes) |
                           _cvtmask64_u64(_mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8(0x20)))) &
                          start_bytes;

    // Break-everywhere fast path: when a window holds only "Other" leads (no class that any WB3-WB16 no-break rule
    // can join, no ignorables, no newline/CR/ZWJ/RI, no WSegSpace, and a non-joining left neighbour), every
    // codepoint start is a boundary. This is the overwhelmingly common CJK / ideographic case and skips the entire
    // smear/bridge algebra. A single mask test keeps it branchless per lane and fully vectorized.
    sz_u64_t const joinable_any = class_aletter | class_numeric | class_katakana | class_extendnumlet | mid_any |
                                  lead_ignorable | class_newline | class_zwj | class_regional | wseg;
    int const left_joins = left_property != sz_utf8_word_break_other_k;
    if (joinable_any == 0 && !left_joins) {
        sz_size_t const resolved_fast = right_high + 1;
        result.boundary = start_bytes & sz_u64_mask_until_(resolved_fast) & anchor_floor;
        result.start_bytes = start_bytes;
        result.lead_ignorable = 0; // fast path is gated on `lead_ignorable == 0` (joinable_any)
        result.resolved = resolved_fast;
        result.carry.left_property = sz_utf8_word_break_seam_left_from_classes_(classes, start_bytes, 0, resolved_fast,
                                                                                left_property);
        result.carry.ri_parity = 0;
        return result;
    }

    // WB4: Extend/Format/ZWJ are transparent. A "flow" mask lets a class run reach across ignorables (and the
    // continuation bytes of a multi-byte codepoint) so neighbour shifts read the effective adjacent class.
    sz_u64_t const flow_ignorable = continuation | lead_ignorable;
    sz_u64_t aletter_right_base = sz_utf8_codepoints_fill_right_(class_aletter, flow_ignorable);
    sz_u64_t numeric_right_base = sz_utf8_codepoints_fill_right_(class_numeric, flow_ignorable);
    sz_u64_t const aletter_left_base = sz_utf8_codepoints_fill_left_(class_aletter, flow_ignorable);
    sz_u64_t const numeric_left_base = sz_utf8_codepoints_fill_left_(class_numeric, flow_ignorable);
    sz_u64_t hebrew_right_base = sz_utf8_codepoints_fill_right_(class_hebrew, flow_ignorable);
    sz_u64_t const hebrew_left_base = sz_utf8_codepoints_fill_left_(class_hebrew, flow_ignorable);
    int const left_is_aletter = left_property == sz_utf8_word_break_aletter_k;
    int const left_is_hebrew = left_property == sz_utf8_word_break_hebrew_letter_k;
    int const left_is_numeric = left_property == sz_utf8_word_break_numeric_k;
    aletter_right_base = sz_u64_or_if_(aletter_right_base, seam, left_is_aletter || left_is_hebrew);
    numeric_right_base = sz_u64_or_if_(numeric_right_base, seam, left_is_numeric);
    hebrew_right_base = sz_u64_or_if_(hebrew_right_base, seam, left_is_hebrew);

    // WB6/WB7 and WB11/WB12: a Mid* lane between two letters (or numerics) bridges the run; WB13a/13b binding
    // (ExtendNumLet) must not let a Mid bridge form. The bridge "Mid" for letters/numerics is MidNumLetQ
    // (MidNumLet + Single_Quote), which excludes Double_Quote; Double_Quote bridges ONLY Hebrew x " x Hebrew
    // (WB7b/WB7c), so isolate the U+0022 byte and route it through a Hebrew-only bridge.
    sz_u64_t const double_quote = mid_quotes & _cvtmask64_u64(_mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8(0x22)));
    sz_u64_t const mid_letter_quotes_no_double = mid_letter_or_quotes & ~double_quote;
    sz_u64_t const mid_num_quotes_no_double = mid_num_or_quotes & ~double_quote;
    sz_u64_t bridge = (mid_letter_quotes_no_double & (aletter_right_base << 1) & (aletter_left_base >> 1)) |
                      (mid_num_quotes_no_double & (numeric_right_base << 1) & (numeric_left_base >> 1)) |
                      (double_quote & (hebrew_right_base << 1) & (hebrew_left_base >> 1));

    sz_u64_t const flow = flow_ignorable | bridge;
    sz_u64_t const significant_aletter = class_aletter;
    sz_u64_t const significant_numeric = class_numeric;
    sz_u64_t const significant_katakana = class_katakana;
    sz_u64_t const significant_extendnumlet = class_extendnumlet;

    // bridge==0 reuse: when no Mid* bridges a letter/numeric run, `flow == flow_ignorable` exactly, so the four
    // ALetter/Numeric left/right smears over `flow` are bit-identical to the `_base` smears already computed over
    // `flow_ignorable` (and the ALetter/Numeric right `_base` already carry the same seam injection). This is the
    // overwhelmingly common case (text with no MidLetter/MidNum between letters); preferred over re-smearing
    // because it elides 4 of the 12 bounded smears (48 shift-or steps) with NO change to the 12-lane reach bound,
    // so it stays correct even on >12-lane ignorable runs (where a Kogge-Stone `fill` substitute would diverge).
    // Katakana / ExtendNumLet / Hebrew have no `_base`, so they are smeared fresh regardless.
    sz_u64_t aletter_right, numeric_right, aletter_left, numeric_left;
    if (bridge == 0) {
        aletter_right = aletter_right_base; // seam already folded into the `_base` above
        numeric_right = numeric_right_base; // seam already folded into the `_base` above
        aletter_left = aletter_left_base;
        numeric_left = numeric_left_base;
    }
    else {
        aletter_right = sz_utf8_codepoints_fill_right_(significant_aletter, flow);
        numeric_right = sz_utf8_codepoints_fill_right_(significant_numeric, flow);
        aletter_left = sz_utf8_codepoints_fill_left_(significant_aletter, flow);
        numeric_left = sz_utf8_codepoints_fill_left_(significant_numeric, flow);
        aletter_right = sz_u64_or_if_(aletter_right, seam, left_is_aletter || left_is_hebrew);
        numeric_right = sz_u64_or_if_(numeric_right, seam, left_is_numeric);
    }
    sz_u64_t katakana_right = sz_utf8_codepoints_fill_right_(significant_katakana, flow);
    sz_u64_t extendnumlet_right = sz_utf8_codepoints_fill_right_(significant_extendnumlet, flow);
    sz_u64_t const katakana_left = sz_utf8_codepoints_fill_left_(significant_katakana, flow);
    sz_u64_t const extendnumlet_left = sz_utf8_codepoints_fill_left_(significant_extendnumlet, flow);
    sz_u64_t hebrew_right = sz_utf8_codepoints_fill_right_(class_hebrew, flow);
    katakana_right = sz_u64_or_if_(katakana_right, seam, left_property == sz_utf8_word_break_katakana_k);
    extendnumlet_right = sz_u64_or_if_(extendnumlet_right, seam, left_property == sz_utf8_word_break_extendnumlet_k);
    hebrew_right = sz_u64_or_if_(hebrew_right, seam, left_is_hebrew);

    // WB3/WB3a/WB3b/WB3c: Newline (CR/LF/Newline) forces a break on both sides except CR x LF, and ZWJ x
    // suppresses the break after a ZWJ. The left neighbour of the anchor is taken from `left_property`.
    sz_u64_t const seam_anchor = seam << 1; // the anchor lane itself, for the already-shifted `previous_*` signals
    sz_u64_t previous_newline = (class_newline << 1);
    sz_u64_t previous_cr = (class_cr << 1);
    // WB3c "ZWJ x" suppresses the break before the codepoint after a ZWJ. Per the serial oracle's WB4 lookbehind,
    // ZWJ (itself ignorable) is the effective `previous_property` only when no significant codepoint precedes it
    // (a leading ZWJ at the buffer head). That is delivered through the `left_property == ZWJ` seam below; an
    // interior ZWJ is skipped over as an ignorable, so the in-window `previous_zwj` stays a plain neighbour shift.
    // ZWJ is 3 bytes, so reach across its OWN continuation bytes (only `continuation`, not the wider `flow`) before
    // the `<< 1` neighbour shift. WB3c requires the ZWJ immediately before the pictograph: an intervening Extend
    // (which `flow` would smear across) breaks the adjacency, so the pictograph after it is NOT joined.
    sz_u64_t previous_zwj = (sz_utf8_codepoints_smear_right_(class_zwj, continuation, sz_utf8_word_break_smear_steps_k)
                             << 1);
    int const left_is_cr = left_property == sz_utf8_word_break_cr_k;
    int const left_is_newline = left_is_cr || left_property == sz_utf8_word_break_lf_k ||
                                left_property == sz_utf8_word_break_newline_k;
    previous_cr = sz_u64_or_if_(previous_cr, seam_anchor, left_is_cr);
    previous_newline = sz_u64_or_if_(previous_newline, seam_anchor, left_is_newline);
    // No ZWJ seam injection: WB3c needs the previous element to END in ZWJ, but `left_property` carries its BASE,
    // which is ZWJ even when trailing Extends follow it. The anchor's immediate predecessor is always in-window, so
    // the in-window `previous_zwj` (smeared over the real continuation bytes) already resolves WB3c exactly.

    sz_u64_t const previous_aletter = aletter_right << 1;
    sz_u64_t const previous_numeric = numeric_right << 1;
    sz_u64_t const previous_katakana = katakana_right << 1;
    sz_u64_t const previous_extendnumlet = extendnumlet_right << 1;
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
    // WB3c: ZWJ x Extended_Pictographic. The pictograph set is not in the 4-bit Word_Break model, so it is detected
    // by branchless (high,low) range membership, gated behind an in-window ZWJ — pure text has no ZWJ and pays
    // nothing. SMP pictographs live in plane 1; their window `high`/`low` are the codepoint's low 16 bits.
    if (class_zwj) {
        sz_u64_t const four_byte = _cvtmask64_u64(decoded.four_byte_starts);
        __m512i const plane = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(window, _mm512_set1_epi8(0x07)), 2),
                                              sz_utf8_codepoints_srl8_(next1, 4, 0x03));
        sz_u64_t const plane_one = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(plane, _mm512_set1_epi8(0x01)));
        sz_u64_t const pictographic_bmp = sz_utf8_word_break_range16_mask_( //
            high, low, sz_utf8_word_break_pict_bmp_lo_, sz_utf8_word_break_pict_bmp_hi_,
            sz_utf8_word_break_pict_bmp_count_k);
        // For 4-byte leads the decoder's `high`/`low` are not the codepoint's low 16 bits; reconstruct them the
        // same way the SMP classifier does (`mid` = cp[15:8], `low` = cp[7:0]) and match the plane-1 ranges.
        __m512i const smp_mid = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(next1, _mm512_set1_epi8(0x0F)), 4),
                                                sz_utf8_codepoints_srl8_(next2, 2, 0x0F));
        __m512i const smp_low = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(next2, _mm512_set1_epi8(0x03)), 6),
                                                _mm512_and_si512(next3, _mm512_set1_epi8(0x3F)));
        sz_u64_t const pictographic_smp = sz_utf8_word_break_range16_mask_( //
            smp_mid, smp_low, sz_utf8_word_break_pict_smp_lo_, sz_utf8_word_break_pict_smp_hi_,
            sz_utf8_word_break_pict_smp_count_k);
        // BMP pictographs are all >= U+00A9 (non-ASCII); gate to non-ASCII lanes so an ASCII lane's garbage `low`
        // cannot spuriously land in a pictographic range.
        sz_u64_t const pictographic = ((pictographic_bmp & non_ascii_lanes & ~four_byte) |
                                       (pictographic_smp & four_byte & plane_one)) &
                                      start_bytes;
        join |= previous_zwj & pictographic; // WB3c
    }

    // WB3d: WSegSpace x WSegSpace (runs of horizontal spaces stay one segment). Like WB3c, the spaces must be
    // directly adjacent — reach only across a space's OWN continuation bytes (`continuation`), not the wider
    // `flow`, so an intervening Extend (which attaches to the first space via WB4) still forces a break.
    join |= (sz_utf8_codepoints_smear_right_(wseg, continuation, sz_utf8_word_break_smear_steps_k) << 1) & wseg;

    // WB15/WB16: Regional_Indicator pairs. Resolved in-register through the segmented parity scan; the run gate
    // spans the RI starts and the ignorables/continuations between two RIs. The first RI of a window continues the
    // inbound run via `seam_anchor`, so the parity seed threads the run state across the window seam.
    sz_u64_t ri_inclusive = 0;
    // The run gate must connect lane 0 to the first in-window RI through any LEADING ignorables, or the inbound
    // parity seed (a virtual RI below lane 0) is stranded in a separate lane-0 run and never reaches the RIs — the
    // WB15/WB16 seam bug when `base` lands on the absorbed ignorables between an off-window RI and the first
    // in-window RI. `fill_right` only floods RIs upward, so add `fill_left` to flood the first RI down across the
    // leading ignorable run to lane 0 (it stops at any non-flow lane, so distinct RI runs stay separate).
    sz_u64_t const ri_run_gate = sz_u64_or_if_(sz_utf8_codepoints_fill_right_(class_regional, flow_ignorable) |
                                                   sz_utf8_codepoints_fill_left_(class_regional, flow_ignorable) |
                                                   class_regional,
                                               seam, carry.left_property == sz_utf8_word_break_regional_ind_k);
    sz_u64_t const ri_join = sz_utf8_word_break_ri_join_(class_regional, ri_run_gate, carry.ri_parity, &ri_inclusive);
    join |= ri_join;

    // WB3a/WB3b force a break before and after every Newline/CR/LF lane, overriding any join (CR x LF kept).
    sz_u64_t const force_break = ((previous_newline | class_newline) & ~(previous_cr & class_lf)) & start_bytes;

    sz_u64_t const start_bytes_no_bridge = start_bytes & ~bridge;
    sz_u64_t boundary = (~join) & start_bytes_no_bridge;
    // WB7a is Hebrew_Letter x Single_Quote (U+0027) only; the `mid_quotes` class also covers Double_Quote and
    // MidNumLet (e.g. '.'), so suppressing on the whole class wrongly merged "Hebrew ." / "Hebrew \"". Isolate the
    // single-quote byte. (WB7b/WB7c Hebrew x Double_Quote x Hebrew is handled by the letter bridge, not here.)
    sz_u64_t const single_quote = mid_quotes & _cvtmask64_u64(_mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8(0x27)));
    boundary &= ~(single_quote & (hebrew_right << 1)); // WB7a: Hebrew_Letter x Single_Quote
    // WB4: never break before an Extend/Format/ZWJ lane — it attaches to the preceding codepoint. The only breaks
    // before an ignorable are the newline-driven ones (CR/LF/Newline x; the first ignorable after a Newline is
    // "de-ignored"), which `force_break` re-adds immediately below, and the buffer-head sot break (driver-owned).
    boundary &= ~lead_ignorable;
    boundary |= force_break & start_bytes_no_bridge; // WB3a/WB3b break around Newline/CR/LF

    // Trust only the band [anchor_lane, right_high]: lanes below the anchor are pure left context and lanes above
    // `right_high` lack full forward reach (handled by the next window).
    sz_size_t const resolved = right_high + 1;
    sz_u64_t const trusted_run = sz_u64_mask_until_(resolved) & anchor_floor;
    boundary &= trusted_run;

    result.boundary = boundary;
    result.start_bytes = start_bytes;
    result.lead_ignorable = lead_ignorable;
    result.resolved = resolved;
    result.carry.left_property = sz_utf8_word_break_seam_left_from_classes_(classes, start_bytes, lead_ignorable,
                                                                            resolved, left_property);
    // Outbound RI parity: the inclusive RI parity (inbound seed included) at the last significant lane below
    // `resolved`, but only if that lane's RI run is still open at the seam (i.e. the run continues past `resolved`
    // or ends exactly at the last RI). Read directly from the segmented `ri_inclusive` vector.
    // The old `left_property==RI || class_regional` outer guard is redundant: the `(class_regional >> seam_lane) &
    // 1` test below already yields parity 0 whenever the top significant lane is not an RI start (covering both the
    // no-RI-in-window and the RI-only-via-left_property cases). Preferred branchless form keeps only the empty-set
    // guard (`clz(0)` is UB); `seam_significant | 1ull` pins lane 0 when the set is empty, then the final ternary
    // discards it. Drops the data-dependent branch and the `left_property` dependency; bit-exact, ~24% fewer instr.
    sz_u64_t const seam_significant = start_bytes & ~lead_ignorable & sz_u64_mask_until_(resolved);
    sz_size_t const seam_lane = (sz_size_t)(63 - sz_u64_clz(seam_significant | 1ull));
    sz_u8_t const top_is_regional = (sz_u8_t)((class_regional >> seam_lane) & 1ull);
    sz_u8_t const seam_parity = (sz_u8_t)((ri_inclusive >> seam_lane) & 1ull);
    result.carry.ri_parity = seam_significant ? (top_is_regional ? seam_parity : 0) : 0;
    return result;
}

#pragma endregion Word_Break Boundary Algebra

#pragma region Word_Break Forward Driver

// Scans `[scan_start, length)` forward, threading carry from `scan_start`. Words `[skip_words, ...)` (0-based in
// the scan order) are stored into `starts[]/lengths[]`; the first `skip_words` words are walked over (carry threaded,
// not stored) so a caller can retain a trailing slice without re-seeding from a possibly-mid-run boundary. Returns
// the number of STORED words (<= capacity), stopping early once `capacity` stored words are reached.
//
// Correctness note: for `scan_start == 0` the init reduces to the forward driver's (`base == 0`, `ri_parity == 0`,
// `left_property == previous_property`), so a byte-0 scan reproduces the forward boundary set EXACTLY. Starting at a
// non-zero `scan_start` re-seeds the first window's carry from the serial primitives, which is exact for ordinary
// text but NOT for a `scan_start` sitting mid-Regional_Indicator-run (the window's RI seam-gate cannot always rebuild
// long-run parity from a re-seed). The reverse driver therefore always calls this with `scan_start == 0` and uses
// `skip_words` to discard the leading words — never re-seeding mid-run.
SZ_INTERNAL sz_size_t sz_utf8_word_scan_forward_(                                                                //
    sz_u8_t const *text_u8, sz_size_t length, sz_size_t scan_start, sz_size_t skip_words, __m512i lane_identity, //
    sz_size_t *starts, sz_size_t *lengths, sz_size_t capacity, sz_size_t *total_words_out) {

    sz_size_t stored = 0; // words written to the output (index in scan order >= skip_words)
    sz_size_t seen = 0;   // words finalized so far (their start drained); used to honor `skip_words`
    sz_size_t word_start = scan_start;
    sz_size_t position = scan_start + sz_utf8_codepoint_length_(text_u8[scan_start]);
    // The whole suffix is a single word when no further codepoint exists past `scan_start`.
    if (position >= length) {
        if (total_words_out) *total_words_out = 1;
        if (skip_words == 0 && capacity > 0) {
            starts[0] = scan_start, lengths[0] = length - scan_start;
            return 1;
        }
        return 0;
    }

    // First-window carry: derive the exact left context entering `position` locally. The first window must seed its
    // RI parity at a CODEPOINT START — a mid-Regional_Indicator base would mis-decode the run and break WB15/16 —
    // yet `anchor_lane` must stay `<= left_context_k` (the window's RI accumulator and seam carry assume the seed
    // covers exactly the lanes below the anchor). So snap `base` UP to the first codepoint start at-or-after
    // `position - left_context_k`: keeps `anchor_lane <= left_context_k`, lands `base` on a codepoint start so the
    // seed counts whole codepoints, and lets the in-window scan count the remaining RIs.
    while (position < length) {
        // Snap `base` to a codepoint start and re-derive the EXACT carry EVERY window — not just the first. A
        // mid-codepoint base drops a Regional_Indicator straddling the base from the WB15/16 parity (it is counted
        // neither "before" base, since it straddles, nor in-window, where only its tail lanes appear), and a
        // threaded parity double-counts the RIs in the window's left-context overlap; either corrupts RI-run parity
        // across a window seam. Re-deriving at a codepoint-aligned base is exact and cheap (the serial walks stop at
        // the current ignorable/RI run's start). Snap UP so `anchor_lane <= left_context_k`, which the window's RI
        // accumulator and seam carry assume.
        sz_size_t const aim = position >= sz_utf8_word_break_left_context_k
                                  ? position - sz_utf8_word_break_left_context_k
                                  : 0;
        // Snap DOWN to the enclosing codepoint start: this keeps a codepoint straddling `aim` FULLY in-window so
        // the in-window RI scan counts it directly (snapping UP would push it below `base`, where the WB15/16 seed
        // must reconnect across leading ignorables — fragile). `anchor_lane` grows by at most 3 bytes past
        // `left_context_k`, still ample left context for WB6/WB7.
        sz_size_t base = sz_utf8_codepoints_start_at_or_before_(text_u8, aim);
        // Snap `base` to the START of the Regional_Indicator run OPEN AT THE ANCHOR so the WHOLE run is in-window and
        // the WB15/16 parity seed is a clean 0 — the seed-via-gate reconnection across leading ignorables is
        // unreliable. The run open at `position` is the maximal trailing sequence of RIs and the WB4 ignorables
        // between them; find its start by walking back over RIs and ignorables, stopping at the first non-ignorable
        // non-RI codepoint (or buffer head). Only snap when that start sits below the budget-derived `base` AND keeps
        // the whole run within the 64-byte window: a run already closed below the anchor needs no seed threading
        // (snapping would only inflate `anchor_lane`), and a run longer than the window cannot be fully captured (the
        // seed then threads the residual parity through the in-window gate). The serial walks stop at codepoint
        // starts, so this is exact and cheap.
        if (sz_utf8_word_break_count_regional_indicators_before_((sz_cptr_t)text_u8, position) > 0) {
            // The anchor's own codepoint must stay fully inside the 64-byte window or the edge truncation guard drops
            // it and forward progress stalls. So the snap floor is `position + anchor_length - 64`: a run that ends
            // within the window snaps to its clean (seed 0) start; a run longer than the window snaps only to this
            // floor, leaving the residual RI parity below `base` to thread through the in-window gate via the seed.
            sz_size_t const anchor_length = sz_utf8_codepoint_length_(text_u8[position]);
            sz_size_t const snap_floor = position + anchor_length > 64 ? position + anchor_length - 64 : 0;
            sz_size_t run_start = position;
            while (run_start > snap_floor) {
                sz_size_t const previous = sz_utf8_codepoints_start_at_or_before_(text_u8, run_start - 1);
                if (previous < snap_floor) break;
                sz_size_t previous_decode = previous;
                sz_u8_t const previous_property = sz_rune_word_break_property(
                    sz_utf8_decode_((sz_cptr_t)text_u8, run_start, &previous_decode));
                int const in_run = previous_property == sz_utf8_word_break_regional_ind_k ||
                                   sz_utf8_word_break_is_ignorable_(previous_property);
                if (!in_run) break;
                run_start = previous;
            }
            if (run_start < base) base = run_start;
        }
        // WB6/WB7/WB11/WB12 bridge lookbehind: the boundary at the anchor can be suppressed by a `letter Mid letter`
        // (or `numeric Mid numeric`) bridge whose LEFT letter sits two significant codepoints below the anchor —
        // across a Mid* neighbour and an arbitrarily long WB4 ignorable run. The `left_property` carry only delivers
        // ONE codepoint back (the Mid), so the bridge cannot form in-window unless that left letter is also in-window.
        // Pull `base` back until at least two significant (non-ignorable) codepoints precede the anchor (or the buffer
        // head is reached), bounded by the 64-byte window. A fixed byte budget under-reserves on long ignorable runs;
        // this data-dependent walk reserves exactly the bridge's reach. The serial walks stop at codepoint starts.
        sz_size_t significant_below = 0;
        {
            // Count the significant codepoints already in `[base, position)`, then keep pulling `base` to the
            // previous codepoint start (folding in one more codepoint each step) until two significant ones lie below
            // the anchor or the window edge / buffer head stops the walk.
            sz_size_t scan = base;
            while (scan < position) {
                sz_size_t next_scan = scan;
                sz_rune_t const scan_rune = sz_utf8_decode_((sz_cptr_t)text_u8, position, &next_scan);
                if (!sz_utf8_word_break_is_ignorable_(sz_rune_word_break_property(scan_rune))) ++significant_below;
                scan = next_scan;
            }
        }
        while (base > 0 && significant_below < sz_utf8_word_break_reserve_codepoints_k) {
            sz_size_t const previous = sz_utf8_codepoints_start_at_or_before_(text_u8, base - 1);
            if (position - previous >= 64) break;
            sz_size_t previous_decode = previous;
            sz_rune_t const previous_rune = sz_utf8_decode_((sz_cptr_t)text_u8, base, &previous_decode);
            if (!sz_utf8_word_break_is_ignorable_(sz_rune_word_break_property(previous_rune))) ++significant_below;
            base = previous;
        }
        sz_size_t const anchor_lane = position - base;
        sz_utf8_word_break_carry_t carry;
        carry.left_property = sz_utf8_word_break_previous_property_((sz_cptr_t)text_u8, position);
        carry.ri_parity = (sz_u8_t)(sz_utf8_word_break_count_regional_indicators_before_((sz_cptr_t)text_u8, base) &
                                    1u);

        sz_utf8_word_break_window_t const win = sz_utf8_word_break_window_boundaries_(
            text_u8, length, base, anchor_lane, lane_identity, carry);

        // The trusted band always begins at the anchor (== `position`); advance to the first untrusted codepoint
        // start so the next window re-resolves the seam with full left context.
        sz_u64_t const resolved_starts = win.start_bytes & sz_u64_mask_until_(win.resolved) &
                                         ~sz_u64_mask_until_(anchor_lane);
        if (resolved_starts == 0) break; // no forward progress possible: only the final word remains

        sz_u64_t boundary = win.boundary;
        if (position == word_start) boundary &= ~(1ull << anchor_lane);
        // Drain this window's boundaries into a stack scratch (a window resolves at most 64 boundaries), then copy
        // only the words at scan-order index >= skip_words into the output. This threads carry across ALL words yet
        // retains just the requested trailing slice without re-seeding.
        sz_size_t scratch_starts[64], scratch_lengths[64];
        sz_size_t const drained = sz_utf8_codepoints_drain_forward_(boundary, base, lane_identity, scratch_starts,
                                                                    scratch_lengths, 0, 64, &word_start);
        for (sz_size_t i = 0; i < drained; ++i, ++seen)
            if (seen >= skip_words && stored < capacity)
                starts[stored] = scratch_starts[i], lengths[stored] = scratch_lengths[i], ++stored;
        // When the caller only wants the count, keep threading to enumerate every word; otherwise stop once the
        // requested trailing slice is full.
        if (stored == capacity && total_words_out == SZ_NULL) return stored;
        sz_size_t const last_start_lane = (sz_size_t)(63 - sz_u64_clz(resolved_starts));
        position = base + last_start_lane + sz_utf8_codepoint_length_(text_u8[base + last_start_lane]);
    }

    // The trailing (still-open) word `[word_start, length)` finalizes the count.
    if (total_words_out) *total_words_out = seen + 1;
    if (seen >= skip_words && stored < capacity) {
        starts[stored] = word_start;
        lengths[stored] = length - word_start;
        ++stored;
    }
    return stored;
}

SZ_PUBLIC sz_size_t sz_utf8_words_icelake( //
    sz_cptr_t text, sz_size_t length,                     //
    sz_size_t *word_starts, sz_size_t *word_lengths,      //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    __m512i const lane_identity = sz_utf8_codepoints_lane_identity_icelake_();
    sz_size_t const words = sz_utf8_word_scan_forward_(text_u8, length, 0, 0, lane_identity, word_starts, word_lengths,
                                                       words_capacity, SZ_NULL);
    // The scan fills [0, length): when capacity was hit mid-stream the consumed prefix ends exactly at the end of
    // the last emitted word (== the start of the pending unfinished word); otherwise the whole input was consumed.
    sz_size_t const consumed_end = word_starts[words - 1] + word_lengths[words - 1];
    if (words == words_capacity && consumed_end != length) {
        if (bytes_consumed) *bytes_consumed = consumed_end;
        return words;
    }
    if (bytes_consumed) *bytes_consumed = length;
    return words;
}

#pragma endregion Word_Break Forward Driver

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_ICELAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDS_ICELAKE_H_
