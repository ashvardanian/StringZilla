/**
 *  @file include/stringzilla/utf8_words/icelake.h
 *  @author Ash Vardanian
 *  @brief  Fully-vectorized UAX-29 Word_Break segmentation for AVX-512 (Ice Lake and later). No path scalar-walks
 *          codepoints, spills a ZMM to the stack, calls the serial oracle, or issues a gather.
 *
 *  The text is consumed in 64-byte windows. Each window is decoded by the shared codepoint substrate and classified
 *  to a per-codepoint Word_Break property (ALetter, Numeric, MidLetter, Extend, Regional_Indicator, ...) entirely
 *  in-register: an ASCII permute, big arithmetic ranges, a codepoint < 0x800 page LUT, the shared two-stage trie for the
 *  scattered BMP residue, and a aligned `.rodata` astral trie for the Supplementary Plane. Rules WB3-WB16 then run as
 *  branchless 64-lane bit algebra. The driver emits the boundaries it can fully trust and advances to the last
 *  resolved codepoint start; a small register carry (@ref sz_utf8_word_break_carry_t) threads the cross-window state.
 *
 *  The hard part is that a boundary can depend on bytes beyond the current window. Three cross-window dependencies
 *  drive almost all of the complexity below.
 *
 *  The first is cross-window left context. Judging the boundary at the start of window N needs the last codepoint of
 *  window N-1:
 *  @code
 *      window N-1: "...the ca" | window N: "t's fine"
 *                                ^ is this a boundary?
 *  @endcode
 *  The apostrophe bridge (WB6/7, `AHLetter x MidNumLetQ x AHLetter`) at lane 0 of window N needs the `t` that lives
 *  in window N-1. It is carried in `carry.left_property` (plus the Regional_Indicator parity and the WSegSpace/ZWJ
 *  raw-adjacency bits) and injected as a branchless lane-0 seed, so the next window never re-scans the window edge.
 *
 *  The second is an unbounded right lookahead, the bridge shadow. A Mid between two letters joins them, but the right
 *  letter can be arbitrarily far away, because Extend/Format/ZWJ (combining marks) are transparent under WB4:
 *  @code
 *      "a" "." [200 combining marks] "b" → one word (WB6 still bridges across the marks)
 *      "a" "." [200 combining marks] " " → "a." | " " (the bridge fails)
 *  @endcode
 *  In-window the bridge is resolved by an Extend-transparent `fill` that runs through the marks. Only when the open
 *  Mid run reaches the 64-byte edge still undecided is `resolved` clamped before it and the open-shadow run-state
 *  carried (`carry.bridge_open` / `carry.bridge_kind`); the next, fully-contextual window re-resolves it. This mirrors
 *  the sentence kernel's SB8 mechanism - no per-word scalar bookkeeping and no deferred emission.
 *
 *  The third is malformed UTF-8, where even the codepoint boundaries are ambiguous. The partition must be
 *  capacity-independent, so a streamed or resumed call reproduces the single-pass segmentation:
 *  @code
 *      "a" 0x80 "b" → a | U+FFFD | b (a stray continuation is its own replacement char)
 *      0xF0 "x" → U+FFFD | x (a lone 4-byte lead; "x" starts fresh)
 *      0xC2 0xB7 → U+00B7 (a valid 2-byte codepoint, one element)
 *  @endcode
 *  The kernel uses the canonical maximal-subpart partition (the serial reference's model). The common well-formed
 *  case collapses to `valid & ~continuation` (O(1)); only a genuinely malformed window falls back to a data-dependent
 *  reachability fixpoint. The subtlety is telling a benign window-edge straddle (a valid lead whose continuations
 *  spill into the next window, handled by `complete_limit`) apart from a true end-of-text truncation (a 1-byte
 *  replacement char).
 *
 *  The overwhelmingly common case - well-formed BMP/ASCII text with no open bridge - is kept cheap: the astral
 *  classifier, the page LUT, the trie, the per-lead validity machinery and the partition fixpoint are all rare-class
 *  gated, so the per-window cost is proportional to the window's content, not to the size of the Unicode tables.
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

#pragma region Gather Free Word_Break Classifier

/** @brief  64-byte ZMM tile counts spanning the flattened ASTRAL stage tables (BMP keeps the cheap arithmetic
 *          ranges + 2-byte page LUT, which already classify faster than a trie). */
enum {
    sz_utf8_word_break_astral_stage1_tiles_k = sizeof(sz_utf8_word_break_astral_s1_) / 64,
    sz_utf8_word_break_astral_stage2_tiles_k = sizeof(sz_utf8_word_break_astral_s2_) / 64,
    sz_utf8_word_break_astral_leaf_tiles_k = sizeof(sz_utf8_word_break_astral_leaf_) / 64,
};

/** @brief  Classify 16 astral codepoints (u32 lanes) into 16 Word_Break classes via the 4-stage astral trie:
 *          `astral_s0` → stage1 → stage2 → leaf, addressed by the 8/4/4/4 split of `offset = codepoint - 0x10000`,
 *          every tile read straight from aligned `.rodata` (re-init-free — no per-call `luts`). Byte-identical to
 *          `sz_rune_word_break_property` for astral, replacing the 476-range linear fold. */
SZ_INTERNAL __m512i sz_utf8_word_break_classify_astral16_icelake_(__m512i codepoints) {
    __m512i const offset = _mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x10000));
    __m512i const stage1 = sz_utf8_codepoints_permute256_icelake_(
        sz_utf8_word_break_astral_s0_, _mm512_and_si512(_mm512_srli_epi32(offset, 12), _mm512_set1_epi32(0xFF)));
    __m512i const stage2_index = _mm512_add_epi32(
        _mm512_slli_epi32(stage1, 4), _mm512_and_si512(_mm512_srli_epi32(offset, 8), _mm512_set1_epi32(0xF)));
    __m512i const stage2 = sz_utf8_codepoints_lut_cascade_icelake_(
        sz_utf8_word_break_astral_s1_, sz_utf8_word_break_astral_stage1_tiles_k, stage2_index);
    __m512i const leaf_index = _mm512_add_epi32(_mm512_slli_epi32(stage2, 4),
                                                _mm512_and_si512(_mm512_srli_epi32(offset, 4), _mm512_set1_epi32(0xF)));
    __m512i const leaf = sz_utf8_codepoints_lut_cascade_icelake_(sz_utf8_word_break_astral_s2_,
                                                                 sz_utf8_word_break_astral_stage2_tiles_k, leaf_index);
    __m512i const class_index = _mm512_add_epi32(_mm512_slli_epi32(leaf, 4),
                                                 _mm512_and_si512(offset, _mm512_set1_epi32(0xF)));
    return sz_utf8_codepoints_lut_cascade_icelake_(sz_utf8_word_break_astral_leaf_,
                                                   sz_utf8_word_break_astral_leaf_tiles_k, class_index);
}

/** @brief  AVX-512 classification of an all-ASCII 64-byte vector to WB properties via table lookup. */
SZ_INTERNAL __m512i sz_utf8_word_break_classify_ascii_icelake_(__m512i ascii_bytes) {
    __m512i const low_table = _mm512_loadu_epi8(sz_utf8_word_break_property_ascii_);
    __m512i const high_table = _mm512_loadu_epi8(sz_utf8_word_break_property_ascii_ + 64);
    __mmask64 const high_half = _mm512_test_epi8_mask(ascii_bytes, _mm512_set1_epi8(0x40));
    __m512i const low_result = _mm512_permutexvar_epi8(ascii_bytes, low_table);
    __m512i const high_result = _mm512_permutexvar_epi8(ascii_bytes, high_table);
    return _mm512_mask_blend_epi8(high_half, low_result, high_result);
}

/** @brief  64-lane mask of bytes whose class equals @p value. */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_class_mask_icelake_(__m512i classes, sz_u8_t value) {
    return _cvtmask64_u64(_mm512_cmpeq_epi8_mask(classes, _mm512_set1_epi8((char)value)));
}

/** @brief  The Word_Break class byte held at lane @p lane, extracted in-register by a single byte permute. */
SZ_INTERNAL sz_u8_t sz_utf8_word_break_class_at_icelake_(__m512i classes, sz_size_t lane) {
    __m512i const broadcast = _mm512_permutexvar_epi8(_mm512_set1_epi8((char)lane), classes);
    return (sz_u8_t)_mm_cvtsi128_si32(_mm512_castsi512_si128(broadcast));
}

/** @brief  64-lane mask of lanes whose `(high, low)` 16-bit value lies inside any sorted `[lo, hi]` range (WSegSpace
 *          WB3d and Extended_Pictographic WB3c, which are NOT part of the 4-bit Word_Break model). */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_range16_mask_icelake_( //
    __m512i high, __m512i low, sz_u16_t const *lo_table, sz_u16_t const *hi_table, int count) {
    __mmask64 hit = _cvtu64_mask64(0);
    for (int range = 0; range < count; ++range) {
        __m512i const lo_high = _mm512_set1_epi8((char)(lo_table[range] >> 8));
        __m512i const lo_low = _mm512_set1_epi8((char)(lo_table[range] & 0xFF));
        __m512i const hi_high = _mm512_set1_epi8((char)(hi_table[range] >> 8));
        __m512i const hi_low = _mm512_set1_epi8((char)(hi_table[range] & 0xFF));
        __mmask64 const at_or_above_low = _kor_mask64(
            _mm512_cmp_epu8_mask(high, lo_high, _MM_CMPINT_NLE),
            _kand_mask64(_mm512_cmpeq_epi8_mask(high, lo_high), _mm512_cmp_epu8_mask(low, lo_low, _MM_CMPINT_NLT)));
        __mmask64 const at_or_below_high = _kor_mask64(
            _mm512_cmp_epu8_mask(high, hi_high, _MM_CMPINT_LT),
            _kand_mask64(_mm512_cmpeq_epi8_mask(high, hi_high), _mm512_cmp_epu8_mask(low, hi_low, _MM_CMPINT_LE)));
        hit = _kor_mask64(hit, _kand_mask64(at_or_above_low, at_or_below_high));
    }
    return _cvtmask64_u64(hit);
}

/** @brief  Look up one cp < 0x800 page over @ref sz_utf8_word_break_flat_lut_0800_ via an in-register `vpermi2b`
 *          network (faster than a trie for the dense 2-byte scripts). */
SZ_INTERNAL __m512i sz_utf8_word_break_small_page_icelake_(__m512i high, __m512i low) {
    __m512i const in_seven = _mm512_and_si512(low, _mm512_set1_epi8(0x7F));
    __m512i const low_high_bit = sz_utf8_codepoints_srl8_(low, 7, 0x01);
    __m512i const page = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(high, _mm512_set1_epi8(0x07)), 1),
                                         low_high_bit);
    __m512i segment[32];
    for (int segment_index = 0; segment_index < 32; ++segment_index)
        segment[segment_index] = _mm512_loadu_si512(sz_utf8_word_break_flat_lut_0800_ + segment_index * 64);
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

/** @brief  Classify the 4-byte (astral) lanes of a window via the aligned `.rodata` astral trie. Four 16-lane chunks
 *          reconstruct the 21-bit codepoint and walk the 4-stage trie; the caller blends the result onto `is_four_byte`
 *          lanes. Replaces the per-window 476-range linear fold. */
SZ_INTERNAL __m512i sz_utf8_word_break_classify_four_byte_icelake_(__m512i window, __m512i next1, __m512i next2,
                                                                   __m512i next3) {
    __m512i const byte0 = _mm512_and_si512(window, _mm512_set1_epi8(0x07));
    __m512i const byte1 = _mm512_and_si512(next1, _mm512_set1_epi8(0x3F));
    __m512i const byte2 = _mm512_and_si512(next2, _mm512_set1_epi8(0x3F));
    __m512i const byte3 = _mm512_and_si512(next3, _mm512_set1_epi8(0x3F));
    __m512i const lane_identity = sz_utf8_codepoints_lane_identity_icelake_();
    __m512i astral_class = _mm512_set1_epi8((char)sz_utf8_word_break_other_k);
    for (int chunk = 0; chunk < 4; ++chunk) {
        __m512i const select = _mm512_add_epi8(lane_identity, _mm512_set1_epi8((char)(chunk * 16)));
        __m512i const lead_bits = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_permutexvar_epi8(select, byte0)));
        __m512i const continuation1 = _mm512_cvtepu8_epi32(
            _mm512_castsi512_si128(_mm512_permutexvar_epi8(select, byte1)));
        __m512i const continuation2 = _mm512_cvtepu8_epi32(
            _mm512_castsi512_si128(_mm512_permutexvar_epi8(select, byte2)));
        __m512i const continuation3 = _mm512_cvtepu8_epi32(
            _mm512_castsi512_si128(_mm512_permutexvar_epi8(select, byte3)));
        __m512i const codepoint = _mm512_or_si512(
            _mm512_or_si512(_mm512_slli_epi32(lead_bits, 18), _mm512_slli_epi32(continuation1, 12)),
            _mm512_or_si512(_mm512_slli_epi32(continuation2, 6), continuation3));
        __m512i const chunk_class = sz_utf8_word_break_classify_astral16_icelake_(codepoint);
        __m128i const class_bytes = _mm512_cvtepi32_epi8(chunk_class);
        __m512i const class_broadcast = _mm512_maskz_expand_epi8(_cvtu64_mask64(0xFFFFull << (chunk * 16)),
                                                                 _mm512_castsi128_si512(class_bytes));
        astral_class = _mm512_mask_mov_epi8(astral_class, _cvtu64_mask64((sz_u64_t)0xFFFFull << (chunk * 16)),
                                            class_broadcast);
    }
    return astral_class;
}

/**
 *  @brief  Classify a 64-byte window to per-lane Word_Break properties. ASCII through the ASCII permute, BMP through
 *          arithmetic big ranges (Latin / Hangul / CJK) + the codepoint < 0x800 page LUT + the shared two-stage BMP trie for
 *          the residue; 4-byte leads through the aligned `.rodata` astral trie. All cheap paths are rare-class gated.
 */
SZ_INTERNAL __m512i sz_utf8_word_break_classify_window_icelake_( //
    __m512i window, __m512i high, __m512i low, __mmask64 is_four_byte, __m512i next1, __m512i next2, __m512i next3) {
    __mmask64 const is_ascii = ~_mm512_movepi8_mask(window);
    __mmask64 const high_eq_01 = _mm512_cmpeq_epi8_mask(high, _mm512_set1_epi8(0x01));
    __mmask64 const high_eq_02 = _mm512_cmpeq_epi8_mask(high, _mm512_set1_epi8(0x02));
    __mmask64 const low_le_4f = _mm512_cmp_epu8_mask(low, _mm512_set1_epi8(0x4F), _MM_CMPINT_LE);
    __mmask64 const latin = high_eq_01 | (high_eq_02 & low_le_4f);
    __mmask64 const hangul = (_mm512_cmp_epu8_mask(high, _mm512_set1_epi8((char)0xAC), _MM_CMPINT_NLT) &
                              _mm512_cmp_epu8_mask(high, _mm512_set1_epi8((char)0xD7), _MM_CMPINT_LE)) &
                             ~(_mm512_cmpeq_epi8_mask(high, _mm512_set1_epi8((char)0xD7)) &
                               _mm512_cmp_epu8_mask(low, _mm512_set1_epi8((char)0xA3), _MM_CMPINT_NLE));
    __mmask64 const cjk_combined = (_mm512_cmp_epu8_mask(high, _mm512_set1_epi8(0x34), _MM_CMPINT_NLT) &
                                    _mm512_cmp_epu8_mask(high, _mm512_set1_epi8((char)0x9F), _MM_CMPINT_LE)) &
                                   ~(_mm512_cmpeq_epi8_mask(high, _mm512_set1_epi8(0x4D)) &
                                     _mm512_cmp_epu8_mask(low, _mm512_set1_epi8((char)0xBF), _MM_CMPINT_NLE));
    __mmask64 const fast_aletter = latin | hangul;

    __m512i classes = _mm512_setzero_si512();
    classes = _mm512_mask_blend_epi8(cjk_combined, classes, _mm512_set1_epi8((char)sz_utf8_word_break_other_k));
    classes = _mm512_mask_blend_epi8(fast_aletter, classes, _mm512_set1_epi8((char)sz_utf8_word_break_aletter_k));
    if (is_four_byte)
        classes = _mm512_mask_mov_epi8(classes, is_four_byte,
                                       sz_utf8_word_break_classify_four_byte_icelake_(window, next1, next2, next3));
    if (is_ascii)
        classes = _mm512_mask_blend_epi8(is_ascii, classes, sz_utf8_word_break_classify_ascii_icelake_(window));

    __mmask64 const is_small = _mm512_cmp_epu8_mask(high, _mm512_set1_epi8(0x08), _MM_CMPINT_LT) & ~is_ascii &
                               ~is_four_byte;
    if (is_small) classes = _mm512_mask_mov_epi8(classes, is_small, sz_utf8_word_break_small_page_icelake_(high, low));

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

#pragma region Word_Break Codepoint Partition

/**
 *  @brief  Canonical maximal-subpart codepoint partition (Unicode U+FFFD substitution, the serial reference's exact
 *          model after its malformed-UTF-8 fix). The serial decode (`sz_rune_parse` inside `sz_utf8_decode_`) is
 *          ALL-OR-NOTHING per lead: a multi-byte lead consumes its FULL declared length iff the sequence is well-formed
 *          (every declared continuation byte is a real `10xxxxxx` byte present in the buffer AND the lead is not
 *          overlong / surrogate / out-of-range); otherwise the lead is a single 1-byte U+FFFD and the very next byte
 *          starts a fresh codepoint. Every stray continuation byte is likewise its own 1-byte U+FFFD. For well-formed
 *          UTF-8 this equals the decoder's continuation-bit partition; on malformed input it is capacity-independent
 *          (a streamed resume always lands on the same grid because the partition never depends on the buffer length).
 *
 *  Implemented as a reachability fixpoint: lane 0 is a start (the window is codepoint-aligned), and each start at lane
 *  `i` makes lane `i + subpart(i)` a start, where `subpart(i)` is the declared length when @p claims_full holds at `i`
 *  and 1 otherwise. The four subpart-length masks are the declared-length masks gated by @p claims_full; the fixpoint
 *  is the same shift-OR doubling as the declared-length partition, with no scalar per-lane loop and no gather.
 *
 *  @param claims_full  Lead lanes whose full declared multi-byte sequence is well-formed (computed by the driver,
 *                      mirroring `sz_rune_parse`). A length>=2 lead NOT in this set collapses to a 1-byte U+FFFD.
 */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_subpart_starts_(sz_u64_t length_one, sz_u64_t length_two, sz_u64_t length_three,
                                                        sz_u64_t length_four, sz_u64_t claims_full, sz_u64_t valid) {
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
    sz_u64_t start_bytes;  /**< Codepoint-start lanes under the canonical maximal-subpart partition. */
    sz_u64_t continuation; /**< Claimed continuation bytes (the interior of valid multi-byte codepoints). */
    sz_u64_t forced_other; /**< Lanes whose class must be forced to U+FFFD/Other: strays + short/ill-formed leads. */
    sz_u64_t length_two;   /**< High-nibble declared-length lead masks, reused by the block resolver's truncation. */
    sz_u64_t length_three;
    sz_u64_t length_four;
} sz_utf8_word_break_partition_t;

/**
 *  @brief  Resolve one window into the canonical maximal-subpart codepoint partition (the serial reference's model:
 *          ill-formed bytes become 1-byte U+FFFD). A well-formed window collapses to the O(1) continuation-bit
 *          partition `valid & ~real_continuation`; only a stray continuation, a short lead, or an overlong / surrogate
 *          / range lead takes the data-dependent reachability fixpoint. @p at_end_of_text distinguishes a benign
 *          interior straddle (the next window completes it) from a true end-of-text truncation.
 */
SZ_INTERNAL sz_utf8_word_break_partition_t sz_utf8_word_break_partition_icelake_( //
    __m512i window, __m512i next1, sz_u64_t valid, int at_end_of_text) {
    sz_u64_t const real_continuation = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(
                                           _mm512_and_si512(window, _mm512_set1_epi8((char)0xC0)),
                                           _mm512_set1_epi8((char)0x80))) &
                                       valid;
    // Declared length keyed purely on the HIGH NIBBLE (serial `codepoint_length_`): 0xC/0xD → 2, 0xE → 3, 0xF → 4,
    // everything else → 1. A never-valid lead like 0xFE still declares a 4-byte span; its maximal subpart truncates
    // at the first non-continuation inside that span.
    __m512i const high_nibble = sz_utf8_codepoints_srl8_(window, 4, 0x0F);
    sz_u64_t const length_two = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(high_nibble, _mm512_set1_epi8(0x0C)) |
                                               _mm512_cmpeq_epi8_mask(high_nibble, _mm512_set1_epi8(0x0D))) &
                                valid;
    sz_u64_t const length_three = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(high_nibble, _mm512_set1_epi8(0x0E))) & valid;
    sz_u64_t const length_four = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(high_nibble, _mm512_set1_epi8(0x0F))) & valid;
    sz_u64_t const length_one = valid & ~length_two & ~length_three & ~length_four;
    sz_u64_t const length_ge_two = length_two | length_three | length_four;

    // Declared continuation spans of EVERY multi-byte lead (validity-agnostic). On well-formed UTF-8 this exactly
    // equals the real continuation bytes, so the partition is the continuation-bit one. A mismatch means a stray
    // (run longer than declared) or a lead short of its declared length. A benign edge straddle's missing far
    // continuations sit past `valid` in BOTH masks, so it stays on the O(1) path (a 3-byte-CJK window always ends
    // mid-codepoint and must not be forced onto the fixpoint).
    sz_u64_t const claimed_unmasked = (length_ge_two << 1) | ((length_three | length_four) << 2) | (length_four << 3);
    sz_u64_t const claimed_optimistic = claimed_unmasked & valid;
    // At end-of-text a lead whose declared span runs PAST `valid` is genuinely truncated (a 1-byte U+FFFD), not a
    // benign straddle; force the malformed path so its in-window continuations become strays.
    int const end_of_text_truncated_lead = at_end_of_text && claimed_unmasked != claimed_optimistic;
    // Overlong / surrogate / out-of-range leads (rare) also diverge; detect them only when a multi-byte lead exists.
    sz_u64_t bad_second_byte = 0ull;
    if (length_ge_two) {
        sz_u64_t const next1_at_least_a0 = _cvtmask64_u64(_mm512_cmpge_epu8_mask(next1, _mm512_set1_epi8((char)0xA0)));
        sz_u64_t const next1_at_least_90 = _cvtmask64_u64(_mm512_cmpge_epu8_mask(next1, _mm512_set1_epi8((char)0x90)));
        sz_u64_t const lead_c0_c1 = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8((char)0xC0)) |
                                                   _mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8((char)0xC1))) &
                                    valid;
        sz_u64_t const lead_e0 = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8((char)0xE0))) & valid;
        sz_u64_t const lead_ed = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8((char)0xED))) & valid;
        sz_u64_t const lead_f0 = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8((char)0xF0))) & valid;
        sz_u64_t const lead_f4 = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8((char)0xF4))) & valid;
        sz_u64_t const lead_f5_or_more = _cvtmask64_u64(_mm512_cmpge_epu8_mask(window, _mm512_set1_epi8((char)0xF5))) &
                                         valid;
        bad_second_byte = lead_c0_c1 | (lead_e0 & ~next1_at_least_a0) | (lead_ed & next1_at_least_a0) |
                          (lead_f0 & ~next1_at_least_90) | (lead_f4 & next1_at_least_90) | lead_f5_or_more;
    }

    sz_utf8_word_break_partition_t result;
    result.length_two = length_two;
    result.length_three = length_three;
    result.length_four = length_four;
    if (real_continuation == claimed_optimistic && bad_second_byte == 0 && !end_of_text_truncated_lead) {
        // Well-formed (including the benign edge straddle): maximal-subpart == continuation-bit partition, O(1).
        result.start_bytes = valid & ~real_continuation;
        result.continuation = real_continuation;
        result.forced_other = 0ull;
    }
    else {
        // Malformed window: run the reachability fixpoint with full per-lead validity.
        sz_u64_t const continuation_at_1 = real_continuation >> 1, continuation_at_2 = real_continuation >> 2,
                       continuation_at_3 = real_continuation >> 3;
        sz_u64_t const claims_full = ((length_two & continuation_at_1) |
                                      (length_three & continuation_at_1 & continuation_at_2) |
                                      (length_four & continuation_at_1 & continuation_at_2 & continuation_at_3)) &
                                     ~bad_second_byte;
        sz_u64_t const start_bytes = sz_utf8_word_break_subpart_starts_(length_one, length_two, length_three,
                                                                        length_four, claims_full, valid);
        // A stray continuation makes its own 1-byte U+FFFD start; a short / ill-formed lead is likewise a 1-byte U+FFFD.
        sz_u64_t const stray_continuation = start_bytes & real_continuation;
        sz_u64_t const short_lead = length_ge_two & ~claims_full & start_bytes;
        result.start_bytes = start_bytes;
        result.continuation = real_continuation & ~stray_continuation;
        result.forced_other = short_lead | stray_continuation;
    }
    return result;
}

#pragma endregion Word_Break Codepoint Partition

#pragma region Word_Break Boundary Algebra

/**
 *  @brief  Cross-window left-context register carry: the effective `previous_property` at the next window's anchor,
 *          the open bridge-shadow run-state (`bridge_open` / `bridge_kind`), the open Regional_Indicator run parity
 *          (WB15/WB16), and the WB3d/WB3c raw-adjacency bits. Threaded forward so the vectorized path never re-walks
 *          a straddle, re-derives a carry, or calls the serial oracle.
 */
typedef struct sz_utf8_word_break_carry_t {
    sz_u8_t bridge_open;      /**< 1 if a WB6/7/11/12 bridge shadow is open at the previous block's edge. */
    sz_u8_t bridge_kind;      /**< Which left letter opened the shadow (@ref sz_utf8_word_break_bridge_none_k ...). */
    sz_u8_t left_property;    /**< Effective `previous_property` at the next window's first emitted lane. */
    sz_u8_t have_prev;        /**< 0 only at the very start of the text (WB1 sot). */
    sz_u8_t ri_parity;        /**< Parity of the contiguous RI run open at that lane (0 or 1). */
    sz_u8_t prev_is_wseg;     /**< The codepoint immediately below lane 0 is a WSegSpace (WB3d across the edge). */
    sz_u8_t prev_ends_in_zwj; /**< The previous element's LAST codepoint is a bare ZWJ (WB3c across the edge). */
} sz_utf8_word_break_carry_t;

/** @brief  One classified block resolved into per-lane boundary bits plus the byte the driver advances to. */
typedef struct sz_utf8_word_break_window_t {
    sz_u64_t breaks;    /**< Bit `i` set => a UAX-29 word boundary begins at codepoint-lead lane `i`. */
    sz_size_t resolved; /**< Exclusive upper bound, in bytes, on lanes whose break bit is fully trusted. */
} sz_utf8_word_break_window_t;

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
    // it contributes a CONSTANT toggle to every lane of that run, so it is XORed in AFTER the prefix-XOR.
    if (inbound_parity) bits ^= sz_utf8_codepoints_fill_right_(run_gate & 1ull, run_gate);
    *inclusive_parity_out = bits;
    return ri & (bits ^ ri);
}

/** @brief  Flood up to 8 independent 64-bit seeds (one per `epi64` lane) rightward through one shared @p gate in a
 *          single vector pass: lane `j` of the result equals `sz_utf8_codepoints_fill_right_(seed_j, gate)`. The
 *          Kogge-Stone shift is per-lane (`slli_epi64` never crosses lanes), so all eight letter/numeric/script fills
 *          that share a gate resolve at once on the vector ports instead of one-at-a-time on the integer ports. */
SZ_INTERNAL __m512i sz_utf8_word_break_fill_right_packed_icelake_(__m512i seeds, sz_u64_t gate) {
    __m512i bits = seeds;
    __m512i reach = _mm512_set1_epi64((long long)gate);
    for (int shift = 1; shift < 64; shift <<= 1) {
        bits = _mm512_or_si512(bits, _mm512_and_si512(_mm512_slli_epi64(bits, shift), reach));
        reach = _mm512_and_si512(reach, _mm512_slli_epi64(reach, shift));
    }
    return bits;
}

/** @brief  Leftward counterpart of @ref sz_utf8_word_break_fill_right_packed_ (`srli_epi64` doubling). */
SZ_INTERNAL __m512i sz_utf8_word_break_fill_left_packed_icelake_(__m512i seeds, sz_u64_t gate) {
    __m512i bits = seeds;
    __m512i reach = _mm512_set1_epi64((long long)gate);
    for (int shift = 1; shift < 64; shift <<= 1) {
        bits = _mm512_or_si512(bits, _mm512_and_si512(_mm512_srli_epi64(bits, shift), reach));
        reach = _mm512_and_si512(reach, _mm512_srli_epi64(reach, shift));
    }
    return bits;
}

/** @brief  Lane slots packed into the batched base-letter flood vector: ALetter / Numeric / Hebrew_Letter share one
 *          `fill_*_packed_icelake_` pass over the ignorable flow, the same lanes on the right and left floods. */
enum {
    sz_utf8_word_break_base_aletter_lane_k = 0,
    sz_utf8_word_break_base_numeric_lane_k = 1,
    sz_utf8_word_break_base_hebrew_lane_k = 2,
};

/** @brief  Lane slots packed into the batched flow flood vector: Katakana / ExtendNumLet / Hebrew_Letter floods over
 *          `flow`. The left flood leaves the Hebrew lane unused. */
enum {
    sz_utf8_word_break_flow_katakana_lane_k = 0,
    sz_utf8_word_break_flow_extendnumlet_lane_k = 1,
    sz_utf8_word_break_flow_hebrew_lane_k = 2,
};

/**
 *  @brief  Resolve a block of @p loaded classified bytes (codepoint-lead classes in @p classes, the open bridge
 *          shadow / RI parity / left context arriving in @p carry) into per-lane word-break bits, mirroring the
 *          serial WB1-WB16 over the WB4 element model.
 *
 *  All of WB3-WB16 are pure 64-bit bit algebra over the per-class lane masks: the WB4 transparency `flow`, the
 *  WB6/7/11/12 letter-Mid-letter bridge resolved in-window plus a SHADOW that floods rightward when the right
 *  letter is past the edge, the WB13a/13b ExtendNumLet binds, the WB7a Hebrew single-quote, the WB3/3a/3b newline
 *  force-breaks, the WB3c ZWJ-pictograph and WB3d WSegSpace raw adjacencies, and the WB15/WB16 RI parity scan. The
 *  cross-window left context arrives as register-carry seeds at lane 0; nothing is scalar re-walked.
 *
 *  The bridge's right context is unbounded; the in-window `bridge` is exact unless an open shadow reaches the block
 *  edge undecided. `resolved` is then clamped before that lane so the next, fully-contextual window re-resolves it
 *  (effective-window<64 + register carry; no oracle, no scalar). At true end-of-text (`more_text == 0`) there is no
 *  next block, so a trailing `letter Mid*` force-breaks in-window and nothing is clamped. @p carry is updated with
 *  the trailing run-state read in-register at the block edge by plain shifts.
 */
SZ_INTERNAL sz_utf8_word_break_window_t sz_utf8_word_break_block_breaks_icelake_( //
    __m512i window, __m512i high, __m512i low, __mmask64 four_byte_starts, __m512i next1, __m512i next2, __m512i next3,
    __m512i classes, sz_u64_t start_bytes_all, sz_u64_t continuation_all, sz_u64_t forced_other, sz_u64_t length_two,
    sz_u64_t length_three, sz_u64_t length_four, sz_size_t loaded, sz_utf8_word_break_carry_t *carry,
    sz_bool_t more_text) {

    sz_u64_t const valid = sz_u64_mask_until_(loaded);
    sz_u64_t const start_bytes = start_bytes_all & valid;
    sz_u64_t const continuation = continuation_all & valid;
    sz_size_t const high_lane = loaded - 1;
    int const at_tail = !more_text; // no text past this block: a trailing undecided shadow is decided (force-break)

    // Strict U+FFFD substitution for an ill-formed multi-byte lead truncated by the buffer end: when a lead's
    // declared byte span runs past `loaded`, the serial reference decodes it blind as a 1-byte U+FFFD (Other) rather
    // than folding the zero-padded tail into a spurious BMP value (e.g. a lone `0xCC` at EOF must be Other, not the
    // combining mark U+0300 = Extend). Only the true tail reaches here for such a lead (the driver's `complete_limit`
    // excludes truncated leads whenever more text follows), so reclassify those starts to Other in-register.
    // Declared length per the serial `codepoint_length_` high-nibble table (0xC/0xD → 2, 0xE → 3, 0xF → 4); the
    // driver already derived these masks for the partition, so just restrict them to in-window leads here (no second
    // high-nibble pass). A never-valid lead like 0xFE still declares a 4-byte span and is truncated when the buffer
    // ends inside it.
    sz_u64_t const lead_two = length_two & start_bytes;
    sz_u64_t const lead_three = length_three & start_bytes;
    sz_u64_t const lead_four = length_four & start_bytes;
    sz_u64_t const truncated_raw = ((lead_two & ~sz_u64_mask_until_(loaded > 1 ? loaded - 1 : 0)) |
                                    (lead_three & ~sz_u64_mask_until_(loaded > 2 ? loaded - 2 : 0)) |
                                    (lead_four & ~sz_u64_mask_until_(loaded > 3 ? loaded - 3 : 0))) &
                                   valid;
    if (truncated_raw)
        classes = _mm512_mask_mov_epi8(classes, _cvtu64_mask64(truncated_raw),
                                       _mm512_set1_epi8((char)sz_utf8_word_break_other_k));
    // Every U+FFFD lane — truncated at the edge OR ill-formed (overlong/surrogate/range/bad continuation, already
    // forced to Other by the driver) — must be excluded from the raw-byte WB3c (pictograph) and WB3d (WSegSpace)
    // range checks, which read `high`/`low`/`window` directly and would otherwise match the blind codepoint value.
    sz_u64_t const truncated = truncated_raw | (forced_other & valid);

    sz_u8_t const left_property = carry->left_property;

    sz_u64_t const class_aletter = (sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_aletter_k) |
                                    sz_utf8_word_break_class_mask_icelake_(classes,
                                                                           sz_utf8_word_break_hebrew_letter_k)) &
                                   start_bytes;
    sz_u64_t const class_hebrew = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_hebrew_letter_k) &
                                  start_bytes;
    sz_u64_t const class_numeric = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_numeric_k) &
                                   start_bytes;
    sz_u64_t const class_katakana = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_katakana_k) &
                                    start_bytes;
    sz_u64_t const class_extendnumlet =
        sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_extendnumlet_k) & start_bytes;
    sz_u64_t const lead_ignorable = (sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_extend_k) |
                                     sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_zwj_k) |
                                     sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_format_k)) &
                                    start_bytes;

    sz_u64_t const mid_letter_or_quotes =
        (sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_midletter_k) |
         sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_mid_quotes_k)) &
        start_bytes;
    sz_u64_t const mid_num_or_quotes = (sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_midnum_k) |
                                        sz_utf8_word_break_class_mask_icelake_(classes,
                                                                               sz_utf8_word_break_mid_quotes_k)) &
                                       start_bytes;
    sz_u64_t const mid_quotes = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_mid_quotes_k) &
                                start_bytes;
    sz_u64_t const mid_any = (sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_midletter_k) |
                              sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_midnum_k) |
                              sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_mid_quotes_k)) &
                             start_bytes;
    sz_u64_t const class_cr = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_cr_k) & start_bytes;
    sz_u64_t const class_lf = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_lf_k) & start_bytes;
    sz_u64_t const class_newline = (class_cr | class_lf |
                                    sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_newline_k)) &
                                   start_bytes;
    sz_u64_t const class_zwj = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_zwj_k) & start_bytes;
    sz_u64_t const class_regional = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_regional_ind_k) &
                                    start_bytes;

    // WSegSpace lanes (WB3d) are outside the 4-bit Word_Break model; detect them by branchless (high,low) range
    // membership. The decoder fills `high`/`low` for >=2-byte leads only, so detect the sole ASCII WSegSpace
    // (U+0020) from the raw window byte and gate the multibyte range check to non-ASCII lanes.
    sz_u64_t const non_ascii_lanes = _cvtmask64_u64(_mm512_movepi8_mask(window)) & valid;
    // Rare-class gate: the multibyte WSegSpace set (U+1680, U+2000-200A, U+202F, U+205F, U+3000) needs the range scan
    // only on non-ASCII lanes; an all-ASCII window skips it. The sole ASCII WSegSpace (U+0020) is a raw byte compare.
    sz_u64_t wseg_multibyte = 0ull;
    if (non_ascii_lanes)
        wseg_multibyte = sz_utf8_word_break_range16_mask_icelake_(high, low, sz_utf8_word_break_wseg_lo_,
                                                                  sz_utf8_word_break_wseg_hi_,
                                                                  sz_utf8_word_break_wseg_count_k) &
                         non_ascii_lanes;
    sz_u64_t const wseg = (wseg_multibyte |
                           (_cvtmask64_u64(_mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8(0x20))) & valid)) &
                          start_bytes & ~truncated; // a truncated lead is U+FFFD, not WSegSpace

    // WB4: Extend/Format/ZWJ are transparent. A "flow" mask lets a class run reach across ignorables (and the
    // CLAIMED continuation bytes of a multi-byte codepoint) so neighbour shifts read the effective adjacent class.
    // Under the canonical maximal-subpart partition a STRAY continuation byte is its OWN 1-byte U+FFFD (Other) start,
    // claimed by no lead, so it is excluded from `flow_ignorable` and breaks every join/bridge SYMMETRICALLY on both
    // sides — exactly matching the serial forward decode and its reach-checked backward look-back. On well-formed
    // UTF-8 `stray_continuation == 0` and `continuation_all` is already strays-free, so this is a no-op.
    sz_u64_t const flow_ignorable = continuation | lead_ignorable;
    // Base ALetter/Numeric/Hebrew right- and left-floods over `flow_ignorable`, batched three-per-vector-pass into
    // the lanes named by `sz_utf8_word_break_base_*_lane_k`.
    sz_u512_vec_t base_seeds;
    base_seeds.zmm = _mm512_setzero_si512();
    base_seeds.u64s[sz_utf8_word_break_base_aletter_lane_k] = class_aletter;
    base_seeds.u64s[sz_utf8_word_break_base_numeric_lane_k] = class_numeric;
    base_seeds.u64s[sz_utf8_word_break_base_hebrew_lane_k] = class_hebrew;
    sz_u512_vec_t base_right, base_left;
    base_right.zmm = sz_utf8_word_break_fill_right_packed_icelake_(base_seeds.zmm, flow_ignorable);
    base_left.zmm = sz_utf8_word_break_fill_left_packed_icelake_(base_seeds.zmm, flow_ignorable);
    sz_u64_t const aletter_right_base = base_right.u64s[sz_utf8_word_break_base_aletter_lane_k];
    sz_u64_t const numeric_right_base = base_right.u64s[sz_utf8_word_break_base_numeric_lane_k];
    sz_u64_t const hebrew_right_base = base_right.u64s[sz_utf8_word_break_base_hebrew_lane_k];
    sz_u64_t const aletter_left_base = base_left.u64s[sz_utf8_word_break_base_aletter_lane_k];
    sz_u64_t const numeric_left_base = base_left.u64s[sz_utf8_word_break_base_numeric_lane_k];
    sz_u64_t const hebrew_left_base = base_left.u64s[sz_utf8_word_break_base_hebrew_lane_k];
    int const left_is_aletter = left_property == sz_utf8_word_break_aletter_k;
    int const left_is_hebrew = left_property == sz_utf8_word_break_hebrew_letter_k;
    int const left_is_numeric = left_property == sz_utf8_word_break_numeric_k;

    // The leading window-edge region runs from lane 0 up to and including the first significant (non-ignorable)
    // lead; the cross-window carried class is injected here so lane-0 left context needs no scalar re-walk.
    sz_u64_t const significant_leads = start_bytes & ~lead_ignorable;
    sz_u64_t const edge_region = significant_leads ? sz_u64_mask_until_((sz_size_t)sz_u64_ctz(significant_leads) + 1)
                                                   : valid;

    // Pre-bridge carry seeds for the bridge's left-letter detection. A Mid's previous-X signal is
    // `(X_right_base << 1) | edge_region`: the carried class is the Mid's effective previous across the LEADING
    // ignorable run, and a carried OPEN bridge of kind X also presents X as the left letter to a lane-0 Mid run.
    int const carry_bridge_aletter = carry->bridge_open && carry->bridge_kind == sz_utf8_word_break_bridge_aletter_k;
    int const carry_bridge_numeric = carry->bridge_open && carry->bridge_kind == sz_utf8_word_break_bridge_numeric_k;
    int const carry_bridge_hebrew = carry->bridge_open && carry->bridge_kind == sz_utf8_word_break_bridge_hebrew_k;
    sz_u64_t const seed_aletter_pre = sz_u64_or_if_(0ull, edge_region,
                                                    left_is_aletter || left_is_hebrew || carry_bridge_aletter);
    sz_u64_t const seed_numeric_pre = sz_u64_or_if_(0ull, edge_region, left_is_numeric || carry_bridge_numeric);
    sz_u64_t const seed_hebrew_pre = sz_u64_or_if_(0ull, edge_region, left_is_hebrew || carry_bridge_hebrew);

    // WB6/WB7 and WB11/WB12: a Mid* lane between two letters (or numerics) bridges the run. The bridge "Mid" for
    // letters/numerics is MidNumLetQ (MidNumLet + Single_Quote), which excludes Double_Quote; Double_Quote bridges
    // ONLY Hebrew x " x Hebrew (WB7b/WB7c), so isolate U+0022 and route it through a Hebrew-only bridge.
    sz_u64_t const double_quote = mid_quotes & _cvtmask64_u64(_mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8(0x22)));
    sz_u64_t const mid_letter_quotes_no_double = mid_letter_or_quotes & ~double_quote;
    sz_u64_t const mid_num_quotes_no_double = mid_num_or_quotes & ~double_quote;
    // A Mid* lane has an open LEFT letter when the lane just below it (across `flow_ignorable`) is the matching
    // letter, or it sits in the leading edge region after a carried letter / open bridge.
    sz_u64_t const mid_open_aletter = mid_letter_quotes_no_double & ((aletter_right_base << 1) | seed_aletter_pre);
    sz_u64_t const mid_open_numeric = mid_num_quotes_no_double & ((numeric_right_base << 1) | seed_numeric_pre);
    sz_u64_t const mid_open_hebrew = double_quote & ((hebrew_right_base << 1) | seed_hebrew_pre);
    sz_u64_t const mid_open = mid_open_aletter | mid_open_numeric | mid_open_hebrew;
    sz_u64_t bridge = (mid_open_aletter & (aletter_left_base >> 1)) | (mid_open_numeric & (numeric_left_base >> 1)) |
                      (mid_open_hebrew & (hebrew_left_base >> 1));

    sz_u64_t const flow = flow_ignorable | bridge;

    // bridge==0 reuse: when no Mid* bridges a letter/numeric run, `flow == flow_ignorable` exactly, so the four
    // ALetter/Numeric left/right smears over `flow` are bit-identical to the `_base` smears already computed over
    // `flow_ignorable`. Katakana / ExtendNumLet / Hebrew have no `_base`, so they are smeared fresh regardless.
    sz_u64_t aletter_right, numeric_right, aletter_left, numeric_left;
    if (bridge == 0) {
        aletter_right = aletter_right_base;
        numeric_right = numeric_right_base;
        aletter_left = aletter_left_base;
        numeric_left = numeric_left_base;
    }
    else {
        aletter_right = sz_utf8_codepoints_fill_right_(class_aletter, flow);
        numeric_right = sz_utf8_codepoints_fill_right_(class_numeric, flow);
        aletter_left = sz_utf8_codepoints_fill_left_(class_aletter, flow);
        numeric_left = sz_utf8_codepoints_fill_left_(class_numeric, flow);
    }
    // Katakana / ExtendNumLet / Hebrew floods over `flow`, batched into the lanes named by
    // `sz_utf8_word_break_flow_*_lane_k` (the left flood has no Hebrew lane).
    sz_u512_vec_t flow_right_seeds, flow_left_seeds;
    flow_right_seeds.zmm = _mm512_setzero_si512();
    flow_right_seeds.u64s[sz_utf8_word_break_flow_katakana_lane_k] = class_katakana;
    flow_right_seeds.u64s[sz_utf8_word_break_flow_extendnumlet_lane_k] = class_extendnumlet;
    flow_right_seeds.u64s[sz_utf8_word_break_flow_hebrew_lane_k] = class_hebrew;
    flow_left_seeds.zmm = _mm512_setzero_si512();
    flow_left_seeds.u64s[sz_utf8_word_break_flow_katakana_lane_k] = class_katakana;
    flow_left_seeds.u64s[sz_utf8_word_break_flow_extendnumlet_lane_k] = class_extendnumlet;
    sz_u512_vec_t flow_right, flow_left;
    flow_right.zmm = sz_utf8_word_break_fill_right_packed_icelake_(flow_right_seeds.zmm, flow);
    flow_left.zmm = sz_utf8_word_break_fill_left_packed_icelake_(flow_left_seeds.zmm, flow);
    sz_u64_t const katakana_right = flow_right.u64s[sz_utf8_word_break_flow_katakana_lane_k];
    sz_u64_t const extendnumlet_right = flow_right.u64s[sz_utf8_word_break_flow_extendnumlet_lane_k];
    sz_u64_t const hebrew_right = flow_right.u64s[sz_utf8_word_break_flow_hebrew_lane_k];
    sz_u64_t const katakana_left = flow_left.u64s[sz_utf8_word_break_flow_katakana_lane_k];
    sz_u64_t const extendnumlet_left = flow_left.u64s[sz_utf8_word_break_flow_extendnumlet_lane_k];

    // WB3/WB3a/WB3b/WB3c: Newline (CR/LF/Newline) forces a break on both sides except CR x LF, and ZWJ x suppresses
    // the break after a ZWJ. The left neighbour of lane 0 is taken from `left_property` via a lane-0 seed. A
    // multi-byte Newline (e.g. U+2028/U+2029 = 3 bytes) reaches the next lead only after smearing across its OWN
    // continuation bytes (NOT the wider `flow`, which would also hop ignorables and break WB3a's de-ignore: the
    // first ignorable after a newline must still see the newline as its raw predecessor). The shift `<< 1` past the
    // continuation-smear then lands on the next codepoint start — be it a significant lead or a de-ignored ignorable.
    sz_u64_t previous_newline =
        (sz_utf8_codepoints_smear_right_(class_newline, continuation, sz_utf8_word_break_smear_steps_k) << 1);
    sz_u64_t previous_cr = (sz_utf8_codepoints_smear_right_(class_cr, continuation, sz_utf8_word_break_smear_steps_k)
                            << 1);
    // WB3c "ZWJ x" reaches only across the ZWJ's OWN continuation bytes (not the wider `flow`); an intervening
    // Extend breaks the adjacency. Seed lane 0 from `carry->prev_ends_in_zwj` (precisely "ends in ZWJ").
    sz_u64_t previous_zwj = (sz_utf8_codepoints_smear_right_(class_zwj, continuation, sz_utf8_word_break_smear_steps_k)
                             << 1) |
                            sz_u64_or_if_(0ull, 1ull, carry->prev_ends_in_zwj != 0);
    int const left_is_cr = left_property == sz_utf8_word_break_cr_k;
    int const left_is_newline = left_is_cr || left_property == sz_utf8_word_break_lf_k ||
                                left_property == sz_utf8_word_break_newline_k;
    previous_cr = sz_u64_or_if_(previous_cr, 1ull, left_is_cr);
    previous_newline = sz_u64_or_if_(previous_newline, 1ull, left_is_newline);

    // Cross-window previous-class seed. The carried class is the effective previous of every lead in the leading
    // `flow` run (the `edge_region`), AND of the first lead immediately after a leading BRIDGED Mid (WB6/7 makes the
    // Mid transparent, so the letter after a `letter Mid` bridge spanning the window edge sees the carried letter).
    int const leading_mid_bridged = (edge_region & bridge) != 0;
    sz_u64_t const carry_seed_extension = leading_mid_bridged
                                              ? ((sz_utf8_codepoints_fill_right_(edge_region, flow) << 1) & start_bytes)
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

    // WB3c: ZWJ x Extended_Pictographic. The pictograph set is not in the 4-bit Word_Break model, so it is detected
    // by branchless (high,low) range membership, gated behind an in-window ZWJ (or a carried ZWJ).
    if (class_zwj || carry->prev_ends_in_zwj) {
        sz_u64_t const four_byte = _cvtmask64_u64(four_byte_starts) & valid;
        __m512i const plane = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(window, _mm512_set1_epi8(0x07)), 2),
                                              sz_utf8_codepoints_srl8_(next1, 4, 0x03));
        sz_u64_t const plane_one = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(plane, _mm512_set1_epi8(0x01)));
        sz_u64_t const pictographic_bmp = sz_utf8_word_break_range16_mask_icelake_( //
            high, low, sz_utf8_word_break_pict_bmp_lo_, sz_utf8_word_break_pict_bmp_hi_,
            sz_utf8_word_break_pict_bmp_count_k);
        __m512i const smp_mid = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(next1, _mm512_set1_epi8(0x0F)), 4),
                                                sz_utf8_codepoints_srl8_(next2, 2, 0x0F));
        __m512i const smp_low = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(next2, _mm512_set1_epi8(0x03)), 6),
                                                _mm512_and_si512(next3, _mm512_set1_epi8(0x3F)));
        sz_u64_t const pictographic_smp = sz_utf8_word_break_range16_mask_icelake_( //
            smp_mid, smp_low, sz_utf8_word_break_pict_smp_lo_, sz_utf8_word_break_pict_smp_hi_,
            sz_utf8_word_break_pict_smp_count_k);
        sz_u64_t const pictographic = ((pictographic_bmp & non_ascii_lanes & ~four_byte) |
                                       (pictographic_smp & four_byte & plane_one)) &
                                      start_bytes & ~truncated; // a truncated lead is U+FFFD, not pictographic
        join |= previous_zwj & pictographic;                    // WB3c
    }

    // WB3d: WSegSpace x WSegSpace (runs of horizontal spaces stay one segment), raw adjacency over the space's OWN
    // continuation bytes only, with a carried "previous is WSegSpace" lane-0 seed.
    sz_u64_t const previous_wseg =
        (sz_utf8_codepoints_smear_right_(wseg, continuation, sz_utf8_word_break_smear_steps_k) << 1) |
        sz_u64_or_if_(0ull, 1ull, carry->prev_is_wseg != 0);
    join |= previous_wseg & wseg;

    // WB15/WB16: Regional_Indicator pairs, resolved in-register by the segmented parity scan; the run gate spans the
    // RI starts and the ignorables/continuations between two RIs, the first RI of the window continuing the inbound
    // run via a lane-0 seed when lane 0 is itself a run lane.
    sz_u64_t ri_inclusive = 0;
    int const lane0_continues_ri = carry->left_property == sz_utf8_word_break_regional_ind_k &&
                                   ((class_regional | flow_ignorable) & 1ull) != 0;
    sz_u64_t const ri_run_gate = sz_u64_or_if_(sz_utf8_codepoints_fill_right_(class_regional, flow_ignorable) |
                                                   sz_utf8_codepoints_fill_left_(class_regional, flow_ignorable) |
                                                   class_regional,
                                               1ull, lane0_continues_ri);
    sz_u64_t const ri_join = sz_utf8_word_break_ri_join_(
        class_regional, ri_run_gate, (sz_u8_t)(lane0_continues_ri ? carry->ri_parity : 0), &ri_inclusive);
    join |= ri_join;

    // WB3a/WB3b force a break before and after every Newline/CR/LF lane, overriding any join (CR x LF kept).
    sz_u64_t const force_break = ((previous_newline | class_newline) & ~(previous_cr & class_lf)) & start_bytes;

    sz_u64_t const start_bytes_no_bridge = start_bytes & ~bridge;
    sz_u64_t boundary = (~join) & start_bytes_no_bridge;
    // WB7a is Hebrew_Letter x Single_Quote (U+0027) only; isolate the single-quote byte (the `mid_quotes` class also
    // covers Double_Quote and MidNumLet). WB7b/WB7c Hebrew x Double_Quote x Hebrew is handled by the letter bridge.
    sz_u64_t const single_quote = mid_quotes & _cvtmask64_u64(_mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8(0x27)));
    boundary &= ~(single_quote & previous_hebrew); // WB7a: Hebrew_Letter x Single_Quote
    // WB4: never break before an Extend/Format/ZWJ lane — it attaches to the preceding codepoint. The only breaks
    // before an ignorable are the newline-driven ones, which `force_break` re-adds below.
    boundary &= ~lead_ignorable;
    boundary |= force_break & start_bytes_no_bridge; // WB3a/WB3b break around Newline/CR/LF

    // Bridge shadow (the deferred WB6/7/11/12 right context). When an open Mid* (its left letter present, in-window
    // or carried) is followed only by Mid*/ignorables/continuations up to the block edge with NO matching right
    // letter yet, the verdict on the break AFTER the Mid (before its right letter, which lies past the edge) depends
    // on the next block. Flood the open Mid* rightward across the bridge gate; if the flood reaches the top valid
    // lane the shadow is undecided. Flooding it back left gives the contiguous undecided region, whose LOWEST open
    // Mid lane is where the trust horizon ends: the next window re-anchors there and sees the right letter. At the
    // buffer tail (`more_text == 0`) there is no next block, so the shadow is decided: the `letter Mid*`
    // force-breaks (no clamp), which is what `boundary` already holds.
    sz_u64_t const bridge_gate = flow_ignorable | mid_any;
    sz_u64_t const top_bit = at_tail ? 0ull : (1ull << high_lane);
    sz_u64_t const open_to_edge = sz_utf8_codepoints_fill_right_(mid_open, bridge_gate) & top_bit;
    sz_u64_t const undecided = (!at_tail && open_to_edge)
                                   ? (mid_open & sz_utf8_codepoints_fill_left_(open_to_edge, bridge_gate))
                                   : 0ull;

    // Boundaries only at codepoint leads; SOT (lane 0) is the open word's start, never an interior boundary, unless
    // it is carried (have_prev). Clamp the trusted band before the first undecided bridge shadow.
    sz_u64_t const lowbit = carry->have_prev ? 1ull : 0ull;
    sz_u64_t const produced = start_bytes & (~1ull | lowbit);
    boundary &= produced;
    sz_size_t resolved = loaded;
    if (undecided) {
        sz_size_t const lane = (sz_size_t)sz_u64_ctz(undecided);
        resolved = lane;
        boundary &= sz_u64_mask_until_(lane);
    }

    // Update the carry from the block's high edge (resolved lane and below). `edge` is the last byte trusted; the
    // last codepoint-start lane below `resolved` is the cross-window anchor's predecessor.
    sz_u64_t const resolved_valid = sz_u64_mask_until_(resolved);
    sz_u64_t const resolved_starts = start_bytes & resolved_valid;
    sz_u64_t const resolved_significant = resolved_starts & ~lead_ignorable;
    // WB4 effective `previous_property`: the BASE class of the last element below `resolved`. Trailing
    // ignorables absorb into the last significant lead, EXCEPT the first ignorable directly after a Newline/CR/LF
    // (de-ignored — it becomes its own base), so that first ignorable's class is threaded instead. Newline
    // classes are not in any `flow` gate, so the first post-newline ignorable already carries its own class via
    // the `previous_*` shifts in the NEXT window — but the BASE we thread must reflect the de-ignore here so the
    // next window does not re-break before every trailing ignorable.
    sz_u8_t left_out;
    if (resolved_significant == 0) {
        // No significant lead resolved: keep the inbound carry across an all-ignorable block, unless a leading
        // ignorable here is de-ignored after a carried newline (then it becomes the base).
        int const carry_is_newline = left_property == sz_utf8_word_break_cr_k ||
                                     left_property == sz_utf8_word_break_lf_k ||
                                     left_property == sz_utf8_word_break_newline_k;
        sz_u64_t const ignorable_here = lead_ignorable & resolved_valid;
        if ((carry_is_newline || carry->have_prev == 0) && ignorable_here)
            // The first ignorable after a Newline (or at sot) is de-ignored - it becomes the element base, so
            // thread ITS class (Extend / Format / ZWJ) onward, read straight from that lane.
            left_out = sz_utf8_word_break_class_at_icelake_(classes, (sz_size_t)sz_u64_ctz(ignorable_here));
        else left_out = left_property;
    }
    else {
        // The base class is whatever class the last resolved significant lead holds, read in one byte permute.
        sz_size_t const last_lane = (sz_size_t)(63 - sz_u64_clz(resolved_significant));
        sz_u8_t base_class = sz_utf8_word_break_class_at_icelake_(classes, last_lane);
        int const base_is_newline = base_class == sz_utf8_word_break_cr_k || base_class == sz_utf8_word_break_lf_k ||
                                    base_class == sz_utf8_word_break_newline_k;
        // A post-newline ignorable above the last significant lead is itself de-ignored to its own base class.
        sz_u64_t const ignorable_after = lead_ignorable & resolved_valid & ~sz_u64_mask_until_(last_lane + 1);
        if (base_is_newline && ignorable_after)
            base_class = sz_utf8_word_break_class_at_icelake_(classes, (sz_size_t)sz_u64_ctz(ignorable_after));
        left_out = base_class;
    }
    carry->left_property = left_out;
    carry->have_prev = 1;

    // Outbound RI parity: the inclusive parity at the last significant lane below `resolved`, when that lane's
    // RI run is still open. When no significant lead resolved, thread the inbound parity (a >64-byte RI run).
    if (resolved_significant) {
        sz_size_t const edge_lane = (sz_size_t)(63 - sz_u64_clz(resolved_significant));
        sz_u8_t const top_is_regional = (sz_u8_t)((class_regional >> edge_lane) & 1ull);
        sz_u8_t const edge_parity = (sz_u8_t)((ri_inclusive >> edge_lane) & 1ull);
        carry->ri_parity = top_is_regional ? edge_parity : 0;
    }
    // else: keep carry->ri_parity unchanged.

    // Outbound WB3d / WB3c raw-adjacency: is the last resolved codepoint a WSegSpace / a bare ZWJ?
    if (resolved_starts) {
        sz_size_t const edge_start = (sz_size_t)(63 - sz_u64_clz(resolved_starts));
        carry->prev_is_wseg = (sz_u8_t)((wseg >> edge_start) & 1ull);
        carry->prev_ends_in_zwj = (sz_u8_t)((class_zwj >> edge_start) & 1ull);
    }
    else {
        carry->prev_is_wseg = 0;
        carry->prev_ends_in_zwj = 0;
    }

    // Outbound bridge shadow: is a Mid* bridge open and undecided at the trusted edge? A shadow open at
    // `resolved` floods the open Mid* rightward through `bridge_gate`; if it reaches the resolved edge its left
    // letter kind is carried so the next window's first matching letter completes the bridge. The shadow's left
    // letter is the carried `mid_open_*` kind at the highest open Mid* below `resolved`.
    sz_u64_t const open_at_edge = sz_utf8_codepoints_fill_right_(mid_open, bridge_gate) & resolved_valid;
    sz_u64_t const open_run = open_at_edge & ~sz_u64_mask_until_(resolved > 0 ? resolved - 1 : 0);
    if (open_run) {
        // The bridge is open at the edge. Pick the kind of the highest open Mid* whose shadow reaches the edge.
        sz_u64_t const reaching = mid_open & sz_utf8_codepoints_fill_left_(open_run, bridge_gate);
        if (reaching) {
            sz_size_t const mid_lane = (sz_size_t)(63 - sz_u64_clz(reaching));
            sz_u64_t const mid_bit = 1ull << mid_lane;
            carry->bridge_open = 1;
            carry->bridge_kind = (sz_u8_t)((mid_open_hebrew & mid_bit)
                                               ? sz_utf8_word_break_bridge_hebrew_k
                                               : ((mid_open_numeric & mid_bit) ? sz_utf8_word_break_bridge_numeric_k
                                                                               : sz_utf8_word_break_bridge_aletter_k));
        }
        else {
            carry->bridge_open = 0;
            carry->bridge_kind = sz_utf8_word_break_bridge_none_k;
        }
    }
    else if (resolved_significant) {
        // A significant lead resolved with no open shadow at the edge: any inbound bridge is decided/closed.
        carry->bridge_open = 0;
        carry->bridge_kind = sz_utf8_word_break_bridge_none_k;
    }
    // else: an all-ignorable block keeps the inbound bridge open for the driver to thread.

    sz_utf8_word_break_window_t result;
    result.breaks = boundary;
    result.resolved = resolved;
    return result;
}

#pragma endregion Word_Break Boundary Algebra

#pragma region Word_Break Forward Driver

/**
 *  @brief  Forward UAX-29 word segmentation over `[0, length)`, mirroring the sentence driver's clean-advance loop:
 *          each iteration decodes one 64-byte window at a codepoint-aligned `position`, classifies it in-register,
 *          resolves the no-break algebra over the complete-codepoint region, drains the trusted band, and advances
 *          by `win.resolved` with the small register carry threaded forward. The only cross-window state is the
 *          register carry; there is no scalar back-walk, no deferred-emission state machine, and no carry
 *          re-derivation by re-reading the text. Every iteration advances at least one codepoint on all inputs.
 */
SZ_PUBLIC sz_size_t sz_utf8_words_icelake(           //
    sz_cptr_t text, sz_size_t length,                //
    sz_size_t *word_starts, sz_size_t *word_lengths, //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    __m512i const lane_identity = sz_utf8_codepoints_lane_identity_icelake_();

    sz_size_t words = 0;      // words written to the output
    sz_size_t word_start = 0; // start byte of the currently open (unfinished) word
    sz_size_t position = 0;   // codepoint-aligned anchor of the next window (advances cleanly)

    sz_utf8_word_break_carry_t carry;
    carry.bridge_open = 0;
    carry.bridge_kind = sz_utf8_word_break_bridge_none_k;
    carry.left_property = (sz_u8_t)sz_utf8_word_break_other_k; // WB1: sot has no left context
    carry.have_prev = 0;
    carry.ri_parity = 0;
    carry.prev_is_wseg = 0;
    carry.prev_ends_in_zwj = 0;

    while (position < length) {
        sz_utf8_codepoints_window_t const decoded = sz_utf8_codepoints_decode_window_(text_u8 + position,
                                                                                      length - position, lane_identity);
        sz_size_t const loaded = decoded.loaded;
        __m512i const window = decoded.window;
        sz_u64_t const valid = sz_u64_mask_until_(loaded);
        __m512i const high = decoded.high;
        __m512i const low = decoded.low;
        __m512i const next1 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(1)), window);
        __m512i const next2 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(2)), window);
        __m512i const next3 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(3)), window);
        __m512i classes = sz_utf8_word_break_classify_window_icelake_(window, high, low, decoded.four_byte_starts,
                                                                      next1, next2, next3);

        // Canonical maximal-subpart partition (the serial reference's exact model after its malformed-UTF-8 fix):
        // a multi-byte lead claims continuation bytes only up to the first non-continuation OR its declared length,
        // and every unclaimed byte is its own 1-byte U+FFFD start. On well-formed UTF-8 this equals the decoder's
        // continuation-bit partition, so the no-break ALGEBRA (which reads the per-codepoint classes) is unchanged.
        sz_utf8_word_break_partition_t const partition = sz_utf8_word_break_partition_icelake_(
            window, next1, valid, position + loaded >= length);
        sz_u64_t const start_bytes_all = partition.start_bytes;
        sz_u64_t const continuation_all = partition.continuation;
        sz_u64_t const forced_other = partition.forced_other;
        sz_u64_t const length_two = partition.length_two;
        sz_u64_t const length_three = partition.length_three;
        sz_u64_t const length_four = partition.length_four;
        if (forced_other)
            classes = _mm512_mask_mov_epi8(classes, _cvtu64_mask64(forced_other),
                                           _mm512_set1_epi8((char)sz_utf8_word_break_other_k));

        sz_bool_t const more_text = (position + loaded < length) ? sz_true_k : sz_false_k;

        // Effective-window<64: when more text follows, the final codepoint's blind byte span may spill past the
        // 64-byte edge — either its declared continuation bytes are missing (a multi-byte lead at the tail) or
        // trailing continuation bytes run into the next window. Trust and classify only up to the last fully
        // decoded codepoint (`complete_limit`) and re-anchor on that lead. Guaranteed-progress: never below 1.
        sz_size_t complete_limit = loaded;
        if (more_text) {
            sz_u64_t const valid_lanes = sz_u64_mask_until_(loaded);
            // Use the declared-length (high-nibble) lead masks restricted to the maximal-subpart starts, so a
            // straddling never-valid lead (e.g. a trailing 0xFE) is excluded exactly as the serial walk would skip it.
            // A lead whose declared span exceeds `loaded` (it has fewer real continuations than declared, or runs off
            // the edge) is straddling — the next window may complete or truncate it, so trust ends before it.
            sz_u64_t const two = length_two & start_bytes_all;
            sz_u64_t const three = length_three & start_bytes_all;
            sz_u64_t const four = length_four & start_bytes_all;
            sz_u64_t const straddle = ((two & ~sz_u64_mask_until_(loaded > 1 ? loaded - 1 : 0)) |
                                       (three & ~sz_u64_mask_until_(loaded > 2 ? loaded - 2 : 0)) |
                                       (four & ~sz_u64_mask_until_(loaded > 3 ? loaded - 3 : 0))) &
                                      valid_lanes;
            sz_size_t limit = straddle ? (sz_size_t)sz_u64_ctz(straddle) : loaded;
            // A continuation byte just past the window edge extends the LAST codepoint's blind span into the next
            // window only when that codepoint actually claims it (its declared span reaches the edge). The straddle
            // check already excludes any lead whose declared span exceeds `loaded`, so here we cap to the last start
            // only if the edge byte is a continuation AND the last codepoint ends exactly at `loaded` (a multi-byte
            // codepoint flush against the edge whose blind decode may absorb the next byte).
            if ((text_u8[position + loaded] & 0xC0) == 0x80) {
                sz_size_t const last_lead = (sz_size_t)(63 - sz_u64_clz(start_bytes_all));
                sz_size_t const last_lead_length = sz_utf8_codepoint_length_(text_u8[position + last_lead]);
                if (last_lead + last_lead_length > loaded && last_lead < limit) limit = last_lead;
            }
            if (limit > 0) complete_limit = limit; // `limit == 0` keeps the whole window for guaranteed progress
        }

        // Resolve over the complete region; `carry_full` is the exact run-state at the complete edge.
        sz_utf8_word_break_carry_t carry_full = carry;
        sz_utf8_word_break_window_t const win = sz_utf8_word_break_block_breaks_icelake_(
            window, high, low, decoded.four_byte_starts, next1, next2, next3, classes, start_bytes_all,
            continuation_all, forced_other, length_two, length_three, length_four, complete_limit, &carry_full,
            more_text);

        sz_size_t const adv = win.resolved;
        sz_u64_t boundary_lanes = win.breaks & sz_u64_mask_until_(adv);

        words = sz_utf8_codepoints_drain_forward_(boundary_lanes, position, lane_identity, word_starts, word_lengths,
                                                  words, words_capacity, &word_start);
        if (words == words_capacity) {
            if (bytes_consumed) *bytes_consumed = word_start;
            return words;
        }

        if (adv > 0 && adv < complete_limit) {
            // Bridge-shadow clamp strictly before the complete edge: rebuild the carry to the clamp byte so the next
            // window's left context + open-shadow run-state are exact (break bits discarded; carry side-effect only).
            sz_utf8_word_break_carry_t carry_to_edge = carry;
            sz_utf8_word_break_block_breaks_icelake_(
                window, high, low, decoded.four_byte_starts, next1, next2, next3, classes, start_bytes_all,
                continuation_all, forced_other, length_two, length_three, length_four, adv, &carry_to_edge, sz_true_k);
            carry = carry_to_edge;
            position += adv;
        }
        else {
            // The whole complete region resolved (adv == complete_limit), or the clamp sits at lane 0 (adv == 0, a
            // Mid* shadow longer than one window): step past the complete span with guaranteed progress, keeping the
            // window-end run-state (`carry_full`) so any open shadow stays exact.
            carry = carry_full;
            position += complete_limit ? complete_limit : loaded;
        }
    }

    // The trailing (still-open) word `[word_start, length)` finalizes the output (end of text is a boundary).
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
