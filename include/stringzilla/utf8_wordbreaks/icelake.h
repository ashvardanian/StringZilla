/**
 *  @file include/stringzilla/utf8_wordbreaks/icelake.h
 *  @author Ash Vardanian
 *  @brief  Fully-vectorized UAX-29 Word_Break segmentation for AVX-512 (Ice Lake and later). No path scalar-walks
 *          codepoints, spills a ZMM to the stack, or calls the serial oracle.
 *
 *  The text is consumed in 64-byte windows. Each window is decoded by the shared codepoint substrate and classified
 *  to a per-codepoint Word_Break property (ALetter, Numeric, MidLetter, Extend, Regional_Indicator, ...): an ASCII
 *  permute, big arithmetic ranges, a codepoint < 0x800 page LUT, a page-compressed FLAT table for the scattered cold
 *  BMP residue, and an aligned `.rodata` astral trie for the Supplementary Plane. Rules WB3-WB16 then run as
 *  branchless 64-lane bit algebra. The driver emits the boundaries it can fully trust and advances to the last
 *  resolved codepoint start; a small register carry (@ref sz_utf8_word_break_carry_t) threads the cross-window state.
 *
 *  Design note: the cold BMP classifier is `bmp_page_lut_[cp >> 8]` selecting one of 52 distinct 256-byte pages, then
 *  `flat_bmp_[page * 256 + (cp & 0xFF)]` fetched by one `vpgatherdd` - a single indexed lookup, chosen for port
 *  pressure over instruction count: cross-lane shuffles are port-5-only, so any dependent shuffle cascade saturates
 *  that one port, while the gather issues on the load ports and leaves the shuffle port to the decode. The
 *  `cp < 0x800` gate means small-page scripts never reach it. Gathers pay off on multi-KB tables in general - see
 *  less_slow.cpp v0.3.0 "Gather and Scatter": https://github.com/ashvardanian/less_slow.cpp/releases/tag/v0.3.0
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
 *  classifier, the page LUT, the flat-table gather, the per-lead validity machinery and the partition fixpoint are
 *  all rare-class gated, so the per-window cost is proportional to the window's content, not to the size of the
 *  Unicode tables.
 */
#ifndef STRINGZILLA_UTF8_WORDBREAKS_ICELAKE_H_
#define STRINGZILLA_UTF8_WORDBREAKS_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_wordbreaks/tables.h"
#include "stringzilla/utf8_wordbreaks/serial.h"
#include "stringzilla/utf8_runes/icelake.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_ICELAKE
#if defined(__clang__) && SZ_CLANG_HAS_EVEX512_
#pragma clang attribute push(                                                                                         \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,evex512,popcnt"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(                                                                                 \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,popcnt"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2", \
                   "popcnt")
#endif

#pragma region Word_Break Classifier

/** @brief  64-byte ZMM tile counts spanning the flattened ASTRAL stage tables (the BMP resolves through the cheap
 *          arithmetic ranges, the 2-byte page LUT, and the flat-table gather). */
enum {
    sz_utf8_word_break_astral_stage1_tiles_k = sizeof(sz_utf8_word_break_astral_s1_) / 64,
    sz_utf8_word_break_astral_stage2_tiles_k = sizeof(sz_utf8_word_break_astral_s2_) / 64,
    sz_utf8_word_break_astral_leaf_packed_tiles_k = sizeof(sz_utf8_word_break_astral_leaf_packed_) / 64,
};

/** @brief  Classify 16 astral codepoints (u32 lanes) into 16 Word_Break classes via the 4-stage astral trie:
 *          `astral_s0` → stage1 → stage2 → leaf, addressed by the 8/4/4/4 split of `offset = codepoint - 0x10000`,
 *          every tile read straight from aligned `.rodata` (re-init-free — no per-call `luts`). Byte-identical to
 *          `sz_rune_word_break_property` for astral, replacing the 476-range linear fold. */
SZ_HELPER_INLINE __m512i sz_utf8_word_break_classify_astral16_icelake_(__m512i codepoints) {
    __m512i const offset = _mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x10000));
    __m512i const stage1 = sz_utf8_rune_permute256_icelake_(
        sz_utf8_word_break_astral_s0_, _mm512_and_si512(_mm512_srli_epi32(offset, 12), _mm512_set1_epi32(0xFF)));
    __m512i const stage2_index = _mm512_add_epi32(
        _mm512_slli_epi32(stage1, 4), _mm512_and_si512(_mm512_srli_epi32(offset, 8), _mm512_set1_epi32(0xF)));
    __m512i const stage2 = sz_utf8_rune_lut_cascade_icelake_(sz_utf8_word_break_astral_s1_,
                                                             sz_utf8_word_break_astral_stage1_tiles_k, stage2_index);
    __m512i const leaf_index = _mm512_add_epi32(_mm512_slli_epi32(stage2, 4),
                                                _mm512_and_si512(_mm512_srli_epi32(offset, 4), _mm512_set1_epi32(0xF)));
    __m512i const leaf = sz_utf8_rune_lut_cascade_icelake_(sz_utf8_word_break_astral_s2_,
                                                           sz_utf8_word_break_astral_stage2_tiles_k, leaf_index);
    __m512i const class_index = _mm512_add_epi32(_mm512_slli_epi32(leaf, 4),
                                                 _mm512_and_si512(offset, _mm512_set1_epi32(0xF)));
    // The Word_Break class fits in 4 bits, so the leaf is stored two cells per byte (half the tiles); the nibble
    // cascade halves this stage's port-5 cost on the astral / emoji path.
    return sz_utf8_rune_lut_cascade_nibble_icelake_(sz_utf8_word_break_astral_leaf_packed_,
                                                    sz_utf8_word_break_astral_leaf_packed_tiles_k, class_index);
}

/** @brief  AVX-512 classification of an all-ASCII 64-byte vector to WB properties via table lookup. */
SZ_HELPER_INLINE __m512i sz_utf8_word_break_classify_ascii_icelake_(__m512i ascii_bytes) {
    __m512i const low_table = _mm512_loadu_epi8(sz_utf8_word_break_property_ascii_);
    __m512i const high_table = _mm512_loadu_epi8(sz_utf8_word_break_property_ascii_ + 64);
    __mmask64 const high_half = _mm512_test_epi8_mask(ascii_bytes, _mm512_set1_epi8(0x40));
    __m512i const low_result = _mm512_permutexvar_epi8(ascii_bytes, low_table);
    __m512i const high_result = _mm512_permutexvar_epi8(ascii_bytes, high_table);
    return _mm512_mask_blend_epi8(high_half, low_result, high_result);
}

/** @brief  64-lane mask of bytes whose class equals @p value. */
SZ_HELPER_INLINE sz_u64_t sz_utf8_word_break_class_mask_icelake_(__m512i classes, sz_u8_t value) {
    return _cvtmask64_u64(_mm512_cmpeq_epi8_mask(classes, _mm512_set1_epi8((char)value)));
}

/** @brief  64-lane mask of lanes whose `(high, low)` 16-bit value lies inside any sorted `[lo, hi]` range (WSegSpace
 *          WB3d and Extended_Pictographic WB3c, which are NOT part of the 4-bit Word_Break model). */
SZ_HELPER_AUTO sz_u64_t sz_utf8_word_break_range16_mask_icelake_( //
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
 *          network (cheaper than the flat gather for the dense 2-byte scripts). */
SZ_HELPER_AUTO __m512i sz_utf8_word_break_small_page_icelake_(__m512i high, __m512i low) {
    __m512i const in_seven = _mm512_and_si512(low, _mm512_set1_epi8(0x7F));
    __m512i const low_high_bit = sz_utf8_srl8_icelake_(low, 7, 0x01);
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
 *          lanes. Bit-exact with `sz_rune_word_break_property` over the astral planes. */
SZ_HELPER_AUTO __m512i sz_utf8_word_break_classify_four_byte_icelake_(__m512i window, __m512i next1, __m512i next2,
                                                                      __m512i next3) {
    __m512i const byte0 = _mm512_and_si512(window, _mm512_set1_epi8(0x07));
    __m512i const byte1 = _mm512_and_si512(next1, _mm512_set1_epi8(0x3F));
    __m512i const byte2 = _mm512_and_si512(next2, _mm512_set1_epi8(0x3F));
    __m512i const byte3 = _mm512_and_si512(next3, _mm512_set1_epi8(0x3F));
    __m512i const lane_identity = sz_utf8_lane_identity_icelake_();
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

/** @brief  Start-compacting flat-lookup classify for the COLD `0x800..0xFFFF` residue (Devanagari, Bengali, Thai,
 *          Tamil, ...). The rule engine reads classes only at codepoint-START lanes, and every cold lane is a 3-byte
 *          lead, so a 64-byte window holds at most 21 cold starts. Their `high`/`low` bytes are `vpcompressb`-
 *          compacted into the low lanes, widened to full 32-bit codepoints in up to two 16-lane registers, resolved
 *          by the shared @ref sz_utf8_rune_flat_lookup_icelake_ (page LUT `vpermb` + one `vpgatherdd` each), then
 *          `vpexpandb`-scattered back onto @p classes at their original byte-lane positions. The second half only
 *          runs when more than sixteen cold starts are present. Cold continuation lanes are don't-cares (`decide`
 *          reads only start lanes) and keep their prior value. */
SZ_HELPER_AUTO __m512i sz_utf8_word_break_cold_compact_icelake_( //
    __m512i classes_u8x64, __m512i high_bytes_u8x64, __m512i low_bytes_u8x64, sz_u64_t cold_starts) {
    __mmask64 const cold_start_mask = _cvtu64_mask64(cold_starts);
    __m512i const high_packed_u8x64 = _mm512_maskz_compress_epi8(cold_start_mask, high_bytes_u8x64);
    __m512i const low_packed_u8x64 = _mm512_maskz_compress_epi8(cold_start_mask, low_bytes_u8x64);

    //  Unused (zeroed) compacted lanes decode to codepoint 0, a safe in-bounds flat index whose class is discarded.
    __m512i const codepoints_first_u32x16 = _mm512_or_si512(
        _mm512_slli_epi32(_mm512_cvtepu8_epi32(_mm512_castsi512_si128(high_packed_u8x64)), 8),
        _mm512_cvtepu8_epi32(_mm512_castsi512_si128(low_packed_u8x64)));
    __m128i const classes_first_u8x16 = _mm512_cvtepi32_epi8(sz_utf8_rune_flat_lookup_icelake_(
        sz_utf8_word_break_bmp_page_lut_, sz_utf8_word_break_flat_bmp_, codepoints_first_u32x16));
    __m512i classes_packed_u8x64 = _mm512_castsi128_si512(classes_first_u8x16);

    if (_mm_popcnt_u64(cold_starts) > 16) {
        __m512i const codepoints_second_u32x16 = _mm512_or_si512(
            _mm512_slli_epi32(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(high_packed_u8x64, 1)), 8),
            _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(low_packed_u8x64, 1)));
        __m128i const classes_second_u8x16 = _mm512_cvtepi32_epi8(sz_utf8_rune_flat_lookup_icelake_(
            sz_utf8_word_break_bmp_page_lut_, sz_utf8_word_break_flat_bmp_, codepoints_second_u32x16));
        classes_packed_u8x64 = _mm512_inserti32x4(classes_packed_u8x64, classes_second_u8x16, 1);
    }
    return _mm512_mask_expand_epi8(classes_u8x64, cold_start_mask, classes_packed_u8x64);
}

/**
 *  @brief  Classify a 64-byte window to per-lane Word_Break properties. ASCII through the ASCII permute, BMP through
 *          arithmetic big ranges (Latin / Hangul / CJK) + the codepoint < 0x800 page LUT + the start-compacted flat
 *          table for the residue; 4-byte leads through the aligned `.rodata` astral trie. All cheap paths are
 *          rare-class gated.
 */
SZ_HELPER_AUTO __m512i sz_utf8_word_break_classify_window_icelake_( //
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
    if (cold) {
        __mmask64 const continuation = _mm512_cmpeq_epi8_mask(_mm512_and_si512(window, _mm512_set1_epi8((char)0xC0)),
                                                              _mm512_set1_epi8((char)0x80));
        classes = sz_utf8_word_break_cold_compact_icelake_(classes, high, low, _cvtmask64_u64(cold & ~continuation));
    }
    return classes;
}

#pragma endregion Word_Break Classifier

#pragma region Word_Break Codepoint Partition

/**
 *  @brief  Resolve one window into the canonical maximal-subpart codepoint partition (the serial reference's model:
 *          ill-formed bytes become 1-byte U+FFFD): compute the per-ISA `sz_u64_t` masks (real continuation, high-nibble
 *          declared lengths, bad-second-byte overlong/surrogate/range) and delegate the partition algebra to the
 *          portable @ref sz_utf8_word_break_partition_from_masks_. @p at_end_of_text distinguishes a benign interior
 *          straddle (the next window completes it) from a true end-of-text truncation.
 */
SZ_HELPER_AUTO sz_utf8_word_break_partition_t sz_utf8_word_break_partition_icelake_( //
    __m512i window, __m512i next1, sz_u64_t valid, int at_end_of_text) {
    sz_u64_t const real_continuation = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(
                                           _mm512_and_si512(window, _mm512_set1_epi8((char)0xC0)),
                                           _mm512_set1_epi8((char)0x80))) &
                                       valid;
    // Declared length keyed purely on the HIGH NIBBLE (serial `codepoint_length_`): 0xC/0xD → 2, 0xE → 3, 0xF → 4.
    __m512i const high_nibble = sz_utf8_srl8_icelake_(window, 4, 0x0F);
    sz_u64_t const length_two = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(high_nibble, _mm512_set1_epi8(0x0C)) |
                                               _mm512_cmpeq_epi8_mask(high_nibble, _mm512_set1_epi8(0x0D))) &
                                valid;
    sz_u64_t const length_three = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(high_nibble, _mm512_set1_epi8(0x0E))) & valid;
    sz_u64_t const length_four = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(high_nibble, _mm512_set1_epi8(0x0F))) & valid;
    sz_u64_t const length_ge_two = length_two | length_three | length_four;
    // Overlong / surrogate / out-of-range leads (rare); detect them only when a multi-byte lead exists.
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
    return sz_utf8_word_break_partition_from_masks_(real_continuation, length_two, length_three, length_four,
                                                    bad_second_byte, valid, at_end_of_text);
}

#pragma endregion Word_Break Codepoint Partition

#pragma region Word_Break Boundary Algebra

/**
 *  @brief  Per-ISA extractor: lower one classified 64-byte window to the portable @ref sz_utf8_word_break_frame_t.
 *          Applies the truncated-edge U+FFFD reclassify to the class vector, then materializes every per-class lane
 *          mask, the raw-byte WSegSpace / Extended_Pictographic / Single_Quote / Double_Quote membership masks, and
 *          the per-lane class byte array the carry edge reads.
 */
SZ_HELPER_INLINE sz_utf8_word_break_frame_t sz_utf8_word_break_build_frame_icelake_( //
    __m512i window, __m512i high, __m512i low, __mmask64 four_byte_starts, __m512i next1, __m512i next2, __m512i next3,
    __m512i classes, sz_u64_t start_bytes_all, sz_u64_t length_two, sz_u64_t length_three, sz_u64_t length_four,
    sz_size_t loaded, int want_pictographic) {

    sz_u64_t const valid = sz_u64_mask_until_(loaded);
    sz_u64_t const start_bytes = start_bytes_all & valid;

    // Strict U+FFFD substitution for an ill-formed multi-byte lead truncated by the buffer end: when a lead's declared
    // byte span runs past `loaded`, the serial reference decodes it blind as a 1-byte U+FFFD (Other). Reclassify those
    // starts to Other in-register BEFORE the per-class masks are extracted, so the portable engine sees them as Other.
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

    sz_utf8_word_break_frame_t frame;
    frame.class_aletter = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_aletter_k);
    frame.class_hebrew = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_hebrew_letter_k);
    frame.class_numeric = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_numeric_k);
    frame.class_katakana = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_katakana_k);
    frame.class_extendnumlet = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_extendnumlet_k);
    frame.class_extend = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_extend_k);
    frame.class_zwj = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_zwj_k);
    frame.class_format = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_format_k);
    frame.class_midletter = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_midletter_k);
    frame.class_midnum = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_midnum_k);
    frame.class_mid_quotes = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_mid_quotes_k);
    frame.class_cr = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_cr_k);
    frame.class_lf = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_lf_k);
    frame.class_newline = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_newline_k);
    frame.class_regional = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_regional_ind_k);

    sz_u64_t const non_ascii_lanes = _cvtmask64_u64(_mm512_movepi8_mask(window)) & valid;
    frame.non_ascii_lanes = non_ascii_lanes;
    frame.double_quote_byte = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8(0x22))) & valid;
    frame.single_quote_byte = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8(0x27))) & valid;

    // WB3d WSegSpace raw membership (the ASCII U+0020 byte compare OR the multibyte (high,low) range scan).
    sz_u64_t wseg_multibyte = 0ull;
    if (non_ascii_lanes)
        wseg_multibyte = sz_utf8_word_break_range16_mask_icelake_(high, low, sz_utf8_word_break_wseg_lo_,
                                                                  sz_utf8_word_break_wseg_hi_,
                                                                  sz_utf8_word_break_wseg_count_k) &
                         non_ascii_lanes;
    frame.wseg = (wseg_multibyte | (_cvtmask64_u64(_mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8(0x20))) & valid));

    // WB3c Extended_Pictographic raw membership (BMP range scan on non-4-byte lanes, SMP range scan on plane-one
    // 4-byte lanes). Rare-class gated: the engine reads `pictographic` only when a ZWJ is present in-window or carried,
    // so the driver passes `want_pictographic` (an in-window ZWJ or the carried `prev_ends_in_zwj`) and the ~156-range
    // scan is skipped entirely on the overwhelmingly common no-ZWJ window. The engine applies the final gating.
    frame.pictographic = 0ull;
    if (want_pictographic) {
        sz_u64_t const four_byte = _cvtmask64_u64(four_byte_starts) & valid;
        __m512i const plane = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(window, _mm512_set1_epi8(0x07)), 2),
                                              sz_utf8_srl8_icelake_(next1, 4, 0x03));
        sz_u64_t const plane_one = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(plane, _mm512_set1_epi8(0x01)));
        sz_u64_t const pictographic_bmp = sz_utf8_word_break_range16_mask_icelake_( //
            high, low, sz_utf8_word_break_pict_bmp_lo_, sz_utf8_word_break_pict_bmp_hi_,
            sz_utf8_word_break_pict_bmp_count_k);
        __m512i const smp_mid = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(next1, _mm512_set1_epi8(0x0F)), 4),
                                                sz_utf8_srl8_icelake_(next2, 2, 0x0F));
        __m512i const smp_low = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(next2, _mm512_set1_epi8(0x03)), 6),
                                                _mm512_and_si512(next3, _mm512_set1_epi8(0x3F)));
        sz_u64_t const pictographic_smp = sz_utf8_word_break_range16_mask_icelake_( //
            smp_mid, smp_low, sz_utf8_word_break_pict_smp_lo_, sz_utf8_word_break_pict_smp_hi_,
            sz_utf8_word_break_pict_smp_count_k);
        frame.pictographic = (pictographic_bmp & non_ascii_lanes & ~four_byte) |
                             (pictographic_smp & four_byte & plane_one);
    }

    _mm512_storeu_si512(frame.classes_byte, classes);
    return frame;
}

/**
 *  @brief  Byte-level UAX-29 rule engine, Ice Lake entry: extract the portable frame in-register, then delegate every
 *          WB1-WB16 decision to the portable @ref sz_utf8_word_break_decide_window_. Bit-exact with the serial reference.
 */
SZ_HELPER_INLINE sz_utf8_word_break_window_t sz_utf8_word_break_block_breaks_icelake_( //
    __m512i window, __m512i high, __m512i low, __mmask64 four_byte_starts, __m512i next1, __m512i next2, __m512i next3,
    __m512i classes, sz_u64_t start_bytes_all, sz_u64_t continuation_all, sz_u64_t forced_other, sz_u64_t length_two,
    sz_u64_t length_three, sz_u64_t length_four, sz_size_t loaded, sz_utf8_word_break_carry_t *carry,
    sz_bool_t more_text) {
    int const want_pictographic = sz_utf8_word_break_class_mask_icelake_(classes, sz_utf8_word_break_zwj_k) != 0 ||
                                  carry->prev_ends_in_zwj;
    sz_utf8_word_break_frame_t const frame = sz_utf8_word_break_build_frame_icelake_(
        window, high, low, four_byte_starts, next1, next2, next3, classes, start_bytes_all, length_two, length_three,
        length_four, loaded, want_pictographic);
    return sz_utf8_word_break_decide_window_(&frame, start_bytes_all, continuation_all, forced_other, length_two,
                                             length_three, length_four, loaded, carry, more_text);
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
SZ_API_COMPTIME sz_size_t sz_utf8_wordbreaks_icelake( //
    sz_cptr_t text, sz_size_t length,                 //
    sz_size_t *word_starts, sz_size_t *word_lengths,  //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    __m512i const lane_identity = sz_utf8_lane_identity_icelake_();

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
        sz_utf8_rune_window_t const decoded = sz_utf8_rune_decode_window_icelake_(text_u8 + position, length - position,
                                                                                  lane_identity);
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
                sz_size_t const last_lead_length = sz_utf8_lead_length_(text_u8[position + last_lead]);
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

        words = sz_utf8_rune_drain_forward_(boundary_lanes, position, lane_identity, word_starts, word_lengths, words,
                                            words_capacity, &word_start);
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

#endif // STRINGZILLA_UTF8_WORDBREAKS_ICELAKE_H_
