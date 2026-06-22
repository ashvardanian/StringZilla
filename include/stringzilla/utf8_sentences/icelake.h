/**
 *  @brief Ice Lake backend for UAX-29 sentence boundaries.
 *  @file include/stringzilla/utf8_sentences/icelake.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_SENTENCES_ICELAKE_H_
#define STRINGZILLA_UTF8_SENTENCES_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_sentences/tables.h"
#include "stringzilla/utf8_sentences/serial.h"
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

#pragma region Sentence_Break gather free classifier

/**
 *  @brief  Classify up to 64 codepoints (held one-per-lane in the decoded @p high / @p low byte halves) into
 *          per-lane Sentence_Break properties, fully in-register and gather-free. The dominant `cp < 0x800`
 *          region (Latin/Greek/Cyrillic/Arabic/Hebrew) is resolved by an in-register `vpermi2b` page network
 *          over `sz_utf8_sentence_break_flat_lut_0800_`; the big homogeneous OLetter blocks (CJK/Hangul/...) by
 *          arithmetic range compares (zero data); the 3-byte-BMP residue by the shared two-stage trie substrate
 *          (`sz_utf8_codepoints_trie_walk_icelake_`) over all 64 lanes at once. No scalar per-lane loop, no
 *          stack round-trip, no `vpgather`. Astral (4-byte) lanes are reconstructed to full 21-bit codepoints in
 *          register (four 16-lane chunks) and resolved through the canonical sorted astral range list with
 *          arithmetic compares - still gather-free, still all 64 lanes uniformly.
 *
 *  @param  raw_window         The raw 64 input bytes (codepoint lead/continuation bytes, one lane each here).
 *  @param  raw_next1          Byte at lane+1 (first continuation), @p raw_next2 lane+2, @p raw_next3 lane+3.
 *  @param  high               Per-lane high byte of the reconstructed codepoint (`cp >> 8`, BMP) from the driver.
 *  @param  low                Per-lane low byte of the reconstructed codepoint (`cp & 0xFF`) from the driver.
 *  @param  four_byte_starts   Lanes that begin a 4-byte UTF-8 sequence (gates the >= 0x10000 astral test).
 *  @param  codepoint_starts   Lanes that begin any codepoint (non-continuation, in range).
 */
SZ_INTERNAL __m512i sz_utf8_sentence_break_classify_window_icelake_(             //
    __m512i raw_window, __m512i raw_next1, __m512i raw_next2, __m512i raw_next3, //
    __m512i high, __m512i low,                                                   //
    __mmask64 four_byte_starts, __mmask64 codepoint_starts) {

    // ASCII / 2-byte lanes have `high < 0x08`, so the low 11 bits address the 2048-byte page LUT directly. The
    // index is `high[2:0]:low[7:0]`; `vpermi2b` selects within a 128-byte segment-pair, masked moves the pair.
    __m512i const in_seven = _mm512_and_si512(low, _mm512_set1_epi8(0x7F));
    __m512i const low_high_bit = sz_utf8_codepoints_srl8_(low, 7, 0x01);
    __m512i const page = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(high, _mm512_set1_epi8(0x07)), 1),
                                         low_high_bit);
    __m512i small_class = _mm512_setzero_si512();
    for (int pair = 0; pair < 16; ++pair) {
        __m512i const seg_lo = _mm512_loadu_si512(sz_utf8_sentence_break_flat_lut_0800_ + (2 * pair) * 64);
        __m512i const seg_hi = _mm512_loadu_si512(sz_utf8_sentence_break_flat_lut_0800_ + (2 * pair + 1) * 64);
        __m512i const permuted = _mm512_permutex2var_epi8(seg_lo, in_seven, seg_hi);
        __mmask64 const hit = _mm512_cmpeq_epi8_mask(page, _mm512_set1_epi8((char)pair));
        small_class = _mm512_mask_mov_epi8(small_class, hit, permuted);
    }

    // 3-byte BMP residue: the shared two-stage trie classifies all 64 lanes at once (no scalar cold loop).
    __m512i const trie_class = sz_utf8_codepoints_trie_walk_icelake_(
        high, low, sz_utf8_sentence_break_trie_l1_, sz_utf8_sentence_break_trie_l2_, sz_utf8_sentence_break_trie_leaf_,
        sz_utf8_sentence_break_trie_block_k, sz_utf8_sentence_break_trie_subblock_k, 496, 1376, 2248);

    // Big homogeneous OLetter ranges as arithmetic compares (zero data) over the per-lane (high<<8|low).
    __mmask64 oletter = 0;
    for (int range = 0; range < sz_utf8_sentence_break_big_oletter_count_k; ++range) {
        sz_u32_t const lo = sz_utf8_sentence_break_big_oletter_lo_[range];
        sz_u32_t const hi = sz_utf8_sentence_break_big_oletter_hi_[range];
        if (lo >= 0x10000u) continue; // astral OLetter handled by the astral sweep below
        __m512i const lo_hi = _mm512_set1_epi8((char)(sz_u8_t)(lo >> 8));
        __m512i const lo_lo = _mm512_set1_epi8((char)(sz_u8_t)lo);
        __m512i const hi_hi = _mm512_set1_epi8((char)(sz_u8_t)(hi >> 8));
        __m512i const hi_lo = _mm512_set1_epi8((char)(sz_u8_t)hi);
        __mmask64 const ge = _mm512_cmpgt_epu8_mask(high, lo_hi) |
                             (_mm512_cmpeq_epi8_mask(high, lo_hi) & _mm512_cmp_epu8_mask(low, lo_lo, _MM_CMPINT_NLT));
        __mmask64 const le = _mm512_cmplt_epu8_mask(high, hi_hi) |
                             (_mm512_cmpeq_epi8_mask(high, hi_hi) & _mm512_cmp_epu8_mask(low, hi_lo, _MM_CMPINT_LE));
        oletter |= ge & le;
    }
    // Dispatch by codepoint VALUE, not by byte-length, so overlong / malformed sequences classify exactly like the
    // serial reference: an n-byte lead whose blind value lands in a shorter range is resolved by that range's table.
    // `is_astral` selects 4-byte leads whose value is truly >= 0x10000 (cp bit 16+ set, from `(b0 & 7) | (b1 & 0x30)`);
    // every other lead - including a 4-byte lead decoding to a BMP value or a 3-byte lead decoding below 0x800 - is a
    // BMP lane routed to the page LUT (high < 0x08) or the trie (high >= 0x08). For well-formed UTF-8 the value split
    // is identical to the byte-length split, so this leaves conformance untouched.
    __mmask64 const is_astral = four_byte_starts & (_mm512_test_epi8_mask(raw_window, _mm512_set1_epi8(0x07)) |
                                                    _mm512_test_epi8_mask(raw_next1, _mm512_set1_epi8(0x30)));
    __mmask64 const bmp_starts = codepoint_starts & ~is_astral;
    __mmask64 const small_lanes = bmp_starts & _mm512_cmplt_epu8_mask(high, _mm512_set1_epi8(0x08));
    __mmask64 const trie_lanes = bmp_starts & _mm512_cmp_epu8_mask(high, _mm512_set1_epi8(0x08), _MM_CMPINT_NLT);
    oletter &= bmp_starts;

    // Layer the resolutions: default Other, BMP-value lanes from the page LUT (<0x800) or trie (0x800..0xFFFF), then
    // the big OLetter blocks; the astral sweep below overwrites the true >= 0x10000 lanes.
    __m512i classes = _mm512_set1_epi8((char)sz_sentence_break_other_k);
    classes = _mm512_mask_mov_epi8(classes, small_lanes, small_class);
    classes = _mm512_mask_mov_epi8(classes, trie_lanes, trie_class);
    classes = _mm512_mask_mov_epi8(classes, oletter, _mm512_set1_epi8((char)sz_sentence_break_oletter_k));

    // Astral (4-byte) lanes: reconstruct the full 21-bit codepoint per lane and resolve through the sorted astral
    // range list with arithmetic compares (no gather). cp = ((b0&7)<<18)|((b1&0x3F)<<12)|((b2&0x3F)<<6)|(b3&0x3F).
    // The 21-bit value exceeds a byte lane, so we widen to 32-bit lanes in four 16-lane chunks and compare each
    // chunk against every astral range, blending the matching class back. The whole block only fires when a
    // 4-byte lead is present (corpus astral residue < 0.1%), keeping the common path branch-free.
    if (is_astral) {
        __m512i const b0 = _mm512_and_si512(raw_window, _mm512_set1_epi8(0x07));
        __m512i const b1 = _mm512_and_si512(raw_next1, _mm512_set1_epi8(0x3F));
        __m512i const b2 = _mm512_and_si512(raw_next2, _mm512_set1_epi8(0x3F));
        __m512i const b3 = _mm512_and_si512(raw_next3, _mm512_set1_epi8(0x3F));
        __m512i const lane_identity = sz_utf8_codepoints_lane_identity_icelake_();
        for (int chunk = 0; chunk < 4; ++chunk) {
            // Bring this chunk's 16 bytes to the low 16 lanes via a runtime permute, then widen to 32-bit lanes.
            __m512i const select = _mm512_add_epi8(lane_identity, _mm512_set1_epi8((char)(chunk * 16)));
            __m512i const w0 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_permutexvar_epi8(select, b0)));
            __m512i const w1 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_permutexvar_epi8(select, b1)));
            __m512i const w2 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_permutexvar_epi8(select, b2)));
            __m512i const w3 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_permutexvar_epi8(select, b3)));
            __m512i const cp = _mm512_or_si512(_mm512_or_si512(_mm512_slli_epi32(w0, 18), _mm512_slli_epi32(w1, 12)),
                                               _mm512_or_si512(_mm512_slli_epi32(w2, 6), w3));
            __mmask16 const lane_is_four = (__mmask16)((is_astral >> (chunk * 16)) & 0xFFFFu);
            __m512i astral_class = _mm512_setzero_si512();
            __mmask16 matched = 0;
            // Big homogeneous OLetter blocks that live above the BMP (CJK Extension B+, astral Hangul/Kana, ...) are
            // skipped by the BMP OLetter loop above and are NOT duplicated in the astral range list; the serial
            // reference resolves them from `big_oletter` FIRST, so check them here before the astral list with the
            // same first-match-wins precedence.
            for (int range = 0; range < sz_utf8_sentence_break_big_oletter_count_k; ++range) {
                sz_u32_t const lo = sz_utf8_sentence_break_big_oletter_lo_[range];
                sz_u32_t const hi = sz_utf8_sentence_break_big_oletter_hi_[range];
                if (lo < 0x10000u) continue; // BMP OLetter blocks are resolved by the page LUT / trie path
                __mmask16 const in_range = _mm512_cmpge_epu32_mask(cp, _mm512_set1_epi32((int)lo)) &
                                           _mm512_cmple_epu32_mask(cp, _mm512_set1_epi32((int)hi)) & lane_is_four &
                                           ~matched;
                astral_class = _mm512_mask_mov_epi32(astral_class, in_range,
                                                     _mm512_set1_epi32((int)sz_sentence_break_oletter_k));
                matched |= in_range;
            }
            for (int range = 0; range < sz_utf8_sentence_break_astral_count_k; ++range) {
                sz_u32_t const lo = sz_utf8_sentence_break_astral_lo_[range];
                sz_u32_t const hi = sz_utf8_sentence_break_astral_hi_[range];
                if (hi < 0x10000u) continue; // BMP ranges are already resolved by the page LUT / trie / OLetter
                __mmask16 const in_range = _mm512_cmpge_epu32_mask(cp, _mm512_set1_epi32((int)lo)) &
                                           _mm512_cmple_epu32_mask(cp, _mm512_set1_epi32((int)hi)) & lane_is_four &
                                           ~matched;
                astral_class = _mm512_mask_mov_epi32(astral_class, in_range,
                                                     _mm512_set1_epi32((int)sz_utf8_sentence_break_astral_cls_[range]));
                matched |= in_range;
            }
            // Narrow the 16 32-bit astral classes to 16 bytes (`vpmovdb`), broadcast them back to this chunk's byte
            // lanes via `vpexpandb`, and blend - fully vectorized, no scalar per-lane writeback.
            __m128i const astral_bytes = _mm512_cvtepi32_epi8(astral_class);
            __m512i const astral_broadcast = _mm512_maskz_expand_epi8(_cvtu64_mask64(0xFFFFull << (chunk * 16)),
                                                                      _mm512_castsi128_si512(astral_bytes));
            classes = _mm512_mask_mov_epi8(classes, _cvtu64_mask64((sz_u64_t)matched << (chunk * 16)),
                                           astral_broadcast);
        }
    }
    return classes;
}

#pragma endregion Sentence_Break gather free classifier

#pragma region Sentence_Break boundary algebra

/** @brief  Membership mask of lanes whose class byte equals @p value, restricted to @p valid lanes. */
SZ_INTERNAL sz_u64_t sz_utf8_sentence_break_class_mask_(__m512i classes, sz_u8_t value, sz_u64_t valid) {
    return _cvtmask64_u64(_mm512_cmpeq_epi8_mask(classes, _mm512_set1_epi8((char)value))) & valid;
}

/** @brief  Class byte held at lane @p lane of @p classes, read in-register via masked compress (no stack spill). */
SZ_INTERNAL sz_u8_t sz_utf8_sentence_break_class_at_(__m512i classes, int lane) {
    __m512i const isolated = _mm512_maskz_compress_epi8(_cvtu64_mask64((sz_u64_t)1 << lane), classes);
    return (sz_u8_t)_mm_cvtsi128_si32(_mm512_castsi512_si128(isolated));
}

/**
 *  @brief  Membership at each codepoint lead of "the nearest significant codepoint before it has the seeded class":
 *          flood the per-class significant leads @p class_significant rightward through @p flow (continuation bytes
 *          plus Extend/Format codepoints) and read one lane on. @p edge_region seeds the leading window-edge run
 *          with the cross-window carried class when @p edge_is_class, so lane-0 left context arrives as a register
 *          carry - no scalar re-walk, no oracle. Used for SB6 (ATerm) and the SB7 two-back seeds.
 */
SZ_INTERNAL sz_u64_t sz_utf8_sentence_break_eff_prev_(sz_u64_t class_significant, sz_u64_t flow, sz_u64_t edge_region,
                                                      int edge_is_class) {
    return (sz_utf8_codepoints_fill_right_(class_significant, flow) << 1) |
           sz_u64_or_if_(0ull, edge_region, edge_is_class);
}

/**
 *  @brief  Cross-window register carry: the open `SATerm Close* Sp*` shadow run-state, the trailing significant /
 *          raw classes the bounded-lookback rules (SB3/SB4/SB6/SB7) need at the next block head, plus the SB8
 *          "neutral chain awaiting a Lower verdict" pending state that threads an unbounded right context across
 *          windows without any serial re-walk or oracle call.
 */
typedef struct sz_utf8_sentence_break_carry_t {
    sz_u8_t in_shadow;     /**< A SATerm has opened a shadow that is still open at the previous block's edge. */
    sz_u8_t shadow_aterm;  /**< The opening terminator was ATerm (matters for SB8). */
    sz_u8_t shadow_saw_sp; /**< At least one Sp seen since the terminator (SB9 vs SB10). */
    sz_u8_t have_prev;     /**< False only at the very start of the text (SB1). */
    sz_u8_t prev_eff;      /**< Class of the last significant codepoint of the previous block. */
    sz_u8_t prev_prev_eff; /**< Class of the second-last significant codepoint. */
    sz_u8_t prev_raw;      /**< Class of the last raw codepoint of the previous block (SB3/SB4 raw-before). */
} sz_utf8_sentence_break_carry_t;

/** @brief  One classified block resolved into per-byte break bits plus the metadata the driver stitches. */
typedef struct sz_utf8_sentence_break_window_t {
    sz_u64_t breaks; /**< Bit `i` set => a sentence boundary begins before the codepoint whose lead is byte-lane `i`. */
    sz_size_t resolved; /**< Exclusive upper bound, in bytes, on lanes whose break bit is fully trusted (SB8 edge). */
} sz_utf8_sentence_break_window_t;

/**
 *  @brief  Resolve a block of @p span classified codepoints (lanes `0..span-1` in @p classes, the open shadow and
 *          left context arriving in @p carry) into per-lane sentence-break bits, mirroring `sb_block_breaks`.
 *
 *  All of SB3-SB998 are pure 64-bit bit algebra over the per-class lane masks: the two-phase `SATerm Close* Sp*`
 *  shadow (`close_phase` then `shadow`), the SB8 in-window lower-ahead (`fill_left` of Lower through neutral via
 *  the shared substrate), the SB6/SB7 effective-previous chains across Extend/Format, and the SB3/SB4 raw-before.
 *  `@p carry` is updated with the trailing run-state. `breaks` bit 0 is the inter-block boundary (only meaningful
 *  when `have_prev`).
 *
 *  SB8's right context is unbounded; the in-window `fill_left` is exact unless a trailing neutral run reaches the
 *  block edge (the Lower may lie in the next block). `resolved` is clamped before such an undecided lane so the
 *  driver re-anchors the next window with full forward context - a register carry only, never a scalar re-walk or
 *  oracle call.
 */
SZ_INTERNAL sz_utf8_sentence_break_window_t sz_utf8_sentence_break_block_breaks_( //
    __m512i classes, sz_u64_t start_bytes, sz_u64_t continuation, sz_size_t loaded,
    sz_utf8_sentence_break_carry_t *carry, sz_bool_t more_text) {

    sz_u64_t const valid = sz_u64_mask_until_(loaded);
    sz_u64_t const sb = start_bytes & valid;    // codepoint-lead lanes
    sz_u64_t const cont = continuation & valid; // UTF-8 continuation-byte lanes
    sz_u64_t const m_extend = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_extend_k, sb);
    sz_u64_t const m_format = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_format_k, sb);
    sz_u64_t const m_ignorable = m_extend | m_format;
    sz_u64_t const m_cr = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_cr_k, sb);
    sz_u64_t const m_lf = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_lf_k, sb);
    sz_u64_t const m_sep = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_sep_k, sb);
    sz_u64_t const m_sp = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_sp_k, sb);
    sz_u64_t const m_lower = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_lower_k, sb);
    sz_u64_t const m_upper = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_upper_k, sb);
    sz_u64_t const m_oletter = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_oletter_k, sb);
    sz_u64_t const m_numeric = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_numeric_k, sb);
    sz_u64_t const m_aterm = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_aterm_k, sb);
    sz_u64_t const m_scont = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_scontinue_k, sb);
    sz_u64_t const m_sterm = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_sterm_k, sb);
    sz_u64_t const m_close = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_close_k, sb);
    sz_u64_t const m_parasep = m_sep | m_cr | m_lf;
    sz_u64_t const m_saterm = m_aterm | m_sterm;
    sz_u64_t const significant = sb & ~m_ignorable;

    // Adjacency in the byte domain rides a `flow` gate that hops over Extend/Format codepoints (SB5 transparency)
    // and the continuation bytes of every multi-byte codepoint, so a class flooded right then shifted one lane
    // reads the effective previous codepoint at the next lead.
    sz_u64_t const flow = cont | m_ignorable;
    sz_u8_t const prev_eff = carry->prev_eff;
    sz_u8_t const prev_prev_eff = carry->prev_prev_eff;
    sz_u8_t const prev_raw = carry->prev_raw;
    int const have_prev = carry->have_prev;

    // The leading window-edge region runs from lane 0 up to and including the first significant lead; the
    // cross-window carried class is injected here so lane-0 left context needs no scalar re-walk.
    int const first_sig = significant ? sz_u64_ctz(significant) : -1;
    sz_u64_t const edge_region = (first_sig < 0) ? sb : sz_u64_mask_until_((sz_size_t)first_sig + 1);

    // Effective-previous significant codepoint (SB6 ATerm, and the SB7 two-back seeds Upper/Lower).
    sz_u64_t const eb_aterm = sz_utf8_sentence_break_eff_prev_(m_aterm & significant, flow, edge_region,
                                                               have_prev && prev_eff == sz_sentence_break_aterm_k);
    sz_u64_t const eb_upper = sz_utf8_sentence_break_eff_prev_(m_upper & significant, flow, edge_region,
                                                               have_prev && prev_eff == sz_sentence_break_upper_k);
    sz_u64_t const eb_lower = sz_utf8_sentence_break_eff_prev_(m_lower & significant, flow, edge_region,
                                                               have_prev && prev_eff == sz_sentence_break_lower_k);

    // SB7 second-significant-back: relabel each significant lead with its own previous-significant class (`eb_X &
    // significant`), flood that one more codepoint forward, read one lane on. The first significant lead's two-back
    // is the carried `prev_prev_eff`; the carried `prev_eff` reaches the second lead automatically through the
    // `eb_X` edge seed, mirroring the serial two-back lookbehind.
    sz_u64_t const eb2_upper = (sz_utf8_codepoints_fill_right_(eb_upper & significant, flow) << 1) |
                               sz_u64_or_if_(0ull, edge_region,
                                             have_prev && prev_prev_eff == sz_sentence_break_upper_k);
    sz_u64_t const eb2_lower = (sz_utf8_codepoints_fill_right_(eb_lower & significant, flow) << 1) |
                               sz_u64_or_if_(0ull, edge_region,
                                             have_prev && prev_prev_eff == sz_sentence_break_lower_k);

    // Two-phase monotone `SATerm Close* Sp*` shadow. `flow` already spans ignorables and all continuation bytes,
    // so only the Close / Sp lead classes widen each phase's gate. The carried open-run state seeds lane 0.
    sz_u64_t const gate_close = flow | m_close;
    sz_u64_t const gate_sp = flow | m_sp;
    sz_u64_t const in_shadow_carry = (sz_u64_t)(carry->in_shadow != 0);
    sz_u64_t const saw_sp_carry = (sz_u64_t)(carry->shadow_saw_sp != 0);
    sz_u64_t const aterm_carry = (sz_u64_t)(carry->shadow_aterm != 0);
    sz_u64_t const lane0_close = (m_ignorable | m_close) & 1ull;
    sz_u64_t const lane0_sp = (m_ignorable | m_sp) & 1ull;
    sz_u64_t const lane0_sp_only = m_sp & 1ull;
    sz_u64_t const open_no_sp = in_shadow_carry & ~saw_sp_carry;  // open shadow still in its Close phase
    sz_u64_t const open_with_sp = in_shadow_carry & saw_sp_carry; // open shadow already in its Sp phase
    sz_u64_t const carry_close_seed = sz_u64_or_if_(0ull, lane0_close, (int)open_no_sp);
    sz_u64_t const carry_sp_seed = sz_u64_or_if_(sz_u64_or_if_(0ull, lane0_sp_only, (int)open_no_sp), lane0_sp,
                                                 (int)open_with_sp);
    sz_u64_t const carry_aterm_close = sz_u64_or_if_(0ull, lane0_close, (int)(open_no_sp & aterm_carry));
    sz_u64_t const carry_aterm_sp = sz_u64_or_if_(sz_u64_or_if_(0ull, lane0_sp_only, (int)(open_no_sp & aterm_carry)),
                                                  lane0_sp, (int)(open_with_sp & aterm_carry));

    sz_u64_t const close_phase = sz_utf8_codepoints_fill_right_(m_saterm | carry_close_seed, gate_close) |
                                 carry_sp_seed;
    sz_u64_t const shadow = sz_utf8_codepoints_fill_right_(close_phase | carry_sp_seed, gate_sp);
    sz_u64_t const sp_in_shadow = (m_sp & shadow) | carry_sp_seed;
    sz_u64_t const saw_sp_upto = sz_utf8_codepoints_fill_right_(sp_in_shadow, gate_sp) & shadow;
    sz_u64_t const aterm_close_phase = sz_utf8_codepoints_fill_right_(m_aterm | carry_aterm_close, gate_close) |
                                       carry_aterm_sp;
    sz_u64_t const aterm_shadow = sz_utf8_codepoints_fill_right_(aterm_close_phase | carry_aterm_sp, gate_sp);

    // SB8 in-window lower-ahead: `fill_left` of Lower through the SB8 neutral set. A multi-byte stop's continuation
    // bytes are excluded from the neutral gate so the flood cannot leak past a stop.
    sz_u64_t const sb8_stop = m_lower | m_upper | m_oletter | m_parasep | m_saterm;
    sz_u64_t const stop_cont = sz_utf8_codepoints_fill_right_(sb8_stop, cont) & cont;
    sz_u64_t const neutral = valid & ~sb8_stop & ~stop_cont;
    sz_u64_t const lower_ahead = sz_utf8_codepoints_fill_left_(m_lower, neutral);
    // An ATerm-shadow boundary is SB8-undecided if, from its lead, the neutral chain reaches the last valid byte
    // without hitting a Lower or any stop in-window: the verdict depends on the next block (effective-window<64).
    sz_u64_t const top_bit = (loaded >= 64) ? (1ull << 63) : (1ull << (loaded - 1));
    sz_u64_t const neutral_to_edge = sz_utf8_codepoints_fill_left_(top_bit & neutral, neutral) & ~lower_ahead;

    // Raw immediately-preceding codepoint for SB3/SB4 (NOT Extend/Format-skipped): flood the class right over the
    // full continuation run - its own trailing bytes plus any stray continuations folded into it - so the shift to
    // the next lead reads it regardless of gap width (serial back-scans every continuation to the lead identically).
    sz_u64_t raw_before_cr = sz_utf8_codepoints_fill_right_(m_cr, cont) << 1;
    sz_u64_t raw_before_parasep = sz_utf8_codepoints_fill_right_(m_parasep, cont) << 1;
    raw_before_cr = sz_u64_or_if_(raw_before_cr, 1ull, have_prev && prev_raw == sz_sentence_break_cr_k);
    raw_before_parasep = sz_u64_or_if_(
        raw_before_parasep, 1ull,
        have_prev && (prev_raw == sz_sentence_break_sep_k || prev_raw == sz_sentence_break_cr_k ||
                      prev_raw == sz_sentence_break_lf_k));

    sz_u64_t const r_sb3 = raw_before_cr & m_lf;
    sz_u64_t const r_sb4 = raw_before_parasep & sb;
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
    sz_u64_t const r_sb11 = in_shadow_before & sb;

    // Serial precedence: the first matching rule wins. SB3 nobreak, SB4 break, SB5 nobreak, SB6/SB7 nobreak,
    // SB8/8a/9/10 nobreak, SB11 break, SB998 nobreak. `decided` accumulates settled lanes; `brk` the breaks.
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
    brk |= r_sb11 & ~decided, decided |= r_sb11 & ~decided;

    sz_u64_t const lowbit = have_prev ? 1ull : 0ull;
    sz_u64_t const produced = sb & (~1ull | lowbit); // boundaries only at codepoint leads; clear SOT unless carried
    sz_u64_t breaks = brk & produced;

    // Trust the window only up to the first SB8-undecided ATerm-shadow boundary; the next, fully-contextual window
    // re-resolves it (effective-window<64 + register carry; no oracle, no scalar). At true end-of-text there is no
    // next block, so a trailing neutral run is decided (no Lower can follow): never clamp then.
    sz_u64_t const undecided = more_text ? (aterm_shadow_before & in_shadow_before & neutral_to_edge & produced) : 0ull;
    sz_size_t resolved = loaded;
    if (undecided) {
        sz_size_t const lane = sz_u64_ctz(undecided);
        resolved = lane;
        breaks &= sz_u64_mask_until_(lane);
    }

    // Update the carry for the next block from this window's high edge: the trailing shadow run-state at the last
    // valid byte and the last two significant / last raw codepoint classes (read in-register, no stack round-trip).
    {
        int const edge = (int)loaded - 1;
        if (significant) {
            int const last = 63 - sz_u64_clz(significant);
            carry->prev_eff = sz_utf8_sentence_break_class_at_(classes, last);
            sz_u64_t const sig2 = significant & ~(1ull << last);
            if (sig2) carry->prev_prev_eff = sz_utf8_sentence_break_class_at_(classes, 63 - sz_u64_clz(sig2));
        }
        if (sb) carry->prev_raw = sz_utf8_sentence_break_class_at_(classes, 63 - sz_u64_clz(sb));
        carry->have_prev = 1;
        carry->in_shadow = (sz_u8_t)((shadow >> edge) & 1ull);
        carry->shadow_aterm = (sz_u8_t)((aterm_shadow >> edge) & 1ull);
        carry->shadow_saw_sp = (sz_u8_t)((saw_sp_upto >> edge) & 1ull);
    }

    sz_utf8_sentence_break_window_t result;
    result.breaks = breaks;
    result.resolved = resolved;
    return result;
}

#pragma endregion Sentence_Break boundary algebra

#pragma region Sentence_Break forward driver

SZ_PUBLIC sz_size_t sz_utf8_sentences_icelake(               //
    sz_cptr_t text, sz_size_t length,                        //
    sz_size_t *sentence_starts, sz_size_t *sentence_lengths, //
    sz_size_t sentences_capacity, sz_size_t *bytes_consumed) {

    sz_size_t sentences = 0;
    if (length == 0 || sentences_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t sentence_start = 0;
    sz_size_t position = 0;
    __m512i const lane_identity = sz_utf8_codepoints_lane_identity_icelake_();

    sz_utf8_sentence_break_carry_t carry;
    carry.in_shadow = carry.shadow_aterm = carry.shadow_saw_sp = 0;
    carry.have_prev = 0;
    carry.prev_eff = carry.prev_prev_eff = carry.prev_raw = (sz_u8_t)sz_sentence_break_other_k;

    while (position < length) {
        // Decode a 64-byte window in-register (no scalar per-codepoint walk): codepoint-start / continuation /
        // lead-length masks plus the byte-domain `high`/`low` halves, all from the shared substrate helper.
        sz_utf8_codepoints_window_t const decoded = sz_utf8_codepoints_decode_window_(text_u8 + position,
                                                                                      length - position, lane_identity);
        sz_size_t const loaded = decoded.loaded;
        // Text may begin mid-codepoint: if byte 0 of the whole input is a continuation, the serial reference still
        // treats it as a lone 1-byte codepoint (blind decode), so force lane 0 to a codepoint start on the first
        // window and drop it from the continuation set. Every later window re-anchors on a true lead, so this only
        // ever promotes the very first byte.
        __mmask64 const lead_continuation = (position == 0) ? (__mmask64)1 : (__mmask64)0;
        __mmask64 const codepoint_starts = decoded.codepoint_starts | lead_continuation;
        sz_u64_t const start_bytes = _cvtmask64_u64(codepoint_starts);
        sz_u64_t const cont_mask = _cvtmask64_u64(decoded.continuation) & ~(sz_u64_t)lead_continuation;

        // Gather each lead's three forward neighbours via in-register permutes, zeroing any that fall at or past
        // `loaded`: the substrate permute wraps the window tail (lane 63's `+1` neighbour is lane 0), which would
        // mis-decode a multi-byte lead truncated at the buffer end. The serial reference 0-pads the missing bytes,
        // so the maskz keeps `high`/`low` and the astral reconstruction bit-identical there.
        __m512i const window = decoded.window;
        __mmask64 const keep1 = sz_u64_mask_until_(loaded >= 1 ? loaded - 1 : 0);
        __mmask64 const keep2 = sz_u64_mask_until_(loaded >= 2 ? loaded - 2 : 0);
        __mmask64 const keep3 = sz_u64_mask_until_(loaded >= 3 ? loaded - 3 : 0);
        __m512i const next1 = _mm512_maskz_permutexvar_epi8(keep1, _mm512_add_epi8(lane_identity, _mm512_set1_epi8(1)),
                                                            window);
        __m512i const next2 = _mm512_maskz_permutexvar_epi8(keep2, _mm512_add_epi8(lane_identity, _mm512_set1_epi8(2)),
                                                            window);
        __m512i const next3 = _mm512_maskz_permutexvar_epi8(keep3, _mm512_add_epi8(lane_identity, _mm512_set1_epi8(3)),
                                                            window);

        // Reconstruct the byte-domain `high`/`low` halves the page-LUT/trie classifier indexes from the 0-padded
        // neighbours: ASCII keeps `high = 0`, `low = raw byte`; 2/3-byte blend their folded halves; 4-byte leads keep
        // the ASCII halves (the astral path resolves them from the raw bytes instead).
        __m512i const two_high = sz_utf8_codepoints_srl8_(_mm512_and_si512(window, _mm512_set1_epi8(0x1F)), 2, 0x07);
        __m512i const two_low = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(window, _mm512_set1_epi8(0x03)), 6),
                                                _mm512_and_si512(next1, _mm512_set1_epi8(0x3F)));
        __m512i const three_high = _mm512_or_si512(
            _mm512_slli_epi16(_mm512_and_si512(window, _mm512_set1_epi8(0x0F)), 4),
            sz_utf8_codepoints_srl8_(next1, 2, 0x0F));
        __m512i const three_low = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(next1, _mm512_set1_epi8(0x03)), 6),
                                                  _mm512_and_si512(next2, _mm512_set1_epi8(0x3F)));
        // 4-byte: the LOW 16 bits of the reconstructed codepoint, so an overlong / malformed 4-byte lead whose value
        // is actually BMP classifies through the page LUT / trie by value (matching the serial reference). True
        // astral leads (value >= 0x10000) are routed past these halves to the astral sweep by the classifier.
        __m512i const four_low = _mm512_or_si512(_mm512_and_si512(next3, _mm512_set1_epi8(0x3F)),
                                                 _mm512_slli_epi16(_mm512_and_si512(next2, _mm512_set1_epi8(0x03)), 6));
        __m512i const four_high = _mm512_or_si512(
            sz_utf8_codepoints_srl8_(next2, 2, 0x0F),
            _mm512_slli_epi16(_mm512_and_si512(next1, _mm512_set1_epi8(0x0F)), 4));
        __m512i high = _mm512_setzero_si512();
        high = _mm512_mask_mov_epi8(high, decoded.two_byte_starts, two_high);
        high = _mm512_mask_mov_epi8(high, decoded.three_byte_starts, three_high);
        high = _mm512_mask_mov_epi8(high, decoded.four_byte_starts, four_high);
        __m512i low = window;
        low = _mm512_mask_mov_epi8(low, decoded.two_byte_starts, two_low);
        low = _mm512_mask_mov_epi8(low, decoded.three_byte_starts, three_low);
        low = _mm512_mask_mov_epi8(low, decoded.four_byte_starts, four_low);
        __m512i const classes = sz_utf8_sentence_break_classify_window_icelake_(
            window, next1, next2, next3, high, low, decoded.four_byte_starts, codepoint_starts);

        sz_bool_t const more_text = (position + loaded < length) ? sz_true_k : sz_false_k;

        // Effective-window<64: when more text follows, the final codepoint's blind byte span may spill past the
        // 64-byte edge - either its own declared continuation bytes are missing (a multi-byte lead at the tail) or
        // trailing (possibly stray) continuation bytes run into the next window. Either way its class / span is not
        // yet final, so trust and classify only up to the last fully-resolved codepoint (`complete_limit`) and
        // re-anchor on that lead, letting the next window decode it with its full byte run. The `last_lead > 0`
        // guard keeps progress when the whole window is a single codepoint trailed by continuations.
        sz_size_t complete_limit = loaded;
        if (more_text && start_bytes) {
            sz_u64_t const valid_lanes = sz_u64_mask_until_(loaded);
            sz_u64_t const two = _cvtmask64_u64(decoded.two_byte_starts) & valid_lanes;
            sz_u64_t const three = _cvtmask64_u64(decoded.three_byte_starts) & valid_lanes;
            sz_u64_t const four = _cvtmask64_u64(decoded.four_byte_starts) & valid_lanes;
            // A lead at lane L is incomplete when L + declared_length(L) > loaded: its final byte(s) live in the next
            // window, so a 64-byte decode reconstructs it from zero-padded tails (a 3-byte STerm one lane from the
            // edge would otherwise mis-decode to a 2-byte Extend). EVERY trailing straddling lead must be excluded,
            // not just the last one, so the carry's run-state is built only over fully-decoded codepoints.
            sz_u64_t const straddle = ((two & ~sz_u64_mask_until_(loaded > 1 ? loaded - 1 : 0)) |
                                       (three & ~sz_u64_mask_until_(loaded > 2 ? loaded - 2 : 0)) |
                                       (four & ~sz_u64_mask_until_(loaded > 3 ? loaded - 3 : 0))) &
                                      valid_lanes;
            sz_size_t limit = straddle ? (sz_size_t)sz_u64_ctz(straddle) : loaded;
            // A stray continuation at the window edge extends the last codepoint's blind span into the next window.
            if ((text_u8[position + loaded] & 0xC0) == 0x80) {
                sz_size_t const last_lead = (sz_size_t)(63 - sz_u64_clz(start_bytes));
                if (last_lead < limit) limit = last_lead;
            }
            if (limit > 0) complete_limit = limit; // `limit == 0` keeps the whole window for guaranteed progress
        }

        // Resolve over the complete-codepoint region only, so a garbage straddling lead never enters the algebra;
        // `carry_full` is therefore the exact run-state at the complete edge (no rebuild needed unless SB8 clamps).
        sz_utf8_sentence_break_carry_t carry_full = carry;
        sz_utf8_sentence_break_window_t const win = sz_utf8_sentence_break_block_breaks_(
            classes, start_bytes, cont_mask, complete_limit, &carry_full, more_text);

        // `win.resolved` is already bounded by `complete_limit`; an SB8 clamp lowers it further. Boundaries strictly
        // before it are fully decided; the boundary at the advance point is re-resolved as the next window's lane 0.
        sz_size_t const adv = win.resolved;
        sz_u64_t boundary_lanes = win.breaks & sz_u64_mask_until_(adv);
        if (!carry.have_prev) boundary_lanes &= ~1ull; // SOT is never emitted as an interior boundary

        sentences = sz_utf8_codepoints_drain_forward_(boundary_lanes, position, lane_identity, sentence_starts,
                                                      sentence_lengths, sentences, sentences_capacity, &sentence_start);
        if (sentences == sentences_capacity) {
            if (bytes_consumed) *bytes_consumed = sentence_start;
            return sentences;
        }

        if (adv == complete_limit) {
            carry = carry_full; // carry already computed at the complete edge
            position += complete_limit ? complete_limit : loaded;
        }
        else if (adv > 0) {
            // SB8 clamp strictly before the complete edge: rebuild the carry to the clamp byte so the next window's
            // left context + open-shadow run-state are exact (break bits discarded; carry side-effect only).
            sz_utf8_sentence_break_carry_t carry_to_edge = carry;
            sz_utf8_sentence_break_block_breaks_(classes, start_bytes, cont_mask, adv, &carry_to_edge, sz_true_k);
            carry = carry_to_edge;
            position += adv;
        }
        else {
            // The clamp sits at lane 0 (an open shadow / neutral run longer than one window). Step past the whole
            // complete span with guaranteed progress, keeping the window-end run-state so the open shadow stays exact.
            carry = carry_full;
            position += complete_limit ? complete_limit : loaded;
        }
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

#pragma endregion Sentence_Break forward driver
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_ICELAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_SENTENCES_ICELAKE_H_
