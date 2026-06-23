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

    // Partition the lanes FIRST so the expensive page LUT and two-stage trie only run for lanes that need them.
    // Dispatch by codepoint VALUE, not byte-length, so overlong / malformed sequences classify exactly like the
    // serial reference: an n-byte lead whose blind value lands in a shorter range is resolved by that range's table.
    // `is_astral` selects 4-byte leads whose value is truly >= 0x10000 (cp bit 16+ set, from `(b0 & 7) | (b1 & 0x30)`);
    // every other lead is a BMP lane routed to the page LUT (high < 0x08) or the trie (high >= 0x08). For well-formed
    // UTF-8 the value split is identical to the byte-length split, so this leaves conformance untouched.
    __mmask64 const is_astral = four_byte_starts & (_mm512_test_epi8_mask(raw_window, _mm512_set1_epi8(0x07)) |
                                                    _mm512_test_epi8_mask(raw_next1, _mm512_set1_epi8(0x30)));
    __mmask64 const bmp_starts = codepoint_starts & ~is_astral;
    __mmask64 const small_lanes = bmp_starts & _mm512_cmplt_epu8_mask(high, _mm512_set1_epi8(0x08));
    __mmask64 const trie_lanes = bmp_starts & _mm512_cmp_epu8_mask(high, _mm512_set1_epi8(0x08), _MM_CMPINT_NLT);

    // Big homogeneous OLetter ranges (CJK, Kana, ...) as arithmetic compares (zero data) over the per-lane
    // (high<<8|low). These resolve the vast majority of CJK / Kana without the trie, so a pure-CJK window skips the
    // trie walk entirely below.
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
    oletter &= bmp_starts;

    __m512i classes = _mm512_set1_epi8((char)sz_sentence_break_other_k);

    // Page LUT for codepoint < 0x800 (`high[2:0]:low[7:0]` via a `vpermi2b` segment-pair cascade), gated on a small
    // lane being present so a pure-3-byte (CJK) window pays nothing here.
    if (small_lanes) {
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
        classes = _mm512_mask_mov_epi8(classes, small_lanes, small_class);
    }

    // Two-stage trie for the 0x800..0xFFFF residue, gated on a 3-byte lane the OLetter ranges did NOT already resolve,
    // so CJK / Kana windows skip the ~36-`vpermi2b` trie walk (the port-5 hot spot). When every 3-byte lane is OLetter
    // the trie output would be overwritten by the OLetter overlay anyway, so skipping it is byte-identical.
    __mmask64 const trie_residual = trie_lanes & ~oletter;
    if (trie_residual) {
        __m512i const trie_class = sz_utf8_codepoints_trie_walk_icelake_(
            high, low, sz_utf8_sentence_break_trie_l1_, sz_utf8_sentence_break_trie_l2_,
            sz_utf8_sentence_break_trie_leaf_, sz_utf8_sentence_break_trie_block_k,
            sz_utf8_sentence_break_trie_subblock_k, 496, 1376, 2248);
        classes = _mm512_mask_mov_epi8(classes, trie_lanes, trie_class);
    }

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
 *  @brief  Cross-window register carry: the open `SATerm Close* Sp*` shadow run-state, the trailing significant /
 *          raw classes the bounded-lookback rules (SB3/SB4/SB6/SB7) need at the next block head, plus the SB8
 *          "neutral chain awaiting a Lower verdict" pending state that threads an unbounded right context across
 *          windows without any serial re-walk or oracle call.
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

/** @brief  One classified block resolved into per-byte break bits plus the metadata the driver stitches. */
typedef struct sz_utf8_sentence_break_window_t {
    /** Bit `i` set => a boundary begins before dense codepoint `i` (scattered to byte lanes by the driver). */
    sz_u64_t breaks;
    /** Exclusive upper bound, in dense codepoints, on lanes whose break bit is fully trusted (SB8 edge). */
    sz_size_t resolved;
    /** Verdict for an entering `carry->sb8_pending`: 0 still pending, 1 break, 2 no break. */
    sz_u8_t sb8_resolution;
} sz_utf8_sentence_break_window_t;

/**
 *  @brief  Resolve a block of @p count classified codepoints (dense lanes `0..count-1` in @p classes, the open shadow
 *          and left context arriving in @p carry) into dense per-codepoint sentence-break bits. The driver compacts
 *          the byte-lane classes to this dense stream (one `vpcompressb`) and scatters the result back with `_pdep_u64`.
 *
 *  All of SB3-SB998 are pure 64-bit bit algebra over the per-class lane masks: the two-phase `SATerm Close* Sp*`
 *  shadow (`close_phase` then `shadow`), the SB8 in-window lower-ahead (`fill_left` of Lower through neutral via
 *  the shared substrate), the SB6/SB7 effective-previous chains across Extend/Format (the `flow` gate hops only the
 *  remaining ignorable codepoints, no continuation bytes), and the SB3/SB4 raw-before (a single `<< 1` on the dense stream).
 *  `@p carry` is updated with the trailing run-state. `breaks` bit 0 is the inter-block boundary (only meaningful
 *  when `have_prev`).
 *
 *  SB8's right context is unbounded; the in-window `fill_left` is exact unless a trailing neutral run reaches the
 *  block edge (the Lower may lie in the next block). `resolved` is clamped before such an undecided lane so the
 *  driver re-anchors the next window with full forward context - a register carry only, never a scalar re-walk or
 *  oracle call. Force-inlined so the 24-byte window result stays in registers instead of spilling through a hidden
 *  `sret` pointer.
 */
SZ_FORCE_INLINE sz_utf8_sentence_break_window_t sz_utf8_sentence_break_block_breaks_( //
    __m512i classes, sz_size_t count, sz_utf8_sentence_break_carry_t *carry, sz_bool_t more_text) {

    sz_u64_t const valid = (count >= 64) ? ~0ull : ((1ull << count) - 1);
    sz_u64_t const m_extend = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_extend_k, valid);
    sz_u64_t const m_format = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_format_k, valid);
    sz_u64_t const m_ignorable = m_extend | m_format;
    sz_u64_t const m_cr = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_cr_k, valid);
    sz_u64_t const m_lf = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_lf_k, valid);
    sz_u64_t const m_sep = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_sep_k, valid);
    sz_u64_t const m_sp = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_sp_k, valid);
    sz_u64_t const m_lower = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_lower_k, valid);
    sz_u64_t const m_upper = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_upper_k, valid);
    sz_u64_t const m_oletter = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_oletter_k, valid);
    sz_u64_t const m_numeric = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_numeric_k, valid);
    sz_u64_t const m_aterm = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_aterm_k, valid);
    sz_u64_t const m_scont = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_scontinue_k, valid);
    sz_u64_t const m_sterm = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_sterm_k, valid);
    sz_u64_t const m_close = sz_utf8_sentence_break_class_mask_(classes, sz_sentence_break_close_k, valid);
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
    sz_u64_t const edge_region = (first_sig < 0) ? valid : sz_u64_mask_until_((sz_size_t)first_sig + 1);

    // Effective-previous significant codepoint (SB6 ATerm, SB7 Upper/Lower two-back seeds).
    sz_u64_t const eb_aterm = (sz_utf8_codepoints_fill_right_(m_aterm & significant, flow) << 1) |
                              sz_u64_or_if_(0ull, edge_region, have_prev && prev_eff == sz_sentence_break_aterm_k);
    sz_u64_t const eb_upper = (sz_utf8_codepoints_fill_right_(m_upper & significant, flow) << 1) |
                              sz_u64_or_if_(0ull, edge_region, have_prev && prev_eff == sz_sentence_break_upper_k);
    sz_u64_t const eb_lower = (sz_utf8_codepoints_fill_right_(m_lower & significant, flow) << 1) |
                              sz_u64_or_if_(0ull, edge_region, have_prev && prev_eff == sz_sentence_break_lower_k);

    sz_u64_t const eb2_upper = (sz_utf8_codepoints_fill_right_(eb_upper & significant, flow) << 1) |
                               sz_u64_or_if_(0ull, edge_region,
                                             have_prev && prev_prev_eff == sz_sentence_break_upper_k);
    sz_u64_t const eb2_lower = (sz_utf8_codepoints_fill_right_(eb_lower & significant, flow) << 1) |
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

    sz_u64_t const close_phase = sz_utf8_codepoints_fill_right_(m_saterm | carry_close_seed, gate_close) |
                                 carry_sp_seed;
    sz_u64_t const shadow = sz_utf8_codepoints_fill_right_(close_phase | carry_sp_seed, gate_sp);
    sz_u64_t const sp_in_shadow = (m_sp & shadow) | carry_sp_seed;
    sz_u64_t const saw_sp_upto = sz_utf8_codepoints_fill_right_(sp_in_shadow, gate_sp) & shadow;
    sz_u64_t const aterm_close_phase = sz_utf8_codepoints_fill_right_(m_aterm | carry_aterm_close, gate_close) |
                                       carry_aterm_sp;
    sz_u64_t const aterm_shadow = sz_utf8_codepoints_fill_right_(aterm_close_phase | carry_aterm_sp, gate_sp);

    // SB8 in-window lower-ahead: `fill_left` of Lower through the SB8 neutral set. In dense space a stop is one
    // lane, so no continuation-byte exclusion is needed; the neutral gate is just every non-stop codepoint.
    sz_u64_t const sb8_stop = m_lower | m_upper | m_oletter | m_parasep | m_saterm;
    sz_u64_t const neutral = valid & ~sb8_stop;
    sz_u64_t const lower_ahead = sz_utf8_codepoints_fill_left_(m_lower, neutral);
    sz_u64_t const top_bit = (count >= 64) ? (1ull << 63) : (1ull << (count - 1));
    sz_u64_t const neutral_to_edge = sz_utf8_codepoints_fill_left_(top_bit & neutral, neutral) & ~lower_ahead;

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
        breaks &= sz_u64_mask_until_(lane);
    }

    // Update the carry from this window's high edge: trailing shadow run-state and the trailing class context.
    {
        int const edge = (int)count - 1;
        if (valid) carry->prev_raw = sz_utf8_sentence_break_class_at_(classes, edge);
        if (significant) {
            int const last = 63 - sz_u64_clz(significant);
            carry->prev_eff = sz_utf8_sentence_break_class_at_(classes, last);
            sz_u64_t const sig2 = significant & ~(1ull << last);
            if (sig2) carry->prev_prev_eff = sz_utf8_sentence_break_class_at_(classes, 63 - sz_u64_clz(sig2));
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
    carry.have_prev = 0;
    carry.prev_raw = carry.prev_eff = carry.prev_prev_eff = (sz_u8_t)sz_sentence_break_other_k;
    carry.in_shadow = carry.shadow_aterm = carry.shadow_saw_sp = 0;
    carry.sb8_pending = 0;

    sz_size_t sb8_pending_position = 0;
    int sb8_pending_active = 0;

    while (position < length) {
        sz_utf8_codepoints_window_t const decoded = sz_utf8_codepoints_decode_window_(text_u8 + position,
                                                                                      length - position, lane_identity);
        sz_size_t const loaded = decoded.loaded;
        __mmask64 const lead_continuation = (position == 0) ? (__mmask64)1 : (__mmask64)0;
        __mmask64 const codepoint_starts = decoded.codepoint_starts | lead_continuation;
        sz_u64_t const start_bytes = _cvtmask64_u64(codepoint_starts);

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

        __m512i const two_high = sz_utf8_codepoints_srl8_(_mm512_and_si512(window, _mm512_set1_epi8(0x1F)), 2, 0x07);
        __m512i const two_low = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(window, _mm512_set1_epi8(0x03)), 6),
                                                _mm512_and_si512(next1, _mm512_set1_epi8(0x3F)));
        __m512i const three_high = _mm512_or_si512(
            _mm512_slli_epi16(_mm512_and_si512(window, _mm512_set1_epi8(0x0F)), 4),
            sz_utf8_codepoints_srl8_(next1, 2, 0x0F));
        __m512i const three_low = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(next1, _mm512_set1_epi8(0x03)), 6),
                                                  _mm512_and_si512(next2, _mm512_set1_epi8(0x3F)));
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

        // Effective-window<64 trim (identical to the shipped driver): exclude every straddling trailing lead and a
        // stray continuation at the window edge, so only fully-decoded codepoints enter the dense stream.
        sz_size_t complete_limit = loaded;
        if (more_text && start_bytes) {
            sz_u64_t const valid_lanes = sz_u64_mask_until_(loaded);
            sz_u64_t const two = _cvtmask64_u64(decoded.two_byte_starts) & valid_lanes;
            sz_u64_t const three = _cvtmask64_u64(decoded.three_byte_starts) & valid_lanes;
            sz_u64_t const four = _cvtmask64_u64(decoded.four_byte_starts) & valid_lanes;
            sz_u64_t const straddle = ((two & ~sz_u64_mask_until_(loaded > 1 ? loaded - 1 : 0)) |
                                       (three & ~sz_u64_mask_until_(loaded > 2 ? loaded - 2 : 0)) |
                                       (four & ~sz_u64_mask_until_(loaded > 3 ? loaded - 3 : 0))) &
                                      valid_lanes;
            sz_size_t limit = straddle ? (sz_size_t)sz_u64_ctz(straddle) : loaded;
            if ((text_u8[position + loaded] & 0xC0) == 0x80) {
                sz_size_t const last_lead = (sz_size_t)(63 - sz_u64_clz(start_bytes));
                if (last_lead < limit) limit = last_lead;
            }
            if (limit > 0) complete_limit = limit;
        }

        // Dense compaction: keep only codepoint starts inside the complete region, compress their class bytes to a
        // dense codepoint stream (one vpcompressb), and remember the byte lanes for the scatter back.
        sz_u64_t const complete_mask = sz_u64_mask_until_(complete_limit);
        sz_u64_t const dense_start_lanes = start_bytes & complete_mask;
        sz_size_t const dense_count = (sz_size_t)_mm_popcnt_u64(dense_start_lanes);
        __m512i const dense_classes = _mm512_maskz_compress_epi8(_cvtu64_mask64(dense_start_lanes), classes);

        sz_utf8_sentence_break_carry_t carry_full = carry;
        sz_utf8_sentence_break_window_t const win = sz_utf8_sentence_break_block_breaks_(dense_classes, dense_count,
                                                                                         &carry_full, more_text);

        // Resolve a previously deferred SB8 boundary before any of this window's boundaries.
        if (sb8_pending_active && win.sb8_resolution != 0) {
            if (win.sb8_resolution == 1) {
                if (sentences == sentences_capacity) {
                    if (bytes_consumed) *bytes_consumed = sentence_start;
                    return sentences;
                }
                sentence_starts[sentences] = sentence_start;
                sentence_lengths[sentences] = sb8_pending_position - sentence_start;
                ++sentences;
                sentence_start = sb8_pending_position;
            }
            sb8_pending_active = 0;
        }

        // Scatter the trusted dense breaks back to byte lanes. `win.resolved` is dense; deposit the trusted dense
        // bits into the dense start lanes via `_pdep_u64`, then mask to the byte lanes of the trusted dense prefix.
        sz_size_t const dense_adv = win.resolved;
        sz_u64_t const dense_breaks = win.breaks & sz_u64_mask_until_(dense_adv);
        sz_u64_t boundary_lanes = _pdep_u64(dense_breaks, dense_start_lanes);
        if (!carry.have_prev) boundary_lanes &= ~1ull;

        // The byte-domain advance is the byte offset just past the last trusted dense codepoint. When the whole
        // complete region is trusted (dense_adv == dense_count), advance by the complete region; otherwise advance
        // to the byte lane of the first untrusted dense codepoint (the SB8 clamp re-anchor point).
        sz_size_t byte_adv;
        if (dense_adv >= dense_count) { byte_adv = complete_limit ? complete_limit : loaded; }
        else {
            // Byte offset of dense codepoint `dense_adv`: the (dense_adv)-th set bit of `dense_start_lanes`.
            sz_u64_t const upto = _pdep_u64((1ull << dense_adv), dense_start_lanes);
            byte_adv = (sz_size_t)sz_u64_ctz(upto);
        }

        sentences = sz_utf8_codepoints_drain_forward_(boundary_lanes, position, lane_identity, sentence_starts,
                                                      sentence_lengths, sentences, sentences_capacity, &sentence_start);
        if (sentences == sentences_capacity) {
            if (bytes_consumed) *bytes_consumed = sentence_start;
            return sentences;
        }

        if (dense_adv >= dense_count) {
            carry = carry_full;
            position += byte_adv;
        }
        else if (dense_adv > 0) {
            // SB8 clamp strictly before the dense edge: rebuild the carry to the clamp point.
            sz_utf8_sentence_break_carry_t carry_to_edge = carry;
            sz_utf8_sentence_break_block_breaks_(dense_classes, dense_adv, &carry_to_edge, sz_true_k);
            carry = carry_to_edge;
            position += byte_adv;
        }
        else {
            // Clamp at dense lane 0: defer the ATerm SB8 boundary and step past the whole complete span.
            if (!sb8_pending_active) {
                sb8_pending_active = 1;
                sb8_pending_position = position;
            }
            carry = carry_full;
            carry.sb8_pending = 1;
            position += complete_limit ? complete_limit : loaded;
        }
    }

    if (sb8_pending_active) {
        if (sentences == sentences_capacity) {
            if (bytes_consumed) *bytes_consumed = sentence_start;
            return sentences;
        }
        sentence_starts[sentences] = sentence_start;
        sentence_lengths[sentences] = sb8_pending_position - sentence_start;
        ++sentences;
        sentence_start = sb8_pending_position;
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
