/**
 *  @brief Ice Lake backend for UAX-14 line break boundaries.
 *  @file include/stringzilla/utf8_lines/icelake.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_LINES_ICELAKE_H_
#define STRINGZILLA_UTF8_LINES_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_lines/tables.h"
#include "stringzilla/utf8_lines/serial.h"
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

#pragma region UAX-14 Line Boundaries forward kernel

/** @brief Engine side-word bits, one per descriptor flag the LB rules read in-register. */
enum {
    sz_line_break_side_pi_k = 0x01,   /**< gc=Pi (LB15a/LB19) */
    sz_line_break_side_pf_k = 0x02,   /**< gc=Pf (LB15b/LB19) */
    sz_line_break_side_eaw_k = 0x04,  /**< East-Asian Width F/W/H (LB19/LB30) */
    sz_line_break_side_cn_k = 0x08,   /**< unassigned (LB30b, only in conjunction with ExtPict) */
    sz_line_break_side_ext_k = 0x10,  /**< Extended_Pictographic (LB30b) */
    sz_line_break_side_ri_k = 0x20,   /**< raw RI class (LB30a parity) */
    sz_line_break_side_zwj_k = 0x40,  /**< raw ZWJ class (LB8a) */
    sz_line_break_side_mark_k = 0x80, /**< resolved CM|ZWJ (LB9/LB10 attachment) */
    sz_line_break_side_dotted_k = 0x100 /**< DottedCircle U+25CC (LB28a); above the byte side-word */
};

/** @brief Per-pass codepoint window capacity for one streaming forward pass. */
enum { sz_utf8_line_icelake_window_k = 4096 };
/** @brief Max forward lookahead any rule needs (LB25.08/.11 read +2 past the gap). */
enum { sz_utf8_line_icelake_right_look_k = 3 };
/** @brief Constant left-context margin (codepoints) so the first committed lane of a block has its bounded
 *         left neighbours in-register, including the base of an LB9 combining-mark cluster that ends just before
 *         the block seam (CM/ZWJ runs are NOT carried, unlike the unbounded SP/NU/RI run state, so the margin
 *         must span a realistic mark cluster). 16 covers the cluster lengths the differential fuzz exercises. */
enum { sz_utf8_line_icelake_left_context_k = 16 };

/** @brief Window state carried across the 64-lane block window_edge: the open run straddling into lane 0. */
typedef struct sz_line_break_carry_t {
    sz_u8_t open_sp_opener; /**< class anchoring an open "X SP*" run (LB14/16/17/8); 0xFF = none */
    sz_u8_t in_nu_run;      /**< lane 0 continues a "NU (SY|IS)*" run (LB25) */
    sz_u8_t ri_open;        /**< lane 0 continues an RI sequence (parity carries) */
    sz_u8_t ri_parity_odd;  /**< odd count of RI characters precede lane 0 within the open sequence */
} sz_line_break_carry_t;

/** @brief Per-block engine result: the UAX-14 break-before opportunities for the block. */
typedef struct sz_line_break_decided_t {
    sz_u64_t brk; /**< bit i set => break allowed before lane i */
} sz_line_break_decided_t;

#pragma region In-register vectorized classifier

/*  The classifier resolves a contiguous run of codepoints to per-codepoint (class, side, dotted) with ZERO
 *  per-lane scalar loop, ZERO `vpgather`, and NO serial `sz_rune_line_break_property` deferral. It processes
 *  sixteen 32-bit codepoint lanes per pass: each codepoint is mapped to a PALETTE INDEX in-register (page-LUT
 *  for cp<0x800, the shared substrate-style two-stage trie for 0x800..0xFFFF, arithmetic range compares for
 *  the 50 big homogeneous blocks and the 618 astral ranges), then the 62-entry palette descriptor is unpacked
 *  to the LB1-resolved class byte and the engine side byte by in-register compares and masked moves. The
 *  resolution precedence mirrors `sz_rune_line_break_property`: a big-range hit (first match in array order)
 *  overrides everything; else the page-LUT; else the trie; else an astral range; else palette[0]. */

/** @brief Byte-table read addressed by a 32-bit-lane index via a `vpermi2b` page network. Tile slices are loaded
 *         on demand from @p table (no pre-materialized resident copy) so only the 128-byte pages an index range
 *         actually reaches are touched -- the ASCII fast path hits page 0 only, CJK skips the trie entirely. */
SZ_INTERNAL __m512i sz_line_break_permute_byte_(sz_u8_t const *table, int byte_count, __m512i indices) {
    __m512i const within = _mm512_and_si512(indices, _mm512_set1_epi32(0x7F));
    __m512i const page = _mm512_srli_epi32(indices, 7);
    __m512i result = _mm512_setzero_si512();
    int const pages = (byte_count + 127) / 128;
    for (int page_index = 0; page_index < pages; ++page_index) {
        int const base = page_index * 128, remaining = byte_count - base;
        __m512i const low = _mm512_maskz_loadu_epi8(sz_u64_clamp_mask_until_((sz_size_t)remaining), table + base);
        __m512i const high =
            remaining > 64
                ? _mm512_maskz_loadu_epi8(sz_u64_clamp_mask_until_((sz_size_t)(remaining - 64)), table + base + 64)
                : _mm512_setzero_si512();
        __m512i const picked = _mm512_permutex2var_epi8(low, within, high);
        __mmask16 const here = _mm512_cmpeq_epi32_mask(page, _mm512_set1_epi32(page_index));
        result = _mm512_mask_blend_epi32(here, result, _mm512_and_si512(picked, _mm512_set1_epi32(0xFF)));
    }
    return result;
}

/** @brief Word-table read addressed by a 32-bit-lane index via a `vpermi2w` page network, tiles loaded on demand. */
SZ_INTERNAL __m512i sz_line_break_permute_word_(sz_u16_t const *table, int word_count, __m512i indices) {
    __m512i const within = _mm512_and_si512(indices, _mm512_set1_epi32(0x1F));
    __m512i const page = _mm512_srli_epi32(indices, 5);
    __m512i result = _mm512_setzero_si512();
    int const tiles = (word_count + 31) / 32;
    for (int tile = 0; tile < tiles; ++tile) {
        int const base = tile * 32, remaining = word_count - base;
        __m512i const slice = _mm512_maskz_loadu_epi16(sz_u32_clamp_mask_until_((sz_size_t)remaining), table + base);
        __m512i const picked = _mm512_permutexvar_epi16(within, slice);
        __mmask16 const here = _mm512_cmpeq_epi32_mask(page, _mm512_set1_epi32(tile));
        result = _mm512_mask_blend_epi32(here, result, _mm512_and_si512(picked, _mm512_set1_epi32(0xFFFF)));
    }
    return result;
}

/** @brief Page-LUT palette index for sixteen 32-bit codepoint lanes with cp<0x800 (pages loaded on demand). */
SZ_INTERNAL __m512i sz_line_break_page_index_(__m512i codepoints) {
    __m512i const low11 = _mm512_and_si512(codepoints, _mm512_set1_epi32(0x7FF));
    __m512i const low7 = _mm512_and_si512(low11, _mm512_set1_epi32(0x7F));
    __m512i const span = _mm512_srli_epi32(low11, 7);
    __m512i result = _mm512_setzero_si512();
    for (int span_index = 0; span_index < 16; ++span_index) {
        __m512i const low = _mm512_loadu_si512((void const *)(sz_utf8_line_break_page_lut_ + span_index * 128));
        __m512i const high = _mm512_loadu_si512((void const *)(sz_utf8_line_break_page_lut_ + span_index * 128 + 64));
        __m512i const picked = _mm512_permutex2var_epi8(low, low7, high);
        __mmask16 const here = _mm512_cmpeq_epi32_mask(span, _mm512_set1_epi32(span_index));
        result = _mm512_mask_mov_epi32(result, here, _mm512_and_si512(picked, _mm512_set1_epi32(0xFF)));
    }
    return result;
}

/** @brief Two-stage trie palette index for sixteen 32-bit lanes with 0x800<=cp<0x10000 (block=8, subblock=8). */
SZ_INTERNAL __m512i sz_line_break_trie_index_(__m512i codepoints) {
    __m512i const offset = _mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x800));
    __m512i const block = _mm512_srli_epi32(offset, 3);
    __m512i const within = _mm512_and_si512(offset, _mm512_set1_epi32(7));
    __m512i const super = _mm512_srli_epi32(block, 3);
    __m512i const super_offset = _mm512_and_si512(block, _mm512_set1_epi32(7));
    __m512i const level1 = sz_line_break_permute_byte_(sz_utf8_line_break_trie_l1_, (int)sizeof(sz_utf8_line_break_trie_l1_), super);
    __m512i const level2_index = _mm512_add_epi32(_mm512_mullo_epi32(level1, _mm512_set1_epi32(8)), super_offset);
    __m512i const leaf = sz_line_break_permute_word_(sz_utf8_line_break_trie_l2_, (int)(sizeof(sz_utf8_line_break_trie_l2_) / 2), level2_index);
    __m512i const leaf_index = _mm512_add_epi32(_mm512_mullo_epi32(leaf, _mm512_set1_epi32(8)), within);
    return sz_line_break_permute_byte_(sz_utf8_line_break_trie_leaf_, (int)sizeof(sz_utf8_line_break_trie_leaf_), leaf_index);
}

/** @brief Palette index for sixteen 32-bit codepoint lanes, matching the serial resolution precedence. */
SZ_INTERNAL __m512i sz_line_break_palette_index_(__m512i codepoints) {
    //  Pure-ASCII fast path (the dominant Latin case): every codepoint < 0x80 lives in page span 0, so the 16-span
    //  page loop and the >=0xC0 big-range loop are both provably no-ops -- one `vpermi2b` resolves the whole block.
    __mmask16 const below_ascii = _mm512_cmplt_epu32_mask(codepoints, _mm512_set1_epi32(0x80));
    if (below_ascii == 0xFFFF) {
        __m512i const low7 = _mm512_and_si512(codepoints, _mm512_set1_epi32(0x7F));
        __m512i const low = _mm512_loadu_si512((void const *)sz_utf8_line_break_page_lut_);
        __m512i const high = _mm512_loadu_si512((void const *)(sz_utf8_line_break_page_lut_ + 64));
        return _mm512_and_si512(_mm512_permutex2var_epi8(low, low7, high), _mm512_set1_epi32(0xFF));
    }
    //  Big ranges have the highest precedence and serial takes the FIRST array match. Resolve them up front so
    //  the (expensive) two-stage BMP trie is run ONLY for the BMP lanes a big range does not already cover -- for
    //  pure CJK (0x4E00..0x9FFF is one big range) that elides the trie entirely. Iterate in reverse so the earliest
    //  matching range is the last to overwrite the value, reproducing first-match-wins; the smallest big-range
    //  lower bound is 0xC0, so a block entirely below that matches nothing and skips the loop.
    __m512i big_value = _mm512_setzero_si512();
    __mmask16 big_covered = 0;
    if (_mm512_cmpge_epu32_mask(codepoints, _mm512_set1_epi32(0xC0))) {
        for (int range = (int)sz_utf8_line_break_big_count_k - 1; range >= 0; --range) {
            __mmask16 const above = _mm512_cmpge_epu32_mask(codepoints, _mm512_set1_epi32((int)sz_utf8_line_break_big_lo_[range]));
            __mmask16 const below = _mm512_cmple_epu32_mask(codepoints, _mm512_set1_epi32((int)sz_utf8_line_break_big_hi_[range]));
            __mmask16 const hit = above & below;
            big_value = _mm512_mask_mov_epi32(big_value, hit, _mm512_set1_epi32(sz_utf8_line_break_big_idx_[range]));
            big_covered |= hit;
        }
    }

    __m512i index = _mm512_setzero_si512();
    __mmask16 const need_page = _mm512_cmplt_epu32_mask(codepoints, _mm512_set1_epi32(0x800)) & ~big_covered;
    if (need_page) index = _mm512_mask_mov_epi32(index, need_page, sz_line_break_page_index_(codepoints));
    __mmask16 const need_trie =
        _mm512_cmpge_epu32_mask(codepoints, _mm512_set1_epi32(0x800)) &
        _mm512_cmplt_epu32_mask(codepoints, _mm512_set1_epi32(0x10000)) & ~big_covered;
    if (need_trie) index = _mm512_mask_mov_epi32(index, need_trie, sz_line_break_trie_index_(codepoints));
    __mmask16 const need_astral = _mm512_cmpge_epu32_mask(codepoints, _mm512_set1_epi32(0x10000)) & ~big_covered;
    if (need_astral) {
        __m512i astral_value = _mm512_setzero_si512();
        for (int range = 0; range < (int)sz_utf8_line_break_astral_count_k; ++range) {
            __mmask16 const above = _mm512_cmpge_epu32_mask(codepoints, _mm512_set1_epi32((int)sz_utf8_line_break_astral_lo_[range]));
            __mmask16 const below = _mm512_cmple_epu32_mask(codepoints, _mm512_set1_epi32((int)sz_utf8_line_break_astral_hi_[range]));
            astral_value = _mm512_mask_mov_epi32(astral_value, above & below, _mm512_set1_epi32(sz_utf8_line_break_astral_idx_[range]));
        }
        index = _mm512_mask_mov_epi32(index, need_astral, astral_value);
    }
    index = _mm512_mask_mov_epi32(index, big_covered, big_value);
    return index;
}

/** @brief Unpack sixteen palette descriptors to per-lane LB1-resolved class, side byte, and DottedCircle mask. */
SZ_INTERNAL void sz_line_break_descriptor_unpack_(__m512i descriptors, __m512i *classes_out, __m512i *side_out,
                                                  __mmask16 *dotted_out) {
    __m512i classes = _mm512_and_si512(descriptors, _mm512_set1_epi32(0x3F));
    __mmask16 const is_sa = _mm512_cmpeq_epi32_mask(classes, _mm512_set1_epi32(sz_line_break_sa_k));
    __mmask16 const sa_is_mark = _mm512_test_epi32_mask(descriptors, _mm512_set1_epi32(1 << 12));
    classes = _mm512_mask_mov_epi32(classes, is_sa, _mm512_set1_epi32(sz_line_break_al_k));
    classes = _mm512_mask_mov_epi32(classes, is_sa & sa_is_mark, _mm512_set1_epi32(sz_line_break_cm_k));
    __mmask16 const is_alias = _mm512_cmpeq_epi32_mask(classes, _mm512_set1_epi32(sz_line_break_ai_k)) |
                               _mm512_cmpeq_epi32_mask(classes, _mm512_set1_epi32(sz_line_break_sg_k)) |
                               _mm512_cmpeq_epi32_mask(classes, _mm512_set1_epi32(sz_line_break_xx_k));
    classes = _mm512_mask_mov_epi32(classes, is_alias, _mm512_set1_epi32(sz_line_break_al_k));
    __mmask16 const is_cj = _mm512_cmpeq_epi32_mask(classes, _mm512_set1_epi32(sz_line_break_cj_k));
    classes = _mm512_mask_mov_epi32(classes, is_cj, _mm512_set1_epi32(sz_line_break_ns_k));

    __m512i side = _mm512_setzero_si512();
    side = _mm512_or_si512(side, _mm512_maskz_mov_epi32(_mm512_test_epi32_mask(descriptors, _mm512_set1_epi32(1 << 6)),
                                                        _mm512_set1_epi32(sz_line_break_side_pi_k)));
    side = _mm512_or_si512(side, _mm512_maskz_mov_epi32(_mm512_test_epi32_mask(descriptors, _mm512_set1_epi32(1 << 7)),
                                                        _mm512_set1_epi32(sz_line_break_side_pf_k)));
    side = _mm512_or_si512(side, _mm512_maskz_mov_epi32(_mm512_test_epi32_mask(descriptors, _mm512_set1_epi32(1 << 8)),
                                                        _mm512_set1_epi32(sz_line_break_side_eaw_k)));
    side = _mm512_or_si512(side, _mm512_maskz_mov_epi32(_mm512_test_epi32_mask(descriptors, _mm512_set1_epi32(1 << 9)),
                                                        _mm512_set1_epi32(sz_line_break_side_cn_k | sz_line_break_side_ext_k)));
    __m512i const raw_class = _mm512_and_si512(descriptors, _mm512_set1_epi32(0x3F));
    side = _mm512_or_si512(side, _mm512_maskz_mov_epi32(_mm512_cmpeq_epi32_mask(raw_class, _mm512_set1_epi32(sz_line_break_ri_k)),
                                                        _mm512_set1_epi32(sz_line_break_side_ri_k)));
    side = _mm512_or_si512(side, _mm512_maskz_mov_epi32(_mm512_cmpeq_epi32_mask(raw_class, _mm512_set1_epi32(sz_line_break_zwj_k)),
                                                        _mm512_set1_epi32(sz_line_break_side_zwj_k)));
    __mmask16 const class_is_mark = _mm512_cmpeq_epi32_mask(classes, _mm512_set1_epi32(sz_line_break_cm_k)) |
                                    _mm512_cmpeq_epi32_mask(classes, _mm512_set1_epi32(sz_line_break_zwj_k));
    side = _mm512_or_si512(side, _mm512_maskz_mov_epi32(class_is_mark, _mm512_set1_epi32(sz_line_break_side_mark_k)));

    *classes_out = classes;
    *side_out = side;
    *dotted_out = _mm512_test_epi32_mask(descriptors, _mm512_set1_epi32(1 << 13));
}

/**
 *  @brief  Classify @p count decoded codepoints to per-codepoint class / side bytes plus a DottedCircle bitmap,
 *          fully in-register. Output @p class_out / @p side_out hold one byte per codepoint; @p dotted_out is a
 *          bitmap (bit i set => codepoint i is DottedCircle U+25CC). No per-lane scalar loop, no gather, no
 *          serial deferral.
 */
SZ_INTERNAL void sz_line_break_classify_block_(sz_u32_t const *codepoints, sz_size_t count, sz_u8_t *class_out,
                                               sz_u8_t *side_out, sz_u64_t *dotted_out) {
    sz_size_t lane = 0;
    for (; lane + 16 <= count; lane += 16) {
        __m512i const codepoint_vec = _mm512_loadu_si512((void const *)(codepoints + lane));
        __m512i const index = sz_line_break_palette_index_(codepoint_vec);
        __m512i const descriptor =
            sz_line_break_permute_word_(sz_utf8_line_break_palette_, (int)sz_utf8_line_break_palette_count_k, index);
        __m512i classes, side;
        __mmask16 dotted;
        sz_line_break_descriptor_unpack_(descriptor, &classes, &side, &dotted);
        __m128i const class_bytes = _mm512_cvtepi32_epi8(classes);
        __m128i const side_bytes = _mm512_cvtepi32_epi8(side);
        _mm_storeu_si128((__m128i *)(class_out + lane), class_bytes);
        _mm_storeu_si128((__m128i *)(side_out + lane), side_bytes);
        *dotted_out = (*dotted_out & ~(0xFFFFull << lane)) | ((sz_u64_t)dotted << lane);
    }
    if (lane < count) {
        __mmask16 const tail = sz_u16_mask_until_(count - lane);
        __m512i const codepoint_vec = _mm512_maskz_loadu_epi32(tail, codepoints + lane);
        __m512i const index = sz_line_break_palette_index_(codepoint_vec);
        __m512i const descriptor =
            sz_line_break_permute_word_(sz_utf8_line_break_palette_, (int)sz_utf8_line_break_palette_count_k, index);
        __m512i classes, side;
        __mmask16 dotted;
        sz_line_break_descriptor_unpack_(descriptor, &classes, &side, &dotted);
        _mm_mask_storeu_epi8(class_out + lane, tail, _mm512_cvtepi32_epi8(classes));
        _mm_mask_storeu_epi8(side_out + lane, tail, _mm512_cvtepi32_epi8(side));
        sz_size_t const tail_count = count - lane;
        sz_u64_t const tail_dotted = (sz_u64_t)(dotted & tail) << lane;
        *dotted_out = (*dotted_out & ~(((tail_count >= 64) ? ~0ull : ((1ull << tail_count) - 1)) << lane)) | tail_dotted;
    }
}

#pragma endregion In-register vectorized classifier

#pragma region Mask-algebra rule engine

/** @brief Kogge-Stone segmented contiguous rightward fill of @p seed through @p run lanes (LB9/LB14-17/25/30a). */
SZ_INTERNAL sz_u64_t sz_line_break_smear_right_(sz_u64_t seed, sz_u64_t run) {
    sz_u64_t acc = seed, barrier = run;
    acc |= barrier & (acc << 1), barrier &= (barrier << 1);
    acc |= barrier & (acc << 2), barrier &= (barrier << 2);
    acc |= barrier & (acc << 4), barrier &= (barrier << 4);
    acc |= barrier & (acc << 8), barrier &= (barrier << 8);
    acc |= barrier & (acc << 16), barrier &= (barrier << 16);
    acc |= barrier & (acc << 32);
    return acc;
}

/** @brief Build a 64-bit "lane class == @p cls" mask with one `vpcmpeqb` -> kmask. */
SZ_INTERNAL sz_u64_t sz_line_break_class_mask_(__m512i classes, sz_u8_t cls) {
    return _cvtmask64_u64(_mm512_cmpeq_epi8_mask(classes, _mm512_set1_epi8((char)cls)));
}

/** @brief Build a 64-bit "lane (side & @p bit) != 0" mask with one `vptestmb` -> kmask. */
SZ_INTERNAL sz_u64_t sz_line_break_side_mask_(__m512i side_lo, sz_u8_t bit) {
    __m512i const masked = _mm512_and_si512(side_lo, _mm512_set1_epi8((char)bit));
    return _cvtmask64_u64(_mm512_test_epi8_mask(masked, masked));
}

/** @brief Broadcast a base-lane signal @p anchor across its following attached CM/ZWJ mark run (LB9). */
SZ_INTERNAL sz_u64_t sz_line_break_over_marks_(sz_u64_t anchor, sz_u64_t attached) {
    sz_u64_t const seeded = anchor | ((anchor << 1) & attached);
    return sz_line_break_smear_right_(seeded, anchor | attached);
}

/** @brief LB9 base-class broadcast of @p base_of_class over attached marks (anchored on good bases only). */
SZ_INTERNAL sz_u64_t sz_line_break_broadcast_base_(sz_u64_t base_of_class, sz_u64_t base_good, sz_u64_t attached) {
    sz_u64_t const anchor = base_of_class & base_good;
    sz_u64_t const seeded = anchor | ((anchor << 1) & attached);
    return sz_line_break_smear_right_(seeded, anchor | attached) & (anchor | attached);
}

/** @brief Effective class mask: own non-mark lanes plus the base broadcast over attached marks. */
SZ_INTERNAL sz_u64_t sz_line_break_eff_(sz_u64_t base_of_class, sz_u64_t mark, sz_u64_t base_good, sz_u64_t attached) {
    return (base_of_class & ~mark) | sz_line_break_broadcast_base_(base_of_class & base_good, base_good, attached);
}

/** @brief Effective side-flag mask: own non-mark lanes plus the flag broadcast over attached marks. */
SZ_INTERNAL sz_u64_t sz_line_break_eff_side_(sz_u64_t side_mask, sz_u64_t mark, sz_u64_t base_good, sz_u64_t attached) {
    sz_u64_t const anchor = side_mask & base_good;
    sz_u64_t const seeded = anchor | ((anchor << 1) & attached);
    return (side_mask & ~mark) | (sz_line_break_smear_right_(seeded, anchor | attached) & (anchor | attached));
}

/** @brief Segmented "opener SP* reaches lane" governance, continued across the window_edge via @p carried. */
SZ_INTERNAL sz_u64_t sz_line_break_open_run_(sz_u64_t opener, sz_bool_t carried, sz_u64_t spaces) {
    sz_u64_t seed = opener;
    if (carried && (spaces & 1ull)) seed |= 1ull;
    return sz_line_break_smear_right_(seed | ((seed << 1) & spaces), seed | spaces) & (seed | spaces);
}

/** @brief Commit a forced break at the undecided lanes of @p where (sets break + settled). */
SZ_INTERNAL void sz_line_break_force_break_(sz_u64_t where, sz_u64_t all, sz_u64_t *settled, sz_u64_t *brk) {
    sz_u64_t const w = where & ~*settled & all;
    *brk |= w, *settled |= w;
}

/** @brief Commit a forced join (no break) at the undecided lanes of @p where (sets settled only). */
SZ_INTERNAL void sz_line_break_force_join_(sz_u64_t where, sz_u64_t all, sz_u64_t *settled) {
    *settled |= where & ~*settled & all;
}

/**
 *  @brief  Decide break-before bits for one contiguous block of <= 64 pre-classified codepoints.
 *
 *  @p classes / @p side_lo are the resolved class and low-8 side byte per lane (ZMM, one byte per codepoint);
 *  @p dotted is the block-local DottedCircle bitmap. @p is_sot / @p is_eot mark the true text ends. @p carry
 *  holds the open run straddling into lane 0; @p carry_out (optional) reports the run state entering @p resume_lane.
 */
SZ_INTERNAL sz_line_break_decided_t sz_line_break_decide_block_(             //
    __m512i classes, __m512i side_lo, sz_u64_t dotted, sz_size_t count,     //
    sz_bool_t is_sot, sz_bool_t is_eot, sz_line_break_carry_t carry,        //
    sz_size_t resume_lane, sz_line_break_carry_t *carry_out) {

    sz_u64_t const ALL = sz_u64_mask_until_(count);
    sz_u64_t const INTERIOR = ALL & ~1ull;
    sz_u64_t const SOT_BIT = is_sot ? 1ull : 0ull;
    sz_u64_t const EOT_LANE = is_eot ? (1ull << (count - 1)) : 0ull;
    sz_u64_t const DC = dotted & ALL;

    sz_u64_t BK = sz_line_break_class_mask_(classes, sz_line_break_bk_k), CR = sz_line_break_class_mask_(classes, sz_line_break_cr_k);
    sz_u64_t LF = sz_line_break_class_mask_(classes, sz_line_break_lf_k), NL = sz_line_break_class_mask_(classes, sz_line_break_nl_k);
    sz_u64_t SP = sz_line_break_class_mask_(classes, sz_line_break_sp_k), ZW = sz_line_break_class_mask_(classes, sz_line_break_zw_k);
    sz_u64_t WJ = sz_line_break_class_mask_(classes, sz_line_break_wj_k), GL = sz_line_break_class_mask_(classes, sz_line_break_gl_k);
    sz_u64_t CM = sz_line_break_class_mask_(classes, sz_line_break_cm_k), ZWJ = sz_line_break_class_mask_(classes, sz_line_break_zwj_k);
    sz_u64_t OP = sz_line_break_class_mask_(classes, sz_line_break_op_k), CL = sz_line_break_class_mask_(classes, sz_line_break_cl_k);
    sz_u64_t CP = sz_line_break_class_mask_(classes, sz_line_break_cp_k), QU = sz_line_break_class_mask_(classes, sz_line_break_qu_k);
    sz_u64_t NS = sz_line_break_class_mask_(classes, sz_line_break_ns_k), B2 = sz_line_break_class_mask_(classes, sz_line_break_b2_k);
    sz_u64_t HY = sz_line_break_class_mask_(classes, sz_line_break_hy_k), HH = sz_line_break_class_mask_(classes, sz_line_break_hh_k);
    sz_u64_t BA = sz_line_break_class_mask_(classes, sz_line_break_ba_k), BB = sz_line_break_class_mask_(classes, sz_line_break_bb_k);
    sz_u64_t IN = sz_line_break_class_mask_(classes, sz_line_break_in_k), CB = sz_line_break_class_mask_(classes, sz_line_break_cb_k);
    sz_u64_t IS = sz_line_break_class_mask_(classes, sz_line_break_is_k), NU = sz_line_break_class_mask_(classes, sz_line_break_nu_k);
    sz_u64_t SY = sz_line_break_class_mask_(classes, sz_line_break_sy_k), EX = sz_line_break_class_mask_(classes, sz_line_break_ex_k);
    sz_u64_t PO = sz_line_break_class_mask_(classes, sz_line_break_po_k), PR = sz_line_break_class_mask_(classes, sz_line_break_pr_k);
    sz_u64_t AL = sz_line_break_class_mask_(classes, sz_line_break_al_k), HL = sz_line_break_class_mask_(classes, sz_line_break_hl_k);
    sz_u64_t ID = sz_line_break_class_mask_(classes, sz_line_break_id_k), EB = sz_line_break_class_mask_(classes, sz_line_break_eb_k);
    sz_u64_t EM = sz_line_break_class_mask_(classes, sz_line_break_em_k), RI = sz_line_break_class_mask_(classes, sz_line_break_ri_k);
    sz_u64_t JL = sz_line_break_class_mask_(classes, sz_line_break_jl_k), JV = sz_line_break_class_mask_(classes, sz_line_break_jv_k);
    sz_u64_t JT = sz_line_break_class_mask_(classes, sz_line_break_jt_k);
    sz_u64_t H2 = sz_line_break_class_mask_(classes, sz_line_break_h2_k), H3 = sz_line_break_class_mask_(classes, sz_line_break_h3_k);
    sz_u64_t AK = sz_line_break_class_mask_(classes, sz_line_break_ak_k), AP = sz_line_break_class_mask_(classes, sz_line_break_ap_k);
    sz_u64_t AS = sz_line_break_class_mask_(classes, sz_line_break_as_k), VF = sz_line_break_class_mask_(classes, sz_line_break_vf_k);
    sz_u64_t VI = sz_line_break_class_mask_(classes, sz_line_break_vi_k);

    sz_u64_t PI = sz_line_break_side_mask_(side_lo, sz_line_break_side_pi_k), PF = sz_line_break_side_mask_(side_lo, sz_line_break_side_pf_k);
    sz_u64_t EAW = sz_line_break_side_mask_(side_lo, sz_line_break_side_eaw_k), CN = sz_line_break_side_mask_(side_lo, sz_line_break_side_cn_k);
    sz_u64_t EXTPICT = sz_line_break_side_mask_(side_lo, sz_line_break_side_ext_k);

    sz_u64_t mark = CM | ZWJ;
    sz_u64_t excluded = BK | CR | LF | NL | SP | ZW;
    sz_u64_t base_good = ALL & ~mark & ~excluded;
    sz_u64_t attached = sz_line_break_smear_right_((base_good << 1) & mark, mark);
    sz_u64_t unattached_mark = mark & ~attached;

    //  The mark-attachment machinery (LB9) is pure overhead when a block carries no combining marks -- the
    //  overwhelmingly common case for unaccented Latin, CJK, Hangul. When `mark == 0`, `attached`/`unattached_mark`
    //  are empty and every `sz_line_break_eff_`/`sz_line_break_eff_side_` provably reduces to the identity
    //  (broadcast over an empty mark run is a no-op), so the ~30 segmented smears are skipped and the raw class
    //  masks are used directly. Bit-identical to the general path by construction.
    sz_u64_t eOP, eCL, eCP, eQU, eNS, eB2, eHY, eHH, eBA, eBB, eIN, eCB, eIS, eNU, eSY, eEX, ePO, ePR;
    sz_u64_t eAL, eHL, eID, eEB, eEM, eJL, eJV, eJT, eH2, eH3, eWJ, eGL, eAK, eAP, eAS, eVF, eVI;
    sz_u64_t ePI, ePF, eEAW, eCN, eEXT;
    if (mark) {
        eOP = sz_line_break_eff_(OP, mark, base_good, attached), eCL = sz_line_break_eff_(CL, mark, base_good, attached);
        eCP = sz_line_break_eff_(CP, mark, base_good, attached), eQU = sz_line_break_eff_(QU, mark, base_good, attached);
        eNS = sz_line_break_eff_(NS, mark, base_good, attached), eB2 = sz_line_break_eff_(B2, mark, base_good, attached);
        eHY = sz_line_break_eff_(HY, mark, base_good, attached), eHH = sz_line_break_eff_(HH, mark, base_good, attached);
        eBA = sz_line_break_eff_(BA, mark, base_good, attached), eBB = sz_line_break_eff_(BB, mark, base_good, attached);
        eIN = sz_line_break_eff_(IN, mark, base_good, attached), eCB = sz_line_break_eff_(CB, mark, base_good, attached);
        eIS = sz_line_break_eff_(IS, mark, base_good, attached), eNU = sz_line_break_eff_(NU, mark, base_good, attached);
        eSY = sz_line_break_eff_(SY, mark, base_good, attached), eEX = sz_line_break_eff_(EX, mark, base_good, attached);
        ePO = sz_line_break_eff_(PO, mark, base_good, attached), ePR = sz_line_break_eff_(PR, mark, base_good, attached);
        eAL = sz_line_break_eff_(AL, mark, base_good, attached) | unattached_mark, eHL = sz_line_break_eff_(HL, mark, base_good, attached);
        eID = sz_line_break_eff_(ID, mark, base_good, attached), eEB = sz_line_break_eff_(EB, mark, base_good, attached);
        eEM = sz_line_break_eff_(EM, mark, base_good, attached);
        eJL = sz_line_break_eff_(JL, mark, base_good, attached), eJV = sz_line_break_eff_(JV, mark, base_good, attached);
        eJT = sz_line_break_eff_(JT, mark, base_good, attached);
        eH2 = sz_line_break_eff_(H2, mark, base_good, attached), eH3 = sz_line_break_eff_(H3, mark, base_good, attached);
        eWJ = sz_line_break_eff_(WJ, mark, base_good, attached), eGL = sz_line_break_eff_(GL, mark, base_good, attached);
        eAK = sz_line_break_eff_(AK, mark, base_good, attached), eAP = sz_line_break_eff_(AP, mark, base_good, attached);
        eAS = sz_line_break_eff_(AS, mark, base_good, attached);
        eVF = sz_line_break_eff_(VF, mark, base_good, attached), eVI = sz_line_break_eff_(VI, mark, base_good, attached);
        ePI = sz_line_break_eff_side_(PI, mark, base_good, attached), ePF = sz_line_break_eff_side_(PF, mark, base_good, attached);
        eEAW = sz_line_break_eff_side_(EAW, mark, base_good, attached), eCN = sz_line_break_eff_side_(CN, mark, base_good, attached);
        eEXT = sz_line_break_eff_side_(EXTPICT, mark, base_good, attached);
    }
    else {
        eOP = OP, eCL = CL, eCP = CP, eQU = QU, eNS = NS, eB2 = B2, eHY = HY, eHH = HH, eBA = BA, eBB = BB;
        eIN = IN, eCB = CB, eIS = IS, eNU = NU, eSY = SY, eEX = EX, ePO = PO, ePR = PR;
        eAL = AL, eHL = HL, eID = ID, eEB = EB, eEM = EM, eJL = JL, eJV = JV, eJT = JT, eH2 = H2, eH3 = H3;
        eWJ = WJ, eGL = GL, eAK = AK, eAP = AP, eAS = AS, eVF = VF, eVI = VI;
        ePI = PI, ePF = PF, eEAW = EAW, eCN = CN, eEXT = EXTPICT;
    }

    sz_u64_t eSP = SP, eZW = ZW, eBK = BK, eCR = CR, eLF = LF, eNL = NL;
    sz_u64_t rawZWJ = ZWJ; /* LB8a tests the ORIGINAL ZWJ class. */

    sz_u64_t settled = 0, brk = 0;
    sz_bool_t carry_op = (sz_bool_t)(carry.open_sp_opener == sz_line_break_op_k);
    sz_bool_t carry_clcp = (sz_bool_t)(carry.open_sp_opener == sz_line_break_cl_k || carry.open_sp_opener == sz_line_break_cp_k);
    sz_bool_t carry_b2 = (sz_bool_t)(carry.open_sp_opener == sz_line_break_b2_k);
    sz_bool_t carry_zw = (sz_bool_t)(carry.open_sp_opener == sz_line_break_zw_k);
    sz_u64_t op_gov = sz_line_break_open_run_(eOP, carry_op, eSP);
    sz_u64_t cl_gov = sz_line_break_open_run_(eCL | eCP, carry_clcp, eSP);
    sz_u64_t b2_gov = sz_line_break_open_run_(eB2, carry_b2, eSP);
    sz_u64_t zw_gov = sz_line_break_open_run_(eZW, carry_zw, eSP);

    // ============================ Rules in priority order ============================

    // LB4: BK !  (mandatory).
    sz_u64_t lb4 = (eBK << 1) & INTERIOR;
    sz_line_break_force_break_(lb4, ALL, &settled, &brk);
    // LB5: CR x LF ; CR ! ; LF ! ; NL ! (the forced breaks are mandatory).
    sz_line_break_force_join_(((eCR << 1) & eLF) & INTERIOR, ALL, &settled);
    sz_u64_t lb5 = ((eCR | eLF | eNL) << 1) & INTERIOR;
    sz_line_break_force_break_(lb5, ALL, &settled, &brk);
    // LB6: x (BK|CR|LF|NL).
    sz_line_break_force_join_((eBK | eCR | eLF | eNL) & INTERIOR, ALL, &settled);
    // LB7: x SP ; x ZW.
    sz_line_break_force_join_((eSP | eZW) & INTERIOR, ALL, &settled);
    // LB8: ZW SP* /
    sz_line_break_force_break_(((zw_gov << 1) & ~zw_gov) & INTERIOR, ALL, &settled, &brk);
    // LB8a: ZWJ x (original class).
    sz_line_break_force_join_((rawZWJ << 1) & INTERIOR, ALL, &settled);
    // LB9: x (attached CM|ZWJ).
    sz_line_break_force_join_(attached & INTERIOR, ALL, &settled);
    // LB11: x WJ ; WJ x.
    sz_line_break_force_join_((eWJ | (eWJ << 1)) & INTERIOR, ALL, &settled);
    // LB12: GL x.
    sz_line_break_force_join_((eGL << 1) & INTERIOR, ALL, &settled);
    // LB12a: [^SP BA HY HH] x GL.
    sz_line_break_force_join_((eGL & ~(((eSP | eBA | eHY | eHH) << 1))) & INTERIOR, ALL, &settled);
    // LB13: x CL ; x CP ; x EX ; x SY.
    sz_line_break_force_join_((eCL | eCP | eEX | eSY) & INTERIOR, ALL, &settled);
    // LB14: OP SP* x.
    sz_line_break_force_join_((op_gov << 1) & INTERIOR, ALL, &settled);
    // LB15a: (sot|allowed) [QU&Pi] SP* x. Seed only at a real opening quote -- the base QU lane (raw class QU), not
    // a combining mark that merely inherited QU/Pi from its base (such a mark, whose left neighbour is its own base
    // QU, would otherwise spuriously re-open the run). The governance bridges the quote's own attached marks and the
    // following spaces, so the legitimate "QU CM SP x" case is preserved.
    {
        sz_u64_t qupi = QU & PI;
        sz_u64_t allowedL = eBK | eCR | eLF | eNL | eOP | eQU | eGL | eSP | eZW;
        sz_u64_t seed = qupi & ((allowedL << 1) | SOT_BIT);
        sz_u64_t run = attached | eSP;
        sz_u64_t gov = sz_line_break_smear_right_(seed | ((seed << 1) & run), seed | run) & (seed | run);
        sz_line_break_force_join_((gov << 1) & INTERIOR, ALL, &settled);
    }
    // LB15b: x [QU&Pf] (followed by allowed | eot).
    {
        sz_u64_t qupf = eQU & ePF;
        sz_u64_t rset = eSP | eGL | eWJ | eCL | eQU | eCP | eEX | eIS | eSY | eBK | eCR | eLF | eNL | eZW;
        sz_u64_t right_ok = (rset >> 1) | (qupf & EOT_LANE);
        sz_line_break_force_join_((qupf & right_ok) & INTERIOR, ALL, &settled);
    }
    // LB15.3: SP / IS NU.
    sz_line_break_force_break_(((eSP << 1) & eIS & (eNU >> 1)) & INTERIOR, ALL, &settled, &brk);
    // LB15.4: x IS.
    sz_line_break_force_join_(eIS & INTERIOR, ALL, &settled);
    // LB16: (CL|CP) SP* NS.
    sz_line_break_force_join_((eNS & (cl_gov << 1)) & INTERIOR, ALL, &settled);
    // LB17: B2 SP* B2.
    sz_line_break_force_join_((eB2 & (b2_gov << 1)) & INTERIOR, ALL, &settled);
    // LB18: SP /.
    sz_line_break_force_break_((eSP << 1) & INTERIOR, ALL, &settled, &brk);
    // LB19 group (East-Asian-aware quotation).
    sz_line_break_force_join_((eQU & ~ePI) & INTERIOR, ALL, &settled);
    sz_line_break_force_join_(((eQU & ~ePF) << 1) & INTERIOR, ALL, &settled);
    sz_line_break_force_join_((eQU & ~(eEAW << 1)) & INTERIOR, ALL, &settled);
    sz_line_break_force_join_((eQU & (~(eEAW >> 1) | EOT_LANE)) & INTERIOR, ALL, &settled);
    sz_line_break_force_join_(((eQU << 1) & ~eEAW) & INTERIOR, ALL, &settled);
    sz_line_break_force_join_(((eQU << 1) & (~(eEAW << 2) | (SOT_BIT << 1))) & INTERIOR, ALL, &settled);
    // LB20: / CB ; CB /.
    sz_line_break_force_break_((eCB | (eCB << 1)) & INTERIOR, ALL, &settled, &brk);
    // LB20a: (sot|allowed) (HY|HH) x (AL|HL).
    {
        sz_u64_t hy_base = (eHY | eHH) & ~attached;
        sz_u64_t leftset = eBK | eCR | eLF | eNL | eSP | eZW | eCB | eGL;
        sz_u64_t left_ok = (leftset << 1) | SOT_BIT;
        sz_u64_t hy_ok = sz_line_break_over_marks_(hy_base & left_ok, attached);
        sz_line_break_force_join_(((hy_ok << 1) & (eAL | eHL)) & INTERIOR, ALL, &settled);
    }
    // LB21: x BA ; x HY ; x HH ; x NS ; BB x.
    sz_line_break_force_join_((eBA | eHY | eHH | eNS | (eBB << 1)) & INTERIOR, ALL, &settled);
    // LB21a: HL (HY|HH) x [^HL].
    {
        sz_u64_t hy_base = (eHY | eHH) & ~attached;
        sz_u64_t hl_hy = sz_line_break_over_marks_((eHL << 1) & hy_base, attached);
        sz_line_break_force_join_(((hl_hy << 1) & ~eHL) & INTERIOR, ALL, &settled);
    }
    // LB21b: SY x HL.
    sz_line_break_force_join_(((eSY << 1) & eHL) & INTERIOR, ALL, &settled);
    // LB22: x IN.
    sz_line_break_force_join_(eIN & INTERIOR, ALL, &settled);
    // LB23: (AL|HL) x NU ; NU x (AL|HL).
    sz_line_break_force_join_((((eAL | eHL) << 1) & eNU) & INTERIOR, ALL, &settled);
    sz_line_break_force_join_(((eNU << 1) & (eAL | eHL)) & INTERIOR, ALL, &settled);
    // LB23a: PR x (ID|EB|EM) ; (ID|EB|EM) x PO.
    sz_line_break_force_join_(((ePR << 1) & (eID | eEB | eEM)) & INTERIOR, ALL, &settled);
    sz_line_break_force_join_((((eID | eEB | eEM) << 1) & ePO) & INTERIOR, ALL, &settled);
    // LB24: (PR|PO) x (AL|HL) ; (AL|HL) x (PR|PO).
    sz_line_break_force_join_((((ePR | ePO) << 1) & (eAL | eHL)) & INTERIOR, ALL, &settled);
    sz_line_break_force_join_((((eAL | eHL) << 1) & (ePR | ePO)) & INTERIOR, ALL, &settled);
    // LB25: numeric clusters. Empty unless a NU lane is present (or an NU run straddles in via the carry).
    if (NU || carry.in_nu_run) {
        sz_u64_t syis = eSY | eIS;
        sz_u64_t nu_seed = eNU;
        if (carry.in_nu_run && ((eNU | syis) & 1ull)) nu_seed |= 1ull;
        sz_u64_t nu_run = sz_line_break_smear_right_(nu_seed | ((nu_seed << 1) & syis), nu_seed | syis) & (nu_seed | syis);
        sz_u64_t nu_run_close = (nu_run << 1) & (eCL | eCP);
        sz_line_break_force_join_(((nu_run_close << 1) & (ePO | ePR)) & INTERIOR, ALL, &settled);
        sz_line_break_force_join_(((nu_run << 1) & (ePO | ePR)) & INTERIOR, ALL, &settled);
        sz_line_break_force_join_(((ePO << 1) & eOP & (eNU >> 1)) & INTERIOR, ALL, &settled);
        sz_line_break_force_join_(((ePO << 1) & eOP & (eIS >> 1) & (eNU >> 2)) & INTERIOR, ALL, &settled);
        sz_line_break_force_join_(((ePO << 1) & eNU) & INTERIOR, ALL, &settled);
        sz_line_break_force_join_(((ePR << 1) & eOP & (eNU >> 1)) & INTERIOR, ALL, &settled);
        sz_line_break_force_join_(((ePR << 1) & eOP & (eIS >> 1) & (eNU >> 2)) & INTERIOR, ALL, &settled);
        sz_line_break_force_join_(((ePR << 1) & eNU) & INTERIOR, ALL, &settled);
        sz_line_break_force_join_(((eHY << 1) & eNU) & INTERIOR, ALL, &settled);
        sz_line_break_force_join_(((eIS << 1) & eNU) & INTERIOR, ALL, &settled);
        sz_line_break_force_join_(((nu_run << 1) & eNU) & INTERIOR, ALL, &settled);
    }
    // LB26 + LB27: Hangul syllable structure + PR/PO affixes. Empty unless a Hangul jamo/syllable lane is present.
    if (JL | JV | JT | H2 | H3) {
        sz_line_break_force_join_(((eJL << 1) & (eJL | eJV | eH2 | eH3)) & INTERIOR, ALL, &settled);
        sz_line_break_force_join_((((eJV | eH2) << 1) & (eJV | eJT)) & INTERIOR, ALL, &settled);
        sz_line_break_force_join_((((eJT | eH3) << 1) & eJT) & INTERIOR, ALL, &settled);
        sz_line_break_force_join_((((eJL | eJV | eJT | eH2 | eH3) << 1) & ePO) & INTERIOR, ALL, &settled);
        sz_line_break_force_join_(((ePR << 1) & (eJL | eJV | eJT | eH2 | eH3)) & INTERIOR, ALL, &settled);
    }
    // LB28: (AL|HL) x (AL|HL).
    sz_line_break_force_join_((((eAL | eHL) << 1) & (eAL | eHL)) & INTERIOR, ALL, &settled);
    // LB28a: aksara (Brahmic); DottedCircle acts as an aksara base. Empty unless a Brahmic/DottedCircle lane present.
    if (AK | AP | AS | VF | VI | DC) {
        sz_u64_t eDC = sz_line_break_eff_side_(DC, mark, base_good, attached);
        sz_u64_t base = eAK | eAS | eDC;
        sz_u64_t a_base = eAK | eAS | eDC;
        sz_u64_t a_base_dc = eAK | eDC;
        sz_line_break_force_join_(((eAP << 1) & a_base) & INTERIOR, ALL, &settled);
        sz_line_break_force_join_(((base << 1) & (eVF | eVI)) & INTERIOR, ALL, &settled);
        {
            sz_u64_t vi_base = eVI & ~attached;
            sz_u64_t base_vi = sz_line_break_over_marks_((base << 1) & vi_base, attached);
            sz_line_break_force_join_(((base_vi << 1) & a_base_dc) & INTERIOR, ALL, &settled);
        }
        sz_line_break_force_join_(((base << 1) & a_base & (eVF >> 1)) & INTERIOR, ALL, &settled);
    }
    // LB29: IS x (AL|HL).
    sz_line_break_force_join_(((eIS << 1) & (eAL | eHL)) & INTERIOR, ALL, &settled);
    // LB30: (AL|HL|NU) x OP[^EAW] ; CP[^EAW] x (AL|HL|NU).
    sz_line_break_force_join_((((eAL | eHL | eNU) << 1) & eOP & ~eEAW) & INTERIOR, ALL, &settled);
    sz_line_break_force_join_((((eCP & ~eEAW) << 1) & (eAL | eHL | eNU)) & INTERIOR, ALL, &settled);
    // LB30a: RI RI even-parity pairing. Empty unless an RI lane is present (a straddling RI run with no RI lane in
    // this block contributes no join, and the carry-out below is computed unconditionally).
    if (RI) {
        sz_u64_t ri_base = RI & ~attached;
        sz_u64_t ri_ext = sz_line_break_over_marks_(ri_base, attached);
        sz_u64_t parity_scan = ri_base;
        parity_scan ^= parity_scan << 1, parity_scan ^= parity_scan << 2, parity_scan ^= parity_scan << 4;
        parity_scan ^= parity_scan << 8, parity_scan ^= parity_scan << 16, parity_scan ^= parity_scan << 32;
        sz_u64_t run_start = ri_ext & ~(ri_ext << 1);
        sz_u64_t carry_in = sz_line_break_smear_right_((parity_scan << 1) & run_start, ri_ext);
        sz_u64_t par = (parity_scan ^ carry_in) & ri_ext;
        // Branchless cross-window odd-parity fold: when the inbound run is open with odd parity, flip lane 0's
        // run parity (the seed `1 & ri_ext` is empty unless lane 0 actually continues the RI run).
        sz_u64_t const carry_open_odd = (sz_u64_t)0 - (sz_u64_t)(carry.ri_open & carry.ri_parity_odd);
        sz_u64_t lane0_run = sz_line_break_smear_right_(1ull & ri_ext, ri_ext);
        par ^= lane0_run & ri_ext & carry_open_odd;
        sz_u64_t odd_left = par & ri_base;
        sz_u64_t reach = sz_line_break_over_marks_(odd_left, attached);
        sz_line_break_force_join_(((reach << 1) & ri_base) & INTERIOR, ALL, &settled);
    }
    // LB30b: EB x EM ; [ExtPict & Cn] x EM. Both joins require an EM lane on the right, so empty unless EM present.
    if (EM) {
        sz_line_break_force_join_(((eEB << 1) & eEM) & INTERIOR, ALL, &settled);
        sz_line_break_force_join_((((eEXT & eCN) << 1) & eEM) & INTERIOR, ALL, &settled);
    }

    // LB31: break everywhere else.
    brk |= INTERIOR & ~settled;

    // ---- Resume carry: run state entering `resume_lane` (O(1) bit tests on the segmented masks). ----
    if (carry_out) {
        sz_line_break_carry_t out;
        out.open_sp_opener = 0xFF, out.in_nu_run = 0, out.ri_open = 0, out.ri_parity_odd = 0;
        if (resume_lane > 0 && resume_lane <= count) {
            sz_size_t prev = resume_lane - 1;
            sz_u64_t pbit = (1ull << prev);
            if ((op_gov & pbit) && ((eSP | eOP) & pbit)) out.open_sp_opener = sz_line_break_op_k;
            else if ((cl_gov & pbit) && ((eSP | eCL | eCP) & pbit)) {
                sz_bool_t const prev_is_cp = (sz_bool_t)((CP & pbit) != 0);
                out.open_sp_opener = prev_is_cp ? (sz_u8_t)sz_line_break_cp_k : (sz_u8_t)sz_line_break_cl_k;
            }
            else if ((b2_gov & pbit) && ((eSP | eB2) & pbit)) out.open_sp_opener = sz_line_break_b2_k;
            else if ((zw_gov & pbit) && ((eSP | eZW) & pbit)) out.open_sp_opener = sz_line_break_zw_k;
            sz_u64_t syis2 = eSY | eIS;
            sz_u64_t nu_seed2 = eNU;
            if (carry.in_nu_run && ((eNU | syis2) & 1ull)) nu_seed2 |= 1ull;
            sz_u64_t nu_run2 = sz_line_break_smear_right_(nu_seed2 | ((nu_seed2 << 1) & syis2), nu_seed2 | syis2) & (nu_seed2 | syis2);
            out.in_nu_run = (sz_u8_t)((nu_run2 & pbit) != 0);
            sz_u64_t ri_base2 = RI & ~attached, ri_ext2 = sz_line_break_over_marks_(ri_base2, attached);
            sz_u64_t parity_scan2 = ri_base2;
            parity_scan2 ^= parity_scan2 << 1, parity_scan2 ^= parity_scan2 << 2, parity_scan2 ^= parity_scan2 << 4;
            parity_scan2 ^= parity_scan2 << 8, parity_scan2 ^= parity_scan2 << 16, parity_scan2 ^= parity_scan2 << 32;
            sz_u64_t rs = ri_ext2 & ~(ri_ext2 << 1);
            sz_u64_t cin = sz_line_break_smear_right_((parity_scan2 << 1) & rs, ri_ext2);
            sz_u64_t par2 = (parity_scan2 ^ cin) & ri_ext2;
            if (carry.ri_open && carry.ri_parity_odd && (ri_ext2 & 1ull))
                par2 ^= sz_line_break_smear_right_(1ull & ri_ext2, ri_ext2) & ri_ext2;
            out.ri_open = (sz_u8_t)((ri_ext2 & pbit) != 0);
            out.ri_parity_odd = (sz_u8_t)((par2 & pbit) != 0);
        }
        *carry_out = out;
    }

    sz_line_break_decided_t decided;
    decided.brk = brk;
    return decided;
}

#pragma endregion Mask-algebra rule engine

#pragma region Forward driver

/**
 *  @brief  Forward UAX-14 line-break-opportunity kernel (Ice Lake AVX-512).
 *
 *  Bit-exact with `sz_utf8_linewraps_serial`. Emits every UAX-14 break opportunity (no per-segment
 *  mandatory flag). The classifier and rule engine are fully vectorized (no per-lane scalar loop, no `vpgather`,
 *  no serial-oracle deferral); the 64-codepoint block engine threads cross-block state through a register carry
 *  (no halo back-scan), so throughput is flat in run length. Emits at most @p capacity segments; sets
 *  *@p bytes_consumed to the resume.
 */
SZ_PUBLIC sz_size_t sz_utf8_linewraps_icelake( //
    sz_cptr_t text, sz_size_t length,                     //
    sz_size_t *starts, sz_size_t *lengths,               //
    sz_size_t capacity, sz_size_t *bytes_consumed) {

    sz_size_t lines = 0;
    if (length == 0 || capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_size_t byte_offset[sz_utf8_line_icelake_window_k];
    sz_u32_t codepoint[sz_utf8_line_icelake_window_k];
    sz_u8_t class_bytes[sz_utf8_line_icelake_window_k + 64];
    sz_u8_t side_bytes[sz_utf8_line_icelake_window_k + 64];
    sz_u8_t break_before[sz_utf8_line_icelake_window_k];

    sz_size_t line_start = 0; /* absolute byte offset anchoring the current open line */
    while (line_start < length && lines < capacity) {

        /*  Decode a window of codepoints starting at `line_start`. The UTF-8 length-advance + decode is the
         *  inherent serial input stage (it mirrors the serial routine's decode); classification and the rule
         *  engine that follow are fully vectorized. */
        sz_size_t count = 0;
        sz_size_t position = line_start;
        sz_bool_t window_full = sz_false_k;
        while (position < length) {
            if (count == sz_utf8_line_icelake_window_k) {
                window_full = sz_true_k;
                break;
            }
            byte_offset[count] = position;
            sz_size_t decode = position;
            codepoint[count] = (sz_u32_t)sz_utf8_decode_(text, length, &decode);
            position = decode > position ? decode : position + 1;
            ++count;
        }
        sz_size_t const window_end = position;
        sz_bool_t const window_is_tail = (sz_bool_t)(!window_full && window_end == length);

        sz_u64_t dotted_local[(sz_utf8_line_icelake_window_k + 63) / 64];
        sz_size_t const used_words = (count + 63) / 64;
        for (sz_size_t word = 0; word < used_words; ++word) dotted_local[word] = 0;
        sz_line_break_classify_block_(codepoint, count, class_bytes, side_bytes, dotted_local);

        /*  LB9/LB10 cluster collapse: a combining mark that attaches to a preceding base is dropped so the rule
         *  engine sees exactly one lane per cluster -- a mark must never act as an independent rule participant
         *  (otherwise a mark resolved to its base's class would re-trigger a pair rule, e.g. a quote's trailing
         *  mark spuriously governing the following break). A lone mark (LB10) is kept but reclassified to AL.
         *  `kept_orig` maps each surviving lane back to its codepoint index; dropped marks inherit "no break
         *  before" (LB9) below. */
        sz_u8_t kept_class[sz_utf8_line_icelake_window_k];
        sz_u8_t kept_side[sz_utf8_line_icelake_window_k];
        sz_u32_t kept_orig[sz_utf8_line_icelake_window_k];
        sz_u64_t kept_dotted[(sz_utf8_line_icelake_window_k + 63) / 64];
        for (sz_size_t word = 0; word < used_words; ++word) kept_dotted[word] = 0;
        for (sz_size_t i = 0; i < count; ++i) break_before[i] = 0;
        sz_size_t count_c = 0;
        sz_bool_t base_attachable = sz_false_k;
        for (sz_size_t i = 0; i < count; ++i) {
            sz_u8_t cls = class_bytes[i];
            sz_bool_t const is_mark = (sz_bool_t)(cls == sz_line_break_cm_k || cls == sz_line_break_zwj_k);
            sz_u8_t kept_cls = cls, kept_sd = side_bytes[i];
            if (is_mark) {
                if (base_attachable) continue;                                          // LB9: attached -> collapse
                kept_cls = sz_line_break_al_k, kept_sd = 0, base_attachable = sz_true_k; // LB10: lone mark acts as AL
            }
            else
                base_attachable = (sz_bool_t)(cls != sz_line_break_bk_k && cls != sz_line_break_cr_k &&
                                              cls != sz_line_break_lf_k && cls != sz_line_break_nl_k &&
                                              cls != sz_line_break_sp_k && cls != sz_line_break_zw_k);
            if ((dotted_local[i >> 6] >> (i & 63)) & 1ull) kept_dotted[count_c >> 6] |= 1ull << (count_c & 63);
            kept_class[count_c] = kept_cls, kept_side[count_c] = kept_sd, kept_orig[count_c] = (sz_u32_t)i, ++count_c;
        }
        if (count_c == 0) {
            line_start = window_end;
            continue;
        }

        /*  64-base block engine over the collapsed stream. The carry threads all unbounded run state across the
         *  window_edge; a constant LEFT_CONTEXT margin gives the first committed lane its bounded left neighbours
         *  in-register. No variable-length halo back-scan -> O(1) per block, flat throughput in run length. */
        sz_line_break_carry_t carry;
        carry.open_sp_opener = 0xFF, carry.in_nu_run = 0, carry.ri_open = 0, carry.ri_parity_odd = 0;
        sz_size_t commit_from = 1;
        while (commit_from < count_c) {
            sz_size_t lo = (commit_from > sz_utf8_line_icelake_left_context_k)
                               ? (commit_from - sz_utf8_line_icelake_left_context_k)
                               : 0;
            sz_bool_t const window_edge = (sz_bool_t)(lo > 0);
            sz_size_t block_end = lo + 64;
            if (block_end > count_c) block_end = count_c;
            sz_size_t block_count = block_end - lo;
            sz_size_t commit_end =
                (block_end == count_c) ? block_end : (block_end - sz_utf8_line_icelake_right_look_k);
            if (commit_end <= commit_from) commit_end = commit_from + 1;
            if (commit_end > block_end) commit_end = block_end;

            sz_line_break_carry_t use_carry;
            if (window_edge) use_carry = carry;
            else use_carry.open_sp_opener = 0xFF, use_carry.in_nu_run = 0, use_carry.ri_open = 0,
                 use_carry.ri_parity_odd = 0;

            /*  Next block anchors at (commit_end - LEFT_CONTEXT); report the carry for THAT lane so it is valid
             *  as the next block's lane 0. */
            sz_size_t next_lo = (commit_end > sz_utf8_line_icelake_left_context_k)
                                    ? (commit_end - sz_utf8_line_icelake_left_context_k)
                                    : 0;
            sz_size_t emit_carry_lane = next_lo - lo;

            //  Load the block's class/side bytes into ZMM (tail lanes load as class 0xFF / side 0).
            __mmask64 const load_mask = sz_u64_mask_until_(block_count);
            __m512i const block_classes = _mm512_mask_loadu_epi8(_mm512_set1_epi8((char)0xFF), load_mask, kept_class + lo);
            __m512i const block_side = _mm512_maskz_loadu_epi8(load_mask, kept_side + lo);
            /*  The block's DottedCircle bits are a contiguous 64-bit slice of the collapsed bitmap starting at
             *  `lo`; extract it branch/loop-free across the (at most) two spanning words (no gather). */
            sz_size_t const dotted_word = lo >> 6;
            sz_size_t const dotted_shift = lo & 63;
            sz_u64_t block_dotted = kept_dotted[dotted_word] >> dotted_shift;
            if (dotted_shift && (dotted_word + 1) < (sz_size_t)((sz_utf8_line_icelake_window_k + 63) / 64))
                block_dotted |= kept_dotted[dotted_word + 1] << (64 - dotted_shift);
            block_dotted &= sz_u64_mask_until_(block_count);

            sz_line_break_carry_t next_carry;
            sz_line_break_decided_t decided = sz_line_break_decide_block_(
                block_classes, block_side, block_dotted, block_count,
                /*is_sot=*/(sz_bool_t)(lo == 0), /*is_eot=*/(sz_bool_t)(block_end == count_c), use_carry,
                emit_carry_lane, &next_carry);

            for (sz_size_t i = commit_from; i < commit_end; ++i) {
                sz_size_t const orig = kept_orig[i];
                sz_u8_t bit = (sz_u8_t)((decided.brk >> (i - lo)) & 1);
                // LB8a: ZWJ x. The collapse folds a ZWJ into its base (LB9), but LB8a -- "no break after ZWJ" --
                // is a separate, earlier rule that the collapse must not lose, so suppress the break before any
                // base whose immediately preceding codepoint is a ZWJ.
                if (orig > 0 && class_bytes[orig - 1] == sz_line_break_zwj_k) bit = 0;
                break_before[orig] = bit;
            }
            carry = next_carry;
            commit_from = commit_end;
        }

        //  Emit committed opportunities, mirroring the serial commit policy.
        sz_size_t local_line_start = line_start;
        sz_bool_t hit_capacity = sz_false_k;
        for (sz_size_t k = 1; k < count; ++k) {
            if (!break_before[k]) continue;
            if (!(window_is_tail || k + 2 < count)) continue;
            if (lines == capacity) {
                hit_capacity = sz_true_k;
                break;
            }
            starts[lines] = local_line_start;
            lengths[lines] = byte_offset[k] - local_line_start;
            ++lines;
            local_line_start = byte_offset[k];
        }

        if (hit_capacity) {
            if (bytes_consumed) *bytes_consumed = local_line_start;
            return lines;
        }
        if (window_is_tail) {
            if (lines == capacity) {
                if (bytes_consumed) *bytes_consumed = local_line_start;
                return lines;
            }
            starts[lines] = local_line_start;
            lengths[lines] = length - local_line_start;
            ++lines;
            if (bytes_consumed) *bytes_consumed = length;
            return lines;
        }
        line_start = (local_line_start > line_start) ? local_line_start : window_end;
    }

    if (bytes_consumed) *bytes_consumed = line_start;
    return lines;
}

#pragma endregion Forward driver

#pragma endregion UAX-14 Line Boundaries forward kernel
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_ICELAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_LINES_ICELAKE_H_
