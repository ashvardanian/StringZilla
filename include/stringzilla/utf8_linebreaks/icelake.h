/**
 *  @brief Ice Lake backend for UAX-14 line break boundaries.
 *  @file include/stringzilla/utf8_linebreaks/icelake.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_LINEBREAKS_ICELAKE_H_
#define STRINGZILLA_UTF8_LINEBREAKS_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_linebreaks/tables.h"
#include "stringzilla/utf8_linebreaks/serial.h"
#include "stringzilla/utf8_runes/icelake.h"

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

#pragma region UAX 14 Line Boundaries forward kernel

#pragma region In register vectorized classifier

/*  The classifier resolves a contiguous run of codepoints to per-codepoint (class, side, dotted) with ZERO
 *  per-lane scalar loop, ZERO `vpgather`, and NO serial `sz_rune_line_break_property` deferral. It processes
 *  sixteen 32-bit codepoint lanes per pass: each codepoint is mapped to a PALETTE INDEX in-register (page-LUT
 *  for cp<0x800, the shared substrate-style two-stage trie for 0x800..0xFFFF, arithmetic range compares for
 *  the 50 big homogeneous blocks and the 618 astral ranges), then the 62-entry palette descriptor is unpacked
 *  to the LB1-resolved class byte and the engine side byte by in-register compares and masked moves. The
 *  resolution precedence mirrors `sz_rune_line_break_property`: a big-range hit (first match in array order)
 *  overrides everything; else the page-LUT; else the trie; else an astral range; else palette[0]. */

/** @brief Palette index for sixteen astral (>=0x10000) codepoints via a register-resident 8/4/4/4 trie over
 *         offset = codepoint - 0x10000 (s0 -> s1 -> s2 -> leaf). Re-init-free: every tile is read straight from
 *         aligned .rodata through the substrate permute256_/lut_cascade_ helpers. Bit-exact with the legacy
 *         618-range linear astral fold; replaces that per-window scan. */
SZ_HELPER_AUTO __m512i sz_line_break_classify_astral16_icelake_(__m512i codepoints) {
    __m512i const offset = _mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x10000));
    __m512i const stage1 = sz_utf8_rune_permute256_icelake_(
        sz_utf8_line_break_astral_s0_, _mm512_and_si512(_mm512_srli_epi32(offset, 12), _mm512_set1_epi32(0xFF)));
    __m512i const stage2_index = _mm512_add_epi32(
        _mm512_slli_epi32(stage1, 4), _mm512_and_si512(_mm512_srli_epi32(offset, 8), _mm512_set1_epi32(0xF)));
    __m512i const stage2 = sz_utf8_rune_lut_cascade_icelake_(sz_utf8_line_break_astral_s1_,
                                                             (int)sz_utf8_line_break_astral_s1_tiles_k, stage2_index);
    __m512i const leaf_index = _mm512_add_epi32(_mm512_slli_epi32(stage2, 4),
                                                _mm512_and_si512(_mm512_srli_epi32(offset, 4), _mm512_set1_epi32(0xF)));
    __m512i const leaf = sz_utf8_rune_lut_cascade_icelake_(sz_utf8_line_break_astral_s2_,
                                                           (int)sz_utf8_line_break_astral_s2_tiles_k, leaf_index);
    __m512i const class_index = _mm512_add_epi32(_mm512_slli_epi32(leaf, 4),
                                                 _mm512_and_si512(offset, _mm512_set1_epi32(0xF)));
    return sz_utf8_rune_lut_cascade_icelake_(sz_utf8_line_break_astral_leaf_,
                                             (int)sz_utf8_line_break_astral_leaf_tiles_k, class_index);
}

/** @brief All-64-lane complete-BMP palette index for `cp < 0x10000` over the full-BMP trie (offset = the codepoint
 *         itself, no `-0x800` subtract), so a SINGLE pass resolves the whole BMP -- the page LUT and the 0x800 split
 *         are gone. This mirrors `sz_utf8_rune_trie_walk_icelake_` exactly with `block = superblock = 8` and
 *         `offset = (high << 8) | low`; the substrate function bakes in a hard `-0x800`, so the zero-base form lives
 *         here. Lanes whose codepoint is >= 0x10000 are undefined (the caller blends in the astral path). */
SZ_HELPER_AUTO __m512i sz_line_break_bmp_full_index_icelake_(__m512i high, __m512i low) {
    __m128i const block_log2 = _mm_cvtsi32_si128(sz_u64_ctz((sz_u64_t)sz_utf8_line_break_trie_block_k));
    __m128i const super_log2 = _mm_cvtsi32_si128(sz_u64_ctz((sz_u64_t)sz_utf8_line_break_trie_subblock_k));
    __m512i const within_mask = _mm512_set1_epi16((short)(sz_utf8_line_break_trie_block_k - 1));
    __m512i const super_off_mask = _mm512_set1_epi16((short)(sz_utf8_line_break_trie_subblock_k - 1));
    __m512i const super_v16 = _mm512_set1_epi16((short)sz_utf8_line_break_trie_subblock_k);
    __m512i const block_v16 = _mm512_set1_epi16((short)sz_utf8_line_break_trie_block_k);

    __m512i const zero = _mm512_setzero_si512();
    __m512i const offset_lo = _mm512_or_si512(_mm512_slli_epi16(_mm512_unpacklo_epi8(high, zero), 8),
                                              _mm512_unpacklo_epi8(low, zero));
    __m512i const offset_hi = _mm512_or_si512(_mm512_slli_epi16(_mm512_unpackhi_epi8(high, zero), 8),
                                              _mm512_unpackhi_epi8(low, zero));

    __m512i const within_lo = _mm512_and_si512(offset_lo, within_mask);
    __m512i const within_hi = _mm512_and_si512(offset_hi, within_mask);
    __m512i const block_idx_lo = _mm512_srl_epi16(offset_lo, block_log2);
    __m512i const block_idx_hi = _mm512_srl_epi16(offset_hi, block_log2);
    __m512i const super_off_lo = _mm512_and_si512(block_idx_lo, super_off_mask);
    __m512i const super_off_hi = _mm512_and_si512(block_idx_hi, super_off_mask);
    __m512i const super_lo = _mm512_srl_epi16(block_idx_lo, super_log2);
    __m512i const super_hi = _mm512_srl_epi16(block_idx_hi, super_log2);

    __m512i const level1_lo = sz_utf8_rune_gather_byte_(sz_utf8_line_break_bmp_full_trie_l1_,
                                                        (int)sz_utf8_line_break_bmp_full_trie_l1_count_k, super_lo);
    __m512i const level1_hi = sz_utf8_rune_gather_byte_(sz_utf8_line_break_bmp_full_trie_l1_,
                                                        (int)sz_utf8_line_break_bmp_full_trie_l1_count_k, super_hi);

    __m512i const l2_index_lo = _mm512_add_epi16(_mm512_mullo_epi16(level1_lo, super_v16), super_off_lo);
    __m512i const l2_index_hi = _mm512_add_epi16(_mm512_mullo_epi16(level1_hi, super_v16), super_off_hi);
    __m512i const leaf_idx_lo = sz_utf8_rune_gather_word_(
        sz_utf8_line_break_bmp_full_trie_l2_, (int)sz_utf8_line_break_bmp_full_trie_l2_count_k, l2_index_lo);
    __m512i const leaf_idx_hi = sz_utf8_rune_gather_word_(
        sz_utf8_line_break_bmp_full_trie_l2_, (int)sz_utf8_line_break_bmp_full_trie_l2_count_k, l2_index_hi);

    __m512i const leaf_byte_lo = _mm512_add_epi16(_mm512_mullo_epi16(leaf_idx_lo, block_v16), within_lo);
    __m512i const leaf_byte_hi = _mm512_add_epi16(_mm512_mullo_epi16(leaf_idx_hi, block_v16), within_hi);
    __m512i const class_lo = sz_utf8_rune_gather_byte_(
        sz_utf8_line_break_bmp_full_trie_leaf_, (int)sz_utf8_line_break_bmp_full_trie_leaf_count_k, leaf_byte_lo);
    __m512i const class_hi = sz_utf8_rune_gather_byte_(
        sz_utf8_line_break_bmp_full_trie_leaf_, (int)sz_utf8_line_break_bmp_full_trie_leaf_count_k, leaf_byte_hi);

    return _mm512_packus_epi16(class_lo, class_hi);
}

/** @brief Start-compacting complete-BMP palette index: the rule engine reads classes only at codepoint-START lanes,
 *         and the dense 3-byte scripts that dominate the trie cost carry at most 32 starts per 64-byte window, so the
 *         start lanes' high/low bytes are `vpcompressb`-compacted into the low <=32 lanes, widened to ONE 32x16-bit
 *         register, walked through the full-BMP trie a SINGLE time (the L1/L2/leaf gathers run once, not the lo+hi
 *         twice of @ref sz_line_break_bmp_full_index_icelake_), then the per-start class bytes are
 *         `vpexpandb`-scattered back to their original byte-lane positions. Windows with more than 32 starts (dense
 *         2-byte scripts, ASCII-interleaved runs) cannot fit one 16-bit register and fall back to the unconditional
 *         two-pass walk, which costs no more than today. Continuation lanes hold an undefined index (never consumed:
 *         `base` is a subset of `starts`). Bit-identical to the two-pass walk on every start lane. */
SZ_HELPER_AUTO __m512i sz_line_break_bmp_full_index_compact_icelake_(__m512i high, __m512i low, sz_u64_t starts) {
    if (_mm_popcnt_u64(starts) > 32) return sz_line_break_bmp_full_index_icelake_(high, low);
    __mmask64 const start_mask = _cvtu64_mask64(starts);
    __m512i const high_packed = _mm512_maskz_compress_epi8(start_mask, high);
    __m512i const low_packed = _mm512_maskz_compress_epi8(start_mask, low);

    __m128i const block_log2 = _mm_cvtsi32_si128(sz_u64_ctz((sz_u64_t)sz_utf8_line_break_trie_block_k));
    __m128i const super_log2 = _mm_cvtsi32_si128(sz_u64_ctz((sz_u64_t)sz_utf8_line_break_trie_subblock_k));
    __m512i const within_mask = _mm512_set1_epi16((short)(sz_utf8_line_break_trie_block_k - 1));
    __m512i const super_off_mask = _mm512_set1_epi16((short)(sz_utf8_line_break_trie_subblock_k - 1));
    __m512i const super_v16 = _mm512_set1_epi16((short)sz_utf8_line_break_trie_subblock_k);
    __m512i const block_v16 = _mm512_set1_epi16((short)sz_utf8_line_break_trie_block_k);

    //  The <=32 compacted starts sit in the low 32 bytes; widen them to a single 32x16-bit codepoint register.
    __m512i const offset = _mm512_or_si512(
        _mm512_slli_epi16(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(high_packed)), 8),
        _mm512_cvtepu8_epi16(_mm512_castsi512_si256(low_packed)));

    __m512i const within = _mm512_and_si512(offset, within_mask);
    __m512i const block_idx = _mm512_srl_epi16(offset, block_log2);
    __m512i const super_off = _mm512_and_si512(block_idx, super_off_mask);
    __m512i const super = _mm512_srl_epi16(block_idx, super_log2);

    __m512i const level1 = sz_utf8_rune_gather_byte_(sz_utf8_line_break_bmp_full_trie_l1_,
                                                     (int)sz_utf8_line_break_bmp_full_trie_l1_count_k, super);
    __m512i const l2_index = _mm512_add_epi16(_mm512_mullo_epi16(level1, super_v16), super_off);
    __m512i const leaf_idx = sz_utf8_rune_gather_word_(sz_utf8_line_break_bmp_full_trie_l2_,
                                                       (int)sz_utf8_line_break_bmp_full_trie_l2_count_k, l2_index);
    __m512i const leaf_byte = _mm512_add_epi16(_mm512_mullo_epi16(leaf_idx, block_v16), within);
    __m512i const class_word = sz_utf8_rune_gather_byte_(sz_utf8_line_break_bmp_full_trie_leaf_,
                                                         (int)sz_utf8_line_break_bmp_full_trie_leaf_count_k, leaf_byte);

    //  Narrow the 32 class words back to 32 contiguous bytes (low half), then scatter to the original start lanes.
    __m512i const class_packed = _mm512_castsi256_si512(_mm512_cvtepi16_epi8(class_word));
    return _mm512_maskz_expand_epi8(start_mask, class_packed);
}

/** @brief Unpack thirty-two 16-bit palette descriptors (one half of the 64-byte block) to the LB1-resolved class and
 *         the engine side byte, both as 16-bit lanes, plus the per-lane DottedCircle predicate. Applies the serial
 *         resolution aliasing (SA->AL/CM, AI/SG/XX->AL, CJ->NS); Pi/Pf/EAW/Cn|Ext side bits come from descriptor
 *         bits 6/7/8/9; RI/ZWJ side from the raw class; CM|ZWJ -> mark side bit; DottedCircle from bit 13. */
SZ_HELPER_INLINE void sz_line_break_descriptor_unpack_half_icelake_(__m512i descriptors, __m512i *classes_out,
                                                                    __m512i *side_out, __mmask32 *dotted_out) {
    __m512i classes = _mm512_and_si512(descriptors, _mm512_set1_epi16(0x3F));
    __mmask32 const is_sa = _mm512_cmpeq_epi16_mask(classes, _mm512_set1_epi16(sz_line_break_sa_k));
    __mmask32 const sa_is_mark = _mm512_test_epi16_mask(descriptors, _mm512_set1_epi16(1 << 12));
    classes = _mm512_mask_mov_epi16(classes, is_sa, _mm512_set1_epi16(sz_line_break_al_k));
    classes = _mm512_mask_mov_epi16(classes, is_sa & sa_is_mark, _mm512_set1_epi16(sz_line_break_cm_k));
    __mmask32 const is_alias = _mm512_cmpeq_epi16_mask(classes, _mm512_set1_epi16(sz_line_break_ai_k)) |
                               _mm512_cmpeq_epi16_mask(classes, _mm512_set1_epi16(sz_line_break_sg_k)) |
                               _mm512_cmpeq_epi16_mask(classes, _mm512_set1_epi16(sz_line_break_xx_k));
    classes = _mm512_mask_mov_epi16(classes, is_alias, _mm512_set1_epi16(sz_line_break_al_k));
    __mmask32 const is_cj = _mm512_cmpeq_epi16_mask(classes, _mm512_set1_epi16(sz_line_break_cj_k));
    classes = _mm512_mask_mov_epi16(classes, is_cj, _mm512_set1_epi16(sz_line_break_ns_k));

    __m512i side = _mm512_setzero_si512();
    side = _mm512_or_si512(side, _mm512_maskz_mov_epi16(_mm512_test_epi16_mask(descriptors, _mm512_set1_epi16(1 << 6)),
                                                        _mm512_set1_epi16(sz_line_break_side_pi_k)));
    side = _mm512_or_si512(side, _mm512_maskz_mov_epi16(_mm512_test_epi16_mask(descriptors, _mm512_set1_epi16(1 << 7)),
                                                        _mm512_set1_epi16(sz_line_break_side_pf_k)));
    side = _mm512_or_si512(side, _mm512_maskz_mov_epi16(_mm512_test_epi16_mask(descriptors, _mm512_set1_epi16(1 << 8)),
                                                        _mm512_set1_epi16(sz_line_break_side_eaw_k)));
    side = _mm512_or_si512(
        side, _mm512_maskz_mov_epi16(_mm512_test_epi16_mask(descriptors, _mm512_set1_epi16(1 << 9)),
                                     _mm512_set1_epi16(sz_line_break_side_cn_k | sz_line_break_side_ext_k)));
    __m512i const raw_class = _mm512_and_si512(descriptors, _mm512_set1_epi16(0x3F));
    side = _mm512_or_si512(
        side, _mm512_maskz_mov_epi16(_mm512_cmpeq_epi16_mask(raw_class, _mm512_set1_epi16(sz_line_break_ri_k)),
                                     _mm512_set1_epi16(sz_line_break_side_ri_k)));
    side = _mm512_or_si512(
        side, _mm512_maskz_mov_epi16(_mm512_cmpeq_epi16_mask(raw_class, _mm512_set1_epi16(sz_line_break_zwj_k)),
                                     _mm512_set1_epi16(sz_line_break_side_zwj_k)));
    __mmask32 const class_is_mark = _mm512_cmpeq_epi16_mask(classes, _mm512_set1_epi16(sz_line_break_cm_k)) |
                                    _mm512_cmpeq_epi16_mask(classes, _mm512_set1_epi16(sz_line_break_zwj_k));
    side = _mm512_or_si512(side, _mm512_maskz_mov_epi16(class_is_mark, _mm512_set1_epi16(sz_line_break_side_mark_k)));

    *classes_out = classes;
    *side_out = side;
    *dotted_out = _mm512_test_epi16_mask(descriptors, _mm512_set1_epi16(1 << 13));
}

/** @brief Per-window byte-lane classification: class/side per lane, plus the effective-start and U+FFFD masks. */
typedef struct sz_line_break_classified_t {
    __m512i classes;      /**< Per-byte-lane Line_Break class (valid only on `starts` lanes). */
    __m512i side;         /**< Per-byte-lane engine side byte. */
    sz_u64_t dotted;      /**< Bit i set => lane i is DottedCircle U+25CC. */
    sz_u64_t starts;      /**< Effective codepoint starts: valid leads (at their lane) + 1-byte U+FFFD units. */
    sz_u64_t replacement; /**< Effective-start lanes that are ill-formed (decoded as U+FFFD, class AL). */
    sz_u64_t non_start;   /**< Bytes that are NOT effective starts (consumed continuations) within `loaded`. */
    sz_size_t loaded;     /**< Bytes loaded into this window (<= 64). */
} sz_line_break_classified_t;

/**
 *  @brief  Classify a decoded 64-byte window onto byte-start lanes, fully in-register and zero-scalar. Reproduces
 *          the serial `sz_utf8_next_rune_` "consume-1 U+FFFD" policy: an invalid lead, a short/stray continuation, an
 *          overlong / surrogate / out-of-range lead each become a single-byte U+FFFD unit (class AL), so serial and
 *          icelake agree on malformed input. Valid leads classify by decoded VALUE (page / trie / big / astral),
 *          matching the serial resolution precedence. The BMP trie uses the shared substrate `trie_walk_icelake_`.
 */
SZ_HELPER_AUTO sz_line_break_classified_t sz_line_break_classify_window_icelake_(sz_utf8_rune_window_t window,
                                                                                 __m512i lane_identity) {
    sz_u64_t const loaded_mask = sz_u64_mask_until_(window.loaded);
    sz_u64_t const continuation = _cvtmask64_u64(window.continuation) & loaded_mask;
    sz_u64_t const two_byte = _cvtmask64_u64(window.two_byte_starts);
    sz_u64_t const three_byte = _cvtmask64_u64(window.three_byte_starts);
    sz_u64_t const four_byte = _cvtmask64_u64(window.four_byte_starts);
    __m512i const raw = window.window;
    __m512i const next1 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(1)), raw);

    //  "next k lanes are continuations" tests, shifted into each lead's lane.
    sz_u64_t const next1_continuation = continuation >> 1, next2_continuation = continuation >> 2,
                   next3_continuation = continuation >> 3;

    //  Ill-formed-lead gate (LEVER B): the overlong / surrogate / out-of-range value checks only fire on the lead
    //  bytes C0/C1 (overlong-2), E0/ED (overlong-3 / surrogate-3) and F0/F4/>=F5 (overlong-4 / above-range-4). Real
    //  text never carries those, so detect their presence with ONE `raw`-only test and, when absent, take the cheap
    //  "lead + enough continuations" validity path. The gated branch runs the exact same overlong/surrogate/above
    //  algebra as before, so the result is bit-identical regardless of which side fires.
    sz_u64_t const lead_c0_c1 = _cvtmask64_u64(
        _kand_mask64(_mm512_cmp_epu8_mask(raw, _mm512_set1_epi8((char)0xC0), _MM_CMPINT_NLT),
                     _mm512_cmp_epu8_mask(raw, _mm512_set1_epi8((char)0xC1), _MM_CMPINT_LE)));
    sz_u64_t const lead_e0 = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(raw, _mm512_set1_epi8((char)0xE0)));
    sz_u64_t const lead_ed = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(raw, _mm512_set1_epi8((char)0xED)));
    sz_u64_t const lead_f0 = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(raw, _mm512_set1_epi8((char)0xF0)));
    sz_u64_t const lead_f4_or_above = _cvtmask64_u64(
        _mm512_cmp_epu8_mask(raw, _mm512_set1_epi8((char)0xF4), _MM_CMPINT_NLT));
    sz_u64_t const danger_leads = (lead_c0_c1 | lead_e0 | lead_ed | lead_f0 | lead_f4_or_above) & loaded_mask;

    sz_u64_t valid2, valid3, valid4;
    if (!danger_leads) {
        //  Common case: no overlong / surrogate / out-of-range leads -> validity is purely "lead + continuations".
        valid2 = two_byte & next1_continuation;
        valid3 = three_byte & next1_continuation & next2_continuation;
        valid4 = four_byte & next1_continuation & next2_continuation & next3_continuation;
    }
    else {
        //  Per-lane value-bit predicates for overlong / surrogate / out-of-range detection (exact prior algebra).
        sz_u64_t const lead_not_overlong2 = _cvtmask64_u64(_mm512_test_epi8_mask(raw, _mm512_set1_epi8(0x1E)));
        sz_u64_t const b0_e0 = lead_e0;
        sz_u64_t const b0_ed = lead_ed;
        sz_u64_t const b0_f0 = lead_f0;
        sz_u64_t const b0_f4 = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(raw, _mm512_set1_epi8((char)0xF4)));
        sz_u64_t const b0_above_f4 = _cvtmask64_u64(
            _mm512_cmp_epu8_mask(raw, _mm512_set1_epi8((char)0xF4), _MM_CMPINT_NLE));
        sz_u64_t const b1_lt_a0 = _cvtmask64_u64(_mm512_cmplt_epu8_mask(next1, _mm512_set1_epi8((char)0xA0)));
        sz_u64_t const b1_ge_a0 = _cvtmask64_u64(
            _mm512_cmp_epu8_mask(next1, _mm512_set1_epi8((char)0xA0), _MM_CMPINT_NLT));
        sz_u64_t const b1_lt_90 = _cvtmask64_u64(_mm512_cmplt_epu8_mask(next1, _mm512_set1_epi8((char)0x90)));
        sz_u64_t const b1_ge_90 = _cvtmask64_u64(
            _mm512_cmp_epu8_mask(next1, _mm512_set1_epi8((char)0x90), _MM_CMPINT_NLT));

        sz_u64_t const overlong3 = three_byte & b0_e0 & b1_lt_a0;
        sz_u64_t const surrogate3 = three_byte & b0_ed & b1_ge_a0;
        sz_u64_t const overlong4 = four_byte & b0_f0 & b1_lt_90;
        sz_u64_t const above4 = four_byte & ((b0_f4 & b1_ge_90) | b0_above_f4);
        valid2 = two_byte & next1_continuation & lead_not_overlong2;
        valid3 = three_byte & next1_continuation & next2_continuation & ~overlong3 & ~surrogate3;
        valid4 = four_byte & next1_continuation & next2_continuation & next3_continuation & ~overlong4 & ~above4;
    }
    sz_u64_t const true_ascii = _cvtmask64_u64(_mm512_cmplt_epu8_mask(raw, _mm512_set1_epi8((char)0x80))) &
                                loaded_mask & ~continuation;

    //  `decode_window_` reconstructs high/low with the 2/3-byte formula only: correct for 2/3-byte leads, garbage on
    //  ASCII lanes (codepoint IS the raw byte) and on 4-byte leads (true low 16 bits come from b1..b3). Rebuild both
    //  halves per lead length so cp = (plane<<16)|(high<<8)|low is exact. (Replacement lanes are overridden to
    //  U+FFFD downstream, so their garbage never reaches the palette.)
    __m512i const next2 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(2)), raw);
    __m512i const next3 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(3)), raw);
    __m512i const low_four = _mm512_or_si512(
        _mm512_and_si512(_mm512_slli_epi16(_mm512_and_si512(next2, _mm512_set1_epi8(0x03)), 6),
                         _mm512_set1_epi8((char)0xC0)),
        _mm512_and_si512(next3, _mm512_set1_epi8(0x3F)));
    __m512i const high_four = _mm512_or_si512(
        _mm512_and_si512(_mm512_slli_epi16(_mm512_and_si512(next1, _mm512_set1_epi8(0x0F)), 4),
                         _mm512_set1_epi8((char)0xF0)),
        sz_utf8_srl8_icelake_(next2, 2, 0x0F));
    __m512i low_fixed = _mm512_mask_mov_epi8(window.low, _cvtu64_mask64(true_ascii), raw);
    low_fixed = _mm512_mask_mov_epi8(low_fixed, _cvtu64_mask64(four_byte), low_four);
    __m512i high_fixed = _mm512_maskz_mov_epi8(_cvtu64_mask64(~true_ascii), window.high);
    high_fixed = _mm512_mask_mov_epi8(high_fixed, _cvtu64_mask64(four_byte), high_four);

    sz_u64_t const valid_start = true_ascii | valid2 | valid3 | valid4;
    //  Continuation bytes consumed by a valid multibyte lead's body (so they are NOT independent units).
    sz_u64_t const consumed = (((valid2 | valid3 | valid4) << 1) | ((valid3 | valid4) << 2) | (valid4 << 3)) &
                              continuation & loaded_mask;
    sz_u64_t const starts = loaded_mask & ~consumed;
    sz_u64_t const replacement = starts & ~valid_start;

    //  4-byte plane bits (bits 16..20 of the codepoint); zero on every non-4-byte lane so cp = (high<<8)|low there.
    __m512i const plane_all = _mm512_or_si512(
        _mm512_and_si512(_mm512_slli_epi16(_mm512_and_si512(raw, _mm512_set1_epi8(0x07)), 2), _mm512_set1_epi8(0x1C)),
        sz_utf8_srl8_icelake_(next1, 4, 0x03));
    __m512i const previous_cluster_lane = _mm512_maskz_mov_epi8(_cvtu64_mask64(four_byte), plane_all);

    //  Build the 64-lane palette-index byte vector in ONE pass. The whole BMP (cp < 0x10000) resolves through ONE
    //  full-BMP trie walk: the sub-0x800 page LUT and its 0x800 split are folded into the trie leaf, so there is no
    //  page-LUT gather on this path. Astral lanes (cp >= 0x10000) are blended from the astral trie below; replacement
    //  lanes are forced to U+FFFD's palette index so malformed input matches the serial U+FFFD policy.
    sz_u64_t const is_astral = four_byte & loaded_mask;
    __m512i index;
    //  All-ASCII fast path (the dominant Latin case): every loaded lane is a 1-byte unit below 0x80, so the page LUT
    //  resolves the whole block with one `vpermi2b` over span 0 -- no lo/hi unpack, no BMP trie, no astral scan.
    if ((true_ascii & loaded_mask) == loaded_mask) {
        __m512i const low = _mm512_loadu_si512((void const *)sz_utf8_line_break_page_lut_);
        __m512i const high = _mm512_loadu_si512((void const *)(sz_utf8_line_break_page_lut_ + 64));
        index = _mm512_permutex2var_epi8(low, raw, high);
    }
    else if (!three_byte && !four_byte && !replacement) {
        //  Sub-0x800 fast path (window is ASCII + valid 2-byte only): address the 2048-byte page LUT directly by
        //  codepoint = (high << 8) | low, so two `gather_byte` cascades replace the three-level BMP trie. Bit-identical
        //  (the trie leaf folds in this same page LUT below 0x800).
        __m512i const zero_bytes = _mm512_setzero_si512();
        __m512i const codepoint_low_half = _mm512_or_si512(
            _mm512_slli_epi16(_mm512_unpacklo_epi8(high_fixed, zero_bytes), 8),
            _mm512_unpacklo_epi8(low_fixed, zero_bytes));
        __m512i const codepoint_high_half = _mm512_or_si512(
            _mm512_slli_epi16(_mm512_unpackhi_epi8(high_fixed, zero_bytes), 8),
            _mm512_unpackhi_epi8(low_fixed, zero_bytes));
        __m512i const class_low_half = sz_utf8_rune_gather_byte_(sz_utf8_line_break_page_lut_, 0x800,
                                                                 codepoint_low_half);
        __m512i const class_high_half = sz_utf8_rune_gather_byte_(sz_utf8_line_break_page_lut_, 0x800,
                                                                  codepoint_high_half);
        index = _mm512_packus_epi16(class_low_half, class_high_half);
    }
    else { index = sz_line_break_bmp_full_index_compact_icelake_(high_fixed, low_fixed, starts); }
    if (is_astral) {
        //  Astral is rare: reconstruct the full 32-bit codepoint per sixteen-lane group and resolve through the
        //  shared astral trie, then blend the resulting indices back into the byte vector. No big-range scan. The
        //  groups are unrolled because `vextracti32x4`/`vinserti32x4` take an immediate lane selector.
        __m512i astral_index = _mm512_setzero_si512();
        __m512i high32, low32, plane32, codepoints, group_index;
        high32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(high_fixed, 0));
        low32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(low_fixed, 0));
        plane32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(previous_cluster_lane, 0));
        codepoints = _mm512_or_si512(_mm512_or_si512(_mm512_slli_epi32(plane32, 16), _mm512_slli_epi32(high32, 8)),
                                     low32);
        group_index = sz_line_break_classify_astral16_icelake_(codepoints);
        astral_index = _mm512_inserti32x4(astral_index, _mm512_cvtepi32_epi8(group_index), 0);
        high32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(high_fixed, 1));
        low32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(low_fixed, 1));
        plane32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(previous_cluster_lane, 1));
        codepoints = _mm512_or_si512(_mm512_or_si512(_mm512_slli_epi32(plane32, 16), _mm512_slli_epi32(high32, 8)),
                                     low32);
        group_index = sz_line_break_classify_astral16_icelake_(codepoints);
        astral_index = _mm512_inserti32x4(astral_index, _mm512_cvtepi32_epi8(group_index), 1);
        high32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(high_fixed, 2));
        low32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(low_fixed, 2));
        plane32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(previous_cluster_lane, 2));
        codepoints = _mm512_or_si512(_mm512_or_si512(_mm512_slli_epi32(plane32, 16), _mm512_slli_epi32(high32, 8)),
                                     low32);
        group_index = sz_line_break_classify_astral16_icelake_(codepoints);
        astral_index = _mm512_inserti32x4(astral_index, _mm512_cvtepi32_epi8(group_index), 2);
        high32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(high_fixed, 3));
        low32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(low_fixed, 3));
        plane32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(previous_cluster_lane, 3));
        codepoints = _mm512_or_si512(_mm512_or_si512(_mm512_slli_epi32(plane32, 16), _mm512_slli_epi32(high32, 8)),
                                     low32);
        group_index = sz_line_break_classify_astral16_icelake_(codepoints);
        astral_index = _mm512_inserti32x4(astral_index, _mm512_cvtepi32_epi8(group_index), 3);
        index = _mm512_mask_mov_epi8(index, _cvtu64_mask64(is_astral), astral_index);
    }
    if (replacement) {
        //  Force U+FFFD's palette index on the malformed lanes, derived by walking the BMP trie for cp = 0xFFFD so
        //  the malformed policy stays table-derived rather than hard-coded.
        __m512i const fffd_index = sz_line_break_bmp_full_index_icelake_(_mm512_set1_epi8((char)0xFF),
                                                                         _mm512_set1_epi8((char)0xFD));
        index = _mm512_mask_mov_epi8(index, _cvtu64_mask64(replacement), fffd_index);
    }

    //  Resolve class / side / DottedCircle by permuting the precomputed palette tables with the per-lane palette index:
    //  three `vpermb` over 64-entry .rodata tables replace the u16 descriptor permute + bitfield unpack. The LB1
    //  resolution (SA->AL/CM, CJ->NS, AI/SG/XX->AL) and every side/dotted bit are baked into the tables, so this is
    //  bit-identical to the descriptor unpack. (Garbage indices on non-start lanes fold to &63 and are ignored.)
    sz_line_break_classified_t result;
    result.classes = _mm512_permutexvar_epi8(index, _mm512_load_si512((void const *)sz_utf8_line_break_palette_class_));
    result.side = _mm512_permutexvar_epi8(index, _mm512_load_si512((void const *)sz_utf8_line_break_palette_side_));
    __m512i const dotted_bytes = _mm512_permutexvar_epi8(
        index, _mm512_load_si512((void const *)sz_utf8_line_break_palette_dotted_));
    sz_u64_t const dotted = _cvtmask64_u64(_mm512_test_epi8_mask(dotted_bytes, dotted_bytes));
    result.dotted = dotted & starts;
    result.starts = starts;
    result.replacement = replacement;
    result.non_start = loaded_mask & ~starts;
    result.loaded = window.loaded;
    return result;
}

#pragma endregion In register vectorized classifier

#pragma region Mask algebra rule engine

/** @brief Build a 64-bit "lane class == @p cls" mask with one `vpcmpeqb` -> kmask. */
SZ_HELPER_INLINE sz_u64_t sz_line_break_class_mask_icelake_(__m512i classes, sz_u8_t cls) {
    return _cvtmask64_u64(_mm512_cmpeq_epi8_mask(classes, _mm512_set1_epi8((char)cls)));
}

/** @brief The class/side byte held at lane @p lane, extracted in-register by a single byte permute (no scalar loop). */
SZ_HELPER_INLINE sz_u8_t sz_line_break_byte_at_icelake_(__m512i lanes, sz_size_t lane) {
    __m512i const broadcast = _mm512_permutexvar_epi8(_mm512_set1_epi8((char)lane), lanes);
    return (sz_u8_t)_mm_cvtsi128_si32(_mm512_castsi512_si128(broadcast));
}

/** @brief Build a 64-bit "lane (side & @p bit) != 0" mask with one `vptestmb` -> kmask. */
SZ_HELPER_INLINE sz_u64_t sz_line_break_side_mask_icelake_(__m512i side_lo, sz_u8_t bit) {
    __m512i const masked = _mm512_and_si512(side_lo, _mm512_set1_epi8((char)bit));
    return _cvtmask64_u64(_mm512_test_epi8_mask(masked, masked));
}

/** @brief Build a 64-bit "lane class is in the inclusive byte range [@p lo, @p hi]" mask with two `vpcmpub` -> kmask.
 *         Used as a cheap combined presence test that gates the rarely-fired script blocks (Hangul, Brahmic) without
 *         extracting each individual per-class mask on the common Latin/CJK path. */
SZ_HELPER_INLINE sz_u64_t sz_line_break_class_range_mask_icelake_(__m512i classes, sz_u8_t lo, sz_u8_t hi) {
    __mmask64 const ge = _mm512_cmp_epu8_mask(classes, _mm512_set1_epi8((char)lo), _MM_CMPINT_NLT);
    __mmask64 const le = _mm512_cmp_epu8_mask(classes, _mm512_set1_epi8((char)hi), _MM_CMPINT_LE);
    return _cvtmask64_u64(_kand_mask64(ge, le));
}

/** @brief Byte-lane gate/base derivation for the byte-level rule engine: identifies cluster bases, the transparent
 *         gate (continuations + attached combining marks), and reclassifies lone marks (LB10) to AL in @p classes. */
typedef struct sz_line_break_byte_frame_t {
    __m512i classes;   /**< Class per lane with lone marks reclassified to AL (LB10). */
    sz_u64_t base;     /**< Cluster-base lanes (every effective start except an attached CM/ZWJ). */
    sz_u64_t gate;     /**< Transparent lanes for neighbour fills: continuations + attached-mark starts. */
    sz_u64_t attached; /**< Attached CM/ZWJ start lanes (LB9). */
    sz_u64_t
        lone_mark; /**< LB10 lone marks reclassified to AL; their side bits must be cleared (serial zeros the descriptor). */
} sz_line_break_byte_frame_t;

SZ_HELPER_INLINE sz_line_break_byte_frame_t sz_line_break_byte_frame_icelake_(sz_line_break_classified_t classified) {
    sz_u64_t const starts = classified.starts, non_start = classified.non_start;
    sz_u64_t const mark_start = (sz_line_break_class_mask_icelake_(classified.classes, sz_line_break_cm_k) |
                                 sz_line_break_class_mask_icelake_(classified.classes, sz_line_break_zwj_k)) &
                                starts;
    sz_u64_t const excluded = (sz_line_break_class_mask_icelake_(classified.classes, sz_line_break_bk_k) |
                               sz_line_break_class_mask_icelake_(classified.classes, sz_line_break_cr_k) |
                               sz_line_break_class_mask_icelake_(classified.classes, sz_line_break_lf_k) |
                               sz_line_break_class_mask_icelake_(classified.classes, sz_line_break_nl_k) |
                               sz_line_break_class_mask_icelake_(classified.classes, sz_line_break_sp_k) |
                               sz_line_break_class_mask_icelake_(classified.classes, sz_line_break_zw_k)) &
                              starts;
    sz_u64_t const good_base = starts & ~excluded & ~mark_start;
    //  A mark attaches (LB9) when reachable from a good base across only continuations and other marks. Flood each
    //  good base rightward over (continuations | mark starts); the mark starts it reaches are the attached marks.
    sz_u64_t const mark_bytes = sz_u64_fill_right_(mark_start, non_start) | mark_start;
    sz_u64_t const flood = sz_u64_fill_right_(good_base, non_start | mark_bytes);
    sz_u64_t const attached = flood & mark_start;
    sz_u64_t const lone_mark = mark_start & ~attached; // LB10: a mark with no attachable base acts as AL

    sz_line_break_byte_frame_t frame;
    frame.classes = _mm512_mask_mov_epi8(classified.classes, _cvtu64_mask64(lone_mark),
                                         _mm512_set1_epi8((char)sz_line_break_al_k));
    frame.base = starts & ~attached;
    frame.gate = non_start | attached;
    frame.attached = attached;
    frame.lone_mark = lone_mark;
    return frame;
}

/**
 *  @brief  Per-ISA extractor: lower one classified 64-byte window to the portable @ref sz_line_break_frame_t. Builds
 *          the byte-level cluster frame (LB9/LB10), then materializes per-class membership (after the LB10 lone-mark
 *          -> AL reclassify), the raw ZWJ mask, the five side-bit masks, and the per-lane class/side bytes for the
 *          carry-out reads. All `__m512i` -> `sz_u64_t` work lives here; the rule engine that consumes the frame is
 *          fully portable.
 */
SZ_HELPER_INLINE sz_line_break_frame_t sz_line_break_build_frame_icelake_(sz_line_break_classified_t classified,
                                                                          sz_u8_t *effective_class_byte_out,
                                                                          sz_u8_t *side_byte_out) {
    sz_line_break_byte_frame_t const byte_frame = sz_line_break_byte_frame_icelake_(classified);
    __m512i const classes = byte_frame.classes;
    //  LB10 reclassifies a lone CM/ZWJ to AL; its descriptor side bits (EAW/Pi/Pf/...) must go with it, else LB19/LB15
    //  see a phantom East-Asian / quote cluster. Mirrors the serial path zeroing `codepoint_descriptors` on LB10.
    __m512i const side = _mm512_maskz_mov_epi8(_cvtu64_mask64(~byte_frame.lone_mark), classified.side);

    sz_line_break_frame_t frame;
    frame.base = byte_frame.base;
    frame.gate = byte_frame.gate;
    frame.attached = byte_frame.attached;
    frame.lone_mark = byte_frame.lone_mark;
    frame.non_start = classified.non_start;
    frame.dotted = classified.dotted;
    frame.starts = classified.starts;
    frame.replacement = classified.replacement;
    //  Full unroll is load-bearing: with a runtime loop index the compiler keeps `effective_class` stack-resident, so
    //  every per-class read in the rule engine becomes a load (~7% slower). Unrolled, SROA promotes the array to
    //  registers and drops the `vpcmpeqb` for classes the engine never reads.
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 64
#endif
    for (sz_size_t cls = 0; cls < sz_line_break_class_count_k; ++cls)
        frame.effective_class[cls] = sz_line_break_class_mask_icelake_(classes, (sz_u8_t)cls);
    frame.raw_zwj = sz_line_break_class_mask_icelake_(classified.classes, sz_line_break_zwj_k);
    frame.side_pi = sz_line_break_side_mask_icelake_(side, sz_line_break_side_pi_k);
    frame.side_pf = sz_line_break_side_mask_icelake_(side, sz_line_break_side_pf_k);
    frame.side_eaw = sz_line_break_side_mask_icelake_(side, sz_line_break_side_eaw_k);
    frame.side_cn = sz_line_break_side_mask_icelake_(side, sz_line_break_side_cn_k);
    frame.side_ext = sz_line_break_side_mask_icelake_(side, sz_line_break_side_ext_k);
    _mm512_storeu_si512((void *)effective_class_byte_out, classes);
    _mm512_storeu_si512((void *)side_byte_out, side);
    return frame;
}

/**
 *  @brief  Byte-level UAX-14 rule engine, Ice Lake entry: extract the portable frame in-register, then delegate every
 *          LB1-LB31 decision to the portable @ref sz_line_break_decide_window_. Kept as the driver's call target so the
 *          forward driver is unchanged.
 */
SZ_HELPER_INLINE sz_line_break_window_t sz_line_break_decide_window_icelake_(sz_line_break_classified_t classified,
                                                                             sz_line_break_carry_t carry,
                                                                             sz_line_break_carry_t *carry_out,
                                                                             sz_size_t complete_limit,
                                                                             sz_bool_t more_text) {
    sz_u8_t effective_class_byte[64], side_byte[64];
    sz_line_break_frame_t const frame = sz_line_break_build_frame_icelake_(classified, effective_class_byte, side_byte);
    return sz_line_break_decide_window_(&frame, effective_class_byte, side_byte, carry, carry_out, complete_limit,
                                        more_text);
}

#pragma endregion Mask algebra rule engine

#pragma region Forward driver

/**
 *  @brief  Largest byte prefix of the window whose codepoints are all fully loaded (no multi-byte lead straddles the
 *          64-byte edge). Mirrors the word kernel's complete-limit: a declared-length lead whose span exceeds `loaded`
 *          ends the trusted region just before it; with no more text the whole window is complete. Never below 1.
 */
SZ_HELPER_AUTO sz_size_t sz_line_break_complete_limit_(sz_utf8_rune_window_t window, sz_bool_t more_text) {
    sz_size_t const loaded = window.loaded;
    if (!more_text) return loaded;
    sz_u64_t const valid = sz_u64_mask_until_(loaded);
    sz_u64_t const starts = _cvtmask64_u64(window.codepoint_starts) & valid;
    sz_u64_t const two = _cvtmask64_u64(window.two_byte_starts) & starts;
    sz_u64_t const three = _cvtmask64_u64(window.three_byte_starts) & starts;
    sz_u64_t const four = _cvtmask64_u64(window.four_byte_starts) & starts;
    //  A multi-byte lead straddles when its declared span runs past `loaded`: a 2-byte lead in the top 1 lane, a
    //  3-byte lead in the top 2 lanes, a 4-byte lead in the top 3 lanes. Trust ends at the lowest such lead.
    sz_u64_t const straddle = ((two & ~sz_u64_mask_until_(loaded > 1 ? loaded - 1 : 0)) |
                               (three & ~sz_u64_mask_until_(loaded > 2 ? loaded - 2 : 0)) |
                               (four & ~sz_u64_mask_until_(loaded > 3 ? loaded - 3 : 0))) &
                              valid;
    sz_size_t const limit = straddle ? (sz_size_t)sz_u64_ctz(straddle) : loaded;
    return limit > 0 ? limit : loaded; // `limit == 0` keeps the whole window for guaranteed progress
}

/**
 *  @brief  Byte-level zero-scalar forward UAX-14 kernel: an overlap-free advancing driver. Each iteration decodes one
 *          64-byte window at the codepoint-aligned `position`, classifies it in-register, decides the break-before mask
 *          over the complete-codepoint region with a carry-aware rule engine, drains the trusted band below the trust
 *          horizon, and advances by `win.resolved` with the small register carry threaded forward. There is NO byte
 *          re-read (no `start_at_or_before_`, no left-context back-walk): the carry alone supplies lane-0 left context.
 *          `bytes_consumed` is always a confirmed break (`line_start`), so resume is bit-identical and capacity-free.
 */
SZ_API_COMPTIME sz_size_t sz_utf8_linebreaks_icelake_bytes_( //
    sz_cptr_t text, sz_size_t length,                        //
    sz_size_t *starts, sz_size_t *lengths,                   //
    sz_size_t capacity, sz_size_t *bytes_consumed) {

    if (length == 0 || capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *bytes = (sz_u8_t const *)text;
    __m512i const lane_identity = sz_utf8_lane_identity_icelake_();
    sz_size_t produced = 0;
    sz_size_t line_start = 0;                                 // open line's first byte: a confirmed break (or 0)
    sz_size_t position = 0;                                   // codepoint-aligned anchor of the next window
    sz_line_break_carry_t carry = sz_line_break_carry_sot_(); // LB2: start-of-text has no left context

    while (position < length) {
        sz_utf8_rune_window_t const window = sz_utf8_rune_decode_window_icelake_(bytes + position, length - position,
                                                                                 lane_identity);
        sz_bool_t const more_text = (sz_bool_t)(position + window.loaded < length);
        sz_size_t const complete_limit = sz_line_break_complete_limit_(window, more_text);
        sz_line_break_classified_t const classified = sz_line_break_classify_window_icelake_(window, lane_identity);

        sz_line_break_carry_t carry_next = carry;
        sz_line_break_window_t const win = sz_line_break_decide_window_icelake_(classified, carry, &carry_next,
                                                                                complete_limit, more_text);
        sz_u64_t const commit = win.breaks & sz_u64_mask_until_(win.resolved);

        produced = sz_utf8_rune_drain_forward_(commit, position, lane_identity, starts, lengths, produced, capacity,
                                               &line_start);
        if (produced >= capacity) {
            if (bytes_consumed) *bytes_consumed = line_start;
            return produced;
        }

        //  Advance by the trust horizon when it bit before the complete edge, else the whole complete span (guaranteed
        //  progress). `carry_next` is already anchored at exactly this byte by `decide_window_` -- one decision per
        //  window, no second pass.
        sz_size_t const advance = win.resolved ? win.resolved : complete_limit;
        carry = carry_next;
        position += advance ? advance : window.loaded;
    }

    //  The trailing (still-open) line `[line_start, length)` finalizes the output (end of text is a break).
    if (produced < capacity) starts[produced] = line_start, lengths[produced] = length - line_start, ++produced;
    if (bytes_consumed) *bytes_consumed = length;
    return produced;
}

/**
 *  @brief  Forward UAX-14 line-break-opportunity kernel (Ice Lake AVX-512).
 *
 *  Bit-exact with `sz_utf8_linebreaks_serial`. Emits every UAX-14 break opportunity (no per-segment
 *  mandatory flag). The classifier and rule engine are fully vectorized (no per-lane scalar loop, no `vpgather`,
 *  no serial-oracle deferral); the 64-codepoint block engine threads cross-block state through a register carry
 *  (no halo back-scan), so throughput is flat in run length. Emits at most @p capacity segments; sets
 *  *@p bytes_consumed to the resume.
 */
SZ_API_COMPTIME sz_size_t sz_utf8_linebreaks_icelake( //
    sz_cptr_t text, sz_size_t length,                 //
    sz_size_t *starts, sz_size_t *lengths,            //
    sz_size_t capacity, sz_size_t *bytes_consumed) {

    return sz_utf8_linebreaks_icelake_bytes_(text, length, starts, lengths, capacity, bytes_consumed);
}

#pragma endregion Forward driver

#pragma endregion UAX 14 Line Boundaries forward kernel
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_ICELAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_LINEBREAKS_ICELAKE_H_
