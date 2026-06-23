/**
 *  @brief Ice Lake backend for UAX-14 line break boundaries.
 *  @file include/stringzilla/utf8_linewraps/icelake.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_LINEWRAPS_ICELAKE_H_
#define STRINGZILLA_UTF8_LINEWRAPS_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_linewraps/tables.h"
#include "stringzilla/utf8_linewraps/serial.h"
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

#pragma region UAX 14 Line Boundaries forward kernel

/** @brief Engine side-word bits, one per descriptor flag the LB rules read in-register. */
enum {
    sz_line_break_side_pi_k = 0x01,     /**< gc=Pi (LB15a/LB19) */
    sz_line_break_side_pf_k = 0x02,     /**< gc=Pf (LB15b/LB19) */
    sz_line_break_side_eaw_k = 0x04,    /**< East-Asian Width F/W/H (LB19/LB30) */
    sz_line_break_side_cn_k = 0x08,     /**< unassigned (LB30b, only in conjunction with ExtPict) */
    sz_line_break_side_ext_k = 0x10,    /**< Extended_Pictographic (LB30b) */
    sz_line_break_side_ri_k = 0x20,     /**< raw RI class (LB30a parity) */
    sz_line_break_side_zwj_k = 0x40,    /**< raw ZWJ class (LB8a) */
    sz_line_break_side_mark_k = 0x80,   /**< resolved CM|ZWJ (LB9/LB10 attachment) */
    sz_line_break_side_dotted_k = 0x100 /**< DottedCircle U+25CC (LB28a); above the byte side-word */
};

/** @brief Window state carried across the 64-lane block window_edge: the left context + open runs straddling into
 *         lane 0, so the next window decides its first cluster's break-before with NO byte re-read (no overlap). */
typedef struct sz_line_break_carry_t {
    sz_u8_t have_prev; /**< 0 only at start-of-text (LB2); 1 once a cluster precedes lane 0 */
    sz_u64_t
        previous_class_bit; /**< one-hot effective LB class of the cluster ending just before lane 0 (post LB9/10) */
    sz_u64_t
        previous2_class_bit; /**< one-hot LB class of the cluster two before lane 0 (2-left rules LB19/LB20a/LB21a) */
    sz_u8_t left_eaw;        /**< the cluster before lane 0 is East-Asian F/W/H (LB19 side bit) */
    sz_u8_t left2_eaw;       /**< the cluster two before lane 0 is East-Asian F/W/H (LB19 ~prev2EAW) */
    sz_u8_t left_pf;         /**< the cluster before lane 0 carries gc=Pf (LB19 `prev_(QU & ~PF)`) */
    sz_u8_t left_aksara;     /**< the cluster before lane 0 is an aksara base AK|AS|DottedCircle (LB28a) */
    sz_u8_t left2_aksara;    /**< the cluster two before lane 0 is an aksara base (LB28a `prev_(base_vi)`) */
    sz_u8_t left_extpict_cn; /**< the cluster before lane 0 is Extended_Pictographic & unassigned (LB30b) */
    sz_u8_t prev_is_zwj;     /**< the cluster before lane 0 is a bare ZWJ (LB8a no-break-after) */
    sz_u8_t open_sp_opener;  /**< class anchoring an open "X SP*" run (LB14/16/17/8); 0xFF = none */
    sz_u8_t in_nu_run;       /**< lane 0 continues a "NU (SY|IS)*" run (LB25) */
    sz_u8_t in_nu_close;     /**< the cluster before lane 0 is a CL/CP closing a NU run (LB25 nu_run_close) */
    sz_u8_t ri_open;         /**< lane 0 continues an RI sequence (parity carries) */
    sz_u8_t ri_parity_odd;   /**< odd count of RI characters precede lane 0 within the open sequence */
    sz_u8_t qupi_sp_open;    /**< lane 0 continues an open "[QU&Pi] SP*" governed run (LB15a) */
} sz_line_break_carry_t;

/** @brief Is the carried one-hot class word @p class_bits set for class `cls`? `class_bits` is hoisted once per
 *         window into a register (0 at start-of-text), so every call is a pure register shift/and the compiler CSEs. */
SZ_INTERNAL sz_bool_t sz_line_break_class_is_(sz_u64_t class_bits, sz_u8_t cls) {
    return (sz_bool_t)((class_bits >> cls) & 1ull);
}

/** @brief Is the carried one-hot class word @p class_bits a member of @p class_set? */
SZ_INTERNAL sz_bool_t sz_line_break_class_in_(sz_u64_t class_bits, sz_u64_t class_set) {
    return (sz_bool_t)((class_bits & class_set) != 0);
}

/** @brief Result of one window decision: break-before bits + the trust horizon (lanes [0,resolved) are committable). */
typedef struct sz_line_break_window_t {
    sz_u64_t breaks;    /**< break-before bits at cluster-base lanes */
    sz_size_t resolved; /**< lowest lane whose verdict needs context past the window edge; commit below this */
} sz_line_break_window_t;

/** @brief Start-of-text carry: no previous cluster, all runs closed. */
SZ_INTERNAL sz_line_break_carry_t sz_line_break_carry_sot_(void) {
    sz_line_break_carry_t carry;
    carry.have_prev = 0, carry.previous_class_bit = 1ull << sz_line_break_xx_k,
    carry.previous2_class_bit = 1ull << sz_line_break_xx_k;
    carry.left_eaw = 0, carry.left2_eaw = 0, carry.left_pf = 0, carry.left_aksara = 0, carry.left2_aksara = 0,
    carry.left_extpict_cn = 0;
    carry.prev_is_zwj = 0, carry.open_sp_opener = 0xFF, carry.in_nu_run = 0, carry.in_nu_close = 0, carry.ri_open = 0,
    carry.ri_parity_odd = 0, carry.qupi_sp_open = 0;
    return carry;
}

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
SZ_INTERNAL __m512i sz_line_break_classify_astral16_icelake_(__m512i codepoints) {
    __m512i const offset = _mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x10000));
    __m512i const stage1 = sz_utf8_codepoints_permute256_icelake_(
        sz_utf8_line_break_astral_s0_, _mm512_and_si512(_mm512_srli_epi32(offset, 12), _mm512_set1_epi32(0xFF)));
    __m512i const stage2_index = _mm512_add_epi32(
        _mm512_slli_epi32(stage1, 4), _mm512_and_si512(_mm512_srli_epi32(offset, 8), _mm512_set1_epi32(0xF)));
    __m512i const stage2 = sz_utf8_codepoints_lut_cascade_icelake_(
        sz_utf8_line_break_astral_s1_, (int)sz_utf8_line_break_astral_s1_tiles_k, stage2_index);
    __m512i const leaf_index = _mm512_add_epi32(_mm512_slli_epi32(stage2, 4),
                                                _mm512_and_si512(_mm512_srli_epi32(offset, 4), _mm512_set1_epi32(0xF)));
    __m512i const leaf = sz_utf8_codepoints_lut_cascade_icelake_(sz_utf8_line_break_astral_s2_,
                                                                 (int)sz_utf8_line_break_astral_s2_tiles_k, leaf_index);
    __m512i const class_index = _mm512_add_epi32(_mm512_slli_epi32(leaf, 4),
                                                 _mm512_and_si512(offset, _mm512_set1_epi32(0xF)));
    return sz_utf8_codepoints_lut_cascade_icelake_(sz_utf8_line_break_astral_leaf_,
                                                   (int)sz_utf8_line_break_astral_leaf_tiles_k, class_index);
}

/** @brief All-64-lane complete-BMP palette index for `cp < 0x10000` over the full-BMP trie (offset = the codepoint
 *         itself, no `-0x800` subtract), so a SINGLE pass resolves the whole BMP -- the page LUT and the 0x800 split
 *         are gone. This mirrors `sz_utf8_codepoints_trie_walk_icelake_` exactly with `block = superblock = 8` and
 *         `offset = (high << 8) | low`; the substrate function bakes in a hard `-0x800`, so the zero-base form lives
 *         here. Lanes whose codepoint is >= 0x10000 are undefined (the caller blends in the astral path). */
SZ_INTERNAL __m512i sz_line_break_bmp_full_index_icelake_(__m512i high, __m512i low) {
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

    __m512i const level1_lo = sz_utf8_codepoints_gather_byte_(
        sz_utf8_line_break_bmp_full_trie_l1_, (int)sz_utf8_line_break_bmp_full_trie_l1_count_k, super_lo);
    __m512i const level1_hi = sz_utf8_codepoints_gather_byte_(
        sz_utf8_line_break_bmp_full_trie_l1_, (int)sz_utf8_line_break_bmp_full_trie_l1_count_k, super_hi);

    __m512i const l2_index_lo = _mm512_add_epi16(_mm512_mullo_epi16(level1_lo, super_v16), super_off_lo);
    __m512i const l2_index_hi = _mm512_add_epi16(_mm512_mullo_epi16(level1_hi, super_v16), super_off_hi);
    __m512i const leaf_idx_lo = sz_utf8_codepoints_gather_word_(
        sz_utf8_line_break_bmp_full_trie_l2_, (int)sz_utf8_line_break_bmp_full_trie_l2_count_k, l2_index_lo);
    __m512i const leaf_idx_hi = sz_utf8_codepoints_gather_word_(
        sz_utf8_line_break_bmp_full_trie_l2_, (int)sz_utf8_line_break_bmp_full_trie_l2_count_k, l2_index_hi);

    __m512i const leaf_byte_lo = _mm512_add_epi16(_mm512_mullo_epi16(leaf_idx_lo, block_v16), within_lo);
    __m512i const leaf_byte_hi = _mm512_add_epi16(_mm512_mullo_epi16(leaf_idx_hi, block_v16), within_hi);
    __m512i const class_lo = sz_utf8_codepoints_gather_byte_(
        sz_utf8_line_break_bmp_full_trie_leaf_, (int)sz_utf8_line_break_bmp_full_trie_leaf_count_k, leaf_byte_lo);
    __m512i const class_hi = sz_utf8_codepoints_gather_byte_(
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
SZ_INTERNAL __m512i sz_line_break_bmp_full_index_compact_icelake_(__m512i high, __m512i low, sz_u64_t starts) {
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

    __m512i const level1 = sz_utf8_codepoints_gather_byte_(sz_utf8_line_break_bmp_full_trie_l1_,
                                                           (int)sz_utf8_line_break_bmp_full_trie_l1_count_k, super);
    __m512i const l2_index = _mm512_add_epi16(_mm512_mullo_epi16(level1, super_v16), super_off);
    __m512i const leaf_idx = sz_utf8_codepoints_gather_word_(
        sz_utf8_line_break_bmp_full_trie_l2_, (int)sz_utf8_line_break_bmp_full_trie_l2_count_k, l2_index);
    __m512i const leaf_byte = _mm512_add_epi16(_mm512_mullo_epi16(leaf_idx, block_v16), within);
    __m512i const class_word = sz_utf8_codepoints_gather_byte_(
        sz_utf8_line_break_bmp_full_trie_leaf_, (int)sz_utf8_line_break_bmp_full_trie_leaf_count_k, leaf_byte);

    //  Narrow the 32 class words back to 32 contiguous bytes (low half), then scatter to the original start lanes.
    __m512i const class_packed = _mm512_castsi256_si512(_mm512_cvtepi16_epi8(class_word));
    return _mm512_maskz_expand_epi8(start_mask, class_packed);
}

/** @brief Unpack thirty-two 16-bit palette descriptors (one half of the 64-byte block) to the LB1-resolved class and
 *         the engine side byte, both as 16-bit lanes, plus the per-lane DottedCircle predicate. Applies the serial
 *         resolution aliasing (SA->AL/CM, AI/SG/XX->AL, CJ->NS); Pi/Pf/EAW/Cn|Ext side bits come from descriptor
 *         bits 6/7/8/9; RI/ZWJ side from the raw class; CM|ZWJ -> mark side bit; DottedCircle from bit 13. */
SZ_INTERNAL void sz_line_break_descriptor_unpack_half_icelake_(__m512i descriptors, __m512i *classes_out,
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
 *          the serial `sz_utf8_decode_` "consume-1 U+FFFD" policy: an invalid lead, a short/stray continuation, an
 *          overlong / surrogate / out-of-range lead each become a single-byte U+FFFD unit (class AL), so serial and
 *          icelake agree on malformed input. Valid leads classify by decoded VALUE (page / trie / big / astral),
 *          matching the serial resolution precedence. The BMP trie uses the shared substrate `trie_walk_icelake_`.
 */
SZ_INTERNAL sz_line_break_classified_t sz_line_break_classify_window_icelake_(sz_utf8_codepoints_window_t window,
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
        sz_utf8_codepoints_srl8_(next2, 2, 0x0F));
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
        sz_utf8_codepoints_srl8_(next1, 4, 0x03));
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
SZ_INTERNAL sz_u64_t sz_line_break_class_mask_icelake_(__m512i classes, sz_u8_t cls) {
    return _cvtmask64_u64(_mm512_cmpeq_epi8_mask(classes, _mm512_set1_epi8((char)cls)));
}

/** @brief The class/side byte held at lane @p lane, extracted in-register by a single byte permute (no scalar loop). */
SZ_INTERNAL sz_u8_t sz_line_break_byte_at_icelake_(__m512i lanes, sz_size_t lane) {
    __m512i const broadcast = _mm512_permutexvar_epi8(_mm512_set1_epi8((char)lane), lanes);
    return (sz_u8_t)_mm_cvtsi128_si32(_mm512_castsi512_si128(broadcast));
}

/** @brief Build a 64-bit "lane (side & @p bit) != 0" mask with one `vptestmb` -> kmask. */
SZ_INTERNAL sz_u64_t sz_line_break_side_mask_icelake_(__m512i side_lo, sz_u8_t bit) {
    __m512i const masked = _mm512_and_si512(side_lo, _mm512_set1_epi8((char)bit));
    return _cvtmask64_u64(_mm512_test_epi8_mask(masked, masked));
}

/** @brief Build a 64-bit "lane class is in the inclusive byte range [@p lo, @p hi]" mask with two `vpcmpub` -> kmask.
 *         Used as a cheap combined presence test that gates the rarely-fired script blocks (Hangul, Brahmic) without
 *         extracting each individual per-class mask on the common Latin/CJK path. */
SZ_INTERNAL sz_u64_t sz_line_break_class_range_mask_icelake_(__m512i classes, sz_u8_t lo, sz_u8_t hi) {
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

SZ_INTERNAL sz_line_break_byte_frame_t sz_line_break_byte_frame_icelake_(sz_line_break_classified_t classified) {
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
    sz_u64_t const mark_bytes = sz_utf8_codepoints_fill_right_(mark_start, non_start) | mark_start;
    sz_u64_t const flood = sz_utf8_codepoints_fill_right_(good_base, non_start | mark_bytes);
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

/** @brief Previous-cluster mask: bit k set => the cluster ending just before lane k has a base in @p class_base. */
SZ_INTERNAL sz_u64_t sz_line_break_prev_(sz_u64_t class_base, sz_u64_t gate) {
    return sz_utf8_codepoints_fill_right_(class_base, gate) << 1;
}

/** @brief Next-cluster mask: bit k set => the cluster starting just after lane k has a base in @p class_base. */
SZ_INTERNAL sz_u64_t sz_line_break_next_(sz_u64_t class_base, sz_u64_t gate) {
    return sz_utf8_codepoints_fill_left_(class_base, gate) >> 1;
}

/** @brief Carry-aware previous-cluster mask: like @ref sz_line_break_prev_, but additionally marks @p edge (lane 0)
 *         when the carried left cluster matches (@p left_in_set), so cross-window left context needs no byte re-read. */
SZ_INTERNAL sz_u64_t sz_line_break_prevc_(sz_u64_t class_base, sz_u64_t gate, sz_bool_t left_in_set, sz_u64_t edge) {
    return (sz_utf8_codepoints_fill_right_(class_base, gate) << 1) | (left_in_set ? edge : 0);
}

/** @brief Commit a forced break at the undecided lanes of @p where (sets break + settled). */
SZ_INTERNAL void sz_line_break_force_break_(sz_u64_t where, sz_u64_t all, sz_u64_t *settled, sz_u64_t *breaks) {
    sz_u64_t const w = where & ~*settled & all;
    *breaks |= w, *settled |= w;
}

/** @brief Commit a forced join (no break) at the undecided lanes of @p where (sets settled only). */
SZ_INTERNAL void sz_line_break_force_join_(sz_u64_t where, sz_u64_t all, sz_u64_t *settled) {
    *settled |= where & ~*settled & all;
}

/** @brief Opener-governed "X SP*" run over byte-start lanes (LB8/14/16/17): flood the opener rightward across the
 *         transparent gate and the space bases; the result marks the opener + the governed space base lanes. */
SZ_INTERNAL sz_u64_t sz_line_break_run_byte_(sz_u64_t opener, sz_u64_t spaces, sz_u64_t gate) {
    return sz_utf8_codepoints_fill_right_(opener, gate | spaces) & (opener | spaces);
}

/** @brief Inclusive segmented prefix-XOR of @p members over each contiguous @p run_gate run (LB30a RI parity); when
 *         @p inbound_parity, an odd virtual member below lane 0 toggles the whole leading run. Returns the per-lane
 *         inclusive parity. The prefix-XOR scan is the shared substrate primitive; this wrapper adds the RI-pairing
 *         inbound-run seed (mirrors the word kernel's `ri_join_`). */
SZ_INTERNAL sz_u64_t sz_line_break_segmented_parity_(sz_u64_t members, sz_u64_t run_gate, sz_bool_t inbound_parity) {
    sz_u64_t bits = sz_utf8_codepoints_segmented_parity_icelake_(members, run_gate);
    if (inbound_parity) bits ^= sz_utf8_codepoints_fill_right_(run_gate & 1ull, run_gate);
    return bits;
}

/**
 *  @brief  Byte-level UAX-14 rule engine: decide break-before bits at the cluster-base lanes of one decoded window.
 *          Every LB1-LB31 rule is expressed over byte-start lanes, reading effective neighbours across `gate` via
 *          `sz_line_break_prev_`/`_next_` (no scalar collapse). The returned mask has bits only at base lanes (a
 *          break opportunity precedes that cluster).
 */
SZ_INTERNAL sz_line_break_window_t sz_line_break_decide_window_icelake_(sz_line_break_classified_t classified,
                                                                        sz_line_break_carry_t carry,
                                                                        sz_line_break_carry_t *carry_out,
                                                                        sz_size_t complete_limit, sz_bool_t more_text) {
    sz_line_break_byte_frame_t const frame = sz_line_break_byte_frame_icelake_(classified);
    __m512i const classes = frame.classes;
    //  LB10 reclassifies a lone CM/ZWJ to AL; its descriptor side bits (EAW/Pi/Pf/...) must go with it, else LB19/LB15
    //  see a phantom East-Asian / quote cluster. Mirrors the serial path zeroing `codepoint_descriptors` on LB10.
    __m512i const side = _mm512_maskz_mov_epi8(_cvtu64_mask64(~frame.lone_mark), classified.side);
    sz_u64_t const base = frame.base, gate = frame.gate, non_start = classified.non_start;
    sz_line_break_window_t empty;
    empty.breaks = 0, empty.resolved = complete_limit;
    if (!base) {
        //  All-ignorable window: nothing to decide, thread the inbound carry forward unchanged.
        if (carry_out) *carry_out = carry;
        return empty;
    }
    sz_bool_t const is_sot = (sz_bool_t)(!carry.have_prev);
    sz_bool_t const is_eot = (sz_bool_t)(!more_text);
    //  Hoist the carried one-hot class words into registers once. At start-of-text (`!have_prev`) they read as 0, so
    //  every `sz_line_break_class_is_/in_` test folds `have_prev` in and the repeated shifts/ands CSE in-register.
    sz_u64_t const left_bit = carry.have_prev ? carry.previous_class_bit : 0ull;
    sz_u64_t const left2_bit = carry.have_prev ? carry.previous2_class_bit : 0ull;
    sz_u64_t const first_base = base & (0ull - base);
    sz_u64_t const last_base = 1ull << (63 - sz_u64_clz(base));
    //  When a cluster precedes lane 0 (carried left context), lane 0's break-before is decided by the rules just like
    //  any interior lane; only at true start-of-text is lane 0 exempt (LB2).
    sz_u64_t const interior_lanes = carry.have_prev ? base : (base & ~first_base);
    sz_u64_t const edge = carry.have_prev ? first_base : 0;
    sz_u64_t const start_of_text = is_sot ? first_base : 0;
    sz_u64_t const end_of_text = is_eot ? last_base : 0;
    sz_u64_t const dotted_circle = classified.dotted & base & ~frame.lone_mark;

    //  The carried-left class is now read on demand through `sz_line_break_left_is_` / `sz_line_break_left_in_` (and the
    //  `left2_` variants) at each rule's call site, so the per-class boolean wall is gone. Only the East-Asian-width
    //  side bits, which live outside the class field, keep dedicated booleans here.
    sz_bool_t const left_is_east_asian_width = (sz_bool_t)(carry.have_prev && carry.left_eaw != 0);
    sz_bool_t const left2_is_east_asian_width = (sz_bool_t)(carry.have_prev && carry.left2_eaw != 0);
    //  Shared class sets for the two-left LB20a allow-list and (below) the one-left LB20a allow-list. Listing the SAME
    //  classes the original boolean tested: BK, CR, LF, NL, SP, ZW, CB, GL.
    sz_u64_t const lb20a_allowed_mask = (1ull << sz_line_break_bk_k) | (1ull << sz_line_break_cr_k) |
                                        (1ull << sz_line_break_lf_k) | (1ull << sz_line_break_nl_k) |
                                        (1ull << sz_line_break_sp_k) | (1ull << sz_line_break_zw_k) |
                                        (1ull << sz_line_break_cb_k) | (1ull << sz_line_break_gl_k);
    sz_bool_t const left2_is_lb20a_allowed = sz_line_break_class_in_(left2_bit, lb20a_allowed_mask);

    sz_u64_t const class_break_mandatory = sz_line_break_class_mask_icelake_(classes, sz_line_break_bk_k) & base;
    sz_u64_t const class_carriage_return = sz_line_break_class_mask_icelake_(classes, sz_line_break_cr_k) & base;
    sz_u64_t const class_line_feed = sz_line_break_class_mask_icelake_(classes, sz_line_break_lf_k) & base;
    sz_u64_t const class_next_line = sz_line_break_class_mask_icelake_(classes, sz_line_break_nl_k) & base;
    sz_u64_t const class_space = sz_line_break_class_mask_icelake_(classes, sz_line_break_sp_k) & base;
    sz_u64_t const class_zero_width_space = sz_line_break_class_mask_icelake_(classes, sz_line_break_zw_k) & base;
    sz_u64_t const class_word_joiner = sz_line_break_class_mask_icelake_(classes, sz_line_break_wj_k) & base;
    sz_u64_t const class_glue = sz_line_break_class_mask_icelake_(classes, sz_line_break_gl_k) & base;
    sz_u64_t const class_open_punctuation = sz_line_break_class_mask_icelake_(classes, sz_line_break_op_k) & base;
    sz_u64_t const class_close_punctuation = sz_line_break_class_mask_icelake_(classes, sz_line_break_cl_k) & base;
    sz_u64_t const class_close_parenthesis = sz_line_break_class_mask_icelake_(classes, sz_line_break_cp_k) & base;
    sz_u64_t const class_quotation = sz_line_break_class_mask_icelake_(classes, sz_line_break_qu_k) & base;
    sz_u64_t const class_nonstarter = sz_line_break_class_mask_icelake_(classes, sz_line_break_ns_k) & base;
    sz_u64_t const class_break_both = sz_line_break_class_mask_icelake_(classes, sz_line_break_b2_k) & base;
    sz_u64_t const class_hyphen = sz_line_break_class_mask_icelake_(classes, sz_line_break_hy_k) & base;
    sz_u64_t const class_unambiguous_hyphen = sz_line_break_class_mask_icelake_(classes, sz_line_break_hh_k) & base;
    sz_u64_t const class_break_after = sz_line_break_class_mask_icelake_(classes, sz_line_break_ba_k) & base;
    sz_u64_t const class_break_before = sz_line_break_class_mask_icelake_(classes, sz_line_break_bb_k) & base;
    sz_u64_t const class_inseparable = sz_line_break_class_mask_icelake_(classes, sz_line_break_in_k) & base;
    sz_u64_t const class_contingent_break = sz_line_break_class_mask_icelake_(classes, sz_line_break_cb_k) & base;
    sz_u64_t const class_infix_separator = sz_line_break_class_mask_icelake_(classes, sz_line_break_is_k) & base;
    sz_u64_t const class_numeric = sz_line_break_class_mask_icelake_(classes, sz_line_break_nu_k) & base;
    sz_u64_t const class_symbol = sz_line_break_class_mask_icelake_(classes, sz_line_break_sy_k) & base;
    sz_u64_t const class_exclamation = sz_line_break_class_mask_icelake_(classes, sz_line_break_ex_k) & base;
    sz_u64_t const class_postfix_numeric = sz_line_break_class_mask_icelake_(classes, sz_line_break_po_k) & base;
    sz_u64_t const class_prefix_numeric = sz_line_break_class_mask_icelake_(classes, sz_line_break_pr_k) & base;
    sz_u64_t const class_alphabetic = sz_line_break_class_mask_icelake_(classes, sz_line_break_al_k) & base;
    sz_u64_t const class_hebrew_letter = sz_line_break_class_mask_icelake_(classes, sz_line_break_hl_k) & base;
    sz_u64_t const class_ideographic = sz_line_break_class_mask_icelake_(classes, sz_line_break_id_k) & base;
    sz_u64_t const class_emoji_base = sz_line_break_class_mask_icelake_(classes, sz_line_break_eb_k) & base;
    sz_u64_t const class_emoji_modifier = sz_line_break_class_mask_icelake_(classes, sz_line_break_em_k) & base;
    sz_u64_t const class_regional_indicator = sz_line_break_class_mask_icelake_(classes, sz_line_break_ri_k) & base;
    //  class_aksara and class_aksara_start are also read by the cross-window carry-out below, so they stay at the top
    //  even though their rules live in the gated LB28a block. The remaining Brahmic masks (AP/VF/VI) and all five
    //  Hangul masks (JL/JV/JT/H2/H3) are used only inside their respective gated blocks and are extracted there.
    sz_u64_t const class_aksara = sz_line_break_class_mask_icelake_(classes, sz_line_break_ak_k) & base;
    sz_u64_t const class_aksara_start = sz_line_break_class_mask_icelake_(classes, sz_line_break_as_k) & base;
    sz_u64_t const zwj_starts = sz_line_break_class_mask_icelake_(classified.classes, sz_line_break_zwj_k) &
                                classified.starts;

    sz_u64_t const side_quote_initial = sz_line_break_side_mask_icelake_(side, sz_line_break_side_pi_k) & base;
    sz_u64_t const side_quote_final = sz_line_break_side_mask_icelake_(side, sz_line_break_side_pf_k) & base;
    sz_u64_t const side_east_asian_width = sz_line_break_side_mask_icelake_(side, sz_line_break_side_eaw_k) & base;

    sz_u64_t settled = 0, breaks = 0;
    sz_bool_t const carry_op = (sz_bool_t)(carry.open_sp_opener == sz_line_break_op_k);
    sz_bool_t const carry_clcp = (sz_bool_t)(carry.open_sp_opener == sz_line_break_cl_k ||
                                             carry.open_sp_opener == sz_line_break_cp_k);
    sz_bool_t const carry_b2 = (sz_bool_t)(carry.open_sp_opener == sz_line_break_b2_k);
    sz_bool_t const carry_zw = (sz_bool_t)(carry.open_sp_opener == sz_line_break_zw_k);
    //  "X SP*" run governance (LB8/14/16/17). The fill_right floods are skipped whenever no opener is present in the
    //  window AND none is carried open across the edge -- the dominant CJK/plain-text case -- leaving every gov empty
    //  (bit-identical, since prev_(0) contributes nothing and carry-out then finds no opener).
    sz_u64_t opener_governance = 0, close_governance = 0, break_both_governance = 0, zero_width_governance = 0;
    if ((class_open_punctuation | class_close_punctuation | class_close_parenthesis | class_break_both |
         class_zero_width_space) ||
        carry_op || carry_clcp || carry_b2 || carry_zw) {
        sz_u64_t const op_seed = class_open_punctuation | (carry_op && (class_space & first_base) ? first_base : 0);
        sz_u64_t const clcp_seed = (class_close_punctuation | class_close_parenthesis) |
                                   (carry_clcp && (class_space & first_base) ? first_base : 0);
        sz_u64_t const b2_seed = class_break_both | (carry_b2 && (class_space & first_base) ? first_base : 0);
        sz_u64_t const zw_seed = class_zero_width_space | (carry_zw && (class_space & first_base) ? first_base : 0);
        opener_governance = sz_line_break_run_byte_(op_seed, class_space, gate);
        close_governance = sz_line_break_run_byte_(clcp_seed, class_space, gate);
        break_both_governance = sz_line_break_run_byte_(b2_seed, class_space, gate);
        zero_width_governance = sz_line_break_run_byte_(zw_seed, class_space, gate);
    }
    //  LB15a "(sot|allowed) [QU&Pi] SP* x" governance, seeded by a carried open QU·Pi run when lane 0 continues it.
    sz_u64_t const quote_initial = class_quotation & side_quote_initial;
    sz_u64_t quote_initial_governance = 0;
    if (quote_initial || carry.qupi_sp_open) {
        sz_u64_t const qupi_allowed_left = class_break_mandatory | class_carriage_return | class_line_feed |
                                           class_next_line | class_open_punctuation | class_quotation | class_glue |
                                           class_space | class_zero_width_space;
        //  A QU·Pi opening at lane 0 may have its allowed-left cluster carried across the edge (LB15a left context).
        sz_bool_t const carry_qupi_left = (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_bk_k) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_cr_k) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_lf_k) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_nl_k) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_op_k) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_qu_k) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_gl_k) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_sp_k) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_zw_k));
        sz_u64_t qupi_seed = quote_initial &
                             (sz_line_break_prevc_(qupi_allowed_left, gate, carry_qupi_left, edge) | start_of_text);
        if (carry.qupi_sp_open && (class_space & first_base)) qupi_seed |= first_base;
        quote_initial_governance = sz_line_break_run_byte_(qupi_seed, class_space, gate);
    }

    // LB4: BK !
    sz_line_break_force_break_(
        sz_line_break_prevc_(class_break_mandatory, gate, sz_line_break_class_is_(left_bit, sz_line_break_bk_k), edge) &
            interior_lanes,
        base, &settled, &breaks);
    // LB5: CR x LF ; CR ! ; LF ! ; NL !
    sz_line_break_force_join_((sz_line_break_prevc_(class_carriage_return, gate,
                                                    sz_line_break_class_is_(left_bit, sz_line_break_cr_k), edge) &
                               class_line_feed) &
                                  interior_lanes,
                              base, &settled);
    sz_line_break_force_break_(sz_line_break_prevc_(class_carriage_return | class_line_feed | class_next_line, gate,
                                                    (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_cr_k) ||
                                                                sz_line_break_class_is_(left_bit, sz_line_break_lf_k) ||
                                                                sz_line_break_class_is_(left_bit, sz_line_break_nl_k)),
                                                    edge) &
                                   interior_lanes,
                               base, &settled, &breaks);
    // LB6: x (BK|CR|LF|NL)
    sz_line_break_force_join_(
        (class_break_mandatory | class_carriage_return | class_line_feed | class_next_line) & interior_lanes, base,
        &settled);
    // LB7: x SP ; x ZW
    sz_line_break_force_join_((class_space | class_zero_width_space) & interior_lanes, base, &settled);
    // LB8: ZW SP* /
    sz_line_break_force_break_(
        (sz_line_break_prevc_(zero_width_governance, gate, carry_zw, edge) & ~zero_width_governance) & interior_lanes,
        base, &settled, &breaks);
    // LB8a: ZWJ x (codepoint-level: previous codepoint is the original ZWJ class). A carried bare ZWJ joins lane 0.
    {
        sz_u64_t zwj_join_targets = (sz_utf8_codepoints_fill_right_(zwj_starts, non_start) << 1) & base;
        if (carry.prev_is_zwj) zwj_join_targets |= edge;
        sz_line_break_force_join_(zwj_join_targets & interior_lanes, base, &settled);
    }
    // LB9: x (attached CM|ZWJ) -- attached marks are never base lanes, so no break-before exists there (implicit).
    // LB11: x WJ ; WJ x
    sz_line_break_force_join_(
        (class_word_joiner |
         sz_line_break_prevc_(class_word_joiner, gate, sz_line_break_class_is_(left_bit, sz_line_break_wj_k), edge)) &
            interior_lanes,
        base, &settled);
    // LB12: GL x
    sz_line_break_force_join_(
        sz_line_break_prevc_(class_glue, gate, sz_line_break_class_is_(left_bit, sz_line_break_gl_k), edge) &
            interior_lanes,
        base, &settled);
    // LB12a: [^SP BA HY HH] x GL
    sz_u64_t const space_break_hyphen_prev = sz_line_break_prevc_(
        class_space | class_break_after | class_hyphen | class_unambiguous_hyphen, gate,
        (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_sp_k) ||
                    sz_line_break_class_is_(left_bit, sz_line_break_ba_k) ||
                    sz_line_break_class_is_(left_bit, sz_line_break_hy_k) ||
                    sz_line_break_class_is_(left_bit, sz_line_break_hh_k)),
        edge);
    sz_line_break_force_join_((class_glue & ~space_break_hyphen_prev) & interior_lanes, base, &settled);
    // LB13: x CL ; x CP ; x EX ; x SY
    sz_line_break_force_join_(
        (class_close_punctuation | class_close_parenthesis | class_exclamation | class_symbol) & interior_lanes, base,
        &settled);
    // LB14: OP SP* x. A carried open "OP SP*" run governs lane 0 (the opener/space may lie before the window edge).
    sz_line_break_force_join_(sz_line_break_prevc_(opener_governance, gate, carry_op, edge) & interior_lanes, base,
                              &settled);
    // LB15a: (sot|allowed) [QU&Pi] SP* x  (governance hoisted above with the other SP-run openers)
    sz_line_break_force_join_(
        sz_line_break_prevc_(quote_initial_governance, gate, (sz_bool_t)carry.qupi_sp_open, edge) & interior_lanes,
        base, &settled);
    // LB15b: x [QU&Pf] (followed by allowed | eot)
    {
        sz_u64_t const quote_final = class_quotation & side_quote_final;
        sz_u64_t const right_context_set = class_space | class_glue | class_word_joiner | class_close_punctuation |
                                           class_quotation | class_close_parenthesis | class_exclamation |
                                           class_infix_separator | class_symbol | class_break_mandatory |
                                           class_carriage_return | class_line_feed | class_next_line |
                                           class_zero_width_space;
        sz_u64_t const right_ok = sz_line_break_next_(right_context_set, gate) | (quote_final & end_of_text);
        sz_line_break_force_join_((quote_final & right_ok) & interior_lanes, base, &settled);
    }
    // LB15.3: SP / IS NU
    sz_u64_t const space_prev = sz_line_break_prevc_(class_space, gate,
                                                     sz_line_break_class_is_(left_bit, sz_line_break_sp_k), edge);
    sz_line_break_force_break_(
        (space_prev & class_infix_separator & sz_line_break_next_(class_numeric, gate)) & interior_lanes, base,
        &settled, &breaks);
    // LB15.4: x IS
    sz_line_break_force_join_(class_infix_separator & interior_lanes, base, &settled);
    // LB16: (CL|CP) SP* NS
    sz_line_break_force_join_(
        (class_nonstarter & sz_line_break_prevc_(close_governance, gate, carry_clcp, edge)) & interior_lanes, base,
        &settled);
    // LB17: B2 SP* B2
    sz_line_break_force_join_(
        (class_break_both & sz_line_break_prevc_(break_both_governance, gate, carry_b2, edge)) & interior_lanes, base,
        &settled);
    // LB18: SP /
    sz_line_break_force_break_(space_prev & interior_lanes, base, &settled, &breaks);
    // LB19 group (East-Asian-aware quotation); every term reads a QU cluster, so skip when none is present in the
    // window AND the carried left is neither a QU nor an East-Asian cluster that LB19 reads across the edge.
    if (class_quotation || sz_line_break_class_is_(left_bit, sz_line_break_qu_k) || left_is_east_asian_width ||
        left2_is_east_asian_width) {
        sz_u64_t const previous_east_asian_width = sz_line_break_prevc_(side_east_asian_width, gate,
                                                                        left_is_east_asian_width, edge);
        sz_u64_t const next_east_asian_width = sz_line_break_next_(side_east_asian_width, gate);
        sz_u64_t previous2_east_asian_width = sz_line_break_prev_(previous_east_asian_width & base, gate);
        if (left2_is_east_asian_width)
            previous2_east_asian_width |= edge; // lane 0's two-left cluster is the carried left2 (East-Asian)
        sz_u64_t const sot_next = is_sot ? (sz_line_break_prev_(first_base, gate) & base) : 0;
        sz_u64_t const quotation_prev = sz_line_break_prevc_(
            class_quotation, gate, sz_line_break_class_is_(left_bit, sz_line_break_qu_k), edge);
        sz_line_break_force_join_((class_quotation & ~side_quote_initial) & interior_lanes, base, &settled);
        sz_line_break_force_join_(
            sz_line_break_prevc_(class_quotation & ~side_quote_final, gate,
                                 (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_qu_k) && !carry.left_pf),
                                 edge) &
                interior_lanes,
            base, &settled);
        sz_line_break_force_join_((class_quotation & ~previous_east_asian_width) & interior_lanes, base, &settled);
        sz_line_break_force_join_((class_quotation & (~next_east_asian_width | end_of_text)) & interior_lanes, base,
                                  &settled);
        sz_line_break_force_join_((quotation_prev & ~side_east_asian_width) & interior_lanes, base, &settled);
        sz_line_break_force_join_((quotation_prev & (~previous2_east_asian_width | sot_next)) & interior_lanes, base,
                                  &settled);
    }
    // LB20: / CB ; CB /
    sz_line_break_force_break_(
        (class_contingent_break | sz_line_break_prevc_(class_contingent_break, gate,
                                                       sz_line_break_class_is_(left_bit, sz_line_break_cb_k), edge)) &
            interior_lanes,
        base, &settled, &breaks);
    // LB20a: (sot|allowed) (HY|HH) x (AL|HL); fires when a HY/HH cluster precedes an AL|HL, in-window or carried.
    if ((class_hyphen | class_unambiguous_hyphen) || sz_line_break_class_is_(left_bit, sz_line_break_hy_k) ||
        sz_line_break_class_is_(left_bit, sz_line_break_hh_k)) {
        sz_u64_t const leftset = class_break_mandatory | class_carriage_return | class_line_feed | class_next_line |
                                 class_space | class_zero_width_space | class_contingent_break | class_glue;
        //  The HY/HH cluster qualifies when preceded by an allowed-left cluster (or sot). At lane 0 the HY/HH is the
        //  carried left, so the allowed-left is the carried left2 (or sot if no left2).
        sz_bool_t const carry_hy_ok = (sz_bool_t)((sz_line_break_class_is_(left_bit, sz_line_break_hy_k) ||
                                                   sz_line_break_class_is_(left_bit, sz_line_break_hh_k)) &&
                                                  (left2_is_lb20a_allowed ||
                                                   sz_line_break_class_is_(left2_bit, sz_line_break_xx_k)));
        //  When the HY/HH itself is at lane 0, its allowed-left predecessor is the CARRIED left cluster -- the in-window
        //  prev_ cannot see it, so qualify the lane-0 HY/HH via the carried left class in the allowed set (same classes
        //  as `lb20a_allowed_mask`: BK, CR, LF, NL, SP, ZW, CB, GL).
        sz_bool_t const left_is_lb20a_allowed = sz_line_break_class_in_(left_bit, lb20a_allowed_mask);
        sz_u64_t const hy_ok = (class_hyphen | class_unambiguous_hyphen) &
                               (sz_line_break_prevc_(leftset, gate, left_is_lb20a_allowed, edge) | start_of_text);
        sz_line_break_force_join_(
            (sz_line_break_prevc_(hy_ok, gate, carry_hy_ok, edge) & (class_alphabetic | class_hebrew_letter)) &
                interior_lanes,
            base, &settled);
    }
    // LB21: x BA ; x HY ; x HH ; x NS ; BB x
    sz_line_break_force_join_(
        (class_break_after | class_hyphen | class_unambiguous_hyphen | class_nonstarter |
         sz_line_break_prevc_(class_break_before, gate, sz_line_break_class_is_(left_bit, sz_line_break_bb_k), edge)) &
            interior_lanes,
        base, &settled);
    // LB21a: HL (HY|HH) x [^HL]; fires when an HL-preceded HY/HH cluster precedes a non-HL, in-window or carried.
    // The HL anchor may itself be the carried left cluster while the HY/HH sits at in-window lane 0, so the gate
    // must also fire on a carried-left HL (otherwise the block is skipped and LB31 spuriously breaks before the x).
    if ((class_hebrew_letter && (class_hyphen | class_unambiguous_hyphen)) ||
        sz_line_break_class_is_(left_bit, sz_line_break_hl_k) ||
        ((sz_line_break_class_is_(left_bit, sz_line_break_hy_k) ||
          sz_line_break_class_is_(left_bit, sz_line_break_hh_k)) &&
         sz_line_break_class_is_(left2_bit, sz_line_break_hl_k))) {
        sz_bool_t const carry_hl_hy = (sz_bool_t)((sz_line_break_class_is_(left_bit, sz_line_break_hy_k) ||
                                                   sz_line_break_class_is_(left_bit, sz_line_break_hh_k)) &&
                                                  sz_line_break_class_is_(left2_bit, sz_line_break_hl_k));
        //  A lane-0 HY/HH whose HL predecessor is the carried left cluster: prev_ misses it, so use prevc_ with Lc_HL.
        sz_u64_t const hl_hy = sz_line_break_prevc_(class_hebrew_letter, gate,
                                                    sz_line_break_class_is_(left_bit, sz_line_break_hl_k), edge) &
                               (class_hyphen | class_unambiguous_hyphen);
        sz_line_break_force_join_(
            (sz_line_break_prevc_(hl_hy, gate, carry_hl_hy, edge) & ~class_hebrew_letter) & interior_lanes, base,
            &settled);
    }
    // LB21b: SY x HL
    sz_line_break_force_join_(
        (sz_line_break_prevc_(class_symbol, gate, sz_line_break_class_is_(left_bit, sz_line_break_sy_k), edge) &
         class_hebrew_letter) &
            interior_lanes,
        base, &settled);
    // LB22: x IN
    sz_line_break_force_join_(class_inseparable & interior_lanes, base, &settled);
    // LB23 + LB24: alphabetic/numeric adjacency; fires when an AL/HL cluster is present, in-window or carried left.
    sz_bool_t const alpha_hebrew_active = (sz_bool_t)((class_alphabetic | class_hebrew_letter) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_al_k) ||
                                                      sz_line_break_class_is_(left_bit, sz_line_break_hl_k));
    sz_u64_t const alpha_hebrew_prev = alpha_hebrew_active
                                           ? sz_line_break_prevc_(
                                                 class_alphabetic | class_hebrew_letter, gate,
                                                 (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_al_k) ||
                                                             sz_line_break_class_is_(left_bit, sz_line_break_hl_k)),
                                                 edge)
                                           : 0;
    if (alpha_hebrew_active) {
        sz_u64_t const prefix_postfix_prev = sz_line_break_prevc_(
            class_prefix_numeric | class_postfix_numeric, gate,
            (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_pr_k) ||
                        sz_line_break_class_is_(left_bit, sz_line_break_po_k)),
            edge);
        // LB23: (AL|HL) x NU ; NU x (AL|HL)
        sz_line_break_force_join_((alpha_hebrew_prev & class_numeric) & interior_lanes, base, &settled);
        sz_line_break_force_join_(
            (sz_line_break_prevc_(class_numeric, gate, sz_line_break_class_is_(left_bit, sz_line_break_nu_k), edge) &
             (class_alphabetic | class_hebrew_letter)) &
                interior_lanes,
            base, &settled);
        // LB24: (PR|PO) x (AL|HL) ; (AL|HL) x (PR|PO)
        sz_line_break_force_join_((prefix_postfix_prev & (class_alphabetic | class_hebrew_letter)) & interior_lanes,
                                  base, &settled);
        sz_line_break_force_join_((alpha_hebrew_prev & (class_prefix_numeric | class_postfix_numeric)) & interior_lanes,
                                  base, &settled);
    }
    // LB23a: PR x (ID|EB|EM) ; (ID|EB|EM) x PO  (fires for CJK ID, kept ungated)
    sz_u64_t const prefix_numeric_prev = sz_line_break_prevc_(
        class_prefix_numeric, gate, sz_line_break_class_is_(left_bit, sz_line_break_pr_k), edge);
    sz_line_break_force_join_(
        (prefix_numeric_prev & (class_ideographic | class_emoji_base | class_emoji_modifier)) & interior_lanes, base,
        &settled);
    sz_line_break_force_join_((sz_line_break_prevc_(class_ideographic | class_emoji_base | class_emoji_modifier, gate,
                                                    (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_id_k) ||
                                                                sz_line_break_class_is_(left_bit, sz_line_break_eb_k) ||
                                                                sz_line_break_class_is_(left_bit, sz_line_break_em_k)),
                                                    edge) &
                               class_postfix_numeric) &
                                  interior_lanes,
                              base, &settled);
    // LB25: numeric clusters
    if (class_numeric || carry.in_nu_run || carry.in_nu_close) {
        sz_u64_t const symbol_or_infix = class_symbol | class_infix_separator;
        sz_u64_t const nu_seed = class_numeric |
                                 (carry.in_nu_run && ((class_numeric | symbol_or_infix) & first_base) ? first_base : 0);
        sz_u64_t const numeric_run = sz_utf8_codepoints_fill_right_(nu_seed, gate | symbol_or_infix) &
                                     (nu_seed | symbol_or_infix);
        sz_u64_t const numeric_run_prev = sz_line_break_prevc_(numeric_run, gate, (sz_bool_t)carry.in_nu_run, edge);
        //  A CL/CP closing a NU run; the run may have closed at the carried left (carry.in_nu_close).
        sz_u64_t const numeric_run_close = numeric_run_prev & (class_close_punctuation | class_close_parenthesis);
        //  Hoisted lookahead predicates for the PO/PR x OP IS? NU forms (LB25): the cluster one ahead is NU, one
        //  ahead is IS, and (for the OP IS NU form) the cluster two ahead is NU.
        sz_u64_t const next_numeric = sz_line_break_next_(class_numeric, gate);
        sz_u64_t const next_infix = sz_line_break_next_(class_infix_separator, gate);
        sz_u64_t const numeric_two_ahead = sz_line_break_next_(next_numeric & base, gate);
        sz_u64_t const postfix_prev = sz_line_break_prevc_(class_postfix_numeric, gate,
                                                           sz_line_break_class_is_(left_bit, sz_line_break_po_k), edge);
        sz_u64_t const prefix_prev = prefix_numeric_prev;
        sz_line_break_force_join_((sz_line_break_prevc_(numeric_run_close, gate, (sz_bool_t)carry.in_nu_close, edge) &
                                   (class_postfix_numeric | class_prefix_numeric)) &
                                      interior_lanes,
                                  base, &settled);
        sz_line_break_force_join_((numeric_run_prev & (class_postfix_numeric | class_prefix_numeric)) & interior_lanes,
                                  base, &settled);
        sz_line_break_force_join_((postfix_prev & class_open_punctuation & next_numeric) & interior_lanes, base,
                                  &settled);
        sz_line_break_force_join_(
            (postfix_prev & class_open_punctuation & next_infix & numeric_two_ahead) & interior_lanes, base, &settled);
        sz_line_break_force_join_((postfix_prev & class_numeric) & interior_lanes, base, &settled);
        sz_line_break_force_join_((prefix_prev & class_open_punctuation & next_numeric) & interior_lanes, base,
                                  &settled);
        sz_line_break_force_join_(
            (prefix_prev & class_open_punctuation & next_infix & numeric_two_ahead) & interior_lanes, base, &settled);
        sz_line_break_force_join_((prefix_prev & class_numeric) & interior_lanes, base, &settled);
        sz_line_break_force_join_(
            (sz_line_break_prevc_(class_hyphen, gate, sz_line_break_class_is_(left_bit, sz_line_break_hy_k), edge) &
             class_numeric) &
                interior_lanes,
            base, &settled);
        sz_line_break_force_join_((sz_line_break_prevc_(class_infix_separator, gate,
                                                        sz_line_break_class_is_(left_bit, sz_line_break_is_k), edge) &
                                   class_numeric) &
                                      interior_lanes,
                                  base, &settled);
        sz_line_break_force_join_((numeric_run_prev & class_numeric) & interior_lanes, base, &settled);
    }
    // LB26 + LB27: Hangul. The window-presence half of the gate is a cheap combined range test over the Hangul class
    // bytes -- H2/H3 (33,34) and JL/JV/JT (37,38,39), with HL(35)/ID(36) excluded by the gap between the two ranges --
    // so the five individual JL/JV/JT/H2/H3 masks are extracted only when the block actually fires.
    sz_u64_t const class_hangul_present =
        (sz_line_break_class_range_mask_icelake_(classes, sz_line_break_h2_k, sz_line_break_h3_k) |
         sz_line_break_class_range_mask_icelake_(classes, sz_line_break_jl_k, sz_line_break_jt_k)) &
        base;
    if (class_hangul_present || sz_line_break_class_is_(left_bit, sz_line_break_jl_k) ||
        sz_line_break_class_is_(left_bit, sz_line_break_jv_k) ||
        sz_line_break_class_is_(left_bit, sz_line_break_jt_k) ||
        sz_line_break_class_is_(left_bit, sz_line_break_h2_k) ||
        sz_line_break_class_is_(left_bit, sz_line_break_h3_k) ||
        sz_line_break_class_is_(left_bit, sz_line_break_pr_k)) {
        sz_u64_t const class_hangul_l_jamo = sz_line_break_class_mask_icelake_(classes, sz_line_break_jl_k) & base;
        sz_u64_t const class_hangul_v_jamo = sz_line_break_class_mask_icelake_(classes, sz_line_break_jv_k) & base;
        sz_u64_t const class_hangul_t_jamo = sz_line_break_class_mask_icelake_(classes, sz_line_break_jt_k) & base;
        sz_u64_t const class_hangul_lv_syllable = sz_line_break_class_mask_icelake_(classes, sz_line_break_h2_k) & base;
        sz_u64_t const class_hangul_lvt_syllable = sz_line_break_class_mask_icelake_(classes, sz_line_break_h3_k) &
                                                   base;
        sz_line_break_force_join_(
            (sz_line_break_prevc_(class_hangul_l_jamo, gate, sz_line_break_class_is_(left_bit, sz_line_break_jl_k),
                                  edge) &
             (class_hangul_l_jamo | class_hangul_v_jamo | class_hangul_lv_syllable | class_hangul_lvt_syllable)) &
                interior_lanes,
            base, &settled);
        sz_line_break_force_join_(
            (sz_line_break_prevc_(class_hangul_v_jamo | class_hangul_lv_syllable, gate,
                                  (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_jv_k) ||
                                              sz_line_break_class_is_(left_bit, sz_line_break_h2_k)),
                                  edge) &
             (class_hangul_v_jamo | class_hangul_t_jamo)) &
                interior_lanes,
            base, &settled);
        sz_line_break_force_join_(
            (sz_line_break_prevc_(class_hangul_t_jamo | class_hangul_lvt_syllable, gate,
                                  (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_jt_k) ||
                                              sz_line_break_class_is_(left_bit, sz_line_break_h3_k)),
                                  edge) &
             class_hangul_t_jamo) &
                interior_lanes,
            base, &settled);
        sz_line_break_force_join_(
            (sz_line_break_prevc_(class_hangul_l_jamo | class_hangul_v_jamo | class_hangul_t_jamo |
                                      class_hangul_lv_syllable | class_hangul_lvt_syllable,
                                  gate,
                                  (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_jl_k) ||
                                              sz_line_break_class_is_(left_bit, sz_line_break_jv_k) ||
                                              sz_line_break_class_is_(left_bit, sz_line_break_jt_k) ||
                                              sz_line_break_class_is_(left_bit, sz_line_break_h2_k) ||
                                              sz_line_break_class_is_(left_bit, sz_line_break_h3_k)),
                                  edge) &
             class_postfix_numeric) &
                interior_lanes,
            base, &settled);
        sz_line_break_force_join_(
            (prefix_numeric_prev & (class_hangul_l_jamo | class_hangul_v_jamo | class_hangul_t_jamo |
                                    class_hangul_lv_syllable | class_hangul_lvt_syllable)) &
                interior_lanes,
            base, &settled);
    }
    // LB28: (AL|HL) x (AL|HL)
    if (alpha_hebrew_active)
        sz_line_break_force_join_((alpha_hebrew_prev & (class_alphabetic | class_hebrew_letter)) & interior_lanes, base,
                                  &settled);
    // LB28a: aksara (Brahmic); DottedCircle acts as an aksara base
    {
        sz_bool_t const left_is_aksara = (sz_bool_t)(carry.have_prev && carry.left_aksara != 0);
        sz_bool_t const left2_is_aksara = (sz_bool_t)(carry.have_prev && carry.left2_aksara != 0);
        //  The window-presence half of the gate is a cheap combined range test over the contiguous Brahmic class bytes
        //  AK..VI (43..47), which covers AK/AP/AS/VF/VI exactly, ORed with the separately-held DottedCircle. The AP/VF/VI
        //  masks live only inside this block, so they are extracted here rather than at the top.
        sz_u64_t const class_brahmic_present =
            (sz_line_break_class_range_mask_icelake_(classes, sz_line_break_ak_k, sz_line_break_vi_k) & base) |
            dotted_circle;
        if (class_brahmic_present || sz_line_break_class_is_(left_bit, sz_line_break_ap_k) || left_is_aksara ||
            (sz_line_break_class_is_(left_bit, sz_line_break_vi_k) && left2_is_aksara)) {
            sz_u64_t const class_aksara_prebase = sz_line_break_class_mask_icelake_(classes, sz_line_break_ap_k) & base;
            sz_u64_t const class_virama_final = sz_line_break_class_mask_icelake_(classes, sz_line_break_vf_k) & base;
            sz_u64_t const class_virama = sz_line_break_class_mask_icelake_(classes, sz_line_break_vi_k) & base;
            sz_u64_t const aksara = class_aksara | class_aksara_start | dotted_circle;
            sz_u64_t const aksara_dc = class_aksara | dotted_circle;
            sz_u64_t const aksara_prev = sz_line_break_prevc_(aksara, gate, left_is_aksara, edge);
            sz_line_break_force_join_(
                (sz_line_break_prevc_(class_aksara_prebase, gate, sz_line_break_class_is_(left_bit, sz_line_break_ap_k),
                                      edge) &
                 aksara) &
                    interior_lanes,
                base, &settled);
            sz_line_break_force_join_((aksara_prev & (class_virama_final | class_virama)) & interior_lanes, base,
                                      &settled);
            {
                sz_u64_t base_vi = aksara_prev & class_virama;
                sz_line_break_force_join_(
                    (sz_line_break_prevc_(
                         base_vi, gate,
                         (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_vi_k) && left2_is_aksara), edge) &
                     aksara_dc) &
                        interior_lanes,
                    base, &settled);
            }
            sz_line_break_force_join_(
                (aksara_prev & aksara & sz_line_break_next_(class_virama_final, gate)) & interior_lanes, base,
                &settled);
        }
    }
    // LB29: IS x (AL|HL)
    if ((class_alphabetic | class_hebrew_letter) || sz_line_break_class_is_(left_bit, sz_line_break_is_k))
        sz_line_break_force_join_((sz_line_break_prevc_(class_infix_separator, gate,
                                                        sz_line_break_class_is_(left_bit, sz_line_break_is_k), edge) &
                                   (class_alphabetic | class_hebrew_letter)) &
                                      interior_lanes,
                                  base, &settled);
    // LB30: (AL|HL|NU) x OP[^EAW] ; CP[^EAW] x (AL|HL|NU); requires an OP or CP cluster, in-window or carried CP.
    if ((class_open_punctuation | class_close_parenthesis) || sz_line_break_class_is_(left_bit, sz_line_break_cp_k)) {
        sz_line_break_force_join_(
            (sz_line_break_prevc_(class_alphabetic | class_hebrew_letter | class_numeric, gate,
                                  (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_al_k) ||
                                              sz_line_break_class_is_(left_bit, sz_line_break_hl_k) ||
                                              sz_line_break_class_is_(left_bit, sz_line_break_nu_k)),
                                  edge) &
             class_open_punctuation & ~side_east_asian_width) &
                interior_lanes,
            base, &settled);
        //  CP[^EAW]: a carried left CP qualifies only when it is NOT East-Asian-wide.
        sz_line_break_force_join_(
            (sz_line_break_prevc_(class_close_parenthesis & ~side_east_asian_width, gate,
                                  (sz_bool_t)(sz_line_break_class_is_(left_bit, sz_line_break_cp_k) && !carry.left_eaw),
                                  edge) &
             (class_alphabetic | class_hebrew_letter | class_numeric)) &
                interior_lanes,
            base, &settled);
    }
    // LB30a: RI RI even-parity pairing. The inclusive segmented prefix-XOR over each maximal RI run gives, at each RI,
    // the parity of indicators at-or-before it; the 2nd of each pair (even inclusive parity) is joined. A run
    // straddling in from the previous window toggles the leading run via the inbound-parity seed.
    if (class_regional_indicator) {
        sz_u64_t const ri_run_gate = sz_utf8_codepoints_fill_right_(class_regional_indicator, gate) |
                                     class_regional_indicator;
        sz_bool_t const inbound = (sz_bool_t)(carry.ri_open && carry.ri_parity_odd);
        sz_u64_t const inclusive = sz_line_break_segmented_parity_(class_regional_indicator, ri_run_gate, inbound);
        sz_line_break_force_join_((class_regional_indicator & ~inclusive) & interior_lanes, base, &settled);
    }
    // LB30b: EB x EM ; [ExtPict & Cn] x EM. The ExtPict/Cn side masks feed only this block, so extract them here.
    if (class_emoji_modifier) {
        sz_u64_t const side_unassigned = sz_line_break_side_mask_icelake_(side, sz_line_break_side_cn_k) & base;
        sz_u64_t const side_extended_pictographic = sz_line_break_side_mask_icelake_(side, sz_line_break_side_ext_k) &
                                                    base;
        sz_line_break_force_join_(
            (sz_line_break_prevc_(class_emoji_base, gate, sz_line_break_class_is_(left_bit, sz_line_break_eb_k), edge) &
             class_emoji_modifier) &
                interior_lanes,
            base, &settled);
        sz_line_break_force_join_((sz_line_break_prevc_(side_extended_pictographic & side_unassigned, gate,
                                                        (sz_bool_t)carry.left_extpict_cn, edge) &
                                   class_emoji_modifier) &
                                      interior_lanes,
                                  base, &settled);
    }

    // LB31: break everywhere else.
    breaks |= interior_lanes & ~settled;

    //  Trust horizon: the lowest lane whose break-before verdict needs context past the window edge. Open unbounded
    //  lookahead runs (NU runs, "X SP*" opener runs, "[QU&Pi] SP*" runs) that flood to the top valid lane are
    //  undecided, and the bounded +2-codepoint lookahead (LB19 East-Asian, LB25 PO/PR OP IS? NU) leaves the top two
    //  codepoints of the complete region undecided. At the buffer tail there is no next window, so nothing clamps.
    sz_u64_t const symbol_or_infix_all = class_symbol | class_infix_separator;
    sz_u64_t const nu_seed_all =
        class_numeric | (carry.in_nu_run && ((class_numeric | symbol_or_infix_all) & first_base) ? first_base : 0);
    sz_u64_t const numeric_run_all = sz_utf8_codepoints_fill_right_(nu_seed_all, gate | symbol_or_infix_all) &
                                     (nu_seed_all | symbol_or_infix_all);
    sz_size_t resolved = complete_limit;
    if (more_text) {
        sz_u64_t const complete_mask = sz_u64_mask_until_(complete_limit);
        sz_u64_t const complete_base = base & complete_mask;
        sz_size_t const top_base = complete_base ? (sz_size_t)(63 - sz_u64_clz(complete_base)) : 0;
        sz_u64_t const top_bit = complete_base ? (1ull << top_base) : 0ull;
        //  (a) NU run open at the top of the complete region: its closer / PO|PR suffix may lie past the edge.
        sz_u64_t undecided = (numeric_run_all & top_bit) ? top_bit : 0ull;
        //  (b) "X SP*" opener runs open at the top: the governed `x` lies past the edge.
        undecided |= ((opener_governance | close_governance | break_both_governance | zero_width_governance |
                       quote_initial_governance) &
                      top_bit);
        if (undecided) {
            //  Flood the open construct back left across its gate to the run's lowest lane: the next window re-derives
            //  the verdict there with the carried run-state, so trust ends at that base.
            sz_u64_t const run_gate = gate | symbol_or_infix_all | class_space;
            sz_u64_t const reaching = base & sz_utf8_codepoints_fill_left_(undecided, run_gate);
            sz_size_t const low_lane = reaching ? (sz_size_t)sz_u64_ctz(reaching) : top_base;
            if (low_lane < resolved) resolved = low_lane;
        }
        //  (d) Bounded +2-codepoint lookahead: hold back the top two complete codepoints (their right context may be
        //  out of window) unless they were already settled by an unconditional rule (which never reads past the edge).
        sz_size_t const lookahead = 2;
        sz_size_t count = 0, lane = top_base;
        for (;;) {
            sz_u64_t const lane_mask = sz_u64_mask_until_(lane);
            sz_u64_t const lower = complete_base & lane_mask;
            if (!lower || count + 1 >= lookahead) {
                if (lane < resolved) resolved = lane;
                break;
            }
            lane = (sz_size_t)(63 - sz_u64_clz(lower));
            ++count;
        }
        if (resolved > complete_limit) resolved = complete_limit;
    }
    breaks &= sz_u64_mask_until_(resolved);

    //  The driver advances by `advance` bytes after this call: the clamp horizon `resolved` when it bit before the
    //  complete edge, else the whole complete span. The carry anchors at THAT byte so a single decision rebuilds the
    //  next window's exact left context -- no second pass. (`resolved == 0` is an unbounded run longer than one window;
    //  the driver steps the full complete span for progress, so the carry anchors at the complete edge too.)
    sz_size_t const advance = resolved ? resolved : complete_limit;

    //  Cross-window carry: run state of the cluster ending at the highest base lane below `advance` (the next window's
    //  left context). An all-ignorable region below `advance` threads the inbound carry unchanged.
    if (carry_out) {
        sz_line_break_carry_t out = carry;
        sz_u64_t const below = base & sz_u64_mask_until_(advance);
        if (below) {
            //  At least one cluster resolved: rebuild the full left context + run-state at the complete edge.
            out.open_sp_opener = 0xFF, out.in_nu_run = 0, out.in_nu_close = 0, out.ri_open = 0, out.ri_parity_odd = 0,
            out.qupi_sp_open = 0;
            sz_size_t const previous_cluster_lane = (sz_size_t)(63 - sz_u64_clz(below));
            sz_u64_t const previous_cluster_bit = 1ull << previous_cluster_lane; // prev cluster's base lane
            sz_u64_t const below2 = below & ~previous_cluster_bit;
            sz_size_t const previous2_cluster_lane = below2 ? (sz_size_t)(63 - sz_u64_clz(below2)) : 0;
            sz_u64_t const previous2_cluster_bit = below2 ? (1ull << previous2_cluster_lane) : 0ull;

            out.have_prev = 1;
            out.previous_class_bit = 1ull << sz_line_break_byte_at_icelake_(classes, previous_cluster_lane);
            sz_u8_t const side_pbit = sz_line_break_byte_at_icelake_(side, previous_cluster_lane);
            out.left_eaw = (sz_u8_t)((side_pbit & sz_line_break_side_eaw_k) != 0);
            out.left_pf = (sz_u8_t)((side_pbit & sz_line_break_side_pf_k) != 0);
            out.left_extpict_cn = (sz_u8_t)(((side_pbit & sz_line_break_side_ext_k) != 0) &&
                                            ((side_pbit & sz_line_break_side_cn_k) != 0));
            out.left_aksara = (sz_u8_t)(((class_aksara | class_aksara_start) & previous_cluster_bit) != 0 ||
                                        (dotted_circle & previous_cluster_bit) != 0);
            //  LB8a is codepoint-level: the previous CLUSTER joins lane 0 when its LAST codepoint is a bare ZWJ -- that
            //  ZWJ is an attached mark (LB9), so it sits ABOVE the base `pbit`, not on it. Test the highest codepoint
            //  START below the edge, not the highest base.
            sz_u64_t const starts_below = classified.starts & sz_u64_mask_until_(advance);
            sz_u64_t const last_start = starts_below ? (1ull << (63 - sz_u64_clz(starts_below))) : 0ull;
            out.prev_is_zwj = (sz_u8_t)((zwj_starts & last_start) != 0);
            //  The two-left context is the prior resolved base, or -- when only one cluster resolved here -- the inbound
            //  carry's left cluster (which lay just before that single resolved base).
            if (below2) {
                out.previous2_class_bit = 1ull << sz_line_break_byte_at_icelake_(classes, previous2_cluster_lane);
                sz_u8_t const side_p2bit = sz_line_break_byte_at_icelake_(side, previous2_cluster_lane);
                out.left2_eaw = (sz_u8_t)((side_p2bit & sz_line_break_side_eaw_k) != 0);
                out.left2_aksara = (sz_u8_t)(((class_aksara | class_aksara_start) & previous2_cluster_bit) != 0 ||
                                             (dotted_circle & previous2_cluster_bit) != 0);
            }
            else {
                out.previous2_class_bit = carry.have_prev ? carry.previous_class_bit : (1ull << sz_line_break_xx_k);
                out.left2_eaw = carry.have_prev ? carry.left_eaw : (sz_u8_t)0;
                out.left2_aksara = carry.have_prev ? carry.left_aksara : (sz_u8_t)0;
            }

            out.qupi_sp_open = (sz_u8_t)((quote_initial_governance & previous_cluster_bit) &&
                                         ((class_space | quote_initial) & previous_cluster_bit));
            if ((opener_governance & previous_cluster_bit) &&
                ((class_space | class_open_punctuation) & previous_cluster_bit))
                out.open_sp_opener = sz_line_break_op_k;
            else if ((close_governance & previous_cluster_bit) &&
                     ((class_space | class_close_punctuation | class_close_parenthesis) & previous_cluster_bit))
                out.open_sp_opener = (class_close_parenthesis & previous_cluster_bit) ? (sz_u8_t)sz_line_break_cp_k
                                                                                      : (sz_u8_t)sz_line_break_cl_k;
            else if ((break_both_governance & previous_cluster_bit) &&
                     ((class_space | class_break_both) & previous_cluster_bit))
                out.open_sp_opener = sz_line_break_b2_k;
            else if ((zero_width_governance & previous_cluster_bit) &&
                     ((class_space | class_zero_width_space) & previous_cluster_bit))
                out.open_sp_opener = sz_line_break_zw_k;
            out.in_nu_run = (sz_u8_t)((numeric_run_all & previous_cluster_bit) != 0);
            //  A CL/CP closing a NU run (its predecessor cluster is in the run): seeds LB25 nu_run_close next window.
            sz_u64_t const nu_run_close_all = sz_line_break_prevc_(numeric_run_all, gate, (sz_bool_t)carry.in_nu_run,
                                                                   edge) &
                                              (class_close_punctuation | class_close_parenthesis);
            out.in_nu_close = (sz_u8_t)((nu_run_close_all & previous_cluster_bit) != 0);
            sz_u64_t const ri_run_gate = sz_utf8_codepoints_fill_right_(class_regional_indicator, gate) |
                                         class_regional_indicator;
            sz_bool_t const inbound = (sz_bool_t)(carry.ri_open && carry.ri_parity_odd);
            sz_u64_t const inclusive = sz_line_break_segmented_parity_(class_regional_indicator, ri_run_gate, inbound);
            out.ri_open = (sz_u8_t)((class_regional_indicator & previous_cluster_bit) != 0);
            out.ri_parity_odd = (sz_u8_t)((inclusive & previous_cluster_bit) != 0);
        }
        *carry_out = out;
    }
    sz_line_break_window_t result;
    result.breaks = breaks;
    result.resolved = resolved;
    return result;
}

#pragma endregion Mask algebra rule engine

#pragma region Forward driver

/**
 *  @brief  Largest byte prefix of the window whose codepoints are all fully loaded (no multi-byte lead straddles the
 *          64-byte edge). Mirrors the word kernel's complete-limit: a declared-length lead whose span exceeds `loaded`
 *          ends the trusted region just before it; with no more text the whole window is complete. Never below 1.
 */
SZ_INTERNAL sz_size_t sz_line_break_complete_limit_(sz_utf8_codepoints_window_t window, sz_bool_t more_text) {
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
SZ_PUBLIC sz_size_t sz_utf8_linewraps_icelake_bytes_( //
    sz_cptr_t text, sz_size_t length,                 //
    sz_size_t *starts, sz_size_t *lengths,            //
    sz_size_t capacity, sz_size_t *bytes_consumed) {

    if (length == 0 || capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *bytes = (sz_u8_t const *)text;
    __m512i const lane_identity = sz_utf8_codepoints_lane_identity_icelake_();
    sz_size_t produced = 0;
    sz_size_t line_start = 0;                                 // open line's first byte: a confirmed break (or 0)
    sz_size_t position = 0;                                   // codepoint-aligned anchor of the next window
    sz_line_break_carry_t carry = sz_line_break_carry_sot_(); // LB2: start-of-text has no left context

    while (position < length) {
        sz_utf8_codepoints_window_t const window = sz_utf8_codepoints_decode_window_(bytes + position,
                                                                                     length - position, lane_identity);
        sz_bool_t const more_text = (sz_bool_t)(position + window.loaded < length);
        sz_size_t const complete_limit = sz_line_break_complete_limit_(window, more_text);
        sz_line_break_classified_t const classified = sz_line_break_classify_window_icelake_(window, lane_identity);

        sz_line_break_carry_t carry_next = carry;
        sz_line_break_window_t const win = sz_line_break_decide_window_icelake_(classified, carry, &carry_next,
                                                                                complete_limit, more_text);
        sz_u64_t const commit = win.breaks & sz_u64_mask_until_(win.resolved);

        produced = sz_utf8_codepoints_drain_forward_(commit, position, lane_identity, starts, lengths, produced,
                                                     capacity, &line_start);
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
 *  Bit-exact with `sz_utf8_linewraps_serial`. Emits every UAX-14 break opportunity (no per-segment
 *  mandatory flag). The classifier and rule engine are fully vectorized (no per-lane scalar loop, no `vpgather`,
 *  no serial-oracle deferral); the 64-codepoint block engine threads cross-block state through a register carry
 *  (no halo back-scan), so throughput is flat in run length. Emits at most @p capacity segments; sets
 *  *@p bytes_consumed to the resume.
 */
SZ_PUBLIC sz_size_t sz_utf8_linewraps_icelake( //
    sz_cptr_t text, sz_size_t length,          //
    sz_size_t *starts, sz_size_t *lengths,     //
    sz_size_t capacity, sz_size_t *bytes_consumed) {

    return sz_utf8_linewraps_icelake_bytes_(text, length, starts, lengths, capacity, bytes_consumed);
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

#endif // STRINGZILLA_UTF8_LINEWRAPS_ICELAKE_H_
