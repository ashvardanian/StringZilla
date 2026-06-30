/**
 *  @brief LoongArch LASX backend for UAX-29 word boundaries.
 *  @file include/stringzilla/utf8_wordbreaks/lasx.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_WORDS_LASX_H_
#define STRINGZILLA_UTF8_WORDS_LASX_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_wordbreaks/tables.h"
#include "stringzilla/utf8_wordbreaks/serial.h"
#include "stringzilla/utf8_runes/lasx.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_LASX
/** @brief  Per-class 32-bit lane boundary mask over an all-ASCII LASX byte compare. Each Word_Break class is a
 *          small ASCII byte set (ALetter A-Za-z, Numeric 0-9, ExtendNumLet '_', Mid* punctuation, CR and LF). */
SZ_HELPER_AUTO sz_u32_t sz_utf8_word_break_boundary_mask_lasx_(__m256i window) {
    __m256i lowered = __lasx_xvor_v(window, __lasx_xvreplgr2vr_b(0x20)); // fold A-Z onto a-z
    sz_u64_t aletter_mask = (sz_u64_t)sz_xvmovemask_b_utf8_lasx_(__lasx_xvand_v(
        __lasx_xvsle_bu(__lasx_xvreplgr2vr_b(0x61), lowered), __lasx_xvsle_bu(lowered, __lasx_xvreplgr2vr_b(0x7A))));
    sz_u64_t numeric_mask = (sz_u64_t)sz_xvmovemask_b_utf8_lasx_(__lasx_xvand_v(
        __lasx_xvsle_bu(__lasx_xvreplgr2vr_b(0x30), window), __lasx_xvsle_bu(window, __lasx_xvreplgr2vr_b(0x39))));
    sz_u64_t extendnumlet_mask = (sz_u64_t)sz_xvmovemask_b_utf8_lasx_(
        __lasx_xvseq_b(window, __lasx_xvreplgr2vr_b(0x5F)));
    sz_u64_t midletter_mask = (sz_u64_t)sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, __lasx_xvreplgr2vr_b(0x3A)));
    sz_u64_t midnum_mask = (sz_u64_t)sz_xvmovemask_b_utf8_lasx_(__lasx_xvor_v(
        __lasx_xvseq_b(window, __lasx_xvreplgr2vr_b(0x2C)), __lasx_xvseq_b(window, __lasx_xvreplgr2vr_b(0x3B))));
    sz_u64_t mid_quotes_mask = (sz_u64_t)sz_xvmovemask_b_utf8_lasx_(
        __lasx_xvor_v(__lasx_xvor_v(__lasx_xvseq_b(window, __lasx_xvreplgr2vr_b(0x22)),
                                    __lasx_xvseq_b(window, __lasx_xvreplgr2vr_b(0x27))),
                      __lasx_xvseq_b(window, __lasx_xvreplgr2vr_b(0x2E))));
    sz_u64_t carriage_return_mask = (sz_u64_t)sz_xvmovemask_b_utf8_lasx_(
        __lasx_xvseq_b(window, __lasx_xvreplgr2vr_b(0x0D)));
    sz_u64_t line_feed_mask = (sz_u64_t)sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, __lasx_xvreplgr2vr_b(0x0A)));
    sz_u64_t join = sz_utf8_word_break_join_from_class_masks_(aletter_mask, numeric_mask, extendnumlet_mask,
                                                              midletter_mask, midnum_mask, mid_quotes_mask,
                                                              carriage_return_mask, line_feed_mask);
    return (sz_u32_t)((~join) & 0x7FFFFFFCu); // trusted lanes [2,30]
}

/** @brief  Ascending dword-index left-pack permutation for `__lasx_xvperm_w`: row `[submask]` gathers the
 *          `submask`-selected u64 lanes (of 4) to the front in LOW-to-HIGH lane order. Mirrors the Haswell
 *          `sz_utf8_word_compact4_permutation_haswell_` table. */
SZ_HELPER_INLINE __m256i sz_utf8_word_compact4_permutation_lasx_(sz_u32_t submask) {
    static sz_u32_t const compact_lut[16][8] = {
        {0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0}, {2, 3, 0, 0, 0, 0, 0, 0}, {0, 1, 2, 3, 0, 0, 0, 0},
        {4, 5, 0, 0, 0, 0, 0, 0}, {0, 1, 4, 5, 0, 0, 0, 0}, {2, 3, 4, 5, 0, 0, 0, 0}, {0, 1, 2, 3, 4, 5, 0, 0},
        {6, 7, 0, 0, 0, 0, 0, 0}, {0, 1, 6, 7, 0, 0, 0, 0}, {2, 3, 6, 7, 0, 0, 0, 0}, {0, 1, 2, 3, 6, 7, 0, 0},
        {4, 5, 6, 7, 0, 0, 0, 0}, {0, 1, 4, 5, 6, 7, 0, 0}, {2, 3, 4, 5, 6, 7, 0, 0}, {0, 1, 2, 3, 4, 5, 6, 7},
    };
    return __lasx_xvld(compact_lut[submask & 0xFu], 0);
}

/** @brief  Descending-order counterpart of `sz_utf8_word_compact4_permutation_lasx_`: row `[submask]` gathers
 *          the `submask`-selected u64 lanes (of 4) to the front in HIGH-to-LOW lane order, for the reverse
 *          word scan. Mirrors the Haswell `sz_utf8_word_compact4_permutation_descending_haswell_` table. */
SZ_HELPER_INLINE __m256i sz_utf8_word_compact4_permutation_descending_lasx_(sz_u32_t submask) {
    static sz_u32_t const compact_lut[16][8] = {
        {0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0}, {2, 3, 0, 0, 0, 0, 0, 0}, {2, 3, 0, 1, 0, 0, 0, 0},
        {4, 5, 0, 0, 0, 0, 0, 0}, {4, 5, 0, 1, 0, 0, 0, 0}, {4, 5, 2, 3, 0, 0, 0, 0}, {4, 5, 2, 3, 0, 1, 0, 0},
        {6, 7, 0, 0, 0, 0, 0, 0}, {6, 7, 0, 1, 0, 0, 0, 0}, {6, 7, 2, 3, 0, 0, 0, 0}, {6, 7, 2, 3, 0, 1, 0, 0},
        {6, 7, 4, 5, 0, 0, 0, 0}, {6, 7, 4, 5, 0, 1, 0, 0}, {6, 7, 4, 5, 2, 3, 0, 0}, {6, 7, 4, 5, 2, 3, 0, 1},
    };
    return __lasx_xvld(compact_lut[submask & 0xFu], 0);
}

/** @brief  Shift the four u64 lanes of `boundaries` right by one (lanes 1..3 ← 0..2) and insert `carry` into
 *          lane 0 — the LASX form of Haswell's `lane_shift_right` permute + lane-0 blend. */
SZ_HELPER_INLINE __m256i sz_utf8_word_shift_right_insert_lasx_(__m256i boundaries, sz_size_t carry) {
    static sz_u32_t const lane_shift_right[8] = {0, 0, 0, 1, 2, 3, 4, 5};
    __m256i const shifted = __lasx_xvperm_w(boundaries, __lasx_xvld(lane_shift_right, 0));
    return __lasx_xvinsgr2vr_d(shifted, (long long)carry, 0);
}

SZ_API_COMPTIME sz_size_t sz_utf8_wordbreaks_lasx(   //
    sz_cptr_t text, sz_size_t length,                //
    sz_size_t *word_starts, sz_size_t *word_lengths, //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t word_start = 0; // Start of the word currently being accumulated (always a boundary).
    sz_size_t position = sz_utf8_lead_length_(text_u8[0]);

    // Oracle-free fast path: an all-ASCII window [position-2, position+30) resolves boundaries at positions
    // [position, position+28]; one fixed sub-block loop compacts each group and emits it as a shifted-difference,
    // carrying the open `word_start` into lane 0 and the previous boundary into lanes 1..3.
    static sz_u64_t const lane_ramp[4] = {0, 1, 2, 3};
    __m256i const lane_ramp_u64x4 = __lasx_xvld(lane_ramp, 0);
    while (position < length) {
        int ascii_window = position >= 2 && position + 30 <= length;
        __m256i window = __lasx_xvreplgr2vr_b(0);
        if (ascii_window) {
            window = __lasx_xvld(text_u8 + position - 2, 0); // lane j = byte position-2+j
            ascii_window = sz_xvmovemask_b_utf8_lasx_(window) == 0;
        }
        if (!ascii_window) { // Non-ASCII window or near the edges: one scalar codepoint step.
            if (sz_utf8_is_word_boundary_serial(text, length, position)) {
                if (words == words_capacity) {
                    if (bytes_consumed) *bytes_consumed = word_start;
                    return words;
                }
                word_starts[words] = word_start, word_lengths[words] = position - word_start, ++words;
                word_start = position;
            }
            position += sz_utf8_lead_length_(text_u8[position]);
            continue;
        }

        sz_u32_t boundary = sz_utf8_word_break_boundary_mask_lasx_(window); // trusted lanes [2,30]
        for (sz_size_t sub_block = 0; sub_block < 8; ++sub_block) {
            sz_u32_t const submask = (boundary >> (sub_block * 4)) & 0xFu;
            if (!submask) continue;
            sz_size_t const taken = (sz_size_t)sz_u32_popcount(submask);
            sz_size_t const stored = sz_min_of_two(taken, words_capacity - words);

            __m256i const positions = __lasx_xvadd_d(__lasx_xvreplgr2vr_d((long long)(position - 2 + sub_block * 4)),
                                                     lane_ramp_u64x4);
            __m256i const boundaries = __lasx_xvperm_w(positions, sz_utf8_word_compact4_permutation_lasx_(submask));
            __m256i const starts = sz_utf8_word_shift_right_insert_lasx_(boundaries, word_start);
            __m256i const lengths = __lasx_xvsub_d(boundaries, starts);
            sz_utf8_iterate_store_group_lasx_(starts, stored, word_starts + words);
            sz_utf8_iterate_store_group_lasx_(lengths, stored, word_lengths + words);
            words += stored;
            if (stored) word_start = word_starts[words - 1] + word_lengths[words - 1];
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_start;
                return words;
            }
        }
        position += 29; // Resolved [position, position+28]; next unresolved boundary is at position+29.
    }

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
#endif // SZ_USE_LASX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDS_LASX_H_
