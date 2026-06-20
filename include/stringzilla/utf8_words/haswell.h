/**
 *  @brief Haswell (AVX2) backend for UAX-29 word boundaries.
 *  @file include/stringzilla/utf8_words/haswell.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_WORDS_HASWELL_H_
#define STRINGZILLA_UTF8_WORDS_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_words/tables.h"
#include "stringzilla/utf8_words/serial.h"
#include "stringzilla/utf8_codepoints/haswell.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2,bmi,bmi2,popcnt")
#endif

/* Per-byte Word_Break property of ASCII codepoints, laid out as eight 16-entry rows indexed by the
 * low nibble; row R serves high nibble R (covering bytes 0x00-0x7F). */
SZ_INTERNAL __m256i sz_utf8_word_break_classify_ascii_haswell_(__m256i bytes) {
    // Eight rows of the ASCII Word_Break property table (high nibble → 16 low-nibble entries).
    __m256i row0 = _mm256_setr_epi8(                    //
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 1, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 1, 0, 0);
    __m256i row1 = _mm256_setzero_si256();
    __m256i row2 = _mm256_setr_epi8(                        //
        0, 0, 15, 0, 0, 0, 0, 15, 0, 0, 0, 0, 14, 0, 15, 0, //
        0, 0, 15, 0, 0, 0, 0, 15, 0, 0, 0, 0, 14, 0, 15, 0);
    __m256i row3 = _mm256_setr_epi8(                                //
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 13, 14, 0, 0, 0, 0, //
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 13, 14, 0, 0, 0, 0);
    __m256i row4 = _mm256_setr_epi8(                    //
        0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, //
        0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8);
    __m256i row5 = _mm256_setr_epi8(                     //
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 12, //
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 12);
    __m256i row6 = _mm256_setr_epi8(                    //
        0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, //
        0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8);
    __m256i row7 = _mm256_setr_epi8(                    //
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, //
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0);

    __m256i lo_nibble = _mm256_and_si256(bytes, _mm256_set1_epi8(0x0F));
    __m256i hi_nibble = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), _mm256_set1_epi8(0x0F));

    __m256i result = _mm256_setzero_si256();
    __m256i shuf0 = _mm256_shuffle_epi8(row0, lo_nibble);
    __m256i shuf1 = _mm256_shuffle_epi8(row1, lo_nibble);
    __m256i shuf2 = _mm256_shuffle_epi8(row2, lo_nibble);
    __m256i shuf3 = _mm256_shuffle_epi8(row3, lo_nibble);
    __m256i shuf4 = _mm256_shuffle_epi8(row4, lo_nibble);
    __m256i shuf5 = _mm256_shuffle_epi8(row5, lo_nibble);
    __m256i shuf6 = _mm256_shuffle_epi8(row6, lo_nibble);
    __m256i shuf7 = _mm256_shuffle_epi8(row7, lo_nibble);

    result = _mm256_blendv_epi8(result, shuf0, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(0)));
    result = _mm256_blendv_epi8(result, shuf1, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(1)));
    result = _mm256_blendv_epi8(result, shuf2, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(2)));
    result = _mm256_blendv_epi8(result, shuf3, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(3)));
    result = _mm256_blendv_epi8(result, shuf4, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(4)));
    result = _mm256_blendv_epi8(result, shuf5, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(5)));
    result = _mm256_blendv_epi8(result, shuf6, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(6)));
    result = _mm256_blendv_epi8(result, shuf7, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(7)));
    return result;
}

/* Per-class lane mask: bit i set where lane i has Word_Break property `value`. */
SZ_INTERNAL sz_u32_t sz_utf8_word_break_class_mask_haswell_(__m256i classes, int value) {
    return (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(classes, _mm256_set1_epi8((char)value)));
}

/* 32-bit "joined" (guaranteed non-boundary) mask for an all-ASCII 32-byte window: bit i set => the boundary
 * before lane i is suppressed by a UAX-29 no-break rule. Exact for lanes whose i-2 and i+1 neighbours are
 * in-window; the caller restricts the result to its trusted lane window. */
SZ_INTERNAL sz_u32_t sz_utf8_word_break_join_mask_ascii_haswell_(__m256i classes) {
    // Reduce each class to a 32-lane bitmask and defer the rule logic to the shared portable routine; the
    // caller restricts the result to its trusted lane window, so the wider u64 math is harmless.
    return (sz_u32_t)sz_utf8_word_break_join_from_class_masks_(
        sz_utf8_word_break_class_mask_haswell_(classes, sz_utf8_word_break_aletter_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_utf8_word_break_numeric_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_utf8_word_break_extendnumlet_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_utf8_word_break_midletter_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_utf8_word_break_midnum_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_utf8_word_break_mid_quotes_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_utf8_word_break_cr_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_utf8_word_break_lf_k));
}

/* Per-file copy of the sorting backend's left-pack table for compacting u64 lanes: row `[m]` holds the 8 dword
 * indices that gather the `m`-selected u64 lanes (of 4, each a dword pair) to the front for
 * `_mm256_permutevar8x32_epi32`. Identical to the table in `sz_utf8_iterate_peel_haswell_`. */
SZ_INTERNAL __m256i sz_utf8_word_compact4_permutation_haswell_(sz_u32_t submask) {
    static sz_u32_t const compact_lut[16][8] = {
        {0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0}, {2, 3, 0, 0, 0, 0, 0, 0}, {0, 1, 2, 3, 0, 0, 0, 0},
        {4, 5, 0, 0, 0, 0, 0, 0}, {0, 1, 4, 5, 0, 0, 0, 0}, {2, 3, 4, 5, 0, 0, 0, 0}, {0, 1, 2, 3, 4, 5, 0, 0},
        {6, 7, 0, 0, 0, 0, 0, 0}, {0, 1, 6, 7, 0, 0, 0, 0}, {2, 3, 6, 7, 0, 0, 0, 0}, {0, 1, 2, 3, 6, 7, 0, 0},
        {4, 5, 6, 7, 0, 0, 0, 0}, {0, 1, 4, 5, 6, 7, 0, 0}, {2, 3, 4, 5, 6, 7, 0, 0}, {0, 1, 2, 3, 4, 5, 6, 7},
    };
    return _mm256_loadu_si256((__m256i const *)compact_lut[submask & 0xFu]);
}

/** @brief  Descending-order counterpart of `sz_utf8_word_compact4_permutation_haswell_`: row `[m]` gathers the
 *          `m`-selected u64 lanes (of 4) to the front in HIGH-to-LOW lane order, for the reverse word scan. */
SZ_INTERNAL __m256i sz_utf8_word_compact4_permutation_descending_haswell_(sz_u32_t submask) {
    static sz_u32_t const compact_lut[16][8] = {
        {0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0}, {2, 3, 0, 0, 0, 0, 0, 0}, {2, 3, 0, 1, 0, 0, 0, 0},
        {4, 5, 0, 0, 0, 0, 0, 0}, {4, 5, 0, 1, 0, 0, 0, 0}, {4, 5, 2, 3, 0, 0, 0, 0}, {4, 5, 2, 3, 0, 1, 0, 0},
        {6, 7, 0, 0, 0, 0, 0, 0}, {6, 7, 0, 1, 0, 0, 0, 0}, {6, 7, 2, 3, 0, 0, 0, 0}, {6, 7, 2, 3, 0, 1, 0, 0},
        {6, 7, 4, 5, 0, 0, 0, 0}, {6, 7, 4, 5, 0, 1, 0, 0}, {6, 7, 4, 5, 2, 3, 0, 0}, {6, 7, 4, 5, 2, 3, 0, 1},
    };
    return _mm256_loadu_si256((__m256i const *)compact_lut[submask & 0xFu]);
}

SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_haswell( //
    sz_cptr_t text, sz_size_t length,                     //
    sz_size_t *word_starts, sz_size_t *word_lengths,      //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t word_start = 0; // Start of the word currently being accumulated (always a boundary).
    // Skip first codepoint (position 0 is always a boundary), matching the serial reference.
    sz_size_t position = sz_utf8_codepoint_length_(text_u8[0]);

    // Oracle-free fast path: an all-ASCII window [position-2, position+30) resolves boundaries at positions
    // [position, position+28]; one fixed sub-block loop compacts each group and emits it as a shifted-difference,
    // carrying the open `word_start` into lane 0 and the previous boundary into lanes 1..3.
    __m256i const lane_ramp = _mm256_setr_epi64x(0, 1, 2, 3);
    __m256i const lane_shift_right = _mm256_setr_epi32(0, 0, 0, 1, 2, 3, 4, 5);
    while (position < length) {
        int ascii_window = position >= 2 && position + 30 <= length;
        __m256i window = _mm256_setzero_si256();
        if (ascii_window) {
            window = _mm256_loadu_si256((__m256i const *)(text_u8 + position - 2)); // lane j = byte position-2+j
            ascii_window = _mm256_movemask_epi8(window) == 0;
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
            position += sz_utf8_codepoint_length_(text_u8[position]);
            continue;
        }

        sz_u32_t boundary =
            (~sz_utf8_word_break_join_mask_ascii_haswell_(sz_utf8_word_break_classify_ascii_haswell_(window))) &
            0x7FFFFFFCu; // trusted lanes [2,30]
        for (sz_size_t sub_block = 0; sub_block < 8; ++sub_block) {
            sz_u32_t const submask = (boundary >> (sub_block * 4)) & 0xFu;
            if (!submask) continue;
            sz_size_t const taken = (sz_size_t)_mm_popcnt_u32(submask);
            sz_size_t const stored = sz_min_of_two(taken, words_capacity - words);

            __m256i const positions = _mm256_add_epi64(_mm256_set1_epi64x((long long)(position - 2 + sub_block * 4)),
                                                       lane_ramp);
            __m256i const boundaries = _mm256_permutevar8x32_epi32(positions,
                                                                   sz_utf8_word_compact4_permutation_haswell_(submask));
            __m256i const starts = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(boundaries, lane_shift_right),
                                                      _mm256_set1_epi64x((long long)word_start), 0x03);
            __m256i const store_mask = sz_mm256_store_mask_epi64_(stored);
            _mm256_maskstore_epi64((long long *)(word_starts + words), store_mask, starts);
            _mm256_maskstore_epi64((long long *)(word_lengths + words), store_mask,
                                   _mm256_sub_epi64(boundaries, starts));
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

SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_haswell( //
    sz_cptr_t text, sz_size_t length,                      //
    sz_size_t *word_starts, sz_size_t *word_lengths,       //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = length;
        return 0;
    }

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t word_end = length; // End of the word currently being accumulated (always a boundary).
    sz_size_t position = length - 1;
    while (position > 0 && (text_u8[position] & 0xC0) == 0x80) position--;

    // Oracle-free fast path: an all-ASCII window [position-30, position+2) resolves boundaries at positions
    // [position-28, position]; one fixed sub-block loop walks high-to-low, compacting each group in descending
    // lane order and emitting it as a shifted-difference (lane 0 carries the open `word_end`).
    __m256i const lane_ramp = _mm256_setr_epi64x(0, 1, 2, 3);
    __m256i const lane_shift_right = _mm256_setr_epi32(0, 0, 0, 1, 2, 3, 4, 5);
    while (position > 0) {
        sz_size_t base = position - 30; // lane j = byte base+j; trusted lanes [2,30] → [position-28, position]
        int ascii_window = position >= 30 && position + 2 <= length;
        __m256i window = _mm256_setzero_si256();
        if (ascii_window) {
            window = _mm256_loadu_si256((__m256i const *)(text_u8 + base));
            ascii_window = _mm256_movemask_epi8(window) == 0;
        }
        if (!ascii_window) { // Non-ASCII window or near the edges: one scalar codepoint step.
            if (sz_utf8_is_word_boundary_serial(text, length, position)) {
                if (words == words_capacity) {
                    if (bytes_consumed) *bytes_consumed = word_end;
                    return words;
                }
                word_starts[words] = position, word_lengths[words] = word_end - position, ++words;
                word_end = position;
            }
            position--;
            while (position > 0 && (text_u8[position] & 0xC0) == 0x80) position--;
            continue;
        }

        sz_u32_t boundary =
            (~sz_utf8_word_break_join_mask_ascii_haswell_(sz_utf8_word_break_classify_ascii_haswell_(window))) &
            0x7FFFFFFCu;                                  // trusted lanes [2,30]
        for (sz_size_t sub_block = 8; sub_block-- > 0;) { // high-to-low for descending emission
            sz_u32_t const submask = (boundary >> (sub_block * 4)) & 0xFu;
            if (!submask) continue;
            sz_size_t const taken = (sz_size_t)_mm_popcnt_u32(submask);
            sz_size_t const stored = sz_min_of_two(taken, words_capacity - words);

            __m256i const positions = _mm256_add_epi64(_mm256_set1_epi64x((long long)(base + sub_block * 4)),
                                                       lane_ramp);
            __m256i const boundaries = _mm256_permutevar8x32_epi32(
                positions, sz_utf8_word_compact4_permutation_descending_haswell_(submask));
            __m256i const previous = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(boundaries, lane_shift_right),
                                                        _mm256_set1_epi64x((long long)word_end), 0x03);
            __m256i const store_mask = sz_mm256_store_mask_epi64_(stored);
            _mm256_maskstore_epi64((long long *)(word_starts + words), store_mask, boundaries);
            _mm256_maskstore_epi64((long long *)(word_lengths + words), store_mask,
                                   _mm256_sub_epi64(previous, boundaries));
            words += stored;
            if (stored) word_end = word_starts[words - 1];
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_end;
                return words;
            }
        }
        position = base + 1; // Resolved down to position-28; next unresolved boundary is at position-29.
    }

    if (words == words_capacity) {
        if (bytes_consumed) *bytes_consumed = word_end;
        return words;
    }
    word_starts[words] = 0;
    word_lengths[words] = word_end;
    ++words;
    if (bytes_consumed) *bytes_consumed = 0;
    return words;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDS_HASWELL_H_
