/**
 *  @brief WebAssembly SIMD128 backend for UAX-29 word boundaries.
 *  @file include/stringzilla/utf8_words/v128.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_WORDS_V128_H_
#define STRINGZILLA_UTF8_WORDS_V128_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_words/tables.h"
#include "stringzilla/utf8_words/serial.h"
#include "stringzilla/utf8_runes/v128.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

/*  Per-class 16-bit lane mask for an all-ASCII window via `wasm_i8x16_bitmask`. */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_class_bitmask_v128_(v128_t equal_vec) {
    return (sz_u64_t)((sz_u32_t)wasm_i8x16_bitmask(equal_vec) & 0xFFFFu);
}

/*  Boundary mask for the trusted lanes [2,14] of an all-ASCII 16-byte window: bit i set => a word boundary
 *  precedes lane i. Computed branchlessly via the shared portable join routine. */
SZ_INTERNAL sz_u32_t sz_utf8_word_break_boundary_mask_v128_(v128_t window) {
    v128_t lowered = wasm_v128_or(window, wasm_i8x16_splat(0x20)); // fold A-Z onto a-z
    sz_u64_t aletter_mask = sz_utf8_word_break_class_bitmask_v128_(
        wasm_v128_and(wasm_u8x16_ge(lowered, wasm_i8x16_splat(0x61)), wasm_u8x16_le(lowered, wasm_i8x16_splat(0x7A))));
    sz_u64_t numeric_mask = sz_utf8_word_break_class_bitmask_v128_(
        wasm_v128_and(wasm_u8x16_ge(window, wasm_i8x16_splat(0x30)), wasm_u8x16_le(window, wasm_i8x16_splat(0x39))));
    sz_u64_t extendnumlet_mask = sz_utf8_word_break_class_bitmask_v128_(wasm_i8x16_eq(window, wasm_i8x16_splat(0x5F)));
    sz_u64_t midletter_mask = sz_utf8_word_break_class_bitmask_v128_(wasm_i8x16_eq(window, wasm_i8x16_splat(0x3A)));
    sz_u64_t midnum_mask = sz_utf8_word_break_class_bitmask_v128_(
        wasm_v128_or(wasm_i8x16_eq(window, wasm_i8x16_splat(0x2C)), wasm_i8x16_eq(window, wasm_i8x16_splat(0x3B))));
    sz_u64_t mid_quotes_mask = sz_utf8_word_break_class_bitmask_v128_(wasm_v128_or(
        wasm_v128_or(wasm_i8x16_eq(window, wasm_i8x16_splat(0x22)), wasm_i8x16_eq(window, wasm_i8x16_splat(0x27))),
        wasm_i8x16_eq(window, wasm_i8x16_splat(0x2E))));
    sz_u64_t carriage_return_mask = sz_utf8_word_break_class_bitmask_v128_(
        wasm_i8x16_eq(window, wasm_i8x16_splat(0x0D)));
    sz_u64_t line_feed_mask = sz_utf8_word_break_class_bitmask_v128_(wasm_i8x16_eq(window, wasm_i8x16_splat(0x0A)));
    sz_u64_t join = sz_utf8_word_break_join_from_class_masks_(aletter_mask, numeric_mask, extendnumlet_mask,
                                                              midletter_mask, midnum_mask, mid_quotes_mask,
                                                              carriage_return_mask, line_feed_mask);
    return (sz_u32_t)((~join) & 0x7FFCu); // trusted lanes [2,14]
}

#pragma region Word boundary left pack

/** @brief  Ascending `wasm_i8x16_swizzle` permutation that gathers a 4-bit sub-block's set u32 lanes to the front,
 *          preserving low-to-high lane order. The v128 analog of `sz_utf8_word_compact4_permutation_haswell_`. */
SZ_INTERNAL v128_t sz_utf8_word_compact4_permutation_v128_(sz_u32_t submask) {
    static sz_u8_t const compact_lut[16][16] = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},       {0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},       {0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0},
        {8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},     {0, 1, 2, 3, 8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0},
        {4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0},     {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0},
        {12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},   {0, 1, 2, 3, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0},
        {4, 5, 6, 7, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0},   {0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 0, 0, 0, 0},
        {8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0},
        {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    };
    return wasm_v128_load(compact_lut[submask & 0xFu]);
}

/** @brief  Descending counterpart of `sz_utf8_word_compact4_permutation_v128_`: gathers a 4-bit sub-block's set
 *          u32 lanes to the front in HIGH-to-LOW lane order, for the reverse word scan. */
SZ_INTERNAL v128_t sz_utf8_word_compact4_permutation_descending_v128_(sz_u32_t submask) {
    static sz_u8_t const compact_lut[16][16] = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},       {0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},       {4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0},
        {8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},     {8, 9, 10, 11, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0},
        {8, 9, 10, 11, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0},     {8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0},
        {12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},   {12, 13, 14, 15, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0},
        {12, 13, 14, 15, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0},   {12, 13, 14, 15, 4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0},
        {12, 13, 14, 15, 8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0}, {12, 13, 14, 15, 8, 9, 10, 11, 0, 1, 2, 3, 0, 0, 0, 0},
        {12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 0, 0, 0}, {12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3},
    };
    return wasm_v128_load(compact_lut[submask & 0xFu]);
}

/** @brief  Shift four packed u32 boundary lanes right by one and seat @p carry in the freed lane 0, building
 *          `[carry, b0, b1, b2]` from `boundaries = [b0, b1, b2, b3]` with one `wasm_i8x16_shuffle`. */
SZ_INTERNAL v128_t sz_utf8_word_shift_right_carry_v128_(v128_t boundaries, v128_t carry) {
    return wasm_i8x16_shuffle(carry, boundaries, 0, 1, 2, 3, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27);
}

#pragma endregion // Word boundary left pack

SZ_PUBLIC sz_size_t sz_utf8_words_v128(              //
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
    // Skip the first codepoint (position 0 is always a boundary, WB1).
    sz_size_t position = sz_utf8_lead_length_(text_u8[0]);

    // Oracle-free fast path: an all-ASCII window [position-2, position+14) resolves boundaries at positions
    // [position, position+12]; one fixed sub-block loop compacts each group and emits it as a shifted-difference,
    // carrying the open `word_start` into lane 0 and the previous boundary into lanes 1..3.
    while (position < length) {
        int ascii_window = position >= 2 && position + 14 <= length;
        v128_t window = wasm_i8x16_splat(0);
        if (ascii_window) {
            window = wasm_v128_load(text_u8 + position - 2); // lane j = byte position-2+j
            ascii_window = ((sz_u32_t)wasm_i8x16_bitmask(window) & 0xFFFFu) == 0;
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

        sz_u32_t boundary = sz_utf8_word_break_boundary_mask_v128_(window); // trusted lanes [2,14]
        // Each sub-block compacts its set boundaries (at most four) and emits the shifted-difference straight into
        // `word_starts + words`: `word_start` itself is the carry seeded into lane 0, then re-read from the stored
        // tail so a capacity cut lands on the last emit. WASM has no masked store, so a 4-lane vector store is used
        // only when the whole vector fits; otherwise a ≤4-element tail buffer is copied out without overshoot.
        for (sz_size_t sub_block = 0; sub_block < 4; ++sub_block) {
            sz_u32_t const submask = (boundary >> (sub_block * 4)) & 0xFu;
            if (!submask) continue;
            sz_size_t const taken = (sz_size_t)sz_u32_popcount(submask);
            sz_size_t const stored = sz_min_of_two(taken, words_capacity - words);

            v128_t const positions = wasm_u32x4_make(
                (sz_u32_t)(position - 2 + sub_block * 4 + 0), (sz_u32_t)(position - 2 + sub_block * 4 + 1),
                (sz_u32_t)(position - 2 + sub_block * 4 + 2), (sz_u32_t)(position - 2 + sub_block * 4 + 3));
            v128_t const boundaries = wasm_i8x16_swizzle(positions, sz_utf8_word_compact4_permutation_v128_(submask));
            v128_t const starts = sz_utf8_word_shift_right_carry_v128_(boundaries,
                                                                       wasm_u32x4_splat((sz_u32_t)word_start));
            v128_t const lengths = wasm_i32x4_sub(boundaries, starts);
            if (words + 4 <= words_capacity) {
                wasm_v128_store(word_starts + words, starts);
                wasm_v128_store(word_lengths + words, lengths);
            }
            else {
                sz_size_t tail_starts[4], tail_lengths[4];
                wasm_v128_store(tail_starts, starts);
                wasm_v128_store(tail_lengths, lengths);
                for (sz_size_t i = 0; i < stored; ++i)
                    word_starts[words + i] = tail_starts[i], word_lengths[words + i] = tail_lengths[i];
            }
            words += stored;
            if (stored) word_start = word_starts[words - 1] + word_lengths[words - 1]; // last boundary carries on
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_start;
                return words;
            }
        }
        position += 13; // Resolved [position, position+12]; next unresolved boundary is at position+13.
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

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDS_V128_H_
