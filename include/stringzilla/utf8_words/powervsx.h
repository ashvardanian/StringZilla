/**
 *  @brief POWER VSX backend for UAX-29 word boundaries.
 *  @file include/stringzilla/utf8_words/powervsx.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_WORDS_POWERVSX_H_
#define STRINGZILLA_UTF8_WORDS_POWERVSX_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_words/tables.h"
#include "stringzilla/utf8_words/serial.h"
#include "stringzilla/utf8_runes/powervsx.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

/*  x86-`movemask`-equivalent for VSX gathering each byte's MSB into bit `i` via `vec_vbpermq`; named distinctly
 *  from `find/powervsx.h`'s identical helper so both coexist in one translation unit. */
SZ_HELPER_INLINE sz_u64_t sz_utf8_movemask_powervsx_(__vector unsigned char compared) {
    __vector unsigned char const indices = {120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0};
    __vector unsigned long long const gathered = vec_vbpermq(compared, indices);
#if SZ_IS_BIG_ENDIAN_
    return (sz_u64_t)gathered[0] & 0xFFFFull;
#else
    return (sz_u64_t)gathered[1] & 0xFFFFull;
#endif
}

/*  Boundary mask for the trusted lanes [2,14] of an all-ASCII 16-byte window. Each Word_Break class is a few
 *  VSX range/equality compares; the eight per-class lane bitmasks are extracted with the single-instruction
 *  `vec_vbpermq` movemask and fed to the shared portable join routine - matching the Haswell/v128/LASX path
 *  (extracting to the wide integer ALU measured faster than evaluating the rules in-vector). */
SZ_HELPER_AUTO sz_u32_t sz_utf8_word_break_boundary_mask_powervsx_(__vector unsigned char bytes_vec) {
    __vector unsigned char lowered = vec_or(bytes_vec, vec_splats((unsigned char)0x20));
    __vector unsigned char is_aletter = vec_and(
        (__vector unsigned char)vec_cmpge(lowered, vec_splats((unsigned char)0x61)),
        (__vector unsigned char)vec_cmple(lowered, vec_splats((unsigned char)0x7A)));
    __vector unsigned char is_numeric = vec_and(
        (__vector unsigned char)vec_cmpge(bytes_vec, vec_splats((unsigned char)0x30)),
        (__vector unsigned char)vec_cmple(bytes_vec, vec_splats((unsigned char)0x39)));
    __vector unsigned char is_extendnumlet = (__vector unsigned char)vec_cmpeq(bytes_vec,
                                                                               vec_splats((unsigned char)0x5F));
    __vector unsigned char is_midletter = (__vector unsigned char)vec_cmpeq(bytes_vec, vec_splats((unsigned char)0x3A));
    __vector unsigned char is_midnum = vec_or(
        (__vector unsigned char)vec_cmpeq(bytes_vec, vec_splats((unsigned char)0x2C)),
        (__vector unsigned char)vec_cmpeq(bytes_vec, vec_splats((unsigned char)0x3B)));
    __vector unsigned char is_mid_quotes = vec_or(
        vec_or((__vector unsigned char)vec_cmpeq(bytes_vec, vec_splats((unsigned char)0x22)),
               (__vector unsigned char)vec_cmpeq(bytes_vec, vec_splats((unsigned char)0x27))),
        (__vector unsigned char)vec_cmpeq(bytes_vec, vec_splats((unsigned char)0x2E)));
    __vector unsigned char is_cr = (__vector unsigned char)vec_cmpeq(bytes_vec, vec_splats((unsigned char)0x0D));
    __vector unsigned char is_lf = (__vector unsigned char)vec_cmpeq(bytes_vec, vec_splats((unsigned char)0x0A));
    sz_u64_t join = sz_utf8_word_break_join_from_class_masks_(
        sz_utf8_movemask_powervsx_(is_aletter), sz_utf8_movemask_powervsx_(is_numeric),
        sz_utf8_movemask_powervsx_(is_extendnumlet), sz_utf8_movemask_powervsx_(is_midletter),
        sz_utf8_movemask_powervsx_(is_midnum), sz_utf8_movemask_powervsx_(is_mid_quotes),
        sz_utf8_movemask_powervsx_(is_cr), sz_utf8_movemask_powervsx_(is_lf));
    return (sz_u32_t)((~join) & 0x7FFCu); // trusted lanes [2,14]
}

SZ_API_COMPTIME sz_size_t sz_utf8_words_powervsx(    //
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

    // Byte-permutation rows compacting a 2-lane sub-block's set `u64` boundary positions to the front of a
    // `vector unsigned long long` via `vec_perm` (a VSX vector holds two `u64`): row `[m]` in ascending lane
    // order, indexed by the dense 2-bit submask. Mirrors the newline peel's `compact2_lut`.
    static unsigned char const compact2_lut[4][16] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    };

    // Oracle-free fast path: an all-ASCII window [position-2, position+14) resolves boundaries at positions
    // [position, position+12]; one fixed sub-block loop compacts each 2-lane group and emits it as a
    // shifted-difference, carrying the open `word_start` into lane 0 and the group's first boundary into lane 1.
    while (position < length) {
        int ascii_window = position >= 2 && position + 14 <= length;
        __vector unsigned char window = vec_splats((unsigned char)0);
        if (ascii_window) {
            window = vec_xl(0, text_u8 + position - 2); // lane j = byte position-2+j
            ascii_window = !vec_any_ge(window, vec_splats((unsigned char)0x80));
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

        sz_u32_t boundary = sz_utf8_word_break_boundary_mask_powervsx_(window); // trusted lanes [2,14]

        // Compact every trusted boundary directly into `word_starts + words` (one full-vector store per
        // non-empty 2-lane sub-block), capping at `room` so the masked-store-less 2-lane spill never overshoots
        // capacity; the final sub-block stages through a 2-element tail when its store would cross `room`.
        sz_size_t const room = words_capacity - words;
        sz_size_t filled = 0;
        for (sz_size_t sub_block = 0; sub_block < 8 && filled < room; ++sub_block) {
            sz_u32_t const submask = (boundary >> (sub_block * 2)) & 0x3u;
            if (!submask) continue;

            sz_size_t const base = position - 2 + sub_block * 2; // lane k of this sub-block = byte base+k
            sz_size_t const taken = (sz_size_t)sz_u32_popcount(submask);
            __vector unsigned long long const positions = {(unsigned long long)base, (unsigned long long)(base + 1)};
            __vector unsigned char const permutation = vec_xl(0, compact2_lut[submask]);
            __vector unsigned long long const boundaries = (__vector unsigned long long)vec_perm(
                (__vector unsigned char)positions, (__vector unsigned char)positions, permutation);
            if (filled + 2 <= room) { vec_xst(boundaries, 0, (unsigned long long *)(word_starts + words + filled)); }
            else { // Final sub-block: a 2-lane store would cross `room`, so emit ≤2 valid lanes via a tail buffer.
                sz_size_t tail[2];
                vec_xst(boundaries, 0, (unsigned long long *)tail);
                sz_size_t const copy = sz_min_of_two(taken, room - filled);
                for (sz_size_t k = 0; k < copy; ++k) word_starts[words + filled + k] = tail[k];
            }
            filled += taken;
        }

        // In-place shifted-difference: boundary `i`'s start is boundary `i-1` (or the open `word_start` for
        // `i == 0`). Read boundary `i` before overwriting its slot; carry the previous boundary to dodge the
        // clobbered-slot read.
        sz_size_t const stored = sz_min_of_two(filled, room);
        sz_size_t previous_boundary = word_start;
        for (sz_size_t i = 0; i < stored; ++i) {
            sz_size_t const boundary_i = word_starts[words + i];
            word_starts[words + i] = previous_boundary;
            word_lengths[words + i] = boundary_i - previous_boundary;
            previous_boundary = boundary_i;
        }
        words += stored;
        if (stored) word_start = previous_boundary;
        if (words == words_capacity) {
            if (bytes_consumed) *bytes_consumed = word_start;
            return words;
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
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_POWERVSX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDS_POWERVSX_H_
