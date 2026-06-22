/**
 *  @brief WebAssembly SIMD128 uncased UTF-8 search, comparison & invariance backend.
 *  @file include/stringzilla/utf8_uncased/v128.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_V128_H_
#define STRINGZILLA_UTF8_UNCASED_V128_H_

#include "stringzilla/find/v128.h"           // `sz_find_v128`
#include "stringzilla/utf8_uncased_fold/v128.h" //
#include "stringzilla/utf8_uncased/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

/*  The uncased substring search mirrors the NEON/RVV port: a scripted driver walks the haystack
 *  in 32-byte chunks; each per-script kernel supplies a `fold` callback (case-folds a length-preserving
 *  chunk in place, one folded byte per source byte) and an `alarm` callback (flags length-CHANGING folds
 *  like ß→ss, ligatures, Kelvin). Clean chunks are folded and probe-filtered (four needle bytes broadcast
 *  and compared at their offsets, AND-ed, walked low-to-high); each survivor re-checks its folded window
 *  and defers to the value-exact serial `sz_utf8_uncased_verify_match_`. Alarmed chunks and the
 *  sub-window tail go to the serial danger-zone scanner. The fold math is the length-preserving subset of
 *  `utf8_uncased_fold/v128.h`; the cross-chunk predecessor/successor are read straight from the padded chunk
 *  buffer so every byte folds with the same neighbours the whole-chunk vector would have seen. Matches the
 *  Ice Lake / NEON / RVV SIMD backends exactly (which may differ from the serial folded-rune scanner on a
 *  few expanding-needle ties; that is a pre-existing cross-backend property, not wasm-specific). */

#pragma region Helpers

/** @brief `result[0] = carry`, `result[i] = vector[i-1]` — slide-up carrying a real predecessor across windows. */
SZ_INTERNAL v128_t sz_utf8_uncased_slide1up_v128_(v128_t vector, sz_u8_t carry) {
    return wasm_i8x16_shuffle(vector, wasm_i8x16_splat((sz_i8_t)carry), 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                              13, 14);
}

/** @brief 0/1 byte (not 0xFF): 1 where `bytes == value`. */
SZ_INTERNAL v128_t sz_utf8_uncased_eq01_v128_(v128_t bytes, sz_u8_t value) {
    return wasm_v128_and(wasm_i8x16_eq(bytes, wasm_i8x16_splat((sz_i8_t)value)), wasm_i8x16_splat(1));
}
/** @brief 0/1 byte: 1 where `bytes` in `[start, start+length)`. */
SZ_INTERNAL v128_t sz_utf8_uncased_inrange01_v128_(v128_t bytes, sz_u8_t start, sz_u8_t length) {
    return wasm_v128_and(sz_utf8_in_range_v128_(bytes, start, length), wasm_i8x16_splat(1));
}

/** @brief Zero a scratch buffer then copy `length` bytes, so strip kernels can read one byte past. */
SZ_INTERNAL sz_u8_t const *sz_utf8_uncased_load_padded_v128_(sz_cptr_t source, sz_size_t length, sz_u8_t *buffer,
                                                        sz_size_t buffer_capacity) {
    for (sz_size_t byte_index = 0; byte_index < buffer_capacity; ++byte_index) buffer[byte_index] = 0;
    for (sz_size_t byte_index = 0; byte_index < length; ++byte_index) buffer[byte_index] = (sz_u8_t)source[byte_index];
    return buffer;
}

/** @brief Gather the C4/C5/C6 +1 parity delta for the continuation byte's low 6 bits (irregular flag kept). */
SZ_INTERNAL v128_t sz_utf8_uncased_latin_delta_v128_(v128_t source, v128_t after_c4, v128_t after_c5, v128_t after_c6,
                                                v128_t is_continuation) {
    v128_t low6 = wasm_v128_and(source, wasm_i8x16_splat(0x3F));
    v128_t c4 = sz_utf8_gather64_v128_(wasm_v128_load(&sz_utf8_fold_latin_c4_deltas_v128_[0]),
                                       wasm_v128_load(&sz_utf8_fold_latin_c4_deltas_v128_[16]),
                                       wasm_v128_load(&sz_utf8_fold_latin_c4_deltas_v128_[32]),
                                       wasm_v128_load(&sz_utf8_fold_latin_c4_deltas_v128_[48]), low6);
    v128_t c5 = sz_utf8_gather64_v128_(wasm_v128_load(&sz_utf8_fold_latin_c5_deltas_v128_[0]),
                                       wasm_v128_load(&sz_utf8_fold_latin_c5_deltas_v128_[16]),
                                       wasm_v128_load(&sz_utf8_fold_latin_c5_deltas_v128_[32]),
                                       wasm_v128_load(&sz_utf8_fold_latin_c5_deltas_v128_[48]), low6);
    v128_t c6 = sz_utf8_gather64_v128_(wasm_v128_load(&sz_utf8_fold_latin_c6_deltas_v128_[0]),
                                       wasm_v128_load(&sz_utf8_fold_latin_c6_deltas_v128_[16]),
                                       wasm_v128_load(&sz_utf8_fold_latin_c6_deltas_v128_[32]),
                                       wasm_v128_load(&sz_utf8_fold_latin_c6_deltas_v128_[48]), low6);
    v128_t delta = wasm_i8x16_splat(0);
    delta = wasm_v128_bitselect(c4, delta, after_c4);
    delta = wasm_v128_bitselect(c5, delta, after_c5);
    delta = wasm_v128_bitselect(c6, delta, after_c6);
    return wasm_v128_and(delta, is_continuation);
}

/** @brief A 16-byte fold window: the chunk at `src + pos` plus its cross-boundary neighbours. */
typedef struct {
    v128_t source;   // The 16 bytes at `src + pos`.
    v128_t previous; // `source` slid up one lane, carrying the real predecessor byte (0 at `pos == 0`).
    v128_t next;     // `source` slid down one lane, carrying the real successor byte.
} sz_utf8_uncased_window_v128_t_;

/** @brief Loads the fold window at `src + pos`, reading one byte on each side for cross-window folds. */
SZ_INTERNAL sz_utf8_uncased_window_v128_t_ sz_utf8_uncased_load_window_v128_(sz_u8_t const *src, sz_size_t pos) {
    sz_utf8_uncased_window_v128_t_ window;
    window.source = wasm_v128_load(src + pos);
    window.previous = sz_utf8_uncased_slide1up_v128_(window.source, pos > 0 ? src[pos - 1] : 0);
    window.next = sz_utf8_slide1down_v128_(window.source, src[pos + 16]);
    return window;
}

#pragma endregion // Helpers

#pragma region Per script fold strips

SZ_INTERNAL void sz_utf8_uncased_fold_ascii_strip_v128_(sz_u8_t const *src, sz_size_t vector_length, sz_u8_t *dst) {
    for (sz_size_t pos = 0; pos < vector_length; pos += 16) {
        sz_size_t window = vector_length - pos < 16 ? vector_length - pos : 16;
        sz_store_partial_v128_((sz_ptr_t)(dst + pos), sz_ascii_fold_v128_(wasm_v128_load(src + pos)), window);
    }
}

SZ_INTERNAL void sz_utf8_uncased_fold_western_europe_strip_v128_(sz_u8_t const *src, sz_size_t vector_length, sz_u8_t *dst) {
    for (sz_size_t pos = 0; pos < vector_length; pos += 16) {
        sz_size_t window = vector_length - pos < 16 ? vector_length - pos : 16;
        sz_utf8_uncased_window_v128_t_ chunk = sz_utf8_uncased_load_window_v128_(src, pos);
        v128_t source = chunk.source, previous = chunk.previous, next = chunk.next;
        v128_t after_c3 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xC3));
        v128_t folded = sz_ascii_fold_v128_(source);
        v128_t latin1_range = sz_utf8_in_range_v128_(source, 0x80, 0x1F);
        v128_t is_latin1_upper = wasm_v128_andnot(wasm_v128_and(after_c3, latin1_range),
                                                  wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0x97)));
        folded = sz_utf8_masked_add_v128_(folded, is_latin1_upper, 0x20);
        v128_t eszett_lead = wasm_v128_and(wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xC3)),
                                           wasm_i8x16_eq(next, wasm_i8x16_splat((sz_i8_t)0x9F)));
        v128_t eszett_second = wasm_v128_and(after_c3, wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0x9F)));
        folded = wasm_v128_bitselect(wasm_i8x16_splat('s'), folded, wasm_v128_or(eszett_lead, eszett_second));
        sz_store_partial_v128_((sz_ptr_t)(dst + pos), folded, window);
    }
}

SZ_INTERNAL void sz_utf8_uncased_fold_central_europe_strip_v128_(sz_u8_t const *src, sz_size_t vector_length, sz_u8_t *dst) {
    for (sz_size_t pos = 0; pos < vector_length; pos += 16) {
        sz_size_t window = vector_length - pos < 16 ? vector_length - pos : 16;
        sz_utf8_uncased_window_v128_t_ chunk = sz_utf8_uncased_load_window_v128_(src, pos);
        v128_t source = chunk.source, previous = chunk.previous, next = chunk.next;
        (void)next;
        v128_t is_continuation = wasm_i8x16_eq(wasm_v128_and(source, wasm_i8x16_splat((sz_i8_t)0xC0)),
                                               wasm_i8x16_splat((sz_i8_t)0x80));
        v128_t after_c3 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xC3));
        v128_t after_c4 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xC4));
        v128_t after_c5 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xC5));
        v128_t folded = sz_ascii_fold_v128_(source);
        v128_t latin1_range = sz_utf8_in_range_v128_(source, 0x80, 0x1F);
        v128_t is_latin1_upper = wasm_v128_andnot(wasm_v128_and(after_c3, latin1_range),
                                                  wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0x97)));
        folded = sz_utf8_masked_add_v128_(folded, is_latin1_upper, 0x20);
        v128_t delta = sz_utf8_uncased_latin_delta_v128_(source, after_c4, after_c5, wasm_i8x16_splat(0), is_continuation);
        folded = wasm_i8x16_add(folded, wasm_v128_and(delta, wasm_i8x16_splat(0x01)));
        sz_store_partial_v128_((sz_ptr_t)(dst + pos), folded, window);
    }
}

SZ_INTERNAL void sz_utf8_uncased_fold_cyrillic_strip_v128_(sz_u8_t const *src, sz_size_t vector_length, sz_u8_t *dst) {
    static sz_align_(16) sz_u8_t const offsets[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x20, 0xE0, 0, 0, 0, 0, 0};
    v128_t offsets_lut = wasm_v128_load(offsets);
    for (sz_size_t pos = 0; pos < vector_length; pos += 16) {
        sz_size_t window = vector_length - pos < 16 ? vector_length - pos : 16;
        sz_utf8_uncased_window_v128_t_ chunk = sz_utf8_uncased_load_window_v128_(src, pos);
        v128_t source = chunk.source, previous = chunk.previous, next = chunk.next;
        v128_t after_d0 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xD0));
        v128_t is_d0 = wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xD0));
        v128_t folded = sz_ascii_fold_v128_(source);
        v128_t offset = wasm_v128_and(wasm_i8x16_swizzle(offsets_lut, wasm_u8x16_shr(source, 4)), after_d0);
        folded = wasm_i8x16_add(folded, offset);
        v128_t needs_d1 = wasm_v128_and(
            is_d0, wasm_v128_or(sz_utf8_in_range_v128_(next, 0x80, 0x10), sz_utf8_in_range_v128_(next, 0xA0, 0x10)));
        folded = sz_utf8_masked_add_v128_(folded, needs_d1, 1);
        sz_store_partial_v128_((sz_ptr_t)(dst + pos), folded, window);
    }
}

SZ_INTERNAL void sz_utf8_uncased_fold_greek_strip_v128_(sz_u8_t const *src, sz_size_t vector_length, sz_u8_t *dst) {
    // Greek monotonic second-byte deltas after a CE lead, indexed by `text & 0x3F`.
    static sz_align_(16) sz_u8_t const ce_deltas[64] = {
        0,    0,    0,    0,    0,    0,    0x26, 0,    0x25, 0x25, 0x25, 0,    0,    0,    0xFF, 0xFF, // CE 80-8F
        0,    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, // CE 90-9F
        0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0,    0,    0,    0,    // CE A0-AF
        0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0};   // CE B0-BF
    // CE→CF (+1) lead-promotion flags for second-byte classes whose lowercase lands in the CF block.
    static sz_align_(16) sz_u8_t const ce_promotes[64] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, // CE 80-8F: 8C, 8E, 8F promote
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // CE 90-9F stay
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, // CE A0-AB promote
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (sz_size_t pos = 0; pos < vector_length; pos += 16) {
        sz_size_t window = vector_length - pos < 16 ? vector_length - pos : 16;
        sz_utf8_uncased_window_v128_t_ chunk = sz_utf8_uncased_load_window_v128_(src, pos);
        v128_t source = chunk.source, previous = chunk.previous, next = chunk.next;
        v128_t is_continuation = wasm_i8x16_eq(wasm_v128_and(source, wasm_i8x16_splat((sz_i8_t)0xC0)),
                                               wasm_i8x16_splat((sz_i8_t)0x80));
        v128_t after_ce = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xCE));
        v128_t after_cf = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xCF));
        v128_t after_c2 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xC2));
        v128_t is_ce_lead = wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xCE));
        v128_t after_ce_cont = wasm_v128_and(after_ce, is_continuation);
        v128_t folded = sz_ascii_fold_v128_(source);

        v128_t deltas_lo = wasm_v128_load(&ce_deltas[0]);
        v128_t deltas_1 = wasm_v128_load(&ce_deltas[16]);
        v128_t deltas_2 = wasm_v128_load(&ce_deltas[32]);
        v128_t deltas_3 = wasm_v128_load(&ce_deltas[48]);
        v128_t promo_lo = wasm_v128_load(&ce_promotes[0]);
        v128_t promo_1 = wasm_v128_load(&ce_promotes[16]);
        v128_t promo_2 = wasm_v128_load(&ce_promotes[32]);
        v128_t promo_3 = wasm_v128_load(&ce_promotes[48]);
        v128_t ce_delta = wasm_v128_and(sz_utf8_gather64_v128_(deltas_lo, deltas_1, deltas_2, deltas_3,
                                                               wasm_v128_and(source, wasm_i8x16_splat(0x3F))),
                                        after_ce_cont);
        folded = wasm_i8x16_add(folded, ce_delta);

        v128_t final_sigma = wasm_v128_and(after_cf, wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0x82)));
        folded = sz_utf8_masked_add_v128_(folded, final_sigma, 0x01);
        v128_t micro_second = wasm_v128_and(after_c2, wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xB5)));
        folded = wasm_v128_bitselect(wasm_i8x16_splat((sz_i8_t)0xBC), folded, micro_second);

        // Lead rewrites computed directly from the lead + its `next` byte (cross-window safe).
        v128_t next_promote = sz_utf8_gather64_v128_(promo_lo, promo_1, promo_2, promo_3,
                                                     wasm_v128_and(next, wasm_i8x16_splat(0x3F)));
        v128_t promote_lead = wasm_v128_and(is_ce_lead, wasm_i8x16_ne(next_promote, wasm_i8x16_splat(0)));
        v128_t micro_lead = wasm_v128_and(wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xC2)),
                                          wasm_i8x16_eq(next, wasm_i8x16_splat((sz_i8_t)0xB5)));
        folded = sz_utf8_masked_add_v128_(folded, promote_lead, 0x01);
        folded = sz_utf8_masked_add_v128_(folded, micro_lead, 0x0C);
        sz_store_partial_v128_((sz_ptr_t)(dst + pos), folded, window);
    }
}

SZ_INTERNAL void sz_utf8_uncased_fold_armenian_strip_v128_(sz_u8_t const *src, sz_size_t vector_length, sz_u8_t *dst) {
    for (sz_size_t pos = 0; pos < vector_length; pos += 16) {
        sz_size_t window = vector_length - pos < 16 ? vector_length - pos : 16;
        sz_utf8_uncased_window_v128_t_ chunk = sz_utf8_uncased_load_window_v128_(src, pos);
        v128_t source = chunk.source, previous = chunk.previous, next = chunk.next;
        v128_t is_d4 = wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xD4));
        v128_t is_d5 = wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xD5));
        v128_t after_d4 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xD4));
        v128_t after_d5 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xD5));
        v128_t folded = sz_ascii_fold_v128_(source);
        v128_t d4_second = wasm_v128_and(after_d4, wasm_u8x16_ge(source, wasm_i8x16_splat((sz_i8_t)0xB1)));
        v128_t d5_plus30 = wasm_v128_and(after_d5, sz_utf8_in_range_v128_(source, 0x80, 0x10));
        v128_t d5_minus10 = wasm_v128_and(after_d5, sz_utf8_in_range_v128_(source, 0x90, 0x07));
        folded = sz_utf8_masked_add_v128_(folded, wasm_v128_or(d4_second, d5_minus10), 0xF0);
        folded = sz_utf8_masked_add_v128_(folded, d5_plus30, 0x30);
        v128_t promotes_d4 = wasm_v128_and(is_d4, wasm_u8x16_ge(next, wasm_i8x16_splat((sz_i8_t)0xB1)));
        v128_t promotes_d5 = wasm_v128_and(is_d5, sz_utf8_in_range_v128_(next, 0x90, 0x07));
        folded = sz_utf8_masked_add_v128_(folded, wasm_v128_or(promotes_d4, promotes_d5), 0x01);
        sz_store_partial_v128_((sz_ptr_t)(dst + pos), folded, window);
    }
}

SZ_INTERNAL void sz_utf8_uncased_fold_vietnamese_strip_v128_(sz_u8_t const *src, sz_size_t vector_length, sz_u8_t *dst) {
    for (sz_size_t pos = 0; pos < vector_length; pos += 16) {
        sz_size_t window = vector_length - pos < 16 ? vector_length - pos : 16;
        v128_t source = wasm_v128_load(src + pos);
        v128_t previous = sz_utf8_uncased_slide1up_v128_(source, pos > 0 ? src[pos - 1] : 0);
        v128_t previous2 = sz_utf8_uncased_slide1up_v128_(previous, pos > 1 ? src[pos - 2] : 0);
        v128_t is_continuation = wasm_i8x16_eq(wasm_v128_and(source, wasm_i8x16_splat((sz_i8_t)0xC0)),
                                               wasm_i8x16_splat((sz_i8_t)0x80));
        v128_t after_c3 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xC3));
        v128_t after_c4 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xC4));
        v128_t after_c5 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xC5));
        v128_t after_c6 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xC6));
        v128_t folded = sz_ascii_fold_v128_(source);
        v128_t latin1_range = sz_utf8_in_range_v128_(source, 0x80, 0x1F);
        v128_t is_latin1_upper = wasm_v128_andnot(wasm_v128_and(after_c3, latin1_range),
                                                  wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0x97)));
        folded = sz_utf8_masked_add_v128_(folded, is_latin1_upper, 0x20);
        v128_t delta = sz_utf8_uncased_latin_delta_v128_(source, after_c4, after_c5, after_c6, is_continuation);
        folded = wasm_i8x16_add(folded, wasm_v128_and(delta, wasm_i8x16_splat(0x01)));
        v128_t after_e1_pair = wasm_v128_and(wasm_i8x16_eq(previous2, wasm_i8x16_splat((sz_i8_t)0xE1)),
                                             sz_utf8_in_range_v128_(previous, 0xB8, 0x04));
        v128_t third_even = wasm_i8x16_eq(wasm_v128_and(source, wasm_i8x16_splat(0x01)), wasm_i8x16_splat(0));
        folded = sz_utf8_masked_add_v128_(folded, wasm_v128_and(after_e1_pair, third_even), 0x01);
        sz_store_partial_v128_((sz_ptr_t)(dst + pos), folded, window);
    }
}

#pragma endregion // Per script fold strips

#pragma region Per script alarm strips

/** @brief Fold the first lead-danger from a window's 0/1 second-byte danger vector into `*best` (min). */
SZ_INTERNAL void sz_utf8_uncased_alarm_window_(v128_t danger_second, sz_size_t pos, sz_size_t window, long *best) {
    sz_u32_t bits = (sz_u32_t)wasm_i8x16_bitmask(wasm_i8x16_ne(danger_second, wasm_i8x16_splat(0)));
    if (window < 16) bits &= ((sz_u32_t)1 << window) - 1;
    while (bits) {
        sz_size_t g = pos + sz_u32_ctz(bits);
        if (g >= 1) { // index-0 danger has its lead before the strip
            long lead = (long)(g - 1);
            if (*best < 0 || lead < *best) *best = lead;
            return;
        }
        bits &= bits - 1;
    }
}

SZ_INTERNAL long sz_utf8_uncased_alarm_western_europe_strip_v128_(sz_u8_t const *src, sz_size_t vector_length) {
    long best = -1;
    for (sz_size_t pos = 0; pos < vector_length; pos += 16) {
        sz_size_t window = vector_length - pos < 16 ? vector_length - pos : 16;
        sz_utf8_uncased_window_v128_t_ chunk = sz_utf8_uncased_load_window_v128_(src, pos);
        v128_t source = chunk.source, previous = chunk.previous, next = chunk.next;
        v128_t after_c3 = sz_utf8_uncased_eq01_v128_(previous, 0xC3);
        v128_t after_c5 = sz_utf8_uncased_eq01_v128_(previous, 0xC5);
        v128_t danger = wasm_v128_and(
            wasm_v128_and(sz_utf8_uncased_eq01_v128_(source, 0xBA), sz_utf8_uncased_eq01_v128_(previous, 0xE1)),
            sz_utf8_uncased_inrange01_v128_(next, 0x96, 0x09));
        danger = wasm_v128_or(
            danger,
            wasm_v128_and(wasm_v128_and(sz_utf8_uncased_eq01_v128_(source, 0x84), sz_utf8_uncased_eq01_v128_(previous, 0xE2)),
                          wasm_v128_or(sz_utf8_uncased_eq01_v128_(next, 0xAA), sz_utf8_uncased_eq01_v128_(next, 0xAB))));
        danger = wasm_v128_or(
            danger, wasm_v128_and(sz_utf8_uncased_eq01_v128_(source, 0xAC), sz_utf8_uncased_eq01_v128_(previous, 0xEF)));
        danger = wasm_v128_or(danger, wasm_v128_and(after_c5, sz_utf8_uncased_eq01_v128_(source, 0xBF)));
        danger = wasm_v128_or(danger, wasm_v128_and(after_c5, sz_utf8_uncased_eq01_v128_(source, 0xB8)));
        danger = wasm_v128_or(danger, wasm_v128_and(after_c3, sz_utf8_uncased_eq01_v128_(source, 0x9F)));
        sz_utf8_uncased_alarm_window_(danger, pos, window, &best);
        if (best >= 0) break;
    }
    return best;
}

SZ_INTERNAL long sz_utf8_uncased_alarm_central_europe_strip_v128_(sz_u8_t const *src, sz_size_t vector_length) {
    long best = -1;
    for (sz_size_t pos = 0; pos < vector_length; pos += 16) {
        sz_size_t window = vector_length - pos < 16 ? vector_length - pos : 16;
        v128_t source = wasm_v128_load(src + pos);
        v128_t previous = sz_utf8_uncased_slide1up_v128_(source, pos > 0 ? src[pos - 1] : 0);
        v128_t after_c3 = sz_utf8_uncased_eq01_v128_(previous, 0xC3);
        v128_t after_c4 = sz_utf8_uncased_eq01_v128_(previous, 0xC4);
        v128_t after_c5 = sz_utf8_uncased_eq01_v128_(previous, 0xC5);
        v128_t danger = wasm_v128_and(sz_utf8_uncased_eq01_v128_(source, 0x84), sz_utf8_uncased_eq01_v128_(previous, 0xE2));
        danger = wasm_v128_or(danger, wasm_v128_and(after_c3, sz_utf8_uncased_eq01_v128_(source, 0x9F)));
        danger = wasm_v128_or(danger, wasm_v128_and(after_c4, sz_utf8_uncased_eq01_v128_(source, 0xB0)));
        danger = wasm_v128_or(danger, wasm_v128_and(after_c4, sz_utf8_uncased_eq01_v128_(source, 0xBF)));
        danger = wasm_v128_or(danger, wasm_v128_and(after_c5, sz_utf8_uncased_eq01_v128_(source, 0xBF)));
        danger = wasm_v128_or(danger, wasm_v128_and(after_c5, sz_utf8_uncased_eq01_v128_(source, 0xB8)));
        danger = wasm_v128_or(
            danger, wasm_v128_and(sz_utf8_uncased_eq01_v128_(source, 0xAC), sz_utf8_uncased_eq01_v128_(previous, 0xEF)));
        sz_utf8_uncased_alarm_window_(danger, pos, window, &best);
        if (best >= 0) break;
    }
    return best;
}

SZ_INTERNAL long sz_utf8_uncased_alarm_cyrillic_strip_v128_(sz_u8_t const *src, sz_size_t vector_length) {
    long best = -1;
    for (sz_size_t pos = 0; pos < vector_length; pos += 16) {
        sz_size_t window = vector_length - pos < 16 ? vector_length - pos : 16;
        sz_utf8_uncased_window_v128_t_ chunk = sz_utf8_uncased_load_window_v128_(src, pos);
        v128_t source = chunk.source, previous = chunk.previous, next = chunk.next;
        v128_t danger = wasm_v128_and(
            wasm_v128_and(sz_utf8_uncased_eq01_v128_(source, 0xB2), sz_utf8_uncased_eq01_v128_(previous, 0xE1)),
            sz_utf8_uncased_inrange01_v128_(next, 0x80, 0x09));
        sz_utf8_uncased_alarm_window_(danger, pos, window, &best);
        if (best >= 0) break;
    }
    return best;
}

SZ_INTERNAL long sz_utf8_uncased_alarm_greek_strip_v128_(sz_u8_t const *src, sz_size_t vector_length) {
    long best = -1, blanket = -1;
    for (sz_size_t pos = 0; pos < vector_length; pos += 16) {
        sz_size_t window = vector_length - pos < 16 ? vector_length - pos : 16;
        v128_t source = wasm_v128_load(src + pos);
        v128_t previous = sz_utf8_uncased_slide1up_v128_(source, pos > 0 ? src[pos - 1] : 0);
        v128_t after_ce = sz_utf8_uncased_eq01_v128_(previous, 0xCE);
        v128_t after_cf = sz_utf8_uncased_eq01_v128_(previous, 0xCF);
        v128_t second_90_b0_a2 = wasm_v128_or(
            wasm_v128_or(sz_utf8_uncased_eq01_v128_(source, 0x90), sz_utf8_uncased_eq01_v128_(source, 0xB0)),
            sz_utf8_uncased_eq01_v128_(source, 0xA2));
        v128_t danger = wasm_v128_and(after_ce, second_90_b0_a2);
        v128_t second_9x = wasm_v128_or(
            wasm_v128_or(sz_utf8_uncased_eq01_v128_(source, 0x90), sz_utf8_uncased_eq01_v128_(source, 0x91)),
            wasm_v128_or(sz_utf8_uncased_eq01_v128_(source, 0x95), sz_utf8_uncased_eq01_v128_(source, 0x96)));
        v128_t second_bx = wasm_v128_or(
            wasm_v128_or(sz_utf8_uncased_eq01_v128_(source, 0xB0), sz_utf8_uncased_eq01_v128_(source, 0xB1)),
            wasm_v128_or(sz_utf8_uncased_eq01_v128_(source, 0xB4), sz_utf8_uncased_eq01_v128_(source, 0xB5)));
        danger = wasm_v128_or(danger, wasm_v128_and(after_cf, wasm_v128_or(second_9x, second_bx)));
        danger = wasm_v128_or(
            danger, wasm_v128_and(sz_utf8_uncased_eq01_v128_(source, 0x84), sz_utf8_uncased_eq01_v128_(previous, 0xE2)));
        sz_utf8_uncased_alarm_window_(danger, pos, window, &best);
        if (blanket < 0) {
            v128_t lead_blanket = wasm_v128_or(wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xE1)),
                                               wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xCD)));
            int first = sz_utf8_first_set_v128_(lead_blanket, window);
            if (first >= 0) blanket = (long)(pos + (sz_size_t)first);
        }
        if (best >= 0 || blanket >= 0) break;
    }
    if (best < 0) return blanket;
    if (blanket < 0) return best;
    return best < blanket ? best : blanket;
}

SZ_INTERNAL long sz_utf8_uncased_alarm_armenian_strip_v128_(sz_u8_t const *src, sz_size_t vector_length) {
    long best = -1;
    for (sz_size_t pos = 0; pos < vector_length; pos += 16) {
        sz_size_t window = vector_length - pos < 16 ? vector_length - pos : 16;
        v128_t source = wasm_v128_load(src + pos);
        v128_t previous = sz_utf8_uncased_slide1up_v128_(source, pos > 0 ? src[pos - 1] : 0);
        v128_t danger = wasm_v128_and(sz_utf8_uncased_eq01_v128_(source, 0x87), sz_utf8_uncased_eq01_v128_(previous, 0xD6));
        danger = wasm_v128_or(
            danger, wasm_v128_and(sz_utf8_uncased_eq01_v128_(source, 0xAC), sz_utf8_uncased_eq01_v128_(previous, 0xEF)));
        sz_utf8_uncased_alarm_window_(danger, pos, window, &best);
        if (best >= 0) break;
    }
    return best;
}

SZ_INTERNAL long sz_utf8_uncased_alarm_vietnamese_strip_v128_(sz_u8_t const *src, sz_size_t vector_length) {
    long best = -1;
    for (sz_size_t pos = 0; pos < vector_length; pos += 16) {
        sz_size_t window = vector_length - pos < 16 ? vector_length - pos : 16;
        sz_utf8_uncased_window_v128_t_ chunk = sz_utf8_uncased_load_window_v128_(src, pos);
        v128_t source = chunk.source, previous = chunk.previous, next = chunk.next;
        v128_t danger = wasm_v128_and(
            wasm_v128_and(sz_utf8_uncased_eq01_v128_(source, 0xBA), sz_utf8_uncased_eq01_v128_(previous, 0xE1)),
            sz_utf8_uncased_inrange01_v128_(next, 0x96, 0x0A));
        danger = wasm_v128_or(
            danger, wasm_v128_and(sz_utf8_uncased_eq01_v128_(source, 0x9F), sz_utf8_uncased_eq01_v128_(previous, 0xC3)));
        danger = wasm_v128_or(
            danger, wasm_v128_and(sz_utf8_uncased_eq01_v128_(source, 0xBF), sz_utf8_uncased_eq01_v128_(previous, 0xC5)));
        danger = wasm_v128_or(
            danger, wasm_v128_and(sz_utf8_uncased_eq01_v128_(source, 0xAC), sz_utf8_uncased_eq01_v128_(previous, 0xEF)));
        danger = wasm_v128_or(
            danger, wasm_v128_and(sz_utf8_uncased_eq01_v128_(source, 0x84), sz_utf8_uncased_eq01_v128_(previous, 0xE2)));
        v128_t is_cont = wasm_i8x16_eq(wasm_v128_and(source, wasm_i8x16_splat((sz_i8_t)0xC0)),
                                       wasm_i8x16_splat((sz_i8_t)0x80));
        v128_t delta = sz_utf8_uncased_latin_delta_v128_(source, wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xC4)),
                                                    wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xC5)),
                                                    wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xC6)), is_cont);
        v128_t irregular = wasm_v128_and(
            wasm_i8x16_eq(wasm_v128_and(delta, wasm_i8x16_splat((sz_i8_t)0x80)), wasm_i8x16_splat((sz_i8_t)0x80)),
            wasm_i8x16_splat(1));
        danger = wasm_v128_or(danger, irregular);
        sz_utf8_uncased_alarm_window_(danger, pos, window, &best);
        if (best >= 0) break;
    }
    return best;
}

SZ_INTERNAL long sz_utf8_uncased_alarm_georgian_strip_v128_(sz_u8_t const *src, sz_size_t vector_length) {
    long best = -1;
    for (sz_size_t pos = 0; pos < vector_length; pos += 16) {
        sz_size_t window = vector_length - pos < 16 ? vector_length - pos : 16;
        sz_utf8_uncased_window_v128_t_ chunk = sz_utf8_uncased_load_window_v128_(src, pos);
        v128_t source = chunk.source, previous = chunk.previous, next = chunk.next;
        v128_t after_e1 = sz_utf8_uncased_eq01_v128_(previous, 0xE1);
        v128_t danger = wasm_v128_and(sz_utf8_uncased_eq01_v128_(source, 0xB2), after_e1);
        danger = wasm_v128_or(danger, wasm_v128_and(wasm_v128_and(sz_utf8_uncased_eq01_v128_(source, 0x82), after_e1),
                                                    sz_utf8_uncased_inrange01_v128_(next, 0xA0, 0x46)));
        danger = wasm_v128_or(
            danger, wasm_v128_and(sz_utf8_uncased_eq01_v128_(source, 0xB4), sz_utf8_uncased_eq01_v128_(previous, 0xE2)));
        sz_utf8_uncased_alarm_window_(danger, pos, window, &best);
        if (best >= 0) break;
    }
    return best;
}

#pragma endregion // Per script alarm strips

#pragma region Scripted driver

typedef void (*sz_utf8_uncased_fold_strip_v128_t_)(sz_u8_t const *, sz_size_t, sz_u8_t *);
typedef long (*sz_utf8_uncased_alarm_strip_v128_t_)(sz_u8_t const *, sz_size_t);

SZ_INTERNAL sz_cptr_t sz_utf8_uncased_find_scripted_v128_(                                         //
    sz_utf8_uncased_fold_strip_v128_t_ fold, sz_utf8_uncased_alarm_strip_v128_t_ alarm,                 //
    sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length, //
    sz_utf8_uncased_needle_metadata_t const *needle_metadata, sz_size_t *matched_length) {

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_size_t const offset_second = needle_metadata->probe_second;
    sz_size_t const offset_third = needle_metadata->probe_third;
    sz_size_t const offset_last = folded_window_length - 1;
    sz_u8_t const probe_first = needle_metadata->folded_slice[0];
    sz_u8_t const probe_second = needle_metadata->folded_slice[offset_second];
    sz_u8_t const probe_third = needle_metadata->folded_slice[offset_third];
    sz_u8_t const probe_last = needle_metadata->folded_slice[offset_last];

    sz_rune_t needle_first_safe_folded_rune = 0;
    if (alarm) sz_rune_parse_unchecked((sz_cptr_t)needle_metadata->folded_slice, &needle_first_safe_folded_rune);

    // 32-byte chunks (NEON/Ice Lake granularity): finer than RVV strips so an alarmed chunk only serial-scans
    // ~32 bytes, and the danger retreat `step = valid_starts - 2` keeps cross-chunk patterns visible.
    enum { chunk_capacity_k = 32, fold_pad_k = 16 };
    sz_u8_t source_buffer[chunk_capacity_k + fold_pad_k];
    sz_u8_t folded_buffer[chunk_capacity_k + fold_pad_k];

    sz_cptr_t haystack_ptr = haystack;
    while (haystack_ptr < haystack_end) {
        sz_size_t const available = (sz_size_t)(haystack_end - haystack_ptr);
        if (available < folded_window_length) break;
        sz_size_t chunk_size = available < chunk_capacity_k ? available : chunk_capacity_k;
        sz_size_t const valid_starts = chunk_size - folded_window_length + 1;
        sz_size_t const step = !alarm ? valid_starts : valid_starts > 2 ? valid_starts - 2 : 1;

        sz_u8_t const *source = sz_utf8_uncased_load_padded_v128_(haystack_ptr, chunk_size, source_buffer,
                                                             chunk_capacity_k + fold_pad_k);

        if (alarm) {
            if (alarm(source, chunk_size) >= 0) {
                sz_cptr_t match = sz_utf8_uncased_find_in_danger_zone_( //
                    haystack, haystack_length, needle, needle_length, haystack_ptr, chunk_size,
                    needle_first_safe_folded_rune, needle_metadata->offset_in_unfolded, matched_length);
                if (match) return match;
                haystack_ptr += step;
                continue;
            }
        }

        fold(source, chunk_size, folded_buffer);

        for (sz_size_t position = 0; position < valid_starts; position += 16) {
            sz_size_t lanes = valid_starts - position < 16 ? valid_starts - position : 16;
            v128_t match_vec = wasm_i8x16_eq(wasm_v128_load(folded_buffer + position),
                                             wasm_i8x16_splat((sz_i8_t)probe_first));
            match_vec = wasm_v128_and(match_vec, wasm_i8x16_eq(wasm_v128_load(folded_buffer + position + offset_second),
                                                               wasm_i8x16_splat((sz_i8_t)probe_second)));
            match_vec = wasm_v128_and(match_vec, wasm_i8x16_eq(wasm_v128_load(folded_buffer + position + offset_third),
                                                               wasm_i8x16_splat((sz_i8_t)probe_third)));
            match_vec = wasm_v128_and(match_vec, wasm_i8x16_eq(wasm_v128_load(folded_buffer + position + offset_last),
                                                               wasm_i8x16_splat((sz_i8_t)probe_last)));
            sz_u32_t matches = (sz_u32_t)wasm_i8x16_bitmask(match_vec);
            if (lanes < 16) matches &= ((sz_u32_t)1 << lanes) - 1;
            while (matches) {
                sz_size_t const candidate_offset = position + sz_u32_ctz(matches);
                matches &= matches - 1;
                int window_ok = 1;
                for (sz_size_t byte_index = 0; byte_index < folded_window_length; ++byte_index)
                    if (folded_buffer[candidate_offset + byte_index] != needle_metadata->folded_slice[byte_index]) {
                        window_ok = 0;
                        break;
                    }
                if (!window_ok) continue;
                sz_cptr_t match = sz_utf8_uncased_verify_match_( //
                    haystack, haystack_length, needle, needle_length,
                    (sz_size_t)(haystack_ptr - haystack) + candidate_offset, folded_window_length,
                    needle_metadata->offset_in_unfolded,
                    needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded,
                    matched_length);
                if (match) return match;
            }
        }
        haystack_ptr += step;
    }

    if (alarm && haystack_ptr < haystack_end) {
        sz_cptr_t match = sz_utf8_uncased_find_in_danger_zone_( //
            haystack, haystack_length, needle, needle_length, haystack_ptr, (sz_size_t)(haystack_end - haystack_ptr),
            needle_first_safe_folded_rune, needle_metadata->offset_in_unfolded, matched_length);
        if (match) return match;
    }
    return SZ_NULL_CHAR;
}

#pragma endregion // Scripted driver

SZ_PUBLIC sz_cptr_t sz_utf8_uncased_violation_v128(sz_cptr_t str, sz_size_t length);

SZ_PUBLIC sz_cptr_t sz_utf8_uncased_find_v128( //
    sz_cptr_t haystack, sz_size_t haystack_length,      //
    sz_cptr_t needle, sz_size_t needle_length,          //
    sz_utf8_uncased_needle_metadata_t *needle_metadata, sz_size_t *matched_length) {

    if (needle_length == 0) {
        *matched_length = 0;
        return haystack;
    }

    int const is_unknown = needle_metadata->kernel_id == sz_utf8_uncased_rune_unknown_k;
    int const known_agnostic = needle_metadata->kernel_id == sz_utf8_uncased_rune_invariant_k;
    if (known_agnostic || (is_unknown && sz_utf8_uncased_violation_v128(needle, needle_length) == SZ_NULL_CHAR)) {
        sz_cptr_t result = sz_find_v128(haystack, haystack_length, needle, needle_length);
        *matched_length = result ? needle_length : 0;
        return result;
    }

    if (is_unknown) {
        sz_utf8_uncased_needle_metadata_(needle, needle_length, needle_metadata);
        if (needle_metadata->kernel_id == sz_utf8_uncased_rune_fallback_serial_k)
            return sz_utf8_uncased_find_serial(haystack, haystack_length, needle, needle_length,
                                                        needle_metadata, matched_length);
    }

    switch (needle_metadata->kernel_id) {
    case sz_utf8_uncased_rune_ascii_invariant_k:
        return sz_utf8_uncased_find_scripted_v128_(sz_utf8_uncased_fold_ascii_strip_v128_, SZ_NULL, haystack, haystack_length,
                                              needle, needle_length, needle_metadata, matched_length);
    case sz_utf8_uncased_rune_safe_western_europe_k:
        return sz_utf8_uncased_find_scripted_v128_(sz_utf8_uncased_fold_western_europe_strip_v128_,
                                              sz_utf8_uncased_alarm_western_europe_strip_v128_, haystack, haystack_length,
                                              needle, needle_length, needle_metadata, matched_length);
    case sz_utf8_uncased_rune_safe_central_europe_k:
        return sz_utf8_uncased_find_scripted_v128_(sz_utf8_uncased_fold_central_europe_strip_v128_,
                                              sz_utf8_uncased_alarm_central_europe_strip_v128_, haystack, haystack_length,
                                              needle, needle_length, needle_metadata, matched_length);
    case sz_utf8_uncased_rune_safe_cyrillic_k:
        return sz_utf8_uncased_find_scripted_v128_(sz_utf8_uncased_fold_cyrillic_strip_v128_,
                                              sz_utf8_uncased_alarm_cyrillic_strip_v128_, haystack, haystack_length, needle,
                                              needle_length, needle_metadata, matched_length);
    case sz_utf8_uncased_rune_safe_greek_k:
        return sz_utf8_uncased_find_scripted_v128_(sz_utf8_uncased_fold_greek_strip_v128_, sz_utf8_uncased_alarm_greek_strip_v128_,
                                              haystack, haystack_length, needle, needle_length, needle_metadata,
                                              matched_length);
    case sz_utf8_uncased_rune_safe_armenian_k:
        return sz_utf8_uncased_find_scripted_v128_(sz_utf8_uncased_fold_armenian_strip_v128_,
                                              sz_utf8_uncased_alarm_armenian_strip_v128_, haystack, haystack_length, needle,
                                              needle_length, needle_metadata, matched_length);
    case sz_utf8_uncased_rune_safe_vietnamese_k:
        return sz_utf8_uncased_find_scripted_v128_(sz_utf8_uncased_fold_vietnamese_strip_v128_,
                                              sz_utf8_uncased_alarm_vietnamese_strip_v128_, haystack, haystack_length,
                                              needle, needle_length, needle_metadata, matched_length);
    case sz_utf8_uncased_rune_safe_georgian_k:
        return sz_utf8_uncased_find_scripted_v128_(sz_utf8_uncased_fold_ascii_strip_v128_, sz_utf8_uncased_alarm_georgian_strip_v128_,
                                              haystack, haystack_length, needle, needle_length, needle_metadata,
                                              matched_length);
    default: break;
    }

    needle_metadata->kernel_id = sz_utf8_uncased_rune_fallback_serial_k;
    return sz_utf8_uncased_find_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                                matched_length);
}

#pragma region Case Invariance

/** @brief 32-bit movemask of two 16-byte registers: bit `i` = lane `i` of `low`, bit `16+i` = lane `i` of `high`. */
SZ_INTERNAL sz_u32_t sz_utf8_uncased_movemask_v128x2_(v128_t low, v128_t high) {
    return (sz_u32_t)(sz_u16_t)wasm_i8x16_bitmask(low) | ((sz_u32_t)(sz_u16_t)wasm_i8x16_bitmask(high) << 16);
}

SZ_PUBLIC sz_cptr_t sz_utf8_uncased_violation_v128(sz_cptr_t str, sz_size_t length) {
    sz_u8_t const *text_cursor = (sz_u8_t const *)str;
    while (length) {
        sz_size_t block_length = length < 29 ? length : 29;
        sz_u32_t lead_mask = block_length >= 32 ? 0xFFFFFFFFu : (((sz_u32_t)1 << block_length) - 1);
        v128_t low, high;
        if (length >= 32) { low = wasm_v128_load(text_cursor), high = wasm_v128_load(text_cursor + 16); }
        else {
            low = sz_load_partial_v128_((sz_cptr_t)text_cursor, length < 16 ? length : 16);
            high = length > 16 ? sz_load_partial_v128_((sz_cptr_t)(text_cursor + 16), length - 16)
                               : wasm_i8x16_splat(0);
        }
        sz_u32_t is_upper = sz_utf8_uncased_movemask_v128x2_(sz_utf8_in_range_v128_(low, 'A', 26),
                                                        sz_utf8_in_range_v128_(high, 'A', 26));
        sz_u32_t is_lower = sz_utf8_uncased_movemask_v128x2_(sz_utf8_in_range_v128_(low, 'a', 26),
                                                        sz_utf8_in_range_v128_(high, 'a', 26));
        if (is_upper | is_lower) return sz_utf8_uncased_violation_serial(str, length);

        v128_t x80 = wasm_i8x16_splat((sz_i8_t)0x80);
        sz_u32_t is_non_ascii = sz_utf8_uncased_movemask_v128x2_(wasm_u8x16_ge(low, x80), wasm_u8x16_ge(high, x80)) &
                                lead_mask;
        if (is_non_ascii) {
            v128_t xe0 = wasm_i8x16_splat((sz_i8_t)0xE0), xf0 = wasm_i8x16_splat((sz_i8_t)0xF0);
            v128_t xc0 = wasm_i8x16_splat((sz_i8_t)0xC0), xf8 = wasm_i8x16_splat((sz_i8_t)0xF8);
            sz_u32_t is_two = sz_utf8_uncased_movemask_v128x2_(wasm_i8x16_eq(wasm_v128_and(low, xe0), xc0),
                                                          wasm_i8x16_eq(wasm_v128_and(high, xe0), xc0)) &
                              lead_mask;
            sz_u32_t is_three = sz_utf8_uncased_movemask_v128x2_(wasm_i8x16_eq(wasm_v128_and(low, xf0), xe0),
                                                            wasm_i8x16_eq(wasm_v128_and(high, xf0), xe0)) &
                                lead_mask;
            sz_u32_t is_four = sz_utf8_uncased_movemask_v128x2_(wasm_i8x16_eq(wasm_v128_and(low, xf8), xf0),
                                                           wasm_i8x16_eq(wasm_v128_and(high, xf8), xf0)) &
                               lead_mask;
            if (is_four) {
                static sz_u8_t const seconds[5] = {0x90, 0x91, 0x96, 0x9D, 0x9E};
                sz_u32_t hit = 0;
                for (int value = 0; value < 5; ++value) {
                    v128_t b = wasm_i8x16_splat((sz_i8_t)seconds[value]);
                    hit |= sz_utf8_uncased_movemask_v128x2_(wasm_i8x16_eq(low, b), wasm_i8x16_eq(high, b));
                }
                if ((is_four << 1) & hit) return sz_utf8_uncased_violation_serial(str, length);
            }
            if (is_two) {
                sz_u32_t is_bicameral = sz_utf8_uncased_movemask_v128x2_(sz_utf8_in_range_v128_(low, 0xC3, 0x14),
                                                                    sz_utf8_in_range_v128_(high, 0xC3, 0x14));
                v128_t xc2 = wasm_i8x16_splat((sz_i8_t)0xC2), xb5 = wasm_i8x16_splat((sz_i8_t)0xB5);
                sz_u32_t is_c2 = sz_utf8_uncased_movemask_v128x2_(wasm_i8x16_eq(low, xc2), wasm_i8x16_eq(high, xc2)) &
                                 is_two;
                if (is_c2) {
                    sz_u32_t is_b5 = sz_utf8_uncased_movemask_v128x2_(wasm_i8x16_eq(low, xb5), wasm_i8x16_eq(high, xb5));
                    if ((is_c2 << 1) & is_b5) return sz_utf8_uncased_violation_serial(str, length);
                }
                if (is_bicameral & is_two) return sz_utf8_uncased_violation_serial(str, length);
            }
            if (is_three) {
                v128_t xe1 = wasm_i8x16_splat((sz_i8_t)0xE1), xef = wasm_i8x16_splat((sz_i8_t)0xEF);
                sz_u32_t is_e1 = sz_utf8_uncased_movemask_v128x2_(wasm_i8x16_eq(low, xe1), wasm_i8x16_eq(high, xe1));
                if (is_e1 & is_three) return sz_utf8_uncased_violation_serial(str, length);
                sz_u32_t is_ef = sz_utf8_uncased_movemask_v128x2_(wasm_i8x16_eq(low, xef), wasm_i8x16_eq(high, xef));
                if (is_ef & is_three) return sz_utf8_uncased_violation_serial(str, length);
                v128_t xe2 = wasm_i8x16_splat((sz_i8_t)0xE2);
                sz_u32_t is_e2 = sz_utf8_uncased_movemask_v128x2_(wasm_i8x16_eq(low, xe2), wasm_i8x16_eq(high, xe2)) &
                                 is_three;
                if (is_e2) {
                    sz_u32_t e2_safe = sz_utf8_uncased_movemask_v128x2_(sz_utf8_in_range_v128_(low, 0x80, 0x04),
                                                                   sz_utf8_in_range_v128_(high, 0x80, 0x04));
                    if ((is_e2 << 1) & ~e2_safe) return sz_utf8_uncased_violation_serial(str, length);
                }
                v128_t xea = wasm_i8x16_splat((sz_i8_t)0xEA);
                sz_u32_t is_ea = sz_utf8_uncased_movemask_v128x2_(wasm_i8x16_eq(low, xea), wasm_i8x16_eq(high, xea)) &
                                 is_three;
                if (is_ea) {
                    sz_u32_t is_99 = sz_utf8_uncased_movemask_v128x2_(sz_utf8_in_range_v128_(low, 0x99, 0x07),
                                                                 sz_utf8_in_range_v128_(high, 0x99, 0x07));
                    sz_u32_t is_ac = sz_utf8_uncased_movemask_v128x2_(sz_utf8_in_range_v128_(low, 0xAC, 0x03),
                                                                 sz_utf8_in_range_v128_(high, 0xAC, 0x03));
                    if ((is_ea << 1) & (is_99 | is_ac)) return sz_utf8_uncased_violation_serial(str, length);
                }
            }
        }
        text_cursor += block_length;
        length -= block_length;
    }
    return SZ_NULL_CHAR;
}

#pragma endregion // Case Invariance

SZ_PUBLIC sz_ordering_t sz_utf8_uncased_order_v128(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                            sz_size_t b_length) {
    return sz_utf8_uncased_order_serial(a, a_length, b, b_length);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_UNCASED_V128_H_
