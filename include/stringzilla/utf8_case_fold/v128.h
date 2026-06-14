/**
 *  @brief WebAssembly SIMD128 backend for UTF-8 case folding.
 *  @file include/stringzilla/utf8_case_fold/v128.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_case_fold.h
 */
#ifndef STRINGZILLA_UTF8_CASE_FOLD_V128_H_
#define STRINGZILLA_UTF8_CASE_FOLD_V128_H_

#include "stringzilla/utf8_case_fold/serial.h"
#include "stringzilla/memory/v128.h" // `sz_load_partial_v128_`, `sz_store_partial_v128_`

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

/*  Byte-for-byte equivalent to `sz_utf8_case_fold_serial`, mirroring the RISC-V Vector kernel's
 *  "stop-and-serial" structure on fixed 16-byte windows. Maximal ASCII runs fold in-vector; Latin,
 *  Cyrillic, Greek, Armenian and Georgian text fold in place by their `_strip_v128_` handlers
 *  (dispatched on the lead byte); any other script or length-changing fold (ß→"ss", ΐ→3 runes, final
 *  sigma, ligatures, …) is a "stop" that the handler stops before, deferring that one codepoint to the
 *  serial decode/fold/encode. RVV uses a 64-entry `vrgather`; wasm has only a 16-entry swizzle, so the
 *  Latin delta tables are split into 4×16 sub-tables selected by the index's high two bits. */

#pragma region Helpers

/** @brief Per-codepoint deltas for 2-byte Latin Extended sequences, indexed by the continuation byte's low
 *  6 bits: 0x00 identity, 0x01 fold by +1, 0x80 irregular (route to serial). Identical to the RVV/NEON LUTs. */
static sz_align_(16) sz_u8_t const sz_utf8_fold_latin_c4_deltas_v128_[64] = {
    1,    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, //
    1,    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, //
    1,    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, //
    0x80, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0x80};
static sz_align_(16) sz_u8_t const sz_utf8_fold_latin_c5_deltas_v128_[64] = {
    0, 1, 0, 1, 0, 1, 0, 1, 0,    0x80, 1, 0, 1, 0, 1, 0, //
    1, 0, 1, 0, 1, 0, 1, 0, 1,    0,    1, 0, 1, 0, 1, 0, //
    1, 0, 1, 0, 1, 0, 1, 0, 1,    0,    1, 0, 1, 0, 1, 0, //
    1, 0, 1, 0, 1, 0, 1, 0, 0x80, 1,    0, 1, 0, 1, 0, 0x80};
static sz_align_(16) sz_u8_t const sz_utf8_fold_latin_c6_deltas_v128_[64] = {
    0,    0x80, 1,    0,    1,    0, 0x80, 1,    0, 0x80, 0x80, 1, 0,    0,    0x80, 0x80, //
    0x80, 1,    0,    0x80, 0x80, 0, 0x80, 0x80, 1, 0,    0,    0, 0x80, 0x80, 0,    0x80, //
    1,    0,    1,    0,    1,    0, 0x80, 1,    0, 0x80, 0,    0, 1,    0,    0x80, 1,    //
    0,    0x80, 0x80, 1,    0,    1, 0,    0x80, 1, 0,    0,    0, 1,    0,    0,    0};

/** @brief Fold 16 ASCII bytes: lowercase `A`..`Z` by +0x20, identical to `sz_ascii_fold_`. */
SZ_INTERNAL v128_t sz_ascii_fold_v128_(v128_t bytes) {
    // `(c - 'A') <= 25` (unsigned) == `c >= 'A' && c <= 'Z'`; add 0x20 there, nowhere else.
    v128_t is_upper = wasm_u8x16_le(wasm_i8x16_sub(bytes, wasm_i8x16_splat('A')), wasm_i8x16_splat(25));
    return wasm_i8x16_add(bytes, wasm_v128_and(is_upper, wasm_i8x16_splat(0x20)));
}

/** @brief `result[0] = 0`, `result[i] = vector[i-1]` — the constant-shuffle twin of RVV `vslide1up(vector, 0)`. */
SZ_INTERNAL v128_t sz_utf8_slide1up_v128_(v128_t vector) {
    return wasm_i8x16_shuffle(vector, wasm_i8x16_splat(0), 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
}
/** @brief `result[i] = vector[i+1]`, `result[15] = carry` — the twin of RVV `vslide1down(vector, carry)`. */
SZ_INTERNAL v128_t sz_utf8_slide1down_v128_(v128_t vector, sz_u8_t carry) {
    return wasm_i8x16_shuffle(vector, wasm_i8x16_splat((sz_i8_t)carry), 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                              15, 16);
}

/** @brief 0xFF where `(byte - start)` is unsigned-`< length`, i.e. `byte` in `[start, start+length)`. */
SZ_INTERNAL v128_t sz_utf8_in_range_v128_(v128_t bytes, sz_u8_t start, sz_u8_t length) {
    return wasm_u8x16_lt(wasm_i8x16_sub(bytes, wasm_i8x16_splat((sz_i8_t)start)), wasm_i8x16_splat((sz_i8_t)length));
}

/** @brief Add `value` to the lanes flagged by `mask` (0xFF), leave the rest — twin of RVV `vadd_vx_m`. */
SZ_INTERNAL v128_t sz_utf8_masked_add_v128_(v128_t bytes, v128_t mask, sz_u8_t value) {
    return wasm_i8x16_add(bytes, wasm_v128_and(mask, wasm_i8x16_splat((sz_i8_t)value)));
}

/** @brief 64-entry table lookup via four 16-entry swizzles selected by the index's high two bits. */
SZ_INTERNAL v128_t sz_utf8_gather64_v128_(v128_t lut0, v128_t lut1, v128_t lut2, v128_t lut3, v128_t index) {
    v128_t local = wasm_v128_and(index, wasm_i8x16_splat(0x0F));
    v128_t sub = wasm_u8x16_shr(index, 4); // index in [0, 63] -> sub in [0, 3]
    v128_t result = wasm_i8x16_swizzle(lut0, local);
    result = wasm_v128_bitselect(wasm_i8x16_swizzle(lut1, local), result, wasm_i8x16_eq(sub, wasm_i8x16_splat(1)));
    result = wasm_v128_bitselect(wasm_i8x16_swizzle(lut2, local), result, wasm_i8x16_eq(sub, wasm_i8x16_splat(2)));
    result = wasm_v128_bitselect(wasm_i8x16_swizzle(lut3, local), result, wasm_i8x16_eq(sub, wasm_i8x16_splat(3)));
    return result;
}

/** @brief Index of the first set lane in `mask` among the low `valid` lanes, or -1 — twin of RVV `vfirst_m`. */
SZ_INTERNAL int sz_utf8_first_set_v128_(v128_t mask, sz_size_t vector_length) {
    sz_u32_t bits = (sz_u32_t)wasm_i8x16_bitmask(mask);
    if (vector_length < 16) bits &= ((sz_u32_t)1 << vector_length) - 1;
    return bits ? (int)sz_u32_ctz(bits) : -1;
}

/** @brief Load up to 16 bytes (zero-padded past `available`), so strip handlers never over-read. */
SZ_INTERNAL v128_t sz_utf8_load_window_v128_(sz_u8_t const *source_ptr, sz_size_t available) {
    return available >= 16 ? wasm_v128_load(source_ptr) : sz_load_partial_v128_((sz_cptr_t)source_ptr, available);
}

/** @brief Largest prefix of a window that does not split a trailing multi-byte sequence (twin of RVV trim). */
SZ_INTERNAL sz_size_t sz_utf8_trim_incomplete_v128_(sz_u8_t const *source_ptr, sz_size_t vector_length,
                                                    sz_size_t remaining) {
    if (vector_length >= remaining) return vector_length;
    sz_size_t boundary = vector_length;
    while (boundary && (source_ptr[boundary - 1] & 0xC0) == 0x80) --boundary; // back up to the last lead
    if (!boundary) return vector_length;
    sz_u8_t lead = source_ptr[boundary - 1];
    sz_size_t needed = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 1;
    return ((boundary - 1) + needed > vector_length) ? (boundary - 1) : vector_length;
}

/** @brief Common tail of every strip handler: resolve `consumed` from the first stop, store, set the flag. */
SZ_INTERNAL sz_size_t sz_utf8_strip_finish_v128_(sz_u8_t const *source_ptr, sz_size_t vector_length,
                                                 sz_size_t remaining, v128_t folded, int first_stop,
                                                 sz_u8_t *destination_ptr, int *needs_serial) {
    sz_size_t consumed;
    if (first_stop >= 0) {
        consumed = (sz_size_t)first_stop;
        while (consumed && (source_ptr[consumed] & 0xC0) == 0x80) --consumed; // back up to the lead
        *needs_serial = 1;
    }
    else {
        consumed = sz_utf8_trim_incomplete_v128_(source_ptr, vector_length, remaining);
        *needs_serial = 0;
    }
    if (consumed) sz_store_partial_v128_((sz_ptr_t)destination_ptr, folded, consumed);
    return consumed;
}

#pragma endregion // Helpers

#pragma region Per-script strip handlers

/** @brief Fold one window of Latin (ASCII + Latin-1 C2/C3 + Latin Extended-A/B C4-C6). @sa RVV latin strip. */
SZ_INTERNAL sz_size_t sz_utf8_fold_latin_strip_v128_(sz_u8_t const *source_ptr, sz_size_t remaining,
                                                     sz_u8_t *destination_ptr, int *needs_serial) {
    sz_size_t vector_length = remaining < 16 ? remaining : 16;
    v128_t source = sz_utf8_load_window_v128_(source_ptr, remaining);
    v128_t previous = sz_utf8_slide1up_v128_(source);
    v128_t next = sz_utf8_slide1down_v128_(source, (vector_length < remaining) ? source_ptr[vector_length] : 0);

    v128_t is_continuation = wasm_i8x16_eq(wasm_v128_and(source, wasm_i8x16_splat((sz_i8_t)0xC0)),
                                           wasm_i8x16_splat((sz_i8_t)0x80));
    v128_t after_c2 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xC2));
    v128_t after_c3 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xC3));
    v128_t after_c4 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xC4));
    v128_t after_c5 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xC5));
    v128_t after_c6 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xC6));

    // Latin Extended-A/B continuation deltas: pick the LUT of the preceding lead, only on continuation bytes.
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
    delta = wasm_v128_and(delta, is_continuation); // deltas apply only on continuation bytes
    v128_t is_irregular = wasm_i8x16_ne(wasm_v128_and(delta, wasm_i8x16_splat((sz_i8_t)0x80)), wasm_i8x16_splat(0));

    // Build the folded window: ASCII A-Z, Latin-1 upper +0x20, ß/µ replacements, then the Extended delta.
    v128_t folded = sz_ascii_fold_v128_(source);
    // Latin-1 Supplement 'À'-'Þ' (C3 80-9E, excluding '×' at 0x97) get +0x20.
    v128_t latin1_range = sz_utf8_in_range_v128_(source, 0x80, 0x1F);
    v128_t is_latin1_upper = wasm_v128_andnot(wasm_v128_and(after_c3, latin1_range),
                                              wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0x97)));
    folded = sz_utf8_masked_add_v128_(folded, is_latin1_upper, 0x20);
    // ß (C3 9F) -> "ss": both bytes become 's'.
    v128_t eszett_lead = wasm_v128_and(wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xC3)),
                                       wasm_i8x16_eq(next, wasm_i8x16_splat((sz_i8_t)0x9F)));
    v128_t eszett_second = wasm_v128_and(after_c3, wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0x9F)));
    folded = wasm_v128_bitselect(wasm_i8x16_splat('s'), folded, wasm_v128_or(eszett_lead, eszett_second));
    // µ (C2 B5) -> μ (CE BC): lead byte becomes 0xCE, continuation 0xBC.
    folded = wasm_v128_bitselect(wasm_i8x16_splat((sz_i8_t)0xCE), folded,
                                 wasm_v128_and(wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xC2)),
                                               wasm_i8x16_eq(next, wasm_i8x16_splat((sz_i8_t)0xB5))));
    folded = wasm_v128_bitselect(wasm_i8x16_splat((sz_i8_t)0xBC), folded,
                                 wasm_v128_and(after_c2, wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xB5))));
    // Latin Extended +1 parity delta (0x80 irregulars corrupt their lane but are trimmed off below).
    folded = wasm_i8x16_add(folded, delta);

    // Stops: any lead outside the C2-C6 family, or an irregular continuation.
    v128_t is_non_ascii = wasm_u8x16_gt(source, wasm_i8x16_splat((sz_i8_t)0x7F));
    v128_t is_lead = wasm_v128_andnot(is_non_ascii, is_continuation);
    v128_t is_family_lead = sz_utf8_in_range_v128_(source, 0xC2, 5);
    v128_t is_foreign_lead = wasm_v128_andnot(is_lead, is_family_lead);
    v128_t stop = wasm_v128_or(is_foreign_lead, is_irregular);

    int first_stop = sz_utf8_first_set_v128_(stop, vector_length);
    return sz_utf8_strip_finish_v128_(source_ptr, vector_length, remaining, folded, first_stop, destination_ptr,
                                      needs_serial);
}

/** @brief Fold one window of basic Cyrillic (D0/D1 leads). @sa RVV cyrillic strip. */
SZ_INTERNAL sz_size_t sz_utf8_fold_cyrillic_strip_v128_(sz_u8_t const *source_ptr, sz_size_t remaining,
                                                        sz_u8_t *destination_ptr, int *needs_serial) {
    static
        sz_align_(16) sz_u8_t const second_byte_offsets[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0x10, 0x20, 0xE0, 0, 0, 0, 0, 0};
    sz_size_t vector_length = remaining < 16 ? remaining : 16;
    v128_t source = sz_utf8_load_window_v128_(source_ptr, remaining);
    v128_t next = sz_utf8_slide1down_v128_(source, (vector_length < remaining) ? source_ptr[vector_length] : 0);
    v128_t previous = sz_utf8_slide1up_v128_(source);

    v128_t is_continuation = wasm_i8x16_eq(wasm_v128_and(source, wasm_i8x16_splat((sz_i8_t)0xC0)),
                                           wasm_i8x16_splat((sz_i8_t)0x80));
    v128_t is_d0 = wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xD0));
    v128_t is_d1 = wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xD1));
    v128_t after_d0 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xD0));
    v128_t is_non_ascii = wasm_u8x16_gt(source, wasm_i8x16_splat((sz_i8_t)0x7F));
    v128_t is_lead = wasm_v128_andnot(is_non_ascii, is_continuation);
    v128_t is_foreign_lead = wasm_v128_andnot(is_lead, wasm_v128_or(is_d0, is_d1));
    // Cyrillic Extended-A (D1 A0+) folds by parity across blocks — leave it to serial.
    v128_t is_extended = wasm_v128_and(is_d1, wasm_u8x16_ge(next, wasm_i8x16_splat((sz_i8_t)0xA0)));
    v128_t stop = wasm_v128_or(is_foreign_lead, is_extended);

    // Second-byte offset by high nibble, applied only after a D0 lead.
    v128_t offsets_lut = wasm_v128_load(second_byte_offsets);
    v128_t offset = wasm_i8x16_swizzle(offsets_lut, wasm_u8x16_shr(source, 4));
    offset = wasm_v128_and(offset, after_d0);
    v128_t folded = sz_ascii_fold_v128_(source);
    folded = wasm_i8x16_add(folded, offset);
    // Lead rewrite D0 -> D1 (+1) where the next byte is 80-8F or A0-AF.
    v128_t next_80_8f = sz_utf8_in_range_v128_(next, 0x80, 0x10);
    v128_t next_a0_af = sz_utf8_in_range_v128_(next, 0xA0, 0x10);
    v128_t needs_d1 = wasm_v128_and(is_d0, wasm_v128_or(next_80_8f, next_a0_af));
    folded = sz_utf8_masked_add_v128_(folded, needs_d1, 1);

    int first_stop = sz_utf8_first_set_v128_(stop, vector_length);
    return sz_utf8_strip_finish_v128_(source_ptr, vector_length, remaining, folded, first_stop, destination_ptr,
                                      needs_serial);
}

/** @brief Fold one window of basic Greek (CE/CF leads). @sa RVV greek strip. */
SZ_INTERNAL sz_size_t sz_utf8_fold_greek_strip_v128_(sz_u8_t const *source_ptr, sz_size_t remaining,
                                                     sz_u8_t *destination_ptr, int *needs_serial) {
    sz_size_t vector_length = remaining < 16 ? remaining : 16;
    v128_t source = sz_utf8_load_window_v128_(source_ptr, remaining);
    v128_t next = sz_utf8_slide1down_v128_(source, (vector_length < remaining) ? source_ptr[vector_length] : 0);
    v128_t previous = sz_utf8_slide1up_v128_(source);

    v128_t is_continuation = wasm_i8x16_eq(wasm_v128_and(source, wasm_i8x16_splat((sz_i8_t)0xC0)),
                                           wasm_i8x16_splat((sz_i8_t)0x80));
    v128_t is_ce = wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xCE));
    v128_t is_cf = wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xCF));
    v128_t after_ce = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xCE));
    v128_t after_cf = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xCF));
    v128_t is_non_ascii = wasm_u8x16_gt(source, wasm_i8x16_splat((sz_i8_t)0x7F));
    v128_t is_lead = wasm_v128_andnot(is_non_ascii, is_continuation);
    v128_t is_foreign_lead = wasm_v128_andnot(is_lead, wasm_v128_or(is_ce, is_cf));
    // CE excluded: next < 0x91 (accented uppercase) or next == 0xB0 ('ΰ' expands). CF excluded: next >= 0x8F.
    v128_t ce_excluded = wasm_v128_and(is_ce, wasm_v128_or(wasm_u8x16_lt(next, wasm_i8x16_splat((sz_i8_t)0x91)),
                                                           wasm_i8x16_eq(next, wasm_i8x16_splat((sz_i8_t)0xB0))));
    v128_t cf_excluded = wasm_v128_and(is_cf, wasm_u8x16_ge(next, wasm_i8x16_splat((sz_i8_t)0x8F)));
    v128_t stop = wasm_v128_or(is_foreign_lead, wasm_v128_or(ce_excluded, cf_excluded));

    // Promoting ranges (second byte A0-A1 or A3-AB), used both for the second-byte -0x20 and the lead +1.
    v128_t in_promote_a0 = sz_utf8_in_range_v128_(source, 0xA0, 0x02);
    v128_t in_promote_a3 = sz_utf8_in_range_v128_(source, 0xA3, 0x09);
    v128_t next_promote_a0 = sz_utf8_in_range_v128_(next, 0xA0, 0x02);
    v128_t next_promote_a3 = sz_utf8_in_range_v128_(next, 0xA3, 0x09);

    v128_t folded = sz_ascii_fold_v128_(source);
    v128_t basic_upper = wasm_v128_and(after_ce, sz_utf8_in_range_v128_(source, 0x91, 0x0F));
    folded = sz_utf8_masked_add_v128_(folded, basic_upper, 0x20);
    v128_t promoting_upper = wasm_v128_and(after_ce, wasm_v128_or(in_promote_a0, in_promote_a3));
    folded = sz_utf8_masked_add_v128_(folded, promoting_upper, 0xE0); // -0x20
    v128_t final_sigma = wasm_v128_and(after_cf, wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0x82)));
    folded = sz_utf8_masked_add_v128_(folded, final_sigma, 0x01);
    v128_t promotes_lead = wasm_v128_and(is_ce, wasm_v128_or(next_promote_a0, next_promote_a3));
    folded = sz_utf8_masked_add_v128_(folded, promotes_lead, 0x01); // CE -> CF

    int first_stop = sz_utf8_first_set_v128_(stop, vector_length);
    return sz_utf8_strip_finish_v128_(source_ptr, vector_length, remaining, folded, first_stop, destination_ptr,
                                      needs_serial);
}

/** @brief Fold one window of Armenian (D4/D5/D6 leads). @sa RVV armenian strip. */
SZ_INTERNAL sz_size_t sz_utf8_fold_armenian_strip_v128_(sz_u8_t const *source_ptr, sz_size_t remaining,
                                                        sz_u8_t *destination_ptr, int *needs_serial) {
    sz_size_t vector_length = remaining < 16 ? remaining : 16;
    v128_t source = sz_utf8_load_window_v128_(source_ptr, remaining);
    v128_t next = sz_utf8_slide1down_v128_(source, (vector_length < remaining) ? source_ptr[vector_length] : 0);
    v128_t previous = sz_utf8_slide1up_v128_(source);

    v128_t is_continuation = wasm_i8x16_eq(wasm_v128_and(source, wasm_i8x16_splat((sz_i8_t)0xC0)),
                                           wasm_i8x16_splat((sz_i8_t)0x80));
    v128_t is_d4 = wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xD4));
    v128_t is_d5 = wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xD5));
    v128_t is_d6 = wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xD6));
    v128_t after_d4 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xD4));
    v128_t after_d5 = wasm_i8x16_eq(previous, wasm_i8x16_splat((sz_i8_t)0xD5));
    v128_t is_non_ascii = wasm_u8x16_gt(source, wasm_i8x16_splat((sz_i8_t)0x7F));
    v128_t is_lead = wasm_v128_andnot(is_non_ascii, is_continuation);
    v128_t is_family = wasm_v128_or(wasm_v128_or(is_d4, is_d5), is_d6);
    v128_t is_foreign_lead = wasm_v128_andnot(is_lead, is_family);
    v128_t d4_stop = wasm_v128_and(is_d4, wasm_u8x16_lt(next, wasm_i8x16_splat((sz_i8_t)0xB1)));
    v128_t ligature_stop = wasm_v128_and(is_d6, wasm_i8x16_eq(next, wasm_i8x16_splat((sz_i8_t)0x87)));
    v128_t stop = wasm_v128_or(is_foreign_lead, wasm_v128_or(d4_stop, ligature_stop));

    v128_t folded = sz_ascii_fold_v128_(source);
    // Second-byte offsets (disjoint lanes).
    v128_t d4_second = wasm_v128_and(after_d4, wasm_u8x16_ge(source, wasm_i8x16_splat((sz_i8_t)0xB1)));
    v128_t d5_plus30 = wasm_v128_and(after_d5, sz_utf8_in_range_v128_(source, 0x80, 0x10));
    v128_t d5_minus10 = wasm_v128_and(after_d5, sz_utf8_in_range_v128_(source, 0x90, 0x07));
    folded = sz_utf8_masked_add_v128_(folded, wasm_v128_or(d4_second, d5_minus10), 0xF0); // -0x10
    folded = sz_utf8_masked_add_v128_(folded, d5_plus30, 0x30);
    // Lead +1 rewrites D4->D5 and D5->D6.
    v128_t promotes_d4 = wasm_v128_and(is_d4, wasm_u8x16_ge(next, wasm_i8x16_splat((sz_i8_t)0xB1)));
    v128_t promotes_d5 = wasm_v128_and(is_d5, sz_utf8_in_range_v128_(next, 0x90, 0x07));
    folded = sz_utf8_masked_add_v128_(folded, wasm_v128_or(promotes_d4, promotes_d5), 0x01);

    int first_stop = sz_utf8_first_set_v128_(stop, vector_length);
    return sz_utf8_strip_finish_v128_(source_ptr, vector_length, remaining, folded, first_stop, destination_ptr,
                                      needs_serial);
}

/** @brief Fold one window of Georgian (3-byte E1 82/83 sequences, uppercase keyed by the third byte). @sa RVV. */
SZ_INTERNAL sz_size_t sz_utf8_fold_georgian_strip_v128_(sz_u8_t const *source_ptr, sz_size_t remaining,
                                                        sz_u8_t *destination_ptr, int *needs_serial) {
    sz_size_t vector_length = remaining < 16 ? remaining : 16;
    v128_t source = sz_utf8_load_window_v128_(source_ptr, remaining);
    v128_t next = sz_utf8_slide1down_v128_(source, (vector_length < remaining) ? source_ptr[vector_length] : 0);
    v128_t next_next = sz_utf8_slide1down_v128_(next,
                                                (vector_length + 1 < remaining) ? source_ptr[vector_length + 1] : 0);

    v128_t is_continuation = wasm_i8x16_eq(wasm_v128_and(source, wasm_i8x16_splat((sz_i8_t)0xC0)),
                                           wasm_i8x16_splat((sz_i8_t)0x80));
    v128_t is_e1 = wasm_i8x16_eq(source, wasm_i8x16_splat((sz_i8_t)0xE1));
    v128_t is_non_ascii = wasm_u8x16_gt(source, wasm_i8x16_splat((sz_i8_t)0x7F));
    v128_t is_lead = wasm_v128_andnot(is_non_ascii, is_continuation);
    v128_t is_foreign_lead = wasm_v128_andnot(is_lead, is_e1);
    v128_t is_82_lead = wasm_v128_and(is_e1, wasm_i8x16_eq(next, wasm_i8x16_splat((sz_i8_t)0x82)));
    v128_t is_83_lead = wasm_v128_and(is_e1, wasm_i8x16_eq(next, wasm_i8x16_splat((sz_i8_t)0x83)));
    v128_t is_foreign_e1 = wasm_v128_andnot(is_e1, wasm_v128_or(is_82_lead, is_83_lead));
    v128_t stop = wasm_v128_or(is_foreign_lead, is_foreign_e1);

    // Uppercase keyed by the third byte: E1 82 third >= A0; E1 83 third in 80-85, or 87, or 8D.
    v128_t is_82_upper_lead = wasm_v128_and(is_82_lead, wasm_u8x16_ge(next_next, wasm_i8x16_splat((sz_i8_t)0xA0)));
    v128_t third_83_range = sz_utf8_in_range_v128_(next_next, 0x80, 0x06);
    v128_t third_83_extra = wasm_v128_or(wasm_i8x16_eq(next_next, wasm_i8x16_splat((sz_i8_t)0x87)),
                                         wasm_i8x16_eq(next_next, wasm_i8x16_splat((sz_i8_t)0x8D)));
    v128_t is_83_upper_lead = wasm_v128_and(is_83_lead, wasm_v128_or(third_83_range, third_83_extra));

    // Carry the per-window uppercase flag forward: lane +1 (second) and lane +2 (third).
    v128_t flag82_lead = wasm_v128_and(is_82_upper_lead, wasm_i8x16_splat(1));
    v128_t flag83_lead = wasm_v128_and(is_83_upper_lead, wasm_i8x16_splat(1));
    v128_t flag82_second = sz_utf8_slide1up_v128_(flag82_lead);
    v128_t flag83_second = sz_utf8_slide1up_v128_(flag83_lead);
    v128_t flag82_third = sz_utf8_slide1up_v128_(flag82_second);
    v128_t flag83_third = sz_utf8_slide1up_v128_(flag83_second);

    v128_t is_upper_lead = wasm_v128_or(is_82_upper_lead, is_83_upper_lead);
    v128_t is_upper_second = wasm_i8x16_ne(wasm_v128_or(flag82_second, flag83_second), wasm_i8x16_splat(0));
    v128_t is_82_upper_third = wasm_i8x16_ne(flag82_third, wasm_i8x16_splat(0));
    v128_t is_83_upper_third = wasm_i8x16_ne(flag83_third, wasm_i8x16_splat(0));

    v128_t folded = sz_ascii_fold_v128_(source);
    folded = wasm_v128_bitselect(wasm_i8x16_splat((sz_i8_t)0xE2), folded, is_upper_lead);   // lead E1 -> E2
    folded = wasm_v128_bitselect(wasm_i8x16_splat((sz_i8_t)0xB4), folded, is_upper_second); // second 82/83 -> B4
    folded = sz_utf8_masked_add_v128_(folded, is_82_upper_third, 0xE0);                     // third -0x20
    folded = sz_utf8_masked_add_v128_(folded, is_83_upper_third, 0x20);                     // third +0x20

    int first_stop = sz_utf8_first_set_v128_(stop, vector_length);
    return sz_utf8_strip_finish_v128_(source_ptr, vector_length, remaining, folded, first_stop, destination_ptr,
                                      needs_serial);
}

#pragma endregion // Per-script strip handlers

SZ_PUBLIC sz_size_t sz_utf8_case_fold_v128(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {

    sz_u8_t const *source_ptr = (sz_u8_t const *)source;
    sz_u8_t const *source_end = source_ptr + source_length;
    sz_u8_t *destination_ptr = (sz_u8_t *)destination;

    // Assumes valid UTF-8 input; use sz_utf8_valid() first if validation is needed.
    while (source_ptr < source_end) {
        // Fold a maximal ASCII run, 16 bytes per window, stopping at the first non-ASCII byte.
        sz_size_t remaining = (sz_size_t)(source_end - source_ptr);
        sz_size_t window = remaining < 16 ? remaining : 16;
        v128_t loaded = sz_utf8_load_window_v128_(source_ptr, remaining);
        int first_non_ascii = sz_utf8_first_set_v128_(loaded, window);
        sz_size_t ascii_run = first_non_ascii < 0 ? window : (sz_size_t)first_non_ascii;
        if (ascii_run) {
            sz_store_partial_v128_((sz_ptr_t)destination_ptr, sz_ascii_fold_v128_(loaded), ascii_run);
            source_ptr += ascii_run, destination_ptr += ascii_run;
        }
        if (source_ptr >= source_end) break;
        if (first_non_ascii < 0) continue; // a full ASCII window; fetch the next

        // Dispatch the non-ASCII region to the matching vectorized script handler by its lead byte.
        int needs_serial = 1;
        sz_size_t consumed = 0;
        remaining = (sz_size_t)(source_end - source_ptr);
        sz_u8_t lead = source_ptr[0];
        if (lead >= 0xC2 && lead <= 0xC6)
            consumed = sz_utf8_fold_latin_strip_v128_(source_ptr, remaining, destination_ptr, &needs_serial);
        else if (lead == 0xD0 || lead == 0xD1)
            consumed = sz_utf8_fold_cyrillic_strip_v128_(source_ptr, remaining, destination_ptr, &needs_serial);
        else if (lead == 0xCE || lead == 0xCF)
            consumed = sz_utf8_fold_greek_strip_v128_(source_ptr, remaining, destination_ptr, &needs_serial);
        else if (lead == 0xD4 || lead == 0xD5 || lead == 0xD6)
            consumed = sz_utf8_fold_armenian_strip_v128_(source_ptr, remaining, destination_ptr, &needs_serial);
        else if (lead == 0xE1)
            consumed = sz_utf8_fold_georgian_strip_v128_(source_ptr, remaining, destination_ptr, &needs_serial);
        source_ptr += consumed, destination_ptr += consumed;
        if (!needs_serial && consumed) continue; // a clean vectorized strip (or trailing-incomplete trim)

        // One codepoint the vector path can't fold in place: full serial decode / Unicode fold / encode.
        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse((sz_cptr_t)source_ptr, (sz_cptr_t)source_end, &rune, &rune_length);
        source_ptr += rune_length;
        sz_rune_t folded_runes[3]; // Unicode case folding produces at most 3 runes
        sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
        for (sz_size_t rune_index = 0; rune_index != folded_count; ++rune_index)
            destination_ptr += sz_rune_export(folded_runes[rune_index], destination_ptr);
    }

    return (sz_size_t)(destination_ptr - (sz_u8_t *)destination);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_CASE_FOLD_V128_H_
