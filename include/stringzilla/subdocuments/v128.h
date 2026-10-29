/**
 *  @file include/stringzilla/subdocuments/v128.h
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief WASM SIMD128 backend for path queries: the JSON block classifier and the XML byte search,
 *      under the serial walks.
 *
 *  A 64-byte block is four registers, each classified by one pair of nibble swizzles. Shifting a
 *  class bit to the top of its byte makes that class's mask one bitmask per register.
 *
 *  @sa c/stringzilla/subdocuments.h
 */
#ifndef STRINGZILLA_SUBDOCUMENTS_V128_H_
#define STRINGZILLA_SUBDOCUMENTS_V128_H_

#include "stringzilla/types.h"

#include "stringzilla/find/v128.h" // `sz_find_byte_v128_`
#include "stringzilla/subdocuments/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_WASM_
#if STRINGZILLA_ARCH_WASM_V128_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

#pragma region V128

/** One class bit of a 64-byte block held as four quarters of classes, the first quarter in the
 *  lowest bits: each quarter's bit moves to the top of its bytes for a bitmask. */
STRINGZILLA_INLINE sz_u64_t sz_subdocuments_class_v128_(v128_t first_u8x16, v128_t second_u8x16, v128_t third_u8x16,
                                                        v128_t fourth_u8x16, sz_u32_t bit) {
    sz_u64_t const first = wasm_i8x16_bitmask(wasm_i8x16_shl(first_u8x16, 7 - bit));
    sz_u64_t const second = wasm_i8x16_bitmask(wasm_i8x16_shl(second_u8x16, 7 - bit));
    sz_u64_t const third = wasm_i8x16_bitmask(wasm_i8x16_shl(third_u8x16, 7 - bit));
    sz_u64_t const fourth = wasm_i8x16_bitmask(wasm_i8x16_shl(fourth_u8x16, 7 - bit));
    return fourth << 48 | third << 32 | second << 16 | first;
}

/** The class bits of 16 bytes, one pair of nibble lookups. */
STRINGZILLA_INLINE v128_t sz_subdocuments_classes_v128_(sz_u8_t const *text) {
    v128_t const low_nibbles_u8x16 = wasm_v128_load(sz_subdocuments_low_nibble_classes_k);
    v128_t const high_nibbles_u8x16 = wasm_v128_load(sz_subdocuments_high_nibble_classes_k);
    v128_t const text_u8x16 = wasm_v128_load(text);
    return wasm_v128_and(wasm_i8x16_swizzle(low_nibbles_u8x16, wasm_v128_and(text_u8x16, wasm_i8x16_splat(0x0F))),
                         wasm_i8x16_swizzle(high_nibbles_u8x16, wasm_u8x16_shr(text_u8x16, 4)));
}

/** The classes of 16 bytes ORed with themselves shifted down by one, so bit 2 holds colons or
 *  commas and bit 6 spaces or other blanks; bytes shift alone, so no neighbour leaks in. */
STRINGZILLA_INLINE v128_t sz_subdocuments_pairs_v128_(v128_t classes_u8x16) {
    return wasm_v128_or(classes_u8x16, wasm_u8x16_shr(classes_u8x16, 1));
}

/** Classifies a block as four quarters, moving each class bit to the top of its byte for a
 *  bitmask. */
STRINGZILLA_INLINE sz_subdocuments_block_t sz_subdocuments_classify_v128_(sz_u8_t const *text,
                                                                          sz_subdocuments_carry_t *carry) {
    v128_t const first_u8x16 = sz_subdocuments_classes_v128_(text);
    v128_t const second_u8x16 = sz_subdocuments_classes_v128_(text + 16);
    v128_t const third_u8x16 = sz_subdocuments_classes_v128_(text + 32);
    v128_t const fourth_u8x16 = sz_subdocuments_classes_v128_(text + 48);
    v128_t const first_pairs_u8x16 = sz_subdocuments_pairs_v128_(first_u8x16);
    v128_t const second_pairs_u8x16 = sz_subdocuments_pairs_v128_(second_u8x16);
    v128_t const third_pairs_u8x16 = sz_subdocuments_pairs_v128_(third_u8x16);
    v128_t const fourth_pairs_u8x16 = sz_subdocuments_pairs_v128_(fourth_u8x16);
    sz_u64_t const opens = sz_subdocuments_class_v128_(first_u8x16, second_u8x16, third_u8x16, fourth_u8x16, 0);
    sz_u64_t const closes = sz_subdocuments_class_v128_(first_u8x16, second_u8x16, third_u8x16, fourth_u8x16, 1);
    sz_u64_t const separators = sz_subdocuments_class_v128_(first_pairs_u8x16, second_pairs_u8x16, third_pairs_u8x16,
                                                            fourth_pairs_u8x16, 2);
    sz_u64_t const quotes = sz_subdocuments_class_v128_(first_u8x16, second_u8x16, third_u8x16, fourth_u8x16, 4);
    sz_u64_t const backslashes = sz_subdocuments_class_v128_(first_u8x16, second_u8x16, third_u8x16, fourth_u8x16, 5);
    sz_u64_t const blanks = sz_subdocuments_class_v128_(first_pairs_u8x16, second_pairs_u8x16, third_pairs_u8x16,
                                                        fourth_pairs_u8x16, 6);
    return sz_subdocuments_block_settle_(quotes, backslashes, opens, closes, separators, blanks, carry);
}

#if STRINGZILLA_TARGET_V128

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_v128(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_subdocuments_engine_compile_(engine, paths, format, options, malformed_index, sz_cap_v128_k, allocator,
                                           stream);
}

STRINGZILLA_API sz_status_t sz_subdocuments_contains_v128(sz_subdocuments_engine_t *engine,
                                                          sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                          sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_contains_with_(engine, &sz_subdocuments_classify_v128_, &sz_subdocuments_classify_v128_,
                                          &sz_find_byte_v128_, documents, contained, malformed_index);
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts_v128(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *documents, sz_size_t *counts,
                                                        sz_size_t counts_stride, sz_size_t *malformed_index,
                                                        sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_counts_with_(engine, &sz_subdocuments_classify_v128_, &sz_subdocuments_classify_v128_,
                                        &sz_find_byte_v128_, documents, counts, counts_stride, malformed_index);
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_v128(sz_subdocuments_engine_t *engine,
                                                      sz_strings_view_t const *documents, sz_match_t *matches,
                                                      sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                      sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_find_with_(engine, &sz_subdocuments_classify_v128_, &sz_subdocuments_classify_v128_,
                                      &sz_find_byte_v128_, documents, matches, matches_capacity, matches_offsets,
                                      malformed_index);
}

#endif // STRINGZILLA_TARGET_V128

#pragma endregion V128

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // STRINGZILLA_ARCH_WASM_V128_
#endif // STRINGZILLA_ARCH_WASM_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_SUBDOCUMENTS_V128_H_
