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
 *  @sa include/stringzilla/subdocuments.h
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

/** Classifies a block in quarters, moving each class bit to the top of its byte for a bitmask. */
STRINGZILLA_INLINE sz_subdocuments_block_t sz_subdocuments_classify_v128_(sz_u8_t const *text,
                                                                          sz_subdocuments_carry_t *carry) {
    v128_t const low_nibbles_u8x16 = wasm_v128_load(sz_subdocuments_low_nibble_classes_k);
    v128_t const high_nibbles_u8x16 = wasm_v128_load(sz_subdocuments_high_nibble_classes_k);
    v128_t const nibble_u8x16 = wasm_i8x16_splat(0x0F);
    sz_u64_t quotes = 0, backslashes = 0, opens = 0, closes = 0, separators = 0, blanks = 0;
    sz_size_t part;
    for (part = 0; part != 4; ++part) {
        v128_t const text_u8x16 = wasm_v128_load(text + part * 16);
        v128_t const classes_u8x16 = wasm_v128_and(
            wasm_i8x16_swizzle(low_nibbles_u8x16, wasm_v128_and(text_u8x16, nibble_u8x16)),
            wasm_i8x16_swizzle(high_nibbles_u8x16, wasm_u8x16_shr(text_u8x16, 4)));
        sz_size_t const shift = part * 16;
        opens |= (sz_u64_t)wasm_i8x16_bitmask(wasm_i8x16_shl(classes_u8x16, 7)) << shift;
        closes |= (sz_u64_t)wasm_i8x16_bitmask(wasm_i8x16_shl(classes_u8x16, 6)) << shift;
        separators |= (sz_u64_t)wasm_i8x16_bitmask(
                          wasm_v128_or(wasm_i8x16_shl(classes_u8x16, 5), wasm_i8x16_shl(classes_u8x16, 4)))
                      << shift;
        quotes |= (sz_u64_t)wasm_i8x16_bitmask(wasm_i8x16_shl(classes_u8x16, 3)) << shift;
        backslashes |= (sz_u64_t)wasm_i8x16_bitmask(wasm_i8x16_shl(classes_u8x16, 2)) << shift;
        blanks |= (sz_u64_t)wasm_i8x16_bitmask(wasm_v128_or(wasm_i8x16_shl(classes_u8x16, 1), classes_u8x16)) << shift;
    }
    return sz_subdocuments_block_settle_(quotes, backslashes, opens, closes, separators, blanks, carry);
}

#if STRINGZILLA_TARGET_V128

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_v128(sz_subdocuments_engine_t *engine,
                                                             sz_strings_view_t const *paths,
                                                             sz_subdocuments_format_t format,
                                                             sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                             sz_stream_t stream) {
    return sz_subdocuments_engine_compile_(engine, paths, format, malformed_index, sz_cap_v128_k, allocator, stream);
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts_v128(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *documents, sz_size_t *counts,
                                                        sz_size_t counts_stride, sz_size_t *malformed_index,
                                                        sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_counts_with_(engine, &sz_subdocuments_classify_v128_, &sz_find_byte_v128_, documents, counts,
                                        counts_stride, malformed_index);
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_v128(sz_subdocuments_engine_t *engine,
                                                      sz_strings_view_t const *documents, sz_match_t *matches,
                                                      sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                      sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_find_with_(engine, &sz_subdocuments_classify_v128_, &sz_find_byte_v128_, documents, matches,
                                      matches_capacity, matches_offsets, malformed_index);
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
