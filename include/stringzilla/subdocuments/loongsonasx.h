/**
 *  @file include/stringzilla/subdocuments/loongsonasx.h
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief LoongArch LASX backend for path queries: the JSON block classifier and the XML byte
 *      search, under the serial walks.
 *
 *  A 64-byte block is two registers, each classified by one pair of nibble lookups, which LASX
 *  takes within each 128-bit lane. Shifting a class bit to the top of its byte makes that class's
 *  mask one movemask per register.
 *
 *  @sa c/stringzilla/subdocuments.h
 */
#ifndef STRINGZILLA_SUBDOCUMENTS_LOONGSONASX_H_
#define STRINGZILLA_SUBDOCUMENTS_LOONGSONASX_H_

#include "stringzilla/types.h"

#include "stringzilla/find/loongsonasx.h" // `sz_find_byte_loongsonasx_`, `sz_xvmovemask_b_find_loongsonasx_`
#include "stringzilla/subdocuments/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_LOONGARCH64_
#if STRINGZILLA_TARGET_LOONGSONASX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("lasx"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("lasx")
#endif

#pragma region LASX

/** The sign bits of a 64-byte block held as two halves, the lower half in the lower 32 bits. */
STRINGZILLA_INLINE sz_u64_t sz_subdocuments_movemask_loongsonasx_(__m256i low_u8x32, __m256i high_u8x32) {
    sz_u64_t const low = sz_xvmovemask_b_find_loongsonasx_(low_u8x32);
    sz_u64_t const high = sz_xvmovemask_b_find_loongsonasx_(high_u8x32);
    return high << 32 | low;
}

/** The class bits of 32 bytes, one pair of nibble lookups within each 128-bit lane. */
STRINGZILLA_INLINE __m256i sz_subdocuments_classes_loongsonasx_(sz_u8_t const *text, __m256i low_nibbles_u8x32,
                                                                __m256i high_nibbles_u8x32) {
    __m256i const text_u8x32 = __lasx_xvld(text, 0);
    return __lasx_xvand_v(__lasx_xvshuf_b(low_nibbles_u8x32, low_nibbles_u8x32, __lasx_xvandi_b(text_u8x32, 0x0F)),
                          __lasx_xvshuf_b(high_nibbles_u8x32, high_nibbles_u8x32, __lasx_xvsrli_b(text_u8x32, 4)));
}

/** Classifies a block as two halves, moving each class bit to the top of its byte for a
 *  movemask. */
STRINGZILLA_INLINE sz_subdocuments_block_t sz_subdocuments_classify_loongsonasx_(sz_u8_t const *text,
                                                                                 sz_subdocuments_carry_t *carry) {
    // Each table fills both 128-bit lanes, as `__lasx_xvshuf_b` looks up within a lane.
    sz_u8_t tables[64];
    sz_size_t index;
    for (index = 0; index != 16; ++index)
        tables[index] = tables[index + 16] = sz_subdocuments_low_nibble_classes_k[index],
        tables[index + 32] = tables[index + 48] = sz_subdocuments_high_nibble_classes_k[index];
    __m256i const low_nibbles_u8x32 = __lasx_xvld(tables, 0);
    __m256i const high_nibbles_u8x32 = __lasx_xvld(tables, 32);
    __m256i const low_u8x32 = sz_subdocuments_classes_loongsonasx_(text, low_nibbles_u8x32, high_nibbles_u8x32);
    __m256i const high_u8x32 = sz_subdocuments_classes_loongsonasx_(text + 32, low_nibbles_u8x32, high_nibbles_u8x32);
    // Bit 2 then holds colons or commas and bit 6 spaces or other blanks; bytes shift alone.
    __m256i const low_pairs_u8x32 = __lasx_xvor_v(low_u8x32, __lasx_xvsrli_b(low_u8x32, 1));
    __m256i const high_pairs_u8x32 = __lasx_xvor_v(high_u8x32, __lasx_xvsrli_b(high_u8x32, 1));
    sz_u64_t const opens = sz_subdocuments_movemask_loongsonasx_(__lasx_xvslli_b(low_u8x32, 7),
                                                                 __lasx_xvslli_b(high_u8x32, 7));
    sz_u64_t const closes = sz_subdocuments_movemask_loongsonasx_(__lasx_xvslli_b(low_u8x32, 6),
                                                                  __lasx_xvslli_b(high_u8x32, 6));
    sz_u64_t const separators = sz_subdocuments_movemask_loongsonasx_(__lasx_xvslli_b(low_pairs_u8x32, 5),
                                                                      __lasx_xvslli_b(high_pairs_u8x32, 5));
    sz_u64_t const quotes = sz_subdocuments_movemask_loongsonasx_(__lasx_xvslli_b(low_u8x32, 3),
                                                                  __lasx_xvslli_b(high_u8x32, 3));
    sz_u64_t const backslashes = sz_subdocuments_movemask_loongsonasx_(__lasx_xvslli_b(low_u8x32, 2),
                                                                       __lasx_xvslli_b(high_u8x32, 2));
    sz_u64_t const blanks = sz_subdocuments_movemask_loongsonasx_(__lasx_xvslli_b(low_pairs_u8x32, 1),
                                                                  __lasx_xvslli_b(high_pairs_u8x32, 1));
    return sz_subdocuments_block_settle_(quotes, backslashes, opens, closes, separators, blanks, carry);
}

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_loongsonasx(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_subdocuments_engine_compile_(engine, paths, format, options, malformed_index, sz_cap_loongsonasx_k,
                                           allocator, stream);
}

STRINGZILLA_API sz_status_t sz_subdocuments_contains_loongsonasx(sz_subdocuments_engine_t *engine,
                                                                 sz_strings_view_t const *documents,
                                                                 sz_u1x8_t *contained, sz_size_t *malformed_index,
                                                                 sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_contains_with_(engine, &sz_subdocuments_classify_loongsonasx_,
                                          &sz_subdocuments_classify_loongsonasx_, &sz_find_byte_loongsonasx_, documents,
                                          contained, malformed_index);
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts_loongsonasx(sz_subdocuments_engine_t *engine,
                                                               sz_strings_view_t const *documents, sz_size_t *counts,
                                                               sz_size_t counts_stride, sz_size_t *malformed_index,
                                                               sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_counts_with_(engine, &sz_subdocuments_classify_loongsonasx_,
                                        &sz_subdocuments_classify_loongsonasx_, &sz_find_byte_loongsonasx_, documents,
                                        counts, counts_stride, malformed_index);
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_loongsonasx(sz_subdocuments_engine_t *engine,
                                                             sz_strings_view_t const *documents, sz_match_t *matches,
                                                             sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                             sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_find_with_(engine, &sz_subdocuments_classify_loongsonasx_,
                                      &sz_subdocuments_classify_loongsonasx_, &sz_find_byte_loongsonasx_, documents,
                                      matches, matches_capacity, matches_offsets, malformed_index);
}

#pragma endregion LASX

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_LOONGSONASX
#endif // STRINGZILLA_ARCH_LOONGARCH64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_SUBDOCUMENTS_LOONGSONASX_H_
