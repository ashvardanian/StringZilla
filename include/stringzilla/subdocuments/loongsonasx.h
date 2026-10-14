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
 *  @sa include/stringzilla/subdocuments.h
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

/** Classifies a block in halves, moving each class bit to the top of its byte for a movemask. */
STRINGZILLA_INLINE sz_subdocuments_block_t sz_subdocuments_classify_loongsonasx_(sz_u8_t const *text,
                                                                                 sz_subdocuments_carry_t *carry) {
    // Each table fills both 128-bit lanes, as `__lasx_xvshuf_b` looks up within a lane.
    sz_u8_t tables[64];
    sz_u64_t quotes = 0, backslashes = 0, opens = 0, closes = 0, separators = 0, blanks = 0;
    sz_size_t index, half;
    for (index = 0; index != 16; ++index)
        tables[index] = tables[index + 16] = sz_subdocuments_low_nibble_classes_k[index],
        tables[index + 32] = tables[index + 48] = sz_subdocuments_high_nibble_classes_k[index];
    __m256i const low_nibbles_u8x32 = __lasx_xvld(tables, 0);
    __m256i const high_nibbles_u8x32 = __lasx_xvld(tables, 32);
    for (half = 0; half != 2; ++half) {
        __m256i const text_u8x32 = __lasx_xvld(text + half * 32, 0);
        __m256i const classes_u8x32 = __lasx_xvand_v(
            __lasx_xvshuf_b(low_nibbles_u8x32, low_nibbles_u8x32, __lasx_xvandi_b(text_u8x32, 0x0F)),
            __lasx_xvshuf_b(high_nibbles_u8x32, high_nibbles_u8x32, __lasx_xvsrli_b(text_u8x32, 4)));
        sz_size_t const shift = half * 32;
        opens |= (sz_u64_t)sz_xvmovemask_b_find_loongsonasx_(__lasx_xvslli_b(classes_u8x32, 7)) << shift;
        closes |= (sz_u64_t)sz_xvmovemask_b_find_loongsonasx_(__lasx_xvslli_b(classes_u8x32, 6)) << shift;
        separators |= (sz_u64_t)sz_xvmovemask_b_find_loongsonasx_(
                          __lasx_xvor_v(__lasx_xvslli_b(classes_u8x32, 5), __lasx_xvslli_b(classes_u8x32, 4)))
                      << shift;
        quotes |= (sz_u64_t)sz_xvmovemask_b_find_loongsonasx_(__lasx_xvslli_b(classes_u8x32, 3)) << shift;
        backslashes |= (sz_u64_t)sz_xvmovemask_b_find_loongsonasx_(__lasx_xvslli_b(classes_u8x32, 2)) << shift;
        blanks |=
            (sz_u64_t)sz_xvmovemask_b_find_loongsonasx_(__lasx_xvor_v(__lasx_xvslli_b(classes_u8x32, 1), classes_u8x32))
            << shift;
    }
    return sz_subdocuments_block_settle_(quotes, backslashes, opens, closes, separators, blanks, carry);
}

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_loongsonasx(sz_subdocuments_engine_t *engine,
                                                                    sz_strings_view_t const *paths,
                                                                    sz_subdocuments_format_t format,
                                                                    sz_size_t *malformed_index,
                                                                    sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_subdocuments_engine_compile_(engine, paths, format, malformed_index, sz_cap_loongsonasx_k, allocator,
                                           stream);
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts_loongsonasx(sz_subdocuments_engine_t *engine,
                                                               sz_strings_view_t const *documents, sz_size_t *counts,
                                                               sz_size_t counts_stride, sz_size_t *malformed_index,
                                                               sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_counts_with_(engine, &sz_subdocuments_classify_loongsonasx_, &sz_find_byte_loongsonasx_,
                                        documents, counts, counts_stride, malformed_index);
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_loongsonasx(sz_subdocuments_engine_t *engine,
                                                             sz_strings_view_t const *documents, sz_match_t *matches,
                                                             sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                             sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_find_with_(engine, &sz_subdocuments_classify_loongsonasx_, &sz_find_byte_loongsonasx_,
                                      documents, matches, matches_capacity, matches_offsets, malformed_index);
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
