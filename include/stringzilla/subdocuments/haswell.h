/**
 *  @file include/stringzilla/subdocuments/haswell.h
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief AVX2 (Haswell) backend for path queries: the JSON block classifier and the XML byte
 *      search, under the serial walks.
 *
 *  A 64-byte block is two registers, each classified by one pair of nibble lookups. Shifting a
 *  class bit to the top of its byte makes that class's mask one movemask per register.
 *
 *  @sa include/stringzilla/subdocuments.h
 */
#ifndef STRINGZILLA_SUBDOCUMENTS_HASWELL_H_
#define STRINGZILLA_SUBDOCUMENTS_HASWELL_H_

#include "stringzilla/types.h"

#include "stringzilla/find/haswell.h" // `sz_find_byte_haswell_`
#include "stringzilla/subdocuments/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_X8664_
#if STRINGZILLA_ARCH_X8664_HASWELL_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,lzcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "bmi", "bmi2", "lzcnt")
#endif

#pragma region Haswell

/** Classifies a block as two halves, moving each class bit to the top of its byte for a movemask; a
 *  16-bit shift by at most 7 leaves every byte's top bit its own. */
STRINGZILLA_INLINE sz_subdocuments_block_t sz_subdocuments_classify_haswell_(sz_u8_t const *text,
                                                                             sz_subdocuments_carry_t *carry) {
    __m256i const low_nibbles_u8x32 = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((__m128i const *)sz_subdocuments_low_nibble_classes_k));
    __m256i const high_nibbles_u8x32 = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((__m128i const *)sz_subdocuments_high_nibble_classes_k));
    __m256i const nibble_u8x32 = _mm256_set1_epi8(0x0F);
    sz_u64_t quotes = 0, backslashes = 0, opens = 0, closes = 0, separators = 0, blanks = 0;
    sz_size_t half;
    for (half = 0; half != 2; ++half) {
        __m256i const text_u8x32 = _mm256_loadu_si256((__m256i const *)(text + half * 32));
        __m256i const classes_u8x32 = _mm256_and_si256(
            _mm256_shuffle_epi8(low_nibbles_u8x32, _mm256_and_si256(text_u8x32, nibble_u8x32)),
            _mm256_shuffle_epi8(high_nibbles_u8x32, _mm256_and_si256(_mm256_srli_epi16(text_u8x32, 4), nibble_u8x32)));
        sz_size_t const shift = half * 32;
        opens |= (sz_u64_t)(sz_u32_t)_mm256_movemask_epi8(_mm256_slli_epi16(classes_u8x32, 7)) << shift;
        closes |= (sz_u64_t)(sz_u32_t)_mm256_movemask_epi8(_mm256_slli_epi16(classes_u8x32, 6)) << shift;
        separators |= (sz_u64_t)(sz_u32_t)_mm256_movemask_epi8(
                          _mm256_or_si256(_mm256_slli_epi16(classes_u8x32, 5), _mm256_slli_epi16(classes_u8x32, 4)))
                      << shift;
        quotes |= (sz_u64_t)(sz_u32_t)_mm256_movemask_epi8(_mm256_slli_epi16(classes_u8x32, 3)) << shift;
        backslashes |= (sz_u64_t)(sz_u32_t)_mm256_movemask_epi8(_mm256_slli_epi16(classes_u8x32, 2)) << shift;
        blanks |= (sz_u64_t)(sz_u32_t)_mm256_movemask_epi8(
                      _mm256_or_si256(_mm256_slli_epi16(classes_u8x32, 1), classes_u8x32))
                  << shift;
    }
    return sz_subdocuments_block_settle_(quotes, backslashes, opens, closes, separators, blanks, carry);
}

#if STRINGZILLA_TARGET_HASWELL

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_haswell(sz_subdocuments_engine_t *engine,
                                                                sz_strings_view_t const *paths,
                                                                sz_subdocuments_format_t format,
                                                                sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                                sz_stream_t stream) {
    return sz_subdocuments_engine_compile_(engine, paths, format, malformed_index, sz_cap_haswell_k, allocator, stream);
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts_haswell(sz_subdocuments_engine_t *engine,
                                                           sz_strings_view_t const *documents, sz_size_t *counts,
                                                           sz_size_t counts_stride, sz_size_t *malformed_index,
                                                           sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_counts_with_(engine, &sz_subdocuments_classify_haswell_, &sz_find_byte_haswell_, documents,
                                        counts, counts_stride, malformed_index);
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_haswell(sz_subdocuments_engine_t *engine,
                                                         sz_strings_view_t const *documents, sz_match_t *matches,
                                                         sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                         sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_find_with_(engine, &sz_subdocuments_classify_haswell_, &sz_find_byte_haswell_, documents,
                                      matches, matches_capacity, matches_offsets, malformed_index);
}

#endif // STRINGZILLA_TARGET_HASWELL

#pragma endregion Haswell

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_X8664_HASWELL_
#endif // STRINGZILLA_ARCH_X8664_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_SUBDOCUMENTS_HASWELL_H_
