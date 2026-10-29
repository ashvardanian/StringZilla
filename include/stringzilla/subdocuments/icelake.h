/**
 *  @file include/stringzilla/subdocuments/icelake.h
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief AVX-512 (Ice Lake) backend for path queries: the JSON block classifier and the XML byte
 *      search, under the serial walks.
 *
 *  A 64-byte block is one register, so one pair of nibble lookups classifies all of it and each
 *  class mask is a single byte test into a mask register.
 *
 *  @sa c/stringzilla/subdocuments.h
 */
#ifndef STRINGZILLA_SUBDOCUMENTS_ICELAKE_H_
#define STRINGZILLA_SUBDOCUMENTS_ICELAKE_H_

#include "stringzilla/types.h"

#include "stringzilla/find/skylake.h" // `sz_find_byte_skylake_`
#include "stringzilla/subdocuments/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_X8664_
#if STRINGZILLA_TARGET_ICELAKE
#if defined(__clang__) && STRINGZILLA_HAS_CLANG_EVEX512_
#pragma clang attribute push(                                                                                        \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,lzcnt,evex512"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(                                                                                \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,lzcnt"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2", \
                   "lzcnt")
#endif

#pragma region Ice Lake

/** Classifies a block with one pair of nibble lookups, then tests each class bit into its mask. */
STRINGZILLA_INLINE sz_subdocuments_block_t sz_subdocuments_classify_icelake_(sz_u8_t const *text,
                                                                             sz_subdocuments_carry_t *carry) {
    __m512i const low_nibbles_u8x64 = _mm512_broadcast_i32x4(
        _mm_loadu_si128((__m128i const *)sz_subdocuments_low_nibble_classes_k));
    __m512i const high_nibbles_u8x64 = _mm512_broadcast_i32x4(
        _mm_loadu_si128((__m128i const *)sz_subdocuments_high_nibble_classes_k));
    __m512i const nibble_u8x64 = _mm512_set1_epi8(0x0F);
    __m512i const text_u8x64 = _mm512_loadu_si512(text);
    __m512i const classes_u8x64 = _mm512_and_si512(
        _mm512_shuffle_epi8(low_nibbles_u8x64, _mm512_and_si512(text_u8x64, nibble_u8x64)),
        _mm512_shuffle_epi8(high_nibbles_u8x64, _mm512_and_si512(_mm512_srli_epi16(text_u8x64, 4), nibble_u8x64)));
    return sz_subdocuments_block_settle_(_mm512_test_epi8_mask(classes_u8x64, _mm512_set1_epi8(0x10)),
                                         _mm512_test_epi8_mask(classes_u8x64, _mm512_set1_epi8(0x20)),
                                         _mm512_test_epi8_mask(classes_u8x64, _mm512_set1_epi8(0x01)),
                                         _mm512_test_epi8_mask(classes_u8x64, _mm512_set1_epi8(0x02)),
                                         _mm512_test_epi8_mask(classes_u8x64, _mm512_set1_epi8(0x0C)),
                                         _mm512_test_epi8_mask(classes_u8x64, _mm512_set1_epi8((char)0xC0)), carry);
}

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_icelake(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_subdocuments_engine_compile_(engine, paths, format, options, malformed_index, sz_cap_icelake_k, allocator,
                                           stream);
}

STRINGZILLA_API sz_status_t sz_subdocuments_contains_icelake(sz_subdocuments_engine_t *engine,
                                                             sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                             sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_contains_with_(engine, &sz_subdocuments_classify_icelake_,
                                          &sz_subdocuments_classify_icelake_, &sz_find_byte_skylake_, documents,
                                          contained, malformed_index);
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts_icelake(sz_subdocuments_engine_t *engine,
                                                           sz_strings_view_t const *documents, sz_size_t *counts,
                                                           sz_size_t counts_stride, sz_size_t *malformed_index,
                                                           sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_counts_with_(engine, &sz_subdocuments_classify_icelake_, &sz_subdocuments_classify_icelake_,
                                        &sz_find_byte_skylake_, documents, counts, counts_stride, malformed_index);
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_icelake(sz_subdocuments_engine_t *engine,
                                                         sz_strings_view_t const *documents, sz_match_t *matches,
                                                         sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                         sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_find_with_(engine, &sz_subdocuments_classify_icelake_, &sz_subdocuments_classify_icelake_,
                                      &sz_find_byte_skylake_, documents, matches, matches_capacity, matches_offsets,
                                      malformed_index);
}

#pragma endregion Ice Lake

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_ICELAKE
#endif // STRINGZILLA_ARCH_X8664_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_SUBDOCUMENTS_ICELAKE_H_
