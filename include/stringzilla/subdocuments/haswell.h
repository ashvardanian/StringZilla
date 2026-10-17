/**
 *  @file include/stringzilla/subdocuments/haswell.h
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief AVX2 (Haswell) backend for path queries: the JSON block classifier and the XML byte
 *      search, under the serial walks.
 *
 *  A 64-byte block is two registers, each classified by one pair of nibble lookups. Shifting a
 *  class bit to the top of its byte makes that class's mask one movemask per register, and the
 *  bytes inside strings are one carry-less multiply of the quotes by all ones.
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
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,lzcnt,pclmul"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "bmi", "bmi2", "lzcnt", "pclmul")
#endif

#pragma region Haswell

/** @ref sz_subdocuments_prefix_xor_serial_ as one carry-less multiply by all ones. */
STRINGZILLA_INLINE sz_u64_t sz_subdocuments_prefix_xor_haswell_(sz_u64_t bits) {
    return (sz_u64_t)_mm_cvtsi128_si64(
        _mm_clmulepi64_si128(_mm_cvtsi64_si128((long long)bits), _mm_set1_epi8((char)0xFF), 0));
}

/** The sign bits of a 64-byte block held as two halves, the lower half in the lower 32 bits. */
STRINGZILLA_INLINE sz_u64_t sz_subdocuments_movemask_haswell_(__m256i low_u8x32, __m256i high_u8x32) {
    sz_u32_t const low = (sz_u32_t)_mm256_movemask_epi8(low_u8x32), high = (sz_u32_t)_mm256_movemask_epi8(high_u8x32);
    return (sz_u64_t)high << 32 | low;
}

/** The class bits of 32 bytes, one pair of nibble lookups. */
STRINGZILLA_INLINE __m256i sz_subdocuments_classes_haswell_(__m256i text_u8x32) {
    __m256i const low_nibbles_u8x32 = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((__m128i const *)sz_subdocuments_low_nibble_classes_k));
    __m256i const high_nibbles_u8x32 = _mm256_broadcastsi128_si256(
        _mm_loadu_si128((__m128i const *)sz_subdocuments_high_nibble_classes_k));
    __m256i const nibble_u8x32 = _mm256_set1_epi8(0x0F);
    return _mm256_and_si256(
        _mm256_shuffle_epi8(low_nibbles_u8x32, _mm256_and_si256(text_u8x32, nibble_u8x32)),
        _mm256_shuffle_epi8(high_nibbles_u8x32, _mm256_and_si256(_mm256_srli_epi16(text_u8x32, 4), nibble_u8x32)));
}

/** Classifies a block, moving each class bit to the top of its byte for a movemask; a 16-bit shift
 *  by at most 7 leaves every byte's top bit its own. */
STRINGZILLA_INLINE sz_subdocuments_block_t sz_subdocuments_classify_haswell_(sz_u8_t const *text,
                                                                             sz_subdocuments_carry_t *carry) {
    __m256i const low_u8x32 = sz_subdocuments_classes_haswell_(_mm256_loadu_si256((__m256i const *)text));
    __m256i const high_u8x32 = sz_subdocuments_classes_haswell_(_mm256_loadu_si256((__m256i const *)(text + 32)));
    // Colons and commas are bits 2 and 3, spaces and other blanks 6 and 7, so ORing the classes
    // with themselves shifted down by one puts each pair in one bit; the 16-bit shift leaks a
    // byte's bit 0 into its lower neighbour's bit 7 only, which no pair reads.
    __m256i const low_pairs_u8x32 = _mm256_or_si256(low_u8x32, _mm256_srli_epi16(low_u8x32, 1));
    __m256i const high_pairs_u8x32 = _mm256_or_si256(high_u8x32, _mm256_srli_epi16(high_u8x32, 1));
    sz_u64_t const opens = sz_subdocuments_movemask_haswell_(_mm256_slli_epi16(low_u8x32, 7),
                                                             _mm256_slli_epi16(high_u8x32, 7));
    sz_u64_t const closes = sz_subdocuments_movemask_haswell_(_mm256_slli_epi16(low_u8x32, 6),
                                                              _mm256_slli_epi16(high_u8x32, 6));
    sz_u64_t const separators = sz_subdocuments_movemask_haswell_(_mm256_slli_epi16(low_pairs_u8x32, 5),
                                                                  _mm256_slli_epi16(high_pairs_u8x32, 5));
    sz_u64_t const quotes = sz_subdocuments_movemask_haswell_(_mm256_slli_epi16(low_u8x32, 3),
                                                              _mm256_slli_epi16(high_u8x32, 3));
    sz_u64_t const backslashes = sz_subdocuments_movemask_haswell_(_mm256_slli_epi16(low_u8x32, 2),
                                                                   _mm256_slli_epi16(high_u8x32, 2));
    sz_u64_t const blanks = sz_subdocuments_movemask_haswell_(_mm256_slli_epi16(low_pairs_u8x32, 1),
                                                              _mm256_slli_epi16(high_pairs_u8x32, 1));
    return sz_subdocuments_block_settle_with_(quotes, backslashes, opens, closes, separators, blanks,
                                              &sz_subdocuments_prefix_xor_haswell_, carry);
}

/** The brackets outside strings of a block a skip passes whole: five byte compares per half
 *  instead of the six classes. */
STRINGZILLA_INLINE sz_subdocuments_block_t sz_subdocuments_skim_haswell_(sz_u8_t const *text,
                                                                         sz_subdocuments_carry_t *carry) {
    __m256i const quote_u8x32 = _mm256_set1_epi8('"'), backslash_u8x32 = _mm256_set1_epi8('\\');
    __m256i const open_u8x32 = _mm256_set1_epi8('{'), close_u8x32 = _mm256_set1_epi8('}');
    __m256i const slash_u8x32 = _mm256_set1_epi8('/'), fold_u8x32 = _mm256_set1_epi8(0x20);
    __m256i const low_u8x32 = _mm256_loadu_si256((__m256i const *)text);
    __m256i const high_u8x32 = _mm256_loadu_si256((__m256i const *)(text + 32));
    // `[` and `]` differ from `{` and `}` only in the 0x20 bit, and no other byte folds there.
    __m256i const low_folded_u8x32 = _mm256_or_si256(low_u8x32, fold_u8x32);
    __m256i const high_folded_u8x32 = _mm256_or_si256(high_u8x32, fold_u8x32);
    sz_u64_t const quotes = sz_subdocuments_movemask_haswell_(_mm256_cmpeq_epi8(low_u8x32, quote_u8x32),
                                                              _mm256_cmpeq_epi8(high_u8x32, quote_u8x32));
    sz_u64_t const backslashes = sz_subdocuments_movemask_haswell_(_mm256_cmpeq_epi8(low_u8x32, backslash_u8x32),
                                                                   _mm256_cmpeq_epi8(high_u8x32, backslash_u8x32));
    sz_u64_t const opens = sz_subdocuments_movemask_haswell_(_mm256_cmpeq_epi8(low_folded_u8x32, open_u8x32),
                                                             _mm256_cmpeq_epi8(high_folded_u8x32, open_u8x32));
    sz_u64_t const closes = sz_subdocuments_movemask_haswell_(_mm256_cmpeq_epi8(low_folded_u8x32, close_u8x32),
                                                              _mm256_cmpeq_epi8(high_folded_u8x32, close_u8x32));
    sz_u64_t const slashes = sz_subdocuments_movemask_haswell_(_mm256_cmpeq_epi8(low_u8x32, slash_u8x32),
                                                               _mm256_cmpeq_epi8(high_u8x32, slash_u8x32));
    return sz_subdocuments_brackets_settle_with_(quotes, backslashes, opens, closes, slashes,
                                                 &sz_subdocuments_prefix_xor_haswell_, carry);
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
    return sz_subdocuments_counts_with_(engine, &sz_subdocuments_classify_haswell_, &sz_subdocuments_skim_haswell_,
                                        &sz_find_byte_haswell_, documents, counts, counts_stride, malformed_index);
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_haswell(sz_subdocuments_engine_t *engine,
                                                         sz_strings_view_t const *documents, sz_match_t *matches,
                                                         sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                         sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_find_with_(engine, &sz_subdocuments_classify_haswell_, &sz_subdocuments_skim_haswell_,
                                      &sz_find_byte_haswell_, documents, matches, matches_capacity, matches_offsets,
                                      malformed_index);
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
