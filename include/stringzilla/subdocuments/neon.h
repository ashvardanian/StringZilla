/**
 *  @file include/stringzilla/subdocuments/neon.h
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief NEON backend for path queries: the JSON block classifier and the XML byte search,
 *      under the serial walks.
 *
 *  One pair of nibble lookups gives every byte a class byte with one bit per class, and one 8x8
 *  bit-plane transpose turns the class bytes of a whole 64-byte block into all eight class masks
 *  at once, where extracting a movemask per class would cost twice the instructions. XML documents
 *  need no classification, only the NEON byte search.
 *
 *  @sa c/stringzilla/subdocuments.h
 */
#ifndef STRINGZILLA_SUBDOCUMENTS_NEON_H_
#define STRINGZILLA_SUBDOCUMENTS_NEON_H_

#include "stringzilla/types.h"

#include "stringzilla/find/neon.h" // `sz_find_byte_neon_`
#include "stringzilla/subdocuments/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_ARM64_
#if STRINGZILLA_ARCH_ARM64_NEON_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

#pragma region NEON

/** The class bits of 16 bytes, one pair of nibble lookups. */
STRINGZILLA_INLINE uint8x16_t sz_subdocuments_classes_neon_(uint8x16_t text_u8x16) {
    uint8x16_t const low_nibbles_u8x16 = vld1q_u8(sz_subdocuments_low_nibble_classes_k);
    uint8x16_t const high_nibbles_u8x16 = vld1q_u8(sz_subdocuments_high_nibble_classes_k);
    return vandq_u8(vqtbl1q_u8(low_nibbles_u8x16, vandq_u8(text_u8x16, vdupq_n_u8(0x0F))),
                    vqtbl1q_u8(high_nibbles_u8x16, vshrq_n_u8(text_u8x16, 4)));
}

/** Two vectors of nibbles, each lane holding one class's bits of four bytes, as one class's mask
 *  in each half of the two 64-bit words: the low word's classes first, then the high word's. */
STRINGZILLA_INLINE void sz_subdocuments_masks_neon_(uint8x16_t first_u8x16, uint8x16_t second_u8x16,
                                                    uint64x2_t *low_u64x2, uint64x2_t *high_u64x2) {
    uint8x16_t const evens_u8x16 = vuzp1q_u8(first_u8x16, second_u8x16);
    uint8x16_t const odds_u8x16 = vuzp2q_u8(first_u8x16, second_u8x16);
    *low_u64x2 = vreinterpretq_u64_u8(vsliq_n_u8(evens_u8x16, odds_u8x16, 4));
    *high_u64x2 = vreinterpretq_u64_u8(vsriq_n_u8(odds_u8x16, evens_u8x16, 4));
}

/**
 *  @brief Classifies a block from a deinterleaved load: lane @c l of vector @c k is byte <tt>4l + k</tt>.
 *
 *  Two rounds of masked shifts between vectors gather each class's bit of the four vectors into one
 *  nibble per lane, and neighbouring lanes' nibbles are then whole bytes of that class's mask.
 */
STRINGZILLA_INLINE sz_subdocuments_block_t sz_subdocuments_classify_neon_(sz_u8_t const *text,
                                                                          sz_subdocuments_carry_t *carry) {
    uint8x16x4_t const text_u8x16x4 = vld4q_u8(text);
    uint8x16_t const first_u8x16 = sz_subdocuments_classes_neon_(text_u8x16x4.val[0]);
    uint8x16_t const second_u8x16 = sz_subdocuments_classes_neon_(text_u8x16x4.val[1]);
    uint8x16_t const third_u8x16 = sz_subdocuments_classes_neon_(text_u8x16x4.val[2]);
    uint8x16_t const fourth_u8x16 = sz_subdocuments_classes_neon_(text_u8x16x4.val[3]);

    // Pairs of vectors trade single bits, leaving even classes in one and odd classes in the other.
    uint8x16_t const low_evens_u8x16 = vbslq_u8(vdupq_n_u8(0x55), first_u8x16, vshlq_n_u8(second_u8x16, 1));
    uint8x16_t const low_odds_u8x16 = vbslq_u8(vdupq_n_u8(0xAA), second_u8x16, vshrq_n_u8(first_u8x16, 1));
    uint8x16_t const high_evens_u8x16 = vbslq_u8(vdupq_n_u8(0x55), third_u8x16, vshlq_n_u8(fourth_u8x16, 1));
    uint8x16_t const high_odds_u8x16 = vbslq_u8(vdupq_n_u8(0xAA), fourth_u8x16, vshrq_n_u8(third_u8x16, 1));
    // Then bit pairs, leaving the nibbles of classes 0 and 4, 2 and 6, 1 and 5, and 3 and 7.
    uint8x16_t const opens_quotes_u8x16 = vbslq_u8(vdupq_n_u8(0x33), low_evens_u8x16, vshlq_n_u8(high_evens_u8x16, 2));
    uint8x16_t const colons_spaces_u8x16 = vbslq_u8(vdupq_n_u8(0xCC), high_evens_u8x16, vshrq_n_u8(low_evens_u8x16, 2));
    uint8x16_t const closes_backslashes_u8x16 = vbslq_u8(vdupq_n_u8(0x33), low_odds_u8x16,
                                                         vshlq_n_u8(high_odds_u8x16, 2));
    uint8x16_t const commas_blanks_u8x16 = vbslq_u8(vdupq_n_u8(0xCC), high_odds_u8x16, vshrq_n_u8(low_odds_u8x16, 2));

    // Masks by class bit: opens, closes, colons, commas, quotes, backslashes, spaces, other blanks.
    uint64x2_t even_low_u64x2, even_high_u64x2, odd_low_u64x2, odd_high_u64x2;
    sz_subdocuments_masks_neon_(opens_quotes_u8x16, colons_spaces_u8x16, &even_low_u64x2, &even_high_u64x2);
    sz_subdocuments_masks_neon_(closes_backslashes_u8x16, commas_blanks_u8x16, &odd_low_u64x2, &odd_high_u64x2);
    sz_u64_t const opens = vgetq_lane_u64(even_low_u64x2, 0), colons = vgetq_lane_u64(even_low_u64x2, 1);
    sz_u64_t const closes = vgetq_lane_u64(odd_low_u64x2, 0), commas = vgetq_lane_u64(odd_low_u64x2, 1);
    sz_u64_t const quotes = vgetq_lane_u64(even_high_u64x2, 0), spaces = vgetq_lane_u64(even_high_u64x2, 1);
    sz_u64_t const backslashes = vgetq_lane_u64(odd_high_u64x2, 0), other_blanks = vgetq_lane_u64(odd_high_u64x2, 1);
    return sz_subdocuments_block_settle_(quotes, backslashes, opens, closes, colons | commas, spaces | other_blanks,
                                         carry);
}

#if STRINGZILLA_TARGET_NEON

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_neon(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_subdocuments_engine_compile_(engine, paths, format, options, malformed_index, sz_cap_neon_k, allocator,
                                           stream);
}

STRINGZILLA_API sz_status_t sz_subdocuments_contains_neon(sz_subdocuments_engine_t *engine,
                                                          sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                          sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_contains_with_(engine, &sz_subdocuments_classify_neon_, &sz_subdocuments_classify_neon_,
                                          &sz_find_byte_neon_, documents, contained, malformed_index);
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts_neon(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *documents, sz_size_t *counts,
                                                        sz_size_t counts_stride, sz_size_t *malformed_index,
                                                        sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_counts_with_(engine, &sz_subdocuments_classify_neon_, &sz_subdocuments_classify_neon_,
                                        &sz_find_byte_neon_, documents, counts, counts_stride, malformed_index);
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_neon(sz_subdocuments_engine_t *engine,
                                                      sz_strings_view_t const *documents, sz_match_t *matches,
                                                      sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                      sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_find_with_(engine, &sz_subdocuments_classify_neon_, &sz_subdocuments_classify_neon_,
                                      &sz_find_byte_neon_, documents, matches, matches_capacity, matches_offsets,
                                      malformed_index);
}

#endif // STRINGZILLA_TARGET_NEON

#pragma endregion NEON

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_ARM64_NEON_
#endif // STRINGZILLA_ARCH_ARM64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_SUBDOCUMENTS_NEON_H_
