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
 *  @sa include/stringzilla/subdocuments.h
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

/**
 *  @brief Classifies a block from a deinterleaved load: lane @c l of vector @c k is byte `4l + k`.
 *
 *  Two rounds of masked shifts between vectors gather each class's bit of the four vectors into one
 *  nibble per lane, and neighbouring lanes' nibbles are then whole bytes of that class's mask.
 */
STRINGZILLA_INLINE sz_subdocuments_block_t sz_subdocuments_classify_neon_(sz_u8_t const *text,
                                                                          sz_subdocuments_carry_t *carry) {
    uint8x16_t const low_nibbles_u8x16 = vld1q_u8(sz_subdocuments_low_nibble_classes_k);
    uint8x16_t const high_nibbles_u8x16 = vld1q_u8(sz_subdocuments_high_nibble_classes_k);
    uint8x16_t const nibble_u8x16 = vdupq_n_u8(0x0F);
    uint8x16x4_t const text_u8x16x4 = vld4q_u8(text);
    uint8x16_t classes_u8x16[4], even_u8x16[2], odd_u8x16[2], nibbles_u8x16[4];
    sz_u64_t masks[8];
    sz_size_t part;
    for (part = 0; part != 4; ++part)
        classes_u8x16[part] = vandq_u8(vqtbl1q_u8(low_nibbles_u8x16, vandq_u8(text_u8x16x4.val[part], nibble_u8x16)),
                                       vqtbl1q_u8(high_nibbles_u8x16, vshrq_n_u8(text_u8x16x4.val[part], 4)));

    // Pairs of vectors trade single bits, leaving even classes in one and odd classes in the other.
    for (part = 0; part != 2; ++part) {
        even_u8x16[part] = vbslq_u8(vdupq_n_u8(0x55), classes_u8x16[2 * part],
                                    vshlq_n_u8(classes_u8x16[2 * part + 1], 1));
        odd_u8x16[part] = vbslq_u8(vdupq_n_u8(0xAA), classes_u8x16[2 * part + 1],
                                   vshrq_n_u8(classes_u8x16[2 * part], 1));
    }
    // Then bit pairs, leaving the nibbles of classes 0 and 4, 2 and 6, 1 and 5, and 3 and 7.
    nibbles_u8x16[0] = vbslq_u8(vdupq_n_u8(0x33), even_u8x16[0], vshlq_n_u8(even_u8x16[1], 2));
    nibbles_u8x16[1] = vbslq_u8(vdupq_n_u8(0xCC), even_u8x16[1], vshrq_n_u8(even_u8x16[0], 2));
    nibbles_u8x16[2] = vbslq_u8(vdupq_n_u8(0x33), odd_u8x16[0], vshlq_n_u8(odd_u8x16[1], 2));
    nibbles_u8x16[3] = vbslq_u8(vdupq_n_u8(0xCC), odd_u8x16[1], vshrq_n_u8(odd_u8x16[0], 2));

    // Masks by class bit: opens, closes, colons, commas, quotes, backslashes, spaces, other blanks.
    for (part = 0; part != 2; ++part) {
        uint8x16_t const evens_u8x16 = vuzp1q_u8(nibbles_u8x16[2 * part], nibbles_u8x16[2 * part + 1]);
        uint8x16_t const odds_u8x16 = vuzp2q_u8(nibbles_u8x16[2 * part], nibbles_u8x16[2 * part + 1]);
        uint64x2_t const low_u64x2 = vreinterpretq_u64_u8(vsliq_n_u8(evens_u8x16, odds_u8x16, 4));
        uint64x2_t const high_u64x2 = vreinterpretq_u64_u8(vsriq_n_u8(odds_u8x16, evens_u8x16, 4));
        masks[part] = vgetq_lane_u64(low_u64x2, 0), masks[part + 2] = vgetq_lane_u64(low_u64x2, 1);
        masks[part + 4] = vgetq_lane_u64(high_u64x2, 0), masks[part + 6] = vgetq_lane_u64(high_u64x2, 1);
    }
    return sz_subdocuments_block_settle_(masks[4], masks[5], masks[0], masks[1], masks[2] | masks[3],
                                         masks[6] | masks[7], carry);
}

#if STRINGZILLA_TARGET_NEON

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_neon(sz_subdocuments_engine_t *engine,
                                                             sz_strings_view_t const *paths,
                                                             sz_subdocuments_format_t format,
                                                             sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                             sz_stream_t stream) {
    return sz_subdocuments_engine_compile_(engine, paths, format, malformed_index, sz_cap_neon_k, allocator, stream);
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts_neon(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *documents, sz_size_t *counts,
                                                        sz_size_t counts_stride, sz_size_t *malformed_index,
                                                        sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_counts_with_(engine, &sz_subdocuments_classify_neon_, &sz_find_byte_neon_, documents, counts,
                                        counts_stride, malformed_index);
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_neon(sz_subdocuments_engine_t *engine,
                                                      sz_strings_view_t const *documents, sz_match_t *matches,
                                                      sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                      sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_find_with_(engine, &sz_subdocuments_classify_neon_, &sz_find_byte_neon_, documents, matches,
                                      matches_capacity, matches_offsets, malformed_index);
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
