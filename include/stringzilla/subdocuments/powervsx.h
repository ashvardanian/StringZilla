/**
 *  @file include/stringzilla/subdocuments/powervsx.h
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief Power VSX backend for path queries: the JSON block classifier and the XML byte search,
 *      under the serial walks.
 *
 *  A 64-byte block is four registers, each classified by one pair of nibble permutes. Shifting a
 *  class bit to the top of its byte makes that class's mask one bit gather per register.
 *
 *  @sa c/stringzilla/subdocuments.h
 */
#ifndef STRINGZILLA_SUBDOCUMENTS_POWERVSX_H_
#define STRINGZILLA_SUBDOCUMENTS_POWERVSX_H_

#include "stringzilla/types.h"

#include "stringzilla/find/powervsx.h" // `sz_find_byte_powervsx_`, `sz_movemask_powervsx_`
#include "stringzilla/subdocuments/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_PPC64_
#if STRINGZILLA_TARGET_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

#pragma region VSX

/** One class bit of a 64-byte block held as four quarters of classes, the first quarter in the
 *  lowest bits: each quarter's bit moves to the top of its bytes for a movemask. */
STRINGZILLA_INLINE sz_u64_t sz_subdocuments_class_powervsx_(__vector unsigned char first_u8x16,
                                                            __vector unsigned char second_u8x16,
                                                            __vector unsigned char third_u8x16,
                                                            __vector unsigned char fourth_u8x16, unsigned char bit) {
    __vector unsigned char const shift_u8x16 = vec_splats((unsigned char)(7 - bit));
    sz_u64_t const first = sz_movemask_powervsx_(vec_sl(first_u8x16, shift_u8x16));
    sz_u64_t const second = sz_movemask_powervsx_(vec_sl(second_u8x16, shift_u8x16));
    sz_u64_t const third = sz_movemask_powervsx_(vec_sl(third_u8x16, shift_u8x16));
    sz_u64_t const fourth = sz_movemask_powervsx_(vec_sl(fourth_u8x16, shift_u8x16));
    return fourth << 48 | third << 32 | second << 16 | first;
}

/** The class bits of 16 bytes, one pair of nibble lookups. */
STRINGZILLA_INLINE __vector unsigned char sz_subdocuments_classes_powervsx_(sz_u8_t const *text) {
    __vector unsigned char const low_nibbles_u8x16 = vec_xl(0, sz_subdocuments_low_nibble_classes_k);
    __vector unsigned char const high_nibbles_u8x16 = vec_xl(0, sz_subdocuments_high_nibble_classes_k);
    __vector unsigned char const text_u8x16 = vec_xl(0, text);
    return vec_and(vec_perm(low_nibbles_u8x16, low_nibbles_u8x16, vec_and(text_u8x16, vec_splats((unsigned char)0x0F))),
                   vec_perm(high_nibbles_u8x16, high_nibbles_u8x16, vec_sr(text_u8x16, vec_splats((unsigned char)4))));
}

/** The classes of 16 bytes ORed with themselves shifted down by one, so bit 2 holds colons or
 *  commas and bit 6 spaces or other blanks; bytes shift alone, so no neighbour leaks in. */
STRINGZILLA_INLINE __vector unsigned char sz_subdocuments_pairs_powervsx_(__vector unsigned char classes_u8x16) {
    return vec_or(classes_u8x16, vec_sr(classes_u8x16, vec_splats((unsigned char)1)));
}

/** Classifies a block as four quarters, moving each class bit to the top of its byte for a
 *  movemask. */
STRINGZILLA_INLINE sz_subdocuments_block_t sz_subdocuments_classify_powervsx_(sz_u8_t const *text,
                                                                              sz_subdocuments_carry_t *carry) {
    __vector unsigned char const first_u8x16 = sz_subdocuments_classes_powervsx_(text);
    __vector unsigned char const second_u8x16 = sz_subdocuments_classes_powervsx_(text + 16);
    __vector unsigned char const third_u8x16 = sz_subdocuments_classes_powervsx_(text + 32);
    __vector unsigned char const fourth_u8x16 = sz_subdocuments_classes_powervsx_(text + 48);
    __vector unsigned char const first_pairs_u8x16 = sz_subdocuments_pairs_powervsx_(first_u8x16);
    __vector unsigned char const second_pairs_u8x16 = sz_subdocuments_pairs_powervsx_(second_u8x16);
    __vector unsigned char const third_pairs_u8x16 = sz_subdocuments_pairs_powervsx_(third_u8x16);
    __vector unsigned char const fourth_pairs_u8x16 = sz_subdocuments_pairs_powervsx_(fourth_u8x16);
    sz_u64_t const opens = sz_subdocuments_class_powervsx_(first_u8x16, second_u8x16, third_u8x16, fourth_u8x16, 0);
    sz_u64_t const closes = sz_subdocuments_class_powervsx_(first_u8x16, second_u8x16, third_u8x16, fourth_u8x16, 1);
    sz_u64_t const separators = sz_subdocuments_class_powervsx_(first_pairs_u8x16, second_pairs_u8x16,
                                                                third_pairs_u8x16, fourth_pairs_u8x16, 2);
    sz_u64_t const quotes = sz_subdocuments_class_powervsx_(first_u8x16, second_u8x16, third_u8x16, fourth_u8x16, 4);
    sz_u64_t const backslashes = sz_subdocuments_class_powervsx_(first_u8x16, second_u8x16, third_u8x16, fourth_u8x16,
                                                                 5);
    sz_u64_t const blanks = sz_subdocuments_class_powervsx_(first_pairs_u8x16, second_pairs_u8x16, third_pairs_u8x16,
                                                            fourth_pairs_u8x16, 6);
    return sz_subdocuments_block_settle_(quotes, backslashes, opens, closes, separators, blanks, carry);
}

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_powervsx(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_subdocuments_engine_compile_(engine, paths, format, options, malformed_index, sz_cap_powervsx_k,
                                           allocator, stream);
}

STRINGZILLA_API sz_status_t sz_subdocuments_contains_powervsx(sz_subdocuments_engine_t *engine,
                                                              sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                              sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_contains_with_(engine, &sz_subdocuments_classify_powervsx_,
                                          &sz_subdocuments_classify_powervsx_, &sz_find_byte_powervsx_, documents,
                                          contained, malformed_index);
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts_powervsx(sz_subdocuments_engine_t *engine,
                                                            sz_strings_view_t const *documents, sz_size_t *counts,
                                                            sz_size_t counts_stride, sz_size_t *malformed_index,
                                                            sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_counts_with_(engine, &sz_subdocuments_classify_powervsx_,
                                        &sz_subdocuments_classify_powervsx_, &sz_find_byte_powervsx_, documents, counts,
                                        counts_stride, malformed_index);
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_powervsx(sz_subdocuments_engine_t *engine,
                                                          sz_strings_view_t const *documents, sz_match_t *matches,
                                                          sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                          sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_find_with_(engine, &sz_subdocuments_classify_powervsx_, &sz_subdocuments_classify_powervsx_,
                                      &sz_find_byte_powervsx_, documents, matches, matches_capacity, matches_offsets,
                                      malformed_index);
}

#pragma endregion VSX

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_POWERVSX
#endif // STRINGZILLA_ARCH_PPC64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_SUBDOCUMENTS_POWERVSX_H_
