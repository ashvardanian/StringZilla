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
 *  @sa include/stringzilla/subdocuments.h
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

/** Classifies a block in quarters, moving each class bit to the top of its byte for a movemask. */
STRINGZILLA_INLINE sz_subdocuments_block_t sz_subdocuments_classify_powervsx_(sz_u8_t const *text,
                                                                              sz_subdocuments_carry_t *carry) {
    __vector unsigned char const low_nibbles_u8x16 = vec_xl(0, sz_subdocuments_low_nibble_classes_k);
    __vector unsigned char const high_nibbles_u8x16 = vec_xl(0, sz_subdocuments_high_nibble_classes_k);
    __vector unsigned char const nibble_u8x16 = vec_splats((unsigned char)0x0F);
    __vector unsigned char const four_u8x16 = vec_splats((unsigned char)4);
    sz_u64_t quotes = 0, backslashes = 0, opens = 0, closes = 0, separators = 0, blanks = 0;
    sz_size_t part;
    for (part = 0; part != 4; ++part) {
        __vector unsigned char const text_u8x16 = vec_xl(0, text + part * 16);
        __vector unsigned char const classes_u8x16 = vec_and(
            vec_perm(low_nibbles_u8x16, low_nibbles_u8x16, vec_and(text_u8x16, nibble_u8x16)),
            vec_perm(high_nibbles_u8x16, high_nibbles_u8x16, vec_sr(text_u8x16, four_u8x16)));
        sz_size_t const shift = part * 16;
        opens |= sz_movemask_powervsx_(vec_sl(classes_u8x16, vec_splats((unsigned char)7))) << shift;
        closes |= sz_movemask_powervsx_(vec_sl(classes_u8x16, vec_splats((unsigned char)6))) << shift;
        separators |= sz_movemask_powervsx_(vec_or(vec_sl(classes_u8x16, vec_splats((unsigned char)5)),
                                                   vec_sl(classes_u8x16, four_u8x16)))
                      << shift;
        quotes |= sz_movemask_powervsx_(vec_sl(classes_u8x16, vec_splats((unsigned char)3))) << shift;
        backslashes |= sz_movemask_powervsx_(vec_sl(classes_u8x16, vec_splats((unsigned char)2))) << shift;
        blanks |= sz_movemask_powervsx_(vec_or(vec_sl(classes_u8x16, vec_splats((unsigned char)1)), classes_u8x16))
                  << shift;
    }
    return sz_subdocuments_block_settle_(quotes, backslashes, opens, closes, separators, blanks, carry);
}

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_powervsx(sz_subdocuments_engine_t *engine,
                                                                 sz_strings_view_t const *paths,
                                                                 sz_subdocuments_format_t format,
                                                                 sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                                 sz_stream_t stream) {
    return sz_subdocuments_engine_compile_(engine, paths, format, malformed_index, sz_cap_powervsx_k, allocator,
                                           stream);
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts_powervsx(sz_subdocuments_engine_t *engine,
                                                            sz_strings_view_t const *documents, sz_size_t *counts,
                                                            sz_size_t counts_stride, sz_size_t *malformed_index,
                                                            sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_counts_with_(engine, &sz_subdocuments_classify_powervsx_, &sz_find_byte_powervsx_, documents,
                                        counts, counts_stride, malformed_index);
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_powervsx(sz_subdocuments_engine_t *engine,
                                                          sz_strings_view_t const *documents, sz_match_t *matches,
                                                          sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                          sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_find_with_(engine, &sz_subdocuments_classify_powervsx_, &sz_find_byte_powervsx_, documents,
                                      matches, matches_capacity, matches_offsets, malformed_index);
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
