/**
 *  @file include/stringzilla/subdocuments/sve2.h
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief SVE2 backend for path queries: the JSON block classifier and the XML byte search,
 *      under the serial walks.
 *
 *  Each class is one compare or one MATCH against its set of bytes, so no lookup table is loaded. A
 *  predicate becomes its bits of the block mask by one carry-free multiply per 64-bit lane, and
 *  each mask by one OR reduction once the block's vectors are done, at any vector length.
 *
 *  @sa include/stringzilla/subdocuments.h
 */
#ifndef STRINGZILLA_SUBDOCUMENTS_SVE2_H_
#define STRINGZILLA_SUBDOCUMENTS_SVE2_H_

#include "stringzilla/types.h"

#include "stringzilla/find/sve.h" // `sz_find_byte_sve_`
#include "stringzilla/subdocuments/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_ARM64_
#if STRINGZILLA_ARCH_ARM64_SVE2_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve+sve2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve+sve2")
#endif

#pragma region SVE2

/** Places the lanes of @p bytes_b8x as bits of a block mask, each 64-bit lane's eight
 *  at its shift in @p shifts_u64x; flag bytes of 0 or 1 times this constant land in the
 *  top byte without carries. */
STRINGZILLA_INLINE svuint64_t sz_subdocuments_place_sve2_(svbool_t bytes_b8x, svuint64_t shifts_u64x) {
    svbool_t const all_b64x = svptrue_b64();
    svuint64_t const flags_u64x = svreinterpret_u64_u8(svdup_n_u8_z(bytes_b8x, 1));
    svuint64_t const packed_u64x = svlsr_n_u64_x(all_b64x, svmul_n_u64_x(all_b64x, flags_u64x, 0x0102040810204080ull),
                                                 56);
    return svlsl_u64_x(all_b64x, packed_u64x, shifts_u64x);
}

/** Classifies a block one vector at a time, so the loop runs once at 512 bits or more; lanes past
 *  the block load nothing, and their shifts past 63 place nothing. */
STRINGZILLA_OUTLINED_ sz_subdocuments_block_t sz_subdocuments_classify_sve2_(sz_u8_t const *text,
                                                                             sz_subdocuments_carry_t *carry) {
    svbool_t const all_b64x = svptrue_b64();
    // MATCH tests each byte against its 128-bit segment's set, so short sets repeat to fill one.
    // `/` joins both the separators and the blanks, as the class tables give it both bits.
    svuint8_t const opens_set_u8x = svreinterpret_u8_u16(svdup_n_u16(0x5B7B));
    svuint8_t const closes_set_u8x = svreinterpret_u8_u16(svdup_n_u16(0x5D7D));
    svuint8_t const separators_set_u8x = svreinterpret_u8_u32(svdup_n_u32(0x2F2F2C3A));
    svuint8_t const blanks_set_u8x = svreinterpret_u8_u64(svdup_n_u64(0x2F2F2F2F0D0A0920ull));
    svuint64_t quotes_u64x = svdup_n_u64(0), backslashes_u64x = svdup_n_u64(0), opens_u64x = svdup_n_u64(0),
               closes_u64x = svdup_n_u64(0), separators_u64x = svdup_n_u64(0), blanks_u64x = svdup_n_u64(0);
    sz_size_t offset;
    for (offset = 0; offset < 64; offset += svcntb()) {
        svbool_t const loaded_b8x = svwhilelt_b8_u64(offset, 64);
        svuint8_t const text_u8x = svld1_u8(loaded_b8x, text + offset);
        svuint64_t const shifts_u64x = svindex_u64(offset, 8);
        quotes_u64x = svorr_u64_x(all_b64x, quotes_u64x,
                                  sz_subdocuments_place_sve2_(svcmpeq_n_u8(loaded_b8x, text_u8x, '"'), shifts_u64x));
        backslashes_u64x = svorr_u64_x(
            all_b64x, backslashes_u64x,
            sz_subdocuments_place_sve2_(svcmpeq_n_u8(loaded_b8x, text_u8x, '\\'), shifts_u64x));
        opens_u64x = svorr_u64_x(
            all_b64x, opens_u64x,
            sz_subdocuments_place_sve2_(svmatch_u8(loaded_b8x, text_u8x, opens_set_u8x), shifts_u64x));
        closes_u64x = svorr_u64_x(
            all_b64x, closes_u64x,
            sz_subdocuments_place_sve2_(svmatch_u8(loaded_b8x, text_u8x, closes_set_u8x), shifts_u64x));
        separators_u64x = svorr_u64_x(
            all_b64x, separators_u64x,
            sz_subdocuments_place_sve2_(svmatch_u8(loaded_b8x, text_u8x, separators_set_u8x), shifts_u64x));
        blanks_u64x = svorr_u64_x(
            all_b64x, blanks_u64x,
            sz_subdocuments_place_sve2_(svmatch_u8(loaded_b8x, text_u8x, blanks_set_u8x), shifts_u64x));
    }
    return sz_subdocuments_block_settle_(svorv_u64(all_b64x, quotes_u64x), svorv_u64(all_b64x, backslashes_u64x),
                                         svorv_u64(all_b64x, opens_u64x), svorv_u64(all_b64x, closes_u64x),
                                         svorv_u64(all_b64x, separators_u64x), svorv_u64(all_b64x, blanks_u64x), carry);
}

#if STRINGZILLA_TARGET_SVE2

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_sve2(sz_subdocuments_engine_t *engine,
                                                             sz_strings_view_t const *paths,
                                                             sz_subdocuments_format_t format,
                                                             sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                             sz_stream_t stream) {
    return sz_subdocuments_engine_compile_(engine, paths, format, malformed_index, sz_cap_sve2_k, allocator, stream);
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts_sve2(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *documents, sz_size_t *counts,
                                                        sz_size_t counts_stride, sz_size_t *malformed_index,
                                                        sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_counts_with_(engine, &sz_subdocuments_classify_sve2_, &sz_subdocuments_classify_sve2_,
                                        &sz_find_byte_sve_, documents, counts, counts_stride, malformed_index);
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_sve2(sz_subdocuments_engine_t *engine,
                                                      sz_strings_view_t const *documents, sz_match_t *matches,
                                                      sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                      sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_find_with_(engine, &sz_subdocuments_classify_sve2_, &sz_subdocuments_classify_sve2_,
                                      &sz_find_byte_sve_, documents, matches, matches_capacity, matches_offsets,
                                      malformed_index);
}

#endif // STRINGZILLA_TARGET_SVE2

#pragma endregion SVE2

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_ARM64_SVE2_
#endif // STRINGZILLA_ARCH_ARM64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_SUBDOCUMENTS_SVE2_H_
