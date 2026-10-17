/**
 *  @file include/stringzilla/subdocuments/rvv.h
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief RVV backend for path queries: the JSON block classifier and the XML byte search,
 *      under the serial walks.
 *
 *  A 64-byte block is one group of four registers at any vector length the V extension allows,
 *  and each class is a compare or two into a mask register, which reads out as the class's mask
 *  in one move. A nibble lookup would cost a permutation across the group, which grows with the
 *  square of its size.
 *
 *  @sa include/stringzilla/subdocuments.h
 */
#ifndef STRINGZILLA_SUBDOCUMENTS_RVV_H_
#define STRINGZILLA_SUBDOCUMENTS_RVV_H_

#include "stringzilla/types.h"

#include "stringzilla/find/rvv.h" // `sz_find_byte_rvv_`
#include "stringzilla/subdocuments/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_RISCV64_
#if STRINGZILLA_ARCH_RISCV64_RVV_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#pragma region RVV

/** The 64 lanes of @p lanes_b2 as one word, read from the mask register's low 64 bits. */
STRINGZILLA_INLINE sz_u64_t sz_subdocuments_mask_bits_rvv_(vbool2_t lanes_b2) {
    return __riscv_vmv_x_s_u64m1_u64(__riscv_vreinterpret_v_u8m1_u64m1(__riscv_vreinterpret_v_b2_u8m1(lanes_b2)));
}

/** Classifies a block at `vl = 64`, so all 64 bits of every mask are defined. */
STRINGZILLA_INLINE sz_subdocuments_block_t sz_subdocuments_classify_rvv_(sz_u8_t const *text,
                                                                         sz_subdocuments_carry_t *carry) {
    sz_size_t const vector_length = __riscv_vsetvl_e8m4(64);
    vuint8m4_t const text_u8m4 = __riscv_vle8_v_u8m4(text, vector_length);
    // `[` and `]` differ from `{` and `}` only in the 0x20 bit, and no other byte folds there.
    vuint8m4_t const folded_u8m4 = __riscv_vor_vx_u8m4(text_u8m4, 0x20, vector_length);
    // `/` joins both the separators and the blanks, as the class tables give it both bits.
    vbool2_t const slashes_b2 = __riscv_vmseq_vx_u8m4_b2(text_u8m4, '/', vector_length);
    vbool2_t const separators_b2 = __riscv_vmor_mm_b2(
        __riscv_vmor_mm_b2(__riscv_vmseq_vx_u8m4_b2(text_u8m4, ':', vector_length),
                           __riscv_vmseq_vx_u8m4_b2(text_u8m4, ',', vector_length), vector_length),
        slashes_b2, vector_length);
    vbool2_t const blanks_b2 = __riscv_vmor_mm_b2(
        __riscv_vmor_mm_b2(__riscv_vmor_mm_b2(__riscv_vmseq_vx_u8m4_b2(text_u8m4, ' ', vector_length),
                                              __riscv_vmseq_vx_u8m4_b2(text_u8m4, '\t', vector_length), vector_length),
                           __riscv_vmor_mm_b2(__riscv_vmseq_vx_u8m4_b2(text_u8m4, '\n', vector_length),
                                              __riscv_vmseq_vx_u8m4_b2(text_u8m4, '\r', vector_length), vector_length),
                           vector_length),
        slashes_b2, vector_length);
    return sz_subdocuments_block_settle_(
        sz_subdocuments_mask_bits_rvv_(__riscv_vmseq_vx_u8m4_b2(text_u8m4, '"', vector_length)),
        sz_subdocuments_mask_bits_rvv_(__riscv_vmseq_vx_u8m4_b2(text_u8m4, '\\', vector_length)),
        sz_subdocuments_mask_bits_rvv_(__riscv_vmseq_vx_u8m4_b2(folded_u8m4, '{', vector_length)),
        sz_subdocuments_mask_bits_rvv_(__riscv_vmseq_vx_u8m4_b2(folded_u8m4, '}', vector_length)),
        sz_subdocuments_mask_bits_rvv_(separators_b2), sz_subdocuments_mask_bits_rvv_(blanks_b2), carry);
}

#if STRINGZILLA_TARGET_RVV

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_rvv(sz_subdocuments_engine_t *engine,
                                                            sz_strings_view_t const *paths,
                                                            sz_subdocuments_format_t format, sz_size_t *malformed_index,
                                                            sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_subdocuments_engine_compile_(engine, paths, format, malformed_index, sz_cap_rvv_k, allocator, stream);
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts_rvv(sz_subdocuments_engine_t *engine,
                                                       sz_strings_view_t const *documents, sz_size_t *counts,
                                                       sz_size_t counts_stride, sz_size_t *malformed_index,
                                                       sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_counts_with_(engine, &sz_subdocuments_classify_rvv_, &sz_subdocuments_classify_rvv_,
                                        &sz_find_byte_rvv_, documents, counts, counts_stride, malformed_index);
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_rvv(sz_subdocuments_engine_t *engine,
                                                     sz_strings_view_t const *documents, sz_match_t *matches,
                                                     sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                     sz_size_t *malformed_index, sz_stream_t stream) {
    sz_unused_(stream);
    return sz_subdocuments_find_with_(engine, &sz_subdocuments_classify_rvv_, &sz_subdocuments_classify_rvv_,
                                      &sz_find_byte_rvv_, documents, matches, matches_capacity, matches_offsets,
                                      malformed_index);
}

#endif // STRINGZILLA_TARGET_RVV

#pragma endregion RVV

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_RISCV64_RVV_
#endif // STRINGZILLA_ARCH_RISCV64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_SUBDOCUMENTS_RVV_H_
