/**
 *  @file include/stringzilla/overlap/skylake.h
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief AVX-512 (Skylake) backend for window overlap: eight prefix hashes from one word, the
 *      window hashes eight positions at a time, the sort over sixteen-key registers, and the B-tree
 *      probe sixteen keys at a time.
 *
 *  @sa include/stringzilla/overlap.h
 */
#ifndef STRINGZILLA_OVERLAP_SKYLAKE_H_
#define STRINGZILLA_OVERLAP_SKYLAKE_H_

#include "stringzilla/types.h"

#include "stringzilla/overlap/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_TARGET_SKYLAKE
#if defined(__clang__) && STRINGZILLA_HAS_CLANG_EVEX512_
#pragma clang attribute push(                                                                       \
    __attribute__((target("avx,avx2,avx512f,avx512vl,avx512dq,avx512bw,bmi,bmi2,popcnt,evex512"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx2,avx512f,avx512vl,avx512dq,avx512bw,bmi,bmi2,popcnt"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx2", "avx512f", "avx512vl", "avx512dq", "avx512bw", "bmi", "bmi2", "popcnt")
#endif

#pragma region Skylake

/** Positions one ZMM step advances the chain by: the eight prefix hashes ending inside one word. */
enum { sz_overlap_skylake_f64x8_positions_per_step_k = 8 };

/** Keys one ZMM register holds, and keys the sixteen-register sorting network orders at once. */
enum { sz_overlap_skylake_keys_per_register_k = 16, sz_overlap_skylake_keys_per_run_k = 256 };

/** (multiplier · multiplicand + addend) mod p, in [0, p), exact for every input below 2³²: the
 *  product's rounded head and exact tail reduce together, so the 53-bit mantissa never binds. */
STRINGZILLA_API_COMPTIME __m512d sz_overlap_skylake_multiply_add_(__m512d multiplier_vec, __m512d multiplicand_vec,
                                                                  __m512d addend_vec) {
    sz_u512_vec_t modulus_vec, reciprocal_vec, high_vec, low_vec, quotient_vec, folded_vec, second_vec, residue_vec;
    modulus_vec.zmm_pd = _mm512_set1_pd((sz_f64_t)sz_overlap_modulus_k);
    reciprocal_vec.zmm_pd = _mm512_set1_pd(1.0 / (sz_f64_t)sz_overlap_modulus_k);
    high_vec.zmm_pd = _mm512_mul_pd(multiplier_vec, multiplicand_vec);
    low_vec.zmm_pd = _mm512_fmsub_pd(multiplier_vec, multiplicand_vec, high_vec.zmm_pd);
    quotient_vec.zmm_pd = _mm512_roundscale_pd(_mm512_mul_pd(high_vec.zmm_pd, reciprocal_vec.zmm_pd),
                                               _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    folded_vec.zmm_pd = _mm512_add_pd(
        _mm512_add_pd(_mm512_fnmadd_pd(quotient_vec.zmm_pd, modulus_vec.zmm_pd, high_vec.zmm_pd), low_vec.zmm_pd),
        addend_vec);
    second_vec.zmm_pd = _mm512_roundscale_pd(_mm512_mul_pd(folded_vec.zmm_pd, reciprocal_vec.zmm_pd),
                                             _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    residue_vec.zmm_pd = _mm512_fnmadd_pd(second_vec.zmm_pd, modulus_vec.zmm_pd, folded_vec.zmm_pd);
    __mmask8 const negative_mask_m8 = _mm512_cmp_pd_mask(residue_vec.zmm_pd, _mm512_setzero_pd(), _CMP_LT_OQ);
    return _mm512_mask_add_pd(residue_vec.zmm_pd, negative_mask_m8, residue_vec.zmm_pd, modulus_vec.zmm_pd);
}

/** Advances the chain over eight bytes, writing the eight prefix hashes ending inside that word. */
STRINGZILLA_API_COMPTIME sz_f64_t sz_overlap_f64x8_prefix_hash_step_skylake(sz_f64_t prior, sz_cptr_t text,
                                                                            sz_f64_t *prefix_hashes) {

    // The word's halves as big-endian integers, each exact in `f64`; shifts past 32 answer zero.
    sz_u32_t const high_word = sz_u32_bytes_reverse(sz_u32_load(text).u32);
    sz_u32_t const low_word = sz_u32_bytes_reverse(sz_u32_load(text + 4).u32);
    __m256i const high_shifts_u32x8 = _mm256_setr_epi32(24, 16, 8, 0, 0, 0, 0, 0);
    __m256i const low_shifts_u32x8 = _mm256_setr_epi32(32, 32, 32, 32, 24, 16, 8, 0);
    sz_u512_vec_t high_parts_vec, low_parts_vec, joining_vec, powers_vec, chunks_vec, values_vec;
    high_parts_vec.zmm_pd = _mm512_cvtepu32_pd(_mm256_srlv_epi32(_mm256_set1_epi32((int)high_word), high_shifts_u32x8));
    low_parts_vec.zmm_pd = _mm512_cvtepu32_pd(_mm256_srlv_epi32(_mm256_set1_epi32((int)low_word), low_shifts_u32x8));

    // Positions 0-3 take the high half alone; positions 4-7 scale it past however much of the low half they cover.
    joining_vec.zmm_pd = _mm512_setr_pd(1.0, 1.0, 1.0, 1.0, sz_overlap_powers_of_256_k[1],
                                        sz_overlap_powers_of_256_k[2], sz_overlap_powers_of_256_k[3],
                                        sz_overlap_powers_of_256_k[4]);
    chunks_vec.zmm_pd = sz_overlap_skylake_multiply_add_(high_parts_vec.zmm_pd, joining_vec.zmm_pd,
                                                         low_parts_vec.zmm_pd);
    powers_vec.zmm_pd = _mm512_loadu_pd(sz_overlap_powers_of_256_k + 1);
    values_vec.zmm_pd = sz_overlap_skylake_multiply_add_(_mm512_set1_pd(prior), powers_vec.zmm_pd, chunks_vec.zmm_pd);
    _mm512_storeu_pd(prefix_hashes, values_vec.zmm_pd);
    return prefix_hashes[7];
}

/** The last step over @p count positions, fewer than a full step's. */
STRINGZILLA_API_COMPTIME sz_f64_t sz_overlap_f64x8_prefix_hash_step_tail_skylake(sz_f64_t prior, sz_cptr_t text,
                                                                                 sz_size_t count,
                                                                                 sz_f64_t *prefix_hashes) {
    for (sz_size_t position = 0; position != count; ++position)
        prior = sz_overlap_f64x1_prefix_hash_step_serial(prior, text + position, prefix_hashes + position);
    return prior;
}

/** Eight positions' window hashes: H(i, w) = P(i + w) − P(i) · bʷ, a full-width hash
 *  under the modulus. */
STRINGZILLA_API_COMPTIME void sz_overlap_f64x8_window_hash_step_skylake(sz_f64_t const *prefix_hashes_at_start,
                                                                        sz_f64_t const *prefix_hashes_at_end,
                                                                        sz_f64_t window_power,
                                                                        sz_u32_t *window_hashes) {
    sz_u512_vec_t start_vec, end_vec, shifted_vec, difference_vec, residues_vec;
    start_vec.zmm_pd = _mm512_loadu_pd(prefix_hashes_at_start);
    end_vec.zmm_pd = _mm512_loadu_pd(prefix_hashes_at_end);
    shifted_vec.zmm_pd = sz_overlap_skylake_multiply_add_(start_vec.zmm_pd, _mm512_set1_pd(window_power),
                                                          _mm512_setzero_pd());
    difference_vec.zmm_pd = _mm512_sub_pd(end_vec.zmm_pd, shifted_vec.zmm_pd);
    __mmask8 const negative_mask_m8 = _mm512_cmp_pd_mask(difference_vec.zmm_pd, _mm512_setzero_pd(), _CMP_LT_OQ);
    residues_vec.zmm_pd = _mm512_mask_add_pd(difference_vec.zmm_pd, negative_mask_m8, difference_vec.zmm_pd,
                                             _mm512_set1_pd((sz_f64_t)sz_overlap_modulus_k));
    _mm256_storeu_si256((__m256i *)window_hashes, _mm512_cvttpd_epu32(residues_vec.zmm_pd));
}

/** The last step over @p count positions, fewer than a full step's. */
STRINGZILLA_API_COMPTIME void sz_overlap_f64x8_window_hash_step_tail_skylake(sz_f64_t const *prefix_hashes_at_start,
                                                                             sz_f64_t const *prefix_hashes_at_end,
                                                                             sz_f64_t window_power, sz_size_t count,
                                                                             sz_u32_t *window_hashes) {
    __mmask8 const tail_mask_m8 = sz_u8_mask_until_(count);
    sz_u512_vec_t start_vec, end_vec, shifted_vec, difference_vec, residues_vec;
    start_vec.zmm_pd = _mm512_maskz_loadu_pd(tail_mask_m8, prefix_hashes_at_start);
    end_vec.zmm_pd = _mm512_maskz_loadu_pd(tail_mask_m8, prefix_hashes_at_end);
    shifted_vec.zmm_pd = sz_overlap_skylake_multiply_add_(start_vec.zmm_pd, _mm512_set1_pd(window_power),
                                                          _mm512_setzero_pd());
    difference_vec.zmm_pd = _mm512_sub_pd(end_vec.zmm_pd, shifted_vec.zmm_pd);
    __mmask8 const negative_mask_m8 = _mm512_cmp_pd_mask(difference_vec.zmm_pd, _mm512_setzero_pd(), _CMP_LT_OQ);
    residues_vec.zmm_pd = _mm512_mask_add_pd(difference_vec.zmm_pd, negative_mask_m8, difference_vec.zmm_pd,
                                             _mm512_set1_pd((sz_f64_t)sz_overlap_modulus_k));
    _mm256_mask_storeu_epi32(window_hashes, tail_mask_m8, _mm512_cvttpd_epu32(residues_vec.zmm_pd));
}

/** Positions @c i ^ @p flip for the sixteen keys of a register: the partner at a distance, or
 *  across a mirror. */
STRINGZILLA_API_COMPTIME __m512i sz_overlap_skylake_partners_(sz_u32_t flip) {
    return _mm512_setr_epi32((int)(0u ^ flip), (int)(1u ^ flip), (int)(2u ^ flip), (int)(3u ^ flip),   //
                             (int)(4u ^ flip), (int)(5u ^ flip), (int)(6u ^ flip), (int)(7u ^ flip),   //
                             (int)(8u ^ flip), (int)(9u ^ flip), (int)(10u ^ flip), (int)(11u ^ flip), //
                             (int)(12u ^ flip), (int)(13u ^ flip), (int)(14u ^ flip), (int)(15u ^ flip));
}

/** The sixteen keys of a register in reverse order. */
STRINGZILLA_API_COMPTIME __m512i sz_overlap_skylake_reverse_(__m512i keys_u32x16) {
    return _mm512_permutexvar_epi32(sz_overlap_skylake_partners_(15), keys_u32x16);
}

/** Compare-exchange inside a register: every key against the one @p partners_u32x16 names, masked
 *  positions take the larger. */
STRINGZILLA_API_COMPTIME __m512i sz_overlap_skylake_exchange_within_(__m512i keys_u32x16, __m512i partners_u32x16,
                                                                     __mmask16 take_larger_mask_m16) {
    __m512i const partner_keys_u32x16 = _mm512_permutexvar_epi32(partners_u32x16, keys_u32x16);
    return _mm512_mask_mov_epi32(_mm512_min_epu32(keys_u32x16, partner_keys_u32x16), take_larger_mask_m16,
                                 _mm512_max_epu32(keys_u32x16, partner_keys_u32x16));
}

/** Two registers a fixed distance apart: @p lower keeps the minima, @p upper the maxima. */
STRINGZILLA_HELPER_INLINE void sz_overlap_skylake_exchange_(__m512i *lower_u32x16, __m512i *upper_u32x16) {
    __m512i const smaller_u32x16 = _mm512_min_epu32(*lower_u32x16, *upper_u32x16);
    __m512i const larger_u32x16 = _mm512_max_epu32(*lower_u32x16, *upper_u32x16);
    *lower_u32x16 = smaller_u32x16, *upper_u32x16 = larger_u32x16;
}

/** The mirrored stage opening a merge of two ascending runs: @p upper is read
 *  and written reversed. */
STRINGZILLA_HELPER_INLINE void sz_overlap_skylake_exchange_mirrored_(__m512i *lower_u32x16, __m512i *upper_u32x16) {
    __m512i const reversed_u32x16 = sz_overlap_skylake_reverse_(*upper_u32x16);
    __m512i const smaller_u32x16 = _mm512_min_epu32(*lower_u32x16, reversed_u32x16);
    __m512i const larger_u32x16 = _mm512_max_epu32(*lower_u32x16, reversed_u32x16);
    *lower_u32x16 = smaller_u32x16, *upper_u32x16 = sz_overlap_skylake_reverse_(larger_u32x16);
}

/** Sorts the sixteen keys of one register ascending: the merges of two, four, eight and sixteen
 *  keys, all inside it. */
STRINGZILLA_API_COMPTIME __m512i sz_overlap_skylake_sort_within_(__m512i keys_u32x16) {
    keys_u32x16 = sz_overlap_skylake_exchange_within_(keys_u32x16, sz_overlap_skylake_partners_(1), 0xAAAA);
    keys_u32x16 = sz_overlap_skylake_exchange_within_(keys_u32x16, sz_overlap_skylake_partners_(3), 0xCCCC);
    keys_u32x16 = sz_overlap_skylake_exchange_within_(keys_u32x16, sz_overlap_skylake_partners_(1), 0xAAAA);
    keys_u32x16 = sz_overlap_skylake_exchange_within_(keys_u32x16, sz_overlap_skylake_partners_(7), 0xF0F0);
    keys_u32x16 = sz_overlap_skylake_exchange_within_(keys_u32x16, sz_overlap_skylake_partners_(2), 0xCCCC);
    keys_u32x16 = sz_overlap_skylake_exchange_within_(keys_u32x16, sz_overlap_skylake_partners_(1), 0xAAAA);
    keys_u32x16 = sz_overlap_skylake_exchange_within_(keys_u32x16, sz_overlap_skylake_partners_(15), 0xFF00);
    keys_u32x16 = sz_overlap_skylake_exchange_within_(keys_u32x16, sz_overlap_skylake_partners_(4), 0xF0F0);
    keys_u32x16 = sz_overlap_skylake_exchange_within_(keys_u32x16, sz_overlap_skylake_partners_(2), 0xCCCC);
    return sz_overlap_skylake_exchange_within_(keys_u32x16, sz_overlap_skylake_partners_(1), 0xAAAA);
}

/** The ascending half-cleaners at distances eight, four, two and one, closing a merge
 *  inside the register. */
STRINGZILLA_API_COMPTIME __m512i sz_overlap_skylake_merge_within_(__m512i keys_u32x16) {
    keys_u32x16 = sz_overlap_skylake_exchange_within_(keys_u32x16, sz_overlap_skylake_partners_(8), 0xFF00);
    keys_u32x16 = sz_overlap_skylake_exchange_within_(keys_u32x16, sz_overlap_skylake_partners_(4), 0xF0F0);
    keys_u32x16 = sz_overlap_skylake_exchange_within_(keys_u32x16, sz_overlap_skylake_partners_(2), 0xCCCC);
    return sz_overlap_skylake_exchange_within_(keys_u32x16, sz_overlap_skylake_partners_(1), 0xAAAA);
}

/** Loads four registers, sixty-four keys. */
STRINGZILLA_HELPER_INLINE void sz_overlap_skylake_load_4_(sz_u32_t const *keys, __m512i *registers_u32x16) {
    registers_u32x16[0] = _mm512_loadu_si512(keys + 0);
    registers_u32x16[1] = _mm512_loadu_si512(keys + 16);
    registers_u32x16[2] = _mm512_loadu_si512(keys + 32);
    registers_u32x16[3] = _mm512_loadu_si512(keys + 48);
}

/** Stores four registers, sixty-four keys. */
STRINGZILLA_HELPER_INLINE void sz_overlap_skylake_store_4_(__m512i const *registers_u32x16, sz_u32_t *keys) {
    _mm512_storeu_si512(keys + 0, registers_u32x16[0]);
    _mm512_storeu_si512(keys + 16, registers_u32x16[1]);
    _mm512_storeu_si512(keys + 32, registers_u32x16[2]);
    _mm512_storeu_si512(keys + 48, registers_u32x16[3]);
}

/** Closes a merge inside each of four registers. */
STRINGZILLA_HELPER_INLINE void sz_overlap_skylake_merge_within_4_(__m512i *registers_u32x16) {
    registers_u32x16[0] = sz_overlap_skylake_merge_within_(registers_u32x16[0]);
    registers_u32x16[1] = sz_overlap_skylake_merge_within_(registers_u32x16[1]);
    registers_u32x16[2] = sz_overlap_skylake_merge_within_(registers_u32x16[2]);
    registers_u32x16[3] = sz_overlap_skylake_merge_within_(registers_u32x16[3]);
}

/** Ascending stage, pairing each of the first eight @p registers_u32x16 with the eighth after. */
STRINGZILLA_HELPER_INLINE void sz_overlap_skylake_exchange_stride_8_(__m512i *registers_u32x16) {
    sz_overlap_skylake_exchange_(&registers_u32x16[0], &registers_u32x16[8]);
    sz_overlap_skylake_exchange_(&registers_u32x16[1], &registers_u32x16[9]);
    sz_overlap_skylake_exchange_(&registers_u32x16[2], &registers_u32x16[10]);
    sz_overlap_skylake_exchange_(&registers_u32x16[3], &registers_u32x16[11]);
    sz_overlap_skylake_exchange_(&registers_u32x16[4], &registers_u32x16[12]);
    sz_overlap_skylake_exchange_(&registers_u32x16[5], &registers_u32x16[13]);
    sz_overlap_skylake_exchange_(&registers_u32x16[6], &registers_u32x16[14]);
    sz_overlap_skylake_exchange_(&registers_u32x16[7], &registers_u32x16[15]);
}

/** Ascending stage, pairing each of the first four @p registers_u32x16 with the fourth after. */
STRINGZILLA_HELPER_INLINE void sz_overlap_skylake_exchange_stride_4_(__m512i *registers_u32x16) {
    sz_overlap_skylake_exchange_(&registers_u32x16[0], &registers_u32x16[4]);
    sz_overlap_skylake_exchange_(&registers_u32x16[1], &registers_u32x16[5]);
    sz_overlap_skylake_exchange_(&registers_u32x16[2], &registers_u32x16[6]);
    sz_overlap_skylake_exchange_(&registers_u32x16[3], &registers_u32x16[7]);
}

/** The ascending stages at register strides two and one over four registers, then within each. */
STRINGZILLA_HELPER_INLINE void sz_overlap_skylake_merge_4_(__m512i *registers_u32x16) {
    sz_overlap_skylake_exchange_(&registers_u32x16[0], &registers_u32x16[2]);
    sz_overlap_skylake_exchange_(&registers_u32x16[1], &registers_u32x16[3]);
    sz_overlap_skylake_exchange_(&registers_u32x16[0], &registers_u32x16[1]);
    sz_overlap_skylake_exchange_(&registers_u32x16[2], &registers_u32x16[3]);
    sz_overlap_skylake_merge_within_4_(registers_u32x16);
}

/** Sorts sixty-four keys in four registers, in place. */
STRINGZILLA_API_COMPTIME void sz_overlap_skylake_sort_4_(sz_u32_t *keys) {
    __m512i registers_u32x16[4];
    sz_overlap_skylake_load_4_(keys, registers_u32x16);
    registers_u32x16[0] = sz_overlap_skylake_sort_within_(registers_u32x16[0]);
    registers_u32x16[1] = sz_overlap_skylake_sort_within_(registers_u32x16[1]);
    registers_u32x16[2] = sz_overlap_skylake_sort_within_(registers_u32x16[2]);
    registers_u32x16[3] = sz_overlap_skylake_sort_within_(registers_u32x16[3]);
    // Thirty-two keys: register pairs mirrored, then within.
    sz_overlap_skylake_exchange_mirrored_(&registers_u32x16[0], &registers_u32x16[1]);
    sz_overlap_skylake_exchange_mirrored_(&registers_u32x16[2], &registers_u32x16[3]);
    sz_overlap_skylake_merge_within_4_(registers_u32x16);
    // Sixty-four keys: the quad mirrored, then the stages below.
    sz_overlap_skylake_exchange_mirrored_(&registers_u32x16[0], &registers_u32x16[3]);
    sz_overlap_skylake_exchange_mirrored_(&registers_u32x16[1], &registers_u32x16[2]);
    sz_overlap_skylake_exchange_(&registers_u32x16[0], &registers_u32x16[1]);
    sz_overlap_skylake_exchange_(&registers_u32x16[2], &registers_u32x16[3]);
    sz_overlap_skylake_merge_within_4_(registers_u32x16);
    sz_overlap_skylake_store_4_(registers_u32x16, keys);
}

/** Sorts one hundred twenty-eight keys in eight registers, in place: two sorted quads,
 *  then their merge. */
STRINGZILLA_API_COMPTIME void sz_overlap_skylake_sort_8_(sz_u32_t *keys) {
    sz_overlap_skylake_sort_4_(keys);
    sz_overlap_skylake_sort_4_(keys + 64);
    __m512i registers_u32x16[8];
    sz_overlap_skylake_load_4_(keys, registers_u32x16);
    sz_overlap_skylake_load_4_(keys + 64, registers_u32x16 + 4);
    sz_overlap_skylake_exchange_mirrored_(&registers_u32x16[0], &registers_u32x16[7]);
    sz_overlap_skylake_exchange_mirrored_(&registers_u32x16[1], &registers_u32x16[6]);
    sz_overlap_skylake_exchange_mirrored_(&registers_u32x16[2], &registers_u32x16[5]);
    sz_overlap_skylake_exchange_mirrored_(&registers_u32x16[3], &registers_u32x16[4]);
    sz_overlap_skylake_merge_4_(registers_u32x16);
    sz_overlap_skylake_merge_4_(registers_u32x16 + 4);
    sz_overlap_skylake_store_4_(registers_u32x16, keys);
    sz_overlap_skylake_store_4_(registers_u32x16 + 4, keys + 64);
}

/** Sorts one 256-key run in sixteen registers, in place: two sorted halves, then their merge. */
STRINGZILLA_API_COMPTIME void sz_overlap_skylake_sort_16_(sz_u32_t *keys) {
    sz_overlap_skylake_sort_8_(keys);
    sz_overlap_skylake_sort_8_(keys + 128);
    __m512i registers_u32x16[16];
    sz_overlap_skylake_load_4_(keys, registers_u32x16);
    sz_overlap_skylake_load_4_(keys + 64, registers_u32x16 + 4);
    sz_overlap_skylake_load_4_(keys + 128, registers_u32x16 + 8);
    sz_overlap_skylake_load_4_(keys + 192, registers_u32x16 + 12);
    sz_overlap_skylake_exchange_mirrored_(&registers_u32x16[0], &registers_u32x16[15]);
    sz_overlap_skylake_exchange_mirrored_(&registers_u32x16[1], &registers_u32x16[14]);
    sz_overlap_skylake_exchange_mirrored_(&registers_u32x16[2], &registers_u32x16[13]);
    sz_overlap_skylake_exchange_mirrored_(&registers_u32x16[3], &registers_u32x16[12]);
    sz_overlap_skylake_exchange_mirrored_(&registers_u32x16[4], &registers_u32x16[11]);
    sz_overlap_skylake_exchange_mirrored_(&registers_u32x16[5], &registers_u32x16[10]);
    sz_overlap_skylake_exchange_mirrored_(&registers_u32x16[6], &registers_u32x16[9]);
    sz_overlap_skylake_exchange_mirrored_(&registers_u32x16[7], &registers_u32x16[8]);
    sz_overlap_skylake_exchange_stride_4_(registers_u32x16);
    sz_overlap_skylake_exchange_stride_4_(registers_u32x16 + 8);
    sz_overlap_skylake_merge_4_(registers_u32x16);
    sz_overlap_skylake_merge_4_(registers_u32x16 + 4);
    sz_overlap_skylake_merge_4_(registers_u32x16 + 8);
    sz_overlap_skylake_merge_4_(registers_u32x16 + 12);
    sz_overlap_skylake_store_4_(registers_u32x16, keys);
    sz_overlap_skylake_store_4_(registers_u32x16 + 4, keys + 64);
    sz_overlap_skylake_store_4_(registers_u32x16 + 8, keys + 128);
    sz_overlap_skylake_store_4_(registers_u32x16 + 12, keys + 192);
}

/** Closes a merge inside one 256-key run: the ascending stages at distances one hundred
 *  twenty-eight down to one. */
STRINGZILLA_API_COMPTIME void sz_overlap_skylake_merge_run_(sz_u32_t *keys) {
    __m512i registers_u32x16[16];
    sz_overlap_skylake_load_4_(keys, registers_u32x16);
    sz_overlap_skylake_load_4_(keys + 64, registers_u32x16 + 4);
    sz_overlap_skylake_load_4_(keys + 128, registers_u32x16 + 8);
    sz_overlap_skylake_load_4_(keys + 192, registers_u32x16 + 12);
    sz_overlap_skylake_exchange_stride_8_(registers_u32x16);
    sz_overlap_skylake_exchange_stride_4_(registers_u32x16);
    sz_overlap_skylake_exchange_stride_4_(registers_u32x16 + 8);
    sz_overlap_skylake_merge_4_(registers_u32x16);
    sz_overlap_skylake_merge_4_(registers_u32x16 + 4);
    sz_overlap_skylake_merge_4_(registers_u32x16 + 8);
    sz_overlap_skylake_merge_4_(registers_u32x16 + 12);
    sz_overlap_skylake_store_4_(registers_u32x16, keys);
    sz_overlap_skylake_store_4_(registers_u32x16 + 4, keys + 64);
    sz_overlap_skylake_store_4_(registers_u32x16 + 8, keys + 128);
    sz_overlap_skylake_store_4_(registers_u32x16 + 12, keys + 192);
}

/** Sorts @p count keys ascending in place, unsigned, and drops repeats, answering how many remain;
 *  the buffer holds @ref sz_overlap_btree_sorted_capacity entries. */
STRINGZILLA_API_COMPTIME sz_size_t sz_overlap_u32x16_btree_sort_skylake(sz_u32_t *keys, sz_size_t count) {
    sz_size_t const keys_per_register = sz_overlap_skylake_keys_per_register_k;
    sz_size_t const keys_per_run = sz_overlap_skylake_keys_per_run_k;
    sz_size_t const capacity = sz_overlap_btree_sorted_capacity(count);
    for (sz_size_t position = count; position != capacity; ++position) keys[position] = sz_overlap_padding_key_k;
    if (capacity == 64) sz_overlap_skylake_sort_4_(keys);
    else if (capacity == 128) sz_overlap_skylake_sort_8_(keys);
    else
        for (sz_size_t run = 0; run != capacity / keys_per_run; ++run)
            sz_overlap_skylake_sort_16_(keys + run * keys_per_run);

    // Every merge opens mirrored over memory, so no run is ever descending; the stages that reach past a run
    // exchange whole registers in memory, and the rest close inside one run's registers.
    for (sz_size_t phase = 2 * keys_per_run; phase <= capacity; phase *= 2) {
        for (sz_size_t start = 0; start != capacity; start += phase)
            for (sz_size_t offset = 0; offset != phase / 2; offset += keys_per_register) {
                __m512i lower_u32x16 = _mm512_loadu_si512(keys + start + offset);
                __m512i upper_u32x16 = _mm512_loadu_si512(keys + start + phase - keys_per_register - offset);
                sz_overlap_skylake_exchange_mirrored_(&lower_u32x16, &upper_u32x16);
                _mm512_storeu_si512(keys + start + offset, lower_u32x16);
                _mm512_storeu_si512(keys + start + phase - keys_per_register - offset, upper_u32x16);
            }
        for (sz_size_t distance = phase / 4; distance >= keys_per_run; distance /= 2)
            for (sz_size_t start = 0; start != capacity; start += 2 * distance)
                for (sz_size_t offset = 0; offset != distance; offset += keys_per_register) {
                    __m512i lower_u32x16 = _mm512_loadu_si512(keys + start + offset);
                    __m512i upper_u32x16 = _mm512_loadu_si512(keys + start + distance + offset);
                    sz_overlap_skylake_exchange_(&lower_u32x16, &upper_u32x16);
                    _mm512_storeu_si512(keys + start + offset, lower_u32x16);
                    _mm512_storeu_si512(keys + start + distance + offset, upper_u32x16);
                }
        for (sz_size_t run = 0; run != capacity / keys_per_run; ++run)
            sz_overlap_skylake_merge_run_(keys + run * keys_per_run);
    }
    return sz_overlap_btree_unique_(keys, count);
}

/** One branch level: the child ordinal a flipped @p key_u32x16 descends into, the count of
 *  separators below it. */
STRINGZILLA_API_COMPTIME sz_size_t sz_overlap_skylake_branch_step_(sz_u32_t const *node, __m512i key_u32x16) {
    __mmask16 const below_mask_m16 = _mm512_cmpgt_epi32_mask(key_u32x16, _mm512_loadu_si512(node));
    return (sz_size_t)_mm_popcnt_u32((unsigned)below_mask_m16);
}

/** The leaf compare: one when the flipped @p key_u32x16 sits in this node, zero otherwise. */
STRINGZILLA_API_COMPTIME sz_size_t sz_overlap_skylake_leaf_step_(sz_u32_t const *node, __m512i key_u32x16) {
    return _mm512_cmpeq_epi32_mask(key_u32x16, _mm512_loadu_si512(node)) != 0;
}

/** Walks the whole tree for one flipped key, the root included. */
STRINGZILLA_API_COMPTIME sz_size_t sz_overlap_skylake_probe_key_(sz_overlap_btree_t const *btree,
                                                                 sz_u32_t flipped_key) {
    sz_size_t const keys_per_node = sz_overlap_keys_per_node_k, branches_per_node = sz_overlap_branches_per_node_k;
    __m512i const key_u32x16 = _mm512_set1_epi32((int)flipped_key);
    sz_size_t node = 0;
    for (sz_size_t level = 0; level + 1 != btree->levels; ++level)
        node = node * branches_per_node +
               sz_overlap_skylake_branch_step_(btree->nodes + (btree->level_bases[level] + node) * keys_per_node,
                                               key_u32x16);
    return sz_overlap_skylake_leaf_step_(btree->nodes + (btree->level_bases[btree->levels - 1] + node) * keys_per_node,
                                         key_u32x16);
}

/** Counts how many of @p count raw keys of one candidate the tree holds. */
STRINGZILLA_API_COMPTIME sz_size_t sz_overlap_u32x16_btree_probe_skylake(sz_overlap_btree_t const *btree,
                                                                         sz_u32_t const *keys, sz_size_t count) {
    sz_size_t const keys_per_node = sz_overlap_keys_per_node_k, branches_per_node = sz_overlap_branches_per_node_k;
    sz_size_t const keys_per_register = sz_overlap_skylake_keys_per_register_k;
    sz_u32_t const *const root = btree->nodes + btree->level_bases[0] * keys_per_node;
    sz_u32_t const *const leaves = btree->nodes + btree->level_bases[btree->levels - 1] * keys_per_node;
    __m512i const flip_u32x16 = _mm512_set1_epi32((int)sz_overlap_sign_flip_k);
    __m512i const ones_u32x16 = _mm512_set1_epi32(1);
    sz_size_t matches = 0, index = 0;

    // Sixteen keys meet the root transposed, each separator broadcast against all sixteen; the deeper levels are
    // sixteen interleaved walks.
    if (btree->levels > 1)
        for (; index + keys_per_register <= count; index += keys_per_register) {
            __m512i const keys_u32x16 = _mm512_xor_si512(_mm512_loadu_si512(keys + index), flip_u32x16);
            __m512i children_u32x16 = _mm512_setzero_si512();
            for (sz_size_t separator = 0; separator != keys_per_node; ++separator) {
                __mmask16 const below_mask_m16 = _mm512_cmpgt_epi32_mask(keys_u32x16,
                                                                         _mm512_set1_epi32((int)root[separator]));
                children_u32x16 = _mm512_mask_add_epi32(children_u32x16, below_mask_m16, children_u32x16, ones_u32x16);
            }

            sz_u32_t flipped_keys[16], walks[16];
            _mm512_storeu_si512(flipped_keys, keys_u32x16);
            _mm512_storeu_si512(walks, children_u32x16);
            for (sz_size_t level = 1; level + 1 != btree->levels; ++level) {
                sz_u32_t const *const level_nodes = btree->nodes + btree->level_bases[level] * keys_per_node;
                for (sz_size_t walk = 0; walk != keys_per_register; ++walk)
                    walks[walk] = (sz_u32_t)(walks[walk] * branches_per_node +
                                             sz_overlap_skylake_branch_step_(
                                                 level_nodes + walks[walk] * keys_per_node,
                                                 _mm512_set1_epi32((int)flipped_keys[walk])));
            }
            for (sz_size_t walk = 0; walk != keys_per_register; ++walk)
                matches += sz_overlap_skylake_leaf_step_(leaves + walks[walk] * keys_per_node,
                                                         _mm512_set1_epi32((int)flipped_keys[walk]));
        }
    for (; index != count; ++index)
        matches += sz_overlap_skylake_probe_key_(btree, keys[index] ^ sz_overlap_sign_flip_k);
    return matches;
}

/**
 *  @brief Prepares every query of @p queries into one block, hashing and sorting on
 *      the Skylake tier.
 *  @param[in] alloc Where the forest's block comes from, or @c STRINGZILLA_NULL for the
 *      default host allocator.
 *  @sa sz_overlap_engine_init_cpu
 */
STRINGZILLA_API_COMPTIME sz_status_t sz_overlap_engine_init_skylake(sz_sequence_t const *queries,
                                                                    sz_size_t const *window_widths,
                                                                    sz_size_t window_widths_count,
                                                                    sz_memory_allocator_t *alloc,
                                                                    sz_overlap_engine_t *engine) {
    sz_size_t const step = sz_overlap_skylake_f64x8_positions_per_step_k;
    sz_memory_allocator_t host;
    if (alloc) host = *alloc;
    else sz_memory_allocator_init_default(&host);
    sz_status_t const opened = sz_overlap_engine_open_(queries, window_widths, window_widths_count, 0, &host, engine);
    if (opened != sz_success_k) return opened;

    // The chain is the engine's own round block, so the first round reuses what the longest query already asked for.
    sz_size_t longest_query = 0;
    for (sz_size_t index = 0; index != engine->count; ++index)
        if (engine->lengths[index] > longest_query) longest_query = engine->lengths[index];
    sz_status_t const grown = sz_overlap_engine_grow_(engine, (longest_query + 1) * sizeof(sz_f64_t));
    if (grown != sz_success_k) {
        sz_overlap_engine_close_(engine);
        return grown;
    }

    // The arena and the key counts stay writable until the engine is handed back; its readers see them const.
    sz_u32_t *const nodes = (sz_u32_t *)engine->nodes;
    sz_u32_t *const keys_counts = (sz_u32_t *)engine->keys_counts;
    sz_f64_t *const chain = (sz_f64_t *)engine->scratch;
    for (sz_size_t index = 0; index != engine->count; ++index) {
        sz_cptr_t const text = queries->get_start(queries->handle, index);
        sz_size_t const length = engine->lengths[index];
        sz_u32_t *const arena = nodes + engine->nodes_offsets[index];
        chain[0] = 0.0;
        sz_f64_t prior = 0.0;
        sz_size_t position = 0;
        for (; position + step <= length; position += step)
            prior = sz_overlap_f64x8_prefix_hash_step_skylake(prior, text + position, chain + position + 1);
        if (position != length)
            sz_overlap_f64x8_prefix_hash_step_tail_skylake(prior, text + position, length - position,
                                                           chain + position + 1);

        // Every width's window hashes land in one tree; a candidate window hash carries its own width, so a hit is
        // attributed to that width and a cross-width coincidence costs `2^-32`.
        sz_size_t written = 0;
        for (sz_size_t width_index = 0; width_index != engine->widths_count; ++width_index) {
            sz_size_t const width = engine->widths[width_index];
            if (!width || width > length) continue;
            sz_f64_t const power = (sz_f64_t)engine->powers[width_index];
            sz_size_t const query_windows = length - width + 1;
            sz_size_t window = 0;
            for (; window + step <= query_windows; window += step)
                sz_overlap_f64x8_window_hash_step_skylake(chain + window, chain + window + width, power,
                                                          arena + written + window);
            if (window != query_windows)
                sz_overlap_f64x8_window_hash_step_tail_skylake(chain + window, chain + window + width, power,
                                                               query_windows - window, arena + written + window);
            written += query_windows;
        }
        sz_overlap_btree_t btree;
        sz_size_t const distinct = sz_overlap_u32x16_btree_sort_skylake(arena, written);
        sz_overlap_btree_prepare(arena, distinct, &btree);
        keys_counts[index] = (sz_u32_t)distinct;
    }

    engine->capability = sz_cap_skylake_k;
    return sz_success_k;
}

STRINGZILLA_API_COMPTIME sz_status_t sz_overlap_scores_skylake(sz_overlap_engine_t *engine,
                                                               sz_sequence_t const *candidates, sz_f32_t *scores,
                                                               sz_size_t scores_query_stride,
                                                               sz_size_t scores_candidate_stride) {
    sz_status_t const dimensions = sz_overlap_engine_strides_(engine, candidates->count, scores_query_stride,
                                                              scores_candidate_stride);
    if (dimensions != sz_success_k) return dimensions;
    if (!candidates->count || !engine->count) return sz_success_k;
    sz_size_t const step = sz_overlap_skylake_f64x8_positions_per_step_k;
    sz_size_t const chains = sz_overlap_interleaved_chains_k;

    sz_size_t longest_candidate = 0;
    for (sz_size_t index = 0; index != candidates->count; ++index) {
        sz_size_t const length = candidates->get_length(candidates->handle, index);
        if (length > longest_candidate) longest_candidate = length;
    }
    sz_status_t const grown = sz_overlap_engine_grow_(
        engine, sz_overlap_engine_round_bytes_(longest_candidate, chains));
    if (grown != sz_success_k) return grown;

    sz_size_t const chain_stride = longest_candidate + 1;
    sz_f64_t *const prefix_hashes = (sz_f64_t *)engine->scratch;
    sz_u32_t *const window_hashes = (sz_u32_t *)(prefix_hashes + chain_stride * chains);

    // Four candidates' chains advance together, each carrying its own prior, so their latency-bound steps overlap.
    for (sz_size_t first = 0; first < candidates->count; first += chains) {
        sz_size_t const interleaved = candidates->count - first < chains ? candidates->count - first : chains;
        sz_cptr_t texts[sz_overlap_interleaved_chains_k];
        sz_size_t lengths[sz_overlap_interleaved_chains_k];
        sz_f64_t priors[sz_overlap_interleaved_chains_k];
        sz_size_t shortest = STRINGZILLA_SIZE_MAX;
        for (sz_size_t chain = 0; chain != interleaved; ++chain) {
            texts[chain] = candidates->get_start(candidates->handle, first + chain);
            lengths[chain] = candidates->get_length(candidates->handle, first + chain);
            priors[chain] = prefix_hashes[chain * chain_stride] = 0.0;
            if (lengths[chain] < shortest) shortest = lengths[chain];
        }
        sz_size_t walked = 0;
        if (interleaved == chains)
            for (; walked + step <= shortest; walked += step)
                for (sz_size_t chain = 0; chain != chains; ++chain)
                    priors[chain] = sz_overlap_f64x8_prefix_hash_step_skylake(
                        priors[chain], texts[chain] + walked, prefix_hashes + chain * chain_stride + walked + 1);
        for (sz_size_t chain = 0; chain != interleaved; ++chain) {
            sz_f64_t *const chain_prefix_hashes = prefix_hashes + chain * chain_stride;
            sz_f64_t running = priors[chain];
            sz_size_t own = walked;
            for (; own + step <= lengths[chain]; own += step)
                running = sz_overlap_f64x8_prefix_hash_step_skylake(running, texts[chain] + own,
                                                                    chain_prefix_hashes + own + 1);
            if (own != lengths[chain])
                sz_overlap_f64x8_prefix_hash_step_tail_skylake(running, texts[chain] + own, lengths[chain] - own,
                                                               chain_prefix_hashes + own + 1);
        }

        // One candidate's window hashes at one width serve every query's tree, so the hashing runs once here and
        // the probe runs `count` times over what it wrote.
        for (sz_size_t chain = 0; chain != interleaved; ++chain) {
            sz_f64_t const *const chain_prefix_hashes = prefix_hashes + chain * chain_stride;
            sz_size_t const length = lengths[chain];
            sz_f32_t *const candidate_scores = scores + (first + chain) * scores_candidate_stride;
            for (sz_size_t width_index = 0; width_index != engine->widths_count; ++width_index) {
                sz_size_t const width = engine->widths[width_index];
                sz_size_t const windows = width && width <= length ? length - width + 1 : 0;
                sz_f64_t const power = (sz_f64_t)engine->powers[width_index];
                sz_size_t window = 0;
                for (; window + step <= windows; window += step)
                    sz_overlap_f64x8_window_hash_step_skylake(chain_prefix_hashes + window,
                                                              chain_prefix_hashes + window + width, power,
                                                              window_hashes + window);
                if (window != windows)
                    sz_overlap_f64x8_window_hash_step_tail_skylake(chain_prefix_hashes + window,
                                                                   chain_prefix_hashes + window + width, power,
                                                                   windows - window, window_hashes + window);
                for (sz_size_t query = 0; query != engine->count; ++query) {
                    sz_size_t const query_length = engine->lengths[query];
                    sz_f32_t *const slot = candidate_scores + query * scores_query_stride + width_index;
                    if (!windows || width > query_length) {
                        *slot = 0.0f;
                        continue;
                    }
                    sz_overlap_btree_t const btree = sz_overlap_engine_row_(engine, query);
                    sz_size_t const matches = sz_overlap_u32x16_btree_probe_skylake(&btree, window_hashes, windows);
                    *slot = sz_overlap_share_(matches, windows, query_length - width + 1);
                }
            }
        }
    }
    return sz_success_k;
}

#pragma endregion Skylake

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_SKYLAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_OVERLAP_SKYLAKE_H_
