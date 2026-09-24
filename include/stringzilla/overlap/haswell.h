/**
 *  @brief AVX2 (Haswell) backend for window overlap: four prefix hashes from one word, the window hashes four
 *      positions at a time, the sort over eight-key registers, and the B-tree probe eight keys at a time.
 *  @file include/stringzilla/overlap/haswell.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/overlap.h
 */
#ifndef STRINGZILLA_OVERLAP_HASWELL_H_
#define STRINGZILLA_OVERLAP_HASWELL_H_

#include "stringzilla/types.h"

#include "stringzilla/overlap/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "fma", "bmi", "bmi2", "popcnt")
#endif

#pragma region Haswell

/** Positions one YMM step advances the chain by: the four prefix hashes ending inside one word. */
enum { sz_overlap_haswell_f64x4_positions_per_step_k = 4 };

/** Keys one YMM register holds, and keys the eight-register sorting network orders at once. */
enum { sz_overlap_haswell_keys_per_register_k = 8, sz_overlap_haswell_keys_per_run_k = 64 };

/** Exact @c u32 → @c f64, since AVX2 offers only the signed conversion. */
SZ_API_COMPTIME __m256d sz_overlap_haswell_u32x4_to_f64x4_(__m128i values_vec) {
    sz_u256_vec_t signed_vec, wrapped_vec;
    signed_vec.ymm_pd = _mm256_cvtepi32_pd(values_vec);
    wrapped_vec.ymm_pd = _mm256_cmp_pd(signed_vec.ymm_pd, _mm256_setzero_pd(), _CMP_LT_OQ);
    return _mm256_add_pd(signed_vec.ymm_pd, _mm256_and_pd(wrapped_vec.ymm_pd, _mm256_set1_pd(4294967296.0)));
}

/** Exact @c f64 → @c u32 for values below 2^32, biasing around the signed conversion's ceiling. */
SZ_API_COMPTIME __m128i sz_overlap_haswell_f64x4_to_u32x4_(__m256d values_vec) {
    __m128i const biased_vec = _mm256_cvttpd_epi32(_mm256_sub_pd(values_vec, _mm256_set1_pd(2147483648.0)));
    return _mm_xor_si128(biased_vec, _mm_set1_epi32((int)0x80000000u));
}

/** @c (multiplier · multiplicand + addend) mod p in @c [0, p), exact for every input below 2^32; the product's
 *  rounded head and its exact tail reduce together, so the 53-bit mantissa never binds. */
SZ_API_COMPTIME __m256d sz_overlap_haswell_multiply_add_(__m256d multiplier_vec, __m256d multiplicand_vec,
                                                         __m256d addend_vec) {
    sz_u256_vec_t modulus_vec, reciprocal_vec, high_vec, low_vec, quotient_vec, folded_vec;
    sz_u256_vec_t second_vec, residue_vec, negative_vec;
    modulus_vec.ymm_pd = _mm256_set1_pd((sz_f64_t)sz_overlap_modulus_k);
    reciprocal_vec.ymm_pd = _mm256_set1_pd(1.0 / (sz_f64_t)sz_overlap_modulus_k);
    high_vec.ymm_pd = _mm256_mul_pd(multiplier_vec, multiplicand_vec);
    low_vec.ymm_pd = _mm256_fmsub_pd(multiplier_vec, multiplicand_vec, high_vec.ymm_pd);
    quotient_vec.ymm_pd = _mm256_round_pd(_mm256_mul_pd(high_vec.ymm_pd, reciprocal_vec.ymm_pd),
                                          _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    folded_vec.ymm_pd = _mm256_add_pd(
        _mm256_add_pd(_mm256_fnmadd_pd(quotient_vec.ymm_pd, modulus_vec.ymm_pd, high_vec.ymm_pd), low_vec.ymm_pd),
        addend_vec);
    second_vec.ymm_pd = _mm256_round_pd(_mm256_mul_pd(folded_vec.ymm_pd, reciprocal_vec.ymm_pd),
                                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    residue_vec.ymm_pd = _mm256_fnmadd_pd(second_vec.ymm_pd, modulus_vec.ymm_pd, folded_vec.ymm_pd);
    negative_vec.ymm_pd = _mm256_cmp_pd(residue_vec.ymm_pd, _mm256_setzero_pd(), _CMP_LT_OQ);
    return _mm256_add_pd(residue_vec.ymm_pd, _mm256_and_pd(negative_vec.ymm_pd, modulus_vec.ymm_pd));
}

/** Advances the chain over four bytes, writing the four prefix hashes ending inside that word. */
SZ_API_COMPTIME sz_f64_t sz_overlap_f64x4_prefix_hash_step_haswell(sz_f64_t prior, sz_cptr_t text,
                                                                   sz_f64_t *prefix_hashes) {
    sz_u32_t const word = sz_u32_bytes_reverse(sz_u32_load(text).u32);
    __m128i const shifted_vec = _mm_srlv_epi32(_mm_set1_epi32((int)word), _mm_setr_epi32(24, 16, 8, 0));
    sz_u256_vec_t chunks_vec, powers_vec, values_vec;
    chunks_vec.ymm_pd = sz_overlap_haswell_u32x4_to_f64x4_(shifted_vec);
    powers_vec.ymm_pd = _mm256_loadu_pd(sz_overlap_powers_of_256_k + 1);
    values_vec.ymm_pd = sz_overlap_haswell_multiply_add_(_mm256_set1_pd(prior), powers_vec.ymm_pd, chunks_vec.ymm_pd);
    _mm256_storeu_pd(prefix_hashes, values_vec.ymm_pd);
    return prefix_hashes[3];
}

/** The last step over @p count positions, fewer than a full step's. */
SZ_API_COMPTIME sz_f64_t sz_overlap_f64x4_prefix_hash_step_tail_haswell(sz_f64_t prior, sz_cptr_t text, sz_size_t count,
                                                                        sz_f64_t *prefix_hashes) {
    for (sz_size_t position = 0; position != count; ++position)
        prior = sz_overlap_f64x1_prefix_hash_step_serial(prior, text + position, prefix_hashes + position);
    return prior;
}

/** Four positions' window hashes: @c H(i,w) = P(i+w) - P(i)·b^w, a full-width hash under the modulus. */
SZ_API_COMPTIME void sz_overlap_f64x4_window_hash_step_haswell(sz_f64_t const *prefix_hashes_at_start,
                                                               sz_f64_t const *prefix_hashes_at_end,
                                                               sz_f64_t window_power, sz_u32_t *window_hashes) {
    sz_u256_vec_t start_vec, end_vec, shifted_vec, difference_vec, negative_vec, residues_vec;
    start_vec.ymm_pd = _mm256_loadu_pd(prefix_hashes_at_start);
    end_vec.ymm_pd = _mm256_loadu_pd(prefix_hashes_at_end);
    shifted_vec.ymm_pd = sz_overlap_haswell_multiply_add_(start_vec.ymm_pd, _mm256_set1_pd(window_power),
                                                          _mm256_setzero_pd());
    difference_vec.ymm_pd = _mm256_sub_pd(end_vec.ymm_pd, shifted_vec.ymm_pd);
    negative_vec.ymm_pd = _mm256_cmp_pd(difference_vec.ymm_pd, _mm256_setzero_pd(), _CMP_LT_OQ);
    residues_vec.ymm_pd = _mm256_add_pd(
        difference_vec.ymm_pd, _mm256_and_pd(negative_vec.ymm_pd, _mm256_set1_pd((sz_f64_t)sz_overlap_modulus_k)));
    _mm_storeu_si128((__m128i *)window_hashes, sz_overlap_haswell_f64x4_to_u32x4_(residues_vec.ymm_pd));
}

/** The last step over @p count positions, fewer than a full step's. */
SZ_API_COMPTIME void sz_overlap_f64x4_window_hash_step_tail_haswell(sz_f64_t const *prefix_hashes_at_start,
                                                                    sz_f64_t const *prefix_hashes_at_end,
                                                                    sz_f64_t window_power, sz_size_t count,
                                                                    sz_u32_t *window_hashes) {
    for (sz_size_t position = 0; position != count; ++position)
        sz_overlap_f64x1_window_hash_step_serial(prefix_hashes_at_start + position, prefix_hashes_at_end + position,
                                                 window_power, window_hashes + position);
}

/** The eight keys of a register in reverse order. */
SZ_API_COMPTIME __m256i sz_overlap_haswell_reverse_(__m256i keys_vec) {
    return _mm256_permutevar8x32_epi32(keys_vec, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
}

/** Compare-exchange at distance one inside a register: the odd positions take the larger key. */
SZ_API_COMPTIME __m256i sz_overlap_haswell_exchange_at_1_(__m256i keys_vec) {
    __m256i const partners_vec = _mm256_shuffle_epi32(keys_vec, 0xB1);
    return _mm256_blend_epi32(_mm256_min_epu32(keys_vec, partners_vec), _mm256_max_epu32(keys_vec, partners_vec), 0xAA);
}

/** Compare-exchange at distance two inside a register. */
SZ_API_COMPTIME __m256i sz_overlap_haswell_exchange_at_2_(__m256i keys_vec) {
    __m256i const partners_vec = _mm256_shuffle_epi32(keys_vec, 0x4E);
    return _mm256_blend_epi32(_mm256_min_epu32(keys_vec, partners_vec), _mm256_max_epu32(keys_vec, partners_vec), 0xCC);
}

/** Compare-exchange at distance four inside a register. */
SZ_API_COMPTIME __m256i sz_overlap_haswell_exchange_at_4_(__m256i keys_vec) {
    __m256i const partners_vec = _mm256_permute2x128_si256(keys_vec, keys_vec, 0x01);
    return _mm256_blend_epi32(_mm256_min_epu32(keys_vec, partners_vec), _mm256_max_epu32(keys_vec, partners_vec), 0xF0);
}

/** The mirrored stage opening a merge of two-key runs into four: key @c i against key @c 3-i. */
SZ_API_COMPTIME __m256i sz_overlap_haswell_mirror_over_4_(__m256i keys_vec) {
    __m256i const partners_vec = _mm256_shuffle_epi32(keys_vec, 0x1B);
    return _mm256_blend_epi32(_mm256_min_epu32(keys_vec, partners_vec), _mm256_max_epu32(keys_vec, partners_vec), 0xCC);
}

/** The mirrored stage opening a merge of four-key runs into eight: key @c i against key @c 7-i. */
SZ_API_COMPTIME __m256i sz_overlap_haswell_mirror_over_8_(__m256i keys_vec) {
    __m256i const partners_vec = sz_overlap_haswell_reverse_(keys_vec);
    return _mm256_blend_epi32(_mm256_min_epu32(keys_vec, partners_vec), _mm256_max_epu32(keys_vec, partners_vec), 0xF0);
}

/** Two registers a fixed distance apart: @p lower keeps the minima, @p upper the maxima. */
SZ_HELPER_INLINE void sz_overlap_haswell_exchange_(__m256i *lower_vec, __m256i *upper_vec) {
    __m256i const smaller_vec = _mm256_min_epu32(*lower_vec, *upper_vec);
    __m256i const larger_vec = _mm256_max_epu32(*lower_vec, *upper_vec);
    *lower_vec = smaller_vec, *upper_vec = larger_vec;
}

/** The mirrored stage opening a merge of two ascending runs: @p upper is read and written reversed. */
SZ_HELPER_INLINE void sz_overlap_haswell_exchange_mirrored_(__m256i *lower_vec, __m256i *upper_vec) {
    __m256i const reversed_vec = sz_overlap_haswell_reverse_(*upper_vec);
    __m256i const smaller_vec = _mm256_min_epu32(*lower_vec, reversed_vec);
    __m256i const larger_vec = _mm256_max_epu32(*lower_vec, reversed_vec);
    *lower_vec = smaller_vec, *upper_vec = sz_overlap_haswell_reverse_(larger_vec);
}

/** Sorts the eight keys of one register ascending: the merges of two, four and eight keys, all inside it. */
SZ_API_COMPTIME __m256i sz_overlap_haswell_sort_within_(__m256i keys_vec) {
    keys_vec = sz_overlap_haswell_exchange_at_1_(keys_vec);
    keys_vec = sz_overlap_haswell_mirror_over_4_(keys_vec);
    keys_vec = sz_overlap_haswell_exchange_at_1_(keys_vec);
    keys_vec = sz_overlap_haswell_mirror_over_8_(keys_vec);
    keys_vec = sz_overlap_haswell_exchange_at_2_(keys_vec);
    return sz_overlap_haswell_exchange_at_1_(keys_vec);
}

/** The ascending half-cleaners at distances four, two and one, closing a merge inside the register. */
SZ_API_COMPTIME __m256i sz_overlap_haswell_merge_within_(__m256i keys_vec) {
    keys_vec = sz_overlap_haswell_exchange_at_4_(keys_vec);
    keys_vec = sz_overlap_haswell_exchange_at_2_(keys_vec);
    return sz_overlap_haswell_exchange_at_1_(keys_vec);
}

/** Loads the eight registers of one 64-key run. */
SZ_HELPER_INLINE void sz_overlap_haswell_load_run_(sz_u32_t const *keys, __m256i *registers_vec) {
    registers_vec[0] = _mm256_loadu_si256((__m256i const *)(keys + 0));
    registers_vec[1] = _mm256_loadu_si256((__m256i const *)(keys + 8));
    registers_vec[2] = _mm256_loadu_si256((__m256i const *)(keys + 16));
    registers_vec[3] = _mm256_loadu_si256((__m256i const *)(keys + 24));
    registers_vec[4] = _mm256_loadu_si256((__m256i const *)(keys + 32));
    registers_vec[5] = _mm256_loadu_si256((__m256i const *)(keys + 40));
    registers_vec[6] = _mm256_loadu_si256((__m256i const *)(keys + 48));
    registers_vec[7] = _mm256_loadu_si256((__m256i const *)(keys + 56));
}

/** Stores the eight registers of one 64-key run. */
SZ_HELPER_INLINE void sz_overlap_haswell_store_run_(__m256i const *registers_vec, sz_u32_t *keys) {
    _mm256_storeu_si256((__m256i *)(keys + 0), registers_vec[0]);
    _mm256_storeu_si256((__m256i *)(keys + 8), registers_vec[1]);
    _mm256_storeu_si256((__m256i *)(keys + 16), registers_vec[2]);
    _mm256_storeu_si256((__m256i *)(keys + 24), registers_vec[3]);
    _mm256_storeu_si256((__m256i *)(keys + 32), registers_vec[4]);
    _mm256_storeu_si256((__m256i *)(keys + 40), registers_vec[5]);
    _mm256_storeu_si256((__m256i *)(keys + 48), registers_vec[6]);
    _mm256_storeu_si256((__m256i *)(keys + 56), registers_vec[7]);
}

/** Closes a merge inside every register of a run. */
SZ_HELPER_INLINE void sz_overlap_haswell_merge_within_run_(__m256i *registers_vec) {
    registers_vec[0] = sz_overlap_haswell_merge_within_(registers_vec[0]);
    registers_vec[1] = sz_overlap_haswell_merge_within_(registers_vec[1]);
    registers_vec[2] = sz_overlap_haswell_merge_within_(registers_vec[2]);
    registers_vec[3] = sz_overlap_haswell_merge_within_(registers_vec[3]);
    registers_vec[4] = sz_overlap_haswell_merge_within_(registers_vec[4]);
    registers_vec[5] = sz_overlap_haswell_merge_within_(registers_vec[5]);
    registers_vec[6] = sz_overlap_haswell_merge_within_(registers_vec[6]);
    registers_vec[7] = sz_overlap_haswell_merge_within_(registers_vec[7]);
}

/** Sorts one 64-key run ascending in eight registers: every stage is a fixed register pair or an immediate blend. */
SZ_API_COMPTIME void sz_overlap_haswell_sort_run_(sz_u32_t *keys) {
    __m256i registers_vec[8];
    sz_overlap_haswell_load_run_(keys, registers_vec);
    registers_vec[0] = sz_overlap_haswell_sort_within_(registers_vec[0]);
    registers_vec[1] = sz_overlap_haswell_sort_within_(registers_vec[1]);
    registers_vec[2] = sz_overlap_haswell_sort_within_(registers_vec[2]);
    registers_vec[3] = sz_overlap_haswell_sort_within_(registers_vec[3]);
    registers_vec[4] = sz_overlap_haswell_sort_within_(registers_vec[4]);
    registers_vec[5] = sz_overlap_haswell_sort_within_(registers_vec[5]);
    registers_vec[6] = sz_overlap_haswell_sort_within_(registers_vec[6]);
    registers_vec[7] = sz_overlap_haswell_sort_within_(registers_vec[7]);
    // Sixteen keys: register pairs mirrored, then within.
    sz_overlap_haswell_exchange_mirrored_(&registers_vec[0], &registers_vec[1]);
    sz_overlap_haswell_exchange_mirrored_(&registers_vec[2], &registers_vec[3]);
    sz_overlap_haswell_exchange_mirrored_(&registers_vec[4], &registers_vec[5]);
    sz_overlap_haswell_exchange_mirrored_(&registers_vec[6], &registers_vec[7]);
    sz_overlap_haswell_merge_within_run_(registers_vec);
    // Thirty-two keys: quads mirrored, one stage of pairs, then within.
    sz_overlap_haswell_exchange_mirrored_(&registers_vec[0], &registers_vec[3]);
    sz_overlap_haswell_exchange_mirrored_(&registers_vec[1], &registers_vec[2]);
    sz_overlap_haswell_exchange_mirrored_(&registers_vec[4], &registers_vec[7]);
    sz_overlap_haswell_exchange_mirrored_(&registers_vec[5], &registers_vec[6]);
    sz_overlap_haswell_exchange_(&registers_vec[0], &registers_vec[1]);
    sz_overlap_haswell_exchange_(&registers_vec[2], &registers_vec[3]);
    sz_overlap_haswell_exchange_(&registers_vec[4], &registers_vec[5]);
    sz_overlap_haswell_exchange_(&registers_vec[6], &registers_vec[7]);
    sz_overlap_haswell_merge_within_run_(registers_vec);
    // Sixty-four keys: the whole run mirrored, the stages at distance sixteen and eight, then within.
    sz_overlap_haswell_exchange_mirrored_(&registers_vec[0], &registers_vec[7]);
    sz_overlap_haswell_exchange_mirrored_(&registers_vec[1], &registers_vec[6]);
    sz_overlap_haswell_exchange_mirrored_(&registers_vec[2], &registers_vec[5]);
    sz_overlap_haswell_exchange_mirrored_(&registers_vec[3], &registers_vec[4]);
    sz_overlap_haswell_exchange_(&registers_vec[0], &registers_vec[2]);
    sz_overlap_haswell_exchange_(&registers_vec[1], &registers_vec[3]);
    sz_overlap_haswell_exchange_(&registers_vec[4], &registers_vec[6]);
    sz_overlap_haswell_exchange_(&registers_vec[5], &registers_vec[7]);
    sz_overlap_haswell_exchange_(&registers_vec[0], &registers_vec[1]);
    sz_overlap_haswell_exchange_(&registers_vec[2], &registers_vec[3]);
    sz_overlap_haswell_exchange_(&registers_vec[4], &registers_vec[5]);
    sz_overlap_haswell_exchange_(&registers_vec[6], &registers_vec[7]);
    sz_overlap_haswell_merge_within_run_(registers_vec);
    sz_overlap_haswell_store_run_(registers_vec, keys);
}

/** Closes a merge inside one 64-key run: the ascending stages at distances thirty-two down to one. */
SZ_API_COMPTIME void sz_overlap_haswell_merge_run_(sz_u32_t *keys) {
    __m256i registers_vec[8];
    sz_overlap_haswell_load_run_(keys, registers_vec);
    sz_overlap_haswell_exchange_(&registers_vec[0], &registers_vec[4]);
    sz_overlap_haswell_exchange_(&registers_vec[1], &registers_vec[5]);
    sz_overlap_haswell_exchange_(&registers_vec[2], &registers_vec[6]);
    sz_overlap_haswell_exchange_(&registers_vec[3], &registers_vec[7]);
    sz_overlap_haswell_exchange_(&registers_vec[0], &registers_vec[2]);
    sz_overlap_haswell_exchange_(&registers_vec[1], &registers_vec[3]);
    sz_overlap_haswell_exchange_(&registers_vec[4], &registers_vec[6]);
    sz_overlap_haswell_exchange_(&registers_vec[5], &registers_vec[7]);
    sz_overlap_haswell_exchange_(&registers_vec[0], &registers_vec[1]);
    sz_overlap_haswell_exchange_(&registers_vec[2], &registers_vec[3]);
    sz_overlap_haswell_exchange_(&registers_vec[4], &registers_vec[5]);
    sz_overlap_haswell_exchange_(&registers_vec[6], &registers_vec[7]);
    sz_overlap_haswell_merge_within_run_(registers_vec);
    sz_overlap_haswell_store_run_(registers_vec, keys);
}

/** Sorts @p count keys ascending in place, unsigned, and drops repeats, answering how many remain; the buffer
 *  holds @ref sz_overlap_btree_sorted_capacity entries. */
SZ_API_COMPTIME sz_size_t sz_overlap_u32x8_btree_sort_haswell(sz_u32_t *keys, sz_size_t count) {
    sz_size_t const keys_per_register = sz_overlap_haswell_keys_per_register_k;
    sz_size_t const keys_per_run = sz_overlap_haswell_keys_per_run_k;
    sz_size_t const capacity = sz_overlap_btree_sorted_capacity(count);
    for (sz_size_t position = count; position != capacity; ++position) keys[position] = sz_overlap_padding_key_k;
    for (sz_size_t run = 0; run != capacity / keys_per_run; ++run)
        sz_overlap_haswell_sort_run_(keys + run * keys_per_run);

    // Every merge opens mirrored over memory, so no run is ever descending; the stages that reach past a run
    // exchange whole registers in memory, and the rest close inside one run's registers.
    for (sz_size_t phase = 2 * keys_per_run; phase <= capacity; phase *= 2) {
        for (sz_size_t start = 0; start != capacity; start += phase)
            for (sz_size_t offset = 0; offset != phase / 2; offset += keys_per_register) {
                __m256i lower_vec = _mm256_loadu_si256((__m256i const *)(keys + start + offset));
                __m256i upper_vec = _mm256_loadu_si256(
                    (__m256i const *)(keys + start + phase - keys_per_register - offset));
                sz_overlap_haswell_exchange_mirrored_(&lower_vec, &upper_vec);
                _mm256_storeu_si256((__m256i *)(keys + start + offset), lower_vec);
                _mm256_storeu_si256((__m256i *)(keys + start + phase - keys_per_register - offset), upper_vec);
            }
        for (sz_size_t distance = phase / 4; distance >= keys_per_run; distance /= 2)
            for (sz_size_t start = 0; start != capacity; start += 2 * distance)
                for (sz_size_t offset = 0; offset != distance; offset += keys_per_register) {
                    __m256i lower_vec = _mm256_loadu_si256((__m256i const *)(keys + start + offset));
                    __m256i upper_vec = _mm256_loadu_si256((__m256i const *)(keys + start + distance + offset));
                    sz_overlap_haswell_exchange_(&lower_vec, &upper_vec);
                    _mm256_storeu_si256((__m256i *)(keys + start + offset), lower_vec);
                    _mm256_storeu_si256((__m256i *)(keys + start + distance + offset), upper_vec);
                }
        for (sz_size_t run = 0; run != capacity / keys_per_run; ++run)
            sz_overlap_haswell_merge_run_(keys + run * keys_per_run);
    }
    return sz_overlap_btree_unique_(keys, count);
}

/** One branch level: the child ordinal a flipped @p key_vec descends into, the count of separators below it. */
SZ_API_COMPTIME sz_size_t sz_overlap_haswell_branch_step_(sz_u32_t const *node, __m256i key_vec) {
    __m256i const first_below_vec = _mm256_cmpgt_epi32(key_vec, _mm256_loadu_si256((__m256i const *)node));
    __m256i const second_below_vec = _mm256_cmpgt_epi32(key_vec, _mm256_loadu_si256((__m256i const *)(node + 8)));
    unsigned const below_mask = (unsigned)_mm256_movemask_epi8(_mm256_packs_epi32(first_below_vec, second_below_vec));
    return (sz_size_t)_mm_popcnt_u32(below_mask) / 2;
}

/** The leaf compare: one when the flipped @p key_vec sits in this node, zero otherwise. */
SZ_API_COMPTIME sz_size_t sz_overlap_haswell_leaf_step_(sz_u32_t const *node, __m256i key_vec) {
    __m256i const first_equal_vec = _mm256_cmpeq_epi32(key_vec, _mm256_loadu_si256((__m256i const *)node));
    __m256i const second_equal_vec = _mm256_cmpeq_epi32(key_vec, _mm256_loadu_si256((__m256i const *)(node + 8)));
    return _mm256_movemask_epi8(_mm256_or_si256(first_equal_vec, second_equal_vec)) != 0;
}

/** Walks the whole tree for one flipped key, the root included. */
SZ_API_COMPTIME sz_size_t sz_overlap_haswell_probe_key_(sz_overlap_btree_t const *btree, sz_u32_t flipped_key) {
    sz_size_t const keys_per_node = sz_overlap_keys_per_node_k, branches_per_node = sz_overlap_branches_per_node_k;
    __m256i const key_vec = _mm256_set1_epi32((int)flipped_key);
    sz_size_t node = 0;
    for (sz_size_t level = 0; level + 1 != btree->levels; ++level)
        node = node * branches_per_node +
               sz_overlap_haswell_branch_step_(btree->nodes + (btree->level_bases[level] + node) * keys_per_node,
                                               key_vec);
    return sz_overlap_haswell_leaf_step_(btree->nodes + (btree->level_bases[btree->levels - 1] + node) * keys_per_node,
                                         key_vec);
}

/** Counts how many of @p count raw keys of one candidate the tree holds. */
SZ_API_COMPTIME sz_size_t sz_overlap_u32x8_btree_probe_haswell(sz_overlap_btree_t const *btree, sz_u32_t const *keys,
                                                               sz_size_t count) {
    sz_size_t const keys_per_node = sz_overlap_keys_per_node_k, branches_per_node = sz_overlap_branches_per_node_k;
    sz_size_t const keys_per_register = sz_overlap_haswell_keys_per_register_k;
    sz_u32_t const *const root = btree->nodes + btree->level_bases[0] * keys_per_node;
    sz_u32_t const *const leaves = btree->nodes + btree->level_bases[btree->levels - 1] * keys_per_node;
    __m256i const flip_vec = _mm256_set1_epi32((int)sz_overlap_sign_flip_k);
    sz_size_t matches = 0, index = 0;

    // Eight keys meet the root transposed, each separator broadcast against all eight; the first eight separators
    // stay in registers and the rest broadcast from memory. The deeper levels are eight interleaved walks.
    __m256i root_vecs[8];
    for (sz_size_t separator = 0; separator != 8; ++separator)
        root_vecs[separator] = _mm256_set1_epi32((int)root[separator]);
    if (btree->levels > 1)
        for (; index + keys_per_register <= count; index += keys_per_register) {
            __m256i const keys_vec = _mm256_xor_si256(_mm256_loadu_si256((__m256i const *)(keys + index)), flip_vec);
            __m256i children_vec = _mm256_setzero_si256();
            for (sz_size_t separator = 0; separator != 8; ++separator)
                children_vec = _mm256_sub_epi32(children_vec, _mm256_cmpgt_epi32(keys_vec, root_vecs[separator]));
            for (sz_size_t separator = 8; separator != keys_per_node; ++separator)
                children_vec = _mm256_sub_epi32(children_vec,
                                                _mm256_cmpgt_epi32(keys_vec, _mm256_set1_epi32((int)root[separator])));

            sz_u32_t flipped_keys[8], walks[8];
            _mm256_storeu_si256((__m256i *)flipped_keys, keys_vec);
            _mm256_storeu_si256((__m256i *)walks, children_vec);
            for (sz_size_t level = 1; level + 1 != btree->levels; ++level) {
                sz_u32_t const *const level_nodes = btree->nodes + btree->level_bases[level] * keys_per_node;
                for (sz_size_t walk = 0; walk != keys_per_register; ++walk)
                    walks[walk] = (sz_u32_t)(walks[walk] * branches_per_node +
                                             sz_overlap_haswell_branch_step_(
                                                 level_nodes + walks[walk] * keys_per_node,
                                                 _mm256_set1_epi32((int)flipped_keys[walk])));
            }
            for (sz_size_t walk = 0; walk != keys_per_register; ++walk)
                matches += sz_overlap_haswell_leaf_step_(leaves + walks[walk] * keys_per_node,
                                                         _mm256_set1_epi32((int)flipped_keys[walk]));
        }
    for (; index != count; ++index)
        matches += sz_overlap_haswell_probe_key_(btree, keys[index] ^ sz_overlap_sign_flip_k);
    return matches;
}

/**
 *  @brief Prepares every query of @p queries into one block, hashing and sorting on the Haswell tier.
 *  @param[in] alloc Where the forest's block comes from, or @c SZ_NULL for the default host allocator.
 *  @sa sz_overlap_engine_init_cpu
 */
SZ_API_COMPTIME sz_status_t sz_overlap_engine_init_haswell(sz_sequence_t const *queries,
                                                           sz_size_t const *window_widths,
                                                           sz_size_t window_widths_count,
                                                           sz_memory_allocator_t *alloc,
                                                           sz_overlap_engine_t *engine) {
    sz_size_t const step = sz_overlap_haswell_f64x4_positions_per_step_k;
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
            prior = sz_overlap_f64x4_prefix_hash_step_haswell(prior, text + position, chain + position + 1);
        if (position != length)
            sz_overlap_f64x4_prefix_hash_step_tail_haswell(prior, text + position, length - position,
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
                sz_overlap_f64x4_window_hash_step_haswell(chain + window, chain + window + width, power,
                                                          arena + written + window);
            if (window != query_windows)
                sz_overlap_f64x4_window_hash_step_tail_haswell(chain + window, chain + window + width, power,
                                                               query_windows - window, arena + written + window);
            written += query_windows;
        }
        sz_overlap_btree_t btree;
        sz_size_t const distinct = sz_overlap_u32x8_btree_sort_haswell(arena, written);
        sz_overlap_btree_prepare(arena, distinct, &btree);
        keys_counts[index] = (sz_u32_t)distinct;
    }

    engine->capability = sz_cap_haswell_k;
    return sz_success_k;
}

SZ_API_COMPTIME sz_status_t sz_overlap_scores_haswell(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                                      sz_f32_t *scores, sz_size_t scores_query_stride,
                                                      sz_size_t scores_candidate_stride) {
    sz_status_t const dimensions = sz_overlap_engine_strides_(engine, candidates->count, scores_query_stride,
                                                              scores_candidate_stride);
    if (dimensions != sz_success_k) return dimensions;
    if (!candidates->count || !engine->count) return sz_success_k;
    sz_size_t const step = sz_overlap_haswell_f64x4_positions_per_step_k;
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
        sz_size_t shortest = SZ_SIZE_MAX;
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
                    priors[chain] = sz_overlap_f64x4_prefix_hash_step_haswell(
                        priors[chain], texts[chain] + walked, prefix_hashes + chain * chain_stride + walked + 1);
        for (sz_size_t chain = 0; chain != interleaved; ++chain) {
            sz_f64_t *const chain_prefix_hashes = prefix_hashes + chain * chain_stride;
            sz_f64_t running = priors[chain];
            sz_size_t own = walked;
            for (; own + step <= lengths[chain]; own += step)
                running = sz_overlap_f64x4_prefix_hash_step_haswell(running, texts[chain] + own,
                                                                    chain_prefix_hashes + own + 1);
            if (own != lengths[chain])
                sz_overlap_f64x4_prefix_hash_step_tail_haswell(running, texts[chain] + own, lengths[chain] - own,
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
                    sz_overlap_f64x4_window_hash_step_haswell(chain_prefix_hashes + window,
                                                              chain_prefix_hashes + window + width, power,
                                                              window_hashes + window);
                if (window != windows)
                    sz_overlap_f64x4_window_hash_step_tail_haswell(chain_prefix_hashes + window,
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
                    sz_size_t const matches = sz_overlap_u32x8_btree_probe_haswell(&btree, window_hashes, windows);
                    *slot = sz_overlap_share_(matches, windows, query_length - width + 1);
                }
            }
        }
    }
    return sz_success_k;
}

#pragma endregion Haswell

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_OVERLAP_HASWELL_H_
