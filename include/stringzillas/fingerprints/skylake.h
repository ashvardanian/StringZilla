/**
 *  @file include/stringzillas/fingerprints/skylake.h
 *  @brief AVX-512 (Skylake) rolling-hash Min-Hash / Count-Min fingerprint kernels.
 *  @author Ash Vardanian
 *  @sa include/stringzillas/fingerprints/haswell.h
 *
 *  Processes 8x f64 per ZMM register. Mirrors `floating_rolling_hashers<sz_cap_skylake_k, dims>::roll_group`
 *  EXACTLY, including the magic-number floor, the fused `fnmadd` Barrett reduction with `fpclass`-masked
 *  negative clamp, and the K-mask-based Min-Hash / Count-Min update over a 32-bit count vector.
 */
#ifndef STRINGZILLAS_FINGERPRINTS_SKYLAKE_H_
#define STRINGZILLAS_FINGERPRINTS_SKYLAKE_H_

#include "stringzillas/fingerprints/haswell.h"

#if SZ_USE_SKYLAKE
#ifdef __cplusplus
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512dq,avx512bw,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512dq", "avx512bw", "bmi", "bmi2")
#endif

#pragma region - AVX-512 Helpers

SZ_INTERNAL __m512d szs_fh_floor_magic_f64x8_(__m512d x) {
    __m512d const magic_f64x8 = _mm512_set1_pd(6755399441055744.0); // 2^52 + 2^51
    __m512d rounded_f64x8 = _mm512_sub_pd(_mm512_add_pd(x, magic_f64x8), magic_f64x8);
    __mmask8 neg_mask = _mm512_cmp_pd_mask(rounded_f64x8, x, _CMP_GT_OQ);
    return _mm512_mask_sub_pd(rounded_f64x8, neg_mask, rounded_f64x8, _mm512_set1_pd(1.0));
}

SZ_INTERNAL __m512d szs_fh_barrett_mod_f64x8_(__m512d xs_f64x8, __m512d modulos_f64x8, __m512d inverse_modulos_f64x8) {
    __m512d qs_f64x8 = szs_fh_floor_magic_f64x8_(_mm512_mul_pd(xs_f64x8, inverse_modulos_f64x8));
    __m512d results_f64x8 = _mm512_fnmadd_pd(qs_f64x8, modulos_f64x8, xs_f64x8);
    __mmask8 overflow_mask = _mm512_cmp_pd_mask(results_f64x8, modulos_f64x8, _CMP_GE_OQ);
    results_f64x8 = _mm512_mask_sub_pd(results_f64x8, overflow_mask, results_f64x8, modulos_f64x8);
    __mmask8 negative_mask = _mm512_fpclass_pd_mask(results_f64x8, 0x44); // Negative
    results_f64x8 = _mm512_mask_add_pd(results_f64x8, negative_mask, results_f64x8, modulos_f64x8);
    return results_f64x8;
}

#pragma endregion

#pragma region - AVX-512 Sweep Megakernel

SZ_INTERNAL void szs_fingerprints_sweep_skylake_(         //
    sz_byte_t const *text, sz_size_t text_length,         //
    sz_fh_dim_t const *dims, sz_size_t dimensions,        //
    sz_size_t window_width,                               //
    sz_f64_t *rolling_states, sz_f64_t *rolling_minimums, //
    sz_u32_t *min_hashes, sz_u32_t *min_counts) {

    sz_size_t const hashes_per_zmm_k = 8;
    sz_size_t const groups_count = (dimensions + hashes_per_zmm_k - 1) / hashes_per_zmm_k;
    sz_size_t dim, group_index;

    if (text_length < window_width) {
        for (dim = 0; dim < dimensions; ++dim) min_hashes[dim] = SZ_FH_MAX_HASH_K;
        for (dim = 0; dim < dimensions; ++dim) min_counts[dim] = 0;
        return;
    }

    for (dim = 0; dim < dimensions; ++dim)
        rolling_states[dim] = 0.0, rolling_minimums[dim] = SZ_FH_SKIPPED_ROLLING_HASH_K, min_counts[dim] = 0;

    for (group_index = 0; group_index < groups_count; ++group_index) {
        sz_size_t const first_dim = group_index * hashes_per_zmm_k;
        sz_size_t const group_dims = dimensions - first_dim < hashes_per_zmm_k ? dimensions - first_dim
                                                                               : hashes_per_zmm_k;

        sz_u512_vec_t last_states_vec, rolling_minimums_vec;
        sz_u256_vec_t rolling_counts_vec;
        sz_u512_vec_t multipliers_vec, negative_discarding_multipliers_vec, modulos_vec, inverse_modulos_vec;
        sz_size_t new_char_offset = 0;
        sz_size_t const prefix_length = text_length < window_width ? text_length : window_width;
        __m256i const ones_i32x8 = _mm256_set1_epi32(1);
        sz_size_t word_index;

        for (word_index = 0; word_index < hashes_per_zmm_k; ++word_index) {
            sz_size_t const local = first_dim + (word_index < group_dims ? word_index : group_dims - 1);
            multipliers_vec.f64s[word_index] = dims[local].multiplier;
            negative_discarding_multipliers_vec.f64s[word_index] = dims[local].negative_discarding_multiplier;
            modulos_vec.f64s[word_index] = dims[local].modulo;
            inverse_modulos_vec.f64s[word_index] = dims[local].inverse_modulo;
            last_states_vec.f64s[word_index] = 0.0;
            rolling_minimums_vec.f64s[word_index] = SZ_FH_SKIPPED_ROLLING_HASH_K;
            rolling_counts_vec.u32s[word_index] = 0;
        }

        for (; new_char_offset < prefix_length; ++new_char_offset) {
            sz_byte_t const new_char = text[new_char_offset];
            sz_f64_t const new_term = (sz_f64_t)new_char + 1.0;
            __m512d new_term_f64x8 = _mm512_set1_pd(new_term);
            last_states_vec.zmm_pd = _mm512_fmadd_pd(last_states_vec.zmm_pd, multipliers_vec.zmm_pd, new_term_f64x8);
            last_states_vec.zmm_pd = szs_fh_barrett_mod_f64x8_(last_states_vec.zmm_pd, modulos_vec.zmm_pd,
                                                               inverse_modulos_vec.zmm_pd);
        }

        if (new_char_offset == window_width)
            rolling_minimums_vec.zmm_pd = last_states_vec.zmm_pd, rolling_counts_vec.ymm = ones_i32x8;

        for (; new_char_offset < text_length; ++new_char_offset) {
            sz_byte_t const new_char = text[new_char_offset];
            sz_byte_t const old_char = text[new_char_offset - window_width];
            sz_f64_t const new_term = (sz_f64_t)new_char + 1.0;
            sz_f64_t const old_term = (sz_f64_t)old_char + 1.0;
            __m512d new_term_f64x8 = _mm512_set1_pd(new_term);
            __m512d old_term_f64x8 = _mm512_set1_pd(old_term);

            last_states_vec.zmm_pd = _mm512_fmadd_pd(negative_discarding_multipliers_vec.zmm_pd, old_term_f64x8,
                                                     last_states_vec.zmm_pd);
            last_states_vec.zmm_pd = szs_fh_barrett_mod_f64x8_(last_states_vec.zmm_pd, modulos_vec.zmm_pd,
                                                               inverse_modulos_vec.zmm_pd);
            last_states_vec.zmm_pd = _mm512_fmadd_pd(last_states_vec.zmm_pd, multipliers_vec.zmm_pd, new_term_f64x8);
            last_states_vec.zmm_pd = szs_fh_barrett_mod_f64x8_(last_states_vec.zmm_pd, modulos_vec.zmm_pd,
                                                               inverse_modulos_vec.zmm_pd);

            __mmask8 found_mask = _mm512_cmp_pd_mask(last_states_vec.zmm_pd, rolling_minimums_vec.zmm_pd, _CMP_LE_OQ);
            __mmask8 discard_mask = _mm512_cmp_pd_mask(last_states_vec.zmm_pd, rolling_minimums_vec.zmm_pd, _CMP_GE_OQ);
            rolling_minimums_vec.zmm_pd = _mm512_mask_mov_pd(rolling_minimums_vec.zmm_pd, found_mask,
                                                             last_states_vec.zmm_pd);
            rolling_counts_vec.ymm = _mm256_maskz_mov_epi32(discard_mask, rolling_counts_vec.ymm);
            rolling_counts_vec.ymm = _mm256_mask_add_epi32(rolling_counts_vec.ymm, found_mask, rolling_counts_vec.ymm,
                                                           ones_i32x8);
        }

        for (word_index = 0; word_index < group_dims; ++word_index) {
            rolling_states[first_dim + word_index] = last_states_vec.f64s[word_index];
            rolling_minimums[first_dim + word_index] = rolling_minimums_vec.f64s[word_index];
            min_counts[first_dim + word_index] = rolling_counts_vec.u32s[word_index];
        }
    }

    for (dim = 0; dim < dimensions; ++dim) {
        sz_f64_t const rolling_minimum = rolling_minimums[dim];
        sz_u64_t const rolling_minimum_as_uint = (sz_u64_t)rolling_minimum;
        min_hashes[dim] = rolling_minimum == SZ_FH_SKIPPED_ROLLING_HASH_K
                              ? SZ_FH_MAX_HASH_K
                              : (sz_u32_t)(rolling_minimum_as_uint & SZ_FH_MAX_HASH_K);
    }
}

#pragma endregion

#pragma region - Skylake Entry

SZ_INTERNAL sz_status_t szs_fingerprints_skylake(                          //
    szs_fingerprints_engine_t const *engine, szs_device_scope_impl_t *device, //
    szs_sequence_view_t inputs,                                            //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                     //
    sz_u32_t *min_counts, sz_size_t min_counts_stride,                     //
    char const **error) {

    sz_memory_allocator_t fallback_alloc;
    sz_memory_allocator_t *alloc = szs_fingerprints_resolve_alloc_((sz_memory_allocator_t *)&engine->alloc,
                                                                   &fallback_alloc);
    sz_size_t const count = szs_fingerprints_count_(&inputs);
    szs_fingerprints_ctx_t ctx = szs_fingerprints_make_ctx_(engine, inputs, alloc, &szs_fingerprints_sweep_skylake_,
                                                            min_hashes, min_hashes_stride, min_counts,
                                                            min_counts_stride, error);
    return szs_parallel_for_(device, count, &szs_fingerprints_body_, &ctx);
}

#pragma endregion

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#ifdef __cplusplus
} // extern "C"
#endif
#endif // SZ_USE_SKYLAKE

#endif // STRINGZILLAS_FINGERPRINTS_SKYLAKE_H_
