/**
 *  @file include/stringzillas/fingerprints/haswell.h
 *  @brief AVX2 (Haswell) rolling-hash Min-Hash / Count-Min fingerprint kernels.
 *  @author Ash Vardanian
 *  @sa include/stringzillas/fingerprints/serial.h
 *
 *  Processes 4x f64 per YMM register. Mirrors `floating_rolling_hashers<sz_cap_haswell_k, dims>::roll_group`
 *  EXACTLY, including the magic-number floor, the fused `fnmadd` Barrett reduction, and the blend-based
 *  Min-Hash / Count-Min update. Falls through to the serial sweep for the incomplete tail group's scalars.
 */
#ifndef STRINGZILLAS_FINGERPRINTS_HASWELL_H_
#define STRINGZILLAS_FINGERPRINTS_HASWELL_H_

#include "stringzillas/fingerprints/serial.h"

#if SZ_USE_HASWELL
#ifdef __cplusplus
extern "C" {
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "fma")
#endif

#pragma region - AVX2 Helpers

SZ_INTERNAL __m256d szs_fh_floor_magic_f64x4_(__m256d x) {
    __m256d const magic_f64x4 = _mm256_set1_pd(6755399441055744.0); // 2^52 + 2^51
    __m256d rounded_f64x4 = _mm256_sub_pd(_mm256_add_pd(x, magic_f64x4), magic_f64x4);
    __m256d neg_mask_f64x4 = _mm256_cmp_pd(rounded_f64x4, x, _CMP_GT_OQ);
    return _mm256_sub_pd(rounded_f64x4, _mm256_and_pd(neg_mask_f64x4, _mm256_set1_pd(1.0)));
}

SZ_INTERNAL __m256d szs_fh_barrett_mod_f64x4_(__m256d xs_f64x4, __m256d modulos_f64x4, __m256d inverse_modulos_f64x4) {
    __m256d qs_f64x4 = szs_fh_floor_magic_f64x4_(_mm256_mul_pd(xs_f64x4, inverse_modulos_f64x4));
    __m256d results_f64x4 = _mm256_fnmadd_pd(qs_f64x4, modulos_f64x4, xs_f64x4);
    __m256d overflow_mask_f64x4 = _mm256_cmp_pd(results_f64x4, modulos_f64x4, _CMP_GE_OQ);
    results_f64x4 = _mm256_sub_pd(results_f64x4, _mm256_and_pd(overflow_mask_f64x4, modulos_f64x4));
    __m256d negative_mask_f64x4 = _mm256_cmp_pd(results_f64x4, _mm256_setzero_pd(), _CMP_LT_OQ);
    results_f64x4 = _mm256_add_pd(results_f64x4, _mm256_and_pd(negative_mask_f64x4, modulos_f64x4));
    return results_f64x4;
}

#pragma endregion

#pragma region - AVX2 Sweep Megakernel

SZ_INTERNAL void szs_fingerprints_sweep_haswell_(         //
    sz_byte_t const *text, sz_size_t text_length,         //
    sz_fh_dim_t const *dims, sz_size_t dimensions,        //
    sz_size_t window_width,                               //
    sz_f64_t *rolling_states, sz_f64_t *rolling_minimums, //
    sz_u32_t *min_hashes, sz_u32_t *min_counts) {

    sz_size_t const hashes_per_ymm_k = 4;
    sz_size_t const groups_count = (dimensions + hashes_per_ymm_k - 1) / hashes_per_ymm_k;
    sz_size_t dim, group_index;

    if (text_length < window_width) {
        for (dim = 0; dim < dimensions; ++dim) min_hashes[dim] = SZ_FH_MAX_HASH_K;
        for (dim = 0; dim < dimensions; ++dim) min_counts[dim] = 0;
        return;
    }

    for (dim = 0; dim < dimensions; ++dim)
        rolling_states[dim] = 0.0, rolling_minimums[dim] = SZ_FH_SKIPPED_ROLLING_HASH_K, min_counts[dim] = 0;

    for (group_index = 0; group_index < groups_count; ++group_index) {
        sz_size_t const first_dim = group_index * hashes_per_ymm_k;
        sz_size_t const group_dims = dimensions - first_dim < hashes_per_ymm_k ? dimensions - first_dim
                                                                               : hashes_per_ymm_k;

        sz_u256_vec_t last_states_vec, rolling_minimums_vec, rolling_counts_vec;
        sz_u256_vec_t multipliers_vec, negative_discarding_multipliers_vec, modulos_vec, inverse_modulos_vec;
        sz_size_t new_char_offset = 0;
        sz_size_t const prefix_length = text_length < window_width ? text_length : window_width;
        __m256i const ones_i64x4 = _mm256_set1_epi64x(1);
        sz_size_t word_index;

        // Gather this register's coefficients; pad lanes clamp to the last live dim (masked out on store).
        for (word_index = 0; word_index < hashes_per_ymm_k; ++word_index) {
            sz_size_t const local = first_dim + (word_index < group_dims ? word_index : group_dims - 1);
            multipliers_vec.f64s[word_index] = dims[local].multiplier;
            negative_discarding_multipliers_vec.f64s[word_index] = dims[local].negative_discarding_multiplier;
            modulos_vec.f64s[word_index] = dims[local].modulo;
            inverse_modulos_vec.f64s[word_index] = dims[local].inverse_modulo;
        }

        for (word_index = 0; word_index < hashes_per_ymm_k; ++word_index) {
            last_states_vec.f64s[word_index] = 0.0;
            rolling_minimums_vec.f64s[word_index] = SZ_FH_SKIPPED_ROLLING_HASH_K;
            rolling_counts_vec.u64s[word_index] = 0;
        }

        // Branching prefix: only push.
        for (; new_char_offset < prefix_length; ++new_char_offset) {
            sz_byte_t const new_char = text[new_char_offset];
            sz_f64_t const new_term = (sz_f64_t)new_char + 1.0;
            __m256d new_term_f64x4 = _mm256_set1_pd(new_term);
            last_states_vec.ymm_pd = _mm256_fmadd_pd(last_states_vec.ymm_pd, multipliers_vec.ymm_pd, new_term_f64x4);
            last_states_vec.ymm_pd = szs_fh_barrett_mod_f64x4_(last_states_vec.ymm_pd, modulos_vec.ymm_pd,
                                                               inverse_modulos_vec.ymm_pd);
        }

        if (new_char_offset == window_width)
            rolling_minimums_vec.ymm_pd = last_states_vec.ymm_pd, rolling_counts_vec.ymm = ones_i64x4;

        // Branchless central loop.
        for (; new_char_offset < text_length; ++new_char_offset) {
            sz_byte_t const new_char = text[new_char_offset];
            sz_byte_t const old_char = text[new_char_offset - window_width];
            sz_f64_t const new_term = (sz_f64_t)new_char + 1.0;
            sz_f64_t const old_term = (sz_f64_t)old_char + 1.0;
            __m256d new_term_f64x4 = _mm256_set1_pd(new_term);
            __m256d old_term_f64x4 = _mm256_set1_pd(old_term);

            last_states_vec.ymm_pd = _mm256_fmadd_pd(negative_discarding_multipliers_vec.ymm_pd, old_term_f64x4,
                                                     last_states_vec.ymm_pd);
            last_states_vec.ymm_pd = szs_fh_barrett_mod_f64x4_(last_states_vec.ymm_pd, modulos_vec.ymm_pd,
                                                               inverse_modulos_vec.ymm_pd);
            last_states_vec.ymm_pd = _mm256_fmadd_pd(last_states_vec.ymm_pd, multipliers_vec.ymm_pd, new_term_f64x4);
            last_states_vec.ymm_pd = szs_fh_barrett_mod_f64x4_(last_states_vec.ymm_pd, modulos_vec.ymm_pd,
                                                               inverse_modulos_vec.ymm_pd);

            __m256d found_f64x4 = _mm256_cmp_pd(last_states_vec.ymm_pd, rolling_minimums_vec.ymm_pd, _CMP_LE_OQ);
            __m256d discard_f64x4 = _mm256_cmp_pd(last_states_vec.ymm_pd, rolling_minimums_vec.ymm_pd, _CMP_GE_OQ);
            rolling_minimums_vec.ymm_pd = _mm256_blendv_pd(rolling_minimums_vec.ymm_pd, last_states_vec.ymm_pd,
                                                           found_f64x4);
            rolling_counts_vec.ymm_pd = _mm256_blendv_pd(_mm256_setzero_pd(), rolling_counts_vec.ymm_pd, discard_f64x4);
            rolling_counts_vec.ymm_pd = _mm256_blendv_pd(
                rolling_counts_vec.ymm_pd, _mm256_castsi256_pd(_mm256_add_epi64(rolling_counts_vec.ymm, ones_i64x4)),
                found_f64x4);
        }

        // Scatter the live lanes back into the per-dimension buffers.
        for (word_index = 0; word_index < group_dims; ++word_index) {
            rolling_states[first_dim + word_index] = last_states_vec.f64s[word_index];
            rolling_minimums[first_dim + word_index] = rolling_minimums_vec.f64s[word_index];
            min_counts[first_dim + word_index] = (sz_u32_t)rolling_counts_vec.u64s[word_index];
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

#pragma region - Haswell Entry

SZ_INTERNAL sz_status_t szs_fingerprints_haswell(                          //
    szs_fingerprints_engine_t const *engine, szs_device_scope_impl_t *device, //
    szs_sequence_view_t inputs,                                            //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,                     //
    sz_u32_t *min_counts, sz_size_t min_counts_stride,                     //
    char const **error) {

    sz_memory_allocator_t fallback_alloc;
    sz_memory_allocator_t *alloc = szs_fingerprints_resolve_alloc_((sz_memory_allocator_t *)&engine->alloc,
                                                                   &fallback_alloc);
    sz_size_t const count = szs_fingerprints_count_(&inputs);
    szs_fingerprints_ctx_t ctx = szs_fingerprints_make_ctx_(engine, inputs, alloc, &szs_fingerprints_sweep_haswell_,
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
#endif // SZ_USE_HASWELL

#endif // STRINGZILLAS_FINGERPRINTS_HASWELL_H_
