/**
 *  @brief AVX-512 (Ice Lake) string-similarity backend.
 *  @file include/stringzillas/similarities/icelake.hpp
 *  @author Ash Vardanian
 *  @sa include/stringzillas/similarities/serial.hpp
 */
#ifndef STRINGZILLAS_SIMILARITIES_ICELAKE_HPP_
#define STRINGZILLAS_SIMILARITIES_ICELAKE_HPP_

#include "stringzillas/similarities/serial.hpp"

namespace ashvardanian {
namespace stringzillas {

/*  AVX512 implementation of the string similarity algorithms for Ice Lake and newer CPUs.
 *  Includes extensions:
 *      - 2017 Skylake: F, CD, ER, PF, VL, DQ, BW,
 *      - 2018 CannonLake: IFMA, VBMI,
 *      - 2019 Ice Lake: VPOPCNTDQ, VNNI, VBMI2, BITALG, GFNI, VPCLMULQDQ, VAES.
 */
#pragma region Ice Lake Implementation
#if SZ_USE_ICELAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,bmi,bmi2"))), \
                             apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "bmi", "bmi2")
#endif

/**
 *  @brief Variant of `tile_scorer` - Minimizes Levenshtein distance for inputs under 256 bytes.
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, sz_u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, sz_u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, sz_u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 64;

    /**
     *  @brief Computes one diagonal of the `u8` DM matrix for exactly 64 characters,
     *         using unaligned loads, but forcing @b aligned stores.
     */
    SZ_INLINE void slice_aligned64chars(                                             //
        char const *first_reversed_slice, char const *second_slice,                  //
        sz_u8_t const *scores_pre_substitution, sz_u8_t const *scores_pre_insertion, //
        sz_u8_t const *scores_pre_deletion, sz_u8_t *scores_new,                     //
        sz_u512_vec_t match_cost_vec, sz_u512_vec_t mismatch_cost_vec,               //
        sz_u512_vec_t gap_cost_vec) const noexcept {

        __mmask64 match_mask;
        sz_u512_vec_t first_vec, second_vec;
        sz_u512_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        sz_u512_vec_t cost_of_substitution_vec;
        sz_u512_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

        // ? Note that here we are still traversing both buffers in the same order,
        // ? because one of the strings has been reversed beforehand.
        first_vec.zmm = _mm512_loadu_epi8(first_reversed_slice);
        second_vec.zmm = _mm512_loadu_epi8(second_slice);
        pre_substitution_vec.zmm = _mm512_loadu_epi8(scores_pre_substitution);
        pre_insert_vec.zmm = _mm512_loadu_epi8(scores_pre_insertion);
        pre_delete_vec.zmm = _mm512_loadu_epi8(scores_pre_deletion);

        match_mask = _mm512_cmpeq_epi8_mask(first_vec.zmm, second_vec.zmm);
        cost_of_substitution_vec.zmm = _mm512_mask_blend_epi8(match_mask, mismatch_cost_vec.zmm, match_cost_vec.zmm);
        cost_if_substitution_vec.zmm = _mm512_add_epi8(pre_substitution_vec.zmm, cost_of_substitution_vec.zmm);
        cost_if_gap_vec.zmm =
            _mm512_add_epi8(_mm512_min_epu8(pre_insert_vec.zmm, pre_delete_vec.zmm), gap_cost_vec.zmm);
        cell_score_vec.zmm = _mm512_min_epu8(cost_if_substitution_vec.zmm, cost_if_gap_vec.zmm);
        _mm512_store_si512(scores_new, cell_score_vec.zmm);
    }

    /**
     *  @brief Computes one diagonal of the `u8` DM matrix for up to 64 characters,
     *         using unaligned loads and stores.
     */
    SZ_INLINE void slice_upto64chars(                                                //
        char const *first_reversed_slice, char const *second_slice, size_t n,        //
        sz_u8_t const *scores_pre_substitution, sz_u8_t const *scores_pre_insertion, //
        sz_u8_t const *scores_pre_deletion, sz_u8_t *scores_new,                     //
        sz_u512_vec_t match_cost_vec, sz_u512_vec_t mismatch_cost_vec,               //
        sz_u512_vec_t gap_cost_vec) const noexcept {

        __mmask64 load_mask, match_mask;
        sz_u512_vec_t first_vec, second_vec;
        sz_u512_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        sz_u512_vec_t cost_of_substitution_vec;
        sz_u512_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

        // ? Note that here we are still traversing both buffers in the same order,
        // ? because one of the strings has been reversed beforehand.
        load_mask = sz_u64_mask_until_(n);
        first_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, first_reversed_slice);
        second_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, second_slice);
        pre_substitution_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, scores_pre_substitution);
        pre_insert_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, scores_pre_insertion);
        pre_delete_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, scores_pre_deletion);

        match_mask = _mm512_cmpeq_epi8_mask(first_vec.zmm, second_vec.zmm);
        cost_of_substitution_vec.zmm = _mm512_mask_blend_epi8(match_mask, mismatch_cost_vec.zmm, match_cost_vec.zmm);
        cost_if_substitution_vec.zmm = _mm512_add_epi8(pre_substitution_vec.zmm, cost_of_substitution_vec.zmm);
        cost_if_gap_vec.zmm =
            _mm512_add_epi8(_mm512_min_epu8(pre_insert_vec.zmm, pre_delete_vec.zmm), gap_cost_vec.zmm);
        cell_score_vec.zmm = _mm512_min_epu8(cost_if_substitution_vec.zmm, cost_if_gap_vec.zmm);
        _mm512_mask_storeu_epi8(scores_new, load_mask, cell_score_vec.zmm);
    }

    inline void operator()(                                                              //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        sz_u8_t const *scores_pre_substitution, sz_u8_t const *scores_pre_insertion,     //
        sz_u8_t const *scores_pre_deletion, sz_u8_t *scores_new,                         //
        dummy_executor_t executor = {}) noexcept {

        sz_unused_(executor); // On such small inputs, we don't need to worry about parallelism.

        // Initialize constats:
        sz_u512_vec_t match_cost_vec, mismatch_cost_vec, gap_cost_vec;
        match_cost_vec.zmm = _mm512_set1_epi8(this->substituter_.match);
        mismatch_cost_vec.zmm = _mm512_set1_epi8(this->substituter_.mismatch);
        gap_cost_vec.zmm = _mm512_set1_epi8(this->gap_costs_.open_or_extend);

        // On very small inputs, avoid the headache of splitting the input into chunks:
        if (length <= step_k) {
            slice_upto64chars(                                                                  //
                first_reversed_slice, second_slice, length,                                     //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);
            // The last element of the last chunk is the result of the global alignment.
            this->last_score_ = scores_new[0];
            return;
        }

        // First handle the misaligned slice of the output buffer:
        head_body_tail_t hbt = head_body_tail<step_k>(scores_new, length);

        // Misaligned head:
        if (hbt.head)
            slice_upto64chars(                                                                  //
                first_reversed_slice, second_slice, hbt.head,                                   //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);
        first_reversed_slice += hbt.head, second_slice += hbt.head, scores_pre_substitution += hbt.head,
            scores_pre_insertion += hbt.head, scores_pre_deletion += hbt.head, scores_new += hbt.head;

        // In this variant we will need at most 4 loops per diagonal:
        for (size_t progress = 0; progress < hbt.body; //
             progress += step_k,                       //
             first_reversed_slice += step_k, second_slice += step_k, scores_pre_substitution += step_k,
                    scores_pre_insertion += step_k, scores_pre_deletion += step_k, scores_new += step_k)
            slice_aligned64chars(                                                               //
                first_reversed_slice, second_slice,                                             //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);

        // Shorter tail:
        if (hbt.tail)
            slice_upto64chars(                                                                  //
                first_reversed_slice, second_slice, hbt.tail,                                   //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);

        // The last element of the last chunk is the result of the global alignment.
        if (length == 1) this->last_score_ = scores_new[0];
    }
};

/**
 *  @brief Variant of `scorer` - Minimizes Levenshtein distance for inputs under 256 runes.
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<sz_rune_t const *, sz_rune_t const *, sz_u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<sz_rune_t const *, sz_rune_t const *, sz_u8_t, uniform_substitution_costs_t,
                         linear_gap_costs_t, sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<sz_rune_t const *, sz_rune_t const *, sz_u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 16;

    /**
     *  @brief Computes one diagonal of the `u8` DM matrix for exactly 16 characters,
     *         using unaligned loads, but forcing @b aligned stores.
     */
    SZ_INLINE void slice_aligned16chars(                                             //
        sz_rune_t const *first_reversed_slice, sz_rune_t const *second_slice,        //
        sz_u8_t const *scores_pre_substitution, sz_u8_t const *scores_pre_insertion, //
        sz_u8_t const *scores_pre_deletion, sz_u8_t *scores_new,                     //
        sz_u128_vec_t match_cost_vec, sz_u128_vec_t mismatch_cost_vec,               //
        sz_u128_vec_t gap_cost_vec) const noexcept {

        __mmask16 match_mask;
        sz_u512_vec_t first_vec, second_vec;
        sz_u128_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        sz_u128_vec_t cost_of_substitution_vec;
        sz_u128_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

        // ? Note that here we are still traversing both buffers in the same order,
        // ? because one of the strings has been reversed beforehand.
        first_vec.zmm = _mm512_loadu_epi32(first_reversed_slice);
        second_vec.zmm = _mm512_loadu_epi32(second_slice);
        pre_substitution_vec.xmm = _mm_lddqu_si128((__m128i const *)(scores_pre_substitution));
        pre_insert_vec.xmm = _mm_lddqu_si128((__m128i const *)(scores_pre_insertion));
        pre_delete_vec.xmm = _mm_lddqu_si128((__m128i const *)(scores_pre_deletion));

        match_mask = _mm512_cmpeq_epi32_mask(first_vec.zmm, second_vec.zmm);
        cost_of_substitution_vec.xmm = _mm_mask_blend_epi8(match_mask, mismatch_cost_vec.xmm, match_cost_vec.xmm);
        cost_if_substitution_vec.xmm = _mm_add_epi8(pre_substitution_vec.xmm, cost_of_substitution_vec.xmm);
        cost_if_gap_vec.xmm = _mm_add_epi8(_mm_min_epu8(pre_insert_vec.xmm, pre_delete_vec.xmm), gap_cost_vec.xmm);
        cell_score_vec.xmm = _mm_min_epu8(cost_if_substitution_vec.xmm, cost_if_gap_vec.xmm);
        _mm_store_si128((__m128i *)scores_new, cell_score_vec.xmm);
    }

    /**
     *  @brief Computes one diagonal of the `u8` DM matrix for up to 16 characters,
     *         using unaligned loads and stores.
     */
    SZ_INLINE void slice_upto16chars(                                                   //
        sz_rune_t const *first_reversed_slice, sz_rune_t const *second_slice, size_t n, //
        sz_u8_t const *scores_pre_substitution, sz_u8_t const *scores_pre_insertion,    //
        sz_u8_t const *scores_pre_deletion, sz_u8_t *scores_new,                        //
        sz_u128_vec_t match_cost_vec, sz_u128_vec_t mismatch_cost_vec,                  //
        sz_u128_vec_t gap_cost_vec) const noexcept {

        __mmask16 load_mask, match_mask;
        sz_u512_vec_t first_vec, second_vec;
        sz_u128_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        sz_u128_vec_t cost_of_substitution_vec;
        sz_u128_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

        // ? Note that here we are still traversing both buffers in the same order,
        // ? because one of the strings has been reversed beforehand.
        load_mask = sz_u16_mask_until_(n);
        first_vec.zmm = _mm512_maskz_loadu_epi32(load_mask, first_reversed_slice);
        second_vec.zmm = _mm512_maskz_loadu_epi32(load_mask, second_slice);
        pre_substitution_vec.xmm = _mm_maskz_loadu_epi8(load_mask, scores_pre_substitution);
        pre_insert_vec.xmm = _mm_maskz_loadu_epi8(load_mask, scores_pre_insertion);
        pre_delete_vec.xmm = _mm_maskz_loadu_epi8(load_mask, scores_pre_deletion);

        match_mask = _mm512_cmpeq_epi32_mask(first_vec.zmm, second_vec.zmm);
        cost_of_substitution_vec.xmm = _mm_mask_blend_epi8(match_mask, mismatch_cost_vec.xmm, match_cost_vec.xmm);
        cost_if_substitution_vec.xmm = _mm_add_epi8(pre_substitution_vec.xmm, cost_of_substitution_vec.xmm);
        cost_if_gap_vec.xmm = _mm_add_epi8(_mm_min_epu8(pre_insert_vec.xmm, pre_delete_vec.xmm), gap_cost_vec.xmm);
        cell_score_vec.xmm = _mm_min_epu8(cost_if_substitution_vec.xmm, cost_if_gap_vec.xmm);
        _mm_mask_storeu_epi8(scores_new, load_mask, cell_score_vec.xmm);
    }

    inline void operator()(                                                                        //
        sz_rune_t const *first_reversed_slice, sz_rune_t const *second_slice, size_t const length, //
        sz_u8_t const *scores_pre_substitution, sz_u8_t const *scores_pre_insertion,               //
        sz_u8_t const *scores_pre_deletion, sz_u8_t *scores_new,                                   //
        dummy_executor_t executor = {}) noexcept {

        sz_unused_(executor); // On such small inputs, we don't need to worry about parallelism.

        // Initialize constats:
        sz_u128_vec_t match_cost_vec, mismatch_cost_vec, gap_cost_vec;
        match_cost_vec.xmm = _mm_set1_epi8(this->substituter_.match);
        mismatch_cost_vec.xmm = _mm_set1_epi8(this->substituter_.mismatch);
        gap_cost_vec.xmm = _mm_set1_epi8(this->gap_costs_.open_or_extend);

        // On very small inputs, avoid the headache of splitting the input into chunks:
        if (length <= step_k) {
            slice_upto16chars(                                                                  //
                first_reversed_slice, second_slice, length,                                     //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);
            // The last element of the last chunk is the result of the global alignment.
            this->last_score_ = scores_new[0];
            return;
        }

        // First handle the misaligned slice of the output buffer:
        head_body_tail_t hbt = head_body_tail<step_k>(scores_new, length);

        // Misaligned head:
        if (hbt.head)
            slice_upto16chars(                                                                  //
                first_reversed_slice, second_slice, hbt.head,                                   //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);
        first_reversed_slice += hbt.head, second_slice += hbt.head, scores_pre_substitution += hbt.head,
            scores_pre_insertion += hbt.head, scores_pre_deletion += hbt.head, scores_new += hbt.head;

        // In this variant we will need at most (256 / 16) = 16 loops per diagonal.
        for (size_t progress = 0; progress < hbt.body; //
             progress += step_k,                       //
             first_reversed_slice += step_k, second_slice += step_k, scores_pre_substitution += step_k,
                    scores_pre_insertion += step_k, scores_pre_deletion += step_k, scores_new += step_k)
            slice_aligned16chars(                                                               //
                first_reversed_slice, second_slice,                                             //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);

        // Shorter tail:
        if (hbt.tail)
            slice_upto16chars(                                                                  //
                first_reversed_slice, second_slice, hbt.tail,                                   //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);

        // The last element of the last chunk is the result of the global alignment.
        if (length == 1) this->last_score_ = scores_new[0];
    }
};

/**
 *  @brief Variant of `scorer` - Minimizes Levenshtein distance for inputs in [256, 65K] bytes.
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, sz_u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, sz_u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, sz_u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 32;

    /**
     *  @brief Computes one diagonal of the `u16` DM matrix for exactly 16 characters,
     *         using unaligned loads, but forcing @b aligned stores.
     */
    SZ_INLINE void slice_aligned32chars(                                               //
        char const *first_reversed_slice, char const *second_slice,                    //
        sz_u16_t const *scores_pre_substitution, sz_u16_t const *scores_pre_insertion, //
        sz_u16_t const *scores_pre_deletion, sz_u16_t *scores_new,                     //
        sz_u512_vec_t match_cost_vec, sz_u512_vec_t mismatch_cost_vec, sz_u512_vec_t gap_cost_vec) const noexcept {

        __mmask32 match_mask;
        sz_u256_vec_t first_vec, second_vec;
        sz_u512_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        sz_u512_vec_t cost_of_substitution_vec;
        sz_u512_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

        // ? Note that here we are still traversing both buffers in the same order,
        // ? because one of the strings has been reversed beforehand.
        first_vec.ymm = _mm256_loadu_epi8(first_reversed_slice);
        second_vec.ymm = _mm256_loadu_epi8(second_slice);
        pre_substitution_vec.zmm = _mm512_loadu_epi16(scores_pre_substitution);
        pre_insert_vec.zmm = _mm512_loadu_epi16(scores_pre_insertion);
        pre_delete_vec.zmm = _mm512_loadu_epi16(scores_pre_deletion);

        match_mask = _mm256_cmpeq_epi8_mask(first_vec.ymm, second_vec.ymm);
        cost_of_substitution_vec.zmm = _mm512_mask_blend_epi16(match_mask, mismatch_cost_vec.zmm, match_cost_vec.zmm);
        cost_if_substitution_vec.zmm = _mm512_add_epi16(pre_substitution_vec.zmm, cost_of_substitution_vec.zmm);
        cost_if_gap_vec.zmm =
            _mm512_add_epi16(_mm512_min_epu16(pre_insert_vec.zmm, pre_delete_vec.zmm), gap_cost_vec.zmm);
        cell_score_vec.zmm = _mm512_min_epu16(cost_if_substitution_vec.zmm, cost_if_gap_vec.zmm);
        _mm512_store_si512(scores_new, cell_score_vec.zmm);
    }

    SZ_INLINE void slice_upto32chars(                                                  //
        char const *first_reversed_slice, char const *second_slice, size_t n,          //
        sz_u16_t const *scores_pre_substitution, sz_u16_t const *scores_pre_insertion, //
        sz_u16_t const *scores_pre_deletion, sz_u16_t *scores_new,                     //
        sz_u512_vec_t match_cost_vec, sz_u512_vec_t mismatch_cost_vec, sz_u512_vec_t gap_cost_vec) const noexcept {

        __mmask32 load_mask, match_mask;
        sz_u256_vec_t first_vec, second_vec;
        sz_u512_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        sz_u512_vec_t cost_of_substitution_vec;
        sz_u512_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

        // ? Note that here we are still traversing both buffers in the same order,
        // ? because one of the strings has been reversed beforehand.
        load_mask = sz_u32_mask_until_(n);
        first_vec.ymm = _mm256_maskz_loadu_epi8(load_mask, first_reversed_slice);
        second_vec.ymm = _mm256_maskz_loadu_epi8(load_mask, second_slice);
        pre_substitution_vec.zmm = _mm512_maskz_loadu_epi16(load_mask, scores_pre_substitution);
        pre_insert_vec.zmm = _mm512_maskz_loadu_epi16(load_mask, scores_pre_insertion);
        pre_delete_vec.zmm = _mm512_maskz_loadu_epi16(load_mask, scores_pre_deletion);

        match_mask = _mm256_cmpeq_epi8_mask(first_vec.ymm, second_vec.ymm);
        cost_of_substitution_vec.zmm = _mm512_mask_blend_epi16(match_mask, mismatch_cost_vec.zmm, match_cost_vec.zmm);
        cost_if_substitution_vec.zmm = _mm512_add_epi16(pre_substitution_vec.zmm, cost_of_substitution_vec.zmm);
        cost_if_gap_vec.zmm =
            _mm512_add_epi16(_mm512_min_epu16(pre_insert_vec.zmm, pre_delete_vec.zmm), gap_cost_vec.zmm);
        cell_score_vec.zmm = _mm512_min_epu16(cost_if_substitution_vec.zmm, cost_if_gap_vec.zmm);
        _mm512_mask_storeu_epi16(scores_new, load_mask, cell_score_vec.zmm);
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    inline void operator()(                                                              //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        sz_u16_t const *scores_pre_substitution, sz_u16_t const *scores_pre_insertion,   //
        sz_u16_t const *scores_pre_deletion, sz_u16_t *scores_new,                       //
        executor_type_ &&executor = {}) noexcept {

        // Initialize constats:
        sz_u512_vec_t match_cost_vec, mismatch_cost_vec, gap_cost_vec;
        match_cost_vec.zmm = _mm512_set1_epi16(this->substituter_.match);
        mismatch_cost_vec.zmm = _mm512_set1_epi16(this->substituter_.mismatch);
        gap_cost_vec.zmm = _mm512_set1_epi16(this->gap_costs_.open_or_extend);

        // On very small inputs, avoid the headache of splitting the input into chunks:
        if (length <= step_k) {
            slice_upto32chars(                                                                  //
                first_reversed_slice, second_slice, length,                                     //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);
            // The last element of the last chunk is the result of the global alignment.
            this->last_score_ = scores_new[0];
            return;
        }

        // First handle the misaligned slice of the output buffer:
        head_body_tail_t hbt = head_body_tail<step_k>(scores_new, length);

        // Misaligned head and tail:
        if (hbt.head)
            slice_upto32chars(                                                                  //
                first_reversed_slice, second_slice, hbt.head,                                   //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);
        first_reversed_slice += hbt.head, second_slice += hbt.head, scores_pre_substitution += hbt.head,
            scores_pre_insertion += hbt.head, scores_pre_deletion += hbt.head, scores_new += hbt.head;
        if (hbt.tail)
            slice_upto32chars(                                                       //
                first_reversed_slice + hbt.body, second_slice + hbt.body, hbt.tail,  //
                scores_pre_substitution + hbt.body, scores_pre_insertion + hbt.body, //
                scores_pre_deletion + hbt.body, scores_new + hbt.body,               //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);

        // In this variant we will need at most (64 * 1024 / 32) = 2048 loops per diagonal.
        size_t const body_pages = hbt.body / step_k;
        executor.for_n(body_pages, [&](size_t const page) noexcept {
            size_t const progress = page * step_k;
            slice_aligned32chars(                                                                             //
                first_reversed_slice + progress, second_slice + progress, scores_pre_substitution + progress, //
                scores_pre_insertion + progress, scores_pre_deletion + progress, scores_new + progress,       //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);
        });

        // The last element of the last chunk is the result of the global alignment.
        if (length == 1) this->last_score_ = scores_new[0];
    }
};

/**
 *  @brief Variant of `scorer` - Minimizes Levenshtein distance for inputs in [256, 65K] runes in parallel.
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<sz_rune_t const *, sz_rune_t const *, sz_u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<sz_rune_t const *, sz_rune_t const *, sz_u16_t, uniform_substitution_costs_t,
                         linear_gap_costs_t, sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<sz_rune_t const *, sz_rune_t const *, sz_u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 16;

    /**
     *  @brief Computes one diagonal of the `u16` DM matrix for exactly 16 characters,
     *         using unaligned loads, but forcing @b aligned stores.
     */
    SZ_INLINE void slice_aligned16chars(                                               //
        sz_rune_t const *first_reversed_slice, sz_rune_t const *second_slice,          //
        sz_u16_t const *scores_pre_substitution, sz_u16_t const *scores_pre_insertion, //
        sz_u16_t const *scores_pre_deletion, sz_u16_t *scores_new,                     //
        sz_u256_vec_t match_cost_vec, sz_u256_vec_t mismatch_cost_vec, sz_u256_vec_t gap_cost_vec) const noexcept {

        __mmask16 match_mask;
        sz_u512_vec_t first_vec, second_vec;
        sz_u256_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        sz_u256_vec_t cost_of_substitution_vec;
        sz_u256_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

        // ? Note that here we are still traversing both buffers in the same order,
        // ? because one of the strings has been reversed beforehand.
        first_vec.zmm = _mm512_loadu_epi32(first_reversed_slice);
        second_vec.zmm = _mm512_loadu_epi32(second_slice);
        pre_substitution_vec.ymm = _mm256_loadu_epi16(scores_pre_substitution);
        pre_insert_vec.ymm = _mm256_loadu_epi16(scores_pre_insertion);
        pre_delete_vec.ymm = _mm256_loadu_epi16(scores_pre_deletion);

        match_mask = _mm512_cmpeq_epi32_mask(first_vec.zmm, second_vec.zmm);
        cost_of_substitution_vec.ymm = _mm256_mask_blend_epi16(match_mask, mismatch_cost_vec.ymm, match_cost_vec.ymm);
        cost_if_substitution_vec.ymm = _mm256_add_epi16(pre_substitution_vec.ymm, cost_of_substitution_vec.ymm);
        cost_if_gap_vec.ymm =
            _mm256_add_epi16(_mm256_min_epu16(pre_insert_vec.ymm, pre_delete_vec.ymm), gap_cost_vec.ymm);
        cell_score_vec.ymm = _mm256_min_epu16(cost_if_substitution_vec.ymm, cost_if_gap_vec.ymm);
        _mm256_store_si256((__m256i *)scores_new, cell_score_vec.ymm);
    }

    /**
     *  @brief Computes one diagonal of the `u16` DM matrix for up to 16 characters,
     *         using unaligned loads and stores.
     */
    SZ_INLINE void slice_upto16chars(                                                   //
        sz_rune_t const *first_reversed_slice, sz_rune_t const *second_slice, size_t n, //
        sz_u16_t const *scores_pre_substitution, sz_u16_t const *scores_pre_insertion,  //
        sz_u16_t const *scores_pre_deletion, sz_u16_t *scores_new,                      //
        sz_u256_vec_t match_cost_vec, sz_u256_vec_t mismatch_cost_vec, sz_u256_vec_t gap_cost_vec) const noexcept {

        __mmask16 load_mask, match_mask;
        sz_u512_vec_t first_vec, second_vec;
        sz_u256_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        sz_u256_vec_t cost_of_substitution_vec;
        sz_u256_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

        // ? Note that here we are still traversing both buffers in the same order,
        // ? because one of the strings has been reversed beforehand.
        load_mask = sz_u16_mask_until_(n);
        first_vec.zmm = _mm512_maskz_loadu_epi32(load_mask, first_reversed_slice);
        second_vec.zmm = _mm512_maskz_loadu_epi32(load_mask, second_slice);
        pre_substitution_vec.ymm = _mm256_maskz_loadu_epi16(load_mask, scores_pre_substitution);
        pre_insert_vec.ymm = _mm256_maskz_loadu_epi16(load_mask, scores_pre_insertion);
        pre_delete_vec.ymm = _mm256_maskz_loadu_epi16(load_mask, scores_pre_deletion);

        match_mask = _mm512_cmpeq_epi32_mask(first_vec.zmm, second_vec.zmm);
        cost_of_substitution_vec.ymm = _mm256_mask_blend_epi16(match_mask, mismatch_cost_vec.ymm, match_cost_vec.ymm);
        cost_if_substitution_vec.ymm = _mm256_add_epi16(pre_substitution_vec.ymm, cost_of_substitution_vec.ymm);
        cost_if_gap_vec.ymm =
            _mm256_add_epi16(_mm256_min_epu16(pre_insert_vec.ymm, pre_delete_vec.ymm), gap_cost_vec.ymm);
        cell_score_vec.ymm = _mm256_min_epu16(cost_if_substitution_vec.ymm, cost_if_gap_vec.ymm);
        _mm256_mask_storeu_epi16(scores_new, load_mask, cell_score_vec.ymm);
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    inline void operator()(                                                                        //
        sz_rune_t const *first_reversed_slice, sz_rune_t const *second_slice, size_t const length, //
        sz_u16_t const *scores_pre_substitution, sz_u16_t const *scores_pre_insertion,             //
        sz_u16_t const *scores_pre_deletion, sz_u16_t *scores_new,                                 //
        executor_type_ &&executor = {}) noexcept {

        // Initialize constats:
        sz_u256_vec_t match_cost_vec, mismatch_cost_vec, gap_cost_vec;
        match_cost_vec.ymm = _mm256_set1_epi16(this->substituter_.match);
        mismatch_cost_vec.ymm = _mm256_set1_epi16(this->substituter_.mismatch);
        gap_cost_vec.ymm = _mm256_set1_epi16(this->gap_costs_.open_or_extend);

        // On very small inputs, avoid the headache of splitting the input into chunks:
        if (length <= step_k) {
            slice_upto16chars(                                                                  //
                first_reversed_slice, second_slice, length,                                     //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);
            // The last element of the last chunk is the result of the global alignment.
            this->last_score_ = scores_new[0];
            return;
        }

        // First handle the misaligned slice of the output buffer:
        head_body_tail_t hbt = head_body_tail<step_k>(scores_new, length);

        // Misaligned head and tail:
        if (hbt.head)
            slice_upto16chars(                                                                  //
                first_reversed_slice, second_slice, hbt.head,                                   //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);
        first_reversed_slice += hbt.head, second_slice += hbt.head, scores_pre_substitution += hbt.head,
            scores_pre_insertion += hbt.head, scores_pre_deletion += hbt.head, scores_new += hbt.head;
        if (hbt.tail)
            slice_upto16chars(                                                       //
                first_reversed_slice + hbt.body, second_slice + hbt.body, hbt.tail,  //
                scores_pre_substitution + hbt.body, scores_pre_insertion + hbt.body, //
                scores_pre_deletion + hbt.body, scores_new + hbt.body,               //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);

        // In this variant we will need at most (64 * 1024 / 16) = 4096 loops per diagonal.
        size_t const body_pages = hbt.body / step_k;
        executor.for_n(body_pages, [&](size_t const page) noexcept {
            size_t const progress = page * step_k;
            slice_aligned16chars(                                                                             //
                first_reversed_slice + progress, second_slice + progress, scores_pre_substitution + progress, //
                scores_pre_insertion + progress, scores_pre_deletion + progress, scores_new + progress,       //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);
        });

        // The last element of the last chunk is the result of the global alignment.
        if (length == 1) this->last_score_ = scores_new[0];
    }
};

/**
 *  @brief Variant of `scorer` - Minimizes Levenshtein distance for inputs in [65K, 4B] bytes.
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, sz_u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, sz_u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, sz_u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 16;

    /**
     *  @brief Computes one diagonal of the `u32` DM matrix for exactly 16 characters,
     *         using unaligned loads, but forcing @b aligned stores.
     */
    SZ_INLINE void slice_aligned16chars(                                               //
        char const *first_reversed_slice, char const *second_slice,                    //
        sz_u32_t const *scores_pre_substitution, sz_u32_t const *scores_pre_insertion, //
        sz_u32_t const *scores_pre_deletion, sz_u32_t *scores_new,                     //
        sz_u512_vec_t match_cost_vec, sz_u512_vec_t mismatch_cost_vec, sz_u512_vec_t gap_cost_vec) const noexcept {

        __mmask16 match_mask;
        sz_u128_vec_t first_vec, second_vec;
        sz_u512_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        sz_u512_vec_t cost_of_substitution_vec;
        sz_u512_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

        // ? Note that here we are still traversing both buffers in the same order,
        // ? because one of the strings has been reversed beforehand.
        first_vec.xmm = _mm_lddqu_si128((__m128i const *)first_reversed_slice);
        second_vec.xmm = _mm_lddqu_si128((__m128i const *)second_slice);
        pre_substitution_vec.zmm = _mm512_loadu_epi32(scores_pre_substitution);
        pre_insert_vec.zmm = _mm512_loadu_epi32(scores_pre_insertion);
        pre_delete_vec.zmm = _mm512_loadu_epi32(scores_pre_deletion);

        match_mask = _mm_cmpeq_epi8_mask(first_vec.xmm, second_vec.xmm);
        cost_of_substitution_vec.zmm = _mm512_mask_blend_epi32(match_mask, mismatch_cost_vec.zmm, match_cost_vec.zmm);
        cost_if_substitution_vec.zmm = _mm512_add_epi32(pre_substitution_vec.zmm, cost_of_substitution_vec.zmm);
        cost_if_gap_vec.zmm =
            _mm512_add_epi32(_mm512_min_epu32(pre_insert_vec.zmm, pre_delete_vec.zmm), gap_cost_vec.zmm);
        cell_score_vec.zmm = _mm512_min_epu32(cost_if_substitution_vec.zmm, cost_if_gap_vec.zmm);
        _mm512_store_si512((__m512i *)scores_new, cell_score_vec.zmm);
    }

    /**
     *  @brief Computes one diagonal of the `u32` DM matrix for up to 16 characters,
     *         using unaligned loads and stores.
     */
    SZ_INLINE void slice_upto16chars(                                                  //
        char const *first_reversed_slice, char const *second_slice, size_t n,          //
        sz_u32_t const *scores_pre_substitution, sz_u32_t const *scores_pre_insertion, //
        sz_u32_t const *scores_pre_deletion, sz_u32_t *scores_new,                     //
        sz_u512_vec_t match_cost_vec, sz_u512_vec_t mismatch_cost_vec, sz_u512_vec_t gap_cost_vec) const noexcept {

        __mmask16 load_mask, match_mask;
        sz_u128_vec_t first_vec, second_vec;
        sz_u512_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        sz_u512_vec_t cost_of_substitution_vec;
        sz_u512_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

        // ? Note that here we are still traversing both buffers in the same order,
        // ? because one of the strings has been reversed beforehand.
        load_mask = sz_u16_mask_until_(n);
        first_vec.xmm = _mm_maskz_loadu_epi8(load_mask, first_reversed_slice);
        second_vec.xmm = _mm_maskz_loadu_epi8(load_mask, second_slice);
        pre_substitution_vec.zmm = _mm512_maskz_loadu_epi32(load_mask, scores_pre_substitution);
        pre_insert_vec.zmm = _mm512_maskz_loadu_epi32(load_mask, scores_pre_insertion);
        pre_delete_vec.zmm = _mm512_maskz_loadu_epi32(load_mask, scores_pre_deletion);

        match_mask = _mm_cmpeq_epi8_mask(first_vec.xmm, second_vec.xmm);
        cost_of_substitution_vec.zmm = _mm512_mask_blend_epi32(match_mask, mismatch_cost_vec.zmm, match_cost_vec.zmm);
        cost_if_substitution_vec.zmm = _mm512_add_epi32(pre_substitution_vec.zmm, cost_of_substitution_vec.zmm);
        cost_if_gap_vec.zmm =
            _mm512_add_epi32(_mm512_min_epu32(pre_insert_vec.zmm, pre_delete_vec.zmm), gap_cost_vec.zmm);
        cell_score_vec.zmm = _mm512_min_epu32(cost_if_substitution_vec.zmm, cost_if_gap_vec.zmm);
        _mm512_mask_storeu_epi32(scores_new, load_mask, cell_score_vec.zmm);
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    inline void operator()(                                                              //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        sz_u32_t const *scores_pre_substitution, sz_u32_t const *scores_pre_insertion,   //
        sz_u32_t const *scores_pre_deletion, sz_u32_t *scores_new,                       //
        executor_type_ &&executor = {}) noexcept {

        // Initialize constats:
        sz_u512_vec_t match_cost_vec, mismatch_cost_vec, gap_cost_vec;
        match_cost_vec.zmm = _mm512_set1_epi32(this->substituter_.match);
        mismatch_cost_vec.zmm = _mm512_set1_epi32(this->substituter_.mismatch);
        gap_cost_vec.zmm = _mm512_set1_epi32(this->gap_costs_.open_or_extend);

        // On very small inputs, avoid the headache of splitting the input into chunks:
        if (length <= step_k) {
            slice_upto16chars(                                                                  //
                first_reversed_slice, second_slice, length,                                     //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);
            // The last element of the last chunk is the result of the global alignment.
            this->last_score_ = scores_new[0];
            return;
        }

        // First handle the misaligned slice of the output buffer:
        head_body_tail_t hbt = head_body_tail<step_k>(scores_new, length);

        // Misaligned head and tail:
        if (hbt.head)
            slice_upto16chars(                                                                  //
                first_reversed_slice, second_slice, hbt.head,                                   //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);
        first_reversed_slice += hbt.head, second_slice += hbt.head, scores_pre_substitution += hbt.head,
            scores_pre_insertion += hbt.head, scores_pre_deletion += hbt.head, scores_new += hbt.head;
        if (hbt.tail)
            slice_upto16chars(                                                       //
                first_reversed_slice + hbt.body, second_slice + hbt.body, hbt.tail,  //
                scores_pre_substitution + hbt.body, scores_pre_insertion + hbt.body, //
                scores_pre_deletion + hbt.body, scores_new + hbt.body,               //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);

        size_t const body_pages = hbt.body / step_k;
        executor.for_n(body_pages, [&](size_t const page) noexcept {
            size_t const progress = page * step_k;
            slice_aligned16chars(                                                                             //
                first_reversed_slice + progress, second_slice + progress, scores_pre_substitution + progress, //
                scores_pre_insertion + progress, scores_pre_deletion + progress, scores_new + progress,       //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);
        });

        // The last element of the last chunk is the result of the global alignment.
        if (length == 1) this->last_score_ = scores_new[0];
    }
};

/**
 *  @brief Variant of `tile_scorer` - Minimizes Levenshtein distance for inputs under 256 bytes.
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, sz_u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, sz_u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, sz_u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 64;

    /**
     *  @brief Computes one diagonal of the `u8` DM matrix for up to 64 characters,
     *         using unaligned loads and stores.
     */
    SZ_INLINE void slice_upto64chars(                                         //
        char const *first_reversed_slice, char const *second_slice, size_t n, //
        sz_u8_t const *scores_pre_substitution,                               //
        sz_u8_t const *scores_pre_insertion,                                  //
        sz_u8_t const *scores_pre_deletion,                                   //
        sz_u8_t const *scores_running_insertions,                             //
        sz_u8_t const *scores_running_deletions,                              //
        sz_u8_t *scores_new,                                                  //
        sz_u8_t *scores_new_insertions,                                       //
        sz_u8_t *scores_new_deletions,                                        //
        sz_u512_vec_t match_cost_vec, sz_u512_vec_t mismatch_cost_vec,        //
        sz_u512_vec_t gap_open_vec, sz_u512_vec_t gap_expand_vec) const noexcept {

        __mmask64 load_mask, match_mask;
        sz_u512_vec_t first_vec, second_vec;
        sz_u512_vec_t pre_substitution_vec, pre_insert_open_vec, pre_delete_open_vec, pre_insert_expand_vec,
            pre_delete_expand_vec;
        sz_u512_vec_t cost_of_substitution_vec;
        sz_u512_vec_t cost_if_substitution_vec, cost_if_insert, cost_if_delete, cell_score_vec;

        // ? Note that here we are still traversing both buffers in the same order,
        // ? because one of the strings has been reversed beforehand.
        load_mask = sz_u64_mask_until_(n);
        first_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, first_reversed_slice);
        second_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, second_slice);
        pre_substitution_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, scores_pre_substitution);
        pre_insert_open_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, scores_pre_insertion);
        pre_delete_open_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, scores_pre_deletion);
        pre_insert_expand_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, scores_running_insertions);
        pre_delete_expand_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, scores_running_deletions);

        match_mask = _mm512_cmpeq_epi8_mask(first_vec.zmm, second_vec.zmm);
        cost_of_substitution_vec.zmm = _mm512_mask_blend_epi8(match_mask, mismatch_cost_vec.zmm, match_cost_vec.zmm);
        cost_if_substitution_vec.zmm = _mm512_add_epi8(pre_substitution_vec.zmm, cost_of_substitution_vec.zmm);
        cost_if_insert.zmm = _mm512_min_epu8(_mm512_add_epi8(pre_insert_expand_vec.zmm, gap_expand_vec.zmm),
                                             _mm512_add_epi8(pre_insert_open_vec.zmm, gap_open_vec.zmm));
        cost_if_delete.zmm = _mm512_min_epu8(_mm512_add_epi8(pre_delete_expand_vec.zmm, gap_expand_vec.zmm),
                                             _mm512_add_epi8(pre_delete_open_vec.zmm, gap_open_vec.zmm));
        cell_score_vec.zmm =
            _mm512_min_epu8(cost_if_substitution_vec.zmm, _mm512_min_epu8(cost_if_insert.zmm, cost_if_delete.zmm));

        // Export results.
        _mm512_mask_storeu_epi8(scores_new, load_mask, cell_score_vec.zmm);
        _mm512_mask_storeu_epi8(scores_new_insertions, load_mask, cost_if_insert.zmm);
        _mm512_mask_storeu_epi8(scores_new_deletions, load_mask, cost_if_delete.zmm);
    }

    inline void operator()(                                                              //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        sz_u8_t const *scores_pre_substitution,                                          //
        sz_u8_t const *scores_pre_insertion,                                             //
        sz_u8_t const *scores_pre_deletion,                                              //
        sz_u8_t const *scores_running_insertions,                                        //
        sz_u8_t const *scores_running_deletions,                                         //
        sz_u8_t *scores_new,                                                             //
        sz_u8_t *scores_new_insertions,                                                  //
        sz_u8_t *scores_new_deletions,                                                   //
        dummy_executor_t executor = {}) noexcept {

        sz_unused_(executor); // On such small inputs, we don't need to worry about parallelism.

        // Initialize constats:
        sz_u512_vec_t match_cost_vec, mismatch_cost_vec, gap_open_vec, gap_expand_vec;
        match_cost_vec.zmm = _mm512_set1_epi8(this->substituter_.match);
        mismatch_cost_vec.zmm = _mm512_set1_epi8(this->substituter_.mismatch);
        gap_open_vec.zmm = _mm512_set1_epi8(this->gap_costs_.open);
        gap_expand_vec.zmm = _mm512_set1_epi8(this->gap_costs_.extend);

        // In this variant we will need at most 4 loops per diagonal:
        size_t progress = 0;
        for (; progress + step_k <= length; progress += step_k)
            slice_upto64chars(                                                                                       //
                first_reversed_slice, second_slice, step_k,                                                          //
                scores_pre_substitution + progress, scores_pre_insertion + progress, scores_pre_deletion + progress, //
                scores_running_insertions + progress, scores_running_deletions + progress,                           //
                scores_new + progress, scores_new_insertions + progress, scores_new_deletions + progress,            //
                match_cost_vec, mismatch_cost_vec, gap_open_vec, gap_expand_vec);

        // Shorter tail:
        size_t const tail = length - progress;
        if (tail)
            slice_upto64chars(                                                                                       //
                first_reversed_slice + progress, second_slice + progress, tail,                                      //
                scores_pre_substitution + progress, scores_pre_insertion + progress, scores_pre_deletion + progress, //
                scores_running_insertions + progress, scores_running_deletions + progress,                           //
                scores_new + progress, scores_new_insertions + progress, scores_new_deletions + progress,            //
                match_cost_vec, mismatch_cost_vec, gap_open_vec, gap_expand_vec);

        // The last element of the last chunk is the result of the global alignment.
        if (length == 1) this->last_score_ = scores_new[0];
    }
};

/**
 *  @brief Variant of `scorer` - Minimizes Levenshtein distance for inputs in [256, 65K] bytes.
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, sz_u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, sz_u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, sz_u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 32;

    SZ_INLINE void slice_upto32chars(                                         //
        char const *first_reversed_slice, char const *second_slice, size_t n, //
        sz_u16_t const *scores_pre_substitution,                              //
        sz_u16_t const *scores_pre_insertion,                                 //
        sz_u16_t const *scores_pre_deletion,                                  //
        sz_u16_t const *scores_running_insertions,                            //
        sz_u16_t const *scores_running_deletions,                             //
        sz_u16_t *scores_new,                                                 //
        sz_u16_t *scores_new_insertions,                                      //
        sz_u16_t *scores_new_deletions,                                       //
        sz_u512_vec_t match_cost_vec, sz_u512_vec_t mismatch_cost_vec,        //
        sz_u512_vec_t gap_open_vec, sz_u512_vec_t gap_expand_vec) const noexcept {

        __mmask32 load_mask, match_mask;
        sz_u256_vec_t first_vec, second_vec;
        sz_u512_vec_t pre_substitution_vec, pre_insert_open_vec, pre_delete_open_vec, pre_insert_expand_vec,
            pre_delete_expand_vec;
        sz_u512_vec_t cost_of_substitution_vec;
        sz_u512_vec_t cost_if_substitution_vec, cost_if_insert, cost_if_delete, cell_score_vec;

        // ? Note that here we are still traversing both buffers in the same order,
        // ? because one of the strings has been reversed beforehand.
        load_mask = sz_u32_mask_until_(n);
        first_vec.ymm = _mm256_maskz_loadu_epi8(load_mask, first_reversed_slice);
        second_vec.ymm = _mm256_maskz_loadu_epi8(load_mask, second_slice);
        pre_substitution_vec.zmm = _mm512_maskz_loadu_epi16(load_mask, scores_pre_substitution);
        pre_insert_open_vec.zmm = _mm512_maskz_loadu_epi16(load_mask, scores_pre_insertion);
        pre_delete_open_vec.zmm = _mm512_maskz_loadu_epi16(load_mask, scores_pre_deletion);
        pre_insert_expand_vec.zmm = _mm512_maskz_loadu_epi16(load_mask, scores_running_insertions);
        pre_delete_expand_vec.zmm = _mm512_maskz_loadu_epi16(load_mask, scores_running_deletions);

        match_mask = _mm256_cmpeq_epi8_mask(first_vec.ymm, second_vec.ymm);
        cost_of_substitution_vec.zmm = _mm512_mask_blend_epi16(match_mask, mismatch_cost_vec.zmm, match_cost_vec.zmm);
        cost_if_substitution_vec.zmm = _mm512_add_epi16(pre_substitution_vec.zmm, cost_of_substitution_vec.zmm);
        cost_if_insert.zmm = _mm512_min_epu16(_mm512_add_epi16(pre_insert_expand_vec.zmm, gap_expand_vec.zmm),
                                              _mm512_add_epi16(pre_insert_open_vec.zmm, gap_open_vec.zmm));
        cost_if_delete.zmm = _mm512_min_epu16(_mm512_add_epi16(pre_delete_expand_vec.zmm, gap_expand_vec.zmm),
                                              _mm512_add_epi16(pre_delete_open_vec.zmm, gap_open_vec.zmm));
        cell_score_vec.zmm =
            _mm512_min_epu16(cost_if_substitution_vec.zmm, _mm512_min_epu16(cost_if_insert.zmm, cost_if_delete.zmm));

        // Export results.
        _mm512_mask_storeu_epi16(scores_new, load_mask, cell_score_vec.zmm);
        _mm512_mask_storeu_epi16(scores_new_insertions, load_mask, cost_if_insert.zmm);
        _mm512_mask_storeu_epi16(scores_new_deletions, load_mask, cost_if_delete.zmm);
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    inline void operator()(                                                              //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        sz_u16_t const *scores_pre_substitution,                                         //
        sz_u16_t const *scores_pre_insertion,                                            //
        sz_u16_t const *scores_pre_deletion,                                             //
        sz_u16_t const *scores_running_insertions,                                       //
        sz_u16_t const *scores_running_deletions,                                        //
        sz_u16_t *scores_new,                                                            //
        sz_u16_t *scores_new_insertions,                                                 //
        sz_u16_t *scores_new_deletions,                                                  //
        executor_type_ &&executor = {}) noexcept {

        // Initialize constats:
        sz_u512_vec_t match_cost_vec, mismatch_cost_vec, gap_open_vec, gap_expand_vec;
        match_cost_vec.zmm = _mm512_set1_epi16(this->substituter_.match);
        mismatch_cost_vec.zmm = _mm512_set1_epi16(this->substituter_.mismatch);
        gap_open_vec.zmm = _mm512_set1_epi16(this->gap_costs_.open);
        gap_expand_vec.zmm = _mm512_set1_epi16(this->gap_costs_.extend);

        // In this variant we will need at most (64 * 1024 / 32) = 2048 loops per diagonal.
        size_t const body_pages = length / step_k;
        executor.for_n(body_pages, [&](size_t const page) noexcept {
            size_t const progress = page * step_k;
            slice_upto32chars(                                                                                       //
                first_reversed_slice + progress, second_slice + progress, step_k,                                    //
                scores_pre_substitution + progress, scores_pre_insertion + progress, scores_pre_deletion + progress, //
                scores_running_insertions + progress, scores_running_deletions + progress,                           //
                scores_new + progress, scores_new_insertions + progress, scores_new_deletions + progress,            //
                match_cost_vec, mismatch_cost_vec, gap_open_vec, gap_expand_vec);
        });

        // Shorter tail:
        size_t const progress = body_pages * step_k;
        size_t const tail = length - progress;
        if (tail)
            slice_upto32chars(                                                                                       //
                first_reversed_slice + progress, second_slice + progress, tail,                                      //
                scores_pre_substitution + progress, scores_pre_insertion + progress, scores_pre_deletion + progress, //
                scores_running_insertions + progress, scores_running_deletions + progress,                           //
                scores_new + progress, scores_new_insertions + progress, scores_new_deletions + progress,            //
                match_cost_vec, mismatch_cost_vec, gap_open_vec, gap_expand_vec);

        // The last element of the last chunk is the result of the global alignment.
        if (length == 1) this->last_score_ = scores_new[0];
    }
};

/**
 *  @brief Variant of `scorer` - Minimizes Levenshtein distance for inputs in [65K, 4B] bytes.
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, sz_u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, sz_u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, sz_u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 16;

    /**
     *  @brief Computes one diagonal of the `u32` DM matrix for up to 16 characters,
     *         using unaligned loads and stores.
     */
    SZ_INLINE void slice_upto16chars(                                         //
        char const *first_reversed_slice, char const *second_slice, size_t n, //
        sz_u32_t const *scores_pre_substitution,                              //
        sz_u32_t const *scores_pre_insertion,                                 //
        sz_u32_t const *scores_pre_deletion,                                  //
        sz_u32_t const *scores_running_insertions,                            //
        sz_u32_t const *scores_running_deletions,                             //
        sz_u32_t *scores_new,                                                 //
        sz_u32_t *scores_new_insertions,                                      //
        sz_u32_t *scores_new_deletions,                                       //
        sz_u512_vec_t match_cost_vec, sz_u512_vec_t mismatch_cost_vec,        //
        sz_u512_vec_t gap_open_vec, sz_u512_vec_t gap_expand_vec) const noexcept {

        __mmask16 load_mask, match_mask;
        sz_u128_vec_t first_vec, second_vec;
        sz_u512_vec_t pre_substitution_vec, pre_insert_open_vec, pre_delete_open_vec, pre_insert_expand_vec,
            pre_delete_expand_vec;
        sz_u512_vec_t cost_of_substitution_vec;
        sz_u512_vec_t cost_if_substitution_vec, cost_if_insert, cost_if_delete, cell_score_vec;

        // ? Note that here we are still traversing both buffers in the same order,
        // ? because one of the strings has been reversed beforehand.
        load_mask = sz_u16_mask_until_(n);
        first_vec.xmm = _mm_maskz_loadu_epi8(load_mask, first_reversed_slice);
        second_vec.xmm = _mm_maskz_loadu_epi8(load_mask, second_slice);
        pre_substitution_vec.zmm = _mm512_maskz_loadu_epi32(load_mask, scores_pre_substitution);
        pre_insert_open_vec.zmm = _mm512_maskz_loadu_epi32(load_mask, scores_pre_insertion);
        pre_delete_open_vec.zmm = _mm512_maskz_loadu_epi32(load_mask, scores_pre_deletion);
        pre_insert_expand_vec.zmm = _mm512_maskz_loadu_epi32(load_mask, scores_running_insertions);
        pre_delete_expand_vec.zmm = _mm512_maskz_loadu_epi32(load_mask, scores_running_deletions);

        match_mask = _mm_cmpeq_epi8_mask(first_vec.xmm, second_vec.xmm);
        cost_of_substitution_vec.zmm = _mm512_mask_blend_epi32(match_mask, mismatch_cost_vec.zmm, match_cost_vec.zmm);
        cost_if_substitution_vec.zmm = _mm512_add_epi32(pre_substitution_vec.zmm, cost_of_substitution_vec.zmm);
        cost_if_insert.zmm = _mm512_min_epu32(_mm512_add_epi32(pre_insert_expand_vec.zmm, gap_expand_vec.zmm),
                                              _mm512_add_epi32(pre_insert_open_vec.zmm, gap_open_vec.zmm));
        cost_if_delete.zmm = _mm512_min_epu32(_mm512_add_epi32(pre_delete_expand_vec.zmm, gap_expand_vec.zmm),
                                              _mm512_add_epi32(pre_delete_open_vec.zmm, gap_open_vec.zmm));
        cell_score_vec.zmm =
            _mm512_min_epu32(cost_if_substitution_vec.zmm, _mm512_min_epu32(cost_if_insert.zmm, cost_if_delete.zmm));

        // Export results.
        _mm512_mask_storeu_epi32(scores_new, load_mask, cell_score_vec.zmm);
        _mm512_mask_storeu_epi32(scores_new_insertions, load_mask, cost_if_insert.zmm);
        _mm512_mask_storeu_epi32(scores_new_deletions, load_mask, cost_if_delete.zmm);
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    inline void operator()(                                                              //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        sz_u32_t const *scores_pre_substitution,                                         //
        sz_u32_t const *scores_pre_insertion,                                            //
        sz_u32_t const *scores_pre_deletion,                                             //
        sz_u32_t const *scores_running_insertions,                                       //
        sz_u32_t const *scores_running_deletions,                                        //
        sz_u32_t *scores_new,                                                            //
        sz_u32_t *scores_new_insertions,                                                 //
        sz_u32_t *scores_new_deletions,                                                  //
        executor_type_ &&executor = {}) noexcept {

        // Initialize constats:
        sz_u512_vec_t match_cost_vec, mismatch_cost_vec, gap_open_vec, gap_expand_vec;
        match_cost_vec.zmm = _mm512_set1_epi32(this->substituter_.match);
        mismatch_cost_vec.zmm = _mm512_set1_epi32(this->substituter_.mismatch);
        gap_open_vec.zmm = _mm512_set1_epi32(this->gap_costs_.open);
        gap_expand_vec.zmm = _mm512_set1_epi32(this->gap_costs_.extend);

        // Handle the body in parallel, despite having misaligned writes:
        size_t const body_pages = length / step_k;
        executor.for_n(body_pages, [&](size_t const page) noexcept {
            size_t const progress = page * step_k;
            slice_upto16chars(                                                             //
                first_reversed_slice + progress, second_slice + progress, step_k,          //
                scores_pre_substitution + progress,                                        //
                scores_pre_insertion + progress, scores_pre_deletion + progress,           //
                scores_running_insertions + progress, scores_running_deletions + progress, //
                scores_new + progress,                                                     //
                scores_new_insertions + progress, scores_new_deletions + progress,         //
                match_cost_vec, mismatch_cost_vec, gap_open_vec, gap_expand_vec);
        });

        // Handle the tail:
        size_t const progress = body_pages * step_k;
        size_t const tail = length - progress;
        if (tail)
            slice_upto16chars(                                                        //
                first_reversed_slice + progress, second_slice + progress, tail,       //
                scores_pre_substitution + progress, scores_pre_insertion + progress,  //
                scores_pre_deletion + progress, scores_running_insertions + progress, //
                scores_running_deletions + progress, scores_new + progress,           //
                scores_new_insertions + progress, scores_new_deletions + progress,    //
                match_cost_vec, mismatch_cost_vec, gap_open_vec, gap_expand_vec);

        // The last element of the last chunk is the result of the global alignment.
        if (length == 1) this->last_score_ = scores_new[0];
    }
};

/**
 *  @brief Computes the @b byte-level Levenshtein distance between two strings using the CPU backend.
 *  @sa `levenshtein_distance_utf8` for UTF-8 strings.
 */
template <typename gap_costs_type_, typename allocator_type_, sz_capability_t capability_>
struct levenshtein_distance<char, gap_costs_type_, allocator_type_, capability_,
                            std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>> {

    using char_t = char;
    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_wout_simd_k = (sz_capability_t)(capability_k & ~sz_cap_icelake_k);

    using diagonal_u8_t =                                                                        //
        diagonal_walker<char_t, sz_u8_t, uniform_substitution_costs_t, gap_costs_t, allocator_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_k>;
    using diagonal_u16_t =                                                                        //
        diagonal_walker<char_t, sz_u16_t, uniform_substitution_costs_t, gap_costs_t, allocator_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_k>;
    using diagonal_u32_t =                                                                        //
        diagonal_walker<char_t, sz_u32_t, uniform_substitution_costs_t, gap_costs_t, allocator_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_k>;
    using diagonal_u64_t =                                                                        //
        diagonal_walker<char_t, sz_u64_t, uniform_substitution_costs_t, gap_costs_t, allocator_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_wout_simd_k>;

    uniform_substitution_costs_t substituter_ {};
    gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    levenshtein_distance(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    levenshtein_distance(uniform_substitution_costs_t subs, gap_costs_t gaps,
                         allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score. Pointer-sized for compatibility with C APIs.
     */
    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, size_t &result_ref,
                        executor_type_ &&executor = {}) const noexcept {

        // Estimate the maximum dimension of the DP matrix and choose the best type for it.
        using similarity_memory_requirements_t = similarity_memory_requirements<size_t, false>;
        similarity_memory_requirements_t requirements(                                 //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), SZ_MAX_REGISTER_WIDTH);

        // When dealing with larger arrays, we need to differentiate kernel with different cost aggregation types.
        // Smaller ones will overflow for larger inputs, but using larger-than-needed types will waste memory.
        if (requirements.bytes_per_cell <= 1) {
            sz_u8_t result_u8;
            status_t status = diagonal_u8_t {substituter_, gap_costs_, alloc_}(first, second, result_u8 /* executor */);
            if (status != status_t::success_k) return status;
            result_ref = result_u8;
        }
        else if (requirements.bytes_per_cell == 2) {
            sz_u16_t result_u16;
            status_t status = diagonal_u16_t {substituter_, gap_costs_, alloc_}(first, second, result_u16, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_u16;
        }
        else if (requirements.bytes_per_cell == 4) {
            sz_u32_t result_u32;
            status_t status = diagonal_u32_t {substituter_, gap_costs_, alloc_}(first, second, result_u32, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_u32;
        }
        else if (requirements.bytes_per_cell == 8) {
            sz_u64_t result_u64;
            status_t status = diagonal_u64_t {substituter_, gap_costs_, alloc_}(first, second, result_u64, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_u64;
        }

        return status_t::success_k;
    }
};

/**
 *  @brief Computes the @b rune-level Levenshtein distance between two UTF-8 strings using the CPU backend.
 *  @sa `levenshtein_distance` for binary strings.
 */
template <typename allocator_type_, sz_capability_t capability_>
struct levenshtein_distance_utf8<char, linear_gap_costs_t, allocator_type_, capability_,
                                 std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>> {

    using char_t = char;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using allocator_traits_t = std::allocator_traits<allocator_t>;
    using rune_allocator_t = typename allocator_traits_t::template rebind_alloc<sz_rune_t>;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_wout_simd_k = (sz_capability_t)(capability_k & ~sz_cap_icelake_k);

    using diagonal_u8_t =                                                                                  //
        diagonal_walker<sz_rune_t, sz_u8_t, uniform_substitution_costs_t, linear_gap_costs_t, allocator_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_k>;
    using diagonal_u16_t =                                                                                  //
        diagonal_walker<sz_rune_t, sz_u16_t, uniform_substitution_costs_t, linear_gap_costs_t, allocator_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_k>;
    using diagonal_u32_t =                                                                                  //
        diagonal_walker<sz_rune_t, sz_u32_t, uniform_substitution_costs_t, linear_gap_costs_t, allocator_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_wout_simd_k>;
    using diagonal_u64_t =                                                                                  //
        diagonal_walker<sz_rune_t, sz_u64_t, uniform_substitution_costs_t, linear_gap_costs_t, allocator_t, //
                        sz_minimize_distance_k, sz_similarity_global_k, capability_wout_simd_k>;

    using ascii_fallback_t = levenshtein_distance<char_t, linear_gap_costs_t, allocator_t, capability_k>;

    uniform_substitution_costs_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    mutable allocator_t alloc_ {};

    levenshtein_distance_utf8(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    levenshtein_distance_utf8(uniform_substitution_costs_t subs, linear_gap_costs_t gaps,
                              allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score. Pointer-sized for compatibility with C APIs.
     *  @retval status_t::success_k On successful computation.
     *  @retval status_t::invalid_utf8_k If either input contains invalid UTF-8 sequences.
     *  @retval status_t::bad_alloc_k If memory allocation fails.
     */
    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, size_t &result_ref,
                        executor_type_ &&executor = {}) const noexcept {

        // Check if the strings are entirely composed of ASCII characters,
        // and default to a simpler algorithm in that case.
        if (sz_isascii(first.data(), first.size()) && sz_isascii(second.data(), second.size()))
            return ascii_fallback_t {substituter_, gap_costs_, alloc_}(first, second, result_ref, executor);

        // Allocate some memory to expand UTF-8 strings into UTF-32.
        safe_vector<sz_rune_t, rune_allocator_t> unpacked_utf32(alloc_);
        if (unpacked_utf32.try_resize(first.size() + second.size()) != status_t::success_k)
            return status_t::bad_alloc_k;
        sz_rune_t *const first_data_utf32 = unpacked_utf32.data();
        sz_rune_t *const second_data_utf32 = first_data_utf32 + first.size();

        // Export into UTF-32 buffer.
        sz_rune_length_t rune_length;
        size_t first_length_utf32 = 0, second_length_utf32 = 0;
        for (size_t progress_utf8 = 0, progress_utf32 = 0; progress_utf8 < first.size();
             progress_utf8 += rune_length, ++progress_utf32, ++first_length_utf32) {
            sz_rune_parse(first.data() + progress_utf8, first_data_utf32 + progress_utf32, &rune_length);
            if (rune_length == sz_utf8_invalid_k) return status_t::invalid_utf8_k;
        }
        for (size_t progress_utf8 = 0, progress_utf32 = 0; progress_utf8 < second.size();
             progress_utf8 += rune_length, ++progress_utf32, ++second_length_utf32) {
            sz_rune_parse(second.data() + progress_utf8, second_data_utf32 + progress_utf32, &rune_length);
            if (rune_length == sz_utf8_invalid_k) return status_t::invalid_utf8_k;
        }

        // Estimate the maximum dimension of the DP matrix and choose the best type for it.
        using similarity_memory_requirements_t = similarity_memory_requirements<size_t, false>;
        similarity_memory_requirements_t requirements(                                 //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(sz_rune_t), SZ_MAX_REGISTER_WIDTH);

        // When dealing with larger arrays, we need to differentiate kernel with different cost aggregation types.
        // Smaller ones will overflow for larger inputs, but using larger-than-needed types will waste memory.
        span<sz_rune_t const> const first_utf32 {first_data_utf32, first_length_utf32};
        span<sz_rune_t const> const second_utf32 {second_data_utf32, second_length_utf32};
        if (requirements.bytes_per_cell <= 1) {
            sz_u8_t result_u8;
            status_t status =
                diagonal_u8_t {substituter_, gap_costs_, alloc_}(first_utf32, second_utf32, result_u8 /* executor */);
            if (status != status_t::success_k) return status;
            result_ref = result_u8;
        }
        else if (requirements.bytes_per_cell == 2) {
            sz_u16_t result_u16;
            status_t status =
                diagonal_u16_t {substituter_, gap_costs_, alloc_}(first_utf32, second_utf32, result_u16, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_u16;
        }
        else if (requirements.bytes_per_cell == 4) {
            sz_u32_t result_u32;
            status_t status =
                diagonal_u32_t {substituter_, gap_costs_, alloc_}(first_utf32, second_utf32, result_u32, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_u32;
        }
        else if (requirements.bytes_per_cell == 8) {
            sz_u64_t result_u64;
            status_t status =
                diagonal_u64_t {substituter_, gap_costs_, alloc_}(first_utf32, second_utf32, result_u64, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_u64;
        }

        return status_t::success_k;
    }
};

/**
 *  @brief Helper object optimizing the most expensive part of variable-substitution-cost alignment methods for
 *         Ice Lake CPUs. It's designed for horizontal layout "walkers", where we look at just one row
 *         of (256 x 256) substitution matrix and can fit 256 bytes worth of costs in the registers.
 *
 *  This is a common abstraction for both:
 *  - Local SW and global NW alignment.
 *  - Serial and parallel implementations.
 *  - 8-bit, 16-bit, 32-bit, and even 64-bit costs.
 *  - Any memory allocator used.
 */
struct lookup_in256bytes_icelake_t_ {
    sz_u512_vec_t row_subs_vecs_[4];
    sz_u512_vec_t is_third_or_fourth_vec_, is_second_or_fourth_vec_;

    inline lookup_in256bytes_icelake_t_() noexcept {
        char is_third_or_fourth_check, is_second_or_fourth_check;
        *(sz_u8_t *)&is_third_or_fourth_check = 0x80, *(sz_u8_t *)&is_second_or_fourth_check = 0x40;
        is_third_or_fourth_vec_.zmm = _mm512_set1_epi8(is_third_or_fourth_check);
        is_second_or_fourth_vec_.zmm = _mm512_set1_epi8(is_second_or_fourth_check);
    }

    inline void reload(sz_error_cost_t const *row_subs) noexcept {
        row_subs_vecs_[0].zmm = _mm512_loadu_si512(row_subs + 64 * 0);
        row_subs_vecs_[1].zmm = _mm512_loadu_si512(row_subs + 64 * 1);
        row_subs_vecs_[2].zmm = _mm512_loadu_si512(row_subs + 64 * 2);
        row_subs_vecs_[3].zmm = _mm512_loadu_si512(row_subs + 64 * 3);
    }

    inline sz_u512_vec_t lookup64(sz_u512_vec_t const &text_vec) const noexcept {

        sz_u512_vec_t shuffled_subs_vecs[4];
        sz_u512_vec_t substituted_vec;
        __mmask64 is_third_or_fourth, is_second_or_fourth;

        // Only the bottom 6 bits of a byte are used in `VPERB`, so we don't even need to mask.
        shuffled_subs_vecs[0].zmm = _mm512_permutexvar_epi8(text_vec.zmm, row_subs_vecs_[0].zmm);
        shuffled_subs_vecs[1].zmm = _mm512_permutexvar_epi8(text_vec.zmm, row_subs_vecs_[1].zmm);
        shuffled_subs_vecs[2].zmm = _mm512_permutexvar_epi8(text_vec.zmm, row_subs_vecs_[2].zmm);
        shuffled_subs_vecs[3].zmm = _mm512_permutexvar_epi8(text_vec.zmm, row_subs_vecs_[3].zmm);

        // To blend we can invoke three `_mm512_cmplt_epu8_mask`, but we can also achieve the same using
        // the AND logical operation, checking the top two bits of every byte. Continuing this thought,
        // we can use the `VPTESTMB` instruction to output the mask after the AND.
        is_third_or_fourth = _mm512_test_epi8_mask(text_vec.zmm, is_third_or_fourth_vec_.zmm);
        is_second_or_fourth = _mm512_test_epi8_mask(text_vec.zmm, is_second_or_fourth_vec_.zmm);
        substituted_vec.zmm = _mm512_mask_blend_epi8(
            is_third_or_fourth,
            // Choose between the first and the second.
            _mm512_mask_blend_epi8(is_second_or_fourth, shuffled_subs_vecs[0].zmm, shuffled_subs_vecs[1].zmm),
            // Choose between the third and the fourth.
            _mm512_mask_blend_epi8(is_second_or_fourth, shuffled_subs_vecs[2].zmm, shuffled_subs_vecs[3].zmm));

        return substituted_vec;
    }
};

/**
 *  @brief Helper object for Ice Lake CPUs, designed for horizontal layout "walkers", operating over 16-bit costs.
 *         It's based on the idea, that substitutions are the most expensive part of the algorithm, so those are
 *         parallelized, while the running minimums within a row are computed in a serial fashion.
 *
 *  This is a common abstraction for both:
 *  - Local SW and global NW alignment.
 *  - Serial and parallel implementations.
 *  - Any memory allocator used.
 */
template <sz_similarity_locality_t locality_>
struct tile_scorer<constant_iterator<char>, char const *, sz_i16_t, error_costs_256x256_t, linear_gap_costs_t,
                   sz_maximize_score_k, locality_, sz_cap_icelake_k>
    : public tile_scorer<constant_iterator<char>, char const *, sz_i16_t, error_costs_256x256_t, linear_gap_costs_t,
                         sz_maximize_score_k, locality_, sz_cap_serial_k, void> {

    using tile_scorer<constant_iterator<char>, char const *, sz_i16_t, error_costs_256x256_t, linear_gap_costs_t,
                      sz_maximize_score_k, locality_, sz_cap_serial_k,
                      void>::tile_scorer; // Make the constructors visible

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_icelake_k;

    lookup_in256bytes_icelake_t_ lookup_;

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                   //
        constant_iterator<char> first_char, char const *second_slice, size_t n,        //
        sz_i16_t const *scores_pre_substitution, sz_i16_t const *scores_pre_insertion, //
        sz_i16_t const *scores_pre_deletion, sz_i16_t *scores_new, executor_type_ &&executor = {}) noexcept {

        // Load a new substitution row.
        sz_i16_t const gap = static_cast<sz_i16_t>(this->gap_costs_.open_or_extend);
        error_cost_t const *substitutions_row = &this->substituter_.cells[(sz_u8_t)*first_char][0];
        lookup_.reload(substitutions_row);

        // Progress through the row 64 characters at a time.
        size_t const count_slices = n / 64;
        executor.for_n(count_slices, [&](size_t idx_slice) noexcept {
            slice_64chars(second_slice, idx_slice * 64, gap, scores_pre_substitution, scores_pre_insertion, scores_new);
        });

        // Handle the tail with a less efficient kernel - at most 2 iterations of the following loop:
        for (size_t idx_half_slice = count_slices * 2; idx_half_slice * 32 < n; ++idx_half_slice)
            slice_upto32chars(second_slice, idx_half_slice * 32, n, gap, scores_pre_substitution, scores_pre_insertion,
                              scores_new);

        // Horizontally compute the running minimum of the last row.
        // Simply disabling this operation results in 5x performance improvement, meaning
        //
        // To perform the same operation in vectorized form, we need to perform a tree-like reduction,
        // that will involve multiple steps. It's quite expensive and should be first tested in the
        // "experimental" section.
        sz_assert_(scores_pre_substitution + 1 == scores_pre_insertion && "Expects horizontal traversal of DP matrix");
        sz_assert_(scores_pre_deletion + 1 == scores_new && "Expects horizontal traversal of DP matrix");
        sz_i16_t last_in_row = scores_pre_deletion[0];
        for (size_t i = 0; i < n; ++i) scores_new[i] = last_in_row = sz_max_of_two(scores_new[i], last_in_row + gap);
        this->last_score_ = last_in_row;
    }

    void slice_64chars(char const *second_slice, size_t i, sz_i16_t gap,                              //
                       sz_i16_t const *scores_pre_substitution, sz_i16_t const *scores_pre_insertion, //
                       sz_i16_t *scores_new) const noexcept {

        sz_u512_vec_t second_vec;
        sz_u512_vec_t pre_substitution_vecs[2], pre_gap_vecs[2];
        sz_u512_vec_t cost_of_substitution_i8_vec, cost_of_substitution_i16_vecs[2];
        sz_u512_vec_t cost_if_substitution_vecs[2], cost_if_gap_vecs[2], cell_score_vecs[2];

        // Initialize constats:
        sz_u512_vec_t gap_cost_vec;
        gap_cost_vec.zmm = _mm512_set1_epi16(gap);

        // Load the data without any masks:
        second_vec.zmm = _mm512_loadu_epi8(second_slice + i);
        pre_substitution_vecs[0].zmm = _mm512_loadu_epi16(scores_pre_substitution + i + 0);
        pre_substitution_vecs[1].zmm = _mm512_loadu_epi16(scores_pre_substitution + i + 32);
        pre_gap_vecs[0].zmm = _mm512_loadu_epi16(scores_pre_insertion + i + 0);
        pre_gap_vecs[1].zmm = _mm512_loadu_epi16(scores_pre_insertion + i + 32);

        // First, sign-extend the substitution cost vector.
        cost_of_substitution_i8_vec = lookup_.lookup64(second_vec);
        cost_of_substitution_i16_vecs[0].zmm =
            _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(cost_of_substitution_i8_vec.zmm, 0));
        cost_of_substitution_i16_vecs[1].zmm =
            _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(cost_of_substitution_i8_vec.zmm, 1));

        // Then compute the data-parallel part, assuming the cost of deletions will be propagated
        // left to right outside of this loop.
        cost_if_substitution_vecs[0].zmm =
            _mm512_add_epi16(pre_substitution_vecs[0].zmm, cost_of_substitution_i16_vecs[0].zmm);
        cost_if_substitution_vecs[1].zmm =
            _mm512_add_epi16(pre_substitution_vecs[1].zmm, cost_of_substitution_i16_vecs[1].zmm);
        cost_if_gap_vecs[0].zmm = _mm512_add_epi16(pre_gap_vecs[0].zmm, gap_cost_vec.zmm);
        cost_if_gap_vecs[1].zmm = _mm512_add_epi16(pre_gap_vecs[1].zmm, gap_cost_vec.zmm);
        cell_score_vecs[0].zmm = _mm512_max_epi16(cost_if_substitution_vecs[0].zmm, cost_if_gap_vecs[0].zmm);
        cell_score_vecs[1].zmm = _mm512_max_epi16(cost_if_substitution_vecs[1].zmm, cost_if_gap_vecs[1].zmm);

        // In Local Alignment for SW we also need to compare to zero and set the result to zero if negative.
        if constexpr (locality_ == sz_similarity_local_k)
            cell_score_vecs[0].zmm = _mm512_max_epi16(cell_score_vecs[0].zmm, _mm512_setzero_epi32()),
            cell_score_vecs[1].zmm = _mm512_max_epi16(cell_score_vecs[1].zmm, _mm512_setzero_epi32());

        // Dump partial results to the output buffer.
        _mm512_storeu_epi16(scores_new + i + 0, cell_score_vecs[0].zmm);
        _mm512_storeu_epi16(scores_new + i + 32, cell_score_vecs[1].zmm);
    }

    void slice_upto32chars(char const *second_slice, size_t i, size_t n, sz_i16_t gap,                    //
                           sz_i16_t const *scores_pre_substitution, sz_i16_t const *scores_pre_insertion, //
                           sz_i16_t *scores_new) const noexcept {

        __mmask32 load_mask;
        sz_u512_vec_t second_vec; // ! Only up to 32 bytes in the low YMM section will be used
        sz_u512_vec_t pre_substitution_vec, pre_gap_vec;
        sz_u512_vec_t cost_of_substitution_vec;
        sz_u512_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

        // Initialize constats:
        sz_u512_vec_t gap_cost_vec;
        gap_cost_vec.zmm = _mm512_set1_epi16(gap);

        // Load the data with a mask:
        load_mask = sz_u32_mask_until_(n - i);
        second_vec.ymms[0] = _mm256_maskz_loadu_epi8(load_mask, second_slice + i);
        pre_substitution_vec.zmm = _mm512_maskz_loadu_epi16(load_mask, scores_pre_substitution + i);
        pre_gap_vec.zmm = _mm512_maskz_loadu_epi16(load_mask, scores_pre_insertion + i);

        // First, sign-extend the substitution cost vector.
        cost_of_substitution_vec.zmm = _mm512_cvtepi8_epi16(lookup_.lookup64(second_vec).ymms[0]);

        // Then compute the data-parallel part, assuming the cost of deletions will be propagated
        // left to right outside of this loop.
        cost_if_substitution_vec.zmm = _mm512_add_epi16(pre_substitution_vec.zmm, cost_of_substitution_vec.zmm);
        cost_if_gap_vec.zmm = _mm512_add_epi16(pre_gap_vec.zmm, gap_cost_vec.zmm);
        cell_score_vec.zmm = _mm512_max_epi16(cost_if_substitution_vec.zmm, cost_if_gap_vec.zmm);

        // In Local Alignment for SW we also need to compare to zero and set the result to zero if negative.
        if constexpr (locality_ == sz_similarity_local_k)
            cell_score_vec.zmm = _mm512_max_epi16(cell_score_vec.zmm, _mm512_setzero_epi32());

        // Dump partial results to the output buffer.
        _mm512_mask_storeu_epi16(scores_new + i, load_mask, cell_score_vec.zmm);
    }
};

template <sz_similarity_locality_t locality_>
struct tile_scorer<constant_iterator<char>, char const *, sz_i32_t, error_costs_256x256_t, linear_gap_costs_t,
                   sz_maximize_score_k, locality_, sz_cap_icelake_k, void>
    : public tile_scorer<constant_iterator<char>, char const *, sz_i32_t, error_costs_256x256_t, linear_gap_costs_t,
                         sz_maximize_score_k, locality_, sz_cap_serial_k, void> {

    using tile_scorer<constant_iterator<char>, char const *, sz_i32_t, error_costs_256x256_t, linear_gap_costs_t,
                      sz_maximize_score_k, locality_, sz_cap_serial_k,
                      void>::tile_scorer; // Make the constructors visible

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_icelake_k;

    lookup_in256bytes_icelake_t_ lookup_;

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                   //
        constant_iterator<char> first_char, char const *second_slice, size_t n,        //
        sz_i32_t const *scores_pre_substitution, sz_i32_t const *scores_pre_insertion, //
        sz_i32_t const *scores_pre_deletion, sz_i32_t *scores_new, executor_type_ &&executor = {}) noexcept {

        // Load a new substitution row.
        sz_i32_t const gap = static_cast<sz_i32_t>(this->gap_costs_.open_or_extend);
        error_cost_t const *substitutions_row = &this->substituter_.cells[(sz_u8_t)*first_char][0];
        lookup_.reload(substitutions_row);

        // Progress through the row 64 characters at a time.
        size_t const count_slices = n / 64;
        executor.for_n(count_slices, [&](size_t idx_slice) noexcept {
            slice_64chars(second_slice, idx_slice * 64, gap, scores_pre_substitution, scores_pre_insertion, scores_new);
        });

        // Handle the tail with a less efficient kernel - at most 4 iterations of the following loop:
        for (size_t idx_quarter_slice = count_slices * 4; idx_quarter_slice * 16 < n; ++idx_quarter_slice)
            slice_upto16chars(second_slice, idx_quarter_slice * 16, n, gap, scores_pre_substitution,
                              scores_pre_insertion, scores_new);

        // Horizontally compute the running minimum of the last row.
        // Simply disabling this operation results in 5x performance improvement, meaning
        //
        // To perform the same operation in vectorized form, we need to perform a tree-like reduction,
        // that will involve multiple steps. It's quite expensive and should be first tested in the
        // "experimental" section.
        sz_assert_(scores_pre_substitution + 1 == scores_pre_insertion && "Expects horizontal traversal of DP matrix");
        sz_assert_(scores_pre_deletion + 1 == scores_new && "Expects horizontal traversal of DP matrix");
        sz_i32_t last_in_row = scores_pre_deletion[0];
        for (size_t i = 0; i < n; ++i) scores_new[i] = last_in_row = sz_max_of_two(scores_new[i], last_in_row + gap);
        this->last_score_ = last_in_row;
    }

    void slice_64chars(char const *second_slice, size_t i, sz_i32_t gap,                              //
                       sz_i32_t const *scores_pre_substitution, sz_i32_t const *scores_pre_insertion, //
                       sz_i32_t *scores_new) const noexcept {

        sz_u512_vec_t second_vec;
        sz_u512_vec_t pre_substitution_vecs[4], pre_gap_vecs[4];
        sz_u512_vec_t cost_of_substitution_i8_vec, cost_of_substitution_i32_vecs[4];
        sz_u512_vec_t cost_if_substitution_vecs[4], cost_if_gap_vecs[4], cell_score_vecs[4];

        // Initialize constats:
        sz_u512_vec_t gap_cost_vec;
        gap_cost_vec.zmm = _mm512_set1_epi32(gap);

        // Load the data without any masks:
        second_vec.zmm = _mm512_loadu_epi8(second_slice + i);
        pre_substitution_vecs[0].zmm = _mm512_loadu_epi32(scores_pre_substitution + i + 16 * 0);
        pre_substitution_vecs[1].zmm = _mm512_loadu_epi32(scores_pre_substitution + i + 16 * 1);
        pre_substitution_vecs[2].zmm = _mm512_loadu_epi32(scores_pre_substitution + i + 16 * 2);
        pre_substitution_vecs[3].zmm = _mm512_loadu_epi32(scores_pre_substitution + i + 16 * 3);
        pre_gap_vecs[0].zmm = _mm512_loadu_epi32(scores_pre_insertion + i + 16 * 0);
        pre_gap_vecs[1].zmm = _mm512_loadu_epi32(scores_pre_insertion + i + 16 * 1);
        pre_gap_vecs[2].zmm = _mm512_loadu_epi32(scores_pre_insertion + i + 16 * 2);
        pre_gap_vecs[3].zmm = _mm512_loadu_epi32(scores_pre_insertion + i + 16 * 3);

        // First, sign-extend the substitution cost vector.
        cost_of_substitution_i8_vec = lookup_.lookup64(second_vec);
        cost_of_substitution_i32_vecs[0].zmm =
            _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 0));
        cost_of_substitution_i32_vecs[1].zmm =
            _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 1));
        cost_of_substitution_i32_vecs[2].zmm =
            _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 2));
        cost_of_substitution_i32_vecs[3].zmm =
            _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 3));

        // Then compute the data-parallel part, assuming the cost of deletions will be propagated
        // left to right outside of this loop.
        cost_if_substitution_vecs[0].zmm =
            _mm512_add_epi32(pre_substitution_vecs[0].zmm, cost_of_substitution_i32_vecs[0].zmm);
        cost_if_substitution_vecs[1].zmm =
            _mm512_add_epi32(pre_substitution_vecs[1].zmm, cost_of_substitution_i32_vecs[1].zmm);
        cost_if_substitution_vecs[2].zmm =
            _mm512_add_epi32(pre_substitution_vecs[2].zmm, cost_of_substitution_i32_vecs[2].zmm);
        cost_if_substitution_vecs[3].zmm =
            _mm512_add_epi32(pre_substitution_vecs[3].zmm, cost_of_substitution_i32_vecs[3].zmm);
        cost_if_gap_vecs[0].zmm = _mm512_add_epi32(pre_gap_vecs[0].zmm, gap_cost_vec.zmm);
        cost_if_gap_vecs[1].zmm = _mm512_add_epi32(pre_gap_vecs[1].zmm, gap_cost_vec.zmm);
        cost_if_gap_vecs[2].zmm = _mm512_add_epi32(pre_gap_vecs[2].zmm, gap_cost_vec.zmm);
        cost_if_gap_vecs[3].zmm = _mm512_add_epi32(pre_gap_vecs[3].zmm, gap_cost_vec.zmm);
        cell_score_vecs[0].zmm = _mm512_max_epi32(cost_if_substitution_vecs[0].zmm, cost_if_gap_vecs[0].zmm);
        cell_score_vecs[1].zmm = _mm512_max_epi32(cost_if_substitution_vecs[1].zmm, cost_if_gap_vecs[1].zmm);
        cell_score_vecs[2].zmm = _mm512_max_epi32(cost_if_substitution_vecs[2].zmm, cost_if_gap_vecs[2].zmm);
        cell_score_vecs[3].zmm = _mm512_max_epi32(cost_if_substitution_vecs[3].zmm, cost_if_gap_vecs[3].zmm);

        // In Local Alignment for SW we also need to compare to zero and set the result to zero if negative.
        if constexpr (locality_ == sz_similarity_local_k)
            cell_score_vecs[0].zmm = _mm512_max_epi32(cell_score_vecs[0].zmm, _mm512_setzero_epi32()),
            cell_score_vecs[1].zmm = _mm512_max_epi32(cell_score_vecs[1].zmm, _mm512_setzero_epi32()),
            cell_score_vecs[2].zmm = _mm512_max_epi32(cell_score_vecs[2].zmm, _mm512_setzero_epi32()),
            cell_score_vecs[3].zmm = _mm512_max_epi32(cell_score_vecs[3].zmm, _mm512_setzero_epi32());

        // Dump partial results to the output buffer.
        _mm512_storeu_epi32(scores_new + i + 16 * 0, cell_score_vecs[0].zmm);
        _mm512_storeu_epi32(scores_new + i + 16 * 1, cell_score_vecs[1].zmm);
        _mm512_storeu_epi32(scores_new + i + 16 * 2, cell_score_vecs[2].zmm);
        _mm512_storeu_epi32(scores_new + i + 16 * 3, cell_score_vecs[3].zmm);
    }

    void slice_upto16chars(char const *second_slice, size_t i, size_t n, sz_i32_t gap,                    //
                           sz_i32_t const *scores_pre_substitution, sz_i32_t const *scores_pre_insertion, //
                           sz_i32_t *scores_new) const noexcept {

        __mmask16 load_mask;
        sz_u512_vec_t second_vec; // ! Only up to 16 bytes in the low YMM section will be used
        sz_u512_vec_t pre_substitution_vec, pre_gap_vec;
        sz_u512_vec_t cost_of_substitution_vec;
        sz_u512_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

        // Initialize constats:
        sz_u512_vec_t gap_cost_vec;
        gap_cost_vec.zmm = _mm512_set1_epi32(gap);

        // Load the data with a mask:
        load_mask = sz_u16_clamp_mask_until_(n - i);
        second_vec.xmms[0] = _mm_maskz_loadu_epi8(load_mask, second_slice + i);
        pre_substitution_vec.zmm = _mm512_maskz_loadu_epi32(load_mask, scores_pre_substitution + i);
        pre_gap_vec.zmm = _mm512_maskz_loadu_epi32(load_mask, scores_pre_insertion + i);

        // First, sign-extend the substitution cost vector.
        cost_of_substitution_vec.zmm = _mm512_cvtepi8_epi32(lookup_.lookup64(second_vec).xmms[0]);

        // Then compute the data-parallel part, assuming the cost of deletions will be propagated
        // left to right outside of this loop.
        cost_if_substitution_vec.zmm = _mm512_add_epi32(pre_substitution_vec.zmm, cost_of_substitution_vec.zmm);
        cost_if_gap_vec.zmm = _mm512_add_epi32(pre_gap_vec.zmm, gap_cost_vec.zmm);
        cell_score_vec.zmm = _mm512_max_epi32(cost_if_substitution_vec.zmm, cost_if_gap_vec.zmm);

        // In Local Alignment for SW we also need to compare to zero and set the result to zero if negative.
        if constexpr (locality_ == sz_similarity_local_k)
            cell_score_vec.zmm = _mm512_max_epi32(cell_score_vec.zmm, _mm512_setzero_epi32());

        // Dump partial results to the output buffer.
        _mm512_mask_storeu_epi32(scores_new + i, load_mask, cell_score_vec.zmm);
    }
};

template <sz_similarity_locality_t locality_>
struct tile_scorer<constant_iterator<char>, char const *, sz_i64_t, error_costs_256x256_t, linear_gap_costs_t,
                   sz_maximize_score_k, locality_, sz_cap_icelake_k>
    : public tile_scorer<constant_iterator<char>, char const *, sz_i64_t, error_costs_256x256_t, linear_gap_costs_t,
                         sz_maximize_score_k, locality_, sz_cap_serial_k, void> {

    using tile_scorer<constant_iterator<char>, char const *, sz_i64_t, error_costs_256x256_t, linear_gap_costs_t,
                      sz_maximize_score_k, locality_, sz_cap_serial_k,
                      void>::tile_scorer; // Make the constructors visible
};

/** @brief Redirects the Ice Lake template specialization to the serial version. */
template <typename char_type_, typename score_type_, typename substituter_type_, typename gap_costs_type_,
          typename allocator_type_, sz_similarity_objective_t objective_, sz_similarity_locality_t locality_>
struct horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, allocator_type_, objective_,
                         locality_, sz_cap_icelake_k, void>
    : public horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, allocator_type_, objective_,
                               locality_, sz_cap_serial_k, void> {

    using base_t = horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, allocator_type_,
                                     objective_, locality_, sz_cap_serial_k, void>;

    using base_t::base_t;
    using base_t::operator();
};

/**
 *  @brief Computes the @b byte-level Needleman-Wunsch score between two strings using the Ice Lake backend.
 *  @sa `levenshtein_distance` for uniform substitution and gap costs.
 */
template <typename allocator_type_>
struct needleman_wunsch_score<char, error_costs_256x256_t, linear_gap_costs_t, allocator_type_, sz_caps_sil_k> {

    using char_t = char;
    using substituter_t = error_costs_256x256_t;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;

    using horizontal_i16_t =                                                         //
        horizontal_walker<char_t, sz_i16_t, substituter_t, gap_costs_t, allocator_t, //
                          sz_maximize_score_k, sz_similarity_global_k, sz_cap_icelake_k>;
    using horizontal_i32_t =                                                         //
        horizontal_walker<char_t, sz_i32_t, substituter_t, gap_costs_t, allocator_t, //
                          sz_maximize_score_k, sz_similarity_global_k, sz_cap_icelake_k>;
    using horizontal_i64_t =                                                         //
        horizontal_walker<char_t, sz_i64_t, substituter_t, gap_costs_t, allocator_t, //
                          sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    needleman_wunsch_score(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    needleman_wunsch_score(substituter_t subs, linear_gap_costs_t gaps, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score. Pointer-sized for compatibility with C APIs.
     */
    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, sz_ssize_t &result_ref,
                        executor_type_ &&executor = {}) const noexcept {

        // Estimate the maximum dimension of the DP matrix and choose the best type for it.
        using similarity_memory_requirements_t = similarity_memory_requirements<size_t, true>;
        similarity_memory_requirements_t requirements(                                 //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), SZ_MAX_REGISTER_WIDTH);

        // When dealing with larger arrays, we need to differentiate kernel with different cost aggregation types.
        // Smaller ones will overflow for larger inputs, but using larger-than-needed types will waste memory.
        if (requirements.bytes_per_cell <= 2) {
            sz_i16_t result_i16;
            status_t status = horizontal_i16_t {substituter_, gap_costs_, alloc_}(first, second, result_i16, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            sz_i32_t result_i32;
            status_t status = horizontal_i32_t {substituter_, gap_costs_, alloc_}(first, second, result_i32, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            sz_i64_t result_i64;
            status_t status = horizontal_i64_t {substituter_, gap_costs_, alloc_}(first, second, result_i64, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_i64;
        }

        return status_t::success_k;
    }
};

/**
 *  @brief Computes the @b byte-level Smith-Waterman score between two strings using the Ice Lake backend.
 *  @sa `levenshtein_distance` for uniform substitution and gap costs.
 */
template <typename allocator_type_>
struct smith_waterman_score<char, error_costs_256x256_t, linear_gap_costs_t, allocator_type_, sz_caps_sil_k> {

    using char_t = char;
    using substituter_t = error_costs_256x256_t;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;

    using horizontal_i16_t =                                                                //
        horizontal_walker<char_t, sz_i16_t, substituter_t, linear_gap_costs_t, allocator_t, //
                          sz_maximize_score_k, sz_similarity_local_k, sz_cap_icelake_k>;
    using horizontal_i32_t =                                                                //
        horizontal_walker<char_t, sz_i32_t, substituter_t, linear_gap_costs_t, allocator_t, //
                          sz_maximize_score_k, sz_similarity_local_k, sz_cap_icelake_k>;
    using horizontal_i64_t =                                                                //
        horizontal_walker<char_t, sz_i64_t, substituter_t, linear_gap_costs_t, allocator_t, //
                          sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    smith_waterman_score(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    smith_waterman_score(substituter_t subs, linear_gap_costs_t gaps, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score. Pointer-sized for compatibility with C APIs.
     */
    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, sz_ssize_t &result_ref,
                        executor_type_ &&executor = {}) const noexcept {

        // Estimate the maximum dimension of the DP matrix and choose the best type for it.
        using similarity_memory_requirements_t = similarity_memory_requirements<size_t, true>;
        similarity_memory_requirements_t requirements(                                 //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), SZ_MAX_REGISTER_WIDTH);

        // When dealing with larger arrays, we need to differentiate kernel with different cost aggregation types.
        // Smaller ones will overflow for larger inputs, but using larger-than-needed types will waste memory.
        if (requirements.bytes_per_cell <= 2) {
            sz_i16_t result_i16;
            status_t status = horizontal_i16_t {substituter_, gap_costs_, alloc_}(first, second, result_i16, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            sz_i32_t result_i32;
            status_t status = horizontal_i32_t {substituter_, gap_costs_, alloc_}(first, second, result_i32, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            sz_i64_t result_i64;
            status_t status = horizontal_i64_t {substituter_, gap_costs_, alloc_}(first, second, result_i64, executor);
            if (status != status_t::success_k) return status;
            result_ref = result_i64;
        }

        return status_t::success_k;
    }
};

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_ICELAKE
#pragma endregion // Ice Lake Implementation

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SIMILARITIES_ICELAKE_HPP_
