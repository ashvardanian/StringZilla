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
struct tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 64;

    /**
     *  @brief Computes one diagonal of the `u8` DM matrix for exactly 64 characters,
     *         using unaligned loads, but forcing @b aligned stores.
     */
    SZ_INLINE void slice_aligned64chars(                                       //
        char const *first_reversed_slice, char const *second_slice,            //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion, //
        u8_t const *scores_pre_deletion, u8_t *scores_new,                     //
        u512_vec_t match_cost_vec, u512_vec_t mismatch_cost_vec,               //
        u512_vec_t gap_cost_vec) const noexcept {

        __mmask64 match_mask;
        u512_vec_t first_vec, second_vec;
        u512_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        u512_vec_t cost_of_substitution_vec;
        u512_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

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
        cost_if_gap_vec.zmm = _mm512_add_epi8(_mm512_min_epu8(pre_insert_vec.zmm, pre_delete_vec.zmm),
                                              gap_cost_vec.zmm);
        cell_score_vec.zmm = _mm512_min_epu8(cost_if_substitution_vec.zmm, cost_if_gap_vec.zmm);
        _mm512_store_si512(scores_new, cell_score_vec.zmm);
    }

    /**
     *  @brief Computes one diagonal of the `u8` DM matrix for up to 64 characters,
     *         using unaligned loads and stores.
     */
    SZ_INLINE void slice_upto64chars(                                          //
        char const *first_reversed_slice, char const *second_slice, size_t n,  //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion, //
        u8_t const *scores_pre_deletion, u8_t *scores_new,                     //
        u512_vec_t match_cost_vec, u512_vec_t mismatch_cost_vec,               //
        u512_vec_t gap_cost_vec) const noexcept {

        __mmask64 load_mask, match_mask;
        u512_vec_t first_vec, second_vec;
        u512_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        u512_vec_t cost_of_substitution_vec;
        u512_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

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
        cost_if_gap_vec.zmm = _mm512_add_epi8(_mm512_min_epu8(pre_insert_vec.zmm, pre_delete_vec.zmm),
                                              gap_cost_vec.zmm);
        cell_score_vec.zmm = _mm512_min_epu8(cost_if_substitution_vec.zmm, cost_if_gap_vec.zmm);
        _mm512_mask_storeu_epi8(scores_new, load_mask, cell_score_vec.zmm);
    }

    template <typename executor_type_ = dummy_executor_t>
    inline void operator()(                                                              //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion,           //
        u8_t const *scores_pre_deletion, u8_t *scores_new,                               //
        executor_type_ &&executor = {}) noexcept {

        sz_unused_(executor); // On such small inputs, we don't need to worry about parallelism.

        // Initialize constats:
        u512_vec_t match_cost_vec, mismatch_cost_vec, gap_cost_vec;
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
struct tile_scorer<rune_t const *, rune_t const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<rune_t const *, rune_t const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<rune_t const *, rune_t const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 16;

    /**
     *  @brief Computes one diagonal of the `u8` DM matrix for exactly 16 characters,
     *         using unaligned loads, but forcing @b aligned stores.
     */
    SZ_INLINE void slice_aligned16chars(                                       //
        rune_t const *first_reversed_slice, rune_t const *second_slice,        //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion, //
        u8_t const *scores_pre_deletion, u8_t *scores_new,                     //
        u128_vec_t match_cost_vec, u128_vec_t mismatch_cost_vec,               //
        u128_vec_t gap_cost_vec) const noexcept {

        __mmask16 match_mask;
        u512_vec_t first_vec, second_vec;
        u128_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        u128_vec_t cost_of_substitution_vec;
        u128_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

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
    SZ_INLINE void slice_upto16chars(                                             //
        rune_t const *first_reversed_slice, rune_t const *second_slice, size_t n, //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion,    //
        u8_t const *scores_pre_deletion, u8_t *scores_new,                        //
        u128_vec_t match_cost_vec, u128_vec_t mismatch_cost_vec,                  //
        u128_vec_t gap_cost_vec) const noexcept {

        __mmask16 load_mask, match_mask;
        u512_vec_t first_vec, second_vec;
        u128_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        u128_vec_t cost_of_substitution_vec;
        u128_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

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

    template <typename executor_type_ = dummy_executor_t>
    inline void operator()(                                                                  //
        rune_t const *first_reversed_slice, rune_t const *second_slice, size_t const length, //
        u8_t const *scores_pre_substitution, u8_t const *scores_pre_insertion,               //
        u8_t const *scores_pre_deletion, u8_t *scores_new,                                   //
        executor_type_ &&executor = {}) noexcept {

        sz_unused_(executor); // On such small inputs, we don't need to worry about parallelism.

        // Initialize constats:
        u128_vec_t match_cost_vec, mismatch_cost_vec, gap_cost_vec;
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
struct tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 32;

    /**
     *  @brief Computes one diagonal of the `u16` DM matrix for exactly 16 characters,
     *         using unaligned loads, but forcing @b aligned stores.
     */
    SZ_INLINE void slice_aligned32chars(                                         //
        char const *first_reversed_slice, char const *second_slice,              //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion, //
        u16_t const *scores_pre_deletion, u16_t *scores_new,                     //
        u512_vec_t match_cost_vec, u512_vec_t mismatch_cost_vec, u512_vec_t gap_cost_vec) const noexcept {

        __mmask32 match_mask;
        u256_vec_t first_vec, second_vec;
        u512_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        u512_vec_t cost_of_substitution_vec;
        u512_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

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
        cost_if_gap_vec.zmm = _mm512_add_epi16(_mm512_min_epu16(pre_insert_vec.zmm, pre_delete_vec.zmm),
                                               gap_cost_vec.zmm);
        cell_score_vec.zmm = _mm512_min_epu16(cost_if_substitution_vec.zmm, cost_if_gap_vec.zmm);
        _mm512_store_si512(scores_new, cell_score_vec.zmm);
    }

    SZ_INLINE void slice_upto32chars(                                            //
        char const *first_reversed_slice, char const *second_slice, size_t n,    //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion, //
        u16_t const *scores_pre_deletion, u16_t *scores_new,                     //
        u512_vec_t match_cost_vec, u512_vec_t mismatch_cost_vec, u512_vec_t gap_cost_vec) const noexcept {

        __mmask32 load_mask, match_mask;
        u256_vec_t first_vec, second_vec;
        u512_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        u512_vec_t cost_of_substitution_vec;
        u512_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

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
        cost_if_gap_vec.zmm = _mm512_add_epi16(_mm512_min_epu16(pre_insert_vec.zmm, pre_delete_vec.zmm),
                                               gap_cost_vec.zmm);
        cell_score_vec.zmm = _mm512_min_epu16(cost_if_substitution_vec.zmm, cost_if_gap_vec.zmm);
        _mm512_mask_storeu_epi16(scores_new, load_mask, cell_score_vec.zmm);
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                             //
        char const *first_reversed_slice, char const *second_slice,                       //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion,          //
        u16_t const *scores_pre_deletion, u16_t *scores_new,                              //
        u512_vec_t match_cost_vec, u512_vec_t mismatch_cost_vec, u512_vec_t gap_cost_vec, //
        size_t from, size_t to) noexcept {

        for (size_t page = from; page < to; ++page) {
            size_t const progress = page * step_k;
            slice_aligned32chars(                                                                             //
                first_reversed_slice + progress, second_slice + progress, scores_pre_substitution + progress, //
                scores_pre_insertion + progress, scores_pre_deletion + progress, scores_new + progress,       //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    inline void operator()(                                                              //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion,         //
        u16_t const *scores_pre_deletion, u16_t *scores_new,                             //
        executor_type_ &&executor = {}) noexcept {

        // Initialize constats:
        u512_vec_t match_cost_vec, mismatch_cost_vec, gap_cost_vec;
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
        executor.for_slices(body_pages, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_slice, second_slice, scores_pre_substitution, scores_pre_insertion,
                                    scores_pre_deletion, scores_new, match_cost_vec, mismatch_cost_vec, gap_cost_vec,
                                    from, to);
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
struct tile_scorer<rune_t const *, rune_t const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<rune_t const *, rune_t const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<rune_t const *, rune_t const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 16;

    /**
     *  @brief Computes one diagonal of the `u16` DM matrix for exactly 16 characters,
     *         using unaligned loads, but forcing @b aligned stores.
     */
    SZ_INLINE void slice_aligned16chars(                                         //
        rune_t const *first_reversed_slice, rune_t const *second_slice,          //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion, //
        u16_t const *scores_pre_deletion, u16_t *scores_new,                     //
        u256_vec_t match_cost_vec, u256_vec_t mismatch_cost_vec, u256_vec_t gap_cost_vec) const noexcept {

        __mmask16 match_mask;
        u512_vec_t first_vec, second_vec;
        u256_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        u256_vec_t cost_of_substitution_vec;
        u256_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

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
        cost_if_gap_vec.ymm = _mm256_add_epi16(_mm256_min_epu16(pre_insert_vec.ymm, pre_delete_vec.ymm),
                                               gap_cost_vec.ymm);
        cell_score_vec.ymm = _mm256_min_epu16(cost_if_substitution_vec.ymm, cost_if_gap_vec.ymm);
        _mm256_store_si256((__m256i *)scores_new, cell_score_vec.ymm);
    }

    /**
     *  @brief Computes one diagonal of the `u16` DM matrix for up to 16 characters,
     *         using unaligned loads and stores.
     */
    SZ_INLINE void slice_upto16chars(                                             //
        rune_t const *first_reversed_slice, rune_t const *second_slice, size_t n, //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion,  //
        u16_t const *scores_pre_deletion, u16_t *scores_new,                      //
        u256_vec_t match_cost_vec, u256_vec_t mismatch_cost_vec, u256_vec_t gap_cost_vec) const noexcept {

        __mmask16 load_mask, match_mask;
        u512_vec_t first_vec, second_vec;
        u256_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        u256_vec_t cost_of_substitution_vec;
        u256_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

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
        cost_if_gap_vec.ymm = _mm256_add_epi16(_mm256_min_epu16(pre_insert_vec.ymm, pre_delete_vec.ymm),
                                               gap_cost_vec.ymm);
        cell_score_vec.ymm = _mm256_min_epu16(cost_if_substitution_vec.ymm, cost_if_gap_vec.ymm);
        _mm256_mask_storeu_epi16(scores_new, load_mask, cell_score_vec.ymm);
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                             //
        rune_t const *first_reversed_slice, rune_t const *second_slice,                   //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion,          //
        u16_t const *scores_pre_deletion, u16_t *scores_new,                              //
        u256_vec_t match_cost_vec, u256_vec_t mismatch_cost_vec, u256_vec_t gap_cost_vec, //
        size_t from, size_t to) noexcept {

        for (size_t page = from; page < to; ++page) {
            size_t const progress = page * step_k;
            slice_aligned16chars(                                                                             //
                first_reversed_slice + progress, second_slice + progress, scores_pre_substitution + progress, //
                scores_pre_insertion + progress, scores_pre_deletion + progress, scores_new + progress,       //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    inline void operator()(                                                                  //
        rune_t const *first_reversed_slice, rune_t const *second_slice, size_t const length, //
        u16_t const *scores_pre_substitution, u16_t const *scores_pre_insertion,             //
        u16_t const *scores_pre_deletion, u16_t *scores_new,                                 //
        executor_type_ &&executor = {}) noexcept {

        // Initialize constats:
        u256_vec_t match_cost_vec, mismatch_cost_vec, gap_cost_vec;
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
        executor.for_slices(body_pages, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_slice, second_slice, scores_pre_substitution, scores_pre_insertion,
                                    scores_pre_deletion, scores_new, match_cost_vec, mismatch_cost_vec, gap_cost_vec,
                                    from, to);
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
struct tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 16;

    /**
     *  @brief Computes one diagonal of the `u32` DM matrix for exactly 16 characters,
     *         using unaligned loads, but forcing @b aligned stores.
     */
    SZ_INLINE void slice_aligned16chars(                                         //
        char const *first_reversed_slice, char const *second_slice,              //
        u32_t const *scores_pre_substitution, u32_t const *scores_pre_insertion, //
        u32_t const *scores_pre_deletion, u32_t *scores_new,                     //
        u512_vec_t match_cost_vec, u512_vec_t mismatch_cost_vec, u512_vec_t gap_cost_vec) const noexcept {

        __mmask16 match_mask;
        u128_vec_t first_vec, second_vec;
        u512_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        u512_vec_t cost_of_substitution_vec;
        u512_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

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
        cost_if_gap_vec.zmm = _mm512_add_epi32(_mm512_min_epu32(pre_insert_vec.zmm, pre_delete_vec.zmm),
                                               gap_cost_vec.zmm);
        cell_score_vec.zmm = _mm512_min_epu32(cost_if_substitution_vec.zmm, cost_if_gap_vec.zmm);
        _mm512_store_si512((__m512i *)scores_new, cell_score_vec.zmm);
    }

    /**
     *  @brief Computes one diagonal of the `u32` DM matrix for up to 16 characters,
     *         using unaligned loads and stores.
     */
    SZ_INLINE void slice_upto16chars(                                            //
        char const *first_reversed_slice, char const *second_slice, size_t n,    //
        u32_t const *scores_pre_substitution, u32_t const *scores_pre_insertion, //
        u32_t const *scores_pre_deletion, u32_t *scores_new,                     //
        u512_vec_t match_cost_vec, u512_vec_t mismatch_cost_vec, u512_vec_t gap_cost_vec) const noexcept {

        __mmask16 load_mask, match_mask;
        u128_vec_t first_vec, second_vec;
        u512_vec_t pre_substitution_vec, pre_insert_vec, pre_delete_vec;
        u512_vec_t cost_of_substitution_vec;
        u512_vec_t cost_if_substitution_vec, cost_if_gap_vec, cell_score_vec;

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
        cost_if_gap_vec.zmm = _mm512_add_epi32(_mm512_min_epu32(pre_insert_vec.zmm, pre_delete_vec.zmm),
                                               gap_cost_vec.zmm);
        cell_score_vec.zmm = _mm512_min_epu32(cost_if_substitution_vec.zmm, cost_if_gap_vec.zmm);
        _mm512_mask_storeu_epi32(scores_new, load_mask, cell_score_vec.zmm);
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                             //
        char const *first_reversed_slice, char const *second_slice,                       //
        u32_t const *scores_pre_substitution, u32_t const *scores_pre_insertion,          //
        u32_t const *scores_pre_deletion, u32_t *scores_new,                              //
        u512_vec_t match_cost_vec, u512_vec_t mismatch_cost_vec, u512_vec_t gap_cost_vec, //
        size_t from, size_t to) noexcept {

        for (size_t page = from; page < to; ++page) {
            size_t const progress = page * step_k;
            slice_aligned16chars(                                                                             //
                first_reversed_slice + progress, second_slice + progress, scores_pre_substitution + progress, //
                scores_pre_insertion + progress, scores_pre_deletion + progress, scores_new + progress,       //
                match_cost_vec, mismatch_cost_vec, gap_cost_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    inline void operator()(                                                              //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        u32_t const *scores_pre_substitution, u32_t const *scores_pre_insertion,         //
        u32_t const *scores_pre_deletion, u32_t *scores_new,                             //
        executor_type_ &&executor = {}) noexcept {

        // Initialize constats:
        u512_vec_t match_cost_vec, mismatch_cost_vec, gap_cost_vec;
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
        executor.for_slices(body_pages, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_slice, second_slice, scores_pre_substitution, scores_pre_insertion,
                                    scores_pre_deletion, scores_new, match_cost_vec, mismatch_cost_vec, gap_cost_vec,
                                    from, to);
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
struct tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
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
        u8_t const *scores_pre_substitution,                                  //
        u8_t const *scores_pre_insertion,                                     //
        u8_t const *scores_pre_deletion,                                      //
        u8_t const *scores_running_insertions,                                //
        u8_t const *scores_running_deletions,                                 //
        u8_t *scores_new,                                                     //
        u8_t *scores_new_insertions,                                          //
        u8_t *scores_new_deletions,                                           //
        u512_vec_t match_cost_vec, u512_vec_t mismatch_cost_vec,              //
        u512_vec_t gap_open_vec, u512_vec_t gap_expand_vec) const noexcept {

        __mmask64 load_mask, match_mask;
        u512_vec_t first_vec, second_vec;
        u512_vec_t pre_substitution_vec, pre_insert_open_vec, pre_delete_open_vec, pre_insert_expand_vec,
            pre_delete_expand_vec;
        u512_vec_t cost_of_substitution_vec;
        u512_vec_t cost_if_substitution_vec, cost_if_insert, cost_if_delete, cell_score_vec;

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
        cell_score_vec.zmm = _mm512_min_epu8(cost_if_substitution_vec.zmm,
                                             _mm512_min_epu8(cost_if_insert.zmm, cost_if_delete.zmm));

        // Export results.
        _mm512_mask_storeu_epi8(scores_new, load_mask, cell_score_vec.zmm);
        _mm512_mask_storeu_epi8(scores_new_insertions, load_mask, cost_if_insert.zmm);
        _mm512_mask_storeu_epi8(scores_new_deletions, load_mask, cost_if_delete.zmm);
    }

    template <typename executor_type_ = dummy_executor_t>
    inline void operator()(                                                              //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        u8_t const *scores_pre_substitution,                                             //
        u8_t const *scores_pre_insertion,                                                //
        u8_t const *scores_pre_deletion,                                                 //
        u8_t const *scores_running_insertions,                                           //
        u8_t const *scores_running_deletions,                                            //
        u8_t *scores_new,                                                                //
        u8_t *scores_new_insertions,                                                     //
        u8_t *scores_new_deletions,                                                      //
        executor_type_ &&executor = {}) noexcept {

        sz_unused_(executor); // On such small inputs, we don't need to worry about parallelism.

        // Initialize constats:
        u512_vec_t match_cost_vec, mismatch_cost_vec, gap_open_vec, gap_expand_vec;
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
struct tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                      sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 32;

    SZ_INLINE void slice_upto32chars(                                         //
        char const *first_reversed_slice, char const *second_slice, size_t n, //
        u16_t const *scores_pre_substitution,                                 //
        u16_t const *scores_pre_insertion,                                    //
        u16_t const *scores_pre_deletion,                                     //
        u16_t const *scores_running_insertions,                               //
        u16_t const *scores_running_deletions,                                //
        u16_t *scores_new,                                                    //
        u16_t *scores_new_insertions,                                         //
        u16_t *scores_new_deletions,                                          //
        u512_vec_t match_cost_vec, u512_vec_t mismatch_cost_vec,              //
        u512_vec_t gap_open_vec, u512_vec_t gap_expand_vec) const noexcept {

        __mmask32 load_mask, match_mask;
        u256_vec_t first_vec, second_vec;
        u512_vec_t pre_substitution_vec, pre_insert_open_vec, pre_delete_open_vec, pre_insert_expand_vec,
            pre_delete_expand_vec;
        u512_vec_t cost_of_substitution_vec;
        u512_vec_t cost_if_substitution_vec, cost_if_insert, cost_if_delete, cell_score_vec;

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
        cell_score_vec.zmm = _mm512_min_epu16(cost_if_substitution_vec.zmm,
                                              _mm512_min_epu16(cost_if_insert.zmm, cost_if_delete.zmm));

        // Export results.
        _mm512_mask_storeu_epi16(scores_new, load_mask, cell_score_vec.zmm);
        _mm512_mask_storeu_epi16(scores_new_insertions, load_mask, cost_if_insert.zmm);
        _mm512_mask_storeu_epi16(scores_new_deletions, load_mask, cost_if_delete.zmm);
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                             //
        char const *first_reversed_slice, char const *second_slice,                       //
        u16_t const *scores_pre_substitution,                                             //
        u16_t const *scores_pre_insertion,                                                //
        u16_t const *scores_pre_deletion,                                                 //
        u16_t const *scores_running_insertions,                                           //
        u16_t const *scores_running_deletions,                                            //
        u16_t *scores_new,                                                                //
        u16_t *scores_new_insertions,                                                     //
        u16_t *scores_new_deletions,                                                      //
        u512_vec_t match_cost_vec, u512_vec_t mismatch_cost_vec, u512_vec_t gap_open_vec, //
        u512_vec_t gap_expand_vec, size_t from, size_t to) noexcept {

        for (size_t page = from; page < to; ++page) {
            size_t const progress = page * step_k;
            slice_upto32chars(                                                                                       //
                first_reversed_slice + progress, second_slice + progress, step_k,                                    //
                scores_pre_substitution + progress, scores_pre_insertion + progress, scores_pre_deletion + progress, //
                scores_running_insertions + progress, scores_running_deletions + progress,                           //
                scores_new + progress, scores_new_insertions + progress, scores_new_deletions + progress,            //
                match_cost_vec, mismatch_cost_vec, gap_open_vec, gap_expand_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    inline void operator()(                                                              //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        u16_t const *scores_pre_substitution,                                            //
        u16_t const *scores_pre_insertion,                                               //
        u16_t const *scores_pre_deletion,                                                //
        u16_t const *scores_running_insertions,                                          //
        u16_t const *scores_running_deletions,                                           //
        u16_t *scores_new,                                                               //
        u16_t *scores_new_insertions,                                                    //
        u16_t *scores_new_deletions,                                                     //
        executor_type_ &&executor = {}) noexcept {

        // Initialize constats:
        u512_vec_t match_cost_vec, mismatch_cost_vec, gap_open_vec, gap_expand_vec;
        match_cost_vec.zmm = _mm512_set1_epi16(this->substituter_.match);
        mismatch_cost_vec.zmm = _mm512_set1_epi16(this->substituter_.mismatch);
        gap_open_vec.zmm = _mm512_set1_epi16(this->gap_costs_.open);
        gap_expand_vec.zmm = _mm512_set1_epi16(this->gap_costs_.extend);

        // In this variant we will need at most (64 * 1024 / 32) = 2048 loops per diagonal.
        size_t const body_pages = length / step_k;
        executor.for_slices(body_pages, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_slice, second_slice, scores_pre_substitution, scores_pre_insertion,
                                    scores_pre_deletion, scores_running_insertions, scores_running_deletions,
                                    scores_new, scores_new_insertions, scores_new_deletions, match_cost_vec,
                                    mismatch_cost_vec, gap_open_vec, gap_expand_vec, from, to);
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
struct tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, capability_,
                   std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
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
        u32_t const *scores_pre_substitution,                                 //
        u32_t const *scores_pre_insertion,                                    //
        u32_t const *scores_pre_deletion,                                     //
        u32_t const *scores_running_insertions,                               //
        u32_t const *scores_running_deletions,                                //
        u32_t *scores_new,                                                    //
        u32_t *scores_new_insertions,                                         //
        u32_t *scores_new_deletions,                                          //
        u512_vec_t match_cost_vec, u512_vec_t mismatch_cost_vec,              //
        u512_vec_t gap_open_vec, u512_vec_t gap_expand_vec) const noexcept {

        __mmask16 load_mask, match_mask;
        u128_vec_t first_vec, second_vec;
        u512_vec_t pre_substitution_vec, pre_insert_open_vec, pre_delete_open_vec, pre_insert_expand_vec,
            pre_delete_expand_vec;
        u512_vec_t cost_of_substitution_vec;
        u512_vec_t cost_if_substitution_vec, cost_if_insert, cost_if_delete, cell_score_vec;

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
        cell_score_vec.zmm = _mm512_min_epu32(cost_if_substitution_vec.zmm,
                                              _mm512_min_epu32(cost_if_insert.zmm, cost_if_delete.zmm));

        // Export results.
        _mm512_mask_storeu_epi32(scores_new, load_mask, cell_score_vec.zmm);
        _mm512_mask_storeu_epi32(scores_new_insertions, load_mask, cost_if_insert.zmm);
        _mm512_mask_storeu_epi32(scores_new_deletions, load_mask, cost_if_delete.zmm);
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                             //
        char const *first_reversed_slice, char const *second_slice,                       //
        u32_t const *scores_pre_substitution,                                             //
        u32_t const *scores_pre_insertion,                                                //
        u32_t const *scores_pre_deletion,                                                 //
        u32_t const *scores_running_insertions,                                           //
        u32_t const *scores_running_deletions,                                            //
        u32_t *scores_new,                                                                //
        u32_t *scores_new_insertions,                                                     //
        u32_t *scores_new_deletions,                                                      //
        u512_vec_t match_cost_vec, u512_vec_t mismatch_cost_vec, u512_vec_t gap_open_vec, //
        u512_vec_t gap_expand_vec, size_t from, size_t to) noexcept {

        for (size_t page = from; page < to; ++page) {
            size_t const progress = page * step_k;
            slice_upto16chars(                                                             //
                first_reversed_slice + progress, second_slice + progress, step_k,          //
                scores_pre_substitution + progress,                                        //
                scores_pre_insertion + progress, scores_pre_deletion + progress,           //
                scores_running_insertions + progress, scores_running_deletions + progress, //
                scores_new + progress,                                                     //
                scores_new_insertions + progress, scores_new_deletions + progress,         //
                match_cost_vec, mismatch_cost_vec, gap_open_vec, gap_expand_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    inline void operator()(                                                              //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        u32_t const *scores_pre_substitution,                                            //
        u32_t const *scores_pre_insertion,                                               //
        u32_t const *scores_pre_deletion,                                                //
        u32_t const *scores_running_insertions,                                          //
        u32_t const *scores_running_deletions,                                           //
        u32_t *scores_new,                                                               //
        u32_t *scores_new_insertions,                                                    //
        u32_t *scores_new_deletions,                                                     //
        executor_type_ &&executor = {}) noexcept {

        // Initialize constats:
        u512_vec_t match_cost_vec, mismatch_cost_vec, gap_open_vec, gap_expand_vec;
        match_cost_vec.zmm = _mm512_set1_epi32(this->substituter_.match);
        mismatch_cost_vec.zmm = _mm512_set1_epi32(this->substituter_.mismatch);
        gap_open_vec.zmm = _mm512_set1_epi32(this->gap_costs_.open);
        gap_expand_vec.zmm = _mm512_set1_epi32(this->gap_costs_.extend);

        // Handle the body in parallel, despite having misaligned writes:
        size_t const body_pages = length / step_k;
        executor.for_slices(body_pages, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_slice, second_slice, scores_pre_substitution, scores_pre_insertion,
                                    scores_pre_deletion, scores_running_insertions, scores_running_deletions,
                                    scores_new, scores_new_insertions, scores_new_deletions, match_cost_vec,
                                    mismatch_cost_vec, gap_open_vec, gap_expand_vec, from, to);
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
template <typename gap_costs_type_, sz_capability_t capability_>
struct levenshtein_distance<char, gap_costs_type_, capability_,
                            std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>> {

    using char_t = char;
    using gap_costs_t = gap_costs_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_wout_simd_k = (sz_capability_t)(capability_k & ~sz_cap_icelake_k);

    using diagonal_u8_t = diagonal_walker<char_t, u8_t, uniform_substitution_costs_t, gap_costs_t,
                                          sz_minimize_distance_k, sz_similarity_global_k, capability_k>;
    using diagonal_u16_t = diagonal_walker<char_t, u16_t, uniform_substitution_costs_t, gap_costs_t,
                                           sz_minimize_distance_k, sz_similarity_global_k, capability_k>;
    using diagonal_u32_t = diagonal_walker<char_t, u32_t, uniform_substitution_costs_t, gap_costs_t,
                                           sz_minimize_distance_k, sz_similarity_global_k, capability_k>;
    using diagonal_u64_t = diagonal_walker<char_t, u64_t, uniform_substitution_costs_t, gap_costs_t,
                                           sz_minimize_distance_k, sz_similarity_global_k, capability_wout_simd_k>;

    uniform_substitution_costs_t substituter_ {};
    gap_costs_t gap_costs_ {};

    levenshtein_distance() noexcept {}
    levenshtein_distance(uniform_substitution_costs_t subs, gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    size_t scratch_space_needed(span<char_t const> first, span<char_t const> second,
                                cpu_specs_t const &specs) const noexcept {
        using diagonal_memory_requirements_t = diagonal_memory_requirements<size_t>;
        diagonal_memory_requirements_t requirements(                                   //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), specs.cache_line_width);
        return requirements.total;
    }

    template <typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, size_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &executor,
                        cpu_specs_t const &specs) const noexcept {

        using diagonal_memory_requirements_t = diagonal_memory_requirements<size_t>;
        diagonal_memory_requirements_t requirements(                                   //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), specs.cache_line_width);

        if (requirements.bytes_per_cell <= 1) {
            u8_t result_u8;
            status_t status = diagonal_u8_t {substituter_, gap_costs_}(first, second, result_u8, scratch_space,
                                                                       executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_u8;
        }
        else if (requirements.bytes_per_cell == 2) {
            u16_t result_u16;
            status_t status = diagonal_u16_t {substituter_, gap_costs_}(first, second, result_u16, scratch_space,
                                                                        executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_u16;
        }
        else if (requirements.bytes_per_cell == 4) {
            u32_t result_u32;
            status_t status = diagonal_u32_t {substituter_, gap_costs_}(first, second, result_u32, scratch_space,
                                                                        executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_u32;
        }
        else if (requirements.bytes_per_cell == 8) {
            u64_t result_u64;
            status_t status = diagonal_u64_t {substituter_, gap_costs_}(first, second, result_u64, scratch_space,
                                                                        executor, specs);
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
template <sz_capability_t capability_>
struct levenshtein_distance_utf8<linear_gap_costs_t, capability_,
                                 std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>> {

    using char_t = char;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_wout_simd_k = (sz_capability_t)(capability_k & ~sz_cap_icelake_k);

    using diagonal_u8_t = diagonal_walker<rune_t, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                                          sz_minimize_distance_k, sz_similarity_global_k, capability_k>;
    using diagonal_u16_t = diagonal_walker<rune_t, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                                           sz_minimize_distance_k, sz_similarity_global_k, capability_k>;
    using diagonal_u32_t = diagonal_walker<rune_t, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                                           sz_minimize_distance_k, sz_similarity_global_k, capability_wout_simd_k>;
    using diagonal_u64_t = diagonal_walker<rune_t, u64_t, uniform_substitution_costs_t, linear_gap_costs_t,
                                           sz_minimize_distance_k, sz_similarity_global_k, capability_wout_simd_k>;

    using ascii_fallback_t = levenshtein_distance<char_t, linear_gap_costs_t, capability_k>;

    uniform_substitution_costs_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    levenshtein_distance_utf8() noexcept {}
    levenshtein_distance_utf8(uniform_substitution_costs_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    size_t scratch_space_needed(span<char_t const> first, span<char_t const> second,
                                cpu_specs_t const &specs) const noexcept {
        size_t const first_unpacking_ceiling = round_up_to_multiple(sizeof(rune_t) * first.size(),
                                                                    specs.cache_line_width);
        size_t const second_unpacking_ceiling = round_up_to_multiple(sizeof(rune_t) * second.size(),
                                                                     specs.cache_line_width);
        return ascii_fallback_t {substituter_, gap_costs_}.scratch_space_needed(first, second, specs) +
               first_unpacking_ceiling + second_unpacking_ceiling;
    }

    template <typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, size_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &executor,
                        cpu_specs_t const &specs) const noexcept {

        // Check if the strings are entirely composed of ASCII characters,
        // and default to a simpler algorithm in that case.
        if (sz_isascii(first.data(), first.size()) && sz_isascii(second.data(), second.size()))
            return ascii_fallback_t {substituter_, gap_costs_}(first, second, result_ref, scratch_space, executor,
                                                               specs);

        // Carve the transcode region off the front of scratch, then pass the remainder to walkers.
        size_t const first_unpacking_ceiling = round_up_to_multiple(sizeof(rune_t) * first.size(),
                                                                    specs.cache_line_width);
        size_t const second_unpacking_ceiling = round_up_to_multiple(sizeof(rune_t) * second.size(),
                                                                     specs.cache_line_width);
        size_t const transcode_bytes = first_unpacking_ceiling + second_unpacking_ceiling;
        if (scratch_space.size() < transcode_bytes) return status_t::bad_alloc_k;
        rune_t *const first_data_utf32 = reinterpret_cast<rune_t *>(scratch_space.data());
        rune_t *const second_data_utf32 = reinterpret_cast<rune_t *>(scratch_space.data() + first_unpacking_ceiling);
        scratch_space_t const walker_scratch = scratch_space.subspan(transcode_bytes,
                                                                     scratch_space.size() - transcode_bytes);

        // Export into UTF-32 buffer.
        rune_length_t rune_length;
        size_t first_length_utf32 = 0, second_length_utf32 = 0;
        for (size_t progress_utf8 = 0; progress_utf8 < first.size();
             progress_utf8 += rune_length, ++first_length_utf32) {
            sz_rune_parse(first.data() + progress_utf8, first_data_utf32 + first_length_utf32, &rune_length);
            if (rune_length == sz_utf8_invalid_k) return status_t::invalid_utf8_k;
        }
        for (size_t progress_utf8 = 0; progress_utf8 < second.size();
             progress_utf8 += rune_length, ++second_length_utf32) {
            sz_rune_parse(second.data() + progress_utf8, second_data_utf32 + second_length_utf32, &rune_length);
            if (rune_length == sz_utf8_invalid_k) return status_t::invalid_utf8_k;
        }

        using diagonal_memory_requirements_t = diagonal_memory_requirements<size_t>;
        diagonal_memory_requirements_t requirements(                                   //
            first_length_utf32, second_length_utf32,                                   //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(rune_t), specs.cache_line_width);

        span<rune_t const> const first_utf32 {first_data_utf32, first_length_utf32};
        span<rune_t const> const second_utf32 {second_data_utf32, second_length_utf32};
        if (requirements.bytes_per_cell <= 1) {
            u8_t result_u8;
            status_t status = diagonal_u8_t {substituter_, gap_costs_}(first_utf32, second_utf32, result_u8,
                                                                       walker_scratch, executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_u8;
        }
        else if (requirements.bytes_per_cell == 2) {
            u16_t result_u16;
            status_t status = diagonal_u16_t {substituter_, gap_costs_}(first_utf32, second_utf32, result_u16,
                                                                        walker_scratch, executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_u16;
        }
        else if (requirements.bytes_per_cell == 4) {
            u32_t result_u32;
            status_t status = diagonal_u32_t {substituter_, gap_costs_}(first_utf32, second_utf32, result_u32,
                                                                        walker_scratch, executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_u32;
        }
        else if (requirements.bytes_per_cell == 8) {
            u64_t result_u64;
            status_t status = diagonal_u64_t {substituter_, gap_costs_}(first_utf32, second_utf32, result_u64,
                                                                        walker_scratch, executor, specs);
            if (status != status_t::success_k) return status;
            result_ref = result_u64;
        }

        return status_t::success_k;
    }
};

/**
 *  @brief Helper object for Ice Lake CPUs, designed for @b diagonal layout "walkers", where @b both operands
 *         of every substitution vary per cell, so a single resident cost row is not enough and the entire
 *         (32 x 32) `class_substitution_costs` matrix must stay resident.
 *
 *  The byte-to-class mapping uses the 256-entry `byte_to_class` table (the 4x `VPERMB`-blend technique, since
 *  a single `VPERMB` register only holds 64 bytes) and is exposed via `classify64`,
 *  so diagonal walkers can pre-classify both strings @b once and feed class-index buffers into the hot loop. The
 *  second stage looks up the cost for two varying class operands at once by keeping the matrix folded into 16
 *  loop-invariant 64-byte windows, addressed as `idx = ((first_class & 1) << 5) | second_class` with the window
 *  `window = first_class >> 1`, resolved through 16x `VPERMB` and a balanced tree of mask-blends.
 *
 *  This is a common abstraction for both:
 *  - Local SW and global NW alignment.
 *  - Serial and parallel implementations.
 *  - 16-bit and 32-bit costs.
 *  - Any memory allocator used.
 */
struct substitution_lookup_icelake_t {
    u512_vec_t byte_to_class_vecs_[4];
    u512_vec_t is_third_or_fourth_vec_, is_second_or_fourth_vec_;
    u512_vec_t cost_windows_vecs_[16];

    inline substitution_lookup_icelake_t() noexcept {
        char is_third_or_fourth_check, is_second_or_fourth_check;
        *(u8_t *)&is_third_or_fourth_check = 0x80, *(u8_t *)&is_second_or_fourth_check = 0x40;
        is_third_or_fourth_vec_.zmm = _mm512_set1_epi8(is_third_or_fourth_check);
        is_second_or_fourth_vec_.zmm = _mm512_set1_epi8(is_second_or_fourth_check);
    }

    inline void reload_classes(u8_t const *byte_to_class) noexcept {
        byte_to_class_vecs_[0].zmm = _mm512_loadu_si512(byte_to_class + 64 * 0);
        byte_to_class_vecs_[1].zmm = _mm512_loadu_si512(byte_to_class + 64 * 1);
        byte_to_class_vecs_[2].zmm = _mm512_loadu_si512(byte_to_class + 64 * 2);
        byte_to_class_vecs_[3].zmm = _mm512_loadu_si512(byte_to_class + 64 * 3);
    }

    /**
     *  @brief Folds the (32 x 32) class cost matrix into 16 resident 64-byte windows for `lookup64`.
     *  @param transpose When set, the matrix is loaded as @b transposed, so that `lookup64(first, second)`
     *         still returns the original `class_substitution_costs[first][second]` even when the diagonal walker
     *         has swapped the shorter and longer strings (which flips the order of the two class operands).
     */
    inline void reload_costs(
        error_cost_t const (&class_substitution_costs)[error_costs_classes_count_k][error_costs_classes_count_k],
        bool transpose) noexcept {
        alignas(64) error_cost_t windows[16 * 64];
        for (size_t window = 0; window != 16; ++window)
            for (size_t low_bit = 0; low_bit != 2; ++low_bit) {
                size_t const first_class = window * 2 + low_bit;
                for (size_t second_class = 0; second_class != error_costs_classes_count_k; ++second_class)
                    windows[window * 64 + low_bit * 32 + second_class] =
                        transpose ? class_substitution_costs[second_class][first_class]
                                  : class_substitution_costs[first_class][second_class];
            }
        for (size_t window = 0; window != 16; ++window)
            cost_windows_vecs_[window].zmm = _mm512_load_si512(windows + window * 64);
    }

    inline u512_vec_t classify64(u512_vec_t const &text_vec) const noexcept {

        u512_vec_t shuffled_class_vecs[4];
        u512_vec_t class_vec;
        __mmask64 is_third_or_fourth, is_second_or_fourth;

        // Map each input byte to its class using the 256-entry `byte_to_class` table.
        // Only the bottom 6 bits of a byte are used in `VPERMB`, so we don't even need to mask.
        shuffled_class_vecs[0].zmm = _mm512_permutexvar_epi8(text_vec.zmm, byte_to_class_vecs_[0].zmm);
        shuffled_class_vecs[1].zmm = _mm512_permutexvar_epi8(text_vec.zmm, byte_to_class_vecs_[1].zmm);
        shuffled_class_vecs[2].zmm = _mm512_permutexvar_epi8(text_vec.zmm, byte_to_class_vecs_[2].zmm);
        shuffled_class_vecs[3].zmm = _mm512_permutexvar_epi8(text_vec.zmm, byte_to_class_vecs_[3].zmm);

        is_third_or_fourth = _mm512_test_epi8_mask(text_vec.zmm, is_third_or_fourth_vec_.zmm);
        is_second_or_fourth = _mm512_test_epi8_mask(text_vec.zmm, is_second_or_fourth_vec_.zmm);
        class_vec.zmm = _mm512_mask_blend_epi8(
            is_third_or_fourth,
            _mm512_mask_blend_epi8(is_second_or_fourth, shuffled_class_vecs[0].zmm, shuffled_class_vecs[1].zmm),
            _mm512_mask_blend_epi8(is_second_or_fourth, shuffled_class_vecs[2].zmm, shuffled_class_vecs[3].zmm));
        return class_vec;
    }

    inline u512_vec_t lookup64(u512_vec_t const &first_class_vec, u512_vec_t const &second_class_vec) const noexcept {

        u512_vec_t index_vec, substituted_vec;
        u512_vec_t permuted_vecs[16], blend4_vecs[8], blend3_vecs[4], blend2_vecs[2];
        u512_vec_t window_vec;

        // The permute index inside every window is `((first_class & 1) << 5) | second_class`.
        index_vec.zmm = _mm512_or_si512(                                                                   //
            _mm512_and_si512(_mm512_slli_epi16(_mm512_and_si512(first_class_vec.zmm, _mm512_set1_epi8(1)), //
                                               5),                                                         //
                             _mm512_set1_epi8((char)0x20)),                                                //
            second_class_vec.zmm);
        for (size_t window = 0; window != 16; ++window)
            permuted_vecs[window].zmm = _mm512_permutexvar_epi8(index_vec.zmm, cost_windows_vecs_[window].zmm);

        // Select the correct window per lane via a balanced tree of mask-blends over `window = first_class >> 1`.
        window_vec.zmm = _mm512_and_si512(_mm512_srli_epi16(first_class_vec.zmm, 1), _mm512_set1_epi8(15));
        __mmask64 const window_bit0 = _mm512_test_epi8_mask(window_vec.zmm, _mm512_set1_epi8(1));
        for (size_t pair = 0; pair != 8; ++pair)
            blend4_vecs[pair].zmm = _mm512_mask_blend_epi8(window_bit0, permuted_vecs[2 * pair].zmm,
                                                           permuted_vecs[2 * pair + 1].zmm);
        __mmask64 const window_bit1 = _mm512_test_epi8_mask(window_vec.zmm, _mm512_set1_epi8(2));
        for (size_t pair = 0; pair != 4; ++pair)
            blend3_vecs[pair].zmm = _mm512_mask_blend_epi8(window_bit1, blend4_vecs[2 * pair].zmm,
                                                           blend4_vecs[2 * pair + 1].zmm);
        __mmask64 const window_bit2 = _mm512_test_epi8_mask(window_vec.zmm, _mm512_set1_epi8(4));
        for (size_t pair = 0; pair != 2; ++pair)
            blend2_vecs[pair].zmm = _mm512_mask_blend_epi8(window_bit2, blend3_vecs[2 * pair].zmm,
                                                           blend3_vecs[2 * pair + 1].zmm);
        __mmask64 const window_bit3 = _mm512_test_epi8_mask(window_vec.zmm, _mm512_set1_epi8(8));
        substituted_vec.zmm = _mm512_mask_blend_epi8(window_bit3, blend2_vecs[0].zmm, blend2_vecs[1].zmm);
        return substituted_vec;
    }
};

/**
 *  @brief Variant of `tile_scorer` - maximizes the @b global Needleman-Wunsch score with class-based
 *         substitution costs, for inputs whose diagonal exceeds the tiny horizontal threshold.
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 *
 *  Mirrors the uniform `u16_t` diagonal scorer (reversed-first / forward-second loads, aligned stores,
 *  head-body-tail split), but replaces the `cmpeq(first, second) -> select(match, mismatch)` substitution
 *  term with `sign_extend_i8_i16(lookup(first_class, second_class))`, evaluated over @b pre-classified
 *  class-index buffers and the resident (32 x 32) cost table built by `prepare`. The objective is maximization,
 *  so the recurrence uses `max`/`add` with a negative gap, exactly like the horizontal class scorer above.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 64;

    substitution_lookup_icelake_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_aligned64chars(                                         //
        u8_t const *first_reversed_slice, u8_t const *second_slice,              //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t const *scores_pre_deletion, i16_t *scores_new,                     //
        u512_vec_t gap_cost_vec) const noexcept {

        u512_vec_t first_vec, second_vec;
        u512_vec_t cost_of_substitution_i8_vec, cost_of_substitution_i16_vecs[2];
        u512_vec_t pre_substitution_vecs[2], pre_insert_vecs[2], pre_delete_vecs[2];
        u512_vec_t cost_if_substitution_vecs[2], cost_if_gap_vecs[2], cell_score_vecs[2];

        // ? Note that here we are still traversing both buffers in the same order,
        // ? because one of the strings has been reversed beforehand.
        first_vec.zmm = _mm512_loadu_epi8(first_reversed_slice);
        second_vec.zmm = _mm512_loadu_epi8(second_slice);
        pre_substitution_vecs[0].zmm = _mm512_loadu_epi16(scores_pre_substitution + 0);
        pre_substitution_vecs[1].zmm = _mm512_loadu_epi16(scores_pre_substitution + 32);
        pre_insert_vecs[0].zmm = _mm512_loadu_epi16(scores_pre_insertion + 0);
        pre_insert_vecs[1].zmm = _mm512_loadu_epi16(scores_pre_insertion + 32);
        pre_delete_vecs[0].zmm = _mm512_loadu_epi16(scores_pre_deletion + 0);
        pre_delete_vecs[1].zmm = _mm512_loadu_epi16(scores_pre_deletion + 32);

        // First, sign-extend the 64 class-pair substitution costs into two `i16` halves.
        cost_of_substitution_i8_vec = lookup_.lookup64(first_vec, second_vec);
        cost_of_substitution_i16_vecs[0].zmm = _mm512_cvtepi8_epi16(
            _mm512_extracti64x4_epi64(cost_of_substitution_i8_vec.zmm, 0));
        cost_of_substitution_i16_vecs[1].zmm = _mm512_cvtepi8_epi16(
            _mm512_extracti64x4_epi64(cost_of_substitution_i8_vec.zmm, 1));

        cost_if_substitution_vecs[0].zmm = _mm512_add_epi16(pre_substitution_vecs[0].zmm,
                                                            cost_of_substitution_i16_vecs[0].zmm);
        cost_if_substitution_vecs[1].zmm = _mm512_add_epi16(pre_substitution_vecs[1].zmm,
                                                            cost_of_substitution_i16_vecs[1].zmm);
        cost_if_gap_vecs[0].zmm = _mm512_add_epi16(_mm512_max_epi16(pre_insert_vecs[0].zmm, pre_delete_vecs[0].zmm),
                                                   gap_cost_vec.zmm);
        cost_if_gap_vecs[1].zmm = _mm512_add_epi16(_mm512_max_epi16(pre_insert_vecs[1].zmm, pre_delete_vecs[1].zmm),
                                                   gap_cost_vec.zmm);
        cell_score_vecs[0].zmm = _mm512_max_epi16(cost_if_substitution_vecs[0].zmm, cost_if_gap_vecs[0].zmm);
        cell_score_vecs[1].zmm = _mm512_max_epi16(cost_if_substitution_vecs[1].zmm, cost_if_gap_vecs[1].zmm);
        _mm512_store_si512(scores_new + 0, cell_score_vecs[0].zmm);
        _mm512_store_si512(scores_new + 32, cell_score_vecs[1].zmm);
    }

    SZ_INLINE void slice_upto64chars(                                            //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t n,    //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t const *scores_pre_deletion, i16_t *scores_new,                     //
        u512_vec_t gap_cost_vec) const noexcept {

        __mmask64 load_mask;
        __mmask32 load_masks[2];
        u512_vec_t first_vec, second_vec;
        u512_vec_t cost_of_substitution_i8_vec, cost_of_substitution_i16_vecs[2];
        u512_vec_t pre_substitution_vecs[2], pre_insert_vecs[2], pre_delete_vecs[2];
        u512_vec_t cost_if_substitution_vecs[2], cost_if_gap_vecs[2], cell_score_vecs[2];

        load_mask = sz_u64_mask_until_(n);
        load_masks[0] = sz_u32_mask_until_(n);
        load_masks[1] = sz_u32_mask_until_(n > 32 ? n - 32 : 0);
        first_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, first_reversed_slice);
        second_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, second_slice);
        pre_substitution_vecs[0].zmm = _mm512_maskz_loadu_epi16(load_masks[0], scores_pre_substitution + 0);
        pre_substitution_vecs[1].zmm = _mm512_maskz_loadu_epi16(load_masks[1], scores_pre_substitution + 32);
        pre_insert_vecs[0].zmm = _mm512_maskz_loadu_epi16(load_masks[0], scores_pre_insertion + 0);
        pre_insert_vecs[1].zmm = _mm512_maskz_loadu_epi16(load_masks[1], scores_pre_insertion + 32);
        pre_delete_vecs[0].zmm = _mm512_maskz_loadu_epi16(load_masks[0], scores_pre_deletion + 0);
        pre_delete_vecs[1].zmm = _mm512_maskz_loadu_epi16(load_masks[1], scores_pre_deletion + 32);

        cost_of_substitution_i8_vec = lookup_.lookup64(first_vec, second_vec);
        cost_of_substitution_i16_vecs[0].zmm = _mm512_cvtepi8_epi16(
            _mm512_extracti64x4_epi64(cost_of_substitution_i8_vec.zmm, 0));
        cost_of_substitution_i16_vecs[1].zmm = _mm512_cvtepi8_epi16(
            _mm512_extracti64x4_epi64(cost_of_substitution_i8_vec.zmm, 1));

        cost_if_substitution_vecs[0].zmm = _mm512_add_epi16(pre_substitution_vecs[0].zmm,
                                                            cost_of_substitution_i16_vecs[0].zmm);
        cost_if_substitution_vecs[1].zmm = _mm512_add_epi16(pre_substitution_vecs[1].zmm,
                                                            cost_of_substitution_i16_vecs[1].zmm);
        cost_if_gap_vecs[0].zmm = _mm512_add_epi16(_mm512_max_epi16(pre_insert_vecs[0].zmm, pre_delete_vecs[0].zmm),
                                                   gap_cost_vec.zmm);
        cost_if_gap_vecs[1].zmm = _mm512_add_epi16(_mm512_max_epi16(pre_insert_vecs[1].zmm, pre_delete_vecs[1].zmm),
                                                   gap_cost_vec.zmm);
        cell_score_vecs[0].zmm = _mm512_max_epi16(cost_if_substitution_vecs[0].zmm, cost_if_gap_vecs[0].zmm);
        cell_score_vecs[1].zmm = _mm512_max_epi16(cost_if_substitution_vecs[1].zmm, cost_if_gap_vecs[1].zmm);
        _mm512_mask_storeu_epi16(scores_new + 0, load_masks[0], cell_score_vecs[0].zmm);
        _mm512_mask_storeu_epi16(scores_new + 32, load_masks[1], cell_score_vecs[1].zmm);
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                    //
        u8_t const *first_reversed_classes, u8_t const *second_classes,          //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t const *scores_pre_deletion, i16_t *scores_new,                     //
        u512_vec_t gap_cost_vec, size_t from, size_t to) noexcept {

        for (size_t page = from; page < to; ++page) {
            size_t const progress = page * step_k;
            slice_aligned64chars(                                                                                 //
                first_reversed_classes + progress, second_classes + progress, scores_pre_substitution + progress, //
                scores_pre_insertion + progress, scores_pre_deletion + progress, scores_new + progress,           //
                gap_cost_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,         //
        i16_t const *scores_pre_deletion, i16_t *scores_new, executor_type_ &&executor = {}) noexcept {

        // ! Both slices already carry @b class bytes, pre-classified once by the diagonal walker.
        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;

        u512_vec_t gap_cost_vec;
        gap_cost_vec.zmm = _mm512_set1_epi16(this->gap_costs_.open_or_extend);

        // On very small inputs, avoid the headache of splitting the input into chunks:
        if (length <= step_k) {
            slice_upto64chars(                                                                  //
                first_reversed_classes, second_classes, length,                                 //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                gap_cost_vec);
            this->last_score_ = scores_new[0];
            return;
        }

        // First handle the misaligned slice of the output buffer:
        head_body_tail_t hbt = head_body_tail<step_k>(scores_new, length);

        // Misaligned head and tail:
        if (hbt.head)
            slice_upto64chars(                                                                  //
                first_reversed_classes, second_classes, hbt.head,                               //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                gap_cost_vec);
        first_reversed_classes += hbt.head, second_classes += hbt.head, scores_pre_substitution += hbt.head,
            scores_pre_insertion += hbt.head, scores_pre_deletion += hbt.head, scores_new += hbt.head;
        if (hbt.tail)
            slice_upto64chars(                                                          //
                first_reversed_classes + hbt.body, second_classes + hbt.body, hbt.tail, //
                scores_pre_substitution + hbt.body, scores_pre_insertion + hbt.body,    //
                scores_pre_deletion + hbt.body, scores_new + hbt.body,                  //
                gap_cost_vec);

        size_t const body_pages = hbt.body / step_k;
        executor.for_slices(body_pages, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_new, gap_cost_vec, from, to);
        });

        if (length == 1) this->last_score_ = scores_new[0];
    }
};

/**
 *  @brief Variant of `tile_scorer` - maximizes the @b local Smith-Waterman score with class-based
 *         substitution costs, for inputs whose diagonal exceeds the tiny horizontal threshold.
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 *
 *  Identical to the global diagonal scorer above, but adds the local zero-clamp on every cell and tracks
 *  the running best across the whole matrix, mirroring the uniform local diagonal scorer and the horizontal
 *  class scorer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_local_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 64;

    substitution_lookup_icelake_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_aligned64chars(                                         //
        u8_t const *first_reversed_slice, u8_t const *second_slice,              //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t const *scores_pre_deletion, i16_t *scores_new,                     //
        u512_vec_t gap_cost_vec) const noexcept {

        u512_vec_t first_vec, second_vec;
        u512_vec_t cost_of_substitution_i8_vec, cost_of_substitution_i16_vecs[2];
        u512_vec_t pre_substitution_vecs[2], pre_insert_vecs[2], pre_delete_vecs[2];
        u512_vec_t cost_if_substitution_vecs[2], cost_if_gap_vecs[2], cell_score_vecs[2];

        first_vec.zmm = _mm512_loadu_epi8(first_reversed_slice);
        second_vec.zmm = _mm512_loadu_epi8(second_slice);
        pre_substitution_vecs[0].zmm = _mm512_loadu_epi16(scores_pre_substitution + 0);
        pre_substitution_vecs[1].zmm = _mm512_loadu_epi16(scores_pre_substitution + 32);
        pre_insert_vecs[0].zmm = _mm512_loadu_epi16(scores_pre_insertion + 0);
        pre_insert_vecs[1].zmm = _mm512_loadu_epi16(scores_pre_insertion + 32);
        pre_delete_vecs[0].zmm = _mm512_loadu_epi16(scores_pre_deletion + 0);
        pre_delete_vecs[1].zmm = _mm512_loadu_epi16(scores_pre_deletion + 32);

        cost_of_substitution_i8_vec = lookup_.lookup64(first_vec, second_vec);
        cost_of_substitution_i16_vecs[0].zmm = _mm512_cvtepi8_epi16(
            _mm512_extracti64x4_epi64(cost_of_substitution_i8_vec.zmm, 0));
        cost_of_substitution_i16_vecs[1].zmm = _mm512_cvtepi8_epi16(
            _mm512_extracti64x4_epi64(cost_of_substitution_i8_vec.zmm, 1));

        cost_if_substitution_vecs[0].zmm = _mm512_add_epi16(pre_substitution_vecs[0].zmm,
                                                            cost_of_substitution_i16_vecs[0].zmm);
        cost_if_substitution_vecs[1].zmm = _mm512_add_epi16(pre_substitution_vecs[1].zmm,
                                                            cost_of_substitution_i16_vecs[1].zmm);
        cost_if_gap_vecs[0].zmm = _mm512_add_epi16(_mm512_max_epi16(pre_insert_vecs[0].zmm, pre_delete_vecs[0].zmm),
                                                   gap_cost_vec.zmm);
        cost_if_gap_vecs[1].zmm = _mm512_add_epi16(_mm512_max_epi16(pre_insert_vecs[1].zmm, pre_delete_vecs[1].zmm),
                                                   gap_cost_vec.zmm);
        cell_score_vecs[0].zmm = _mm512_max_epi16(cost_if_substitution_vecs[0].zmm, cost_if_gap_vecs[0].zmm);
        cell_score_vecs[1].zmm = _mm512_max_epi16(cost_if_substitution_vecs[1].zmm, cost_if_gap_vecs[1].zmm);

        // In Local Alignment for SW we also need to compare to zero and set the result to zero if negative.
        cell_score_vecs[0].zmm = _mm512_max_epi16(cell_score_vecs[0].zmm, _mm512_setzero_epi32());
        cell_score_vecs[1].zmm = _mm512_max_epi16(cell_score_vecs[1].zmm, _mm512_setzero_epi32());
        _mm512_store_si512(scores_new + 0, cell_score_vecs[0].zmm);
        _mm512_store_si512(scores_new + 32, cell_score_vecs[1].zmm);
    }

    SZ_INLINE void slice_upto64chars(                                            //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t n,    //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t const *scores_pre_deletion, i16_t *scores_new,                     //
        u512_vec_t gap_cost_vec) const noexcept {

        __mmask64 load_mask;
        __mmask32 load_masks[2];
        u512_vec_t first_vec, second_vec;
        u512_vec_t cost_of_substitution_i8_vec, cost_of_substitution_i16_vecs[2];
        u512_vec_t pre_substitution_vecs[2], pre_insert_vecs[2], pre_delete_vecs[2];
        u512_vec_t cost_if_substitution_vecs[2], cost_if_gap_vecs[2], cell_score_vecs[2];

        load_mask = sz_u64_mask_until_(n);
        load_masks[0] = sz_u32_mask_until_(n);
        load_masks[1] = sz_u32_mask_until_(n > 32 ? n - 32 : 0);
        first_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, first_reversed_slice);
        second_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, second_slice);
        pre_substitution_vecs[0].zmm = _mm512_maskz_loadu_epi16(load_masks[0], scores_pre_substitution + 0);
        pre_substitution_vecs[1].zmm = _mm512_maskz_loadu_epi16(load_masks[1], scores_pre_substitution + 32);
        pre_insert_vecs[0].zmm = _mm512_maskz_loadu_epi16(load_masks[0], scores_pre_insertion + 0);
        pre_insert_vecs[1].zmm = _mm512_maskz_loadu_epi16(load_masks[1], scores_pre_insertion + 32);
        pre_delete_vecs[0].zmm = _mm512_maskz_loadu_epi16(load_masks[0], scores_pre_deletion + 0);
        pre_delete_vecs[1].zmm = _mm512_maskz_loadu_epi16(load_masks[1], scores_pre_deletion + 32);

        cost_of_substitution_i8_vec = lookup_.lookup64(first_vec, second_vec);
        cost_of_substitution_i16_vecs[0].zmm = _mm512_cvtepi8_epi16(
            _mm512_extracti64x4_epi64(cost_of_substitution_i8_vec.zmm, 0));
        cost_of_substitution_i16_vecs[1].zmm = _mm512_cvtepi8_epi16(
            _mm512_extracti64x4_epi64(cost_of_substitution_i8_vec.zmm, 1));

        cost_if_substitution_vecs[0].zmm = _mm512_add_epi16(pre_substitution_vecs[0].zmm,
                                                            cost_of_substitution_i16_vecs[0].zmm);
        cost_if_substitution_vecs[1].zmm = _mm512_add_epi16(pre_substitution_vecs[1].zmm,
                                                            cost_of_substitution_i16_vecs[1].zmm);
        cost_if_gap_vecs[0].zmm = _mm512_add_epi16(_mm512_max_epi16(pre_insert_vecs[0].zmm, pre_delete_vecs[0].zmm),
                                                   gap_cost_vec.zmm);
        cost_if_gap_vecs[1].zmm = _mm512_add_epi16(_mm512_max_epi16(pre_insert_vecs[1].zmm, pre_delete_vecs[1].zmm),
                                                   gap_cost_vec.zmm);
        cell_score_vecs[0].zmm = _mm512_max_epi16(cost_if_substitution_vecs[0].zmm, cost_if_gap_vecs[0].zmm);
        cell_score_vecs[1].zmm = _mm512_max_epi16(cost_if_substitution_vecs[1].zmm, cost_if_gap_vecs[1].zmm);

        // In Local Alignment for SW we also need to compare to zero and set the result to zero if negative.
        cell_score_vecs[0].zmm = _mm512_max_epi16(cell_score_vecs[0].zmm, _mm512_setzero_epi32());
        cell_score_vecs[1].zmm = _mm512_max_epi16(cell_score_vecs[1].zmm, _mm512_setzero_epi32());
        _mm512_mask_storeu_epi16(scores_new + 0, load_masks[0], cell_score_vecs[0].zmm);
        _mm512_mask_storeu_epi16(scores_new + 32, load_masks[1], cell_score_vecs[1].zmm);
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                    //
        u8_t const *first_reversed_classes, u8_t const *second_classes,          //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion, //
        i16_t const *scores_pre_deletion, i16_t *scores_new,                     //
        u512_vec_t gap_cost_vec, size_t from, size_t to) noexcept {

        for (size_t page = from; page < to; ++page) {
            size_t const progress = page * step_k;
            slice_aligned64chars(                                                                                 //
                first_reversed_classes + progress, second_classes + progress, scores_pre_substitution + progress, //
                scores_pre_insertion + progress, scores_pre_deletion + progress, scores_new + progress,           //
                gap_cost_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,         //
        i16_t const *scores_pre_deletion, i16_t *scores_new, executor_type_ &&executor = {}) noexcept {

        // ! Both slices already carry @b class bytes, pre-classified once by the diagonal walker.
        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;
        i16_t *const scores_new_begin = scores_new;

        u512_vec_t gap_cost_vec;
        gap_cost_vec.zmm = _mm512_set1_epi16(this->gap_costs_.open_or_extend);

        if (length <= step_k) {
            slice_upto64chars(                                                                  //
                first_reversed_classes, second_classes, length,                                 //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                gap_cost_vec);
        }
        else {
            head_body_tail_t hbt = head_body_tail<step_k>(scores_new, length);
            if (hbt.head)
                slice_upto64chars(                                                                  //
                    first_reversed_classes, second_classes, hbt.head,                               //
                    scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                    gap_cost_vec);
            first_reversed_classes += hbt.head, second_classes += hbt.head, scores_pre_substitution += hbt.head,
                scores_pre_insertion += hbt.head, scores_pre_deletion += hbt.head, scores_new += hbt.head;
            if (hbt.tail)
                slice_upto64chars(                                                          //
                    first_reversed_classes + hbt.body, second_classes + hbt.body, hbt.tail, //
                    scores_pre_substitution + hbt.body, scores_pre_insertion + hbt.body,    //
                    scores_pre_deletion + hbt.body, scores_new + hbt.body,                  //
                    gap_cost_vec);
            size_t const body_pages = hbt.body / step_k;
            executor.for_slices(body_pages, [&](size_t from, size_t to) noexcept {
                score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                        scores_pre_insertion, scores_pre_deletion, scores_new, gap_cost_vec, from, to);
            });
        }

        // The running best across the whole matrix is the reported local-alignment score.
        i16_t best_in_diagonal = this->best_score_;
        for (size_t i = 0; i != length; ++i) best_in_diagonal = sz_max_of_two(best_in_diagonal, scores_new_begin[i]);
        this->best_score_ = best_in_diagonal;
    }
};

/**
 *  @brief Variant of the global diagonal class scorer over @b `i32_t` cells, for inputs whose scores
 *         exceed the 16-bit range. Mirrors the `i16_t` scorer but folds the 64 class-pair costs into
 *         four 16-lane `i32` halves via `_mm512_cvtepi8_epi32`.
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 64;

    substitution_lookup_icelake_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_aligned64chars(                                         //
        u8_t const *first_reversed_slice, u8_t const *second_slice,              //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t const *scores_pre_deletion, i32_t *scores_new,                     //
        u512_vec_t gap_cost_vec) const noexcept {

        u512_vec_t first_vec, second_vec, cost_of_substitution_i8_vec;
        u512_vec_t cost_of_substitution_i32_vecs[4];
        u512_vec_t pre_substitution_vecs[4], pre_insert_vecs[4], pre_delete_vecs[4];
        u512_vec_t cost_if_substitution_vecs[4], cost_if_gap_vecs[4], cell_score_vecs[4];

        first_vec.zmm = _mm512_loadu_epi8(first_reversed_slice);
        second_vec.zmm = _mm512_loadu_epi8(second_slice);
        for (size_t part = 0; part != 4; ++part) {
            pre_substitution_vecs[part].zmm = _mm512_loadu_epi32(scores_pre_substitution + part * 16);
            pre_insert_vecs[part].zmm = _mm512_loadu_epi32(scores_pre_insertion + part * 16);
            pre_delete_vecs[part].zmm = _mm512_loadu_epi32(scores_pre_deletion + part * 16);
        }

        cost_of_substitution_i8_vec = lookup_.lookup64(first_vec, second_vec);
        cost_of_substitution_i32_vecs[0].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 0));
        cost_of_substitution_i32_vecs[1].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 1));
        cost_of_substitution_i32_vecs[2].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 2));
        cost_of_substitution_i32_vecs[3].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 3));

        for (size_t part = 0; part != 4; ++part) {
            cost_if_substitution_vecs[part].zmm = _mm512_add_epi32(pre_substitution_vecs[part].zmm,
                                                                   cost_of_substitution_i32_vecs[part].zmm);
            cost_if_gap_vecs[part].zmm = _mm512_add_epi32(
                _mm512_max_epi32(pre_insert_vecs[part].zmm, pre_delete_vecs[part].zmm), gap_cost_vec.zmm);
            cell_score_vecs[part].zmm = _mm512_max_epi32(cost_if_substitution_vecs[part].zmm,
                                                         cost_if_gap_vecs[part].zmm);
            _mm512_store_si512(scores_new + part * 16, cell_score_vecs[part].zmm);
        }
    }

    SZ_INLINE void slice_upto64chars(                                            //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t n,    //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t const *scores_pre_deletion, i32_t *scores_new,                     //
        u512_vec_t gap_cost_vec) const noexcept {

        u512_vec_t first_vec, second_vec, cost_of_substitution_i8_vec;
        u512_vec_t cost_of_substitution_i32_vecs[4];
        u512_vec_t pre_substitution_vecs[4], pre_insert_vecs[4], pre_delete_vecs[4];
        u512_vec_t cost_if_substitution_vecs[4], cost_if_gap_vecs[4], cell_score_vecs[4];

        // The four 16-lane score sub-masks are just consecutive slices of the single 64-lane byte mask.
        __mmask64 const load_mask = sz_u64_mask_until_(n);
        __mmask16 const load_masks[4] = {(__mmask16)load_mask, (__mmask16)(load_mask >> 16),
                                         (__mmask16)(load_mask >> 32), (__mmask16)(load_mask >> 48)};
        first_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, first_reversed_slice);
        second_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, second_slice);
        for (size_t part = 0; part != 4; ++part) {
            pre_substitution_vecs[part].zmm = _mm512_maskz_loadu_epi32(load_masks[part],
                                                                       scores_pre_substitution + part * 16);
            pre_insert_vecs[part].zmm = _mm512_maskz_loadu_epi32(load_masks[part], scores_pre_insertion + part * 16);
            pre_delete_vecs[part].zmm = _mm512_maskz_loadu_epi32(load_masks[part], scores_pre_deletion + part * 16);
        }

        cost_of_substitution_i8_vec = lookup_.lookup64(first_vec, second_vec);
        cost_of_substitution_i32_vecs[0].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 0));
        cost_of_substitution_i32_vecs[1].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 1));
        cost_of_substitution_i32_vecs[2].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 2));
        cost_of_substitution_i32_vecs[3].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 3));

        for (size_t part = 0; part != 4; ++part) {
            cost_if_substitution_vecs[part].zmm = _mm512_add_epi32(pre_substitution_vecs[part].zmm,
                                                                   cost_of_substitution_i32_vecs[part].zmm);
            cost_if_gap_vecs[part].zmm = _mm512_add_epi32(
                _mm512_max_epi32(pre_insert_vecs[part].zmm, pre_delete_vecs[part].zmm), gap_cost_vec.zmm);
            cell_score_vecs[part].zmm = _mm512_max_epi32(cost_if_substitution_vecs[part].zmm,
                                                         cost_if_gap_vecs[part].zmm);
            _mm512_mask_storeu_epi32(scores_new + part * 16, load_masks[part], cell_score_vecs[part].zmm);
        }
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                    //
        u8_t const *first_reversed_classes, u8_t const *second_classes,          //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t const *scores_pre_deletion, i32_t *scores_new,                     //
        u512_vec_t gap_cost_vec, size_t from, size_t to) noexcept {

        for (size_t page = from; page < to; ++page) {
            size_t const progress = page * step_k;
            slice_aligned64chars(                                                                                 //
                first_reversed_classes + progress, second_classes + progress, scores_pre_substitution + progress, //
                scores_pre_insertion + progress, scores_pre_deletion + progress, scores_new + progress,           //
                gap_cost_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,         //
        i32_t const *scores_pre_deletion, i32_t *scores_new, executor_type_ &&executor = {}) noexcept {

        // ! Both slices already carry @b class bytes, pre-classified once by the diagonal walker.
        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;

        u512_vec_t gap_cost_vec;
        gap_cost_vec.zmm = _mm512_set1_epi32(this->gap_costs_.open_or_extend);

        if (length <= step_k) {
            slice_upto64chars(                                                                  //
                first_reversed_classes, second_classes, length,                                 //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                gap_cost_vec);
            this->last_score_ = scores_new[0];
            return;
        }

        head_body_tail_t hbt = head_body_tail<step_k>(scores_new, length);
        if (hbt.head)
            slice_upto64chars(                                                                  //
                first_reversed_classes, second_classes, hbt.head,                               //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                gap_cost_vec);
        first_reversed_classes += hbt.head, second_classes += hbt.head, scores_pre_substitution += hbt.head,
            scores_pre_insertion += hbt.head, scores_pre_deletion += hbt.head, scores_new += hbt.head;
        if (hbt.tail)
            slice_upto64chars(                                                          //
                first_reversed_classes + hbt.body, second_classes + hbt.body, hbt.tail, //
                scores_pre_substitution + hbt.body, scores_pre_insertion + hbt.body,    //
                scores_pre_deletion + hbt.body, scores_new + hbt.body,                  //
                gap_cost_vec);

        size_t const body_pages = hbt.body / step_k;
        executor.for_slices(body_pages, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_new, gap_cost_vec, from, to);
        });

        if (length == 1) this->last_score_ = scores_new[0];
    }
};

/**
 *  @brief Variant of the local diagonal class scorer over @b `i32_t` cells. Mirrors the `i32_t` global
 *         scorer, plus the per-cell zero-clamp and the running-best reduction of Smith-Waterman.
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, linear_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_local_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 64;

    substitution_lookup_icelake_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_aligned64chars(                                         //
        u8_t const *first_reversed_slice, u8_t const *second_slice,              //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t const *scores_pre_deletion, i32_t *scores_new,                     //
        u512_vec_t gap_cost_vec) const noexcept {

        u512_vec_t first_vec, second_vec, cost_of_substitution_i8_vec;
        u512_vec_t cost_of_substitution_i32_vecs[4];
        u512_vec_t pre_substitution_vecs[4], pre_insert_vecs[4], pre_delete_vecs[4];
        u512_vec_t cost_if_substitution_vecs[4], cost_if_gap_vecs[4], cell_score_vecs[4];

        first_vec.zmm = _mm512_loadu_epi8(first_reversed_slice);
        second_vec.zmm = _mm512_loadu_epi8(second_slice);
        for (size_t part = 0; part != 4; ++part) {
            pre_substitution_vecs[part].zmm = _mm512_loadu_epi32(scores_pre_substitution + part * 16);
            pre_insert_vecs[part].zmm = _mm512_loadu_epi32(scores_pre_insertion + part * 16);
            pre_delete_vecs[part].zmm = _mm512_loadu_epi32(scores_pre_deletion + part * 16);
        }

        cost_of_substitution_i8_vec = lookup_.lookup64(first_vec, second_vec);
        cost_of_substitution_i32_vecs[0].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 0));
        cost_of_substitution_i32_vecs[1].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 1));
        cost_of_substitution_i32_vecs[2].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 2));
        cost_of_substitution_i32_vecs[3].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 3));

        for (size_t part = 0; part != 4; ++part) {
            cost_if_substitution_vecs[part].zmm = _mm512_add_epi32(pre_substitution_vecs[part].zmm,
                                                                   cost_of_substitution_i32_vecs[part].zmm);
            cost_if_gap_vecs[part].zmm = _mm512_add_epi32(
                _mm512_max_epi32(pre_insert_vecs[part].zmm, pre_delete_vecs[part].zmm), gap_cost_vec.zmm);
            cell_score_vecs[part].zmm = _mm512_max_epi32(cost_if_substitution_vecs[part].zmm,
                                                         cost_if_gap_vecs[part].zmm);
            cell_score_vecs[part].zmm = _mm512_max_epi32(cell_score_vecs[part].zmm, _mm512_setzero_epi32());
            _mm512_store_si512(scores_new + part * 16, cell_score_vecs[part].zmm);
        }
    }

    SZ_INLINE void slice_upto64chars(                                            //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t n,    //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t const *scores_pre_deletion, i32_t *scores_new,                     //
        u512_vec_t gap_cost_vec) const noexcept {

        u512_vec_t first_vec, second_vec, cost_of_substitution_i8_vec;
        u512_vec_t cost_of_substitution_i32_vecs[4];
        u512_vec_t pre_substitution_vecs[4], pre_insert_vecs[4], pre_delete_vecs[4];
        u512_vec_t cost_if_substitution_vecs[4], cost_if_gap_vecs[4], cell_score_vecs[4];

        // The four 16-lane score sub-masks are just consecutive slices of the single 64-lane byte mask.
        __mmask64 const load_mask = sz_u64_mask_until_(n);
        __mmask16 const load_masks[4] = {(__mmask16)load_mask, (__mmask16)(load_mask >> 16),
                                         (__mmask16)(load_mask >> 32), (__mmask16)(load_mask >> 48)};
        first_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, first_reversed_slice);
        second_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, second_slice);
        for (size_t part = 0; part != 4; ++part) {
            pre_substitution_vecs[part].zmm = _mm512_maskz_loadu_epi32(load_masks[part],
                                                                       scores_pre_substitution + part * 16);
            pre_insert_vecs[part].zmm = _mm512_maskz_loadu_epi32(load_masks[part], scores_pre_insertion + part * 16);
            pre_delete_vecs[part].zmm = _mm512_maskz_loadu_epi32(load_masks[part], scores_pre_deletion + part * 16);
        }

        cost_of_substitution_i8_vec = lookup_.lookup64(first_vec, second_vec);
        cost_of_substitution_i32_vecs[0].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 0));
        cost_of_substitution_i32_vecs[1].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 1));
        cost_of_substitution_i32_vecs[2].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 2));
        cost_of_substitution_i32_vecs[3].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 3));

        for (size_t part = 0; part != 4; ++part) {
            cost_if_substitution_vecs[part].zmm = _mm512_add_epi32(pre_substitution_vecs[part].zmm,
                                                                   cost_of_substitution_i32_vecs[part].zmm);
            cost_if_gap_vecs[part].zmm = _mm512_add_epi32(
                _mm512_max_epi32(pre_insert_vecs[part].zmm, pre_delete_vecs[part].zmm), gap_cost_vec.zmm);
            cell_score_vecs[part].zmm = _mm512_max_epi32(cost_if_substitution_vecs[part].zmm,
                                                         cost_if_gap_vecs[part].zmm);
            cell_score_vecs[part].zmm = _mm512_max_epi32(cell_score_vecs[part].zmm, _mm512_setzero_epi32());
            _mm512_mask_storeu_epi32(scores_new + part * 16, load_masks[part], cell_score_vecs[part].zmm);
        }
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                    //
        u8_t const *first_reversed_classes, u8_t const *second_classes,          //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion, //
        i32_t const *scores_pre_deletion, i32_t *scores_new,                     //
        u512_vec_t gap_cost_vec, size_t from, size_t to) noexcept {

        for (size_t page = from; page < to; ++page) {
            size_t const progress = page * step_k;
            slice_aligned64chars(                                                                                 //
                first_reversed_classes + progress, second_classes + progress, scores_pre_substitution + progress, //
                scores_pre_insertion + progress, scores_pre_deletion + progress, scores_new + progress,           //
                gap_cost_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,         //
        i32_t const *scores_pre_deletion, i32_t *scores_new, executor_type_ &&executor = {}) noexcept {

        // ! Both slices already carry @b class bytes, pre-classified once by the diagonal walker.
        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;
        i32_t *const scores_new_begin = scores_new;

        u512_vec_t gap_cost_vec;
        gap_cost_vec.zmm = _mm512_set1_epi32(this->gap_costs_.open_or_extend);

        if (length <= step_k) {
            slice_upto64chars(                                                                  //
                first_reversed_classes, second_classes, length,                                 //
                scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                gap_cost_vec);
        }
        else {
            head_body_tail_t hbt = head_body_tail<step_k>(scores_new, length);
            if (hbt.head)
                slice_upto64chars(                                                                  //
                    first_reversed_classes, second_classes, hbt.head,                               //
                    scores_pre_substitution, scores_pre_insertion, scores_pre_deletion, scores_new, //
                    gap_cost_vec);
            first_reversed_classes += hbt.head, second_classes += hbt.head, scores_pre_substitution += hbt.head,
                scores_pre_insertion += hbt.head, scores_pre_deletion += hbt.head, scores_new += hbt.head;
            if (hbt.tail)
                slice_upto64chars(                                                          //
                    first_reversed_classes + hbt.body, second_classes + hbt.body, hbt.tail, //
                    scores_pre_substitution + hbt.body, scores_pre_insertion + hbt.body,    //
                    scores_pre_deletion + hbt.body, scores_new + hbt.body,                  //
                    gap_cost_vec);
            size_t const body_pages = hbt.body / step_k;
            executor.for_slices(body_pages, [&](size_t from, size_t to) noexcept {
                score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                        scores_pre_insertion, scores_pre_deletion, scores_new, gap_cost_vec, from, to);
            });
        }

        // The running best across the whole matrix is the reported local-alignment score.
        i32_t best_in_diagonal = this->best_score_;
        for (size_t i = 0; i != length; ++i) best_in_diagonal = sz_max_of_two(best_in_diagonal, scores_new_begin[i]);
        this->best_score_ = best_in_diagonal;
    }
};

/**
 *  @brief Ice Lake @b affine-gap diagonal class scorer - maximizes the global Needleman-Wunsch score over
 *         `i16_t` cells. Mirrors the linear class scorer, but threads the separate insertion and deletion
 *         gap diagonals of the Gotoh recurrence (open vs extend) alongside the main score diagonal.
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 64;

    substitution_lookup_icelake_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_upto64chars(                                             //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t n,     //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,  //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions, //
        i16_t const *scores_running_deletions, i16_t *scores_new,                 //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                //
        u512_vec_t gap_open_vec, u512_vec_t gap_expand_vec) const noexcept {

        u512_vec_t first_vec, second_vec, cost_of_substitution_i8_vec, cost_of_substitution_i16_vecs[2];

        // The two 32-lane score sub-masks are consecutive slices of the single 64-lane byte mask.
        __mmask64 const load_mask = sz_u64_mask_until_(n);
        __mmask32 const load_masks[2] = {(__mmask32)load_mask, (__mmask32)(load_mask >> 32)};
        first_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, first_reversed_slice);
        second_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, second_slice);
        cost_of_substitution_i8_vec = lookup_.lookup64(first_vec, second_vec);
        cost_of_substitution_i16_vecs[0].zmm = _mm512_cvtepi8_epi16(
            _mm512_extracti64x4_epi64(cost_of_substitution_i8_vec.zmm, 0));
        cost_of_substitution_i16_vecs[1].zmm = _mm512_cvtepi8_epi16(
            _mm512_extracti64x4_epi64(cost_of_substitution_i8_vec.zmm, 1));

        for (size_t part = 0; part != 2; ++part) {
            __mmask32 const part_mask = load_masks[part];
            size_t const offset = part * 32;
            u512_vec_t pre_substitution, pre_insert_open, pre_delete_open, run_insert, run_delete;
            u512_vec_t cost_if_insert, cost_if_delete, cell_score;
            pre_substitution.zmm = _mm512_maskz_loadu_epi16(part_mask, scores_pre_substitution + offset);
            pre_insert_open.zmm = _mm512_maskz_loadu_epi16(part_mask, scores_pre_insertion + offset);
            pre_delete_open.zmm = _mm512_maskz_loadu_epi16(part_mask, scores_pre_deletion + offset);
            run_insert.zmm = _mm512_maskz_loadu_epi16(part_mask, scores_running_insertions + offset);
            run_delete.zmm = _mm512_maskz_loadu_epi16(part_mask, scores_running_deletions + offset);
            cost_if_insert.zmm = _mm512_max_epi16(_mm512_add_epi16(run_insert.zmm, gap_expand_vec.zmm),
                                                  _mm512_add_epi16(pre_insert_open.zmm, gap_open_vec.zmm));
            cost_if_delete.zmm = _mm512_max_epi16(_mm512_add_epi16(run_delete.zmm, gap_expand_vec.zmm),
                                                  _mm512_add_epi16(pre_delete_open.zmm, gap_open_vec.zmm));
            cell_score.zmm = _mm512_max_epi16(
                _mm512_add_epi16(pre_substitution.zmm, cost_of_substitution_i16_vecs[part].zmm),
                _mm512_max_epi16(cost_if_insert.zmm, cost_if_delete.zmm));
            _mm512_mask_storeu_epi16(scores_new + offset, part_mask, cell_score.zmm);
            _mm512_mask_storeu_epi16(scores_new_insertions + offset, part_mask, cost_if_insert.zmm);
            _mm512_mask_storeu_epi16(scores_new_deletions + offset, part_mask, cost_if_delete.zmm);
        }
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                     //
        u8_t const *first_reversed_classes, u8_t const *second_classes,           //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,  //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions, //
        i16_t const *scores_running_deletions, i16_t *scores_new,                 //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                //
        u512_vec_t gap_open_vec, u512_vec_t gap_expand_vec, size_t from, size_t to) noexcept {

        for (size_t page = from; page < to; ++page) {
            size_t const progress = page * step_k;
            slice_upto64chars(                                                        //
                first_reversed_classes + progress, second_classes + progress, step_k, //
                scores_pre_substitution + progress, scores_pre_insertion + progress,  //
                scores_pre_deletion + progress, scores_running_insertions + progress, //
                scores_running_deletions + progress, scores_new + progress,           //
                scores_new_insertions + progress, scores_new_deletions + progress,    //
                gap_open_vec, gap_expand_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,         //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions,        //
        i16_t const *scores_running_deletions, i16_t *scores_new,                        //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                       //
        executor_type_ &&executor = {}) noexcept {

        // ! Both slices already carry @b class bytes, pre-classified once by the diagonal walker.
        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;

        u512_vec_t gap_open_vec, gap_expand_vec;
        gap_open_vec.zmm = _mm512_set1_epi16(this->gap_costs_.open);
        gap_expand_vec.zmm = _mm512_set1_epi16(this->gap_costs_.extend);

        size_t const body_pages = length / step_k;
        executor.for_slices(body_pages, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_running_insertions,
                                    scores_running_deletions, scores_new, scores_new_insertions, scores_new_deletions,
                                    gap_open_vec, gap_expand_vec, from, to);
        });

        size_t const progress = body_pages * step_k;
        size_t const tail = length - progress;
        if (tail)
            slice_upto64chars(                                                        //
                first_reversed_classes + progress, second_classes + progress, tail,   //
                scores_pre_substitution + progress, scores_pre_insertion + progress,  //
                scores_pre_deletion + progress, scores_running_insertions + progress, //
                scores_running_deletions + progress, scores_new + progress,           //
                scores_new_insertions + progress, scores_new_deletions + progress,    //
                gap_open_vec, gap_expand_vec);

        if (length == 1) this->last_score_ = scores_new[0];
    }
};

/**
 *  @brief Ice Lake @b affine-gap diagonal class scorer - maximizes the local Smith-Waterman score over
 *         `i16_t` cells. Mirrors the global affine class scorer above, but adds the Smith-Waterman-Gotoh
 *         zero-reset on @b only the substitution term and tracks the running best across the whole matrix.
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i16_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_local_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 64;

    substitution_lookup_icelake_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_upto64chars(                                             //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t n,     //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,  //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions, //
        i16_t const *scores_running_deletions, i16_t *scores_new,                 //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                //
        u512_vec_t gap_open_vec, u512_vec_t gap_expand_vec) const noexcept {

        u512_vec_t first_vec, second_vec, cost_of_substitution_i8_vec, cost_of_substitution_i16_vecs[2];

        // The two 32-lane score sub-masks are consecutive slices of the single 64-lane byte mask.
        __mmask64 const load_mask = sz_u64_mask_until_(n);
        __mmask32 const load_masks[2] = {(__mmask32)load_mask, (__mmask32)(load_mask >> 32)};
        first_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, first_reversed_slice);
        second_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, second_slice);
        cost_of_substitution_i8_vec = lookup_.lookup64(first_vec, second_vec);
        cost_of_substitution_i16_vecs[0].zmm = _mm512_cvtepi8_epi16(
            _mm512_extracti64x4_epi64(cost_of_substitution_i8_vec.zmm, 0));
        cost_of_substitution_i16_vecs[1].zmm = _mm512_cvtepi8_epi16(
            _mm512_extracti64x4_epi64(cost_of_substitution_i8_vec.zmm, 1));

        for (size_t part = 0; part != 2; ++part) {
            __mmask32 const part_mask = load_masks[part];
            size_t const offset = part * 32;
            u512_vec_t pre_substitution, pre_insert_open, pre_delete_open, run_insert, run_delete;
            u512_vec_t cost_if_insert, cost_if_delete, cell_score;
            pre_substitution.zmm = _mm512_maskz_loadu_epi16(part_mask, scores_pre_substitution + offset);
            pre_insert_open.zmm = _mm512_maskz_loadu_epi16(part_mask, scores_pre_insertion + offset);
            pre_delete_open.zmm = _mm512_maskz_loadu_epi16(part_mask, scores_pre_deletion + offset);
            run_insert.zmm = _mm512_maskz_loadu_epi16(part_mask, scores_running_insertions + offset);
            run_delete.zmm = _mm512_maskz_loadu_epi16(part_mask, scores_running_deletions + offset);
            cost_if_insert.zmm = _mm512_max_epi16(_mm512_add_epi16(run_insert.zmm, gap_expand_vec.zmm),
                                                  _mm512_add_epi16(pre_insert_open.zmm, gap_open_vec.zmm));
            cost_if_delete.zmm = _mm512_max_epi16(_mm512_add_epi16(run_delete.zmm, gap_expand_vec.zmm),
                                                  _mm512_add_epi16(pre_delete_open.zmm, gap_open_vec.zmm));
            // In Local Alignment for SW the zero-reset is applied to @b only the substitution term;
            // the insertion/deletion gap matrices are not clamped, exactly like the serial scorer.
            cell_score.zmm = _mm512_max_epi16(
                _mm512_max_epi16(_mm512_add_epi16(pre_substitution.zmm, cost_of_substitution_i16_vecs[part].zmm),
                                 _mm512_setzero_epi32()),
                _mm512_max_epi16(cost_if_insert.zmm, cost_if_delete.zmm));
            _mm512_mask_storeu_epi16(scores_new + offset, part_mask, cell_score.zmm);
            _mm512_mask_storeu_epi16(scores_new_insertions + offset, part_mask, cost_if_insert.zmm);
            _mm512_mask_storeu_epi16(scores_new_deletions + offset, part_mask, cost_if_delete.zmm);
        }
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                     //
        u8_t const *first_reversed_classes, u8_t const *second_classes,           //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,  //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions, //
        i16_t const *scores_running_deletions, i16_t *scores_new,                 //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                //
        u512_vec_t gap_open_vec, u512_vec_t gap_expand_vec, size_t from, size_t to) noexcept {

        for (size_t page = from; page < to; ++page) {
            size_t const progress = page * step_k;
            slice_upto64chars(                                                        //
                first_reversed_classes + progress, second_classes + progress, step_k, //
                scores_pre_substitution + progress, scores_pre_insertion + progress,  //
                scores_pre_deletion + progress, scores_running_insertions + progress, //
                scores_running_deletions + progress, scores_new + progress,           //
                scores_new_insertions + progress, scores_new_deletions + progress,    //
                gap_open_vec, gap_expand_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i16_t const *scores_pre_substitution, i16_t const *scores_pre_insertion,         //
        i16_t const *scores_pre_deletion, i16_t const *scores_running_insertions,        //
        i16_t const *scores_running_deletions, i16_t *scores_new,                        //
        i16_t *scores_new_insertions, i16_t *scores_new_deletions,                       //
        executor_type_ &&executor = {}) noexcept {

        // ! Both slices already carry @b class bytes, pre-classified once by the diagonal walker.
        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;
        i16_t *const scores_new_begin = scores_new;

        u512_vec_t gap_open_vec, gap_expand_vec;
        gap_open_vec.zmm = _mm512_set1_epi16(this->gap_costs_.open);
        gap_expand_vec.zmm = _mm512_set1_epi16(this->gap_costs_.extend);

        size_t const body_pages = length / step_k;
        executor.for_slices(body_pages, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_running_insertions,
                                    scores_running_deletions, scores_new, scores_new_insertions, scores_new_deletions,
                                    gap_open_vec, gap_expand_vec, from, to);
        });

        size_t const progress = body_pages * step_k;
        size_t const tail = length - progress;
        if (tail)
            slice_upto64chars(                                                        //
                first_reversed_classes + progress, second_classes + progress, tail,   //
                scores_pre_substitution + progress, scores_pre_insertion + progress,  //
                scores_pre_deletion + progress, scores_running_insertions + progress, //
                scores_running_deletions + progress, scores_new + progress,           //
                scores_new_insertions + progress, scores_new_deletions + progress,    //
                gap_open_vec, gap_expand_vec);

        // The running best across the whole matrix is the reported local-alignment score.
        i16_t best_in_diagonal = this->best_score_;
        for (size_t i = 0; i != length; ++i) best_in_diagonal = sz_max_of_two(best_in_diagonal, scores_new_begin[i]);
        this->best_score_ = best_in_diagonal;
    }
};

/**
 *  @brief Ice Lake @b affine-gap diagonal class scorer - maximizes the global Needleman-Wunsch score over
 *         `i32_t` cells, for inputs whose scores exceed the 16-bit range. Mirrors the `i16_t` affine
 *         scorer but folds the 64 class-pair costs into four 16-lane `i32` halves via `_mm512_cvtepi8_epi32`.
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_global_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_global_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 64;

    substitution_lookup_icelake_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_upto64chars(                                             //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t n,     //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,  //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions, //
        i32_t const *scores_running_deletions, i32_t *scores_new,                 //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                //
        u512_vec_t gap_open_vec, u512_vec_t gap_expand_vec) const noexcept {

        u512_vec_t first_vec, second_vec, cost_of_substitution_i8_vec, cost_of_substitution_i32_vecs[4];

        // The four 16-lane score sub-masks are just consecutive slices of the single 64-lane byte mask.
        __mmask64 const load_mask = sz_u64_mask_until_(n);
        __mmask16 const load_masks[4] = {(__mmask16)load_mask, (__mmask16)(load_mask >> 16),
                                         (__mmask16)(load_mask >> 32), (__mmask16)(load_mask >> 48)};
        first_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, first_reversed_slice);
        second_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, second_slice);
        cost_of_substitution_i8_vec = lookup_.lookup64(first_vec, second_vec);
        cost_of_substitution_i32_vecs[0].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 0));
        cost_of_substitution_i32_vecs[1].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 1));
        cost_of_substitution_i32_vecs[2].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 2));
        cost_of_substitution_i32_vecs[3].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 3));

        for (size_t part = 0; part != 4; ++part) {
            __mmask16 const part_mask = load_masks[part];
            size_t const offset = part * 16;
            u512_vec_t pre_substitution, pre_insert_open, pre_delete_open, run_insert, run_delete;
            u512_vec_t cost_if_insert, cost_if_delete, cell_score;
            pre_substitution.zmm = _mm512_maskz_loadu_epi32(part_mask, scores_pre_substitution + offset);
            pre_insert_open.zmm = _mm512_maskz_loadu_epi32(part_mask, scores_pre_insertion + offset);
            pre_delete_open.zmm = _mm512_maskz_loadu_epi32(part_mask, scores_pre_deletion + offset);
            run_insert.zmm = _mm512_maskz_loadu_epi32(part_mask, scores_running_insertions + offset);
            run_delete.zmm = _mm512_maskz_loadu_epi32(part_mask, scores_running_deletions + offset);
            cost_if_insert.zmm = _mm512_max_epi32(_mm512_add_epi32(run_insert.zmm, gap_expand_vec.zmm),
                                                  _mm512_add_epi32(pre_insert_open.zmm, gap_open_vec.zmm));
            cost_if_delete.zmm = _mm512_max_epi32(_mm512_add_epi32(run_delete.zmm, gap_expand_vec.zmm),
                                                  _mm512_add_epi32(pre_delete_open.zmm, gap_open_vec.zmm));
            cell_score.zmm = _mm512_max_epi32(
                _mm512_add_epi32(pre_substitution.zmm, cost_of_substitution_i32_vecs[part].zmm),
                _mm512_max_epi32(cost_if_insert.zmm, cost_if_delete.zmm));
            _mm512_mask_storeu_epi32(scores_new + offset, part_mask, cell_score.zmm);
            _mm512_mask_storeu_epi32(scores_new_insertions + offset, part_mask, cost_if_insert.zmm);
            _mm512_mask_storeu_epi32(scores_new_deletions + offset, part_mask, cost_if_delete.zmm);
        }
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                     //
        u8_t const *first_reversed_classes, u8_t const *second_classes,           //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,  //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions, //
        i32_t const *scores_running_deletions, i32_t *scores_new,                 //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                //
        u512_vec_t gap_open_vec, u512_vec_t gap_expand_vec, size_t from, size_t to) noexcept {

        for (size_t page = from; page < to; ++page) {
            size_t const progress = page * step_k;
            slice_upto64chars(                                                        //
                first_reversed_classes + progress, second_classes + progress, step_k, //
                scores_pre_substitution + progress, scores_pre_insertion + progress,  //
                scores_pre_deletion + progress, scores_running_insertions + progress, //
                scores_running_deletions + progress, scores_new + progress,           //
                scores_new_insertions + progress, scores_new_deletions + progress,    //
                gap_open_vec, gap_expand_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,         //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions,        //
        i32_t const *scores_running_deletions, i32_t *scores_new,                        //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                       //
        executor_type_ &&executor = {}) noexcept {

        // ! Both slices already carry @b class bytes, pre-classified once by the diagonal walker.
        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;

        u512_vec_t gap_open_vec, gap_expand_vec;
        gap_open_vec.zmm = _mm512_set1_epi32(this->gap_costs_.open);
        gap_expand_vec.zmm = _mm512_set1_epi32(this->gap_costs_.extend);

        size_t const body_pages = length / step_k;
        executor.for_slices(body_pages, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_running_insertions,
                                    scores_running_deletions, scores_new, scores_new_insertions, scores_new_deletions,
                                    gap_open_vec, gap_expand_vec, from, to);
        });

        size_t const progress = body_pages * step_k;
        size_t const tail = length - progress;
        if (tail)
            slice_upto64chars(                                                        //
                first_reversed_classes + progress, second_classes + progress, tail,   //
                scores_pre_substitution + progress, scores_pre_insertion + progress,  //
                scores_pre_deletion + progress, scores_running_insertions + progress, //
                scores_running_deletions + progress, scores_new + progress,           //
                scores_new_insertions + progress, scores_new_deletions + progress,    //
                gap_open_vec, gap_expand_vec);

        if (length == 1) this->last_score_ = scores_new[0];
    }
};

/**
 *  @brief Ice Lake @b affine-gap diagonal class scorer - maximizes the local Smith-Waterman score over
 *         `i32_t` cells. Mirrors the `i32_t` global affine scorer, plus the Smith-Waterman-Gotoh
 *         zero-reset on the substitution term and the running-best reduction.
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 */
template <sz_capability_t capability_>
struct tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                   sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>>
    : public tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t,
                         sz_maximize_score_k, sz_similarity_local_k, sz_cap_serial_k, void> {

    using tile_scorer<char const *, char const *, i32_t, error_costs_32x32_t, affine_gap_costs_t, sz_maximize_score_k,
                      sz_similarity_local_k, sz_cap_serial_k, void>::tile_scorer;

    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    static constexpr size_t step_k = 64;

    substitution_lookup_icelake_t lookup_;

    void prepare(bool transpose) noexcept {
        lookup_.reload_costs(this->substituter_.class_substitution_costs, transpose);
    }

    SZ_INLINE void slice_upto64chars(                                             //
        u8_t const *first_reversed_slice, u8_t const *second_slice, size_t n,     //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,  //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions, //
        i32_t const *scores_running_deletions, i32_t *scores_new,                 //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                //
        u512_vec_t gap_open_vec, u512_vec_t gap_expand_vec) const noexcept {

        u512_vec_t first_vec, second_vec, cost_of_substitution_i8_vec, cost_of_substitution_i32_vecs[4];

        // The four 16-lane score sub-masks are just consecutive slices of the single 64-lane byte mask.
        __mmask64 const load_mask = sz_u64_mask_until_(n);
        __mmask16 const load_masks[4] = {(__mmask16)load_mask, (__mmask16)(load_mask >> 16),
                                         (__mmask16)(load_mask >> 32), (__mmask16)(load_mask >> 48)};
        first_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, first_reversed_slice);
        second_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, second_slice);
        cost_of_substitution_i8_vec = lookup_.lookup64(first_vec, second_vec);
        cost_of_substitution_i32_vecs[0].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 0));
        cost_of_substitution_i32_vecs[1].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 1));
        cost_of_substitution_i32_vecs[2].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 2));
        cost_of_substitution_i32_vecs[3].zmm = _mm512_cvtepi8_epi32(
            _mm512_extracti32x4_epi32(cost_of_substitution_i8_vec.zmm, 3));

        for (size_t part = 0; part != 4; ++part) {
            __mmask16 const part_mask = load_masks[part];
            size_t const offset = part * 16;
            u512_vec_t pre_substitution, pre_insert_open, pre_delete_open, run_insert, run_delete;
            u512_vec_t cost_if_insert, cost_if_delete, cell_score;
            pre_substitution.zmm = _mm512_maskz_loadu_epi32(part_mask, scores_pre_substitution + offset);
            pre_insert_open.zmm = _mm512_maskz_loadu_epi32(part_mask, scores_pre_insertion + offset);
            pre_delete_open.zmm = _mm512_maskz_loadu_epi32(part_mask, scores_pre_deletion + offset);
            run_insert.zmm = _mm512_maskz_loadu_epi32(part_mask, scores_running_insertions + offset);
            run_delete.zmm = _mm512_maskz_loadu_epi32(part_mask, scores_running_deletions + offset);
            cost_if_insert.zmm = _mm512_max_epi32(_mm512_add_epi32(run_insert.zmm, gap_expand_vec.zmm),
                                                  _mm512_add_epi32(pre_insert_open.zmm, gap_open_vec.zmm));
            cost_if_delete.zmm = _mm512_max_epi32(_mm512_add_epi32(run_delete.zmm, gap_expand_vec.zmm),
                                                  _mm512_add_epi32(pre_delete_open.zmm, gap_open_vec.zmm));
            // In Local Alignment for SW the zero-reset is applied to @b only the substitution term;
            // the insertion/deletion gap matrices are not clamped, exactly like the serial scorer.
            cell_score.zmm = _mm512_max_epi32(
                _mm512_max_epi32(_mm512_add_epi32(pre_substitution.zmm, cost_of_substitution_i32_vecs[part].zmm),
                                 _mm512_setzero_epi32()),
                _mm512_max_epi32(cost_if_insert.zmm, cost_if_delete.zmm));
            _mm512_mask_storeu_epi32(scores_new + offset, part_mask, cell_score.zmm);
            _mm512_mask_storeu_epi32(scores_new_insertions + offset, part_mask, cost_if_insert.zmm);
            _mm512_mask_storeu_epi32(scores_new_deletions + offset, part_mask, cost_if_delete.zmm);
        }
    }

    /** @brief Executor-independent trampoline, computing one anti-diagonal of the DP matrix. */
    SZ_NOINLINE void score_slice_trampoline_(                                     //
        u8_t const *first_reversed_classes, u8_t const *second_classes,           //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,  //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions, //
        i32_t const *scores_running_deletions, i32_t *scores_new,                 //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                //
        u512_vec_t gap_open_vec, u512_vec_t gap_expand_vec, size_t from, size_t to) noexcept {

        for (size_t page = from; page < to; ++page) {
            size_t const progress = page * step_k;
            slice_upto64chars(                                                        //
                first_reversed_classes + progress, second_classes + progress, step_k, //
                scores_pre_substitution + progress, scores_pre_insertion + progress,  //
                scores_pre_deletion + progress, scores_running_insertions + progress, //
                scores_running_deletions + progress, scores_new + progress,           //
                scores_new_insertions + progress, scores_new_deletions + progress,    //
                gap_open_vec, gap_expand_vec);
        }
    }

    template <typename executor_type_ = dummy_executor_t>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    void operator()(                                                                     //
        char const *first_reversed_slice, char const *second_slice, size_t const length, //
        i32_t const *scores_pre_substitution, i32_t const *scores_pre_insertion,         //
        i32_t const *scores_pre_deletion, i32_t const *scores_running_insertions,        //
        i32_t const *scores_running_deletions, i32_t *scores_new,                        //
        i32_t *scores_new_insertions, i32_t *scores_new_deletions,                       //
        executor_type_ &&executor = {}) noexcept {

        // ! Both slices already carry @b class bytes, pre-classified once by the diagonal walker.
        u8_t const *first_reversed_classes = (u8_t const *)first_reversed_slice;
        u8_t const *second_classes = (u8_t const *)second_slice;
        i32_t *const scores_new_begin = scores_new;

        u512_vec_t gap_open_vec, gap_expand_vec;
        gap_open_vec.zmm = _mm512_set1_epi32(this->gap_costs_.open);
        gap_expand_vec.zmm = _mm512_set1_epi32(this->gap_costs_.extend);

        size_t const body_pages = length / step_k;
        executor.for_slices(body_pages, [&](size_t from, size_t to) noexcept {
            score_slice_trampoline_(first_reversed_classes, second_classes, scores_pre_substitution,
                                    scores_pre_insertion, scores_pre_deletion, scores_running_insertions,
                                    scores_running_deletions, scores_new, scores_new_insertions, scores_new_deletions,
                                    gap_open_vec, gap_expand_vec, from, to);
        });

        size_t const progress = body_pages * step_k;
        size_t const tail = length - progress;
        if (tail)
            slice_upto64chars(                                                        //
                first_reversed_classes + progress, second_classes + progress, tail,   //
                scores_pre_substitution + progress, scores_pre_insertion + progress,  //
                scores_pre_deletion + progress, scores_running_insertions + progress, //
                scores_running_deletions + progress, scores_new + progress,           //
                scores_new_insertions + progress, scores_new_deletions + progress,    //
                gap_open_vec, gap_expand_vec);

        // The running best across the whole matrix is the reported local-alignment score.
        i32_t best_in_diagonal = this->best_score_;
        for (size_t i = 0; i != length; ++i) best_in_diagonal = sz_max_of_two(best_in_diagonal, scores_new_begin[i]);
        this->best_score_ = best_in_diagonal;
    }
};

/** @brief Redirects the Ice Lake template specialization to the serial version. */
template <typename char_type_, typename score_type_, typename substituter_type_, typename gap_costs_type_,
          sz_similarity_objective_t objective_, sz_similarity_locality_t locality_>
struct horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, objective_, locality_,
                         sz_cap_icelake_k, void>
    : public horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, objective_, locality_,
                               sz_cap_serial_k, void> {

    using base_t = horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, objective_, locality_,
                                     sz_cap_serial_k, void>;

    using base_t::base_t;
    using base_t::operator();
};

/**
 *  @brief Ice Lake diagonal "walker" for class-based substitution costs with linear gaps.
 *         Mirrors the serial diagonal walker, but pre-classifies @b both strings once into class-index buffers
 *         feeding the AVX-512 diagonal scorers above, which the generic serial walker (operating on raw
 *         characters) would otherwise leave on the scalar serial path.
 *
 *  Because the walker swaps the shorter string into the reversed `first` operand, an @b asymmetric (32 x 32)
 *  cost matrix would be looked up as `T[second][first]` after a swap. We compensate by threading a `transpose`
 *  bit into `tile_scorer_t::prepare`, which folds the resident table from `T` transposed (one-time, off the
 *  hot path), so the recurrence keeps reading the original `class_substitution_costs[first][second]`.
 */
template <typename score_type_, sz_similarity_objective_t objective_, sz_similarity_locality_t locality_>
struct diagonal_walker<char, score_type_, error_costs_32x32_t, linear_gap_costs_t, objective_, locality_,
                       sz_cap_icelake_k, void> {

    using char_t = char;
    using score_t = score_type_;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_icelake_k;
    static constexpr size_t step_classes_k = 64;

    using tile_scorer_t = tile_scorer<char_t const *, char_t const *, score_t, substituter_t, gap_costs_t, objective_k,
                                      locality_k, capability_k>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    diagonal_walker() noexcept {}
    diagonal_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Byte offsets of this walker's scratch sub-buffers within its `scratch_space`. */
    struct layout_t {
        size_t previous_scores = 0; // ? The 3 rotating score diagonals.
        size_t current_scores = 0;
        size_t next_scores = 0;
        size_t shorter_reversed = 0;         // ? Reversed shorter string (carries a step_classes_k overread guard).
        size_t shorter_reversed_classes = 0; // ? Its pre-classified class indices.
        size_t longer_classes = 0;           // ? The longer string's pre-classified class indices.
        size_t total = 0;                    // ? Bytes this walker touches; doubles as its scratch-size estimate.
        constexpr operator size_t() const noexcept { return total; }
    };

    /** @brief The single source of truth for this walker's scratch size and sub-buffer offsets. */
    layout_t layout(span<char_t const> first, span<char_t const> second, cpu_specs_t const &specs) const noexcept {
        size_t const shorter_length = sz_min_of_two(first.size(), second.size());
        size_t const longer_length = sz_max_of_two(first.size(), second.size());
        size_t const diagonal_bytes = sizeof(score_t) * (shorter_length + 1); // one anti-diagonal, unpadded
        size_t const shorter_stream_bytes = shorter_length + step_classes_k;  // string + SIMD overread guard
        size_t const longer_stream_bytes = longer_length + step_classes_k;
        scratch_amount_t amount {specs.cache_line_width};
        layout_t at;
        at.previous_scores = amount, amount += diagonal_bytes;
        at.current_scores = amount, amount += diagonal_bytes;
        at.next_scores = amount, amount += diagonal_bytes;
        at.shorter_reversed = amount, amount += shorter_stream_bytes;
        at.shorter_reversed_classes = amount, amount += shorter_stream_bytes;
        at.longer_classes = amount, amount += longer_stream_bytes;
        at.total = amount;
        return at;
    }

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score.
     */
    template <typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, score_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &&executor,
                        cpu_specs_t const &specs) const noexcept {

        // Early exit for empty strings.
        if (first.empty() || second.empty()) {
            result_ref = 0;
            if constexpr (locality_k == sz_similarity_global_k) {
                if (!first.empty() && second.empty()) { result_ref = gap_costs_.open_or_extend * first.size(); }
                else if (first.empty() && !second.empty()) { result_ref = gap_costs_.open_or_extend * second.size(); }
            }
            return status_t::success_k;
        }

        // Make sure the size relation between the strings is correct.
        char_t const *shorter = first.data(), *longer = second.data();
        size_t shorter_length = first.size(), longer_length = second.size();
        bool transpose = false;
        if (shorter_length > longer_length) {
            trivial_swap(shorter, longer);
            trivial_swap(shorter_length, longer_length);
            transpose = true;
        }

        // We are going to store 3 diagonals of the matrix.
        // The length of the longest (main) diagonal would be `shorter_dim = (shorter_length + 1)`.
        size_t const shorter_dim = shorter_length + 1;
        size_t const longer_dim = longer_length + 1;
        size_t const diagonals_count = shorter_dim + longer_dim - 1;
        size_t const max_diagonal_length = shorter_length + 1;

        // One `layout()` describes every sub-buffer (3 diagonals, the reversed shorter string, and both
        // class-index streams, each padded for the step_classes_k SIMD overread). We validate the walker's own
        // footprint and place the pointers from it.
        layout_t const at = layout(first, second, specs);
        if (scratch_space.size() < at.total) return status_t::bad_alloc_k;
        score_t *previous_scores = (score_t *)(scratch_space.data() + at.previous_scores);
        score_t *current_scores = (score_t *)(scratch_space.data() + at.current_scores);
        score_t *next_scores = (score_t *)(scratch_space.data() + at.next_scores);
        char_t *const shorter_reversed = (char_t *)(scratch_space.data() + at.shorter_reversed);
        char_t *const shorter_reversed_classes = (char_t *)(scratch_space.data() + at.shorter_reversed_classes);
        char_t *const longer_classes = (char_t *)(scratch_space.data() + at.longer_classes);

        // Export the reversed shorter string, then classify both strings @b once into their class-index buffers.
        for (size_t i = 0; i != shorter_length; ++i) shorter_reversed[i] = shorter[shorter_length - 1 - i];

        tile_scorer_t scorer {substituter_, gap_costs_};
        scorer.lookup_.reload_classes(substituter_.byte_to_class);
        scorer.prepare(transpose);
        classify_into_(scorer.lookup_, shorter_reversed, shorter_length, shorter_reversed_classes);
        classify_into_(scorer.lookup_, longer, longer_length, longer_classes);

        // Initialize the first two diagonals:
        scorer.init_score(previous_scores[0], 0);
        scorer.init_score(current_scores[0], 1);
        scorer.init_score(current_scores[1], 1);

        size_t next_diagonal_index = 2;

        // Progress through the upper-left triangle of the matrix.
        for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {

            size_t const next_diagonal_length = next_diagonal_index + 1;
            scorer(                                                                  //
                shorter_reversed_classes + shorter_length - next_diagonal_index + 1, // first sequence of classes
                longer_classes,                                                      // second sequence of classes
                next_diagonal_length - 2,           // number of elements to compute with the `scorer`
                previous_scores,                    // costs pre substitution
                current_scores, current_scores + 1, // costs pre insertion/deletion
                next_scores + 1,                    // new scores for the next diagonal
                executor);                          // parallel execution within the diagonal

            // Don't forget to populate the first row and the first column of the matrix.
            scorer.init_score(next_scores[0], next_diagonal_index);
            scorer.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);
            rotate_three(previous_scores, current_scores, next_scores);
        }

        // Now let's handle the anti-diagonal band of the matrix.
        for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {

            size_t const next_diagonal_length = shorter_dim;
            scorer(                                                          //
                shorter_reversed_classes + shorter_length - shorter_dim + 1, // first sequence of classes
                longer_classes + next_diagonal_index - shorter_dim,          // second sequence of classes
                next_diagonal_length - 1,                                    // number of elements to compute
                previous_scores,                                             // costs pre substitution
                current_scores, current_scores + 1,                          // costs pre insertion/deletion
                next_scores,                                                 // new scores for the next diagonal
                executor);                                                   // parallel execution within the diagonal

            scorer.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);
            rotate_three(previous_scores, current_scores, next_scores);
            sz_move_serial((ptr_t)(previous_scores), (ptr_t)(previous_scores + 1),
                           (max_diagonal_length - 1) * sizeof(score_t));
        }

        // Now let's handle the bottom-right triangle of the matrix.
        for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {

            size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
            scorer(                                                          //
                shorter_reversed_classes + shorter_length - shorter_dim + 1, // first sequence of classes
                longer_classes + next_diagonal_index - shorter_dim,          // second sequence of classes
                next_diagonal_length,                                        // number of elements to compute
                previous_scores,                                             // costs pre substitution
                current_scores, current_scores + 1,                          // costs pre insertion/deletion
                next_scores,                                                 // new scores for the next diagonal
                executor);                                                   // parallel execution within the diagonal

            rotate_three(previous_scores, current_scores, next_scores);
            previous_scores++;
        }

        result_ref = scorer.score();
        return status_t::success_k;
    }

  private:
    /** @brief Maps a raw byte string into class bytes using the resident `byte_to_class` lookup, @b amortized. */
    static void classify_into_(substitution_lookup_icelake_t const &lookup, char_t const *source, size_t length,
                               char_t *classes) noexcept {
        u512_vec_t source_vec, classes_vec;
        size_t progress = 0;
        for (; progress + step_classes_k <= length; progress += step_classes_k) {
            source_vec.zmm = _mm512_loadu_epi8(source + progress);
            classes_vec = lookup.classify64(source_vec);
            _mm512_storeu_epi8(classes + progress, classes_vec.zmm);
        }
        if (progress < length) {
            __mmask64 const load_mask = sz_u64_mask_until_(length - progress);
            source_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, source + progress);
            classes_vec = lookup.classify64(source_vec);
            _mm512_mask_storeu_epi8(classes + progress, load_mask, classes_vec.zmm);
        }
    }
};

/**
 *  @brief Ice Lake diagonal "walker" for class-based substitution costs with @b affine gaps.
 *         Mirrors the linear class walker above for the classify-once + `prepare(transpose)` machinery,
 *         but threads the 7-diagonal Gotoh layout (3 main score diagonals + 2 insertion + 2 deletion) of
 *         the serial affine walker, feeding the AVX-512 affine diagonal scorers (5-in / 3-out).
 *
 *  Because the walker swaps the shorter string into the reversed `first` operand, an @b asymmetric (32 x 32)
 *  cost matrix would be looked up as `T[second][first]` after a swap. We compensate by threading a `transpose`
 *  bit into `tile_scorer_t::prepare`, which folds the resident table from `T` transposed (one-time, off the
 *  hot path), so the recurrence keeps reading the original `class_substitution_costs[first][second]`.
 */
template <typename score_type_, sz_similarity_objective_t objective_, sz_similarity_locality_t locality_>
struct diagonal_walker<char, score_type_, error_costs_32x32_t, affine_gap_costs_t, objective_, locality_,
                       sz_cap_icelake_k, void> {

    using char_t = char;
    using score_t = score_type_;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = sz_cap_icelake_k;
    static constexpr size_t step_classes_k = 64;

    using tile_scorer_t = tile_scorer<char_t const *, char_t const *, score_t, substituter_t, gap_costs_t, objective_k,
                                      locality_k, capability_k>;

    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};

    diagonal_walker() noexcept {}
    diagonal_walker(substituter_t subs, affine_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Byte offsets of this walker's scratch sub-buffers within its `scratch_space`. */
    struct layout_t {
        size_t previous_scores = 0; // ? The 3 rotating score diagonals.
        size_t current_scores = 0;
        size_t next_scores = 0;
        size_t current_inserts = 0; // ? The 2 rotating insertion-gap diagonals.
        size_t next_inserts = 0;
        size_t current_deletes = 0; // ? The 2 rotating deletion-gap diagonals.
        size_t next_deletes = 0;
        size_t shorter_reversed = 0;         // ? Reversed shorter string (carries a step_classes_k overread guard).
        size_t shorter_reversed_classes = 0; // ? Its pre-classified class indices.
        size_t longer_classes = 0;           // ? The longer string's pre-classified class indices.
        size_t total = 0;                    // ? Bytes this walker touches; doubles as its scratch-size estimate.
        constexpr operator size_t() const noexcept { return total; }
    };

    /** @brief The single source of truth for this walker's scratch size and sub-buffer offsets. */
    layout_t layout(span<char_t const> first, span<char_t const> second, cpu_specs_t const &specs) const noexcept {
        size_t const shorter_length = sz_min_of_two(first.size(), second.size());
        size_t const longer_length = sz_max_of_two(first.size(), second.size());
        size_t const diagonal_bytes = sizeof(score_t) * (shorter_length + 1); // one anti-diagonal, unpadded
        size_t const shorter_stream_bytes = shorter_length + step_classes_k;  // string + SIMD overread guard
        size_t const longer_stream_bytes = longer_length + step_classes_k;
        scratch_amount_t amount {specs.cache_line_width};
        layout_t at;
        at.previous_scores = amount, amount += diagonal_bytes;
        at.current_scores = amount, amount += diagonal_bytes;
        at.next_scores = amount, amount += diagonal_bytes;
        at.current_inserts = amount, amount += diagonal_bytes;
        at.next_inserts = amount, amount += diagonal_bytes;
        at.current_deletes = amount, amount += diagonal_bytes;
        at.next_deletes = amount, amount += diagonal_bytes;
        at.shorter_reversed = amount, amount += shorter_stream_bytes;
        at.shorter_reversed_classes = amount, amount += shorter_stream_bytes;
        at.longer_classes = amount, amount += longer_stream_bytes;
        at.total = amount;
        return at;
    }

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score.
     */
    template <typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, score_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &&executor,
                        cpu_specs_t const &specs) const noexcept {

        // Early exit for empty strings.
        if (first.empty() || second.empty()) {
            result_ref = 0;
            if constexpr (locality_k == sz_similarity_global_k) {
                if (!first.empty() && second.empty()) {
                    result_ref = gap_costs_.open + gap_costs_.extend * (first.size() - 1);
                }
                else if (first.empty() && !second.empty()) {
                    result_ref = gap_costs_.open + gap_costs_.extend * (second.size() - 1);
                }
            }
            return status_t::success_k;
        }

        // Make sure the size relation between the strings is correct.
        char_t const *shorter = first.data(), *longer = second.data();
        size_t shorter_length = first.size(), longer_length = second.size();
        bool transpose = false;
        if (shorter_length > longer_length) {
            trivial_swap(shorter, longer);
            trivial_swap(shorter_length, longer_length);
            transpose = true;
        }

        // We are going to store 7 diagonals of the matrix (3 main + 2 insert + 2 delete).
        // The length of the longest (main) diagonal would be `shorter_dim = (shorter_length + 1)`.
        size_t const shorter_dim = shorter_length + 1;
        size_t const longer_dim = longer_length + 1;
        size_t const diagonals_count = shorter_dim + longer_dim - 1;
        size_t const max_diagonal_length = shorter_length + 1;

        // One `layout()` describes every sub-buffer (7 diagonals, the reversed shorter string, and both
        // class-index streams, each padded for the step_classes_k SIMD overread). We validate the walker's own
        // footprint and place the pointers from it.
        layout_t const at = layout(first, second, specs);
        if (scratch_space.size() < at.total) return status_t::bad_alloc_k;
        score_t *previous_scores = (score_t *)(scratch_space.data() + at.previous_scores);
        score_t *current_scores = (score_t *)(scratch_space.data() + at.current_scores);
        score_t *next_scores = (score_t *)(scratch_space.data() + at.next_scores);
        score_t *current_inserts = (score_t *)(scratch_space.data() + at.current_inserts);
        score_t *next_inserts = (score_t *)(scratch_space.data() + at.next_inserts);
        score_t *current_deletes = (score_t *)(scratch_space.data() + at.current_deletes);
        score_t *next_deletes = (score_t *)(scratch_space.data() + at.next_deletes);
        char_t *const shorter_reversed = (char_t *)(scratch_space.data() + at.shorter_reversed);
        char_t *const shorter_reversed_classes = (char_t *)(scratch_space.data() + at.shorter_reversed_classes);
        char_t *const longer_classes = (char_t *)(scratch_space.data() + at.longer_classes);

        // Export the reversed shorter string, then classify both strings @b once into their class-index buffers.
        for (size_t i = 0; i != shorter_length; ++i) shorter_reversed[i] = shorter[shorter_length - 1 - i];

        tile_scorer_t scorer {substituter_, gap_costs_};
        scorer.lookup_.reload_classes(substituter_.byte_to_class);
        scorer.prepare(transpose);
        classify_into_(scorer.lookup_, shorter_reversed, shorter_length, shorter_reversed_classes);
        classify_into_(scorer.lookup_, longer, longer_length, longer_classes);

        // Initialize the first two diagonals:
        scorer.init_score(previous_scores[0], 0);
        scorer.init_score(current_scores[0], 1);
        scorer.init_score(current_scores[1], 1);
        scorer.init_gap(current_inserts[0], 1);
        scorer.init_gap(current_deletes[1], 1);

        size_t next_diagonal_index = 2;

        // Progress through the upper-left triangle of the matrix.
        for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {

            size_t const next_diagonal_length = next_diagonal_index + 1;
            scorer(                                                                  //
                shorter_reversed_classes + shorter_length - next_diagonal_index + 1, // first sequence of classes
                longer_classes,                                                      // second sequence of classes
                next_diagonal_length - 2,             // number of elements to compute with the `scorer`
                previous_scores,                      // costs pre substitution
                current_scores, current_scores + 1,   // costs pre insertion/deletion opening
                current_inserts, current_deletes + 1, // costs pre insertion/deletion extension
                next_scores + 1,                      // updated similarity scores
                next_inserts + 1, next_deletes + 1,   // updated insertion/deletion extensions
                executor);                            // parallel execution within the diagonal

            // Don't forget to populate the first row and the first column of the matrix.
            scorer.init_score(next_scores[0], next_diagonal_index);
            scorer.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);
            scorer.init_gap(next_inserts[0], next_diagonal_index);
            scorer.init_gap(next_deletes[next_diagonal_length - 1], next_diagonal_index);

            rotate_three(previous_scores, current_scores, next_scores);
            trivial_swap(current_inserts, next_inserts);
            trivial_swap(current_deletes, next_deletes);
        }

        // Now let's handle the anti-diagonal band of the matrix.
        for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {

            size_t const next_diagonal_length = shorter_dim;
            scorer(                                                          //
                shorter_reversed_classes + shorter_length - shorter_dim + 1, // first sequence of classes
                longer_classes + next_diagonal_index - shorter_dim,          // second sequence of classes
                next_diagonal_length - 1,                                    // number of elements to compute
                previous_scores,                                             // costs pre substitution
                current_scores, current_scores + 1,                          // costs pre insertion/deletion opening
                current_inserts, current_deletes + 1,                        // costs pre insertion/deletion extension
                next_scores,                                                 // updated similarity scores
                next_inserts, next_deletes,                                  // updated insertion/deletion extensions
                executor);                                                   // parallel execution within the diagonal

            scorer.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);
            scorer.init_gap(next_deletes[next_diagonal_length - 1], next_diagonal_index);

            rotate_three(previous_scores, current_scores, next_scores);
            trivial_swap(current_inserts, next_inserts);
            trivial_swap(current_deletes, next_deletes);
            sz_move_serial((ptr_t)(previous_scores), (ptr_t)(previous_scores + 1),
                           (max_diagonal_length - 1) * sizeof(score_t));
        }

        // Now let's handle the bottom-right triangle of the matrix.
        for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {

            size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
            scorer(                                                          //
                shorter_reversed_classes + shorter_length - shorter_dim + 1, // first sequence of classes
                longer_classes + next_diagonal_index - shorter_dim,          // second sequence of classes
                next_diagonal_length,                                        // number of elements to compute
                previous_scores,                                             // costs pre substitution
                current_scores, current_scores + 1,                          // costs pre insertion/deletion opening
                current_inserts, current_deletes + 1,                        // costs pre insertion/deletion extension
                next_scores,                                                 // updated similarity scores
                next_inserts, next_deletes,                                  // updated insertion/deletion extensions
                executor);                                                   // parallel execution within the diagonal

            rotate_three(previous_scores, current_scores, next_scores);
            trivial_swap(current_inserts, next_inserts);
            trivial_swap(current_deletes, next_deletes);
            previous_scores++;
        }

        result_ref = scorer.score();
        return status_t::success_k;
    }

  private:
    /** @brief Maps a raw byte string into class bytes using the resident `byte_to_class` lookup, @b amortized. */
    static void classify_into_(substitution_lookup_icelake_t const &lookup, char_t const *source, size_t length,
                               char_t *classes) noexcept {
        u512_vec_t source_vec, classes_vec;
        size_t progress = 0;
        for (; progress + step_classes_k <= length; progress += step_classes_k) {
            source_vec.zmm = _mm512_loadu_epi8(source + progress);
            classes_vec = lookup.classify64(source_vec);
            _mm512_storeu_epi8(classes + progress, classes_vec.zmm);
        }
        if (progress < length) {
            __mmask64 const load_mask = sz_u64_mask_until_(length - progress);
            source_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, source + progress);
            classes_vec = lookup.classify64(source_vec);
            _mm512_mask_storeu_epi8(classes + progress, load_mask, classes_vec.zmm);
        }
    }
};

/**
 *  @brief Computes the @b byte-level Needleman-Wunsch score between two strings using the Ice Lake backend.
 *  @sa `levenshtein_distance` for uniform substitution and gap costs.
 */
template <>
struct needleman_wunsch_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sil_k> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr size_t diagonal_buffers_count_k = 3;
    using diagonal_i16_t = diagonal_walker<char_t, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_global_k, sz_cap_icelake_k>;
    using diagonal_i32_t = diagonal_walker<char_t, i32_t, substituter_t, gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_global_k, sz_cap_icelake_k>;
    using diagonal_i64_t = diagonal_walker<char_t, i64_t, substituter_t, gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_global_k, sz_cap_serial_k>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    needleman_wunsch_score() noexcept {}
    needleman_wunsch_score(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    size_t scratch_space_needed(span<char_t const> first, span<char_t const> second,
                                cpu_specs_t const &specs) const noexcept {
        size_t const shorter_length = std::min(first.size(), second.size());
        size_t const longer_length = std::max(first.size(), second.size());
        size_t const max_diagonal_length = shorter_length + 1;
        size_t const padded_diagonal_length =
            round_up_to_multiple(sizeof(i64_t) * max_diagonal_length, specs.cache_line_width) / sizeof(i64_t);
        size_t const padded_shorter_stream_length = round_up_to_multiple(
            shorter_length + diagonal_i16_t::step_classes_k, specs.cache_line_width);
        size_t const padded_longer_stream_length = round_up_to_multiple(longer_length + diagonal_i16_t::step_classes_k,
                                                                        specs.cache_line_width);
        return sizeof(i64_t) * padded_diagonal_length * diagonal_buffers_count_k + padded_shorter_stream_length * 2 +
               padded_longer_stream_length;
    }

    template <typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, ssize_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &executor,
                        cpu_specs_t const &specs) const noexcept {

        using diagonal_memory_requirements_t = diagonal_memory_requirements<ssize_t>;
        diagonal_memory_requirements_t requirements(                                   //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), specs.cache_line_width);

        status_t status = status_t::success_k;
        if (requirements.bytes_per_cell <= 2) {
            i16_t result_i16;
            status = diagonal_i16_t {substituter_, gap_costs_}(first, second, result_i16, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            i32_t result_i32;
            status = diagonal_i32_t {substituter_, gap_costs_}(first, second, result_i32, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            i64_t result_i64;
            status = diagonal_i64_t {substituter_, gap_costs_}(first, second, result_i64, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i64;
        }

        return status;
    }
};

/**
 *  @brief Computes the @b byte-level Needleman-Wunsch score with @b affine gaps using the Ice Lake backend.
 *  @sa `levenshtein_distance` for uniform substitution and gap costs.
 */
template <>
struct needleman_wunsch_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sil_k> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr size_t diagonal_buffers_count_k = 7;
    using diagonal_i16_t = diagonal_walker<char_t, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_global_k, sz_cap_icelake_k>;
    using diagonal_i32_t = diagonal_walker<char_t, i32_t, substituter_t, gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_global_k, sz_cap_icelake_k>;
    using diagonal_i64_t = diagonal_walker<char_t, i64_t, substituter_t, gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_global_k, sz_cap_serial_k>;

    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};

    needleman_wunsch_score() noexcept {}
    needleman_wunsch_score(substituter_t subs, affine_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    size_t scratch_space_needed(span<char_t const> first, span<char_t const> second,
                                cpu_specs_t const &specs) const noexcept {
        size_t const shorter_length = std::min(first.size(), second.size());
        size_t const longer_length = std::max(first.size(), second.size());
        size_t const max_diagonal_length = shorter_length + 1;
        size_t const padded_diagonal_length =
            round_up_to_multiple(sizeof(i64_t) * max_diagonal_length, specs.cache_line_width) / sizeof(i64_t);
        size_t const padded_shorter_stream_length = round_up_to_multiple(
            shorter_length + diagonal_i16_t::step_classes_k, specs.cache_line_width);
        size_t const padded_longer_stream_length = round_up_to_multiple(longer_length + diagonal_i16_t::step_classes_k,
                                                                        specs.cache_line_width);
        return sizeof(i64_t) * padded_diagonal_length * diagonal_buffers_count_k + padded_shorter_stream_length * 2 +
               padded_longer_stream_length;
    }

    template <typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, ssize_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &executor,
                        cpu_specs_t const &specs) const noexcept {

        using diagonal_memory_requirements_t = diagonal_memory_requirements<ssize_t>;
        diagonal_memory_requirements_t requirements(                                   //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), specs.cache_line_width);

        status_t status = status_t::success_k;
        if (requirements.bytes_per_cell <= 2) {
            i16_t result_i16;
            status = diagonal_i16_t {substituter_, gap_costs_}(first, second, result_i16, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            i32_t result_i32;
            status = diagonal_i32_t {substituter_, gap_costs_}(first, second, result_i32, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            i64_t result_i64;
            status = diagonal_i64_t {substituter_, gap_costs_}(first, second, result_i64, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i64;
        }

        return status;
    }
};

/**
 *  @brief Computes the @b byte-level Smith-Waterman score between two strings using the Ice Lake backend.
 *  @sa `levenshtein_distance` for uniform substitution and gap costs.
 */
template <>
struct smith_waterman_score<char, error_costs_32x32_t, linear_gap_costs_t, sz_caps_sil_k> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr size_t diagonal_buffers_count_k = 3;
    using diagonal_i16_t = diagonal_walker<char_t, i16_t, substituter_t, linear_gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_local_k, sz_cap_icelake_k>;
    using diagonal_i32_t = diagonal_walker<char_t, i32_t, substituter_t, linear_gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_local_k, sz_cap_icelake_k>;
    using diagonal_i64_t = diagonal_walker<char_t, i64_t, substituter_t, linear_gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_local_k, sz_cap_serial_k>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    smith_waterman_score() noexcept {}
    smith_waterman_score(substituter_t subs, linear_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    size_t scratch_space_needed(span<char_t const> first, span<char_t const> second,
                                cpu_specs_t const &specs) const noexcept {
        size_t const shorter_length = std::min(first.size(), second.size());
        size_t const longer_length = std::max(first.size(), second.size());
        size_t const max_diagonal_length = shorter_length + 1;
        size_t const padded_diagonal_length =
            round_up_to_multiple(sizeof(i64_t) * max_diagonal_length, specs.cache_line_width) / sizeof(i64_t);
        size_t const padded_shorter_stream_length = round_up_to_multiple(
            shorter_length + diagonal_i16_t::step_classes_k, specs.cache_line_width);
        size_t const padded_longer_stream_length = round_up_to_multiple(longer_length + diagonal_i16_t::step_classes_k,
                                                                        specs.cache_line_width);
        return sizeof(i64_t) * padded_diagonal_length * diagonal_buffers_count_k + padded_shorter_stream_length * 2 +
               padded_longer_stream_length;
    }

    template <typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, ssize_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &executor,
                        cpu_specs_t const &specs) const noexcept {

        using diagonal_memory_requirements_t = diagonal_memory_requirements<ssize_t>;
        diagonal_memory_requirements_t requirements(                                   //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), specs.cache_line_width);

        status_t status = status_t::success_k;
        if (requirements.bytes_per_cell <= 2) {
            i16_t result_i16;
            status = diagonal_i16_t {substituter_, gap_costs_}(first, second, result_i16, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            i32_t result_i32;
            status = diagonal_i32_t {substituter_, gap_costs_}(first, second, result_i32, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            i64_t result_i64;
            status = diagonal_i64_t {substituter_, gap_costs_}(first, second, result_i64, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i64;
        }

        return status;
    }
};

/**
 *  @brief Computes the @b byte-level Smith-Waterman score with @b affine gaps using the Ice Lake backend.
 *  @sa `levenshtein_distance` for uniform substitution and gap costs.
 */
template <>
struct smith_waterman_score<char, error_costs_32x32_t, affine_gap_costs_t, sz_caps_sil_k> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr size_t diagonal_buffers_count_k = 7;
    using diagonal_i16_t = diagonal_walker<char_t, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_local_k, sz_cap_icelake_k>;
    using diagonal_i32_t = diagonal_walker<char_t, i32_t, substituter_t, gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_local_k, sz_cap_icelake_k>;
    using diagonal_i64_t = diagonal_walker<char_t, i64_t, substituter_t, gap_costs_t, sz_maximize_score_k,
                                           sz_similarity_local_k, sz_cap_serial_k>;

    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};

    smith_waterman_score() noexcept {}
    smith_waterman_score(substituter_t subs, affine_gap_costs_t gaps) noexcept : substituter_(subs), gap_costs_(gaps) {}

    size_t scratch_space_needed(span<char_t const> first, span<char_t const> second,
                                cpu_specs_t const &specs) const noexcept {
        size_t const shorter_length = std::min(first.size(), second.size());
        size_t const longer_length = std::max(first.size(), second.size());
        size_t const max_diagonal_length = shorter_length + 1;
        size_t const padded_diagonal_length =
            round_up_to_multiple(sizeof(i64_t) * max_diagonal_length, specs.cache_line_width) / sizeof(i64_t);
        size_t const padded_shorter_stream_length = round_up_to_multiple(
            shorter_length + diagonal_i16_t::step_classes_k, specs.cache_line_width);
        size_t const padded_longer_stream_length = round_up_to_multiple(longer_length + diagonal_i16_t::step_classes_k,
                                                                        specs.cache_line_width);
        return sizeof(i64_t) * padded_diagonal_length * diagonal_buffers_count_k + padded_shorter_stream_length * 2 +
               padded_longer_stream_length;
    }

    template <typename executor_type_>
#if SZ_HAS_CONCEPTS_
        requires executor_like<executor_type_>
#endif
    status_t operator()(span<char_t const> first, span<char_t const> second, ssize_t &result_ref,
                        scratch_space_t scratch_space, executor_type_ &executor,
                        cpu_specs_t const &specs) const noexcept {

        using diagonal_memory_requirements_t = diagonal_memory_requirements<ssize_t>;
        diagonal_memory_requirements_t requirements(                                   //
            first.size(), second.size(),                                               //
            gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
            sizeof(char_t), specs.cache_line_width);

        status_t status = status_t::success_k;
        if (requirements.bytes_per_cell <= 2) {
            i16_t result_i16;
            status = diagonal_i16_t {substituter_, gap_costs_}(first, second, result_i16, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            i32_t result_i32;
            status = diagonal_i32_t {substituter_, gap_costs_}(first, second, result_i32, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            i64_t result_i64;
            status = diagonal_i64_t {substituter_, gap_costs_}(first, second, result_i64, scratch_space, executor,
                                                               specs);
            if (status == status_t::success_k) result_ref = result_i64;
        }

        return status;
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
