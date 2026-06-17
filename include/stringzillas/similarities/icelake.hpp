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
 *  @brief AVX-512 @b lockstep Myers/Hyyrö unit-cost Levenshtein for Ice Lake, with one dedicated routine per
 *      shorter-side tier so each carries exactly the cross-lane machinery it needs:
 *      - `distances_8x64_`  - 8 independent single-word Myers (shorter <= 64), no cross-lane logic at all;
 *      - `distances_4x128_` - 4 pairs of 2 lanes (shorter <= 128), 128-bit Myers integers;
 *      - `distances_2x256_` - 2 pairs of 4 lanes (shorter <= 256), 256-bit Myers integers;
 *      - `distances_1x512_` - 1 pair of 8 lanes (shorter <= 512), a single 512-bit Myers integer.
 *      The three multi-lane tiers share the `lockstep_pairs_` core: only the carry of `(Eq & VP) + VP` and the
 *      `<< 1` ripple cross a pair's lanes (masked at pair boundaries); everything else is per-lane bit-logic.
 *      Variable per-pair lengths are handled with an active mask that freezes finished pairs.
 */
template <sz_capability_t capability_>
struct levenshtein_distance_myers<char, capability_, std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>> {

    using char_t = char;
    using index_t = u32_t;
    static constexpr index_t lanes_k = 8;
    static constexpr size_t match_masks_bytes_k = sizeof(u64_t) * lanes_k * 256; // ? 16 KB `Peq` table.

    // VPTERNLOGQ truth tables for the Myers booleans; the imm8 is indexed by `(A << 2) | (B << 1) | C`.
    static constexpr int ternlog_xor_or_k = 0xBE; // ? `(A ^ B) | C` - the diagonal tail `(sum ^ VP) | Eq`.
    static constexpr int ternlog_or_nor_k = 0xF1; // ? `A | ~(B | C)` - the shared `Ph` and `Pv'` shape.

    levenshtein_distance_myers() noexcept {}

    /** @brief Compile-time mask of the lanes that begin a pair (every `lanes_per_pair_`-th lane). */
    template <index_t lanes_per_pair_>
    static constexpr __mmask8 pair_starts_mask_() noexcept {
        __mmask8 mask = 0;
        for (index_t lane = 0; lane != lanes_k; ++lane)
            if (lane % lanes_per_pair_ == 0) mask = (__mmask8)(mask | (1u << lane));
        return mask;
    }

    /**
     *  @brief Eight independent single-word Myers distances, one per ZMM lane (each shorter side <= 64).
     *      The hot path for short words: no carry, shift, or boundary masking crosses lanes - every lane is
     *      its own 64-bit Myers integer. @p scratch_space holds the `Peq[8][256]` table (`match_masks_bytes_k`)
     *      followed by a transposed-text buffer of at least `max_longer * 8` bytes.
     */
    template <typename results_writer_>
    status_t distances_8x64_(span<char_t const> const *shorters, span<char_t const> const *longers,
                             size_t const *positions, index_t pairs_active, results_writer_ &results,
                             scratch_space_t scratch_space) const noexcept {

        size_t max_longer = 0;
        for (index_t lane = 0; lane != pairs_active; ++lane)
            max_longer = sz_max_of_two(max_longer, longers[lane].size());
        if (scratch_space.size() < match_masks_bytes_k + max_longer * lanes_k) return status_t::bad_alloc_k;

        u64_t *const match_masks = reinterpret_cast<u64_t *>(scratch_space.data()); // ? Indexed `lane * 256 + symbol`.
        u8_t *const transposed_text = reinterpret_cast<u8_t *>(scratch_space.data() + match_masks_bytes_k);
        alignas(64) u64_t top_bits[lanes_k] = {0}, shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
        for (size_t position = 0; position != max_longer * lanes_k; ++position) transposed_text[position] = 0;

        for (index_t lane = 0; lane != pairs_active; ++lane) {
            index_t const shorter_length = (index_t)shorters[lane].size();
            size_t const longer_length = longers[lane].size();
            char_t const *const shorter = shorters[lane].data();
            char_t const *const longer = longers[lane].data();
            // Zero this lane's `Peq` entries for every character its text may read (see the scalar Myers).
            for (index_t position = 0; position != shorter_length; ++position)
                match_masks[lane * 256 + (u8_t)shorter[position]] = 0;
            for (size_t position = 0; position != longer_length; ++position)
                match_masks[lane * 256 + (u8_t)longer[position]] = 0;
            for (index_t position = 0; position != shorter_length; ++position)
                match_masks[lane * 256 + (u8_t)shorter[position]] |= (u64_t)1 << position;
            top_bits[lane] = (u64_t)1 << (shorter_length - 1);
            shorter_lengths[lane] = shorter_length;
            longer_lengths[lane] = longer_length;
            for (size_t position = 0; position != longer_length; ++position)
                transposed_text[position * lanes_k + lane] = (u8_t)longer[position];
        }

        __m512i const lane_offsets = _mm512_set_epi64(7 * 256, 6 * 256, 5 * 256, 4 * 256, 3 * 256, 2 * 256, 1 * 256, 0);
        __m512i const one = _mm512_set1_epi64(1);
        __m512i const top_mask = _mm512_load_si512(top_bits), longer_vec = _mm512_load_si512(longer_lengths);
        __m512i const length_vec = _mm512_load_si512(shorter_lengths);
        // VP = the low `shorter_length` bits set (a shift count of 64 yields 0, so `(1 << 64) - 1` == ~0).
        __m512i vertical_positive = _mm512_sub_epi64(_mm512_sllv_epi64(one, length_vec), one);
        __m512i vertical_negative = _mm512_setzero_si512();
        __m512i score = length_vec;

        for (size_t position = 0; position != max_longer; ++position) {
            __mmask8 const active = _mm512_cmpgt_epi64_mask(longer_vec, _mm512_set1_epi64((long long)position));
            __m512i const symbols = _mm512_cvtepu8_epi64(
                _mm_loadl_epi64((__m128i const *)(transposed_text + position * lanes_k)));
            __m512i const equality = _mm512_i64gather_epi64(_mm512_add_epi64(lane_offsets, symbols), match_masks, 8);
            __m512i const carry_in = _mm512_or_si512(equality, vertical_negative);
            // Xh = (((Eq & VP) + VP) ^ VP) | Eq; the trailing `(sum ^ VP) | Eq` folds into one VPTERNLOGQ (0xBE).
            __m512i const sum = _mm512_add_epi64(_mm512_and_si512(equality, vertical_positive), vertical_positive);
            __m512i const diagonal = _mm512_ternarylogic_epi64(sum, vertical_positive, equality, ternlog_xor_or_k);
            // Ph = Mv | ~(Xh | VP) is `A | ~(B | C)` -> VPTERNLOGQ(0xF1).
            __m512i horizontal_positive = _mm512_ternarylogic_epi64(vertical_negative, diagonal, vertical_positive,
                                                                    ternlog_or_nor_k);
            __m512i horizontal_negative = _mm512_and_si512(vertical_positive, diagonal);
            score = _mm512_mask_add_epi64(score, active & _mm512_test_epi64_mask(horizontal_positive, top_mask), score,
                                          one);
            score = _mm512_mask_sub_epi64(score, active & _mm512_test_epi64_mask(horizontal_negative, top_mask), score,
                                          one);
            horizontal_positive = _mm512_or_si512(_mm512_slli_epi64(horizontal_positive, 1), one);
            horizontal_negative = _mm512_slli_epi64(horizontal_negative, 1);
            // Pv' = Mh | ~(Xv | Ph), the same `A | ~(B | C)` shape -> VPTERNLOGQ(0xF1).
            __m512i const next_positive = _mm512_ternarylogic_epi64(horizontal_negative, carry_in, horizontal_positive,
                                                                    ternlog_or_nor_k);
            __m512i const next_negative = _mm512_and_si512(horizontal_positive, carry_in);
            vertical_positive = _mm512_mask_blend_epi64(active, vertical_positive, next_positive);
            vertical_negative = _mm512_mask_blend_epi64(active, vertical_negative, next_negative);
        }

        alignas(64) u64_t final_scores[lanes_k];
        _mm512_store_si512(final_scores, score);
        for (index_t lane = 0; lane != pairs_active; ++lane) results[positions[lane]] = (size_t)final_scores[lane];
        return status_t::success_k;
    }

    /**
     *  @brief Cross-product single-word Myers: one shared @p query against up to 8 @p candidates per call, building
     *      the 256-entry `Peq` table once from the query and reusing it across all 8 ZMM lanes. For short strings the
     *      per-lane `Peq` rebuild of `distances_8x64_` dominates the lockstep scan; here every lane gathers from the
     *      same single table, so the build cost is paid once per query rather than once per candidate.
     *
     *      The query is the Myers pattern for every lane and each candidate is that lane's text. Unit-cost edit
     *      distance is symmetric, so this is correct even when a candidate is shorter than the query.
     *
     *      @pre `query.size() <= 64` (single 64-bit Myers integer per lane). No runtime check is performed.
     *      @p scratch_space holds the single `Peq[256]` table (256 `u64`) followed by a transposed-text buffer of
     *      at least `max_candidate_length * 8` bytes.
     */
    template <typename results_writer_>
    status_t distances_8x64_shared_query_(span<char_t const> query, span<char_t const> const *candidates,
                                          size_t const *positions, index_t candidates_active,
                                          results_writer_ &results, scratch_space_t scratch_space) const noexcept {

        static constexpr size_t shared_match_masks_bytes_k = sizeof(u64_t) * 256;

        size_t max_candidate_length = 0;
        for (index_t lane = 0; lane != candidates_active; ++lane)
            max_candidate_length = sz_max_of_two(max_candidate_length, candidates[lane].size());
        if (scratch_space.size() < shared_match_masks_bytes_k + max_candidate_length * lanes_k)
            return status_t::bad_alloc_k;

        u64_t *const match_masks = reinterpret_cast<u64_t *>(scratch_space.data()); // ? Indexed by `symbol` alone.
        u8_t *const transposed_text = reinterpret_cast<u8_t *>(scratch_space.data() + shared_match_masks_bytes_k);
        alignas(64) u64_t top_bits[lanes_k] = {0}, shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
        for (size_t position = 0; position != max_candidate_length * lanes_k; ++position) transposed_text[position] = 0;

        index_t const query_length = (index_t)query.size();
        char_t const *const query_data = query.data();
        // Clear every symbol the query and any candidate may read so stale bits from a prior query cannot leak in.
        for (index_t position = 0; position != query_length; ++position)
            match_masks[(u8_t)query_data[position]] = 0;
        for (index_t lane = 0; lane != candidates_active; ++lane) {
            size_t const candidate_length = candidates[lane].size();
            char_t const *const candidate = candidates[lane].data();
            for (size_t position = 0; position != candidate_length; ++position)
                match_masks[(u8_t)candidate[position]] = 0;
        }
        for (index_t position = 0; position != query_length; ++position)
            match_masks[(u8_t)query_data[position]] |= (u64_t)1 << position;

        for (index_t lane = 0; lane != candidates_active; ++lane) {
            size_t const candidate_length = candidates[lane].size();
            char_t const *const candidate = candidates[lane].data();
            top_bits[lane] = (u64_t)1 << (query_length - 1);
            shorter_lengths[lane] = query_length;
            longer_lengths[lane] = candidate_length;
            for (size_t position = 0; position != candidate_length; ++position)
                transposed_text[position * lanes_k + lane] = (u8_t)candidate[position];
        }

        __m512i const lane_offsets = _mm512_setzero_si512(); // ? Every lane reads the one shared table at `symbol`.
        __m512i const one = _mm512_set1_epi64(1);
        __m512i const top_mask = _mm512_load_si512(top_bits), longer_vec = _mm512_load_si512(longer_lengths);
        __m512i const length_vec = _mm512_load_si512(shorter_lengths);
        // VP = the low `query_length` bits set (a shift count of 64 yields 0, so `(1 << 64) - 1` == ~0).
        __m512i vertical_positive = _mm512_sub_epi64(_mm512_sllv_epi64(one, length_vec), one);
        __m512i vertical_negative = _mm512_setzero_si512();
        __m512i score = length_vec;

        for (size_t position = 0; position != max_candidate_length; ++position) {
            __mmask8 const active = _mm512_cmpgt_epi64_mask(longer_vec, _mm512_set1_epi64((long long)position));
            __m512i const symbols = _mm512_cvtepu8_epi64(
                _mm_loadl_epi64((__m128i const *)(transposed_text + position * lanes_k)));
            __m512i const equality = _mm512_i64gather_epi64(_mm512_add_epi64(lane_offsets, symbols), match_masks, 8);
            __m512i const carry_in = _mm512_or_si512(equality, vertical_negative);
            // Xh = (((Eq & VP) + VP) ^ VP) | Eq; the trailing `(sum ^ VP) | Eq` folds into one VPTERNLOGQ (0xBE).
            __m512i const sum = _mm512_add_epi64(_mm512_and_si512(equality, vertical_positive), vertical_positive);
            __m512i const diagonal = _mm512_ternarylogic_epi64(sum, vertical_positive, equality, ternlog_xor_or_k);
            // Ph = Mv | ~(Xh | VP) is `A | ~(B | C)` -> VPTERNLOGQ(0xF1).
            __m512i horizontal_positive = _mm512_ternarylogic_epi64(vertical_negative, diagonal, vertical_positive,
                                                                    ternlog_or_nor_k);
            __m512i horizontal_negative = _mm512_and_si512(vertical_positive, diagonal);
            score = _mm512_mask_add_epi64(score, active & _mm512_test_epi64_mask(horizontal_positive, top_mask), score,
                                          one);
            score = _mm512_mask_sub_epi64(score, active & _mm512_test_epi64_mask(horizontal_negative, top_mask), score,
                                          one);
            horizontal_positive = _mm512_or_si512(_mm512_slli_epi64(horizontal_positive, 1), one);
            horizontal_negative = _mm512_slli_epi64(horizontal_negative, 1);
            // Pv' = Mh | ~(Xv | Ph), the same `A | ~(B | C)` shape -> VPTERNLOGQ(0xF1).
            __m512i const next_positive = _mm512_ternarylogic_epi64(horizontal_negative, carry_in, horizontal_positive,
                                                                    ternlog_or_nor_k);
            __m512i const next_negative = _mm512_and_si512(horizontal_positive, carry_in);
            vertical_positive = _mm512_mask_blend_epi64(active, vertical_positive, next_positive);
            vertical_negative = _mm512_mask_blend_epi64(active, vertical_negative, next_negative);
        }

        alignas(64) u64_t final_scores[lanes_k];
        _mm512_store_si512(final_scores, score);
        for (index_t lane = 0; lane != candidates_active; ++lane) results[positions[lane]] = (size_t)final_scores[lane];
        return status_t::success_k;
    }

    /**
     *  @brief `8 / lanes_per_pair_` distances in lockstep, each pair spanning `lanes_per_pair_` consecutive lanes
     *      as one `lanes_per_pair_ * 64`-bit Myers integer. Only the carry of `(Eq & VP) + VP` and the `<< 1` ripple
     *      cross a pair's lanes; both are masked at pair boundaries so the pairs stay independent. `lanes_per_pair_`
     *      = 2/4/8 covers shorter sides <= 128/256/512 (4/2/1 pairs per ZMM).
     */
    template <index_t lanes_per_pair_, typename results_writer_>
    status_t lockstep_pairs_(span<char_t const> const *shorters, span<char_t const> const *longers,
                             size_t const *positions, index_t pairs_active, results_writer_ &results,
                             scratch_space_t scratch_space) const noexcept {

        constexpr __mmask8 pair_starts = pair_starts_mask_<lanes_per_pair_>();
        constexpr __mmask8 receivers = (__mmask8)(~pair_starts & 0xFFu); // ? Lanes that take an incoming carry/bit.

        size_t max_longer = 0;
        for (index_t pair = 0; pair != pairs_active; ++pair)
            max_longer = sz_max_of_two(max_longer, longers[pair].size());
        if (scratch_space.size() < match_masks_bytes_k + max_longer * lanes_k) return status_t::bad_alloc_k;

        u64_t *const match_masks = reinterpret_cast<u64_t *>(scratch_space.data()); // ? Indexed `lane * 256 + symbol`.
        u8_t *const transposed_text = reinterpret_cast<u8_t *>(scratch_space.data() + match_masks_bytes_k);
        alignas(64) u64_t vertical_positives[lanes_k] = {0}, top_bits[lanes_k] = {0};
        alignas(64) u64_t shorter_lengths[lanes_k] = {0}, longer_lengths[lanes_k] = {0};
        for (size_t position = 0; position != max_longer * lanes_k; ++position) transposed_text[position] = 0;

        for (index_t pair = 0; pair != pairs_active; ++pair) {
            index_t const shorter_length = (index_t)shorters[pair].size();
            size_t const longer_length = longers[pair].size();
            char_t const *const shorter = shorters[pair].data();
            char_t const *const longer = longers[pair].data();
            for (index_t word = 0; word != lanes_per_pair_; ++word) {
                index_t const lane = pair * lanes_per_pair_ + word;
                // Zero this lane's `Peq` entries for every character its text may read (see the scalar Myers).
                for (index_t position = 0; position != shorter_length; ++position)
                    match_masks[lane * 256 + (u8_t)shorter[position]] = 0;
                for (size_t position = 0; position != longer_length; ++position)
                    match_masks[lane * 256 + (u8_t)longer[position]] = 0;
                // VP = the low `shorter_length` bits, spread across the pair's words.
                index_t const word_low = word * 64;
                if (shorter_length >= word_low + 64) vertical_positives[lane] = ~(u64_t)0;
                else if (shorter_length <= word_low) vertical_positives[lane] = 0;
                else vertical_positives[lane] = ((u64_t)1 << (shorter_length - word_low)) - 1;
                shorter_lengths[lane] = shorter_length;
                longer_lengths[lane] = longer_length;
            }
            for (index_t position = 0; position != shorter_length; ++position)
                match_masks[(pair * lanes_per_pair_ + (position >> 6)) * 256 + (u8_t)shorter[position]] |=
                    (u64_t)1 << (position & 63);
            index_t const top_word = (shorter_length - 1) >> 6, top_bit = (shorter_length - 1) & 63;
            top_bits[pair * lanes_per_pair_ + top_word] = (u64_t)1 << top_bit;
            for (size_t position = 0; position != longer_length; ++position)
                for (index_t word = 0; word != lanes_per_pair_; ++word)
                    transposed_text[position * lanes_k + pair * lanes_per_pair_ + word] = (u8_t)longer[position];
        }

        __m512i const lane_offsets = _mm512_set_epi64(7 * 256, 6 * 256, 5 * 256, 4 * 256, 3 * 256, 2 * 256, 1 * 256, 0);
        __m512i const one = _mm512_set1_epi64(1);
        __m512i const shift_up = _mm512_set_epi64(6, 5, 4, 3, 2, 1, 0, 0); // ? Lane i receives lane (i-1)'s top bit.
        __m512i const boundary_one = _mm512_maskz_mov_epi64(pair_starts, one);
        __m512i const top_mask = _mm512_load_si512(top_bits), longer_vec = _mm512_load_si512(longer_lengths);
        __m512i vertical_positive = _mm512_load_si512(vertical_positives);
        __m512i vertical_negative = _mm512_setzero_si512();
        __m512i score = _mm512_load_si512(shorter_lengths);

        for (size_t position = 0; position != max_longer; ++position) {
            __mmask8 const active = _mm512_cmpgt_epi64_mask(longer_vec, _mm512_set1_epi64((long long)position));
            __m512i const symbols = _mm512_cvtepu8_epi64(
                _mm_loadl_epi64((__m128i const *)(transposed_text + position * lanes_k)));
            __m512i const equality = _mm512_i64gather_epi64(_mm512_add_epi64(lane_offsets, symbols), match_masks, 8);
            __m512i const carry_in = _mm512_or_si512(equality, vertical_negative);
            // (Eq & VP) + VP, rippling the carry across each pair's lanes (blocked at pair starts).
            __m512i const addend = _mm512_and_si512(equality, vertical_positive);
            __m512i sum = _mm512_add_epi64(addend, vertical_positive);
            __mmask8 carry = _mm512_cmplt_epu64_mask(sum, addend);
            for (index_t iteration = 1; iteration != lanes_per_pair_; ++iteration) {
                __mmask8 const into_next = (__mmask8)((carry << 1) & receivers);
                if (!into_next) break;
                __m512i const advanced = _mm512_add_epi64(sum, _mm512_maskz_mov_epi64(into_next, one));
                carry = _mm512_cmplt_epu64_mask(advanced, sum);
                sum = advanced;
            }
            // Xh = (sum ^ VP) | Eq -> VPTERNLOGQ(0xBE); Ph = Mv | ~(Xh | VP) is `A | ~(B | C)` -> VPTERNLOGQ(0xF1).
            __m512i const diagonal = _mm512_ternarylogic_epi64(sum, vertical_positive, equality, ternlog_xor_or_k);
            __m512i horizontal_positive = _mm512_ternarylogic_epi64(vertical_negative, diagonal, vertical_positive,
                                                                    ternlog_or_nor_k);
            __m512i horizontal_negative = _mm512_and_si512(vertical_positive, diagonal);
            score = _mm512_mask_add_epi64(score, active & _mm512_test_epi64_mask(horizontal_positive, top_mask), score,
                                          one);
            score = _mm512_mask_sub_epi64(score, active & _mm512_test_epi64_mask(horizontal_negative, top_mask), score,
                                          one);
            // (Ph | Mh) << 1 across each pair's lanes; Ph also takes a +1 at each pair start.
            __m512i const positive_carry = _mm512_maskz_mov_epi64(
                receivers, _mm512_permutexvar_epi64(shift_up, _mm512_srli_epi64(horizontal_positive, 63)));
            __m512i const negative_carry = _mm512_maskz_mov_epi64(
                receivers, _mm512_permutexvar_epi64(shift_up, _mm512_srli_epi64(horizontal_negative, 63)));
            horizontal_positive = _mm512_or_si512(
                _mm512_or_si512(_mm512_slli_epi64(horizontal_positive, 1), positive_carry), boundary_one);
            horizontal_negative = _mm512_or_si512(_mm512_slli_epi64(horizontal_negative, 1), negative_carry);
            // Pv' = Mh | ~(Xv | Ph), the same `A | ~(B | C)` shape -> VPTERNLOGQ(0xF1).
            __m512i const next_positive = _mm512_ternarylogic_epi64(horizontal_negative, carry_in, horizontal_positive,
                                                                    ternlog_or_nor_k);
            __m512i const next_negative = _mm512_and_si512(horizontal_positive, carry_in);
            vertical_positive = _mm512_mask_blend_epi64(active, vertical_positive, next_positive);
            vertical_negative = _mm512_mask_blend_epi64(active, vertical_negative, next_negative);
        }

        alignas(64) u64_t final_scores[lanes_k];
        _mm512_store_si512(final_scores, score);
        for (index_t pair = 0; pair != pairs_active; ++pair) {
            index_t const top_word = ((index_t)shorters[pair].size() - 1) >> 6;
            results[positions[pair]] = (size_t)final_scores[pair * lanes_per_pair_ + top_word];
        }
        return status_t::success_k;
    }

    /** @brief Four distances in lockstep, each pair a 128-bit Myers integer over two lanes (shorter side <= 128). */
    template <typename results_writer_>
    status_t distances_4x128_(span<char_t const> const *shorters, span<char_t const> const *longers,
                              size_t const *positions, index_t pairs_active, results_writer_ &results,
                              scratch_space_t scratch_space) const noexcept {
        return lockstep_pairs_<2>(shorters, longers, positions, pairs_active, results, scratch_space);
    }

    /** @brief Two distances in lockstep, each pair a 256-bit Myers integer over four lanes (shorter side <= 256). */
    template <typename results_writer_>
    status_t distances_2x256_(span<char_t const> const *shorters, span<char_t const> const *longers,
                              size_t const *positions, index_t pairs_active, results_writer_ &results,
                              scratch_space_t scratch_space) const noexcept {
        return lockstep_pairs_<4>(shorters, longers, positions, pairs_active, results, scratch_space);
    }

    /** @brief One distance as a single 512-bit Myers integer over all eight lanes (shorter side <= 512). */
    template <typename results_writer_>
    status_t distances_1x512_(span<char_t const> const *shorters, span<char_t const> const *longers,
                              size_t const *positions, index_t pairs_active, results_writer_ &results,
                              scratch_space_t scratch_space) const noexcept {
        return lockstep_pairs_<8>(shorters, longers, positions, pairs_active, results, scratch_space);
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
 *  @brief Batched byte-level Levenshtein distances on Ice Lake, using the @b AVX-512 lockstep Myers family for
 *      unit-cost pairs (8/4/2/1 distances at once for shorter sides <= 64/128/256/512) and the anti-diagonal DP
 *      for everything else (non-unit costs, or shorter side > 512). Pairs are grouped by their shorter-side tier.
 */
template <typename allocator_type_, sz_capability_t capability_>
struct levenshtein_distances<linear_gap_costs_t, allocator_type_, capability_,
                             std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>> {

    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    using scoring_t = levenshtein_distance<char, gap_costs_t, capability_k>; // ? Per-pair DP fallback.
    using myers_t = levenshtein_distance_myers<char, capability_k>;          // ? AVX-512 lockstep Myers.
    using scratch_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;

    uniform_substitution_costs_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    safe_vector<std::byte, scratch_allocator_t> score_scratch_ {alloc_}; // grow-only, reused; partitioned per worker

    levenshtein_distances(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    levenshtein_distances(uniform_substitution_costs_t subs, linear_gap_costs_t gaps,
                          allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    bool is_unit_cost_() const noexcept {
        return substituter_.match == 0 && substituter_.mismatch == 1 && gap_costs_.open_or_extend == 1;
    }

    /**
     *  @brief Worst-case scratch for a single cell over the whole input, in O(Q+C): the Myers `Peq` + transposed-text
     *      buffer for the longest string, or the anti-diagonal DP fallback for the longest query × longest candidate
     *      (a real cell in both the full and symmetric grids, so this is a safe upper bound for every slice).
     */
    template <typename queries_type_, typename candidates_type_>
    size_t worst_cell_scratch_(queries_type_ const &queries, candidates_type_ const &candidates,
                               cpu_specs_t const &specs) const noexcept {
        size_t longest_query = 0, longest_query_index = 0, longest_candidate = 0, longest_candidate_index = 0;
        for (size_t index = 0; index < queries.size(); ++index)
            if (to_view(queries[index]).size() > longest_query)
                longest_query = to_view(queries[index]).size(), longest_query_index = index;
        for (size_t index = 0; index < candidates.size(); ++index)
            if (to_view(candidates[index]).size() > longest_candidate)
                longest_candidate = to_view(candidates[index]).size(), longest_candidate_index = index;
        size_t const max_longer = sz_max_of_two(longest_query, longest_candidate);
        size_t const myers_scratch = myers_t::match_masks_bytes_k + max_longer * (size_t)myers_t::lanes_k;
        size_t dp_scratch = 0;
        if (queries.size() && candidates.size() && sz_min_of_two(longest_query, longest_candidate) > 512) {
            scoring_t dp {substituter_, gap_costs_};
            dp_scratch = dp.scratch_space_needed(to_view(queries[longest_query_index]),
                                                 to_view(candidates[longest_candidate_index]), specs);
        }
        return sz_max_of_two(myers_scratch, dp_scratch);
    }

#pragma region - Cross-Product Cell Addressing

    /**
     *  @brief A destination for one scored cell: the primary matrix slot plus an optional mirror slot.
     *      The Myers kernels assign `writer[group_local_index] = distance`, so the writer holds one of
     *      these per active lane and fans the score out to both slots on assignment.
     */
    template <typename value_type_>
    struct cross_cell_destination_ {
        value_type_ *primary = nullptr;
        value_type_ *mirror = nullptr;
    };

    /**
     *  @brief An indexable adapter handed to the Myers kernels and the DP fallback so they can stay
     *      grouping-agnostic: a lane's group-local index selects its destination, and assigning a score
     *      writes the primary cell and, for symmetric self-similarity, the mirrored cell too.
     */
    template <typename value_type_>
    struct cross_cell_writer_ {
        cross_cell_destination_<value_type_> const *destinations = nullptr;

        struct cell_proxy_ {
            cross_cell_destination_<value_type_> destination;
            cell_proxy_ &operator=(size_t value) noexcept {
                *destination.primary = static_cast<value_type_>(value);
                if (destination.mirror) *destination.mirror = static_cast<value_type_>(value);
                return *this;
            }
        };

        cell_proxy_ operator[](size_t group_local_index) const noexcept {
            return cell_proxy_ {destinations[group_local_index]};
        }
    };

    /** @brief The number of live cells: the full rectangle, or the lower triangle (incl. diagonal) when symmetric. */
    static size_t live_cells_count_(size_t queries_count, size_t candidates_count,
                                    cross_similarities_t cross_kind) noexcept {
        if (cross_kind == cross_similarities_t::symmetric_k) return queries_count * (queries_count + 1) / 2;
        return queries_count * candidates_count;
    }

    /**
     *  @brief Decodes a flat live-cell index into its `(query_index, candidate_index)` grid coordinates.
     *      For the full rectangle the layout is row-major; for the symmetric case it walks the lower triangle
     *      (including the diagonal) row by row, so consecutive cells stay on the same query row when possible.
     */
    static void cell_to_indices_(size_t cell_index, size_t candidates_count, cross_similarities_t cross_kind,
                                 size_t &query_index, size_t &candidate_index) noexcept {
        if (cross_kind == cross_similarities_t::symmetric_k) {
            // Triangular inverse: find the row whose prefix `row * (row + 1) / 2 <= cell_index`.
            size_t row = 0;
            while ((row + 1) * (row + 2) / 2 <= cell_index) ++row;
            query_index = row;
            candidate_index = cell_index - row * (row + 1) / 2;
        }
        else {
            query_index = cell_index / candidates_count;
            candidate_index = cell_index % candidates_count;
        }
    }

#pragma endregion - Cross-Product Cell Addressing

#pragma region - Cross-Product Scoring

    /**
     *  @brief Scores the live cells `[cell_begin, cell_end)` of the cross-product with the lockstep Myers family,
     *      falling back to the anti-diagonal DP for the long tail. Each cell scores `dist(query, candidate)` and
     *      writes it into the strided @p results matrix (plus the mirror slot for symmetric self-similarity).
     */
    template <typename queries_type_, typename candidates_type_, typename results_type_>
    [[gnu::noinline]] status_t score_range_(queries_type_ const &queries, candidates_type_ const &candidates,
                                            results_type_ &&results, cross_similarities_t cross_kind, size_t cell_begin,
                                            size_t cell_end, scratch_space_t scratch, cpu_specs_t const &specs) noexcept {

        using value_t = remove_cvref<decltype(results.data[0])>;
        size_t const candidates_count = candidates.size();

        // `scratch` is provided by the caller, already sized to the worst single cell (`worst_cell_scratch_`).
        scoring_t dp {substituter_, gap_costs_};

        // Maps a query row and candidate column to their primary (and mirrored) destination slots.
        auto const destination_for = [&](size_t query_index, size_t candidate_index) noexcept {
            cross_cell_destination_<value_t> destination;
            destination.primary = results.data + query_index * results.row_stride + candidate_index;
            if (cross_kind == cross_similarities_t::symmetric_k && candidate_index != query_index)
                destination.mirror = results.data + candidate_index * results.row_stride + query_index;
            return destination;
        };

        myers_t myers;
        cross_cell_writer_<value_t> writer;
        dummy_executor_t dummy;
        for (size_t cell_index = cell_begin; cell_index != cell_end;) {
            size_t query_index = 0, candidate_index = 0;
            cell_to_indices_(cell_index, candidates_count, cross_kind, query_index, candidate_index);
            auto const query = to_view(queries[query_index]);
            auto const candidate = to_view(candidates[candidate_index]);
            size_t const shorter = sz_min_of_two(query.size(), candidate.size());

            if (shorter == 0) {
                cross_cell_destination_<value_t> const destination = destination_for(query_index, candidate_index);
                cross_cell_writer_<value_t> {&destination}[0] = sz_max_of_two(query.size(), candidate.size());
                ++cell_index;
                continue;
            }
            if (shorter > 512) { // Anti-diagonal DP for the long tail.
                size_t result_score = 0;
                if (status_t status = dp(query, candidate, result_score, scratch, dummy, specs);
                    status != status_t::success_k)
                    return status;
                cross_cell_destination_<value_t> const destination = destination_for(query_index, candidate_index);
                cross_cell_writer_<value_t> {&destination}[0] = result_score;
                ++cell_index;
                continue;
            }

            // Pick the tier from this first cell, seed the group with it (it is non-empty and fits its tier), then
            // extend with consecutive live cells that also fit. Seeding first guarantees forward progress.
            size_t const tier_limit = shorter <= 64 ? 64 : shorter <= 128 ? 128 : shorter <= 256 ? 256 : 512;
            index_t const lanes_per_pair = shorter <= 64 ? 1 : shorter <= 128 ? 2 : shorter <= 256 ? 4 : 8;
            index_t const pairs_capacity = myers_t::lanes_k / lanes_per_pair;
            // The kernel is grouping-agnostic: it takes the per-lane spans plus a group-local position that selects
            // each lane's destination cell, so a future grouping policy can hand it any set of cells. This greedy
            // pass still seeds with the current cell and extends over consecutive live cells.
            span<char const> shorters[myers_t::lanes_k], longers[myers_t::lanes_k];
            span<char const> candidate_views[myers_t::lanes_k]; // ? Candidate's own view, for the shared-query kernel.
            size_t positions[myers_t::lanes_k];
            cross_cell_destination_<value_t> destinations[myers_t::lanes_k];
            size_t const seed_query_index = query_index;
            bool const seed_query_shorter = query.size() <= candidate.size();
            shorters[0] = seed_query_shorter ? query : candidate;
            longers[0] = seed_query_shorter ? candidate : query;
            candidate_views[0] = candidate;
            positions[0] = 0;
            destinations[0] = destination_for(query_index, candidate_index);
            index_t group = 1;
            ++cell_index;
            // Keep every group query-major: one shared query against a run of its candidates, so the case-1 tier can
            // dispatch the build-once `distances_8x64_shared_query_` kernel that pays the `Peq` build once per query.
            for (; cell_index != cell_end && group != pairs_capacity; ++cell_index, ++group) {
                size_t next_query_index = 0, next_candidate_index = 0;
                cell_to_indices_(cell_index, candidates_count, cross_kind, next_query_index, next_candidate_index);
                if (next_query_index != seed_query_index) break;
                auto const next_query = to_view(queries[next_query_index]);
                auto const next_candidate = to_view(candidates[next_candidate_index]);
                size_t const next_shorter = sz_min_of_two(next_query.size(), next_candidate.size());
                if (next_shorter == 0 || next_shorter > tier_limit) break;
                bool const next_query_shorter = next_query.size() <= next_candidate.size();
                shorters[group] = next_query_shorter ? next_query : next_candidate;
                longers[group] = next_query_shorter ? next_candidate : next_query;
                candidate_views[group] = next_candidate;
                positions[group] = group;
                destinations[group] = destination_for(next_query_index, next_candidate_index);
            }

            writer.destinations = destinations;
            status_t status = status_t::success_k;
            switch (lanes_per_pair) {
            case 1: {
                // Query-major group: every lane shares `queries[seed_query_index]`. When that query fits a single
                // 64-bit Myers word, build its `Peq` table once via the shared-query kernel; otherwise (query is the
                // longer side and exceeds 64 while a candidate is short) fall back to the per-pair build.
                auto const query_view = to_view(queries[seed_query_index]);
                if (query_view.size() <= 64)
                    status = myers.distances_8x64_shared_query_(query_view, candidate_views, positions, group, writer,
                                                                scratch);
                else
                    status = myers.distances_8x64_(shorters, longers, positions, group, writer, scratch);
                break;
            }
            case 2: status = myers.distances_4x128_(shorters, longers, positions, group, writer, scratch); break;
            case 4: status = myers.distances_2x256_(shorters, longers, positions, group, writer, scratch); break;
            default: status = myers.distances_1x512_(shorters, longers, positions, group, writer, scratch); break;
            }
            if (status != status_t::success_k) return status;
        }
        return status_t::success_k;
    }

    /** @brief Scores the cross-product in parallel: each worker takes a contiguous slice of the live-cell range. */
    template <typename queries_type_, typename candidates_type_, typename results_type_, typename executor_type_>
    [[gnu::noinline]] status_t score_parallel_(queries_type_ const &queries, candidates_type_ const &candidates,
                                               results_type_ &&results, cross_similarities_t cross_kind,
                                               executor_type_ &&executor, cpu_specs_t const &specs) noexcept {
        size_t const cells_count = live_cells_count_(queries.size(), candidates.size(), cross_kind);
        // One hoisted buffer carved into a per-worker slice; `for_slices` invokes the body at most `threads_count`
        // times, and the atomic counter hands each invocation a disjoint slice (no aliasing, no per-worker alloc).
        size_t const worker_scratch = worst_cell_scratch_(queries, candidates, specs);
        size_t const workers = sz_max_of_two(sz_min_of_two(executor.threads_count(), cells_count), (size_t)1);
        if (status_t status = score_scratch_.try_resize(worker_scratch * workers); status != status_t::success_k)
            return status;
        std::atomic<size_t> next_worker {0};
        std::atomic<status_t> error {status_t::success_k};
        executor.for_slices(cells_count, [&](size_t cell_begin, size_t length) noexcept {
            if (length == 0) return; // empty slice: no work, and it must not consume a scratch partition
            size_t const worker = next_worker.fetch_add(1, std::memory_order_relaxed);
            scratch_space_t slice = scratch_space_t(score_scratch_).subspan(worker * worker_scratch, worker_scratch);
            status_t status =
                score_range_(queries, candidates, results, cross_kind, cell_begin, cell_begin + length, slice, specs);
            if (status != status_t::success_k) error.store(status);
        });
        return error.load();
    }

#pragma endregion - Cross-Product Scoring

#pragma region - Public Cross-Product Overloads

    // Concrete `strided_rows<value_type_>` results parameter disambiguates the two-set and symmetric overloads by
    // type alone (no concepts needed; `SZ_HAS_CONCEPTS_` is off repo-wide for the GCC concepts bug).
    template <typename queries_type_, typename candidates_type_, typename value_type_>
    status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                        strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        if (!is_unit_cost_())
            return cross_sequentially_<size_t>(scoring_t {substituter_, gap_costs_}, queries, candidates, results,
                                               cross_similarities_t::all_pairs_k, score_scratch_, specs);
        if (status_t status = score_scratch_.try_resize(worst_cell_scratch_(queries, candidates, specs));
            status != status_t::success_k)
            return status;
        return score_range_(queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
                            live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
                            scratch_space_t(score_scratch_), specs);
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_, typename executor_type_>
    status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                        strided_rows<value_type_> results, executor_type_ &&executor,
                        cpu_specs_t const &specs = {}) noexcept {
        if (!is_unit_cost_())
            return cross_in_parallel_<size_t>(scoring_t {substituter_, gap_costs_}, queries, candidates, results,
                                              cross_similarities_t::all_pairs_k, score_scratch_,
                                              std::forward<executor_type_>(executor), specs);
        return score_parallel_(queries, candidates, results, cross_similarities_t::all_pairs_k,
                               std::forward<executor_type_>(executor), specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                        cpu_specs_t const &specs = {}) noexcept {
        if (!is_unit_cost_())
            return cross_sequentially_<size_t>(scoring_t {substituter_, gap_costs_}, sequences, sequences, results,
                                               cross_similarities_t::symmetric_k, score_scratch_, specs);
        if (status_t status = score_scratch_.try_resize(worst_cell_scratch_(sequences, sequences, specs));
            status != status_t::success_k)
            return status;
        return score_range_(sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
                            live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
                            scratch_space_t(score_scratch_), specs);
    }

    template <typename sequences_type_, typename value_type_, typename executor_type_>
    status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                        executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        if (!is_unit_cost_())
            return cross_in_parallel_<size_t>(scoring_t {substituter_, gap_costs_}, sequences, sequences, results,
                                              cross_similarities_t::symmetric_k, score_scratch_,
                                              std::forward<executor_type_>(executor), specs);
        return score_parallel_(sequences, sequences, results, cross_similarities_t::symmetric_k,
                               std::forward<executor_type_>(executor), specs);
    }

#pragma endregion - Public Cross-Product Overloads
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
            rune_length = sz_rune_parse_unchecked(first.data() + progress_utf8, first_data_utf32 + first_length_utf32);
            if (rune_length == sz_utf8_invalid_k) return status_t::invalid_utf8_k;
        }
        for (size_t progress_utf8 = 0; progress_utf8 < second.size();
             progress_utf8 += rune_length, ++second_length_utf32) {
            rune_length =
                sz_rune_parse_unchecked(second.data() + progress_utf8, second_data_utf32 + second_length_utf32);
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

#pragma region - Inter-Sequence Candidate Lanes

/**
 *  @brief Inter-sequence Ice Lake walker: one query against up to 64 candidates packed one-per-lane.
 *
 *  Computes the @b global unit-cost Levenshtein distance of a single shared query against a transposed
 *  `candidate_lanes_block_t` of up to 64 candidates. Each `__m512i` holds 64 `u8` cells - lane @p lane_index
 *  carries that candidate's running Dynamic Programming column. The query characters index the rows; for every
 *  row the candidate column is broadcast-compared against the query character and the SWIPE recurrence
 *  `cell = min(substitution, min(deletion, insertion))` advances all 64 lanes in lockstep.
 *
 *  @note The cells are `u8`, so distances saturate at 255: this kernel is only valid when the query and every
 *      candidate are at most 255 characters long. Enforcing that bound is the caller's dispatch contract; the
 *      kernel performs no runtime length check.
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<char, u8_t, uniform_substitution_costs_t, linear_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_icelake_k, 64, void> {

    using char_t = char;
    using score_t = u8_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_icelake_k;
    static constexpr size_t candidate_lanes_k = 64;

    // The `u8` lane recurrence hardcodes `_mm512_min_epu8`; maximization would need a different blend.
    static_assert(objective_ == sz_minimize_distance_k,
                  "The 8-bit candidate-lane kernel only implements distance minimization (Levenshtein).");

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch holds two score rows of `longest_candidate + 1` lane-vectors (64 `u8` cells each). */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const row_bytes = candidate_lanes_k * (longest_candidate + 1);
        scratch_amount_t amount {specs.cache_line_width};
        amount += row_bytes; // previous row
        amount += row_bytes; // current row
        return amount;
    }

    /**
     *  @param[in] query The shared query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 64 candidates (see `candidate_lanes_block_t`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block_t<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        // Two row buffers carved from the byte span; each lane-vector lives at `row + column * 64`.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;

        __m512i const one_vec = _mm512_set1_epi8(1);

        // Row 0: the empty query prefix against every candidate prefix is a run of `column` gaps, identical
        // across lanes (later masked per-lane at latch time by reading each lane's own final column).
        for (size_t column = 0; column <= longest_candidate; ++column)
            _mm512_storeu_si512(previous_row + column * candidate_lanes_k,
                                _mm512_set1_epi8(static_cast<char>(static_cast<u8_t>(column))));

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            __m512i const query_char_vec = _mm512_set1_epi8(query[query_position - 1]);
            _mm512_storeu_si512(current_row, _mm512_set1_epi8(static_cast<char>(static_cast<u8_t>(query_position))));
            for (size_t column = 1; column <= longest_candidate; ++column) {
                __m512i const candidate_chars_vec =
                    _mm512_loadu_si512(candidates.position(column - 1));
                __m512i const diagonal_vec = _mm512_loadu_si512(previous_row + (column - 1) * candidate_lanes_k);
                __m512i const deletion_source_vec = _mm512_loadu_si512(previous_row + column * candidate_lanes_k);
                __m512i const insertion_source_vec = _mm512_loadu_si512(current_row + (column - 1) * candidate_lanes_k);

                __mmask64 const mismatch_mask = _mm512_cmpneq_epi8_mask(query_char_vec, candidate_chars_vec);
                __m512i const cost_if_substitution_vec =
                    _mm512_mask_add_epi8(diagonal_vec, mismatch_mask, diagonal_vec, one_vec);
                __m512i const cost_if_deletion_vec = _mm512_add_epi8(deletion_source_vec, one_vec);
                __m512i const cost_if_insertion_vec = _mm512_add_epi8(insertion_source_vec, one_vec);
                __m512i const cell_score_vec = _mm512_min_epu8(
                    cost_if_substitution_vec, _mm512_min_epu8(cost_if_deletion_vec, cost_if_insertion_vec));
                _mm512_storeu_si512(current_row + column * candidate_lanes_k, cell_score_vec);
            }
            trivial_swap(previous_row, current_row);
        }

        // Latch each live lane's result from its own final column; ragged lengths mean different columns per lane.
        for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index) {
            size_t const candidate_length = candidates.lengths[lane_index];
            result_lanes[lane_index] = previous_row[candidate_length * candidate_lanes_k + lane_index];
        }
        return status_t::success_k;
    }
};

/**
 *  @brief Inter-sequence Ice Lake walker: one query against up to 32 candidates packed one-per-lane, 16-bit cells.
 *
 *  Structural twin of the 64-lane `u8` walker, widened to `u16` cells so each `__m512i` holds 32 lanes. The
 *  candidate characters remain `char`/`u8`: 32 of them load as a 256-bit `__m256i` and compare against the
 *  broadcast query character with `_mm256_cmpneq_epi8_mask`, yielding a `__mmask32` that drives the 16-bit
 *  substitution add. The SWIPE recurrence `cell = min(substitution, min(deletion, insertion))` advances all 32
 *  lanes in lockstep over `_mm512_min_epu16`.
 *
 *  @note The cells are `u16`, so distances saturate at 65535: this kernel is only valid when the query and every
 *      candidate are at most 65535 characters long. Enforcing that bound is the caller's dispatch contract; the
 *      kernel performs no runtime length check.
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<char, u16_t, uniform_substitution_costs_t, linear_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_icelake_k, 32, void> {

    using char_t = char;
    using score_t = u16_t;
    using substituter_t = uniform_substitution_costs_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_icelake_k;
    static constexpr size_t candidate_lanes_k = 32;

    // The `u16` lane recurrence hardcodes `_mm512_min_epu16`; maximization would need a different blend.
    static_assert(objective_ == sz_minimize_distance_k,
                  "The 16-bit candidate-lane kernel only implements distance minimization (Levenshtein).");

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch holds two score rows of `longest_candidate + 1` lane-vectors (32 `u16` cells each). */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += row_bytes; // previous row
        amount += row_bytes; // current row
        return amount;
    }

    /**
     *  @param[in] query The shared query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 32 candidates (see `candidate_lanes_block_t`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block_t<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        // Two row buffers carved from the byte span; each lane-vector lives at `row + column * 32`.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;

        __m512i const one_vec = _mm512_set1_epi16(1);

        // Row 0: the empty query prefix against every candidate prefix is a run of `column` gaps, identical
        // across lanes (later masked per-lane at latch time by reading each lane's own final column).
        for (size_t column = 0; column <= longest_candidate; ++column)
            _mm512_storeu_si512(previous_row + column * candidate_lanes_k,
                                _mm512_set1_epi16(static_cast<short>(static_cast<u16_t>(column))));

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            __m256i const query_char_vec = _mm256_set1_epi8(query[query_position - 1]);
            _mm512_storeu_si512(current_row,
                                _mm512_set1_epi16(static_cast<short>(static_cast<u16_t>(query_position))));
            for (size_t column = 1; column <= longest_candidate; ++column) {
                __m256i const candidate_chars_vec =
                    _mm256_loadu_si256(reinterpret_cast<__m256i const *>(candidates.position(column - 1)));
                __m512i const diagonal_vec = _mm512_loadu_si512(previous_row + (column - 1) * candidate_lanes_k);
                __m512i const deletion_source_vec = _mm512_loadu_si512(previous_row + column * candidate_lanes_k);
                __m512i const insertion_source_vec = _mm512_loadu_si512(current_row + (column - 1) * candidate_lanes_k);

                __mmask32 const mismatch_mask = _mm256_cmpneq_epi8_mask(query_char_vec, candidate_chars_vec);
                __m512i const cost_if_substitution_vec =
                    _mm512_mask_add_epi16(diagonal_vec, mismatch_mask, diagonal_vec, one_vec);
                __m512i const cost_if_deletion_vec = _mm512_add_epi16(deletion_source_vec, one_vec);
                __m512i const cost_if_insertion_vec = _mm512_add_epi16(insertion_source_vec, one_vec);
                __m512i const cell_score_vec = _mm512_min_epu16(
                    cost_if_substitution_vec, _mm512_min_epu16(cost_if_deletion_vec, cost_if_insertion_vec));
                _mm512_storeu_si512(current_row + column * candidate_lanes_k, cell_score_vec);
            }
            trivial_swap(previous_row, current_row);
        }

        // Latch each live lane's result from its own final column; ragged lengths mean different columns per lane.
        for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index) {
            size_t const candidate_length = candidates.lengths[lane_index];
            result_lanes[lane_index] = previous_row[candidate_length * candidate_lanes_k + lane_index];
        }
        return status_t::success_k;
    }
};

/**
 *  @brief Inter-sequence Ice Lake walker: one query against up to 32 candidates packed one-per-lane, weighted
 *         Needleman-Wunsch with 32-class substitution costs and a linear gap, 16-bit signed cells.
 *
 *  Structural twin of the 32-lane uniform `u16` walker, but the unit `cmpneq -> +1` substitution term is replaced
 *  by a two-level class lookup against the resident (32 x 32) `class_substitution_costs` matrix, and the objective
 *  is @b maximization (Needleman-Wunsch), so the recurrence is `cell = max(diag + substitution, max(up, left) + gap)`
 *  over signed `i16` cells with a (typically negative) `error_cost_t` gap.
 *
 *  The query character is fixed for an entire Dynamic Programming row, so its class is scalar: per row we load the
 *  single cost row `class_substitution_costs[query_class][0..31]` (32 signed `i8`) into the low half of a `zmm`.
 *  The 32 candidate characters (one per lane) are loop-invariant across rows, so their classes are computed @b once
 *  before the row loop (the 256-entry `byte_to_class` table via the 4x `VPERMB`-blend technique) and cached. Inside
 *  the column loop the cached candidate-class vector indexes the resident cost row with a single `VPERMB`, yielding
 *  32 signed `i8` substitution costs that sign-extend to 32 `i16` lanes.
 *
 *  @note The cells are signed `i16`, so the kernel is only valid while every reachable score stays within
 *      `[-32768, 32767]`. With a cost magnitude `m` and gap magnitude `g`, the worst-case score grows like
 *      `(query_length + candidate_length) * max(m, g)`; enforcing that bound is the caller's dispatch contract,
 *      the kernel performs no runtime range check.
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<char, i16_t, error_costs_32x32_t, linear_gap_costs_t, objective_,
                             sz_similarity_global_k, sz_cap_icelake_k, 32, void> {

    using char_t = char;
    using score_t = i16_t;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = sz_cap_icelake_k;
    static constexpr size_t candidate_lanes_k = 32;

    // The signed `i16` recurrence hardcodes `_mm512_max_epi16`; minimization would need a different blend.
    static_assert(objective_ == sz_maximize_score_k,
                  "The weighted candidate-lane kernel only implements score maximization (Needleman-Wunsch).");

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /**
     *  @brief Scratch holds two score rows of `longest_candidate + 1` lane-vectors (32 `i16` cells each), plus one
     *      cached candidate-class lane-vector (32 `u8`) per candidate position.
     */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const score_row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        size_t const class_bytes = candidate_lanes_k * longest_candidate * sizeof(u8_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += score_row_bytes; // previous row
        amount += score_row_bytes; // current row
        amount += class_bytes;     // cached candidate classes
        return amount;
    }

    /** @brief Maps the 32 candidate bytes of one column to their 32 classes via the 256-entry table. */
    SZ_INLINE __m256i classify32_(__m512i const text_vec, __m512i const (&byte_to_class_vecs)[4],
                                  __m512i const is_third_or_fourth_vec,
                                  __m512i const is_second_or_fourth_vec) const noexcept {
        __m512i const shuffled0 = _mm512_permutexvar_epi8(text_vec, byte_to_class_vecs[0]);
        __m512i const shuffled1 = _mm512_permutexvar_epi8(text_vec, byte_to_class_vecs[1]);
        __m512i const shuffled2 = _mm512_permutexvar_epi8(text_vec, byte_to_class_vecs[2]);
        __m512i const shuffled3 = _mm512_permutexvar_epi8(text_vec, byte_to_class_vecs[3]);
        __mmask64 const is_third_or_fourth = _mm512_test_epi8_mask(text_vec, is_third_or_fourth_vec);
        __mmask64 const is_second_or_fourth = _mm512_test_epi8_mask(text_vec, is_second_or_fourth_vec);
        __m512i const class_vec = _mm512_mask_blend_epi8(
            is_third_or_fourth, _mm512_mask_blend_epi8(is_second_or_fourth, shuffled0, shuffled1),
            _mm512_mask_blend_epi8(is_second_or_fourth, shuffled2, shuffled3));
        return _mm512_castsi512_si256(class_vec);
    }

    /**
     *  @param[in] query The shared query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 32 candidates (see `candidate_lanes_block_t`).
     *  @param[out] result_lanes One score per live lane (`candidates.lanes_count` of them); the caller maps each
     *      lane back to its candidate index for the strided result matrix.
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block_t<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        error_cost_t const gap = gap_costs_.open_or_extend;
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        // Two `i16` score row buffers, then a `u8` candidate-class cache, all carved from the byte span.
        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;
        u8_t *candidate_classes = reinterpret_cast<u8_t *>(current_row + row_stride);

        // Load the 256-entry `byte_to_class` table for the 4x `VPERMB`-blend classifier.
        __m512i byte_to_class_vecs[4];
        byte_to_class_vecs[0] = _mm512_loadu_si512(substituter_.byte_to_class + 64 * 0);
        byte_to_class_vecs[1] = _mm512_loadu_si512(substituter_.byte_to_class + 64 * 1);
        byte_to_class_vecs[2] = _mm512_loadu_si512(substituter_.byte_to_class + 64 * 2);
        byte_to_class_vecs[3] = _mm512_loadu_si512(substituter_.byte_to_class + 64 * 3);
        __m512i const is_third_or_fourth_vec = _mm512_set1_epi8((char)0x80);
        __m512i const is_second_or_fourth_vec = _mm512_set1_epi8((char)0x40);

        // Pre-classify every candidate column once; the 32 candidate bytes per column are loop-invariant in rows.
        for (size_t column = 0; column < longest_candidate; ++column) {
            __m256i const candidate_chars_vec =
                _mm256_loadu_si256(reinterpret_cast<__m256i const *>(candidates.position(column)));
            __m512i const candidate_chars_zvec = _mm512_castsi256_si512(candidate_chars_vec);
            __m256i const candidate_classes_vec =
                classify32_(candidate_chars_zvec, byte_to_class_vecs, is_third_or_fourth_vec, is_second_or_fourth_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(candidate_classes + column * candidate_lanes_k),
                                candidate_classes_vec);
        }

        __m512i const gap_vec = _mm512_set1_epi16(static_cast<short>(gap));

        // Row 0: the empty query prefix against every candidate prefix is a run of `gap * column`, identical across
        // lanes (later masked per-lane at latch time by reading each lane's own final column).
        for (size_t column = 0; column <= longest_candidate; ++column)
            _mm512_storeu_si512(previous_row + column * candidate_lanes_k,
                                _mm512_set1_epi16(static_cast<short>(static_cast<i16_t>(gap * (i16_t)column))));

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            // The query character is fixed for the whole row, so its class and cost row are scalar.
            u8_t const query_class = substituter_.byte_to_class[(u8_t)query[query_position - 1]];
            __m512i const cost_row_vec = _mm512_castsi256_si512(
                _mm256_loadu_si256(reinterpret_cast<__m256i const *>(&substituter_.class_substitution_costs[query_class][0])));

            _mm512_storeu_si512(current_row, _mm512_set1_epi16(static_cast<short>(
                                                 static_cast<i16_t>(gap * (i16_t)query_position))));
            for (size_t column = 1; column <= longest_candidate; ++column) {
                __m256i const candidate_classes_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(candidate_classes + (column - 1) * candidate_lanes_k));
                __m512i const diagonal_vec = _mm512_loadu_si512(previous_row + (column - 1) * candidate_lanes_k);
                __m512i const up_vec = _mm512_loadu_si512(previous_row + column * candidate_lanes_k);
                __m512i const left_vec = _mm512_loadu_si512(current_row + (column - 1) * candidate_lanes_k);

                // Gather the 32 substitution costs of this column from the resident query-class cost row, then
                // sign-extend the low 32 `i8` lanes into 32 `i16` cells.
                __m256i const cost_i8_vec =
                    _mm512_castsi512_si256(_mm512_permutexvar_epi8(_mm512_castsi256_si512(candidate_classes_vec),
                                                                   cost_row_vec));
                __m512i const cost_i16_vec = _mm512_cvtepi8_epi16(cost_i8_vec);

                __m512i const cost_if_substitution_vec = _mm512_add_epi16(diagonal_vec, cost_i16_vec);
                __m512i const cost_if_gap_vec = _mm512_add_epi16(_mm512_max_epi16(up_vec, left_vec), gap_vec);
                __m512i const cell_score_vec = _mm512_max_epi16(cost_if_substitution_vec, cost_if_gap_vec);
                _mm512_storeu_si512(current_row + column * candidate_lanes_k, cell_score_vec);
            }
            trivial_swap(previous_row, current_row);
        }

        // Latch each live lane's result from its own final column; ragged lengths mean different columns per lane.
        for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index) {
            size_t const candidate_length = candidates.lengths[lane_index];
            result_lanes[lane_index] = previous_row[candidate_length * candidate_lanes_k + lane_index];
        }
        return status_t::success_k;
    }
};

/**
 *  @brief Inter-sequence Ice Lake walker: one query against up to 32 candidates packed one-per-lane, weighted
 *         @b Smith-Waterman (local) alignment with 32-class substitution costs and a linear gap, 16-bit signed cells.
 *
 *  Local sibling of the weighted Needleman-Wunsch walker: the recurrence clamps to zero,
 *  `cell = max(0, diag + substitution, max(up, left) + gap)`, the boundary row and column are zero (an alignment may
 *  start anywhere), and the score is the maximum cell over the whole matrix rather than the bottom-right corner. A
 *  per-lane running maximum accumulates that, updated only for columns within each lane's own candidate length so a
 *  shorter candidate's zero-padded tail columns cannot inflate its score.
 *
 *  @note Signed `i16` cells; valid while every reachable score stays within `[0, 32767]` (caller's dispatch contract).
 *  @note Requires Intel Ice Lake generation CPUs or newer.
 */
template <sz_similarity_objective_t objective_>
struct candidate_lane_walker<char, i16_t, error_costs_32x32_t, linear_gap_costs_t, objective_,
                             sz_similarity_local_k, sz_cap_icelake_k, 32, void> {

    using char_t = char;
    using score_t = i16_t;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = sz_cap_icelake_k;
    static constexpr size_t candidate_lanes_k = 32;

    static_assert(objective_ == sz_maximize_score_k,
                  "The weighted local candidate-lane kernel only implements score maximization (Smith-Waterman).");

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};

    candidate_lane_walker() noexcept {}
    candidate_lane_walker(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /** @brief Scratch: two `i16` score rows of `longest_candidate + 1` lane-vectors plus one `u8` candidate-class
     *      cache per candidate position (same layout as the global weighted walker). */
    size_t scratch_space_needed(size_t longest_candidate, cpu_specs_t const &specs) const noexcept {
        size_t const score_row_bytes = candidate_lanes_k * (longest_candidate + 1) * sizeof(score_t);
        size_t const class_bytes = candidate_lanes_k * longest_candidate * sizeof(u8_t);
        scratch_amount_t amount {specs.cache_line_width};
        amount += score_row_bytes;
        amount += score_row_bytes;
        amount += class_bytes;
        return amount;
    }

    /** @brief Maps the 32 candidate bytes of one column to their 32 classes via the 256-entry table. */
    SZ_INLINE __m256i classify32_(__m512i const text_vec, __m512i const (&byte_to_class_vecs)[4],
                                  __m512i const is_third_or_fourth_vec,
                                  __m512i const is_second_or_fourth_vec) const noexcept {
        __m512i const shuffled0 = _mm512_permutexvar_epi8(text_vec, byte_to_class_vecs[0]);
        __m512i const shuffled1 = _mm512_permutexvar_epi8(text_vec, byte_to_class_vecs[1]);
        __m512i const shuffled2 = _mm512_permutexvar_epi8(text_vec, byte_to_class_vecs[2]);
        __m512i const shuffled3 = _mm512_permutexvar_epi8(text_vec, byte_to_class_vecs[3]);
        __mmask64 const is_third_or_fourth = _mm512_test_epi8_mask(text_vec, is_third_or_fourth_vec);
        __mmask64 const is_second_or_fourth = _mm512_test_epi8_mask(text_vec, is_second_or_fourth_vec);
        __m512i const class_vec = _mm512_mask_blend_epi8(
            is_third_or_fourth, _mm512_mask_blend_epi8(is_second_or_fourth, shuffled0, shuffled1),
            _mm512_mask_blend_epi8(is_second_or_fourth, shuffled2, shuffled3));
        return _mm512_castsi512_si256(class_vec);
    }

    /**
     *  @param[in] query The shared query; its length is the number of Dynamic Programming rows.
     *  @param[in] candidates Transposed block of up to 32 candidates (see `candidate_lanes_block_t`).
     *  @param[out] result_lanes One local-alignment score per live lane (`candidates.lanes_count` of them).
     */
    status_t operator()(span<char_t const> query, candidate_lanes_block_t<char_t> candidates, score_t *result_lanes,
                        scratch_space_t scratch_space, cpu_specs_t const &specs) const noexcept {
        sz_unused_(specs);
        error_cost_t const gap = gap_costs_.open_or_extend;
        size_t const query_length = query.size();
        size_t const longest_candidate = candidates.longest_candidate;
        size_t const row_stride = candidate_lanes_k * (longest_candidate + 1);

        score_t *previous_row = reinterpret_cast<score_t *>(scratch_space.data());
        score_t *current_row = previous_row + row_stride;
        u8_t *candidate_classes = reinterpret_cast<u8_t *>(current_row + row_stride);

        __m512i byte_to_class_vecs[4];
        byte_to_class_vecs[0] = _mm512_loadu_si512(substituter_.byte_to_class + 64 * 0);
        byte_to_class_vecs[1] = _mm512_loadu_si512(substituter_.byte_to_class + 64 * 1);
        byte_to_class_vecs[2] = _mm512_loadu_si512(substituter_.byte_to_class + 64 * 2);
        byte_to_class_vecs[3] = _mm512_loadu_si512(substituter_.byte_to_class + 64 * 3);
        __m512i const is_third_or_fourth_vec = _mm512_set1_epi8((char)0x80);
        __m512i const is_second_or_fourth_vec = _mm512_set1_epi8((char)0x40);

        for (size_t column = 0; column < longest_candidate; ++column) {
            __m256i const candidate_chars_vec =
                _mm256_loadu_si256(reinterpret_cast<__m256i const *>(candidates.position(column)));
            __m512i const candidate_chars_zvec = _mm512_castsi256_si512(candidate_chars_vec);
            __m256i const candidate_classes_vec =
                classify32_(candidate_chars_zvec, byte_to_class_vecs, is_third_or_fourth_vec, is_second_or_fourth_vec);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(candidate_classes + column * candidate_lanes_k),
                                candidate_classes_vec);
        }

        __m512i const gap_vec = _mm512_set1_epi16(static_cast<short>(gap));
        __m512i const zero_vec = _mm512_setzero_si512();

        // Per-lane candidate lengths gate the running-maximum so a shorter candidate's padded columns are excluded.
        alignas(64) i16_t lane_lengths[candidate_lanes_k] = {0};
        for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index)
            lane_lengths[lane_index] = static_cast<i16_t>(candidates.lengths[lane_index]);
        __m512i const lane_lengths_vec = _mm512_load_si512(lane_lengths);
        __m512i running_max_vec = zero_vec;

        // Local alignment: the boundary row and column are zero (an alignment may begin at any cell).
        for (size_t column = 0; column <= longest_candidate; ++column)
            _mm512_storeu_si512(previous_row + column * candidate_lanes_k, zero_vec);

        for (size_t query_position = 1; query_position <= query_length; ++query_position) {
            u8_t const query_class = substituter_.byte_to_class[(u8_t)query[query_position - 1]];
            __m512i const cost_row_vec = _mm512_castsi256_si512(_mm256_loadu_si256(
                reinterpret_cast<__m256i const *>(&substituter_.class_substitution_costs[query_class][0])));

            _mm512_storeu_si512(current_row, zero_vec);
            for (size_t column = 1; column <= longest_candidate; ++column) {
                __m256i const candidate_classes_vec = _mm256_loadu_si256(
                    reinterpret_cast<__m256i const *>(candidate_classes + (column - 1) * candidate_lanes_k));
                __m512i const diagonal_vec = _mm512_loadu_si512(previous_row + (column - 1) * candidate_lanes_k);
                __m512i const up_vec = _mm512_loadu_si512(previous_row + column * candidate_lanes_k);
                __m512i const left_vec = _mm512_loadu_si512(current_row + (column - 1) * candidate_lanes_k);

                __m256i const cost_i8_vec =
                    _mm512_castsi512_si256(_mm512_permutexvar_epi8(_mm512_castsi256_si512(candidate_classes_vec),
                                                                   cost_row_vec));
                __m512i const cost_i16_vec = _mm512_cvtepi8_epi16(cost_i8_vec);

                __m512i const cost_if_substitution_vec = _mm512_add_epi16(diagonal_vec, cost_i16_vec);
                __m512i const cost_if_gap_vec = _mm512_add_epi16(_mm512_max_epi16(up_vec, left_vec), gap_vec);
                __m512i const cell_score_vec = _mm512_max_epi16(
                    zero_vec, _mm512_max_epi16(cost_if_substitution_vec, cost_if_gap_vec));
                _mm512_storeu_si512(current_row + column * candidate_lanes_k, cell_score_vec);

                // Fold this column into the running maximum only for lanes whose candidate reaches it.
                __mmask32 const column_live =
                    _mm512_cmpgt_epi16_mask(lane_lengths_vec, _mm512_set1_epi16(static_cast<short>(column - 1)));
                running_max_vec = _mm512_mask_max_epi16(running_max_vec, column_live, running_max_vec, cell_score_vec);
            }
            trivial_swap(previous_row, current_row);
        }

        alignas(64) i16_t final_max[candidate_lanes_k];
        _mm512_store_si512(final_max, running_max_vec);
        for (size_t lane_index = 0; lane_index < candidates.lanes_count; ++lane_index)
            result_lanes[lane_index] = final_max[lane_index];
        return status_t::success_k;
    }
};

/**
 *  @brief Batched byte-level @b weighted Needleman-Wunsch scores on Ice Lake, packing up to 32 candidates of a
 *      shared query into the signed-`i16` `candidate_lane_walker` and falling back to the per-pair anti-diagonal
 *      Dynamic Programming scorer for the long tail (scores that escape `i16`, or sparse query rows).
 *
 *  Mirrors the structure of the Ice Lake `levenshtein_distances` cross-product engine: live cells are walked
 *  query-major, grouped into shared-query blocks, transposed into a reusable scratch buffer (the column-major
 *  layout the lane walker reads), scored, and scattered into the strided result matrix (plus the mirrored slot
 *  for symmetric self-similarity). The lane walker maximizes the signed score; the value type is the wide
 *  `ssize_t`, so the `i16` lane results are widened on scatter.
 */
template <typename allocator_type_, sz_capability_t capability_>
struct needleman_wunsch_scores<error_costs_32x32_t, linear_gap_costs_t, allocator_type_, capability_,
                               std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 32;
    static constexpr ssize_t score_range_limit_k = 30000; // ? `i16` headroom for the lane walker.

    using scoring_t = needleman_wunsch_score<char, substituter_t, gap_costs_t, sz_caps_sil_k>; // ? Per-pair DP fallback.
    using lane_walker_t =
        candidate_lane_walker<char, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_global_k,
                              sz_cap_icelake_k, (int)candidate_lanes_k, void>; // ? AVX-512 shared-query lanes.

    using scratch_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    safe_vector<std::byte, scratch_allocator_t> score_scratch_ {alloc_}; // grow-only, reused; partitioned per worker

    needleman_wunsch_scores(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    needleman_wunsch_scores(substituter_t subs, linear_gap_costs_t gaps, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /** @brief The largest per-cell substitution/gap magnitude; scales the worst-case score against the `i16` range. */
    error_cost_magnitude_t cost_magnitude_() const noexcept {
        return sz_max_of_two(substituter_.magnitude(), gap_costs_.magnitude());
    }

    /** @brief Whether a `(query, candidate)` cell's worst-case score stays inside the lane walker's `i16` headroom. */
    bool fits_lane_range_(size_t query_length, size_t candidate_length) const noexcept {
        ssize_t const magnitude = (ssize_t)cost_magnitude_();
        ssize_t const reach = (ssize_t)(query_length + candidate_length) * magnitude;
        return reach <= score_range_limit_k;
    }

    /**
     *  @brief Worst-case scratch for a single cell over the whole input, in O(Q+C): the transposed-block + lane-walker
     *      arena for the longest candidate, or the anti-diagonal DP fallback for the longest query × longest candidate
     *      (a real cell in both grids, so this is a safe upper bound for every slice).
     */
    template <typename queries_type_, typename candidates_type_>
    size_t worst_cell_scratch_(queries_type_ const &queries, candidates_type_ const &candidates,
                               cpu_specs_t const &specs) const noexcept {
        size_t longest_query = 0, longest_query_index = 0, longest_candidate = 0, longest_candidate_index = 0;
        for (size_t index = 0; index < queries.size(); ++index)
            if (to_view(queries[index]).size() > longest_query)
                longest_query = to_view(queries[index]).size(), longest_query_index = index;
        for (size_t index = 0; index < candidates.size(); ++index)
            if (to_view(candidates[index]).size() > longest_candidate)
                longest_candidate = to_view(candidates[index]).size(), longest_candidate_index = index;
        lane_walker_t lane_walker {substituter_, gap_costs_};
        size_t const transpose_bytes = candidate_lanes_k * longest_candidate * sizeof(char_t);
        size_t const walker_scratch = longest_candidate ? lane_walker.scratch_space_needed(longest_candidate, specs) : 0;
        size_t dp_scratch = 0;
        if (queries.size() && candidates.size() && !fits_lane_range_(longest_query, longest_candidate)) {
            scoring_t dp {substituter_, gap_costs_};
            dp_scratch = dp.scratch_space_needed(to_view(queries[longest_query_index]),
                                                 to_view(candidates[longest_candidate_index]), specs);
        }
        return sz_max_of_two(transpose_bytes + walker_scratch, dp_scratch);
    }

#pragma region - Cross-Product Cell Addressing

    /**
     *  @brief A destination for one scored cell: the primary matrix slot plus an optional mirror slot. The lane
     *      walker writes one score per lane, and the scatter fans it out to both slots on assignment.
     */
    template <typename value_type_>
    struct cross_cell_destination_ {
        value_type_ *primary = nullptr;
        value_type_ *mirror = nullptr;
    };

    /** @brief The number of live cells: the full rectangle, or the lower triangle (incl. diagonal) when symmetric. */
    static size_t live_cells_count_(size_t queries_count, size_t candidates_count,
                                    cross_similarities_t cross_kind) noexcept {
        if (cross_kind == cross_similarities_t::symmetric_k) return queries_count * (queries_count + 1) / 2;
        return queries_count * candidates_count;
    }

    /**
     *  @brief Decodes a flat live-cell index into its `(query_index, candidate_index)` grid coordinates.
     *      For the full rectangle the layout is row-major; for the symmetric case it walks the lower triangle
     *      (including the diagonal) row by row, so consecutive cells stay on the same query row when possible.
     */
    static void cell_to_indices_(size_t cell_index, size_t candidates_count, cross_similarities_t cross_kind,
                                 size_t &query_index, size_t &candidate_index) noexcept {
        if (cross_kind == cross_similarities_t::symmetric_k) {
            size_t row = 0;
            while ((row + 1) * (row + 2) / 2 <= cell_index) ++row;
            query_index = row;
            candidate_index = cell_index - row * (row + 1) / 2;
        }
        else {
            query_index = cell_index / candidates_count;
            candidate_index = cell_index % candidates_count;
        }
    }

#pragma endregion - Cross-Product Cell Addressing

#pragma region - Cross-Product Scoring

    /**
     *  @brief Scores the live cells `[cell_begin, cell_end)` of the cross-product with the shared-query lane walker,
     *      falling back to the anti-diagonal Dynamic Programming scorer for the long tail. Each cell scores
     *      `score(query, candidate)` and writes it into the strided @p results matrix (plus the mirror slot for
     *      symmetric self-similarity).
     */
    template <typename queries_type_, typename candidates_type_, typename results_type_>
    [[gnu::noinline]] status_t score_range_(queries_type_ const &queries, candidates_type_ const &candidates,
                                            results_type_ &&results, cross_similarities_t cross_kind, size_t cell_begin,
                                            size_t cell_end, scratch_space_t scratch, cpu_specs_t const &specs) noexcept {

        using value_t = remove_cvref<decltype(results.data[0])>;
        size_t const candidates_count = candidates.size();

        // The caller provides `scratch`, sized to the worst single cell (`worst_cell_scratch_`). We still scan the
        // slice for its longest lane-eligible candidate to carve the transpose/walker sub-arenas; the global-worst
        // sizing guarantees those offsets stay in bounds.
        scoring_t dp {substituter_, gap_costs_};
        lane_walker_t lane_walker {substituter_, gap_costs_};
        size_t longest_candidate = 0;
        for (size_t cell_index = cell_begin; cell_index != cell_end; ++cell_index) {
            size_t query_index = 0, candidate_index = 0;
            cell_to_indices_(cell_index, candidates_count, cross_kind, query_index, candidate_index);
            size_t const query_length = to_view(queries[query_index]).size();
            size_t const candidate_length = to_view(candidates[candidate_index]).size();
            if (query_length != 0 && candidate_length != 0 && fits_lane_range_(query_length, candidate_length))
                longest_candidate = sz_max_of_two(longest_candidate, candidate_length);
        }

        size_t const transpose_bytes = candidate_lanes_k * longest_candidate * sizeof(char_t);
        size_t const walker_scratch = longest_candidate ? lane_walker.scratch_space_needed(longest_candidate, specs) : 0;
        char_t *transposed = reinterpret_cast<char_t *>(scratch.data());
        scratch_space_t walker_scratch_space = scratch.subspan(transpose_bytes, walker_scratch);
        scratch_space_t dp_scratch_space = scratch;

        // Maps a query row and candidate column to their primary (and mirrored) destination slots.
        auto const destination_for = [&](size_t query_index, size_t candidate_index) noexcept {
            cross_cell_destination_<value_t> destination;
            destination.primary = results.data + query_index * results.row_stride + candidate_index;
            if (cross_kind == cross_similarities_t::symmetric_k && candidate_index != query_index)
                destination.mirror = results.data + candidate_index * results.row_stride + query_index;
            return destination;
        };

        auto const scatter = [&](cross_cell_destination_<value_t> const &destination, ssize_t score) noexcept {
            *destination.primary = static_cast<value_t>(score);
            if (destination.mirror) *destination.mirror = static_cast<value_t>(score);
        };

        dummy_executor_t dummy;
        size_t lengths[candidate_lanes_k];
        cross_cell_destination_<value_t> destinations[candidate_lanes_k];
        i16_t result_lanes[candidate_lanes_k];
        for (size_t cell_index = cell_begin; cell_index != cell_end;) {
            size_t query_index = 0, candidate_index = 0;
            cell_to_indices_(cell_index, candidates_count, cross_kind, query_index, candidate_index);
            auto const query = to_view(queries[query_index]);
            auto const candidate = to_view(candidates[candidate_index]);
            size_t const query_length = query.size(), candidate_length = candidate.size();

            // Empty cell: the only alignment is all-gaps, so the score is `gap * other_length`.
            if (query_length == 0 || candidate_length == 0) {
                ssize_t const other_length = (ssize_t)sz_max_of_two(query_length, candidate_length);
                scatter(destination_for(query_index, candidate_index),
                        (ssize_t)gap_costs_.open_or_extend * other_length);
                ++cell_index;
                continue;
            }

            // Long tail: any cell whose worst-case score escapes the `i16` range walks the per-pair scorer.
            if (!fits_lane_range_(query_length, candidate_length)) {
                ssize_t result_score = 0;
                if (status_t status = dp(query, candidate, result_score, dp_scratch_space, dummy, specs);
                    status != status_t::success_k)
                    return status;
                scatter(destination_for(query_index, candidate_index), result_score);
                ++cell_index;
                continue;
            }

            // Seed a shared-query block with this cell, then extend over consecutive same-query candidates that
            // also fit the `i16` range, up to the 32-lane capacity. Seeding first guarantees forward progress.
            size_t const seed_query_index = query_index;
            size_t const block_begin = cell_index;
            size_t block_longest = candidate_length;
            lengths[0] = candidate_length;
            destinations[0] = destination_for(query_index, candidate_index);
            index_t lanes_count = 1;
            ++cell_index;
            for (; cell_index != cell_end && lanes_count != candidate_lanes_k; ++cell_index, ++lanes_count) {
                size_t next_query_index = 0, next_candidate_index = 0;
                cell_to_indices_(cell_index, candidates_count, cross_kind, next_query_index, next_candidate_index);
                if (next_query_index != seed_query_index) break;
                auto const next_candidate = to_view(candidates[next_candidate_index]);
                if (next_candidate.size() == 0 || !fits_lane_range_(query_length, next_candidate.size())) break;
                lengths[lanes_count] = next_candidate.size();
                destinations[lanes_count] = destination_for(next_query_index, next_candidate_index);
                block_longest = sz_max_of_two(block_longest, next_candidate.size());
            }

            // Transpose the block into column-major lane order, zero-padding the ragged tails of every lane.
            for (size_t position = 0; position != candidate_lanes_k * block_longest; ++position) transposed[position] = 0;
            for (index_t lane_index = 0; lane_index < lanes_count; ++lane_index) {
                size_t fill_query_index = 0, fill_candidate_index = 0;
                cell_to_indices_(block_begin + lane_index, candidates_count, cross_kind, fill_query_index,
                                 fill_candidate_index);
                auto const lane_candidate = to_view(candidates[fill_candidate_index]);
                for (size_t position = 0; position < lane_candidate.size(); ++position)
                    transposed[position * candidate_lanes_k + lane_index] = lane_candidate[position];
            }

            candidate_lanes_block_t<char_t> block;
            block.transposed = transposed;
            block.lane_capacity = candidate_lanes_k;
            block.lanes_count = lanes_count;
            block.lengths = lengths;
            block.longest_candidate = block_longest;

            if (status_t status = lane_walker(query, block, result_lanes, walker_scratch_space, specs);
                status != status_t::success_k)
                return status;
            for (index_t lane_index = 0; lane_index < lanes_count; ++lane_index)
                scatter(destinations[lane_index], (ssize_t)result_lanes[lane_index]);
        }
        return status_t::success_k;
    }

    /** @brief Scores the cross-product in parallel: each worker takes a contiguous slice of the live-cell range. */
    template <typename queries_type_, typename candidates_type_, typename results_type_, typename executor_type_>
    [[gnu::noinline]] status_t score_parallel_(queries_type_ const &queries, candidates_type_ const &candidates,
                                               results_type_ &&results, cross_similarities_t cross_kind,
                                               executor_type_ &&executor, cpu_specs_t const &specs) noexcept {
        size_t const cells_count = live_cells_count_(queries.size(), candidates.size(), cross_kind);
        // One hoisted buffer carved into a per-worker slice; `for_slices` invokes the body at most `threads_count`
        // times, and the atomic counter hands each invocation a disjoint slice (no aliasing, no per-worker alloc).
        size_t const worker_scratch = worst_cell_scratch_(queries, candidates, specs);
        size_t const workers = sz_max_of_two(sz_min_of_two(executor.threads_count(), cells_count), (size_t)1);
        if (status_t status = score_scratch_.try_resize(worker_scratch * workers); status != status_t::success_k)
            return status;
        std::atomic<size_t> next_worker {0};
        std::atomic<status_t> error {status_t::success_k};
        executor.for_slices(cells_count, [&](size_t cell_begin, size_t length) noexcept {
            if (length == 0) return; // empty slice: no work, and it must not consume a scratch partition
            size_t const worker = next_worker.fetch_add(1, std::memory_order_relaxed);
            scratch_space_t slice = scratch_space_t(score_scratch_).subspan(worker * worker_scratch, worker_scratch);
            status_t status =
                score_range_(queries, candidates, results, cross_kind, cell_begin, cell_begin + length, slice, specs);
            if (status != status_t::success_k) error.store(status);
        });
        return error.load();
    }

#pragma endregion - Cross-Product Scoring

#pragma region - Public Cross-Product Overloads

    // Concrete `strided_rows<value_type_>` results parameter disambiguates the two-set and symmetric overloads by
    // type alone (no concepts needed; `SZ_HAS_CONCEPTS_` is off repo-wide for the GCC concepts bug).
    template <typename queries_type_, typename candidates_type_, typename value_type_>
    status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                        strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        if (status_t status = score_scratch_.try_resize(worst_cell_scratch_(queries, candidates, specs));
            status != status_t::success_k)
            return status;
        return score_range_(queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
                            live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
                            scratch_space_t(score_scratch_), specs);
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_, typename executor_type_>
    status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                        strided_rows<value_type_> results, executor_type_ &&executor,
                        cpu_specs_t const &specs = {}) noexcept {
        return score_parallel_(queries, candidates, results, cross_similarities_t::all_pairs_k,
                               std::forward<executor_type_>(executor), specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                        cpu_specs_t const &specs = {}) noexcept {
        if (status_t status = score_scratch_.try_resize(worst_cell_scratch_(sequences, sequences, specs));
            status != status_t::success_k)
            return status;
        return score_range_(sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
                            live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
                            scratch_space_t(score_scratch_), specs);
    }

    template <typename sequences_type_, typename value_type_, typename executor_type_>
    status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                        executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        return score_parallel_(sequences, sequences, results, cross_similarities_t::symmetric_k,
                               std::forward<executor_type_>(executor), specs);
    }

#pragma endregion - Public Cross-Product Overloads
};

/**
 *  @brief Batched byte-level @b weighted Smith-Waterman (local) scores on Ice Lake, packing up to 32 candidates of a
 *      shared query into the signed-`i16` local `candidate_lane_walker` and falling back to the per-pair anti-diagonal
 *      Dynamic Programming scorer for the long tail (scores that escape `i16`, or sparse query rows).
 *
 *  Local sibling of the Ice Lake `needleman_wunsch_scores` cross-product engine: identical query-major grouping,
 *  transpose, and scatter, but it dispatches the @b local lane walker (zero-clamped recurrence, score = the maximum
 *  cell of the matrix) and an empty `(query, candidate)` cell scores @b 0 (a local alignment may align nothing).
 */
template <typename allocator_type_, sz_capability_t capability_>
struct smith_waterman_scores<error_costs_32x32_t, linear_gap_costs_t, allocator_type_, capability_,
                             std::enable_if_t<(capability_ & sz_cap_icelake_k) != 0>> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = linear_gap_costs_t;
    using allocator_t = allocator_type_;
    using index_t = u32_t;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr size_t candidate_lanes_k = 32;
    static constexpr ssize_t score_range_limit_k = 30000; // ? `i16` headroom for the lane walker.

    using scoring_t = smith_waterman_score<char, substituter_t, gap_costs_t, sz_caps_sil_k>; // ? Per-pair DP fallback.
    using lane_walker_t =
        candidate_lane_walker<char, i16_t, substituter_t, gap_costs_t, sz_maximize_score_k, sz_similarity_local_k,
                              sz_cap_icelake_k, (int)candidate_lanes_k, void>; // ? AVX-512 shared-query local lanes.

    using scratch_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;

    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    safe_vector<std::byte, scratch_allocator_t> score_scratch_ {alloc_}; // grow-only, reused; partitioned per worker

    smith_waterman_scores(allocator_t alloc = {}) noexcept : alloc_(alloc) {}
    smith_waterman_scores(substituter_t subs, linear_gap_costs_t gaps, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    /** @brief The largest per-cell substitution/gap magnitude; scales the worst-case score against the `i16` range. */
    error_cost_magnitude_t cost_magnitude_() const noexcept {
        return sz_max_of_two(substituter_.magnitude(), gap_costs_.magnitude());
    }

    /** @brief Whether a `(query, candidate)` cell's worst-case score stays inside the lane walker's `i16` headroom. */
    bool fits_lane_range_(size_t query_length, size_t candidate_length) const noexcept {
        ssize_t const magnitude = (ssize_t)cost_magnitude_();
        ssize_t const reach = (ssize_t)(query_length + candidate_length) * magnitude;
        return reach <= score_range_limit_k;
    }

    /**
     *  @brief Worst-case scratch for a single cell over the whole input, in O(Q+C): the transposed-block + lane-walker
     *      arena for the longest candidate, or the anti-diagonal DP fallback for the longest query × longest candidate
     *      (a real cell in both grids, so this is a safe upper bound for every slice).
     */
    template <typename queries_type_, typename candidates_type_>
    size_t worst_cell_scratch_(queries_type_ const &queries, candidates_type_ const &candidates,
                               cpu_specs_t const &specs) const noexcept {
        size_t longest_query = 0, longest_query_index = 0, longest_candidate = 0, longest_candidate_index = 0;
        for (size_t index = 0; index < queries.size(); ++index)
            if (to_view(queries[index]).size() > longest_query)
                longest_query = to_view(queries[index]).size(), longest_query_index = index;
        for (size_t index = 0; index < candidates.size(); ++index)
            if (to_view(candidates[index]).size() > longest_candidate)
                longest_candidate = to_view(candidates[index]).size(), longest_candidate_index = index;
        lane_walker_t lane_walker {substituter_, gap_costs_};
        size_t const transpose_bytes = candidate_lanes_k * longest_candidate * sizeof(char_t);
        size_t const walker_scratch = longest_candidate ? lane_walker.scratch_space_needed(longest_candidate, specs) : 0;
        size_t dp_scratch = 0;
        if (queries.size() && candidates.size() && !fits_lane_range_(longest_query, longest_candidate)) {
            scoring_t dp {substituter_, gap_costs_};
            dp_scratch = dp.scratch_space_needed(to_view(queries[longest_query_index]),
                                                 to_view(candidates[longest_candidate_index]), specs);
        }
        return sz_max_of_two(transpose_bytes + walker_scratch, dp_scratch);
    }

#pragma region - Cross-Product Cell Addressing

    template <typename value_type_>
    struct cross_cell_destination_ {
        value_type_ *primary = nullptr;
        value_type_ *mirror = nullptr;
    };

    /** @brief The number of live cells: the full rectangle, or the lower triangle (incl. diagonal) when symmetric. */
    static size_t live_cells_count_(size_t queries_count, size_t candidates_count,
                                    cross_similarities_t cross_kind) noexcept {
        if (cross_kind == cross_similarities_t::symmetric_k) return queries_count * (queries_count + 1) / 2;
        return queries_count * candidates_count;
    }

    /** @brief Decodes a flat live-cell index into its `(query_index, candidate_index)` grid coordinates. */
    static void cell_to_indices_(size_t cell_index, size_t candidates_count, cross_similarities_t cross_kind,
                                 size_t &query_index, size_t &candidate_index) noexcept {
        if (cross_kind == cross_similarities_t::symmetric_k) {
            size_t row = 0;
            while ((row + 1) * (row + 2) / 2 <= cell_index) ++row;
            query_index = row;
            candidate_index = cell_index - row * (row + 1) / 2;
        }
        else {
            query_index = cell_index / candidates_count;
            candidate_index = cell_index % candidates_count;
        }
    }

#pragma endregion - Cross-Product Cell Addressing

#pragma region - Cross-Product Scoring

    /**
     *  @brief Scores the live cells `[cell_begin, cell_end)` with the shared-query local lane walker, falling back
     *      to the anti-diagonal Dynamic Programming scorer for the long tail. Writes each `score(query, candidate)`
     *      into the strided @p results matrix (plus the mirror slot for symmetric self-similarity).
     */
    template <typename queries_type_, typename candidates_type_, typename results_type_>
    [[gnu::noinline]] status_t score_range_(queries_type_ const &queries, candidates_type_ const &candidates,
                                            results_type_ &&results, cross_similarities_t cross_kind, size_t cell_begin,
                                            size_t cell_end, scratch_space_t scratch, cpu_specs_t const &specs) noexcept {

        using value_t = remove_cvref<decltype(results.data[0])>;
        size_t const candidates_count = candidates.size();

        // The caller provides `scratch`, sized to the worst single cell (`worst_cell_scratch_`). We still scan the
        // slice for its longest lane-eligible candidate to carve the transpose/walker sub-arenas; the global-worst
        // sizing guarantees those offsets stay in bounds.
        scoring_t dp {substituter_, gap_costs_};
        lane_walker_t lane_walker {substituter_, gap_costs_};
        size_t longest_candidate = 0;
        for (size_t cell_index = cell_begin; cell_index != cell_end; ++cell_index) {
            size_t query_index = 0, candidate_index = 0;
            cell_to_indices_(cell_index, candidates_count, cross_kind, query_index, candidate_index);
            size_t const query_length = to_view(queries[query_index]).size();
            size_t const candidate_length = to_view(candidates[candidate_index]).size();
            if (query_length != 0 && candidate_length != 0 && fits_lane_range_(query_length, candidate_length))
                longest_candidate = sz_max_of_two(longest_candidate, candidate_length);
        }

        size_t const transpose_bytes = candidate_lanes_k * longest_candidate * sizeof(char_t);
        size_t const walker_scratch = longest_candidate ? lane_walker.scratch_space_needed(longest_candidate, specs) : 0;
        char_t *transposed = reinterpret_cast<char_t *>(scratch.data());
        scratch_space_t walker_scratch_space = scratch.subspan(transpose_bytes, walker_scratch);
        scratch_space_t dp_scratch_space = scratch;

        auto const destination_for = [&](size_t query_index, size_t candidate_index) noexcept {
            cross_cell_destination_<value_t> destination;
            destination.primary = results.data + query_index * results.row_stride + candidate_index;
            if (cross_kind == cross_similarities_t::symmetric_k && candidate_index != query_index)
                destination.mirror = results.data + candidate_index * results.row_stride + query_index;
            return destination;
        };

        auto const scatter = [&](cross_cell_destination_<value_t> const &destination, ssize_t score) noexcept {
            *destination.primary = static_cast<value_t>(score);
            if (destination.mirror) *destination.mirror = static_cast<value_t>(score);
        };

        dummy_executor_t dummy;
        size_t lengths[candidate_lanes_k];
        cross_cell_destination_<value_t> destinations[candidate_lanes_k];
        i16_t result_lanes[candidate_lanes_k];
        for (size_t cell_index = cell_begin; cell_index != cell_end;) {
            size_t query_index = 0, candidate_index = 0;
            cell_to_indices_(cell_index, candidates_count, cross_kind, query_index, candidate_index);
            auto const query = to_view(queries[query_index]);
            auto const candidate = to_view(candidates[candidate_index]);
            size_t const query_length = query.size(), candidate_length = candidate.size();

            // Empty cell: a local alignment may align nothing, so the score is 0.
            if (query_length == 0 || candidate_length == 0) {
                scatter(destination_for(query_index, candidate_index), 0);
                ++cell_index;
                continue;
            }

            // Long tail: any cell whose worst-case score escapes the `i16` range walks the per-pair scorer.
            if (!fits_lane_range_(query_length, candidate_length)) {
                ssize_t result_score = 0;
                if (status_t status = dp(query, candidate, result_score, dp_scratch_space, dummy, specs);
                    status != status_t::success_k)
                    return status;
                scatter(destination_for(query_index, candidate_index), result_score);
                ++cell_index;
                continue;
            }

            size_t const seed_query_index = query_index;
            size_t const block_begin = cell_index;
            size_t block_longest = candidate_length;
            lengths[0] = candidate_length;
            destinations[0] = destination_for(query_index, candidate_index);
            index_t lanes_count = 1;
            ++cell_index;
            for (; cell_index != cell_end && lanes_count != candidate_lanes_k; ++cell_index, ++lanes_count) {
                size_t next_query_index = 0, next_candidate_index = 0;
                cell_to_indices_(cell_index, candidates_count, cross_kind, next_query_index, next_candidate_index);
                if (next_query_index != seed_query_index) break;
                auto const next_candidate = to_view(candidates[next_candidate_index]);
                if (next_candidate.size() == 0 || !fits_lane_range_(query_length, next_candidate.size())) break;
                lengths[lanes_count] = next_candidate.size();
                destinations[lanes_count] = destination_for(next_query_index, next_candidate_index);
                block_longest = sz_max_of_two(block_longest, next_candidate.size());
            }

            for (size_t position = 0; position != candidate_lanes_k * block_longest; ++position) transposed[position] = 0;
            for (index_t lane_index = 0; lane_index < lanes_count; ++lane_index) {
                size_t fill_query_index = 0, fill_candidate_index = 0;
                cell_to_indices_(block_begin + lane_index, candidates_count, cross_kind, fill_query_index,
                                 fill_candidate_index);
                auto const lane_candidate = to_view(candidates[fill_candidate_index]);
                for (size_t position = 0; position < lane_candidate.size(); ++position)
                    transposed[position * candidate_lanes_k + lane_index] = lane_candidate[position];
            }

            candidate_lanes_block_t<char_t> block;
            block.transposed = transposed;
            block.lane_capacity = candidate_lanes_k;
            block.lanes_count = lanes_count;
            block.lengths = lengths;
            block.longest_candidate = block_longest;

            if (status_t status = lane_walker(query, block, result_lanes, walker_scratch_space, specs);
                status != status_t::success_k)
                return status;
            for (index_t lane_index = 0; lane_index < lanes_count; ++lane_index)
                scatter(destinations[lane_index], (ssize_t)result_lanes[lane_index]);
        }
        return status_t::success_k;
    }

    /** @brief Scores the cross-product in parallel: each worker takes a contiguous slice of the live-cell range. */
    template <typename queries_type_, typename candidates_type_, typename results_type_, typename executor_type_>
    [[gnu::noinline]] status_t score_parallel_(queries_type_ const &queries, candidates_type_ const &candidates,
                                               results_type_ &&results, cross_similarities_t cross_kind,
                                               executor_type_ &&executor, cpu_specs_t const &specs) noexcept {
        size_t const cells_count = live_cells_count_(queries.size(), candidates.size(), cross_kind);
        // One hoisted buffer carved into a per-worker slice; `for_slices` invokes the body at most `threads_count`
        // times, and the atomic counter hands each invocation a disjoint slice (no aliasing, no per-worker alloc).
        size_t const worker_scratch = worst_cell_scratch_(queries, candidates, specs);
        size_t const workers = sz_max_of_two(sz_min_of_two(executor.threads_count(), cells_count), (size_t)1);
        if (status_t status = score_scratch_.try_resize(worker_scratch * workers); status != status_t::success_k)
            return status;
        std::atomic<size_t> next_worker {0};
        std::atomic<status_t> error {status_t::success_k};
        executor.for_slices(cells_count, [&](size_t cell_begin, size_t length) noexcept {
            if (length == 0) return; // empty slice: no work, and it must not consume a scratch partition
            size_t const worker = next_worker.fetch_add(1, std::memory_order_relaxed);
            scratch_space_t slice = scratch_space_t(score_scratch_).subspan(worker * worker_scratch, worker_scratch);
            status_t status =
                score_range_(queries, candidates, results, cross_kind, cell_begin, cell_begin + length, slice, specs);
            if (status != status_t::success_k) error.store(status);
        });
        return error.load();
    }

#pragma endregion - Cross-Product Scoring

#pragma region - Public Cross-Product Overloads

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                        strided_rows<value_type_> results, cpu_specs_t const &specs = {}) noexcept {
        if (status_t status = score_scratch_.try_resize(worst_cell_scratch_(queries, candidates, specs));
            status != status_t::success_k)
            return status;
        return score_range_(queries, candidates, results, cross_similarities_t::all_pairs_k, 0,
                            live_cells_count_(queries.size(), candidates.size(), cross_similarities_t::all_pairs_k),
                            scratch_space_t(score_scratch_), specs);
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_, typename executor_type_>
    status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                        strided_rows<value_type_> results, executor_type_ &&executor,
                        cpu_specs_t const &specs = {}) noexcept {
        return score_parallel_(queries, candidates, results, cross_similarities_t::all_pairs_k,
                               std::forward<executor_type_>(executor), specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                        cpu_specs_t const &specs = {}) noexcept {
        if (status_t status = score_scratch_.try_resize(worst_cell_scratch_(sequences, sequences, specs));
            status != status_t::success_k)
            return status;
        return score_range_(sequences, sequences, results, cross_similarities_t::symmetric_k, 0,
                            live_cells_count_(sequences.size(), sequences.size(), cross_similarities_t::symmetric_k),
                            scratch_space_t(score_scratch_), specs);
    }

    template <typename sequences_type_, typename value_type_, typename executor_type_>
    status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                        executor_type_ &&executor, cpu_specs_t const &specs = {}) noexcept {
        return score_parallel_(sequences, sequences, results, cross_similarities_t::symmetric_k,
                               std::forward<executor_type_>(executor), specs);
    }

#pragma endregion - Public Cross-Product Overloads
};

#pragma endregion - Inter-Sequence Candidate Lanes

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
