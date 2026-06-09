/**
 *  @brief CUDA base-tier string-similarity backend (SIMT, pre-Kepler).
 *  @file include/stringzillas/similarities/cuda.cuh
 *  @author Ash Vardanian
 *  @sa include/stringzillas/similarities/serial.hpp
 */
#ifndef STRINGZILLAS_SIMILARITIES_CUDA_CUH_
#define STRINGZILLAS_SIMILARITIES_CUDA_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>        // `cuda::pipeline`
#include <cooperative_groups.h> // `cooperative_groups::this_grid()`

#include "stringzillas/types.cuh"
#include "stringzillas/similarities/serial.hpp"

namespace ashvardanian {
namespace stringzillas {

#pragma region - Common Aliases

using ualloc_t = unified_alloc_t;

/**
 *  In @b CUDA:
 *  - for GPUs before Hopper, we can use the @b SIMT model for warp-level parallelism using diagonal "walkers"
 *  - for GPUs after Hopper, we compound that with thread-level @b SIMD via @b DPX instructions for min-max
 */
using levenshtein_cuda_t = levenshtein_distances<char, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
using affine_levenshtein_cuda_t = levenshtein_distances<char, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k>;

using levenshtein_kepler_t = levenshtein_distances<char, linear_gap_costs_t, ualloc_t, sz_caps_ck_k>;
using affine_levenshtein_kepler_t = levenshtein_distances<char, affine_gap_costs_t, ualloc_t, sz_caps_ck_k>;

using levenshtein_hopper_t = levenshtein_distances<char, linear_gap_costs_t, ualloc_t, sz_caps_ckh_k>;
using affine_levenshtein_hopper_t = levenshtein_distances<char, affine_gap_costs_t, ualloc_t, sz_caps_ckh_k>;

using needleman_wunsch_cuda_t =
    needleman_wunsch_scores<char, error_costs_32x32_t, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
using smith_waterman_cuda_t =
    smith_waterman_scores<char, error_costs_32x32_t, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k>;

using affine_needleman_wunsch_cuda_t =
    needleman_wunsch_scores<char, error_costs_32x32_t, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
using affine_smith_waterman_cuda_t =
    smith_waterman_scores<char, error_costs_32x32_t, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k>;

using needleman_wunsch_hopper_t =
    needleman_wunsch_scores<char, error_costs_32x32_t, linear_gap_costs_t, ualloc_t, sz_caps_ckh_k>;
using smith_waterman_hopper_t =
    smith_waterman_scores<char, error_costs_32x32_t, linear_gap_costs_t, ualloc_t, sz_caps_ckh_k>;

using affine_needleman_wunsch_hopper_t =
    needleman_wunsch_scores<char, error_costs_32x32_t, affine_gap_costs_t, ualloc_t, sz_caps_ckh_k>;
using affine_smith_waterman_hopper_t =
    smith_waterman_scores<char, error_costs_32x32_t, affine_gap_costs_t, ualloc_t, sz_caps_ckh_k>;

#pragma endregion - Common Aliases

#pragma region - Common Helpers

/**
 *  @brief Dispatches min or max operation based on the compile-time objective.
 */
template <sz_similarity_objective_t objective_, typename scalar_type_>
__forceinline__ __device__ scalar_type_ pick_best_(scalar_type_ a, scalar_type_ b) noexcept {
    if constexpr (objective_ == sz_minimize_distance_k) { return std::min(a, b); }
    else { return std::max(a, b); }
}

template <sz_similarity_objective_t objective_, typename scalar_type_>
__forceinline__ __device__ scalar_type_ pick_best_in_warp_(scalar_type_ x) noexcept {
    // The `__shfl_down_sync` replaces `__shfl_down`
    // https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
    x = pick_best_<objective_, scalar_type_>(__shfl_down_sync(0xffffffff, x, 16), x);
    x = pick_best_<objective_, scalar_type_>(__shfl_down_sync(0xffffffff, x, 8), x);
    x = pick_best_<objective_, scalar_type_>(__shfl_down_sync(0xffffffff, x, 4), x);
    x = pick_best_<objective_, scalar_type_>(__shfl_down_sync(0xffffffff, x, 2), x);
    x = pick_best_<objective_, scalar_type_>(__shfl_down_sync(0xffffffff, x, 1), x);
    return x;
}

/**
 *  @brief Loads data with a hint, that it's frequently accessed and immutable throughout the kernel.
 *  @see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#read-only-data-cache-load-function
 *  @see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-memory-5-x
 */
template <typename scalar_type_>
__forceinline__ __device__ scalar_type_ load_immutable_(scalar_type_ const *ptr) noexcept {
    // The `__ldg` intrinsic translates into the `ld.global.nc` PTX instruction.
    // It reads a value from global memory and caches it in the non-coherent cache.
    // return __ldg(ptr);
    return *ptr;
}

/**
 *  @brief Loads data with a cache hint, that it will not be accessed again.
 *  @see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#load-functions-using-cache-hints
 *  @see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators
 */
template <typename scalar_type_>
__forceinline__ __device__ scalar_type_ load_last_use_(scalar_type_ const *ptr) noexcept {
    // return __ldlu(ptr);
    return *ptr;
}

#pragma endregion - Common Helpers

#pragma region - Algorithm Building Blocks

/**
 *  @brief GPU adaptation of the `tile_scorer` on CUDA, avoiding warp-level shuffles and DPX.
 *  @note Uses 32-bit `unsigned` counter to iterate through the string slices, so it can't be over 4 billion characters.
 */
template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_, sz_capability_t capability_>
#if SZ_HAS_CONCEPTS_
    requires pointer_like<first_iterator_type_> && pointer_like<second_iterator_type_> && score_like<score_type_> &&
             substituter_like<substituter_type_>
#endif
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, linear_gap_costs_t,
                   objective_, sz_similarity_global_k, capability_, std::enable_if_t<capability_ == sz_cap_cuda_k>> {

    using first_iterator_t = first_iterator_type_;
    using second_iterator_t = second_iterator_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = linear_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    using first_char_t = typename std::iterator_traits<first_iterator_t>::value_type;
    using second_char_t = typename std::iterator_traits<second_iterator_t>::value_type;
    static_assert(is_same_type<first_char_t, second_char_t>::value, "String characters must be of the same type.");
    using char_t = remove_cvref<first_char_t>;

    using cuda_warp_scorer_t = tile_scorer<first_iterator_t, second_iterator_t, score_t, substituter_t,
                                           linear_gap_costs_t, objective_k, sz_similarity_global_k, capability_k>;

  protected:
    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    score_t final_score_ {0};

  public:
    __forceinline__ __device__ tile_scorer(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /**
     *  @brief Initializes a boundary value within a certain diagonal.
     *  @note Should only be called for the diagonals outside of the bottom-right triangle.
     *  @note Should only be called for the top row and left column of the matrix.
     */
    __forceinline__ __device__ void init_score(score_t &cell, size_t diagonal_index) const noexcept {
        cell = gap_costs_.open_or_extend * diagonal_index;
    }

    /**
     *  @brief Extract the final result of the scoring operation which will be always in the bottom-right corner.
     */
    __forceinline__ __device__ score_t score() const noexcept { return final_score_; }

    /**
     *  @brief Computes one diagonal of the DP matrix, using the results of the previous 2x diagonals.
     *  @param first_slice The first string, unlike the CPU variant @b NOT reversed.
     *  @param second_slice The second string.
     *
     *  @param tasks_offset The offset of the first character to compare from each string.
     *  @param tasks_step The step size for the next character to compare from each string.
     *  @param tasks_count The total number of characters to compare from input slices.
     *
     *  @tparam index_type_ @b `unsigned` is recommended if the strings are under 4 billion characters.
     */
    template <typename index_type_>
    __forceinline__ __device__ void operator()(                                                      //
        first_iterator_t first_slice, second_iterator_t second_slice,                                //
        index_type_ const tasks_offset, index_type_ const tasks_step, index_type_ const tasks_count, //
        score_t const *scores_pre_substitution, score_t const *scores_pre_insertion,                 //
        score_t const *scores_pre_deletion, score_t *scores_new) noexcept {

        // Make sure we are called for an anti-diagonal traversal order
        score_t const gap_costs = gap_costs_.open_or_extend;
        sz_assert_(scores_pre_insertion + 1 == scores_pre_deletion);

        // ? One weird observation, is that even though we can avoid fetching `pre_insertion`
        // ? from shared memory on each cycle, by slicing the work differently between the threads,
        // ? and allowing them to reuse the previous `pre_deletion` as the new `pre_insertion`,
        // ? that code ends up being slower than the one below.
        for (index_type_ i = tasks_offset; i < tasks_count; i += tasks_step) {
            score_t pre_substitution = load_last_use_(scores_pre_substitution + i);
            score_t pre_insertion = scores_pre_insertion[i];
            score_t pre_deletion = scores_pre_deletion[i];
            char first_char = load_immutable_(first_slice + tasks_count - i - 1);
            char second_char = load_immutable_(second_slice + i);

            error_cost_t cost_of_substitution = substituter_(first_char, second_char);
            score_t if_substitution = pre_substitution + cost_of_substitution;
            score_t if_deletion_or_insertion = pick_best_<objective_k>(pre_deletion, pre_insertion) + gap_costs;
            score_t cell_score = pick_best_<objective_k>(if_deletion_or_insertion, if_substitution);
            scores_new[i] = cell_score;
        }

        // The last element of the last chunk is the result of the global alignment.
        if (tasks_offset == 0) final_score_ = scores_new[0];
    }
};

/**
 *  @brief GPU adaptation of the `local_scorer` on CUDA, avoiding warp-level shuffles and DPX.
 *  @note Uses 32-bit `unsigned` counter to iterate through the string slices, so it can't be over 4 billion characters.
 */
template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_, sz_capability_t capability_>
#if SZ_HAS_CONCEPTS_
    requires pointer_like<first_iterator_type_> && pointer_like<second_iterator_type_> && score_like<score_type_> &&
             substituter_like<substituter_type_>
#endif
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, linear_gap_costs_t,
                   objective_, sz_similarity_local_k, capability_, std::enable_if_t<capability_ == sz_cap_cuda_k>> {

    using first_iterator_t = first_iterator_type_;
    using second_iterator_t = second_iterator_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    using first_char_t = typename std::iterator_traits<first_iterator_t>::value_type;
    using second_char_t = typename std::iterator_traits<second_iterator_t>::value_type;
    static_assert(is_same_type<first_char_t, second_char_t>::value, "String characters must be of the same type.");
    using char_t = remove_cvref<first_char_t>;

    using cuda_warp_scorer_t = tile_scorer<first_iterator_t, second_iterator_t, score_t, substituter_t,
                                           linear_gap_costs_t, objective_k, sz_similarity_local_k, capability_k>;

  protected:
    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    score_t final_score_ {0};

  public:
    __forceinline__ __device__ tile_scorer(substituter_t subs, linear_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /**
     *  @brief Initializes a boundary value within a certain diagonal.
     *  @note Should only be called for the diagonals outside of the bottom-right triangle.
     *  @note Should only be called for the top row and left column of the matrix.
     */
    __forceinline__ __device__ void init_score(score_t &cell, size_t diagonal_index) const noexcept { cell = 0; }

    /**
     *  @brief Extract the final result of the scoring operation which will be always in the bottom-right corner.
     */
    __forceinline__ __device__ score_t score() const noexcept { return final_score_; }

    /**
     *  @brief Computes one diagonal of the DP matrix, using the results of the previous 2x diagonals.
     *  @param first_slice The first string, unlike the CPU variant @b NOT reversed.
     *  @param second_slice The second string.
     *
     *  @param tasks_offset The offset of the first character to compare from each string.
     *  @param tasks_step The step size for the next character to compare from each string.
     *  @param tasks_count The total number of characters to compare from input slices.
     *
     *  @tparam index_type_ @b `unsigned` is recommended if the strings are under 4 billion characters.
     */
    template <typename index_type_>
    __forceinline__ __device__ void operator()(                                                      //
        first_iterator_t first_slice, second_iterator_t second_slice,                                //
        index_type_ const tasks_offset, index_type_ const tasks_step, index_type_ const tasks_count, //
        score_t const *scores_pre_substitution, score_t const *scores_pre_insertion,                 //
        score_t const *scores_pre_deletion, score_t *scores_new) noexcept {

        // Make sure we are called for an anti-diagonal traversal order
        error_cost_t const gap_cost = gap_costs_.open_or_extend;
        sz_assert_(scores_pre_insertion + 1 == scores_pre_deletion);

        // ? One weird observation, is that even though we can avoid fetching `pre_insertion`
        // ? from shared memory on each cycle, by slicing the work differently between the threads,
        // ? and allowing them to reuse the previous `pre_deletion` as the new `pre_insertion`,
        // ? that code ends up being slower than the one below.
        for (index_type_ i = tasks_offset; i < tasks_count; i += tasks_step) {
            score_t pre_substitution = load_last_use_(scores_pre_substitution + i);
            score_t pre_insertion = scores_pre_insertion[i];
            score_t pre_deletion = scores_pre_deletion[i];
            char first_char = load_immutable_(first_slice + tasks_count - i - 1);
            char second_char = load_immutable_(second_slice + i);

            error_cost_t cost_of_substitution = substituter_(first_char, second_char);
            score_t if_substitution = pre_substitution + cost_of_substitution;
            score_t if_deletion_or_insertion = pick_best_<objective_k>(pre_deletion, pre_insertion) + gap_cost;
            score_t if_substitution_or_reset = pick_best_<objective_k, score_t>(if_substitution, 0);
            score_t cell_score = pick_best_<objective_k>(if_deletion_or_insertion, if_substitution_or_reset);
            scores_new[i] = cell_score;

            // Update the global maximum score if this cell beats it.
            final_score_ = pick_best_<objective_k>(final_score_, cell_score);
        }

        // ! Don't forget to pick the best among the best scores per thread.
        final_score_ = pick_best_in_warp_<objective_k>(final_score_);
    }
};

/**
 *  @brief GPU adaptation of the `tile_scorer` on CUDA, avoiding warp-level shuffles and DPX.
 *  @note Uses 32-bit `unsigned` counter to iterate through the string slices, so it can't be over 4 billion characters.
 */
template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_, sz_capability_t capability_>
#if SZ_HAS_CONCEPTS_
    requires pointer_like<first_iterator_type_> && pointer_like<second_iterator_type_> && score_like<score_type_> &&
             substituter_like<substituter_type_>
#endif
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, affine_gap_costs_t,
                   objective_, sz_similarity_global_k, capability_, std::enable_if_t<capability_ == sz_cap_cuda_k>> {

    using first_iterator_t = first_iterator_type_;
    using second_iterator_t = second_iterator_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = affine_gap_costs_t;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    using first_char_t = typename std::iterator_traits<first_iterator_t>::value_type;
    using second_char_t = typename std::iterator_traits<second_iterator_t>::value_type;
    static_assert(is_same_type<first_char_t, second_char_t>::value, "String characters must be of the same type.");
    using char_t = remove_cvref<first_char_t>;

    using cuda_warp_scorer_t = tile_scorer<first_iterator_t, second_iterator_t, score_t, substituter_t,
                                           affine_gap_costs_t, objective_k, sz_similarity_global_k, capability_k>;

  protected:
    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};
    score_t final_score_ {0};

  public:
    __forceinline__ __device__ tile_scorer(substituter_t subs, affine_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /**
     *  @brief Initializes a boundary value within a certain diagonal.
     *  @note Should only be called for the diagonals outside of the bottom-right triangle.
     *  @note Should only be called for the top row and left column of the matrix.
     */
    __forceinline__ __device__ void init_score(score_t &cell, size_t diagonal_index) const noexcept {
        cell = diagonal_index ? gap_costs_.open + gap_costs_.extend * (diagonal_index - 1) : 0;
    }

    __forceinline__ __device__ void init_gap(score_t &cell, size_t diagonal_index) const noexcept {
        // Make sure the initial value of the gap is not smaller in magnitude than the primary.
        // The supplementary matrices are initialized with values of higher magnitude,
        // which is equivalent to discarding them. That's better than using `SIZE_MAX`
        // as subsequent additions won't overflow.
        cell = (gap_costs_.open + gap_costs_.extend) +
               (diagonal_index ? gap_costs_.open + gap_costs_.extend * (diagonal_index - 1) : 0);
    }

    /**
     *  @brief Extract the final result of the scoring operation which will be always in the bottom-right corner.
     */
    __forceinline__ __device__ score_t score() const noexcept { return final_score_; }

    /**
     *  @brief Computes one diagonal of the DP matrix, using the results of the previous 2x diagonals.
     *  @param first_slice The first string, unlike the CPU variant @b NOT reversed.
     *  @param second_slice The second string.
     *
     *  @param tasks_offset The offset of the first character to compare from each string.
     *  @param tasks_step The step size for the next character to compare from each string.
     *  @param tasks_count The total number of characters to compare from input slices.
     *
     *  @tparam index_type_ @b `unsigned` is recommended if the strings are under 4 billion characters.
     */
    template <typename index_type_>
    __forceinline__ __device__ void operator()(                                                      //
        first_iterator_t first_slice, second_iterator_t second_slice,                                //
        index_type_ const tasks_offset, index_type_ const tasks_step, index_type_ const tasks_count, //
        score_t const *scores_pre_substitution,                                                      //
        score_t const *scores_pre_insertion,                                                         //
        score_t const *scores_pre_deletion,                                                          //
        score_t const *scores_running_insertions,                                                    //
        score_t const *scores_running_deletions,                                                     //
        score_t *scores_new,                                                                         //
        score_t *scores_new_insertions,                                                              //
        score_t *scores_new_deletions) noexcept {

        // Make sure we are called for an anti-diagonal traversal order
        sz_assert_(scores_pre_insertion + 1 == scores_pre_deletion);

        // ? One weird observation, is that even though we can avoid fetching `pre_insertion`
        // ? from shared memory on each cycle, by slicing the work differently between the threads,
        // ? and allowing them to reuse the previous `pre_deletion` as the new `pre_insertion`,
        // ? that code ends up being slower than the one below.
        for (index_type_ i = tasks_offset; i < tasks_count; i += tasks_step) {
            score_t pre_substitution = load_last_use_(scores_pre_substitution + i);
            score_t pre_insertion_opening = scores_pre_insertion[i];
            score_t pre_deletion_opening = scores_pre_deletion[i];
            score_t pre_insertion_expansion = scores_running_insertions[i];
            score_t pre_deletion_expansion = scores_running_deletions[i];
            char first_char = load_immutable_(first_slice + tasks_count - i - 1);
            char second_char = load_immutable_(second_slice + i);

            error_cost_t cost_of_substitution = substituter_(first_char, second_char);
            score_t if_substitution = pre_substitution + cost_of_substitution;
            score_t if_insertion = min_or_max<objective_k>(pre_insertion_opening + gap_costs_.open,
                                                           pre_insertion_expansion + gap_costs_.extend);
            score_t if_deletion = min_or_max<objective_k>(pre_deletion_opening + gap_costs_.open,
                                                          pre_deletion_expansion + gap_costs_.extend);
            score_t if_deletion_or_insertion = min_or_max<objective_k>(if_deletion, if_insertion);
            score_t cell_score = pick_best_<objective_k>(if_deletion_or_insertion, if_substitution);

            // Export results.
            scores_new[i] = cell_score;
            scores_new_insertions[i] = if_insertion;
            scores_new_deletions[i] = if_deletion;
        }

        // The last element of the last chunk is the result of the global alignment.
        if (tasks_offset == 0) final_score_ = scores_new[0];
    }
};

/**
 *  @brief GPU adaptation of the `local_scorer` on CUDA, avoiding warp-level shuffles and DPX.
 *  @note Uses 32-bit `unsigned` counter to iterate through the string slices, so it can't be over 4 billion characters.
 */
template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_, sz_capability_t capability_>
#if SZ_HAS_CONCEPTS_
    requires pointer_like<first_iterator_type_> && pointer_like<second_iterator_type_> && score_like<score_type_> &&
             substituter_like<substituter_type_>
#endif
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, affine_gap_costs_t,
                   objective_, sz_similarity_local_k, capability_, std::enable_if_t<capability_ == sz_cap_cuda_k>> {

    using first_iterator_t = first_iterator_type_;
    using second_iterator_t = second_iterator_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    using first_char_t = typename std::iterator_traits<first_iterator_t>::value_type;
    using second_char_t = typename std::iterator_traits<second_iterator_t>::value_type;
    static_assert(is_same_type<first_char_t, second_char_t>::value, "String characters must be of the same type.");
    using char_t = remove_cvref<first_char_t>;

    using cuda_warp_scorer_t = tile_scorer<first_iterator_t, second_iterator_t, score_t, substituter_t,
                                           affine_gap_costs_t, objective_k, sz_similarity_local_k, capability_k>;

  protected:
    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};
    score_t final_score_ {0};

  public:
    __forceinline__ __device__ tile_scorer(substituter_t subs, affine_gap_costs_t gaps) noexcept
        : substituter_(subs), gap_costs_(gaps) {}

    /**
     *  @brief Initializes a boundary value within a certain diagonal.
     *  @note Should only be called for the diagonals outside of the bottom-right triangle.
     *  @note Should only be called for the top row and left column of the matrix.
     */
    __forceinline__ __device__ void init_score(score_t &cell, size_t diagonal_index) const noexcept { cell = 0; }
    __forceinline__ __device__ void init_gap(score_t &cell, size_t /* diagonal_index */) const noexcept {
        // Make sure the initial value of the gap is not smaller in magnitude than the primary.
        // The supplementary matrices are initialized with values of higher magnitude,
        // which is equivalent to discarding them. That's better than using `SIZE_MAX`
        // as subsequent additions won't overflow.
        cell = gap_costs_.open + gap_costs_.extend;
    }

    /**
     *  @brief Extract the final result of the scoring operation which will be always in the bottom-right corner.
     */
    __forceinline__ __device__ score_t score() const noexcept { return final_score_; }

    /**
     *  @brief Computes one diagonal of the DP matrix, using the results of the previous 2x diagonals.
     *  @param first_slice The first string, unlike the CPU variant @b NOT reversed.
     *  @param second_slice The second string.
     *
     *  @param tasks_offset The offset of the first character to compare from each string.
     *  @param tasks_step The step size for the next character to compare from each string.
     *  @param tasks_count The total number of characters to compare from input slices.
     *
     *  @tparam index_type_ @b `unsigned` is recommended if the strings are under 4 billion characters.
     */
    template <typename index_type_>
    __forceinline__ __device__ void operator()(                                                      //
        first_iterator_t first_slice, second_iterator_t second_slice,                                //
        index_type_ const tasks_offset, index_type_ const tasks_step, index_type_ const tasks_count, //
        score_t const *scores_pre_substitution,                                                      //
        score_t const *scores_pre_insertion,                                                         //
        score_t const *scores_pre_deletion,                                                          //
        score_t const *scores_running_insertions,                                                    //
        score_t const *scores_running_deletions,                                                     //
        score_t *scores_new,                                                                         //
        score_t *scores_new_insertions,                                                              //
        score_t *scores_new_deletions) noexcept {

        // Make sure we are called for an anti-diagonal traversal order
        sz_assert_(scores_pre_insertion + 1 == scores_pre_deletion);

        // ? One weird observation, is that even though we can avoid fetching `pre_insertion`
        // ? from shared memory on each cycle, by slicing the work differently between the threads,
        // ? and allowing them to reuse the previous `pre_deletion` as the new `pre_insertion`,
        // ? that code ends up being slower than the one below.
        for (index_type_ i = tasks_offset; i < tasks_count; i += tasks_step) {
            score_t pre_substitution = load_last_use_(scores_pre_substitution + i);
            score_t pre_insertion_opening = scores_pre_insertion[i];
            score_t pre_deletion_opening = scores_pre_deletion[i];
            score_t pre_insertion_expansion = scores_running_insertions[i];
            score_t pre_deletion_expansion = scores_running_deletions[i];
            char first_char = load_immutable_(first_slice + tasks_count - i - 1);
            char second_char = load_immutable_(second_slice + i);

            error_cost_t cost_of_substitution = substituter_(first_char, second_char);
            score_t if_substitution = pre_substitution + cost_of_substitution;
            score_t if_deletion = min_or_max<objective_k>(pre_deletion_opening + gap_costs_.open,
                                                          pre_deletion_expansion + gap_costs_.extend);
            score_t if_insertion = min_or_max<objective_k>(pre_insertion_opening + gap_costs_.open,
                                                           pre_insertion_expansion + gap_costs_.extend);
            score_t if_deletion_or_insertion = min_or_max<objective_k>(if_deletion, if_insertion);
            score_t if_substitution_or_reset = pick_best_<objective_k, score_t>(if_substitution, 0);
            score_t cell_score = pick_best_<objective_k>(if_deletion_or_insertion, if_substitution_or_reset);

            // Export results.
            scores_new[i] = cell_score;
            scores_new_insertions[i] = if_insertion;
            scores_new_deletions[i] = if_deletion;

            // Update the global maximum score if this cell beats it.
            final_score_ = pick_best_<objective_k>(final_score_, cell_score);
        }

        // ! Don't forget to pick the best among the best scores per thread.
        final_score_ = pick_best_in_warp_<objective_k>(final_score_);
    }
};

/**
 *  @brief String similarity scoring algorithm evaluating a @b single Dynamic Programming matrix
 *          @b three skewed (reverse) diagonals at a time on a GPU, leveraging CUDA for parallelization.
 *          Unlike the `_levenshtein_in_cuda_warp` is designed to take one pair of very-longs string,
 *          ideally @b Tens-of-Megabytes in size or more.
 *
 *  @param[in] shorter_string The shorter string in the pair for score calculation.
 *  @param[in] longer_string The longer string in the pair for score calculation.
 *  @param[out] result_ptr Output address of the score for the pair of strings.
 *
 *  The ideal plan is:
 *  - Use cooperative groups abstractions for grid level synchronization between iterations.
 *  - Keep 3 diagonals in shared memory, but not the texts, as depending on the diagonal, different
 *    characters will be needed. Asynchronous copy of the characters from global memory to shared memory
 *    will help hide that latency.
 *  - Each block of threads takes its own slice of those 3 diagonals, constrained by the amount of shared
 *    memory available on the device.
 *  - Every neighboring pair of diagonals has a different length, and boundary elements of each slice need
 *    to be exchanged through shared memory.
 *
 *  The current starter plan is much simpler:
 *  - Keep everything in global memory - the strings and the diagonals.
 *  - Execute the naive algorithm, expecting the hardware to handle coalescing the memory accesses.
 */
template <                                                       //
    typename char_type_ = char,                                  //
    typename index_type_ = unsigned,                             //
    typename score_type_ = size_t,                               //
    typename final_score_type_ = size_t,                         //
    typename substituter_type_ = uniform_substitution_costs_t,   //
    sz_similarity_objective_t objective_ = sz_maximize_score_k,  //
    sz_similarity_locality_t locality_ = sz_similarity_global_k, //
    sz_capability_t capability_ = sz_cap_cuda_k                  //
    >
__global__ void linear_score_across_cuda_device_(              //
    char_type_ const *shorter_ptr, index_type_ shorter_length, //
    char_type_ const *longer_ptr, index_type_ longer_length,   //
    final_score_type_ *result_ptr, score_type_ *diagonals_ptr, //
    substituter_type_ const substituter, linear_gap_costs_t const gap_costs) {

    namespace cg = cooperative_groups;

    sz_assert_(shorter_length > 0);
    sz_assert_(longer_length > 0);
    sz_assert_(shorter_length <= longer_length);
    using char_t = char_type_;
    using index_t = index_type_;
    using score_t = score_type_;
    using final_score_t = final_score_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_similarity_objective_t objective_k = objective_;

    // Pre-load the substituter and gap costs.
    using substituter_t = substituter_type_;
    using gap_costs_t = linear_gap_costs_t;
    static_assert(std::is_trivially_copyable<substituter_t>::value, "Substituter must be trivially copyable.");
    static_assert(std::is_trivially_copyable<gap_costs_t>::value, "Gap costs must be trivially copyable.");

    using cuda_warp_scorer_t = tile_scorer<char_t const *, char_t const *, score_t, substituter_t, gap_costs_t,
                                           objective_k, locality_k, capability_k>;

    // Cooperatively copy the substitution costs into static shared memory for the inner-loop lookups.
    substituter_t const substituter_shared = load_substituter_into_shared_(substituter);

    // Only one thread will be initializing the top row and left column and outputting the result.
    bool const is_main_thread = blockIdx.x == 0 && threadIdx.x == 0;

    // We are going to store 3 diagonals of the matrix, assuming each would fit into a single ZMM register.
    // The length of the longest (main) diagonal would be `shorter_dim = (shorter_length + 1)`.
    index_t const shorter_dim = shorter_length + 1, longer_dim = longer_length + 1;

    // Let's say we are dealing with 3 and 5 letter words.
    // The matrix will have size 4 x 6, parameterized as (shorter_dim x longer_dim).
    // It will have:
    // - 4 diagonals of increasing length, at positions: 0, 1, 2, 3.
    // - 2 diagonals of fixed length, at positions: 4, 5.
    // - 3 diagonals of decreasing length, at positions: 6, 7, 8.
    index_t const diagonals_count = shorter_dim + longer_dim - 1;
    index_t const max_diagonal_length = shorter_length + 1;

    // The next few pointers will be swapped around.
    score_t *previous_scores = diagonals_ptr;
    score_t *current_scores = diagonals_ptr + max_diagonal_length;
    score_t *next_scores = diagonals_ptr + 2 * max_diagonal_length;

    // Initialize the first two diagonals:
    cuda_warp_scorer_t diagonal_aligner {substituter_shared, gap_costs};
    if (is_main_thread) {
        diagonal_aligner.init_score(previous_scores[0], 0);
        diagonal_aligner.init_score(current_scores[0], 1);
        diagonal_aligner.init_score(current_scores[1], 1);
    }

    cg::grid_group grid = cg::this_grid();

    // We skip diagonals 0 and 1, as they are trivial.
    // We will start with diagonal 2, which has length 3, with the first and last elements being preset,
    // so we are effectively computing just one value, as will be marked by a single set bit in
    // the `next_diagonal_mask` on the very first iteration.
    index_t next_diagonal_index = 2;
    index_t const global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t const global_thread_step = blockDim.x * gridDim.x;

    // Progress through the upper-left triangle of the Levenshtein matrix.
    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {

        index_t const next_diagonal_length = next_diagonal_index + 1;
        diagonal_aligner(                            //
            shorter_ptr,                             // first sequence of characters
            longer_ptr,                              // second sequence of characters
            global_thread_index, global_thread_step, //
            (index_t)(next_diagonal_length - 2),     // number of elements to compute with the `diagonal_aligner`
            previous_scores,                         // costs pre substitution
            current_scores, current_scores + 1,      // costs pre insertion/deletion
            next_scores + 1);                        // ! notice unaligned write destination

        // Don't forget to populate the first row and the first column of the Levenshtein matrix.
        if (is_main_thread) {
            diagonal_aligner.init_score(next_scores[0], next_diagonal_index);
            diagonal_aligner.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);
        }
        // Guarantee that all the writes have finished, before progressing to the next diagonal.
        grid.sync();

        // Perform a circular rotation of those buffers, to reuse the memory.
        rotate_three(previous_scores, current_scores, next_scores);
    }

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_system, 2> memcpy_pipeline_state;
    auto memcpy_pipeline = cuda::make_pipeline(grid, &memcpy_pipeline_state);

    // Now let's handle the anti-diagonal band of the matrix, between the top and bottom-right triangles.
    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {

        index_t const next_diagonal_length = shorter_dim;
        diagonal_aligner(                                   //
            shorter_ptr,                                    // first sequence of characters
            longer_ptr + next_diagonal_index - shorter_dim, // second sequence of characters
            global_thread_index, global_thread_step,        //
            (index_t)(next_diagonal_length - 1),            // number of elements to compute with the `diagonal_aligner`
            previous_scores,                                // costs pre substitution
            current_scores, current_scores + 1,             // costs pre insertion/deletion
            next_scores);

        // Don't forget to populate the first row of the Levenshtein matrix.
        if (is_main_thread) diagonal_aligner.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);

        // Guarantee that all the writes have finished, before progressing to the next diagonal.
        grid.sync();

        // ! In the central anti-diagonal band, we can't just set the `current_scores + 1` to `previous_scores`
        // ! for the circular shift, as we will end up spilling outside of the diagonal a few iterations later.
        // ! Assuming in-place `memmove` is tricky on the GPU, so we will copy the data.
        memcpy_pipeline.producer_acquire();
        cuda::memcpy_async(grid, (void *)previous_scores, (void const *)(current_scores + 1),
                           (next_diagonal_length - 1) * sizeof(score_t), memcpy_pipeline);
        cuda::memcpy_async(grid, (void *)current_scores, (void const *)(next_scores),
                           (next_diagonal_length) * sizeof(score_t), memcpy_pipeline);
        memcpy_pipeline.producer_commit();
        memcpy_pipeline.consumer_wait();
        memcpy_pipeline.consumer_release();
    }

    // Now let's handle the bottom-right triangle of the matrix.
    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {

        index_t const next_diagonal_length = diagonals_count - next_diagonal_index;
        diagonal_aligner(                                   //
            shorter_ptr + next_diagonal_index - longer_dim, // first sequence of characters
            longer_ptr + next_diagonal_index - shorter_dim, // second sequence of characters
            global_thread_index, global_thread_step,        //
            next_diagonal_length,                           // number of elements to compute with the `diagonal_aligner`
            previous_scores,                                // costs pre substitution
            current_scores, current_scores + 1,             // costs pre insertion/deletion
            next_scores);

        // Guarantee that all the writes have finished, before progressing to the next diagonal.
        grid.sync();

        // Perform a circular rotation of those buffers, to reuse the memory.
        rotate_three(previous_scores, current_scores, next_scores);

        // ! Drop the first entry among the current scores.
        // ! Assuming every next diagonal is shorter by one element,
        // ! we don't need a full-blown `sz_move` to shift the array by one element.
        previous_scores++;
    }

    // Export one result per each block.
    if (is_main_thread) *result_ptr = static_cast<final_score_t>(diagonal_aligner.score());
}

/**
 *  @brief String similarity scoring algorithm evaluating a @b single Dynamic Programming matrix
 *          @b three skewed (reverse) diagonals at a time on a GPU, leveraging CUDA for parallelization.
 *          Unlike the `_levenshtein_in_cuda_warp` is designed to take one pair of very-longs string,
 *          ideally @b Tens-of-Megabytes in size or more.
 *
 *  @param[in] shorter_string The shorter string in the pair for score calculation.
 *  @param[in] longer_string The longer string in the pair for score calculation.
 *  @param[out] result_ptr Output address of the score for the pair of strings.
 */
template <                                                       //
    typename char_type_ = char,                                  //
    typename index_type_ = unsigned,                             //
    typename score_type_ = size_t,                               //
    typename final_score_type_ = size_t,                         //
    typename substituter_type_ = uniform_substitution_costs_t,   //
    sz_similarity_objective_t objective_ = sz_maximize_score_k,  //
    sz_similarity_locality_t locality_ = sz_similarity_global_k, //
    sz_capability_t capability_ = sz_cap_cuda_k                  //
    >
__global__ void affine_score_across_cuda_device(               //
    char_type_ const *shorter_ptr, index_type_ shorter_length, //
    char_type_ const *longer_ptr, index_type_ longer_length,   //
    final_score_type_ *result_ptr, score_type_ *diagonals_ptr, //
    substituter_type_ const substituter, affine_gap_costs_t const gap_costs) {

    namespace cg = cooperative_groups;

    sz_assert_(shorter_length > 0);
    sz_assert_(longer_length > 0);
    sz_assert_(shorter_length <= longer_length);
    using char_t = char_type_;
    using index_t = index_type_;
    using score_t = score_type_;
    using final_score_t = final_score_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_similarity_objective_t objective_k = objective_;

    // Pre-load the substituter and gap costs.
    using substituter_t = substituter_type_;
    using gap_costs_t = affine_gap_costs_t;
    static_assert(std::is_trivially_copyable<substituter_t>::value, "Substituter must be trivially copyable.");
    static_assert(std::is_trivially_copyable<gap_costs_t>::value, "Gap costs must be trivially copyable.");

    using cuda_warp_scorer_t = tile_scorer<char_t const *, char_t const *, score_t, substituter_t, gap_costs_t,
                                           objective_k, locality_k, capability_k>;

    // Cooperatively copy the substitution costs into static shared memory for the inner-loop lookups.
    substituter_t const substituter_shared = load_substituter_into_shared_(substituter);

    // Only one thread will be initializing the top row and left column and outputting the result.
    bool const is_main_thread = blockIdx.x == 0 && threadIdx.x == 0; // ! Differs for warp-wide

    // We are going to store 3 diagonals of the matrix, assuming each would fit into a single ZMM register.
    // The length of the longest (main) diagonal would be `shorter_dim = (shorter_length + 1)`.
    index_t const shorter_dim = shorter_length + 1, longer_dim = longer_length + 1;

    // Let's say we are dealing with 3 and 5 letter words.
    // The matrix will have size 4 x 6, parameterized as (shorter_dim x longer_dim).
    // It will have:
    // - 4 diagonals of increasing length, at positions: 0, 1, 2, 3.
    // - 2 diagonals of fixed length, at positions: 4, 5.
    // - 3 diagonals of decreasing length, at positions: 6, 7, 8.
    index_t const diagonals_count = shorter_dim + longer_dim - 1;
    index_t const max_diagonal_length = shorter_length + 1;

    // The next few pointers will be swapped around.
    score_t *previous_scores = diagonals_ptr;
    score_t *current_scores = diagonals_ptr + max_diagonal_length;
    score_t *next_scores = diagonals_ptr + 2 * max_diagonal_length;
    score_t *current_inserts = diagonals_ptr + 3 * max_diagonal_length;
    score_t *next_inserts = diagonals_ptr + 4 * max_diagonal_length;
    score_t *current_deletes = diagonals_ptr + 5 * max_diagonal_length;
    score_t *next_deletes = diagonals_ptr + 6 * max_diagonal_length;

    // Initialize the first two diagonals:
    cuda_warp_scorer_t diagonal_aligner {substituter_shared, gap_costs};
    if (is_main_thread) {
        diagonal_aligner.init_score(previous_scores[0], 0);
        diagonal_aligner.init_score(current_scores[0], 1);
        diagonal_aligner.init_score(current_scores[1], 1);
        diagonal_aligner.init_gap(current_inserts[0], 1);
        diagonal_aligner.init_gap(current_deletes[1], 1);
    }

    cg::grid_group grid = cg::this_grid();

    // We skip diagonals 0 and 1, as they are trivial.
    // We will start with diagonal 2, which has length 3, with the first and last elements being preset,
    // so we are effectively computing just one value, as will be marked by a single set bit in
    // the `next_diagonal_mask` on the very first iteration.
    index_t next_diagonal_index = 2;
    index_t const global_thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t const global_thread_step = blockDim.x * gridDim.x;

    // Progress through the upper-left triangle of the Levenshtein matrix.
    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {

        index_t const next_diagonal_length = next_diagonal_index + 1;
        diagonal_aligner(                            //
            shorter_ptr,                             // first sequence of characters
            longer_ptr,                              // second sequence of characters
            global_thread_index, global_thread_step, //
            (index_t)(next_diagonal_length - 2),     // number of elements to compute with the `diagonal_aligner`
            previous_scores,                         // costs pre substitution
            current_scores, current_scores + 1,      // costs pre insertion/deletion opening
            current_inserts, current_deletes + 1,    // costs pre insertion/deletion extension
            next_scores + 1,                         // ! notice unaligned write destination
            next_inserts + 1, next_deletes + 1       // ! notice unaligned write destination
        );

        // Don't forget to populate the first row and the first column of the Levenshtein matrix.
        if (is_main_thread) {
            diagonal_aligner.init_score(next_scores[0], next_diagonal_index);
            diagonal_aligner.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);
            diagonal_aligner.init_gap(next_inserts[0], next_diagonal_index);
            diagonal_aligner.init_gap(next_deletes[next_diagonal_length - 1], next_diagonal_index);
        }
        // Guarantee that all the writes have finished, before progressing to the next diagonal.
        grid.sync();

        // Perform a circular rotation of those buffers, to reuse the memory.
        rotate_three(previous_scores, current_scores, next_scores);
        trivial_swap(current_inserts, next_inserts);
        trivial_swap(current_deletes, next_deletes);
    }

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_system, 2> memcpy_pipeline_state;
    auto memcpy_pipeline = cuda::make_pipeline(grid, &memcpy_pipeline_state);

    // Now let's handle the anti-diagonal band of the matrix, between the top and bottom-right triangles.
    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {

        index_t const next_diagonal_length = shorter_dim;
        diagonal_aligner(                                   //
            shorter_ptr,                                    // first sequence of characters
            longer_ptr + next_diagonal_index - shorter_dim, // second sequence of characters
            global_thread_index, global_thread_step,        //
            (index_t)(next_diagonal_length - 1),            // number of elements to compute with the `diagonal_aligner`
            previous_scores,                                // costs pre substitution
            current_scores, current_scores + 1,             // costs pre insertion/deletion opening
            current_inserts, current_deletes + 1,           // costs pre insertion/deletion extension
            next_scores,                                    // updated similarity scores
            next_inserts, next_deletes                      // updated insertion/deletion extensions
        );

        // Don't forget to populate the first row of the Levenshtein matrix.
        if (is_main_thread) {
            diagonal_aligner.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);
            diagonal_aligner.init_gap(next_deletes[next_diagonal_length - 1], next_diagonal_index);
        }

        trivial_swap(current_inserts, next_inserts);
        trivial_swap(current_deletes, next_deletes);

        // Guarantee that all the writes have finished, before progressing to the next diagonal.
        grid.sync();

        // ! In the central anti-diagonal band, we can't just set the `current_scores + 1` to `previous_scores`
        // ! for the circular shift, as we will end up spilling outside of the diagonal a few iterations later.
        // ! Assuming in-place `memmove` is tricky on the GPU, so we will copy the data.
        memcpy_pipeline.producer_acquire();
        cuda::memcpy_async(grid, (void *)previous_scores, (void const *)(current_scores + 1),
                           (next_diagonal_length - 1) * sizeof(score_t), memcpy_pipeline);
        cuda::memcpy_async(grid, (void *)current_scores, (void const *)(next_scores),
                           (next_diagonal_length) * sizeof(score_t), memcpy_pipeline);
        memcpy_pipeline.producer_commit();
        memcpy_pipeline.consumer_wait();
        memcpy_pipeline.consumer_release();
    }

    // Now let's handle the bottom-right triangle of the matrix.
    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {

        index_t const next_diagonal_length = diagonals_count - next_diagonal_index;
        diagonal_aligner(                                   //
            shorter_ptr + next_diagonal_index - longer_dim, // first sequence of characters
            longer_ptr + next_diagonal_index - shorter_dim, // second sequence of characters
            global_thread_index, global_thread_step,        //
            next_diagonal_length,                           // number of elements to compute with the `diagonal_aligner`
            previous_scores,                                // costs pre substitution
            current_scores, current_scores + 1,             // costs pre insertion/deletion opening
            current_inserts, current_deletes + 1,           // costs pre insertion/deletion extension
            next_scores,                                    // updated similarity scores
            next_inserts, next_deletes                      // updated insertion/deletion extensions
        );

        // Guarantee that all the writes have finished, before progressing to the next diagonal.
        grid.sync();

        // Perform a circular rotation of those buffers, to reuse the memory.
        rotate_three(previous_scores, current_scores, next_scores);
        trivial_swap(current_inserts, next_inserts);
        trivial_swap(current_deletes, next_deletes);

        // ! Drop the first entry among the current scores.
        // ! Assuming every next diagonal is shorter by one element,
        // ! we don't need a full-blown `sz_move` to shift the array by one element.
        previous_scores++;
    }

    // Export one result per each block.
    if (is_main_thread) *result_ptr = static_cast<final_score_t>(diagonal_aligner.score());
}

/**
 *  @brief Levenshtein edit distances algorithm evaluating the Dynamic Programming matrix
 *          @b three skewed (reverse) diagonals at a time on a GPU, leveraging CUDA for parallelization.
 *          Each pair of strings gets its own @b "block" of CUDA threads forming one @b warp and shared memory.
 *
 *  @param[in] tasks Tasks containing the strings and output locations.
 *  @param[in] tasks_count The number of tasks to process.
 *  @param[in] substituter The substitution costs.
 *  @param[in] gap_costs The @b linear gap costs.
 */
template < //
    typename task_type_,
    typename char_type_ = char,                                  //
    typename index_type_ = unsigned,                             //
    typename score_type_ = size_t,                               //
    typename substituter_type_ = uniform_substitution_costs_t,   //
    sz_similarity_objective_t objective_ = sz_maximize_score_k,  //
    sz_similarity_locality_t locality_ = sz_similarity_global_k, //
    sz_capability_t capability_ = sz_cap_cuda_k                  //
    >
__global__ void linear_score_on_each_cuda_warp_(                             //
    task_type_ *tasks, size_t tasks_count,                                   //
    substituter_type_ const substituter, linear_gap_costs_t const gap_costs, //
    unsigned const shared_memory_size) {

    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    using task_t = task_type_;
    using char_t = char_type_;
    using index_t = index_type_;
    using score_t = score_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_similarity_objective_t objective_k = objective_;

    // Pre-load the substituter and gap costs.
    using substituter_t = substituter_type_;
    using gap_costs_t = linear_gap_costs_t;
    static_assert(std::is_trivially_copyable<substituter_t>::value, "Substituter must be trivially copyable.");
    static_assert(std::is_trivially_copyable<gap_costs_t>::value, "Gap costs must be trivially copyable.");

    using cuda_warp_scorer_t = tile_scorer<char_t const *, char_t const *, score_t, substituter_t, gap_costs_t,
                                           objective_k, locality_k, capability_k>;

    // We may have multiple warps operating in the same block.
    unsigned const warp_size = warpSize;
    unsigned const global_thread_index = static_cast<unsigned>(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned const global_warp_index = static_cast<unsigned>(global_thread_index / warp_size);
    unsigned const warps_per_block = static_cast<unsigned>(blockDim.x / warp_size);
    unsigned const warps_per_device = static_cast<unsigned>(gridDim.x * warps_per_block);
    unsigned const thread_in_warp_index = static_cast<unsigned>(global_thread_index % warp_size);

    // Allocating shared memory is handled on the host side.
    extern __shared__ char shared_memory_for_block[];
    char *const shared_memory_for_warp = shared_memory_for_block +
                                         (global_warp_index % warps_per_block) * (shared_memory_size / warps_per_block);

    // Only one thread will be initializing the top row and left column and outputting the result.
    bool const is_main_thread = thread_in_warp_index == 0;

    // Cooperatively copy the substitution costs into static shared memory for the inner-loop lookups.
    substituter_t const substituter_shared = load_substituter_into_shared_(substituter);

    // We are computing N edit distances for N pairs of strings. Not a cartesian product!
    // Each block/warp may end up receiving a different number of strings.
    for (size_t task_idx = global_warp_index; task_idx < tasks_count; task_idx += warps_per_device) {
        task_t &task = tasks[task_idx];
        char_t const *shorter_global = task.shorter_ptr;
        char_t const *longer_global = task.longer_ptr;
        u32_t const shorter_length = static_cast<u32_t>(task.shorter_length);
        u32_t const longer_length = static_cast<u32_t>(task.longer_length);
        auto &result_ref = task.result;

        // We are going to store 3 diagonals of the matrix, assuming each would fit into a single ZMM register.
        // The length of the longest (main) diagonal would be `shorter_dim = (shorter_length + 1)`.
        unsigned const shorter_dim = static_cast<unsigned>(shorter_length + 1);
        unsigned const longer_dim = static_cast<unsigned>(longer_length + 1);

        // Let's say we are dealing with 3 and 5 letter words.
        // The matrix will have size 4 x 6, parameterized as (shorter_dim x longer_dim).
        // It will have:
        // - 4 diagonals of increasing length, at positions: 0, 1, 2, 3.
        // - 2 diagonals of fixed length, at positions: 4, 5.
        // - 3 diagonals of decreasing length, at positions: 6, 7, 8.
        unsigned const diagonals_count = shorter_dim + longer_dim - 1;
        unsigned const max_diagonal_length = shorter_length + 1;
        unsigned const bytes_per_diagonal = round_up_to_multiple<unsigned>(max_diagonal_length * sizeof(score_t), 4);

        // The next few pointers will be swapped around.
        score_t *previous_scores = reinterpret_cast<score_t *>(shared_memory_for_warp);
        score_t *current_scores = reinterpret_cast<score_t *>(shared_memory_for_warp + bytes_per_diagonal);
        score_t *next_scores = reinterpret_cast<score_t *>(shared_memory_for_warp + 2 * bytes_per_diagonal);

        // Read the strings straight from global memory (L1-cached) instead of staging them in shared, freeing
        // ~(longer+shorter) bytes/warp of shared memory for higher occupancy. The small strings stay hot in L1.
        char_t const *const longer = longer_global;
        char_t const *const shorter = shorter_global;

        // Initialize the first two diagonals:
        cuda_warp_scorer_t diagonal_aligner {substituter_shared, gap_costs};
        if (is_main_thread) {
            diagonal_aligner.init_score(previous_scores[0], 0);
            diagonal_aligner.init_score(current_scores[0], 1);
            diagonal_aligner.init_score(current_scores[1], 1);
        }

        // Make sure the shared memory is fully loaded.
        __syncwarp();

        // We skip diagonals 0 and 1, as they are trivial.
        // We will start with diagonal 2, which has length 3, with the first and last elements being preset,
        // so we are effectively computing just one value, as will be marked by a single set bit in
        // the `next_diagonal_mask` on the very first iteration.
        unsigned next_diagonal_index = 2;

        // Progress through the upper-left triangle of the Levenshtein matrix.
        for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {

            unsigned const next_diagonal_length = next_diagonal_index + 1;
            diagonal_aligner(                       //
                shorter,                            // first sequence of characters
                longer,                             // second sequence of characters
                thread_in_warp_index, warp_size,    //
                next_diagonal_length - 2,           // number of elements to compute with the `diagonal_aligner`
                previous_scores,                    // costs pre substitution
                current_scores, current_scores + 1, // costs pre insertion/deletion
                next_scores + 1);                   // ! notice unaligned write destination

            // Don't forget to populate the first row and the first column of the Levenshtein matrix.
            if (is_main_thread) {
                diagonal_aligner.init_score(next_scores[0], next_diagonal_index);
                diagonal_aligner.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);
            }
            __syncwarp();

            // Perform a circular rotation of those buffers, to reuse the memory.
            rotate_three(previous_scores, current_scores, next_scores);
        }

        // Now let's handle the anti-diagonal band of the matrix, between the top and bottom-right triangles.
        for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {

            unsigned const next_diagonal_length = shorter_dim;
            diagonal_aligner(                               //
                shorter,                                    // first sequence of characters
                longer + next_diagonal_index - shorter_dim, // second sequence of characters
                thread_in_warp_index, warp_size,            //
                next_diagonal_length - 1,                   // number of elements to compute with the `diagonal_aligner`
                previous_scores,                            // costs pre substitution
                current_scores, current_scores + 1,         // costs pre insertion/deletion
                next_scores);

            // Don't forget to populate the first row of the Levenshtein matrix.
            if (is_main_thread) diagonal_aligner.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);

            __syncwarp();
            // ! In the central anti-diagonal band, we can't just set the `current_scores + 1` to `previous_scores`
            // ! for the circular shift, as we will end up spilling outside of the diagonal a few iterations later.
            // ! Assuming in-place `memmove` is tricky on the GPU, so we will copy the data.
            for (unsigned i = thread_in_warp_index; i + 1 < next_diagonal_length; i += warp_size)
                previous_scores[i] = current_scores[i + 1];
            __syncwarp();
            for (unsigned i = thread_in_warp_index; i < next_diagonal_length; i += warp_size)
                current_scores[i] = next_scores[i];
            __syncwarp();
        }

        // Now let's handle the bottom-right triangle of the matrix.
        for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {

            unsigned const next_diagonal_length = diagonals_count - next_diagonal_index;
            diagonal_aligner(                               //
                shorter + next_diagonal_index - longer_dim, // first sequence of characters
                longer + next_diagonal_index - shorter_dim, // second sequence of characters
                thread_in_warp_index, warp_size,            //
                next_diagonal_length,                       // number of elements to compute with the `diagonal_aligner`
                previous_scores,                            // costs pre substitution
                current_scores, current_scores + 1,         // costs pre insertion/deletion
                next_scores);

            // Perform a circular rotation of those buffers, to reuse the memory.
            rotate_three(previous_scores, current_scores, next_scores);

            // ! Drop the first entry among the current scores.
            // ! Assuming every next diagonal is shorter by one element,
            // ! we don't need a full-blown `sz_move` to shift the array by one element.
            previous_scores++;
            __syncwarp();
        }

        // Export one result per each block.
        if (is_main_thread) result_ref = diagonal_aligner.score();
    }
}

/**
 *  @brief Levenshtein edit distances algorithm evaluating the Dynamic Programming matrix
 *          @b three skewed (reverse) diagonals at a time on a GPU, leveraging CUDA for parallelization.
 *          Each pair of strings gets its own @b "block" of CUDA threads forming one @b warp and shared memory.
 *
 *  @param[in] tasks Tasks containing the strings and output locations.
 *  @param[in] tasks_count The number of tasks to process.
 *  @param[in] substituter The substitution costs.
 *  @param[in] gap_costs The @b affine gap costs.
 */
template < //
    typename task_type_,
    typename char_type_ = char,                                  //
    typename index_type_ = unsigned,                             //
    typename score_type_ = size_t,                               //
    typename substituter_type_ = uniform_substitution_costs_t,   //
    sz_similarity_objective_t objective_ = sz_maximize_score_k,  //
    sz_similarity_locality_t locality_ = sz_similarity_global_k, //
    sz_capability_t capability_ = sz_cap_cuda_k                  //
    >
__global__ void affine_score_on_each_cuda_warp_(                             //
    task_type_ *tasks, size_t tasks_count,                                   //
    substituter_type_ const substituter, affine_gap_costs_t const gap_costs, //
    unsigned const shared_memory_size) {

    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    using task_t = task_type_;
    using char_t = char_type_;
    using index_t = index_type_;
    using score_t = score_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_similarity_objective_t objective_k = objective_;

    // Pre-load the substituter and gap costs.
    using substituter_t = substituter_type_;
    using gap_costs_t = affine_gap_costs_t;
    static_assert(std::is_trivially_copyable<substituter_t>::value, "Substituter must be trivially copyable.");
    static_assert(std::is_trivially_copyable<gap_costs_t>::value, "Gap costs must be trivially copyable.");

    using cuda_warp_scorer_t = tile_scorer<char_t const *, char_t const *, score_t, substituter_t, gap_costs_t,
                                           objective_k, locality_k, capability_k>;

    // We may have multiple warps operating in the same block.
    unsigned const warp_size = warpSize;
    unsigned const global_thread_index = static_cast<unsigned>(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned const global_warp_index = static_cast<unsigned>(global_thread_index / warp_size);
    unsigned const warps_per_block = static_cast<unsigned>(blockDim.x / warp_size);
    unsigned const warps_per_device = static_cast<unsigned>(gridDim.x * warps_per_block);
    unsigned const thread_in_warp_index = static_cast<unsigned>(global_thread_index % warp_size);

    // Allocating shared memory is handled on the host side.
    extern __shared__ char shared_memory_for_block[];
    char *const shared_memory_for_warp = shared_memory_for_block +
                                         (global_warp_index % warps_per_block) * (shared_memory_size / warps_per_block);

    // Only one thread will be initializing the top row and left column and outputting the result.
    bool const is_main_thread = thread_in_warp_index == 0;

    // Cooperatively copy the substitution costs into static shared memory for the inner-loop lookups.
    substituter_t const substituter_shared = load_substituter_into_shared_(substituter);

    // We are computing N edit distances for N pairs of strings. Not a cartesian product!
    // Each block/warp may end up receiving a different number of strings.
    for (size_t task_idx = global_warp_index; task_idx < tasks_count; task_idx += warps_per_device) {
        task_t &task = tasks[task_idx];
        char_t const *shorter_global = task.shorter_ptr;
        char_t const *longer_global = task.longer_ptr;
        u32_t const shorter_length = static_cast<u32_t>(task.shorter_length);
        u32_t const longer_length = static_cast<u32_t>(task.longer_length);
        auto &result_ref = task.result;

        // We are going to store 3 diagonals of the matrix, assuming each would fit into a single ZMM register.
        // The length of the longest (main) diagonal would be `shorter_dim = (shorter_length + 1)`.
        unsigned const shorter_dim = static_cast<unsigned>(shorter_length + 1);
        unsigned const longer_dim = static_cast<unsigned>(longer_length + 1);

        // Let's say we are dealing with 3 and 5 letter words.
        // The matrix will have size 4 x 6, parameterized as (shorter_dim x longer_dim).
        // It will have:
        // - 4 diagonals of increasing length, at positions: 0, 1, 2, 3.
        // - 2 diagonals of fixed length, at positions: 4, 5.
        // - 3 diagonals of decreasing length, at positions: 6, 7, 8.
        unsigned const diagonals_count = shorter_dim + longer_dim - 1;
        unsigned const max_diagonal_length = shorter_length + 1;
        unsigned const bytes_per_diagonal = round_up_to_multiple<unsigned>(max_diagonal_length * sizeof(score_t), 4);

        // The next few pointers will be swapped around.
        score_t *previous_scores = reinterpret_cast<score_t *>(shared_memory_for_warp);
        score_t *current_scores = reinterpret_cast<score_t *>(shared_memory_for_warp + bytes_per_diagonal);
        score_t *next_scores = reinterpret_cast<score_t *>(shared_memory_for_warp + 2 * bytes_per_diagonal);
        score_t *current_inserts = reinterpret_cast<score_t *>(shared_memory_for_warp + 3 * bytes_per_diagonal);
        score_t *next_inserts = reinterpret_cast<score_t *>(shared_memory_for_warp + 4 * bytes_per_diagonal);
        score_t *current_deletes = reinterpret_cast<score_t *>(shared_memory_for_warp + 5 * bytes_per_diagonal);
        score_t *next_deletes = reinterpret_cast<score_t *>(shared_memory_for_warp + 6 * bytes_per_diagonal);
        // Read the strings straight from global memory (L1-cached) instead of staging them in shared, freeing
        // ~(longer+shorter) bytes/warp of shared memory for higher occupancy. The small strings stay hot in L1.
        char_t const *const longer = longer_global;
        char_t const *const shorter = shorter_global;

        // Initialize the first two diagonals:
        cuda_warp_scorer_t diagonal_aligner {substituter_shared, gap_costs};
        if (is_main_thread) {
            diagonal_aligner.init_score(previous_scores[0], 0);
            diagonal_aligner.init_score(current_scores[0], 1);
            diagonal_aligner.init_score(current_scores[1], 1);
            diagonal_aligner.init_gap(current_inserts[0], 1);
            diagonal_aligner.init_gap(current_deletes[1], 1);
        }

        // Make sure the shared memory is fully loaded.
        __syncwarp();

        // We skip diagonals 0 and 1, as they are trivial.
        // We will start with diagonal 2, which has length 3, with the first and last elements being preset,
        // so we are effectively computing just one value, as will be marked by a single set bit in
        // the `next_diagonal_mask` on the very first iteration.
        unsigned next_diagonal_index = 2;

        // Progress through the upper-left triangle of the Levenshtein matrix.
        for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {

            unsigned const next_diagonal_length = next_diagonal_index + 1;
            diagonal_aligner(                         //
                shorter,                              // first sequence of characters
                longer,                               // second sequence of characters
                thread_in_warp_index, warp_size,      //
                next_diagonal_length - 2,             // number of elements to compute with the `diagonal_aligner`
                previous_scores,                      // costs pre substitution
                current_scores, current_scores + 1,   // costs pre insertion/deletion opening
                current_inserts, current_deletes + 1, // costs pre insertion/deletion extension
                next_scores + 1,                      // ! notice unaligned write destination
                next_inserts + 1, next_deletes + 1    // ! notice unaligned write destination
            );

            // Don't forget to populate the first row and the first column of the Levenshtein matrix.
            if (is_main_thread) {
                diagonal_aligner.init_score(next_scores[0], next_diagonal_index);
                diagonal_aligner.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);
                diagonal_aligner.init_gap(next_inserts[0], next_diagonal_index);
                diagonal_aligner.init_gap(next_deletes[next_diagonal_length - 1], next_diagonal_index);
            }
            __syncwarp();

            // Perform a circular rotation of those buffers, to reuse the memory.
            rotate_three(previous_scores, current_scores, next_scores);
            trivial_swap(current_inserts, next_inserts);
            trivial_swap(current_deletes, next_deletes);
        }

        // Now let's handle the anti-diagonal band of the matrix, between the top and bottom-right triangles.
        for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {

            unsigned const next_diagonal_length = shorter_dim;
            diagonal_aligner(                               //
                shorter,                                    // first sequence of characters
                longer + next_diagonal_index - shorter_dim, // second sequence of characters
                thread_in_warp_index, warp_size,            //
                next_diagonal_length - 1,                   // number of elements to compute with the `diagonal_aligner`
                previous_scores,                            // costs pre substitution
                current_scores, current_scores + 1,         // costs pre insertion/deletion opening
                current_inserts, current_deletes + 1,       // costs pre insertion/deletion extension
                next_scores,                                // updated similarity scores
                next_inserts, next_deletes                  // updated insertion/deletion extensions
            );

            // Don't forget to populate the first row of the Levenshtein matrix.
            if (is_main_thread) {
                diagonal_aligner.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);
                diagonal_aligner.init_gap(next_deletes[next_diagonal_length - 1], next_diagonal_index);
            }

            trivial_swap(current_inserts, next_inserts);
            trivial_swap(current_deletes, next_deletes);

            __syncwarp();
            // ! In the central anti-diagonal band, we can't just set the `current_scores + 1` to `previous_scores`
            // ! for the circular shift, as we will end up spilling outside of the diagonal a few iterations later.
            // ! Assuming in-place `memmove` is tricky on the GPU, so we will copy the data.
            for (unsigned i = thread_in_warp_index; i + 1 < next_diagonal_length; i += warp_size)
                previous_scores[i] = current_scores[i + 1];
            __syncwarp();
            for (unsigned i = thread_in_warp_index; i < next_diagonal_length; i += warp_size)
                current_scores[i] = next_scores[i];
            __syncwarp();
        }

        // Now let's handle the bottom-right triangle of the matrix.
        for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {

            unsigned const next_diagonal_length = diagonals_count - next_diagonal_index;
            diagonal_aligner(                               //
                shorter + next_diagonal_index - longer_dim, // first sequence of characters
                longer + next_diagonal_index - shorter_dim, // second sequence of characters
                thread_in_warp_index, warp_size,            //
                next_diagonal_length,                       // number of elements to compute with the `diagonal_aligner`
                previous_scores,                            // costs pre substitution
                current_scores, current_scores + 1,         // costs pre insertion/deletion opening
                current_inserts, current_deletes + 1,       // costs pre insertion/deletion extension
                next_scores,                                // updated similarity scores
                next_inserts, next_deletes                  // updated insertion/deletion extensions
            );

            // Perform a circular rotation of those buffers, to reuse the memory.
            rotate_three(previous_scores, current_scores, next_scores);
            trivial_swap(current_inserts, next_inserts);
            trivial_swap(current_deletes, next_deletes);

            // ! Drop the first entry among the current scores.
            // ! Assuming every next diagonal is shorter by one element,
            // ! we don't need a full-blown `sz_move` to shift the array by one element.
            previous_scores++;
            __syncwarp();
        }

        // Export one result per each block.
        if (is_main_thread) result_ref = diagonal_aligner.score();
    }
}

#pragma endregion

#pragma region - Levenshtein Distance in CUDA

/**
 *  @brief Wraps a single task for the CUDA-based @b byte-level "similarity" kernels.
 *  @note Used to allow sorting/grouping inputs to differentiate device-wide and warp-wide tasks.
 */
/** @brief Max string length (chars) for which Levenshtein runs as a register-only thread-per-pair kernel. */
inline static constexpr unsigned levenshtein_on_each_cuda_thread_default_text_limit_k = 128;

template <typename char_type_>
struct cuda_similarity_task_ {
    using char_t = char_type_;

    char_t const *shorter_ptr = nullptr;
    size_t shorter_length = 0;
    char_t const *longer_ptr = nullptr;
    size_t longer_length = 0;
    size_t memory_requirement = 0;
    size_t original_index = 0;
    size_t result = std::numeric_limits<size_t>::max();       // ? Signal that we are not done yet.
    bytes_per_cell_t bytes_per_cell = eight_bytes_per_cell_k; // ? Worst case, need the most memory per scalar.
    warp_tasks_density_t density = warps_working_together_k;  // ? Worst case, we are not using shared memory.

    constexpr cuda_similarity_task_() = default;
    constexpr cuda_similarity_task_(                  //
        char_t const *first_ptr, size_t first_length, //
        char_t const *second_ptr, size_t second_length) noexcept {
        if (first_length < second_length)
            shorter_ptr = first_ptr, shorter_length = first_length, longer_ptr = second_ptr,
            longer_length = second_length;
        else
            shorter_ptr = second_ptr, shorter_length = second_length, longer_ptr = first_ptr,
            longer_length = first_length;
    }

    constexpr size_t max_diagonal_length() const noexcept { return sz_max_of_two(shorter_length, longer_length) + 1; }

    /** @brief Whether this task is small enough for the register-only thread-per-pair Levenshtein kernels. */
    constexpr bool fits_in_registers() const noexcept {
        return (bytes_per_cell == one_byte_per_cell_k || bytes_per_cell == two_bytes_per_cell_k) &&
               shorter_length <= levenshtein_on_each_cuda_thread_default_text_limit_k &&
               longer_length <= levenshtein_on_each_cuda_thread_default_text_limit_k;
    }
};

/**
 *  @brief Byte-wise SIMD helpers (4× `sz_u8_t` packed in a `sz_u32_t`) for the register-only Levenshtein kernel.
 *         On the device they map to the `__vcmpeq4`/`__vminu4`/`__vaddus4`/`__byte_perm` video instructions; the
 *         host fallbacks keep the kernel unit-testable on the CPU.
 */
__forceinline__ __device__ __host__ sz_u32_t sz_u32_vcmpeq4_(sz_u32_t a, sz_u32_t b) noexcept {
#ifdef __CUDA_ARCH__
    return __vcmpeq4(a, b);
#else
    sz_u32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        sz_u8_t byte_a = (a >> (i * 8)) & 0xFF, byte_b = (b >> (i * 8)) & 0xFF;
        if (byte_a == byte_b) result |= 0xFFu << (i * 8);
    }
    return result;
#endif
}

__forceinline__ __device__ __host__ sz_u32_t sz_u32_vminu4_(sz_u32_t a, sz_u32_t b) noexcept {
#ifdef __CUDA_ARCH__
    return __vminu4(a, b);
#else
    sz_u32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        sz_u8_t byte_a = (a >> (i * 8)) & 0xFF, byte_b = (b >> (i * 8)) & 0xFF;
        result |= (sz_u32_t)(byte_a < byte_b ? byte_a : byte_b) << (i * 8);
    }
    return result;
#endif
}

__forceinline__ __device__ __host__ sz_u32_t sz_u32_vaddus4_(sz_u32_t a, sz_u32_t b) noexcept {
#ifdef __CUDA_ARCH__
    return __vaddus4(a, b);
#else
    sz_u32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        sz_u32_t sum = ((a >> (i * 8)) & 0xFF) + ((b >> (i * 8)) & 0xFF);
        result |= (sz_u32_t)(sum > 0xFF ? 0xFF : sum) << (i * 8);
    }
    return result;
#endif
}

__forceinline__ __device__ __host__ sz_u32_t sz_u32_byte_perm_(sz_u32_t x, sz_u32_t y, sz_u32_t selector) noexcept {
#ifdef __CUDA_ARCH__
    return __byte_perm(x, y, selector);
#else
    sz_u8_t source[8];
    for (int i = 0; i < 4; ++i) source[i] = (x >> (i * 8)) & 0xFF, source[i + 4] = (y >> (i * 8)) & 0xFF;
    sz_u32_t result = 0;
    for (int i = 0; i < 4; ++i) result |= (sz_u32_t)source[(selector >> (i * 4)) & 0x7] << (i * 8);
    return result;
#endif
}

/**
 *  @brief Register-only Levenshtein distance for strings up to @p max_text_length_ bytes, one thread per pair.
 *
 *  Wagner-Fischer with a single DP row kept entirely in registers, the (longer) string cached in registers, and
 *  cells stored as `sz_u8_t` packed 4-per-`sz_u32_t` so each video instruction advances four columns at once.
 *  No shared memory and no `__syncwarp` - ideal for short inputs where the anti-diagonal warp kernel starves the
 *  warp. Saturates at 255, so callers must gate on @b `fits_in_registers` (≤1-byte cells, lengths ≤ the limit).
 */
template <unsigned max_text_length_>
struct register_optimal_levenshtein {
    static constexpr unsigned max_text_length_k = max_text_length_;
    static constexpr unsigned vec_count_k = max_text_length_k / sizeof(sz_u32_vec_t);

    sz_u32_vec_t row_vec_[vec_count_k];
    sz_u32_vec_t longer_string_vec_[vec_count_k];

    __forceinline__ __device__ __host__ sz_u8_t operator()(     //
        sz_u8_t const *longer_string, unsigned longer_length,   //
        sz_u8_t const *shorter_string, unsigned shorter_length, //
        uniform_substitution_costs_t const substituter, linear_gap_costs_t const gap_costs) noexcept {

        // Initialize the first row with the column-index * gap-cost ladder, four cells at a time.
        for (unsigned i = 0, running_gap = gap_costs.open_or_extend; i < vec_count_k; ++i) {
            row_vec_[i].u32 = running_gap, running_gap += gap_costs.open_or_extend;
            row_vec_[i].u32 |= running_gap << 8, running_gap += gap_costs.open_or_extend;
            row_vec_[i].u32 |= running_gap << 16, running_gap += gap_costs.open_or_extend;
            row_vec_[i].u32 |= running_gap << 24, running_gap += gap_costs.open_or_extend;
        }

        // Cache the longer string in registers (packed), accessed in the inner loop.
        for (unsigned i = 0; i < longer_length; ++i) longer_string_vec_[0].u8s[i] = longer_string[i];

        error_cost_t const gap_cost = gap_costs.open_or_extend;
        error_cost_t const match_cost = substituter.match, mismatch_cost = substituter.mismatch;
        sz_u32_vec_t gap_vec, match_vec, mismatch_vec;
        gap_vec.u32 = gap_cost | (gap_cost << 8) | (gap_cost << 16) | (gap_cost << 24);
        match_vec.u32 = match_cost | (match_cost << 8) | (match_cost << 16) | (match_cost << 24);
        mismatch_vec.u32 = mismatch_cost | (mismatch_cost << 8) | (mismatch_cost << 16) | (mismatch_cost << 24);

        // Outer loop over the shorter string (fewer iterations).
        for (unsigned row_idx = 1; row_idx <= shorter_length; ++row_idx) {
            sz_u8_t const shorter_char = shorter_string[row_idx - 1];
            sz_u32_t const shorter_char_vec = shorter_char | (shorter_char << 8) | (shorter_char << 16) |
                                              (shorter_char << 24);
            sz_u8_t const col0_curr = row_idx * gap_cost;
            sz_u8_t const col0_prev = (row_idx - 1) * gap_cost;
            sz_u32_t prev_u32vec = col0_prev | (col0_prev << 8) | (col0_prev << 16) | (col0_prev << 24);

            // Inner loop over the longer string, four columns at a time.
            for (unsigned vec_idx = 0; vec_idx < vec_count_k; ++vec_idx) {
                sz_u32_t const top_u32vec = row_vec_[vec_idx].u32;
                sz_u32_t const diag_u32vec = sz_u32_byte_perm_(prev_u32vec, top_u32vec, 0x6543);
                sz_u32_t const match_u32vec = sz_u32_vcmpeq4_(shorter_char_vec, longer_string_vec_[vec_idx].u32);
                sz_u32_t const subst_u32vec = (match_vec.u32 & match_u32vec) | (mismatch_vec.u32 & ~match_u32vec);
                sz_u32_t const from_diag = sz_u32_vaddus4_(diag_u32vec, subst_u32vec);
                sz_u32_t const from_top = sz_u32_vaddus4_(top_u32vec, gap_vec.u32);
                sz_u32_t result_u32vec = sz_u32_vminu4_(from_diag, from_top);

                // Propagate the left dependency across the four packed cells (sequential scan).
                sz_u32_t const left_source = (vec_idx == 0) ? col0_curr : (row_vec_[vec_idx - 1].u32 >> 24);
                sz_u32_t left_u32vec = left_source | (left_source << 8) | (left_source << 16) | (left_source << 24);
                sz_u32_t shifted = sz_u32_byte_perm_(left_u32vec, result_u32vec, 0x6540);
                result_u32vec = sz_u32_vminu4_(result_u32vec, sz_u32_vaddus4_(shifted, gap_vec.u32));
                shifted = sz_u32_byte_perm_(result_u32vec, result_u32vec, 0x2100);
                result_u32vec = sz_u32_vminu4_(result_u32vec, sz_u32_vaddus4_(shifted, gap_vec.u32));
                shifted = sz_u32_byte_perm_(result_u32vec, result_u32vec, 0x2110);
                result_u32vec = sz_u32_vminu4_(result_u32vec, sz_u32_vaddus4_(shifted, gap_vec.u32));
                shifted = sz_u32_byte_perm_(result_u32vec, result_u32vec, 0x2221);
                result_u32vec = sz_u32_vminu4_(result_u32vec, sz_u32_vaddus4_(shifted, gap_vec.u32));

                prev_u32vec = top_u32vec;
                row_vec_[vec_idx].u32 = result_u32vec;
            }
        }

        unsigned const result_vec_idx = (longer_length - 1) / 4, result_byte_idx = (longer_length - 1) % 4;
        return (row_vec_[result_vec_idx].u32 >> (result_byte_idx * 8)) & 0xFF;
    }
};

/**
 *  @brief Levenshtein distances with @b one-thread-per-pair using only register memory, for short inputs.
 *
 *  Each thread runs a full register-resident DP (@ref register_optimal_levenshtein). Inputs are conceptually
 *  padded to a fixed @p max_text_length_ so the inner loop is branch-free; the divergent dimension is the outer
 *  (row) loop, so callers minimize divergence by keeping per-warp row counts similar (or putting a shared query
 *  on the outer axis when there are more targets than queries).
 */
template <                                                                          //
    typename task_type_,                                                            //
    typename char_type_ = char,                                                     //
    typename score_type_ = sz_u8_t,                                                 //
    sz_capability_t capability_ = sz_cap_cuda_k,                                    //
    unsigned max_text_length_ = levenshtein_on_each_cuda_thread_default_text_limit_k //
    >
__global__ __launch_bounds__(256, 4) void levenshtein_on_each_cuda_thread_( //
    task_type_ *tasks, size_t tasks_count,                                 //
    uniform_substitution_costs_t const substituter, linear_gap_costs_t const gap_costs) {

    using task_t = task_type_;
    register_optimal_levenshtein<max_text_length_> levenshtein_computer;
    size_t const threads_per_device = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t task_idx = blockIdx.x * blockDim.x + threadIdx.x; task_idx < tasks_count;
         task_idx += threads_per_device) {
        task_t &task = tasks[task_idx];
        task.result = levenshtein_computer(                                                    //
            reinterpret_cast<sz_u8_t const *>(task.longer_ptr), static_cast<unsigned>(task.longer_length),   //
            reinterpret_cast<sz_u8_t const *>(task.shorter_ptr), static_cast<unsigned>(task.shorter_length), //
            substituter, gap_costs);
    }
}

/**
 *  @brief Register-only Levenshtein for strings up to @p max_text_length_ bytes with @b 2-byte cells, one thread
 *         per pair. Like @ref register_optimal_levenshtein but packs @b two `sz_u16_t` cells per `sz_u32_t` (via the
 *         16-bit video instructions), so it covers non-unit costs / distances that overflow the 1-byte variant.
 */
template <unsigned max_text_length_>
struct register_optimal_levenshtein_u16 {
    static constexpr unsigned max_text_length_k = max_text_length_;
    static constexpr unsigned vec_count_k = max_text_length_k / 2; // ? two `sz_u16_t` cells per `sz_u32_t`

    sz_u32_t row_vec_[vec_count_k];
    sz_u8_t longer_chars_[max_text_length_k];

    __forceinline__ __device__ sz_u16_t operator()(             //
        sz_u8_t const *longer_string, unsigned longer_length,   //
        sz_u8_t const *shorter_string, unsigned shorter_length, //
        uniform_substitution_costs_t const substituter, linear_gap_costs_t const gap_costs) noexcept {

        sz_u16_t const gap_cost = gap_costs.open_or_extend;
        sz_u32_t const gap_vec = (sz_u32_t)gap_cost | ((sz_u32_t)gap_cost << 16);
        sz_u32_t const match_vec = (sz_u32_t)substituter.match | ((sz_u32_t)substituter.match << 16);
        sz_u32_t const mismatch_vec = (sz_u32_t)substituter.mismatch | ((sz_u32_t)substituter.mismatch << 16);

        for (unsigned i = 0, running_gap = gap_cost; i < vec_count_k; ++i) {
            sz_u16_t const low = running_gap;
            running_gap += gap_cost;
            sz_u16_t const high = running_gap;
            running_gap += gap_cost;
            row_vec_[i] = (sz_u32_t)low | ((sz_u32_t)high << 16);
        }
        for (unsigned i = 0; i < longer_length; ++i) longer_chars_[i] = longer_string[i];

        for (unsigned row_idx = 1; row_idx <= shorter_length; ++row_idx) {
            sz_u8_t const shorter_char = shorter_string[row_idx - 1];
            sz_u32_t const shorter_char_vec = (sz_u32_t)shorter_char | ((sz_u32_t)shorter_char << 16);
            sz_u16_t left_carry = row_idx * gap_cost;       // cell[j-1] for the first pair = column 0 of this row
            sz_u16_t diag_carry = (row_idx - 1) * gap_cost; // top[j-1] for the first pair = column 0 of previous row
            for (unsigned vec_idx = 0; vec_idx < vec_count_k; ++vec_idx) {
                sz_u32_t const top = row_vec_[vec_idx];
                sz_u16_t const top0 = top & 0xFFFF, top1 = top >> 16;
                sz_u32_t const diag = (sz_u32_t)diag_carry | ((sz_u32_t)top0 << 16);
                sz_u32_t const chars = (sz_u32_t)longer_chars_[2 * vec_idx] |
                                       ((sz_u32_t)longer_chars_[2 * vec_idx + 1] << 16);
                sz_u32_t const equal = __vcmpeq2(shorter_char_vec, chars);
                sz_u32_t const subst = (match_vec & equal) | (mismatch_vec & ~equal);
                sz_u32_t const r = __vminu2(__vaddus2(diag, subst), __vaddus2(top, gap_vec));
                sz_u16_t r0 = r & 0xFFFF, r1 = r >> 16;
                unsigned const from_left0 = (unsigned)left_carry + gap_cost;
                if (from_left0 < r0) r0 = (sz_u16_t)from_left0;
                unsigned const from_left1 = (unsigned)r0 + gap_cost;
                if (from_left1 < r1) r1 = (sz_u16_t)from_left1;
                row_vec_[vec_idx] = (sz_u32_t)r0 | ((sz_u32_t)r1 << 16);
                left_carry = r1;
                diag_carry = top1;
            }
        }
        unsigned const result_vec_idx = (longer_length - 1) / 2, result_byte_idx = (longer_length - 1) % 2;
        return (row_vec_[result_vec_idx] >> (result_byte_idx * 16)) & 0xFFFF;
    }
};

/** @brief One-thread-per-pair Levenshtein with @b 2-byte register cells; see @ref levenshtein_on_each_cuda_thread_. */
template <                                                                           //
    typename task_type_,                                                             //
    typename char_type_ = char,                                                      //
    sz_capability_t capability_ = sz_cap_cuda_k,                                     //
    unsigned max_text_length_ = levenshtein_on_each_cuda_thread_default_text_limit_k //
    >
__global__ __launch_bounds__(256, 2) void levenshtein_u16_on_each_cuda_thread_( //
    task_type_ *tasks, size_t tasks_count,                                     //
    uniform_substitution_costs_t const substituter, linear_gap_costs_t const gap_costs) {

    using task_t = task_type_;
    register_optimal_levenshtein_u16<max_text_length_> levenshtein_computer;
    size_t const threads_per_device = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t task_idx = blockIdx.x * blockDim.x + threadIdx.x; task_idx < tasks_count;
         task_idx += threads_per_device) {
        task_t &task = tasks[task_idx];
        task.result = levenshtein_computer(                                                                 //
            reinterpret_cast<sz_u8_t const *>(task.longer_ptr), static_cast<unsigned>(task.longer_length),   //
            reinterpret_cast<sz_u8_t const *>(task.shorter_ptr), static_cast<unsigned>(task.shorter_length), //
            substituter, gap_costs);
    }
}

/**
 *  @brief Register-only @b affine-gap Levenshtein for strings up to @p max_text_length_ bytes with @b 2-byte cells,
 *         one thread per pair. Like @ref register_optimal_levenshtein_u16 but runs the Gotoh recurrence: it keeps a
 *         second register row for the insertion matrix @b I (`ins_vec_`) and carries the deletion matrix @b D as a
 *         scalar across the row, so gap opening and extension are priced separately.
 */
template <unsigned max_text_length_>
struct register_optimal_levenshtein_u16_affine {
    static constexpr unsigned max_text_length_k = max_text_length_;
    static constexpr unsigned vec_count_k = max_text_length_k / 2; // ? two `sz_u16_t` cells per `sz_u32_t`

    sz_u32_t row_vec_[vec_count_k]; // ? the score matrix M
    sz_u32_t ins_vec_[vec_count_k]; // ? the insertion matrix I (gap from the row above)
    sz_u8_t longer_chars_[max_text_length_k];

    __forceinline__ __device__ sz_u16_t operator()(             //
        sz_u8_t const *longer_string, unsigned longer_length,   //
        sz_u8_t const *shorter_string, unsigned shorter_length, //
        uniform_substitution_costs_t const substituter, affine_gap_costs_t const gap_costs) noexcept {

        sz_u16_t const open = gap_costs.open, extend = gap_costs.extend;
        sz_u32_t const open_vec = (sz_u32_t)open | ((sz_u32_t)open << 16);
        sz_u32_t const extend_vec = (sz_u32_t)extend | ((sz_u32_t)extend << 16);
        sz_u32_t const match_vec = (sz_u32_t)substituter.match | ((sz_u32_t)substituter.match << 16);
        sz_u32_t const mismatch_vec = (sz_u32_t)substituter.mismatch | ((sz_u32_t)substituter.mismatch << 16);

        // Row 0: M[0][j] = open + extend*(j-1); the gap matrix gets the higher-magnitude "discard" boundary so it
        // never wins, but stays bounded (no overflow on later additions) - matching the serial/warp affine scorer.
        for (unsigned vec_idx = 0; vec_idx < vec_count_k; ++vec_idx) {
            unsigned const col0 = 2 * vec_idx + 1, col1 = 2 * vec_idx + 2;
            sz_u16_t const m0 = open + extend * (col0 - 1), m1 = open + extend * (col1 - 1);
            row_vec_[vec_idx] = (sz_u32_t)m0 | ((sz_u32_t)m1 << 16);
            sz_u16_t const g0 = (open + extend) + (open + extend * (col0 - 1));
            sz_u16_t const g1 = (open + extend) + (open + extend * (col1 - 1));
            ins_vec_[vec_idx] = (sz_u32_t)g0 | ((sz_u32_t)g1 << 16);
        }
        for (unsigned i = 0; i < longer_length; ++i) longer_chars_[i] = longer_string[i];

        for (unsigned row_idx = 1; row_idx <= shorter_length; ++row_idx) {
            sz_u8_t const shorter_char = shorter_string[row_idx - 1];
            sz_u32_t const shorter_char_vec = (sz_u32_t)shorter_char | ((sz_u32_t)shorter_char << 16);
            sz_u16_t left_M = open + extend * (row_idx - 1);                  // M[row][0]
            sz_u16_t diag_M = row_idx == 1 ? 0 : (open + extend * (row_idx - 2)); // M[row-1][0]
            sz_u16_t left_D = (open + extend) + (open + extend * (row_idx - 1));  // D[row][0] (discard boundary)
            for (unsigned vec_idx = 0; vec_idx < vec_count_k; ++vec_idx) {
                sz_u32_t const top = row_vec_[vec_idx];
                sz_u16_t const top0 = top & 0xFFFF, top1 = top >> 16;
                sz_u32_t const prev_ins = ins_vec_[vec_idx];
                // I[row][j] = min(M[row-1][j] + open, I[row-1][j] + extend) - independent per cell, so packed.
                sz_u32_t const ins = __vminu2(__vaddus2(top, open_vec), __vaddus2(prev_ins, extend_vec));
                // diag = (M[row-1][2v], M[row-1][2v+1]); subst cost per cell; the substitution candidate, packed.
                sz_u32_t const diag = (sz_u32_t)diag_M | ((sz_u32_t)top0 << 16);
                sz_u32_t const chars = (sz_u32_t)longer_chars_[2 * vec_idx] |
                                       ((sz_u32_t)longer_chars_[2 * vec_idx + 1] << 16);
                sz_u32_t const equal = __vcmpeq2(shorter_char_vec, chars);
                sz_u32_t const subst = (match_vec & equal) | (mismatch_vec & ~equal);
                // min(diag + subst, I) is independent per cell, so packed; only the deletion fold below is sequential.
                sz_u32_t const partial = __vminu2(__vaddus2(diag, subst), ins);
                sz_u16_t const partial0 = partial & 0xFFFF, partial1 = partial >> 16;
                // Deletion D carries left across the row (sequential): cell 0 then cell 1.
                sz_u16_t const d0_open = left_M + open, d0_ext = left_D + extend;
                sz_u16_t const d0 = sz_min_of_two(d0_open, d0_ext);
                sz_u16_t const m0 = sz_min_of_two(partial0, d0);
                sz_u16_t const d1_open = m0 + open, d1_ext = d0 + extend;
                sz_u16_t const d1 = sz_min_of_two(d1_open, d1_ext);
                sz_u16_t const m1 = sz_min_of_two(partial1, d1);
                row_vec_[vec_idx] = (sz_u32_t)m0 | ((sz_u32_t)m1 << 16);
                ins_vec_[vec_idx] = ins;
                left_M = m1;
                left_D = d1;
                diag_M = top1;
            }
        }
        unsigned const result_vec_idx = (longer_length - 1) / 2, result_byte_idx = (longer_length - 1) % 2;
        return (row_vec_[result_vec_idx] >> (result_byte_idx * 16)) & 0xFFFF;
    }
};

/** @brief One-thread-per-pair @b affine Levenshtein with 2-byte register cells; see @ref levenshtein_u16_on_each_cuda_thread_. */
template <                                                                           //
    typename task_type_,                                                            //
    typename char_type_ = char,                                                      //
    sz_capability_t capability_ = sz_cap_cuda_k,                                     //
    unsigned max_text_length_ = levenshtein_on_each_cuda_thread_default_text_limit_k //
    >
__global__ __launch_bounds__(256, 2) void levenshtein_u16_affine_on_each_cuda_thread_( //
    task_type_ *tasks, size_t tasks_count,                                            //
    uniform_substitution_costs_t const substituter, affine_gap_costs_t const gap_costs) {

    using task_t = task_type_;
    register_optimal_levenshtein_u16_affine<max_text_length_> levenshtein_computer;
    size_t const threads_per_device = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t task_idx = blockIdx.x * blockDim.x + threadIdx.x; task_idx < tasks_count;
         task_idx += threads_per_device) {
        task_t &task = tasks[task_idx];
        task.result = levenshtein_computer(                                                                 //
            reinterpret_cast<sz_u8_t const *>(task.longer_ptr), static_cast<unsigned>(task.longer_length),   //
            reinterpret_cast<sz_u8_t const *>(task.shorter_ptr), static_cast<unsigned>(task.shorter_length), //
            substituter, gap_costs);
    }
}

/**
 *  @brief Dispatches baseline Levenshtein edit distance algorithm to the GPU.
 *         Before starting the kernels, bins them by size to maximize the number of blocks
 *         per grid that can run simultaneously, while fitting into the shared memory.
 */
template <typename char_type_, typename gap_costs_type_, typename allocator_type_, sz_capability_t capability_>
struct levenshtein_distances<char_type_, gap_costs_type_, allocator_type_, capability_,
                             std::enable_if_t<capability_ & sz_cap_cuda_k>> {

    using char_t = char_type_;
    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;
    using scores_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<size_t>;
    static constexpr sz_capability_t capability_k = capability_;

    using task_t = cuda_similarity_task_<char_t>;
    using tasks_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<task_t>;

    uniform_substitution_costs_t substituter_ {};
    gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    // Host-side scratch reused across calls (single-user contract): the launch is driven by one host thread,
    // so these are filled and consumed serially while the GPU does the parallel work.
    mutable safe_vector<task_t, tasks_allocator_t> tasks_ {alloc_};
    mutable safe_vector<u64_t, scores_allocator_t> diagonals_u64_buffer_ {alloc_};

    levenshtein_distances(uniform_substitution_costs_t subs = {}, gap_costs_t gaps = {},
                          allocator_t const &alloc = {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    // The engine owns its scratch buffers, so it is move-only and tied to a single user.
    levenshtein_distances(levenshtein_distances const &) = delete;
    levenshtein_distances &operator=(levenshtein_distances const &) = delete;
    levenshtein_distances(levenshtein_distances &&) noexcept = default;
    levenshtein_distances &operator=(levenshtein_distances &&) noexcept = default;

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
#if SZ_HAS_CONCEPTS_
        requires indexed_results_like<results_type_>
#endif
    cuda_status_t operator()(                                                                 //
        first_strings_type_ const &first_strings, second_strings_type_ const &second_strings, //
        results_type_ &&results,                                                              //
        cuda_executor_t const &executor = {}, gpu_specs_t specs = {}) const noexcept {

        constexpr bool is_affine_k = is_same_type<gap_costs_t, affine_gap_costs_t>::value;
        constexpr size_t count_diagonals_k = is_affine_k ? 7 : 3;

        // RAII pair of GPU timing events, released on every return path by the destructor.
        cuda_timer_t timer;

        using final_score_t = typename indexed_results_type<results_type_>::type;
        tasks_.clear();
        auto &tasks = tasks_;
        if (tasks.try_resize(first_strings.size()) == status_t::bad_alloc_k) return {status_t::bad_alloc_k};

        // Record the start event
        cudaError_t start_event_error = timer.record_start(executor.stream());
        if (start_event_error != cudaSuccess) return make_cuda_status(start_event_error);

        // Ensure inputs are device-accessible (Unified/Device memory). Both strings of every pair come from
        // contiguous tapes/arrays, so probing each pair with `cudaPointerGetAttributes` - a per-pointer driver
        // round-trip - would cost two driver calls per pair (~65k calls at batch 32768). We validate the base
        // pointers of the first pair once, which covers the whole tape for each side.
        if (first_strings.size()) {
            if (!is_device_accessible_memory((void const *)first_strings[0].data()) ||
                !is_device_accessible_memory((void const *)second_strings[0].data()))
                return {status_t::device_memory_mismatch_k, cudaSuccess};
        }

        // Export all the tasks and sort them by decreasing memory requirement.
        size_t count_empty_tasks = 0;
        using similarity_memory_requirements_t = similarity_memory_requirements<size_t, false>;
        for (size_t i = 0; i < first_strings.size(); ++i) {
            task_t task(                                            //
                first_strings[i].data(), first_strings[i].length(), //
                second_strings[i].data(), second_strings[i].length());
            similarity_memory_requirements_t requirement(                                  //
                task.shorter_length, task.longer_length,                                   //
                gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
                sizeof(char_t), 4);

            task.original_index = i;
            // Both warp kernels (linear & affine) read the strings straight from global (L1-cached), so they only
            // need shared memory for the DP diagonals - not the input strings.
            task.memory_requirement = requirement.bytes_for_diagonals;
            task.bytes_per_cell = requirement.bytes_per_cell;
            task.density = warp_tasks_density(requirement.bytes_for_diagonals, specs);
            if (task.density == infinite_warps_per_multiprocessor_k) {
                if constexpr (!is_affine_k) { task.result = task.longer_length * gap_costs_.open_or_extend; }
                else if (!task.longer_length) { task.result = 0; }
                else { task.result = (task.longer_length - 1) * gap_costs_.extend + gap_costs_.open; }
                count_empty_tasks++;
            }
            tasks[i] = task;
        }

        // Tiny tasks (both strings within the register limit) run as a register-only thread-per-pair kernel - far
        // faster than the warp anti-diagonal wavefront for short inputs. Partition them to the front, launch the
        // register kernel, and let the warp/device tiers handle only the rest.
        size_t count_register_level_tasks = 0;
        for (size_t i = 0; i < tasks.size(); ++i) count_register_level_tasks += tasks[i].fits_in_registers();
        if (count_register_level_tasks) {
            std::partition(tasks.begin(), tasks.end(), [](task_t const &task) { return task.fits_in_registers(); });

            // Launches a register thread-per-pair kernel over `tasks[first, first + count)`.
            cuda_status_t register_status {status_t::success_k, cudaSuccess};
            auto const launch_register_kernel = [&](void *kernel, size_t first, size_t count) noexcept {
                if (!count || register_status.status != status_t::success_k) return;
                cudaFunction_t function = nullptr;
                cudaError_t function_error = cudaGetFuncBySymbol(&function, kernel);
                if (function_error != cudaSuccess) return (void)(register_status = make_cuda_status(function_error));
                task_t *tasks_ptr = tasks.data() + first;
                void *kernel_args[4] = {(void *)(&tasks_ptr), (void *)(&count), (void *)(&substituter_),
                                        (void *)(&gap_costs_)};
                unsigned const threads_per_block = 256;
                unsigned blocks_per_grid = 0;
                cuda_status_t occupancy_status = occupancy_grid(blocks_per_grid, kernel, threads_per_block, 0, specs);
                if (occupancy_status.status != status_t::success_k) return (void)(register_status = occupancy_status);
                CUresult launch_error = cuda_launch_t {}
                                            .grid(blocks_per_grid)
                                            .block(threads_per_block)
                                            .shared(0)
                                            .stream(executor.stream())
                                            .launch(function, kernel_args);
                if (launch_error != CUDA_SUCCESS) register_status = make_cuda_status(launch_error);
            };
            if constexpr (!is_affine_k) {
                // Within the register tier, 1-byte cells go to the (4×-SIMD) u8 kernel and the rest to the u16 one.
                size_t const count_u8_register_tasks =
                    std::partition(tasks.begin(), tasks.begin() + count_register_level_tasks,
                                   [](task_t const &task) { return task.bytes_per_cell == one_byte_per_cell_k; }) -
                    tasks.begin();
                launch_register_kernel(
                    (void *)&levenshtein_on_each_cuda_thread_<task_t, char_t, u8_t, capability_k,
                                                             levenshtein_on_each_cuda_thread_default_text_limit_k>,
                    0, count_u8_register_tasks);
                launch_register_kernel(
                    (void *)&levenshtein_u16_on_each_cuda_thread_<task_t, char_t, capability_k,
                                                                 levenshtein_on_each_cuda_thread_default_text_limit_k>,
                    count_u8_register_tasks, count_register_level_tasks - count_u8_register_tasks);
            }
            else {
                // Affine gaps need 2-byte cells (gap opening overflows the 1-byte variant), so the whole register
                // tier runs the u16 affine kernel.
                launch_register_kernel(
                    (void *)&levenshtein_u16_affine_on_each_cuda_thread_<
                        task_t, char_t, capability_k, levenshtein_on_each_cuda_thread_default_text_limit_k>,
                    0, count_register_level_tasks);
            }
            if (register_status.status != status_t::success_k) return register_status;
        }

        auto [device_level_tasks, warp_level_tasks, empty_tasks] = warp_tasks_grouping<task_t>(
            {tasks.data() + count_register_level_tasks, tasks.size() - count_register_level_tasks}, specs);

        if (device_level_tasks.size()) {
            auto device_level_u16_kernel =
                is_affine_k //
                    ? (void *)&affine_score_across_cuda_device<char_t, u16_t, u16_t, final_score_t,
                                                               uniform_substitution_costs_t, sz_minimize_distance_k,
                                                               sz_similarity_global_k, capability_k>
                    : (void *)&linear_score_across_cuda_device_<char_t, u16_t, u16_t, final_score_t,
                                                                uniform_substitution_costs_t, sz_minimize_distance_k,
                                                                sz_similarity_global_k, capability_k>;
            auto device_level_u32_kernel =
                is_affine_k //
                    ? (void *)&affine_score_across_cuda_device<char_t, u32_t, u32_t, final_score_t,
                                                               uniform_substitution_costs_t, sz_minimize_distance_k,
                                                               sz_similarity_global_k, capability_k>
                    : (void *)&linear_score_across_cuda_device_<char_t, u32_t, u32_t, final_score_t,
                                                                uniform_substitution_costs_t, sz_minimize_distance_k,
                                                                sz_similarity_global_k, capability_k>;
            auto device_level_u64_kernel =
                is_affine_k //
                    ? (void *)&affine_score_across_cuda_device<char_t, u64_t, u64_t, final_score_t,
                                                               uniform_substitution_costs_t, sz_minimize_distance_k,
                                                               sz_similarity_global_k, capability_k>
                    : (void *)&linear_score_across_cuda_device_<char_t, u64_t, u64_t, final_score_t,
                                                                uniform_substitution_costs_t, sz_minimize_distance_k,
                                                                sz_similarity_global_k, capability_k>;
            void *device_level_kernel_args[8];

            // On very large inputs we can't fit the diagonals in shared memory, and use the global one.
            diagonals_u64_buffer_.clear();
            auto &diagonals_u64_buffer = diagonals_u64_buffer_;
            task_t const &largest_task = device_level_tasks[0];
            sz_assert_(largest_task.max_diagonal_length() >= device_level_tasks.back().max_diagonal_length());
            if (diagonals_u64_buffer.try_resize(largest_task.max_diagonal_length() * count_diagonals_k) ==
                status_t::bad_alloc_k)
                return {status_t::bad_alloc_k};

            // Individually submit each task to the GPU.
            void *const diagonals_buffer_ptr = (void *)diagonals_u64_buffer.data();
            for (size_t i = 0; i < device_level_tasks.size(); ++i) {
                task_t const &task = device_level_tasks[i];
                device_level_kernel_args[0] = (void *)(&task.shorter_ptr);
                device_level_kernel_args[1] = (void *)(&task.shorter_length);
                device_level_kernel_args[2] = (void *)(&task.longer_ptr);
                device_level_kernel_args[3] = (void *)(&task.longer_length);
                device_level_kernel_args[4] = (void *)(&task.result);
                device_level_kernel_args[5] = (void *)(&diagonals_buffer_ptr);
                device_level_kernel_args[6] = (void *)(&substituter_);
                device_level_kernel_args[7] = (void *)(&gap_costs_);

                // Pick the smallest fitting type for the diagonals.
                void *device_level_kernel = reinterpret_cast<void *>(device_level_u16_kernel);
                if (task.bytes_per_cell >= sizeof(u32_t))
                    device_level_kernel = reinterpret_cast<void *>(device_level_u32_kernel);
                if (task.bytes_per_cell >= sizeof(u64_t))
                    device_level_kernel = reinterpret_cast<void *>(device_level_u64_kernel);

                // Bridge the runtime kernel symbol to a driver `CUfunction`. Calling `cudaGetFuncBySymbol` also
                // guarantees the primary CUDA context is current on this thread, which `cuLaunchKernelEx` requires.
                cudaFunction_t device_level_function = nullptr;
                cudaError_t function_error = cudaGetFuncBySymbol(&device_level_function, device_level_kernel);
                if (function_error != cudaSuccess) return make_cuda_status(function_error);

                // A cooperative launch REQUIRES the whole grid to be co-resident, so the grid must not exceed
                // `co_resident_blocks_per_multiprocessor * streaming_multiprocessors`. The previous hardcoded
                // 32-blocks-per-multiprocessor grid silently violated this on GPUs with lower occupancy.
                unsigned const device_level_block_size = 128;
                unsigned blocks_per_grid = 0;
                cuda_status_t occupancy_status =
                    occupancy_grid(blocks_per_grid, device_level_kernel, device_level_block_size, 0, specs);
                if (occupancy_status.status != status_t::success_k) return occupancy_status;

                // This kernel calls `cg::this_grid().sync()`, so it MUST be launched cooperatively.
                CUresult launch_error = cuda_launch_t {}
                                            .grid(blocks_per_grid)
                                            .block(device_level_block_size)
                                            .shared(0)
                                            .stream(executor.stream())
                                            .cooperative()
                                            .launch(device_level_function, device_level_kernel_args);
                if (launch_error != CUDA_SUCCESS) return make_cuda_status(launch_error);
            }
        }

        // Now process remaining warp-level tasks, checking warp densities in reverse order.
        // From the highest possible number of warps per multiprocessor to the lowest.
        // TODO(v5+): This per-call template selection (u8 vs u16, linear vs affine) is resolved eagerly on the host
        // TODO(v5+): for a single compiled SASS. A future redesign should ship a fatbin with multiple per-SM SASS
        // TODO(v5+): variants and select among them at runtime through a dispatch table, mirroring the pure
        // TODO(v5+): `stringzilla/` foundation's `sz_dispatch_table_*` machinery.
        if (warp_level_tasks.size()) {
            auto warp_level_u8_kernel =
                is_affine_k
                    ? (void *)&affine_score_on_each_cuda_warp_<task_t, char_t, u8_t, u8_t, uniform_substitution_costs_t,
                                                               sz_minimize_distance_k, sz_similarity_global_k,
                                                               capability_k>
                    : (void *)&linear_score_on_each_cuda_warp_<task_t, char_t, u8_t, u8_t, uniform_substitution_costs_t,
                                                               sz_minimize_distance_k, sz_similarity_global_k,
                                                               capability_k>;
            auto warp_level_u16_kernel =
                is_affine_k
                    ? (void *)&affine_score_on_each_cuda_warp_<task_t, char_t, u16_t, u16_t,
                                                               uniform_substitution_costs_t, sz_minimize_distance_k,
                                                               sz_similarity_global_k, capability_k>
                    : (void *)&linear_score_on_each_cuda_warp_<task_t, char_t, u16_t, u16_t,
                                                               uniform_substitution_costs_t, sz_minimize_distance_k,
                                                               sz_similarity_global_k, capability_k>;
            void *warp_level_kernel_args[5];

            cuda_status_t result;
            auto const task_size_equality = [](task_t const &lhs, task_t const &rhs) {
                return lhs.bytes_per_cell == rhs.bytes_per_cell && lhs.density == rhs.density;
            };

            // Selects the warp-kernel template for a group and the indicative (most memory-hungry) task within it.
            auto const pick_kernel_for_group = [&](task_t const *tasks_begin, task_t const *tasks_end) {
                task_t const &indicative_task = *std::max_element(
                    tasks_begin, tasks_end, [](task_t const &lhs, task_t const &rhs) {
                        return lhs.memory_requirement < rhs.memory_requirement;
                    });
                void *warp_level_kernel = reinterpret_cast<void *>(warp_level_u8_kernel);
                if (indicative_task.bytes_per_cell >= sizeof(u16_t))
                    warp_level_kernel = reinterpret_cast<void *>(warp_level_u16_kernel);
                return std::make_pair(warp_level_kernel, &indicative_task);
            };

            // ! CONFIGURE phase: `cuFuncSetAttribute` is a synchronous, function-global operation, so calling it
            // ! once per group between launches would serialize the whole stream. Instead we precompute the MAX
            // ! dynamic shared memory each distinct kernel template will ever request and raise its ceiling a single
            // ! time. The attribute is a monotone-safe upper bound; the actual per-launch dynamic-smem argument stays
            // ! per-group, so a smaller group still only reserves what it needs.
            unsigned shared_memory_ceiling_per_kernel[2] = {0, 0};
            auto const configure_callback = [&](task_t const *tasks_begin, task_t const *tasks_end) {
                auto const [warp_level_kernel, indicative_task] = pick_kernel_for_group(tasks_begin, tasks_end);
                unsigned const kernel_index = warp_level_kernel == warp_level_u16_kernel ? 1u : 0u;
                auto const [optimal_density,
                            speculative_factor] = speculation_friendly_density(indicative_task->density);
                unsigned const shared_memory_per_block = static_cast<unsigned>(indicative_task->memory_requirement *
                                                                               optimal_density);
                sz_assert_(shared_memory_per_block > 0);
                sz_assert_(shared_memory_per_block < specs.shared_memory_per_multiprocessor());
                if (shared_memory_per_block > shared_memory_ceiling_per_kernel[kernel_index])
                    shared_memory_ceiling_per_kernel[kernel_index] = shared_memory_per_block;
            };
            group_by(warp_level_tasks.begin(), warp_level_tasks.end(), task_size_equality, configure_callback);

            // Bridge both warp kernel symbols to driver `CUfunction` handles. Calling `cudaGetFuncBySymbol` also
            // guarantees the primary CUDA context is current on this thread, which the driver launches require.
            void *const warp_level_kernels[2] = {warp_level_u8_kernel, warp_level_u16_kernel};
            cudaFunction_t warp_level_functions[2] = {nullptr, nullptr};
            for (unsigned kernel_index = 0; kernel_index < 2 && result.status == status_t::success_k; ++kernel_index) {
                cudaError_t function_error = cudaGetFuncBySymbol(&warp_level_functions[kernel_index],
                                                                 warp_level_kernels[kernel_index]);
                if (function_error != cudaSuccess) result = make_cuda_status(function_error);
            }
            for (unsigned kernel_index = 0; kernel_index < 2 && result.status == status_t::success_k; ++kernel_index) {
                if (!shared_memory_ceiling_per_kernel[kernel_index]) continue;
                CUresult attribute_error = cuFuncSetAttribute(warp_level_functions[kernel_index],
                                                              CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                                              shared_memory_ceiling_per_kernel[kernel_index]);
                if (attribute_error != CUDA_SUCCESS) result = make_cuda_status(attribute_error);
            }

            // ! ENQUEUE phase: now that every template's shared-memory ceiling is fixed, launch all groups onto the
            // ! stream back-to-back with NO per-group synchronization. The single trailing event sync below drains
            // ! the whole stream, so the host never round-trips between consecutive launches.
            auto const enqueue_callback = [&](task_t const *tasks_begin, task_t const *tasks_end) {
                // Check if we need to stop processing.
                if (result.status != status_t::success_k) return;

                // Make sure all tasks can be handled by the same kernel template.
                task_t const &first_task = *tasks_begin;
                sz_assert_(std::all_of(tasks_begin, tasks_end, [&](task_t const &task) {
                    return task.bytes_per_cell == first_task.bytes_per_cell && task.density == first_task.density;
                }));

                auto const [warp_level_kernel, indicative_task] = pick_kernel_for_group(tasks_begin, tasks_end);

                // Even if we can fit more warps per block we sometimes should not.
                auto const [optimal_density,
                            speculative_factor] = speculation_friendly_density(indicative_task->density);

                unsigned const shared_memory_per_block = static_cast<unsigned>(indicative_task->memory_requirement *
                                                                               optimal_density);
                size_t const count_tasks = tasks_end - tasks_begin;
                warp_level_kernel_args[0] = (void *)(&tasks_begin);
                warp_level_kernel_args[1] = (void *)(&count_tasks);
                warp_level_kernel_args[2] = (void *)(&substituter_);
                warp_level_kernel_args[3] = (void *)(&gap_costs_);
                warp_level_kernel_args[4] = (void *)(&shared_memory_per_block);

                // Warp-level algorithm clearly aligns with the warp size.
                unsigned const threads_per_block = static_cast<unsigned>(specs.warp_size * optimal_density);

                // Size the grid from the actual device occupancy: extra blocks beyond the supplied tasks simply
                // exit through the grid-stride `task_idx < tasks_count` loop guard, so filling the device once is
                // never harmful, while the previous `streaming_multiprocessors * speculative_factor` grid could be
                // smaller than the co-resident block limit and leave warp slots idle.
                unsigned occupancy_blocks_per_grid = 0;
                cuda_status_t occupancy_status = occupancy_grid(occupancy_blocks_per_grid,
                                                               reinterpret_cast<void *>(warp_level_kernel),
                                                               threads_per_block, shared_memory_per_block, specs);
                if (occupancy_status.status != status_t::success_k) {
                    result = occupancy_status;
                    return;
                }
                unsigned blocks_per_grid = static_cast<unsigned>(specs.streaming_multiprocessors) * speculative_factor;
                if (occupancy_blocks_per_grid > blocks_per_grid) blocks_per_grid = occupancy_blocks_per_grid;

                // Map the selected runtime kernel symbol to its already-bridged driver `CUfunction`. These
                // warp-level kernels do NOT use `grid.sync`, so they launch non-cooperatively via `cuLaunchKernelEx`.
                cudaFunction_t warp_level_function = warp_level_kernel == warp_level_u16_kernel
                                                         ? warp_level_functions[1]
                                                         : warp_level_functions[0];
                CUresult launch_error = cuda_launch_t {}
                                            .grid(blocks_per_grid)
                                            .block(threads_per_block)
                                            .shared(shared_memory_per_block)
                                            .stream(executor.stream())
                                            .launch(warp_level_function, warp_level_kernel_args);
                if (launch_error != CUDA_SUCCESS) result = make_cuda_status(launch_error);
            };
            if (result.status == status_t::success_k)
                group_by(warp_level_tasks.begin(), warp_level_tasks.end(), task_size_equality, enqueue_callback);
            if (result.status != status_t::success_k) return result;
        }

        // Calculate the duration. This single trailing sync replaces the per-group `cudaStreamSynchronize`
        // calls: it drains every group's kernels enqueued during the ENQUEUE phase before we read results.
        cudaError_t stop_event_error = timer.record_stop(executor.stream());
        if (stop_event_error != cudaSuccess) return make_cuda_status(stop_event_error);
        cudaError_t execution_error = timer.synchronize();
        if (execution_error != cudaSuccess) return make_cuda_status(execution_error);
        float execution_milliseconds = timer.elapsed_milliseconds();

        // Now that everything went well, export the results back into the `results` array.
        for (size_t i = 0; i < tasks.size(); ++i) {
            task_t const &task = tasks[i];
            results[task.original_index] = task.result;
        }
        return {status_t::success_k, cudaSuccess, CUDA_SUCCESS, execution_milliseconds};
    }
};

#pragma endregion

#pragma region - Needleman Wunsch and Smith Waterman Scores in CUDA

/**
 *  @brief Device-side view of a compact @b `error_costs_32x32_t` substitution table.
 *
 *  Holds raw pointers to a `byte_to_class[256]` map and an `class_substitution_costs[32][32]` table,
 *  which may live either in global or shared memory. The fixed @b stride of 32 entries per row matches
 *  @b `error_costs_classes_count_k`, so a substitution cost is looked up as:
 *
 *      class_substitution_costs[byte_to_class[a] * 32 + byte_to_class[b]]
 *
 *  The whole table (1 KB) plus the class map (256 B) fits in shared memory, replacing the divergent
 *  serialization of a full 256 x 256 table in CUDA @b constant memory.
 */
struct error_costs_classes_in_cuda_shared_memory_t {
    static constexpr unsigned classes_count_k = static_cast<unsigned>(error_costs_classes_count_k);

    sz_u8_t const *byte_to_class = nullptr;
    error_cost_t const *class_substitution_costs = nullptr;

    __host__ error_cost_t magnitude() const noexcept { return 0; }

    /**
     *  @brief Looks up the substitution cost between two bytes. Accepts any byte-like integral type,
     *         so the Hopper kernels can pass packed 16-bit lanes without ambiguous overload resolution.
     */
    template <typename first_byte_type_, typename second_byte_type_>
    __forceinline__ __host__ __device__ error_cost_t operator()(first_byte_type_ a,
                                                                second_byte_type_ b) const noexcept {
#if defined(__CUDA_ARCH__)
        unsigned const class_a = byte_to_class[static_cast<sz_u8_t>(a)];
        unsigned const class_b = byte_to_class[static_cast<sz_u8_t>(b)];
        return class_substitution_costs[class_a * classes_count_k + class_b];
#else
        sz_unused_(a && b);
        return 0;
#endif
    }
};

/**
 *  @brief Cooperatively materializes a substituter into block-local @b static shared memory.
 *
 *  Generic substituters (e.g. `uniform_substitution_costs_t`) carry no out-of-band state, so the
 *  pass-through overload simply returns them unchanged. The `error_costs_classes_in_cuda_shared_memory_t`
 *  overload copies the 256 B class map and the 1 KB cost table into static `__shared__` buffers,
 *  letting every thread in the block read the cost table from fast shared memory.
 */
template <typename substituter_type_>
__forceinline__ __device__ substituter_type_
load_substituter_into_shared_(substituter_type_ const substituter) noexcept {
    return substituter;
}

__forceinline__ __device__ error_costs_classes_in_cuda_shared_memory_t
load_substituter_into_shared_(error_costs_classes_in_cuda_shared_memory_t const substituter) noexcept {

    constexpr unsigned classes_count_k = error_costs_classes_in_cuda_shared_memory_t::classes_count_k;
    __shared__ sz_u8_t shared_byte_to_class[256];
    __shared__ error_cost_t shared_class_substitution_costs[classes_count_k * classes_count_k];

    for (unsigned i = threadIdx.x; i < 256; i += blockDim.x) shared_byte_to_class[i] = substituter.byte_to_class[i];
    for (unsigned i = threadIdx.x; i < classes_count_k * classes_count_k; i += blockDim.x)
        shared_class_substitution_costs[i] = substituter.class_substitution_costs[i];
    __syncthreads();

    error_costs_classes_in_cuda_shared_memory_t shared_substituter;
    shared_substituter.byte_to_class = shared_byte_to_class;
    shared_substituter.class_substitution_costs = shared_class_substitution_costs;
    return shared_substituter;
}

/**
 *  @brief Register-only NW/SW scoring for strings up to @p max_text_length_ bytes with @b signed 2-byte cells, one
 *         thread per pair. Like @ref register_optimal_levenshtein_u16 but @b maximizes (`__vmaxs2`/`__vaddss2`),
 *         gathers the per-cell substitution cost via the class substituter, and - for local scoring - clamps cells
 *         to zero and tracks a column-guarded running maximum (padded columns >ll are excluded).
 */
template <unsigned max_text_length_, sz_similarity_locality_t locality_>
struct register_optimal_nw_sw {
    static constexpr unsigned max_text_length_k = max_text_length_;
    static constexpr unsigned vec_count_k = max_text_length_k / 2; // ? two signed `sz_i16_t` cells per `sz_u32_t`
    static constexpr bool is_local_k = locality_ == sz_similarity_local_k;

    sz_u32_t row_vec_[vec_count_k];
    sz_u8_t longer_chars_[max_text_length_k];

    __forceinline__ __device__ sz_i16_t operator()(             //
        sz_u8_t const *longer_string, unsigned longer_length,   //
        sz_u8_t const *shorter_string, unsigned shorter_length, //
        error_costs_classes_in_cuda_shared_memory_t const substituter, linear_gap_costs_t const gap_costs) noexcept {

        sz_i16_t const gap_cost = gap_costs.open_or_extend; // ? stored as a (negative) penalty, added to scores
        sz_u32_t const gap_vec = (sz_u16_t)gap_cost | ((sz_u32_t)(sz_u16_t)gap_cost << 16);

        for (unsigned i = 0; i < vec_count_k; ++i) {
            if constexpr (is_local_k) { row_vec_[i] = 0; }
            else {
                sz_i16_t const low = (sz_i16_t)((2 * i + 1) * gap_cost), high = (sz_i16_t)((2 * i + 2) * gap_cost);
                row_vec_[i] = (sz_u16_t)low | ((sz_u32_t)(sz_u16_t)high << 16);
            }
        }
        for (unsigned i = 0; i < longer_length; ++i) longer_chars_[i] = longer_string[i];

        sz_i16_t global_max = 0;
        for (unsigned row_idx = 1; row_idx <= shorter_length; ++row_idx) {
            sz_u8_t const shorter_char = shorter_string[row_idx - 1];
            sz_i16_t left_carry = is_local_k ? 0 : (sz_i16_t)(row_idx * gap_cost);
            sz_i16_t diag_carry = is_local_k ? 0 : (sz_i16_t)((row_idx - 1) * gap_cost);
            for (unsigned vec_idx = 0; vec_idx < vec_count_k; ++vec_idx) {
                sz_u32_t const top = row_vec_[vec_idx];
                sz_i16_t const top0 = (sz_i16_t)(top & 0xFFFF), top1 = (sz_i16_t)(top >> 16);
                sz_u32_t const diag = (sz_u16_t)diag_carry | ((sz_u32_t)(sz_u16_t)top0 << 16);
                sz_i16_t const sub0 = substituter(shorter_char, longer_chars_[2 * vec_idx]);
                sz_i16_t const sub1 = substituter(shorter_char, longer_chars_[2 * vec_idx + 1]);
                sz_u32_t const subst = (sz_u16_t)sub0 | ((sz_u32_t)(sz_u16_t)sub1 << 16);
                sz_u32_t const r = __vmaxs2(__vaddss2(diag, subst), __vaddss2(top, gap_vec));
                sz_i16_t r0 = (sz_i16_t)(r & 0xFFFF), r1 = (sz_i16_t)(r >> 16);
                sz_i16_t const from_left0 = (sz_i16_t)(left_carry + gap_cost);
                if (from_left0 > r0) r0 = from_left0;
                sz_i16_t const from_left1 = (sz_i16_t)(r0 + gap_cost);
                if (from_left1 > r1) r1 = from_left1;
                if constexpr (is_local_k) {
                    if (r0 < 0) r0 = 0;
                    if (r1 < 0) r1 = 0;
                    unsigned const col0 = 2 * vec_idx + 1, col1 = 2 * vec_idx + 2;
                    if (col0 <= longer_length && r0 > global_max) global_max = r0;
                    if (col1 <= longer_length && r1 > global_max) global_max = r1;
                }
                row_vec_[vec_idx] = (sz_u16_t)r0 | ((sz_u32_t)(sz_u16_t)r1 << 16);
                left_carry = r1;
                diag_carry = top1;
            }
        }
        if constexpr (is_local_k) return global_max;
        unsigned const result_vec_idx = (longer_length - 1) / 2, result_byte_idx = (longer_length - 1) % 2;
        return (sz_i16_t)((row_vec_[result_vec_idx] >> (result_byte_idx * 16)) & 0xFFFF);
    }
};

/** @brief One-thread-per-pair NW/SW scoring with signed 2-byte register cells; see @ref register_optimal_nw_sw. */
template <                                                                           //
    typename task_type_,                                                             //
    typename char_type_ = char,                                                      //
    sz_similarity_locality_t locality_ = sz_similarity_global_k,                      //
    sz_capability_t capability_ = sz_cap_cuda_k,                                     //
    unsigned max_text_length_ = levenshtein_on_each_cuda_thread_default_text_limit_k //
    >
__global__ __launch_bounds__(256, 2) void nw_sw_score_on_each_cuda_thread_( //
    task_type_ *tasks, size_t tasks_count,                                 //
    error_costs_classes_in_cuda_shared_memory_t const substituter, linear_gap_costs_t const gap_costs) {

    using task_t = task_type_;
    register_optimal_nw_sw<max_text_length_, locality_> nw_sw_computer;
    size_t const threads_per_device = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t task_idx = blockIdx.x * blockDim.x + threadIdx.x; task_idx < tasks_count;
         task_idx += threads_per_device) {
        task_t &task = tasks[task_idx];
        task.result = nw_sw_computer(                                                                       //
            reinterpret_cast<sz_u8_t const *>(task.longer_ptr), static_cast<unsigned>(task.longer_length),   //
            reinterpret_cast<sz_u8_t const *>(task.shorter_ptr), static_cast<unsigned>(task.shorter_length), //
            substituter, gap_costs);
    }
}

/**
 *  @brief Register-only @b affine-gap NW/SW scoring for strings up to @p max_text_length_ bytes with @b signed
 *         2-byte cells, one thread per pair. Like @ref register_optimal_nw_sw but runs the Gotoh recurrence: a
 *         second register row holds the insertion matrix @b I (`ins_vec_`) and the deletion matrix @b D is carried
 *         as a scalar across the row, so gap opening and extension are priced separately.
 */
template <unsigned max_text_length_, sz_similarity_locality_t locality_>
struct register_optimal_nw_sw_affine {
    static constexpr unsigned max_text_length_k = max_text_length_;
    static constexpr unsigned vec_count_k = max_text_length_k / 2; // ? two signed `sz_i16_t` cells per `sz_u32_t`
    static constexpr bool is_local_k = locality_ == sz_similarity_local_k;

    sz_u32_t row_vec_[vec_count_k]; // ? the score matrix M
    sz_u32_t ins_vec_[vec_count_k]; // ? the insertion matrix I (gap from the row above)
    sz_u8_t longer_chars_[max_text_length_k];

    __forceinline__ __device__ sz_i16_t operator()(             //
        sz_u8_t const *longer_string, unsigned longer_length,   //
        sz_u8_t const *shorter_string, unsigned shorter_length, //
        error_costs_classes_in_cuda_shared_memory_t const substituter, affine_gap_costs_t const gap_costs) noexcept {

        sz_i16_t const open = gap_costs.open, extend = gap_costs.extend; // ? stored as (negative) penalties
        sz_u32_t const open_vec = (sz_u16_t)open | ((sz_u32_t)(sz_u16_t)open << 16);
        sz_u32_t const extend_vec = (sz_u16_t)extend | ((sz_u32_t)(sz_u16_t)extend << 16);

        // Row 0: global M[0][j] = open + extend*(j-1) (local resets to 0); the gap matrix gets the higher-magnitude
        // "discard" boundary so it never wins, but stays bounded - matching the serial/warp affine scorer.
        for (unsigned vec_idx = 0; vec_idx < vec_count_k; ++vec_idx) {
            unsigned const col0 = 2 * vec_idx + 1, col1 = 2 * vec_idx + 2;
            if constexpr (is_local_k) { row_vec_[vec_idx] = 0; }
            else {
                sz_i16_t const m0 = open + extend * (sz_i16_t)(col0 - 1), m1 = open + extend * (sz_i16_t)(col1 - 1);
                row_vec_[vec_idx] = (sz_u16_t)m0 | ((sz_u32_t)(sz_u16_t)m1 << 16);
            }
            sz_i16_t const g0 = (open + extend) + (open + extend * (sz_i16_t)(col0 - 1));
            sz_i16_t const g1 = (open + extend) + (open + extend * (sz_i16_t)(col1 - 1));
            ins_vec_[vec_idx] = (sz_u16_t)g0 | ((sz_u32_t)(sz_u16_t)g1 << 16);
        }
        for (unsigned i = 0; i < longer_length; ++i) longer_chars_[i] = longer_string[i];

        sz_i16_t global_max = 0;
        for (unsigned row_idx = 1; row_idx <= shorter_length; ++row_idx) {
            sz_u8_t const shorter_char = shorter_string[row_idx - 1];
            sz_i16_t left_M = is_local_k ? (sz_i16_t)0 : (sz_i16_t)(open + extend * (sz_i16_t)(row_idx - 1)); // M[row][0]
            sz_i16_t diag_M = is_local_k ? (sz_i16_t)0
                                         : (sz_i16_t)(row_idx == 1 ? 0 : (open + extend * (sz_i16_t)(row_idx - 2)));
            sz_i16_t left_D = (open + extend) + (open + extend * (sz_i16_t)(row_idx - 1)); // D[row][0] (discard)
            for (unsigned vec_idx = 0; vec_idx < vec_count_k; ++vec_idx) {
                sz_u32_t const top = row_vec_[vec_idx];
                sz_i16_t const top0 = (sz_i16_t)(top & 0xFFFF), top1 = (sz_i16_t)(top >> 16);
                sz_u32_t const prev_ins = ins_vec_[vec_idx];
                // I[row][j] = max(M[row-1][j] + open, I[row-1][j] + extend) - independent per cell, so packed.
                sz_u32_t const ins = __vmaxs2(__vaddss2(top, open_vec), __vaddss2(prev_ins, extend_vec));
                // diag = (M[row-1][2v], M[row-1][2v+1]); per-cell substitution cost; the substitution candidate.
                sz_u32_t const diag = (sz_u16_t)diag_M | ((sz_u32_t)(sz_u16_t)top0 << 16);
                sz_i16_t const sub0 = substituter(shorter_char, longer_chars_[2 * vec_idx]);
                sz_i16_t const sub1 = substituter(shorter_char, longer_chars_[2 * vec_idx + 1]);
                sz_u32_t const subst = (sz_u16_t)sub0 | ((sz_u32_t)(sz_u16_t)sub1 << 16);
                // max(diag + subst, I) is independent per cell, so packed; only the deletion fold below is sequential.
                sz_u32_t const partial = __vmaxs2(__vaddss2(diag, subst), ins);
                sz_i16_t const partial0 = (sz_i16_t)(partial & 0xFFFF), partial1 = (sz_i16_t)(partial >> 16);
                // Deletion D carries left across the row (sequential): cell 0 then cell 1.
                sz_i16_t const d0_open = left_M + open, d0_ext = left_D + extend;
                sz_i16_t const d0 = sz_max_of_two(d0_open, d0_ext);
                sz_i16_t m0 = sz_max_of_two(partial0, d0);
                sz_i16_t const d1_open = m0 + open, d1_ext = d0 + extend;
                sz_i16_t const d1 = sz_max_of_two(d1_open, d1_ext);
                sz_i16_t m1 = sz_max_of_two(partial1, d1);
                if constexpr (is_local_k) {
                    if (m0 < 0) m0 = 0;
                    if (m1 < 0) m1 = 0;
                    unsigned const col0 = 2 * vec_idx + 1, col1 = 2 * vec_idx + 2;
                    if (col0 <= longer_length && m0 > global_max) global_max = m0;
                    if (col1 <= longer_length && m1 > global_max) global_max = m1;
                }
                row_vec_[vec_idx] = (sz_u16_t)m0 | ((sz_u32_t)(sz_u16_t)m1 << 16);
                ins_vec_[vec_idx] = ins;
                left_M = m1;
                left_D = d1;
                diag_M = top1;
            }
        }
        if constexpr (is_local_k) return global_max;
        unsigned const result_vec_idx = (longer_length - 1) / 2, result_byte_idx = (longer_length - 1) % 2;
        return (sz_i16_t)((row_vec_[result_vec_idx] >> (result_byte_idx * 16)) & 0xFFFF);
    }
};

/** @brief One-thread-per-pair @b affine NW/SW scoring with signed 2-byte register cells; see @ref nw_sw_score_on_each_cuda_thread_. */
template <                                                                           //
    typename task_type_,                                                            //
    typename char_type_ = char,                                                      //
    sz_similarity_locality_t locality_ = sz_similarity_global_k,                      //
    sz_capability_t capability_ = sz_cap_cuda_k,                                     //
    unsigned max_text_length_ = levenshtein_on_each_cuda_thread_default_text_limit_k //
    >
__global__ __launch_bounds__(256, 2) void nw_sw_score_affine_on_each_cuda_thread_( //
    task_type_ *tasks, size_t tasks_count,                                        //
    error_costs_classes_in_cuda_shared_memory_t const substituter, affine_gap_costs_t const gap_costs) {

    using task_t = task_type_;
    register_optimal_nw_sw_affine<max_text_length_, locality_> nw_sw_computer;
    size_t const threads_per_device = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t task_idx = blockIdx.x * blockDim.x + threadIdx.x; task_idx < tasks_count;
         task_idx += threads_per_device) {
        task_t &task = tasks[task_idx];
        task.result = nw_sw_computer(                                                                       //
            reinterpret_cast<sz_u8_t const *>(task.longer_ptr), static_cast<unsigned>(task.longer_length),   //
            reinterpret_cast<sz_u8_t const *>(task.shorter_ptr), static_cast<unsigned>(task.shorter_length), //
            substituter, gap_costs);
    }
}

/**
 *  @brief Dispatches baseline NW or SW scoring algorithm to the GPU.
 *         Before starting the kernels, bins them by size to maximize the number of blocks
 *         per grid that can run simultaneously, while fitting into the shared memory.
 *         Unlike the Levenshtein distances, also uploads the compact byte-level @b `error_costs_32x32_t`
 *         substitution costs to device memory, addressing them via
 *          @b `error_costs_classes_in_cuda_shared_memory_t`, which every block then mirrors into
 *         its own static shared memory.
 */
template <typename gap_costs_type_, typename allocator_type_, sz_similarity_locality_t locality_,
          sz_capability_t capability_>
struct cuda_nw_or_sw_byte_level_scores_ {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using device_substituter_t = error_costs_classes_in_cuda_shared_memory_t;
    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;
    using scores_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<size_t>;
    using byte_to_class_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<sz_u8_t>;
    using class_costs_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<error_cost_t>;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = capability_;

    using task_t = cuda_similarity_task_<char_t>;
    using tasks_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<task_t>;

    error_costs_32x32_t substituter_ {};
    gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    // Host-side scratch reused across calls (single-user contract): the launch is driven by one host thread,
    // so these are filled and consumed serially while the GPU does the parallel work.
    mutable safe_vector<task_t, tasks_allocator_t> tasks_ {alloc_};
    mutable safe_vector<u64_t, scores_allocator_t> diagonals_u64_buffer_ {alloc_};
    mutable safe_vector<sz_u8_t, byte_to_class_allocator_t> byte_to_class_buffer_ {alloc_};
    mutable safe_vector<error_cost_t, class_costs_allocator_t> class_substitution_costs_buffer_ {alloc_};

    cuda_nw_or_sw_byte_level_scores_(error_costs_32x32_t subs = {}, gap_costs_t gaps = {},
                                     allocator_t const &alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    // The engine owns its scratch buffers, so it is move-only and tied to a single user.
    cuda_nw_or_sw_byte_level_scores_(cuda_nw_or_sw_byte_level_scores_ const &) = delete;
    cuda_nw_or_sw_byte_level_scores_ &operator=(cuda_nw_or_sw_byte_level_scores_ const &) = delete;
    cuda_nw_or_sw_byte_level_scores_(cuda_nw_or_sw_byte_level_scores_ &&) noexcept = default;
    cuda_nw_or_sw_byte_level_scores_ &operator=(cuda_nw_or_sw_byte_level_scores_ &&) noexcept = default;

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
#if SZ_HAS_CONCEPTS_
        requires indexed_results_like<results_type_>
#endif
    cuda_status_t operator()(                                                                 //
        first_strings_type_ const &first_strings, second_strings_type_ const &second_strings, //
        results_type_ &&results,                                                              //
        cuda_executor_t const &executor = {}, gpu_specs_t specs = {}) const noexcept {

        constexpr bool is_local_k = locality_k == sz_similarity_local_k;
        constexpr bool is_affine_k = is_same_type<gap_costs_t, affine_gap_costs_t>::value;
        constexpr size_t count_diagonals_k = is_affine_k ? 7 : 3;

        // RAII pair of GPU timing events, released on every return path by the destructor.
        cuda_timer_t timer;

        using final_score_t = typename indexed_results_type<results_type_>::type;
        tasks_.clear();
        auto &tasks = tasks_;
        if (tasks.try_resize(first_strings.size()) == status_t::bad_alloc_k) return {status_t::bad_alloc_k};

        // Record the start event
        cudaError_t start_event_error = timer.record_start(executor.stream());
        if (start_event_error != cudaSuccess) return make_cuda_status(start_event_error);

        // Ensure inputs are device-accessible (Unified/Device memory). Both strings of every pair come from
        // contiguous tapes/arrays, so probing each pair with `cudaPointerGetAttributes` - a per-pointer driver
        // round-trip - would cost two driver calls per pair (~65k calls at batch 32768). We validate the base
        // pointers of the first pair once, which covers the whole tape for each side.
        if (first_strings.size()) {
            if (!is_device_accessible_memory((void const *)first_strings[0].data()) ||
                !is_device_accessible_memory((void const *)second_strings[0].data()))
                return {status_t::device_memory_mismatch_k, cudaSuccess};
        }

        // Upload the compact substituter into device-accessible memory. Every block will later mirror
        // these tiny buffers into its own static shared memory for the inner-loop lookups.
        constexpr size_t classes_count_k = error_costs_32x32_t::classes_count_k;
        byte_to_class_buffer_.clear();
        class_substitution_costs_buffer_.clear();
        auto &byte_to_class_buffer = byte_to_class_buffer_;
        auto &class_substitution_costs_buffer = class_substitution_costs_buffer_;
        if (byte_to_class_buffer.try_resize(256) == status_t::bad_alloc_k) return {status_t::bad_alloc_k};
        if (class_substitution_costs_buffer.try_resize(classes_count_k * classes_count_k) == status_t::bad_alloc_k)
            return {status_t::bad_alloc_k};
        for (size_t i = 0; i < 256; ++i) byte_to_class_buffer[i] = substituter_.byte_to_class[i];
        for (size_t i = 0; i < classes_count_k; ++i)
            for (size_t j = 0; j < classes_count_k; ++j)
                class_substitution_costs_buffer[i * classes_count_k + j] = substituter_.class_substitution_costs[i][j];

        device_substituter_t device_substituter;
        device_substituter.byte_to_class = byte_to_class_buffer.data();
        device_substituter.class_substitution_costs = class_substitution_costs_buffer.data();

        // Export all the tasks and sort them by decreasing memory requirement.
        size_t count_empty_tasks = 0;
        using similarity_memory_requirements_t = similarity_memory_requirements<size_t, true>;
        for (size_t i = 0; i < first_strings.size(); ++i) {
            task_t task(                                            //
                first_strings[i].data(), first_strings[i].length(), //
                second_strings[i].data(), second_strings[i].length());
            similarity_memory_requirements_t requirement(                                  //
                task.shorter_length, task.longer_length,                                   //
                gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
                sizeof(char_t), 4, two_bytes_per_cell_k);

            task.original_index = i;
            // Both warp kernels (linear & affine) read the strings straight from global (L1-cached), so they only
            // need shared memory for the DP diagonals - not the input strings.
            task.memory_requirement = requirement.bytes_for_diagonals;
            task.bytes_per_cell = requirement.bytes_per_cell;
            task.density = warp_tasks_density(requirement.bytes_for_diagonals, specs);
            if (task.density == infinite_warps_per_multiprocessor_k) {
                if constexpr (is_local_k) { task.result = 0; }
                else if constexpr (!is_affine_k) { task.result = task.longer_length * gap_costs_.open_or_extend; }
                else if (!task.longer_length) { task.result = 0; }
                else { task.result = (task.longer_length - 1) * gap_costs_.extend + gap_costs_.open; }
                count_empty_tasks++;
            }
            tasks[i] = task;
        }

        // Tiny tasks (within the register length limit) run as a register-only thread-per-pair NW/SW kernel - far
        // faster than the warp anti-diagonal wavefront for short inputs. Partition them to the front, launch the
        // register kernel, and let the warp/device tiers handle only the rest.
        size_t count_register_level_tasks = 0;
        for (size_t i = 0; i < tasks.size(); ++i) count_register_level_tasks += tasks[i].fits_in_registers();
        if (count_register_level_tasks) {
            std::partition(tasks.begin(), tasks.end(), [](task_t const &task) { return task.fits_in_registers(); });
            void *thread_level_kernel =
                is_affine_k //
                    ? (void *)&nw_sw_score_affine_on_each_cuda_thread_<
                          task_t, char_t, locality_k, capability_k, levenshtein_on_each_cuda_thread_default_text_limit_k>
                    : (void *)&nw_sw_score_on_each_cuda_thread_<
                          task_t, char_t, locality_k, capability_k, levenshtein_on_each_cuda_thread_default_text_limit_k>;
            cudaFunction_t thread_level_function = nullptr;
            cudaError_t function_error = cudaGetFuncBySymbol(&thread_level_function, thread_level_kernel);
            if (function_error != cudaSuccess) return make_cuda_status(function_error);
            task_t *tasks_ptr = tasks.data();
            void *thread_level_kernel_args[4] = {(void *)(&tasks_ptr), (void *)(&count_register_level_tasks),
                                                 (void *)(&device_substituter), (void *)(&gap_costs_)};
            unsigned const threads_per_block = 256;
            unsigned blocks_per_grid = 0;
            cuda_status_t occupancy_status =
                occupancy_grid(blocks_per_grid, thread_level_kernel, threads_per_block, 0, specs);
            if (occupancy_status.status != status_t::success_k) return occupancy_status;
            CUresult launch_error = cuda_launch_t {}
                                        .grid(blocks_per_grid)
                                        .block(threads_per_block)
                                        .shared(0)
                                        .stream(executor.stream())
                                        .launch(thread_level_function, thread_level_kernel_args);
            if (launch_error != CUDA_SUCCESS) return make_cuda_status(launch_error);
        }

        auto [device_level_tasks, warp_level_tasks, empty_tasks] = warp_tasks_grouping<task_t>(
            {tasks.data() + count_register_level_tasks, tasks.size() - count_register_level_tasks}, specs);

        if (device_level_tasks.size()) {
            auto device_level_i32_kernel =
                is_affine_k //
                    ? (void *)&affine_score_across_cuda_device<char_t, u32_t, sz_i32_t, final_score_t,
                                                               error_costs_classes_in_cuda_shared_memory_t,
                                                               sz_maximize_score_k, locality_k, capability_k>
                    : (void *)&linear_score_across_cuda_device_<char_t, u32_t, sz_i32_t, final_score_t,
                                                                error_costs_classes_in_cuda_shared_memory_t,
                                                                sz_maximize_score_k, locality_k, capability_k>;
            auto device_level_i64_kernel =
                is_affine_k //
                    ? (void *)&affine_score_across_cuda_device<char_t, u64_t, sz_i64_t, final_score_t,
                                                               error_costs_classes_in_cuda_shared_memory_t,
                                                               sz_maximize_score_k, locality_k, capability_k>
                    : (void *)&linear_score_across_cuda_device_<char_t, u64_t, sz_i64_t, final_score_t,
                                                                error_costs_classes_in_cuda_shared_memory_t,
                                                                sz_maximize_score_k, locality_k, capability_k>;
            void *device_level_kernel_args[8];

            // On very large inputs we can't fit the diagonals in shared memory, and use the global one.
            diagonals_u64_buffer_.clear();
            auto &diagonals_u64_buffer = diagonals_u64_buffer_;
            task_t const &largest_task = device_level_tasks[0];
            sz_assert_(largest_task.max_diagonal_length() >= device_level_tasks.back().max_diagonal_length());
            if (diagonals_u64_buffer.try_resize(largest_task.max_diagonal_length() * count_diagonals_k) ==
                status_t::bad_alloc_k)
                return {status_t::bad_alloc_k};

            // Individually submit each task to the GPU.
            void *const diagonals_buffer_ptr = (void *)diagonals_u64_buffer.data();
            for (size_t i = 0; i < device_level_tasks.size(); ++i) {
                task_t const &task = device_level_tasks[i];
                device_level_kernel_args[0] = (void *)(&task.shorter_ptr);
                device_level_kernel_args[1] = (void *)(&task.shorter_length);
                device_level_kernel_args[2] = (void *)(&task.longer_ptr);
                device_level_kernel_args[3] = (void *)(&task.longer_length);
                device_level_kernel_args[4] = (void *)(&task.result);
                device_level_kernel_args[5] = (void *)(&diagonals_buffer_ptr);
                device_level_kernel_args[6] = (void *)(&device_substituter);
                device_level_kernel_args[7] = (void *)(&gap_costs_);

                // Pick the smallest fitting type for the diagonals.
                void *device_level_kernel = reinterpret_cast<void *>(device_level_i32_kernel);
                if (task.bytes_per_cell >= sizeof(sz_i64_t))
                    device_level_kernel = reinterpret_cast<void *>(device_level_i64_kernel);

                // Bridge the runtime kernel symbol to a driver `CUfunction`. Calling `cudaGetFuncBySymbol` also
                // guarantees the primary CUDA context is current on this thread, which `cuLaunchKernelEx` requires.
                cudaFunction_t device_level_function = nullptr;
                cudaError_t function_error = cudaGetFuncBySymbol(&device_level_function, device_level_kernel);
                if (function_error != cudaSuccess) return make_cuda_status(function_error);

                // A cooperative launch REQUIRES the whole grid to be co-resident, so the grid must not exceed
                // `co_resident_blocks_per_multiprocessor * streaming_multiprocessors`. The previous hardcoded
                // 32-blocks-per-multiprocessor grid silently violated this on GPUs with lower occupancy.
                unsigned const device_level_block_size = 128;
                unsigned blocks_per_grid = 0;
                cuda_status_t occupancy_status =
                    occupancy_grid(blocks_per_grid, device_level_kernel, device_level_block_size, 0, specs);
                if (occupancy_status.status != status_t::success_k) return occupancy_status;

                // This kernel calls `cg::this_grid().sync()`, so it MUST be launched cooperatively.
                CUresult launch_error = cuda_launch_t {}
                                            .grid(blocks_per_grid)
                                            .block(device_level_block_size)
                                            .shared(0)
                                            .stream(executor.stream())
                                            .cooperative()
                                            .launch(device_level_function, device_level_kernel_args);
                if (launch_error != CUDA_SUCCESS) return make_cuda_status(launch_error);
            }
        }

        // Now process remaining warp-level tasks, checking warp densities in reverse order.
        // From the highest possible number of warps per multiprocessor to the lowest.
        // TODO(v5+): This per-call template selection (i16 vs i32, linear vs affine) is resolved eagerly on the host
        // TODO(v5+): for a single compiled SASS. A future redesign should ship a fatbin with multiple per-SM SASS
        // TODO(v5+): variants and select among them at runtime through a dispatch table, mirroring the pure
        // TODO(v5+): `stringzilla/` foundation's `sz_dispatch_table_*` machinery.
        if (warp_level_tasks.size()) {
            auto warp_level_i16_kernel =
                is_affine_k ? (void *)&affine_score_on_each_cuda_warp_<task_t, char_t, u16_t, sz_i16_t,
                                                                       error_costs_classes_in_cuda_shared_memory_t,
                                                                       sz_maximize_score_k, locality_k, capability_k>
                            : (void *)&linear_score_on_each_cuda_warp_<task_t, char_t, u16_t, sz_i16_t,
                                                                       error_costs_classes_in_cuda_shared_memory_t,
                                                                       sz_maximize_score_k, locality_k, capability_k>;
            auto warp_level_i32_kernel =
                is_affine_k ? (void *)&affine_score_on_each_cuda_warp_<task_t, char_t, u32_t, sz_i32_t,
                                                                       error_costs_classes_in_cuda_shared_memory_t,
                                                                       sz_maximize_score_k, locality_k, capability_k>
                            : (void *)&linear_score_on_each_cuda_warp_<task_t, char_t, u32_t, sz_i32_t,
                                                                       error_costs_classes_in_cuda_shared_memory_t,
                                                                       sz_maximize_score_k, locality_k, capability_k>;
            void *warp_level_kernel_args[5];

            cuda_status_t result;
            auto const task_size_equality = [](task_t const &lhs, task_t const &rhs) {
                return lhs.bytes_per_cell == rhs.bytes_per_cell && lhs.density == rhs.density;
            };

            // Selects the warp-kernel template for a group and the indicative (most memory-hungry) task within it.
            auto const pick_kernel_for_group = [&](task_t const *tasks_begin, task_t const *tasks_end) {
                task_t const &indicative_task = *std::max_element(
                    tasks_begin, tasks_end, [](task_t const &lhs, task_t const &rhs) {
                        return lhs.memory_requirement < rhs.memory_requirement;
                    });
                void *warp_level_kernel = reinterpret_cast<void *>(warp_level_i16_kernel);
                if (indicative_task.bytes_per_cell >= sizeof(sz_i32_t))
                    warp_level_kernel = reinterpret_cast<void *>(warp_level_i32_kernel);
                return std::make_pair(warp_level_kernel, &indicative_task);
            };

            // ! CONFIGURE phase: `cuFuncSetAttribute` is a synchronous, function-global operation, so calling it
            // ! once per group between launches would serialize the whole stream. Instead we precompute the MAX
            // ! dynamic shared memory each distinct kernel template will ever request and raise its ceiling a single
            // ! time. The attribute is a monotone-safe upper bound; the actual per-launch dynamic-smem argument stays
            // ! per-group, so a smaller group still only reserves what it needs.
            unsigned shared_memory_ceiling_per_kernel[2] = {0, 0};
            auto const configure_callback = [&](task_t const *tasks_begin, task_t const *tasks_end) {
                auto const [warp_level_kernel, indicative_task] = pick_kernel_for_group(tasks_begin, tasks_end);
                unsigned const kernel_index = warp_level_kernel == warp_level_i32_kernel ? 1u : 0u;
                auto const [optimal_density,
                            speculative_factor] = speculation_friendly_density(indicative_task->density);
                unsigned const shared_memory_per_block = static_cast<unsigned>(indicative_task->memory_requirement *
                                                                               optimal_density);
                sz_assert_(shared_memory_per_block > 0);
                sz_assert_(shared_memory_per_block < specs.shared_memory_per_multiprocessor());
                if (shared_memory_per_block > shared_memory_ceiling_per_kernel[kernel_index])
                    shared_memory_ceiling_per_kernel[kernel_index] = shared_memory_per_block;
            };
            group_by(warp_level_tasks.begin(), warp_level_tasks.end(), task_size_equality, configure_callback);

            // Bridge both warp kernel symbols to driver `CUfunction` handles. Calling `cudaGetFuncBySymbol` also
            // guarantees the primary CUDA context is current on this thread, which the driver launches require.
            void *const warp_level_kernels[2] = {warp_level_i16_kernel, warp_level_i32_kernel};
            cudaFunction_t warp_level_functions[2] = {nullptr, nullptr};
            for (unsigned kernel_index = 0; kernel_index < 2 && result.status == status_t::success_k; ++kernel_index) {
                cudaError_t function_error = cudaGetFuncBySymbol(&warp_level_functions[kernel_index],
                                                                 warp_level_kernels[kernel_index]);
                if (function_error != cudaSuccess) result = make_cuda_status(function_error);
            }
            for (unsigned kernel_index = 0; kernel_index < 2 && result.status == status_t::success_k; ++kernel_index) {
                if (!shared_memory_ceiling_per_kernel[kernel_index]) continue;
                CUresult attribute_error = cuFuncSetAttribute(warp_level_functions[kernel_index],
                                                              CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                                              shared_memory_ceiling_per_kernel[kernel_index]);
                if (attribute_error != CUDA_SUCCESS) result = make_cuda_status(attribute_error);
            }

            // ! ENQUEUE phase: now that every template's shared-memory ceiling is fixed, launch all groups onto the
            // ! stream back-to-back with NO per-group synchronization. The single trailing event sync below drains
            // ! the whole stream, so the host never round-trips between consecutive launches.
            auto const enqueue_callback = [&](task_t const *tasks_begin, task_t const *tasks_end) {
                // Check if we need to stop processing.
                if (result.status != status_t::success_k) return;

                // Make sure all tasks can be handled by the same kernel template.
                task_t const &first_task = *tasks_begin;
                sz_assert_(std::all_of(tasks_begin, tasks_end, [&](task_t const &task) {
                    return task.bytes_per_cell == first_task.bytes_per_cell && task.density == first_task.density;
                }));

                auto const [warp_level_kernel, indicative_task] = pick_kernel_for_group(tasks_begin, tasks_end);

                // Even if we can fit more warps per block we sometimes should not.
                auto const [optimal_density,
                            speculative_factor] = speculation_friendly_density(indicative_task->density);

                unsigned const shared_memory_per_block = static_cast<unsigned>(indicative_task->memory_requirement *
                                                                               optimal_density);
                size_t const count_tasks = tasks_end - tasks_begin;
                warp_level_kernel_args[0] = (void *)(&tasks_begin);
                warp_level_kernel_args[1] = (void *)(&count_tasks);
                warp_level_kernel_args[2] = (void *)(&device_substituter);
                warp_level_kernel_args[3] = (void *)(&gap_costs_);
                warp_level_kernel_args[4] = (void *)(&shared_memory_per_block);

                // Warp-level algorithm clearly aligns with the warp size.
                unsigned const threads_per_block = static_cast<unsigned>(specs.warp_size * optimal_density);

                // Size the grid from the actual device occupancy: extra blocks beyond the supplied tasks simply
                // exit through the grid-stride `task_idx < tasks_count` loop guard, so filling the device once is
                // never harmful, while the previous `streaming_multiprocessors * speculative_factor` grid could be
                // smaller than the co-resident block limit and leave warp slots idle.
                unsigned occupancy_blocks_per_grid = 0;
                cuda_status_t occupancy_status = occupancy_grid(occupancy_blocks_per_grid,
                                                               reinterpret_cast<void *>(warp_level_kernel),
                                                               threads_per_block, shared_memory_per_block, specs);
                if (occupancy_status.status != status_t::success_k) {
                    result = occupancy_status;
                    return;
                }
                unsigned blocks_per_grid = static_cast<unsigned>(specs.streaming_multiprocessors) * speculative_factor;
                if (occupancy_blocks_per_grid > blocks_per_grid) blocks_per_grid = occupancy_blocks_per_grid;

                // Map the selected runtime kernel symbol to its already-bridged driver `CUfunction`. These
                // warp-level kernels do NOT use `grid.sync`, so they launch non-cooperatively via `cuLaunchKernelEx`.
                cudaFunction_t warp_level_function = warp_level_kernel == warp_level_i32_kernel
                                                         ? warp_level_functions[1]
                                                         : warp_level_functions[0];
                CUresult launch_error = cuda_launch_t {}
                                            .grid(blocks_per_grid)
                                            .block(threads_per_block)
                                            .shared(shared_memory_per_block)
                                            .stream(executor.stream())
                                            .launch(warp_level_function, warp_level_kernel_args);
                if (launch_error != CUDA_SUCCESS) result = make_cuda_status(launch_error);
            };
            if (result.status == status_t::success_k)
                group_by(warp_level_tasks.begin(), warp_level_tasks.end(), task_size_equality, enqueue_callback);
            if (result.status != status_t::success_k) return result;
        }

        // Calculate the duration. This single trailing sync replaces the per-group `cudaStreamSynchronize`
        // calls: it drains every group's kernels enqueued during the ENQUEUE phase before we read results.
        cudaError_t stop_event_error = timer.record_stop(executor.stream());
        if (stop_event_error != cudaSuccess) return make_cuda_status(stop_event_error);
        cudaError_t execution_error = timer.synchronize();
        if (execution_error != cudaSuccess) return make_cuda_status(execution_error);
        float execution_milliseconds = timer.elapsed_milliseconds();

        // Now that everything went well, export the results back into the `results` array.
        for (size_t task_index = 0; task_index < tasks.size(); ++task_index) {
            task_t const &task = tasks[task_index];
            results[task.original_index] = task.result;
        }
        return {status_t::success_k, cudaSuccess, CUDA_SUCCESS, execution_milliseconds};
    }
};

/** @brief Dispatches baseline Needleman Wunsch algorithm to the GPU. */
template <typename gap_costs_type_, typename allocator_type_, sz_capability_t capability_>
struct needleman_wunsch_scores<char, error_costs_32x32_t, gap_costs_type_, allocator_type_, capability_,
                               std::enable_if_t<capability_ & sz_cap_cuda_k>>
    : public cuda_nw_or_sw_byte_level_scores_<gap_costs_type_, allocator_type_, sz_similarity_global_k, capability_> {

    using cuda_nw_or_sw_byte_level_scores_<gap_costs_type_, allocator_type_, sz_similarity_global_k,
                                           capability_>::cuda_nw_or_sw_byte_level_scores_;
};

/** @brief Dispatches baseline Smith Waterman algorithm to the GPU. */
template <typename gap_costs_type_, typename allocator_type_, sz_capability_t capability_>
struct smith_waterman_scores<char, error_costs_32x32_t, gap_costs_type_, allocator_type_, capability_,
                             std::enable_if_t<capability_ & sz_cap_cuda_k>>
    : public cuda_nw_or_sw_byte_level_scores_<gap_costs_type_, allocator_type_, sz_similarity_local_k, capability_> {

    using cuda_nw_or_sw_byte_level_scores_<gap_costs_type_, allocator_type_, sz_similarity_local_k,
                                           capability_>::cuda_nw_or_sw_byte_level_scores_;
};

#pragma endregion

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SIMILARITIES_CUDA_CUH_
