/**
 *  @brief  CUDA-accelerated string similarity utilities.
 *  @file   similarities.cuh
 *  @author Ash Vardanian
 *
 *  Unlike the CPU backed, which also has single-pair similarity scores, the CUDA backend focuses @b only on
 *  batch-processing of large collections of strings, generally, assigning a single warp to each string pair:
 *
 *  - `sz::levenshtein_distances` & `sz::levenshtein_distances_utf8` for Levenshtein edit-distances.
 *  - `sz::needleman_wunsch_score` for weighted Needleman-Wunsch global alignment scores.
 *  - `sz::smith_waterman_score` for weighted Smith-Waterman local alignment scores.
 *
 *  Unlike the trivially parallelizable CPU kernels in `stringzilla/similarities.hpp`, the GPU kernels in this file are
 *  designed for batch-processing of large collections of strings, assigning a single warp to each string pair.
 *  Thus, they should be used when hundreds of pairwise comparisons are needed, and the strings are long enough to
 *  amortize the cost of copying them to the GPU.
 *
 *  @section Abstraction layers
 *
 *  Under the hood, each @b dense high-level algorithm, like Levenshtein, NW, or SW, builds on top of a "walker"
 *  template object, which in turn builds on top of an "scorer" template object:
 *
 *  - the "walker" chooses the order in which the DP matrix is evaluated - row-wise or diagonal-wise.
 *  - the "scorer" evaluates the actual DP matrix cells, taking 3+ inputs, for "local" and "global" alignment,
 *    or the "affine local" and "affine global" alignments, differentiating the cost of gap opening & extension.
 *
 *  Those are later wrapped via 2 functions:
 *
 *  - `*_in_cuda` for on-GPU execution - implementing the actual similarity scoring.
 *  - `*_via_cuda` for on-host execution - allocating memory, and dispatching the kernel.
 *
 *  Those are in-turn wrapped into the same-named function objects:
 *
 *  - `levenshtein_distances`: {CUDA and Kepler} for any chars and lengths, {Hopper} for 8-bit and 16-bit lengths.
 *  - `needleman_wunsch_score`.
 */
#ifndef STRINGZILLAS_SIMILARITIES_CUH_
#define STRINGZILLAS_SIMILARITIES_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>        // `cuda::pipeline`
#include <cooperative_groups.h> // `cooperative_groups::this_grid()`

#include "stringzillas/types.cuh"
#include "stringzillas/similarities.hpp"

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
    needleman_wunsch_scores<char, error_costs_256x256_t, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
using smith_waterman_cuda_t =
    smith_waterman_scores<char, error_costs_256x256_t, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k>;

using affine_needleman_wunsch_cuda_t =
    needleman_wunsch_scores<char, error_costs_256x256_t, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
using affine_smith_waterman_cuda_t =
    smith_waterman_scores<char, error_costs_256x256_t, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k>;

using needleman_wunsch_hopper_t =
    needleman_wunsch_scores<char, error_costs_256x256_t, linear_gap_costs_t, ualloc_t, sz_caps_ckh_k>;
using smith_waterman_hopper_t =
    smith_waterman_scores<char, error_costs_256x256_t, linear_gap_costs_t, ualloc_t, sz_caps_ckh_k>;

using affine_needleman_wunsch_hopper_t =
    needleman_wunsch_scores<char, error_costs_256x256_t, affine_gap_costs_t, ualloc_t, sz_caps_ckh_k>;
using affine_smith_waterman_hopper_t =
    smith_waterman_scores<char, error_costs_256x256_t, affine_gap_costs_t, ualloc_t, sz_caps_ckh_k>;

#pragma endregion - Common Aliases

#pragma region - Common Helpers

/**
 *  @brief  Dispatches min or max operation based on the compile-time objective.
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
 *  @brief  Loads data with a hint, that it's frequently accessed and immutable throughout the kernel.
 *  @see    https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#read-only-data-cache-load-function
 *  @see    https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#global-memory-5-x
 */
template <typename scalar_type_>
__forceinline__ __device__ scalar_type_ load_immutable_(scalar_type_ const *ptr) noexcept {
    // The `__ldg` intrinsic translates into the `ld.global.nc` PTX instruction.
    // It reads a value from global memory and caches it in the non-coherent cache.
    // return __ldg(ptr);
    return *ptr;
}

/**
 *  @brief  Loads data with a cache hint, that it will not be accessed again.
 *  @see    https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#load-functions-using-cache-hints
 *  @see    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators
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

/*  On Kepler and newer GPUs we benefit from the following:
 *  - processing 4x 8-bit values or 2x 16-bit values at a time, packed as 32-bit words.
 *  - warp-level exchange primitives for fast reduction of the best score.
 */

#if SZ_USE_KEPLER

/**
 *  @brief GPU adaptation of the `tile_scorer` - Minimizes Global Levenshtein distance with linear gap costs.
 *  @note Requires Kepler generation GPUs to handle 4x `u8` scores at a time.
 *
 *  Relies on following instruction families to output 4x @b `u8` scores per call:
 *  - @b `prmt` to shuffle bytes in 32 bit registers.
 *  - @b `vmax4,vmin4,vadd4` video-processing instructions.
 */
template <>
struct tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
    using kepler_warp_scorer_t =
        tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                    sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>;

    __forceinline__ __device__ void operator()(            //
        char const *first_slice, char const *second_slice, //
        unsigned const tasks_offset, unsigned const tasks_step,
        unsigned const tasks_count,          // ! Unlike CPU, uses `unsigned`
        u8_t const *scores_pre_substitution, //
        u8_t const *scores_pre_insertion,    //
        u8_t const *scores_pre_deletion,     //
        u8_t *scores_new) noexcept {

        u8_t const match_cost = this->substituter_.match;
        u8_t const mismatch_cost = this->substituter_.mismatch;
        u8_t const gap_cost = this->gap_costs_.open_or_extend;

        sz_u32_vec_t match_cost_vec, mismatch_cost_vec, gap_cost_vec, equality_vec;
        match_cost_vec.u32 = match_cost * 0x01010101u;       // ! 4x `u8` match costs
        mismatch_cost_vec.u32 = mismatch_cost * 0x01010101u; // ! 4x `u8` mismatch costs
        gap_cost_vec.u32 = gap_cost * 0x01010101u;           // ! 4x `u8` gap costs

        // The hardest part of this kernel is dealing with unaligned loads!
        // We want to minimize single-byte processing in favor of 4-byte SIMD loads and min/max operations.
        // Assuming we are reading consecutive values from a buffer, in every cycle, most likely, we will be
        // dealing with most values being unaligned!
        sz_u32_vec_t pre_substitution_vec, pre_insertion_vec, pre_deletion_vec;
        sz_u32_vec_t first_vec, second_vec;
        sz_u32_vec_t cost_of_substitution_vec, if_substitution_vec, if_deletion_or_insertion_vec;
        sz_u32_vec_t cell_score_vec;

        // ! As we are processing 4 bytes per loop, and have at least 32 threads per block (32 * 4 = 128),
        // ! and deal with strings only under 256 bytes, this loop will fire at most twice per input.
        for (unsigned i = tasks_offset * 4; i < tasks_count; i += tasks_step * 4) { // ! it's OK to spill beyond bounds
            pre_substitution_vec = sz_u32_load_unaligned(scores_pre_substitution + i);
            pre_insertion_vec = sz_u32_load_unaligned(scores_pre_insertion + i);
            pre_deletion_vec = sz_u32_load_unaligned(scores_pre_deletion + i);
            first_vec = sz_u32_load_unaligned(first_slice + tasks_count - i - 4); // ! this may be OOB
            second_vec = sz_u32_load_unaligned(second_slice + i);                 // ! this may be OOB, but padded
            first_vec.u32 = __nv_bswap32(first_vec.u32); // ! reverse the order of bytes in the first vector

            // Equality comparison will output 0xFF for each matching byte.
            equality_vec.u32 = __vcmpeq4(first_vec.u32, second_vec.u32);
            cost_of_substitution_vec.u32 =                //
                (equality_vec.u32 & match_cost_vec.u32) + //
                (~equality_vec.u32 & mismatch_cost_vec.u32);
            if_substitution_vec.u32 = __vaddus4(pre_substitution_vec.u32, cost_of_substitution_vec.u32);
            if_deletion_or_insertion_vec.u32 =
                __vaddus4(__vminu4(pre_deletion_vec.u32, pre_insertion_vec.u32), gap_cost_vec.u32);
            cell_score_vec.u32 = __vminu4(if_deletion_or_insertion_vec.u32, if_substitution_vec.u32);

            // When walking through the top-left triangle of the matrix, our output addresses are misaligned.
            scores_new[i + 0] = cell_score_vec.u8s[0];
            scores_new[i + 1] = cell_score_vec.u8s[1];
            scores_new[i + 2] = cell_score_vec.u8s[2];
            scores_new[i + 3] = cell_score_vec.u8s[3];
        }

        // Extract the bottom-right corner of the matrix, which is the result of the global alignment.
        if (tasks_offset == 0) this->final_score_ = scores_new[0];
    }
};

/**
 *  @brief GPU adaptation of the `tile_scorer` - Minimizes Global Levenshtein distance with linear gap costs.
 *  @note Requires Kepler generation GPUs to handle 2x `u16` scores at a time.
 *
 *  Relies on following instruction families to output 2x @b `u16` scores per call:
 *  - @b `prmt` to shuffle bytes in 32 bit registers.
 *  - @b `vmax2,vmin2,vadd2` video-processing instructions.
 */
template <>
struct tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
    using kepler_warp_scorer_t =
        tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                    sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>;

    __forceinline__ __device__ void operator()(            //
        char const *first_slice, char const *second_slice, //
        unsigned const tasks_offset, unsigned const tasks_step,
        unsigned const tasks_count,           // ! Unlike CPU, uses `unsigned`
        u16_t const *scores_pre_substitution, //
        u16_t const *scores_pre_insertion,    //
        u16_t const *scores_pre_deletion,     //
        u16_t *scores_new) noexcept {

        u16_t const match_cost = this->substituter_.match;
        u16_t const mismatch_cost = this->substituter_.mismatch;
        u16_t const gap_cost = this->gap_costs_.open_or_extend;

        sz_u32_vec_t match_cost_vec, mismatch_cost_vec, gap_cost_vec, equality_vec;
        match_cost_vec.u32 = match_cost * 0x00010001;       // ! 2x `u16` match costs
        mismatch_cost_vec.u32 = mismatch_cost * 0x00010001; // ! 2x `u16` mismatch costs
        gap_cost_vec.u32 = gap_cost * 0x00010001;           // ! 2x `u16` gap costs

        // The hardest part of this kernel is dealing with unaligned loads!
        // We want to minimize single-byte processing in favor of 2-byte SIMD loads and min/max operations.
        // Assuming we are reading consecutive values from a buffer, in every cycle, most likely, we will be
        // dealing with most values being unaligned!
        sz_u32_vec_t pre_substitution_vec, pre_insertion_vec, pre_deletion_vec;
        sz_u32_vec_t first_vec, second_vec;
        sz_u32_vec_t cost_of_substitution_vec, if_substitution_vec, if_deletion_or_insertion_vec;
        sz_u32_vec_t cell_score_vec;

        // ! As we are processing 2 bytes per loop, and have at least 32 threads per block (32 * 2 = 64),
        // ! and deal with strings only under 64k bytes, this loop will fire at most 1K times per input
        for (unsigned i = tasks_offset * 2; i < tasks_count; i += tasks_step * 2) { // ! it's OK to spill beyond bounds
            pre_substitution_vec.u16s[0] = scores_pre_substitution[i + 0];
            pre_substitution_vec.u16s[1] = scores_pre_substitution[i + 1];
            pre_insertion_vec.u16s[0] = scores_pre_insertion[i + 0];
            pre_insertion_vec.u16s[1] = scores_pre_insertion[i + 1];
            pre_deletion_vec.u16s[0] = scores_pre_deletion[i + 0];
            pre_deletion_vec.u16s[1] = scores_pre_deletion[i + 1];
            first_vec.u16s[0] = load_immutable_(first_slice + tasks_count - i - 1);
            first_vec.u16s[1] = load_immutable_(first_slice + tasks_count - i - 2); // ! this may be OOB
            second_vec.u16s[0] = load_immutable_(second_slice + i + 0);
            second_vec.u16s[1] = load_immutable_(second_slice + i + 1); // ! this may be OOB, but padded

            // Equality comparison will output 0xFFFF for each matching byte-pair.
            equality_vec.u32 = __vcmpeq2(first_vec.u32, second_vec.u32);
            cost_of_substitution_vec.u32 =                //
                (equality_vec.u32 & match_cost_vec.u32) + //
                (~equality_vec.u32 & mismatch_cost_vec.u32);
            if_substitution_vec.u32 = __vaddus2(pre_substitution_vec.u32, cost_of_substitution_vec.u32);
            if_deletion_or_insertion_vec.u32 =
                __vaddus2(__vminu2(pre_deletion_vec.u32, pre_insertion_vec.u32), gap_cost_vec.u32);
            cell_score_vec.u32 = __vminu2(if_deletion_or_insertion_vec.u32, if_substitution_vec.u32);

            // When walking through the top-left triangle of the matrix, our output addresses are misaligned.
            scores_new[i + 0] = cell_score_vec.u16s[0];
            scores_new[i + 1] = cell_score_vec.u16s[1];
        }

        // Extract the bottom-right corner of the matrix, which is the result of the global alignment.
        if (tasks_offset == 0) this->final_score_ = scores_new[0];
    }
};

template <>
struct tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
    using kepler_warp_scorer_t =
        tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                    sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>;
};

template <>
struct tile_scorer<char const *, char const *, u64_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, u64_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
    using kepler_warp_scorer_t =
        tile_scorer<char const *, char const *, u64_t, uniform_substitution_costs_t, linear_gap_costs_t,
                    sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>;
};

/**
 *  @brief GPU adaptation of the `tile_scorer` - Minimizes Global Levenshtein distance with affine gap costs.
 *  @note Requires Kepler generation GPUs to handle 4x `u8` scores at a time.
 *
 *  Relies on following instruction families to output 4x @b `u8` scores per call:
 *  - @b `prmt` to shuffle bytes in 32 bit registers.
 *  - @b `vmax4,vmin4,vadd4` video-processing instructions.
 */
template <>
struct tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
    using kepler_warp_scorer_t =
        tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                    sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>;

    __forceinline__ __device__ void operator()(            //
        char const *first_slice, char const *second_slice, //
        unsigned const tasks_offset, unsigned const tasks_step,
        unsigned const tasks_count,            // ! Unlike CPU, uses `unsigned`
        u8_t const *scores_pre_substitution,   //
        u8_t const *scores_pre_insertion,      //
        u8_t const *scores_pre_deletion,       //
        u8_t const *scores_running_insertions, //
        u8_t const *scores_running_deletions,  //
        u8_t *scores_new,                      //
        u8_t *scores_new_insertions,           //
        u8_t *scores_new_deletions) noexcept {

        u8_t const match_cost = this->substituter_.match;
        u8_t const mismatch_cost = this->substituter_.mismatch;
        u8_t const gap_open_cost = this->gap_costs_.open;
        u8_t const gap_extend_cost = this->gap_costs_.extend;
        sz_u32_vec_t match_cost_vec, mismatch_cost_vec, gap_open_cost_vec, gap_extend_cost_vec, equality_vec;
        match_cost_vec.u32 = match_cost * 0x01010101u;           // ! 4x `u8` match costs
        mismatch_cost_vec.u32 = mismatch_cost * 0x01010101u;     // ! 4x `u8` mismatch costs
        gap_open_cost_vec.u32 = gap_open_cost * 0x01010101u;     // ! 4x `u8` gap costs
        gap_extend_cost_vec.u32 = gap_extend_cost * 0x01010101u; // ! 4x `u8` gap costs

        // The hardest part of this kernel is dealing with unaligned loads!
        // We want to minimize single-byte processing in favor of 4-byte SIMD loads and min/max operations.
        // Assuming we are reading consecutive values from a buffer, in every cycle, most likely, we will be
        // dealing with most values being unaligned!
        sz_u32_vec_t pre_substitution_vec, pre_insertion_opening_vec, pre_deletion_opening_vec;
        sz_u32_vec_t pre_insertion_expansion_vec, pre_deletion_expansion_vec;
        sz_u32_vec_t first_vec, second_vec;
        sz_u32_vec_t cost_of_substitution_vec, if_substitution_vec, if_insertion_vec, if_deletion_vec;
        sz_u32_vec_t cell_score_vec;

        // ! As we are processing 4 bytes per loop, and have at least 32 threads per block (32 * 4 = 128),
        // ! and deal with strings only under 256 bytes, this loop will fire at most twice per input.
        for (unsigned i = tasks_offset * 4; i < tasks_count; i += tasks_step * 4) { // ! it's OK to spill beyond bounds
            pre_substitution_vec = sz_u32_load_unaligned(scores_pre_substitution + i);
            pre_insertion_opening_vec = sz_u32_load_unaligned(scores_pre_insertion + i);
            pre_deletion_opening_vec = sz_u32_load_unaligned(scores_pre_deletion + i);
            pre_insertion_expansion_vec = sz_u32_load_unaligned(scores_running_insertions + i);
            pre_deletion_expansion_vec = sz_u32_load_unaligned(scores_running_deletions + i);
            first_vec = sz_u32_load_unaligned(first_slice + tasks_count - i - 4); // ! this may be OOB
            second_vec = sz_u32_load_unaligned(second_slice + i);                 // ! this may be OOB, but padded
            first_vec.u32 = __nv_bswap32(first_vec.u32); // ! reverse the order of bytes in the first vector

            // Equality comparison will output 0xFF for each matching byte.
            equality_vec.u32 = __vcmpeq4(first_vec.u32, second_vec.u32);
            cost_of_substitution_vec.u32 =                //
                (equality_vec.u32 & match_cost_vec.u32) + //
                (~equality_vec.u32 & mismatch_cost_vec.u32);
            if_substitution_vec.u32 = __vaddus4(pre_substitution_vec.u32, cost_of_substitution_vec.u32);
            if_insertion_vec.u32 = __vminu4(__vaddus4(pre_insertion_opening_vec.u32, gap_open_cost_vec.u32),
                                            __vaddus4(pre_insertion_expansion_vec.u32, gap_extend_cost_vec.u32));
            if_deletion_vec.u32 = __vminu4(__vaddus4(pre_deletion_opening_vec.u32, gap_open_cost_vec.u32),
                                           __vaddus4(pre_deletion_expansion_vec.u32, gap_extend_cost_vec.u32));
            cell_score_vec.u32 = __vminu4(if_substitution_vec.u32, __vminu4(if_insertion_vec.u32, if_deletion_vec.u32));

            // When walking through the top-left triangle of the matrix, our output addresses are misaligned.
            scores_new[i + 0] = cell_score_vec.u8s[0];
            scores_new[i + 1] = cell_score_vec.u8s[1];
            scores_new[i + 2] = cell_score_vec.u8s[2];
            scores_new[i + 3] = cell_score_vec.u8s[3];
            scores_new_insertions[i + 0] = if_insertion_vec.u8s[0];
            scores_new_insertions[i + 1] = if_insertion_vec.u8s[1];
            scores_new_insertions[i + 2] = if_insertion_vec.u8s[2];
            scores_new_insertions[i + 3] = if_insertion_vec.u8s[3];
            scores_new_deletions[i + 0] = if_deletion_vec.u8s[0];
            scores_new_deletions[i + 1] = if_deletion_vec.u8s[1];
            scores_new_deletions[i + 2] = if_deletion_vec.u8s[2];
            scores_new_deletions[i + 3] = if_deletion_vec.u8s[3];
        }

        // Extract the bottom-right corner of the matrix, which is the result of the global alignment.
        if (tasks_offset == 0) this->final_score_ = scores_new[0];
    }
};

/**
 *  @brief GPU adaptation of the `tile_scorer` - Minimizes Global Levenshtein distance with affine gap costs.
 *  @note Requires Kepler generation GPUs to handle 2x `u16` scores at a time.
 *
 *  Relies on following instruction families to output 2x @b `u16` scores per call:
 *  - @b `prmt` to shuffle bytes in 32 bit registers.
 *  - @b `vmax2,vmin2,vadd2` video-processing instructions.
 */
template <>
struct tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
    using kepler_warp_scorer_t =
        tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                    sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>;

    __forceinline__ __device__ void operator()(            //
        char const *first_slice, char const *second_slice, //
        unsigned const tasks_offset, unsigned const tasks_step,
        unsigned const tasks_count,             // ! Unlike CPU, uses `unsigned`
        u16_t const *scores_pre_substitution,   //
        u16_t const *scores_pre_insertion,      //
        u16_t const *scores_pre_deletion,       //
        u16_t const *scores_running_insertions, //
        u16_t const *scores_running_deletions,  //
        u16_t *scores_new,                      //
        u16_t *scores_new_insertions,           //
        u16_t *scores_new_deletions) noexcept {

        u16_t const match_cost = this->substituter_.match;
        u16_t const mismatch_cost = this->substituter_.mismatch;
        u16_t const gap_open_cost = this->gap_costs_.open;
        u16_t const gap_extend_cost = this->gap_costs_.extend;

        sz_u32_vec_t match_cost_vec, mismatch_cost_vec, gap_open_cost_vec, gap_extend_cost_vec, equality_vec;
        match_cost_vec.u32 = match_cost * 0x00010001;           // ! 2x `u16` match costs
        mismatch_cost_vec.u32 = mismatch_cost * 0x00010001;     // ! 2x `u16` mismatch costs
        gap_open_cost_vec.u32 = gap_open_cost * 0x00010001;     // ! 2x `u16` gap costs
        gap_extend_cost_vec.u32 = gap_extend_cost * 0x00010001; // ! 2x `u16` gap costs

        // The hardest part of this kernel is dealing with unaligned loads!
        // We want to minimize single-byte processing in favor of 2-byte SIMD loads and min/max operations.
        // Assuming we are reading consecutive values from a buffer, in every cycle, most likely, we will be
        // dealing with most values being unaligned!
        sz_u32_vec_t pre_substitution_vec, pre_insertion_opening_vec, pre_deletion_opening_vec;
        sz_u32_vec_t pre_insertion_expansion_vec, pre_deletion_expansion_vec;
        sz_u32_vec_t first_vec, second_vec;
        sz_u32_vec_t cost_of_substitution_vec, if_substitution_vec, if_insertion_vec, if_deletion_vec;
        sz_u32_vec_t cell_score_vec;

        // ! As we are processing 2 bytes per loop, and have at least 32 threads per block (32 * 2 = 64),
        // ! and deal with strings only under 64k bytes, this loop will fire at most 1K times per input
        for (unsigned i = tasks_offset * 2; i < tasks_count; i += tasks_step * 2) { // ! it's OK to spill beyond bounds
            pre_substitution_vec.u16s[0] = scores_pre_substitution[i + 0];
            pre_substitution_vec.u16s[1] = scores_pre_substitution[i + 1];
            pre_insertion_opening_vec.u16s[0] = scores_pre_insertion[i + 0];
            pre_insertion_opening_vec.u16s[1] = scores_pre_insertion[i + 1];
            pre_deletion_opening_vec.u16s[0] = scores_pre_deletion[i + 0];
            pre_deletion_opening_vec.u16s[1] = scores_pre_deletion[i + 1];
            pre_insertion_expansion_vec.u16s[0] = scores_running_insertions[i + 0];
            pre_insertion_expansion_vec.u16s[1] = scores_running_insertions[i + 1];
            pre_deletion_expansion_vec.u16s[0] = scores_running_deletions[i + 0];
            pre_deletion_expansion_vec.u16s[1] = scores_running_deletions[i + 1];
            first_vec.u16s[0] = load_immutable_(first_slice + tasks_count - i - 1);
            first_vec.u16s[1] = load_immutable_(first_slice + tasks_count - i - 2); // ! this may be OOB
            second_vec.u16s[0] = load_immutable_(second_slice + i + 0);
            second_vec.u16s[1] = load_immutable_(second_slice + i + 1); // ! this may be OOB, but padded

            // Equality comparison will output 0xFFFF for each matching byte-pair.
            equality_vec.u32 = __vcmpeq2(first_vec.u32, second_vec.u32);
            cost_of_substitution_vec.u32 =                //
                (equality_vec.u32 & match_cost_vec.u32) + //
                (~equality_vec.u32 & mismatch_cost_vec.u32);
            if_substitution_vec.u32 = __vaddus2(pre_substitution_vec.u32, cost_of_substitution_vec.u32);
            if_insertion_vec.u32 = __vminu2(__vaddus2(pre_insertion_opening_vec.u32, gap_open_cost_vec.u32),
                                            __vaddus2(pre_insertion_expansion_vec.u32, gap_extend_cost_vec.u32));
            if_deletion_vec.u32 = __vminu2(__vaddus2(pre_deletion_opening_vec.u32, gap_open_cost_vec.u32),
                                           __vaddus2(pre_deletion_expansion_vec.u32, gap_extend_cost_vec.u32));
            cell_score_vec.u32 = __vminu2(if_substitution_vec.u32, __vminu2(if_insertion_vec.u32, if_deletion_vec.u32));

            // When walking through the top-left triangle of the matrix, our output addresses are misaligned.
            scores_new[i + 0] = cell_score_vec.u16s[0];
            scores_new[i + 1] = cell_score_vec.u16s[1];
            scores_new_insertions[i + 0] = if_insertion_vec.u16s[0];
            scores_new_insertions[i + 1] = if_insertion_vec.u16s[1];
            scores_new_deletions[i + 0] = if_deletion_vec.u16s[0];
            scores_new_deletions[i + 1] = if_deletion_vec.u16s[1];
        }

        // Extract the bottom-right corner of the matrix, which is the result of the global alignment.
        if (tasks_offset == 0) this->final_score_ = scores_new[0];
    }
};

template <>
struct tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {
    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
};

template <>
struct tile_scorer<char const *, char const *, u64_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, u64_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {
    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
};

#endif

/*  Hopper-generation optimizations are quite different from Kepler.
 *  Our Kepler optimizations are mostly designed for 8-bit and 16-bit scalars packed as 32-bit words,
 *  while Hopper optimizations are designed for 16-bit and 32-bit scalars, grouping chains of add/min/max
 *  operations using DPX instructions.
 */

#if SZ_USE_HOPPER

template <>
struct tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k> {
    using kepler_warp_scorer_t::tile_scorer; // Make the constructors visible
};

/**
 *  @brief GPU adaptation of the `tile_scorer` - Minimizes Global Levenshtein distance with linear gap costs.
 *  @note Requires Hopper generation GPUs to handle 2x `u16` scores at a time.
 */
template <>
struct tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {
    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible

    __forceinline__ __device__ void operator()(            //
        char const *first_slice, char const *second_slice, //
        unsigned const tasks_offset, unsigned const tasks_step,
        unsigned const tasks_count,           // ! Unlike CPU, uses `unsigned`
        u16_t const *scores_pre_substitution, //
        u16_t const *scores_pre_insertion,    //
        u16_t const *scores_pre_deletion,     //
        u16_t *scores_new) noexcept {

        u16_t const match_cost = this->substituter_.match;
        u16_t const mismatch_cost = this->substituter_.mismatch;
        u16_t const gap_cost = this->gap_costs_.open_or_extend;

        sz_u32_vec_t match_cost_vec, mismatch_cost_vec, gap_cost_vec, equality_vec;
        match_cost_vec.u32 = match_cost * 0x00010001;       // ! 2x `u16` match costs
        mismatch_cost_vec.u32 = mismatch_cost * 0x00010001; // ! 2x `u16` mismatch costs
        gap_cost_vec.u32 = gap_cost * 0x00010001;           // ! 2x `u16` gap costs

        // The hardest part of this kernel is dealing with unaligned loads!
        // We want to minimize single-byte processing in favor of 2-byte SIMD loads and min/max operations.
        // Assuming we are reading consecutive values from a buffer, in every cycle, most likely, we will be
        // dealing with most values being unaligned!
        sz_u32_vec_t pre_substitution_vec, pre_insertion_vec, pre_deletion_vec;
        sz_u32_vec_t first_vec, second_vec;
        sz_u32_vec_t cost_of_substitution_vec, if_substitution_vec, if_insertion_vec, if_deletion_vec;
        sz_u32_vec_t cell_score_vec;

        // ! As we are processing 2 bytes per loop, and have at least 32 threads per block (32 * 2 = 64),
        // ! and deal with strings only under 64k bytes, this loop will fire at most 1K times per input
        for (unsigned i = tasks_offset * 2; i < tasks_count; i += tasks_step * 2) { // ! it's OK to spill beyond bounds
            pre_substitution_vec.u16s[0] = scores_pre_substitution[i + 0];
            pre_substitution_vec.u16s[1] = scores_pre_substitution[i + 1];
            pre_insertion_vec.u16s[0] = scores_pre_insertion[i + 0];
            pre_insertion_vec.u16s[1] = scores_pre_insertion[i + 1];
            pre_deletion_vec.u16s[0] = scores_pre_deletion[i + 0];
            pre_deletion_vec.u16s[1] = scores_pre_deletion[i + 1];
            first_vec.u16s[0] = load_immutable_(first_slice + tasks_count - i - 1);
            first_vec.u16s[1] = load_immutable_(first_slice + tasks_count - i - 2); // ! this may be OOB
            second_vec.u16s[0] = load_immutable_(second_slice + i + 0);
            second_vec.u16s[1] = load_immutable_(second_slice + i + 1); // ! this may be OOB, but padded

            // Equality comparison will output 0xFFFF for each matching byte-pair.
            equality_vec.u32 = __vcmpeq2(first_vec.u32, second_vec.u32);
            cost_of_substitution_vec.u32 =                //
                (equality_vec.u32 & match_cost_vec.u32) + //
                (~equality_vec.u32 & mismatch_cost_vec.u32);
            if_substitution_vec.u32 = __vaddus2(pre_substitution_vec.u32, cost_of_substitution_vec.u32);
            if_insertion_vec.u32 = __vaddus2(pre_insertion_vec.u32, gap_cost_vec.u32);
            if_deletion_vec.u32 = __vaddus2(pre_deletion_vec.u32, gap_cost_vec.u32);
            cell_score_vec.u32 = __vimin3_u16x2(if_substitution_vec.u32, if_insertion_vec.u32, if_deletion_vec.u32);

            // When walking through the top-left triangle of the matrix, our output addresses are misaligned.
            scores_new[i + 0] = cell_score_vec.u16s[0];
            scores_new[i + 1] = cell_score_vec.u16s[1];
        }

        // Extract the bottom-right corner of the matrix, which is the result of the global alignment.
        if (tasks_offset == 0) this->final_score_ = scores_new[0];
    }
};

template <>
struct tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
};

template <>
struct tile_scorer<char const *, char const *, u64_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, u64_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
};

template <>
struct tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, u8_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k> {
    using kepler_warp_scorer_t::tile_scorer; // Make the constructors visible
};

/**
 *  @brief GPU adaptation of the `tile_scorer` - Minimizes Global Levenshtein distance with affine gap costs.
 *  @note Requires Hopper generation GPUs to handle 2x `u8` scores at a time.
 */
template <>
struct tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, u16_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible

    __forceinline__ __device__ void operator()(            //
        char const *first_slice, char const *second_slice, //
        unsigned const tasks_offset, unsigned const tasks_step,
        unsigned const tasks_count,             // ! Unlike CPU, uses `unsigned`
        u16_t const *scores_pre_substitution,   //
        u16_t const *scores_pre_insertion,      //
        u16_t const *scores_pre_deletion,       //
        u16_t const *scores_running_insertions, //
        u16_t const *scores_running_deletions,  //
        u16_t *scores_new,                      //
        u16_t *scores_new_insertions,           //
        u16_t *scores_new_deletions) noexcept {

        u16_t const match_cost = this->substituter_.match;
        u16_t const mismatch_cost = this->substituter_.mismatch;
        u16_t const gap_open_cost = this->gap_costs_.open;
        u16_t const gap_extend_cost = this->gap_costs_.extend;

        sz_u32_vec_t match_cost_vec, mismatch_cost_vec, gap_open_cost_vec, gap_extend_cost_vec, equality_vec;
        match_cost_vec.u32 = match_cost * 0x00010001;           // ! 2x `u16` match costs
        mismatch_cost_vec.u32 = mismatch_cost * 0x00010001;     // ! 2x `u16` mismatch costs
        gap_open_cost_vec.u32 = gap_open_cost * 0x00010001;     // ! 2x `u16` gap costs
        gap_extend_cost_vec.u32 = gap_extend_cost * 0x00010001; // ! 2x `u16` gap costs

        // The hardest part of this kernel is dealing with unaligned loads!
        // We want to minimize single-byte processing in favor of 2-byte SIMD loads and min/max operations.
        // Assuming we are reading consecutive values from a buffer, in every cycle, most likely, we will be
        // dealing with most values being unaligned!
        sz_u32_vec_t pre_substitution_vec, pre_insertion_opening_vec, pre_deletion_opening_vec;
        sz_u32_vec_t pre_insertion_expansion_vec, pre_deletion_expansion_vec;
        sz_u32_vec_t first_vec, second_vec;
        sz_u32_vec_t cost_of_substitution_vec, if_substitution_vec, if_insertion_vec, if_deletion_vec;
        sz_u32_vec_t cell_score_vec;

        // ! As we are processing 2 bytes per loop, and have at least 32 threads per block (32 * 2 = 64),
        // ! and deal with strings only under 64k bytes, this loop will fire at most 1K times per input
        for (unsigned i = tasks_offset * 2; i < tasks_count; i += tasks_step * 2) { // ! it's OK to spill beyond bounds
            pre_substitution_vec.u16s[0] = scores_pre_substitution[i + 0];
            pre_substitution_vec.u16s[1] = scores_pre_substitution[i + 1];
            pre_insertion_opening_vec.u16s[0] = scores_pre_insertion[i + 0];
            pre_insertion_opening_vec.u16s[1] = scores_pre_insertion[i + 1];
            pre_deletion_opening_vec.u16s[0] = scores_pre_deletion[i + 0];
            pre_deletion_opening_vec.u16s[1] = scores_pre_deletion[i + 1];
            pre_insertion_expansion_vec.u16s[0] = scores_running_insertions[i + 0];
            pre_insertion_expansion_vec.u16s[1] = scores_running_insertions[i + 1];
            pre_deletion_expansion_vec.u16s[0] = scores_running_deletions[i + 0];
            pre_deletion_expansion_vec.u16s[1] = scores_running_deletions[i + 1];
            first_vec.u16s[0] = load_immutable_(first_slice + tasks_count - i - 1);
            first_vec.u16s[1] = load_immutable_(first_slice + tasks_count - i - 2); // ! this may be OOB
            second_vec.u16s[0] = load_immutable_(second_slice + i + 0);
            second_vec.u16s[1] = load_immutable_(second_slice + i + 1); // ! this may be OOB, but padded

            // Equality comparison will output 0xFFFF for each matching byte-pair.
            equality_vec.u32 = __vcmpeq2(first_vec.u32, second_vec.u32);
            cost_of_substitution_vec.u32 =                //
                (equality_vec.u32 & match_cost_vec.u32) + //
                (~equality_vec.u32 & mismatch_cost_vec.u32);
            if_substitution_vec.u32 = __vaddus2(pre_substitution_vec.u32, cost_of_substitution_vec.u32);
            if_insertion_vec.u32 = //
                __viaddmin_u16x2(pre_insertion_opening_vec.u32, gap_open_cost_vec.u32,
                                 __vaddus2(pre_insertion_expansion_vec.u32, gap_extend_cost_vec.u32));
            if_deletion_vec.u32 = //
                __viaddmin_u16x2(pre_deletion_opening_vec.u32, gap_open_cost_vec.u32,
                                 __vaddus2(pre_deletion_expansion_vec.u32, gap_extend_cost_vec.u32));
            cell_score_vec.u32 = __vimin3_u16x2(if_substitution_vec.u32, if_insertion_vec.u32, if_deletion_vec.u32);

            // When walking through the top-left triangle of the matrix, our output addresses are misaligned.
            scores_new[i + 0] = cell_score_vec.u16s[0];
            scores_new[i + 1] = cell_score_vec.u16s[1];
            scores_new_insertions[i + 0] = if_insertion_vec.u16s[0];
            scores_new_insertions[i + 1] = if_insertion_vec.u16s[1];
            scores_new_deletions[i + 0] = if_deletion_vec.u16s[0];
            scores_new_deletions[i + 1] = if_deletion_vec.u16s[1];
        }

        // Extract the bottom-right corner of the matrix, which is the result of the global alignment.
        if (tasks_offset == 0) this->final_score_ = scores_new[0];
    }
};

template <>
struct tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, u32_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
};

template <>
struct tile_scorer<char const *, char const *, u64_t, uniform_substitution_costs_t, affine_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, u64_t, uniform_substitution_costs_t, affine_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using cuda_warp_scorer_t::tile_scorer; // Make the constructors visible
};

#endif

/**
 *  @brief  String similarity scoring algorithm evaluating a @b single Dynamic Programming matrix
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
    cuda_warp_scorer_t diagonal_aligner {substituter, gap_costs};
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
 *  @brief  String similarity scoring algorithm evaluating a @b single Dynamic Programming matrix
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
    cuda_warp_scorer_t diagonal_aligner {substituter, gap_costs};
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
 *  @brief  Levenshtein edit distances algorithm evaluating the Dynamic Programming matrix
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
    size_t const global_thread_index = static_cast<unsigned>(blockIdx.x * blockDim.x + threadIdx.x);
    size_t const global_warp_index = static_cast<unsigned>(global_thread_index / warp_size);
    size_t const warps_per_block = static_cast<unsigned>(blockDim.x / warp_size);
    size_t const warps_per_device = static_cast<unsigned>(gridDim.x * warps_per_block);
    unsigned const thread_in_warp_index = static_cast<unsigned>(global_thread_index % warp_size);

    // Allocating shared memory is handled on the host side.
    extern __shared__ char shared_memory_for_block[];
    char *const shared_memory_for_warp =
        shared_memory_for_block + (global_warp_index % warps_per_block) * (shared_memory_size / warps_per_block);

    // Only one thread will be initializing the top row and left column and outputting the result.
    bool const is_main_thread = thread_in_warp_index == 0;

    // We are computing N edit distances for N pairs of strings. Not a cartesian product!
    // Each block/warp may end up receiving a different number of strings.
    for (size_t task_idx = global_warp_index; task_idx < tasks_count; task_idx += warps_per_device) {
        task_t &task = tasks[task_idx];
        char_t const *shorter_global = task.shorter_ptr;
        char_t const *longer_global = task.longer_ptr;
        size_t const shorter_length = task.shorter_length;
        size_t const longer_length = task.longer_length;
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
        char_t *const longer = reinterpret_cast<char_t *>(shared_memory_for_warp + 3 * bytes_per_diagonal);
        char_t *const shorter = longer + longer_length;

        // Each thread in the warp will be loading it's own set of strided characters into shared memory.
        for (unsigned i = thread_in_warp_index; i < longer_length; i += warp_size) longer[i] = longer_global[i];
        for (unsigned i = thread_in_warp_index; i < shorter_length; i += warp_size) shorter[i] = shorter_global[i];

        // Initialize the first two diagonals:
        cuda_warp_scorer_t diagonal_aligner {substituter, gap_costs};
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
            for (size_t i = thread_in_warp_index; i + 1 < next_diagonal_length; i += warp_size)
                previous_scores[i] = current_scores[i + 1];
            __syncwarp();
            for (size_t i = thread_in_warp_index; i < next_diagonal_length; i += warp_size)
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
 *  @brief  Levenshtein edit distances algorithm evaluating the Dynamic Programming matrix
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
    size_t const global_thread_index = static_cast<unsigned>(blockIdx.x * blockDim.x + threadIdx.x);
    size_t const global_warp_index = static_cast<unsigned>(global_thread_index / warp_size);
    size_t const warps_per_block = static_cast<unsigned>(blockDim.x / warp_size);
    size_t const warps_per_device = static_cast<unsigned>(gridDim.x * warps_per_block);
    unsigned const thread_in_warp_index = static_cast<unsigned>(global_thread_index % warp_size);

    // Allocating shared memory is handled on the host side.
    extern __shared__ char shared_memory_for_block[];
    char *const shared_memory_for_warp =
        shared_memory_for_block + (global_warp_index % warps_per_block) * (shared_memory_size / warps_per_block);

    // Only one thread will be initializing the top row and left column and outputting the result.
    bool const is_main_thread = thread_in_warp_index == 0;

    // We are computing N edit distances for N pairs of strings. Not a cartesian product!
    // Each block/warp may end up receiving a different number of strings.
    for (size_t task_idx = global_warp_index; task_idx < tasks_count; task_idx += warps_per_device) {
        task_t &task = tasks[task_idx];
        char_t const *shorter_global = task.shorter_ptr;
        char_t const *longer_global = task.longer_ptr;
        size_t const shorter_length = task.shorter_length;
        size_t const longer_length = task.longer_length;
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
        char_t *const longer = reinterpret_cast<char_t *>(shared_memory_for_warp + 7 * bytes_per_diagonal);
        char_t *const shorter = longer + longer_length;

        // Each thread in the warp will be loading it's own set of strided characters into shared memory.
        for (unsigned i = thread_in_warp_index; i < longer_length; i += warp_size) longer[i] = longer_global[i];
        for (unsigned i = thread_in_warp_index; i < shorter_length; i += warp_size) shorter[i] = shorter_global[i];

        // Initialize the first two diagonals:
        cuda_warp_scorer_t diagonal_aligner {substituter, gap_costs};
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
            for (size_t i = thread_in_warp_index; i + 1 < next_diagonal_length; i += warp_size)
                previous_scores[i] = current_scores[i + 1];
            __syncwarp();
            for (size_t i = thread_in_warp_index; i < next_diagonal_length; i += warp_size)
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
};

/**
 *  @brief  Dispatches baseline Levenshtein edit distance algorithm to the GPU.
 *          Before starting the kernels, bins them by size to maximize the number of blocks
 *          per grid that can run simultaneously, while fitting into the shared memory.
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

    levenshtein_distances(uniform_substitution_costs_t subs = {}, gap_costs_t gaps = {},
                          allocator_t const &alloc = {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
#if SZ_HAS_CONCEPTS_
        requires indexed_results_like<results_type_>
#endif
    cuda_status_t operator()(                                                                 //
        first_strings_type_ const &first_strings, second_strings_type_ const &second_strings, //
        results_type_ &&results,                                                              //
        cuda_executor_t executor = {}, gpu_specs_t specs = {}) const noexcept {

        constexpr bool is_affine_k = is_same_type<gap_costs_t, affine_gap_costs_t>::value;
        constexpr size_t count_diagonals_k = is_affine_k ? 7 : 3;

        // Preallocate the events for GPU timing.
        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event, cudaEventBlockingSync);
        cudaEventCreate(&stop_event, cudaEventBlockingSync);

        using final_score_t = typename indexed_results_type<results_type_>::type;
        safe_vector<task_t, tasks_allocator_t> tasks(alloc_);
        if (tasks.try_resize(first_strings.size()) == status_t::bad_alloc_k) return {status_t::bad_alloc_k};

        // Record the start event
        cudaError_t start_event_error = cudaEventRecord(start_event, executor.stream());
        if (start_event_error != cudaSuccess) return {status_t::unknown_k, start_event_error};

        // Export all the tasks and sort them by decreasing memory requirement.
        size_t count_empty_tasks = 0;
        using similarity_memory_requirements_t = similarity_memory_requirements<size_t, false>;
        for (size_t i = 0; i < first_strings.size(); ++i) {
            // Ensure inputs are device-accessible (Unified/Device memory)
            if (!is_device_accessible_memory((void const *)first_strings[i].data()) ||
                !is_device_accessible_memory((void const *)second_strings[i].data()))
                return {status_t::device_memory_mismatch_k, cudaSuccess};
            task_t task(                                            //
                first_strings[i].data(), first_strings[i].length(), //
                second_strings[i].data(), second_strings[i].length());
            similarity_memory_requirements_t requirement(                                  //
                task.shorter_length, task.longer_length,                                   //
                gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
                sizeof(char_t), 4);

            task.original_index = i;
            task.memory_requirement = requirement.total;
            task.bytes_per_cell = requirement.bytes_per_cell;
            task.density = warp_tasks_density(requirement.total, specs);
            if (task.density == infinite_warps_per_multiprocessor_k) {
                if constexpr (!is_affine_k) { task.result = task.longer_length * gap_costs_.open_or_extend; }
                else if (!task.longer_length) { task.result = 0; }
                else { task.result = (task.longer_length - 1) * gap_costs_.extend + gap_costs_.open; }
                count_empty_tasks++;
            }
            tasks[i] = task;
        }

        auto [device_level_tasks, warp_level_tasks, empty_tasks] =
            warp_tasks_grouping<task_t>({tasks.data(), tasks.size()}, specs);

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
            safe_vector<u64_t, scores_allocator_t> diagonals_u64_buffer(alloc_);
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

                // TODO: We can be wiser about the dimensions of this grid.
                unsigned const random_block_size = 128;
                unsigned const random_blocks_per_multiprocessor = 32;
                cudaError_t launch_error = cudaLaunchCooperativeKernel(                       //
                    reinterpret_cast<void *>(device_level_kernel),                            // Kernel function pointer
                    dim3(random_blocks_per_multiprocessor * specs.streaming_multiprocessors), // Grid dimensions
                    dim3(random_block_size),                                                  // Block dimensions
                    device_level_kernel_args, // Array of kernel argument pointers
                    0,                        // Shared memory per block (in bytes)
                    executor.stream());       // CUDA stream
                if (launch_error != cudaSuccess)
                    if (launch_error == cudaErrorMemoryAllocation) { return {status_t::bad_alloc_k, launch_error}; }
                    else { return {status_t::unknown_k, launch_error}; }
            }
        }

        // Now process remaining warp-level tasks, checking warp densities in reverse order.
        // From the highest possible number of warps per multiprocessor to the lowest.
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
            auto const task_group_callback = [&](task_t const *tasks_begin, task_t const *tasks_end) {
                // Check if we need to stop processing.
                if (result.status != status_t::success_k) return;

                // Make sure all tasks can be handled by the same kernel template.
                task_t const &first_task = *tasks_begin;
                sz_assert_(std::all_of(tasks_begin, tasks_end, [&](task_t const &task) {
                    return task.bytes_per_cell == first_task.bytes_per_cell && task.density == first_task.density;
                }));

                // Find the task in the batch that requires the most memory.
                task_t const &indicative_task =
                    *std::max_element(tasks_begin, tasks_end, [](task_t const &lhs, task_t const &rhs) {
                        return lhs.memory_requirement < rhs.memory_requirement;
                    });

                // Pick the smallest fitting type for the diagonals.
                void *warp_level_kernel = reinterpret_cast<void *>(warp_level_u8_kernel);
                if (indicative_task.bytes_per_cell >= sizeof(u16_t))
                    warp_level_kernel = reinterpret_cast<void *>(warp_level_u16_kernel);

                // Even if we can fit more warps per block we sometimes should not.
                auto const [optimal_density, speculative_factor] =
                    speculation_friendly_density(indicative_task.density);

                // Update the selected kernels properties.
                unsigned const shared_memory_per_block =
                    static_cast<unsigned>(indicative_task.memory_requirement * optimal_density);
                sz_assert_(shared_memory_per_block > 0);
                sz_assert_(shared_memory_per_block < specs.shared_memory_per_multiprocessor());
                cudaError_t attribute_error = cudaFuncSetAttribute(
                    warp_level_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_per_block);
                if (attribute_error != cudaSuccess) {
                    result = {status_t::unknown_k, attribute_error};
                    return;
                }

                size_t const count_tasks = tasks_end - tasks_begin;
                warp_level_kernel_args[0] = (void *)(&tasks_begin);
                warp_level_kernel_args[1] = (void *)(&count_tasks);
                warp_level_kernel_args[2] = (void *)(&substituter_);
                warp_level_kernel_args[3] = (void *)(&gap_costs_);
                warp_level_kernel_args[4] = (void *)(&shared_memory_per_block);

                // Warp-level algorithm clearly aligns with the warp size.
                unsigned const threads_per_block = static_cast<unsigned>(specs.warp_size * optimal_density);
                cudaError_t launch_error = cudaLaunchKernel(                    //
                    reinterpret_cast<void *>(warp_level_kernel),                // Kernel function pointer
                    dim3(specs.streaming_multiprocessors * speculative_factor), // Grid dimensions
                    dim3(threads_per_block),                                    // Block dimensions
                    warp_level_kernel_args,                                     // Array of kernel argument pointers
                    shared_memory_per_block,                                    // Shared memory per block (in bytes)
                    executor.stream());                                         // CUDA stream
                if (launch_error != cudaSuccess) {
                    result = {launch_error == cudaErrorMemoryAllocation ? status_t::bad_alloc_k : status_t::unknown_k,
                              launch_error};
                    return;
                }

                // Wait until everything completes, as on the next iteration we will update the properties again.
                cudaError_t execution_error = cudaStreamSynchronize(executor.stream());
                if (execution_error != cudaSuccess) {
                    result = {status_t::unknown_k, execution_error};
                    return;
                }
            };
            group_by(warp_level_tasks.begin(), warp_level_tasks.end(), task_size_equality, task_group_callback);
            if (result.status != status_t::success_k) return result;
        }

        // Calculate the duration:
        cudaError_t stop_event_error = cudaEventRecord(stop_event, executor.stream());
        if (stop_event_error != cudaSuccess) return {status_t::unknown_k, stop_event_error};
        float execution_milliseconds = 0;
        cudaEventElapsedTime(&execution_milliseconds, start_event, stop_event);

        // Now that everything went well, export the results back into the `results` array.
        for (size_t i = 0; i < tasks.size(); ++i) {
            task_t const &task = tasks[i];
            results[task.original_index] = task.result;
        }
        return {status_t::success_k, cudaSuccess, execution_milliseconds};
    }
};

#pragma endregion

#pragma region - Needleman Wunsch and Smith Waterman Scores in CUDA

/**
 *  @brief  Convenience buffer of the size matching the size of the CUDA constant memory,
 *          used to cheaper store and access the substitution costs for the characters.
 *  @see    CUDA constant memory docs: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#constant
 */
__constant__ char error_costs_in_cuda_constant_memory_[256 * 256];

struct error_costs_256x256_in_cuda_constant_memory_t {
    __host__ error_cost_t magnitude() const noexcept { return 0; }
    __forceinline__ __host__ __device__ error_cost_t operator()(char a, char b) const noexcept {
#if defined(__CUDA_ARCH__)
        return error_costs_in_cuda_constant_memory_[static_cast<u8_t>(a) * 256 + static_cast<u8_t>(b)];
#else
        sz_unused_(a && b);
        return 0;
#endif
    }
};

#if SZ_USE_HOPPER

/**
 *  @brief GPU adaptation of the `tile_scorer` - Maximizes Global or Local score with linear gap costs.
 *  @note Requires Hopper generation GPUs to handle 2x `i16` scores at a time.
 */
template <sz_similarity_locality_t locality_>
struct tile_scorer<char const *, char const *, sz_i16_t, error_costs_256x256_in_cuda_constant_memory_t,
                   linear_gap_costs_t, sz_maximize_score_k, locality_, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, sz_i16_t, error_costs_256x256_in_cuda_constant_memory_t,
                         linear_gap_costs_t, sz_maximize_score_k, locality_, sz_cap_cuda_k> {

    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;

    using tile_scorer<char const *, char const *, sz_i16_t, error_costs_256x256_in_cuda_constant_memory_t,
                      linear_gap_costs_t, sz_maximize_score_k, locality_,
                      sz_cap_cuda_k>::tile_scorer; // Make the constructors visible

    __forceinline__ __device__ void operator()(            //
        char const *first_slice, char const *second_slice, //
        unsigned const tasks_offset, unsigned const tasks_step,
        unsigned const tasks_count,              // ! Unlike CPU, uses `unsigned`
        sz_i16_t const *scores_pre_substitution, //
        sz_i16_t const *scores_pre_insertion,    //
        sz_i16_t const *scores_pre_deletion,     //
        sz_i16_t *scores_new) noexcept {

        error_costs_256x256_in_cuda_constant_memory_t substituter;
        sz_i16_t const gap_cost = this->gap_costs_.open_or_extend;
        sz_u32_vec_t gap_cost_vec;
        gap_cost_vec.i16s[0] = gap_cost_vec.i16s[1] = gap_cost;

        // The hardest part of this kernel is dealing with unaligned loads!
        // We want to minimize single-byte processing in favor of 2-byte SIMD loads and min/max operations.
        // Assuming we are reading consecutive values from a buffer, in every cycle, most likely, we will be
        // dealing with most values being unaligned!
        sz_u32_vec_t pre_substitution_vec, pre_insertion_vec, pre_deletion_vec;
        sz_u32_vec_t first_vec, second_vec;
        sz_u32_vec_t cost_of_substitution_vec, if_deletion_or_insertion_vec;
        sz_u32_vec_t cell_score_vec, final_score_vec;
        final_score_vec.i16s[0] = final_score_vec.i16s[1] = 0;

        // ! As we are processing 2 bytes per loop, and have at least 32 threads per block (32 * 2 = 64),
        // ! and deal with strings only under 64k bytes, this loop will fire at most 1K times per input
        for (unsigned i = tasks_offset * 2; i < tasks_count; i += tasks_step * 2) { // ! it's OK to spill beyond bounds
            pre_substitution_vec.i16s[0] = load_last_use_(scores_pre_substitution + i + 0);
            pre_substitution_vec.i16s[1] = load_last_use_(scores_pre_substitution + i + 1);
            pre_insertion_vec.i16s[0] = scores_pre_insertion[i + 0];
            pre_insertion_vec.i16s[1] = scores_pre_insertion[i + 1];
            pre_deletion_vec.i16s[0] = scores_pre_deletion[i + 0];
            pre_deletion_vec.i16s[1] = scores_pre_deletion[i + 1];
            first_vec.u16s[0] = load_immutable_(first_slice + tasks_count - i - 1);
            first_vec.u16s[1] = load_immutable_(first_slice + tasks_count - i - 2); // ! this may be OOB
            second_vec.u16s[0] = load_immutable_(second_slice + i + 0);
            second_vec.u16s[1] = load_immutable_(second_slice + i + 1); // ! this may be OOB, but padded

            cost_of_substitution_vec.i16s[0] = substituter(first_vec.u16s[0], second_vec.u16s[0]);
            cost_of_substitution_vec.i16s[1] = substituter(first_vec.u16s[1], second_vec.u16s[1]);
            if_deletion_or_insertion_vec.u32 =
                __vaddss2(__vmaxs2(pre_insertion_vec.u32, pre_deletion_vec.u32), gap_cost_vec.u32);

            // For local scoring we should use the ReLU variants of 3-way `max`.
            if constexpr (locality_k == sz_similarity_global_k) {
                cell_score_vec.u32 = __viaddmax_s16x2(pre_substitution_vec.u32, cost_of_substitution_vec.u32,
                                                      if_deletion_or_insertion_vec.u32);
                sz_unused_(final_score_vec);
            }
            else {
                cell_score_vec.u32 = __viaddmax_s16x2_relu(pre_substitution_vec.u32, cost_of_substitution_vec.u32,
                                                           if_deletion_or_insertion_vec.u32);
                // In the last iteration of the loop the second half-word contains noise,
                // so we have to discard it from affecting the final score.
                bool const is_tail = i + 1 == tasks_count;
                final_score_vec.i16s[0] = (std::max)(cell_score_vec.i16s[0], final_score_vec.i16s[0]);
                final_score_vec.i16s[1] = (std::max)(cell_score_vec.i16s[1 - is_tail], final_score_vec.i16s[1]);
            }

            // When walking through the top-left triangle of the matrix, our output addresses are misaligned.
            scores_new[i + 0] = cell_score_vec.i16s[0];
            scores_new[i + 1] = cell_score_vec.i16s[1];
        }

        // Extract the bottom-right corner of the matrix, which is the result of the global alignment.
        if constexpr (locality_k == sz_similarity_global_k) {
            if (tasks_offset == 0) this->final_score_ = scores_new[0];
        }
        else { // Or the best score for local alignment.
            this->final_score_ = __vimax3_s32(this->final_score_, final_score_vec.i16s[0], final_score_vec.i16s[1]);
            // On Hopper we can use specialized warp reductions for up-to 32-bit values:
            // this->final_score_ = pick_best_in_warp_<sz_maximize_score_k>(this->final_score_);
            this->final_score_ = __reduce_max_sync(0xFFFFFFFF, this->final_score_);
        }
    }
};

template <sz_similarity_locality_t locality_>
struct tile_scorer<char const *, char const *, sz_i32_t, error_costs_256x256_in_cuda_constant_memory_t,
                   linear_gap_costs_t, sz_maximize_score_k, locality_, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, sz_i32_t, error_costs_256x256_in_cuda_constant_memory_t,
                         linear_gap_costs_t, sz_maximize_score_k, locality_, sz_cap_cuda_k> {

    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;

    using tile_scorer<char const *, char const *, sz_i32_t, error_costs_256x256_in_cuda_constant_memory_t,
                      linear_gap_costs_t, sz_maximize_score_k, locality_,
                      sz_cap_cuda_k>::tile_scorer; // Make the constructors visible

    __forceinline__ __device__ void operator()(            //
        char const *first_slice, char const *second_slice, //
        unsigned const tasks_offset, unsigned const tasks_step,
        unsigned const tasks_count,              // ! Unlike CPU, uses `unsigned`
        sz_i32_t const *scores_pre_substitution, //
        sz_i32_t const *scores_pre_insertion,    //
        sz_i32_t const *scores_pre_deletion,     //
        sz_i32_t *scores_new) noexcept {

        // Make sure we are called for an anti-diagonal traversal order
        sz_assert_(scores_pre_insertion + 1 == scores_pre_deletion);
        error_costs_256x256_in_cuda_constant_memory_t substituter;
        sz_i32_t const gap_costs = this->gap_costs_.open_or_extend;
        sz_i32_t final_score = 0;

        for (unsigned i = tasks_offset; i < tasks_count; i += tasks_step) {
            sz_i32_t pre_substitution = load_last_use_(scores_pre_substitution + i);
            sz_i32_t pre_insertion = scores_pre_insertion[i];
            sz_i32_t pre_deletion = scores_pre_deletion[i];
            char first_char = load_immutable_(first_slice + tasks_count - i - 1);
            char second_char = load_immutable_(second_slice + i);

            error_cost_t cost_of_substitution = substituter(first_char, second_char);
            sz_i32_t if_deletion_or_insertion = (std::max)(pre_deletion, pre_insertion) + gap_costs;
            sz_i32_t cell_score;

            // For local scoring we should use the ReLU variants of 3-way `max`.
            if constexpr (locality_k == sz_similarity_global_k) {
                cell_score = __viaddmax_s32(pre_substitution, cost_of_substitution, if_deletion_or_insertion);
                sz_unused_(final_score);
            }
            else {
                cell_score = __viaddmax_s32_relu(pre_substitution, cost_of_substitution, if_deletion_or_insertion);
                final_score = (std::max)(cell_score, final_score);
            }

            // When walking through the top-left triangle of the matrix, our output addresses are misaligned.
            scores_new[i] = cell_score;
        }

        // Extract the bottom-right corner of the matrix, which is the result of the global alignment.
        if constexpr (locality_k == sz_similarity_global_k) {
            if (tasks_offset == 0) this->final_score_ = scores_new[0];
        }
        else { // Or the best score for local alignment.
            this->final_score_ = (std::max)(this->final_score_, final_score);
            // On Hopper we can use specialized warp reductions for up-to 32-bit values:
            // this->final_score_ = pick_best_in_warp_<sz_maximize_score_k>(this->final_score_);
            this->final_score_ = __reduce_max_sync(0xFFFFFFFF, this->final_score_);
        }
    }
};

/**
 *  @brief GPU adaptation of the `tile_scorer` - Maximizes Global or Local score with affine gap costs.
 *  @note Requires Hopper generation GPUs to handle 2x `i16` scores at a time.
 */
template <sz_similarity_locality_t locality_>
struct tile_scorer<char const *, char const *, sz_i16_t, error_costs_256x256_in_cuda_constant_memory_t,
                   affine_gap_costs_t, sz_maximize_score_k, locality_, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, sz_i16_t, error_costs_256x256_in_cuda_constant_memory_t,
                         affine_gap_costs_t, sz_maximize_score_k, locality_, sz_cap_cuda_k> {

    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;

    using tile_scorer<char const *, char const *, sz_i16_t, error_costs_256x256_in_cuda_constant_memory_t,
                      affine_gap_costs_t, sz_maximize_score_k, locality_,
                      sz_cap_cuda_k>::tile_scorer; // Make the constructors visible

    __forceinline__ __device__ void operator()(            //
        char const *first_slice, char const *second_slice, //
        unsigned const tasks_offset, unsigned const tasks_step,
        unsigned const tasks_count,                // ! Unlike CPU, uses `unsigned`
        sz_i16_t const *scores_pre_substitution,   //
        sz_i16_t const *scores_pre_insertion,      //
        sz_i16_t const *scores_pre_deletion,       //
        sz_i16_t const *scores_running_insertions, //
        sz_i16_t const *scores_running_deletions,  //
        sz_i16_t *scores_new,                      //
        sz_i16_t *scores_new_insertions,           //
        sz_i16_t *scores_new_deletions) noexcept {

        error_costs_256x256_in_cuda_constant_memory_t substituter;
        sz_i16_t const gap_open_cost = this->gap_costs_.open;
        sz_i16_t const gap_extend_cost = this->gap_costs_.extend;
        sz_u32_vec_t gap_open_cost_vec, gap_extend_cost_vec;
        gap_open_cost_vec.i16s[0] = gap_open_cost_vec.i16s[1] = gap_open_cost;
        gap_extend_cost_vec.i16s[0] = gap_extend_cost_vec.i16s[1] = gap_extend_cost;

        // The hardest part of this kernel is dealing with unaligned loads!
        // We want to minimize single-byte processing in favor of 2-byte SIMD loads and min/max operations.
        // Assuming we are reading consecutive values from a buffer, in every cycle, most likely, we will be
        // dealing with most values being unaligned!
        sz_u32_vec_t pre_substitution_vec, pre_insertion_opening_vec, pre_deletion_opening_vec;
        sz_u32_vec_t pre_insertion_expansion_vec, pre_deletion_expansion_vec;
        sz_u32_vec_t first_vec, second_vec;
        sz_u32_vec_t cost_of_substitution_vec, if_substitution_vec, if_insertion_vec, if_deletion_vec;
        sz_u32_vec_t cell_score_vec, final_score_vec;
        final_score_vec.i16s[0] = final_score_vec.i16s[1] = 0;

        // ! As we are processing 2 bytes per loop, and have at least 32 threads per block (32 * 2 = 64),
        // ! and deal with strings only under 64k bytes, this loop will fire at most 1K times per input
        for (unsigned i = tasks_offset * 2; i < tasks_count; i += tasks_step * 2) { // ! it's OK to spill beyond bounds
            pre_substitution_vec.i16s[0] = load_last_use_(scores_pre_substitution + i + 0);
            pre_substitution_vec.i16s[1] = load_last_use_(scores_pre_substitution + i + 1);
            pre_insertion_opening_vec.i16s[0] = scores_pre_insertion[i + 0];
            pre_insertion_opening_vec.i16s[1] = scores_pre_insertion[i + 1];
            pre_deletion_opening_vec.i16s[0] = scores_pre_deletion[i + 0];
            pre_deletion_opening_vec.i16s[1] = scores_pre_deletion[i + 1];
            pre_insertion_expansion_vec.i16s[0] = scores_running_insertions[i + 0];
            pre_insertion_expansion_vec.i16s[1] = scores_running_insertions[i + 1];
            pre_deletion_expansion_vec.i16s[0] = scores_running_deletions[i + 0];
            pre_deletion_expansion_vec.i16s[1] = scores_running_deletions[i + 1];
            first_vec.u16s[0] = load_immutable_(first_slice + tasks_count - i - 1);
            first_vec.u16s[1] = load_immutable_(first_slice + tasks_count - i - 2); // ! this may be OOB
            second_vec.u16s[0] = load_immutable_(second_slice + i + 0);
            second_vec.u16s[1] = load_immutable_(second_slice + i + 1); // ! this may be OOB, but padded

            cost_of_substitution_vec.i16s[0] = substituter(first_vec.u16s[0], second_vec.u16s[0]);
            cost_of_substitution_vec.i16s[1] = substituter(first_vec.u16s[1], second_vec.u16s[1]);
            if_substitution_vec.u32 = __vaddss2(pre_substitution_vec.u32, cost_of_substitution_vec.u32);
            if_insertion_vec.u32 = //
                __viaddmax_s16x2(pre_insertion_opening_vec.u32, gap_open_cost_vec.u32,
                                 __vaddss2(pre_insertion_expansion_vec.u32, gap_extend_cost_vec.u32));
            if_deletion_vec.u32 = //
                __viaddmax_s16x2(pre_deletion_opening_vec.u32, gap_open_cost_vec.u32,
                                 __vaddss2(pre_deletion_expansion_vec.u32, gap_extend_cost_vec.u32));

            // For local scoring we should use the ReLU variants of 3-way `max`.
            if constexpr (locality_k == sz_similarity_global_k) {
                cell_score_vec.u32 = __vimax3_s16x2(if_substitution_vec.u32, if_insertion_vec.u32, if_deletion_vec.u32);
                sz_unused_(final_score_vec);
            }
            else {
                cell_score_vec.u32 =
                    __vimax3_s16x2_relu(if_substitution_vec.u32, if_insertion_vec.u32, if_deletion_vec.u32);
                // In the last iteration of the loop the second half-word contains noise,
                // so we have to discard it from affecting the final score.
                bool const is_tail = i + 1 == tasks_count;
                final_score_vec.i16s[0] = (std::max)(cell_score_vec.i16s[0], final_score_vec.i16s[0]);
                final_score_vec.i16s[1] = (std::max)(cell_score_vec.i16s[1 - is_tail], final_score_vec.i16s[1]);
            }

            // When walking through the top-left triangle of the matrix, our output addresses are misaligned.
            scores_new[i + 0] = cell_score_vec.i16s[0];
            scores_new[i + 1] = cell_score_vec.i16s[1];
            scores_new_insertions[i + 0] = if_insertion_vec.i16s[0];
            scores_new_insertions[i + 1] = if_insertion_vec.i16s[1];
            scores_new_deletions[i + 0] = if_deletion_vec.i16s[0];
            scores_new_deletions[i + 1] = if_deletion_vec.i16s[1];
        }

        // Extract the bottom-right corner of the matrix, which is the result of the global alignment.
        if constexpr (locality_k == sz_similarity_global_k) {
            if (tasks_offset == 0) this->final_score_ = scores_new[0];
        }
        else { // Or the best score for local alignment.
            this->final_score_ = __vimax3_s32(this->final_score_, final_score_vec.i16s[0], final_score_vec.i16s[1]);
            // On Hopper we can use specialized warp reductions for up-to 32-bit values:
            // this->final_score_ = pick_best_in_warp_<sz_maximize_score_k>(this->final_score_);
            this->final_score_ = __reduce_max_sync(0xFFFFFFFF, this->final_score_);
        }
    }
};

template <sz_similarity_locality_t locality_>
struct tile_scorer<char const *, char const *, sz_i32_t, error_costs_256x256_in_cuda_constant_memory_t,
                   affine_gap_costs_t, sz_maximize_score_k, locality_, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, sz_i32_t, error_costs_256x256_in_cuda_constant_memory_t,
                         affine_gap_costs_t, sz_maximize_score_k, locality_, sz_cap_cuda_k> {

    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;

    using tile_scorer<char const *, char const *, sz_i32_t, error_costs_256x256_in_cuda_constant_memory_t,
                      affine_gap_costs_t, sz_maximize_score_k, locality_,
                      sz_cap_cuda_k>::tile_scorer; // Make the constructors visible

    __forceinline__ __device__ void operator()(            //
        char const *first_slice, char const *second_slice, //
        unsigned const tasks_offset, unsigned const tasks_step,
        unsigned const tasks_count,                // ! Unlike CPU, uses `unsigned`
        sz_i32_t const *scores_pre_substitution,   //
        sz_i32_t const *scores_pre_insertion,      //
        sz_i32_t const *scores_pre_deletion,       //
        sz_i32_t const *scores_running_insertions, //
        sz_i32_t const *scores_running_deletions,  //
        sz_i32_t *scores_new,                      //
        sz_i32_t *scores_new_insertions,           //
        sz_i32_t *scores_new_deletions) noexcept {

        // Make sure we are called for an anti-diagonal traversal order
        sz_assert_(scores_pre_insertion + 1 == scores_pre_deletion);
        sz_i32_t const gap_open_cost = this->gap_costs_.open;
        sz_i32_t const gap_extend_cost = this->gap_costs_.extend;
        error_costs_256x256_in_cuda_constant_memory_t substituter;
        sz_i32_t final_score = 0;

        for (unsigned i = tasks_offset; i < tasks_count; i += tasks_step) {
            sz_i32_t pre_substitution = load_last_use_(scores_pre_substitution + i);
            sz_i32_t pre_insertion_opening = scores_pre_insertion[i];
            sz_i32_t pre_deletion_opening = scores_pre_deletion[i];
            sz_i32_t pre_insertion_expansion = scores_running_insertions[i];
            sz_i32_t pre_deletion_expansion = scores_running_deletions[i];
            char first_char = load_immutable_(first_slice + tasks_count - i - 1);
            char second_char = load_immutable_(second_slice + i);

            error_cost_t cost_of_substitution = substituter(first_char, second_char);
            sz_i32_t if_substitution = pre_substitution + cost_of_substitution;
            sz_i32_t if_insertion =
                __viaddmax_s32(pre_insertion_opening, gap_open_cost, pre_insertion_expansion + gap_extend_cost);
            sz_i32_t if_deletion =
                __viaddmax_s32(pre_deletion_opening, gap_open_cost, pre_deletion_expansion + gap_extend_cost);
            sz_i32_t cell_score;

            // For local scoring we should use the ReLU variants of 3-way `max`.
            if constexpr (locality_k == sz_similarity_global_k) {
                cell_score = __vimax3_s32(if_substitution, if_insertion, if_deletion);
                sz_unused_(final_score);
            }
            else {
                cell_score = __vimax3_s32_relu(if_substitution, if_insertion, if_deletion);
                final_score = (std::max)(cell_score, final_score);
            }

            // When walking through the top-left triangle of the matrix, our output addresses are misaligned.
            scores_new[i] = cell_score;
            scores_new_insertions[i] = if_insertion;
            scores_new_deletions[i] = if_deletion;
        }

        // Extract the bottom-right corner of the matrix, which is the result of the global alignment.
        if constexpr (locality_k == sz_similarity_global_k) {
            if (tasks_offset == 0) this->final_score_ = scores_new[0];
        }
        else { // Or the best score for local alignment.
            this->final_score_ = (std::max)(this->final_score_, final_score);
            // On Hopper we can use specialized warp reductions for up-to 32-bit values:
            // this->final_score_ = pick_best_in_warp_<sz_maximize_score_k>(this->final_score_);
            this->final_score_ = __reduce_max_sync(0xFFFFFFFF, this->final_score_);
        }
    }
};

template <sz_similarity_locality_t locality_>
struct tile_scorer<char const *, char const *, sz_i64_t, error_costs_256x256_in_cuda_constant_memory_t,
                   linear_gap_costs_t, sz_maximize_score_k, locality_, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, sz_i64_t, error_costs_256x256_in_cuda_constant_memory_t,
                         linear_gap_costs_t, sz_maximize_score_k, locality_, sz_cap_cuda_k> {

    using tile_scorer<char const *, char const *, sz_i64_t, error_costs_256x256_in_cuda_constant_memory_t,
                      linear_gap_costs_t, sz_maximize_score_k, locality_,
                      sz_cap_cuda_k>::tile_scorer; // Make the constructors visible
};

template <sz_similarity_locality_t locality_>
struct tile_scorer<char const *, char const *, sz_i64_t, error_costs_256x256_in_cuda_constant_memory_t,
                   affine_gap_costs_t, sz_maximize_score_k, locality_, sz_caps_ckh_k>
    : public tile_scorer<char const *, char const *, sz_i64_t, error_costs_256x256_in_cuda_constant_memory_t,
                         affine_gap_costs_t, sz_maximize_score_k, locality_, sz_cap_cuda_k> {

    using tile_scorer<char const *, char const *, sz_i64_t, error_costs_256x256_in_cuda_constant_memory_t,
                      affine_gap_costs_t, sz_maximize_score_k, locality_,
                      sz_cap_cuda_k>::tile_scorer; // Make the constructors visible
};

#endif

/**
 *  @brief  Dispatches baseline NW or SW scoring algorithm to the GPU.
 *          Before starting the kernels, bins them by size to maximize the number of blocks
 *          per grid that can run simultaneously, while fitting into the shared memory.
 *          Unlike the Levenshtein distances, also places byte-level @b `error_costs_256x256_t`
 *          substitution costs into the CUDA @b constant memory, addressing it via
 *          @b `error_costs_256x256_in_cuda_constant_memory_t` struct.
 */
template <typename gap_costs_type_, typename allocator_type_, sz_similarity_locality_t locality_,
          sz_capability_t capability_>
struct cuda_nw_or_sw_byte_level_scores_ {

    using char_t = char;
    using substituter_t = error_costs_256x256_t;
    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;
    using scores_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<size_t>;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = capability_;

    using task_t = cuda_similarity_task_<char_t>;
    using tasks_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<task_t>;

    error_costs_256x256_t substituter_ {};
    gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    cuda_nw_or_sw_byte_level_scores_(error_costs_256x256_t subs = {}, gap_costs_t gaps = {},
                                     allocator_t const &alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
#if SZ_HAS_CONCEPTS_
        requires indexed_results_like<results_type_>
#endif
    cuda_status_t operator()(                                                                 //
        first_strings_type_ const &first_strings, second_strings_type_ const &second_strings, //
        results_type_ &&results,                                                              //
        cuda_executor_t executor = {}, gpu_specs_t specs = {}) const noexcept {

        constexpr bool is_local_k = locality_k == sz_similarity_local_k;
        constexpr bool is_affine_k = is_same_type<gap_costs_t, affine_gap_costs_t>::value;
        constexpr size_t count_diagonals_k = is_affine_k ? 7 : 3;

        // Preallocate the events for GPU timing.
        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event, cudaEventBlockingSync);
        cudaEventCreate(&stop_event, cudaEventBlockingSync);

        using final_score_t = typename indexed_results_type<results_type_>::type;
        safe_vector<task_t, tasks_allocator_t> tasks(alloc_);
        if (tasks.try_resize(first_strings.size()) == status_t::bad_alloc_k) return {status_t::bad_alloc_k};

        // Record the start event
        cudaError_t start_event_error = cudaEventRecord(start_event, executor.stream());
        if (start_event_error != cudaSuccess) return {status_t::unknown_k, start_event_error};

        // Enqueue the transfer of the substituter to the constant memory:
        cudaError_t copy_error =
            cudaMemcpyToSymbolAsync(error_costs_in_cuda_constant_memory_, (void const *)&substituter_,
                                    sizeof(substituter_t), 0, cudaMemcpyHostToDevice, executor.stream());
        if (copy_error != cudaSuccess) return {status_t::unknown_k, copy_error};

        // Export all the tasks and sort them by decreasing memory requirement.
        size_t count_empty_tasks = 0;
        using similarity_memory_requirements_t = similarity_memory_requirements<size_t, true>;
        for (size_t i = 0; i < first_strings.size(); ++i) {
            // Ensure inputs are device-accessible (Unified/Device memory)
            if (!is_device_accessible_memory((void const *)first_strings[i].data()) ||
                !is_device_accessible_memory((void const *)second_strings[i].data()))
                return {status_t::device_memory_mismatch_k, cudaSuccess};
            task_t task(                                            //
                first_strings[i].data(), first_strings[i].length(), //
                second_strings[i].data(), second_strings[i].length());
            similarity_memory_requirements_t requirement(                                  //
                task.shorter_length, task.longer_length,                                   //
                gap_type<gap_costs_t>(), substituter_.magnitude(), gap_costs_.magnitude(), //
                sizeof(char_t), 4, two_bytes_per_cell_k);

            task.original_index = i;
            task.memory_requirement = requirement.total;
            task.bytes_per_cell = requirement.bytes_per_cell;
            task.density = warp_tasks_density(requirement.total, specs);
            if (task.density == infinite_warps_per_multiprocessor_k) {
                if constexpr (is_local_k) { task.result = 0; }
                else if constexpr (!is_affine_k) { task.result = task.longer_length * gap_costs_.open_or_extend; }
                else if (!task.longer_length) { task.result = 0; }
                else { task.result = (task.longer_length - 1) * gap_costs_.extend + gap_costs_.open; }
                count_empty_tasks++;
            }
            tasks[i] = task;
        }

        auto [device_level_tasks, warp_level_tasks, empty_tasks] =
            warp_tasks_grouping<task_t>({tasks.data(), tasks.size()}, specs);

        if (device_level_tasks.size()) {
            auto device_level_i32_kernel =
                is_affine_k //
                    ? (void *)&affine_score_across_cuda_device<char_t, u32_t, sz_i32_t, final_score_t,
                                                               error_costs_256x256_in_cuda_constant_memory_t,
                                                               sz_maximize_score_k, locality_k, capability_k>
                    : (void *)&linear_score_across_cuda_device_<char_t, u32_t, sz_i32_t, final_score_t,
                                                                error_costs_256x256_in_cuda_constant_memory_t,
                                                                sz_maximize_score_k, locality_k, capability_k>;
            auto device_level_i64_kernel =
                is_affine_k //
                    ? (void *)&affine_score_across_cuda_device<char_t, u64_t, sz_i64_t, final_score_t,
                                                               error_costs_256x256_in_cuda_constant_memory_t,
                                                               sz_maximize_score_k, locality_k, capability_k>
                    : (void *)&linear_score_across_cuda_device_<char_t, u64_t, sz_i64_t, final_score_t,
                                                                error_costs_256x256_in_cuda_constant_memory_t,
                                                                sz_maximize_score_k, locality_k, capability_k>;
            void *device_level_kernel_args[8];

            // On very large inputs we can't fit the diagonals in shared memory, and use the global one.
            safe_vector<u64_t, scores_allocator_t> diagonals_u64_buffer(alloc_);
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
                void *device_level_kernel = reinterpret_cast<void *>(device_level_i32_kernel);
                if (task.bytes_per_cell >= sizeof(sz_i64_t))
                    device_level_kernel = reinterpret_cast<void *>(device_level_i64_kernel);

                // TODO: We can be wiser about the dimensions of this grid.
                unsigned const random_block_size = 128;
                unsigned const random_blocks_per_multiprocessor = 32;
                cudaError_t launch_error = cudaLaunchCooperativeKernel(                       //
                    reinterpret_cast<void *>(device_level_kernel),                            // Kernel function pointer
                    dim3(random_blocks_per_multiprocessor * specs.streaming_multiprocessors), // Grid dimensions
                    dim3(random_block_size),                                                  // Block dimensions
                    device_level_kernel_args, // Array of kernel argument pointers
                    0,                        // Shared memory per block (in bytes)
                    executor.stream());       // CUDA stream
                if (launch_error != cudaSuccess)
                    if (launch_error == cudaErrorMemoryAllocation) { return {status_t::bad_alloc_k, launch_error}; }
                    else { return {status_t::unknown_k, launch_error}; }
            }
        }

        // Now process remaining warp-level tasks, checking warp densities in reverse order.
        // From the highest possible number of warps per multiprocessor to the lowest.
        if (warp_level_tasks.size()) {
            auto warp_level_i16_kernel =
                is_affine_k ? (void *)&affine_score_on_each_cuda_warp_<task_t, char_t, u16_t, sz_i16_t,
                                                                       error_costs_256x256_in_cuda_constant_memory_t,
                                                                       sz_maximize_score_k, locality_k, capability_k>
                            : (void *)&linear_score_on_each_cuda_warp_<task_t, char_t, u16_t, sz_i16_t,
                                                                       error_costs_256x256_in_cuda_constant_memory_t,
                                                                       sz_maximize_score_k, locality_k, capability_k>;
            auto warp_level_i32_kernel =
                is_affine_k ? (void *)&affine_score_on_each_cuda_warp_<task_t, char_t, u32_t, sz_i32_t,
                                                                       error_costs_256x256_in_cuda_constant_memory_t,
                                                                       sz_maximize_score_k, locality_k, capability_k>
                            : (void *)&linear_score_on_each_cuda_warp_<task_t, char_t, u32_t, sz_i32_t,
                                                                       error_costs_256x256_in_cuda_constant_memory_t,
                                                                       sz_maximize_score_k, locality_k, capability_k>;
            void *warp_level_kernel_args[5];

            cuda_status_t result;
            auto const task_size_equality = [](task_t const &lhs, task_t const &rhs) {
                return lhs.bytes_per_cell == rhs.bytes_per_cell && lhs.density == rhs.density;
            };
            auto const task_group_callback = [&](task_t const *tasks_begin, task_t const *tasks_end) {
                // Check if we need to stop processing.
                if (result.status != status_t::success_k) return;

                // Make sure all tasks can be handled by the same kernel template.
                task_t const &first_task = *tasks_begin;
                sz_assert_(std::all_of(tasks_begin, tasks_end, [&](task_t const &task) {
                    return task.bytes_per_cell == first_task.bytes_per_cell && task.density == first_task.density;
                }));

                // Find the task in the batch that requires the most memory.
                task_t const &indicative_task =
                    *std::max_element(tasks_begin, tasks_end, [](task_t const &lhs, task_t const &rhs) {
                        return lhs.memory_requirement < rhs.memory_requirement;
                    });

                // Pick the smallest fitting type for the diagonals.
                void *warp_level_kernel = reinterpret_cast<void *>(warp_level_i16_kernel);
                if (indicative_task.bytes_per_cell >= sizeof(sz_i32_t))
                    warp_level_kernel = reinterpret_cast<void *>(warp_level_i32_kernel);

                // Even if we can fit more warps per block we sometimes should not.
                auto const [optimal_density, speculative_factor] =
                    speculation_friendly_density(indicative_task.density);

                // Update the selected kernels properties.
                unsigned const shared_memory_per_block =
                    static_cast<unsigned>(indicative_task.memory_requirement * optimal_density);
                sz_assert_(shared_memory_per_block > 0);
                sz_assert_(shared_memory_per_block < specs.shared_memory_per_multiprocessor());
                cudaError_t attribute_error = cudaFuncSetAttribute(
                    warp_level_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_per_block);
                if (attribute_error != cudaSuccess) {
                    result = {status_t::unknown_k, attribute_error};
                    return;
                }

                size_t const count_tasks = tasks_end - tasks_begin;
                warp_level_kernel_args[0] = (void *)(&tasks_begin);
                warp_level_kernel_args[1] = (void *)(&count_tasks);
                warp_level_kernel_args[2] = (void *)(&substituter_);
                warp_level_kernel_args[3] = (void *)(&gap_costs_);
                warp_level_kernel_args[4] = (void *)(&shared_memory_per_block);

                // Warp-level algorithm clearly aligns with the warp size.
                unsigned const threads_per_block = static_cast<unsigned>(specs.warp_size * optimal_density);
                cudaError_t launch_error = cudaLaunchKernel(                    //
                    reinterpret_cast<void *>(warp_level_kernel),                // Kernel function pointer
                    dim3(specs.streaming_multiprocessors * speculative_factor), // Grid dimensions
                    dim3(threads_per_block),                                    // Block dimensions
                    warp_level_kernel_args,                                     // Array of kernel argument pointers
                    shared_memory_per_block,                                    // Shared memory per block (in bytes)
                    executor.stream());                                         // CUDA stream
                if (launch_error != cudaSuccess) {
                    result = {launch_error == cudaErrorMemoryAllocation ? status_t::bad_alloc_k : status_t::unknown_k,
                              launch_error};
                    return;
                }

                // Wait until everything completes, as on the next iteration we will update the properties again.
                cudaError_t execution_error = cudaStreamSynchronize(executor.stream());
                if (execution_error != cudaSuccess) {
                    result = {status_t::unknown_k, execution_error};
                    return;
                }
            };
            group_by(warp_level_tasks.begin(), warp_level_tasks.end(), task_size_equality, task_group_callback);
            if (result.status != status_t::success_k) return result;
        }

        // Calculate the duration:
        cudaError_t stop_event_error = cudaEventRecord(stop_event, executor.stream());
        if (stop_event_error != cudaSuccess) return {status_t::unknown_k, stop_event_error};
        float execution_milliseconds = 0;
        cudaEventElapsedTime(&execution_milliseconds, start_event, stop_event);

        // Now that everything went well, export the results back into the `results` array.
        for (size_t task_index = 0; task_index < tasks.size(); ++task_index) {
            task_t const &task = tasks[task_index];
            results[task.original_index] = task.result;
        }
        return {status_t::success_k, cudaSuccess, execution_milliseconds};
    }
};

/** @brief Dispatches baseline Needleman Wunsch algorithm to the GPU. */
template <typename gap_costs_type_, typename allocator_type_, sz_capability_t capability_>
struct needleman_wunsch_scores<char, error_costs_256x256_t, gap_costs_type_, allocator_type_, capability_,
                               std::enable_if_t<capability_ & sz_cap_cuda_k>>
    : public cuda_nw_or_sw_byte_level_scores_<gap_costs_type_, allocator_type_, sz_similarity_global_k, capability_> {

    using cuda_nw_or_sw_byte_level_scores_<gap_costs_type_, allocator_type_, sz_similarity_global_k,
                                           capability_>::cuda_nw_or_sw_byte_level_scores_;
};

/** @brief Dispatches baseline Smith Waterman algorithm to the GPU. */
template <typename gap_costs_type_, typename allocator_type_, sz_capability_t capability_>
struct smith_waterman_scores<char, error_costs_256x256_t, gap_costs_type_, allocator_type_, capability_,
                             std::enable_if_t<capability_ & sz_cap_cuda_k>>
    : public cuda_nw_or_sw_byte_level_scores_<gap_costs_type_, allocator_type_, sz_similarity_local_k, capability_> {

    using cuda_nw_or_sw_byte_level_scores_<gap_costs_type_, allocator_type_, sz_similarity_local_k,
                                           capability_>::cuda_nw_or_sw_byte_level_scores_;
};

#pragma endregion

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SIMILARITIES_CUH_
