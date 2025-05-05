/**
 *  @brief  CUDA-accelerated string similarity utilities.
 *  @file   similarity.cuh
 *  @author Ash Vardanian
 *
 *  Unlike th OpenMP backed, which also has single-pair similarity scores, the CUDA backend focuses @b only on
 *  batch-processing of large collections of strings, generally, assigning a single warp to each string pair:
 *
 *  - `sz::levenshtein_distances` & `sz::levenshtein_distances_utf8` for Levenshtein edit-distances.
 *  - `sz::needleman_wunsch_score` for weighted Needleman-Wunsch global alignment scores.
 *  - `sz::smith_waterman_score` for weighted Smith-Waterman local alignment scores.
 *
 *  Unlike the trivially parallelizable CPU kernels in `stringzilla/similarity.hpp`, the GPU kernels in this file are
 *  designed for batch-processing of large collections of strings, assigning a single warp to each string pair.
 *  Thus, they should be used when hundreds of pairwise comparisons are needed, and the strings are long enough to
 *  amortize the cost of copying them to the GPU.
 *
 *  @section    Abstraction layers
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
#ifndef STRINGCUZILLA_SIMILARITY_CUH_
#define STRINGCUZILLA_SIMILARITY_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>        // `cuda::pipeline`
#include <cooperative_groups.h> // `cooperative_groups::this_grid()`

#include "stringcuzilla/types.cuh"
#include "stringcuzilla/similarity.hpp"

namespace ashvardanian {
namespace stringzilla {

#pragma region - Common Aliases

using ualloc_t = unified_alloc<char>;

/**
 *  In @b CUDA:
 *  - for GPUs before Hopper, we can use the @b SIMT model for warp-level parallelism using diagonal "walkers"
 *  - for GPUs after Hopper, we compound that with thread-level @b SIMD via @b DPX instructions for min-max
 */
using levenshtein_cuda_t = levenshtein_distances<char, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
using levenshtein_utf8_cuda_t = levenshtein_distances_utf8<char, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
using needleman_wunsch_cuda_t =
    needleman_wunsch_scores<char, error_costs_256x256_t, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
using smith_waterman_cuda_t =
    smith_waterman_scores<char, error_costs_256x256_t, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k>;

using affine_levenshtein_cuda_t = levenshtein_distances<char, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
using affine_levenshtein_utf8_cuda_t = levenshtein_distances_utf8<char, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
using affine_needleman_wunsch_cuda_t =
    needleman_wunsch_scores<char, error_costs_256x256_t, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
using affine_smith_waterman_cuda_t =
    smith_waterman_scores<char, error_costs_256x256_t, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k>;

using levenshtein_kepler_t = levenshtein_distances<char, linear_gap_costs_t, ualloc_t, sz_caps_ck_k>;
using levenshtein_utf8_hopper_t = levenshtein_distances_utf8<char, linear_gap_costs_t, ualloc_t, sz_caps_ckh_k>;
using needleman_wunsch_hopper_t =
    needleman_wunsch_scores<char, error_costs_256x256_t, linear_gap_costs_t, ualloc_t, sz_caps_ckh_k>;
using smith_waterman_hopper_t =
    smith_waterman_scores<char, error_costs_256x256_t, linear_gap_costs_t, ualloc_t, sz_caps_ckh_k>;

#pragma endregion - Common Aliases

#pragma region - Algorithm Building Blocks

/**
 *  @brief GPU adaptation of the `tile_scorer` on CUDA, avoiding warp-level shuffles and DPX.
 *  @note Uses 32-bit `uint` counter to iterate through the string slices, so it can't be over 4 billion characters.
 */
template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_, sz_capability_t capability_>
#if _SZ_IS_CPP20
    requires pointer_like<first_iterator_type_> && pointer_like<second_iterator_type_> && score_like<score_type_> &&
             substituter_like<substituter_type_>
#endif
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, linear_gap_costs_t,
                   objective_, sz_similarity_global_k, capability_, std::enable_if_t<capability_ & sz_cap_cuda_k>> {

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
    using char_t = typename std::remove_cvref<first_char_t>::type;

    using warp_scorer_t = tile_scorer<first_iterator_t, second_iterator_t, score_t, substituter_t, linear_gap_costs_t,
                                      objective_k, sz_similarity_global_k, capability_k>;

  protected:
    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    score_t last_cell_ {0};

    __forceinline__ __device__ score_t pick_best(score_t a, score_t b) const noexcept {
        if constexpr (objective_ == sz_minimize_distance_k) { return std::min(a, b); }
        else { return std::max(a, b); }
    }

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
    __forceinline__ __device__ score_t score() const noexcept { return last_cell_; }

    /**
     *  @brief Computes one diagonal of the DP matrix, using the results of the previous 2x diagonals.
     *  @param first_slice The first string, unlike the CPU variant @b NOT reversed.
     *  @param second_slice The second string.
     *
     *  @param tasks_offset The offset of the first character to compare from each string.
     *  @param tasks_step The step size for the next character to compare from each string.
     *  @param tasks_count The total number of characters to compare from input slices.
     *
     *  @tparam index_type_ @b `uint` is recommended if the strings are under 4 billion characters.
     */
    template <typename index_type_>
    __forceinline__ __device__ void operator()(                                                      //
        first_iterator_t first_slice, second_iterator_t second_slice,                                //
        index_type_ const tasks_offset, index_type_ const tasks_step, index_type_ const tasks_count, //
        score_t const *scores_pre_substitution, score_t const *scores_pre_insertion,                 //
        score_t const *scores_pre_deletion, score_t *scores_new) noexcept {

        // Make sure we are called for an anti-diagonal traversal order
        score_t const gap_costs = gap_costs_.open_or_extend;
        _sz_assert(scores_pre_insertion + 1 == scores_pre_deletion);

        // ? One weird observation, is that even though we can avoid fetching `pre_insertion`
        // ? from shared memory on each cycle, by slicing the work differently between the threads,
        // ? and allowing them to reuse the previous `pre_deletion` as the new `pre_insertion`,
        // ? that code ends up being slower than the one below.
        for (index_type_ i = tasks_offset; i < tasks_count; i += tasks_step) {
            score_t pre_substitution = scores_pre_substitution[i];
            score_t pre_insertion = scores_pre_insertion[i];
            score_t pre_deletion = scores_pre_deletion[i];

            error_cost_t cost_of_substitution = substituter_(first_slice[tasks_count - i - 1], second_slice[i]);
            score_t if_substitution = pre_substitution + cost_of_substitution;
            score_t if_deletion_or_insertion = pick_best(pre_deletion, pre_insertion) + gap_costs;
            score_t cell_score = pick_best(if_deletion_or_insertion, if_substitution);
            scores_new[i] = cell_score;
        }

        // The last element of the last chunk is the result of the global alignment.
        if (tasks_offset == 0) last_cell_ = scores_new[0];
    }
};

/**
 *  @brief GPU adaptation of the `local_scorer` on CUDA, avoiding warp-level shuffles and DPX.
 *  @note Uses 32-bit `uint` counter to iterate through the string slices, so it can't be over 4 billion characters.
 */
template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_, sz_capability_t capability_>
#if _SZ_IS_CPP20
    requires pointer_like<first_iterator_type_> && pointer_like<second_iterator_type_> && score_like<score_type_> &&
             substituter_like<substituter_type_>
#endif
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, linear_gap_costs_t,
                   objective_, sz_similarity_local_k, capability_, std::enable_if_t<capability_ & sz_cap_cuda_k>> {

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
    using char_t = typename std::remove_cvref<first_char_t>::type;

    using warp_scorer_t = tile_scorer<first_iterator_t, second_iterator_t, score_t, substituter_t, linear_gap_costs_t,
                                      objective_k, sz_similarity_local_k, capability_k>;

  protected:
    substituter_t substituter_ {};
    linear_gap_costs_t gap_costs_ {};
    score_t best_score_ {0};

    __forceinline__ __device__ score_t pick_best(score_t a, score_t b) const noexcept {
        if constexpr (objective_k == sz_minimize_distance_k) { return std::min(a, b); }
        else { return std::max(a, b); }
    }

    __forceinline__ __device__ score_t pick_best_in_warp(score_t val) const noexcept {
        // The `__shfl_down_sync` replaces `__shfl_down`
        // https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
        val = pick_best(__shfl_down_sync(0xffffffff, val, 16));
        val = pick_best(__shfl_down_sync(0xffffffff, val, 8));
        val = pick_best(__shfl_down_sync(0xffffffff, val, 4));
        val = pick_best(__shfl_down_sync(0xffffffff, val, 2));
        val = pick_best(__shfl_down_sync(0xffffffff, val, 1));
        return val;
    }

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
    __forceinline__ __device__ score_t score() const noexcept { return best_score_; }

    /**
     *  @brief Computes one diagonal of the DP matrix, using the results of the previous 2x diagonals.
     *  @param first_slice The first string, unlike the CPU variant @b NOT reversed.
     *  @param second_slice The second string.
     *
     *  @param tasks_offset The offset of the first character to compare from each string.
     *  @param tasks_step The step size for the next character to compare from each string.
     *  @param tasks_count The total number of characters to compare from input slices.
     *
     *  @tparam index_type_ @b `uint` is recommended if the strings are under 4 billion characters.
     */
    template <typename index_type_>
    __forceinline__ __device__ void operator()(                                                      //
        first_iterator_t first_slice, second_iterator_t second_slice,                                //
        index_type_ const tasks_offset, index_type_ const tasks_step, index_type_ const tasks_count, //
        score_t const *scores_pre_substitution, score_t const *scores_pre_insertion,                 //
        score_t const *scores_pre_deletion, score_t *scores_new) noexcept {

        // Make sure we are called for an anti-diagonal traversal order
        error_cost_t const gap_cost = gap_costs_.open_or_extend;
        _sz_assert(scores_pre_insertion + 1 == scores_pre_deletion);

        // ? One weird observation, is that even though we can avoid fetching `pre_insertion`
        // ? from shared memory on each cycle, by slicing the work differently between the threads,
        // ? and allowing them to reuse the previous `pre_deletion` as the new `pre_insertion`,
        // ? that code ends up being slower than the one below.
        for (index_type_ i = tasks_offset; i < tasks_count; i += tasks_step) {
            score_t pre_substitution = scores_pre_substitution[i];
            score_t pre_insertion = scores_pre_insertion[i];
            score_t pre_deletion = scores_pre_deletion[i];

            error_cost_t cost_of_substitution = substituter_(first_slice[tasks_count - i - 1], second_slice[i]);
            score_t if_substitution = pre_substitution + cost_of_substitution;
            score_t if_deletion_or_insertion = pick_best(pre_deletion, pre_insertion) + gap_cost;
            score_t if_substitution_or_reset = pick_best(if_substitution, 0);
            score_t cell_score = pick_best(if_deletion_or_insertion, if_substitution_or_reset);
            scores_new[i] = cell_score;

            // Update the global maximum score if this cell beats it.
            best_score_ = pick_best(best_score_, cell_score);
        }

        // ! Don't forget to pick the best among the best scores per thread.
        best_score_ = pick_best_in_warp(best_score_);
    }
};

/**
 *  @brief GPU adaptation of the `tile_scorer` on CUDA, avoiding warp-level shuffles and DPX.
 *  @note Uses 32-bit `uint` counter to iterate through the string slices, so it can't be over 4 billion characters.
 */
template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_, sz_capability_t capability_>
#if _SZ_IS_CPP20
    requires pointer_like<first_iterator_type_> && pointer_like<second_iterator_type_> && score_like<score_type_> &&
             substituter_like<substituter_type_>
#endif
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, affine_gap_costs_t,
                   objective_, sz_similarity_global_k, capability_, std::enable_if_t<capability_ & sz_cap_cuda_k>> {

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
    using char_t = typename std::remove_cvref<first_char_t>::type;

    using warp_scorer_t = tile_scorer<first_iterator_t, second_iterator_t, score_t, substituter_t, affine_gap_costs_t,
                                      objective_k, sz_similarity_global_k, capability_k>;

  protected:
    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};
    score_t last_cell_ {0};

    __forceinline__ __device__ score_t pick_best(score_t a, score_t b) const noexcept {
        if constexpr (objective_ == sz_minimize_distance_k) { return std::min(a, b); }
        else { return std::max(a, b); }
    }

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
    __forceinline__ __device__ score_t score() const noexcept { return last_cell_; }

    /**
     *  @brief Computes one diagonal of the DP matrix, using the results of the previous 2x diagonals.
     *  @param first_slice The first string, unlike the CPU variant @b NOT reversed.
     *  @param second_slice The second string.
     *
     *  @param tasks_offset The offset of the first character to compare from each string.
     *  @param tasks_step The step size for the next character to compare from each string.
     *  @param tasks_count The total number of characters to compare from input slices.
     *
     *  @tparam index_type_ @b `uint` is recommended if the strings are under 4 billion characters.
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
        _sz_assert(scores_pre_insertion + 1 == scores_pre_deletion);

        // ? One weird observation, is that even though we can avoid fetching `pre_insertion`
        // ? from shared memory on each cycle, by slicing the work differently between the threads,
        // ? and allowing them to reuse the previous `pre_deletion` as the new `pre_insertion`,
        // ? that code ends up being slower than the one below.
        for (index_type_ i = tasks_offset; i < tasks_count; i += tasks_step) {
            score_t pre_substitution = scores_pre_substitution[i];
            score_t pre_insertion_opening = scores_pre_insertion[i];
            score_t pre_deletion_opening = scores_pre_deletion[i];
            score_t pre_insertion_expansion = scores_running_insertions[i];
            score_t pre_deletion_expansion = scores_running_deletions[i];

            error_cost_t cost_of_substitution = substituter_(first_slice[tasks_count - i - 1], second_slice[i]);
            score_t if_substitution = pre_substitution + cost_of_substitution;
            score_t if_insertion = min_or_max<objective_k>(pre_insertion_opening + gap_costs_.open,
                                                           pre_insertion_expansion + gap_costs_.extend);
            score_t if_deletion = min_or_max<objective_k>(pre_deletion_opening + gap_costs_.open,
                                                          pre_deletion_expansion + gap_costs_.extend);
            score_t if_deletion_or_insertion = min_or_max<objective_k>(if_deletion, if_insertion);
            score_t cell_score = pick_best(if_deletion_or_insertion, if_substitution);

            // Export results.
            scores_new[i] = cell_score;
            scores_new_insertions[i] = if_insertion;
            scores_new_deletions[i] = if_deletion;
        }

        // The last element of the last chunk is the result of the global alignment.
        if (tasks_offset == 0) last_cell_ = scores_new[0];
    }
};

/**
 *  @brief GPU adaptation of the `local_scorer` on CUDA, avoiding warp-level shuffles and DPX.
 *  @note Uses 32-bit `uint` counter to iterate through the string slices, so it can't be over 4 billion characters.
 */
template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_, sz_capability_t capability_>
#if _SZ_IS_CPP20
    requires pointer_like<first_iterator_type_> && pointer_like<second_iterator_type_> && score_like<score_type_> &&
             substituter_like<substituter_type_>
#endif
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, affine_gap_costs_t,
                   objective_, sz_similarity_local_k, capability_, std::enable_if_t<capability_ & sz_cap_cuda_k>> {

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
    using char_t = typename std::remove_cvref<first_char_t>::type;

    using warp_scorer_t = tile_scorer<first_iterator_t, second_iterator_t, score_t, substituter_t, affine_gap_costs_t,
                                      objective_k, sz_similarity_local_k, capability_k>;

  protected:
    substituter_t substituter_ {};
    affine_gap_costs_t gap_costs_ {};
    score_t best_score_ {0};

    __forceinline__ __device__ score_t pick_best(score_t a, score_t b) const noexcept {
        if constexpr (objective_k == sz_minimize_distance_k) { return std::min(a, b); }
        else { return std::max(a, b); }
    }

    __forceinline__ __device__ score_t pick_best_in_warp(score_t val) const noexcept {
        // The `__shfl_down_sync` replaces `__shfl_down`
        // https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
        val = pick_best(__shfl_down_sync(0xffffffff, val, 16));
        val = pick_best(__shfl_down_sync(0xffffffff, val, 8));
        val = pick_best(__shfl_down_sync(0xffffffff, val, 4));
        val = pick_best(__shfl_down_sync(0xffffffff, val, 2));
        val = pick_best(__shfl_down_sync(0xffffffff, val, 1));
        return val;
    }

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
    __forceinline__ __device__ score_t score() const noexcept { return best_score_; }

    /**
     *  @brief Computes one diagonal of the DP matrix, using the results of the previous 2x diagonals.
     *  @param first_slice The first string, unlike the CPU variant @b NOT reversed.
     *  @param second_slice The second string.
     *
     *  @param tasks_offset The offset of the first character to compare from each string.
     *  @param tasks_step The step size for the next character to compare from each string.
     *  @param tasks_count The total number of characters to compare from input slices.
     *
     *  @tparam index_type_ @b `uint` is recommended if the strings are under 4 billion characters.
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
        _sz_assert(scores_pre_insertion + 1 == scores_pre_deletion);

        // ? One weird observation, is that even though we can avoid fetching `pre_insertion`
        // ? from shared memory on each cycle, by slicing the work differently between the threads,
        // ? and allowing them to reuse the previous `pre_deletion` as the new `pre_insertion`,
        // ? that code ends up being slower than the one below.
        for (index_type_ i = tasks_offset; i < tasks_count; i += tasks_step) {
            score_t pre_substitution = scores_pre_substitution[i];
            score_t pre_insertion_opening = scores_pre_insertion[i];
            score_t pre_deletion_opening = scores_pre_deletion[i];
            score_t pre_insertion_expansion = scores_running_insertions[i];
            score_t pre_deletion_expansion = scores_running_deletions[i];

            error_cost_t cost_of_substitution = substituter_(first_slice[tasks_count - i - 1], second_slice[i]);
            score_t if_substitution = pre_substitution + cost_of_substitution;
            score_t if_deletion = min_or_max<objective_k>(pre_deletion_opening + gap_costs_.open,
                                                          pre_deletion_expansion + gap_costs_.extend);
            score_t if_insertion = min_or_max<objective_k>(pre_insertion_opening + gap_costs_.open,
                                                           pre_insertion_expansion + gap_costs_.extend);
            score_t if_deletion_or_insertion = min_or_max<objective_k>(if_deletion, if_insertion);
            score_t if_substitution_or_reset = pick_best(if_substitution, 0);
            score_t cell_score = pick_best(if_deletion_or_insertion, if_substitution_or_reset);

            // Export results.
            scores_new[i] = cell_score;
            scores_new_insertions[i] = if_insertion;
            scores_new_deletions[i] = if_deletion;

            // Update the global maximum score if this cell beats it.
            best_score_ = pick_best(best_score_, cell_score);
        }

        // ! Don't forget to pick the best among the best scores per thread.
        best_score_ = pick_best_in_warp(best_score_);
    }
};

#if SZ_USE_KEPLER

/**
 *  @brief GPU adaptation of the `tile_scorer` - Minimizes Global Levenshtein distance.
 *  @note Requires Kepler generation GPUs to handle 4x `u8` scores at a time.
 *
 *  Relies on following instruction families to output 4x @b `u8` scores per call:
 *  - @b `prmt` to shuffle bytes in 32 bit registers.
 *  - @b `vmax4,vmin4,vadd4` video-processing instructions.
 */
template <>
struct tile_scorer<char const *, char const *, sz_u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, sz_u8_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {

    using warp_scorer_t::tile_scorer; // Make the constructors visible

    __forceinline__ __device__ void operator()(                                 //
        char const *first_slice, char const *second_slice,                      //
        uint const tasks_offset, uint const tasks_step, uint const tasks_count, // ! Unlike CPU, uses `uint`
        sz_u8_t const *scores_pre_substitution, sz_u8_t const *scores_pre_insertion, sz_u8_t const *scores_pre_deletion,
        sz_u8_t *scores_new) noexcept {

        sz_u8_t const match_cost = this->substituter_.match;
        sz_u8_t const mismatch_cost = this->substituter_.mismatch;
        sz_u8_t const gap_cost = this->gap_costs_.open_or_extend;
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
        for (uint i = tasks_offset * 4; i < tasks_count; i += tasks_step * 4) { // ! it's OK to spill beyond bounds
            pre_substitution_vec = sz_u32_load_unaligned(scores_pre_substitution + i);
            pre_insertion_vec = sz_u32_load_unaligned(scores_pre_insertion + i);
            pre_deletion_vec = sz_u32_load_unaligned(scores_pre_deletion + i);
            first_vec = sz_u32_load_unaligned(first_slice + tasks_count - i - 4);
            second_vec = sz_u32_load_unaligned(second_slice + i);
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
        if (tasks_offset == 0) this->last_cell_ = scores_new[0];
    }
};

template <>
struct tile_scorer<char const *, char const *, sz_u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, sz_u16_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {
    using warp_scorer_t::tile_scorer; // Make the constructors visible

    __forceinline__ __device__ void operator()(                                 //
        char const *first_slice, char const *second_slice,                      //
        uint const tasks_offset, uint const tasks_step, uint const tasks_count, // ! Unlike CPU, uses `uint`
        sz_u16_t const *scores_pre_substitution, sz_u16_t const *scores_pre_insertion,
        sz_u16_t const *scores_pre_deletion, sz_u16_t *scores_new) noexcept {

        sz_u16_t const match_cost = this->substituter_.match;
        sz_u16_t const mismatch_cost = this->substituter_.mismatch;
        sz_u16_t const gap_cost = this->gap_costs_.open_or_extend;
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
        for (uint i = tasks_offset * 2; i < tasks_count; i += tasks_step * 2) { // ! it's OK to spill beyond bounds
            pre_substitution_vec = sz_u32_load_unaligned(scores_pre_substitution + i);
            pre_insertion_vec = sz_u32_load_unaligned(scores_pre_insertion + i);
            pre_deletion_vec = sz_u32_load_unaligned(scores_pre_deletion + i);
            first_vec.u16s[0] = *(first_slice + tasks_count - i - 1); // ! with a [] lookup would underflow
            first_vec.u16s[1] = *(first_slice + tasks_count - i - 2); // ! with a [] lookup would underflow
            second_vec.u16s[0] = second_slice[i + 0];
            second_vec.u16s[1] = second_slice[i + 1];

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
        if (tasks_offset == 0) this->last_cell_ = scores_new[0];
    }
};

template <>
struct tile_scorer<char const *, char const *, sz_u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                   sz_minimize_distance_k, sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, sz_u32_t, uniform_substitution_costs_t, linear_gap_costs_t,
                         sz_minimize_distance_k, sz_similarity_global_k, sz_cap_cuda_k> {
    using warp_scorer_t::tile_scorer; // Make the constructors visible
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
    typename index_type_ = uint,                                 //
    typename score_type_ = size_t,                               //
    typename final_score_type_ = size_t,                         //
    typename substituter_type_ = uniform_substitution_costs_t,   //
    sz_similarity_objective_t objective_ = sz_maximize_score_k,  //
    sz_similarity_locality_t locality_ = sz_similarity_global_k, //
    sz_capability_t capability_ = sz_cap_cuda_k                  //
    >
__global__ void _linear_score_across_cuda_device(              //
    char_type_ const *shorter_ptr, index_type_ shorter_length, //
    char_type_ const *longer_ptr, index_type_ longer_length,   //
    final_score_type_ *result_ptr, score_type_ *diagonals_ptr, //
    substituter_type_ const substituter, linear_gap_costs_t const gap_costs) {

    namespace cg = cooperative_groups;

    _sz_assert(shorter_length > 0);
    _sz_assert(longer_length > 0);
    _sz_assert(shorter_length <= longer_length);
    using char_t = char_type_;
    using index_t = index_type_;
    using score_t = score_type_;
    using final_score_t = final_score_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;

    // Pre-load the substituter and gap costs.
    using substituter_t = substituter_type_;
    using gap_costs_t = linear_gap_costs_t;
    static_assert(std::is_trivially_copyable<substituter_t>::value, "Substituter must be trivially copyable.");
    static_assert(std::is_trivially_copyable<gap_costs_t>::value, "Gap costs must be trivially copyable.");

    using warp_scorer_t = tile_scorer<char_t const *, char_t const *, score_t, substituter_t, gap_costs_t, objective_k,
                                      locality_k, capability_k>;

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
    warp_scorer_t diagonal_aligner {substituter, gap_costs};
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
    typename index_type_ = uint,                                 //
    typename score_type_ = size_t,                               //
    typename final_score_type_ = size_t,                         //
    typename substituter_type_ = uniform_substitution_costs_t,   //
    sz_similarity_objective_t objective_ = sz_maximize_score_k,  //
    sz_similarity_locality_t locality_ = sz_similarity_global_k, //
    sz_capability_t capability_ = sz_cap_cuda_k                  //
    >
__global__ void _affine_score_across_cuda_device(              //
    char_type_ const *shorter_ptr, index_type_ shorter_length, //
    char_type_ const *longer_ptr, index_type_ longer_length,   //
    final_score_type_ *result_ptr, score_type_ *diagonals_ptr, //
    substituter_type_ const substituter, affine_gap_costs_t const gap_costs) {

    namespace cg = cooperative_groups;

    _sz_assert(shorter_length > 0);
    _sz_assert(longer_length > 0);
    _sz_assert(shorter_length <= longer_length);
    using char_t = char_type_;
    using index_t = index_type_;
    using score_t = score_type_;
    using final_score_t = final_score_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;

    // Pre-load the substituter and gap costs.
    using substituter_t = substituter_type_;
    using gap_costs_t = affine_gap_costs_t;
    static_assert(std::is_trivially_copyable<substituter_t>::value, "Substituter must be trivially copyable.");
    static_assert(std::is_trivially_copyable<gap_costs_t>::value, "Gap costs must be trivially copyable.");

    using warp_scorer_t = tile_scorer<char_t const *, char_t const *, score_t, substituter_t, gap_costs_t, objective_k,
                                      locality_k, capability_k>;

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
    warp_scorer_t diagonal_aligner {substituter, gap_costs};
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
        std::swap(current_inserts, next_inserts);
        std::swap(current_deletes, next_deletes);
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

        std::swap(current_inserts, next_inserts);
        std::swap(current_deletes, next_deletes);

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
        std::swap(current_inserts, next_inserts);
        std::swap(current_deletes, next_deletes);

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
    typename index_type_ = uint,                                 //
    typename score_type_ = size_t,                               //
    typename substituter_type_ = uniform_substitution_costs_t,   //
    sz_similarity_objective_t objective_ = sz_maximize_score_k,  //
    sz_similarity_locality_t locality_ = sz_similarity_global_k, //
    sz_capability_t capability_ = sz_cap_cuda_k                  //
    >
__global__ void _linear_score_on_each_cuda_warp( //
    task_type_ *tasks, size_t tasks_count,       //
    substituter_type_ const substituter, linear_gap_costs_t const gap_costs) {

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

    using warp_scorer_t = tile_scorer<char_t const *, char_t const *, score_t, substituter_t, gap_costs_t, objective_k,
                                      locality_k, capability_k>;

    // Allocating shared memory is handled on the host side.
    extern __shared__ char shared_memory_buffer[];

    // We are computing N edit distances for N pairs of strings. Not a cartesian product!
    // Each block/warp may end up receiving a different number of strings.
    for (size_t task_idx = blockIdx.x; task_idx < tasks_count; task_idx += gridDim.x) {
        task_t &task = tasks[task_idx];
        char_t const *shorter_global = task.shorter_ptr;
        char_t const *longer_global = task.longer_ptr;
        size_t const shorter_length = task.shorter_length;
        size_t const longer_length = task.longer_length;
        auto &result_ref = task.result;

        // We are going to store 3 diagonals of the matrix, assuming each would fit into a single ZMM register.
        // The length of the longest (main) diagonal would be `shorter_dim = (shorter_length + 1)`.
        uint const shorter_dim = static_cast<uint>(shorter_length + 1);
        uint const longer_dim = static_cast<uint>(longer_length + 1);

        // Let's say we are dealing with 3 and 5 letter words.
        // The matrix will have size 4 x 6, parameterized as (shorter_dim x longer_dim).
        // It will have:
        // - 4 diagonals of increasing length, at positions: 0, 1, 2, 3.
        // - 2 diagonals of fixed length, at positions: 4, 5.
        // - 3 diagonals of decreasing length, at positions: 6, 7, 8.
        uint const diagonals_count = shorter_dim + longer_dim - 1;
        uint const max_diagonal_length = shorter_length + 1;
        uint const bytes_per_diagonal = round_up_to_multiple<uint>(max_diagonal_length * sizeof(score_t), 4);

        // The next few pointers will be swapped around.
        score_t *previous_scores = reinterpret_cast<score_t *>(shared_memory_buffer);
        score_t *current_scores = reinterpret_cast<score_t *>(shared_memory_buffer + bytes_per_diagonal);
        score_t *next_scores = reinterpret_cast<score_t *>(shared_memory_buffer + 2 * bytes_per_diagonal);
        char_t *const longer = reinterpret_cast<char_t *>(shared_memory_buffer + 3 * bytes_per_diagonal);
        char_t *const shorter = longer + longer_length;

        // Each thread in the warp will be loading it's own set of strided characters into shared memory.
        for (uint i = threadIdx.x; i < longer_length; i += blockDim.x) longer[i] = longer_global[i];
        for (uint i = threadIdx.x; i < shorter_length; i += blockDim.x) shorter[i] = shorter_global[i];

        // Initialize the first two diagonals:
        warp_scorer_t diagonal_aligner {substituter, gap_costs};
        if (threadIdx.x == 0) {
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
        uint next_diagonal_index = 2;

        // Progress through the upper-left triangle of the Levenshtein matrix.
        for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {

            uint const next_diagonal_length = next_diagonal_index + 1;
            diagonal_aligner(                       //
                shorter,                            // first sequence of characters
                longer,                             // second sequence of characters
                threadIdx.x, blockDim.x,            //
                next_diagonal_length - 2,           // number of elements to compute with the `diagonal_aligner`
                previous_scores,                    // costs pre substitution
                current_scores, current_scores + 1, // costs pre insertion/deletion
                next_scores + 1);                   // ! notice unaligned write destination

            // Don't forget to populate the first row and the first column of the Levenshtein matrix.
            if (threadIdx.x == 0) {
                diagonal_aligner.init_score(next_scores[0], next_diagonal_index);
                diagonal_aligner.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);
            }
            __syncwarp();

            // Perform a circular rotation of those buffers, to reuse the memory.
            rotate_three(previous_scores, current_scores, next_scores);
        }

        // Now let's handle the anti-diagonal band of the matrix, between the top and bottom-right triangles.
        for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {

            uint const next_diagonal_length = shorter_dim;
            diagonal_aligner(                               //
                shorter,                                    // first sequence of characters
                longer + next_diagonal_index - shorter_dim, // second sequence of characters
                threadIdx.x, blockDim.x,                    //
                next_diagonal_length - 1,                   // number of elements to compute with the `diagonal_aligner`
                previous_scores,                            // costs pre substitution
                current_scores, current_scores + 1,         // costs pre insertion/deletion
                next_scores);

            // Don't forget to populate the first row of the Levenshtein matrix.
            if (threadIdx.x == 0)
                diagonal_aligner.init_score(next_scores[next_diagonal_length - 1], next_diagonal_index);

            __syncwarp();
            // ! In the central anti-diagonal band, we can't just set the `current_scores + 1` to `previous_scores`
            // ! for the circular shift, as we will end up spilling outside of the diagonal a few iterations later.
            // ! Assuming in-place `memmove` is tricky on the GPU, so we will copy the data.
            for (size_t i = threadIdx.x; i + 1 < next_diagonal_length; i += blockDim.x)
                previous_scores[i] = current_scores[i + 1];
            __syncwarp();
            for (size_t i = threadIdx.x; i < next_diagonal_length; i += blockDim.x) current_scores[i] = next_scores[i];
            __syncwarp();
        }

        // Now let's handle the bottom-right triangle of the matrix.
        for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {

            uint const next_diagonal_length = diagonals_count - next_diagonal_index;
            diagonal_aligner(                               //
                shorter + next_diagonal_index - longer_dim, // first sequence of characters
                longer + next_diagonal_index - shorter_dim, // second sequence of characters
                threadIdx.x, blockDim.x,                    //
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
        if (threadIdx.x == 0) result_ref = diagonal_aligner.score();
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
    typename index_type_ = uint,                                 //
    typename score_type_ = size_t,                               //
    typename substituter_type_ = uniform_substitution_costs_t,   //
    sz_similarity_objective_t objective_ = sz_maximize_score_k,  //
    sz_similarity_locality_t locality_ = sz_similarity_global_k, //
    sz_capability_t capability_ = sz_cap_cuda_k                  //
    >
__global__ void _affine_score_on_each_cuda_warp( //
    task_type_ *tasks, size_t tasks_count,       //
    substituter_type_ const substituter, affine_gap_costs_t const gap_costs) {

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

    using warp_scorer_t = tile_scorer<char_t const *, char_t const *, score_t, substituter_t, gap_costs_t, objective_k,
                                      locality_k, capability_k>;

    // Only one thread will be initializing the top row and left column and outputting the result.
    bool const is_main_thread = threadIdx.x == 0; // ! Differs for device-wide

    // Allocating shared memory is handled on the host side.
    extern __shared__ char shared_memory_buffer[];

    // We are computing N edit distances for N pairs of strings. Not a cartesian product!
    // Each block/warp may end up receiving a different number of strings.
    for (size_t task_idx = blockIdx.x; task_idx < tasks_count; task_idx += gridDim.x) {
        task_t &task = tasks[task_idx];
        char_t const *shorter_global = task.shorter_ptr;
        char_t const *longer_global = task.longer_ptr;
        size_t const shorter_length = task.shorter_length;
        size_t const longer_length = task.longer_length;
        auto &result_ref = task.result;

        // We are going to store 3 diagonals of the matrix, assuming each would fit into a single ZMM register.
        // The length of the longest (main) diagonal would be `shorter_dim = (shorter_length + 1)`.
        uint const shorter_dim = static_cast<uint>(shorter_length + 1);
        uint const longer_dim = static_cast<uint>(longer_length + 1);

        // Let's say we are dealing with 3 and 5 letter words.
        // The matrix will have size 4 x 6, parameterized as (shorter_dim x longer_dim).
        // It will have:
        // - 4 diagonals of increasing length, at positions: 0, 1, 2, 3.
        // - 2 diagonals of fixed length, at positions: 4, 5.
        // - 3 diagonals of decreasing length, at positions: 6, 7, 8.
        uint const diagonals_count = shorter_dim + longer_dim - 1;
        uint const max_diagonal_length = shorter_length + 1;
        uint const bytes_per_diagonal = round_up_to_multiple<uint>(max_diagonal_length * sizeof(score_t), 4);

        // The next few pointers will be swapped around.
        score_t *previous_scores = reinterpret_cast<score_t *>(shared_memory_buffer);
        score_t *current_scores = reinterpret_cast<score_t *>(shared_memory_buffer + bytes_per_diagonal);
        score_t *next_scores = reinterpret_cast<score_t *>(shared_memory_buffer + 2 * bytes_per_diagonal);
        score_t *current_inserts = reinterpret_cast<score_t *>(shared_memory_buffer + 3 * bytes_per_diagonal);
        score_t *next_inserts = reinterpret_cast<score_t *>(shared_memory_buffer + 4 * bytes_per_diagonal);
        score_t *current_deletes = reinterpret_cast<score_t *>(shared_memory_buffer + 5 * bytes_per_diagonal);
        score_t *next_deletes = reinterpret_cast<score_t *>(shared_memory_buffer + 6 * bytes_per_diagonal);
        char_t *const longer = reinterpret_cast<char_t *>(shared_memory_buffer + 7 * bytes_per_diagonal);
        char_t *const shorter = longer + longer_length;

        // Each thread in the warp will be loading it's own set of strided characters into shared memory.
        for (uint i = threadIdx.x; i < longer_length; i += blockDim.x) longer[i] = longer_global[i];
        for (uint i = threadIdx.x; i < shorter_length; i += blockDim.x) shorter[i] = shorter_global[i];

        // Initialize the first two diagonals:
        warp_scorer_t diagonal_aligner {substituter, gap_costs};
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
        uint next_diagonal_index = 2;

        // Progress through the upper-left triangle of the Levenshtein matrix.
        for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {

            uint const next_diagonal_length = next_diagonal_index + 1;
            diagonal_aligner(                         //
                shorter,                              // first sequence of characters
                longer,                               // second sequence of characters
                threadIdx.x, blockDim.x,              //
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
            std::swap(current_inserts, next_inserts);
            std::swap(current_deletes, next_deletes);
        }

        // Now let's handle the anti-diagonal band of the matrix, between the top and bottom-right triangles.
        for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {

            uint const next_diagonal_length = shorter_dim;
            diagonal_aligner(                               //
                shorter,                                    // first sequence of characters
                longer + next_diagonal_index - shorter_dim, // second sequence of characters
                threadIdx.x, blockDim.x,                    //
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

            std::swap(current_inserts, next_inserts);
            std::swap(current_deletes, next_deletes);

            __syncwarp();
            // ! In the central anti-diagonal band, we can't just set the `current_scores + 1` to `previous_scores`
            // ! for the circular shift, as we will end up spilling outside of the diagonal a few iterations later.
            // ! Assuming in-place `memmove` is tricky on the GPU, so we will copy the data.
            for (size_t i = threadIdx.x; i + 1 < next_diagonal_length; i += blockDim.x)
                previous_scores[i] = current_scores[i + 1];
            __syncwarp();
            for (size_t i = threadIdx.x; i < next_diagonal_length; i += blockDim.x) current_scores[i] = next_scores[i];
            __syncwarp();
        }

        // Now let's handle the bottom-right triangle of the matrix.
        for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {

            uint const next_diagonal_length = diagonals_count - next_diagonal_index;
            diagonal_aligner(                               //
                shorter + next_diagonal_index - longer_dim, // first sequence of characters
                longer + next_diagonal_index - shorter_dim, // second sequence of characters
                threadIdx.x, blockDim.x,                    //
                next_diagonal_length,                       // number of elements to compute with the `diagonal_aligner`
                previous_scores,                            // costs pre substitution
                current_scores, current_scores + 1,         // costs pre insertion/deletion opening
                current_inserts, current_deletes + 1,       // costs pre insertion/deletion extension
                next_scores,                                // updated similarity scores
                next_inserts, next_deletes                  // updated insertion/deletion extensions
            );

            // Perform a circular rotation of those buffers, to reuse the memory.
            rotate_three(previous_scores, current_scores, next_scores);
            std::swap(current_inserts, next_inserts);
            std::swap(current_deletes, next_deletes);

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
    using scores_allocator_t = typename allocator_t::template rebind<size_t>::other;
    static constexpr sz_capability_t capability_k = capability_;

    struct task_t {
        char_t const *shorter_ptr = nullptr;
        size_t shorter_length = 0;
        char_t const *longer_ptr = nullptr;
        size_t longer_length = 0;
        size_t memory_requirement = 0;
        size_t original_index = 0;
        size_t result = std::numeric_limits<size_t>::max();       // ? Signal that we are not done yet.
        bytes_per_cell_t bytes_per_cell = eight_bytes_per_cell_k; // ? Worst case, need the most memory per scalar.
        warp_tasks_density_t density = warps_working_together_k;  // ? Worst case, we are not using shared memory.

        constexpr task_t() = default;
        constexpr task_t(                                 //
            char_t const *first_ptr, size_t first_length, //
            char_t const *second_ptr, size_t second_length) noexcept {
            if (first_length < second_length)
                shorter_ptr = first_ptr, shorter_length = first_length, longer_ptr = second_ptr,
                longer_length = second_length;
            else
                shorter_ptr = second_ptr, shorter_length = second_length, longer_ptr = first_ptr,
                longer_length = first_length;
        }

        constexpr size_t max_diagonal_length() const noexcept {
            return sz_max_of_two(shorter_length, longer_length) + 1;
        }
    };
    using tasks_allocator_t = typename allocator_t::template rebind<task_t>::other;

    uniform_substitution_costs_t substituter_ {};
    gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    levenshtein_distances(uniform_substitution_costs_t subs = {}, gap_costs_t gaps = {},
                          allocator_t const &alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    cuda_status_t operator()(                                                                 //
        first_strings_type_ const &first_strings, second_strings_type_ const &second_strings, //
        results_type_ *results_ptr,                                                           //
        gpu_specs_t specs = {}, cuda_executor_t executor = {}) const noexcept {

        constexpr bool is_affine_k = std::is_same<gap_costs_t, affine_gap_costs_t>::value;
        constexpr size_t count_diagonals_k = is_affine_k ? 7 : 3;

        // Preallocate the events for GPU timing.
        cudaEvent_t start_event, stop_event;
        cudaEventCreate(&start_event, cudaEventBlockingSync);
        cudaEventCreate(&stop_event, cudaEventBlockingSync);

        using final_score_t = results_type_;
        safe_vector<task_t, tasks_allocator_t> tasks(alloc_);
        if (tasks.try_resize(first_strings.size()) == status_t::bad_alloc_k) return {status_t::bad_alloc_k};

        // Record the start event
        cudaError_t start_event_error = cudaEventRecord(start_event, executor.stream);
        if (start_event_error != cudaSuccess) return {status_t::unknown_k, start_event_error};

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
            task.memory_requirement = requirement.total;
            task.bytes_per_cell = requirement.bytes_per_cell;
            task.density = warp_tasks_density(requirement.total, specs);
            if (task.density == infinite_warps_per_multiprocessor_k) {
                if constexpr (!is_affine_k) task.result = task.longer_length * gap_costs_.open_or_extend;
                else
                    task.result =
                        task.longer_length ? (task.longer_length - 1) * gap_costs_.extend + gap_costs_.open : 0;
                count_empty_tasks++;
            }
            tasks[i] = task;
        }

        auto [device_level_tasks, warp_level_tasks, empty_tasks] =
            warp_tasks_grouping<task_t>({tasks.data(), tasks.size()}, specs);

        if (device_level_tasks.size()) {
            auto device_level_u16_kernel =
                is_affine_k //
                    ? (void *)&_affine_score_across_cuda_device<char_t, sz_u16_t, sz_u16_t, final_score_t,
                                                                uniform_substitution_costs_t, sz_minimize_distance_k,
                                                                sz_similarity_global_k, capability_k>
                    : (void *)&_linear_score_across_cuda_device<char_t, sz_u16_t, sz_u16_t, final_score_t,
                                                                uniform_substitution_costs_t, sz_minimize_distance_k,
                                                                sz_similarity_global_k, capability_k>;
            auto device_level_u32_kernel =
                is_affine_k //
                    ? (void *)&_affine_score_across_cuda_device<char_t, sz_u32_t, sz_u32_t, final_score_t,
                                                                uniform_substitution_costs_t, sz_minimize_distance_k,
                                                                sz_similarity_global_k, capability_k>
                    : (void *)&_linear_score_across_cuda_device<char_t, sz_u32_t, sz_u32_t, final_score_t,
                                                                uniform_substitution_costs_t, sz_minimize_distance_k,
                                                                sz_similarity_global_k, capability_k>;
            auto device_level_u64_kernel =
                is_affine_k //
                    ? (void *)&_affine_score_across_cuda_device<char_t, sz_u64_t, sz_u64_t, final_score_t,
                                                                uniform_substitution_costs_t, sz_minimize_distance_k,
                                                                sz_similarity_global_k, capability_k>
                    : (void *)&_linear_score_across_cuda_device<char_t, sz_u64_t, sz_u64_t, final_score_t,
                                                                uniform_substitution_costs_t, sz_minimize_distance_k,
                                                                sz_similarity_global_k, capability_k>;
            void *device_level_kernel_args[8];

            // On very large inputs we can't fit the diagonals in shared memory, and use the global one.
            safe_vector<sz_u64_t, scores_allocator_t> diagonals_u64_buffer(alloc_);
            task_t const &largest_task = device_level_tasks[0];
            _sz_assert(largest_task.max_diagonal_length() >= device_level_tasks.back().max_diagonal_length());
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
                if (task.bytes_per_cell >= sizeof(sz_u32_t))
                    device_level_kernel = reinterpret_cast<void *>(device_level_u32_kernel);
                if (task.bytes_per_cell >= sizeof(sz_u64_t))
                    device_level_kernel = reinterpret_cast<void *>(device_level_u64_kernel);

                // TODO: We can be wiser about the dimensions of this grid.
                uint const random_block_size = 128;
                uint const random_blocks_per_multiprocessor = 32;
                cudaError_t launch_error = cudaLaunchCooperativeKernel(                       //
                    reinterpret_cast<void *>(device_level_kernel),                            // Kernel function pointer
                    dim3(random_blocks_per_multiprocessor * specs.streaming_multiprocessors), // Grid dimensions
                    dim3(random_block_size),                                                  // Block dimensions
                    device_level_kernel_args, // Array of kernel argument pointers
                    0,                        // Shared memory per block (in bytes)
                    executor.stream);         // CUDA stream
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
                    ? (void *)&_affine_score_on_each_cuda_warp<task_t, char_t, sz_u8_t, sz_u8_t,
                                                               uniform_substitution_costs_t, sz_minimize_distance_k,
                                                               sz_similarity_global_k, capability_k>
                    : (void *)&_linear_score_on_each_cuda_warp<task_t, char_t, sz_u8_t, sz_u8_t,
                                                               uniform_substitution_costs_t, sz_minimize_distance_k,
                                                               sz_similarity_global_k, capability_k>;
            auto warp_level_u16_kernel =
                is_affine_k
                    ? (void *)&_affine_score_on_each_cuda_warp<task_t, char_t, sz_u16_t, sz_u16_t,
                                                               uniform_substitution_costs_t, sz_minimize_distance_k,
                                                               sz_similarity_global_k, capability_k>
                    : (void *)&_linear_score_on_each_cuda_warp<task_t, char_t, sz_u16_t, sz_u16_t,
                                                               uniform_substitution_costs_t, sz_minimize_distance_k,
                                                               sz_similarity_global_k, capability_k>;
            void *warp_level_kernel_args[4];

            cuda_status_t result;
            auto const task_size_equality = [](task_t const &lhs, task_t const &rhs) {
                return lhs.bytes_per_cell == rhs.bytes_per_cell && lhs.density == rhs.density;
            };
            auto const task_group_callback = [&](task_t const *tasks_begin, task_t const *tasks_end) {
                // Check if we need to stop processing.
                if (result.status != status_t::success_k) return;

                size_t const count_tasks = tasks_end - tasks_begin;
                warp_level_kernel_args[0] = (void *)(&tasks_begin);
                warp_level_kernel_args[1] = (void *)(&count_tasks);
                warp_level_kernel_args[2] = (void *)(&substituter_);
                warp_level_kernel_args[3] = (void *)(&gap_costs_);

                // Pick the smallest fitting type for the diagonals.
                task_t const &indicative_task = tasks_begin[0];
                void *warp_level_kernel = reinterpret_cast<void *>(warp_level_u8_kernel);
                if (indicative_task.bytes_per_cell >= sizeof(sz_u16_t))
                    warp_level_kernel = reinterpret_cast<void *>(warp_level_u16_kernel);

                // Update the selected kernels properties.
                size_t const shared_memory_per_block =
                    indicative_task.memory_requirement * static_cast<size_t>(indicative_task.density);
                _sz_assert(shared_memory_per_block > 0);
                _sz_assert(shared_memory_per_block < specs.shared_memory_per_multiprocessor());
                cudaError_t attribute_error = cudaFuncSetAttribute(
                    warp_level_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_per_block);
                if (attribute_error != cudaSuccess) {
                    result = {status_t::unknown_k, attribute_error};
                    return;
                }

                // Warp-level algorithm clearly aligns with the warp size.
                uint const warp_block_size = static_cast<uint>(specs.warp_size);
                uint const warp_blocks_per_multiprocessor = static_cast<uint>(indicative_task.density);
                cudaError_t launch_error = cudaLaunchKernel(                                //
                    reinterpret_cast<void *>(warp_level_kernel),                            // Kernel function pointer
                    dim3(warp_blocks_per_multiprocessor * specs.streaming_multiprocessors), // Grid dimensions
                    dim3(warp_block_size),                                                  // Block dimensions
                    warp_level_kernel_args,  // Array of kernel argument pointers
                    shared_memory_per_block, // Shared memory per block (in bytes)
                    executor.stream);        // CUDA stream
                if (launch_error != cudaSuccess) {
                    result = {launch_error == cudaErrorMemoryAllocation ? status_t::bad_alloc_k : status_t::unknown_k,
                              launch_error};
                    return;
                }

                // Wait until everything completes, as on the next iteration we will update the properties again.
                cudaError_t execution_error = cudaStreamSynchronize(executor.stream);
                if (execution_error != cudaSuccess) {
                    result = {status_t::unknown_k, execution_error};
                    return;
                }
            };
            group_by(warp_level_tasks.begin(), warp_level_tasks.end(), task_size_equality, task_group_callback);
            if (result.status != status_t::success_k) return result;
        }

        // Calculate the duration:
        cudaError_t stop_event_error = cudaEventRecord(stop_event, executor.stream);
        if (stop_event_error != cudaSuccess) return {status_t::unknown_k, stop_event_error};
        float execution_milliseconds = 0;
        cudaEventElapsedTime(&execution_milliseconds, start_event, stop_event);

        // Now that everything went well, export the results back into the `results` array.
        for (size_t i = 0; i < tasks.size(); ++i) {
            task_t const &task = tasks[i];
            results_ptr[task.original_index] = task.result;
        }
        return {status_t::success_k, cudaSuccess, execution_milliseconds};
    }
};

#pragma endregion

#pragma region - Needleman Wunsch Scores in CUDA
#if 0
/**
 *  @brief  Convenience buffer of the size matching the size of the CUDA constant memory,
 *          used to cheaper store and access the substitution costs for the characters.
 *  @see    CUDA constant memory docs: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#constant
 */
__constant__ char _error_costs_in_cuda_constant_memory[256 * 256];

/**
 *  @brief  Needleman-Wunsch alignment cores algorithm evaluating the Dynamic Programming matrix
 *          @b three skewed (reverse) diagonals at a time on a GPU, leveraging CUDA for parallelization.
 *          Each pair of strings gets its own @b "block" of CUDA threads forming one @b warp and shared memory.
 *
 *  @param[in] first_strings Array of first strings in each pair for score calculation.
 *  @param[in] second_strings Array of second strings in each pair for score calculation.
 *  @param[out] results_ptr Output array of scores for each pair of strings.
 *
 *  @tparam substituter_type_ Must be a trivially copyable object already placed in the GPU @b constant memory.
 */
template <                                                      //
    typename first_strings_type_,                               //
    typename second_strings_type_,                              //
    typename score_type_ = sz_ssize_t,                        //
    typename substituter_type_ = error_costs_256x256_t, //
    typename gap_costs_type_ = linear_gap_costs_t,        //
    sz_capability_t capability_ = sz_cap_cuda_k                 //
    >
__global__ void _needleman_wunsch_in_cuda_warp( //
    first_strings_type_ first_strings,          //
    second_strings_type_ second_strings,        //
    score_type_ *results_ptr,                   //
    gap_costs_type_ gaps) {

    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    using first_string_t = typename first_strings_type_::value_type;
    using second_string_t = typename second_strings_type_::value_type;
    using first_char_t = typename first_string_t::value_type;
    using second_char_t = typename second_string_t::value_type;
    static_assert(sizeof(first_char_t) == sizeof(second_char_t), "Character types don't match");
    using char_t = typename std::remove_cvref<first_char_t>::type;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    static_assert(std::is_trivially_copyable<substituter_t>::value, "Substituter must be a trivially copyable type.");
    static_assert(std::is_signed<score_t>::value, "Score must be a signed type.");

    static constexpr sz_capability_t cap_k = capability_;
    static constexpr sz_similarity_objective_t obj_k = sz_maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    using walker_i8_t = diagonal_walker_per_warp<char_t, sz_i8_t, substituter_t const &, obj_k, locality_k, cap_k>;
    using walker_i16_t = diagonal_walker_per_warp<char_t, sz_i16_t, substituter_t const &, obj_k, locality_k, cap_k>;
    using walker_i32_t = diagonal_walker_per_warp<char_t, sz_i32_t, substituter_t const &, obj_k, locality_k, cap_k>;
    using walker_i64_t = diagonal_walker_per_warp<char_t, sz_i64_t, substituter_t const &, obj_k, locality_k, cap_k>;

    // Allocating shared memory is handled on the host side.
    extern __shared__ char shared_memory_buffer[];

    // We expect the substituter state to be already in the GPU constant memory.
    substituter_t const &substituter_constant =
        *reinterpret_cast<substituter_t const *>(_error_costs_in_cuda_constant_memory);

    // We are computing N edit distances for N pairs of strings. Not a cartesian product!
    // Each block/warp may end up receiving a different number of strings.
    for (size_t pair_idx = blockIdx.x; pair_idx < first_strings.size(); pair_idx += gridDim.x) {
        first_string_t const first_global = first_strings[pair_idx];
        second_string_t const second_global = second_strings[pair_idx];
        score_t &result_ref = results_ptr[pair_idx];

        // Skip empty strings.
        size_t const first_length = first_global.length();
        size_t const second_length = second_global.length();
        if (first_length == 0) {
            result_ref = second_length * gap_cost;
            continue;
        }
        if (second_length == 0) {
            result_ref = first_length * gap_cost;
            continue;
        }

        // Estimate the maximum dimension of the DP matrix to pick the smallest fitting type.
        using similarity_memory_requirements_t = similarity_memory_requirements<uint, true>;
        similarity_memory_requirements_t requirements(first_length, second_length, static_cast<uint>(magnitude()),
                                                      sizeof(char_t), 4);

        // Estimate the maximum dimension of the DP matrix to pick the smallest fitting type.
        span<char const> const first = {first_global.data(), first_length};
        span<char const> const second = {second_global.data(), second_length};

        if (requirements.bytes_per_cell == 1) {
            sz_i8_t result_i8 = std::numeric_limits<sz_i8_t>::min();
            walker_i8_t walker(substituter_constant, gap_cost);
            walker(first, second, result_i8, shared_memory_buffer);
            if (threadIdx.x == 0) result_ref = result_i8;
        }
        else if (requirements.bytes_per_cell == 2) {
            sz_i16_t result_i16 = std::numeric_limits<sz_i16_t>::min();
            walker_i16_t walker(substituter_constant, gap_cost);
            walker(first, second, result_i16, shared_memory_buffer);
            if (threadIdx.x == 0) result_ref = result_i16;
        }
        else if (requirements.bytes_per_cell == 4) {
            sz_i32_t result_i32 = std::numeric_limits<sz_i32_t>::min();
            walker_i32_t walker(substituter_constant, gap_cost);
            walker(first, second, result_i32, shared_memory_buffer);
            if (threadIdx.x == 0) result_ref = result_i32;
        }
        else if (requirements.bytes_per_cell == 8) {
            sz_i64_t result_i64 = std::numeric_limits<sz_i64_t>::min();
            walker_i64_t walker(substituter_constant, gap_cost);
            walker(first, second, result_i64, shared_memory_buffer);
            if (threadIdx.x == 0) result_ref = result_i64;
        }
    }
}

/** @brief Dispatches on @b `_needleman_wunsch_in_cuda_warp` on the device side from the host side. */
template <                                                      //
    sz_capability_t capability_ = sz_cap_cuda_k,                //
    typename first_strings_type_,                               //
    typename second_strings_type_,                              //
    score_like score_type_ = sz_ssize_t,                        //
    substituter_like substituter_type_ = error_costs_256x256_t, //
    gap_costs_like gap_costs_type_ = linear_gap_costs_t         //
    >
cuda_status_t _needleman_wunsch_via_cuda_warp(                                            //
    first_strings_type_ const &first_strings, second_strings_type_ const &second_strings, //
    score_type_ *results,
    substituter_type_ const &substituter, //
    error_cost_t gap_cost = 1,            //
    gpu_specs_t specs = {}, cuda_executor_t executor = {}) noexcept(false) {

    // We need to be able to copy these function arguments into GPU memory:
    static constexpr sz_capability_t capability_k = capability_;
    using first_strings_t = first_strings_type_;
    using second_strings_t = second_strings_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    static_assert(std::is_trivially_copyable<first_strings_t>() && std::is_trivially_copyable<second_strings_t>(),
                  "The first and second strings must be trivially copyable types - consider `arrow_strings_view`.");

    // Make sure that we don't string pairs that are too large to fit 3 matrix diagonals into shared memory.
    // H100 Streaming Multiprocessor can have up to 128 active warps concurrently and only 256 KB of shared memory.
    // A100 SMs had only 192 KB. We can't deal with blocks that require more memory than the SM can provide.
    size_t shared_memory_per_block =
        _scores_diagonally_warp_shared_memory_requirement<true>(first_strings, second_strings, substituter.magnitude());
    if (shared_memory_per_block > specs.shared_memory_per_multiprocessor()) return {status_t::bad_alloc_k};

    // It may be the case that we've only received empty strings.
    if (shared_memory_per_block == 0) {
        for (size_t i = 0; i < first_strings.size(); ++i)
            if (first_strings[i].length() == 0) { results[i] = second_strings[i].length(); }
            else if (second_strings[i].length() == 0) { results[i] = first_strings[i].length(); }
        return {status_t::success_k};
    }

    // In most cases we should be able to fit many blocks per SM.
    size_t count_blocks_per_multiprocessor = specs.shared_memory_per_multiprocessor() / shared_memory_per_block;
    if (count_blocks_per_multiprocessor > specs.max_blocks_per_multiprocessor)
        count_blocks_per_multiprocessor = specs.max_blocks_per_multiprocessor;
    if (count_blocks_per_multiprocessor > first_strings.size()) count_blocks_per_multiprocessor = first_strings.size();

    // Let's use all 32 threads in a warp.
    constexpr size_t threads_per_block = 32u;
    size_t const magnitude() = substituter.magnitude();
    auto kernel =
        &_needleman_wunsch_in_cuda_warp<first_strings_t, second_strings_t, score_t, substituter_t, capability_k>;
    void *kernel_args[] = {
        (void *)&first_strings, (void *)&second_strings, (void *)&results, (void *)&gap_cost, (void *)&magnitude(),
    };

    // On Volta and newer GPUs, there is an extra flag to be set to use more than 48 KB of shared memory per block.
    // CUDA reserves 1 KB of shared memory per thread block, so on H100 we can use up to 227 KB of shared memory.
    // https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#unified-shared-memory-l1-texture-cache
    cudaError_t attribute_error =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             specs.shared_memory_per_multiprocessor() - count_blocks_per_multiprocessor * 1024);
    if (attribute_error != cudaSuccess) return {status_t::unknown_k, attribute_error};

    // Create CUDA events for timing
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // Record the start event
    cudaEventRecord(start_event, executor.stream);

    // Enqueue the transfer of the substituter to the constant memory:
    cudaError_t copy_error = cudaMemcpyToSymbolAsync(_error_costs_in_cuda_constant_memory, (void const *)&substituter,
                                                     sizeof(substituter_t), 0, cudaMemcpyHostToDevice, executor.stream);
    if (copy_error != cudaSuccess) return {status_t::unknown_k, copy_error};

    // Enqueue the kernel for execution:
    cudaError_t launch_error = cudaLaunchKernel(                                 //
        reinterpret_cast<void *>(kernel),                                        // Kernel function pointer
        dim3(count_blocks_per_multiprocessor * specs.streaming_multiprocessors), // Grid dimensions
        dim3(threads_per_block),                                                 // Block dimensions
        kernel_args,                                                             // Array of kernel argument pointers
        shared_memory_per_block,                                                 // Shared memory per block (in bytes)
        executor.stream);                                                        // CUDA stream
    if (launch_error != cudaSuccess)
        if (launch_error == cudaErrorMemoryAllocation) { return {status_t::bad_alloc_k, launch_error}; }
        else { return {status_t::unknown_k, launch_error}; }

    cudaEventRecord(stop_event, executor.stream);

    // Fetch the execution error:
    float execution_milliseconds = 0;
    cudaError_t execution_error = cudaStreamSynchronize(executor.stream);
    cudaEventElapsedTime(&execution_milliseconds, start_event, stop_event);

    if (execution_error != cudaSuccess)
        if (execution_error == cudaErrorMemoryAllocation) {
            return {status_t::bad_alloc_k, execution_error, execution_milliseconds};
        }
        else { return {status_t::unknown_k, execution_error, execution_milliseconds}; }
    return {status_t::success_k, cudaSuccess, execution_milliseconds};
}

/** @brief Dispatches baseline Needleman Wunsch algorithm to the GPU. */
template <typename char_type_, substituter_like substituter_type_, gap_costs_like gap_costs_type_,
          typename allocator_type_>
struct needleman_wunsch_scores<char_type_, substituter_type_, gap_costs_type_, allocator_type_, sz_cap_cuda_k> {

    using char_t = char_type_;
    using substituter_t = substituter_type_;
    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;

    substituter_t substituter_ {};
    error_cost_t gap_costs_ {1};
    allocator_t alloc_ {};

    needleman_wunsch_scores(allocator_t const &alloc = {}) noexcept : alloc_(alloc) {}
    needleman_wunsch_scores(substituter_t subs, gap_costs_t gaps, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    cuda_status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                             results_type_ &&results, gpu_specs_t specs = {},
                             cuda_executor_t executor = {}) const noexcept {
        return _needleman_wunsch_via_cuda_warp<sz_cap_cuda_k>(first_strings, second_strings, results, substituter_,
                                                              gap_costs_, specs, executor);
    }
};
#endif
#pragma endregion

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGCUZILLA_SIMILARITY_CUH_