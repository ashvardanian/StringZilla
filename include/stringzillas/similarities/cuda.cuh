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

#include <cub/device/device_partition.cuh>         // `cub::DevicePartition` — register/device task split
#include <cub/device/device_radix_sort.cuh>        // `cub::DeviceRadixSort` — single-pass device tier sort
#include <cub/device/device_reduce.cuh>            // `cub::DeviceReduce` — device-tier shape maxima
#include <cub/device/device_run_length_encode.cuh> // `cub::DeviceRunLengthEncode` — dense tier counts
#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/transform_input_iterator.cuh>

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
using levenshtein_cuda_t = levenshtein_distances<linear_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
using affine_levenshtein_cuda_t = levenshtein_distances<affine_gap_costs_t, ualloc_t, sz_cap_cuda_k>;

using levenshtein_kepler_t = levenshtein_distances<linear_gap_costs_t, ualloc_t, sz_caps_ck_k>;
using affine_levenshtein_kepler_t = levenshtein_distances<affine_gap_costs_t, ualloc_t, sz_caps_ck_k>;

using levenshtein_hopper_t = levenshtein_distances<linear_gap_costs_t, ualloc_t, sz_caps_ckh_k>;
using affine_levenshtein_hopper_t = levenshtein_distances<affine_gap_costs_t, ualloc_t, sz_caps_ckh_k>;

using needleman_wunsch_cuda_t =
    needleman_wunsch_scores<error_costs_32x32_t, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
using smith_waterman_cuda_t = smith_waterman_scores<error_costs_32x32_t, linear_gap_costs_t, ualloc_t, sz_cap_cuda_k>;

using affine_needleman_wunsch_cuda_t =
    needleman_wunsch_scores<error_costs_32x32_t, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k>;
using affine_smith_waterman_cuda_t =
    smith_waterman_scores<error_costs_32x32_t, affine_gap_costs_t, ualloc_t, sz_cap_cuda_k>;

using needleman_wunsch_hopper_t =
    needleman_wunsch_scores<error_costs_32x32_t, linear_gap_costs_t, ualloc_t, sz_caps_ckh_k>;
using smith_waterman_hopper_t = smith_waterman_scores<error_costs_32x32_t, linear_gap_costs_t, ualloc_t, sz_caps_ckh_k>;

using affine_needleman_wunsch_hopper_t =
    needleman_wunsch_scores<error_costs_32x32_t, affine_gap_costs_t, ualloc_t, sz_caps_ckh_k>;
using affine_smith_waterman_hopper_t =
    smith_waterman_scores<error_costs_32x32_t, affine_gap_costs_t, ualloc_t, sz_caps_ckh_k>;

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

#pragma region Tiled large-input device kernel (register micro-tiles)

/**
 *  @brief One linear-gap DP cell: `opt(diag + sub, top + gap, left + gap)` (+ a Smith-Waterman ReLU clamp to 0 for the
 *         local objective). On @b Hopper with <=32-bit cells it fuses the whole thing into a single DPX
 *         `__viaddmin/max_s32[_relu]` (the gaps share `gap`, so `opt(top,left)+gap == opt(top+gap,left+gap)`); on every
 *         other capability or 64-bit cell it stays scalar. Mirrors the warp-tier `tile_scorer` Hopper specializations.
 */
template <sz_similarity_objective_t objective_, sz_similarity_locality_t locality_, sz_capability_t capability_,
          typename score_type_>
__forceinline__ __device__ score_type_ tiled_dp_cell_(score_type_ diag, score_type_ top, score_type_ left,
                                                      score_type_ substitution, score_type_ gap) noexcept {
    using score_t = score_type_;
    static constexpr bool is_local_k = locality_ == sz_similarity_local_k;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if constexpr ((capability_ & sz_cap_hopper_k) != 0 && sizeof(score_t) <= 4) {
        int const d = static_cast<int>(diag), s = static_cast<int>(substitution), g = static_cast<int>(gap);
        int const t = static_cast<int>(top), l = static_cast<int>(left);
        if constexpr (objective_ == sz_minimize_distance_k)
            return static_cast<score_t>(__viaddmin_s32(d, s, min(t, l) + g)); // min(diag+sub, opt(top,left)+gap)
        else if constexpr (is_local_k)
            return static_cast<score_t>(__viaddmax_s32_relu(d, s, max(t, l) + g)); // Smith-Waterman: clamp to 0
        else return static_cast<score_t>(__viaddmax_s32(d, s, max(t, l) + g));
    }
    else
#endif
    {
        score_t cell = min_or_max<objective_>(
            static_cast<score_t>(diag + substitution),
            min_or_max<objective_>(static_cast<score_t>(top + gap), static_cast<score_t>(left + gap)));
        if constexpr (is_local_k) cell = min_or_max<objective_, score_t>(cell, 0);
        return cell;
    }
}

/**
 *  @brief One affine-gap (Gotoh) DP cell. Returns @b M and writes the running vertical/horizontal gaps @p v_out /
 *         @p h_out. On @b Hopper with <=32-bit cells each of V, H, and M is a single fused DPX
 *         `__viaddmin/max_s32[_relu]`; scalar otherwise. @sa tiled_dp_cell_.
 */
template <sz_similarity_objective_t objective_, sz_similarity_locality_t locality_, sz_capability_t capability_,
          typename score_type_>
__forceinline__ __device__ score_type_ tiled_dp_cell_affine_( //
    score_type_ diag, score_type_ top_m, score_type_ top_v, score_type_ left_m, score_type_ left_h,
    score_type_ substitution, score_type_ open, score_type_ extend, score_type_ &v_out, score_type_ &h_out) noexcept {
    using score_t = score_type_;
    static constexpr bool is_local_k = locality_ == sz_similarity_local_k;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    if constexpr ((capability_ & sz_cap_hopper_k) != 0 && sizeof(score_t) <= 4) {
        int const d = static_cast<int>(diag), s = static_cast<int>(substitution);
        int const o = static_cast<int>(open), e = static_cast<int>(extend);
        if constexpr (objective_ == sz_minimize_distance_k) {
            int const v = __viaddmin_s32(static_cast<int>(top_m), o, static_cast<int>(top_v) + e);
            int const h = __viaddmin_s32(static_cast<int>(left_m), o, static_cast<int>(left_h) + e);
            v_out = static_cast<score_t>(v), h_out = static_cast<score_t>(h);
            return static_cast<score_t>(__viaddmin_s32(d, s, min(v, h)));
        }
        else {
            int const v = __viaddmax_s32(static_cast<int>(top_m), o, static_cast<int>(top_v) + e);
            int const h = __viaddmax_s32(static_cast<int>(left_m), o, static_cast<int>(left_h) + e);
            v_out = static_cast<score_t>(v), h_out = static_cast<score_t>(h);
            int const gap_best = max(v, h);
            return is_local_k ? static_cast<score_t>(__viaddmax_s32_relu(d, s, gap_best))
                              : static_cast<score_t>(__viaddmax_s32(d, s, gap_best));
        }
    }
    else
#endif
    {
        score_t const v = min_or_max<objective_>(static_cast<score_t>(top_m + open),
                                                 static_cast<score_t>(top_v + extend));
        score_t const h = min_or_max<objective_>(static_cast<score_t>(left_m + open),
                                                 static_cast<score_t>(left_h + extend));
        v_out = v, h_out = h;
        score_t if_substitution = static_cast<score_t>(diag + substitution);
        if constexpr (is_local_k) if_substitution = min_or_max<objective_, score_t>(if_substitution, 0);
        return min_or_max<objective_>(min_or_max<objective_>(v, h), if_substitution);
    }
}

/**
 *  @brief Tiled large-matrix linear-gap scorer: one @b warp owns a 128-wide tile-COLUMN and marches it top-to-bottom,
 *         computing each 128x128 tile via 4x4 register @b micro-tiles (lane @e l owns micro-column @e l; the left
 *         neighbour's right column + top-right corner arrive by @b `__shfl_up`; no shared micro-halos). The top edge
 *         is free (carried in registers down the column); only the left edge + corner cross warps, via the global
 *         @p row_frontier / @p corner_frontier gated by acquire/release @p progress counters - so there is no
 *         `grid.sync` and no cooperative launch. One launch handles one pair (grid over its tile-columns).
 *
 *  Reaches ~190 GCUPS on one 50K^2 pair and ~650 on 200K^2 (vs ~8 for the anti-diagonal device kernel); a batch of
 *  pairs run concurrently approaches the ~1.4 TCUPS warp-batch ceiling. @sa register_levenshtein for the short-input tier.
 */
template <                                                         //
    unsigned warps_per_block_,                                     //
    typename char_type_ = char,                                    //
    typename score_type_ = u32_t,                                  //
    typename final_score_type_ = size_t,                           //
    typename substituter_type_ = uniform_substitution_costs_t,     //
    sz_similarity_objective_t objective_ = sz_minimize_distance_k, //
    sz_similarity_locality_t locality_ = sz_similarity_global_k,   //
    sz_capability_t capability_ = sz_cap_cuda_k,                   //
    typename task_type_ = void                                     //
    >
__global__ __launch_bounds__(warps_per_block_ * 32) void tiled_score_across_cuda_device_(    //
    task_type_ *tasks,                                                                       //
    score_type_ *row_frontier_base, score_type_ *corner_frontier_base, u32_t *progress_base, //
    u32_t row_stride, u32_t corner_stride,                                                   //
    substituter_type_ const substituter, linear_gap_costs_t const gap_costs) {

    using score_t = score_type_;
    static constexpr unsigned tile_side_k = 128, micro_side_k = 4, lanes_k = 32,
                              micro_rows_k = tile_side_k / micro_side_k;
    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr bool is_local_k = locality_ == sz_similarity_local_k;
    score_t const gap = gap_costs.open_or_extend;
    // Each warp stages, once per tile-row, BOTH its current tile's query window AND its incoming left boundary into
    // shared. Staging the left boundary (a coalesced warp-wide read of the global `row_frontier` slice) and then serving
    // the inner loop's lane-0 left/corner reads from `shared_left` is a measured +15% over reading `row_frontier`
    // directly per micro-tile: it amortizes the scattered, repeatedly-latent global loads the profiler flagged. (An
    // on-chip ring that also moves the right-boundary hand-off off-chip was measured *slower* - the small frontier is
    // L2-hot, so the ring's extra shared pressure and producer/consumer coupling outweigh the saved traffic.)
    __shared__ char_type_ shared_query[warps_per_block_][tile_side_k];
    __shared__ score_t shared_left[warps_per_block_][tile_side_k]; // staged incoming left boundary, per warp
    unsigned const warp_in_block = threadIdx.x >> 5;

    // Mirror the substitution table into shared once (a no-op for uniform costs); ALL threads must reach this before any
    // early return, as the class-cost path runs a block-wide `__syncthreads` inside.
    substituter_type_ const substituter_shared = load_substituter_into_shared_(substituter);

    // Cross-pair batching: `blockIdx.y` selects one (shorter, longer) pair from the task array; its frontier scratch
    // is a `pair * stride` slice of the shared buffers (sized to the largest pair in the batch). A single-pair launch
    // (`gridDim.y == 1`) reduces to the original behaviour. Per-pair lengths/pointers/result are read here so the
    // proven micro-tile body below stays byte-for-byte identical.
    u32_t const pair = blockIdx.y;
    char_type_ const *const shorter_ptr = tasks[pair].shorter.data();
    char_type_ const *const longer_ptr = tasks[pair].longer.data();
    u32_t const shorter_length = static_cast<u32_t>(tasks[pair].shorter.size());
    u32_t const longer_length = static_cast<u32_t>(tasks[pair].longer.size());
    final_score_type_ *const result_ptr = reinterpret_cast<final_score_type_ *>(&tasks[pair].result);
    u32_t const tile_grid_rows = (shorter_length + tile_side_k - 1) / tile_side_k;
    u32_t const tile_grid_columns = (longer_length + tile_side_k - 1) / tile_side_k;
    score_type_ *const row_frontier = row_frontier_base + static_cast<size_t>(pair) * row_stride;
    score_type_ *const corner_frontier = corner_frontier_base + static_cast<size_t>(pair) * corner_stride;
    u32_t *const progress = progress_base + static_cast<size_t>(pair) * corner_stride;

    unsigned const lane = threadIdx.x & 31u;
    u32_t const tile_column = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; // one warp per tile-column
    if (tile_column >= tile_grid_columns) return;
    u32_t const tile_first_column = tile_column * tile_side_k;

    // This lane's target characters (its micro-column), constant across the whole column march. Columns past
    // `longer_length` (the last tile may be partial) read a sentinel that never matches - those padded cells are
    // computed but never feed a valid cell, and are excluded from the result.
    char_type_ target_chars[micro_side_k];
    for (unsigned a = 0; a < micro_side_k; ++a) {
        u32_t const target_index = tile_first_column + lane * micro_side_k + a;
        target_chars[a] = target_index < longer_length ? longer_ptr[target_index] : static_cast<char_type_>(0xFF);
    }
    // Top edge carried in registers down the column (free vertical hand-off); row 0 is the matrix boundary - the same
    // value `tile_scorer::init_score` produces (0 for local, gap·column for the linear global gap ladder).
    score_t carry_top[micro_side_k];
    for (unsigned a = 0; a < micro_side_k; ++a)
        carry_top[a] = is_local_k ? score_t {0}
                                  : static_cast<score_t>(gap * (tile_first_column + lane * micro_side_k + a + 1));
    score_t running_best = 0; // local (SW) keeps the global maximum across all cells

    // The corner cell M[shorter_length][longer_length] lives in exactly one tile; only that warp/tile pays the per-cell
    // corner test. A "full" tile (no partial edge) skips all bounds checks in the hot loop.
    u32_t const corner_tile_row = (shorter_length - 1) / tile_side_k;
    u32_t const corner_tile_column = (longer_length - 1) / tile_side_k;
    bool const owns_corner_column = tile_column == corner_tile_column;

    for (u32_t tile_row = 0; tile_row < tile_grid_rows; ++tile_row) {
        u32_t const tile_first_row = tile_row * tile_side_k;
        bool const tile_is_full = tile_first_row + tile_side_k <= shorter_length &&
                                  tile_first_column + tile_side_k <= longer_length;
        bool const tile_has_corner = owns_corner_column && tile_row == corner_tile_row;
        // Wait for the left column to publish this tile-row (acquire orders its halo writes before our reads).
        if (tile_column > 0 && lane == 0) {
            cuda::atomic_ref<u32_t, cuda::thread_scope_device> left_progress(progress[tile_column - 1]);
            while (left_progress.load(cuda::memory_order_acquire) <= tile_row) {}
        }
        __syncwarp();
        // Stage this tile-row's left boundary (a coalesced read of the global `row_frontier` slice) and the query rows
        // into shared. `shared_left[r]` holds M[tile_first_row + 1 + r][tile_first_column]; the inner loop's lane-0 then
        // serves its left/corner reads from shared instead of re-touching the latent global frontier per micro-tile.
        for (unsigned r = lane; r < tile_side_k; r += 32) {
            shared_left[warp_in_block][r] = row_frontier[tile_first_row + 1 + r];
            shared_query[warp_in_block][r] = tile_first_row + r < shorter_length ? shorter_ptr[tile_first_row + r]
                                                                                 : static_cast<char_type_>(0xFE);
        }
        __syncwarp();
        score_t const tile_corner = corner_frontier[tile_column];
        // This tile's bottom-left boundary M[tile_row·128+128][tile_column·128] is the last staged left-boundary value
        // (the global slice was read into `shared_left` before the march writes its own right edge into `row_frontier`);
        // it becomes the diagonal corner for the tile directly below.
        score_t const tile_bottom_left = shared_left[warp_in_block][tile_side_k - 1];

        // Micro-tile anti-diagonal wavefront within the tile (lane = micro-column, step skew = micro-row), specialized on
        // `tile_fast_`. A FULL non-corner tile (global) - or any full tile (local) - needs NO per-cell result/bounds work,
        // so the `tile_fast_` instantiation's inner loop is provably free of the guarded global store / bounds branch that
        // otherwise inhibits register optimization of the hot path (measured +17% @batch-8, +33% @batch-32). The corner
        // and partial-edge tiles - a vanishing fraction - take the checked path.
        auto const march_tile = [&]<bool tile_fast_>() {
            score_t last_right[micro_side_k];
            for (unsigned a = 0; a < micro_side_k; ++a) last_right[a] = 0;
            score_t last_topright = 0;
            unsigned const steps = micro_rows_k + lanes_k - 1;
            for (unsigned step = 0; step < steps; ++step) {
                unsigned const micro_row = step - lane;
                bool const active = (step >= lane) && (micro_row < micro_rows_k);
                score_t neighbor_right[micro_side_k];
                for (unsigned a = 0; a < micro_side_k; ++a)
                    neighbor_right[a] = __shfl_up_sync(0xffffffff, last_right[a], 1);
                score_t neighbor_topright = __shfl_up_sync(0xffffffff, last_topright, 1);
                if (active) {
                    u32_t const micro_first_row = tile_first_row + micro_row * micro_side_k;
                    score_t left[micro_side_k], corner;
                    if (lane == 0) {
                        // Lane 0's left boundary + diagonal corner come from the on-chip staged `shared_left` (indexed by
                        // the micro-row within the tile), not the global frontier; the tile-top corner is the carried diagonal.
                        for (unsigned a = 0; a < micro_side_k; ++a)
                            left[a] = shared_left[warp_in_block][micro_row * micro_side_k + a];
                        corner = micro_row == 0 ? tile_corner
                                                : shared_left[warp_in_block][micro_row * micro_side_k - 1];
                    }
                    else {
                        for (unsigned a = 0; a < micro_side_k; ++a) left[a] = neighbor_right[a];
                        corner = neighbor_topright;
                    }
                    score_t const send_topright =
                        carry_top[micro_side_k - 1]; // M[micro_first_row][col+4], corner for lane+1
                    score_t above[micro_side_k + 1];
                    above[0] = corner;
                    for (unsigned a = 0; a < micro_side_k; ++a) above[a + 1] = carry_top[a];
                    score_t my_right[micro_side_k];
                    for (unsigned i = 1; i <= micro_side_k; ++i) {
                        score_t row[micro_side_k + 1];
                        row[0] = left[i - 1];
                        [[maybe_unused]] u32_t const matrix_row = micro_first_row +
                                                                  i; // 1-based DP row; > len on a partial tile
                        char_type_ const query_char = shared_query[warp_in_block][micro_row * micro_side_k + i - 1];
                        for (unsigned j = 1; j <= micro_side_k; ++j) {
                            // Polymorphic cost: `uniform_substitution_costs_t` for Levenshtein, the 32-class table for NW/SW.
                            score_t const substitution = static_cast<score_t>(
                                substituter_shared(query_char, target_chars[j - 1]));
                            // One fused DP cell (Hopper DPX when the capability + cell width allow; the helper also applies
                            // the Smith-Waterman clamp-to-0 for the local objective).
                            score_t cell = tiled_dp_cell_<objective_k, locality_, capability_, score_t>(
                                above[j - 1], above[j], row[j - 1], substitution, gap);
                            if constexpr (is_local_k) {
                                if constexpr (tile_fast_)
                                    running_best = min_or_max<objective_k, score_t>(running_best, cell);
                                else {
                                    u32_t const matrix_column = tile_first_column + lane * micro_side_k + j;
                                    if (tile_is_full ||
                                        (matrix_row <= shorter_length && matrix_column <= longer_length))
                                        running_best = min_or_max<objective_k, score_t>(running_best, cell);
                                }
                            }
                            else if constexpr (!tile_fast_) {
                                if (tile_has_corner && matrix_row == shorter_length &&
                                    tile_first_column + lane * micro_side_k + j == longer_length)
                                    *result_ptr = static_cast<final_score_type_>(
                                        cell); // global result = the true corner cell
                            }
                            row[j] = cell;
                        }
                        my_right[i - 1] = row[micro_side_k];
                        for (unsigned a = 0; a <= micro_side_k; ++a) above[a] = row[a];
                    }
                    for (unsigned a = 0; a < micro_side_k; ++a)
                        carry_top[a] = above[a + 1]; // bottom row -> next micro-row top
                    for (unsigned a = 0; a < micro_side_k; ++a) last_right[a] = my_right[a];
                    last_topright = send_topright;
                    if (lane ==
                        lanes_k - 1) // rightmost micro-column: publish the tile's right column for the next warp
                        for (unsigned a = 0; a < micro_side_k; ++a) row_frontier[micro_first_row + a + 1] = my_right[a];
                }
                __syncwarp();
            }
        };
        // Full tiles with no result cell to capture (global: not the corner tile; local: every full tile) take the fast,
        // store-free path; corner / partial-edge tiles take the checked path.
        if (tile_is_full && (is_local_k || !tile_has_corner)) march_tile.template operator()<true>();
        else march_tile.template operator()<false>();
        if (lane == 0)
            corner_frontier[tile_column] = tile_bottom_left; // bottom-left corner -> diagonal for the tile below
        __syncwarp();
        if (lane == 0) { // release: publish this tile-row so the right neighbour column can proceed
            cuda::atomic_ref<u32_t, cuda::thread_scope_device> my_progress(progress[tile_column]);
            my_progress.store(tile_row + 1, cuda::memory_order_release);
        }
    }
    if constexpr (
        is_local_k) { // Smith-Waterman: reduce the per-lane maxima across the warp and publish the global best
        running_best = pick_best_in_warp_<objective_k>(running_best);
        if (lane == 0)
            atomicMax(reinterpret_cast<unsigned long long *>(result_ptr),
                      static_cast<unsigned long long>(running_best));
    }
}

/**
 *  @brief Seeds the global frontier for `tiled_score_across_cuda_device_`: the left boundary column `M[i][0]`, the
 *         per-tile-column diagonal corners `M[0][tc·128]`, the `progress` counters, and the local-result slot. A
 *         separate launch supplies the grid-wide barrier the (cooperative-free) tiled kernel deliberately avoids.
 */
template <typename score_type_, typename final_score_type_, sz_similarity_objective_t objective_,
          sz_similarity_locality_t locality_, typename task_type_ = void>
__global__ void tiled_frontier_init_(task_type_ *tasks, score_type_ *row_frontier_base,
                                     score_type_ *corner_frontier_base, u32_t *progress_base, u32_t row_stride,
                                     u32_t corner_stride, linear_gap_costs_t const gap_costs) {
    using score_t = score_type_;
    static constexpr unsigned tile_side_k = 128;
    static constexpr bool is_local_k = locality_ == sz_similarity_local_k;
    score_t const gap = gap_costs.open_or_extend;
    // One pair per `blockIdx.y`; seed only that pair's slice (sized to its own matrix, within the padded stride).
    u32_t const pair = blockIdx.y;
    u32_t const shorter_length = static_cast<u32_t>(tasks[pair].shorter.size());
    u32_t const longer_length = static_cast<u32_t>(tasks[pair].longer.size());
    u32_t const padded_rows = ((shorter_length + tile_side_k - 1) / tile_side_k) * tile_side_k;
    u32_t const tile_grid_columns = (longer_length + tile_side_k - 1) / tile_side_k;
    score_t *const row_frontier = row_frontier_base + static_cast<size_t>(pair) * row_stride;
    score_t *const corner_frontier = corner_frontier_base + static_cast<size_t>(pair) * corner_stride;
    u32_t *const progress = progress_base + static_cast<size_t>(pair) * corner_stride;
    u32_t const global_index = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (u32_t row = global_index; row <= padded_rows; row += stride)
        row_frontier[row] = is_local_k ? score_t {0} : static_cast<score_t>(gap * row);
    for (u32_t tile_column = global_index; tile_column < tile_grid_columns; tile_column += stride) {
        corner_frontier[tile_column] = is_local_k ? score_t {0} : static_cast<score_t>(gap * tile_column * 128u);
        progress[tile_column] = 0;
    }
    if (global_index == 0)
        *reinterpret_cast<final_score_type_ *>(&tasks[pair].result) = final_score_type_ {
            0}; // local seed; global overwritten
}

/**
 *  @brief Affine-gap (Gotoh) sibling of `tiled_score_across_cuda_device_`: same warp-per-tile-column data-flow and
 *         4x4 register micro-tiles, but each cell carries three matrices - the primary @b M plus the running vertical
 *         (@b V, opens from M above) and horizontal (@b H, opens from M to the left) gap matrices. The top edge carries
 *         M+V down the column in registers; the left edge + corner cross warps via @p row_frontier_m / @p row_frontier_d
 *         (M and H right-edges) and @p corner_frontier_m (the diagonal M), gated by acquire/release @p progress.
 *
 *  Matches the library's serial/cooperative affine convention exactly: M boundary is the gap ladder
 *  `open + extend·(k-1)`, the gap matrices are seeded one `open+extend` "worse" (the discard sentinel that never
 *  overflows), and the recurrence is `V = opt(M_up+open, V_up+extend)`, `H = opt(M_left+open, H_left+extend)`,
 *  `M = pick_best(opt(V,H), M_diag+sub)` (with an extra `pick_best(·,0)` reset on the substitution path for SW).
 */
template <                                                         //
    unsigned warps_per_block_,                                     //
    typename char_type_ = char,                                    //
    typename score_type_ = u32_t,                                  //
    typename final_score_type_ = size_t,                           //
    typename substituter_type_ = uniform_substitution_costs_t,     //
    sz_similarity_objective_t objective_ = sz_minimize_distance_k, //
    sz_similarity_locality_t locality_ = sz_similarity_global_k,   //
    sz_capability_t capability_ = sz_cap_cuda_k,                   //
    typename task_type_ = void                                     //
    >
__global__ __launch_bounds__(warps_per_block_ * 32) void tiled_score_affine_across_cuda_device_( //
    task_type_ *tasks,                                                                           //
    score_type_ *row_frontier_m_base, score_type_ *row_frontier_d_base,                          //
    score_type_ *corner_frontier_m_base, u32_t *progress_base,                                   //
    u32_t row_stride, u32_t corner_stride,                                                       //
    substituter_type_ const substituter, affine_gap_costs_t const gap_costs) {

    using score_t = score_type_;
    static constexpr unsigned tile_side_k = 128, micro_side_k = 4, lanes_k = 32,
                              micro_rows_k = tile_side_k / micro_side_k;
    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr bool is_local_k = locality_ == sz_similarity_local_k;
    score_t const open = gap_costs.open, extend = gap_costs.extend;
    // M boundary = the affine gap ladder `open + extend·(d-1)`; gap (V/H) boundary = one `open+extend` "worse" - the
    // discard sentinel that, unlike `INF`, never overflows on subsequent `+extend` (mirrors `tile_scorer::init_gap`).
    auto const boundary_m = [&](u32_t d) -> score_t {
        return is_local_k ? score_t {0} : static_cast<score_t>(d ? open + extend * (d - 1) : 0);
    };
    auto const boundary_gap = [&](u32_t d) -> score_t {
        return is_local_k ? static_cast<score_t>(open + extend)
                          : static_cast<score_t>((open + extend) + (d ? open + extend * (d - 1) : 0));
    };
    // Stage the query window and BOTH incoming left boundaries (the primary M and the horizontal-gap H) into shared once
    // per tile-row, then serve the inner loop's lane-0 reads on-chip - the same measured +15% frontier-staging win as the
    // linear kernel, applied to the two affine frontier slices.
    __shared__ char_type_ shared_query[warps_per_block_][tile_side_k];
    __shared__ score_t shared_left_m[warps_per_block_][tile_side_k]; // staged left M boundary, per warp
    __shared__ score_t shared_left_h[warps_per_block_][tile_side_k]; // staged left H (deletion) boundary, per warp
    unsigned const warp_in_block = threadIdx.x >> 5;
    substituter_type_ const substituter_shared = load_substituter_into_shared_(substituter);

    u32_t const pair = blockIdx.y;
    char_type_ const *const shorter_ptr = tasks[pair].shorter.data();
    char_type_ const *const longer_ptr = tasks[pair].longer.data();
    u32_t const shorter_length = static_cast<u32_t>(tasks[pair].shorter.size());
    u32_t const longer_length = static_cast<u32_t>(tasks[pair].longer.size());
    final_score_type_ *const result_ptr = reinterpret_cast<final_score_type_ *>(&tasks[pair].result);
    u32_t const tile_grid_rows = (shorter_length + tile_side_k - 1) / tile_side_k;
    u32_t const tile_grid_columns = (longer_length + tile_side_k - 1) / tile_side_k;
    score_type_ *const row_frontier_m = row_frontier_m_base + static_cast<size_t>(pair) * row_stride;
    score_type_ *const row_frontier_d = row_frontier_d_base + static_cast<size_t>(pair) * row_stride;
    score_type_ *const corner_frontier_m = corner_frontier_m_base + static_cast<size_t>(pair) * corner_stride;
    u32_t *const progress = progress_base + static_cast<size_t>(pair) * corner_stride;

    unsigned const lane = threadIdx.x & 31u;
    u32_t const tile_column = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (tile_column >= tile_grid_columns) return;
    u32_t const tile_first_column = tile_column * tile_side_k;

    char_type_ target_chars[micro_side_k];
    for (unsigned a = 0; a < micro_side_k; ++a) {
        u32_t const target_index = tile_first_column + lane * micro_side_k + a;
        target_chars[a] = target_index < longer_length ? longer_ptr[target_index] : static_cast<char_type_>(0xFF);
    }
    // Top edge carried in registers down the column: M and the running vertical gap V (the row directly above).
    score_t carry_top_m[micro_side_k], carry_top_v[micro_side_k];
    for (unsigned a = 0; a < micro_side_k; ++a) {
        u32_t const column = tile_first_column + lane * micro_side_k + a + 1;
        carry_top_m[a] = boundary_m(column);
        carry_top_v[a] = boundary_gap(column);
    }
    score_t running_best = 0;

    u32_t const corner_tile_row = (shorter_length - 1) / tile_side_k;
    u32_t const corner_tile_column = (longer_length - 1) / tile_side_k;
    bool const owns_corner_column = tile_column == corner_tile_column;

    for (u32_t tile_row = 0; tile_row < tile_grid_rows; ++tile_row) {
        u32_t const tile_first_row = tile_row * tile_side_k;
        bool const tile_is_full = tile_first_row + tile_side_k <= shorter_length &&
                                  tile_first_column + tile_side_k <= longer_length;
        bool const tile_has_corner = owns_corner_column && tile_row == corner_tile_row;
        if (tile_column > 0 && lane == 0) {
            cuda::atomic_ref<u32_t, cuda::thread_scope_device> left_progress(progress[tile_column - 1]);
            while (left_progress.load(cuda::memory_order_acquire) <= tile_row) {}
        }
        __syncwarp();
        score_t const tile_corner_m = corner_frontier_m[tile_column];
        // Stage both left boundaries (coalesced reads of the global M/H frontier slices) and the query rows into shared.
        for (unsigned r = lane; r < tile_side_k; r += 32) {
            shared_left_m[warp_in_block][r] = row_frontier_m[tile_first_row + 1 + r];
            shared_left_h[warp_in_block][r] = row_frontier_d[tile_first_row + 1 + r];
            shared_query[warp_in_block][r] = tile_first_row + r < shorter_length ? shorter_ptr[tile_first_row + r]
                                                                                 : static_cast<char_type_>(0xFE);
        }
        __syncwarp();
        // Bottom-left M boundary (the staged copy survives the march writing its own right edge into `row_frontier_m`);
        // becomes the diagonal corner for the tile below.
        score_t const tile_bottom_left_m = shared_left_m[warp_in_block][tile_side_k - 1];

        // Full non-corner (global) / full (local) tiles take a `tile_fast_` march with the per-cell result/bounds work
        // compile-time elided - the same store-free hot-loop speedup as the linear kernel (see its comment).
        auto const march_tile = [&]<bool tile_fast_>() {
            score_t last_right_m[micro_side_k], last_right_h[micro_side_k];
            for (unsigned a = 0; a < micro_side_k; ++a) last_right_m[a] = 0, last_right_h[a] = 0;
            score_t last_topright_m = 0;
            unsigned const steps = micro_rows_k + lanes_k - 1;
            for (unsigned step = 0; step < steps; ++step) {
                unsigned const micro_row = step - lane;
                bool const active = (step >= lane) && (micro_row < micro_rows_k);
                score_t neighbor_right_m[micro_side_k], neighbor_right_h[micro_side_k];
                for (unsigned a = 0; a < micro_side_k; ++a) {
                    neighbor_right_m[a] = __shfl_up_sync(0xffffffff, last_right_m[a], 1);
                    neighbor_right_h[a] = __shfl_up_sync(0xffffffff, last_right_h[a], 1);
                }
                score_t neighbor_topright_m = __shfl_up_sync(0xffffffff, last_topright_m, 1);
                if (active) {
                    u32_t const micro_first_row = tile_first_row + micro_row * micro_side_k;
                    score_t left_m[micro_side_k], left_h[micro_side_k], corner_m;
                    if (lane == 0) {
                        // Lane 0's left M/H boundaries + diagonal M corner come from the on-chip staged slices.
                        for (unsigned a = 0; a < micro_side_k; ++a) {
                            left_m[a] = shared_left_m[warp_in_block][micro_row * micro_side_k + a];
                            left_h[a] = shared_left_h[warp_in_block][micro_row * micro_side_k + a];
                        }
                        corner_m = micro_row == 0 ? tile_corner_m
                                                  : shared_left_m[warp_in_block][micro_row * micro_side_k - 1];
                    }
                    else {
                        for (unsigned a = 0; a < micro_side_k; ++a)
                            left_m[a] = neighbor_right_m[a], left_h[a] = neighbor_right_h[a];
                        corner_m = neighbor_topright_m;
                    }
                    score_t const send_topright_m =
                        carry_top_m[micro_side_k - 1]; // M[micro_first_row][col+4] -> lane+1's corner
                    score_t above_m[micro_side_k + 1], above_v[micro_side_k + 1];
                    above_m[0] = corner_m; // the diagonal M; above_v[0] is never read
                    for (unsigned a = 0; a < micro_side_k; ++a)
                        above_m[a + 1] = carry_top_m[a], above_v[a + 1] = carry_top_v[a];
                    score_t my_right_m[micro_side_k], my_right_h[micro_side_k];
                    for (unsigned i = 1; i <= micro_side_k; ++i) {
                        score_t row_m[micro_side_k + 1], row_h[micro_side_k + 1], row_v[micro_side_k + 1];
                        row_m[0] = left_m[i - 1], row_h[0] = left_h[i - 1]; // row_v[0] never read
                        [[maybe_unused]] u32_t const matrix_row = micro_first_row + i;
                        char_type_ const query_char = shared_query[warp_in_block][micro_row * micro_side_k + i - 1];
                        for (unsigned j = 1; j <= micro_side_k; ++j) {
                            score_t const substitution = static_cast<score_t>(
                                substituter_shared(query_char, target_chars[j - 1]));
                            // One fused Gotoh cell (Hopper DPX when capability + cell width allow); writes the running V/H gaps.
                            score_t v_new, h_new;
                            score_t cell = tiled_dp_cell_affine_<objective_k, locality_, capability_, score_t>(
                                above_m[j - 1], above_m[j], above_v[j], row_m[j - 1], row_h[j - 1], substitution, open,
                                extend, v_new, h_new);
                            if constexpr (is_local_k) {
                                if constexpr (tile_fast_)
                                    running_best = min_or_max<objective_k, score_t>(running_best, cell);
                                else {
                                    u32_t const matrix_column = tile_first_column + lane * micro_side_k + j;
                                    if (tile_is_full ||
                                        (matrix_row <= shorter_length && matrix_column <= longer_length))
                                        running_best = min_or_max<objective_k, score_t>(running_best, cell);
                                }
                            }
                            else if constexpr (!tile_fast_) {
                                if (tile_has_corner && matrix_row == shorter_length &&
                                    tile_first_column + lane * micro_side_k + j == longer_length)
                                    *result_ptr = static_cast<final_score_type_>(cell);
                            }
                            row_m[j] = cell, row_h[j] = h_new, row_v[j] = v_new;
                        }
                        my_right_m[i - 1] = row_m[micro_side_k], my_right_h[i - 1] = row_h[micro_side_k];
                        for (unsigned a = 0; a <= micro_side_k; ++a) above_m[a] = row_m[a], above_v[a] = row_v[a];
                    }
                    for (unsigned a = 0; a < micro_side_k; ++a)
                        carry_top_m[a] = above_m[a + 1], carry_top_v[a] = above_v[a + 1];
                    for (unsigned a = 0; a < micro_side_k; ++a)
                        last_right_m[a] = my_right_m[a], last_right_h[a] = my_right_h[a];
                    last_topright_m = send_topright_m;
                    if (lane == lanes_k - 1)
                        for (unsigned a = 0; a < micro_side_k; ++a) {
                            row_frontier_m[micro_first_row + a + 1] = my_right_m[a];
                            row_frontier_d[micro_first_row + a + 1] = my_right_h[a];
                        }
                }
                __syncwarp();
            }
        };
        if (tile_is_full && (is_local_k || !tile_has_corner)) march_tile.template operator()<true>();
        else march_tile.template operator()<false>();
        if (lane == 0) corner_frontier_m[tile_column] = tile_bottom_left_m;
        __syncwarp();
        if (lane == 0) {
            cuda::atomic_ref<u32_t, cuda::thread_scope_device> my_progress(progress[tile_column]);
            my_progress.store(tile_row + 1, cuda::memory_order_release);
        }
    }
    if constexpr (is_local_k) {
        running_best = pick_best_in_warp_<objective_k>(running_best);
        if (lane == 0)
            atomicMax(reinterpret_cast<unsigned long long *>(result_ptr),
                      static_cast<unsigned long long>(running_best));
    }
}

/**
 *  @brief Seeds the affine frontier for `tiled_score_affine_across_cuda_device_`: the left-column M and H (deletion)
 *         boundaries, the per-tile-column diagonal M corners, the `progress` counters, and the local-result slot.
 */
template <typename score_type_, typename final_score_type_, sz_similarity_objective_t objective_,
          sz_similarity_locality_t locality_, typename task_type_ = void>
__global__ void tiled_frontier_init_affine_(task_type_ *tasks, score_type_ *row_frontier_m_base,
                                            score_type_ *row_frontier_d_base, score_type_ *corner_frontier_m_base,
                                            u32_t *progress_base, u32_t row_stride, u32_t corner_stride,
                                            affine_gap_costs_t const gap_costs) {
    using score_t = score_type_;
    static constexpr unsigned tile_side_k = 128;
    static constexpr bool is_local_k = locality_ == sz_similarity_local_k;
    score_t const open = gap_costs.open, extend = gap_costs.extend;
    auto const boundary_m = [&](u32_t d) -> score_t {
        return is_local_k ? score_t {0} : static_cast<score_t>(d ? open + extend * (d - 1) : 0);
    };
    auto const boundary_gap = [&](u32_t d) -> score_t {
        return is_local_k ? static_cast<score_t>(open + extend)
                          : static_cast<score_t>((open + extend) + (d ? open + extend * (d - 1) : 0));
    };
    u32_t const pair = blockIdx.y;
    u32_t const shorter_length = static_cast<u32_t>(tasks[pair].shorter.size());
    u32_t const longer_length = static_cast<u32_t>(tasks[pair].longer.size());
    u32_t const padded_rows = ((shorter_length + tile_side_k - 1) / tile_side_k) * tile_side_k;
    u32_t const tile_grid_columns = (longer_length + tile_side_k - 1) / tile_side_k;
    score_t *const row_frontier_m = row_frontier_m_base + static_cast<size_t>(pair) * row_stride;
    score_t *const row_frontier_d = row_frontier_d_base + static_cast<size_t>(pair) * row_stride;
    score_t *const corner_frontier_m = corner_frontier_m_base + static_cast<size_t>(pair) * corner_stride;
    u32_t *const progress = progress_base + static_cast<size_t>(pair) * corner_stride;
    u32_t const global_index = blockIdx.x * blockDim.x + threadIdx.x, stride = gridDim.x * blockDim.x;
    for (u32_t row = global_index; row <= padded_rows; row += stride) {
        row_frontier_m[row] = boundary_m(row);
        row_frontier_d[row] = boundary_gap(row);
    }
    for (u32_t tile_column = global_index; tile_column < tile_grid_columns; tile_column += stride) {
        corner_frontier_m[tile_column] = boundary_m(tile_column * tile_side_k);
        progress[tile_column] = 0;
    }
    if (global_index == 0) *reinterpret_cast<final_score_type_ *>(&tasks[pair].result) = final_score_type_ {0};
}

#pragma endregion

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
__global__ void linear_score_per_cuda_warp_(                                 //
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
        char_t const *shorter_global = task.shorter.data();
        char_t const *longer_global = task.longer.data();
        u32_t const shorter_length = static_cast<u32_t>(task.shorter.size());
        u32_t const longer_length = static_cast<u32_t>(task.longer.size());
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
__global__ void affine_score_per_cuda_warp_(                                 //
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
        char_t const *shorter_global = task.shorter.data();
        char_t const *longer_global = task.longer.data();
        u32_t const shorter_length = static_cast<u32_t>(task.shorter.size());
        u32_t const longer_length = static_cast<u32_t>(task.longer.size());
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
inline static constexpr unsigned register_text_limit_k = 128;

/**
 *  @brief Max string length (chars) for the NW/SW @b thread-per-pair @b batch tier (above @ref register_text_limit_k).
 *
 *  Measured on H100: the NW/SW thread-per-pair scorer beats the tiled wavefront for every pair below ~512 (160 GCUPS
 *  @L<=256, 115 @512), then collapses as the per-thread DP row spills to local memory; the tiled wavefront is flat
 *  ~128 GCUPS at all lengths and wins from ~512 up. So the register tier (<=128) and this batch tier (<=512) both run
 *  thread-per-pair, and only pairs at/above this crossover are promoted to the tiled device wavefront. Tunable.
 */
inline static constexpr unsigned batch_text_limit_k = 512;

/**
 *  @brief Shorter-length at/above which a pair is promoted from the warp tier to the @b device (tiled) tier.
 *
 *  The warp kernel runs one warp per pair down a single anti-diagonal, so its throughput collapses as the pair
 *  grows (measured saturated GCUPS: 438 @2560², 322 @4096², 170 @8192², 54 @16384²) while the multi-warp tiled
 *  device kernel stays flat at ~1050 - a 2-19x gap. The native warp-vs-device split is purely memory-fit (a pair
 *  stays on the warp tier while its diagonal fits shared), which ignores that the tiled kernel is simply faster
 *  here. `shorter >= 4096` guarantees >=32 tile-columns (good multi-warp occupancy); below it the tiled per-tile
 *  overhead isn't amortized and the warp tier (or Myers, for <=2048 unit-cost Levenshtein) is preferable. Tunable.
 */
inline static constexpr size_t tiled_promotion_min_shorter_k = 4096;

template <typename char_type_>
struct cuda_similarity_task {
    using char_t = char_type_;

    using string_t = span<char_t const>;

    /** @brief Shorter of the two strings (scoring kernels assume shorter.size() <= longer.size()). */
    string_t shorter;
    /** @brief Longer of the two strings. */
    string_t longer;
    /** @brief Shared query of this cell's row (Peq side for Myers reuse); empty when reuse is not applicable. */
    string_t query;
    /** @brief Shared-memory bytes for this cell's DP diagonals. */
    size_t memory_requirement;
    /** @brief Flat index into the row-major results matrix. */
    size_t result_offset;
    /** @brief Second slot for symmetric self-similarity (== result_offset otherwise). */
    size_t mirror_offset;
    /** @brief The scored distance/score, written by the scoring kernels (reinterpreted as signed for NW/SW). */
    size_t result;
    /** @brief DP cell-width tier selected by the sizing pass. */
    bytes_per_cell_t bytes_per_cell;
    /** @brief Warps-per-multiprocessor scheduling tier (infinite_* marks an empty, pre-seeded cell). */
    warp_tasks_density_t density;

    constexpr cuda_similarity_task() = default;
    constexpr cuda_similarity_task(                   //
        char_t const *first_ptr, size_t first_length, //
        char_t const *second_ptr, size_t second_length) noexcept {
        bool const first_is_shorter = first_length < second_length;
        shorter = first_is_shorter ? string_t {first_ptr, first_length} : string_t {second_ptr, second_length};
        longer = first_is_shorter ? string_t {second_ptr, second_length} : string_t {first_ptr, first_length};
        // The remaining fields are filled by the caller (materialization / host loop); a 4-arg-constructed task
        // still starts fully defined here so it is never read uninitialized.
        query = {}, memory_requirement = 0, result_offset = 0, mirror_offset = 0,
        result = std::numeric_limits<size_t>::max(), bytes_per_cell = eight_bytes_per_cell_k,
        density = warps_working_together_k;
    }

    /** @brief Length of the longest anti-diagonal of this cell's DP matrix. */
    constexpr size_t max_diagonal_length() const noexcept { return sz_max_of_two(shorter.size(), longer.size()) + 1; }

    /** @brief Whether this task is small enough for the register-only thread-per-pair Levenshtein kernels. */
    constexpr bool fits_in_registers() const noexcept {
        return (bytes_per_cell == one_byte_per_cell_k || bytes_per_cell == two_bytes_per_cell_k) &&
               shorter.size() <= register_text_limit_k && longer.size() <= register_text_limit_k;
    }

    /**
     *  @brief Whether this task belongs to the NW/SW thread-per-pair @b batch tier: longer than the register limit but
     *         still below the tiled-wavefront crossover (@ref batch_text_limit_k) and within signed 2-byte cells.
     *         Disjoint from @ref fits_in_registers so a partition over each leaves the tiled remainder untouched.
     */
    constexpr bool fits_in_batch() const noexcept {
        return (bytes_per_cell == one_byte_per_cell_k || bytes_per_cell == two_bytes_per_cell_k) &&
               !fits_in_registers() && shorter.size() <= batch_text_limit_k && longer.size() <= batch_text_limit_k;
    }
};

// The task array lives in (eventually device) memory the host must not construct/destruct element-wise; it is
// filled only by the materialization kernel and resized via `try_resize_uninitialized`. Trivial destructibility
// lets `safe_vector` shrink/free it without running a host destructor over device memory.
static_assert(std::is_trivially_destructible<cuda_similarity_task<char>>::value,
              "cuda_similarity_task must be trivially destructible (device_alloc + try_resize_uninitialized).");

/**
 *  @brief Engine-owned, grow-only device buffer bundle shared by every CUDA cross-product engine (Levenshtein and
 *         the weighted NW/SW family). Allocated once and reused across calls so the host-orchestration free
 *         functions never allocate on the hot path; every CUB `DoubleBuffer` / temporary below wraps one of these
 *         hoisted members and is launched stream-async. Holding it by value in each engine and passing it
 *         @b by-reference into the shared free functions is what replaces the retired `cuda_weighted_scores`
 *         inheritance (no base class, no CRTP).
 *
 *  @tparam task_type_ The trivially-destructible @ref cuda_similarity_task instantiation (R4: `device_alloc` +
 *                     `try_resize_uninitialized` require trivial destructibility).
 */
template <typename task_type_, typename allocator_type_>
struct cuda_cross_buffers {
    using task_t = task_type_;
    using allocator_t = allocator_type_;
    using scores_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<u64_t>;
    using tasks_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<task_t>;
    using cub_buffer_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<std::byte>;
    using desc_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<span<char const>>;
    using counts_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<u32_t>;
    using warp_size_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<size_t>;
    using warp_descriptor_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<warp_tasks_group_descriptor_t>;

    allocator_t alloc_ {};

    /** @brief O(Q·C) device-resident task array, one cell per live (query, candidate) pair. */
    safe_vector<task_t, tasks_allocator_t> tasks_ {alloc_};
    /** @brief Grow-only reordered-task ping-pong (router gather target / partition selection output). */
    safe_vector<task_t, tasks_allocator_t> tasks_alt_ {alloc_};
    /** @brief Grow-only CUB temporary storage shared by every device-wide sort / scan / partition / RLE / reduce. */
    safe_vector<std::byte, cub_buffer_allocator_t> cub_buffer_ {alloc_};

    /** @brief Grow-only host-built device task-build descriptors for the query side. */
    safe_vector<span<char const>, desc_allocator_t> query_descs_ {alloc_};
    /** @brief Grow-only host-built device task-build descriptors for the candidate side. */
    safe_vector<span<char const>, desc_allocator_t> candidate_descs_ {alloc_};

    /** @brief Device-accessible 3-slot scratch the `cub::DeviceReduce::Max` device-tier shape reductions write into. */
    safe_vector<u32_t, counts_allocator_t> device_maxima_scratch_ {alloc_};
    /** @brief Grow-only task ping-pong for the warp-grouping `cub::DevicePartition::If` device/warp/empty split. */
    safe_vector<task_t, tasks_allocator_t> warp_partition_scratch_ {alloc_};
    /** @brief Device-accessible 2-slot partition selected-counts for the warp-grouping split. */
    safe_vector<u32_t, counts_allocator_t> warp_split_counts_ {alloc_};
    /** @brief Device-accessible unique composite group keys + run count from the warp-tier run-length-encode. */
    safe_vector<u32_t, counts_allocator_t> warp_group_keys_ {alloc_};
    /** @brief Device-accessible warp-group run lengths, begin offsets, and per-group max memory requirement. */
    safe_vector<size_t, warp_size_allocator_t> warp_group_sizes_ {alloc_};
    /** @brief Host-readable warp-tier launch group descriptors (one per merged group). */
    safe_vector<warp_tasks_group_descriptor_t, warp_descriptor_allocator_t> warp_group_descriptors_ {alloc_};

    /** @brief Grow-only `u64` frontier scratch for the tiled device-tier wavefront (carved per cell width). */
    safe_vector<u64_t, scores_allocator_t> diagonals_u64_buffer_ {alloc_};

    /**
     *  @brief Grow-only device-resident dense results staging for the host-output (non-device-accessible) scatter
     *         fallback: the device scatter kernel writes the full row-major matrix here, then one
     *         `cudaMemcpy2DAsync` strides it into the caller's (host) `strided_rows`. Byte-typed since the element
     *         width (`value_type_`) varies per call; never touched on the device-accessible fast path.
     */
    safe_vector<std::byte, cub_buffer_allocator_t> scatter_staging_ {alloc_};

    cuda_cross_buffers() noexcept = default;
    explicit cuda_cross_buffers(allocator_t const &alloc) noexcept : alloc_(alloc) {}

    cuda_cross_buffers(cuda_cross_buffers const &) = delete;
    cuda_cross_buffers &operator=(cuda_cross_buffers const &) = delete;
    cuda_cross_buffers(cuda_cross_buffers &&) noexcept = default;
    cuda_cross_buffers &operator=(cuda_cross_buffers &&) noexcept = default;
};

#pragma region Levenshtein Device Tier Router

/**
 *  @brief On-device replacement for the host `std::partition` / `std::sort` / `std::upper_bound` Levenshtein
 *         tiering. One `cub::DeviceRadixSort::SortPairs` over a packed `u64` key lays out every task in its final
 *         contiguous tier order; a `cub::DeviceRunLengthEncode::Encode` over a transform iterator emitting the
 *         dense tier id yields the per-tier counts. Mirrors the verified `/tmp/tier_router.cu` design.
 *
 *  Five contiguous output tiers, in final order:
 *    0 myers_word1  : unit-cost linear and shorter <=  64 (register-resident single-word Myers)
 *    1 myers_generic: unit-cost linear and shorter >   64 (size-generic Myers, any length)
 *    2 register_u8  : non-myers, fits_in_registers, one-byte cells
 *    3 register_u16 : non-myers, fits_in_registers, two-byte cells
 *    4 device_rest  : non-myers, everything else
 *
 *  Packed `u64` sort key, MSB -> LSB:
 *    tier_priority   : 3 bits  [63:61]   group order (both myers tiers collapse to 0; register/device 5/6/7)
 *    shorter_length  : 13 bits [60:48]   ascending intra-myers sub-sort (splits word1 / generic at 64; also fine elsewhere)
 *    original_index  : 32 bits [47:16]   stable tiebreaker / gather source
 *    (16 low bits unused, zero)
 */

/** @brief Below this shorter length the register-resident single-word Myers runs; above it, the size-generic Myers. */
static constexpr u32_t levenshtein_myers_word1_cap_k = 64;

static constexpr int levenshtein_original_index_bits_k = 32;
static constexpr int levenshtein_shorter_length_bits_k = 13; // 8192 > the ~6000 realistic maximum
static constexpr int levenshtein_tier_priority_bits_k = 3;

static constexpr int levenshtein_original_index_shift_k = 16;
static constexpr int levenshtein_shorter_length_shift_k = levenshtein_original_index_shift_k +
                                                          levenshtein_original_index_bits_k; // 48
static constexpr int levenshtein_tier_priority_shift_k = levenshtein_shorter_length_shift_k +
                                                         levenshtein_shorter_length_bits_k; // 61

/** @brief Group-order priority placed in the key MSBs (all myers tasks share priority 0). */
static constexpr u32_t levenshtein_priority_myers_k = 0;
static constexpr u32_t levenshtein_priority_register_u8_k = 5;
static constexpr u32_t levenshtein_priority_register_u16_k = 6;
static constexpr u32_t levenshtein_priority_device_k = 7;

/** @brief Final dense tier ids 0..4 used for run-length counting. */
static constexpr u32_t levenshtein_tier_myers_word1_k = 0;
static constexpr u32_t levenshtein_tier_myers_generic_k = 1;
static constexpr u32_t levenshtein_tier_register_u8_k = 2;
static constexpr u32_t levenshtein_tier_register_u16_k = 3;
static constexpr u32_t levenshtein_tier_device_k = 4;
static constexpr int levenshtein_tier_count_k = 5;

/**
 *  @brief Whether a task routes to the bit-parallel Myers tiers. Only unit-cost linear Levenshtein can; affine and
 *         non-unit-cost linear skip Myers entirely and fall through to the register/device split (exactly as the host
 *         code did with `myers_count == 0`). Every unit-cost linear task takes a Myers tier regardless of size: short
 *         (`shorter <= 64`) on the register-resident single-word kernel, the rest on the size-generic kernel.
 */
enum class levenshtein_tier_mode_t {
    /** @brief Unit-cost linear: shorter <= 64 -> register Myers, shorter > 64 -> generic Myers; never register/device. */
    myers_and_registers_k,
    /** @brief Affine or non-unit-cost linear: no Myers; every task -> register / device split. */
    registers_only_k,
};

/** @brief Group-ordering priority placed in the key MSBs for one task. */
template <typename char_type_>
__host__ __device__ __forceinline__ u32_t levenshtein_task_tier_priority(cuda_similarity_task<char_type_> const &task,
                                                                         levenshtein_tier_mode_t mode) noexcept {
    if (mode == levenshtein_tier_mode_t::myers_and_registers_k) return levenshtein_priority_myers_k;
    if (task.fits_in_registers())
        return task.bytes_per_cell == one_byte_per_cell_k ? levenshtein_priority_register_u8_k
                                                          : levenshtein_priority_register_u16_k;
    return levenshtein_priority_device_k;
}

/** @brief Dense final tier id 0..4, including the Myers word1 / generic split at 64, for one task. */
template <typename char_type_>
__host__ __device__ __forceinline__ u32_t levenshtein_task_dense_tier(cuda_similarity_task<char_type_> const &task,
                                                                      levenshtein_tier_mode_t mode) noexcept {
    if (mode == levenshtein_tier_mode_t::myers_and_registers_k)
        return task.shorter.size() <= levenshtein_myers_word1_cap_k ? levenshtein_tier_myers_word1_k
                                                                    : levenshtein_tier_myers_generic_k;
    if (task.fits_in_registers())
        return task.bytes_per_cell == one_byte_per_cell_k ? levenshtein_tier_register_u8_k
                                                          : levenshtein_tier_register_u16_k;
    return levenshtein_tier_device_k;
}

/**
 *  @brief Dyadic length bucket `bit_width(length - 1)` for one task, mirroring the CPU `candidate_length_bucket_`.
 *         Two lengths in one bucket differ by less than 2x, so radix-sorting on this field (instead of the raw
 *         length) groups tasks into power-of-two length bands within a priority tier -> uniform-depth launches.
 *         Host-portable: uses `__clzll` on device and the `sz_u64_clz` SWAR/intrinsic wrapper on the host.
 */
__host__ __device__ __forceinline__ u64_t levenshtein_length_dyadic_bucket(u64_t length) noexcept {
    if (length <= 1) return 0;
#ifdef __CUDA_ARCH__
    return static_cast<u64_t>(64 - __clzll(static_cast<unsigned long long>(length - 1)));
#else
    return static_cast<u64_t>(64 - sz_u64_clz(static_cast<sz_u64_t>(length - 1)));
#endif
}

/** @brief Packs one task into the MSB-ordered radix-sort key described above. */
template <typename char_type_>
__host__ __device__ __forceinline__ u64_t levenshtein_pack_tier_key(cuda_similarity_task<char_type_> const &task,
                                                                    u32_t original_index,
                                                                    levenshtein_tier_mode_t mode) noexcept {
    u64_t const priority = levenshtein_task_tier_priority(task, mode);
    // Bucket the length dyadically (`bit_width(L - 1)`) instead of using the raw length, so the ascending sub-sort
    // groups tasks into power-of-two length bands - uniform launch depth - while staying monotonic. The bucket is
    // already saturated by construction (it never exceeds ~bit_width(SZ_SIZE_MAX), far under the field width), and
    // the size-generic Myers tier still sorts above the `<= 64` word1 tasks.
    u64_t const shorter_max = (1ull << levenshtein_shorter_length_bits_k) - 1ull;
    u64_t const shorter = sz_min_of_two(levenshtein_length_dyadic_bucket(static_cast<u64_t>(task.shorter.size())),
                                        shorter_max);
    u64_t const index = original_index;
    u64_t key = 0;
    key |= priority << levenshtein_tier_priority_shift_k;
    key |= shorter << levenshtein_shorter_length_shift_k;
    key |= index << levenshtein_original_index_shift_k;
    return key;
}

/** @brief Builds the packed sort key for every task and seeds the identity permutation values. */
template <typename char_type_>
__global__ void levenshtein_build_tier_keys_kernel(cuda_similarity_task<char_type_> const *tasks, size_t count,
                                                   levenshtein_tier_mode_t mode, u64_t *keys, u32_t *task_indices) {
    size_t const stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += stride) {
        keys[index] = levenshtein_pack_tier_key(tasks[index], static_cast<u32_t>(index), mode);
        task_indices[index] = static_cast<u32_t>(index);
    }
}

/** @brief Materializes the reordered task array from the sorted permutation of indices. */
template <typename char_type_>
__global__ void levenshtein_gather_tasks_kernel(cuda_similarity_task<char_type_> const *tasks,
                                                u32_t const *sorted_indices, size_t count,
                                                cuda_similarity_task<char_type_> *reordered) {
    size_t const stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += stride)
        reordered[index] = tasks[sorted_indices[index]];
}

/**
 *  @brief Reads the dense tier id 0..7 straight from each reordered task. Driving the run-length encode off the
 *         reordered tasks (not the key) lets the single ascending Myers sub-sort yield the 64/128/256/512 word
 *         boundaries without a second sort.
 */
template <typename char_type_>
struct levenshtein_dense_tier_functor {
    cuda_similarity_task<char_type_> const *tasks;
    levenshtein_tier_mode_t mode;
    __host__ __device__ __forceinline__ u32_t operator()(size_t index) const {
        return levenshtein_task_dense_tier(tasks[index], mode);
    }
};

/**
 *  @brief On-device dyadic length-bucket router shared by the Levenshtein and weighted NW/SW GPU engines. Lays
 *         `buffers.tasks_` out in contiguous tier order with one `cub::DeviceRadixSort::SortPairs` over a packed
 *         `u64` key, gathers the sorted permutation back into `buffers.tasks_`, and run-length-encodes the dense
 *         tier id of the reordered tasks into the host-readable @p tier_counts. A free function taking the engine's
 *         `cuda_cross_buffers` plus its grow-only key / value / RLE scratch @b by-reference (the no-inheritance
 *         substrate); the caller supplies its own packed-key build kernel (@p build_keys_shape, invoked with @p mode)
 *         and dense-tier functor (@p dense_tier_functor), so the Lev five-tier Myers/register/device taxonomy and the
 *         weighted three-tier register/batch/device taxonomy share one machinery without sharing tier semantics.
 *
 *  @param[in,out] buffers           Cross-product bundle; `tasks_` is reordered in place, `tasks_alt_`/`cub_buffer_` reused.
 *  @param[in,out] keys_scratch      Grow-only packed `u64` keys (DoubleBuffer A/B halves `[0,count)` / `[count,2*count)`).
 *  @param[in,out] values_scratch    Grow-only `u32` permutation values (same DoubleBuffer A/B layout as the keys).
 *  @param[in,out] rle_scratch       Grow-only `u32` RLE outputs: unique ids + run lengths + run count (`2*tier_count+1`).
 *  @param[in]     build_keys_shape  Engine-owned kernel packing one key + identity index per task; called with @p mode.
 *  @param[in]     gather_shape      Kernel gathering the sorted permutation into the reordered array (generic).
 *  @param[in]     mode             Opaque tier-mode value forwarded to the build-keys kernel (engine-specific).
 *  @param[in]     dense_tier_functor  Host-constructed functor mapping a reordered task index to its dense tier id.
 *  @param[in]     begin_bit/end_bit   Informative radix-sort bit range of the packed key.
 *  @param[in]     tier_count       Number of dense tiers (slots written into @p tier_counts).
 *  @param[out]    tier_counts      Host-readable dense per-tier counts (empty tiers expand to zero).
 */
template <typename buffers_type_, typename keys_scratch_type_, typename values_scratch_type_,
          typename rle_scratch_type_, typename mode_type_, typename dense_tier_functor_type_>
cuda_status_t cuda_route_tasks_into_tiers_(
    buffers_type_ &buffers, //
    keys_scratch_type_ &keys_scratch, values_scratch_type_ &values_scratch, rle_scratch_type_ &rle_scratch, //
    kernel_shape_t const &build_keys_shape, kernel_shape_t const &gather_shape, mode_type_ mode, //
    dense_tier_functor_type_ dense_tier_functor, int begin_bit, int end_bit, int tier_count, //
    cuda_executor_t const &executor, size_t *tier_counts) noexcept {

    using task_t = typename buffers_type_::task_t;
    size_t const count = buffers.tasks_.size();
    for (int tier = 0; tier < tier_count; ++tier) tier_counts[tier] = 0;
    if (!count) return {status_t::success_k, cudaSuccess};

    // Grow the device scratch: keys/values each need 2*count (DoubleBuffer A/B), the reordered task array needs
    // count, and the RLE outputs need a unique-id + run-length slot per tier plus one run-count slot.
    if (keys_scratch.try_resize_uninitialized(2 * count) == status_t::bad_alloc_k ||
        values_scratch.try_resize_uninitialized(2 * count) == status_t::bad_alloc_k ||
        buffers.tasks_alt_.try_resize_uninitialized(count) == status_t::bad_alloc_k ||
        rle_scratch.try_resize_uninitialized(2 * tier_count + 1) == status_t::bad_alloc_k)
        return {status_t::bad_alloc_k};

    u64_t *const keys_a = keys_scratch.data();
    u64_t *const keys_b = keys_a + count;
    u32_t *const values_a = values_scratch.data();
    u32_t *const values_b = values_a + count;
    u32_t *const unique_tiers = rle_scratch.data();
    u32_t *const tier_run_lengths = unique_tiers + tier_count;
    u32_t *const run_count = tier_run_lengths + tier_count;

    int const block = 256;
    unsigned const grid = static_cast<unsigned>(sz_min_of_two((count + block - 1) / block, size_t(1024)));

    // Build the packed sort keys + identity permutation via the engine-owned build-keys kernel.
    {
        task_t const *tasks_ptr = buffers.tasks_.data();
        size_t count_arg = count;
        mode_type_ mode_arg = mode;
        u64_t *keys_ptr = keys_a;
        u32_t *values_ptr = values_a;
        void *args[5] = {(void *)&tasks_ptr, (void *)&count_arg, (void *)&mode_arg, (void *)&keys_ptr,
                         (void *)&values_ptr};
        CUresult launch_error = cuda_launch_t {}
                                    .grid(grid)
                                    .block(block)
                                    .shared(0)
                                    .stream(executor.stream())
                                    .launch(build_keys_shape.function, args);
        if (launch_error != CUDA_SUCCESS) return make_cuda_status(launch_error);
    }

    // One radix sort over the only informative bits: [begin_bit, end_bit).
    cub::DoubleBuffer<u64_t> keys_buffer(keys_a, keys_b);
    cub::DoubleBuffer<u32_t> values_buffer(values_a, values_b);
    {
        size_t sort_bytes = 0;
        if (cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, keys_buffer, values_buffer, static_cast<int>(count),
                                            begin_bit, end_bit, executor.stream()) != cudaSuccess)
            return make_cuda_status(cudaGetLastError());
        if (buffers.cub_buffer_.try_resize_uninitialized(sort_bytes) == status_t::bad_alloc_k)
            return {status_t::bad_alloc_k};
        if (cub::DeviceRadixSort::SortPairs(buffers.cub_buffer_.data(), sort_bytes, keys_buffer, values_buffer,
                                            static_cast<int>(count), begin_bit, end_bit,
                                            executor.stream()) != cudaSuccess)
            return make_cuda_status(cudaGetLastError());
    }

    // Gather the reordered tasks via the sorted permutation, then swap the layout into `tasks_`.
    {
        task_t const *tasks_ptr = buffers.tasks_.data();
        u32_t const *sorted_indices = values_buffer.Current();
        size_t count_arg = count;
        task_t *reordered_ptr = buffers.tasks_alt_.data();
        void *args[4] = {(void *)&tasks_ptr, (void *)&sorted_indices, (void *)&count_arg, (void *)&reordered_ptr};
        CUresult launch_error = cuda_launch_t {}
                                    .grid(grid)
                                    .block(block)
                                    .shared(0)
                                    .stream(executor.stream())
                                    .launch(gather_shape.function, args);
        if (launch_error != CUDA_SUCCESS) return make_cuda_status(launch_error);
    }
    std::swap(buffers.tasks_, buffers.tasks_alt_);

    // One run-length encode over the dense tier id of the (now reordered) tasks.
    dense_tier_functor.tasks = buffers.tasks_.data();
    cub::CountingInputIterator<size_t> counting_iterator(0);
    cub::TransformInputIterator<u32_t, dense_tier_functor_type_, cub::CountingInputIterator<size_t>> tier_iterator(
        counting_iterator, dense_tier_functor);
    {
        size_t rle_bytes = 0;
        if (cub::DeviceRunLengthEncode::Encode(nullptr, rle_bytes, tier_iterator, unique_tiers, tier_run_lengths,
                                               run_count, static_cast<int>(count), executor.stream()) != cudaSuccess)
            return make_cuda_status(cudaGetLastError());
        if (buffers.cub_buffer_.try_resize_uninitialized(rle_bytes) == status_t::bad_alloc_k)
            return {status_t::bad_alloc_k};
        if (cub::DeviceRunLengthEncode::Encode(buffers.cub_buffer_.data(), rle_bytes, tier_iterator, unique_tiers,
                                               tier_run_lengths, run_count, static_cast<int>(count),
                                               executor.stream()) != cudaSuccess)
            return make_cuda_status(cudaGetLastError());
    }

    // The tasks live in unified memory and the RLE outputs are engine-owned unified buffers; synchronize the
    // stream, then expand the (empty-tier-omitting) runs into the dense host count array.
    if (cudaStreamSynchronize(executor.stream()) != cudaSuccess) return make_cuda_status(cudaGetLastError());
    u32_t const host_run_count = *run_count;
    for (u32_t run = 0; run < host_run_count; ++run) tier_counts[unique_tiers[run]] = tier_run_lengths[run];
    return {status_t::success_k, cudaSuccess};
}

#pragma endregion Levenshtein Device Tier Router

#pragma region Device Tier Shape Maxima

/**
 *  @brief Transform-iterator extractors that read one shape field of a device-resident task so the device-tier
 *         frontier sizing (`max_shorter`, `max_longer`, widest `bytes_per_cell`) can be computed with
 *         `cub::DeviceReduce::Max` instead of a host scan over the device task array. Iterating directly over
 *         `cuda_similarity_task const *` makes the functor argument a task, never a host dereference.
 */
template <typename char_type_>
struct task_shorter_length_functor {
    __host__ __device__ __forceinline__ u32_t operator()(cuda_similarity_task<char_type_> const &task) const {
        return static_cast<u32_t>(task.shorter.size());
    }
};

template <typename char_type_>
struct task_longer_length_functor {
    __host__ __device__ __forceinline__ u32_t operator()(cuda_similarity_task<char_type_> const &task) const {
        return static_cast<u32_t>(task.longer.size());
    }
};

template <typename char_type_>
struct task_bytes_per_cell_functor {
    __host__ __device__ __forceinline__ u32_t operator()(cuda_similarity_task<char_type_> const &task) const {
        return static_cast<u32_t>(task.bytes_per_cell);
    }
};

/**
 *  @brief Per-task shape maxima a device tier needs to size its batched frontier: the longest shorter/longer side
 *         and the widest cell type across the whole device-promoted subspan. Replaces the former host scan loops.
 */
struct device_tier_maxima_t {
    u32_t max_shorter = 0;
    u32_t max_longer = 0;
    u32_t max_bytes_per_cell = 0;
};

/**
 *  @brief Computes @ref device_tier_maxima_t over a device-resident task subspan with three `cub::DeviceReduce::Max`
 *         passes (one per shape field) over transform iterators, copying only the three small maxima back to the
 *         host. @p cub_temp is grown in place and reused; @p maxima_scratch holds the three device-side reduction
 *         outputs (engine-owned, device-accessible). The stream is synchronized before the host reads the results.
 */
template <typename char_type_, typename cub_temp_type_, typename maxima_scratch_type_>
cuda_status_t reduce_device_tier_maxima_(cuda_similarity_task<char_type_> const *tasks, size_t count,
                                         cub_temp_type_ &cub_temp, maxima_scratch_type_ &maxima_scratch,
                                         cudaStream_t stream, device_tier_maxima_t &maxima) noexcept {
    maxima = {};
    if (!count) return {status_t::success_k, cudaSuccess};
    if (maxima_scratch.try_resize_uninitialized(3) == status_t::bad_alloc_k) return {status_t::bad_alloc_k};
    u32_t *const max_shorter_out = maxima_scratch.data();
    u32_t *const max_longer_out = max_shorter_out + 1;
    u32_t *const max_bytes_per_cell_out = max_shorter_out + 2;

    using task_t = cuda_similarity_task<char_type_>;
    cub::TransformInputIterator<u32_t, task_shorter_length_functor<char_type_>, task_t const *> shorter_iterator(
        tasks, task_shorter_length_functor<char_type_> {});
    cub::TransformInputIterator<u32_t, task_longer_length_functor<char_type_>, task_t const *> longer_iterator(
        tasks, task_longer_length_functor<char_type_> {});
    cub::TransformInputIterator<u32_t, task_bytes_per_cell_functor<char_type_>, task_t const *> bytes_per_cell_iterator(
        tasks, task_bytes_per_cell_functor<char_type_> {});

    auto const reduce_max = [&](auto input_iterator, u32_t *output) -> cuda_status_t {
        size_t reduce_bytes = 0;
        if (cub::DeviceReduce::Max(nullptr, reduce_bytes, input_iterator, output, static_cast<int>(count), stream) !=
            cudaSuccess)
            return make_cuda_status(cudaGetLastError());
        if (cub_temp.try_resize_uninitialized(reduce_bytes) == status_t::bad_alloc_k) return {status_t::bad_alloc_k};
        if (cub::DeviceReduce::Max(cub_temp.data(), reduce_bytes, input_iterator, output, static_cast<int>(count),
                                   stream) != cudaSuccess)
            return make_cuda_status(cudaGetLastError());
        return {status_t::success_k, cudaSuccess};
    };
    cuda_status_t status = reduce_max(shorter_iterator, max_shorter_out);
    if (status.status != status_t::success_k) return status;
    status = reduce_max(longer_iterator, max_longer_out);
    if (status.status != status_t::success_k) return status;
    status = reduce_max(bytes_per_cell_iterator, max_bytes_per_cell_out);
    if (status.status != status_t::success_k) return status;

    if (cudaStreamSynchronize(stream) != cudaSuccess) return make_cuda_status(cudaGetLastError());
    maxima.max_shorter = *max_shorter_out;
    maxima.max_longer = *max_longer_out;
    maxima.max_bytes_per_cell = *max_bytes_per_cell_out;
    return {status_t::success_k, cudaSuccess};
}

#pragma endregion Device Tier Shape Maxima

/**
 *  @brief Device sibling of `warp_tasks_density` (types.cuh) so the materialization kernel sizes each cell with
 *         the same predicate as the host. `gpu_specs_t::shared_memory_per_multiprocessor()` is host-only, so the
 *         shared-per-SM divide is inlined here from the POD `specs` fields.
 */
__forceinline__ __device__ warp_tasks_density_t warp_tasks_density_device_(size_t task_memory_requirement,
                                                                           gpu_specs_t const &specs) noexcept {
    warp_tasks_density_t const densities[] {
        sixty_four_warps_per_multiprocessor_k, thirty_two_warps_per_multiprocessor_k,
        sixteen_warps_per_multiprocessor_k,    eight_warps_per_multiprocessor_k,
        four_warps_per_multiprocessor_k,       two_warps_per_multiprocessor_k,
        one_warp_per_multiprocessor_k,
    };
    if (task_memory_requirement == 0) return infinite_warps_per_multiprocessor_k;
    size_t const shared_per_multiprocessor = specs.shared_memory_bytes / specs.streaming_multiprocessors;
    for (warp_tasks_density_t density : densities) {
        if (density > specs.max_blocks_per_multiprocessor) continue;
        size_t required_block_memory = task_memory_requirement * density + specs.reserved_memory_per_block * density;
        if (required_block_memory < shared_per_multiprocessor) return density;
    }
    return warps_working_together_k;
}

/**
 *  @brief Materializes the per-cell `cuda_similarity_task` array of a cross-product entirely on the device (one
 *         thread per live cell) for @b all-pairs OR @b symmetric (lower-triangle) shapes, and for any engine
 *         family - uniform or weighted, linear or affine, global or local. Replaces the O(queries*candidates)
 *         host loop with an O(queries+candidates) host descriptor build; per-cell sizing/tiering mirrors the host
 *         path exactly. For symmetric shapes the flat cell index is mapped to a lower-triangle (row, column) and
 *         the result is mirrored on write.
 */
template <sz_similarity_objective_t objective_, sz_similarity_locality_t locality_, bool is_affine_,
          typename task_type_, typename gap_costs_type_>
__global__ void materialize_similarity_tasks_(                                         //
    task_type_ *tasks,                                                                 //
    span<char const> const *queries, span<char const> const *candidates,           //
    size_t queries_count, size_t candidates_count, size_t row_stride,                  //
    cross_similarities_t cross_kind,                                                   //
    error_cost_magnitude_t substitute_magnitude, error_cost_magnitude_t gap_magnitude, //
    sz_similarity_gaps_t gap_type_value, bytes_per_cell_t min_bytes_per_cell,          //
    gap_costs_type_ gap_costs, gpu_specs_t specs) {

    using score_t = typename std::conditional<objective_ == sz_minimize_distance_k, size_t, ssize_t>::type;
    constexpr bool is_local_k = locality_ == sz_similarity_local_k;
    bool const is_symmetric = cross_kind == cross_similarities_t::symmetric_k;

    size_t const total_cells = is_symmetric ? queries_count * (queries_count + 1) / 2
                                            : queries_count * candidates_count;
    size_t const cell_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell_index >= total_cells) return;

    size_t query_index, candidate_index;
    if (is_symmetric) {
        // Lower triangle (incl. diagonal), filled row-major: recover (row, column) from the flat cell index via a
        // double-precision seed + integer fixup (sqrtf is too lossy for large cell indices).
        double const discriminant = 8.0 * static_cast<double>(cell_index) + 1.0;
        size_t row = static_cast<size_t>((sqrt(discriminant) - 1.0) * 0.5);
        while (row * (row + 1) / 2 > cell_index) row -= 1;
        while ((row + 1) * (row + 2) / 2 <= cell_index) row += 1;
        query_index = row;
        candidate_index = cell_index - row * (row + 1) / 2;
    }
    else {
        query_index = cell_index / candidates_count;
        candidate_index = cell_index % candidates_count;
    }
    span<char const> const query = queries[query_index];
    span<char const> const candidate = candidates[candidate_index];

    task_type_ task(query.data(), query.size(), candidate.data(), candidate.size());
    diagonal_memory_requirements<score_t> const requirement(task.shorter.size(), task.longer.size(), gap_type_value,
                                                            substitute_magnitude, gap_magnitude, sizeof(char), 4,
                                                            min_bytes_per_cell);
    task.result_offset = query_index * row_stride + candidate_index;
    task.mirror_offset = is_symmetric ? candidate_index * row_stride + query_index : task.result_offset;
    task.query = query;
    task.memory_requirement = requirement.bytes_for_diagonals;
    task.bytes_per_cell = requirement.bytes_per_cell;
    task.density = warp_tasks_density_device_(requirement.bytes_for_diagonals, specs);
    if (task.density != infinite_warps_per_multiprocessor_k && task.shorter.size() >= tiled_promotion_min_shorter_k)
        task.density = warps_working_together_k;
    if (task.density == infinite_warps_per_multiprocessor_k) {
        if constexpr (is_local_k) { task.result = 0; }
        else if constexpr (!is_affine_) { task.result = task.longer.size() * gap_costs.open_or_extend; }
        else if (!task.longer.size()) { task.result = 0; }
        else { task.result = (task.longer.size() - 1) * gap_costs.extend + gap_costs.open; }
    }
    tasks[cell_index] = task;
}

/**
 *  @brief Scatters each task's result into the row-major matrix on the device (one thread per task), so the host
 *         never reads the (large, GPU-resident) task array back - avoiding a full unified-memory page migration.
 */
template <typename task_type_, typename value_type_>
__global__ void scatter_similarity_results_(task_type_ const *tasks, size_t tasks_count, value_type_ *results) {
    size_t const task_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (task_index >= tasks_count) return;
    task_type_ const &task = tasks[task_index];
    value_type_ const value = static_cast<value_type_>(task.result);
    results[task.result_offset] = value;
    if (task.mirror_offset != task.result_offset) results[task.mirror_offset] = value;
}

/**
 *  @brief Host-output (non-device-accessible) scatter fallback shared by every CUDA cross-product engine: instead of a
 *         per-cell host loop, the device @ref scatter_similarity_results_ kernel writes the full row-major matrix into a
 *         hoisted device-resident staging buffer (laid out at the caller's `row_stride`, so each task's precomputed
 *         `result_offset` / `mirror_offset` stays valid), then a single strided `cudaMemcpy2DAsync` copies the valid
 *         `rows x columns` region into the caller's host `strided_rows`. Fully stream-async; staging grows-and-reuses.
 *
 *  @param buffers Engine buffer bundle holding the grow-only `scatter_staging_` device buffer (passed by reference).
 *  @param tasks Device-resident task array (already scored); read-only.
 *  @param tasks_count Number of live tasks.
 *  @param results The caller's host-side strided output matrix.
 */
template <typename task_type_, typename value_type_, typename buffers_task_type_, typename allocator_type_>
cuda_status_t cuda_scatter_results_to_host_strided_(
    cuda_cross_buffers<buffers_task_type_, allocator_type_> &buffers, task_type_ const *tasks, size_t tasks_count,
    strided_rows<value_type_> const &results, cuda_executor_t const &executor, unsigned block) noexcept {

    if (!tasks_count) return {status_t::success_k, cudaSuccess};

    // Size the dense staging matrix at the caller's `row_stride` (not the tighter `columns`) so the device kernel's
    // precomputed `result_offset = query_index * row_stride + candidate_index` indexes it without any remapping.
    size_t const staging_elements = results.rows * results.row_stride;
    if (buffers.scatter_staging_.try_resize_uninitialized(staging_elements * sizeof(value_type_)) ==
        status_t::bad_alloc_k)
        return {status_t::bad_alloc_k};
    value_type_ *const staging_ptr = static_cast<value_type_ *>(static_cast<void *>(buffers.scatter_staging_.data()));

    kernel_shape_t scatter_shape;
    cuda_status_t const scatter_resolve = resolve_kernel_shape(
        scatter_shape, (void const *)&scatter_similarity_results_<task_type_, value_type_>, 256, 0, false);
    if (scatter_resolve.status != status_t::success_k) return scatter_resolve;

    task_type_ const *tasks_ptr = tasks;
    value_type_ *results_ptr = staging_ptr;
    size_t tasks_size = tasks_count;
    void *scatter_args[3] = {(void *)&tasks_ptr, (void *)&tasks_size, (void *)&results_ptr};
    unsigned const scatter_grid = static_cast<unsigned>((tasks_size + block - 1) / block);
    CUresult const scatter_error = cuda_launch_t {}
                                       .grid(scatter_grid)
                                       .block(block)
                                       .shared(0)
                                       .stream(executor.stream())
                                       .launch(scatter_shape.function, scatter_args);
    if (scatter_error != CUDA_SUCCESS) return make_cuda_status(scatter_error);

    // Strided copy: only the valid `columns`-wide prefix of each of the `rows` rows is transferred; the padding
    // between `columns` and `row_stride` is skipped on both sides (matching the per-cell host loop's behavior).
    size_t const valid_row_bytes = results.columns * sizeof(value_type_);
    size_t const stride_bytes = results.row_stride * sizeof(value_type_);
    if (cudaMemcpy2DAsync(results.data, stride_bytes, staging_ptr, stride_bytes, valid_row_bytes, results.rows,
                          cudaMemcpyDeviceToHost, executor.stream()) != cudaSuccess)
        return make_cuda_status(cudaGetLastError());
    if (cudaStreamSynchronize(executor.stream()) != cudaSuccess) return make_cuda_status(cudaGetLastError());
    return {status_t::success_k, cudaSuccess};
}

/**
 *  @brief Byte-wise SIMD helpers (4× `u8_t` packed in a `u32_t`) for the register-only Levenshtein kernel.
 *         On the device they map to the `__vcmpeq4`/`__vminu4`/`__vaddus4`/`__byte_perm` video instructions; the
 *         host fallbacks keep the kernel unit-testable on the CPU.
 */
__forceinline__ __device__ __host__ u32_t sz_u32_vcmpeq4_(u32_t a, u32_t b) noexcept {
#ifdef __CUDA_ARCH__
    return __vcmpeq4(a, b);
#else
    u32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        u8_t byte_a = (a >> (i * 8)) & 0xFF, byte_b = (b >> (i * 8)) & 0xFF;
        if (byte_a == byte_b) result |= 0xFFu << (i * 8);
    }
    return result;
#endif
}

__forceinline__ __device__ __host__ u32_t sz_u32_vminu4_(u32_t a, u32_t b) noexcept {
#ifdef __CUDA_ARCH__
    return __vminu4(a, b);
#else
    u32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        u8_t byte_a = (a >> (i * 8)) & 0xFF, byte_b = (b >> (i * 8)) & 0xFF;
        result |= (u32_t)(byte_a < byte_b ? byte_a : byte_b) << (i * 8);
    }
    return result;
#endif
}

__forceinline__ __device__ __host__ u32_t sz_u32_vaddus4_(u32_t a, u32_t b) noexcept {
#ifdef __CUDA_ARCH__
    return __vaddus4(a, b);
#else
    u32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        u32_t sum = ((a >> (i * 8)) & 0xFF) + ((b >> (i * 8)) & 0xFF);
        result |= (u32_t)(sum > 0xFF ? 0xFF : sum) << (i * 8);
    }
    return result;
#endif
}

__forceinline__ __device__ __host__ u32_t sz_u32_byte_perm_(u32_t x, u32_t y, u32_t selector) noexcept {
#ifdef __CUDA_ARCH__
    return __byte_perm(x, y, selector);
#else
    u8_t source[8];
    for (int i = 0; i < 4; ++i) source[i] = (x >> (i * 8)) & 0xFF, source[i + 4] = (y >> (i * 8)) & 0xFF;
    u32_t result = 0;
    for (int i = 0; i < 4; ++i) result |= (u32_t)source[(selector >> (i * 4)) & 0x7] << (i * 8);
    return result;
#endif
}

/** @brief Broadcasts a byte-wide cost into all four lanes of a packed `u32_t`. */
__forceinline__ __device__ __host__ u32_t broadcast_cost_u8x4_(u8_t value) noexcept {
    return (u32_t)value * 0x01010101u;
}

/** @brief Broadcasts a 16-bit cost into both lanes of a packed `u32_t`. */
__forceinline__ __device__ __host__ u32_t broadcast_cost_u16x2_(u16_t value) noexcept {
    return (u32_t)value | ((u32_t)value << 16);
}

/**
 *  @brief Fills a packed DP row with the leading `(column index + 1) * gap` ladder, shared by the register kernels.
 *  @param[out] packed_row The row of packed cells, @p lanes_per_pack cells of @p bits_per_lane bits per `u32_t`.
 */
__forceinline__ __device__ __host__ void fill_gap_ladder_(u32_t *packed_row, unsigned pack_count,
                                                          unsigned lanes_per_pack, unsigned bits_per_lane,
                                                          error_cost_t gap_cost) noexcept {
    unsigned gap_ladder = gap_cost;
    for (unsigned pack = 0; pack < pack_count; ++pack) {
        u32_t value = 0;
        for (unsigned lane = 0; lane < lanes_per_pack; ++lane) {
            value |= (u32_t)gap_ladder << (lane * bits_per_lane);
            gap_ladder += gap_cost;
        }
        packed_row[pack] = value;
    }
}

/**
 *  @brief Register-only Levenshtein distance for strings up to @p max_text_length_ bytes, one thread per pair.
 *
 *  Wagner-Fischer with a single DP row kept entirely in registers, the (longer) string cached in registers, and
 *  cells stored as `u8_t` packed 4-per-`u32_t` so each video instruction advances four columns at once.
 *  No shared memory and no `__syncwarp` - ideal for short inputs where the anti-diagonal warp kernel starves the
 *  warp. Saturates at 255, so callers must gate on @b `fits_in_registers` (≤1-byte cells, lengths ≤ the limit).
 */
template <unsigned max_text_length_>
struct register_levenshtein {
    static constexpr unsigned max_text_length_k = max_text_length_;
    static constexpr unsigned pack_count_k = max_text_length_k / sizeof(u32_vec_t);

    // `__byte_perm` lane selectors (one nibble per result byte; sources 0..3 are the first operand's bytes,
    // 4..7 the second operand's). Used to build the diagonal and to run the packed left-dependency prefix scan.
    static constexpr unsigned diagonal_selector_k = 0x6543;    // diag = {previous_row[3], top[0], top[1], top[2]}
    static constexpr unsigned left_carry_selector_k = 0x6540;  // {left_cell, cell[0], cell[1], cell[2]}
    static constexpr unsigned left_scan_selector_1_k = 0x2100; // {cell[0], cell[0], cell[1], cell[2]}
    static constexpr unsigned left_scan_selector_2_k = 0x2110; // {cell[0], cell[1], cell[1], cell[2]}
    static constexpr unsigned left_scan_selector_3_k = 0x2221; // {cell[1], cell[2], cell[2], cell[2]}

    u32_vec_t row_cells_[pack_count_k];
    u32_vec_t longer_chars_[pack_count_k];

    __forceinline__ __device__ __host__ u8_t operator()(     //
        u8_t const *longer_string, unsigned longer_length,   //
        u8_t const *shorter_string, unsigned shorter_length, //
        uniform_substitution_costs_t const substituter, linear_gap_costs_t const gap_costs) noexcept {

        error_cost_t const gap_cost = gap_costs.open_or_extend;

        // Initialize the first row with the (column index + 1) * gap ladder, four cells per pack.
        fill_gap_ladder_(reinterpret_cast<u32_t *>(row_cells_), pack_count_k, 4, 8, gap_cost);

        // Cache the longer string in registers (packed), accessed in the inner loop.
        for (unsigned i = 0; i < longer_length; ++i) longer_chars_[0].u8s[i] = longer_string[i];

        u32_t const gap_cost_vec = broadcast_cost_u8x4_(gap_cost);
        u32_t const match_cost_vec = broadcast_cost_u8x4_(substituter.match);
        u32_t const mismatch_cost_vec = broadcast_cost_u8x4_(substituter.mismatch);

        // Outer loop over the shorter string (fewer iterations).
        for (unsigned row_idx = 1; row_idx <= shorter_length; ++row_idx) {
            u8_t const shorter_char = shorter_string[row_idx - 1];
            u32_t const shorter_char_vec = broadcast_cost_u8x4_(shorter_char);
            u8_t const first_col_current = row_idx * gap_cost;
            u8_t const first_col_previous = (row_idx - 1) * gap_cost;
            u32_t previous_row_vec = broadcast_cost_u8x4_(first_col_previous);

            // Inner loop over the longer string, four columns per pack.
            for (unsigned pack_idx = 0; pack_idx < pack_count_k; ++pack_idx) {
                u32_t const top_vec = row_cells_[pack_idx].u32;
                u32_t const diagonal_vec = sz_u32_byte_perm_(previous_row_vec, top_vec, diagonal_selector_k);
                u32_t const match_mask_vec = sz_u32_vcmpeq4_(shorter_char_vec, longer_chars_[pack_idx].u32);
                u32_t const cost_of_substitution_vec = (match_cost_vec & match_mask_vec) |
                                                       (mismatch_cost_vec & ~match_mask_vec);
                u32_t const cost_if_substitution_vec = sz_u32_vaddus4_(diagonal_vec, cost_of_substitution_vec);
                u32_t const cost_if_top_gap_vec = sz_u32_vaddus4_(top_vec, gap_cost_vec);
                u32_t cell_score_vec = sz_u32_vminu4_(cost_if_substitution_vec, cost_if_top_gap_vec);

                // Propagate the left dependency across the four packed cells (sequential prefix scan).
                u8_t const left_cell = (pack_idx == 0) ? first_col_current : (row_cells_[pack_idx - 1].u32 >> 24);
                u32_t cost_if_left_gap_vec = sz_u32_byte_perm_(broadcast_cost_u8x4_(left_cell), cell_score_vec,
                                                               left_carry_selector_k);
                cell_score_vec = sz_u32_vminu4_(cell_score_vec, sz_u32_vaddus4_(cost_if_left_gap_vec, gap_cost_vec));
                cost_if_left_gap_vec = sz_u32_byte_perm_(cell_score_vec, cell_score_vec, left_scan_selector_1_k);
                cell_score_vec = sz_u32_vminu4_(cell_score_vec, sz_u32_vaddus4_(cost_if_left_gap_vec, gap_cost_vec));
                cost_if_left_gap_vec = sz_u32_byte_perm_(cell_score_vec, cell_score_vec, left_scan_selector_2_k);
                cell_score_vec = sz_u32_vminu4_(cell_score_vec, sz_u32_vaddus4_(cost_if_left_gap_vec, gap_cost_vec));
                cost_if_left_gap_vec = sz_u32_byte_perm_(cell_score_vec, cell_score_vec, left_scan_selector_3_k);
                cell_score_vec = sz_u32_vminu4_(cell_score_vec, sz_u32_vaddus4_(cost_if_left_gap_vec, gap_cost_vec));

                previous_row_vec = top_vec;
                row_cells_[pack_idx].u32 = cell_score_vec;
            }
        }

        unsigned const result_pack_idx = (longer_length - 1) / 4, result_lane_idx = (longer_length - 1) % 4;
        return (row_cells_[result_pack_idx].u32 >> (result_lane_idx * 8)) & 0xFF;
    }
};

/**
 *  @brief Bit-parallel Myers/Hyyrö @b unit-cost Levenshtein, one pair per thread, for shorter <= `words_ * 64` runes.
 *         A GPU port of the serial `levenshtein_distance_myers::unrolled_`: exact edit distance in
 *         O(longer * words_) 64-bit word operations, 64 DP cells per machine word, no DP matrix. The
 *         256-entry-per-word `Peq` (`match_masks`) table lives in a per-thread @b global-scratch slice (L1-cached);
 *         the dispatch zeroes the scratch once, and each pair clears its own shorter-character entries at the end so
 *         the slice stays clean between grid-stride pairs.
 *  @note Unit-cost Levenshtein ONLY (match 0, mismatch 1, gap 1, single-byte) - the dispatch gates on that predicate.
 *        Myers cannot encode weighted/affine costs, so there is deliberately no NW/SW variant.
 */
template <typename task_type_, typename char_type_ = char, u32_t words_ = 1,
          sz_capability_t capability_ = sz_cap_cuda_k>
__global__ __launch_bounds__(256, 4) void levenshtein_myers_per_cuda_thread_( //
    task_type_ *tasks, size_t tasks_count, u64_t *peq_scratch, size_t peq_stride) {

    size_t const thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const threads_per_device = static_cast<size_t>(gridDim.x) * blockDim.x;
    u64_t *const match_masks = peq_scratch + thread_index * peq_stride; // this thread's flattened [words_][256] Peq
    for (size_t task_index = thread_index; task_index < tasks_count; task_index += threads_per_device) {
        task_type_ &task = tasks[task_index];
        char_type_ const *const shorter_ptr =
            task.shorter.data(); // tasks are pre-sorted: shorter_length <= longer_length
        char_type_ const *const longer_ptr = task.longer.data();
        u32_t const shorter_length = static_cast<u32_t>(task.shorter.size());
        size_t const longer_length = task.longer.size();
        if (shorter_length == 0) {
            task.result = longer_length;
            continue;
        } // empty pattern -> distance is the text length

        // Build the `Peq` table: set one bit per shorter position (the slice is clean coming in).
        for (u32_t position = 0; position != shorter_length; ++position)
            match_masks[(position >> 6) * 256 + static_cast<u8_t>(shorter_ptr[position])] |= (u64_t)1
                                                                                             << (position & 63);

        u64_t vertical_positives[words_], vertical_negatives[words_]; // Myers' VP / VN, per 64-cell block
        for (u32_t word = 0; word != words_; ++word) vertical_positives[word] = ~(u64_t)0, vertical_negatives[word] = 0;
        u32_t const last_word = (shorter_length - 1) >> 6, last_bit = (shorter_length - 1) & 63;
        size_t distance = shorter_length;
        for (size_t longer_position = 0; longer_position != longer_length; ++longer_position) {
            u8_t const symbol = static_cast<u8_t>(longer_ptr[longer_position]);
            u64_t horizontal_positive_carry = 1, horizontal_negative_carry = 0; // top-row boundary into block 0 is +1
            for (u32_t word = 0; word != words_; ++word) {
                u64_t const pattern_matches = match_masks[word * 256 + symbol];
                u64_t const vertical_carry = pattern_matches | vertical_negatives[word];
                u64_t const matched_with_carry = pattern_matches | horizontal_negative_carry;
                u64_t const diagonal_zero =
                    (((matched_with_carry & vertical_positives[word]) + vertical_positives[word]) ^
                     vertical_positives[word]) |
                    matched_with_carry;
                u64_t horizontal_positive = vertical_negatives[word] | ~(diagonal_zero | vertical_positives[word]);
                u64_t horizontal_negative = vertical_positives[word] & diagonal_zero;
                if (word == last_word) {
                    distance += (horizontal_positive >> last_bit) & 1;
                    distance -= (horizontal_negative >> last_bit) & 1;
                }
                u64_t const hp_carry_next = horizontal_positive >> 63, hn_carry_next = horizontal_negative >> 63;
                horizontal_positive = (horizontal_positive << 1) | horizontal_positive_carry;
                horizontal_negative = (horizontal_negative << 1) | horizontal_negative_carry;
                horizontal_positive_carry = hp_carry_next, horizontal_negative_carry = hn_carry_next;
                vertical_positives[word] = horizontal_negative | ~(vertical_carry | horizontal_positive);
                vertical_negatives[word] = horizontal_positive & vertical_carry;
            }
        }
        task.result = distance;

        // Clear this pair's shorter entries so the slice is clean for the next grid-stride pair.
        for (u32_t position = 0; position != shorter_length; ++position)
            match_masks[(position >> 6) * 256 + static_cast<u8_t>(shorter_ptr[position])] = 0;
    }
}

/**
 *  @brief Size-generic bit-parallel Myers/Hyyrö @b unit-cost Levenshtein, one pair per thread, for @b any shorter
 *         length (`shorter > 64`, no word-count cap). The companion of `levenshtein_myers_per_cuda_thread_<words_=1>`:
 *         where the single-word kernel keeps VP/VN in registers, this one holds the per-word `Peq` table and the
 *         per-word VP/VN state in a per-thread @b global-scratch slice and loops over `words_count = ceil(shorter/64)`
 *         machine words, so it covers shorter lengths that would otherwise overflow a fixed register array.
 *
 *  Per text character the words are swept low-to-high, rippling two distinct cross-word carries:
 *    - the arithmetic carry-out of the `(Eq & VP) + VP` 65-bit add into the next word's add, and
 *    - the +1 / -1 horizontal score carries leaving the top bit of one word and entering bit 0 of the next word's
 *      shifted HP / HN.
 *  The running distance is updated from the horizontal delta at the pattern's final bit (top bit of the last,
 *  possibly partial, word). Validated 0-error vs full-DP for shorter lengths in {1..9000}.
 *
 *  @note Unit-cost Levenshtein ONLY (match 0, mismatch 1, gap 1, single-byte). Myers cannot encode weighted/affine
 *        costs, so there is deliberately no NW/SW variant. The scratch slice (`peq_stride` `u64_t` per thread, laid
 *        out as `[words_count][256]` `Peq` followed by `[words_count]` VP and `[words_count]` VN) is zeroed once by
 *        the dispatch; each pair clears its own `Peq` entries at the end so the slice stays clean between pairs.
 */
template <typename task_type_, typename char_type_ = char, sz_capability_t capability_ = sz_cap_cuda_k>
__global__ __launch_bounds__(256, 4) void levenshtein_myers_generic_per_cuda_thread_( //
    task_type_ *tasks, size_t tasks_count, u64_t *peq_scratch, size_t peq_stride) {

    size_t const thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const threads_per_device = static_cast<size_t>(gridDim.x) * blockDim.x;
    u64_t *const scratch = peq_scratch + thread_index * peq_stride;
    for (size_t task_index = thread_index; task_index < tasks_count; task_index += threads_per_device) {
        task_type_ &task = tasks[task_index];
        char_type_ const *const shorter_ptr = task.shorter.data();
        char_type_ const *const longer_ptr = task.longer.data();
        u32_t const shorter_length = static_cast<u32_t>(task.shorter.size());
        size_t const longer_length = task.longer.size();
        if (shorter_length == 0) {
            task.result = longer_length;
            continue;
        }
        if (longer_length == 0) {
            task.result = shorter_length;
            continue;
        }

        u32_t const words_count = (shorter_length + 63u) >> 6;
        u64_t *const match_masks = scratch;                                 // [words_count][256] flattened `Peq`
        u64_t *const vertical_positives = scratch + words_count * 256;      // [words_count] Myers' VP
        u64_t *const vertical_negatives = vertical_positives + words_count; // [words_count] Myers' VN

        // Build the `Peq` table: set one bit per shorter position (the slice is clean coming in).
        for (u32_t position = 0; position != shorter_length; ++position)
            match_masks[(position >> 6) * 256 + static_cast<u8_t>(shorter_ptr[position])] |= (u64_t)1
                                                                                             << (position & 63);
        for (u32_t word = 0; word != words_count; ++word)
            vertical_positives[word] = ~(u64_t)0, vertical_negatives[word] = 0;

        u32_t const last_word = words_count - 1u, last_bit = (shorter_length - 1u) & 63u;
        size_t distance = shorter_length;
        for (size_t longer_position = 0; longer_position != longer_length; ++longer_position) {
            u8_t const symbol = static_cast<u8_t>(longer_ptr[longer_position]);
            u64_t addition_carry = 0;
            u64_t horizontal_positive_carry = 1, horizontal_negative_carry = 0; // top-row boundary into word 0 is +1
            for (u32_t word = 0; word != words_count; ++word) {
                u64_t const pattern_matches = match_masks[word * 256 + symbol];
                u64_t const vertical_positive = vertical_positives[word];
                u64_t const vertical_negative = vertical_negatives[word];

                // 65-bit add: (Eq & VP) + VP + addition_carry, rippling the carry-out into the next word.
                u64_t const addition_term = pattern_matches & vertical_positive;
                u64_t const sum_low = addition_term + vertical_positive;
                u64_t const sum = sum_low + addition_carry;
                addition_carry = (sum_low < addition_term) | (sum < sum_low);

                u64_t const diagonal_zero = (sum ^ vertical_positive) | pattern_matches | vertical_negative;
                u64_t horizontal_positive = vertical_negative | ~(diagonal_zero | vertical_positive);
                u64_t horizontal_negative = vertical_positive & diagonal_zero;
                if (word == last_word) {
                    distance += (horizontal_positive >> last_bit) & 1;
                    distance -= (horizontal_negative >> last_bit) & 1;
                }
                u64_t const next_horizontal_positive_carry = horizontal_positive >> 63;
                u64_t const next_horizontal_negative_carry = horizontal_negative >> 63;
                horizontal_positive = (horizontal_positive << 1) | horizontal_positive_carry;
                horizontal_negative = (horizontal_negative << 1) | horizontal_negative_carry;
                horizontal_positive_carry = next_horizontal_positive_carry;
                horizontal_negative_carry = next_horizontal_negative_carry;
                vertical_positives[word] = horizontal_negative | ~(diagonal_zero | horizontal_positive);
                vertical_negatives[word] = horizontal_positive & diagonal_zero;
            }
        }
        task.result = distance;

        // Clear this pair's `Peq` entries so the slice is clean for the next grid-stride pair.
        for (u32_t position = 0; position != shorter_length; ++position)
            match_masks[(position >> 6) * 256 + static_cast<u8_t>(shorter_ptr[position])] = 0;
    }
}

/**
 *  @brief Bit-parallel Myers (unit-cost Levenshtein) sharing one query's @b Peq across a whole row of candidates.
 *         One @b WARP owns one query: its 32 lanes cooperatively build the query's 256-entry match-bitmask table
 *         once into shared memory, then stride over the query's candidates (one candidate per lane in flight),
 *         each lane running an independent single-word Myers scan that reuses the shared table.
 *
 *  @note Cross-product, non-symmetric, unit-cost ONLY, single-word queries (`query_length <= 64`). Myers is
 *        symmetric in its two operands, so the table is built on the query regardless of which side is shorter;
 *        the candidate is the scanned text and may be of any length. Tasks are query-major: query `q` owns the
 *        contiguous run `[q * candidates_count, (q + 1) * candidates_count)`, so runs are implicit (no sort).
 *        This amortizes the per-query table build (the per-pair kernels rebuild it for every candidate).
 */
template <typename task_type_, typename char_type_ = char, sz_capability_t capability_ = sz_cap_cuda_k>
__global__ __launch_bounds__(256, 4) void levenshtein_myers_candidates_per_cuda_warp_( //
    task_type_ *tasks, size_t queries_count, size_t candidates_count) {

    unsigned const warp_in_block = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    extern __shared__ u64_t shared_match_masks[]; // [warps_per_block][256]
    u64_t *const match_masks = shared_match_masks + static_cast<size_t>(warp_in_block) * 256;
    size_t const warp_index = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    size_t const warps_per_device = (static_cast<size_t>(gridDim.x) * blockDim.x) >> 5;

    for (size_t query_index = warp_index; query_index < queries_count; query_index += warps_per_device) {
        task_type_ const &row_head = tasks[query_index * candidates_count];
        char_type_ const *const query_ptr = row_head.query.data();
        u32_t const query_length = static_cast<u32_t>(row_head.query.size());

        // Build this query's `Peq` once into shared (all 32 lanes cooperate); the row reuses it.
        for (unsigned symbol = lane; symbol < 256u; symbol += 32u) match_masks[symbol] = 0;
        __syncwarp();
        for (u32_t position = lane; position < query_length; position += 32u)
            atomicOr(reinterpret_cast<unsigned long long *>(&match_masks[static_cast<u8_t>(query_ptr[position])]),
                     (u64_t)1 << position);
        __syncwarp();

        u64_t const top_bit = query_length ? ((u64_t)1 << (query_length - 1)) : 0;
        for (size_t candidate_in_row = lane; candidate_in_row < candidates_count; candidate_in_row += 32u) {
            task_type_ &task = tasks[query_index * candidates_count + candidate_in_row];
            // The candidate is the side that is not the query (pointer identity; query is one of shorter/longer).
            char_type_ const *const candidate_ptr = task.shorter.data() == query_ptr ? task.longer.data()
                                                                                     : task.shorter.data();
            size_t const candidate_length = task.shorter.data() == query_ptr ? task.longer.size() : task.shorter.size();
            if (query_length == 0) {
                task.result = candidate_length;
                continue;
            }

            u64_t vertical_positive = ~(u64_t)0, vertical_negative = 0;
            size_t distance = query_length;
            for (size_t candidate_position = 0; candidate_position != candidate_length; ++candidate_position) {
                u64_t const pattern_matches = match_masks[static_cast<u8_t>(candidate_ptr[candidate_position])];
                u64_t const vertical_carry = pattern_matches | vertical_negative;
                u64_t const diagonal_zero =
                    (((pattern_matches & vertical_positive) + vertical_positive) ^ vertical_positive) | pattern_matches;
                u64_t horizontal_positive = vertical_negative | ~(diagonal_zero | vertical_positive);
                u64_t horizontal_negative = vertical_positive & diagonal_zero;
                distance += (horizontal_positive & top_bit) ? 1 : 0;
                distance -= (horizontal_negative & top_bit) ? 1 : 0;
                horizontal_positive = (horizontal_positive << 1) | (u64_t)1; // boundary D[0][j] = j
                horizontal_negative = horizontal_negative << 1;
                vertical_positive = horizontal_negative | ~(vertical_carry | horizontal_positive);
                vertical_negative = horizontal_positive & vertical_carry;
            }
            task.result = distance;
        }
    }
}

/**
 *  @brief Levenshtein distances with @b one-thread-per-pair using only register memory, for short inputs.
 *
 *  Each thread runs a full register-resident DP (@ref register_levenshtein). Inputs are conceptually
 *  padded to a fixed @p max_text_length_ so the inner loop is branch-free; the divergent dimension is the outer
 *  (row) loop, so callers minimize divergence by keeping per-warp row counts similar (or putting a shared query
 *  on the outer axis when there are more targets than queries).
 */
template <                                            //
    typename task_type_,                              //
    typename char_type_ = char,                       //
    typename score_type_ = u8_t,                      //
    sz_capability_t capability_ = sz_cap_cuda_k,      //
    unsigned max_text_length_ = register_text_limit_k //
    >
__global__ __launch_bounds__(256, 4) void levenshtein_per_cuda_thread_( //
    task_type_ *tasks, size_t tasks_count,                              //
    uniform_substitution_costs_t const substituter, linear_gap_costs_t const gap_costs) {

    using task_t = task_type_;
    register_levenshtein<max_text_length_> levenshtein_computer;
    size_t const threads_per_device = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t task_idx = blockIdx.x * blockDim.x + threadIdx.x; task_idx < tasks_count;
         task_idx += threads_per_device) {
        task_t &task = tasks[task_idx];
        task.result = levenshtein_computer(                                                                  //
            reinterpret_cast<u8_t const *>(task.longer.data()), static_cast<unsigned>(task.longer.size()),   //
            reinterpret_cast<u8_t const *>(task.shorter.data()), static_cast<unsigned>(task.shorter.size()), //
            substituter, gap_costs);
    }
}

/**
 *  @brief Register-only Levenshtein for strings up to @p max_text_length_ bytes with @b 2-byte cells, one thread
 *         per pair. Like @ref register_levenshtein but packs @b two `u16_t` cells per `u32_t` (via the
 *         16-bit video instructions), so it covers non-unit costs / distances that overflow the 1-byte variant.
 */
template <unsigned max_text_length_>
struct register_levenshtein_u16 {
    static constexpr unsigned max_text_length_k = max_text_length_;
    static constexpr unsigned pack_count_k = max_text_length_k / 2; // ? two `u16_t` cells per `u32_t`

    u32_t row_cells_[pack_count_k];
    u8_t longer_chars_[max_text_length_k];

    __forceinline__ __device__ u16_t operator()(             //
        u8_t const *longer_string, unsigned longer_length,   //
        u8_t const *shorter_string, unsigned shorter_length, //
        uniform_substitution_costs_t const substituter, linear_gap_costs_t const gap_costs) noexcept {

        u16_t const gap_cost = gap_costs.open_or_extend;
        u32_t const gap_cost_vec = broadcast_cost_u16x2_(gap_cost);
        u32_t const match_cost_vec = broadcast_cost_u16x2_(substituter.match);
        u32_t const mismatch_cost_vec = broadcast_cost_u16x2_(substituter.mismatch);

        // Initialize the first row with the (column index + 1) * gap ladder, two cells per pack.
        fill_gap_ladder_(row_cells_, pack_count_k, 2, 16, gap_cost);
        for (unsigned i = 0; i < longer_length; ++i) longer_chars_[i] = longer_string[i];

        for (unsigned row_idx = 1; row_idx <= shorter_length; ++row_idx) {
            u8_t const shorter_char = shorter_string[row_idx - 1];
            u32_t const shorter_char_vec = broadcast_cost_u16x2_(shorter_char);
            u16_t left_cell = row_idx * gap_cost;            // west neighbor: column 0 of this row
            u16_t diagonal_carry = (row_idx - 1) * gap_cost; // NW neighbor: column 0 of the previous row
            for (unsigned pack_idx = 0; pack_idx < pack_count_k; ++pack_idx) {
                u32_t const top_vec = row_cells_[pack_idx];
                u16_t const top_low = top_vec & 0xFFFF, top_high = top_vec >> 16;
                u32_t const diagonal_vec = (u32_t)diagonal_carry | ((u32_t)top_low << 16);
                u32_t const longer_pair_vec = (u32_t)longer_chars_[2 * pack_idx] |
                                              ((u32_t)longer_chars_[2 * pack_idx + 1] << 16);
                u32_t const match_mask = __vcmpeq2(shorter_char_vec, longer_pair_vec);
                u32_t const cost_of_substitution_vec = (match_cost_vec & match_mask) |
                                                       (mismatch_cost_vec & ~match_mask);
                u32_t const cell_score_vec = __vminu2(__vaddus2(diagonal_vec, cost_of_substitution_vec),
                                                      __vaddus2(top_vec, gap_cost_vec));
                u16_t cell_low = cell_score_vec & 0xFFFF, cell_high = cell_score_vec >> 16;
                cell_low = (u16_t)std::min<unsigned>((unsigned)left_cell + gap_cost, cell_low);
                cell_high = (u16_t)std::min<unsigned>((unsigned)cell_low + gap_cost, cell_high);
                row_cells_[pack_idx] = (u32_t)cell_low | ((u32_t)cell_high << 16);
                left_cell = cell_high;
                diagonal_carry = top_high;
            }
        }
        unsigned const result_pack_idx = (longer_length - 1) / 2, result_lane_idx = (longer_length - 1) % 2;
        return (row_cells_[result_pack_idx] >> (result_lane_idx * 16)) & 0xFFFF;
    }
};

/** @brief One-thread-per-pair Levenshtein with @b 2-byte register cells; see @ref levenshtein_per_cuda_thread_. */
template <                                            //
    typename task_type_,                              //
    typename char_type_ = char,                       //
    sz_capability_t capability_ = sz_cap_cuda_k,      //
    unsigned max_text_length_ = register_text_limit_k //
    >
__global__ __launch_bounds__(256, 2) void levenshtein_u16_per_cuda_thread_( //
    task_type_ *tasks, size_t tasks_count,                                  //
    uniform_substitution_costs_t const substituter, linear_gap_costs_t const gap_costs) {

    using task_t = task_type_;
    register_levenshtein_u16<max_text_length_> levenshtein_computer;
    size_t const threads_per_device = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t task_idx = blockIdx.x * blockDim.x + threadIdx.x; task_idx < tasks_count;
         task_idx += threads_per_device) {
        task_t &task = tasks[task_idx];
        task.result = levenshtein_computer(                                                                  //
            reinterpret_cast<u8_t const *>(task.longer.data()), static_cast<unsigned>(task.longer.size()),   //
            reinterpret_cast<u8_t const *>(task.shorter.data()), static_cast<unsigned>(task.shorter.size()), //
            substituter, gap_costs);
    }
}

/**
 *  @brief Register-only @b affine-gap Levenshtein for strings up to @p max_text_length_ bytes with @b 2-byte cells,
 *         one thread per pair. Like @ref register_levenshtein_u16 but runs the Gotoh recurrence: it keeps a
 *         second register row for the insertion matrix @b I (`ins_vec_`) and carries the deletion matrix @b D as a
 *         scalar across the row, so gap opening and extension are priced separately.
 */
template <unsigned max_text_length_>
struct register_levenshtein_u16_affine {
    static constexpr unsigned max_text_length_k = max_text_length_;
    static constexpr unsigned pack_count_k = max_text_length_k / 2; // ? two `u16_t` cells per `u32_t`

    u32_t row_cells_[pack_count_k];       // ? the score matrix M
    u32_t insertion_cells_[pack_count_k]; // ? the insertion matrix I (gap from the row above)
    u8_t longer_chars_[max_text_length_k];

    __forceinline__ __device__ u16_t operator()(             //
        u8_t const *longer_string, unsigned longer_length,   //
        u8_t const *shorter_string, unsigned shorter_length, //
        uniform_substitution_costs_t const substituter, affine_gap_costs_t const gap_costs) noexcept {

        u16_t const open = gap_costs.open, extend = gap_costs.extend;
        u32_t const open_cost_vec = broadcast_cost_u16x2_(open);
        u32_t const extend_cost_vec = broadcast_cost_u16x2_(extend);
        u32_t const match_cost_vec = broadcast_cost_u16x2_(substituter.match);
        u32_t const mismatch_cost_vec = broadcast_cost_u16x2_(substituter.mismatch);

        // Row 0: M[0][j] = open + extend*(j-1); the gap matrix gets the higher-magnitude "discard" boundary so it
        // never wins, but stays bounded (no overflow on later additions) - matching the serial/warp affine scorer.
        for (unsigned pack_idx = 0; pack_idx < pack_count_k; ++pack_idx) {
            unsigned const column_low = 2 * pack_idx + 1, column_high = 2 * pack_idx + 2;
            u16_t const cell_low = open + extend * (column_low - 1);
            u16_t const cell_high = open + extend * (column_high - 1);
            row_cells_[pack_idx] = (u32_t)cell_low | ((u32_t)cell_high << 16);
            u16_t const insertion_low = (open + extend) + (open + extend * (column_low - 1));
            u16_t const insertion_high = (open + extend) + (open + extend * (column_high - 1));
            insertion_cells_[pack_idx] = (u32_t)insertion_low | ((u32_t)insertion_high << 16);
        }
        for (unsigned i = 0; i < longer_length; ++i) longer_chars_[i] = longer_string[i];

        for (unsigned row_idx = 1; row_idx <= shorter_length; ++row_idx) {
            u8_t const shorter_char = shorter_string[row_idx - 1];
            u32_t const shorter_char_vec = broadcast_cost_u16x2_(shorter_char);
            u16_t left_cell = open + extend * (row_idx - 1);                           // M[row][0]
            u16_t diagonal_carry = row_idx == 1 ? 0 : (open + extend * (row_idx - 2)); // M[row-1][0]
            u16_t left_deletion = (open + extend) + (open + extend * (row_idx - 1));   // D[row][0] (discard boundary)
            for (unsigned pack_idx = 0; pack_idx < pack_count_k; ++pack_idx) {
                u32_t const top_vec = row_cells_[pack_idx];
                u16_t const top_low = top_vec & 0xFFFF, top_high = top_vec >> 16;
                u32_t const previous_insertion_vec = insertion_cells_[pack_idx];
                // I[row][j] = min(M[row-1][j] + open, I[row-1][j] + extend) - independent per cell, so packed.
                u32_t const insertion_vec = __vminu2(__vaddus2(top_vec, open_cost_vec),
                                                     __vaddus2(previous_insertion_vec, extend_cost_vec));
                // diagonal = (M[row-1][2v], M[row-1][2v+1]); per-cell substitution cost; the substitution candidate.
                u32_t const diagonal_vec = (u32_t)diagonal_carry | ((u32_t)top_low << 16);
                u32_t const longer_pair_vec = (u32_t)longer_chars_[2 * pack_idx] |
                                              ((u32_t)longer_chars_[2 * pack_idx + 1] << 16);
                u32_t const match_mask = __vcmpeq2(shorter_char_vec, longer_pair_vec);
                u32_t const cost_of_substitution_vec = (match_cost_vec & match_mask) |
                                                       (mismatch_cost_vec & ~match_mask);
                // min(diagonal + subst, I) is independent per cell, so packed; only the deletion fold below is sequential.
                u32_t const match_or_insert_vec = __vminu2(__vaddus2(diagonal_vec, cost_of_substitution_vec),
                                                           insertion_vec);
                u16_t const match_or_insert_low = match_or_insert_vec & 0xFFFF;
                u16_t const match_or_insert_high = match_or_insert_vec >> 16;
                // Deletion D carries left across the row (sequential): cell 0 then cell 1.
                u16_t const deletion_low = std::min<u16_t>(left_cell + open, left_deletion + extend);
                u16_t const cell_low = std::min(match_or_insert_low, deletion_low);
                u16_t const deletion_high = std::min<u16_t>(cell_low + open, deletion_low + extend);
                u16_t const cell_high = std::min(match_or_insert_high, deletion_high);
                row_cells_[pack_idx] = (u32_t)cell_low | ((u32_t)cell_high << 16);
                insertion_cells_[pack_idx] = insertion_vec;
                left_cell = cell_high;
                left_deletion = deletion_high;
                diagonal_carry = top_high;
            }
        }
        unsigned const result_pack_idx = (longer_length - 1) / 2, result_lane_idx = (longer_length - 1) % 2;
        return (row_cells_[result_pack_idx] >> (result_lane_idx * 16)) & 0xFFFF;
    }
};

/** @brief One-thread-per-pair @b affine Levenshtein with 2-byte register cells; see @ref levenshtein_u16_per_cuda_thread_. */
template <                                            //
    typename task_type_,                              //
    typename char_type_ = char,                       //
    sz_capability_t capability_ = sz_cap_cuda_k,      //
    unsigned max_text_length_ = register_text_limit_k //
    >
__global__ __launch_bounds__(256, 2) void levenshtein_u16_affine_per_cuda_thread_( //
    task_type_ *tasks, size_t tasks_count,                                         //
    uniform_substitution_costs_t const substituter, affine_gap_costs_t const gap_costs) {

    using task_t = task_type_;
    register_levenshtein_u16_affine<max_text_length_> levenshtein_computer;
    size_t const threads_per_device = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t task_idx = blockIdx.x * blockDim.x + threadIdx.x; task_idx < tasks_count;
         task_idx += threads_per_device) {
        task_t &task = tasks[task_idx];
        task.result = levenshtein_computer(                                                                  //
            reinterpret_cast<u8_t const *>(task.longer.data()), static_cast<unsigned>(task.longer.size()),   //
            reinterpret_cast<u8_t const *>(task.shorter.data()), static_cast<unsigned>(task.shorter.size()), //
            substituter, gap_costs);
    }
}

#pragma region Shared Cross-Product Host Orchestration

/**
 *  @brief Tiled device-tier wavefront launch shared by every CUDA cross-product engine. Reduces the device-level
 *         tasks' shape maxima, carves the (hoisted, grow-only) `diagonals_u64_buffer_` into per-cell-width frontier
 *         slices, and chunks the batch along `blockIdx.y`, seeding then scoring each chunk. One body covers the
 *         linear (single row frontier + corners) and affine Gotoh (M + H row frontiers + M corners) data-flows via
 *         `if constexpr (is_affine_)`; the two engine families differ only in the resolved score / init
 *         `cudaFunction_t`s and the substituter type, both passed by reference. Bit-identical to the four former
 *         in-line `launch_batched` clones.
 *
 *  @param[in] init_fn,score_fn The resolved frontier-seed and tiled-score `CUfunction`s for @p score_type_.
 *  @param[in] substituter The cell-cost substituter (uniform for Levenshtein, the device 32-class map for NW/SW).
 */
template <typename score_type_, bool is_affine_, typename task_type_, typename allocator_type_,
          typename substituter_type_, typename gap_costs_type_>
cuda_status_t cuda_launch_tiled_device_tier_(cuda_cross_buffers<task_type_, allocator_type_> &buffers,
                                             span<task_type_> device_level_tasks, u32_t row_stride, u32_t corner_stride,
                                             unsigned grid_columns_blocks, cudaFunction_t init_fn,
                                             cudaFunction_t score_fn, substituter_type_ const &substituter,
                                             gap_costs_type_ const &gap_costs,
                                             cuda_executor_t const &executor) noexcept {
    using score_t = score_type_;
    static constexpr unsigned tiled_warps_per_block_k = 8;
    static constexpr u32_t tiled_grid_y_max_k = 65535u; // CUDA `gridDim.y` ceiling; larger batches chunk

    size_t const batch = device_level_tasks.size();
    task_type_ *const tasks_base = device_level_tasks.data();
    cuda_status_t tiled_status {status_t::success_k, cudaSuccess};

    size_t const chunk_pairs = batch < tiled_grid_y_max_k ? batch : tiled_grid_y_max_k;
    size_t const score_words = is_affine_ ? chunk_pairs * (2 * static_cast<size_t>(row_stride) + corner_stride)
                                          : chunk_pairs * (static_cast<size_t>(row_stride) + corner_stride);
    size_t const progress_words = chunk_pairs * corner_stride; // one `u32_t` progress counter per tile-column
    size_t const frontier_u64 = (score_words * sizeof(score_t) + progress_words * sizeof(u32_t)) / sizeof(u64_t) + 2;
    buffers.diagonals_u64_buffer_.clear();
    if (buffers.diagonals_u64_buffer_.try_resize(frontier_u64) == status_t::bad_alloc_k)
        return {status_t::bad_alloc_k, cudaSuccess};
    u64_t *const frontier_scratch = buffers.diagonals_u64_buffer_.data();

    auto const launch = [&](cudaFunction_t function, unsigned bx, unsigned by, unsigned threads,
                            void **args) noexcept -> bool {
        CUresult e =
            cuda_launch_t {}.grid(bx, by).block(threads).shared(0).stream(executor.stream()).launch(function, args);
        return e == CUDA_SUCCESS ? true : (tiled_status = make_cuda_status(e), false);
    };

    if constexpr (is_affine_) {
        // Affine Gotoh: the frontier carries M and H (deletion) left-edges (two `row` arrays) plus the diagonal M corners.
        score_t *row_frontier_m_base = reinterpret_cast<score_t *>(frontier_scratch);
        score_t *row_frontier_d_base = row_frontier_m_base + chunk_pairs * row_stride;
        score_t *corner_frontier_m_base = row_frontier_d_base + chunk_pairs * row_stride;
        u32_t *progress_base = reinterpret_cast<u32_t *>(
            (reinterpret_cast<uintptr_t>(corner_frontier_m_base + chunk_pairs * corner_stride) + alignof(u32_t) - 1) &
            ~static_cast<uintptr_t>(alignof(u32_t) - 1));
        for (size_t start = 0; start < batch && tiled_status.status == status_t::success_k; start += chunk_pairs) {
            unsigned const this_chunk = static_cast<unsigned>((batch - start) < chunk_pairs ? (batch - start)
                                                                                            : chunk_pairs);
            task_type_ *chunk_tasks = tasks_base + start;
            void *init_args[8] = {(void *)&chunk_tasks,         (void *)&row_frontier_m_base,
                                  (void *)&row_frontier_d_base, (void *)&corner_frontier_m_base,
                                  (void *)&progress_base,       (void *)&row_stride,
                                  (void *)&corner_stride,       (void *)&gap_costs};
            if (!launch(init_fn, 132, this_chunk, 256, init_args)) break;
            void *tiled_args[9] = {(void *)&chunk_tasks,
                                   (void *)&row_frontier_m_base,
                                   (void *)&row_frontier_d_base,
                                   (void *)&corner_frontier_m_base,
                                   (void *)&progress_base,
                                   (void *)&row_stride,
                                   (void *)&corner_stride,
                                   (void *)&substituter,
                                   (void *)&gap_costs};
            launch(score_fn, grid_columns_blocks, this_chunk, tiled_warps_per_block_k * 32, tiled_args);
        }
    }
    else {
        score_t *row_frontier_base = reinterpret_cast<score_t *>(frontier_scratch);
        score_t *corner_frontier_base = row_frontier_base + chunk_pairs * row_stride;
        // `progress` is `u32_t`; align its base up to 4 bytes - a `score_t == u16_t` carving can otherwise leave
        // the boundary only 2-byte aligned, misaligning the `atomic_ref<u32_t>` writes (a launch failure).
        u32_t *progress_base = reinterpret_cast<u32_t *>(
            (reinterpret_cast<uintptr_t>(corner_frontier_base + chunk_pairs * corner_stride) + alignof(u32_t) - 1) &
            ~static_cast<uintptr_t>(alignof(u32_t) - 1));
        for (size_t start = 0; start < batch && tiled_status.status == status_t::success_k; start += chunk_pairs) {
            unsigned const this_chunk = static_cast<unsigned>((batch - start) < chunk_pairs ? (batch - start)
                                                                                            : chunk_pairs);
            task_type_ *chunk_tasks = tasks_base + start;
            void *init_args[7] = {(void *)&chunk_tasks,   (void *)&row_frontier_base, (void *)&corner_frontier_base,
                                  (void *)&progress_base, (void *)&row_stride,        (void *)&corner_stride,
                                  (void *)&gap_costs};
            if (!launch(init_fn, 132, this_chunk, 256, init_args)) break;
            void *tiled_args[8] = {(void *)&chunk_tasks,   (void *)&row_frontier_base, (void *)&corner_frontier_base,
                                   (void *)&progress_base, (void *)&row_stride,        (void *)&corner_stride,
                                   (void *)&substituter,   (void *)&gap_costs};
            launch(score_fn, grid_columns_blocks, this_chunk, tiled_warps_per_block_k * 32, tiled_args);
        }
    }
    return tiled_status;
}

/**
 *  @brief Warp-tier launch loop shared by every CUDA cross-product engine: one non-cooperative launch per
 *         device-built `warp_tasks_group_descriptor_t` (densest groups first by the sort order). The host reads only
 *         the small descriptor array (kernel family, density, max shared memory, task subrange) and NEVER a device
 *         task; launches go onto the stream back-to-back with no per-group synchronization. The two engine families
 *         differ only in the warp kernel-shape pair (selected by `bytes_per_cell`) and the substituter, both passed
 *         by reference. Bit-identical to the two former in-line warp loops.
 */
template <typename task_type_, typename allocator_type_, typename substituter_type_, typename gap_costs_type_>
cuda_status_t cuda_launch_warp_groups_(cuda_cross_buffers<task_type_, allocator_type_> &buffers,
                                       span<task_type_> device_level_tasks, size_t warp_group_count,
                                       kernel_shape_t const &warp_narrow_shape, kernel_shape_t const &warp_wide_shape,
                                       size_t wide_bytes_per_cell, substituter_type_ const &substituter,
                                       gap_costs_type_ const &gap_costs, gpu_specs_t const &specs,
                                       cuda_executor_t const &executor) noexcept {
    cuda_status_t result {status_t::success_k, cudaSuccess};
    void *warp_level_kernel_args[5];
    task_type_ *const warp_tasks_base = device_level_tasks.data();
    warp_tasks_group_descriptor_t const *const descriptors = buffers.warp_group_descriptors_.data();

    for (size_t group = 0; group < warp_group_count && result.status == status_t::success_k; ++group) {
        warp_tasks_group_descriptor_t const &descriptor = descriptors[group];
        task_type_ *tasks_begin = warp_tasks_base + descriptor.begin_offset;
        size_t const count_tasks = descriptor.count;
        kernel_shape_t const &shape = descriptor.bytes_per_cell >= wide_bytes_per_cell ? warp_wide_shape
                                                                                       : warp_narrow_shape;
        auto const [optimal_density, speculative_factor] = speculation_friendly_density(descriptor.density);
        unsigned const shared_memory_per_block = static_cast<unsigned>(descriptor.max_memory_requirement *
                                                                       optimal_density);
        warp_level_kernel_args[0] = (void *)(&tasks_begin);
        warp_level_kernel_args[1] = (void *)(&count_tasks);
        warp_level_kernel_args[2] = (void *)(&substituter);
        warp_level_kernel_args[3] = (void *)(&gap_costs);
        warp_level_kernel_args[4] = (void *)(&shared_memory_per_block);
        unsigned const threads_per_block = static_cast<unsigned>(specs.warp_size * optimal_density);

        // The grid is sized per group from the actual shared memory (which varies with the data); extra
        // blocks beyond the supplied tasks exit through the grid-stride loop guard, so filling the device
        // once is never harmful. The handle is already resolved, so the query goes through the driver.
        unsigned occupancy_blocks_per_grid = 0;
        cuda_status_t occupancy_status = occupancy_grid_for(occupancy_blocks_per_grid, shape.function,
                                                            threads_per_block, shared_memory_per_block, specs);
        if (occupancy_status.status != status_t::success_k) {
            result = occupancy_status;
            break;
        }
        unsigned blocks_per_grid = static_cast<unsigned>(specs.streaming_multiprocessors) * speculative_factor;
        if (occupancy_blocks_per_grid > blocks_per_grid) blocks_per_grid = occupancy_blocks_per_grid;

        // Warp-level kernels do NOT use `grid.sync`, so they launch non-cooperatively via `cuLaunchKernelEx`.
        CUresult launch_error = cuda_launch_t {}
                                    .grid(blocks_per_grid)
                                    .block(threads_per_block)
                                    .shared(shared_memory_per_block)
                                    .stream(executor.stream())
                                    .launch(shape.function, warp_level_kernel_args);
        if (launch_error != CUDA_SUCCESS) result = make_cuda_status(launch_error);
    }
    return result;
}

#pragma endregion Shared Cross - Product Host Orchestration

/**
 *  @brief Dispatches baseline Levenshtein edit distance algorithm to the GPU.
 *         Before starting the kernels, bins them by size to maximize the number of blocks
 *         per grid that can run simultaneously, while fitting into the shared memory.
 */
template <typename gap_costs_type_, typename allocator_type_, sz_capability_t capability_>
struct levenshtein_distances<gap_costs_type_, allocator_type_, capability_,
                             std::enable_if_t<capability_ & sz_cap_cuda_k>> {

    using char_t = char;
    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;
    using scores_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<size_t>;
    static constexpr sz_capability_t capability_k = capability_;

    using task_t = cuda_similarity_task<char_t>;
    using buffers_t = cuda_cross_buffers<task_t, allocator_t>;
    using tier_keys_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<u64_t>;
    using tier_values_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<u32_t>;

    uniform_substitution_costs_t substituter_ {};
    gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    /** @brief The cross-product device buffer bundle shared with the weighted NW/SW engines (passed by reference). */
    buffers_t buffers_ {alloc_};

    /** @brief Grow-only packed `u64` radix-sort keys (DoubleBuffer A/B occupy `[0, count)` / `[count, 2*count)`). */
    safe_vector<u64_t, tier_keys_allocator_t> tier_keys_ {alloc_};
    /** @brief Grow-only `u32` permutation values for the radix sort (DoubleBuffer A/B, same layout as the keys). */
    safe_vector<u32_t, tier_values_allocator_t> tier_values_ {alloc_};
    /** @brief Device-side run-length-encode outputs: unique tier ids, run lengths, run count (`levenshtein_tier_count_k + 1`). */
    safe_vector<u32_t, tier_values_allocator_t> tier_rle_ {alloc_};
    /** @brief Host-side dense per-tier counts (empty tiers absent from the RLE expand to zero). */
    size_t tier_counts_[levenshtein_tier_count_k] {};

    safe_vector<u64_t, scores_allocator_t> myers_peq_buffer_ {alloc_}; // per-thread Myers `Peq` scratch (zeroed once)
    /** @brief Row width of the current cross-product (for the Myers-reuse kernel). */
    size_t cross_candidates_count_ = 0;
    /** @brief Longest query this call (reuse needs every query single-word, <= 64). */
    size_t cross_max_query_length_ = 0;
    /** @brief How the current call pairs its inputs (all-pairs vs symmetric self-similarity). */
    cross_similarities_t cross_kind_ = cross_similarities_t::all_pairs_k;
    cuda_timer_t timer_ {};

    levenshtein_distances(uniform_substitution_costs_t subs = {}, gap_costs_t gaps = {},
                          allocator_t const &alloc = {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    levenshtein_distances(levenshtein_distances const &) = delete;
    levenshtein_distances &operator=(levenshtein_distances const &) = delete;
    levenshtein_distances(levenshtein_distances &&) noexcept = default;
    levenshtein_distances &operator=(levenshtein_distances &&) noexcept = default;

    using final_score_t = size_t;

    struct cuda_kernels_t {
        kernel_shape_t register_u8;
        kernel_shape_t register_u16;
        kernel_shape_t warp_u8;
        kernel_shape_t warp_u16;
        // Device tier: the tiled register-micro-tile scorer + its frontier-seed kernel, one shape per cell width.
        // (Resolves to the linear `tiled_score_across_cuda_device_` or the affine sibling, matching `affine_k`.)
        // The old cooperative `[affine_]score_across_cuda_device_` kernels were retired - they were dead weight.
        kernel_shape_t tiled_u16, tiled_u32, tiled_u64;
        kernel_shape_t tiled_init_u16, tiled_init_u32, tiled_init_u64;
        // Bit-parallel Myers fast path (linear unit-cost Levenshtein only): a register-resident single-word
        // thread-per-pair kernel for shorter <= 64, and one size-generic thread-per-pair kernel for shorter > 64
        // of any length (no word-count cap). Null for the affine engine, which never routes to Myers.
        kernel_shape_t myers_word1, myers_generic;
        /** @brief Per-query `Peq`-reuse across a row of candidates (cross-product). */
        kernel_shape_t myers_candidates_warp;
        /** @brief Device-side per-cell task build (all-pairs or symmetric). */
        kernel_shape_t materialize_tasks;
        /** @brief Device-side scatter of task results into the row-major matrix. */
        kernel_shape_t scatter_results;
        /** @brief Device tier-router: packs the radix-sort keys + identity permutation over `tasks_`. */
        kernel_shape_t build_tier_keys;
        /** @brief Device tier-router: gathers the sorted permutation into the reordered task array. */
        kernel_shape_t gather_tasks;
    };

    /** @brief Resolves the engine's CUDA kernel table once and returns it with the resolution status. */
    static expected<cuda_kernels_t const &, cuda_status_t> kernels() noexcept {
        static cuda_kernels_t cuda_kernels;
        static bool resolved = false;
        if (resolved) return {cuda_kernels, {}};
        constexpr unsigned text_limit_k = register_text_limit_k;
        constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
        constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
        constexpr bool affine_k = is_same_type<gap_costs_t, affine_gap_costs_t>::value;
        int device = 0, warp_ceiling = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&warp_ceiling, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        cuda_status_t status {status_t::success_k, cudaSuccess};

        // Register tier (thread-per-pair, fixed 256-thread / no-shared shape).
        if constexpr (affine_k)
            status = resolve_kernel_shape(
                cuda_kernels.register_u16,
                (void const *)&levenshtein_u16_affine_per_cuda_thread_<task_t, char_t, capability_k, text_limit_k>, 256,
                0, true);
        else {
            status = resolve_kernel_shape(
                cuda_kernels.register_u8,
                (void const *)&levenshtein_per_cuda_thread_<task_t, char_t, u8_t, capability_k, text_limit_k>, 256, 0,
                true);
            if (status.status != status_t::success_k) return {cuda_kernels, status};
            status = resolve_kernel_shape(
                cuda_kernels.register_u16,
                (void const *)&levenshtein_u16_per_cuda_thread_<task_t, char_t, capability_k, text_limit_k>, 256, 0,
                true);
        }
        if (status.status != status_t::success_k) return {cuda_kernels, status};

        // Warp tier (anti-diagonal; dynamic-shared ceiling raised once, grid sized per group at launch).
        status = resolve_kernel_shape(
            cuda_kernels.warp_u8,
            affine_k
                ? (void const *)&affine_score_per_cuda_warp_<task_t, char_t, u8_t, u8_t, uniform_substitution_costs_t,
                                                             objective_k, locality_k, capability_k>
                : (void const *)&linear_score_per_cuda_warp_<task_t, char_t, u8_t, u8_t, uniform_substitution_costs_t,
                                                             objective_k, locality_k, capability_k>,
            0, (unsigned)warp_ceiling, false);
        if (status.status != status_t::success_k) return {cuda_kernels, status};
        status = resolve_kernel_shape(
            cuda_kernels.warp_u16,
            affine_k
                ? (void const *)&affine_score_per_cuda_warp_<task_t, char_t, u16_t, u16_t, uniform_substitution_costs_t,
                                                             objective_k, locality_k, capability_k>
                : (void const *)&linear_score_per_cuda_warp_<task_t, char_t, u16_t, u16_t, uniform_substitution_costs_t,
                                                             objective_k, locality_k, capability_k>,
            0, (unsigned)warp_ceiling, false);
        if (status.status != status_t::success_k) return {cuda_kernels, status};

        // Device tier: the tiled micro-tile scorer + its frontier-seed kernel, one shape per cell width. We only need
        // the resolved `CUfunction` (the launch sizes its grid from the tile-column count, not occupancy), so
        // `precompute_occupancy` is false. Resolves to the linear or affine sibling, matching `affine_k`.
        static constexpr unsigned tiled_warps_k = 8;
        auto const resolve_tiled = [&]<typename score_t>(kernel_shape_t &score_shape,
                                                         kernel_shape_t &init_shape) -> cuda_status_t {
            void const *score_sym =
                affine_k ? (void const *)&tiled_score_affine_across_cuda_device_<
                               tiled_warps_k, char_t, score_t, final_score_t, uniform_substitution_costs_t, objective_k,
                               locality_k, capability_k, task_t>
                         : (void const *)&tiled_score_across_cuda_device_<tiled_warps_k, char_t, score_t, final_score_t,
                                                                          uniform_substitution_costs_t, objective_k,
                                                                          locality_k, capability_k, task_t>;
            void const *init_sym =
                affine_k ? (void const
                                *)&tiled_frontier_init_affine_<score_t, final_score_t, objective_k, locality_k, task_t>
                         : (void const *)&tiled_frontier_init_<score_t, final_score_t, objective_k, locality_k, task_t>;
            cuda_status_t s = resolve_kernel_shape(score_shape, score_sym, tiled_warps_k * 32, 0, false);
            if (s.status != status_t::success_k) return s;
            return resolve_kernel_shape(init_shape, init_sym, 256, 0, false);
        };
        status = resolve_tiled.template operator()<u16_t>(cuda_kernels.tiled_u16, cuda_kernels.tiled_init_u16);
        if (status.status != status_t::success_k) return {cuda_kernels, status};
        status = resolve_tiled.template operator()<u32_t>(cuda_kernels.tiled_u32, cuda_kernels.tiled_init_u32);
        if (status.status != status_t::success_k) return {cuda_kernels, status};
        status = resolve_tiled.template operator()<u64_t>(cuda_kernels.tiled_u64, cuda_kernels.tiled_init_u64);
        if (status.status != status_t::success_k) return {cuda_kernels, status};

        // Bit-parallel Myers fast path (linear unit-cost Levenshtein only); occupancy precomputed for grid sizing.
        if constexpr (!affine_k) {
            status = resolve_kernel_shape(
                cuda_kernels.myers_word1,
                (void const *)&levenshtein_myers_per_cuda_thread_<task_t, char_t, 1u, capability_k>, 256, 0, true);
            if (status.status != status_t::success_k) return {cuda_kernels, status};
            status = resolve_kernel_shape(
                cuda_kernels.myers_generic,
                (void const *)&levenshtein_myers_generic_per_cuda_thread_<task_t, char_t, capability_k>, 256, 0, true);
            if (status.status != status_t::success_k) return {cuda_kernels, status};
            // Per-query `Peq`-reuse kernel: 256 threads = 8 warps, each warp owns 256 `u64_t` of shared `Peq`.
            static constexpr unsigned myers_candidates_shared_k = (256u / 32u) * 256u * sizeof(u64_t);
            status = resolve_kernel_shape(
                cuda_kernels.myers_candidates_warp,
                (void const *)&levenshtein_myers_candidates_per_cuda_warp_<task_t, char_t, capability_k>, 256,
                myers_candidates_shared_k, true);
            if (status.status != status_t::success_k) return {cuda_kernels, status};
        }
        // Device-side task materialization (all-pairs or symmetric); grid is sized from the cell count, so no
        // occupancy precompute is needed. Levenshtein is always minimize / global.
        status = resolve_kernel_shape(
            cuda_kernels.materialize_tasks,
            (void const *)&materialize_similarity_tasks_<sz_minimize_distance_k, sz_similarity_global_k, affine_k,
                                                         task_t, gap_costs_t>,
            256, 0, false);
        if (status.status != status_t::success_k) return {cuda_kernels, status};
        // Device tier-router kernels (grid-stride, fixed 256-thread blocks; the router caps the grid itself).
        status = resolve_kernel_shape(cuda_kernels.build_tier_keys,
                                      (void const *)&levenshtein_build_tier_keys_kernel<char_t>, 256, 0, false);
        if (status.status != status_t::success_k) return {cuda_kernels, status};
        status = resolve_kernel_shape(cuda_kernels.gather_tasks, (void const *)&levenshtein_gather_tasks_kernel<char_t>,
                                      256, 0, false);
        if (status.status != status_t::success_k) return {cuda_kernels, status};
        resolved = true;
        return {cuda_kernels, {}};
    }

    /** @brief Container-independent GPU pipeline trampoline over the packed `tasks_`; compiles once per engine. */
    cuda_status_t run_trampoline_(cuda_executor_t const &executor, gpu_specs_t specs) noexcept;

    /**
     *  @brief Device tier-router: reorders `tasks_` into the eight contiguous tier ranges and fills `tier_counts_`,
     *         entirely on the GPU, replacing the host `std::partition` / `std::sort` / `std::upper_bound` tiering.
     *         One `cub::DeviceRadixSort::SortPairs` over the packed key lays out every task in final tier order; a
     *         gather materializes the reordered array (swapped into `tasks_`); one `cub::DeviceRunLengthEncode`
     *         over the dense-tier transform iterator yields the per-tier counts. @p mode selects whether the short
     *         (`shorter <= 2048`) tasks route to the Myers word tiers or fall through to the register/device split.
     */
    cuda_status_t route_tiers_(cuda_executor_t const &executor, [[maybe_unused]] gpu_specs_t const &specs,
                               levenshtein_tier_mode_t mode, kernel_shape_t const &build_keys_shape,
                               kernel_shape_t const &gather_shape) noexcept {
        // Delegate to the shared dyadic length-bucket router free function (also used by the weighted NW/SW engines):
        // it reorders `tasks_` with one radix sort + gather and run-length-encodes the dense tier id into the host
        // `tier_counts_`. The Levenshtein five-tier Myers/register/device taxonomy is supplied here via the
        // engine-owned build-keys kernel + `levenshtein_dense_tier_functor`; the radix bits cover the packed key from
        // the original-index field up through the tier-priority field.
        int const begin_bit = levenshtein_original_index_shift_k;
        int const end_bit = levenshtein_tier_priority_shift_k + levenshtein_tier_priority_bits_k;
        levenshtein_dense_tier_functor<char_t> dense_tier_functor {nullptr, mode};
        return cuda_route_tasks_into_tiers_(buffers_, tier_keys_, tier_values_, tier_rle_, build_keys_shape,
                                            gather_shape, mode, dense_tier_functor, begin_bit, end_bit,
                                            levenshtein_tier_count_k, executor, tier_counts_);
    }

    /**
     *  @brief Shared cross-product driver: builds one task per live (query, candidate) cell, runs the
     *         container-independent GPU pipeline, and scatters each result into the row-major matrix. For
     *         @b symmetric_k only the lower triangle (incl. the diagonal) is built and each result is mirrored.
     */
    template <typename queries_type_, typename candidates_type_, typename results_type_>
    cuda_status_t cross_(queries_type_ const &queries, candidates_type_ const &candidates, results_type_ &&results,
                         cross_similarities_t cross_kind, cuda_executor_t const &executor, gpu_specs_t specs) noexcept {

        bool const is_symmetric = cross_kind == cross_similarities_t::symmetric_k;
        size_t const queries_count = queries.size();
        size_t const candidates_count = candidates.size();
        size_t const row_stride = results.row_stride;
        size_t const live_cells = is_symmetric ? queries_count * (queries_count + 1) / 2
                                               : queries_count * candidates_count;

        // No `clear()`: every live cell is fully overwritten by the materialization below, and a same-size
        // `try_resize` of a trivially-constructible task does zero (re-)construction - so the host never touches
        // the (large) task array, which would otherwise force a fault-driven unified-memory ping-pong.
        auto &tasks = buffers_.tasks_;
        if (tasks.try_resize_uninitialized(live_cells) == status_t::bad_alloc_k) return {status_t::bad_alloc_k};

        // Ensure inputs are device-accessible (Unified/Device memory). Both sides come from contiguous
        // tapes/arrays, so we validate the base pointers of the first element once (covers the whole tape)
        // instead of a per-cell `cudaPointerGetAttributes` driver round-trip.
        if (queries_count != 0 && candidates_count != 0) {
            if (!is_device_accessible_memory((void const *)queries[0].data()) ||
                !is_device_accessible_memory((void const *)candidates[0].data()))
                return {status_t::device_memory_mismatch_k, cudaSuccess};
        }

        // Export one task per live cell; the per-cell sizing/tiering is unchanged from the pairwise design.
        using diagonal_memory_requirements_t = diagonal_memory_requirements<size_t>;
        cross_max_query_length_ = 0;
        for (size_t query_index = 0; query_index < queries_count; ++query_index)
            if (queries[query_index].length() > cross_max_query_length_)
                cross_max_query_length_ = queries[query_index].length();

        // Device-side materialization for BOTH all-pairs and symmetric: build the O(queries+candidates)
        // descriptors on the host and let one thread per live cell fill the O(queries*candidates) task array on
        // the GPU (the symmetric path maps the flat cell index into the lower triangle). The descriptor buffers
        // are unified, so the host writes them and the kernel reads them with no extra copy.
        if (buffers_.query_descs_.try_resize(queries_count) == status_t::bad_alloc_k ||
            buffers_.candidate_descs_.try_resize(candidates_count) == status_t::bad_alloc_k)
            return {status_t::bad_alloc_k};
        for (size_t query_index = 0; query_index < queries_count; ++query_index)
            buffers_.query_descs_[query_index] = {queries[query_index].data(), queries[query_index].length()};
        for (size_t candidate_index = 0; candidate_index < candidates_count; ++candidate_index)
            buffers_.candidate_descs_[candidate_index] = {candidates[candidate_index].data(),
                                                          candidates[candidate_index].length()};

        auto [kernel_table, kernels_status] = kernels();
        if (kernels_status.status != status_t::success_k) return kernels_status;

        span<char const> *queries_ptr = buffers_.query_descs_.data(),
                           *candidates_ptr = buffers_.candidate_descs_.data();
        task_t *tasks_ptr = tasks.data();
        error_cost_magnitude_t substitute_magnitude = substituter_.magnitude(), gap_magnitude = gap_costs_.magnitude();
        sz_similarity_gaps_t gap_type_value = gap_type<gap_costs_t>();
        bytes_per_cell_t min_bytes_per_cell = one_byte_per_cell_k; // Levenshtein cells start at 1 byte.
        cross_similarities_t cross_kind_copy = cross_kind;
        gap_costs_t gap_costs_copy = gap_costs_;
        gpu_specs_t specs_copy = specs;
        void *materialize_args[13] = {(void *)&tasks_ptr,       (void *)&queries_ptr,          (void *)&candidates_ptr,
                                      (void *)&queries_count,   (void *)&candidates_count,     (void *)&row_stride,
                                      (void *)&cross_kind_copy, (void *)&substitute_magnitude, (void *)&gap_magnitude,
                                      (void *)&gap_type_value,  (void *)&min_bytes_per_cell,   (void *)&gap_costs_copy,
                                      (void *)&specs_copy};
        unsigned const block = 256;
        unsigned const grid = static_cast<unsigned>((live_cells + block - 1) / block);
        CUresult materialize_error = cuda_launch_t {}
                                         .grid(grid)
                                         .block(block)
                                         .shared(0)
                                         .stream(executor.stream())
                                         .launch(kernel_table.materialize_tasks.function, materialize_args);
        if (materialize_error != CUDA_SUCCESS) return make_cuda_status(materialize_error);
        // The downstream tiering reads `tasks_` on the host (partition/sort), so drain the kernel first.
        if (cudaStreamSynchronize(executor.stream()) != cudaSuccess) return make_cuda_status(cudaGetLastError());

        cross_candidates_count_ = candidates_count;
        cross_kind_ = cross_kind;
        cuda_status_t status = run_trampoline_(executor, specs);
        if (status.status != status_t::success_k) return status;

        // Scatter on the device when the output is device-accessible (the common unified-memory case): the host
        // then never reads the large GPU-resident task array, avoiding a full unified-memory page migration. The
        // scatter kernel depends on `value_type_`, so it is resolved at the call site rather than the cached table.
        if (tasks.size() && is_device_accessible_memory((void const *)results.data)) {
            using results_value_t = typename std::remove_reference_t<results_type_>::value_type;
            kernel_shape_t scatter_shape;
            cuda_status_t scatter_resolve = resolve_kernel_shape(
                scatter_shape, (void const *)&scatter_similarity_results_<task_t, results_value_t>, 256, 0, false);
            if (scatter_resolve.status != status_t::success_k) return scatter_resolve;
            results_value_t *results_ptr = results.data;
            task_t const *tasks_ptr = tasks.data();
            size_t tasks_size = tasks.size();
            void *scatter_args[3] = {(void *)&tasks_ptr, (void *)&tasks_size, (void *)&results_ptr};
            unsigned const block = 256;
            unsigned const grid = static_cast<unsigned>((tasks_size + block - 1) / block);
            CUresult scatter_error = cuda_launch_t {}
                                         .grid(grid)
                                         .block(block)
                                         .shared(0)
                                         .stream(executor.stream())
                                         .launch(scatter_shape.function, scatter_args);
            if (scatter_error != CUDA_SUCCESS) return make_cuda_status(scatter_error);
            if (cudaStreamSynchronize(executor.stream()) != cudaSuccess) return make_cuda_status(cudaGetLastError());
        }
        else if (tasks.size()) {
            // Host-only output: device-scatter into a dense staging matrix, then one strided `cudaMemcpy2DAsync`
            // into the caller's host matrix - stream-async, no hot-path alloc.
            using results_value_t = typename std::remove_reference_t<results_type_>::value_type;
            cuda_status_t const fallback_status = cuda_scatter_results_to_host_strided_<task_t, results_value_t>(
                buffers_, tasks.data(), tasks.size(), results, executor, 256u);
            if (fallback_status.status != status_t::success_k) return fallback_status;
        }
        return status;
    }

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    cuda_status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                             strided_rows<value_type_> results, cuda_executor_t const &executor = {},
                             gpu_specs_t specs = {}) noexcept {
        return cross_(queries, candidates, results, cross_similarities_t::all_pairs_k, executor, specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    cuda_status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                             cuda_executor_t const &executor = {}, gpu_specs_t specs = {}) noexcept {
        return cross_(sequences, sequences, results, cross_similarities_t::symmetric_k, executor, specs);
    }
};

template <typename gap_costs_type_, typename allocator_type_, sz_capability_t capability_>
cuda_status_t levenshtein_distances<gap_costs_type_, allocator_type_, capability_,
                                    std::enable_if_t<capability_ & sz_cap_cuda_k>>::run_trampoline_( //
    cuda_executor_t const &executor, gpu_specs_t specs) noexcept {

    constexpr bool is_affine_k = is_same_type<gap_costs_t, affine_gap_costs_t>::value;
    [[maybe_unused]] constexpr size_t count_diagonals_k = is_affine_k ? 7 : 3;

    // Create the engine-owned timing events on first use; the kernel table resolves itself on first access.
    CUresult timer_error = timer_.ensure_created();
    if (timer_error != CUDA_SUCCESS) return make_cuda_status(timer_error);
    auto [kernel_table, kernels_status] = kernels();
    if (kernels_status.status != status_t::success_k) return kernels_status;
    auto &tasks = buffers_.tasks_;

    // Record the start event
    CUresult start_event_error = timer_.record_start(executor.stream());
    if (start_event_error != CUDA_SUCCESS) return make_cuda_status(start_event_error);

    // Per-query Myers `Peq`-reuse fast path (the retrieval regime): a non-symmetric unit-cost cross-product whose
    // queries are all single-word (`<= 64`), with at least a warp's worth of candidates per row to amortize the
    // shared-table build. One warp owns a query's whole row and reuses its `Peq` across every candidate - bit-exact
    // with the per-pair Myers kernels (identical recurrence), so it covers the whole tier pipeline when it fires.
    if constexpr (!is_affine_k) {
        bool const is_unit_cost = substituter_.match == 0 && substituter_.mismatch == 1 &&
                                  gap_costs_.open_or_extend == 1;
        static constexpr size_t reuse_min_candidates_k = 32; // below a warp's width, reuse cannot fill its lanes
        size_t const candidates_count = cross_candidates_count_;
        size_t const queries_count = candidates_count ? tasks.size() / candidates_count : 0;
        bool const reuse_eligible = is_unit_cost && cross_kind_ == cross_similarities_t::all_pairs_k &&
                                    cross_max_query_length_ <= 64 && candidates_count >= reuse_min_candidates_k &&
                                    queries_count != 0 && queries_count * candidates_count == tasks.size();
        if (reuse_eligible) {
            kernel_shape_t const &shape = kernel_table.myers_candidates_warp;
            unsigned const blocks = shape.blocks_per_multiprocessor * specs.streaming_multiprocessors;
            static constexpr unsigned reuse_shared_k = (256u / 32u) * 256u * sizeof(u64_t);
            task_t *tasks_ptr = tasks.data();
            size_t queries_count_arg = queries_count, candidates_count_arg = candidates_count;
            void *reuse_args[3] = {(void *)&tasks_ptr, (void *)&queries_count_arg, (void *)&candidates_count_arg};
            CUresult launch_error = cuda_launch_t {}
                                        .grid(blocks)
                                        .block(256)
                                        .shared(reuse_shared_k)
                                        .stream(executor.stream())
                                        .launch(shape.function, reuse_args);
            if (launch_error != CUDA_SUCCESS) return make_cuda_status(launch_error);
            CUresult stop_event_error = timer_.record_stop(executor.stream());
            if (stop_event_error != CUDA_SUCCESS) return make_cuda_status(stop_event_error);
            CUresult execution_error = timer_.synchronize(executor.stream());
            if (execution_error != CUDA_SUCCESS) return make_cuda_status(execution_error);
            return {status_t::success_k, cudaSuccess, CUDA_SUCCESS, timer_.elapsed_milliseconds()};
        }
    }

    {
        // Device tier-router: one `cub::DeviceRadixSort` lays every task out in its final contiguous tier order and
        // one `cub::DeviceRunLengthEncode` fills `tier_counts_`. Unit-cost linear Levenshtein routes short pairs
        // (shorter <= 2048) to the bit-parallel Myers word tiers; affine and non-unit-cost linear skip Myers and split
        // only into register / device tiers. The tier-kernel launches below consume the router's offsets + counts directly.
        bool is_unit_cost = false;
        if constexpr (!is_affine_k)
            is_unit_cost = substituter_.match == 0 && substituter_.mismatch == 1 && gap_costs_.open_or_extend == 1;
        levenshtein_tier_mode_t const tier_mode = is_unit_cost ? levenshtein_tier_mode_t::myers_and_registers_k
                                                               : levenshtein_tier_mode_t::registers_only_k;
        cuda_status_t const router_status = route_tiers_(executor, specs, tier_mode, kernel_table.build_tier_keys,
                                                         kernel_table.gather_tasks);
        if (router_status.status != status_t::success_k) return router_status;

        // Myers tier ends, derived from the dense per-tier counts (contiguous prefix sums): the register-resident
        // single-word tier `[0, word1_count)` holds shorter <= 64, the size-generic tier `[word1_count, myers_count)`
        // holds the rest of any length.
        size_t const word1_count = tier_counts_[levenshtein_tier_myers_word1_k];
        size_t const generic_count = tier_counts_[levenshtein_tier_myers_generic_k];
        size_t const myers_count = word1_count + generic_count;

        // Bit-parallel Myers fast path: claim unit-cost Levenshtein pairs (match 0 / mismatch 1 / gap 1) of ANY size
        // and run the register-resident single-word kernel (shorter <= 64) plus the size-generic kernel (the rest) -
        // far faster than the anti-diagonal register/device tiers. The router places them at the front; the
        // register/device tiers below operate on the remaining `[myers_count, size)` tasks. (Linear engine only -
        // Myers can't encode affine or weighted costs.)
        if constexpr (!is_affine_k) {
            if (myers_count) {
                // The size-generic kernel sizes its per-thread scratch to the widest pair it may run: words_count =
                // ceil(max_generic_shorter / 64) `Peq` rows of 256 plus 2*words_count VP/VN slots. The single-word
                // kernel needs a flat 256-entry `Peq` per thread. Both launches reuse the buffer sequentially (each
                // pair cleans up its own entries), so size it to the larger per-thread slice times the grid threads.
                unsigned const word1_threads = kernel_table.myers_word1.blocks_per_multiprocessor *
                                               specs.streaming_multiprocessors * 256u;
                size_t const word1_peq = word1_count ? static_cast<size_t>(word1_threads) * 256 : 0;
                size_t generic_stride = 0, generic_peq = 0;
                if (generic_count) {
                    device_tier_maxima_t generic_maxima {};
                    cuda_status_t const maxima_status = reduce_device_tier_maxima_<char_t>(
                        tasks.data() + word1_count, generic_count, buffers_.cub_buffer_,
                        buffers_.device_maxima_scratch_, executor.stream(), generic_maxima);
                    if (maxima_status.status != status_t::success_k) return maxima_status;
                    u32_t const words_count = (generic_maxima.max_shorter + 63u) >> 6;
                    generic_stride = static_cast<size_t>(words_count) * 256 + 2u * words_count; // Peq + VP + VN
                    unsigned const generic_threads = kernel_table.myers_generic.blocks_per_multiprocessor *
                                                     specs.streaming_multiprocessors * 256u;
                    generic_peq = static_cast<size_t>(generic_threads) * generic_stride;
                }
                size_t const peq_words_total = word1_peq > generic_peq ? word1_peq : generic_peq;
                myers_peq_buffer_.clear();
                if (myers_peq_buffer_.try_resize(peq_words_total) == status_t::bad_alloc_k)
                    return {status_t::bad_alloc_k};
                if (cudaMemsetAsync(myers_peq_buffer_.data(), 0, peq_words_total * sizeof(u64_t), executor.stream()) !=
                    cudaSuccess)
                    return make_cuda_status(cudaGetLastError());
                u64_t *peq = myers_peq_buffer_.data();

                cuda_status_t myers_status {status_t::success_k, cudaSuccess};
                auto const launch_myers = [&](kernel_shape_t const &shape, size_t first, size_t count,
                                              size_t stride) noexcept {
                    if (!count || myers_status.status != status_t::success_k) return;
                    task_t *tasks_ptr = tasks.data() + first;
                    void *args[4] = {(void *)&tasks_ptr, (void *)&count, (void *)&peq, (void *)&stride};
                    unsigned const blocks = shape.blocks_per_multiprocessor * specs.streaming_multiprocessors;
                    CUresult e = cuda_launch_t {}
                                     .grid(blocks)
                                     .block(256)
                                     .shared(0)
                                     .stream(executor.stream())
                                     .launch(shape.function, args);
                    if (e != CUDA_SUCCESS) myers_status = make_cuda_status(e);
                };
                launch_myers(kernel_table.myers_word1, 0, word1_count, 256);
                launch_myers(kernel_table.myers_generic, word1_count, generic_count, generic_stride);
                if (myers_status.status != status_t::success_k) return myers_status;
            }
        }

        // Tiny tasks (both strings within the register limit) run as a register-only thread-per-pair kernel - far
        // faster than the warp anti-diagonal wavefront for short inputs. The router places them right after the Myers
        // range (one-byte cells first), so the warp/device tiers handle only the rest.
        size_t const count_u8_register_tasks = tier_counts_[levenshtein_tier_register_u8_k];
        size_t const count_register_level_tasks = count_u8_register_tasks +
                                                  tier_counts_[levenshtein_tier_register_u16_k];
        if (count_register_level_tasks) {
            cuda_status_t register_status {status_t::success_k, cudaSuccess};

            // Launches a resolved register thread-per-pair kernel over `tasks[first, first + count)`.
            auto const launch_register_kernel = [&](kernel_shape_t const &shape, size_t first, size_t count) noexcept {
                if (!count || register_status.status != status_t::success_k) return;
                task_t *tasks_ptr = tasks.data() + first;
                void *kernel_args[4] = {(void *)(&tasks_ptr), (void *)(&count), (void *)(&substituter_),
                                        (void *)(&gap_costs_)};
                unsigned const blocks_per_grid = shape.blocks_per_multiprocessor * specs.streaming_multiprocessors;
                CUresult launch_error = cuda_launch_t {}
                                            .grid(blocks_per_grid)
                                            .block(256)
                                            .shared(0)
                                            .stream(executor.stream())
                                            .launch(shape.function, kernel_args);
                if (launch_error != CUDA_SUCCESS) register_status = make_cuda_status(launch_error);
            };
            if constexpr (!is_affine_k) {
                // Within the register tier, 1-byte cells go to the (4×-SIMD) u8 kernel and the rest to the u16 one;
                // the router already laid the one-byte tasks first, so the counts give the exact sub-ranges.
                launch_register_kernel(kernel_table.register_u8, myers_count, count_u8_register_tasks);
                launch_register_kernel(kernel_table.register_u16, myers_count + count_u8_register_tasks,
                                       count_register_level_tasks - count_u8_register_tasks);
            }
            else {
                // Affine gaps need 2-byte cells (gap opening overflows the 1-byte variant), so the whole register
                // tier runs the u16 affine kernel.
                launch_register_kernel(kernel_table.register_u16, myers_count, count_register_level_tasks);
            }
            if (register_status.status != status_t::success_k) return register_status;
        }

        size_t const non_myers_register = myers_count + count_register_level_tasks;
        size_t warp_group_count = 0;
        [[maybe_unused]] auto [device_level_tasks, warp_level_tasks, empty_tasks] = warp_tasks_grouping<task_t>(
            {tasks.data() + non_myers_register, tasks.size() - non_myers_register}, specs, executor.stream(),
            buffers_.cub_buffer_, buffers_.warp_partition_scratch_, buffers_.warp_split_counts_,
            buffers_.warp_group_keys_, buffers_.warp_group_sizes_, buffers_.warp_group_descriptors_, warp_group_count);

        if (device_level_tasks.size()) {
            // Device tier -> the tiled register-micro-tile kernel (linear) or its Gotoh sibling (affine), batched
            // across pairs on `blockIdx.y`: one warp owns a tile-column, `gridDim.y` pairs run concurrently (the
            // batch regime). `buffers_.diagonals_u64_buffer_` holds every pair's frontier slice padded to the batch's
            // largest pair, run at the widest cell type any pair needs. Floor the device-tier cell at 32-bit: GPU
            // scalar u16 min/add promote to 32-bit anyway, so a u16 tile is ~1.3-1.4x SLOWER than u32 (measured) and
            // the narrower frontier buys nothing - the tiled kernel is compute/occupancy-bound with 0% DRAM.
            static constexpr unsigned tiled_warps_per_block_k = 8, tiled_tile_side_k = 128;
            device_tier_maxima_t maxima;
            cuda_status_t const maxima_status = reduce_device_tier_maxima_<char_t>(
                device_level_tasks.data(), device_level_tasks.size(), buffers_.cub_buffer_,
                buffers_.device_maxima_scratch_, executor.stream(), maxima);
            if (maxima_status.status != status_t::success_k) return maxima_status;
            unsigned const max_bytes_per_cell = sz_max_of_two(maxima.max_bytes_per_cell,
                                                              static_cast<u32_t>(sizeof(u32_t)));
            u32_t const max_columns = divide_round_up<u32_t>(maxima.max_longer, tiled_tile_side_k);
            u32_t const max_padded_rows = round_up_to_multiple<u32_t>(maxima.max_shorter, tiled_tile_side_k);
            u32_t const row_stride = max_padded_rows + 1, corner_stride = max_columns;
            unsigned const grid_columns_blocks = divide_round_up<unsigned>(max_columns, tiled_warps_per_block_k);

            // Resolve the per-cell-width tiled score + frontier-seed `CUfunction`s from the table and launch the
            // shared batched driver; the linear and affine frontier carvings live in the driver, gated on `affine_k`.
            auto const launch_batched = [&]<typename score_t>() noexcept -> cuda_status_t {
                cudaFunction_t const init_fn = sizeof(score_t) == 8   ? kernel_table.tiled_init_u64.function
                                               : sizeof(score_t) == 4 ? kernel_table.tiled_init_u32.function
                                                                      : kernel_table.tiled_init_u16.function;
                cudaFunction_t const score_fn = sizeof(score_t) == 8   ? kernel_table.tiled_u64.function
                                                : sizeof(score_t) == 4 ? kernel_table.tiled_u32.function
                                                                       : kernel_table.tiled_u16.function;
                return cuda_launch_tiled_device_tier_<score_t, is_affine_k>(
                    buffers_, device_level_tasks, row_stride, corner_stride, grid_columns_blocks, init_fn, score_fn,
                    substituter_, gap_costs_, executor);
            };

            cuda_status_t tiled_status {status_t::success_k, cudaSuccess};
            if (max_bytes_per_cell >= sizeof(u64_t)) tiled_status = launch_batched.template operator()<u64_t>();
            else if (max_bytes_per_cell >= sizeof(u32_t)) tiled_status = launch_batched.template operator()<u32_t>();
            else tiled_status = launch_batched.template operator()<u16_t>();
            if (tiled_status.status != status_t::success_k) return tiled_status;
        }

        // Now process the remaining warp-level tasks via the shared per-descriptor launch loop: the host reads only
        // the small descriptor array (never a device task) and the launches go onto the stream back-to-back.
        if (warp_group_count) {
            cuda_status_t const warp_status = cuda_launch_warp_groups_(
                buffers_, device_level_tasks, warp_group_count, kernel_table.warp_u8, kernel_table.warp_u16,
                sizeof(u16_t), substituter_, gap_costs_, specs, executor);
            if (warp_status.status != status_t::success_k) return warp_status;
        }

        // Single trailing sync: drains every group's kernels enqueued during the ENQUEUE phase before we read results.
        CUresult stop_event_error = timer_.record_stop(executor.stream());
        if (stop_event_error != CUDA_SUCCESS) return make_cuda_status(stop_event_error);
        CUresult execution_error = timer_.synchronize(executor.stream());
        if (execution_error != CUDA_SUCCESS) return make_cuda_status(execution_error);
        float execution_milliseconds = timer_.elapsed_milliseconds();
        return {status_t::success_k, cudaSuccess, CUDA_SUCCESS, execution_milliseconds};
    }
}

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

    u8_t const *byte_to_class = nullptr;
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
        unsigned const class_a = byte_to_class[static_cast<u8_t>(a)];
        unsigned const class_b = byte_to_class[static_cast<u8_t>(b)];
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
    __shared__ u8_t shared_byte_to_class[256];
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
 *         thread per pair. Like @ref register_levenshtein_u16 but @b maximizes (`__vmaxs2`/`__vaddss2`),
 *         gathers the per-cell substitution cost via the class substituter, and - for local scoring - clamps cells
 *         to zero and tracks a column-guarded running maximum (padded columns >ll are excluded).
 */
template <unsigned max_text_length_, sz_similarity_locality_t locality_, sz_capability_t capability_ = sz_cap_cuda_k>
struct register_weighted {
    static constexpr unsigned max_text_length_k = max_text_length_;
    static constexpr unsigned pack_count_k = max_text_length_k / 2; // ? two signed `i16_t` cells per `u32_t`
    static constexpr bool is_local_k = locality_ == sz_similarity_local_k;

    u32_t row_cells_[pack_count_k];
    u8_t longer_chars_[max_text_length_k];

    __forceinline__ __device__ i16_t operator()(             //
        u8_t const *longer_string, unsigned longer_length,   //
        u8_t const *shorter_string, unsigned shorter_length, //
        error_costs_classes_in_cuda_shared_memory_t const substituter, linear_gap_costs_t const gap_costs) noexcept {

        i16_t const gap_cost = gap_costs.open_or_extend; // ? stored as a (negative) penalty, added to scores
        u32_t const gap_cost_vec = broadcast_cost_u16x2_((u16_t)gap_cost);

        // Initialize the first row: local alignment starts at zero, global with the signed `column * gap` ladder.
        for (unsigned pack_idx = 0; pack_idx < pack_count_k; ++pack_idx) {
            if constexpr (is_local_k) { row_cells_[pack_idx] = 0; }
            else {
                i16_t const cell_low = (i16_t)((2 * pack_idx + 1) * gap_cost);
                i16_t const cell_high = (i16_t)((2 * pack_idx + 2) * gap_cost);
                row_cells_[pack_idx] = (u16_t)cell_low | ((u32_t)(u16_t)cell_high << 16);
            }
        }
        for (unsigned i = 0; i < longer_length; ++i) longer_chars_[i] = longer_string[i];

        i16_t best_score = 0;
        for (unsigned row_idx = 1; row_idx <= shorter_length; ++row_idx) {
            u8_t const shorter_char = shorter_string[row_idx - 1];
            i16_t left_cell = is_local_k ? 0 : (i16_t)(row_idx * gap_cost);
            i16_t diagonal_carry = is_local_k ? 0 : (i16_t)((row_idx - 1) * gap_cost);
            for (unsigned pack_idx = 0; pack_idx < pack_count_k; ++pack_idx) {
                u32_t const top_vec = row_cells_[pack_idx];
                i16_t const top_low = (i16_t)(top_vec & 0xFFFF), top_high = (i16_t)(top_vec >> 16);
                u32_t const diagonal_vec = (u16_t)diagonal_carry | ((u32_t)(u16_t)top_low << 16);
                i16_t const substitution_cost_low = substituter(shorter_char, longer_chars_[2 * pack_idx]);
                i16_t const substitution_cost_high = substituter(shorter_char, longer_chars_[2 * pack_idx + 1]);
                u32_t const cost_of_substitution_vec = (u16_t)substitution_cost_low |
                                                       ((u32_t)(u16_t)substitution_cost_high << 16);
                u32_t const cell_score_vec = __vmaxs2(__vaddss2(diagonal_vec, cost_of_substitution_vec),
                                                      __vaddss2(top_vec, gap_cost_vec));
                i16_t cell_low = (i16_t)(cell_score_vec & 0xFFFF), cell_high = (i16_t)(cell_score_vec >> 16);
                // Sequential left (horizontal) fold + local clamp. Mirrors the maximize branch of `tiled_dp_cell_`:
                // `__viaddmax_s32(a,b,c) == max(a + b, c)` for the gap fold; the Smith-Waterman clamp-to-0 stays at the
                // end (the high-cell fold reads the un-clamped low cell, matching the scalar path bit-for-bit).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
                if constexpr ((capability_ & sz_cap_hopper_k) != 0 && sizeof(i16_t) <= 4) {
                    int const gap = (int)gap_cost;
                    cell_low = (i16_t)__viaddmax_s32((int)left_cell, gap, (int)cell_low);
                    cell_high = (i16_t)__viaddmax_s32((int)cell_low, gap, (int)cell_high);
                    if constexpr (is_local_k) {
                        cell_low = std::max<i16_t>(cell_low, 0);
                        cell_high = std::max<i16_t>(cell_high, 0);
                    }
                }
                else
#endif
                {
                    cell_low = std::max<i16_t>(cell_low, left_cell + gap_cost);
                    cell_high = std::max<i16_t>(cell_high, cell_low + gap_cost);
                    if constexpr (is_local_k) {
                        cell_low = std::max<i16_t>(cell_low, 0);
                        cell_high = std::max<i16_t>(cell_high, 0);
                    }
                }
                if constexpr (is_local_k) {
                    unsigned const column_low = 2 * pack_idx + 1, column_high = 2 * pack_idx + 2;
                    if (column_low <= longer_length) best_score = std::max(best_score, cell_low);
                    if (column_high <= longer_length) best_score = std::max(best_score, cell_high);
                }
                row_cells_[pack_idx] = (u16_t)cell_low | ((u32_t)(u16_t)cell_high << 16);
                left_cell = cell_high;
                diagonal_carry = top_high;
            }
        }
        if constexpr (is_local_k) return best_score;
        unsigned const result_pack_idx = (longer_length - 1) / 2, result_lane_idx = (longer_length - 1) % 2;
        return (i16_t)((row_cells_[result_pack_idx] >> (result_lane_idx * 16)) & 0xFFFF);
    }
};

/** @brief One-thread-per-pair NW/SW scoring with signed 2-byte register cells; see @ref register_weighted. */
template <                                                       //
    typename task_type_,                                         //
    typename char_type_ = char,                                  //
    sz_similarity_locality_t locality_ = sz_similarity_global_k, //
    sz_capability_t capability_ = sz_cap_cuda_k,                 //
    unsigned max_text_length_ = register_text_limit_k            //
    >
__global__ __launch_bounds__(256, 2) void weighted_per_cuda_thread_( //
    task_type_ *tasks, size_t tasks_count,                           //
    error_costs_classes_in_cuda_shared_memory_t const substituter, linear_gap_costs_t const gap_costs) {

    using task_t = task_type_;
    register_weighted<max_text_length_, locality_, capability_> nw_sw_computer;
    size_t const threads_per_device = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t task_idx = blockIdx.x * blockDim.x + threadIdx.x; task_idx < tasks_count;
         task_idx += threads_per_device) {
        task_t &task = tasks[task_idx];
        task.result = nw_sw_computer(                                                                        //
            reinterpret_cast<u8_t const *>(task.longer.data()), static_cast<unsigned>(task.longer.size()),   //
            reinterpret_cast<u8_t const *>(task.shorter.data()), static_cast<unsigned>(task.shorter.size()), //
            substituter, gap_costs);
    }
}

/**
 *  @brief Register-only @b affine-gap NW/SW scoring for strings up to @p max_text_length_ bytes with @b signed
 *         2-byte cells, one thread per pair. Like @ref register_weighted but runs the Gotoh recurrence: a
 *         second register row holds the insertion matrix @b I (`ins_vec_`) and the deletion matrix @b D is carried
 *         as a scalar across the row, so gap opening and extension are priced separately.
 */
template <unsigned max_text_length_, sz_similarity_locality_t locality_, sz_capability_t capability_ = sz_cap_cuda_k>
struct register_weighted_affine {
    static constexpr unsigned max_text_length_k = max_text_length_;
    static constexpr unsigned pack_count_k = max_text_length_k / 2; // ? two signed `i16_t` cells per `u32_t`
    static constexpr bool is_local_k = locality_ == sz_similarity_local_k;

    u32_t row_cells_[pack_count_k];       // ? the score matrix M
    u32_t insertion_cells_[pack_count_k]; // ? the insertion matrix I (gap from the row above)
    u8_t longer_chars_[max_text_length_k];

    __forceinline__ __device__ i16_t operator()(             //
        u8_t const *longer_string, unsigned longer_length,   //
        u8_t const *shorter_string, unsigned shorter_length, //
        error_costs_classes_in_cuda_shared_memory_t const substituter, affine_gap_costs_t const gap_costs) noexcept {

        i16_t const open = gap_costs.open, extend = gap_costs.extend; // ? stored as (negative) penalties
        u32_t const open_cost_vec = broadcast_cost_u16x2_((u16_t)open);
        u32_t const extend_cost_vec = broadcast_cost_u16x2_((u16_t)extend);

        // Row 0: global M[0][j] = open + extend*(j-1) (local resets to 0); the gap matrix gets the higher-magnitude
        // "discard" boundary so it never wins, but stays bounded - matching the serial/warp affine scorer.
        for (unsigned pack_idx = 0; pack_idx < pack_count_k; ++pack_idx) {
            unsigned const column_low = 2 * pack_idx + 1, column_high = 2 * pack_idx + 2;
            if constexpr (is_local_k) { row_cells_[pack_idx] = 0; }
            else {
                i16_t const cell_low = open + extend * (i16_t)(column_low - 1);
                i16_t const cell_high = open + extend * (i16_t)(column_high - 1);
                row_cells_[pack_idx] = (u16_t)cell_low | ((u32_t)(u16_t)cell_high << 16);
            }
            i16_t const insertion_low = (open + extend) + (open + extend * (i16_t)(column_low - 1));
            i16_t const insertion_high = (open + extend) + (open + extend * (i16_t)(column_high - 1));
            insertion_cells_[pack_idx] = (u16_t)insertion_low | ((u32_t)(u16_t)insertion_high << 16);
        }
        for (unsigned i = 0; i < longer_length; ++i) longer_chars_[i] = longer_string[i];

        i16_t best_score = 0;
        for (unsigned row_idx = 1; row_idx <= shorter_length; ++row_idx) {
            u8_t const shorter_char = shorter_string[row_idx - 1];
            i16_t left_cell = is_local_k ? (i16_t)0 : (i16_t)(open + extend * (i16_t)(row_idx - 1)); // M[row][0]
            i16_t diagonal_carry = is_local_k ? (i16_t)0
                                              : (i16_t)(row_idx == 1 ? 0 : (open + extend * (i16_t)(row_idx - 2)));
            i16_t left_deletion = (open + extend) + (open + extend * (i16_t)(row_idx - 1)); // D[row][0] (discard)
            for (unsigned pack_idx = 0; pack_idx < pack_count_k; ++pack_idx) {
                u32_t const top_vec = row_cells_[pack_idx];
                i16_t const top_low = (i16_t)(top_vec & 0xFFFF), top_high = (i16_t)(top_vec >> 16);
                u32_t const previous_insertion_vec = insertion_cells_[pack_idx];
                // I[row][j] = max(M[row-1][j] + open, I[row-1][j] + extend) - independent per cell, so packed.
                u32_t const insertion_vec = __vmaxs2(__vaddss2(top_vec, open_cost_vec),
                                                     __vaddss2(previous_insertion_vec, extend_cost_vec));
                // diagonal = (M[row-1][2v], M[row-1][2v+1]); per-cell substitution cost; the substitution candidate.
                u32_t const diagonal_vec = (u16_t)diagonal_carry | ((u32_t)(u16_t)top_low << 16);
                i16_t const substitution_cost_low = substituter(shorter_char, longer_chars_[2 * pack_idx]);
                i16_t const substitution_cost_high = substituter(shorter_char, longer_chars_[2 * pack_idx + 1]);
                u32_t const cost_of_substitution_vec = (u16_t)substitution_cost_low |
                                                       ((u32_t)(u16_t)substitution_cost_high << 16);
                // max(diagonal + subst, I) is independent per cell, so packed; only the deletion fold below is sequential.
                u32_t const match_or_insert_vec = __vmaxs2(__vaddss2(diagonal_vec, cost_of_substitution_vec),
                                                           insertion_vec);
                i16_t const match_or_insert_low = (i16_t)(match_or_insert_vec & 0xFFFF);
                i16_t const match_or_insert_high = (i16_t)(match_or_insert_vec >> 16);
                // Deletion D carries left across the row (sequential): cell 0 then cell 1. The open/extend fold is the
                // affine H-track of `tiled_dp_cell_affine_`: `__viaddmax_s32(left_m, open, left_h + extend)`, then the
                // 2-way `max(match_or_insert, deletion)`. The Smith-Waterman clamp-to-0 stays at the end (the next
                // deletion fold reads the un-clamped `cell_low`, matching the scalar path bit-for-bit).
                i16_t deletion_low, cell_low, deletion_high, cell_high;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
                if constexpr ((capability_ & sz_cap_hopper_k) != 0 && sizeof(i16_t) <= 4) {
                    int const open_i = (int)open, extend_i = (int)extend;
                    deletion_low = (i16_t)__viaddmax_s32((int)left_cell, open_i, (int)left_deletion + extend_i);
                    cell_low = std::max(match_or_insert_low, deletion_low);
                    deletion_high = (i16_t)__viaddmax_s32((int)cell_low, open_i, (int)deletion_low + extend_i);
                    cell_high = std::max(match_or_insert_high, deletion_high);
                    if constexpr (is_local_k) {
                        cell_low = std::max<i16_t>(cell_low, 0);
                        cell_high = std::max<i16_t>(cell_high, 0);
                    }
                }
                else
#endif
                {
                    deletion_low = std::max<i16_t>(left_cell + open, left_deletion + extend);
                    cell_low = std::max(match_or_insert_low, deletion_low);
                    deletion_high = std::max<i16_t>(cell_low + open, deletion_low + extend);
                    cell_high = std::max(match_or_insert_high, deletion_high);
                    if constexpr (is_local_k) {
                        cell_low = std::max<i16_t>(cell_low, 0);
                        cell_high = std::max<i16_t>(cell_high, 0);
                    }
                }
                if constexpr (is_local_k) {
                    unsigned const column_low = 2 * pack_idx + 1, column_high = 2 * pack_idx + 2;
                    if (column_low <= longer_length) best_score = std::max(best_score, cell_low);
                    if (column_high <= longer_length) best_score = std::max(best_score, cell_high);
                }
                row_cells_[pack_idx] = (u16_t)cell_low | ((u32_t)(u16_t)cell_high << 16);
                insertion_cells_[pack_idx] = insertion_vec;
                left_cell = cell_high;
                left_deletion = deletion_high;
                diagonal_carry = top_high;
            }
        }
        if constexpr (is_local_k) return best_score;
        unsigned const result_pack_idx = (longer_length - 1) / 2, result_lane_idx = (longer_length - 1) % 2;
        return (i16_t)((row_cells_[result_pack_idx] >> (result_lane_idx * 16)) & 0xFFFF);
    }
};

/** @brief One-thread-per-pair @b affine NW/SW scoring with signed 2-byte register cells; see @ref weighted_per_cuda_thread_. */
template <                                                       //
    typename task_type_,                                         //
    typename char_type_ = char,                                  //
    sz_similarity_locality_t locality_ = sz_similarity_global_k, //
    sz_capability_t capability_ = sz_cap_cuda_k,                 //
    unsigned max_text_length_ = register_text_limit_k            //
    >
__global__ __launch_bounds__(256, 2) void weighted_affine_per_cuda_thread_( //
    task_type_ *tasks, size_t tasks_count,                                  //
    error_costs_classes_in_cuda_shared_memory_t const substituter, affine_gap_costs_t const gap_costs) {

    using task_t = task_type_;
    register_weighted_affine<max_text_length_, locality_, capability_> nw_sw_computer;
    size_t const threads_per_device = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t task_idx = blockIdx.x * blockDim.x + threadIdx.x; task_idx < tasks_count;
         task_idx += threads_per_device) {
        task_t &task = tasks[task_idx];
        task.result = nw_sw_computer(                                                                        //
            reinterpret_cast<u8_t const *>(task.longer.data()), static_cast<unsigned>(task.longer.size()),   //
            reinterpret_cast<u8_t const *>(task.shorter.data()), static_cast<unsigned>(task.shorter.size()), //
            substituter, gap_costs);
    }
}

#pragma region Weighted Device Tier Router

/**
 *  @brief Weighted NW/SW three-tier taxonomy fed to the shared @ref cuda_route_tasks_into_tiers_ dyadic router,
 *         replacing the former two sequential `cub::DevicePartition::If` passes. In final contiguous order:
 *    0 register : `fits_in_registers()` (both sides <= @ref register_text_limit_k) -> thread-per-pair register scorer
 *    1 batch    : `fits_in_batch()`     (register < L <= @ref batch_text_limit_k)  -> thread-per-pair batch scorer
 *    2 device   : everything else (warp anti-diagonal + tiled device wavefront grouping downstream)
 *
 *  The packed `u64` key reuses the Levenshtein field layout (tier-priority MSBs, ascending shorter-length sub-sort,
 *  original-index tiebreaker) so the radix bits and the run-length-encode dense-id scheme are identical machinery.
 */
static constexpr u32_t weighted_tier_register_k = 0;
static constexpr u32_t weighted_tier_batch_k = 1;
static constexpr u32_t weighted_tier_device_k = 2;
static constexpr int weighted_tier_count_k = 3;

/** @brief Dense tier id 0..2 (register / batch / device) for one weighted task. */
template <typename char_type_>
__host__ __device__ __forceinline__ u32_t
weighted_task_dense_tier(cuda_similarity_task<char_type_> const &task) noexcept {
    if (task.fits_in_registers()) return weighted_tier_register_k;
    if (task.fits_in_batch()) return weighted_tier_batch_k;
    return weighted_tier_device_k;
}

/** @brief Packs one weighted task into the Levenshtein-layout MSB-ordered radix-sort key (priority = dense tier id). */
template <typename char_type_>
__host__ __device__ __forceinline__ u64_t
weighted_pack_tier_key(cuda_similarity_task<char_type_> const &task, u32_t original_index) noexcept {
    u64_t const priority = weighted_task_dense_tier(task);
    // Dyadic length bucket (see @ref levenshtein_length_dyadic_bucket) so both engine families bucket identically.
    u64_t const shorter_max = (1ull << levenshtein_shorter_length_bits_k) - 1ull;
    u64_t const shorter = sz_min_of_two(levenshtein_length_dyadic_bucket(static_cast<u64_t>(task.shorter.size())),
                                        shorter_max);
    u64_t const index = original_index;
    u64_t key = 0;
    key |= priority << levenshtein_tier_priority_shift_k;
    key |= shorter << levenshtein_shorter_length_shift_k;
    key |= index << levenshtein_original_index_shift_k;
    return key;
}

/**
 *  @brief Build-keys kernel for the weighted router: packs one key + identity index per task. The fourth-from-last
 *         signature mirrors the Levenshtein build-keys kernel (a `mode` slot, unused here) so both reuse the shared
 *         five-argument launch path in @ref cuda_route_tasks_into_tiers_.
 */
template <typename char_type_>
__global__ void weighted_build_tier_keys_kernel(cuda_similarity_task<char_type_> const *tasks, size_t count,
                                                u32_t /*mode_unused*/, u64_t *keys, u32_t *task_indices) {
    size_t const stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count; index += stride) {
        keys[index] = weighted_pack_tier_key(tasks[index], static_cast<u32_t>(index));
        task_indices[index] = static_cast<u32_t>(index);
    }
}

/** @brief Run-length-encode source mapping a reordered weighted task index to its dense tier id; `tasks` set by router. */
template <typename char_type_>
struct weighted_dense_tier_functor {
    cuda_similarity_task<char_type_> const *tasks;
    u32_t mode; // unused; present so the router can construct/forward it uniformly with the Levenshtein functor
    __host__ __device__ __forceinline__ u32_t operator()(size_t index) const {
        return weighted_task_dense_tier(tasks[index]);
    }
};

#pragma endregion Weighted Device Tier Router

/**
 *  @brief Resolved CUDA kernel table shared by the weighted NW/SW engines (filled by @ref cuda_weighted_kernels_).
 *         Decoupled from any engine struct so the trampoline free function can take it by const reference - the
 *         no-inheritance substrate that replaced the retired `cuda_weighted_scores` base class.
 */
struct cuda_weighted_kernels_t {
    kernel_shape_t register_score;
    /** @brief Thread-per-pair @b batch scorer (same kernel, @ref batch_text_limit_k storage) for register<L<=~512. */
    kernel_shape_t batch_score;
    kernel_shape_t warp_i16;
    kernel_shape_t warp_i32;
    // Device tier: the tiled register-micro-tile scorer (objective=maximize, locality=global/local) + its
    // frontier-seed kernel, one shape per signed cell width. Resolves to the linear `tiled_score_across_cuda_device_`
    // or the affine sibling, matching `affine_k`. The old cooperative device kernels were retired as dead weight.
    kernel_shape_t tiled_i32, tiled_i64;
    kernel_shape_t tiled_init_i32, tiled_init_i64;
    /** @brief Device-side per-cell task build (all-pairs or symmetric). */
    kernel_shape_t materialize_tasks;
    /** @brief Device-side scatter of task results into the row-major matrix. */
    kernel_shape_t scatter_results;
    /** @brief Device tier-router: packs the radix-sort keys + identity permutation over `tasks_`. */
    kernel_shape_t build_tier_keys;
    /** @brief Device tier-router: gathers the sorted permutation into the reordered task array. */
    kernel_shape_t gather_tasks;
};

/**
 *  @brief Resolves the weighted NW/SW CUDA kernel table once per (gap-cost x locality x capability) and returns it
 *         with the resolution status. A free function template (state-by-reference substrate, no base class): both
 *         the `needleman_wunsch_scores` and `smith_waterman_scores` GPU engines call it with their own `locality_`.
 */
template <typename gap_costs_type_, sz_similarity_locality_t locality_, sz_capability_t capability_>
expected<cuda_weighted_kernels_t const &, cuda_status_t> cuda_weighted_kernels_() noexcept {
    using task_t = cuda_similarity_task<char>;
    using final_score_t = ssize_t;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_capability_t capability_k = capability_;

    static cuda_weighted_kernels_t cuda_kernels;
    static bool resolved = false;
    if (resolved) return {cuda_kernels, {}};
    using char_t = char;
    using substituter_t = error_costs_classes_in_cuda_shared_memory_t;
    constexpr unsigned text_limit_k = register_text_limit_k;
    constexpr sz_similarity_objective_t objective_k = sz_maximize_score_k;
    constexpr bool affine_k = is_same_type<gap_costs_type_, affine_gap_costs_t>::value;
    int device = 0, warp_ceiling = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&warp_ceiling, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    cuda_status_t status {status_t::success_k, cudaSuccess};

    // Register tier (single signed i16 thread-per-pair kernel, fixed 256-thread / no-shared shape).
    status = resolve_kernel_shape(
        cuda_kernels.register_score,
        affine_k ? (void const *)&weighted_affine_per_cuda_thread_<task_t, char_t, locality_k, capability_k, text_limit_k>
                 : (void const *)&weighted_per_cuda_thread_<task_t, char_t, locality_k, capability_k, text_limit_k>,
        256, 0, true);
    if (status.status != status_t::success_k) return {cuda_kernels, status};

    // Batch tier: the SAME thread-per-pair kernel instantiated with the larger `batch_text_limit_k` register/local
    // storage, covering register<L<=~512 where the measured thread-per-pair scorer still beats the tiled wavefront.
    constexpr unsigned batch_limit_k = batch_text_limit_k;
    status = resolve_kernel_shape(
        cuda_kernels.batch_score,
        affine_k ? (void const *)&weighted_affine_per_cuda_thread_<task_t, char_t, locality_k, capability_k, batch_limit_k>
                 : (void const *)&weighted_per_cuda_thread_<task_t, char_t, locality_k, capability_k, batch_limit_k>,
        256, 0, true);
    if (status.status != status_t::success_k) return {cuda_kernels, status};

    // Warp tier (anti-diagonal; dynamic-shared ceiling raised once, grid sized per group at launch).
    status = resolve_kernel_shape(
        cuda_kernels.warp_i16,
        affine_k ? (void const *)&affine_score_per_cuda_warp_<task_t, char_t, u16_t, i16_t, substituter_t, objective_k,
                                                              locality_k, capability_k>
                 : (void const *)&linear_score_per_cuda_warp_<task_t, char_t, u16_t, i16_t, substituter_t, objective_k,
                                                              locality_k, capability_k>,
        0, (unsigned)warp_ceiling, false);
    if (status.status != status_t::success_k) return {cuda_kernels, status};
    status = resolve_kernel_shape(
        cuda_kernels.warp_i32,
        affine_k ? (void const *)&affine_score_per_cuda_warp_<task_t, char_t, u32_t, i32_t, substituter_t, objective_k,
                                                              locality_k, capability_k>
                 : (void const *)&linear_score_per_cuda_warp_<task_t, char_t, u32_t, i32_t, substituter_t, objective_k,
                                                              locality_k, capability_k>,
        0, (unsigned)warp_ceiling, false);
    if (status.status != status_t::success_k) return {cuda_kernels, status};

    // Device tier: the tiled micro-tile scorer + its frontier-seed kernel, one shape per signed cell width. Only
    // the resolved `CUfunction` is needed (the launch sizes its grid from the tile-column count, not occupancy),
    // so `precompute_occupancy` is false. Resolves to the linear or affine sibling, matching `affine_k`.
    static constexpr unsigned tiled_warps_k = 8;
    auto const resolve_tiled = [&]<typename score_type_>(kernel_shape_t &score_shape,
                                                         kernel_shape_t &init_shape) -> cuda_status_t {
        void const *score_sym =
            affine_k
                ? (void const *)&tiled_score_affine_across_cuda_device_<tiled_warps_k, char_t, score_type_,
                                                                        final_score_t, substituter_t, objective_k,
                                                                        locality_k, capability_k, task_t>
                : (void const *)&tiled_score_across_cuda_device_<tiled_warps_k, char_t, score_type_, final_score_t,
                                                                 substituter_t, objective_k, locality_k, capability_k,
                                                                 task_t>;
        void const *init_sym =
            affine_k ? (void const
                            *)&tiled_frontier_init_affine_<score_type_, final_score_t, objective_k, locality_k, task_t>
                     : (void const *)&tiled_frontier_init_<score_type_, final_score_t, objective_k, locality_k, task_t>;
        cuda_status_t s = resolve_kernel_shape(score_shape, score_sym, tiled_warps_k * 32, 0, false);
        if (s.status != status_t::success_k) return s;
        return resolve_kernel_shape(init_shape, init_sym, 256, 0, false);
    };
    status = resolve_tiled.template operator()<i32_t>(cuda_kernels.tiled_i32, cuda_kernels.tiled_init_i32);
    if (status.status != status_t::success_k) return {cuda_kernels, status};
    status = resolve_tiled.template operator()<i64_t>(cuda_kernels.tiled_i64, cuda_kernels.tiled_init_i64);
    if (status.status != status_t::success_k) return {cuda_kernels, status};

    // Device-side task materialization (all-pairs or symmetric) + scatter; grid sized from the cell count.
    status = resolve_kernel_shape(
        cuda_kernels.materialize_tasks,
        (void const *)&materialize_similarity_tasks_<objective_k, locality_k, affine_k, task_t, gap_costs_type_>, 256, 0,
        false);
    if (status.status != status_t::success_k) return {cuda_kernels, status};
    status = resolve_kernel_shape(cuda_kernels.scatter_results,
                                  (void const *)&scatter_similarity_results_<task_t, final_score_t>, 256, 0, false);
    if (status.status != status_t::success_k) return {cuda_kernels, status};

    // Dyadic length-bucket router kernels (shared machinery with Levenshtein): the weighted build-keys kernel packs
    // the three-tier register/batch/device key; the gather kernel is the generic permutation-gather reused as-is.
    status = resolve_kernel_shape(cuda_kernels.build_tier_keys,
                                  (void const *)&weighted_build_tier_keys_kernel<char_t>, 256, 0, false);
    if (status.status != status_t::success_k) return {cuda_kernels, status};
    status = resolve_kernel_shape(cuda_kernels.gather_tasks,
                                  (void const *)&levenshtein_gather_tasks_kernel<char_t>, 256, 0, false);
    if (status.status != status_t::success_k) return {cuda_kernels, status};
    resolved = true;
    return {cuda_kernels, {}};
}

/**
 *  @brief Container-independent GPU pipeline trampoline over a weighted engine's packed `tasks_`. A free function
 *         taking every piece of engine state @b by-reference (the no-inheritance substrate that replaced the
 *         `cuda_weighted_scores` base): NW and SW call it identically, differing only in the resolved kernel table.
 */
template <typename gap_costs_type_, typename allocator_type_>
cuda_status_t cuda_weighted_run_trampoline_(                                                                     //
    cuda_cross_buffers<cuda_similarity_task<char>, allocator_type_> &buffers, error_costs_32x32_t const &substituter, //
    gap_costs_type_ const &gap_costs, cuda_weighted_kernels_t const &kernel_table, cuda_timer_t &timer,          //
    safe_vector<u64_t, typename std::allocator_traits<allocator_type_>::template rebind_alloc<u64_t>> &tier_keys, //
    safe_vector<u32_t, typename std::allocator_traits<allocator_type_>::template rebind_alloc<u32_t>> &tier_values, //
    safe_vector<u32_t, typename std::allocator_traits<allocator_type_>::template rebind_alloc<u32_t>> &tier_rle, //
    safe_vector<u8_t, typename std::allocator_traits<allocator_type_>::template rebind_alloc<u8_t>>                //
        &byte_to_class_buffer,                                                                                     //
    safe_vector<error_cost_t, typename std::allocator_traits<allocator_type_>::template rebind_alloc<error_cost_t>> //
        &class_substitution_costs_buffer,                                                                          //
    cuda_executor_t const &executor, gpu_specs_t specs) noexcept {

    using char_t = char;
    using task_t = cuda_similarity_task<char_t>;
    using device_substituter_t = error_costs_classes_in_cuda_shared_memory_t;
    constexpr bool is_affine_k = is_same_type<gap_costs_type_, affine_gap_costs_t>::value;
    [[maybe_unused]] constexpr size_t count_diagonals_k = is_affine_k ? 7 : 3;

    // Create the engine-owned timing events on first use; the kernel table is already resolved by the caller.
    CUresult timer_error = timer.ensure_created();
    if (timer_error != CUDA_SUCCESS) return make_cuda_status(timer_error);
    auto &tasks = buffers.tasks_;

    // Record the start event
    CUresult start_event_error = timer.record_start(executor.stream());
    if (start_event_error != CUDA_SUCCESS) return make_cuda_status(start_event_error);

    {
        // Upload the compact substituter into device-accessible memory. Every block will later mirror
        // these tiny buffers into its own static shared memory for the inner-loop lookups.
        constexpr size_t classes_count_k = error_costs_32x32_t::classes_count_k;
        byte_to_class_buffer.clear();
        class_substitution_costs_buffer.clear();
        if (byte_to_class_buffer.try_resize(256) == status_t::bad_alloc_k) return {status_t::bad_alloc_k};
        if (class_substitution_costs_buffer.try_resize(classes_count_k * classes_count_k) == status_t::bad_alloc_k)
            return {status_t::bad_alloc_k};
        for (size_t i = 0; i < 256; ++i) byte_to_class_buffer[i] = substituter.byte_to_class[i];
        for (size_t i = 0; i < classes_count_k; ++i)
            for (size_t j = 0; j < classes_count_k; ++j)
                class_substitution_costs_buffer[i * classes_count_k + j] = substituter.class_substitution_costs[i][j];

        device_substituter_t device_substituter;
        device_substituter.byte_to_class = byte_to_class_buffer.data();
        device_substituter.class_substitution_costs = class_substitution_costs_buffer.data();

        // Dyadic length-bucket router (shared with the Levenshtein engine): one `cub::DeviceRadixSort` lays every
        // task out in the contiguous `[register | batch | device]` order and one `cub::DeviceRunLengthEncode` fills
        // the host-side per-tier counts. The register kernel consumes the register prefix, the batch kernel the
        // next slice, and the warp/device grouping the remaining tail. The final scatter reads `tasks_` by
        // `result_offset`, which the router preserves (it carries `result_offset` along with each reordered task).
        size_t count_register_level_tasks = 0;
        size_t count_batch_level_tasks = 0;
        {
            size_t weighted_tier_counts[weighted_tier_count_k] = {};
            int const begin_bit = levenshtein_original_index_shift_k;
            int const end_bit = levenshtein_tier_priority_shift_k + levenshtein_tier_priority_bits_k;
            weighted_dense_tier_functor<char_t> dense_tier_functor {nullptr, 0u};
            cuda_status_t const router_status = cuda_route_tasks_into_tiers_(
                buffers, tier_keys, tier_values, tier_rle, kernel_table.build_tier_keys, kernel_table.gather_tasks,
                u32_t {0}, dense_tier_functor, begin_bit, end_bit, weighted_tier_count_k, executor,
                weighted_tier_counts);
            if (router_status.status != status_t::success_k) return router_status;
            count_register_level_tasks = weighted_tier_counts[weighted_tier_register_k];
            count_batch_level_tasks = weighted_tier_counts[weighted_tier_batch_k];
        }
        if (count_register_level_tasks) {
            kernel_shape_t const &shape = kernel_table.register_score;
            task_t *tasks_ptr = tasks.data();
            void *thread_level_kernel_args[4] = {(void *)(&tasks_ptr), (void *)(&count_register_level_tasks),
                                                 (void *)(&device_substituter), (void *)(&gap_costs)};
            unsigned const blocks_per_grid = shape.blocks_per_multiprocessor * specs.streaming_multiprocessors;
            CUresult launch_error = cuda_launch_t {}
                                        .grid(blocks_per_grid)
                                        .block(256)
                                        .shared(0)
                                        .stream(executor.stream())
                                        .launch(shape.function, thread_level_kernel_args);
            if (launch_error != CUDA_SUCCESS) return make_cuda_status(launch_error);
        }
        if (count_batch_level_tasks) {
            // Thread-per-pair batch tier: same kernel, larger per-thread storage; consumes the `[register, register+batch)`
            // prefix of the remainder. Below the measured ~512 GPU crossover this beats the tiled wavefront below.
            kernel_shape_t const &shape = kernel_table.batch_score;
            task_t *tasks_ptr = tasks.data() + count_register_level_tasks;
            void *thread_level_kernel_args[4] = {(void *)(&tasks_ptr), (void *)(&count_batch_level_tasks),
                                                 (void *)(&device_substituter), (void *)(&gap_costs)};
            unsigned const blocks_per_grid = shape.blocks_per_multiprocessor * specs.streaming_multiprocessors;
            CUresult launch_error = cuda_launch_t {}
                                        .grid(blocks_per_grid)
                                        .block(256)
                                        .shared(0)
                                        .stream(executor.stream())
                                        .launch(shape.function, thread_level_kernel_args);
            if (launch_error != CUDA_SUCCESS) return make_cuda_status(launch_error);
        }

        size_t const count_thread_level_tasks = count_register_level_tasks + count_batch_level_tasks;
        size_t warp_group_count = 0;
        [[maybe_unused]] auto [device_level_tasks, warp_level_tasks, empty_tasks] = warp_tasks_grouping<task_t>(
            {tasks.data() + count_thread_level_tasks, tasks.size() - count_thread_level_tasks}, specs,
            executor.stream(), buffers.cub_buffer_, buffers.warp_partition_scratch_, buffers.warp_split_counts_,
            buffers.warp_group_keys_, buffers.warp_group_sizes_, buffers.warp_group_descriptors_, warp_group_count);

        if (device_level_tasks.size()) {
            // Device tier -> the same tiled register-micro-tile kernel as Levenshtein, here in NW/SW form: maximize
            // (`sz_maximize_score_k`), the 32-class substituter, and signed results. Linear uses one row frontier +
            // corners; affine (Gotoh) carries M + H left-edges + M corners. `buffers.diagonals_u64_buffer_` holds
            // every pair's frontier slice padded to the batch's largest pair, run at the widest signed cell width.
            // NW/SW floor at i32 (GPU scalar i16 min/add promote to 32-bit anyway, so a narrower frontier buys nothing).
            static constexpr unsigned tiled_warps_per_block_k = 8, tiled_tile_side_k = 128;
            device_tier_maxima_t maxima;
            cuda_status_t const maxima_status = reduce_device_tier_maxima_<char_t>(
                device_level_tasks.data(), device_level_tasks.size(), buffers.cub_buffer_,
                buffers.device_maxima_scratch_, executor.stream(), maxima);
            if (maxima_status.status != status_t::success_k) return maxima_status;
            unsigned const max_bytes_per_cell = sz_max_of_two(maxima.max_bytes_per_cell,
                                                              static_cast<u32_t>(sizeof(i32_t)));
            u32_t const max_columns = divide_round_up<u32_t>(maxima.max_longer, tiled_tile_side_k);
            u32_t const max_padded_rows = round_up_to_multiple<u32_t>(maxima.max_shorter, tiled_tile_side_k);
            u32_t const row_stride = max_padded_rows + 1, corner_stride = max_columns;
            unsigned const grid_columns_blocks = divide_round_up<unsigned>(max_columns, tiled_warps_per_block_k);

            // Resolve the per-signed-cell-width tiled score + frontier-seed `CUfunction`s and launch the shared
            // batched driver; the linear and affine frontier carvings live in the driver, gated on `affine_k`.
            auto const launch_batched = [&]<typename score_type_>() noexcept -> cuda_status_t {
                cudaFunction_t const init_fn = sizeof(score_type_) == 8 ? kernel_table.tiled_init_i64.function
                                                                        : kernel_table.tiled_init_i32.function;
                cudaFunction_t const score_fn = sizeof(score_type_) == 8 ? kernel_table.tiled_i64.function
                                                                         : kernel_table.tiled_i32.function;
                return cuda_launch_tiled_device_tier_<score_type_, is_affine_k>(
                    buffers, device_level_tasks, row_stride, corner_stride, grid_columns_blocks, init_fn, score_fn,
                    device_substituter, gap_costs, executor);
            };

            cuda_status_t tiled_status {status_t::success_k, cudaSuccess};
            if (max_bytes_per_cell >= sizeof(i64_t)) tiled_status = launch_batched.template operator()<i64_t>();
            else tiled_status = launch_batched.template operator()<i32_t>();
            if (tiled_status.status != status_t::success_k) return tiled_status;
        }

        // Now process remaining warp-level tasks via the shared per-descriptor launch loop (densest groups first by
        // the sort order): the host reads only the small descriptor array, never a device task.
        if (warp_group_count) {
            cuda_status_t const warp_status = cuda_launch_warp_groups_(
                buffers, device_level_tasks, warp_group_count, kernel_table.warp_i16, kernel_table.warp_i32,
                sizeof(i32_t), device_substituter, gap_costs, specs, executor);
            if (warp_status.status != status_t::success_k) return warp_status;
        }

        // Single trailing sync: drains every group's kernels enqueued during the ENQUEUE phase before we read results.
        CUresult stop_event_error = timer.record_stop(executor.stream());
        if (stop_event_error != CUDA_SUCCESS) return make_cuda_status(stop_event_error);
        CUresult execution_error = timer.synchronize(executor.stream());
        if (execution_error != CUDA_SUCCESS) return make_cuda_status(execution_error);
        float execution_milliseconds = timer.elapsed_milliseconds();
        return {status_t::success_k, cudaSuccess, CUDA_SUCCESS, execution_milliseconds};
    }
}

/**
 *  @brief Shared cross-product driver for the weighted NW/SW GPU engines: builds one task per live (query, candidate)
 *         cell, runs the container-independent GPU pipeline, and scatters each result into the row-major matrix. For
 *         @b symmetric_k only the lower triangle (incl. the diagonal) is built and each result is mirrored. A free
 *         function taking engine state @b by-reference (the no-inheritance substrate); NW and SW call it with their
 *         own `locality_` (folded into the kernel table) - the bodies are otherwise identical.
 */
template <typename gap_costs_type_, sz_similarity_locality_t locality_, sz_capability_t capability_,
          typename allocator_type_, typename queries_type_, typename candidates_type_, typename results_type_>
cuda_status_t cuda_weighted_cross_(                                                                              //
    cuda_cross_buffers<cuda_similarity_task<char>, allocator_type_> &buffers, error_costs_32x32_t const &substituter, //
    gap_costs_type_ const &gap_costs, cuda_timer_t &timer,                                                       //
    safe_vector<u64_t, typename std::allocator_traits<allocator_type_>::template rebind_alloc<u64_t>> &tier_keys, //
    safe_vector<u32_t, typename std::allocator_traits<allocator_type_>::template rebind_alloc<u32_t>> &tier_values, //
    safe_vector<u32_t, typename std::allocator_traits<allocator_type_>::template rebind_alloc<u32_t>> &tier_rle, //
    safe_vector<u8_t, typename std::allocator_traits<allocator_type_>::template rebind_alloc<u8_t>>                //
        &byte_to_class_buffer,                                                                                     //
    safe_vector<error_cost_t, typename std::allocator_traits<allocator_type_>::template rebind_alloc<error_cost_t>> //
        &class_substitution_costs_buffer,                                                                          //
    queries_type_ const &queries, candidates_type_ const &candidates, results_type_ &&results,                   //
    cross_similarities_t cross_kind, cuda_executor_t const &executor, gpu_specs_t specs) noexcept {

    using char_t = char;
    using task_t = cuda_similarity_task<char_t>;

    bool const is_symmetric = cross_kind == cross_similarities_t::symmetric_k;
    size_t const queries_count = queries.size();
    size_t const candidates_count = candidates.size();
    size_t const row_stride = results.row_stride;
    size_t const live_cells = is_symmetric ? queries_count * (queries_count + 1) / 2
                                           : queries_count * candidates_count;

    // No `clear()`: every live cell is fully overwritten by the materialization below, and a same-size
    // `try_resize` of a trivially-constructible task does zero (re-)construction - so the host never touches
    // the (large) task array, which would otherwise force a fault-driven unified-memory ping-pong.
    auto &tasks = buffers.tasks_;
    if (tasks.try_resize_uninitialized(live_cells) == status_t::bad_alloc_k) return {status_t::bad_alloc_k};

    // Ensure inputs are device-accessible (Unified/Device memory). Both sides come from contiguous
    // tapes/arrays, so we validate the base pointers of the first element once (covers the whole tape).
    if (queries_count != 0 && candidates_count != 0) {
        if (!is_device_accessible_memory((void const *)queries[0].data()) ||
            !is_device_accessible_memory((void const *)candidates[0].data()))
            return {status_t::device_memory_mismatch_k, cudaSuccess};
    }

    // Device-side materialization for BOTH all-pairs and symmetric (one thread per live cell; the symmetric
    // path maps the flat cell index into the lower triangle). Build the O(queries+candidates) descriptors on
    // the host; the kernel reads them from unified memory. Weighted cells start at 2 bytes (signed scores).
    if (buffers.query_descs_.try_resize(queries_count) == status_t::bad_alloc_k ||
        buffers.candidate_descs_.try_resize(candidates_count) == status_t::bad_alloc_k)
        return {status_t::bad_alloc_k};
    for (size_t query_index = 0; query_index < queries_count; ++query_index)
        buffers.query_descs_[query_index] = {queries[query_index].data(), queries[query_index].length()};
    for (size_t candidate_index = 0; candidate_index < candidates_count; ++candidate_index)
        buffers.candidate_descs_[candidate_index] = {candidates[candidate_index].data(),
                                                     candidates[candidate_index].length()};

    auto [kernel_table, kernels_status] = cuda_weighted_kernels_<gap_costs_type_, locality_, capability_>();
    if (kernels_status.status != status_t::success_k) return kernels_status;

    span<char const> *queries_ptr = buffers.query_descs_.data(), *candidates_ptr = buffers.candidate_descs_.data();
    task_t *tasks_ptr = tasks.data();
    error_cost_magnitude_t substitute_magnitude = substituter.magnitude(), gap_magnitude = gap_costs.magnitude();
    sz_similarity_gaps_t gap_type_value = gap_type<gap_costs_type_>();
    bytes_per_cell_t min_bytes_per_cell = two_bytes_per_cell_k;
    cross_similarities_t cross_kind_copy = cross_kind;
    gap_costs_type_ gap_costs_copy = gap_costs;
    gpu_specs_t specs_copy = specs;
    void *materialize_args[13] = {(void *)&tasks_ptr,       (void *)&queries_ptr,          (void *)&candidates_ptr,
                                  (void *)&queries_count,   (void *)&candidates_count,     (void *)&row_stride,
                                  (void *)&cross_kind_copy, (void *)&substitute_magnitude, (void *)&gap_magnitude,
                                  (void *)&gap_type_value,  (void *)&min_bytes_per_cell,   (void *)&gap_costs_copy,
                                  (void *)&specs_copy};
    unsigned const block = 256;
    unsigned const grid = static_cast<unsigned>((live_cells + block - 1) / block);
    CUresult materialize_error = cuda_launch_t {}
                                     .grid(grid)
                                     .block(block)
                                     .shared(0)
                                     .stream(executor.stream())
                                     .launch(kernel_table.materialize_tasks.function, materialize_args);
    if (materialize_error != CUDA_SUCCESS) return make_cuda_status(materialize_error);
    // No host sync after materialization: the dyadic router tiers `tasks_` entirely on-device (radix sort +
    // gather + run-length-encode), all enqueued on the same stream after this kernel. The router performs the single
    // stream sync it needs internally before reading its small per-tier counts back on the host.

    cuda_status_t status = cuda_weighted_run_trampoline_<gap_costs_type_, allocator_type_>(
        buffers, substituter, gap_costs, kernel_table, timer, tier_keys, tier_values, tier_rle, byte_to_class_buffer,
        class_substitution_costs_buffer, executor, specs);
    if (status.status != status_t::success_k) return status;

    // Scatter on the device when the output is device-accessible (the common unified-memory case): the device
    // scatter kernel writes each result by its `result_offset` so the host never reads the large task array back.
    // When the output is host-only, the same kernel scatters into a device-resident dense staging matrix (laid out
    // at the caller's `row_stride` so the precomputed offsets stay valid), then a single strided `cudaMemcpy2DAsync`
    // strides the valid `rows x columns` region into the host matrix - no per-cell host loop, fully stream-async.
    using results_value_t = typename std::remove_reference_t<results_type_>::value_type;
    if (tasks.size() && is_device_accessible_memory((void const *)results.data)) {
        results_value_t *results_ptr = results.data;
        task_t const *tasks_const_ptr = tasks.data();
        size_t tasks_size = tasks.size();
        void *scatter_args[3] = {(void *)&tasks_const_ptr, (void *)&tasks_size, (void *)&results_ptr};
        unsigned const scatter_grid = static_cast<unsigned>((tasks_size + block - 1) / block);
        CUresult scatter_error = cuda_launch_t {}
                                     .grid(scatter_grid)
                                     .block(block)
                                     .shared(0)
                                     .stream(executor.stream())
                                     .launch(kernel_table.scatter_results.function, scatter_args);
        if (scatter_error != CUDA_SUCCESS) return make_cuda_status(scatter_error);
        // Single completion barrier: the scatter writes the caller's output buffer and results must be host-readable
        // on return, so removing it would let the host read `results.data` before the stream-ordered scatter completes.
        if (cudaStreamSynchronize(executor.stream()) != cudaSuccess) return make_cuda_status(cudaGetLastError());
    }
    else if (tasks.size()) {
        cuda_status_t const fallback_status = cuda_scatter_results_to_host_strided_<task_t, results_value_t>(
            buffers, tasks.data(), tasks.size(), results, executor, block);
        if (fallback_status.status != status_t::success_k) return fallback_status;
    }
    return status;
}

/**
 *  @brief Dispatches baseline Needleman Wunsch (global) algorithm to the GPU. Independent struct (no base class):
 *         holds its own `cuda_cross_buffers` + substituter/costs and forwards to the shared weighted free functions
 *         with @ref sz_similarity_global_k.
 */
template <typename gap_costs_type_, typename allocator_type_, sz_capability_t capability_>
struct needleman_wunsch_scores<error_costs_32x32_t, gap_costs_type_, allocator_type_, capability_,
                               std::enable_if_t<capability_ & sz_cap_cuda_k>> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;
    using byte_to_class_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<u8_t>;
    using class_costs_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<error_cost_t>;
    using tier_keys_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<u64_t>;
    using tier_values_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<u32_t>;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    using task_t = cuda_similarity_task<char_t>;
    using buffers_t = cuda_cross_buffers<task_t, allocator_t>;

    error_costs_32x32_t substituter_ {};
    gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    /** @brief The cross-product device buffer bundle shared with the other GPU engines (passed by reference). */
    buffers_t buffers_ {alloc_};

    /** @brief Grow-only dyadic-router scratch (shared machinery with Levenshtein): radix-sort keys, permutation
     *         values, and run-length-encode outputs feeding @ref cuda_route_tasks_into_tiers_. */
    safe_vector<u64_t, tier_keys_allocator_t> tier_keys_ {alloc_};
    safe_vector<u32_t, tier_values_allocator_t> tier_values_ {alloc_};
    safe_vector<u32_t, tier_values_allocator_t> tier_rle_ {alloc_};
    safe_vector<u8_t, byte_to_class_allocator_t> byte_to_class_buffer_ {alloc_};
    safe_vector<error_cost_t, class_costs_allocator_t> class_substitution_costs_buffer_ {alloc_};
    cuda_timer_t timer_ {};

    needleman_wunsch_scores(error_costs_32x32_t subs = {}, gap_costs_t gaps = {},
                            allocator_t const &alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    needleman_wunsch_scores(needleman_wunsch_scores const &) = delete;
    needleman_wunsch_scores &operator=(needleman_wunsch_scores const &) = delete;
    needleman_wunsch_scores(needleman_wunsch_scores &&) noexcept = default;
    needleman_wunsch_scores &operator=(needleman_wunsch_scores &&) noexcept = default;

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    cuda_status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                             strided_rows<value_type_> results, cuda_executor_t const &executor = {},
                             gpu_specs_t specs = {}) noexcept {
        return cuda_weighted_cross_<gap_costs_t, locality_k, capability_k>(
            buffers_, substituter_, gap_costs_, timer_, tier_keys_, tier_values_, tier_rle_, byte_to_class_buffer_,
            class_substitution_costs_buffer_, queries, candidates, results, cross_similarities_t::all_pairs_k, executor,
            specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    cuda_status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                             cuda_executor_t const &executor = {}, gpu_specs_t specs = {}) noexcept {
        return cuda_weighted_cross_<gap_costs_t, locality_k, capability_k>(
            buffers_, substituter_, gap_costs_, timer_, tier_keys_, tier_values_, tier_rle_, byte_to_class_buffer_,
            class_substitution_costs_buffer_, sequences, sequences, results, cross_similarities_t::symmetric_k, executor,
            specs);
    }
};

/**
 *  @brief Dispatches baseline Smith Waterman (local) algorithm to the GPU. Independent struct (no base class):
 *         holds its own `cuda_cross_buffers` + substituter/costs and forwards to the shared weighted free functions
 *         with @ref sz_similarity_local_k.
 */
template <typename gap_costs_type_, typename allocator_type_, sz_capability_t capability_>
struct smith_waterman_scores<error_costs_32x32_t, gap_costs_type_, allocator_type_, capability_,
                             std::enable_if_t<capability_ & sz_cap_cuda_k>> {

    using char_t = char;
    using substituter_t = error_costs_32x32_t;
    using gap_costs_t = gap_costs_type_;
    using allocator_t = allocator_type_;
    using byte_to_class_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<u8_t>;
    using class_costs_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<error_cost_t>;
    using tier_keys_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<u64_t>;
    using tier_values_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<u32_t>;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_local_k;
    static constexpr sz_capability_t capability_k = capability_;

    using task_t = cuda_similarity_task<char_t>;
    using buffers_t = cuda_cross_buffers<task_t, allocator_t>;

    error_costs_32x32_t substituter_ {};
    gap_costs_t gap_costs_ {};
    allocator_t alloc_ {};

    /** @brief The cross-product device buffer bundle shared with the other GPU engines (passed by reference). */
    buffers_t buffers_ {alloc_};

    /** @brief Grow-only dyadic-router scratch (shared machinery with Levenshtein): radix-sort keys, permutation
     *         values, and run-length-encode outputs feeding @ref cuda_route_tasks_into_tiers_. */
    safe_vector<u64_t, tier_keys_allocator_t> tier_keys_ {alloc_};
    safe_vector<u32_t, tier_values_allocator_t> tier_values_ {alloc_};
    safe_vector<u32_t, tier_values_allocator_t> tier_rle_ {alloc_};
    safe_vector<u8_t, byte_to_class_allocator_t> byte_to_class_buffer_ {alloc_};
    safe_vector<error_cost_t, class_costs_allocator_t> class_substitution_costs_buffer_ {alloc_};
    cuda_timer_t timer_ {};

    smith_waterman_scores(error_costs_32x32_t subs = {}, gap_costs_t gaps = {},
                          allocator_t const &alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_costs_(gaps), alloc_(alloc) {}

    smith_waterman_scores(smith_waterman_scores const &) = delete;
    smith_waterman_scores &operator=(smith_waterman_scores const &) = delete;
    smith_waterman_scores(smith_waterman_scores &&) noexcept = default;
    smith_waterman_scores &operator=(smith_waterman_scores &&) noexcept = default;

    template <typename queries_type_, typename candidates_type_, typename value_type_>
    cuda_status_t operator()(queries_type_ const &queries, candidates_type_ const &candidates,
                             strided_rows<value_type_> results, cuda_executor_t const &executor = {},
                             gpu_specs_t specs = {}) noexcept {
        return cuda_weighted_cross_<gap_costs_t, locality_k, capability_k>(
            buffers_, substituter_, gap_costs_, timer_, tier_keys_, tier_values_, tier_rle_, byte_to_class_buffer_,
            class_substitution_costs_buffer_, queries, candidates, results, cross_similarities_t::all_pairs_k, executor,
            specs);
    }

    /** @brief Symmetric self-similarity: one set scored against itself (lower triangle + mirror). */
    template <typename sequences_type_, typename value_type_>
    cuda_status_t operator()(sequences_type_ const &sequences, strided_rows<value_type_> results,
                             cuda_executor_t const &executor = {}, gpu_specs_t specs = {}) noexcept {
        return cuda_weighted_cross_<gap_costs_t, locality_k, capability_k>(
            buffers_, substituter_, gap_costs_, timer_, tier_keys_, tier_values_, tier_rle_, byte_to_class_buffer_,
            class_substitution_costs_buffer_, sequences, sequences, results, cross_similarities_t::symmetric_k, executor,
            specs);
    }
};

#pragma endregion

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SIMILARITIES_CUDA_CUH_
