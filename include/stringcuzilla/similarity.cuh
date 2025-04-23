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
#ifndef STRINGZILLA_SIMILARITIES_CUH_
#define STRINGZILLA_SIMILARITIES_CUH_

#include "stringcuzilla/types.cuh"
#include "stringcuzilla/similarity.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>        // `cuda::pipeline`
#include <cooperative_groups.h> // `cooperative_groups::this_grid()`

namespace ashvardanian {
namespace stringzilla {

#pragma region - Algorithm Building Blocks

/**
 *  @brief GPU adaptation of the `scorer` on CUDA, avoiding warp-level shuffles and DPX.
 *  @note Uses 32-bit `uint` counter to iterate through the string slices, so it can't be over 4 billion characters.
 */
template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_, sz_capability_t capability_>
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, objective_,
                   sz_similarity_global_k, capability_, std::enable_if_t<capability_ & sz_cap_cuda_k>> {

    using first_iterator_t = first_iterator_type_;
    using second_iterator_t = second_iterator_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_capability_t capability_k = capability_;

    using first_char_t = typename std::iterator_traits<first_iterator_t>::value_type;
    using second_char_t = typename std::iterator_traits<second_iterator_t>::value_type;
    static_assert(is_same_type<first_char_t, second_char_t>::value, "String characters must be of the same type.");
    using char_t = typename std::remove_cvref<first_char_t>::type;

    using warp_scorer_t = tile_scorer<first_iterator_t, second_iterator_t, score_t, substituter_t, objective_k,
                                      sz_similarity_global_k, capability_k>;

  protected:
    substituter_t substituter_;
    error_cost_t gap_cost_ {1};
    score_t last_cell_ {0};

    __forceinline__ __device__ score_t pick_best(score_t a, score_t b) const noexcept {
        if constexpr (objective_ == sz_minimize_distance_k) { return std::min(a, b); }
        else { return std::max(a, b); }
    }

  public:
    __forceinline__ __device__ tile_scorer(substituter_t substituter, error_cost_t gap_cost) noexcept
        : substituter_(substituter), gap_cost_(gap_cost) {}

    /**
     *  @brief Initializes a boundary value within a certain diagonal.
     *  @note Should only be called for the diagonals outside of the bottom-right triangle.
     *  @note Should only be called for the top row and left column of the matrix.
     */
    __forceinline__ __device__ void init(score_t &cell, sz_size_t diagonal_index) const noexcept {
        cell = gap_cost_ * diagonal_index;
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
            score_t if_deletion_or_insertion = pick_best(pre_deletion, pre_insertion) + gap_cost_;
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
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, objective_,
                   sz_similarity_local_k, capability_, std::enable_if_t<capability_ & sz_cap_cuda_k>> {

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

    using warp_scorer_t = tile_scorer<first_iterator_t, second_iterator_t, score_t, substituter_t, objective_k,
                                      sz_similarity_local_k, capability_k>;

  protected:
    substituter_t substituter_;
    error_cost_t gap_cost_ {1};
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
    __forceinline__ __device__ tile_scorer(substituter_t substituter, error_cost_t gap_cost) noexcept
        : substituter_(substituter), gap_cost_(gap_cost) {}

    /**
     *  @brief Initializes a boundary value within a certain diagonal.
     *  @note Should only be called for the diagonals outside of the bottom-right triangle.
     *  @note Should only be called for the top row and left column of the matrix.
     */
    __forceinline__ __device__ void init(score_t &cell, sz_size_t diagonal_index) const noexcept { cell = 0; }

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
            score_t if_deletion_or_insertion = pick_best(pre_deletion, pre_insertion) + gap_cost_;
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

#if SZ_USE_KEPLER

/**
 *  @brief GPU adaptation of the `scorer` - Minimizes Global Levenshtein distance.
 *  @note Requires Kepler generation GPUs to handle 4x `u8` scores at a time.
 *
 *  Relies on following instruction families to output 4x @b `u8` scores per call:
 *  - @b `prmt` to shuffle bytes in 32 bit registers.
 *  - @b `vmax4,vmin4,vadd4` video-processing instructions.
 */
template <>
struct tile_scorer<char const *, char const *, sz_u8_t, error_costs_uniform_t, sz_minimize_distance_k,
                   sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, sz_u8_t, error_costs_uniform_t, sz_minimize_distance_k,
                         sz_similarity_global_k, sz_cap_cuda_k> {

    using warp_scorer_t::tile_scorer; // Make the constructors visible

    __forceinline__ __device__ void operator()(                                 //
        char const *first_slice, char const *second_slice,                      //
        uint const tasks_offset, uint const tasks_step, uint const tasks_count, // ! Unlike CPU, uses `uint`
        sz_u8_t const *scores_pre_substitution, sz_u8_t const *scores_pre_insertion, sz_u8_t const *scores_pre_deletion,
        sz_u8_t *scores_new) noexcept {

        sz_u8_t const gap_cost = this->gap_cost_;
        _sz_assert(gap_cost == 1);

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
            // Adding one to it will make it 0x00 for each matching byte, and 0x01 for each non-matching byte.
            // Perfect for substitution cost!
            cost_of_substitution_vec.u32 = __vadd4(__vcmpeq4(first_vec.u32, second_vec.u32), 0x01010101);
            if_substitution_vec.u32 = __vaddus4(pre_substitution_vec.u32, cost_of_substitution_vec.u32);
            if_deletion_or_insertion_vec.u32 =
                __vaddus4(__vminu4(pre_deletion_vec.u32, pre_insertion_vec.u32), 0x01010101);
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
struct tile_scorer<char const *, char const *, sz_u16_t, error_costs_uniform_t, sz_minimize_distance_k,
                   sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, sz_u16_t, error_costs_uniform_t, sz_minimize_distance_k,
                         sz_similarity_global_k, sz_cap_cuda_k> {
    using warp_scorer_t::tile_scorer; // Make the constructors visible

    __forceinline__ __device__ void operator()(                                 //
        char const *first_slice, char const *second_slice,                      //
        uint const tasks_offset, uint const tasks_step, uint const tasks_count, // ! Unlike CPU, uses `uint`
        sz_u16_t const *scores_pre_substitution, sz_u16_t const *scores_pre_insertion,
        sz_u16_t const *scores_pre_deletion, sz_u16_t *scores_new) noexcept {

        sz_u16_t const gap_cost = this->gap_cost_;
        _sz_assert(gap_cost == 1);

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
            // Adding one to it will make it 0x0000 for each matching byte-pair,
            // and 0x0001 for each non-matching byte-pair. Perfect for substitution cost!
            cost_of_substitution_vec.u32 = __vadd2(__vcmpeq2(first_vec.u32, second_vec.u32), 0x00010001);
            if_substitution_vec.u32 = __vaddus2(pre_substitution_vec.u32, cost_of_substitution_vec.u32);
            if_deletion_or_insertion_vec.u32 =
                __vaddus2(__vminu2(pre_deletion_vec.u32, pre_insertion_vec.u32), 0x00010001);
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
struct tile_scorer<char const *, char const *, sz_u32_t, error_costs_uniform_t, sz_minimize_distance_k,
                   sz_similarity_global_k, sz_caps_ck_k>
    : public tile_scorer<char const *, char const *, sz_u32_t, error_costs_uniform_t, sz_minimize_distance_k,
                         sz_similarity_global_k, sz_cap_cuda_k> {
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
    typename score_type_ = sz_size_t,                            //
    typename substituter_type_ = error_costs_uniform_t,          //
    sz_similarity_objective_t objective_ = sz_maximize_score_k,  //
    sz_similarity_locality_t locality_ = sz_similarity_global_k, //
    sz_capability_t capability_ = sz_cap_cuda_k                  //
    >
__global__ void _score_across_cuda_device(                     //
    char_type_ const *shorter_ptr, index_type_ shorter_length, //
    char_type_ const *longer_ptr, index_type_ longer_length,   //
    score_type_ *result_ptr, score_type_ *diagonals_ptr,       //
    substituter_type_ const *substituter_ptr, error_cost_t const gap_cost) noexcept {

    namespace cg = cooperative_groups;

    _sz_assert(shorter_length > 0);
    _sz_assert(longer_length > 0);
    _sz_assert(shorter_length <= longer_length);
    using char_t = char_type_;
    using index_t = index_type_;
    using score_t = score_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    static constexpr sz_similarity_objective_t objective_k = sz_minimize_distance_k;
    using substituter_t = error_costs_uniform_t;
    using warp_scorer_t =
        tile_scorer<char_t const *, char_t const *, score_t, substituter_t, objective_k, locality_k, capability_k>;

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
    substituter_t const substituter;
    error_cost_t const gap_cost = 1;
    warp_scorer_t diagonal_aligner {substituter, gap_cost};
    if (is_main_thread) {
        diagonal_aligner.init(previous_scores[0], 0);
        diagonal_aligner.init(current_scores[0], 1);
        diagonal_aligner.init(current_scores[1], 1);
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
            diagonal_aligner.init(next_scores[0], next_diagonal_index);
            diagonal_aligner.init(next_scores[next_diagonal_length - 1], next_diagonal_index);
        }
        // Guarantee that all the writes have finished, before progressing to the next diagonal.
        grid.sync();

        // Perform a circular rotation of those buffers, to reuse the memory.
        score_t *temporary = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = temporary;
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
        if (is_main_thread) diagonal_aligner.init(next_scores[next_diagonal_length - 1], next_diagonal_index);

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

        // Perform a circular rotation of those buffers, to reuse the memory, this time, with a shift,
        // dropping the first element in the current array.
        score_t *temporary = previous_scores;
        // ! Drop the first entry among the current distances.
        // ! Assuming every next diagonal is shorter by one element, we don't need a full-blown `sz_move`.
        // ! to shift the array by one element.
        previous_scores = current_scores + 1;
        current_scores = next_scores;
        next_scores = temporary;
    }

    // Export one result per each block.
    if (is_main_thread) *result_ptr = diagonal_aligner.score();
}

/**
 *  @brief  Levenshtein edit distances algorithm evaluating the Dynamic Programming matrix
 *          @b three skewed (reverse) diagonals at a time on a GPU, leveraging CUDA for parallelization.
 *          Each pair of strings gets its own @b "block" of CUDA threads forming one @b warp and shared memory.
 *
 *  @param[in] tasks Tasks containing the strings and output locations.
 */
template < //
    typename task_type_,
    typename char_type_ = char,                                  //
    typename index_type_ = uint,                                 //
    typename score_type_ = sz_size_t,                            //
    typename substituter_type_ = error_costs_uniform_t,          //
    sz_similarity_objective_t objective_ = sz_maximize_score_k,  //
    sz_similarity_locality_t locality_ = sz_similarity_global_k, //
    sz_capability_t capability_ = sz_cap_cuda_k                  //
    >
__global__ void _score_on_each_cuda_warp(task_type_ *tasks, sz_size_t tasks_count) {

    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    using task_t = task_type_;
    using char_t = char_type_;
    using index_t = index_type_;
    using score_t = score_type_;
    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_similarity_objective_t objective_k = objective_;

    // Allocating shared memory is handled on the host side.
    extern __shared__ char shared_memory_buffer[];

    // We are computing N edit distances for N pairs of strings. Not a cartesian product!
    // Each block/warp may end up receiving a different number of strings.
    for (sz_size_t task_idx = blockIdx.x; task_idx < first_strings.size(); task_idx += gridDim.x) {
        task_t const &task = tasks[task_idx];
        char_t const *shorter_global = task.shorter_ptr;
        char_t const *longer_global = task.longer_ptr;
        sz_size_t const shorter_length = task.shorter_length;
        sz_size_t const longer_length = task.longer_length;
        score_t &result_ref = *task.result_ptr;

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
        warp_scorer_t diagonal_aligner {substituter_, gap_cost_};
        if (threadIdx.x == 0) {
            diagonal_aligner.init(previous_scores[0], 0);
            diagonal_aligner.init(current_scores[0], 1);
            diagonal_aligner.init(current_scores[1], 1);
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
                diagonal_aligner.init(next_scores[0], next_diagonal_index);
                diagonal_aligner.init(next_scores[next_diagonal_length - 1], next_diagonal_index);
            }
            __syncwarp();

            // Perform a circular rotation of those buffers, to reuse the memory.
            score_t *temporary = previous_scores;
            previous_scores = current_scores;
            current_scores = next_scores;
            next_scores = temporary;
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
            if (threadIdx.x == 0) diagonal_aligner.init(next_scores[next_diagonal_length - 1], next_diagonal_index);

            __syncwarp();
            // ! In the central anti-diagonal band, we can't just set the `current_scores + 1` to `previous_scores`
            // ! for the circular shift, as we will end up spilling outside of the diagonal a few iterations later.
            // ! Assuming in-place `memmove` is tricky on the GPU, so we will copy the data.
            for (sz_size_t i = threadIdx.x; i + 1 < next_diagonal_length; i += blockDim.x)
                previous_scores[i] = current_scores[i + 1];
            __syncwarp();
            for (sz_size_t i = threadIdx.x; i < next_diagonal_length; i += blockDim.x)
                current_scores[i] = next_scores[i];
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

            // Perform a circular rotation of those buffers, to reuse the memory, this time, with a shift,
            // dropping the first element in the current array.
            score_t *temporary = previous_scores;
            // ! Drop the first entry among the current distances.
            // ! Assuming every next diagonal is shorter by one element, we don't need a full-blown `sz_move`.
            // ! to shift the array by one element.
            previous_scores = current_scores + 1;
            current_scores = next_scores;
            next_scores = temporary;
            __syncwarp();
        }

        // Export one result per each block.
        if (threadIdx.x == 0) result_ref = diagonal_aligner.score();
    }
}

#pragma endregion

#pragma region - Levenshtein Distance in CUDA

/** @brief Dispatches on @b `_levenshtein_in_cuda_warp` on the device side from the host side. */
template <                                       //
    sz_capability_t capability_ = sz_cap_cuda_k, //
    typename first_strings_type_,                //
    typename second_strings_type_,               //
    typename allocator_type_,                    //
    typename score_type_ = sz_size_t             //
    >
cuda_status_t _levenshtein_distances_implementation(                                      //
    first_strings_type_ const &first_strings, second_strings_type_ const &second_strings, //
    score_type_ *results, allocator_type_ const &allocator = {}, gpu_specs_t specs = {},  //
    cudaStream_t stream = 0) noexcept(false) {

    // We need to be able to copy these function arguments into GPU memory:
    static constexpr sz_capability_t capability_k = capability_;
    using first_strings_t = first_strings_type_;
    using second_strings_t = second_strings_type_;
    using allocator_t = allocator_type_;
    static_assert(std::is_trivially_copyable<first_strings_t>() && std::is_trivially_copyable<second_strings_t>(),
                  "The first and second strings must be trivially copyable types - consider `arrow_strings_view`.");
    using first_string_t = typename first_strings_t::value_type;
    using second_string_t = typename second_strings_t::value_type;
    static_assert(std::is_trivially_copyable<first_string_t>() && std::is_trivially_copyable<second_string_t>(),
                  "The first and second strings must be trivially copyable types - consider `span<char>`.");
    using first_char_t = typename first_string_t::value_type;
    using second_char_t = typename second_string_t::value_type;
    static_assert(sizeof(first_char_t) == sizeof(second_char_t), "Character types don't match");
    using char_t = typename std::remove_cvref<first_char_t>::type;
    using score_t = score_type_;

    // Make sure that we don't string pairs that are too large to fit 3 matrix diagonals into shared memory.
    // H100 Streaming Multiprocessor can have up to 128 active warps concurrently and only 256 KB of shared memory.
    // A100 SMs had only 192 KB. We can't deal with blocks that require more memory than the SM can provide.
    using similarity_memory_requirements_t = similarity_memory_requirements<sz_size_t, true>;
    sz_size_t shared_memory_per_block =
        _scores_diagonally_warp_shared_memory_requirement<false>(first_strings, second_strings, 1);
    if (shared_memory_per_block > specs.shared_memory_per_multiprocessor()) return {status_t::bad_alloc_k};

    // It may be the case that we've only received empty strings.
    if (shared_memory_per_block == 0) {
        for (sz_size_t i = 0; i < first_strings.size(); ++i)
            if (first_strings[i].length() == 0) { results[i] = second_strings[i].length(); }
            else if (second_strings[i].length() == 0) { results[i] = first_strings[i].length(); }
        return {status_t::success_k};
    }

    // In most cases we should be able to fit many blocks per SM.
    sz_size_t count_blocks_per_multiprocessor = specs.shared_memory_per_multiprocessor() / shared_memory_per_block;
    if (count_blocks_per_multiprocessor > specs.max_blocks_per_multiprocessor)
        count_blocks_per_multiprocessor = specs.max_blocks_per_multiprocessor;
    if (count_blocks_per_multiprocessor > first_strings.size()) count_blocks_per_multiprocessor = first_strings.size();
    _sz_assert(count_blocks_per_multiprocessor > 0);

    // Let's use all 32 threads in a warp.
    constexpr sz_size_t threads_per_block = 32u;
    sz_size_t const max_input_length = 1024u * 16u;
    auto warp_level_kernel = &_levenshtein_in_cuda_warp<first_strings_t, second_strings_t, score_t, capability_k>;
    void *warp_level_kernel_args[] = {
        (void *)&first_strings,
        (void *)&second_strings,
        (void *)&results,
        (void *)&max_input_length,
    };

    // On Volta and newer GPUs, there is an extra flag to be set to use more than 48 KB of shared memory per block.
    // CUDA reserves 1 KB of shared memory per thread block, so on H100 we can use up to 227 KB of shared memory.
    // https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#unified-shared-memory-l1-texture-cache
    cudaError_t attribute_error =
        cudaFuncSetAttribute(warp_level_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             specs.shared_memory_per_multiprocessor() - count_blocks_per_multiprocessor * 1024);
    if (attribute_error != cudaSuccess) return {status_t::unknown_k, attribute_error};

    // Create CUDA events for timing
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    // Record the start event
    cudaEventRecord(start_event, stream);

    // Enqueue the warp_level_kernel for execution:
    cudaError_t launch_error = cudaLaunchKernel(                                 //
        reinterpret_cast<void *>(warp_level_kernel),                             // Kernel function pointer
        dim3(count_blocks_per_multiprocessor * specs.streaming_multiprocessors), // Grid dimensions
        dim3(threads_per_block),                                                 // Block dimensions
        warp_level_kernel_args,                                                  // Array of kernel argument pointers
        shared_memory_per_block,                                                 // Shared memory per block (in bytes)
        stream);                                                                 // CUDA stream
    if (launch_error != cudaSuccess)
        if (launch_error == cudaErrorMemoryAllocation) { return {status_t::bad_alloc_k, launch_error}; }
        else { return {status_t::unknown_k, launch_error}; }

    // Fetch the execution error:
    float execution_milliseconds = 0;
    cudaError_t shorts_execution_error = cudaStreamSynchronize(stream);
    if (shorts_execution_error != cudaSuccess)
        if (shorts_execution_error == cudaErrorMemoryAllocation) {
            return {status_t::bad_alloc_k, shorts_execution_error, execution_milliseconds};
        }
        else { return {status_t::unknown_k, shorts_execution_error, execution_milliseconds}; }

    // Go through the results and check if any of them were too big to be processed
    // by the warp-level implementation, and should be processed by the `_score_across_cuda_device`.
    sz_size_t count_longer_strings = 0;
    sz_size_t longest_string_length = 0;
    for (sz_size_t i = 0; i < first_strings.size(); ++i) {
        count_longer_strings += results[i] == std::numeric_limits<score_t>::max();
        longest_string_length =
            sz_max_of_two(longest_string_length, sz_max_of_two(first_strings[i].length(), second_strings[i].length()));
    }

    if (count_longer_strings) {
        auto device_level_u16index_kernel = &_score_across_cuda_device<char_t, sz_u16_t, score_t, capability_k>;
        auto device_level_u32index_kernel = &_score_across_cuda_device<char_t, sz_u32_t, score_t, capability_k>;
        auto device_level_u64index_kernel = &_score_across_cuda_device<char_t, sz_u64_t, score_t, capability_k>;
        void *device_level_kernel_args[6];
        safe_vector<score_t, allocator_t> diagonals_buffer(allocator);
        if (diagonals_buffer.try_resize((longest_string_length + 1) * 3) == status_t::bad_alloc_k)
            return {status_t::bad_alloc_k};

        // We will enqueue many such kernels one after another, without waiting for a completion of the previous one.
        for (sz_size_t i = 0; i < first_strings.size(); ++i) {
            if (results[i] != std::numeric_limits<score_t>::max()) continue;
            // We need to process this string pair separately.
            auto const &first_global = first_strings[i];
            auto const &second_global = second_strings[i];

            // Pick the shorter and longer string.
            char_t const *shorter_ptr, *longer_ptr;
            sz_size_t shorter_length, longer_length;
            if (first_global.length() < second_global.length()) {
                shorter_ptr = first_global.data(), longer_ptr = second_global.data(),
                shorter_length = first_global.length(), longer_length = second_global.length();
            }
            else {
                shorter_ptr = second_global.data(), longer_ptr = first_global.data(),
                shorter_length = second_global.length(), longer_length = first_global.length();
            }
        }
    }

    // Wait until everything is done.
    cudaEventRecord(stop_event, stream);
    cudaEventElapsedTime(&execution_milliseconds, start_event, stop_event);
    cudaError_t longs_execution_error = cudaStreamSynchronize(stream);
    if (longs_execution_error != cudaSuccess)
        if (longs_execution_error == cudaErrorMemoryAllocation) {
            return {status_t::bad_alloc_k, longs_execution_error, execution_milliseconds};
        }
        else { return {status_t::unknown_k, longs_execution_error, execution_milliseconds}; }

    return {status_t::success_k, cudaSuccess, execution_milliseconds};
}

/**
 *  @brief  Dispatches baseline Levenshtein edit distance algorithm to the GPU.
 *          Before starting the kernels, bins them by size to maximize the number of blocks
 *          per grid that can run simultaneously, while fitting into the shared memory.
 */
template <typename char_type_, typename allocator_type_, sz_capability_t capability_>
struct levenshtein_distances<char_type_, allocator_type_, capability_, std::enable_if_t<capability_ & sz_cap_cuda_k>> {

    using char_t = char_type_;
    using allocator_t = allocator_type_;
    using scores_allocator_t = typename allocator_t::template rebind<sz_size_t>::other;

    struct task_t {
        char_t const *shorter_ptr = nullptr;
        size_t shorter_length = 0;
        char_t const *longer_ptr = nullptr;
        size_t longer_length = 0;
        size_t memory_requirement = 0;
        size_t original_index = 0;

        constexpr task_t(char_t const *first_ptr, size_t first_length, char_t const *second_ptr,
                         size_t second_length) noexcept {
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

    allocator_t allocator = {};

    levenshtein_distances(allocator_t const &allocator = {}) noexcept : allocator(allocator) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    cuda_status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                             results_type_ &&results, gpu_specs_t specs = {}, cudaStream_t stream = 0) const noexcept {

        score_t *const results_ptr = results.data();
        safe_vector<task_t, tasks_allocator_t> tasks(allocator_);
        if (!tasks.try_resize(first_strings.size()) == status_t::bad_alloc_k) return {status_t::bad_alloc_k};

        // Export all the tasks and sort them by decreasing memory requirement.
        using similarity_memory_requirements_t = similarity_memory_requirements<sz_size_t, is_signed_>;
        for (sz_size_t i = 0; i < first_strings.size(); ++i) {
            tasks[i] = task_t(                                      //
                first_strings[i].data(), first_strings[i].length(), //
                second_strings[i].data(), second_strings[i].length());
            tasks[i].original_index = i;
            tasks[i].memory_requirement = //
                similarity_memory_requirements_t(tasks[i].shorter_length, tasks[i].longer_length, 1, sizeof(char_t), 4)
                    .total;
        }
        std::sort(tasks.begin(), tasks.end(),
                  [](task_t const &a, task_t const &b) { return a.memory_requirement > b.memory_requirement; });

        // On very large inputs we will keep the diagonals in shared memory.
        safe_vector<sz_u64_t, scores_allocator_t> diagonals_u64_buffer(allocator_);
        auto device_level_u16_kernel = &_score_across_cuda_device<char_t, sz_u16_t, sz_u16_t, capability_k>;
        auto device_level_u32_kernel = &_score_across_cuda_device<char_t, sz_u32_t, sz_u32_t, capability_k>;
        auto device_level_u64_kernel = &_score_across_cuda_device<char_t, sz_u64_t, sz_u64_t, capability_k>;
        void *device_level_kernel_args[6];

        // Now we need to bin them based on the number of blocks per multiprocessor,
        // starting with problems, that can't fit into the memory of a single SM.
        size_t count_tasks_processed = 0;
        size_t count_tasks_for_entire_device = 0;
        for (; count_tasks_processed != tasks.size(); ++count_tasks_processed, ++count_tasks_for_entire_device) {
            // Check if we've finally reached small-enough inputs.
            task_t const &task = tasks[count_tasks_processed];
            size_t const requirement_with_one_warp = task.memory_requirement + specs.reserved_memory_per_block * 1;
            if (requirement_with_one_warp < specs.shared_memory_per_multiprocessor()) break;

            // As tasks decrease in size, this can only fail on first iteration.
            if (diagonals_u64_buffer.try_resize(task.max_diagonal_length() * 3) == status_t::bad_alloc_k)
                return {status_t::bad_alloc_k};
            device_level_kernel_args[0] = (void *)(task.shorter_ptr);
            device_level_kernel_args[1] = (void *)(&task.shorter_length);
            device_level_kernel_args[2] = (void *)(task.longer_ptr);
            device_level_kernel_args[3] = (void *)(&task.longer_length);
            device_level_kernel_args[4] = (void *)(results_ptr + i);
            device_level_kernel_args[5] = (void *)(diagonals_u64_buffer.data());

            // Pick the smallest fitting type for the diagonals.
            void *device_level_kernel = reinterpret_cast<void *>(device_level_u16_kernel);
            if (task.max_diagonal_length() >= std::numeric_limits<sz_u16_t>::max())
                device_level_kernel = reinterpret_cast<void *>(device_level_u32_kernel);
            if (task.max_diagonal_length() >= std::numeric_limits<sz_u32_t>::max())
                device_level_kernel = reinterpret_cast<void *>(device_level_u64_kernel);

            // TODO: We can be wiser about the dimensions of this grid.
            uint const random_block_size = 128;
            uint const random_blocks_per_multiprocessor = 32;
            cudaError_t launch_error = cudaLaunchCooperativeKernel(                       //
                reinterpret_cast<void *>(device_level_u64index_kernel),                   // Kernel function pointer
                dim3(random_blocks_per_multiprocessor * specs.streaming_multiprocessors), // Grid dimensions
                dim3(random_block_size),                                                  // Block dimensions
                device_level_kernel_args, // Array of kernel argument pointers
                0,                        // Shared memory per block (in bytes)
                stream);                  // CUDA stream
            if (launch_error != cudaSuccess)
                if (launch_error == cudaErrorMemoryAllocation) { return {status_t::bad_alloc_k, launch_error}; }
                else { return {status_t::unknown_k, launch_error}; }
        }

        // Now process remaining warp-level tasks, checking warp densities in reverse order.
        // From the highest possible number of warps per multiprocessor to the lowest.
        std::initializer_list<size_t> warps_per_multiprocessor_densities = {32, 16, 8, 4, 2, 1};
        auto warp_level_u8_kernel = &_score_on_each_cuda_warp<task_t, char_t, sz_u8_t, sz_u8_t, capability_k>;
        auto warp_level_u16_kernel = &_score_on_each_cuda_warp<task_t, char_t, sz_u16_t, sz_u16_t, capability_k>;
        size_t count_tasks_for_this_density = 0;
        for (size_t warps_per_multiprocessor_density : warps_per_multiprocessor_densities) {
            for (; count_tasks_processed != tasks.size(); ++count_tasks_processed, ++count_tasks_for_this_density) {
                // Check if the current warp density is still optimal.
                task_t const &task = tasks[count_tasks_processed];
                size_t const requirement_with_current_warp_density =
                    task.memory_requirement + specs.reserved_memory_per_block * warps_per_multiprocessor_density;
                if (requirement_with_current_warp_density > specs.shared_memory_per_multiprocessor()) break;
            }

            // Now check if any tasks of that size have been found and if their quantity is sufficient
            // to fill the entire device with warps. If we don't have enough tasks for this density,
            // part of our GPU will just sit idle...
            //
            //     Theoretically, that isn't true. On paper, several CUDA kernels can run concurrently
            //     on the same physical device. It makes sense, assuming the Multi-Instance GPU (MIG)
            //     virtualization feature on them. But even when pulling `cudaGetDeviceProperties`
            //     and logging `cudaDeviceProp::concurrentKernels` for the H200 GPU with 140 GB of VRAM,
            //     it states 1!
            //
            // Moreover, we need to overwrite the maximum addressable shared memory for that kernel
            // as the default limits our blocks (in our case, single-warp blocks) to 48 KB, while
            // device supports 4-5x more! Still, that API is synchronous and we must block the current
            // thread to what until the task completes to change the shared memory amount.

            device_level_kernel_args[0] = (void *)(task.shorter_ptr);
            device_level_kernel_args[1] = (void *)(&task.shorter_length);
            device_level_kernel_args[2] = (void *)(task.longer_ptr);
            device_level_kernel_args[3] = (void *)(&task.longer_length);
            device_level_kernel_args[4] = (void *)(results_ptr + i);
            device_level_kernel_args[5] = (void *)(diagonals_u64_buffer.data());

            // Pick the smallest fitting type for the diagonals.
            void *device_level_kernel = reinterpret_cast<void *>(warp_level_u8_kernel);
            if (task.max_diagonal_length() >= std::numeric_limits<sz_u8_t>::max())
                device_level_kernel = reinterpret_cast<void *>(warp_level_u16_kernel);
        }

        // Now that everything went well, export the results back into the `results` array.
        for (size_t i = 0; i < count_tasks_processed; ++i) {
            task_t const &task = tasks[i];
            results[task.original_index] = *task.result_ptr;
        }
    }
};

#pragma endregion

#pragma region - Needleman Wunsch Scores in CUDA

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
template <                                              //
    typename first_strings_type_,                       //
    typename second_strings_type_,                      //
    typename score_type_ = sz_ssize_t,                  //
    typename substituter_type_ = error_costs_256x256_t, //
    sz_capability_t capability_ = sz_cap_cuda_k         //
    >
__global__ void _needleman_wunsch_in_cuda_warp( //
    first_strings_type_ first_strings,          //
    second_strings_type_ second_strings,        //
    score_type_ *results_ptr,                   //
    error_cost_t gap_cost,                      //
    sz_size_t max_magnitude_change) {

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
    for (sz_size_t pair_idx = blockIdx.x; pair_idx < first_strings.size(); pair_idx += gridDim.x) {
        first_string_t const first_global = first_strings[pair_idx];
        second_string_t const second_global = second_strings[pair_idx];
        score_t &result_ref = results_ptr[pair_idx];

        // Skip empty strings.
        sz_size_t const first_length = first_global.length();
        sz_size_t const second_length = second_global.length();
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
        similarity_memory_requirements_t requirements(first_length, second_length,
                                                      static_cast<uint>(max_magnitude_change), sizeof(char_t), 4);

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
template <                                             //
    sz_capability_t capability_ = sz_cap_cuda_k,       //
    typename first_strings_type_,                      //
    typename second_strings_type_,                     //
    typename score_type_ = sz_ssize_t,                 //
    typename substituter_type_ = error_costs_256x256_t //
    >
cuda_status_t _needleman_wunsch_via_cuda_warp(                                            //
    first_strings_type_ const &first_strings, second_strings_type_ const &second_strings, //
    score_type_ *results,
    substituter_type_ const &substituter, //
    error_cost_t gap_cost = 1,            //
    gpu_specs_t specs = {}, cudaStream_t stream = 0) noexcept(false) {

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
    sz_size_t shared_memory_per_block = _scores_diagonally_warp_shared_memory_requirement<true>(
        first_strings, second_strings, substituter.max_magnitude_change());
    if (shared_memory_per_block > specs.shared_memory_per_multiprocessor()) return {status_t::bad_alloc_k};

    // It may be the case that we've only received empty strings.
    if (shared_memory_per_block == 0) {
        for (sz_size_t i = 0; i < first_strings.size(); ++i)
            if (first_strings[i].length() == 0) { results[i] = second_strings[i].length(); }
            else if (second_strings[i].length() == 0) { results[i] = first_strings[i].length(); }
        return {status_t::success_k};
    }

    // In most cases we should be able to fit many blocks per SM.
    sz_size_t count_blocks_per_multiprocessor = specs.shared_memory_per_multiprocessor() / shared_memory_per_block;
    if (count_blocks_per_multiprocessor > specs.max_blocks_per_multiprocessor)
        count_blocks_per_multiprocessor = specs.max_blocks_per_multiprocessor;
    if (count_blocks_per_multiprocessor > first_strings.size()) count_blocks_per_multiprocessor = first_strings.size();

    // Let's use all 32 threads in a warp.
    constexpr sz_size_t threads_per_block = 32u;
    sz_size_t const max_magnitude_change = substituter.max_magnitude_change();
    auto kernel =
        &_needleman_wunsch_in_cuda_warp<first_strings_t, second_strings_t, score_t, substituter_t, capability_k>;
    void *kernel_args[] = {
        (void *)&first_strings, (void *)&second_strings,       (void *)&results,
        (void *)&gap_cost,      (void *)&max_magnitude_change,
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
    cudaEventRecord(start_event, stream);

    // Enqueue the transfer of the substituter to the constant memory:
    cudaError_t copy_error = cudaMemcpyToSymbolAsync(_error_costs_in_cuda_constant_memory, (void const *)&substituter,
                                                     sizeof(substituter_t), 0, cudaMemcpyHostToDevice, stream);
    if (copy_error != cudaSuccess) return {status_t::unknown_k, copy_error};

    // Enqueue the kernel for execution:
    cudaError_t launch_error = cudaLaunchKernel(                                 //
        reinterpret_cast<void *>(kernel),                                        // Kernel function pointer
        dim3(count_blocks_per_multiprocessor * specs.streaming_multiprocessors), // Grid dimensions
        dim3(threads_per_block),                                                 // Block dimensions
        kernel_args,                                                             // Array of kernel argument pointers
        shared_memory_per_block,                                                 // Shared memory per block (in bytes)
        stream);                                                                 // CUDA stream
    if (launch_error != cudaSuccess)
        if (launch_error == cudaErrorMemoryAllocation) { return {status_t::bad_alloc_k, launch_error}; }
        else { return {status_t::unknown_k, launch_error}; }

    cudaEventRecord(stop_event, stream);

    // Fetch the execution error:
    float execution_milliseconds = 0;
    cudaError_t execution_error = cudaStreamSynchronize(stream);
    cudaEventElapsedTime(&execution_milliseconds, start_event, stop_event);

    if (execution_error != cudaSuccess)
        if (execution_error == cudaErrorMemoryAllocation) {
            return {status_t::bad_alloc_k, execution_error, execution_milliseconds};
        }
        else { return {status_t::unknown_k, execution_error, execution_milliseconds}; }
    return {status_t::success_k, cudaSuccess, execution_milliseconds};
}

/** @brief Dispatches baseline Needleman Wunsch algorithm to the GPU. */
template <typename char_type_, typename allocator_type_, typename substituter_type_>
struct needleman_wunsch_scores<char_type_, substituter_type_, allocator_type_, sz_cap_cuda_k> {

    using char_t = char_type_;
    using substituter_t = substituter_type_;

    substituter_t substituter_ {};
    error_cost_t gap_cost_ {1};

    needleman_wunsch_scores() noexcept {}
    needleman_wunsch_scores(substituter_t subs, error_cost_t gap_cost) noexcept
        : substituter_(subs), gap_cost_(gap_cost) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    cuda_status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                             results_type_ &&results, gpu_specs_t specs = {}, cudaStream_t stream = 0) const noexcept {
        return _needleman_wunsch_via_cuda_warp<sz_cap_cuda_k>(first_strings, second_strings, results, substituter_,
                                                              gap_cost_, specs, stream);
    }
};

#pragma endregion

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_SIMILARITIES_CUH_