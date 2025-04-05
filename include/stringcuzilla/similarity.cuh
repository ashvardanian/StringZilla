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

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief GPU adaptation of the `global_scorer` on CUDA, avoiding warp-level shuffles and DPX.
 *  @note Uses 32-bit `uint` counter to iterate through the string slices, so it can't be over 4 billion characters.
 */
template <                               //
    typename first_iterator_type_,       //
    typename second_iterator_type_,      //
    typename score_type_,                //
    typename substituter_type_,          //
    sz_similarity_objective_t objective_ //
    >
struct global_scorer<      //
    first_iterator_type_,  //
    second_iterator_type_, //
    score_type_,           //
    substituter_type_,     //
    objective_,            //
    sz_cap_cuda_k> {

    using first_iterator_t = first_iterator_type_;
    using second_iterator_t = second_iterator_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_capability_t capability_k = sz_cap_cuda_k;

    using first_char_t = typename std::iterator_traits<first_iterator_t>::value_type;
    using second_char_t = typename std::iterator_traits<second_iterator_t>::value_type;
    static_assert(is_same_type<first_char_t, second_char_t>::value, "String characters must be of the same type.");
    using char_t = first_char_t;

  private:
    substituter_t substituter_;
    error_cost_t gap_cost_ {1};
    score_t last_cell_ {0};

    static constexpr score_t pick_best(score_t a, score_t b) noexcept {
        if constexpr (objective_ == minimize_distance_k) { return sz_min_of_two(a, b); }
        else { return sz_max_of_two(a, b); }
    }

  public:
    __device__ global_scorer(substituter_t substituter, error_cost_t gap_cost) noexcept
        : substituter_(substituter), gap_cost_(gap_cost) {}

    /**
     *  @brief Initializes a boundary value within a certain diagonal.
     *  @note Should only be called for the diagonals outside of the bottom-right triangle.
     *  @note Should only be called for the top row and left column of the matrix.
     */
    __device__ void init(score_t &cell, sz_size_t diagonal_index) const noexcept { cell = gap_cost_ * diagonal_index; }

    /**
     *  @brief Extract the final result of the scoring operation which will be always in the bottom-right corner.
     */
    __device__ score_t score() const noexcept { return last_cell_; }

    /**
     *  @brief Computes one diagonal of the DP matrix, using the results of the previous 2x diagonals.
     *  @param first_reversed_slice The first string, @b reversed.
     *  @param second_slice The second string.
     *  @param n The length of the diagonal to evaluate and the number of characters to compare from each string.
     */
    __device__ void operator()(                                                        //
        first_iterator_t first_reversed_slice, second_iterator_t second_slice, uint n, // ! Unlike CPU, uses `uint`
        score_t const *scores_pre_substitution, score_t const *scores_pre_insertion, score_t const *scores_pre_deletion,
        score_t *scores_new) noexcept {

        for (uint i = threadIdx.x; i < n; i += blockDim.x) {
            score_t pre_substitution = scores_pre_substitution[i];
            score_t pre_insertion = scores_pre_insertion[i];
            score_t pre_deletion = scores_pre_deletion[i];

            // ? Note that here we are still traversing both buffers in the same order,
            // ? because one of the strings has been reversed beforehand.
            error_cost_t cost_of_substitution = substituter_(first_reversed_slice[i], second_slice[i]);
            score_t if_substitution = pre_substitution + cost_of_substitution;
            score_t if_deletion_or_insertion = pick_best(pre_deletion, pre_insertion) + gap_cost_;
            score_t cell_score = pick_best(if_deletion_or_insertion, if_substitution);
            scores_new[i] = cell_score;
        }

        // The last element of the last chunk is the result of the global alignment.
        if (threadIdx.x == 0) last_cell_ = scores_new[n - 1];
    }
};

/**
 *  @brief GPU adaptation of the `local_scorer` on CUDA, avoiding warp-level shuffles and DPX.
 *  @note Uses 32-bit `uint` counter to iterate through the string slices, so it can't be over 4 billion characters.
 */
template <                               //
    typename first_iterator_type_,       //
    typename second_iterator_type_,      //
    typename score_type_,                //
    typename substituter_type_,          //
    sz_similarity_objective_t objective_ //
    >
struct local_scorer<       //
    first_iterator_type_,  //
    second_iterator_type_, //
    score_type_,           //
    substituter_type_,     //
    objective_,            //
    sz_cap_cuda_k> {

    using first_iterator_t = first_iterator_type_;
    using second_iterator_t = second_iterator_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;

    static constexpr sz_similarity_objective_t objective_k = objective_;
    static constexpr sz_capability_t capability_k = sz_cap_cuda_k;

    using first_char_t = typename std::iterator_traits<first_iterator_t>::value_type;
    using second_char_t = typename std::iterator_traits<second_iterator_t>::value_type;
    static_assert(is_same_type<first_char_t, second_char_t>::value, "String characters must be of the same type.");
    using char_t = first_char_t;

  private:
    substituter_t substituter_;
    error_cost_t gap_cost_ {1};
    score_t best_score_ {0};

    static constexpr score_t pick_best(score_t a, score_t b) noexcept {
        if constexpr (objective_k == minimize_distance_k) { return sz_min_of_two(a, b); }
        else { return sz_max_of_two(a, b); }
    }

  public:
    __device__ local_scorer(substituter_t substituter, error_cost_t gap_cost) noexcept
        : substituter_(substituter), gap_cost_(gap_cost) {}

    /**
     *  @brief Initializes a boundary value within a certain diagonal.
     *  @note Should only be called for the diagonals outside of the bottom-right triangle.
     *  @note Should only be called for the top row and left column of the matrix.
     */
    __device__ void init(score_t &cell, sz_size_t diagonal_index) const noexcept { cell = 0; }

    /**
     *  @brief Extract the final result of the scoring operation which will be always in the bottom-right corner.
     */
    __device__ score_t score() const noexcept { return best_score_; }

    /**
     *  @brief Computes one diagonal of the DP matrix, using the results of the previous 2x diagonals.
     *  @param first_reversed_slice The first string, @b reversed.
     *  @param second_slice The second string.
     *  @param n The length of the diagonal to evaluate and the number of characters to compare from each string.
     */
    __device__ void operator()(                                                        //
        first_iterator_t first_reversed_slice, second_iterator_t second_slice, uint n, // ! Unlike CPU, uses `uint`
        score_t const *scores_pre_substitution, score_t const *scores_pre_insertion, score_t const *scores_pre_deletion,
        score_t *scores_new) noexcept {

        for (uint i = threadIdx.x; i < n; i += blockDim.x) {
            score_t pre_substitution = scores_pre_substitution[i];
            score_t pre_insertion = scores_pre_insertion[i];
            score_t pre_deletion = scores_pre_deletion[i];

            // ? Note that here we are still traversing both buffers in the same order,
            // ? because one of the strings has been reversed beforehand.
            error_cost_t cost_of_substitution = substituter_(first_reversed_slice[i], second_slice[i]);
            score_t if_substitution = pre_substitution + cost_of_substitution;
            score_t if_deletion_or_insertion = pick_best(pre_deletion, pre_insertion) + gap_cost_;
            score_t if_substitution_or_reset = pick_best(if_substitution, 0);
            score_t cell_score = pick_best(if_deletion_or_insertion, if_substitution_or_reset);
            scores_new[i] = cell_score;

            // Update the global maximum score if this cell beats it.
            best_score_ = pick_best(best_score_, cell_score);
        }

        // ! Don't forget to pick the best among the best scores per thread.
        // ! Assuming, that reducing across the warp is not possible, let's output the best score per thread
        // ! into the expired set of cells in `scores_pre_substitution`, and sequentially reduce it afterwards.
        scores_pre_substitution[threadIdx.x] = best_score_;
        __syncthreads();
        if (threadIdx.x == 0)
            for (uint i = 1; i < blockDim.x; ++i) best_score_ = pick_best(best_score_, scores_pre_substitution[i]);
    }
};

/**
 *  @brief  Alignment Score and Edit Distance algorithm evaluating the Dynamic Programming matrix
 *          @b three skewed (reverse) diagonals at a time on a GPU, leveraging CUDA for parallelization.
 *          This function implements a logic for a single pair of strings.
 *
 *  @param[in] first The first string.
 *  @param[in] second The second string.
 *  @param[out] result_ref Location to dump the calculated score from the first thread in warp.
 *  @param[in] gap_cost The uniform cost of a gap (insertion or deletion).
 *  @param[in] get_substitution_cost A commutative function returning the cost of substituting one char with another.
 *
 *  We could have implemented the logic of this function as part of the `scores_diagonally` kernel,
 *  but we want to control the used @b `distance_type_` at the level of each warp and score computation.
 *  If all of the strings except for one are 100-ish bytes, but one is 1000-ish bytes, we want to use
 *  the 8-bit `distance_type_` for the smaller strings, and 16-bit `distance_type_` for the larger one.
 *  The smaller the type, the more likely we are to use specialized @b SIMD instructions, like
 */
template <                                                       //
    sz_capability_t capability_ = sz_cap_cuda_k,                 //
    sz_similarity_objective_t objective_ = maximize_score_k,     //
    sz_similarity_locality_t locality_ = sz_similarity_global_k, //
    typename char_type_ = char,                                  //
    typename score_type_ = sz_size_t,                            //
    typename substituter_type_ = error_costs_uniform_t           //
    >
struct diagonal_walker_per_warp {

    using char_t = char_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_similarity_locality_t locality_k = locality_;
    static constexpr sz_similarity_objective_t objective_k = objective_;

    using global_scorer_t =
        global_scorer<char_t const *, char_t const *, score_t, substituter_t, objective_k, capability_k>;
    using local_scorer_t =
        local_scorer<char_t const *, char_t const *, score_t, substituter_t, objective_k, capability_k>;
    using scorer_t = std::conditional_t<locality_k == sz_similarity_local_k, local_scorer_t, global_scorer_t>;

  private:
    substituter_t substituter_;
    error_cost_t gap_cost_ {1};

  public:
    /**
     *  @param[in] substituter A commutative function returning the cost of substituting one char with another.
     *  @param[in] gap_cost The uniform cost of a gap (insertion or deletion).
     *  @param[in] alloc A default-constructible allocator for the internal buffers.
     *
     */
    __device__ diagonal_walker_per_warp(substituter_t substituter, error_cost_t gap_cost) noexcept
        : substituter_(substituter), gap_cost_(gap_cost) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score.
     */
    __device__ void operator()(span<char_t const> first, span<char_t const> second, score_t &result_ref,
                               char *shared_memory_buffer) const noexcept {

        // Make sure the size relation between the strings is correct.
        char_t const *shorter_global = first.data(), *longer_global = second.data();
        sz_size_t shorter_length = first.size(), longer_length = second.size();
        if (shorter_length > longer_length) {
            std::swap(shorter_global, longer_global);
            std::swap(shorter_length, longer_length);
        }
        constexpr sz_size_t max_allowed_length_k = std::numeric_limits<uint>::max();
        _sz_assert(shorter_length <= longer_length);
        _sz_assert(shorter_length > 0 && longer_length > 0);
        _sz_assert(longer_length < max_allowed_length_k);

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

        // The next few pointers will be swapped around.
        score_t *previous_scores = reinterpret_cast<score_t *>(shared_memory_buffer);
        score_t *current_scores = previous_scores + max_diagonal_length;
        score_t *next_scores = current_scores + max_diagonal_length;
        char_t *const longer = (char_t *)(next_scores + max_diagonal_length);
        char_t *const shorter_reversed = longer + longer_length;

        // Each thread in the warp will be loading it's own set of strided characters into shared memory.
        for (uint i = threadIdx.x; i < longer_length; i += blockDim.x) longer[i] = longer_global[i];
        for (uint i = threadIdx.x; i < shorter_length; i += blockDim.x)
            shorter_reversed[i] = shorter_global[shorter_length - i - 1];

        // Initialize the first two diagonals:
        scorer_t diagonal_aligner {substituter_, gap_cost_};
        if (threadIdx.x == 0) {
            diagonal_aligner.init(previous_scores[0], 0);
            diagonal_aligner.init(current_scores[0], 1);
            diagonal_aligner.init(current_scores[1], 1);
        }

        // Make sure the shared memory is fully loaded.
        __syncthreads();

        // We skip diagonals 0 and 1, as they are trivial.
        // We will start with diagonal 2, which has length 3, with the first and last elements being preset,
        // so we are effectively computing just one value, as will be marked by a single set bit in
        // the `next_diagonal_mask` on the very first iteration.
        uint next_diagonal_index = 2;

        // Progress through the upper-left triangle of the Levenshtein matrix.
        for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {

            uint const next_diagonal_length = next_diagonal_index + 1;
            diagonal_aligner(                                                //
                shorter_reversed + shorter_length - next_diagonal_index + 1, // first sequence of characters
                longer,                                                      // second sequence of characters
                next_diagonal_length - 2,           // number of elements to compute with the `diagonal_aligner`
                previous_scores,                    // costs pre substitution
                current_scores, current_scores + 1, // costs pre insertion/deletion
                next_scores + 1);

            // Don't forget to populate the first row and the first column of the Levenshtein matrix.
            if (threadIdx.x == 0) {
                diagonal_aligner.init(next_scores[0], next_diagonal_index);
                diagonal_aligner.init(next_scores[next_diagonal_length - 1], next_diagonal_index);
            }
            __syncthreads();

            // Perform a circular rotation of those buffers, to reuse the memory.
            score_t *temporary = previous_scores;
            previous_scores = current_scores;
            current_scores = next_scores;
            next_scores = temporary;
        }

        // Now let's handle the anti-diagonal band of the matrix, between the top and bottom-right triangles.
        for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {

            uint const next_diagonal_length = shorter_dim;
            diagonal_aligner(                                        //
                shorter_reversed + shorter_length - shorter_dim + 1, // first sequence of characters
                longer + next_diagonal_index - shorter_dim,          // second sequence of characters
                next_diagonal_length - 1,           // number of elements to compute with the `diagonal_aligner`
                previous_scores,                    // costs pre substitution
                current_scores, current_scores + 1, // costs pre insertion/deletion
                next_scores);

            // Don't forget to populate the first row of the Levenshtein matrix.
            if (threadIdx.x == 0) diagonal_aligner.init(next_scores[next_diagonal_length - 1], next_diagonal_index);

            __syncthreads();
            // ! In the central anti-diagonal band, we can't just set the `current_scores + 1` to `previous_scores`
            // ! for the circular shift, as we will end up spilling outside of the diagonal a few iterations later.
            // ! Assuming in-place `memmove` is tricky on the GPU, so we will copy the data.
            for (sz_size_t i = threadIdx.x; i + 1 < next_diagonal_length; i += blockDim.x)
                previous_scores[i] = current_scores[i + 1];
            __syncthreads();
            for (sz_size_t i = threadIdx.x; i < next_diagonal_length; i += blockDim.x)
                current_scores[i] = next_scores[i];
            __syncthreads();
        }

        // Now let's handle the bottom-right triangle of the matrix.
        for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {

            uint const next_diagonal_length = diagonals_count - next_diagonal_index;
            diagonal_aligner(                                        //
                shorter_reversed + shorter_length - shorter_dim + 1, // first sequence of characters
                longer + next_diagonal_index - shorter_dim,          // second sequence of characters
                next_diagonal_length,               // number of elements to compute with the `diagonal_aligner`
                previous_scores,                    // costs pre substitution
                current_scores, current_scores + 1, // costs pre insertion/deletion
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
            __syncthreads();
        }

        // Export one result per each block.
        if (threadIdx.x == 0) result_ref = diagonal_aligner.score();
    }
};

template <typename first_strings_type_,
          typename second_strings_type_>
sz_size_t _scores_diagonally_warp_shared_memory_requirement( //
    first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
    error_cost_t max_cell_difference) noexcept {

    sz_size_t max_required_shared_memory = 0;
    for (sz_size_t i = 0; i < first_strings.size(); ++i) {
        sz_size_t const first_length = first_strings[i].length();
        sz_size_t const second_length = second_strings[i].length();
        sz_size_t const shorter_length = sz_min_of_two(first_length, second_length);
        sz_size_t const longer_length = sz_max_of_two(first_length, second_length);
        sz_size_t const max_diagonal_length = shorter_length + 1;
        sz_size_t const max_cell_value = (longer_length + 1) * max_cell_difference;
        sz_size_t const bytes_per_cell = max_cell_value < 256 ? 1 : max_cell_value < 65536 ? 2 : 4;
        // For each string we need to copy its contents, and allocate 3 bands proportional to the length
        // of the shorter string with each cell being big enough to hold the length of the longer one.
        sz_size_t const shared_memory_requirement = 3 * max_diagonal_length * bytes_per_cell + //
                                                    first_length + second_length;
        max_required_shared_memory = sz_max_of_two(max_required_shared_memory, shared_memory_requirement);
    }
    return max_required_shared_memory;
}

/**
 *  @brief  Levenshtein edit distances algorithm evaluating the Dynamic Programming matrix
 *          @b three skewed (reverse) diagonals at a time on a GPU, leveraging CUDA for parallelization.
 *          Each pair of strings gets its own @b "block" of CUDA threads and shared memory.
 *
 *  @param[in] first_strings Array of first strings in each pair for score calculation.
 *  @param[in] second_strings Array of second strings in each pair for score calculation.
 *  @param[out] results_ptr Output array of scores for each pair of strings.
 */
template <                                         //
    sz_capability_t capability_ = sz_cap_serial_k, //
    typename first_strings_type_,                  //
    typename second_strings_type_,                 //
    typename score_type_ = sz_size_t               //
    >
__global__ void _levenshtein_in_cuda(    //
    first_strings_type_ first_strings,   //
    second_strings_type_ second_strings, //
    score_type_ *results_ptr             //
) {
    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    using first_string_t = typename first_strings_type_::value_type;
    using second_string_t = typename second_strings_type_::value_type;
    using first_char_t = typename first_string_t::value_type;
    using second_char_t = typename second_string_t::value_type;
    static_assert(sizeof(first_char_t) == sizeof(second_char_t), "Character types don't match");
    using char_t = first_char_t;
    using score_t = score_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_similarity_objective_t objective_k = minimize_distance_k;
    using walker_u8_t = diagonal_walker_per_warp<capability_k, objective_k, sz_similarity_global_k, char_t, sz_u8_t>;
    using walker_u16_t = diagonal_walker_per_warp<capability_k, objective_k, sz_similarity_global_k, char_t, sz_u16_t>;
    using walker_u32_t = diagonal_walker_per_warp<capability_k, objective_k, sz_similarity_global_k, char_t, sz_u32_t>;

    // Allocating shared memory is handled on the host side.
    extern __shared__ char shared_memory_buffer[];
    printf("results_ptr: %p\n", results_ptr);

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
            result_ref = second_length;
            continue;
        }
        if (second_length == 0) {
            result_ref = first_length;
            continue;
        }

        // Estimate the maximum dimension of the DP matrix to pick the smallest fitting type.
        sz_size_t const max_cell_value = sz_max_of_two(first_length, second_length) + 1;
        span<char const> const first = {first_global.data(), first_length};
        span<char const> const second = {second_global.data(), second_length};
        if (max_cell_value < 256u) {
            sz_u8_t result_u8 = (sz_u8_t)-1;
            walker_u8_t walker({}, 1);
            walker(first, second, result_u8, shared_memory_buffer);
            if (threadIdx.x == 0) result_ref = result_u8;
        }
        else if (max_cell_value < 65536u) {
            sz_u16_t result_u16 = (sz_u16_t)-1;
            walker_u16_t walker({}, 1);
            walker(first, second, result_u16, shared_memory_buffer);
            if (threadIdx.x == 0) result_ref = result_u16;
        }
        else {
            sz_u32_t result_u32 = (sz_u32_t)-1;
            walker_u32_t walker({}, 1);
            walker(first, second, result_u32, shared_memory_buffer);
            if (threadIdx.x == 0) result_ref = result_u32;
        }
    }
}

/** @brief Dispatches on @b `_levenshtein_in_cuda` on the device side from the host side. */
template <                                         //
    sz_capability_t capability_ = sz_cap_serial_k, //
    typename first_strings_type_,                  //
    typename second_strings_type_,                 //
    typename score_type_ = sz_size_t               //
    >
status_t _levenshtein_via_cuda(                                                           //
    first_strings_type_ const &first_strings, second_strings_type_ const &second_strings, //
    score_type_ *results, gpu_specs_t specs = {}, cudaStream_t stream = 0) noexcept(false) {

    // We need to be able to copy these function arguments into GPU memory:
    static constexpr sz_capability_t capability_k = capability_;
    using first_strings_t = first_strings_type_;
    using second_strings_t = second_strings_type_;
    using score_t = score_type_;
    static_assert(std::is_trivially_copyable<first_strings_t>() && std::is_trivially_copyable<second_strings_t>(),
                  "The first and second strings must be trivially copyable types - consider `arrow_strings_view`.");

    // Make sure that we don't string pairs that are too large to fit 3 matrix diagonals into shared memory.
    // H100 Streaming Multiprocessor can have up to 128 active warps concurrently and only 256 KB of shared memory.
    // A100 SMs had only 192 KB. We can't deal with blocks that require more memory than the SM can provide.
    sz_size_t shared_memory_per_block =
        _scores_diagonally_warp_shared_memory_requirement(first_strings, second_strings, 1);
    if (shared_memory_per_block > specs.shared_memory_per_sm) return status_t::bad_alloc_k;

    // It may be the case that we've only received empty strings.
    if (shared_memory_per_block == 0) {
        for (sz_size_t i = 0; i < first_strings.size(); ++i)
            if (first_strings[i].length() == 0) { results[i] = second_strings[i].length(); }
            else if (second_strings[i].length() == 0) { results[i] = first_strings[i].length(); }
        return status_t::success_k;
    }

    // In most cases we should be able to fir many blocks per SM.
    sz_size_t count_blocks = specs.shared_memory_per_sm / shared_memory_per_block;
    if (count_blocks > specs.blocks_per_sm) count_blocks = specs.blocks_per_sm;
    if (count_blocks > first_strings.size()) count_blocks = first_strings.size();

    // Let's use all 32 threads in a warp.
    constexpr sz_size_t threads_per_block = 32u;
    auto kernel = &_levenshtein_in_cuda<capability_k, first_strings_t, second_strings_t, score_t>;
    void *kernel_args[] = {(void *)&first_strings, (void *)&second_strings, (void *)&results};

    // Enqueue the kernel for execution:
    cudaError_t launch_error = cudaLaunchKernel( //
        reinterpret_cast<void *>(kernel),        // Kernel function pointer
        dim3(count_blocks),                      // Grid dimensions
        dim3(threads_per_block),                 // Block dimensions
        kernel_args,                             // Array of kernel argument pointers
        shared_memory_per_block,                 // Shared memory per block (in bytes)
        stream);                                 // CUDA stream
    if (launch_error != cudaSuccess)
        if (launch_error == cudaErrorMemoryAllocation) { return status_t::bad_alloc_k; }
        else { return status_t::unknown_k; }

    // Fetch the execution error:
    cudaError_t execution_error = cudaStreamSynchronize(stream);
    if (execution_error != cudaSuccess)
        if (execution_error == cudaErrorMemoryAllocation) { return status_t::bad_alloc_k; }
        else { return status_t::unknown_k; }
    return status_t::success_k;
}

/** @brief Dispatches baseline Levenshtein edit distance algorithm to the GPU. */
template <typename char_type_>
struct levenshtein_distances<sz_cap_cuda_k, char_type_, dummy_alloc_t> {

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results, gpu_specs_t specs = {}, cudaStream_t stream = 0) const noexcept {
        return _levenshtein_via_cuda<sz_cap_cuda_k>(first_strings, second_strings, results, specs, stream);
    }
};

/** @brief Dispatches Levenshtein edit distance algorithm to the GPU with Kepler shuffles. */
template <typename char_type_>
struct levenshtein_distances<sz_cap_kepler_k, char_type_, dummy_alloc_t> {

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results, gpu_specs_t specs = {}, cudaStream_t stream = 0) const noexcept {
        return _levenshtein_via_cuda<sz_cap_kepler_k>(first_strings, second_strings, results, specs, stream);
    }
};

/** @brief Dispatches Levenshtein edit distance algorithm to the GPU with Kepler shuffles and Hopper DPX. */
template <>
struct levenshtein_distances<sz_cap_hopper_k, char, dummy_alloc_t> {

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results, gpu_specs_t specs = {}, cudaStream_t stream = 0) const noexcept {
        return _levenshtein_via_cuda<sz_cap_hopper_k>(first_strings, second_strings, results, specs, stream);
    }
};

/**
 *  @brief  Needleman-Wunsch alignment cores algorithm evaluating the Dynamic Programming matrix
 *          @b three skewed (reverse) diagonals at a time on a GPU, leveraging CUDA for parallelization.
 *          Each pair of strings gets its own @b "block" of CUDA threads and shared memory.
 *
 *  @param[in] first_strings Array of first strings in each pair for score calculation.
 *  @param[in] second_strings Array of second strings in each pair for score calculation.
 *  @param[out] results_ptr Output array of scores for each pair of strings.
 *
 *  @tparam substituter_type_ Must be a trivially copyable object already placed in the GPU global memory.
 */
template <                                             //
    sz_capability_t capability_ = sz_cap_serial_k,     //
    typename first_strings_type_,                      //
    typename second_strings_type_,                     //
    typename score_type_ = sz_size_t,                  //
    typename substituter_type_ = error_costs_uniform_t //
    >
__global__ void _needleman_wunsch_in_cuda(           //
    first_strings_type_ first_strings,               //
    second_strings_type_ second_strings,             //
    score_type_ *results_ptr,                        //
    substituter_type_ const *substituter_global_ptr, //
    error_cost_t gap_cost = 1                        //
) {
    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    using first_string_t = typename first_strings_type_::value_type;
    using second_string_t = typename second_strings_type_::value_type;
    using first_char_t = typename first_string_t::value_type;
    using second_char_t = typename second_string_t::value_type;
    static_assert(sizeof(first_char_t) == sizeof(second_char_t), "Character types don't match");
    using char_t = first_char_t;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    static_assert(std::is_trivially_copyable<substituter_t>::value, "Substituter must be a trivially copyable type.");
    static_assert(std::is_signed<score_t>::value, "Score must be a signed type.");

    static constexpr sz_capability_t cap_k = capability_;
    static constexpr sz_similarity_objective_t obj_k = maximize_score_k;
    static constexpr sz_similarity_locality_t locality_k = sz_similarity_global_k;
    using walker_i16_t = diagonal_walker_per_warp<cap_k, obj_k, locality_k, char_t, sz_i16_t, substituter_t &>;
    using walker_i32_t = diagonal_walker_per_warp<cap_k, obj_k, locality_k, char_t, sz_i32_t, substituter_t &>;
    using walker_i64_t = diagonal_walker_per_warp<cap_k, obj_k, locality_k, char_t, sz_i64_t, substituter_t &>;

    // Allocating shared memory is handled on the host side.
    extern __shared__ char shared_memory_buffer[];

    // Assuming, all pairwise computations use the same substituter, load it into shared memory.
    static constexpr uint substituter_size = sizeof(substituter_type_) / sizeof(error_cost_t);
    error_cost_t *substituter_costs = reinterpret_cast<error_cost_t *>(shared_memory_buffer);
    for (uint i = threadIdx.x; i < substituter_size; i += blockDim.x)
        substituter_costs[i] = reinterpret_cast<error_cost_t const *>(substituter_global_ptr)[i];
    substituter_t &substituter_shared = *reinterpret_cast<substituter_t *>(substituter_costs);
    __syncthreads();

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
        sz_size_t const max_cell_value = sz_max_of_two(first_length, second_length) + 1;
        span<char const> const first = {first_global.data(), first_length};
        span<char const> const second = {second_global.data(), second_length};
        if (max_cell_value < 256u) {
            sz_i16_t result_i16 = std::numeric_limits<sz_i16_t>::min();
            walker_i16_t walker(substituter_shared, gap_cost);
            walker(first, second, result_i16, shared_memory_buffer + substituter_size);
            if (threadIdx.x == 0) result_ref = result_i16;
        }
        else if (max_cell_value < 65536u) {
            sz_i32_t result_i32 = std::numeric_limits<sz_i32_t>::min();
            walker_i32_t walker(substituter_shared, gap_cost);
            walker(first, second, result_i32, shared_memory_buffer + substituter_size);
            if (threadIdx.x == 0) result_ref = result_i32;
        }
        else {
            sz_i64_t result_i64 = std::numeric_limits<sz_i64_t>::min();
            walker_i64_t walker(substituter_shared, gap_cost);
            walker(first, second, result_i64, shared_memory_buffer + substituter_size);
            if (threadIdx.x == 0) result_ref = result_i64;
        }
    }
}

/** @brief Dispatches on @b `_needleman_wunsch_in_cuda` on the device side from the host side. */
template <                                             //
    sz_capability_t capability_ = sz_cap_serial_k,     //
    typename first_strings_type_,                      //
    typename second_strings_type_,                     //
    typename score_type_ = sz_ssize_t,                 //
    typename substituter_type_ = error_costs_uniform_t //
    >
status_t _needleman_wunsch_via_cuda(                                                      //
    first_strings_type_ const &first_strings, second_strings_type_ const &second_strings, //
    score_type_ *results,
    substituter_type_ const *substituter_global_ptr, //
    error_cost_t gap_cost = 1,                       //
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
    sz_size_t shared_memory_per_block =
        _scores_diagonally_warp_shared_memory_requirement(first_strings, second_strings, 1);
    if (shared_memory_per_block > specs.shared_memory_per_sm) return status_t::bad_alloc_k;

    // It may be the case that we've only received empty strings.
    if (shared_memory_per_block == 0) {
        for (sz_size_t i = 0; i < first_strings.size(); ++i)
            if (first_strings[i].length() == 0) { results[i] = second_strings[i].length(); }
            else if (second_strings[i].length() == 0) { results[i] = first_strings[i].length(); }
        return status_t::success_k;
    }

    // ? Include the size of the substituter in the shared memory requirement.
    shared_memory_per_block += sizeof(substituter_t);

    // In most cases we should be able to fir many blocks per SM.
    sz_size_t count_blocks = specs.shared_memory_per_sm / shared_memory_per_block;
    if (count_blocks > specs.blocks_per_sm) count_blocks = specs.blocks_per_sm;
    if (count_blocks > first_strings.size()) count_blocks = first_strings.size();

    // Let's use all 32 threads in a warp.
    constexpr sz_size_t threads_per_block = 32u;
    auto kernel = &_needleman_wunsch_in_cuda<capability_k, first_strings_t, second_strings_t, score_t, substituter_t>;
    void *kernel_args[] = {
        (void *)&first_strings,         (void *)&second_strings, (void *)&results,
        (void *)substituter_global_ptr, (void *)&gap_cost,
    };

    // Enqueue the kernel for execution:
    cudaError_t launch_error = cudaLaunchKernel( //
        reinterpret_cast<void *>(kernel),        // Kernel function pointer
        dim3(count_blocks),                      // Grid dimensions
        dim3(threads_per_block),                 // Block dimensions
        kernel_args,                             // Array of kernel argument pointers
        shared_memory_per_block,                 // Shared memory per block (in bytes)
        stream);                                 // CUDA stream
    if (launch_error != cudaSuccess)
        if (launch_error == cudaErrorMemoryAllocation) { return status_t::bad_alloc_k; }
        else { return status_t::unknown_k; }

    // Fetch the execution error:
    cudaError_t execution_error = cudaStreamSynchronize(stream);
    if (execution_error != cudaSuccess)
        if (execution_error == cudaErrorMemoryAllocation) { return status_t::bad_alloc_k; }
        else { return status_t::unknown_k; }
    return status_t::success_k;
}

/** @brief Dispatches baseline Needleman Wunsch algorithm to the GPU. */
template <typename char_type_, typename substituter_type_>
struct needleman_wunsch_scores<sz_cap_cuda_k, char_type_, substituter_type_ *, dummy_alloc_t> {

    using char_t = char_type_;
    using substituter_t = substituter_type_ *; // ! Note the pointer, it must be in the global memory

    substituter_t substituter_ {};
    error_cost_t gap_cost_ {1};

    needleman_wunsch_scores() noexcept {}
    needleman_wunsch_scores(substituter_t subs, error_cost_t gap_cost) noexcept
        : substituter_(subs), gap_cost_(gap_cost) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results, gpu_specs_t specs = {}, cudaStream_t stream = 0) const noexcept {
        return _needleman_wunsch_via_cuda<sz_cap_cuda_k>(first_strings, second_strings, results, substituter_,
                                                         gap_cost_, specs, stream);
    }
};

/** @brief Dispatches Needleman Wunsch algorithm to the GPU with Kepler shuffles. */
template <typename char_type_, typename substituter_type_>
struct needleman_wunsch_scores<sz_cap_kepler_k, char_type_, substituter_type_ *, dummy_alloc_t> {

    using char_t = char_type_;
    using substituter_t = substituter_type_ *; // ! Note the pointer, it must be in the global memory

    substituter_t substituter_ {};
    error_cost_t gap_cost_ {1};

    needleman_wunsch_scores() noexcept {}
    needleman_wunsch_scores(substituter_t subs, error_cost_t gap_cost) noexcept
        : substituter_(subs), gap_cost_(gap_cost) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results, gpu_specs_t specs = {}, cudaStream_t stream = 0) const noexcept {
        return _needleman_wunsch_via_cuda<sz_cap_kepler_k>(first_strings, second_strings, results, substituter_,
                                                           gap_cost_, specs, stream);
    }
};

/** @brief Dispatches Needleman Wunsch algorithm to the GPU with Kepler shuffles and Hopper DPX. */
template <typename substituter_type_>
struct needleman_wunsch_scores<sz_cap_hopper_k, char, substituter_type_, dummy_alloc_t> {

    using char_t = char;
    using substituter_t = substituter_type_ *; // ! Note the pointer, it must be in the global memory

    substituter_t substituter_ {};
    error_cost_t gap_cost_ {1};

    needleman_wunsch_scores() noexcept {}
    needleman_wunsch_scores(substituter_t subs, error_cost_t gap_cost) noexcept
        : substituter_(subs), gap_cost_(gap_cost) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results, gpu_specs_t specs = {}, cudaStream_t stream = 0) const noexcept {
        return _needleman_wunsch_via_cuda<sz_cap_hopper_k>(first_strings, second_strings, results, substituter_,
                                                           gap_cost_, specs, stream);
    }
};

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_SIMILARITIES_CUH_