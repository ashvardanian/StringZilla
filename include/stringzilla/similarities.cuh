/**
 *  @brief  CUDA-accelerated string similarity utilities.
 *  @file   similarities.cuh
 *  @author Ash Vardanian
 *
 *  Includes core APIs:
 *
 *  - `sz::cuda::levenshtein_distances` & `sz::cuda::levenshtein_distances_utf8` for Levenshtein edit-distances.
 *  - `sz::cuda::needleman_wunsch_score` for weighted Needleman-Wunsch global alignment.
 *
 *  Unlike the trivially parallelizable CPU kernels in `stringzilla/similarity.h`, the GPU kernels in this file are
 *  designed for batch-processing of large collections of strings, assigning a single warp to each string pair.
 *  Thus, they should be used when hundreds of pairwise comparisons are needed, and the strings are long enough to
 *  amortize the cost of copying them to the GPU.
 */
#ifndef STRINGZILLA_SIMILARITIES_CUH_
#define STRINGZILLA_SIMILARITIES_CUH_

#include "types.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

namespace ashvardanian {
namespace stringzilla {
namespace cuda {

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
template <                                                        //
    typename char_type_,                                          //
    typename distance_type_ = sz_size_t,                          //
    typename get_substitution_cost_ = uniform_substitution_cost_t //

    >
__device__ void score_diagonally(                                                        //
    span<char_type_ const> const &first, span<char_type_ const> const &second,           //
    distance_type_ &result_ref,                                                          //
    sz_error_cost_t gap_cost = 1,                                                        //
    get_substitution_cost_ const &get_substitution_cost = uniform_substitution_cost_t {} //
) {
    using char_type = char_type_;
    using distance_type = distance_type_;

    // Make sure the size relation between the strings is correct.
    char_type const *shorter_global = first.data(), *longer_global = second.data();
    sz_size_t shorter_length = first.size(), longer_length = second.size();
    if (shorter_length > longer_length) {
        std::swap(shorter_global, longer_global);
        std::swap(shorter_length, longer_length);
    }

    // We are going to store 3 diagonals of the matrix, assuming each would fit into a single ZMM register.
    // The length of the longest (main) diagonal would be `shorter_dim = (shorter_length + 1)`.
    sz_size_t const shorter_dim = shorter_length + 1;
    sz_size_t const longer_dim = longer_length + 1;

    // Let's say we are dealing with 3 and 5 letter words.
    // The matrix will have size 4 x 6, parameterized as (shorter_dim x longer_dim).
    // It will have:
    // - 4 diagonals of increasing length, at positions: 0, 1, 2, 3.
    // - 2 diagonals of fixed length, at positions: 4, 5.
    // - 3 diagonals of decreasing length, at positions: 6, 7, 8.
    sz_size_t const diagonals_count = shorter_dim + longer_dim - 1;
    sz_size_t const max_diagonal_length = shorter_length + 1;

    // Allocating shared memory is handled on the host side.
    extern __shared__ char buffer[];

    // The next few pointers will be swapped around.
    distance_type *previous_distances = reinterpret_cast<distance_type *>(buffer);
    distance_type *current_distances = previous_distances + max_diagonal_length;
    distance_type *next_distances = current_distances + max_diagonal_length;
    char_type *const longer = (char_type *)(next_distances + max_diagonal_length);
    char_type *const shorter_reversed = longer + longer_length;

    // Each thread in the warp will be loading it's own set of strided characters into shared memory.
    for (sz_size_t i = threadIdx.x; i < longer_length; i += blockDim.x) longer[i] = longer_global[i];
    for (sz_size_t i = threadIdx.x; i < shorter_length; i += blockDim.x)
        shorter_reversed[i] = shorter_global[shorter_length - i - 1];

    // Initialize the first two diagonals:
    if (threadIdx.x == 0) {
        previous_distances[0] = 0;
        current_distances[0] = current_distances[1] = 1;
    }

    // Make sure the shared memory is fully loaded.
    __syncthreads();

    // We skip diagonals 0 and 1, as they are trivial.
    // We will start with diagonal 2, which has length 3, with the first and last elements being preset,
    // so we are effectively computing just one value, as will be marked by a single set bit in
    // the `next_diagonal_mask` on the very first iteration.
    sz_size_t next_diagonal_index = 2;

    // Progress through the upper-left triangle of the Levenshtein matrix.
    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = next_diagonal_index + 1;
        for (sz_size_t offset_in_diagonal = threadIdx.x + 1; offset_in_diagonal + 1 < next_diagonal_length;
             offset_in_diagonal += blockDim.x) {
            // ? Note that here we are still traversing both buffers in the same order,
            // ? because the shorter string has been reversed into `shorter_reversed`.
            char_type shorter_char = shorter_reversed[shorter_length - next_diagonal_index + offset_in_diagonal];
            char_type longer_char = longer[offset_in_diagonal - 1];
            sz_error_cost_t cost_of_substitution = get_substitution_cost(shorter_char, longer_char);
            distance_type cost_if_substitution = previous_distances[offset_in_diagonal - 1] + cost_of_substitution;
            distance_type cost_if_deletion_or_insertion =      //
                sz_min_of_two(                                 //
                    current_distances[offset_in_diagonal - 1], //
                    current_distances[offset_in_diagonal]      //
                    ) +
                gap_cost;
            next_distances[offset_in_diagonal] = sz_min_of_two(cost_if_deletion_or_insertion, cost_if_substitution);
        }
        // Don't forget to populate the first row and the first column of the Levenshtein matrix.
        if (threadIdx.x == 0) next_distances[0] = next_distances[next_diagonal_length - 1] = next_diagonal_index;
        __syncthreads();

        // Perform a circular rotation of those buffers, to reuse the memory.
        distance_type *temporary = previous_distances;
        previous_distances = current_distances;
        current_distances = next_distances;
        next_distances = temporary;
    }

    // Now let's handle the anti-diagonal band of the matrix, between the top and bottom-right triangles.
    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = shorter_dim;
        for (sz_size_t offset_in_diagonal = threadIdx.x; offset_in_diagonal + 1 < next_diagonal_length;
             offset_in_diagonal += blockDim.x) {
            char_type shorter_char = shorter_reversed[shorter_length - shorter_dim + offset_in_diagonal + 1];
            char_type longer_char = longer[next_diagonal_index - shorter_dim + offset_in_diagonal];
            sz_error_cost_t cost_of_substitution = get_substitution_cost(shorter_char, longer_char);
            distance_type cost_if_substitution = previous_distances[offset_in_diagonal] + cost_of_substitution;
            distance_type cost_if_deletion_or_insertion =     //
                sz_min_of_two(                                //
                    current_distances[offset_in_diagonal],    //
                    current_distances[offset_in_diagonal + 1] //
                    ) +
                gap_cost;
            next_distances[offset_in_diagonal] = sz_min_of_two(cost_if_deletion_or_insertion, cost_if_substitution);
        }
        if (threadIdx.x == 0) next_distances[next_diagonal_length - 1] = next_diagonal_index;

        __syncthreads();
        // ! In the central anti-diagonal band, we can't just assign the `current_distances + 1` to `previous_distances`
        // ! for the circular shift, as we will end up spilling outside of the diagonal's buffer a few iterations later.
        // ! Assuming in-place `memmove` is tricky on the GPU, so we will copy the data.
        for (sz_size_t i = threadIdx.x; i + 1 < next_diagonal_length; i += blockDim.x)
            previous_distances[i] = current_distances[i + 1];
        __syncthreads();
        for (sz_size_t i = threadIdx.x; i < next_diagonal_length; i += blockDim.x)
            current_distances[i] = next_distances[i];
        __syncthreads();
    }

    // Now let's handle the bottom-right triangle of the matrix.
    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
#pragma omp simd
        for (sz_size_t offset_in_diagonal = threadIdx.x; offset_in_diagonal < next_diagonal_length;
             offset_in_diagonal += blockDim.x) {
            char_type shorter_char = shorter_reversed[shorter_length - shorter_dim + offset_in_diagonal + 1];
            char_type longer_char = longer[next_diagonal_index - shorter_dim + offset_in_diagonal];
            sz_error_cost_t cost_of_substitution = get_substitution_cost(shorter_char, longer_char);
            distance_type cost_if_substitution = previous_distances[offset_in_diagonal] + cost_of_substitution;
            distance_type cost_if_deletion_or_insertion =     //
                sz_min_of_two(                                //
                    current_distances[offset_in_diagonal],    //
                    current_distances[offset_in_diagonal + 1] //
                    ) +
                gap_cost;
            next_distances[offset_in_diagonal] = sz_min_of_two(cost_if_deletion_or_insertion, cost_if_substitution);
        }
        // Perform a circular rotation of those buffers, to reuse the memory, this time, with a shift,
        // dropping the first element in the current array.
        distance_type *temporary = previous_distances;
        // ! Drop the first entry among the current distances.
        // ! Assuming every next diagonal is shorter by one element, we don't need a full-blown `sz_move`.
        // ! to shift the array by one element.
        previous_distances = current_distances + 1;
        current_distances = next_distances;
        next_distances = temporary;
        __syncthreads();
    }

    // Export one result per each block.
    if (threadIdx.x == 0) result_ref = current_distances[0];
}

/**
 *  @brief  Alignment Scores and Edit Distances algorithm evaluating the Dynamic Programming matrix
 *          @b three skewed (reverse) diagonals at a time on a GPU, leveraging CUDA for parallelization.
 *          Each pair of strings gets its own @b "block" of CUDA threads and shared memory.
 *
 *  @note   Unlike the `openmp::score_diagonally` kernel, 32-bit integers are used for offsets and lengths,
 *          as GPUs are often better optimized for 32-bit arithmetic.
 *
 *  @param[in] first_strings Array of first strings in each pair for score calculation.
 *  @param[in] second_strings Array of second strings in each pair for score calculation.
 *  @param[out] results_ptr Output array of scores for each pair of strings.
 */
template <                                                        //
    typename first_strings_type_,                                 //
    typename second_strings_type_,                                //
    typename global_distance_type_ = sz_size_t,                   //
    typename get_substitution_cost_ = uniform_substitution_cost_t //
    >
__global__ void scores_diagonally(                                                //
    first_strings_type_ const &first_strings,                                     //
    second_strings_type_ const &second_strings,                                   //
    global_distance_type_ *results_ptr,                                           //
    sz_error_cost_t gap_cost = 1,                                                 //
    get_substitution_cost_ get_substitution_cost = uniform_substitution_cost_t {} //
) {
    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    using first_string_type = typename first_strings_type_::value_type;
    using second_string_type = typename second_strings_type_::value_type;
    using first_char_type = typename first_string_type::value_type;
    using second_char_type = typename second_string_type::value_type;
    static_assert(sizeof(first_char_type) == sizeof(second_char_type), "Character types don't match");
    using char_type = first_char_type;
    using distance_type = global_distance_type_;

    // We are computing N edit distances for N pairs of strings. Not a cartesian product!
    // Each block/warp may end up receiving a different number of strings.
    for (sz_size_t pair_idx = threadIdx.x; pair_idx < first_strings.size(); pair_idx += blockDim.x) {
        first_string_type const &first_global = first_strings[pair_idx];
        second_string_type const &second_global = second_strings[pair_idx];
        distance_type &results_ref = results_ptr[pair_idx];

        // Skip empty strings.
        sz_size_t const first_length = first_global.length();
        sz_size_t const second_length = second_global.length();
        if (first_length == 0) {
            results_ref = second_length * gap_cost;
            continue;
        }
        if (second_length == 0) {
            results_ref = first_length * gap_cost;
            continue;
        }

        // Estimate the maximum dimension of the DP matrix to pick the smallest fitting type.
        sz_size_t const max_dim = sz_max_of_two(first_length, second_length) + 1;
        span<char const> const first = {first_global.data(), first_length};
        span<char const> const second = {second_global.data(), second_length};
        if (max_dim < 256u) {
            sz_u8_t result_u8;
            score_diagonally(first, second, result_u8, gap_cost, get_substitution_cost);
            results_ref = result_u8;
        }
        else if (max_dim < 65536u) {
            sz_u16_t result_u16;
            score_diagonally(first, second, result_u16, gap_cost, get_substitution_cost);
            results_ref = result_u16;
        }
        else {
            sz_size_t result_size;
            score_diagonally(first, second, result_size, gap_cost, get_substitution_cost);
            results_ref = result_size;
        }
    }
}

template <typename first_strings_type_,
          typename second_strings_type_>
size_t scores_diagonally_shared_memory_requirement( //
    first_strings_type_ const &first_strings, second_strings_type_ const &second_strings) noexcept {
    sz_size_t max_required_shared_memory = 0;
    for (sz_size_t i = 0; i < first_strings.size(); ++i) {
        sz_size_t const first_length = first_strings[i].length();
        sz_size_t const second_length = second_strings[i].length();
        sz_size_t const shorter_length = sz_min_of_two(first_length, second_length);
        sz_size_t const longer_length = sz_max_of_two(first_length, second_length);
        sz_size_t const max_diagonal_length = shorter_length + 1;
        sz_size_t const max_cell_value = longer_length + 1;
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
 *  @brief Computes the @b byte-level Levenshtein distances between pairs of strings using the CUDA backend.
 *  @param[in] first_strings Array of first strings in each pair for score calculation.
 *  @param[in] second_strings Array of second strings in each pair for score calculation.
 *  @param[in] alloc An allocator for the internal buffers.
 *  @sa `levenshtein_distance_utf8` for UTF-8 strings.
 *  @sa `scores_diagonally` for the core algorithm.
 */
template <                             //
    typename first_strings_type_,      //
    typename second_strings_type_,     //
    typename results_type_ = sz_size_t //
    >
status_t levenshtein_distances(                                                           //
    first_strings_type_ const &first_strings, second_strings_type_ const &second_strings, //
    results_type_ *results,                                                               //
    specs_t specs = {}, cudaStream_t stream = 0) noexcept(false) {

    // We need to be able to copy these function arguments into GPU memory:
    static_assert(
        std::is_trivially_copyable<first_strings_type_>() && std::is_trivially_copyable<second_strings_type_>(),
        "The first and second strings must be trivially copyable types - consider `arrow_strings_view`.");

    // Make sure that we don't string pairs that are too large to fit 3 matrix diagonals into shared memory.
    // H100 Streaming Multiprocessor can have up to 128 active warps concurrently and only 256 KB of shared memory.
    // A100 SMs had only 192 KB. We can't deal with blocks that require more memory than the SM can provide.
    sz_size_t shared_memory_per_block = scores_diagonally_shared_memory_requirement(first_strings, second_strings);
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
    auto kernel =
        &scores_diagonally<first_strings_type_, second_strings_type_, results_type_, uniform_substitution_cost_t>;
    kernel<<<count_blocks, threads_per_block, shared_memory_per_block, stream>>>(first_strings, second_strings, results,
                                                                                 1, {});

    // Fetch the error:
    cudaError_t error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
        if (error == cudaErrorMemoryAllocation) { return status_t::bad_alloc_k; }
        else { return status_t::unknown_error_k; }
    }
    return status_t::success_k;
}

/**
 *  @brief Computes the @b byte-level Needleman-Wunsch scores between pairs of strings using the CUDA backend.
 *  @param[in] first_strings Array of first strings in each pair for score calculation.
 *  @param[in] second_strings Array of second strings in each pair for score calculation.
 *  @param[in] alloc An allocator for the internal buffers.
 *  @sa `scores_diagonally` for the core algorithm.
 */
template <                              //
    typename first_strings_type_,       //
    typename second_strings_type_,      //
    typename results_type_ = sz_ssize_t //
    >
status_t needleman_wunsch_scores(                                                         //
    first_strings_type_ const &first_strings, second_strings_type_ const &second_strings, //
    results_type_ *results,                                                               //
    sz_error_cost_t const *subs, sz_error_cost_t gap,                                     //
    specs_t specs = {}, cudaStream_t stream = 0) noexcept(false) {

    // We need to be able to copy these function arguments into GPU memory:
    static_assert(
        std::is_trivially_copyable<first_strings_type_>() && std::is_trivially_copyable<second_strings_type_>(),
        "The first and second strings must be trivially copyable types - consider `arrow_strings_view`.");

    // Make sure that we don't string pairs that are too large to fit 3 matrix diagonals into shared memory.
    // H100 Streaming Multiprocessor can have up to 128 active warps concurrently and only 256 KB of shared memory.
    // A100 SMs had only 192 KB. We can't deal with blocks that require more memory than the SM can provide.
    sz_size_t shared_memory_per_block = scores_diagonally_shared_memory_requirement(first_strings, second_strings);
    // ! Add the space we need to preload the substitution costs.
    // shared_memory_per_block += 256 * 256 * sizeof(sz_error_cost_t);
    if (shared_memory_per_block > specs.shared_memory_per_sm) return status_t::bad_alloc_k;

    // It may be the case that we've only received empty strings.
    if (shared_memory_per_block == 0) {
        for (sz_size_t i = 0; i < first_strings.size(); ++i)
            if (first_strings[i].length() == 0) { results[i] = second_strings[i].length() * gap; }
            else if (second_strings[i].length() == 0) { results[i] = first_strings[i].length() * gap; }
        return status_t::success_k;
    }

    // In most cases we should be able to fir many blocks per SM.
    sz_size_t count_blocks = specs.shared_memory_per_sm / shared_memory_per_block;
    if (count_blocks > specs.blocks_per_sm) count_blocks = specs.blocks_per_sm;
    if (count_blocks > first_strings.size()) count_blocks = first_strings.size();

    // Let's use all 32 threads in a warp.
    constexpr sz_size_t threads_per_block = 32u;
    auto kernel =
        &scores_diagonally<first_strings_type_, second_strings_type_, results_type_, lookup_substitution_cost_t>;
    kernel<<<count_blocks, threads_per_block, shared_memory_per_block, stream>>>(
        first_strings, second_strings, results, gap, lookup_substitution_cost_t {subs});

    // Fetch the error:
    cudaError_t error = cudaStreamSynchronize(stream);
    if (error != cudaSuccess) {
        if (error == cudaErrorMemoryAllocation) { return status_t::bad_alloc_k; }
        else { return status_t::unknown_error_k; }
    }
    return status_t::success_k;
}

} // namespace cuda
} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_SIMILARITIES_CUH_