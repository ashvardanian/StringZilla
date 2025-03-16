/**
 *  @brief  OpenMP-accelerated string similarity utilities.
 *  @file   similarity.hpp
 *  @author Ash Vardanian
 *
 *  Includes core APIs:
 *
 *  - `sz::openmp::levenshtein_distances` & `sz::openmp::levenshtein_distances_utf8` for Levenshtein edit-distances.
 *  - `sz::openmp::needleman_wunsch_score` for weighted Needleman-Wunsch global alignment.
 *
 *  Those are mostly providing specialized overloads of the @b `sz::openmp::score_diagonally` template.
 */
#ifndef STRINGZILLA_SIMILARITY_HPP_
#define STRINGZILLA_SIMILARITY_HPP_

#include "types.h"

namespace ashvardanian {
namespace stringzilla {
namespace openmp {

struct uniform_substitution_cost_t {
    sz_error_cost_t operator()(char a, char b) const { return a == b ? 0 : 1; }
};

/**
 *  @brief  Alignment Score and Edit Distance algorithm evaluating the Dynamic Programming matrix
 *          @b three skewed (reverse) diagonals at a time on a CPU, leveraging OpenMP for parallelization.
 *  @sa     sz_levenshtein_distance, sz_levenshtein_distance_utf8, sz_needleman_wunsch_score
 *
 *  @param[in] first The first string.
 *  @param[in] first_length The length of the first string.
 *  @param[in] second The second string.
 *  @param[in] second_length The length of the second string.
 *
 *  There are smarter algorithms for computing the Levenshtein distance, mostly based on bit-level operations.
 *  Those, however, don't generalize well to arbitrary length inputs or non-uniform substitution costs.
 *  This algorithm provides a more flexible baseline implementation for future SIMD and GPGPU optimizations.
 */
template <                                                         //
    typename char_type_ = char,                                    //
    typename distance_type_ = sz_size_t,                           //
    typename get_substitution_cost_ = uniform_substitution_cost_t, //
    sz_error_cost_t gap_cost_ = 1                                  //
    >
sz_status_t score_diagonally(                            //
    char_type_ const *shorter, sz_size_t shorter_length, //
    char_type_ const *longer, sz_size_t longer_length,   //
    get_substitution_cost_ &&get_substitution_cost,      //
    sz_memory_allocator_t *alloc,                        //
    distance_type_ *result_ptr) {

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

    // We want to avoid reverse-order iteration over the shorter string.
    // Let's allocate a bit more memory and reverse-export our shorter string into that buffer.
    sz_size_t const buffer_length =
        sizeof(distance_type_) * max_diagonal_length * 3 + shorter_length * sizeof(char_type_);
    distance_type_ *const distances = (distance_type_ *)alloc->allocate(buffer_length, alloc->handle);
    if (!distances) return sz_bad_alloc_k;

    // The next few pointers will be swapped around.
    distance_type_ *previous_distances = distances;
    distance_type_ *current_distances = previous_distances + longer_dim;
    distance_type_ *next_distances = current_distances + longer_dim;
    char_type_ const *const shorter_reversed = (char_type_ const *)(next_distances + longer_dim);

    // Export the reversed string into the buffer.
    for (sz_size_t i = 0; i != shorter_length; ++i) shorter_reversed[i] = shorter[shorter_length - 1 - i];

    // Initialize the first two diagonals:
    previous_distances[0] = 0;
    current_distances[0] = current_distances[1] = 1;

    // We skip diagonals 0 and 1, as they are trivial.
    // We will start with diagonal 2, which has length 3, with the first and last elements being preset,
    // so we are effectively computing just one value, as will be marked by a single set bit in
    // the `next_diagonal_mask` on the very first iteration.
    sz_size_t next_diagonal_index = 2;

    // Progress through the upper triangle of the Levenshtein matrix.
    for (; next_diagonal_index != shorter_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = next_diagonal_index + 1;
#pragma omp simd
        for (sz_size_t offset_in_diagonal = 1; offset_in_diagonal + 1 < next_diagonal_length; ++offset_in_diagonal) {
            // ? Note that here we are still traversing both buffers in the same order,
            // ? because the shorter string has been reversed into `shorter_reversed`.
            char_type_ shorter_char = shorter_reversed[shorter_length - next_diagonal_index + offset_within_diagonal];
            char_type_ longer_char = longer[offset_in_diagonal - 1];
            sz_error_cost_t cost_of_substitution = get_substitution_cost(shorter_char, longer_char);
            distance_type_ cost_if_substitution = previous_distances[offset_in_diagonal] + cost_of_substitution;
            distance_type_ cost_if_deletion_or_insertion =     //
                sz_min_of_two(                                 //
                    current_distances[offset_in_diagonal - 1], //
                    current_distances[offset_in_diagonal]      //
                    ) +
                gap_cost_;
            next_distances[offset_in_diagonal] = sz_min_of_two(cost_if_deletion_or_insertion, cost_if_substitution);
        }
        // Don't forget to populate the first row and the first column of the Levenshtein matrix.
        next_distances[0] = next_distances[next_diagonal_length - 1] = next_diagonal_index;
        // Perform a circular rotation of those buffers, to reuse the memory.
        distance_type_ *temporary = previous_distances;
        previous_distances = current_distances;
        current_distances = next_distances;
        next_distances = temporary;
    }

    // Now let's handle the anti-diagonal band of the matrix, between the top and bottom triangles.
    for (; next_diagonal_index != longer_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = shorter_dim;
#pragma omp simd
        for (sz_size_t offset_in_diagonal = 0; offset_in_diagonal + 1 < next_diagonal_length; ++offset_in_diagonal) {
            char_type_ shorter_char = shorter_reversed[shorter_length - shorter_dim + offset_in_diagonal + 1];
            char_type_ longer_char = longer[next_diagonal_index - shorter_dim + offset_in_diagonal];
            sz_error_cost_t cost_of_substitution = get_substitution_cost(shorter_char, longer_char);
            distance_type_ cost_if_substitution = previous_distances[offset_in_diagonal] + cost_of_substitution;
            distance_type_ cost_if_deletion_or_insertion =    //
                sz_min_of_two(                                //
                    current_distances[offset_in_diagonal],    //
                    current_distances[offset_in_diagonal + 1] //
                    ) +
                gap_cost_;
            next_distances[i] = sz_min_of_two(cost_if_deletion_or_insertion, cost_if_substitution);
        }
        next_distances[next_diagonal_length - 1] = next_diagonal_index;
        // Perform a circular rotation of those buffers, to reuse the memory, this time, with a shift,
        // dropping the first element in the current array.
        distance_type_ *temporary = previous_distances;
        previous_distances = current_distances + 1; // ! Note how we shift forward here
        current_distances = next_distances;
        next_distances = temporary;
    }

    // Now let's handle the bottom-right triangle of the matrix.
    for (; next_diagonal_index != longer_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
#pragma omp simd
        for (sz_size_t offset_in_diagonal = 0; offset_in_diagonal < next_diagonal_length; ++offset_in_diagonal) {
            char_type_ shorter_char = shorter_reversed[shorter_length - shorter_dim + offset_in_diagonal + 1];
            char_type_ longer_char = longer[next_diagonal_index - shorter_dim + offset_in_diagonal];
            sz_error_cost_t cost_of_substitution = get_substitution_cost(shorter_char, longer_char);
            distance_type_ cost_if_substitution = previous_distances[offset_in_diagonal] + cost_of_substitution;
            distance_type_ cost_if_deletion_or_insertion =    //
                sz_min_of_two(                                //
                    current_distances[offset_in_diagonal],    //
                    current_distances[offset_in_diagonal + 1] //
                    ) +
                gap_cost_;
            next_distances[i] = sz_min_of_two(cost_if_deletion_or_insertion, cost_if_substitution);
        }
        // Perform a circular rotation of those buffers, to reuse the memory, this time, with a shift,
        // dropping the first element in the current array.
        distance_type_ *temporary = previous_distances;
        previous_distances = current_distances + 1; // ! Note how we shift forward here
        current_distances = next_distances;
        next_distances = temporary;
    }

    // Cache scalar before `free` call.
    distance_type_ result = current_distances[0];
    alloc->free(distances, buffer_length, alloc->handle);
    *result_ptr = result;
    return sz_success_k;
}

} // namespace openmp
} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_SIMILARITY_HPP_