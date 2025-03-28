/**
 *  @brief  OpenMP-accelerated string similarity utilities.
 *  @file   similarity.hpp
 *  @author Ash Vardanian
 *
 *  Includes core APIs:
 *
 *  - `sz::levenshtein_distances` & `sz::levenshtein_distances_utf8` for Levenshtein edit-distances.
 *  - `sz::needleman_wunsch_score` for weighted Needleman-Wunsch global alignment.
 *
 *  Those are mostly providing specialized overloads of the @b `sz::score_diagonally` template.
 */
#ifndef STRINGZILLA_SIMILARITY_HPP_
#define STRINGZILLA_SIMILARITY_HPP_

#include "types.h"

namespace ashvardanian {
namespace stringzilla {
namespace openmp {

struct dummy_allocator_t {
    using value_type = char;
    inline char *allocate(sz_size_t) const noexcept { return nullptr; }
    inline void deallocate(char *, sz_size_t) const noexcept {}
};

struct uniform_substitution_cost_t {
    inline sz_error_cost_t operator()(char a, char b) const noexcept { return a == b ? 0 : 1; }
};

struct lookup_substitution_cost_t {
    sz_error_cost_t const *costs;
    inline sz_error_cost_t operator()(char a, char b) const noexcept { return costs[(sz_u8_t)a * 256 + (sz_u8_t)b]; }
};

template <typename char_type_>
struct span {
    char_type_ const *data_;
    sz_size_t size_;

    char_type_ const *begin() const noexcept { return data_; }
    char_type_ const *end() const noexcept { return data_ + size_; }
    char_type_ const *data() const noexcept { return data_; }
    sz_size_t size() const noexcept { return size_; }
};

/**
 *  @brief  Alignment Score and Edit Distance algorithm evaluating the Dynamic Programming matrix
 *          @b three skewed (reverse) diagonals at a time on a CPU, leveraging OpenMP for parallelization.
 *
 *  @param[in] first The first string.
 *  @param[in] second The second string.
 *  @param[out] result_ref Location to dump the calculated score.
 *  @param[in] gap_cost The uniform cost of a gap (insertion or deletion).
 *  @param[in] get_substitution_cost A commutative function returning the cost of substituting one char with another.
 *  @param[in] alloc A default-constructible allocator for the internal buffers.
 *
 *  There are smarter algorithms for computing the Levenshtein distance, mostly based on bit-level operations.
 *  Those, however, don't generalize well to arbitrary length inputs or non-uniform substitution costs.
 *  This algorithm provides a more flexible baseline implementation for future SIMD and GPGPU optimizations.
 *
 *  @note   The API of this algorithm is a bit weird, but it's designed to minimize the reliance on the definitions
 *          in the `stringzilla.hpp` header, making compilation times shorter for the end-user.
 *  @sa     For lower-level API, check `sz_levenshtein_distance[_utf8]` and `sz_needleman_wunsch_score`.
 *  @sa     For simplicity, use the `sz::levenshtein_distance[_utf8]` and `sz::needleman_wunsch_score`.
 *  @sa     For bulk API, use `sz::levenshtein_distances[_utf8]`.
 */
template <                                                         //
    typename char_type_,                                           //
    typename distance_type_ = sz_size_t,                           //
    typename get_substitution_cost_ = uniform_substitution_cost_t, //
    typename allocator_type_ = dummy_allocator_t                   //
    >
sz_status_t score_diagonally(                                                        //
    span<char_type_ const> first, span<char_type_ const> second,                     //
    distance_type_ &result_ref,                                                      //
    sz_error_cost_t gap_cost = 1,                                                    //
    get_substitution_cost_ &&get_substitution_cost = uniform_substitution_cost_t {}, //
    allocator_type_ &&alloc = allocator_type_ {}                                     //
) {
    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    using allocated_type = typename allocator_type_::value_type;
    static_assert(sizeof(allocated_type) == sizeof(char), "Allocator must be byte-aligned");
    using char_type = char_type_;
    using distance_type = distance_type_;

    // Make sure the size relation between the strings is correct.
    char_type const *shorter = first.data(), *longer = second.data();
    sz_size_t shorter_length = first.size(), longer_length = second.size();
    if (shorter_length > longer_length) {
        std::swap(shorter, longer);
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

    // We want to avoid reverse-order iteration over the shorter string.
    // Let's allocate a bit more memory and reverse-export our shorter string into that buffer.
    sz_size_t const buffer_length =
        sizeof(distance_type) * max_diagonal_length * 3 + shorter_length * sizeof(char_type);
    distance_type *const buffer = (distance_type *)alloc.allocate(buffer_length);
    if (!buffer) return sz_bad_alloc_k;

    // The next few pointers will be swapped around.
    distance_type *previous_distances = buffer;
    distance_type *current_distances = previous_distances + max_diagonal_length;
    distance_type *next_distances = current_distances + max_diagonal_length;
    char_type *const shorter_reversed = (char_type *)(next_distances + max_diagonal_length);

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

    // Progress through the upper-left triangle of the Levenshtein matrix.
    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = next_diagonal_index + 1;
#pragma omp simd
        for (sz_size_t offset_in_diagonal = 1; offset_in_diagonal + 1 < next_diagonal_length; ++offset_in_diagonal) {
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
        next_distances[0] = next_distances[next_diagonal_length - 1] = next_diagonal_index;
        // Perform a circular rotation of those buffers, to reuse the memory.
        distance_type *temporary = previous_distances;
        previous_distances = current_distances;
        current_distances = next_distances;
        next_distances = temporary;
    }

    // Now let's handle the anti-diagonal band of the matrix, between the top and bottom-right triangles.
    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = shorter_dim;
#pragma omp simd
        for (sz_size_t offset_in_diagonal = 0; offset_in_diagonal + 1 < next_diagonal_length; ++offset_in_diagonal) {
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
        next_distances[next_diagonal_length - 1] = next_diagonal_index;
        // Perform a circular rotation of those buffers, to reuse the memory, this time, with a shift,
        // dropping the first element in the current array.
        distance_type *temporary = previous_distances;
        previous_distances = current_distances;
        current_distances = next_distances;
        next_distances = temporary;
        // ! Drop the first entry among the current distances.
        sz_move((sz_ptr_t)(previous_distances), (sz_ptr_t)(previous_distances + 1),
                (max_diagonal_length - 1) * sizeof(distance_type));
    }

    // Now let's handle the bottom-right triangle of the matrix.
    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
#pragma omp simd
        for (sz_size_t offset_in_diagonal = 0; offset_in_diagonal < next_diagonal_length; ++offset_in_diagonal) {
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
    }

    // Export the scalar before `free` call.
    result_ref = current_distances[0];
    alloc.deallocate((allocated_type *)buffer, buffer_length);
    return sz_success_k;
}

template <                                       //
    typename first_type_,                        //
    typename second_type_,                       //
    typename allocator_type_ = dummy_allocator_t //
    >
inline sz_size_t levenshtein_distance( //
    first_type_ const &first, second_type_ const &second,
    allocator_type_ &&alloc = allocator_type_ {}) noexcept(false) {

    sz_size_t const first_length = first.length();
    sz_size_t const second_length = second.length();
    if (first_length == 0) return second_length;
    if (second_length == 0) return first_length;

    // Estimate the maximum dimension of the DP matrix
    sz_size_t const max_dim = sz_max_of_two(first_length, second_length) + 1;
    if (max_dim < 256u) {
        sz_u8_t result_u8;
        sz_status_t status = score_diagonally<char, sz_u8_t, uniform_substitution_cost_t, allocator_type_>(
            {first.data(), first_length}, {second.data(), second_length}, result_u8, 1, uniform_substitution_cost_t {},
            std::forward<allocator_type_>(alloc));
        if (status == sz_bad_alloc_k) throw std::bad_alloc();
        return result_u8;
    }
    else if (max_dim < 65536u) {
        sz_u16_t result_u16;
        sz_status_t status = score_diagonally<char, sz_u16_t, uniform_substitution_cost_t, allocator_type_>(
            {first.data(), first_length}, {second.data(), second_length}, result_u16, 1, uniform_substitution_cost_t {},
            std::forward<allocator_type_>(alloc));
        if (status == sz_bad_alloc_k) throw std::bad_alloc();
        return result_u16;
    }
    else {
        sz_size_t result_size;
        sz_status_t status = score_diagonally<char, sz_size_t, uniform_substitution_cost_t, allocator_type_>(
            {first.data(), first_length}, {second.data(), second_length}, result_size, 1,
            uniform_substitution_cost_t {}, std::forward<allocator_type_>(alloc));
        if (status == sz_bad_alloc_k) throw std::bad_alloc();
        return result_size;
    }
}

template <                                       //
    typename first_type_,                        //
    typename second_type_,                       //
    typename allocator_type_ = dummy_allocator_t //
    >
inline sz_size_t levenshtein_distance_utf8( //
    first_type_ const &first, second_type_ const &second,
    allocator_type_ &&alloc = allocator_type_ {}) noexcept(false) {

    sz_size_t const first_length = first.length();
    sz_size_t const second_length = second.length();
    if (first_length == 0) return second_length;
    if (second_length == 0) return first_length;

    // Check if the strings are entirely composed of ASCII characters,
    // and default to a simpler algorithm in that case.
    if (sz_isascii(first.data(), first.length()) && sz_isascii(second.data(), second.length()))
        return levenshtein_distance(first, second);

    // Allocate some memory to expand UTF-8 strings into UTF-32.
    sz_size_t const max_utf32_bytes = first.size() * 4 + second.size() * 4;
    sz_rune_t const *const first_utf32 = (sz_rune_t *)alloc.allocate(max_utf32_bytes);
    sz_rune_t const *const second_utf32 = first_utf32 + first.size();

    // Export into UTF-32 buffer.
    sz_rune_length_t rune_length;
    sz_size_t first_length_utf32 = 0, second_length_utf32 = 0;
    for (sz_size_t progress_utf8 = 0, progress_utf32 = 0; progress_utf8 < first.size();
         progress_utf8 += rune_length, ++progress_utf32, ++first_length_utf32)
        sz_rune_parse(first.data() + progress_utf8, first_utf32 + progress_utf32, &rune_length);
    for (sz_size_t progress_utf8 = 0, progress_utf32 = 0; progress_utf8 < second.size();
         progress_utf8 += rune_length, ++progress_utf32, ++second_length_utf32)
        sz_rune_parse(second.data() + progress_utf8, second_utf32 + progress_utf32, &rune_length);

    // Infer the largest distance type we may need fr aggregated error costs.
    // Estimate the maximum dimension of the DP matrix
    sz_size_t const max_dim = sz_max_of_two(first_length_utf32, second_length_utf32) + 1;
    if (max_dim < 256u) {
        sz_u8_t result_u8;
        sz_status_t status = score_diagonally<sz_rune_t, sz_u8_t, uniform_substitution_cost_t, allocator_type_>(
            {first_utf32, first_length_utf32}, {second_utf32, second_length_utf32}, result_u8, 1,
            uniform_substitution_cost_t {}, std::forward<allocator_type_>(alloc));
        if (status == sz_bad_alloc_k) throw std::bad_alloc();
        return result_u8;
    }
    else if (max_dim < 65536u) {
        sz_u16_t result_u16;
        sz_status_t status = score_diagonally<sz_rune_t, sz_u16_t, uniform_substitution_cost_t, allocator_type_>(
            {first_utf32, first_length_utf32}, {second_utf32, second_length_utf32}, result_u16, 1,
            uniform_substitution_cost_t {}, std::forward<allocator_type_>(alloc));
        if (status == sz_bad_alloc_k) throw std::bad_alloc();
        return result_u16;
    }
    else {
        sz_size_t result_size;
        sz_status_t status = score_diagonally<sz_rune_t, sz_size_t, uniform_substitution_cost_t, allocator_type_>(
            {first_utf32, first_length_utf32}, {second_utf32, second_length_utf32}, result_size, 1,
            uniform_substitution_cost_t {}, std::forward<allocator_type_>(alloc));
        if (status == sz_bad_alloc_k) throw std::bad_alloc();
        return result_size;
    }
}

template <                                       //
    typename first_type_,                        //
    typename second_type_,                       //
    typename allocator_type_ = dummy_allocator_t //
    >
inline sz_ssize_t needleman_wunsch_score(                 //
    first_type_ const &first, second_type_ const &second, //
    sz_error_cost_t const *subs, sz_error_cost_t gap,     //
    allocator_type_ &&alloc = allocator_type_ {}) noexcept(false) {

    sz_size_t const first_length = first.length();
    sz_size_t const second_length = second.length();
    if (first_length == 0) return second_length * gap;
    if (second_length == 0) return first_length * gap;

    // Estimate the maximum dimension of the DP matrix
    sz_size_t const max_dim = sz_max_of_two(first_length, second_length) + 1;
    if (max_dim < 256u) {
        sz_u8_t result_u8;
        sz_status_t status = score_diagonally<char, sz_u8_t, lookup_substitution_cost_t, allocator_type_>(
            {first.data(), first_length}, {second.data(), second_length}, result_u8, gap,
            lookup_substitution_cost_t {subs}, std::forward<allocator_type_>(alloc));
        if (status == sz_bad_alloc_k) throw std::bad_alloc();
        return result_u8;
    }
    else if (max_dim < 65536u) {
        sz_u16_t result_u16;
        sz_status_t status = score_diagonally<char, sz_u16_t, lookup_substitution_cost_t, allocator_type_>(
            {first.data(), first_length}, {second.data(), second_length}, result_u16, gap,
            lookup_substitution_cost_t {subs}, std::forward<allocator_type_>(alloc));
        if (status == sz_bad_alloc_k) throw std::bad_alloc();
        return result_u16;
    }
    else {
        sz_size_t result_size;
        sz_status_t status = score_diagonally<char, sz_size_t, lookup_substitution_cost_t, allocator_type_>(
            {first.data(), first_length}, {second.data(), second_length}, result_size, gap,
            lookup_substitution_cost_t {subs}, std::forward<allocator_type_>(alloc));
        if (status == sz_bad_alloc_k) throw std::bad_alloc();
        return result_size;
    }
}

inline void levenshtein_distances() {}
inline void levenshtein_distances_utf8() {}
inline void needleman_wunsch_scores() {}

} // namespace openmp
} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_SIMILARITY_HPP_