/**
 *  @brief  OpenMP-accelerated string similarity scores in C++.
 *  @file   similarity.hpp
 *  @author Ash Vardanian
 *
 *  Includes core APIs:
 *
 *  - `sz::levenshtein_distance` & `sz::levenshtein_distance_utf8` for Levenshtein edit-scores.
 *  - `sz::needleman_wunsch_score` for weighted Needleman-Wunsch global alignment.
 *  - `sz::smith_waterman_score` for weighted Smith-Waterman local alignment.
 *
 *  Also includes their batch-capable and parallel versions:
 *
 *  - `sz::levenshtein_scores` & `sz::levenshtein_scores_utf8` for Levenshtein edit-scores.
 *  - `sz::needleman_wunsch_scores` for weighted Needleman-Wunsch global alignment.
 *  - `sz::smith_waterman_scores` for weighted Smith-Waterman local alignment.
 *
 *  Those are mostly providing specialized overloads of the @b `sz::score_diagonally` wavefront-like template
 *  or @b `sz::score_horizontally` conventional Wagner-Fischer algorithm template, that may be more suitable
 *  for large 256x256 substitution matrices on x86 CPUs.
 *
 *  @section    Why not reimplement this in pure C 99?
 *
 *  In bioinformatics and other string processing applications we are exposed to too much variability in the
 *  form of inputs and the kind of processing optimizations we want to apply. Many of those optimizations are
 *  independent from the core logic and can be composed together in a modular way. Doing that in C 99 would
 *  require a lot of boilerplate code and would be hard to maintain.
 *
 *  - The core algorithm for byte-level and UTF-32 alignment scoring is identical.
 *  - Local and global alignment algorithms are almost identical, only differing in one more `min`/`max`
 *    operation and the way the top row and left column of the DP matrix are initialized.
 *  - Different CPU cores may be scheduled to process different pairs individually, or collaborate to
 *    align very large strings, still using the same core logic.
 *  - Different substitution cost models require very different SIMD implementations in case of uniform
 *    costs, DNA scoring with 4x4 matrix, protein scoring with 20x20 matrix, or custom costs.
 *
 *  Each of those may just be a 2 line change in the core logic, but can produce a @b 1000 lines of boilerplate!
 */
#ifndef STRINGZILLA_SIMILARITY_HPP_
#define STRINGZILLA_SIMILARITY_HPP_

#include "types.hpp"

namespace ashvardanian {
namespace stringzilla {
namespace openmp {

/**
 *  @brief  An operator to be applied to be applied to all 2x2 blocks of the DP matrix to produce
 *          the bottom-right value from the 3x others in case of Global Alignment algorithms, like
 *          the Needleman-Wunsch or Levenshtein distance calculations.
 *
 *  It updates the internal state to remember the last calculated value, as in Global Alignment it's
 *  always in the bottom-right corner of the DP matrix, which is evaluated last.
 */
template <                                                    //
    typename first_iterator_type_ = char const *,             //
    typename second_iterator_type_ = char const *,            //
    typename score_type_ = sz_size_t,                         //
    typename substituter_type_ = uniform_substitution_cost_t, //
    sz_capability_t capability_ = sz_cap_serial_k             //
    >
struct global_aligner {

    using first_iterator_type = first_iterator_type_;
    using second_iterator_type = second_iterator_type_;
    using score_type = score_type_;
    using substituter_type = substituter_type_;
    static constexpr sz_capability_t capability = capability_;

    substituter_type substituter_ {};
    error_cost_t gap_cost_ {1};
    score_type last_cell_ {0};

    global_aligner() = default;
    global_aligner(substituter_type &&substituter, error_cost_t gap_cost) noexcept
        : substituter_(std::move(substituter)), gap_cost_(gap_cost) {}

    static constexpr bool is_parallel() { return capability & sz_cap_parallel_k; }

    /**
     *  @brief Initializes a boundary value within a certain diagonal.
     *  @note Should only be called for the diagonals outside of the bottom-right triangle.
     *  @note Should only be called for the top row and left column of the matrix.
     */
    void init(score_type &cell, sz_size_t diagonal_index) const noexcept { cell = gap_cost_ * diagonal_index; }

    /**
     *  @brief Extract the final result of the scoring operation which will be always in the bottom-right corner.
     */
    score_type score() const noexcept { return last_cell_; }

    /**
     *  @brief Computes one diagonal of the DP matrix, using the results of the previous 2x diagonals.
     *  @param first_reversed_slice The first string, @b reversed.
     *  @param second_slice The second string.
     *  @param n The length of the diagonal to evaluate and the number of characters to compare from each string.
     */
    void operator()(                                                                             //
        char_type const *first_reversed_slice, char_type const *second_slice, sz_size_t const n, //
        score_type const *scores_pre_substitution,                                               //
        score_type const *scores_pre_insertion, score_type const *scores_pre_deletion,           //
        score_type *scores_new) noexcept {

#pragma omp parallel for simd schedule(dynamic, 1) if (is_parallel())
        for (sz_size_t i = 0; i < n; ++i) {
            score_type score_pre_substitution = scores_pre_substitution[i];
            score_type score_pre_insertion = scores_pre_insertion[i];
            score_type score_pre_deletion = scores_pre_deletion[i];

            // ? Note that here we are still traversing both buffers in the same order,
            // ? because one of the strings has been reversed beforehand.
            error_cost_t cost_of_substitution = substituter_(first_reversed_slice[i], second_slice[i]);
            score_type score_if_substitution = score_pre_substitution + cost_of_substitution;
            score_type score_if_deletion_or_insertion =
                sz_min_of_two(score_pre_deletion, score_pre_insertion) + gap_cost_;
            score_type cell_score = sz_min_of_two(score_if_deletion_or_insertion, score_if_substitution);
            scores_new[i] = cell_score;
        }

        // The last element of the last chunk is the result of the global alignment.
        last_cell_ = scores_new[n - 1];
    }
};

/**
 *  @brief  An operator to be applied to be applied to all 2x2 blocks of the DP matrix to produce
 *          the bottom-right value from the 3x others in case of Global Alignment algorithms, like
 *          the Smith-Waterman score.
 *
 *  It updates the internal state to remember the minimum/maximum calculated value, as in Local Alignment
 *  it's always in the bottom-right corner of the DP matrix, which is evaluated last.
 */
template <                                                    //
    typename first_iterator_type_ = char const *,             //
    typename second_iterator_type_ = char const *,            //
    typename score_type_ = sz_size_t,                         //
    typename substituter_type_ = uniform_substitution_cost_t, //
    sz_capability_t capability_ = sz_cap_serial_k             //
    >
struct local_aligner {

    using first_iterator_type = first_iterator_type_;
    using second_iterator_type = second_iterator_type_;
    using score_type = score_type_;
    using substituter_type = substituter_type_;
    static constexpr sz_capability_t capability = capability_;

    substituter_type substituter_ {};
    error_cost_t gap_cost_ {-1};
    score_type max_cell_ {0};

    local_aligner() = default;
    local_aligner(substituter_type &&substituter, error_cost_t gap_cost) noexcept
        : substituter_(std::move(substituter)), gap_cost_(gap_cost) {}

    static constexpr bool is_parallel() { return capability & sz_cap_parallel_k; }

    /**
     *  @brief Initializes a boundary value within a certain diagonal.
     *  @note Should only be called for the diagonals outside of the bottom-right triangle.
     *  @note Should only be called for the top row and left column of the matrix.
     */
    void init(score_type &cell, sz_size_t /* diagonal_index */) const noexcept { cell = 0; }

    /**
     *  @brief Extract the final result of the scoring operation which will be maximum encountered value.
     */
    score_type score() const noexcept { return max_cell_; }

    void operator()(char_type const *first_reversed_slice, char_type const *second_slice, sz_size_t const n,
                    score_type const *scores_pre_substitution, score_type const *scores_pre_insertion,
                    score_type const *scores_pre_deletion, score_type *scores_new) noexcept {

#pragma omp parallel for simd schedule(dynamic, 1) if (is_parallel())
        for (sz_size_t i = 0; i < n; ++i) {
            score_type score_pre_substitution = scores_pre_substitution[i];
            score_type score_pre_insertion = scores_pre_insertion[i];
            score_type score_pre_deletion = scores_pre_deletion[i];

            // ? Note that here we are still traversing both buffers in the same order,
            // ? because one of the strings has been reversed beforehand.
            error_cost_t substitution = substituter_(first_reversed_slice[i], second_slice[i]);
            score_type score_if_substitution = score_pre_substitution + cost_of_substitution;
            score_type score_if_deletion_or_insertion =
                sz_min_of_two(score_pre_deletion, score_pre_insertion) + gap_cost_;
            score_type cell_score = sz_min_of_three(score_if_deletion_or_insertion, score_if_substitution, 0);
            scores_new[i] = cell_score;

            // Update the global maximum score if this cell beats it.
#pragma omp critical
            {
                if (cell_score > max_cell_) max_cell_ = cell_score;
            }
        }
    }
};

/**
 *  @brief  Alignment Score and Edit Distance algorithm evaluating the Dynamic Programming matrix
 *          @b three skewed (reverse) diagonals at a time on a CPU, leveraging OpenMP for parallelization.
 *          Can be used for both global and local alignment, like Needleman-Wunsch and Smith-Waterman.
 *
 *  ? There are smarter algorithms for computing the Levenshtein distance, mostly based on bit-level operations.
 *  ? Those, however, don't generalize well to arbitrary length inputs or non-uniform substitution costs.
 *  ? This algorithm provides a more flexible baseline implementation for future SIMD and GPGPU optimizations.
 *  ! This algorithm can't handle different "gap opening" and "gap extension" costs, those need 3x more memory.
 *  ! This algorithm may be suboptimal for very small strings, where a conventional Wagner-Fischer algorithm
 *  ! with horizontal traversal order and fewer loops may be faster. That one, however, can't be parallel!
 *
 *  @param[in] first The first string.
 *  @param[in] second The second string.
 *  @param[out] result_ref Location to dump the calculated score.
 *  @param[in] gap_cost The uniform cost of a gap (insertion or deletion).
 *  @param[in] substituter A commutative function returning the cost of substituting one char with another.
 *  @param[in] alloc A default-constructible allocator for the internal buffers.
 *
 *  @tparam char_type_ The type of the characters in the strings, generally `char` or @b `rune_t` for UTF-8.
 *  @tparam score_type_ The smallest type that can hold the distance, ideally `sz_i8_t` or `sz_u8_t`.
 *  @tparam substituter_ A callable type that takes two characters and returns the substitution cost.
 *  @tparam allocator_type_ A default-constructible allocator type for the internal buffers.
 *  @tparam multi_threaded_ Whether to use OpenMP for @b multi-threading or just vectorization.
 *  @tparam global_alignment_ Whether to use the global alignment algorithm or the local one.
 *
 *  @note   The API of this algorithm is a bit weird, but it's designed to minimize the reliance on the definitions
 *          in the `stringzilla.hpp` header, making compilation times shorter for the end-user.
 *  @sa     For lower-level API, check `sz_levenshtein_distance[_utf8]` and `sz_needleman_wunsch_score`.
 *  @sa     For simplicity, use the `sz::levenshtein_distance[_utf8]` and `sz::needleman_wunsch_score`.
 *  @sa     For bulk API, use `sz::levenshtein_scores[_utf8]`.
 */
template <                                                 //
    sz_capability_t capability_ = sz_cap_serial_k,         //
    sz_alignment_locality_t locality_ = sz_align_global_k, //
    typename char_type_ = char,                            //
    typename score_type_ = sz_size_t,                      //
    typename substituter_ = uniform_substitution_cost_t,   //
    typename allocator_type_ = dummy_alloc_t               //
    >
sz_status_t score_diagonally(                                    //
    span<char_type_ const> first, span<char_type_ const> second, //
    score_type_ &result_ref,                                     //
    sz_error_cost_t gap_cost = 1,                                //
    substituter_ &&substituter = uniform_substitution_cost_t {}, //
    allocator_type_ &&alloc = allocator_type_ {}                 //
    ) noexcept {

    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    using allocated_type = typename allocator_type_::value_type;
    static_assert(sizeof(allocated_type) == sizeof(char), "Allocator must be byte-aligned");
    using char_type = char_type_;
    using score_type = score_type_;

    using aligner_t = global_aligner<char_type, score_type, substituter_, capability_>;

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
    sz_size_t const buffer_length = sizeof(score_type) * max_diagonal_length * 3 + shorter_length * sizeof(char_type);
    score_type *const buffer = (score_type *)alloc.allocate(buffer_length);
    if (!buffer) return sz_bad_alloc_k;

    // The next few pointers will be swapped around.
    score_type *previous_scores = buffer;
    score_type *current_scores = previous_scores + max_diagonal_length;
    score_type *next_scores = current_scores + max_diagonal_length;
    char_type *const shorter_reversed = (char_type *)(next_scores + max_diagonal_length);

    // Export the reversed string into the buffer.
    for (sz_size_t i = 0; i != shorter_length; ++i) shorter_reversed[i] = shorter[shorter_length - 1 - i];

    // Initialize the first two diagonals:
    aligner_t diagonal_aligner;
    diagonal_aligner.init(previous_scores[0], 0);
    diagonal_aligner.init(current_scores[0], 1);
    diagonal_aligner.init(current_scores[1], 1);

    // We skip diagonals 0 and 1, as they are trivial.
    // We will start with diagonal 2, which has length 3, with the first and last elements being preset,
    // so we are effectively computing just one value, as will be marked by a single set bit in
    // the `next_diagonal_mask` on the very first iteration.
    sz_size_t next_diagonal_index = 2;

    // Progress through the upper-left triangle of the Levenshtein matrix.
    for (; next_diagonal_index < shorter_dim; ++next_diagonal_index) {

        sz_size_t const next_diagonal_length = next_diagonal_index + 1;
        diagonal_aligner(                                                //
            shorter_reversed + shorter_length - next_diagonal_index + 1, // first sequence of characters
            longer,                                                      // second sequence of characters
            next_diagonal_length - 2,           // number of elements to compute with the `diagonal_aligner`
            previous_scores,                    // costs pre substitution
            current_scores, current_scores + 1, // costs pre insertion/deletion
            next_scores + 1);

        // Don't forget to populate the first row and the first column of the Levenshtein matrix.
        diagonal_aligner.init(next_scores[0], next_diagonal_index);
        diagonal_aligner.init(next_scores[next_diagonal_length - 1], next_diagonal_index);

        // Perform a circular rotation of those buffers, to reuse the memory.
        score_type *temporary = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = temporary;
    }

    // Now let's handle the anti-diagonal band of the matrix, between the top and bottom-right triangles.
    for (; next_diagonal_index < longer_dim; ++next_diagonal_index) {

        sz_size_t const next_diagonal_length = shorter_dim;
        diagonal_aligner(                                        //
            shorter_reversed + shorter_length - shorter_dim + 1, // first sequence of characters
            longer + next_diagonal_index - shorter_dim,          // second sequence of characters
            next_diagonal_length - 1,           // number of elements to compute with the `diagonal_aligner`
            previous_scores,                    // costs pre substitution
            current_scores, current_scores + 1, // costs pre insertion/deletion
            next_scores);

        // Don't forget to populate the first row of the Levenshtein matrix.
        diagonal_aligner.init(next_scores[next_diagonal_length - 1], next_diagonal_index);

        // Perform a circular rotation of those buffers, to reuse the memory, this time, with a shift,
        // dropping the first element in the current array.
        score_type *temporary = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = temporary;

        // ! Drop the first entry among the current scores.
        sz_move((sz_ptr_t)(previous_scores), (sz_ptr_t)(previous_scores + 1),
                (max_diagonal_length - 1) * sizeof(score_type));
    }

    // Now let's handle the bottom-right triangle of the matrix.
    for (; next_diagonal_index < diagonals_count; ++next_diagonal_index) {

        sz_size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
        diagonal_aligner(                                        //
            shorter_reversed + shorter_length - shorter_dim + 1, // first sequence of characters
            longer + next_diagonal_index - shorter_dim,          // second sequence of characters
            next_diagonal_length,               // number of elements to compute with the `diagonal_aligner`
            previous_scores,                    // costs pre substitution
            current_scores, current_scores + 1, // costs pre insertion/deletion
            next_scores);

        // Perform a circular rotation of those buffers, to reuse the memory, this time, with a shift,
        // dropping the first element in the current array.
        score_type *temporary = previous_scores;

        // ! Drop the first entry among the current scores.
        // ! Assuming every next diagonal is shorter by one element, we don't need a full-blown `sz_move`.
        // ! to shift the array by one element.
        previous_scores = current_scores + 1;
        current_scores = next_scores;
        next_scores = temporary;
    }

    // Export the scalar before `free` call.
    result_ref = diagonal_aligner.score();
    alloc.deallocate((allocated_type *)buffer, buffer_length);
    return sz_success_k;
}

/**
 *  @brief  Alignment Score and Edit Distance algorithm evaluating the Dynamic Programming matrix
 *          @b two rows at a time on a CPU, using the conventional Wagner Fischer algorithm.
 *
 *  ! This algorithm can't handle different "gap opening" and "gap extension" costs, those need 3x more memory.
 *  ! This algorithm doesn't parallelize well, check out the diagonal variants!
 *
 *  @param[in] first The first string.
 *  @param[in] second The second string.
 *  @param[out] result_ref Location to dump the calculated score.
 *  @param[in] gap_cost The uniform cost of a gap (insertion or deletion).
 *  @param[in] substituter A commutative function returning the cost of substituting one char with another.
 *  @param[in] alloc A default-constructible allocator for the internal buffers.
 *
 *  @tparam char_type_ The type of the characters in the strings, generally `char` or @b `rune_t` for UTF-8.
 *  @tparam score_type_ The smallest type that can hold the distance, ideally `sz_i8_t` or `sz_u8_t`.
 *  @tparam substituter_ A callable type that takes two characters and returns the substitution cost.
 *  @tparam allocator_type_ A default-constructible allocator type for the internal buffers.
 *  @tparam multi_threaded_ Whether to use OpenMP for @b multi-threading or just vectorization.
 *  @tparam global_alignment_ Whether to use the global alignment algorithm or the local one.
 *
 *  @note   The API of this algorithm is a bit weird, but it's designed to minimize the reliance on the definitions
 *          in the `stringzilla.hpp` header, making compilation times shorter for the end-user.
 *  @sa     For lower-level API, check `sz_levenshtein_distance[_utf8]` and `sz_needleman_wunsch_score`.
 *  @sa     For simplicity, use the `sz::levenshtein_distance[_utf8]` and `sz::needleman_wunsch_score`.
 *  @sa     For bulk API, use `sz::levenshtein_scores[_utf8]`.
 */
template <                                                 //
    sz_capability_t capability_ = sz_cap_serial_k,         //
    sz_alignment_locality_t locality_ = sz_align_global_k, //
    typename char_type_ = char,                            //
    typename score_type_ = sz_size_t,                      //
    typename substituter_ = uniform_substitution_cost_t,   //
    typename allocator_type_ = dummy_alloc_t               //
    >
sz_status_t score_horizontally(                                  //
    span<char_type_ const> first, span<char_type_ const> second, //
    score_type_ &result_ref,                                     //
    sz_error_cost_t gap_cost = 1,                                //
    substituter_ &&substituter = uniform_substitution_cost_t {}, //
    allocator_type_ &&alloc = allocator_type_ {}                 //
    ) noexcept {

    static_assert((capability_ & sz_cap_parallel_k) == 0, "This algorithm is not parallelized!");

    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    using allocated_type = typename allocator_type_::value_type;
    static_assert(sizeof(allocated_type) == sizeof(char), "Allocator must be byte-aligned");
    using char_type = char_type_;
    using score_type = score_type_;

    using aligner_t = global_aligner<char_type, score_type, substituter_, capability_>;

    // Make sure the size relation between the strings is correct.
    char_type const *shorter = first.data(), *longer = second.data();
    sz_size_t shorter_length = first.size(), longer_length = second.size();
    if (shorter_length > longer_length) {
        std::swap(shorter, longer);
        std::swap(shorter_length, longer_length);
    }

    // We are going to store 2 rows of the matrix. It will be either 2 rows of length `shorter_length + 1`
    // or 2 rows of length `longer_length + 1`, depending on our preference - either minimizing the memory
    // consumption or the inner loop performance.
    sz_size_t const shorter_dim = shorter_length + 1;
    sz_size_t const longer_dim = longer_length + 1;

    // We decide to use less memory!
    sz_size_t const buffer_length = sizeof(score_type) * shorter_dim * 2;
    score_type *const buffer = (score_type *)alloc.allocate(buffer_length);
    if (!buffer) return sz_bad_alloc_k;

    // The next few pointers will be swapped around.
    score_type *previous_scores = buffer;
    score_type *current_scores = previous_scores + shorter_dim;

    // Initialize the first row:
    aligner_t horizontal_aligner;
    for (sz_size_t col_idx = 0; col_idx < shorter_dim; ++col_idx)
        horizontal_aligner.init(previous_scores[col_idx], col_idx);

    // Progress through the matrix row-by-row:
    for (sz_size_t row_idx = 1; row_idx < longer_dim; ++row_idx) {

        // Don't forget to populate the first column of each row:
        horizontal_aligner.init(next_scores[0], 1);

        horizontal_aligner(           //
            repeat {&shorter[i - 1]}, // first sequence of characters
            longer,                   // second sequence of characters
            next_diagonal_length - 2, // number of elements to compute with the `horizontal_aligner`
            previous_scores,          // costs pre substitution
            previous_scores + 1,      // costs pre insertion
            current_scores,           // costs pre deletion
            current_scores + 1);

        // Perform a circular rotation of those buffers, to reuse the memory.
        score_type *temporary = previous_scores;
        previous_scores = current_scores;
        current_scores = next_scores;
        next_scores = temporary;
    }

    // Export the scalar before `free` call.
    result_ref = horizontal_aligner.score();
    alloc.deallocate((allocated_type *)buffer, buffer_length);
    return sz_success_k;
}

/**
 *  @brief Computes the @b byte-level Levenshtein distance between two strings using the OpenMP backend.
 *  @param[in] first The first string.
 *  @param[in] second The second string.
 *  @param[in] alloc An allocator for the internal buffers.
 *  @return The Levenshtein distance between the two strings.
 *  @throws `std::bad_alloc` if the allocator fails to allocate memory.
 *  @sa `levenshtein_distance_utf8` for UTF-8 strings.
 *  @sa `score_diagonally` for the core algorithm.
 */
template < //
    sz_capability_t capability_ = sz_cap_serial_k,
    typename first_type_ = span<char const>,  //
    typename second_type_ = span<char const>, //
    typename allocator_type_ = dummy_alloc_t  //
    >
inline sz_size_t levenshtein_distance( //
    first_type_ const &first, second_type_ const &second,
    allocator_type_ &&alloc = allocator_type_ {}) noexcept(false) {

    sz_size_t const first_length = first.length();
    sz_size_t const second_length = second.length();
    if (first_length == 0) return second_length;
    if (second_length == 0) return first_length;

    // Estimate the maximum dimension of the DP matrix
    sz_size_t const min_dim = sz_min_of_two(first_length, second_length) + 1;
    sz_size_t const max_dim = sz_max_of_two(first_length, second_length) + 1;

    // When dealing with very small inputs, we may want to use a simpler Wagner-Fischer algorithm.
    if (min_dim < 16u) {
        sz_u8_t result_u8;
        sz_status_t status = score_horizontally<capability_, sz_align_global_k, char, sz_u8_t,
                                                uniform_substitution_cost_t, allocator_type_>(
            {first.data(), first_length}, {second.data(), second_length}, result_u8, 1, uniform_substitution_cost_t {},
            std::forward<allocator_type_>(alloc));
        if (status == sz_bad_alloc_k) throw std::bad_alloc();
        return result_u8;
    }

    // When dealing with larger arrays, we need to differentiate kernel with different cost aggregation types.
    // Smaller ones will overflow for larger inputs, but using larger-than-needed types will waste memory.
    if (max_dim < 256u) {
        sz_u8_t result_u8;
        sz_status_t status =
            score_diagonally<capability_, sz_align_global_k, char, sz_u8_t, uniform_substitution_cost_t,
                             allocator_type_>({first.data(), first_length}, {second.data(), second_length}, result_u8,
                                              1, uniform_substitution_cost_t {}, std::forward<allocator_type_>(alloc));
        if (status == sz_bad_alloc_k) throw std::bad_alloc();
        return result_u8;
    }
    else if (max_dim < 65536u) {
        sz_u16_t result_u16;
        sz_status_t status =
            score_diagonally<capability_, sz_align_global_k, char, sz_u16_t, uniform_substitution_cost_t,
                             allocator_type_>({first.data(), first_length}, {second.data(), second_length}, result_u16,
                                              1, uniform_substitution_cost_t {}, std::forward<allocator_type_>(alloc));
        if (status == sz_bad_alloc_k) throw std::bad_alloc();
        return result_u16;
    }
    else {
        sz_size_t result_size;
        sz_status_t status =
            score_diagonally<capability_, sz_align_global_k, char, sz_size_t, uniform_substitution_cost_t,
                             allocator_type_>({first.data(), first_length}, {second.data(), second_length}, result_size,
                                              1, uniform_substitution_cost_t {}, std::forward<allocator_type_>(alloc));
        if (status == sz_bad_alloc_k) throw std::bad_alloc();
        return result_size;
    }
}

/**
 *  @brief Computes the @b rune-level Levenshtein distance between two UTF-8 strings using the OpenMP backend.
 *  @param[in] first The first string.
 *  @param[in] second The second string.
 *  @param[in] alloc An allocator for the internal buffers.
 *  @return The Levenshtein distance between the two strings.
 *  @throws `std::bad_alloc` if the allocator fails to allocate memory.
 *  @sa `levenshtein_distance` for binary strings.
 *  @sa `score_diagonally` for the core algorithm.
 */
template < //
    sz_capability_t capability_ = sz_cap_serial_k,
    typename first_type_ = span<char const>,  //
    typename second_type_ = span<char const>, //
    typename allocator_type_ = dummy_alloc_t  //
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
        sz_status_t status = score_diagonally<capability_, sz_align_global_k, sz_rune_t, sz_u8_t,
                                              uniform_substitution_cost_t, allocator_type_>(
            {first_utf32, first_length_utf32}, {second_utf32, second_length_utf32}, result_u8, 1,
            uniform_substitution_cost_t {}, std::forward<allocator_type_>(alloc));
        if (status == sz_bad_alloc_k) throw std::bad_alloc();
        return result_u8;
    }
    else if (max_dim < 65536u) {
        sz_u16_t result_u16;
        sz_status_t status = score_diagonally<capability_, sz_align_global_k, sz_rune_t, sz_u16_t,
                                              uniform_substitution_cost_t, allocator_type_>(
            {first_utf32, first_length_utf32}, {second_utf32, second_length_utf32}, result_u16, 1,
            uniform_substitution_cost_t {}, std::forward<allocator_type_>(alloc));
        if (status == sz_bad_alloc_k) throw std::bad_alloc();
        return result_u16;
    }
    else {
        sz_size_t result_size;
        sz_status_t status = score_diagonally<capability_, sz_align_global_k, sz_rune_t, sz_size_t,
                                              uniform_substitution_cost_t, allocator_type_>(
            {first_utf32, first_length_utf32}, {second_utf32, second_length_utf32}, result_size, 1,
            uniform_substitution_cost_t {}, std::forward<allocator_type_>(alloc));
        if (status == sz_bad_alloc_k) throw std::bad_alloc();
        return result_size;
    }
}

/**
 *  @brief Computes the @b byte-level Needleman-Wunsch score between two strings using the OpenMP backend.
 *  @param[in] first The first string.
 *  @param[in] second The second string.
 *  @param[in] alloc An allocator for the internal buffers.
 *  @return The Needleman-Wunsch global alignment score between the two strings.
 *  @throws `std::bad_alloc` if the allocator fails to allocate memory.
 *  @sa `levenshtein_distance` for uniform substitution and gap costs.
 *  @sa `score_diagonally` for the core algorithm.
 */
template < //
    sz_capability_t capability_ = sz_cap_serial_k,
    typename first_type_ = span<char const>,  //
    typename second_type_ = span<char const>, //
    typename allocator_type_ = dummy_alloc_t  //
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
        sz_status_t status =
            score_diagonally<capability_, sz_align_local_k, char, sz_u8_t, lookup_substitution_cost_t, allocator_type_>(
                {first.data(), first_length}, {second.data(), second_length}, result_u8, gap,
                lookup_substitution_cost_t {subs}, std::forward<allocator_type_>(alloc));
        if (status == sz_bad_alloc_k) throw std::bad_alloc();
        return result_u8;
    }
    else if (max_dim < 65536u) {
        sz_u16_t result_u16;
        sz_status_t status = score_diagonally<capability_, sz_align_local_k, char, sz_u16_t, lookup_substitution_cost_t,
                                              allocator_type_>(
            {first.data(), first_length}, {second.data(), second_length}, result_u16, gap,
            lookup_substitution_cost_t {subs}, std::forward<allocator_type_>(alloc));
        if (status == sz_bad_alloc_k) throw std::bad_alloc();
        return result_u16;
    }
    else {
        sz_size_t result_size;
        sz_status_t status = score_diagonally<capability_, sz_align_local_k, char, sz_size_t,
                                              lookup_substitution_cost_t, allocator_type_>(
            {first.data(), first_length}, {second.data(), second_length}, result_size, gap,
            lookup_substitution_cost_t {subs}, std::forward<allocator_type_>(alloc));
        if (status == sz_bad_alloc_k) throw std::bad_alloc();
        return result_size;
    }
}

inline void levenshtein_scores() {}
inline void levenshtein_scores_utf8() {}
inline void needleman_wunsch_scores() {}

} // namespace openmp
} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_SIMILARITY_HPP_