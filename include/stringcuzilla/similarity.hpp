/**
 *  @brief  OpenMP-accelerated string similarity scores in C++.
 *  @file   similarity.hpp
 *  @author Ash Vardanian
 *
 *  Includes core APIs, defined as the following template objects:
 *
 *  - `sz::levenshtein_distance` & `sz::levenshtein_distance_utf8` for Levenshtein edit-scores.
 *  - `sz::needleman_wunsch_score` for weighted Needleman-Wunsch @b (NW) global alignment.
 *  - `sz::smith_waterman_score` for weighted Smith-Waterman @b (SW) local alignment.
 *
 *  Also includes their batch-capable and parallel versions:
 *
 *  - `sz::levenshtein_scores` & `sz::levenshtein_scores_utf8` for Levenshtein edit-scores.
 *  - `sz::needleman_wunsch_scores` for weighted Needleman-Wunsch global alignment.
 *  - `sz::smith_waterman_scores` for weighted Smith-Waterman local alignment.
 *
 *  Those are mostly providing specialized overloads of the @b `sz::diagonal_walker` wavefront-like template
 *  or @b `sz::horizontal_walker` conventional Wagner-Fischer algorithm template, that may be more suitable
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
 *    operation and the way the top row and left column of the Dynamic Programming @b (DP) matrix are initialized.
 *  - Different CPU cores may be scheduled to process different pairs individually, or collaborate to
 *    align very large strings, still using the same core logic.
 *  - Different substitution cost models require very different SIMD implementations in case of uniform
 *    costs, DNA scoring with 4x4 matrix, protein scoring with 20x20 matrix, or custom costs.
 *
 *  Each of those may just be a 2 line change in the core logic, but can produce a @b 1000 lines of boilerplate!
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
 *  TODO: For @b sparse algorithms, the algorithms are constructed differently.
 */
#ifndef STRINGZILLA_SIMILARITY_HPP_
#define STRINGZILLA_SIMILARITY_HPP_

#include "stringzilla/memory.h"  // `sz_move`
#include "stringzilla/types.hpp" // `sz::error_cost_t`

#include <atomic> // `std::atomic` to synchronize OpenMP threads

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
template <                                                     //
    typename first_iterator_type_ = char const *,              //
    typename second_iterator_type_ = char const *,             //
    typename score_type_ = sz_size_t,                          //
    typename substituter_type_ = error_costs_uniform_t,        //
    sz_alignment_direction_t direction_ = sz_align_maximize_k, //
    sz_capability_t capability_ = sz_cap_serial_k              //
    >
struct global_scorer {

    using first_iterator_t = first_iterator_type_;
    using second_iterator_t = second_iterator_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    static constexpr sz_capability_t capability_k = capability_;

    using first_char_t = typename std::iterator_traits<first_iterator_t>::value_type;
    using second_char_t = typename std::iterator_traits<second_iterator_t>::value_type;
    static_assert(is_same_type<first_char_t, second_char_t>::value, "String characters must be of the same type.");
    using char_t = first_char_t;

  private:
    substituter_t substituter_ {};
    error_cost_t gap_cost_ {1};
    score_t last_cell_ {0};

    static inline score_t pick_best(score_t a, score_t b) noexcept {
        if constexpr (direction_ == sz_align_minimize_k) { return sz_min_of_two(a, b); }
        else { return sz_max_of_two(a, b); }
    }

  public:
    global_scorer() = default;
    global_scorer(substituter_t substituter, error_cost_t gap_cost) noexcept
        : substituter_(substituter), gap_cost_(gap_cost) {}

    static constexpr bool is_parallel() { return capability_k & sz_cap_parallel_k; }

    /**
     *  @brief Initializes a boundary value within a certain diagonal.
     *  @note Should only be called for the diagonals outside of the bottom-right triangle.
     *  @note Should only be called for the top row and left column of the matrix.
     */
    void init(score_t &cell, sz_size_t diagonal_index) const noexcept { cell = gap_cost_ * diagonal_index; }

    /**
     *  @brief Extract the final result of the scoring operation which will be always in the bottom-right corner.
     */
    score_t score() const noexcept { return last_cell_; }

    /**
     *  @brief Computes one diagonal of the DP matrix, using the results of the previous 2x diagonals.
     *  @param first_reversed_slice The first string, @b reversed.
     *  @param second_slice The second string.
     *  @param n The length of the diagonal to evaluate and the number of characters to compare from each string.
     */
    void operator()(                                                                        //
        first_iterator_t first_reversed_slice, second_iterator_t second_slice, sz_size_t n, //
        score_t const *scores_pre_substitution, score_t const *scores_pre_insertion, score_t const *scores_pre_deletion,
        score_t *scores_new) noexcept {

#pragma omp parallel for simd schedule(dynamic, 1) if (is_parallel())
        for (sz_size_t i = 0; i < n; ++i) {
            score_t score_pre_substitution = scores_pre_substitution[i];
            score_t score_pre_insertion = scores_pre_insertion[i];
            score_t score_pre_deletion = scores_pre_deletion[i];

            // ? Note that here we are still traversing both buffers in the same order,
            // ? because one of the strings has been reversed beforehand.
            error_cost_t cost_of_substitution = substituter_(first_reversed_slice[i], second_slice[i]);
            score_t score_if_substitution = score_pre_substitution + cost_of_substitution;
            score_t score_if_deletion_or_insertion = pick_best(score_pre_deletion, score_pre_insertion) + gap_cost_;
            score_t cell_score = pick_best(score_if_deletion_or_insertion, score_if_substitution);
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
template <                                                     //
    typename first_iterator_type_ = char const *,              //
    typename second_iterator_type_ = char const *,             //
    typename score_type_ = sz_size_t,                          //
    typename substituter_type_ = error_costs_uniform_t,        //
    sz_alignment_direction_t direction_ = sz_align_maximize_k, //
    sz_capability_t capability_ = sz_cap_serial_k              //
    >
struct local_scorer {

    using first_iterator_t = first_iterator_type_;
    using second_iterator_t = second_iterator_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    static constexpr sz_capability_t capability_k = capability_;

    using first_char_t = typename std::iterator_traits<first_iterator_t>::value_type;
    using second_char_t = typename std::iterator_traits<second_iterator_t>::value_type;
    static_assert(std::is_same<first_char_t, second_char_t>(), "String characters must be of the same type.");
    using char_t = first_char_t;

  private:
    substituter_t substituter_ {};
    error_cost_t gap_cost_ {1};
    score_t best_score_ {0};

    static inline score_t pick_best(score_t a, score_t b) noexcept {
        if constexpr (direction_ == sz_align_minimize_k) { return sz_min_of_two(a, b); }
        else { return sz_max_of_two(a, b); }
    }

  public:
    local_scorer() = default;
    local_scorer(substituter_t substituter, error_cost_t gap_cost) noexcept
        : substituter_(substituter), gap_cost_(gap_cost) {}

    static constexpr bool is_parallel() { return capability_k & sz_cap_parallel_k; }

    /**
     *  @brief Initializes a boundary value within a certain diagonal.
     *  @note Should only be called for the diagonals outside of the bottom-right triangle.
     *  @note Should only be called for the top row and left column of the matrix.
     */
    void init(score_t &cell, sz_size_t /* diagonal_index */) const noexcept { cell = 0; }

    /**
     *  @brief Extract the final result of the scoring operation which will be maximum encountered value.
     */
    score_t score() const noexcept { return best_score_; }

    void operator()(                                                                              //
        first_iterator_t first_reversed_slice, second_iterator_t second_slice, sz_size_t const n, //
        score_t const *scores_pre_substitution, score_t const *scores_pre_insertion, score_t const *scores_pre_deletion,
        score_t *scores_new) noexcept {

#pragma omp parallel for schedule(dynamic, 1) if (is_parallel())
        for (sz_size_t i = 0; i < n; ++i) {
            score_t score_pre_substitution = scores_pre_substitution[i];
            score_t score_pre_insertion = scores_pre_insertion[i];
            score_t score_pre_deletion = scores_pre_deletion[i];

            // ? Note that here we are still traversing both buffers in the same order,
            // ? because one of the strings has been reversed beforehand.
            error_cost_t cost_of_substitution = substituter_(first_reversed_slice[i], second_slice[i]);
            score_t score_if_substitution = score_pre_substitution + cost_of_substitution;
            score_t score_if_deletion_or_insertion = pick_best(score_pre_deletion, score_pre_insertion) + gap_cost_;
            score_t score_if_substitution_or_reset = pick_best(score_if_substitution, 0);
            score_t cell_score = pick_best(score_if_deletion_or_insertion, score_if_substitution_or_reset);
            scores_new[i] = cell_score;

            // Update the global maximum score if this cell beats it.
#pragma omp critical
            { best_score_ = pick_best(best_score_, cell_score); }
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
 *  @tparam char_type_ The type of the characters in the strings, generally `char` or @b `rune_t` for UTF-8.
 *  @tparam score_type_ The smallest type that can hold the distance, ideally `sz_i8_t` or `sz_u8_t`.
 *  @tparam substituter_type_ A callable type that takes two characters and returns the substitution cost.
 *  @tparam allocator_type_ A default-constructible allocator type for the internal buffers.
 *  @tparam capability_ Whether to use OpenMP for @b multi-threading or some form of @b SIMD vectorization, or both.
 *  @tparam locality_ Whether to use the global alignment algorithm or the local one.
 *
 *  @note   The API of this algorithm is a bit weird, but it's designed to minimize the reliance on the definitions
 *          in the `stringzilla.hpp` header, making compilation times shorter for the end-user.
 *  @sa     For lower-level API, check `sz_levenshtein_distance[_utf8]` and `sz_needleman_wunsch_score`.
 *  @sa     For simplicity, use the `sz::levenshtein_distance[_utf8]` and `sz::needleman_wunsch_score`.
 *  @sa     For bulk API, use `sz::levenshtein_scores[_utf8]`.
 */
template <                                                     //
    sz_capability_t capability_ = sz_cap_serial_k,             //
    sz_alignment_direction_t direction_ = sz_align_maximize_k, //
    sz_alignment_locality_t locality_ = sz_align_global_k,     //
    typename char_type_ = char,                                //
    typename score_type_ = sz_size_t,                          //
    typename substituter_type_ = error_costs_uniform_t,        //
    typename allocator_type_ = dummy_alloc_t                   //
    >
struct diagonal_walker {

    using char_t = char_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_alignment_locality_t locality_k = locality_;
    static constexpr sz_alignment_direction_t direction_k = direction_;

    using allocated_t = typename allocator_t::value_type;
    static_assert(sizeof(allocated_t) == sizeof(char), "Allocator must be byte-aligned");
    using global_scorer_t =
        global_scorer<char_t const *, char_t const *, score_t, substituter_t, direction_k, capability_k>;
    using local_scorer_t =
        local_scorer<char_t const *, char_t const *, score_t, substituter_t, direction_k, capability_k>;
    using scorer_t = std::conditional_t<locality_k == sz_align_local_k, local_scorer_t, global_scorer_t>;

    substituter_t substituter_ {};
    error_cost_t gap_cost_ {1};
    allocator_t alloc_ {};

    diagonal_walker(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}

    /**
     *  @param[in] substituter A commutative function returning the cost of substituting one char with another.
     *  @param[in] gap_cost The uniform cost of a gap (insertion or deletion).
     *  @param[in] alloc A default-constructible allocator for the internal buffers.
     *
     */
    diagonal_walker(substituter_t substituter, error_cost_t gap_cost, allocator_t alloc) noexcept
        : substituter_(substituter), gap_cost_(gap_cost), alloc_(alloc) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score.
     */
    status_t operator()(span<char_t const> first, span<char_t const> second, score_t &result_ref) noexcept {

        // Make sure the size relation between the strings is correct.
        char_t const *shorter = first.data(), *longer = second.data();
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
        sz_size_t const buffer_length = sizeof(score_t) * max_diagonal_length * 3 + shorter_length * sizeof(char_t);
        score_t *const buffer = (score_t *)alloc_.allocate(buffer_length);
        if (!buffer) return status_t::bad_alloc_k;

        // The next few pointers will be swapped around.
        score_t *previous_scores = buffer;
        score_t *current_scores = previous_scores + max_diagonal_length;
        score_t *next_scores = current_scores + max_diagonal_length;
        char_t *const shorter_reversed = (char_t *)(next_scores + max_diagonal_length);

        // Export the reversed string into the buffer.
        for (sz_size_t i = 0; i != shorter_length; ++i) shorter_reversed[i] = shorter[shorter_length - 1 - i];

        // Initialize the first two diagonals:
        scorer_t diagonal_aligner {substituter_, gap_cost_};
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
            score_t *temporary = previous_scores;
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
            score_t *temporary = previous_scores;
            previous_scores = current_scores;
            current_scores = next_scores;
            next_scores = temporary;

            // ! Drop the first entry among the current scores.
            sz_move((sz_ptr_t)(previous_scores), (sz_ptr_t)(previous_scores + 1),
                    (max_diagonal_length - 1) * sizeof(score_t));
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
            score_t *temporary = previous_scores;

            // ! Drop the first entry among the current scores.
            // ! Assuming every next diagonal is shorter by one element, we don't need a full-blown `sz_move`.
            // ! to shift the array by one element.
            previous_scores = current_scores + 1;
            current_scores = next_scores;
            next_scores = temporary;
        }

        // Export the scalar before `free` call.
        result_ref = diagonal_aligner.score();
        alloc_.deallocate((allocated_t *)buffer, buffer_length);
        return status_t::success_k;
    }
};

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
 *  @tparam substituter_type_ A callable type that takes two characters and returns the substitution cost.
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
template <                                                     //
    sz_capability_t capability_ = sz_cap_serial_k,             //
    sz_alignment_direction_t direction_ = sz_align_maximize_k, //
    sz_alignment_locality_t locality_ = sz_align_global_k,     //
    typename char_type_ = char,                                //
    typename score_type_ = sz_size_t,                          //
    typename substituter_type_ = error_costs_uniform_t,        //
    typename allocator_type_ = dummy_alloc_t                   //
    >
struct horizontal_walker {

    using char_t = char_type_;
    using score_t = score_type_;
    using substituter_t = substituter_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_alignment_locality_t locality_k = locality_;
    static constexpr sz_alignment_direction_t direction_k = direction_;
    static_assert((capability_k & sz_cap_parallel_k) == 0, "This algorithm is not parallelized!");

    using allocated_t = typename allocator_t::value_type;
    static_assert(sizeof(allocated_t) == sizeof(char), "Allocator must be byte-aligned");
    using global_scorer_t =
        global_scorer<constant_iterator<char_t>, char_t const *, score_t, substituter_t, direction_k, capability_k>;
    using local_scorer_t =
        local_scorer<constant_iterator<char_t>, char_t const *, score_t, substituter_t, direction_k, capability_k>;
    using scorer_t = std::conditional_t<locality_k == sz_align_local_k, local_scorer_t, global_scorer_t>;

    substituter_t substituter_ {};
    error_cost_t gap_cost_ {1};
    allocator_t alloc_ {};

    horizontal_walker(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}

    /**
     *  @param[in] substituter A commutative function returning the cost of substituting one char with another.
     *  @param[in] gap_cost The uniform cost of a gap (insertion or deletion).
     *  @param[in] alloc A default-constructible allocator for the internal buffers.
     *
     */
    horizontal_walker(substituter_t substituter, error_cost_t gap_cost, allocator_t alloc) noexcept
        : substituter_(substituter), gap_cost_(gap_cost), alloc_(alloc) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score.
     */
    status_t operator()(span<char_t const> first, span<char_t const> second, score_t &result_ref) noexcept {

        // Make sure the size relation between the strings is correct.
        char_t const *shorter = first.data(), *longer = second.data();
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
        sz_size_t const buffer_length = sizeof(score_t) * shorter_dim * 2;
        score_t *const buffer = (score_t *)alloc_.allocate(buffer_length);
        if (!buffer) return status_t::bad_alloc_k;

        // The next few pointers will be swapped around.
        score_t *previous_scores = buffer;
        score_t *current_scores = previous_scores + shorter_dim;

        // Initialize the first row:
        scorer_t horizontal_aligner {substituter_, gap_cost_};
        for (sz_size_t col_idx = 0; col_idx < shorter_dim; ++col_idx)
            horizontal_aligner.init(previous_scores[col_idx], col_idx);

        // Progress through the matrix row-by-row:
        for (sz_size_t row_idx = 1; row_idx < longer_dim; ++row_idx) {

            // Don't forget to populate the first column of each row:
            horizontal_aligner.init(current_scores[0], 1);

            horizontal_aligner(                                  //
                constant_iterator<char_t> {longer[row_idx - 1]}, // first sequence of characters
                shorter,                                         // second sequence of characters
                shorter_dim - 1,     // number of elements to compute with the `horizontal_aligner`
                previous_scores,     // costs pre substitution
                previous_scores + 1, // costs pre insertion
                current_scores,      // costs pre deletion
                current_scores + 1);

            // Reuse the memory.
            std::swap(previous_scores, current_scores);
        }

        // Export the scalar before `free` call.
        result_ref = horizontal_aligner.score();
        alloc_.deallocate((allocated_t *)buffer, buffer_length);
        return status_t::success_k;
    }
};

/**
 *  @brief  Computes the @b byte-level Levenshtein distance between two strings using the OpenMP backend.
 *  @sa     `levenshtein_distance_utf8` for UTF-8 strings.
 */
template <                                         //
    sz_capability_t capability_ = sz_cap_serial_k, //
    typename char_type_ = char,                    //
    typename allocator_type_ = dummy_alloc_t       //
    >
struct levenshtein_distance {

    using char_t = char_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_serialized_k = (sz_capability_t)(capability_k & ~sz_cap_parallel_k);

    using horizontal_u8_t = horizontal_walker<capability_serialized_k, sz_align_minimize_k, sz_align_global_k, char_t,
                                              sz_u8_t, error_costs_uniform_t, allocator_t>;
    using diagonal_u8_t = diagonal_walker<capability_serialized_k, sz_align_minimize_k, sz_align_global_k, char_t,
                                          sz_u8_t, error_costs_uniform_t, allocator_t>;
    using diagonal_u16_t = diagonal_walker<capability_k, sz_align_minimize_k, sz_align_global_k, char_t, sz_u16_t,
                                           error_costs_uniform_t, allocator_t>;
    using diagonal_u32_t = diagonal_walker<capability_k, sz_align_minimize_k, sz_align_global_k, char_t, sz_u32_t,
                                           error_costs_uniform_t, allocator_t>;
    using diagonal_u64_t = diagonal_walker<capability_k, sz_align_minimize_k, sz_align_global_k, char_t, sz_u64_t,
                                           error_costs_uniform_t, allocator_t>;

    allocator_t alloc_ {};

    levenshtein_distance(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score. Pointer-sized for compatibility with C APIs.
     */
    status_t operator()(span<char_t const> first, span<char_t const> second, sz_size_t &result_ref) const noexcept {

        sz_size_t const first_length = first.length();
        sz_size_t const second_length = second.length();
        if (first_length == 0) {
            result_ref = second_length;
            return status_t::success_k;
        }
        if (second_length == 0) {
            result_ref = first_length;
            return status_t::success_k;
        }

        // Estimate the maximum dimension of the DP matrix
        sz_size_t const min_dim = sz_min_of_two(first_length, second_length) + 1;
        sz_size_t const max_dim = sz_max_of_two(first_length, second_length) + 1;

        // When dealing with very small inputs, we may want to use a simpler Wagner-Fischer algorithm.
        status_t status = status_t::success_k;
        if (min_dim < 16u) {
            sz_u8_t result_u8;
            status = horizontal_u8_t {error_costs_uniform_t {}, 1, alloc_}(first, second, result_u8);
            if (status == status_t::success_k) result_ref = result_u8;
        }

        // When dealing with larger arrays, we need to differentiate kernel with different cost aggregation types.
        // Smaller ones will overflow for larger inputs, but using larger-than-needed types will waste memory.
        if (max_dim < 256u) {
            sz_u8_t result_u8;
            status = diagonal_u8_t {error_costs_uniform_t {}, 1, alloc_}(first, second, result_u8);
            if (status == status_t::success_k) result_ref = result_u8;
        }
        else if (max_dim < 65536u) {
            sz_u16_t result_u16;
            status = diagonal_u16_t {error_costs_uniform_t {}, 1, alloc_}(first, second, result_u16);
            if (status == status_t::success_k) result_ref = result_u16;
        }
        else if (max_dim < 4294967296u) {
            sz_u32_t result_u32;
            status = diagonal_u32_t {error_costs_uniform_t {}, 1, alloc_}(first, second, result_u32);
            if (status == status_t::success_k) result_ref = result_u32;
        }
        else {
            sz_u64_t result_u64;
            status = diagonal_u64_t {error_costs_uniform_t {}, 1, alloc_}(first, second, result_u64);
            if (status == status_t::success_k) result_ref = result_u64;
        }

        return status;
    }
};

/**
 *  @brief  Computes the @b rune-level Levenshtein distance between two UTF-8 strings using the OpenMP backend.
 *  @sa     `levenshtein_distance` for binary strings.
 */
template <                                         //
    sz_capability_t capability_ = sz_cap_serial_k, //
    typename char_type_ = char,                    //
    typename allocator_type_ = dummy_alloc_t       //
    >
struct levenshtein_distance_utf8 {

    using char_t = char_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_serialized_k = (sz_capability_t)(capability_k & ~sz_cap_parallel_k);

    using horizontal_u8_t = horizontal_walker<capability_serialized_k, sz_align_minimize_k, sz_align_global_k,
                                              sz_rune_t, sz_u8_t, error_costs_uniform_t, allocator_t>;
    using diagonal_u8_t = diagonal_walker<capability_serialized_k, sz_align_minimize_k, sz_align_global_k, sz_rune_t,
                                          sz_u8_t, error_costs_uniform_t, allocator_t>;
    using diagonal_u16_t = diagonal_walker<capability_k, sz_align_minimize_k, sz_align_global_k, sz_rune_t, sz_u16_t,
                                           error_costs_uniform_t, allocator_t>;
    using diagonal_u32_t = diagonal_walker<capability_k, sz_align_minimize_k, sz_align_global_k, sz_rune_t, sz_u32_t,
                                           error_costs_uniform_t, allocator_t>;
    using diagonal_u64_t = diagonal_walker<capability_k, sz_align_minimize_k, sz_align_global_k, sz_rune_t, sz_u64_t,
                                           error_costs_uniform_t, allocator_t>;

    using ascii_fallback_t = levenshtein_distance_utf8<capability_k, char_t, allocator_t>;

    allocator_t alloc_ {};

    levenshtein_distance_utf8(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score. Pointer-sized for compatibility with C APIs.
     */
    status_t operator()(span<char_t const> first, span<char_t const> second, sz_size_t &result_ref) const noexcept {

        sz_size_t const first_length = first.length();
        sz_size_t const second_length = second.length();
        if (first_length == 0) {
            result_ref = second_length;
            return status_t::success_k;
        }
        if (second_length == 0) {
            result_ref = first_length;
            return status_t::success_k;
        }

        // Check if the strings are entirely composed of ASCII characters,
        // and default to a simpler algorithm in that case.
        if (sz_isascii(first.data(), first.length()) && sz_isascii(second.data(), second.length()))
            return ascii_fallback_t {alloc_}(first, second);

        // Allocate some memory to expand UTF-8 strings into UTF-32.
        sz_size_t const max_utf32_bytes = first.size() * 4 + second.size() * 4;
        sz_rune_t const *const first_data_utf32 = (sz_rune_t *)alloc_.allocate(max_utf32_bytes);
        sz_rune_t const *const second_data_utf32 = first_data_utf32 + first.size();

        // Export into UTF-32 buffer.
        sz_rune_length_t rune_length;
        sz_size_t first_length_utf32 = 0, second_length_utf32 = 0;
        for (sz_size_t progress_utf8 = 0, progress_utf32 = 0; progress_utf8 < first.size();
             progress_utf8 += rune_length, ++progress_utf32, ++first_length_utf32)
            sz_rune_parse(first.data() + progress_utf8, first_data_utf32 + progress_utf32, &rune_length);
        for (sz_size_t progress_utf8 = 0, progress_utf32 = 0; progress_utf8 < second.size();
             progress_utf8 += rune_length, ++progress_utf32, ++second_length_utf32)
            sz_rune_parse(second.data() + progress_utf8, second_data_utf32 + progress_utf32, &rune_length);

        // Estimate the maximum dimension of the DP matrix
        sz_size_t const min_dim = sz_min_of_two(first_length, second_length) + 1;
        sz_size_t const max_dim = sz_max_of_two(first_length, second_length) + 1;
        span<sz_rune_t const> const first_utf32 {first_data_utf32, first_length_utf32};
        span<sz_rune_t const> const second_utf32 {second_data_utf32, second_length_utf32};

        // When dealing with very small inputs, we may want to use a simpler Wagner-Fischer algorithm.
        status_t status = status_t::success_k;
        if (min_dim < 16u) {
            sz_u8_t result_u8;
            status = horizontal_u8_t {error_costs_uniform_t {}, 1, alloc_}(first_utf32, second_utf32, result_u8);
            if (status == status_t::success_k) result_ref = result_u8;
        }

        // When dealing with larger arrays, we need to differentiate kernel with different cost aggregation types.
        // Smaller ones will overflow for larger inputs, but using larger-than-needed types will waste memory.
        if (max_dim < 256u) {
            sz_u8_t result_u8;
            status = diagonal_u8_t {error_costs_uniform_t {}, 1, alloc_}(first_utf32, second_utf32, result_u8);
            if (status == status_t::success_k) result_ref = result_u8;
        }
        else if (max_dim < 65536u) {
            sz_u16_t result_u16;
            status = diagonal_u16_t {error_costs_uniform_t {}, 1, alloc_}(first_utf32, second_utf32, result_u16);
            if (status == status_t::success_k) result_ref = result_u16;
        }
        else if (max_dim < 4294967296u) {
            sz_u32_t result_u32;
            status = diagonal_u32_t {error_costs_uniform_t {}, 1, alloc_}(first_utf32, second_utf32, result_u32);
            if (status == status_t::success_k) result_ref = result_u32;
        }
        else {
            sz_u64_t result_u64;
            status = diagonal_u64_t {error_costs_uniform_t {}, 1, alloc_}(first_utf32, second_utf32, result_u64);
            if (status == status_t::success_k) result_ref = result_u64;
        }

        return status;
    }
};

/**
 *  @brief  Computes the @b byte-level Needleman-Wunsch score between two strings using the OpenMP backend.
 *  @sa     `levenshtein_distance` for uniform substitution and gap costs.
 */
template <                                                     //
    sz_capability_t capability_ = sz_cap_serial_k,             //
    typename char_type_ = char,                                //
    typename substituter_type_ = error_costs_256x256_lookup_t, //
    typename allocator_type_ = dummy_alloc_t                   //
    >
struct needleman_wunsch_score {

    using char_t = char_type_;
    using substituter_t = substituter_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_serialized_k = (sz_capability_t)(capability_k & ~sz_cap_parallel_k);

    using horizontal_i16_t = horizontal_walker<capability_serialized_k, sz_align_maximize_k, sz_align_global_k, char_t,
                                               sz_i16_t, substituter_t, allocator_t>;
    using diagonal_i16_t = diagonal_walker<capability_serialized_k, sz_align_maximize_k, sz_align_global_k, char_t,
                                           sz_i16_t, substituter_t, allocator_t>;
    using diagonal_i32_t = diagonal_walker<capability_k, sz_align_maximize_k, sz_align_global_k, char_t, sz_i32_t,
                                           substituter_t, allocator_t>;
    using diagonal_i64_t = diagonal_walker<capability_k, sz_align_maximize_k, sz_align_global_k, char_t, sz_i64_t,
                                           substituter_t, allocator_t>;

    substituter_t substituter_ {};
    error_cost_t gap_cost_ {1};
    allocator_t alloc_ {};

    needleman_wunsch_score(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    needleman_wunsch_score(substituter_t subs, error_cost_t gap_cost, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_cost_(gap_cost), alloc_(alloc) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score. Pointer-sized for compatibility with C APIs.
     */
    status_t operator()(span<char_t const> first, span<char_t const> second, sz_ssize_t &result_ref) const noexcept {

        sz_size_t const first_length = first.length();
        sz_size_t const second_length = second.length();
        if (first_length == 0) {
            result_ref = second_length * gap_cost_;
            return status_t::success_k;
        }
        if (second_length == 0) {
            result_ref = first_length * gap_cost_;
            return status_t::success_k;
        }

        // Estimate the maximum dimension of the DP matrix
        sz_size_t const min_dim = sz_min_of_two(first_length, second_length) + 1;
        sz_size_t const max_dim = sz_max_of_two(first_length, second_length) + 1;

        // When dealing with very small inputs, we may want to use a simpler Wagner-Fischer algorithm.
        status_t status = status_t::success_k;
        if (min_dim < 16u) {
            sz_i16_t result_i16;
            status = horizontal_i16_t {substituter_, gap_cost_, alloc_}(first, second, result_i16);
            if (status == status_t::success_k) result_ref = result_i16;
        }

        // When dealing with larger arrays, we need to differentiate kernel with different cost aggregation types.
        // Smaller ones will overflow for larger inputs, but using larger-than-needed types will waste memory.
        // Assuming each individual cost falls in [-128, 127], the `i16` range of [-32768, 32767] is sufficient
        // for inputs under (32768 / 128) = 256 characters.
        if (max_dim < 256u) {
            sz_i16_t result_i16;
            status = diagonal_i16_t {substituter_, gap_cost_, alloc_}(first, second, result_i16);
            if (status == status_t::success_k) result_ref = result_i16;
        }
        // Assuming each individual cost falls in [-128, 127], the `i32` range of [-2147483648, 2147483647] is
        // sufficient for inputs under (2147483648 / 128) = 16777216 characters.
        else if (max_dim < 16777216u) {
            sz_i32_t result_i32;
            status = diagonal_i32_t {substituter_, gap_cost_, alloc_}(first, second, result_i32);
            if (status == status_t::success_k) result_ref = result_i32;
        }
        else {
            sz_i64_t result_i64;
            status = diagonal_i64_t {substituter_, gap_cost_, alloc_}(first, second, result_i64);
            if (status == status_t::success_k) result_ref = result_i64;
        }

        return status;
    }
};

/**
 *  @brief  Computes the @b byte-level Needleman-Wunsch score between two strings using the OpenMP backend.
 *  @sa     `levenshtein_distance` for uniform substitution and gap costs.
 */
template <                                                     //
    sz_capability_t capability_ = sz_cap_serial_k,             //
    typename char_type_ = char,                                //
    typename substituter_type_ = error_costs_256x256_lookup_t, //
    typename allocator_type_ = dummy_alloc_t                   //
    >
struct smith_waterman_score {

    using char_t = char_type_;
    using substituter_t = substituter_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_serialized_k = (sz_capability_t)(capability_k & ~sz_cap_parallel_k);

    using horizontal_i16_t = horizontal_walker<capability_serialized_k, sz_align_maximize_k, sz_align_local_k, char_t,
                                               sz_i16_t, substituter_t, allocator_t>;
    using diagonal_i16_t = diagonal_walker<capability_serialized_k, sz_align_maximize_k, sz_align_local_k, char_t,
                                           sz_i16_t, substituter_t, allocator_t>;
    using diagonal_i32_t = diagonal_walker<capability_k, sz_align_maximize_k, sz_align_local_k, char_t, sz_i32_t,
                                           substituter_t, allocator_t>;
    using diagonal_i64_t = diagonal_walker<capability_k, sz_align_maximize_k, sz_align_local_k, char_t, sz_i64_t,
                                           substituter_t, allocator_t>;

    substituter_t substituter_ {};
    error_cost_t gap_cost_ {1};
    allocator_t alloc_ {};

    smith_waterman_score(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    smith_waterman_score(substituter_t subs, error_cost_t gap_cost, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_cost_(gap_cost), alloc_(alloc) {}

    /**
     *  @param[in] first The first string.
     *  @param[in] second The second string.
     *  @param[out] result_ref Location to dump the calculated score. Pointer-sized for compatibility with C APIs.
     */
    status_t operator()(span<char_t const> first, span<char_t const> second, sz_ssize_t &result_ref) const noexcept {

        sz_size_t const first_length = first.length();
        sz_size_t const second_length = second.length();
        if (first_length == 0 || second_length == 0) {
            result_ref = 0;
            return status_t::success_k;
        }

        // Estimate the maximum dimension of the DP matrix
        sz_size_t const min_dim = sz_min_of_two(first_length, second_length) + 1;
        sz_size_t const max_dim = sz_max_of_two(first_length, second_length) + 1;

        // When dealing with very small inputs, we may want to use a simpler Wagner-Fischer algorithm.
        status_t status = status_t::success_k;
        if (min_dim < 16u) {
            sz_i16_t result_i16;
            status = horizontal_i16_t {substituter_, gap_cost_, alloc_}(first, second, result_i16);
            if (status == status_t::success_k) result_ref = result_i16;
        }

        // When dealing with larger arrays, we need to differentiate kernel with different cost aggregation types.
        // Smaller ones will overflow for larger inputs, but using larger-than-needed types will waste memory.
        // Assuming each individual cost falls in [-128, 127], the `i16` range of [-32768, 32767] is sufficient
        // for inputs under (32768 / 128) = 256 characters.
        if (max_dim < 256u) {
            sz_i16_t result_i16;
            status = diagonal_i16_t {substituter_, gap_cost_, alloc_}(first, second, result_i16);
            if (status == status_t::success_k) result_ref = result_i16;
        }
        // Assuming each individual cost falls in [-128, 127], the `i32` range of [-2147483648, 2147483647] is
        // sufficient for inputs under (2147483648 / 128) = 16777216 characters.
        else if (max_dim < 16777216u) {
            sz_i32_t result_i32;
            status = diagonal_i32_t {substituter_, gap_cost_, alloc_}(first, second, result_i32);
            if (status == status_t::success_k) result_ref = result_i32;
        }
        else {
            sz_i64_t result_i64;
            status = diagonal_i64_t {substituter_, gap_cost_, alloc_}(first, second, result_i64);
            if (status == status_t::success_k) result_ref = result_i64;
        }

        return status;
    }
};

/**
 *  @brief  Helper method, applying the desired pairwise scoring kernel to all input pairs,
 *          differentiating multi-threaded and single-threaded cases.
 *          For pairs of very large strings, all cores cooperate to compute one distance maximizing
 *          cache hits. For smaller strings, each core computes its own distance.
 */
template <                                                                              //
    typename score_type_,                                                               //
    typename inter_pair_parallel_type_,                                                 //
    typename intra_pair_parallel_type_,                                                 //
    typename first_strings_type_, typename second_strings_type_, typename results_type_ //
    >
status_t _score_in_parallel(                         //
    inter_pair_parallel_type_ &&intra_pair_parallel, //
    intra_pair_parallel_type_ &&inter_pair_parallel, //
    first_strings_type_ const &first_strings, second_strings_type_ const &second_strings, results_type_ &&results,
    cpu_specs_t specs = {}) noexcept {

    using score_t = score_type_;

    auto first_size = first_strings.size();
    auto second_size = second_strings.size();
    _sz_assert(first_size == second_size && "Expect equal number of strings");

    // Separately handle shorter and longer strings.
    constexpr sz_size_t threshold_size = 256;

    // Use an atomic to store any error encountered.
    std::atomic<status_t> error {status_t::success_k};

    // ? There may be a huge variance in the lengths of the strings,
    // ? so we need to use a dynamic schedule.
#pragma omp parallel for schedule(dynamic, 1)
    for (sz_size_t i = 0; i < first_size; ++i) {
        if (error.load() != status_t::success_k) continue;
        score_t result = 0;
        auto const &first = first_strings[i];
        auto const &second = second_strings[i];
        auto largest_dimension = sz_max_of_two(first.length(), second.length());
        if (largest_dimension >= threshold_size) continue;
        status_t status = inter_pair_parallel({first.data(), first.length()}, {second.data(), second.length()}, result);
        if (status == status_t::success_k) { results[i] = result; }
        else { error.store(status); }
    }

    // Now handle the larger strings.
    for (sz_size_t i = 0; i < first_size && error.load() != status_t::success_k; ++i) {
        score_t result = 0;
        auto const &first = first_strings[i];
        auto const &second = second_strings[i];
        auto largest_dimension = sz_max_of_two(first.length(), second.length());
        if (largest_dimension < threshold_size) continue;
        status_t status = intra_pair_parallel({first.data(), first.length()}, {second.data(), second.length()}, result);
        if (status == status_t::success_k) { results[i] = result; }
        else { error.store(status); }
    }
    return error.load();
}

template <                                                                              //
    typename score_type_,                                                               //
    typename scoring_type_,                                                             //
    typename first_strings_type_, typename second_strings_type_, typename results_type_ //
    >
status_t _score_sequentially(scoring_type_ &&scoring, first_strings_type_ const &first_strings,
                             second_strings_type_ const &second_strings, results_type_ &&results) noexcept {
    using scoring_t = scoring_type_;
    using score_t = score_type_;

    auto first_size = first_strings.size();
    auto second_size = second_strings.size();
    _sz_assert(first_size == second_size && "Expect equal number of strings");

    for (sz_size_t i = 0; i < first_size; ++i) {
        score_t result = 0;
        auto const &first = first_strings[i];
        auto const &second = second_strings[i];
        status_t status = scoring({first.data(), first.length()}, {second.data(), second.length()}, result);
        if (status == status_t::success_k) { results[i] = result; }
        else { return status; }
    }
    return status_t::success_k;
}

/**
 *  @brief  Computes one or many pairwise Levenshtein distances in parallel using the OpenMP backend.
 *          For pairs of very large strings, all cores cooperate to compute one distance maximizing
 *          cache hits. For smaller strings, each core computes its own distance.
 */
template <                                         //
    sz_capability_t capability_ = sz_cap_serial_k, //
    typename char_type_ = char,                    //
    typename allocator_type_ = dummy_alloc_t       //
    >
struct levenshtein_distances {

    using char_t = char_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_serialized_k = (sz_capability_t)(capability_k & ~sz_cap_parallel_k);

    static constexpr bool is_parallel() { return (capability_k & sz_cap_parallel_k) != 0; }

    using intra_pair_parallel_t = levenshtein_distance<capability_serialized_k, char_t, allocator_t>;
    using inter_pair_parallel_t = levenshtein_distance<capability_k, char_t, allocator_t>;

    allocator_t alloc_ {};

    levenshtein_distances(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results) const noexcept {

        if constexpr (is_parallel())
            return _score_in_parallel<sz_size_t>(inter_pair_parallel_t {alloc_}, intra_pair_parallel_t {alloc_},
                                                 first_strings, second_strings, std::forward<results_type_>(results));
        else
            return _score_sequentially<sz_size_t>(intra_pair_parallel_t {alloc_}, first_strings, second_strings,
                                                  std::forward<results_type_>(results));
    }
};

template <                                         //
    sz_capability_t capability_ = sz_cap_serial_k, //
    typename char_type_ = char,                    //
    typename allocator_type_ = dummy_alloc_t       //
    >
struct levenshtein_distances_utf8 {

    using char_t = char_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_serialized_k = (sz_capability_t)(capability_k & ~sz_cap_parallel_k);

    static constexpr bool is_parallel() { return (capability_k & sz_cap_parallel_k) != 0; }

    using intra_pair_parallel_t = levenshtein_distance_utf8<capability_serialized_k, char_t, allocator_t>;
    using inter_pair_parallel_t = levenshtein_distance_utf8<capability_k, char_t, allocator_t>;

    allocator_t alloc_ {};

    levenshtein_distances_utf8(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results) const noexcept {

        if constexpr (is_parallel())
            return _score_in_parallel<sz_size_t>(inter_pair_parallel_t {alloc_}, intra_pair_parallel_t {alloc_},
                                                 first_strings, second_strings, std::forward<results_type_>(results));
        else
            return _score_sequentially<sz_size_t>(intra_pair_parallel_t {alloc_}, first_strings, second_strings,
                                                  std::forward<results_type_>(results));
    }
};

template <                                                     //
    sz_capability_t capability_ = sz_cap_serial_k,             //
    typename char_type_ = char,                                //
    typename substituter_type_ = error_costs_256x256_lookup_t, //
    typename allocator_type_ = dummy_alloc_t                   //
    >
struct needleman_wunsch_scores {

    using char_t = char_type_;
    using substituter_t = substituter_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_serialized_k = (sz_capability_t)(capability_k & ~sz_cap_parallel_k);

    static constexpr bool is_parallel() { return (capability_k & sz_cap_parallel_k) != 0; }

    using intra_pair_parallel_t = needleman_wunsch_score<capability_serialized_k, char_t, substituter_t, allocator_t>;
    using inter_pair_parallel_t = needleman_wunsch_score<capability_k, char_t, substituter_t, allocator_t>;

    substituter_t substituter_ {};
    error_cost_t gap_cost_ {1};
    allocator_t alloc_ {};

    needleman_wunsch_scores(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    needleman_wunsch_scores(substituter_t subs, error_cost_t gap_cost, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_cost_(gap_cost), alloc_(alloc) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results) const noexcept {

        if constexpr (is_parallel())
            return _score_in_parallel<sz_ssize_t>(inter_pair_parallel_t {substituter_, gap_cost_, alloc_},
                                                  intra_pair_parallel_t {substituter_, gap_cost_, alloc_},
                                                  first_strings, second_strings, std::forward<results_type_>(results));
        else
            return _score_sequentially<sz_ssize_t>(intra_pair_parallel_t {substituter_, gap_cost_, alloc_},
                                                   first_strings, second_strings, std::forward<results_type_>(results));
    }
};

template <                                                     //
    sz_capability_t capability_ = sz_cap_serial_k,             //
    typename char_type_ = char,                                //
    typename substituter_type_ = error_costs_256x256_lookup_t, //
    typename allocator_type_ = dummy_alloc_t                   //
    >
struct smith_waterman_scores {

    using char_t = char_type_;
    using substituter_t = substituter_type_;
    using allocator_t = allocator_type_;

    static constexpr sz_capability_t capability_k = capability_;
    static constexpr sz_capability_t capability_serialized_k = (sz_capability_t)(capability_k & ~sz_cap_parallel_k);

    static constexpr bool is_parallel() { return (capability_k & sz_cap_parallel_k) != 0; }

    using intra_pair_parallel_t = smith_waterman_score<capability_serialized_k, char_t, substituter_t, allocator_t>;
    using inter_pair_parallel_t = smith_waterman_score<capability_k, char_t, substituter_t, allocator_t>;

    substituter_t substituter_ {};
    error_cost_t gap_cost_ {1};
    allocator_t alloc_ {};

    smith_waterman_scores(allocator_t alloc = allocator_t {}) noexcept : alloc_(alloc) {}
    smith_waterman_scores(substituter_t subs, error_cost_t gap_cost, allocator_t alloc = allocator_t {}) noexcept
        : substituter_(subs), gap_cost_(gap_cost), alloc_(alloc) {}

    template <typename first_strings_type_, typename second_strings_type_, typename results_type_>
    status_t operator()(first_strings_type_ const &first_strings, second_strings_type_ const &second_strings,
                        results_type_ &&results) const noexcept {

        if constexpr (is_parallel())
            return _score_in_parallel<sz_ssize_t>(inter_pair_parallel_t {substituter_, gap_cost_, alloc_},
                                                  intra_pair_parallel_t {substituter_, gap_cost_, alloc_},
                                                  first_strings, second_strings, std::forward<results_type_>(results));
        else
            return _score_sequentially<sz_ssize_t>(intra_pair_parallel_t {substituter_, gap_cost_, alloc_},
                                                   first_strings, second_strings, std::forward<results_type_>(results));
    }
};

} // namespace openmp
} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_SIMILARITY_HPP_