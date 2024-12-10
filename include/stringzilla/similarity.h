/**
 *  @brief  Hardware-accelerated string similarity utilities.
 *  @file   similarity.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs:
 *
 *  - `sz_edit_distance` & `sz_edit_distance_utf8` for Levenshtein edit-distance computation.
 *  - `sz_alignment_score` for weighted Needleman-Wunsch global alignment.
 *  - `sz_hamming_distance` & `sz_hamming_distance_utf8` for Hamming distance computation.
 *
 *  The Hamming distance is rarely used in string processing, so only minimal compatibility is provided.
 *  The Levenshtein distance, however, is much more popular and computationally intensive.
 *  So a huge part of this file is focused on optimizing it for different input alphabet sizes and input lengths.
 */
#ifndef STRINGZILLA_SIMILARITY_H_
#define STRINGZILLA_SIMILARITY_H_

#include "find.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief  Computes the Hamming distance between two strings - number of not matching characters.
 *          Difference in length is is counted as a mismatch.
 *
 *  @param a        First string to compare.
 *  @param a_length Number of bytes in the first string.
 *  @param b        Second string to compare.
 *  @param b_length Number of bytes in the second string.
 *
 *  @param bound    Exclusive upper bound on the distance, that allows us to exit early.
 *                  Pass `SZ_SIZE_MAX` or any value greater than `(max(a_length, b_length))` to ignore.
 *                  Pass zero to check if the strings are equal.
 *  @return         Returns an unsigned integer for the edit distance. Zero means the strings are equal.
 *                  Returns the `(max(a_length, b_length)) + 1` if the distance limit was reached.
 *
 *  @see    sz_hamming_distance_utf8
 *  @see    https://en.wikipedia.org/wiki/Hamming_distance
 */
SZ_DYNAMIC sz_size_t sz_hamming_distance( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, sz_size_t bound);

/**
 *  @brief  Computes the Hamming distance between two @b UTF8 strings - number of not matching characters.
 *          Difference in length is is counted as a mismatch.
 *
 *  @param a        First string to compare.
 *  @param a_length Number of bytes in the first string.
 *  @param b        Second string to compare.
 *  @param b_length Number of bytes in the second string.
 *
 *  @param bound    Exclusive upper bound on the distance, that allows us to exit early.
 *                  Pass `SZ_SIZE_MAX` or any value greater than `(max(a_length, b_length))` to ignore.
 *                  Pass zero to check if the strings are equal.
 *  @return         Returns an unsigned integer for the edit distance. Zero means the strings are equal.
 *                  Returns the `(max(a_length, b_length)) + 1` if the distance limit was reached.
 *
 *  @see    sz_hamming_distance
 *  @see    https://en.wikipedia.org/wiki/Hamming_distance
 */
SZ_DYNAMIC sz_size_t sz_hamming_distance_utf8( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, sz_size_t bound);

/**
 *  @brief  Computes the Levenshtein edit-distance between two strings using the Wagner-Fisher algorithm.
 *          Similar to the Needleman-Wunsch alignment algorithm. Often used in fuzzy string matching.
 *
 *  @param a        First string to compare.
 *  @param a_length Number of bytes in the first string.
 *  @param b        Second string to compare.
 *  @param b_length Number of bytes in the second string.
 *
 *  @param alloc    Temporary memory allocator. Only some of the rows of the matrix will be allocated,
 *                  so the memory usage is linear in relation to ::a_length and ::b_length.
 *                  If SZ_NULL is passed, will initialize to the systems default `malloc`.
 *
 *  @param bound    Exclusive upper bound on the distance, that allows us to exit early.
 *                  Pass `SZ_SIZE_MAX` or any value greater than `(max(a_length, b_length))` to ignore.
 *                  Pass zero to check if the strings are equal.
 *  @return         Returns an unsigned integer for the edit distance. Zero means the strings are equal.
 *                  Returns the `(max(a_length, b_length)) + 1` if the distance limit was reached.
 *                  Returns `SZ_SIZE_MAX` if the memory allocation failed.
 *
 *  @see    sz_memory_allocator_init_fixed, sz_memory_allocator_init_default
 *  @see    https://en.wikipedia.org/wiki/Levenshtein_distance
 */
SZ_DYNAMIC sz_size_t sz_edit_distance(                                //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, //
    sz_size_t bound, sz_memory_allocator_t *alloc);

/**
 *  @brief  Computes the Levenshtein edit-distance between two @b UTF8 strings.
 *          Unlike `sz_edit_distance`, reports the distance in Unicode codepoints, and not in bytes.
 *
 *  @param a        First string to compare.
 *  @param a_length Number of bytes in the first string.
 *  @param b        Second string to compare.
 *  @param b_length Number of bytes in the second string.
 *
 *  @param alloc    Temporary memory allocator. Only some of the rows of the matrix will be allocated,
 *                  so the memory usage is linear in relation to ::a_length and ::b_length.
 *                  If SZ_NULL is passed, will initialize to the systems default `malloc`.
 *
 *  @param bound    Exclusive upper bound on the distance, that allows us to exit early.
 *                  Pass `SZ_SIZE_MAX` or any value greater than `(max(a_length, b_length))` to ignore.
 *                  Pass zero to check if the strings are equal.
 *  @return         Returns an unsigned integer for the edit distance. Zero means the strings are equal.
 *                  Returns the `(max(a_length, b_length)) + 1` if the distance limit was reached.
 *                  Returns `SZ_SIZE_MAX` if the memory allocation failed.
 *
 *  @see    sz_memory_allocator_init_fixed, sz_memory_allocator_init_default, sz_edit_distance
 *  @see    https://en.wikipedia.org/wiki/Levenshtein_distance
 */
SZ_DYNAMIC sz_size_t sz_edit_distance_utf8(                           //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, //
    sz_size_t bound, sz_memory_allocator_t *alloc);

/**
 *  @brief  Computes Needleman–Wunsch alignment score for two string. Often used in bioinformatics and cheminformatics.
 *          Similar to the Levenshtein edit-distance, parameterized for gap and substitution penalties.
 *
 *  Not commutative in the general case, as the order of the strings matters, as `sz_alignment_score(a, b)` may
 *  not be equal to `sz_alignment_score(b, a)`. Becomes @b commutative, if the substitution costs are symmetric.
 *  Equivalent to the negative Levenshtein distance, if: `gap == -1` and `subs[i][j] == (i == j ? 0: -1)`.
 *
 *  @param a        First string to compare.
 *  @param a_length Number of bytes in the first string.
 *  @param b        Second string to compare.
 *  @param b_length Number of bytes in the second string.
 *  @param gap      Penalty cost for gaps - insertions and removals.
 *  @param subs     Substitution costs matrix with 256 x 256 values for all pairs of characters.
 *
 *  @param alloc    Temporary memory allocator. Only some of the rows of the matrix will be allocated,
 *                  so the memory usage is linear in relation to ::a_length and ::b_length.
 *                  If SZ_NULL is passed, will initialize to the systems default `malloc`.
 *
 *  @return         Signed similarity score. Can be negative, depending on the substitution costs.
 *                  Returns `SZ_SSIZE_MAX` if the memory allocation failed.
 *
 *  @see    sz_memory_allocator_init_fixed, sz_memory_allocator_init_default
 *  @see    https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm
 */
SZ_DYNAMIC sz_ssize_t sz_alignment_score(                             //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, //
    sz_error_cost_t const *subs, sz_error_cost_t gap,                 //
    sz_memory_allocator_t *alloc);

/**
 *  @brief  Checks if all characters in the range are valid ASCII characters.
 *
 *  @param text     String to be analyzed.
 *  @param length   Number of bytes in the string.
 *  @return         Whether all characters are valid ASCII characters.
 */
SZ_PUBLIC sz_bool_t sz_isascii(sz_cptr_t text, sz_size_t length);

/** @copydoc sz_hamming_distance */
SZ_PUBLIC sz_size_t sz_hamming_distance_serial( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, sz_size_t bound);

/** @copydoc sz_hamming_distance_utf8 */
SZ_PUBLIC sz_size_t sz_hamming_distance_utf8_serial( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, sz_size_t bound);

/** @copydoc sz_edit_distance */
SZ_PUBLIC sz_size_t sz_edit_distance_serial(                          //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, //
    sz_size_t bound, sz_memory_allocator_t *alloc);

/** @copydoc sz_edit_distance_utf8 */
SZ_PUBLIC sz_size_t sz_edit_distance_utf8_serial(                     //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, //
    sz_size_t bound, sz_memory_allocator_t *alloc);

/** @copydoc sz_alignment_score */
SZ_PUBLIC sz_ssize_t sz_alignment_score_serial(                       //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, //
    sz_error_cost_t const *subs, sz_error_cost_t gap,                 //
    sz_memory_allocator_t *alloc);

#if SZ_USE_ICE

SZ_INTERNAL sz_size_t sz_edit_distance_ice(      //
    sz_cptr_t shorter, sz_size_t shorter_length, //
    sz_cptr_t longer, sz_size_t longer_length,   //
    sz_size_t bound, sz_memory_allocator_t *alloc);

SZ_INTERNAL sz_ssize_t sz_alignment_score_ice(   //
    sz_cptr_t shorter, sz_size_t shorter_length, //
    sz_cptr_t longer, sz_size_t longer_length,   //
    sz_error_cost_t const *subs, sz_error_cost_t gap, sz_memory_allocator_t *alloc);

#endif

#pragma endregion // Core API

#pragma region Serial Implementation

SZ_INTERNAL sz_size_t _sz_edit_distance_skewed_diagonals_serial( //
    sz_cptr_t shorter, sz_size_t shorter_length,                 //
    sz_cptr_t longer, sz_size_t longer_length,                   //
    sz_size_t bound, sz_memory_allocator_t *alloc) {

    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    // TODO: Generalize to remove the following asserts!
    _sz_assert(!bound && "For bounded search the method should only evaluate one band of the matrix.");
    _sz_assert(shorter_length == longer_length && "The method hasn't been generalized to different length inputs yet.");
    sz_unused(longer_length && bound);

    // We are going to store 3 diagonals of the matrix.
    // The length of the longest (main) diagonal would be `n = (shorter_length + 1)`.
    sz_size_t n = shorter_length + 1;
    sz_size_t buffer_length = sizeof(sz_size_t) * n * 3;
    sz_size_t *distances = (sz_size_t *)alloc->allocate(buffer_length, alloc->handle);
    if (!distances) return SZ_SIZE_MAX;

    sz_size_t *previous_distances = distances;
    sz_size_t *current_distances = previous_distances + n;
    sz_size_t *next_distances = previous_distances + n * 2;

    // Initialize the first two diagonals:
    previous_distances[0] = 0;
    current_distances[0] = current_distances[1] = 1;

    // Progress through the upper triangle of the Levenshtein matrix.
    sz_size_t next_diagonal_index = 2;
    for (; next_diagonal_index != n; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = next_diagonal_index + 1;
        for (sz_size_t i = 0; i + 2 < next_diagonal_length; ++i) {
            sz_size_t cost_of_substitution = shorter[next_diagonal_index - i - 2] != longer[i];
            sz_size_t cost_if_substitution = previous_distances[i] + cost_of_substitution;
            sz_size_t cost_if_deletion_or_insertion = sz_min_of_two(current_distances[i], current_distances[i + 1]) + 1;
            next_distances[i + 1] = sz_min_of_two(cost_if_deletion_or_insertion, cost_if_substitution);
        }
        // Don't forget to populate the first row and the first column of the Levenshtein matrix.
        next_distances[0] = next_distances[next_diagonal_length - 1] = next_diagonal_index;
        // Perform a circular rotation of those buffers, to reuse the memory.
        sz_size_t *temporary = previous_distances;
        previous_distances = current_distances;
        current_distances = next_distances;
        next_distances = temporary;
    }

    // By now we've scanned through the upper triangle of the matrix, where each subsequent iteration results in a
    // larger diagonal. From now onwards, we will be shrinking. Instead of adding value equal to the skewed diagonal
    // index on either side, we will be cropping those values out.
    sz_size_t diagonals_count = n + n - 1;
    for (; next_diagonal_index != diagonals_count; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
        for (sz_size_t i = 0; i != next_diagonal_length; ++i) {
            sz_size_t cost_of_substitution = shorter[shorter_length - 1 - i] != longer[next_diagonal_index - n + i];
            sz_size_t cost_if_substitution = previous_distances[i] + cost_of_substitution;
            sz_size_t cost_if_deletion_or_insertion = sz_min_of_two(current_distances[i], current_distances[i + 1]) + 1;
            next_distances[i] = sz_min_of_two(cost_if_deletion_or_insertion, cost_if_substitution);
        }
        // Perform a circular rotation of those buffers, to reuse the memory, this time, with a shift,
        // dropping the first element in the current array.
        sz_size_t *temporary = previous_distances;
        previous_distances = current_distances + 1;
        current_distances = next_distances;
        next_distances = temporary;
    }

    // Cache scalar before `free` call.
    sz_size_t result = current_distances[0];
    alloc->free(distances, buffer_length, alloc->handle);
    return result;
}

/**
 *  @brief  Compute the Levenshtein distance between two strings using the Wagner-Fisher algorithm.
 *          Stores only 2 rows of the Levenshtein matrix, but uses 64-bit integers for the distance values,
 *          and upcasts UTF8 variable-length codepoints to 64-bit integers for faster addressing.
 *
 *  ! In the worst case for 2 strings of length 100, that contain just one 16-bit codepoint this will result in
 * extra:
 *      + 2 rows * 100 slots * 8 bytes/slot = 1600 bytes of memory for the two rows of the Levenshtein matrix rows.
 *      + 100 codepoints * 2 strings * 4 bytes/codepoint = 800 bytes of memory for the UTF8 buffer.
 *      = 2400 bytes of memory or @b 12x memory amplification!
 */
SZ_INTERNAL sz_size_t _sz_edit_distance_wagner_fisher_serial( //
    sz_cptr_t longer, sz_size_t longer_length,                //
    sz_cptr_t shorter, sz_size_t shorter_length,              //
    sz_size_t bound, sz_bool_t can_be_unicode, sz_memory_allocator_t *alloc) {

    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    // A good idea may be to dispatch different kernels for different string lengths.
    // Like using `uint8_t` counters for strings under 255 characters long.
    // Good in theory, this results in frequent upcasts and downcasts in serial code.
    // On strings over 20 bytes, using `uint8` over `uint64` on 64-bit x86 CPU doubles the execution time.
    // So one must be very cautious with such optimizations.
    typedef sz_size_t _distance_t;

    // Compute the number of columns in our Levenshtein matrix.
    sz_size_t const n = shorter_length + 1;

    // If a buffering memory-allocator is provided, this operation is practically free,
    // and cheaper than allocating even 512 bytes (for small distance matrices) on stack.
    sz_size_t buffer_length = sizeof(_distance_t) * (n * 2);

    // If the strings contain Unicode characters, let's estimate the max character width,
    // and use it to allocate a larger buffer to decode UTF8.
    sz_charset_t ascii_charset;
    sz_charset_init_ascii(&ascii_charset);
    sz_charset_invert(&ascii_charset);
    int const longer_is_ascii = sz_find_charset_serial(longer, longer_length, &ascii_charset) == SZ_NULL_CHAR;
    int const shorter_is_ascii = sz_find_charset_serial(shorter, shorter_length, &ascii_charset) == SZ_NULL_CHAR;
    int const will_convert_to_unicode = can_be_unicode == sz_true_k && (!longer_is_ascii || !shorter_is_ascii);
    if (will_convert_to_unicode) { buffer_length += (shorter_length + longer_length) * sizeof(sz_rune_t); }
    else { can_be_unicode = sz_false_k; }

    // If the allocation fails, return the maximum distance.
    sz_ptr_t const buffer = (sz_ptr_t)alloc->allocate(buffer_length, alloc->handle);
    if (!buffer) return SZ_SIZE_MAX;

    // Let's export the UTF8 sequence into the newly allocated buffer at the end.
    if (can_be_unicode == sz_true_k) {
        sz_rune_t *const longer_utf32 = (sz_rune_t *)(buffer + sizeof(_distance_t) * (n * 2));
        sz_rune_t *const shorter_utf32 = longer_utf32 + longer_length;
        // Export the UTF8 sequences into the newly allocated buffer.
        longer_length = _sz_export_utf8_to_utf32(longer, longer_length, longer_utf32);
        shorter_length = _sz_export_utf8_to_utf32(shorter, shorter_length, shorter_utf32);
        longer = (sz_cptr_t)longer_utf32;
        shorter = (sz_cptr_t)shorter_utf32;
    }

    // Let's parameterize the core logic for different character types and distance types.
#define _wagner_fisher_unbounded(_distance_t, _char_t)                                                                \
    /* Now let's cast our pointer to avoid it in subsequent sections. */                                              \
    _char_t const *const longer_chars = (_char_t const *)longer;                                                      \
    _char_t const *const shorter_chars = (_char_t const *)shorter;                                                    \
    _distance_t *previous_distances = (_distance_t *)buffer;                                                          \
    _distance_t *current_distances = previous_distances + n;                                                          \
    /*  Initialize the first row of the Levenshtein matrix with `iota`-style arithmetic progression. */               \
    for (_distance_t idx_shorter = 0; idx_shorter != n; ++idx_shorter) previous_distances[idx_shorter] = idx_shorter; \
    /* The main loop of the algorithm with quadratic complexity. */                                                   \
    for (_distance_t idx_longer = 0; idx_longer != longer_length; ++idx_longer) {                                     \
        _char_t const longer_char = longer_chars[idx_longer];                                                         \
        /* Using pure pointer arithmetic is faster than iterating with an index. */                                   \
        _char_t const *shorter_ptr = shorter_chars;                                                                   \
        _distance_t const *previous_ptr = previous_distances;                                                         \
        _distance_t *current_ptr = current_distances;                                                                 \
        _distance_t *const current_end = current_ptr + shorter_length;                                                \
        current_ptr[0] = idx_longer + 1;                                                                              \
        for (; current_ptr != current_end; ++previous_ptr, ++current_ptr, ++shorter_ptr) {                            \
            _distance_t cost_substitution = previous_ptr[0] + (_distance_t)(longer_char != shorter_ptr[0]);           \
            /* We can avoid `+1` for costs here, shifting it to post-minimum computation, */                          \
            /* saving one increment operation. */                                                                     \
            _distance_t cost_deletion = previous_ptr[1];                                                              \
            _distance_t cost_insertion = current_ptr[0];                                                              \
            /* ? It might be a good idea to enforce branchless execution here. */                                     \
            /* ? The caveat being that the benchmarks on longer sequences backfire and more research is needed. */    \
            current_ptr[1] = sz_min_of_two(cost_substitution, sz_min_of_two(cost_deletion, cost_insertion) + 1);      \
        }                                                                                                             \
        /* Swap `previous_distances` and `current_distances` pointers. */                                             \
        _distance_t *temporary = previous_distances;                                                                  \
        previous_distances = current_distances;                                                                       \
        current_distances = temporary;                                                                                \
    }                                                                                                                 \
    /* Cache scalar before `free` call. */                                                                            \
    sz_size_t result = previous_distances[shorter_length];                                                            \
    alloc->free(buffer, buffer_length, alloc->handle);                                                                \
    return result;

    // Let's define a separate variant for bounded distance computation.
    // Practically the same as unbounded, but also collecting the running minimum within each row for early exit.
#define _wagner_fisher_bounded(_distance_t, _char_t)                                                                  \
    _char_t const *const longer_chars = (_char_t const *)longer;                                                      \
    _char_t const *const shorter_chars = (_char_t const *)shorter;                                                    \
    _distance_t *previous_distances = (_distance_t *)buffer;                                                          \
    _distance_t *current_distances = previous_distances + n;                                                          \
    for (_distance_t idx_shorter = 0; idx_shorter != n; ++idx_shorter) previous_distances[idx_shorter] = idx_shorter; \
    for (_distance_t idx_longer = 0; idx_longer != longer_length; ++idx_longer) {                                     \
        _char_t const longer_char = longer_chars[idx_longer];                                                         \
        _char_t const *shorter_ptr = shorter_chars;                                                                   \
        _distance_t const *previous_ptr = previous_distances;                                                         \
        _distance_t *current_ptr = current_distances;                                                                 \
        _distance_t *const current_end = current_ptr + shorter_length;                                                \
        current_ptr[0] = idx_longer + 1;                                                                              \
        /* Initialize min_distance with a value greater than bound */                                                 \
        _distance_t min_distance = bound - 1;                                                                         \
        for (; current_ptr != current_end; ++previous_ptr, ++current_ptr, ++shorter_ptr) {                            \
            _distance_t cost_substitution = previous_ptr[0] + (_distance_t)(longer_char != shorter_ptr[0]);           \
            _distance_t cost_deletion = previous_ptr[1];                                                              \
            _distance_t cost_insertion = current_ptr[0];                                                              \
            current_ptr[1] = sz_min_of_two(cost_substitution, sz_min_of_two(cost_deletion, cost_insertion) + 1);      \
            /* Keep track of the minimum distance seen so far in this row */                                          \
            min_distance = sz_min_of_two(current_ptr[1], min_distance);                                               \
        }                                                                                                             \
        /* If the minimum distance in this row exceeded the bound, return early */                                    \
        if (min_distance >= bound) {                                                                                  \
            alloc->free(buffer, buffer_length, alloc->handle);                                                        \
            return longer_length + 1;                                                                                 \
        }                                                                                                             \
        _distance_t *temporary = previous_distances;                                                                  \
        previous_distances = current_distances;                                                                       \
        current_distances = temporary;                                                                                \
    }                                                                                                                 \
    sz_size_t result = previous_distances[shorter_length];                                                            \
    alloc->free(buffer, buffer_length, alloc->handle);                                                                \
    return result;

    // Dispatch the actual computation.
    if (!bound) {
        if (can_be_unicode == sz_true_k) { _wagner_fisher_unbounded(sz_size_t, sz_rune_t); }
        else { _wagner_fisher_unbounded(sz_size_t, sz_u8_t); }
    }
    else {
        if (can_be_unicode == sz_true_k) { _wagner_fisher_bounded(sz_size_t, sz_rune_t); }
        else { _wagner_fisher_bounded(sz_size_t, sz_u8_t); }
    }
}

SZ_PUBLIC sz_size_t sz_edit_distance_serial(     //
    sz_cptr_t longer, sz_size_t longer_length,   //
    sz_cptr_t shorter, sz_size_t shorter_length, //
    sz_size_t bound, sz_memory_allocator_t *alloc) {

    // Let's make sure that we use the amount proportional to the
    // number of elements in the shorter string, not the larger.
    if (shorter_length > longer_length) {
        sz_pointer_swap((void **)&longer_length, (void **)&shorter_length);
        sz_pointer_swap((void **)&longer, (void **)&shorter);
    }

    // Skip the matching prefixes and suffixes, they won't affect the distance.
    for (sz_cptr_t a_end = longer + longer_length, b_end = shorter + shorter_length;
         longer != a_end && shorter != b_end && *longer == *shorter;
         ++longer, ++shorter, --longer_length, --shorter_length);
    for (; longer_length && shorter_length && longer[longer_length - 1] == shorter[shorter_length - 1];
         --longer_length, --shorter_length);

    // Bounded computations may exit early.
    int const is_bounded = bound < longer_length;
    if (is_bounded) {
        // If one of the strings is empty - the edit distance is equal to the length of the other one.
        if (longer_length == 0) return sz_min_of_two(shorter_length, bound);
        if (shorter_length == 0) return sz_min_of_two(longer_length, bound);
        // If the difference in length is beyond the `bound`, there is no need to check at all.
        if (longer_length - shorter_length > bound) return bound;
    }

    if (shorter_length == 0) return longer_length; // If no mismatches were found - the distance is zero.
    if (shorter_length == longer_length && !is_bounded)
        return _sz_edit_distance_skewed_diagonals_serial(longer, longer_length, shorter, shorter_length, bound, alloc);
    return _sz_edit_distance_wagner_fisher_serial( //
        longer, longer_length, shorter, shorter_length, bound, sz_false_k, alloc);
}

SZ_PUBLIC sz_ssize_t sz_alignment_score_serial(       //
    sz_cptr_t longer, sz_size_t longer_length,        //
    sz_cptr_t shorter, sz_size_t shorter_length,      //
    sz_error_cost_t const *subs, sz_error_cost_t gap, //
    sz_memory_allocator_t *alloc) {

    // If one of the strings is empty - the edit distance is equal to the length of the other one
    if (longer_length == 0) return (sz_ssize_t)shorter_length * gap;
    if (shorter_length == 0) return (sz_ssize_t)longer_length * gap;

    // Let's make sure that we use the amount proportional to the
    // number of elements in the shorter string, not the larger.
    if (shorter_length > longer_length) {
        sz_pointer_swap((void **)&longer_length, (void **)&shorter_length);
        sz_pointer_swap((void **)&longer, (void **)&shorter);
    }

    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    sz_size_t n = shorter_length + 1;
    sz_size_t buffer_length = sizeof(sz_ssize_t) * n * 2;
    sz_ssize_t *distances = (sz_ssize_t *)alloc->allocate(buffer_length, alloc->handle);
    sz_ssize_t *previous_distances = distances;
    sz_ssize_t *current_distances = previous_distances + n;

    for (sz_size_t idx_shorter = 0; idx_shorter != n; ++idx_shorter)
        previous_distances[idx_shorter] = (sz_ssize_t)idx_shorter * gap;

    sz_u8_t const *shorter_unsigned = (sz_u8_t const *)shorter;
    sz_u8_t const *longer_unsigned = (sz_u8_t const *)longer;
    for (sz_size_t idx_longer = 0; idx_longer != longer_length; ++idx_longer) {
        current_distances[0] = ((sz_ssize_t)idx_longer + 1) * gap;

        // Initialize min_distance with a value greater than bound
        sz_error_cost_t const *a_subs = subs + longer_unsigned[idx_longer] * 256ul;
        for (sz_size_t idx_shorter = 0; idx_shorter != shorter_length; ++idx_shorter) {
            sz_ssize_t cost_deletion = previous_distances[idx_shorter + 1] + gap;
            sz_ssize_t cost_insertion = current_distances[idx_shorter] + gap;
            sz_ssize_t cost_substitution = previous_distances[idx_shorter] + a_subs[shorter_unsigned[idx_shorter]];
            current_distances[idx_shorter + 1] = sz_max_of_three(cost_deletion, cost_insertion, cost_substitution);
        }

        // Swap previous_distances and current_distances pointers
        sz_pointer_swap((void **)&previous_distances, (void **)&current_distances);
    }

    // Cache scalar before `free` call.
    sz_ssize_t result = previous_distances[shorter_length];
    alloc->free(distances, buffer_length, alloc->handle);
    return result;
}

SZ_PUBLIC sz_size_t sz_hamming_distance_serial( //
    sz_cptr_t a, sz_size_t a_length,            //
    sz_cptr_t b, sz_size_t b_length,            //
    sz_size_t bound) {

    sz_size_t const min_length = sz_min_of_two(a_length, b_length);
    sz_size_t const max_length = sz_max_of_two(a_length, b_length);
    sz_cptr_t const a_end = a + min_length;
    bound = bound == 0 ? max_length : bound;

    // Walk through both strings using SWAR and counting the number of differing characters.
    sz_size_t distance = max_length - min_length;
#if SZ_USE_MISALIGNED_LOADS && !_SZ_IS_BIG_ENDIAN
    if (min_length >= SZ_SWAR_THRESHOLD) {
        sz_u64_vec_t a_vec, b_vec, match_vec;
        for (; a + 8 <= a_end && distance < bound; a += 8, b += 8) {
            a_vec.u64 = sz_u64_load(a).u64;
            b_vec.u64 = sz_u64_load(b).u64;
            match_vec = _sz_u64_each_byte_equal(a_vec, b_vec);
            distance += sz_u64_popcount((~match_vec.u64) & 0x8080808080808080ull);
        }
    }
#endif

    for (; a != a_end && distance < bound; ++a, ++b) { distance += (*a != *b); }
    return sz_min_of_two(distance, bound);
}

SZ_PUBLIC sz_size_t sz_hamming_distance_utf8_serial( //
    sz_cptr_t a, sz_size_t a_length,                 //
    sz_cptr_t b, sz_size_t b_length,                 //
    sz_size_t bound) {

    sz_cptr_t const a_end = a + a_length;
    sz_cptr_t const b_end = b + b_length;
    sz_size_t distance = 0;

    sz_rune_t a_rune, b_rune;
    sz_rune_length_t a_rune_length, b_rune_length;

    if (bound) {
        for (; a < a_end && b < b_end && distance < bound; a += a_rune_length, b += b_rune_length) {
            _sz_extract_utf8_rune(a, &a_rune, &a_rune_length);
            _sz_extract_utf8_rune(b, &b_rune, &b_rune_length);
            distance += (a_rune != b_rune);
        }
        // If one string has more runes, we need to go through the tail.
        if (distance < bound) {
            for (; a < a_end && distance < bound; a += a_rune_length, ++distance)
                _sz_extract_utf8_rune(a, &a_rune, &a_rune_length);

            for (; b < b_end && distance < bound; b += b_rune_length, ++distance)
                _sz_extract_utf8_rune(b, &b_rune, &b_rune_length);
        }
    }
    else {
        for (; a < a_end && b < b_end; a += a_rune_length, b += b_rune_length) {
            _sz_extract_utf8_rune(a, &a_rune, &a_rune_length);
            _sz_extract_utf8_rune(b, &b_rune, &b_rune_length);
            distance += (a_rune != b_rune);
        }
        // If one string has more runes, we need to go through the tail.
        for (; a < a_end; a += a_rune_length, ++distance) _sz_extract_utf8_rune(a, &a_rune, &a_rune_length);
        for (; b < b_end; b += b_rune_length, ++distance) _sz_extract_utf8_rune(b, &b_rune, &b_rune_length);
    }
    return distance;
}

#pragma endregion // Serial Implementation

/*  AVX2 implementation of the string similarity algorithms for Haswell processors and newer.
 *  Very minimalistic (compared to AVX-512), but still faster than the serial implementation.
 */
#pragma region Haswell Implementation
#if SZ_USE_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2")
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_HASWELL
#pragma endregion // Haswell Implementation

/*  AVX512 implementation of the string similarity algorithms for Skylake and newer CPUs.
 *  Includes extensions: F, CD, ER, PF, VL, DQ, BW.
 *
 *  This is the "starting level" for the advanced algorithms using K-mask registers on x86.
 */
#pragma region Skylake Implementation
#if SZ_USE_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2")
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_SKYLAKE
#pragma endregion // Skylake Implementation

/*  AVX512 implementation of the string similarity algorithms for Ice Lake and newer CPUs.
 *  Includes extensions:
 *      - 2017 Skylake: F, CD, ER, PF, VL, DQ, BW,
 *      - 2018 CannonLake: IFMA, VBMI,
 *      - 2019 Ice Lake: VPOPCNTDQ, VNNI, VBMI2, BITALG, GFNI, VPCLMULQDQ, VAES.
 */
#pragma region Ice Lake Implementation
#if SZ_USE_ICE
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "bmi", "bmi2")
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,bmi,bmi2"))), \
                             apply_to = function)

/**
 *  @brief  Computes the edit distance between two very short byte-strings using the AVX-512VBMI extensions.
 *
 *  Applies to string lengths up to 63, and evaluates at most (63 * 2 + 1 = 127) diagonals, or just as many loop
 * cycles. Supports an early exit, if the distance is bounded. Keeps all of the data and Levenshtein matrices skew
 * diagonal in just a couple of registers. Benefits from the @b `vpermb` instructions, that can rotate the bytes
 * across the entire ZMM register.
 */
SZ_INTERNAL sz_size_t _sz_edit_distance_skewed_diagonals_upto63_ice( //
    sz_cptr_t shorter, sz_size_t shorter_length,                     //
    sz_cptr_t longer, sz_size_t longer_length,                       //
    sz_size_t bound) {

    sz_size_t const max_length = 63u;
    _sz_assert(shorter_length <= longer_length && "The 'shorter' string is longer than the 'longer' one.");
    _sz_assert(shorter_length < max_length && "The length must fit into 16-bit integer. Otherwise use serial variant.");

    // We are going to store 3 diagonals of the matrix, assuming each would fit into a single ZMM register.
    // The length of the longest (main) diagonal would be `shorter_dim = (shorter_length + 1)`.
    sz_size_t const shorter_dim = shorter_length + 1;
    sz_size_t const longer_dim = longer_length + 1;

    // The next few buffers will be swapped around.
    sz_u512_vec_t previous_vec, current_vec, next_vec;
    sz_u512_vec_t gaps_vec, substitutions_vec;

    // Load the strings into ZMM registers - just once.
    sz_u512_vec_t longer_vec, shorter_vec, shorter_rotated_vec, rotate_left_vec, rotate_right_vec, ones_vec, bound_vec;
    longer_vec.zmm = _mm512_maskz_loadu_epi8(_sz_u64_mask_until(longer_length), longer);
    rotate_left_vec.zmm = _mm512_set_epi8(                              //
        0, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49,  //
        48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, //
        32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, //
        16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
    rotate_right_vec.zmm = _mm512_set_epi8(                             //
        62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48,     //
        47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, //
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, //
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 63);
    ones_vec.zmm = _mm512_set1_epi8(1);
    bound_vec.zmm = _mm512_set1_epi8(bound <= 255 ? (sz_u8_t)bound : 255);

    // To simplify comparisons and traversals, we want to reverse the order of bytes in the shorter string.
    for (sz_size_t i = 0; i != shorter_length; ++i) shorter_vec.u8s[63 - i] = shorter[i];
    shorter_rotated_vec.zmm = _mm512_permutexvar_epi8(rotate_right_vec.zmm, shorter_vec.zmm);

    // Let's say we are dealing with 3 and 5 letter words.
    // The matrix will have size 4 x 6, parameterized as (shorter_dim x longer_dim).
    // It will have:
    // - 4 diagonals of increasing length, at positions: 0, 1, 2, 3.
    // - 2 diagonals of fixed length, at positions: 4, 5.
    // - 3 diagonals of decreasing length, at positions: 6, 7, 8.
    sz_size_t const diagonals_count = shorter_dim + longer_dim - 1;

    // Initialize the first two diagonals:
    //
    //      previous_vec.u8s[0] = 0;
    //      current_vec.u8s[0] = current_vec.u8s[1] = 1;
    //
    // We can do a similar thing with vector ops:
    previous_vec.zmm = _mm512_setzero_si512();
    current_vec.zmm = _mm512_set1_epi8(1);

    // We skip diagonals 0 and 1, as they are trivial.
    // We will start with diagonal 2, which has length 3, with the first and last elements being preset,
    // so we are effectively computing just one value, as will be marked by a single set bit in
    // the `next_diagonal_mask` on the very first iteration.
    sz_size_t next_diagonal_index = 2;
    __mmask64 next_diagonal_mask = 0;

    // Progress through the upper triangle of the Levenshtein matrix.
    for (; next_diagonal_index != shorter_dim; ++next_diagonal_index) {
        // After this iteration, the values at offset `0` and `next_diagonal_index` in the `next_vec`
        // should be set to `next_diagonal_index`, but it's easier to broadcast the value to the whole vector,
        // and later merge with a mask with new values.
        next_vec.zmm = _mm512_set1_epi8((sz_u8_t)next_diagonal_index);

        // The mask also adds one set bit.
        next_diagonal_mask = _kor_mask64(next_diagonal_mask, 1);
        next_diagonal_mask = _kshiftli_mask64(next_diagonal_mask, 1);

        // Check for equality between string slices.
        __mmask64 conflict_mask = _mm512_cmpneq_epi8_mask(longer_vec.zmm, shorter_rotated_vec.zmm);
        substitutions_vec.zmm = _mm512_mask_add_epi8(previous_vec.zmm, conflict_mask, previous_vec.zmm, ones_vec.zmm);
        substitutions_vec.zmm = _mm512_permutexvar_epi8(rotate_right_vec.zmm, substitutions_vec.zmm);
        gaps_vec.zmm = _mm512_add_epi8(
            // Insertions or deletions
            _mm512_min_epu8(_mm512_permutexvar_epi8(rotate_right_vec.zmm, current_vec.zmm), current_vec.zmm),
            ones_vec.zmm);
        next_vec.zmm = _mm512_mask_min_epu8(next_vec.zmm, next_diagonal_mask, gaps_vec.zmm, substitutions_vec.zmm);

        // Mark the current skewed diagonal as the previous one and the next one as the current one.
        previous_vec.zmm = current_vec.zmm;
        current_vec.zmm = next_vec.zmm;

        // Shift the shorter string
        shorter_rotated_vec.zmm = _mm512_permutexvar_epi8(rotate_right_vec.zmm, shorter_rotated_vec.zmm);

        // Check if we can exit early - if none of the diagonals values are smaller than the upper distance bound.
        __mmask64 within_bound_mask = _mm512_cmple_epu8_mask(next_vec.zmm, bound_vec.zmm);
        if (_ktestz_mask64_u8(within_bound_mask, next_diagonal_mask) == 1) return longer_length + 1;
    }

    // Now let's handle the anti-diagonal band of the matrix, between the top and bottom triangles.
    for (; next_diagonal_index != longer_dim; ++next_diagonal_index) {
        // After this iteration, the value `shorted_dim - 1` in the `next_vec`
        // should be set to `next_diagonal_index`, but it's easier to broadcast the value to the whole vector,
        // and later merge with a mask with new values.
        next_vec.zmm = _mm512_set1_epi8((sz_u8_t)next_diagonal_index);

        // Make sure we update the first entry.
        next_diagonal_mask = _kor_mask64(next_diagonal_mask, 1);

        // Check for equality between string slices.
        __mmask64 conflict_mask = _mm512_cmpneq_epi8_mask(longer_vec.zmm, shorter_rotated_vec.zmm);
        substitutions_vec.zmm = _mm512_mask_add_epi8(previous_vec.zmm, conflict_mask, previous_vec.zmm, ones_vec.zmm);
        gaps_vec.zmm = _mm512_add_epi8(
            // Insertions or deletions
            _mm512_min_epu8(current_vec.zmm, _mm512_permutexvar_epi8(rotate_left_vec.zmm, current_vec.zmm)),
            ones_vec.zmm);
        next_vec.zmm = _mm512_mask_min_epu8(next_vec.zmm, next_diagonal_mask, gaps_vec.zmm, substitutions_vec.zmm);

        // Mark the current skewed diagonal as the previous one and the next one as the current one.
        previous_vec.zmm = _mm512_permutexvar_epi8(rotate_left_vec.zmm, current_vec.zmm);
        current_vec.zmm = next_vec.zmm;

        // Let's shift the longer string now.
        longer_vec.zmm = _mm512_permutexvar_epi8(rotate_left_vec.zmm, longer_vec.zmm);

        // Check if we can exit early - if none of the diagonals values are smaller than the upper distance bound.
        __mmask64 within_bound_mask = _mm512_cmple_epu8_mask(next_vec.zmm, bound_vec.zmm);
        if (_ktestz_mask64_u8(within_bound_mask, next_diagonal_mask) == 1) return longer_length + 1;
    }

    // Now let's handle the bottom right triangle.
    for (; next_diagonal_index != diagonals_count; ++next_diagonal_index) {

        // Check for equality between string slices.
        __mmask64 conflict_mask = _mm512_cmpneq_epi8_mask(longer_vec.zmm, shorter_rotated_vec.zmm);
        substitutions_vec.zmm = _mm512_mask_add_epi8(previous_vec.zmm, conflict_mask, previous_vec.zmm, ones_vec.zmm);
        gaps_vec.zmm = _mm512_add_epi8(
            // Insertions or deletions
            _mm512_min_epu8(current_vec.zmm, _mm512_permutexvar_epi8(rotate_left_vec.zmm, current_vec.zmm)),
            ones_vec.zmm);
        next_vec.zmm = _mm512_min_epu8(gaps_vec.zmm, substitutions_vec.zmm);

        // Mark the current skewed diagonal as the previous one and the next one as the current one.
        previous_vec.zmm = _mm512_permutexvar_epi8(rotate_left_vec.zmm, current_vec.zmm);
        current_vec.zmm = next_vec.zmm;

        // Let's shift the longer string now.
        longer_vec.zmm = _mm512_permutexvar_epi8(rotate_left_vec.zmm, longer_vec.zmm);

        // Check if we can exit early - if none of the diagonals values are smaller than the upper distance bound.
        __mmask64 within_bound_mask = _mm512_cmple_epu8_mask(next_vec.zmm, bound_vec.zmm);
        if (_ktestz_mask64_u8(within_bound_mask, next_diagonal_mask) == 1) return longer_length + 1;

        // In every following iterations we take use a shorter prefix of each register,
        // but we don't need to update the `next_diagonal_mask` anymore... except for the early exit.
        next_diagonal_mask = _kshiftri_mask64(next_diagonal_mask, 1);
    }
    return current_vec.u8s[0];
}

/**
 *  @brief  Computes the edit distance between two somewhat short bytes-strings using the AVX-512VBMI extensions.
 *
 *  Applies to string lengths up to 127, and evaluates at most (127 * 2 + 1 = 255) diagonals.
 *  Supports an early exit, if the distance is bounded.
 *  Uses a lot more CPU registers space, than the `upto63` variant.
 *  Benefits from the @b `vpermi2b` instructions, that can rotate the bytes in 2 registers at once.
 *
 *  This may be one of the most frequently called kernels for:
 *  - source code analysis, assuming most lines are either under 80 or under 120 characters long.
 *  - DNA sequence alignment, as most short reads are 50-300 characters long.
 */
SZ_INTERNAL sz_size_t _sz_edit_distance_skewed_diagonals_upto127_ice( //
    sz_cptr_t shorter, sz_size_t shorter_length,                      //
    sz_cptr_t longer, sz_size_t longer_length,                        //
    sz_size_t bound) {
    sz_unused(shorter && shorter_length && longer && longer_length && bound);
    return 0;
}

/**
 *  @brief  Computes the edit distance between two longer bytes-strings using the AVX-512VBMI extensions.
 *
 *  Applies to string lengths up to 255, and evaluates at most (255 * 2 + 1 = 511) diagonals.
 *  Supports an early exit, if the distance is bounded.
 *  Uses a lot more CPU registers space, than the `upto63` variant.
 *
 *  Each of 2x string ends up occupying 4 ZMM registers, and each of 3x diagonals uses 4 ZMM registers.
 *  So 20x of the 32x are persistently occupied, and the rest are used for math temporarily.
 *  This is the largest space-efficient variant, as strings beyond 255 characters may require
 *  16-bit accumulators, which would be a significant bottleneck.
 */
SZ_INTERNAL sz_size_t _sz_edit_distance_skewed_diagonals_upto_ice( //
    sz_cptr_t shorter, sz_size_t shorter_length,                   //
    sz_cptr_t longer, sz_size_t longer_length,                     //
    sz_size_t bound) {
    sz_unused(shorter && shorter_length && longer && longer_length && bound);
    return 0;
}

/**
 *  @brief  Computes the edit distance between two longer bytes-strings using the AVX-512VBMI extensions,
 *          assuming the upper distance bound can not exceed 255, but the string length can be arbitrary.
 *
 *  Applies to string lengths up to 255, and evaluates at most (255 * 2 + 1 = 511) diagonals.
 *  Supports an early exit, if the distance is bounded.
 *  Uses a lot more CPU registers space, than the `upto63` variant.
 *
 *  Each of 2x string ends up occupying 4 ZMM registers, and each of 3x diagonals uses 4 ZMM registers.
 *  So 20x of the 32x are persistently occupied, and the rest are used for math temporarily.
 *  This is the largest space-efficient variant, as strings beyond 255 characters may require
 *  16-bit accumulators, which would be a significant bottleneck.
 */
SZ_INTERNAL sz_size_t _sz_edit_distance_skewed_diagonals_upto255bound_ice( //
    sz_cptr_t shorter, sz_size_t shorter_length,                           //
    sz_cptr_t longer, sz_size_t longer_length,                             //
    sz_size_t bound) {
    sz_unused(shorter && shorter_length && longer && longer_length && bound);
    return 0;
}

/**
 *  @brief  Computes the edit distance between two mid-length UTF-8-strings using the AVX-512VBMI extensions.
 *
 *  Applies to string lengths up to 127, and evaluates at most (127 * 2 + 1 = 511) diagonals.
 *  Supports an early exit, if the distance is bounded.
 *  Benefits from the @b `valignd` instructions used to rotate UTF-32 unpacked unicode codepoints.
 *
 *  Each string is unpacked into 128 characters * 4 bytes per character / 64 bytes per register = 8 registers.
 */
SZ_INTERNAL sz_size_t _sz_edit_distance_utf8_skewed_diagonals_upto127_ice( //
    sz_cptr_t shorter, sz_size_t shorter_length,                           //
    sz_cptr_t longer, sz_size_t longer_length,                             //
    sz_size_t bound) {
    sz_unused(shorter && shorter_length && longer && longer_length && bound);
    return 0;
}

SZ_INTERNAL sz_size_t _sz_edit_distance_skewed_diagonals_upto65k_ice( //
    sz_cptr_t shorter, sz_size_t shorter_length,                      //
    sz_cptr_t longer, sz_size_t longer_length,                        //
    sz_size_t bound, sz_memory_allocator_t *alloc) {

    sz_unused(shorter && longer && bound && alloc);

    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    // TODO: Generalize!
    sz_size_t const max_length = 256u * 256u;
    _sz_assert(shorter_length <= longer_length && "The 'shorter' string is longer than the 'longer' one.");
    _sz_assert(shorter_length < max_length && "The length must fit into 16-bit integer. Otherwise use serial variant.");
    sz_unused(longer_length && bound && max_length);

#if 0
    // We are going to store 3 diagonals of the matrix.
    // The length of the longest (main) diagonal would be `shorter_dim = (shorter_length + 1)`.
    sz_size_t const shorter_dim = shorter_length + 1;
    sz_size_t const longer_dim = longer_length + 1;
    // Unlike the serial version, we also want to avoid reverse-order iteration over teh shorter string.
    // So let's allocate a bit more memory and reverse-export our shorter string into that buffer.
    sz_size_t const buffer_length = sizeof(sz_u16_t) * longer_dim * 3 + shorter_length;
    sz_u16_t *const distances = (sz_u16_t *)alloc->allocate(buffer_length, alloc->handle);
    if (!distances) return SZ_SIZE_MAX;

    // The next few pointers will be swapped around.
    sz_u16_t *previous_distances = distances;
    sz_u16_t *current_distances = previous_distances + longer_dim;
    sz_u16_t *next_distances = current_distances + longer_dim;
    sz_ptr_t const shorter_reversed = (sz_ptr_t)(next_distances + longer_dim);

    // Export the reversed string into the buffer.
    for (sz_size_t i = 0; i != shorter_length; ++i) shorter_reversed[i] = shorter[shorter_length - 1 - i];

    // Initialize the first two diagonals:
    previous_distances[0] = 0;
    current_distances[0] = current_distances[1] = 1;

    // Using ZMM registers, we can process 32x 16-bit values at once,
    // storing 16 bytes of each string in YMM registers.
    sz_u512_vec_t insertions_vec, deletions_vec, substitutions_vec, next_vec;
    sz_u512_vec_t ones_u16_vec;
    ones_u16_vec.zmm = _mm512_set1_epi16(1);

    // This is a mixed-precision implementation, using 8-bit representations for part of the operations.
    // Even there, in case `SZ_USE_HASWELL=0`, let's use the `sz_u512_vec_t` type, addressing the first YMM halfs.
    sz_u512_vec_t shorter_vec, longer_vec;
    sz_u512_vec_t ones_u8_vec;
    ones_u8_vec.ymms[0] = _mm256_set1_epi8(1);

    // Let's say we are dealing with 3 and 5 letter words.
    // The matrix will have size 4 x 6, parameterized as (shorter_dim x longer_dim).
    // It will have:
    // - 4 diagonals of increasing length, at positions: 0, 1, 2, 3.
    // - 2 diagonals of fixed length, at positions: 4, 5.
    // - 3 diagonals of decreasing length, at positions: 6, 7, 8.
    sz_size_t const diagonals_count = shorter_dim + longer_dim - 1;

    // Progress through the upper triangle of the Levenshtein matrix.
    sz_size_t next_diagonal_index = 2;
    for (; next_diagonal_index != shorter_dim; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = next_diagonal_index + 1;
        for (sz_size_t offset_within_diagonal = 0; offset_within_diagonal + 2 < next_diagonal_length;) {
            sz_u32_t remaining_length = (sz_u32_t)(next_diagonal_length - offset_within_diagonal - 2);
            sz_u32_t register_length = remaining_length < 32 ? remaining_length : 32;
            sz_u32_t remaining_length_mask = _bzhi_u32(0xFFFFFFFFu, register_length);
            longer_vec.ymms[0] = _mm256_maskz_loadu_epi8(remaining_length_mask, longer + offset_within_diagonal);
            // Our original code addressed the shorter string `[next_diagonal_index - offset_within_diagonal - 2]`
            // for growing `offset_within_diagonal`. If the `shorter` string was reversed, the
            // `[next_diagonal_index - offset_within_diagonal - 2]` would be equal to `[shorter_length - 1 -
            // next_diagonal_index + offset_within_diagonal + 2]`. Which simplified would be equal to
            // `[shorter_length - next_diagonal_index + offset_within_diagonal + 1]`.
            shorter_vec.ymms[0] = _mm256_maskz_loadu_epi8( //
                remaining_length_mask,
                shorter_reversed + shorter_length - next_diagonal_index + offset_within_diagonal + 1);
            // For substitutions, perform the equality comparison using AVX2 instead of AVX-512
            // to get the result as a vector, instead of a bitmask. Adding 1 to every scalar we can overflow
            // transforming from {0xFF, 0} values to {0, 1} values - exactly what we need. Then - upcast to 16-bit.
            substitutions_vec.zmm = _mm512_cvtepi8_epi16( //
                _mm256_add_epi8(_mm256_cmpeq_epi8(longer_vec.ymms[0], shorter_vec.ymms[0]), ones_u8_vec.ymms[0]));
            substitutions_vec.zmm = _mm512_add_epi16( //
                substitutions_vec.zmm,
                _mm512_maskz_loadu_epi16(remaining_length_mask, previous_distances + offset_within_diagonal));
            // For insertions and deletions, on modern hardware, it's faster to issue two separate loads,
            // than rotate the bytes in the ZMM register.
            insertions_vec.zmm =
                _mm512_maskz_loadu_epi16(remaining_length_mask, current_distances + offset_within_diagonal);
            deletions_vec.zmm =
                _mm512_maskz_loadu_epi16(remaining_length_mask, current_distances + offset_within_diagonal + 1);
            // First get the minimum of insertions and deletions.
            next_vec.zmm = _mm512_add_epi16(_mm512_min_epu16(insertions_vec.zmm, deletions_vec.zmm), ones_u16_vec.zmm);
            next_vec.zmm = _mm512_min_epu16(next_vec.zmm, substitutions_vec.zmm);
            _mm512_mask_storeu_epi16(next_distances + offset_within_diagonal + 1, remaining_length_mask, next_vec.zmm);
            offset_within_diagonal += register_length;
        }
        // Don't forget to populate the first row and the first column of the Levenshtein matrix.
        next_distances[0] = next_distances[next_diagonal_length - 1] = (sz_u16_t)next_diagonal_index;
        // Perform a circular rotation (three-way swap) of those buffers, to reuse the memory.
        sz_u16_t *temporary = previous_distances;
        previous_distances = current_distances;
        current_distances = next_distances;
        next_distances = temporary;
    }

    // By now we've scanned through the upper triangle of the matrix, where each subsequent iteration results in a
    // larger diagonal. From now onwards, we will be shrinking. Instead of adding value equal to the skewed diagonal
    // index on either side, we will be cropping those values out.
    for (; next_diagonal_index != diagonals_count; ++next_diagonal_index) {
        sz_size_t const next_diagonal_length = diagonals_count - next_diagonal_index;
        for (sz_size_t i = 0; i != next_diagonal_length;) {
            sz_u32_t remaining_length = (sz_u32_t)(next_diagonal_length - i);
            sz_u32_t register_length = remaining_length < 32 ? remaining_length : 32;
            sz_u32_t remaining_length_mask = _bzhi_u32(0xFFFFFFFFu, register_length);
            longer_vec.ymms[0] = _mm256_maskz_loadu_epi8(remaining_length_mask, longer + next_diagonal_index - n + i);
            // Our original code addressed the shorter string `[shorter_length - 1 - i]` for growing `i`.
            // If the `shorter` string was reversed, the `[shorter_length - 1 - i]` would
            // be equal to `[shorter_length - 1 - shorter_length + 1 + i]`.
            // Which simplified would be equal to just `[i]`. Beautiful!
            shorter_vec.ymms[0] = _mm256_maskz_loadu_epi8(remaining_length_mask, shorter_reversed + i);
            // For substitutions, perform the equality comparison using AVX2 instead of AVX-512
            // to get the result as a vector, instead of a bitmask. The compare it against the accumulated
            // substitution costs.
            substitutions_vec.zmm = _mm512_cvtepi8_epi16( //
                _mm256_add_epi8(_mm256_cmpeq_epi8(longer_vec.ymms[0], shorter_vec.ymms[0]), ones_u8_vec.ymms[0]));
            substitutions_vec.zmm = _mm512_add_epi16( //
                substitutions_vec.zmm, _mm512_maskz_loadu_epi16(remaining_length_mask, previous_distances + i));
            // For insertions and deletions, on modern hardware, it's faster to issue two separate loads,
            // than rotate the bytes in the ZMM register.
            insertions_vec.zmm = _mm512_maskz_loadu_epi16(remaining_length_mask, current_distances + i);
            deletions_vec.zmm = _mm512_maskz_loadu_epi16(remaining_length_mask, current_distances + i + 1);
            // First get the minimum of insertions and deletions.
            next_vec.zmm = _mm512_add_epi16(_mm512_min_epu16(insertions_vec.zmm, deletions_vec.zmm), ones_u16_vec.zmm);
            next_vec.zmm = _mm512_min_epu16(next_vec.zmm, substitutions_vec.zmm);
            _mm512_mask_storeu_epi16(next_distances + i, remaining_length_mask, next_vec.zmm);
            i += register_length;
        }

        // Perform a circular rotation (three-way swap) of those buffers, to reuse the memory, this time, with a shift,
        // dropping the first element in the current array.
        sz_u16_t *temporary = previous_distances;
        previous_distances = current_distances + 1;
        current_distances = next_distances;
        next_distances = temporary;
    }

    // Cache scalar before `free` call.
    sz_size_t result = current_distances[0];
    alloc->free(distances, buffer_length, alloc->handle);
    return result;
#endif
    return 0;
}

SZ_INTERNAL sz_size_t sz_edit_distance_ice(      //
    sz_cptr_t shorter, sz_size_t shorter_length, //
    sz_cptr_t longer, sz_size_t longer_length,   //
    sz_size_t bound, sz_memory_allocator_t *alloc) {

    // Bounded computations may exit early.
    int const is_bounded = bound < longer_length;
    if (is_bounded) {
        // If one of the strings is empty - the edit distance is equal to the length of the other one.
        if (longer_length == 0) return sz_min_of_two(shorter_length, bound);
        if (shorter_length == 0) return sz_min_of_two(longer_length, bound);
        // If the difference in length is beyond the `bound`, there is no need to check at all.
        if (longer_length - shorter_length > bound) return bound;
    }

    // Make sure the shorter string is actually shorter.
    if (shorter_length > longer_length) {
        sz_cptr_t temporary = shorter;
        shorter = longer;
        longer = temporary;
        sz_size_t temporary_length = shorter_length;
        shorter_length = longer_length;
        longer_length = temporary_length;
    }

    // Dispatch the right implementation based on the length of the strings.
    if (longer_length < 64u)
        return _sz_edit_distance_skewed_diagonals_upto63_ice( //
            shorter, shorter_length, longer, longer_length, bound);
    // else if (longer_length < 256u * 256u)
    //     return _sz_edit_distance_skewed_diagonals_upto65k_ice( //
    //         shorter, shorter_length, longer, longer_length, bound, alloc);
    else
        return sz_edit_distance_serial(shorter, shorter_length, longer, longer_length, bound, alloc);
}

/**
 *  Computes the Needleman Wunsch alignment score between two strings.
 *  The method uses 32-bit integers to accumulate the running score for every cell in the matrix.
 *  Assuming the costs of substitutions can be arbitrary signed 8-bit integers, the method is expected to be used
 *  on strings not exceeding 2^24 length or 16.7 million characters.
 *
 *  Unlike the `_sz_edit_distance_skewed_diagonals_upto65k_avx512` method, this one uses signed integers to store
 *  the accumulated score. Moreover, it's primary bottleneck is the latency of gathering the substitution costs
 *  from the substitution matrix. If we use the diagonal order, we will be comparing a slice of the first string
 * with a slice of the second. If we stick to the conventional horizontal order, we will be comparing one character
 * against a slice, which is much easier to optimize. In that case we are sampling costs not from arbitrary parts of
 *  a 256 x 256 matrix, but from a single row!
 */
SZ_INTERNAL sz_ssize_t _sz_alignment_score_wagner_fisher_upto17m_ice( //
    sz_cptr_t shorter, sz_size_t shorter_length,                      //
    sz_cptr_t longer, sz_size_t longer_length,                        //
    sz_error_cost_t const *subs, sz_error_cost_t gap, sz_memory_allocator_t *alloc) {

    // If one of the strings is empty - the edit distance is equal to the length of the other one
    if (longer_length == 0) return (sz_ssize_t)shorter_length * gap;
    if (shorter_length == 0) return (sz_ssize_t)longer_length * gap;

    // Let's make sure that we use the amount proportional to the
    // number of elements in the shorter string, not the larger.
    if (shorter_length > longer_length) {
        sz_pointer_swap((void **)&longer_length, (void **)&shorter_length);
        sz_pointer_swap((void **)&longer, (void **)&shorter);
    }

    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    sz_size_t const max_length = 256ull * 256ull * 256ull;
    sz_size_t const n = longer_length + 1;
    _sz_assert(n < max_length && "The length must fit into 24-bit integer. Otherwise use serial variant.");
    sz_unused(longer_length && max_length);

    sz_size_t buffer_length = sizeof(sz_i32_t) * n * 2;
    sz_i32_t *distances = (sz_i32_t *)alloc->allocate(buffer_length, alloc->handle);
    sz_i32_t *previous_distances = distances;
    sz_i32_t *current_distances = previous_distances + n;

    // Initialize the first row of the Levenshtein matrix with `iota`.
    for (sz_size_t idx_longer = 0; idx_longer != n; ++idx_longer)
        previous_distances[idx_longer] = (sz_i32_t)idx_longer * gap;

    /// Contains up to 16 consecutive characters from the longer string.
    sz_u512_vec_t longer_vec;
    sz_u512_vec_t cost_deletion_vec, cost_substitution_vec, lookup_substitution_vec, current_vec;
    sz_u512_vec_t row_first_subs_vec, row_second_subs_vec, row_third_subs_vec, row_fourth_subs_vec;
    sz_u512_vec_t shuffled_first_subs_vec, shuffled_second_subs_vec, shuffled_third_subs_vec, shuffled_fourth_subs_vec;

    // Prepare constants and masks.
    sz_u512_vec_t is_third_or_fourth_vec, is_second_or_fourth_vec, gap_vec;
    {
        char is_third_or_fourth_check, is_second_or_fourth_check;
        *(sz_u8_t *)&is_third_or_fourth_check = 0x80, *(sz_u8_t *)&is_second_or_fourth_check = 0x40;
        is_third_or_fourth_vec.zmm = _mm512_set1_epi8(is_third_or_fourth_check);
        is_second_or_fourth_vec.zmm = _mm512_set1_epi8(is_second_or_fourth_check);
        gap_vec.zmm = _mm512_set1_epi32(gap);
    }

    sz_u8_t const *shorter_unsigned = (sz_u8_t const *)shorter;
    for (sz_size_t idx_shorter = 0; idx_shorter != shorter_length; ++idx_shorter) {
        sz_i32_t last_in_row = current_distances[0] = (sz_i32_t)(idx_shorter + 1) * gap;

        // Load one row of the substitution matrix into four ZMM registers.
        sz_error_cost_t const *row_subs = subs + shorter_unsigned[idx_shorter] * 256u;
        row_first_subs_vec.zmm = _mm512_loadu_si512(row_subs + 64 * 0);
        row_second_subs_vec.zmm = _mm512_loadu_si512(row_subs + 64 * 1);
        row_third_subs_vec.zmm = _mm512_loadu_si512(row_subs + 64 * 2);
        row_fourth_subs_vec.zmm = _mm512_loadu_si512(row_subs + 64 * 3);

        // In the serial version we have one forward pass, that computes the deletion,
        // insertion, and substitution costs at once.
        //    for (sz_size_t idx_longer = 0; idx_longer < longer_length; ++idx_longer) {
        //        sz_ssize_t cost_deletion = previous_distances[idx_longer + 1] + gap;
        //        sz_ssize_t cost_insertion = current_distances[idx_longer] + gap;
        //        sz_ssize_t cost_substitution = previous_distances[idx_longer] +
        //        row_subs[longer_unsigned[idx_longer]]; current_distances[idx_longer + 1] =
        //        sz_min_of_three(cost_deletion, cost_insertion, cost_substitution);
        //    }
        //
        // Given the complexity of handling the data-dependency between consecutive insertion cost computations
        // within a Levenshtein matrix, the simplest design would be to vectorize every kind of cost computation
        // separately.
        //      1. Compute substitution costs for up to 64 characters at once, upcasting from 8-bit integers to 32.
        //      2. Compute the pairwise minimum with deletion costs.
        //      3. Inclusive prefix minimum computation to combine with addition costs.
        // Proceeding with substitutions:
        for (sz_size_t idx_longer = 0; idx_longer < longer_length; idx_longer += 64) {
            sz_size_t register_length = sz_min_of_two(longer_length - idx_longer, 64);
            __mmask64 mask = _sz_u64_mask_until(register_length);
            longer_vec.zmm = _mm512_maskz_loadu_epi8(mask, longer + idx_longer);

            // Blend the `row_(first|second|third|fourth)_subs_vec` into `current_vec`, picking the right source
            // for every character in `longer_vec`. Before that, we need to permute the subsititution vectors.
            // Only the bottom 6 bits of a byte are used in VPERB, so we don't even need to mask.
            shuffled_first_subs_vec.zmm = _mm512_maskz_permutexvar_epi8(mask, longer_vec.zmm, row_first_subs_vec.zmm);
            shuffled_second_subs_vec.zmm = _mm512_maskz_permutexvar_epi8(mask, longer_vec.zmm, row_second_subs_vec.zmm);
            shuffled_third_subs_vec.zmm = _mm512_maskz_permutexvar_epi8(mask, longer_vec.zmm, row_third_subs_vec.zmm);
            shuffled_fourth_subs_vec.zmm = _mm512_maskz_permutexvar_epi8(mask, longer_vec.zmm, row_fourth_subs_vec.zmm);

            // To blend we can invoke three `_mm512_cmplt_epu8_mask`, but we can also achieve the same using
            // the AND logical operation, checking the top two bits of every byte.
            // Continuing this thought, we can use the VPTESTMB instruction to output the mask after the AND.
            __mmask64 is_third_or_fourth = _mm512_mask_test_epi8_mask(mask, longer_vec.zmm, is_third_or_fourth_vec.zmm);
            __mmask64 is_second_or_fourth =
                _mm512_mask_test_epi8_mask(mask, longer_vec.zmm, is_second_or_fourth_vec.zmm);
            lookup_substitution_vec.zmm = _mm512_mask_blend_epi8(
                is_third_or_fourth,
                // Choose between the first and the second.
                _mm512_mask_blend_epi8(is_second_or_fourth, shuffled_first_subs_vec.zmm, shuffled_second_subs_vec.zmm),
                // Choose between the third and the fourth.
                _mm512_mask_blend_epi8(is_second_or_fourth, shuffled_third_subs_vec.zmm, shuffled_fourth_subs_vec.zmm));

            // First, sign-extend lower and upper 16 bytes to 16-bit integers.
            __m512i current_0_31_vec = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(lookup_substitution_vec.zmm, 0));
            __m512i current_32_63_vec = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(lookup_substitution_vec.zmm, 1));

            // Now extend those 16-bit integers to 32-bit.
            // This isn't free, same as the subsequent store, so we only want to do that for the populated lanes.
            // To minimize the number of loads and stores, we can combine our substitution costs with the previous
            // distances, containing the deletion costs.
            {
                cost_substitution_vec.zmm = _mm512_maskz_loadu_epi32((__mmask16)mask, previous_distances + idx_longer);
                cost_substitution_vec.zmm = _mm512_add_epi32(
                    cost_substitution_vec.zmm, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(current_0_31_vec, 0)));
                cost_deletion_vec.zmm = _mm512_maskz_loadu_epi32((__mmask16)mask, previous_distances + 1 + idx_longer);
                cost_deletion_vec.zmm = _mm512_add_epi32(cost_deletion_vec.zmm, gap_vec.zmm);
                current_vec.zmm = _mm512_max_epi32(cost_substitution_vec.zmm, cost_deletion_vec.zmm);

                // Inclusive prefix minimum computation to combine with insertion costs.
                // Simply disabling this operation results in 5x performance improvement, meaning
                // that this operation is responsible for 80% of the total runtime.
                //    for (sz_size_t idx_longer = 0; idx_longer < longer_length; ++idx_longer) {
                //        current_distances[idx_longer + 1] =
                //            sz_max_of_two(current_distances[idx_longer] + gap, current_distances[idx_longer + 1]);
                //    }
                //
                // To perform the same operation in vectorized form, we need to perform a tree-like reduction,
                // that will involve multiple steps. It's quite expensive and should be first tested in the
                // "experimental" section.
                //
                // Another approach might be loop unrolling:
                //      current_vec.i32s[0] = last_in_row = sz_i32_max_of_two(current_vec.i32s[0], last_in_row +
                //      gap); current_vec.i32s[1] = last_in_row = sz_i32_max_of_two(current_vec.i32s[1], last_in_row
                //      + gap); current_vec.i32s[2] = last_in_row = sz_i32_max_of_two(current_vec.i32s[2],
                //      last_in_row + gap);
                //      ... yet this approach is also quite expensive.
                for (int i = 0; i != 16; ++i)
                    current_vec.i32s[i] = last_in_row = sz_max_of_two(current_vec.i32s[i], last_in_row + gap);
                _mm512_mask_storeu_epi32(current_distances + idx_longer + 1, (__mmask16)mask, current_vec.zmm);
            }

            // Export the values from 16 to 31.
            if (register_length > 16) {
                mask = _kshiftri_mask64(mask, 16);
                cost_substitution_vec.zmm =
                    _mm512_maskz_loadu_epi32((__mmask16)mask, previous_distances + idx_longer + 16);
                cost_substitution_vec.zmm = _mm512_add_epi32(
                    cost_substitution_vec.zmm, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(current_0_31_vec, 1)));
                cost_deletion_vec.zmm =
                    _mm512_maskz_loadu_epi32((__mmask16)mask, previous_distances + 1 + idx_longer + 16);
                cost_deletion_vec.zmm = _mm512_add_epi32(cost_deletion_vec.zmm, gap_vec.zmm);
                current_vec.zmm = _mm512_max_epi32(cost_substitution_vec.zmm, cost_deletion_vec.zmm);

                // Aggregate running insertion costs within the register.
                for (int i = 0; i != 16; ++i)
                    current_vec.i32s[i] = last_in_row = sz_max_of_two(current_vec.i32s[i], last_in_row + gap);
                _mm512_mask_storeu_epi32(current_distances + idx_longer + 1 + 16, (__mmask16)mask, current_vec.zmm);
            }

            // Export the values from 32 to 47.
            if (register_length > 32) {
                mask = _kshiftri_mask64(mask, 16);
                cost_substitution_vec.zmm =
                    _mm512_maskz_loadu_epi32((__mmask16)mask, previous_distances + idx_longer + 32);
                cost_substitution_vec.zmm = _mm512_add_epi32(
                    cost_substitution_vec.zmm, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(current_32_63_vec, 0)));
                cost_deletion_vec.zmm =
                    _mm512_maskz_loadu_epi32((__mmask16)mask, previous_distances + 1 + idx_longer + 32);
                cost_deletion_vec.zmm = _mm512_add_epi32(cost_deletion_vec.zmm, gap_vec.zmm);
                current_vec.zmm = _mm512_max_epi32(cost_substitution_vec.zmm, cost_deletion_vec.zmm);

                // Aggregate running insertion costs within the register.
                for (int i = 0; i != 16; ++i)
                    current_vec.i32s[i] = last_in_row = sz_max_of_two(current_vec.i32s[i], last_in_row + gap);
                _mm512_mask_storeu_epi32(current_distances + idx_longer + 1 + 32, (__mmask16)mask, current_vec.zmm);
            }

            // Export the values from 32 to 47.
            if (register_length > 48) {
                mask = _kshiftri_mask64(mask, 16);
                cost_substitution_vec.zmm =
                    _mm512_maskz_loadu_epi32((__mmask16)mask, previous_distances + idx_longer + 48);
                cost_substitution_vec.zmm = _mm512_add_epi32(
                    cost_substitution_vec.zmm, _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(current_32_63_vec, 1)));
                cost_deletion_vec.zmm =
                    _mm512_maskz_loadu_epi32((__mmask16)mask, previous_distances + 1 + idx_longer + 48);
                cost_deletion_vec.zmm = _mm512_add_epi32(cost_deletion_vec.zmm, gap_vec.zmm);
                current_vec.zmm = _mm512_max_epi32(cost_substitution_vec.zmm, cost_deletion_vec.zmm);

                // Aggregate running insertion costs within the register.
                for (int i = 0; i != 16; ++i)
                    current_vec.i32s[i] = last_in_row = sz_max_of_two(current_vec.i32s[i], last_in_row + gap);
                _mm512_mask_storeu_epi32(current_distances + idx_longer + 1 + 48, (__mmask16)mask, current_vec.zmm);
            }
        }

        // Swap previous_distances and current_distances pointers
        sz_pointer_swap((void **)&previous_distances, (void **)&current_distances);
    }

    // Cache scalar before `free` call.
    sz_ssize_t result = previous_distances[longer_length];
    alloc->free(distances, buffer_length, alloc->handle);
    return result;
}

SZ_INTERNAL sz_ssize_t sz_alignment_score_ice(   //
    sz_cptr_t shorter, sz_size_t shorter_length, //
    sz_cptr_t longer, sz_size_t longer_length,   //
    sz_error_cost_t const *subs, sz_error_cost_t gap, sz_memory_allocator_t *alloc) {

    if (sz_max_of_two(shorter_length, longer_length) < (256ull * 256ull * 256ull))
        return _sz_alignment_score_wagner_fisher_upto17m_ice(shorter, shorter_length, longer, longer_length, subs, gap,
                                                             alloc);
    else
        return sz_alignment_score_serial(shorter, shorter_length, longer, longer_length, subs, gap, alloc);
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_ICE
#pragma endregion // Ice Lake Implementation

/*  Implementation of the similarity algorithms using the Arm NEON instruction set, available on 64-bit
 *  Arm processors. Covers billions of mobile CPUs worldwide, including Apple's A-series, and Qualcomm's Snapdragon.
 */
#pragma region NEON Implementation
#if SZ_USE_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_NEON
#pragma endregion // NEON Implementation

/*  Implementation of the string search algorithms using the Arm SVE variable-length registers,
 *  available in Arm v9 processors, like in Apple M4+ and Graviton 3+ CPUs.
 */
#pragma region SVE Implementation
#if SZ_USE_SVE
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_SVE
#pragma endregion // SVE Implementation

/*  Pick the right implementation for the string search algorithms.
 *  To override this behavior and precompile all backends - set `SZ_DYNAMIC_DISPATCH` to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_size_t sz_hamming_distance( //
    sz_cptr_t a, sz_size_t a_length,      //
    sz_cptr_t b, sz_size_t b_length,      //
    sz_size_t bound) {
    return sz_hamming_distance_serial(a, a_length, b, b_length, bound);
}

SZ_DYNAMIC sz_size_t sz_hamming_distance_utf8( //
    sz_cptr_t a, sz_size_t a_length,           //
    sz_cptr_t b, sz_size_t b_length,           //
    sz_size_t bound) {
    return sz_hamming_distance_utf8_serial(a, a_length, b, b_length, bound);
}

SZ_DYNAMIC sz_size_t sz_edit_distance( //
    sz_cptr_t a, sz_size_t a_length,   //
    sz_cptr_t b, sz_size_t b_length,   //
    sz_size_t bound, sz_memory_allocator_t *alloc) {
#if SZ_USE_ICE
    return sz_edit_distance_ice(a, a_length, b, b_length, bound, alloc);
#else
    return sz_edit_distance_serial(a, a_length, b, b_length, bound, alloc);
#endif
}

SZ_DYNAMIC sz_size_t sz_edit_distance_utf8( //
    sz_cptr_t a, sz_size_t a_length,        //
    sz_cptr_t b, sz_size_t b_length,        //
    sz_size_t bound, sz_memory_allocator_t *alloc) {
    return _sz_edit_distance_wagner_fisher_serial(a, a_length, b, b_length, bound, sz_true_k, alloc);
}

SZ_DYNAMIC sz_ssize_t sz_alignment_score( //
    sz_cptr_t a, sz_size_t a_length,      //
    sz_cptr_t b, sz_size_t b_length,      //
    sz_error_cost_t const *subs, sz_error_cost_t gap, sz_memory_allocator_t *alloc) {
#if SZ_USE_ICE
    return sz_alignment_score_ice(a, a_length, b, b_length, subs, gap, alloc);
#else
    return sz_alignment_score_serial(a, a_length, b, b_length, subs, gap, alloc);
#endif
}

#endif            // !SZ_DYNAMIC_DISPATCH
#pragma endregion // Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_SIMISLARITY_H_
