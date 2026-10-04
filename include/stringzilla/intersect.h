/**
 *  @file include/stringzilla/intersect.h
 *  @author Ash Vardanian
 *  @date March 7, 2025
 *  @brief Hardware-accelerated string collection intersections for JOIN-like DBMS operations.
 *
 *  Includes core APIs for @c sz_sequence_t string collections with hardware-specific backends:
 *
 *  - @c sz_sequence_intersect_best - to compute the strict, distinct-set intersection of two string
 *    collections, tolerating duplicates within either side and emitting each shared value once.
 *  - TODO: @c sz_sequence_join - to compute the full join, all matching pairs, of two collections.
 */
#ifndef STRINGZILLA_INTERSECT_H_
#define STRINGZILLA_INTERSECT_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h"     // `sz_capability_t`
#include "stringzilla/intersect/serial.h" // `STRINGZILLA_SEQUENCE_INTERSECT_BUDGET`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Intersects two binary @b string sequences, using a hash table.
 *
 *  Outputs the @p first_positions from the @p first_sequence and @p second_positions from the
 *  @p second_sequence, that contain matched strings. Missing matches are represented as
 *  @c STRINGZILLA_SIZE_MAX. Tolerates duplicate strings within either sequence: each distinct
 *  shared value is emitted exactly once, a distinct-set intersection, so @p intersection_count
 *  never exceeds the smaller of the two sequence counts and can't overflow the output arrays.
 *
 *  @param[in] first_sequence First immutable sequence of strings to intersect.
 *  @param[in] second_sequence Second immutable sequence of strings to intersect.
 *  @param[in] allocator Optional memory allocator for temporary storage.
 *  @param[in] seed Optional seed for the hash table to avoid attacks.
 *  @param[out] intersection_count Number of matching strings in both sequences.
 *  @param[out] first_positions Offset positions of the matching strings from the @p first_sequence.
 *  @param[out] second_positions Offset positions of the matching strings from @p second_sequence.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k on success, @c sz_bad_alloc_k if memory allocation failed, or
 *      @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *  @pre The @p first_positions array must fit as many items as the smaller sequence holds.
 *  @pre The @p second_positions array must fit as many items as the smaller sequence holds.
 *
 *  For example, intersecting two small collections of fruit names:
 *
 *  @code{.c}
 *      #include <stringzilla/stringzilla.h>
 *      int main() {
 *          char const *first[] = {"banana", "apple", "cherry"};
 *          char const *second[] = {"cherry", "orange", "pineapple", "banana"};
 *          sz_sequence_t first_sequence, second_sequence;
 *          sz_sequence_from_null_terminated_strings(first, 3, &first_sequence);
 *          sz_sequence_from_null_terminated_strings(second, 4, &second_sequence);
 *          sz_capability_t capabilities;
 *          sz_capabilities_enabled_cpu(&capabilities);
 *          sz_size_t intersection_count;
 *          sz_sorted_idx_t first_positions[3], second_positions[3]; //? 3 is the size of the smaller sequence
 *          sz_status_t status = sz_sequence_intersect_best(&first_sequence, &second_sequence, NULL, 0,
 *              &intersection_count, first_positions, second_positions, capabilities, NULL);
 *          return status == sz_success_k && intersection_count == 2 ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note The algorithm has linear memory complexity and linear time complexity.
 *  @see SQL joins: https://en.wikipedia.org/wiki/Join_(SQL)
 *
 *  @sa sz_sequence_intersect_serial, sz_sequence_intersect_westmere, sz_sequence_intersect_icelake,
 *      sz_sequence_intersect_neonaes
 */
STRINGZILLA_API sz_status_t sz_sequence_intersect_best(                        //
    sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence, //
    sz_allocator_t *allocator, sz_u64_t seed, sz_size_t *intersection_count,   //
    sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions,       //
    sz_capability_t capabilities, sz_stream_t stream);

/**
 *  @brief Defines various JOIN semantics for string sequences, including handling of duplicates.
 *  @sa sz_join_inner_strict_k, sz_join_inner_k, sz_join_left_outer_k, sz_join_right_outer_k,
 *      sz_join_full_outer_k, sz_join_cross_k
 */
typedef enum {

    /**
     *  @brief Strict inner join with uniqueness enforcement.
     *
     *  In this mode, only unique matching strings from both sequences are returned.
     *  If either sequence contains duplicate strings, the operation will fail.
     *
     *  For example, joining these two sequences:
     *
     *  @verbatim
     *  first_sequence:  { "apple", "banana", "cherry" }
     *  second_sequence: { "banana", "cherry", "date" }
     *  result:          { ("banana", "banana"), ("cherry", "cherry") }
     *  @endverbatim
     *
     *  The SQL equivalent:
     *
     *  @code{.sql}
     *  -- Returns unique matching rows only.
     *  SELECT DISTINCT a.*
     *  FROM first_sequence a
     *  INNER JOIN second_sequence b ON a.string = b.string;
     *  @endcode
     */
    sz_join_inner_strict_k = 0,

    /**
     *  @brief Conventional inner join allowing duplicate entries.
     *
     *  This mode returns all pairs of matching strings from both sequences.
     *  Each occurrence in the first sequence is paired with every matching occurrence
     *  in the second sequence. Order stability is not guaranteed.
     *
     *  For example, joining these two sequences:
     *
     *  @verbatim
     *  first_sequence:  { "apple", "banana", "banana" }
     *  second_sequence: { "banana", "banana", "cherry" }
     *  result:          { ("banana", "banana"), ("banana", "banana"),
     *                     ("banana", "banana"), ("banana", "banana") }
     *  @endverbatim
     *
     *  Two occurrences of "banana" in the first sequence × two in the second give four pairs.
     *
     *  The SQL equivalent:
     *
     *  @code{.sql}
     *  SELECT a.*, b.*
     *  FROM first_sequence a
     *  INNER JOIN second_sequence b ON a.string = b.string;
     *  @endcode
     */
    sz_join_inner_k = 1,

    /**
     *  @brief Left outer join preserving all entries from the first sequence.
     *
     *  This mode returns every string from the first sequence along with matching strings
     *  from the second sequence. If no match is found for an element in the first sequence,
     *  the corresponding output for the second sequence is NULL (or its equivalent).
     *
     *  For example, joining these two sequences:
     *
     *  @verbatim
     *  first_sequence:  { "apple", "banana", "cherry" }
     *  second_sequence: { "banana", "cherry", "date" }
     *  result:          { ("apple", NULL), ("banana", "banana"), ("cherry", "cherry") }
     *  @endverbatim
     *
     *  The SQL equivalent:
     *
     *  @code{.sql}
     *  SELECT a.*, b.*
     *  FROM first_sequence a
     *  LEFT OUTER JOIN second_sequence b ON a.string = b.string;
     *  @endcode
     */
    sz_join_left_outer_k = 2,

    /**
     *  @brief Right outer join preserving all entries from the second sequence.
     *
     *  This mode returns every string from the second sequence along with matching strings
     *  from the first sequence. If no match is found for an element in the second sequence,
     *  the corresponding output for the first sequence is NULL (or its equivalent).
     *
     *  For example, joining these two sequences:
     *
     *  @verbatim
     *  first_sequence:  { "apple", "banana" }
     *  second_sequence: { "banana", "cherry", "date" }
     *  result:          { ("banana", "banana"), (NULL, "cherry"), (NULL, "date") }
     *  @endverbatim
     *
     *  The SQL equivalent:
     *
     *  @code{.sql}
     *  SELECT a.*, b.*
     *  FROM first_sequence a
     *  RIGHT OUTER JOIN second_sequence b ON a.string = b.string;
     *  @endcode
     */
    sz_join_right_outer_k = 3,

    /**
     *  @brief Full outer join combining all entries from both sequences.
     *
     *  This mode returns all matching pairs along with unmatched strings from both sequences.
     *  For unmatched strings, the corresponding result from the other sequence is NULL.
     *
     *  For example, joining these two sequences:
     *
     *  @verbatim
     *  first_sequence:  { "apple", "banana" }
     *  second_sequence: { "banana", "cherry" }
     *  result:          { ("apple", NULL), ("banana", "banana"), (NULL, "cherry") }
     *  @endverbatim
     *
     *  The SQL equivalent:
     *
     *  @code{.sql}
     *  SELECT a.*, b.*
     *  FROM first_sequence a
     *  FULL OUTER JOIN second_sequence b ON a.string = b.string;
     *  @endcode
     */
    sz_join_full_outer_k = 4,

    /**
     *  @brief Cross join (Cartesian product) of two sequences.
     *
     *  This mode returns the Cartesian product of both sequences, pairing every string in the first
     *  sequence with every string in the second sequence regardless of any matching condition.
     *
     *  For example, joining these two sequences:
     *
     *  @verbatim
     *  first_sequence:  { "apple", "banana" }
     *  second_sequence: { "cherry", "date" }
     *  result:          { ("apple", "cherry"), ("apple", "date"),
     *                     ("banana", "cherry"), ("banana", "date") }
     *  @endverbatim
     *
     *  The SQL equivalent:
     *
     *  @code{.sql}
     *  SELECT a.*, b.*
     *  FROM first_sequence a, second_sequence b;
     *  @endcode
     */
    sz_join_cross_k = 5,
} sz_sequence_join_semantics_t;

/** @copydoc sz_sequence_intersect_best */
STRINGZILLA_API sz_status_t sz_sequence_intersect_serial(                      //
    sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence, //
    sz_allocator_t *allocator, sz_u64_t seed, sz_size_t *intersection_count,   //
    sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions, sz_stream_t stream);

#if STRINGZILLA_TARGET_WESTMERE

/** @copydoc sz_sequence_intersect_best */
STRINGZILLA_API sz_status_t sz_sequence_intersect_westmere(                    //
    sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence, //
    sz_allocator_t *allocator, sz_u64_t seed, sz_size_t *intersection_count,   //
    sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions, sz_stream_t stream);

#endif

#if STRINGZILLA_TARGET_ICELAKE

/** @copydoc sz_sequence_intersect_best */
STRINGZILLA_API sz_status_t sz_sequence_intersect_icelake(                     //
    sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence, //
    sz_allocator_t *allocator, sz_u64_t seed, sz_size_t *intersection_count,   //
    sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions, sz_stream_t stream);

#endif

#if STRINGZILLA_TARGET_NEONAES

/** @copydoc sz_sequence_intersect_best */
STRINGZILLA_API sz_status_t sz_sequence_intersect_neonaes(                     //
    sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence, //
    sz_allocator_t *allocator, sz_u64_t seed, sz_size_t *intersection_count,   //
    sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions, sz_stream_t stream);

#endif

/**
 *  @brief Finds the intersection kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_intersect_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                     sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion

#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/intersect/westmere.h"
#include "stringzilla/intersect/icelake.h"
#include "stringzilla/intersect/neonaes.h"
#endif // STRINGZILLA_HEADER_ONLY

#pragma region Dispatch

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_sequence_intersect_best(                        //
    sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence, //
    sz_allocator_t *allocator, sz_u64_t seed, sz_size_t *intersection_count,   //
    sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions,       //
    sz_capability_t capabilities, sz_stream_t stream) {
    sz_unused_(first_sequence), sz_unused_(second_sequence), sz_unused_(allocator), sz_unused_(seed),
        sz_unused_(intersection_count), sz_unused_(first_positions), sz_unused_(second_positions),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_intersect_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                     sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY
#pragma endregion Dispatch

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_INTERSECT_H_
