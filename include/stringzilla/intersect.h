/**
 *  @brief Hardware-accelerated string collection intersections for JOIN-like DBMS operations.
 *  @file include/stringzilla/intersect.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs for `sz_sequence_t` string collections with hardware-specific backends:
 *
 *  - `sz_sequence_intersect` - to compute the strict (distinct-set) intersection of two string collections,
 *    tolerating duplicates within either side and emitting each shared value once.
 *  - TODO: `sz_sequence_join` - to compute the full join (all matching pairs) of two string collections.
 */
#ifndef STRINGZILLA_INTERSECT_H_
#define STRINGZILLA_INTERSECT_H_

#include "stringzilla/types.h"

#include "stringzilla/compare.h" // `sz_compare`
#include "stringzilla/memory.h"  // `sz_fill`
#include "stringzilla/hash.h"    // `sz_hash`

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief The @b power-of-two memory-usage budget @b multiple for the hash table.
 *
 *  The behaviour of hashing-based approaches can often be tuned with different "hyper-parameter" values.
 *  For "unordered set intersections" implemented here, the @p budget argument controls the balance between
 *  throughput and memory usage. The higher the budget, the more memory is used, but the fewer collisions
 *  will be observed
 */
#if !defined(SZ_SEQUENCE_INTERSECT_BUDGET)
#define SZ_SEQUENCE_INTERSECT_BUDGET (1)
#endif

#pragma region Core API

/**
 *  @brief Intersects two binary @b string sequences, using a hash table.
 *         Outputs the @p first_positions from the @p first_sequence and @p second_positions from
 *         the @p second_sequence, that contain matched strings. Missing matches are represented as `SZ_SIZE_MAX`.
 *
 *  @param first_sequence First immutable sequence of strings to intersection.
 *  @param second_sequence Second immutable sequence of strings to intersection.
 *  @param semantics JOIN semantics for the intersection, including handling of duplicates.
 *  @param alloc Optional memory allocator for temporary storage.
 *  @param seed Optional seed for the hash table to avoid attacks.
 *  @param intersection_size Number of matching strings in both sequences.
 *  @param first_positions Offset positions of the matching strings from the @p first_sequence.
 *  @param second_positions Offset positions of the matching strings from the @p second_sequence.
 *
 *  @retval `sz_success_k` if the operation was successful.
 *  @retval `sz_bad_alloc_k` if the operation failed due to memory allocation failure.
 *  @pre The @p first_positions array must fit at least `min(first_sequence->count, second_sequence->count)` items.
 *  @pre The @p second_positions array must fit at least `min(first_sequence->count, second_sequence->count)` items.
 *  @note Tolerates duplicate strings within either sequence: each distinct shared value is emitted exactly
 *        once (a distinct-set intersection), so `intersection_size` never exceeds
 *        `min(first_sequence->count, second_sequence->count)` and can't overflow the output arrays.
 *
 *  Example usage:
 *
 *  @code{.c}
 *      #include <stringzilla/intersect.h>
 *      int main() {
 *          char const *first[] = {"banana", "apple", "cherry"};
 *          char const *second[] = {"cherry", "orange", "pineapple", "banana"};
 *          sz_sequence_t first_sequence, second_sequence;
 *          sz_sequence_from_null_terminated_strings(first, 3, &first_sequence);
 *          sz_sequence_from_null_terminated_strings(second, 4, &second_sequence);
 *          sz_size_t intersection_size;
 *          sz_sorted_idx_t first_positions[3], second_positions[3]; //? 3 is the size of the smaller sequence
 *          sz_status_t status = sz_sequence_intersect(&first_sequence, &second_sequence,
 *              sz_join_inner_strict_k, NULL, 0,
 *              &intersection_size, first_positions, second_positions);
 *          return status == sz_success_k && intersection_size == 2 ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note The algorithm has linear memory complexity and linear time complexity.
 *  @see https://en.wikipedia.org/wiki/Join_(SQL)
 *
 *  @note Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa sz_sequence_intersect_serial, sz_sequence_intersect_icelake, sz_sequence_intersect_sve
 */
SZ_API_RUNTIME sz_status_t sz_sequence_intersect(sz_sequence_t const *first_sequence,
                                                 sz_sequence_t const *second_sequence, sz_memory_allocator_t *alloc,
                                                 sz_u64_t seed, sz_size_t *intersection_size,
                                                 sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions);

/**
 *  @brief Defines various JOIN semantics for string sequences, including handling of duplicates.
 *  @sa sz_join_inner_strict_k, sz_join_inner_k, sz_join_left_outer_k, sz_join_right_outer_k, sz_join_full_outer_k,
 *      sz_join_cross_k
 */
typedef enum {
    /**
     *  @brief Strict inner join with uniqueness enforcement.
     *
     *  In this mode, only unique matching strings from both sequences are returned.
     *  If either sequence contains duplicate strings, the operation will fail.
     *
     *  Example:
     *  - Input:
     *      first_sequence: { "apple", "banana", "cherry" }
     *      second_sequence: { "banana", "cherry", "date" }
     *  - Output:
     *      Result: { ("banana", "banana"), ("cherry", "cherry") }
     *
     *  SQL equivalent:
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
     *  Example:
     *  - Input:
     *      first_sequence: { "apple", "banana", "banana" }
     *      second_sequence: { "banana", "banana", "cherry" }
     *  - Output:
     *      Result: { ("banana", "banana"), ("banana", "banana"),
     *                ("banana", "banana"), ("banana", "banana") }
     *      (2 occurrences of "banana" in the first sequence × 2 in the second = 4 pairs)
     *
     *  SQL equivalent:
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
     *  Example:
     *  - Input:
     *      first_sequence: { "apple", "banana", "cherry" }
     *      second_sequence: { "banana", "cherry", "date" }
     *  - Output:
     *      Result: { ("apple", NULL), ("banana", "banana"), ("cherry", "cherry") }
     *
     *  SQL equivalent:
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
     *  Example:
     *  - Input:
     *      first_sequence: { "apple", "banana" }
     *      second_sequence: { "banana", "cherry", "date" }
     *  - Output:
     *      Result: { ("banana", "banana"), (NULL, "cherry"), (NULL, "date") }
     *
     *  SQL equivalent:
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
     *  Example:
     *  - Input:
     *      first_sequence: { "apple", "banana" }
     *      second_sequence: { "banana", "cherry" }
     *  - Output:
     *      Result: { ("apple", NULL), ("banana", "banana"), (NULL, "cherry") }
     *
     *  SQL equivalent:
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
     *  This mode returns the Cartesian product of both sequences, pairing every string in the first sequence
     *  with every string in the second sequence regardless of any matching condition.
     *
     *  Example:
     *  - Input:
     *      first_sequence: { "apple", "banana" }
     *      second_sequence: { "cherry", "date" }
     *  - Output:
     *      Result: { ("apple", "cherry"), ("apple", "date"),
     *                ("banana", "cherry"), ("banana", "date") }
     *
     *  SQL equivalent:
     *  @code{.sql}
     *  SELECT a.*, b.*
     *  FROM first_sequence a, second_sequence b;
     *  @endcode
     */
    sz_join_cross_k = 5,
} sz_sequence_join_semantics_t;

/** @copydoc sz_sequence_intersect */
SZ_API_COMPTIME sz_status_t sz_sequence_intersect_serial(                      //
    sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence, //
    sz_memory_allocator_t *alloc, sz_u64_t seed, sz_size_t *intersection_size, //
    sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions);

#if SZ_USE_ICELAKE

/** @copydoc sz_sequence_intersect */
SZ_API_COMPTIME sz_status_t sz_sequence_intersect_icelake(                     //
    sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence, //
    sz_memory_allocator_t *alloc, sz_u64_t seed, sz_size_t *intersection_size, //
    sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions);

#endif

#if SZ_USE_SVE

/** @copydoc sz_sequence_intersect */
SZ_API_COMPTIME sz_status_t sz_sequence_intersect_sve(                         //
    sz_sequence_t const *first_sequence, sz_sequence_t const *second_sequence, //
    sz_memory_allocator_t *alloc, sz_u64_t seed, sz_size_t *intersection_size, //
    sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions);

#endif

#pragma endregion

#include "stringzilla/intersect/serial.h"
#include "stringzilla/intersect/icelake.h"
#include "stringzilla/intersect/sve.h"

/*  Pick the right implementation for the string search algorithms.
 *  To override this behavior and precompile all backends - set `SZ_DYNAMIC_DISPATCH` to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_API_RUNTIME sz_status_t sz_sequence_intersect(sz_sequence_t const *first_sequence,
                                                 sz_sequence_t const *second_sequence, sz_memory_allocator_t *alloc,
                                                 sz_u64_t seed, sz_size_t *intersection_size,
                                                 sz_sorted_idx_t *first_positions, sz_sorted_idx_t *second_positions) {
#if SZ_USE_ICELAKE
    return sz_sequence_intersect_icelake( //
        first_sequence, second_sequence,  //
        alloc, seed, intersection_size,   //
        first_positions, second_positions);
#elif SZ_USE_SVE
    return sz_sequence_intersect_sve(    //
        first_sequence, second_sequence, //
        alloc, seed, intersection_size,  //
        first_positions, second_positions);
#else
    return sz_sequence_intersect_serial( //
        first_sequence, second_sequence, //
        alloc, seed, intersection_size,  //
        first_positions, second_positions);
#endif
}

#endif            // !SZ_DYNAMIC_DISPATCH
#pragma endregion // Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_INTERSECT_H_
