/**
 *  @brief Hardware-accelerated string collection sorting.
 *  @file include/stringzilla/sort.h
 *  @author Ash Vardanian
 *
 *  Provides the @b `sz_sequence_argsort` API to get the sorting permutation of `sz_sequence_t` binary
 *  string collections in lexicographical order.
 *
 *  The core idea of all following string algorithms is to process strings not based on 1 character at a time,
 *  but on a larger "Pointer-sized N-grams" fitting in 4 or 8 bytes at once, on 32-bit or 64-bit architectures,
 *  respectively. In reality we may not use the full pointer size, but only a few bytes from it, and keep the
 *  rest for some metadata.
 *
 *  That, however, means, that unsigned integer sorting is a constituent part of our sequence algorithms.
 *  The per-backend `sz_pgrams_sort_serial`/`_skylake`/`_sve` helpers expose that integer-sort core for
 *  direct benchmarking, but it is an internal building block - not a runtime-dispatched public API.
 *
 *  Beyond plain byte-lexicographic ordering, `sz_sequence_argsort_utf8_uncased` sorts UTF-8
 *  strings under Unicode case-folding, progressively folding small chunks of each string on the fly so
 *  callers don't have to materialize a pre-folded copy of the whole collection. Malformed UTF-8 is
 *  well-defined: a byte that does not begin a well-formed codepoint sorts by its raw byte value as a
 *  single one-byte unit, keeping the order total and deterministic.
 *
 *  All `sz_sequence_argsort*` entry points are @b stable (equal elements keep their input order), support
 *  descending order via the `reverse` flag, and accept a `top_count` to only fully order the leading
 *  `top_count` elements - a partial-sort / top-K mode that prunes work on the unwanted tail.
 *
 *  Other helpers include:
 *
 *  - `sz_pgrams_sort_with_insertion` - for quadratic-complexity sorting of small continuous integer arrays.
 *  - `sz_sequence_argsort_with_insertion` - for quadratic-complexity sorting of small string collections.
 */
#ifndef STRINGZILLA_SORT_H_
#define STRINGZILLA_SORT_H_

#include "stringzilla/types.h"

#include "stringzilla/compare.h" // `sz_compare`
#include "stringzilla/memory.h"  // `sz_copy`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Faster @b stable arg-sort for an arbitrary @b string sequence, using QuickSort.
 *         Outputs the @p order of elements in the immutable @p sequence, that would sort it.
 *
 *  @param sequence Immutable sequence of strings to sort.
 *  @param alloc Optional memory allocator for temporary storage.
 *  @param order Output permutation that sorts the elements.
 *  @param top_count Number of leading elements to fully order, or 0 to sort the whole sequence.
 *  @param reverse Whether to sort in descending order.
 *
 *  @retval `sz_success_k` if the operation was successful.
 *  @retval `sz_bad_alloc_k` if the operation failed due to memory allocation failure.
 *  @pre The @p order array must fit at least `sequence->count` integers.
 *  @post The @p order array will contain a valid permutation of `[0, sequence->count - 1]`.
 *  @post If `top_count` is non-zero and smaller than the count, only `order[0, top_count)` are sorted;
 *        the remaining entries are an arbitrary permutation of the leftover indices.
 *
 *  Example usage:
 *
 *  @code{.c}
 *      #include <stringzilla/sort.h>
 *      int main() {
 *          char const *strings[] = {"banana", "apple", "cherry"};
 *          sz_sequence_t sequence;
 *          sz_sequence_from_null_terminated_strings(strings, 3, &sequence);
 *          sz_sorted_idx_t order[3];
 *          sz_status_t status = sz_sequence_argsort(&sequence, NULL, order, 0, sz_false_k);
 *          return status == sz_success_k && order[0] == 1 && order[1] == 0 && order[2] == 2 ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note The algorithm has linear memory complexity, quadratic worst-case and log-linear average time complexity.
 *  @see https://en.wikipedia.org/wiki/Quicksort
 *
 *  @note This algorithm is @b stable: equal elements keep their relative order, ascending by original index.
 *
 *  @note Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa sz_sequence_argsort_serial, sz_sequence_argsort_skylake, sz_sequence_argsort_sve
 *  @sa sz_sequence_argsort_utf8_uncased
 */
SZ_DYNAMIC sz_status_t sz_sequence_argsort(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                           sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse);

/**
 *  @brief Faster @b stable @b uncased arg-sort for a UTF-8 @b string sequence, using QuickSort.
 *         Orders strings under Unicode case-folding, equivalent to folding every string and sorting the
 *         folded bytes lexicographically - but folding only small chunks on the fly, never materializing
 *         a fully pre-folded copy of the collection.
 *
 *  @param sequence Immutable sequence of UTF-8 strings to sort.
 *  @param alloc Optional memory allocator for temporary storage.
 *  @param order Output permutation that sorts the elements.
 *  @param top_count Number of leading elements to fully order, or 0 to sort the whole sequence.
 *  @param reverse Whether to sort in descending order.
 *
 *  @retval `sz_success_k` if the operation was successful.
 *  @retval `sz_bad_alloc_k` if the operation failed due to memory allocation failure.
 *  @pre The @p order array must fit at least `sequence->count` integers.
 *  @post The @p order array will contain a valid permutation of `[0, sequence->count - 1]`.
 *
 *  @note This algorithm is @b stable: equal (case-folded-equal) elements keep their input order.
 *  @note Malformed UTF-8 is handled losslessly: any byte that does not begin a well-formed codepoint is
 *        compared and sorted by its raw byte value as a single one-byte unit, and decoding resyncs at the
 *        next byte. The result stays a total, deterministic order; valid input sorts byte-identically.
 *  @sa sz_utf8_uncased_fold, sz_utf8_uncased_order
 *  @sa sz_sequence_argsort_utf8_uncased_serial
 */
SZ_DYNAMIC sz_status_t sz_sequence_argsort_utf8_uncased( //
    sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,  //
    sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse);

/** @copydoc sz_sequence_argsort */
SZ_PUBLIC sz_status_t sz_sequence_argsort_serial(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                                 sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse);

/** @copydoc sz_sequence_argsort_utf8_uncased */
SZ_PUBLIC sz_status_t sz_sequence_argsort_utf8_uncased_serial( //
    sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,        //
    sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse);

/**
 *  @brief Internal @b inplace QuickSort for a continuous @b unsigned-integer sequence, backing the string
 *         arg-sorts. Overwrites the input @p pgrams with the sorted sequence and exports the @p order
 *         permutation. Not part of the stable, public ordering contract and not runtime-dispatched - the
 *         per-backend variants exist for direct benchmarking of the integer-sort core.
 *
 *  @param pgrams Continuous buffer of unsigned integers to sort in place.
 *  @param count Number of elements in the sequence.
 *  @param alloc Optional memory allocator for temporary storage.
 *  @param order Output permutation that sorts the elements.
 *
 *  @retval `sz_success_k` if the operation was successful.
 *  @retval `sz_bad_alloc_k` if the operation failed due to memory allocation failure.
 */
SZ_PUBLIC sz_status_t sz_pgrams_sort_serial(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                            sz_sorted_idx_t *order);

#if SZ_USE_HASWELL

/** @copydoc sz_sequence_argsort */
SZ_PUBLIC sz_status_t sz_sequence_argsort_haswell(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                                  sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse);

/** @copydoc sz_sequence_argsort_utf8_uncased */
SZ_PUBLIC sz_status_t sz_sequence_argsort_utf8_uncased_haswell( //
    sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,         //
    sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse);

/** @copydoc sz_pgrams_sort_serial */
SZ_PUBLIC sz_status_t sz_pgrams_sort_haswell(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                             sz_sorted_idx_t *order);

#endif

#if SZ_USE_SKYLAKE

/** @copydoc sz_sequence_argsort */
SZ_PUBLIC sz_status_t sz_sequence_argsort_skylake(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                                  sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse);

/** @copydoc sz_sequence_argsort_utf8_uncased */
SZ_PUBLIC sz_status_t sz_sequence_argsort_utf8_uncased_skylake( //
    sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,         //
    sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse);

/** @copydoc sz_pgrams_sort_serial */
SZ_PUBLIC sz_status_t sz_pgrams_sort_skylake(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                             sz_sorted_idx_t *order);

#endif

#if SZ_USE_SVE

/** @copydoc sz_sequence_argsort */
SZ_PUBLIC sz_status_t sz_sequence_argsort_sve(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                              sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse);

/** @copydoc sz_sequence_argsort_utf8_uncased */
SZ_PUBLIC sz_status_t sz_sequence_argsort_utf8_uncased_sve( //
    sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,     //
    sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse);

/** @copydoc sz_pgrams_sort_serial */
SZ_PUBLIC sz_status_t sz_pgrams_sort_sve(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                         sz_sorted_idx_t *order);

#endif

#if SZ_USE_NEON

/** @copydoc sz_sequence_argsort */
SZ_PUBLIC sz_status_t sz_sequence_argsort_neon(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                               sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse);

/** @copydoc sz_sequence_argsort_utf8_uncased */
SZ_PUBLIC sz_status_t sz_sequence_argsort_utf8_uncased_neon( //
    sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,      //
    sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse);

/** @copydoc sz_pgrams_sort_serial */
SZ_PUBLIC sz_status_t sz_pgrams_sort_neon(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                          sz_sorted_idx_t *order);

#endif

#if SZ_USE_RVV

/** @copydoc sz_sequence_argsort */
SZ_PUBLIC sz_status_t sz_sequence_argsort_rvv(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                              sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse);

/** @copydoc sz_sequence_argsort_utf8_uncased */
SZ_PUBLIC sz_status_t sz_sequence_argsort_utf8_uncased_rvv( //
    sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,     //
    sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse);

/** @copydoc sz_pgrams_sort_serial */
SZ_PUBLIC sz_status_t sz_pgrams_sort_rvv(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                         sz_sorted_idx_t *order);

#endif

#pragma endregion

#include "stringzilla/sort/serial.h"
#include "stringzilla/sort/haswell.h"
#include "stringzilla/sort/skylake.h"
#include "stringzilla/sort/sve.h"
#include "stringzilla/sort/neon.h"
#include "stringzilla/sort/rvv.h"

/*  Pick the right implementation for the string search algorithms.
 *  To override this behavior and precompile all backends - set `SZ_DYNAMIC_DISPATCH` to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_status_t sz_sequence_argsort(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                           sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse) {
#if SZ_USE_SKYLAKE
    return sz_sequence_argsort_skylake(sequence, alloc, order, top_count, reverse);
#elif SZ_USE_HASWELL
    return sz_sequence_argsort_haswell(sequence, alloc, order, top_count, reverse);
#elif SZ_USE_SVE
    return sz_sequence_argsort_sve(sequence, alloc, order, top_count, reverse);
#elif SZ_USE_NEON
    return sz_sequence_argsort_neon(sequence, alloc, order, top_count, reverse);
#elif SZ_USE_RVV
    return sz_sequence_argsort_rvv(sequence, alloc, order, top_count, reverse);
#else
    return sz_sequence_argsort_serial(sequence, alloc, order, top_count, reverse);
#endif
}

SZ_DYNAMIC sz_status_t sz_sequence_argsort_utf8_uncased( //
    sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,  //
    sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse) {
#if SZ_USE_SKYLAKE
    return sz_sequence_argsort_utf8_uncased_skylake(sequence, alloc, order, top_count, reverse);
#elif SZ_USE_HASWELL
    return sz_sequence_argsort_utf8_uncased_haswell(sequence, alloc, order, top_count, reverse);
#elif SZ_USE_SVE
    return sz_sequence_argsort_utf8_uncased_sve(sequence, alloc, order, top_count, reverse);
#elif SZ_USE_NEON
    return sz_sequence_argsort_utf8_uncased_neon(sequence, alloc, order, top_count, reverse);
#elif SZ_USE_RVV
    return sz_sequence_argsort_utf8_uncased_rvv(sequence, alloc, order, top_count, reverse);
#else
    return sz_sequence_argsort_utf8_uncased_serial(sequence, alloc, order, top_count, reverse);
#endif
}

#endif            // !SZ_DYNAMIC_DISPATCH
#pragma endregion // Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_SORT_H_
