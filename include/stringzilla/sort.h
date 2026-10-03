/**
 *  @file include/stringzilla/sort.h
 *  @author Ash Vardanian
 *  @date February 15, 2025
 *  @brief Hardware-accelerated string collection sorting.
 *
 *  Provides the @b sz_sequence_argsort_best API to get the sorting permutation of @c sz_sequence_t
 *  binary string collections in lexicographical order.
 *
 *  The core idea of all following string algorithms is to process strings not based on 1 character
 *  at a time, but on a larger "Pointer-sized N-grams" fitting in 4 or 8 bytes at once, on 32-bit or
 *  64-bit architectures, respectively. In reality we may not use the full pointer size, but only a
 *  few bytes from it, and keep the rest for some metadata.
 *
 *  That, however, means, that unsigned integer sorting is a constituent part of our sequence
 *  algorithms. The per-backend @c sz_pgrams_sort_serial_ helpers are that integer-sort core, an
 *  internal building block the benchmarks reach directly, with no dispatch point of its own.
 *
 *  Beyond plain byte-lexicographic ordering, @c sz_sequence_argsort_uncased_best sorts UTF-8
 *  strings under Unicode case-folding, progressively folding small chunks of each string on the fly
 *  so callers don't have to materialize a pre-folded copy of the whole collection. Malformed UTF-8
 *  is well-defined: a byte that does not begin a well-formed codepoint sorts by its raw byte value
 *  as a single one-byte unit, keeping the order total and deterministic.
 *
 *  All `sz_sequence_argsort*` entry points are @b stable, so equal elements keep their input order,
 *  support descending order via the @c reverse flag, and accept a @c top_count to only fully order
 *  the leading @c top_count elements - a partial-sort or top-K mode that prunes work on the tail
 *  the caller does not want.
 *
 *  Other helpers include:
 *
 *  - @c sz_pgrams_sort_with_insertion - quadratic-complexity sorting of small integer arrays.
 *  - @c sz_sequence_argsort_with_insertion - quadratic-complexity sorting of small string arrays.
 */
#ifndef STRINGZILLA_SORT_H_
#define STRINGZILLA_SORT_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Faster @b stable arg-sort for an arbitrary @b string sequence, using QuickSort.
 *
 *  Outputs the @p order of elements in the immutable @p sequence, that would sort it.
 *
 *  @param[in] sequence Immutable sequence of strings to sort.
 *  @param[in] top_count Number of leading elements to fully order, or 0 to sort the whole sequence.
 *  @param[in] reverse Whether to sort in descending order.
 *  @param[in] allocator Optional memory allocator for temporary storage.
 *  @param[out] order Output permutation that sorts the elements.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k on success, @c sz_bad_alloc_k if memory allocation failed, or
 *      @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *  @pre The @p order array must fit at least as many integers as the @p sequence holds.
 *  @post The @p order array will contain a valid permutation of all indices of the @p sequence.
 *  @post If @p top_count is non-zero and smaller than the count, only the first @p top_count
 *      entries are sorted; the rest are an arbitrary permutation of the leftover indices.
 *
 *  For example, sorting three fruit names:
 *
 *  @code{.c}
 *      #include <stringzilla/stringzilla.h>
 *      int main() {
 *          char const *strings[] = {"banana", "apple", "cherry"};
 *          sz_sequence_t sequence;
 *          sz_sequence_from_null_terminated_strings(strings, 3, &sequence);
 *          sz_capability_t capabilities;
 *          sz_cpu_capabilities_enabled(&capabilities);
 *          sz_sorted_idx_t order[3];
 *          sz_status_t status =
 *              sz_sequence_argsort_best(&sequence, 0, sz_false_k, NULL, order, capabilities, NULL);
 *          return status == sz_success_k && order[0] == 1 && order[1] == 0 && order[2] == 2 ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note Takes linear memory, quadratic worst-case and log-linear average time.
 *  @see Quicksort: https://en.wikipedia.org/wiki/Quicksort
 *
 *  @note This algorithm is @b stable: equal elements keep their relative order, ascending by index.
 *
 *  @sa sz_sequence_argsort_serial, sz_sequence_argsort_haswell, sz_sequence_argsort_skylake,
 *      sz_sequence_argsort_neon, sz_sequence_argsort_sve, sz_sequence_argsort_rvv
 *  @sa sz_sequence_argsort_uncased_best
 */
STRINGZILLA_API sz_status_t sz_sequence_argsort_best(sz_sequence_t const *sequence, sz_size_t top_count,
                                                     sz_bool_t reverse, sz_allocator_t *allocator,
                                                     sz_sorted_idx_t *order, sz_capability_t capabilities,
                                                     void *stream);

/**
 *  @brief Faster @b stable @b uncased arg-sort for a UTF-8 @b string sequence, using QuickSort.
 *
 *  Orders strings under Unicode case-folding, equivalent to folding every string and sorting the
 *  folded bytes lexicographically - but folding only small chunks on the fly, never materializing a
 *  fully pre-folded copy of the collection.
 *
 *  @param[in] sequence Immutable sequence of UTF-8 strings to sort.
 *  @param[in] top_count Number of leading elements to fully order, or 0 to sort the whole sequence.
 *  @param[in] reverse Whether to sort in descending order.
 *  @param[in] allocator Optional memory allocator for temporary storage.
 *  @param[out] order Output permutation that sorts the elements.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k on success, @c sz_bad_alloc_k if memory allocation failed, or
 *      @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *  @pre The @p order array must fit at least as many integers as the @p sequence holds.
 *  @post The @p order array will contain a valid permutation of all indices of the @p sequence.
 *  @note This algorithm is @b stable: case-folded-equal elements keep their input order.
 *
 *  Malformed UTF-8 is handled losslessly: any byte that does not begin a well-formed codepoint is
 *  compared and sorted by its raw byte value as a single one-byte unit, and decoding resyncs at the
 *  next byte. The result stays a total, deterministic order; valid input sorts byte-identically.
 *
 *  @sa sz_utf8_uncased_fold, sz_utf8_uncased_order
 *  @sa sz_sequence_argsort_uncased_serial
 */
STRINGZILLA_API sz_status_t sz_sequence_argsort_uncased_best(              //
    sz_sequence_t const *sequence, sz_size_t top_count, sz_bool_t reverse, //
    sz_allocator_t *allocator, sz_sorted_idx_t *order,                     //
    sz_capability_t capabilities, void *stream);

/** @copydoc sz_sequence_argsort_best */
STRINGZILLA_API sz_status_t sz_sequence_argsort_serial(sz_sequence_t const *sequence, sz_size_t top_count,
                                                       sz_bool_t reverse, sz_allocator_t *allocator,
                                                       sz_sorted_idx_t *order, void *stream);

/** @copydoc sz_sequence_argsort_uncased_best */
STRINGZILLA_API sz_status_t sz_sequence_argsort_uncased_serial(sz_sequence_t const *sequence, sz_size_t top_count,
                                                               sz_bool_t reverse, sz_allocator_t *allocator,
                                                               sz_sorted_idx_t *order, void *stream);

#if STRINGZILLA_TARGET_HASWELL

/** @copydoc sz_sequence_argsort_best */
STRINGZILLA_API sz_status_t sz_sequence_argsort_haswell(sz_sequence_t const *sequence, sz_size_t top_count,
                                                        sz_bool_t reverse, sz_allocator_t *allocator,
                                                        sz_sorted_idx_t *order, void *stream);

/** @copydoc sz_sequence_argsort_uncased_best */
STRINGZILLA_API sz_status_t sz_sequence_argsort_uncased_haswell(sz_sequence_t const *sequence, sz_size_t top_count,
                                                                sz_bool_t reverse, sz_allocator_t *allocator,
                                                                sz_sorted_idx_t *order, void *stream);

#endif

#if STRINGZILLA_TARGET_SKYLAKE

/** @copydoc sz_sequence_argsort_best */
STRINGZILLA_API sz_status_t sz_sequence_argsort_skylake(sz_sequence_t const *sequence, sz_size_t top_count,
                                                        sz_bool_t reverse, sz_allocator_t *allocator,
                                                        sz_sorted_idx_t *order, void *stream);

/** @copydoc sz_sequence_argsort_uncased_best */
STRINGZILLA_API sz_status_t sz_sequence_argsort_uncased_skylake(sz_sequence_t const *sequence, sz_size_t top_count,
                                                                sz_bool_t reverse, sz_allocator_t *allocator,
                                                                sz_sorted_idx_t *order, void *stream);

#endif

#if STRINGZILLA_TARGET_SVE

/** @copydoc sz_sequence_argsort_best */
STRINGZILLA_API sz_status_t sz_sequence_argsort_sve(sz_sequence_t const *sequence, sz_size_t top_count,
                                                    sz_bool_t reverse, sz_allocator_t *allocator,
                                                    sz_sorted_idx_t *order, void *stream);

/** @copydoc sz_sequence_argsort_uncased_best */
STRINGZILLA_API sz_status_t sz_sequence_argsort_uncased_sve(sz_sequence_t const *sequence, sz_size_t top_count,
                                                            sz_bool_t reverse, sz_allocator_t *allocator,
                                                            sz_sorted_idx_t *order, void *stream);

#endif

#if STRINGZILLA_TARGET_NEON

/** @copydoc sz_sequence_argsort_best */
STRINGZILLA_API sz_status_t sz_sequence_argsort_neon(sz_sequence_t const *sequence, sz_size_t top_count,
                                                     sz_bool_t reverse, sz_allocator_t *allocator,
                                                     sz_sorted_idx_t *order, void *stream);

/** @copydoc sz_sequence_argsort_uncased_best */
STRINGZILLA_API sz_status_t sz_sequence_argsort_uncased_neon(sz_sequence_t const *sequence, sz_size_t top_count,
                                                             sz_bool_t reverse, sz_allocator_t *allocator,
                                                             sz_sorted_idx_t *order, void *stream);

#endif

#if STRINGZILLA_TARGET_RVV

/** @copydoc sz_sequence_argsort_best */
STRINGZILLA_API sz_status_t sz_sequence_argsort_rvv(sz_sequence_t const *sequence, sz_size_t top_count,
                                                    sz_bool_t reverse, sz_allocator_t *allocator,
                                                    sz_sorted_idx_t *order, void *stream);

/** @copydoc sz_sequence_argsort_uncased_best */
STRINGZILLA_API sz_status_t sz_sequence_argsort_uncased_rvv(sz_sequence_t const *sequence, sz_size_t top_count,
                                                            sz_bool_t reverse, sz_allocator_t *allocator,
                                                            sz_sorted_idx_t *order, void *stream);

#endif

/**
 *  @brief Finds the sorting kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_sort_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion

#include "stringzilla/sort/serial.h"
#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/sort/haswell.h"
#include "stringzilla/sort/skylake.h"
#include "stringzilla/sort/sve.h"
#include "stringzilla/sort/neon.h"
#include "stringzilla/sort/rvv.h"
#endif // STRINGZILLA_HEADER_ONLY

#pragma region Dispatch

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_sequence_argsort_best(sz_sequence_t const *sequence, sz_size_t top_count,
                                                     sz_bool_t reverse, sz_allocator_t *allocator,
                                                     sz_sorted_idx_t *order, sz_capability_t capabilities,
                                                     void *stream) {
    sz_unused_(sequence), sz_unused_(top_count), sz_unused_(reverse), sz_unused_(allocator), sz_unused_(order),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_sequence_argsort_uncased_best(              //
    sz_sequence_t const *sequence, sz_size_t top_count, sz_bool_t reverse, //
    sz_allocator_t *allocator, sz_sorted_idx_t *order,                     //
    sz_capability_t capabilities, void *stream) {
    sz_unused_(sequence), sz_unused_(top_count), sz_unused_(reverse), sz_unused_(allocator), sz_unused_(order),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_sort_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
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
#endif // STRINGZILLA_SORT_H_
