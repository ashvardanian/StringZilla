/**
 *  @file include/stringzilla/sort.h
 *  @author Ash Vardanian
 *  @date February 15, 2025
 *  @brief Hardware-accelerated sorting of string collections.
 *
 *  Two dispatch points return the stable permutation that sorts a @c sz_strings_view_t:
 *
 *  - @c sz_sort_order_best orders strings byte-lexicographically, like @c memcmp would.
 *  - @c sz_sort_uncased_order_best orders UTF-8 strings under Unicode case folding, folding
 *    small chunks of each string on the fly instead of materializing a folded copy.
 *
 *  A byte that does not begin a well-formed codepoint sorts by its raw value as a one-byte unit,
 *  keeping the uncased order total and deterministic.
 *
 *  Both are stable, take @c reverse for descending order, and accept @c top_count to fully order
 *  only the leading elements, a partial sort that prunes work on the tail the caller skips.
 *
 *  Internally both compare several bytes at a time: each pass packs the next bytes, or folded
 *  codepoints, of every string into a pointer-sized integer key and sorts the keys, recursing
 *  only into runs of equal keys.
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
 *  @brief Faster @b stable arg-sort for an arbitrary @b string collection, using QuickSort.
 *
 *  Outputs the @p order of elements in the immutable @p strings, that would sort them.
 *
 *  @param[in] strings Immutable collection of strings to sort, in any layout.
 *  @param[in] top_count Number of leading elements to fully order, or 0 to sort all of them.
 *  @param[in] reverse Whether to sort in descending order.
 *  @param[in] allocator Optional memory allocator for temporary storage.
 *  @param[out] order Output permutation that sorts the elements.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k on success, @c sz_bad_alloc_k if memory allocation failed, or
 *      @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *  @pre The @p order array must fit at least as many integers as @p strings holds.
 *  @post The @p order array will contain a valid permutation of all indices of @p strings.
 *  @post If @p top_count is non-zero and smaller than the count, only the first @p top_count
 *      entries are sorted; the rest are an arbitrary permutation of the leftover indices.
 *
 *  For example, sorting three fruit names:
 *
 *  @code{.c}
 *      #include <stringzilla/stringzilla.h>
 *      int main() {
 *          char const *fruits[] = {"banana", "apple", "cherry"};
 *          sz_strings_view_t strings;
 *          sz_strings_from_null_terminated_strings(fruits, 3, &strings);
 *          sz_capability_t capabilities;
 *          sz_capabilities_enabled_cpu(&capabilities);
 *          sz_size_t order[3];
 *          sz_status_t status =
 *              sz_sort_order_best(&strings, 0, sz_false_k, NULL, order, capabilities, NULL);
 *          return status == sz_success_k && order[0] == 1 && order[1] == 0 && order[2] == 2 ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note Takes linear memory, quadratic worst-case and log-linear average time.
 *  @see Quicksort: https://en.wikipedia.org/wiki/Quicksort
 *
 *  @note This algorithm is @b stable: equal elements keep their relative order, ascending by index.
 *
 *  @sa sz_sort_order_serial, sz_sort_order_haswell, sz_sort_order_skylake, sz_sort_order_neon,
 *      sz_sort_order_sve, sz_sort_order_rvv
 *  @sa sz_sort_uncased_order_best
 */
STRINGZILLA_API sz_status_t sz_sort_order_best(sz_strings_view_t const *strings, sz_size_t top_count, sz_bool_t reverse,
                                               sz_allocator_t *allocator, sz_size_t *order,
                                               sz_capability_t capabilities, sz_stream_t stream);

/**
 *  @brief Faster @b stable @b uncased arg-sort for a UTF-8 @b string collection, using QuickSort.
 *
 *  Orders strings under Unicode case-folding, equivalent to folding every string and sorting the
 *  folded bytes lexicographically - but folding only small chunks on the fly, never materializing a
 *  fully pre-folded copy of the collection.
 *
 *  @param[in] strings Immutable collection of UTF-8 strings to sort, in any layout.
 *  @param[in] top_count Number of leading elements to fully order, or 0 to sort all of them.
 *  @param[in] reverse Whether to sort in descending order.
 *  @param[in] allocator Optional memory allocator for temporary storage.
 *  @param[out] order Output permutation that sorts the elements.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k on success, @c sz_bad_alloc_k if memory allocation failed, or
 *      @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *  @pre The @p order array must fit at least as many integers as @p strings holds.
 *  @post The @p order array will contain a valid permutation of all indices of @p strings.
 *  @note This algorithm is @b stable: case-folded-equal elements keep their input order.
 *
 *  Malformed UTF-8 is handled losslessly: any byte that does not begin a well-formed codepoint is
 *  compared and sorted by its raw byte value as a single one-byte unit, and decoding resyncs at the
 *  next byte. The result stays a total, deterministic order; valid input sorts byte-identically.
 *
 *  @sa sz_utf8_uncased_fold, sz_utf8_uncased_order
 *  @sa sz_sort_uncased_order_serial
 */
STRINGZILLA_API sz_status_t sz_sort_uncased_order_best(sz_strings_view_t const *strings, sz_size_t top_count,
                                                       sz_bool_t reverse, sz_allocator_t *allocator, sz_size_t *order,
                                                       sz_capability_t capabilities, sz_stream_t stream);

/** @copydoc sz_sort_order_best */
STRINGZILLA_API sz_status_t sz_sort_order_serial(sz_strings_view_t const *strings, sz_size_t top_count,
                                                 sz_bool_t reverse, sz_allocator_t *allocator, sz_size_t *order,
                                                 sz_stream_t stream);

/** @copydoc sz_sort_uncased_order_best */
STRINGZILLA_API sz_status_t sz_sort_uncased_order_serial(sz_strings_view_t const *strings, sz_size_t top_count,
                                                         sz_bool_t reverse, sz_allocator_t *allocator, sz_size_t *order,
                                                         sz_stream_t stream);

#if STRINGZILLA_TARGET_HASWELL

/** @copydoc sz_sort_order_best */
STRINGZILLA_API sz_status_t sz_sort_order_haswell(sz_strings_view_t const *strings, sz_size_t top_count,
                                                  sz_bool_t reverse, sz_allocator_t *allocator, sz_size_t *order,
                                                  sz_stream_t stream);

/** @copydoc sz_sort_uncased_order_best */
STRINGZILLA_API sz_status_t sz_sort_uncased_order_haswell(sz_strings_view_t const *strings, sz_size_t top_count,
                                                          sz_bool_t reverse, sz_allocator_t *allocator,
                                                          sz_size_t *order, sz_stream_t stream);

#endif

#if STRINGZILLA_TARGET_SKYLAKE

/** @copydoc sz_sort_order_best */
STRINGZILLA_API sz_status_t sz_sort_order_skylake(sz_strings_view_t const *strings, sz_size_t top_count,
                                                  sz_bool_t reverse, sz_allocator_t *allocator, sz_size_t *order,
                                                  sz_stream_t stream);

/** @copydoc sz_sort_uncased_order_best */
STRINGZILLA_API sz_status_t sz_sort_uncased_order_skylake(sz_strings_view_t const *strings, sz_size_t top_count,
                                                          sz_bool_t reverse, sz_allocator_t *allocator,
                                                          sz_size_t *order, sz_stream_t stream);

#endif

#if STRINGZILLA_TARGET_SVE

/** @copydoc sz_sort_order_best */
STRINGZILLA_API sz_status_t sz_sort_order_sve(sz_strings_view_t const *strings, sz_size_t top_count, sz_bool_t reverse,
                                              sz_allocator_t *allocator, sz_size_t *order, sz_stream_t stream);

/** @copydoc sz_sort_uncased_order_best */
STRINGZILLA_API sz_status_t sz_sort_uncased_order_sve(sz_strings_view_t const *strings, sz_size_t top_count,
                                                      sz_bool_t reverse, sz_allocator_t *allocator, sz_size_t *order,
                                                      sz_stream_t stream);

#endif

#if STRINGZILLA_TARGET_NEON

/** @copydoc sz_sort_order_best */
STRINGZILLA_API sz_status_t sz_sort_order_neon(sz_strings_view_t const *strings, sz_size_t top_count, sz_bool_t reverse,
                                               sz_allocator_t *allocator, sz_size_t *order, sz_stream_t stream);

/** @copydoc sz_sort_uncased_order_best */
STRINGZILLA_API sz_status_t sz_sort_uncased_order_neon(sz_strings_view_t const *strings, sz_size_t top_count,
                                                       sz_bool_t reverse, sz_allocator_t *allocator, sz_size_t *order,
                                                       sz_stream_t stream);

#endif

#if STRINGZILLA_TARGET_RVV

/** @copydoc sz_sort_order_best */
STRINGZILLA_API sz_status_t sz_sort_order_rvv(sz_strings_view_t const *strings, sz_size_t top_count, sz_bool_t reverse,
                                              sz_allocator_t *allocator, sz_size_t *order, sz_stream_t stream);

/** @copydoc sz_sort_uncased_order_best */
STRINGZILLA_API sz_status_t sz_sort_uncased_order_rvv(sz_strings_view_t const *strings, sz_size_t top_count,
                                                      sz_bool_t reverse, sz_allocator_t *allocator, sz_size_t *order,
                                                      sz_stream_t stream);

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

STRINGZILLA_API sz_status_t sz_sort_order_best(sz_strings_view_t const *strings, sz_size_t top_count, sz_bool_t reverse,
                                               sz_allocator_t *allocator, sz_size_t *order,
                                               sz_capability_t capabilities, sz_stream_t stream) {
    sz_unused_(strings), sz_unused_(top_count), sz_unused_(reverse), sz_unused_(allocator), sz_unused_(order),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_sort_uncased_order_best(sz_strings_view_t const *strings, sz_size_t top_count,
                                                       sz_bool_t reverse, sz_allocator_t *allocator, sz_size_t *order,
                                                       sz_capability_t capabilities, sz_stream_t stream) {
    sz_unused_(strings), sz_unused_(top_count), sz_unused_(reverse), sz_unused_(allocator), sz_unused_(order),
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
