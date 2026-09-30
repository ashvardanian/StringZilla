/**
 *  @file include/stringzilla/compare.h
 *  @author Ash Vardanian
 *  @date August 14, 2020
 *  @brief Hardware-accelerated string comparison utilities.
 *
 *  Includes core APIs:
 *
 *  - @c sz_equal_best - for equality comparison of two strings.
 *  - @c sz_order_best - for the relative order of two strings, similar to @c memcmp.
 *
 *  A valid suggestion may be to add an @c sz_mismatch, as the shared part of @c sz_order_best and
 *  @c sz_equal_best. That would be great for a general-purpose library, but string processing has
 *  little practical use for it.
 *
 *  The functions in this file can be used for both UTF-8 and other inputs. On platforms without
 *  masked loads they use interleaved prefix and suffix vector-loads to avoid scalar code, similar
 *  to the kernels in `memory.h`.
 */
#ifndef STRINGZILLA_COMPARE_H_
#define STRINGZILLA_COMPARE_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Checks if two strings are equal, like `memcmp(a, b, length) == 0`, or `a == b` in STL.
 *
 *  @param[in] a First string to compare.
 *  @param[in] b Second string to compare.
 *  @param[in] length Number of bytes to compare in both strings.
 *  @param[out] equal @c sz_true_k if the strings are equal, @c sz_false_k if they differ.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *  @see memcmp: https://en.cppreference.com/w/c/string/byte/memcmp
 *
 *  Telling an equal pair from a different one:
 *
 *  @code{.c}
 *      #include <stringzilla/compare.h>
 *      int main() {
 *          sz_capability_t capabilities;
 *          sz_bool_t same, different;
 *          sz_cpu_capabilities_enabled(&capabilities);
 *          sz_equal_best("hello", "hello", 5, &same, capabilities, NULL);
 *          sz_equal_best("hello", "world", 5, &different, capabilities, NULL);
 *          return same && !different;
 *      }
 *  @endcode
 *
 *  @sa sz_equal_serial, sz_equal_westmere, sz_equal_haswell, sz_equal_skylake,
 *      sz_equal_neon, sz_equal_sve, sz_equal_v128, sz_equal_v128relaxed, sz_equal_rvv,
 *      sz_equal_loongsonasx, sz_equal_powervsx
 */
STRINGZILLA_API sz_status_t sz_equal_best(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal,
                                          sz_capability_t capabilities, void *stream);

/**
 *  @brief Compares two strings lexicographically, like @c memcmp in LibC.
 *
 *  @param[in] a First string to compare.
 *  @param[in] a_length Number of bytes in the first string.
 *  @param[in] b Second string to compare.
 *  @param[in] b_length Number of bytes in the second string.
 *  @param[out] ordering @c sz_less_k if @p a sorts before @p b, @c sz_greater_k if after, or
 *      @c sz_equal_k if the strings are identical.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *  @see memcmp: https://en.cppreference.com/w/c/string/byte/memcmp
 *
 *  Mostly used in sorting and associative containers, and can be used for @b UTF-8 inputs. This
 *  function uses scalar code on most platforms, as in the majority of cases the strings that differ
 *  will have differences among the very first characters and fetching more than one cache line may
 *  not be justified.
 *
 *  Ordering three pairs of words:
 *
 *  @code{.c}
 *      #include <stringzilla/compare.h>
 *      int main() {
 *          sz_capability_t capabilities;
 *          sz_ordering_t first, second, third;
 *          sz_cpu_capabilities_enabled(&capabilities);
 *          sz_order_best("apple", 5, "banana", 6, &first, capabilities, NULL);
 *          sz_order_best("grape", 5, "grape", 5, &second, capabilities, NULL);
 *          sz_order_best("zebra", 5, "apple", 5, &third, capabilities, NULL);
 *          return first < 0 && second == 0 && third > 0;
 *      }
 *  @endcode
 *
 *  @sa sz_order_serial, sz_order_westmere, sz_order_haswell, sz_order_skylake,
 *      sz_order_neon, sz_order_sve, sz_order_v128, sz_order_v128relaxed, sz_order_rvv,
 *      sz_order_loongsonasx, sz_order_powervsx
 */
STRINGZILLA_API sz_status_t sz_order_best(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                          sz_ordering_t *ordering, sz_capability_t capabilities, void *stream);

/** @copydoc sz_equal_best */
STRINGZILLA_API sz_status_t sz_equal_serial(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal, void *stream);

/** @copydoc sz_order_best */
STRINGZILLA_API sz_status_t sz_order_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                            sz_ordering_t *ordering, void *stream);

#if STRINGZILLA_TARGET_WESTMERE

/** @copydoc sz_equal_best */
STRINGZILLA_API sz_status_t sz_equal_westmere(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal,
                                              void *stream);

/** @copydoc sz_order_best */
STRINGZILLA_API sz_status_t sz_order_westmere(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                              sz_ordering_t *ordering, void *stream);
#endif

#if STRINGZILLA_TARGET_HASWELL

/** @copydoc sz_equal_best */
STRINGZILLA_API sz_status_t sz_equal_haswell(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal,
                                             void *stream);

/** @copydoc sz_order_best */
STRINGZILLA_API sz_status_t sz_order_haswell(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                             sz_ordering_t *ordering, void *stream);
#endif

#if STRINGZILLA_TARGET_SKYLAKE

/** @copydoc sz_equal_best */
STRINGZILLA_API sz_status_t sz_equal_skylake(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal,
                                             void *stream);

/** @copydoc sz_order_best */
STRINGZILLA_API sz_status_t sz_order_skylake(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                             sz_ordering_t *ordering, void *stream);
#endif

#if STRINGZILLA_TARGET_NEON

/** @copydoc sz_equal_best */
STRINGZILLA_API sz_status_t sz_equal_neon(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal, void *stream);

/** @copydoc sz_order_best */
STRINGZILLA_API sz_status_t sz_order_neon(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                          sz_ordering_t *ordering, void *stream);
#endif

#if STRINGZILLA_TARGET_SVE

/** @copydoc sz_equal_best */
STRINGZILLA_API sz_status_t sz_equal_sve(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal, void *stream);

/** @copydoc sz_order_best */
STRINGZILLA_API sz_status_t sz_order_sve(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                         sz_ordering_t *ordering, void *stream);
#endif

#if STRINGZILLA_TARGET_RVV

/** @copydoc sz_equal_best */
STRINGZILLA_API sz_status_t sz_equal_rvv(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal, void *stream);

/** @copydoc sz_order_best */
STRINGZILLA_API sz_status_t sz_order_rvv(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                         sz_ordering_t *ordering, void *stream);
#endif

#if STRINGZILLA_TARGET_V128

/** @copydoc sz_equal_best */
STRINGZILLA_API sz_status_t sz_equal_v128(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal, void *stream);

/** @copydoc sz_order_best */
STRINGZILLA_API sz_status_t sz_order_v128(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                          sz_ordering_t *ordering, void *stream);
#endif

#if STRINGZILLA_TARGET_V128RELAXED

/** @copydoc sz_equal_best */
STRINGZILLA_API sz_status_t sz_equal_v128relaxed(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal,
                                                 void *stream);

/** @copydoc sz_order_best */
STRINGZILLA_API sz_status_t sz_order_v128relaxed(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                                 sz_ordering_t *ordering, void *stream);
#endif

#if STRINGZILLA_TARGET_LOONGSONASX

/** @copydoc sz_equal_best */
STRINGZILLA_API sz_status_t sz_equal_loongsonasx(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal,
                                                 void *stream);

/** @copydoc sz_order_best */
STRINGZILLA_API sz_status_t sz_order_loongsonasx(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                                 sz_ordering_t *ordering, void *stream);
#endif

#if STRINGZILLA_TARGET_POWERVSX

/** @copydoc sz_equal_best */
STRINGZILLA_API sz_status_t sz_equal_powervsx(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal,
                                              void *stream);

/** @copydoc sz_order_best */
STRINGZILLA_API sz_status_t sz_order_powervsx(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                              sz_ordering_t *ordering, void *stream);
#endif

/**
 *  @brief Finds the comparison kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_compare_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                   sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion Core API

#include "stringzilla/compare/serial.h"
#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/compare/westmere.h"
#include "stringzilla/compare/haswell.h"
#include "stringzilla/compare/skylake.h"
#include "stringzilla/compare/neon.h"
#include "stringzilla/compare/sve.h"
#include "stringzilla/compare/v128relaxed.h"
#include "stringzilla/compare/v128.h"
#include "stringzilla/compare/rvv.h"
#include "stringzilla/compare/loongsonasx.h"
#include "stringzilla/compare/powervsx.h"
#endif // STRINGZILLA_HEADER_ONLY

/*  Header-only builds link no library, so their dispatch points only report it missing. */
#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_equal_best(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal,
                                          sz_capability_t capabilities, void *stream) {
    sz_unused_(a), sz_unused_(b), sz_unused_(length), sz_unused_(equal), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_order_best(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                          sz_ordering_t *ordering, sz_capability_t capabilities, void *stream) {
    sz_unused_(a), sz_unused_(a_length), sz_unused_(b), sz_unused_(b_length), sz_unused_(ordering),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_compare_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                   sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_COMPARE_H_
