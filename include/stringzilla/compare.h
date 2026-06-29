/**
 *  @brief Hardware-accelerated string comparison utilities.
 *  @file include/stringzilla/compare.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs:
 *
 *  - `sz_equal` - for equality comparison of two strings.
 *  - `sz_order` - for the relative order of two strings, similar to `memcmp`.
 *
 *  A valid suggestion may be to add an `sz_mismatch`, as the shared part of the `sz_order` and `sz_equal`.
 *  That would be great for a general-purpose library, but has little practical use for string processing.
 *
 *  The functions in this file can be used for both UTF-8 and other inputs.
 *  On platforms without masked loads they use interleaved prefix and suffix vector-loads
 *  to avoid scalar code, similar to the kernels in `memory.h`.
 */
#ifndef STRINGZILLA_COMPARE_H_
#define STRINGZILLA_COMPARE_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Checks if two strings are equal. Equivalent to `memcmp(a, b, length) == 0` in LibC and `a == b` in STL.
 *  @see https://en.cppreference.com/w/c/string/byte/memcmp
 *
 *  @param a First string to compare.
 *  @param b Second string to compare.
 *  @param length Number of bytes to compare in both strings.
 *
 *  @retval `sz_true_k` if strings are equal.
 *  @retval `sz_false_k` if strings are different.
 *
 *  Example usage:
 *
 *  @code{.c}
 *      #include <stringzilla/compare.h>
 *      int main() {
 *          return sz_equal("hello", "hello", 5) && !sz_equal("hello", "world", 5);
 *      }
 *  @endcode
 *
 *  @note Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa sz_equal_serial, sz_equal_westmere, sz_equal_haswell, sz_equal_skylake, sz_equal_neon, sz_equal_sve,
 *      sz_equal_v128, sz_equal_v128relaxed, sz_equal_rvv, sz_equal_lasx, sz_equal_powervsx
 */
SZ_API_RUNTIME sz_bool_t sz_equal(sz_cptr_t a, sz_cptr_t b, sz_size_t length);

/**
 *  @brief Compares two strings lexicographically. Equivalent to `memcmp(a, b, length)` in LibC.
 *         Mostly used in sorting and associative containers. Can be used for @b UTF-8 inputs.
 *  @see https://en.cppreference.com/w/c/string/byte/memcmp
 *
 *  This function uses scalar code on most platforms, as in the majority of cases the strings that
 *  differ - will have differences among the very first characters and fetching more than one cache
 *  line may not be justified.
 *
 *  @param a First string to compare.
 *  @param a_length Number of bytes in the first string.
 *  @param b Second string to compare.
 *  @param b_length Number of bytes in the second string.
 *
 *  @retval `sz_less_k` if @p a is lexicographically smaller than @p b.
 *  @retval `sz_greater_k` if @p a is lexicographically greater than @p b.
 *  @retval `sz_equal_k` if strings @p a and @p b are identical.
 *
 *  Example usage:
 *
 *  @code{.c}
 *      #include <stringzilla/compare.h>
 *      int main() {
 *          return sz_order("apple", 5, "banana", 6) < 0 &&
 *                 sz_order("grape", 5, "grape", 5) == 0 &&
 *                 sz_order("zebra", 5, "apple", 5) > 0;
 *      }
 *  @endcode
 *
 *  @note Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa sz_order_serial, sz_order_westmere, sz_order_haswell, sz_order_skylake, sz_order_neon, sz_order_sve,
 *      sz_order_v128, sz_order_v128relaxed, sz_order_rvv, sz_order_lasx, sz_order_powervsx
 */
SZ_API_RUNTIME sz_ordering_t sz_order(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);

/** @copydoc sz_equal */
SZ_API_COMPTIME sz_bool_t sz_equal_serial(sz_cptr_t a, sz_cptr_t b, sz_size_t length);
/** @copydoc sz_order */
SZ_API_COMPTIME sz_ordering_t sz_order_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);

#if SZ_USE_WESTMERE
/** @copydoc sz_equal */
SZ_API_COMPTIME sz_bool_t sz_equal_westmere(sz_cptr_t a, sz_cptr_t b, sz_size_t length);
/** @copydoc sz_order */
SZ_API_COMPTIME sz_ordering_t sz_order_westmere(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);
#endif

#if SZ_USE_HASWELL
/** @copydoc sz_equal */
SZ_API_COMPTIME sz_bool_t sz_equal_haswell(sz_cptr_t a, sz_cptr_t b, sz_size_t length);
/** @copydoc sz_order */
SZ_API_COMPTIME sz_ordering_t sz_order_haswell(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);
#endif

#if SZ_USE_SKYLAKE
/** @copydoc sz_equal */
SZ_API_COMPTIME sz_bool_t sz_equal_skylake(sz_cptr_t a, sz_cptr_t b, sz_size_t length);
/** @copydoc sz_order */
SZ_API_COMPTIME sz_ordering_t sz_order_skylake(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);
#endif

#if SZ_USE_NEON
/** @copydoc sz_equal */
SZ_API_COMPTIME sz_bool_t sz_equal_neon(sz_cptr_t a, sz_cptr_t b, sz_size_t length);
/** @copydoc sz_order */
SZ_API_COMPTIME sz_ordering_t sz_order_neon(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);
#endif

#pragma endregion // Core API

#include "stringzilla/compare/serial.h"
#include "stringzilla/compare/westmere.h"
#include "stringzilla/compare/haswell.h"
#include "stringzilla/compare/skylake.h"
#include "stringzilla/compare/neon.h"
#include "stringzilla/compare/sve.h"
#include "stringzilla/compare/v128relaxed.h"
#include "stringzilla/compare/v128.h"
#include "stringzilla/compare/rvv.h"
#include "stringzilla/compare/lasx.h"
#include "stringzilla/compare/powervsx.h"

/*  Pick the right implementation for the string search algorithms.
 *  To override this behavior and precompile all backends - set `SZ_DYNAMIC_DISPATCH` to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_API_RUNTIME sz_bool_t sz_equal(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
#if SZ_USE_V128RELAXED
    return sz_equal_v128relaxed(a, b, length);
#elif SZ_USE_V128
    return sz_equal_v128(a, b, length);
#elif SZ_USE_RVV
    return sz_equal_rvv(a, b, length);
#elif SZ_USE_LASX
    return sz_equal_lasx(a, b, length);
#elif SZ_USE_POWERVSX
    return sz_equal_powervsx(a, b, length);
#elif SZ_USE_SKYLAKE
    return sz_equal_skylake(a, b, length);
#elif SZ_USE_HASWELL
    return sz_equal_haswell(a, b, length);
#elif SZ_USE_SVE && SZ_ENFORCE_SVE_OVER_NEON
    return sz_equal_sve(a, b, length);
#elif SZ_USE_NEON
    return sz_equal_neon(a, b, length);
#else
    return sz_equal_serial(a, b, length);
#endif
}

SZ_API_RUNTIME sz_ordering_t sz_order(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
#if SZ_USE_V128RELAXED
    return sz_order_v128relaxed(a, a_length, b, b_length);
#elif SZ_USE_V128
    return sz_order_v128(a, a_length, b, b_length);
#elif SZ_USE_RVV
    return sz_order_rvv(a, a_length, b, b_length);
#elif SZ_USE_LASX
    return sz_order_lasx(a, a_length, b, b_length);
#elif SZ_USE_POWERVSX
    return sz_order_powervsx(a, a_length, b, b_length);
#elif SZ_USE_SKYLAKE
    return sz_order_skylake(a, a_length, b, b_length);
#elif SZ_USE_HASWELL
    return sz_order_haswell(a, a_length, b, b_length);
#elif SZ_USE_SVE && SZ_ENFORCE_SVE_OVER_NEON
    return sz_order_sve(a, a_length, b, b_length);
#elif SZ_USE_NEON
    return sz_order_neon(a, a_length, b, b_length);
#else
    return sz_order_serial(a, a_length, b, b_length);
#endif
}

#endif            // !SZ_DYNAMIC_DISPATCH
#pragma endregion // Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_COMPARE_H_
