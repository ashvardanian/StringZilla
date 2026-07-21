/**
 *  @brief Hardware-accelerated sub-string and character-set search utilities.
 *  @file include/stringzilla/find.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs:
 *
 *  - `sz_find` and reverse-order `sz_rfind`
 *  - `sz_find_byte` and reverse-order `sz_rfind_byte`
 *  - `sz_find_byteset` and reverse-order `sz_rfind_byteset`
 *
 *  Convenience functions for character-set matching:
 *
 *  - `sz_find_byte_from` shortcut for `sz_find_byteset`
 *  - `sz_find_byte_not_from` shortcut for `sz_find_byteset` with inverted set
 *  - `sz_rfind_byte_from` shortcut for `sz_rfind_byteset`
 *  - `sz_rfind_byte_not_from` shortcut for `sz_rfind_byteset` with inverted set
 */
#ifndef STRINGZILLA_FIND_H_
#define STRINGZILLA_FIND_H_

#include "stringzilla/types.h"

#include "stringzilla/compare.h" // `sz_equal`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Locates first matching byte in a string. Equivalent to `memchr(haystack, *needle, haystack_length)` in LibC.
 *
 *  @see X86_64 implementation: https://github.com/lattera/glibc/blob/master/sysdeps/x86_64/memchr.S
 *  @see Aarch64 implementation: https://github.com/lattera/glibc/blob/master/sysdeps/aarch64/memchr.S
 *
 *  @param haystack Haystack - the string to search in.
 *  @param haystack_length Number of bytes in the haystack.
 *  @param needle Needle - single-byte substring to find.
 *  @return Address of the first match. NULL if not found.
 */
SZ_API_RUNTIME sz_cptr_t sz_find_byte(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);

/**
 *  @brief Locates last matching byte in a string. Equivalent to `memrchr(haystack, *needle, haystack_length)` in LibC.
 *
 *  @see X86_64 implementation: https://github.com/lattera/glibc/blob/master/sysdeps/x86_64/memrchr.S
 *  @see Aarch64 implementation: missing
 *
 *  @param haystack Haystack - the string to search in.
 *  @param haystack_length Number of bytes in the haystack.
 *  @param needle Needle - single-byte substring to find.
 *  @return Address of the last match. NULL if not found.
 */
SZ_API_RUNTIME sz_cptr_t sz_rfind_byte(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);

/** @copydoc sz_find_byte */
SZ_API_COMPTIME sz_cptr_t sz_find_byte_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);

#if SZ_USE_WESTMERE
/** @copydoc sz_find_byte */
SZ_API_COMPTIME sz_cptr_t sz_find_byte_westmere(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_westmere(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
#endif

#if SZ_USE_HASWELL
/** @copydoc sz_find_byte */
SZ_API_COMPTIME sz_cptr_t sz_find_byte_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
#endif

#if SZ_USE_SKYLAKE
/** @copydoc sz_find_byte */
SZ_API_COMPTIME sz_cptr_t sz_find_byte_skylake(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_skylake(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
#endif

#if SZ_USE_NEON
/** @copydoc sz_find_byte */
SZ_API_COMPTIME sz_cptr_t sz_find_byte_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
#endif

#if SZ_USE_SVE
/** @copydoc sz_find_byte */
SZ_API_COMPTIME sz_cptr_t sz_find_byte_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
#endif

#if SZ_USE_V128RELAXED
/** @copydoc sz_find_byte */
SZ_API_COMPTIME sz_cptr_t sz_find_byte_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
#endif

#if SZ_USE_V128
/** @copydoc sz_find_byte */
SZ_API_COMPTIME sz_cptr_t sz_find_byte_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
#endif

#if SZ_USE_RVV
/** @copydoc sz_find_byte */
SZ_API_COMPTIME sz_cptr_t sz_find_byte_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
#endif

#if SZ_USE_LASX
/** @copydoc sz_find_byte */
SZ_API_COMPTIME sz_cptr_t sz_find_byte_lasx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_lasx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
#endif

#if SZ_USE_POWERVSX
/** @copydoc sz_find_byte */
SZ_API_COMPTIME sz_cptr_t sz_find_byte_powervsx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_powervsx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
#endif

/**
 *  @brief Locates first matching substring.
 *         Equivalent to `memmem(haystack, haystack_length, needle, needle_length)` in LibC.
 *         Similar to `strstr(haystack, needle)` in LibC, but requires known length.
 *
 *  @param haystack Haystack - the string to search in.
 *  @param haystack_length Number of bytes in the haystack.
 *  @param needle Needle - substring to find.
 *  @param needle_length Number of bytes in the needle.
 *  @return Address of the first match.
 */
SZ_API_RUNTIME sz_cptr_t sz_find(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                 sz_size_t needle_length);

/**
 *  @brief Locates the last matching substring.
 *
 *  @param haystack Haystack - the string to search in.
 *  @param haystack_length Number of bytes in the haystack.
 *  @param needle Needle - substring to find.
 *  @param needle_length Number of bytes in the needle.
 *  @return Address of the last match.
 */
SZ_API_RUNTIME sz_cptr_t sz_rfind(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                  sz_size_t needle_length);

/** @copydoc sz_find */
SZ_API_COMPTIME sz_cptr_t sz_find_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                         sz_size_t needle_length);
/** @copydoc sz_rfind */
SZ_API_COMPTIME sz_cptr_t sz_rfind_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                          sz_size_t needle_length);

#if SZ_USE_WESTMERE
/** @copydoc sz_find */
SZ_API_COMPTIME sz_cptr_t sz_find_westmere(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                           sz_size_t needle_length);
/** @copydoc sz_rfind */
SZ_API_COMPTIME sz_cptr_t sz_rfind_westmere(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                            sz_size_t needle_length);
#endif

#if SZ_USE_HASWELL
/** @copydoc sz_find */
SZ_API_COMPTIME sz_cptr_t sz_find_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                          sz_size_t needle_length);
/** @copydoc sz_rfind */
SZ_API_COMPTIME sz_cptr_t sz_rfind_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                           sz_size_t needle_length);
#endif

#if SZ_USE_SKYLAKE
/** @copydoc sz_find */
SZ_API_COMPTIME sz_cptr_t sz_find_skylake(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                          sz_size_t needle_length);
/** @copydoc sz_rfind */
SZ_API_COMPTIME sz_cptr_t sz_rfind_skylake(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                           sz_size_t needle_length);
#endif

#if SZ_USE_NEON
/** @copydoc sz_find */
SZ_API_COMPTIME sz_cptr_t sz_find_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                       sz_size_t needle_length);
/** @copydoc sz_rfind */
SZ_API_COMPTIME sz_cptr_t sz_rfind_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                        sz_size_t needle_length);
#endif

#if SZ_USE_SVE
/** @copydoc sz_find */
SZ_API_COMPTIME sz_cptr_t sz_find_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                      sz_size_t needle_length);
/** @copydoc sz_rfind */
SZ_API_COMPTIME sz_cptr_t sz_rfind_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                       sz_size_t needle_length);
#endif

#if SZ_USE_V128RELAXED
/** @copydoc sz_find */
SZ_API_COMPTIME sz_cptr_t sz_find_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                              sz_size_t needle_length);
/** @copydoc sz_rfind */
SZ_API_COMPTIME sz_cptr_t sz_rfind_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                               sz_size_t needle_length);
#endif

#if SZ_USE_V128
/** @copydoc sz_find */
SZ_API_COMPTIME sz_cptr_t sz_find_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                       sz_size_t needle_length);
/** @copydoc sz_rfind */
SZ_API_COMPTIME sz_cptr_t sz_rfind_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                        sz_size_t needle_length);
#endif

#if SZ_USE_RVV
/** @copydoc sz_find */
SZ_API_COMPTIME sz_cptr_t sz_find_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                      sz_size_t needle_length);
/** @copydoc sz_rfind */
SZ_API_COMPTIME sz_cptr_t sz_rfind_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                       sz_size_t needle_length);
#endif

#if SZ_USE_LASX
/** @copydoc sz_find */
SZ_API_COMPTIME sz_cptr_t sz_find_lasx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                       sz_size_t needle_length);
/** @copydoc sz_rfind */
SZ_API_COMPTIME sz_cptr_t sz_rfind_lasx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                        sz_size_t needle_length);
#endif

#if SZ_USE_POWERVSX
/** @copydoc sz_find */
SZ_API_COMPTIME sz_cptr_t sz_find_powervsx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                           sz_size_t needle_length);
/** @copydoc sz_rfind */
SZ_API_COMPTIME sz_cptr_t sz_rfind_powervsx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                            sz_size_t needle_length);
#endif

/**
 *  @brief Finds the first character present from the @p set, present in @p text.
 *         Equivalent to `strspn(text, accepted)` and `strcspn(text, rejected)` in LibC.
 *         May have identical implementation and performance to ::sz_rfind_byteset.
 *
 *  Useful for parsing, when we want to skip a set of characters. Examples:
 *  - 6 whitespaces: " \t\n\r\v\f".
 *  - 16 digits forming a float number: "0123456789,.eE+-".
 *  - 5 HTML reserved characters: "\"'&<>", of which "<>" can be useful for parsing.
 *  - 2 JSON string special characters useful to locate the end of the string: "\"\\".
 *
 *  @param text String to be scanned.
 *  @param length Number of bytes in the string.
 *  @param set Set of relevant characters.
 *  @return Pointer to the first matching character from @p set.
 */
SZ_API_RUNTIME sz_cptr_t sz_find_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set);

/**
 *  @brief Finds the last character present from the @p set, present in @p text.
 *         Equivalent to `strspn(text, accepted)` and `strcspn(text, rejected)` in LibC.
 *         May have identical implementation and performance to ::sz_find_byteset.
 *
 *  Useful for parsing, when we want to skip a set of characters. Examples:
 *  - 6 whitespaces: " \t\n\r\v\f".
 *  - 16 digits forming a float number: "0123456789,.eE+-".
 *  - 5 HTML reserved characters: "\"'&<>", of which "<>" can be useful for parsing.
 *  - 2 JSON string special characters useful to locate the end of the string: "\"\\".
 *
 *  @param text String to be scanned.
 *  @param length Number of bytes in the string.
 *  @param set Set of relevant characters.
 *  @return Pointer to the last matching character from @p set.
 */
SZ_API_RUNTIME sz_cptr_t sz_rfind_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set);

/** @copydoc sz_find_byteset */
SZ_API_COMPTIME sz_cptr_t sz_find_byteset_serial(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set);
/** @copydoc sz_rfind_byteset */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_serial(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set);

#if SZ_USE_HASWELL
/** @copydoc sz_find_byteset */
SZ_API_COMPTIME sz_cptr_t sz_find_byteset_haswell(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
/** @copydoc sz_rfind_byteset */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_haswell(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

#if SZ_USE_ICELAKE
/** @copydoc sz_find_byteset */
SZ_API_COMPTIME sz_cptr_t sz_find_byteset_icelake(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
/** @copydoc sz_rfind_byteset */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_icelake(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

#if SZ_USE_NEON
/** @copydoc sz_find_byteset */
SZ_API_COMPTIME sz_cptr_t sz_find_byteset_neon(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
/** @copydoc sz_rfind_byteset */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_neon(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

#if SZ_USE_SVE2
/** @copydoc sz_find_byteset */
SZ_API_COMPTIME sz_cptr_t sz_find_byteset_sve2(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
/** @copydoc sz_rfind_byteset */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_sve2(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

#if SZ_USE_V128RELAXED
/** @copydoc sz_find_byteset */
SZ_API_COMPTIME sz_cptr_t sz_find_byteset_v128relaxed(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
/** @copydoc sz_rfind_byteset */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_v128relaxed(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

#if SZ_USE_V128
/** @copydoc sz_find_byteset */
SZ_API_COMPTIME sz_cptr_t sz_find_byteset_v128(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
/** @copydoc sz_rfind_byteset */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_v128(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

#if SZ_USE_RVV
/** @copydoc sz_find_byteset */
SZ_API_COMPTIME sz_cptr_t sz_find_byteset_rvv(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
/** @copydoc sz_rfind_byteset */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_rvv(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

#if SZ_USE_LASX
/** @copydoc sz_find_byteset */
SZ_API_COMPTIME sz_cptr_t sz_find_byteset_lasx(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
/** @copydoc sz_rfind_byteset */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_lasx(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

#if SZ_USE_POWERVSX
/** @copydoc sz_find_byteset */
SZ_API_COMPTIME sz_cptr_t sz_find_byteset_powervsx(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
/** @copydoc sz_rfind_byteset */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_powervsx(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

/*  `sz_utf8_delimiters` (UTF-8 punctuation/symbol/separator/whitespace enumeration) lives in
 *  "stringzilla/utf8_tokens.h" alongside its per-ISA backends and property tables. */

#pragma endregion // Core API

#pragma region Helper Shortcuts

/**
 *  @brief Finds the first byte in @p haystack that is present in @p needle.
 *  @param haystack String to be scanned.
 *  @param haystack_length Number of bytes in the haystack.
 *  @param needle String whose bytes form the accepted set.
 *  @param needle_length Number of bytes in the needle.
 *  @return Pointer to the first matching byte, or NULL if not found.
 */
SZ_API_COMPTIME sz_cptr_t sz_find_byte_from(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                            sz_size_t needle_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; needle_length; ++needle, --needle_length) sz_byteset_add(&set, *needle);
    return sz_find_byteset(haystack, haystack_length, &set);
}

/**
 *  @brief Finds the first byte in @p haystack that is NOT present in @p needle.
 *  @param haystack String to be scanned.
 *  @param haystack_length Number of bytes in the haystack.
 *  @param needle String whose bytes form the rejected set.
 *  @param needle_length Number of bytes in the needle.
 *  @return Pointer to the first non-matching byte, or NULL if not found.
 */
SZ_API_COMPTIME sz_cptr_t sz_find_byte_not_from(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                sz_size_t needle_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; needle_length; ++needle, --needle_length) sz_byteset_add(&set, *needle);
    sz_byteset_invert(&set);
    return sz_find_byteset(haystack, haystack_length, &set);
}

/**
 *  @brief Finds the last byte in @p haystack that is present in @p needle.
 *  @param haystack String to be scanned.
 *  @param haystack_length Number of bytes in the haystack.
 *  @param needle String whose bytes form the accepted set.
 *  @param needle_length Number of bytes in the needle.
 *  @return Pointer to the last matching byte, or NULL if not found.
 */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_from(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                             sz_size_t needle_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; needle_length; ++needle, --needle_length) sz_byteset_add(&set, *needle);
    return sz_rfind_byteset(haystack, haystack_length, &set);
}

/**
 *  @brief Finds the last byte in @p haystack that is NOT present in @p needle.
 *  @param haystack String to be scanned.
 *  @param haystack_length Number of bytes in the haystack.
 *  @param needle String whose bytes form the rejected set.
 *  @param needle_length Number of bytes in the needle.
 *  @return Pointer to the last non-matching byte, or NULL if not found.
 */
SZ_API_COMPTIME sz_cptr_t sz_rfind_byte_not_from(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                 sz_size_t needle_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; needle_length; ++needle, --needle_length) sz_byteset_add(&set, *needle);
    sz_byteset_invert(&set);
    return sz_rfind_byteset(haystack, haystack_length, &set);
}

#pragma endregion // Helper Shortcuts

#include "stringzilla/find/serial.h"
#include "stringzilla/find/westmere.h"
#include "stringzilla/find/haswell.h"
#include "stringzilla/find/skylake.h"
#include "stringzilla/find/icelake.h"
#include "stringzilla/find/neon.h"
#include "stringzilla/find/sve.h"
#include "stringzilla/find/sve2.h"
#include "stringzilla/find/v128relaxed.h"
#include "stringzilla/find/v128.h"
#include "stringzilla/find/rvv.h"
#include "stringzilla/find/lasx.h"
#include "stringzilla/find/powervsx.h"

/*  Pick the right implementation for the string search algorithms.
 *  To override this behavior and precompile all backends - set `SZ_DYNAMIC_DISPATCH` to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

#pragma region Core Functionality

SZ_API_RUNTIME sz_cptr_t sz_find_byte(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
#if SZ_USE_V128RELAXED
    return sz_find_byte_v128relaxed(haystack, haystack_length, needle);
#elif SZ_USE_V128
    return sz_find_byte_v128(haystack, haystack_length, needle);
#elif SZ_USE_RVV
    return sz_find_byte_rvv(haystack, haystack_length, needle);
#elif SZ_USE_LASX
    return sz_find_byte_lasx(haystack, haystack_length, needle);
#elif SZ_USE_POWERVSX
    return sz_find_byte_powervsx(haystack, haystack_length, needle);
#elif SZ_USE_SKYLAKE
    return sz_find_byte_skylake(haystack, haystack_length, needle);
#elif SZ_USE_HASWELL
    return sz_find_byte_haswell(haystack, haystack_length, needle);
#elif SZ_USE_WESTMERE
    return sz_find_byte_westmere(haystack, haystack_length, needle);
#elif SZ_USE_SVE // ? actually faster than NEON on most machines
    return sz_find_byte_sve(haystack, haystack_length, needle);
#elif SZ_USE_NEON
    return sz_find_byte_neon(haystack, haystack_length, needle);
#else
    return sz_find_byte_serial(haystack, haystack_length, needle);
#endif
}

SZ_API_RUNTIME sz_cptr_t sz_rfind_byte(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
#if SZ_USE_V128RELAXED
    return sz_rfind_byte_v128relaxed(haystack, haystack_length, needle);
#elif SZ_USE_V128
    return sz_rfind_byte_v128(haystack, haystack_length, needle);
#elif SZ_USE_RVV
    return sz_rfind_byte_rvv(haystack, haystack_length, needle);
#elif SZ_USE_LASX
    return sz_rfind_byte_lasx(haystack, haystack_length, needle);
#elif SZ_USE_POWERVSX
    return sz_rfind_byte_powervsx(haystack, haystack_length, needle);
#elif SZ_USE_SKYLAKE
    return sz_rfind_byte_skylake(haystack, haystack_length, needle);
#elif SZ_USE_HASWELL
    return sz_rfind_byte_haswell(haystack, haystack_length, needle);
#elif SZ_USE_WESTMERE
    return sz_rfind_byte_westmere(haystack, haystack_length, needle);
#elif SZ_USE_SVE // ? actually faster than NEON on most machines
    return sz_rfind_byte_sve(haystack, haystack_length, needle);
#elif SZ_USE_NEON
    return sz_rfind_byte_neon(haystack, haystack_length, needle);
#else
    return sz_rfind_byte_serial(haystack, haystack_length, needle);
#endif
}

SZ_API_RUNTIME sz_cptr_t sz_find(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                 sz_size_t needle_length) {
#if SZ_USE_V128RELAXED
    return sz_find_v128relaxed(haystack, haystack_length, needle, needle_length);
#elif SZ_USE_V128
    return sz_find_v128(haystack, haystack_length, needle, needle_length);
#elif SZ_USE_RVV
    return sz_find_rvv(haystack, haystack_length, needle, needle_length);
#elif SZ_USE_LASX
    return sz_find_lasx(haystack, haystack_length, needle, needle_length);
#elif SZ_USE_POWERVSX
    return sz_find_powervsx(haystack, haystack_length, needle, needle_length);
#elif SZ_USE_SKYLAKE
    return sz_find_skylake(haystack, haystack_length, needle, needle_length);
#elif SZ_USE_HASWELL
    return sz_find_haswell(haystack, haystack_length, needle, needle_length);
#elif SZ_USE_WESTMERE
    return sz_find_westmere(haystack, haystack_length, needle, needle_length);
#elif SZ_USE_SVE && SZ_SVE_WIDER_THAN_NEON_
    return sz_find_sve(haystack, haystack_length, needle, needle_length);
#elif SZ_USE_NEON
    return sz_find_neon(haystack, haystack_length, needle, needle_length);
#else
    return sz_find_serial(haystack, haystack_length, needle, needle_length);
#endif
}

SZ_API_RUNTIME sz_cptr_t sz_rfind(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                  sz_size_t needle_length) {
#if SZ_USE_V128RELAXED
    return sz_rfind_v128relaxed(haystack, haystack_length, needle, needle_length);
#elif SZ_USE_V128
    return sz_rfind_v128(haystack, haystack_length, needle, needle_length);
#elif SZ_USE_RVV
    return sz_rfind_rvv(haystack, haystack_length, needle, needle_length);
#elif SZ_USE_LASX
    return sz_rfind_lasx(haystack, haystack_length, needle, needle_length);
#elif SZ_USE_POWERVSX
    return sz_rfind_powervsx(haystack, haystack_length, needle, needle_length);
#elif SZ_USE_SKYLAKE
    return sz_rfind_skylake(haystack, haystack_length, needle, needle_length);
#elif SZ_USE_HASWELL
    return sz_rfind_haswell(haystack, haystack_length, needle, needle_length);
#elif SZ_USE_WESTMERE
    return sz_rfind_westmere(haystack, haystack_length, needle, needle_length);
#elif SZ_USE_SVE // ? actually faster than NEON on most machines
    return sz_rfind_sve(haystack, haystack_length, needle, needle_length);
#elif SZ_USE_NEON
    return sz_rfind_neon(haystack, haystack_length, needle, needle_length);
#else
    return sz_rfind_serial(haystack, haystack_length, needle, needle_length);
#endif
}

SZ_API_RUNTIME sz_cptr_t sz_find_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
#if SZ_USE_V128RELAXED
    return sz_find_byteset_v128relaxed(text, length, set);
#elif SZ_USE_V128
    return sz_find_byteset_v128(text, length, set);
#elif SZ_USE_RVV
    return sz_find_byteset_rvv(text, length, set);
#elif SZ_USE_LASX
    return sz_find_byteset_lasx(text, length, set);
#elif SZ_USE_POWERVSX
    return sz_find_byteset_powervsx(text, length, set);
#elif SZ_USE_ICELAKE
    return sz_find_byteset_icelake(text, length, set);
#elif SZ_USE_HASWELL
    return sz_find_byteset_haswell(text, length, set);
#elif SZ_USE_SVE2 // ? `MATCH` covers a whole small set per instruction
    return sz_find_byteset_sve2(text, length, set);
#elif SZ_USE_NEON
    return sz_find_byteset_neon(text, length, set);
#else
    return sz_find_byteset_serial(text, length, set);
#endif
}

SZ_API_RUNTIME sz_cptr_t sz_rfind_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
#if SZ_USE_V128RELAXED
    return sz_rfind_byteset_v128relaxed(text, length, set);
#elif SZ_USE_V128
    return sz_rfind_byteset_v128(text, length, set);
#elif SZ_USE_RVV
    return sz_rfind_byteset_rvv(text, length, set);
#elif SZ_USE_LASX
    return sz_rfind_byteset_lasx(text, length, set);
#elif SZ_USE_POWERVSX
    return sz_rfind_byteset_powervsx(text, length, set);
#elif SZ_USE_ICELAKE
    return sz_rfind_byteset_icelake(text, length, set);
#elif SZ_USE_HASWELL
    return sz_rfind_byteset_haswell(text, length, set);
#elif SZ_USE_SVE2 // ? `MATCH` covers a whole small set per instruction
    return sz_rfind_byteset_sve2(text, length, set);
#elif SZ_USE_NEON
    return sz_rfind_byteset_neon(text, length, set);
#else
    return sz_rfind_byteset_serial(text, length, set);
#endif
}

#pragma endregion
#endif            // !SZ_DYNAMIC_DISPATCH
#pragma endregion // Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_FIND_H_
