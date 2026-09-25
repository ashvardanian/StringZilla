/**
 *  @file include/stringzilla/find.h
 *  @author Ash Vardanian
 *  @date August 14, 2020
 *  @brief Hardware-accelerated sub-string and character-set search utilities.
 *
 *  Includes core APIs:
 *
 *  - @c sz_find and reverse-order @c sz_rfind
 *  - @c sz_find_byte and reverse-order @c sz_rfind_byte
 *  - @c sz_find_byteset and reverse-order @c sz_rfind_byteset
 *
 *  Convenience functions for character-set matching:
 *
 *  - @c sz_find_byte_from shortcut for @c sz_find_byteset
 *  - @c sz_find_byte_not_from shortcut for @c sz_find_byteset with inverted set
 *  - @c sz_rfind_byte_from shortcut for @c sz_rfind_byteset
 *  - @c sz_rfind_byte_not_from shortcut for @c sz_rfind_byteset with inverted set
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
 *  @brief Locates the first byte equal to the one at @p needle in @p haystack, like @c memchr.
 *
 *  @param[in] haystack Haystack - the string to search in.
 *  @param[in] haystack_length Number of bytes in the haystack.
 *  @param[in] needle Needle - single-byte substring to find.
 *  @return Address of the first match, or NULL if not found.
 *
 *  @see x86_64 implementation in glibc: https://github.com/lattera/glibc/blob/master/sysdeps/x86_64/memchr.S
 *  @see AArch64 implementation in glibc: https://github.com/lattera/glibc/blob/master/sysdeps/aarch64/memchr.S
 */
STRINGZILLA_API_RUNTIME sz_cptr_t sz_find_byte(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);

/**
 *  @brief Locates the last byte equal to the one at @p needle in @p haystack, like @c memrchr.
 *
 *  @param[in] haystack Haystack - the string to search in.
 *  @param[in] haystack_length Number of bytes in the haystack.
 *  @param[in] needle Needle - single-byte substring to find.
 *  @return Address of the last match, or NULL if not found.
 *
 *  There is no AArch64 reference implementation to link here.
 *
 *  @see x86_64 implementation in glibc: https://github.com/lattera/glibc/blob/master/sysdeps/x86_64/memrchr.S
 */
STRINGZILLA_API_RUNTIME sz_cptr_t sz_rfind_byte(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);

/** @copydoc sz_find_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byte_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);

/** @copydoc sz_rfind_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byte_serial(sz_cptr_t haystack, sz_size_t haystack_length,
                                                        sz_cptr_t needle);

#if STRINGZILLA_TARGET_WESTMERE

/** @copydoc sz_find_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byte_westmere(sz_cptr_t haystack, sz_size_t haystack_length,
                                                         sz_cptr_t needle);

/** @copydoc sz_rfind_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byte_westmere(sz_cptr_t haystack, sz_size_t haystack_length,
                                                          sz_cptr_t needle);
#endif

#if STRINGZILLA_TARGET_HASWELL

/** @copydoc sz_find_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byte_haswell(sz_cptr_t haystack, sz_size_t haystack_length,
                                                        sz_cptr_t needle);

/** @copydoc sz_rfind_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byte_haswell(sz_cptr_t haystack, sz_size_t haystack_length,
                                                         sz_cptr_t needle);
#endif

#if STRINGZILLA_TARGET_SKYLAKE

/** @copydoc sz_find_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byte_skylake(sz_cptr_t haystack, sz_size_t haystack_length,
                                                        sz_cptr_t needle);

/** @copydoc sz_rfind_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byte_skylake(sz_cptr_t haystack, sz_size_t haystack_length,
                                                         sz_cptr_t needle);
#endif

#if STRINGZILLA_TARGET_NEON

/** @copydoc sz_find_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byte_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);

/** @copydoc sz_rfind_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byte_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
#endif

#if STRINGZILLA_TARGET_SVE

/** @copydoc sz_find_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byte_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);

/** @copydoc sz_rfind_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byte_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
#endif

#if STRINGZILLA_TARGET_V128RELAXED

/** @copydoc sz_find_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byte_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length,
                                                            sz_cptr_t needle);

/** @copydoc sz_rfind_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byte_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length,
                                                             sz_cptr_t needle);
#endif

#if STRINGZILLA_TARGET_V128

/** @copydoc sz_find_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byte_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);

/** @copydoc sz_rfind_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byte_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
#endif

#if STRINGZILLA_TARGET_RVV

/** @copydoc sz_find_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byte_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);

/** @copydoc sz_rfind_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byte_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
#endif

#if STRINGZILLA_TARGET_LASX

/** @copydoc sz_find_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byte_lasx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);

/** @copydoc sz_rfind_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byte_lasx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle);
#endif

#if STRINGZILLA_TARGET_POWERVSX

/** @copydoc sz_find_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byte_powervsx(sz_cptr_t haystack, sz_size_t haystack_length,
                                                         sz_cptr_t needle);

/** @copydoc sz_rfind_byte */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byte_powervsx(sz_cptr_t haystack, sz_size_t haystack_length,
                                                          sz_cptr_t needle);
#endif

/**
 *  @brief Locates the first occurrence of @p needle in @p haystack, like @c memmem in LibC, or like
 *      @c strstr for strings of known length.
 *
 *  @param[in] haystack Haystack - the string to search in.
 *  @param[in] haystack_length Number of bytes in the haystack.
 *  @param[in] needle Needle - substring to find.
 *  @param[in] needle_length Number of bytes in the needle.
 *  @return Address of the first match.
 */
STRINGZILLA_API_RUNTIME sz_cptr_t sz_find(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                          sz_size_t needle_length);

/**
 *  @brief Locates the last matching substring.
 *
 *  @param[in] haystack Haystack - the string to search in.
 *  @param[in] haystack_length Number of bytes in the haystack.
 *  @param[in] needle Needle - substring to find.
 *  @param[in] needle_length Number of bytes in the needle.
 *  @return Address of the last match.
 */
STRINGZILLA_API_RUNTIME sz_cptr_t sz_rfind(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                           sz_size_t needle_length);

/** @copydoc sz_find */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                  sz_size_t needle_length);

/** @copydoc sz_rfind */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                   sz_size_t needle_length);

#if STRINGZILLA_TARGET_WESTMERE

/** @copydoc sz_find */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_westmere(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                    sz_size_t needle_length);

/** @copydoc sz_rfind */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_westmere(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                     sz_size_t needle_length);
#endif

#if STRINGZILLA_TARGET_HASWELL

/** @copydoc sz_find */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                   sz_size_t needle_length);

/** @copydoc sz_rfind */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_haswell(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                    sz_size_t needle_length);
#endif

#if STRINGZILLA_TARGET_SKYLAKE

/** @copydoc sz_find */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_skylake(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                   sz_size_t needle_length);

/** @copydoc sz_rfind */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_skylake(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                    sz_size_t needle_length);
#endif

#if STRINGZILLA_TARGET_NEON

/** @copydoc sz_find */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                sz_size_t needle_length);

/** @copydoc sz_rfind */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                 sz_size_t needle_length);
#endif

#if STRINGZILLA_TARGET_SVE

/** @copydoc sz_find */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                               sz_size_t needle_length);

/** @copydoc sz_rfind */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_sve(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                sz_size_t needle_length);
#endif

#if STRINGZILLA_TARGET_V128RELAXED

/** @copydoc sz_find */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                       sz_size_t needle_length);

/** @copydoc sz_rfind */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_v128relaxed(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                        sz_size_t needle_length);
#endif

#if STRINGZILLA_TARGET_V128

/** @copydoc sz_find */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                sz_size_t needle_length);

/** @copydoc sz_rfind */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_v128(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                 sz_size_t needle_length);
#endif

#if STRINGZILLA_TARGET_RVV

/** @copydoc sz_find */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                               sz_size_t needle_length);

/** @copydoc sz_rfind */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_rvv(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                sz_size_t needle_length);
#endif

#if STRINGZILLA_TARGET_LASX

/** @copydoc sz_find */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_lasx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                sz_size_t needle_length);

/** @copydoc sz_rfind */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_lasx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                 sz_size_t needle_length);
#endif

#if STRINGZILLA_TARGET_POWERVSX

/** @copydoc sz_find */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_powervsx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                    sz_size_t needle_length);

/** @copydoc sz_rfind */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_powervsx(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                     sz_size_t needle_length);
#endif

/**
 *  @brief Finds the first character present from the @p set, present in @p text.
 *
 *  @param[in] text String to be scanned.
 *  @param[in] length Number of bytes in the string.
 *  @param[in] set Set of relevant characters.
 *  @return Pointer to the first matching character from @p set.
 *
 *  Equivalent to `strspn(text, accepted)` and `strcspn(text, rejected)` in LibC. May have identical
 *  implementation and performance to @c sz_rfind_byteset. Useful for parsing, when we want to skip
 *  a set of characters, such as:
 *
 *  - 6 whitespaces: " \t\n\r\v\f".
 *  - 16 digits forming a float number: "0123456789,.eE+-".
 *  - 5 HTML reserved characters: "\"'&<>", of which "<>" can be useful for parsing.
 *  - 2 JSON string special characters useful to locate the end of the string: "\"\\".
 */
STRINGZILLA_API_RUNTIME sz_cptr_t sz_find_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set);

/**
 *  @brief Finds the last character present from the @p set, present in @p text.
 *
 *  @param[in] text String to be scanned.
 *  @param[in] length Number of bytes in the string.
 *  @param[in] set Set of relevant characters.
 *  @return Pointer to the last matching character from @p set.
 *
 *  Equivalent to `strspn(text, accepted)` and `strcspn(text, rejected)` in LibC. May have identical
 *  implementation and performance to @c sz_find_byteset. Useful for parsing, when we want to skip a
 *  set of characters, such as:
 *
 *  - 6 whitespaces: " \t\n\r\v\f".
 *  - 16 digits forming a float number: "0123456789,.eE+-".
 *  - 5 HTML reserved characters: "\"'&<>", of which "<>" can be useful for parsing.
 *  - 2 JSON string special characters useful to locate the end of the string: "\"\\".
 */
STRINGZILLA_API_RUNTIME sz_cptr_t sz_rfind_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set);

/** @copydoc sz_find_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byteset_serial(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set);

/** @copydoc sz_rfind_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byteset_serial(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set);

#if STRINGZILLA_TARGET_HASWELL

/** @copydoc sz_find_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byteset_haswell(sz_cptr_t haystack, sz_size_t length,
                                                           sz_byteset_t const *set);

/** @copydoc sz_rfind_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byteset_haswell(sz_cptr_t haystack, sz_size_t length,
                                                            sz_byteset_t const *set);
#endif

#if STRINGZILLA_TARGET_ICELAKE

/** @copydoc sz_find_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byteset_icelake(sz_cptr_t haystack, sz_size_t length,
                                                           sz_byteset_t const *set);

/** @copydoc sz_rfind_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byteset_icelake(sz_cptr_t haystack, sz_size_t length,
                                                            sz_byteset_t const *set);
#endif

#if STRINGZILLA_TARGET_NEON

/** @copydoc sz_find_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byteset_neon(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);

/** @copydoc sz_rfind_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byteset_neon(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

#if STRINGZILLA_TARGET_SVE2

/** @copydoc sz_find_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byteset_sve2(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);

/** @copydoc sz_rfind_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byteset_sve2(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

#if STRINGZILLA_TARGET_V128RELAXED

/** @copydoc sz_find_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byteset_v128relaxed(sz_cptr_t haystack, sz_size_t length,
                                                               sz_byteset_t const *set);

/** @copydoc sz_rfind_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byteset_v128relaxed(sz_cptr_t haystack, sz_size_t length,
                                                                sz_byteset_t const *set);
#endif

#if STRINGZILLA_TARGET_V128

/** @copydoc sz_find_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byteset_v128(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);

/** @copydoc sz_rfind_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byteset_v128(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

#if STRINGZILLA_TARGET_RVV

/** @copydoc sz_find_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byteset_rvv(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);

/** @copydoc sz_rfind_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byteset_rvv(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

#if STRINGZILLA_TARGET_LASX

/** @copydoc sz_find_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byteset_lasx(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);

/** @copydoc sz_rfind_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byteset_lasx(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

#if STRINGZILLA_TARGET_POWERVSX

/** @copydoc sz_find_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byteset_powervsx(sz_cptr_t haystack, sz_size_t length,
                                                            sz_byteset_t const *set);

/** @copydoc sz_rfind_byteset */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byteset_powervsx(sz_cptr_t haystack, sz_size_t length,
                                                             sz_byteset_t const *set);
#endif

/*  @c sz_utf8_delimiters (UTF-8 punctuation/symbol/separator/whitespace enumeration) lives in
 *  "stringzilla/utf8_tokens.h" alongside its per-ISA backends and property tables. */

#pragma endregion Core API

#pragma region Helper Shortcuts

/**
 *  @brief Finds the first byte in @p haystack that is present in @p needle.
 *
 *  @param[in] haystack String to be scanned.
 *  @param[in] haystack_length Number of bytes in the haystack.
 *  @param[in] needle String whose bytes form the accepted set.
 *  @param[in] needle_length Number of bytes in the needle.
 *  @return Pointer to the first matching byte, or NULL if not found.
 */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byte_from(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                     sz_size_t needle_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; needle_length; ++needle, --needle_length) sz_byteset_add(&set, *needle);
    return sz_find_byteset(haystack, haystack_length, &set);
}

/**
 *  @brief Finds the first byte in @p haystack that is not present in @p needle.
 *
 *  @param[in] haystack String to be scanned.
 *  @param[in] haystack_length Number of bytes in the haystack.
 *  @param[in] needle String whose bytes form the rejected set.
 *  @param[in] needle_length Number of bytes in the needle.
 *  @return Pointer to the first non-matching byte, or NULL if not found.
 */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_find_byte_not_from(sz_cptr_t haystack, sz_size_t haystack_length,
                                                         sz_cptr_t needle, sz_size_t needle_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; needle_length; ++needle, --needle_length) sz_byteset_add(&set, *needle);
    sz_byteset_invert(&set);
    return sz_find_byteset(haystack, haystack_length, &set);
}

/**
 *  @brief Finds the last byte in @p haystack that is present in @p needle.
 *
 *  @param[in] haystack String to be scanned.
 *  @param[in] haystack_length Number of bytes in the haystack.
 *  @param[in] needle String whose bytes form the accepted set.
 *  @param[in] needle_length Number of bytes in the needle.
 *  @return Pointer to the last matching byte, or NULL if not found.
 */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byte_from(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                      sz_size_t needle_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; needle_length; ++needle, --needle_length) sz_byteset_add(&set, *needle);
    return sz_rfind_byteset(haystack, haystack_length, &set);
}

/**
 *  @brief Finds the last byte in @p haystack that is not present in @p needle.
 *
 *  @param[in] haystack String to be scanned.
 *  @param[in] haystack_length Number of bytes in the haystack.
 *  @param[in] needle String whose bytes form the rejected set.
 *  @param[in] needle_length Number of bytes in the needle.
 *  @return Pointer to the last non-matching byte, or NULL if not found.
 */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_rfind_byte_not_from(sz_cptr_t haystack, sz_size_t haystack_length,
                                                          sz_cptr_t needle, sz_size_t needle_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; needle_length; ++needle, --needle_length) sz_byteset_add(&set, *needle);
    sz_byteset_invert(&set);
    return sz_rfind_byteset(haystack, haystack_length, &set);
}

#pragma endregion Helper Shortcuts

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

/*  Pick the right implementation for the string search algorithms. To override this behavior and
 *  precompile all backends - set @c STRINGZILLA_RUNTIME_DISPATCH to 1. */
#pragma region Compile Time Dispatching
#if !STRINGZILLA_RUNTIME_DISPATCH

#pragma region Core Functionality

STRINGZILLA_API_RUNTIME sz_cptr_t sz_find_byte(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
#if STRINGZILLA_TARGET_V128RELAXED
    return sz_find_byte_v128relaxed(haystack, haystack_length, needle);
#elif STRINGZILLA_TARGET_V128
    return sz_find_byte_v128(haystack, haystack_length, needle);
#elif STRINGZILLA_TARGET_RVV
    return sz_find_byte_rvv(haystack, haystack_length, needle);
#elif STRINGZILLA_TARGET_LASX
    return sz_find_byte_lasx(haystack, haystack_length, needle);
#elif STRINGZILLA_TARGET_POWERVSX
    return sz_find_byte_powervsx(haystack, haystack_length, needle);
#elif STRINGZILLA_TARGET_SKYLAKE
    return sz_find_byte_skylake(haystack, haystack_length, needle);
#elif STRINGZILLA_TARGET_HASWELL
    return sz_find_byte_haswell(haystack, haystack_length, needle);
#elif STRINGZILLA_TARGET_WESTMERE
    return sz_find_byte_westmere(haystack, haystack_length, needle);
#elif STRINGZILLA_TARGET_SVE // ? actually faster than NEON on most machines
    return sz_find_byte_sve(haystack, haystack_length, needle);
#elif STRINGZILLA_TARGET_NEON
    return sz_find_byte_neon(haystack, haystack_length, needle);
#else
    return sz_find_byte_serial(haystack, haystack_length, needle);
#endif
}

STRINGZILLA_API_RUNTIME sz_cptr_t sz_rfind_byte(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
#if STRINGZILLA_TARGET_V128RELAXED
    return sz_rfind_byte_v128relaxed(haystack, haystack_length, needle);
#elif STRINGZILLA_TARGET_V128
    return sz_rfind_byte_v128(haystack, haystack_length, needle);
#elif STRINGZILLA_TARGET_RVV
    return sz_rfind_byte_rvv(haystack, haystack_length, needle);
#elif STRINGZILLA_TARGET_LASX
    return sz_rfind_byte_lasx(haystack, haystack_length, needle);
#elif STRINGZILLA_TARGET_POWERVSX
    return sz_rfind_byte_powervsx(haystack, haystack_length, needle);
#elif STRINGZILLA_TARGET_SKYLAKE
    return sz_rfind_byte_skylake(haystack, haystack_length, needle);
#elif STRINGZILLA_TARGET_HASWELL
    return sz_rfind_byte_haswell(haystack, haystack_length, needle);
#elif STRINGZILLA_TARGET_WESTMERE
    return sz_rfind_byte_westmere(haystack, haystack_length, needle);
#elif STRINGZILLA_TARGET_SVE // ? actually faster than NEON on most machines
    return sz_rfind_byte_sve(haystack, haystack_length, needle);
#elif STRINGZILLA_TARGET_NEON
    return sz_rfind_byte_neon(haystack, haystack_length, needle);
#else
    return sz_rfind_byte_serial(haystack, haystack_length, needle);
#endif
}

STRINGZILLA_API_RUNTIME sz_cptr_t sz_find(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                          sz_size_t needle_length) {
#if STRINGZILLA_TARGET_V128RELAXED
    return sz_find_v128relaxed(haystack, haystack_length, needle, needle_length);
#elif STRINGZILLA_TARGET_V128
    return sz_find_v128(haystack, haystack_length, needle, needle_length);
#elif STRINGZILLA_TARGET_RVV
    return sz_find_rvv(haystack, haystack_length, needle, needle_length);
#elif STRINGZILLA_TARGET_LASX
    return sz_find_lasx(haystack, haystack_length, needle, needle_length);
#elif STRINGZILLA_TARGET_POWERVSX
    return sz_find_powervsx(haystack, haystack_length, needle, needle_length);
#elif STRINGZILLA_TARGET_SKYLAKE
    return sz_find_skylake(haystack, haystack_length, needle, needle_length);
#elif STRINGZILLA_TARGET_HASWELL
    return sz_find_haswell(haystack, haystack_length, needle, needle_length);
#elif STRINGZILLA_TARGET_WESTMERE
    return sz_find_westmere(haystack, haystack_length, needle, needle_length);
#elif STRINGZILLA_TARGET_SVE && STRINGZILLA_SVE_WIDER_THAN_NEON_
    return sz_find_sve(haystack, haystack_length, needle, needle_length);
#elif STRINGZILLA_TARGET_NEON
    return sz_find_neon(haystack, haystack_length, needle, needle_length);
#else
    return sz_find_serial(haystack, haystack_length, needle, needle_length);
#endif
}

STRINGZILLA_API_RUNTIME sz_cptr_t sz_rfind(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                           sz_size_t needle_length) {
#if STRINGZILLA_TARGET_V128RELAXED
    return sz_rfind_v128relaxed(haystack, haystack_length, needle, needle_length);
#elif STRINGZILLA_TARGET_V128
    return sz_rfind_v128(haystack, haystack_length, needle, needle_length);
#elif STRINGZILLA_TARGET_RVV
    return sz_rfind_rvv(haystack, haystack_length, needle, needle_length);
#elif STRINGZILLA_TARGET_LASX
    return sz_rfind_lasx(haystack, haystack_length, needle, needle_length);
#elif STRINGZILLA_TARGET_POWERVSX
    return sz_rfind_powervsx(haystack, haystack_length, needle, needle_length);
#elif STRINGZILLA_TARGET_SKYLAKE
    return sz_rfind_skylake(haystack, haystack_length, needle, needle_length);
#elif STRINGZILLA_TARGET_HASWELL
    return sz_rfind_haswell(haystack, haystack_length, needle, needle_length);
#elif STRINGZILLA_TARGET_WESTMERE
    return sz_rfind_westmere(haystack, haystack_length, needle, needle_length);
#elif STRINGZILLA_TARGET_SVE // ? actually faster than NEON on most machines
    return sz_rfind_sve(haystack, haystack_length, needle, needle_length);
#elif STRINGZILLA_TARGET_NEON
    return sz_rfind_neon(haystack, haystack_length, needle, needle_length);
#else
    return sz_rfind_serial(haystack, haystack_length, needle, needle_length);
#endif
}

STRINGZILLA_API_RUNTIME sz_cptr_t sz_find_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
#if STRINGZILLA_TARGET_V128RELAXED
    return sz_find_byteset_v128relaxed(text, length, set);
#elif STRINGZILLA_TARGET_V128
    return sz_find_byteset_v128(text, length, set);
#elif STRINGZILLA_TARGET_RVV
    return sz_find_byteset_rvv(text, length, set);
#elif STRINGZILLA_TARGET_LASX
    return sz_find_byteset_lasx(text, length, set);
#elif STRINGZILLA_TARGET_POWERVSX
    return sz_find_byteset_powervsx(text, length, set);
#elif STRINGZILLA_TARGET_ICELAKE
    return sz_find_byteset_icelake(text, length, set);
#elif STRINGZILLA_TARGET_HASWELL
    return sz_find_byteset_haswell(text, length, set);
#elif STRINGZILLA_TARGET_SVE2 // ? `MATCH` covers a whole small set per instruction
    return sz_find_byteset_sve2(text, length, set);
#elif STRINGZILLA_TARGET_NEON
    return sz_find_byteset_neon(text, length, set);
#else
    return sz_find_byteset_serial(text, length, set);
#endif
}

STRINGZILLA_API_RUNTIME sz_cptr_t sz_rfind_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
#if STRINGZILLA_TARGET_V128RELAXED
    return sz_rfind_byteset_v128relaxed(text, length, set);
#elif STRINGZILLA_TARGET_V128
    return sz_rfind_byteset_v128(text, length, set);
#elif STRINGZILLA_TARGET_RVV
    return sz_rfind_byteset_rvv(text, length, set);
#elif STRINGZILLA_TARGET_LASX
    return sz_rfind_byteset_lasx(text, length, set);
#elif STRINGZILLA_TARGET_POWERVSX
    return sz_rfind_byteset_powervsx(text, length, set);
#elif STRINGZILLA_TARGET_ICELAKE
    return sz_rfind_byteset_icelake(text, length, set);
#elif STRINGZILLA_TARGET_HASWELL
    return sz_rfind_byteset_haswell(text, length, set);
#elif STRINGZILLA_TARGET_SVE2 // ? `MATCH` covers a whole small set per instruction
    return sz_rfind_byteset_sve2(text, length, set);
#elif STRINGZILLA_TARGET_NEON
    return sz_rfind_byteset_neon(text, length, set);
#else
    return sz_rfind_byteset_serial(text, length, set);
#endif
}

#pragma endregion
#endif // !STRINGZILLA_RUNTIME_DISPATCH
#pragma endregion Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_FIND_H_
