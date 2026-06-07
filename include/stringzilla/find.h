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
 *  @brief Locates first matching byte in a string. Equivalent to `memchr(haystack, *needle, h_length)` in LibC.
 *
 *  @see X86_64 implementation: https://github.com/lattera/glibc/blob/master/sysdeps/x86_64/memchr.S
 *  @see Aarch64 implementation: https://github.com/lattera/glibc/blob/master/sysdeps/aarch64/memchr.S
 *
 *  @param[in] haystack Haystack - the string to search in.
 *  @param[in] h_length Number of bytes in the haystack.
 *  @param[in] needle Needle - single-byte substring to find.
 *  @return Address of the first match. NULL if not found.
 */
SZ_DYNAMIC sz_cptr_t sz_find_byte(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);

/**
 *  @brief Locates last matching byte in a string. Equivalent to `memrchr(haystack, *needle, h_length)` in LibC.
 *
 *  @see X86_64 implementation: https://github.com/lattera/glibc/blob/master/sysdeps/x86_64/memrchr.S
 *  @see Aarch64 implementation: missing
 *
 *  @param[in] haystack Haystack - the string to search in.
 *  @param[in] h_length Number of bytes in the haystack.
 *  @param[in] needle Needle - single-byte substring to find.
 *  @return Address of the last match. NULL if not found.
 */
SZ_DYNAMIC sz_cptr_t sz_rfind_byte(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);

/** @copydoc sz_find_byte */
SZ_PUBLIC sz_cptr_t sz_find_byte_serial(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_PUBLIC sz_cptr_t sz_rfind_byte_serial(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);

#if SZ_USE_WESTMERE
/** @copydoc sz_find_byte */
SZ_PUBLIC sz_cptr_t sz_find_byte_westmere(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_PUBLIC sz_cptr_t sz_rfind_byte_westmere(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);
#endif

#if SZ_USE_HASWELL
/** @copydoc sz_find_byte */
SZ_PUBLIC sz_cptr_t sz_find_byte_haswell(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_PUBLIC sz_cptr_t sz_rfind_byte_haswell(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);
#endif

#if SZ_USE_SKYLAKE
/** @copydoc sz_find_byte */
SZ_PUBLIC sz_cptr_t sz_find_byte_skylake(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_PUBLIC sz_cptr_t sz_rfind_byte_skylake(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);
#endif

#if SZ_USE_NEON
/** @copydoc sz_find_byte */
SZ_PUBLIC sz_cptr_t sz_find_byte_neon(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_PUBLIC sz_cptr_t sz_rfind_byte_neon(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);
#endif

/**
 *  @brief Locates first matching substring.
 *         Equivalent to `memmem(haystack, h_length, needle, n_length)` in LibC.
 *         Similar to `strstr(haystack, needle)` in LibC, but requires known length.
 *
 *  @param[in] haystack Haystack - the string to search in.
 *  @param[in] h_length Number of bytes in the haystack.
 *  @param[in] needle Needle - substring to find.
 *  @param[in] n_length Number of bytes in the needle.
 *  @return Address of the first match.
 */
SZ_DYNAMIC sz_cptr_t sz_find(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);

/**
 *  @brief Locates the last matching substring.
 *
 *  @param[in] haystack Haystack - the string to search in.
 *  @param[in] h_length Number of bytes in the haystack.
 *  @param[in] needle Needle - substring to find.
 *  @param[in] n_length Number of bytes in the needle.
 *  @return Address of the last match.
 */
SZ_DYNAMIC sz_cptr_t sz_rfind(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);

/** @copydoc sz_find */
SZ_PUBLIC sz_cptr_t sz_find_serial(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);
/** @copydoc sz_rfind */
SZ_PUBLIC sz_cptr_t sz_rfind_serial(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);

#if SZ_USE_WESTMERE
/** @copydoc sz_find */
SZ_PUBLIC sz_cptr_t sz_find_westmere(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);
/** @copydoc sz_rfind */
SZ_PUBLIC sz_cptr_t sz_rfind_westmere(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);
#endif

#if SZ_USE_HASWELL
/** @copydoc sz_find */
SZ_PUBLIC sz_cptr_t sz_find_haswell(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);
/** @copydoc sz_rfind */
SZ_PUBLIC sz_cptr_t sz_rfind_haswell(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);
#endif

#if SZ_USE_SKYLAKE
/** @copydoc sz_find */
SZ_PUBLIC sz_cptr_t sz_find_skylake(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);
/** @copydoc sz_rfind */
SZ_PUBLIC sz_cptr_t sz_rfind_skylake(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);
#endif

#if SZ_USE_NEON
/** @copydoc sz_find */
SZ_PUBLIC sz_cptr_t sz_find_neon(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);
/** @copydoc sz_rfind */
SZ_PUBLIC sz_cptr_t sz_rfind_neon(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);
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
 *  @param[in] text String to be scanned.
 *  @param[in] set Set of relevant characters.
 *  @return Pointer to the first matching character from @p set.
 */
SZ_DYNAMIC sz_cptr_t sz_find_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set);

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
 *  @param[in] text String to be scanned.
 *  @param[in] set Set of relevant characters.
 *  @return Pointer to the last matching character from @p set.
 */
SZ_DYNAMIC sz_cptr_t sz_rfind_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set);

/** @copydoc sz_find_byteset */
SZ_PUBLIC sz_cptr_t sz_find_byteset_serial(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set);
/** @copydoc sz_rfind_byteset */
SZ_PUBLIC sz_cptr_t sz_rfind_byteset_serial(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set);

#if SZ_USE_HASWELL
/** @copydoc sz_find_byteset */
SZ_PUBLIC sz_cptr_t sz_find_byteset_haswell(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
/** @copydoc sz_rfind_byteset */
SZ_PUBLIC sz_cptr_t sz_rfind_byteset_haswell(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

#if SZ_USE_ICELAKE
/** @copydoc sz_find_byteset */
SZ_PUBLIC sz_cptr_t sz_find_byteset_icelake(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
/** @copydoc sz_rfind_byteset */
SZ_PUBLIC sz_cptr_t sz_rfind_byteset_icelake(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

#if SZ_USE_NEON
/** @copydoc sz_find_byteset */
SZ_PUBLIC sz_cptr_t sz_find_byteset_neon(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
/** @copydoc sz_rfind_byteset */
SZ_PUBLIC sz_cptr_t sz_rfind_byteset_neon(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

/**
 *  @brief Finds the first occurrence of a UTF-8 whitespace or punctuation character in a string.
 *
 *  Delimiters include all of the above, plus common "punctuation" characters, "symbols", and "separators",
 *  as defined by the "Unicode UAX #29" word segmentation standard and implemented in the ICU.
 *  - around 30 ASCII characters
 *  - around 130 2-byte characters
 *  - around 4.3k 3-byte characters
 *  - around 4.1k 4-byte characters
 *
 *  @param[in] text String to be scanned.
 *  @param[in] length Number of bytes in the string.
 *  @param[out] matched_length Number of bytes in the matched newline delimiter.
 *  @return Pointer to the first matching newline character from @p text.
 */
SZ_DYNAMIC sz_cptr_t sz_find_delimiter_utf8(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);

/** @copydoc sz_find_delimiters_utf8 */
SZ_PUBLIC sz_cptr_t sz_find_delimiters_utf8_serial(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);

#if SZ_USE_HASWELL
/** @copydoc sz_find_delimiters_utf8 */
SZ_PUBLIC sz_cptr_t sz_find_delimiters_utf8_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
#endif

#if SZ_USE_ICELAKE
/** @copydoc sz_find_delimiters_utf8 */
SZ_PUBLIC sz_cptr_t sz_find_delimiters_utf8_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
#endif

#if SZ_USE_NEON
/** @copydoc sz_find_delimiters_utf8 */
SZ_PUBLIC sz_cptr_t sz_find_delimiters_utf8_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
#endif

#pragma endregion // Core API

#pragma region Helper Shortcuts

SZ_PUBLIC sz_cptr_t sz_find_byte_from(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; n_length; ++n, --n_length) sz_byteset_add(&set, *n);
    return sz_find_byteset(h, h_length, &set);
}

SZ_PUBLIC sz_cptr_t sz_find_byte_not_from(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; n_length; ++n, --n_length) sz_byteset_add(&set, *n);
    sz_byteset_invert(&set);
    return sz_find_byteset(h, h_length, &set);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byte_from(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; n_length; ++n, --n_length) sz_byteset_add(&set, *n);
    return sz_rfind_byteset(h, h_length, &set);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byte_not_from(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; n_length; ++n, --n_length) sz_byteset_add(&set, *n);
    sz_byteset_invert(&set);
    return sz_rfind_byteset(h, h_length, &set);
}

#pragma endregion // Helper Shortcuts

#include "stringzilla/find/serial.h"
#include "stringzilla/find/westmere.h"
#include "stringzilla/find/haswell.h"
#include "stringzilla/find/skylake.h"
#include "stringzilla/find/icelake.h"
#include "stringzilla/find/neon.h"
#include "stringzilla/find/sve.h"
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

SZ_DYNAMIC sz_cptr_t sz_find_byte(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle) {
#if SZ_USE_V128RELAXED
    return sz_find_byte_v128relaxed(haystack, h_length, needle);
#elif SZ_USE_V128
    return sz_find_byte_v128(haystack, h_length, needle);
#elif SZ_USE_RVV
    return sz_find_byte_rvv(haystack, h_length, needle);
#elif SZ_USE_LASX
    return sz_find_byte_lasx(haystack, h_length, needle);
#elif SZ_USE_POWERVSX
    return sz_find_byte_powervsx(haystack, h_length, needle);
#elif SZ_USE_SKYLAKE
    return sz_find_byte_skylake(haystack, h_length, needle);
#elif SZ_USE_HASWELL
    return sz_find_byte_haswell(haystack, h_length, needle);
#elif SZ_USE_WESTMERE
    return sz_find_byte_westmere(haystack, h_length, needle);
#elif SZ_USE_SVE // ? actually faster than NEON on most machines
    return sz_find_byte_sve(haystack, h_length, needle);
#elif SZ_USE_NEON
    return sz_find_byte_neon(haystack, h_length, needle);
#else
    return sz_find_byte_serial(haystack, h_length, needle);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_rfind_byte(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle) {
#if SZ_USE_V128RELAXED
    return sz_rfind_byte_v128relaxed(haystack, h_length, needle);
#elif SZ_USE_V128
    return sz_rfind_byte_v128(haystack, h_length, needle);
#elif SZ_USE_RVV
    return sz_rfind_byte_rvv(haystack, h_length, needle);
#elif SZ_USE_LASX
    return sz_rfind_byte_lasx(haystack, h_length, needle);
#elif SZ_USE_POWERVSX
    return sz_rfind_byte_powervsx(haystack, h_length, needle);
#elif SZ_USE_SKYLAKE
    return sz_rfind_byte_skylake(haystack, h_length, needle);
#elif SZ_USE_HASWELL
    return sz_rfind_byte_haswell(haystack, h_length, needle);
#elif SZ_USE_WESTMERE
    return sz_rfind_byte_westmere(haystack, h_length, needle);
#elif SZ_USE_SVE // ? actually faster than NEON on most machines
    return sz_rfind_byte_sve(haystack, h_length, needle);
#elif SZ_USE_NEON
    return sz_rfind_byte_neon(haystack, h_length, needle);
#else
    return sz_rfind_byte_serial(haystack, h_length, needle);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_find(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length) {
#if SZ_USE_V128RELAXED
    return sz_find_v128relaxed(haystack, h_length, needle, n_length);
#elif SZ_USE_V128
    return sz_find_v128(haystack, h_length, needle, n_length);
#elif SZ_USE_RVV
    return sz_find_rvv(haystack, h_length, needle, n_length);
#elif SZ_USE_LASX
    return sz_find_lasx(haystack, h_length, needle, n_length);
#elif SZ_USE_POWERVSX
    return sz_find_powervsx(haystack, h_length, needle, n_length);
#elif SZ_USE_SKYLAKE
    return sz_find_skylake(haystack, h_length, needle, n_length);
#elif SZ_USE_HASWELL
    return sz_find_haswell(haystack, h_length, needle, n_length);
#elif SZ_USE_WESTMERE
    return sz_find_westmere(haystack, h_length, needle, n_length);
#elif SZ_USE_SVE && SZ_ENFORCE_SVE_OVER_NEON
    return sz_find_sve(haystack, h_length, needle, n_length);
#elif SZ_USE_NEON
    return sz_find_neon(haystack, h_length, needle, n_length);
#else
    return sz_find_serial(haystack, h_length, needle, n_length);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_rfind(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length) {
#if SZ_USE_V128RELAXED
    return sz_rfind_v128relaxed(haystack, h_length, needle, n_length);
#elif SZ_USE_V128
    return sz_rfind_v128(haystack, h_length, needle, n_length);
#elif SZ_USE_RVV
    return sz_rfind_rvv(haystack, h_length, needle, n_length);
#elif SZ_USE_LASX
    return sz_rfind_lasx(haystack, h_length, needle, n_length);
#elif SZ_USE_POWERVSX
    return sz_rfind_powervsx(haystack, h_length, needle, n_length);
#elif SZ_USE_SKYLAKE
    return sz_rfind_skylake(haystack, h_length, needle, n_length);
#elif SZ_USE_HASWELL
    return sz_rfind_haswell(haystack, h_length, needle, n_length);
#elif SZ_USE_WESTMERE
    return sz_rfind_westmere(haystack, h_length, needle, n_length);
#elif SZ_USE_NEON
    return sz_rfind_neon(haystack, h_length, needle, n_length);
#else
    return sz_rfind_serial(haystack, h_length, needle, n_length);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_find_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
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
#elif SZ_USE_NEON
    return sz_find_byteset_neon(text, length, set);
#else
    return sz_find_byteset_serial(text, length, set);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_rfind_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
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
