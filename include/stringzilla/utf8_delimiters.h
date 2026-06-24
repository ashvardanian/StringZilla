/**
 *  @brief Hardware-accelerated UTF-8 delimiter (punctuation/symbol/separator/whitespace) scanning.
 *  @file utf8_delimiters.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_DELIMITERS_H_
#define STRINGZILLA_UTF8_DELIMITERS_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Finds the first occurrence of a UTF-8 delimiter codepoint in a string.
 *
 *  A "delimiter" is any whitespace, punctuation, symbol or separator codepoint - precisely the codepoints
 *  whose Unicode General_Category is one of {Pc, Pd, Ps, Pe, Pi, Pf, Po, Sm, Sc, Sk, So, Zs, Zl, Zp},
 *  unioned with the White_Space property. This matches the set used by "Unicode UAX #29" word segmentation
 *  and implemented in the ICU. It is the Unicode-property analog of `sz_find_byteset`.
 *  - around 30 ASCII characters
 *  - around 130 2-byte characters
 *  - around 4.3k 3-byte characters
 *  - around 4.1k 4-byte characters
 *
 *  Scanning is bounds-checked: a truncated trailing UTF-8 sequence is never over-read, and any byte that does
 *  not begin a well-formed codepoint (lone continuation, overlong, surrogate, truncated tail) is skipped and
 *  never reported as a delimiter.
 *
 *  @param text String to be scanned.
 *  @param length Number of bytes in the string.
 *  @param matched_length Output: number of bytes in the matched delimiter codepoint (0 if none found).
 *  @return Pointer to the first matching delimiter codepoint from @p text, or NULL if none.
 */
SZ_DYNAMIC sz_cptr_t sz_find_delimiter_utf8(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);

/** @copydoc sz_find_delimiter_utf8 */
SZ_PUBLIC sz_cptr_t sz_find_delimiters_utf8_serial(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);

#if SZ_USE_HASWELL
/** @copydoc sz_find_delimiter_utf8 */
SZ_PUBLIC sz_cptr_t sz_find_delimiters_utf8_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
#endif

#if SZ_USE_ICELAKE
/** @copydoc sz_find_delimiter_utf8 */
SZ_PUBLIC sz_cptr_t sz_find_delimiters_utf8_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
#endif

#if SZ_USE_NEON
/** @copydoc sz_find_delimiter_utf8 */
SZ_PUBLIC sz_cptr_t sz_find_delimiters_utf8_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
#endif

#pragma endregion // Core API

#ifdef __cplusplus
}
#endif

#include "stringzilla/utf8_delimiters/serial.h"
#include "stringzilla/utf8_delimiters/haswell.h"
#include "stringzilla/utf8_delimiters/icelake.h"
#include "stringzilla/utf8_delimiters/neon.h"

#pragma region Dynamic Dispatch

#if !SZ_DYNAMIC_DISPATCH

#ifdef __cplusplus
extern "C" {
#endif

SZ_DYNAMIC sz_cptr_t sz_find_delimiter_utf8(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
#if SZ_USE_ICELAKE
    return sz_find_delimiters_utf8_icelake(text, length, matched_length);
#elif SZ_USE_HASWELL
    return sz_find_delimiters_utf8_haswell(text, length, matched_length);
#elif SZ_USE_NEON
    return sz_find_delimiters_utf8_neon(text, length, matched_length);
#else
    return sz_find_delimiters_utf8_serial(text, length, matched_length);
#endif
}

#ifdef __cplusplus
}
#endif

#endif // !SZ_DYNAMIC_DISPATCH

#pragma endregion // Dynamic Dispatch

#endif // STRINGZILLA_UTF8_DELIMITERS_H_
