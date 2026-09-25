/**
 *  @file include/stringzilla/utf8_uncased_fold.h
 *  @author Ash Vardanian
 *  @date November 23, 2025
 *  @brief Hardware-accelerated UTF-8 case folding.
 *
 *  Includes core APIs:
 *
 *  - @c sz_utf8_uncased_fold - Unicode case folding for uncased comparisons
 */
#ifndef STRINGZILLA_UTF8_UNCASED_FOLD_H_
#define STRINGZILLA_UTF8_UNCASED_FOLD_H_

#include "stringzilla/utf8_runes/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Apply Unicode case folding to a UTF-8 string.
 *
 *  Case folding normalizes text for uncased comparisons by mapping uppercase letters to their
 *  lowercase equivalents and handling special expansions defined in Unicode CaseFolding.txt.
 *
 *  @section utf8_uncased_fold_buffer_sizing Buffer Sizing
 *
 *  The destination buffer must be at least `source_length * 3` bytes to guarantee sufficient space
 *  for worst-case expansion. The maximum expansion ratio is 3:1, which occurs with Greek characters
 *  that expand to three codepoints under case folding. For example, 'ΐ' (U+0390, CE 90) folds to
 *  "ΐ" (U+03B9 U+0308 U+0301, CE B9 CC 88 CC 81), so 2 bytes → 6 bytes, and a string of N such
 *  characters expands from 2N to 6N bytes.
 *
 *  Malformed UTF-8 is handled losslessly: any byte that does not begin a well-formed codepoint is
 *  copied through unchanged (it folds to itself) and folding resyncs at the next byte. Valid UTF-8
 *  folds as usual. Basic usage:
 *
 *  @code{.c}
 *      char const *source = "HELLO";
 *      sz_size_t capacity = 5 * 3; // Safe overestimate
 *      char destination[15];
 *      sz_size_t result_length = sz_utf8_uncased_fold(source, 5, destination);
 *      // destination now contains "hello", result_length = 5
 *  @endcode
 *
 *  @param[in] source UTF-8 string to be case-folded.
 *  @param[in] source_length Number of bytes in the source buffer.
 *  @param[out] destination Buffer to write the case-folded UTF-8 string.
 *  @return Number of bytes written to the destination buffer.
 *  @warning The caller must ensure the destination buffer is large enough. No bounds checking is
 *      performed. Use `source_length * 3` for safety.
 */
STRINGZILLA_API_RUNTIME sz_size_t sz_utf8_uncased_fold( //
    sz_cptr_t source, sz_size_t source_length,          //
    sz_ptr_t destination);

/** @copydoc sz_utf8_uncased_fold */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_uncased_fold_serial( //
    sz_cptr_t source, sz_size_t source_length,                  //
    sz_ptr_t destination);

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_utf8_uncased_fold */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_uncased_fold_icelake( //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination);
#endif

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_utf8_uncased_fold */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_uncased_fold_haswell( //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_utf8_uncased_fold */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_uncased_fold_neon( //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination);
#endif

#if STRINGZILLA_TARGET_SVE2
/** @copydoc sz_utf8_uncased_fold */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_uncased_fold_sve2( //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination);
#endif

#if STRINGZILLA_TARGET_V128
/** @copydoc sz_utf8_uncased_fold */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_uncased_fold_v128( //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination);
#endif

#if STRINGZILLA_TARGET_RVV
/** @copydoc sz_utf8_uncased_fold */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_uncased_fold_rvv( //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination);
#endif

#if STRINGZILLA_TARGET_LASX
/** @copydoc sz_utf8_uncased_fold */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_uncased_fold_lasx( //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination);
#endif

#if STRINGZILLA_TARGET_POWERVSX
/** @copydoc sz_utf8_uncased_fold */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_uncased_fold_powervsx( //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination);
#endif

#pragma endregion

#pragma region Backends

#include "stringzilla/utf8_uncased_fold/serial.h"
#include "stringzilla/utf8_uncased_fold/icelake.h"
#include "stringzilla/utf8_uncased_fold/haswell.h"
#include "stringzilla/utf8_uncased_fold/neon.h"
#include "stringzilla/utf8_uncased_fold/sve2.h"
#include "stringzilla/utf8_uncased_fold/v128.h"
#include "stringzilla/utf8_uncased_fold/rvv.h"
#include "stringzilla/utf8_uncased_fold/lasx.h"
#include "stringzilla/utf8_uncased_fold/powervsx.h"

#pragma endregion

#pragma region Dynamic Dispatch

#if !STRINGZILLA_RUNTIME_DISPATCH

STRINGZILLA_API_RUNTIME sz_size_t sz_utf8_uncased_fold(sz_cptr_t source, sz_size_t source_length,
                                                       sz_ptr_t destination) {
#if STRINGZILLA_TARGET_ICELAKE
    return sz_utf8_uncased_fold_icelake(source, source_length, destination);
#elif STRINGZILLA_TARGET_HASWELL
    return sz_utf8_uncased_fold_haswell(source, source_length, destination);
#elif STRINGZILLA_TARGET_SVE2 && STRINGZILLA_SVE_WIDER_THAN_NEON_
    return sz_utf8_uncased_fold_sve2(source, source_length, destination);
#elif STRINGZILLA_TARGET_NEON
    return sz_utf8_uncased_fold_neon(source, source_length, destination);
#elif STRINGZILLA_TARGET_V128
    return sz_utf8_uncased_fold_v128(source, source_length, destination);
#elif STRINGZILLA_TARGET_RVV
    return sz_utf8_uncased_fold_rvv(source, source_length, destination);
#elif STRINGZILLA_TARGET_LASX
    return sz_utf8_uncased_fold_lasx(source, source_length, destination);
#elif STRINGZILLA_TARGET_POWERVSX
    return sz_utf8_uncased_fold_powervsx(source, source_length, destination);
#else
    return sz_utf8_uncased_fold_serial(source, source_length, destination);
#endif
}

#endif // !STRINGZILLA_RUNTIME_DISPATCH

#pragma endregion

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_UNCASED_FOLD_H_
