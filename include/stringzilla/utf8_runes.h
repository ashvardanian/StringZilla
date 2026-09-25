/**
 *  @file include/stringzilla/utf8_runes.h
 *  @author Ash Vardanian
 *  @date November 19, 2025
 *  @brief Hardware-accelerated UTF-8 codepoint mechanics: count, find-nth, and chunk unpacking.
 */
#ifndef STRINGZILLA_UTF8_RUNES_H_
#define STRINGZILLA_UTF8_RUNES_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Count the number of UTF-8 characters in a string.
 *
 *  The logic is to count the number of "continuation bytes" matching the 10xxxxxx pattern, and then
 *  subtract that from the total byte length to get the number of "start bytes" - coinciding with
 *  the number of UTF-8 characters. Counting the characters of a string:
 *
 *  @code{.c}
 *      size_t char_count = sz_utf8_count(text, length);
 *      printf("String has %zu characters\n", char_count);
 *  @endcode
 *
 *  @param[in] text String to be scanned.
 *  @param[in] length Number of bytes in the string.
 *  @return Number of UTF-8 characters in the string.
 */
STRINGZILLA_API_RUNTIME sz_size_t sz_utf8_count(sz_cptr_t text, sz_size_t length);

/**
 *  @brief Skip forward to the Nth UTF-8 character.
 *
 *  Skipping to character 1000, e.g. for pagination:
 *
 *  @code{.c}
 *      char const *pos = sz_utf8_seek(text, length, 1000);
 *      if (!pos) {
 *          // String has fewer than 1000 characters
 *      }
 *  @endcode
 *
 *  Truncating to 280 characters, Twitter-style:
 *
 *  @code{.c}
 *      char const *end = sz_utf8_seek(text, length, 280);
 *      size_t truncated_bytes = end ? (end - text) : length;
 *  @endcode
 *
 *  @param[in] text String to be scanned.
 *  @param[in] length Number of bytes in the string.
 *  @param[in] n Number of UTF-8 characters to skip, 0-indexed, so `n = 0` returns @p text.
 *  @return Pointer to the Nth character, or NULL if the string has fewer than @p n characters.
 */
STRINGZILLA_API_RUNTIME sz_cptr_t sz_utf8_seek(sz_cptr_t text, sz_size_t length, sz_size_t n);

/**
 *  @brief Unpack a UTF-8 string into UTF-32 codepoints.
 *
 *  This function is designed for streaming-like decoding with smart iterators built on top of it.
 *  The iterator would unpack a continuous slice of UTF-8 text into UTF-32 codepoints in chunks,
 *  yielding them upstream - only one at a time. This avoids allocating large buffers for the entire
 *  UTF-32 string, which can be 4x the size of the UTF-8 input.
 *
 *  This functionality is similar to the @c simdutf library's UTF-8 to UTF-32 conversion routines,
 *  and leverages an assumption that the absolute majority of written text doesn't mix codepoints of
 *  every length in each register-sized chunk:
 *
 *  - English text and source code is predominantly 1-byte ASCII characters.
 *  - Broader European languages with diacritics: mostly 2-byte characters, 1-byte punctuation.
 *  - Chinese & Japanese mostly use 3-byte characters with rare 1- or 3-byte punctuation.
 *  - Korean uses 3-byte characters with 1-byte spaces; word are 2-6 syllables or 6-16 bytes.
 *
 *  It's a different story for emoji-heavy texts, which can mix 4-byte characters more frequently.
 *
 *  Every backend shares one contract, the basis for every codepoint iterator built on it:
 *
 *  - @b Fill-or-drain: emits runes until @p runes_capacity is reached or @p text is exhausted,
 *    looping internally - one call fills the buffer however many byte-widths the text mixes.
 *  - @b Total @b and @b safe: never reads past `text + length`; substitutes one @b U+FFFD per
 *    maximal ill-formed subpart (overlong, surrogate, out-of-range, or a stray byte) and resyncs.
 *  - @b Valid @b scalar @b values: every emitted rune is a Unicode scalar value (incl. U+FFFD),
 *    so callers may convert without re-validation.
 *  - @b Resumable @b truncation: a well-formed but truncated trailing prefix is left unconsumed,
 *    the cursor stopping before it, so a streaming caller resumes once more bytes arrive.
 *
 *  @param[in] text UTF-8 string to unpack.
 *  @param[in] length Number of bytes in the string.
 *  @param[out] runes Buffer for UTF-32 codepoints, recommended to be at least @b 64 entries wide.
 *  @param[in] runes_capacity Capacity of the @p runes buffer, in @c sz_rune_t entries.
 *  @param[out] runes_unpacked Number of runes unpacked.
 *  @return Pointer to the byte after the last unpacked byte in @p text (the resume cursor).
 */
STRINGZILLA_API_RUNTIME sz_cptr_t sz_utf8_decode( //
    sz_cptr_t text, sz_size_t length,             //
    sz_rune_t *runes, sz_size_t runes_capacity,   //
    sz_size_t *runes_unpacked);

#pragma endregion

#pragma region Platform Specific Backends

/** @copydoc sz_utf8_count */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_count_serial(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_seek */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_seek_serial(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_decode */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_decode_serial( //
    sz_cptr_t text, sz_size_t length,                     //
    sz_rune_t *runes, sz_size_t runes_capacity,           //
    sz_size_t *runes_unpacked);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_utf8_count */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_count_haswell(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_seek */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_seek_haswell(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_decode */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_decode_haswell( //
    sz_cptr_t text, sz_size_t length,                      //
    sz_rune_t *runes, sz_size_t runes_capacity, sz_size_t *runes_unpacked);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_utf8_count */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_count_icelake(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_seek */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_seek_icelake(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_decode */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_decode_icelake( //
    sz_cptr_t text, sz_size_t length,                      //
    sz_rune_t *runes, sz_size_t runes_capacity, sz_size_t *runes_unpacked);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_utf8_count */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_count_neon(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_seek */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_seek_neon(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_decode */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_decode_neon( //
    sz_cptr_t text, sz_size_t length,                   //
    sz_rune_t *runes, sz_size_t runes_capacity, sz_size_t *runes_unpacked);
#endif

#if STRINGZILLA_TARGET_SVE2
/** @copydoc sz_utf8_count */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_count_sve2(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_seek */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_seek_sve2(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_decode */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_decode_sve2( //
    sz_cptr_t text, sz_size_t length,                   //
    sz_rune_t *runes, sz_size_t runes_capacity, sz_size_t *runes_unpacked);
#endif

#if STRINGZILLA_TARGET_V128
/** @copydoc sz_utf8_count */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_count_v128(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_seek */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_seek_v128(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_decode */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_decode_v128( //
    sz_cptr_t text, sz_size_t length,                   //
    sz_rune_t *runes, sz_size_t runes_capacity, sz_size_t *runes_unpacked);
#endif

#if STRINGZILLA_TARGET_V128RELAXED
/** @copydoc sz_utf8_count */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_count_v128relaxed(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_seek */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_seek_v128relaxed(sz_cptr_t text, sz_size_t length, sz_size_t n);
#endif

#if STRINGZILLA_TARGET_RVV
/** @copydoc sz_utf8_count */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_count_rvv(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_seek */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_seek_rvv(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_decode */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_decode_rvv( //
    sz_cptr_t text, sz_size_t length,                  //
    sz_rune_t *runes, sz_size_t runes_capacity, sz_size_t *runes_unpacked);
#endif

#if STRINGZILLA_TARGET_LASX
/** @copydoc sz_utf8_count */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_count_lasx(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_seek */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_seek_lasx(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_decode */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_decode_lasx( //
    sz_cptr_t text, sz_size_t length,                   //
    sz_rune_t *runes, sz_size_t runes_capacity, sz_size_t *runes_unpacked);
#endif

#if STRINGZILLA_TARGET_POWERVSX
/** @copydoc sz_utf8_count */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_count_powervsx(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_seek */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_seek_powervsx(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_decode */
STRINGZILLA_API_COMPTIME sz_cptr_t sz_utf8_decode_powervsx( //
    sz_cptr_t text, sz_size_t length,                       //
    sz_rune_t *runes, sz_size_t runes_capacity, sz_size_t *runes_unpacked);
#endif

#pragma endregion

/*  Implementation Section - each ISA backend lives in its own header, included serial-first. */
#include "stringzilla/utf8_runes/serial.h"
#include "stringzilla/utf8_runes/icelake.h"
#include "stringzilla/utf8_runes/haswell.h"
#include "stringzilla/utf8_runes/neon.h"
#include "stringzilla/utf8_runes/sve2.h"
#include "stringzilla/utf8_runes/v128.h"
#include "stringzilla/utf8_runes/v128relaxed.h"
#include "stringzilla/utf8_runes/rvv.h"
#include "stringzilla/utf8_runes/lasx.h"
#include "stringzilla/utf8_runes/powervsx.h"

#pragma region Dynamic Dispatch

#if !STRINGZILLA_RUNTIME_DISPATCH

STRINGZILLA_API_RUNTIME sz_size_t sz_utf8_count(sz_cptr_t text, sz_size_t length) {
#if STRINGZILLA_TARGET_V128RELAXED
    return sz_utf8_count_v128relaxed(text, length);
#elif STRINGZILLA_TARGET_V128
    return sz_utf8_count_v128(text, length);
#elif STRINGZILLA_TARGET_RVV
    return sz_utf8_count_rvv(text, length);
#elif STRINGZILLA_TARGET_LASX
    return sz_utf8_count_lasx(text, length);
#elif STRINGZILLA_TARGET_POWERVSX
    return sz_utf8_count_powervsx(text, length);
#elif STRINGZILLA_TARGET_ICELAKE
    return sz_utf8_count_icelake(text, length);
#elif STRINGZILLA_TARGET_HASWELL
    return sz_utf8_count_haswell(text, length);
#elif STRINGZILLA_TARGET_SVE2
    return sz_utf8_count_sve2(text, length);
#elif STRINGZILLA_TARGET_NEON
    return sz_utf8_count_neon(text, length);
#else
    return sz_utf8_count_serial(text, length);
#endif
}

STRINGZILLA_API_RUNTIME sz_cptr_t sz_utf8_seek(sz_cptr_t text, sz_size_t length, sz_size_t n) {
#if STRINGZILLA_TARGET_V128RELAXED
    return sz_utf8_seek_v128relaxed(text, length, n);
#elif STRINGZILLA_TARGET_V128
    return sz_utf8_seek_v128(text, length, n);
#elif STRINGZILLA_TARGET_RVV
    return sz_utf8_seek_rvv(text, length, n);
#elif STRINGZILLA_TARGET_LASX
    return sz_utf8_seek_lasx(text, length, n);
#elif STRINGZILLA_TARGET_POWERVSX
    return sz_utf8_seek_powervsx(text, length, n);
#elif STRINGZILLA_TARGET_ICELAKE
    return sz_utf8_seek_icelake(text, length, n);
#elif STRINGZILLA_TARGET_HASWELL
    return sz_utf8_seek_haswell(text, length, n);
#elif STRINGZILLA_TARGET_SVE2
    return sz_utf8_seek_sve2(text, length, n);
#elif STRINGZILLA_TARGET_NEON
    return sz_utf8_seek_neon(text, length, n);
#else
    return sz_utf8_seek_serial(text, length, n);
#endif
}

STRINGZILLA_API_RUNTIME sz_cptr_t sz_utf8_decode(sz_cptr_t text, sz_size_t length, sz_rune_t *runes,
                                                 sz_size_t runes_capacity, sz_size_t *runes_unpacked) {
#if STRINGZILLA_TARGET_V128
    return sz_utf8_decode_v128(text, length, runes, runes_capacity, runes_unpacked);
#elif STRINGZILLA_TARGET_RVV
    return sz_utf8_decode_rvv(text, length, runes, runes_capacity, runes_unpacked);
#elif STRINGZILLA_TARGET_LASX
    return sz_utf8_decode_lasx(text, length, runes, runes_capacity, runes_unpacked);
#elif STRINGZILLA_TARGET_POWERVSX
    return sz_utf8_decode_powervsx(text, length, runes, runes_capacity, runes_unpacked);
#elif STRINGZILLA_TARGET_ICELAKE
    return sz_utf8_decode_icelake(text, length, runes, runes_capacity, runes_unpacked);
#elif STRINGZILLA_TARGET_HASWELL
    return sz_utf8_decode_haswell(text, length, runes, runes_capacity, runes_unpacked);
#elif STRINGZILLA_TARGET_SVE2
    return sz_utf8_decode_sve2(text, length, runes, runes_capacity, runes_unpacked);
#elif STRINGZILLA_TARGET_NEON
    return sz_utf8_decode_neon(text, length, runes, runes_capacity, runes_unpacked);
#else
    return sz_utf8_decode_serial(text, length, runes, runes_capacity, runes_unpacked);
#endif
}

#endif // !STRINGZILLA_RUNTIME_DISPATCH

#pragma endregion

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_RUNES_H_
