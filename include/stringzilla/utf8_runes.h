/**
 *  @brief Hardware-accelerated UTF-8 codepoint mechanics: count, find-nth, and chunk unpacking.
 *  @file utf8_runes.h
 *  @author Ash Vardanian
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
 *  The logic is to count the number of "continuation bytes" matching the 10xxxxxx pattern,
 *  and then subtract that from the total byte length to get the number of "start bytes" -
 *  coinciding with the number of UTF-8 characters.
 *
 *  @param text String to be scanned.
 *  @param length Number of bytes in the string.
 *  @return Number of UTF-8 characters in the string.
 *
 *  @example Count characters:
 *  @code
 *      size_t char_count = sz_utf8_count(text, length);
 *      printf("String has %zu characters\n", char_count);
 *  @endcode
 */
SZ_API_RUNTIME sz_size_t sz_utf8_count(sz_cptr_t text, sz_size_t length);

/**
 *  @brief Skip forward to the Nth UTF-8 character.
 *
 *  @param text String to be scanned.
 *  @param length Number of bytes in the string.
 *  @param n Number of UTF-8 characters to skip (0-indexed, so n=0 returns text).
 *  @return Pointer to the Nth character, or NULL if the string has fewer than n characters.
 *
 *  @example Skip to character 1000 (e.g., pagination):
 *  @code
 *      char const *pos = sz_utf8_seek(text, length, 1000);
 *      if (!pos) {
 *          // String has fewer than 1000 characters
 *      }
 *  @endcode
 *
 *  @example Truncate to 280 characters (Twitter-style):
 *  @code
 *      char const *end = sz_utf8_seek(text, length, 280);
 *      size_t truncated_bytes = end ? (end - text) : length;
 *  @endcode
 */
SZ_API_RUNTIME sz_cptr_t sz_utf8_seek(sz_cptr_t text, sz_size_t length, sz_size_t n);

/**
 *  @brief Unpack a UTF-8 string into UTF-32 codepoints.
 *
 *  This function is designed for streaming-like decoding with smart iterators built on top of it.
 *  The iterator would unpack a continuous slice of UTF-8 text into UTF-32 codepoints in chunks,
 *  yielding them upstream - only one at a time. This avoids allocating large buffers for the entire
 *  UTF-32 string, which can be 4x the size of the UTF-8 input.
 *
 *  This functionality is similar to the `simdutf` library's UTF-8 to UTF-32 conversion routines, and
 *  leverages an assumption that the absolute majority of written text doesn't mix codepoints of every
 *  length in each register-sized chunk:
 *
 *  - English text and source code is predominantly 1-byte ASCII characters.
 *  - Broader European languages with diacritics mostly use 2-byte characters with 1-byte punctuation.
 *  - Chinese & Jamapanese mostly use 3-byte characters with rare punctuation, which can be 1- or 3-byte.
 *  - Korean uses 3-byte characters with 1-byte spaces; word are 2-6 syllables or 6-16 bytes.
 *
 *  It's a different story for emoji-heavy texts, which can mix 4-byte characters more frequently.
 *
 *  @b Contract (uniform across every backend, and the basis for the C++/Rust/Python codepoint iterators):
 *  - @b Fill-or-drain: emits runes until @p runes_capacity is reached or @p text is exhausted, looping
 *    internally - one call fills the buffer regardless of how many byte-widths the text mixes.
 *  - @b Total @b and @b safe: never reads past `text + length`; substitutes one @b U+FFFD per maximal
 *    ill-formed subpart (overlong, surrogate, out-of-range, or a stray byte) and resyncs.
 *  - @b Valid @b scalar @b values: every emitted rune is a Unicode scalar value (incl. U+FFFD), so callers
 *    may convert without re-validation.
 *  - @b Resumable @b truncation: a well-formed but truncated trailing prefix is left unconsumed (the cursor
 *    stops before it) so a streaming caller resumes once more bytes arrive.
 *
 *  @param text UTF-8 string to unpack.
 *  @param length Number of bytes in the string.
 *  @param runes Output buffer for UTF-32 codepoints (recommended to be at least @b 64 entries wide).
 *  @param runes_capacity Capacity of the @p runes buffer (number of sz_rune_t entries).
 *  @param runes_unpacked Number of runes unpacked.
 *  @return Pointer to the byte after the last unpacked byte in @p text (the resume cursor).
 */
SZ_API_RUNTIME sz_cptr_t sz_utf8_decode(        //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked);

#pragma endregion

#pragma region Platform Specific Backends

/** @copydoc sz_utf8_count */
SZ_API_COMPTIME sz_size_t sz_utf8_count_serial(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_seek */
SZ_API_COMPTIME sz_cptr_t sz_utf8_seek_serial(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_decode */
SZ_API_COMPTIME sz_cptr_t sz_utf8_decode_serial( //
    sz_cptr_t text, sz_size_t length,            //
    sz_rune_t *runes, sz_size_t runes_capacity,  //
    sz_size_t *runes_unpacked);

#if SZ_USE_HASWELL
/** @copydoc sz_utf8_count */
SZ_API_COMPTIME sz_size_t sz_utf8_count_haswell(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_seek */
SZ_API_COMPTIME sz_cptr_t sz_utf8_seek_haswell(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_decode */
SZ_API_COMPTIME sz_cptr_t sz_utf8_decode_haswell( //
    sz_cptr_t text, sz_size_t length,             //
    sz_rune_t *runes, sz_size_t runes_capacity, sz_size_t *runes_unpacked);
#endif

#if SZ_USE_ICELAKE
/** @copydoc sz_utf8_count */
SZ_API_COMPTIME sz_size_t sz_utf8_count_icelake(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_seek */
SZ_API_COMPTIME sz_cptr_t sz_utf8_seek_icelake(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_decode */
SZ_API_COMPTIME sz_cptr_t sz_utf8_decode_icelake( //
    sz_cptr_t text, sz_size_t length,             //
    sz_rune_t *runes, sz_size_t runes_capacity, sz_size_t *runes_unpacked);
#endif

#if SZ_USE_NEON
/** @copydoc sz_utf8_count */
SZ_API_COMPTIME sz_size_t sz_utf8_count_neon(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_seek */
SZ_API_COMPTIME sz_cptr_t sz_utf8_seek_neon(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_decode */
SZ_API_COMPTIME sz_cptr_t sz_utf8_decode_neon( //
    sz_cptr_t text, sz_size_t length,          //
    sz_rune_t *runes, sz_size_t runes_capacity, sz_size_t *runes_unpacked);
#endif

#if SZ_USE_SVE2
/** @copydoc sz_utf8_count */
SZ_API_COMPTIME sz_size_t sz_utf8_count_sve2(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_seek */
SZ_API_COMPTIME sz_cptr_t sz_utf8_seek_sve2(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_decode */
SZ_API_COMPTIME sz_cptr_t sz_utf8_decode_sve2( //
    sz_cptr_t text, sz_size_t length,          //
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

#if !SZ_DYNAMIC_DISPATCH

SZ_API_RUNTIME sz_size_t sz_utf8_count(sz_cptr_t text, sz_size_t length) {
#if SZ_USE_V128RELAXED
    return sz_utf8_count_v128relaxed(text, length);
#elif SZ_USE_V128
    return sz_utf8_count_v128(text, length);
#elif SZ_USE_RVV
    return sz_utf8_count_rvv(text, length);
#elif SZ_USE_LASX
    return sz_utf8_count_lasx(text, length);
#elif SZ_USE_POWERVSX
    return sz_utf8_count_powervsx(text, length);
#elif SZ_USE_ICELAKE
    return sz_utf8_count_icelake(text, length);
#elif SZ_USE_HASWELL
    return sz_utf8_count_haswell(text, length);
#elif SZ_USE_SVE2
    return sz_utf8_count_sve2(text, length);
#elif SZ_USE_NEON
    return sz_utf8_count_neon(text, length);
#else
    return sz_utf8_count_serial(text, length);
#endif
}

SZ_API_RUNTIME sz_cptr_t sz_utf8_seek(sz_cptr_t text, sz_size_t length, sz_size_t n) {
#if SZ_USE_V128RELAXED
    return sz_utf8_seek_v128relaxed(text, length, n);
#elif SZ_USE_V128
    return sz_utf8_seek_v128(text, length, n);
#elif SZ_USE_RVV
    return sz_utf8_seek_rvv(text, length, n);
#elif SZ_USE_LASX
    return sz_utf8_seek_lasx(text, length, n);
#elif SZ_USE_POWERVSX
    return sz_utf8_seek_powervsx(text, length, n);
#elif SZ_USE_ICELAKE
    return sz_utf8_seek_icelake(text, length, n);
#elif SZ_USE_HASWELL
    return sz_utf8_seek_haswell(text, length, n);
#elif SZ_USE_SVE2
    return sz_utf8_seek_sve2(text, length, n);
#elif SZ_USE_NEON
    return sz_utf8_seek_neon(text, length, n);
#else
    return sz_utf8_seek_serial(text, length, n);
#endif
}

SZ_API_RUNTIME sz_cptr_t sz_utf8_decode(sz_cptr_t text, sz_size_t length, sz_rune_t *runes, sz_size_t runes_capacity,
                                        sz_size_t *runes_unpacked) {
#if SZ_USE_V128
    return sz_utf8_decode_v128(text, length, runes, runes_capacity, runes_unpacked);
#elif SZ_USE_RVV
    return sz_utf8_decode_rvv(text, length, runes, runes_capacity, runes_unpacked);
#elif SZ_USE_LASX
    return sz_utf8_decode_lasx(text, length, runes, runes_capacity, runes_unpacked);
#elif SZ_USE_POWERVSX
    return sz_utf8_decode_powervsx(text, length, runes, runes_capacity, runes_unpacked);
#elif SZ_USE_ICELAKE
    return sz_utf8_decode_icelake(text, length, runes, runes_capacity, runes_unpacked);
#elif SZ_USE_HASWELL
    return sz_utf8_decode_haswell(text, length, runes, runes_capacity, runes_unpacked);
#elif SZ_USE_SVE2
    return sz_utf8_decode_sve2(text, length, runes, runes_capacity, runes_unpacked);
#elif SZ_USE_NEON
    return sz_utf8_decode_neon(text, length, runes, runes_capacity, runes_unpacked);
#else
    return sz_utf8_decode_serial(text, length, runes, runes_capacity, runes_unpacked);
#endif
}

#endif // !SZ_DYNAMIC_DISPATCH

#pragma endregion

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_RUNES_H_
