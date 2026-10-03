/**
 *  @file include/stringzilla/utf8_runes.h
 *  @author Ash Vardanian
 *  @date November 19, 2025
 *  @brief Hardware-accelerated UTF-8 codepoint mechanics: count, find-nth, and chunk unpacking.
 */
#ifndef STRINGZILLA_UTF8_RUNES_H_
#define STRINGZILLA_UTF8_RUNES_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`

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
 *      sz_capability_t capabilities;
 *      sz_cpu_capabilities_enabled(&capabilities);
 *      sz_size_t char_count;
 *      sz_utf8_count_best(text, length, &char_count, capabilities, NULL);
 *      printf("String has %zu characters\n", char_count);
 *  @endcode
 *
 *  @param[in] text String to be scanned.
 *  @param[in] length Number of bytes in the string.
 *  @param[out] count Number of UTF-8 characters in the string.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 */
STRINGZILLA_API sz_status_t sz_utf8_count_best(sz_cptr_t text, sz_size_t length, sz_size_t *count,
                                               sz_capability_t capabilities, void *stream);

/**
 *  @brief Skip forward to the Nth UTF-8 character.
 *
 *  Skipping to character 1000, e.g. for pagination:
 *
 *  @code{.c}
 *      char const *position;
 *      sz_utf8_seek_best(text, length, 1000, &position, capabilities, NULL);
 *      if (!position) {
 *          // String has fewer than 1000 characters
 *      }
 *  @endcode
 *
 *  Truncating to 280 characters, Twitter-style:
 *
 *  @code{.c}
 *      char const *end;
 *      sz_utf8_seek_best(text, length, 280, &end, capabilities, NULL);
 *      size_t truncated_bytes = end ? (end - text) : length;
 *  @endcode
 *
 *  @param[in] text String to be scanned.
 *  @param[in] length Number of bytes in the string.
 *  @param[in] n Number of UTF-8 characters to skip, 0-indexed, so `n = 0` yields @p text.
 *  @param[out] position The Nth character, or NULL if the string has fewer than @p n characters.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 */
STRINGZILLA_API sz_status_t sz_utf8_seek_best(sz_cptr_t text, sz_size_t length, sz_size_t n, sz_cptr_t *position,
                                              sz_capability_t capabilities, void *stream);

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
 *
 *  End of text is end of input: a well-formed but truncated final sequence decodes to one U+FFFD
 *  over its maximal subpart and is consumed, so `"a\xE2\x82"` yields `{'a', U+FFFD}` over all 3
 *  bytes. Only a full @p runes buffer stops early, so @p bytes_consumed falls short of @p length
 *  only when @p runes_count reaches @p runes_capacity, and then at a rune boundary. Callers
 *  decoding a stream in chunks split it at a rune boundary.
 *
 *  @param[in] text UTF-8 string to unpack.
 *  @param[in] length Number of bytes in the string.
 *  @param[out] runes Buffer for UTF-32 codepoints, recommended to be at least @b 64 entries wide.
 *  @param[in] runes_capacity Capacity of the @p runes buffer, in @c sz_rune_t entries.
 *  @param[out] runes_count Number of runes unpacked.
 *  @param[out] bytes_consumed Bytes of @p text decoded, the offset the next call resumes at.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 */
STRINGZILLA_API sz_status_t sz_utf8_decode_best(       //
    sz_cptr_t text, sz_size_t length,                  //
    sz_rune_t *runes, sz_size_t runes_capacity,        //
    sz_size_t *runes_count, sz_size_t *bytes_consumed, //
    sz_capability_t capabilities, void *stream);

/**
 *  @brief Finds the UTF-8 runes kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_utf8_runes_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                      sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion

#pragma region Platform Specific Backends

/** @copydoc sz_utf8_count_best */
STRINGZILLA_API sz_status_t sz_utf8_count_serial(sz_cptr_t text, sz_size_t length, sz_size_t *count, void *stream);
/** @copydoc sz_utf8_seek_best */
STRINGZILLA_API sz_status_t sz_utf8_seek_serial(sz_cptr_t text, sz_size_t length, sz_size_t n, sz_cptr_t *position,
                                                void *stream);
/** @copydoc sz_utf8_decode_best */
STRINGZILLA_API sz_status_t sz_utf8_decode_serial( //
    sz_cptr_t text, sz_size_t length,              //
    sz_rune_t *runes, sz_size_t runes_capacity,    //
    sz_size_t *runes_count, sz_size_t *bytes_consumed, void *stream);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_utf8_count_best */
STRINGZILLA_API sz_status_t sz_utf8_count_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *count, void *stream);
/** @copydoc sz_utf8_seek_best */
STRINGZILLA_API sz_status_t sz_utf8_seek_haswell(sz_cptr_t text, sz_size_t length, sz_size_t n, sz_cptr_t *position,
                                                 void *stream);
/** @copydoc sz_utf8_decode_best */
STRINGZILLA_API sz_status_t sz_utf8_decode_haswell( //
    sz_cptr_t text, sz_size_t length,               //
    sz_rune_t *runes, sz_size_t runes_capacity,     //
    sz_size_t *runes_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_utf8_count_best */
STRINGZILLA_API sz_status_t sz_utf8_count_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *count, void *stream);
/** @copydoc sz_utf8_seek_best */
STRINGZILLA_API sz_status_t sz_utf8_seek_icelake(sz_cptr_t text, sz_size_t length, sz_size_t n, sz_cptr_t *position,
                                                 void *stream);
/** @copydoc sz_utf8_decode_best */
STRINGZILLA_API sz_status_t sz_utf8_decode_icelake( //
    sz_cptr_t text, sz_size_t length,               //
    sz_rune_t *runes, sz_size_t runes_capacity,     //
    sz_size_t *runes_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_utf8_count_best */
STRINGZILLA_API sz_status_t sz_utf8_count_neon(sz_cptr_t text, sz_size_t length, sz_size_t *count, void *stream);
/** @copydoc sz_utf8_seek_best */
STRINGZILLA_API sz_status_t sz_utf8_seek_neon(sz_cptr_t text, sz_size_t length, sz_size_t n, sz_cptr_t *position,
                                              void *stream);
/** @copydoc sz_utf8_decode_best */
STRINGZILLA_API sz_status_t sz_utf8_decode_neon( //
    sz_cptr_t text, sz_size_t length,            //
    sz_rune_t *runes, sz_size_t runes_capacity,  //
    sz_size_t *runes_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_SVE2
/** @copydoc sz_utf8_count_best */
STRINGZILLA_API sz_status_t sz_utf8_count_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *count, void *stream);
/** @copydoc sz_utf8_seek_best */
STRINGZILLA_API sz_status_t sz_utf8_seek_sve2(sz_cptr_t text, sz_size_t length, sz_size_t n, sz_cptr_t *position,
                                              void *stream);
/** @copydoc sz_utf8_decode_best */
STRINGZILLA_API sz_status_t sz_utf8_decode_sve2( //
    sz_cptr_t text, sz_size_t length,            //
    sz_rune_t *runes, sz_size_t runes_capacity,  //
    sz_size_t *runes_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_RVV
/** @copydoc sz_utf8_count_best */
STRINGZILLA_API sz_status_t sz_utf8_count_rvv(sz_cptr_t text, sz_size_t length, sz_size_t *count, void *stream);
/** @copydoc sz_utf8_seek_best */
STRINGZILLA_API sz_status_t sz_utf8_seek_rvv(sz_cptr_t text, sz_size_t length, sz_size_t n, sz_cptr_t *position,
                                             void *stream);
/** @copydoc sz_utf8_decode_best */
STRINGZILLA_API sz_status_t sz_utf8_decode_rvv( //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_V128
/** @copydoc sz_utf8_count_best */
STRINGZILLA_API sz_status_t sz_utf8_count_v128(sz_cptr_t text, sz_size_t length, sz_size_t *count, void *stream);
/** @copydoc sz_utf8_seek_best */
STRINGZILLA_API sz_status_t sz_utf8_seek_v128(sz_cptr_t text, sz_size_t length, sz_size_t n, sz_cptr_t *position,
                                              void *stream);
/** @copydoc sz_utf8_decode_best */
STRINGZILLA_API sz_status_t sz_utf8_decode_v128( //
    sz_cptr_t text, sz_size_t length,            //
    sz_rune_t *runes, sz_size_t runes_capacity,  //
    sz_size_t *runes_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_V128RELAXED
/** @copydoc sz_utf8_count_best */
STRINGZILLA_API sz_status_t sz_utf8_count_v128relaxed(sz_cptr_t text, sz_size_t length, sz_size_t *count, void *stream);
/** @copydoc sz_utf8_seek_best */
STRINGZILLA_API sz_status_t sz_utf8_seek_v128relaxed(sz_cptr_t text, sz_size_t length, sz_size_t n, sz_cptr_t *position,
                                                     void *stream);
#endif

#if STRINGZILLA_TARGET_LOONGSONASX
/** @copydoc sz_utf8_count_best */
STRINGZILLA_API sz_status_t sz_utf8_count_loongsonasx(sz_cptr_t text, sz_size_t length, sz_size_t *count, void *stream);
/** @copydoc sz_utf8_seek_best */
STRINGZILLA_API sz_status_t sz_utf8_seek_loongsonasx(sz_cptr_t text, sz_size_t length, sz_size_t n, sz_cptr_t *position,
                                                     void *stream);
/** @copydoc sz_utf8_decode_best */
STRINGZILLA_API sz_status_t sz_utf8_decode_loongsonasx( //
    sz_cptr_t text, sz_size_t length,                   //
    sz_rune_t *runes, sz_size_t runes_capacity,         //
    sz_size_t *runes_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_POWERVSX
/** @copydoc sz_utf8_count_best */
STRINGZILLA_API sz_status_t sz_utf8_count_powervsx(sz_cptr_t text, sz_size_t length, sz_size_t *count, void *stream);
/** @copydoc sz_utf8_seek_best */
STRINGZILLA_API sz_status_t sz_utf8_seek_powervsx(sz_cptr_t text, sz_size_t length, sz_size_t n, sz_cptr_t *position,
                                                  void *stream);
/** @copydoc sz_utf8_decode_best */
STRINGZILLA_API sz_status_t sz_utf8_decode_powervsx( //
    sz_cptr_t text, sz_size_t length,                //
    sz_rune_t *runes, sz_size_t runes_capacity,      //
    sz_size_t *runes_count, sz_size_t *bytes_consumed, void *stream);
#endif

#pragma endregion

/*  Header-only builds define each kernel inline from its tier header, while the library defines
 *  every kernel once, in its capability's unit under `c/target/`. */
#include "stringzilla/utf8_runes/serial.h"
#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/utf8_runes/icelake.h"
#include "stringzilla/utf8_runes/haswell.h"
#include "stringzilla/utf8_runes/neon.h"
#include "stringzilla/utf8_runes/sve2.h"
#include "stringzilla/utf8_runes/v128.h"
#include "stringzilla/utf8_runes/v128relaxed.h"
#include "stringzilla/utf8_runes/rvv.h"
#include "stringzilla/utf8_runes/loongsonasx.h"
#include "stringzilla/utf8_runes/powervsx.h"
#endif // STRINGZILLA_HEADER_ONLY

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_utf8_count_best(sz_cptr_t text, sz_size_t length, sz_size_t *count,
                                               sz_capability_t capabilities, void *stream) {
    sz_unused_(text), sz_unused_(length), sz_unused_(count), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_utf8_seek_best(sz_cptr_t text, sz_size_t length, sz_size_t n, sz_cptr_t *position,
                                              sz_capability_t capabilities, void *stream) {
    sz_unused_(text), sz_unused_(length), sz_unused_(n), sz_unused_(position), sz_unused_(capabilities),
        sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_utf8_decode_best(       //
    sz_cptr_t text, sz_size_t length,                  //
    sz_rune_t *runes, sz_size_t runes_capacity,        //
    sz_size_t *runes_count, sz_size_t *bytes_consumed, //
    sz_capability_t capabilities, void *stream) {
    sz_unused_(text), sz_unused_(length), sz_unused_(runes), sz_unused_(runes_capacity), sz_unused_(runes_count),
        sz_unused_(bytes_consumed), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_utf8_runes_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                      sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_RUNES_H_
