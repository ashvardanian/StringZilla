/**
 *  @file include/stringzilla/utf8_wordbreaks.h
 *  @author Ash Vardanian
 *  @date November 30, 2025
 *  @brief Hardware-accelerated UAX-29 word boundary segmentation.
 */
#ifndef STRINGZILLA_UTF8_WORDBREAKS_H_
#define STRINGZILLA_UTF8_WORDBREAKS_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Segment UTF-8 text into UAX-29 words in a single pass.
 *
 *  Walks the input left-to-right and writes the byte length of each word into @p lengths. Words
 *  are the spans between consecutive TR29 boundaries and tile the input: each starts where the
 *  previous one ended, the first at @p text. A call covers all of @p text unless it fills
 *  @p lengths, in which case it covers the sum of the lengths it wrote, and the caller resumes
 *  from @p text advanced by that sum.
 *
 *  @param[in] text UTF-8 encoded text.
 *  @param[in] length Byte length of @p text.
 *  @param[out] lengths Word byte lengths, at least @p capacity entries.
 *  @param[in] capacity Capacity of @p lengths, in entries.
 *  @param[out] count Number of words written, at most @p capacity.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note No zero-length words are emitted; @p length == 0 yields no words.
 */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_best(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                    sz_size_t capacity, sz_size_t *count, sz_capability_t capabilities,
                                                    void *stream);

/**
 *  @brief Get the Unicode TR29 Word_Break property for a codepoint.
 *
 *  Returns one of the 16 Word_Break property values, @c sz_utf8_word_break_other_k through
 *  @c sz_utf8_word_break_mid_quotes_k: the foundation of TR29-compliant word boundary detection.
 *
 *  @param[in] rune The Unicode codepoint to classify.
 *  @return The Word_Break property value (0-15).
 *
 *  @see Unicode Text Segmentation: https://www.unicode.org/reports/tr29/
 */
STRINGZILLA_CONSTEXPR sz_u8_t sz_rune_word_break_property(sz_rune_t rune);

/**
 *  @brief Check if a codepoint is a "word character" (has word-forming property).
 *
 *  Returns true if the codepoint has a Word_Break property that typically forms words:
 *  ALetter, Hebrew_Letter, Numeric, Katakana, ExtendNumLet, or mid-word punctuation.
 *
 *  @param[in] rune The Unicode codepoint to check.
 *  @return @c sz_true_k if the codepoint is a word character, @c sz_false_k otherwise.
 */
STRINGZILLA_CONSTEXPR sz_bool_t sz_rune_is_word_char(sz_rune_t rune);

/**
 *  @brief Suggested batch size for streaming boundaries through the segmenting kernels.
 *
 *  Iterators that emit one segment/delimiter at a time buffer this many boundaries per call so
 *  the per-item overhead amortizes without an unbounded output buffer. It is only a default -
 *  any capacity works, and every kernel reports how far it got, so a caller can resume past a
 *  full output buffer.
 */
enum { sz_iterators_default_steps_k = 64 };

/**
 *  @brief Check if a position in UTF-8 text is a word boundary per Unicode TR29.
 *
 *  Implements the full TR29 word boundary algorithm including:
 *  - WB3: Do not break between CR and LF
 *  - WB4: Ignore Extend/Format/ZWJ characters for boundary purposes
 *  - WB5-WB13: Letter, number, and punctuation rules
 *  - WB15-WB16: Regional Indicator pair rules
 *
 *  @param[in] text UTF-8 encoded text.
 *  @param[in] length Byte length of @p text.
 *  @param[in] position Byte offset to check (must be start of a UTF-8 codepoint).
 *  @return @c sz_true_k if @p position is a word boundary, @c sz_false_k otherwise.
 *
 *  @note Position 0 and position == length are always boundaries (SOT/EOT).
 *  @note This is an internal helper used by the iterators; not part of stable ABI.
 */
STRINGZILLA_CONSTEXPR sz_bool_t sz_utf8_is_word_boundary_serial(sz_cptr_t text, sz_size_t length, sz_size_t position);

/**
 *  @brief Finds the UTF-8 word break kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                           sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion

#pragma region Platform Specific Backends

/** @copydoc sz_utf8_wordbreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_serial(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                      sz_size_t capacity, sz_size_t *count, void *stream);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_utf8_wordbreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                       sz_size_t capacity, sz_size_t *count, void *stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_utf8_wordbreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                       sz_size_t capacity, sz_size_t *count, void *stream);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_utf8_wordbreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_neon(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                    sz_size_t capacity, sz_size_t *count, void *stream);
#endif

#if STRINGZILLA_TARGET_SVE2
/** @copydoc sz_utf8_wordbreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                    sz_size_t capacity, sz_size_t *count, void *stream);
#endif

#if STRINGZILLA_TARGET_RVV
/** @copydoc sz_utf8_wordbreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_rvv(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                   sz_size_t capacity, sz_size_t *count, void *stream);
#endif

#if STRINGZILLA_TARGET_V128
/** @copydoc sz_utf8_wordbreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_v128(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                    sz_size_t capacity, sz_size_t *count, void *stream);
#endif

#if STRINGZILLA_TARGET_LOONGSONASX
/** @copydoc sz_utf8_wordbreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_loongsonasx(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                           sz_size_t capacity, sz_size_t *count, void *stream);
#endif

#if STRINGZILLA_TARGET_POWERVSX
/** @copydoc sz_utf8_wordbreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_powervsx(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                        sz_size_t capacity, sz_size_t *count, void *stream);
#endif

#pragma endregion

/*  Header-only builds define each kernel inline from its tier header, while the library defines
 *  every kernel once, in its capability's unit under `c/cpu/`. */
#include "stringzilla/utf8_wordbreaks/serial.h"
#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/utf8_wordbreaks/icelake.h"
#include "stringzilla/utf8_wordbreaks/haswell.h"
#include "stringzilla/utf8_wordbreaks/neon.h"
#include "stringzilla/utf8_wordbreaks/sve2.h"
#include "stringzilla/utf8_wordbreaks/v128.h"
#include "stringzilla/utf8_wordbreaks/rvv.h"
#include "stringzilla/utf8_wordbreaks/loongsonasx.h"
#include "stringzilla/utf8_wordbreaks/powervsx.h"
#endif // STRINGZILLA_HEADER_ONLY

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_best(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                    sz_size_t capacity, sz_size_t *count, sz_capability_t capabilities,
                                                    void *stream) {
    sz_unused_(text), sz_unused_(length), sz_unused_(lengths), sz_unused_(capacity), sz_unused_(count),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                           sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDBREAKS_H_
