/**
 *  @file include/stringzilla/utf8_uncased.h
 *  @author Ash Vardanian
 *  @date November 23, 2025
 *  @brief Uncased UTF-8 substring search, comparison & case-invariance checks.
 *
 *  Public Core API:
 *
 *  - @c sz_utf8_uncased_needle_init_best - one-time analysis of a needle for uncased search
 *  - @c sz_utf8_uncased_search_best - uncased substring search in UTF-8 strings
 *  - @c sz_utf8_uncased_order_best - uncased lexicographical comparison of UTF-8 strings
 *  - @c sz_utf8_find_cased_best - the first cased (foldable) codepoint, or NULL if caseless
 *
 *  All comparison and matching uses full Unicode Case Folding (UAX #21 / CaseFolding.txt),
 *  including one-to-many expansions (e.g., 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)).
 *  Folding is locale-independent and deterministic across platforms.
 *
 *  On fast vectorized paths there may be significant algorithmic differences between ISA versions;
 *  the per-script SIMD kernels (Ice Lake, ...) consume the needle analysis produced once by the
 *  ISA-agnostic classifier in `utf8_uncased/serial.h`.
 *
 *  @sa include/stringzilla/utf8_uncased_fold.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_H_
#define STRINGZILLA_UTF8_UNCASED_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h"        // `sz_capability_t`
#include "stringzilla/utf8_uncased/serial.h" // `sz_utf8_uncased_needle_t`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Analyses a needle once for any number of uncased searches, under any capabilities.
 *
 *  Picks the per-script search kernel and the probe positions it matches on, so that
 *  @c sz_utf8_uncased_search_best never re-analyses the needle. The prepared needle points at
 *  @p needle rather than copying it, so the needle must outlive it:
 *
 *  @code{.c}
 *      sz_utf8_uncased_needle_t prepared;
 *      sz_utf8_uncased_needle_init_best("STRASSE", 7, &prepared, capabilities, STRINGZILLA_NULL);
 *      sz_cptr_t match;
 *      sz_size_t match_length;
 *      sz_utf8_uncased_search_best("die Straße", 11, &prepared, &match, &match_length, capabilities,
 *                                  STRINGZILLA_NULL);
 *      // match points at "Straße", match_length == 7
 *  @endcode
 *
 *  @param[in] needle UTF-8 substring to search for, possibly empty.
 *  @param[in] needle_length Number of bytes in @p needle.
 *  @param[out] prepared The prepared needle, every field filled.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 */
STRINGZILLA_API sz_status_t sz_utf8_uncased_needle_init_best(sz_cptr_t needle, sz_size_t needle_length,
                                                             sz_utf8_uncased_needle_t *prepared,
                                                             sz_capability_t capabilities, sz_stream_t stream);

/**
 *  @brief Uncased substring search in UTF-8 strings.
 *
 *  In applications where the haystack remains largely static and memory/storage is cheap, it is
 *  recommended to pre-process the haystack into a case-folded version using Unicode Case Folding
 *  (e.g., via the ICU library) and subsequently use the simpler @c sz_find_best function for
 *  repeated searches, which avoids performing the full folding logic during every search.
 *
 *  This function applies full Unicode Case Folding as defined in the Unicode Standard (UAX #21 and
 *  CaseFolding.txt), covering all bicameral scripts, all offset-based one-to-one folds, all
 *  table-based one-to-one folds, and all normative one-to-many expansions. It doesn't however
 *  perform any normalization, like NFKC or NFC, so combining marks are treated as-is. StringZilla
 *  is intentionally locale-independent: case folding produces identical results regardless of
 *  runtime locale settings, ensuring deterministic behavior across platforms and simplifying use in
 *  multi-threaded and distributed systems.
 *
 *  The following character mappings are supported:
 *
 *  - ASCII Latin letters A–Z (U+0041–U+005A) are folded to a–z (U+0061–U+007A) using a
 *    trivial +32 offset.
 *  - Fullwidth Latin letters Ａ–Ｚ (U+FF21–U+FF3A) are folded to ａ–ｚ (U+FF41–U+FF5A) with the
 *    same +32 offset.
 *  - Cyrillic uppercase А–Я (U+0410–U+042F) are folded to а–я (U+0430–U+044F) using a +32 offset.
 *  - Armenian uppercase Ա–Ֆ (U+0531–U+0556) are folded to ա–ֆ (U+0561–U+0586) using a +48 offset.
 *  - Georgian Mtavruli letters Ა-Ჿ (U+1C90–U+1CBF, excluding 2) are folded to their Mkhedruli
 *    equivalents (U+10D0–U+10FF) using a fixed linear translation defined by the Unicode Standard.
 *  - Greek uppercase Α–Ω (U+0391–U+03A9) are folded to α–ω (U+03B1–U+03C9) via a +32 offset. Both Σ
 *    (U+03A3) and ς (U+03C2, final sigma) fold to σ (U+03C3) for consistent matching.
 *  - Latin Extended characters include numerous one-to-one folds, mixed-case digraphs and
 *    trigraphs normalized to lowercase sequences, and one-to-many expansions like:
 *    - 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
 *    - 'ẞ' (U+1E9E, E1 BA 9E) → "ss" (U+0073 U+0073, 73 73)
 *  - Turkic dotted/dotless-I characters are handled per Unicode Case Folding (not locale-specific):
 *    - 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87) — Full case folding with combining dot
 *    - 'I' (U+0049, 49) → 'i' (U+0069, 69) — Standard folding (not Turkic 'I' (U+0049, 49) → 'ı'
 *      (U+0131, C4 B1))
 *    - 'ı' (U+0131, C4 B1) → 'ı' (U+0131, C4 B1) — Already lowercase, unchanged
 *  - Lithuanian accented I/J characters with combining dots are processed as multi-codepoint
 *    expansions per CaseFolding.txt.
 *  - Additional bicameral scripts, Cherokee, Deseret, Osage, Warang Citi, and Adlam, use their
 *    normative one-to-one uppercase-to-lowercase mappings defined in CaseFolding.txt.
 *
 *  Folding is applied during matching without rewriting the entire haystack. Multi-codepoint
 *  expansions, contextual folds, and combining-mark adjustments are handled at comparison time.
 *
 *  @section utf8_uncased_search_algo Algorithmic Considerations
 *
 *  Uncased search with full Unicode case folding is fundamentally harder than byte-level search
 *  because one-to-many expansions (e.g., 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)) break
 *  core assumptions of fast string algorithms:
 *
 *  - Boyer-Moore/Horspool skip tables assume 1:1 character mapping
 *  - Two-Way critical factorization assumes fixed pattern length
 *  - Rabin-Karp rolling hash assumes fixed character widths
 *  - Volnitsky bigram hashing assumes consistent byte patterns
 *
 *  Industry approaches vary:
 *
 *  - ICU abandoned Boyer-Moore for Unicode, reverting to linear search for correctness
 *  - ClickHouse uses Volnitsky with fallback to naive search for problematic characters
 *  - RipGrep uses simple case folding only (no expansion handling) leveraging the Rust RegEx engine
 *
 *  StringZilla implements several algorithms. Most importantly it first locates the longest
 *  expansion-free slice of the needle to locate against.
 *
 *  The references below cover ICU's abandonment of Boyer-Moore for Unicode, ClickHouse's hash-based
 *  Volnitsky search with UTF-8 case folding, glibc's O(n) time and O(1) space Two-Way matcher, and
 *  the uncased search of the uni-algo Unicode algorithms library.
 *
 *  Malformed UTF-8 is handled losslessly: any byte that does not begin a well-formed codepoint
 *  is matched byte-for-byte as a single literal byte (never as a Unicode codepoint) and
 *  processing resyncs at the next byte, so a search over invalid input is well-defined and
 *  identical across all backends.
 *
 *  @param[in] haystack UTF-8 string to be searched.
 *  @param[in] haystack_length Number of bytes in the haystack buffer.
 *  @param[in] needle The needle, prepared by @c sz_utf8_uncased_needle_init_best; an empty one
 *      matches at the start of @p haystack with a zero @p match_length.
 *  @param[out] match The first match in @p haystack, or @c STRINGZILLA_NULL_CHAR if not found.
 *  @param[out] match_length Number of bytes in the matched region.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @see ICU String Search: https://unicode-org.github.io/icu/userguide/collation/string-search.html
 *  @see ClickHouse Volnitsky: https://github.com/ClickHouse/ClickHouse/blob/master/src/Common/Volnitsky.h
 *  @see glibc Two-Way Algorithm: https://github.com/lattera/glibc/blob/master/string/str-two-way.h
 *  @see Efficient Parameterized Pattern Matching in Sublinear Space: https://arxiv.org/abs/2306.10714
 *  @see uni-algo: https://github.com/uni-algo/uni-algo
 */
STRINGZILLA_API sz_status_t sz_utf8_uncased_search_best(sz_cptr_t haystack, sz_size_t haystack_length,
                                                        sz_utf8_uncased_needle_t const *needle, sz_cptr_t *match,
                                                        sz_size_t *match_length, sz_capability_t capabilities,
                                                        sz_stream_t stream);

/**
 *  @brief Uncased lexicographic comparison of two UTF-8 strings.
 *
 *  Compares strings using Unicode case folding rules, producing consistent ordering regardless of
 *  letter case. Implements the same full Unicode Case Folding as @c sz_utf8_uncased_fold_best,
 *  including all one-to-many expansions (e.g. 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73))
 *  and bicameral script mappings.
 *
 *  Unlike simple byte comparison, this function correctly handles multi-byte UTF-8 sequences and
 *  expansion characters. Comparison is performed codepoint-by-codepoint after folding, not
 *  byte-by-byte, ensuring linguistically correct results.
 *
 *  Malformed UTF-8 is handled losslessly: any byte that does not begin a well-formed codepoint is
 *  treated as a single literal byte (folded to itself and compared byte-for-byte, never as a
 *  Unicode codepoint), and processing resyncs at the next byte. All uncased operations - find,
 *  order, and violation - share this contract, so results on invalid input are well-defined and
 *  consistent across backends. Basic usage:
 *
 *  @code{.c}
 *      sz_capability_t capabilities;
 *      sz_capabilities_enabled_cpu(&capabilities);
 *      sz_ordering_t ordering;
 *      sz_utf8_uncased_order_best("Hello", 5, "HELLO", 5, &ordering, capabilities, STRINGZILLA_NULL);
 *      // ordering == sz_equal_k
 *
 *      sz_utf8_uncased_order_best("straße", 7, "STRASSE", 7, &ordering, capabilities, STRINGZILLA_NULL);
 *      // ordering == sz_equal_k ('ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73))
 *  @endcode
 *
 *  @param[in] a First UTF-8 string to compare.
 *  @param[in] a_length Number of bytes in the first string.
 *  @param[in] b Second UTF-8 string to compare.
 *  @param[in] b_length Number of bytes in the second string.
 *  @param[out] ordering @c sz_less_k if a < b, @c sz_equal_k if a = b, @c sz_greater_k if a > b.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 */
STRINGZILLA_API sz_status_t sz_utf8_uncased_order_best( //
    sz_cptr_t a, sz_size_t a_length,                    //
    sz_cptr_t b, sz_size_t b_length,                    //
    sz_ordering_t *ordering, sz_capability_t capabilities, sz_stream_t stream);

/**
 *  @brief Locate the first cased, or case-foldable, codepoint in a UTF-8 string.
 *
 *  A codepoint is case-agnostic, or caseless, if all three conditions are true:
 *
 *  1. @b Self-folding: case folding produces exactly the original codepoint.
 *  2. @b Not @b bicameral: it does not belong to any script with case distinctions.
 *  3. @b Not @b an @b expansion @b target: it appears in no multi-rune case fold expansion.
 *
 *  The third condition is subtle but critical. Consider 'ʾ' (U+02BE, CA BE):
 *
 *  - It has no case variant and folds to itself.
 *  - However, 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE).
 *  - A needle containing 'ʾ' must match at position 1 of the folded expansion of 'ẚ'.
 *  - Binary search cannot handle this, so 'ʾ' must not be case-agnostic.
 *
 *  Case-agnostic scripts include: CJK ideographs, Hangul, digits, punctuation, most symbols,
 *  Hebrew, Arabic, Thai, Hindi (Devanagari), and many other scripts without case distinctions.
 *
 *  The function is conservative: it reports any codepoint that participates in case folding, even
 *  if the specific instance wouldn't change. For example, lowercase 'a' is reported because it's
 *  a case-folding target.
 *
 *  @section utf8_find_cased_usage Use Case
 *
 *  This function enables an important optimization: if both haystack and needle are fully
 *  case-agnostic, then @c sz_find_best can be used directly instead of the slower
 *  @c sz_utf8_uncased_search_best. This is particularly valuable for:
 *
 *  - CJK text (Chinese, Japanese, Korean) - always caseless
 *  - Numeric data and punctuation-heavy content
 *  - Middle Eastern scripts (Arabic, Hebrew, Persian)
 *  - South/Southeast Asian scripts (Thai, Hindi, Vietnamese without Latin)
 *
 *  The optimization pattern:
 *
 *  @code{.c}
 *      sz_cptr_t haystack = "价格：¥1234";  // Chinese + punctuation + digits
 *      sz_cptr_t needle = "¥1234";
 *      sz_cptr_t haystack_cased, needle_cased, match;
 *      sz_utf8_find_cased_best(haystack, haystack_length, &haystack_cased, capabilities, STRINGZILLA_NULL);
 *      sz_utf8_find_cased_best(needle, needle_length, &needle_cased, capabilities, STRINGZILLA_NULL);
 *
 *      if (!haystack_cased && !needle_cased) {
 *          // Fast path: both strings are fully caseless, so use binary search
 *          sz_find_best(haystack, haystack_length, needle, needle_length, &match, capabilities, STRINGZILLA_NULL);
 *      } else {
 *          // Slow path: full uncased search
 *          sz_utf8_uncased_needle_t prepared;
 *          sz_size_t match_length;
 *          sz_utf8_uncased_needle_init_best(needle, needle_length, &prepared, capabilities, STRINGZILLA_NULL);
 *          sz_utf8_uncased_search_best(haystack, haystack_length, &prepared, &match, &match_length, capabilities,
 *                                      STRINGZILLA_NULL);
 *      }
 *  @endcode
 *
 *  @param[in] text UTF-8 string to check.
 *  @param[in] length Number of bytes in @p text.
 *  @param[out] match The first cased codepoint, or @c STRINGZILLA_NULL_CHAR if all are caseless.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 */
STRINGZILLA_API sz_status_t sz_utf8_find_cased_best(sz_cptr_t text, sz_size_t length, sz_cptr_t *match,
                                                    sz_capability_t capabilities, sz_stream_t stream);

/**
 *  @brief Finds the UTF-8 uncased kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_utf8_uncased_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                        sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion

#pragma region Platform Specific Backends

/** @copydoc sz_utf8_uncased_needle_init_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_needle_init_serial(sz_cptr_t needle, sz_size_t needle_length,
                                                               sz_utf8_uncased_needle_t *prepared, sz_stream_t stream);
/** @copydoc sz_utf8_uncased_search_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_search_serial(                                 //
    sz_cptr_t haystack, sz_size_t haystack_length, sz_utf8_uncased_needle_t const *needle, //
    sz_cptr_t *match, sz_size_t *match_length, sz_stream_t stream);
/** @copydoc sz_utf8_uncased_order_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_order_serial( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, sz_ordering_t *ordering, sz_stream_t stream);
/** @copydoc sz_utf8_find_cased_best */
STRINGZILLA_API sz_status_t sz_utf8_find_cased_serial(sz_cptr_t text, sz_size_t length, sz_cptr_t *match,
                                                      sz_stream_t stream);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_utf8_uncased_search_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_search_haswell(                                //
    sz_cptr_t haystack, sz_size_t haystack_length, sz_utf8_uncased_needle_t const *needle, //
    sz_cptr_t *match, sz_size_t *match_length, sz_stream_t stream);
/** @copydoc sz_utf8_uncased_order_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_order_haswell( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, sz_ordering_t *ordering, sz_stream_t stream);
/** @copydoc sz_utf8_find_cased_best */
STRINGZILLA_API sz_status_t sz_utf8_find_cased_haswell(sz_cptr_t text, sz_size_t length, sz_cptr_t *match,
                                                       sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_utf8_uncased_search_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_search_icelake(                                //
    sz_cptr_t haystack, sz_size_t haystack_length, sz_utf8_uncased_needle_t const *needle, //
    sz_cptr_t *match, sz_size_t *match_length, sz_stream_t stream);
/** @copydoc sz_utf8_uncased_order_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_order_icelake( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, sz_ordering_t *ordering, sz_stream_t stream);
/** @copydoc sz_utf8_find_cased_best */
STRINGZILLA_API sz_status_t sz_utf8_find_cased_icelake(sz_cptr_t text, sz_size_t length, sz_cptr_t *match,
                                                       sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_utf8_uncased_search_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_search_neon(                                   //
    sz_cptr_t haystack, sz_size_t haystack_length, sz_utf8_uncased_needle_t const *needle, //
    sz_cptr_t *match, sz_size_t *match_length, sz_stream_t stream);
/** @copydoc sz_utf8_uncased_order_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_order_neon( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, sz_ordering_t *ordering, sz_stream_t stream);
/** @copydoc sz_utf8_find_cased_best */
STRINGZILLA_API sz_status_t sz_utf8_find_cased_neon(sz_cptr_t text, sz_size_t length, sz_cptr_t *match,
                                                    sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_SVE2
/** @copydoc sz_utf8_uncased_search_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_search_sve2(                                   //
    sz_cptr_t haystack, sz_size_t haystack_length, sz_utf8_uncased_needle_t const *needle, //
    sz_cptr_t *match, sz_size_t *match_length, sz_stream_t stream);
/** @copydoc sz_utf8_uncased_order_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_order_sve2( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, sz_ordering_t *ordering, sz_stream_t stream);
/** @copydoc sz_utf8_find_cased_best */
STRINGZILLA_API sz_status_t sz_utf8_find_cased_sve2(sz_cptr_t text, sz_size_t length, sz_cptr_t *match,
                                                    sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_RVV
/** @copydoc sz_utf8_uncased_search_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_search_rvv(                                    //
    sz_cptr_t haystack, sz_size_t haystack_length, sz_utf8_uncased_needle_t const *needle, //
    sz_cptr_t *match, sz_size_t *match_length, sz_stream_t stream);
/** @copydoc sz_utf8_uncased_order_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_order_rvv( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, sz_ordering_t *ordering, sz_stream_t stream);
/** @copydoc sz_utf8_find_cased_best */
STRINGZILLA_API sz_status_t sz_utf8_find_cased_rvv(sz_cptr_t text, sz_size_t length, sz_cptr_t *match,
                                                   sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128
/** @copydoc sz_utf8_uncased_search_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_search_v128(                                   //
    sz_cptr_t haystack, sz_size_t haystack_length, sz_utf8_uncased_needle_t const *needle, //
    sz_cptr_t *match, sz_size_t *match_length, sz_stream_t stream);
/** @copydoc sz_utf8_uncased_order_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_order_v128( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, sz_ordering_t *ordering, sz_stream_t stream);
/** @copydoc sz_utf8_find_cased_best */
STRINGZILLA_API sz_status_t sz_utf8_find_cased_v128(sz_cptr_t text, sz_size_t length, sz_cptr_t *match,
                                                    sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_LOONGSONASX
/** @copydoc sz_utf8_uncased_search_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_search_loongsonasx(                            //
    sz_cptr_t haystack, sz_size_t haystack_length, sz_utf8_uncased_needle_t const *needle, //
    sz_cptr_t *match, sz_size_t *match_length, sz_stream_t stream);
/** @copydoc sz_utf8_uncased_order_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_order_loongsonasx( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, sz_ordering_t *ordering, sz_stream_t stream);
/** @copydoc sz_utf8_find_cased_best */
STRINGZILLA_API sz_status_t sz_utf8_find_cased_loongsonasx(sz_cptr_t text, sz_size_t length, sz_cptr_t *match,
                                                           sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_POWERVSX
/** @copydoc sz_utf8_uncased_search_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_search_powervsx(                               //
    sz_cptr_t haystack, sz_size_t haystack_length, sz_utf8_uncased_needle_t const *needle, //
    sz_cptr_t *match, sz_size_t *match_length, sz_stream_t stream);
/** @copydoc sz_utf8_uncased_order_best */
STRINGZILLA_API sz_status_t sz_utf8_uncased_order_powervsx( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length, sz_ordering_t *ordering, sz_stream_t stream);
/** @copydoc sz_utf8_find_cased_best */
STRINGZILLA_API sz_status_t sz_utf8_find_cased_powervsx(sz_cptr_t text, sz_size_t length, sz_cptr_t *match,
                                                        sz_stream_t stream);
#endif

#pragma endregion

/*  Header-only builds define each kernel inline from its tier header, while the library defines
 *  every kernel once, in its capability's unit under `c/target/`. */
#include "stringzilla/utf8_uncased/serial.h"
#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/utf8_uncased/icelake.h"
#include "stringzilla/utf8_uncased/haswell.h"
#include "stringzilla/utf8_uncased/neon.h"
#include "stringzilla/utf8_uncased/sve2.h"
#include "stringzilla/utf8_uncased/v128.h"
#include "stringzilla/utf8_uncased/rvv.h"
#include "stringzilla/utf8_uncased/loongsonasx.h"
#include "stringzilla/utf8_uncased/powervsx.h"
#endif // STRINGZILLA_HEADER_ONLY

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_utf8_uncased_needle_init_best(sz_cptr_t needle, sz_size_t needle_length,
                                                             sz_utf8_uncased_needle_t *prepared,
                                                             sz_capability_t capabilities, sz_stream_t stream) {
    sz_unused_(needle), sz_unused_(needle_length), sz_unused_(prepared), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_utf8_uncased_search_best(sz_cptr_t haystack, sz_size_t haystack_length,
                                                        sz_utf8_uncased_needle_t const *needle, sz_cptr_t *match,
                                                        sz_size_t *match_length, sz_capability_t capabilities,
                                                        sz_stream_t stream) {
    sz_unused_(haystack), sz_unused_(haystack_length), sz_unused_(needle), sz_unused_(match), sz_unused_(match_length),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_utf8_uncased_order_best( //
    sz_cptr_t a, sz_size_t a_length,                    //
    sz_cptr_t b, sz_size_t b_length,                    //
    sz_ordering_t *ordering, sz_capability_t capabilities, sz_stream_t stream) {
    sz_unused_(a), sz_unused_(a_length), sz_unused_(b), sz_unused_(b_length), sz_unused_(ordering),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_utf8_find_cased_best(sz_cptr_t text, sz_size_t length, sz_cptr_t *match,
                                                    sz_capability_t capabilities, sz_stream_t stream) {
    sz_unused_(text), sz_unused_(length), sz_unused_(match), sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_utf8_uncased_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                        sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_UNCASED_H_
