/**
 *  @brief Case-insensitive UTF-8 substring search, comparison & case-invariance checks.
 *  @file include/stringzilla/utf8_case_insensitive.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_case_fold.h
 *
 *  Public Core API:
 *
 *  - `sz_utf8_case_insensitive_find` - case-insensitive substring search in UTF-8 strings
 *  - `sz_utf8_case_insensitive_order` - case-insensitive lexicographical comparison of UTF-8 strings
 *  - `sz_utf8_case_invariant` - check if a string contains only case-agnostic (caseless) codepoints
 *
 *  All comparison and matching uses full Unicode Case Folding (UAX #21 / CaseFolding.txt), including
 *  one-to-many expansions (e.g., 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)). Folding is
 *  locale-independent and deterministic across platforms.
 *
 *  On fast vectorized paths there may be significant algorithmic differences between ISA versions;
 *  the per-script SIMD kernels (Ice Lake, ...) consume needle metadata produced by the ISA-agnostic
 *  classifier in `utf8_case_insensitive/serial.h`.
 */
#ifndef STRINGZILLA_UTF8_CASE_INSENSITIVE_H_
#define STRINGZILLA_UTF8_CASE_INSENSITIVE_H_

#include "stringzilla/utf8_runes.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Case-insensitive lexicographic comparison of two UTF-8 strings.
 *
 *  Compares strings using Unicode case folding rules, producing consistent ordering regardless of
 *  letter case. Implements the same full Unicode Case Folding as `sz_utf8_case_fold`, including
 *  all one-to-many expansions (e.g., 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)) and bicameral script mappings.
 *
 *  Unlike simple byte comparison, this function correctly handles multi-byte UTF-8 sequences
 *  and expansion characters. Comparison is performed codepoint-by-codepoint after folding,
 *  not byte-by-byte, ensuring linguistically correct results.
 *
 *  @param[in] a First UTF-8 string to compare.
 *  @param[in] a_length Number of bytes in the first string.
 *  @param[in] b Second UTF-8 string to compare.
 *  @param[in] b_length Number of bytes in the second string.
 *  @return @c sz_less_k if a < b, @c sz_equal_k if a == b, @c sz_greater_k if a > b.
 *
 *  @warning Both inputs must contain valid UTF-8. Behavior is undefined for invalid input.
 *
 *  @example Basic usage:
 *  @code
 *      sz_ordering_t result = sz_utf8_case_insensitive_order("Hello", 5, "HELLO", 5);
 *      // result == sz_equal_k
 *
 *      result = sz_utf8_case_insensitive_order("straße", 7, "STRASSE", 7);
 *      // result == sz_equal_k ('ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73))
 *  @endcode
 */
SZ_DYNAMIC sz_ordering_t sz_utf8_case_insensitive_order(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                        sz_size_t b_length);

/**
 *  @brief Check if a UTF-8 string contains only case-agnostic (caseless) codepoints.
 *
 *  A codepoint is case-agnostic if ALL three conditions are true:
 *  1. **Self-folding**: Case folding produces exactly the original codepoint
 *  2. **Not bicameral**: It does not belong to any script with case distinctions
 *  3. **Not expansion target**: It does NOT appear in any multi-rune case fold expansion
 *
 *  The third condition is subtle but critical. Consider 'ʾ' (U+02BE, CA BE):
 *  - It has no case variant and folds to itself
 *  - However, 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
 *  - A needle containing 'ʾ' must match at position 1 of the folded expansion of 'ẚ'
 *  - Binary search cannot handle this, so 'ʾ' must NOT be case-agnostic
 *
 *  Case-agnostic scripts include: CJK ideographs, Hangul, digits, punctuation, most symbols,
 *  Hebrew, Arabic, Thai, Hindi (Devanagari), and many other scripts without case distinctions.
 *
 *  @section utf8_case_agnostic_usage Use Case
 *
 *  This function enables an important optimization: if both haystack and needle are fully
 *  case-agnostic, then `sz_find()` can be used directly instead of the slower
 *  `sz_utf8_case_insensitive_find()`. This is particularly valuable for:
 *
 *  - CJK text (Chinese, Japanese, Korean) - always caseless
 *  - Numeric data and punctuation-heavy content
 *  - Middle Eastern scripts (Arabic, Hebrew, Persian)
 *  - South/Southeast Asian scripts (Thai, Hindi, Vietnamese without Latin)
 *
 *  @param[in] str UTF-8 string to check.
 *  @param[in] length Number of bytes in the string.
 *  @return sz_true_k if all codepoints are case-agnostic, sz_false_k otherwise.
 *
 *  @example Optimization pattern:
 *  @code
 *      sz_cptr_t haystack = "价格：¥1234";  // Chinese + punctuation + digits
 *      sz_cptr_t needle = "¥1234";
 *
 *      if (sz_utf8_case_invariant(haystack, haystack_len) &&
 *          sz_utf8_case_invariant(needle, needle_len)) {
 *          // Fast path: use binary search
 *          result = sz_find(haystack, haystack_len, needle, needle_len);
 *      } else {
 *          // Slow path: full case-insensitive search
 *          result = sz_utf8_case_insensitive_find(haystack, haystack_len,
 *                                                  needle, needle_len, &match_len);
 *      }
 *  @endcode
 *
 *  @note This function is conservative: it returns sz_false_k for any codepoint that
 *        participates in case folding, even if the specific instance wouldn't change.
 *        For example, lowercase 'a' returns false because it's a case-folding target.
 */
SZ_DYNAMIC sz_bool_t sz_utf8_case_invariant(sz_cptr_t str, sz_size_t length);

/** @copydoc sz_utf8_case_insensitive_order */
SZ_PUBLIC sz_ordering_t sz_utf8_case_insensitive_order_serial( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);

/** @copydoc sz_utf8_case_invariant */
SZ_PUBLIC sz_bool_t sz_utf8_case_invariant_serial(sz_cptr_t str, sz_size_t length);

#if SZ_USE_ICELAKE
/** @copydoc sz_utf8_case_insensitive_order */
SZ_PUBLIC sz_ordering_t sz_utf8_case_insensitive_order_icelake( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);

/** @copydoc sz_utf8_case_invariant */
SZ_PUBLIC sz_bool_t sz_utf8_case_invariant_icelake(sz_cptr_t str, sz_size_t length);
#endif

#if SZ_USE_NEON
/** @copydoc sz_utf8_case_insensitive_order */
SZ_PUBLIC sz_ordering_t sz_utf8_case_insensitive_order_neon( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);

/** @copydoc sz_utf8_case_invariant */
SZ_PUBLIC sz_bool_t sz_utf8_case_invariant_neon(sz_cptr_t str, sz_size_t length);
#endif

#pragma region Declarations

/**
 *  @brief Case-insensitive substring search in UTF-8 strings.
 *
 *  In applications where the haystack remains largely static and memory/storage is cheap, it is recommended
 *  to pre-process the haystack into a case-folded version using Unicode Case Folding (e.g., via the ICU
 *  library) and subsequently use the simpler `sz_find()` function for repeated searches. This avoids the cost
 *  of performing full folding logic during every search operation.
 *
 *  This function applies full Unicode Case Folding as defined in the Unicode Standard (UAX #21 and
 *  CaseFolding.txt), covering all bicameral scripts, all offset-based one-to-one folds, all table-based
 *  one-to-one folds, and all normative one-to-many expansions. It doesn't however perform any normalization,
 *  like NFKC or NFC, so combining marks are treated as-is. StringZilla is intentionally locale-independent:
 *  case folding produces identical results regardless of runtime locale settings, ensuring deterministic
 *  behavior across platforms and simplifying use in multi-threaded and distributed systems.
 *
 *  The following character mappings are supported:
 *
 *  - ASCII Latin letters A–Z (U+0041–U+005A) are folded to a–z (U+0061–U+007A) using a trivial +32 offset.
 *  - Fullwidth Latin letters Ａ–Ｚ (U+FF21–U+FF3A) are folded to ａ–ｚ (U+FF41–U+FF5A) with the same +32 offset.
 *  - Cyrillic uppercase А–Я (U+0410–U+042F) are folded to а–я (U+0430–U+044F) using a +32 offset.
 *  - Armenian uppercase Ա–Ֆ (U+0531–U+0556) are folded to ա–ֆ (U+0561–U+0586) using a +48 offset.
 *  - Georgian Mtavruli letters Ა-Ჿ (U+1C90–U+1CBF, excluding 2) are folded to their Mkhedruli equivalents
 *    (U+10D0–U+10FF) using a fixed linear translation defined by the Unicode Standard.
 *  - Greek uppercase Α–Ω (U+0391–U+03A9) are folded to α–ω (U+03B1–U+03C9) via a +32 offset.
 *    Both Σ (U+03A3) and ς (U+03C2, final sigma) fold to σ (U+03C3) for consistent matching.
 *  - Latin Extended characters include numerous one-to-one folds and several one-to-many expansions, including:
 *      'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
 *      'ẞ' (U+1E9E, E1 BA 9E) → "ss" (U+0073 U+0073, 73 73)
 *    as well as mixed-case digraphs and trigraphs normalized to lowercase sequences.
 *  - Turkic dotted/dotless-I characters are handled per Unicode Case Folding (not locale-specific):
 *      'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87) — Full case folding with combining dot
 *      'I' (U+0049, 49) → 'i' (U+0069, 69) — Standard folding (not Turkic 'I' (U+0049, 49) → 'ı' (U+0131, C4 B1))
 *      'ı' (U+0131, C4 B1) → 'ı' (U+0131, C4 B1) — Already lowercase, unchanged
 *  - Lithuanian accented I/J characters with combining dots are processed as multi-codepoint expansions
 *    per CaseFolding.txt.
 *  - Additional bicameral scripts—Cherokee, Deseret, Osage, Warang Citi, Adlam—use their normative
 *    one-to-one uppercase-to-lowercase mappings defined in CaseFolding.txt.
 *
 *  Folding is applied during matching without rewriting the entire haystack. Multi-codepoint expansions,
 *  contextual folds, and combining-mark adjustments are handled at comparison time.
 *
 *  @section utf8_case_insensitive_find_algo Algorithmic Considerations
 *
 *  Case-insensitive search with full Unicode case folding is fundamentally harder than byte-level search
 *  because one-to-many expansions (e.g., 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)) break core
 *  assumptions of fast string algorithms:
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
 *  StringZilla implements several algorithms. Most importantly it first locates the longest expansion-free
 *  slice of the needle to locate against.
 *
 *  @see https://unicode-org.github.io/icu/userguide/collation/string-search.html
 *       ICU String Search - discusses why Boyer-Moore was abandoned for Unicode
 *  @see https://github.com/ClickHouse/ClickHouse/blob/master/src/Common/Volnitsky.h
 *       ClickHouse Volnitsky - hash-based substring search with UTF-8 case folding
 *  @see https://github.com/lattera/glibc/blob/master/string/str-two-way.h
 *       glibc Two-Way Algorithm - O(n) time, O(1) space string matching
 *  @see https://arxiv.org/abs/2306.10714
 *       Efficient Parameterized Pattern Matching in Sublinear Space
 *  @see https://github.com/uni-algo/uni-algo
 *       uni-algo - Unicode algorithms implementation with case-insensitive search
 *
 *  @param[in] haystack UTF-8 string to be searched.
 *  @param[in] haystack_length Number of bytes in the haystack buffer.
 *  @param[in] needle UTF-8 substring to search for.
 *  @param[inout] needle_metadata Optional pre-computed needle metadata for reuse across multiple searches.
 *  @param[out] matched_length Number of bytes in the matched region.
 *  @return Pointer to the first matching substring from @p haystack, or @c SZ_NULL_CHAR if not found.
 */
SZ_DYNAMIC sz_cptr_t sz_utf8_case_insensitive_find( //
    sz_cptr_t haystack, sz_size_t haystack_length,  //
    sz_cptr_t needle, sz_size_t needle_length,      //
    sz_utf8_case_insensitive_needle_metadata_t *needle_metadata, sz_size_t *matched_length);

/** @copydoc sz_utf8_case_insensitive_find */
SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_serial( //
    sz_cptr_t haystack, sz_size_t haystack_length,        //
    sz_cptr_t needle, sz_size_t needle_length,            //
    sz_utf8_case_insensitive_needle_metadata_t *needle_metadata, sz_size_t *matched_length);

#if SZ_USE_ICELAKE
/** @copydoc sz_utf8_case_insensitive_find */
SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_icelake( //
    sz_cptr_t haystack, sz_size_t haystack_length,         //
    sz_cptr_t needle, sz_size_t needle_length,             //
    sz_utf8_case_insensitive_needle_metadata_t *needle_metadata, sz_size_t *matched_length);
#endif

#if SZ_USE_NEON
/** @copydoc sz_utf8_case_insensitive_find */
SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_neon( //
    sz_cptr_t haystack, sz_size_t haystack_length,      //
    sz_cptr_t needle, sz_size_t needle_length,          //
    sz_utf8_case_insensitive_needle_metadata_t *needle_metadata, sz_size_t *matched_length);
#endif

#pragma endregion // Core API

/*  Backends in serial-first order: every SIMD/ISA header includes `serial.h` for the shared
 *  scaffolding (verify helpers, hash-free finders, Rabin-Karp serial search, the safety classifier
 *  and needle-metadata builder), then layers its own kernels (or delegates back to serial). */
#include "stringzilla/utf8_case_insensitive/serial.h"
#include "stringzilla/utf8_case_insensitive/icelake.h"
#include "stringzilla/utf8_case_insensitive/neon.h"
#include "stringzilla/utf8_case_insensitive/v128.h"
#include "stringzilla/utf8_case_insensitive/v128relaxed.h"
#include "stringzilla/utf8_case_insensitive/rvv.h"
#include "stringzilla/utf8_case_insensitive/lasx.h"
#include "stringzilla/utf8_case_insensitive/powervsx.h"

/*  Pick the right implementation for the case-insensitive UTF-8 algorithms.
 *  To override this behavior and precompile all backends - set `SZ_DYNAMIC_DISPATCH` to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_cptr_t sz_utf8_case_insensitive_find(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                   sz_size_t needle_length,
                                                   sz_utf8_case_insensitive_needle_metadata_t *needle_metadata,
                                                   sz_size_t *matched_length) {
#if SZ_USE_V128RELAXED
    return sz_utf8_case_insensitive_find_v128relaxed(haystack, haystack_length, needle, needle_length, needle_metadata,
                                                     matched_length);
#elif SZ_USE_V128
    return sz_utf8_case_insensitive_find_v128(haystack, haystack_length, needle, needle_length, needle_metadata,
                                              matched_length);
#elif SZ_USE_RVV
    return sz_utf8_case_insensitive_find_rvv(haystack, haystack_length, needle, needle_length, needle_metadata,
                                             matched_length);
#elif SZ_USE_LASX
    return sz_utf8_case_insensitive_find_lasx(haystack, haystack_length, needle, needle_length, needle_metadata,
                                              matched_length);
#elif SZ_USE_POWERVSX
    return sz_utf8_case_insensitive_find_powervsx(haystack, haystack_length, needle, needle_length, needle_metadata,
                                                  matched_length);
#elif SZ_USE_ICELAKE
    return sz_utf8_case_insensitive_find_icelake(haystack, haystack_length, needle, needle_length, needle_metadata,
                                                 matched_length);
#else
    return sz_utf8_case_insensitive_find_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                                matched_length);
#endif
}

SZ_DYNAMIC sz_ordering_t sz_utf8_case_insensitive_order(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                        sz_size_t b_length) {
    return sz_utf8_case_insensitive_order_serial(a, a_length, b, b_length);
}

SZ_DYNAMIC sz_bool_t sz_utf8_case_invariant(sz_cptr_t str, sz_size_t length) {
#if SZ_USE_V128RELAXED
    return sz_utf8_case_invariant_v128relaxed(str, length);
#elif SZ_USE_V128
    return sz_utf8_case_invariant_v128(str, length);
#elif SZ_USE_RVV
    return sz_utf8_case_invariant_rvv(str, length);
#elif SZ_USE_LASX
    return sz_utf8_case_invariant_lasx(str, length);
#elif SZ_USE_POWERVSX
    return sz_utf8_case_invariant_powervsx(str, length);
#elif SZ_USE_ICELAKE
    return sz_utf8_case_invariant_icelake(str, length);
#else
    return sz_utf8_case_invariant_serial(str, length);
#endif
}

#endif            // !SZ_DYNAMIC_DISPATCH
#pragma endregion // Compile Time Dispatching

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_CASE_INSENSITIVE_H_
