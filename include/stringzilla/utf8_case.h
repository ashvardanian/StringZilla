/**
 *  @brief  Hardware-accelerated UTF-8 text processing utilities, that require unpacking into UTF-32 runes.
 *  @file   utf8_case.h
 *  @author Ash Vardanian
 *
 *  Work in progress:
 *
 *  - `sz_utf8_case_fold` - Unicode case folding for codepoints
 *  - `sz_utf8_case_insensitive_find` - case-insensitive substring search in UTF-8 strings
 *  - `sz_utf8_case_insensitive_order` - case-insensitive lexicographical comparison of UTF-8 strings
 *  - `sz_utf8_case_invariant` - check if a string contains only case-agnostic (caseless) codepoints
 *
 *  It's important to remember that UTF-8 is just one of many possible Unicode encodings.
 *  Unicode is a versioned standard and we implement its locale-independent specification v17.
 *  All algorithms are fully compliant with the specification and handle all edge cases.
 *
 *  On fast vectorized paths, unlike other parts of StringZilla, there may be significant algorithmic
 *  differences between different ISA versions. Most of them are designed to be practical in common
 *  use cases, targeting the most common languages on the Internet.
 *
 *  Rank  Language     Script        UTF-8 Bytes  Has Case?  Case Folding Notes
 *  -------------------------------------------------------------------------------------------
 *  1     English      Latin         1            Yes        Simple +32 offset (A-Z)
 *  2     Russian      Cyrillic      2            Yes        Simple +32 offset (А-Я)
 *  3     Spanish      Latin         1-2          Yes        Mostly ASCII, few 2-byte (ñ, á, é)
 *  4     German       Latin         1-2          Yes        Includes 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
 *  5     French       Latin         1-2          Yes        Mostly ASCII, accents (é, è, ç)
 *  6     Japanese     CJK + Kana    3            No*        No case, but has fullwidth A-Z in addresses
 *  7     Portuguese   Latin         1-2          Yes        Similar to Spanish
 *  8     Chinese      CJK           3            No         No case folding
 *  9     Italian      Latin         1-2          Yes        Similar to Spanish
 *  10    Polish       Latin         1-2          Yes        ASCII + ą, ę, ł, ż, etc.
 *  11    Turkish      Latin         1-2          Yes        Locale-specific İ/ı intentionally not applied
 *  12    Dutch        Latin         1            Yes        Almost pure ASCII
 *  13    Persian      Arabic        2            No         RTL, no case
 *  14    Vietnamese   Latin         2-3          Yes        Heavy diacritics (ă, ơ, ư), odd/even
 *  15    Korean       Hangul        3            No         No case folding
 *  16    Arabic       Arabic        2            No         RTL, no case
 *  17    Indonesian   Latin         1            Yes        Pure ASCII
 *  18    Greek        Greek         2            Yes        +32 offset, σ/ς handling
 *  19    Ukrainian    Cyrillic      2            Yes        Similar to Russian
 *  20    Czech        Latin         1-2          Yes        ASCII + ě, š, č, ř, ž
 *
 *  This doesn't, however, cover many other relevant subranges of Unicode.
 */
#ifndef STRINGZILLA_UTF8_CASE_H_
#define STRINGZILLA_UTF8_CASE_H_

#include "types.h"
#include "find.h" // `sz_find`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Lightweight metadata for a safe window within a script path.
 *
 *  This struct only contains location and length information needed for kernel selection.
 *  The actual case-folding and probe computation is deferred until after the best kernel
 *  is chosen, using `sz_utf8_case_insensitive_needle_metadata_t`.
 */
typedef struct sz_utf8_string_slice_t {
    sz_size_t offset;       // Start offset in original needle (bytes)
    sz_size_t length;       // Byte length in original needle
    sz_size_t runes_within; // Codepoints within this window
} sz_utf8_string_slice_t;

/**
 *  @brief Tiny wrapper for substring search queries with pre-located probing positions.
 *
 *  Reuse this structure to avoid re-computing the probe positions for the same needle multiple times.
 *  It's created internally in a multi-step process of:
 *  1. locating the longest "safe" slice of the needle with respect to different SIMD folding kernels,
 *  2. shrinking it further to find the most diverse slice that fits into a `folded_slice` when case-folded.
 *
 *  Unlike the exact substring search kernels, it uses 4 probe positions instead of 3:
 *    - first: implicit at `folded_slice[0]`
 *    - second: `probe_second`
 *    - third: `probe_third`
 *    - last: implicit at `folded_slice[folded_slice_length - 1]`
 */
typedef struct sz_utf8_case_insensitive_needle_metadata_t {
    sz_size_t offset_in_unfolded; // Number of bytes in the "unsafe LONG NeedLe" before the safe & folded part
    sz_size_t length_in_unfolded; // Number of bytes in the safe part of the actual "NeedLe" before folding
    sz_u8_t folded_slice[16];
    sz_u8_t folded_slice_length;
    sz_u8_t probe_second; // Position of the second relevant character in the folded slice
    sz_u8_t probe_third;  // Position of the third relevant character in the folded slice
    sz_u8_t kernel_id;    // The unique identifier of the kernel best suited for searching this needle
} sz_utf8_case_insensitive_needle_metadata_t;

/**
 *  @brief  Apply Unicode case folding to a UTF-8 string.
 *
 *  Case folding normalizes text for case-insensitive comparisons by mapping uppercase letters
 *  to their lowercase equivalents and handling special expansions defined in Unicode CaseFolding.txt.
 *
 *  @section Buffer Sizing
 *
 *  The destination buffer must be at least `source_length * 3` bytes to guarantee sufficient space
 *  for worst-case expansion. The maximum expansion ratio is 3:1 (3x), which occurs with Greek
 *  characters that expand to three codepoints under case folding.
 *
 *  Worst-case example: 'ΐ' (U+0390, CE 90) → "ΐ" (U+03B9 U+0308 U+0301, CE B9 CC 88 CC 81) (2 bytes → 6 bytes).
 *  A string of N such characters would expand from 2N to 6N bytes (3x expansion).
 *
 *  @param[in] source UTF-8 string to be case-folded.
 *  @param[in] source_length Number of bytes in the source buffer.
 *  @param[out] destination Buffer to write the case-folded UTF-8 string.
 *  @return Number of bytes written to the destination buffer.
 *
 *  @warning The caller must ensure the destination buffer is large enough. No bounds checking
 *           is performed. Use `source_length * 3` for safety.
 *  @warning The source must contain valid UTF-8. Behavior is undefined for invalid input.
 *
 *  @example Basic usage:
 *  @code
 *      char const *source = "HELLO";
 *      sz_size_t capacity = 5 * 3; // Safe overestimate
 *      char destination[15];
 *      sz_size_t result_length = sz_utf8_case_fold(source, 5, destination);
 *      // destination now contains "hello", result_length = 5
 *  @endcode
 */
SZ_DYNAMIC sz_size_t sz_utf8_case_fold(        //
    sz_cptr_t source, sz_size_t source_length, //
    sz_ptr_t destination);

/**
 *  @brief  Convert a UTF-8 string to uppercase using Full Unicode Case Mapping.
 *
 *  This function converts all lowercase letters in the source string to their uppercase equivalents,
 *  including handling of one-to-many expansions (e.g., 'ß' → "SS", 'ﬁ' → "FI").
 *
 *  @param source       Pointer to the source UTF-8 string.
 *  @param source_length Number of bytes in source (not characters).
 *  @param destination  Buffer to write the uppercased result. Must be at least 3x source_length bytes.
 *  @return Number of bytes written to destination.
 */
SZ_DYNAMIC sz_size_t sz_utf8_case_upper(          //
    sz_cptr_t source, sz_size_t source_length, //
    sz_ptr_t destination);

/**
 *  @brief  Case-insensitive substring search in UTF-8 strings.
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

/**
 *  @brief  Case-insensitive lexicographic comparison of two UTF-8 strings.
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
 *  @brief  Check if a UTF-8 string contains only case-agnostic (caseless) codepoints.
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

#pragma endregion

#pragma region Platform-Specific Backends

/** @copydoc sz_utf8_case_fold */
SZ_PUBLIC sz_size_t sz_utf8_case_fold_serial(  //
    sz_cptr_t source, sz_size_t source_length, //
    sz_ptr_t destination);

/** @copydoc sz_utf8_case_upper */
SZ_PUBLIC sz_size_t sz_utf8_case_upper_serial(  //
    sz_cptr_t source, sz_size_t source_length, //
    sz_ptr_t destination);

/** @copydoc sz_utf8_case_insensitive_find */
SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_serial( //
    sz_cptr_t haystack, sz_size_t haystack_length,        //
    sz_cptr_t needle, sz_size_t needle_length,            //
    sz_utf8_case_insensitive_needle_metadata_t *needle_metadata, sz_size_t *matched_length);

/** @copydoc sz_utf8_case_insensitive_order */
SZ_PUBLIC sz_ordering_t sz_utf8_case_insensitive_order_serial( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);

/** @copydoc sz_utf8_case_invariant */
SZ_PUBLIC sz_bool_t sz_utf8_case_invariant_serial(sz_cptr_t str, sz_size_t length);

#if SZ_USE_ICE

/** @copydoc sz_utf8_case_fold */
SZ_PUBLIC sz_size_t sz_utf8_case_fold_ice( //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination);

/** @copydoc sz_utf8_case_upper */
SZ_PUBLIC sz_size_t sz_utf8_case_upper_ice( //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination);

/** @copydoc sz_utf8_case_insensitive_find */
SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_ice( //
    sz_cptr_t haystack, sz_size_t haystack_length,     //
    sz_cptr_t needle, sz_size_t needle_length,         //
    sz_utf8_case_insensitive_needle_metadata_t *needle_metadata, sz_size_t *matched_length);

/** @copydoc sz_utf8_case_insensitive_order */
SZ_PUBLIC sz_ordering_t sz_utf8_case_insensitive_order_ice( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);

/** @copydoc sz_utf8_case_invariant */
SZ_PUBLIC sz_bool_t sz_utf8_case_invariant_ice(sz_cptr_t str, sz_size_t length);

#endif

#if SZ_USE_NEON

/** @copydoc sz_utf8_case_fold */
SZ_PUBLIC sz_size_t sz_utf8_case_fold_neon( //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination);

/** @copydoc sz_utf8_case_insensitive_find */
SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_neon( //
    sz_cptr_t haystack, sz_size_t haystack_length,      //
    sz_cptr_t needle, sz_size_t needle_length,          //
    sz_utf8_case_insensitive_needle_metadata_t *needle_metadata, sz_size_t *matched_length);

/** @copydoc sz_utf8_case_insensitive_order */
SZ_PUBLIC sz_ordering_t sz_utf8_case_insensitive_order_neon( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);

/** @copydoc sz_utf8_case_invariant */
SZ_PUBLIC sz_bool_t sz_utf8_case_invariant_neon(sz_cptr_t str, sz_size_t length);

#endif

#pragma endregion

/*  Implementation Section
 *  Following the same pattern as find.h with implementations in the header file.
 */

#pragma region Serial Implementation

/**  Helper macro for readable assertions - use for SIMD implementation reference */
#define sz_is_in_range_(x, lo, hi) ((x) >= (lo) && (x) <= (hi))

/**
 *  @brief  Fold a Unicode codepoint to its case-folded form (Unicode 17.0).
 *
 *  Optimization strategy:
 *  - Single-comparison range checks: `(sz_u32_t)(rune - base) <= size` instead of two comparisons
 *  - Combined upper+lower ranges: check both cases, apply offset only for uppercase (branchless)
 *  - Combined even/odd ranges: check full range, apply +1 only for uppercase parity
 *  - Hierarchical by UTF-8 byte width for early exit on common cases
 *  - Per-section switches for irregular mappings (better compiler optimization)
 *
 *  Each range check includes an assertion with traditional bounds for SIMD implementation reference.
 */
SZ_INTERNAL sz_size_t sz_unicode_fold_codepoint_(sz_rune_t rune, sz_rune_t *folded) {
    // clang-format off

    // 1-byte UTF-8 (U+0000-007F): ASCII - only A-Z needs folding
    if (rune <= 0x7F) {
        if ((sz_u32_t)(rune - 0x41) <= 25) { // A-Z: 0x41-0x5A (26 chars)
            sz_assert_(sz_is_in_range_(rune, 0x0041, 0x005A));
            folded[0] = rune + 0x20; return 1; }
        folded[0] = rune; return 1;  // digits, punctuation, control chars unchanged
    }

    // 2-byte UTF-8 (U+0080-07FF): Latin, Greek, Cyrillic, Armenian
    if (rune <= 0x7FF) {
        // Cyrillic А-я: 0x0410-0x044F (upper 0x0410-0x042F, lower 0x0430-0x044F)
        if ((sz_u32_t)(rune - 0x0410) <= 0x3F) {
            sz_assert_(sz_is_in_range_(rune, 0x0410, 0x044F));
            folded[0] = rune + ((rune <= 0x042F) * 0x20); return 1; } // +32 if upper, +0 if lower

        // Latin-1 À-þ: 0x00C0-0x00FE (upper 0x00C0-0x00DE, lower 0x00E0-0x00FE)
        if ((sz_u32_t)(rune - 0x00C0) <= 0x3E) {
            sz_assert_(sz_is_in_range_(rune, 0x00C0, 0x00FE));
            if ((rune | 0x20) == 0xF7) { folded[0] = rune; return 1; } // × (D7) and ÷ (F7) unchanged
            // 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
            if (rune == 0x00DF) { folded[0] = 0x0073; folded[1] = 0x0073; return 2; }
            folded[0] = rune + ((rune <= 0x00DE) * 0x20); return 1; }

        // Greek Α-Ρ: 0x0391-0x03A1 → α-ρ (+32)
        if ((sz_u32_t)(rune - 0x0391) <= 0x10) {
            sz_assert_(sz_is_in_range_(rune, 0x0391, 0x03A1));
            folded[0] = rune + 0x20; return 1; }

        // Greek Σ-Ϋ: 0x03A3-0x03AB → σ-ϋ (+32)
        if ((sz_u32_t)(rune - 0x03A3) <= 0x08) {
            sz_assert_(sz_is_in_range_(rune, 0x03A3, 0x03AB));
            folded[0] = rune + 0x20; return 1; }

        // Cyrillic Ѐ-Џ: 0x0400-0x040F → ѐ-џ (+80)
        if ((sz_u32_t)(rune - 0x0400) <= 0x0F) {
            sz_assert_(sz_is_in_range_(rune, 0x0400, 0x040F));
            folded[0] = rune + 0x50; return 1; }

        // Armenian Ա-Ֆ: 0x0531-0x0556 → ա-ֆ (+48)
        if ((sz_u32_t)(rune - 0x0531) <= 0x25) {
            sz_assert_(sz_is_in_range_(rune, 0x0531, 0x0556));
            folded[0] = rune + 0x30; return 1; }

        // Greek Έ-Ί: 0x0388-0x038A (+37)
        if ((sz_u32_t)(rune - 0x0388) <= 0x02) {
            sz_assert_(sz_is_in_range_(rune, 0x0388, 0x038A));
            folded[0] = rune + 0x25; return 1; }

        // Greek Ͻ-Ͽ: 0x03FD-0x03FF → ͻ-Ϳ (-130)
        if ((sz_u32_t)(rune - 0x03FD) <= 0x02) {
            sz_assert_(sz_is_in_range_(rune, 0x03FD, 0x03FF));
            folded[0] = rune - 130; return 1; }

        // Next let's handle the even/odd parity-based ranges
        sz_u32_t is_even = ((rune & 1) == 0);

        // Latin Extended-A: Ā-Į (0x0100-0x012E, even → +1)
        if ((sz_u32_t)(rune - 0x0100) <= 0x2E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0100, 0x012E));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-A: Ĳ-Ķ (0x0132-0x0136, even → +1)
        if ((sz_u32_t)(rune - 0x0132) <= 0x04 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0132, 0x0136));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-A: Ĺ-Ň (0x0139-0x0147, odd → +1)
        if ((sz_u32_t)(rune - 0x0139) <= 0x0E && !is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0139, 0x0147));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-A: Ŋ-Ŷ (0x014A-0x0176, even → +1)
        if ((sz_u32_t)(rune - 0x014A) <= 0x2C && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x014A, 0x0176));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-A: Ź-Ž (0x0179-0x017D, odd → +1)
        if ((sz_u32_t)(rune - 0x0179) <= 0x04 && !is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0179, 0x017D));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-B: Ǎ-Ǜ (0x01CD-0x01DB, odd → +1)
        if ((sz_u32_t)(rune - 0x01CD) <= 0x0E && !is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x01CD, 0x01DB));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-B: Ǟ-Ǯ (0x01DE-0x01EE, even → +1)
        if ((sz_u32_t)(rune - 0x01DE) <= 0x10 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x01DE, 0x01EE));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-B: Ǹ-Ǿ (0x01F8-0x01FE, even → +1)
        if ((sz_u32_t)(rune - 0x01F8) <= 0x06 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x01F8, 0x01FE));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-B: Ȁ-Ȟ (0x0200-0x021E, even → +1)
        if ((sz_u32_t)(rune - 0x0200) <= 0x1E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0200, 0x021E));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-B: Ȣ-Ȳ (0x0222-0x0232, even → +1)
        if ((sz_u32_t)(rune - 0x0222) <= 0x10 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0222, 0x0232));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-B: Ɇ-Ɏ (0x0246-0x024E, even → +1)
        if ((sz_u32_t)(rune - 0x0246) <= 0x08 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0246, 0x024E));
            folded[0] = rune + 1; return 1; }

        // Greek archaic: Ͱ-Ͳ (0x0370-0x0372, even → +1)
        if ((sz_u32_t)(rune - 0x0370) <= 0x02 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0370, 0x0372));
            folded[0] = rune + 1; return 1; }

        // Greek archaic: Ϙ-Ϯ (0x03D8-0x03EE, even → +1)
        if ((sz_u32_t)(rune - 0x03D8) <= 0x16 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x03D8, 0x03EE));
            folded[0] = rune + 1; return 1; }

        // Cyrillic extended: Ѡ-Ҁ (0x0460-0x0480, even → +1)
        if ((sz_u32_t)(rune - 0x0460) <= 0x20 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0460, 0x0480));
            folded[0] = rune + 1; return 1; }

        // Cyrillic extended: Ҋ-Ҿ (0x048A-0x04BE, even → +1)
        if ((sz_u32_t)(rune - 0x048A) <= 0x34 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x048A, 0x04BE));
            folded[0] = rune + 1; return 1; }

        // Cyrillic extended: Ӂ-Ӎ (0x04C1-0x04CD, odd → +1)
        if ((sz_u32_t)(rune - 0x04C1) <= 0x0C && !is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x04C1, 0x04CD));
            folded[0] = rune + 1; return 1; }

        // Cyrillic extended: Ӑ-Ӿ (0x04D0-0x04FE, even → +1)
        if ((sz_u32_t)(rune - 0x04D0) <= 0x2E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x04D0, 0x04FE));
            folded[0] = rune + 1; return 1; }

        // Cyrillic extended: Ԁ-Ԯ (0x0500-0x052E, even → +1)
        if ((sz_u32_t)(rune - 0x0500) <= 0x2E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x0500, 0x052E));
            folded[0] = rune + 1; return 1; }

        // Next let's handle the 2-byte irregular one-to-one mappings
        switch (rune) {
        // Latin-1 Supplement specials
        case 0x00B5: folded[0] = 0x03BC; return 1; // 'µ' (U+00B5, C2 B5) → 'μ' (U+03BC, CE BC)
        case 0x0178: folded[0] = 0x00FF; return 1; // 'Ÿ' (U+0178, C5 B8) → 'ÿ' (U+00FF, C3 BF)
        case 0x017F: folded[0] = 0x0073; return 1; // 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
        // Latin Extended-B: African/IPA letters (0x0181-0x01BF)
        case 0x0181: folded[0] = 0x0253; return 1; // 'Ɓ' (U+0181, C6 81) → 'ɓ' (U+0253, C9 93)
        case 0x0182: folded[0] = 0x0183; return 1; // 'Ƃ' (U+0182, C6 82) → 'ƃ' (U+0183, C6 83)
        case 0x0184: folded[0] = 0x0185; return 1; // 'Ƅ' (U+0184, C6 84) → 'ƅ' (U+0185, C6 85)
        case 0x0186: folded[0] = 0x0254; return 1; // 'Ɔ' (U+0186, C6 86) → 'ɔ' (U+0254, C9 94)
        case 0x0187: folded[0] = 0x0188; return 1; // 'Ƈ' (U+0187, C6 87) → 'ƈ' (U+0188, C6 88)
        case 0x0189: folded[0] = 0x0256; return 1; // 'Ɖ' (U+0189, C6 89) → 'ɖ' (U+0256, C9 96)
        case 0x018A: folded[0] = 0x0257; return 1; // 'Ɗ' (U+018A, C6 8A) → 'ɗ' (U+0257, C9 97)
        case 0x018B: folded[0] = 0x018C; return 1; // 'Ƌ' (U+018B, C6 8B) → 'ƌ' (U+018C, C6 8C)
        case 0x018E: folded[0] = 0x01DD; return 1; // 'Ǝ' (U+018E, C6 8E) → 'ǝ' (U+01DD, C7 9D)
        case 0x018F: folded[0] = 0x0259; return 1; // 'Ə' (U+018F, C6 8F) → 'ə' (U+0259, C9 99)
        case 0x0190: folded[0] = 0x025B; return 1; // 'Ɛ' (U+0190, C6 90) → 'ɛ' (U+025B, C9 9B)
        case 0x0191: folded[0] = 0x0192; return 1; // 'Ƒ' (U+0191, C6 91) → 'ƒ' (U+0192, C6 92)
        case 0x0193: folded[0] = 0x0260; return 1; // 'Ɠ' (U+0193, C6 93) → 'ɠ' (U+0260, C9 A0)
        case 0x0194: folded[0] = 0x0263; return 1; // 'Ɣ' (U+0194, C6 94) → 'ɣ' (U+0263, C9 A3)
        case 0x0196: folded[0] = 0x0269; return 1; // 'Ɩ' (U+0196, C6 96) → 'ɩ' (U+0269, C9 A9)
        case 0x0197: folded[0] = 0x0268; return 1; // 'Ɨ' (U+0197, C6 97) → 'ɨ' (U+0268, C9 A8)
        case 0x0198: folded[0] = 0x0199; return 1; // 'Ƙ' (U+0198, C6 98) → 'ƙ' (U+0199, C6 99)
        case 0x019C: folded[0] = 0x026F; return 1; // 'Ɯ' (U+019C, C6 9C) → 'ɯ' (U+026F, C9 AF)
        case 0x019D: folded[0] = 0x0272; return 1; // 'Ɲ' (U+019D, C6 9D) → 'ɲ' (U+0272, C9 B2)
        case 0x019F: folded[0] = 0x0275; return 1; // 'Ɵ' (U+019F, C6 9F) → 'ɵ' (U+0275, C9 B5)
        case 0x01A0: folded[0] = 0x01A1; return 1; // 'Ơ' (U+01A0, C6 A0) → 'ơ' (U+01A1, C6 A1)
        case 0x01A2: folded[0] = 0x01A3; return 1; // 'Ƣ' (U+01A2, C6 A2) → 'ƣ' (U+01A3, C6 A3)
        case 0x01A4: folded[0] = 0x01A5; return 1; // 'Ƥ' (U+01A4, C6 A4) → 'ƥ' (U+01A5, C6 A5)
        case 0x01A6: folded[0] = 0x0280; return 1; // 'Ʀ' (U+01A6, C6 A6) → 'ʀ' (U+0280, CA 80)
        case 0x01A7: folded[0] = 0x01A8; return 1; // 'Ƨ' (U+01A7, C6 A7) → 'ƨ' (U+01A8, C6 A8)
        case 0x01A9: folded[0] = 0x0283; return 1; // 'Ʃ' (U+01A9, C6 A9) → 'ʃ' (U+0283, CA 83)
        case 0x01AC: folded[0] = 0x01AD; return 1; // 'Ƭ' (U+01AC, C6 AC) → 'ƭ' (U+01AD, C6 AD)
        case 0x01AE: folded[0] = 0x0288; return 1; // 'Ʈ' (U+01AE, C6 AE) → 'ʈ' (U+0288, CA 88)
        case 0x01AF: folded[0] = 0x01B0; return 1; // 'Ư' (U+01AF, C6 AF) → 'ư' (U+01B0, C6 B0)
        case 0x01B1: folded[0] = 0x028A; return 1; // 'Ʊ' (U+01B1, C6 B1) → 'ʊ' (U+028A, CA 8A)
        case 0x01B2: folded[0] = 0x028B; return 1; // 'Ʋ' (U+01B2, C6 B2) → 'ʋ' (U+028B, CA 8B)
        case 0x01B3: folded[0] = 0x01B4; return 1; // 'Ƴ' (U+01B3, C6 B3) → 'ƴ' (U+01B4, C6 B4)
        case 0x01B5: folded[0] = 0x01B6; return 1; // 'Ƶ' (U+01B5, C6 B5) → 'ƶ' (U+01B6, C6 B6)
        case 0x01B7: folded[0] = 0x0292; return 1; // 'Ʒ' (U+01B7, C6 B7) → 'ʒ' (U+0292, CA 92)
        case 0x01B8: folded[0] = 0x01B9; return 1; // 'Ƹ' (U+01B8, C6 B8) → 'ƹ' (U+01B9, C6 B9)
        case 0x01BC: folded[0] = 0x01BD; return 1; // 'Ƽ' (U+01BC, C6 BC) → 'ƽ' (U+01BD, C6 BD)
        // Digraphs: Serbian/Croatian DŽ, LJ, NJ and DZ
        case 0x01C4: folded[0] = 0x01C6; return 1; // 'Ǆ' (U+01C4, C7 84) → 'ǆ' (U+01C6, C7 86)
        case 0x01C5: folded[0] = 0x01C6; return 1; // 'ǅ' (U+01C5, C7 85) → 'ǆ' (U+01C6, C7 86)
        case 0x01C7: folded[0] = 0x01C9; return 1; // 'Ǉ' (U+01C7, C7 87) → 'ǉ' (U+01C9, C7 89)
        case 0x01C8: folded[0] = 0x01C9; return 1; // 'ǈ' (U+01C8, C7 88) → 'ǉ' (U+01C9, C7 89)
        case 0x01CA: folded[0] = 0x01CC; return 1; // 'Ǌ' (U+01CA, C7 8A) → 'ǌ' (U+01CC, C7 8C)
        case 0x01CB: folded[0] = 0x01CC; return 1; // 'ǋ' (U+01CB, C7 8B) → 'ǌ' (U+01CC, C7 8C)
        case 0x01F1: folded[0] = 0x01F3; return 1; // 'Ǳ' (U+01F1, C7 B1) → 'ǳ' (U+01F3, C7 B3)
        case 0x01F2: folded[0] = 0x01F3; return 1; // 'ǲ' (U+01F2, C7 B2) → 'ǳ' (U+01F3, C7 B3)
        // Latin Extended-B: isolated irregulars
        case 0x01F4: folded[0] = 0x01F5; return 1; // 'Ǵ' (U+01F4, C7 B4) → 'ǵ' (U+01F5, C7 B5)
        case 0x01F6: folded[0] = 0x0195; return 1; // 'Ƕ' (U+01F6, C7 B6) → 'ƕ' (U+0195, C6 95)
        case 0x01F7: folded[0] = 0x01BF; return 1; // 'Ƿ' (U+01F7, C7 B7) → 'ƿ' (U+01BF, C6 BF)
        case 0x0220: folded[0] = 0x019E; return 1; // 'Ƞ' (U+0220, C8 A0) → 'ƞ' (U+019E, C6 9E)
        case 0x023A: folded[0] = 0x2C65; return 1; // 'Ⱥ' (U+023A, C8 BA) → 'ⱥ' (U+2C65, E2 B1 A5)
        case 0x023B: folded[0] = 0x023C; return 1; // 'Ȼ' (U+023B, C8 BB) → 'ȼ' (U+023C, C8 BC)
        case 0x023D: folded[0] = 0x019A; return 1; // 'Ƚ' (U+023D, C8 BD) → 'ƚ' (U+019A, C6 9A)
        case 0x023E: folded[0] = 0x2C66; return 1; // 'Ⱦ' (U+023E, C8 BE) → 'ⱦ' (U+2C66, E2 B1 A6)
        case 0x0241: folded[0] = 0x0242; return 1; // 'Ɂ' (U+0241, C9 81) → 'ɂ' (U+0242, C9 82)
        case 0x0243: folded[0] = 0x0180; return 1; // 'Ƀ' (U+0243, C9 83) → 'ƀ' (U+0180, C6 80)
        case 0x0244: folded[0] = 0x0289; return 1; // 'Ʉ' (U+0244, C9 84) → 'ʉ' (U+0289, CA 89)
        case 0x0245: folded[0] = 0x028C; return 1; // 'Ʌ' (U+0245, C9 85) → 'ʌ' (U+028C, CA 8C)
        // Greek: combining iota, accented vowels, variant forms
        case 0x0345: folded[0] = 0x03B9; return 1; // 'ͅ' (U+0345, CD 85) → 'ι' (U+03B9, CE B9)
        case 0x0376: folded[0] = 0x0377; return 1; // 'Ͷ' (U+0376, CD B6) → 'ͷ' (U+0377, CD B7)
        case 0x037F: folded[0] = 0x03F3; return 1; // 'Ϳ' (U+037F, CD BF) → 'ϳ' (U+03F3, CF B3)
        case 0x0386: folded[0] = 0x03AC; return 1; // 'Ά' (U+0386, CE 86) → 'ά' (U+03AC, CE AC)
        case 0x038C: folded[0] = 0x03CC; return 1; // 'Ό' (U+038C, CE 8C) → 'ό' (U+03CC, CF 8C)
        case 0x038E: folded[0] = 0x03CD; return 1; // 'Ύ' (U+038E, CE 8E) → 'ύ' (U+03CD, CF 8D)
        case 0x038F: folded[0] = 0x03CE; return 1; // 'Ώ' (U+038F, CE 8F) → 'ώ' (U+03CE, CF 8E)
        case 0x03C2: folded[0] = 0x03C3; return 1; // 'ς' (U+03C2, CF 82) → 'σ' (U+03C3, CF 83)
        case 0x03CF: folded[0] = 0x03D7; return 1; // 'Ϗ' (U+03CF, CF 8F) → 'ϗ' (U+03D7, CF 97)
        case 0x03D0: folded[0] = 0x03B2; return 1; // 'ϐ' (U+03D0, CF 90) → 'β' (U+03B2, CE B2)
        case 0x03D1: folded[0] = 0x03B8; return 1; // 'ϑ' (U+03D1, CF 91) → 'θ' (U+03B8, CE B8)
        case 0x03D5: folded[0] = 0x03C6; return 1; // 'ϕ' (U+03D5, CF 95) → 'φ' (U+03C6, CF 86)
        case 0x03D6: folded[0] = 0x03C0; return 1; // 'ϖ' (U+03D6, CF 96) → 'π' (U+03C0, CF 80)
        case 0x03F0: folded[0] = 0x03BA; return 1; // 'ϰ' (U+03F0, CF B0) → 'κ' (U+03BA, CE BA)
        case 0x03F1: folded[0] = 0x03C1; return 1; // 'ϱ' (U+03F1, CF B1) → 'ρ' (U+03C1, CF 81)
        case 0x03F4: folded[0] = 0x03B8; return 1; // 'ϴ' (U+03F4, CF B4) → 'θ' (U+03B8, CE B8)
        case 0x03F5: folded[0] = 0x03B5; return 1; // 'ϵ' (U+03F5, CF B5) → 'ε' (U+03B5, CE B5)
        case 0x03F7: folded[0] = 0x03F8; return 1; // 'Ϸ' (U+03F7, CF B7) → 'ϸ' (U+03F8, CF B8)
        case 0x03F9: folded[0] = 0x03F2; return 1; // 'Ϲ' (U+03F9, CF B9) → 'ϲ' (U+03F2, CF B2)
        case 0x03FA: folded[0] = 0x03FB; return 1; // 'Ϻ' (U+03FA, CF BA) → 'ϻ' (U+03FB, CF BB)
        // Cyrillic: palochka
        case 0x04C0: folded[0] = 0x04CF; return 1; // 'Ӏ' (U+04C0, D3 80) → 'ӏ' (U+04CF, D3 8F)
        }

        // 2-byte one-to-many expansions
        switch (rune) {
        // ß handled inline in Latin-1 range above... interestingly the capital Eszett is in the 3-byte range!
        // case 0x00DF: folded[0] = 0x0073; folded[1] = 0x0073; return 2;
        
        // 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
        case 0x0130: folded[0] = 0x0069; folded[1] = 0x0307; return 2;
        // 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
        case 0x0149: folded[0] = 0x02BC; folded[1] = 0x006E; return 2;
        // 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C)
        case 0x01F0: folded[0] = 0x006A; folded[1] = 0x030C; return 2;
        // 'ΐ' (U+0390, CE 90) → "ΐ" (U+03B9 U+0308 U+0301, CE B9 CC 88 CC 81)
        case 0x0390: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0301; return 3;
        // 'ΰ' (U+03B0, CE B0) → "ΰ" (U+03C5 U+0308 U+0301, CF 85 CC 88 CC 81)
        case 0x03B0: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0301; return 3;
        // 'և' (U+0587, D6 87) → "եւ" (U+0565 U+0582, D5 A5 D6 82)
        case 0x0587: folded[0] = 0x0565; folded[1] = 0x0582; return 2;
        }

        folded[0] = rune; return 1;  // 2-byte: no folding needed
    }

    // 3-byte UTF-8 (U+0800-FFFF): Georgian, Cherokee, Greek Extended, etc.
    if (rune <= 0xFFFF) {
        // Georgian Ⴀ-Ⴥ: 0x10A0-0x10C5 (+7264)
        if ((sz_u32_t)(rune - 0x10A0) <= 0x25) {
            sz_assert_(sz_is_in_range_(rune, 0x10A0, 0x10C5));
            folded[0] = rune + 0x1C60; return 1; }

        // Georgian Mtavruli Ა-Ჺ: 0x1C90-0x1CBA (-3008)
        if ((sz_u32_t)(rune - 0x1C90) <= 0x2A) {
            sz_assert_(sz_is_in_range_(rune, 0x1C90, 0x1CBA));
            folded[0] = rune - 0xBC0; return 1; }

        // Georgian Mtavruli Ჽ-Ჿ: 0x1CBD-0x1CBF (-3008)
        if ((sz_u32_t)(rune - 0x1CBD) <= 0x02) {
            sz_assert_(sz_is_in_range_(rune, 0x1CBD, 0x1CBF));
            folded[0] = rune - 0xBC0; return 1; }

        // Cherokee Ᏸ-Ᏽ: 0x13F8-0x13FD (-8)
        if ((sz_u32_t)(rune - 0x13F8) <= 0x05) {
            sz_assert_(sz_is_in_range_(rune, 0x13F8, 0x13FD));
            folded[0] = rune - 8; return 1; }

        // Cherokee Ꭰ-Ᏼ: 0xAB70-0xABBF → Ꭰ-Ᏼ: 0x13A0-0x13EF (-38864)
        if ((sz_u32_t)(rune - 0xAB70) <= 0x4F) {
            sz_assert_(sz_is_in_range_(rune, 0xAB70, 0xABBF));
            folded[0] = rune - 0x97D0; return 1; }

        // Greek Extended: multiple -8 offset ranges
        if ((sz_u32_t)(rune - 0x1F08) <= 0x07) { // Ἀ-Ἇ
            sz_assert_(sz_is_in_range_(rune, 0x1F08, 0x1F0F));
            folded[0] = rune - 8; return 1; }
        if ((sz_u32_t)(rune - 0x1F18) <= 0x05) { // Ἐ-Ἕ
            sz_assert_(sz_is_in_range_(rune, 0x1F18, 0x1F1D));
            folded[0] = rune - 8; return 1; }
        if ((sz_u32_t)(rune - 0x1F28) <= 0x07) { // Ἠ-Ἧ
            sz_assert_(sz_is_in_range_(rune, 0x1F28, 0x1F2F));
            folded[0] = rune - 8; return 1; }
        if ((sz_u32_t)(rune - 0x1F38) <= 0x07) { // Ἰ-Ἷ
            sz_assert_(sz_is_in_range_(rune, 0x1F38, 0x1F3F));
            folded[0] = rune - 8; return 1; }
        if ((sz_u32_t)(rune - 0x1F48) <= 0x05) { // Ὀ-Ὅ
            sz_assert_(sz_is_in_range_(rune, 0x1F48, 0x1F4D));
            folded[0] = rune - 8; return 1; }
        if ((sz_u32_t)(rune - 0x1F68) <= 0x07) { // Ὠ-Ὧ
            sz_assert_(sz_is_in_range_(rune, 0x1F68, 0x1F6F));
            folded[0] = rune - 8; return 1; }

        // Greek Extended Ὲ-Ή: 0x1FC8-0x1FCB (-86)
        if ((sz_u32_t)(rune - 0x1FC8) <= 0x03) {
            sz_assert_(sz_is_in_range_(rune, 0x1FC8, 0x1FCB));
            folded[0] = rune - 86; return 1; }

        // Roman numerals Ⅰ-Ⅿ: 0x2160-0x216F (+16)
        if ((sz_u32_t)(rune - 0x2160) <= 0x0F) {
            sz_assert_(sz_is_in_range_(rune, 0x2160, 0x216F));
            folded[0] = rune + 0x10; return 1; }

        // Circled letters Ⓐ-Ⓩ: 0x24B6-0x24CF (+26)
        if ((sz_u32_t)(rune - 0x24B6) <= 0x19) {
            sz_assert_(sz_is_in_range_(rune, 0x24B6, 0x24CF));
            folded[0] = rune + 0x1A; return 1; }

        // Glagolitic Ⰰ-Ⱟ: 0x2C00-0x2C2F (+48)
        if ((sz_u32_t)(rune - 0x2C00) <= 0x2F) {
            sz_assert_(sz_is_in_range_(rune, 0x2C00, 0x2C2F));
            folded[0] = rune + 0x30; return 1; }

        // Fullwidth Ａ-Ｚ: 0xFF21-0xFF3A (+32)
        if ((sz_u32_t)(rune - 0xFF21) <= 0x19) {
            sz_assert_(sz_is_in_range_(rune, 0xFF21, 0xFF3A));
            folded[0] = rune + 0x20; return 1; }

        // Next let's handle the even/odd parity-based ranges
        sz_u32_t is_even = ((rune & 1) == 0);

        // Latin Extended Additional Ḁ-Ẕ: 0x1E00-0x1E94
        if ((sz_u32_t)(rune - 0x1E00) <= 0x94 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x1E00, 0x1E94));
            folded[0] = rune + 1; return 1; }

        // Latin Extended Additional (Vietnamese) Ạ-Ỿ: 0x1EA0-0x1EFE
        if ((sz_u32_t)(rune - 0x1EA0) <= 0x5E && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x1EA0, 0x1EFE));
            folded[0] = rune + 1; return 1; }

        // Coptic Ⲁ-Ⳣ: 0x2C80-0x2CE2
        if ((sz_u32_t)(rune - 0x2C80) <= 0x62 && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0x2C80, 0x2CE2));
            folded[0] = rune + 1; return 1; }

        // Cyrillic Extended-B Ꙁ-Ꙭ: 0xA640-0xA66C
        if ((sz_u32_t)(rune - 0xA640) <= 0x2C && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0xA640, 0xA66C));
            folded[0] = rune + 1; return 1; }

        // Cyrillic Extended-B Ꚁ-Ꚛ: 0xA680-0xA69A
        if ((sz_u32_t)(rune - 0xA680) <= 0x1A && is_even) {
            sz_assert_(sz_is_in_range_(rune, 0xA680, 0xA69A));
            folded[0] = rune + 1; return 1; }

        // Latin Extended-D ranges
        if ((sz_u32_t)(rune - 0xA722) <= 0x0C && is_even) { // Ꜣ-Ꜯ
            sz_assert_(sz_is_in_range_(rune, 0xA722, 0xA72E));
            folded[0] = rune + 1; return 1; }
        if ((sz_u32_t)(rune - 0xA732) <= 0x3C && is_even) { // Ꜳ-Ꝯ
            sz_assert_(sz_is_in_range_(rune, 0xA732, 0xA76E));
            folded[0] = rune + 1; return 1; }
        if ((sz_u32_t)(rune - 0xA77E) <= 0x08 && is_even) { // Ꝿ-Ꞇ
            sz_assert_(sz_is_in_range_(rune, 0xA77E, 0xA786));
            folded[0] = rune + 1; return 1; }
        if ((sz_u32_t)(rune - 0xA790) <= 0x02 && is_even) { // Ꞑ-Ꞓ
            sz_assert_(sz_is_in_range_(rune, 0xA790, 0xA792));
            folded[0] = rune + 1; return 1; }
        if ((sz_u32_t)(rune - 0xA796) <= 0x12 && is_even) { // Ꞗ-Ꞩ
            sz_assert_(sz_is_in_range_(rune, 0xA796, 0xA7A8));
            folded[0] = rune + 1; return 1; }
        if ((sz_u32_t)(rune - 0xA7B4) <= 0x0E && is_even) { // Ꞵ-Ꟃ
            sz_assert_(sz_is_in_range_(rune, 0xA7B4, 0xA7C2));
            folded[0] = rune + 1; return 1; }

        // Next let's handle the 3-byte irregular one-to-one mappings
        switch (rune) {
        // Georgian irregular
        case 0x10C7: folded[0] = 0x2D27; return 1; // 'Ⴧ' (U+10C7, E1 83 87) → 'ⴧ' (U+2D27, E2 B4 A7)
        case 0x10CD: folded[0] = 0x2D2D; return 1; // 'Ⴭ' (U+10CD, E1 83 8D) → 'ⴭ' (U+2D2D, E2 B4 AD)
        // Cyrillic Extended-C: Old Slavonic variant forms
        case 0x1C80: folded[0] = 0x0432; return 1; // 'ᲀ' (U+1C80, E1 B2 80) → 'в' (U+0432, D0 B2)
        case 0x1C81: folded[0] = 0x0434; return 1; // 'ᲁ' (U+1C81, E1 B2 81) → 'д' (U+0434, D0 B4)
        case 0x1C82: folded[0] = 0x043E; return 1; // 'ᲂ' (U+1C82, E1 B2 82) → 'о' (U+043E, D0 BE)
        case 0x1C83: folded[0] = 0x0441; return 1; // 'ᲃ' (U+1C83, E1 B2 83) → 'с' (U+0441, D1 81)
        case 0x1C84: folded[0] = 0x0442; return 1; // 'ᲄ' (U+1C84, E1 B2 84) → 'т' (U+0442, D1 82)
        case 0x1C85: folded[0] = 0x0442; return 1; // 'ᲅ' (U+1C85, E1 B2 85) → 'т' (U+0442, D1 82)
        case 0x1C86: folded[0] = 0x044A; return 1; // 'ᲆ' (U+1C86, E1 B2 86) → 'ъ' (U+044A, D1 8A)
        case 0x1C87: folded[0] = 0x0463; return 1; // 'ᲇ' (U+1C87, E1 B2 87) → 'ѣ' (U+0463, D1 A3)
        case 0x1C88: folded[0] = 0xA64B; return 1; // 'ᲈ' (U+1C88, E1 B2 88) → 'ꙋ' (U+A64B, EA 99 8B)
        case 0x1C89: folded[0] = 0x1C8A; return 1; // 'Ᲊ' (U+1C89, E1 B2 89) → 'ᲊ' (U+1C8A, E1 B2 8A)
        // Latin Extended Additional: long s with dot
        case 0x1E9B: folded[0] = 0x1E61; return 1; // 'ẛ' (U+1E9B, E1 BA 9B) → 'ṡ' (U+1E61, E1 B9 A1)
        // Greek Extended: vowels with breathing marks (irregular offsets)
        case 0x1F59: folded[0] = 0x1F51; return 1; // 'Ὑ' (U+1F59, E1 BD 99) → 'ὑ' (U+1F51, E1 BD 91)
        case 0x1F5B: folded[0] = 0x1F53; return 1; // 'Ὓ' (U+1F5B, E1 BD 9B) → 'ὓ' (U+1F53, E1 BD 93)
        case 0x1F5D: folded[0] = 0x1F55; return 1; // 'Ὕ' (U+1F5D, E1 BD 9D) → 'ὕ' (U+1F55, E1 BD 95)
        case 0x1F5F: folded[0] = 0x1F57; return 1; // 'Ὗ' (U+1F5F, E1 BD 9F) → 'ὗ' (U+1F57, E1 BD 97)
        case 0x1FB8: folded[0] = 0x1FB0; return 1; // 'Ᾰ' (U+1FB8, E1 BE B8) → 'ᾰ' (U+1FB0, E1 BE B0)
        case 0x1FB9: folded[0] = 0x1FB1; return 1; // 'Ᾱ' (U+1FB9, E1 BE B9) → 'ᾱ' (U+1FB1, E1 BE B1)
        case 0x1FBA: folded[0] = 0x1F70; return 1; // 'Ὰ' (U+1FBA, E1 BE BA) → 'ὰ' (U+1F70, E1 BD B0)
        case 0x1FBB: folded[0] = 0x1F71; return 1; // 'Ά' (U+1FBB, E1 BE BB) → 'ά' (U+1F71, E1 BD B1)
        case 0x1FBE: folded[0] = 0x03B9; return 1; // 'ι' (U+1FBE, E1 BE BE) → 'ι' (U+03B9, CE B9)
        case 0x1FD8: folded[0] = 0x1FD0; return 1; // 'Ῐ' (U+1FD8, E1 BF 98) → 'ῐ' (U+1FD0, E1 BF 90)
        case 0x1FD9: folded[0] = 0x1FD1; return 1; // 'Ῑ' (U+1FD9, E1 BF 99) → 'ῑ' (U+1FD1, E1 BF 91)
        case 0x1FDA: folded[0] = 0x1F76; return 1; // 'Ὶ' (U+1FDA, E1 BF 9A) → 'ὶ' (U+1F76, E1 BD B6)
        case 0x1FDB: folded[0] = 0x1F77; return 1; // 'Ί' (U+1FDB, E1 BF 9B) → 'ί' (U+1F77, E1 BD B7)
        case 0x1FE8: folded[0] = 0x1FE0; return 1; // 'Ῠ' (U+1FE8, E1 BF A8) → 'ῠ' (U+1FE0, E1 BF A0)
        case 0x1FE9: folded[0] = 0x1FE1; return 1; // 'Ῡ' (U+1FE9, E1 BF A9) → 'ῡ' (U+1FE1, E1 BF A1)
        case 0x1FEA: folded[0] = 0x1F7A; return 1; // 'Ὺ' (U+1FEA, E1 BF AA) → 'ὺ' (U+1F7A, E1 BD BA)
        case 0x1FEB: folded[0] = 0x1F7B; return 1; // 'Ύ' (U+1FEB, E1 BF AB) → 'ύ' (U+1F7B, E1 BD BB)
        case 0x1FEC: folded[0] = 0x1FE5; return 1; // 'Ῥ' (U+1FEC, E1 BF AC) → 'ῥ' (U+1FE5, E1 BF A5)
        case 0x1FF8: folded[0] = 0x1F78; return 1; // 'Ὸ' (U+1FF8, E1 BF B8) → 'ὸ' (U+1F78, E1 BD B8)
        case 0x1FF9: folded[0] = 0x1F79; return 1; // 'Ό' (U+1FF9, E1 BF B9) → 'ό' (U+1F79, E1 BD B9)
        case 0x1FFA: folded[0] = 0x1F7C; return 1; // 'Ὼ' (U+1FFA, E1 BF BA) → 'ὼ' (U+1F7C, E1 BD BC)
        case 0x1FFB: folded[0] = 0x1F7D; return 1; // 'Ώ' (U+1FFB, E1 BF BB) → 'ώ' (U+1F7D, E1 BD BD)
        // Letterlike Symbols: compatibility mappings
        case 0x2126: folded[0] = 0x03C9; return 1; // 'Ω' (U+2126, E2 84 A6) → 'ω' (U+03C9, CF 89)
        case 0x212A: folded[0] = 0x006B; return 1; // 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B)
        case 0x212B: folded[0] = 0x00E5; return 1; // 'Å' (U+212B, E2 84 AB) → 'å' (U+00E5, C3 A5)
        case 0x2132: folded[0] = 0x214E; return 1; // 'Ⅎ' (U+2132, E2 84 B2) → 'ⅎ' (U+214E, E2 85 8E)
        case 0x2183: folded[0] = 0x2184; return 1; // 'Ↄ' (U+2183, E2 86 83) → 'ↄ' (U+2184, E2 86 84)
        // Latin Extended-C: irregular mappings to IPA/other blocks
        case 0x2C60: folded[0] = 0x2C61; return 1; // 'Ⱡ' (U+2C60, E2 B1 A0) → 'ⱡ' (U+2C61, E2 B1 A1)
        case 0x2C62: folded[0] = 0x026B; return 1; // 'Ɫ' (U+2C62, E2 B1 A2) → 'ɫ' (U+026B, C9 AB)
        case 0x2C63: folded[0] = 0x1D7D; return 1; // 'Ᵽ' (U+2C63, E2 B1 A3) → 'ᵽ' (U+1D7D, E1 B5 BD)
        case 0x2C64: folded[0] = 0x027D; return 1; // 'Ɽ' (U+2C64, E2 B1 A4) → 'ɽ' (U+027D, C9 BD)
        case 0x2C67: folded[0] = 0x2C68; return 1; // 'Ⱨ' (U+2C67, E2 B1 A7) → 'ⱨ' (U+2C68, E2 B1 A8)
        case 0x2C69: folded[0] = 0x2C6A; return 1; // 'Ⱪ' (U+2C69, E2 B1 A9) → 'ⱪ' (U+2C6A, E2 B1 AA)
        case 0x2C6B: folded[0] = 0x2C6C; return 1; // 'Ⱬ' (U+2C6B, E2 B1 AB) → 'ⱬ' (U+2C6C, E2 B1 AC)
        case 0x2C6D: folded[0] = 0x0251; return 1; // 'Ɑ' (U+2C6D, E2 B1 AD) → 'ɑ' (U+0251, C9 91)
        case 0x2C6E: folded[0] = 0x0271; return 1; // 'Ɱ' (U+2C6E, E2 B1 AE) → 'ɱ' (U+0271, C9 B1)
        case 0x2C6F: folded[0] = 0x0250; return 1; // 'Ɐ' (U+2C6F, E2 B1 AF) → 'ɐ' (U+0250, C9 90)
        case 0x2C70: folded[0] = 0x0252; return 1; // 'Ɒ' (U+2C70, E2 B1 B0) → 'ɒ' (U+0252, C9 92)
        case 0x2C72: folded[0] = 0x2C73; return 1; // 'Ⱳ' (U+2C72, E2 B1 B2) → 'ⱳ' (U+2C73, E2 B1 B3)
        case 0x2C75: folded[0] = 0x2C76; return 1; // 'Ⱶ' (U+2C75, E2 B1 B5) → 'ⱶ' (U+2C76, E2 B1 B6)
        case 0x2C7E: folded[0] = 0x023F; return 1; // 'Ȿ' (U+2C7E, E2 B1 BE) → 'ȿ' (U+023F, C8 BF)
        case 0x2C7F: folded[0] = 0x0240; return 1; // 'Ɀ' (U+2C7F, E2 B1 BF) → 'ɀ' (U+0240, C9 80)
        // Coptic: irregular cases outside even/odd range
        case 0x2CEB: folded[0] = 0x2CEC; return 1; // 'Ⳬ' (U+2CEB, E2 B3 AB) → 'ⳬ' (U+2CEC, E2 B3 AC)
        case 0x2CED: folded[0] = 0x2CEE; return 1; // 'Ⳮ' (U+2CED, E2 B3 AD) → 'ⳮ' (U+2CEE, E2 B3 AE)
        case 0x2CF2: folded[0] = 0x2CF3; return 1; // 'Ⳳ' (U+2CF2, E2 B3 B2) → 'ⳳ' (U+2CF3, E2 B3 B3)
        // Latin Extended-D: isolated irregulars
        case 0xA779: folded[0] = 0xA77A; return 1; // 'Ꝺ' (U+A779, EA 9D B9) → 'ꝺ' (U+A77A, EA 9D BA)
        case 0xA77B: folded[0] = 0xA77C; return 1; // 'Ꝼ' (U+A77B, EA 9D BB) → 'ꝼ' (U+A77C, EA 9D BC)
        case 0xA77D: folded[0] = 0x1D79; return 1; // 'Ᵹ' (U+A77D, EA 9D BD) → 'ᵹ' (U+1D79, E1 B5 B9)
        case 0xA78B: folded[0] = 0xA78C; return 1; // 'Ꞌ' (U+A78B, EA 9E 8B) → 'ꞌ' (U+A78C, EA 9E 8C)
        case 0xA78D: folded[0] = 0x0265; return 1; // 'Ɥ' (U+A78D, EA 9E 8D) → 'ɥ' (U+0265, C9 A5)
        case 0xA7AA: folded[0] = 0x0266; return 1; // 'Ɦ' (U+A7AA, EA 9E AA) → 'ɦ' (U+0266, C9 A6)
        case 0xA7AB: folded[0] = 0x025C; return 1; // 'Ɜ' (U+A7AB, EA 9E AB) → 'ɜ' (U+025C, C9 9C)
        case 0xA7AC: folded[0] = 0x0261; return 1; // 'Ɡ' (U+A7AC, EA 9E AC) → 'ɡ' (U+0261, C9 A1)
        case 0xA7AD: folded[0] = 0x026C; return 1; // 'Ɬ' (U+A7AD, EA 9E AD) → 'ɬ' (U+026C, C9 AC)
        case 0xA7AE: folded[0] = 0x026A; return 1; // 'Ɪ' (U+A7AE, EA 9E AE) → 'ɪ' (U+026A, C9 AA)
        case 0xA7B0: folded[0] = 0x029E; return 1; // 'Ʞ' (U+A7B0, EA 9E B0) → 'ʞ' (U+029E, CA 9E)
        case 0xA7B1: folded[0] = 0x0287; return 1; // 'Ʇ' (U+A7B1, EA 9E B1) → 'ʇ' (U+0287, CA 87)
        case 0xA7B2: folded[0] = 0x029D; return 1; // 'Ʝ' (U+A7B2, EA 9E B2) → 'ʝ' (U+029D, CA 9D)
        case 0xA7B3: folded[0] = 0xAB53; return 1; // 'Ꭓ' (U+A7B3, EA 9E B3) → 'ꭓ' (U+AB53, EA AD 93)
        case 0xA7C4: folded[0] = 0xA794; return 1; // 'Ꞔ' (U+A7C4, EA 9F 84) → 'ꞔ' (U+A794, EA 9E 94)
        case 0xA7C5: folded[0] = 0x0282; return 1; // 'Ʂ' (U+A7C5, EA 9F 85) → 'ʂ' (U+0282, CA 82)
        case 0xA7C6: folded[0] = 0x1D8E; return 1; // 'Ᶎ' (U+A7C6, EA 9F 86) → 'ᶎ' (U+1D8E, E1 B6 8E)
        case 0xA7C7: folded[0] = 0xA7C8; return 1; // 'Ꟈ' (U+A7C7, EA 9F 87) → 'ꟈ' (U+A7C8, EA 9F 88)
        case 0xA7C9: folded[0] = 0xA7CA; return 1; // 'Ꟊ' (U+A7C9, EA 9F 89) → 'ꟊ' (U+A7CA, EA 9F 8A)
        case 0xA7CB: folded[0] = 0x0264; return 1; // 'Ɤ' (U+A7CB, EA 9F 8B) → 'ɤ' (U+0264, C9 A4)
        case 0xA7CC: folded[0] = 0xA7CD; return 1; // 'Ꟍ' (U+A7CC, EA 9F 8C) → 'ꟍ' (U+A7CD, EA 9F 8D)
        case 0xA7CE: folded[0] = 0xA7CF; return 1; // '꟎' (U+A7CE, EA 9F 8E) → '꟏' (U+A7CF, EA 9F 8F)
        case 0xA7D0: folded[0] = 0xA7D1; return 1; // 'Ꟑ' (U+A7D0, EA 9F 90) → 'ꟑ' (U+A7D1, EA 9F 91)
        case 0xA7D2: folded[0] = 0xA7D3; return 1; // '꟒' (U+A7D2, EA 9F 92) → 'ꟓ' (U+A7D3, EA 9F 93)
        case 0xA7D4: folded[0] = 0xA7D5; return 1; // '꟔' (U+A7D4, EA 9F 94) → 'ꟕ' (U+A7D5, EA 9F 95)
        case 0xA7D6: folded[0] = 0xA7D7; return 1; // 'Ꟗ' (U+A7D6, EA 9F 96) → 'ꟗ' (U+A7D7, EA 9F 97)
        case 0xA7D8: folded[0] = 0xA7D9; return 1; // 'Ꟙ' (U+A7D8, EA 9F 98) → 'ꟙ' (U+A7D9, EA 9F 99)
        case 0xA7DA: folded[0] = 0xA7DB; return 1; // 'Ꟛ' (U+A7DA, EA 9F 9A) → 'ꟛ' (U+A7DB, EA 9F 9B)
        case 0xA7DC: folded[0] = 0x019B; return 1; // 'Ƛ' (U+A7DC, EA 9F 9C) → 'ƛ' (U+019B, C6 9B)
        case 0xA7F5: folded[0] = 0xA7F6; return 1; // 'Ꟶ' (U+A7F5, EA 9F B5) → 'ꟶ' (U+A7F6, EA 9F B6)
        }

        // Next let's handle the 3-byte one-to-many expansions
        switch (rune) {
        // Latin Extended Additional
        // 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
        case 0x1E96: folded[0] = 0x0068; folded[1] = 0x0331; return 2;
        // 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
        case 0x1E97: folded[0] = 0x0074; folded[1] = 0x0308; return 2;
        // 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
        case 0x1E98: folded[0] = 0x0077; folded[1] = 0x030A; return 2;
        // 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
        case 0x1E99: folded[0] = 0x0079; folded[1] = 0x030A; return 2;
        // 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
        case 0x1E9A: folded[0] = 0x0061; folded[1] = 0x02BE; return 2;
        // 'ẞ' (U+1E9E, E1 BA 9E) → "ss" (U+0073 U+0073, 73 73)
        case 0x1E9E: folded[0] = 0x0073; folded[1] = 0x0073; return 2;
        // Greek Extended: breathing marks
        // 'ὐ' (U+1F50, E1 BD 90) → "ὐ" (U+03C5 U+0313, CF 85 CC 93)
        case 0x1F50: folded[0] = 0x03C5; folded[1] = 0x0313; return 2;
        // 'ὒ' (U+1F52, E1 BD 92) → "ὒ" (U+03C5 U+0313 U+0300, CF 85 CC 93 CC 80)
        case 0x1F52: folded[0] = 0x03C5; folded[1] = 0x0313; folded[2] = 0x0300; return 3;
        // 'ὔ' (U+1F54, E1 BD 94) → "ὔ" (U+03C5 U+0313 U+0301, CF 85 CC 93 CC 81)
        case 0x1F54: folded[0] = 0x03C5; folded[1] = 0x0313; folded[2] = 0x0301; return 3;
        // 'ὖ' (U+1F56, E1 BD 96) → "ὖ" (U+03C5 U+0313 U+0342, CF 85 CC 93 CD 82)
        case 0x1F56: folded[0] = 0x03C5; folded[1] = 0x0313; folded[2] = 0x0342; return 3;
        // Greek Extended: iota subscript combinations (0x1F80-0x1FAF)
        // 'ᾀ' (U+1F80, E1 BE 80) → "ἀι" (U+1F00 U+03B9, E1 BC 80 CE B9)
        case 0x1F80: folded[0] = 0x1F00; folded[1] = 0x03B9; return 2;
        // 'ᾁ' (U+1F81, E1 BE 81) → "ἁι" (U+1F01 U+03B9, E1 BC 81 CE B9)
        case 0x1F81: folded[0] = 0x1F01; folded[1] = 0x03B9; return 2;
        // 'ᾂ' (U+1F82, E1 BE 82) → "ἂι" (U+1F02 U+03B9, E1 BC 82 CE B9)
        case 0x1F82: folded[0] = 0x1F02; folded[1] = 0x03B9; return 2;
        // 'ᾃ' (U+1F83, E1 BE 83) → "ἃι" (U+1F03 U+03B9, E1 BC 83 CE B9)
        case 0x1F83: folded[0] = 0x1F03; folded[1] = 0x03B9; return 2;
        // 'ᾄ' (U+1F84, E1 BE 84) → "ἄι" (U+1F04 U+03B9, E1 BC 84 CE B9)
        case 0x1F84: folded[0] = 0x1F04; folded[1] = 0x03B9; return 2;
        // 'ᾅ' (U+1F85, E1 BE 85) → "ἅι" (U+1F05 U+03B9, E1 BC 85 CE B9)
        case 0x1F85: folded[0] = 0x1F05; folded[1] = 0x03B9; return 2;
        // 'ᾆ' (U+1F86, E1 BE 86) → "ἆι" (U+1F06 U+03B9, E1 BC 86 CE B9)
        case 0x1F86: folded[0] = 0x1F06; folded[1] = 0x03B9; return 2;
        // 'ᾇ' (U+1F87, E1 BE 87) → "ἇι" (U+1F07 U+03B9, E1 BC 87 CE B9)
        case 0x1F87: folded[0] = 0x1F07; folded[1] = 0x03B9; return 2;
        // 'ᾈ' (U+1F88, E1 BE 88) → "ἀι" (U+1F00 U+03B9, E1 BC 80 CE B9)
        case 0x1F88: folded[0] = 0x1F00; folded[1] = 0x03B9; return 2;
        // 'ᾉ' (U+1F89, E1 BE 89) → "ἁι" (U+1F01 U+03B9, E1 BC 81 CE B9)
        case 0x1F89: folded[0] = 0x1F01; folded[1] = 0x03B9; return 2;
        // 'ᾊ' (U+1F8A, E1 BE 8A) → "ἂι" (U+1F02 U+03B9, E1 BC 82 CE B9)
        case 0x1F8A: folded[0] = 0x1F02; folded[1] = 0x03B9; return 2;
        // 'ᾋ' (U+1F8B, E1 BE 8B) → "ἃι" (U+1F03 U+03B9, E1 BC 83 CE B9)
        case 0x1F8B: folded[0] = 0x1F03; folded[1] = 0x03B9; return 2;
        // 'ᾌ' (U+1F8C, E1 BE 8C) → "ἄι" (U+1F04 U+03B9, E1 BC 84 CE B9)
        case 0x1F8C: folded[0] = 0x1F04; folded[1] = 0x03B9; return 2;
        // 'ᾍ' (U+1F8D, E1 BE 8D) → "ἅι" (U+1F05 U+03B9, E1 BC 85 CE B9)
        case 0x1F8D: folded[0] = 0x1F05; folded[1] = 0x03B9; return 2;
        // 'ᾎ' (U+1F8E, E1 BE 8E) → "ἆι" (U+1F06 U+03B9, E1 BC 86 CE B9)
        case 0x1F8E: folded[0] = 0x1F06; folded[1] = 0x03B9; return 2;
        // 'ᾏ' (U+1F8F, E1 BE 8F) → "ἇι" (U+1F07 U+03B9, E1 BC 87 CE B9)
        case 0x1F8F: folded[0] = 0x1F07; folded[1] = 0x03B9; return 2;
        // 'ᾐ' (U+1F90, E1 BE 90) → "ἠι" (U+1F20 U+03B9, E1 BC A0 CE B9)
        case 0x1F90: folded[0] = 0x1F20; folded[1] = 0x03B9; return 2;
        // 'ᾑ' (U+1F91, E1 BE 91) → "ἡι" (U+1F21 U+03B9, E1 BC A1 CE B9)
        case 0x1F91: folded[0] = 0x1F21; folded[1] = 0x03B9; return 2;
        // 'ᾒ' (U+1F92, E1 BE 92) → "ἢι" (U+1F22 U+03B9, E1 BC A2 CE B9)
        case 0x1F92: folded[0] = 0x1F22; folded[1] = 0x03B9; return 2;
        // 'ᾓ' (U+1F93, E1 BE 93) → "ἣι" (U+1F23 U+03B9, E1 BC A3 CE B9)
        case 0x1F93: folded[0] = 0x1F23; folded[1] = 0x03B9; return 2;
        // 'ᾔ' (U+1F94, E1 BE 94) → "ἤι" (U+1F24 U+03B9, E1 BC A4 CE B9)
        case 0x1F94: folded[0] = 0x1F24; folded[1] = 0x03B9; return 2;
        // 'ᾕ' (U+1F95, E1 BE 95) → "ἥι" (U+1F25 U+03B9, E1 BC A5 CE B9)
        case 0x1F95: folded[0] = 0x1F25; folded[1] = 0x03B9; return 2;
        // 'ᾖ' (U+1F96, E1 BE 96) → "ἦι" (U+1F26 U+03B9, E1 BC A6 CE B9)
        case 0x1F96: folded[0] = 0x1F26; folded[1] = 0x03B9; return 2;
        // 'ᾗ' (U+1F97, E1 BE 97) → "ἧι" (U+1F27 U+03B9, E1 BC A7 CE B9)
        case 0x1F97: folded[0] = 0x1F27; folded[1] = 0x03B9; return 2;
        // 'ᾘ' (U+1F98, E1 BE 98) → "ἠι" (U+1F20 U+03B9, E1 BC A0 CE B9)
        case 0x1F98: folded[0] = 0x1F20; folded[1] = 0x03B9; return 2;
        // 'ᾙ' (U+1F99, E1 BE 99) → "ἡι" (U+1F21 U+03B9, E1 BC A1 CE B9)
        case 0x1F99: folded[0] = 0x1F21; folded[1] = 0x03B9; return 2;
        // 'ᾚ' (U+1F9A, E1 BE 9A) → "ἢι" (U+1F22 U+03B9, E1 BC A2 CE B9)
        case 0x1F9A: folded[0] = 0x1F22; folded[1] = 0x03B9; return 2;
        // 'ᾛ' (U+1F9B, E1 BE 9B) → "ἣι" (U+1F23 U+03B9, E1 BC A3 CE B9)
        case 0x1F9B: folded[0] = 0x1F23; folded[1] = 0x03B9; return 2;
        // 'ᾜ' (U+1F9C, E1 BE 9C) → "ἤι" (U+1F24 U+03B9, E1 BC A4 CE B9)
        case 0x1F9C: folded[0] = 0x1F24; folded[1] = 0x03B9; return 2;
        // 'ᾝ' (U+1F9D, E1 BE 9D) → "ἥι" (U+1F25 U+03B9, E1 BC A5 CE B9)
        case 0x1F9D: folded[0] = 0x1F25; folded[1] = 0x03B9; return 2;
        // 'ᾞ' (U+1F9E, E1 BE 9E) → "ἦι" (U+1F26 U+03B9, E1 BC A6 CE B9)
        case 0x1F9E: folded[0] = 0x1F26; folded[1] = 0x03B9; return 2;
        // 'ᾟ' (U+1F9F, E1 BE 9F) → "ἧι" (U+1F27 U+03B9, E1 BC A7 CE B9)
        case 0x1F9F: folded[0] = 0x1F27; folded[1] = 0x03B9; return 2;
        // 'ᾠ' (U+1FA0, E1 BE A0) → "ὠι" (U+1F60 U+03B9, E1 BD A0 CE B9)
        case 0x1FA0: folded[0] = 0x1F60; folded[1] = 0x03B9; return 2;
        // 'ᾡ' (U+1FA1, E1 BE A1) → "ὡι" (U+1F61 U+03B9, E1 BD A1 CE B9)
        case 0x1FA1: folded[0] = 0x1F61; folded[1] = 0x03B9; return 2;
        // 'ᾢ' (U+1FA2, E1 BE A2) → "ὢι" (U+1F62 U+03B9, E1 BD A2 CE B9)
        case 0x1FA2: folded[0] = 0x1F62; folded[1] = 0x03B9; return 2;
        // 'ᾣ' (U+1FA3, E1 BE A3) → "ὣι" (U+1F63 U+03B9, E1 BD A3 CE B9)
        case 0x1FA3: folded[0] = 0x1F63; folded[1] = 0x03B9; return 2;
        // 'ᾤ' (U+1FA4, E1 BE A4) → "ὤι" (U+1F64 U+03B9, E1 BD A4 CE B9)
        case 0x1FA4: folded[0] = 0x1F64; folded[1] = 0x03B9; return 2;
        // 'ᾥ' (U+1FA5, E1 BE A5) → "ὥι" (U+1F65 U+03B9, E1 BD A5 CE B9)
        case 0x1FA5: folded[0] = 0x1F65; folded[1] = 0x03B9; return 2;
        // 'ᾦ' (U+1FA6, E1 BE A6) → "ὦι" (U+1F66 U+03B9, E1 BD A6 CE B9)
        case 0x1FA6: folded[0] = 0x1F66; folded[1] = 0x03B9; return 2;
        // 'ᾧ' (U+1FA7, E1 BE A7) → "ὧι" (U+1F67 U+03B9, E1 BD A7 CE B9)
        case 0x1FA7: folded[0] = 0x1F67; folded[1] = 0x03B9; return 2;
        // 'ᾨ' (U+1FA8, E1 BE A8) → "ὠι" (U+1F60 U+03B9, E1 BD A0 CE B9)
        case 0x1FA8: folded[0] = 0x1F60; folded[1] = 0x03B9; return 2;
        // 'ᾩ' (U+1FA9, E1 BE A9) → "ὡι" (U+1F61 U+03B9, E1 BD A1 CE B9)
        case 0x1FA9: folded[0] = 0x1F61; folded[1] = 0x03B9; return 2;
        // 'ᾪ' (U+1FAA, E1 BE AA) → "ὢι" (U+1F62 U+03B9, E1 BD A2 CE B9)
        case 0x1FAA: folded[0] = 0x1F62; folded[1] = 0x03B9; return 2;
        // 'ᾫ' (U+1FAB, E1 BE AB) → "ὣι" (U+1F63 U+03B9, E1 BD A3 CE B9)
        case 0x1FAB: folded[0] = 0x1F63; folded[1] = 0x03B9; return 2;
        // 'ᾬ' (U+1FAC, E1 BE AC) → "ὤι" (U+1F64 U+03B9, E1 BD A4 CE B9)
        case 0x1FAC: folded[0] = 0x1F64; folded[1] = 0x03B9; return 2;
        // 'ᾭ' (U+1FAD, E1 BE AD) → "ὥι" (U+1F65 U+03B9, E1 BD A5 CE B9)
        case 0x1FAD: folded[0] = 0x1F65; folded[1] = 0x03B9; return 2;
        // 'ᾮ' (U+1FAE, E1 BE AE) → "ὦι" (U+1F66 U+03B9, E1 BD A6 CE B9)
        case 0x1FAE: folded[0] = 0x1F66; folded[1] = 0x03B9; return 2;
        // 'ᾯ' (U+1FAF, E1 BE AF) → "ὧι" (U+1F67 U+03B9, E1 BD A7 CE B9)
        case 0x1FAF: folded[0] = 0x1F67; folded[1] = 0x03B9; return 2;
        // Greek Extended: vowel + iota subscript (0x1FB2-0x1FFC)
        // 'ᾲ' (U+1FB2, E1 BE B2) → "ὰι" (U+1F70 U+03B9, E1 BD B0 CE B9)
        case 0x1FB2: folded[0] = 0x1F70; folded[1] = 0x03B9; return 2;
        // 'ᾳ' (U+1FB3, E1 BE B3) → "αι" (U+03B1 U+03B9, CE B1 CE B9)
        case 0x1FB3: folded[0] = 0x03B1; folded[1] = 0x03B9; return 2;
        // 'ᾴ' (U+1FB4, E1 BE B4) → "άι" (U+03AC U+03B9, CE AC CE B9)
        case 0x1FB4: folded[0] = 0x03AC; folded[1] = 0x03B9; return 2;
        // 'ᾶ' (U+1FB6, E1 BE B6) → "ᾶ" (U+03B1 U+0342, CE B1 CD 82)
        case 0x1FB6: folded[0] = 0x03B1; folded[1] = 0x0342; return 2;
        // 'ᾷ' (U+1FB7, E1 BE B7) → "ᾶι" (U+03B1 U+0342 U+03B9, CE B1 CD 82 CE B9)
        case 0x1FB7: folded[0] = 0x03B1; folded[1] = 0x0342; folded[2] = 0x03B9; return 3;
        // 'ᾼ' (U+1FBC, E1 BE BC) → "αι" (U+03B1 U+03B9, CE B1 CE B9)
        case 0x1FBC: folded[0] = 0x03B1; folded[1] = 0x03B9; return 2;
        // 'ῂ' (U+1FC2, E1 BF 82) → "ὴι" (U+1F74 U+03B9, E1 BD B4 CE B9)
        case 0x1FC2: folded[0] = 0x1F74; folded[1] = 0x03B9; return 2;
        // 'ῃ' (U+1FC3, E1 BF 83) → "ηι" (U+03B7 U+03B9, CE B7 CE B9)
        case 0x1FC3: folded[0] = 0x03B7; folded[1] = 0x03B9; return 2;
        // 'ῄ' (U+1FC4, E1 BF 84) → "ήι" (U+03AE U+03B9, CE AE CE B9)
        case 0x1FC4: folded[0] = 0x03AE; folded[1] = 0x03B9; return 2;
        // 'ῆ' (U+1FC6, E1 BF 86) → "ῆ" (U+03B7 U+0342, CE B7 CD 82)
        case 0x1FC6: folded[0] = 0x03B7; folded[1] = 0x0342; return 2;
        // 'ῇ' (U+1FC7, E1 BF 87) → "ῆι" (U+03B7 U+0342 U+03B9, CE B7 CD 82 CE B9)
        case 0x1FC7: folded[0] = 0x03B7; folded[1] = 0x0342; folded[2] = 0x03B9; return 3;
        // 'ῌ' (U+1FCC, E1 BF 8C) → "ηι" (U+03B7 U+03B9, CE B7 CE B9)
        case 0x1FCC: folded[0] = 0x03B7; folded[1] = 0x03B9; return 2;
        // 'ῒ' (U+1FD2, E1 BF 92) → "ῒ" (U+03B9 U+0308 U+0300, CE B9 CC 88 CC 80)
        case 0x1FD2: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0300; return 3;
        // 'ΐ' (U+1FD3, E1 BF 93) → "ΐ" (U+03B9 U+0308 U+0301, CE B9 CC 88 CC 81)
        case 0x1FD3: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0301; return 3;
        // 'ῖ' (U+1FD6, E1 BF 96) → "ῖ" (U+03B9 U+0342, CE B9 CD 82)
        case 0x1FD6: folded[0] = 0x03B9; folded[1] = 0x0342; return 2;
        // 'ῗ' (U+1FD7, E1 BF 97) → "ῗ" (U+03B9 U+0308 U+0342, CE B9 CC 88 CD 82)
        case 0x1FD7: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0342; return 3;
        // 'ῢ' (U+1FE2, E1 BF A2) → "ῢ" (U+03C5 U+0308 U+0300, CF 85 CC 88 CC 80)
        case 0x1FE2: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0300; return 3;
        // 'ΰ' (U+1FE3, E1 BF A3) → "ΰ" (U+03C5 U+0308 U+0301, CF 85 CC 88 CC 81)
        case 0x1FE3: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0301; return 3;
        // 'ῤ' (U+1FE4, E1 BF A4) → "ῤ" (U+03C1 U+0313, CF 81 CC 93)
        case 0x1FE4: folded[0] = 0x03C1; folded[1] = 0x0313; return 2;
        // 'ῦ' (U+1FE6, E1 BF A6) → "ῦ" (U+03C5 U+0342, CF 85 CD 82)
        case 0x1FE6: folded[0] = 0x03C5; folded[1] = 0x0342; return 2;
        // 'ῧ' (U+1FE7, E1 BF A7) → "ῧ" (U+03C5 U+0308 U+0342, CF 85 CC 88 CD 82)
        case 0x1FE7: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0342; return 3;
        // 'ῲ' (U+1FF2, E1 BF B2) → "ὼι" (U+1F7C U+03B9, E1 BD BC CE B9)
        case 0x1FF2: folded[0] = 0x1F7C; folded[1] = 0x03B9; return 2;
        // 'ῳ' (U+1FF3, E1 BF B3) → "ωι" (U+03C9 U+03B9, CF 89 CE B9)
        case 0x1FF3: folded[0] = 0x03C9; folded[1] = 0x03B9; return 2;
        // 'ῴ' (U+1FF4, E1 BF B4) → "ώι" (U+03CE U+03B9, CF 8E CE B9)
        case 0x1FF4: folded[0] = 0x03CE; folded[1] = 0x03B9; return 2;
        // 'ῶ' (U+1FF6, E1 BF B6) → "ῶ" (U+03C9 U+0342, CF 89 CD 82)
        case 0x1FF6: folded[0] = 0x03C9; folded[1] = 0x0342; return 2;
        // 'ῷ' (U+1FF7, E1 BF B7) → "ῶι" (U+03C9 U+0342 U+03B9, CF 89 CD 82 CE B9)
        case 0x1FF7: folded[0] = 0x03C9; folded[1] = 0x0342; folded[2] = 0x03B9; return 3;
        // 'ῼ' (U+1FFC, E1 BF BC) → "ωι" (U+03C9 U+03B9, CF 89 CE B9)
        case 0x1FFC: folded[0] = 0x03C9; folded[1] = 0x03B9; return 2;
        // Alphabetic Presentation Forms: ligatures
        // 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
        case 0xFB00: folded[0] = 0x0066; folded[1] = 0x0066; return 2;
        // 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
        case 0xFB01: folded[0] = 0x0066; folded[1] = 0x0069; return 2;
        // 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
        case 0xFB02: folded[0] = 0x0066; folded[1] = 0x006C; return 2;
        // 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
        case 0xFB03: folded[0] = 0x0066; folded[1] = 0x0066; folded[2] = 0x0069; return 3;
        // 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
        case 0xFB04: folded[0] = 0x0066; folded[1] = 0x0066; folded[2] = 0x006C; return 3;
        // 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
        case 0xFB05: folded[0] = 0x0073; folded[1] = 0x0074; return 2;
        // 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
        case 0xFB06: folded[0] = 0x0073; folded[1] = 0x0074; return 2;
        // Armenian ligatures
        // 'ﬓ' (U+FB13, EF AC 93) → "մն" (U+0574 U+0576, D5 B4 D5 B6)
        case 0xFB13: folded[0] = 0x0574; folded[1] = 0x0576; return 2;
        // 'ﬔ' (U+FB14, EF AC 94) → "մե" (U+0574 U+0565, D5 B4 D5 A5)
        case 0xFB14: folded[0] = 0x0574; folded[1] = 0x0565; return 2;
        // 'ﬕ' (U+FB15, EF AC 95) → "մի" (U+0574 U+056B, D5 B4 D5 AB)
        case 0xFB15: folded[0] = 0x0574; folded[1] = 0x056B; return 2;
        // 'ﬖ' (U+FB16, EF AC 96) → "վն" (U+057E U+0576, D5 BE D5 B6)
        case 0xFB16: folded[0] = 0x057E; folded[1] = 0x0576; return 2;
        // 'ﬗ' (U+FB17, EF AC 97) → "մխ" (U+0574 U+056D, D5 B4 D5 AD)
        case 0xFB17: folded[0] = 0x0574; folded[1] = 0x056D; return 2;
        }

        folded[0] = rune; return 1;  // 3-byte: no folding needed
    }

    // 4-byte UTF-8 (U+10000-10FFFF): Deseret, Osage, Vithkuqi, etc.

    // Deseret 𐐀-𐐧: 0x10400-0x10427 (+40)
    if ((sz_u32_t)(rune - 0x10400) <= 0x27) {
        sz_assert_(sz_is_in_range_(rune, 0x10400, 0x10427));
        folded[0] = rune + 0x28; return 1; }

    // Osage 𐒰-𐓓: 0x104B0-0x104D3 (+40)
    if ((sz_u32_t)(rune - 0x104B0) <= 0x23) {
        sz_assert_(sz_is_in_range_(rune, 0x104B0, 0x104D3));
        folded[0] = rune + 0x28; return 1; }

    // Vithkuqi: 3 ranges with gaps, all +39
    if ((sz_u32_t)(rune - 0x10570) <= 0x0A) { // 0x10570-0x1057A
        sz_assert_(sz_is_in_range_(rune, 0x10570, 0x1057A));
        folded[0] = rune + 0x27; return 1; }
    if ((sz_u32_t)(rune - 0x1057C) <= 0x0E) { // 0x1057C-0x1058A
        sz_assert_(sz_is_in_range_(rune, 0x1057C, 0x1058A));
        folded[0] = rune + 0x27; return 1; }
    if ((sz_u32_t)(rune - 0x1058C) <= 0x06) { // 0x1058C-0x10592
        sz_assert_(sz_is_in_range_(rune, 0x1058C, 0x10592));
        folded[0] = rune + 0x27; return 1; }

    // Old Hungarian: 0x10C80-0x10CB2 (+64)
    if ((sz_u32_t)(rune - 0x10C80) <= 0x32) {
        sz_assert_(sz_is_in_range_(rune, 0x10C80, 0x10CB2));
        folded[0] = rune + 0x40; return 1; }

    // Garay: 0x10D50-0x10D65 (+32)
    if ((sz_u32_t)(rune - 0x10D50) <= 0x15) {
        sz_assert_(sz_is_in_range_(rune, 0x10D50, 0x10D65));
        folded[0] = rune + 0x20; return 1; }

    // Warang Citi: 0x118A0-0x118BF (+32)
    if ((sz_u32_t)(rune - 0x118A0) <= 0x1F) {
        sz_assert_(sz_is_in_range_(rune, 0x118A0, 0x118BF));
        folded[0] = rune + 0x20; return 1; }

    // Medefaidrin: 0x16E40-0x16E5F (+32)
    if ((sz_u32_t)(rune - 0x16E40) <= 0x1F) {
        sz_assert_(sz_is_in_range_(rune, 0x16E40, 0x16E5F));
        folded[0] = rune + 0x20; return 1; }

    // Beria Erfe: 0x16EA0-0x16EB8 (+27)
    if ((sz_u32_t)(rune - 0x16EA0) <= 0x18) {
        sz_assert_(sz_is_in_range_(rune, 0x16EA0, 0x16EB8));
        folded[0] = rune + 0x1B; return 1; }

    // Adlam: 0x1E900-0x1E921 (+34)
    if ((sz_u32_t)(rune - 0x1E900) <= 0x21) {
        sz_assert_(sz_is_in_range_(rune, 0x1E900, 0x1E921));
        folded[0] = rune + 0x22; return 1; }

    // Next let's handle the 4-byte irregular mappings
    switch (rune) {
    // Vithkuqi: Albanian historical script
    case 0x10594: folded[0] = 0x105BB; return 1; // '𐖔' (U+010594, F0 90 96 94) → '𐖻' (U+0105BB, F0 90 96 BB)
    case 0x10595: folded[0] = 0x105BC; return 1; // '𐖕' (U+010595, F0 90 96 95) → '𐖼' (U+0105BC, F0 90 96 BC)
    }

    folded[0] = rune; return 1;  // No folding needed
    // clang-format on
}

/**
 * @brief Convert a Unicode codepoint to uppercase (Full Case Mapping).
 * @param rune The input codepoint
 * @param upper Output array for uppercase codepoint(s)
 * @return Number of codepoints written (1-3)
 */
SZ_INTERNAL sz_size_t sz_unicode_upper_codepoint_(sz_rune_t rune, sz_rune_t *upper) {
    // clang-format off

    // ASCII a-z → A-Z
    if (rune <= 0x7F) {
        if ((sz_u32_t)(rune - 0x61) <= 25) {
            upper[0] = rune - 0x20; return 1; }
        upper[0] = rune; return 1;
    }

    // 2-byte UTF-8 (U+0080-07FF)
    if (rune <= 0x7FF) {
        // Cyrillic а-я → uppercase (- 0x20)
        if ((sz_u32_t)(rune - 0x0430) <= 0x1F) {
            upper[0] = rune - 0x20; return 1; }

        // Latin-1 à-þ → uppercase (- 0x20)
        if ((sz_u32_t)(rune - 0x00E0) <= 0x1E) {
            upper[0] = rune - 0x20; return 1; }

        // Greek α-ρ → uppercase (- 0x20)
        if ((sz_u32_t)(rune - 0x03B1) <= 0x10) {
            upper[0] = rune - 0x20; return 1; }

        // Greek σ-ϋ → uppercase (- 0x20)
        if ((sz_u32_t)(rune - 0x03C3) <= 0x8) {
            upper[0] = rune - 0x20; return 1; }

        // Cyrillic ѐ-џ → uppercase (- 0x50)
        if ((sz_u32_t)(rune - 0x0450) <= 0xF) {
            upper[0] = rune - 0x50; return 1; }

        // Armenian ա-ֆ → uppercase (- 0x30)
        if ((sz_u32_t)(rune - 0x0561) <= 0x25) {
            upper[0] = rune - 0x30; return 1; }

        // Greek έ-ί → uppercase (- 0x25)
        if ((sz_u32_t)(rune - 0x03AD) <= 0x2) {
            upper[0] = rune - 0x25; return 1; }

        // Greek ͻ-ͽ → uppercase (+ 0x82)
        if ((sz_u32_t)(rune - 0x037B) <= 0x2) {
            upper[0] = rune + 0x82; return 1; }

        // Parity-based ranges (reversed: odd lowercase → -1)
        sz_u32_t is_odd = (rune & 1);

        // Latin Extended-A: Ā-Į (0x0100-0x012E, even → +1) (reversed)
        if ((sz_u32_t)(rune - 0x0101) <= 0x2E && is_odd) {
            upper[0] = rune - 1; return 1; }

        // Latin Extended-A: Ĳ-Ķ (0x0132-0x0136, even → +1) (reversed)
        if ((sz_u32_t)(rune - 0x0133) <= 0x4 && is_odd) {
            upper[0] = rune - 1; return 1; }

        // Latin Extended-A: Ĺ-Ň (0x0139-0x0147, odd → +1) (reversed)
        if ((sz_u32_t)(rune - 0x013A) <= 0xE && !is_odd) {
            upper[0] = rune - 1; return 1; }

        // Latin Extended-A: Ŋ-Ŷ (0x014A-0x0176, even → +1) (reversed)
        if ((sz_u32_t)(rune - 0x014B) <= 0x2C && is_odd) {
            upper[0] = rune - 1; return 1; }

        // Latin Extended-A: Ź-Ž (0x0179-0x017D, odd → +1) (reversed)
        if ((sz_u32_t)(rune - 0x017A) <= 0x4 && !is_odd) {
            upper[0] = rune - 1; return 1; }

        // Latin Extended-B: Ǎ-Ǜ (0x01CD-0x01DB, odd → +1) (reversed)
        if ((sz_u32_t)(rune - 0x01CE) <= 0xE && !is_odd) {
            upper[0] = rune - 1; return 1; }

        // Latin Extended-B: Ǟ-Ǯ (0x01DE-0x01EE, even → +1) (reversed)
        if ((sz_u32_t)(rune - 0x01DF) <= 0x10 && is_odd) {
            upper[0] = rune - 1; return 1; }

        // Latin Extended-B: Ǹ-Ǿ (0x01F8-0x01FE, even → +1) (reversed)
        if ((sz_u32_t)(rune - 0x01F9) <= 0x6 && is_odd) {
            upper[0] = rune - 1; return 1; }

        // Latin Extended-B: Ȁ-Ȟ (0x0200-0x021E, even → +1) (reversed)
        if ((sz_u32_t)(rune - 0x0201) <= 0x1E && is_odd) {
            upper[0] = rune - 1; return 1; }

        // Latin Extended-B: Ȣ-Ȳ (0x0222-0x0232, even → +1) (reversed)
        if ((sz_u32_t)(rune - 0x0223) <= 0x10 && is_odd) {
            upper[0] = rune - 1; return 1; }

        // Latin Extended-B: Ɇ-Ɏ (0x0246-0x024E, even → +1) (reversed)
        if ((sz_u32_t)(rune - 0x0247) <= 0x8 && is_odd) {
            upper[0] = rune - 1; return 1; }

        // Greek archaic: Ͱ-Ͳ (0x0370-0x0372, even → +1) (reversed)
        if ((sz_u32_t)(rune - 0x0371) <= 0x2 && is_odd) {
            upper[0] = rune - 1; return 1; }

        // Greek archaic: Ϙ-Ϯ (0x03D8-0x03EE, even → +1) (reversed)
        if ((sz_u32_t)(rune - 0x03D9) <= 0x16 && is_odd) {
            upper[0] = rune - 1; return 1; }

        // Cyrillic extended: Ѡ-Ҁ (0x0460-0x0480, even → +1) (reversed)
        if ((sz_u32_t)(rune - 0x0461) <= 0x20 && is_odd) {
            upper[0] = rune - 1; return 1; }

        // Cyrillic extended: Ҋ-Ҿ (0x048A-0x04BE, even → +1) (reversed)
        if ((sz_u32_t)(rune - 0x048B) <= 0x34 && is_odd) {
            upper[0] = rune - 1; return 1; }

        // Cyrillic extended: Ӂ-Ӎ (0x04C1-0x04CD, odd → +1) (reversed)
        if ((sz_u32_t)(rune - 0x04C2) <= 0xC && !is_odd) {
            upper[0] = rune - 1; return 1; }

        // Cyrillic extended: Ӑ-Ӿ (0x04D0-0x04FE, even → +1) (reversed)
        if ((sz_u32_t)(rune - 0x04D1) <= 0x2E && is_odd) {
            upper[0] = rune - 1; return 1; }

        // Cyrillic extended: Ԁ-Ԯ (0x0500-0x052E, even → +1) (reversed)
        if ((sz_u32_t)(rune - 0x0501) <= 0x2E && is_odd) {
            upper[0] = rune - 1; return 1; }

        // Irregular 1:1 mappings (reversed)
        switch (rune) {
        case 0x00FF: upper[0] = 0x0178; return 1;
        case 0x017F: upper[0] = 0x0053; return 1;
        case 0x0180: upper[0] = 0x0243; return 1;
        case 0x0183: upper[0] = 0x0182; return 1;
        case 0x0185: upper[0] = 0x0184; return 1;
        case 0x0188: upper[0] = 0x0187; return 1;
        case 0x018C: upper[0] = 0x018B; return 1;
        case 0x0192: upper[0] = 0x0191; return 1;
        case 0x0195: upper[0] = 0x01F6; return 1;
        case 0x0199: upper[0] = 0x0198; return 1;
        case 0x019A: upper[0] = 0x023D; return 1;
        case 0x019B: upper[0] = 0xA7DC; return 1;
        case 0x019E: upper[0] = 0x0220; return 1;
        case 0x01A1: upper[0] = 0x01A0; return 1;
        case 0x01A3: upper[0] = 0x01A2; return 1;
        case 0x01A5: upper[0] = 0x01A4; return 1;
        case 0x01A8: upper[0] = 0x01A7; return 1;
        case 0x01AD: upper[0] = 0x01AC; return 1;
        case 0x01B0: upper[0] = 0x01AF; return 1;
        case 0x01B4: upper[0] = 0x01B3; return 1;
        case 0x01B6: upper[0] = 0x01B5; return 1;
        case 0x01B9: upper[0] = 0x01B8; return 1;
        case 0x01BD: upper[0] = 0x01BC; return 1;
        case 0x01BF: upper[0] = 0x01F7; return 1;
        case 0x01C6: upper[0] = 0x01C4; return 1;
        case 0x01C9: upper[0] = 0x01C7; return 1;
        case 0x01CC: upper[0] = 0x01CA; return 1;
        case 0x01DD: upper[0] = 0x018E; return 1;
        case 0x01F3: upper[0] = 0x01F1; return 1;
        case 0x01F5: upper[0] = 0x01F4; return 1;
        case 0x023C: upper[0] = 0x023B; return 1;
        case 0x023F: upper[0] = 0x2C7E; return 1;
        case 0x0240: upper[0] = 0x2C7F; return 1;
        case 0x0242: upper[0] = 0x0241; return 1;
        case 0x0250: upper[0] = 0x2C6F; return 1;
        case 0x0251: upper[0] = 0x2C6D; return 1;
        case 0x0252: upper[0] = 0x2C70; return 1;
        case 0x0253: upper[0] = 0x0181; return 1;
        case 0x0254: upper[0] = 0x0186; return 1;
        case 0x0256: upper[0] = 0x0189; return 1;
        case 0x0257: upper[0] = 0x018A; return 1;
        case 0x0259: upper[0] = 0x018F; return 1;
        case 0x025B: upper[0] = 0x0190; return 1;
        case 0x025C: upper[0] = 0xA7AB; return 1;
        case 0x0260: upper[0] = 0x0193; return 1;
        case 0x0261: upper[0] = 0xA7AC; return 1;
        case 0x0263: upper[0] = 0x0194; return 1;
        case 0x0264: upper[0] = 0xA7CB; return 1;
        case 0x0265: upper[0] = 0xA78D; return 1;
        case 0x0266: upper[0] = 0xA7AA; return 1;
        case 0x0268: upper[0] = 0x0197; return 1;
        case 0x0269: upper[0] = 0x0196; return 1;
        case 0x026A: upper[0] = 0xA7AE; return 1;
        case 0x026B: upper[0] = 0x2C62; return 1;
        case 0x026C: upper[0] = 0xA7AD; return 1;
        case 0x026F: upper[0] = 0x019C; return 1;
        case 0x0271: upper[0] = 0x2C6E; return 1;
        case 0x0272: upper[0] = 0x019D; return 1;
        case 0x0275: upper[0] = 0x019F; return 1;
        case 0x027D: upper[0] = 0x2C64; return 1;
        case 0x0280: upper[0] = 0x01A6; return 1;
        case 0x0282: upper[0] = 0xA7C5; return 1;
        case 0x0283: upper[0] = 0x01A9; return 1;
        case 0x0287: upper[0] = 0xA7B1; return 1;
        case 0x0288: upper[0] = 0x01AE; return 1;
        case 0x0289: upper[0] = 0x0244; return 1;
        case 0x028A: upper[0] = 0x01B1; return 1;
        case 0x028B: upper[0] = 0x01B2; return 1;
        case 0x028C: upper[0] = 0x0245; return 1;
        case 0x0292: upper[0] = 0x01B7; return 1;
        case 0x029D: upper[0] = 0xA7B2; return 1;
        case 0x029E: upper[0] = 0xA7B0; return 1;
        case 0x0377: upper[0] = 0x0376; return 1;
        case 0x03AC: upper[0] = 0x0386; return 1;
        case 0x03C2: upper[0] = 0x03A3; return 1;
        case 0x03C3: upper[0] = 0x03A3; return 1;
        case 0x03CC: upper[0] = 0x038C; return 1;
        case 0x03CD: upper[0] = 0x038E; return 1;
        case 0x03CE: upper[0] = 0x038F; return 1;
        case 0x03D7: upper[0] = 0x03CF; return 1;
        case 0x03F2: upper[0] = 0x03F9; return 1;
        case 0x03F3: upper[0] = 0x037F; return 1;
        case 0x03F8: upper[0] = 0x03F7; return 1;
        case 0x03FB: upper[0] = 0x03FA; return 1;
        case 0x04CF: upper[0] = 0x04C0; return 1;
        // ß → SS (full uppercase)
        case 0x00DF: upper[0] = 0x0053; upper[1] = 0x0053; return 2;
        }

        upper[0] = rune; return 1;
    }

    // 3-byte UTF-8 (U+0800-FFFF)
    if (rune <= 0xFFFF) {
        // Georgian Ⴀ-Ⴥ: 0x10A0-0x10C5 (+7264) → uppercase (- 0x1C60)
        if ((sz_u32_t)(rune - 0x2D00) <= 0x25) {
            upper[0] = rune - 0x1C60; return 1; }

        // Georgian Mtavruli Ა-Ჺ: 0x1C90-0x1CBA (-3008) → uppercase (+ 0xBC0)
        if ((sz_u32_t)(rune - 0x10D0) <= 0x2A) {
            upper[0] = rune + 0xBC0; return 1; }

        // Georgian Mtavruli Ჽ-Ჿ: 0x1CBD-0x1CBF (-3008) → uppercase (+ 0xBC0)
        if ((sz_u32_t)(rune - 0x10FD) <= 0x2) {
            upper[0] = rune + 0xBC0; return 1; }

        // Cherokee Ᏸ-Ᏽ: 0x13F8-0x13FD (-8) → uppercase (+ 0x8)
        if ((sz_u32_t)(rune - 0x13F0) <= 0x5) {
            upper[0] = rune + 0x8; return 1; }

        // Cherokee Ꭰ-Ᏼ: 0xAB70-0xABBF → Ꭰ-Ᏼ: 0x13A0-0x13EF (-38864) → uppercase (+ 0x97D0)
        if ((sz_u32_t)(rune - 0x13A0) <= 0x4F) {
            upper[0] = rune + 0x97D0; return 1; }

        // Greek Extended: multiple -8 offset ranges → uppercase (+ 0x8)
        if ((sz_u32_t)(rune - 0x1F00) <= 0x7) {
            upper[0] = rune + 0x8; return 1; }

        // Greek Extended Ὲ-Ή: 0x1FC8-0x1FCB (-86) → uppercase (+ 0x56)
        if ((sz_u32_t)(rune - 0x1F72) <= 0x3) {
            upper[0] = rune + 0x56; return 1; }

        // Roman numerals Ⅰ-Ⅿ: 0x2160-0x216F (+16) → uppercase (- 0x10)
        if ((sz_u32_t)(rune - 0x2170) <= 0xF) {
            upper[0] = rune - 0x10; return 1; }

        // Circled letters Ⓐ-Ⓩ: 0x24B6-0x24CF (+26) → uppercase (- 0x1A)
        if ((sz_u32_t)(rune - 0x24D0) <= 0x19) {
            upper[0] = rune - 0x1A; return 1; }

        // Glagolitic Ⰰ-Ⱟ: 0x2C00-0x2C2F (+48) → uppercase (- 0x30)
        if ((sz_u32_t)(rune - 0x2C30) <= 0x2F) {
            upper[0] = rune - 0x30; return 1; }

        // Fullwidth Ａ-Ｚ: 0xFF21-0xFF3A (+32) → uppercase (- 0x20)
        if ((sz_u32_t)(rune - 0xFF41) <= 0x19) {
            upper[0] = rune - 0x20; return 1; }

        // Parity-based ranges (odd lowercase → -1)
        sz_u32_t is_odd_3 = (rune & 1);

        // Latin Extended Additional Ḁ-Ẕ: 0x1E00-0x1E94 (reversed)
        if ((sz_u32_t)(rune - 0x1E01) <= 0x94 && is_odd_3) {
            upper[0] = rune - 1; return 1; }

        // Latin Extended Additional (Vietnamese) Ạ-Ỿ: 0x1EA0-0x1EFE (reversed)
        if ((sz_u32_t)(rune - 0x1EA1) <= 0x5E && is_odd_3) {
            upper[0] = rune - 1; return 1; }

        // Coptic Ⲁ-Ⳣ: 0x2C80-0x2CE2 (reversed)
        if ((sz_u32_t)(rune - 0x2C81) <= 0x62 && is_odd_3) {
            upper[0] = rune - 1; return 1; }

        // Cyrillic Extended-B Ꙁ-Ꙭ: 0xA640-0xA66C (reversed)
        if ((sz_u32_t)(rune - 0xA641) <= 0x2C && is_odd_3) {
            upper[0] = rune - 1; return 1; }

        // Cyrillic Extended-B Ꚁ-Ꚛ: 0xA680-0xA69A (reversed)
        if ((sz_u32_t)(rune - 0xA681) <= 0x1A && is_odd_3) {
            upper[0] = rune - 1; return 1; }

        // Latin Extended-D ranges (reversed)
        if ((sz_u32_t)(rune - 0xA723) <= 0xC && is_odd_3) {
            upper[0] = rune - 1; return 1; }

        // 3-byte irregular 1:1 mappings (reversed)
        switch (rune) {
        case 0x1C8A: upper[0] = 0x1C89; return 1;
        case 0x1D79: upper[0] = 0xA77D; return 1;
        case 0x1D7D: upper[0] = 0x2C63; return 1;
        case 0x1D8E: upper[0] = 0xA7C6; return 1;
        case 0x1E61: upper[0] = 0x1E9B; return 1;
        case 0x1F51: upper[0] = 0x1F59; return 1;
        case 0x1F53: upper[0] = 0x1F5B; return 1;
        case 0x1F55: upper[0] = 0x1F5D; return 1;
        case 0x1F57: upper[0] = 0x1F5F; return 1;
        case 0x1F70: upper[0] = 0x1FBA; return 1;
        case 0x1F71: upper[0] = 0x1FBB; return 1;
        case 0x1F76: upper[0] = 0x1FDA; return 1;
        case 0x1F77: upper[0] = 0x1FDB; return 1;
        case 0x1F78: upper[0] = 0x1FF8; return 1;
        case 0x1F79: upper[0] = 0x1FF9; return 1;
        case 0x1F7A: upper[0] = 0x1FEA; return 1;
        case 0x1F7B: upper[0] = 0x1FEB; return 1;
        case 0x1F7C: upper[0] = 0x1FFA; return 1;
        case 0x1F7D: upper[0] = 0x1FFB; return 1;
        case 0x1FB0: upper[0] = 0x1FB8; return 1;
        case 0x1FB1: upper[0] = 0x1FB9; return 1;
        case 0x1FD0: upper[0] = 0x1FD8; return 1;
        case 0x1FD1: upper[0] = 0x1FD9; return 1;
        case 0x1FE0: upper[0] = 0x1FE8; return 1;
        case 0x1FE1: upper[0] = 0x1FE9; return 1;
        case 0x1FE5: upper[0] = 0x1FEC; return 1;
        case 0x214E: upper[0] = 0x2132; return 1;
        case 0x2184: upper[0] = 0x2183; return 1;
        case 0x2C61: upper[0] = 0x2C60; return 1;
        case 0x2C65: upper[0] = 0x023A; return 1;
        case 0x2C66: upper[0] = 0x023E; return 1;
        case 0x2C68: upper[0] = 0x2C67; return 1;
        case 0x2C6A: upper[0] = 0x2C69; return 1;
        case 0x2C6C: upper[0] = 0x2C6B; return 1;
        case 0x2C73: upper[0] = 0x2C72; return 1;
        case 0x2C76: upper[0] = 0x2C75; return 1;
        case 0x2CEC: upper[0] = 0x2CEB; return 1;
        case 0x2CEE: upper[0] = 0x2CED; return 1;
        case 0x2CF3: upper[0] = 0x2CF2; return 1;
        case 0x2D27: upper[0] = 0x10C7; return 1;
        case 0x2D2D: upper[0] = 0x10CD; return 1;
        case 0xA77A: upper[0] = 0xA779; return 1;
        case 0xA77C: upper[0] = 0xA77B; return 1;
        case 0xA78C: upper[0] = 0xA78B; return 1;
        case 0xA794: upper[0] = 0xA7C4; return 1;
        case 0xA7C8: upper[0] = 0xA7C7; return 1;
        case 0xA7CA: upper[0] = 0xA7C9; return 1;
        case 0xA7CD: upper[0] = 0xA7CC; return 1;
        case 0xA7CF: upper[0] = 0xA7CE; return 1;
        case 0xA7D1: upper[0] = 0xA7D0; return 1;
        case 0xA7D3: upper[0] = 0xA7D2; return 1;
        case 0xA7D5: upper[0] = 0xA7D4; return 1;
        case 0xA7D7: upper[0] = 0xA7D6; return 1;
        case 0xA7D9: upper[0] = 0xA7D8; return 1;
        case 0xA7DB: upper[0] = 0xA7DA; return 1;
        case 0xA7F6: upper[0] = 0xA7F5; return 1;
        case 0xAB53: upper[0] = 0xA7B3; return 1;
        // Typographic ligatures → uppercase expansion
        case 0xFB00: upper[0] = 0x0046; upper[1] = 0x0046; return 2; // ﬀ → FF
        case 0xFB01: upper[0] = 0x0046; upper[1] = 0x0049; return 2; // ﬁ → FI
        case 0xFB02: upper[0] = 0x0046; upper[1] = 0x004C; return 2; // ﬂ → FL
        case 0xFB03: upper[0] = 0x0046; upper[1] = 0x0046; upper[2] = 0x0049; return 3; // ﬃ → FFI
        case 0xFB04: upper[0] = 0x0046; upper[1] = 0x0046; upper[2] = 0x004C; return 3; // ﬄ → FFL
        case 0xFB05: upper[0] = 0x0053; upper[1] = 0x0054; return 2; // ﬅ → ST
        case 0xFB06: upper[0] = 0x0053; upper[1] = 0x0054; return 2; // ﬆ → ST
        }

        upper[0] = rune; return 1;
    }

    // 4-byte UTF-8 (U+10000+)
    // Deseret 𐐀-𐐧: 0x10400-0x10427 (+40) → uppercase (- 0x28)
    if ((sz_u32_t)(rune - 0x10428) <= 0x27) {
        upper[0] = rune - 0x28; return 1; }

    // Osage 𐒰-𐓓: 0x104B0-0x104D3 (+40) → uppercase (- 0x28)
    if ((sz_u32_t)(rune - 0x104D8) <= 0x23) {
        upper[0] = rune - 0x28; return 1; }

    // Vithkuqi: 3 ranges with gaps, all +39 → uppercase (- 0x27)
    if ((sz_u32_t)(rune - 0x10597) <= 0xA) {
        upper[0] = rune - 0x27; return 1; }

    // Old Hungarian: 0x10C80-0x10CB2 (+64) → uppercase (- 0x40)
    if ((sz_u32_t)(rune - 0x10CC0) <= 0x32) {
        upper[0] = rune - 0x40; return 1; }

    // Garay: 0x10D50-0x10D65 (+32) → uppercase (- 0x20)
    if ((sz_u32_t)(rune - 0x10D70) <= 0x15) {
        upper[0] = rune - 0x20; return 1; }

    // Warang Citi: 0x118A0-0x118BF (+32) → uppercase (- 0x20)
    if ((sz_u32_t)(rune - 0x118C0) <= 0x1F) {
        upper[0] = rune - 0x20; return 1; }

    // Medefaidrin: 0x16E40-0x16E5F (+32) → uppercase (- 0x20)
    if ((sz_u32_t)(rune - 0x16E60) <= 0x1F) {
        upper[0] = rune - 0x20; return 1; }

    // Beria Erfe: 0x16EA0-0x16EB8 (+27) → uppercase (- 0x1B)
    if ((sz_u32_t)(rune - 0x16EBB) <= 0x18) {
        upper[0] = rune - 0x1B; return 1; }

    // Adlam: 0x1E900-0x1E921 (+34) → uppercase (- 0x22)
    if ((sz_u32_t)(rune - 0x1E922) <= 0x21) {
        upper[0] = rune - 0x22; return 1; }

    // 4-byte irregular 1:1 mappings (reversed)
    switch (rune) {
    case 0x105BB: upper[0] = 0x10594; return 1;
    case 0x105BC: upper[0] = 0x10595; return 1;
    }

    upper[0] = rune; return 1;

    // clang-format on
}

/**
 *  @brief  Branchless ASCII case fold - converts A-Z to a-z.
 *  Uses unsigned subtraction trick: (c - 'A') <= 25 is true only for uppercase letters.
 */
SZ_INTERNAL sz_u8_t sz_ascii_fold_(sz_u8_t c) { return c + (((sz_u8_t)(c - 'A') <= 25u) * 0x20); }

/**
 *  @brief  Branchless ASCII uppercase - converts a-z to A-Z.
 *  Uses unsigned subtraction trick: (c - 'a') <= 25 is true only for lowercase letters.
 */
SZ_INTERNAL sz_u8_t sz_ascii_upper_(sz_u8_t c) { return c - (((sz_u8_t)(c - 'a') <= 25u) * 0x20); }

/**
 *  @brief  Iterator state for streaming through folded UTF-8 runes.
 *  Handles one-to-many case folding expansions (e.g., 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)) transparently.
 */
typedef struct {
    sz_cptr_t ptr;           // Current position in UTF-8 string
    sz_cptr_t end;           // End of string
    sz_rune_t pending[4];    // Buffered folded runes from one-to-many expansions
    sz_size_t pending_count; // Number of pending folded runes
    sz_size_t pending_idx;   // Current index into pending buffer
} sz_utf8_folded_iter_t_;

/** @brief Initialize a folded rune iterator. */
SZ_INTERNAL void sz_utf8_folded_iter_init_(sz_utf8_folded_iter_t_ *it, sz_cptr_t str, sz_size_t len) {
    it->ptr = str;
    it->end = str + len;
    it->pending_count = 0;
    it->pending_idx = 0;
}

/** @brief Get next folded rune. Returns `sz_false_k` when exhausted. Assumes valid UTF-8 input. */
SZ_INTERNAL sz_bool_t sz_utf8_folded_iter_next_(sz_utf8_folded_iter_t_ *it, sz_rune_t *out_rune) {
    // Refill pending buffer if exhausted
    if (it->pending_idx >= it->pending_count) {
        if (it->ptr >= it->end) return sz_false_k;

        // ASCII fast-path: fold inline without buffering
        sz_u8_t lead = *(sz_u8_t const *)it->ptr;
        if (lead < 0x80) {
            *out_rune = sz_ascii_fold_(lead);
            it->ptr++;
            it->pending_count = 0; // Clear pending buffer
            it->pending_idx = 0;   // Signal first rune of new codepoint for source tracking
            return sz_true_k;
        }

        // Multi-byte UTF-8: decode, fold, and buffer
        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse(it->ptr, &rune, &rune_length);

        it->ptr += rune_length;
        // Pre-fill pending buffer with sentinel values to prevent stale data from causing false matches.
        // The fold function will overwrite positions it uses; unused positions keep the sentinel.
        // This follows the same pattern as sz_utf8_case_insensitive_find_2folded_serial_ and
        // sz_utf8_case_insensitive_find_3folded_serial_.
        it->pending[0] = 0xFFFFFFFFu;
        it->pending[1] = 0xFFFFFFFEu;
        it->pending[2] = 0xFFFFFFFDu;
        it->pending[3] = 0xFFFFFFFCu;
        it->pending_count = sz_unicode_fold_codepoint_(rune, it->pending);
        it->pending_idx = 0;
    }

    *out_rune = it->pending[it->pending_idx++];
    return sz_true_k;
}

/**
 *  @brief  Reverse iterator state for streaming through folded UTF-8 runes backwards.
 * Handles one-to-many case folding expansions (e.g., 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)) transparently
 * in reverse order.
 */
typedef struct {
    sz_cptr_t ptr;           // Current position (points to byte AFTER current sequence)
    sz_cptr_t start;         // Start of string (stop when ptr reaches this)
    sz_rune_t pending[4];    // Buffered folded runes from one-to-many expansions (in reverse order)
    sz_size_t pending_count; // Number of pending folded runes
    sz_size_t pending_idx;   // Current index into pending buffer
} sz_utf8_folded_reverse_iter_t_;

/** @brief Initialize a reverse folded rune iterator. Iterates from end towards start. */
SZ_INTERNAL void sz_utf8_folded_reverse_iter_init_(sz_utf8_folded_reverse_iter_t_ *it, sz_cptr_t start, sz_cptr_t end) {
    it->ptr = end;
    it->start = start;
    it->pending_count = 0;
    it->pending_idx = 0;
}

/**
 *  @brief Get previous folded rune (walking backwards). Returns `sz_false_k` when exhausted.
 * When a codepoint folds to multiple runes (like 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)), returns them in
 * reverse order ('s', then 's').
 */
SZ_INTERNAL sz_bool_t sz_utf8_folded_reverse_iter_prev_(sz_utf8_folded_reverse_iter_t_ *it, sz_rune_t *out_rune) {
    // Return pending runes if any (stored in reverse order, consumed in reverse)
    if (it->pending_idx < it->pending_count) {
        *out_rune = it->pending[it->pending_count - 1 - it->pending_idx];
        it->pending_idx++;
        return sz_true_k;
    }

    // Refill: find previous codepoint
    if (it->ptr <= it->start) return sz_false_k;

    // Walk backwards to find start of UTF-8 sequence
    // Continuation bytes have form 10xxxxxx (0x80-0xBF)
    it->ptr--;
    while (it->ptr > it->start && (*(sz_u8_t const *)it->ptr & 0xC0) == 0x80) { it->ptr--; }

    // ASCII fast-path
    sz_u8_t lead = *(sz_u8_t const *)it->ptr;
    if (lead < 0x80) {
        *out_rune = sz_ascii_fold_(lead);
        it->pending_count = 0;
        it->pending_idx = 0;
        return sz_true_k;
    }

    // Multi-byte UTF-8: decode and fold
    sz_rune_t rune;
    sz_rune_length_t rune_length;
    sz_rune_parse(it->ptr, &rune, &rune_length);

    // Store folded runes in pending buffer
    it->pending[0] = 0xFFFFFFFFu;
    it->pending[1] = 0xFFFFFFFEu;
    it->pending[2] = 0xFFFFFFFDu;
    it->pending[3] = 0xFFFFFFFCu;
    it->pending_count = sz_unicode_fold_codepoint_(rune, it->pending);
    it->pending_idx = 1; // We'll return the last one now, then the rest in subsequent calls

    // Return the LAST folded rune first (since we're going backwards)
    *out_rune = it->pending[it->pending_count - 1];
    return sz_true_k;
}

/**
 *  @brief Verify head region case-insensitively (backward iteration).
 *
 *  Walks backward from needle_end/haystack_end, comparing folded runes.
 *  Returns true if needle region exhausts (matched), with haystack bytes consumed.
 *
 *  @param[in] needle_start Start of needle head region
 *  @param[in] needle_end End of needle head region (where safe window begins)
 *  @param[in] haystack_start Start of haystack (lower bound for backward scan)
 *  @param[in] haystack_end End of haystack head region (where safe window was found)
 *  @param[out] match_length Haystack bytes consumed by this match
 */
SZ_INTERNAL sz_bool_t sz_utf8_case_insensitive_verify_head_(sz_cptr_t needle_start, sz_cptr_t needle_end,
                                                            sz_cptr_t haystack_start, sz_cptr_t haystack_end,
                                                            sz_size_t *match_length) {

    // If needle head is empty, no haystack bytes needed
    if (needle_end <= needle_start) {
        *match_length = 0;
        return sz_true_k;
    }

    sz_utf8_folded_reverse_iter_t_ needle_riter, haystack_riter;
    sz_utf8_folded_reverse_iter_init_(&needle_riter, needle_start, needle_end);
    sz_utf8_folded_reverse_iter_init_(&haystack_riter, haystack_start, haystack_end);

    sz_rune_t needle_rune = 0, haystack_rune = 0;
    for (;;) {
        sz_bool_t have_needle = sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_rune);

        // Needle exhausted - success! Unconsumed haystack runes are OK.
        // Example: "fi" matches suffix of "ﬃ" (folds to "ffi"), leaving first 'f' unused.
        if (!have_needle) {
            *match_length = (sz_size_t)(haystack_end - haystack_riter.ptr);
            return sz_true_k;
        }

        sz_bool_t have_haystack = sz_utf8_folded_reverse_iter_prev_(&haystack_riter, &haystack_rune);
        if (!have_haystack) return sz_false_k;
        if (needle_rune != haystack_rune) return sz_false_k;
    }
}

/**
 *  @brief Verify tail region case-insensitively (forward iteration).
 *
 *  Walks forward, comparing folded runes. Returns true if needle exhausts.
 *
 *  @param[in] needle_start Start of needle tail region
 *  @param[in] needle_end End of needle tail region (= needle + needle_length)
 *  @param[in] haystack_start Start of haystack tail region
 *  @param[in] haystack_end End of haystack (upper bound for forward scan)
 *  @param[out] match_length Haystack bytes consumed by this match
 */
SZ_INTERNAL sz_bool_t sz_utf8_case_insensitive_verify_tail_(sz_cptr_t needle_start, sz_cptr_t needle_end,
                                                            sz_cptr_t haystack_start, sz_cptr_t haystack_end,
                                                            sz_size_t *match_length) {

    sz_size_t needle_length = (sz_size_t)(needle_end - needle_start);

    // Empty tail is trivially matched
    if (needle_length == 0) {
        *match_length = 0;
        return sz_true_k;
    }

    sz_utf8_folded_iter_t_ needle_iter, haystack_iter;
    sz_utf8_folded_iter_init_(&needle_iter, needle_start, needle_length);
    sz_utf8_folded_iter_init_(&haystack_iter, haystack_start, (sz_size_t)(haystack_end - haystack_start));

    sz_rune_t needle_rune = 0, haystack_rune = 0;
    for (;;) {
        sz_bool_t have_needle = sz_utf8_folded_iter_next_(&needle_iter, &needle_rune);

        if (!have_needle) {
            // Needle exhausted - success!
            *match_length = (sz_size_t)(haystack_iter.ptr - haystack_start);
            return sz_true_k;
        }

        sz_bool_t have_haystack = sz_utf8_folded_iter_next_(&haystack_iter, &haystack_rune);
        if (!have_haystack) return sz_false_k;
        if (needle_rune != haystack_rune) return sz_false_k;
    }
}

/**
 *  @brief Verify a complete match around a SIMD-detected window.
 *
 *  Verifies two regions: "head" (before window) and "tail" (after window).
 *  It's important to note that the middle part may still be in part unprocessed, if its larger
 *  than the "folded slice" of the needle. We handle it as part of the "tail" and the `needle_tail_bytes`
 *  must be calculated accordingly.
 *
 *  @param[in] haystack_ptr Haystack start pointer, arbitrary case
 *  @param[in] haystack_length Haystack length in bytes
 *  @param[in] needle_ptr Needle start pointer, arbitrary case
 *  @param[in] needle_length Needle length in bytes
 *  @param[in] haystack_matched_offset Start offset of matched safe window in haystack in bytes
 *  @param[in] haystack_matched_length Length of matched safe window in haystack in bytes
 *  @param[in] needle_head_bytes Start of matched safe window in needle in bytes
 *  @param[in] needle_tail_bytes Number of bytes in the needle remaining after the matched part
 *  @param[out] match_length Total length of the verified match in haystack bytes
 *  @return Match start pointer, or SZ_NULL_CHAR if validation fails
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_verify_match_(             //
    sz_cptr_t haystack_ptr, sz_size_t haystack_length,                    //
    sz_cptr_t needle_ptr, sz_size_t needle_length,                        //
    sz_size_t haystack_matched_offset, sz_size_t haystack_matched_length, //
    sz_size_t needle_head_bytes, sz_size_t needle_tail_bytes,             //
    sz_size_t *match_length) {

    sz_cptr_t needle_end = needle_ptr + needle_length;
    sz_cptr_t haystack_end = haystack_ptr + haystack_length;

    // Verify head using backward iterators
    sz_size_t head_match_length = 0;
    if (needle_head_bytes)
        if (!sz_utf8_case_insensitive_verify_head_(                   //
                needle_ptr, needle_ptr + needle_head_bytes,           // needle head region
                haystack_ptr, haystack_ptr + haystack_matched_offset, // haystack head region
                &head_match_length))
            return SZ_NULL_CHAR;

    // Verify tail using forward iterators
    sz_size_t tail_match_length = 0;
    sz_cptr_t haystack_tail_start = haystack_ptr + haystack_matched_offset + haystack_matched_length;
    if (needle_tail_bytes)
        if (!sz_utf8_case_insensitive_verify_tail_(                         //
                needle_ptr + needle_length - needle_tail_bytes, needle_end, // needle tail region
                haystack_tail_start, haystack_end,                          // haystack tail region
                &tail_match_length))
            return SZ_NULL_CHAR;

    *match_length = head_match_length + haystack_matched_length + tail_match_length;
    return haystack_ptr + haystack_matched_offset - head_match_length;
}

/**
 *  @brief  Internal helper: checks if a single Unicode codepoint is case-agnostic.
 *
 *  A codepoint is case-agnostic if ALL of the following are true:
 *  1. It folds to exactly itself (no transformation, no expansion)
 *  2. It does NOT belong to any bicameral (cased) script
 *  3. It does NOT appear in any case fold expansion as a target character
 *
 *  The third condition is critical. Consider 'ʾ' (U+02BE, CA BE):
 *  - It has no case variant and folds to itself
 *  - However, 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
 *  - A needle containing 'ʾ' must match at position 1 of the folded expansion of 'ẚ'
 *  - Binary search cannot handle this - it only sees 'ẚ' as a 3-byte sequence (E1 BA 9A)
 *  - Therefore 'ʾ' must NOT be treated as case-agnostic
 *
 *  This function implements the check via explicit range exclusions for all bicameral
 *  scripts and all Unicode blocks containing case fold expansion target characters.
 *
 *  @param[in] rune Unicode codepoint to check
 *  @return sz_true_k if the codepoint is case-agnostic, sz_false_k otherwise
 *
 *  @warning This is an internal function. Use sz_utf8_case_invariant_serial() for string checking.
 *  @see sz_utf8_case_invariant_serial
 *  @see sz_unicode_fold_codepoint_
 */
SZ_INTERNAL sz_bool_t sz_rune_is_case_invariant_(sz_rune_t rune) {

    // Check if this rune participates in case folding
    sz_rune_t folded_runes[3];
    sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);

    // If it expands or changes, it's not caseless
    if (folded_count != 1 || folded_runes[0] != rune) return sz_false_k;

    // Check if this rune is a lowercase target of some uppercase letter.
    // Lowercase letters that don't change when folded still participate in case
    // because uppercase versions fold TO them. We must mark entire bicameral
    // script ranges as "not caseless" to enable proper case-insensitive matching.
    //
    // Important: Combining diacritical marks (U+0300-U+036F) can appear as non-first
    // runes in multi-rune case fold expansions. Example: ǰ (U+01F0) → j + ̌ (U+030C).
    // A needle starting with combining caron could match inside such an expansion,
    // so combining marks must NOT be treated as case-agnostic.
    //
    // Bicameral scripts organized by UTF-8 lead byte for efficient checking:
    //
    // 1-byte sequences with upper and lower case (U+0000-007F): 00-7F
    if (rune >= 0x0041 && rune <= 0x005A) return sz_false_k; // Basic Latin (A-Z)
    if (rune >= 0x0061 && rune <= 0x007A) return sz_false_k; // Basic Latin (a-z)
    //
    // 2-byte sequences (U+0080-07FF): C2-DF lead bytes
    if (rune >= 0x00C0 && rune <= 0x00FF) return sz_false_k; // Latin-1 Supplement (À-ÿ)
    if (rune >= 0x0100 && rune <= 0x024F) return sz_false_k; // Latin Extended-A/B
    if (rune >= 0x0250 && rune <= 0x02AF) return sz_false_k; // IPA Extensions
    if (rune >= 0x02B0 && rune <= 0x02FF) return sz_false_k; // Spacing Modifier Letters (ʾ U+02BE appears in ẚ→aʾ)
    if (rune >= 0x0300 && rune <= 0x036F) return sz_false_k; // Combining Diacritical Marks (can appear in expansions!)
    if (rune >= 0x0370 && rune <= 0x03FF) return sz_false_k; // Greek and Coptic
    if (rune >= 0x0400 && rune <= 0x04FF) return sz_false_k; // Cyrillic
    if (rune >= 0x0500 && rune <= 0x052F) return sz_false_k; // Cyrillic Supplement
    if (rune >= 0x0531 && rune <= 0x0587) return sz_false_k; // Armenian (uppercase + lowercase + ligature)
    //
    // 3-byte sequences (U+0800-FFFF): E0-EF lead bytes
    if (rune >= 0x10A0 && rune <= 0x10FF) return sz_false_k; // Georgian (Asomtavruli + Mkhedruli)
    if (rune >= 0x13A0 && rune <= 0x13FD) return sz_false_k; // Cherokee (folds to uppercase!)
    if (rune >= 0x1C80 && rune <= 0x1C8F) return sz_false_k; // Cyrillic Extended-C
    if (rune >= 0x1C90 && rune <= 0x1CBF) return sz_false_k; // Georgian Extended (Mtavruli)
    if (rune >= 0x1E00 && rune <= 0x1EFF) return sz_false_k; // Latin Extended Additional
    if (rune >= 0x1F00 && rune <= 0x1FFF) return sz_false_k; // Greek Extended
    if (rune >= 0x2C00 && rune <= 0x2C5F) return sz_false_k; // Glagolitic
    if (rune >= 0x2C60 && rune <= 0x2C7F) return sz_false_k; // Latin Extended-C
    if (rune >= 0x2C80 && rune <= 0x2CFF) return sz_false_k; // Coptic
    if (rune >= 0x2D00 && rune <= 0x2D2F) return sz_false_k; // Georgian Supplement (Nuskhuri)
    if (rune >= 0x2DE0 && rune <= 0x2DFF) return sz_false_k; // Cyrillic Extended-A
    if (rune >= 0xA640 && rune <= 0xA69F) return sz_false_k; // Cyrillic Extended-B
    if (rune >= 0xA720 && rune <= 0xA7FF) return sz_false_k; // Latin Extended-D
    if (rune >= 0xAB30 && rune <= 0xAB6F) return sz_false_k; // Latin Extended-E
    if (rune >= 0xAB70 && rune <= 0xABBF) return sz_false_k; // Cherokee Supplement (lowercase)
    if (rune >= 0xFB00 && rune <= 0xFB06) return sz_false_k; // Alphabetic Presentation (ligatures)
    if (rune >= 0xFB13 && rune <= 0xFB17) return sz_false_k; // Armenian ligatures
    if (rune >= 0xFF21 && rune <= 0xFF5A) return sz_false_k; // Fullwidth Latin
    //
    // 4-byte sequences (U+10000-10FFFF): F0-F4 lead bytes
    if (rune >= 0x10400 && rune <= 0x1044F) return sz_false_k; // Deseret
    if (rune >= 0x104B0 && rune <= 0x104FF) return sz_false_k; // Osage
    if (rune >= 0x10570 && rune <= 0x105BF) return sz_false_k; // Vithkuqi
    if (rune >= 0x10780 && rune <= 0x107BF) return sz_false_k; // Latin Extended-F
    if (rune >= 0x10C80 && rune <= 0x10CFF) return sz_false_k; // Old Hungarian
    if (rune >= 0x118A0 && rune <= 0x118FF) return sz_false_k; // Warang Citi
    if (rune >= 0x16E40 && rune <= 0x16E9F) return sz_false_k; // Medefaidrin
    if (rune >= 0x1DF00 && rune <= 0x1DFFF) return sz_false_k; // Latin Extended-G
    if (rune >= 0x1E000 && rune <= 0x1E02F) return sz_false_k; // Glagolitic Supplement
    if (rune >= 0x1E030 && rune <= 0x1E08F) return sz_false_k; // Cyrillic Extended-D
    if (rune >= 0x1E900 && rune <= 0x1E95F) return sz_false_k; // Adlam

    return sz_true_k;
}

SZ_PUBLIC sz_bool_t sz_utf8_case_invariant_serial(sz_cptr_t str, sz_size_t length) {
    sz_u8_t const *ptr = (sz_u8_t const *)str;
    sz_u8_t const *end = ptr + length;

    while (ptr < end) {
        sz_u8_t lead = *ptr;

        // ASCII fast path: only digits, punctuation, and control chars are caseless
        // A-Z (0x41-0x5A) and a-z (0x61-0x7A) participate in case folding
        if (lead < 0x80) {
            if ((lead >= 'A' && lead <= 'Z') || (lead >= 'a' && lead <= 'z')) return sz_false_k;
            ptr++;
            continue;
        }

        // Multi-byte: decode and check
        sz_rune_t rune;
        sz_rune_length_t rune_len;
        sz_rune_parse((sz_cptr_t)ptr, &rune, &rune_len);
        if (sz_rune_is_case_invariant_(rune) == sz_false_k) return sz_false_k;
        ptr += rune_len;
    }

    return sz_true_k;
}

/**
 *  @brief  Hash-free case-insensitive search for needles that fold to exactly 1 rune.
 *          Examples: 'a', 'A', 'б', 'Б' (but NOT 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73) = 2 runes).
 *
 *  Single-pass algorithm: parses each source rune, folds it, checks if it produces
 *  exactly one rune matching the target. No iterator overhead, no verification needed.
 *
 *  @param[in] target_folded The single folded rune to search for.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_1folded_serial_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                   //
    sz_rune_t needle_folded, sz_size_t *match_length) {

    sz_cptr_t const haystack_end = haystack + haystack_length;

    // Each haystack rune may fold in up to 3 runes
    sz_rune_t haystack_rune;
    sz_rune_length_t haystack_rune_length;

    // If we simply initialize the runes for zero, the code will break
    // when the needle itself is the NUL character
    sz_rune_t haystack_folded_runes[3] = {~needle_folded};
    while (haystack < haystack_end) {
        sz_rune_parse(haystack, &haystack_rune, &haystack_rune_length);
        sz_unicode_fold_codepoint_(haystack_rune, haystack_folded_runes);

        // Perform branchless equality check via arithmetic
        sz_u32_t has_match =                              //
            (haystack_folded_runes[0] == needle_folded) + //
            (haystack_folded_runes[1] == needle_folded) + //
            (haystack_folded_runes[2] == needle_folded);

        if (has_match) {
            *match_length = haystack_rune_length;
            return haystack;
        }

        haystack += haystack_rune_length;
    }

    *match_length = 0;
    return SZ_NULL_CHAR;
}

/**
 *  @brief Search a "danger zone" region using 1-folded candidate search + validation.
 *
 *  When SIMD kernels detect potentially problematic bytes (ligatures, Greek Extended, etc.),
 *  they fall back to this serial search within the affected chunk. This function:
 *  1. Extracts the first folded rune from the needle's safe window
 *  2. Searches for candidates matching that rune
 *  3. Validates each candidate using the full verification pipeline
 *
 *  @param[in] haystack_ptr Full haystack string, arbitrary case
 *  @param[in] haystack_length Full haystack length in bytes
 *  @param[in] needle_ptr Full needle string, arbitrary case
 *  @param[in] needle_length Full needle length
 *  @param[in] danger_ptr Start of the danger zone region to search
 *  @param[in] danger_length Length of the danger zone region in bytes
 *  @param[in] needle_first_safe_folded_rune The first rune of the safe window, folded
 *  @param[in] needle_head_bytes Offset of the safe window within the needle
 *  @param[out] match_length Haystack bytes consumed by the match
 *  @return Pointer to match start, or SZ_NULL_CHAR if not found in this region
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_in_danger_zone_( //
    sz_cptr_t haystack_ptr, sz_size_t haystack_length,               //
    sz_cptr_t needle_ptr, sz_size_t needle_length,                   //
    sz_cptr_t danger_ptr, sz_size_t danger_length,                   //
    sz_rune_t needle_first_safe_folded_rune,                         //
    sz_size_t needle_first_safe_folded_rune_offset,                  //
    sz_size_t *match_length) {

    sz_cptr_t const haystack_end = haystack_ptr + haystack_length;
    sz_cptr_t const danger_end = sz_min_of_two(danger_ptr + danger_length, haystack_end);
    while (danger_ptr < danger_end) {

        // Skip continuation bytes - they are mid-sequence, not valid rune starts.
        // Without this check, a continuation byte like 0xBA could be misinterpreted as U+00BA (º),
        // causing false matches when the danger zone starts mid-character.
        sz_u8_t lead_byte = *(sz_u8_t const *)danger_ptr;
        if ((lead_byte & 0xC0) == 0x80) {
            danger_ptr++;
            continue;
        }

        // The following part is practically the unpacked variant of `sz_utf8_case_insensitive_find_1folded_serial_`,
        // that finds the first occurrence of the `needle_first_safe_folded_rune` haystack. The issue is that each one
        // `haystack_rune` may unpack into multiple `haystack_folded_runes`.
        sz_rune_t haystack_rune;
        sz_rune_length_t haystack_rune_length;
        sz_rune_t haystack_folded_runes[3] = {~needle_first_safe_folded_rune};
        sz_rune_parse(danger_ptr, &haystack_rune, &haystack_rune_length);
        sz_size_t haystack_folded_runes_count = sz_unicode_fold_codepoint_(haystack_rune, haystack_folded_runes);

        // The simplest case is when the very first in `haystack_folded_runes` is our target:
        if (haystack_folded_runes[0] == needle_first_safe_folded_rune) {
            // Validate the full match using the unified validator
            sz_cptr_t match = sz_utf8_case_insensitive_verify_match_( //
                haystack_ptr, haystack_length,                        //
                needle_ptr, needle_length,                            //
                danger_ptr - haystack_ptr, 0,                         // No pre-matched middle
                needle_first_safe_folded_rune_offset,
                needle_length - needle_first_safe_folded_rune_offset, // Verify everything after head serially
                match_length);

            if (match) return match;
            else { goto consider_second_haystack_folded_rune; } // We fall through here anyways :)
        }

    consider_second_haystack_folded_rune:

        // Check for a match at the second position in the folded haystack rune sequence
        if (haystack_folded_runes_count > 1 && haystack_folded_runes[1] == needle_first_safe_folded_rune) {
            sz_cptr_t haystack_match_start = 0, haystack_match_end = 0;

            // Check if the previous characters in the needle match the haystack before the danger zone began
            sz_rune_t needle_riter_rune = 0, haystack_riter_rune = 0;
            sz_utf8_folded_reverse_iter_t_ needle_riter, haystack_riter;
            sz_utf8_folded_reverse_iter_init_(&needle_riter, needle_ptr,
                                              needle_ptr + needle_first_safe_folded_rune_offset);
            sz_utf8_folded_reverse_iter_init_(&haystack_riter, haystack_ptr, danger_ptr);

            // Check if we even have needle bytes to check
            {
                sz_bool_t have_needle = sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_riter_rune);
                if (have_needle && needle_riter_rune != haystack_folded_runes[0])
                    goto consider_third_haystack_folded_rune;
            }

            // Loop backwards until we exhaust the needle head or find a mismatch
            for (;;) {
                sz_bool_t have_needle = sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_riter_rune);

                // Needle exhausted - success!
                if (!have_needle) {
                    haystack_match_start = haystack_riter.ptr;
                    break;
                }

                sz_bool_t have_haystack = sz_utf8_folded_reverse_iter_prev_(&haystack_riter, &haystack_riter_rune);
                if (!have_haystack) goto consider_third_haystack_folded_rune;
                if (needle_riter_rune != haystack_riter_rune) goto consider_third_haystack_folded_rune;
            }

            // First match the tail (from safe window start forward)
            sz_rune_t needle_iter_rune = 0, haystack_iter_rune = 0;
            sz_utf8_folded_iter_t_ needle_iter, haystack_iter;
            sz_utf8_folded_iter_init_(&needle_iter, needle_ptr + needle_first_safe_folded_rune_offset,
                                      needle_length - needle_first_safe_folded_rune_offset);
            sz_utf8_folded_iter_init_(&haystack_iter, danger_ptr + haystack_rune_length,
                                      haystack_end - (danger_ptr + haystack_rune_length));

            // Pop the `needle_first_safe_folded_rune` from the forward iterator
            {
                sz_bool_t have_needle = sz_utf8_folded_iter_next_(&needle_iter, &needle_iter_rune);
                sz_assert_(have_needle && needle_iter_rune == needle_first_safe_folded_rune);
            }

            // In some cases we already have the first point of comparison in the `haystack_folded_runes[2]`
            if (haystack_folded_runes_count == 3) {
                sz_bool_t have_needle = sz_utf8_folded_iter_next_(&needle_iter, &needle_iter_rune);
                if (have_needle && needle_iter_rune != haystack_folded_runes[2])
                    goto consider_third_haystack_folded_rune;
            }

            // Match the remaining tail runes
            for (;;) {
                sz_bool_t have_needle = sz_utf8_folded_iter_next_(&needle_iter, &needle_iter_rune);

                // Needle exhausted - success!
                if (!have_needle) {
                    haystack_match_end = haystack_iter.ptr;
                    break;
                }

                sz_bool_t have_haystack = sz_utf8_folded_iter_next_(&haystack_iter, &haystack_iter_rune);
                if (!have_haystack) goto consider_third_haystack_folded_rune;
                if (needle_iter_rune != haystack_iter_rune) goto consider_third_haystack_folded_rune;
            }

            // Check if we have a match to report
            if (haystack_match_start != 0 && haystack_match_end != 0) {
                *match_length = (sz_size_t)(haystack_match_end - haystack_match_start);
                return haystack_match_start;
            }
        }

    consider_third_haystack_folded_rune:

        // Check for a match at the second position in the folded haystack rune sequence
        if (haystack_folded_runes_count > 2 && haystack_folded_runes[2] == needle_first_safe_folded_rune) {
            sz_cptr_t haystack_match_start = 0, haystack_match_end = 0;

            // Check if the previous characters in the needle match the haystack before the danger zone began
            sz_rune_t needle_riter_rune = 0, haystack_riter_rune = 0;
            sz_utf8_folded_reverse_iter_t_ needle_riter, haystack_riter;
            sz_utf8_folded_reverse_iter_init_(&needle_riter, needle_ptr,
                                              needle_ptr + needle_first_safe_folded_rune_offset);
            sz_utf8_folded_reverse_iter_init_(&haystack_riter, haystack_ptr, danger_ptr);

            // Check if we even have needle bytes to check
            {
                sz_bool_t have_needle = sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_riter_rune);
                if (have_needle && needle_riter_rune != haystack_folded_runes[1])
                    goto consider_following_haystack_runes;
                have_needle = sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_riter_rune);
                if (have_needle && needle_riter_rune != haystack_folded_runes[0])
                    goto consider_following_haystack_runes;
            }

            // Loop backwards until we exhaust the needle head or find a mismatch
            for (;;) {
                sz_bool_t have_needle = sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_riter_rune);

                // Needle exhausted - success!
                if (!have_needle) {
                    haystack_match_start = haystack_riter.ptr;
                    break;
                }

                sz_bool_t have_haystack = sz_utf8_folded_reverse_iter_prev_(&haystack_riter, &haystack_riter_rune);
                if (!have_haystack) goto consider_following_haystack_runes;
                if (needle_riter_rune != haystack_riter_rune) goto consider_following_haystack_runes;
            }

            // First match the tail (from safe window start forward)
            sz_rune_t needle_iter_rune = 0, haystack_iter_rune = 0;
            sz_utf8_folded_iter_t_ needle_iter, haystack_iter;
            sz_utf8_folded_iter_init_(&needle_iter, needle_ptr + needle_first_safe_folded_rune_offset,
                                      needle_length - needle_first_safe_folded_rune_offset);
            sz_utf8_folded_iter_init_(&haystack_iter, danger_ptr + haystack_rune_length,
                                      haystack_end - (danger_ptr + haystack_rune_length));

            // Pop the `needle_first_safe_folded_rune` from the forward iterator
            {
                sz_bool_t have_needle = sz_utf8_folded_iter_next_(&needle_iter, &needle_iter_rune);
                sz_assert_(have_needle && needle_iter_rune == needle_first_safe_folded_rune);
            }

            // Match the remaining tail runes
            for (;;) {
                sz_bool_t have_needle = sz_utf8_folded_iter_next_(&needle_iter, &needle_iter_rune);

                // Needle exhausted - success!
                if (!have_needle) {
                    haystack_match_end = haystack_iter.ptr;
                    break;
                }

                sz_bool_t have_haystack = sz_utf8_folded_iter_next_(&haystack_iter, &haystack_iter_rune);
                if (!have_haystack) goto consider_following_haystack_runes;
                if (needle_iter_rune != haystack_iter_rune) goto consider_following_haystack_runes;
            }

            // Check if we have a match to report
            if (haystack_match_start != 0 && haystack_match_end != 0) {
                *match_length = (sz_size_t)(haystack_match_end - haystack_match_start);
                return haystack_match_start;
            }
        }

    consider_following_haystack_runes:
        // Move to next candidate
        danger_ptr += haystack_rune_length;
    }

    return SZ_NULL_CHAR;
}

/**
 *  @brief  Hash-free case-insensitive search for needles that fold to exactly 2 runes.
 * Examples: 'ab', 'AB', 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73), 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066
 * U+0069, 66 69).
 *
 * Single-pass sliding window over the folded rune stream. Handles expansions ('ß' (U+00DF, C3 9F) → "ss" (U+0073
 * U+0073, 73 73))
 *  by buffering folded runes from each source and tracking source boundaries.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_2folded_serial_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                   //
    sz_rune_t first_needle_folded, sz_rune_t second_needle_folded, sz_size_t *match_length) {

    sz_cptr_t const haystack_end = haystack + haystack_length;

    // Each haystack rune may fold in up to 3 runes, but we also keep an extra slot
    // for the last folded rune from the previous iterato step
    sz_rune_t haystack_rune;
    sz_rune_length_t haystack_rune_length, haystack_last_rune_length = sz_utf8_invalid_k;

    // If we simply initialize the runes for zero, the code will break
    // when the needle itself is the NUL character
    sz_rune_t haystack_folded_runes[4] = {~first_needle_folded};
    while (haystack < haystack_end) {
        sz_rune_parse(haystack, &haystack_rune, &haystack_rune_length);

        // Pre-fill positions [2] and [3] with sentinels before folding.
        // The fold will overwrite positions it uses; unused positions keep the sentinel.
        // This branchlessly prevents stale data from causing false matches.
        sz_rune_t sentinel = ~second_needle_folded;
        haystack_folded_runes[2] = sentinel;
        haystack_folded_runes[3] = sentinel;
        // Export into the last 3 rune entries of the 4-element array,
        // keeping the first position with historical data untouched
        sz_size_t folded_count = sz_unicode_fold_codepoint_(haystack_rune, haystack_folded_runes + 1);

        // Perform branchless equality check via arithmetic
        sz_u32_t has_match_f0 = first_needle_folded == haystack_folded_runes[0];
        sz_u32_t has_match_f1 = first_needle_folded == haystack_folded_runes[1];
        sz_u32_t has_match_f2 = first_needle_folded == haystack_folded_runes[2];
        sz_u32_t has_match_s1 = second_needle_folded == haystack_folded_runes[1];
        sz_u32_t has_match_s2 = second_needle_folded == haystack_folded_runes[2];
        sz_u32_t has_match_s3 = second_needle_folded == haystack_folded_runes[3];

        // Branchless match detection: each product is 0 or 1
        sz_u32_t match_at_01 = has_match_f0 * has_match_s1;
        sz_u32_t match_at_12 = has_match_f1 * has_match_s2;
        sz_u32_t match_at_23 = has_match_f2 * has_match_s3;
        sz_u32_t has_match = match_at_01 + match_at_12 + match_at_23;

        if (has_match) {
            // Only `match_at_01` spans sources; others are within current source
            sz_size_t back_offset = match_at_01 * (sz_size_t)haystack_last_rune_length;
            *match_length = (sz_size_t)haystack_rune_length + back_offset;
            return haystack - back_offset;
        }

        haystack_folded_runes[0] = haystack_folded_runes[folded_count];
        haystack_last_rune_length = haystack_rune_length;
        haystack += haystack_rune_length;
    }

    *match_length = 0;
    return SZ_NULL_CHAR;
}

/**
 *  @brief  Hash-free case-insensitive search for needles that fold to exactly 3 runes.
 * Examples: 'abc', 'ABC', "aß" ('ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)) → "ass" (U+0061 U+0073 U+0073, 61 73
 * 73),
 * "ﬁa" ('ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)) → "fia" (U+0066 U+0069 U+0061, 66 69 61).
 *
 *  Single-pass sliding window of 3 folded runes over the haystack's folded stream.
 *  Handles expansions ('ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)) by buffering and tracking source boundaries.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_3folded_serial_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                   //
    sz_rune_t first_needle_folded, sz_rune_t second_needle_folded, sz_rune_t third_needle_folded,
    sz_size_t *match_length) {

    sz_cptr_t const haystack_end = haystack + haystack_length;

    // Each haystack rune may fold in up to 3 runes, but we also keep an extra 2 slots
    // for the last folded rune from the previous iteration step, and the one before that
    sz_rune_t haystack_rune;
    sz_rune_length_t haystack_rune_length, haystack_last_rune_length = sz_utf8_invalid_k,
                                           haystack_preceding_rune_length = sz_utf8_invalid_k;

    // Initialize historical slots with sentinels that can never match their respective needle positions
    // This prevents false matches on first iterations when history is not yet populated
    sz_rune_t haystack_folded_runes[5] = {~first_needle_folded, ~second_needle_folded, 0, 0, 0};
    while (haystack < haystack_end) {
        sz_rune_parse(haystack, &haystack_rune, &haystack_rune_length);

        // Pre-fill positions [3] and [4] with sentinels before folding.
        // The fold will overwrite positions it uses; unused positions keep the sentinel.
        // This branchlessly prevents stale data from causing false matches.
        sz_rune_t sentinel = ~third_needle_folded;
        haystack_folded_runes[3] = sentinel;
        haystack_folded_runes[4] = sentinel;
        // Export into the last 3 rune entries of the 5-element array,
        // keeping the first two positions with historical data untouched
        sz_size_t folded_count = sz_unicode_fold_codepoint_(haystack_rune, haystack_folded_runes + 2);

        // Perform branchless equality check via arithmetic
        sz_u32_t has_match_f0 = first_needle_folded == haystack_folded_runes[0];
        sz_u32_t has_match_f1 = first_needle_folded == haystack_folded_runes[1];
        sz_u32_t has_match_f2 = first_needle_folded == haystack_folded_runes[2];
        sz_u32_t has_match_s1 = second_needle_folded == haystack_folded_runes[1];
        sz_u32_t has_match_s2 = second_needle_folded == haystack_folded_runes[2];
        sz_u32_t has_match_s3 = second_needle_folded == haystack_folded_runes[3];
        sz_u32_t has_match_t2 = third_needle_folded == haystack_folded_runes[2];
        sz_u32_t has_match_t3 = third_needle_folded == haystack_folded_runes[3];
        sz_u32_t has_match_t4 = third_needle_folded == haystack_folded_runes[4];

        // Branchless match detection: each product is 0 or 1
        sz_u32_t match_at_012 = has_match_f0 * has_match_s1 * has_match_t2;
        sz_u32_t match_at_123 = has_match_f1 * has_match_s2 * has_match_t3;
        sz_u32_t match_at_234 = has_match_f2 * has_match_s3 * has_match_t4;
        sz_u32_t has_match = match_at_012 + match_at_123 + match_at_234;

        if (has_match) {
            // Compute back offset based on which position matched:
            // - `match_at_012`: need preceding + last
            // - `match_at_123`: need last
            // - `match_at_234`: stay at current
            sz_size_t back_for_last = (match_at_012 + match_at_123) * (sz_size_t)haystack_last_rune_length;
            sz_size_t back_for_preceding = match_at_012 * (sz_size_t)haystack_preceding_rune_length;
            sz_size_t back_offset = back_for_last + back_for_preceding;
            *match_length = (sz_size_t)haystack_rune_length + back_offset;
            return haystack - back_offset;
        }

        // Historical context update here is a bit trickier than in previous spaces
        if (folded_count >= 2) {
            haystack_folded_runes[0] = haystack_folded_runes[folded_count];
            haystack_folded_runes[1] = haystack_folded_runes[folded_count + 1];
            haystack_preceding_rune_length = sz_utf8_invalid_k;
            haystack_last_rune_length = haystack_rune_length;
        }
        else {
            sz_assert_(folded_count == 1);
            haystack_folded_runes[0] = haystack_folded_runes[1];
            haystack_folded_runes[1] = haystack_folded_runes[2];
            haystack_preceding_rune_length = haystack_last_rune_length;
            haystack_last_rune_length = haystack_rune_length;
        }

        haystack += haystack_rune_length;
    }

    *match_length = 0;
    return SZ_NULL_CHAR;
}

/**
 *  @brief  Rabin-Karp style case-insensitive UTF-8 substring search using a ring buffer.
 *          Uses a rolling hash over casefolded runes with O(1) updates per position.
 *          Ring buffer of 32 runes handles the prefix; longer needles verify tail separately.
 */
SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_serial( //
    sz_cptr_t haystack, sz_size_t haystack_length,        //
    sz_cptr_t needle, sz_size_t needle_length,            //
    sz_utf8_case_insensitive_needle_metadata_t *needle_metadata, sz_size_t *match_length) {

    (void)needle_metadata; // Only used by SIMD kernels for debugging

    if (needle_length == 0) {
        *match_length = 0;
        return haystack;
    }

    if (sz_utf8_case_invariant_serial(needle, needle_length)) {
        sz_cptr_t result = sz_find(haystack, haystack_length, needle, needle_length);
        if (result) {
            *match_length = needle_length;
            return result;
        }
        *match_length = 0;
        return SZ_NULL_CHAR;
    }

    // For short needles (up to 12 bytes which can fold to at most ~6 runes), try hash-free search.
    // We fold the needle first and dispatch based on the folded rune count.
    // This avoids ring buffer setup, hash multiplier computation, and rolling hash updates.
    if (needle_length <= 12) {
        sz_rune_t folded_runes[4]; // 4th slot accessed before loop exit
        sz_size_t folded_count = 0;
        sz_utf8_folded_iter_t_ iter;
        sz_utf8_folded_iter_init_(&iter, needle, needle_length);
        sz_rune_t rune;
        while (folded_count < 4 && sz_utf8_folded_iter_next_(&iter, &rune)) folded_runes[folded_count++] = rune;

        // Dispatch based on folded rune count
        switch (folded_count) {
        case 1:
            return sz_utf8_case_insensitive_find_1folded_serial_( //
                haystack, haystack_length,                        //
                folded_runes[0], match_length);
        case 2:
            return sz_utf8_case_insensitive_find_2folded_serial_( //
                haystack, haystack_length,                        //
                folded_runes[0], folded_runes[1], match_length);
        case 3:
            return sz_utf8_case_insensitive_find_3folded_serial_( //
                haystack, haystack_length,                        //
                folded_runes[0], folded_runes[1], folded_runes[2], match_length);
        default: break; // 4+ folded runes: fall through to Rabin-Karp
        }
    }

    sz_size_t const ring_capacity = 32;
    sz_rune_t needle_runes[32];
    sz_size_t needle_prefix_count = 0, needle_total_count = 0;
    sz_u64_t needle_hash = 0;
    {
        sz_utf8_folded_iter_t_ needle_iter;
        sz_utf8_folded_iter_init_(&needle_iter, needle, needle_length);
        sz_rune_t rune;
        while (needle_prefix_count < ring_capacity && sz_utf8_folded_iter_next_(&needle_iter, &rune)) {
            needle_runes[needle_prefix_count++] = rune;
            needle_hash = needle_hash * 257 + rune;
        }
        needle_total_count = needle_prefix_count;
        // For long needles, count remaining runes beyond ring buffer
        while (sz_utf8_folded_iter_next_(&needle_iter, &rune)) needle_total_count++;
    }
    if (!needle_prefix_count) {
        *match_length = 0;
        return SZ_NULL_CHAR;
    }

    sz_u64_t hash_multiplier = 1;
    for (sz_size_t i = 1; i < needle_prefix_count; ++i) hash_multiplier *= 257;

    sz_rune_t window_runes[32];
    sz_cptr_t window_sources[32];     // Byte position of character that produced each window rune
    sz_size_t window_skip_counts[32]; // Runes to skip from first character's expansion
    sz_size_t ring_head = 0;
    sz_u64_t window_hash = 0;
    sz_utf8_folded_iter_t_ haystack_iter;
    sz_utf8_folded_iter_init_(&haystack_iter, haystack, haystack_length);

    sz_cptr_t window_start = haystack;
    sz_cptr_t current_source = haystack;
    sz_size_t current_skip = 0;
    sz_size_t window_count = 0;

    while (window_count < needle_prefix_count) {
        sz_cptr_t before_ptr = haystack_iter.ptr;
        sz_rune_t rune;
        if (!sz_utf8_folded_iter_next_(&haystack_iter, &rune)) break;
        window_runes[window_count] = rune;
        // Update source and skip only when starting a new character (not mid-expansion)
        if (haystack_iter.pending_idx <= 1 || haystack_iter.pending_count == 0) {
            current_source = before_ptr;
            current_skip = 0;
        }
        window_sources[window_count] = current_source;
        window_skip_counts[window_count] = current_skip;
        window_hash = window_hash * 257 + rune;
        window_count++;
        // For next rune from same expansion, increment skip
        if (haystack_iter.pending_idx > 0 && haystack_iter.pending_idx < haystack_iter.pending_count)
            current_skip = haystack_iter.pending_idx;
    }
    if (window_count < needle_prefix_count) {
        *match_length = 0;
        return SZ_NULL_CHAR;
    }
    sz_cptr_t window_end = haystack_iter.ptr;

    for (;;) {
        if (window_hash == needle_hash) {
            // The ring buffer is a circular array where `ring_head` points to the oldest (first) element.
            // A naive approach would use `window_runes[(ring_head + i) % needle_prefix_count]` for each comparison,
            // but modulo is expensive. Instead, we compare in two contiguous segments:
            //   - First segment:  window_runes[ring_head..needle_prefix_count) maps to needle_runes[0..first_segment)
            //   - Second segment: window_runes[0..ring_head) maps to needle_runes[first_segment..needle_prefix_count)
            sz_size_t first_segment = needle_prefix_count - ring_head;
            sz_size_t mismatches = 0;
            for (sz_size_t i = 0; i < first_segment; ++i) mismatches += window_runes[ring_head + i] != needle_runes[i];
            for (sz_size_t i = 0; i < ring_head; ++i) mismatches += window_runes[i] != needle_runes[first_segment + i];

            if (!mismatches) {
                sz_size_t skip_runes = window_skip_counts[ring_head];
                // Short needle: rune comparison above is sufficient verification
                if (needle_total_count <= ring_capacity) {
                    *match_length = (sz_size_t)(window_end - window_start);
                    return window_start;
                }
                // Long needle: verify FULL needle from window_start, skipping runes if match
                // starts mid-expansion. Example: ẚ→"aʾ", needle starting with "ʾ" must skip "a".
                sz_utf8_folded_iter_t_ verify_haystack_iter;
                sz_utf8_folded_iter_init_(&verify_haystack_iter, window_start,
                                          (sz_size_t)(haystack + haystack_length - window_start));
                // Skip runes within first character's expansion
                sz_rune_t skip_rune;
                for (sz_size_t i = 0; i < skip_runes; ++i) sz_utf8_folded_iter_next_(&verify_haystack_iter, &skip_rune);
                // Now verify full needle against remaining haystack
                sz_utf8_folded_iter_t_ verify_needle_iter;
                sz_utf8_folded_iter_init_(&verify_needle_iter, needle, needle_length);
                sz_rune_t needle_rune_v, haystack_rune_v;
                sz_bool_t match_ok = sz_true_k;
                while (sz_utf8_folded_iter_next_(&verify_needle_iter, &needle_rune_v)) {
                    if (!sz_utf8_folded_iter_next_(&verify_haystack_iter, &haystack_rune_v) ||
                        needle_rune_v != haystack_rune_v) {
                        match_ok = sz_false_k;
                        break;
                    }
                }
                if (match_ok) {
                    *match_length = (sz_size_t)(verify_haystack_iter.ptr - window_start);
                    return window_start;
                }
            }
        }

        sz_cptr_t before_ptr = haystack_iter.ptr;
        sz_rune_t new_rune;
        if (!sz_utf8_folded_iter_next_(&haystack_iter, &new_rune)) break;

        window_hash -= window_runes[ring_head] * hash_multiplier;
        window_hash = window_hash * 257 + new_rune;

        // Advance ring head, avoiding modulo with a conditional (cheaper than integer division)
        sz_size_t next_head = ring_head + 1;
        next_head = next_head == needle_prefix_count ? 0 : next_head;

        window_runes[ring_head] = new_rune;
        // Update source and skip only when starting a new character (not mid-expansion)
        if (haystack_iter.pending_idx <= 1 || haystack_iter.pending_count == 0) {
            current_source = before_ptr;
            current_skip = 0;
        }
        window_sources[ring_head] = current_source;
        window_skip_counts[ring_head] = current_skip;
        // For next rune from same expansion, increment skip
        if (haystack_iter.pending_idx > 0 && haystack_iter.pending_idx < haystack_iter.pending_count)
            current_skip = haystack_iter.pending_idx;

        ring_head = next_head;
        window_start = window_sources[ring_head];
        window_end = haystack_iter.ptr;
    }

    *match_length = 0;
    return SZ_NULL_CHAR;
}

/**
 *  @brief Helper function performing case-folding under the constraint, that no output may be incomplete.
 *
 *  @param[in] source Pointer to the source UTF-8 data, must be valid UTF-8.
 *  @param[in] source_length Length of the source data in bytes.
 *  @param[out] destination Pointer to the destination buffer.
 *  @param[in] destination_length Length of the destination buffer in bytes.
 *  @param[out] codepoints_consumed Number of codepoints read from source.
 *  @param[out] codepoints_exported Number of codepoints written to destination.
 *  @param[out] bytes_consumed Number of bytes read from source.
 *  @param[out] bytes_exported Number of bytes written to destination.
 */
SZ_INTERNAL void sz_utf8_case_fold_upto_(                           //
    sz_cptr_t source, sz_size_t source_length,                      //
    sz_ptr_t destination, sz_size_t destination_length,             //
    sz_size_t *codepoints_consumed, sz_size_t *codepoints_exported, //
    sz_size_t *bytes_consumed, sz_size_t *bytes_exported) {

    sz_u8_t const *const source_start = (sz_u8_t const *)source;
    sz_u8_t const *const source_limit = source_start + source_length;
    sz_u8_t *const destination_start = (sz_u8_t *)destination;
    sz_u8_t *const destination_limit = destination_start + destination_length;

    sz_u8_t const *source_ptr = source_start;
    sz_u8_t *destination_ptr = destination_start;
    sz_size_t codepoints_read = 0;
    sz_size_t codepoints_written = 0;

    while (source_ptr < source_limit && destination_ptr < destination_limit) {

        // Fast path for ASCII optimization
        while (source_ptr < source_limit && destination_ptr < destination_limit && *source_ptr < 0x80) {
            *destination_ptr++ = sz_ascii_fold_(*source_ptr++);
            codepoints_read++;
            codepoints_written++;
        }

        // Check if we hit boundaries
        if (source_ptr >= source_limit || destination_ptr >= destination_limit) break;

        // Multi-byte UTF-8 sequence
        sz_rune_t source_rune;
        sz_rune_length_t source_rune_length;
        sz_rune_parse((sz_cptr_t)source_ptr, &source_rune, &source_rune_length);

        // Perform Unicode folding
        sz_rune_t target_runes[3];
        sz_size_t target_runes_count = sz_unicode_fold_codepoint_(source_rune, target_runes);

        // In the worst case scenario, when folded, the text becomes 3x longer.
        // That's the story of 'ΐ' (U+0390, CE 90) becoming three codepoints (U+03B9 U+0308 U+0301, CE B9 CC 88 CC 81).
        sz_u8_t target_bytes[12]; // 3 runes max, each up to 4 bytes
        sz_size_t target_bytes_count = 0;
        for (sz_size_t i = 0; i < target_runes_count; ++i)
            target_bytes_count += sz_rune_export(target_runes[i], target_bytes + target_bytes_count);

        if (destination_ptr + target_bytes_count > destination_limit) break;
        for (sz_size_t i = 0; i < target_bytes_count; ++i) *destination_ptr++ = target_bytes[i];

        source_ptr += source_rune_length;
        codepoints_read++;
        codepoints_written += target_runes_count;
    }

    if (codepoints_consumed) *codepoints_consumed = codepoints_read;
    if (codepoints_exported) *codepoints_exported = codepoints_written;
    if (bytes_consumed) *bytes_consumed = (sz_size_t)(source_ptr - source_start);
    if (bytes_exported) *bytes_exported = (sz_size_t)(destination_ptr - destination_start);
}

SZ_PUBLIC sz_size_t sz_utf8_case_fold_serial(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {

    sz_u8_t const *source_ptr = (sz_u8_t const *)source;
    sz_u8_t const *source_end = source_ptr + source_length;
    sz_u8_t *destination_ptr = (sz_u8_t *)destination;

    // Assumes valid UTF-8 input; use sz_utf8_valid() first if validation is needed.
    while (source_ptr < source_end) {
        // ASCII fast-path: process consecutive ASCII bytes without UTF-8 decode/encode overhead.
        // This handles ~95% of typical text content with minimal branching.
        while (source_ptr < source_end && *source_ptr < 0x80) *destination_ptr++ = sz_ascii_fold_(*source_ptr++);
        if (source_ptr >= source_end) break;

        // Multi-byte UTF-8 sequence: use full decode/fold/encode path
        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse((sz_cptr_t)source_ptr, &rune, &rune_length);
        source_ptr += rune_length;

        sz_rune_t folded_runes[3]; // Unicode case folding produces at most 3 runes
        sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
        for (sz_size_t i = 0; i != folded_count; ++i)
            destination_ptr += sz_rune_export(folded_runes[i], destination_ptr);
    }

    return (sz_size_t)(destination_ptr - (sz_u8_t *)destination);
}

SZ_PUBLIC sz_size_t sz_utf8_case_upper_serial(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {

    sz_u8_t const *source_ptr = (sz_u8_t const *)source;
    sz_u8_t const *source_end = source_ptr + source_length;
    sz_u8_t *destination_ptr = (sz_u8_t *)destination;

    // Assumes valid UTF-8 input; use sz_utf8_valid() first if validation is needed.
    while (source_ptr < source_end) {
        // ASCII fast-path: process consecutive ASCII bytes without UTF-8 decode/encode overhead.
        // This handles ~95% of typical text content with minimal branching.
        while (source_ptr < source_end && *source_ptr < 0x80) *destination_ptr++ = sz_ascii_upper_(*source_ptr++);
        if (source_ptr >= source_end) break;

        // Multi-byte UTF-8 sequence: use full decode/upper/encode path
        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse((sz_cptr_t)source_ptr, &rune, &rune_length);
        source_ptr += rune_length;

        sz_rune_t upper_runes[3]; // Unicode uppercasing produces at most 3 runes (e.g., ﬃ → FFI)
        sz_size_t upper_count = sz_unicode_upper_codepoint_(rune, upper_runes);
        for (sz_size_t i = 0; i != upper_count; ++i)
            destination_ptr += sz_rune_export(upper_runes[i], destination_ptr);
    }

    return (sz_size_t)(destination_ptr - (sz_u8_t *)destination);
}

SZ_PUBLIC sz_ordering_t sz_utf8_case_insensitive_order_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                              sz_size_t b_length) {
    sz_utf8_folded_iter_t_ a_iterator, b_iterator;
    sz_utf8_folded_iter_init_(&a_iterator, a, a_length);
    sz_utf8_folded_iter_init_(&b_iterator, b, b_length);

    sz_rune_t a_rune, b_rune;
    for (;;) {
        sz_bool_t pulled_from_a = sz_utf8_folded_iter_next_(&a_iterator, &a_rune);
        sz_bool_t pulled_from_b = sz_utf8_folded_iter_next_(&b_iterator, &b_rune);

        if (!pulled_from_a && !pulled_from_b) return sz_equal_k;
        if (!pulled_from_a) return sz_less_k;
        if (!pulled_from_b) return sz_greater_k;
        if (a_rune != b_rune) return sz_order_scalars_(a_rune, b_rune);
    }
}

#pragma endregion // Serial Implementation

#pragma region Ice Lake Implementation
#if SZ_USE_ICE
#if defined(__clang__)
#pragma clang attribute push(                                                                          \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2")
#endif

/**
 *  Helper macros to reduce code duplication in fast-paths.
 *  Detect ASCII uppercase A-Z: returns mask where bytes are in range 0x41-0x5A.
 */
#define sz_ice_is_ascii_upper_(src_zmm) _mm512_cmplt_epu8_mask(_mm512_sub_epi8((src_zmm), a_upper_vec), subtract26_vec)

/** Apply ASCII case folding (+0x20) to masked positions. */
#define sz_ice_fold_ascii_(src_zmm, upper_mask, apply_mask) \
    _mm512_mask_add_epi8((src_zmm), (upper_mask) & (apply_mask), (src_zmm), ascii_case_offset)

/** Fold ASCII A-Z in source vector within a prefix mask, returning folded result. */
#define sz_ice_fold_ascii_in_prefix_(src_zmm, prefix_mask) \
    _mm512_mask_add_epi8((src_zmm), sz_ice_is_ascii_upper_(src_zmm) & (prefix_mask), (src_zmm), ascii_case_offset)

/**
 *  Georgian uppercase transformation: E1 82/83 XX → E2 B4 YY.
 *  Applies to lead byte positions (sets E2), second byte positions (sets B4),
 *  and adjusts third bytes (-0x20 for 82 sequences, +0x20 for 83 sequences).
 */
#define sz_ice_transform_georgian_(folded, georgian_leads, is_82_upper, is_83_upper, prefix_mask)              \
    do {                                                                                                       \
        (folded) = _mm512_mask_blend_epi8((georgian_leads), (folded), _mm512_set1_epi8((char)0xE2));           \
        (folded) = _mm512_mask_blend_epi8((georgian_leads) << 1, (folded), _mm512_set1_epi8((char)0xB4));      \
        (folded) = _mm512_mask_sub_epi8((folded), (is_82_upper) & (prefix_mask), (folded), ascii_case_offset); \
        (folded) = _mm512_mask_add_epi8((folded), (is_83_upper) & (prefix_mask), (folded), ascii_case_offset); \
    } while (0)

/**
 *  @brief  Find the first invalid position within load_mask, returning chunk_size if all valid.
 *
 *  This safely handles the edge case where all 64 bytes are valid (i.e., ctz(0) which is undefined).
 *  The mask `~is_valid | ~load_mask` marks positions that are either invalid OR outside the loaded chunk.
 *  When all loaded bytes are valid, this mask becomes zero, so we return chunk_size instead of calling ctz(0).
 */
SZ_INTERNAL sz_size_t sz_ice_first_invalid_(sz_u64_t is_valid, sz_u64_t load_mask, sz_size_t chunk_size) {
    sz_u64_t invalid_mask = ~is_valid | ~load_mask;
    return invalid_mask ? (sz_size_t)sz_u64_ctz(invalid_mask) : chunk_size;
}

#if SZ_DEBUG

#if !SZ_AVOID_LIBC
SZ_INTERNAL void sz_utf8_case_debug_dump_bytes_hex_(FILE *stream, sz_cptr_t bytes, sz_size_t length) {
    for (sz_size_t i = 0; i < length; ++i) fprintf(stream, "%02X ", (unsigned char)bytes[i]);
}

SZ_INTERNAL void sz_utf8_case_debug_dump_bytes_c_array_(FILE *stream, char const *name, sz_cptr_t bytes,
                                                        sz_size_t length) {
    fprintf(stream, "static unsigned char %s[%zu] = {", name, (size_t)length);
    for (sz_size_t i = 0; i < length; ++i) {
        if ((i % 16) == 0) fprintf(stream, "\n    ");
        fprintf(stream, "0x%02X,", (unsigned char)bytes[i]);
    }
    fprintf(stream, "\n};\n");
}
#endif

/**
 *  @brief  Verifies the SIMD result against the serial implementation.
 *          If they differ, dumps diagnostic info and crashes to help debugging.
 */
SZ_INTERNAL void sz_utf8_case_insensitive_find_assert_and_crash_( //
    sz_cptr_t result, sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length,
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata, char const *file, int line) {

    sz_size_t serial_matched_length;
    sz_cptr_t expected = sz_utf8_case_insensitive_find_serial(haystack, haystack_length, needle, needle_length, 0,
                                                              &serial_matched_length);

    if (result == expected) return;

#if !SZ_AVOID_LIBC
    // Assumes <stdio.h> and <stdlib.h> are included by the user/test suite
    fprintf(stderr, "--------------------------------------------------------\n");
    fprintf(stderr, "SIMD Mismatch at %s:%d\n", file, line);
    fprintf(stderr, "Haystack Length: %zu, Needle Length: %zu\n", haystack_length, needle_length);
    fprintf(stderr, "Expected: %p, Found: %p\n", (void *)expected, (void *)result);

    if (expected) fprintf(stderr, "Expected Offset: %zu\n", (size_t)(expected - haystack));
    else
        fprintf(stderr, "Expected: NULL (Not Found)\n");

    if (result) fprintf(stderr, "Found Offset: %zu\n", (size_t)(result - haystack));
    else
        fprintf(stderr, "Found: NULL (Not Found)\n");

    // Print SIMD metadata for debugging
    if (needle_metadata) {
        fprintf(stderr, "SIMD Metadata: kernel_id=%u, offset_in_unfolded=%zu, length_in_unfolded=%zu\n",
                needle_metadata->kernel_id, needle_metadata->offset_in_unfolded, needle_metadata->length_in_unfolded);
        fprintf(stderr, "SIMD Metadata: folded_slice_length=%u, probe_second=%u, probe_third=%u\n",
                needle_metadata->folded_slice_length, needle_metadata->probe_second, needle_metadata->probe_third);
        fprintf(stderr, "SIMD Metadata folded_slice: ");
        for (sz_size_t i = 0; i < needle_metadata->folded_slice_length && i < 16; ++i)
            fprintf(stderr, "%02X ", needle_metadata->folded_slice[i]);
        fprintf(stderr, "\n");
    }

    // Print full needle and haystack to enable copy-pastable deterministic repros.
    fprintf(stderr, "Needle (Hex): ");
    sz_utf8_case_debug_dump_bytes_hex_(stderr, needle, needle_length);
    fprintf(stderr, "\n");
    fprintf(stderr, "Haystack (Hex): ");
    sz_utf8_case_debug_dump_bytes_hex_(stderr, haystack, haystack_length);
    fprintf(stderr, "\n");

    fprintf(stderr, "Needle (C Array):\n");
    sz_utf8_case_debug_dump_bytes_c_array_(stderr, "needle_bytes", needle, needle_length);
    fprintf(stderr, "Haystack (C Array):\n");
    sz_utf8_case_debug_dump_bytes_c_array_(stderr, "haystack_bytes", haystack, haystack_length);

    // Print Haystack Context (if match found or missed)
    size_t context_padding = 16;
    sz_cptr_t focus = expected ? expected : (result ? result : haystack);
    sz_cptr_t start = (focus - haystack > (sz_ssize_t)context_padding) ? focus - context_padding : haystack;
    sz_cptr_t end = (focus + needle_length + context_padding < haystack + haystack_length)
                        ? focus + needle_length + context_padding
                        : haystack + haystack_length;

    fprintf(stderr, "Haystack Context (Hex): ");
    for (sz_cptr_t p = start; p < end; ++p) {
        if (p == expected) fprintf(stderr, "[EXP] ");
        if (p == result) fprintf(stderr, "[ACT] ");
        fprintf(stderr, "%02X ", (unsigned char)*p);
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "--------------------------------------------------------\n");
    abort();
#else
    sz_unused_(needle_metadata);
    sz_unused_(haystack);
    sz_unused_(haystack_length);
    sz_unused_(needle);
    sz_unused_(needle_length);
    sz_unused_(result);
    sz_unused_(file);
    sz_unused_(line);
    // Force crash without LibC
    *((volatile int *)0) = 0;
#endif
}

#define sz_utf8_case_insensitive_find_assert_(result, haystack, haystack_length, needle, needle_length,       \
                                              needle_metadata)                                                \
    sz_utf8_case_insensitive_find_assert_and_crash_(result, haystack, haystack_length, needle, needle_length, \
                                                    needle_metadata, __FILE__, __LINE__)

#else
#define sz_utf8_case_insensitive_find_assert_(result, haystack, haystack_length, needle, needle_length, needle_metadata)
#endif

SZ_PUBLIC sz_size_t sz_utf8_case_fold_ice(sz_cptr_t source, sz_size_t source_length, sz_ptr_t target) {
    // This algorithm exploits the idea, that most text in a single ZMM register is either:
    //
    // 1. All ASCII single-byte codepoints
    //    Fast-path: detect A-Z (0x41-0x5A) and add 0x20 to convert to lowercase.
    //
    // 2. Mixture of 2-byte codepoints in one language (continuous range) and 1-byte ASCII codepoints
    //
    //    2.1. Latin-1 Supplement (C3 80-BF): À-ß and à-ÿ.
    //         Folding is trivial +32 to second byte for 80-9E (except × at 0x97).
    //         Special case: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73) - handled via mask blend.
    //
    //    2.2. Basic Cyrillic (D0/D1): Ѐ-џ and А-я (U+0400-U+045F).
    //         Folding maps uppercase → lowercase:
    //         - D0 80-8F (Ѐ-Џ) → D1 90-9F (ѐ-џ): +0x10, D0→D1
    //         - D0 90-9F (А-П) → D0 B0-BF (а-п): +0x20
    //         - D0 A0-AF (Р-Я) → D1 80-8F (р-я): −0x20, D0→D1
    //         Excludes Extended-A (D1 A0+) which uses parity (+1) folding.
    //
    //    2.3. Greek (CE/CF): Basic Greek letters Α-Ω and α-ω.
    //         Uppercase is in CE, but lowercase spans both CE and CF:
    //         - CE 91-9F (Α-Ο) → CE B1-BF (α-ο) (+0x20)
    //         - CE A0-AB (Π-Ϋ, skipping A2) → CF 80-8B (π-ϋ) (CE→CF and −0x20)
    //         - CF 82 ('ς' U+03C2) → CF 83 ('σ' U+03C3) (+1)
    //         Excludes archaic letters, tonos/diacritics, and symbol variants needing special handling.
    //
    // 3. Groups of 3-byte codepoints - split into caseless and case-aware paths
    //
    //    3.1. Caseless 3-byte content: CJK (E4-E9), Hangul (EA with safe seconds),
    //         most punctuation (E2 80-83), Thai, Hindi, etc.
    //         Fast-path: copy directly without transformation.
    //
    //    3.2. Georgian uppercase (E1 82 A0-BF, E1 83 80-85/87/8D) → lowercase (E2 B4 80-AF).
    //         Full 3-byte transformation: lead E1→E2, second byte→B4,
    //         third byte ±0x20 depending on original second byte (82 vs 83).
    //
    //    3.3. Fullwidth Latin:
    //         - Ａ-Ｚ: 'Ａ' (U+FF21, EF BC A1) → 'ａ' (U+FF41, EF BD 81) (third byte +0x20, second byte BC→BD)
    //
    // 4. Groups of 4-byte codepoints (emoji, historic scripts)
    //    Generally caseless, but Deseret/Warang Citi have folding rules.
    //    Falls back to serial code for rare case-aware sequences.
    //
    // Within these assumptions, this kernel vectorizes case folding for simple cases, when
    // a sequence of bytes in a certain range can be folded by a single constant addition/subtraction.
    // For more complex cases it periodically switches to serial code.
    //
    // ┌─────────────────────────────────────────────────────────────────────────────┐
    // │ UTF-8 Encoding Quick Reference                                              │
    // ├───────┬────────────────┬─────────────┬──────────────────────────────────────┤
    // │ Bytes │ Range          │ Lead Byte   │ Pattern                              │
    // ├───────┼────────────────┼─────────────┼──────────────────────────────────────┤
    // │  1    │ U+0000-007F    │ 0xxxxxxx    │ ASCII                                │
    // │  2    │ U+0080-07FF    │ 110xxxxx    │ C2-DF lead + 1 continuation          │
    // │  3    │ U+0800-FFFF    │ 1110xxxx    │ E0-EF lead + 2 continuations         │
    // │  4    │ U+10000-10FFFF │ 11110xxx    │ F0-F4 lead + 3 continuations         │
    // ├───────┴────────────────┴─────────────┴──────────────────────────────────────┤
    // │ Continuation bytes: 10xxxxxx (0x80-0xBF)                                    │
    // │ Lead byte detection: (byte & 0xC0) == 0x80 → continuation                   │
    // │                      (byte & 0xF0) == 0xE0 → 3-byte lead                     │
    // │                      (byte & 0xF8) == 0xF0 → 4-byte lead                     │
    // └─────────────────────────────────────────────────────────────────────────────┘
    //
    sz_u512_vec_t source_vec;
    sz_ptr_t target_start = target;

    // Pre-compute constants used in multiple places
    __m512i const indices_vec = _mm512_set_epi8(                        //
        63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, //
        47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, //
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, //
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m512i const a_upper_vec = _mm512_set1_epi8('A');
    __m512i const subtract26_vec = _mm512_set1_epi8(26);
    __m512i const ascii_case_offset = _mm512_set1_epi8(0x20); // Difference between 'A' and 'a'

    // UTF-8 lead byte detection constants:
    // Continuation bytes match: (byte & 0xC0) == 0x80  → pattern 10xxxxxx
    // 3-byte leads match:       (byte & 0xF0) == 0xE0  → pattern 1110xxxx
    // 4-byte leads match:       (byte & 0xF8) == 0xF0  → pattern 11110xxx
    __m512i const utf8_cont_test_mask = _mm512_set1_epi8((char)0xC0);
    __m512i const utf8_cont_pattern = _mm512_set1_epi8((char)0x80);
    __m512i const utf8_3byte_test_mask = _mm512_set1_epi8((char)0xF0);
    __m512i const utf8_3byte_pattern = _mm512_set1_epi8((char)0xE0);
    __m512i const utf8_4byte_test_mask = _mm512_set1_epi8((char)0xF8);

    while (source_length) {
        // Prefetch ahead to hide memory latency on large datasets.
        // This helps when processing multi-GB files that don't fit in cache.
        _mm_prefetch(source + 1024, _MM_HINT_T1); // To L2: 16 cache lines ahead
        _mm_prefetch(source + 512, _MM_HINT_T0);  // To L1: 8 cache lines ahead
        _mm_prefetch(source + 576, _MM_HINT_T0);  // To L1: 9 cache lines ahead
        _mm_prefetch(source + 640, _MM_HINT_T0);  // To L1: 10 cache lines ahead

        sz_size_t chunk_size = sz_min_of_two(source_length, 64);
        __mmask64 load_mask = sz_u64_mask_until_(chunk_size);
        source_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, source);
        __mmask64 is_non_ascii = _mm512_movepi8_mask(source_vec.zmm);

        // FAST PATH: Check for pure ASCII FIRST, before computing any masks.
        // This is the most common case for English and many other Latin-script texts.
        // Avoids computing 6+ masks that would be wasted on pure ASCII chunks.
        if (is_non_ascii == 0) {
            _mm512_mask_storeu_epi8(target, load_mask, sz_ice_fold_ascii_in_prefix_(source_vec.zmm, load_mask));
            target += chunk_size, source += chunk_size, source_length -= chunk_size;
            continue;
        }

        // Compute lead byte masks only for non-ASCII chunks.
        //
        // Optimization: Use VPTERNLOGD (imm8=0x80 = A & B) instead of VPAND.
        // - VPAND executes on p0/p5 only (2 ports)
        // - VPTERNLOGD executes on p0/p1/p5 (3 ports)
        // - VPCMPEQ-to-mask is p5-only (bottleneck)
        // By moving AND operations to VPTERNLOGD, we reduce p5 contention, allowing
        // the subsequent CMPEQ operations to execute with less stalling.
        //
        // Original pattern:
        //   _mm512_cmpeq_epi8_mask(_mm512_and_si512(src, mask), pattern)
        // Optimized pattern:
        //   _mm512_cmpeq_epi8_mask(_mm512_ternarylogic_epi64(src, mask, mask, 0x80), pattern)
        //
        __m512i masked_cont = _mm512_ternarylogic_epi64(source_vec.zmm, utf8_cont_test_mask, utf8_cont_test_mask, 0x80);
        __m512i masked_3byte =
            _mm512_ternarylogic_epi64(source_vec.zmm, utf8_3byte_test_mask, utf8_3byte_test_mask, 0x80);
        __m512i masked_4byte =
            _mm512_ternarylogic_epi64(source_vec.zmm, utf8_4byte_test_mask, utf8_4byte_test_mask, 0x80);
        __mmask64 is_cont_mask = _mm512_cmpeq_epi8_mask(masked_cont, utf8_cont_pattern);
        __mmask64 is_three_byte_lead_mask = _mm512_cmpeq_epi8_mask(masked_3byte, utf8_3byte_pattern);
        __mmask64 is_four_byte_lead_mask = _mm512_cmpeq_epi8_mask(masked_4byte, utf8_3byte_test_mask);

        // Early fast path: Pure 3-byte content (no ASCII, no 2-byte, no 4-byte)
        // This is common for CJK, Hindi (Devanagari), Thai, etc.
        // Check if all non-ASCII bytes are 3-byte leads or continuations
        {
            __mmask64 is_valid_pure_3byte_mask = is_three_byte_lead_mask | is_cont_mask;
            // Quick check: if all bits match, we have pure 3-byte content
            if ((is_valid_pure_3byte_mask & load_mask) == (is_non_ascii & load_mask) && !is_four_byte_lead_mask) {
                // Check for problematic leads that have case folding:
                //   - E1: Georgian, Greek Extended, Latin Extended Additional
                //   - E2: Glagolitic (B0-B1), Coptic (B2-B3), Letterlike (84 = Kelvin/Angstrom)
                //   - EF: Fullwidth A-Z
                // E2 80-83 are safe (General Punctuation quotes); other E2 ranges have folding
                // EA is mostly safe (Hangul B0-BF) but some second bytes have folding:
                //   - 0x99-0x9F: Cyrillic Ext-B, Latin Ext-D (A640-A7FF)
                //   - 0xAD-0xAE: Cherokee Supplement (AB70-ABBF)
                // Optimization: Range compression for E1/E2 detection.
                // E1 (0xE1) and E2 (0xE2) are adjacent, so instead of 2 CMPEQ operations:
                //   is_e1_mask = CMPEQ(src, E1)  // p5-only
                //   is_e2_mask = CMPEQ(src, E2)  // p5-only
                // We use 1 range check + 1 CMPEQ:
                //   is_e1_e2_mask = (src - E1) < 2  // SUB (any port) + CMPLT (p5)
                //   is_e1_mask = CMPEQ(src, E1)     // p5 (still needed for final check)
                //   is_e2_mask = is_e1_e2_mask & ~is_e1_mask  // KANDN (k-unit, not p5)
                // Net effect: saves 1 p5 operation and enables early-out on combined mask.
                //
                // Original pattern:
                //   is_e1_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE1));
                //   is_e2_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE2));
                //
                __mmask64 is_e1_e2_mask = _mm512_cmplt_epu8_mask(
                    _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xE1)), _mm512_set1_epi8(2));
                __mmask64 is_e1_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE1));
                __mmask64 is_ef_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xEF));
                // For E2, only allow 80-83 (General Punctuation quotes) through - many other E2 ranges have folding
                // (84 Letterlike, 93 Enclosed Alphanumerics, B0-B3 Glagolitic/Coptic, etc.)
                __mmask64 is_e2_mask = _kand_mask64(is_e1_e2_mask, _knot_mask64(is_e1_mask));
                __mmask64 e2_second_byte_positions = is_e2_mask << 1;
                // E2 folding needed if second byte is NOT in 80-83 range
                __mmask64 is_e2_folding_mask =
                    e2_second_byte_positions &
                    ~_mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x80)),
                                            _mm512_set1_epi8(0x04)); // NOT 80-83
                // For EA, check if second byte is in problematic ranges
                __mmask64 is_ea_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xEA));
                __mmask64 ea_second_byte_positions = is_ea_mask << 1;
                // EA 99-9F (second byte 0x99-0x9F) or EA AD-AE (second byte 0xAD-0xAE)
                __mmask64 is_ea_folding_mask =
                    ea_second_byte_positions &
                    (_mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x99)),
                                            _mm512_set1_epi8(0x07)) | // 0x99-0x9F
                     _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xAD)),
                                            _mm512_set1_epi8(0x02))); // 0xAD-0xAE
                if (!(is_e1_mask | is_e2_folding_mask | is_ea_folding_mask | is_ef_mask)) {
                    // Safe 3-byte content (E0, E3-E9, EB-EE) - no 3-byte case folding needed
                    // But ASCII mixed in still needs folding! Use sz_ice_fold_ascii_in_prefix_.
                    // Just need to avoid splitting a 3-byte sequence at the end.
                    sz_size_t copy_len = chunk_size;
                    // Check if last 1-2 bytes are an incomplete sequence
                    __mmask64 leads_in_chunk_mask = is_three_byte_lead_mask & load_mask;
                    if (leads_in_chunk_mask) {
                        int last_lead_pos = 63 - sz_u64_clz(leads_in_chunk_mask);
                        if (last_lead_pos + 3 > (int)copy_len) copy_len = last_lead_pos;
                    }
                    if (copy_len > 0) {
                        __mmask64 copy_mask = sz_u64_mask_until_(copy_len);
                        _mm512_mask_storeu_epi8(target, copy_mask,
                                                sz_ice_fold_ascii_in_prefix_(source_vec.zmm, copy_mask));
                        target += copy_len, source += copy_len, source_length -= copy_len;
                        continue;
                    }
                }
            }
        }

        // 2. Two-byte UTF-8 sequences (valid lead bytes C2-DF)
        //
        // 2.1. Latin-1 Supplement (C3 80 - C3 BF) mixed with ASCII
        __mmask64 is_latin1_lead = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xC3));
        __mmask64 latin1_second_byte_positions = is_latin1_lead << 1;
        __mmask64 is_valid_latin1_mix = ~is_non_ascii | is_latin1_lead | latin1_second_byte_positions;
        sz_size_t latin1_length = sz_ice_first_invalid_(is_valid_latin1_mix, load_mask, chunk_size);
        latin1_length -= latin1_length && ((is_latin1_lead >> (latin1_length - 1)) & 1); // Don't split 2-byte seq

        if (latin1_length >= 2) {
            __mmask64 prefix_mask = sz_u64_mask_until_(latin1_length);
            __mmask64 latin1_second_bytes = latin1_second_byte_positions & prefix_mask;

            // ASCII A-Z (0x41-0x5A) and Latin-1 À-Þ (second byte 0x80-0x9E excl. ×=0x97) both get +0x20
            __mmask64 is_upper_ascii = sz_ice_is_ascii_upper_(source_vec.zmm);
            __mmask64 is_latin1_upper = _mm512_mask_cmplt_epu8_mask(
                latin1_second_bytes, _mm512_sub_epi8(source_vec.zmm, utf8_cont_pattern), _mm512_set1_epi8(0x1F));
            is_latin1_upper ^= _mm512_mask_cmpeq_epi8_mask(is_latin1_upper, source_vec.zmm,
                                                           _mm512_set1_epi8((char)0x97)); // Exclude ×
            __m512i folded = sz_ice_fold_ascii_(source_vec.zmm, is_upper_ascii | is_latin1_upper, prefix_mask);

            // 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73): replace both bytes with 's'
            __mmask64 is_eszett_mask =
                _mm512_mask_cmpeq_epi8_mask(latin1_second_bytes, source_vec.zmm, _mm512_set1_epi8((char)0x9F));
            folded = _mm512_mask_set1_epi8(folded, is_eszett_mask | (is_eszett_mask >> 1), 's');

            _mm512_mask_storeu_epi8(target, prefix_mask, folded);
            target += latin1_length, source += latin1_length, source_length -= latin1_length;
            continue;
        }

        // 2.2. Cyrillic fast path (D0/D1 lead bytes for basic Cyrillic 0x0400-0x045F)
        //
        // Cyrillic D0/D1 uppercase → lowercase transformations:
        // ┌──────────────────┬───────────────────────┬─────────────────────┬──────────────┐
        // │ Input Range      │ Codepoints            │ Output              │ Transform    │
        // ├──────────────────┼───────────────────────┼─────────────────────┼──────────────┤
        // │ D0 80-8F         │ Ѐ-Џ (U+0400-040F)     │ D1 90-9F (ѐ-џ)      │ +0x10, D0→D1 │
        // │ D0 90-9F         │ А-П (U+0410-041F)     │ D0 B0-BF (а-п)      │ +0x20        │
        // │ D0 A0-AF         │ Р-Я (U+0420-042F)     │ D1 80-8F (р-я)      │ −0x20, D0→D1 │
        // │ D0 B0-BF         │ а-п (lowercase)       │ unchanged           │ —            │
        // │ D1 80-9F         │ р-џ (lowercase)       │ unchanged           │ —            │
        // │ D1 A0+           │ Extended-A (U+0460+)  │ → serial path       │ +1 parity    │
        // └──────────────────┴───────────────────────┴─────────────────────┴──────────────┘
        //
        // EXCLUDED from fast path: Cyrillic Extended-A (0x0460-04FF) which starts at D1 A0.
        // These use +1 folding for even codepoints and must go through the general 2-byte path.
        {
            // Optimization: Range compression for D0/D1 Cyrillic detection.
            // D0 (0xD0) and D1 (0xD1) are adjacent, so instead of 2 CMPEQ + OR:
            //   is_d0_mask = CMPEQ(src, D0)  // p5-only
            //   is_d1_mask = CMPEQ(src, D1)  // p5-only
            //   is_cyrillic_lead_mask = is_d0_mask | is_d1_mask
            // We use 1 range check + 1 CMPEQ:
            //   is_cyrillic_lead_mask = (src - D0) < 2  // SUB (any port) + CMPLT (p5)
            //   is_d0_mask = CMPEQ(src, D0)             // p5 (still needed for is_after_d0_mask)
            //   is_d1_mask = is_cyrillic_lead_mask & ~is_d0_mask  // KANDN (k-unit, not p5)
            // Net effect: saves 1 p5 operation.
            //
            // Original pattern:
            //   is_d0_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xD0));
            //   is_d1_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xD1));
            //   is_cyrillic_lead_mask = is_d0_mask | is_d1_mask;
            //
            __mmask64 is_cyrillic_lead_mask = _mm512_cmplt_epu8_mask(
                _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xD0)), _mm512_set1_epi8(2));
            __mmask64 is_d0_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xD0));
            __mmask64 is_d1_mask = _kand_mask64(is_cyrillic_lead_mask, _knot_mask64(is_d0_mask));
            __mmask64 cyrillic_second_byte_positions = is_cyrillic_lead_mask << 1;

            // Exclude Cyrillic Extended-A: D1 with second byte >= 0xA0 (U+0460+)
            // These have +1 folding rules, not the basic Cyrillic patterns
            __mmask64 is_d1_extended_mask =
                (is_d1_mask << 1) & _mm512_cmpge_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xA0));

            // Check for pure basic Cyrillic + ASCII mix (no extended)
            __mmask64 is_valid_cyrillic_mix_mask =
                ~is_non_ascii | is_cyrillic_lead_mask | cyrillic_second_byte_positions;
            is_valid_cyrillic_mix_mask &= ~is_d1_extended_mask; // Stop at Cyrillic Extended
            sz_size_t cyrillic_length = sz_ice_first_invalid_(is_valid_cyrillic_mix_mask, load_mask, chunk_size);
            cyrillic_length -= cyrillic_length && ((is_cyrillic_lead_mask >> (cyrillic_length - 1)) & 1);

            if (cyrillic_length >= 2) {
                __mmask64 prefix_mask = sz_u64_mask_until_(cyrillic_length);
                __mmask64 is_after_d0_mask = (is_d0_mask << 1) & prefix_mask;

                // Start with source, apply ASCII folding
                __m512i folded = sz_ice_fold_ascii_in_prefix_(source_vec.zmm, prefix_mask);

                // D0 second bytes: apply Cyrillic uppercase folding
                // Range 0x80-0x8F (Ѐ-Џ): second byte += 0x10, lead becomes D1
                // Range 0x90-0x9F (А-П): second byte += 0x20, lead stays D0
                // Range 0xA0-0xAF (Р-Я): second byte -= 0x20, lead becomes D1
                __mmask64 is_d0_upper1_mask =
                    _mm512_mask_cmplt_epu8_mask(is_after_d0_mask, source_vec.zmm, _mm512_set1_epi8((char)0x90));
                __mmask64 is_d0_upper2_mask = _mm512_mask_cmplt_epu8_mask(
                    is_after_d0_mask, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x90)),
                    _mm512_set1_epi8(0x10));
                __mmask64 is_d0_upper3_mask = _mm512_mask_cmplt_epu8_mask(
                    is_after_d0_mask, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xA0)),
                    _mm512_set1_epi8(0x10));

                // Apply transformations
                // Ѐ-Џ (0x80-0x8F): +0x10 to second byte
                folded = _mm512_mask_add_epi8(folded, is_d0_upper1_mask, folded, _mm512_set1_epi8(0x10));
                // А-П (0x90-0x9F): +0x20 to second byte
                folded = _mm512_mask_add_epi8(folded, is_d0_upper2_mask, folded, _mm512_set1_epi8(0x20));
                // Р-Я (0xA0-0xAF): -0x20 to second byte (wraps to 0x80-0x8F)
                folded = _mm512_mask_sub_epi8(folded, is_d0_upper3_mask, folded, _mm512_set1_epi8(0x20));

                // Fix lead bytes: Ѐ-Џ and Р-Я need D0→D1
                __mmask64 needs_d1_mask = ((is_d0_upper1_mask | is_d0_upper3_mask) >> 1) & (is_d0_mask & prefix_mask);
                folded = _mm512_mask_mov_epi8(folded, needs_d1_mask, _mm512_set1_epi8((char)0xD1));

                _mm512_mask_storeu_epi8(target, prefix_mask, folded);
                target += cyrillic_length, source += cyrillic_length, source_length -= cyrillic_length;
                continue;
            }
        }

        // 2.3. Greek fast path (CE/CF lead bytes for basic Greek 0x0370-0x03FF)
        //
        // Greek CE/CF uppercase → lowercase transformations:
        // ┌──────────────────┬───────────────────────┬─────────────────────┬──────────────┐
        // │ Input Range      │ Codepoints            │ Output              │ Transform    │
        // ├──────────────────┼───────────────────────┼─────────────────────┼──────────────┤
        // │ CE 91-9F         │ Α-Ο (U+0391-039F)     │ CE B1-BF (α-ο)      │ +0x20        │
        // │ CE A0-A1         │ Π-Ρ (U+03A0-03A1)     │ CF 80-81 (π-ρ)      │ −0x20, CE→CF │
        // │ CE A3-AB         │ Σ-Ϋ (U+03A3-03AB)     │ CF 83-8B (σ-ϋ)      │ −0x20, CE→CF │
        // │ CE B1-BF         │ α-ο (lowercase)       │ unchanged           │ —            │
        // │ CF 80-8B         │ π-ϋ (lowercase)       │ unchanged           │ —            │
        // │ CF 82            │ ς (U+03C2, final sigma) │ CF 83 (σ U+03C3)   │ +1           │
        // └──────────────────┴───────────────────────┴─────────────────────┴──────────────┘
        //
        // EXCLUDED from fast path: Greek tonos/diacritics (CE 84-90) and extended symbols (CF 8C+).
        // These have irregular folding rules and must go through the general 2-byte path.
        {
            __mmask64 is_ce_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xCE));
            __mmask64 is_cf_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xCF));
            __mmask64 is_greek_lead_mask = is_ce_mask | is_cf_mask;
            __mmask64 greek_second_byte_positions = is_greek_lead_mask << 1;

            // Exclude problematic Greek ranges that need serial handling:
            // - CE 84-90: tonos/diacritics with irregular folding (U+0384-0390)
            // - CE 8C, 8E-8F: Ό, Ύ, Ώ with non-standard offsets
            // - CE B0: ΰ (U+03B0) expands to 3 codepoints
            // - CF 8C+: symbols and extended Greek with irregular folding
            // Check second bytes after CE leads
            __mmask64 is_ce_problematic =
                (is_ce_mask << 1) &
                (_mm512_cmplt_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0x91)) | // < 0x91
                 _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xB0))); // == 0xB0 (ΰ expands)
            // Check second bytes after CF leads (only allow 80-8B for basic lowercase + final sigma)
            __mmask64 is_cf_problematic =
                (is_cf_mask << 1) & _mm512_cmpge_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0x8C));

            // Check for pure basic Greek + ASCII mix (no problematic ranges)
            __mmask64 is_valid_greek_mix_mask = ~is_non_ascii | is_greek_lead_mask | greek_second_byte_positions;
            is_valid_greek_mix_mask &= ~(is_ce_problematic | is_cf_problematic);
            sz_size_t greek_length = sz_ice_first_invalid_(is_valid_greek_mix_mask, load_mask, chunk_size);
            greek_length -= greek_length && ((is_greek_lead_mask >> (greek_length - 1)) & 1);

            if (greek_length >= 2) {
                __mmask64 prefix_mask = sz_u64_mask_until_(greek_length);
                __mmask64 is_after_ce_mask = (is_ce_mask << 1) & prefix_mask;
                __mmask64 is_after_cf_mask = (is_cf_mask << 1) & prefix_mask;

                // Start with source, apply ASCII folding
                __m512i folded = sz_ice_fold_ascii_in_prefix_(source_vec.zmm, prefix_mask);

                // CE second bytes: Range 91-9F gets +0x20, Range A0-A1 and A3-AB gets -0x20 (lead changes)
                // Note: A2 is unassigned (U+03A2) and must be excluded!
                // 91-9F: (second - 0x91) < 0x0F
                __mmask64 is_ce_upper1_mask = _mm512_mask_cmplt_epu8_mask(
                    is_after_ce_mask, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x91)),
                    _mm512_set1_epi8(0x0F));
                // A0-A1: Π-Ρ (2 chars)
                __mmask64 is_ce_upper2a_mask = _mm512_mask_cmplt_epu8_mask(
                    is_after_ce_mask, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xA0)),
                    _mm512_set1_epi8(0x02));
                // A3-AB: Σ-Ϋ (9 chars) - skip A2 which is unassigned
                __mmask64 is_ce_upper2b_mask = _mm512_mask_cmplt_epu8_mask(
                    is_after_ce_mask, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xA3)),
                    _mm512_set1_epi8(0x09));
                __mmask64 is_ce_upper2_mask = is_ce_upper2a_mask | is_ce_upper2b_mask;

                // Apply +0x20 for CE 91-9F
                folded = _mm512_mask_add_epi8(folded, is_ce_upper1_mask, folded, _mm512_set1_epi8(0x20));
                // Apply -0x20 for CE A0-AB
                folded = _mm512_mask_sub_epi8(folded, is_ce_upper2_mask, folded, _mm512_set1_epi8(0x20));

                // Fix lead bytes: CE A0-AB need CE→CF
                __mmask64 needs_cf_mask = (is_ce_upper2_mask >> 1) & (is_ce_mask & prefix_mask);
                folded = _mm512_mask_mov_epi8(folded, needs_cf_mask, _mm512_set1_epi8((char)0xCF));

                // Handle final sigma: 'ς' (U+03C2, CF 82) → 'σ' (U+03C3, CF 83)
                __mmask64 is_final_sigma =
                    is_after_cf_mask & _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0x82));
                folded = _mm512_mask_add_epi8(folded, is_final_sigma, folded, _mm512_set1_epi8(1));

                _mm512_mask_storeu_epi8(target, prefix_mask, folded);
                target += greek_length, source += greek_length, source_length -= greek_length;
                continue;
            }
        }

        // 2.4. Fast path for 2-byte scripts without case folding (Hebrew, Arabic, Syriac, etc.)
        //
        // Lead bytes D7-DF cover Hebrew (D7), Arabic (D8-DB), Syriac (DC-DD), Thaana/NKo (DE-DF).
        // None of these scripts have case distinctions, so we can just copy them unchanged.
        // Note: D5/D6 cover Armenian which HAS case folding (including U+0587 which expands).
        __mmask64 is_caseless_2byte = _mm512_cmpge_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xD7)) &
                                      _mm512_cmple_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xDF));
        if (is_caseless_2byte) {
            __mmask64 is_caseless_second = is_caseless_2byte << 1;
            __mmask64 is_valid_caseless = ~is_non_ascii | is_caseless_2byte | is_caseless_second;
            sz_size_t caseless_length = sz_ice_first_invalid_(is_valid_caseless, load_mask, chunk_size);
            caseless_length -= caseless_length && ((is_caseless_2byte >> (caseless_length - 1)) & 1);

            if (caseless_length >= 2) {
                __mmask64 prefix_mask = sz_u64_mask_until_(caseless_length);
                // Fold only ASCII A-Z, copy 2-byte unchanged
                _mm512_mask_storeu_epi8(target, prefix_mask, sz_ice_fold_ascii_in_prefix_(source_vec.zmm, prefix_mask));
                target += caseless_length, source += caseless_length, source_length -= caseless_length;
                continue;
            }
        }

        // 2.5. Other 2-byte scripts (Latin Extended, Greek, Cyrillic, Armenian)
        //
        // Unlike Latin-1 where folding is a simple +0x20 to the second byte in-place, these scripts
        // require unpacking to 32-bit codepoints because:
        //   - Different scripts have different folding offsets (+0x20 for Cyrillic/Greek, +0x30 for Armenian, +0x50 for
        //   Cyrillic Ѐ-Џ)
        //   - Latin Extended-A uses parity-based rules (even codepoints → +1, odd → +1 in different ranges)
        //   - Some codepoints expand (İ→iı, ŉ→ʼn) requiring serial handling
        //
        // Strategy: Compress character start positions, gather lead/continuation bytes, expand to 32-bit,
        // decode UTF-8 to codepoints, apply vectorized folding rules, re-encode to UTF-8, scatter back
        // using expand instructions. Processes up to 16 characters per iteration.
        //
        // Detect any 2-byte lead (110xxxxx = 0xC0-0xDF) except C3 (Latin-1, already handled above)
        __mmask64 is_two_byte_lead = _mm512_cmpeq_epi8_mask(
            _mm512_and_si512(source_vec.zmm, _mm512_set1_epi8((char)0xE0)), _mm512_set1_epi8((char)0xC0));
        is_two_byte_lead &= ~is_latin1_lead; // Exclude C3
        __mmask64 two_byte_second_positions = is_two_byte_lead << 1;

        // Accept ALL 2-byte sequences; we'll detect singletons after decoding
        __mmask64 is_valid_two_byte_mix = ~is_non_ascii | is_two_byte_lead | two_byte_second_positions;
        sz_size_t two_byte_length = sz_ice_first_invalid_(is_valid_two_byte_mix, load_mask, chunk_size);
        two_byte_length -= two_byte_length && ((is_two_byte_lead >> (two_byte_length - 1)) & 1);

        if (two_byte_length >= 2) {
            __mmask64 prefix_mask = sz_u64_mask_until_(two_byte_length);
            __mmask64 is_char_start = (~is_non_ascii | is_two_byte_lead) & prefix_mask;
            sz_size_t num_chars = (sz_size_t)sz_u64_popcount(is_char_start);

            // Compress character start positions, gather first/second bytes
            sz_u512_vec_t char_indices;
            char_indices.zmm = _mm512_maskz_compress_epi8(is_char_start, indices_vec);

            // We can only process 16 chars at a time (one ZMM register of 32-bit values)
            // If we have more, truncate the prefix to only include those 16 chars
            if (num_chars > 16) {
                sz_u8_t last_char_idx = char_indices.u8s[15]; // 16th char's byte position
                two_byte_length = last_char_idx + ((is_two_byte_lead >> last_char_idx) & 1 ? 2 : 1);
                prefix_mask = sz_u64_mask_until_(two_byte_length);
                is_char_start &= prefix_mask;
                num_chars = 16;
            }

            sz_u512_vec_t first_bytes, second_bytes;
            first_bytes.zmm = _mm512_permutexvar_epi8(char_indices.zmm, source_vec.zmm);
            second_bytes.zmm =
                _mm512_permutexvar_epi8(_mm512_add_epi8(char_indices.zmm, _mm512_set1_epi8(1)), source_vec.zmm);

            // Expand to 32-bit for arithmetic
            __m512i first_wide = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(first_bytes.zmm));
            __m512i second_wide = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(second_bytes.zmm));
            __mmask16 is_two_byte_char = (__mmask16)_pext_u64(is_two_byte_lead & prefix_mask, is_char_start);

            // Decode: ASCII as-is, 2-byte as ((first & 0x1F) << 6) | (second & 0x3F)
            __m512i decoded =
                _mm512_or_si512(_mm512_slli_epi32(_mm512_and_si512(first_wide, _mm512_set1_epi32(0x1F)), 6),
                                _mm512_and_si512(second_wide, _mm512_set1_epi32(0x3F)));
            __m512i codepoints = _mm512_mask_blend_epi32(is_two_byte_char, first_wide, decoded);

            // Detect codepoints that need serial handling - ONLY ranges with case folding that
            // our vectorized rules don't handle correctly. Ranges without case folding (Arabic,
            // Hebrew, etc.) can pass through unchanged since no folding rules will match.
            __mmask16 needs_serial =
                // Specific singletons/expanders in Latin Extended-A:
                _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x0130)) | // İ (expands)
                _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x0149)) | // ŉ (expands)
                _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x0178)) | // Ÿ → ÿ (0x00FF)
                _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x017F)) | // ſ → s (ASCII)
                // Latin Extended-B: 0x0180-0x024F (irregular case folding patterns)
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0180)),
                                        _mm512_set1_epi32(0x00D0)) | // 0x0180-0x024F
                // Greek singletons with non-standard folding: 0x0345-0x0390
                // (Basic Greek Α-Ω 0x0391-0x03A9 and α-ω 0x03B1-0x03C9 are handled by vectorized +0x20 rules)
                // (Final sigma ς 0x03C2 → σ is also vectorized)
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0345)),
                                        _mm512_set1_epi32(0x4C)) | // 0x0345-0x0390
                // Greek ΰ (0x03B0) expands to 3 codepoints
                _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x03B0)) |
                // Greek symbols with non-standard folding: 0x03CF-0x03FF
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x03CF)),
                                        _mm512_set1_epi32(0x31)) | // 0x03CF-0x03FF
                // Cyrillic Extended: 0x0460-0x052F (has case folding, not covered)
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0460)),
                                        _mm512_set1_epi32(0x00D0)) | // 0x0460-0x052F
                // Armenian ligature that expands: U+0587 (և → եւ, ech + yiwn)
                _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x0587));

            // Only consider characters we're actually processing
            needs_serial &= (__mmask16)((1u << num_chars) - 1);

            // Handle serial-needed characters by processing them one at a time (assumes valid UTF-8)
            if (needs_serial) {
                sz_size_t first_special = (sz_size_t)sz_u64_ctz((sz_u64_t)needs_serial);
                if (first_special == 0) {
                    // First character needs serial - process it and continue the main loop
                    sz_rune_t rune;
                    sz_rune_length_t rune_length;
                    sz_rune_parse(source, &rune, &rune_length);
                    sz_rune_t folded_runes[3]; // Unicode case folding produces at most 3 runes
                    sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
                    for (sz_size_t i = 0; i != folded_count; ++i)
                        target += sz_rune_export(folded_runes[i], (sz_u8_t *)target);
                    source += rune_length;
                    source_length -= rune_length;
                    continue;
                }
                // Truncate to only process characters before the special one
                num_chars = first_special;
                // Recalculate byte length from character positions
                sz_u8_t last_char_idx = char_indices.u8s[num_chars - 1];
                two_byte_length = last_char_idx + ((is_two_byte_lead >> last_char_idx) & 1 ? 2 : 1);
                prefix_mask = sz_u64_mask_until_(two_byte_length);
                is_char_start &= prefix_mask;
                is_two_byte_char = (__mmask16)_pext_u64(is_two_byte_lead & prefix_mask, is_char_start);
            }

            // Apply folding rules - all use range check: (cp - base) < size
            __m512i folded = codepoints;

            // ASCII A-Z: 0x0041-0x005A → +0x20
            folded = _mm512_mask_add_epi32(
                folded,
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0041)), _mm512_set1_epi32(26)),
                folded, _mm512_set1_epi32(0x20));
            // Cyrillic А-Я: 0x0410-0x042F → +0x20
            folded =
                _mm512_mask_add_epi32(folded,
                                      _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0410)),
                                                              _mm512_set1_epi32(0x20)),
                                      folded, _mm512_set1_epi32(0x20));
            // Cyrillic Ѐ-Џ: 0x0400-0x040F → +0x50
            folded =
                _mm512_mask_add_epi32(folded,
                                      _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0400)),
                                                              _mm512_set1_epi32(0x10)),
                                      folded, _mm512_set1_epi32(0x50));
            // Greek Α-Ρ: 0x0391-0x03A1 → +0x20
            folded =
                _mm512_mask_add_epi32(folded,
                                      _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0391)),
                                                              _mm512_set1_epi32(0x11)),
                                      folded, _mm512_set1_epi32(0x20));
            // Greek Σ-Ϋ: 0x03A3-0x03AB → +0x20
            folded =
                _mm512_mask_add_epi32(folded,
                                      _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x03A3)),
                                                              _mm512_set1_epi32(0x09)),
                                      folded, _mm512_set1_epi32(0x20));
            // Armenian Ա-Ֆ: 0x0531-0x0556 → +0x30
            folded =
                _mm512_mask_add_epi32(folded,
                                      _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0531)),
                                                              _mm512_set1_epi32(0x26)),
                                      folded, _mm512_set1_epi32(0x30));
            // Latin Extended-A/B: complex alternating pattern with varying parity
            // Use _mm512_test_epi32_mask for cleaner parity check: returns 1 where (a & b) != 0
            __mmask16 is_odd = _mm512_test_epi32_mask(codepoints, _mm512_set1_epi32(1));
            __mmask16 is_even = ~is_odd;
            // Ranges where EVEN is uppercase (even → +1):
            // 0x0100-0x012F, 0x0132-0x0137, 0x014A-0x0177
            __mmask16 is_latin_even_upper =
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0100)),
                                        _mm512_set1_epi32(0x30)) |
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0132)),
                                        _mm512_set1_epi32(0x06)) |
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x014A)),
                                        _mm512_set1_epi32(0x2E));
            folded = _mm512_mask_add_epi32(folded, is_latin_even_upper & is_even, folded, _mm512_set1_epi32(1));
            // Ranges where ODD is uppercase (odd → +1):
            // 0x0139-0x0148, 0x0179-0x017E
            __mmask16 is_latin_odd_upper =
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0139)),
                                        _mm512_set1_epi32(0x10)) |
                _mm512_cmplt_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0179)),
                                        _mm512_set1_epi32(0x06));
            folded = _mm512_mask_add_epi32(folded, is_latin_odd_upper & is_odd, folded, _mm512_set1_epi32(1));
            // Special case: µ (U+00B5 MICRO SIGN) → μ (U+03BC GREEK SMALL LETTER MU)
            folded = _mm512_mask_mov_epi32(folded, _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x00B5)),
                                           _mm512_set1_epi32(0x03BC));
            // Special case: ς (U+03C2 GREEK SMALL LETTER FINAL SIGMA) → σ (U+03C3)
            folded = _mm512_mask_mov_epi32(folded, _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x03C2)),
                                           _mm512_set1_epi32(0x03C3));

            // Re-encode to UTF-8: lead = 0xC0 | (cp >> 6), second = 0x80 | (cp & 0x3F)
            __m512i new_lead = _mm512_or_si512(_mm512_set1_epi32(0xC0), _mm512_srli_epi32(folded, 6));
            __m512i new_second =
                _mm512_or_si512(_mm512_set1_epi32(0x80), _mm512_and_si512(folded, _mm512_set1_epi32(0x3F)));
            // ASCII stays single-byte
            __mmask16 is_ascii_out = _mm512_cmplt_epu32_mask(folded, _mm512_set1_epi32(0x80));
            new_lead = _mm512_mask_blend_epi32(is_ascii_out, new_lead, folded);

            // Scatter back using expand (inverse of compress) - no scalar loop needed!
            // Pack 32-bit results to bytes
            __m512i lead_zmm = _mm512_zextsi128_si512(_mm512_cvtepi32_epi8(new_lead));
            __m512i second_zmm = _mm512_zextsi128_si512(_mm512_cvtepi32_epi8(new_second));

            // Expand lead bytes to char start positions
            __m512i result = _mm512_mask_expand_epi8(source_vec.zmm, is_char_start, lead_zmm);

            // For second bytes: compress only 2-byte chars' seconds, then expand to continuation positions
            __m512i second_compressed = _mm512_maskz_compress_epi8((__mmask64)is_two_byte_char, second_zmm);
            result = _mm512_mask_expand_epi8(result, two_byte_second_positions & prefix_mask, second_compressed);

            _mm512_mask_storeu_epi8(target, prefix_mask, result);
            target += two_byte_length, source += two_byte_length, source_length -= two_byte_length;
            continue;
        }

        // 3. Handle 3-byte sequences (E0-EF leads), possibly mixed with ASCII
        //
        // Most 3-byte codepoints have NO case folding, making this a fast-path copy:
        //   - CJK Unified Ideographs (U+4E00-9FFF, U+3400-4DBF): E4-E9, some E3
        //   - Hangul Syllables (U+AC00-D7AF): EA-ED
        //   - General Punctuation, Symbols, Currency, etc.
        //
        // Exceptions requiring special handling:
        //   - Latin Extended Additional (U+1E00-1EFF = E1 B8-BB): +1 folding for even → vectorized
        //   - Greek Extended (U+1F00-1FFF = E1 BC-BF): Complex expansions → serial
        //   - Capital Eszett (U+1E9E = E1 BA 9E): Expands to "ss" → serial
        //   - Fullwidth A-Z (U+FF21-FF3A = EF BC A1-BA): +0x20 to third byte → vectorized
        //   - Cyrillic Ext-B, Cherokee (EA 99-9F, AD-AE): +1 folding → serial
        //
        // Strategy:
        //   - Fast path: ASCII + safe 3-byte (E0, E3-E9, EB-EE, most of EA) → fold ASCII, copy 3-byte
        //   - E1 path: Check for Latin Ext Add (vectorized) vs Greek Extended (serial)
        //   - EF path: Check for Fullwidth A-Z (vectorized)
        {
            // Check for any byte that prevents the fast path
            // Problematic bytes: C0-DF (2-byte), E1, E2, EF, F0-FF (4-byte)
            // EA is mostly safe (Hangul B0-BF) but some second bytes have folding:
            //   - 0x99-0x9F: Cyrillic Ext-B, Latin Ext-D (A640-A7FF)
            //   - 0xAD-0xAE: Cherokee Supplement (AB70-ABBF)
            __mmask64 is_two_byte_lead =
                _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xC0)),
                                       _mm512_set1_epi8(0x20)); // C0-DF
            __mmask64 is_e1_lead = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE1));
            __mmask64 is_e2_lead = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE2));
            __mmask64 is_ef_lead = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xEF));
            // For EA, only flag as complex if second byte is in problematic ranges
            __mmask64 is_ea_lead = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xEA));
            __mmask64 ea_second_byte_positions = is_ea_lead << 1;
            __mmask64 is_ea_complex =
                ea_second_byte_positions &
                (_mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x99)),
                                        _mm512_set1_epi8(0x07)) | // 0x99-0x9F
                 _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xAD)),
                                        _mm512_set1_epi8(0x02))); // 0xAD-0xAE
            __mmask64 has_complex =
                (is_two_byte_lead | is_four_byte_lead_mask | is_e1_lead | is_e2_lead | is_ea_complex | is_ef_lead) &
                load_mask;

            // Fast path: No complex bytes - just fold ASCII A-Z and copy everything else
            if (!has_complex) {
                // All bytes are ASCII or safe 3-byte (E0, E3-E9, EB-EE range)
                // Accept: ASCII, safe 3-byte leads (E0, E3-E9, EB-EE), continuations
                __mmask64 is_valid = ~is_non_ascii | is_three_byte_lead_mask | is_cont_mask;
                sz_size_t valid_length = sz_ice_first_invalid_(is_valid, load_mask, chunk_size);

                // Don't split a 3-byte sequence at chunk boundary.
                // Note: `>= 1` is an optimization, not redundant! When valid_length == 0, the mask
                // operations below are no-ops, but skipping them saves ~5% throughput on mixed content.
                if (valid_length >= 1) {
                    __mmask64 all_leads = is_three_byte_lead_mask & sz_u64_mask_until_(valid_length);
                    __mmask64 safe_mask = valid_length >= 3 ? sz_u64_mask_until_(valid_length - 2) : 0;
                    __mmask64 unsafe = all_leads & ~safe_mask;
                    if (unsafe) valid_length = sz_u64_ctz(unsafe);
                }

                if (valid_length >= 2) {
                    __mmask64 mask = sz_u64_mask_until_(valid_length);
                    // Fold ASCII A-Z, copy everything else unchanged
                    _mm512_mask_storeu_epi8(target, mask, sz_ice_fold_ascii_in_prefix_(source_vec.zmm, mask));
                    target += valid_length, source += valid_length, source_length -= valid_length;
                    continue;
                }
            }

            // 3.1. Georgian fast path: handles E1 82/83 content
            //
            // Georgian script uses E1 82 and E1 83 lead sequences:
            //   - E1 82 A0-BF: Uppercase (Ⴀ-Ⴟ) - folds to E2 B4 80-9F
            //   - E1 83 80-85: Uppercase (Ⴠ-Ⴥ) - folds to E2 B4 A0-A5
            //   - E1 83 86-BF: Lowercase/other (ა-ჿ) - no folding needed
            //
            // We include ALL E1 82/83 content in the fast path, but only transform uppercase.
            if (is_e1_lead && source_length >= 3) {
                // Check if E1 leads have Georgian second bytes (82 or 83)
                __m512i georgian_second_bytes =
                    _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(1)), source_vec.zmm);

                // Check for Georgian second bytes: 82 or 83
                // Only check positions where second byte is within chunk (positions 0-62 for 64-byte chunk)
                // Position 63's second byte would wrap around in the permutation
                __mmask64 safe_e1_mask = is_e1_lead & (load_mask >> 1);
                __mmask64 is_82_at_e1 =
                    _mm512_mask_cmpeq_epi8_mask(safe_e1_mask, georgian_second_bytes, _mm512_set1_epi8((char)0x82));
                __mmask64 is_83_at_e1 =
                    _mm512_mask_cmpeq_epi8_mask(safe_e1_mask, georgian_second_bytes, _mm512_set1_epi8((char)0x83));
                __mmask64 is_georgian_e1 = is_82_at_e1 | is_83_at_e1;

                // Check if all E1 leads are Georgian (82/83) with no mixed content.
                // Note: Both conditions are necessary, not redundant!
                // - `!non_georgian_e1`: ensures ALL E1 leads are Georgian (no Greek Extended, etc.)
                // - `is_georgian_e1`: ensures at least ONE Georgian exists (handles empty safe_e1_mask)
                __mmask64 non_georgian_e1 = safe_e1_mask & ~is_georgian_e1;
                if (!non_georgian_e1 && is_georgian_e1) {
                    // All Georgian 3-byte sequences are valid (E1 82 80-BF, E1 83 80-BF)
                    // We only transform the uppercase subset, rest passes through

                    // Find uppercase positions that need transformation:
                    // - E1 82 A0-BF: third byte in A0-BF range (U+10A0-10BF)
                    // - E1 83 80-85: third byte in 80-85 range (U+10C0-10C5)
                    // - E1 83 87: third byte = 87 (U+10C7)
                    // - E1 83 8D: third byte = 8D (U+10CD)
                    __mmask64 third_pos_82 = is_82_at_e1 << 2;
                    __mmask64 third_pos_83 = is_83_at_e1 << 2;

                    __mmask64 is_82_uppercase = _mm512_mask_cmplt_epu8_mask(
                        third_pos_82 & load_mask, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xA0)),
                        _mm512_set1_epi8(0x20));
                    // For E1 83: check 80-85, 87, or 8D
                    __mmask64 is_83_range = _mm512_mask_cmplt_epu8_mask(
                        third_pos_83 & load_mask, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x80)),
                        _mm512_set1_epi8(0x06));
                    __mmask64 is_83_c7 = _mm512_mask_cmpeq_epi8_mask(third_pos_83 & load_mask, source_vec.zmm,
                                                                     _mm512_set1_epi8((char)0x87));
                    __mmask64 is_83_cd = _mm512_mask_cmpeq_epi8_mask(third_pos_83 & load_mask, source_vec.zmm,
                                                                     _mm512_set1_epi8((char)0x8D));
                    __mmask64 is_83_uppercase = is_83_range | is_83_c7 | is_83_cd;

                    // Include ASCII, ALL Georgian E1 (not just uppercase), safe E2, continuations, safe EA
                    // E2 80-83 are safe (General Punctuation), others have case folding (Glagolitic, Coptic, etc.)
                    // Also include C2 leads (Latin-1 Supplement: U+0080-00BF) - no case folding needed
                    __mmask64 is_safe_ea = is_ea_lead & ~(is_ea_complex >> 1);
                    __mmask64 is_c2_lead = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xC2));
                    // E2 is only safe if second byte is 80-83 (General Punctuation quotes)
                    __m512i second_bytes_for_e2 =
                        _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(1)), source_vec.zmm);
                    __mmask64 safe_e2_positions = is_e2_lead & (load_mask >> 1); // Ensure second byte is in chunk
                    __mmask64 is_safe_e2 =
                        safe_e2_positions &
                        _mm512_mask_cmplt_epu8_mask(safe_e2_positions,
                                                    _mm512_sub_epi8(second_bytes_for_e2, _mm512_set1_epi8((char)0x80)),
                                                    _mm512_set1_epi8(0x04)); // 80-83
                    __mmask64 is_valid_georgian_mix =
                        ~is_non_ascii | is_georgian_e1 | is_safe_e2 | is_cont_mask | is_safe_ea | is_c2_lead;
                    // Exclude other 2-byte leads (C3-DF may need folding), 4-byte, EF, and unsafe E2
                    __mmask64 is_foldable_2byte = is_two_byte_lead & ~is_c2_lead;
                    __mmask64 is_unsafe_e2 = is_e2_lead & ~is_safe_e2;
                    is_valid_georgian_mix &= ~(is_foldable_2byte | is_four_byte_lead_mask | is_ef_lead | is_unsafe_e2);
                    sz_size_t georgian_length = sz_ice_first_invalid_(is_valid_georgian_mix, load_mask, chunk_size);

                    // Don't split multi-byte sequences (2-byte C2, 3-byte E1/E2/EA).
                    // Note: `>= 1` is an optimization - skips mask ops when nothing valid.
                    if (georgian_length >= 1) {
                        __mmask64 prefix = sz_u64_mask_until_(georgian_length);
                        // Check for incomplete 3-byte sequences (leads in last 2 positions)
                        __mmask64 leads3_in_prefix = is_three_byte_lead_mask & prefix;
                        __mmask64 safe3_mask = georgian_length >= 3 ? sz_u64_mask_until_(georgian_length - 2) : 0;
                        __mmask64 unsafe3 = leads3_in_prefix & ~safe3_mask;
                        // Check for incomplete 2-byte sequences (leads in last position)
                        __mmask64 leads2_in_prefix = is_c2_lead & prefix;
                        __mmask64 safe2_mask = georgian_length >= 2 ? sz_u64_mask_until_(georgian_length - 1) : 0;
                        __mmask64 unsafe2 = leads2_in_prefix & ~safe2_mask;
                        __mmask64 unsafe = unsafe3 | unsafe2;
                        if (unsafe) georgian_length = sz_u64_ctz(unsafe);
                    }

                    if (georgian_length >= 2) {
                        __mmask64 prefix_mask = sz_u64_mask_until_(georgian_length);

                        // Find uppercase leads that need transformation within prefix
                        __mmask64 uppercase_leads = ((is_82_uppercase | is_83_uppercase) >> 2) & is_georgian_e1;
                        uppercase_leads &= prefix_mask;

                        // Transform only uppercase Georgian: E1 82/83 XX → E2 B4 YY
                        __m512i folded = source_vec.zmm;

                        // Set lead bytes to E2 where uppercase Georgian
                        folded = _mm512_mask_blend_epi8(uppercase_leads, folded, _mm512_set1_epi8((char)0xE2));

                        // Set second bytes to B4 where uppercase Georgian
                        __mmask64 uppercase_second_pos = uppercase_leads << 1;
                        folded = _mm512_mask_blend_epi8(uppercase_second_pos, folded, _mm512_set1_epi8((char)0xB4));

                        // Adjust third bytes for uppercase only: -0x20 for 82, +0x20 for 83
                        __mmask64 prefix_82_upper = is_82_uppercase & prefix_mask;
                        __mmask64 prefix_83_upper = is_83_uppercase & prefix_mask;
                        folded = _mm512_mask_sub_epi8(folded, prefix_82_upper, folded, _mm512_set1_epi8(0x20));
                        folded = _mm512_mask_add_epi8(folded, prefix_83_upper, folded, _mm512_set1_epi8(0x20));

                        // Also fold ASCII A-Z
                        folded = _mm512_mask_add_epi8(folded, sz_ice_is_ascii_upper_(source_vec.zmm) & prefix_mask,
                                                      folded, ascii_case_offset);

                        // Fold Micro Sign: 'µ' (U+00B5, C2 B5) → 'μ' (U+03BC, CE BC)
                        __mmask64 c2_in_prefix = is_c2_lead & prefix_mask;
                        __mmask64 c2_second_pos = c2_in_prefix << 1;
                        __mmask64 is_micro_second =
                            c2_second_pos & _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xB5));
                        __mmask64 is_micro_lead = is_micro_second >> 1;
                        folded = _mm512_mask_blend_epi8(is_micro_lead, folded, _mm512_set1_epi8((char)0xCE));
                        folded = _mm512_mask_blend_epi8(is_micro_second, folded, _mm512_set1_epi8((char)0xBC));

                        _mm512_mask_storeu_epi8(target, prefix_mask, folded);
                        target += georgian_length, source += georgian_length, source_length -= georgian_length;
                        continue;
                    }
                }
            }

            // 3.2. Latin Extended Additional fast path: E1 B8-BB (U+1E00-1EFF)
            //
            // Vietnamese and other Latin Extended Additional characters use parity folding:
            //   - Even codepoints are uppercase → +1 to get lowercase
            //   - Odd codepoints are lowercase → no change
            // UTF-8 third byte determines parity: even third byte = even codepoint.
            //
            // Exceptions that need serial handling:
            //   - U+1E96–U+1E9E (E1 BA 96-9E):
            //     - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
            //     - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
            //     - 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
            //     - 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
            //     - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
            //     - 'ẛ' (U+1E9B, E1 BA 9B) → 'ṡ' (U+1E61, E1 B9 A1)
            //     - 'ẞ' (U+1E9E, E1 BA 9E) → "ss" (U+0073 U+0073, 73 73)
            if (is_e1_lead && source_length >= 3) {
                __m512i latin_ext_second_bytes =
                    _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(1)), source_vec.zmm);
                __m512i latin_ext_third_bytes =
                    _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(2)), source_vec.zmm);

                // Check for Latin Ext Add second bytes: B8, B9, BA, BB
                __mmask64 safe_e1_mask = is_e1_lead & (load_mask >> 1);
                __mmask64 is_latin_ext_e1 = _mm512_mask_cmplt_epu8_mask(
                    safe_e1_mask, _mm512_sub_epi8(latin_ext_second_bytes, _mm512_set1_epi8((char)0xB8)),
                    _mm512_set1_epi8(0x04)); // 0xB8-0xBB

                // Check for U+1E96-U+1E9E (E1 BA 96-9E) which expand to multiple codepoints
                __mmask64 is_ba_second =
                    _mm512_mask_cmpeq_epi8_mask(safe_e1_mask, latin_ext_second_bytes, _mm512_set1_epi8((char)0xBA));
                // Third byte in range 0x96-0x9E (9 values: 0x96 + 0..8)
                // Note: latin_ext_third_bytes holds third byte value at lead positions, so compare at is_ba_second
                // position
                __mmask64 is_special_third =
                    is_ba_second &
                    _mm512_mask_cmplt_epu8_mask(is_ba_second,
                                                _mm512_sub_epi8(latin_ext_third_bytes, _mm512_set1_epi8((char)0x96)),
                                                _mm512_set1_epi8(0x09));

                // All E1 leads must be Latin Ext Add (no Greek Ext BC-BF, Georgian 82/83)
                __mmask64 non_latin_ext_e1 = safe_e1_mask & ~is_latin_ext_e1;
                if (!non_latin_ext_e1 && is_latin_ext_e1 && !is_special_third) {
                    // Pure Latin Extended Additional content
                    // Accept ASCII + Latin Ext Add E1 + continuations
                    __mmask64 is_valid_latin_ext = ~is_non_ascii | is_latin_ext_e1 | is_cont_mask;
                    is_valid_latin_ext &= ~(is_four_byte_lead_mask | is_ef_lead);
                    sz_size_t latin_ext_length = sz_ice_first_invalid_(is_valid_latin_ext, load_mask, chunk_size);

                    // Don't split 3-byte sequences
                    if (latin_ext_length >= 1) {
                        __mmask64 prefix = sz_u64_mask_until_(latin_ext_length);
                        __mmask64 leads_in_prefix = is_three_byte_lead_mask & prefix;
                        __mmask64 safe_mask = latin_ext_length >= 3 ? sz_u64_mask_until_(latin_ext_length - 2) : 0;
                        __mmask64 unsafe = leads_in_prefix & ~safe_mask;
                        if (unsafe) latin_ext_length = sz_u64_ctz(unsafe);
                    }

                    if (latin_ext_length >= 3) {
                        __mmask64 prefix_mask = sz_u64_mask_until_(latin_ext_length);

                        // Find third byte positions of Latin Ext Add chars
                        __mmask64 third_positions = (is_latin_ext_e1 & prefix_mask) << 2;
                        third_positions &= prefix_mask;

                        // Check parity: even third byte = uppercase (fold +1)
                        __mmask64 is_even_third =
                            ~_mm512_test_epi8_mask(source_vec.zmm, _mm512_set1_epi8(0x01)) & third_positions;

                        // Apply folding
                        __m512i folded = source_vec.zmm;
                        folded = _mm512_mask_add_epi8(folded, is_even_third, folded, _mm512_set1_epi8(0x01));

                        // Also fold ASCII A-Z
                        folded = _mm512_mask_add_epi8(folded, sz_ice_is_ascii_upper_(source_vec.zmm) & prefix_mask,
                                                      folded, ascii_case_offset);

                        _mm512_mask_storeu_epi8(target, prefix_mask, folded);
                        target += latin_ext_length, source += latin_ext_length, source_length -= latin_ext_length;
                        continue;
                    }
                }
            }

            // ┌──────────────────────────────────────────────────────────────────────────────┐
            // │ 3.4. Slow Path: Mixed 3-byte content with E1/E2/EF leads                      │
            // │                                                                               │
            // │ This path handles remaining 3-byte content that couldn't use a fast path.    │
            // │ Sub-sections:                                                                 │
            // │   • E1 Greek Extended / Capital Eszett detection → serial fallback            │
            // │   • E2 validation (Glagolitic, Coptic, Letterlike) → serial fallback          │
            // │   • Latin Extended Additional parity folding (vectorized)                     │
            // │   • Georgian transformation (vectorized)                                      │
            // │   • Fullwidth A-Z (EF BC A1-BA) (vectorized)                                  │
            // └──────────────────────────────────────────────────────────────────────────────┘
            //
            // EA with problematic second bytes (is_ea_complex) also needs special handling
            // But plain EA (Hangul) is safe
            __mmask64 is_ea_lead_complex = is_ea_complex >> 1; // Shift back to lead position
            __mmask64 is_safe_three_byte_lead =
                is_three_byte_lead_mask & ~is_e1_lead & ~is_e2_lead & ~is_ea_lead_complex & ~is_ef_lead;
            __mmask64 is_valid_mixed = ~is_non_ascii | is_safe_three_byte_lead | is_cont_mask;
            is_valid_mixed &= ~is_four_byte_lead_mask;
            sz_size_t three_byte_length = sz_ice_first_invalid_(is_valid_mixed, load_mask, chunk_size);

            // Don't split a 3-byte sequence: find first incomplete lead and truncate there.
            // Note: `>= 1` is an optimization - skips mask ops when nothing valid.
            if (three_byte_length >= 1) {
                __mmask64 all_leads = is_three_byte_lead_mask & sz_u64_mask_until_(three_byte_length);
                __mmask64 safe_leads_mask = three_byte_length >= 3 ? sz_u64_mask_until_(three_byte_length - 2) : 0;
                __mmask64 unsafe_leads = all_leads & ~safe_leads_mask;
                if (unsafe_leads) three_byte_length = sz_u64_ctz(unsafe_leads);
            }

            // Need at least 2 bytes
            if (three_byte_length >= 2) {
                __mmask64 prefix_mask_3 = sz_u64_mask_until_(three_byte_length);
                __mmask64 three_byte_leads_in_prefix = is_three_byte_lead_mask & prefix_mask_3;

                // Check for problematic lead bytes in this prefix

                // E2 leads: E2 80-83 are safe (General Punctuation), but others need folding.
                // For safety and simplicity, treating ALL E2s here as unsafe forces serial fallback,
                // which handles both folding and identity cases correctly.
                __mmask64 is_unsafe_e2 = is_e2_lead & three_byte_leads_in_prefix;
                __mmask64 problematic_leads = (is_e1_lead | is_ef_lead | is_unsafe_e2) & three_byte_leads_in_prefix;

                if (!problematic_leads) {
                    // No E1 or EF leads in prefix - fold ASCII A-Z, copy 3-byte chars unchanged
                    _mm512_mask_storeu_epi8(target, prefix_mask_3,
                                            sz_ice_fold_ascii_in_prefix_(source_vec.zmm, prefix_mask_3));
                    target += three_byte_length, source += three_byte_length, source_length -= three_byte_length;
                    continue;
                }

                // Check if E1 leads need special handling:
                // - Greek Extended (E1 BC-BF = 1F00-1FFF): complex expansions
                // - Capital Eszett (E1 BA 9E = U+1E9E): expands to "ss"
                __mmask64 is_e1_in_prefix = is_e1_lead & three_byte_leads_in_prefix;
                if (is_e1_in_prefix) {
                    __m512i e1_second_bytes =
                        _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(1)), source_vec.zmm);
                    __m512i e1_third_bytes =
                        _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(2)), source_vec.zmm);

                    // Check for Greek Extended (second byte BC-BF)
                    __mmask64 is_greek_ext = _mm512_mask_cmplt_epu8_mask(
                        is_e1_in_prefix, _mm512_sub_epi8(e1_second_bytes, _mm512_set1_epi8((char)0xBC)),
                        _mm512_set1_epi8(0x04));

                    // Check for U+1E96-U+1E9E (E1 BA 96-9E) which expand to multiple codepoints
                    __mmask64 is_ba_second =
                        _mm512_mask_cmpeq_epi8_mask(is_e1_in_prefix, e1_second_bytes, _mm512_set1_epi8((char)0xBA));
                    // Third byte in range 0x96-0x9E (9 values)
                    __mmask64 is_special_third = _mm512_mask_cmplt_epu8_mask(
                        is_ba_second, _mm512_sub_epi8(e1_third_bytes, _mm512_set1_epi8((char)0x96)),
                        _mm512_set1_epi8(0x09));
                    __mmask64 needs_serial_e1 = is_greek_ext | is_special_third;

                    if (needs_serial_e1) {
                        sz_size_t first_special = sz_u64_ctz(needs_serial_e1);
                        if (first_special == 0) {
                            // First char needs serial processing (assumes valid UTF-8)
                            sz_rune_t rune;
                            sz_rune_length_t rune_length;
                            sz_rune_parse(source, &rune, &rune_length);
                            sz_rune_t folded_runes[3]; // Unicode case folding produces at most 3 runes
                            sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
                            for (sz_size_t i = 0; i != folded_count; ++i)
                                target += sz_rune_export(folded_runes[i], (sz_u8_t *)target);
                            source += rune_length;
                            source_length -= rune_length;
                            continue;
                        }
                        three_byte_length = first_special;
                        prefix_mask_3 = sz_u64_mask_until_(three_byte_length);
                        three_byte_leads_in_prefix = is_three_byte_lead_mask & prefix_mask_3;
                        is_ef_lead &= prefix_mask_3;
                        is_e1_in_prefix = is_e1_lead & three_byte_leads_in_prefix;
                    }

                    // Vectorized Latin Extended Additional (E1 B8-BB = U+1E00-1EFF)
                    // This covers Vietnamese letters which use +1 folding for even codepoints.
                    // Check for E1 B8/B9/BA/BB (second byte 0xB8-0xBB)
                    __mmask64 is_latin_ext_add = _mm512_mask_cmplt_epu8_mask(
                        is_e1_in_prefix, _mm512_sub_epi8(e1_second_bytes, _mm512_set1_epi8((char)0xB8)),
                        _mm512_set1_epi8(0x04)); // 0xB8-0xBB

                    if (is_latin_ext_add) {
                        // For Latin Ext Add, even codepoints fold to +1
                        // The third byte determines parity: even third byte = even codepoint
                        // Third bytes 80, 82, 84... are even codepoints (uppercase)
                        // Note: U+1E96-1E9E (E1 BA 96-9E) are already filtered above via is_special_third
                        __mmask64 third_positions = is_latin_ext_add << 2;
                        // Check if third byte is even (bit 0 = 0)
                        __mmask64 is_even_third = ~_mm512_test_epi8_mask(source_vec.zmm, _mm512_set1_epi8(0x01));
                        __mmask64 fold_third = third_positions & is_even_third & prefix_mask_3;

                        // Also fold ASCII A-Z (special handling: Latin Ext gets +1, ASCII gets +0x20)
                        __mmask64 is_upper_ascii = sz_ice_is_ascii_upper_(source_vec.zmm);
                        // First apply +1 to both Latin Ext (fold_third) and ASCII positions
                        __m512i folded =
                            _mm512_mask_add_epi8(source_vec.zmm, (fold_third | is_upper_ascii) & prefix_mask_3,
                                                 source_vec.zmm, _mm512_set1_epi8(0x01));
                        // Then add remaining +0x1F to ASCII only (total +0x20)
                        folded = _mm512_mask_add_epi8(folded, is_upper_ascii & prefix_mask_3, folded,
                                                      _mm512_set1_epi8(0x1F));

                        _mm512_mask_storeu_epi8(target, prefix_mask_3, folded);
                        target += three_byte_length, source += three_byte_length, source_length -= three_byte_length;
                        continue;
                    }

                    // Handle Georgian uppercase (E1 82/83) → lowercase (E2 B4)
                    // Georgian Mkhedruli: U+10A0-10C5 (E1 82 A0 - E1 83 85) → U+2D00-2D25 (E2 B4 80 - E2 B4 A5)
                    // This transformation changes the UTF-8 lead byte from E1 to E2, requiring special handling.
                    __mmask64 is_82_second =
                        _mm512_mask_cmpeq_epi8_mask(is_e1_in_prefix, e1_second_bytes, _mm512_set1_epi8((char)0x82));
                    __mmask64 is_83_second =
                        _mm512_mask_cmpeq_epi8_mask(is_e1_in_prefix, e1_second_bytes, _mm512_set1_epi8((char)0x83));
                    __mmask64 is_georgian_second = is_82_second | is_83_second;

                    if (is_georgian_second) {
                        // Validate third byte range for Georgian uppercase:
                        // - E1 82 A0-BF: U+10A0-10BF (32 chars)
                        // - E1 83 80-85: U+10C0-10C5 (6 chars)
                        __mmask64 third_pos_82 = is_82_second << 2;
                        __mmask64 third_pos_83 = is_83_second << 2;

                        // For E1 82: third byte must be A0-BF
                        __mmask64 is_82_valid = _mm512_mask_cmplt_epu8_mask(
                            third_pos_82, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xA0)),
                            _mm512_set1_epi8(0x20));

                        // For E1 83: third byte must be 80-85
                        __mmask64 is_83_valid = _mm512_mask_cmplt_epu8_mask(
                            third_pos_83, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x80)),
                            _mm512_set1_epi8(0x06));

                        __mmask64 georgian_leads = ((is_82_valid | is_83_valid) >> 2) & is_e1_in_prefix;

                        if (georgian_leads) {
                            // Transform: E1 82/83 XX → E2 B4 YY
                            // For 82: YY = XX - 0x20 (A0-BF → 80-9F)
                            // For 83: YY = XX + 0x20 (80-85 → A0-A5)

                            // Start with source, then apply transformations
                            __m512i folded = source_vec.zmm;

                            // Set lead bytes to E2 where Georgian
                            folded = _mm512_mask_blend_epi8(georgian_leads, folded, _mm512_set1_epi8((char)0xE2));

                            // Set second bytes to B4 where Georgian
                            __mmask64 georgian_second_pos = georgian_leads << 1;
                            folded = _mm512_mask_blend_epi8(georgian_second_pos, folded, _mm512_set1_epi8((char)0xB4));

                            // Adjust third bytes based on original second byte
                            // -0x20 for sequences that had 82, +0x20 for sequences that had 83
                            folded = _mm512_mask_sub_epi8(folded, is_82_valid, folded, _mm512_set1_epi8(0x20));
                            folded = _mm512_mask_add_epi8(folded, is_83_valid, folded, _mm512_set1_epi8(0x20));

                            // Also fold any ASCII A-Z that might be mixed in
                            folded =
                                _mm512_mask_add_epi8(folded, sz_ice_is_ascii_upper_(source_vec.zmm) & prefix_mask_3,
                                                     folded, ascii_case_offset);

                            _mm512_mask_storeu_epi8(target, prefix_mask_3, folded);
                            target += three_byte_length, source += three_byte_length,
                                source_length -= three_byte_length;
                            continue;
                        }
                    }
                }

                // Handle EF leads - check for Fullwidth A-Z (FF21-FF3A = EF BC A1 - EF BC BA)
                __mmask64 is_ef_in_prefix = is_ef_lead & three_byte_leads_in_prefix;
                if (is_ef_in_prefix) {
                    __m512i ef_second_bytes =
                        _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(1)), source_vec.zmm);
                    __m512i ef_third_bytes =
                        _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(2)), source_vec.zmm);

                    // Check for EF BC (Fullwidth block FF00-FF3F)
                    __mmask64 is_ef_bc =
                        _mm512_mask_cmpeq_epi8_mask(is_ef_in_prefix, ef_second_bytes, _mm512_set1_epi8((char)0xBC));

                    // Check third byte in A1-BA range (Fullwidth A-Z = FF21-FF3A)
                    // Third byte A1-BA corresponds to codepoints FF21-FF3A
                    __mmask64 is_fullwidth_az = _mm512_mask_cmplt_epu8_mask(
                        is_ef_bc, _mm512_sub_epi8(ef_third_bytes, _mm512_set1_epi8((char)0xA1)),
                        _mm512_set1_epi8(0x1A));

                    if (is_fullwidth_az) {
                        // Has Fullwidth A-Z - apply +0x20 to third byte for those positions
                        // Also fold any ASCII A-Z in the mixed content
                        __mmask64 third_byte_positions = is_fullwidth_az << 2;
                        __mmask64 fold_mask =
                            (third_byte_positions | sz_ice_is_ascii_upper_(source_vec.zmm)) & prefix_mask_3;
                        __m512i folded =
                            _mm512_mask_add_epi8(source_vec.zmm, fold_mask, source_vec.zmm, ascii_case_offset);
                        _mm512_mask_storeu_epi8(target, prefix_mask_3, folded);
                        target += three_byte_length, source += three_byte_length, source_length -= three_byte_length;
                        continue;
                    }
                }

                // No special 3-byte cases found - fold ASCII A-Z, copy 3-byte unchanged
                // But do NOT copy if we detected unsafe E2s that weren't handled!
                if (!is_unsafe_e2) {
                    _mm512_mask_storeu_epi8(target, prefix_mask_3,
                                            sz_ice_fold_ascii_in_prefix_(source_vec.zmm, prefix_mask_3));
                    target += three_byte_length, source += three_byte_length, source_length -= three_byte_length;
                    continue;
                }
            }
        }

        // 4. Handle 4-byte sequences (emoji, rare scripts): detect lead bytes (11110xxx = F0-F7)
        {
            __mmask64 is_valid_four_byte_only = is_four_byte_lead_mask | is_cont_mask;
            sz_size_t four_byte_length = sz_ice_first_invalid_(is_valid_four_byte_only, load_mask, chunk_size);

            // Don't split a 4-byte sequence: find first incomplete lead and truncate there.
            // Note: `>= 1` is an optimization - skips mask ops when nothing valid.
            if (four_byte_length >= 1) {
                __mmask64 all_leads = is_four_byte_lead_mask & sz_u64_mask_until_(four_byte_length);
                __mmask64 safe_leads_mask = four_byte_length >= 4 ? sz_u64_mask_until_(four_byte_length - 3) : 0;
                __mmask64 unsafe_leads = all_leads & ~safe_leads_mask;
                if (unsafe_leads) four_byte_length = sz_u64_ctz(unsafe_leads);
            }

            if (four_byte_length >= 4) {
                // Check if all 4-byte chars are in non-folding ranges (emoji: second byte >= 0x9F)
                __m512i f0_second_bytes =
                    _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(1)), source_vec.zmm);
                __mmask64 is_emoji_lead = _mm512_cmpge_epu8_mask(f0_second_bytes, _mm512_set1_epi8((char)0x9F));
                __mmask64 prefix_mask_4 = sz_u64_mask_until_(four_byte_length);
                __mmask64 four_byte_leads_in_prefix = is_four_byte_lead_mask & prefix_mask_4;

                // All 4-byte leads are emoji (second byte >= 0x9F) - no case folding needed
                if ((four_byte_leads_in_prefix & ~is_emoji_lead) == 0) {
                    _mm512_mask_storeu_epi8(target, prefix_mask_4, source_vec.zmm);
                    target += four_byte_length, source += four_byte_length, source_length -= four_byte_length;
                    continue;
                }
            }
        }

        // Mixed content or expanding characters - process one character serially inline
        {
            // Check for incomplete sequence at end of buffer before parsing
            sz_u8_t lead = (sz_u8_t)*source;
            sz_size_t expected_length = 1;
            if ((lead & 0xE0) == 0xC0) expected_length = 2;
            else if ((lead & 0xF0) == 0xE0)
                expected_length = 3;
            else if ((lead & 0xF8) == 0xF0)
                expected_length = 4;

            if (expected_length > source_length) {
                // Incomplete sequence at end - copy remaining bytes as-is
                while (source_length) {
                    *target++ = *source++;
                    source_length--;
                }
                break;
            }
            // Serial fallback for remaining bytes (assumes valid UTF-8)
            sz_rune_t rune;
            sz_rune_length_t rune_length;
            sz_rune_parse(source, &rune, &rune_length);

            sz_rune_t folded_runes[3]; // Unicode case folding produces at most 3 runes
            sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);

            for (sz_size_t i = 0; i != folded_count; ++i) target += sz_rune_export(folded_runes[i], (sz_u8_t *)target);
            source += rune_length;
            source_length -= rune_length;
        }
    }

    return (sz_size_t)(target - target_start);
}

/*  Undefine local helper macros to avoid namespace pollution */
#undef sz_ice_is_ascii_upper_
#undef sz_ice_fold_ascii_
#undef sz_ice_fold_ascii_in_prefix_
#undef sz_ice_transform_georgian_

SZ_PUBLIC sz_size_t sz_utf8_case_upper_ice(sz_cptr_t source, sz_size_t source_length, sz_ptr_t target) {
    // This is the inverse of sz_utf8_case_fold_ice - converts lowercase to uppercase.
    //
    // Key inversions:
    // 1. ASCII: a-z (0x61-0x7A) → A-Z (0x41-0x5A): subtract 0x20
    // 2. Latin-1 (C3): à-þ (A0-BE) → À-Þ (80-9E): subtract 0x20 from second byte
    // 3. Cyrillic (D0/D1):
    //    - D0 B0-BF (а-п) → D0 90-9F (А-П): -0x20
    //    - D1 80-8F (р-я) → D0 A0-AF (Р-Я): +0x20, D1→D0
    //    - D1 90-9F (ѐ-џ) → D0 80-8F (Ѐ-Џ): -0x10, D1→D0
    // 4. Greek (CE/CF):
    //    - CE B1-BF (α-ο) → CE 91-9F (Α-Ο): -0x20
    //    - CF 80-8B (π-ϋ) → CE A0-AB (Π-Ϋ): +0x20, CF→CE
    // 5. Fullwidth: ａ-ｚ (EF BD 81-9A) → Ａ-Ｚ (EF BC A1-BA): -0x20 third, BD→BC
    //
    sz_u512_vec_t source_vec;
    sz_ptr_t target_start = target;

    __m512i const a_lower_vec = _mm512_set1_epi8('a');
    __m512i const subtract26_vec = _mm512_set1_epi8(26);
    __m512i const ascii_case_offset = _mm512_set1_epi8(0x20);

    __m512i const utf8_cont_test_mask = _mm512_set1_epi8((char)0xC0);
    __m512i const utf8_cont_pattern = _mm512_set1_epi8((char)0x80);
    __m512i const utf8_3byte_test_mask = _mm512_set1_epi8((char)0xF0);
    __m512i const utf8_3byte_pattern = _mm512_set1_epi8((char)0xE0);
    __m512i const utf8_4byte_test_mask = _mm512_set1_epi8((char)0xF8);

    while (source_length) {
        _mm_prefetch(source + 1024, _MM_HINT_T1);
        _mm_prefetch(source + 512, _MM_HINT_T0);

        sz_size_t chunk_size = sz_min_of_two(source_length, 64);
        __mmask64 load_mask = sz_u64_mask_until_(chunk_size);
        source_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, source);
        __mmask64 is_non_ascii = _mm512_movepi8_mask(source_vec.zmm);

        // FAST PATH: Pure ASCII - uppercase a-z to A-Z
        if (is_non_ascii == 0) {
            __mmask64 is_lower = _mm512_cmplt_epu8_mask(
                _mm512_sub_epi8(source_vec.zmm, a_lower_vec), subtract26_vec);
            __m512i upper = _mm512_mask_sub_epi8(source_vec.zmm, is_lower, source_vec.zmm, ascii_case_offset);
            _mm512_mask_storeu_epi8(target, load_mask, upper);
            target += chunk_size, source += chunk_size, source_length -= chunk_size;
            continue;
        }

        // Compute lead byte masks
        __m512i masked_cont = _mm512_ternarylogic_epi64(source_vec.zmm, utf8_cont_test_mask, utf8_cont_test_mask, 0x80);
        __m512i masked_3byte = _mm512_ternarylogic_epi64(source_vec.zmm, utf8_3byte_test_mask, utf8_3byte_test_mask, 0x80);
        __m512i masked_4byte = _mm512_ternarylogic_epi64(source_vec.zmm, utf8_4byte_test_mask, utf8_4byte_test_mask, 0x80);
        __mmask64 is_cont_mask = _mm512_cmpeq_epi8_mask(masked_cont, utf8_cont_pattern);
        __mmask64 is_three_byte_lead_mask = _mm512_cmpeq_epi8_mask(masked_3byte, utf8_3byte_pattern);
        __mmask64 is_four_byte_lead_mask = _mm512_cmpeq_epi8_mask(masked_4byte, utf8_3byte_test_mask);

        // 3-byte caseless fast path (CJK, Thai, Hindi, etc.)
        {
            __mmask64 is_valid_pure_3byte_mask = is_three_byte_lead_mask | is_cont_mask;
            if ((is_valid_pure_3byte_mask & load_mask) == (is_non_ascii & load_mask) && !is_four_byte_lead_mask) {
                __mmask64 is_e1_e2_mask = _mm512_cmplt_epu8_mask(
                    _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xE1)), _mm512_set1_epi8(2));
                __mmask64 is_e1_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE1));
                __mmask64 is_ef_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xEF));
                __mmask64 is_e2_mask = _kand_mask64(is_e1_e2_mask, _knot_mask64(is_e1_mask));
                __mmask64 e2_second_byte_positions = is_e2_mask << 1;
                __mmask64 is_e2_upper_mask = e2_second_byte_positions &
                    ~_mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x80)),
                                            _mm512_set1_epi8(0x04));
                __mmask64 is_ea_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xEA));
                __mmask64 ea_second_byte_positions = is_ea_mask << 1;
                __mmask64 is_ea_upper_mask = ea_second_byte_positions &
                    (_mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x99)),
                                            _mm512_set1_epi8(0x07)) |
                     _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xAD)),
                                            _mm512_set1_epi8(0x02)));
                if (!(is_e1_mask | is_e2_upper_mask | is_ea_upper_mask | is_ef_mask)) {
                    sz_size_t copy_len = chunk_size;
                    __mmask64 leads_in_chunk_mask = is_three_byte_lead_mask & load_mask;
                    if (leads_in_chunk_mask) {
                        int last_lead_pos = 63 - sz_u64_clz(leads_in_chunk_mask);
                        if (last_lead_pos + 3 > (int)copy_len) copy_len = last_lead_pos;
                    }
                    if (copy_len > 0) {
                        __mmask64 copy_mask = sz_u64_mask_until_(copy_len);
                        // Uppercase ASCII within 3-byte content
                        __mmask64 is_lower = _mm512_cmplt_epu8_mask(
                            _mm512_sub_epi8(source_vec.zmm, a_lower_vec), subtract26_vec);
                        __m512i upper = _mm512_mask_sub_epi8(source_vec.zmm, is_lower & copy_mask,
                                                             source_vec.zmm, ascii_case_offset);
                        _mm512_mask_storeu_epi8(target, copy_mask, upper);
                        target += copy_len, source += copy_len, source_length -= copy_len;
                        continue;
                    }
                }
            }
        }

        // Latin-1 Supplement (C3): à-þ → À-Þ (excluding ß=0x9F which expands to SS)
        __mmask64 is_latin1_lead = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xC3));
        __mmask64 latin1_second_byte_positions = is_latin1_lead << 1;
        // Exclude ß (C3 9F) - it expands to SS and must go through fallback
        __mmask64 is_sharp_s = latin1_second_byte_positions &
            _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0x9F));
        __mmask64 is_valid_latin1_mix = ~is_non_ascii | is_latin1_lead | latin1_second_byte_positions;
        is_valid_latin1_mix &= ~is_sharp_s; // Stop before ß
        sz_size_t latin1_length = sz_ice_first_invalid_(is_valid_latin1_mix, load_mask, chunk_size);
        latin1_length -= latin1_length && ((is_latin1_lead >> (latin1_length - 1)) & 1);

        if (latin1_length >= 2) {
            __mmask64 prefix_mask = sz_u64_mask_until_(latin1_length);
            __mmask64 latin1_second_bytes = latin1_second_byte_positions & prefix_mask;

            // ASCII a-z and Latin-1 à-þ (second byte 0xA0-0xBE excl. ÷=0xB7) get -0x20
            __mmask64 is_lower_ascii = _mm512_cmplt_epu8_mask(
                _mm512_sub_epi8(source_vec.zmm, a_lower_vec), subtract26_vec);
            // Latin-1 lowercase: second byte 0xA0-0xBE (à-þ), excluding ÷ at 0xB7
            __mmask64 is_latin1_lower = _mm512_mask_cmpge_epu8_mask(latin1_second_bytes, source_vec.zmm,
                                                                     _mm512_set1_epi8((char)0xA0));
            is_latin1_lower &= _mm512_mask_cmple_epu8_mask(latin1_second_bytes, source_vec.zmm,
                                                           _mm512_set1_epi8((char)0xBE));
            is_latin1_lower ^= _mm512_mask_cmpeq_epi8_mask(is_latin1_lower, source_vec.zmm,
                                                           _mm512_set1_epi8((char)0xB7)); // Exclude ÷

            __m512i upper = _mm512_mask_sub_epi8(source_vec.zmm, (is_lower_ascii | is_latin1_lower) & prefix_mask,
                                                  source_vec.zmm, ascii_case_offset);
            _mm512_mask_storeu_epi8(target, prefix_mask, upper);
            target += latin1_length, source += latin1_length, source_length -= latin1_length;
            continue;
        }

        // Cyrillic (D0/D1): lowercase → uppercase
        {
            __mmask64 is_cyrillic_lead_mask = _mm512_cmplt_epu8_mask(
                _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xD0)), _mm512_set1_epi8(2));
            __mmask64 is_d0_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xD0));
            __mmask64 is_d1_mask = _kand_mask64(is_cyrillic_lead_mask, _knot_mask64(is_d0_mask));
            __mmask64 cyrillic_second_byte_positions = is_cyrillic_lead_mask << 1;

            // Exclude D1 A0+ (Extended Cyrillic)
            __mmask64 is_d1_extended_mask =
                (is_d1_mask << 1) & _mm512_cmpge_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xA0));

            __mmask64 is_valid_cyrillic_mix_mask =
                ~is_non_ascii | is_cyrillic_lead_mask | cyrillic_second_byte_positions;
            is_valid_cyrillic_mix_mask &= ~is_d1_extended_mask;
            sz_size_t cyrillic_length = sz_ice_first_invalid_(is_valid_cyrillic_mix_mask, load_mask, chunk_size);
            cyrillic_length -= cyrillic_length && ((is_cyrillic_lead_mask >> (cyrillic_length - 1)) & 1);

            if (cyrillic_length >= 2) {
                __mmask64 prefix_mask = sz_u64_mask_until_(cyrillic_length);
                __mmask64 is_after_d0_mask = (is_d0_mask << 1) & prefix_mask;
                __mmask64 is_after_d1_mask = (is_d1_mask << 1) & prefix_mask;

                // Start with source, apply ASCII uppercasing
                __mmask64 is_lower_ascii = _mm512_cmplt_epu8_mask(
                    _mm512_sub_epi8(source_vec.zmm, a_lower_vec), subtract26_vec);
                __m512i upper = _mm512_mask_sub_epi8(source_vec.zmm, is_lower_ascii & prefix_mask,
                                                      source_vec.zmm, ascii_case_offset);

                // D0 B0-BF (а-п) → D0 90-9F (А-П): -0x20 to second byte
                __mmask64 is_d0_lower_mask = _mm512_mask_cmpge_epu8_mask(is_after_d0_mask, source_vec.zmm,
                                                                          _mm512_set1_epi8((char)0xB0));
                upper = _mm512_mask_sub_epi8(upper, is_d0_lower_mask, upper, _mm512_set1_epi8(0x20));

                // D1 80-8F (р-я) → D0 A0-AF (Р-Я): +0x20 to second byte, D1→D0 for lead
                __mmask64 is_d1_lower1_mask = _mm512_mask_cmplt_epu8_mask(is_after_d1_mask, source_vec.zmm,
                                                                           _mm512_set1_epi8((char)0x90));
                upper = _mm512_mask_add_epi8(upper, is_d1_lower1_mask, upper, _mm512_set1_epi8(0x20));
                // Change lead byte D1→D0
                upper = _mm512_mask_sub_epi8(upper, is_d1_lower1_mask >> 1, upper, _mm512_set1_epi8(1));

                // D1 90-9F (ѐ-џ) → D0 80-8F (Ѐ-Џ): -0x10 to second byte, D1→D0 for lead
                __mmask64 is_d1_lower2_mask = _mm512_mask_cmplt_epu8_mask(
                    is_after_d1_mask, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x90)),
                    _mm512_set1_epi8(0x10));
                upper = _mm512_mask_sub_epi8(upper, is_d1_lower2_mask, upper, _mm512_set1_epi8(0x10));
                // Change lead byte D1→D0
                upper = _mm512_mask_sub_epi8(upper, is_d1_lower2_mask >> 1, upper, _mm512_set1_epi8(1));

                _mm512_mask_storeu_epi8(target, prefix_mask, upper);
                target += cyrillic_length, source += cyrillic_length, source_length -= cyrillic_length;
                continue;
            }
        }

        // Cyrillic Extended (D2): lowercase → uppercase (parity-based)
        // U+0480-04BF = D2 80-BF, odd second byte is lowercase
        // ҳ (D2 B3) → Ҳ (D2 B2), ҷ (D2 B7) → Ҷ (D2 B6)
        {
            __mmask64 is_d2_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xD2));
            __mmask64 d2_second_byte_positions = is_d2_mask << 1;

            __mmask64 is_valid_d2_mix_mask = ~is_non_ascii | is_d2_mask | d2_second_byte_positions;
            sz_size_t d2_length = sz_ice_first_invalid_(is_valid_d2_mix_mask, load_mask, chunk_size);
            d2_length -= d2_length && ((is_d2_mask >> (d2_length - 1)) & 1);

            if (d2_length >= 2) {
                __mmask64 prefix_mask = sz_u64_mask_until_(d2_length);
                __mmask64 is_after_d2_mask = (is_d2_mask << 1) & prefix_mask;

                // ASCII uppercasing
                __mmask64 is_lower_ascii = _mm512_cmplt_epu8_mask(
                    _mm512_sub_epi8(source_vec.zmm, a_lower_vec), subtract26_vec);
                __m512i upper = _mm512_mask_sub_epi8(source_vec.zmm, is_lower_ascii & prefix_mask,
                                                      source_vec.zmm, ascii_case_offset);

                // Parity-based: odd second byte (8B,8D,...,BD,BF) → subtract 1
                // Range 0x8A-0xBE: uppercase even, lowercase odd
                __mmask64 is_in_range = _mm512_mask_cmpge_epu8_mask(is_after_d2_mask, source_vec.zmm,
                                                                     _mm512_set1_epi8((char)0x8B));
                is_in_range &= _mm512_mask_cmple_epu8_mask(is_after_d2_mask, source_vec.zmm,
                                                           _mm512_set1_epi8((char)0xBF));
                // Check if odd (lowercase)
                __m512i byte_and_1 = _mm512_and_si512(source_vec.zmm, _mm512_set1_epi8(1));
                __mmask64 is_odd = _mm512_cmpeq_epi8_mask(byte_and_1, _mm512_set1_epi8(1));
                __mmask64 is_d2_lower = is_in_range & is_odd;

                upper = _mm512_mask_sub_epi8(upper, is_d2_lower, upper, _mm512_set1_epi8(1));

                _mm512_mask_storeu_epi8(target, prefix_mask, upper);
                target += d2_length, source += d2_length, source_length -= d2_length;
                continue;
            }
        }

        // Cyrillic Extended (D3): lowercase → uppercase (parity-based)
        // U+04C0-04FF = D3 80-BF
        // Two sub-ranges with opposite parity:
        // - 0x04C2-0x04CE (D3 82-8E): even second byte is lowercase → subtract 1
        // - 0x04D1-0x04FF (D3 91-BF): odd second byte is lowercase → subtract 1
        // Example: ӣ (D3 A3, U+04E3) → Ӣ (D3 A2, U+04E2)
        {
            __mmask64 is_d3_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xD3));
            __mmask64 d3_second_byte_positions = is_d3_mask << 1;

            __mmask64 is_valid_d3_mix_mask = ~is_non_ascii | is_d3_mask | d3_second_byte_positions;
            sz_size_t d3_length = sz_ice_first_invalid_(is_valid_d3_mix_mask, load_mask, chunk_size);
            d3_length -= d3_length && ((is_d3_mask >> (d3_length - 1)) & 1);

            if (d3_length >= 2) {
                __mmask64 prefix_mask = sz_u64_mask_until_(d3_length);
                __mmask64 is_after_d3_mask = (is_d3_mask << 1) & prefix_mask;

                // ASCII uppercasing
                __mmask64 is_lower_ascii = _mm512_cmplt_epu8_mask(
                    _mm512_sub_epi8(source_vec.zmm, a_lower_vec), subtract26_vec);
                __m512i upper = _mm512_mask_sub_epi8(source_vec.zmm, is_lower_ascii & prefix_mask,
                                                      source_vec.zmm, ascii_case_offset);

                // Get parity of second byte
                __m512i byte_and_1 = _mm512_and_si512(source_vec.zmm, _mm512_set1_epi8(1));
                __mmask64 is_odd = _mm512_cmpeq_epi8_mask(byte_and_1, _mm512_set1_epi8(1));

                // Range 1: D3 82-8E (0x04C2-0x04CE) - even is lowercase
                __mmask64 is_range1 = _mm512_mask_cmpge_epu8_mask(is_after_d3_mask, source_vec.zmm,
                                                                   _mm512_set1_epi8((char)0x82));
                is_range1 &= _mm512_mask_cmple_epu8_mask(is_after_d3_mask, source_vec.zmm,
                                                         _mm512_set1_epi8((char)0x8E));
                __mmask64 is_d3_lower1 = is_range1 & ~is_odd;  // even bytes are lowercase

                // Range 2: D3 91-BF (0x04D1-0x04FF) - odd is lowercase
                __mmask64 is_range2 = _mm512_mask_cmpge_epu8_mask(is_after_d3_mask, source_vec.zmm,
                                                                   _mm512_set1_epi8((char)0x91));
                is_range2 &= _mm512_mask_cmple_epu8_mask(is_after_d3_mask, source_vec.zmm,
                                                         _mm512_set1_epi8((char)0xBF));
                __mmask64 is_d3_lower2 = is_range2 & is_odd;  // odd bytes are lowercase

                upper = _mm512_mask_sub_epi8(upper, is_d3_lower1 | is_d3_lower2, upper, _mm512_set1_epi8(1));

                _mm512_mask_storeu_epi8(target, prefix_mask, upper);
                target += d3_length, source += d3_length, source_length -= d3_length;
                continue;
            }
        }

        // Cyrillic Extended D1 A0+ (0x0460-0x047F): lowercase → uppercase (parity-based)
        // D1 A0-BF, odd second byte is lowercase → subtract 1
        {
            __mmask64 is_d1_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xD1));
            __mmask64 d1_second_byte_positions = is_d1_mask << 1;
            // Only D1 A0+ (Extended Cyrillic)
            __mmask64 is_d1_extended = d1_second_byte_positions &
                _mm512_cmpge_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xA0));

            __mmask64 is_valid_d1ext_mix_mask = ~is_non_ascii | is_d1_mask | d1_second_byte_positions;
            // Exclude non-extended D1 (D1 80-9F)
            __mmask64 is_d1_basic = d1_second_byte_positions &
                _mm512_cmplt_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xA0));
            is_valid_d1ext_mix_mask &= ~is_d1_basic;

            sz_size_t d1ext_length = sz_ice_first_invalid_(is_valid_d1ext_mix_mask, load_mask, chunk_size);
            d1ext_length -= d1ext_length && ((is_d1_mask >> (d1ext_length - 1)) & 1);

            if (d1ext_length >= 2) {
                __mmask64 prefix_mask = sz_u64_mask_until_(d1ext_length);

                // ASCII uppercasing
                __mmask64 is_lower_ascii = _mm512_cmplt_epu8_mask(
                    _mm512_sub_epi8(source_vec.zmm, a_lower_vec), subtract26_vec);
                __m512i upper = _mm512_mask_sub_epi8(source_vec.zmm, is_lower_ascii & prefix_mask,
                                                      source_vec.zmm, ascii_case_offset);

                // Parity: odd second byte (A1,A3,...,BF) → subtract 1
                __m512i byte_and_1 = _mm512_and_si512(source_vec.zmm, _mm512_set1_epi8(1));
                __mmask64 is_odd = _mm512_cmpeq_epi8_mask(byte_and_1, _mm512_set1_epi8(1));
                __mmask64 is_d1ext_lower = is_d1_extended & prefix_mask & is_odd;

                upper = _mm512_mask_sub_epi8(upper, is_d1ext_lower, upper, _mm512_set1_epi8(1));

                _mm512_mask_storeu_epi8(target, prefix_mask, upper);
                target += d1ext_length, source += d1ext_length, source_length -= d1ext_length;
                continue;
            }
        }

        // Cyrillic Extended D4 (0x0500-0x052E): lowercase → uppercase (parity-based)
        // D4 80-AE, odd second byte is lowercase → subtract 1
        {
            __mmask64 is_d4_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xD4));
            __mmask64 d4_second_byte_positions = is_d4_mask << 1;

            __mmask64 is_valid_d4_mix_mask = ~is_non_ascii | is_d4_mask | d4_second_byte_positions;
            sz_size_t d4_length = sz_ice_first_invalid_(is_valid_d4_mix_mask, load_mask, chunk_size);
            d4_length -= d4_length && ((is_d4_mask >> (d4_length - 1)) & 1);

            if (d4_length >= 2) {
                __mmask64 prefix_mask = sz_u64_mask_until_(d4_length);
                __mmask64 is_after_d4_mask = (is_d4_mask << 1) & prefix_mask;

                // ASCII uppercasing
                __mmask64 is_lower_ascii = _mm512_cmplt_epu8_mask(
                    _mm512_sub_epi8(source_vec.zmm, a_lower_vec), subtract26_vec);
                __m512i upper = _mm512_mask_sub_epi8(source_vec.zmm, is_lower_ascii & prefix_mask,
                                                      source_vec.zmm, ascii_case_offset);

                // Range D4 81-AF (0x0501-0x052F): odd second byte is lowercase
                __mmask64 is_in_range = _mm512_mask_cmpge_epu8_mask(is_after_d4_mask, source_vec.zmm,
                                                                     _mm512_set1_epi8((char)0x81));
                is_in_range &= _mm512_mask_cmple_epu8_mask(is_after_d4_mask, source_vec.zmm,
                                                           _mm512_set1_epi8((char)0xAF));
                __m512i byte_and_1 = _mm512_and_si512(source_vec.zmm, _mm512_set1_epi8(1));
                __mmask64 is_odd = _mm512_cmpeq_epi8_mask(byte_and_1, _mm512_set1_epi8(1));
                __mmask64 is_d4_lower = is_in_range & is_odd;

                upper = _mm512_mask_sub_epi8(upper, is_d4_lower, upper, _mm512_set1_epi8(1));

                _mm512_mask_storeu_epi8(target, prefix_mask, upper);
                target += d4_length, source += d4_length, source_length -= d4_length;
                continue;
            }
        }

        // Greek (CE/CF): lowercase → uppercase
        {
            __mmask64 is_greek_lead_mask = _mm512_cmplt_epu8_mask(
                _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xCE)), _mm512_set1_epi8(2));
            __mmask64 is_ce_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xCE));
            __mmask64 is_cf_mask = _kand_mask64(is_greek_lead_mask, _knot_mask64(is_ce_mask));
            __mmask64 greek_second_byte_positions = is_greek_lead_mask << 1;

            __mmask64 is_valid_greek_mix_mask =
                ~is_non_ascii | is_greek_lead_mask | greek_second_byte_positions;
            sz_size_t greek_length = sz_ice_first_invalid_(is_valid_greek_mix_mask, load_mask, chunk_size);
            greek_length -= greek_length && ((is_greek_lead_mask >> (greek_length - 1)) & 1);

            if (greek_length >= 2) {
                __mmask64 prefix_mask = sz_u64_mask_until_(greek_length);
                __mmask64 is_after_ce_mask = (is_ce_mask << 1) & prefix_mask;
                __mmask64 is_after_cf_mask = (is_cf_mask << 1) & prefix_mask;

                // ASCII uppercasing
                __mmask64 is_lower_ascii = _mm512_cmplt_epu8_mask(
                    _mm512_sub_epi8(source_vec.zmm, a_lower_vec), subtract26_vec);
                __m512i upper = _mm512_mask_sub_epi8(source_vec.zmm, is_lower_ascii & prefix_mask,
                                                      source_vec.zmm, ascii_case_offset);

                // CE B1-BF (α-ο) → CE 91-9F (Α-Ο): -0x20
                __mmask64 is_ce_lower_mask = _mm512_mask_cmpge_epu8_mask(is_after_ce_mask, source_vec.zmm,
                                                                          _mm512_set1_epi8((char)0xB1));
                upper = _mm512_mask_sub_epi8(upper, is_ce_lower_mask, upper, _mm512_set1_epi8(0x20));

                // CF 80-8B (π-ϋ) → CE A0-AB (Π-Ϋ): +0x20, CF→CE
                __mmask64 is_cf_lower_mask = _mm512_mask_cmplt_epu8_mask(
                    is_after_cf_mask, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x80)),
                    _mm512_set1_epi8(0x0C));
                upper = _mm512_mask_add_epi8(upper, is_cf_lower_mask, upper, _mm512_set1_epi8(0x20));
                upper = _mm512_mask_sub_epi8(upper, is_cf_lower_mask >> 1, upper, _mm512_set1_epi8(1));

                // Special case: ς (CF 82, U+03C2) → Σ (CE A3, U+03A3)
                // The +0x20 above gave A2, need to add 1 more to get A3
                __mmask64 is_final_sigma = is_after_cf_mask &
                    _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0x82));
                upper = _mm512_mask_add_epi8(upper, is_final_sigma, upper, _mm512_set1_epi8(1));

                _mm512_mask_storeu_epi8(target, prefix_mask, upper);
                target += greek_length, source += greek_length, source_length -= greek_length;
                continue;
            }
        }

        // Georgian lowercase (E2 B4) → uppercase (E1 82/83)
        // Inverse of fold: E1 82 A0-BF → E2 B4 80-9F, E1 83 80-85 → E2 B4 A0-A5
        // For upper: E2 B4 80-9F → E1 82 A0-BF, E2 B4 A0-A5 → E1 83 80-85
        {
            __mmask64 is_e2_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE2));
            if (is_e2_mask) {
                __mmask64 e2_second_positions = is_e2_mask << 1;
                // Check for B4 second byte (Georgian lowercase range)
                __mmask64 is_b4_second = e2_second_positions &
                    _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xB4));

                if (is_b4_second) {
                    __mmask64 is_valid_georgian = ~is_non_ascii | is_e2_mask | is_cont_mask;
                    sz_size_t georgian_length = sz_ice_first_invalid_(is_valid_georgian, load_mask, chunk_size);
                    georgian_length -= georgian_length && ((is_e2_mask >> (georgian_length - 1)) & 1);

                    if (georgian_length >= 3) {
                        __mmask64 prefix_mask = sz_u64_mask_until_(georgian_length);
                        __mmask64 is_after_e2_mask = (is_e2_mask << 1) & prefix_mask;
                        __mmask64 third_positions = (is_e2_mask << 2) & prefix_mask;

                        // Start with source, apply ASCII uppercasing
                        __mmask64 is_lower_ascii = _mm512_cmplt_epu8_mask(
                            _mm512_sub_epi8(source_vec.zmm, a_lower_vec), subtract26_vec);
                        __m512i upper = _mm512_mask_sub_epi8(source_vec.zmm, is_lower_ascii & prefix_mask,
                                                              source_vec.zmm, ascii_case_offset);

                        // Check if B4 second byte (only process Georgian lowercase)
                        __mmask64 is_georgian_lower = is_after_e2_mask &
                            _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xB4));

                        // Third byte 80-9F → E1 82 A0-BF (add 0x20 to third, change E2→E1, B4→82)
                        // Third byte A0-A5 → E1 83 80-85 (subtract 0x20 from third, change E2→E1, B4→83)
                        __mmask64 third_after_b4 = (is_georgian_lower << 1) & prefix_mask;
                        __mmask64 is_range1 = third_after_b4 &
                            _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x80)),
                                                   _mm512_set1_epi8(0x20)); // 80-9F
                        __mmask64 is_range2 = third_after_b4 &
                            _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xA0)),
                                                   _mm512_set1_epi8(0x06)); // A0-A5

                        // Transform range1: third +0x20, lead E2→E1, second B4→82
                        upper = _mm512_mask_add_epi8(upper, is_range1, upper, _mm512_set1_epi8(0x20));
                        upper = _mm512_mask_sub_epi8(upper, is_range1 >> 2, upper, _mm512_set1_epi8(1)); // E2→E1
                        upper = _mm512_mask_sub_epi8(upper, is_range1 >> 1, upper, _mm512_set1_epi8(0x32)); // B4→82

                        // Transform range2: third -0x20, lead E2→E1, second B4→83
                        upper = _mm512_mask_sub_epi8(upper, is_range2, upper, _mm512_set1_epi8(0x20));
                        upper = _mm512_mask_sub_epi8(upper, is_range2 >> 2, upper, _mm512_set1_epi8(1)); // E2→E1
                        upper = _mm512_mask_sub_epi8(upper, is_range2 >> 1, upper, _mm512_set1_epi8(0x31)); // B4→83

                        _mm512_mask_storeu_epi8(target, prefix_mask, upper);
                        target += georgian_length, source += georgian_length, source_length -= georgian_length;
                        continue;
                    }
                }
            }
        }

        // Latin Extended Additional (E1 B8-BB): lowercase → uppercase (parity-based)
        // Odd codepoints are lowercase → subtract 1 to get uppercase
        // UTF-8 third byte determines parity: odd third byte = odd codepoint = lowercase
        {
            __mmask64 is_e1_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE1));
            if (is_e1_mask) {
                __mmask64 e1_second_positions = is_e1_mask << 1;
                // Check for B8-BB second byte (Latin Extended Additional)
                __mmask64 is_latin_ext = e1_second_positions &
                    _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xB8)),
                                           _mm512_set1_epi8(0x04)); // B8, B9, BA, BB

                // Exclude Georgian (82/83) - they need different handling
                __mmask64 is_georgian = e1_second_positions &
                    _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x82)),
                                           _mm512_set1_epi8(0x02)); // 82, 83

                if (is_latin_ext && !is_georgian) {
                    __mmask64 is_valid_latin_ext = ~is_non_ascii | is_e1_mask | is_cont_mask;
                    sz_size_t latin_ext_length = sz_ice_first_invalid_(is_valid_latin_ext, load_mask, chunk_size);
                    latin_ext_length -= latin_ext_length && ((is_e1_mask >> (latin_ext_length - 1)) & 1);

                    if (latin_ext_length >= 3) {
                        __mmask64 prefix_mask = sz_u64_mask_until_(latin_ext_length);
                        __mmask64 third_positions = (is_e1_mask << 2) & prefix_mask;

                        // Start with source, apply ASCII uppercasing
                        __mmask64 is_lower_ascii = _mm512_cmplt_epu8_mask(
                            _mm512_sub_epi8(source_vec.zmm, a_lower_vec), subtract26_vec);
                        __m512i upper = _mm512_mask_sub_epi8(source_vec.zmm, is_lower_ascii & prefix_mask,
                                                              source_vec.zmm, ascii_case_offset);

                        // Parity: odd third byte is lowercase → subtract 1
                        __mmask64 is_latin_ext_second = (is_e1_mask << 1) & prefix_mask &
                            _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xB8)),
                                                   _mm512_set1_epi8(0x04));
                        __mmask64 third_after_latin = (is_latin_ext_second << 1) & prefix_mask;
                        __m512i byte_and_1 = _mm512_and_si512(source_vec.zmm, _mm512_set1_epi8(1));
                        __mmask64 is_odd = _mm512_cmpeq_epi8_mask(byte_and_1, _mm512_set1_epi8(1));
                        __mmask64 is_lowercase = third_after_latin & is_odd;

                        upper = _mm512_mask_sub_epi8(upper, is_lowercase, upper, _mm512_set1_epi8(1));

                        _mm512_mask_storeu_epi8(target, prefix_mask, upper);
                        target += latin_ext_length, source += latin_ext_length, source_length -= latin_ext_length;
                        continue;
                    }
                }
            }
        }

        // Fullwidth lowercase (EF BD 81-9A) → uppercase (EF BC A1-BA)
        // Inverse of fold: third byte -0x20, second byte BD→BC
        {
            __mmask64 is_ef_mask = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xEF));
            if (is_ef_mask) {
                __mmask64 ef_second_positions = is_ef_mask << 1;
                // Check for BD second byte (Fullwidth lowercase)
                __mmask64 is_bd_second = ef_second_positions &
                    _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xBD));

                if (is_bd_second) {
                    __mmask64 is_valid_fullwidth = ~is_non_ascii | is_ef_mask | is_cont_mask;
                    sz_size_t fullwidth_length = sz_ice_first_invalid_(is_valid_fullwidth, load_mask, chunk_size);
                    fullwidth_length -= fullwidth_length && ((is_ef_mask >> (fullwidth_length - 1)) & 1);

                    if (fullwidth_length >= 3) {
                        __mmask64 prefix_mask = sz_u64_mask_until_(fullwidth_length);

                        // Start with source, apply ASCII uppercasing
                        __mmask64 is_lower_ascii = _mm512_cmplt_epu8_mask(
                            _mm512_sub_epi8(source_vec.zmm, a_lower_vec), subtract26_vec);
                        __m512i upper = _mm512_mask_sub_epi8(source_vec.zmm, is_lower_ascii & prefix_mask,
                                                              source_vec.zmm, ascii_case_offset);

                        // Check for BD second byte after EF
                        __mmask64 is_fullwidth_lower = ((is_ef_mask << 1) & prefix_mask) &
                            _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xBD));

                        // Third byte 81-9A → A1-BA (lowercase ａ-ｚ → uppercase Ａ-Ｚ)
                        __mmask64 third_after_bd = (is_fullwidth_lower << 1) & prefix_mask;
                        __mmask64 is_fullwidth_az = third_after_bd &
                            _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x81)),
                                                   _mm512_set1_epi8(0x1A)); // 81-9A (26 letters)

                        // Transform: third +0x20, second BD→BC
                        upper = _mm512_mask_add_epi8(upper, is_fullwidth_az, upper, _mm512_set1_epi8(0x20));
                        upper = _mm512_mask_sub_epi8(upper, is_fullwidth_az >> 1, upper, _mm512_set1_epi8(1)); // BD→BC

                        _mm512_mask_storeu_epi8(target, prefix_mask, upper);
                        target += fullwidth_length, source += fullwidth_length, source_length -= fullwidth_length;
                        continue;
                    }
                }
            }
        }

        // Fallback to serial for complex cases
        {
            sz_rune_t rune;
            sz_rune_length_t rune_length;
            sz_rune_parse((sz_cptr_t)source, &rune, &rune_length);
            source += rune_length;
            source_length -= rune_length;

            sz_rune_t upper_runes[3];
            sz_size_t upper_count = sz_unicode_upper_codepoint_(rune, upper_runes);
            for (sz_size_t i = 0; i != upper_count; ++i)
                target += sz_rune_export(upper_runes[i], target);
        }
    }

    return (sz_size_t)(target - target_start);
}



SZ_PUBLIC sz_bool_t sz_utf8_case_invariant_ice(sz_cptr_t str, sz_size_t length) {
    sz_u8_t const *ptr = (sz_u8_t const *)str;

    // Pre-computed constants
    __m512i const a_upper_vec = _mm512_set1_epi8('A');
    __m512i const a_lower_vec = _mm512_set1_epi8('a');
    __m512i const z26_vec = _mm512_set1_epi8(26);
    __m512i const x80_vec = _mm512_set1_epi8((char)0x80);
    __m512i const xc0_vec = _mm512_set1_epi8((char)0xC0);
    __m512i const xc3_vec = _mm512_set1_epi8((char)0xC3);
    __m512i const xe0_vec = _mm512_set1_epi8((char)0xE0);
    __m512i const xf0_vec = _mm512_set1_epi8((char)0xF0);
    __m512i const xf8_vec = _mm512_set1_epi8((char)0xF8);

    // Single loop: advance by min(length, 61), check leads in first `step` positions
    while (length) {
        sz_size_t step = sz_min_of_two(length, 61);
        __mmask64 lead_mask = sz_u64_mask_until_(step);
        __mmask64 load_mask = sz_u64_clamp_mask_until_(length); // min(length, 64) bits
        __m512i data = _mm512_maskz_loadu_epi8(load_mask, ptr);

        // 1. ASCII letter check (zeros beyond string are fine - not letters)
        __mmask64 is_upper = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(data, a_upper_vec), z26_vec);
        __mmask64 is_lower = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(data, a_lower_vec), z26_vec);
        if (is_upper | is_lower) return sz_false_k;

        // 2. Check for non-ASCII in lead positions
        __mmask64 is_non_ascii_mask = _mm512_movepi8_mask(data) & lead_mask;
        if (is_non_ascii_mask) {
            // 3. Identify UTF-8 lead bytes
            __mmask64 is_two = _mm512_cmpeq_epi8_mask(_mm512_and_si512(data, xe0_vec), xc0_vec) & lead_mask;
            __mmask64 is_three = _mm512_cmpeq_epi8_mask(_mm512_and_si512(data, xf0_vec), xe0_vec) & lead_mask;
            __mmask64 is_four = _mm512_cmpeq_epi8_mask(_mm512_and_si512(data, xf8_vec), xf0_vec) & lead_mask;

            // 4. Check 4-byte bicameral scripts (SMP): F0 with second byte 90/91/96/9D/9E
            if (is_four) {
                __mmask64 f0_sec = is_four << 1;
                __mmask64 is_90 = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8((char)0x90));
                __mmask64 is_91 = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8((char)0x91));
                __mmask64 is_96 = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8((char)0x96));
                __mmask64 is_9d = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8((char)0x9D));
                __mmask64 is_9e = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8((char)0x9E));
                if (f0_sec & (is_90 | is_91 | is_96 | is_9d | is_9e)) return sz_false_k;
            }

            // 5. Check 2-byte bicameral leads: C3-D6
            // C3-CF: Latin Extended (umlauts, accents, Eszett)
            // D0-D1: Cyrillic, D4-D6: Armenian (D6 needed for small letters U+0580+)
            if (is_two) {
                __mmask64 is_bicameral = _mm512_cmplt_epu8_mask( //
                    _mm512_sub_epi8(data, xc3_vec), _mm512_set1_epi8(0x14));

                // Special case: C2 B5 = U+00B5 MICRO SIGN folds to Greek mu (U+03BC)
                __mmask64 is_c2 = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8((char)0xC2)) & is_two;
                if (is_c2) {
                    __mmask64 c2_sec = is_c2 << 1;
                    __mmask64 is_b5 = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8((char)0xB5));
                    if (c2_sec & is_b5) return sz_false_k;
                }

                // Note: CA 80-BF includes both IPA Extensions (U+0280-02AF) and Spacing Modifier Letters
                // (U+02B0-02BF). Spacing Modifier Letters CAN appear in case fold expansions:
                // e.g., ẚ (U+1E9A) folds to [a, ʾ] where ʾ = U+02BE is a Spacing Modifier Letter.
                // So we must NOT exclude this range from bicameral check.

                if (is_bicameral & is_two) return sz_false_k;
            }

            // 6. Check 3-byte bicameral sequences
            if (is_three) {
                // E1: Georgian, Greek Extended, Latin Extended Additional
                __mmask64 is_e1 = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8((char)0xE1));
                if (is_e1 & is_three) return sz_false_k;

                // EF: Fullwidth Latin
                __mmask64 is_ef = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8((char)0xEF));
                if (is_ef & is_three) return sz_false_k;

                // E2: Safe only for second byte 80-83
                __mmask64 is_e2 = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8((char)0xE2)) & is_three;
                if (is_e2) {
                    __mmask64 e2_sec = is_e2 << 1;
                    __mmask64 safe = _mm512_cmplt_epu8_mask( //
                        _mm512_sub_epi8(data, x80_vec), _mm512_set1_epi8(0x04));
                    if (e2_sec & ~safe) return sz_false_k;
                }

                // EA: Bicameral second bytes 99-9F, AD-AE
                __mmask64 is_ea = _mm512_cmpeq_epi8_mask(data, _mm512_set1_epi8((char)0xEA)) & is_three;
                if (is_ea) {
                    __mmask64 ea_sec = is_ea << 1;
                    __mmask64 is_99 = _mm512_cmplt_epu8_mask( //
                        _mm512_sub_epi8(data, _mm512_set1_epi8((char)0x99)), _mm512_set1_epi8(0x07));
                    __mmask64 is_ad = _mm512_cmplt_epu8_mask( //
                        _mm512_sub_epi8(data, _mm512_set1_epi8((char)0xAD)), _mm512_set1_epi8(0x02));
                    if (ea_sec & (is_99 | is_ad)) return sz_false_k;
                }
            }
        }

        ptr += step;
        length -= step;
    }

    return sz_true_k;
}

#pragma region Character Safety Profiles

/**
 *  @brief Safety profile for a single character across all script paths.
 *
 *  A safety profile for a "needle" is a set of conditions that allow simpler haystack on-the-fly folding
 *  than the proper `sz_utf8_case_fold`, but without losing any possible matches. That's typically achieved
 *  finding parts of the needle, that never appear in any multi-byte expansions of complex characters, so
 *  we don't need to shuffle data within a CPU register - just swap some byte sequences with others.
 *
 *  Assuming the complexity of Unicode, the number of such rules to take care of is quite significant, so
 *  it's hard to achieve matching speeds beyond 500 MB/s for arbitrary needles. However, if separate them
 *  by language groups and Unicode subranges, the 5 GB/s target becomes approachable.
 */
typedef enum {
    sz_utf8_case_rune_unknown_k = 0,
    /**
     *  @brief  Describes a safety-class profile for contextually-safe ASCII characters, mostly for English text,
     *          exclusive to single-byte characters without case-folding "collisions" and ambiguities.
     *
     *  If all of the following @b needle-constraints are satisfied, our case-insensitive UTF-8 substring search
     *  becomes no more than a trivial case-insensitive ASCII substring search, where the only @b haystack-folding
     *  operation to be applied is mapping A-Z to a-z:
     *
     *  - 'a' (U+0061, 61) - can't be last; can't precede 'ʾ' (U+02BE, CA BE) to avoid:
     *    - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
     *  - 'f' (U+0066, 66) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede 'f' (U+0066, 66), 'i' (U+0069, 69), 'l' (U+006C, 6C) to avoid:
     *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'h' (U+0068, 68) - can't be last; can't precede '̱' (U+0331, CC B1) to avoid:
     *    - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede '̇' (U+0307, CC 87) to avoid:
     *    - 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *  - 'j' (U+006A, 6A) - can't be last; can't precede '̌' (U+030C, CC 8C) to avoid:
     *    - 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C)
     *  - 'k' (U+006B, 6B) - can't be present at all, because it's a folding target of the Kelvin sign:
     *    - 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B)
     *  - 'l' (U+006C, 6C) - can't be first; can't follow 'f' (U+0066, 66) to avoid:
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'n' (U+006E, 6E) - can't be first; can't follow 'ʼ' (U+02BC, CA BC) to avoid:
     *    - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
     *  - 's' (U+0073, 73) - can't be present at all, because it's a folding target of the old S sign:
     *    - 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73);
     *    can't precede '̈' (U+0308, CC 88) to avoid:
     *    - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 'w' (U+0077, 77) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
     *  - 'y' (U+0079, 79) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
     *
     *  This means, that all ASCII characters beyond the rules above are considered "safe" for this profile.
     *  This includes English alphabet letters like: b, c, d, e, g, m, o, p, q, r, u, v, x, z,
     *  as well as digits, punctuation, symbols, and control characters.
     */
    sz_utf8_case_rune_ascii_invariant_k = 1,

    /**
     *  @brief  Describes a safety-class profile for contextually-safe ASCII + Latin-1 Supplements designed mostly
     *          for Western European languages (like French, German, Spanish, & Portuguese) with a mixture of
     *          single-byte and double-byte UTF-8 character sequences.
     *
     *  @sa sz_utf8_case_rune_ascii_invariant_k for a simpler variant.
     *
     *  Unlike the ASCII fast path, these kernels fold a wider range of characters:
     *  - 26x original ASCII uppercase letters: 'A' (U+0041, 41) → 'a' (U+0061, 61), 'Z' (U+005A, 5A) → 'z' (U+007A, 7A)
     *  - 30x Latin-1 supplement uppercase letters for French, German, Spanish, & Portuguese, like:
     *    - 'À' (U+00C0, C3 80) → 'à' (U+00E0, C3 A0),
     *    - 'Ñ' (U+00D1, C3 91) → 'ñ' (U+00F1, C3 B1),
     *    - 'Ü' (U+00DC, C3 9C) → 'ü' (U+00FC, C3 BC)
     *  - 1x special case of folding from Latin-1 to ASCII pair, preserving byte-width:
     *    - 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
     *
     *  This doesn't cover Latin-A and Latin-B extensions (like Polish, Czech, Hungarian, & Turkish letters).
     *  This also inherits some of the contextual limitations from `sz_utf8_case_rune_ascii_invariant_k`, but not all!
     *
     *  The lowercase 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73) is folded in-place (2 bytes → 2 bytes).
     *  This creates a mid-expansion matching issue: if a needle starts or ends with 's', the SIMD kernel might find
     *  a match at the second byte of 'ß' (the UTF-8 continuation byte 0x9F) instead of at a codepoint boundary.
     *  Example: haystack "ßStra" folds to "ssstra", needle "sstra" matches at position 1 (the 0x9F byte of 'ß').
     *  To avoid this, 's' is only safe when NOT at the start or end of the needle (contextual restriction).
     *
     *  The uppercase 'ẞ' (U+1E9E, E1 BA 9E) also folds into "ss" (U+0073 U+0073, 73 73), but is outside of Latin-1.
     *  In UTF-8 it is a 3-byte sequence, so it resizes into a 2-byte sequence when folded. Luckily for us,
     *  it's almost never used in practice: introduced to Unicode in 2008 and officially adopted into German
     *  orthography in 2017. When processing the haystack, we check if 'ẞ' appears, and if so, we revert to
     *  serial processing for that tiny block of text.
     *
     *  Another place where 's' (U+0073, 73) appears are ligatures 'ﬅ' (U+FB05, EF AC 85) and 'ﬆ' (U+FB06, EF AC 86)
     *  that both fold into "st" (U+0073 U+0074, 73 74). They also result in serial fallback when detected in the
     *  haystack. If we detect all of those ligatures from 'ﬀ' (U+FB00, EF AC 80) to 'ﬆ' (U+FB06, EF AC 86), we can
     *  safely allow both 'f' (U+0066, 66) and 'l' (U+006C, 6C).
     *
     *  There is one more 3-byte problematic range to consider - from (E1 BA 96) to (E1 BA 9A), which includes:
     *  'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1), 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88),
     *  'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A), 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A),
     *  'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE). If we correctly detect that range in the haystack, we
     *  can safely allow 'h' (U+0068, 68), 't' (U+0074, 74), 'w' (U+0077, 77), 'y' (U+0079, 79), and 'a' (U+0061, 61) in
     *  needles!
     *
     *  There is also a Unicode rule for folding the Kelvin 'K' (U+212A, E2 84 AA) into 'k' (U+006B, 6B).
     *  That sign is extremely rare in Western European languages, while the lowercase 'k' is obviously common
     *  in German and English. In French, Spanish, and Portuguese - less so. So we add one more check
     *  for 'K' (U+212A, E2 84 AA) in the haystack, and if detected, again - revert to serial.
     *  Similarly, we check for 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73).
     *  It's archaic in modern languages but theoretically possible in historical texts.
     *
     *  So we allow 'k' unconditionally and inherit/extend the following limitations from
     *  `sz_utf8_case_rune_ascii_invariant_k`:
     *
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66); can't precede '̇' (U+0307, CC 87)
     *    to avoid 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87).
     *    It's the Turkish dotted capital I that expands into a 3-byte sequence when folded. It typically appears
     *    at the start of words, like: İstanbul (the city), İngilizce (English language).
     *
     *  - 'j' (U+006A, 6A) - can't be last; can't precede '̌' (U+030C, CC 8C)
     *    to avoid: 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C).
     *    It's the "J with Caron", used in phonetic transcripts and romanization of Iranian, Armenian, Georgian.
     *
     *  - 'n' (U+006E, 6E) - can't be first; can't follow 'ʼ' (U+02BC, CA BC)
     *    to avoid: 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E).
     *    It's mostly used in Afrikaans (South Africa/Namibia), contracted from Dutch "een" (one/a), in phrases
     *    like "Dit is 'n boom" (It is a tree), "Dit is 'n appel" (This is an apple).
     *
     *  - 's' (U+0073, 73) - can't be first or last, or part of the folded "ss" (U+0073 U+0073, 73 73) prefix or suffix,
     *    to avoid mid-ß-expansion matches: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73) is folded in-place,
     *    so a needle starting/ending with 's' could match at position 1 (the 0x9F continuation byte).
     *    Example: "ßStra" → "ssstra", needle "sstra" would match at the second byte of 'ß'.
     *    Needles with 's' in the middle are safe.
     *
     *  We also add one more limitation for a special 2-byte character that is an irregular folding target of
     *  codepoints of different length:
     *
     *  - 'å' (U+00E5, C3 A5) - is the folding target of both 'Å' (U+00C5, C3 85) in Latin-1 and
     *    the Angstrom Sign 'Å' (U+212B, E2 84 AB) → 'å' (U+00E5, C3 A5), so needle cannot contain 'å' (U+00E5, C3 A5)
     *    to avoid ambiguity.
     *
     *  There is also a Latin-1 character that doesn't change the width, but we still ban it from the safe strings:
     *
     *  - 'µ' (U+00B5, C2 B5) - the mathematical Micro sign folds to the Greek lowercase 'μ' (U+03BC, CE BC), which
     *    is also a folding target of the uppercase Greek letter 'Μ' (U+039C, CE 9C). To avoid having to filter/check
     *    for Greek symbols in the haystacks, we ban the Micro sign from the needles.
     *
     *  This means, that all of the ASCII and Latin-1 characters beyond the rules above are considered "safe"
     *  for this profile. This includes English alphabet letters like: b, c, d, e, g, k, m, o, p, q, r, u, v, x, z,
     *  as well as digits, punctuation, symbols, and control characters.
     */
    sz_utf8_case_rune_safe_western_europe_k = 2,

    /**
     *  @brief  Describes a safety-class profile for contextually-safe ASCII + Latin-1 + Latin-A Supplements designed
     *          mostly for Central European languages (like Polish, Czech, & Hungarian) and Turkish with a mixture of
     *          single-byte, double-byte, and rare triple-byte UTF-8 character sequences.
     *
     *  @sa sz_utf8_case_rune_ascii_invariant_k for a simpler variant.
     *
     *  Unlike the ASCII fast path, these kernels fold a wider range of characters:
     *  - 26x original ASCII uppercase letters: 'A' (U+0041, 41) → 'a' (U+0061, 61), 'Z' (U+005A, 5A) → 'z' (U+007A, 7A)
     *  - 30x Latin-1 supplement uppercase letters for French, German, Spanish, & Portuguese, like:
     *    - 'À' (U+00C0, C3 80) → 'à' (U+00E0, C3 A0),
     *    - 'Ñ' (U+00D1, C3 91) → 'ñ' (U+00F1, C3 B1),
     *    - 'Ü' (U+00DC, C3 9C) → 'ü' (U+00FC, C3 BC)
     *  - 63x Latin-A extension uppercase letters for Polish, Czech, Hungarian, & Turkish, like:
     *    - 'Ą' (U+0104, C4 84) → 'ą' (U+0105, C4 85),
     *    - 'Ł' (U+0141, C5 81) → 'ł' (U+0142, C5 82),
     *    - 'Č' (U+010C, C4 8C) → 'č' (U+010D, C4 8D)
     *
     *  This doesn't cover Latin-B extensions (like Baltic, Romanian, & Vietnamese letters), and is not optimal
     *  for Western European languages, assuming the lack of "ss" handling for German Eszett 'ß' (U+00DF, C3 9F).
     *  There is, however, a huge overlap between the Central European, Western European, and Turkic scripts:
     *
     *  - Czech has the highest overlap - nearly half of Czech words with Latin-A characters (like Č, Ř, Š, Ž)
     *    also contain Latin-1 characters (Á, É, Í, Ó, Ú, Ý). Examples: sčítání, dalšími, řízení, systémů.
     *  - Polish has minimal word-level overlap because Polish only uses Ó/ó from Latin-1, and most Polish-specific
     *    letters (Ą, Ę, Ł, Ń, Ś, Ź, Ż) are in Latin-A. Example: mieszkańców (has both ń and ó).
     *  - Turkish has moderate overlap from Ç, Ö, Ü (Latin-1) mixing with Ğ, İ, Ş (Latin-A).
     *    Examples: içeriği, öğrencilerden, dönüşüm.
     *
     *  All those languages are not always related linguistically:
     *
     *  - Czech and Polish are Slavic languages, that use Latin script with háčeks since 15th century.
     *  - Hungarian is a Uralic language, that adopted Latin script in 11th century.
     *  - Turkish is a Turkic (Altaic) language, that switched from Arabic to Latin script in 1928.
     *    Atatürk's 1928 alphabet reform:
     *    - borrowed Ç, Ö, Ü from French and German subsets of Latin-1 Supplement (C3 lead byte).
     *    - introduced Ğ, İ, Ş, which ended up in the Latin Extended-A (C4/C5 lead byte).
     *
     *  But due to overlapping character sets, they can all benefit from the same fast path.
     *
     *  There is also a Unicode rule for folding the Kelvin 'K' (U+212A, E2 84 AA) into 'k' (U+006B, 6B).
     *  That sign is extremely rare in Western European languages, while the lowercase 'k' is very common in Turkish,
     *  Czech, Polish. So we add one more check for 'K' (U+212A, E2 84 AA) in the haystack, and if detected,
     *  again - revert to serial. Same logic applies to 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73).
     *
     *  The Turkish dotted 'İ' (U+0130, C4 B0) expands into a 3-byte sequence. We detect it when scanning through the
     *  haystack and fall back to the serial algorithm. That's pretty much the only triple-byte sequence we will
     *  frequently encounter in Turkish text.
     *
     *  We inherit most contextual limitations for some of the ASCII characters from
     *  `sz_utf8_case_rune_ascii_invariant_k`:
     *
     *  - 'a' (U+0061, 61) - can't be last; can't precede 'ʾ' (U+02BE, CA BE) to avoid:
     *    - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
     *  - 'f' (U+0066, 66) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede 'f' (U+0066, 66), 'i' (U+0069, 69), 'l' (U+006C, 6C) to avoid:
     *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'h' (U+0068, 68) - can't be last; can't precede '̱' (U+0331, CC B1) to avoid:
     *    - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede '̇' (U+0307, CC 87) to avoid:
     *    - 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *  - 'j' (U+006A, 6A) - can't be last; can't precede '̌' (U+030C, CC 8C) to avoid:
     *    - 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C)
     *  - 'l' (U+006C, 6C) - can't be first; can't follow 'f' (U+0066, 66) to avoid:
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'n' (U+006E, 6E) - can't be first; can't follow 'ʼ' (U+02BC, CA BC) to avoid:
     *    - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
     *  - 's' (U+0073, 73) - can't be first or last; can't follow 's' (U+0073, 73);
     *    can't precede 's' (U+0073, 73), 't' (U+0074, 74) to avoid:
     *    - 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
     *    - 'ẞ' (U+1E9E, E1 BA 9E) → "ss" (U+0073 U+0073, 73 73)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73);
     *    can't precede '̈' (U+0308, CC 88) to avoid:
     *    - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 'w' (U+0077, 77) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
     *  - 'y' (U+0079, 79) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
     *
     *  We also inherit one more limitation from the Latin-1 profile, same as `sz_utf8_case_rune_safe_western_europe_k`:
     *
     *  - 'å' (U+00E5, C3 A5) - is the folding target of both 'Å' (U+00C5, C3 85) in Latin-1 and
     *    the Angstrom Sign 'Å' (U+212B, E2 84 AB) → 'å' (U+00E5, C3 A5), so needle cannot contain 'å' (U+00E5, C3 A5)
     *    to avoid ambiguity.
     *
     *  This means, that all of the ASCII and Latin-1 characters beyond the rules above are considered "safe"
     *  for this profile. This includes English alphabet letters like: b, c, d, e, g, k, m, o, p, q, r, u, v, x, z,
     *  as well as digits, punctuation, symbols, and control characters.
     */
    sz_utf8_case_rune_safe_central_europe_k = 3,

    /**
     *  @brief  Describes a safety-class profile for contextually-safe ASCII + Basic Cyrillic designed mostly
     *          for East Slavic languages (like Russian, Ukrainian, & Belarusian) and South Slavic languages
     *          (like Serbian, Bulgarian, & Macedonian), but excluding Cyrillic Extensions.
     *
     *  @sa sz_utf8_case_rune_ascii_invariant_k for the inherited ASCII rules.
     *
     *  Unlike the ASCII fast path, these kernels fold a wider range of characters:
     *  - 26x original ASCII uppercase letters: 'A' (U+0041, 41) → 'a' (U+0061, 61), 'Z' (U+005A, 5A) → 'z' (U+007A, 7A)
     *  - 32x Basic Cyrillic uppercase letters:
     *    - 'А' (U+0410, D0 90) → 'а' (U+0430, D0 B0) through 'П' (U+041F, D0 9F) → 'п' (U+043F, D0 BF)
     *    - 'Р' (U+0420, D0 A0) → 'р' (U+0440, D1 80) through 'Я' (U+042F, D0 AF) → 'я' (U+044F, D1 8F)
     *  - 16x Cyrillic extensions for non-Russian Slavic languages:
     *    - 'Ѐ' (U+0400, D0 80) → 'ѐ' (U+0450, D1 90) - Cyrillic E with grave (Macedonian, Serbian)
     *    - 'Ё' (U+0401, D0 81) → 'ё' (U+0451, D1 91) - Cyrillic IO (Russian, Belarusian)
     *    - 'Ђ' (U+0402, D0 82) → 'ђ' (U+0452, D1 92) - Cyrillic DJE (Serbian)
     *    - 'Ѓ' (U+0403, D0 83) → 'ѓ' (U+0453, D1 93) - Cyrillic GJE (Macedonian)
     *    - 'Є' (U+0404, D0 84) → 'є' (U+0454, D1 94) - Cyrillic Ukrainian IE (Ukrainian)
     *    - 'Ѕ' (U+0405, D0 85) → 'ѕ' (U+0455, D1 95) - Cyrillic DZE (Macedonian)
     *    - 'І' (U+0406, D0 86) → 'і' (U+0456, D1 96) - Cyrillic Byelorussian-Ukrainian I (Ukrainian, Belarusian)
     *    - 'Ї' (U+0407, D0 87) → 'ї' (U+0457, D1 97) - Cyrillic YI (Ukrainian)
     *    - 'Ј' (U+0408, D0 88) → 'ј' (U+0458, D1 98) - Cyrillic JE (Serbian, Macedonian)
     *    - 'Љ' (U+0409, D0 89) → 'љ' (U+0459, D1 99) - Cyrillic LJE (Serbian, Macedonian)
     *    - 'Њ' (U+040A, D0 8A) → 'њ' (U+045A, D1 9A) - Cyrillic NJE (Serbian, Macedonian)
     *    - 'Ћ' (U+040B, D0 8B) → 'ћ' (U+045B, D1 9B) - Cyrillic TSHE (Serbian)
     *    - 'Ќ' (U+040C, D0 8C) → 'ќ' (U+045C, D1 9C) - Cyrillic KJE (Macedonian)
     *    - 'Ѝ' (U+040D, D0 8D) → 'ѝ' (U+045D, D1 9D) - Cyrillic I with grave (Bulgarian, Macedonian)
     *    - 'Ў' (U+040E, D0 8E) → 'ў' (U+045E, D1 9E) - Cyrillic short U (Belarusian)
     *    - 'Џ' (U+040F, D0 8F) → 'џ' (U+045F, D1 9F) - Cyrillic DZHE (Serbian, Macedonian)
     *
     *  UTF-8 byte patterns for Basic Cyrillic (D0/D1 lead bytes):
     *  - D0 80-8F: Extensions uppercase 'Ѐ'-'Џ' (U+0400-U+040F) → fold to D1 90-9F
     *  - D0 90-9F: Basic uppercase 'А'-'П' (U+0410-U+041F) → fold to D0 B0-BF (same lead byte)
     *  - D0 A0-AF: Basic uppercase 'Р'-'Я' (U+0420-U+042F) → fold to D1 80-8F (cross lead byte)
     *  - D0 B0-BF: Basic lowercase 'а'-'п' (U+0430-U+043F)
     *  - D1 80-8F: Basic lowercase 'р'-'я' (U+0440-U+044F)
     *  - D1 90-9F: Extensions lowercase 'ѐ'-'џ' (U+0450-U+045F)
     *
     *  We entirely ban all of the Extended Cyrillic (D2/D3 lead bytes), sometimes used in Ukranian,
     *  Kazakh, and Uzbek languages, like the 'Ґ' (U+0490, D2 90) → 'ґ' (U+0491, D2 91) folding with even/odd ordering
     *  of uppercase and lowercase. Similar rules apply to some Chechen, and various Turkic languages.
     *  But there are also exceptions, like the Palochka 'Ӏ' (U+04C0, D3 80) → 'ӏ' (U+04CF, D3 8F).
     *  By omitting those extensions we can make our folding kernel much lighter.
     *
     *  We inherit ALL contextual ASCII limitations from `sz_utf8_case_rune_ascii_invariant_k`:
     *
     *  - 'a' (U+0061, 61) - can't be last; can't precede 'ʾ' (U+02BE, CA BE) to avoid:
     *     - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
     *  - 'f' (U+0066, 66) - can't be first or last; can't follow 'f' (U+0066, 66);
     *     can't precede 'f' (U+0066, 66), 'i' (U+0069, 69), 'l' (U+006C, 6C) to avoid:
     *     - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
     *     - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *     - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *     - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *     - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'h' (U+0068, 68) - can't be last; can't precede '̱' (U+0331, CC B1) to avoid:
     *     - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66);
     *     can't precede '̇' (U+0307, CC 87) to avoid:
     *     - 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
     *     - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *     - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *  - 'j' (U+006A, 6A) - can't be last; can't precede '̌' (U+030C, CC 8C) to avoid:
     *     - 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C)
     *  - 'k' (U+006B, 6B) - can't be present at all, because it's a folding target of the Kelvin sign:
     *     - 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B)
     *  - 'l' (U+006C, 6C) - can't be first; can't follow 'f' (U+0066, 66) to avoid:
     *     - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *     - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'n' (U+006E, 6E) - can't be first; can't follow 'ʼ' (U+02BC, CA BC) to avoid:
     *     - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
     *  - 's' (U+0073, 73) - can't be present at all, because it's a folding target of the old S sign:
     *    - 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73);
     *     can't precede '̈' (U+0308, CC 88) to avoid:
     *     - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *     - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *     - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73);
     *     can't precede '̈' (U+0308, CC 88) to avoid:
     *     - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *     - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *     - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 'w' (U+0077, 77) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *     - 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
     *  - 'y' (U+0079, 79) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *     - 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
     *
     *  These ASCII constraints are necessary because mixed-script documents (Cyrillic + Latin) may contain
     *  Latin ligatures, German Eszett, or Turkish İ that the Cyrillic fold function doesn't handle.
     *
     *  This means, that all ASCII characters beyond the rules above are considered "safe" for this profile.
     *  This includes English alphabet letters like: b, c, d, e, g, m, o, p, q, r, u, v, x, z,
     *  as well as digits, punctuation, symbols, and control characters.
     */
    sz_utf8_case_rune_safe_cyrillic_k = 4,

    /**
     *  @brief  Describes a safety-class profile for contextually-safe ASCII + Basic Greek designed mostly
     *          for Modern Greek (Demotic) text with a mixture of single-byte and double-byte UTF-8
     *          character sequences.
     *
     *  @sa sz_utf8_case_rune_ascii_invariant_k for the inherited ASCII rules.
     *
     *  Unlike the ASCII fast path, these kernels fold a wider range of characters:
     *  - 26x original ASCII uppercase letters: 'A' (U+0041, 41) → 'a' (U+0061, 61), 'Z' (U+005A, 5A) → 'z' (U+007A, 7A)
     *  - 24x Basic Greek uppercase letters (monotonic, without diacritics):
     *    - 'Α' (U+0391, CE 91) → 'α' (U+03B1, CE B1) through 'Ο' (U+039F, CE 9F) → 'ο' (U+03BF, CE BF)
     *    - 'Π' (U+03A0, CE A0) → 'π' (U+03C0, CF 80) through 'Ω' (U+03A9, CE A9) → 'ω' (U+03C9, CF 89)
     *  - 1x Final sigma to regular sigma:
     *    - 'ς' (U+03C2, CF 82) → 'σ' (U+03C3, CF 83)
     *  - 7x Greek accented uppercase letters (tonos only, modern orthography):
     *    - 'Ά' (U+0386, CE 86) → 'ά' (U+03AC, CE AC)
     *    - 'Έ' (U+0388, CE 88) → 'έ' (U+03AD, CE AD)
     *    - 'Ή' (U+0389, CE 89) → 'ή' (U+03AE, CE AE)
     *    - 'Ί' (U+038A, CE 8A) → 'ί' (U+03AF, CE AF)
     *    - 'Ό' (U+038C, CE 8C) → 'ό' (U+03CC, CF 8C)
     *    - 'Ύ' (U+038E, CE 8E) → 'ύ' (U+03CD, CF 8D)
     *    - 'Ώ' (U+038F, CE 8F) → 'ώ' (U+03CE, CF 8E)
     *  - 2x Greek uppercase letters with dialytika:
     *    - 'Ϊ' (U+03AA, CE AA) → 'ϊ' (U+03CA, CF 8A)
     *    - 'Ϋ' (U+03AB, CE AB) → 'ϋ' (U+03CB, CF 8B)
     *
     *  UTF-8 byte patterns for Basic Greek (CE/CF lead bytes):
     *  - CE 86-8F: Accented uppercase 'Ά'-'Ώ' (with gaps) → CE AC-AF or CF 8C-8E
     *  - CE 91-9F: Basic uppercase 'Α'-'Ο' (U+0391-U+039F) → CE B1-BF (same lead byte)
     *  - CE A0-A9: Basic uppercase 'Π'-'Ω' (U+03A0-U+03A9) → CF 80-89 (cross lead byte)
     *  - CE AA-AB: Dialytika uppercase 'Ϊ'-'Ϋ' (U+03AA-U+03AB) → CF 8A-8B (cross lead byte)
     *  - CE AC-AF: Accented lowercase 'ά'-'ί' (U+03AC-U+03AF)
     *  - CE B1-BF: Basic lowercase 'α'-'ο' (U+03B1-U+03BF)
     *  - CF 80-89: Basic lowercase 'π'-'ω' (U+03C0-U+03C9), includes 'ς' (U+03C2, CF 82) and 'σ' (U+03C3, CF 83)
     *  - CF 8A-8E: Accented/dialytika lowercase 'ϊ'-'ώ' (U+03CA-U+03CE)
     *
     *  Greek symbol variants that fold to basic letters (detected in haystack, serial fallback):
     *  - 'ϐ' (U+03D0, CF 90) → 'β' (U+03B2, CE B2) - Greek Beta Symbol
     *  - 'ϑ' (U+03D1, CF 91) → 'θ' (U+03B8, CE B8) - Greek Theta Symbol
     *  - 'ϕ' (U+03D5, CF 95) → 'φ' (U+03C6, CF 86) - Greek Phi Symbol
     *  - 'ϖ' (U+03D6, CF 96) → 'π' (U+03C0, CF 80) - Greek Pi Symbol
     *  - 'ϰ' (U+03F0, CF B0) → 'κ' (U+03BA, CE BA) - Greek Kappa Symbol
     *  - 'ϱ' (U+03F1, CF B1) → 'ρ' (U+03C1, CF 81) - Greek Rho Symbol
     *  - 'ϵ' (U+03F5, CF B5) → 'ε' (U+03B5, CE B5) - Greek Lunate Epsilon Symbol
     *
     *  Excluded from the needle (require serial fallback when detected in haystack):
     *
     *  - 'ΐ' (U+0390, CE 90) → "ΐ" (U+03B9 U+0308 U+0301, CE B9 CC 88 CC 81) - iota with dialytika and tonos
     *    EXPANDS to "ΐ" (U+03B9 U+0308 U+0301) - 3 codepoints!
     *  - 'ΰ' (U+03B0, CE B0) → "ΰ" (U+03C5 U+0308 U+0301, CF 85 CC 88 CC 81) - upsilon with dialytika and tonos
     *    EXPANDS to "ΰ" (U+03C5 U+0308 U+0301) - 3 codepoints!
     *  - Greek Extended / Polytonic (U+1F00-U+1FFF, E1 BC-BF lead bytes):
     *    Ancient Greek with breathing marks, accents, and iota subscript. Many expand to multiple
     *    codepoints, e.g., 'ᾈ' (U+1F88) → "ἀι" (U+1F00 U+03B9, E1 BC 80 CE B9), 'ᾳ' (U+1FB3) → "αι" (U+03B1 U+03B9, CE
     * B1 CE B9). Polytonic Greek is used primarily in academic, religious, and historical texts.
     *
     *  Note on the Micro Sign 'µ' (U+00B5, C2 B5):
     *  The Latin-1 micro sign folds TO Greek mu 'μ' (U+03BC, CE BC). This is handled by the Latin-1
     *  kernel path (sz_utf8_case_rune_safe_western_europe_k), not the Greek path. The Greek kernel
     *  only handles characters that originate in the Greek block.
     *
     *  We inherit @b all contextual ASCII limitations from `sz_utf8_case_rune_ascii_invariant_k`:
     *
     *  - 'a' (U+0061, 61) - can't be last; can't precede 'ʾ' (U+02BE, CA BE) to avoid:
     *    - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
     *  - 'f' (U+0066, 66) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede 'f' (U+0066, 66), 'i' (U+0069, 69), 'l' (U+006C, 6C) to avoid:
     *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'h' (U+0068, 68) - can't be last; can't precede '̱' (U+0331, CC B1) to avoid:
     *    - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede '̇' (U+0307, CC 87) to avoid:
     *    - 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *  - 'j' (U+006A, 6A) - can't be last; can't precede '̌' (U+030C, CC 8C) to avoid:
     *    - 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C)
     *  - 'k' (U+006B, 6B) - can't be present at all, because it's a folding target of the Kelvin sign:
     *    - 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B)
     *  - 'l' (U+006C, 6C) - can't be first; can't follow 'f' (U+0066, 66) to avoid:
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'n' (U+006E, 6E) - can't be first; can't follow 'ʼ' (U+02BC, CA BC) to avoid:
     *    - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
     *  - 's' (U+0073, 73) - can't be present at all, because it's a folding target of the old S sign:
     *    - 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73);
     *    can't precede '̈' (U+0308, CC 88) to avoid:
     *    - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73);
     *    can't precede '̈' (U+0308, CC 88) to avoid:
     *    - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 'w' (U+0077, 77) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
     *  - 'y' (U+0079, 79) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
     *
     *  These ASCII constraints are necessary because mixed-script documents (Greek + Latin) are common
     *  in scientific notation, brand names, and modern Greek text with English loanwords.
     *
     *  This means, that all ASCII characters beyond the rules above are considered "safe" for this profile.
     *  This includes English alphabet letters like: b, c, d, e, g, m, o, p, q, r, u, v, x, z,
     *  as well as digits, punctuation, symbols, and control characters.
     */
    sz_utf8_case_rune_safe_greek_k = 5,

    /**
     *  @brief  Describes a safety-class profile for contextually-safe ASCII + Basic Armenian.
     *  @sa sz_utf8_case_rune_ascii_invariant_k for the inherited ASCII rules.
     *
     *  These kernels fold:
     *  - 26x ASCII uppercase letters: 'A' (U+0041, 41) → 'a' (U+0061, 61), 'Z' (U+005A, 5A) → 'z' (U+007A, 7A)
     *  - 38x Armenian uppercase letters: '
     *    - 'Ա' (U+0531, D4 B1) → 'ա' (U+0561, D5 A1)
     *    - 'Ֆ' (U+0556, D5 96) → 'ֆ' (U+0586, D6 86)
     *
     *  UTF-8 byte ranges handled:
     *  - D4 B1-BF: uppercase 'Ա' (U+0531) through 'Ձ' (U+053F)
     *  - D5 80-96: uppercase 'Ղ' (U+0540) through 'Ֆ' (U+0556)
     *  - D5 A1-BF: lowercase 'ա' (U+0561) through 'ի' (U+057F)
     *  - D6 80-86: lowercase 'լ' (U+0580) through 'ֆ' (U+0586)
     *
     *  We inherit @b all contextual ASCII limitations from `sz_utf8_case_rune_ascii_invariant_k`:
     *
     *  - 'a' (U+0061, 61) - can't be last; can't precede 'ʾ' (U+02BE, CA BE) to avoid:
     *    - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
     *  - 'f' (U+0066, 66) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede 'f' (U+0066, 66), 'i' (U+0069, 69), 'l' (U+006C, 6C) to avoid:
     *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'h' (U+0068, 68) - can't be last; can't precede '̱' (U+0331, CC B1) to avoid:
     *    - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede '̇' (U+0307, CC 87) to avoid:
     *    - 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *  - 'j' (U+006A, 6A) - can't be last; can't precede '̌' (U+030C, CC 8C) to avoid:
     *    - 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C)
     *  - 'k' (U+006B, 6B) - can't be present at all, because it's a folding target of the Kelvin sign:
     *    - 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B)
     *  - 'l' (U+006C, 6C) - can't be first; can't follow 'f' (U+0066, 66) to avoid:
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'n' (U+006E, 6E) - can't be first; can't follow 'ʼ' (U+02BC, CA BC) to avoid:
     *    - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
     *  - 's' (U+0073, 73) - can't be present at all, because it's a folding target of the old S sign:
     *    - 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73);
     *    can't precede '̈' (U+0308, CC 88) to avoid:
     *    - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73);
     *    can't precede '̈' (U+0308, CC 88) to avoid:
     *    - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 'w' (U+0077, 77) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
     *  - 'y' (U+0079, 79) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
     *
     *  We also add rules specific to Armenian ligatures:
     *
     *  - 'և' (U+0587, D6 87) → "եւ" (U+0565 U+0582, D5 A5 D6 82) - very common
     *  - 'ﬓ' (U+FB13, EF AC 93) → "մն" (U+0574 U+0576, D5 B4 D5 B6) - quite rare
     *  - 'ﬔ' (U+FB14, EF AC 94) → "մե" (U+0574 U+0565, D5 B4 D5 A5) - quite rare
     *  - 'ﬕ' (U+FB15, EF AC 95) → "մի" (U+0574 U+056B, D5 B4 D5 AB) - quite rare
     *  - 'ﬖ' (U+FB16, EF AC 96) → "վն" (U+057E U+0576, D5 BE D5 B6) - quite rare
     *  - 'ﬗ' (U+FB17, EF AC 97) → "մխ" (U+0574 U+056D, D5 B4 D5 AD) - quite rare
     *
     *  Specific constraints by character:
     *
     *  - 'ե' (U+0565, D5 A5) - can't be first; can't follow 'մ' (U+0574, D5 B4);
     *     can't precede 'ւ' (U+0582, D6 82) to avoid:
     *     - 'և' (U+0587, D6 87) → "եւ" (U+0565 U+0582, D5 A5 D6 82)
     *     - 'ﬔ' (U+FB14, EF AC 94) → "մե" (U+0574 U+0565, D5 B4 D5 A5)
     *  - 'ւ' (U+0582, D6 82) - can't be last; can't follow 'ե' (U+0565, D5 A5) to avoid:
     *     - 'և' (U+0587, D6 87) → "եւ" (U+0565 U+0582, D5 A5 D6 82)
     *  - 'մ' (U+0574, D5 B4) - can't be last; can't precede 'ն' (U+0576, D5 B6), 'ե' (U+0565, D5 A5),
     *     'ի' (U+056B, D5 AB), 'խ' (U+056D, D5 AD) to avoid:
     *     - 'ﬓ' (U+FB13, EF AC 93) → "մն" (U+0574 U+0576, D5 B4 D5 B6)
     *     - 'ﬔ' (U+FB14, EF AC 94) → "մե" (U+0574 U+0565, D5 B4 D5 A5)
     *     - 'ﬕ' (U+FB15, EF AC 95) → "մի" (U+0574 U+056B, D5 B4 D5 AB)
     *     - 'ﬗ' (U+FB17, EF AC 97) → "մխ" (U+0574 U+056D, D5 B4 D5 AD)
     *  - 'ն' (U+0576, D5 B6) - can't be first; can't follow 'մ' (U+0574, D5 B4), 'վ' (U+057E, D5 BE) to avoid:
     *     - 'ﬓ' (U+FB13, EF AC 93) → "մն" (U+0574 U+0576, D5 B4 D5 B6)
     *     - 'ﬖ' (U+FB16, EF AC 96) → "վն" (U+057E U+0576, D5 BE D5 B6)
     *  - 'ի' (U+056B, D5 AB) - can't be first; can't follow 'մ' (U+0574, D5 B4) to avoid:
     *     - 'ﬕ' (U+FB15, EF AC 95) → "մի" (U+0574 U+056B, D5 B4 D5 AB)
     *  - 'վ' (U+057E, D5 BE) - can't be first; can't precede 'ն' (U+0576, D5 B6) to avoid:
     *     - 'ﬖ' (U+FB16, EF AC 96) → "վն" (U+057E U+0576, D5 BE D5 B6)
     *  - 'խ' (U+056D, D5 AD) - can't be first; can't follow 'մ' (U+0574, D5 B4) to avoid:
     *     - 'ﬗ' (U+FB17, EF AC 97) → "մխ" (U+0574 U+056D, D5 B4 D5 AD)
     *
     *  This means that Armenian needles containing these specific bigrams (եւ, մն, մե, մի, վն, մխ)
     *  cannot use the fast path because finding them separately might miss the precomposed ligatures
     *  present in the haystack.
     */
    sz_utf8_case_rune_safe_armenian_k = 6,

    /**
     *  @brief  Describes a safety-class profile for contextually-safe ASCII + Latin-1 + Latin Extended Additional.
     *  @sa sz_utf8_case_rune_safe_central_europe_k for the inherited Latin rules.
     *
     *  These kernels extend Latin-1/A/B with Vietnamese characters:
     *  - Everything from `sz_utf8_case_rune_safe_central_europe_k` (ASCII + Latin-1/A)
     *  - 166x Latin Extended Additional letters (U+1E00-U+1E95, U+1EA0-U+1EFF) for Vietnamese.
     *    Include precomposed Latin letters with additional diacritics (e.g. Ạ/ạ, Ả/ả, Ấ/ấ).
     *
     *  UTF-8 byte ranges handled:
     *  - 00-7F: ASCII, e.g. 'a' (U+0061, 61)
     *  - C2/C3: Latin-1 Supplement, e.g. 'â' (U+00E2, C3 A2)
     *  - C4-C5: Latin Extended-A, e.g. 'đ' (U+0111, C4 91)
     *  - C6: Latin Extended-B (for ơ, ư), e.g. 'ơ' (U+01A1, C6 A1)
     *  - E1 B8 80 - E1 BA 95: Latin Extended Additional (U+1E00-U+1E95), e.g. 'Ḁ' (U+1E00, E1 B8 80)
     *  - E1 BA A0 - E1 BB BF: Latin Extended Additional (U+1EA0-U+1EFF), e.g. 'ạ' (U+1EA1, E1 BA A1)
     *
     *  There is also a Unicode rule for folding the Kelvin 'K' (U+212A, E2 84 AA) into 'k' (U+006B, 6B).
     *  That sign is extremely rare, while the lowercase 'k' is common in Vietnamese (e.g. "kem", "kéo").
     *  So we add one more check for 'K' (U+212A, E2 84 AA) in the haystack, and if detected, again - revert to serial.
     *
     *  We inherit most contextual limitations for some of the ASCII characters from
     * `sz_utf8_case_rune_ascii_invariant_k`:
     *
     *  - 'a' (U+0061, 61) - can't be last; can't precede 'ʾ' (U+02BE, CA BE) to avoid:
     *    - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
     *  - 'f' (U+0066, 66) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede 'f' (U+0066, 66), 'i' (U+0069, 69), 'l' (U+006C, 6C) to avoid:
     *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'h' (U+0068, 68) - can't be last; can't precede '̱' (U+0331, CC B1) to avoid:
     *    - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66);
     *    can't precede '̇' (U+0307, CC 87) to avoid:
     *    - 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *  - 'j' (U+006A, 6A) - can't be last; can't precede '̌' (U+030C, CC 8C) to avoid:
     *    - 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C)
     *  - 'l' (U+006C, 6C) - can't be first; can't follow 'f' (U+0066, 66) to avoid:
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'n' (U+006E, 6E) - can't be first; can't follow 'ʼ' (U+02BC, CA BC) to avoid:
     *    - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
     *  - 's' (U+0073, 73) - can't be first or last; can't follow 's' (U+0073, 73);
     *    can't precede 's' (U+0073, 73), 't' (U+0074, 74) to avoid:
     *    - 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
     *    - 'ẞ' (U+1E9E, E1 BA 9E) → "ss" (U+0073 U+0073, 73 73)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *    - 'ẛ' (U+1E9B, E1 BA 9B) → 'ṡ' (U+1E61, E1 B9 A1) [Latin Extended Additional]
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73);
     *    can't precede '̈' (U+0308, CC 88) to avoid:
     *    - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 'w' (U+0077, 77) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
     *  - 'y' (U+0079, 79) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
     *
     *  We also inherit one more limitation from the Latin-1 profile:
     *
     *  - 'å' (U+00E5, C3 A5) - is the folding target of both 'Å' (U+00C5, C3 85) in Latin-1 and
     *    the Angstrom Sign 'Å' (U+212B, E2 84 AB) → 'å' (U+00E5, C3 A5), so needle cannot contain 'å' (U+00E5, C3 A5)
     *    to avoid ambiguity.
     *
     *  This means, that all other ASCII and Latin-1/A/Ext-Add characters are "safe" to use with this kernel.
     */
    sz_utf8_case_rune_safe_vietnamese_k = 7,

    /**
     *  @brief  Describes a safety-class profile for Georgian Mkhedruli script.
     *  @sa sz_utf8_case_rune_ascii_invariant_k for inherited ASCII rules.
     *
     *  Georgian Mkhedruli (U+10D0-U+10FF) is caseless - no folding needed for Georgian characters.
     *  Only ASCII A-Z folding for mixed text. Mtavruli (U+1C90-U+1CBF), Asomtavruli (U+10A0-U+10C5),
     *  and Nuskhuri (U+2D00-U+2D25) trigger alarm for serial fallback (rare in modern text).
     *
     *  All Georgian scripts use 3-byte UTF-8 sequences and fold to 3-byte sequences, so there
     *  are no length changes during case folding - making this the simplest non-ASCII kernel.
     */
    sz_utf8_case_rune_safe_georgian_k = 8,

    sz_utf8_case_rune_case_invariant_k = 9,
    sz_utf8_case_rune_fallback_serial_k = 255,
} sz_utf8_case_rune_safety_profile_t_;

/**
 *  @brief  Determine safety profile for a character across all script contexts.
 *
 *  This function encodes the contextual safety rules from the ASCII selector
 *  and applies them consistently to all paths that include ASCII.
 *
 *  @param[in] rune The decoded codepoint
 *  @param[in] rune_bytes UTF-8 byte length of this codepoint (1-4)
 *  @param[in] prev_rune Previous codepoint (0 if at start)
 *  @param[in] next_rune Next codepoint (0 if at end)
 *  @param[in] prev_prev_rune Codepoint before prev_rune (0 if prev is at start)
 *  @param[in] next_next_rune Codepoint after next_rune (0 if next is at end)
 *  @param[out] safety_profiles Safety flags for each script path
 *  @return The primary fast path preferred for this rune
 *
 *  @note Using 0 for boundary markers is safe even though NUL (U+0000) is a valid
 *        codepoint in StringZilla's length-based strings. This works because:
 *        1. NUL is valid ASCII (< 0x80), so boundary and actual NUL are treated identically
 *        2. Ligature checks use inequality (lower_prev != 'f'), and 0 never matches letters
 *        3. NUL doesn't participate in any Unicode case folding or ligature expansions
 *
 *  @note The neighbor-of-neighbor context (prev_prev, next_next) enables position-1 and
 *        position-N-2 detection for the 's' rule: if prev_prev==0 && prev!=0, we're at
 *        position 1; if next_next==0 && next!=0, we're at position N-2.
 */
SZ_INTERNAL sz_utf8_case_rune_safety_profile_t_ sz_utf8_case_rune_safety_profile_( //
    sz_rune_t rune, sz_size_t rune_bytes,                                          //
    sz_rune_t prev_rune, sz_rune_t next_rune,                                      //
    sz_rune_t prev_prev_rune, sz_rune_t next_next_rune,                            //
    unsigned int *safety_profiles) {

    unsigned safety = 0;

    // Bitmasks for profiles that share identical ASCII rules
    unsigned int western_group = //
        (1 << sz_utf8_case_rune_safe_western_europe_k);
    unsigned int central_viet_group =                    //
        (1 << sz_utf8_case_rune_safe_central_europe_k) | //
        (1 << sz_utf8_case_rune_safe_vietnamese_k);
    unsigned int strict_ascii_group =                //
        (1 << sz_utf8_case_rune_ascii_invariant_k) | //
        (1 << sz_utf8_case_rune_safe_cyrillic_k) |   //
        (1 << sz_utf8_case_rune_safe_greek_k) |      //
        (1 << sz_utf8_case_rune_safe_armenian_k) |   //
        (1 << sz_utf8_case_rune_safe_georgian_k);

    // Helper: lowercase ASCII
    sz_rune_t lower = (rune >= 'A' && rune <= 'Z') ? (rune + 0x20) : rune;
    sz_rune_t lower_prev = (prev_rune >= 'A' && prev_rune <= 'Z') ? (prev_rune + 0x20) : prev_rune;
    sz_rune_t lower_next = (next_rune >= 'A' && next_rune <= 'Z') ? (next_rune + 0x20) : next_rune;

    // Helper: is neighbor ASCII letter? (explicit conversion for C++ compatibility)
    // Note: prev_rune/next_rune == 0 means boundary (start/end of needle)
    sz_bool_t prev_ascii = (prev_rune != 0 && prev_rune < 0x80) ? sz_true_k : sz_false_k;
    sz_bool_t next_ascii = (next_rune != 0 && next_rune < 0x80) ? sz_true_k : sz_false_k;
    sz_bool_t at_start = (prev_rune == 0) ? sz_true_k : sz_false_k;
    sz_bool_t at_end = (next_rune == 0) ? sz_true_k : sz_false_k;

    // Helper: position detection for 's' rule (mid-ß-expansion avoidance in Western profile)
    // Position 1: prev exists but prev_prev doesn't (prev is at position 0)
    // Position N-2: next exists but next_next doesn't (next is at position N-1)
    sz_bool_t at_pos_1 = (prev_rune != 0 && prev_prev_rune == 0) ? sz_true_k : sz_false_k;
    sz_bool_t at_pos_n_minus_2 = (next_rune != 0 && next_next_rune == 0) ? sz_true_k : sz_false_k;

    // ASCII character (1-byte UTF-8)
    if (rune < 0x80) {
        if (lower >= 'a' && lower <= 'z') {
            switch (lower) {

            // Unconditionally safe for all profiles.
            // No Unicode chars fold to sequences containing these,
            // and they don't participate in dangerous ligatures.
            // clang-format off
            case 'b': case 'c': case 'd': case 'e': case 'g':
            case 'm': case 'o': case 'p': case 'q': case 'r': case 'u':
            case 'v': case 'x': case 'z':
                // clang-format on
                safety |= strict_ascii_group | central_viet_group | western_group;
                break;

            // 'k':
            // - Strict: UNSAFE. 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B).
            // - Western/Central/Viet: SAFE. Kelvin sign detected in haystack.
            case 'k': safety |= central_viet_group | western_group; break;

            // 'a':
            // - Strict/Central/Viet: Contextual. Can't be last; can't precede 'ʾ' (U+02BE, CA BE).
            //   Avoids: 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE).
            // - Western: SAFE. Expansion detected in haystack.
            case 'a':
                if (at_end == sz_false_k && next_ascii) safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;

            // 'h':
            // - Strict/Central/Viet: Contextual. Can't be last; can't precede '̱' (U+0331, CC B1).
            //   Avoids: 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1).
            // - Western: SAFE. Expansion detected in haystack.
            case 'h':
                if (at_end == sz_false_k && next_ascii) safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;

            // 'j':
            // - All: Contextual. Can't be last; can't precede '̌' (U+030C).
            //   Avoids: 'ǰ' (U+01F0) → "ǰ" (U+006A U+030C, 6A CC 8C).
            //   Western profile does NOT detect this in haystack scan.
            case 'j':
                if (at_end == sz_false_k && next_ascii)
                    safety |= strict_ascii_group | central_viet_group | western_group;
                break;

            // 'w':
            // - Strict/Central/Viet: Contextual. Can't be last; can't precede '̊' (U+030A).
            //   Avoids: 'ẘ' (U+1E98) → "ẘ" (U+0077 U+030A, 77 CC 8A).
            // - Western: SAFE. Expansion detected in haystack.
            case 'w':
                if (at_end == sz_false_k && next_ascii) safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;

            // 'y':
            // - Strict/Central/Viet: Contextual. Can't be last; can't precede '̊' (U+030A).
            //   Avoids: 'ẙ' (U+1E99) → "ẙ" (U+0079 U+030A, 79 CC 8A).
            // - Western: SAFE. Expansion detected in haystack.
            case 'y':
                if (at_end == sz_false_k && next_ascii) safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;

            // 'n':
            // - ASCII/Cyrillic/Greek: Contextual. Can't be first; can't follow 'ʼ' (U+02BC, CA BC).
            //   Avoids: 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E).
            // - Armenian: UNSAFE. Armenian kernel cannot handle 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E).
            //   The character 'n' can match the 2nd part of the expansion, causing false positives.
            // - Western/Central/Viet: Contextual, same as above.
            //   Western profile does NOT detect this in haystack scan.
            case 'n':
                // Exclude Armenian - it cannot handle 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
                if (at_start == sz_false_k && prev_ascii) {
                    safety |= (1 << sz_utf8_case_rune_ascii_invariant_k) | //
                              (1 << sz_utf8_case_rune_safe_cyrillic_k) |   //
                              (1 << sz_utf8_case_rune_safe_greek_k);       //
                    // Armenian EXCLUDED: sz_utf8_case_rune_safe_armenian_k
                    safety |= central_viet_group | western_group;
                }
                break;

            // 'i':
            // - All: Contextual. Can't be first or last; can't follow 'f'; can't precede '̇' (U+0307, CC 87).
            //   Avoids: 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87),
            //   and 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69).
            //   Western profile does NOT detect Turkish 'İ' expansion.
            case 'i':
                if (at_start == sz_false_k && at_end == sz_false_k && next_ascii && lower_prev != 'f')
                    safety |= strict_ascii_group | central_viet_group | western_group;
                break;

            // 'l':
            // - Strict/Central/Viet: Contextual. Can't be first; can't follow 'f'.
            //   Avoids: 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C).
            // - Western: SAFE. Ligatures detected in haystack.
            case 'l':
                if (at_start == sz_false_k && lower_prev != 'f') safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;

            // 't':
            // - Strict/Central/Viet: Contextual. Can't be first/last; can't follow 's';
            //   can't precede '̈' (U+0308, CC 88).
            //   Avoids: 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74),
            //   'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74),
            //   and 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88).
            // - Western: SAFE. Ligatures/expansion detected in haystack.
            case 't':
                if (at_start == sz_false_k && at_end == sz_false_k && next_ascii && lower_prev != 's')
                    safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;

            // 'f':
            // - Strict/Central/Viet: Contextual. Can't be first/last; can't follow 'f';
            //   can't precede 'f', 'i', 'l'.
            //   Avoids:
            //   - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
            //   - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
            //   - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
            //   - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
            //   - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
            // - Western: SAFE. Ligatures detected in haystack.
            case 'f':
                if (at_start == sz_false_k && at_end == sz_false_k && prev_ascii && next_ascii && lower_prev != 'f' &&
                    lower_next != 'f' && lower_next != 'i' && lower_next != 'l')
                    safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;

            // 's'
            // - Strict: UNSAFE. 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73).
            // - Central/Vietnamese: Contextual. Can't be first/last; can't be adjacent to 's'/'t'.
            //   Avoids: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73),
            //   'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74),
            //   'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74), and 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73).
            // - Western: Contextual. Can't be at positions 0, 1 (if prev='s'), N-1, or N-2 (if next='s').
            //   Avoids mid-ß-expansion matches: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73) in-place means
            //   needle with 's' at these positions could match at byte offset 1 (UTF-8 continuation byte 0x9F).
            //   Example: "ßStra" → "ssstra", needle "sstra" matches at pos 1 = mid-character.
            //   Interior 's' like "tesst" or "masse" are safe for SIMD.
            case 's':
                if (at_start == sz_false_k && at_end == sz_false_k && prev_ascii && next_ascii && lower_prev != 's' &&
                    lower_next != 's' && lower_next != 't')
                    safety |= central_viet_group;
                // Western: ban pos 0, pos 1 if prev='s', pos N-1, pos N-2 if next='s'
                if (at_start == sz_false_k && at_end == sz_false_k && //
                    !(at_pos_1 == sz_true_k && lower_prev == 's') &&  //
                    !(at_pos_n_minus_2 == sz_true_k && lower_next == 's'))
                    safety |= western_group;
                break;

            default:
                // Should not happen for a-z
                safety |= strict_ascii_group | central_viet_group | western_group;
                break;
            }
        }
        else {
            // Non-letters (digits, punctuation, whitespace) - always safe for all profiles
            safety |= strict_ascii_group | central_viet_group | western_group;
        }

        *safety_profiles = safety;
        return sz_utf8_case_rune_ascii_invariant_k;
    }

    // 2-byte UTF-8 (U+0080 to U+07FF)
    // Must check EXACT ranges that the fold functions handle, not just lead bytes
    if (rune_bytes == 2) {
        sz_u8_t lead = (rune >> 6) | 0xC0;     // Reconstruct lead byte
        sz_u8_t second = (rune & 0x3F) | 0x80; // Reconstruct continuation byte

        // Latin-1 Supplement (C2/C3 lead bytes)
        // Exclude: 'å' (U+00E5, C3 A5) - Angstrom Sign 'Å' (U+212B, E2 84 AB) → 'å' (U+00E5, C3 A5) also folds to it
        if (lead == 0xC2 || lead == 0xC3) {
            if (rune == 0x00E5) {
                // 'å' excluded from all Latin profiles due to Angstrom ambiguity
            }
            else if (rune == 0x00DF) {
                // 'ß' excluded from Central Europe and Vietnamese, allowed in Western Europe
                safety |= western_group;
            }
            else if (rune == 0x00B5) {
                // 'µ' (U+00B5, C2 B5) → 'μ' (U+03BC, CE BC).
                // Allow only the Greek SIMD path; Latin paths remain unsafe.
                safety |= (1 << sz_utf8_case_rune_safe_greek_k);
            }
            else { safety |= western_group | central_viet_group; }
        }

        // Latin Extended-A (C4/C5 lead bytes) - for central_europe and vietnamese
        if (lead == 0xC4 || lead == 0xC5) {
            // Exclude expansions/length-changes:
            // - 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
            // - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
            // - 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
            if (rune != 0x0130 && rune != 0x0149 && rune != 0x017F) { safety |= central_viet_group; }
        }

        // Latin Extended-B (C6 lead byte) - for vietnamese (supports ơ/ư)
        if (lead == 0xC6) { safety |= (1 << sz_utf8_case_rune_safe_vietnamese_k); }

        // Cyrillic - check exact ranges handled by sz_utf8_case_insensitive_find_ice_cyrillic_fold_zmm_
        // D0 80-BF: U+0400-U+043F (includes uppercase and lowercase)
        // D1 80-9F: U+0440-U+045F (lowercase continuation)
        // Note: D2/D3 Extended Cyrillic BANNED from SIMD kernel - needles with D2/D3 use serial fallback
        if ((lead == 0xD0 && second >= 0x80 && second <= 0xBF) || //
            (lead == 0xD1 && second >= 0x80 && second <= 0x9F)) { //
            safety |= (1 << sz_utf8_case_rune_safe_cyrillic_k);
        }

        // Greek - check exact ranges handled by sz_utf8_case_insensitive_find_ice_greek_fold_zmm_
        // CE 86-8F: accented uppercase Ά-Ώ (with gaps at 87, 8B, 8D)
        //   - EXCLUDE CE 90: 'ΐ' (U+0390) expands to 3 codepoints
        // CE 91-A9: basic uppercase Α-Ω
        // CE AA-AB: dialytika uppercase Ϊ-Ϋ
        // CE AC-AF: accented lowercase ά-ί
        //   - EXCLUDE CE B0: 'ΰ' (U+03B0) expands to 3 codepoints
        // CE B1-BF: basic lowercase α-ο
        // CF 80-89: basic lowercase π-ω (includes final sigma at 82, sigma at 83)
        // CF 8A-8E: accented/dialytika lowercase ϊ-ώ
        if (lead == 0xCE) {
            // Accented uppercase (with gaps) - exclude 87, 8B, 8D, 90
            if ((second >= 0x86 && second <= 0x8F) && second != 0x87 && second != 0x8B && second != 0x8D &&
                second != 0x90) {
                safety |= (1 << sz_utf8_case_rune_safe_greek_k);
            }
            // Basic uppercase Α-Ω
            if (second >= 0x91 && second <= 0xA9) { safety |= (1 << sz_utf8_case_rune_safe_greek_k); }
            // Dialytika uppercase Ϊ-Ϋ
            if (second >= 0xAA && second <= 0xAB) { safety |= (1 << sz_utf8_case_rune_safe_greek_k); }
            // Accented lowercase ά-ί
            if (second >= 0xAC && second <= 0xAF) { safety |= (1 << sz_utf8_case_rune_safe_greek_k); }
            // Basic lowercase α-ο - exclude B0 (ΰ expands)
            if (second >= 0xB1 && second <= 0xBF) { safety |= (1 << sz_utf8_case_rune_safe_greek_k); }
        }
        if (lead == 0xCF) {
            // Basic lowercase π-ω
            if (second >= 0x80 && second <= 0x89) { safety |= (1 << sz_utf8_case_rune_safe_greek_k); }
            // Accented/dialytika lowercase ϊ-ώ
            if (second >= 0x8A && second <= 0x8E) { safety |= (1 << sz_utf8_case_rune_safe_greek_k); }
        }

        // Armenian - check exact ranges with contextual constraints for ligatures
        // D4 B1-BF: uppercase Ա-Ի (U+0531-U+053F)
        // D5 80-96: uppercase Լ-Ֆ (U+0540-U+0556)
        // D5 A1-BF: lowercase ա-տ (U+0561-U+057F)
        // D6 80-86: lowercase ր-ֆ (U+0580-U+0586)
        //
        // Ligature constraints (from spec):
        // - 'ե' (U+0565): can't be first; can't follow 'մ'; can't precede 'ւ'
        // - 'ւ' (U+0582): can't be last; can't follow 'ե'
        // - 'մ' (U+0574): can't be last; can't precede 'ն', 'ե', 'ի', 'խ'
        // - 'ն' (U+0576): can't be first; can't follow 'մ', 'վ'
        // - 'ի' (U+056B): can't be first; can't follow 'մ'
        // - 'վ' (U+057E): can't be first; can't precede 'ն'
        // - 'խ' (U+056D): can't be first; can't follow 'մ'
        {
            sz_bool_t is_armenian_range = sz_false_k;
            sz_bool_t armenian_safe = sz_true_k;

            if ((lead == 0xD4 && second >= 0xB1 && second <= 0xBF) ||
                (lead == 0xD5 && second >= 0x80 && second <= 0x96) ||
                (lead == 0xD5 && second >= 0xA1 && second <= 0xBF) ||
                (lead == 0xD6 && second >= 0x80 && second <= 0x86)) {
                is_armenian_range = sz_true_k;

                // Helper: get lowercase Armenian codepoint for neighbor checks
                sz_rune_t lower_prev_arm = prev_rune;
                sz_rune_t lower_next_arm = next_rune;
                if (prev_rune >= 0x0531 && prev_rune <= 0x0556) lower_prev_arm = prev_rune + 0x30;
                if (next_rune >= 0x0531 && next_rune <= 0x0556) lower_next_arm = next_rune + 0x30;

                // Check ligature constraints
                switch (rune) {
                case 0x0565: // U+0565 ech - can't be first; can't follow U+0574 men; can't precede U+0582 yiwn
                case 0x0535: // U+0535 Ech uppercase
                    if (at_start || lower_prev_arm == 0x0574 || lower_next_arm == 0x0582) armenian_safe = sz_false_k;
                    break;
                case 0x0582: // U+0582 yiwn - can't be first; can't be last; can't follow U+0565 ech
                    // Armenian ligature և (U+0587) → ech + yiwn; needle starting with yiwn matches mid-expansion
                    if (at_start || at_end || lower_prev_arm == 0x0565) armenian_safe = sz_false_k;
                    break;
                case 0x0574: // U+0574 men - can't be last; can't precede U+0576, U+0565, U+056B, U+056D
                case 0x0544: // U+0544 Men uppercase
                    if (at_end || lower_next_arm == 0x0576 || lower_next_arm == 0x0565 || lower_next_arm == 0x056B ||
                        lower_next_arm == 0x056D)
                        armenian_safe = sz_false_k;
                    break;
                case 0x0576: // U+0576 now - can't be first; can't follow U+0574 men, U+057E vew
                case 0x0546: // U+0546 Now uppercase
                    if (at_start || lower_prev_arm == 0x0574 || lower_prev_arm == 0x057E) armenian_safe = sz_false_k;
                    break;
                case 0x056B: // U+056B ini - can't be first; can't follow U+0574 men
                case 0x053B: // U+053B Ini uppercase
                    if (at_start || lower_prev_arm == 0x0574) armenian_safe = sz_false_k;
                    break;
                case 0x057E: // U+057E vew - can't be last; can't precede U+0576 now
                case 0x054E: // U+054E Vew uppercase
                    if (at_end || lower_next_arm == 0x0576) armenian_safe = sz_false_k;
                    break;
                case 0x056D: // U+056D xeh - can't be first; can't follow U+0574 men
                case 0x053D: // U+053D Xeh uppercase
                    if (at_start || lower_prev_arm == 0x0574) armenian_safe = sz_false_k;
                    break;
                default: break;
                }
            }

            if (is_armenian_range && armenian_safe) { safety |= (1 << sz_utf8_case_rune_safe_armenian_k); }
        }

        // Output safety and determine primary script for 2-byte runes
        // For case-invariant non-ASCII runes, add the ASCII-invariant bit.
        // This enables fast ASCII kernel for needles like "中文字" that contain no cased characters.
        // ASCII fold only affects bytes 0x41-0x5A (A-Z), so all other bytes pass through unchanged.
        if (sz_rune_is_case_invariant_(rune)) safety |= (1 << sz_utf8_case_rune_ascii_invariant_k);
        *safety_profiles = safety;
        if (rune >= 0x0080 && rune <= 0x00FF) return sz_utf8_case_rune_safe_western_europe_k; // Latin-1 Supplement
        if (rune >= 0x0100 && rune <= 0x024F) return sz_utf8_case_rune_safe_central_europe_k; // Latin Extended-A/B
        if (rune >= 0x0370 && rune <= 0x03FF) return sz_utf8_case_rune_safe_greek_k;          // Greek
        if (rune >= 0x0400 && rune <= 0x04FF) return sz_utf8_case_rune_safe_cyrillic_k;       // Cyrillic
        if (rune >= 0x0530 && rune <= 0x058F) return sz_utf8_case_rune_safe_armenian_k;       // Armenian
        return sz_utf8_case_rune_case_invariant_k;
    }

    // 3-byte UTF-8 (U+0800 to U+FFFF)
    if (rune_bytes == 3) {
        sz_u8_t lead = (rune >> 12) | 0xE0;
        sz_u8_t second = ((rune >> 6) & 0x3F) | 0x80;
        sz_u8_t third = (rune & 0x3F) | 0x80;

        // Vietnamese/Latin Extended Additional (E1 B8-BB range)
        // U+1E00-U+1EFF maps to E1 B8 80 - E1 BB BF
        if (lead == 0xE1 && (second >= 0xB8 && second <= 0xBB)) {
            // Need detailed check for exclusions in U+1E96-U+1E9F
            // 1E96-1E9F: E1 BA 96 - E1 BA 9F
            if (second == 0xBA && third >= 0x96 && third <= 0x9F) {
                // Excluded: expansions or irregulars
            }
            else { safety |= (1 << sz_utf8_case_rune_safe_vietnamese_k); }
        }

        // Georgian Mkhedruli (E1 83 90-BF range)
        // U+10D0-U+10FF maps to E1 83 90 - E1 83 BF
        // Mkhedruli is caseless, so all characters are safe for the Georgian kernel.
        if (lead == 0xE1 && second == 0x83 && third >= 0x90) { safety |= (1 << sz_utf8_case_rune_safe_georgian_k); }

        // Output safety and determine primary script for 3-byte runes
        // For case-invariant non-ASCII runes (like CJK), add the ASCII-invariant bit.
        if (sz_rune_is_case_invariant_(rune)) safety |= (1 << sz_utf8_case_rune_ascii_invariant_k);
        *safety_profiles = safety;
        if (rune >= 0x10D0 && rune <= 0x10FF) return sz_utf8_case_rune_safe_georgian_k;   // Georgian Mkhedruli
        if (rune >= 0x1E00 && rune <= 0x1EFF) return sz_utf8_case_rune_safe_vietnamese_k; // Latin Extended Additional
        return sz_utf8_case_rune_case_invariant_k;
    }

    // 4-byte UTF-8 - currently no fast paths, but case-invariant 4-byte runes can use ASCII kernel
    if (sz_rune_is_case_invariant_(rune)) safety |= (1 << sz_utf8_case_rune_ascii_invariant_k);
    *safety_profiles = safety;
    return sz_utf8_case_rune_case_invariant_k;
}

/**
 *  @brief Compute diversity score for a byte sequence.
 *
 *  Uses a 256-bit bitmap to efficiently count distinct byte values.
 *  Higher scores indicate more diverse byte values, which lead to better
 *  filtering during SIMD search (fewer false positives).
 *
 *  @param[in] data Pointer to byte sequence.
 *  @param[in] length Length of byte sequence.
 *  @return Count of distinct byte values (0-256).
 */
SZ_INTERNAL sz_size_t sz_utf8_probe_diversity_score_(sz_u8_t const *data, sz_size_t length) {
    if (length <= 1) return length;
    sz_u64_t seen[4] = {0, 0, 0, 0}; // 256-bit bitmap
    sz_size_t distinct = 0;
    for (sz_size_t i = 0; i < length; ++i) {
        sz_u8_t byte = data[i];
        sz_size_t word = byte >> 6;                // Which 64-bit word (0-3)
        sz_u64_t bit = (sz_u64_t)1 << (byte & 63); // Bit within the word
        if (!(seen[word] & bit)) {
            seen[word] |= bit;
            ++distinct;
        }
    }
    return distinct;
}

/**
 *  @brief Find the "best safe window" in the needle for each script path.
 *
 *  The objective is as follows. For a given needle, find a slice, that when folded fits into 16 bytes
 *  and where all characters are "safe" with respect to a certain path. If no such path can be found,
 *  an empty result is returned. It might be the case for a search query like "s" or "n", that by itself
 *  isn't safe for any path given the number of Unicode characters expanding into multiple 's'- or 'n'-containing
 *  sequences. The selected safe folded slice will never begin mid-character in the needle, so if it starts with
 *  an 'ŉ' (U+0149, C5 89), we can't choose - 'n' (6E) - the second half of its folded sequence as a starting point.
 *
 *  The algorithm is as follows. Iterate through the arbitrary-case "ŉEeDlE_WITH_LONG_SUFFIX", unpacking runes.
 *  For each input rune, perform folding, expanding into a sequence, like:
 *
 *      'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E).
 *
 *  Continue unpacking the rest, until we reach a 16-byte limit, like:
 *
 *      ʼ     n  e  e  d  l  e  _  w  i  t  h  _  l  o  n  g
 *      CA BC 6E 45 45 44 4C 45 5F 57 49 54 48 5F 4C 4F 4E 47
 *
 *  At this point, we need to trim it to make sure - its characters satisfy boundary conditions.
 *  Assuming at the next step we'll move the iterator to the next input rune to point to 'E' (U+0045) input character,
 *  we only trim from the end. But also invalidate the whole starting position if a bad character is chosen at start.
 *  For safe window starting position we can have multiple length variants, assuming different safe paths can have
 *  different rules for the last symbol in the safe sequence.
 *
 *  Once we have safe window for a certain script, we evaluate its diversity score - the number of distinct byte
 *  values in the folded window. The more diverse - the better! We keep track of best seen window for each script.
 *
 *  We also track not only the safety with respect to a certain profile, but also applicability. For example,
 *  the needle "xyz" is safe with respect to the Western European path, as well as Central European, Vietnamese,
 *  and potentially others. But it's pure ASCII. We shouldn't pay the cost of complex Vietnamese case-folding of
 *  triple-byte Latin extensions for just "xyz". So we must invalidate the "safe path" if its just "safe", but
 *  not ideal.
 *
 *  In the end, we'll have up to 7 best safe windows, one per script path.
 *  The heuristic is:
 *
 *  - Prefer ASCII, if there is an ASCII-safe path at least 4 bytes wide with at least 4 distinct byte values.
 *    It's only one subtraction, a comparison, and a masked addition. Cheapest of all kernels.
 *  - Pick the most diverse variant from all others, if ASCII variant isn't good enough.
 *
 *  We then identify the four "probe" positions within the <= 16 byte folded safe window, one more than
 *  in exact substring search kernels with Raita heuristics:
 *
 *  1. implicit at `refined->folded_slice[0]`
 *  2. stored in `refined->probe_second` - targets last byte of 2nd character when 4+ chars
 *  3. stored in `refined->probe_third` - targets last byte of 3rd character when 4+ chars
 *  4. implicit at `refined->folded_slice[refined->folded_slice_length - 1]`
 *
 *  By aiming at the last byte of each UTF-8 codepoint we maximize diversity, as in a Russian text almost
 *  all letters will have the same first byte, but mostly different second byte. The same is true for many
 *  other languages. For short strings (< 4 bytes), probes will necessarily overlap - this is expected.
 *  The function also sets `offset_in_unfolded` and `length_in_unfolded` to track where the
 *  selected folded slice came from in the original unfolded input.
 *
 *  @param[in] needle Pointer to needle string (original, not folded)
 *  @param[in] needle_length Length in bytes
 *  @param[out] refined Output metadata structure to populate
 */
SZ_INTERNAL void sz_utf8_case_insensitive_needle_metadata_(sz_cptr_t needle, sz_size_t needle_length, //
                                                           sz_utf8_case_insensitive_needle_metadata_t *refined) {

    // Per-script window state during iteration
    typedef struct {
        sz_size_t start_offset;   // Byte offset in original needle
        sz_size_t input_length;   // Bytes consumed from original needle
        sz_u8_t folded_bytes[16]; // Folded content
        sz_size_t folded_length;  // Length of folded content (bytes)
        sz_bool_t applicable;     // Has >=1 primary-script character
        sz_bool_t broken;         // Window continuity broken - skip further extension
        sz_size_t diversity;      // Distinct byte count (computed at end of each starting position)
    } script_window_t_;

    // Number of script kernels (indices 1-8 used, index 0 reserved)
    sz_size_t const num_scripts = 9;

    // Best window found so far for each script
    script_window_t_ best[9];
    for (sz_size_t i = 0; i < num_scripts; ++i) {
        best[i].start_offset = 0;
        best[i].input_length = 0;
        best[i].folded_length = 0;
        best[i].applicable = sz_false_k;
        best[i].broken = sz_false_k;
        best[i].diversity = 0;
    }

    // Handle empty needle
    if (needle_length == 0) {
        refined->kernel_id = sz_utf8_case_rune_fallback_serial_k;
        refined->offset_in_unfolded = 0;
        refined->length_in_unfolded = 0;
        refined->folded_slice_length = 0;
        refined->probe_second = 0;
        refined->probe_third = 0;
        return;
    }

    sz_u8_t const *needle_bytes = (sz_u8_t const *)needle;
    sz_u8_t const *needle_end = needle_bytes + needle_length;

    // Iterate through each starting position in the needle (stepping by rune)
    for (sz_u8_t const *start_ptr = needle_bytes; start_ptr < needle_end;) {
        // Current window being built for each script at this starting position
        script_window_t_ current[9];
        for (sz_size_t i = 0; i < num_scripts; ++i) {
            current[i].start_offset = (sz_size_t)(start_ptr - needle_bytes);
            current[i].input_length = 0;
            current[i].folded_length = 0;
            current[i].applicable = sz_false_k;
            current[i].broken = sz_false_k;
            current[i].diversity = 0;
        }

        // Track context for safety profile evaluation
        sz_rune_t prev_prev_rune = 0;
        sz_rune_t prev_rune = 0;

        // Fold forward from start_ptr until 16 bytes or needle end
        sz_u8_t const *pos = start_ptr;
        sz_bool_t any_active = sz_true_k;

        while (pos < needle_end && any_active) {
            // Parse current rune
            sz_rune_t rune;
            sz_rune_length_t rune_bytes;
            sz_rune_parse((sz_cptr_t)pos, &rune, &rune_bytes);
            if (pos + rune_bytes > needle_end) break; // Incomplete rune

            // Parse next rune for context (if available)
            sz_rune_t next_rune = 0;
            sz_rune_length_t next_bytes = sz_utf8_invalid_k;
            if (pos + rune_bytes < needle_end) {
                sz_rune_parse((sz_cptr_t)(pos + rune_bytes), &next_rune, &next_bytes);
                if (pos + rune_bytes + next_bytes > needle_end) next_rune = 0;
            }

            // Parse next-next rune for context
            sz_rune_t next_next_rune = 0;
            if (next_rune != 0 && pos + rune_bytes + next_bytes < needle_end) {
                sz_rune_length_t next_next_bytes;
                sz_rune_parse((sz_cptr_t)(pos + rune_bytes + next_bytes), &next_next_rune, &next_next_bytes);
                if (pos + rune_bytes + next_bytes + next_next_bytes > needle_end) next_next_rune = 0;
            }

            // Get safety mask and primary script for this rune
            unsigned safety_mask = 0;
            sz_utf8_case_rune_safety_profile_t_ primary_script = sz_utf8_case_rune_safety_profile_( //
                rune, rune_bytes, prev_rune, next_rune, prev_prev_rune, next_next_rune, &safety_mask);

            // Fold this rune
            sz_rune_t folded_runes[4];
            sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);

            // Convert folded runes to UTF-8 bytes
            sz_u8_t folded_utf8[16];
            sz_size_t folded_utf8_len = 0;
            for (sz_size_t i = 0; i < folded_count; ++i) {
                folded_utf8_len += sz_rune_export(folded_runes[i], folded_utf8 + folded_utf8_len);
            }

            // Update each script's window
            any_active = sz_false_k;
            for (sz_size_t script = 1; script < num_scripts; ++script) {
                if (current[script].broken) continue;

                // Check if this rune is safe for this script
                sz_bool_t is_safe = (safety_mask & (1u << script)) ? sz_true_k : sz_false_k;

                // Check if adding this rune would exceed 16 bytes
                if (is_safe && current[script].folded_length + folded_utf8_len <= 16) {
                    // Extend this script's window
                    for (sz_size_t b = 0; b < folded_utf8_len; ++b) {
                        current[script].folded_bytes[current[script].folded_length + b] = folded_utf8[b];
                    }
                    current[script].folded_length += folded_utf8_len;
                    current[script].input_length += rune_bytes;

                    // Mark as applicable if primary script matches
                    if (primary_script == script) { current[script].applicable = sz_true_k; }
                    any_active = sz_true_k;
                }
                else {
                    // Window broken for this script
                    current[script].broken = sz_true_k;
                }
            }

            // Update context for next iteration
            prev_prev_rune = prev_rune;
            prev_rune = rune;
            pos += rune_bytes;
        }

        // Compare current to best for each script
        for (sz_size_t script = 1; script < num_scripts; ++script) {
            if (!current[script].applicable || current[script].folded_length == 0) continue;

            // Compute diversity score
            current[script].diversity =
                sz_utf8_probe_diversity_score_(current[script].folded_bytes, current[script].folded_length);

            // Update best if this is better (prefer higher diversity, then longer length)
            if (current[script].diversity > best[script].diversity ||
                (current[script].diversity == best[script].diversity &&
                 current[script].folded_length > best[script].folded_length)) {
                best[script] = current[script];
            }
        }

        // Advance to next rune for next starting position
        sz_rune_t skip_rune;
        sz_rune_length_t skip_len;
        sz_rune_parse((sz_cptr_t)start_ptr, &skip_rune, &skip_len);
        start_ptr += skip_len;
    }

    // Select final kernel based on best windows
    // Rule: Prefer ASCII if >=4 bytes with >=4 diversity; otherwise pick most diverse applicable
    sz_size_t chosen_script = 0;
    sz_size_t best_diversity = 0;

    // Check ASCII preference
    if (best[sz_utf8_case_rune_ascii_invariant_k].applicable &&
        best[sz_utf8_case_rune_ascii_invariant_k].folded_length >= 4 &&
        best[sz_utf8_case_rune_ascii_invariant_k].diversity >= 4) {
        chosen_script = sz_utf8_case_rune_ascii_invariant_k;
    }
    else {
        // Find most diverse applicable script
        for (sz_size_t script = 1; script < num_scripts; ++script) {
            if (best[script].applicable && best[script].diversity > best_diversity) {
                best_diversity = best[script].diversity;
                chosen_script = script;
            }
        }
    }

    // If no applicable window found, fall back to serial
    if (chosen_script == 0) {
        refined->kernel_id = sz_utf8_case_rune_fallback_serial_k;
        refined->offset_in_unfolded = 0;
        refined->length_in_unfolded = 0;
        refined->folded_slice_length = 0;
        refined->probe_second = 0;
        refined->probe_third = 0;
        return;
    }

    // Populate output metadata
    refined->kernel_id = (sz_u8_t)chosen_script;
    refined->offset_in_unfolded = best[chosen_script].start_offset;
    refined->length_in_unfolded = best[chosen_script].input_length;
    refined->folded_slice_length = (sz_u8_t)best[chosen_script].folded_length;

    // Copy folded bytes
    for (sz_size_t i = 0; i < best[chosen_script].folded_length; ++i) {
        refined->folded_slice[i] = best[chosen_script].folded_bytes[i];
    }

    // Compute probe positions - target last bytes of UTF-8 codepoints for maximum diversity
    sz_size_t folded_len = best[chosen_script].folded_length;
    if (folded_len == 0) {
        refined->probe_second = 0;
        refined->probe_third = 0;
        return;
    }

    // Find character end positions in the folded slice
    // A byte is a character's last byte if the next byte is a UTF-8 leader (not continuation)
    sz_size_t char_ends[16];
    sz_size_t char_count = 0;
    for (sz_size_t i = 0; i < folded_len; ++i) {
        sz_u8_t next = (i + 1 < folded_len) ? refined->folded_slice[i + 1] : 0xC0; // Fake leader at end
        if ((next & 0xC0) != 0x80) {                                               // Next is not a continuation byte
            if (char_count < 16) char_ends[char_count++] = i;
        }
    }

    // Determine probe positions
    if (char_count >= 4) {
        // 4+ characters: target last bytes of 2nd and 3rd characters
        refined->probe_second = (sz_u8_t)char_ends[1];
        refined->probe_third = (sz_u8_t)char_ends[2];
    }
    else if (folded_len <= 3) {
        // Very short: probes overlap
        refined->probe_second = (folded_len > 1) ? 1 : 0;
        refined->probe_third = (folded_len > 1) ? 1 : 0;
    }
    else {
        // 1-3 characters but 4+ bytes: use byte diversity search
        sz_u8_t byte_first = refined->folded_slice[0];
        sz_u8_t byte_last = refined->folded_slice[folded_len - 1];

        sz_size_t probe_second = folded_len / 3;
        sz_size_t probe_third = (folded_len * 2) / 3;

        // Try to find positions with bytes distinct from first/last
        for (sz_size_t i = 1; i < folded_len - 1; ++i) {
            if (refined->folded_slice[i] != byte_first && refined->folded_slice[i] != byte_last) {
                probe_second = i;
                break;
            }
        }

        sz_u8_t byte_second = refined->folded_slice[probe_second];
        for (sz_size_t i = probe_second + 1; i < folded_len - 1; ++i) {
            if (refined->folded_slice[i] != byte_first && refined->folded_slice[i] != byte_last &&
                refined->folded_slice[i] != byte_second) {
                probe_third = i;
                break;
            }
        }

        // Clamp bounds
        if (probe_second == 0) probe_second = 1;
        if (probe_third >= folded_len - 1) probe_third = folded_len - 2;
        if (probe_third <= probe_second && probe_second + 1 < folded_len - 1) probe_third = probe_second + 1;

        refined->probe_second = (sz_u8_t)probe_second;
        refined->probe_third = (sz_u8_t)probe_third;
    }
}

#pragma endregion Character Safety Profiles
#pragma region ASCII Case-Insensitive Find

/**
 *  @brief Fold a ZMM register using ASCII case folding rules.
 *  @sa sz_utf8_case_rune_ascii_invariant_k
 *
 *  @param[in] text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(__m512i text_zmm) {
    __m512i const a_upper_zmm = _mm512_set1_epi8('A');
    __m512i const range26_zmm = _mm512_set1_epi8(26);
    __m512i const x_20_zmm = _mm512_set1_epi8(0x20);

    // Only fold bytes in range A-Z: (byte - 'A') < 26
    __mmask64 upper_mask = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_zmm, a_upper_zmm), range26_zmm);
    return _mm512_mask_add_epi8(text_zmm, upper_mask, text_zmm, x_20_zmm);
}

/**
 *  @brief ASCII case-insensitive search for needles with safe slices up to 16 bytes.
 *
 *  Scans the entire haystack from byte 0, looking for the folded window pattern.
 *  When found, verifies the head (backwards) and tail (forwards) using codepoint-by-codepoint
 *  comparison to handle variable-width folding correctly.
 *
 *  @param[in] haystack Pointer to the haystack string.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] needle Pointer to the full needle string.
 *  @param[in] needle_length Length of the full needle in bytes.
 *  @param[in] needle_metadata Pre-folded window content with probe positions.
 *  @param[out] matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_ice_ascii_(        //
    sz_cptr_t haystack, sz_size_t haystack_length,                     //
    sz_cptr_t needle, sz_size_t needle_length,                         //
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata, //
    sz_size_t *matched_length) {

    // Validate inputs
    sz_assert_(needle_metadata && "needle_metadata must be provided");
    sz_assert_(needle_metadata->folded_slice_length > 0 && "folded window must be non-empty");
    sz_assert_(needle_metadata->offset_in_unfolded + needle_metadata->length_in_unfolded <= needle_length &&
               "window must be within needle");

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_assert_(folded_window_length <= 16 && "expect folded needle part to fit in XMM registers");

    // Pre-load folded window into XMM
    __mmask16 const folded_window_mask = sz_u16_mask_until_(folded_window_length);
    sz_u128_vec_t needle_window_vec, haystack_candidate_vec;
    needle_window_vec.xmm = _mm_loadu_si128((__m128i const *)needle_metadata->folded_slice);

    // 4 probe positions with meaningful names
    sz_size_t const offset_second = needle_metadata->probe_second;
    sz_size_t const offset_third = needle_metadata->probe_third;
    sz_size_t const offset_last = folded_window_length - 1;

    sz_u512_vec_t probe_first_vec, probe_second_vec, probe_third_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[0]);
    probe_second_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_second]);
    probe_third_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_third]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_last]);

    sz_u512_vec_t haystack_vec;
    sz_cptr_t haystack_ptr = haystack;
    sz_size_t const step = 64 - folded_window_length + 1;
    while (haystack_ptr + 64 <= haystack_end) {
        haystack_vec.zmm = sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(_mm512_loadu_si512(haystack_ptr));

        // 4-way probe filter
        sz_u64_t matches = _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_first_vec.zmm);
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_second_vec.zmm) >> offset_second;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_third_vec.zmm) >> offset_third;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_last_vec.zmm) >> offset_last;
        matches &= sz_u64_mask_until_(step);

        for (; matches; matches &= matches - 1) {
            sz_size_t candidate_offset = sz_u64_ctz(matches);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;

            // Verify window bytes match
            haystack_candidate_vec.xmm = _mm_maskz_loadu_epi8(folded_window_mask, haystack_candidate_ptr);
            haystack_candidate_vec.xmm = _mm512_castsi512_si128(
                sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(_mm512_castsi128_si512(haystack_candidate_vec.xmm)));

            __mmask16 window_mismatch_mask =
                _mm_mask_cmpneq_epi8_mask(folded_window_mask, haystack_candidate_vec.xmm, needle_window_vec.xmm);
            if (window_mismatch_mask) continue;

            // Validate the full match using the unified validator
            sz_cptr_t match = sz_utf8_case_insensitive_verify_match_(                    //
                haystack, haystack_length,                                               //
                needle, needle_length,                                                   //
                haystack_candidate_ptr - haystack, needle_metadata->folded_slice_length, // matched offset & length
                needle_metadata->offset_in_unfolded,                                     // head
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, // tail
                matched_length);
            if (match) {
                sz_utf8_case_insensitive_find_assert_(match, haystack, haystack_length, needle, needle_length,
                                                      needle_metadata);
                return match;
            }
        }
        haystack_ptr += step;
    }

    // Tail processing - remaining bytes < 64
    sz_size_t remaining = (sz_size_t)(haystack_end - haystack_ptr);
    if (remaining >= folded_window_length) {
        sz_size_t valid_starts = remaining - folded_window_length + 1;
        __mmask64 valid_mask = sz_u64_mask_until_(valid_starts);
        __mmask64 load_mask = sz_u64_mask_until_(remaining);

        haystack_vec.zmm =
            sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(_mm512_maskz_loadu_epi8(load_mask, haystack_ptr));

        sz_u64_t matches = _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_first_vec.zmm);
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_second_vec.zmm) >> offset_second;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_third_vec.zmm) >> offset_third;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_last_vec.zmm) >> offset_last;
        matches &= valid_mask;

        for (; matches; matches &= matches - 1) {
            sz_size_t candidate_offset = sz_u64_ctz(matches);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;

            haystack_candidate_vec.xmm = _mm_maskz_loadu_epi8(folded_window_mask, haystack_candidate_ptr);
            haystack_candidate_vec.xmm = _mm512_castsi512_si128(
                sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(_mm512_castsi128_si512(haystack_candidate_vec.xmm)));

            __mmask16 window_mismatch_mask =
                _mm_mask_cmpneq_epi8_mask(folded_window_mask, haystack_candidate_vec.xmm, needle_window_vec.xmm);
            if (window_mismatch_mask) continue;

            // Validate the full match using the unified validator
            sz_cptr_t match = sz_utf8_case_insensitive_verify_match_(                    //
                haystack, haystack_length,                                               //
                needle, needle_length,                                                   //
                haystack_candidate_ptr - haystack, needle_metadata->folded_slice_length, // matched offset & length
                needle_metadata->offset_in_unfolded,                                     // head
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, // tail
                matched_length);
            if (match) {
                sz_utf8_case_insensitive_find_assert_(match, haystack, haystack_length, needle, needle_length,
                                                      needle_metadata);
                return match;
            }
        }
    }

    sz_utf8_case_insensitive_find_assert_(SZ_NULL_CHAR, haystack, haystack_length, needle, needle_length,
                                          needle_metadata);
    return SZ_NULL_CHAR;
}

/**
 *  @brief 3-probe ASCII case-insensitive search using XOR + VPTERNLOG + VPTESTNMB.
 *
 *  For needles with folded_slice_length ≤ 3, probes at positions 0, mid, last cover ALL bytes.
 *  Uses parallel loads from different offsets, XOR for difference detection,
 *  VPTERNLOG to combine all 3, and VPTESTNMB to find matches.
 *  No window verification needed since probes cover the entire window.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_ice_ascii_3probe_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                     //
    sz_cptr_t needle, sz_size_t needle_length,                         //
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata, //
    sz_size_t *matched_length) {

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_size_t const step = 64 - folded_window_length + 1;

    // For ≤3 bytes: positions 0, mid, last cover ALL positions
    // 1-byte: 0=last, 2-byte: 0,1, 3-byte: 0,1,2
    sz_size_t const offset_second = folded_window_length / 2;
    sz_size_t const offset_last = folded_window_length - 1;

    // Broadcast probe bytes
    sz_u512_vec_t probe_first_vec, probe_second_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[0]);
    probe_second_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_second]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_last]);

    sz_u512_vec_t haystack_first_vec, haystack_second_vec, haystack_last_vec;
    sz_cptr_t haystack_ptr = haystack;

    // Main loop
    while (haystack_ptr + 64 + offset_last <= haystack_end) {
        // 3 parallel loads from different offsets - all hit L1 cache
        haystack_first_vec.zmm = sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(_mm512_loadu_si512(haystack_ptr));
        haystack_second_vec.zmm =
            sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(_mm512_loadu_si512(haystack_ptr + offset_second));
        haystack_last_vec.zmm =
            sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(_mm512_loadu_si512(haystack_ptr + offset_last));

        // XOR for difference detection - 0 where probe matches
        __m512i diff_first_zmm = _mm512_xor_si512(haystack_first_vec.zmm, probe_first_vec.zmm);
        __m512i diff_second_zmm = _mm512_xor_si512(haystack_second_vec.zmm, probe_second_vec.zmm);
        __m512i diff_last_zmm = _mm512_xor_si512(haystack_last_vec.zmm, probe_last_vec.zmm);

        // VPTERNLOG: 0xFE = A|B|C - combines all 3 in a single instruction
        __m512i combined_zmm = _mm512_ternarylogic_epi64(diff_first_zmm, diff_second_zmm, diff_last_zmm, 0xFE);

        // VPTESTNMB: find zero bytes (all 3 probes matched)
        __mmask64 matches_mask = _mm512_testn_epi8_mask(combined_zmm, combined_zmm);
        matches_mask &= sz_u64_mask_until_(step);

        for (; matches_mask; matches_mask &= matches_mask - 1) {
            sz_size_t candidate_offset = sz_u64_ctz(matches_mask);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;

            // No window verification needed - probes cover all positions
            // Go directly to head/tail validation
            sz_cptr_t match = sz_utf8_case_insensitive_verify_match_(                                      //
                haystack, haystack_length,                                                                 //
                needle, needle_length,                                                                     //
                haystack_candidate_ptr - haystack, folded_window_length,                                   //
                needle_metadata->offset_in_unfolded,                                                       //
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, //
                matched_length);
            if (match) {
                sz_utf8_case_insensitive_find_assert_(match, haystack, haystack_length, needle, needle_length,
                                                      needle_metadata);
                return match;
            }
        }
        haystack_ptr += step;
    }

    // Tail processing with masked loads
    sz_size_t remaining = (sz_size_t)(haystack_end - haystack_ptr);
    if (remaining >= folded_window_length) {
        sz_size_t valid_starts = remaining - folded_window_length + 1;
        __mmask64 valid_mask = sz_u64_mask_until_(valid_starts);

        __mmask64 load_first_mask = sz_u64_mask_until_(remaining);
        __mmask64 load_second_mask = (remaining > offset_second) ? sz_u64_mask_until_(remaining - offset_second) : 0;
        __mmask64 load_last_mask = (remaining > offset_last) ? sz_u64_mask_until_(remaining - offset_last) : 0;

        haystack_first_vec.zmm =
            sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(_mm512_maskz_loadu_epi8(load_first_mask, haystack_ptr));
        haystack_second_vec.zmm = sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(
            _mm512_maskz_loadu_epi8(load_second_mask, haystack_ptr + offset_second));
        haystack_last_vec.zmm = sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(
            _mm512_maskz_loadu_epi8(load_last_mask, haystack_ptr + offset_last));

        __m512i diff_first_zmm = _mm512_xor_si512(haystack_first_vec.zmm, probe_first_vec.zmm);
        __m512i diff_second_zmm = _mm512_xor_si512(haystack_second_vec.zmm, probe_second_vec.zmm);
        __m512i diff_last_zmm = _mm512_xor_si512(haystack_last_vec.zmm, probe_last_vec.zmm);

        __m512i combined_zmm = _mm512_ternarylogic_epi64(diff_first_zmm, diff_second_zmm, diff_last_zmm, 0xFE);
        __mmask64 matches_mask = _mm512_testn_epi8_mask(combined_zmm, combined_zmm);
        matches_mask &= valid_mask;

        for (; matches_mask; matches_mask &= matches_mask - 1) {
            sz_size_t candidate_offset = sz_u64_ctz(matches_mask);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;

            sz_cptr_t match = sz_utf8_case_insensitive_verify_match_(                                      //
                haystack, haystack_length,                                                                 //
                needle, needle_length,                                                                     //
                haystack_candidate_ptr - haystack, folded_window_length,                                   //
                needle_metadata->offset_in_unfolded,                                                       //
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, //
                matched_length);
            if (match) {
                sz_utf8_case_insensitive_find_assert_(match, haystack, haystack_length, needle, needle_length,
                                                      needle_metadata);
                return match;
            }
        }
    }

    sz_utf8_case_insensitive_find_assert_(SZ_NULL_CHAR, haystack, haystack_length, needle, needle_length,
                                          needle_metadata);
    return SZ_NULL_CHAR;
}

/**
 *  @brief 4-probe ASCII case-insensitive search using XOR + VPTERNLOG + VPTESTNMB.
 *
 *  For needles with folded_slice_length ≥ 4, probes at 4 positions filter candidates quickly.
 *  Uses parallel loads, XOR for difference detection, VPTERNLOG + VPOR to combine,
 *  and VPTESTNMB to find matches. Window verification IS required since probes
 *  don't cover all positions in the folded window.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_ice_ascii_4probe_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                     //
    sz_cptr_t needle, sz_size_t needle_length,                         //
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata, //
    sz_size_t *matched_length) {

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_size_t const step = 64 - folded_window_length + 1;

    // 4 probe positions from metadata
    sz_size_t const offset_second = needle_metadata->probe_second;
    sz_size_t const offset_third = needle_metadata->probe_third;
    sz_size_t const offset_last = folded_window_length - 1;

    // Pre-load folded window for full verification (probes only check 4 positions)
    __mmask16 const folded_window_mask = sz_u16_mask_until_(folded_window_length);
    sz_u128_vec_t needle_window_vec, haystack_candidate_vec;
    needle_window_vec.xmm = _mm_loadu_si128((__m128i const *)needle_metadata->folded_slice);

    // Broadcast probe bytes
    sz_u512_vec_t probe_first_vec, probe_second_vec, probe_third_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[0]);
    probe_second_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_second]);
    probe_third_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_third]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_last]);

    sz_u512_vec_t haystack_first_vec, haystack_second_vec, haystack_third_vec, haystack_last_vec;
    sz_cptr_t haystack_ptr = haystack;

    // Main loop
    while (haystack_ptr + 64 + offset_last <= haystack_end) {
        // 4 parallel loads from different offsets - all hit L1 cache
        haystack_first_vec.zmm = sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(_mm512_loadu_si512(haystack_ptr));
        haystack_second_vec.zmm =
            sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(_mm512_loadu_si512(haystack_ptr + offset_second));
        haystack_third_vec.zmm =
            sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(_mm512_loadu_si512(haystack_ptr + offset_third));
        haystack_last_vec.zmm =
            sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(_mm512_loadu_si512(haystack_ptr + offset_last));

        // XOR for difference detection - 0 where probe matches
        __m512i diff_first_zmm = _mm512_xor_si512(haystack_first_vec.zmm, probe_first_vec.zmm);
        __m512i diff_second_zmm = _mm512_xor_si512(haystack_second_vec.zmm, probe_second_vec.zmm);
        __m512i diff_third_zmm = _mm512_xor_si512(haystack_third_vec.zmm, probe_third_vec.zmm);
        __m512i diff_last_zmm = _mm512_xor_si512(haystack_last_vec.zmm, probe_last_vec.zmm);

        // VPTERNLOG: 0xFE = A|B|C for first 3, then OR the 4th
        __m512i combined_zmm = _mm512_ternarylogic_epi64(diff_first_zmm, diff_second_zmm, diff_third_zmm, 0xFE);
        combined_zmm = _mm512_or_si512(combined_zmm, diff_last_zmm);

        // VPTESTNMB: find zero bytes (all 4 probes matched)
        __mmask64 matches_mask = _mm512_testn_epi8_mask(combined_zmm, combined_zmm);
        matches_mask &= sz_u64_mask_until_(step);

        for (; matches_mask; matches_mask &= matches_mask - 1) {
            sz_size_t candidate_offset = sz_u64_ctz(matches_mask);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;

            // Verify full window - probes don't cover all positions
            haystack_candidate_vec.xmm = _mm_maskz_loadu_epi8(folded_window_mask, haystack_candidate_ptr);
            haystack_candidate_vec.xmm = _mm512_castsi512_si128(
                sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(_mm512_castsi128_si512(haystack_candidate_vec.xmm)));

            __mmask16 window_mismatch_mask =
                _mm_mask_cmpneq_epi8_mask(folded_window_mask, haystack_candidate_vec.xmm, needle_window_vec.xmm);
            if (window_mismatch_mask) continue;

            // Window matched - validate head/tail
            sz_cptr_t match = sz_utf8_case_insensitive_verify_match_(                                      //
                haystack, haystack_length,                                                                 //
                needle, needle_length,                                                                     //
                haystack_candidate_ptr - haystack, folded_window_length,                                   //
                needle_metadata->offset_in_unfolded,                                                       //
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, //
                matched_length);
            if (match) {
                sz_utf8_case_insensitive_find_assert_(match, haystack, haystack_length, needle, needle_length,
                                                      needle_metadata);
                return match;
            }
        }
        haystack_ptr += step;
    }

    // Tail processing with masked loads
    sz_size_t remaining = (sz_size_t)(haystack_end - haystack_ptr);
    if (remaining >= folded_window_length) {
        sz_size_t valid_starts = remaining - folded_window_length + 1;
        __mmask64 valid_mask = sz_u64_mask_until_(valid_starts);

        __mmask64 load_first_mask = sz_u64_mask_until_(remaining);
        __mmask64 load_second_mask = (remaining > offset_second) ? sz_u64_mask_until_(remaining - offset_second) : 0;
        __mmask64 load_third_mask = (remaining > offset_third) ? sz_u64_mask_until_(remaining - offset_third) : 0;
        __mmask64 load_last_mask = (remaining > offset_last) ? sz_u64_mask_until_(remaining - offset_last) : 0;

        haystack_first_vec.zmm =
            sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(_mm512_maskz_loadu_epi8(load_first_mask, haystack_ptr));
        haystack_second_vec.zmm = sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(
            _mm512_maskz_loadu_epi8(load_second_mask, haystack_ptr + offset_second));
        haystack_third_vec.zmm = sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(
            _mm512_maskz_loadu_epi8(load_third_mask, haystack_ptr + offset_third));
        haystack_last_vec.zmm = sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(
            _mm512_maskz_loadu_epi8(load_last_mask, haystack_ptr + offset_last));

        __m512i diff_first_zmm = _mm512_xor_si512(haystack_first_vec.zmm, probe_first_vec.zmm);
        __m512i diff_second_zmm = _mm512_xor_si512(haystack_second_vec.zmm, probe_second_vec.zmm);
        __m512i diff_third_zmm = _mm512_xor_si512(haystack_third_vec.zmm, probe_third_vec.zmm);
        __m512i diff_last_zmm = _mm512_xor_si512(haystack_last_vec.zmm, probe_last_vec.zmm);

        __m512i combined_zmm = _mm512_ternarylogic_epi64(diff_first_zmm, diff_second_zmm, diff_third_zmm, 0xFE);
        combined_zmm = _mm512_or_si512(combined_zmm, diff_last_zmm);
        __mmask64 matches_mask = _mm512_testn_epi8_mask(combined_zmm, combined_zmm);
        matches_mask &= valid_mask;

        for (; matches_mask; matches_mask &= matches_mask - 1) {
            sz_size_t candidate_offset = sz_u64_ctz(matches_mask);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;

            // Verify full window
            haystack_candidate_vec.xmm = _mm_maskz_loadu_epi8(folded_window_mask, haystack_candidate_ptr);
            haystack_candidate_vec.xmm = _mm512_castsi512_si128(
                sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(_mm512_castsi128_si512(haystack_candidate_vec.xmm)));

            __mmask16 window_mismatch_mask =
                _mm_mask_cmpneq_epi8_mask(folded_window_mask, haystack_candidate_vec.xmm, needle_window_vec.xmm);
            if (window_mismatch_mask) continue;

            sz_cptr_t match = sz_utf8_case_insensitive_verify_match_(                                      //
                haystack, haystack_length,                                                                 //
                needle, needle_length,                                                                     //
                haystack_candidate_ptr - haystack, folded_window_length,                                   //
                needle_metadata->offset_in_unfolded,                                                       //
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, //
                matched_length);
            if (match) {
                sz_utf8_case_insensitive_find_assert_(match, haystack, haystack_length, needle, needle_length,
                                                      needle_metadata);
                return match;
            }
        }
    }

    sz_utf8_case_insensitive_find_assert_(SZ_NULL_CHAR, haystack, haystack_length, needle, needle_length,
                                          needle_metadata);
    return SZ_NULL_CHAR;
}

#pragma endregion // ASCII Case-Insensitive Find

#pragma region Western European Case-Insensitive Find

/**
 *  @brief Fold a ZMM register using Western European case-folding rules.
 *  @sa sz_utf8_case_rune_safe_western_europe_k
 *
 *  @param[in] text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_case_insensitive_find_ice_western_europe_fold_naively_zmm_(__m512i text_zmm) {
    // Start with ASCII folded
    __m512i result_zmm = sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(text_zmm);

    // Constants for Latin folding
    __m512i const x_20_zmm = _mm512_set1_epi8(0x20);
    __m512i const x_73_zmm = _mm512_set1_epi8('s');

    // Constants for Latin-1 Supplement (C3 lead byte)
    // Note: µ (Micro Sign, C2 B5) is BANNED - needles with µ use serial fallback
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3); // Latin-1 Supplement (upper half)
    __m512i const x_9f_zmm = _mm512_set1_epi8((char)0x9F); // 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)

    // Range Logic Constants for Uppercase Detection (C3 80..9E):
    __m512i const x_80_zmm = _mm512_set1_epi8((char)0x80); // Lower bound of the range (offset)
    __m512i const x_1f_zmm = _mm512_set1_epi8((char)0x1F); // Length of the relevant range (0x9F - 0x80 = 0x1F)

    // Explicit Exclusions from Range Folding:
    // Note: '÷' Division Sign (C3 B7) is outside the uppercase range (0x80..0x9E), so it's safe.
    __m512i const x_97_zmm = _mm512_set1_epi8((char)0x97); // '×' Multiplication Sign (C3 97) - has no case

    // 1. Handle Eszett: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
    __mmask64 is_c3_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c3_zmm);
    __mmask64 is_after_c3_mask = is_c3_mask << 1;
    __mmask64 is_eszett_second_mask = _mm512_mask_cmpeq_epi8_mask(is_after_c3_mask, text_zmm, x_9f_zmm);
    __mmask64 is_eszett_mask = is_eszett_second_mask | (is_eszett_second_mask >> 1);
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_eszett_mask, x_73_zmm);

    // 2. Handle Latin-1 supplement uppercase letters (C3 80-9E) → add 0x20
    //    We need to map:
    //    - 'À' (C3 80) ... 'Þ' (C3 9E) to 'à' (C3 A0) ... 'þ' (C3 BE)
    //    Exceptions:
    //    - 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73) is already handled above
    //    - '×' (C3 97) is the Multiplication Sign, no case variant (so exclude it)
    __mmask64 is_97_mask = _mm512_mask_cmpeq_epi8_mask(is_after_c3_mask, text_zmm, x_97_zmm);
    __mmask64 is_latin1_upper_mask = _mm512_mask_cmplt_epu8_mask(
        is_after_c3_mask & ~is_eszett_second_mask & ~is_97_mask, _mm512_sub_epi8(text_zmm, x_80_zmm), x_1f_zmm);

    // Apply all folding transforms
    // +0x20 for Latin-1 (ASCII already handled in the base call)
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_latin1_upper_mask, result_zmm, x_20_zmm);
    return result_zmm;
}

/**
 *  @brief Fold a ZMM register using Western European case-folding rules.
 *  @sa sz_utf8_case_rune_safe_western_europe_k
 *
 *  @param[in] text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_case_insensitive_find_ice_western_europe_fold_efficiently_zmm_(__m512i text_zmm) {
    // Start with ASCII folded
    __m512i result_zmm = sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(text_zmm);

    // Constants for Latin folding
    __m512i const x_20_zmm = _mm512_set1_epi8(0x20);
    __m512i const x_73_zmm = _mm512_set1_epi8('s');

    // Constants for Latin-1 Supplement (C3 lead byte)
    // Note: µ (Micro Sign, C2 B5) is BANNED - needles with µ use serial fallback
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3); // Latin-1 Supplement (upper half)
    __m512i const x_9f_zmm = _mm512_set1_epi8((char)0x9F); // 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)

    // Range Logic Constants for Uppercase Detection (C3 80..9E):
    __m512i const x_80_zmm = _mm512_set1_epi8((char)0x80); // Lower bound of the range (offset)
    __m512i const x_1f_zmm = _mm512_set1_epi8((char)0x1F); // Length of the relevant range (0x9F - 0x80 = 0x1F)

    // Explicit Exclusions from Range Folding:
    // Note: '÷' Division Sign (C3 B7) is outside the uppercase range (0x80..0x9E), so it's safe.
    __m512i const x_97_zmm = _mm512_set1_epi8((char)0x97); // '×' Multiplication Sign (C3 97) - has no case

    // 1. Handle Eszett: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
    __mmask64 is_c3_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c3_zmm);
    __mmask64 is_after_c3_mask = is_c3_mask << 1;
    __mmask64 is_eszett_second_mask = _mm512_mask_cmpeq_epi8_mask(is_after_c3_mask, text_zmm, x_9f_zmm);
    __mmask64 is_eszett_mask = is_eszett_second_mask | (is_eszett_second_mask >> 1);
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_eszett_mask, x_73_zmm);

    // 2. Handle Latin-1 supplement uppercase letters (C3 80-9E) → add 0x20
    //    We need to map:
    //    - 'À' (C3 80) ... 'Þ' (C3 9E) to 'à' (C3 A0) ... 'þ' (C3 BE)
    //    Exceptions:
    //    - 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73) is already handled above
    //    - '×' (C3 97) is the Multiplication Sign, no case variant (so exclude it)
    __mmask64 is_97_mask = _mm512_mask_cmpeq_epi8_mask(is_after_c3_mask, text_zmm, x_97_zmm);
    __mmask64 is_latin1_upper_mask = _mm512_mask_cmplt_epu8_mask(
        is_after_c3_mask & ~is_eszett_second_mask & ~is_97_mask, _mm512_sub_epi8(text_zmm, x_80_zmm), x_1f_zmm);

    // Apply all folding transforms
    // +0x20 for Latin-1 (ASCII already handled in the base call)
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_latin1_upper_mask, result_zmm, x_20_zmm);

    sz_assert_(_mm512_cmpeq_epi8_mask(sz_utf8_case_insensitive_find_ice_western_europe_fold_naively_zmm_(text_zmm),
                                      result_zmm) == (__mmask64)-1 &&
               "Western European folding optimized and naive results must match");
    return result_zmm;
}

/**
 *  @brief Naive alarm function for Western Europe danger zone detection.
 *
 *  Detects positions where danger characters occur that require special handling:
 *  - E1 BA 9E: 'ẞ' (U+1E9E, E1 BA 9E) → "ss" (U+0073 U+0073, 73 73) (3 bytes → 2 bytes)
 *  - E2 84 AA: 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B) (3 bytes → 1 byte)
 *  - E2 84 AB: 'Å' (U+212B, E2 84 AB) → 'å' (U+00E5, C3 A5) (3 bytes → 2 bytes)
 *  - EF AC 80-86: Ligatures:
 *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
 *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
 *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
 *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
 *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
 *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
 *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
 *  - C5 BF: 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73) (2 bytes → 1 byte)
 *  - C3 9F: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73) (2 bytes → 2 bytes, 1 rune → 2 runes)
 *
 *  Uses 12 CMPEQ operations (5 lead + 7 second byte checks).
 *
 *  @param[in] text_zmm The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_INTERNAL __mmask64 sz_utf8_case_insensitive_find_ice_western_europe_alarm_naively_zmm_(__m512i text_zmm) {
    // Lead byte constants
    __m512i const x_e1_zmm = _mm512_set1_epi8((char)0xE1);
    __m512i const x_e2_zmm = _mm512_set1_epi8((char)0xE2);
    __m512i const x_ef_zmm = _mm512_set1_epi8((char)0xEF);
    __m512i const x_c5_zmm = _mm512_set1_epi8((char)0xC5);
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3);

    // Second/third byte constants
    __m512i const x_ba_zmm = _mm512_set1_epi8((char)0xBA);
    __m512i const x_84_zmm = _mm512_set1_epi8((char)0x84);
    __m512i const x_ac_zmm = _mm512_set1_epi8((char)0xAC);
    __m512i const x_bf_zmm = _mm512_set1_epi8((char)0xBF);
    __m512i const x_9f_zmm = _mm512_set1_epi8((char)0x9F);
    __m512i const x_aa_zmm = _mm512_set1_epi8((char)0xAA);
    __m512i const x_ab_zmm = _mm512_set1_epi8((char)0xAB);

    // Lead bytes (5 CMPEQ)
    __mmask64 is_e1_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_e1_zmm);
    __mmask64 is_e2_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_e2_zmm);
    __mmask64 is_ef_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ef_zmm);
    __mmask64 is_c5_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c5_zmm);
    __mmask64 is_c3_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c3_zmm);

    // Second/third bytes (7 CMPEQ)
    __mmask64 is_ba_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ba_zmm);
    __mmask64 is_84_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_84_zmm);
    __mmask64 is_ac_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ac_zmm);
    __mmask64 is_bf_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_bf_zmm);
    __mmask64 is_9f_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_9f_zmm);
    __mmask64 is_aa_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_aa_zmm);
    __mmask64 is_ab_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ab_zmm);

    // Danger mask construction (checking third byte for E2 84 to avoid false positives)
    return ((is_e1_mask << 1) & is_ba_mask) |                                    // Capital Sharp S (E1 BA 9E)
           ((is_e2_mask << 1) & is_84_mask & ((is_aa_mask | is_ab_mask) >> 1)) | // Kelvin/Angstrom (E2 84 AA/AB)
           ((is_ef_mask << 1) & is_ac_mask) |                                    // Ligatures (EF AC xx)
           ((is_c5_mask << 1) & is_bf_mask) |                                    // Long S (C5 BF)
           ((is_c3_mask << 1) & is_9f_mask);                                     // Sharp S (C3 9F)
}

/**
 *  @brief Optimized alarm function for Western Europe danger zone detection.
 *
 *  Reduces port 5 pressure from 12 to 8 operations by:
 *  - E1/E2 consecutive: 2 CMPEQ -> 1 CMPLT + 1 VPTESTNMB (p0)
 *  - AA/AB consecutive: 2 CMPEQ -> 1 CMPLT
 *  - C3/C5 range: 2 CMPEQ -> 1 CMPLT + 2 VPTESTNMB (p0)
 *  - 9F/BF bit masking: 2 CMPEQ -> 1 CMPEQ + 1 VPTESTMB (p0)
 *
 *  Port summary: 8 p5 ops + 4 p0 ops (vs 12 p5 originally)
 *
 *  @param[in] text_zmm The text ZMM register to scan for danger bytes.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_INTERNAL __mmask64 sz_utf8_case_insensitive_find_ice_western_europe_alarm_efficiently_zmm_(__m512i text_zmm) {
    // Range constants
    __m512i const x_e1_zmm = _mm512_set1_epi8((char)0xE1);
    __m512i const x_ef_zmm = _mm512_set1_epi8((char)0xEF);
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3);
    __m512i const x_ba_zmm = _mm512_set1_epi8((char)0xBA);
    __m512i const x_84_zmm = _mm512_set1_epi8((char)0x84);
    __m512i const x_ac_zmm = _mm512_set1_epi8((char)0xAC);
    __m512i const x_bf_zmm = _mm512_set1_epi8((char)0xBF);
    __m512i const x_aa_zmm = _mm512_set1_epi8((char)0xAA);
    __m512i const x_20_zmm = _mm512_set1_epi8(0x20);
    __m512i const x_02_zmm = _mm512_set1_epi8(0x02);
    __m512i const x_03_zmm = _mm512_set1_epi8(0x03);

    // Check for E1/E2 range: (byte - 0xE1) < 2  [1 CMPLT on p5]
    __m512i off_e1_zmm = _mm512_sub_epi8(text_zmm, x_e1_zmm);
    __mmask64 is_e1_or_e2_mask = _mm512_cmplt_epu8_mask(off_e1_zmm, x_02_zmm);
    __mmask64 is_e1_mask = is_e1_or_e2_mask & _mm512_testn_epi8_mask(off_e1_zmm, off_e1_zmm); // offset==0 [p0]
    __mmask64 is_e2_mask = is_e1_or_e2_mask & ~is_e1_mask;

    // Check for AA/AB range: (byte - 0xAA) < 2  [1 CMPLT on p5]
    __m512i off_aa_zmm = _mm512_sub_epi8(text_zmm, x_aa_zmm);
    __mmask64 is_aa_or_ab_mask = _mm512_cmplt_epu8_mask(off_aa_zmm, x_02_zmm);

    // Check for C3/C4/C5 range: (byte - 0xC3) < 3  [1 CMPLT on p5]
    // We only need C3 and C5; C4 is captured but unused
    __m512i off_c3_zmm = _mm512_sub_epi8(text_zmm, x_c3_zmm);
    __mmask64 is_c3_c4_c5_mask = _mm512_cmplt_epu8_mask(off_c3_zmm, x_03_zmm);
    __mmask64 is_c3_mask = is_c3_c4_c5_mask & _mm512_testn_epi8_mask(off_c3_zmm, off_c3_zmm); // offset==0 [p0]
    __m512i off_xor_2_zmm = _mm512_xor_si512(off_c3_zmm, x_02_zmm);
    __mmask64 is_c5_mask = is_c3_c4_c5_mask & _mm512_testn_epi8_mask(off_xor_2_zmm, off_xor_2_zmm); // offset==2 [p0]

    // Other lead byte (1 CMPEQ on p5)
    __mmask64 is_ef_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ef_zmm);

    // Second bytes: BA, 84, AC (3 CMPEQ on p5)
    __mmask64 is_ba_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ba_zmm);
    __mmask64 is_84_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_84_zmm);
    __mmask64 is_ac_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ac_zmm);

    // 9F/BF bit masking: (byte | 0x20) == 0xBF catches exactly {0x9F, 0xBF}  [1 CMPEQ on p5]
    // 0x9F = 1001_1111, 0xBF = 1011_1111, differ only in bit 5
    __m512i masked_zmm = _mm512_or_si512(text_zmm, x_20_zmm);
    __mmask64 is_9f_or_bf_mask = _mm512_cmpeq_epi8_mask(masked_zmm, x_bf_zmm);
    __mmask64 has_bit5_mask = _mm512_test_epi8_mask(text_zmm, x_20_zmm); // VPTESTMB [p0]
    __mmask64 is_bf_mask = is_9f_or_bf_mask & has_bit5_mask;
    __mmask64 is_9f_mask = is_9f_or_bf_mask & ~has_bit5_mask;

    // Danger mask construction
    __mmask64 danger_mask =
        ((is_e1_mask << 1) & is_ba_mask) |                           // Capital Sharp S (E1 BA xx)
        ((is_e2_mask << 1) & is_84_mask & (is_aa_or_ab_mask >> 1)) | // Kelvin/Angstrom (E2 84 AA/AB)
        ((is_ef_mask << 1) & is_ac_mask) |                           // Ligatures (EF AC xx)
        ((is_c5_mask << 1) & is_bf_mask) |                           // Long S (C5 BF)
        ((is_c3_mask << 1) & is_9f_mask);                            // Sharp S (C3 9F)

    sz_assert_(danger_mask == sz_utf8_case_insensitive_find_ice_western_europe_alarm_naively_zmm_(text_zmm) &&
               "Efficient Western Europe alarm must match naive implementation");
    return danger_mask;
}

/**
 *  @brief Western European case-insensitive search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_case_rune_safe_western_europe_k
 *
 *  Scans the entire haystack from byte 0, looking for the folded window pattern.
 *  When found, verifies the head (backwards) and tail (forwards) using codepoint-by-codepoint
 *  comparison to handle variable-width folding correctly (e.g., 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)).
 *
 *  @param[in] haystack Pointer to the haystack string.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] needle Pointer to the full needle string.
 *  @param[in] needle_length Length of the full needle in bytes.
 *  @param[in] needle_metadata Pre-folded window content with probe positions.
 *  @param[out] matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_ice_western_europe_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                       //
    sz_cptr_t needle, sz_size_t needle_length,                           //
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata,   //
    sz_size_t *matched_length) {

    // Validate inputs
    sz_assert_(needle_metadata && "needle_metadata must be provided");
    sz_assert_(needle_metadata->folded_slice_length > 0 && "folded window must be non-empty");

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_assert_(folded_window_length <= 16 && "expect folded needle part to fit in XMM registers");

    // Pre-load folded window into XMM
    __mmask16 const folded_window_mask = sz_u16_mask_until_(folded_window_length);
    sz_u128_vec_t needle_window_vec, haystack_candidate_vec;
    needle_window_vec.xmm = _mm_loadu_si128((__m128i const *)needle_metadata->folded_slice);

    // 4 probe positions
    sz_size_t const offset_second = needle_metadata->probe_second;
    sz_size_t const offset_third = needle_metadata->probe_third;
    sz_size_t const offset_last = folded_window_length - 1;

    sz_u512_vec_t probe_first_vec, probe_second_vec, probe_third_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[0]);
    probe_second_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_second]);
    probe_third_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_third]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_last]);

    // Pre-load the first folded rune for danger zone matching
    sz_rune_t needle_first_safe_folded_rune;
    {
        sz_rune_length_t dummy;
        sz_rune_parse((sz_cptr_t)(needle_metadata->folded_slice), &needle_first_safe_folded_rune, &dummy);
    }

    sz_u512_vec_t haystack_vec;
    sz_cptr_t haystack_ptr = haystack;
    while (haystack_ptr < haystack_end) {
        sz_size_t const available = (sz_size_t)(haystack_end - haystack_ptr);
        if (available < folded_window_length) break;

        sz_size_t const chunk_size = available < 64 ? available : 64;
        sz_size_t const valid_starts = chunk_size - folded_window_length + 1;
        // For danger detection across chunk boundaries, reduce step size to ensure
        // 3-byte patterns at chunk end are fully visible in the next chunk.
        // For tail chunks (valid_starts <= 2), step = 1 ensures progress.
        sz_size_t const step = valid_starts > 2 ? valid_starts - 2 : 1;
        __mmask64 const load_mask = sz_u64_mask_until_(chunk_size);
        __mmask64 const valid_mask = sz_u64_mask_until_(valid_starts);

        haystack_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, haystack_ptr);

        // Check for anomalies using the alarm function
        __mmask64 danger_mask =
            sz_utf8_case_insensitive_find_ice_western_europe_alarm_efficiently_zmm_(haystack_vec.zmm);

        if (danger_mask) {
            // The danger zone handler scans for the needle's first safe rune (at offset_in_unfolded).
            // To find all matches that start at valid positions (0 to valid_starts-1), we need to
            // scan up to position valid_starts - 1 + offset_in_unfolded. Use chunk_size as upper bound.
            sz_size_t danger_scan_length =
                sz_min_of_two(valid_starts + needle_metadata->offset_in_unfolded, chunk_size);
            sz_cptr_t match = sz_utf8_case_insensitive_find_in_danger_zone_( //
                haystack, haystack_length,                                   //
                needle, needle_length,                                       //
                haystack_ptr, danger_scan_length,                            // extended danger zone
                needle_first_safe_folded_rune,                               // pivot point
                needle_metadata->offset_in_unfolded,                         // its location in the needle
                matched_length);
            if (match) return match;
            haystack_ptr += step;
            continue;
        }

        // Fold and 4-way probe filter
        haystack_vec.zmm = sz_utf8_case_insensitive_find_ice_western_europe_fold_efficiently_zmm_(haystack_vec.zmm);

        sz_u64_t matches = _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_first_vec.zmm);
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_second_vec.zmm) >> offset_second;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_third_vec.zmm) >> offset_third;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_last_vec.zmm) >> offset_last;
        matches &= valid_mask;

        for (; matches; matches &= matches - 1) {
            sz_size_t candidate_offset = sz_u64_ctz(matches);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;

            haystack_candidate_vec.xmm = _mm_maskz_loadu_epi8(folded_window_mask, haystack_candidate_ptr);
            haystack_candidate_vec.xmm =
                _mm512_castsi512_si128(sz_utf8_case_insensitive_find_ice_western_europe_fold_efficiently_zmm_(
                    _mm512_castsi128_si512(haystack_candidate_vec.xmm)));

            __mmask16 window_mismatch_mask =
                _mm_mask_cmpneq_epi8_mask(folded_window_mask, haystack_candidate_vec.xmm, needle_window_vec.xmm);
            if (window_mismatch_mask) continue;

            sz_cptr_t match = sz_utf8_case_insensitive_verify_match_(                    //
                haystack, haystack_length,                                               //
                needle, needle_length,                                                   //
                haystack_candidate_ptr - haystack, needle_metadata->folded_slice_length, // matched offset & length
                needle_metadata->offset_in_unfolded,                                     // head
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, // tail
                matched_length);
            if (match) {
                sz_utf8_case_insensitive_find_assert_(match, haystack, haystack_length, needle, needle_length,
                                                      needle_metadata);
                return match;
            }
        }
        haystack_ptr += step;
    }

    sz_utf8_case_insensitive_find_assert_(SZ_NULL_CHAR, haystack, haystack_length, needle, needle_length,
                                          needle_metadata);
    return SZ_NULL_CHAR;
}

#pragma endregion // Western European Case-Insensitive Find

#pragma region Central European Case-Insensitive Find

/**
 *  @brief Fold a ZMM register using Central European case-folding rules.
 *  @sa sz_utf8_case_rune_safe_central_europe_k
 *
 *  @param[in] text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_case_insensitive_find_ice_central_europe_fold_naively_zmm_(__m512i text_zmm) {
    // Start with ASCII folded
    __m512i result_zmm = sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(text_zmm);

    // Constants for Latin folding
    __m512i const x_20_zmm = _mm512_set1_epi8((char)0x20); // 32
    __m512i const x_01_zmm = _mm512_set1_epi8((char)0x01); // 1

    // Constants for Latin-1 Supplement (C3 lead byte)
    // Range C3 80-9E (Uppercases), excluding C3 97 (×) and C3 9F (ß - not folded here)
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3);
    __m512i const x_80_zmm = _mm512_set1_epi8((char)0x80);
    __m512i const x_9e_zmm = _mm512_set1_epi8((char)0x9E);
    __m512i const x_97_zmm = _mm512_set1_epi8((char)0x97);

    // Constants for Latin Extended-A (C4, C5 lead bytes)
    __m512i const x_c4_zmm = _mm512_set1_epi8((char)0xC4);
    __m512i const x_c5_zmm = _mm512_set1_epi8((char)0xC5);

    // 1. Latin-1 Supplement: C3 80-9E -> +0x20
    //    We check for C3 lead, then check if 2nd byte is in [80, 9E] and != 97
    __mmask64 is_c3_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c3_zmm);
    __mmask64 is_after_c3_mask = is_c3_mask << 1;

    __mmask64 is_latin1_range = _mm512_mask_cmpge_epu8_mask(is_after_c3_mask, result_zmm, x_80_zmm);
    is_latin1_range &= _mm512_mask_cmple_epu8_mask(is_after_c3_mask, result_zmm, x_9e_zmm);
    __mmask64 is_97 = _mm512_mask_cmpeq_epi8_mask(is_after_c3_mask, result_zmm, x_97_zmm);

    __mmask64 fold_latin1 = is_latin1_range & ~is_97;
    result_zmm = _mm512_mask_add_epi8(result_zmm, fold_latin1, result_zmm, x_20_zmm);

    // 2. Latin Extended-A: C4xx / C5xx case folding
    //    The uppercase/lowercase parity pattern varies within Latin Extended-A:
    //    - C4 range (U+0100-U+013F): uppercase = EVEN second bytes
    //    - C5 81-87 (U+0141-U+0147): uppercase = ODD (Ł,Ń,Ņ,Ň → +1)
    //    - C5 8A-B6 (U+014A-U+0176): uppercase = EVEN (Ŋ-Ŷ → +1)
    //    - C5 B9-BD (U+0179-U+017D): uppercase = ODD (Ź,Ż,Ž → +1)
    //    NOT folded (handled elsewhere or excluded):
    //    - 'ŀ' (U+0140, C5 80)
    //    - 'ň' (U+0148, C5 88)
    //    - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
    //    - 'ŷ' (U+0177, C5 B7)
    //    - 'Ÿ' (U+0178, C5 B8) → 'ÿ' (U+00FF, C3 BF)
    //    - 'ž' (U+017E, C5 BE)
    //    - 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
    __mmask64 is_c4_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c4_zmm);
    __mmask64 is_c5_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c5_zmm);
    __mmask64 is_after_c4_mask = is_c4_mask << 1;
    __mmask64 is_after_c5_mask = is_c5_mask << 1;

    __mmask64 is_even_mask = _mm512_testn_epi8_mask(result_zmm, x_01_zmm); // (val & 1) == 0
    __mmask64 is_odd_mask = ~is_even_mask;

    // C5 sub-range detection for second bytes
    __m512i const x_81_zmm = _mm512_set1_epi8((char)0x81);
    __m512i const x_87_zmm = _mm512_set1_epi8((char)0x87);
    __m512i const x_8a_zmm = _mm512_set1_epi8((char)0x8A);
    __m512i const x_b6_zmm = _mm512_set1_epi8((char)0xB6);
    __m512i const x_b9_zmm = _mm512_set1_epi8((char)0xB9);
    __m512i const x_bd_zmm = _mm512_set1_epi8((char)0xBD);

    // C5 81-87: odd = uppercase (Ł,Ń,Ņ,Ň)
    __mmask64 is_c5_81_87 = _mm512_mask_cmpge_epu8_mask(is_after_c5_mask, result_zmm, x_81_zmm);
    is_c5_81_87 &= _mm512_mask_cmple_epu8_mask(is_after_c5_mask, result_zmm, x_87_zmm);

    // C5 8A-B6: even = uppercase (Ŋ-Ŷ)
    __mmask64 is_c5_8a_b6 = _mm512_mask_cmpge_epu8_mask(is_after_c5_mask, result_zmm, x_8a_zmm);
    is_c5_8a_b6 &= _mm512_mask_cmple_epu8_mask(is_after_c5_mask, result_zmm, x_b6_zmm);

    // C5 B9-BD: odd = uppercase (Ź,Ż,Ž)
    __mmask64 is_c5_b9_bd = _mm512_mask_cmpge_epu8_mask(is_after_c5_mask, result_zmm, x_b9_zmm);
    is_c5_b9_bd &= _mm512_mask_cmple_epu8_mask(is_after_c5_mask, result_zmm, x_bd_zmm);

    // Fold: C4 even, C5 81-87 odd, C5 8A-B6 even, C5 B9-BD odd
    __mmask64 fold_latext = (is_after_c4_mask & is_even_mask) | (is_c5_81_87 & is_odd_mask) |
                            (is_c5_8a_b6 & is_even_mask) | (is_c5_b9_bd & is_odd_mask);

    result_zmm = _mm512_mask_add_epi8(result_zmm, fold_latext, result_zmm, x_01_zmm);

    return result_zmm;
}

/**
 *  @brief Fold a ZMM register using Central European case-folding rules.
 *  @sa sz_utf8_case_rune_safe_central_europe_k
 *
 *  @param[in] text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_case_insensitive_find_ice_central_europe_fold_efficiently_zmm_(__m512i text_zmm) {
    // Inline ASCII fold (avoiding function call overhead)
    __m512i const a_upper_zmm = _mm512_set1_epi8('A');
    __m512i const range26_zmm = _mm512_set1_epi8(26);
    __m512i const x_20_zmm = _mm512_set1_epi8(0x20);
    __m512i const x_01_zmm = _mm512_set1_epi8(0x01);
    __mmask64 is_upper_mask = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_zmm, a_upper_zmm), range26_zmm);
    __m512i result_zmm = _mm512_mask_add_epi8(text_zmm, is_upper_mask, text_zmm, x_20_zmm);

    // Lead bytes
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3);
    __m512i const x_c4_zmm = _mm512_set1_epi8((char)0xC4);
    __m512i const x_c5_zmm = _mm512_set1_epi8((char)0xC5);

    // Range compression constants
    __m512i const x_80_zmm = _mm512_set1_epi8((char)0x80);
    __m512i const x_1f_zmm = _mm512_set1_epi8((char)0x1F); // Range 80-9E (31 bytes)
    __m512i const x_97_zmm = _mm512_set1_epi8((char)0x97);
    __m512i const x_81_zmm = _mm512_set1_epi8((char)0x81);
    __m512i const x_07_zmm = _mm512_set1_epi8((char)0x07); // Range 81-87 (7 bytes)
    __m512i const x_8a_zmm = _mm512_set1_epi8((char)0x8A);
    __m512i const x_2d_zmm = _mm512_set1_epi8((char)0x2D); // Range 8A-B6 (45 bytes)
    __m512i const x_b9_zmm = _mm512_set1_epi8((char)0xB9);
    __m512i const x_05_zmm = _mm512_set1_epi8((char)0x05); // Range B9-BD (5 bytes)

    __mmask64 is_c3_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c3_zmm);
    __mmask64 is_c4_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c4_zmm);
    __mmask64 is_c5_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c5_zmm);
    __mmask64 is_after_c3_mask = is_c3_mask << 1;
    __mmask64 is_after_c4_mask = is_c4_mask << 1;
    __mmask64 is_after_c5_mask = is_c5_mask << 1;

    // 1. Latin-1 Supplement: C3 80-9E -> +0x20 (excluding 97)
    // Range compression: (byte - 0x80) < 0x1F covers 80-9E
    __mmask64 is_latin1_range =
        _mm512_mask_cmplt_epu8_mask(is_after_c3_mask, _mm512_sub_epi8(result_zmm, x_80_zmm), x_1f_zmm);
    __mmask64 fold_latin1 = is_latin1_range & ~_mm512_cmpeq_epi8_mask(result_zmm, x_97_zmm);

    // 2. Latin Extended-A: C4xx / C5xx case folding with varying parity
    __mmask64 is_even_mask = _mm512_testn_epi8_mask(result_zmm, x_01_zmm);
    __mmask64 is_odd_mask = ~is_even_mask;

    // RANGE COMPRESSION for C5 sub-ranges
    // C5 81-87: (byte - 81) < 7
    __mmask64 is_c5_81_87 = is_after_c5_mask & _mm512_cmplt_epu8_mask(_mm512_sub_epi8(result_zmm, x_81_zmm), x_07_zmm);
    // C5 8A-B6: (byte - 8A) < 45
    __mmask64 is_c5_8a_b6 = is_after_c5_mask & _mm512_cmplt_epu8_mask(_mm512_sub_epi8(result_zmm, x_8a_zmm), x_2d_zmm);
    // C5 B9-BD: (byte - B9) < 5
    __mmask64 is_c5_b9_bd = is_after_c5_mask & _mm512_cmplt_epu8_mask(_mm512_sub_epi8(result_zmm, x_b9_zmm), x_05_zmm);

    // Fold: C4 even, C5 81-87 odd, C5 8A-B6 even, C5 B9-BD odd
    __mmask64 fold_latext = (is_after_c4_mask & is_even_mask) | (is_c5_81_87 & is_odd_mask) |
                            (is_c5_8a_b6 & is_even_mask) | (is_c5_b9_bd & is_odd_mask);

    // Apply offsets using Offset Vector pattern
    __m512i offset_zmm = _mm512_setzero_si512();
    offset_zmm = _mm512_mask_mov_epi8(offset_zmm, fold_latin1, x_20_zmm);
    offset_zmm = _mm512_mask_mov_epi8(offset_zmm, fold_latext, x_01_zmm);
    result_zmm = _mm512_add_epi8(result_zmm, offset_zmm);

    sz_assert_(_mm512_cmpeq_epi8_mask(sz_utf8_case_insensitive_find_ice_central_europe_fold_naively_zmm_(text_zmm),
                                      result_zmm) == (__mmask64)-1 &&
               "Efficient Central European fold does not match naive implementation");
    return result_zmm;
}

/**
 *  @brief Naive alarm function for Central Europe danger zone detection.
 *
 *  Detects positions where danger characters occur that require special handling:
 *  - E2 84 AA: 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B) (3 bytes → 1 byte)
 *  - C3 9F: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73) (2 bytes → 2 bytes, 1 rune → 2 runes)
 *  - C4 B0: 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87) (2 bytes → 3 bytes)
 *  - C5 BF: 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73) (2 bytes → 1 byte)
 *  - EF AC 80-86: Ligatures:
 *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
 *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
 *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
 *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
 *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
 *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
 *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
 *
 *  Uses 10 CMPEQ operations (5 lead + 5 second byte checks).
 *
 *  @param[in] text_zmm The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_INTERNAL __mmask64 sz_utf8_case_insensitive_find_ice_central_europe_alarm_naively_zmm_(__m512i text_zmm) {
    // Lead byte constants
    __m512i const x_e2_zmm = _mm512_set1_epi8((char)0xE2); // for Kelvin sign
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3); // for Sharp S
    __m512i const x_c4_zmm = _mm512_set1_epi8((char)0xC4); // for Dotted I
    __m512i const x_c5_zmm = _mm512_set1_epi8((char)0xC5); // for Long S
    __m512i const x_ef_zmm = _mm512_set1_epi8((char)0xEF); // for ligatures

    // Second byte constants
    __m512i const x_84_zmm = _mm512_set1_epi8((char)0x84); // 2nd byte of Kelvin
    __m512i const x_9f_zmm = _mm512_set1_epi8((char)0x9F); // 2nd byte of Sharp S
    __m512i const x_b0_zmm = _mm512_set1_epi8((char)0xB0); // 2nd byte of Dotted I
    __m512i const x_bf_zmm = _mm512_set1_epi8((char)0xBF); // 2nd byte of Long S
    __m512i const x_ac_zmm = _mm512_set1_epi8((char)0xAC); // 2nd byte of ligatures

    // Lead bytes (5 CMPEQ)
    __mmask64 is_e2_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_e2_zmm);
    __mmask64 is_c3_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c3_zmm);
    __mmask64 is_c4_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c4_zmm);
    __mmask64 is_c5_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c5_zmm);
    __mmask64 is_ef_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ef_zmm);

    // Second bytes (5 CMPEQ)
    __mmask64 is_84_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_84_zmm);
    __mmask64 is_9f_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_9f_zmm);
    __mmask64 is_b0_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_b0_zmm);
    __mmask64 is_bf_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_bf_zmm);
    __mmask64 is_ac_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ac_zmm);

    // Danger mask construction
    return ((is_e2_mask << 1) & is_84_mask) | // Kelvin (E2 84 AA)
           ((is_c3_mask << 1) & is_9f_mask) | // Sharp S (C3 9F)
           ((is_c4_mask << 1) & is_b0_mask) | // Dotted I (C4 B0)
           ((is_c5_mask << 1) & is_bf_mask) | // Long S (C5 BF)
           ((is_ef_mask << 1) & is_ac_mask);  // Ligatures (EF AC xx)
}

/**
 *  @brief Optimized alarm function for Central Europe danger zone detection.
 *
 *  Reduces port 5 pressure from 10 to 8 operations by:
 *  - C3/C4/C5 consecutive: 3 CMPEQ -> 1 CMPLT + 2 VPTESTNMB (p0)
 *
 *  Port summary: 8 p5 ops + 2 p0 ops (vs 10 p5 originally)
 *
 *  @param[in] h The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_INTERNAL __mmask64 sz_utf8_case_insensitive_find_ice_central_europe_alarm_efficiently_zmm_(__m512i text_zmm) {
    // Lead byte constants (E2, EF not in consecutive range)
    __m512i const x_e2_zmm = _mm512_set1_epi8((char)0xE2);
    __m512i const x_ef_zmm = _mm512_set1_epi8((char)0xEF);
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3);
    __m512i const x_03_zmm = _mm512_set1_epi8(0x03);
    __m512i const x_02_zmm = _mm512_set1_epi8(0x02);

    // Second byte constants
    __m512i const x_84_zmm = _mm512_set1_epi8((char)0x84);
    __m512i const x_9f_zmm = _mm512_set1_epi8((char)0x9F);
    __m512i const x_b0_zmm = _mm512_set1_epi8((char)0xB0);
    __m512i const x_bf_zmm = _mm512_set1_epi8((char)0xBF);
    __m512i const x_ac_zmm = _mm512_set1_epi8((char)0xAC);

    // Check for C3/C4/C5 range: (byte - 0xC3) < 3  [1 CMPLT on p5]
    __m512i off_c3_zmm = _mm512_sub_epi8(text_zmm, x_c3_zmm);
    __mmask64 in_c3_c5_range_mask = _mm512_cmplt_epu8_mask(off_c3_zmm, x_03_zmm);

    // Derive C3, C4, C5 from range using VPTESTNMB (p0)
    __mmask64 is_c3_mask = in_c3_c5_range_mask & _mm512_testn_epi8_mask(off_c3_zmm, off_c3_zmm); // offset==0
    __m512i off_xor_2_zmm = _mm512_xor_si512(off_c3_zmm, x_02_zmm);
    __mmask64 is_c5_mask = in_c3_c5_range_mask & _mm512_testn_epi8_mask(off_xor_2_zmm, off_xor_2_zmm); // offset==2
    __mmask64 is_c4_mask = in_c3_c5_range_mask & ~is_c3_mask & ~is_c5_mask;                            // offset==1

    // Other lead bytes (2 CMPEQ on p5)
    __mmask64 is_e2_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_e2_zmm);
    __mmask64 is_ef_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ef_zmm);

    // Second bytes (5 CMPEQ on p5)
    __mmask64 is_84_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_84_zmm);
    __mmask64 is_9f_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_9f_zmm);
    __mmask64 is_b0_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_b0_zmm);
    __mmask64 is_bf_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_bf_zmm);
    __mmask64 is_ac_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ac_zmm);

    // Danger mask construction
    __mmask64 danger_mask = ((is_e2_mask << 1) & is_84_mask) | // Kelvin (E2 84 AA)
                            ((is_c3_mask << 1) & is_9f_mask) | // Sharp S (C3 9F)
                            ((is_c4_mask << 1) & is_b0_mask) | // Dotted I (C4 B0)
                            ((is_c5_mask << 1) & is_bf_mask) | // Long S (C5 BF)
                            ((is_ef_mask << 1) & is_ac_mask);  // Ligatures (EF AC xx)

    sz_assert_(danger_mask == sz_utf8_case_insensitive_find_ice_central_europe_alarm_naively_zmm_(text_zmm) &&
               "Efficient Central Europe alarm must match naive implementation");
    return danger_mask;
}

/**
 *  @brief Central European case-insensitive search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_case_rune_safe_central_europe_k
 *
 *  @param[in] haystack Pointer to the haystack string.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] needle Pointer to the full needle string.
 *  @param[in] needle_length Length of the full needle in bytes.
 *  @param[in] needle_metadata Safe window metadata.
 *  @param[out] matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_ice_central_europe_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                       //
    sz_cptr_t needle, sz_size_t needle_length,                           //
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata,   //
    sz_size_t *matched_length) {

    // Validate inputs
    sz_assert_(needle_metadata && "needle_metadata must be provided");
    sz_assert_(needle_metadata->folded_slice_length > 0 && "folded window must be non-empty");

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_assert_(folded_window_length <= 16 && "expect folded needle part to fit in XMM registers");

    // Pre-load folded window into XMM
    __mmask16 const folded_window_mask = sz_u16_mask_until_(folded_window_length);
    sz_u128_vec_t needle_window_vec, haystack_candidate_vec;
    needle_window_vec.xmm = _mm_loadu_si128((__m128i const *)needle_metadata->folded_slice);

    // 4 probe positions
    sz_size_t const offset_second = needle_metadata->probe_second;
    sz_size_t const offset_third = needle_metadata->probe_third;
    sz_size_t const offset_last = folded_window_length - 1;

    sz_u512_vec_t probe_first_vec, probe_second_vec, probe_third_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[0]);
    probe_second_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_second]);
    probe_third_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_third]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_last]);

    // Pre-load the first folded rune for danger zone matching
    sz_rune_t needle_first_safe_folded_rune;
    {
        sz_rune_length_t dummy;
        sz_rune_parse((sz_cptr_t)(needle_metadata->folded_slice), &needle_first_safe_folded_rune, &dummy);
    }

    sz_u512_vec_t haystack_vec;
    sz_cptr_t haystack_ptr = haystack;
    while (haystack_ptr < haystack_end) {
        sz_size_t const available = (sz_size_t)(haystack_end - haystack_ptr);
        if (available < folded_window_length) break;

        sz_size_t const chunk_size = available < 64 ? available : 64;
        sz_size_t const valid_starts = chunk_size - folded_window_length + 1;
        // For danger detection across chunk boundaries, reduce step size to ensure
        // 3-byte patterns at chunk end are fully visible in the next chunk.
        // For tail chunks (valid_starts <= 2), step = 1 ensures progress.
        sz_size_t const step = valid_starts > 2 ? valid_starts - 2 : 1;
        __mmask64 const load_mask = sz_u64_mask_until_(chunk_size);
        __mmask64 const valid_mask = sz_u64_mask_until_(valid_starts);

        haystack_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, haystack_ptr);

        // Check for anomalies using the alarm function
        __mmask64 danger_mask =
            sz_utf8_case_insensitive_find_ice_central_europe_alarm_efficiently_zmm_(haystack_vec.zmm);

        if (danger_mask) {
            // The danger zone handler scans for the needle's first safe rune (at offset_in_unfolded).
            // To find all matches that start at valid positions (0 to valid_starts-1), we need to
            // scan up to position valid_starts - 1 + offset_in_unfolded. Use chunk_size as upper bound.
            sz_size_t danger_scan_length =
                sz_min_of_two(valid_starts + needle_metadata->offset_in_unfolded, chunk_size);
            sz_cptr_t match = sz_utf8_case_insensitive_find_in_danger_zone_( //
                haystack, haystack_length,                                   //
                needle, needle_length,                                       //
                haystack_ptr, danger_scan_length,                            // extended danger zone
                needle_first_safe_folded_rune,                               // pivot point
                needle_metadata->offset_in_unfolded,                         // its location in the needle
                matched_length);
            if (match) return match;
            haystack_ptr += step;
            continue;
        }

        // Fold and Probe
        haystack_vec.zmm = sz_utf8_case_insensitive_find_ice_central_europe_fold_efficiently_zmm_(haystack_vec.zmm);

        sz_u64_t matches = _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_first_vec.zmm);
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_second_vec.zmm) >> offset_second;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_third_vec.zmm) >> offset_third;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_last_vec.zmm) >> offset_last;
        matches &= valid_mask;

        // Candidate Verification
        for (; matches; matches &= matches - 1) {
            sz_size_t candidate_offset = sz_u64_ctz(matches);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;

            haystack_candidate_vec.xmm = _mm_maskz_loadu_epi8(folded_window_mask, haystack_candidate_ptr);
            haystack_candidate_vec.xmm =
                _mm512_castsi512_si128(sz_utf8_case_insensitive_find_ice_central_europe_fold_efficiently_zmm_(
                    _mm512_castsi128_si512(haystack_candidate_vec.xmm)));

            __mmask16 window_mismatch_mask =
                _mm_mask_cmpneq_epi8_mask(folded_window_mask, haystack_candidate_vec.xmm, needle_window_vec.xmm);
            if (window_mismatch_mask) continue;

            sz_cptr_t match = sz_utf8_case_insensitive_verify_match_(                    //
                haystack, haystack_length,                                               //
                needle, needle_length,                                                   //
                haystack_candidate_ptr - haystack, needle_metadata->folded_slice_length, // matched offset & length
                needle_metadata->offset_in_unfolded,                                     // head
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, // tail
                matched_length);
            if (match) {
                sz_utf8_case_insensitive_find_assert_(match, haystack, haystack_length, needle, needle_length,
                                                      needle_metadata);
                return match;
            }
        }
        haystack_ptr += step;
    }

    sz_utf8_case_insensitive_find_assert_(SZ_NULL_CHAR, haystack, haystack_length, needle, needle_length,
                                          needle_metadata);
    return SZ_NULL_CHAR;
}

#pragma endregion // Central European Case-Insensitive Find

#pragma region Cyrillic Case-Insensitive Find

/**
 *  @brief Fold a ZMM register using Cyrillic case-folding rules.
 *  @sa sz_utf8_case_rune_safe_cyrillic_k
 *
 *  Handles Basic Cyrillic (D0/D1) and Extended Cyrillic (D2/D3) ranges.
 *
 *  @param[in] text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_case_insensitive_find_ice_cyrillic_fold_naively_zmm_(__m512i text_zmm) {
    // Start with ASCII folded
    __m512i result_zmm = sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(text_zmm);

    // Constants for Basic Cyrillic Folding (D0/D1 only)
    // Note: Extended Cyrillic (D2/D3) is BANNED - needles with D2/D3 use serial fallback
    __m512i const x_10_zmm = _mm512_set1_epi8((char)0x10); // +16 for Extensions (D0 80-8F → D1 90-9F)
    __m512i const x_20_zmm = _mm512_set1_epi8((char)0x20); // +/-32 for basic block shifts

    // Lead Bytes:
    __m512i const x_d0_zmm = _mm512_set1_epi8((char)0xD0); // Basic Cyrillic upper (U+0400-U+043F)
    __m512i const x_d1_zmm = _mm512_set1_epi8((char)0xD1); // Basic Cyrillic lower (U+0440-U+047F)

    // Range Boundary Constants:
    __m512i const x_80_zmm = _mm512_set1_epi8((char)0x80);    // Base offset (0x80)
    __m512i const x_90_zmm = _mm512_set1_epi8((char)0x90);    // D0: 'А' start
    __m512i const x_a0_zmm = _mm512_set1_epi8((char)0xA0);    // D0: 'Р' start
    __m512i const x_len16_zmm = _mm512_set1_epi8((char)0x10); // 16: D0 block length

    // Basic Cyrillic (D0/D1 lead bytes) - U+0400 to U+047F
    // Three sub-ranges need folding:
    //   - Extensions:  D0 80-8F ('Ѐ'-'Џ') → D1 90-9F ('ѐ'-'џ') : lead D0→D1, second +0x10
    //   - Basic A-Pe:  D0 90-9F ('А'-'П') → D0 B0-BF ('а'-'п') : lead unchanged, second +0x20
    //   - Basic Er-Ya: D0 A0-AF ('Р'-'Я') → D1 80-8F ('р'-'я') : lead D0→D1, second -0x20
    __mmask64 is_lead_d0_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_d0_zmm);
    __mmask64 is_after_d0_mask = is_lead_d0_mask << 1;

    // Detect each uppercase sub-range (second bytes following D0)
    __mmask64 is_ext_range = _mm512_mask_cmplt_epu8_mask( // D0 80-8F: Extensions 'Ѐ'-'Џ'
        is_after_d0_mask, _mm512_sub_epi8(text_zmm, x_80_zmm), x_len16_zmm);
    __mmask64 is_basic1_range = _mm512_mask_cmplt_epu8_mask( // D0 90-9F: Basic 'А'-'П'
        is_after_d0_mask, _mm512_sub_epi8(text_zmm, x_90_zmm), x_len16_zmm);
    __mmask64 is_basic2_range = _mm512_mask_cmplt_epu8_mask( // D0 A0-AF: Basic 'Р'-'Я'
        is_after_d0_mask, _mm512_sub_epi8(text_zmm, x_a0_zmm), x_len16_zmm);

    // Change lead byte D0 → D1 for Extensions and Er-Ya (their lowercase lives in D1)
    __mmask64 change_lead_mask = (is_ext_range >> 1) | (is_basic2_range >> 1);
    result_zmm = _mm512_mask_mov_epi8(result_zmm, change_lead_mask, x_d1_zmm);

    // Apply second-byte transformations
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_ext_range, result_zmm, x_10_zmm);    // +0x10
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_basic1_range, result_zmm, x_20_zmm); // +0x20
    result_zmm = _mm512_mask_sub_epi8(result_zmm, is_basic2_range, result_zmm, x_20_zmm); // -0x20

    return result_zmm;
}

/**
 *  @brief Fold a ZMM register using Cyrillic case-folding rules.
 *  @sa sz_utf8_case_rune_safe_cyrillic_k
 *
 *  This function uses VPSHUFB optimization for Cyrillic case-folding. Why?
 *  Cyrillic has a clean high-nibble pattern: second bytes 8x/9x/Ax map directly
 *  to distinct offsets (+0x10, +0x20, -0x20). A single VPSHUFB lookup replaces
 *  3 range comparisons + 3 masked moves.
 *
 *  @param[in] text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_case_insensitive_find_ice_cyrillic_fold_efficiently_zmm_(__m512i text_zmm) {
    // Start with ASCII folded
    __m512i result_zmm = sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(text_zmm);

    // Constants for Basic Cyrillic Folding (D0/D1 only)
    // Note: Extended Cyrillic (D2/D3) is BANNED - needles with D2/D3 use serial fallback
    __m512i const x_d0_zmm = _mm512_set1_epi8((char)0xD0);
    __m512i const x_d1_zmm = _mm512_set1_epi8((char)0xD1);
    __m512i const x_0f_zmm = _mm512_set1_epi8(0x0F);

    // Detect D0 lead bytes and positions following them
    __mmask64 is_d0_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_d0_zmm);
    __mmask64 is_after_d0_mask = is_d0_mask << 1;

    // VPSHUFB Lookup Table: High nibble of second byte -> offset
    // ---------------------------------------------------------------
    // Second bytes after D0:
    //   80-8F (high nibble 8): Extensions 'Ѐ'-'Џ' -> +0x10
    //   90-9F (high nibble 9): Basic 'А'-'П'      -> +0x20
    //   A0-AF (high nibble A): Basic 'Р'-'Я'      -> -0x20 (0xE0)
    //   B0-BF (high nibble B): lowercase, no change -> 0x00
    //
    // LUT layout: index by high nibble (0-F), repeated across 4 lanes
    __m512i const offset_lut = _mm512_set_epi8(
        // clang-format off
        // Lane 3 (bytes 48-63): indices F E D C B A 9 8 7 6 5 4 3 2 1 0
        0, 0, 0, 0, 0, (char)0xE0, (char)0x20, (char)0x10, 0, 0, 0, 0, 0, 0, 0, 0,
        // Lane 2 (bytes 32-47)
        0, 0, 0, 0, 0, (char)0xE0, (char)0x20, (char)0x10, 0, 0, 0, 0, 0, 0, 0, 0,
        // Lane 1 (bytes 16-31)
        0, 0, 0, 0, 0, (char)0xE0, (char)0x20, (char)0x10, 0, 0, 0, 0, 0, 0, 0, 0,
        // Lane 0 (bytes 0-15)
        0, 0, 0, 0, 0, (char)0xE0, (char)0x20, (char)0x10, 0, 0, 0, 0, 0, 0, 0, 0
        // clang-format on
    );

    // Extract high nibble of each byte: (byte >> 4) & 0x0F
    __m512i high_nibbles = _mm512_and_si512(_mm512_srli_epi16(text_zmm, 4), x_0f_zmm);

    // Single VPSHUFB lookup: high_nibble -> offset value
    __m512i offsets = _mm512_shuffle_epi8(offset_lut, high_nibbles);

    // Zero out offsets for non-continuation positions (not after D0)
    offsets = _mm512_maskz_mov_epi8(is_after_d0_mask, offsets);

    // Apply offsets to continuation bytes
    result_zmm = _mm512_add_epi8(result_zmm, offsets);

    // Lead byte changes: D0 -> D1 for high nibble 8 (ext) or A (basic2)
    // These ranges have lowercase in the D1 block
    __m512i const x_08_zmm = _mm512_set1_epi8(0x08);
    __m512i const x_0a_zmm = _mm512_set1_epi8(0x0A);
    __mmask64 is_8x = _mm512_mask_cmpeq_epi8_mask(is_after_d0_mask, high_nibbles, x_08_zmm);
    __mmask64 is_ax = _mm512_mask_cmpeq_epi8_mask(is_after_d0_mask, high_nibbles, x_0a_zmm);
    __mmask64 change_lead_mask = ((is_8x | is_ax) >> 1) & is_d0_mask;
    result_zmm = _mm512_mask_mov_epi8(result_zmm, change_lead_mask, x_d1_zmm);

    sz_assert_(_mm512_cmpeq_epi8_mask(sz_utf8_case_insensitive_find_ice_cyrillic_fold_naively_zmm_(text_zmm),
                                      result_zmm) == (__mmask64)-1 &&
               "Efficient Cyrillic fold does not match naive implementation");
    return result_zmm;
}

/**
 *  @brief Cyrillic case-insensitive search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_case_rune_safe_cyrillic_k
 *
 *  @param[in] haystack Pointer to the haystack string.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] needle Pointer to the full needle string.
 *  @param[in] needle_length Length of the full needle in bytes.
 *  @param[in] needle_metadata Safe window metadata.
 *  @param[out] matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_ice_cyrillic_(     //
    sz_cptr_t haystack, sz_size_t haystack_length,                     //
    sz_cptr_t needle, sz_size_t needle_length,                         //
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata, //
    sz_size_t *matched_length) {

    // Validate inputs
    sz_assert_(needle_metadata && "needle_metadata must be provided");
    sz_assert_(needle_metadata->folded_slice_length > 0 && "folded window must be non-empty");

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_assert_(folded_window_length <= 16 && "expect folded needle part to fit in XMM registers");

    // Pre-load folded window into XMM
    __mmask16 const folded_window_mask = sz_u16_mask_until_(folded_window_length);
    sz_u128_vec_t needle_window_vec, haystack_candidate_vec;
    needle_window_vec.xmm = _mm_loadu_si128((__m128i const *)needle_metadata->folded_slice);

    // 4 probe positions
    sz_size_t const offset_second = needle_metadata->probe_second;
    sz_size_t const offset_third = needle_metadata->probe_third;
    sz_size_t const offset_last = folded_window_length - 1;

    sz_u512_vec_t probe_first_vec, probe_second_vec, probe_third_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[0]);
    probe_second_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_second]);
    probe_third_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_third]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_last]);

    // Unified loop - handles both full 64-byte chunks and the tail with a single code path
    sz_u512_vec_t haystack_vec;
    sz_cptr_t haystack_ptr = haystack;
    while (haystack_ptr < haystack_end) {
        sz_size_t const available = (sz_size_t)(haystack_end - haystack_ptr);
        if (available < folded_window_length) break;

        sz_size_t const chunk_size = available < 64 ? available : 64;
        sz_size_t const valid_starts = chunk_size - folded_window_length + 1;
        __mmask64 const load_mask = sz_u64_mask_until_(chunk_size);
        __mmask64 const valid_mask = sz_u64_mask_until_(valid_starts);

        haystack_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, haystack_ptr);

        // Check for anomalies (characters that fold to different byte widths)
        // No checks needed for Cyrillic:
        // - 'ẞ' (E1 BA 9E), 'K' (E2 84 AA), 'ﬁ' (EF AC xx), 'ſ' (C5 BF):
        //   Excluded by sz_utf8_case_rune_safety_profile_t_ for cyrillic_k.
        // - 'Ґ' (D2 90): Not supported by this kernel, so needles with it use serial fallback.
        //   Presence in haystack doesn't matter as it won't match our needle's D2 bytes.

        // Fold and 4-way probe filter
        haystack_vec.zmm = sz_utf8_case_insensitive_find_ice_cyrillic_fold_efficiently_zmm_(haystack_vec.zmm);

        sz_u64_t matches = _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_first_vec.zmm);
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_second_vec.zmm) >> offset_second;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_third_vec.zmm) >> offset_third;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_last_vec.zmm) >> offset_last;
        matches &= valid_mask;

        for (; matches; matches &= matches - 1) {
            sz_size_t candidate_offset = sz_u64_ctz(matches);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;

            haystack_candidate_vec.xmm = _mm_maskz_loadu_epi8(folded_window_mask, haystack_candidate_ptr);
            haystack_candidate_vec.xmm =
                _mm512_castsi512_si128(sz_utf8_case_insensitive_find_ice_cyrillic_fold_efficiently_zmm_(
                    _mm512_castsi128_si512(haystack_candidate_vec.xmm)));

            __mmask16 window_mismatch_mask =
                _mm_mask_cmpneq_epi8_mask(folded_window_mask, haystack_candidate_vec.xmm, needle_window_vec.xmm);
            if (window_mismatch_mask) continue;

            sz_cptr_t match = sz_utf8_case_insensitive_verify_match_(                    //
                haystack, haystack_length,                                               //
                needle, needle_length,                                                   //
                haystack_candidate_ptr - haystack, needle_metadata->folded_slice_length, // matched offset & length
                needle_metadata->offset_in_unfolded,                                     // head
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, // tail
                matched_length);
            if (match) {
                sz_utf8_case_insensitive_find_assert_(match, haystack, haystack_length, needle, needle_length,
                                                      needle_metadata);
                return match;
            }
        }
        haystack_ptr += valid_starts;
    }

    sz_utf8_case_insensitive_find_assert_(SZ_NULL_CHAR, haystack, haystack_length, needle, needle_length,
                                          needle_metadata);
    return SZ_NULL_CHAR;
}

#pragma endregion // Cyrillic Case-Insensitive Find

#pragma region Armenian Case-Insensitive Find

/**
 *  @brief Fold a ZMM register using Armenian case-folding rules.
 *  @sa sz_utf8_case_rune_safe_armenian_k
 *
 *  @param[in] text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_case_insensitive_find_ice_armenian_fold_naively_zmm_(__m512i text_zmm) {
    // Start with ASCII folded
    __m512i result_zmm = sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(text_zmm);

    // Constants for Armenian folding
    // ------------------------------
    // Lead bytes:
    __m512i const x_d4_zmm = _mm512_set1_epi8((char)0xD4);
    __m512i const x_d5_zmm = _mm512_set1_epi8((char)0xD5);
    __m512i const x_d6_zmm = _mm512_set1_epi8((char)0xD6);

    // Range offsets:
    __m512i const x_b1_zmm = _mm512_set1_epi8((char)0xB1);
    __m512i const x_96_zmm = _mm512_set1_epi8((char)0x96);

    // Transformation deltas:
    __m512i const x_30_zmm = _mm512_set1_epi8((char)0x30); // 48
    __m512i const x_10_zmm = _mm512_set1_epi8((char)0x10); // 16

    // Identify Lead Bytes
    __mmask64 is_d4_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_d4_zmm);
    __mmask64 is_d5_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_d5_zmm);
    __mmask64 is_after_d4_mask = is_d4_mask << 1;
    __mmask64 is_after_d5_mask = is_d5_mask << 1;

    // 1. D4 ranges (Uppercase D4 B1-BF -> Lowercase D5 A1-AF)
    // --------------------------------------------------------
    // 'Ա' (D4 B1) ... 'Ձ' (D4 BF) -> 'ա' (D5 A1) ... 'ձ' (D5 AF)
    // Lead D4 -> D5, Second B1-BF -> A1-AF (-0x10)
    __mmask64 is_d4_upper = _mm512_mask_cmpge_epu8_mask(is_after_d4_mask, result_zmm, x_b1_zmm);
    // Upper bound implied by 0xFF (mask covers all B1-BF if we just check >= B1)

    // Apply D4 transformations
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_d4_upper >> 1, x_d5_zmm);        // Lead D4 -> D5
    result_zmm = _mm512_mask_sub_epi8(result_zmm, is_d4_upper, result_zmm, x_10_zmm); // Second -0x10

    // 2. D5 ranges (Uppercase D5 80-96 -> Lowercase D5 B0-BF / D6 80-86)
    // ------------------------------------------------------------------
    // Group 2a: D5 80-8F ('Ղ'-'Տ') -> D5 B0-BF ('ղ'-'տ')
    // Offset +0x30
    __mmask64 is_d5_upper_subset1 =
        _mm512_mask_cmple_epu8_mask(is_after_d5_mask, result_zmm, _mm512_set1_epi8((char)0x8F));
    is_d5_upper_subset1 &= _mm512_mask_cmpge_epu8_mask(is_after_d5_mask, result_zmm, _mm512_set1_epi8((char)0x80));

    result_zmm = _mm512_mask_add_epi8(result_zmm, is_d5_upper_subset1, result_zmm, x_30_zmm); // +0x30

    // Group 2b: D5 90-96 ('Ր'-'Ֆ') -> D6 80-86 ('ր'-'ֆ')
    // Lead D5 -> D6, Second 90-96 -> 80-86 (-0x10)
    __mmask64 is_d5_upper_subset2 =
        _mm512_mask_cmpge_epu8_mask(is_after_d5_mask, result_zmm, _mm512_set1_epi8((char)0x90));
    is_d5_upper_subset2 &= _mm512_mask_cmple_epu8_mask(is_after_d5_mask, result_zmm, x_96_zmm);

    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_d5_upper_subset2 >> 1, x_d6_zmm);        // Lead D5 -> D6
    result_zmm = _mm512_mask_sub_epi8(result_zmm, is_d5_upper_subset2, result_zmm, x_10_zmm); // Second -0x10

    return result_zmm;
}

/**
 *  @brief Fold a ZMM register using Armenian case-folding rules.
 *  @sa sz_utf8_case_rune_safe_armenian_k
 *
 *  This function uses VPTERNLOG optimization for Armenian case-folding. Why?
 *  Armenian has 3 distinct lead bytes (D4/D5/D6) with different sub-ranges.
 *  VPTERNLOG allows building 3 offset vectors in parallel (no dependencies),
 *  then combining them with a single OR operation (imm8=0xFE: A | B | C).
 *
 *  @param[in] text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_case_insensitive_find_ice_armenian_fold_efficiently_zmm_(__m512i text_zmm) {
    // Inline ASCII fold (avoiding function call overhead)
    __m512i const a_upper_zmm = _mm512_set1_epi8('A');
    __m512i const range26_zmm = _mm512_set1_epi8(26);
    __m512i const x_20_zmm = _mm512_set1_epi8(0x20);
    __mmask64 is_upper_mask = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_zmm, a_upper_zmm), range26_zmm);
    __m512i result_zmm = _mm512_mask_add_epi8(text_zmm, is_upper_mask, text_zmm, x_20_zmm);

    // Lead bytes
    __m512i const x_d4_zmm = _mm512_set1_epi8((char)0xD4);
    __m512i const x_d5_zmm = _mm512_set1_epi8((char)0xD5);
    __m512i const x_d6_zmm = _mm512_set1_epi8((char)0xD6);

    // Range compression constants
    __m512i const x_b1_zmm = _mm512_set1_epi8((char)0xB1);
    __m512i const x_80_zmm = _mm512_set1_epi8((char)0x80);
    __m512i const x_90_zmm = _mm512_set1_epi8((char)0x90);
    __m512i const x_0f_zmm = _mm512_set1_epi8((char)0x0F); // 15: D4 B1-BF range
    __m512i const x_10_zmm = _mm512_set1_epi8((char)0x10); // 16: D5 80-8F range
    __m512i const x_07_zmm = _mm512_set1_epi8((char)0x07); // 7: D5 90-96 range
    __m512i const x_30_zmm = _mm512_set1_epi8((char)0x30); // +0x30 offset
    __m512i const x_f0_zmm = _mm512_set1_epi8((char)0xF0); // -0x10 offset

    // Lead byte detection (all parallel, no dependencies)
    __mmask64 is_d4_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_d4_zmm);
    __mmask64 is_d5_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_d5_zmm);
    __mmask64 is_after_d4_mask = is_d4_mask << 1;
    __mmask64 is_after_d5_mask = is_d5_mask << 1;

    // Range compression: All 3 ranges computed in parallel
    // D4 B1-BF: (byte - B1) < 15
    __mmask64 is_d4_upper =
        _mm512_mask_cmplt_epu8_mask(is_after_d4_mask, _mm512_sub_epi8(result_zmm, x_b1_zmm), x_0f_zmm);
    // D5 80-8F: (byte - 80) < 16
    __mmask64 is_d5_subset1 =
        _mm512_mask_cmplt_epu8_mask(is_after_d5_mask, _mm512_sub_epi8(result_zmm, x_80_zmm), x_10_zmm);
    // D5 90-96: (byte - 90) < 7
    __mmask64 is_d5_subset2 =
        _mm512_mask_cmplt_epu8_mask(is_after_d5_mask, _mm512_sub_epi8(result_zmm, x_90_zmm), x_07_zmm);

    // Lead byte changes (applied before offset to preserve original lead byte positions)
    // D4 B1-BF: lead D4 -> D5
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_d4_upper >> 1, x_d5_zmm);
    // D5 90-96: lead D5 -> D6
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_d5_subset2 >> 1, x_d6_zmm);

    // Build 3 offset vectors in parallel, combine with OR
    // Each maskz_mov is independent (no data dependencies between them)
    __m512i off_d4_zmm = _mm512_maskz_mov_epi8(is_d4_upper, x_f0_zmm);      // -0x10 for D4 B1-BF
    __m512i off_d5_s1_zmm = _mm512_maskz_mov_epi8(is_d5_subset1, x_30_zmm); // +0x30 for D5 80-8F
    __m512i off_d5_s2_zmm = _mm512_maskz_mov_epi8(is_d5_subset2, x_f0_zmm); // -0x10 for D5 90-96

    // VPTERNLOG: A | B | C (imm8 = 0xFE)
    __m512i offset_zmm = _mm512_ternarylogic_epi64(off_d4_zmm, off_d5_s1_zmm, off_d5_s2_zmm, 0xFE);

    // Single add applies all offsets simultaneously
    result_zmm = _mm512_add_epi8(result_zmm, offset_zmm);

    sz_assert_(_mm512_cmpeq_epi8_mask(sz_utf8_case_insensitive_find_ice_armenian_fold_naively_zmm_(text_zmm),
                                      result_zmm) == (__mmask64)-1 &&
               "Efficient Armenian fold must match naive implementation");
    return result_zmm;
}

/**
 *  @brief Naive alarm function for Armenian danger zone detection.
 *
 *  Detects positions where danger characters occur that require special handling:
 *  - D6 87: 'և' (U+0587, D6 87) → "եւ" (U+0565 U+0582, D5 A5 D6 82) (2 bytes → 4 bytes)
 *  - EF AC 93-97: Armenian ligatures 'ﬓ'..'ﬗ' (U+FB13-U+FB17, EF AC 93-97) → 2 codepoints (4 bytes)
 *
 *  Uses 4 CMPEQ operations (2 lead + 2 second byte checks).
 *
 *  @param[in] h The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_INTERNAL __mmask64 sz_utf8_case_insensitive_find_ice_armenian_alarm_naively_zmm_(__m512i text_zmm) {
    // Lead byte constants
    __m512i const x_d6_zmm = _mm512_set1_epi8((char)0xD6); // for Ech-Yiwn
    __m512i const x_ef_zmm = _mm512_set1_epi8((char)0xEF); // for ligatures

    // Second byte constants
    __m512i const x_87_zmm = _mm512_set1_epi8((char)0x87); // 2nd byte of Ech-Yiwn
    __m512i const x_ac_zmm = _mm512_set1_epi8((char)0xAC); // 2nd byte of ligatures

    // Lead bytes (2 CMPEQ)
    __mmask64 is_d6_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_d6_zmm);
    __mmask64 is_ef_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ef_zmm);

    // Second bytes (2 CMPEQ)
    __mmask64 is_87_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_87_zmm);
    __mmask64 is_ac_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ac_zmm);

    // Danger mask construction
    __mmask64 ech_yiwn_mask = (is_d6_mask << 1) & is_87_mask; // Ech-Yiwn (D6 87)
    __mmask64 ef_ac_mask = (is_ef_mask << 1) & is_ac_mask;    // Ligatures (EF AC)
    return ech_yiwn_mask | ef_ac_mask;
}

/**
 *  @brief Optimized alarm function for Armenian danger zone detection.
 *
 *  Currently delegates to the naive implementation.
 *  Armenian has only 4 CMPEQ ops, so optimization overhead may not be worthwhile.
 *
 *  @param[in] h The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_INTERNAL __mmask64 sz_utf8_case_insensitive_find_ice_armenian_alarm_efficiently_zmm_(__m512i text_zmm) {
    // Only 4 CMPEQ operations - optimization overhead likely not worth it
    return sz_utf8_case_insensitive_find_ice_armenian_alarm_naively_zmm_(text_zmm);
}

/**
 *  @brief Armenian case-insensitive search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_case_rune_safe_armenian_k
 *
 *  @param[in] haystack Pointer to the haystack string.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] needle Pointer to the full needle string.
 *  @param[in] needle_length Length of the full needle in bytes.
 *  @param[in] needle_metadata Pre-folded window content with probe positions.
 *  @param[out] matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_ice_armenian_(     //
    sz_cptr_t haystack, sz_size_t haystack_length,                     //
    sz_cptr_t needle, sz_size_t needle_length,                         //
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata, //
    sz_size_t *matched_length) {

    // Validate inputs
    sz_assert_(needle_metadata && "needle_metadata must be provided");
    sz_assert_(needle_metadata->folded_slice_length > 0 && "folded window must be non-empty");

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_assert_(folded_window_length <= 16 && "expect folded needle part to fit in XMM registers");

    // Pre-load folded window into XMM
    __mmask16 const folded_window_mask = sz_u16_mask_until_(folded_window_length);
    sz_u128_vec_t needle_window_vec, haystack_candidate_vec;
    needle_window_vec.xmm = _mm_loadu_si128((__m128i const *)needle_metadata->folded_slice);

    // 4 probe positions
    sz_size_t const offset_second = needle_metadata->probe_second;
    sz_size_t const offset_third = needle_metadata->probe_third;
    sz_size_t const offset_last = folded_window_length - 1;

    sz_u512_vec_t probe_first_vec, probe_second_vec, probe_third_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[0]);
    probe_second_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_second]);
    probe_third_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_third]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_last]);

    // Pre-load the first folded rune for danger zone matching
    sz_rune_t needle_first_safe_folded_rune;
    {
        sz_rune_length_t dummy;
        sz_rune_parse((sz_cptr_t)(needle_metadata->folded_slice), &needle_first_safe_folded_rune, &dummy);
    }

    sz_u512_vec_t haystack_vec;
    sz_cptr_t haystack_ptr = haystack;
    while (haystack_ptr < haystack_end) {
        sz_size_t const available = (sz_size_t)(haystack_end - haystack_ptr);
        if (available < folded_window_length) break;

        sz_size_t const chunk_size = available < 64 ? available : 64;
        sz_size_t const valid_starts = chunk_size - folded_window_length + 1;
        // For danger detection across chunk boundaries, reduce step size to ensure
        // 3-byte patterns at chunk end are fully visible in the next chunk.
        // For tail chunks (valid_starts <= 2), step = 1 ensures progress.
        sz_size_t const step = valid_starts > 2 ? valid_starts - 2 : 1;
        __mmask64 const load_mask = sz_u64_mask_until_(chunk_size);
        __mmask64 const valid_mask = sz_u64_mask_until_(valid_starts);

        haystack_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, haystack_ptr);

        // Check for anomalies using the alarm function
        __mmask64 danger_mask = sz_utf8_case_insensitive_find_ice_armenian_alarm_efficiently_zmm_(haystack_vec.zmm);

        if (danger_mask) {
            // The danger zone handler scans for the needle's first safe rune (at offset_in_unfolded).
            // To find all matches that start at valid positions (0 to valid_starts-1), we need to
            // scan up to position valid_starts - 1 + offset_in_unfolded. Use chunk_size as upper bound.
            sz_size_t danger_scan_length =
                sz_min_of_two(valid_starts + needle_metadata->offset_in_unfolded, chunk_size);
            sz_cptr_t match = sz_utf8_case_insensitive_find_in_danger_zone_( //
                haystack, haystack_length,                                   //
                needle, needle_length,                                       //
                haystack_ptr, danger_scan_length,                            // extended danger zone
                needle_first_safe_folded_rune,                               // pivot point
                needle_metadata->offset_in_unfolded,                         // its location in the needle
                matched_length);
            if (match) return match;
            haystack_ptr += step;
            continue;
        }

        // Fold and Probe
        haystack_vec.zmm = sz_utf8_case_insensitive_find_ice_armenian_fold_efficiently_zmm_(haystack_vec.zmm);

        sz_u64_t matches = _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_first_vec.zmm);
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_second_vec.zmm) >> offset_second;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_third_vec.zmm) >> offset_third;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_last_vec.zmm) >> offset_last;
        matches &= valid_mask;

        // Candidate Verification
        for (; matches; matches &= matches - 1) {
            sz_size_t candidate_offset = sz_u64_ctz(matches);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;

            haystack_candidate_vec.xmm = _mm_maskz_loadu_epi8(folded_window_mask, haystack_candidate_ptr);
            haystack_candidate_vec.xmm =
                _mm512_castsi512_si128(sz_utf8_case_insensitive_find_ice_armenian_fold_efficiently_zmm_(
                    _mm512_castsi128_si512(haystack_candidate_vec.xmm)));

            __mmask16 window_mismatch_mask =
                _mm_mask_cmpneq_epi8_mask(folded_window_mask, haystack_candidate_vec.xmm, needle_window_vec.xmm);
            if (window_mismatch_mask) continue;

            sz_cptr_t match = sz_utf8_case_insensitive_verify_match_(                    //
                haystack, haystack_length,                                               //
                needle, needle_length,                                                   //
                haystack_candidate_ptr - haystack, needle_metadata->folded_slice_length, // matched offset & length
                needle_metadata->offset_in_unfolded,                                     // head
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, // tail
                matched_length);
            if (match) {
                sz_utf8_case_insensitive_find_assert_(match, haystack, haystack_length, needle, needle_length,
                                                      needle_metadata);
                return match;
            }
        }
        haystack_ptr += step;
    }

    sz_utf8_case_insensitive_find_assert_(SZ_NULL_CHAR, haystack, haystack_length, needle, needle_length,
                                          needle_metadata);
    return SZ_NULL_CHAR;
}

#pragma endregion // Armenian Case-Insensitive Find

#pragma region Greek Case-Insensitive Find

/**
 *  @brief Fold a ZMM register using Greek case-folding rules.
 *  @sa sz_utf8_case_rune_safe_greek_k
 *
 *  @param[in] text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_case_insensitive_find_ice_greek_fold_naively_zmm_(__m512i text_zmm) {
    // Start with ASCII folded
    __m512i result_zmm = sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(text_zmm);

    // Constants for Greek folding
    // ---------------------------
    // Lead bytes:
    __m512i const x_ce_zmm = _mm512_set1_epi8((char)0xCE);
    __m512i const x_cf_zmm = _mm512_set1_epi8((char)0xCF);
    // 'µ' (U+00B5, C2 B5) → 'μ' (U+03BC, CE BC) in-place.
    __m512i const x_c2_zmm = _mm512_set1_epi8((char)0xC2);
    __m512i const x_b5_zmm = _mm512_set1_epi8((char)0xB5);
    __m512i const x_bc_zmm = _mm512_set1_epi8((char)0xBC);

    // Ranges for CE lead byte:
    // CE 86-8F (Accented Upper): 'Ά' (86) ... 'Ώ' (8F)
    // CE 91-9F (Basic Upper 1):  'Α' (91) ... 'Ο' (9F)
    // CE A0-A9 (Basic Upper 2):  'Π' (A0) ... 'Ω' (A9)
    // CE AA-AB (Dialytika Upper): 'Ϊ' (AA) ... 'Ϋ' (AB)
    __m512i const x_86_zmm = _mm512_set1_epi8((char)0x86);
    __m512i const x_8f_zmm = _mm512_set1_epi8((char)0x8F);
    __m512i const x_91_zmm = _mm512_set1_epi8((char)0x91);
    __m512i const x_9f_zmm = _mm512_set1_epi8((char)0x9F);
    __m512i const x_a0_zmm = _mm512_set1_epi8((char)0xA0);
    __m512i const x_a9_zmm = _mm512_set1_epi8((char)0xA9);
    __m512i const x_aa_zmm = _mm512_set1_epi8((char)0xAA);
    __m512i const x_ab_zmm = _mm512_set1_epi8((char)0xAB);

    // Specific characters:
    // 'ς' (U+03C2, CF 82) → 'σ' (U+03C3, CF 83)
    __m512i const x_82_zmm = _mm512_set1_epi8((char)0x82);
    __m512i const x_83_zmm = _mm512_set1_epi8((char)0x83);

    // Offsets:
    __m512i const x_20_zmm = _mm512_set1_epi8((char)0x20); // +32 (0x20)
    __m512i const x_e0_zmm = _mm512_set1_epi8((char)0xE0); // -32 (0xE0)
    __m512i const x_01_zmm = _mm512_set1_epi8((char)0x01); // +1 (0x01)

    // Identify Lead Bytes
    __mmask64 is_ce_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_ce_zmm);
    __mmask64 is_cf_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_cf_zmm);
    __mmask64 is_after_ce_mask = is_ce_mask << 1;
    __mmask64 is_after_cf_mask = is_cf_mask << 1;

    // 1. CE ranges (Uppercase -> Lowercase)
    // -------------------------------------
    // Basic Greek Upper (Range 1): CE 91-9F ('Α'-'Ο') -> CE B1-BF ('α'-'ο') (Add 0x20)
    __mmask64 is_basic1 = _mm512_mask_cmpge_epu8_mask(is_after_ce_mask, result_zmm, x_91_zmm);
    is_basic1 &= _mm512_mask_cmple_epu8_mask(is_after_ce_mask, result_zmm, x_9f_zmm);

    // Basic Greek Upper (Range 2): CE A0-A9 ('Π'-'Ω') -> CF 80-89 ('π'-'ω') (Change lead CE->CF, subtract 0x20 from
    // 2nd)
    __mmask64 is_basic2 = _mm512_mask_cmpge_epu8_mask(is_after_ce_mask, result_zmm, x_a0_zmm);
    is_basic2 &= _mm512_mask_cmple_epu8_mask(is_after_ce_mask, result_zmm, x_a9_zmm);

    // Accented Greek Upper: CE 86-8F ('Ά'-'Ώ') -> Lowercase
    // Most map nicely: CE 8x -> CE Ax (+20) or CE 8x -> CF 8x (Lead change, 2nd same)
    __mmask64 is_accented = _mm512_mask_cmpge_epu8_mask(is_after_ce_mask, result_zmm, x_86_zmm);
    is_accented &= _mm512_mask_cmple_epu8_mask(is_after_ce_mask, result_zmm, x_8f_zmm);

    // Sub-masks for specific transformations within the accented block
    __mmask64 is_86 = is_accented & _mm512_cmpeq_epi8_mask(result_zmm, x_86_zmm);

    __mmask64 is_88_8a = is_accented & _mm512_cmpge_epu8_mask(result_zmm, _mm512_set1_epi8((char)0x88));
    is_88_8a &= _mm512_cmple_epu8_mask(result_zmm, _mm512_set1_epi8((char)0x8A));

    __mmask64 is_8c = is_accented & _mm512_cmpeq_epi8_mask(result_zmm, _mm512_set1_epi8((char)0x8C));

    __mmask64 is_8e_8f = is_accented & _mm512_cmpge_epu8_mask(result_zmm, _mm512_set1_epi8((char)0x8E));

    // Dialytika Greek Upper: CE AA-AB ('Ϊ', 'Ϋ') -> CF 8A-8B (Lead CE->CF, 2nd -0x20)
    __mmask64 is_dialytika = _mm512_mask_cmpge_epu8_mask(is_after_ce_mask, result_zmm, x_aa_zmm);
    is_dialytika &= _mm512_mask_cmple_epu8_mask(is_after_ce_mask, result_zmm, x_ab_zmm);

    // 2. CF ranges (Final Sigma)
    // --------------------------
    // 'ς' (U+03C2, CF 82) → 'σ' (U+03C3, CF 83)
    __mmask64 is_final_sigma = _mm512_mask_cmpeq_epi8_mask(is_after_cf_mask, result_zmm, x_82_zmm);

    // Apply transformations using masked operations
    // ---------------------------------------------
    // Apply Basic Greek Upper (Range 1)
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_basic1, result_zmm, x_20_zmm);

    // Apply Basic Greek Upper (Range 2)
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_basic2 >> 1, x_cf_zmm);        // Lead CE -> CF
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_basic2, result_zmm, x_e0_zmm); // 2nd -0x20 (using add E0)

    // Apply Accented Greek Upper
    // 1. Additions for Same-Block Mappings (CE -> CE)
    // 'Ά' -> +0x26
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_86, result_zmm, _mm512_set1_epi8(0x26));
    // 'Έ', 'Ή', 'Ί' -> +0x25
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_88_8a, result_zmm, _mm512_set1_epi8(0x25));

    // 2. Lead Byte Changes (CE -> CF)
    // 'Ό' (8C), 'Ύ' (8E), 'Ώ' (8F)
    __mmask64 change_lead = (is_8c >> 1) | (is_8e_8f >> 1);
    result_zmm = _mm512_mask_mov_epi8(result_zmm, change_lead, x_cf_zmm);

    // 3. Second Byte Changes for CF targets
    // 'Ύ', 'Ώ' -> -1
    result_zmm = _mm512_mask_sub_epi8(result_zmm, is_8e_8f, result_zmm, x_01_zmm);

    // Apply Dialytika Greek Upper
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_dialytika >> 1, x_cf_zmm);
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_dialytika, result_zmm, x_e0_zmm); // -0x20

    // Apply Final Sigma
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_final_sigma, x_83_zmm);

    // Apply Micro Sign folding: 'µ' (U+00B5, C2 B5) → 'μ' (U+03BC, CE BC)
    __mmask64 is_c2_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c2_zmm);
    __mmask64 is_micro_second = (is_c2_mask << 1) & _mm512_cmpeq_epi8_mask(result_zmm, x_b5_zmm);
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_micro_second >> 1, x_ce_zmm); // Lead C2 -> CE
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_micro_second, x_bc_zmm);      // Second B5 -> BC

    return result_zmm;
}

/**
 *  @brief Fold a ZMM register using Greek case-folding rules.
 *  @sa sz_utf8_case_rune_safe_greek_k
 *
 *  This function uses VPTERNLOG optimization for Greek case-folding. Why?
 *  Greek has 5+ distinct offset values (+0x20, -0x20, +0x26, +0x25, -1) applied
 *  to different ranges. Previous code built `offset_zmm` sequentially with 6
 *  `mask_mov` operations on the same CPU port, creating a 6-deep dependency chain.
 *  VPTERNLOG allows grouping offsets into parallel vectors and combining with OR.
 *
 *  @param[in] text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_case_insensitive_find_ice_greek_fold_efficiently_zmm_(__m512i text_zmm) {
    // Inline ASCII fold (avoiding function call overhead)
    __m512i const a_upper_zmm = _mm512_set1_epi8('A');
    __m512i const range26_zmm = _mm512_set1_epi8(26);
    __m512i const x_20_zmm = _mm512_set1_epi8(0x20);
    __mmask64 is_upper_mask = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_zmm, a_upper_zmm), range26_zmm);
    __m512i result_zmm = _mm512_mask_add_epi8(text_zmm, is_upper_mask, text_zmm, x_20_zmm);

    // Constants for Greek folding
    __m512i const x_ce_zmm = _mm512_set1_epi8((char)0xCE);
    __m512i const x_cf_zmm = _mm512_set1_epi8((char)0xCF);
    __m512i const x_c2_zmm = _mm512_set1_epi8((char)0xC2);

    // Range compression bases and sizes
    __m512i const x_86_zmm = _mm512_set1_epi8((char)0x86);
    __m512i const x_91_zmm = _mm512_set1_epi8((char)0x91);
    __m512i const x_a0_zmm = _mm512_set1_epi8((char)0xA0);
    __m512i const x_aa_zmm = _mm512_set1_epi8((char)0xAA);
    __m512i const x_88_zmm = _mm512_set1_epi8((char)0x88);
    __m512i const x_8e_zmm = _mm512_set1_epi8((char)0x8E);
    __m512i const x_0a_zmm = _mm512_set1_epi8((char)0x0A);
    __m512i const x_0f_zmm = _mm512_set1_epi8((char)0x0F);
    __m512i const x_02_zmm = _mm512_set1_epi8((char)0x02);
    __m512i const x_03_zmm = _mm512_set1_epi8((char)0x03);

    // Lead byte detection
    __mmask64 is_ce_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_ce_zmm);
    __mmask64 is_cf_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_cf_zmm);
    __mmask64 is_after_ce_mask = is_ce_mask << 1;
    __mmask64 is_after_cf_mask = is_cf_mask << 1;

    // Range compression: All ranges computed in parallel (no dependencies)
    __m512i sub_86_zmm = _mm512_sub_epi8(result_zmm, x_86_zmm);
    __m512i sub_91_zmm = _mm512_sub_epi8(result_zmm, x_91_zmm);
    __m512i sub_a0_zmm = _mm512_sub_epi8(result_zmm, x_a0_zmm);
    __m512i sub_aa_zmm = _mm512_sub_epi8(result_zmm, x_aa_zmm);

    __mmask64 is_accented_mask = _mm512_mask_cmplt_epu8_mask(is_after_ce_mask, sub_86_zmm, x_0a_zmm);
    __mmask64 is_basic1_mask = _mm512_mask_cmplt_epu8_mask(is_after_ce_mask, sub_91_zmm, x_0f_zmm);
    __mmask64 is_basic2_mask = _mm512_mask_cmplt_epu8_mask(is_after_ce_mask, sub_a0_zmm, x_0a_zmm);
    __mmask64 is_dialytika_mask = _mm512_mask_cmplt_epu8_mask(is_after_ce_mask, sub_aa_zmm, x_02_zmm);

    // Accented sub-masks
    __mmask64 is_86_mask = is_accented_mask & _mm512_cmpeq_epi8_mask(result_zmm, x_86_zmm);
    __mmask64 is_88_8a_mask =
        is_accented_mask & _mm512_cmplt_epu8_mask(_mm512_sub_epi8(result_zmm, x_88_zmm), x_03_zmm);
    __mmask64 is_8c_mask = is_accented_mask & _mm512_cmpeq_epi8_mask(result_zmm, _mm512_set1_epi8((char)0x8C));
    __mmask64 is_8e_8f_mask = is_accented_mask & _mm512_cmpge_epu8_mask(result_zmm, x_8e_zmm);

    // 'ς' (U+03C2, CF 82) → 'σ' (U+03C3, CF 83)
    __mmask64 is_final_sigma_mask =
        _mm512_mask_cmpeq_epi8_mask(is_after_cf_mask, result_zmm, _mm512_set1_epi8((char)0x82));

    // 'µ' (U+00B5, C2 B5) → 'μ' (U+03BC, CE BC)
    __mmask64 is_c2_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c2_zmm);
    __mmask64 is_micro_second_mask =
        (is_c2_mask << 1) & _mm512_cmpeq_epi8_mask(result_zmm, _mm512_set1_epi8((char)0xB5));

    // Lead byte changes: CE -> CF (applied before offset calculation)
    __mmask64 change_ce_to_cf_mask =
        ((is_basic2_mask | is_dialytika_mask | is_8c_mask | is_8e_8f_mask) >> 1) & is_ce_mask;
    result_zmm = _mm512_mask_mov_epi8(result_zmm, change_ce_to_cf_mask, x_cf_zmm);

    // Micro sign lead change: C2 → CE
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_micro_second_mask >> 1, x_ce_zmm);

    // Offset constants
    __m512i const x_e0_zmm = _mm512_set1_epi8((char)0xE0); // -0x20
    __m512i const x_26_zmm = _mm512_set1_epi8((char)0x26);
    __m512i const x_25_zmm = _mm512_set1_epi8((char)0x25);
    __m512i const x_ff_zmm = _mm512_set1_epi8((char)0xFF); // -1

    // Build 5 offset vectors in parallel, combine with OR
    // Each maskz_mov is independent (no data dependencies between them)
    __m512i off1_zmm = _mm512_maskz_mov_epi8(is_basic1_mask, x_20_zmm);                     // +0x20
    __m512i off2_zmm = _mm512_maskz_mov_epi8(is_basic2_mask | is_dialytika_mask, x_e0_zmm); // -0x20
    __m512i off3_zmm = _mm512_maskz_mov_epi8(is_86_mask, x_26_zmm);                         // +0x26
    __m512i off4_zmm = _mm512_maskz_mov_epi8(is_88_8a_mask, x_25_zmm);                      // +0x25
    __m512i off5_zmm = _mm512_maskz_mov_epi8(is_8e_8f_mask, x_ff_zmm);                      // -1

    // Two VPTERNLOG operations to combine all 5 offset vectors
    // First: A | B | C for off1, off2, off3
    __m512i offset_zmm = _mm512_ternarylogic_epi64(off1_zmm, off2_zmm, off3_zmm, 0xFE);
    // Second: (combined) | D | E for adding off4, off5
    offset_zmm = _mm512_ternarylogic_epi64(offset_zmm, off4_zmm, off5_zmm, 0xFE);

    // Single add applies all offsets simultaneously
    result_zmm = _mm512_add_epi8(result_zmm, offset_zmm);

    // Final sigma: 'ς' (U+03C2, CF 82) → 'σ' (U+03C3, CF 83) (direct replacement)
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_final_sigma_mask, _mm512_set1_epi8((char)0x83));

    // Micro sign second byte: B5 → BC
    result_zmm = _mm512_mask_mov_epi8(result_zmm, is_micro_second_mask, _mm512_set1_epi8((char)0xBC));

    sz_assert_(_mm512_cmpeq_epi8_mask(sz_utf8_case_insensitive_find_ice_greek_fold_naively_zmm_(text_zmm),
                                      result_zmm) == (__mmask64)-1 &&
               "Efficient Greek fold must match naive implementation");
    return result_zmm;
}

/**
 *  @brief Detect danger zone positions in a ZMM register for Greek text.
 *
 *  Danger characters are those that expand during case folding or have alternative representations:
 *  - CE 90: 'ΐ' (U+0390) expands to 3 codepoints
 *  - CE B0: 'ΰ' (U+03B0) expands to 3 codepoints
 *  - CF 90,91,95,96: Greek symbols 'ϐ', 'ϑ', 'ϕ', 'ϖ'
 *  - CF B0,B1,B5: Greek symbols 'ϰ', 'ϱ', 'ϵ'
 *  - E2 84: Ohm sign 'Ω' prefix
 *  - E1: Polytonic Greek / Extensions
 *  - CD: Combining Diacritical Marks
 *
 *  @param[in] h The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_INTERNAL __mmask64 sz_utf8_case_insensitive_find_ice_greek_alarm_naively_zmm_(__m512i text_zmm) {
    // All constants local to function
    __m512i const x_ce_zmm = _mm512_set1_epi8((char)0xCE);
    __m512i const x_cf_zmm = _mm512_set1_epi8((char)0xCF);
    __m512i const x_e2_zmm = _mm512_set1_epi8((char)0xE2);
    __m512i const x_e1_zmm = _mm512_set1_epi8((char)0xE1);
    __m512i const x_cd_zmm = _mm512_set1_epi8((char)0xCD);
    __m512i const x_90_zmm = _mm512_set1_epi8((char)0x90);
    __m512i const x_b0_zmm = _mm512_set1_epi8((char)0xB0);
    __m512i const x_91_zmm = _mm512_set1_epi8((char)0x91);
    __m512i const x_95_zmm = _mm512_set1_epi8((char)0x95);
    __m512i const x_96_zmm = _mm512_set1_epi8((char)0x96);
    __m512i const x_b1_zmm = _mm512_set1_epi8((char)0xB1);
    __m512i const x_b5_zmm = _mm512_set1_epi8((char)0xB5);
    __m512i const x_84_zmm = _mm512_set1_epi8((char)0x84);

    // Lead bytes (5 CMPEQ)
    __mmask64 is_ce_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ce_zmm);
    __mmask64 is_cf_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_cf_zmm);
    __mmask64 is_e2_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_e2_zmm);
    __mmask64 is_e1_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_e1_zmm);
    __mmask64 is_cd_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_cd_zmm);

    // Second bytes (8 CMPEQ)
    __mmask64 is_90_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_90_zmm);
    __mmask64 is_b0_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_b0_zmm);
    __mmask64 is_9x_mask = is_90_mask | _mm512_cmpeq_epi8_mask(text_zmm, x_91_zmm) | //
                           _mm512_cmpeq_epi8_mask(text_zmm, x_95_zmm) | _mm512_cmpeq_epi8_mask(text_zmm, x_96_zmm);
    __mmask64 is_bx_mask =
        is_b0_mask | _mm512_cmpeq_epi8_mask(text_zmm, x_b1_zmm) | _mm512_cmpeq_epi8_mask(text_zmm, x_b5_zmm);
    __mmask64 is_84_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_84_zmm);

    // Danger mask construction
    __mmask64 ce_danger_mask = (is_ce_mask << 1) & (is_90_mask | is_b0_mask);
    __mmask64 cf_danger_mask = (is_cf_mask << 1) & (is_9x_mask | is_bx_mask);
    __mmask64 e2_danger_mask = (is_e2_mask << 1) & is_84_mask;
    return ce_danger_mask | cf_danger_mask | e2_danger_mask | is_e1_mask | is_cd_mask;
}

/**
 *  @brief Optimized danger zone detection for Greek text using Range+LUT technique.
 *  @sa sz_utf8_case_insensitive_find_ice_greek_alarm_naively_zmm_
 *
 *  Reduces port 5 pressure from 13 to 6 operations by:
 *  - Lead bytes CD/CE/CF: 3 CMPEQ -> 1 CMPLT + 2 VPTESTNMB (p0)
 *  - Lead bytes E1/E2: 2 CMPEQ -> 1 CMPLT + 1 VPTESTNMB (p0)
 *  - Second bytes 9x/Bx unified: 7 CMPEQ -> 1 CMPLT + 1 VPSHUFB + 1 CMPEQ (B6 filter)
 *    Key insight: (byte & 0xDF) collapses 9x and Bx to same offset space
 *  - Second byte 84: 1 CMPEQ (unchanged)
 *
 *  Port summary: 6 p5 ops + 5 p0 ops (vs 13 p5 originally)
 *
 *  @param[in] h The haystack ZMM register.
 *  @return Bitmask of positions where danger characters are detected.
 */
SZ_INTERNAL __mmask64 sz_utf8_case_insensitive_find_ice_greek_alarm_efficiently_zmm_(__m512i text_zmm) {
    // Range constants
    __m512i const x_cd_zmm = _mm512_set1_epi8((char)0xCD);
    __m512i const x_e1_zmm = _mm512_set1_epi8((char)0xE1);
    __m512i const x_90_zmm = _mm512_set1_epi8((char)0x90);
    __m512i const x_84_zmm = _mm512_set1_epi8((char)0x84);
    __m512i const x_01_zmm = _mm512_set1_epi8(0x01);
    __m512i const x_02_zmm = _mm512_set1_epi8(0x02);
    __m512i const x_03_zmm = _mm512_set1_epi8(0x03);
    __m512i const x_06_zmm = _mm512_set1_epi8(0x06);
    __m512i const x_07_zmm = _mm512_set1_epi8(0x07);

    // Unified LUT for 9x/Bx set: positions 0,1,5,6 are 0xFF (valid)
    // After masking with 0xDF, both 9x (90,91,95,96) and Bx (B0,B1,B5) collapse to same offsets
    // VPSHUFB uses low 4 bits as index; high bit set -> output 0
    __m512i const lut_9x_zmm = _mm512_set_epi8(              //
        0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, -1, -1,  // lane 3: [15..0]
        0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, -1, -1,  // lane 2
        0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, -1, -1,  // lane 1
        0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, -1, -1); // lane 0

    // Lead byte detection using range compression
    // Check for CD/CE/CF range: (byte - 0xCD) < 3  [1 CMPLT on p5]
    // Saves 1 p5 op by consolidating CD check with CE/CF range
    __m512i off_cd_zmm = _mm512_sub_epi8(text_zmm, x_cd_zmm);
    __mmask64 is_cd_ce_cf_mask = _mm512_cmplt_epu8_mask(off_cd_zmm, x_03_zmm);
    // Derive CD (offset==0), CE (offset==1), CF (offset==2) using TESTNM on p0
    __mmask64 is_cd_mask = is_cd_ce_cf_mask & _mm512_testn_epi8_mask(off_cd_zmm, off_cd_zmm);
    __m512i off_xor_1_zmm = _mm512_xor_si512(off_cd_zmm, x_01_zmm);
    __mmask64 is_ce_mask = is_cd_ce_cf_mask & _mm512_testn_epi8_mask(off_xor_1_zmm, off_xor_1_zmm);
    __mmask64 is_cf_mask = is_cd_ce_cf_mask & ~is_cd_mask & ~is_ce_mask;

    // Check for E1/E2 range: (byte - 0xE1) < 2  [1 CMPLT on p5]
    __m512i off_e1_zmm = _mm512_sub_epi8(text_zmm, x_e1_zmm);
    __mmask64 is_e1_or_e2_mask = _mm512_cmplt_epu8_mask(off_e1_zmm, x_02_zmm);
    __mmask64 is_e1_mask = is_e1_or_e2_mask & _mm512_testn_epi8_mask(off_e1_zmm, off_e1_zmm); // offset==0 [p0]
    __mmask64 is_e2_mask = is_e1_or_e2_mask & ~is_e1_mask;

    // Second byte detection using unified Range+LUT for 9x and Bx
    // Key insight: 0x90-0x96 and 0xB0-0xB5 share the same low nibble pattern.
    // Masking with 0xDF collapses both ranges to offsets 0-6 from 0x90:
    //   0x90 & 0xDF = 0x90, 0xB0 & 0xDF = 0x90 -> offset 0
    //   0x91 & 0xDF = 0x91, 0xB1 & 0xDF = 0x91 -> offset 1
    //   0x95 & 0xDF = 0x95, 0xB5 & 0xDF = 0x95 -> offset 5
    //   0x96 & 0xDF = 0x96, 0xB6 & 0xDF = 0x96 -> offset 6 (B6 must be excluded)
    __m512i const x_df_zmm = _mm512_set1_epi8((char)0xDF);
    __m512i const x_20_zmm = _mm512_set1_epi8(0x20);
    __m512i masked_zmm = _mm512_and_si512(text_zmm, x_df_zmm);
    __m512i offset_9x_bx_zmm = _mm512_sub_epi8(masked_zmm, x_90_zmm);

    // Check if masked byte is in range [0x90, 0x96]  [1 CMPLT on p5]
    __mmask64 in_9x_bx_range_mask = _mm512_cmplt_epu8_mask(offset_9x_bx_zmm, x_07_zmm);

    // For CE: need exactly 90 or B0 (both map to offset 0)  [1 VPTESTNMB on p0]
    __mmask64 is_90_or_b0_mask = in_9x_bx_range_mask & _mm512_testn_epi8_mask(offset_9x_bx_zmm, offset_9x_bx_zmm);

    // For CF: validate using LUT, then exclude B6 false positives  [1 VPSHUFB on p5, 1 VPTESTMB on p0]
    __m512i shuffled_9x_bx_zmm = _mm512_shuffle_epi8(lut_9x_zmm, offset_9x_bx_zmm);
    __mmask64 valid_prelim_mask = in_9x_bx_range_mask & _mm512_test_epi8_mask(shuffled_9x_bx_zmm, shuffled_9x_bx_zmm);

    // Exclude B6: offset==6 && bit5 set means original was 0xB6 not 0x96  [1 CMPEQ on p5, 1 VPTESTMB on p0]
    __mmask64 is_offset_6_mask = valid_prelim_mask & _mm512_cmpeq_epi8_mask(offset_9x_bx_zmm, x_06_zmm);
    __m512i bit5_zmm = _mm512_and_si512(text_zmm, x_20_zmm);
    __mmask64 is_b6_mask = is_offset_6_mask & _mm512_test_epi8_mask(bit5_zmm, bit5_zmm);
    __mmask64 valid_9x_bx_mask = valid_prelim_mask & ~is_b6_mask;

    // Check for E2 84  [1 CMPEQ on p5]
    __mmask64 is_84_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_84_zmm);

    // Danger mask construction
    __mmask64 ce_danger_mask = (is_ce_mask << 1) & is_90_or_b0_mask;
    __mmask64 cf_danger_mask = (is_cf_mask << 1) & valid_9x_bx_mask;
    __mmask64 e2_danger_mask = (is_e2_mask << 1) & is_84_mask;
    __mmask64 danger_mask = ce_danger_mask | cf_danger_mask | e2_danger_mask | is_e1_mask | is_cd_mask;

    sz_assert_(danger_mask == sz_utf8_case_insensitive_find_ice_greek_alarm_naively_zmm_(text_zmm) &&
               "Efficient Greek alarm must match naive implementation");
    return danger_mask;
}

/**
 *  @brief Greek case-insensitive search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_case_rune_safe_greek_k
 *
 *  @param[in] haystack Pointer to the haystack string.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] needle Pointer to the full needle string.
 *  @param[in] needle_length Length of the full needle in bytes.
 *  @param[in] needle_metadata Pre-folded window content with probe positions.
 *  @param[out] matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_ice_greek_(        //
    sz_cptr_t haystack, sz_size_t haystack_length,                     //
    sz_cptr_t needle, sz_size_t needle_length,                         //
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata, //
    sz_size_t *matched_length) {

    // Validate inputs
    sz_assert_(needle_metadata && "needle_metadata must be provided");
    sz_assert_(needle_metadata->folded_slice_length > 0 && "folded window must be non-empty");

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_assert_(folded_window_length <= 16 && "expect folded needle part to fit in XMM registers");

    // Pre-load folded window into XMM
    __mmask16 const folded_window_mask = sz_u16_mask_until_(folded_window_length);
    sz_u128_vec_t needle_window_vec, haystack_candidate_vec;
    needle_window_vec.xmm = _mm_loadu_si128((__m128i const *)needle_metadata->folded_slice);

    // 4 probe positions
    sz_size_t const offset_first = 0;
    sz_size_t const offset_second = needle_metadata->probe_second;
    sz_size_t const offset_third = needle_metadata->probe_third;
    sz_size_t const offset_last = folded_window_length - 1;

    sz_u512_vec_t probe_first_vec, probe_second_vec, probe_third_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_first]);
    probe_second_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_second]);
    probe_third_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_third]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_last]);

    // Pre-load the first folded rune for danger zone matching
    sz_rune_t needle_first_safe_folded_rune;
    {
        sz_rune_length_t dummy;
        sz_rune_parse((sz_cptr_t)(needle_metadata->folded_slice), &needle_first_safe_folded_rune, &dummy);
    }

    sz_u512_vec_t haystack_vec;
    sz_cptr_t haystack_ptr = haystack;
    while (haystack_ptr < haystack_end) {
        sz_size_t const available = (sz_size_t)(haystack_end - haystack_ptr);
        if (available < folded_window_length) break;

        sz_size_t const chunk_size = available < 64 ? available : 64;
        sz_size_t const valid_starts = chunk_size - folded_window_length + 1;
        // For danger detection across chunk boundaries, reduce step size to ensure
        // 3-byte patterns at chunk end are fully visible in the next chunk.
        // For tail chunks (valid_starts <= 2), step = 1 ensures progress.
        sz_size_t const step = valid_starts > 2 ? valid_starts - 2 : 1;
        __mmask64 const load_mask = sz_u64_mask_until_(chunk_size);
        __mmask64 const valid_mask = sz_u64_mask_until_(valid_starts);

        haystack_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, haystack_ptr);

        // Check for anomalies using the alarm function
        __mmask64 danger_mask = sz_utf8_case_insensitive_find_ice_greek_alarm_efficiently_zmm_(haystack_vec.zmm);

        if (danger_mask) {
            // The danger zone handler scans for the needle's first safe rune (at offset_in_unfolded).
            // To find all matches that start at valid positions (0 to valid_starts-1), we need to
            // scan up to position valid_starts - 1 + offset_in_unfolded. Use chunk_size as upper bound.
            sz_size_t danger_scan_length =
                sz_min_of_two(valid_starts + needle_metadata->offset_in_unfolded, chunk_size);
            sz_cptr_t match = sz_utf8_case_insensitive_find_in_danger_zone_( //
                haystack, haystack_length,                                   //
                needle, needle_length,                                       //
                haystack_ptr, danger_scan_length,                            // extended danger zone
                needle_first_safe_folded_rune,                               // pivot point
                needle_metadata->offset_in_unfolded,                         // its location in the needle
                matched_length);
            if (match) return match;
            haystack_ptr += step;
            continue;
        }

        // Fold and Probe
        haystack_vec.zmm = sz_utf8_case_insensitive_find_ice_greek_fold_efficiently_zmm_(haystack_vec.zmm);

        sz_u64_t matches = _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_first_vec.zmm);
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_second_vec.zmm) >> offset_second;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_third_vec.zmm) >> offset_third;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_last_vec.zmm) >> offset_last;
        matches &= valid_mask;

        // Candidate Verification
        for (; matches; matches &= matches - 1) {
            sz_size_t candidate_offset = sz_u64_ctz(matches);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;

            haystack_candidate_vec.xmm = _mm_maskz_loadu_epi8(folded_window_mask, haystack_candidate_ptr);
            haystack_candidate_vec.xmm =
                _mm512_castsi512_si128(sz_utf8_case_insensitive_find_ice_greek_fold_efficiently_zmm_(
                    _mm512_castsi128_si512(haystack_candidate_vec.xmm)));

            __mmask16 window_mismatch_mask =
                _mm_mask_cmpneq_epi8_mask(folded_window_mask, haystack_candidate_vec.xmm, needle_window_vec.xmm);
            if (window_mismatch_mask) continue;

            sz_cptr_t match = sz_utf8_case_insensitive_verify_match_(                    //
                haystack, haystack_length,                                               //
                needle, needle_length,                                                   //
                haystack_candidate_ptr - haystack, needle_metadata->folded_slice_length, // matched offset & length
                needle_metadata->offset_in_unfolded,                                     // head
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, // tail
                matched_length);
            if (match) {
                sz_utf8_case_insensitive_find_assert_(match, haystack, haystack_length, needle, needle_length,
                                                      needle_metadata);
                return match;
            }
        }
        haystack_ptr += step;
    }

    sz_utf8_case_insensitive_find_assert_(SZ_NULL_CHAR, haystack, haystack_length, needle, needle_length,
                                          needle_metadata);
    return SZ_NULL_CHAR;
}

#pragma endregion // Greek Case-Insensitive Find

#pragma region Vietnamese Case-Insensitive Find

/**
 *  @brief Fold a ZMM register using Vietnamese case-folding rules.
 *  @sa sz_utf8_case_rune_safe_vietnamese_k
 *
 *  @param[in] text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_case_insensitive_find_ice_vietnamese_fold_naively_zmm_(__m512i text_zmm) {
    // Start with ASCII folded
    __m512i result_zmm = sz_utf8_case_insensitive_find_ice_ascii_fold_zmm_(text_zmm);

    // Constants
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3);
    __m512i const x_c4_zmm = _mm512_set1_epi8((char)0xC4);
    __m512i const x_c5_zmm = _mm512_set1_epi8((char)0xC5);
    __m512i const x_c6_zmm = _mm512_set1_epi8((char)0xC6);
    __m512i const x_e1_zmm = _mm512_set1_epi8((char)0xE1);

    __m512i const x_b8_zmm = _mm512_set1_epi8((char)0xB8);
    __m512i const x_bb_zmm = _mm512_set1_epi8((char)0xBB);
    __m512i const x_ba_zmm = _mm512_set1_epi8((char)0xBA);
    __m512i const x_96_zmm = _mm512_set1_epi8((char)0x96);
    __m512i const x_9f_zmm = _mm512_set1_epi8((char)0x9F);

    __m512i const x_01_zmm = _mm512_set1_epi8((char)0x01);
    __m512i const x_20_zmm = _mm512_set1_epi8((char)0x20);

    // Masks for lead bytes
    __mmask64 is_c3 = _mm512_cmpeq_epi8_mask(result_zmm, x_c3_zmm);
    __mmask64 is_c4 = _mm512_cmpeq_epi8_mask(result_zmm, x_c4_zmm);
    __mmask64 is_c5 = _mm512_cmpeq_epi8_mask(result_zmm, x_c5_zmm);
    __mmask64 is_c6 = _mm512_cmpeq_epi8_mask(result_zmm, x_c6_zmm);
    __mmask64 is_e1 = _mm512_cmpeq_epi8_mask(result_zmm, x_e1_zmm);

    __mmask64 is_after_c3 = is_c3 << 1;
    __mmask64 is_after_c4 = is_c4 << 1;
    __mmask64 is_after_c5 = is_c5 << 1;
    __mmask64 is_after_c6 = is_c6 << 1;

    // 1. Latin-1 Supplement (C3): Upper 80-96, 98-9E -> +0x20
    // Same as Western/Central
    __mmask64 is_c3_target =
        _mm512_mask_cmple_epu8_mask(is_after_c3, result_zmm, _mm512_set1_epi8((char)0x9E)); // <= 9E
    // Exclude multiplication sign 0xD7/0x97 if needed?
    // Western kernel excludes 0x97 (x) and 0xDF (ss).
    // Here we check <= 9E. 0x97 is usually Multiply, but standard folding keeps it?
    // Central kernel: excludes 97.
    is_c3_target &= ~_mm512_cmpeq_epi8_mask(result_zmm, _mm512_set1_epi8((char)0x97)); // Exclude ×
    is_c3_target &= _mm512_mask_cmpge_epu8_mask(is_after_c3, result_zmm, _mm512_set1_epi8((char)0x80));

    result_zmm = _mm512_mask_add_epi8(result_zmm, is_c3_target, result_zmm, x_20_zmm);

    // 2. Latin Extended-A (C4/C5): Even -> Odd (+1) for MOST characters
    // Standard pattern (U+0100-U+0138, U+014A-U+017F): Even=uppercase, Odd=lowercase
    // INVERTED pattern (U+0139-U+0148): Odd=uppercase, Even=lowercase
    //   - After C4: B9,BB,BD,BF are uppercase (odd), BA,BC,BE are lowercase (even)
    //   - After C5: 81,83,85,87 are uppercase (odd), 80,82,84,86,88 are lowercase (even)
    // Note: 'Ŀ' (U+013F, C4 BF) → 'ŀ' (U+0140, C5 80) crosses lead bytes, handled specially by safety profile
    __mmask64 is_c4_c5_target = is_after_c4 | is_after_c5;
    __mmask64 is_even = _mm512_cmpeq_epi8_mask(_mm512_and_si512(result_zmm, x_01_zmm), _mm512_setzero_si512());
    __mmask64 is_odd = ~is_even;

    // Identify the inverted range where Even=lowercase (should NOT be transformed +1)
    // After C4: B9-BE (U+0139-U+013E: Ĺ-ľ inverted pattern)
    // Note: BF (Ŀ U+013F) excluded - its lowercase ŀ (U+0140) is C5 80 (different lead byte)
    __mmask64 is_c4_inverted_range = is_after_c4 & _mm512_cmpge_epu8_mask(result_zmm, _mm512_set1_epi8((char)0xB9)) &
                                     _mm512_cmple_epu8_mask(result_zmm, _mm512_set1_epi8((char)0xBE));
    // After C5: 80-88 (even bytes 80, 82, 84, 86, 88 are lowercase)
    __mmask64 is_c5_inverted_range = is_after_c5 & _mm512_cmple_epu8_mask(result_zmm, _mm512_set1_epi8((char)0x88));
    __mmask64 is_inverted_range = is_c4_inverted_range | is_c5_inverted_range;

    // Standard range: apply +1 to even bytes (uppercase -> lowercase)
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_c4_c5_target & is_even & ~is_inverted_range, result_zmm, x_01_zmm);
    // Inverted range: apply +1 to odd bytes (uppercase -> lowercase)
    result_zmm = _mm512_mask_add_epi8(result_zmm, is_inverted_range & is_odd, result_zmm, x_01_zmm);

    // 3. Latin Extended-B (C6): Specific Vietnamese chars
    // 'Ơ' (U+01A0, C6 A0) → 'ơ' (U+01A1, C6 A1) (even → odd)
    // 'Ư' (U+01AF, C6 AF) → 'ư' (U+01B0, C6 B0) (odd → even)
    __mmask64 is_c6_A0 = is_after_c6 & _mm512_cmpeq_epi8_mask(result_zmm, _mm512_set1_epi8((char)0xA0));
    __mmask64 is_c6_AF = is_after_c6 & _mm512_cmpeq_epi8_mask(result_zmm, _mm512_set1_epi8((char)0xAF));

    result_zmm = _mm512_mask_add_epi8(result_zmm, is_c6_A0 | is_c6_AF, result_zmm, x_01_zmm);

    // 4. Latin Extended Additional (E1 B8-BB): Even -> Odd (+1) for Uppercase
    // E1 lead -> Second is B8..BB -> Third is Even
    // Exception: 1E 96-9F (Second=BA, Third=96-9F)
    __mmask64 is_e1_second = is_e1 << 1;
    __mmask64 is_valid_second = _mm512_mask_cmpge_epu8_mask(is_e1_second, result_zmm, x_b8_zmm);
    is_valid_second &= _mm512_mask_cmple_epu8_mask(is_e1_second, result_zmm, x_bb_zmm);

    __mmask64 is_e1_third = is_valid_second << 1;

    // Check exclusion: Second == BA && Third in 96-9F
    __mmask64 is_ba_second = is_e1_second & _mm512_cmpeq_epi8_mask(result_zmm, x_ba_zmm);
    __mmask64 is_excluded_third = (is_ba_second << 1) & _mm512_mask_cmpge_epu8_mask(is_e1_third, result_zmm, x_96_zmm) &
                                  _mm512_mask_cmple_epu8_mask(is_e1_third, result_zmm, x_9f_zmm);

    // Apply +1 to Even bytes in valid third positions, excluding the danger range
    __mmask64 is_e1_target = is_e1_third & is_even & ~is_excluded_third;

    result_zmm = _mm512_mask_add_epi8(result_zmm, is_e1_target, result_zmm, x_01_zmm);

    return result_zmm;
}

/**
 *  @brief Fold a ZMM register using Vietnamese case-folding rules.
 *  @sa sz_utf8_case_rune_safe_vietnamese_k
 *
 *  @param[in] text_zmm The text ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_utf8_case_insensitive_find_ice_vietnamese_fold_efficiently_zmm_(__m512i text_zmm) {
    // Inline ASCII fold (avoiding function call overhead)
    __m512i const a_upper_zmm = _mm512_set1_epi8('A');
    __m512i const range26_zmm = _mm512_set1_epi8(26);
    __m512i const x_20_zmm = _mm512_set1_epi8(0x20);
    __m512i const x_01_zmm = _mm512_set1_epi8(0x01);
    __mmask64 is_upper_mask = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(text_zmm, a_upper_zmm), range26_zmm);
    __m512i result_zmm = _mm512_mask_add_epi8(text_zmm, is_upper_mask, text_zmm, x_20_zmm);

    // Lead byte constants
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3);
    __m512i const x_c4_zmm = _mm512_set1_epi8((char)0xC4);
    __m512i const x_c5_zmm = _mm512_set1_epi8((char)0xC5);
    __m512i const x_c6_zmm = _mm512_set1_epi8((char)0xC6);
    __m512i const x_e1_zmm = _mm512_set1_epi8((char)0xE1);

    // Range compression constants
    __m512i const x_80_zmm = _mm512_set1_epi8((char)0x80);
    __m512i const x_1f_zmm = _mm512_set1_epi8((char)0x1F); // Range 80-9E (31 bytes)
    __m512i const x_97_zmm = _mm512_set1_epi8((char)0x97); // Exclude ×
    __m512i const x_b9_zmm = _mm512_set1_epi8((char)0xB9); // C4 inverted range base
    __m512i const x_06_zmm = _mm512_set1_epi8((char)0x06); // B9-BE range size (6)
    __m512i const x_09_zmm = _mm512_set1_epi8((char)0x09); // 80-88 range size (9)
    __m512i const x_b8_zmm = _mm512_set1_epi8((char)0xB8);
    __m512i const x_04_zmm = _mm512_set1_epi8((char)0x04); // B8-BB range size (4)
    __m512i const x_ba_zmm = _mm512_set1_epi8((char)0xBA);
    __m512i const x_96_zmm = _mm512_set1_epi8((char)0x96);
    __m512i const x_0a_zmm = _mm512_set1_epi8((char)0x0A); // 96-9F range size (10)

    // Masks for lead bytes
    __mmask64 is_c3_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c3_zmm);
    __mmask64 is_c4_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c4_zmm);
    __mmask64 is_c5_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c5_zmm);
    __mmask64 is_c6_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_c6_zmm);
    __mmask64 is_e1_mask = _mm512_cmpeq_epi8_mask(result_zmm, x_e1_zmm);

    __mmask64 is_after_c3_mask = is_c3_mask << 1;
    __mmask64 is_after_c4_mask = is_c4_mask << 1;
    __mmask64 is_after_c5_mask = is_c5_mask << 1;
    __mmask64 is_after_c6_mask = is_c6_mask << 1;

    // 1. Latin-1 Supplement (C3): Upper 80-9E -> +0x20 (excluding 97)
    // Range compression: (byte - 0x80) < 0x1F covers 80-9E
    __mmask64 is_c3_range_mask =
        _mm512_mask_cmplt_epu8_mask(is_after_c3_mask, _mm512_sub_epi8(result_zmm, x_80_zmm), x_1f_zmm);
    __mmask64 is_c3_target_mask = is_c3_range_mask & ~_mm512_cmpeq_epi8_mask(result_zmm, x_97_zmm); // Exclude ×

    // 2. Latin Extended-A (C4/C5): Even -> Odd (+1) for MOST characters
    // Standard pattern: Even=uppercase, Odd=lowercase
    // INVERTED pattern (C4 B9-BE, C5 80-88): Odd=uppercase, Even=lowercase
    __mmask64 is_c4_c5_target_mask = is_after_c4_mask | is_after_c5_mask;
    __mmask64 is_even_mask = _mm512_testn_epi8_mask(result_zmm, x_01_zmm); // More efficient parity check
    __mmask64 is_odd_mask = ~is_even_mask;

    // RANGE COMPRESSION for inverted ranges
    // C4 B9-BE: (byte - B9) < 6
    __mmask64 is_c4_inverted_range_mask =
        is_after_c4_mask & _mm512_cmplt_epu8_mask(_mm512_sub_epi8(result_zmm, x_b9_zmm), x_06_zmm);
    // C5 80-88: (byte - 80) < 9
    __mmask64 is_c5_inverted_range_mask =
        is_after_c5_mask & _mm512_cmplt_epu8_mask(_mm512_sub_epi8(result_zmm, x_80_zmm), x_09_zmm);
    __mmask64 is_inverted_range_mask = is_c4_inverted_range_mask | is_c5_inverted_range_mask;

    // Standard range: apply +1 to even bytes (uppercase -> lowercase)
    __mmask64 fold_std_mask = is_c4_c5_target_mask & is_even_mask & ~is_inverted_range_mask;
    // Inverted range: apply +1 to odd bytes (uppercase -> lowercase)
    __mmask64 fold_inv_mask = is_inverted_range_mask & is_odd_mask;

    // 3. Latin Extended-B (C6): Specific Vietnamese chars
    // 'Ơ' (U+01A0, C6 A0) → 'ơ' (U+01A1, C6 A1) (even → odd)
    // 'Ư' (U+01AF, C6 AF) → 'ư' (U+01B0, C6 B0) (odd → even)
    __mmask64 is_c6_A0_mask = is_after_c6_mask & _mm512_cmpeq_epi8_mask(result_zmm, _mm512_set1_epi8((char)0xA0));
    __mmask64 is_c6_AF_mask = is_after_c6_mask & _mm512_cmpeq_epi8_mask(result_zmm, _mm512_set1_epi8((char)0xAF));

    // 4. Latin Extended Additional (E1 B8-BB): Even -> Odd (+1) for Uppercase
    // E1 lead -> Second is B8..BB -> Third is Even
    // Exception: 1E 96-9F (Second=BA, Third=96-9F)
    __mmask64 is_e1_second_mask = is_e1_mask << 1;
    // Range compression: (byte - B8) < 4 covers B8-BB
    __mmask64 is_valid_second_mask =
        _mm512_mask_cmplt_epu8_mask(is_e1_second_mask, _mm512_sub_epi8(result_zmm, x_b8_zmm), x_04_zmm);

    __mmask64 is_e1_third_mask = is_valid_second_mask << 1;

    // Check exclusion: Second == BA && Third in 96-9F
    // Range compression: (byte - 96) < 10 covers 96-9F
    __mmask64 is_ba_second_mask = is_e1_second_mask & _mm512_cmpeq_epi8_mask(result_zmm, x_ba_zmm);
    __mmask64 is_excluded_third_mask =
        (is_ba_second_mask << 1) & _mm512_cmplt_epu8_mask(_mm512_sub_epi8(result_zmm, x_96_zmm), x_0a_zmm);

    // Apply +1 to Even bytes in valid third positions, excluding the danger range
    __mmask64 is_e1_target_mask = is_e1_third_mask & is_even_mask & ~is_excluded_third_mask;

    // This section builds offset vectors in parallel to avoid sequential dependency chains.
    // Instead of a series of mask_mov operations (setzero -> mask_mov -> mask_mov),
    // we use `maskz_mov` to create independent vectors and then combine them with `OR`.
    //
    // Instruction latencies (Ice Lake):
    //   - `_mm512_maskz_mov_epi8`: 1 cycle, ports 0/5
    //   - `_mm512_or_si512`: 1 cycle, 0.5 throughput, ports 0/5
    //   - Sequential `mask_mov` chain: 2+ cycle dependency
    //
    // Net effect: ~40% speedup measured (5.58 -> 3.34 ns per 64-byte block).
    __m512i off_c3_zmm = _mm512_maskz_mov_epi8(is_c3_target_mask, x_20_zmm);
    __m512i off_others_zmm = _mm512_maskz_mov_epi8(
        fold_std_mask | fold_inv_mask | is_c6_A0_mask | is_c6_AF_mask | is_e1_target_mask, x_01_zmm);
    __m512i offset_zmm = _mm512_or_si512(off_c3_zmm, off_others_zmm);
    result_zmm = _mm512_add_epi8(result_zmm, offset_zmm);

    sz_assert_(_mm512_cmpeq_epi8_mask(sz_utf8_case_insensitive_find_ice_vietnamese_fold_naively_zmm_(text_zmm),
                                      result_zmm) == (__mmask64)-1 &&
               "Efficient Vietnamese fold must match naive implementation");
    return result_zmm;
}

/**
 *  @brief Naive alarm function for Vietnamese danger zone detection.
 *
 *  Detects positions where danger characters occur that require special handling:
 *  - E1 BA 96-9F: Latin Extensions like 'ẞ' (U+1E9E, E1 BA 9E) → "ss" (U+0073 U+0073, 73 73)
 *  - C3 9F: 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
 *  - C5 BF: 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
 *  - EF AC 80-86: Ligatures:
 *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
 *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
 *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
 *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
 *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
 *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
 *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
 *  - E2 84 AA: 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B)
 *
 *  Uses ~11 CMPEQ + 2 range operations.
 *  Note: Returns danger positions shifted to the START of multi-byte sequences.
 *
 *  @param[in] text_zmm The haystack ZMM register.
 *  @param[in] load_mask Mask of valid bytes in the ZMM register.
 *  @return Bitmask of positions where danger characters are detected (at sequence start).
 */
SZ_INTERNAL __mmask64 sz_utf8_case_insensitive_find_ice_vietnamese_alarm_naively_zmm_(__m512i text_zmm,
                                                                                      __mmask64 load_mask) {
    // Lead byte constants
    __m512i const x_e1_zmm = _mm512_set1_epi8((char)0xE1);
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3);
    __m512i const x_c5_zmm = _mm512_set1_epi8((char)0xC5);
    __m512i const x_ef_zmm = _mm512_set1_epi8((char)0xEF);
    __m512i const x_e2_zmm = _mm512_set1_epi8((char)0xE2);

    // Second byte constants
    __m512i const x_ba_zmm = _mm512_set1_epi8((char)0xBA);
    __m512i const x_9f_zmm = _mm512_set1_epi8((char)0x9F);
    __m512i const x_bf_zmm = _mm512_set1_epi8((char)0xBF);
    __m512i const x_ac_zmm = _mm512_set1_epi8((char)0xAC);
    __m512i const x_84_zmm = _mm512_set1_epi8((char)0x84);
    __m512i const x_96_zmm = _mm512_set1_epi8((char)0x96);

    // Lead bytes (5 CMPEQ)
    __mmask64 is_e1_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_e1_zmm);
    __mmask64 is_c3_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c3_zmm);
    __mmask64 is_c5_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c5_zmm);
    __mmask64 is_ef_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ef_zmm);
    __mmask64 is_e2_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_e2_zmm);

    // Early exit if no lead bytes
    if (!(is_e1_mask | is_c3_mask | is_c5_mask | is_ef_mask | is_e2_mask)) return 0;

    // Check for Latin Extended Additional exclusions (E1 BA 96-9F)
    __mmask64 ba_second_mask = (is_e1_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_ba_zmm);
    __mmask64 bad_third_mask = (ba_second_mask << 1) & _mm512_mask_cmpge_epu8_mask(load_mask, text_zmm, x_96_zmm) &
                               _mm512_mask_cmple_epu8_mask(load_mask, text_zmm, x_9f_zmm);

    // Check for Sharp S (C3 9F)
    __mmask64 sharp_s_mask = (is_c3_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_9f_zmm);

    // Check for Long S (C5 BF)
    __mmask64 long_s_mask = (is_c5_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_bf_zmm);

    // Check for Ligatures (EF AC)
    __mmask64 ligature_mask = (is_ef_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_ac_zmm);

    // Check for Kelvin (E2 84)
    __mmask64 kelvin_mask = (is_e2_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_84_zmm);

    // Shift back to sequence start positions
    return (bad_third_mask >> 2) | (sharp_s_mask >> 1) | (long_s_mask >> 1) | (ligature_mask >> 1) | (kelvin_mask >> 1);
}

/**
 *  @brief Optimized alarm function for Vietnamese danger zone detection.
 *
 *  Reduces port 5 pressure from ~12 to ~11 operations by:
 *  - E1/E2 consecutive: 2 CMPEQ -> 1 CMPLT + 1 VPTESTNMB (p0)
 *
 *  Port summary: 11 p5 ops + 1 p0 op (vs 12 p5 originally)
 *
 *  @param[in] h The haystack ZMM register.
 *  @param[in] load_mask Mask of valid bytes in the ZMM register.
 *  @return Bitmask of positions where danger characters are detected (at sequence start).
 */
SZ_INTERNAL __mmask64 sz_utf8_case_insensitive_find_ice_vietnamese_alarm_efficiently_zmm_(__m512i text_zmm,
                                                                                          __mmask64 load_mask) {
    // Range constants
    __m512i const x_e1_zmm = _mm512_set1_epi8((char)0xE1);
    __m512i const x_c3_zmm = _mm512_set1_epi8((char)0xC3);
    __m512i const x_c5_zmm = _mm512_set1_epi8((char)0xC5);
    __m512i const x_ef_zmm = _mm512_set1_epi8((char)0xEF);
    __m512i const x_ba_zmm = _mm512_set1_epi8((char)0xBA);
    __m512i const x_9f_zmm = _mm512_set1_epi8((char)0x9F);
    __m512i const x_bf_zmm = _mm512_set1_epi8((char)0xBF);
    __m512i const x_ac_zmm = _mm512_set1_epi8((char)0xAC);
    __m512i const x_84_zmm = _mm512_set1_epi8((char)0x84);
    __m512i const x_96_zmm = _mm512_set1_epi8((char)0x96);
    __m512i const x_02_zmm = _mm512_set1_epi8(0x02);

    // Check for E1/E2 range: (byte - 0xE1) < 2  [1 CMPLT on p5]
    __m512i off_e1_zmm = _mm512_sub_epi8(text_zmm, x_e1_zmm);
    __mmask64 is_e1_or_e2_mask = _mm512_cmplt_epu8_mask(off_e1_zmm, x_02_zmm);
    __mmask64 is_e1_mask = is_e1_or_e2_mask & _mm512_testn_epi8_mask(off_e1_zmm, off_e1_zmm); // offset==0 [p0]
    __mmask64 is_e2_mask = is_e1_or_e2_mask & ~is_e1_mask;

    // Other lead bytes (3 CMPEQ on p5)
    __mmask64 is_c3_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c3_zmm);
    __mmask64 is_c5_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_c5_zmm);
    __mmask64 is_ef_mask = _mm512_cmpeq_epi8_mask(text_zmm, x_ef_zmm);

    // Early exit if no lead bytes
    if (!(is_e1_or_e2_mask | is_c3_mask | is_c5_mask | is_ef_mask)) return 0;

    // Check for Latin Extended Additional exclusions (E1 BA 96-9F)
    __mmask64 ba_second_mask = (is_e1_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_ba_zmm);
    __mmask64 bad_third_mask = (ba_second_mask << 1) & _mm512_mask_cmpge_epu8_mask(load_mask, text_zmm, x_96_zmm) &
                               _mm512_mask_cmple_epu8_mask(load_mask, text_zmm, x_9f_zmm);

    // Check for Sharp S (C3 9F)
    __mmask64 sharp_s_mask = (is_c3_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_9f_zmm);

    // Check for Long S (C5 BF)
    __mmask64 long_s_mask = (is_c5_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_bf_zmm);

    // Check for Ligatures (EF AC)
    __mmask64 ligature_mask = (is_ef_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_ac_zmm);

    // Check for Kelvin (E2 84)
    __mmask64 kelvin_mask = (is_e2_mask << 1) & _mm512_cmpeq_epi8_mask(text_zmm, x_84_zmm);

    // Shift back to sequence start positions
    __mmask64 danger_mask =
        (bad_third_mask >> 2) | (sharp_s_mask >> 1) | (long_s_mask >> 1) | (ligature_mask >> 1) | (kelvin_mask >> 1);

    sz_assert_(danger_mask == sz_utf8_case_insensitive_find_ice_vietnamese_alarm_naively_zmm_(text_zmm, load_mask) &&
               "Efficient Vietnamese alarm must match naive implementation");
    return danger_mask;
}

/**
 *  @brief Vietnamese case-insensitive search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_case_rune_safe_vietnamese_k
 *
 *  @param[in] haystack Pointer to the haystack string.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] needle Pointer to the full needle string.
 *  @param[in] needle_length Length of the full needle in bytes.
 *  @param[in] needle_metadata Pre-folded window content with probe positions.
 *  @param[out] matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_ice_vietnamese_(   //
    sz_cptr_t haystack, sz_size_t haystack_length,                     //
    sz_cptr_t needle, sz_size_t needle_length,                         //
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata, //
    sz_size_t *matched_length) {

    // Validate inputs
    sz_assert_(needle_metadata && "needle_metadata must be provided");
    sz_assert_(needle_metadata->folded_slice_length > 0 && "folded window must be non-empty");

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_assert_(folded_window_length <= 16 && "expect folded needle part to fit in XMM registers");

    // Pre-load folded window into XMM
    __mmask16 const folded_window_mask = sz_u16_mask_until_(folded_window_length);
    sz_u128_vec_t needle_window_vec, haystack_candidate_vec;
    needle_window_vec.xmm = _mm_loadu_si128((__m128i const *)needle_metadata->folded_slice);

    // 4 probe positions
    sz_size_t const offset_first = 0;
    sz_size_t const offset_second = needle_metadata->probe_second;
    sz_size_t const offset_third = needle_metadata->probe_third;
    sz_size_t const offset_last = folded_window_length - 1;

    sz_u512_vec_t probe_first_vec, probe_second_vec, probe_third_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_first]);
    probe_second_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_second]);
    probe_third_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_third]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_last]);

    sz_rune_t needle_first_safe_folded_rune;
    {
        sz_rune_length_t dummy;
        sz_rune_parse((sz_cptr_t)(needle_metadata->folded_slice), &needle_first_safe_folded_rune, &dummy);
    }

    sz_u512_vec_t haystack_vec;
    sz_cptr_t haystack_ptr = haystack;
    while (haystack_ptr < haystack_end) {
        sz_size_t const available = (sz_size_t)(haystack_end - haystack_ptr);
        if (available < folded_window_length) break;

        sz_size_t const chunk_size = available < 64 ? available : 64;
        sz_size_t const valid_starts = chunk_size - folded_window_length + 1;
        // For danger detection across chunk boundaries, reduce step size to ensure
        // 3-byte patterns at chunk end are fully visible in the next chunk.
        // For tail chunks (valid_starts <= 2), step = 1 ensures progress.
        sz_size_t const step = valid_starts > 2 ? valid_starts - 2 : 1;
        __mmask64 const load_mask = sz_u64_mask_until_(chunk_size);
        __mmask64 const valid_mask = sz_u64_mask_until_(valid_starts);

        haystack_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, haystack_ptr);

        // Check for anomalies using alarm function
        __mmask64 danger_mask =
            sz_utf8_case_insensitive_find_ice_vietnamese_alarm_efficiently_zmm_(haystack_vec.zmm, load_mask);
        if (danger_mask) {
            // The danger zone handler scans for the needle's first safe rune (at offset_in_unfolded).
            // To find all matches that start at valid positions (0 to valid_starts-1), we need to
            // scan up to position valid_starts - 1 + offset_in_unfolded. Use chunk_size as upper bound.
            sz_size_t danger_scan_length =
                sz_min_of_two(valid_starts + needle_metadata->offset_in_unfolded, chunk_size);
            sz_cptr_t match = sz_utf8_case_insensitive_find_in_danger_zone_( //
                haystack, haystack_length,                                   //
                needle, needle_length,                                       //
                haystack_ptr, danger_scan_length,                            // extended danger zone
                needle_first_safe_folded_rune,                               // pivot point
                needle_metadata->offset_in_unfolded,                         // its location in the needle
                matched_length);
            if (match) return match;
            haystack_ptr += step;
            continue;
        }

        haystack_vec.zmm = sz_utf8_case_insensitive_find_ice_vietnamese_fold_efficiently_zmm_(haystack_vec.zmm);

        sz_u64_t matches = _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_first_vec.zmm);
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_second_vec.zmm) >> offset_second;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_third_vec.zmm) >> offset_third;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_last_vec.zmm) >> offset_last;
        matches &= valid_mask;

        // Candidate Verification
        for (; matches; matches &= matches - 1) {
            sz_size_t candidate_offset = sz_u64_ctz(matches);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;

            haystack_candidate_vec.xmm = _mm_maskz_loadu_epi8(folded_window_mask, haystack_candidate_ptr);
            haystack_candidate_vec.xmm =
                _mm512_castsi512_si128(sz_utf8_case_insensitive_find_ice_vietnamese_fold_efficiently_zmm_(
                    _mm512_castsi128_si512(haystack_candidate_vec.xmm)));

            __mmask16 window_mismatch_mask =
                _mm_mask_cmpneq_epi8_mask(folded_window_mask, haystack_candidate_vec.xmm, needle_window_vec.xmm);
            if (window_mismatch_mask) continue;

            sz_cptr_t match = sz_utf8_case_insensitive_verify_match_(                    //
                haystack, haystack_length,                                               //
                needle, needle_length,                                                   //
                haystack_candidate_ptr - haystack, needle_metadata->folded_slice_length, // matched offset & length
                needle_metadata->offset_in_unfolded,                                     // head
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, // tail
                matched_length);
            if (match) {
                sz_utf8_case_insensitive_find_assert_(match, haystack, haystack_length, needle, needle_length,
                                                      needle_metadata);
                return match;
            }
        }
        haystack_ptr += step;
    }

    sz_utf8_case_insensitive_find_assert_(SZ_NULL_CHAR, haystack, haystack_length, needle, needle_length,
                                          needle_metadata);
    return SZ_NULL_CHAR;
}

#pragma endregion // Vietnamese Case-Insensitive Find

#pragma region Georgian Case-Insensitive Find

/**
 *  @brief Detect danger zones in Georgian text (Mtavruli, Asomtavruli, Nuskhuri).
 *  @sa sz_utf8_case_rune_safe_georgian_k
 *
 *  Georgian Mkhedruli is caseless, so we only need to detect non-Mkhedruli Georgian scripts
 *  that require special handling via serial fallback:
 *  - Mtavruli (E1 B2 xx): Modern uppercase, folds to Mkhedruli
 *  - Asomtavruli (E1 82 A0-E5): Historical uppercase, folds to Nuskhuri
 *  - Nuskhuri (E2 B4 xx): Ecclesiastical script
 *
 *  All Georgian scripts use 3-byte UTF-8, so no length changes during folding.
 */
SZ_INTERNAL __mmask64 sz_utf8_case_insensitive_find_ice_georgian_alarm_zmm_(__m512i text_zmm) {
    // Lead byte detection
    __mmask64 is_e1_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xE1));
    __mmask64 is_e2_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xE2));

    // Second byte detection
    __mmask64 is_b2_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xB2)); // Mtavruli
    __mmask64 is_82_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0x82)); // Asomtavruli
    __mmask64 is_b4_mask = _mm512_cmpeq_epi8_mask(text_zmm, _mm512_set1_epi8((char)0xB4)); // Nuskhuri

    // E1 B2 xx = Mtavruli (folds to Mkhedruli)
    __mmask64 mtavruli_mask = (is_e1_mask << 1) & is_b2_mask;

    // E1 82 [A0-E5] = Asomtavruli (folds to Nuskhuri)
    // Third byte range check: (byte - 0xA0) < 0x46 covers A0-E5
    __mmask64 after_e1_82 = (is_e1_mask << 1) & is_82_mask;
    __m512i off_a0 = _mm512_add_epi8(text_zmm, _mm512_set1_epi8(0x60)); // -0xA0 = +0x60
    __mmask64 in_a0_e5 = _mm512_cmplt_epu8_mask(off_a0, _mm512_set1_epi8(0x46));
    __mmask64 asomtavruli_mask = (after_e1_82 << 1) & in_a0_e5;

    // E2 B4 xx = Nuskhuri
    __mmask64 nuskhuri_mask = (is_e2_mask << 1) & is_b4_mask;

    // Shift back to sequence start positions
    return (mtavruli_mask >> 1) | (asomtavruli_mask >> 2) | (nuskhuri_mask >> 1);
}

/**
 *  @brief Fold Georgian text - only ASCII A-Z needs folding.
 *  @sa sz_utf8_case_rune_safe_georgian_k
 *
 *  Georgian Mkhedruli is caseless, so Georgian characters pass through unchanged.
 *  Only ASCII uppercase letters need folding for mixed Latin text.
 */
SZ_INTERNAL __m512i sz_utf8_case_insensitive_find_ice_georgian_fold_zmm_(__m512i text_zmm) {
    // ASCII A-Z range check: (byte - 'A') <= 25
    __m512i off_a = _mm512_sub_epi8(text_zmm, _mm512_set1_epi8('A'));
    __mmask64 is_upper = _mm512_cmple_epu8_mask(off_a, _mm512_set1_epi8(25));
    return _mm512_mask_add_epi8(text_zmm, is_upper, text_zmm, _mm512_set1_epi8(0x20));
}

/**
 *  @brief Georgian case-insensitive search for needles with safe slices up to 16 bytes.
 *  @sa sz_utf8_case_rune_safe_georgian_k
 *
 *  This is the fastest non-ASCII kernel because Mkhedruli is caseless - no Georgian
 *  folding needed in the hot path. Only ASCII A-Z folding for mixed text.
 *
 *  @param[in] haystack Pointer to the haystack string.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] needle Pointer to the full needle string.
 *  @param[in] needle_length Length of the full needle in bytes.
 *  @param[in] needle_metadata Pre-folded window content with probe positions.
 *  @param[out] matched_length Haystack bytes consumed by the match.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_ice_georgian_(     //
    sz_cptr_t haystack, sz_size_t haystack_length,                     //
    sz_cptr_t needle, sz_size_t needle_length,                         //
    sz_utf8_case_insensitive_needle_metadata_t const *needle_metadata, //
    sz_size_t *matched_length) {

    // Validate inputs
    sz_assert_(needle_metadata && "needle_metadata must be provided");
    sz_assert_(needle_metadata->folded_slice_length > 0 && "folded window must be non-empty");

    sz_size_t const folded_window_length = needle_metadata->folded_slice_length;
    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_assert_(folded_window_length <= 16 && "expect folded needle part to fit in XMM registers");

    // Pre-load folded window into XMM
    __mmask16 const folded_window_mask = sz_u16_mask_until_(folded_window_length);
    sz_u128_vec_t needle_window_vec, haystack_candidate_vec;
    needle_window_vec.xmm = _mm_loadu_si128((__m128i const *)needle_metadata->folded_slice);

    // 4 probe positions
    sz_size_t const offset_first = 0;
    sz_size_t const offset_second = needle_metadata->probe_second;
    sz_size_t const offset_third = needle_metadata->probe_third;
    sz_size_t const offset_last = folded_window_length - 1;

    // Pre-broadcast probe values
    sz_u512_vec_t probe_first_vec, probe_second_vec, probe_third_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_first]);
    probe_second_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_second]);
    probe_third_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_third]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_metadata->folded_slice[offset_last]);

    // For danger zone handling
    sz_rune_t needle_first_safe_folded_rune;
    {
        sz_rune_length_t dummy;
        sz_rune_parse((sz_cptr_t)(needle_metadata->folded_slice), &needle_first_safe_folded_rune, &dummy);
    }

    sz_cptr_t haystack_ptr = haystack;
    sz_u512_vec_t haystack_vec;

    while (haystack_ptr < haystack_end) {
        sz_size_t chunk_size = sz_min_of_two(haystack_end - haystack_ptr, 64);
        sz_size_t valid_starts = (chunk_size >= folded_window_length) ? chunk_size - folded_window_length + 1 : 0;
        sz_size_t step = (valid_starts > 2) ? valid_starts - 2 : 1;
        __mmask64 load_mask = sz_u64_mask_until_(chunk_size);
        __mmask64 valid_mask = sz_u64_mask_until_(valid_starts);

        haystack_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, haystack_ptr);

        // Check for anomalies using alarm function
        __mmask64 danger_mask = sz_utf8_case_insensitive_find_ice_georgian_alarm_zmm_(haystack_vec.zmm);
        if (danger_mask) {
            sz_size_t danger_scan_length =
                sz_min_of_two(valid_starts + needle_metadata->offset_in_unfolded, chunk_size);
            sz_cptr_t match = sz_utf8_case_insensitive_find_in_danger_zone_( //
                haystack, haystack_length,                                   //
                needle, needle_length,                                       //
                haystack_ptr, danger_scan_length,                            //
                needle_first_safe_folded_rune,                               //
                needle_metadata->offset_in_unfolded,                         //
                matched_length);
            if (match) return match;
            haystack_ptr += step;
            continue;
        }

        haystack_vec.zmm = sz_utf8_case_insensitive_find_ice_georgian_fold_zmm_(haystack_vec.zmm);

        sz_u64_t matches = _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_first_vec.zmm);
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_second_vec.zmm) >> offset_second;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_third_vec.zmm) >> offset_third;
        matches &= _mm512_cmpeq_epi8_mask(haystack_vec.zmm, probe_last_vec.zmm) >> offset_last;
        matches &= valid_mask;

        // Candidate Verification
        for (; matches; matches &= matches - 1) {
            sz_size_t candidate_offset = sz_u64_ctz(matches);
            sz_cptr_t haystack_candidate_ptr = haystack_ptr + candidate_offset;

            haystack_candidate_vec.xmm = _mm_maskz_loadu_epi8(folded_window_mask, haystack_candidate_ptr);
            haystack_candidate_vec.xmm = _mm512_castsi512_si128(sz_utf8_case_insensitive_find_ice_georgian_fold_zmm_(
                _mm512_castsi128_si512(haystack_candidate_vec.xmm)));

            __mmask16 window_mismatch_mask =
                _mm_mask_cmpneq_epi8_mask(folded_window_mask, haystack_candidate_vec.xmm, needle_window_vec.xmm);
            if (window_mismatch_mask) continue;

            sz_cptr_t match = sz_utf8_case_insensitive_verify_match_(                                      //
                haystack, haystack_length,                                                                 //
                needle, needle_length,                                                                     //
                haystack_candidate_ptr - haystack, needle_metadata->folded_slice_length,                   //
                needle_metadata->offset_in_unfolded,                                                       //
                needle_length - needle_metadata->offset_in_unfolded - needle_metadata->length_in_unfolded, //
                matched_length);
            if (match) {
                sz_utf8_case_insensitive_find_assert_(match, haystack, haystack_length, needle, needle_length,
                                                      needle_metadata);
                return match;
            }
        }
        haystack_ptr += step;
    }

    sz_utf8_case_insensitive_find_assert_(SZ_NULL_CHAR, haystack, haystack_length, needle, needle_length,
                                          needle_metadata);
    return SZ_NULL_CHAR;
}

#pragma endregion // Georgian Case-Insensitive Find

SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_ice( //
    sz_cptr_t haystack, sz_size_t haystack_length,     //
    sz_cptr_t needle, sz_size_t needle_length,         //
    sz_utf8_case_insensitive_needle_metadata_t *needle_metadata, sz_size_t *matched_length) {

    // Handle the obvious edge cases first
    if (needle_length == 0) {
        *matched_length = 0;
        return haystack;
    }

    // If the needle is entirely made of case-less characters - perform direct substring search
    int const is_unknown = needle_metadata->kernel_id == sz_utf8_case_rune_unknown_k;
    int const known_agnostic = needle_metadata->kernel_id == sz_utf8_case_rune_case_invariant_k;
    if (known_agnostic || (is_unknown && sz_utf8_case_invariant_ice(needle, needle_length))) {
        sz_cptr_t result = sz_find(haystack, haystack_length, needle, needle_length);
        if (result) *matched_length = needle_length;
        return result;
    }

    // Analyze needle to find the best safe window for each script
    if (is_unknown) {
        sz_utf8_case_insensitive_needle_metadata_(needle, needle_length, needle_metadata);
        // If no SIMD-safe window found, fall back to serial immediately
        if (needle_metadata->kernel_id == sz_utf8_case_rune_fallback_serial_k)
            return sz_utf8_case_insensitive_find_serial(haystack, haystack_length, needle, needle_length,
                                                        needle_metadata, matched_length);
    }

    // Dispatch to appropriate kernel
    if (needle_metadata->kernel_id == sz_utf8_case_rune_ascii_invariant_k) {
        // Use XOR+VPTERNLOG+VPTESTNMB variants for better port distribution
        if (needle_metadata->folded_slice_length <= 3)
            return sz_utf8_case_insensitive_find_ice_ascii_3probe_( //
                haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
        else
            return sz_utf8_case_insensitive_find_ice_ascii_4probe_( //
                haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);
    }

    if (needle_metadata->kernel_id == sz_utf8_case_rune_safe_western_europe_k)
        return sz_utf8_case_insensitive_find_ice_western_europe_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_case_rune_safe_central_europe_k)
        return sz_utf8_case_insensitive_find_ice_central_europe_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_case_rune_safe_greek_k)
        return sz_utf8_case_insensitive_find_ice_greek_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_case_rune_safe_armenian_k)
        return sz_utf8_case_insensitive_find_ice_armenian_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_case_rune_safe_vietnamese_k)
        return sz_utf8_case_insensitive_find_ice_vietnamese_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_case_rune_safe_cyrillic_k)
        return sz_utf8_case_insensitive_find_ice_cyrillic_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    if (needle_metadata->kernel_id == sz_utf8_case_rune_safe_georgian_k)
        return sz_utf8_case_insensitive_find_ice_georgian_( //
            haystack, haystack_length, needle, needle_length, needle_metadata, matched_length);

    // No suitable SIMD path found (needle has complex Unicode), fall back to serial
    needle_metadata->kernel_id = sz_utf8_case_rune_fallback_serial_k;
    return sz_utf8_case_insensitive_find_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                                matched_length);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#endif // SZ_USE_ICE

#pragma endregion // Shared Helpers Case-Insensitive Find
#pragma endregion // Ice Lake Implementation

/**
 *  NEON kernels for Unicode case-folding and case-insensitive search.
 *
 *  On modern Arm CPUs (Apple M-series, Neoverse N1/N2/V1/V2), instructions like `vextq_u8` (extract/shift),
 *  `vbslq_u8` (bitwise select), and `vandq_u8` (bitwise AND) are extremely cheap, often with 1-2 cycle latencies
 *  and high throughput (0.25-0.5 CPI). `vld1q_u8_x4` efficiently loads 64 bytes into 4 registers, decomposing
 *  into 4 `LDR Q` operations with high throughput. `vqtbl1q_u8` (table lookup) is relatively fast (2 cycles) but
 *  uses limited shuffle ports, so it should be used judiciously. `vcgtq_u8` (compare greater) is also efficient
 *  for range checks.
 *
 *  | Instr         | Latency | CPI  | Notes                                |
 *  | :------------ | :------ | :--- | :----------------------------------- |
 *  | `vld1q_u8_x4` | ~5-7    | ~0.5 | Decomposes to 4 LDR Q. High throughput. |
 *  | `vextq_u8`    | 1-2     | 0.5  | Cheap. Ideal for Shift Chains.       |
 *  | `vqtbl1q_u8`  | 2       | 1    | Uses limited Shuffle ports.          |
 *  | `vbslq_u8`    | 1       | 0.25 | Free (4/cycle on Apple/Neoverse).    |
 *  | `vandq_u8`    | 1       | 0.25 | Free (4/cycle on Apple/Neoverse).    |
 *  | `vcgtq_u8`    | 1-2     | ~0.5 | Efficient for range checks.          |
 *
 *  To maximize performance, process data in 64-byte blocks (4x 16-byte NEON registers) per iteration to saturate
 *  execution units. Use `vld1q_u8_x4` for sequential loads instead of slower interleaved `ld4` variants. For multi-byte
 *  sequences spanning register boundaries, create "shifted" views using `vextq_u8` to handle dependencies entirely
 *  within SIMD.
 *
 *  For case-folding, prefer `vaddq_u8` with masks from comparisons and bitwise logic for simple ranges like ASCII.
 *  Use `vqtbl1q_u8` for complex mappings, and `vbslq_u8` for conditional updates without branches. Detect "danger
 * zones" (multi-rune expansions or length changes) using bitwise masks; if a block contains any, fall back to a robust
 * serial handler for that specific region.
 *
 *  For match detection, compare processed registers against the needle using `vceqq_u8`, convert the 16-bit match
 *  masks to 64-bit using the `sz_find_vreinterpretq_u8_u4_` pattern, combine them with bitwise OR, and use `ctz`
 *  to find the first match offset. This balances NEON's high ALU throughput with its lack of a direct `movemask`
 *  instruction.
 */

#pragma region NEON Implementation
#if SZ_USE_NEON

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_neon(  //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {
    return sz_utf8_unpack_chunk_serial(text, length, runes, runes_capacity, runes_unpacked);
}

SZ_PUBLIC sz_size_t sz_utf8_case_fold_neon(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {
    return sz_utf8_case_fold_serial(source, source_length, destination);
}

SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_neon( //
    sz_cptr_t haystack, sz_size_t haystack_length,      //
    sz_cptr_t needle, sz_size_t needle_length,          //
    sz_utf8_case_insensitive_needle_metadata_t *needle_metadata, sz_size_t *matched_length) {
    return sz_utf8_case_insensitive_find_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                                matched_length);
}

SZ_PUBLIC sz_bool_t sz_utf8_case_invariant_neon(sz_cptr_t str, sz_size_t length) {
    return sz_utf8_case_invariant_serial(str, length);
}

#endif            // SZ_USE_NEON
#pragma endregion // NEON Implementation

#pragma region Dynamic Dispatch

#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_cptr_t sz_utf8_unpack_chunk(sz_cptr_t text, sz_size_t length, sz_rune_t *runes, sz_size_t runes_capacity,
                                          sz_size_t *runes_unpacked) {
#if SZ_USE_ICE
    return sz_utf8_unpack_chunk_ice(text, length, runes, runes_capacity, runes_unpacked);
#else
    return sz_utf8_unpack_chunk_serial(text, length, runes, runes_capacity, runes_unpacked);
#endif
}

SZ_DYNAMIC sz_size_t sz_utf8_case_fold(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {
#if SZ_USE_ICE
    return sz_utf8_case_fold_ice(source, source_length, destination);
#else
    return sz_utf8_case_fold_serial(source, source_length, destination);
#endif
}

SZ_DYNAMIC sz_size_t sz_utf8_case_upper(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {
#if SZ_USE_ICE
    return sz_utf8_case_upper_ice(source, source_length, destination);
#else
    return sz_utf8_case_upper_serial(source, source_length, destination);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_utf8_case_insensitive_find(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                   sz_size_t needle_length,
                                                   sz_utf8_case_insensitive_needle_metadata_t *needle_metadata,
                                                   sz_size_t *matched_length) {
#if SZ_USE_ICE
    return sz_utf8_case_insensitive_find_ice(haystack, haystack_length, needle, needle_length, needle_metadata,
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
#if SZ_USE_ICE
    return sz_utf8_case_invariant_ice(str, length);
#else
    return sz_utf8_case_invariant_serial(str, length);
#endif
}

#endif // !SZ_DYNAMIC_DISPATCH

#pragma endregion // Dynamic Dispatch

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_CASE_H_
