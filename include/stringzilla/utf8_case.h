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
 *
 *  It's important to remember that UTF-8 is just one of many possible Unicode encodings.
 *  Unicode is a versioned standard and we implement its locale-independent specification v17.
 *  All algorithms are fully compliant with the specification and handle all edge cases.
 *
 *  On fast vectorized paths, unlike other parts of StringZilla, there may be significant algorithmic
 *  differences between different ISA versions. Most of them are designed to be practical in common
 *  use cases, targeting the most common languages on the Internet.
 *
 *   Rank  Language    Script      UTF-8 Bytes  Has Case?  Case Folding Notes
 *  ------------------------------------------------------------------------------------------
 *   1     English     Latin       1            Yes        Simple +32 offset (A-Z)
 *   2     Russian     Cyrillic    2            Yes        Simple +32 offset (А-Я)
 *   3     Spanish     Latin       1-2          Yes        Mostly ASCII, few 2-byte (ñ, á, é)
 *   4     German      Latin       1-2          Yes        ASCII + ß→ss expansion, ä/ö/ü
 *   5     French      Latin       1-2          Yes        Mostly ASCII, accents (é, è, ç)
 *   6     Japanese    CJK + Kana  3            No*        No case, but has fullwidth A-Z in addresses
 *   7     Portuguese  Latin       1-2          Yes        Like Spanish
 *   8     Chinese     CJK         3            No         No case folding
 *   9     Italian     Latin       1-2          Yes        Like Spanish
 *   10    Polish      Latin       1-2          Yes        ASCII + ą, ę, ł, ż, etc.
 *   11    Turkish     Latin       1-2          Yes        İ/ı special handling, that we don't do
 *   12    Dutch       Latin       1            Yes        Almost pure ASCII
 *   13    Persian     Arabic      2            No         RTL, no case
 *   14    Vietnamese  Latin       2-3          Yes        Heavy diacritics (ă, ơ, ư), odd/even
 *   15    Korean      Hangul      3            No         No case folding
 *   16    Arabic      Arabic      2            No         RTL, no case
 *   17    Indonesian  Latin       1            Yes        Pure ASCII
 *   18    Greek       Greek       2            Yes        +32 offset, σ/ς handling
 *   19    Ukrainian   Cyrillic    2            Yes        Like Russian
 *   20    Czech       Latin       1-2          Yes        ASCII + ě, š, č, ř, ž
 *
 *  This doesn't, however, cover many other relevant subranges of Unicode.
 */
#ifndef STRINGZILLA_UTF8_CASE_H_
#define STRINGZILLA_UTF8_CASE_H_

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

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
 *  Worst-case example: U+0390 (2 bytes: CE 90) expands to U+03B9 + U+0308 + U+0301 (6 bytes total).
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
 *      ß  (U+00DF) → "ss"  (U+0073 U+0073)
 *      ẞ  (U+1E9E) → "ss"
 *    as well as mixed-case digraphs and trigraphs normalized to lowercase sequences.
 *  - Turkic dotted/dotless-I characters are handled per Unicode Case Folding (not locale-specific):
 *      İ (U+0130)  → "i̇"   (U+0069 U+0307) — Full case folding with combining dot
 *      I (U+0049)  →  i    (U+0069)        — Standard folding (not Turkic I→ı)
 *      ı (U+0131)  →  ı    (already lowercase, unchanged)
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
 *  because one-to-many expansions (e.g., U+00DF -> "ss") break core assumptions of fast string algorithms:
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
 *  @param[in] needle_length Number of bytes in the needle substring.
 *  @param[out] matched_length Number of bytes in the matched region.
 *  @return Pointer to the first matching substring from @p haystack, or @c SZ_NULL_CHAR if not found.
 */
SZ_DYNAMIC sz_cptr_t sz_utf8_case_insensitive_find( //
    sz_cptr_t haystack, sz_size_t haystack_length,  //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length);

/**
 *  @brief  Case-insensitive lexicographic comparison of two UTF-8 strings.
 *
 *  Compares strings using Unicode case folding rules, producing consistent ordering regardless of
 *  letter case. Implements the same full Unicode Case Folding as `sz_utf8_case_fold`, including
 *  all one-to-many expansions (e.g., ß → ss) and bicameral script mappings.
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
 *      // result == sz_equal_k (ß folds to ss)
 *  @endcode
 */
SZ_DYNAMIC sz_ordering_t sz_utf8_case_insensitive_order(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                        sz_size_t b_length);

#pragma endregion

#pragma region Platform-Specific Backends

/** @copydoc sz_utf8_case_fold */
SZ_PUBLIC sz_size_t sz_utf8_case_fold_serial(  //
    sz_cptr_t source, sz_size_t source_length, //
    sz_ptr_t destination);

/** @copydoc sz_utf8_case_insensitive_find */
SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_serial( //
    sz_cptr_t haystack, sz_size_t haystack_length,        //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length);

/** @copydoc sz_utf8_case_insensitive_order */
SZ_PUBLIC sz_ordering_t sz_utf8_case_insensitive_order_serial( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);

/** @copydoc sz_utf8_case_fold */
SZ_PUBLIC sz_size_t sz_utf8_case_fold_ice( //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination);

/** @copydoc sz_utf8_case_insensitive_find */
SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_ice( //
    sz_cptr_t haystack, sz_size_t haystack_length,     //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length);

/** @copydoc sz_utf8_case_insensitive_order */
SZ_PUBLIC sz_ordering_t sz_utf8_case_insensitive_order_ice( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);

/** @copydoc sz_utf8_case_fold */
SZ_PUBLIC sz_size_t sz_utf8_case_fold_neon( //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination);

/** @copydoc sz_utf8_case_insensitive_find */
SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_neon( //
    sz_cptr_t haystack, sz_size_t haystack_length,      //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length);

/** @copydoc sz_utf8_case_insensitive_order */
SZ_PUBLIC sz_ordering_t sz_utf8_case_insensitive_order_neon( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);

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
            if (rune == 0x00DF) { folded[0] = 0x0073; folded[1] = 0x0073; return 2; } // ß → ss
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
        case 0x00B5: folded[0] = 0x03BC; return 1; // µ → μ (micro sign to Greek mu)
        case 0x0178: folded[0] = 0x00FF; return 1; // Ÿ → ÿ
        case 0x017F: folded[0] = 0x0073; return 1; // ſ → s (long s)
        // Latin Extended-B: African/IPA letters (0x0181-0x01BF)
        case 0x0181: folded[0] = 0x0253; return 1; // Ɓ → ɓ
        case 0x0182: folded[0] = 0x0183; return 1; // Ƃ → ƃ
        case 0x0184: folded[0] = 0x0185; return 1; // Ƅ → ƅ
        case 0x0186: folded[0] = 0x0254; return 1; // Ɔ → ɔ
        case 0x0187: folded[0] = 0x0188; return 1; // Ƈ → ƈ
        case 0x0189: folded[0] = 0x0256; return 1; // Ɖ → ɖ
        case 0x018A: folded[0] = 0x0257; return 1; // Ɗ → ɗ
        case 0x018B: folded[0] = 0x018C; return 1; // Ƌ → ƌ
        case 0x018E: folded[0] = 0x01DD; return 1; // Ǝ → ǝ
        case 0x018F: folded[0] = 0x0259; return 1; // Ə → ə (schwa)
        case 0x0190: folded[0] = 0x025B; return 1; // Ɛ → ɛ
        case 0x0191: folded[0] = 0x0192; return 1; // Ƒ → ƒ
        case 0x0193: folded[0] = 0x0260; return 1; // Ɠ → ɠ
        case 0x0194: folded[0] = 0x0263; return 1; // Ɣ → ɣ
        case 0x0196: folded[0] = 0x0269; return 1; // Ɩ → ɩ
        case 0x0197: folded[0] = 0x0268; return 1; // Ɨ → ɨ
        case 0x0198: folded[0] = 0x0199; return 1; // Ƙ → ƙ
        case 0x019C: folded[0] = 0x026F; return 1; // Ɯ → ɯ
        case 0x019D: folded[0] = 0x0272; return 1; // Ɲ → ɲ
        case 0x019F: folded[0] = 0x0275; return 1; // Ɵ → ɵ
        case 0x01A0: folded[0] = 0x01A1; return 1; // Ơ → ơ (Vietnamese)
        case 0x01A2: folded[0] = 0x01A3; return 1; // Ƣ → ƣ
        case 0x01A4: folded[0] = 0x01A5; return 1; // Ƥ → ƥ
        case 0x01A6: folded[0] = 0x0280; return 1; // Ʀ → ʀ
        case 0x01A7: folded[0] = 0x01A8; return 1; // Ƨ → ƨ
        case 0x01A9: folded[0] = 0x0283; return 1; // Ʃ → ʃ
        case 0x01AC: folded[0] = 0x01AD; return 1; // Ƭ → ƭ
        case 0x01AE: folded[0] = 0x0288; return 1; // Ʈ → ʈ
        case 0x01AF: folded[0] = 0x01B0; return 1; // Ư → ư (Vietnamese)
        case 0x01B1: folded[0] = 0x028A; return 1; // Ʊ → ʊ
        case 0x01B2: folded[0] = 0x028B; return 1; // Ʋ → ʋ
        case 0x01B3: folded[0] = 0x01B4; return 1; // Ƴ → ƴ
        case 0x01B5: folded[0] = 0x01B6; return 1; // Ƶ → ƶ
        case 0x01B7: folded[0] = 0x0292; return 1; // Ʒ → ʒ
        case 0x01B8: folded[0] = 0x01B9; return 1; // Ƹ → ƹ
        case 0x01BC: folded[0] = 0x01BD; return 1; // Ƽ → ƽ
        // Digraphs: Serbian/Croatian DŽ, LJ, NJ and DZ
        case 0x01C4: folded[0] = 0x01C6; return 1; // Ǆ → ǆ
        case 0x01C5: folded[0] = 0x01C6; return 1; // ǅ → ǆ (titlecase)
        case 0x01C7: folded[0] = 0x01C9; return 1; // Ǉ → ǉ
        case 0x01C8: folded[0] = 0x01C9; return 1; // ǈ → ǉ (titlecase)
        case 0x01CA: folded[0] = 0x01CC; return 1; // Ǌ → ǌ
        case 0x01CB: folded[0] = 0x01CC; return 1; // ǋ → ǌ (titlecase)
        case 0x01F1: folded[0] = 0x01F3; return 1; // Ǳ → ǳ
        case 0x01F2: folded[0] = 0x01F3; return 1; // ǲ → ǳ (titlecase)
        // Latin Extended-B: isolated irregulars
        case 0x01F4: folded[0] = 0x01F5; return 1; // Ǵ → ǵ
        case 0x01F6: folded[0] = 0x0195; return 1; // Ƕ → ƕ (hwair)
        case 0x01F7: folded[0] = 0x01BF; return 1; // Ƿ → ƿ (wynn)
        case 0x0220: folded[0] = 0x019E; return 1; // Ƞ → ƞ
        case 0x023A: folded[0] = 0x2C65; return 1; // Ⱥ → ⱥ
        case 0x023B: folded[0] = 0x023C; return 1; // Ȼ → ȼ
        case 0x023D: folded[0] = 0x019A; return 1; // Ƚ → ƚ
        case 0x023E: folded[0] = 0x2C66; return 1; // Ⱦ → ⱦ
        case 0x0241: folded[0] = 0x0242; return 1; // Ɂ → ɂ
        case 0x0243: folded[0] = 0x0180; return 1; // Ƀ → ƀ
        case 0x0244: folded[0] = 0x0289; return 1; // Ʉ → ʉ
        case 0x0245: folded[0] = 0x028C; return 1; // Ʌ → ʌ
        // Greek: combining iota, accented vowels, variant forms
        case 0x0345: folded[0] = 0x03B9; return 1; // ͅ → ι (combining iota subscript)
        case 0x0376: folded[0] = 0x0377; return 1; // Ͷ → ͷ
        case 0x037F: folded[0] = 0x03F3; return 1; // Ϳ → ϳ
        case 0x0386: folded[0] = 0x03AC; return 1; // Ά → ά
        case 0x038C: folded[0] = 0x03CC; return 1; // Ό → ό
        case 0x038E: folded[0] = 0x03CD; return 1; // Ύ → ύ
        case 0x038F: folded[0] = 0x03CE; return 1; // Ώ → ώ
        case 0x03C2: folded[0] = 0x03C3; return 1; // ς → σ (final sigma)
        case 0x03CF: folded[0] = 0x03D7; return 1; // Ϗ → ϗ
        case 0x03D0: folded[0] = 0x03B2; return 1; // ϐ → β (beta symbol)
        case 0x03D1: folded[0] = 0x03B8; return 1; // ϑ → θ (theta symbol)
        case 0x03D5: folded[0] = 0x03C6; return 1; // ϕ → φ (phi symbol)
        case 0x03D6: folded[0] = 0x03C0; return 1; // ϖ → π (pi symbol)
        case 0x03F0: folded[0] = 0x03BA; return 1; // ϰ → κ (kappa symbol)
        case 0x03F1: folded[0] = 0x03C1; return 1; // ϱ → ρ (rho symbol)
        case 0x03F4: folded[0] = 0x03B8; return 1; // ϴ → θ
        case 0x03F5: folded[0] = 0x03B5; return 1; // ϵ → ε (lunate epsilon)
        case 0x03F7: folded[0] = 0x03F8; return 1; // Ϸ → ϸ
        case 0x03F9: folded[0] = 0x03F2; return 1; // Ϲ → ϲ
        case 0x03FA: folded[0] = 0x03FB; return 1; // Ϻ → ϻ
        // Cyrillic: palochka
        case 0x04C0: folded[0] = 0x04CF; return 1; // Ӏ → ӏ
        }

        // 2-byte one-to-many expansions
        switch (rune) {
        // ß handled inline in Latin-1 range above... interestingly the capital Eszett is in the 3-byte range!
        // case 0x00DF: folded[0] = 0x0073; folded[1] = 0x0073; return 2; // ß → ss (German)
        case 0x0130: folded[0] = 0x0069; folded[1] = 0x0307; return 2; // İ → i + combining dot (Turkish)
        case 0x0149: folded[0] = 0x02BC; folded[1] = 0x006E; return 2; // ŉ → ʼn (Afrikaans)
        case 0x01F0: folded[0] = 0x006A; folded[1] = 0x030C; return 2; // ǰ → j + combining caron
        case 0x0390: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0301; return 3; // ΐ → ι + 2 combining
        case 0x03B0: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0301; return 3; // ΰ → υ + 2 combining
        case 0x0587: folded[0] = 0x0565; folded[1] = 0x0582; return 2; // և → եւ (Armenian)
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
        case 0x10C7: folded[0] = 0x2D27; return 1; // Ⴧ → ⴧ
        case 0x10CD: folded[0] = 0x2D2D; return 1; // Ⴭ → ⴭ
        // Cyrillic Extended-C: Old Slavonic variant forms
        case 0x1C80: folded[0] = 0x0432; return 1; // ᲀ → в
        case 0x1C81: folded[0] = 0x0434; return 1; // ᲁ → д
        case 0x1C82: folded[0] = 0x043E; return 1; // ᲂ → о
        case 0x1C83: folded[0] = 0x0441; return 1; // ᲃ → с
        case 0x1C84: folded[0] = 0x0442; return 1; // ᲄ → т
        case 0x1C85: folded[0] = 0x0442; return 1; // ᲅ → т
        case 0x1C86: folded[0] = 0x044A; return 1; // ᲆ → ъ
        case 0x1C87: folded[0] = 0x0463; return 1; // ᲇ → ѣ
        case 0x1C88: folded[0] = 0xA64B; return 1; // ᲈ → ꙋ
        case 0x1C89: folded[0] = 0x1C8A; return 1; // Ᲊ → ᲊ
        // Latin Extended Additional: long s with dot
        case 0x1E9B: folded[0] = 0x1E61; return 1; // ẛ → ṡ
        // Greek Extended: vowels with breathing marks (irregular offsets)
        case 0x1F59: folded[0] = 0x1F51; return 1; // Ὑ → ὑ
        case 0x1F5B: folded[0] = 0x1F53; return 1; // Ὓ → ὓ
        case 0x1F5D: folded[0] = 0x1F55; return 1; // Ὕ → ὕ
        case 0x1F5F: folded[0] = 0x1F57; return 1; // Ὗ → ὗ
        case 0x1FB8: folded[0] = 0x1FB0; return 1; // Ᾰ → ᾰ
        case 0x1FB9: folded[0] = 0x1FB1; return 1; // Ᾱ → ᾱ
        case 0x1FBA: folded[0] = 0x1F70; return 1; // Ὰ → ὰ
        case 0x1FBB: folded[0] = 0x1F71; return 1; // Ά → ά
        case 0x1FBE: folded[0] = 0x03B9; return 1; // ι → ι
        case 0x1FD8: folded[0] = 0x1FD0; return 1; // Ῐ → ῐ
        case 0x1FD9: folded[0] = 0x1FD1; return 1; // Ῑ → ῑ
        case 0x1FDA: folded[0] = 0x1F76; return 1; // Ὶ → ὶ
        case 0x1FDB: folded[0] = 0x1F77; return 1; // Ί → ί
        case 0x1FE8: folded[0] = 0x1FE0; return 1; // Ῠ → ῠ
        case 0x1FE9: folded[0] = 0x1FE1; return 1; // Ῡ → ῡ
        case 0x1FEA: folded[0] = 0x1F7A; return 1; // Ὺ → ὺ
        case 0x1FEB: folded[0] = 0x1F7B; return 1; // Ύ → ύ
        case 0x1FEC: folded[0] = 0x1FE5; return 1; // Ῥ → ῥ
        case 0x1FF8: folded[0] = 0x1F78; return 1; // Ὸ → ὸ
        case 0x1FF9: folded[0] = 0x1F79; return 1; // Ό → ό
        case 0x1FFA: folded[0] = 0x1F7C; return 1; // Ὼ → ὼ
        case 0x1FFB: folded[0] = 0x1F7D; return 1; // Ώ → ώ
        // Letterlike Symbols: compatibility mappings
        case 0x2126: folded[0] = 0x03C9; return 1; // Ω → ω
        case 0x212A: folded[0] = 0x006B; return 1; // K → k
        case 0x212B: folded[0] = 0x00E5; return 1; // Å → å
        case 0x2132: folded[0] = 0x214E; return 1; // Ⅎ → ⅎ
        case 0x2183: folded[0] = 0x2184; return 1; // Ↄ → ↄ
        // Latin Extended-C: irregular mappings to IPA/other blocks
        case 0x2C60: folded[0] = 0x2C61; return 1; // Ⱡ → ⱡ
        case 0x2C62: folded[0] = 0x026B; return 1; // Ɫ → ɫ
        case 0x2C63: folded[0] = 0x1D7D; return 1; // Ᵽ → ᵽ
        case 0x2C64: folded[0] = 0x027D; return 1; // Ɽ → ɽ
        case 0x2C67: folded[0] = 0x2C68; return 1; // Ⱨ → ⱨ
        case 0x2C69: folded[0] = 0x2C6A; return 1; // Ⱪ → ⱪ
        case 0x2C6B: folded[0] = 0x2C6C; return 1; // Ⱬ → ⱬ
        case 0x2C6D: folded[0] = 0x0251; return 1; // Ɑ → ɑ
        case 0x2C6E: folded[0] = 0x0271; return 1; // Ɱ → ɱ
        case 0x2C6F: folded[0] = 0x0250; return 1; // Ɐ → ɐ
        case 0x2C70: folded[0] = 0x0252; return 1; // Ɒ → ɒ
        case 0x2C72: folded[0] = 0x2C73; return 1; // Ⱳ → ⱳ
        case 0x2C75: folded[0] = 0x2C76; return 1; // Ⱶ → ⱶ
        case 0x2C7E: folded[0] = 0x023F; return 1; // Ȿ → ȿ
        case 0x2C7F: folded[0] = 0x0240; return 1; // Ɀ → ɀ
        // Coptic: irregular cases outside even/odd range
        case 0x2CEB: folded[0] = 0x2CEC; return 1; // Ⳬ → ⳬ
        case 0x2CED: folded[0] = 0x2CEE; return 1; // Ⳮ → ⳮ
        case 0x2CF2: folded[0] = 0x2CF3; return 1; // Ⳳ → ⳳ
        // Latin Extended-D: isolated irregulars
        case 0xA779: folded[0] = 0xA77A; return 1; // Ꝺ → ꝺ
        case 0xA77B: folded[0] = 0xA77C; return 1; // Ꝼ → ꝼ
        case 0xA77D: folded[0] = 0x1D79; return 1; // Ᵹ → ᵹ
        case 0xA78B: folded[0] = 0xA78C; return 1; // Ꞌ → ꞌ
        case 0xA78D: folded[0] = 0x0265; return 1; // Ɥ → ɥ
        case 0xA7AA: folded[0] = 0x0266; return 1; // Ɦ → ɦ
        case 0xA7AB: folded[0] = 0x025C; return 1; // Ɜ → ɜ
        case 0xA7AC: folded[0] = 0x0261; return 1; // Ɡ → ɡ
        case 0xA7AD: folded[0] = 0x026C; return 1; // Ɬ → ɬ
        case 0xA7AE: folded[0] = 0x026A; return 1; // Ɪ → ɪ
        case 0xA7B0: folded[0] = 0x029E; return 1; // Ʞ → ʞ
        case 0xA7B1: folded[0] = 0x0287; return 1; // Ʇ → ʇ
        case 0xA7B2: folded[0] = 0x029D; return 1; // Ʝ → ʝ
        case 0xA7B3: folded[0] = 0xAB53; return 1; // Ꭓ → ꭓ
        case 0xA7C4: folded[0] = 0xA794; return 1; // Ꞔ → ꞔ
        case 0xA7C5: folded[0] = 0x0282; return 1; // Ʂ → ʂ
        case 0xA7C6: folded[0] = 0x1D8E; return 1; // Ᶎ → ᶎ
        case 0xA7C7: folded[0] = 0xA7C8; return 1; // Ꟈ → ꟈ
        case 0xA7C9: folded[0] = 0xA7CA; return 1; // Ꟊ → ꟊ
        case 0xA7CB: folded[0] = 0x0264; return 1; // Ɤ → ɤ
        case 0xA7CC: folded[0] = 0xA7CD; return 1; // Ꟍ → ꟍ
        case 0xA7CE: folded[0] = 0xA7CF; return 1; // ꟏ → ꟏
        case 0xA7D0: folded[0] = 0xA7D1; return 1; // Ꟑ → ꟑ
        case 0xA7D2: folded[0] = 0xA7D3; return 1; // (placeholder)
        case 0xA7D4: folded[0] = 0xA7D5; return 1; // (placeholder)
        case 0xA7D6: folded[0] = 0xA7D7; return 1; // Ꟗ → ꟗ
        case 0xA7D8: folded[0] = 0xA7D9; return 1; // Ꟙ → ꟙ
        case 0xA7DA: folded[0] = 0xA7DB; return 1; // Ꟛ → ꟛ
        case 0xA7DC: folded[0] = 0x019B; return 1; // Ƛ → ƛ
        case 0xA7F5: folded[0] = 0xA7F6; return 1; // Ꟶ → ꟶ
        }

        // Next let's handle the 3-byte one-to-many expansions
        switch (rune) {
        // Latin Extended Additional
        case 0x1E96: folded[0] = 0x0068; folded[1] = 0x0331; return 2; // ẖ → h + combining
        case 0x1E97: folded[0] = 0x0074; folded[1] = 0x0308; return 2; // ẗ → t + combining
        case 0x1E98: folded[0] = 0x0077; folded[1] = 0x030A; return 2; // ẘ → w + combining
        case 0x1E99: folded[0] = 0x0079; folded[1] = 0x030A; return 2; // ẙ → y + combining
        case 0x1E9A: folded[0] = 0x0061; folded[1] = 0x02BE; return 2; // ẚ → aʾ
        case 0x1E9E: folded[0] = 0x0073; folded[1] = 0x0073; return 2; // ẞ → ss (German capital Eszett)
        // Greek Extended: breathing marks
        case 0x1F50: folded[0] = 0x03C5; folded[1] = 0x0313; return 2; // ὐ → υ + combining
        case 0x1F52: folded[0] = 0x03C5; folded[1] = 0x0313; folded[2] = 0x0300; return 3; // ὒ → υ + 2 combining
        case 0x1F54: folded[0] = 0x03C5; folded[1] = 0x0313; folded[2] = 0x0301; return 3; // ὔ → υ + 2 combining
        case 0x1F56: folded[0] = 0x03C5; folded[1] = 0x0313; folded[2] = 0x0342; return 3; // ὖ → υ + 2 combining
        // Greek Extended: iota subscript combinations (0x1F80-0x1FAF)
        case 0x1F80: folded[0] = 0x1F00; folded[1] = 0x03B9; return 2; // ᾀ → ἀι
        case 0x1F81: folded[0] = 0x1F01; folded[1] = 0x03B9; return 2; // ᾁ → ἁι
        case 0x1F82: folded[0] = 0x1F02; folded[1] = 0x03B9; return 2; // ᾂ → ἂι
        case 0x1F83: folded[0] = 0x1F03; folded[1] = 0x03B9; return 2; // ᾃ → ἃι
        case 0x1F84: folded[0] = 0x1F04; folded[1] = 0x03B9; return 2; // ᾄ → ἄι
        case 0x1F85: folded[0] = 0x1F05; folded[1] = 0x03B9; return 2; // ᾅ → ἅι
        case 0x1F86: folded[0] = 0x1F06; folded[1] = 0x03B9; return 2; // ᾆ → ἆι
        case 0x1F87: folded[0] = 0x1F07; folded[1] = 0x03B9; return 2; // ᾇ → ἇι
        case 0x1F88: folded[0] = 0x1F00; folded[1] = 0x03B9; return 2; // ᾈ → ἀι
        case 0x1F89: folded[0] = 0x1F01; folded[1] = 0x03B9; return 2; // ᾉ → ἁι
        case 0x1F8A: folded[0] = 0x1F02; folded[1] = 0x03B9; return 2; // ᾊ → ἂι
        case 0x1F8B: folded[0] = 0x1F03; folded[1] = 0x03B9; return 2; // ᾋ → ἃι
        case 0x1F8C: folded[0] = 0x1F04; folded[1] = 0x03B9; return 2; // ᾌ → ἄι
        case 0x1F8D: folded[0] = 0x1F05; folded[1] = 0x03B9; return 2; // ᾍ → ἅι
        case 0x1F8E: folded[0] = 0x1F06; folded[1] = 0x03B9; return 2; // ᾎ → ἆι
        case 0x1F8F: folded[0] = 0x1F07; folded[1] = 0x03B9; return 2; // ᾏ → ἇι
        case 0x1F90: folded[0] = 0x1F20; folded[1] = 0x03B9; return 2; // ᾐ → ἠι
        case 0x1F91: folded[0] = 0x1F21; folded[1] = 0x03B9; return 2; // ᾑ → ἡι
        case 0x1F92: folded[0] = 0x1F22; folded[1] = 0x03B9; return 2; // ᾒ → ἢι
        case 0x1F93: folded[0] = 0x1F23; folded[1] = 0x03B9; return 2; // ᾓ → ἣι
        case 0x1F94: folded[0] = 0x1F24; folded[1] = 0x03B9; return 2; // ᾔ → ἤι
        case 0x1F95: folded[0] = 0x1F25; folded[1] = 0x03B9; return 2; // ᾕ → ἥι
        case 0x1F96: folded[0] = 0x1F26; folded[1] = 0x03B9; return 2; // ᾖ → ἦι
        case 0x1F97: folded[0] = 0x1F27; folded[1] = 0x03B9; return 2; // ᾗ → ἧι
        case 0x1F98: folded[0] = 0x1F20; folded[1] = 0x03B9; return 2; // ᾘ → ἠι
        case 0x1F99: folded[0] = 0x1F21; folded[1] = 0x03B9; return 2; // ᾙ → ἡι
        case 0x1F9A: folded[0] = 0x1F22; folded[1] = 0x03B9; return 2; // ᾚ → ἢι
        case 0x1F9B: folded[0] = 0x1F23; folded[1] = 0x03B9; return 2; // ᾛ → ἣι
        case 0x1F9C: folded[0] = 0x1F24; folded[1] = 0x03B9; return 2; // ᾜ → ἤι
        case 0x1F9D: folded[0] = 0x1F25; folded[1] = 0x03B9; return 2; // ᾝ → ἥι
        case 0x1F9E: folded[0] = 0x1F26; folded[1] = 0x03B9; return 2; // ᾞ → ἦι
        case 0x1F9F: folded[0] = 0x1F27; folded[1] = 0x03B9; return 2; // ᾟ → ἧι
        case 0x1FA0: folded[0] = 0x1F60; folded[1] = 0x03B9; return 2; // ᾠ → ὠι
        case 0x1FA1: folded[0] = 0x1F61; folded[1] = 0x03B9; return 2; // ᾡ → ὡι
        case 0x1FA2: folded[0] = 0x1F62; folded[1] = 0x03B9; return 2; // ᾢ → ὢι
        case 0x1FA3: folded[0] = 0x1F63; folded[1] = 0x03B9; return 2; // ᾣ → ὣι
        case 0x1FA4: folded[0] = 0x1F64; folded[1] = 0x03B9; return 2; // ᾤ → ὤι
        case 0x1FA5: folded[0] = 0x1F65; folded[1] = 0x03B9; return 2; // ᾥ → ὥι
        case 0x1FA6: folded[0] = 0x1F66; folded[1] = 0x03B9; return 2; // ᾦ → ὦι
        case 0x1FA7: folded[0] = 0x1F67; folded[1] = 0x03B9; return 2; // ᾧ → ὧι
        case 0x1FA8: folded[0] = 0x1F60; folded[1] = 0x03B9; return 2; // ᾨ → ὠι
        case 0x1FA9: folded[0] = 0x1F61; folded[1] = 0x03B9; return 2; // ᾩ → ὡι
        case 0x1FAA: folded[0] = 0x1F62; folded[1] = 0x03B9; return 2; // ᾪ → ὢι
        case 0x1FAB: folded[0] = 0x1F63; folded[1] = 0x03B9; return 2; // ᾫ → ὣι
        case 0x1FAC: folded[0] = 0x1F64; folded[1] = 0x03B9; return 2; // ᾬ → ὤι
        case 0x1FAD: folded[0] = 0x1F65; folded[1] = 0x03B9; return 2; // ᾭ → ὥι
        case 0x1FAE: folded[0] = 0x1F66; folded[1] = 0x03B9; return 2; // ᾮ → ὦι
        case 0x1FAF: folded[0] = 0x1F67; folded[1] = 0x03B9; return 2; // ᾯ → ὧι
        // Greek Extended: vowel + iota subscript (0x1FB2-0x1FFC)
        case 0x1FB2: folded[0] = 0x1F70; folded[1] = 0x03B9; return 2; // ᾲ → ὰι
        case 0x1FB3: folded[0] = 0x03B1; folded[1] = 0x03B9; return 2; // ᾳ → αι
        case 0x1FB4: folded[0] = 0x03AC; folded[1] = 0x03B9; return 2; // ᾴ → άι
        case 0x1FB6: folded[0] = 0x03B1; folded[1] = 0x0342; return 2; // ᾶ → α + combining
        case 0x1FB7: folded[0] = 0x03B1; folded[1] = 0x0342; folded[2] = 0x03B9; return 3; // ᾷ → α + 2 combining
        case 0x1FBC: folded[0] = 0x03B1; folded[1] = 0x03B9; return 2; // ᾼ → αι
        case 0x1FC2: folded[0] = 0x1F74; folded[1] = 0x03B9; return 2; // ῂ → ὴι
        case 0x1FC3: folded[0] = 0x03B7; folded[1] = 0x03B9; return 2; // ῃ → ηι
        case 0x1FC4: folded[0] = 0x03AE; folded[1] = 0x03B9; return 2; // ῄ → ήι
        case 0x1FC6: folded[0] = 0x03B7; folded[1] = 0x0342; return 2; // ῆ → η + combining
        case 0x1FC7: folded[0] = 0x03B7; folded[1] = 0x0342; folded[2] = 0x03B9; return 3; // ῇ → η + 2 combining
        case 0x1FCC: folded[0] = 0x03B7; folded[1] = 0x03B9; return 2; // ῌ → ηι
        case 0x1FD2: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0300; return 3; // ῒ → ι + 2 combining
        case 0x1FD3: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0301; return 3; // ΐ → ι + 2 combining
        case 0x1FD6: folded[0] = 0x03B9; folded[1] = 0x0342; return 2; // ῖ → ι + combining
        case 0x1FD7: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0342; return 3; // ῗ → ι + 2 combining
        case 0x1FE2: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0300; return 3; // ῢ → υ + 2 combining
        case 0x1FE3: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0301; return 3; // ΰ → υ + 2 combining
        case 0x1FE4: folded[0] = 0x03C1; folded[1] = 0x0313; return 2; // ῤ → ρ + combining
        case 0x1FE6: folded[0] = 0x03C5; folded[1] = 0x0342; return 2; // ῦ → υ + combining
        case 0x1FE7: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0342; return 3; // ῧ → υ + 2 combining
        case 0x1FF2: folded[0] = 0x1F7C; folded[1] = 0x03B9; return 2; // ῲ → ὼι
        case 0x1FF3: folded[0] = 0x03C9; folded[1] = 0x03B9; return 2; // ῳ → ωι
        case 0x1FF4: folded[0] = 0x03CE; folded[1] = 0x03B9; return 2; // ῴ → ώι
        case 0x1FF6: folded[0] = 0x03C9; folded[1] = 0x0342; return 2; // ῶ → ω + combining
        case 0x1FF7: folded[0] = 0x03C9; folded[1] = 0x0342; folded[2] = 0x03B9; return 3; // ῷ → ω + 2 combining
        case 0x1FFC: folded[0] = 0x03C9; folded[1] = 0x03B9; return 2; // ῼ → ωι
        // Alphabetic Presentation Forms: ligatures
        case 0xFB00: folded[0] = 0x0066; folded[1] = 0x0066; return 2; // ﬀ → ff
        case 0xFB01: folded[0] = 0x0066; folded[1] = 0x0069; return 2; // ﬁ → fi
        case 0xFB02: folded[0] = 0x0066; folded[1] = 0x006C; return 2; // ﬂ → fl
        case 0xFB03: folded[0] = 0x0066; folded[1] = 0x0066; folded[2] = 0x0069; return 3; // ﬃ → ffi
        case 0xFB04: folded[0] = 0x0066; folded[1] = 0x0066; folded[2] = 0x006C; return 3; // ﬄ → ffl
        case 0xFB05: folded[0] = 0x0073; folded[1] = 0x0074; return 2; // ﬅ → st
        case 0xFB06: folded[0] = 0x0073; folded[1] = 0x0074; return 2; // ﬆ → st
        // Armenian ligatures
        case 0xFB13: folded[0] = 0x0574; folded[1] = 0x0576; return 2; // ﬓ → մdelays (men + nun)
        case 0xFB14: folded[0] = 0x0574; folded[1] = 0x0565; return 2; // ﬔ → մdelays (men + ech)
        case 0xFB15: folded[0] = 0x0574; folded[1] = 0x056B; return 2; // ﬕ → մdelays (men + ini)
        case 0xFB16: folded[0] = 0x057E; folded[1] = 0x0576; return 2; // ﬖ → delays (vew + nun)
        case 0xFB17: folded[0] = 0x0574; folded[1] = 0x056D; return 2; // ﬗ → մdelays (men + xeh)
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
    case 0x10594: folded[0] = 0x105BB; return 1; // 𐖔 → 𐖻
    case 0x10595: folded[0] = 0x105BC; return 1; // 𐖕 → 𐖼
    }

    folded[0] = rune; return 1;  // No folding needed
    // clang-format on
}

/**
 *  @brief  Branchless ASCII case fold - converts A-Z to a-z.
 *  Uses unsigned subtraction trick: (c - 'A') <= 25 is true only for uppercase letters.
 */
SZ_INTERNAL sz_u8_t sz_ascii_fold_(sz_u8_t c) { return c + (((sz_u8_t)(c - 'A') <= 25u) * 0x20); }

/**
 *  @brief  Iterator state for streaming through folded UTF-8 runes.
 *  Handles one-to-many case folding expansions (e.g., ß → ss) transparently.
 */
typedef struct {
    sz_cptr_t ptr;           // Current position in UTF-8 string
    sz_cptr_t end;           // End of string
    sz_rune_t pending[4];    // Buffered folded runes from one-to-many expansions
    sz_size_t pending_count; // Number of pending folded runes
    sz_size_t pending_idx;   // Current index into pending buffer
} sz_utf8_folded_iter_t;

/** @brief Initialize a folded rune iterator. */
SZ_INTERNAL void sz_utf8_folded_iter_init_(sz_utf8_folded_iter_t *it, sz_cptr_t str, sz_size_t len) {
    it->ptr = str;
    it->end = str + len;
    it->pending_count = 0;
    it->pending_idx = 0;
}

/** @brief Get next folded rune. Returns `sz_false_k` when exhausted. Assumes valid UTF-8 input. */
SZ_INTERNAL sz_bool_t sz_utf8_folded_iter_next_(sz_utf8_folded_iter_t *it, sz_rune_t *out_rune) {
    // Refill pending buffer if exhausted
    if (it->pending_idx >= it->pending_count) {
        if (it->ptr >= it->end) return sz_false_k;

        // ASCII fast-path: fold inline without buffering
        sz_u8_t lead = *(sz_u8_t const *)it->ptr;
        if (lead < 0x80) {
            *out_rune = sz_ascii_fold_(lead);
            it->ptr++;
            return sz_true_k;
        }

        // Multi-byte UTF-8: decode, fold, and buffer
        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse(it->ptr, &rune, &rune_length);

        it->ptr += rune_length;
        it->pending_count = sz_unicode_fold_codepoint_(rune, it->pending);
        it->pending_idx = 0;
    }

    *out_rune = it->pending[it->pending_idx++];
    return sz_true_k;
}

/**
 *  @brief  Helper to verify a case-insensitive match by comparing folded runes.
 *  @return sz_true_k if all folded runes match, sz_false_k otherwise.
 */
SZ_INTERNAL sz_bool_t sz_utf8_verify_case_insensitive_match_( //
    sz_cptr_t needle, sz_size_t needle_length,                //
    sz_cptr_t window_start, sz_cptr_t window_end) {

    sz_utf8_folded_iter_t needle_iterator, haystack_iterator;
    sz_utf8_folded_iter_init_(&needle_iterator, needle, needle_length);
    sz_utf8_folded_iter_init_(&haystack_iterator, window_start, (sz_size_t)(window_end - window_start));

    sz_rune_t needle_rune = 0, haystack_rune = 0;
    for (;;) {
        sz_bool_t loaded_from_needle = sz_utf8_folded_iter_next_(&needle_iterator, &needle_rune);
        sz_bool_t loaded_from_haystack = sz_utf8_folded_iter_next_(&haystack_iterator, &haystack_rune);

        if (!loaded_from_needle && !loaded_from_haystack) return sz_true_k;  // Both exhausted = match
        if (!loaded_from_needle || !loaded_from_haystack) return sz_false_k; // One exhausted early = no match
        if (needle_rune != haystack_rune) return sz_false_k;
    }
}

/**
 *  @brief  Checks if a UTF-8 string contains only caseless codepoints.
 *
 *  Caseless codepoints are those that:
 *  1. Fold to themselves (no case transformation)
 *  2. Are not targets of any case folding from other codepoints
 *
 *  This includes: CJK ideographs, Hangul, digits, punctuation, most symbols,
 *  Hebrew, Arabic, Thai, and many other scripts without case distinctions.
 *
 *  If a string is fully caseless, `sz_find` can be used instead of case-insensitive search.
 */
SZ_INTERNAL sz_bool_t sz_utf8_is_fully_caseless_(sz_cptr_t str, sz_size_t len) {
    sz_u8_t const *ptr = (sz_u8_t const *)str;
    sz_u8_t const *end = ptr + len;

    while (ptr < end) {
        sz_u8_t lead = *ptr;

        // ASCII: only digits, punctuation, and control chars are caseless
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
        ptr += rune_len;

        // Check if this rune participates in case folding
        sz_rune_t folded[4];
        sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded);

        // If it expands or changes, it's not caseless
        if (folded_count != 1 || folded[0] != rune) return sz_false_k;

        // Check if this rune is a lowercase target of some uppercase letter.
        // Lowercase letters that don't change when folded still participate in case
        // because uppercase versions fold TO them. We must mark entire bicameral
        // script ranges as "not caseless" to enable proper case-insensitive matching.
        //
        // Bicameral scripts organized by UTF-8 lead byte for efficient checking:
        //
        // 2-byte sequences (U+0080-07FF): C2-DF lead bytes
        if (rune >= 0x00C0 && rune <= 0x00FF) return sz_false_k; // Latin-1 Supplement (À-ÿ)
        if (rune >= 0x0100 && rune <= 0x024F) return sz_false_k; // Latin Extended-A/B
        if (rune >= 0x0250 && rune <= 0x02AF) return sz_false_k; // IPA Extensions
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
    }

    return sz_true_k;
}

/**
 *  @brief  Rabin-Karp style case-insensitive UTF-8 substring search using a ring buffer.
 *          Uses a rolling hash over casefolded runes with O(1) updates per position.
 *          Ring buffer of 32 runes handles the prefix; longer needles verify tail separately.
 */
SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_serial( //
    sz_cptr_t haystack, sz_size_t haystack_length,        //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *match_length) {

    if (needle_length == 0) {
        *match_length = 0;
        return haystack;
    }

    if (sz_utf8_is_fully_caseless_(needle, needle_length)) {
        sz_cptr_t result = sz_find(haystack, haystack_length, needle, needle_length);
        if (result) {
            *match_length = needle_length;
            return result;
        }
        *match_length = 0;
        return SZ_NULL_CHAR;
    }

    sz_size_t const ring_capacity = 32;
    sz_rune_t needle_runes[32];
    sz_size_t needle_prefix_count = 0, needle_total_count = 0;
    sz_u64_t needle_hash = 0;
    sz_cptr_t needle_tail = SZ_NULL_CHAR;
    sz_size_t needle_tail_length = 0;
    {
        sz_utf8_folded_iter_t needle_iter;
        sz_utf8_folded_iter_init_(&needle_iter, needle, needle_length);
        sz_rune_t rune;
        while (needle_prefix_count < ring_capacity && sz_utf8_folded_iter_next_(&needle_iter, &rune)) {
            needle_runes[needle_prefix_count++] = rune;
            needle_hash = needle_hash * 257 + rune;
        }
        needle_total_count = needle_prefix_count;
        if (needle_prefix_count == ring_capacity) {
            needle_tail = needle_iter.ptr;
            while (sz_utf8_folded_iter_next_(&needle_iter, &rune)) needle_total_count++;
            needle_tail_length = needle_length - (sz_size_t)(needle_tail - needle);
        }
    }
    if (!needle_prefix_count) {
        *match_length = 0;
        return SZ_NULL_CHAR;
    }

    sz_u64_t hash_multiplier = 1;
    for (sz_size_t i = 1; i < needle_prefix_count; ++i) hash_multiplier *= 257;

    sz_rune_t window_runes[32];
    sz_cptr_t window_sources[32];
    sz_size_t ring_head = 0;
    sz_u64_t window_hash = 0;
    sz_utf8_folded_iter_t haystack_iter;
    sz_utf8_folded_iter_init_(&haystack_iter, haystack, haystack_length);

    sz_cptr_t window_start = haystack;
    sz_cptr_t current_source = haystack;
    sz_size_t window_count = 0;

    while (window_count < needle_prefix_count) {
        sz_cptr_t before_ptr = haystack_iter.ptr;
        sz_rune_t rune;
        if (!sz_utf8_folded_iter_next_(&haystack_iter, &rune)) break;
        window_runes[window_count] = rune;
        current_source = haystack_iter.pending_idx <= 1 ? before_ptr : current_source;
        window_sources[window_count] = current_source;
        window_hash = window_hash * 257 + rune;
        window_count++;
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
                if (needle_total_count <= ring_capacity) {
                    *match_length = (sz_size_t)(window_end - window_start);
                    return window_start;
                }
                sz_size_t haystack_tail_length = haystack_length - (sz_size_t)(window_end - haystack);
                sz_utf8_folded_iter_t needle_tail_iter, haystack_tail_iter;
                sz_utf8_folded_iter_init_(&needle_tail_iter, needle_tail, needle_tail_length);
                sz_utf8_folded_iter_init_(&haystack_tail_iter, window_end, haystack_tail_length);
                sz_rune_t needle_tail_rune, haystack_tail_rune;
                sz_bool_t tail_matches = sz_true_k;
                while (tail_matches && sz_utf8_folded_iter_next_(&needle_tail_iter, &needle_tail_rune)) {
                    if (!sz_utf8_folded_iter_next_(&haystack_tail_iter, &haystack_tail_rune) ||
                        needle_tail_rune != haystack_tail_rune)
                        tail_matches = sz_false_k;
                }
                if (tail_matches) {
                    *match_length = (sz_size_t)(haystack_tail_iter.ptr - window_start);
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
        window_start = window_sources[next_head];

        window_runes[ring_head] = new_rune;
        current_source = haystack_iter.pending_idx <= 1 ? before_ptr : current_source;
        window_sources[ring_head] = current_source;

        ring_head = next_head;
        window_end = haystack_iter.ptr;
    }

    *match_length = 0;
    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_size_t sz_utf8_case_fold_serial(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {

    sz_u8_t const *src = (sz_u8_t const *)source;
    sz_u8_t const *src_end = src + source_length;
    sz_u8_t *dst = (sz_u8_t *)destination;
    sz_u8_t *dst_start = dst;

    // Assumes valid UTF-8 input; use sz_utf8_valid() first if validation is needed.
    while (src < src_end) {
        // ASCII fast-path: process consecutive ASCII bytes without UTF-8 decode/encode overhead.
        // This handles ~95% of typical text content with minimal branching.
        while (src < src_end && *src < 0x80) *dst++ = sz_ascii_fold_(*src++);
        if (src >= src_end) break;

        // Multi-byte UTF-8 sequence: use full decode/fold/encode path
        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse((sz_cptr_t)src, &rune, &rune_length);
        src += rune_length;

        sz_rune_t folded[4];
        sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded);
        for (sz_size_t i = 0; i != folded_count; ++i) dst += sz_rune_export(folded[i], dst);
    }

    return (sz_size_t)(dst - dst_start);
}

SZ_PUBLIC sz_ordering_t sz_utf8_case_insensitive_order_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                              sz_size_t b_length) {
    sz_utf8_folded_iter_t a_iterator, b_iterator;
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
#define sz_ice_is_ascii_upper_(src_zmm) \
    _mm512_cmp_epu8_mask(_mm512_sub_epi8((src_zmm), a_upper_vec), subtract26_vec, _MM_CMPINT_LT)

/** Apply ASCII case folding (+0x20) to masked positions. */
#define sz_ice_fold_ascii_(src_zmm, upper_mask, apply_mask) \
    _mm512_mask_add_epi8((src_zmm), (upper_mask) & (apply_mask), (src_zmm), x20_vec)

/** Fold ASCII A-Z in source vector within a prefix mask, returning folded result. */
#define sz_ice_fold_ascii_in_prefix_(src_zmm, prefix_mask) \
    _mm512_mask_add_epi8((src_zmm), sz_ice_is_ascii_upper_(src_zmm) & (prefix_mask), (src_zmm), x20_vec)

/**
 *  Georgian uppercase transformation: E1 82/83 XX → E2 B4 YY.
 *  Applies to lead byte positions (sets E2), second byte positions (sets B4),
 *  and adjusts third bytes (-0x20 for 82 sequences, +0x20 for 83 sequences).
 */
#define sz_ice_transform_georgian_(folded, georgian_leads, is_82_upper, is_83_upper, prefix_mask)         \
    do {                                                                                                  \
        (folded) = _mm512_mask_blend_epi8((georgian_leads), (folded), _mm512_set1_epi8((char)0xE2));      \
        (folded) = _mm512_mask_blend_epi8((georgian_leads) << 1, (folded), _mm512_set1_epi8((char)0xB4)); \
        (folded) = _mm512_mask_sub_epi8((folded), (is_82_upper) & (prefix_mask), (folded), x20_vec);      \
        (folded) = _mm512_mask_add_epi8((folded), (is_83_upper) & (prefix_mask), (folded), x20_vec);      \
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
    //         Special case: ß (C3 9F) expands to "ss" - handled via mask blend.
    //
    //    2.2. Basic Cyrillic (D0/D1): А-я and Ѐ-џ (0x0400-0x045F).
    //         Uppercase is D0 90-AF and D1 80-8F. Folding maps lowercase TO uppercase:
    //         - D0 B0-BF (lowercase) → D0 90-AF: subtract 0x20
    //         - D1 80-8F (lowercase) → D0 A0-AF: add 0x20, normalize D1→D0
    //         Excludes Extended-A (D1 A0+) which needs +1 folding.
    //
    //    2.3. Greek (CE/CF): Basic Greek letters Α-Ω and α-ω.
    //         Similar structure: uppercase in CE, lowercase in CF.
    //         Excludes archaic letters and symbols needing special handling.
    //
    // 3. Groups of 3-byte codepoints - split into caseless and case-aware paths
    //
    //    3.1. Caseless 3-byte content: CJK (E4-E9), Hangul (EA with safe seconds),
    //         most punctuation (E2 80-83), Thai, Hindi, etc.
    //         Fast-path: copy directly without transformation.
    //
    //    3.2. Georgian uppercase (E1 82 80-9F, E1 83 80-8F) → lowercase (E2 B4 80-AF).
    //         Full 3-byte transformation: lead E1→E2, second byte→B4,
    //         third byte ±0x20 depending on original second byte (82 vs 83).
    //
    //    3.3. Fullwidth Latin (EF BC-BD): Ａ-Ｚ → ａ-ｚ.
    //         Second byte BC with third 81-9A: add 0x21 to third byte.
    //
    // 4. Groups of 4-byte codepoints (emoji, historic scripts)
    //    Generally caseless, but Deseret/Warang Citi have folding rules.
    //    Falls back to serial code for rare case-aware sequences.
    //
    // Within these assumptions, this kernel vectorizes case folding for simple cases, when
    // a sequence of bytes in a certain range can be folded by a single constant addition/subtraction.
    // For more complex cases it periodically switches to serial code.
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
    __m512i const x20_vec = _mm512_set1_epi8(0x20);

    // Additional pre-computed constants for UTF-8 lead byte detection
    __m512i const xc0_vec = _mm512_set1_epi8((char)0xC0);
    __m512i const x80_vec = _mm512_set1_epi8((char)0x80);
    __m512i const xf0_vec = _mm512_set1_epi8((char)0xF0);
    __m512i const xe0_vec = _mm512_set1_epi8((char)0xE0);
    __m512i const xf8_vec = _mm512_set1_epi8((char)0xF8);

    while (source_length) {
        sz_size_t chunk_size = sz_min_of_two(source_length, 64);
        __mmask64 load_mask = sz_u64_mask_until_(chunk_size);
        source_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, source);
        __mmask64 is_non_ascii = _mm512_movepi8_mask(source_vec.zmm);

        // Compute all lead byte masks once per iteration using pre-computed constants
        __mmask64 is_cont = _mm512_cmpeq_epi8_mask(_mm512_and_si512(source_vec.zmm, xc0_vec), x80_vec);
        __mmask64 is_three_byte_lead = _mm512_cmpeq_epi8_mask(_mm512_and_si512(source_vec.zmm, xf0_vec), xe0_vec);
        __mmask64 is_four_byte_lead = _mm512_cmpeq_epi8_mask(_mm512_and_si512(source_vec.zmm, xf8_vec), xf0_vec);

        // Check that all loaded characters are ASCII
        if (is_non_ascii == 0) {
            _mm512_mask_storeu_epi8(target, load_mask, sz_ice_fold_ascii_in_prefix_(source_vec.zmm, load_mask));
            target += chunk_size, source += chunk_size, source_length -= chunk_size;
            continue;
        }

        // Early fast path: Pure 3-byte content (no ASCII, no 2-byte, no 4-byte)
        // This is common for CJK, Hindi (Devanagari), Thai, etc.
        // Check if all non-ASCII bytes are 3-byte leads or continuations
        {
            __mmask64 is_valid_pure_3byte = is_three_byte_lead | is_cont;
            // Quick check: if all bits match, we have pure 3-byte content
            if ((is_valid_pure_3byte & load_mask) == (is_non_ascii & load_mask) && !is_four_byte_lead) {
                // Check for problematic leads that have case folding:
                //   - E1: Georgian, Greek Extended, Latin Extended Additional
                //   - E2: Glagolitic (B0-B1), Coptic (B2-B3), Letterlike (84 = Kelvin/Angstrom)
                //   - EF: Fullwidth A-Z
                // E2 80-83 are safe (General Punctuation quotes); other E2 ranges have folding
                // EA is mostly safe (Hangul B0-BF) but some second bytes have folding:
                //   - 0x99-0x9F: Cyrillic Ext-B, Latin Ext-D (A640-A7FF)
                //   - 0xAD-0xAE: Cherokee Supplement (AB70-ABBF)
                __mmask64 is_e1 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE1));
                __mmask64 is_ef = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xEF));
                // For E2, only allow 80-83 (General Punctuation quotes) through - many other E2 ranges have folding
                // (84 Letterlike, 93 Enclosed Alphanumerics, B0-B3 Glagolitic/Coptic, etc.)
                __mmask64 is_e2 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE2));
                __mmask64 e2_seconds = is_e2 << 1;
                // E2 folding needed if second byte is NOT in 80-83 range
                __mmask64 is_e2_folding =
                    e2_seconds & ~_mm512_cmp_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x80)),
                                                       _mm512_set1_epi8(0x04), _MM_CMPINT_LT); // NOT 80-83
                // For EA, check if second byte is in problematic ranges
                __mmask64 is_ea = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xEA));
                __mmask64 ea_seconds = is_ea << 1;
                // EA 99-9F (second byte 0x99-0x9F) or EA AD-AE (second byte 0xAD-0xAE)
                __mmask64 is_ea_folding =
                    ea_seconds & (_mm512_cmp_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x99)),
                                                       _mm512_set1_epi8(0x07), _MM_CMPINT_LT) | // 0x99-0x9F
                                  _mm512_cmp_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xAD)),
                                                       _mm512_set1_epi8(0x02), _MM_CMPINT_LT)); // 0xAD-0xAE
                if (!(is_e1 | is_e2_folding | is_ea_folding | is_ef)) {
                    // Safe 3-byte content (E0, E3-E9, EB-EE) - no 3-byte case folding needed
                    // But ASCII mixed in still needs folding! Use sz_ice_fold_ascii_in_prefix_.
                    // Just need to avoid splitting a 3-byte sequence at the end.
                    sz_size_t copy_len = chunk_size;
                    if (copy_len < 64) {
                        // Check if last 1-2 bytes are an incomplete sequence
                        __mmask64 leads_in_chunk = is_three_byte_lead & load_mask;
                        if (leads_in_chunk) {
                            int last_lead_pos = 63 - sz_u64_clz(leads_in_chunk);
                            if (last_lead_pos + 3 > (int)copy_len) copy_len = last_lead_pos;
                        }
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

        // 2. Two-byte UTF-8 sequences (lead bytes C0-DF)
        //
        // 2.1. Latin-1 Supplement (C3 80 - C3 BF) mixed with ASCII
        __mmask64 is_latin1_lead = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xC3));
        __mmask64 is_latin1_second = is_latin1_lead << 1;
        __mmask64 is_valid_latin1_mix = ~is_non_ascii | is_latin1_lead | is_latin1_second;
        sz_size_t latin1_length = sz_ice_first_invalid_(is_valid_latin1_mix, load_mask, chunk_size);
        latin1_length -= latin1_length && ((is_latin1_lead >> (latin1_length - 1)) & 1); // Don't split 2-byte seq

        if (latin1_length >= 2) {
            __mmask64 prefix_mask = sz_u64_mask_until_(latin1_length);
            __mmask64 latin1_second_bytes = is_latin1_second & prefix_mask;

            // ASCII A-Z (0x41-0x5A) and Latin-1 À-Þ (second byte 0x80-0x9E excl. ×=0x97) both get +0x20
            __mmask64 is_upper_ascii = sz_ice_is_ascii_upper_(source_vec.zmm);
            __mmask64 is_latin1_upper = _mm512_mask_cmp_epu8_mask(
                latin1_second_bytes, _mm512_sub_epi8(source_vec.zmm, x80_vec), _mm512_set1_epi8(0x1F), _MM_CMPINT_LT);
            is_latin1_upper ^= _mm512_mask_cmpeq_epi8_mask(is_latin1_upper, source_vec.zmm,
                                                           _mm512_set1_epi8((char)0x97)); // Exclude ×
            __m512i folded = sz_ice_fold_ascii_(source_vec.zmm, is_upper_ascii | is_latin1_upper, prefix_mask);

            // ß (C3 9F) → ss: replace both bytes with 's'
            __mmask64 is_eszett =
                _mm512_mask_cmpeq_epi8_mask(latin1_second_bytes, source_vec.zmm, _mm512_set1_epi8((char)0x9F));
            folded = _mm512_mask_set1_epi8(folded, is_eszett | (is_eszett >> 1), 's');

            _mm512_mask_storeu_epi8(target, prefix_mask, folded);
            target += latin1_length, source += latin1_length, source_length -= latin1_length;
            continue;
        }

        // 2.2. Cyrillic fast path (D0/D1 lead bytes for basic Cyrillic 0x0400-0x045F)
        //
        // Basic Cyrillic has predictable case folding that can be done in-place on second bytes:
        //   D0 80-8F (Ѐ-Џ, 0x0400-0x040F) → D1 90-9F (ѐ-џ)  - second byte +0x10, lead D0→D1
        //   D0 90-9F (А-П, 0x0410-0x041F) → D0 B0-BF (а-п)  - second byte +0x20
        //   D0 A0-AF (Р-Я, 0x0420-0x042F) → D1 80-8F (р-я)  - second byte -0x20, lead D0→D1
        //   D0 B0-BF, D1 80-9F: lowercase (а-я, ѐ-џ) - no change
        //
        // EXCLUDED from fast path: Cyrillic Extended-A (0x0460-04FF) which starts at D1 A0.
        // These use +1 folding for even codepoints and must go through the general 2-byte path.
        {
            __mmask64 is_d0 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xD0));
            __mmask64 is_d1 = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xD1));
            __mmask64 is_cyrillic_lead = is_d0 | is_d1;
            __mmask64 is_cyrillic_second = is_cyrillic_lead << 1;

            // Exclude Cyrillic Extended-A: D1 with second byte >= 0xA0 (U+0460+)
            // These have +1 folding rules, not the basic Cyrillic patterns
            __mmask64 is_d1_extended = (is_d1 << 1) & _mm512_cmp_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xA0),
                                                                           _MM_CMPINT_NLT); // >= 0xA0

            // Check for pure basic Cyrillic + ASCII mix (no extended)
            __mmask64 is_valid_cyrillic_mix = ~is_non_ascii | is_cyrillic_lead | is_cyrillic_second;
            is_valid_cyrillic_mix &= ~is_d1_extended; // Stop at Cyrillic Extended
            sz_size_t cyrillic_length = sz_ice_first_invalid_(is_valid_cyrillic_mix, load_mask, chunk_size);
            cyrillic_length -= cyrillic_length && ((is_cyrillic_lead >> (cyrillic_length - 1)) & 1);

            if (cyrillic_length >= 2) {
                __mmask64 prefix_mask = sz_u64_mask_until_(cyrillic_length);
                __mmask64 d0_seconds = (is_d0 << 1) & prefix_mask;

                // Start with source, apply ASCII folding
                __m512i folded = sz_ice_fold_ascii_in_prefix_(source_vec.zmm, prefix_mask);

                // D0 second bytes: apply Cyrillic uppercase folding
                // Range 0x80-0x8F (Ѐ-Џ): second byte += 0x10, lead becomes D1
                // Range 0x90-0x9F (А-П): second byte += 0x20, lead stays D0
                // Range 0xA0-0xAF (Р-Я): second byte -= 0x20, lead becomes D1
                __mmask64 is_d0_upper1 =
                    _mm512_mask_cmp_epu8_mask(d0_seconds, source_vec.zmm, _mm512_set1_epi8((char)0x90), _MM_CMPINT_LT);
                __mmask64 is_d0_upper2 =
                    _mm512_mask_cmp_epu8_mask(d0_seconds, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x90)),
                                              _mm512_set1_epi8(0x10), _MM_CMPINT_LT);
                __mmask64 is_d0_upper3 =
                    _mm512_mask_cmp_epu8_mask(d0_seconds, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xA0)),
                                              _mm512_set1_epi8(0x10), _MM_CMPINT_LT);

                // Apply transformations
                // Ѐ-Џ (0x80-0x8F): +0x10 to second byte
                folded = _mm512_mask_add_epi8(folded, is_d0_upper1, folded, _mm512_set1_epi8(0x10));
                // А-П (0x90-0x9F): +0x20 to second byte
                folded = _mm512_mask_add_epi8(folded, is_d0_upper2, folded, _mm512_set1_epi8(0x20));
                // Р-Я (0xA0-0xAF): -0x20 to second byte (wraps to 0x80-0x8F)
                folded = _mm512_mask_sub_epi8(folded, is_d0_upper3, folded, _mm512_set1_epi8(0x20));

                // Fix lead bytes: Ѐ-Џ and Р-Я need D0→D1
                __mmask64 needs_d1 = ((is_d0_upper1 | is_d0_upper3) >> 1) & (is_d0 & prefix_mask);
                folded = _mm512_mask_mov_epi8(folded, needs_d1, _mm512_set1_epi8((char)0xD1));

                _mm512_mask_storeu_epi8(target, prefix_mask, folded);
                target += cyrillic_length, source += cyrillic_length, source_length -= cyrillic_length;
                continue;
            }
        }

        // 2.3. Fast path for 2-byte scripts without case folding (Hebrew, Arabic, Syriac, etc.)
        //
        // Lead bytes D7-DF cover Hebrew (D7), Arabic (D8-DB), Syriac (DC-DD), Thaana/NKo (DE-DF).
        // None of these scripts have case distinctions, so we can just copy them unchanged.
        // NOTE: D5/D6 cover Armenian which HAS case folding (including U+0587 which expands).
        __mmask64 is_caseless_2byte =
            _mm512_cmp_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xD7), _MM_CMPINT_GE) &
            _mm512_cmp_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xDF), _MM_CMPINT_LE);
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

        // 2.4. Other 2-byte scripts (Latin Extended, Greek, Cyrillic, Armenian)
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
        __mmask64 is_two_byte_second = is_two_byte_lead << 1;

        // Accept ALL 2-byte sequences; we'll detect singletons after decoding
        __mmask64 is_valid_two_byte_mix = ~is_non_ascii | is_two_byte_lead | is_two_byte_second;
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
                _mm512_cmp_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0180)),
                                      _mm512_set1_epi32(0x00D0), _MM_CMPINT_LT) | // 0x0180-0x024F
                // Greek singletons with non-standard folding: 0x0345-0x0390
                // (Basic Greek Α-Ω 0x0391-0x03A9 and α-ω 0x03B1-0x03C9 are handled by vectorized +0x20 rules)
                // (Final sigma ς 0x03C2 → σ is also vectorized)
                _mm512_cmp_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0345)), _mm512_set1_epi32(0x4C),
                                      _MM_CMPINT_LT) | // 0x0345-0x0390
                // Greek ΰ (0x03B0) expands to 3 codepoints
                _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x03B0)) |
                // Greek symbols with non-standard folding: 0x03CF-0x03FF
                _mm512_cmp_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x03CF)), _mm512_set1_epi32(0x31),
                                      _MM_CMPINT_LT) | // 0x03CF-0x03FF
                // Cyrillic Extended: 0x0460-0x052F (has case folding, not covered)
                _mm512_cmp_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0460)),
                                      _mm512_set1_epi32(0x00D0), _MM_CMPINT_LT) | // 0x0460-0x052F
                // Armenian ligature that expands: U+0587 (և → եdelays, ech + yiwn)
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
                    sz_rune_t folded_runes[4];
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
            folded =
                _mm512_mask_add_epi32(folded,
                                      _mm512_cmp_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0041)),
                                                            _mm512_set1_epi32(26), _MM_CMPINT_LT),
                                      folded, _mm512_set1_epi32(0x20));
            // Cyrillic А-Я: 0x0410-0x042F → +0x20
            folded =
                _mm512_mask_add_epi32(folded,
                                      _mm512_cmp_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0410)),
                                                            _mm512_set1_epi32(0x20), _MM_CMPINT_LT),
                                      folded, _mm512_set1_epi32(0x20));
            // Cyrillic Ѐ-Џ: 0x0400-0x040F → +0x50
            folded =
                _mm512_mask_add_epi32(folded,
                                      _mm512_cmp_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0400)),
                                                            _mm512_set1_epi32(0x10), _MM_CMPINT_LT),
                                      folded, _mm512_set1_epi32(0x50));
            // Greek Α-Ρ: 0x0391-0x03A1 → +0x20
            folded =
                _mm512_mask_add_epi32(folded,
                                      _mm512_cmp_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0391)),
                                                            _mm512_set1_epi32(0x11), _MM_CMPINT_LT),
                                      folded, _mm512_set1_epi32(0x20));
            // Greek Σ-Ϋ: 0x03A3-0x03AB → +0x20
            folded =
                _mm512_mask_add_epi32(folded,
                                      _mm512_cmp_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x03A3)),
                                                            _mm512_set1_epi32(0x09), _MM_CMPINT_LT),
                                      folded, _mm512_set1_epi32(0x20));
            // Armenian Ա-Ֆ: 0x0531-0x0556 → +0x30
            folded =
                _mm512_mask_add_epi32(folded,
                                      _mm512_cmp_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0531)),
                                                            _mm512_set1_epi32(0x26), _MM_CMPINT_LT),
                                      folded, _mm512_set1_epi32(0x30));
            // Latin Extended-A/B: complex alternating pattern with varying parity
            __mmask16 is_even =
                _mm512_cmpeq_epi32_mask(_mm512_and_si512(codepoints, _mm512_set1_epi32(1)), _mm512_setzero_si512());
            __mmask16 is_odd = ~is_even;
            // Ranges where EVEN is uppercase (even → +1):
            // 0x0100-0x012F, 0x0132-0x0137, 0x014A-0x0177
            __mmask16 is_latin_even_upper =
                _mm512_cmp_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0100)), _mm512_set1_epi32(0x30),
                                      _MM_CMPINT_LT) |
                _mm512_cmp_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0132)), _mm512_set1_epi32(0x06),
                                      _MM_CMPINT_LT) |
                _mm512_cmp_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x014A)), _mm512_set1_epi32(0x2E),
                                      _MM_CMPINT_LT);
            folded = _mm512_mask_add_epi32(folded, is_latin_even_upper & is_even, folded, _mm512_set1_epi32(1));
            // Ranges where ODD is uppercase (odd → +1):
            // 0x0139-0x0148, 0x0179-0x017E
            __mmask16 is_latin_odd_upper =
                _mm512_cmp_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0139)), _mm512_set1_epi32(0x10),
                                      _MM_CMPINT_LT) |
                _mm512_cmp_epu32_mask(_mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x0179)), _mm512_set1_epi32(0x06),
                                      _MM_CMPINT_LT);
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
            __mmask16 is_ascii_out = _mm512_cmp_epu32_mask(folded, _mm512_set1_epi32(0x80), _MM_CMPINT_LT);
            new_lead = _mm512_mask_blend_epi32(is_ascii_out, new_lead, folded);

            // Scatter back using expand (inverse of compress) - no scalar loop needed!
            // Pack 32-bit results to bytes
            __m512i lead_zmm = _mm512_zextsi128_si512(_mm512_cvtepi32_epi8(new_lead));
            __m512i second_zmm = _mm512_zextsi128_si512(_mm512_cvtepi32_epi8(new_second));

            // Expand lead bytes to char start positions
            __m512i result = _mm512_mask_expand_epi8(source_vec.zmm, is_char_start, lead_zmm);

            // For second bytes: compress only 2-byte chars' seconds, then expand to continuation positions
            __m512i second_compressed = _mm512_maskz_compress_epi8((__mmask64)is_two_byte_char, second_zmm);
            result = _mm512_mask_expand_epi8(result, is_two_byte_second & prefix_mask, second_compressed);

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
                _mm512_cmp_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xC0)),
                                     _mm512_set1_epi8(0x20), _MM_CMPINT_LT); // C0-DF
            __mmask64 is_e1_lead = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE1));
            __mmask64 is_e2_lead = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xE2));
            __mmask64 is_ef_lead = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xEF));
            // For EA, only flag as complex if second byte is in problematic ranges
            __mmask64 is_ea_lead = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xEA));
            __mmask64 ea_second_bytes = is_ea_lead << 1;
            __mmask64 is_ea_complex =
                ea_second_bytes & (_mm512_cmp_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x99)),
                                                        _mm512_set1_epi8(0x07), _MM_CMPINT_LT) | // 0x99-0x9F
                                   _mm512_cmp_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xAD)),
                                                        _mm512_set1_epi8(0x02), _MM_CMPINT_LT)); // 0xAD-0xAE
            __mmask64 has_complex =
                (is_two_byte_lead | is_four_byte_lead | is_e1_lead | is_e2_lead | is_ea_complex | is_ef_lead) &
                load_mask;

            // Fast path: No complex bytes - just fold ASCII A-Z and copy everything else
            if (!has_complex) {
                // All bytes are ASCII or safe 3-byte (E0, E3-E9, EB-EE range)
                // Accept: ASCII, safe 3-byte leads (E0, E3-E9, EB-EE), continuations
                __mmask64 is_valid = ~is_non_ascii | is_three_byte_lead | is_cont;
                sz_size_t valid_length = sz_ice_first_invalid_(is_valid, load_mask, chunk_size);

                // Don't split a 3-byte sequence at chunk boundary.
                // Note: `>= 1` is an optimization, not redundant! When valid_length == 0, the mask
                // operations below are no-ops, but skipping them saves ~5% throughput on mixed content.
                if (valid_length >= 1) {
                    __mmask64 all_leads = is_three_byte_lead & sz_u64_mask_until_(valid_length);
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
                __m512i second_bytes =
                    _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(1)), source_vec.zmm);

                // Check for Georgian second bytes: 82 or 83
                // Only check positions where second byte is within chunk (positions 0-62 for 64-byte chunk)
                // Position 63's second byte would wrap around in the permutation
                __mmask64 safe_e1_mask = is_e1_lead & (load_mask >> 1);
                __mmask64 is_82_at_e1 =
                    _mm512_mask_cmpeq_epi8_mask(safe_e1_mask, second_bytes, _mm512_set1_epi8((char)0x82));
                __mmask64 is_83_at_e1 =
                    _mm512_mask_cmpeq_epi8_mask(safe_e1_mask, second_bytes, _mm512_set1_epi8((char)0x83));
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

                    __mmask64 is_82_uppercase = _mm512_mask_cmp_epu8_mask(
                        third_pos_82 & load_mask, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xA0)),
                        _mm512_set1_epi8(0x20), _MM_CMPINT_LT);
                    // For E1 83: check 80-85, 87, or 8D
                    __mmask64 is_83_range = _mm512_mask_cmp_epu8_mask(
                        third_pos_83 & load_mask, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x80)),
                        _mm512_set1_epi8(0x06), _MM_CMPINT_LT);
                    __mmask64 is_83_c7 = _mm512_mask_cmpeq_epi8_mask(third_pos_83 & load_mask, source_vec.zmm,
                                                                     _mm512_set1_epi8((char)0x87));
                    __mmask64 is_83_cd = _mm512_mask_cmpeq_epi8_mask(third_pos_83 & load_mask, source_vec.zmm,
                                                                     _mm512_set1_epi8((char)0x8D));
                    __mmask64 is_83_uppercase = is_83_range | is_83_c7 | is_83_cd;

                    // Include ASCII, ALL Georgian E1 (not just uppercase), E2 (punctuation), continuations, safe EA
                    // E2 is mostly safe (punctuation, symbols) - only a few codepoints fold (Kelvin, Angstrom)
                    // but we can safely pass those through unchanged (they're rare in Georgian text)
                    // Also include C2 leads (Latin-1 Supplement: U+0080-00BF) - no case folding needed
                    __mmask64 is_safe_ea = is_ea_lead & ~(is_ea_complex >> 1);
                    __mmask64 is_c2_lead = _mm512_cmpeq_epi8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xC2));
                    __mmask64 is_valid_georgian_mix =
                        ~is_non_ascii | is_georgian_e1 | is_e2_lead | is_cont | is_safe_ea | is_c2_lead;
                    // Exclude other 2-byte leads (C3-DF may need folding), 4-byte, EF
                    __mmask64 is_foldable_2byte = is_two_byte_lead & ~is_c2_lead;
                    is_valid_georgian_mix &= ~(is_foldable_2byte | is_four_byte_lead | is_ef_lead);
                    sz_size_t georgian_length = sz_ice_first_invalid_(is_valid_georgian_mix, load_mask, chunk_size);

                    // Don't split multi-byte sequences (2-byte C2, 3-byte E1/E2/EA).
                    // Note: `>= 1` is an optimization - skips mask ops when nothing valid.
                    if (georgian_length >= 1) {
                        __mmask64 prefix = sz_u64_mask_until_(georgian_length);
                        // Check for incomplete 3-byte sequences (leads in last 2 positions)
                        __mmask64 leads3_in_prefix = is_three_byte_lead & prefix;
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
                                                      folded, x20_vec);

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
            //   - U+1E96-U+1E9E (E1 BA 96-9E): These expand to multiple codepoints
            //     U+1E96 → h + combining, U+1E97 → t + combining, U+1E98 → w + combining,
            //     U+1E99 → y + combining, U+1E9A → a + modifier, U+1E9B → ṡ, U+1E9E → ss
            if (is_e1_lead && source_length >= 3) {
                __m512i second_bytes =
                    _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(1)), source_vec.zmm);
                __m512i third_bytes =
                    _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(2)), source_vec.zmm);

                // Check for Latin Ext Add second bytes: B8, B9, BA, BB
                __mmask64 safe_e1_mask = is_e1_lead & (load_mask >> 1);
                __mmask64 is_latin_ext_e1 =
                    _mm512_mask_cmp_epu8_mask(safe_e1_mask, _mm512_sub_epi8(second_bytes, _mm512_set1_epi8((char)0xB8)),
                                              _mm512_set1_epi8(0x04), _MM_CMPINT_LT); // 0xB8-0xBB

                // Check for U+1E96-U+1E9E (E1 BA 96-9E) which expand to multiple codepoints
                __mmask64 is_ba_second =
                    _mm512_mask_cmpeq_epi8_mask(safe_e1_mask, second_bytes, _mm512_set1_epi8((char)0xBA));
                // Third byte in range 0x96-0x9E (9 values: 0x96 + 0..8)
                // Note: third_bytes holds third byte value at lead positions, so compare at is_ba_second position
                __mmask64 is_special_third =
                    is_ba_second & _mm512_mask_cmp_epu8_mask(is_ba_second,
                                                             _mm512_sub_epi8(third_bytes, _mm512_set1_epi8((char)0x96)),
                                                             _mm512_set1_epi8(0x09), _MM_CMPINT_LT);

                // All E1 leads must be Latin Ext Add (no Greek Ext BC-BF, Georgian 82/83)
                __mmask64 non_latin_ext_e1 = safe_e1_mask & ~is_latin_ext_e1;
                if (!non_latin_ext_e1 && is_latin_ext_e1 && !is_special_third) {
                    // Pure Latin Extended Additional content
                    // Accept ASCII + Latin Ext Add E1 + continuations
                    __mmask64 is_valid_latin_ext = ~is_non_ascii | is_latin_ext_e1 | is_cont;
                    is_valid_latin_ext &= ~(is_four_byte_lead | is_ef_lead);
                    sz_size_t latin_ext_length = sz_ice_first_invalid_(is_valid_latin_ext, load_mask, chunk_size);

                    // Don't split 3-byte sequences
                    if (latin_ext_length >= 1) {
                        __mmask64 prefix = sz_u64_mask_until_(latin_ext_length);
                        __mmask64 leads_in_prefix = is_three_byte_lead & prefix;
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
                                                      folded, x20_vec);

                        _mm512_mask_storeu_epi8(target, prefix_mask, folded);
                        target += latin_ext_length, source += latin_ext_length, source_length -= latin_ext_length;
                        continue;
                    }
                }
            }

            // Slow path: Has 2-byte, 4-byte, or E1/E2/EF leads that need special handling
            // EA with problematic second bytes (is_ea_complex) also needs special handling
            // But plain EA (Hangul) is safe
            __mmask64 is_ea_lead_complex = is_ea_complex >> 1; // Shift back to lead position
            __mmask64 is_safe_three_byte_lead =
                is_three_byte_lead & ~is_e1_lead & ~is_e2_lead & ~is_ea_lead_complex & ~is_ef_lead;
            __mmask64 is_valid_mixed = ~is_non_ascii | is_safe_three_byte_lead | is_cont;
            is_valid_mixed &= ~is_four_byte_lead;
            sz_size_t three_byte_length = sz_ice_first_invalid_(is_valid_mixed, load_mask, chunk_size);

            // Don't split a 3-byte sequence: find first incomplete lead and truncate there.
            // Note: `>= 1` is an optimization - skips mask ops when nothing valid.
            if (three_byte_length >= 1) {
                __mmask64 all_leads = is_three_byte_lead & sz_u64_mask_until_(three_byte_length);
                __mmask64 safe_leads_mask = three_byte_length >= 3 ? sz_u64_mask_until_(three_byte_length - 2) : 0;
                __mmask64 unsafe_leads = all_leads & ~safe_leads_mask;
                if (unsafe_leads) three_byte_length = sz_u64_ctz(unsafe_leads);
            }

            // Need at least 2 bytes
            if (three_byte_length >= 2) {
                __mmask64 prefix_mask_3 = sz_u64_mask_until_(three_byte_length);
                __mmask64 three_byte_leads_in_prefix = is_three_byte_lead & prefix_mask_3;

                // Check for problematic lead bytes in this prefix
                __mmask64 problematic_leads = (is_e1_lead | is_ef_lead) & three_byte_leads_in_prefix;

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
                    __m512i second_bytes =
                        _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(1)), source_vec.zmm);
                    __m512i third_bytes =
                        _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(2)), source_vec.zmm);

                    // Check for Greek Extended (second byte BC-BF)
                    __mmask64 is_greek_ext = _mm512_mask_cmp_epu8_mask(
                        is_e1_in_prefix, _mm512_sub_epi8(second_bytes, _mm512_set1_epi8((char)0xBC)),
                        _mm512_set1_epi8(0x04), _MM_CMPINT_LT);

                    // Check for U+1E96-U+1E9E (E1 BA 96-9E) which expand to multiple codepoints
                    __mmask64 is_ba_second =
                        _mm512_mask_cmpeq_epi8_mask(is_e1_in_prefix, second_bytes, _mm512_set1_epi8((char)0xBA));
                    // Third byte in range 0x96-0x9E (9 values)
                    __mmask64 is_special_third = _mm512_mask_cmp_epu8_mask(
                        is_ba_second, _mm512_sub_epi8(third_bytes, _mm512_set1_epi8((char)0x96)),
                        _mm512_set1_epi8(0x09), _MM_CMPINT_LT);
                    __mmask64 needs_serial_e1 = is_greek_ext | is_special_third;

                    if (needs_serial_e1) {
                        sz_size_t first_special = sz_u64_ctz(needs_serial_e1);
                        if (first_special == 0) {
                            // First char needs serial processing (assumes valid UTF-8)
                            sz_rune_t rune;
                            sz_rune_length_t rune_length;
                            sz_rune_parse(source, &rune, &rune_length);
                            sz_rune_t folded_runes[4];
                            sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
                            for (sz_size_t i = 0; i != folded_count; ++i)
                                target += sz_rune_export(folded_runes[i], (sz_u8_t *)target);
                            source += rune_length;
                            source_length -= rune_length;
                            continue;
                        }
                        three_byte_length = first_special;
                        prefix_mask_3 = sz_u64_mask_until_(three_byte_length);
                        three_byte_leads_in_prefix = is_three_byte_lead & prefix_mask_3;
                        is_ef_lead &= prefix_mask_3;
                        is_e1_in_prefix = is_e1_lead & three_byte_leads_in_prefix;
                    }

                    // Vectorized Latin Extended Additional (E1 B8-BB = U+1E00-1EFF)
                    // This covers Vietnamese letters which use +1 folding for even codepoints.
                    // Check for E1 B8/B9/BA/BB (second byte 0xB8-0xBB)
                    __mmask64 is_latin_ext_add = _mm512_mask_cmp_epu8_mask(
                        is_e1_in_prefix, _mm512_sub_epi8(second_bytes, _mm512_set1_epi8((char)0xB8)),
                        _mm512_set1_epi8(0x04), _MM_CMPINT_LT); // 0xB8-0xBB

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
                        _mm512_mask_cmpeq_epi8_mask(is_e1_in_prefix, second_bytes, _mm512_set1_epi8((char)0x82));
                    __mmask64 is_83_second =
                        _mm512_mask_cmpeq_epi8_mask(is_e1_in_prefix, second_bytes, _mm512_set1_epi8((char)0x83));
                    __mmask64 is_georgian_second = is_82_second | is_83_second;

                    if (is_georgian_second) {
                        // Validate third byte range for Georgian uppercase:
                        // - E1 82 A0-BF: U+10A0-10BF (32 chars)
                        // - E1 83 80-85: U+10C0-10C5 (6 chars)
                        __mmask64 third_pos_82 = is_82_second << 2;
                        __mmask64 third_pos_83 = is_83_second << 2;

                        // For E1 82: third byte must be A0-BF
                        __mmask64 is_82_valid = _mm512_mask_cmp_epu8_mask(
                            third_pos_82, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0xA0)),
                            _mm512_set1_epi8(0x20), _MM_CMPINT_LT);

                        // For E1 83: third byte must be 80-85
                        __mmask64 is_83_valid = _mm512_mask_cmp_epu8_mask(
                            third_pos_83, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x80)),
                            _mm512_set1_epi8(0x06), _MM_CMPINT_LT);

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
                            folded = _mm512_mask_add_epi8(
                                folded, sz_ice_is_ascii_upper_(source_vec.zmm) & prefix_mask_3, folded, x20_vec);

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
                    __m512i second_bytes =
                        _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(1)), source_vec.zmm);
                    __m512i third_bytes =
                        _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(2)), source_vec.zmm);

                    // Check for EF BC (Fullwidth block FF00-FF3F)
                    __mmask64 is_ef_bc =
                        _mm512_mask_cmpeq_epi8_mask(is_ef_in_prefix, second_bytes, _mm512_set1_epi8((char)0xBC));

                    // Check third byte in A1-BA range (Fullwidth A-Z = FF21-FF3A)
                    // Third byte A1-BA corresponds to codepoints FF21-FF3A
                    __mmask64 is_fullwidth_az =
                        _mm512_mask_cmp_epu8_mask(is_ef_bc, _mm512_sub_epi8(third_bytes, _mm512_set1_epi8((char)0xA1)),
                                                  _mm512_set1_epi8(0x1A), _MM_CMPINT_LT);

                    if (is_fullwidth_az) {
                        // Has Fullwidth A-Z - apply +0x20 to third byte for those positions
                        // Also fold any ASCII A-Z in the mixed content
                        __mmask64 third_byte_positions = is_fullwidth_az << 2;
                        __mmask64 fold_mask =
                            (third_byte_positions | sz_ice_is_ascii_upper_(source_vec.zmm)) & prefix_mask_3;
                        __m512i folded = _mm512_mask_add_epi8(source_vec.zmm, fold_mask, source_vec.zmm, x20_vec);
                        _mm512_mask_storeu_epi8(target, prefix_mask_3, folded);
                        target += three_byte_length, source += three_byte_length, source_length -= three_byte_length;
                        continue;
                    }
                }

                // No special 3-byte cases found - fold ASCII A-Z, copy 3-byte unchanged
                _mm512_mask_storeu_epi8(target, prefix_mask_3,
                                        sz_ice_fold_ascii_in_prefix_(source_vec.zmm, prefix_mask_3));
                target += three_byte_length, source += three_byte_length, source_length -= three_byte_length;
                continue;
            }
        }

        // 4. Handle 4-byte sequences (emoji, rare scripts): detect lead bytes (11110xxx = F0-F7)
        {
            __mmask64 is_valid_four_byte_only = is_four_byte_lead | is_cont;
            sz_size_t four_byte_length = sz_ice_first_invalid_(is_valid_four_byte_only, load_mask, chunk_size);

            // Don't split a 4-byte sequence: find first incomplete lead and truncate there.
            // Note: `>= 1` is an optimization - skips mask ops when nothing valid.
            if (four_byte_length >= 1) {
                __mmask64 all_leads = is_four_byte_lead & sz_u64_mask_until_(four_byte_length);
                __mmask64 safe_leads_mask = four_byte_length >= 4 ? sz_u64_mask_until_(four_byte_length - 3) : 0;
                __mmask64 unsafe_leads = all_leads & ~safe_leads_mask;
                if (unsafe_leads) four_byte_length = sz_u64_ctz(unsafe_leads);
            }

            if (four_byte_length >= 4) {
                // Check if all 4-byte chars are in non-folding ranges (emoji: second byte >= 0x9F)
                __m512i second_bytes =
                    _mm512_permutexvar_epi8(_mm512_add_epi8(indices_vec, _mm512_set1_epi8(1)), source_vec.zmm);
                __mmask64 is_emoji_lead =
                    _mm512_cmp_epu8_mask(second_bytes, _mm512_set1_epi8((char)0x9F), _MM_CMPINT_GE);
                __mmask64 prefix_mask_4 = sz_u64_mask_until_(four_byte_length);
                __mmask64 four_byte_leads_in_prefix = is_four_byte_lead & prefix_mask_4;

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

            sz_rune_t folded[4];
            sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded);
            for (sz_size_t i = 0; i != folded_count; ++i) target += sz_rune_export(folded[i], (sz_u8_t *)target);
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

/**
 *  @brief  Helper function to find the longest slice of the needle composed exclusively of "safe" ASCII chacters
 *          so that we can use a simpler ASCII path for case-insensitive UTF-8 search.
 *
 *  @return sz_cptr_t Pointer to the start of the longest safe ASCII slice within the needle.
 *  @retval NULL No safe ASCII slice found.
 *  @param[in] needle Pointer to the UTF-8 needle string.
 *  @param[in] needle_length Length of the needle string in bytes.
 *  @param[out] longest_range_length Length of the longest safe ASCII slice found.
 *
 *  On this algorithmic path, we assume that we are only interested in single-byte UTF-8 characters and don't
 *  even try to detect 2-, 3-, and 4-byte characters, because the needle slice is guaranteed to not contain any.
 *  The letters [bcdegmopqruvxz] and their uppercase counterparts are considered safe because they never appear
 *  in case-folds of other Unicode characters. Following non-letter ASCII characters are also safe:
 *  - 33 control characters: 0x00-0x1F, 0x7F
 *  - 10 digits: 0123456789
 *  - 32 punctuation and sign symbols: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
 *  - whitespaces?
 *
 *  The [afhijklnstwy] letters, on the other hand, all depend on special conditions. We can use them on this safe
 *  path, but if and only if the following conditions are met
 *
 *   Char    Hex    Safe following                           Safe preceding        Examples
 *  ------------------------------------------------------------------------------------------------------------
 *   'a'     0x61   NOT: 'ʾ' (0xCA 0xBE)                     any                   'ẚ'→"aʾ"
 *   'f'     0x66   NOT: 'f' (0x66), 'i' (0x69), 'l' (0x6C)  NOT: 'f' (0x66)       'ﬀ'→"ff", 'ﬁ'→"fi", 'ﬂ'→"fl"
 *   'h'     0x68   NOT: '̱' (0xCC 0xB1)                      any                   'ẖ'→"ẖ"
 *   'i'     0x69   NOT: '̇' (0xCC 0x87)                      NOT: 'f' (0x66)       'İ'→"i̇"
 *   'j'     0x6A   NOT: '̌' (0xCC 0x8C)                      any                   'ǰ'→"ǰ"
 *   'l'     0x6C   any                                      NOT: 'f' (0x66)
 *   'n'     0x6E   any                                      NOT: 'ʼ' (0xCA 0xBC)
 *   's'     0x73   NOT: 's' (0x73), 't' (0x74)              NOT: 's' (0x73)       'ß'→"ss", 'ẞ'→"ss", 'ﬅ'→"st"
 *   't'     0x74   NOT: '̈' (0xCC 0x88)                      NOT: 's' (0x73)       'ẗ'→"ẗ"
 *   'w'     0x77   NOT: '̊' (0xCC 0x8A)                      any                   'ẘ'→"ẘ"
 *   'y'     0x79   NOT: '̊' (0xCC 0x8A)                      any                   'ẙ'→"ẙ"
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_ice_can_use_ascii_path_(sz_cptr_t needle, sz_size_t needle_length,
                                                                            sz_size_t *longest_range_length) {
    sz_cptr_t best_start = SZ_NULL_CHAR;
    sz_size_t best_length = 0;
    sz_cptr_t current_start = needle;
    sz_size_t current_length = 0;

    for (sz_size_t i = 0; i < needle_length;) {
        sz_u8_t byte = (sz_u8_t)needle[i];

        // Non-ASCII: end current slice and skip the multi-byte sequence
        if (byte >= 0x80) {
            if (current_length > best_length) {
                best_start = current_start;
                best_length = current_length;
            }
            // Branchless UTF-8 length (same formula as sz_rune_parse)
            sz_size_t skip = 1 + (byte >= 0xC0) + (byte >= 0xE0) + (byte >= 0xF0);
            i += skip;
            current_start = needle + i;
            current_length = 0;
            continue;
        }

        // ASCII byte - check safety based on neighbors
        sz_u8_t prev = (i > 0) ? (sz_u8_t)needle[i - 1] : 0;
        sz_u8_t next = (i + 1 < needle_length) ? (sz_u8_t)needle[i + 1] : 0;
        int prev_ascii = (i == 0) || (prev < 0x80);
        int next_ascii = (i + 1 >= needle_length) || (next < 0x80);

        // Lowercase for comparison
        sz_u8_t lower = (byte >= 'A' && byte <= 'Z') ? (byte + 0x20) : byte;
        sz_u8_t lower_prev = (prev >= 'A' && prev <= 'Z') ? (prev + 0x20) : prev;
        sz_u8_t lower_next = (next >= 'A' && next <= 'Z') ? (next + 0x20) : next;

        sz_bool_t is_safe;
        switch (lower) {
            // clang-format off
        // Unconditionally safe: no Unicode chars fold to sequences containing these
        case 'b': case 'c': case 'd': case 'e': case 'g': case 'k':
        case 'm': case 'o': case 'p': case 'q': case 'r': case 'u':
        case 'v': case 'x': case 'z':
            is_safe = sz_true_k;
            break;
        // clang-format on

        // Safe if next is ASCII (could be followed by combining chars: CC XX)
        // 'a' → ẚ folds to "aʾ" (a + CA BE)
        // 'h' → ẖ has combining macron below (CC B1)
        // 'j' → ǰ has combining caron (CC 8C)
        // 'w' → ẘ has combining ring above (CC 8A)
        // 'y' → ẙ has combining ring above (CC 8A)
        case 'a':
        case 'h':
        case 'j':
        case 'w':
        case 'y': is_safe = next_ascii ? sz_true_k : sz_false_k; break;

        // 'n': ʼn folds to "ʼn" (CA BC + n), so unsafe if preceded by non-ASCII
        case 'n': is_safe = prev_ascii ? sz_true_k : sz_false_k; break;

        // 'i': İ folds to "i̇" (i + CC 87), fi ligature exists
        case 'i': is_safe = (next_ascii && (prev_ascii == 0 || lower_prev != 'f')) ? sz_true_k : sz_false_k; break;

        // 'l': fl ligature exists
        case 'l': is_safe = (prev_ascii == 0 || lower_prev != 'f') ? sz_true_k : sz_false_k; break;

        // 't': ẗ has combining diaeresis (CC 88), st ligature exists
        case 't': is_safe = (next_ascii && (prev_ascii == 0 || lower_prev != 's')) ? sz_true_k : sz_false_k; break;

        // 'f': ff/fi/fl ligatures exist
        case 'f':
            is_safe = (prev_ascii && next_ascii && lower_prev != 'f' && lower_next != 'f' && lower_next != 'i' &&
                       lower_next != 'l')
                          ? sz_true_k
                          : sz_false_k;
            break;

        // 's': ß/ẞ fold to "ss", ﬅ folds to "st"
        case 's':
            is_safe = (prev_ascii && next_ascii && lower_prev != 's' && lower_next != 's' && lower_next != 't')
                          ? sz_true_k
                          : sz_false_k;
            break;

        // Non-letters: digits, punctuation, whitespace - always safe
        default: is_safe = sz_true_k; break;
        }

        if (is_safe) {
            current_length++;
            i++;
        }
        else {
            if (current_length > best_length) {
                best_start = current_start;
                best_length = current_length;
            }
            current_start = needle + i + 1;
            current_length = 0;
            i++;
        }
    }

    if (current_length > best_length) {
        best_start = current_start;
        best_length = current_length;
    }

    *longest_range_length = best_length;
    return best_start;
}

/**
 *  @brief  Fold a ZMM register using ASCII case folding rules.
 *
 *  Only transforms A-Z (0x41-0x5A) to a-z (0x61-0x7A).
 *  All other bytes pass through unchanged - this is the key fix.
 *
 *  @param[in] source The source ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_ascii_fold_zmm_(__m512i source) {
    __m512i const a_upper_vec = _mm512_set1_epi8('A');
    __m512i const range_26_vec = _mm512_set1_epi8(26);
    __m512i const x20_vec = _mm512_set1_epi8(0x20);

    // Only fold bytes in range A-Z: (byte - 'A') < 26
    __mmask64 is_ascii_upper = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source, a_upper_vec), range_26_vec);

    return _mm512_mask_add_epi8(source, is_ascii_upper, source, x20_vec);
}

/**
 *  @brief  ASCII case-insensitive substring search using AVX-512 with haystack folding.
 *
 *  Uses the Raita heuristic (first/mid/last character probes) with proper masked haystack folding.
 *  This kernel only handles safe ASCII slices where all positions have simple 1:1 case folding.
 *  The safe slice detector already excludes problematic patterns like "ss", ligatures, etc.
 *
 *  @param[in] haystack Pointer to the haystack string.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] needle Pointer to the needle string.
 *  @param[in] needle_length Length of the needle in bytes.
 *  @param[in] safe_start Pointer to start of safe ASCII slice within needle.
 *  @param[in] safe_length Length of the safe slice in bytes.
 *  @param[out] matched_length Length of the match in haystack bytes.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_ascii_ice_(sz_cptr_t haystack, sz_size_t haystack_length,
                                                               sz_cptr_t needle, sz_size_t needle_length,
                                                               sz_cptr_t safe_start, sz_size_t safe_length,
                                                               sz_size_t *matched_length) {

    // Clamp safe slice to single ZMM register
    if (safe_length > 64) safe_length = 64;

    sz_size_t const safe_offset = (sz_size_t)(safe_start - needle);
    sz_size_t const head_length = safe_offset;
    sz_size_t const tail_length = needle_length - safe_offset - safe_length;

    // Pre-fold entire safe slice for vectorized verification, to further reduce calls
    // to serial `sz_utf8_verify_case_insensitive_match_`
    __mmask64 const safe_mask = sz_u64_mask_until_(safe_length);
    __m512i const needle_safe_folded = sz_ascii_fold_zmm_(_mm512_maskz_loadu_epi8(safe_mask, safe_start));

    // Pick random anomalies similar to our case-sensitive algorithm, but only within the folded needle
    char needle_safe_folded_bytes[64];
    _mm512_mask_storeu_epi8(needle_safe_folded_bytes, safe_mask, needle_safe_folded);
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle_safe_folded_bytes, safe_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast needle probe bytes for comparisons with pulled haystack data
    sz_u512_vec_t probe_first_vec, probe_mid_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_safe_folded_bytes[offset_first]);
    probe_mid_vec.zmm = _mm512_set1_epi8(needle_safe_folded_bytes[offset_mid]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_safe_folded_bytes[offset_last]);

    // Main loop - 64 positions per iteration
    sz_u512_vec_t haystack_first_vec, haystack_mid_vec, haystack_last_vec;
    for (; haystack_length >= needle_length + 64; haystack += 64, haystack_length -= 64) {
        // Load haystack parts at different offsets
        haystack_first_vec.zmm = sz_ascii_fold_zmm_(_mm512_loadu_si512(haystack + safe_offset + offset_first));
        haystack_mid_vec.zmm = sz_ascii_fold_zmm_(_mm512_loadu_si512(haystack + safe_offset + offset_mid));
        haystack_last_vec.zmm = sz_ascii_fold_zmm_(_mm512_loadu_si512(haystack + safe_offset + offset_last));

        // 3-point Raita filter: find positions where all 3 probes match
        sz_u64_t matches =
            _kand_mask64(_kand_mask64(_mm512_cmpeq_epi8_mask(haystack_first_vec.zmm, probe_first_vec.zmm),
                                      _mm512_cmpeq_epi8_mask(haystack_mid_vec.zmm, probe_mid_vec.zmm)),
                         _mm512_cmpeq_epi8_mask(haystack_last_vec.zmm, probe_last_vec.zmm));

        // Verify all match candidates found
        while (matches) {
            int const idx = sz_u64_ctz(matches);
            sz_cptr_t const candidate = haystack + idx;
            __m512i const haystack_safe =
                sz_ascii_fold_zmm_(_mm512_maskz_loadu_epi8(safe_mask, candidate + safe_offset));
            if (_mm512_mask_cmpneq_epi8_mask(safe_mask, haystack_safe, needle_safe_folded) == 0) {
                sz_bool_t same_head = //
                    head_length ? sz_utf8_verify_case_insensitive_match_(needle, head_length, candidate,
                                                                         candidate + head_length)
                                : sz_true_k;
                sz_bool_t same_tail = //
                    tail_length ? sz_utf8_verify_case_insensitive_match_(safe_start + safe_length, tail_length,
                                                                         candidate + safe_offset + safe_length,
                                                                         candidate + needle_length)
                                : sz_true_k;
                if (same_head && same_tail) {
                    *matched_length = needle_length;
                    return candidate;
                }
            }
            matches &= matches - 1;
        }
    }

    // Tail: use masked loads for remaining positions (single iteration)
    if (haystack_length >= needle_length) {
        __mmask64 tail_mask = sz_u64_mask_until_(haystack_length - needle_length + 1);
        haystack_first_vec.zmm =
            sz_ascii_fold_zmm_(_mm512_maskz_loadu_epi8(tail_mask, haystack + safe_offset + offset_first));
        haystack_mid_vec.zmm =
            sz_ascii_fold_zmm_(_mm512_maskz_loadu_epi8(tail_mask, haystack + safe_offset + offset_mid));
        haystack_last_vec.zmm =
            sz_ascii_fold_zmm_(_mm512_maskz_loadu_epi8(tail_mask, haystack + safe_offset + offset_last));

        sz_u64_t matches =
            _kand_mask64(_kand_mask64(_mm512_cmpeq_epi8_mask(haystack_first_vec.zmm, probe_first_vec.zmm),
                                      _mm512_cmpeq_epi8_mask(haystack_mid_vec.zmm, probe_mid_vec.zmm)),
                         _mm512_cmpeq_epi8_mask(haystack_last_vec.zmm, probe_last_vec.zmm)) &
            tail_mask;

        while (matches) {
            int const idx = sz_u64_ctz(matches);
            sz_cptr_t const candidate = haystack + idx;
            __m512i const haystack_safe =
                sz_ascii_fold_zmm_(_mm512_maskz_loadu_epi8(safe_mask, candidate + safe_offset));
            if (_mm512_mask_cmpneq_epi8_mask(safe_mask, haystack_safe, needle_safe_folded) == 0) {
                sz_bool_t same_head = head_length ? sz_utf8_verify_case_insensitive_match_(
                                                        needle, head_length, candidate, candidate + head_length)
                                                  : sz_true_k;
                sz_bool_t same_tail =
                    tail_length ? sz_utf8_verify_case_insensitive_match_(safe_start + safe_length, tail_length,
                                                                         candidate + safe_offset + safe_length,
                                                                         candidate + needle_length)
                                : sz_true_k;
                if (same_head && same_tail) {
                    *matched_length = needle_length;
                    return candidate;
                }
            }
            matches &= matches - 1;
        }
    }

    return SZ_NULL_CHAR;
}

/**
 *  @brief  Helper function to find the longest slice of the needle composed exclusively of "safe" Latin 1 chacters
 *          so that we can only match a mixture of 1- and 2-byte characters, only performing case-folding via
 *          subtraction, without explicit codepoint unpacking into the UTF-16 form.
 *
 *  Just like the `sz_utf8_case_fold_ice`, it handles some of the equi-sized case-folds beyond subtraction/addition,
 *  including the German Eszett (ß) and Greek Sigma (Σ).
 *
 *  @return sz_cptr_t Pointer to the start of the longest safe Latin 1 slice within the needle.
 *  @retval NULL No safe Latin 1 slice found.
 *  @param[in] needle Pointer to the UTF-8 needle string.
 *  @param[in] needle_length Length of the needle string in bytes.
 *  @param[out] longest_range_length Length of the longest safe Latin 1 slice found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_ice_can_use_latin1_path_(sz_cptr_t needle, sz_size_t needle_length,
                                                                             sz_size_t *longest_range_length) {
    // Latin-1 Supplement safe path: ASCII + 2-byte Latin-1 (C2/C3 lead bytes)
    // Safe: C3 80-9E (except 97 multiplication sign) - uppercase À-Þ fold via +0x20
    // Safe: C3 9F (ß) - blends to "ss" (both bytes become 's'), same width
    // Safe: C3 A0-BF - lowercase à-ÿ
    // Safe: C2 80-BF - Latin-1 punctuation/symbols, no case folding needed

    sz_cptr_t best_start = SZ_NULL_CHAR;
    sz_size_t best_length = 0;
    sz_cptr_t current_start = needle;
    sz_size_t current_length = 0;

    for (sz_size_t i = 0; i < needle_length;) {
        sz_u8_t byte = (sz_u8_t)needle[i];

        // ASCII byte - safe in Latin1 context
        if (byte < 0x80) {
            current_length++;
            i++;
            continue;
        }

        // Check for 2-byte Latin-1 sequence (C3 lead)
        if (byte == 0xC3 && i + 1 < needle_length) {
            sz_u8_t second = (sz_u8_t)needle[i + 1];
            // C3 80-BF covers À-ÿ range, INCLUDING ß (0x9F) which blends to "ss"
            if (second >= 0x80 && second <= 0xBF) {
                current_length += 2;
                i += 2;
                continue;
            }
        }

        // Check for C2 lead byte (Latin-1 Supplement punctuation/symbols)
        if (byte == 0xC2 && i + 1 < needle_length) {
            sz_u8_t second = (sz_u8_t)needle[i + 1];
            if (second >= 0x80 && second <= 0xBF) {
                current_length += 2;
                i += 2;
                continue;
            }
        }

        // Any other multi-byte sequence breaks the Latin1 path
        if (current_length > best_length) {
            best_start = current_start;
            best_length = current_length;
        }
        // Skip the multi-byte sequence
        sz_size_t skip = 1;
        if ((byte & 0xE0) == 0xC0) skip = 2;
        else if ((byte & 0xF0) == 0xE0)
            skip = 3;
        else if ((byte & 0xF8) == 0xF0)
            skip = 4;
        i += skip;
        current_start = needle + i;
        current_length = 0;
    }

    if (current_length > best_length) {
        best_start = current_start;
        best_length = current_length;
    }

    *longest_range_length = best_length;
    return best_start;
}

/**
 *  @brief  Fold a ZMM register using Latin-1 case folding rules.
 *
 *  Folds ASCII uppercase (A-Z) and Latin-1 uppercase (C3 80-9E) to lowercase.
 *  Also handles eszett: ß (C3 9F) → ss (73 73).
 *
 *  @param[in] source The source ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_latin1_fold_zmm_(__m512i source) {
    __m512i const c3_vec = _mm512_set1_epi8((char)0xC3);
    __m512i const x9f_vec = _mm512_set1_epi8((char)0x9F);
    __m512i const s_vec = _mm512_set1_epi8('s');
    __m512i const x20_vec = _mm512_set1_epi8(0x20);
    __m512i const x80_vec = _mm512_set1_epi8((char)0x80);
    __m512i const x1f_vec = _mm512_set1_epi8(0x1F);
    __m512i const a_upper_vec = _mm512_set1_epi8('A');
    __m512i const range_26_vec = _mm512_set1_epi8(26);

    // Step 1: Detect C3 lead bytes → second byte positions
    __mmask64 is_c3 = _mm512_cmpeq_epi8_mask(source, c3_vec);
    __mmask64 is_second = is_c3 << 1;

    // Step 2: Detect eszett (C3 9F) and fold both bytes to 's'
    __mmask64 is_9f = _mm512_mask_cmpeq_epi8_mask(is_second, source, x9f_vec);
    __mmask64 is_eszett = is_9f | (is_9f >> 1); // both C3 and 9F positions
    __m512i result = _mm512_mask_mov_epi8(source, is_eszett, s_vec);

    // Step 3: Fold Latin-1 uppercase (C3 80-9E) → add 0x20 to second byte
    // Range check: (byte - 0x80) < 0x1F means byte in [0x80, 0x9E]
    // Exclude eszett (0x9F, already handled) and multiplication sign (0x97, no case variant)
    __mmask64 is_x97 = _mm512_mask_cmpeq_epi8_mask(is_second, source, _mm512_set1_epi8((char)0x97));
    __mmask64 is_latin1_upper =
        _mm512_mask_cmplt_epu8_mask(is_second & ~is_9f & ~is_x97, _mm512_sub_epi8(result, x80_vec), x1f_vec);

    // Step 4: Fold ASCII uppercase (A-Z) → add 0x20
    __mmask64 is_ascii_upper =
        _mm512_mask_cmplt_epu8_mask(~is_eszett, _mm512_sub_epi8(result, a_upper_vec), range_26_vec);

    // Step 5: Apply folding (+0x20) to uppercase positions
    return _mm512_mask_add_epi8(result, is_latin1_upper | is_ascii_upper, result, x20_vec);
}

/**
 *  @brief  Safety profile for a single character across all script paths.
 *
 *  Returned by sz_utf8_char_script_safety_() to indicate which fast paths
 *  can safely include this character in their safe window.
 */
typedef struct {
    sz_bool_t ascii;      ///< Safe for ASCII-only fast path
    sz_bool_t latin1;     ///< Safe for Latin-1 (ASCII + C2/C3) path
    sz_bool_t cyrillic;   ///< Safe for Cyrillic (ASCII + D0/D1) path
    sz_bool_t greek;      ///< Safe for Greek (ASCII + CE/CF) path
    sz_bool_t armenian;   ///< Safe for Armenian (ASCII + D4/D5/D6) path
    sz_bool_t vietnamese; ///< Safe for Vietnamese (ASCII + Latin1 + E1 B8-BB) path
} sz_char_script_safety_t;

/**
 *  @brief  A safe window for a single script path with integrated probe info.
 *
 *  Contains both the window boundaries and the 3-point probe positions.
 *  During analysis, we track current (building) and best (longest) windows separately.
 *  After analysis, only the best window is returned in sz_needle_analysis_t.
 */
typedef struct {
    sz_size_t start;  ///< Start offset of window within needle (in bytes)
    sz_size_t length; ///< Length of window in bytes

    // Probe positions for 3-point Raita filter (last byte of first/mid/last codepoints)
    sz_size_t probe_first; ///< Byte offset of first probe (relative to window start)
    sz_size_t probe_mid;   ///< Byte offset of mid probe (relative to window start)
    sz_size_t probe_last;  ///< Byte offset of last probe (relative to window start)
    sz_u8_t prefix_first;  ///< UTF-8 prefix for first probe (0-3)
    sz_u8_t prefix_mid;    ///< UTF-8 prefix for mid probe (0-3)
    sz_u8_t prefix_last;   ///< UTF-8 prefix for last probe (0-3)
} sz_script_window_t;

/**
 *  @brief  Complete needle analysis result for all script paths.
 *
 *  Produced by sz_utf8_analyze_needle_() in a single pass through the needle.
 *  Each script path gets its best (longest) safe window with pre-computed probes.
 */
typedef struct {
    sz_script_window_t ascii;
    sz_script_window_t latin1;
    sz_script_window_t cyrillic;
    sz_script_window_t greek;
    sz_script_window_t armenian;
    sz_script_window_t vietnamese;
} sz_needle_analysis_t;

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
 *  @return Safety flags for each script path
 */
SZ_INTERNAL sz_char_script_safety_t sz_utf8_char_script_safety_(sz_rune_t rune, sz_size_t rune_bytes,
                                                                sz_rune_t prev_rune, sz_rune_t next_rune) {
    sz_char_script_safety_t safety = {sz_false_k, sz_false_k, sz_false_k, sz_false_k, sz_false_k, sz_false_k};

    // Helper: lowercase ASCII
    sz_rune_t lower = (rune >= 'A' && rune <= 'Z') ? (rune + 0x20) : rune;
    sz_rune_t lower_prev = (prev_rune >= 'A' && prev_rune <= 'Z') ? (prev_rune + 0x20) : prev_rune;
    sz_rune_t lower_next = (next_rune >= 'A' && next_rune <= 'Z') ? (next_rune + 0x20) : next_rune;

    // Helper: is neighbor ASCII? (explicit conversion for C++ compatibility)
    sz_bool_t prev_ascii = (prev_rune == 0 || prev_rune < 0x80) ? sz_true_k : sz_false_k;
    sz_bool_t next_ascii = (next_rune == 0 || next_rune < 0x80) ? sz_true_k : sz_false_k;

    // ASCII character (1-byte UTF-8)
    if (rune < 0x80) {
        sz_bool_t ascii_safe = sz_false_k;

        // Letters need contextual checks
        if (lower >= 'a' && lower <= 'z') {
            switch (lower) {
                // clang-format off
            // Unconditionally safe: no Unicode chars fold to sequences containing these
            case 'b': case 'c': case 'd': case 'e': case 'g': case 'k':
            case 'm': case 'o': case 'p': case 'q': case 'r': case 'u':
            case 'v': case 'x': case 'z':
                ascii_safe = sz_true_k;
                break;
                // clang-format on

            // Safe if next is ASCII (could be followed by combining chars)
            case 'a':
            case 'h':
            case 'j':
            case 'w':
            case 'y': ascii_safe = next_ascii; break;

            // 'n': ʼn folds to "ʼn" (CA BC + n), unsafe if preceded by non-ASCII
            case 'n': ascii_safe = prev_ascii; break;

            // 'i': İ folds to "i̇", fi ligature exists
            case 'i':
                ascii_safe = (next_ascii && (prev_rune == 0 || lower_prev != 'f')) ? sz_true_k : sz_false_k;
                break;

            // 'l': fl ligature exists
            case 'l': ascii_safe = (prev_rune == 0 || lower_prev != 'f') ? sz_true_k : sz_false_k; break;

            // 't': ẗ has combining diaeresis, st ligature exists
            case 't':
                ascii_safe = (next_ascii && (prev_rune == 0 || lower_prev != 's')) ? sz_true_k : sz_false_k;
                break;

            // 'f': ff/fi/fl ligatures exist
            case 'f':
                ascii_safe = (prev_ascii && next_ascii && lower_prev != 'f' && lower_next != 'f' && lower_next != 'i' &&
                              lower_next != 'l')
                                 ? sz_true_k
                                 : sz_false_k;
                break;

            // 's': ß/ẞ fold to "ss", ﬅ folds to "st"
            case 's':
                ascii_safe = (prev_ascii && next_ascii && lower_prev != 's' && lower_next != 's' && lower_next != 't')
                                 ? sz_true_k
                                 : sz_false_k;
                break;

            default: ascii_safe = sz_true_k; break;
            }
        }
        else {
            // Non-letters (digits, punctuation, whitespace) - always safe
            ascii_safe = sz_true_k;
        }

        // ASCII is safe for ASCII/Latin1/Vietnamese paths IF it passes contextual checks
        // Key fix: Latin1 path uses same ASCII rules as ASCII path (fixes Eszett bug)
        safety.ascii = ascii_safe;
        safety.latin1 = ascii_safe;
        safety.vietnamese = ascii_safe;
        // Script-specific paths (Cyrillic/Greek/Armenian) include ASCII unconditionally
        // because their fold functions handle ASCII (A-Z → a-z) correctly
        safety.cyrillic = sz_true_k;
        safety.greek = sz_true_k;
        safety.armenian = sz_true_k;
        return safety;
    }

    // 2-byte UTF-8 (U+0080 to U+07FF)
    // Must check EXACT ranges that the fold functions handle, not just lead bytes
    if (rune_bytes == 2) {
        sz_u8_t lead = (rune >> 6) | 0xC0;     // Reconstruct lead byte
        sz_u8_t second = (rune & 0x3F) | 0x80; // Reconstruct continuation byte

        // Latin-1 Supplement (C2/C3 lead bytes) - all are safe
        if (lead == 0xC2 || lead == 0xC3) {
            safety.latin1 = sz_true_k;
            safety.vietnamese = sz_true_k; // Vietnamese includes Latin1
        }

        // Cyrillic - check exact ranges handled by sz_cyrillic_fold_zmm_
        // D0 80-BF: U+0400-U+043F (includes uppercase and lowercase)
        // D1 80-9F: U+0440-U+045F (lowercase continuation)
        if ((lead == 0xD0 && second >= 0x80 && second <= 0xBF) || (lead == 0xD1 && second >= 0x80 && second <= 0x9F)) {
            safety.cyrillic = sz_true_k;
        }

        // Greek - check exact ranges handled by sz_greek_fold_zmm_
        // CE 91-A9: uppercase Α-Ω (without accents)
        // CE B1-BF: lowercase α-ο
        // CF 80-89: lowercase π-ω (includes final sigma at 82, sigma at 83)
        if ((lead == 0xCE && second >= 0x91 && second <= 0xA9) || (lead == 0xCE && second >= 0xB1 && second <= 0xBF) ||
            (lead == 0xCF && second >= 0x80 && second <= 0x89)) {
            safety.greek = sz_true_k;
        }

        // Armenian - check exact ranges handled by sz_armenian_fold_zmm_
        // D4 B1-BF: uppercase Ա-Ձ
        // D5 80-96: uppercase Ղ-Ֆ
        // D5 A1-BF: lowercase ա-կ
        // D6 80-86: lowercase հ-ֆ
        if ((lead == 0xD4 && second >= 0xB1 && second <= 0xBF) || (lead == 0xD5 && second >= 0x80 && second <= 0x96) ||
            (lead == 0xD5 && second >= 0xA1 && second <= 0xBF) || (lead == 0xD6 && second >= 0x80 && second <= 0x86)) {
            safety.armenian = sz_true_k;
        }

        return safety;
    }

    // 3-byte UTF-8 (U+0800 to U+FFFF)
    if (rune_bytes == 3) {
        // Vietnamese/Latin Extended Additional (E1 B8-BB range)
        // U+1E00-U+1EFF maps to E1 B8 00 - E1 BB BF
        // EXCLUDE U+1E96-U+1E9F: these expand to base letter + combining diacritics
        //   ẖ (1E96) → h + ̱, ẗ (1E97) → t + ̈, ẘ (1E98) → w + ̊
        //   ẙ (1E99) → y + ̊, ẚ (1E9A) → a + ʾ, ẛ (1E9B) → ſ (long s)
        //   1E9C-1E9F are also problematic or reserved
        if (rune >= 0x1E00 && rune <= 0x1E95) { safety.vietnamese = sz_true_k; }
        if (rune >= 0x1EA0 && rune <= 0x1EFF) { safety.vietnamese = sz_true_k; }
        return safety;
    }

    // 4-byte UTF-8 - currently no fast paths
    return safety;
}

/**
 *  @brief  Initialize a window structure to empty state.
 */
SZ_INTERNAL void sz_script_window_init_(sz_script_window_t *window) {
    window->start = 0;
    window->length = 0;
    window->probe_first = 0;
    window->probe_mid = 0;
    window->probe_last = 0;
    window->prefix_first = 0;
    window->prefix_mid = 0;
    window->prefix_last = 0;
}

/**
 *  @brief  Copy window from src to dst (excluding codepoint_count which is tracked separately).
 */
SZ_INTERNAL void sz_script_window_copy_(sz_script_window_t *dst, sz_script_window_t const *src) {
    dst->start = src->start;
    dst->length = src->length;
    dst->probe_first = src->probe_first;
    dst->probe_mid = src->probe_mid;
    dst->probe_last = src->probe_last;
    dst->prefix_first = src->prefix_first;
    dst->prefix_mid = src->prefix_mid;
    dst->prefix_last = src->prefix_last;
}

/**
 *  @brief  Finalize probe positions for a best window.
 *
 *  Called after finding the best window to compute first/mid/last probe positions
 *  based on codepoint boundaries within the window.
 *
 *  @param window         Window to finalize (modified in place)
 *  @param needle         Original needle bytes
 *  @param codepoint_count Number of codepoints in the window
 */
SZ_INTERNAL void sz_script_window_finalize_probes_(sz_script_window_t *window, sz_cptr_t needle,
                                                   sz_size_t codepoint_count) {
    if (codepoint_count == 0) return;

    // Parse the window to find first, mid, last codepoints
    sz_u8_t const *ptr = (sz_u8_t const *)needle + window->start;
    sz_u8_t const *end = ptr + window->length;

    sz_size_t codepoint_idx = 0;
    sz_size_t mid_idx = codepoint_count / 2;

    sz_size_t first_start = 0, first_len = 0;
    sz_size_t mid_start = 0, mid_len = 0;
    sz_size_t last_start = 0, last_len = 0;

    // Base pointer for computing offsets relative to window start (not needle start)
    sz_u8_t const *window_base = (sz_u8_t const *)needle + window->start;

    while (ptr < end) {
        sz_u8_t lead = *ptr;
        sz_size_t len = 1 + (lead >= 0xC0) + (lead >= 0xE0) + (lead >= 0xF0);
        if (ptr + len > end) len = end - ptr;

        // Offset relative to window start (kernels expect probes relative to window, not needle)
        sz_size_t offset = (sz_size_t)(ptr - window_base);

        if (codepoint_idx == 0) {
            first_start = offset;
            first_len = len;
        }
        if (codepoint_idx == mid_idx) {
            mid_start = offset;
            mid_len = len;
        }
        // Always update last (final iteration wins)
        last_start = offset;
        last_len = len;

        codepoint_idx++;
        ptr += len;
    }

    // Probe position = last byte of codepoint (relative to window start), prefix = bytes before probe
    window->probe_first = first_start + first_len - 1;
    window->prefix_first = (sz_u8_t)(first_len - 1);
    window->probe_mid = mid_start + mid_len - 1;
    window->prefix_mid = (sz_u8_t)(mid_len - 1);
    window->probe_last = last_start + last_len - 1;
    window->prefix_last = (sz_u8_t)(last_len - 1);
}

/**
 *  @brief  Analyze needle in single pass for all script paths.
 *
 *  This function replaces the 6 separate selector functions by analyzing the needle
 *  once and tracking windows for all 6 script paths simultaneously. Each script gets
 *  its best (longest) contiguous safe window with pre-computed 3-point probe positions.
 *
 *  @param needle        Pointer to needle string (original, not folded)
 *  @param needle_length Length in bytes
 *  @return Analysis result with best windows and probes for each script
 */
SZ_INTERNAL sz_needle_analysis_t sz_utf8_analyze_needle_(sz_cptr_t needle, sz_size_t needle_length) {
    sz_needle_analysis_t result;

    // Initialize all best windows to empty
    sz_script_window_init_(&result.ascii);
    sz_script_window_init_(&result.latin1);
    sz_script_window_init_(&result.cyrillic);
    sz_script_window_init_(&result.greek);
    sz_script_window_init_(&result.armenian);
    sz_script_window_init_(&result.vietnamese);

    if (needle_length == 0) return result;

    // Current windows being built (one per script)
    sz_script_window_t curr_ascii, curr_latin1, curr_cyrillic, curr_greek, curr_armenian, curr_vietnamese;
    sz_script_window_init_(&curr_ascii);
    sz_script_window_init_(&curr_latin1);
    sz_script_window_init_(&curr_cyrillic);
    sz_script_window_init_(&curr_greek);
    sz_script_window_init_(&curr_armenian);
    sz_script_window_init_(&curr_vietnamese);

    // Codepoint counts tracked separately (not stored in final struct to save space)
    // curr_*_cp = count for current window being built, best_*_cp = count for best window found
    sz_size_t curr_ascii_cp = 0, curr_latin1_cp = 0, curr_cyrillic_cp = 0;
    sz_size_t curr_greek_cp = 0, curr_armenian_cp = 0, curr_vietnamese_cp = 0;
    sz_size_t best_ascii_cp = 0, best_latin1_cp = 0, best_cyrillic_cp = 0;
    sz_size_t best_greek_cp = 0, best_armenian_cp = 0, best_vietnamese_cp = 0;

    // Track whether each non-ASCII path has seen its specific chars (for validity)
    sz_bool_t has_latin1 = sz_false_k;     // C2/C3 2-byte chars
    sz_bool_t has_cyrillic = sz_false_k;   // D0/D1 2-byte chars
    sz_bool_t has_greek = sz_false_k;      // CE/CF 2-byte chars
    sz_bool_t has_armenian = sz_false_k;   // D4/D5/D6 2-byte chars
    sz_bool_t has_vietnamese = sz_false_k; // E1 B8-BB 3-byte chars

    sz_u8_t const *ptr = (sz_u8_t const *)needle;
    sz_u8_t const *end = ptr + needle_length;
    sz_rune_t prev_rune = 0;

    // Pre-scan: decode next rune for contextual safety
    // We need a 2-codepoint lookahead to get prev and next for each character
    while (ptr < end) {
        // Decode current rune
        sz_u8_t lead = *ptr;
        sz_size_t rune_bytes = 1 + (lead >= 0xC0) + (lead >= 0xE0) + (lead >= 0xF0);
        if (ptr + rune_bytes > end) rune_bytes = end - ptr;

        sz_rune_t rune = 0;
        if (rune_bytes == 1) { rune = lead; }
        else if (rune_bytes == 2) { rune = ((lead & 0x1F) << 6) | (ptr[1] & 0x3F); }
        else if (rune_bytes == 3) { rune = ((lead & 0x0F) << 12) | ((ptr[1] & 0x3F) << 6) | (ptr[2] & 0x3F); }
        else if (rune_bytes == 4) {
            rune = ((lead & 0x07) << 18) | ((ptr[1] & 0x3F) << 12) | ((ptr[2] & 0x3F) << 6) | (ptr[3] & 0x3F);
        }

        // Peek next rune for contextual safety
        sz_rune_t next_rune = 0;
        sz_u8_t const *next_ptr = ptr + rune_bytes;
        if (next_ptr < end) {
            sz_u8_t next_lead = *next_ptr;
            sz_size_t next_bytes = 1 + (next_lead >= 0xC0) + (next_lead >= 0xE0) + (next_lead >= 0xF0);
            if (next_ptr + next_bytes > end) next_bytes = end - next_ptr;

            if (next_bytes == 1) { next_rune = next_lead; }
            else if (next_bytes == 2) { next_rune = ((next_lead & 0x1F) << 6) | (next_ptr[1] & 0x3F); }
            else if (next_bytes == 3) {
                next_rune = ((next_lead & 0x0F) << 12) | ((next_ptr[1] & 0x3F) << 6) | (next_ptr[2] & 0x3F);
            }
            else if (next_bytes == 4) {
                next_rune = ((next_lead & 0x07) << 18) | ((next_ptr[1] & 0x3F) << 12) | ((next_ptr[2] & 0x3F) << 6) |
                            (next_ptr[3] & 0x3F);
            }
        }

        sz_size_t byte_offset = (sz_size_t)(ptr - (sz_u8_t const *)needle);

        // Track which scripts this character belongs to (for window validity)
        if (lead == 0xC2 || lead == 0xC3) has_latin1 = sz_true_k;
        if (lead == 0xD0 || lead == 0xD1) has_cyrillic = sz_true_k;
        if (lead == 0xCE || lead == 0xCF) has_greek = sz_true_k;
        if (lead == 0xD4 || lead == 0xD5 || lead == 0xD6) has_armenian = sz_true_k;
        if (lead == 0xE1 && rune_bytes >= 2 && (ptr[1] >= 0xB8 && ptr[1] <= 0xBB)) has_vietnamese = sz_true_k;

        // Get safety profile for this character
        sz_char_script_safety_t safety = sz_utf8_char_script_safety_(rune, rune_bytes, prev_rune, next_rune);

// Helper macro to update window for a script
#define sz_update_window_(script)                                                  \
    do {                                                                           \
        if (safety.script) {                                                       \
            /* Safe: extend current window */                                      \
            if (curr_##script##_cp == 0) { curr_##script.start = byte_offset; }    \
            curr_##script.length = byte_offset + rune_bytes - curr_##script.start; \
            curr_##script##_cp++;                                                  \
        }                                                                          \
        else {                                                                     \
            /* Unsafe: finalize current, update best if longer, reset current */   \
            if (curr_##script.length > result.script.length) {                     \
                sz_script_window_copy_(&result.script, &curr_##script);            \
                best_##script##_cp = curr_##script##_cp;                           \
            }                                                                      \
            sz_script_window_init_(&curr_##script);                                \
            curr_##script##_cp = 0;                                                \
        }                                                                          \
    } while (0)

        sz_update_window_(ascii);
        sz_update_window_(latin1);
        sz_update_window_(cyrillic);
        sz_update_window_(greek);
        sz_update_window_(armenian);
        sz_update_window_(vietnamese);

#undef sz_update_window_

        prev_rune = rune;
        ptr += rune_bytes;
    }

    // Finalize remaining current windows (end of needle reached)
#define sz_finalize_window_(script)                                                    \
    do {                                                                               \
        if (curr_##script.length > result.script.length) {                             \
            sz_script_window_copy_(&result.script, &curr_##script);                    \
            best_##script##_cp = curr_##script##_cp;                                   \
        }                                                                              \
        sz_script_window_finalize_probes_(&result.script, needle, best_##script##_cp); \
    } while (0)

    sz_finalize_window_(ascii);
    sz_finalize_window_(latin1);
    sz_finalize_window_(cyrillic);
    sz_finalize_window_(greek);
    sz_finalize_window_(armenian);
    sz_finalize_window_(vietnamese);

#undef sz_finalize_window_

    // Invalidate non-ASCII windows that contain no script-specific chars
    // A window is only valid if it contains at least one char from that script
    if (!has_latin1) { result.latin1.length = 0; }
    if (!has_cyrillic) { result.cyrillic.length = 0; }
    if (!has_greek) { result.greek.length = 0; }
    if (!has_armenian) { result.armenian.length = 0; }
    if (!has_vietnamese) { result.vietnamese.length = 0; }

    return result;
}

/**
 *  @brief  Latin-1 case-insensitive substring search using AVX-512.
 *
 *  Strategy: Use 3-point Raita filter on folded needle bytes, load haystack at probe offsets,
 *  fold and compare to find candidates, then verify full safe slice match.
 *  Steps by 62 bytes to ensure complete coverage at chunk boundaries.
 *
 *  @param[in] haystack Pointer to the haystack string.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] needle Pointer to the needle string.
 *  @param[in] needle_length Length of the needle in bytes.
 *  @param[in] window Pre-computed safe window with probe positions.
 *  @param[out] matched_length Length of the match in haystack bytes.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_latin1_ice_(sz_cptr_t haystack, sz_size_t haystack_length,
                                                                sz_cptr_t needle, sz_size_t needle_length,
                                                                sz_script_window_t const *window,
                                                                sz_size_t *matched_length) {

    sz_cptr_t safe_start = needle + window->start;
    sz_size_t safe_length = window->length;

    // Clamp safe slice to single ZMM register, ensuring we don't clip a 2-byte character
    if (safe_length > 64) {
        safe_length = 64;
        // If byte 63 is a C2/C3 lead byte, back off to not split the 2-byte sequence
        sz_u8_t byte63 = (sz_u8_t)safe_start[63];
        if (byte63 == 0xC2 || byte63 == 0xC3) safe_length = 63;
    }

    sz_size_t const safe_offset = window->start;
    sz_size_t const head_length = safe_offset;
    sz_size_t const tail_length = needle_length - safe_offset - safe_length;

    // Pre-fold entire safe slice for vectorized verification, to further reduce calls
    // to serial `sz_utf8_verify_case_insensitive_match_`
    __mmask64 const safe_mask = sz_u64_mask_until_(safe_length);
    __m512i const needle_safe_folded = sz_latin1_fold_zmm_(_mm512_maskz_loadu_epi8(safe_mask, safe_start));

    // Use pre-computed probe positions from analysis (last byte of first/mid/last codepoints)
    sz_size_t const load_first = window->probe_first - window->prefix_first;
    sz_size_t const load_mid = window->probe_mid - window->prefix_mid;
    sz_size_t const load_last = window->probe_last - window->prefix_last;

    // Extract folded probe bytes for comparison
    char needle_safe_folded_bytes[64];
    _mm512_mask_storeu_epi8(needle_safe_folded_bytes, safe_mask, needle_safe_folded);

    // Broadcast needle probe bytes for comparisons with pulled haystack data
    sz_u512_vec_t probe_first_vec, probe_mid_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_safe_folded_bytes[window->probe_first]);
    probe_mid_vec.zmm = _mm512_set1_epi8(needle_safe_folded_bytes[window->probe_mid]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_safe_folded_bytes[window->probe_last]);

    // Main loop - step by 62 bytes to handle 2-byte chars at boundaries plus 1-byte lookback
    sz_u512_vec_t haystack_first_vec, haystack_mid_vec, haystack_last_vec;
    for (; haystack_length >= needle_length + 64; haystack += 62, haystack_length -= 62) {
        // Load haystack parts at adjusted offsets (aligned to codepoint start)
        haystack_first_vec.zmm = sz_latin1_fold_zmm_(_mm512_loadu_si512(haystack + safe_offset + load_first));
        haystack_mid_vec.zmm = sz_latin1_fold_zmm_(_mm512_loadu_si512(haystack + safe_offset + load_mid));
        haystack_last_vec.zmm = sz_latin1_fold_zmm_(_mm512_loadu_si512(haystack + safe_offset + load_last));

        // 3-point Raita filter: find positions where all 3 probes match
        // Shift masks by prefix to align probe byte position
        __mmask64 match_first =
            _mm512_cmpeq_epi8_mask(haystack_first_vec.zmm, probe_first_vec.zmm) >> window->prefix_first;
        __mmask64 match_mid = _mm512_cmpeq_epi8_mask(haystack_mid_vec.zmm, probe_mid_vec.zmm) >> window->prefix_mid;
        __mmask64 match_last = _mm512_cmpeq_epi8_mask(haystack_last_vec.zmm, probe_last_vec.zmm) >> window->prefix_last;
        sz_u64_t matches = _kand_mask64(_kand_mask64(match_first, match_mid), match_last);

        // Exclude positions 62-63 - handle in next iteration to avoid incomplete match
        matches &= 0x3FFFFFFFFFFFFFFFULL;

        // Verify all match candidates found
        while (matches) {
            int const idx = sz_u64_ctz(matches);
            sz_cptr_t const candidate = haystack + idx;
            __m512i const haystack_safe =
                sz_latin1_fold_zmm_(_mm512_maskz_loadu_epi8(safe_mask, candidate + safe_offset));
            if (_mm512_mask_cmpneq_epi8_mask(safe_mask, haystack_safe, needle_safe_folded) == 0) {
                sz_bool_t same_head = //
                    head_length ? sz_utf8_verify_case_insensitive_match_(needle, head_length, candidate,
                                                                         candidate + head_length)
                                : sz_true_k;
                sz_bool_t same_tail = //
                    tail_length ? sz_utf8_verify_case_insensitive_match_(safe_start + safe_length, tail_length,
                                                                         candidate + safe_offset + safe_length,
                                                                         candidate + needle_length)
                                : sz_true_k;
                if (same_head && same_tail) {
                    *matched_length = needle_length;
                    return candidate;
                }
            }
            matches &= matches - 1;
        }
    }

    // Tail: use masked loads for remaining positions (single iteration)
    if (haystack_length >= needle_length) {
        sz_size_t tail_positions = haystack_length - needle_length + 1;
        // For 2-byte scripts, we need to load at least 2 bytes for the fold function to work correctly.
        // The fold function needs to see lead+continuation pairs. We also add prefix for the shift.
        sz_size_t avail_first = haystack_length - safe_offset - load_first;
        sz_size_t avail_mid = haystack_length - safe_offset - load_mid;
        sz_size_t avail_last = haystack_length - safe_offset - load_last;
        sz_size_t need_first = tail_positions + window->prefix_first;
        sz_size_t need_mid = tail_positions + window->prefix_mid;
        sz_size_t need_last = tail_positions + window->prefix_last;
        __mmask64 tail_mask_first = sz_u64_mask_until_(need_first < avail_first ? need_first : avail_first);
        __mmask64 tail_mask_mid = sz_u64_mask_until_(need_mid < avail_mid ? need_mid : avail_mid);
        __mmask64 tail_mask_last = sz_u64_mask_until_(need_last < avail_last ? need_last : avail_last);
        haystack_first_vec.zmm =
            sz_latin1_fold_zmm_(_mm512_maskz_loadu_epi8(tail_mask_first, haystack + safe_offset + load_first));
        haystack_mid_vec.zmm =
            sz_latin1_fold_zmm_(_mm512_maskz_loadu_epi8(tail_mask_mid, haystack + safe_offset + load_mid));
        haystack_last_vec.zmm =
            sz_latin1_fold_zmm_(_mm512_maskz_loadu_epi8(tail_mask_last, haystack + safe_offset + load_last));

        __mmask64 match_first =
            _mm512_cmpeq_epi8_mask(haystack_first_vec.zmm, probe_first_vec.zmm) >> window->prefix_first;
        __mmask64 match_mid = _mm512_cmpeq_epi8_mask(haystack_mid_vec.zmm, probe_mid_vec.zmm) >> window->prefix_mid;
        __mmask64 match_last = _mm512_cmpeq_epi8_mask(haystack_last_vec.zmm, probe_last_vec.zmm) >> window->prefix_last;
        sz_u64_t matches =
            _kand_mask64(_kand_mask64(match_first, match_mid), match_last) & sz_u64_mask_until_(tail_positions);

        while (matches) {
            int const idx = sz_u64_ctz(matches);
            sz_cptr_t const candidate = haystack + idx;
            __m512i const haystack_safe =
                sz_latin1_fold_zmm_(_mm512_maskz_loadu_epi8(safe_mask, candidate + safe_offset));
            if (_mm512_mask_cmpneq_epi8_mask(safe_mask, haystack_safe, needle_safe_folded) == 0) {
                sz_bool_t same_head = head_length ? sz_utf8_verify_case_insensitive_match_(
                                                        needle, head_length, candidate, candidate + head_length)
                                                  : sz_true_k;
                sz_bool_t same_tail =
                    tail_length ? sz_utf8_verify_case_insensitive_match_(safe_start + safe_length, tail_length,
                                                                         candidate + safe_offset + safe_length,
                                                                         candidate + needle_length)
                                : sz_true_k;
                if (same_head && same_tail) {
                    *matched_length = needle_length;
                    return candidate;
                }
            }
            matches &= matches - 1;
        }
    }

    return SZ_NULL_CHAR;
}

/**
 *  @brief  Helper function to find the longest slice of the needle composed exclusively of "safe" Cyrillic characters
 *          so that we can implement a faster substring search only unpacking and folding that subrange of 2-byte
 *          UTF-8 characters, while skipping the rest.
 *
 *  Basic Cyrillic (U+0400-U+045F) uses 2-byte UTF-8 with predictable case folding:
 *  - D0 80-BF: covers U+0400-U+043F (uppercase + first half of lowercase)
 *  - D1 80-9F: covers U+0440-U+045F (second half of lowercase)
 *  - D1 A0+: Cyrillic Extended-A with +1 folding - unsafe for this kernel
 *
 *  ASCII bytes are always safe since they fold independently.
 *
 *  @return sz_cptr_t Pointer to the start of the longest safe Cyrillic slice within the needle.
 *  @retval NULL No safe Cyrillic slice found.
 *  @param[in] needle Pointer to the UTF-8 needle string.
 *  @param[in] needle_length Length of the needle string in bytes.
 *  @param[out] longest_range_length Length of the longest safe Cyrillic slice found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_ice_can_use_cyrillic_path_(sz_cptr_t needle,
                                                                               sz_size_t needle_length,
                                                                               sz_size_t *longest_range_length) {
    sz_cptr_t best_start = SZ_NULL_CHAR;
    sz_size_t best_length = 0;
    sz_cptr_t current_start = needle;
    sz_size_t current_length = 0;

    for (sz_size_t i = 0; i < needle_length;) {
        sz_u8_t byte = (sz_u8_t)needle[i];

        // ASCII byte - always safe in Cyrillic context
        if (byte < 0x80) {
            if (current_length == 0) current_start = needle + i;
            current_length++;
            i++;
            continue;
        }

        // D0 lead byte: second byte 80-BF is safe (U+0400-U+043F)
        if (byte == 0xD0 && i + 1 < needle_length) {
            sz_u8_t second = (sz_u8_t)needle[i + 1];
            if (second >= 0x80 && second <= 0xBF) {
                if (current_length == 0) current_start = needle + i;
                current_length += 2;
                i += 2;
                continue;
            }
        }

        // D1 lead byte: second byte 80-9F is safe (U+0440-U+045F), A0+ is unsafe
        if (byte == 0xD1 && i + 1 < needle_length) {
            sz_u8_t second = (sz_u8_t)needle[i + 1];
            if (second >= 0x80 && second <= 0x9F) {
                if (current_length == 0) current_start = needle + i;
                current_length += 2;
                i += 2;
                continue;
            }
        }

        // Not safe - finalize current slice
        if (current_length > best_length) {
            best_length = current_length;
            best_start = current_start;
        }
        current_length = 0;

        // Skip this byte/sequence properly
        if (byte < 0x80) { i++; }
        else if ((byte & 0xE0) == 0xC0) { i += 2; }
        else if ((byte & 0xF0) == 0xE0) { i += 3; }
        else if ((byte & 0xF8) == 0xF0) { i += 4; }
        else { i++; }
        current_start = needle + i;
    }

    // Check final slice
    if (current_length > best_length) {
        best_length = current_length;
        best_start = current_start;
    }

    *longest_range_length = best_length;
    return best_start;
}

/**
 *  @brief  Fold a ZMM register using Cyrillic case folding rules.
 *
 *  Cyrillic uppercase А-Я (U+0410-U+042F): UTF-8 D0 90-AF (32 letters)
 *  Cyrillic lowercase а-я (U+0430-U+044F): UTF-8 D0 B0-BF, D1 80-8F
 *
 *  Folding maps all to canonical uppercase form (90-AF range):
 *  - After D0: lowercase B0-BF → subtract 0x20
 *  - After D1: lowercase 80-8F → add 0x20
 *  - Lead bytes D0/D1 → D0
 *  - ASCII A-Z → add 0x20
 *
 *  @param[in] source The source ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_cyrillic_fold_zmm_(__m512i source) {
    __m512i const d0_vec = _mm512_set1_epi8((char)0xD0);
    __m512i const d1_vec = _mm512_set1_epi8((char)0xD1);
    __m512i const xb0_vec = _mm512_set1_epi8((char)0xB0);
    __m512i const x80_vec = _mm512_set1_epi8((char)0x80);
    __m512i const x20_vec = _mm512_set1_epi8(0x20);
    __m512i const x10_vec = _mm512_set1_epi8(0x10);
    __m512i const a_upper_vec = _mm512_set1_epi8('A');
    __m512i const range_26_vec = _mm512_set1_epi8(26);

    // Step 1: Detect D0/D1 lead bytes
    __mmask64 is_d0 = _mm512_cmpeq_epi8_mask(source, d0_vec);
    __mmask64 is_d1 = _mm512_cmpeq_epi8_mask(source, d1_vec);
    __mmask64 is_lead = is_d0 | is_d1;

    // Step 2: Detect positions after D0 and after D1
    __mmask64 is_after_d0 = is_d0 << 1;
    __mmask64 is_after_d1 = is_d1 << 1;

    // Step 3: Fold Cyrillic lowercase after D0 (second bytes B0-BF → subtract 0x20)
    // Range check: (byte - 0xB0) < 0x10 means byte in [0xB0, 0xBF]
    __mmask64 is_cyrillic_d0_lower =
        _mm512_mask_cmplt_epu8_mask(is_after_d0, _mm512_sub_epi8(source, xb0_vec), x10_vec);

    // Step 4: Fold Cyrillic lowercase after D1 (second bytes 80-8F → add 0x20)
    // Range check: (byte - 0x80) < 0x10 means byte in [0x80, 0x8F]
    __mmask64 is_cyrillic_d1_lower =
        _mm512_mask_cmplt_epu8_mask(is_after_d1, _mm512_sub_epi8(source, x80_vec), x10_vec);

    // Step 5: Fold ASCII uppercase (A-Z) → add 0x20
    __mmask64 is_ascii_upper = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source, a_upper_vec), range_26_vec);

    // Step 6: Apply Cyrillic D0 lowercase folding (subtract 0x20)
    __m512i result = _mm512_mask_sub_epi8(source, is_cyrillic_d0_lower, source, x20_vec);

    // Step 7: Apply Cyrillic D1 lowercase folding (add 0x20)
    result = _mm512_mask_add_epi8(result, is_cyrillic_d1_lower, result, x20_vec);

    // Step 8: Apply ASCII folding (add 0x20)
    result = _mm512_mask_add_epi8(result, is_ascii_upper, result, x20_vec);

    // Step 9: Normalize lead bytes D0/D1 → D0
    result = _mm512_mask_mov_epi8(result, is_lead, d0_vec);

    return result;
}

/**
 *  @brief  Cyrillic case-insensitive substring search using AVX-512.
 *
 *  Strategy: Use 3-point Raita filter on folded needle bytes, load haystack at probe offsets,
 *  fold and compare to find candidates, then verify full safe slice match.
 *  Steps by 62 bytes to ensure complete coverage at chunk boundaries.
 *
 *  @param[in] haystack Pointer to the haystack string.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] needle Pointer to the needle string.
 *  @param[in] needle_length Length of the needle in bytes.
 *  @param[in] window Pre-computed safe window with probe positions.
 *  @param[out] matched_length Length of the match in haystack bytes.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_cyrillic_ice_(sz_cptr_t haystack, sz_size_t haystack_length,
                                                                  sz_cptr_t needle, sz_size_t needle_length,
                                                                  sz_script_window_t const *window,
                                                                  sz_size_t *matched_length) {

    sz_cptr_t safe_start = needle + window->start;
    sz_size_t safe_length = window->length;

    // Clamp safe slice to single ZMM register, ensuring we don't clip a 2-byte character
    if (safe_length > 64) {
        safe_length = 64;
        // If byte 63 is a D0/D1 lead byte, back off to not split the 2-byte sequence
        sz_u8_t byte63 = (sz_u8_t)safe_start[63];
        if (byte63 == 0xD0 || byte63 == 0xD1) safe_length = 63;
    }

    sz_size_t const safe_offset = window->start;
    sz_size_t const head_length = safe_offset;
    sz_size_t const tail_length = needle_length - safe_offset - safe_length;

    // Pre-fold entire safe slice for vectorized verification
    __mmask64 const safe_mask = sz_u64_mask_until_(safe_length);
    __m512i const needle_safe_folded = sz_cyrillic_fold_zmm_(_mm512_maskz_loadu_epi8(safe_mask, safe_start));

    // Use pre-computed probe positions from analysis (last byte of first/mid/last codepoints)
    sz_size_t const load_first = window->probe_first - window->prefix_first;
    sz_size_t const load_mid = window->probe_mid - window->prefix_mid;
    sz_size_t const load_last = window->probe_last - window->prefix_last;

    // Extract folded probe bytes for comparison
    char needle_safe_folded_bytes[64];
    _mm512_mask_storeu_epi8(needle_safe_folded_bytes, safe_mask, needle_safe_folded);

    // Broadcast needle probe bytes for comparisons
    sz_u512_vec_t probe_first_vec, probe_mid_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_safe_folded_bytes[window->probe_first]);
    probe_mid_vec.zmm = _mm512_set1_epi8(needle_safe_folded_bytes[window->probe_mid]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_safe_folded_bytes[window->probe_last]);

    // Main loop - step by 62 bytes to handle 2-byte chars at boundaries plus 1-byte lookback
    sz_u512_vec_t haystack_first_vec, haystack_mid_vec, haystack_last_vec;
    for (; haystack_length >= needle_length + 64; haystack += 62, haystack_length -= 62) {
        // Load haystack parts at adjusted offsets (aligned to codepoint start)
        haystack_first_vec.zmm = sz_cyrillic_fold_zmm_(_mm512_loadu_si512(haystack + safe_offset + load_first));
        haystack_mid_vec.zmm = sz_cyrillic_fold_zmm_(_mm512_loadu_si512(haystack + safe_offset + load_mid));
        haystack_last_vec.zmm = sz_cyrillic_fold_zmm_(_mm512_loadu_si512(haystack + safe_offset + load_last));

        // Compare and shift masks by prefix to align probe byte position
        __mmask64 match_first =
            _mm512_cmpeq_epi8_mask(haystack_first_vec.zmm, probe_first_vec.zmm) >> window->prefix_first;
        __mmask64 match_mid = _mm512_cmpeq_epi8_mask(haystack_mid_vec.zmm, probe_mid_vec.zmm) >> window->prefix_mid;
        __mmask64 match_last = _mm512_cmpeq_epi8_mask(haystack_last_vec.zmm, probe_last_vec.zmm) >> window->prefix_last;

        // 3-point Raita filter: find positions where all 3 probes match
        sz_u64_t matches = _kand_mask64(_kand_mask64(match_first, match_mid), match_last);

        // Exclude positions 62-63 - handle in next iteration to avoid incomplete match
        matches &= 0x3FFFFFFFFFFFFFFFULL;

        // Verify all match candidates found
        while (matches) {
            int const idx = sz_u64_ctz(matches);
            sz_cptr_t const candidate = haystack + idx;
            __m512i const haystack_safe =
                sz_cyrillic_fold_zmm_(_mm512_maskz_loadu_epi8(safe_mask, candidate + safe_offset));
            if (_mm512_mask_cmpneq_epi8_mask(safe_mask, haystack_safe, needle_safe_folded) == 0) {
                sz_bool_t same_head = //
                    head_length ? sz_utf8_verify_case_insensitive_match_(needle, head_length, candidate,
                                                                         candidate + head_length)
                                : sz_true_k;
                sz_bool_t same_tail = //
                    tail_length ? sz_utf8_verify_case_insensitive_match_(safe_start + safe_length, tail_length,
                                                                         candidate + safe_offset + safe_length,
                                                                         candidate + needle_length)
                                : sz_true_k;
                if (same_head && same_tail) {
                    *matched_length = needle_length;
                    return candidate;
                }
            }
            matches &= matches - 1;
        }
    }

    // Tail: use masked loads for remaining positions (single iteration)
    if (haystack_length >= needle_length) {
        sz_size_t tail_positions = haystack_length - needle_length + 1;
        // For 2-byte scripts, we need to load at least 2 bytes for the fold function to work correctly.
        // The fold function needs to see lead+continuation pairs. We also add prefix for the shift.
        sz_size_t avail_first = haystack_length - safe_offset - load_first;
        sz_size_t avail_mid = haystack_length - safe_offset - load_mid;
        sz_size_t avail_last = haystack_length - safe_offset - load_last;
        sz_size_t need_first = tail_positions + window->prefix_first;
        sz_size_t need_mid = tail_positions + window->prefix_mid;
        sz_size_t need_last = tail_positions + window->prefix_last;
        __mmask64 tail_mask_first = sz_u64_mask_until_(need_first < avail_first ? need_first : avail_first);
        __mmask64 tail_mask_mid = sz_u64_mask_until_(need_mid < avail_mid ? need_mid : avail_mid);
        __mmask64 tail_mask_last = sz_u64_mask_until_(need_last < avail_last ? need_last : avail_last);

        haystack_first_vec.zmm =
            sz_cyrillic_fold_zmm_(_mm512_maskz_loadu_epi8(tail_mask_first, haystack + safe_offset + load_first));
        haystack_mid_vec.zmm =
            sz_cyrillic_fold_zmm_(_mm512_maskz_loadu_epi8(tail_mask_mid, haystack + safe_offset + load_mid));
        haystack_last_vec.zmm =
            sz_cyrillic_fold_zmm_(_mm512_maskz_loadu_epi8(tail_mask_last, haystack + safe_offset + load_last));

        // Compare and shift masks by prefix
        __mmask64 match_first =
            _mm512_cmpeq_epi8_mask(haystack_first_vec.zmm, probe_first_vec.zmm) >> window->prefix_first;
        __mmask64 match_mid = _mm512_cmpeq_epi8_mask(haystack_mid_vec.zmm, probe_mid_vec.zmm) >> window->prefix_mid;
        __mmask64 match_last = _mm512_cmpeq_epi8_mask(haystack_last_vec.zmm, probe_last_vec.zmm) >> window->prefix_last;

        sz_u64_t matches =
            _kand_mask64(_kand_mask64(match_first, match_mid), match_last) & sz_u64_mask_until_(tail_positions);

        while (matches) {
            int const idx = sz_u64_ctz(matches);
            sz_cptr_t const candidate = haystack + idx;
            __m512i const haystack_safe =
                sz_cyrillic_fold_zmm_(_mm512_maskz_loadu_epi8(safe_mask, candidate + safe_offset));
            if (_mm512_mask_cmpneq_epi8_mask(safe_mask, haystack_safe, needle_safe_folded) == 0) {
                sz_bool_t same_head = head_length ? sz_utf8_verify_case_insensitive_match_(
                                                        needle, head_length, candidate, candidate + head_length)
                                                  : sz_true_k;
                sz_bool_t same_tail =
                    tail_length ? sz_utf8_verify_case_insensitive_match_(safe_start + safe_length, tail_length,
                                                                         candidate + safe_offset + safe_length,
                                                                         candidate + needle_length)
                                : sz_true_k;
                if (same_head && same_tail) {
                    *matched_length = needle_length;
                    return candidate;
                }
            }
            matches &= matches - 1;
        }
    }

    return SZ_NULL_CHAR;
}

/**
 *  @brief  Helper function to find the longest slice of the needle composed exclusively of "safe" Armenian characters
 *          so that we can implement a faster substring search only unpacking and folding that subrange of 2-byte
 *          UTF-8 characters, while skipping the rest.
 *
 *  Armenian (U+0531-U+0586) uses 2-byte UTF-8 with predictable case folding:
 *  - D4 B1-BF (Ա-Խ): uppercase U+0531-053F, folds via +0x30
 *  - D5 80-96 (Ծ-Ֆ): uppercase U+0540-0556, folds via +0x30 (wrapping to D5/D6)
 *  - D5 A1-BF ( delays-տ): lowercase U+0561-057F
 *  - D6 80-86 (delays-ֆ): lowercase U+0580-0586
 *
 *  ASCII bytes are always safe since they fold independently.
 *
 *  @return sz_cptr_t Pointer to the start of the longest safe Armenian slice within the needle.
 *  @retval NULL No safe Armenian slice found.
 *  @param[in] needle Pointer to the UTF-8 needle string.
 *  @param[in] needle_length Length of the needle string in bytes.
 *  @param[out] longest_range_length Length of the longest safe Armenian slice found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_ice_can_use_armenian_path_(sz_cptr_t needle,
                                                                               sz_size_t needle_length,
                                                                               sz_size_t *longest_range_length) {
    sz_cptr_t best_start = SZ_NULL_CHAR;
    sz_size_t best_length = 0;
    sz_cptr_t current_start = needle;
    sz_size_t current_length = 0;

    for (sz_size_t i = 0; i < needle_length;) {
        sz_u8_t byte = (sz_u8_t)needle[i];

        // ASCII byte - always safe in Armenian context
        if (byte < 0x80) {
            if (current_length == 0) current_start = needle + i;
            current_length++;
            i++;
            continue;
        }

        // D4 lead byte: Armenian uppercase first part
        if (byte == 0xD4 && i + 1 < needle_length) {
            sz_u8_t second = (sz_u8_t)needle[i + 1];
            // D4 B1-BF: uppercase Ա-Խ (U+0531-053F)
            if (second >= 0xB1 && second <= 0xBF) {
                if (current_length == 0) current_start = needle + i;
                current_length += 2;
                i += 2;
                continue;
            }
        }

        // D5 lead byte: Armenian uppercase second part + lowercase first part
        if (byte == 0xD5 && i + 1 < needle_length) {
            sz_u8_t second = (sz_u8_t)needle[i + 1];
            // D5 80-96: uppercase Ծ-Ֆ (U+0540-0556)
            // D5 A1-BF: lowercase delays-տ (U+0561-057F)
            if ((second >= 0x80 && second <= 0x96) || (second >= 0xA1 && second <= 0xBF)) {
                if (current_length == 0) current_start = needle + i;
                current_length += 2;
                i += 2;
                continue;
            }
        }

        // D6 lead byte: Armenian lowercase continuation
        if (byte == 0xD6 && i + 1 < needle_length) {
            sz_u8_t second = (sz_u8_t)needle[i + 1];
            // D6 80-86: lowercase delays-ֆ (U+0580-0586)
            if (second >= 0x80 && second <= 0x86) {
                if (current_length == 0) current_start = needle + i;
                current_length += 2;
                i += 2;
                continue;
            }
        }

        // Not safe - finalize current slice
        if (current_length > best_length) {
            best_length = current_length;
            best_start = current_start;
        }
        current_length = 0;

        // Skip this byte/sequence properly
        if (byte < 0x80) { i++; }
        else if ((byte & 0xE0) == 0xC0) { i += 2; }
        else if ((byte & 0xF0) == 0xE0) { i += 3; }
        else if ((byte & 0xF8) == 0xF0) { i += 4; }
        else { i++; }
        current_start = needle + i;
    }

    // Check final slice
    if (current_length > best_length) {
        best_length = current_length;
        best_start = current_start;
    }

    *longest_range_length = best_length;
    return best_start;
}

/**
 *  @brief  Fold a ZMM register using Armenian case folding rules.
 *
 *  Armenian uppercase Ա-Ֆ (U+0531-U+0556): UTF-8 D4 B1-BF, D5 80-96
 *  Armenian lowercase ա-ֆ (U+0561-U+0586): UTF-8 D5 A1-BF, D6 80-86
 *
 *  Folding adds +0x30 to uppercase second bytes, with lead byte changes for boundary crossing:
 *  - D4 B1-BF → D5 81-8F (lead D4→D5, second +0x30 wrapping)
 *  - D5 80-8F → D5 B0-BF (same lead, second +0x30)
 *  - D5 90-96 → D6 80-86 (lead D5→D6, second +0x30-0x10 for wrap)
 *
 *  @param[in] source The source ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_armenian_fold_zmm_(__m512i source) {
    __m512i const d4_vec = _mm512_set1_epi8((char)0xD4);
    __m512i const d5_vec = _mm512_set1_epi8((char)0xD5);
    __m512i const d6_vec = _mm512_set1_epi8((char)0xD6);
    __m512i const xb1_vec = _mm512_set1_epi8((char)0xB1);
    __m512i const x0f_vec = _mm512_set1_epi8(0x0F);
    __m512i const x80_vec = _mm512_set1_epi8((char)0x80);
    __m512i const x30_vec = _mm512_set1_epi8(0x30);
    __m512i const a_upper_vec = _mm512_set1_epi8('A');
    __m512i const range_26_vec = _mm512_set1_epi8(26);

    // Step 1: Detect D4/D5/D6 lead bytes
    __mmask64 is_d4 = _mm512_cmpeq_epi8_mask(source, d4_vec);
    __mmask64 is_d5 = _mm512_cmpeq_epi8_mask(source, d5_vec);
    __mmask64 is_d6 = _mm512_cmpeq_epi8_mask(source, d6_vec);

    // Step 2: Detect positions after D4 and after D5
    __mmask64 is_after_d4 = is_d4 << 1;
    __mmask64 is_after_d5 = is_d5 << 1;

    // Step 3: Armenian uppercase after D4 (B1-BF → fold to D5 A1-AF)
    // Range check: (byte - 0xB1) < 0x0F means byte in [0xB1, 0xBF]
    // Unicode: U+0531-U+053F (D4 B1-BF) → U+0561-U+056F (D5 A1-AF)
    // Mapping: lead D4→D5, continuation subtract 0x10 (B1→A1, etc.)
    __mmask64 is_armenian_d4_upper =
        _mm512_mask_cmplt_epu8_mask(is_after_d4, _mm512_sub_epi8(source, xb1_vec), x0f_vec);

    // Step 4: Armenian uppercase after D5 (80-96 → fold to D5 B0-BF or D6 80-86)
    // Unicode: U+0540-U+0556 (D5 80-96) → U+0570-U+0586 (D5 B0-BF, D6 80-86)
    // Mapping: continuation add 0x30, lead stays D5 for 80-8F, changes to D6 for 90-96
    // Range check: (byte - 0x80) < 0x17 means byte in [0x80, 0x96]
    __m512i const x17_vec = _mm512_set1_epi8(0x17);
    __m512i const x10_vec = _mm512_set1_epi8(0x10);
    __mmask64 is_armenian_d5_upper =
        _mm512_mask_cmplt_epu8_mask(is_after_d5, _mm512_sub_epi8(source, x80_vec), x17_vec);

    // Step 5: Detect which D5 uppercase chars will wrap to D6 (byte >= 0x90)
    // D5 90-96 + 0x30 = D5 C0-C6, which is invalid → wrap to D6 80-86 (subtract 0x40)
    __m512i const x90_vec = _mm512_set1_epi8((char)0x90);
    __mmask64 is_d5_will_wrap = _mm512_mask_cmpge_epu8_mask(is_armenian_d5_upper, source, x90_vec);

    // Step 6: Fold ASCII uppercase (A-Z) → add 0x20
    __mmask64 is_ascii_upper = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source, a_upper_vec), range_26_vec);

    // Step 7: Apply Armenian D4 folding (subtract 0x10 from continuation, later change lead)
    __m512i result = _mm512_mask_sub_epi8(source, is_armenian_d4_upper, source, x10_vec);

    // Step 8: Apply Armenian D5 folding (add 0x30 to continuation)
    result = _mm512_mask_add_epi8(result, is_armenian_d5_upper, result, x30_vec);

    // Step 9: Fix wrapped D5 bytes (subtract 0x40 to bring C0-C6 down to 80-86)
    __m512i const x40_vec = _mm512_set1_epi8(0x40);
    result = _mm512_mask_sub_epi8(result, is_d5_will_wrap, result, x40_vec);

    // Step 10: For D4 uppercase, change lead byte D4 → D5
    __mmask64 is_lead_d4_to_d5 = is_armenian_d4_upper >> 1;
    result = _mm512_mask_mov_epi8(result, is_lead_d4_to_d5, d5_vec);

    // Step 11: For D5 wrapping uppercase (90-96), change lead byte D5 → D6
    __mmask64 is_lead_d5_to_d6 = is_d5_will_wrap >> 1;
    result = _mm512_mask_mov_epi8(result, is_lead_d5_to_d6, d6_vec);

    // Step 12: Apply ASCII folding (add 0x20)
    __m512i const x20_vec = _mm512_set1_epi8(0x20);
    result = _mm512_mask_add_epi8(result, is_ascii_upper, result, x20_vec);

    // Step 13: Preserve lead bytes that didn't change
    // D5 lead bytes should stay D5 (original D5 not wrapping + converted from D4)
    // D6 lead bytes should stay D6 (original D6 + converted from D5 wrapping)
    __mmask64 is_keep_d5 = (is_d5 & ~is_lead_d5_to_d6) | is_lead_d4_to_d5;
    __mmask64 is_keep_d6 = is_d6 | is_lead_d5_to_d6;
    result = _mm512_mask_mov_epi8(result, is_keep_d5, d5_vec);
    result = _mm512_mask_mov_epi8(result, is_keep_d6, d6_vec);

    return result;
}

/**
 *  @brief  Armenian case-insensitive substring search using AVX-512.
 *
 *  Strategy: Use 3-point Raita filter on folded needle bytes, load haystack at probe offsets,
 *  fold and compare to find candidates, then verify full safe slice match.
 *  Steps by 62 bytes to ensure complete coverage at chunk boundaries.
 *
 *  @param[in] haystack Pointer to the haystack string.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] needle Pointer to the needle string.
 *  @param[in] needle_length Length of the needle in bytes.
 *  @param[in] window Pre-computed safe window with probe positions.
 *  @param[out] matched_length Length of the match in haystack bytes.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_armenian_ice_(sz_cptr_t haystack, sz_size_t haystack_length,
                                                                  sz_cptr_t needle, sz_size_t needle_length,
                                                                  sz_script_window_t const *window,
                                                                  sz_size_t *matched_length) {

    sz_cptr_t safe_start = needle + window->start;
    sz_size_t safe_length = window->length;

    // Clamp safe slice to single ZMM register, ensuring we don't clip a 2-byte character
    if (safe_length > 64) {
        safe_length = 64;
        // If byte 63 is a D4/D5/D6 lead byte, back off to not split the 2-byte sequence
        sz_u8_t byte63 = (sz_u8_t)safe_start[63];
        if (byte63 == 0xD4 || byte63 == 0xD5 || byte63 == 0xD6) safe_length = 63;
    }

    sz_size_t const safe_offset = window->start;
    sz_size_t const head_length = safe_offset;
    sz_size_t const tail_length = needle_length - safe_offset - safe_length;

    // Pre-fold entire safe slice for vectorized verification
    __mmask64 const safe_mask = sz_u64_mask_until_(safe_length);
    __m512i const needle_safe_folded = sz_armenian_fold_zmm_(_mm512_maskz_loadu_epi8(safe_mask, safe_start));

    // Use pre-computed probe positions from analysis (last byte of first/mid/last codepoints)
    sz_size_t const load_first = window->probe_first - window->prefix_first;
    sz_size_t const load_mid = window->probe_mid - window->prefix_mid;
    sz_size_t const load_last = window->probe_last - window->prefix_last;

    // Extract folded probe bytes from needle
    char needle_safe_folded_bytes[64];
    _mm512_mask_storeu_epi8(needle_safe_folded_bytes, safe_mask, needle_safe_folded);

    // Broadcast needle probe bytes for comparisons
    sz_u512_vec_t probe_first_vec, probe_mid_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_safe_folded_bytes[window->probe_first]);
    probe_mid_vec.zmm = _mm512_set1_epi8(needle_safe_folded_bytes[window->probe_mid]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_safe_folded_bytes[window->probe_last]);

    // Main loop - step by 62 bytes to handle 2-byte chars at boundaries plus 1-byte lookback
    sz_u512_vec_t haystack_first_vec, haystack_mid_vec, haystack_last_vec;
    for (; haystack_length >= needle_length + 64; haystack += 62, haystack_length -= 62) {
        // Load haystack parts at codepoint-aligned offsets (load_* = probe offset - prefix)
        haystack_first_vec.zmm = sz_armenian_fold_zmm_(_mm512_loadu_si512(haystack + safe_offset + load_first));
        haystack_mid_vec.zmm = sz_armenian_fold_zmm_(_mm512_loadu_si512(haystack + safe_offset + load_mid));
        haystack_last_vec.zmm = sz_armenian_fold_zmm_(_mm512_loadu_si512(haystack + safe_offset + load_last));

        // 3-point Raita filter: find positions where all 3 probes match
        // Shift masks by prefix to align results (compare at continuation byte position)
        __mmask64 match_first =
            _mm512_cmpeq_epi8_mask(haystack_first_vec.zmm, probe_first_vec.zmm) >> window->prefix_first;
        __mmask64 match_mid = _mm512_cmpeq_epi8_mask(haystack_mid_vec.zmm, probe_mid_vec.zmm) >> window->prefix_mid;
        __mmask64 match_last = _mm512_cmpeq_epi8_mask(haystack_last_vec.zmm, probe_last_vec.zmm) >> window->prefix_last;
        sz_u64_t matches = _kand_mask64(_kand_mask64(match_first, match_mid), match_last);

        // Exclude positions 62-63 - handle in next iteration to avoid incomplete match
        matches &= 0x3FFFFFFFFFFFFFFFULL;

        // Verify all match candidates found
        while (matches) {
            int const idx = sz_u64_ctz(matches);
            sz_cptr_t const candidate = haystack + idx;
            __m512i const haystack_safe =
                sz_armenian_fold_zmm_(_mm512_maskz_loadu_epi8(safe_mask, candidate + safe_offset));
            if (_mm512_mask_cmpneq_epi8_mask(safe_mask, haystack_safe, needle_safe_folded) == 0) {
                sz_bool_t same_head = //
                    head_length ? sz_utf8_verify_case_insensitive_match_(needle, head_length, candidate,
                                                                         candidate + head_length)
                                : sz_true_k;
                sz_bool_t same_tail = //
                    tail_length ? sz_utf8_verify_case_insensitive_match_(safe_start + safe_length, tail_length,
                                                                         candidate + safe_offset + safe_length,
                                                                         candidate + needle_length)
                                : sz_true_k;
                if (same_head && same_tail) {
                    *matched_length = needle_length;
                    return candidate;
                }
            }
            matches &= matches - 1;
        }
    }

    // Tail: use masked loads for remaining positions (single iteration)
    if (haystack_length >= needle_length) {
        sz_size_t tail_positions = haystack_length - needle_length + 1;
        // For 2-byte scripts, we need to load at least 2 bytes for the fold function to work correctly.
        // The fold function needs to see lead+continuation pairs. We also add prefix for the shift.
        sz_size_t avail_first = haystack_length - safe_offset - load_first;
        sz_size_t avail_mid = haystack_length - safe_offset - load_mid;
        sz_size_t avail_last = haystack_length - safe_offset - load_last;
        sz_size_t need_first = tail_positions + window->prefix_first;
        sz_size_t need_mid = tail_positions + window->prefix_mid;
        sz_size_t need_last = tail_positions + window->prefix_last;
        __mmask64 tail_mask_first = sz_u64_mask_until_(need_first < avail_first ? need_first : avail_first);
        __mmask64 tail_mask_mid = sz_u64_mask_until_(need_mid < avail_mid ? need_mid : avail_mid);
        __mmask64 tail_mask_last = sz_u64_mask_until_(need_last < avail_last ? need_last : avail_last);
        haystack_first_vec.zmm =
            sz_armenian_fold_zmm_(_mm512_maskz_loadu_epi8(tail_mask_first, haystack + safe_offset + load_first));
        haystack_mid_vec.zmm =
            sz_armenian_fold_zmm_(_mm512_maskz_loadu_epi8(tail_mask_mid, haystack + safe_offset + load_mid));
        haystack_last_vec.zmm =
            sz_armenian_fold_zmm_(_mm512_maskz_loadu_epi8(tail_mask_last, haystack + safe_offset + load_last));

        __mmask64 match_first =
            _mm512_cmpeq_epi8_mask(haystack_first_vec.zmm, probe_first_vec.zmm) >> window->prefix_first;
        __mmask64 match_mid = _mm512_cmpeq_epi8_mask(haystack_mid_vec.zmm, probe_mid_vec.zmm) >> window->prefix_mid;
        __mmask64 match_last = _mm512_cmpeq_epi8_mask(haystack_last_vec.zmm, probe_last_vec.zmm) >> window->prefix_last;
        sz_u64_t matches =
            _kand_mask64(_kand_mask64(match_first, match_mid), match_last) & sz_u64_mask_until_(tail_positions);

        while (matches) {
            int const idx = sz_u64_ctz(matches);
            sz_cptr_t const candidate = haystack + idx;
            __m512i const haystack_safe =
                sz_armenian_fold_zmm_(_mm512_maskz_loadu_epi8(safe_mask, candidate + safe_offset));
            if (_mm512_mask_cmpneq_epi8_mask(safe_mask, haystack_safe, needle_safe_folded) == 0) {
                sz_bool_t same_head = head_length ? sz_utf8_verify_case_insensitive_match_(
                                                        needle, head_length, candidate, candidate + head_length)
                                                  : sz_true_k;
                sz_bool_t same_tail =
                    tail_length ? sz_utf8_verify_case_insensitive_match_(safe_start + safe_length, tail_length,
                                                                         candidate + safe_offset + safe_length,
                                                                         candidate + needle_length)
                                : sz_true_k;
                if (same_head && same_tail) {
                    *matched_length = needle_length;
                    return candidate;
                }
            }
            matches &= matches - 1;
        }
    }

    return SZ_NULL_CHAR;
}

/**
 *  @brief  Helper function to find the longest slice of the needle composed exclusively of "safe" Greek characters
 *          so that we can implement a faster substring search only unpacking and folding that subrange of 2-byte
 *          UTF-8 characters, while skipping the rest.
 *
 *  Basic Greek (U+0391-U+03C9) uses 2-byte UTF-8 with predictable case folding:
 *  - CE 91-A1 (Α-Ρ): uppercase, folds via +0x20
 *  - CE A3-A9 (Σ-Ω): uppercase, folds via +0x20 (wrapping to CF for some)
 *  - CE B1-BF (α-ο): lowercase
 *  - CF 80-89 (π-ω): lowercase
 *  - CF 82 (ς): final sigma, same as CF 83 (σ)
 *
 *  ASCII bytes are always safe since they fold independently.
 *
 *  @return sz_cptr_t Pointer to the start of the longest safe Greek slice within the needle.
 *  @retval NULL No safe Greek slice found.
 *  @param[in] needle Pointer to the UTF-8 needle string.
 *  @param[in] needle_length Length of the needle string in bytes.
 *  @param[out] longest_range_length Length of the longest safe Greek slice found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_ice_can_use_greek_path_(sz_cptr_t needle, sz_size_t needle_length,
                                                                            sz_size_t *longest_range_length) {
    sz_cptr_t best_start = SZ_NULL_CHAR;
    sz_size_t best_length = 0;
    sz_cptr_t current_start = needle;
    sz_size_t current_length = 0;

    for (sz_size_t i = 0; i < needle_length;) {
        sz_u8_t byte = (sz_u8_t)needle[i];

        // ASCII byte - always safe in Greek context
        if (byte < 0x80) {
            if (current_length == 0) current_start = needle + i;
            current_length++;
            i++;
            continue;
        }

        // CE lead byte: Greek uppercase and first part of lowercase
        if (byte == 0xCE && i + 1 < needle_length) {
            sz_u8_t second = (sz_u8_t)needle[i + 1];
            // CE 91-A9: uppercase (except A2 which is rare)
            // CE B1-BF: lowercase α-ο
            if ((second >= 0x91 && second <= 0xA9) || (second >= 0xB1 && second <= 0xBF)) {
                if (current_length == 0) current_start = needle + i;
                current_length += 2;
                i += 2;
                continue;
            }
        }

        // CF lead byte: Greek lowercase continuation
        if (byte == 0xCF && i + 1 < needle_length) {
            sz_u8_t second = (sz_u8_t)needle[i + 1];
            // CF 80-89: lowercase π-ω (includes ς at 82 and σ at 83)
            if (second >= 0x80 && second <= 0x89) {
                if (current_length == 0) current_start = needle + i;
                current_length += 2;
                i += 2;
                continue;
            }
        }

        // Not safe - finalize current slice
        if (current_length > best_length) {
            best_length = current_length;
            best_start = current_start;
        }
        current_length = 0;

        // Skip this byte/sequence properly
        if (byte < 0x80) { i++; }
        else if ((byte & 0xE0) == 0xC0) { i += 2; }
        else if ((byte & 0xF0) == 0xE0) { i += 3; }
        else if ((byte & 0xF8) == 0xF0) { i += 4; }
        else { i++; }
        current_start = needle + i;
    }

    // Check final slice
    if (current_length > best_length) {
        best_length = current_length;
        best_start = current_start;
    }

    *longest_range_length = best_length;
    return best_start;
}

/**
 *  @brief  Fold a ZMM register using Greek case folding rules.
 *
 *  Greek uppercase Α-Ω (U+0391-U+03A9): UTF-8 CE 91-A9
 *  Greek lowercase α-ω (U+03B1-U+03C9): UTF-8 CE B1-BF, CF 80-89
 *
 *  Folding maps uppercase to lowercase via +0x20 (wrapping from CE to CF when needed).
 *  Final sigma ς (CF 82) normalizes to regular sigma σ (CF 83).
 *
 *  @param[in] source The source ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_greek_fold_zmm_(__m512i source) {
    __m512i const ce_vec = _mm512_set1_epi8((char)0xCE);
    __m512i const cf_vec = _mm512_set1_epi8((char)0xCF);
    __m512i const x91_vec = _mm512_set1_epi8((char)0x91);
    __m512i const x19_vec = _mm512_set1_epi8(0x19); // Range: 91-A9 is 25 values
    __m512i const x20_vec = _mm512_set1_epi8(0x20);
    __m512i const x82_vec = _mm512_set1_epi8((char)0x82);
    __m512i const x83_vec = _mm512_set1_epi8((char)0x83);
    __m512i const xa0_vec = _mm512_set1_epi8((char)0xA0);
    __m512i const a_upper_vec = _mm512_set1_epi8('A');
    __m512i const range_26_vec = _mm512_set1_epi8(26);

    // Step 1: Detect CE/CF lead bytes
    __mmask64 is_ce = _mm512_cmpeq_epi8_mask(source, ce_vec);
    __mmask64 is_cf = _mm512_cmpeq_epi8_mask(source, cf_vec);
    (void)is_cf; // Used only for final sigma detection below

    // Step 2: Detect positions after CE and after CF
    __mmask64 is_after_ce = is_ce << 1;
    __mmask64 is_after_cf = is_cf << 1;

    // Step 3: Greek uppercase after CE (91-A9)
    // Range check: (byte - 0x91) < 0x19 means byte in [0x91, 0xA9]
    __mmask64 is_greek_ce_upper = _mm512_mask_cmplt_epu8_mask(is_after_ce, _mm512_sub_epi8(source, x91_vec), x19_vec);

    // Step 4: Handle CE/CF boundary crossing for uppercase A0-A9:
    // CE A0 (Π, U+03A0) → CF 80 (π, U+03C0)
    // CE A1 (Ρ, U+03A1) → CF 81 (ρ, U+03C1)
    // etc. For these, we SUBTRACT 0x20 from second byte and change lead CE → CF
    __mmask64 is_will_wrap = _mm512_mask_cmpge_epu8_mask(is_greek_ce_upper, source, xa0_vec);
    // Non-wrapping: 91-9F, we ADD 0x20 (e.g., CE 91 → CE B1)
    __mmask64 is_no_wrap = is_greek_ce_upper & ~is_will_wrap;

    // Step 5: Final sigma normalization: CF 82 → CF 83
    __mmask64 is_final_sigma = _mm512_mask_cmpeq_epi8_mask(is_after_cf, source, x82_vec);

    // Step 6: Fold ASCII uppercase (A-Z) → add 0x20
    __mmask64 is_ascii_upper = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source, a_upper_vec), range_26_vec);

    // Step 7a: Apply Greek uppercase folding for non-wrapping range (91-9F): ADD 0x20
    __m512i result = _mm512_mask_add_epi8(source, is_no_wrap, source, x20_vec);

    // Step 7b: Apply Greek uppercase folding for wrapping range (A0-A9): SUBTRACT 0x20
    result = _mm512_mask_sub_epi8(result, is_will_wrap, source, x20_vec);

    // Step 8: For positions that wrap, change lead byte CE → CF (shift by 1 position)
    __mmask64 is_lead_wrap = is_will_wrap >> 1;
    result = _mm512_mask_mov_epi8(result, is_lead_wrap, cf_vec);

    // Step 9: Normalize final sigma (CF 82 → CF 83)
    result = _mm512_mask_mov_epi8(result, is_final_sigma, x83_vec);

    // Step 10: Apply ASCII folding (add 0x20)
    result = _mm512_mask_add_epi8(result, is_ascii_upper, result, x20_vec);

    // Step 11: Keep CE lead bytes for non-wrapping uppercase (91-9F range)
    // We only touch CE lead bytes that had uppercase folding applied (shift by 1 from is_no_wrap)
    __mmask64 is_ce_upper_lead = is_no_wrap >> 1;
    result = _mm512_mask_mov_epi8(result, is_ce_upper_lead, ce_vec);

    return result;
}

/**
 *  @brief  Greek case-insensitive substring search using AVX-512.
 *
 *  Strategy: Use 3-point Raita filter on folded needle bytes, load haystack at probe offsets,
 *  fold and compare to find candidates, then verify full safe slice match.
 *  Steps by 62 bytes to ensure complete coverage at chunk boundaries.
 *
 *  @param[in] haystack Pointer to the haystack string.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] needle Pointer to the needle string.
 *  @param[in] needle_length Length of the needle in bytes.
 *  @param[in] window Pre-computed safe window with probe positions.
 *  @param[out] matched_length Length of the match in haystack bytes.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_greek_ice_(sz_cptr_t haystack, sz_size_t haystack_length,
                                                               sz_cptr_t needle, sz_size_t needle_length,
                                                               sz_script_window_t const *window,
                                                               sz_size_t *matched_length) {

    sz_cptr_t safe_start = needle + window->start;
    sz_size_t safe_length = window->length;

    // Clamp safe slice to single ZMM register, ensuring we don't clip a 2-byte character
    if (safe_length > 64) {
        safe_length = 64;
        // If byte 63 is a CE/CF lead byte, back off to not split the 2-byte sequence
        sz_u8_t byte63 = (sz_u8_t)safe_start[63];
        if (byte63 == 0xCE || byte63 == 0xCF) safe_length = 63;
    }

    sz_size_t const safe_offset = window->start;
    sz_size_t const head_length = safe_offset;
    sz_size_t const tail_length = needle_length - safe_offset - safe_length;

    // Pre-fold entire safe slice for vectorized verification
    __mmask64 const safe_mask = sz_u64_mask_until_(safe_length);
    __m512i const needle_safe_folded = sz_greek_fold_zmm_(_mm512_maskz_loadu_epi8(safe_mask, safe_start));

    // Use pre-computed probe positions from analysis (last byte of first/mid/last codepoints)
    sz_size_t const load_first = window->probe_first - window->prefix_first;
    sz_size_t const load_mid = window->probe_mid - window->prefix_mid;
    sz_size_t const load_last = window->probe_last - window->prefix_last;

    // Extract folded probe bytes from needle
    char needle_safe_folded_bytes[64];
    _mm512_mask_storeu_epi8(needle_safe_folded_bytes, safe_mask, needle_safe_folded);

    // Broadcast needle probe bytes for comparisons
    sz_u512_vec_t probe_first_vec, probe_mid_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_safe_folded_bytes[window->probe_first]);
    probe_mid_vec.zmm = _mm512_set1_epi8(needle_safe_folded_bytes[window->probe_mid]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_safe_folded_bytes[window->probe_last]);

    // Main loop - step by 62 bytes to handle 2-byte chars at boundaries plus 1-byte lookback
    sz_u512_vec_t haystack_first_vec, haystack_mid_vec, haystack_last_vec;
    for (; haystack_length >= needle_length + 64; haystack += 62, haystack_length -= 62) {
        // Load haystack parts at codepoint-aligned offsets (load_* = probe offset - prefix)
        haystack_first_vec.zmm = sz_greek_fold_zmm_(_mm512_loadu_si512(haystack + safe_offset + load_first));
        haystack_mid_vec.zmm = sz_greek_fold_zmm_(_mm512_loadu_si512(haystack + safe_offset + load_mid));
        haystack_last_vec.zmm = sz_greek_fold_zmm_(_mm512_loadu_si512(haystack + safe_offset + load_last));

        // 3-point Raita filter: find positions where all 3 probes match
        // Shift masks by prefix to align results (compare at continuation byte position)
        __mmask64 match_first =
            _mm512_cmpeq_epi8_mask(haystack_first_vec.zmm, probe_first_vec.zmm) >> window->prefix_first;
        __mmask64 match_mid = _mm512_cmpeq_epi8_mask(haystack_mid_vec.zmm, probe_mid_vec.zmm) >> window->prefix_mid;
        __mmask64 match_last = _mm512_cmpeq_epi8_mask(haystack_last_vec.zmm, probe_last_vec.zmm) >> window->prefix_last;
        sz_u64_t matches = _kand_mask64(_kand_mask64(match_first, match_mid), match_last);

        // Exclude positions 62-63 - handle in next iteration to avoid incomplete match
        matches &= 0x3FFFFFFFFFFFFFFFULL;

        // Verify all match candidates found
        while (matches) {
            int const idx = sz_u64_ctz(matches);
            sz_cptr_t const candidate = haystack + idx;
            __m512i const haystack_safe =
                sz_greek_fold_zmm_(_mm512_maskz_loadu_epi8(safe_mask, candidate + safe_offset));
            if (_mm512_mask_cmpneq_epi8_mask(safe_mask, haystack_safe, needle_safe_folded) == 0) {
                sz_bool_t same_head = //
                    head_length ? sz_utf8_verify_case_insensitive_match_(needle, head_length, candidate,
                                                                         candidate + head_length)
                                : sz_true_k;
                sz_bool_t same_tail = //
                    tail_length ? sz_utf8_verify_case_insensitive_match_(safe_start + safe_length, tail_length,
                                                                         candidate + safe_offset + safe_length,
                                                                         candidate + needle_length)
                                : sz_true_k;
                if (same_head && same_tail) {
                    *matched_length = needle_length;
                    return candidate;
                }
            }
            matches &= matches - 1;
        }
    }

    // Tail: use masked loads for remaining positions (single iteration)
    if (haystack_length >= needle_length) {
        sz_size_t tail_positions = haystack_length - needle_length + 1;
        // For 2-byte scripts, we need to load at least 2 bytes for the fold function to work correctly.
        // The fold function needs to see lead+continuation pairs. We also add prefix for the shift.
        sz_size_t avail_first = haystack_length - safe_offset - load_first;
        sz_size_t avail_mid = haystack_length - safe_offset - load_mid;
        sz_size_t avail_last = haystack_length - safe_offset - load_last;
        sz_size_t need_first = tail_positions + window->prefix_first;
        sz_size_t need_mid = tail_positions + window->prefix_mid;
        sz_size_t need_last = tail_positions + window->prefix_last;
        __mmask64 tail_mask_first = sz_u64_mask_until_(need_first < avail_first ? need_first : avail_first);
        __mmask64 tail_mask_mid = sz_u64_mask_until_(need_mid < avail_mid ? need_mid : avail_mid);
        __mmask64 tail_mask_last = sz_u64_mask_until_(need_last < avail_last ? need_last : avail_last);
        haystack_first_vec.zmm =
            sz_greek_fold_zmm_(_mm512_maskz_loadu_epi8(tail_mask_first, haystack + safe_offset + load_first));
        haystack_mid_vec.zmm =
            sz_greek_fold_zmm_(_mm512_maskz_loadu_epi8(tail_mask_mid, haystack + safe_offset + load_mid));
        haystack_last_vec.zmm =
            sz_greek_fold_zmm_(_mm512_maskz_loadu_epi8(tail_mask_last, haystack + safe_offset + load_last));

        __mmask64 match_first =
            _mm512_cmpeq_epi8_mask(haystack_first_vec.zmm, probe_first_vec.zmm) >> window->prefix_first;
        __mmask64 match_mid = _mm512_cmpeq_epi8_mask(haystack_mid_vec.zmm, probe_mid_vec.zmm) >> window->prefix_mid;
        __mmask64 match_last = _mm512_cmpeq_epi8_mask(haystack_last_vec.zmm, probe_last_vec.zmm) >> window->prefix_last;
        sz_u64_t matches =
            _kand_mask64(_kand_mask64(match_first, match_mid), match_last) & sz_u64_mask_until_(tail_positions);

        while (matches) {
            int const idx = sz_u64_ctz(matches);
            sz_cptr_t const candidate = haystack + idx;
            __m512i const haystack_safe =
                sz_greek_fold_zmm_(_mm512_maskz_loadu_epi8(safe_mask, candidate + safe_offset));
            if (_mm512_mask_cmpneq_epi8_mask(safe_mask, haystack_safe, needle_safe_folded) == 0) {
                sz_bool_t same_head = head_length ? sz_utf8_verify_case_insensitive_match_(
                                                        needle, head_length, candidate, candidate + head_length)
                                                  : sz_true_k;
                sz_bool_t same_tail =
                    tail_length ? sz_utf8_verify_case_insensitive_match_(safe_start + safe_length, tail_length,
                                                                         candidate + safe_offset + safe_length,
                                                                         candidate + needle_length)
                                : sz_true_k;
                if (same_head && same_tail) {
                    *matched_length = needle_length;
                    return candidate;
                }
            }
            matches &= matches - 1;
        }
    }

    return SZ_NULL_CHAR;
}

/**
 *  @brief  Helper function to find the longest slice of the needle composed exclusively of "safe" Vietnamese characters
 *          so that we can implement a faster substring search only unpacking and folding that subrange of 3-byte
 *          UTF-8 characters, while skipping the rest.
 *
 *  Latin Extended Additional (U+1E00-U+1EFF) uses 3-byte UTF-8 with parity-based folding:
 *  - E1 B8 80-BF: U+1E00-U+1E3F (various Latin letters with diacritics)
 *  - E1 B9 80-BF: U+1E40-U+1E7F
 *  - E1 BA 80-BF: U+1E80-U+1EBF (includes Vietnamese letters)
 *  - E1 BB 80-BF: U+1EC0-U+1EFF (Vietnamese continuation)
 *
 *  Folding rule: even third byte = uppercase, add +1 to fold to lowercase.
 *  Exception: U+1E9E (E1 BA 9E) = Capital Sharp S, expands to "ss" - unsafe.
 *
 *  ASCII and Latin-1 (C2/C3) bytes are also safe since they fold independently.
 *
 *  @return sz_cptr_t Pointer to the start of the longest safe Vietnamese slice within the needle.
 *  @retval NULL No safe Vietnamese slice found.
 *  @param[in] needle Pointer to the UTF-8 needle string.
 *  @param[in] needle_length Length of the needle string in bytes.
 *  @param[out] longest_range_length Length of the longest safe Vietnamese slice found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_ice_can_use_vietnamese_path_(sz_cptr_t needle,
                                                                                 sz_size_t needle_length,
                                                                                 sz_size_t *longest_range_length) {
    sz_cptr_t best_start = SZ_NULL_CHAR;
    sz_size_t best_length = 0;
    sz_cptr_t current_start = needle;
    sz_size_t current_length = 0;

    for (sz_size_t i = 0; i < needle_length;) {
        sz_u8_t byte = (sz_u8_t)needle[i];

        // ASCII byte - always safe
        if (byte < 0x80) {
            if (current_length == 0) current_start = needle + i;
            current_length++;
            i++;
            continue;
        }

        // Latin-1 (C2/C3 lead byte) - safe, handled by Latin-1 folding
        if ((byte == 0xC2 || byte == 0xC3) && i + 1 < needle_length) {
            if (current_length == 0) current_start = needle + i;
            current_length += 2;
            i += 2;
            continue;
        }

        // E1 lead byte: Latin Extended Additional
        if (byte == 0xE1 && i + 2 < needle_length) {
            sz_u8_t second = (sz_u8_t)needle[i + 1];
            sz_u8_t third = (sz_u8_t)needle[i + 2];
            // E1 B8-BB: Latin Extended Additional range
            if (second >= 0xB8 && second <= 0xBB) {
                // Exclude U+1E96-U+1E9E (E1 BA 96-9E) which expand to multiple codepoints
                if (second == 0xBA && third >= 0x96 && third <= 0x9E) {
                    // Unsafe - finalize current slice
                    if (current_length > best_length) {
                        best_length = current_length;
                        best_start = current_start;
                    }
                    current_length = 0;
                    i += 3;
                    current_start = needle + i;
                    continue;
                }
                if (current_length == 0) current_start = needle + i;
                current_length += 3;
                i += 3;
                continue;
            }
        }

        // Not safe - finalize current slice
        if (current_length > best_length) {
            best_length = current_length;
            best_start = current_start;
        }
        current_length = 0;

        // Skip this byte/sequence properly
        if (byte < 0x80) { i++; }
        else if ((byte & 0xE0) == 0xC0) { i += 2; }
        else if ((byte & 0xF0) == 0xE0) { i += 3; }
        else if ((byte & 0xF8) == 0xF0) { i += 4; }
        else { i++; }
        current_start = needle + i;
    }

    // Check final slice
    if (current_length > best_length) {
        best_length = current_length;
        best_start = current_start;
    }

    *longest_range_length = best_length;
    return best_start;
}

/**
 *  @brief  Fold a ZMM register using Vietnamese/Latin Extended Additional case folding rules.
 *
 *  Latin Extended Additional (U+1E00-U+1EFF): UTF-8 E1 B8-BB xx
 *  Parity-based folding: even third byte = uppercase, add +1 to fold to lowercase.
 *
 *  Also handles ASCII and Latin-1 folding for mixed-script needles.
 *
 *  @param[in] source The source ZMM register.
 *  @return The folded ZMM register.
 */
SZ_INTERNAL __m512i sz_vietnamese_fold_zmm_(__m512i source) {
    __m512i const e1_vec = _mm512_set1_epi8((char)0xE1);
    __m512i const xb8_vec = _mm512_set1_epi8((char)0xB8);
    __m512i const x04_vec = _mm512_set1_epi8(0x04);
    __m512i const x01_vec = _mm512_set1_epi8(0x01);
    __m512i const c3_vec = _mm512_set1_epi8((char)0xC3);
    __m512i const x80_vec = _mm512_set1_epi8((char)0x80);
    __m512i const x1f_vec = _mm512_set1_epi8(0x1F);
    __m512i const x20_vec = _mm512_set1_epi8(0x20);
    __m512i const a_upper_vec = _mm512_set1_epi8('A');
    __m512i const range_26_vec = _mm512_set1_epi8(26);

    // Step 1: Detect E1 lead bytes (3-byte sequence start)
    __mmask64 is_e1 = _mm512_cmpeq_epi8_mask(source, e1_vec);

    // Step 2: Detect positions after E1 (second byte position)
    __mmask64 is_after_e1 = is_e1 << 1;

    // Step 3: Check if second byte is B8-BB (Latin Extended Additional)
    // Range check: (byte - 0xB8) < 0x04 means byte in [0xB8, 0xBB]
    __mmask64 is_ext_second = _mm512_mask_cmplt_epu8_mask(is_after_e1, _mm512_sub_epi8(source, xb8_vec), x04_vec);

    // Step 4: Detect third byte positions after valid E1 Bx sequences
    __mmask64 is_ext_third = is_ext_second << 1;

    // Step 5: Parity-based folding - if third byte is even, add 1 (uppercase → lowercase)
    // Even check: (byte & 0x01) == 0
    __mmask64 is_even = ~_mm512_test_epi8_mask(source, x01_vec);
    __mmask64 is_vietnamese_upper = is_ext_third & is_even;

    // Step 6: Detect C3 lead bytes for Latin-1 folding
    __mmask64 is_c3 = _mm512_cmpeq_epi8_mask(source, c3_vec);
    __mmask64 is_after_c3 = is_c3 << 1;

    // Step 7: Latin-1 uppercase after C3 (80-9E → add 0x20)
    __mmask64 is_latin1_upper = _mm512_mask_cmplt_epu8_mask(is_after_c3, _mm512_sub_epi8(source, x80_vec), x1f_vec);

    // Step 8: Fold ASCII uppercase (A-Z) → add 0x20
    __mmask64 is_ascii_upper = _mm512_cmplt_epu8_mask(_mm512_sub_epi8(source, a_upper_vec), range_26_vec);

    // Step 9: Apply Vietnamese folding (+1 to third bytes)
    __m512i result = _mm512_mask_add_epi8(source, is_vietnamese_upper, source, x01_vec);

    // Step 10: Apply Latin-1 folding (+0x20 to second bytes)
    result = _mm512_mask_add_epi8(result, is_latin1_upper, result, x20_vec);

    // Step 11: Apply ASCII folding (+0x20)
    result = _mm512_mask_add_epi8(result, is_ascii_upper, result, x20_vec);

    return result;
}

/**
 *  @brief  Vietnamese/Latin Extended Additional case-insensitive substring search using AVX-512.
 *
 *  Strategy: Use 3-point Raita filter on folded needle bytes, load haystack at probe offsets,
 *  fold and compare to find candidates, then verify full safe slice match.
 *  Steps by 62 bytes to ensure complete coverage at chunk boundaries (3-byte chars).
 *
 *  @param[in] haystack Pointer to the haystack string.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] needle Pointer to the needle string.
 *  @param[in] needle_length Length of the needle in bytes.
 *  @param[in] window Pre-computed safe window with probe positions.
 *  @param[out] matched_length Length of the match in haystack bytes.
 *  @return Pointer to match start or SZ_NULL_CHAR if not found.
 */
SZ_INTERNAL sz_cptr_t sz_utf8_case_insensitive_find_vietnamese_ice_(sz_cptr_t haystack, sz_size_t haystack_length,
                                                                    sz_cptr_t needle, sz_size_t needle_length,
                                                                    sz_script_window_t const *window,
                                                                    sz_size_t *matched_length) {

    sz_cptr_t safe_start = needle + window->start;
    sz_size_t safe_length = window->length;

    // Clamp safe slice to single ZMM register, ensuring we don't clip a 3-byte character
    if (safe_length > 64) {
        safe_length = 64;
        // If byte 63 or 62 is an E1 lead byte, back off appropriately
        sz_u8_t byte63 = (sz_u8_t)safe_start[63];
        sz_u8_t byte62 = (sz_u8_t)safe_start[62];
        if (byte63 == 0xE1) safe_length = 63;
        else if (byte62 == 0xE1)
            safe_length = 62;
        // Also handle C2/C3 (2-byte) at position 63
        else if (byte63 == 0xC2 || byte63 == 0xC3)
            safe_length = 63;
    }

    sz_size_t const safe_offset = window->start;
    sz_size_t const head_length = safe_offset;
    sz_size_t const tail_length = needle_length - safe_offset - safe_length;

    // Pre-fold entire safe slice for vectorized verification
    __mmask64 const safe_mask = sz_u64_mask_until_(safe_length);
    __m512i const needle_safe_folded = sz_vietnamese_fold_zmm_(_mm512_maskz_loadu_epi8(safe_mask, safe_start));

    // Use pre-computed probe positions from analysis (last byte of first/mid/last codepoints)
    sz_size_t const load_first = window->probe_first - window->prefix_first;
    sz_size_t const load_mid = window->probe_mid - window->prefix_mid;
    sz_size_t const load_last = window->probe_last - window->prefix_last;

    // Extract folded probe bytes from needle
    char needle_safe_folded_bytes[64];
    _mm512_mask_storeu_epi8(needle_safe_folded_bytes, safe_mask, needle_safe_folded);

    // Broadcast needle probe bytes for comparisons
    sz_u512_vec_t probe_first_vec, probe_mid_vec, probe_last_vec;
    probe_first_vec.zmm = _mm512_set1_epi8(needle_safe_folded_bytes[window->probe_first]);
    probe_mid_vec.zmm = _mm512_set1_epi8(needle_safe_folded_bytes[window->probe_mid]);
    probe_last_vec.zmm = _mm512_set1_epi8(needle_safe_folded_bytes[window->probe_last]);

    // Main loop - step by 62 bytes to handle 3-byte chars at boundaries
    sz_u512_vec_t haystack_first_vec, haystack_mid_vec, haystack_last_vec;
    for (; haystack_length >= needle_length + 64; haystack += 62, haystack_length -= 62) {
        // Load haystack parts at codepoint-aligned offsets (load_* = probe offset - prefix)
        haystack_first_vec.zmm = sz_vietnamese_fold_zmm_(_mm512_loadu_si512(haystack + safe_offset + load_first));
        haystack_mid_vec.zmm = sz_vietnamese_fold_zmm_(_mm512_loadu_si512(haystack + safe_offset + load_mid));
        haystack_last_vec.zmm = sz_vietnamese_fold_zmm_(_mm512_loadu_si512(haystack + safe_offset + load_last));

        // 3-point Raita filter: find positions where all 3 probes match
        // Shift masks by prefix to align results (compare at continuation byte position)
        __mmask64 match_first =
            _mm512_cmpeq_epi8_mask(haystack_first_vec.zmm, probe_first_vec.zmm) >> window->prefix_first;
        __mmask64 match_mid = _mm512_cmpeq_epi8_mask(haystack_mid_vec.zmm, probe_mid_vec.zmm) >> window->prefix_mid;
        __mmask64 match_last = _mm512_cmpeq_epi8_mask(haystack_last_vec.zmm, probe_last_vec.zmm) >> window->prefix_last;
        sz_u64_t matches = _kand_mask64(_kand_mask64(match_first, match_mid), match_last);

        // Exclude positions 62-63 - handle in next iteration
        matches &= 0x3FFFFFFFFFFFFFFFULL;

        // Verify all match candidates found
        while (matches) {
            int const idx = sz_u64_ctz(matches);
            sz_cptr_t const candidate = haystack + idx;
            __m512i const haystack_safe =
                sz_vietnamese_fold_zmm_(_mm512_maskz_loadu_epi8(safe_mask, candidate + safe_offset));
            if (_mm512_mask_cmpneq_epi8_mask(safe_mask, haystack_safe, needle_safe_folded) == 0) {
                sz_bool_t same_head = //
                    head_length ? sz_utf8_verify_case_insensitive_match_(needle, head_length, candidate,
                                                                         candidate + head_length)
                                : sz_true_k;
                sz_bool_t same_tail = //
                    tail_length ? sz_utf8_verify_case_insensitive_match_(safe_start + safe_length, tail_length,
                                                                         candidate + safe_offset + safe_length,
                                                                         candidate + needle_length)
                                : sz_true_k;
                if (same_head && same_tail) {
                    *matched_length = needle_length;
                    return candidate;
                }
            }
            matches &= matches - 1;
        }
    }

    // Tail: use masked loads for remaining positions (single iteration)
    if (haystack_length >= needle_length) {
        sz_size_t tail_positions = haystack_length - needle_length + 1;
        // For 3-byte scripts, we need to load at least 3 bytes for the fold function to work correctly.
        // The fold function needs to see complete lead+continuation triplets. We also add prefix for the shift.
        sz_size_t avail_first = haystack_length - safe_offset - load_first;
        sz_size_t avail_mid = haystack_length - safe_offset - load_mid;
        sz_size_t avail_last = haystack_length - safe_offset - load_last;
        sz_size_t need_first = tail_positions + window->prefix_first;
        sz_size_t need_mid = tail_positions + window->prefix_mid;
        sz_size_t need_last = tail_positions + window->prefix_last;
        __mmask64 tail_mask_first = sz_u64_mask_until_(need_first < avail_first ? need_first : avail_first);
        __mmask64 tail_mask_mid = sz_u64_mask_until_(need_mid < avail_mid ? need_mid : avail_mid);
        __mmask64 tail_mask_last = sz_u64_mask_until_(need_last < avail_last ? need_last : avail_last);
        haystack_first_vec.zmm =
            sz_vietnamese_fold_zmm_(_mm512_maskz_loadu_epi8(tail_mask_first, haystack + safe_offset + load_first));
        haystack_mid_vec.zmm =
            sz_vietnamese_fold_zmm_(_mm512_maskz_loadu_epi8(tail_mask_mid, haystack + safe_offset + load_mid));
        haystack_last_vec.zmm =
            sz_vietnamese_fold_zmm_(_mm512_maskz_loadu_epi8(tail_mask_last, haystack + safe_offset + load_last));

        __mmask64 match_first =
            _mm512_cmpeq_epi8_mask(haystack_first_vec.zmm, probe_first_vec.zmm) >> window->prefix_first;
        __mmask64 match_mid = _mm512_cmpeq_epi8_mask(haystack_mid_vec.zmm, probe_mid_vec.zmm) >> window->prefix_mid;
        __mmask64 match_last = _mm512_cmpeq_epi8_mask(haystack_last_vec.zmm, probe_last_vec.zmm) >> window->prefix_last;
        sz_u64_t matches =
            _kand_mask64(_kand_mask64(match_first, match_mid), match_last) & sz_u64_mask_until_(tail_positions);

        while (matches) {
            int const idx = sz_u64_ctz(matches);
            sz_cptr_t const candidate = haystack + idx;
            __m512i const haystack_safe =
                sz_vietnamese_fold_zmm_(_mm512_maskz_loadu_epi8(safe_mask, candidate + safe_offset));
            if (_mm512_mask_cmpneq_epi8_mask(safe_mask, haystack_safe, needle_safe_folded) == 0) {
                sz_bool_t same_head = head_length ? sz_utf8_verify_case_insensitive_match_(
                                                        needle, head_length, candidate, candidate + head_length)
                                                  : sz_true_k;
                sz_bool_t same_tail =
                    tail_length ? sz_utf8_verify_case_insensitive_match_(safe_start + safe_length, tail_length,
                                                                         candidate + safe_offset + safe_length,
                                                                         candidate + needle_length)
                                : sz_true_k;
                if (same_head && same_tail) {
                    *matched_length = needle_length;
                    return candidate;
                }
            }
            matches &= matches - 1;
        }
    }

    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_ice( //
    sz_cptr_t haystack, sz_size_t haystack_length,     //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length) {

    // There is a way to perform case-insensitive substring search faster than case-folding both strings
    // and calling a standard substring search algorithm on them. Case-folding bicameral scripts is typically
    // a multi-step procedure:
    //
    // 1. find letter boundaries
    // 2. unpack them into Unicode runes
    // 3. case-fold the runes
    // 4. repack the case-folded runes into UTF-8
    // 5. perform substring search, finally!
    //
    // In most practical cases you can skip steps 1-4 for the haystack, and only case-fold the needle on-the-fly
    // during substring search, but there are many (!!) caveats! For example, if the needle contains a letter 'f',
    // and the haystack contains the letter 'ﬁ' (U+FB01 LATIN SMALL LIGATURE FI), then you have to case-fold the
    // haystack letter - expanding into a wider sequence of letters - in order to find a match. These are
    // performace-killers for SIMD code!
    //
    // Instead, we've made a huge exploration effort of all possible rules we can leverage to speed up case-insensitive
    // substing search in `expolore_unicode_case.ipynb`. The shared idea is to find a slice of a needle that contains
    // only characters that have identical width in all of the scripts that case-fold into it.
    //
    // Here are some examples of fast paths:
    //
    // 1. Safe ASCII sequences.
    //    The ASCII letters [bcdegmopqruvxz] don't ever appear in some multi-byte case-folds.
    //    We can safely match chunks of such letters without case folding. Moreover, even when encountering
    //    other letters like [afhijklnstwy], we may be able to match them safely depending on the surrounding
    //    context in the needle.
    // 2. Safe Latin extensions of mixed 1-byte and 2-byte letters.
    // 3. Safe Cyrillic sequences.
    // 4. Safe Armenian sequences.
    // 5. Safe Georgian sequences.
    // 6. Safe Vietnamese sequences.
    //
    // Assuming, unpacking the characters in non-ASCII sequences can be costly, only the first variant uses Raita
    // heuristic like the `sz_find_skylake` - matching the first, last, and middle point of the safe ASCII slice,
    // simultaneously loading haystack parts at different offsets into 3 different ZMM registers. If those parts match,
    // the part in-between start and end of the safe region is matched via a AVX-512 as well. The head and tail outside
    // of the safe zone are unpacked serially via `sz_rune_parse` and its reverse-order variant - `sz_rune_rparse`.
    //
    // In other variants, we only load one ZMM register of a haystack at a time. We unpack and case-fold in a
    // language-specific fashion, checking which haystack character can match which of the 3 different characters from
    // the needle (start, middle, end of the safe zone).
    //
    // If none of those work out - fall back to the serial implementation.

    // Handle edge cases first
    if (needle_length == 0) {
        *matched_length = 0;
        return haystack;
    }
    if (haystack_length < needle_length) { return SZ_NULL_CHAR; }

    // Fast-path: if needle is fully caseless (CJK, Hangul, digits, symbols, etc.),
    // use direct sz_find since case folding won't affect the match
    if (sz_utf8_is_fully_caseless_(needle, needle_length)) {
        sz_cptr_t result = sz_find(haystack, haystack_length, needle, needle_length);
        if (result) {
            *matched_length = needle_length;
            return result;
        }
        *matched_length = 0;
        return SZ_NULL_CHAR;
    }

    // Single-pass needle analysis: finds best safe windows for all 6 script paths
    sz_needle_analysis_t analysis = sz_utf8_analyze_needle_(needle, needle_length);

    // Path selection strategy:
    // 1. Prefer ASCII/Latin1/Vietnamese when they meet thresholds - they have broader haystack compatibility
    //    (can handle Latin1 characters like ß, accented letters, etc.)
    // 2. For script-specific paths (Cyrillic, Greek, Armenian), select the longest one
    // 3. Fall back to serial if no path meets its threshold

    // Priority 1: ASCII path (broadest haystack compatibility, handles all byte values)
    if (analysis.ascii.length >= 3)
        return sz_utf8_case_insensitive_find_ascii_ice_(haystack, haystack_length, needle, needle_length,
                                                        needle + analysis.ascii.start, analysis.ascii.length,
                                                        matched_length);

    // Priority 2: Latin1 path (includes Latin-1 Supplement: ß, accented letters, etc.)
    if (analysis.latin1.length >= 4)
        return sz_utf8_case_insensitive_find_latin1_ice_(haystack, haystack_length, needle, needle_length,
                                                         &analysis.latin1, matched_length);

    // Priority 3: Vietnamese path (includes Latin1 + Latin Extended Additional)
    if (analysis.vietnamese.length >= 6)
        return sz_utf8_case_insensitive_find_vietnamese_ice_(haystack, haystack_length, needle, needle_length,
                                                             &analysis.vietnamese, matched_length);

    // Priority 4: Script-specific paths - select longest among Cyrillic/Greek/Armenian
    // These only handle their specific script + ASCII, so less versatile for mixed-script haystacks
    // Note: Windows without actual script chars have length=0 (set during analysis), so the length
    // check is sufficient - no need for separate has_script scanning loops.
    sz_size_t best_script_len = 0;
    if (analysis.cyrillic.length > best_script_len) best_script_len = analysis.cyrillic.length;
    if (analysis.greek.length > best_script_len) best_script_len = analysis.greek.length;
    if (analysis.armenian.length > best_script_len) best_script_len = analysis.armenian.length;

    // Select among script-specific paths based on longest window
    if (analysis.cyrillic.length == best_script_len && analysis.cyrillic.length >= 4)
        return sz_utf8_case_insensitive_find_cyrillic_ice_(haystack, haystack_length, needle, needle_length,
                                                           &analysis.cyrillic, matched_length);
    if (analysis.greek.length == best_script_len && analysis.greek.length >= 4)
        return sz_utf8_case_insensitive_find_greek_ice_(haystack, haystack_length, needle, needle_length,
                                                        &analysis.greek, matched_length);
    if (analysis.armenian.length == best_script_len && analysis.armenian.length >= 4)
        return sz_utf8_case_insensitive_find_armenian_ice_(haystack, haystack_length, needle, needle_length,
                                                           &analysis.armenian, matched_length);

    // No suitable SIMD path found (needle has complex Unicode), fall back to serial
    return sz_utf8_case_insensitive_find_serial(haystack, haystack_length, needle, needle_length, matched_length);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_ICE
#pragma endregion // Ice Lake Implementation

#pragma region NEON Implementation

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
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length) {
    return sz_utf8_case_insensitive_find_serial(haystack, haystack_length, needle, needle_length, matched_length);
}

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

SZ_DYNAMIC sz_cptr_t sz_utf8_case_insensitive_find(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                   sz_size_t needle_length, sz_size_t *matched_length) {
#if SZ_USE_ICE
    return sz_utf8_case_insensitive_find_ice(haystack, haystack_length, needle, needle_length, matched_length);
#else
    return sz_utf8_case_insensitive_find_serial(haystack, haystack_length, needle, needle_length, matched_length);
#endif
}

SZ_DYNAMIC sz_ordering_t sz_utf8_case_insensitive_order(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                        sz_size_t b_length) {
    return sz_utf8_case_insensitive_order_serial(a, a_length, b, b_length);
}

#endif // !SZ_DYNAMIC_DISPATCH

#pragma endregion // Dynamic Dispatch

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_CASE_H_
