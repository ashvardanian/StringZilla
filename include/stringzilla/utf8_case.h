/**
 *  @brief  Hardware-accelerated UTF-8 text processing utilities, that require unpacking into UTF-32 runes.
 *  @file   utf8_unpack.h
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
 *  All algorithms are fully complaint with the specification and handle all edge cases.
 *
 *  On fast vectorized paths, unlike other parts of StringZilla, there may be significant algorithmic
 *  differences between different ISA versions. Most of them are designed to be practical in common
 *  usecases, targetting the most common languages on the Internet.
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
 *
 *  Huge part of case-insensitive operations can be performed
 */
#ifndef STRINGZILLA_UTF8_UNPACK_H_
#define STRINGZILLA_UTF8_UNPACK_H_

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
SZ_PUBLIC sz_ordering_t sz_utf8_case_insensitive_order_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                              sz_size_t b_length);

/** @copydoc sz_utf8_case_fold */
SZ_PUBLIC sz_size_t sz_utf8_case_fold_ice( //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination);
/** @copydoc sz_utf8_case_insensitive_find */
SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_ice( //
    sz_cptr_t haystack, sz_size_t haystack_length,     //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length);
/** @copydoc sz_utf8_case_insensitive_order */
SZ_PUBLIC sz_ordering_t sz_utf8_case_insensitive_order_ice(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                           sz_size_t b_length);

/** @copydoc sz_utf8_case_fold */
SZ_PUBLIC sz_size_t sz_utf8_case_fold_neon( //
    sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination);
/** @copydoc sz_utf8_case_insensitive_find */
SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_neon( //
    sz_cptr_t haystack, sz_size_t haystack_length,      //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length);
/** @copydoc sz_utf8_case_insensitive_order */
SZ_PUBLIC sz_ordering_t sz_utf8_case_insensitive_order_neon(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                            sz_size_t b_length);

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
        case 0xFB13: folded[0] = 0x0574; folded[1] = 0x0576; return 2; // ﬓ → մdelays
        case 0xFB14: folded[0] = 0x0574; folded[1] = 0x0565; return 2; // ﬔ → մdelays
        case 0xFB15: folded[0] = 0x0574; folded[1] = 0x056B; return 2; // ﬕ → մdelays
        case 0xFB16: folded[0] = 0x057E; folded[1] = 0x0576; return 2; // ﬖ → delays
        case 0xFB17: folded[0] = 0x0574; folded[1] = 0x056D; return 2; // ﬗ → մdelays
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

/** @brief Get next folded rune. Returns `sz_false_k` when exhausted or on invalid UTF-8. */
SZ_INTERNAL sz_bool_t sz_utf8_folded_iter_next_(sz_utf8_folded_iter_t *it, sz_rune_t *out_rune) {
    // Refill pending buffer if exhausted
    if (it->pending_idx >= it->pending_count) {
        if (it->ptr >= it->end) return sz_false_k;

        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse(it->ptr, &rune, &rune_length);
        if (rune_length == sz_utf8_invalid_k) return sz_false_k;

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

SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_serial( //
    sz_cptr_t haystack, sz_size_t haystack_length,        //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length) {

    if (needle_length == 0) {
        *matched_length = 0;
        return haystack;
    }

    // Phase 1: Compute needle hash and folded rune count using iterator
    sz_u64_t needle_hash = 0;
    sz_size_t needle_folded_count = 0;
    {
        sz_utf8_folded_iter_t needle_iterator;
        sz_utf8_folded_iter_init_(&needle_iterator, needle, needle_length);
        sz_rune_t needle_rune;
        while (sz_utf8_folded_iter_next_(&needle_iterator, &needle_rune)) {
            needle_hash = needle_hash * 257 + needle_rune;
            needle_folded_count++;
        }
    }
    if (needle_folded_count == 0) {
        *matched_length = 0;
        return SZ_NULL_CHAR;
    }

    // Precompute highest_power = 257^(needle_folded_count - 1)
    sz_u64_t highest_power = 1;
    for (sz_size_t i = 1; i < needle_folded_count; ++i) highest_power *= 257;

    // Phase 2: Initialize haystack window using iterator
    sz_utf8_folded_iter_t window_iterator;
    sz_utf8_folded_iter_init_(&window_iterator, haystack, haystack_length);

    sz_cptr_t window_start = haystack;
    sz_u64_t window_hash = 0;
    sz_size_t window_count = 0;
    sz_rune_t rune;

    while (window_count < needle_folded_count && sz_utf8_folded_iter_next_(&window_iterator, &rune)) {
        window_hash = window_hash * 257 + rune;
        window_count++;
    }
    sz_cptr_t window_end = window_iterator.ptr;

    if (window_count < needle_folded_count) {
        *matched_length = 0;
        return SZ_NULL_CHAR;
    }

    // Phase 3: Slide window through haystack
    while (window_count == needle_folded_count) {
        // Check for hash match and verify
        if (window_hash == needle_hash &&
            sz_utf8_verify_case_insensitive_match_(needle, needle_length, window_start, window_end)) {
            *matched_length = (sz_size_t)(window_end - window_start);
            return window_start;
        }

        // Advance window_start by one source rune
        sz_rune_t old_rune;
        sz_rune_length_t old_len;
        sz_rune_parse(window_start, &old_rune, &old_len);
        if (old_len == sz_utf8_invalid_k) break;

        sz_rune_t old_folded[4];
        sz_size_t old_folded_count = sz_unicode_fold_codepoint_(old_rune, old_folded);

        // Remove old folded runes from hash
        for (sz_size_t i = 0; i < old_folded_count; ++i) {
            window_hash = (window_hash - old_folded[i] * highest_power) * 257;
            window_count--;
        }
        window_start += old_len;

        // Add new folded runes to refill window
        while (window_count < needle_folded_count && sz_utf8_folded_iter_next_(&window_iterator, &rune)) {
            window_hash += rune;
            window_count++;
        }
        window_end = window_iterator.ptr;
    }

    *matched_length = 0;
    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_size_t sz_utf8_case_fold_serial(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination) {

    sz_u8_t const *src = (sz_u8_t const *)source;
    sz_u8_t const *src_end = src + source_length;
    sz_u8_t *dst = (sz_u8_t *)destination;
    sz_u8_t *dst_start = dst;

    while (src < src_end) {
        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse((sz_cptr_t)src, &rune, &rune_length);
        sz_assert_(rune_length != sz_utf8_invalid_k && "Input text is not valid UTF-8");
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

SZ_PUBLIC sz_size_t sz_utf8_case_fold_ice(sz_cptr_t source, sz_size_t source_length, sz_ptr_t target) {
    // This algorithm exploits the idea, that most text in a single ZMM register is either:
    // 1. All ASCII single-byte codepoints
    // 2. Mixture of 2-byte codepoints in one language (continuous range) and 1-byte ASCII codepoints
    //    2.1. Only Latin-1 supplements, where excluding three characters (×,÷,ß) folding is a trivial
    //         addition of +32 to the second byte, which doesn't even require packing/unpacking to UTF-16.
    //         The three exceptions are handled via blends.
    //    2.2. Other scripts (Cyrillic, Greek, etc.) where we unpack into UTF-16 codepoints, fold, and repack.
    // 3. Groups of 3- and 4-byte codepoints without folding rules defined
    // Within those assumptions, this kernel vectorizes case folding for simple cases, when
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

    while (source_length) {
        sz_size_t chunk_size = sz_min_of_two(source_length, 64);
        __mmask64 load_mask = sz_u64_mask_until_(chunk_size);
        source_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, source);
        __mmask64 is_non_ascii = _mm512_movepi8_mask(source_vec.zmm);

        // Compute all lead byte masks once per iteration
        __mmask64 is_cont = _mm512_cmpeq_epi8_mask(_mm512_and_si512(source_vec.zmm, _mm512_set1_epi8((char)0xC0)),
                                                   _mm512_set1_epi8((char)0x80));
        __mmask64 is_three_byte_lead = _mm512_cmpeq_epi8_mask(
            _mm512_and_si512(source_vec.zmm, _mm512_set1_epi8((char)0xF0)), _mm512_set1_epi8((char)0xE0));
        __mmask64 is_four_byte_lead = _mm512_cmpeq_epi8_mask(
            _mm512_and_si512(source_vec.zmm, _mm512_set1_epi8((char)0xF8)), _mm512_set1_epi8((char)0xF0));

        // Check that all loaded characters are ASCII
        if (is_non_ascii == 0) {
            __mmask64 is_upper =
                _mm512_cmp_epu8_mask(_mm512_sub_epi8(source_vec.zmm, a_upper_vec), subtract26_vec, _MM_CMPINT_LT);
            _mm512_mask_storeu_epi8(target, load_mask,
                                    _mm512_mask_add_epi8(source_vec.zmm, is_upper, source_vec.zmm, x20_vec));
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
                // E2 80-83, 85-9F, A0-AF are safe (punctuation, symbols, currency, math)
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
                    // Pure safe 3-byte content (E0, E3-E9, EB-EE) - no case folding needed
                    // Just need to avoid splitting a 3-byte sequence at the end
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
                        _mm512_mask_storeu_epi8(target, copy_mask, source_vec.zmm);
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
        sz_size_t latin1_length = sz_u64_ctz(~is_valid_latin1_mix | ~load_mask);
        latin1_length -= latin1_length && ((is_latin1_lead >> (latin1_length - 1)) & 1); // Don't split 2-byte seq

        if (latin1_length >= 2) {
            __mmask64 prefix_mask = sz_u64_mask_until_(latin1_length);
            __mmask64 latin1_second_bytes = is_latin1_second & prefix_mask;

            // ASCII A-Z (0x41-0x5A) and Latin-1 À-Þ (second byte 0x80-0x9E excl. ×=0x97) both get +0x20
            __mmask64 is_upper_ascii = _mm512_cmp_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8('A')),
                                                            _mm512_set1_epi8(26), _MM_CMPINT_LT);
            __mmask64 is_latin1_upper = _mm512_mask_cmp_epu8_mask(
                latin1_second_bytes, _mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8((char)0x80)),
                _mm512_set1_epi8(0x1F), _MM_CMPINT_LT);
            is_latin1_upper ^= _mm512_mask_cmpeq_epi8_mask(is_latin1_upper, source_vec.zmm,
                                                           _mm512_set1_epi8((char)0x97)); // Exclude ×
            __m512i folded = _mm512_mask_add_epi8(source_vec.zmm, (is_upper_ascii | is_latin1_upper) & prefix_mask,
                                                  source_vec.zmm, _mm512_set1_epi8(0x20));

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
            sz_size_t cyrillic_length = sz_u64_ctz(~is_valid_cyrillic_mix | ~load_mask);
            cyrillic_length -= cyrillic_length && ((is_cyrillic_lead >> (cyrillic_length - 1)) & 1);

            if (cyrillic_length >= 2) {
                __mmask64 prefix_mask = sz_u64_mask_until_(cyrillic_length);
                __mmask64 d0_seconds = (is_d0 << 1) & prefix_mask;

                // Start with source, apply ASCII folding
                __mmask64 is_upper_ascii = _mm512_cmp_epu8_mask(_mm512_sub_epi8(source_vec.zmm, _mm512_set1_epi8('A')),
                                                                _mm512_set1_epi8(26), _MM_CMPINT_LT);
                __m512i folded = _mm512_mask_add_epi8(source_vec.zmm, is_upper_ascii & prefix_mask, source_vec.zmm,
                                                      _mm512_set1_epi8(0x20));

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
        __mmask64 is_caseless_2byte = _mm512_cmp_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xD7), _MM_CMPINT_GE) &
                                      _mm512_cmp_epu8_mask(source_vec.zmm, _mm512_set1_epi8((char)0xDF), _MM_CMPINT_LE);
        if (is_caseless_2byte) {
            __mmask64 is_caseless_second = is_caseless_2byte << 1;
            __mmask64 is_valid_caseless = ~is_non_ascii | is_caseless_2byte | is_caseless_second;
            sz_size_t caseless_length = sz_u64_ctz(~is_valid_caseless | ~load_mask);
            caseless_length -= caseless_length && ((is_caseless_2byte >> (caseless_length - 1)) & 1);

            if (caseless_length >= 2) {
                __mmask64 prefix_mask = sz_u64_mask_until_(caseless_length);
                // Fold only ASCII A-Z, copy 2-byte unchanged
                __mmask64 is_upper_ascii =
                    _mm512_cmp_epu8_mask(_mm512_sub_epi8(source_vec.zmm, a_upper_vec), subtract26_vec, _MM_CMPINT_LT);
                __m512i folded =
                    _mm512_mask_add_epi8(source_vec.zmm, is_upper_ascii & prefix_mask, source_vec.zmm, x20_vec);
                _mm512_mask_storeu_epi8(target, prefix_mask, folded);
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
        sz_size_t two_byte_length = sz_u64_ctz(~is_valid_two_byte_mix | ~load_mask);
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
                // Armenian ligature that expands: U+0587 (և → եdelays)
                _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x0587));

            // Only consider characters we're actually processing
            needs_serial &= (__mmask16)((1u << num_chars) - 1);

            // Handle serial-needed characters by processing them one at a time
            if (needs_serial) {
                sz_size_t first_special = (sz_size_t)sz_u64_ctz((sz_u64_t)needs_serial);
                if (first_special == 0) {
                    // First character needs serial - process it and continue the main loop
                    sz_rune_t rune;
                    sz_rune_length_t rune_length;
                    sz_rune_parse(source, &rune, &rune_length);
                    if (rune_length == sz_utf8_invalid_k) {
                        *target++ = *source++;
                        source_length--;
                    }
                    else {
                        sz_rune_t folded_runes[4];
                        sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
                        for (sz_size_t i = 0; i != folded_count; ++i)
                            target += sz_rune_export(folded_runes[i], (sz_u8_t *)target);
                        source += rune_length;
                        source_length -= rune_length;
                    }
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
                sz_size_t valid_length = sz_u64_ctz(~is_valid | ~load_mask);

                // Don't split a 3-byte sequence
                if (valid_length >= 1 && valid_length < 64) {
                    __mmask64 all_leads = is_three_byte_lead & sz_u64_mask_until_(valid_length);
                    __mmask64 safe_mask = valid_length >= 3 ? sz_u64_mask_until_(valid_length - 2) : 0;
                    __mmask64 unsafe = all_leads & ~safe_mask;
                    if (unsafe) valid_length = sz_u64_ctz(unsafe);
                }

                if (valid_length >= 2) {
                    __mmask64 mask = sz_u64_mask_until_(valid_length);
                    // Fold ASCII A-Z, copy everything else unchanged
                    __mmask64 is_upper_ascii = _mm512_cmp_epu8_mask(_mm512_sub_epi8(source_vec.zmm, a_upper_vec),
                                                                    subtract26_vec, _MM_CMPINT_LT);
                    __m512i folded =
                        _mm512_mask_add_epi8(source_vec.zmm, is_upper_ascii & mask, source_vec.zmm, x20_vec);
                    _mm512_mask_storeu_epi8(target, mask, folded);
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
                __m512i indices_vec =
                    _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43,
                                    42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22,
                                    21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
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

                // If all checkable E1 leads are Georgian (82/83) and no other complex content
                // E1 leads at the edge (position 63) are handled by not including them in this check
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
                    __mmask64 is_83_c7 = _mm512_mask_cmpeq_epi8_mask(
                        third_pos_83 & load_mask, source_vec.zmm, _mm512_set1_epi8((char)0x87));
                    __mmask64 is_83_cd = _mm512_mask_cmpeq_epi8_mask(
                        third_pos_83 & load_mask, source_vec.zmm, _mm512_set1_epi8((char)0x8D));
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
                    sz_size_t georgian_length = sz_u64_ctz(~is_valid_georgian_mix | ~load_mask);

                    // Don't split multi-byte sequences (2-byte C2, 3-byte E1/E2/EA)
                    if (georgian_length >= 1 && georgian_length < 64) {
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
                        __mmask64 is_upper_ascii = _mm512_cmp_epu8_mask(_mm512_sub_epi8(source_vec.zmm, a_upper_vec),
                                                                        subtract26_vec, _MM_CMPINT_LT);
                        folded = _mm512_mask_add_epi8(folded, is_upper_ascii & prefix_mask, folded, x20_vec);

                        _mm512_mask_storeu_epi8(target, prefix_mask, folded);
                        target += georgian_length, source += georgian_length, source_length -= georgian_length;
                        // DEBUG: printf("Georgian path: processed %zu bytes\n", georgian_length);
                        continue;
                    }
                }
            }
            // DEBUG: Add counter for when Georgian path is skipped
            // static int skip_count = 0; if (is_e1_lead && ++skip_count < 10) printf("Skipped Georgian path at len=%zu\n", source_length);

            // Slow path: Has 2-byte, 4-byte, or E1/E2/EF leads that need special handling
            // EA with problematic second bytes (is_ea_complex) also needs special handling
            // But plain EA (Hangul) is safe
            __mmask64 is_ea_lead_complex = is_ea_complex >> 1; // Shift back to lead position
            __mmask64 is_safe_three_byte_lead =
                is_three_byte_lead & ~is_e1_lead & ~is_e2_lead & ~is_ea_lead_complex & ~is_ef_lead;
            __mmask64 is_valid_mixed = ~is_non_ascii | is_safe_three_byte_lead | is_cont;
            is_valid_mixed &= ~is_four_byte_lead;
            sz_size_t three_byte_length = sz_u64_ctz(~is_valid_mixed | ~load_mask);

            // Don't split a 3-byte sequence: find first incomplete lead and truncate there
            if (three_byte_length >= 1 && three_byte_length < 64) {
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

                if (problematic_leads == 0) {
                    // No E1 or EF leads in prefix - fold ASCII A-Z, copy 3-byte chars unchanged
                    __mmask64 is_upper_ascii = _mm512_cmp_epu8_mask(_mm512_sub_epi8(source_vec.zmm, a_upper_vec),
                                                                    subtract26_vec, _MM_CMPINT_LT);
                    __m512i folded =
                        _mm512_mask_add_epi8(source_vec.zmm, is_upper_ascii & prefix_mask_3, source_vec.zmm, x20_vec);
                    _mm512_mask_storeu_epi8(target, prefix_mask_3, folded);
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

                    // Check for Capital Eszett ẞ (E1 BA 9E = U+1E9E) which expands to "ss"
                    __mmask64 is_ba_second =
                        _mm512_mask_cmpeq_epi8_mask(is_e1_in_prefix, second_bytes, _mm512_set1_epi8((char)0xBA));
                    __mmask64 is_9e_third =
                        _mm512_mask_cmpeq_epi8_mask(is_ba_second, third_bytes, _mm512_set1_epi8((char)0x9E));
                    __mmask64 needs_serial_e1 = is_greek_ext | is_9e_third;

                    if (needs_serial_e1) {
                        sz_size_t first_special = sz_u64_ctz(needs_serial_e1);
                        if (first_special == 0) {
                            // First char needs serial processing
                            sz_rune_t rune;
                            sz_rune_length_t rune_length;
                            sz_rune_parse(source, &rune, &rune_length);
                            if (rune_length == sz_utf8_invalid_k) {
                                *target++ = *source++;
                                source_length--;
                            }
                            else {
                                sz_rune_t folded_runes[4];
                                sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);
                                for (sz_size_t i = 0; i != folded_count; ++i)
                                    target += sz_rune_export(folded_runes[i], (sz_u8_t *)target);
                                source += rune_length;
                                source_length -= rune_length;
                            }
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
                        // Exceptions at E1 BA 96-9B (U+1E96-1E9B) need serial - but these are odd or expand
                        // Actually 96,98,9A are even but expand, so we should have already filtered 9E
                        // For simplicity, apply +1 to even third bytes in Latin Ext Add positions
                        __mmask64 third_positions = is_latin_ext_add << 2;
                        // Check if third byte is even (bit 0 = 0)
                        __mmask64 is_even_third = ~_mm512_test_epi8_mask(source_vec.zmm, _mm512_set1_epi8(0x01));
                        __mmask64 fold_third = third_positions & is_even_third & prefix_mask_3;

                        // Also fold ASCII A-Z
                        __mmask64 is_upper_ascii = _mm512_cmp_epu8_mask(_mm512_sub_epi8(source_vec.zmm, a_upper_vec),
                                                                        subtract26_vec, _MM_CMPINT_LT);
                        __m512i folded =
                            _mm512_mask_add_epi8(source_vec.zmm, (fold_third | is_upper_ascii) & prefix_mask_3,
                                                 source_vec.zmm, _mm512_set1_epi8(0x01));
                        // ASCII needs +0x20 not +1, fix the ASCII positions
                        folded = _mm512_mask_add_epi8(folded, is_upper_ascii & prefix_mask_3, folded,
                                                      _mm512_set1_epi8(0x1F)); // 0x20 - 0x01 = 0x1F more

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
                            folded =
                                _mm512_mask_blend_epi8(georgian_second_pos, folded, _mm512_set1_epi8((char)0xB4));

                            // Adjust third bytes based on original second byte
                            // -0x20 for sequences that had 82, +0x20 for sequences that had 83
                            folded = _mm512_mask_sub_epi8(folded, is_82_valid, folded, _mm512_set1_epi8(0x20));
                            folded = _mm512_mask_add_epi8(folded, is_83_valid, folded, _mm512_set1_epi8(0x20));

                            // Also fold any ASCII A-Z that might be mixed in
                            __mmask64 is_upper_ascii = _mm512_cmp_epu8_mask(
                                _mm512_sub_epi8(source_vec.zmm, a_upper_vec), subtract26_vec, _MM_CMPINT_LT);
                            folded = _mm512_mask_add_epi8(folded, is_upper_ascii & prefix_mask_3, folded, x20_vec);

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
                        __mmask64 is_upper_ascii = _mm512_cmp_epu8_mask(_mm512_sub_epi8(source_vec.zmm, a_upper_vec),
                                                                        subtract26_vec, _MM_CMPINT_LT);
                        __mmask64 fold_mask = (third_byte_positions | is_upper_ascii) & prefix_mask_3;
                        __m512i folded =
                            _mm512_mask_add_epi8(source_vec.zmm, fold_mask, source_vec.zmm, _mm512_set1_epi8(0x20));
                        _mm512_mask_storeu_epi8(target, prefix_mask_3, folded);
                        target += three_byte_length, source += three_byte_length, source_length -= three_byte_length;
                        continue;
                    }
                }

                // No special 3-byte cases found - fold ASCII A-Z, copy 3-byte unchanged
                __mmask64 is_upper_ascii =
                    _mm512_cmp_epu8_mask(_mm512_sub_epi8(source_vec.zmm, a_upper_vec), subtract26_vec, _MM_CMPINT_LT);
                __m512i folded =
                    _mm512_mask_add_epi8(source_vec.zmm, is_upper_ascii & prefix_mask_3, source_vec.zmm, x20_vec);
                _mm512_mask_storeu_epi8(target, prefix_mask_3, folded);
                target += three_byte_length, source += three_byte_length, source_length -= three_byte_length;
                continue;
            }
        }

        // 4. Handle 4-byte sequences (emoji, rare scripts): detect lead bytes (11110xxx = F0-F7)
        {
            __mmask64 is_valid_four_byte_only = is_four_byte_lead | is_cont;
            sz_size_t four_byte_length = sz_u64_ctz(~is_valid_four_byte_only | ~load_mask);

            // Don't split a 4-byte sequence: find first incomplete lead and truncate there
            if (four_byte_length >= 1 && four_byte_length < 64) {
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

                if ((four_byte_leads_in_prefix & ~(is_emoji_lead & is_four_byte_lead)) == 0) {
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

            sz_rune_t rune;
            sz_rune_length_t rune_length;
            sz_rune_parse(source, &rune, &rune_length);
            if (rune_length == sz_utf8_invalid_k) {
                // Invalid UTF-8: copy byte as-is and continue
                *target++ = *source++;
                source_length--;
                continue;
            }

            sz_rune_t folded[4];
            sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded);
            for (sz_size_t i = 0; i != folded_count; ++i) target += sz_rune_export(folded[i], (sz_u8_t *)target);
            source += rune_length;
            source_length -= rune_length;
        }
    }

    return (sz_size_t)(target - target_start);
}

SZ_PUBLIC sz_cptr_t sz_utf8_case_insensitive_find_ice( //
    sz_cptr_t haystack, sz_size_t haystack_length,     //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length) {
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

#endif // STRINGZILLA_UTF8_UNPACK_H_
