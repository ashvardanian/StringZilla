/**
 *  @brief Uncased UTF-8 substring search, comparison & case-invariance checks.
 *  @file include/stringzilla/utf8_uncased.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/utf8_uncased_fold.h
 *
 *  Public Core API:
 *
 *  - `sz_utf8_uncased_search` - uncased substring search in UTF-8 strings
 *  - `sz_utf8_uncased_order` - uncased lexicographical comparison of UTF-8 strings
 *  - `sz_utf8_find_cased` - pointer to the first cased (foldable) codepoint, or NULL if fully caseless
 *
 *  All comparison and matching uses full Unicode Case Folding (UAX #21 / CaseFolding.txt), including
 *  one-to-many expansions (e.g., 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)). Folding is
 *  locale-independent and deterministic across platforms.
 *
 *  On fast vectorized paths there may be significant algorithmic differences between ISA versions;
 *  the per-script SIMD kernels (Ice Lake, ...) consume needle metadata produced by the ISA-agnostic
 *  classifier in `utf8_uncased/serial.h`.
 */
#ifndef STRINGZILLA_UTF8_UNCASED_H_
#define STRINGZILLA_UTF8_UNCASED_H_

#include "stringzilla/utf8_runes/serial.h"

#ifdef __cplusplus
extern "C" {
#endif
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
typedef struct sz_utf8_uncased_needle_metadata_t {
    sz_size_t offset_in_unfolded; // Number of bytes in the "unsafe LONG NeedLe" before the safe & folded part
    sz_size_t length_in_unfolded; // Number of bytes in the safe part of the actual "NeedLe" before folding
    sz_u8_t folded_slice[16];
    sz_u8_t folded_slice_length;
    sz_u8_t probe_second; // Position of the second relevant character in the folded slice
    sz_u8_t probe_third;  // Position of the third relevant character in the folded slice
    sz_u8_t kernel_id;    // The unique identifier of the kernel best suited for searching this needle
} sz_utf8_uncased_needle_metadata_t;
/**
 *  @brief Safety profile for a single character across all script paths.
 *
 *  A safety profile for a "needle" is a set of conditions that allow simpler haystack on-the-fly folding
 *  than the proper `sz_utf8_uncased_fold`, but without losing any possible matches. That's typically achieved
 *  finding parts of the needle, that never appear in any multi-byte expansions of complex characters, so
 *  we don't need to shuffle data within a CPU register - just swap some byte sequences with others.
 *
 *  Assuming the complexity of Unicode, the number of such rules to take care of is quite significant, so
 *  it's hard to achieve matching speeds beyond 500 MB/s for arbitrary needles. However, if separate them
 *  by language groups and Unicode subranges, the 5 GB/s target becomes approachable.
 */
typedef enum {
    sz_utf8_uncased_rune_unknown_k = 0,
    /**
     *  @brief Describes a safety-class profile for contextually-safe ASCII characters, mostly for English text,
     *         exclusive to single-byte characters without case-folding "collisions" and ambiguities.
     *
     *  If all of the following @b needle-constraints are satisfied, our uncased UTF-8 substring search
     *  becomes no more than a trivial uncased ASCII substring search, where the only @b haystack-folding
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
    sz_utf8_uncased_rune_ascii_invariant_k = 1,

    /**
     *  @brief Describes a safety-class profile for contextually-safe ASCII + Latin-1 Supplements designed mostly
     *         for Western European languages (like French, German, Spanish, & Portuguese) with a mixture of
     *         single-byte and double-byte UTF-8 character sequences.
     *
     *  @sa sz_utf8_uncased_rune_ascii_invariant_k for a simpler variant.
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
     *  This also inherits some of the contextual limitations from `sz_utf8_uncased_rune_ascii_invariant_k`, but not all!
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
     *  `sz_utf8_uncased_rune_ascii_invariant_k`:
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
    sz_utf8_uncased_rune_safe_western_europe_k = 2,

    /**
     *  @brief Describes a safety-class profile for contextually-safe ASCII + Latin-1 + Latin-A Supplements designed
     *         mostly for Central European languages (like Polish, Czech, & Hungarian) and Turkish with a mixture of
     *         single-byte, double-byte, and rare triple-byte UTF-8 character sequences.
     *
     *  @sa sz_utf8_uncased_rune_ascii_invariant_k for a simpler variant.
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
     *  `sz_utf8_uncased_rune_ascii_invariant_k`:
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
     *  We also inherit one more limitation from the Latin-1 profile, same as `sz_utf8_uncased_rune_safe_western_europe_k`:
     *
     *  - 'å' (U+00E5, C3 A5) - is the folding target of both 'Å' (U+00C5, C3 85) in Latin-1 and
     *    the Angstrom Sign 'Å' (U+212B, E2 84 AB) → 'å' (U+00E5, C3 A5), so needle cannot contain 'å' (U+00E5, C3 A5)
     *    to avoid ambiguity.
     *
     *  This means, that all of the ASCII and Latin-1 characters beyond the rules above are considered "safe"
     *  for this profile. This includes English alphabet letters like: b, c, d, e, g, k, m, o, p, q, r, u, v, x, z,
     *  as well as digits, punctuation, symbols, and control characters.
     */
    sz_utf8_uncased_rune_safe_central_europe_k = 3,

    /**
     *  @brief Describes a safety-class profile for contextually-safe ASCII + Basic Cyrillic designed mostly
     *         for East Slavic languages (like Russian, Ukrainian, & Belarusian) and South Slavic languages
     *         (like Serbian, Bulgarian, & Macedonian), but excluding Cyrillic Extensions.
     *
     *  @sa sz_utf8_uncased_rune_ascii_invariant_k for the inherited ASCII rules.
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
     *  We inherit ALL contextual ASCII limitations from `sz_utf8_uncased_rune_ascii_invariant_k`:
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
    sz_utf8_uncased_rune_safe_cyrillic_k = 4,

    /**
     *  @brief Describes a safety-class profile for contextually-safe ASCII + Basic Greek designed mostly
     *         for Modern Greek (Demotic) text with a mixture of single-byte and double-byte UTF-8
     *         character sequences.
     *
     *  @sa sz_utf8_uncased_rune_ascii_invariant_k for the inherited ASCII rules.
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
     *  kernel path (sz_utf8_uncased_rune_safe_western_europe_k), not the Greek path. The Greek kernel
     *  only handles characters that originate in the Greek block.
     *
     *  We inherit @b all contextual ASCII limitations from `sz_utf8_uncased_rune_ascii_invariant_k`:
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
    sz_utf8_uncased_rune_safe_greek_k = 5,

    /**
     *  @brief Describes a safety-class profile for contextually-safe ASCII + Basic Armenian.
     *  @sa sz_utf8_uncased_rune_ascii_invariant_k for the inherited ASCII rules.
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
     *  We inherit @b all contextual ASCII limitations from `sz_utf8_uncased_rune_ascii_invariant_k`:
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
    sz_utf8_uncased_rune_safe_armenian_k = 6,

    /**
     *  @brief Describes a safety-class profile for contextually-safe ASCII + Latin-1 + Latin Extended Additional.
     *  @sa sz_utf8_uncased_rune_safe_central_europe_k for the inherited Latin rules.
     *
     *  These kernels extend Latin-1/A/B with Vietnamese characters:
     *  - Everything from `sz_utf8_uncased_rune_safe_central_europe_k` (ASCII + Latin-1/A)
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
     * `sz_utf8_uncased_rune_ascii_invariant_k`:
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
    sz_utf8_uncased_rune_safe_vietnamese_k = 7,

    /**
     *  @brief Describes a safety-class profile for Georgian Mkhedruli script.
     *  @sa sz_utf8_uncased_rune_ascii_invariant_k for inherited ASCII rules.
     *
     *  Georgian Mkhedruli (U+10D0-U+10FF) is caseless - no folding needed for Georgian characters.
     *  Only ASCII A-Z folding for mixed text. Mtavruli (U+1C90-U+1CBF), Asomtavruli (U+10A0-U+10C5),
     *  and Nuskhuri (U+2D00-U+2D25) trigger alarm for serial fallback (rare in modern text).
     *
     *  All Georgian scripts use 3-byte UTF-8 sequences and fold to 3-byte sequences, so there
     *  are no length changes during case folding - making this the simplest non-ASCII kernel.
     */
    sz_utf8_uncased_rune_safe_georgian_k = 8,

    sz_utf8_uncased_rune_invariant_k = 9,
    sz_utf8_uncased_rune_fallback_serial_k = 255,
} sz_utf8_uncased_rune_safety_profile_t_;

#pragma region Core API

/**
 *  @brief Uncased lexicographic comparison of two UTF-8 strings.
 *
 *  Compares strings using Unicode case folding rules, producing consistent ordering regardless of
 *  letter case. Implements the same full Unicode Case Folding as `sz_utf8_uncased_fold`, including
 *  all one-to-many expansions (e.g., 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)) and bicameral script mappings.
 *
 *  Unlike simple byte comparison, this function correctly handles multi-byte UTF-8 sequences
 *  and expansion characters. Comparison is performed codepoint-by-codepoint after folding,
 *  not byte-by-byte, ensuring linguistically correct results.
 *
 *  @param a First UTF-8 string to compare.
 *  @param a_length Number of bytes in the first string.
 *  @param b Second UTF-8 string to compare.
 *  @param b_length Number of bytes in the second string.
 *  @return @c sz_less_k if a < b, @c sz_equal_k if a == b, @c sz_greater_k if a > b.
 *
 *  @note Malformed UTF-8 is handled losslessly: any byte that does not begin a well-formed codepoint is treated
 *        as a single literal byte (folded to itself and compared byte-for-byte, never as a Unicode codepoint),
 *        and processing resyncs at the next byte. All uncased operations - find, order, and violation - share
 *        this contract, so results on invalid input are well-defined and consistent across backends.
 *
 *  @example Basic usage:
 *  @code
 *      sz_ordering_t result = sz_utf8_uncased_order("Hello", 5, "HELLO", 5);
 *      // result == sz_equal_k
 *
 *      result = sz_utf8_uncased_order("straße", 7, "STRASSE", 7);
 *      // result == sz_equal_k ('ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73))
 *  @endcode
 */
SZ_DYNAMIC sz_ordering_t sz_utf8_uncased_order(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);

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
 *  @section utf8_find_cased_usage Use Case
 *
 *  This function enables an important optimization: if both haystack and needle are fully
 *  case-agnostic, then `sz_find()` can be used directly instead of the slower
 *  `sz_utf8_uncased_search()`. This is particularly valuable for:
 *
 *  - CJK text (Chinese, Japanese, Korean) - always caseless
 *  - Numeric data and punctuation-heavy content
 *  - Middle Eastern scripts (Arabic, Hebrew, Persian)
 *  - South/Southeast Asian scripts (Thai, Hindi, Vietnamese without Latin)
 *
 *  @param str UTF-8 string to check.
 *  @param length Number of bytes in the string.
 *  @return sz_true_k if all codepoints are case-agnostic, sz_false_k otherwise.
 *
 *  @example Optimization pattern:
 *  @code
 *      sz_cptr_t haystack = "价格：¥1234";  // Chinese + punctuation + digits
 *      sz_cptr_t needle = "¥1234";
 *
 *      if (sz_utf8_find_cased(haystack, haystack_len) == SZ_NULL_CHAR &&
 *          sz_utf8_find_cased(needle, needle_len) == SZ_NULL_CHAR) {
 *          // Fast path: both strings are fully caseless, so use binary search
 *          result = sz_find(haystack, haystack_len, needle, needle_len);
 *      } else {
 *          // Slow path: full uncased search
 *          result = sz_utf8_uncased_search(haystack, haystack_len,
 *                                                  needle, needle_len, &match_len);
 *      }
 *  @endcode
 *
 *  @note This function is conservative: it returns sz_false_k for any codepoint that
 *        participates in case folding, even if the specific instance wouldn't change.
 *        For example, lowercase 'a' returns false because it's a case-folding target.
 */
SZ_DYNAMIC sz_cptr_t sz_utf8_find_cased(sz_cptr_t str, sz_size_t length);

/** @copydoc sz_utf8_uncased_order */
SZ_PUBLIC sz_ordering_t sz_utf8_uncased_order_serial( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);

/** @copydoc sz_utf8_find_cased */
SZ_PUBLIC sz_cptr_t sz_utf8_find_cased_serial(sz_cptr_t str, sz_size_t length);

#if SZ_USE_ICELAKE
/** @copydoc sz_utf8_uncased_order */
SZ_PUBLIC sz_ordering_t sz_utf8_uncased_order_icelake( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);

/** @copydoc sz_utf8_find_cased */
SZ_PUBLIC sz_cptr_t sz_utf8_find_cased_icelake(sz_cptr_t str, sz_size_t length);
#endif

#if SZ_USE_HASWELL
/** @copydoc sz_utf8_uncased_order */
SZ_PUBLIC sz_ordering_t sz_utf8_uncased_order_haswell( //
    sz_cptr_t a, sz_size_t a_length,                   //
    sz_cptr_t b, sz_size_t b_length);
/** @copydoc sz_utf8_find_cased */
SZ_PUBLIC sz_cptr_t sz_utf8_find_cased_haswell(sz_cptr_t str, sz_size_t length);
#endif

#if SZ_USE_NEON
/** @copydoc sz_utf8_uncased_order */
SZ_PUBLIC sz_ordering_t sz_utf8_uncased_order_neon( //
    sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);

/** @copydoc sz_utf8_find_cased */
SZ_PUBLIC sz_cptr_t sz_utf8_find_cased_neon(sz_cptr_t str, sz_size_t length);
#endif

#pragma region Declarations

/**
 *  @brief Uncased substring search in UTF-8 strings.
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
 *  @section utf8_uncased_search_algo Algorithmic Considerations
 *
 *  Uncased search with full Unicode case folding is fundamentally harder than byte-level search
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
 *       uni-algo - Unicode algorithms implementation with uncased search
 *
 *  @param haystack UTF-8 string to be searched.
 *  @param haystack_length Number of bytes in the haystack buffer.
 *  @param needle UTF-8 substring to search for.
 *  @param needle_metadata Optional pre-computed needle metadata for reuse across multiple searches.
 *  @param matched_length Number of bytes in the matched region.
 *  @return Pointer to the first matching substring from @p haystack, or @c SZ_NULL_CHAR if not found.
 *
 *  @note Malformed UTF-8 is handled losslessly: any byte that does not begin a well-formed codepoint is matched
 *        byte-for-byte as a single literal byte (never as a Unicode codepoint) and processing resyncs at the next
 *        byte, so a search over invalid input is well-defined and identical across all backends.
 */
SZ_DYNAMIC sz_cptr_t sz_utf8_uncased_search(       //
    sz_cptr_t haystack, sz_size_t haystack_length, //
    sz_cptr_t needle, sz_size_t needle_length,     //
    sz_utf8_uncased_needle_metadata_t *needle_metadata, sz_size_t *matched_length);

/** @copydoc sz_utf8_uncased_search */
SZ_PUBLIC sz_cptr_t sz_utf8_uncased_search_serial( //
    sz_cptr_t haystack, sz_size_t haystack_length, //
    sz_cptr_t needle, sz_size_t needle_length,     //
    sz_utf8_uncased_needle_metadata_t *needle_metadata, sz_size_t *matched_length);

#if SZ_USE_ICELAKE
/** @copydoc sz_utf8_uncased_search */
SZ_PUBLIC sz_cptr_t sz_utf8_uncased_search_icelake( //
    sz_cptr_t haystack, sz_size_t haystack_length,  //
    sz_cptr_t needle, sz_size_t needle_length,      //
    sz_utf8_uncased_needle_metadata_t *needle_metadata, sz_size_t *matched_length);
#endif

#if SZ_USE_HASWELL
/** @copydoc sz_utf8_uncased_search */
SZ_PUBLIC sz_cptr_t sz_utf8_uncased_search_haswell( //
    sz_cptr_t haystack, sz_size_t haystack_length,  //
    sz_cptr_t needle, sz_size_t needle_length,      //
    sz_utf8_uncased_needle_metadata_t *needle_metadata, sz_size_t *matched_length);
#endif

#if SZ_USE_NEON
/** @copydoc sz_utf8_uncased_search */
SZ_PUBLIC sz_cptr_t sz_utf8_uncased_search_neon(   //
    sz_cptr_t haystack, sz_size_t haystack_length, //
    sz_cptr_t needle, sz_size_t needle_length,     //
    sz_utf8_uncased_needle_metadata_t *needle_metadata, sz_size_t *matched_length);
#endif

#pragma endregion // Core API

/*  Backends in serial-first order: every SIMD/ISA header includes `serial.h` for the shared
 *  scaffolding (verify helpers, hash-free finders, Rabin-Karp serial search, the safety classifier
 *  and needle-metadata builder), then layers its own kernels (or delegates back to serial). */
#include "stringzilla/utf8_uncased/serial.h"
#include "stringzilla/utf8_uncased/icelake.h"
#include "stringzilla/utf8_uncased/haswell.h"
#include "stringzilla/utf8_uncased/neon.h"
#include "stringzilla/utf8_uncased/v128.h"
#include "stringzilla/utf8_uncased/rvv.h"
#include "stringzilla/utf8_uncased/lasx.h"
#include "stringzilla/utf8_uncased/powervsx.h"

/*  Pick the right implementation for the uncased UTF-8 algorithms.
 *  To override this behavior and precompile all backends - set `SZ_DYNAMIC_DISPATCH` to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_cptr_t sz_utf8_uncased_search(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                            sz_size_t needle_length, sz_utf8_uncased_needle_metadata_t *needle_metadata,
                                            sz_size_t *matched_length) {
#if SZ_USE_V128
    return sz_utf8_uncased_search_v128(haystack, haystack_length, needle, needle_length, needle_metadata,
                                       matched_length);
#elif SZ_USE_RVV
    return sz_utf8_uncased_search_rvv(haystack, haystack_length, needle, needle_length, needle_metadata,
                                      matched_length);
#elif SZ_USE_LASX
    return sz_utf8_uncased_search_lasx(haystack, haystack_length, needle, needle_length, needle_metadata,
                                       matched_length);
#elif SZ_USE_POWERVSX
    return sz_utf8_uncased_search_powervsx(haystack, haystack_length, needle, needle_length, needle_metadata,
                                           matched_length);
#elif SZ_USE_ICELAKE
    return sz_utf8_uncased_search_icelake(haystack, haystack_length, needle, needle_length, needle_metadata,
                                          matched_length);
#elif SZ_USE_HASWELL
    return sz_utf8_uncased_search_haswell(haystack, haystack_length, needle, needle_length, needle_metadata,
                                          matched_length);
#elif SZ_USE_NEON
    return sz_utf8_uncased_search_neon(haystack, haystack_length, needle, needle_length, needle_metadata,
                                       matched_length);
#else
    return sz_utf8_uncased_search_serial(haystack, haystack_length, needle, needle_length, needle_metadata,
                                         matched_length);
#endif
}

SZ_DYNAMIC sz_ordering_t sz_utf8_uncased_order(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    return sz_utf8_uncased_order_serial(a, a_length, b, b_length);
}

SZ_DYNAMIC sz_cptr_t sz_utf8_find_cased(sz_cptr_t str, sz_size_t length) {
#if SZ_USE_V128
    return sz_utf8_find_cased_v128(str, length);
#elif SZ_USE_RVV
    return sz_utf8_find_cased_rvv(str, length);
#elif SZ_USE_LASX
    return sz_utf8_find_cased_lasx(str, length);
#elif SZ_USE_POWERVSX
    return sz_utf8_find_cased_powervsx(str, length);
#elif SZ_USE_ICELAKE
    return sz_utf8_find_cased_icelake(str, length);
#elif SZ_USE_HASWELL
    return sz_utf8_find_cased_haswell(str, length);
#elif SZ_USE_NEON
    return sz_utf8_find_cased_neon(str, length);
#else
    return sz_utf8_find_cased_serial(str, length);
#endif
}

#endif            // !SZ_DYNAMIC_DISPATCH
#pragma endregion // Compile Time Dispatching

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_UNCASED_H_
