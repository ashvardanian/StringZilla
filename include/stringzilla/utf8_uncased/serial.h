/**
 *  @file include/stringzilla/utf8_uncased/serial.h
 *  @author Ash Vardanian
 *  @date November 23, 2025
 *  @brief Uncased UTF-8 search, comparison & invariance checks: serial scaffolding.
 *
 *  @sa include/stringzilla/utf8_uncased.h
 */
#ifndef STRINGZILLA_UTF8_UNCASED_SERIAL_H_
#define STRINGZILLA_UTF8_UNCASED_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h"
#include "stringzilla/utf8_uncased_fold/serial.h" // `sz_unicode_fold_codepoint_`
#include "stringzilla/find/serial.h"              // `sz_find_serial_`

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief A needle analysed once by @c sz_utf8_uncased_needle_init_best for any number of
 *      uncased searches, under any capabilities.
 *
 *  Preparing picks the per-script kernel for the needle in two steps:
 *
 *  1. locating the longest "safe" slice of the needle for the different SIMD folding kernels,
 *  2. shrinking it to the most diverse slice that fits into @c folded_slice when case-folded.
 *
 *  Unlike the exact substring search kernels, it uses 4 probe positions instead of 3:
 *
 *  - first: implicit at `folded_slice[0]`
 *  - second: @c probe_second
 *  - third: @c probe_third
 *  - last: implicit at `folded_slice[folded_slice_length - 1]`
 *
 *  The struct points at the needle rather than copying it, so the needle must outlive it.
 */
typedef struct sz_utf8_uncased_needle_t {

    /** The needle's text, read by every search. */
    sz_cptr_t start;
    sz_size_t length;

    /** Bytes of the needle before the slice that @c folded_slice holds folded. */
    sz_size_t offset_in_unfolded;

    /** Bytes of the needle that @c folded_slice holds folded. */
    sz_size_t length_in_unfolded;
    sz_u8_t folded_slice[16];
    sz_u8_t folded_slice_length;

    /** Offset of the second probe in @c folded_slice. */
    sz_u8_t probe_second;

    /** Offset of the third probe in @c folded_slice. */
    sz_u8_t probe_third;

    /** The @c sz_utf8_uncased_rune_*_k profile whose kernel searches this needle. */
    sz_u8_t script;
} sz_utf8_uncased_needle_t;

/**
 *  @brief Safety profile for a single character across all script paths.
 *
 *  A safety profile for a "needle" is a set of conditions that allow simpler haystack on-the-fly
 *  folding than the proper @c sz_utf8_uncased_fold_best, but without losing any possible matches.
 *  That's typically achieved finding parts of the needle, that never appear in any multi-byte
 *  expansions of complex characters, so we don't need to shuffle data within a CPU register - just
 *  swap some byte sequences with others.
 *
 *  Given the complexity of Unicode, the number of such rules to take care of is quite significant,
 *  so it's hard to achieve matching speeds beyond 500 MB/s for arbitrary needles. However, if we
 *  separate them by language groups and Unicode subranges, the 5 GB/s target becomes approachable.
 */
typedef enum {

    /**
     *  @brief Safety profile for contextually-safe ASCII characters, mostly for English text,
     *      exclusive to single-byte characters without case-folding "collisions" and ambiguities.
     *
     *  If all of the following @b needle-constraints are satisfied, our uncased UTF-8 substring
     *  search becomes no more than a trivial uncased ASCII substring search, where the only
     *  @b haystack-folding operation to be applied is mapping A-Z to a-z:
     *
     *  - 'a' (U+0061, 61) - can't be last; can't precede 'ʾ' (U+02BE, CA BE) to avoid:
     *    - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
     *  - 'f' (U+0066, 66) - can't be first or last; can't follow 'f' (U+0066, 66); can't precede
     *    'f' (U+0066, 66), 'i' (U+0069, 69), 'l' (U+006C, 6C) to avoid:
     *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'h' (U+0068, 68) - can't be last; can't precede '̱' (U+0331, CC B1) to avoid:
     *    - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66); can't precede
     *    '̇' (U+0307, CC 87) to avoid:
     *    - 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *  - 'j' (U+006A, 6A) - can't be last; can't precede '̌' (U+030C, CC 8C) to avoid:
     *    - 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C)
     *  - 'k' (U+006B, 6B) - can't be present at all, as a folding target of the Kelvin sign:
     *    - 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B)
     *  - 'l' (U+006C, 6C) - can't be first; can't follow 'f' (U+0066, 66) to avoid:
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'n' (U+006E, 6E) - can't be first; can't follow 'ʼ' (U+02BC, CA BC) to avoid:
     *    - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
     *  - 's' (U+0073, 73) - can't be present at all, because it's a folding target of the
     *    old S sign:
     *    - 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73); can't precede
     *    '̈' (U+0308, CC 88) to avoid:
     *    - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 'w' (U+0077, 77) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
     *  - 'y' (U+0079, 79) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
     *
     *  This means, that all ASCII characters beyond the rules above are considered "safe" for this
     *  profile, including English letters b, c, d, e, g, m, o, p, q, r, u, v, x, and z, as well as
     *  digits, punctuation, symbols, and control characters.
     */
    sz_utf8_uncased_rune_ascii_invariant_k = 1,

    /**
     *  @brief Safety profile for contextually-safe ASCII + Latin-1 Supplements designed mostly for
     *      Western European languages (like French, German, Spanish, & Portuguese) with a mixture
     *      of single-byte and double-byte UTF-8 character sequences.
     *
     *  Unlike the ASCII fast path, these kernels fold a wider range of characters:
     *  - 26x original ASCII uppercase letters: 'A' (U+0041, 41) → 'a' (U+0061, 61), 'Z' (U+005A,
     *    5A) → 'z' (U+007A, 7A)
     *  - 30x Latin-1 supplement uppercase letters for French, German, Spanish, & Portuguese, like:
     *    - 'À' (U+00C0, C3 80) → 'à' (U+00E0, C3 A0),
     *    - 'Ñ' (U+00D1, C3 91) → 'ñ' (U+00F1, C3 B1),
     *    - 'Ü' (U+00DC, C3 9C) → 'ü' (U+00FC, C3 BC)
     *  - 1x special case of folding from Latin-1 to ASCII pair, preserving byte-width:
     *    - 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
     *
     *  This doesn't cover Latin-A and Latin-B extensions (like Polish, Czech, Hungarian, & Turkish
     *  letters). This also inherits some of the contextual limitations from
     *  @c sz_utf8_uncased_rune_ascii_invariant_k, but not all!
     *
     *  The lowercase 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73) is folded in-place (2 bytes
     *  → 2 bytes). This creates a mid-expansion matching issue: if a needle starts or ends with
     *  's', the SIMD kernel might find a match at the second byte of 'ß' (the UTF-8 continuation
     *  byte 0x9F) instead of at a codepoint boundary. Example: haystack "ßStra" folds to "ssstra",
     *  needle "sstra" matches at position 1 (the 0x9F byte of 'ß'). To avoid this, 's' is only safe
     *  when not at the start or end of the needle (contextual restriction).
     *
     *  The uppercase 'ẞ' (U+1E9E, E1 BA 9E) also folds into "ss" (U+0073 U+0073, 73 73), but is
     *  outside of Latin-1. In UTF-8 it is a 3-byte sequence, so it resizes into a 2-byte
     *  sequence when folded. Luckily for us, it's almost never used in practice: introduced to
     *  Unicode in 2008 and officially adopted into German orthography in 2017. When processing
     *  the haystack, we check if 'ẞ' appears, and if so, we revert to serial processing for that
     *  tiny block of text.
     *
     *  Another place where 's' (U+0073, 73) appears are ligatures 'ﬅ' (U+FB05, EF AC 85) and 'ﬆ'
     *  (U+FB06, EF AC 86) that both fold into "st" (U+0073 U+0074, 73 74). They also result in
     *  serial fallback when detected in the haystack. If we detect all of those ligatures from 'ﬀ'
     *  (U+FB00, EF AC 80) to 'ﬆ' (U+FB06, EF AC 86), we can safely allow both 'f' (U+0066, 66) and
     *  'l' (U+006C, 6C).
     *
     *  There is one more 3-byte problematic range to consider - from (E1 BA 96) to (E1 BA 9A),
     *  which includes: 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1), 'ẗ' (U+1E97, E1 BA
     *  97) → "ẗ" (U+0074 U+0308, 74 CC 88), 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC
     *  8A), 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A), 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ"
     *  (U+0061 U+02BE, 61 CA BE). If we correctly detect that range in the haystack, we can safely
     *  allow 'h' (U+0068, 68), 't' (U+0074, 74), 'w' (U+0077, 77), 'y' (U+0079, 79), and 'a'
     *  (U+0061, 61) in needles.
     *
     *  There is also a Unicode rule for folding the Kelvin 'K' (U+212A, E2 84 AA) into 'k' (U+006B,
     *  6B). That sign is extremely rare in Western European languages, while the lowercase 'k' is
     *  obviously common in German and English. In French, Spanish, and Portuguese - less so. So we
     *  add one more check for 'K' (U+212A, E2 84 AA) in the haystack, and if detected, again -
     *  revert to serial. Similarly, we check for 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73). It's
     *  archaic in modern languages but theoretically possible in historical texts.
     *
     *  So we allow 'k' unconditionally and inherit/extend the following limitations from
     *  @c sz_utf8_uncased_rune_ascii_invariant_k:
     *
     *  'i' (U+0069, 69) can't be first or last; can't follow 'f' (U+0066, 66); can't precede '̇'
     *  (U+0307, CC 87) to avoid 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87). It's the
     *  Turkish dotted capital I that expands into a 3-byte sequence when folded. It typically
     *  appears at the start of words, like: İstanbul (the city), İngilizce (English language).
     *
     *  'j' (U+006A, 6A) can't be last; can't precede '̌' (U+030C, CC 8C) to avoid: 'ǰ' (U+01F0, C7
     *  B0) → "ǰ" (U+006A U+030C, 6A CC 8C). It's the "J with Caron", used in phonetic transcripts
     *  and romanization of Iranian, Armenian, Georgian.
     *
     *  'n' (U+006E, 6E) can't be first; can't follow 'ʼ' (U+02BC, CA BC) to avoid: 'ŉ' (U+0149, C5
     *  89) → "ʼn" (U+02BC U+006E, CA BC 6E). It's mostly used in Afrikaans (South Africa/Namibia),
     *  contracted from Dutch "een" (one/a), in phrases like "Dit is 'n boom" (It is a tree), "Dit
     *  is 'n appel" (This is an apple).
     *
     *  's' (U+0073, 73) can't be first or last, or part of the folded "ss" (U+0073 U+0073, 73 73)
     *  prefix or suffix, to avoid mid-ß-expansion matches: 'ß' (U+00DF, C3 9F) → "ss" (U+0073
     *  U+0073, 73 73) is folded in-place, so a needle starting/ending with 's' could match at
     *  position 1 (the 0x9F continuation byte). Example: "ßStra" → "ssstra", needle "sstra" would
     *  match at the second byte of 'ß'. Needles with 's' in the middle are safe.
     *
     *  We also add one more limitation for a special 2-byte character that is an irregular folding
     *  target of codepoints of different length:
     *
     *  - 'å' (U+00E5, C3 A5) - is the folding target of both 'Å' (U+00C5, C3 85) in Latin-1 and the
     *    Angstrom Sign 'Å' (U+212B, E2 84 AB) → 'å' (U+00E5, C3 A5), so needle cannot contain 'å'
     *    (U+00E5, C3 A5) to avoid ambiguity.
     *
     *  There is also a Latin-1 character that doesn't change the width, but we still ban it from
     *  the safe strings:
     *
     *  'µ' (U+00B5, C2 B5), the mathematical Micro sign, folds to the Greek lowercase 'μ' (U+03BC,
     *  CE BC), which is also a folding target of the uppercase Greek letter 'Μ' (U+039C, CE 9C).
     *  To avoid having to filter/check for Greek symbols in the haystacks, we ban the Micro sign
     *  from the needles.
     *
     *  This means, that all ASCII and Latin-1 characters beyond the rules above are "safe" for this
     *  profile, including English letters b, c, d, e, g, k, m, o, p, q, r, u, v, x, and z, as well
     *  as digits, punctuation, symbols, and control characters.
     *
     *  @sa sz_utf8_uncased_rune_ascii_invariant_k for a simpler variant.
     */
    sz_utf8_uncased_rune_safe_western_europe_k = 2,

    /**
     *  @brief Safety profile for contextually-safe ASCII + Latin-1 + Latin-A Supplements designed
     *      mostly for Central European languages (like Polish, Czech, & Hungarian) and Turkish with
     *      a mixture of single-byte, double-byte, and rare triple-byte UTF-8 character sequences.
     *
     *  Unlike the ASCII fast path, these kernels fold a wider range of characters:
     *  - 26x original ASCII uppercase letters: 'A' (U+0041, 41) → 'a' (U+0061, 61), 'Z' (U+005A,
     *    5A) → 'z' (U+007A, 7A)
     *  - 30x Latin-1 supplement uppercase letters for French, German, Spanish, & Portuguese, like:
     *    - 'À' (U+00C0, C3 80) → 'à' (U+00E0, C3 A0),
     *    - 'Ñ' (U+00D1, C3 91) → 'ñ' (U+00F1, C3 B1),
     *    - 'Ü' (U+00DC, C3 9C) → 'ü' (U+00FC, C3 BC)
     *  - 63x Latin-A extension uppercase letters for Polish, Czech, Hungarian, & Turkish, like:
     *    - 'Ą' (U+0104, C4 84) → 'ą' (U+0105, C4 85),
     *    - 'Ł' (U+0141, C5 81) → 'ł' (U+0142, C5 82),
     *    - 'Č' (U+010C, C4 8C) → 'č' (U+010D, C4 8D)
     *
     *  This doesn't cover Latin-B extensions (like Baltic, Romanian, & Vietnamese letters), and is
     *  not optimal for Western European languages, assuming the lack of "ss" handling for German
     *  Eszett 'ß' (U+00DF, C3 9F). There is, however, a huge overlap between the Central European,
     *  Western European, and Turkic scripts:
     *
     *  - Czech has the highest overlap - nearly half of Czech words with Latin-A characters (like
     *    Č, Ř, Š, Ž) also contain Latin-1 characters (Á, É, Í, Ó, Ú, Ý). Examples: sčítání,
     *    dalšími, řízení, systémů.
     *  - Polish has minimal word-level overlap because Polish only uses Ó/ó from Latin-1, and most
     *    Polish-specific letters (Ą, Ę, Ł, Ń, Ś, Ź, Ż) are in Latin-A. Example: mieszkańców (has
     *    both ń and ó).
     *  - Turkish has moderate overlap from Ç, Ö, Ü (Latin-1) mixing with Ğ, İ, Ş (Latin-A).
     *    Examples: içeriği, öğrencilerden, dönüşüm.
     *
     *  All those languages are not always related linguistically:
     *
     *  - Czech and Polish are Slavic languages, using Latin script with háčeks since 15th century.
     *  - Hungarian is a Uralic language, that adopted Latin script in 11th century.
     *  - Turkish is a Turkic (Altaic) language, that switched from Arabic to Latin script in 1928.
     *    Atatürk's 1928 alphabet reform:
     *    - borrowed Ç, Ö, Ü from French and German subsets of Latin-1 Supplement (C3 lead byte).
     *    - introduced Ğ, İ, Ş, which ended up in the Latin Extended-A (C4/C5 lead byte).
     *
     *  But due to overlapping character sets, they can all benefit from the same fast path.
     *
     *  There is also a Unicode rule for folding the Kelvin 'K' (U+212A, E2 84 AA) into 'k' (U+006B,
     *  6B). That sign is extremely rare in Western European languages, while the lowercase 'k' is
     *  very common in Turkish, Czech, Polish. So we add one more check for 'K' (U+212A, E2 84 AA)
     *  in the haystack, and if detected, again - revert to serial. Same logic applies to 'ſ'
     *  (U+017F, C5 BF) → 's' (U+0073, 73).
     *
     *  The Turkish dotted 'İ' (U+0130, C4 B0) expands into a 3-byte sequence. We detect it when
     *  scanning through the haystack and fall back to the serial algorithm. That's pretty much the
     *  only triple-byte sequence we will frequently encounter in Turkish text.
     *
     *  We inherit most contextual limitations for some of the ASCII characters from
     *  @c sz_utf8_uncased_rune_ascii_invariant_k:
     *
     *  - 'a' (U+0061, 61) - can't be last; can't precede 'ʾ' (U+02BE, CA BE) to avoid:
     *    - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
     *  - 'f' (U+0066, 66) - can't be first or last; can't follow 'f' (U+0066, 66); can't precede
     *    'f' (U+0066, 66), 'i' (U+0069, 69), 'l' (U+006C, 6C) to avoid:
     *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'h' (U+0068, 68) - can't be last; can't precede '̱' (U+0331, CC B1) to avoid:
     *    - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66); can't precede
     *    '̇' (U+0307, CC 87) to avoid:
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
     *  - 's' (U+0073, 73) - can't be first or last; can't follow 's' (U+0073, 73); can't precede
     *    's' (U+0073, 73), 't' (U+0074, 74) to avoid:
     *    - 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
     *    - 'ẞ' (U+1E9E, E1 BA 9E) → "ss" (U+0073 U+0073, 73 73)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73); can't precede
     *    '̈' (U+0308, CC 88) to avoid:
     *    - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 'w' (U+0077, 77) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
     *  - 'y' (U+0079, 79) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
     *
     *  Like @c sz_utf8_uncased_rune_safe_western_europe_k, we inherit one more Latin-1 limitation:
     *
     *  - 'å' (U+00E5, C3 A5) - is the folding target of both 'Å' (U+00C5, C3 85) in Latin-1 and the
     *    Angstrom Sign 'Å' (U+212B, E2 84 AB) → 'å' (U+00E5, C3 A5), so needle cannot contain 'å'
     *    (U+00E5, C3 A5) to avoid ambiguity.
     *
     *  This means, that all ASCII and Latin-1 characters beyond the rules above are "safe" for this
     *  profile, including English letters b, c, d, e, g, k, m, o, p, q, r, u, v, x, and z, as well
     *  as digits, punctuation, symbols, and control characters.
     *
     *  @sa sz_utf8_uncased_rune_ascii_invariant_k for a simpler variant.
     */
    sz_utf8_uncased_rune_safe_central_europe_k = 3,

    /**
     *  @brief Safety profile for contextually-safe ASCII + Basic Cyrillic designed mostly for East
     *      Slavic languages (like Russian, Ukrainian, & Belarusian) and South Slavic languages
     *      (like Serbian, Bulgarian, & Macedonian), but excluding Cyrillic Extensions.
     *
     *  Unlike the ASCII fast path, these kernels fold a wider range of characters:
     *  - 26x original ASCII uppercase letters: 'A' (U+0041, 41) → 'a' (U+0061, 61), 'Z' (U+005A,
     *    5A) → 'z' (U+007A, 7A)
     *  - 32x Basic Cyrillic uppercase letters:
     *    - 'А' (U+0410, D0 90) → 'а' (U+0430, D0 B0) through 'П' (U+041F, D0 9F) → 'п'
     *      (U+043F, D0 BF)
     *    - 'Р' (U+0420, D0 A0) → 'р' (U+0440, D1 80) through 'Я' (U+042F, D0 AF) → 'я'
     *      (U+044F, D1 8F)
     *  - 16x Cyrillic extensions for non-Russian Slavic languages:
     *    - 'Ѐ' (U+0400, D0 80) → 'ѐ' (U+0450, D1 90) - Cyrillic E with grave (Macedonian, Serbian)
     *    - 'Ё' (U+0401, D0 81) → 'ё' (U+0451, D1 91) - Cyrillic IO (Russian, Belarusian)
     *    - 'Ђ' (U+0402, D0 82) → 'ђ' (U+0452, D1 92) - Cyrillic DJE (Serbian)
     *    - 'Ѓ' (U+0403, D0 83) → 'ѓ' (U+0453, D1 93) - Cyrillic GJE (Macedonian)
     *    - 'Є' (U+0404, D0 84) → 'є' (U+0454, D1 94) - Cyrillic Ukrainian IE (Ukrainian)
     *    - 'Ѕ' (U+0405, D0 85) → 'ѕ' (U+0455, D1 95) - Cyrillic DZE (Macedonian)
     *    - 'І' (U+0406, D0 86) → 'і' (U+0456, D1 96) - Cyrillic Byelorussian-Ukrainian
     *      I (Ukrainian, Belarusian)
     *    - 'Ї' (U+0407, D0 87) → 'ї' (U+0457, D1 97) - Cyrillic YI (Ukrainian)
     *    - 'Ј' (U+0408, D0 88) → 'ј' (U+0458, D1 98) - Cyrillic JE (Serbian, Macedonian)
     *    - 'Љ' (U+0409, D0 89) → 'љ' (U+0459, D1 99) - Cyrillic LJE (Serbian, Macedonian)
     *    - 'Њ' (U+040A, D0 8A) → 'њ' (U+045A, D1 9A) - Cyrillic NJE (Serbian, Macedonian)
     *    - 'Ћ' (U+040B, D0 8B) → 'ћ' (U+045B, D1 9B) - Cyrillic TSHE (Serbian)
     *    - 'Ќ' (U+040C, D0 8C) → 'ќ' (U+045C, D1 9C) - Cyrillic KJE (Macedonian)
     *    - 'Ѝ' (U+040D, D0 8D) → 'ѝ' (U+045D, D1 9D) - Cyrillic I with
     *      grave (Bulgarian, Macedonian)
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
     *  We entirely ban all of the Extended Cyrillic (D2/D3 lead bytes), sometimes used in
     *  Ukranian, Kazakh, and Uzbek languages, like the 'Ґ' (U+0490, D2 90) → 'ґ' (U+0491, D2 91)
     *  folding with even/odd ordering of uppercase and lowercase. Similar rules apply to some
     *  Chechen, and various Turkic languages. But there are also exceptions, like the Palochka 'Ӏ'
     *  (U+04C0, D3 80) → 'ӏ' (U+04CF, D3 8F). By omitting those extensions we can make our folding
     *  kernel much lighter.
     *
     *  We inherit @b all contextual ASCII limitations from
     *  @c sz_utf8_uncased_rune_ascii_invariant_k:
     *
     *  - 'a' (U+0061, 61) - can't be last; can't precede 'ʾ' (U+02BE, CA BE) to avoid:
     *     - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
     *  - 'f' (U+0066, 66) - can't be first or last; can't follow 'f' (U+0066, 66); can't precede
     *    'f' (U+0066, 66), 'i' (U+0069, 69), 'l' (U+006C, 6C) to avoid:
     *     - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
     *     - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *     - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *     - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *     - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'h' (U+0068, 68) - can't be last; can't precede '̱' (U+0331, CC B1) to avoid:
     *     - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66); can't precede
     *    '̇' (U+0307, CC 87) to avoid:
     *     - 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
     *     - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *     - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *  - 'j' (U+006A, 6A) - can't be last; can't precede '̌' (U+030C, CC 8C) to avoid:
     *     - 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C)
     *  - 'k' (U+006B, 6B) - can't be present at all, as a folding target of the Kelvin sign:
     *     - 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B)
     *  - 'l' (U+006C, 6C) - can't be first; can't follow 'f' (U+0066, 66) to avoid:
     *     - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *     - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'n' (U+006E, 6E) - can't be first; can't follow 'ʼ' (U+02BC, CA BC) to avoid:
     *     - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
     *  - 's' (U+0073, 73) - can't be present at all, because it's a folding target of the
     *    old S sign:
     *    - 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73); can't precede
     *    '̈' (U+0308, CC 88) to avoid:
     *     - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *     - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *     - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73); can't precede
     *    '̈' (U+0308, CC 88) to avoid:
     *     - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *     - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *     - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 'w' (U+0077, 77) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *     - 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
     *  - 'y' (U+0079, 79) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *     - 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
     *
     *  These ASCII constraints are necessary because mixed-script documents (Cyrillic + Latin)
     *  may contain Latin ligatures, German Eszett, or Turkish İ that the Cyrillic fold
     *  function doesn't handle.
     *
     *  This means, that all ASCII characters beyond the rules above are considered "safe" for this
     *  profile, including English letters b, c, d, e, g, m, o, p, q, r, u, v, x, and z, as well as
     *  digits, punctuation, symbols, and control characters.
     *
     *  @sa sz_utf8_uncased_rune_ascii_invariant_k for the inherited ASCII rules.
     */
    sz_utf8_uncased_rune_safe_cyrillic_k = 4,

    /**
     *  @brief Safety profile for contextually-safe ASCII + Basic Greek designed mostly
     *      for Modern Greek (Demotic) text with a mixture of single-byte and double-byte
     *      UTF-8 character sequences.
     *
     *  Unlike the ASCII fast path, these kernels fold a wider range of characters:
     *  - 26x original ASCII uppercase letters: 'A' (U+0041, 41) → 'a' (U+0061, 61), 'Z' (U+005A,
     *    5A) → 'z' (U+007A, 7A)
     *  - 24x Basic Greek uppercase letters (monotonic, without diacritics):
     *    - 'Α' (U+0391, CE 91) → 'α' (U+03B1, CE B1) through 'Ο' (U+039F, CE 9F) → 'ο'
     *      (U+03BF, CE BF)
     *    - 'Π' (U+03A0, CE A0) → 'π' (U+03C0, CF 80) through 'Ω' (U+03A9, CE A9) → 'ω'
     *      (U+03C9, CF 89)
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
     *  - CF 80-89: Basic lowercase 'π'-'ω' (U+03C0-U+03C9), includes 'ς' (U+03C2, CF 82) and 'σ'
     *    (U+03C3, CF 83)
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
     *  - 'ΐ' (U+0390, CE 90), iota with dialytika and tonos, expands to 3 codepoints: "ΐ" (U+03B9
     *    U+0308 U+0301, CE B9 CC 88 CC 81).
     *  - 'ΰ' (U+03B0, CE B0), upsilon with dialytika and tonos, expands to 3 codepoints: "ΰ"
     *    (U+03C5 U+0308 U+0301, CF 85 CC 88 CC 81).
     *  - Greek Extended / Polytonic (U+1F00-U+1FFF, E1 BC-BF lead bytes): Ancient Greek with
     *    breathing marks, accents, and iota subscript, used mostly in academic, religious,
     *    and historical texts.
     *
     *  Many Polytonic letters expand to multiple codepoints, e.g. 'ᾈ' (U+1F88) → "ἀι" (U+1F00
     *  U+03B9, E1 BC 80 CE B9) and 'ᾳ' (U+1FB3) → "αι" (U+03B1 U+03B9, CE B1 CE B9).
     *
     *  The Latin-1 Micro Sign 'µ' (U+00B5, C2 B5) folds to Greek mu 'μ' (U+03BC, CE BC). This is
     *  handled by the Latin-1 kernel path, @c sz_utf8_uncased_rune_safe_western_europe_k, not the
     *  Greek path. The Greek kernel only handles characters that originate in the Greek block.
     *
     *  We inherit @b all contextual ASCII limitations from
     *  @c sz_utf8_uncased_rune_ascii_invariant_k:
     *
     *  - 'a' (U+0061, 61) - can't be last; can't precede 'ʾ' (U+02BE, CA BE) to avoid:
     *    - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
     *  - 'f' (U+0066, 66) - can't be first or last; can't follow 'f' (U+0066, 66); can't precede
     *    'f' (U+0066, 66), 'i' (U+0069, 69), 'l' (U+006C, 6C) to avoid:
     *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'h' (U+0068, 68) - can't be last; can't precede '̱' (U+0331, CC B1) to avoid:
     *    - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66); can't precede
     *    '̇' (U+0307, CC 87) to avoid:
     *    - 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *  - 'j' (U+006A, 6A) - can't be last; can't precede '̌' (U+030C, CC 8C) to avoid:
     *    - 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C)
     *  - 'k' (U+006B, 6B) - can't be present at all, as a folding target of the Kelvin sign:
     *    - 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B)
     *  - 'l' (U+006C, 6C) - can't be first; can't follow 'f' (U+0066, 66) to avoid:
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'n' (U+006E, 6E) - can't be first; can't follow 'ʼ' (U+02BC, CA BC) to avoid:
     *    - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
     *  - 's' (U+0073, 73) - can't be present at all, because it's a folding target of the
     *    old S sign:
     *    - 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73); can't precede
     *    '̈' (U+0308, CC 88) to avoid:
     *    - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73); can't precede
     *    '̈' (U+0308, CC 88) to avoid:
     *    - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 'w' (U+0077, 77) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẘ' (U+1E98, E1 BA 98) → "ẘ" (U+0077 U+030A, 77 CC 8A)
     *  - 'y' (U+0079, 79) - can't be last; can't precede '̊' (U+030A, CC 8A) to avoid:
     *    - 'ẙ' (U+1E99, E1 BA 99) → "ẙ" (U+0079 U+030A, 79 CC 8A)
     *
     *  These ASCII constraints are necessary because mixed-script documents (Greek + Latin) are
     *  common in scientific notation, brand names, and modern Greek text with English loanwords.
     *
     *  This means, that all ASCII characters beyond the rules above are considered "safe" for this
     *  profile, including English letters b, c, d, e, g, m, o, p, q, r, u, v, x, and z, as well as
     *  digits, punctuation, symbols, and control characters.
     *
     *  @sa sz_utf8_uncased_rune_ascii_invariant_k for the inherited ASCII rules.
     */
    sz_utf8_uncased_rune_safe_greek_k = 5,

    /**
     *  @brief Safety profile for contextually-safe ASCII + Basic Armenian.
     *
     *  These kernels fold:
     *  - 26x ASCII uppercase letters: 'A' (U+0041, 41) → 'a' (U+0061, 61), 'Z' (U+005A, 5A) →
     *    'z' (U+007A, 7A)
     *  - 38x Armenian uppercase letters:
     *    - 'Ա' (U+0531, D4 B1) → 'ա' (U+0561, D5 A1)
     *    - 'Ֆ' (U+0556, D5 96) → 'ֆ' (U+0586, D6 86)
     *
     *  UTF-8 byte ranges handled:
     *  - D4 B1-BF: uppercase 'Ա' (U+0531) through 'Ձ' (U+053F)
     *  - D5 80-96: uppercase 'Ղ' (U+0540) through 'Ֆ' (U+0556)
     *  - D5 A1-BF: lowercase 'ա' (U+0561) through 'ի' (U+057F)
     *  - D6 80-86: lowercase 'լ' (U+0580) through 'ֆ' (U+0586)
     *
     *  We inherit @b all contextual ASCII limitations from
     *  @c sz_utf8_uncased_rune_ascii_invariant_k:
     *
     *  - 'a' (U+0061, 61) - can't be last; can't precede 'ʾ' (U+02BE, CA BE) to avoid:
     *    - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
     *  - 'f' (U+0066, 66) - can't be first or last; can't follow 'f' (U+0066, 66); can't precede
     *    'f' (U+0066, 66), 'i' (U+0069, 69), 'l' (U+006C, 6C) to avoid:
     *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'h' (U+0068, 68) - can't be last; can't precede '̱' (U+0331, CC B1) to avoid:
     *    - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66); can't precede
     *    '̇' (U+0307, CC 87) to avoid:
     *    - 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *  - 'j' (U+006A, 6A) - can't be last; can't precede '̌' (U+030C, CC 8C) to avoid:
     *    - 'ǰ' (U+01F0, C7 B0) → "ǰ" (U+006A U+030C, 6A CC 8C)
     *  - 'k' (U+006B, 6B) - can't be present at all, as a folding target of the Kelvin sign:
     *    - 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B)
     *  - 'l' (U+006C, 6C) - can't be first; can't follow 'f' (U+0066, 66) to avoid:
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'n' (U+006E, 6E) - can't be first; can't follow 'ʼ' (U+02BC, CA BC) to avoid:
     *    - 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
     *  - 's' (U+0073, 73) - can't be present at all, because it's a folding target of the
     *    old S sign:
     *    - 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73); can't precede
     *    '̈' (U+0308, CC 88) to avoid:
     *    - 'ẗ' (U+1E97, E1 BA 97) → "ẗ" (U+0074 U+0308, 74 CC 88)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73); can't precede
     *    '̈' (U+0308, CC 88) to avoid:
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
     *  - 'ե' (U+0565, D5 A5) - can't be first; can't follow 'մ' (U+0574, D5 B4); can't precede 'ւ'
     *    (U+0582, D6 82) to avoid:
     *     - 'և' (U+0587, D6 87) → "եւ" (U+0565 U+0582, D5 A5 D6 82)
     *     - 'ﬔ' (U+FB14, EF AC 94) → "մե" (U+0574 U+0565, D5 B4 D5 A5)
     *  - 'ւ' (U+0582, D6 82) - can't be last; can't follow 'ե' (U+0565, D5 A5) to avoid:
     *     - 'և' (U+0587, D6 87) → "եւ" (U+0565 U+0582, D5 A5 D6 82)
     *  - 'մ' (U+0574, D5 B4) - can't be last; can't precede 'ն' (U+0576, D5 B6), 'ե' (U+0565, D5
     *    A5), 'ի' (U+056B, D5 AB), 'խ' (U+056D, D5 AD) to avoid:
     *     - 'ﬓ' (U+FB13, EF AC 93) → "մն" (U+0574 U+0576, D5 B4 D5 B6)
     *     - 'ﬔ' (U+FB14, EF AC 94) → "մե" (U+0574 U+0565, D5 B4 D5 A5)
     *     - 'ﬕ' (U+FB15, EF AC 95) → "մի" (U+0574 U+056B, D5 B4 D5 AB)
     *     - 'ﬗ' (U+FB17, EF AC 97) → "մխ" (U+0574 U+056D, D5 B4 D5 AD)
     *  - 'ն' (U+0576, D5 B6) - can't be first; can't follow 'մ' (U+0574, D5 B4), 'վ' (U+057E, D5
     *    BE) to avoid:
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
     *  cannot use the fast path because finding them separately might miss the precomposed
     *  ligatures present in the haystack.
     *
     *  @sa sz_utf8_uncased_rune_ascii_invariant_k for the inherited ASCII rules.
     */
    sz_utf8_uncased_rune_safe_armenian_k = 6,

    /**
     *  @brief Safety profile for contextually-safe ASCII + Latin-1 + Latin Extended Additional.
     *
     *  These kernels extend Latin-1/A/B with Vietnamese characters:
     *  - Everything from @c sz_utf8_uncased_rune_safe_central_europe_k (ASCII + Latin-1/A)
     *  - 166x Latin Extended Additional letters (U+1E00-U+1E95, U+1EA0-U+1EFF) for Vietnamese.
     *    Include precomposed Latin letters with additional diacritics (e.g. Ạ/ạ, Ả/ả, Ấ/ấ).
     *
     *  UTF-8 byte ranges handled:
     *  - 00-7F: ASCII, e.g. 'a' (U+0061, 61)
     *  - C2/C3: Latin-1 Supplement, e.g. 'â' (U+00E2, C3 A2)
     *  - C4-C5: Latin Extended-A, e.g. 'đ' (U+0111, C4 91)
     *  - C6: Latin Extended-B (for ơ, ư), e.g. 'ơ' (U+01A1, C6 A1)
     *  - E1 B8 80 - E1 BA 95: Latin Extended Additional (U+1E00-U+1E95), e.g. 'Ḁ' (U+1E00,
     *    E1 B8 80)
     *  - E1 BA A0 - E1 BB BF: Latin Extended Additional (U+1EA0-U+1EFF), e.g. 'ạ' (U+1EA1,
     *    E1 BA A1)
     *
     *  There is also a Unicode rule for folding the Kelvin 'K' (U+212A, E2 84 AA) into 'k' (U+006B,
     *  6B). That sign is extremely rare, while the lowercase 'k' is common in Vietnamese (e.g.
     *  "kem", "kéo"). So we add one more check for 'K' (U+212A, E2 84 AA) in the haystack, and if
     *  detected, again - revert to serial.
     *
     *  We inherit most contextual limitations for some of the ASCII characters from
     *  @c sz_utf8_uncased_rune_ascii_invariant_k:
     *
     *  - 'a' (U+0061, 61) - can't be last; can't precede 'ʾ' (U+02BE, CA BE) to avoid:
     *    - 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
     *  - 'f' (U+0066, 66) - can't be first or last; can't follow 'f' (U+0066, 66); can't precede
     *    'f' (U+0066, 66), 'i' (U+0069, 69), 'l' (U+006C, 6C) to avoid:
     *    - 'ﬀ' (U+FB00, EF AC 80) → "ff" (U+0066 U+0066, 66 66)
     *    - 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69)
     *    - 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C)
     *    - 'ﬃ' (U+FB03, EF AC 83) → "ffi" (U+0066 U+0066 U+0069, 66 66 69)
     *    - 'ﬄ' (U+FB04, EF AC 84) → "ffl" (U+0066 U+0066 U+006C, 66 66 6C)
     *  - 'h' (U+0068, 68) - can't be last; can't precede '̱' (U+0331, CC B1) to avoid:
     *    - 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1)
     *  - 'i' (U+0069, 69) - can't be first or last; can't follow 'f' (U+0066, 66); can't precede
     *    '̇' (U+0307, CC 87) to avoid:
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
     *  - 's' (U+0073, 73) - can't be first or last; can't follow 's' (U+0073, 73); can't precede
     *    's' (U+0073, 73), 't' (U+0074, 74) to avoid:
     *    - 'ß' (U+00DF, C3 9F) → "ss" (U+0073 U+0073, 73 73)
     *    - 'ẞ' (U+1E9E, E1 BA 9E) → "ss" (U+0073 U+0073, 73 73)
     *    - 'ﬅ' (U+FB05, EF AC 85) → "st" (U+0073 U+0074, 73 74)
     *    - 'ﬆ' (U+FB06, EF AC 86) → "st" (U+0073 U+0074, 73 74)
     *    - 'ẛ' (U+1E9B, E1 BA 9B) → 'ṡ' (U+1E61, E1 B9 A1) [Latin Extended Additional]
     *  - 't' (U+0074, 74) - can't be first or last; can't follow 's' (U+0073, 73); can't precede
     *    '̈' (U+0308, CC 88) to avoid:
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
     *  - 'å' (U+00E5, C3 A5) - is the folding target of both 'Å' (U+00C5, C3 85) in Latin-1 and the
     *    Angstrom Sign 'Å' (U+212B, E2 84 AB) → 'å' (U+00E5, C3 A5), so needle cannot contain 'å'
     *    (U+00E5, C3 A5) to avoid ambiguity.
     *
     *  This means, that all other ASCII, Latin-1, Latin-A, and Latin Extended Additional characters
     *  are "safe" to use with this kernel.
     *
     *  @sa sz_utf8_uncased_rune_safe_central_europe_k for the inherited Latin rules.
     */
    sz_utf8_uncased_rune_safe_vietnamese_k = 7,

    /**
     *  @brief Safety profile for Georgian Mkhedruli script.
     *
     *  Georgian Mkhedruli (U+10D0-U+10FF) is caseless - no folding needed for Georgian
     *  characters. Only ASCII A-Z folding for mixed text. Mtavruli (U+1C90-U+1CBF),
     *  Asomtavruli (U+10A0-U+10C5), and Nuskhuri (U+2D00-U+2D25) trigger alarm for serial
     *  fallback (rare in modern text).
     *
     *  All Georgian scripts use 3-byte UTF-8 sequences and fold to 3-byte sequences, so there are
     *  no length changes during case folding - making this the simplest non-ASCII kernel.
     *
     *  @sa sz_utf8_uncased_rune_ascii_invariant_k for inherited ASCII rules.
     */
    sz_utf8_uncased_rune_safe_georgian_k = 8,

    /** The needle has no cased codepoints, so an exact substring search finds every match. */
    sz_utf8_uncased_rune_invariant_k = 9,

    /** No SIMD profile fits the needle, so the search falls back to the serial kernel. */
    sz_utf8_uncased_rune_fallback_serial_k = 255,
} sz_utf8_uncased_rune_safety_profile_t;

#pragma region Case Invariance and Ordering

/**
 *  @brief Internal helper: checks if a single Unicode codepoint is case-agnostic.
 *
 *  A codepoint is case-agnostic if all of the following are true:
 *
 *  1. It folds to exactly itself (no transformation, no expansion)
 *  2. It does not belong to any bicameral (cased) script
 *  3. It does not appear in any case fold expansion as a target character
 *
 *  The third condition is critical. Consider 'ʾ' (U+02BE, CA BE):
 *
 *  - It has no case variant and folds to itself
 *  - However, 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE)
 *  - A needle containing 'ʾ' must match at position 1 of the folded expansion of 'ẚ'
 *  - Binary search cannot handle this - it only sees 'ẚ' as a 3-byte sequence (E1 BA 9A)
 *  - Therefore 'ʾ' must not be treated as case-agnostic
 *
 *  This function implements the check via explicit range exclusions for all bicameral scripts and
 *  all Unicode blocks containing case fold expansion target characters.
 *
 *  @param[in] rune Unicode codepoint to check.
 *  @return @c sz_true_k if the codepoint is case-agnostic, @c sz_false_k otherwise.
 *
 *  @warning This is an internal function. Use @ref sz_utf8_find_cased_serial for string checking.
 *  @sa sz_utf8_find_cased_serial
 *  @sa sz_unicode_fold_codepoint_
 */
STRINGZILLA_CONSTEXPR sz_bool_t sz_rune_is_uncased_(sz_rune_t rune) {

    // Check if this rune participates in case folding
    sz_rune_t folded_runes[3];
    sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);

    // If it expands or changes, it's not caseless
    if (folded_count != 1 || folded_runes[0] != rune) return sz_false_k;

    // Check if this rune is a lowercase target of some uppercase letter.
    // Lowercase letters that don't change when folded still participate in case
    // because uppercase versions fold TO them. We must mark entire bicameral
    // script ranges as "not caseless" to enable proper uncased matching.
    //
    // Important: Combining diacritical marks (U+0300-U+036F) can appear as non-first runes in
    // multi-rune case fold expansions. Example: ǰ (U+01F0) → j + ̌ (U+030C). A needle starting with
    // a combining caron could match inside such an expansion, so combining marks must not be
    // treated as case-agnostic.
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
    if (rune == 0x1D79 || rune == 0x1D7D || rune == 0x1D8E)
        return sz_false_k; // Phonetic Extensions ᵹ ᵽ ᶎ (fold targets of Ᵹ U+A77D, Ᵽ U+2C63, Ᶎ U+A7C6)
    if (rune >= 0x1E00 && rune <= 0x1EFF) return sz_false_k; // Latin Extended Additional
    if (rune >= 0x1F00 && rune <= 0x1FFF) return sz_false_k; // Greek Extended
    if (rune == 0x214E) return sz_false_k;                   // ⅎ (fold target of Ⅎ U+2132)
    if (rune >= 0x2170 && rune <= 0x217F) return sz_false_k; // small Roman numerals (fold targets of U+2160-216F)
    if (rune == 0x2184) return sz_false_k;                   // ↄ (fold target of Ↄ U+2183)
    if (rune >= 0x24D0 && rune <= 0x24E9) return sz_false_k; // circled small Latin (fold targets of U+24B6-24CF)
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
    if (rune >= 0x10D70 && rune <= 0x10D85) return sz_false_k; // Garay small letters (fold targets of U+10D50-10D65)
    if (rune >= 0x118A0 && rune <= 0x118FF) return sz_false_k; // Warang Citi
    if (rune >= 0x16E40 && rune <= 0x16E9F) return sz_false_k; // Medefaidrin
    if (rune >= 0x16EBB && rune <= 0x16ED3) return sz_false_k; // Beria Erfe small letters (Unicode 17 fold targets)
    if (rune >= 0x1DF00 && rune <= 0x1DFFF) return sz_false_k; // Latin Extended-G
    if (rune >= 0x1E000 && rune <= 0x1E02F) return sz_false_k; // Glagolitic Supplement
    if (rune >= 0x1E030 && rune <= 0x1E08F) return sz_false_k; // Cyrillic Extended-D
    if (rune >= 0x1E900 && rune <= 0x1E95F) return sz_false_k; // Adlam

    return sz_true_k;
}

STRINGZILLA_OUTLINED_ sz_cptr_t sz_utf8_find_cased_serial_(sz_cptr_t str, sz_size_t length) {
    sz_u8_t const *text_cursor = (sz_u8_t const *)str;
    sz_u8_t const *text_end = text_cursor + length;

    while (text_cursor < text_end) {
        sz_u8_t lead = *text_cursor;

        // ASCII fast path: only digits, punctuation, and control chars are caseless
        // A-Z (0x41-0x5A) and a-z (0x61-0x7A) participate in case folding
        if (lead < 0x80) {
            if ((lead >= 'A' && lead <= 'Z') || (lead >= 'a' && lead <= 'z')) return (sz_cptr_t)text_cursor;
            text_cursor++;
            continue;
        }

        // Multi-byte: decode and check. A byte that does not begin a well-formed codepoint is its own
        // 1-byte maximal subpart - it folds to itself, so it is caseless and never a violation; resync by one byte.
        sz_rune_t rune;
        sz_rune_length_t const rune_length = sz_rune_decode((sz_cptr_t)text_cursor, (sz_cptr_t)text_end, &rune);
        if (rune_length == sz_rune_invalid_k) {
            text_cursor++;
            continue;
        }
        if (sz_rune_is_uncased_(rune) == sz_false_k) return (sz_cptr_t)text_cursor;
        text_cursor += rune_length;
    }

    return STRINGZILLA_NULL_CHAR;
}

STRINGZILLA_INLINE sz_ordering_t sz_utf8_uncased_order_serial_(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                               sz_size_t b_length) {
    sz_utf8_folded_iter_t a_iterator, b_iterator;
    sz_utf8_folded_iter_init_(&a_iterator, a, a_length);
    sz_utf8_folded_iter_init_(&b_iterator, b, b_length);

    sz_rune_t a_rune = 0,
              b_rune = 0; // Initialized to satisfy GCC's -Wmaybe-uninitialized; the iterators always set them.
    for (;;) {
        sz_bool_t pulled_from_a = sz_utf8_folded_iter_next_(&a_iterator, &a_rune);
        sz_bool_t pulled_from_b = sz_utf8_folded_iter_next_(&b_iterator, &b_rune);

        if (!pulled_from_a && !pulled_from_b) return sz_equal_k;
        if (!pulled_from_a) return sz_less_k;
        if (!pulled_from_b) return sz_greater_k;
        if (a_rune != b_rune) return sz_order_scalars_(a_rune, b_rune);
    }
}

#pragma endregion Case Invariance and Ordering

/** Pops the lowest candidate position from @p matches, returning its bit index: the scalar walk
 *  shared by every ISA probe filter, so vector kernels never materialize their own bit scans. */
STRINGZILLA_INLINE sz_size_t sz_utf8_uncased_pop_candidate_(sz_u64_t *matches) {
    sz_size_t const position = (sz_size_t)sz_u64_ctz(*matches);
    *matches &= *matches - 1;
    return position;
}

/** Per-codepoint Latin Extended-A fold deltas after a C4/C5 lead, indexed by the continuation
 *  byte's low 6 bits, `text & 0x3F`. Entry value is the in-place add: 0 = identity, 1 = fold by +1.
 *  The cross-block irregulars that the case-fold tables flag with 0x80 ('İ' C4 B0, 'Ŀ' C4 BF, 'Ÿ'
 *  C5 B8, 'ſ' C5 BF) are 0 here because the alarm routes them to the danger-zone handler, so the
 *  fold leaves them untouched. The parity matches explicit range checks but resolves in one
 *  @c vqtbl4q_u8 per lead family, verified against the serial reference in tests. */
static sz_u8_t const sz_utf8_uncased_central_c4_deltas_lut_[64] = {
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, // C4 80-8F: 'Ā'-'ď' even-parity pairs
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, // C4 90-9F
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, // C4 A0-AF
    0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, // C4 B0-BF: 'İ'/'ĸ'/'Ŀ' caseless or cross-block
};
static sz_u8_t const sz_utf8_uncased_central_c5_deltas_lut_[64] = {
    0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, // C5 80-8F: odd head, 'ŉ' (C5 89) irregular → 0
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, // C5 90-9F
    1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, // C5 A0-AF
    1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, // C5 B0-BF: 'Ÿ'/'ſ' cross-block → 0
};

/**
 *  @brief Monotonic-Greek second-byte fold deltas after a CE lead, indexed by `text & 0x3F`.
 *
 *  The per-rule range-check deltas, resolved in one @c vqtbl4q_u8: 'Ά' (86) +0x26, 'Έ'-'Ί' (88-8A)
 *  +0x25, 'Ύ'/'Ώ' (8E-8F) −1, 'Α'-'Ο' (91-9F) +0x20, 'Π'-'Ω' (A0-A9) and 'Ϊ'/'Ϋ' (AA-AB) −0x20. 'Ό'
 *  (8C) keeps its byte, as only its lead changes. The window at offset 64 flags the leads that
 *  promote from CE to CF, for the classes whose lowercase lands in CF.
 */
static sz_u8_t const sz_utf8_uncased_greek_ce_lut_[128] = {
    0,    0,    0,    0,    0,    0,    0x26, 0,
    0x25, 0x25, 0x25, 0,    0,    0,    0xFF, 0xFF, // CE 80-8F
    0,    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, // CE 90-9F
    0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0, 0xE0,
    0xE0, 0xE0, 0xE0, 0xE0, 0,    0,    0,    0, // CE A0-AF
    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0, // CE B0-BF
    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    1,    0,    1,    1, // CE 80-8F: 8C, 8E, 8F
    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0, // CE 90-9F: stay under CE
    1,    1,    1,    1,    1,    1,    1,    1,
    1,    1,    1,    1,    0,    0,    0,    0, // CE A0-AB: promote
    0,    0,    0,    0,    0,    0,    0,    0,
    0,    0,    0,    0,    0,    0,    0,    0, // CE B0-BF
};
static sz_u8_t const *const sz_utf8_uncased_greek_ce_deltas_lut_ = sz_utf8_uncased_greek_ce_lut_ + 0;
static sz_u8_t const *const sz_utf8_uncased_greek_ce_promotes_lut_ = sz_utf8_uncased_greek_ce_lut_ + 64;

#pragma region Substring Search

/**
 *  @brief Verify head region uncasedly, iterating backward.
 *
 *  Walks backward from @p needle_end and @p haystack_end, comparing folded runes. Returns true if
 *  the needle region exhausts (matched), with haystack bytes consumed.
 *
 *  @param[in] needle_start Start of needle head region.
 *  @param[in] needle_end End of needle head region, where the safe window begins.
 *  @param[in] haystack_start Start of haystack, the lower bound for the backward scan.
 *  @param[in] haystack_end End of haystack head region, where the safe window was found.
 *  @param[out] match_length Haystack bytes consumed by this match.
 */
STRINGZILLA_CONSTEXPR sz_bool_t sz_utf8_uncased_verify_head_(sz_cptr_t needle_start, sz_cptr_t needle_end,
                                                             sz_cptr_t haystack_start, sz_cptr_t haystack_end,
                                                             sz_size_t *match_length) {

    // If needle head is empty, no haystack bytes needed
    if (needle_end <= needle_start) {
        *match_length = 0;
        return sz_true_k;
    }

    sz_utf8_folded_reverse_iter_t needle_riter, haystack_riter;
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
 *  @brief Verify tail region uncasedly, iterating forward.
 *
 *  Walks forward, comparing folded runes. Returns true if the needle exhausts.
 *
 *  @param[in] needle_start Start of needle tail region.
 *  @param[in] needle_end End of needle tail region, equal to `needle + needle_length`.
 *  @param[in] haystack_start Start of haystack tail region.
 *  @param[in] haystack_end End of haystack, the upper bound for the forward scan.
 *  @param[out] match_length Haystack bytes consumed by this match.
 */
STRINGZILLA_CONSTEXPR sz_bool_t sz_utf8_uncased_verify_tail_(sz_cptr_t needle_start, sz_cptr_t needle_end,
                                                             sz_cptr_t haystack_start, sz_cptr_t haystack_end,
                                                             sz_size_t *match_length) {

    sz_size_t needle_length = (sz_size_t)(needle_end - needle_start);

    // Empty tail is trivially matched
    if (needle_length == 0) {
        *match_length = 0;
        return sz_true_k;
    }

    sz_utf8_folded_iter_t needle_iter, haystack_iter;
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
 *  Verifies two regions: "head" before the window and "tail" after it. The middle part may still be
 *  partly unprocessed if it is larger than the "folded slice" of the needle; it is handled as part
 *  of the "tail", and @p needle_tail_bytes must be calculated accordingly.
 *
 *  @param[in] haystack Haystack start pointer, arbitrary case.
 *  @param[in] haystack_length Haystack length in bytes.
 *  @param[in] needle Needle start pointer, arbitrary case.
 *  @param[in] needle_length Needle length in bytes.
 *  @param[in] haystack_matched_offset Start offset of matched safe window in haystack in bytes.
 *  @param[in] haystack_matched_length Length of matched safe window in haystack in bytes.
 *  @param[in] needle_head_bytes Start of matched safe window in needle in bytes.
 *  @param[in] needle_tail_bytes Number of bytes in the needle remaining after the matched part.
 *  @param[out] match_length Total length of the verified match in haystack bytes.
 *  @return Match start pointer, or @c STRINGZILLA_NULL_CHAR if validation fails.
 */
STRINGZILLA_CONSTEXPR sz_cptr_t sz_utf8_uncased_verify_match_(            //
    sz_cptr_t haystack, sz_size_t haystack_length,                        //
    sz_cptr_t needle, sz_size_t needle_length,                            //
    sz_size_t haystack_matched_offset, sz_size_t haystack_matched_length, //
    sz_size_t needle_head_bytes, sz_size_t needle_tail_bytes,             //
    sz_size_t *match_length) {

    sz_cptr_t needle_end = needle + needle_length;
    sz_cptr_t haystack_end = haystack + haystack_length;

    // Verify head using backward iterators
    sz_size_t head_match_length = 0;
    if (needle_head_bytes)
        if (!sz_utf8_uncased_verify_head_(                    //
                needle, needle + needle_head_bytes,           // needle head region
                haystack, haystack + haystack_matched_offset, // haystack head region
                &head_match_length))
            return STRINGZILLA_NULL_CHAR;

    // Verify tail using forward iterators
    sz_size_t tail_match_length = 0;
    sz_cptr_t haystack_tail_start = haystack + haystack_matched_offset + haystack_matched_length;
    if (needle_tail_bytes)
        if (!sz_utf8_uncased_verify_tail_(                              //
                needle + needle_length - needle_tail_bytes, needle_end, // needle tail region
                haystack_tail_start, haystack_end,                      // haystack tail region
                &tail_match_length))
            return STRINGZILLA_NULL_CHAR;

    *match_length = head_match_length + haystack_matched_length + tail_match_length;
    return haystack + haystack_matched_offset - head_match_length;
}

/**
 *  @brief Hash-free uncased search for needles that fold to exactly 1 rune.
 *
 *  Examples: 'a', 'A', 'б', 'Б', but not 'ß' (U+00DF, C3 9F) → "ss", which is 2 runes.
 *
 *  Single-pass algorithm: parses each source rune, folds it, checks if it produces exactly one rune
 *  matching the target. No iterator overhead, no verification needed.
 *
 *  @param[in] haystack Pointer to the haystack string to search within.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] needle_folded The single folded rune to search for.
 *  @param[out] match_length Length of the matched rune in haystack bytes on success.
 *  @return Pointer to the first matching rune, or @c STRINGZILLA_NULL_CHAR if not found.
 */
STRINGZILLA_CONSTEXPR sz_cptr_t sz_utf8_uncased_search_1folded_serial_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                      //
    sz_rune_t needle_folded, sz_size_t *match_length) {

    sz_cptr_t const haystack_end = haystack + haystack_length;

    // Each haystack rune may fold in up to 3 runes
    sz_rune_t haystack_rune;
    sz_rune_length_t haystack_rune_length;

    // If we simply initialize the runes for zero, the code will break
    // when the needle itself is the NUL character
    sz_rune_t haystack_folded_runes[3] = {~needle_folded};
    while (haystack < haystack_end) {
        // A byte that does not begin a well-formed codepoint folds to itself and matches byte-for-byte;
        // resync by one byte. Fill the unused fold slots with sentinels so they never false-match.
        haystack_rune_length = sz_rune_decode(haystack, haystack_end, &haystack_rune);
        if (haystack_rune_length == sz_rune_invalid_k) {
            haystack_folded_runes[0] = sz_rune_malformed_byte_((sz_u8_t)*haystack);
            haystack_folded_runes[1] = ~needle_folded;
            haystack_folded_runes[2] = ~needle_folded;
            haystack_rune_length = sz_rune_1byte_k;
        }
        else { sz_unicode_fold_codepoint_(haystack_rune, haystack_folded_runes); }

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
    return STRINGZILLA_NULL_CHAR;
}

/**
 *  @brief Verifies the needle anchored at folded rune @p anchor_index of a danger-zone codepoint.
 *
 *  The codepoint starts at @p danger_cursor. Anchoring on rune zero starts on a codepoint boundary,
 *  which the shared validator already handles. Past that, the runes of this one codepoint on either
 *  side of the anchor never reach a folded iterator, as the iterators step over the codepoint
 *  whole, so they are compared against the image directly.
 *
 *  @param[in] haystack_folded_runes The codepoint's folded image; @p anchor_index selects the rune
 *      to anchor on.
 *  @return Match start, or @c STRINGZILLA_NULL_CHAR when this anchor carries no match.
 */
STRINGZILLA_CONSTEXPR sz_cptr_t sz_utf8_uncased_verify_at_folded_rune_(            //
    sz_cptr_t haystack, sz_size_t haystack_length,                                 //
    sz_cptr_t needle, sz_size_t needle_length,                                     //
    sz_cptr_t danger_cursor, sz_size_t haystack_rune_length,                       //
    sz_rune_t const *haystack_folded_runes, sz_size_t haystack_folded_runes_count, //
    sz_size_t anchor_index,                                                        //
    sz_rune_t needle_first_safe_folded_rune,                                       //
    sz_size_t needle_first_safe_folded_rune_offset,                                //
    sz_size_t *match_length) {

    sz_cptr_t const haystack_end = haystack + haystack_length;

    if (anchor_index == 0)
        return sz_utf8_uncased_verify_match_(                     //
            haystack, haystack_length,                            //
            needle, needle_length,                                //
            (sz_size_t)(danger_cursor - haystack), 0,             // No pre-matched middle
            needle_first_safe_folded_rune_offset,                 //
            needle_length - needle_first_safe_folded_rune_offset, // Verify everything after head serially
            match_length);

    sz_cptr_t haystack_match_start = 0, haystack_match_end = 0;

    // Walk the needle head backwards against the haystack before the danger zone began.
    sz_rune_t needle_riter_rune = 0, haystack_riter_rune = 0;
    sz_utf8_folded_reverse_iter_t needle_riter, haystack_riter;
    sz_utf8_folded_reverse_iter_init_(&needle_riter, needle, needle + needle_first_safe_folded_rune_offset);
    sz_utf8_folded_reverse_iter_init_(&haystack_riter, haystack, danger_cursor);

    // This codepoint's own runes before the anchor, newest first.
    for (sz_size_t before = anchor_index; before-- > 0;) {
        if (!sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_riter_rune)) break;
        if (needle_riter_rune != haystack_folded_runes[before]) return STRINGZILLA_NULL_CHAR;
    }

    for (;;) {
        // Needle exhausted - success!
        if (!sz_utf8_folded_reverse_iter_prev_(&needle_riter, &needle_riter_rune)) {
            haystack_match_start = haystack_riter.ptr;
            break;
        }
        if (!sz_utf8_folded_reverse_iter_prev_(&haystack_riter, &haystack_riter_rune)) return STRINGZILLA_NULL_CHAR;
        if (needle_riter_rune != haystack_riter_rune) return STRINGZILLA_NULL_CHAR;
    }

    // Walk the needle tail forwards from the safe window's start.
    sz_rune_t needle_iter_rune = 0, haystack_iter_rune = 0;
    sz_utf8_folded_iter_t needle_iter, haystack_iter;
    sz_utf8_folded_iter_init_(&needle_iter, needle + needle_first_safe_folded_rune_offset,
                              needle_length - needle_first_safe_folded_rune_offset);
    sz_utf8_folded_iter_init_(&haystack_iter, danger_cursor + haystack_rune_length,
                              (sz_size_t)(haystack_end - (danger_cursor + haystack_rune_length)));

    // Pop the `needle_first_safe_folded_rune` from the forward iterator
    {
        sz_bool_t have_needle = sz_utf8_folded_iter_next_(&needle_iter, &needle_iter_rune);
        sz_assert_(have_needle && needle_iter_rune == needle_first_safe_folded_rune);
        sz_unused_(have_needle);
    }

    // This codepoint's own runes after the anchor, oldest first.
    for (sz_size_t after = anchor_index + 1; after < haystack_folded_runes_count; ++after) {
        if (!sz_utf8_folded_iter_next_(&needle_iter, &needle_iter_rune)) break;
        if (needle_iter_rune != haystack_folded_runes[after]) return STRINGZILLA_NULL_CHAR;
    }

    for (;;) {
        // Needle exhausted - success!
        if (!sz_utf8_folded_iter_next_(&needle_iter, &needle_iter_rune)) {
            haystack_match_end = haystack_iter.ptr;
            break;
        }
        if (!sz_utf8_folded_iter_next_(&haystack_iter, &haystack_iter_rune)) return STRINGZILLA_NULL_CHAR;
        if (needle_iter_rune != haystack_iter_rune) return STRINGZILLA_NULL_CHAR;
    }

    if (haystack_match_start == 0 || haystack_match_end == 0) return STRINGZILLA_NULL_CHAR;
    *match_length = (sz_size_t)(haystack_match_end - haystack_match_start);
    return haystack_match_start;
}

/**
 *  @brief Search a "danger zone" region using 1-folded candidate search and validation.
 *
 *  When SIMD kernels detect potentially problematic bytes (ligatures, Greek Extended, etc.), they
 *  fall back to this serial search within the affected chunk. This function:
 *
 *  1. Extracts the first folded rune from the needle's safe window
 *  2. Searches for candidates matching that rune
 *  3. Validates each candidate using the full verification pipeline
 *
 *  @param[in] haystack Full haystack string, arbitrary case.
 *  @param[in] haystack_length Full haystack length in bytes.
 *  @param[in] needle Full needle string, arbitrary case.
 *  @param[in] needle_length Full needle length.
 *  @param[in] danger_cursor Start of the danger zone region to search.
 *  @param[in] danger_length Length of the danger zone region in bytes.
 *  @param[in] needle_first_safe_folded_rune The first rune of the safe window, folded.
 *  @param[in] needle_first_safe_folded_rune_offset Offset of the safe window within the needle.
 *  @param[out] match_length Haystack bytes consumed by the match.
 *  @return Pointer to match start, or @c STRINGZILLA_NULL_CHAR if not found in this region.
 */
STRINGZILLA_CONSTEXPR sz_cptr_t sz_utf8_uncased_search_in_danger_zone_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                      //
    sz_cptr_t needle, sz_size_t needle_length,                          //
    sz_cptr_t danger_cursor, sz_size_t danger_length,                   //
    sz_rune_t needle_first_safe_folded_rune,                            //
    sz_size_t needle_first_safe_folded_rune_offset,                     //
    sz_size_t *match_length) {

    sz_cptr_t const haystack_end = haystack + haystack_length;
    sz_cptr_t const danger_end = sz_min_of_two(danger_cursor + danger_length, haystack_end);
    while (danger_cursor < danger_end) {

        // Skip continuation bytes - they are mid-sequence, not valid rune starts.
        // Without this check, a continuation byte like 0xBA could be misinterpreted as U+00BA (º),
        // causing false matches when the danger zone starts mid-character.
        sz_u8_t lead_byte = *(sz_u8_t const *)danger_cursor;
        if ((lead_byte & 0xC0) == 0x80) {
            danger_cursor++;
            continue;
        }

        // The following part is practically the unpacked variant of `sz_utf8_uncased_search_1folded_serial_`,
        // that finds the first occurrence of the `needle_first_safe_folded_rune` haystack. The issue is that each one
        // `haystack_rune` may unpack into multiple `haystack_folded_runes`.
        sz_rune_t haystack_rune;
        sz_rune_length_t haystack_rune_length;
        sz_rune_t haystack_folded_runes[3] = {~needle_first_safe_folded_rune};
        // A byte that does not begin a well-formed codepoint folds to itself and resyncs by one byte.
        haystack_rune_length = sz_rune_decode(danger_cursor, haystack_end, &haystack_rune);
        sz_size_t haystack_folded_runes_count;
        if (haystack_rune_length == sz_rune_invalid_k) {
            haystack_folded_runes[0] = sz_rune_malformed_byte_((sz_u8_t)*danger_cursor);
            haystack_folded_runes_count = 1;
            haystack_rune_length = sz_rune_1byte_k;
        }
        else { haystack_folded_runes_count = sz_unicode_fold_codepoint_(haystack_rune, haystack_folded_runes); }

        // The needle's anchor may sit at any rune of this codepoint's folded image, and a candidate that
        // fails only rules out that rune - the next one is still open.
        for (sz_size_t anchor_index = 0; anchor_index < haystack_folded_runes_count; ++anchor_index) {
            if (haystack_folded_runes[anchor_index] != needle_first_safe_folded_rune) continue;
            sz_cptr_t const match = sz_utf8_uncased_verify_at_folded_rune_( //
                haystack, haystack_length,                                  //
                needle, needle_length,                                      //
                danger_cursor, (sz_size_t)haystack_rune_length,             //
                haystack_folded_runes, haystack_folded_runes_count,         //
                anchor_index,                                               //
                needle_first_safe_folded_rune,                              //
                needle_first_safe_folded_rune_offset,                       //
                match_length);
            if (match) return match;
        }

        // Move to next candidate
        danger_cursor += haystack_rune_length;
    }

    return STRINGZILLA_NULL_CHAR;
}

/**
 *  @brief Hash-free uncased search for needles that fold to exactly 2 runes.
 *
 *  Examples: 'ab', 'AB', 'ß' (U+00DF) → "ss", 'ﬁ' (U+FB01) → "fi".
 *
 *  Single-pass sliding window over the folded rune stream. Handles expansions by buffering folded
 *  runes from each source and tracking source boundaries.
 *
 *  @param[in] haystack Pointer to the haystack string to search within.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] first_needle_folded First folded rune of the 2-rune needle.
 *  @param[in] second_needle_folded Second folded rune of the 2-rune needle.
 *  @param[out] match_length Length of the matched region in haystack bytes on success.
 *  @return Pointer to the first match, or @c STRINGZILLA_NULL_CHAR if not found.
 */
STRINGZILLA_CONSTEXPR sz_cptr_t sz_utf8_uncased_search_2folded_serial_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                      //
    sz_rune_t first_needle_folded, sz_rune_t second_needle_folded, sz_size_t *match_length) {

    sz_cptr_t const haystack_end = haystack + haystack_length;

    sz_rune_t haystack_rune;
    sz_rune_length_t haystack_rune_length;

    // Source-codepoint begin pointer for the single history slot (slot [0]). It is never read
    // until at least one codepoint has been processed, because the sentinel `~first_needle_folded`
    // in slot [0] can never equal `first_needle_folded`, so `match_at_01` is 0 on the first step.
    sz_cptr_t first_history_source_begin = haystack;

    // If we simply initialize the runes for zero, the code will break
    // when the needle itself is the NUL character
    sz_rune_t haystack_folded_runes[4] = {~first_needle_folded};
    while (haystack < haystack_end) {
        // A byte that does not begin a well-formed codepoint folds to itself and resyncs by one byte.
        haystack_rune_length = sz_rune_decode(haystack, haystack_end, &haystack_rune);

        // Pre-fill positions [2] and [3] with sentinels before folding.
        // The fold will overwrite positions it uses; unused positions keep the sentinel.
        // This branchlessly prevents stale data from causing false matches.
        sz_rune_t sentinel = ~second_needle_folded;
        haystack_folded_runes[2] = sentinel;
        haystack_folded_runes[3] = sentinel;
        // Export into the last 3 rune entries of the 4-element array,
        // keeping the first position with historical data untouched
        sz_size_t folded_count;
        if (haystack_rune_length == sz_rune_invalid_k) {
            haystack_folded_runes[1] = sz_rune_malformed_byte_((sz_u8_t)*haystack);
            folded_count = 1;
            haystack_rune_length = sz_rune_1byte_k;
        }
        else { folded_count = sz_unicode_fold_codepoint_(haystack_rune, haystack_folded_runes + 1); }

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
            // The first matched rune is in history slot [0] only for `match_at_01`; for `match_at_12`
            // and `match_at_23` it is in the current codepoint. The last matched rune is always within
            // the current codepoint, so the match always ends at the current codepoint's end.
            sz_cptr_t match_begin = match_at_01 ? first_history_source_begin : haystack;
            sz_cptr_t match_end = haystack + haystack_rune_length;
            *match_length = (sz_size_t)(match_end - match_begin);
            return match_begin;
        }

        // The history slot is always fed by the codepoint just processed.
        haystack_folded_runes[0] = haystack_folded_runes[folded_count];
        first_history_source_begin = haystack;
        haystack += haystack_rune_length;
    }

    *match_length = 0;
    return STRINGZILLA_NULL_CHAR;
}

/**
 *  @brief Hash-free uncased search for needles that fold to exactly 3 runes.
 *
 *  Examples: 'abc', 'ABC', "aß" → "ass", "ﬁa" (U+FB01) → "fia".
 *
 *  Single-pass sliding window of 3 folded runes over the haystack's folded stream. Handles
 *  expansions by buffering folded runes and tracking source boundaries.
 *
 *  @param[in] haystack Pointer to the haystack string to search within.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] first_needle_folded First folded rune of the 3-rune needle.
 *  @param[in] second_needle_folded Second folded rune of the 3-rune needle.
 *  @param[in] third_needle_folded Third folded rune of the 3-rune needle.
 *  @param[out] match_length Length of the matched region in haystack bytes on success.
 *  @return Pointer to the first match, or @c STRINGZILLA_NULL_CHAR if not found.
 */
STRINGZILLA_CONSTEXPR sz_cptr_t sz_utf8_uncased_search_3folded_serial_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                      //
    sz_rune_t first_needle_folded, sz_rune_t second_needle_folded, sz_rune_t third_needle_folded,
    sz_size_t *match_length) {

    sz_cptr_t const haystack_end = haystack + haystack_length;

    sz_rune_t haystack_rune;
    sz_rune_length_t haystack_rune_length;

    // Source-codepoint begin pointers for the two history slots ([0] and [1]). Never read until the
    // corresponding slot holds a real (non-sentinel) rune, because the sentinels in slots [0],[1] can
    // never match their needle runes.
    sz_cptr_t history_source_begin[2] = {haystack, haystack};

    // Initialize historical slots with sentinels that can never match their respective needle positions
    // This prevents false matches on first iterations when history is not yet populated
    sz_rune_t haystack_folded_runes[5] = {~first_needle_folded, ~second_needle_folded, 0, 0, 0};
    while (haystack < haystack_end) {
        // A byte that does not begin a well-formed codepoint folds to itself and resyncs by one byte.
        haystack_rune_length = sz_rune_decode(haystack, haystack_end, &haystack_rune);

        // Pre-fill positions [3] and [4] with sentinels before folding.
        // The fold will overwrite positions it uses; unused positions keep the sentinel.
        // This branchlessly prevents stale data from causing false matches.
        sz_rune_t sentinel = ~third_needle_folded;
        haystack_folded_runes[3] = sentinel;
        haystack_folded_runes[4] = sentinel;
        // Export into the last 3 rune entries of the 5-element array,
        // keeping the first two positions with historical data untouched
        sz_size_t folded_count;
        if (haystack_rune_length == sz_rune_invalid_k) {
            haystack_folded_runes[2] = sz_rune_malformed_byte_((sz_u8_t)*haystack);
            folded_count = 1;
            haystack_rune_length = sz_rune_1byte_k;
        }
        else { folded_count = sz_unicode_fold_codepoint_(haystack_rune, haystack_folded_runes + 2); }

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
            // First matched rune slot: [0] for `match_at_012`, [1] for `match_at_123` (both history),
            // [2] for `match_at_234` (the current codepoint). The last matched rune is always within
            // the current codepoint, so the match always ends at the current codepoint's end.
            sz_cptr_t match_begin;
            if (match_at_012) match_begin = history_source_begin[0];
            else if (match_at_123) match_begin = history_source_begin[1];
            else match_begin = haystack;
            sz_cptr_t match_end = haystack + haystack_rune_length;
            *match_length = (sz_size_t)(match_end - match_begin);
            return match_begin;
        }

        // Mirror the folded-rune shift for the per-slot source-begin pointers.
        if (folded_count >= 2) {
            // Both new history runes come from the current codepoint.
            haystack_folded_runes[0] = haystack_folded_runes[folded_count];
            haystack_folded_runes[1] = haystack_folded_runes[folded_count + 1];
            history_source_begin[0] = haystack;
            history_source_begin[1] = haystack;
        }
        else {
            sz_assert_(folded_count == 1);
            // Slot [0] inherits old slot [1]; slot [1] takes the current codepoint.
            haystack_folded_runes[0] = haystack_folded_runes[1];
            haystack_folded_runes[1] = haystack_folded_runes[2];
            history_source_begin[0] = history_source_begin[1];
            history_source_begin[1] = haystack;
        }

        haystack += haystack_rune_length;
    }

    *match_length = 0;
    return STRINGZILLA_NULL_CHAR;
}

STRINGZILLA_OUTLINED_ sz_cptr_t sz_utf8_uncased_search_serial_( //
    sz_cptr_t haystack, sz_size_t haystack_length,              //
    sz_cptr_t needle, sz_size_t needle_length,                  //
    sz_utf8_uncased_needle_t const *needle_metadata, sz_size_t *match_length) {

    if (needle_length == 0) {
        *match_length = 0;
        return haystack;
    }

    if (needle_metadata->script == sz_utf8_uncased_rune_invariant_k) {
        sz_cptr_t result = sz_find_serial_(haystack, haystack_length, needle, needle_length);
        if (result) {
            *match_length = needle_length;
            return result;
        }
        *match_length = 0;
        return STRINGZILLA_NULL_CHAR;
    }

    // For short needles (up to 12 bytes which can fold to at most ~6 runes), try hash-free search.
    // We fold the needle first and dispatch based on the folded rune count.
    // This avoids ring buffer setup, hash multiplier computation, and rolling hash updates.
    if (needle_length <= 12) {
        sz_rune_t folded_runes[4]; // 4th slot accessed before loop exit
        sz_size_t folded_count = 0;
        sz_utf8_folded_iter_t iter;
        sz_utf8_folded_iter_init_(&iter, needle, needle_length);
        sz_rune_t rune;
        while (folded_count < 4 && sz_utf8_folded_iter_next_(&iter, &rune)) folded_runes[folded_count++] = rune;

        // Dispatch based on folded rune count
        switch (folded_count) {
        case 1:
            return sz_utf8_uncased_search_1folded_serial_( //
                haystack, haystack_length,                 //
                folded_runes[0], match_length);
        case 2:
            return sz_utf8_uncased_search_2folded_serial_( //
                haystack, haystack_length,                 //
                folded_runes[0], folded_runes[1], match_length);
        case 3:
            return sz_utf8_uncased_search_3folded_serial_( //
                haystack, haystack_length,                 //
                folded_runes[0], folded_runes[1], folded_runes[2], match_length);
        default: break; // 4+ folded runes: fall through to Rabin-Karp
        }
    }

    sz_size_t const ring_capacity = 32;
    sz_rune_t needle_runes[32];
    sz_size_t needle_prefix_count = 0, needle_total_count = 0;
    sz_u64_t needle_hash = 0;
    {
        sz_utf8_folded_iter_t needle_iter;
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
        return STRINGZILLA_NULL_CHAR;
    }

    sz_u64_t hash_multiplier = 1;
    for (sz_size_t lane_index = 1; lane_index < needle_prefix_count; ++lane_index) hash_multiplier *= 257;

    sz_rune_t window_runes[32];
    sz_cptr_t window_sources[32];     // Byte position of character that produced each window rune
    sz_size_t window_skip_counts[32]; // Runes to skip from first character's expansion
    sz_size_t ring_head = 0;
    sz_u64_t window_hash = 0;
    sz_utf8_folded_iter_t haystack_iter;
    sz_utf8_folded_iter_init_(&haystack_iter, haystack, haystack_length);

    sz_cptr_t window_start = haystack;
    sz_cptr_t current_source = haystack;
    sz_size_t current_skip = 0;
    sz_size_t window_count = 0;

    while (window_count < needle_prefix_count) {
        sz_cptr_t pre_advance_cursor = haystack_iter.ptr;
        sz_rune_t rune;
        if (!sz_utf8_folded_iter_next_(&haystack_iter, &rune)) break;
        window_runes[window_count] = rune;
        // Update source and skip only when starting a new character (not mid-expansion)
        if (haystack_iter.pending_idx <= 1 || haystack_iter.pending_count == 0) {
            current_source = pre_advance_cursor;
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
        return STRINGZILLA_NULL_CHAR;
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
            for (sz_size_t lane_index = 0; lane_index < first_segment; ++lane_index)
                mismatches += window_runes[ring_head + lane_index] != needle_runes[lane_index];
            for (sz_size_t lane_index = 0; lane_index < ring_head; ++lane_index)
                mismatches += window_runes[lane_index] != needle_runes[first_segment + lane_index];

            if (!mismatches) {
                sz_size_t skip_runes = window_skip_counts[ring_head];
                // Short needle: rune comparison above is sufficient verification
                if (needle_total_count <= ring_capacity) {
                    *match_length = (sz_size_t)(window_end - window_start);
                    return window_start;
                }
                // Long needle: verify full needle from window_start, skipping runes if match
                // starts mid-expansion. Example: ẚ→"aʾ", needle starting with "ʾ" must skip "a".
                sz_utf8_folded_iter_t verify_haystack_iter;
                sz_utf8_folded_iter_init_(&verify_haystack_iter, window_start,
                                          (sz_size_t)(haystack + haystack_length - window_start));
                // Skip runes within first character's expansion
                sz_rune_t skip_rune;
                for (sz_size_t skip_index = 0; skip_index < skip_runes; ++skip_index)
                    sz_utf8_folded_iter_next_(&verify_haystack_iter, &skip_rune);
                // Now verify full needle against remaining haystack
                sz_utf8_folded_iter_t verify_needle_iter;
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

        sz_cptr_t pre_advance_cursor = haystack_iter.ptr;
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
            current_source = pre_advance_cursor;
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
    return STRINGZILLA_NULL_CHAR;
}

#pragma endregion Substring Search

/*  The character safety classifier and needle-metadata builder are ISA-agnostic: they only
 *  depend on the serial Unicode core (rune parsing and folding). The SIMD kernels (Ice Lake,
 *  etc.) consume the metadata it produces, so it lives here in the serial scaffolding rather
 *  than behind any `STRINGZILLA_TARGET_*` gate, keeping it reachable for every backend including
 *  pure serial builds. */
#pragma region Character Safety Profiles

/**
 *  @brief Determine safety profile for a character across all script contexts.
 *
 *  This function encodes the contextual safety rules from the ASCII selector and applies them
 *  consistently to all paths that include ASCII.
 *
 *  Using 0 for boundary markers is safe even though NUL (U+0000) is a valid codepoint in
 *  StringZilla's length-based strings. This works because:
 *
 *  1. NUL is valid ASCII (< 0x80), so boundary and actual NUL are treated identically
 *  2. Ligature checks use inequality (lower_prev ≠ 'f'), and 0 never matches letters
 *  3. NUL doesn't participate in any Unicode case folding or ligature expansions
 *
 *  The neighbor-of-neighbor context, @p prev_prev_rune and @p next_next_rune, enables position-1
 *  and position-N-2 detection for the 's' rule: if prev_prev = 0 and prev ≠ 0, we're at position 1;
 *  if next_next = 0 and next ≠ 0, we're at position N-2.
 *
 *  @param[in] rune The decoded codepoint.
 *  @param[in] rune_bytes UTF-8 byte length of this codepoint (1-4).
 *  @param[in] prev_rune Previous codepoint, 0 if at start.
 *  @param[in] next_rune Next codepoint, 0 if at end.
 *  @param[in] prev_prev_rune Codepoint before @p prev_rune, 0 if prev is at start.
 *  @param[in] next_next_rune Codepoint after @p next_rune, 0 if next is at end.
 *  @param[out] safety_profiles Safety flags for each script path.
 *  @return The primary fast path preferred for this rune.
 */
STRINGZILLA_CONSTEXPR sz_utf8_uncased_rune_safety_profile_t sz_utf8_uncased_rune_safety_profile_( //
    sz_rune_t rune, sz_size_t rune_bytes,                                                         //
    sz_rune_t prev_rune, sz_rune_t next_rune,                                                     //
    sz_rune_t prev_prev_rune, sz_rune_t next_next_rune,                                           //
    unsigned int *safety_profiles) {

    unsigned safety = 0;

    // Bitmasks for profiles that share identical ASCII rules
    unsigned int western_group = //
        (1 << sz_utf8_uncased_rune_safe_western_europe_k);
    unsigned int central_viet_group =                       //
        (1 << sz_utf8_uncased_rune_safe_central_europe_k) | //
        (1 << sz_utf8_uncased_rune_safe_vietnamese_k);
    unsigned int strict_ascii_group =                   //
        (1 << sz_utf8_uncased_rune_ascii_invariant_k) | //
        (1 << sz_utf8_uncased_rune_safe_cyrillic_k) |   //
        (1 << sz_utf8_uncased_rune_safe_greek_k) |      //
        (1 << sz_utf8_uncased_rune_safe_armenian_k) |   //
        (1 << sz_utf8_uncased_rune_safe_georgian_k);

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
            case 'b':
            case 'c':
            case 'd':
            case 'e':
            case 'g':
            case 'm':
            case 'o':
            case 'p':
            case 'q':
            case 'r':
            case 'u':
            case 'v':
            case 'x':
            case 'z': safety |= strict_ascii_group | central_viet_group | western_group; break;

            // 'k':
            // - Strict: unsafe. 'K' (U+212A, E2 84 AA) → 'k' (U+006B, 6B).
            // - Western/Central/Viet: safe. Kelvin sign detected in haystack.
            case 'k': safety |= central_viet_group | western_group; break;

            // 'a':
            // - Strict/Central/Viet: Contextual. Can't be last; can't precede 'ʾ' (U+02BE, CA BE).
            //   Avoids: 'ẚ' (U+1E9A, E1 BA 9A) → "aʾ" (U+0061 U+02BE, 61 CA BE).
            // - Western: safe. Expansion detected in haystack.
            case 'a':
                if (at_end == sz_false_k && next_ascii) safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;

            // 'h':
            // - Strict/Central/Viet: Contextual. Can't be last; can't precede '̱' (U+0331, CC B1).
            //   Avoids: 'ẖ' (U+1E96, E1 BA 96) → "ẖ" (U+0068 U+0331, 68 CC B1).
            // - Western: safe. Expansion detected in haystack.
            case 'h':
                if (at_end == sz_false_k && next_ascii) safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;

            // 'j':
            // - All: Contextual. Can't be last; can't precede '̌' (U+030C).
            //   Avoids: 'ǰ' (U+01F0) → "ǰ" (U+006A U+030C, 6A CC 8C).
            //   Western profile does not detect this in haystack scan.
            case 'j':
                if (at_end == sz_false_k && next_ascii)
                    safety |= strict_ascii_group | central_viet_group | western_group;
                break;

            // 'w':
            // - Strict/Central/Viet: Contextual. Can't be last; can't precede '̊' (U+030A).
            //   Avoids: 'ẘ' (U+1E98) → "ẘ" (U+0077 U+030A, 77 CC 8A).
            // - Western: safe. Expansion detected in haystack.
            case 'w':
                if (at_end == sz_false_k && next_ascii) safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;

            // 'y':
            // - Strict/Central/Viet: Contextual. Can't be last; can't precede '̊' (U+030A).
            //   Avoids: 'ẙ' (U+1E99) → "ẙ" (U+0079 U+030A, 79 CC 8A).
            // - Western: safe. Expansion detected in haystack.
            case 'y':
                if (at_end == sz_false_k && next_ascii) safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;

            // 'n':
            // - ASCII/Cyrillic/Greek: Contextual. Can't be first; can't follow 'ʼ' (U+02BC, CA BC).
            //   Avoids: 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E).
            // - Armenian: unsafe. The Armenian kernel cannot handle the expansion of 'ŉ' (U+0149,
            //   C5 89) into "ʼn" (U+02BC U+006E, CA BC 6E), and the character 'n' can match its 2nd
            //   part, causing false positives.
            // - Western/Central/Viet: Contextual, same as above.
            //   Western profile does not detect this in haystack scan.
            case 'n':
                // Exclude Armenian - it cannot handle 'ŉ' (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E)
                if (at_start == sz_false_k && prev_ascii) {
                    safety |= (1 << sz_utf8_uncased_rune_ascii_invariant_k) | //
                              (1 << sz_utf8_uncased_rune_safe_cyrillic_k) |   //
                              (1 << sz_utf8_uncased_rune_safe_greek_k);       //
                    // Armenian EXCLUDED: sz_utf8_uncased_rune_safe_armenian_k
                    safety |= central_viet_group | western_group;
                }
                break;

            // 'i':
            // - All: Contextual. Can't be first or last; can't follow 'f'; can't precede '̇' (U+0307, CC 87).
            //   Avoids: 'İ' (U+0130, C4 B0) → "i̇" (U+0069 U+0307, 69 CC 87),
            //   and 'ﬁ' (U+FB01, EF AC 81) → "fi" (U+0066 U+0069, 66 69).
            //   Western profile does not detect Turkish 'İ' expansion.
            case 'i':
                if (at_start == sz_false_k && at_end == sz_false_k && next_ascii && lower_prev != 'f')
                    safety |= strict_ascii_group | central_viet_group | western_group;
                break;

            // 'l':
            // - Strict/Central/Viet: Contextual. Can't be first; can't follow 'f'.
            //   Avoids: 'ﬂ' (U+FB02, EF AC 82) → "fl" (U+0066 U+006C, 66 6C).
            // - Western: safe. Ligatures detected in haystack.
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
            // - Western: safe. Ligatures/expansion detected in haystack.
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
            // - Western: safe. Ligatures detected in haystack.
            case 'f':
                if (at_start == sz_false_k && at_end == sz_false_k && prev_ascii && next_ascii && lower_prev != 'f' &&
                    lower_next != 'f' && lower_next != 'i' && lower_next != 'l')
                    safety |= strict_ascii_group | central_viet_group;
                safety |= western_group;
                break;

            // 's'
            // - Strict: unsafe. 'ſ' (U+017F, C5 BF) → 's' (U+0073, 73).
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
        return sz_utf8_uncased_rune_ascii_invariant_k;
    }

    // 2-byte UTF-8 (U+0080 to U+07FF):
    // Must check exact ranges that the fold functions handle, not just lead bytes
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
                safety |= (1 << sz_utf8_uncased_rune_safe_greek_k);
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
        if (lead == 0xC6) { safety |= (1 << sz_utf8_uncased_rune_safe_vietnamese_k); }

        // Cyrillic - check the exact ranges that
        // sz_utf8_uncased_search_cyrillic_fold_naively_zmm_icelake_ handles
        // D0 80-BF: U+0400-U+043F (includes uppercase and lowercase)
        // D1 80-9F: U+0440-U+045F (lowercase continuation)
        // Note: D2/D3 Extended Cyrillic BANNED from SIMD kernel - needles with D2/D3 use serial fallback
        if ((lead == 0xD0 && second >= 0x80 && second <= 0xBF) || //
            (lead == 0xD1 && second >= 0x80 && second <= 0x9F)) { //
            safety |= (1 << sz_utf8_uncased_rune_safe_cyrillic_k);
        }

        // Greek - check the exact ranges that
        // sz_utf8_uncased_search_greek_fold_naively_zmm_icelake_ handles
        // CE 86-8F: accented uppercase Ά-Ώ (with gaps at 87, 8B, 8D)
        //   - exclude CE 90: 'ΐ' (U+0390) expands to 3 codepoints.
        // CE 91-A9: basic uppercase Α-Ω
        // CE AA-AB: dialytika uppercase Ϊ-Ϋ
        // CE AC-AF: accented lowercase ά-ί
        //   - exclude CE B0: 'ΰ' (U+03B0) expands to 3 codepoints.
        // CE B1-BF: basic lowercase α-ο
        // CF 80-89: basic lowercase π-ω (includes final sigma at 82, sigma at 83)
        // CF 8A-8E: accented/dialytika lowercase ϊ-ώ
        if (lead == 0xCE) {
            // Accented uppercase (with gaps) - exclude 87, 8B, 8D, 90
            if ((second >= 0x86 && second <= 0x8F) && second != 0x87 && second != 0x8B && second != 0x8D &&
                second != 0x90) {
                safety |= (1 << sz_utf8_uncased_rune_safe_greek_k);
            }
            // Basic uppercase Α-Ω
            if (second >= 0x91 && second <= 0xA9) { safety |= (1 << sz_utf8_uncased_rune_safe_greek_k); }
            // Dialytika uppercase Ϊ-Ϋ
            if (second >= 0xAA && second <= 0xAB) { safety |= (1 << sz_utf8_uncased_rune_safe_greek_k); }
            // Accented lowercase ά-ί
            if (second >= 0xAC && second <= 0xAF) { safety |= (1 << sz_utf8_uncased_rune_safe_greek_k); }
            // Basic lowercase α-ο - exclude B0 (ΰ expands)
            if (second >= 0xB1 && second <= 0xBF) { safety |= (1 << sz_utf8_uncased_rune_safe_greek_k); }
        }
        if (lead == 0xCF) {
            // Basic lowercase π-ω
            if (second >= 0x80 && second <= 0x89) { safety |= (1 << sz_utf8_uncased_rune_safe_greek_k); }
            // Accented/dialytika lowercase ϊ-ώ
            if (second >= 0x8A && second <= 0x8E) { safety |= (1 << sz_utf8_uncased_rune_safe_greek_k); }
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

            if (is_armenian_range && armenian_safe) { safety |= (1 << sz_utf8_uncased_rune_safe_armenian_k); }
        }

        // Output safety and determine primary script for 2-byte runes
        // For case-invariant non-ASCII runes, add the ASCII-invariant bit.
        // This enables fast ASCII kernel for needles like "中文字" that contain no cased characters.
        // ASCII fold only affects bytes 0x41-0x5A (A-Z), so all other bytes pass through unchanged.
        if (sz_rune_is_uncased_(rune)) safety |= (1 << sz_utf8_uncased_rune_ascii_invariant_k);
        *safety_profiles = safety;
        if (rune >= 0x0080 && rune <= 0x00FF) return sz_utf8_uncased_rune_safe_western_europe_k; // Latin-1 Supplement
        if (rune >= 0x0100 && rune <= 0x024F) return sz_utf8_uncased_rune_safe_central_europe_k; // Latin Extended-A/B
        if (rune >= 0x0370 && rune <= 0x03FF) return sz_utf8_uncased_rune_safe_greek_k;          // Greek
        if (rune >= 0x0400 && rune <= 0x04FF) return sz_utf8_uncased_rune_safe_cyrillic_k;       // Cyrillic
        if (rune >= 0x0530 && rune <= 0x058F) return sz_utf8_uncased_rune_safe_armenian_k;       // Armenian
        return sz_utf8_uncased_rune_invariant_k;
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
            else { safety |= (1 << sz_utf8_uncased_rune_safe_vietnamese_k); }
        }

        // Georgian Mkhedruli (E1 83 90-BF range)
        // U+10D0-U+10FF maps to E1 83 90 - E1 83 BF
        // Mkhedruli is caseless, so all characters are safe for the Georgian kernel.
        if (lead == 0xE1 && second == 0x83 && third >= 0x90) { safety |= (1 << sz_utf8_uncased_rune_safe_georgian_k); }

        // Output safety and determine primary script for 3-byte runes
        // For case-invariant non-ASCII runes (like CJK), add the ASCII-invariant bit.
        if (sz_rune_is_uncased_(rune)) safety |= (1 << sz_utf8_uncased_rune_ascii_invariant_k);
        *safety_profiles = safety;
        if (rune >= 0x10D0 && rune <= 0x10FF) return sz_utf8_uncased_rune_safe_georgian_k; // Georgian Mkhedruli
        if (rune >= 0x1E00 && rune <= 0x1EFF)
            return sz_utf8_uncased_rune_safe_vietnamese_k; // Latin Extended Additional
        return sz_utf8_uncased_rune_invariant_k;
    }

    // 4-byte UTF-8 - currently no fast paths, but case-invariant 4-byte runes can use ASCII kernel
    if (sz_rune_is_uncased_(rune)) safety |= (1 << sz_utf8_uncased_rune_ascii_invariant_k);
    *safety_profiles = safety;
    return sz_utf8_uncased_rune_invariant_k;
}

/**
 *  @brief Compute diversity score for a byte sequence.
 *
 *  Uses a 256-bit bitmap to efficiently count distinct byte values. Higher scores indicate more
 *  diverse byte values, which lead to better filtering during SIMD search (fewer false positives).
 *
 *  @param[in] data Pointer to byte sequence.
 *  @param[in] length Length of byte sequence.
 *  @return Count of distinct byte values (0-256).
 */
STRINGZILLA_CONSTEXPR sz_size_t sz_utf8_probe_diversity_score_(sz_u8_t const *data, sz_size_t length) {
    if (length <= 1) return length;
    sz_u64_t seen[4] = {0, 0, 0, 0}; // 256-bit bitmap
    sz_size_t distinct = 0;
    for (sz_size_t byte_index = 0; byte_index < length; ++byte_index) {
        sz_u8_t byte = data[byte_index];
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
 *  The objective is as follows. For a given needle, find a slice that, when folded, fits into 16
 *  bytes and where all characters are "safe" with respect to a certain path. If no such path can be
 *  found, an empty result is returned. It might be the case for a search query like "s" or "n",
 *  that by itself isn't safe for any path given the number of Unicode characters expanding into
 *  multiple 's'- or 'n'-containing sequences. The selected safe folded slice will never begin
 *  mid-character in the needle, so if it starts with an 'ŉ' (U+0149, C5 89), we can't choose 'n'
 *  (6E), the second half of its folded sequence, as a starting point.
 *
 *  The algorithm is as follows. Iterate through the arbitrary-case "ŉEeDlE_WITH_LONG_SUFFIX",
 *  unpacking runes. For each input rune, perform folding, expanding into a sequence, like 'ŉ'
 *  (U+0149, C5 89) → "ʼn" (U+02BC U+006E, CA BC 6E). Continue unpacking the rest, until we reach a
 *  16-byte limit, like:
 *
 *  @verbatim
 *  ʼ     n  e  e  d  l  e  _  w  i  t  h  _  l  o  n  g
 *  CA BC 6E 45 45 44 4C 45 5F 57 49 54 48 5F 4C 4F 4E 47
 *  @endverbatim
 *
 *  At this point, we need to trim it to make sure its characters satisfy boundary conditions.
 *  Assuming at the next step we'll move the iterator to the next input rune to point to the 'E'
 *  (U+0045) input character, we only trim from the end, but also invalidate the whole starting
 *  position if a bad character is chosen at start. For a safe window starting position we can have
 *  multiple length variants, assuming different safe paths can have different rules for the last
 *  symbol in the safe sequence.
 *
 *  Once we have a safe window for a certain script, we evaluate its diversity score - the number of
 *  distinct byte values in the folded window. The more diverse, the better! We keep track of the
 *  best seen window for each script.
 *
 *  We also track not only the safety with respect to a certain profile, but also applicability. For
 *  example, the needle "xyz" is safe with respect to the Western European path, as well as Central
 *  European, Vietnamese, and potentially others. But it's pure ASCII, and we shouldn't pay the cost
 *  of complex Vietnamese case-folding of triple-byte Latin extensions for just "xyz". So we must
 *  invalidate the "safe path" if it's just "safe", but not ideal.
 *
 *  In the end, we'll have up to 7 best safe windows, one per script path. The heuristic is:
 *
 *  - Prefer ASCII, if there is an ASCII-safe path at least 4 bytes wide with at least 4 distinct
 *    byte values. It's only one subtraction, a comparison, and a masked addition. Cheapest of all.
 *  - Pick the most diverse variant from all others, if the ASCII variant isn't good enough.
 *
 *  We then identify the four "probe" positions within the ≤ 16 byte folded safe window, one more
 *  than in exact substring search kernels with Raita heuristics:
 *
 *  1. implicit at `refined->folded_slice[0]`
 *  2. stored in `refined->probe_second` - targets last byte of 2nd character when 4+ chars
 *  3. stored in `refined->probe_third` - targets last byte of 3rd character when 4+ chars
 *  4. implicit at `refined->folded_slice[refined->folded_slice_length - 1]`
 *
 *  By aiming at the last byte of each UTF-8 codepoint we maximize diversity, as in a Russian text
 *  almost all letters will have the same first byte, but mostly different second byte. The same is
 *  true for many other languages. For short strings (< 4 bytes), probes will necessarily overlap -
 *  this is expected. The function also sets @c offset_in_unfolded and @c length_in_unfolded to
 *  track where the selected folded slice came from in the original unfolded input.
 *
 *  A caseless needle skips the windows altogether, profiled @c sz_utf8_uncased_rune_invariant_k.
 *
 *  @param[in] needle Pointer to needle string (original, not folded).
 *  @param[in] needle_length Length in bytes.
 *  @param[out] refined The prepared needle, every field filled.
 */
STRINGZILLA_OUTLINED_ void sz_utf8_uncased_needle_init_serial_(sz_cptr_t needle, sz_size_t needle_length, //
                                                               sz_utf8_uncased_needle_t *refined) {

    // Per-script window state during iteration
    typedef struct {

        /** Byte offset in the original needle. */
        sz_size_t start_offset;

        /** Bytes consumed from the original needle. */
        sz_size_t input_length;
        sz_u8_t folded_bytes[16];
        sz_size_t folded_length;

        /** Whether the window holds at least one character of the script. */
        sz_bool_t applicable;

        /** Whether the window's continuity broke, so it extends no further. */
        sz_bool_t broken;

        /** Distinct byte count, computed at the end of each starting position. */
        sz_size_t diversity;
    } script_window_t_;

    // Number of script kernels (indices 1-8 used, index 0 reserved)
    sz_size_t const num_scripts = 9;

    // Best window found so far for each script
    script_window_t_ best[9];
    for (sz_size_t script_index = 0; script_index < num_scripts; ++script_index) {
        best[script_index].start_offset = 0;
        best[script_index].input_length = 0;
        best[script_index].folded_length = 0;
        best[script_index].applicable = sz_false_k;
        best[script_index].broken = sz_false_k;
        best[script_index].diversity = 0;
    }

    refined->start = needle, refined->length = needle_length;
    refined->offset_in_unfolded = refined->length_in_unfolded = 0;
    refined->folded_slice_length = refined->probe_second = refined->probe_third = 0;

    // A caseless needle, the empty one included, is found by an exact substring search.
    if (sz_utf8_find_cased_serial_(needle, needle_length) == STRINGZILLA_NULL_CHAR) {
        refined->script = sz_utf8_uncased_rune_invariant_k;
        return;
    }

    // A needle containing any byte that does not begin a well-formed codepoint cannot be window-analyzed by the
    // unchecked decode below; route it to the serial kernel, which handles malformed bytes losslessly (each is
    // folded to itself and resyncs by one byte), keeping SIMD and serial results identical.
    if (sz_utf8_find_malformed(needle, needle_length) != STRINGZILLA_NULL_CHAR) {
        refined->script = sz_utf8_uncased_rune_fallback_serial_k;
        return;
    }

    sz_u8_t const *needle_start = (sz_u8_t const *)needle;
    sz_u8_t const *needle_end = needle_start + needle_length;

    // Iterate through each starting position in the needle (stepping by rune)
    for (sz_u8_t const *needle_cursor = needle_start; needle_cursor < needle_end;) {
        // Current window being built for each script at this starting position
        script_window_t_ current[9];
        for (sz_size_t script_index = 0; script_index < num_scripts; ++script_index) {
            current[script_index].start_offset = (sz_size_t)(needle_cursor - needle_start);
            current[script_index].input_length = 0;
            current[script_index].folded_length = 0;
            current[script_index].applicable = sz_false_k;
            current[script_index].broken = sz_false_k;
            current[script_index].diversity = 0;
        }

        // Track context for safety profile evaluation
        sz_rune_t prev_prev_rune = 0;
        sz_rune_t prev_rune = 0;

        // Fold forward from needle_cursor until 16 bytes or needle end
        sz_u8_t const *position = needle_cursor;
        sz_bool_t any_active = sz_true_k;

        while (position < needle_end && any_active) {
            // Parse current rune
            sz_rune_t rune;
            sz_rune_length_t const rune_bytes = sz_rune_decode_unchecked((sz_cptr_t)position, &rune);
            if (position + rune_bytes > needle_end) break; // Incomplete rune

            // Parse next rune for context (if available)
            sz_rune_t next_rune = 0;
            sz_rune_length_t next_bytes = sz_rune_invalid_k;
            if (position + rune_bytes < needle_end) {
                next_bytes = sz_rune_decode_unchecked((sz_cptr_t)(position + rune_bytes), &next_rune);
                if (position + rune_bytes + next_bytes > needle_end) next_rune = 0;
            }

            // Parse next-next rune for context
            sz_rune_t next_next_rune = 0;
            if (next_rune != 0 && position + rune_bytes + next_bytes < needle_end) {
                sz_rune_length_t const next_next_bytes = sz_rune_decode_unchecked(
                    (sz_cptr_t)(position + rune_bytes + next_bytes), &next_next_rune);
                if (position + rune_bytes + next_bytes + next_next_bytes > needle_end) next_next_rune = 0;
            }

            // Get safety mask and primary script for this rune
            unsigned safety_mask = 0;
            sz_utf8_uncased_rune_safety_profile_t primary_script = sz_utf8_uncased_rune_safety_profile_( //
                rune, rune_bytes, prev_rune, next_rune, prev_prev_rune, next_next_rune, &safety_mask);

            // Fold this rune
            sz_rune_t folded_runes[4];
            sz_size_t folded_count = sz_unicode_fold_codepoint_(rune, folded_runes);

            // Convert folded runes to UTF-8 bytes
            sz_u8_t folded_utf8[16];
            sz_size_t folded_utf8_length = 0;
            for (sz_size_t rune_index = 0; rune_index < folded_count; ++rune_index) {
                folded_utf8_length += sz_rune_encode(folded_runes[rune_index], folded_utf8 + folded_utf8_length);
            }

            // Update each script's window
            any_active = sz_false_k;
            for (sz_size_t script_index = 1; script_index < num_scripts; ++script_index) {
                if (current[script_index].broken) continue;

                // Check if this rune is safe for this script
                sz_bool_t is_safe = (safety_mask & (1u << script_index)) ? sz_true_k : sz_false_k;

                // Check if adding this rune would exceed 16 bytes
                if (is_safe && current[script_index].folded_length + folded_utf8_length <= 16) {
                    // Extend this script's window
                    for (sz_size_t byte_index = 0; byte_index < folded_utf8_length; ++byte_index) {
                        current[script_index].folded_bytes[current[script_index].folded_length + byte_index] =
                            folded_utf8[byte_index];
                    }
                    current[script_index].folded_length += folded_utf8_length;
                    current[script_index].input_length += rune_bytes;

                    // Mark as applicable if primary script matches
                    if (primary_script == script_index) { current[script_index].applicable = sz_true_k; }
                    any_active = sz_true_k;
                }
                else {
                    // Window broken for this script
                    current[script_index].broken = sz_true_k;
                }
            }

            // Update context for next iteration
            prev_prev_rune = prev_rune;
            prev_rune = rune;
            position += rune_bytes;
        }

        // Compare current to best for each script
        for (sz_size_t script_index = 1; script_index < num_scripts; ++script_index) {
            if (!current[script_index].applicable || current[script_index].folded_length == 0) continue;

            // Compute diversity score
            current[script_index].diversity = sz_utf8_probe_diversity_score_(current[script_index].folded_bytes,
                                                                             current[script_index].folded_length);

            // Update best if this is better (prefer higher diversity, then longer length)
            if (current[script_index].diversity > best[script_index].diversity ||
                (current[script_index].diversity == best[script_index].diversity &&
                 current[script_index].folded_length > best[script_index].folded_length)) {
                best[script_index] = current[script_index];
            }
        }

        // Advance to next rune for next starting position
        sz_rune_t skip_rune;
        sz_rune_length_t const skip_length = sz_rune_decode_unchecked((sz_cptr_t)needle_cursor, &skip_rune);
        needle_cursor += skip_length;
    }

    // Select final kernel based on best windows
    // Rule: Prefer ASCII if >=4 bytes with >=4 diversity; otherwise pick most diverse applicable
    sz_size_t chosen_script = 0;
    sz_size_t best_diversity = 0;

    // Check ASCII preference
    if (best[sz_utf8_uncased_rune_ascii_invariant_k].applicable &&
        best[sz_utf8_uncased_rune_ascii_invariant_k].folded_length >= 4 &&
        best[sz_utf8_uncased_rune_ascii_invariant_k].diversity >= 4) {
        chosen_script = sz_utf8_uncased_rune_ascii_invariant_k;
    }
    else {
        // Find most diverse applicable script
        for (sz_size_t script_index = 1; script_index < num_scripts; ++script_index) {
            if (best[script_index].applicable && best[script_index].diversity > best_diversity) {
                best_diversity = best[script_index].diversity;
                chosen_script = script_index;
            }
        }
    }

    // If no applicable window found, fall back to serial
    if (chosen_script == 0) {
        refined->script = sz_utf8_uncased_rune_fallback_serial_k;
        return;
    }

    // Populate output metadata
    refined->script = (sz_u8_t)chosen_script;
    refined->offset_in_unfolded = best[chosen_script].start_offset;
    refined->length_in_unfolded = best[chosen_script].input_length;
    refined->folded_slice_length = (sz_u8_t)best[chosen_script].folded_length;

    // Copy folded bytes
    for (sz_size_t byte_index = 0; byte_index < best[chosen_script].folded_length; ++byte_index) {
        refined->folded_slice[byte_index] = best[chosen_script].folded_bytes[byte_index];
    }

    // Compute probe positions - target last bytes of UTF-8 codepoints for maximum diversity
    sz_size_t folded_length = best[chosen_script].folded_length;
    if (folded_length == 0) {
        refined->probe_second = 0;
        refined->probe_third = 0;
        return;
    }

    // Find character end positions in the folded slice
    // A byte is a character's last byte if the next byte is a UTF-8 leader (not continuation)
    sz_size_t char_ends[16];
    sz_size_t char_count = 0;
    for (sz_size_t byte_index = 0; byte_index < folded_length; ++byte_index) {
        sz_u8_t next = (byte_index + 1 < folded_length) ? refined->folded_slice[byte_index + 1]
                                                        : 0xC0; // Fake leader at end
        if ((next & 0xC0) != 0x80) {                            // Next is not a continuation byte
            if (char_count < 16) char_ends[char_count++] = byte_index;
        }
    }

    // Determine probe positions
    if (char_count >= 4) {
        // 4+ characters: target last bytes of 2nd and 3rd characters
        refined->probe_second = (sz_u8_t)char_ends[1];
        refined->probe_third = (sz_u8_t)char_ends[2];
    }
    else if (folded_length <= 3) {
        // Very short: probes overlap
        refined->probe_second = (folded_length > 1) ? 1 : 0;
        refined->probe_third = (folded_length > 1) ? 1 : 0;
    }
    else {
        // 1-3 characters but 4+ bytes: use byte diversity search
        sz_u8_t byte_first = refined->folded_slice[0];
        sz_u8_t byte_last = refined->folded_slice[folded_length - 1];

        sz_size_t probe_second = folded_length / 3;
        sz_size_t probe_third = (folded_length * 2) / 3;

        // Try to find positions with bytes distinct from first/last
        for (sz_size_t byte_index = 1; byte_index < folded_length - 1; ++byte_index) {
            if (refined->folded_slice[byte_index] != byte_first && refined->folded_slice[byte_index] != byte_last) {
                probe_second = byte_index;
                break;
            }
        }

        sz_u8_t byte_second = refined->folded_slice[probe_second];
        for (sz_size_t byte_index = probe_second + 1; byte_index < folded_length - 1; ++byte_index) {
            if (refined->folded_slice[byte_index] != byte_first && refined->folded_slice[byte_index] != byte_last &&
                refined->folded_slice[byte_index] != byte_second) {
                probe_third = byte_index;
                break;
            }
        }

        // Clamp bounds
        if (probe_second == 0) probe_second = 1;
        if (probe_third >= folded_length - 1) probe_third = folded_length - 2;
        if (probe_third <= probe_second && probe_second + 1 < folded_length - 1) probe_third = probe_second + 1;

        refined->probe_second = (sz_u8_t)probe_second;
        refined->probe_third = (sz_u8_t)probe_third;
    }
}

#pragma endregion Character Safety Profiles

#if STRINGZILLA_TARGET_SERIAL

STRINGZILLA_API sz_status_t sz_utf8_uncased_needle_init_serial(sz_cptr_t needle, sz_size_t needle_length,
                                                               sz_utf8_uncased_needle_t *prepared, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_utf8_uncased_needle_init_serial_(needle, needle_length, prepared);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_utf8_uncased_search_serial(                                 //
    sz_cptr_t haystack, sz_size_t haystack_length, sz_utf8_uncased_needle_t const *needle, //
    sz_cptr_t *match, sz_size_t *match_length, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_utf8_uncased_search_serial_(haystack, haystack_length, needle->start, needle->length, needle,
                                            match_length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_utf8_uncased_order_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                         sz_size_t b_length, sz_ordering_t *ordering,
                                                         sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *ordering = sz_utf8_uncased_order_serial_(a, a_length, b, b_length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_utf8_find_cased_serial(sz_cptr_t text, sz_size_t length, sz_cptr_t *match,
                                                      sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_utf8_find_cased_serial_(text, length);
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_SERIAL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_UNCASED_SERIAL_H_
