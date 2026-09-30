/**
 *  @file test/utf8_uncased.cpp
 *  @author Ash Vardanian
 *  @date November 23, 2025
 *  @brief Uncased UTF-8 folding, search and order through the dispatch points and the C++ wrappers.
 */
#undef NDEBUG // ! Enable all assertions for testing

/** The Visual C++ run-time library detects incorrect iterator use, and asserts and displays a
 *  dialog box at run time on Windows. */
#if !defined(_ITERATOR_DEBUG_LEVEL) || _ITERATOR_DEBUG_LEVEL == 0
#define _ITERATOR_DEBUG_LEVEL 1
#endif

#if defined(STRINGZILLA_DEBUG)
#undef STRINGZILLA_DEBUG
#endif
#define STRINGZILLA_DEBUG 1 // ! Enforce aggressive logging in this translation unit

/*  Make sure to include the StringZilla headers before anything else, to intercept missing
 *  `#include` directives and other issues. */
#include <stringzilla/stringzilla.h>   // Primary C API
#include <stringzilla/stringzilla.hpp> // C++ string class replacement

#include <string> // Baseline
#include <vector> // `std::vector`

#include "cross.hpp" // `check_utf8_uncased_unit_`, `check_utf8_uncased_equivalence_`, `check_utf8_uncased_safety_`

namespace sz = ashvardanian::stringzilla;
using namespace sz::test;

/** The uncased dispatch points, in the shape of one backend's kernels. */
static utf8_uncased_kernels_t const utf8_uncased_dispatched {
    cpu_best<sz_utf8_uncased_fold_best>, cpu_best<sz_utf8_uncased_search_best>, cpu_best<sz_utf8_uncased_order_best>,
    cpu_best<sz_utf8_find_cased_best>};

#pragma region Unit

/**
 *  @brief Known-answer and C++ API coverage for the uncased UTF-8 family on hand-verifiable inputs.
 *
 *  Runs the dispatch points through the known-answer battery of @c check_utf8_uncased_unit_, then
 *  the C++ wrappers over the same cases, so a regression that the serial-vs-SIMD agreement tests
 *  would miss, as both share a wrong constant, is still caught against a hand-derived answer.
 */
void test_utf8_uncased_unit() {
    using str = sz::string_view_t;
    check_utf8_uncased_unit_(utf8_uncased_dispatched);

    // `utf8_uncased_search`: "world" in "Hello World", and "SS" over the 'ß' (U+00DF) of "ßfox".
    {
        let_verify(auto match = str("Hello World").utf8_uncased_search("world"),
                   match.offset == 6 && match.length == 5);
    }
    { let_verify(auto match = str("ßfox").utf8_uncased_search("SS"), match.offset == 0 && match.length == 2); }

    // `utf8_uncased_matches` yields every non-overlapping case-insensitive match, the twin of
    // Rust/Python `Utf8UncasedMatches`. Folding can change a match's byte length, so each yielded
    // slice spans the matched bytes, not the needle's length.
    {
        let_verify(
            auto hits = str("Ab aB AB xyz ab").utf8_uncased_matches("ab").template to<std::vector<std::string>>(),
            hits == (std::vector<std::string> {"Ab", "aB", "AB", "ab"}));
        verify(str("xyz").utf8_uncased_matches("ab").empty());
        // An empty needle matches at codepoint boundaries, so 3 times in the 3 bytes of "aé".
        verify(str("a\xC3\xA9").utf8_uncased_matches("").size() == 3);
        // A truncated rune is one step, capped at the end, where byte steps give 4.
        {
            str const truncated("a\xE4\xB8");
            verify(offsets_within(truncated, truncated.utf8_uncased_matches("")) ==
                   (std::vector<std::ptrdiff_t> {0, 1, 3}));
        }

        // 'ß' (U+00DF, 2 bytes) folds to "ss": each one is a length-2 match of the needle "SS".
        let_verify(auto folded = str("ßox ß").utf8_uncased_matches("SS").template to<std::vector<std::string>>(),
                   folded == (std::vector<std::string> {"ß", "ß"}));
    }

    // In-place fold on a mutable `sz::string_t`.
    {
        sz::string_t folded("HeLLo");
        verify(sz::succeeded(folded.try_utf8_uncased_fold()));
        verify(folded == "hello");
    }
    {
        sz::string_t folded("\xC3\x9F");
        verify(sz::succeeded(folded.try_utf8_uncased_fold()));
        verify(folded == "ss");
    }

    verify(str("Hello").utf8_uncased_order("HELLO") == sz_equal_k);
}

/**
 *  @brief Known-answer sweep of the uncased C++ wrappers across the world's scripts.
 *
 *  Ordering, finding, ligatures and expansions over Latin-1, Central European, German Eszett, math
 *  symbols, Greek, Cyrillic, Turkish, Armenian, Vietnamese, Georgian, Cherokee, Coptic, Glagolitic
 *  and the caseless scripts - CJK, Arabic, Hebrew and emoji - each with a hand-derived expected
 *  offset and byte length, including the runs that straddle a 64-byte SIMD block.
 */
void test_utf8_uncased_scripts_unit() {
    using str = sz::string_view_t;

    // Equal strings (ASCII)
    verify(str("hello").utf8_uncased_order("HELLO") == sz_equal_k);
    verify(str("abc").utf8_uncased_order("ABC") == sz_equal_k);
    verify(str("HeLLo WoRLd").utf8_uncased_order("hello world") == sz_equal_k);

    // ASCII Extensions
    let_verify(auto m = str("prefixhello").utf8_uncased_search("HELLO"), m.offset == 6 && m.length == 5);
    let_verify(auto m = str("hello_suffix").utf8_uncased_search("HELLO"), m.offset == 0 && m.length == 5);
    let_verify(auto m = str("mid_hello_mid").utf8_uncased_search("HELLO"), m.offset == 4 && m.length == 5);

    // Less than
    verify(str("abc").utf8_uncased_order("abd") == sz_less_k);
    verify(str("ab").utf8_uncased_order("abc") == sz_less_k);
    verify(str("ABC").utf8_uncased_order("abd") == sz_less_k);

    // Greater than
    verify(str("abd").utf8_uncased_order("abc") == sz_greater_k);
    verify(str("abcd").utf8_uncased_order("abc") == sz_greater_k);
    verify(str("ABD").utf8_uncased_order("abc") == sz_greater_k);

    // Latin-1 Supplement and Latin Extended-A, starting with German umlauts
    verify(str("schöner").utf8_uncased_order("SCHÖNER") == sz_equal_k);
    let_verify(auto m = str("Das ist ein schöner Tag").utf8_uncased_search("SCHÖNER"),
               m.offset == 12 && m.length == 8); // 'ö' (U+00F6, C3 B6) is 2 bytes

    // French Accents
    verify(str("café").utf8_uncased_order("CAFÉ") == sz_equal_k);
    verify(str("naïve").utf8_uncased_order("NAÏVE") == sz_equal_k);
    verify(str("À la carte").utf8_uncased_order("à la CARTE") == sz_equal_k);

    // Spanish/Portuguese
    verify(str("niño").utf8_uncased_order("NIÑO") == sz_equal_k);

    // Polish / Central European (Latin Extended-A):
    // "ĄĆĘŁŃÓŚŹŻ" → "ąćęłńóśźż", and "Zaółć gęślą jaźń" is a classic Polish pangram fragment
    verify(str("Zaółć gęślą jaźń").utf8_uncased_order("ZAÓŁĆ GĘŚLĄ JAŹŃ") == sz_equal_k);

    // Czech characters: ř (U+0159, C5 99), ž (U+017E, C5 BE), č (U+010D, C4 8D), ě (U+011B, C4 9B)
    verify(str("řžčě").utf8_uncased_order("ŘŽČĚ") == sz_equal_k);
    let_verify(auto m = str("Příklad").utf8_uncased_search("PŘÍKLAD"), m.offset == 0 && m.length == 9);
    let_verify(auto m = str("žena").utf8_uncased_search("ŽENA"), m.offset == 0 && m.length == 5);

    // Polish ł (U+0142, C5 82) in city name
    verify(str("Łódź").utf8_uncased_order("ŁÓDŹ") == sz_equal_k);
    let_verify(auto m = str("miasto Łódź").utf8_uncased_search("łódź"), m.offset == 7 && m.length == 7);

    // Hungarian: ő (U+0151, C5 91), ű (U+0171, C5 B1)
    verify(str("őű").utf8_uncased_order("ŐŰ") == sz_equal_k);
    let_verify(auto m = str("Erdő").utf8_uncased_search("ERDŐ"), m.offset == 0 && m.length == 5);
    let_verify(auto m = str("Győr").utf8_uncased_search("GYŐR"), m.offset == 0 && m.length == 5);

    // Central European at SIMD boundary (64 bytes)
    {
        std::string prefix(62, 'a');
        let_verify(auto m = str(prefix + "ž").utf8_uncased_search("Ž"), m.offset == 62 && m.length == 2);
        let_verify(auto m = str(prefix + "řž").utf8_uncased_search("ŘŽ"), m.offset == 62 && m.length == 4);
    }

    // German (Eszett 'ß'):
    // 'ß' (U+00DF, C3 9F) → "ss"
    // "straße" → "strasse"
    // "STRASSE" → "strasse"
    verify(str("straße").utf8_uncased_order("STRASSE") == sz_equal_k);
    verify(str("STRASSE").utf8_uncased_order("straße") == sz_equal_k);

    // Uppercase 'ẞ' (U+1E9E, E1 BA 9E) → "ss" or "ß" depending on fold
    // StringZilla generally folds to lowercase first. 'ẞ' → 'ss'.
    // Haystack uses 'ß' (2 bytes), Needle "SS".
    let_verify(auto m = str("straße").utf8_uncased_search("SS"),
               m.offset == 4 && m.length == 2); // Matches 'ß' (2 bytes)

    // Eszett Context Extensions
    let_verify(auto m = str("Eine straße").utf8_uncased_search("SS"),
               m.offset == 9 && m.length == 2); // "Eine " is 5 chars → 5 bytes + "stra" (4) = 9
    let_verify(auto m = str("straßebahn").utf8_uncased_search("SS"), m.offset == 4 && m.length == 2);
    let_verify(auto m = str("Eine straßebahn").utf8_uncased_search("SS"), m.offset == 9 && m.length == 2);

    // Same case-folding, but different relation
    let_verify(auto m = str("HelloäeßHelloL").utf8_uncased_search("helloäesshellol"), m.offset == 0 && m.length == 16);
    let_verify(auto m = str("helloäesshellol").utf8_uncased_search("HelloäeßHelloL"), m.offset == 0 && m.length == 16);

    // Same case-folding, but a different relation and needle length, due to the uppercase
    // three-byte 'ẞ' (U+1E9E, E1 BA 9E)
    let_verify(auto m = str("HelloäeẞHelloL").utf8_uncased_search("helloäesshellol"), m.offset == 0 && m.length == 17);
    let_verify(auto m = str("helloäesshellol").utf8_uncased_search("HelloäeẞHelloL"), m.offset == 0 && m.length == 16);

    // Haystack "STRASSE", Needle "straße"
    let_verify(auto m = str("STRASSE").utf8_uncased_search("straße"),
               m.offset == 0 && m.length == 7); // Matches "STRASSE" (7 bytes)

    // "Maße" → "MASSE"
    let_verify(auto m = str("Maße").utf8_uncased_search("MASSE"),
               m.offset == 0 && m.length == 5); // Matches "Maße" (5 bytes)

    // Haystack: "Fuss" (4 bytes), ending in "u", "s", "s".
    // Needle: "Fuß" (4 bytes), ending in "u", "ß".
    // They are equal in order, and searching "Fuß" in "Fuss" works.
    let_verify(auto m = str("Fuss").utf8_uncased_search("Fuß"),
               m.offset == 0 && m.length == 4); // Matches "Fuss"

    // Mid-expansion matching: the needle starts with 's', which takes the serial fallback.
    // Haystack: "ßfox" (5 bytes) folds to "ssfox".
    // Needle "sfox" matches at folded position 1, but we report offset 0, the start of ß.
    // Length is 5 because we consume the entire ß character, never pointing to half of it.
    let_verify(auto m = str("ßfox").utf8_uncased_search("sfox"), m.offset == 0 && m.length == 5);

    // Needle ends with 's': the suffix case, which takes the serial fallback.
    // Haystack: "foxß" folds to "foxss".
    // Needle "foxs" matches through the first 's' of the expansion.
    let_verify(auto m = str("fox\xC3\x9F").utf8_uncased_search("foxs"), m.offset == 0 && m.length == 5);

    // Cross-boundary case: "ßS" folds to "sss", which takes the serial fallback.
    // Haystack: "ßStra" (6 bytes) folds to "ssstra".
    // Needle "sstra" starts with 's', and would match at position 1, mid-ß, without the rule.
    // Length is 6 because we consume the entire haystack, as ß expands but is consumed whole.
    let_verify(auto m = str("ßStra").utf8_uncased_search("sstra"), m.offset == 0 && m.length == 6);

    // Needle with 's' not at boundary - should use fast SIMD path
    let_verify(auto m = str("teßt").utf8_uncased_search("tesst"), m.offset == 0 && m.length == 5);
    let_verify(auto m = str("maße").utf8_uncased_search("masse"), m.offset == 0 && m.length == 5);

    // Needle with 'ss' at boundary - also uses serial (can't match across ß boundary)
    let_verify(auto m = str("fo\xC3\x9F").utf8_uncased_search("foss"), m.offset == 0 && m.length == 4);
    let_verify(auto m = str("ßfo").utf8_uncased_search("ssfo"), m.offset == 0 && m.length == 4);

    // Math symbols:
    // Multiplication × (U+00D7, C3 97) and Division ÷ (U+00F7, C3 B7).
    // Often confusable with 'x' and '+'/'=', but strictly they are distinct.
    // They should equal themselves but not each other.
    verify(str("×").utf8_uncased_order("×") == sz_equal_k); // × == ×
    verify(str("÷").utf8_uncased_order("÷") == sz_equal_k); // ÷ == ÷
    verify(str("×").utf8_uncased_order("÷") != sz_equal_k); // × ≠ ÷
    verify(str("a×b").utf8_uncased_order("A×B") == sz_equal_k);

    // Math Context Extensions
    let_verify(auto m = str("2×3=6").utf8_uncased_search("×"), m.offset == 1 && m.length == 2);
    let_verify(auto m = str("6÷2=3").utf8_uncased_search("÷"), m.offset == 1 && m.length == 2);

    // Empty strings
    verify(str("").utf8_uncased_order("") == sz_equal_k);
    verify(str("a").utf8_uncased_order("") == sz_greater_k);
    verify(str("").utf8_uncased_order("a") == sz_less_k);

    // Greek
    // Basic casing: "αβγδ" vs "ΑΒΓΔ"
    verify(str("αβγδ").utf8_uncased_order("ΑΒΓΔ") == sz_equal_k);
    let_verify(auto m = str("αβγδ").utf8_uncased_search("ΑΒΓΔ"),
               m.offset == 0 && m.length == 8); // 4 * 2 bytes = 8 bytes

    // Greek Context Extensions
    // "prefix " is 7 bytes.
    let_verify(auto m = str("prefix αβγδ").utf8_uncased_search("ΑΒΓΔ"), m.offset == 7 && m.length == 8);
    // " suffix" is 7 bytes. "αβγδ" is 8 bytes.
    let_verify(auto m = str("αβγδ suffix").utf8_uncased_search("ΑΒΓΔ"), m.offset == 0 && m.length == 8);
    let_verify(auto m = str("prefix αβγδ suffix").utf8_uncased_search("ΑΒΓΔ"), m.offset == 7 && m.length == 8);

    // Sigma: 'Σ' (U+03A3, CE A3) matches both the medial 'σ' (U+03C3, CF 83) and the final
    // 'ς' (U+03C2, CF 82).
    // Haystack: "ΟΔΥΣΣΕΥΣ", Odysseus in uppercase.
    // Needle: "οδυσσευς", lowercase with the final sigma.
    // Lengths match byte-for-byte in this case.
    let_verify(auto m = str("ΟΔΥΣΣΕΥΣ").utf8_uncased_search("οδυσσευς"),
               m.offset == 0 && m.length == 16); // 8 chars * 2 bytes

    // Micro Sign 'µ' (U+00B5) vs Greek Mu 'μ' (U+03BC) vs 'Μ' (U+039C)
    // These should all fold to the same canonical representation.
    let_verify(auto m = str("µ").utf8_uncased_search("μ"), m.offset == 0 && m.length == 2);
    let_verify(auto m = str("μ").utf8_uncased_search("µ"), m.offset == 0 && m.length == 2);
    let_verify(auto m = str("µ").utf8_uncased_search("Μ"), m.offset == 0 && m.length == 2);
    let_verify(auto m = str("Μ").utf8_uncased_search("µ"), m.offset == 0 && m.length == 2);
    // Context: Head/Tail/Middle
    let_verify(auto m = str("123µ456").utf8_uncased_search("123μ456"), m.offset == 0 && m.length == 8);
    let_verify(auto m = str("LongPrefix Μ Suffix").utf8_uncased_search("Prefix µ Suf"),
               m.offset == 4 && m.length == 13);

    // Greek Lunate Epsilon 'ϵ' (U+03F5) → 'ε' (U+03B5)
    let_verify(auto m = str("ϵ").utf8_uncased_search("ε"), m.offset == 0 && m.length == 2);
    let_verify(auto m = str("start ϵ end").utf8_uncased_search("start ε end"), m.offset == 0 && m.length == 12);
    let_verify(auto m = str("...ϵ...").utf8_uncased_search(".ε."), m.offset == 2 && m.length == 4);
    // Greek Kappa Symbol 'ϰ' (U+03F0) → 'κ' (U+03BA)
    let_verify(auto m = str("ϰ").utf8_uncased_search("κ"), m.offset == 0 && m.length == 2);
    let_verify(auto m = str("text ϰ").utf8_uncased_search("text κ"), m.offset == 0 && m.length == 7); // 5 + 2
    let_verify(auto m = str("ϰ text").utf8_uncased_search("κ text"), m.offset == 0 && m.length == 7);

    // Greek Symbols & Anomalies
    // 'ϐ' (CF 90) → 'β' (CE B2)
    let_verify(auto m = str("ϐ").utf8_uncased_search("β"), m.offset == 0 && m.length == 2);
    let_verify(auto m = str("alpha ϐ").utf8_uncased_search("alpha β"), m.offset == 0 && m.length == 8);
    let_verify(auto m = str("ϐ beta").utf8_uncased_search("β beta"), m.offset == 0 && m.length == 7);
    // 'ϑ' (CF 91) → 'θ' (CE B8)
    let_verify(auto m = str("ϑ").utf8_uncased_search("θ"), m.offset == 0 && m.length == 2);
    let_verify(auto m = str("1ϑ2").utf8_uncased_search("1θ2"), m.offset == 0 && m.length == 4);
    let_verify(auto m = str("prefix ϑ suffix").utf8_uncased_search("fix θ suf"), m.offset == 3 && m.length == 10);
    // 'ϖ' (CF 96) → 'π' (CF 80)
    let_verify(auto m = str("ϖ").utf8_uncased_search("π"), m.offset == 0 && m.length == 2);
    let_verify(auto m = str("AϖB").utf8_uncased_search("AπB"), m.offset == 0 && m.length == 4);
    let_verify(auto m = str("Long string with ϖ in it").utf8_uncased_search("th π in"),
               m.offset == 14 && m.length == 8);

    // Greek Context Extensions (Symbols)
    let_verify(auto m = str("alpha ϖ omega").utf8_uncased_search("π"), m.offset == 6 && m.length == 2);

    // Dialytika with Tonos 'ΐ' (CE 90) → Identity check mostly
    verify(str("ΐ").utf8_uncased_order("ΐ") == sz_equal_k);

    // Greek in Mixed Scripts (boundary checks)
    let_verify(auto m = str("ABCαβγ").utf8_uncased_search("abcΑΒΓ"),
               m.offset == 0 && m.length == 9); // 3 + 3*2 bytes

    // Cyrillic
    // Basic: "привет" vs "ПРИВЕТ"
    verify(str("привет").utf8_uncased_order("ПРИВЕТ") == sz_equal_k);
    let_verify(auto m = str("привет мир").utf8_uncased_search("ПРИВЕТ"),
               m.offset == 0 && m.length == 12); // 6 chars * 2 bytes

    // Cyrillic Context Extensions
    // "Check " is 6 bytes.
    let_verify(auto m = str("Check привет").utf8_uncased_search("ПРИВЕТ"), m.offset == 6 && m.length == 12);
    let_verify(auto m = str("привет check").utf8_uncased_search("ПРИВЕТ"), m.offset == 0 && m.length == 12);

    // Palochka 'Ӏ' (U+04C0, D3 80) → 'ӏ' (U+04CF, D3 8F)
    // Used in Caucasian languages. Case agnostic.
    let_verify(auto m = str("Ӏ").utf8_uncased_search("ӏ"), m.offset == 0 && m.length == 2);
    let_verify(auto m = str("ӏ").utf8_uncased_search("Ӏ"), m.offset == 0 && m.length == 2);

    // Ukrainian Ґ (U+0490) → ґ (U+0491)
    let_verify(auto m = str("Ґ").utf8_uncased_search("ґ"), m.offset == 0 && m.length == 2);

    // Mixed Cyrillic
    let_verify(auto m = str("Москва is beautiful").utf8_uncased_search("МОСКВА"),
               m.offset == 0 && m.length == 12); // 6 chars * 2

    // Turkish:
    // Dotted 'İ' (U+0130, C4 B0) → 'i' (ASCII) + combining dot (U+0307, CC 87).
    // "İstanbul" (starts with İ) vs "i̇stanbul" (starts with i + dot).
    // StringZilla finds canonical equivalence. 'İ' (2 bytes) matches 'i̇' (3 bytes).
    // The needle is "i" and a combining dot, and the haystack is 2 bytes of 'İ' and 7 of "stanbul".
    let_verify(auto m = str("İstanbul").utf8_uncased_search("i̇stanbul"), m.offset == 0 && m.length == 9);
    // A needle starting with the combining dot, mid-expansion of 'İ', still anchors to 'İ'.
    let_verify(auto m = str("İstanbul").utf8_uncased_search("\xCC\x87stanbul"), m.offset == 0 && m.length == 9);

    // Turkish Context Extensions
    // "Welcome to " is 11 bytes.
    let_verify(auto m = str("Welcome to İstanbul").utf8_uncased_search("i̇stanbul"), m.offset == 11 && m.length == 9);
    let_verify(auto m = str("Welcome to İstanbul").utf8_uncased_search("\xCC\x87stanbul"),
               m.offset == 11 && m.length == 9);
    // "İstanbul city"
    let_verify(auto m = str("İstanbul city").utf8_uncased_search("i̇stanbul"), m.offset == 0 && m.length == 9);

    // Undotted 'ı' (U+0131):
    // Typically 'I' (ASCII) folds to 'i' (ASCII).
    // 'ı' folds to... itself? Or 'I' if we are in Turkish mode?
    // Default fold often treats 'ı' as distinct from 'i'.
    // 'I' → 'i'. 'ı' → 'ı'. So 'I' != 'ı'.
    let_verify(auto m = str("I").utf8_uncased_search("ı"), m.offset == str::npos);

    // Turkish Ğ (U+011E) → ğ (U+011F) and Ş (U+015E) → ş (U+015F)
    let_verify(auto m = str("ĞŞ").utf8_uncased_search("ğş"), m.offset == 0 && m.length == 4);

    // Armenian
    // Ligature: 'և' (U+0587, D6 87) → 'ե' (U+0565, D5 A5) + 'ւ' (U+0582, D6 82)
    // Haystack: "և" (2 bytes). Needle: "եւ" (2 + 2 = 4 bytes).
    // Match should return haystack slice (2 bytes).
    let_verify(auto m = str("և").utf8_uncased_search("եւ"), m.offset == 0 && m.length == 2);

    // Armenian Context Extensions
    let_verify(auto m = str("abcև").utf8_uncased_search("եւ"), m.offset == 3 && m.length == 2);
    let_verify(auto m = str("ևabc").utf8_uncased_search("եւ"), m.offset == 0 && m.length == 2);
    // Reverse: Haystack "եւ" (4 bytes). Needle "և" (2 bytes).
    // Match should return haystack slice (4 bytes).
    let_verify(auto m = str("եւ").utf8_uncased_search("և"), m.offset == 0 && m.length == 4);

    // Armenian Context Extensions Reverse
    let_verify(auto m = str("abcեւ").utf8_uncased_search("և"), m.offset == 3 && m.length == 4);

    // Ligature: 'ﬓ' (U+FB13 Men-Now) → 'մ' (U+0574) + 'ն' (U+0576)
    // Haystack 3 bytes (EF AC 93). Needle 4 bytes (D5 B4 D5 B6).
    let_verify(auto m = str("ﬓ").utf8_uncased_search("մն"), m.offset == 0 && m.length == 3);
    let_verify(auto m = str("abcﬓdef").utf8_uncased_search("մն"), m.offset == 3 && m.length == 3);
    let_verify(auto m = str("ﬓ start").utf8_uncased_search("մն start"), m.offset == 0 && m.length == 9);

    // Ligature: 'ﬔ' (U+FB14 Men-Ech) → 'մ' (U+0574) + 'ե' (U+0565)
    let_verify(auto m = str("ﬔ").utf8_uncased_search("մե"), m.offset == 0 && m.length == 3);
    let_verify(auto m = str("Some ﬔ text").utf8_uncased_search("մե"), m.offset == 5 && m.length == 3);
    let_verify(auto m = str("End ﬔ").utf8_uncased_search("End մե"), m.offset == 0 && m.length == 7);

    // Ligature: 'ﬕ' (U+FB15 Men-Ini) → 'մ' (U+0574) + 'ի' (U+056B)
    let_verify(auto m = str("ﬕ").utf8_uncased_search("մի"), m.offset == 0 && m.length == 3);
    let_verify(auto m = str("123 ﬕ 456").utf8_uncased_search("123 մի 456"), m.offset == 0 && m.length == 11);
    let_verify(auto m = str("prefixﬕ").utf8_uncased_search("մի"), m.offset == 6 && m.length == 3);

    // Ligature: 'ﬖ' (U+FB16 Vew-Now) → 'վ' (U+057E) + 'ն' (U+0576)
    let_verify(auto m = str("ﬖ").utf8_uncased_search("վն"), m.offset == 0 && m.length == 3);
    let_verify(auto m = str("Test ﬖ Case").utf8_uncased_search("Test վն Case"), m.offset == 0 && m.length == 13);
    let_verify(auto m = str("ﬖ").utf8_uncased_search("վն"),
               m.offset == 0 && m.length == 3); // Redundant but safe

    // Ligature: 'ﬗ' (U+FB17 Men-Xeh) → 'մ' (U+0574) + 'խ' (U+056D)
    let_verify(auto m = str("ﬗ").utf8_uncased_search("մխ"), m.offset == 0 && m.length == 3);
    let_verify(auto m = str("Mid ﬗ dle").utf8_uncased_search("մխ"), m.offset == 4 && m.length == 3);
    let_verify(auto m = str("Start ﬗ").utf8_uncased_search("Start մխ"), m.offset == 0 && m.length == 9);

    // Vietnamese / Latin Extended Additional:
    // 'Ạ' (U+1EA0, E1 BA A0) → 'ạ' (U+1EA1, E1 BA A1)
    let_verify(auto m = str("Ạ").utf8_uncased_search("ạ"), m.offset == 0 && m.length == 3);
    let_verify(auto m = str("Word Ạ End").utf8_uncased_search("Word ạ End"), m.offset == 0 && m.length == 12);
    let_verify(auto m = str("PrefixẠ").utf8_uncased_search("ạ"), m.offset == 6 && m.length == 3);

    // 'Ấ' (U+1EA4, E1 BA A4) → 'ấ' (U+1EA5, E1 BA A5)
    let_verify(auto m = str("Ấ").utf8_uncased_search("ấ"), m.offset == 0 && m.length == 3);
    let_verify(auto m = str("Ấ Start").utf8_uncased_search("ấ Start"), m.offset == 0 && m.length == 9);
    let_verify(auto m = str("Mid Ấ dle").utf8_uncased_search("Mid ấ dle"), m.offset == 0 && m.length == 11);

    // Horn letters: Ơ (U+01A0, C6 A0) → ơ (U+01A1, C6 A1), Ư (U+01AF, C6 AF) → ư (U+01B0, C6 B0)
    let_verify(auto m = str("ƠƯ").utf8_uncased_search("ơư"), m.offset == 0 && m.length == 4);
    let_verify(auto m = str("Big ƠƯ Horns").utf8_uncased_search("Big ơư Horns"), m.offset == 0 && m.length == 14);
    let_verify(auto m = str("Prefix ƠƯ").utf8_uncased_search("ơư"), m.offset == 7 && m.length == 4);

    // Latin Extended Additional: Ḁ (U+1E80, E1 BA 80) → ḁ (U+1E81, E1 BA 81)
    let_verify(auto m = str("Ḁ").utf8_uncased_search("ḁ"), m.offset == 0 && m.length == 3);
    let_verify(auto m = str("Code Ḁ").utf8_uncased_search("Code ḁ"), m.offset == 0 && m.length == 8);
    let_verify(auto m = str("StartḀ").utf8_uncased_search("Startḁ"), m.offset == 0 && m.length == 8);

    // Vietnamese Context Extensions
    let_verify(auto m = str("xin chào Ḁ").utf8_uncased_search("ḁ"), m.offset == 10 && m.length == 3);

    // Special symbols (Latin):
    // Kelvin Sign U+212A (E2 84 AA) folds to 'k' (1 byte); the match spans the 3-byte source rune.
    let_verify(auto m = str("273 \xE2\x84\xAA").utf8_uncased_search("273 k"), m.offset == 0 && m.length == 7);

    // Reverse: haystack "273 k" (5 bytes), needle "273 " + Kelvin U+212A.
    let_verify(auto m = str("273 k").utf8_uncased_search("273 \xE2\x84\xAA"), m.offset == 0 && m.length == 5);

    // Angstrom Sign U+212B (E2 84 AB) folds to 'a' with ring U+00E5 (C3 A5).
    let_verify(auto m = str("\xE2\x84\xAB").utf8_uncased_search("\xC3\xA5"), m.offset == 0 && m.length == 3);
    let_verify(auto m = str("\xE2\x84\xAB").utf8_uncased_search("\xC3\x85"), m.offset == 0 && m.length == 3);

    // Context Extensions (Special Symbols)
    let_verify(auto m = str("Heat: 273 \xE2\x84\xAA").utf8_uncased_search("k"), m.offset == 10 && m.length == 3);
    let_verify(auto m = str("Unit: \xE2\x84\xAB").utf8_uncased_search("\xC3\xA5"), m.offset == 6 && m.length == 3);

    // Long S 'ſ' (U+017F) → 's':
    // "Messer" vs "Meſſer".
    // Haystack "Meſſer": M(1) e(1) ſ(2) ſ(2) e(1) r(1) = 8 bytes.
    // Needle "MESSER": 6 bytes.
    let_verify(auto m = str("Meſſer").utf8_uncased_search("MESSER"), m.offset == 0 && m.length == 8);
    let_verify(auto m = str("Ein Meſſer").utf8_uncased_search("MESSER"), m.offset == 4 && m.length == 8);
    let_verify(auto m = str("Meſſer block").utf8_uncased_search("MESSER"), m.offset == 0 && m.length == 8);

    // Ligature 'ﬅ' (U+FB05 "st") → "st":
    // Haystack "ﬅ" (3 bytes). Needle "st" (2 bytes).
    let_verify(auto m = str("ﬅ").utf8_uncased_search("st"), m.offset == 0 && m.length == 3);
    let_verify(auto m = str("Test ﬅ").utf8_uncased_search("Test st"), m.offset == 0 && m.length == 8);
    let_verify(auto m = str("ﬅart").utf8_uncased_search("start"), m.offset == 0 && m.length == 6);

    // Ligature 'ﬆ' (U+FB06, EF AC 86) → "st"
    let_verify(auto m = str("ﬆ").utf8_uncased_search("st"), m.offset == 0 && m.length == 3);
    let_verify(auto m = str("My ﬆyle").utf8_uncased_search("My style"), m.offset == 0 && m.length == 9);
    let_verify(auto m = str("Faﬆ").utf8_uncased_search("Fast"), m.offset == 0 && m.length == 5);

    // Extended ligature contexts:
    // "Messer" vs "Meſſer" ('ſ' is U+017F, C5 BF)
    let_verify(auto m = str("Das Meſſer schneidet").utf8_uncased_search("MESSER"),
               m.offset == 4 && m.length == 8); // "Das " (4) + "Meſſer" (8) = 12, start at 4
    let_verify(auto m = str("Meſſer").utf8_uncased_search("MESSER"), m.offset == 0 && m.length == 8);
    let_verify(auto m = str("Großes Meſſer").utf8_uncased_search("MESSER"),
               m.offset == 8 && m.length == 8); // "Großes " is 8 bytes, as 'ß' is 2

    // 'ﬅ' (U+FB05, EF AC 85)
    let_verify(auto m = str("Ligature ﬅ check").utf8_uncased_search("st"), m.offset == 9 && m.length == 3);
    let_verify(auto m = str("end with ﬅ").utf8_uncased_search("st"), m.offset == 9 && m.length == 3);

    // More complex ligatures
    let_verify(auto m = str("ﬃJaCä").utf8_uncased_search("fija"), m.offset == 0 && m.length == 5);
    let_verify(auto m = str("ﬃJaCä").utf8_uncased_search("ﬁja"), m.offset == 0 && m.length == 5);
    let_verify(auto m = str("alﬃJaCä").utf8_uncased_search("fija"), m.offset == 2 && m.length == 5);
    let_verify(auto m = str("alﬃJaCä").utf8_uncased_search("ﬁja"), m.offset == 2 && m.length == 5);

    // Mid-expansion matches inside a single ligature: we still report the source rune span.
    // 'ﬃ' (EF AC 83) folds to "ffi", so "fi" occurs starting at index 1.
    let_verify(auto m = str("ﬃ").utf8_uncased_search("fi"), m.offset == 0 && m.length == 3);
    // 'ﬄ' (EF AC 84) folds to "ffl", so "fl" occurs starting at index 1.
    let_verify(auto m = str("ﬄ").utf8_uncased_search("fl"), m.offset == 0 && m.length == 3);

    // Combining diacritical marks: ǰ (U+01F0) folds to 'j' + combining caron (U+030C)
    // Needle starts with combining caron - can match mid-expansion of ǰ
    let_verify(auto m = str("ǰ0").utf8_uncased_search("\xCC\x8C" "0"),       // Caron + '0', split to end the escape
               m.offset == 0 && m.length == 3);                              // Match entire ǰ0 (2 byte ǰ + 1 byte 0)
    let_verify(auto m = str("abcǰ0def").utf8_uncased_search("\xCC\x8C" "0"), // Split to end the escape
               m.offset == 3 && m.length == 3);                              // "abc" = 3 bytes

    // Mid-expansion matches with ß (U+00DF) → "ss":
    // Needle "sfoxeepmº" should match "ßfoxeEPMº", folded to "ssfoxeepmº", at folded position 1.
    // Return position is byte 0 where ß starts (first contributing character)
    let_verify(auto m = str("ßfoxeEPMº").utf8_uncased_search("sfoxeepmº"),
               m.offset == 0 && m.length == 11); // Entire haystack

    // 'ﬆ' (U+FB06, EF AC 86)
    let_verify(auto m = str("Big ﬆ").utf8_uncased_search("st"), m.offset == 4 && m.length == 3);

    // Georgian Mtavruli (upper) folds to Mkhedruli (lower), both 3 bytes in UTF-8:
    // 'Ა' (U+1C90, E1 B2 90) → 'ა' (U+10D0, E1 83 90).
    let_verify(auto m = str("Text Ა").utf8_uncased_search("ა"), m.offset == 5 && m.length == 3);

    // Cherokee:
    // Cherokee Supplement (Lower, U+AB70, EA AD B0, 'ꭰ') → Cherokee (Upper, U+13A0, E1 8E A0, 'Ꭰ')
    // Both 3 bytes.
    let_verify(auto m = str("ꭰ").utf8_uncased_search("Ꭰ"), m.offset == 0 && m.length == 3);

    // Cherokee Context
    let_verify(auto m = str("Syllable ꭰ").utf8_uncased_search("Ꭰ"), m.offset == 9 && m.length == 3);

    // Coptic (Extended):
    // Coptic Ⲡ (U+2C80, E2 B2 80) → ⲡ (U+2C81, E2 B2 81)
    let_verify(auto m = str("Ⲡ").utf8_uncased_search("ⲡ"), m.offset == 0 && m.length == 3);

    // Glagolitic:
    // Ⰰ (U+2C00, E2 B0 80) → ⰰ (U+2C30, E2 B0 B0)
    let_verify(auto m = str("Ⰰ").utf8_uncased_search("ⰰ"), m.offset == 0 && m.length == 3);

    // Glagolitic Context
    let_verify(auto m = str("Letter Ⰰ").utf8_uncased_search("ⰰ"), m.offset == 7 && m.length == 3);

    // Caseless scripts (CJK, Arabic, Hebrew, Emoji):
    // These generally don't fold, so they must match exactly or effectively be uncased by identity.

    // Arabic "Salam"
    verify(str("السلام").utf8_uncased_order("السلام") == sz_equal_k);

    // Hebrew "Shalom"
    verify(str("שלום").utf8_uncased_order("שלום") == sz_equal_k);

    // Numbers & Punctuation
    let_verify(auto m = str("12345!@#$%").utf8_uncased_search("345"), m.offset == 2 && m.length == 3);

    // Negative tests:
    // Not found in Cyrillic
    let_verify(auto m = str("Привет").utf8_uncased_search("xyz"), m.offset == str::npos);
    // Not found Cyrillic in ASCII
    let_verify(auto m = str("Hello World").utf8_uncased_search("При"), m.offset == str::npos);

    // CJK "Chinese"
    let_verify(auto m = str("中文测试").utf8_uncased_search("中文"), m.offset == 0 && m.length == 6);

    // Emoji
    let_verify(auto m = str("😀😁😂").utf8_uncased_search("😁"), m.offset == 4 && m.length == 4);

    // Emoji Context
    let_verify(auto m = str("smile 😀😁😂").utf8_uncased_search("😁"), m.offset == 10 && m.length == 4);

    // "Fuzz Regression": Needle "nԱԲՐԵշ" (Mixed case Armenian + ASCII)
    let_verify(auto m = str("nԱԲՐԵշ").utf8_uncased_search("nաբրեշ"), m.offset == 0 && m.length == 11);

    // Complex SIMD Regression Trigger
    // Needle includes: ǰ (Latin B), ẞ (Sharp S), Turkish ı, Emoji
    std::string complex_haystack =
        "\x66\x6F\x78\x74\xD0\xB2\x58\x77\x58\x20\x67\x31\x5A\xEF\xAC\x82\x46\x21\xC3\xA0\x31\x21\xC6\xA0\xEF\xAC" //
        "\x85\x57\x6F\x72\x6C\x64\xC4\x91\xE4\xB8\xAD\xE6\x96\x87\x43\xCF\x83\xE3\x81\x82\xE3\x81\x84\xD4\xB2\xD4" //
        "\xB1\xD5\x90\xD4\xB5\xD5\x8E\xC4\xB1\x6E\x32\xE4\xB8\xAD\xE6\x96\x87\x42\x30\x6E\xC3\x9F\x55\xCE\xBA\xCF" //
        "\x8C\xCF\x83\xCE\xBC\x30\x62\x72\x6F\x77\x6E\xCF\x83\x67\x66\x6F\x78\x21\xC2\xB5\x4D\xE4\xB8\xAD\xE6\x96" //
        "\x87\xC7\xB0\xE1\xBB\x86\xC4\xB0\x6A\x75\x6D\x70\x73\xC7\xB0\xC3\xA9\x6D\xC3\xB6\xC4\xB1\xF0\x9F\x98\x80" //
        "\x3F\xC4\xB1\xE1\xBA\x9E\x74\x68\x65\xC3\xB1\x45\x7A\xC3\xBC\x49\x74\x68\x65\x61\xC5\xBF\xC3\x80\xC3\x85" //
        "\xD0\x91\xC5\xBF\x4C\x20\xC4\xB0\xCE\x91\x2C\x67\xE1\xBA\x96\xC3\xA0\x77\xC3\x91\x4D\x52\xE1\xBA\xA1\x4A" //
        "\xC6\xA0\xEF\xAC\x85\xE1\xBA\x9E\xF0\x9F\x98\x80\xEF\xAC\x80\xD0\xB1\xCF\x82\x65\x4B\x7A\xC3\xB1\x65\xC3" //
        "\x9C\x64\xC3\xB1\x55\xD0\xB0\xC3\xA4\x67\x41\x7A\xE1\xBB\x87\x5A\x4A\x71\x76\xC3\x89\xC6\xA0\x45\xCE\x91" //
        "\x66\x67\x6F\x41\xC3\x85\x4F\x6B\x58\xC3\xB1\x52\xE1\xBA\x98\xE1\xBA\xA1\x63\x47\xC2\xAA\xD4\xB2\xD4\xB1" //
        "\xD5\x90\xD4\xB5\xD5\x8E\xC3\x89\x77\x31\x46\xCF\x82\x76\xCE\xA3\x56\x56\xCA\xBE\xE1\xBA\x96\xD0\x91\x6F" //
        "\xCE\x92\x6A\x75\x6D\x70\x73\x33\xE1\xBA\xA1\x6A\x75\x6D\x70\x73\xE1\xBA\x98\xC3\x9F\xC3\x9C\xC6\xA1\x59" //
        "\xEF\xAC\x86\x59\x56\x2E\x33\xC3\xA9\x7A\x4C\x4C";

    std::string complex_needle =
        "\x6D\x70\x73\xC7\xB0\xC3\xA9\x6D\xC3\xB6\xC4\xB1\xF0\x9F\x98\x80\x3F\xC4\xB1\xE1\xBA\x9E\x74\x68\x65\xC3" //
        "\xB1\x45\x7A\xC3\xBC\x49\x74\x68\x65";

    let_verify(auto m = str(complex_haystack).utf8_uncased_search(complex_needle), m.length != 0);

    // Cross-Script Mixed Needles (Regression tests for kernel selection issues)

    // Capital Eszett (U+1E9E, E1 BA 9E) - folds to "ss"
    // Single Capital Eszett
    let_verify(auto m = str("\xE1\xBA\x9E").utf8_uncased_search("ss"), m.offset == 0 && m.length == 3);
    let_verify(auto m = str("ss").utf8_uncased_search("\xE1\xBA\x9E"), m.offset == 0 && m.length == 2);

    // Capital Eszett vs lowercase ß (C3 9F)
    let_verify(auto m = str("\xE1\xBA\x9E").utf8_uncased_search("\xC3\x9F"), m.offset == 0 && m.length == 3);
    let_verify(auto m = str("\xC3\x9F").utf8_uncased_search("\xE1\xBA\x9E"), m.offset == 0 && m.length == 2);

    // Double Capital Eszett
    let_verify(auto m = str("\xE1\xBA\x9E\xE1\xBA\x9E").utf8_uncased_search("ssss"), m.offset == 0 && m.length == 6);

    // Capital Eszett at boundaries
    let_verify(auto m = str("prefixẞsuffix").utf8_uncased_search("xss"),
               m.offset == 5 && m.length == 4); // 'x'(1) + ẞ(3) = 4

    // Capital Eszett + Vietnamese (Western + Vietnamese kernels):
    // ẞ (E1 BA 9E) + ệ (E1 BB 87) - the exact failing pattern from fuzz tests
    let_verify(auto m = str("testẞệend").utf8_uncased_search("ss\xE1\xBB\x86"),
               m.offset == 4 && m.length == 6); // ẞ(3) + ệ(3) searched as ss + Ệ

    // Micro Sign + Greek (Western + Greek kernels):
    // µ (C2 B5) surrounded by Greek α (CE B1) and β (CE B2)
    let_verify(auto m = str("\xCE\xB1\xC2\xB5\xCE\xB2").utf8_uncased_search("\xCE\xB1\xCE\xBC\xCE\xB2"),
               m.offset == 0 && m.length == 6); // αµβ vs αμβ

    // Long S (C5 BF) + non-ASCII context
    let_verify(auto m = str("meſſage").utf8_uncased_search("MESSAGE"),
               m.offset == 0 && m.length == 9); // meſſage (9 bytes)

    // One-to-Many Expansions (U+1E96-1E9A range):
    // h with line below (U+1E96, E1 BA 96) → h + combining line below (CC B1)
    let_verify(auto m = str("\xE1\xBA\x96").utf8_uncased_search("h\xCC\xB1"), m.offset == 0 && m.length == 3);

    // t with diaeresis (U+1E97, E1 BA 97) → t + combining diaeresis (CC 88)
    let_verify(auto m = str("\xE1\xBA\x97").utf8_uncased_search("t\xCC\x88"), m.offset == 0 && m.length == 3);

    // w with ring above (U+1E98, E1 BA 98) → w + combining ring above (CC 8A)
    let_verify(auto m = str("\xE1\xBA\x98").utf8_uncased_search("w\xCC\x8A"), m.offset == 0 && m.length == 3);

    // y with ring above (U+1E99, E1 BA 99) → y + combining ring above (CC 8A)
    let_verify(auto m = str("\xE1\xBA\x99").utf8_uncased_search("y\xCC\x8A"), m.offset == 0 && m.length == 3);

    // Kelvin Sign (E2 84 AA) in mixed context
    let_verify(auto m = str("273 \xE2\x84\xAA test").utf8_uncased_search("273 k"),
               m.offset == 0 && m.length == 7); // K is 3 bytes

    // Angstrom Sign (E2 84 AB) with accented chars
    let_verify(auto m = str("10 \xE2\x84\xAB unit").utf8_uncased_search("10 \xC3\xA5"),
               m.offset == 0 && m.length == 6); // Å (3) vs å (2)

    // 64-byte Boundary Stress Tests

    // Capital Eszett at position 63 (just at SIMD boundary)
    {
        std::string prefix(63, 'x');
        let_verify(auto m = str((prefix + "ẞend").c_str()).utf8_uncased_search("xss"),
                   m.offset == 62 && m.length == 4); // last 'x' + ẞ(3)
    }

    // Vietnamese char at position 62
    {
        std::string prefix(62, 'a');
        let_verify(auto m = str((prefix + "ệb").c_str()).utf8_uncased_search("ỆB"),
                   m.offset == 62 && m.length == 4); // ệ(3) + b(1)
    }

    // Micro Sign at position 64 (just past SIMD boundary)
    {
        std::string prefix(64, 'z');
        let_verify(auto m = str((prefix + "µtest").c_str()).utf8_uncased_search("μ"),
                   m.offset == 64 && m.length == 2); // µ matches μ
    }

    // 'ﬄ' at position 63 (just at SIMD boundary), matching from inside its fold.
    {
        std::string prefix(63, 'x');
        let_verify(auto m = str((prefix + "ﬄend").c_str()).utf8_uncased_search("fl"),
                   m.offset == 63 && m.length == 3); // consume whole ligature
    }

    // ASCII + ligature spanning the SIMD boundary: 'P' at 62 and 'ﬄ' at 63.
    {
        std::string prefix(62, 'x');
        let_verify(auto m = str((prefix + "Pﬄend").c_str()).utf8_uncased_search("pf"),
                   m.offset == 62 && m.length == 4); // "P"(1) + "ﬄ"(3)
    }

    // Basic ASCII search
    let_verify(auto m = str("Hello World").utf8_uncased_search("WORLD"), m.offset == 6 && m.length == 5);
    let_verify(auto m = str("Hello World").utf8_uncased_search("world"), m.offset == 6 && m.length == 5);
    let_verify(auto m = str("HELLO").utf8_uncased_search("hello"), m.offset == 0 && m.length == 5);
    let_verify(auto m = str("Hello").utf8_uncased_search("xyz"), m.offset == str::npos);
    let_verify(auto m = str("Hello").utf8_uncased_search(""), m.offset == 0 && m.length == 0);
}

/**
 *  @brief Minimized known-answer vectors pinning serial-vs-SIMD mismatches the find fuzzers found.
 *
 *  Each numbered pattern pins the smallest input reproducing a serial-vs-SIMD disagreement -
 *  ligature and Eszett expansions, one-to-many folds with combining marks, ring-buffer-length
 *  needles, and runs landing on a 64-byte block edge - so the fix stays nailed down cheaply.
 */
void test_utf8_uncased_regressions_unit() {
    using str = sz::string_view_t;

    // Fuzz-Discovered Regressions (Serial vs SIMD mismatches)

    // Pattern 0: Ligature tail-match in mixed-case context; pins a verify crash on this input.
    // Haystack: C3 96 45 47 76 C3 91 2C 50 EF AC 84 ... EF AC 82 70
    // Needle:   67 76 C3 B1 2C 70 66
    {
        let_verify(
            auto m = str("ÖEGv\xC3\x91,P\xEF\xAC\x84quickWorld\xEF\xAC\x82p").utf8_uncased_search("gv\xC3\xB1,pf"),
            m.offset == 3 && m.length == 9);
        let_verify(auto m = str("ÖEGv\xC3\x91,P\xEF\xAC\x84quickWorld\xEF\xAC\x82p").utf8_uncased_search("pf"),
                   m.offset == 8 && m.length == 4);
    }

    // Pattern 1: "st" + Latin-1 char (st ligature expansion issue?)
    // Needle: 73 74 C2 BA = "st" + º (masculine ordinal indicator)
    // Needle: 73 74 C3 B1 = "st" + ñ
    // Needle: 73 74 C3 A5 = "st" + å
    // Needle: 73 74 C3 A9 = "st" + é
    // Needle: 73 74 D5 A2 = "st" + Armenian բ
    // Needle: 73 74 CE B1 = "st" + Greek α.
    // These trigger kernel=2 (Central Europe) with safe_window issues
    {
        // "st" followed by º - should this match "st" ligature + º?
        let_verify(auto m = str("testﬅºend").utf8_uncased_search("st\xC2\xBA"),
                   m.offset == 4 && m.length == 5); // st ligature (3) + º (2)

        // "st" followed by ñ
        let_verify(auto m = str("testﬅñend").utf8_uncased_search("st\xC3\xB1"),
                   m.offset == 4 && m.length == 5); // st ligature (3) + ñ (2)

        // "st" followed by Greek α
        let_verify(auto m = str("prefixﬅαsuffix").utf8_uncased_search("st\xCE\xB1"),
                   m.offset == 6 && m.length == 5); // st ligature (3) + α (2)
    }

    // Pattern 2: "ss" + Latin-1/Greek (Eszett expansion)
    // Needle: 73 73 CE B1 = "ss" + Greek α
    // Needle: 73 73 C3 A5 = "ss" + å
    {
        // "ss" followed by Greek α - should match ß + α
        let_verify(auto m = str("testßαend").utf8_uncased_search("ss\xCE\xB1"),
                   m.offset == 4 && m.length == 4); // ß (2) + α (2)

        // "ss" followed by å
        let_verify(auto m = str("prefixßåsuffix").utf8_uncased_search("ss\xC3\xA5"),
                   m.offset == 6 && m.length == 4); // ß (2) + å (2)
    }

    // Pattern 3: ASCII + combining diacritical + other char
    // Needle: 68 CC B1 D5 A5 = "h" + combining macron below + Armenian ե
    // Needle: 77 CC 8A CE B2 = "w" + combining ring above + Greek β
    // Needle: 6A CC 8C D5 A2 = "j" + combining caron + Armenian բ.
    // These test one-to-many expansions (U+1E96 range) mixed with other scripts
    {
        // h + combining macron below should match ẖ (U+1E96)
        let_verify(auto m = str("\xE1\xBA\x96\xD5\xA5").utf8_uncased_search("h\xCC\xB1\xD5\xA5"),
                   m.offset == 0 && m.length == 5); // ẖ (3) + ե (2)
        // Needle starts with the combining mark (mid-expansion of ẖ).
        let_verify(auto m = str("\xE1\xBA\x96\xD5\xA5").utf8_uncased_search("\xCC\xB1\xD5\xA5"),
                   m.offset == 0 && m.length == 5);

        // w + combining ring above should match ẘ (U+1E98)
        let_verify(auto m = str("\xE1\xBA\x98\xCE\xB2").utf8_uncased_search("w\xCC\x8A\xCE\xB2"),
                   m.offset == 0 && m.length == 5); // ẘ (3) + β (2)
        // Needle starts with the combining mark (mid-expansion of ẘ).
        let_verify(auto m = str("\xE1\xBA\x98\xCE\xB2").utf8_uncased_search("\xCC\x8A\xCE\xB2"),
                   m.offset == 0 && m.length == 5);

        // j + combining caron should match ǰ (U+01F0)
        let_verify(auto m = str("\xC7\xB0\xD5\xA2").utf8_uncased_search("j\xCC\x8C\xD5\xA2"),
                   m.offset == 0 && m.length == 4); // ǰ (2) + բ (2)
        // Needle starts with the combining mark (mid-expansion of ǰ).
        let_verify(auto m = str("\xC7\xB0\xD5\xA2").utf8_uncased_search("\xCC\x8C\xD5\xA2"),
                   m.offset == 0 && m.length == 4);
    }

    // Pattern 4: Modifier letters + other chars
    // Needle: CA BC 6E CE BC = modifier apostrophe + "n" + Greek μ
    // Needle: 61 CA BE D5 A5 = "a" + modifier right half ring + Armenian ե.
    // These test n-apostrophe (U+0149) and a-right-half-ring (U+1E9A) expansions
    {
        // 'n (U+0149) expands to modifier apostrophe + n
        let_verify(auto m = str("\xC5\x89\xCE\xBC").utf8_uncased_search("\xCA\xBCn\xCE\xBC"),
                   m.offset == 0 && m.length == 4); // ʼn (2) + μ (2)
        // Needle starts at the second rune of the expansion ("n..."), so it still anchors to 'ŉ'.
        let_verify(auto m = str("\xC5\x89\xCE\xBC").utf8_uncased_search("n\xCE\xBC"), m.offset == 0 && m.length == 4);

        // a + modifier right half ring should match ẚ (U+1E9A)
        let_verify(auto m = str("\xE1\xBA\x9A\xD5\xA5").utf8_uncased_search("a\xCA\xBE\xD5\xA5"),
                   m.offset == 0 && m.length == 5); // ẚ (3) + ե (2)
        // Needle starts at the second rune of the expansion ("ʾ..."), so it still anchors to 'ẚ'.
        let_verify(auto m = str("\xE1\xBA\x9A\xD5\xA5").utf8_uncased_search("\xCA\xBE\xD5\xA5"),
                   m.offset == 0 && m.length == 5);
    }

    // Pattern 5: Armenian + combining chars / ligatures
    // Needle: D5 A5 D6 82 CE B2 = Armenian ech+yiwn + Greek β
    {
        // Armenian ech+yiwn characters followed by Greek
        let_verify(auto m = str("\xD5\xA5\xD6\x82\xCE\xB2").utf8_uncased_search("\xD5\xA5\xD6\x82\xCE\xB2"),
                   m.offset == 0 && m.length == 6);
    }

    // Pattern 6: Long complex needles crossing multiple scripts.
    // These stress test the kernel selection and danger zone handling
    {
        // Armenian barev + Latin ligatures + Vietnamese
        std::string haystack = "\xD5\xA2\xD5\xA1\xD6\x80\xD5\xA5\xD5\xBE" // barev
                               "\xEF\xAC\x83"                             // ffi ligature
                               "\xE1\xBB\x87";                            // Vietnamese ệ
        std::string needle = "\xD5\xA2\xD5\xA1\xD6\x80\xD5\xA5\xD5\xBE"   // barev
                             "ffi"                                        // expanded
                             "\xE1\xBB\x86";                              // Vietnamese Ệ
        let_verify(auto m = str(haystack).utf8_uncased_search(needle), m.offset == 0 && m.length == 16);
    }

    // Long needle tests at ring buffer boundary (32 folded runes):
    // The serial implementation uses a 32-rune ring buffer for fold comparisons
    {
        // Exactly 32 ASCII characters (32 folded runes)
        std::string hay32(32, 'a');
        let_verify(auto m = str(hay32 + "xyz").utf8_uncased_search(hay32), m.offset == 0 && m.length == 32);

        // 33 ASCII characters (crosses ring buffer boundary)
        std::string hay33(33, 'a');
        let_verify(auto m = str(hay33 + "xyz").utf8_uncased_search(hay33), m.offset == 0 && m.length == 33);

        // 16 eszett characters → 32 folded runes (ss × 16), exactly at boundary
        std::string hay_16_ss(16, '\xC3');
        for (size_t i = 0; i < 16; ++i) hay_16_ss.insert(i * 2 + 1, 1, '\x9F'); // Build "ßßßßßßßßßßßßßßßß"
        std::string needle_32_s(32, 's');
        let_verify(auto m = str(hay_16_ss + "end").utf8_uncased_search(needle_32_s), m.offset == 0 && m.length == 32);

        // 64 ASCII characters (tests double boundary)
        std::string hay64(64, 'b');
        let_verify(auto m = str(hay64 + "xyz").utf8_uncased_search(hay64), m.offset == 0 && m.length == 64);
    }

    // Eszett at SIMD 64-byte chunk boundaries
    {
        // ß at position 62 (ends exactly at 64-byte boundary)
        std::string prefix62(62, 'a');
        let_verify(auto m = str(prefix62 + "\xC3\x9F" + "xyz").utf8_uncased_search("ss"),
                   m.offset == 62 && m.length == 2);

        // ß straddling 64-byte boundary (starts at 63)
        std::string prefix63(63, 'a');
        let_verify(auto m = str(prefix63 + "\xC3\x9F" + "xyz").utf8_uncased_search("ss"),
                   m.offset == 63 && m.length == 2);

        // ß exactly at 64-byte boundary
        std::string prefix64(64, 'a');
        let_verify(auto m = str(prefix64 + "\xC3\x9F" + "xyz").utf8_uncased_search("ss"),
                   m.offset == 64 && m.length == 2);

        // Word with ß crossing boundary: "straße" starting at position 60
        std::string prefix60(60, 'a');
        let_verify(auto m = str(prefix60 + "straßezzz").utf8_uncased_search("strasse"),
                   m.offset == 60 && m.length == 7);
    }

    // Cross-script boundary tests (different SIMD kernels)
    {
        // ASCII → Greek transition at SIMD boundary
        std::string ascii60(60, 'x');
        let_verify(auto m = str(ascii60 + "\xCE\xB1\xCE\xB2\xCE\xB3").utf8_uncased_search("\xCE\x91\xCE\x92\xCE\x93"),
                   m.offset == 60 && m.length == 6); // ΑΒΓ matching αβγ

        // Latin-1 → Cyrillic transition
        std::string latin58(58, '\xC3');                                      // Build Latin-1 prefix
        for (size_t i = 0; i < 58; ++i) latin58.insert(i * 2 + 1, 1, '\xA4'); // "äääää..."
        // This creates 116-byte prefix of ä characters
    }

    // Minimal divergence cases (Ice Lake vs serial):
    // Minimal inputs pinning Serial/SIMD agreement on expansion boundaries that generic fuzzing
    // rarely hits: a fold that grows the needle mid-match, straddling a kernel's block edge.

    // Pattern 7: "sss" prefix matching "Sß" - a triple-s expansion from the Eszett fold.
    // Haystack: "brown Sßà jumps" - bytes at [6]: 53 C3 9F C3 A0 ("Sßà") = 5 bytes
    // Needle: "sssà" - bytes: 73 73 73 C3 A0 = 5 bytes.
    // 'S' → 's', 'ß' → "ss", 'à' → 'à', so "Sßà" → "sssà" (should match!)
    {
        // Simple case: "Sßà" should match "sssà" - match is 5 bytes (53 C3 9F C3 A0)
        let_verify(auto m = str("brown S\xC3\x9F\xC3\xA0 jumps").utf8_uncased_search("sss\xC3\xA0"),
                   m.offset == 6 && m.length == 5);

        // Lowercase: "sßà" should match "sssà" - match is 5 bytes
        let_verify(auto m = str("brown s\xC3\x9F\xC3\xA0 jumps").utf8_uncased_search("sss\xC3\xA0"),
                   m.offset == 6 && m.length == 5);

        // Uppercase ß (U+1E9E) when it exists - "ẞà" should match "ssà"
        let_verify(auto m = str("brown \xE1\xBA\x9E\xC3\xA0 jumps").utf8_uncased_search("ss\xC3\xA0"),
                   m.offset == 6 && m.length == 5);

        // Triple-s with space: "sß " should match "sss ".
        // Match starts at byte 7 where 's' is (byte 6 is space before 's')
        let_verify(auto m = str("ǰbee3 s\xC3\x9F ee\xC3\xA9 nc").utf8_uncased_search("sss ee\xC3\xA9"),
                   m.offset == 7 && m.length == 8);

        // "ss" needle vs "ß" haystack (basic case)
        let_verify(auto m = str("ßabc").utf8_uncased_search("ssabc"), m.offset == 0 && m.length == 5);

        // "sss" needle vs "sß" haystack
        let_verify(auto m = str("sßabc").utf8_uncased_search("sssabc"), m.offset == 0 && m.length == 6);
    }

    // Pattern 8: Greek Mu UTF-8 boundary.
    // Needle: CE BC (Greek μ - U+03BC).
    // Pins that a match only lands on a valid UTF-8 character boundary, never on the mid-byte
    // BC that also appears as the trailing byte of an unrelated codepoint such as ¼.
    {
        // Simple Greek mu search
        let_verify(auto m = str("hello \xCE\xBC world").utf8_uncased_search("\xCE\xBC"),
                   m.offset == 6 && m.length == 2);

        // Greek mu not at position where 0xBC appears as second byte of another char
        // Create haystack with Latin-1 char ending in 0xBC, then Greek mu.
        // This ensures we only match at valid UTF-8 boundaries
        let_verify(auto m = str("test \xC2\xBC thing \xCE\xBC end").utf8_uncased_search("\xCE\xBC"),
                   m.offset == 14 && m.length == 2); // Only at actual μ, not at ¼

        // Multiple Greek chars around mu
        let_verify(auto m = str("\xCE\xB1\xCE\xBC\xCE\xB2").utf8_uncased_search("\xCE\xBC"),
                   m.offset == 2 && m.length == 2);
    }

    // Pattern 9: Cyrillic Moscow case folding.
    // Haystack: "се Москва" (uppercase М - D0 9C)
    // Needle: "се москва" (lowercase м - D0 BC)
    // Should match uncasedly
    {
        // Simple Moscow: Москва vs москва
        let_verify(auto m = str("\xD0\x9C\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0")
                                .utf8_uncased_search("\xD0\xBC\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0"),
                   m.offset == 0 && m.length == 12);

        // Moscow with Latin prefix
        let_verify(auto m = str("se \xD0\x9C\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0")
                                .utf8_uncased_search("se \xD0\xBC\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0"),
                   m.offset == 0 && m.length == 15);

        // All Cyrillic uppercase vs lowercase
        let_verify(auto m = str("\xD0\x90\xD0\x91\xD0\x92")                       // АБВ
                                .utf8_uncased_search("\xD0\xB0\xD0\xB1\xD0\xB2"), // абв
                   m.offset == 0 && m.length == 6);

        // Mixed: ПРИВЕТ vs привет
        let_verify(auto m = str("\xD0\x9F\xD0\xA0\xD0\x98\xD0\x92\xD0\x95\xD0\xA2")                       // ПРИВЕТ
                                .utf8_uncased_search("\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82"), // привет
                   m.offset == 0 && m.length == 12);
    }

    // Pattern 10: Ligature fi expansion.
    // Haystack contains ﬁ (EF AC 81 - U+FB01)
    // Needle has "fi" (66 69).
    // ﬁ should case-fold to "fi"
    {
        // Simple: ﬁ vs fi
        let_verify(auto m = str("\xEF\xAC\x81nd").utf8_uncased_search("find"), m.offset == 0 && m.length == 5);

        // With uppercase: ﬁ vs FI
        let_verify(auto m = str("\xEF\xAC\x81nd").utf8_uncased_search("FInd"), m.offset == 0 && m.length == 5);

        // ff ligature: ﬀ (EF AC 80) vs ff
        let_verify(auto m = str("\xEF\xAC\x80oo").utf8_uncased_search("ffoo"), m.offset == 0 && m.length == 5);

        // ffi ligature: ﬃ (EF AC 83) vs ffi
        let_verify(auto m = str("ﬃce").utf8_uncased_search("ffice"), m.offset == 0 && m.length == 5);

        // fl ligature: ﬂ (EF AC 82) vs fl
        let_verify(auto m = str("\xEF\xAC\x82oor").utf8_uncased_search("floor"), m.offset == 0 && m.length == 6);

        // ffl ligature: ﬄ (EF AC 84) vs ffl
        let_verify(auto m = str("waﬄe").utf8_uncased_search("waffle"), m.offset == 0 && m.length == 6);
    }

    // Pattern 11: Combining marks vs precomposed.
    // j + combining caron (6A CC 8C) vs ǰ (C7 B0 - U+01F0)
    // These are canonically equivalent in Unicode
    // Note: StringZilla may or may not perform normalization - document behavior
    {
        // Precomposed ǰ vs decomposed j+caron:
        // If normalization is performed, these should match.
        // If not, they won't match (current behavior TBD)
        let_verify(auto m = str("\xC7\xB0ump").utf8_uncased_search("j\xCC\x8Cump"), m.offset == 0 && m.length == 5);

        // é precomposed (C3 A9) vs e+acute (65 CC 81)
        let_verify(auto m = str("\xC3\xA9lan").utf8_uncased_search("e\xCC\x81lan"), m.offset == str::npos);
    }

    // Pattern 12: Mixed script verification.
    // These test that case folding works correctly when multiple scripts are mixed
    {
        // Greek κόσμ mixed with Latin
        let_verify(auto m = str("brown \xCE\xBA\xCF\x8C\xCF\x83 end").utf8_uncased_search("\xCE\xBA\xCF\x8C\xCF\x83"),
                   m.offset == 6 && m.length == 6);

        // Greek sigma case: Σ (CE A3) vs σ (CF 83) vs ς (CF 82 - final sigma)
        let_verify(auto m = str("\xCE\xA3\xCE\xB5").utf8_uncased_search("\xCF\x83\xCE\xB5"),
                   m.offset == 0 && m.length == 4);

        // Armenian + Latin mixed
        let_verify(auto m = str("test \xD5\xA2\xD5\xA1\xD6\x80\xD5\xA5\xD5\xBE world")
                                .utf8_uncased_search("\xD5\xA2\xD5\xA1\xD6\x80\xD5\xA5\xD5\xBE"),
                   m.offset == 5 && m.length == 10);

        // Armenian ligature: և (D6 87 - U+0587) vs ե+ւ (D5 A5 D6 82)
        let_verify(auto m = str("\xD6\x87nd").utf8_uncased_search("\xD5\xA5\xD6\x82nd"),
                   m.offset == 0 && m.length == 4);
    }
}

#pragma endregion Unit

#pragma region Drivers

/** Adversarial invalid-input safety probe through the dispatch points. */
void test_utf8_uncased_safety(test_context_t &context) { check_utf8_uncased_safety_(context, utf8_uncased_dispatched); }

/** The dispatch points against serial: the case-fold differential and the full find battery. */
void test_utf8_uncased_all(test_context_t &context) {
    check_utf8_uncased_equivalence_(context, utf8_uncased_dispatched);
}

#pragma endregion Drivers
