/**
 *  @file test/utf8_uncased.cpp
 *  @author Ash Vardanian
 *  @date November 23, 2025
 *  @brief Uncased UTF-8 folding, search and order through the dispatch points and the C++ wrappers.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#undef STRINGZILLA_DEBUG
#define STRINGZILLA_DEBUG 1 // ! Enforce aggressive logging in this translation unit

/*  Make sure to include the StringZilla headers before anything else, to intercept missing
 *  `#include` directives and other issues. */
#include <stringzilla/stringzilla.h>   // Primary C API
#include <stringzilla/stringzilla.hpp> // C++ string class replacement

#include <string> // Baseline
#include <vector> // `std::vector`

#include "cross.hpp" // `check_utf8_uncased_unit_`, `check_utf8_uncased_equivalence_`, `check_utf8_uncased_safety_`

namespace ashvardanian::stringzilla::test {

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
        {
            auto match = sz::utf8_uncased_search(str("Hello World"), "world");
            expect_true(match.offset == 6 && match.length == 5);
        }
    }
    {
        {
            auto match = sz::utf8_uncased_search(str("ßfox"), "SS");
            expect_true(match.offset == 0 && match.length == 2);
        }
    }

    // `utf8_uncased_matches` yields every non-overlapping case-insensitive match, the twin of
    // Rust/Python `Utf8UncasedMatches`. Folding can change a match's byte length, so each yielded
    // slice spans the matched bytes, not the needle's length.
    {
        expect_equal(sz::utf8_uncased_matches(str("Ab aB AB xyz ab"), "ab").template to<std::vector<std::string>>(),
                     std::vector<std::string> {"Ab", "aB", "AB", "ab"});
        expect_true(sz::utf8_uncased_matches(str("xyz"), "ab").empty());
        // An empty needle matches at codepoint boundaries, so 3 times in the 3 bytes of "aé".
        expect_equal(sz::utf8_uncased_matches(str("a\xC3\xA9"), "").size(), 3);
        // A truncated rune is one step, capped at the end, where byte steps give 4.
        {
            str const truncated("a\xE4\xB8");
            expect_equal(offsets_within(truncated, sz::utf8_uncased_matches(truncated, "")),
                         std::vector<std::ptrdiff_t> {0, 1, 3});
        }

        // 'ß' (U+00DF, 2 bytes) folds to "ss": each one is a length-2 match of the needle "SS".
        expect_equal(sz::utf8_uncased_matches(str("ßox ß"), "SS").template to<std::vector<std::string>>(),
                     std::vector<std::string> {"ß", "ß"});
    }

    // In-place fold on a mutable `sz::string_t`.
    {
        sz::string_t folded;
        expect_true(sz::succeeded(folded.assign("HeLLo")));
        expect_true(sz::succeeded(sz::utf8_uncased_fold(folded)));
        expect_equal(folded, "hello");
    }
    {
        sz::string_t folded;
        expect_true(sz::succeeded(folded.assign("\xC3\x9F")));
        expect_true(sz::succeeded(sz::utf8_uncased_fold(folded)));
        expect_equal(folded, "ss");
    }

    expect_equal(sz::utf8_uncased_order(str("Hello"), "HELLO"), sz_equal_k);
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
    expect_equal(sz::utf8_uncased_order(str("hello"), "HELLO"), sz_equal_k);
    expect_equal(sz::utf8_uncased_order(str("abc"), "ABC"), sz_equal_k);
    expect_equal(sz::utf8_uncased_order(str("HeLLo WoRLd"), "hello world"), sz_equal_k);

    // ASCII Extensions
    {
        auto m = sz::utf8_uncased_search(str("prefixhello"), "HELLO");
        expect_true(m.offset == 6 && m.length == 5);
    }
    {
        auto m = sz::utf8_uncased_search(str("hello_suffix"), "HELLO");
        expect_true(m.offset == 0 && m.length == 5);
    }
    {
        auto m = sz::utf8_uncased_search(str("mid_hello_mid"), "HELLO");
        expect_true(m.offset == 4 && m.length == 5);
    }

    // Less than
    expect_equal(sz::utf8_uncased_order(str("abc"), "abd"), sz_less_k);
    expect_equal(sz::utf8_uncased_order(str("ab"), "abc"), sz_less_k);
    expect_equal(sz::utf8_uncased_order(str("ABC"), "abd"), sz_less_k);

    // Greater than
    expect_equal(sz::utf8_uncased_order(str("abd"), "abc"), sz_greater_k);
    expect_equal(sz::utf8_uncased_order(str("abcd"), "abc"), sz_greater_k);
    expect_equal(sz::utf8_uncased_order(str("ABD"), "abc"), sz_greater_k);

    // Latin-1 Supplement and Latin Extended-A, starting with German umlauts
    expect_equal(sz::utf8_uncased_order(str("schöner"), "SCHÖNER"), sz_equal_k);
    {
        auto m = sz::utf8_uncased_search(str("Das ist ein schöner Tag"), "SCHÖNER");
        expect_true(m.offset == 12 && m.length == 8);
    } // 'ö' (U+00F6, C3 B6) is 2 bytes

    // French Accents
    expect_equal(sz::utf8_uncased_order(str("café"), "CAFÉ"), sz_equal_k);
    expect_equal(sz::utf8_uncased_order(str("naïve"), "NAÏVE"), sz_equal_k);
    expect_equal(sz::utf8_uncased_order(str("À la carte"), "à la CARTE"), sz_equal_k);

    // Spanish/Portuguese
    expect_equal(sz::utf8_uncased_order(str("niño"), "NIÑO"), sz_equal_k);

    // Polish / Central European (Latin Extended-A):
    // "ĄĆĘŁŃÓŚŹŻ" → "ąćęłńóśźż", and "Zaółć gęślą jaźń" is a classic Polish pangram fragment
    expect_equal(sz::utf8_uncased_order(str("Zaółć gęślą jaźń"), "ZAÓŁĆ GĘŚLĄ JAŹŃ"), sz_equal_k);

    // Czech characters: ř (U+0159, C5 99), ž (U+017E, C5 BE), č (U+010D, C4 8D), ě (U+011B, C4 9B)
    expect_equal(sz::utf8_uncased_order(str("řžčě"), "ŘŽČĚ"), sz_equal_k);
    {
        auto m = sz::utf8_uncased_search(str("Příklad"), "PŘÍKLAD");
        expect_true(m.offset == 0 && m.length == 9);
    }
    {
        auto m = sz::utf8_uncased_search(str("žena"), "ŽENA");
        expect_true(m.offset == 0 && m.length == 5);
    }

    // Polish ł (U+0142, C5 82) in city name
    expect_equal(sz::utf8_uncased_order(str("Łódź"), "ŁÓDŹ"), sz_equal_k);
    {
        auto m = sz::utf8_uncased_search(str("miasto Łódź"), "łódź");
        expect_true(m.offset == 7 && m.length == 7);
    }

    // Hungarian: ő (U+0151, C5 91), ű (U+0171, C5 B1)
    expect_equal(sz::utf8_uncased_order(str("őű"), "ŐŰ"), sz_equal_k);
    {
        auto m = sz::utf8_uncased_search(str("Erdő"), "ERDŐ");
        expect_true(m.offset == 0 && m.length == 5);
    }
    {
        auto m = sz::utf8_uncased_search(str("Győr"), "GYŐR");
        expect_true(m.offset == 0 && m.length == 5);
    }

    // Central European at SIMD boundary (64 bytes)
    {
        std::string prefix(62, 'a');
        {
            auto m = sz::utf8_uncased_search(str(prefix + "ž"), "Ž");
            expect_true(m.offset == 62 && m.length == 2);
        }
        {
            auto m = sz::utf8_uncased_search(str(prefix + "řž"), "ŘŽ");
            expect_true(m.offset == 62 && m.length == 4);
        }
    }

    // German (Eszett 'ß'):
    // 'ß' (U+00DF, C3 9F) → "ss"
    // "straße" → "strasse"
    // "STRASSE" → "strasse"
    expect_equal(sz::utf8_uncased_order(str("straße"), "STRASSE"), sz_equal_k);
    expect_equal(sz::utf8_uncased_order(str("STRASSE"), "straße"), sz_equal_k);

    // Uppercase 'ẞ' (U+1E9E, E1 BA 9E) → "ss" or "ß" depending on fold
    // StringZilla generally folds to lowercase first. 'ẞ' → 'ss'.
    // Haystack uses 'ß' (2 bytes), Needle "SS".
    {
        auto m = sz::utf8_uncased_search(str("straße"), "SS");
        expect_true(m.offset == 4 && m.length == 2);
    } // Matches 'ß' (2 bytes)

    // Eszett Context Extensions
    {
        auto m = sz::utf8_uncased_search(str("Eine straße"), "SS");
        expect_true(m.offset == 9 && m.length == 2);
    } // "Eine " is 5 chars → 5 bytes + "stra" (4) = 9
    {
        auto m = sz::utf8_uncased_search(str("straßebahn"), "SS");
        expect_true(m.offset == 4 && m.length == 2);
    }
    {
        auto m = sz::utf8_uncased_search(str("Eine straßebahn"), "SS");
        expect_true(m.offset == 9 && m.length == 2);
    }

    // Same case-folding, but different relation
    {
        auto m = sz::utf8_uncased_search(str("HelloäeßHelloL"), "helloäesshellol");
        expect_true(m.offset == 0 && m.length == 16);
    }
    {
        auto m = sz::utf8_uncased_search(str("helloäesshellol"), "HelloäeßHelloL");
        expect_true(m.offset == 0 && m.length == 16);
    }

    // Same case-folding, but a different relation and needle length, due to the uppercase
    // three-byte 'ẞ' (U+1E9E, E1 BA 9E)
    {
        auto m = sz::utf8_uncased_search(str("HelloäeẞHelloL"), "helloäesshellol");
        expect_true(m.offset == 0 && m.length == 17);
    }
    {
        auto m = sz::utf8_uncased_search(str("helloäesshellol"), "HelloäeẞHelloL");
        expect_true(m.offset == 0 && m.length == 16);
    }

    // Haystack "STRASSE", Needle "straße"
    {
        auto m = sz::utf8_uncased_search(str("STRASSE"), "straße");
        expect_true(m.offset == 0 && m.length == 7);
    } // Matches "STRASSE" (7 bytes)

    // "Maße" → "MASSE"
    {
        auto m = sz::utf8_uncased_search(str("Maße"), "MASSE");
        expect_true(m.offset == 0 && m.length == 5);
    } // Matches "Maße" (5 bytes)

    // Haystack: "Fuss" (4 bytes), ending in "u", "s", "s".
    // Needle: "Fuß" (4 bytes), ending in "u", "ß".
    // They are equal in order, and searching "Fuß" in "Fuss" works.
    {
        auto m = sz::utf8_uncased_search(str("Fuss"), "Fuß");
        expect_true(m.offset == 0 && m.length == 4);
    } // Matches "Fuss"

    // Mid-expansion matching: the needle starts with 's', which takes the serial fallback.
    // Haystack: "ßfox" (5 bytes) folds to "ssfox".
    // Needle "sfox" matches at folded position 1, but we report offset 0, the start of ß.
    // Length is 5 because we consume the entire ß character, never pointing to half of it.
    {
        auto m = sz::utf8_uncased_search(str("ßfox"), "sfox");
        expect_true(m.offset == 0 && m.length == 5);
    }

    // Needle ends with 's': the suffix case, which takes the serial fallback.
    // Haystack: "foxß" folds to "foxss".
    // Needle "foxs" matches through the first 's' of the expansion.
    {
        auto m = sz::utf8_uncased_search(str("fox\xC3\x9F"), "foxs");
        expect_true(m.offset == 0 && m.length == 5);
    }

    // Cross-boundary case: "ßS" folds to "sss", which takes the serial fallback.
    // Haystack: "ßStra" (6 bytes) folds to "ssstra".
    // Needle "sstra" starts with 's', and would match at position 1, mid-ß, without the rule.
    // Length is 6 because we consume the entire haystack, as ß expands but is consumed whole.
    {
        auto m = sz::utf8_uncased_search(str("ßStra"), "sstra");
        expect_true(m.offset == 0 && m.length == 6);
    }

    // Needle with 's' not at boundary - should use fast SIMD path
    {
        auto m = sz::utf8_uncased_search(str("teßt"), "tesst");
        expect_true(m.offset == 0 && m.length == 5);
    }
    {
        auto m = sz::utf8_uncased_search(str("maße"), "masse");
        expect_true(m.offset == 0 && m.length == 5);
    }

    // Needle with 'ss' at boundary - also uses serial (can't match across ß boundary)
    {
        auto m = sz::utf8_uncased_search(str("fo\xC3\x9F"), "foss");
        expect_true(m.offset == 0 && m.length == 4);
    }
    {
        auto m = sz::utf8_uncased_search(str("ßfo"), "ssfo");
        expect_true(m.offset == 0 && m.length == 4);
    }

    // Math symbols:
    // Multiplication × (U+00D7, C3 97) and Division ÷ (U+00F7, C3 B7).
    // Often confusable with 'x' and '+'/'=', but strictly they are distinct.
    // They should equal themselves but not each other.
    expect_equal(sz::utf8_uncased_order(str("×"), "×"), sz_equal_k);  // × == ×
    expect_equal(sz::utf8_uncased_order(str("÷"), "÷"), sz_equal_k);  // ÷ == ÷
    expect_true(sz::utf8_uncased_order(str("×"), "÷") != sz_equal_k); // × ≠ ÷
    expect_equal(sz::utf8_uncased_order(str("a×b"), "A×B"), sz_equal_k);

    // Math Context Extensions
    {
        auto m = sz::utf8_uncased_search(str("2×3=6"), "×");
        expect_true(m.offset == 1 && m.length == 2);
    }
    {
        auto m = sz::utf8_uncased_search(str("6÷2=3"), "÷");
        expect_true(m.offset == 1 && m.length == 2);
    }

    // Empty strings
    expect_equal(sz::utf8_uncased_order(str(""), ""), sz_equal_k);
    expect_equal(sz::utf8_uncased_order(str("a"), ""), sz_greater_k);
    expect_equal(sz::utf8_uncased_order(str(""), "a"), sz_less_k);

    // Greek
    // Basic casing: "αβγδ" vs "ΑΒΓΔ"
    expect_equal(sz::utf8_uncased_order(str("αβγδ"), "ΑΒΓΔ"), sz_equal_k);
    {
        auto m = sz::utf8_uncased_search(str("αβγδ"), "ΑΒΓΔ");
        expect_true(m.offset == 0 && m.length == 8);
    } // 4 * 2 bytes = 8 bytes

    // Greek Context Extensions
    // "prefix " is 7 bytes.
    {
        auto m = sz::utf8_uncased_search(str("prefix αβγδ"), "ΑΒΓΔ");
        expect_true(m.offset == 7 && m.length == 8);
    }
    // " suffix" is 7 bytes. "αβγδ" is 8 bytes.
    {
        auto m = sz::utf8_uncased_search(str("αβγδ suffix"), "ΑΒΓΔ");
        expect_true(m.offset == 0 && m.length == 8);
    }
    {
        auto m = sz::utf8_uncased_search(str("prefix αβγδ suffix"), "ΑΒΓΔ");
        expect_true(m.offset == 7 && m.length == 8);
    }

    // Sigma: 'Σ' (U+03A3, CE A3) matches both the medial 'σ' (U+03C3, CF 83) and the final
    // 'ς' (U+03C2, CF 82).
    // Haystack: "ΟΔΥΣΣΕΥΣ", Odysseus in uppercase.
    // Needle: "οδυσσευς", lowercase with the final sigma.
    // Lengths match byte-for-byte in this case.
    {
        auto m = sz::utf8_uncased_search(str("ΟΔΥΣΣΕΥΣ"), "οδυσσευς");
        expect_true(m.offset == 0 && m.length == 16);
    } // 8 chars * 2 bytes

    // Micro Sign 'µ' (U+00B5) vs Greek Mu 'μ' (U+03BC) vs 'Μ' (U+039C)
    // These should all fold to the same canonical representation.
    {
        auto m = sz::utf8_uncased_search(str("µ"), "μ");
        expect_true(m.offset == 0 && m.length == 2);
    }
    {
        auto m = sz::utf8_uncased_search(str("μ"), "µ");
        expect_true(m.offset == 0 && m.length == 2);
    }
    {
        auto m = sz::utf8_uncased_search(str("µ"), "Μ");
        expect_true(m.offset == 0 && m.length == 2);
    }
    {
        auto m = sz::utf8_uncased_search(str("Μ"), "µ");
        expect_true(m.offset == 0 && m.length == 2);
    }
    // Context: Head/Tail/Middle
    {
        auto m = sz::utf8_uncased_search(str("123µ456"), "123μ456");
        expect_true(m.offset == 0 && m.length == 8);
    }
    {
        auto m = sz::utf8_uncased_search(str("LongPrefix Μ Suffix"), "Prefix µ Suf");
        expect_true(m.offset == 4 && m.length == 13);
    }

    // Greek Lunate Epsilon 'ϵ' (U+03F5) → 'ε' (U+03B5)
    {
        auto m = sz::utf8_uncased_search(str("ϵ"), "ε");
        expect_true(m.offset == 0 && m.length == 2);
    }
    {
        auto m = sz::utf8_uncased_search(str("start ϵ end"), "start ε end");
        expect_true(m.offset == 0 && m.length == 12);
    }
    {
        auto m = sz::utf8_uncased_search(str("...ϵ..."), ".ε.");
        expect_true(m.offset == 2 && m.length == 4);
    }
    // Greek Kappa Symbol 'ϰ' (U+03F0) → 'κ' (U+03BA)
    {
        auto m = sz::utf8_uncased_search(str("ϰ"), "κ");
        expect_true(m.offset == 0 && m.length == 2);
    }
    {
        auto m = sz::utf8_uncased_search(str("text ϰ"), "text κ");
        expect_true(m.offset == 0 && m.length == 7);
    } // 5 + 2
    {
        auto m = sz::utf8_uncased_search(str("ϰ text"), "κ text");
        expect_true(m.offset == 0 && m.length == 7);
    }

    // Greek Symbols & Anomalies
    // 'ϐ' (CF 90) → 'β' (CE B2)
    {
        auto m = sz::utf8_uncased_search(str("ϐ"), "β");
        expect_true(m.offset == 0 && m.length == 2);
    }
    {
        auto m = sz::utf8_uncased_search(str("alpha ϐ"), "alpha β");
        expect_true(m.offset == 0 && m.length == 8);
    }
    {
        auto m = sz::utf8_uncased_search(str("ϐ beta"), "β beta");
        expect_true(m.offset == 0 && m.length == 7);
    }
    // 'ϑ' (CF 91) → 'θ' (CE B8)
    {
        auto m = sz::utf8_uncased_search(str("ϑ"), "θ");
        expect_true(m.offset == 0 && m.length == 2);
    }
    {
        auto m = sz::utf8_uncased_search(str("1ϑ2"), "1θ2");
        expect_true(m.offset == 0 && m.length == 4);
    }
    {
        auto m = sz::utf8_uncased_search(str("prefix ϑ suffix"), "fix θ suf");
        expect_true(m.offset == 3 && m.length == 10);
    }
    // 'ϖ' (CF 96) → 'π' (CF 80)
    {
        auto m = sz::utf8_uncased_search(str("ϖ"), "π");
        expect_true(m.offset == 0 && m.length == 2);
    }
    {
        auto m = sz::utf8_uncased_search(str("AϖB"), "AπB");
        expect_true(m.offset == 0 && m.length == 4);
    }
    {
        auto m = sz::utf8_uncased_search(str("Long string with ϖ in it"), "th π in");
        expect_true(m.offset == 14 && m.length == 8);
    }

    // Greek Context Extensions (Symbols)
    {
        auto m = sz::utf8_uncased_search(str("alpha ϖ omega"), "π");
        expect_true(m.offset == 6 && m.length == 2);
    }

    // Dialytika with Tonos 'ΐ' (CE 90) → Identity check mostly
    expect_equal(sz::utf8_uncased_order(str("ΐ"), "ΐ"), sz_equal_k);

    // Greek in Mixed Scripts (boundary checks)
    {
        auto m = sz::utf8_uncased_search(str("ABCαβγ"), "abcΑΒΓ");
        expect_true(m.offset == 0 && m.length == 9);
    } // 3 + 3*2 bytes

    // Cyrillic
    // Basic: "привет" vs "ПРИВЕТ"
    expect_equal(sz::utf8_uncased_order(str("привет"), "ПРИВЕТ"), sz_equal_k);
    {
        auto m = sz::utf8_uncased_search(str("привет мир"), "ПРИВЕТ");
        expect_true(m.offset == 0 && m.length == 12);
    } // 6 chars * 2 bytes

    // Cyrillic Context Extensions
    // "Check " is 6 bytes.
    {
        auto m = sz::utf8_uncased_search(str("Check привет"), "ПРИВЕТ");
        expect_true(m.offset == 6 && m.length == 12);
    }
    {
        auto m = sz::utf8_uncased_search(str("привет check"), "ПРИВЕТ");
        expect_true(m.offset == 0 && m.length == 12);
    }

    // Palochka 'Ӏ' (U+04C0, D3 80) → 'ӏ' (U+04CF, D3 8F)
    // Used in Caucasian languages. Case agnostic.
    {
        auto m = sz::utf8_uncased_search(str("Ӏ"), "ӏ");
        expect_true(m.offset == 0 && m.length == 2);
    }
    {
        auto m = sz::utf8_uncased_search(str("ӏ"), "Ӏ");
        expect_true(m.offset == 0 && m.length == 2);
    }

    // Ukrainian Ґ (U+0490) → ґ (U+0491)
    {
        auto m = sz::utf8_uncased_search(str("Ґ"), "ґ");
        expect_true(m.offset == 0 && m.length == 2);
    }

    // Mixed Cyrillic
    {
        auto m = sz::utf8_uncased_search(str("Москва is beautiful"), "МОСКВА");
        expect_true(m.offset == 0 && m.length == 12);
    } // 6 chars * 2

    // Turkish:
    // Dotted 'İ' (U+0130, C4 B0) → 'i' (ASCII) + combining dot (U+0307, CC 87).
    // "İstanbul" (starts with İ) vs "i̇stanbul" (starts with i + dot).
    // StringZilla finds canonical equivalence. 'İ' (2 bytes) matches 'i̇' (3 bytes).
    // The needle is "i" and a combining dot, and the haystack is 2 bytes of 'İ' and 7 of "stanbul".
    {
        auto m = sz::utf8_uncased_search(str("İstanbul"), "i̇stanbul");
        expect_true(m.offset == 0 && m.length == 9);
    }
    // A needle starting with the combining dot, mid-expansion of 'İ', still anchors to 'İ'.
    {
        auto m = sz::utf8_uncased_search(str("İstanbul"), "\xCC\x87stanbul");
        expect_true(m.offset == 0 && m.length == 9);
    }

    // Turkish Context Extensions
    // "Welcome to " is 11 bytes.
    {
        auto m = sz::utf8_uncased_search(str("Welcome to İstanbul"), "i̇stanbul");
        expect_true(m.offset == 11 && m.length == 9);
    }
    {
        auto m = sz::utf8_uncased_search(str("Welcome to İstanbul"), "\xCC\x87stanbul");
        expect_true(m.offset == 11 && m.length == 9);
    }
    // "İstanbul city"
    {
        auto m = sz::utf8_uncased_search(str("İstanbul city"), "i̇stanbul");
        expect_true(m.offset == 0 && m.length == 9);
    }

    // Undotted 'ı' (U+0131):
    // Typically 'I' (ASCII) folds to 'i' (ASCII).
    // 'ı' folds to... itself? Or 'I' if we are in Turkish mode?
    // Default fold often treats 'ı' as distinct from 'i'.
    // 'I' → 'i'. 'ı' → 'ı'. So 'I' != 'ı'.
    {
        auto m = sz::utf8_uncased_search(str("I"), "ı");
        expect_equal(m.offset, str::npos);
    }

    // Turkish Ğ (U+011E) → ğ (U+011F) and Ş (U+015E) → ş (U+015F)
    {
        auto m = sz::utf8_uncased_search(str("ĞŞ"), "ğş");
        expect_true(m.offset == 0 && m.length == 4);
    }

    // Armenian
    // Ligature: 'և' (U+0587, D6 87) → 'ե' (U+0565, D5 A5) + 'ւ' (U+0582, D6 82)
    // Haystack: "և" (2 bytes). Needle: "եւ" (2 + 2 = 4 bytes).
    // Match should return haystack slice (2 bytes).
    {
        auto m = sz::utf8_uncased_search(str("և"), "եւ");
        expect_true(m.offset == 0 && m.length == 2);
    }

    // Armenian Context Extensions
    {
        auto m = sz::utf8_uncased_search(str("abcև"), "եւ");
        expect_true(m.offset == 3 && m.length == 2);
    }
    {
        auto m = sz::utf8_uncased_search(str("ևabc"), "եւ");
        expect_true(m.offset == 0 && m.length == 2);
    }
    // Reverse: Haystack "եւ" (4 bytes). Needle "և" (2 bytes).
    // Match should return haystack slice (4 bytes).
    {
        auto m = sz::utf8_uncased_search(str("եւ"), "և");
        expect_true(m.offset == 0 && m.length == 4);
    }

    // Armenian Context Extensions Reverse
    {
        auto m = sz::utf8_uncased_search(str("abcեւ"), "և");
        expect_true(m.offset == 3 && m.length == 4);
    }

    // Ligature: 'ﬓ' (U+FB13 Men-Now) → 'մ' (U+0574) + 'ն' (U+0576)
    // Haystack 3 bytes (EF AC 93). Needle 4 bytes (D5 B4 D5 B6).
    {
        auto m = sz::utf8_uncased_search(str("ﬓ"), "մն");
        expect_true(m.offset == 0 && m.length == 3);
    }
    {
        auto m = sz::utf8_uncased_search(str("abcﬓdef"), "մն");
        expect_true(m.offset == 3 && m.length == 3);
    }
    {
        auto m = sz::utf8_uncased_search(str("ﬓ start"), "մն start");
        expect_true(m.offset == 0 && m.length == 9);
    }

    // Ligature: 'ﬔ' (U+FB14 Men-Ech) → 'մ' (U+0574) + 'ե' (U+0565)
    {
        auto m = sz::utf8_uncased_search(str("ﬔ"), "մե");
        expect_true(m.offset == 0 && m.length == 3);
    }
    {
        auto m = sz::utf8_uncased_search(str("Some ﬔ text"), "մե");
        expect_true(m.offset == 5 && m.length == 3);
    }
    {
        auto m = sz::utf8_uncased_search(str("End ﬔ"), "End մե");
        expect_true(m.offset == 0 && m.length == 7);
    }

    // Ligature: 'ﬕ' (U+FB15 Men-Ini) → 'մ' (U+0574) + 'ի' (U+056B)
    {
        auto m = sz::utf8_uncased_search(str("ﬕ"), "մի");
        expect_true(m.offset == 0 && m.length == 3);
    }
    {
        auto m = sz::utf8_uncased_search(str("123 ﬕ 456"), "123 մի 456");
        expect_true(m.offset == 0 && m.length == 11);
    }
    {
        auto m = sz::utf8_uncased_search(str("prefixﬕ"), "մի");
        expect_true(m.offset == 6 && m.length == 3);
    }

    // Ligature: 'ﬖ' (U+FB16 Vew-Now) → 'վ' (U+057E) + 'ն' (U+0576)
    {
        auto m = sz::utf8_uncased_search(str("ﬖ"), "վն");
        expect_true(m.offset == 0 && m.length == 3);
    }
    {
        auto m = sz::utf8_uncased_search(str("Test ﬖ Case"), "Test վն Case");
        expect_true(m.offset == 0 && m.length == 13);
    }
    {
        auto m = sz::utf8_uncased_search(str("ﬖ"), "վն");
        expect_true(m.offset == 0 && m.length == 3);
    } // Redundant but safe

    // Ligature: 'ﬗ' (U+FB17 Men-Xeh) → 'մ' (U+0574) + 'խ' (U+056D)
    {
        auto m = sz::utf8_uncased_search(str("ﬗ"), "մխ");
        expect_true(m.offset == 0 && m.length == 3);
    }
    {
        auto m = sz::utf8_uncased_search(str("Mid ﬗ dle"), "մխ");
        expect_true(m.offset == 4 && m.length == 3);
    }
    {
        auto m = sz::utf8_uncased_search(str("Start ﬗ"), "Start մխ");
        expect_true(m.offset == 0 && m.length == 9);
    }

    // Vietnamese / Latin Extended Additional:
    // 'Ạ' (U+1EA0, E1 BA A0) → 'ạ' (U+1EA1, E1 BA A1)
    {
        auto m = sz::utf8_uncased_search(str("Ạ"), "ạ");
        expect_true(m.offset == 0 && m.length == 3);
    }
    {
        auto m = sz::utf8_uncased_search(str("Word Ạ End"), "Word ạ End");
        expect_true(m.offset == 0 && m.length == 12);
    }
    {
        auto m = sz::utf8_uncased_search(str("PrefixẠ"), "ạ");
        expect_true(m.offset == 6 && m.length == 3);
    }

    // 'Ấ' (U+1EA4, E1 BA A4) → 'ấ' (U+1EA5, E1 BA A5)
    {
        auto m = sz::utf8_uncased_search(str("Ấ"), "ấ");
        expect_true(m.offset == 0 && m.length == 3);
    }
    {
        auto m = sz::utf8_uncased_search(str("Ấ Start"), "ấ Start");
        expect_true(m.offset == 0 && m.length == 9);
    }
    {
        auto m = sz::utf8_uncased_search(str("Mid Ấ dle"), "Mid ấ dle");
        expect_true(m.offset == 0 && m.length == 11);
    }

    // Horn letters: Ơ (U+01A0, C6 A0) → ơ (U+01A1, C6 A1), Ư (U+01AF, C6 AF) → ư (U+01B0, C6 B0)
    {
        auto m = sz::utf8_uncased_search(str("ƠƯ"), "ơư");
        expect_true(m.offset == 0 && m.length == 4);
    }
    {
        auto m = sz::utf8_uncased_search(str("Big ƠƯ Horns"), "Big ơư Horns");
        expect_true(m.offset == 0 && m.length == 14);
    }
    {
        auto m = sz::utf8_uncased_search(str("Prefix ƠƯ"), "ơư");
        expect_true(m.offset == 7 && m.length == 4);
    }

    // Latin Extended Additional: Ḁ (U+1E80, E1 BA 80) → ḁ (U+1E81, E1 BA 81)
    {
        auto m = sz::utf8_uncased_search(str("Ḁ"), "ḁ");
        expect_true(m.offset == 0 && m.length == 3);
    }
    {
        auto m = sz::utf8_uncased_search(str("Code Ḁ"), "Code ḁ");
        expect_true(m.offset == 0 && m.length == 8);
    }
    {
        auto m = sz::utf8_uncased_search(str("StartḀ"), "Startḁ");
        expect_true(m.offset == 0 && m.length == 8);
    }

    // Vietnamese Context Extensions
    {
        auto m = sz::utf8_uncased_search(str("xin chào Ḁ"), "ḁ");
        expect_true(m.offset == 10 && m.length == 3);
    }

    // Special symbols (Latin):
    // Kelvin Sign U+212A (E2 84 AA) folds to 'k' (1 byte); the match spans the 3-byte source rune.
    {
        auto m = sz::utf8_uncased_search(str("273 \xE2\x84\xAA"), "273 k");
        expect_true(m.offset == 0 && m.length == 7);
    }

    // Reverse: haystack "273 k" (5 bytes), needle "273 " + Kelvin U+212A.
    {
        auto m = sz::utf8_uncased_search(str("273 k"), "273 \xE2\x84\xAA");
        expect_true(m.offset == 0 && m.length == 5);
    }

    // Angstrom Sign U+212B (E2 84 AB) folds to 'a' with ring U+00E5 (C3 A5).
    {
        auto m = sz::utf8_uncased_search(str("\xE2\x84\xAB"), "\xC3\xA5");
        expect_true(m.offset == 0 && m.length == 3);
    }
    {
        auto m = sz::utf8_uncased_search(str("\xE2\x84\xAB"), "\xC3\x85");
        expect_true(m.offset == 0 && m.length == 3);
    }

    // Context Extensions (Special Symbols)
    {
        auto m = sz::utf8_uncased_search(str("Heat: 273 \xE2\x84\xAA"), "k");
        expect_true(m.offset == 10 && m.length == 3);
    }
    {
        auto m = sz::utf8_uncased_search(str("Unit: \xE2\x84\xAB"), "\xC3\xA5");
        expect_true(m.offset == 6 && m.length == 3);
    }

    // Long S 'ſ' (U+017F) → 's':
    // "Messer" vs "Meſſer".
    // Haystack "Meſſer": M(1) e(1) ſ(2) ſ(2) e(1) r(1) = 8 bytes.
    // Needle "MESSER": 6 bytes.
    {
        auto m = sz::utf8_uncased_search(str("Meſſer"), "MESSER");
        expect_true(m.offset == 0 && m.length == 8);
    }
    {
        auto m = sz::utf8_uncased_search(str("Ein Meſſer"), "MESSER");
        expect_true(m.offset == 4 && m.length == 8);
    }
    {
        auto m = sz::utf8_uncased_search(str("Meſſer block"), "MESSER");
        expect_true(m.offset == 0 && m.length == 8);
    }

    // Ligature 'ﬅ' (U+FB05 "st") → "st":
    // Haystack "ﬅ" (3 bytes). Needle "st" (2 bytes).
    {
        auto m = sz::utf8_uncased_search(str("ﬅ"), "st");
        expect_true(m.offset == 0 && m.length == 3);
    }
    {
        auto m = sz::utf8_uncased_search(str("Test ﬅ"), "Test st");
        expect_true(m.offset == 0 && m.length == 8);
    }
    {
        auto m = sz::utf8_uncased_search(str("ﬅart"), "start");
        expect_true(m.offset == 0 && m.length == 6);
    }

    // Ligature 'ﬆ' (U+FB06, EF AC 86) → "st"
    {
        auto m = sz::utf8_uncased_search(str("ﬆ"), "st");
        expect_true(m.offset == 0 && m.length == 3);
    }
    {
        auto m = sz::utf8_uncased_search(str("My ﬆyle"), "My style");
        expect_true(m.offset == 0 && m.length == 9);
    }
    {
        auto m = sz::utf8_uncased_search(str("Faﬆ"), "Fast");
        expect_true(m.offset == 0 && m.length == 5);
    }

    // Extended ligature contexts:
    // "Messer" vs "Meſſer" ('ſ' is U+017F, C5 BF)
    {
        auto m = sz::utf8_uncased_search(str("Das Meſſer schneidet"), "MESSER");
        expect_true(m.offset == 4 && m.length == 8);
    } // "Das " (4) + "Meſſer" (8) = 12, start at 4
    {
        auto m = sz::utf8_uncased_search(str("Meſſer"), "MESSER");
        expect_true(m.offset == 0 && m.length == 8);
    }
    {
        auto m = sz::utf8_uncased_search(str("Großes Meſſer"), "MESSER");
        expect_true(m.offset == 8 && m.length == 8);
    } // "Großes " is 8 bytes, as 'ß' is 2

    // 'ﬅ' (U+FB05, EF AC 85)
    {
        auto m = sz::utf8_uncased_search(str("Ligature ﬅ check"), "st");
        expect_true(m.offset == 9 && m.length == 3);
    }
    {
        auto m = sz::utf8_uncased_search(str("end with ﬅ"), "st");
        expect_true(m.offset == 9 && m.length == 3);
    }

    // More complex ligatures
    {
        auto m = sz::utf8_uncased_search(str("ﬃJaCä"), "fija");
        expect_true(m.offset == 0 && m.length == 5);
    }
    {
        auto m = sz::utf8_uncased_search(str("ﬃJaCä"), "ﬁja");
        expect_true(m.offset == 0 && m.length == 5);
    }
    {
        auto m = sz::utf8_uncased_search(str("alﬃJaCä"), "fija");
        expect_true(m.offset == 2 && m.length == 5);
    }
    {
        auto m = sz::utf8_uncased_search(str("alﬃJaCä"), "ﬁja");
        expect_true(m.offset == 2 && m.length == 5);
    }

    // Mid-expansion matches inside a single ligature: we still report the source rune span.
    // 'ﬃ' (EF AC 83) folds to "ffi", so "fi" occurs starting at index 1.
    {
        auto m = sz::utf8_uncased_search(str("ﬃ"), "fi");
        expect_true(m.offset == 0 && m.length == 3);
    }
    // 'ﬄ' (EF AC 84) folds to "ffl", so "fl" occurs starting at index 1.
    {
        auto m = sz::utf8_uncased_search(str("ﬄ"), "fl");
        expect_true(m.offset == 0 && m.length == 3);
    }

    // Combining diacritical marks: ǰ (U+01F0) folds to 'j' + combining caron (U+030C)
    // Needle starts with combining caron - can match mid-expansion of ǰ
    {
        auto m = sz::utf8_uncased_search(str("ǰ0"), "\xCC\x8C" "0"); // Caron + '0', split to end the escape
        expect_true(m.offset == 0 && m.length == 3);                 // Match entire ǰ0 (2 byte ǰ + 1 byte 0)
    }
    {
        auto m = sz::utf8_uncased_search(str("abcǰ0def"), "\xCC\x8C" "0"); // Split to end the escape
        expect_true(m.offset == 3 && m.length == 3);                       // "abc" = 3 bytes
    }

    // Mid-expansion matches with ß (U+00DF) → "ss":
    // Needle "sfoxeepmº" should match "ßfoxeEPMº", folded to "ssfoxeepmº", at folded position 1.
    // Return position is byte 0 where ß starts (first contributing character)
    {
        auto m = sz::utf8_uncased_search(str("ßfoxeEPMº"), "sfoxeepmº");
        expect_true(m.offset == 0 && m.length == 11);
    } // Entire haystack

    // 'ﬆ' (U+FB06, EF AC 86)
    {
        auto m = sz::utf8_uncased_search(str("Big ﬆ"), "st");
        expect_true(m.offset == 4 && m.length == 3);
    }

    // Georgian Mtavruli (upper) folds to Mkhedruli (lower), both 3 bytes in UTF-8:
    // 'Ა' (U+1C90, E1 B2 90) → 'ა' (U+10D0, E1 83 90).
    {
        auto m = sz::utf8_uncased_search(str("Text Ა"), "ა");
        expect_true(m.offset == 5 && m.length == 3);
    }

    // Cherokee:
    // Cherokee Supplement (Lower, U+AB70, EA AD B0, 'ꭰ') → Cherokee (Upper, U+13A0, E1 8E A0, 'Ꭰ')
    // Both 3 bytes.
    {
        auto m = sz::utf8_uncased_search(str("ꭰ"), "Ꭰ");
        expect_true(m.offset == 0 && m.length == 3);
    }

    // Cherokee Context
    {
        auto m = sz::utf8_uncased_search(str("Syllable ꭰ"), "Ꭰ");
        expect_true(m.offset == 9 && m.length == 3);
    }

    // Coptic (Extended):
    // Coptic Ⲡ (U+2C80, E2 B2 80) → ⲡ (U+2C81, E2 B2 81)
    {
        auto m = sz::utf8_uncased_search(str("Ⲡ"), "ⲡ");
        expect_true(m.offset == 0 && m.length == 3);
    }

    // Glagolitic:
    // Ⰰ (U+2C00, E2 B0 80) → ⰰ (U+2C30, E2 B0 B0)
    {
        auto m = sz::utf8_uncased_search(str("Ⰰ"), "ⰰ");
        expect_true(m.offset == 0 && m.length == 3);
    }

    // Glagolitic Context
    {
        auto m = sz::utf8_uncased_search(str("Letter Ⰰ"), "ⰰ");
        expect_true(m.offset == 7 && m.length == 3);
    }

    // Caseless scripts (CJK, Arabic, Hebrew, Emoji):
    // These generally don't fold, so they must match exactly or effectively be uncased by identity.

    // Arabic "Salam"
    expect_equal(sz::utf8_uncased_order(str("السلام"), "السلام"), sz_equal_k);

    // Hebrew "Shalom"
    expect_equal(sz::utf8_uncased_order(str("שלום"), "שלום"), sz_equal_k);

    // Numbers & Punctuation
    {
        auto m = sz::utf8_uncased_search(str("12345!@#$%"), "345");
        expect_true(m.offset == 2 && m.length == 3);
    }

    // Negative tests:
    // Not found in Cyrillic
    {
        auto m = sz::utf8_uncased_search(str("Привет"), "xyz");
        expect_equal(m.offset, str::npos);
    }
    // Not found Cyrillic in ASCII
    {
        auto m = sz::utf8_uncased_search(str("Hello World"), "При");
        expect_equal(m.offset, str::npos);
    }

    // CJK "Chinese"
    {
        auto m = sz::utf8_uncased_search(str("中文测试"), "中文");
        expect_true(m.offset == 0 && m.length == 6);
    }

    // Emoji
    {
        auto m = sz::utf8_uncased_search(str("😀😁😂"), "😁");
        expect_true(m.offset == 4 && m.length == 4);
    }

    // Emoji Context
    {
        auto m = sz::utf8_uncased_search(str("smile 😀😁😂"), "😁");
        expect_true(m.offset == 10 && m.length == 4);
    }

    // "Fuzz Regression": Needle "nԱԲՐԵշ" (Mixed case Armenian + ASCII)
    {
        auto m = sz::utf8_uncased_search(str("nԱԲՐԵշ"), "nաբրեշ");
        expect_true(m.offset == 0 && m.length == 11);
    }

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

    {
        auto m = sz::utf8_uncased_search(str(complex_haystack), complex_needle);
        expect_true(m.length != 0);
    }

    // Cross-Script Mixed Needles (Regression tests for kernel selection issues)

    // Capital Eszett (U+1E9E, E1 BA 9E) - folds to "ss"
    // Single Capital Eszett
    {
        auto m = sz::utf8_uncased_search(str("\xE1\xBA\x9E"), "ss");
        expect_true(m.offset == 0 && m.length == 3);
    }
    {
        auto m = sz::utf8_uncased_search(str("ss"), "\xE1\xBA\x9E");
        expect_true(m.offset == 0 && m.length == 2);
    }

    // Capital Eszett vs lowercase ß (C3 9F)
    {
        auto m = sz::utf8_uncased_search(str("\xE1\xBA\x9E"), "\xC3\x9F");
        expect_true(m.offset == 0 && m.length == 3);
    }
    {
        auto m = sz::utf8_uncased_search(str("\xC3\x9F"), "\xE1\xBA\x9E");
        expect_true(m.offset == 0 && m.length == 2);
    }

    // Double Capital Eszett
    {
        auto m = sz::utf8_uncased_search(str("\xE1\xBA\x9E\xE1\xBA\x9E"), "ssss");
        expect_true(m.offset == 0 && m.length == 6);
    }

    // Capital Eszett at boundaries
    {
        auto m = sz::utf8_uncased_search(str("prefixẞsuffix"), "xss");
        expect_true(m.offset == 5 && m.length == 4);
    } // 'x'(1) + ẞ(3) = 4

    // Capital Eszett + Vietnamese (Western + Vietnamese kernels):
    // ẞ (E1 BA 9E) + ệ (E1 BB 87) - the exact failing pattern from fuzz tests
    {
        auto m = sz::utf8_uncased_search(str("testẞệend"), "ss\xE1\xBB\x86");
        expect_true(m.offset == 4 && m.length == 6);
    } // ẞ(3) + ệ(3) searched as ss + Ệ

    // Micro Sign + Greek (Western + Greek kernels):
    // µ (C2 B5) surrounded by Greek α (CE B1) and β (CE B2)
    {
        auto m = sz::utf8_uncased_search(str("\xCE\xB1\xC2\xB5\xCE\xB2"), "\xCE\xB1\xCE\xBC\xCE\xB2");
        expect_true(m.offset == 0 && m.length == 6);
    } // αµβ vs αμβ

    // Long S (C5 BF) + non-ASCII context
    {
        auto m = sz::utf8_uncased_search(str("meſſage"), "MESSAGE");
        expect_true(m.offset == 0 && m.length == 9);
    } // meſſage (9 bytes)

    // One-to-Many Expansions (U+1E96-1E9A range):
    // h with line below (U+1E96, E1 BA 96) → h + combining line below (CC B1)
    {
        auto m = sz::utf8_uncased_search(str("\xE1\xBA\x96"), "h\xCC\xB1");
        expect_true(m.offset == 0 && m.length == 3);
    }

    // t with diaeresis (U+1E97, E1 BA 97) → t + combining diaeresis (CC 88)
    {
        auto m = sz::utf8_uncased_search(str("\xE1\xBA\x97"), "t\xCC\x88");
        expect_true(m.offset == 0 && m.length == 3);
    }

    // w with ring above (U+1E98, E1 BA 98) → w + combining ring above (CC 8A)
    {
        auto m = sz::utf8_uncased_search(str("\xE1\xBA\x98"), "w\xCC\x8A");
        expect_true(m.offset == 0 && m.length == 3);
    }

    // y with ring above (U+1E99, E1 BA 99) → y + combining ring above (CC 8A)
    {
        auto m = sz::utf8_uncased_search(str("\xE1\xBA\x99"), "y\xCC\x8A");
        expect_true(m.offset == 0 && m.length == 3);
    }

    // Kelvin Sign (E2 84 AA) in mixed context
    {
        auto m = sz::utf8_uncased_search(str("273 \xE2\x84\xAA test"), "273 k");
        expect_true(m.offset == 0 && m.length == 7);
    } // K is 3 bytes

    // Angstrom Sign (E2 84 AB) with accented chars
    {
        auto m = sz::utf8_uncased_search(str("10 \xE2\x84\xAB unit"), "10 \xC3\xA5");
        expect_true(m.offset == 0 && m.length == 6);
    } // Å (3) vs å (2)

    // 64-byte Boundary Stress Tests

    // Capital Eszett at position 63 (just at SIMD boundary)
    {
        std::string prefix(63, 'x');
        {
            auto m = sz::utf8_uncased_search(str((prefix + "ẞend").c_str()), "xss");
            expect_true(m.offset == 62 && m.length == 4);
        } // last 'x' + ẞ(3)
    }

    // Vietnamese char at position 62
    {
        std::string prefix(62, 'a');
        {
            auto m = sz::utf8_uncased_search(str((prefix + "ệb").c_str()), "ỆB");
            expect_true(m.offset == 62 && m.length == 4);
        } // ệ(3) + b(1)
    }

    // Micro Sign at position 64 (just past SIMD boundary)
    {
        std::string prefix(64, 'z');
        {
            auto m = sz::utf8_uncased_search(str((prefix + "µtest").c_str()), "μ");
            expect_true(m.offset == 64 && m.length == 2);
        } // µ matches μ
    }

    // 'ﬄ' at position 63 (just at SIMD boundary), matching from inside its fold.
    {
        std::string prefix(63, 'x');
        {
            auto m = sz::utf8_uncased_search(str((prefix + "ﬄend").c_str()), "fl");
            expect_true(m.offset == 63 && m.length == 3);
        } // consume whole ligature
    }

    // ASCII + ligature spanning the SIMD boundary: 'P' at 62 and 'ﬄ' at 63.
    {
        std::string prefix(62, 'x');
        {
            auto m = sz::utf8_uncased_search(str((prefix + "Pﬄend").c_str()), "pf");
            expect_true(m.offset == 62 && m.length == 4);
        } // "P"(1) + "ﬄ"(3)
    }

    // Basic ASCII search
    {
        auto m = sz::utf8_uncased_search(str("Hello World"), "WORLD");
        expect_true(m.offset == 6 && m.length == 5);
    }
    {
        auto m = sz::utf8_uncased_search(str("Hello World"), "world");
        expect_true(m.offset == 6 && m.length == 5);
    }
    {
        auto m = sz::utf8_uncased_search(str("HELLO"), "hello");
        expect_true(m.offset == 0 && m.length == 5);
    }
    {
        auto m = sz::utf8_uncased_search(str("Hello"), "xyz");
        expect_equal(m.offset, str::npos);
    }
    {
        auto m = sz::utf8_uncased_search(str("Hello"), "");
        expect_true(m.offset == 0 && m.length == 0);
    }
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
        {
            auto m = sz::utf8_uncased_search(str("ÖEGv\xC3\x91,P\xEF\xAC\x84quickWorld\xEF\xAC\x82p"), "gv\xC3\xB1,pf");
            expect_true(m.offset == 3 && m.length == 9);
        }
        {
            auto m = sz::utf8_uncased_search(str("ÖEGv\xC3\x91,P\xEF\xAC\x84quickWorld\xEF\xAC\x82p"), "pf");
            expect_true(m.offset == 8 && m.length == 4);
        }
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
        {
            auto m = sz::utf8_uncased_search(str("testﬅºend"), "st\xC2\xBA");
            expect_true(m.offset == 4 && m.length == 5);
        } // st ligature (3) + º (2)

        // "st" followed by ñ
        {
            auto m = sz::utf8_uncased_search(str("testﬅñend"), "st\xC3\xB1");
            expect_true(m.offset == 4 && m.length == 5);
        } // st ligature (3) + ñ (2)

        // "st" followed by Greek α
        {
            auto m = sz::utf8_uncased_search(str("prefixﬅαsuffix"), "st\xCE\xB1");
            expect_true(m.offset == 6 && m.length == 5);
        } // st ligature (3) + α (2)
    }

    // Pattern 2: "ss" + Latin-1/Greek (Eszett expansion)
    // Needle: 73 73 CE B1 = "ss" + Greek α
    // Needle: 73 73 C3 A5 = "ss" + å
    {
        // "ss" followed by Greek α - should match ß + α
        {
            auto m = sz::utf8_uncased_search(str("testßαend"), "ss\xCE\xB1");
            expect_true(m.offset == 4 && m.length == 4);
        } // ß (2) + α (2)

        // "ss" followed by å
        {
            auto m = sz::utf8_uncased_search(str("prefixßåsuffix"), "ss\xC3\xA5");
            expect_true(m.offset == 6 && m.length == 4);
        } // ß (2) + å (2)
    }

    // Pattern 3: ASCII + combining diacritical + other char
    // Needle: 68 CC B1 D5 A5 = "h" + combining macron below + Armenian ե
    // Needle: 77 CC 8A CE B2 = "w" + combining ring above + Greek β
    // Needle: 6A CC 8C D5 A2 = "j" + combining caron + Armenian բ.
    // These test one-to-many expansions (U+1E96 range) mixed with other scripts
    {
        // h + combining macron below should match ẖ (U+1E96)
        {
            auto m = sz::utf8_uncased_search(str("\xE1\xBA\x96\xD5\xA5"), "h\xCC\xB1\xD5\xA5");
            expect_true(m.offset == 0 && m.length == 5);
        } // ẖ (3) + ե (2)
        // Needle starts with the combining mark (mid-expansion of ẖ).
        {
            auto m = sz::utf8_uncased_search(str("\xE1\xBA\x96\xD5\xA5"), "\xCC\xB1\xD5\xA5");
            expect_true(m.offset == 0 && m.length == 5);
        }

        // w + combining ring above should match ẘ (U+1E98)
        {
            auto m = sz::utf8_uncased_search(str("\xE1\xBA\x98\xCE\xB2"), "w\xCC\x8A\xCE\xB2");
            expect_true(m.offset == 0 && m.length == 5);
        } // ẘ (3) + β (2)
        // Needle starts with the combining mark (mid-expansion of ẘ).
        {
            auto m = sz::utf8_uncased_search(str("\xE1\xBA\x98\xCE\xB2"), "\xCC\x8A\xCE\xB2");
            expect_true(m.offset == 0 && m.length == 5);
        }

        // j + combining caron should match ǰ (U+01F0)
        {
            auto m = sz::utf8_uncased_search(str("\xC7\xB0\xD5\xA2"), "j\xCC\x8C\xD5\xA2");
            expect_true(m.offset == 0 && m.length == 4);
        } // ǰ (2) + բ (2)
        // Needle starts with the combining mark (mid-expansion of ǰ).
        {
            auto m = sz::utf8_uncased_search(str("\xC7\xB0\xD5\xA2"), "\xCC\x8C\xD5\xA2");
            expect_true(m.offset == 0 && m.length == 4);
        }
    }

    // Pattern 4: Modifier letters + other chars
    // Needle: CA BC 6E CE BC = modifier apostrophe + "n" + Greek μ
    // Needle: 61 CA BE D5 A5 = "a" + modifier right half ring + Armenian ե.
    // These test n-apostrophe (U+0149) and a-right-half-ring (U+1E9A) expansions
    {
        // 'n (U+0149) expands to modifier apostrophe + n
        {
            auto m = sz::utf8_uncased_search(str("\xC5\x89\xCE\xBC"), "\xCA\xBCn\xCE\xBC");
            expect_true(m.offset == 0 && m.length == 4);
        } // ʼn (2) + μ (2)
        // Needle starts at the second rune of the expansion ("n..."), so it still anchors to 'ŉ'.
        {
            auto m = sz::utf8_uncased_search(str("\xC5\x89\xCE\xBC"), "n\xCE\xBC");
            expect_true(m.offset == 0 && m.length == 4);
        }

        // a + modifier right half ring should match ẚ (U+1E9A)
        {
            auto m = sz::utf8_uncased_search(str("\xE1\xBA\x9A\xD5\xA5"), "a\xCA\xBE\xD5\xA5");
            expect_true(m.offset == 0 && m.length == 5);
        } // ẚ (3) + ե (2)
        // Needle starts at the second rune of the expansion ("ʾ..."), so it still anchors to 'ẚ'.
        {
            auto m = sz::utf8_uncased_search(str("\xE1\xBA\x9A\xD5\xA5"), "\xCA\xBE\xD5\xA5");
            expect_true(m.offset == 0 && m.length == 5);
        }
    }

    // Pattern 5: Armenian + combining chars / ligatures
    // Needle: D5 A5 D6 82 CE B2 = Armenian ech+yiwn + Greek β
    {
        // Armenian ech+yiwn characters followed by Greek
        {
            auto m = sz::utf8_uncased_search(str("\xD5\xA5\xD6\x82\xCE\xB2"), "\xD5\xA5\xD6\x82\xCE\xB2");
            expect_true(m.offset == 0 && m.length == 6);
        }
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
        {
            auto m = sz::utf8_uncased_search(str(haystack), needle);
            expect_true(m.offset == 0 && m.length == 16);
        }
    }

    // Long needle tests at ring buffer boundary (32 folded runes):
    // The serial implementation uses a 32-rune ring buffer for fold comparisons
    {
        // Exactly 32 ASCII characters (32 folded runes)
        std::string hay32(32, 'a');
        {
            auto m = sz::utf8_uncased_search(str(hay32 + "xyz"), hay32);
            expect_true(m.offset == 0 && m.length == 32);
        }

        // 33 ASCII characters (crosses ring buffer boundary)
        std::string hay33(33, 'a');
        {
            auto m = sz::utf8_uncased_search(str(hay33 + "xyz"), hay33);
            expect_true(m.offset == 0 && m.length == 33);
        }

        // 16 eszett characters → 32 folded runes (ss × 16), exactly at boundary
        std::string hay_16_ss(16, '\xC3');
        for (size_t i = 0; i < 16; ++i) hay_16_ss.insert(i * 2 + 1, 1, '\x9F'); // Build "ßßßßßßßßßßßßßßßß"
        std::string needle_32_s(32, 's');
        {
            auto m = sz::utf8_uncased_search(str(hay_16_ss + "end"), needle_32_s);
            expect_true(m.offset == 0 && m.length == 32);
        }

        // 64 ASCII characters (tests double boundary)
        std::string hay64(64, 'b');
        {
            auto m = sz::utf8_uncased_search(str(hay64 + "xyz"), hay64);
            expect_true(m.offset == 0 && m.length == 64);
        }
    }

    // Eszett at SIMD 64-byte chunk boundaries
    {
        // ß at position 62 (ends exactly at 64-byte boundary)
        std::string prefix62(62, 'a');
        {
            auto m = sz::utf8_uncased_search(str(prefix62 + "\xC3\x9F" + "xyz"), "ss");
            expect_true(m.offset == 62 && m.length == 2);
        }

        // ß straddling 64-byte boundary (starts at 63)
        std::string prefix63(63, 'a');
        {
            auto m = sz::utf8_uncased_search(str(prefix63 + "\xC3\x9F" + "xyz"), "ss");
            expect_true(m.offset == 63 && m.length == 2);
        }

        // ß exactly at 64-byte boundary
        std::string prefix64(64, 'a');
        {
            auto m = sz::utf8_uncased_search(str(prefix64 + "\xC3\x9F" + "xyz"), "ss");
            expect_true(m.offset == 64 && m.length == 2);
        }

        // Word with ß crossing boundary: "straße" starting at position 60
        std::string prefix60(60, 'a');
        {
            auto m = sz::utf8_uncased_search(str(prefix60 + "straßezzz"), "strasse");
            expect_true(m.offset == 60 && m.length == 7);
        }
    }

    // Cross-script boundary tests (different SIMD kernels)
    {
        // ASCII → Greek transition at SIMD boundary
        std::string ascii60(60, 'x');
        {
            auto m = sz::utf8_uncased_search(str(ascii60 + "\xCE\xB1\xCE\xB2\xCE\xB3"), "\xCE\x91\xCE\x92\xCE\x93");
            expect_true(m.offset == 60 && m.length == 6);
        } // ΑΒΓ matching αβγ

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
        {
            auto m = sz::utf8_uncased_search(str("brown S\xC3\x9F\xC3\xA0 jumps"), "sss\xC3\xA0");
            expect_true(m.offset == 6 && m.length == 5);
        }

        // Lowercase: "sßà" should match "sssà" - match is 5 bytes
        {
            auto m = sz::utf8_uncased_search(str("brown s\xC3\x9F\xC3\xA0 jumps"), "sss\xC3\xA0");
            expect_true(m.offset == 6 && m.length == 5);
        }

        // Uppercase ß (U+1E9E) when it exists - "ẞà" should match "ssà"
        {
            auto m = sz::utf8_uncased_search(str("brown \xE1\xBA\x9E\xC3\xA0 jumps"), "ss\xC3\xA0");
            expect_true(m.offset == 6 && m.length == 5);
        }

        // Triple-s with space: "sß " should match "sss ".
        // Match starts at byte 7 where 's' is (byte 6 is space before 's')
        {
            auto m = sz::utf8_uncased_search(str("ǰbee3 s\xC3\x9F ee\xC3\xA9 nc"), "sss ee\xC3\xA9");
            expect_true(m.offset == 7 && m.length == 8);
        }

        // "ss" needle vs "ß" haystack (basic case)
        {
            auto m = sz::utf8_uncased_search(str("ßabc"), "ssabc");
            expect_true(m.offset == 0 && m.length == 5);
        }

        // "sss" needle vs "sß" haystack
        {
            auto m = sz::utf8_uncased_search(str("sßabc"), "sssabc");
            expect_true(m.offset == 0 && m.length == 6);
        }
    }

    // Pattern 8: Greek Mu UTF-8 boundary.
    // Needle: CE BC (Greek μ - U+03BC).
    // Pins that a match only lands on a valid UTF-8 character boundary, never on the mid-byte
    // BC that also appears as the trailing byte of an unrelated codepoint such as ¼.
    {
        // Simple Greek mu search
        {
            auto m = sz::utf8_uncased_search(str("hello \xCE\xBC world"), "\xCE\xBC");
            expect_true(m.offset == 6 && m.length == 2);
        }

        // Greek mu not at position where 0xBC appears as second byte of another char
        // Create haystack with Latin-1 char ending in 0xBC, then Greek mu.
        // This ensures we only match at valid UTF-8 boundaries
        {
            auto m = sz::utf8_uncased_search(str("test \xC2\xBC thing \xCE\xBC end"), "\xCE\xBC");
            expect_true(m.offset == 14 && m.length == 2);
        } // Only at actual μ, not at ¼

        // Multiple Greek chars around mu
        {
            auto m = sz::utf8_uncased_search(str("\xCE\xB1\xCE\xBC\xCE\xB2"), "\xCE\xBC");
            expect_true(m.offset == 2 && m.length == 2);
        }
    }

    // Pattern 9: Cyrillic Moscow case folding.
    // Haystack: "се Москва" (uppercase М - D0 9C)
    // Needle: "се москва" (lowercase м - D0 BC)
    // Should match uncasedly
    {
        // Simple Moscow: Москва vs москва
        {
            auto m = sz::utf8_uncased_search(str("\xD0\x9C\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0"),
                                             "\xD0\xBC\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0");
            expect_true(m.offset == 0 && m.length == 12);
        }

        // Moscow with Latin prefix
        {
            auto m = sz::utf8_uncased_search(str("se \xD0\x9C\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0"),
                                             "se \xD0\xBC\xD0\xBE\xD1\x81\xD0\xBA\xD0\xB2\xD0\xB0");
            expect_true(m.offset == 0 && m.length == 15);
        }

        // All Cyrillic uppercase vs lowercase
        {
            auto m = sz::utf8_uncased_search(str("\xD0\x90\xD0\x91\xD0\x92"),
                                             "\xD0\xB0\xD0\xB1\xD0\xB2"); // АБВ and абв
            expect_true(m.offset == 0 && m.length == 6);
        }

        // Mixed: ПРИВЕТ vs привет
        {
            auto m = sz::utf8_uncased_search(str("\xD0\x9F\xD0\xA0\xD0\x98\xD0\x92\xD0\x95\xD0\xA2"),
                                             "\xD0\xBF\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82"); // ПРИВЕТ and привет
            expect_true(m.offset == 0 && m.length == 12);
        }
    }

    // Pattern 10: Ligature fi expansion.
    // Haystack contains ﬁ (EF AC 81 - U+FB01)
    // Needle has "fi" (66 69).
    // ﬁ should case-fold to "fi"
    {
        // Simple: ﬁ vs fi
        {
            auto m = sz::utf8_uncased_search(str("\xEF\xAC\x81nd"), "find");
            expect_true(m.offset == 0 && m.length == 5);
        }

        // With uppercase: ﬁ vs FI
        {
            auto m = sz::utf8_uncased_search(str("\xEF\xAC\x81nd"), "FInd");
            expect_true(m.offset == 0 && m.length == 5);
        }

        // ff ligature: ﬀ (EF AC 80) vs ff
        {
            auto m = sz::utf8_uncased_search(str("\xEF\xAC\x80oo"), "ffoo");
            expect_true(m.offset == 0 && m.length == 5);
        }

        // ffi ligature: ﬃ (EF AC 83) vs ffi
        {
            auto m = sz::utf8_uncased_search(str("ﬃce"), "ffice");
            expect_true(m.offset == 0 && m.length == 5);
        }

        // fl ligature: ﬂ (EF AC 82) vs fl
        {
            auto m = sz::utf8_uncased_search(str("\xEF\xAC\x82oor"), "floor");
            expect_true(m.offset == 0 && m.length == 6);
        }

        // ffl ligature: ﬄ (EF AC 84) vs ffl
        {
            auto m = sz::utf8_uncased_search(str("waﬄe"), "waffle");
            expect_true(m.offset == 0 && m.length == 6);
        }
    }

    // Pattern 11: Combining marks vs precomposed.
    // j + combining caron (6A CC 8C) vs ǰ (C7 B0 - U+01F0)
    // These are canonically equivalent in Unicode
    // Note: StringZilla may or may not perform normalization - document behavior
    {
        // Precomposed ǰ vs decomposed j+caron:
        // If normalization is performed, these should match.
        // If not, they won't match (current behavior TBD)
        {
            auto m = sz::utf8_uncased_search(str("\xC7\xB0ump"), "j\xCC\x8Cump");
            expect_true(m.offset == 0 && m.length == 5);
        }

        // é precomposed (C3 A9) vs e+acute (65 CC 81)
        {
            auto m = sz::utf8_uncased_search(str("\xC3\xA9lan"), "e\xCC\x81lan");
            expect_equal(m.offset, str::npos);
        }
    }

    // Pattern 12: Mixed script verification.
    // These test that case folding works correctly when multiple scripts are mixed
    {
        // Greek κόσμ mixed with Latin
        {
            auto m = sz::utf8_uncased_search(str("brown \xCE\xBA\xCF\x8C\xCF\x83 end"), "\xCE\xBA\xCF\x8C\xCF\x83");
            expect_true(m.offset == 6 && m.length == 6);
        }

        // Greek sigma case: Σ (CE A3) vs σ (CF 83) vs ς (CF 82 - final sigma)
        {
            auto m = sz::utf8_uncased_search(str("\xCE\xA3\xCE\xB5"), "\xCF\x83\xCE\xB5");
            expect_true(m.offset == 0 && m.length == 4);
        }

        // Armenian + Latin mixed
        {
            auto m = sz::utf8_uncased_search(str("test \xD5\xA2\xD5\xA1\xD6\x80\xD5\xA5\xD5\xBE world"),
                                             "\xD5\xA2\xD5\xA1\xD6\x80\xD5\xA5\xD5\xBE");
            expect_true(m.offset == 5 && m.length == 10);
        }

        // Armenian ligature: և (D6 87 - U+0587) vs ե+ւ (D5 A5 D6 82)
        {
            auto m = sz::utf8_uncased_search(str("\xD6\x87nd"), "\xD5\xA5\xD6\x82nd");
            expect_true(m.offset == 0 && m.length == 4);
        }
    }
}

#pragma endregion Unit

#pragma region Drivers

/** Adversarial invalid-input safety probe through the dispatch points. */
void test_utf8_uncased_safety() {
    split_mix_generator_t generator {case_stream_key()};
    check_utf8_uncased_safety_(generator, utf8_uncased_dispatched);
}

/** The dispatch points against serial: the case-fold differential and the full find battery. */
void test_utf8_uncased_all() {
    split_mix_generator_t generator {case_stream_key()};
    check_utf8_uncased_equivalence_(generator, utf8_uncased_dispatched);
}

#pragma endregion Drivers

} // namespace ashvardanian::stringzilla::test
