/**
 *  @file test/utf8_wordbreaks.cpp
 *  @author Ash Vardanian
 *  @date June 22, 2026
 *  @brief UAX-29 word-boundary (Word_Break) tests: known-answer goldens, malformed-input safety,
 *      and the serial-vs-dispatched differential over hardened corpora.
 */
#undef NDEBUG // ! Enable all assertions for testing

#if !defined(_ITERATOR_DEBUG_LEVEL) || _ITERATOR_DEBUG_LEVEL == 0
#define _ITERATOR_DEBUG_LEVEL 1
#endif

#if defined(STRINGZILLA_DEBUG)
#undef STRINGZILLA_DEBUG
#endif
#define STRINGZILLA_DEBUG 1 // ! Enforce aggressive logging in this translation unit

#include <stringzilla/stringzilla.hpp> // `sz::string_view_t`

#include <string> // `std::string`
#include <vector> // `std::vector`

#include "cross.hpp" // `check_utf8_wordbreaks_unit_` and the shared segmentation harness

using namespace sz::test;

/** The word-break dispatch point in the shape of a segmentation backend. */
static utf8_segment_backend_t const utf8_wordbreaks_dispatched = {"dispatched", cpu_best<sz_utf8_wordbreaks_best>};

#pragma region Unit

/** Known-answer table for @c sz_rune_is_word_char, the UAX-29 word-character classification. */
static void check_utf8_wordbreaks_classification_() {
    // ASCII letters, digits, underscore, and the mid-word apostrophe are word characters.
    verify(sz_rune_is_word_char('A') == sz_true_k);
    verify(sz_rune_is_word_char('z') == sz_true_k);
    verify(sz_rune_is_word_char('0') == sz_true_k);
    verify(sz_rune_is_word_char('9') == sz_true_k);
    verify(sz_rune_is_word_char('_') == sz_true_k);
    verify(sz_rune_is_word_char('\'') == sz_true_k);

    // ASCII whitespace and punctuation are boundaries.
    verify(sz_rune_is_word_char(' ') == sz_false_k);
    verify(sz_rune_is_word_char('\n') == sz_false_k);
    verify(sz_rune_is_word_char('\t') == sz_false_k);
    verify(sz_rune_is_word_char('!') == sz_false_k);
    verify(sz_rune_is_word_char('-') == sz_false_k);

    // Latin Extended, Greek, Cyrillic, Hebrew, Arabic letters and Hangul syllables are word characters.
    verify(sz_rune_is_word_char(0x00DF) == sz_true_k); // ß
    verify(sz_rune_is_word_char(0x0100) == sz_true_k); // Latin Extended-A start
    verify(sz_rune_is_word_char(0x03B1) == sz_true_k); // Greek alpha
    verify(sz_rune_is_word_char(0x0430) == sz_true_k); // Cyrillic a
    verify(sz_rune_is_word_char(0x05D0) == sz_true_k); // Hebrew alef
    verify(sz_rune_is_word_char(0x0627) == sz_true_k); // Arabic alef
    verify(sz_rune_is_word_char(0xAC00) == sz_true_k); // Hangul first
    verify(sz_rune_is_word_char(0xD7A3) == sz_true_k); // Hangul last

    // CJK ideographs, spaces, dashes, and emoji are boundaries (not word characters under TR29).
    verify(sz_rune_is_word_char(0x4E00) == sz_false_k);  // CJK first
    verify(sz_rune_is_word_char(0x9FFF) == sz_false_k);  // CJK last
    verify(sz_rune_is_word_char(0x3000) == sz_false_k);  // Ideographic space
    verify(sz_rune_is_word_char(0x2014) == sz_false_k);  // Em dash
    verify(sz_rune_is_word_char(0x1F600) == sz_false_k); // emoji

    // Edge cases.
    verify(sz_rune_is_word_char(0x0000) == sz_false_k); // NUL
    verify(sz_rune_is_word_char(0x007F) == sz_false_k); // DEL
    verify(sz_rune_is_word_char(0xFFFF) == sz_false_k); // BMP max
}

/** Known-answer word-break vectors via the dispatch point and the C++ range. */
void test_utf8_wordbreaks_unit() {
    check_utf8_wordbreaks_classification_();
    check_utf8_wordbreaks_unit_(utf8_wordbreaks_dispatched);

    // C++ range wrapper known-answer: `utf8_wordbreaks()` faithfully tiles the input into the UAX-29 segments
    // (words and the separators between them); concatenating them reconstructs the input.
    {
        sz::string_view_t const text("Hello, world!");
        auto segments = text.utf8_wordbreaks().template to<std::vector<std::string>>();
        verify(segments.size() == 5 && segments[0] == "Hello" && segments[3] == "world" && "C++ utf8_wordbreaks");
        std::string rejoined;
        for (auto const &s : segments) rejoined += s;
        verify(rejoined == "Hello, world!" && "wordbreaks tile losslessly");
    }

    // Word-break counts for the shared prose fixtures; per-fixture rationale lives in test/utf8.hpp.
    auto count_wordbreaks = [](std::string_view text) {
        return sz::string_view_t(text).utf8_wordbreaks().template to<std::vector<std::string>>().size();
    };
    verify(count_wordbreaks(utf8_prose_hotel_review()) == 100 && "hotel_review wordbreaks");
    verify(count_wordbreaks(utf8_prose_news_lede()) == 83 && "news_lede wordbreaks");
    verify(count_wordbreaks(utf8_prose_concert_post()) == 69 && "concert_post wordbreaks");
    verify(count_wordbreaks(utf8_prose_rtl_scripts()) == 98 && "rtl_scripts wordbreaks");
    verify(count_wordbreaks(utf8_prose_micro_apostrophe()) == 5 && "micro_apostrophe wordbreaks");
}

#pragma endregion Unit

#pragma region Rule coverage

/** Rule-coverage gate: every WB motif agrees serial-vs-dispatched at window phases. */
void test_utf8_wordbreaks_rules() { check_utf8_wordbreaks_rules_(utf8_wordbreaks_dispatched); }

#pragma endregion Rule coverage

#pragma region Safety

/** Malformed-input safety of the word-break dispatch point. */
void test_utf8_wordbreaks_safety(test_context_t &context) {
    check_utf8_wordbreaks_safety_(context, utf8_wordbreaks_dispatched);
}

#pragma endregion Safety

#pragma region Drivers

/** Serial-vs-dispatched word differential over the dense, long-range and seam corpora. */
void test_utf8_wordbreaks_all(test_context_t &context) {
    check_utf8_wordbreaks_equivalence_(context, utf8_wordbreaks_dispatched);
}

#pragma endregion Drivers
