/**
 *  @file test/utf8_sentences.cpp
 *  @author Ash Vardanian
 *  @date June 22, 2026
 *  @brief UAX-29 sentence-boundary (Sentence_Break) tests: known-answer goldens, malformed-input
 *      safety, and the serial-vs-dispatched differential over hardened corpora.
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

#include "cross.hpp" // `check_utf8_sentences_unit_` and the shared segmentation harness

using namespace sz::test;

/** The sentence-break dispatch point in the shape of a segmentation backend. */
static utf8_segment_backend_t const utf8_sentences_dispatched = {"dispatched", cpu_best<sz_utf8_sentences_best>};

#pragma region Unit

/** Known-answer sentence-break vectors via the dispatch point and the C++ range. */
void test_utf8_sentences_unit() {
    check_utf8_sentences_unit_(utf8_sentences_dispatched);

    // C++ range wrapper known-answer: the view must faithfully expose the kernel's sentence segments.
    std::vector<std::string> const sentences =
        sz::string_view_t("Hi. Yo.").utf8_sentences().template to<std::vector<std::string>>();
    verify(sentences.size() == 2 && "C++ utf8_sentences range");

    // Sentence counts for the shared prose fixtures; per-fixture rationale lives in test/utf8.hpp.
    auto count_sentences = [](std::string_view text) {
        return sz::string_view_t(text).utf8_sentences().template to<std::vector<std::string>>().size();
    };
    verify(count_sentences(utf8_prose_hotel_review()) == 5 && "hotel_review sentences");
    verify(count_sentences(utf8_prose_concert_post()) == 4 && "concert_post sentences");
    verify(count_sentences(utf8_prose_news_lede()) == 6 && "news_lede sentences");
    verify(count_sentences(utf8_prose_micro_hardbreaks()) == 3 && "micro_hardbreaks sentences");
}

#pragma endregion Unit

#pragma region Rule coverage

/** Rule-coverage gate: every SB motif agrees serial-vs-dispatched at window phases. */
void test_utf8_sentences_rules() { check_utf8_sentences_rules_(utf8_sentences_dispatched); }

#pragma endregion Rule coverage

#pragma region Safety

/** Malformed-input safety of the sentence-break dispatch point. */
void test_utf8_sentences_safety(test_context_t &context) {
    check_utf8_sentences_safety_(context, utf8_sentences_dispatched);
}

#pragma endregion Safety

#pragma region Drivers

/** Serial-vs-dispatched sentence differential over the dense and long-range corpora. */
void test_utf8_sentences_all(test_context_t &context) {
    check_utf8_sentences_equivalence_(context, utf8_sentences_dispatched);
}

#pragma endregion Drivers
