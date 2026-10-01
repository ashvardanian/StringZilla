/**
 *  @file test/utf8_linebreaks.cpp
 *  @author Ash Vardanian
 *  @date June 22, 2026
 *  @brief UAX-14 line-break (linewrap) tests: known-answer goldens, malformed-input safety, and the
 *      serial-vs-dispatched differential over hardened corpora.
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

#include "cross.hpp" // `check_utf8_linebreaks_unit_` and the shared segmentation harness

namespace ashvardanian::stringzilla::test {

/** The line-break dispatch point in the shape of a segmentation backend. */
static utf8_segment_backend_t const utf8_linebreaks_dispatched = {"dispatched", cpu_best<sz_utf8_linebreaks_best>};

#pragma region Unit

/** Known-answer line-break vectors via the dispatch point and the C++ range. */
void test_utf8_linebreaks_unit() {
    check_utf8_linebreaks_unit_(utf8_linebreaks_dispatched);

    // C++ range wrapper known-answer: the view must faithfully expose the kernel's segments.
    std::vector<std::string> const wrapped =
        sz::string_view_t("a\nb").utf8_linebreaks().template to<std::vector<std::string>>();
    verify(wrapped.size() == 2 && wrapped[0] == "a\n" && wrapped[1] == "b" && "C++ utf8_linebreaks range");

    // Line-break counts for the shared prose fixtures; per-fixture rationale lives in test/utf8.hpp.
    auto count_linebreaks = [](std::string_view text) {
        return sz::string_view_t(text).utf8_linebreaks().template to<std::vector<std::string>>().size();
    };
    verify(count_linebreaks(utf8_prose_hotel_review()) == 45 && "hotel_review linebreaks");
    verify(count_linebreaks(utf8_prose_science_abstract()) == 43 && "science_abstract linebreaks");
    verify(count_linebreaks(utf8_prose_news_lede()) == 32 && "news_lede linebreaks");
}

#pragma endregion Unit

#pragma region Rule coverage

/** Rule-coverage gate: every LB motif agrees serial-vs-dispatched at window phases. */
void test_utf8_linebreaks_rules() { check_utf8_linebreaks_rules_(utf8_linebreaks_dispatched); }

#pragma endregion Rule coverage

#pragma region Safety

/** Malformed-input safety of the line-break dispatch point. */
void test_utf8_linebreaks_safety(test_context_t &context) {
    check_utf8_linebreaks_safety_(context, utf8_linebreaks_dispatched);
}

#pragma endregion Safety

#pragma region Drivers

/** Serial-vs-dispatched line differential over the dense and long-range corpora. */
void test_utf8_linebreaks_all(test_context_t &context) {
    check_utf8_linebreaks_equivalence_(context, utf8_linebreaks_dispatched);
}

#pragma endregion Drivers

} // namespace ashvardanian::stringzilla::test
