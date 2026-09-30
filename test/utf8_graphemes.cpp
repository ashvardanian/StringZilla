/**
 *  @file test/utf8_graphemes.cpp
 *  @author Ash Vardanian
 *  @date June 22, 2026
 *  @brief UAX-29 grapheme-cluster (Grapheme_Cluster_Break) tests: known-answer goldens,
 *      malformed-input safety, and the serial-vs-dispatched differential over hardened corpora.
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

#include "cross.hpp" // `check_utf8_graphemes_unit_` and the shared segmentation harness

using namespace sz::test;

/** The grapheme-cluster dispatch point in the shape of a segmentation backend. */
static utf8_segment_backend_t const utf8_graphemes_dispatched = {"dispatched", cpu_best<sz_utf8_graphemes_best>};

#pragma region Unit

/** Known-answer grapheme-cluster vectors via the dispatch point and the C++ range. */
void test_utf8_graphemes_unit() {
    check_utf8_graphemes_unit_(utf8_graphemes_dispatched);

    // C++ range wrapper known-answer: the view must faithfully expose the kernel's clusters.
    std::vector<std::string> const clusters =
        sz::string_view_t("ab").utf8_graphemes().template to<std::vector<std::string>>();
    verify(clusters.size() == 2 && clusters[0] == "a" && clusters[1] == "b" && "C++ utf8_graphemes range");

    // Grapheme-cluster counts for the shared prose fixtures; per-fixture rationale lives in test/utf8.hpp.
    auto count_graphemes = [](std::string_view text) {
        return sz::string_view_t(text).utf8_graphemes().template to<std::vector<std::string>>().size();
    };
    verify(count_graphemes(utf8_prose_pride_caption()) == 206 && "pride_caption graphemes");
    verify(count_graphemes(utf8_prose_devanagari_tip()) == 252 && "devanagari_tip graphemes");
    verify(count_graphemes(utf8_prose_concert_post()) == 134 && "concert_post graphemes");
    verify(count_graphemes(utf8_prose_rtl_scripts()) == 256 && "rtl_scripts graphemes");
    verify(count_graphemes(utf8_prose_micro_prepend()) == 3 && "micro_prepend graphemes");

    // Codepoints are not clusters: the emoji paragraph has more runes than grapheme clusters.
    std::size_t const pride_caption_runes =
        sz::string_view_t(utf8_prose_pride_caption()).utf8_runes().template to<std::vector<sz_rune_t>>().size();
    verify(pride_caption_runes == 222 && "pride_caption runes");
    verify(pride_caption_runes > count_graphemes(utf8_prose_pride_caption()) && "pride_caption runes exceed graphemes");
}

#pragma endregion Unit

#pragma region Rule coverage

/** Rule-coverage gate: every GB motif agrees serial-vs-dispatched at window phases. */
void test_utf8_graphemes_rules() { check_utf8_graphemes_rules_(utf8_graphemes_dispatched); }

#pragma endregion Rule coverage

#pragma region Safety

/** Malformed-input safety of the grapheme-cluster dispatch point. */
void test_utf8_graphemes_safety(test_context_t &context) {
    check_utf8_graphemes_safety_(context, utf8_graphemes_dispatched);
}

#pragma endregion Safety

#pragma region Drivers

/** Serial-vs-dispatched grapheme differential over the dense and long-range corpora. */
void test_utf8_graphemes_all(test_context_t &context) {
    check_utf8_graphemes_equivalence_(context, utf8_graphemes_dispatched);
}

#pragma endregion Drivers
