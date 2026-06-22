/**
 *  @brief  UAX-29 sentence-boundary (Sentence_Break) tests: known-answer goldens, malformed-input safety, and the
 *          serial-vs-ISA differential over hardened corpora.
 *  @file   scripts/test_utf8_sentences.cpp
 *  @author Ash Vardanian
 */
#undef NDEBUG // ! Enable all assertions for testing

#if !defined(_ITERATOR_DEBUG_LEVEL) || _ITERATOR_DEBUG_LEVEL == 0
#define _ITERATOR_DEBUG_LEVEL 1
#endif

#define SZ_USE_MISALIGNED_LOADS 0
#if defined(SZ_DEBUG)
#undef SZ_DEBUG
#endif
#define SZ_DEBUG 1 // ! Enforce aggressive logging in this translation unit

#include <cassert> // `assert`
#include <cstddef> // `std::size_t`
#include <cstdio>  // `std::printf`

#include <random> // `std::mt19937`, `std::uniform_int_distribution`
#include <string> // `std::string`
#include <vector> // `std::vector`

#include "test_utf8.hpp" // shared segmentation harness (pulls in StringZilla + `test_stringzilla.hpp`)

#pragma region Unit

/** @brief Hand-checked UAX-29 sentence-break golden vectors: each source text and its expected sentence segments. */
static utf8_unit_case_t const utf8_sentences_unit_cases[] = {
    {""_sv, {}},
    {"Hello."_sv, {"Hello."_sv}},
    {"Hello. World."_sv, {"Hello. "_sv, "World."_sv}}, // terminator + space ends the sentence
    {"Mr. Smith went."_sv,
     {"Mr. "_sv, "Smith went."_sv}},                // UAX-29 has no abbreviation list: '. ' before uppercase breaks
    {"etc. and so on"_sv, {"etc. and so on"_sv}},   // SB8: '. ' before a lowercase word does NOT break
    {"One.\nTwo."_sv, {"One.\n"_sv, "Two."_sv}},    // ParaSep after terminator
    {"A! B? C."_sv, {"A! "_sv, "B? "_sv, "C."_sv}}, // multiple terminators
};
static constexpr std::size_t utf8_sentences_unit_cases_count = sizeof(utf8_sentences_unit_cases) /
                                                               sizeof(utf8_sentences_unit_cases[0]);

/** @brief Known-answer sentence-break vectors through dispatched, serial, and each ISA backend + the C++ range. */
void test_utf8_sentences_unit() {
    std::printf("  - testing UTF-8 sentence-break known-answer vectors...\n");

    check_utf8_segment_unit_("sentence", sz_utf8_sentences, utf8_sentences_unit_cases,
                             utf8_sentences_unit_cases_count); // Dispatched
    check_utf8_segment_unit_("sentence", sz_utf8_sentences_serial, utf8_sentences_unit_cases,
                             utf8_sentences_unit_cases_count);
#if SZ_USE_ICELAKE
    check_utf8_segment_unit_("sentence", sz_utf8_sentences_icelake, utf8_sentences_unit_cases,
                             utf8_sentences_unit_cases_count);
#endif

    // C++ range wrapper known-answer: the view must faithfully expose the kernel's sentence segments.
    std::vector<std::string> const sentences =
        sz::string_view("Hi. Yo.").utf8_sentences().template to<std::vector<std::string>>();
    assert(sentences.size() == 2 && "C++ utf8_sentences range");
}

#pragma endregion // Unit

#pragma region Equivalence

/**
 *  @brief UAX-29 sentence-break corner motifs (sprinkled into the random corpus): ATerm vs STerm, terminator +
 *         Close + Space + case, paragraph separators after a terminator, SContinue, abbreviation-like and numeric.
 */
static sz::string_view const utf8_sentences_motifs[] = {
    "End. Next"_sv,               // ATerm + space + uppercase (SB break)
    "End! Next"_sv,               // STerm + space + uppercase (SB break)
    "End? Next"_sv,               // STerm '?' + space + uppercase
    "end. next"_sv,               // SB8: '. ' before lowercase does not break
    "(End.) Next"_sv,             // terminator + Close ')' + space + uppercase
    "\"End.\" Next"_sv,           // terminator + Close '"' + space + uppercase
    "End.\xE2\x80\xA8" "Next"_sv, // ParaSep U+2028 after terminator
    "End.\xE2\x80\xA9" "Next"_sv, // ParaSep U+2029 after terminator
    "End.\xC2\x85" "Next"_sv,     // ParaSep U+0085 NEL after terminator
    "Item, more"_sv,              // SContinue ',' continuation
    "Item: more"_sv,              // SContinue ':' continuation
    "Mr. Smith"_sv,               // abbreviation-like
    "Dr. House"_sv,               // abbreviation-like
    "U.S.A. ok"_sv,               // dotted acronym
    "3.14 value"_sv,              // numeric ATerm (no break)
};
static constexpr std::size_t utf8_sentences_motifs_count = sizeof(utf8_sentences_motifs) /
                                                           sizeof(utf8_sentences_motifs[0]);

/** @brief ATerm + Close* + Sp* run @p link_count wide, then a Lower (SB8 continuation, no break). */
static std::string utf8_sentences_dense_aterm_sp_lower_(std::size_t link_count) {
    std::string text;
    append_codepoint_(text, 0x0055);                                                           // 'U' Upper
    append_codepoint_(text, 0x002E);                                                           // '.' ATerm
    for (std::size_t index = 0; index != link_count; ++index) append_codepoint_(text, 0x0020); // Sp run
    append_codepoint_(text, 0x0061);                                                           // 'a' Lower
    return text;
}

/** @brief Terminator-dense `A. A. A. ...` repeated @p link_count times (one break per terminator). */
static std::string utf8_sentences_dense_terminators_(std::size_t link_count) {
    std::string text;
    for (std::size_t index = 0; index != link_count; ++index) text.append("A. ");
    return text;
}

/** @brief Ideographic full stop (U+3002) + Sp run + CJK, repeated @p link_count times (multibyte STerm). */
static std::string utf8_sentences_dense_cjk_term_(std::size_t link_count) {
    std::string text;
    for (std::size_t index = 0; index != link_count; ++index) {
        append_codepoint_(text, 0x4E2D); // 中
        append_codepoint_(text, 0x3002); // 。 ideographic full stop (STerm)
        append_codepoint_(text, 0x0020); // Sp
    }
    return text;
}

/** @brief The sentence family's high-density homogeneous runs, sized from @p rng to span several 64-byte windows. */
static std::vector<std::string> utf8_sentences_dense_runs_(std::mt19937 &rng) {
    std::uniform_int_distribution<std::size_t> wide(60, 220);
    std::size_t const wide_count = wide(rng);
    return {utf8_sentences_dense_aterm_sp_lower_(wide_count), utf8_sentences_dense_terminators_(wide_count),
            utf8_sentences_dense_cjk_term_(wide_count)};
}

/** @brief SB8: `Upper ATerm Sp{gap} Lower` — long Sp run before a lowercase keeps one sentence. */
static std::string utf8_straddle_sb8_lower_(std::size_t gap) {
    std::string text;
    append_codepoint_(text, 0x0055);                                                    // 'U'
    append_codepoint_(text, 0x002E);                                                    // '.'
    for (std::size_t index = 0; index != gap; ++index) append_codepoint_(text, 0x0020); // Sp run across windows
    append_codepoint_(text, 0x0061);                                                    // 'a' Lower
    return text;
}

/** @brief SB11: `Upper ATerm Sp{gap} Upper` — same run before an uppercase must break. */
static std::string utf8_straddle_sb11_upper_(std::size_t gap) {
    std::string text;
    append_codepoint_(text, 0x0055);                                                    // 'U'
    append_codepoint_(text, 0x002E);                                                    // '.'
    for (std::size_t index = 0; index != gap; ++index) append_codepoint_(text, 0x0020); // Sp run across windows
    append_codepoint_(text, 0x0042);                                                    // 'B' Upper
    return text;
}

/** @brief The sentence family's long-range straddling constructions for a given @p gap (rng kept to match the type). */
static std::vector<std::string> utf8_sentences_straddles_(std::mt19937 & /*rng*/, std::size_t gap) {
    return {utf8_straddle_sb8_lower_(gap), utf8_straddle_sb11_upper_(gap)};
}

/** @brief Assemble the sentence family's differential corpora (motifs + dense + straddle; no seam regressions). */
static utf8_segment_corpora_t utf8_sentences_corpora_() {
    utf8_segment_corpora_t corpora = {"sentence",
                                      utf8_sentences_motifs,
                                      utf8_sentences_motifs_count,
                                      &utf8_sentences_dense_runs_,
                                      &utf8_sentences_straddles_,
                                      nullptr,
                                      0};
    return corpora;
}

#pragma endregion // Equivalence

#pragma region Safety

/** @brief Malformed-input safety of the UTF-8 sentence kernels (serial / dispatched / icelake). */
void test_utf8_sentences_safety() {
    std::printf("  - testing malformed-input safety of UTF-8 sentence kernels...\n");
    check_utf8_segment_safety_("sentence (serial)", sz_utf8_sentences_serial);
    check_utf8_segment_safety_("sentence (dispatched)", sz_utf8_sentences);
#if SZ_USE_ICELAKE
    check_utf8_segment_safety_("sentence (icelake)", sz_utf8_sentences_icelake);
#endif
    std::printf("    sentence safety passed!\n");
}

#pragma endregion // Safety

#pragma region Drivers

/** @brief Serial-vs-ISA sentence differential over the hardened corpora (high-density + long-range). */
void test_utf8_sentences_all() {
#if SZ_USE_ICELAKE
    utf8_segment_corpora_t const corpora = utf8_sentences_corpora_();
    test_utf8_segment_equivalence_(sz_utf8_sentences_serial, sz_utf8_sentences_icelake, corpora);
#endif
}

#pragma endregion // Drivers
