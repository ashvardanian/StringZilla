/**
 *  @brief  UAX-14 line-break (linewrap) tests: known-answer goldens, malformed-input safety, and the
 *          serial-vs-ISA differential over hardened corpora.
 *  @file   scripts/test_utf8_linewraps.cpp
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

/** @brief Hand-checked UAX-14 line-break golden vectors: each source text and its expected line segments. */
static utf8_unit_case_t const utf8_linewraps_unit_cases[] = {
    {""_sv, {}},
    {"hello world"_sv, {"hello "_sv, "world"_sv}},          // soft wrap after the space
    {"a\nb"_sv, {"a\n"_sv, "b"_sv}},                        // LF hard break
    {"a\r\nb"_sv, {"a\r\n"_sv, "b"_sv}},                    // CRLF hard break, one segment
    {"a\xE2\x80\xA8" "b"_sv, {"a\xE2\x80\xA8"_sv, "b"_sv}}, // U+2028 LINE SEPARATOR
    {"a\xE2\x80\xA9" "b"_sv, {"a\xE2\x80\xA9"_sv, "b"_sv}}, // U+2029 PARAGRAPH SEPARATOR
};
static constexpr std::size_t utf8_linewraps_unit_cases_count = sizeof(utf8_linewraps_unit_cases) /
                                                               sizeof(utf8_linewraps_unit_cases[0]);

/** @brief Known-answer line-break vectors through dispatched, serial, and each ISA backend + the C++ range. */
void test_utf8_linewraps_unit() {
    std::printf("  - testing UTF-8 line-break known-answer vectors...\n");

    check_utf8_segment_unit_("linewrap", sz_utf8_linewraps, utf8_linewraps_unit_cases,
                             utf8_linewraps_unit_cases_count); // Dispatched
    check_utf8_segment_unit_("linewrap", sz_utf8_linewraps_serial, utf8_linewraps_unit_cases,
                             utf8_linewraps_unit_cases_count);
#if SZ_USE_ICELAKE
    check_utf8_segment_unit_("linewrap", sz_utf8_linewraps_icelake, utf8_linewraps_unit_cases,
                             utf8_linewraps_unit_cases_count);
#endif

    // C++ range wrapper known-answer: the view must faithfully expose the kernel's segments.
    std::vector<std::string> const wrapped =
        sz::string_view("a\nb").utf8_linewraps().template to<std::vector<std::string>>();
    assert(wrapped.size() == 2 && wrapped[0] == "a\n" && wrapped[1] == "b" && "C++ utf8_linewraps range");
}

#pragma endregion // Unit

#pragma region Equivalence

/** @brief UAX-14 line-break corner motifs (sprinkled into the random corpus): mandatory breaks, OP/CL, HY, GL, NU. */
static sz::string_view const utf8_linewraps_motifs[] = {
    "a\nb"_sv,              // LF mandatory break
    "a\rb"_sv,              // CR mandatory break
    "a\r\nb"_sv,            // CRLF mandatory break (one segment)
    "a\xE2\x80\xA8" "b"_sv, // U+2028 LINE SEPARATOR mandatory
    "a\xE2\x80\xA9" "b"_sv, // U+2029 PARAGRAPH SEPARATOR mandatory
    "a\xC2\x85" "b"_sv,     // U+0085 NEL mandatory
    "a\xE2\x80\x8D" "b"_sv, // ZWJ (U+200D) adjacency
    "(word"_sv,             // OP open-punctuation before
    "word)"_sv,             // CL close-punctuation after
    "co-op"_sv,             // HY hyphen
    "a\xC2\xA0" "b"_sv,     // GL glue NBSP (U+00A0)
    "a b"_sv,               // BA break-after space
    "word!"_sv,             // EX exclamation
    "\"word\""_sv,          // QU quotation
    "a:b"_sv,               // IS infix separator
    "12345"_sv,             // NU numbers
    "$50"_sv,               // PR prefix
    "3,000"_sv,             // numeric grouping comma
};
static constexpr std::size_t utf8_linewraps_motifs_count = sizeof(utf8_linewraps_motifs) /
                                                           sizeof(utf8_linewraps_motifs[0]);

/** @brief Mandatory-break-dense line run cycling CRLF/U+2028/U+2029/U+000B, @p link_count cycles (LB4/5). */
static std::string utf8_linewraps_dense_mandatory_breaks_(std::size_t link_count) {
    std::string text;
    for (std::size_t index = 0; index != link_count; ++index) {
        append_codepoint_(text, 0x0061); // 'a'
        switch (index & 0x3u) {
        case 0: text.append("\r\n"); break;              // CRLF
        case 1: append_codepoint_(text, 0x2028); break;  // LINE SEPARATOR
        case 2: append_codepoint_(text, 0x2029); break;  // PARAGRAPH SEPARATOR
        default: append_codepoint_(text, 0x000B); break; // vertical tab (BK)
        }
    }
    return text;
}

/** @brief OP/CL/QU/HY/BA/GL nesting cycled @p link_count times (LB13/14/15/18 adjacency). */
static std::string utf8_linewraps_dense_nesting_(std::size_t link_count) {
    std::string text;
    static char const *cycle[] = {"(", "word", ")", "\"", "-", " ", "\xC2\xA0"}; // OP CL QU HY BA SP GL(NBSP)
    for (std::size_t index = 0; index != link_count; ++index) text.append(cycle[index % 7u]);
    return text;
}

/** @brief Numeric `NU` runs interleaved with IS (`.`) and SY (`/`), @p link_count groups (LB25 numbers). */
static std::string utf8_linewraps_dense_numeric_(std::size_t link_count) {
    std::string text;
    for (std::size_t index = 0; index != link_count; ++index) text.append("1.234/56 ");
    return text;
}

/** @brief The linewrap family's high-density homogeneous runs, sized from @p rng to span several 64-byte windows. */
static std::vector<std::string> utf8_linewraps_dense_runs_(std::mt19937 &rng) {
    std::uniform_int_distribution<std::size_t> wide(60, 220);
    std::size_t const wide_count = wide(rng);
    return {utf8_linewraps_dense_mandatory_breaks_(wide_count), utf8_linewraps_dense_nesting_(wide_count),
            utf8_linewraps_dense_numeric_(wide_count)};
}

/** @brief The linewrap family's long-range straddling constructions for a given @p gap. */
static std::vector<std::string> utf8_linewraps_straddles_(std::mt19937 & /*rng*/, std::size_t gap) {
    return {utf8_linewraps_dense_mandatory_breaks_(gap), utf8_linewraps_dense_nesting_(gap)};
}

/** @brief Assemble the linewrap family's differential corpora (motifs + dense + straddle). */
static utf8_segment_corpora_t utf8_linewraps_corpora_() {
    utf8_segment_corpora_t corpora = {"linewrap",
                                      utf8_linewraps_motifs,
                                      utf8_linewraps_motifs_count,
                                      &utf8_linewraps_dense_runs_,
                                      &utf8_linewraps_straddles_,
                                      nullptr,
                                      0};
    return corpora;
}

#pragma endregion // Equivalence

#pragma region Safety

/** @brief Malformed-input safety of the UTF-8 line kernels (serial / dispatched / icelake). */
void test_utf8_linewraps_safety() {
    std::printf("  - testing malformed-input safety of UTF-8 line kernels...\n");
    check_utf8_segment_safety_("linewrap (serial)", sz_utf8_linewraps_serial);
    check_utf8_segment_safety_("linewrap (dispatched)", sz_utf8_linewraps);
#if SZ_USE_ICELAKE
    check_utf8_segment_safety_("linewrap (icelake)", sz_utf8_linewraps_icelake);
#endif
    std::printf("    linewrap safety passed!\n");
}

#pragma endregion // Safety

#pragma region Drivers

/** @brief Serial-vs-ISA line differential over the hardened corpora (high-density + long-range). */
void test_utf8_linewraps_all() {
#if SZ_USE_ICELAKE
    utf8_segment_corpora_t const corpora = utf8_linewraps_corpora_();
    test_utf8_segment_equivalence_(sz_utf8_linewraps_serial, sz_utf8_linewraps_icelake, corpora);
#endif
}

#pragma endregion // Drivers
