/**
 *  @file test/utf8_tokens.cpp
 *  @author Ash Vardanian
 *  @date November 18, 2025
 *  @brief UTF-8 newline/whitespace boundary equivalence and C++ line/token splitting semantics.
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

#if defined(__SANITIZE_ADDRESS__)
#include <sanitizer/asan_interface.h> // We use ASAN API to poison memory addresses
#endif

#include <cstdio> // `stderr`
#include <cstring> // `std::memcpy`

#include <algorithm> // `std::transform`
#include <iterator>  // `std::distance`
#include <random>    // `std::random_device`
#include <string>    // Baseline
#include <vector>    // `std::vector`

#include <fmt/format.h>

#include "cross.hpp" // `check_utf8_tokens_unit_`, `check_utf8_delimiters_unit_`, `utf8_tokens_backend_t`

namespace sz = ashvardanian::stringzilla;
using namespace sz::test;
using sz::literals::operator""_sv; // for `sz::string_view_t`

/** The dispatch points in the shape of a count/newline/whitespace backend. */
static utf8_tokens_backend_t const utf8_tokens_dispatched = {
    "dispatched", cpu_best<sz_utf8_count_best>, cpu_best<sz_utf8_newlines_best>, cpu_best<sz_utf8_whitespaces_best>};

/** The delimiter dispatch point in the shape of a token backend. */
static utf8_delimiters_backend_t const utf8_delimiters_dispatched = {"dispatched", cpu_best<sz_utf8_delimiters_best>};

#pragma region Unit

/**
 *  @brief Known-answer coverage for UTF-8 newline/whitespace boundary detection and the C++
 *      line/token splitting iterators.
 *
 *  Exercises the boundary finders through the dispatched C API, so a regression that the
 *  serial-vs-SIMD agreement tests would miss - because both share a wrong constant - is still
 *  caught against an external ground truth; the same anchors hold every capability's kernels in
 *  the cross files. The C++ @c utf8_lines and @c utf8_tokens checks assert against literal
 *  expected segment lists, not another backend.
 */
void test_utf8_tokens_unit() {
    check_utf8_tokens_unit_(utf8_tokens_dispatched);

    // Split by Unicode newlines
    {
        auto lines = [](sz::string_view_t t) {
            return t.utf8_split_newlines().template to<std::vector<std::string>>();
        };

        // Basic newline types
        let_verify(auto l = lines("a\nb\nc"), l.size() == 3 && l[0] == "a" && l[2] == "c");
        let_verify(auto l = lines("a\r\nb\r\nc"), l.size() == 3 && l[1] == "b");
        let_verify(auto l = lines("a\rb\rc"), l.size() == 3 && l[0] == "a");
        let_verify(auto l = lines("a\r\nb"), l.size() == 2 && l[0] == "a" && l[1] == "b"); // CRLF counts as one newline
        let_verify(auto l = lines("a\r\n\r\nb"), l.size() == 3 && l[0] == "a" && l[1].empty() && l[2] == "b");
        let_verify(auto l = lines("\r\na\r\n\r\nb\r\n"),
                   l.size() == 5 && l[0].empty() && l[1] == "a" && l[2].empty() && l[3] == "b" && l[4].empty());

        // Edge cases - N delimiters yield N+1 segments
        let_verify(auto l = lines(""), l.size() == 1 && l[0] == "");
        let_verify(auto l = lines("\n"), l.size() == 2 && l[0] == "" && l[1] == "");
        let_verify(auto l = lines("\n\n"), l.size() == 3 && l[0] == "" && l[1] == "" && l[2] == "");
        let_verify(auto l = lines("a\n"), l.size() == 2 && l[0] == "a" && l[1] == "");
        let_verify(auto l = lines("\na"), l.size() == 2 && l[0] == "" && l[1] == "a");
        let_verify(auto l = lines("a\nb"), l.size() == 2 && l[0] == "a" && l[1] == "b");
        let_verify(auto l = lines("single"), l.size() == 1 && l[0] == "single");

        // Mixed newlines with non-ASCII content
        let_verify(auto l = lines("Hello 世界\nПривет\r\n😀"),
                   l.size() == 3 && l[0] == "Hello 世界" && l[1] == "Привет" && l[2] == "😀");

        // Multiple line types
        let_verify(auto l = lines("a\nb\r\nc\rd"), l.size() == 4 && l[3] == "d");

        // Unicode line separators (U+2028, U+2029)
        let_verify(auto l = lines("a\xE2\x80\xA8" "b"), l.size() >= 1);
        let_verify(auto l = lines("a\xE2\x80\xA9" "b"), l.size() >= 1);

        // Use `_sv` literals for size-aware NUL-containing strings
        let_verify(auto l = lines("a\x00" "b"_sv),
                   l.size() == 1);                                      // NUL in middle - not a newline
        let_verify(auto l = lines("\x00\x00\x00"_sv), l.size() == 1);   // Only NULs - one "line"
        let_verify(auto l = lines("hello\x00world"_sv), l.size() == 1); // NUL between words - not a newline
        let_verify(auto l = lines("\x00\n"_sv), l.size() == 2); // NUL before newline - find \n, yields 2 segments
        let_verify(auto l = lines("\n\x00"_sv), l.size() == 2); // Newline before NUL - split correctly
    }

    // Test with `sz::string_t` - not just `sz::string_view_t`
    {
        sz::string_t multiline = "a\nb\nc";
        let_verify(auto l = multiline.utf8_split_newlines().template to<std::vector<std::string>>(),
                   l.size() == 3 && l[1] == "b");

        sz::string_t words_str = "foo bar baz";
        let_verify(auto w = words_str.utf8_split_whitespaces().template to<std::vector<std::string>>(),
                   w.size() == 3 && w[2] == "baz");
    }

    // The kernel-named accessors yield the delimiter runs themselves (not the segments between).
    {
        // `utf8_newlines` on "a\nb\r\nc": the "\n" and "\r\n".
        let_verify(auto n = sz::string_view_t("a\nb\r\nc").utf8_newlines().template to<std::vector<std::string>>(),
                   n.size() == 2 && n[0] == "\n" && n[1] == "\r\n");
        // `utf8_whitespaces` on "a b  c": each whitespace codepoint is its own delimiter (runs are not coalesced).
        let_verify(auto w = sz::string_view_t("a b  c").utf8_whitespaces().template to<std::vector<std::string>>(),
                   w.size() == 3 && w[0] == " " && w[1] == " " && w[2] == " ");
    }

    // `.with_separators()` interleaves segments and delimiters losslessly: concatenation reconstructs the input.
    {
        for (sz::string_view_t input :
             {sz::string_view_t("Hi, world"), sz::string_view_t("a\nb\nc"), sz::string_view_t("  x  "),
              sz::string_view_t(""), sz::string_view_t("plain")}) {
            std::string rejoined;
            for (auto piece : input.utf8_split_whitespaces().with_separators())
                rejoined.append(piece.data(), piece.size());
            let_verify(std::string round = rejoined, round == std::string(input.data(), input.size()));
        }
    }

    // `.skip_empty()`: a compile-time, branchless variant that drops empty segments, matching Rust/Python.
    {
        // Whitespace tokens across a double space: "a  b" → "a", "b" (the empty middle dropped).
        let_verify(
            auto t =
                sz::string_view_t("a  b").utf8_split_whitespaces().skip_empty().template to<std::vector<std::string>>(),
            t.size() == 2 && t[0] == "a" && t[1] == "b");
    }
}

/**
 *  @brief Known-answer whitespace-splitting vectors covering all 25 Unicode White_Space characters
 *      by byte length.
 *
 *  Walks the 1-byte ASCII set, the 2-byte NEL/NBSP pair and the 17 three-byte space forms through
 *  the C++ @c utf8_split_whitespaces wrapper, and pins the Format characters U+200B/200C/200D as
 *  not whitespace, so a backend that widens the E2 80 [80-8A] block would shatter ZWJ emoji and
 *  Arabic/Indic words is caught here.
 */
void test_utf8_tokens_scripts_unit() {
    // Split by Unicode whitespace (25 total Unicode White_Space characters)
    {
        auto words = [](sz::string_view_t t) {
            return t.utf8_split_whitespaces().template to<std::vector<std::string>>();
        };

        // Basic ASCII whitespace (6 single-byte chars)
        let_verify(auto w = words("Hello World"), w.size() == 2 && w[0] == "Hello" && w[1] == "World");
        let_verify(auto w = words("a\tb"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+0009 TAB
        let_verify(auto w = words("a\nb"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+000A LF
        let_verify(auto w = words("a\vb"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+000B VT
        let_verify(auto w = words("a\fb"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+000C FF
        let_verify(auto w = words("a\rb"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+000D CR
        let_verify(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b");  // U+0020 SPACE
        let_verify(auto w = words("a\r\nb"),
                   w.size() == 3 && w[0] == "a" && w[1].empty() && w[2] == "b"); // CR and LF are both spaces

        // Multiple spaces - N delimiters yield N+1 segments
        let_verify(auto w = words("  a  b  "), w.size() == 7); // 6 spaces: "" "" "a" "" "b" "" ""
        let_verify(auto w = words("a    b"), w.size() == 5);   // 4 spaces: "a" "" "" "" "b"
        let_verify(auto w = words("a\tb\nc\rd"), w.size() == 4 && w[3] == "d");

        // Double-byte whitespace (2 chars)
        let_verify(auto w = words("a\xC2\x85" "b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+0085 NEL (Next Line)
        let_verify(auto w = words("a\xC2\xA0" "b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+00A0 NBSP (No-Break Space)

        // Triple-byte whitespace (17 chars) - various space widths
        let_verify(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+1680 OGHAM SPACE MARK
        let_verify(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2000 EN QUAD
        let_verify(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2001 EM QUAD
        let_verify(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2002 EN SPACE
        let_verify(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2003 EM SPACE
        let_verify(auto w = words("a b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b");                        // U+2004 THREE-PER-EM SPACE
        let_verify(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2005 FOUR-PER-EM SPACE
        let_verify(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2006 SIX-PER-EM SPACE
        let_verify(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2007 FIGURE SPACE
        let_verify(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2008 PUNCTUATION SPACE
        let_verify(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2009 THIN SPACE
        let_verify(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+200A HAIR SPACE
        let_verify(auto w = words("a\xE2\x80\xA8" "b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2028 LINE SEPARATOR
        let_verify(auto w = words("a\xE2\x80\xA9" "b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2029 PARAGRAPH SEPARATOR
        let_verify(auto w = words("a b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+202F NARROW NO-BREAK SPACE
        let_verify(auto w = words("a b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+205F MEDIUM MATHEMATICAL SPACE
        let_verify(auto w = words("a\xE3\x80\x80" "b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+3000 IDEOGRAPHIC SPACE

        // Mixed byte-length whitespace patterns
        let_verify(auto w = words("a \xC2\xA0" " b"), w.size() == 4);                // 1+2+3 byte mix: "a" "" "" "b"
        let_verify(auto w = words("a\t\xC2\x85" "\xE3\x80\x80" "b"), w.size() == 4); // 1+2+3 byte mix: "a" "" "" "b"
        let_verify(auto w = words("Hello 世界\xC2\xA0" "Привет"), w.size() == 3);    // Unicode content + spaces

        // Edge cases
        let_verify(auto w = words(""), w.size() == 1 && w[0] == "");
        let_verify(auto w = words("   "), w.size() == 4);                    // "" "" "" ""
        let_verify(auto w = words("\t\n\r\v\f"), w.size() == 6);             // All single-byte whitespace
        let_verify(auto w = words("\xC2\x85" "\xC2\xA0" ""), w.size() == 3); // All double-byte whitespace
        let_verify(auto w = words("  \xE3\x80\x80" ""), w.size() == 4);      // All triple-byte whitespace
        let_verify(auto w = words("NoSpaces"), w.size() == 1 && w[0] == "NoSpaces");

        // Non-ASCII content with regular spaces
        let_verify(auto w = words("Hello 世界 Привет 😀"),
                   w.size() == 4 && w[1] == "世界" && w[2] == "Привет" && w[3] == "😀");
        let_verify(auto w = words("مرحبا بك"), w.size() == 2);
        let_verify(auto w = words("שלום עולם"), w.size() == 2);

        // U+001C-U+001F are separators, not whitespace
        let_verify(auto w = words("ab"), w.size() == 1); // FILE SEPARATOR - correctly not split
        let_verify(auto w = words("ab"), w.size() == 1); // GROUP SEPARATOR - correctly not split
        let_verify(auto w = words("ab"), w.size() == 1); // RECORD SEPARATOR - correctly not split
        let_verify(auto w = words("ab"), w.size() == 1); // UNIT SEPARATOR - correctly not split

        // Use `_sv` literals for size-aware NUL-containing strings
        let_verify(auto w = words("a\x00" "b"_sv),
                   w.size() == 1);                                      // NUL in middle - not split
        let_verify(auto w = words("\x00\x00\x00"_sv), w.size() == 1);   // Only NULs - one "word"
        let_verify(auto w = words("hello\x00world"_sv), w.size() == 1); // NUL between words - not split
        let_verify(auto w = words("\x00 a"_sv), w.size() == 2);         // NUL before space - yields 2 segments
        let_verify(auto w = words("a \x00"_sv), w.size() == 2);         // Space before NUL - yields 2 segments

        // U+200B/200C/200D (ZWSP/ZWNJ/ZWJ) are Format characters (Unicode White_Space=No): not
        // whitespace, so a word containing one stays a single segment. A regression here would
        // shatter ZWJ emoji and Arabic/Indic words.
        let_verify(auto w = words("a​b"), w.size() == 1); // ZERO WIDTH SPACE - Format char, not whitespace
        let_verify(auto w = words("a‌b"), w.size() == 1); // ZERO WIDTH NON-JOINER - Format char, not whitespace
        let_verify(auto w = words("a‍b"), w.size() == 1); // ZERO WIDTH JOINER - Format char, not whitespace

        // Consecutive different whitespace types - N delimiters yield N+1 segments
        let_verify(auto w = words("a \t\n\r\vb"), w.size() == 6); // 5 whitespace chars between a and b
        let_verify(auto w = words("a \xC2\xA0" " \xE3\x80\x80" "b"), w.size() == 5); // 1+2+3+3 byte: 4 delims → 5 segs

        // Long sequences to test chunk boundaries - N delimiters yield N+1 segments
        scope_verify(
            std::string long_ws, for (int i = 0; i < 100; ++i) long_ws += " ",
            sz::string_view_t(long_ws).utf8_split_whitespaces().template to<std::vector<std::string>>().size() ==
                101); // 100 spaces = 101 empty segments

        scope_verify(
            std::string long_mixed,
            {
                for (int i = 0; i < 50; ++i) long_mixed += "word ";
                long_mixed.pop_back();
            }, // Remove trailing space
            sz::string_view_t(long_mixed).utf8_split_whitespaces().template to<std::vector<std::string>>().size() ==
                50); // 50 words
    }
}

#pragma endregion Unit

#pragma region Safety

/** Drives the dispatched newline and whitespace finders through the malformed-input battery. */
void test_utf8_tokens_safety(test_context_t &context) { check_utf8_tokens_safety_(context, utf8_tokens_dispatched); }

#pragma endregion Safety

#pragma region Drivers

/** Runs the count/newline/whitespace differential of the dispatch points against serial. */
void test_utf8_tokens_all(test_context_t &context) { check_utf8_tokens_equivalence_(context, utf8_tokens_dispatched); }

#pragma endregion Drivers

#pragma region Unit

/** Known-answer unit tests for the UTF-8 delimiter dispatch point and its C++ range wrappers. */
void test_utf8_delimiters_unit() {
    check_utf8_delimiters_unit_(utf8_delimiters_dispatched);

    // The C++ range wrappers over the same kernel, on the same hand-verifiable inputs.
    {
        // "Hi, world" → delimiters at ',' (byte 2) and ' ' (byte 3): segments "Hi", "", "world".
        let_verify(
            auto d = sz::string_view_t("Hi, world").utf8_split_delimiters().template to<std::vector<std::string>>(),
            d.size() == 3 && d[0] == "Hi" && d[2] == "world");
        // U+2014 EM DASH (E2 80 94) is a delimiter: "a—b" → "a", "b".
        let_verify(auto e = sz::string_view_t("a\xE2\x80\x94" "b")
                                .utf8_split_delimiters()
                                .skip_empty()
                                .template to<std::vector<std::string>>(),
                   e.size() == 2 && e[0] == "a" && e[1] == "b");
        // The kernel-named accessor yields the delimiter runs themselves, not the segments between.
        let_verify(auto r = sz::string_view_t("Hi, world").utf8_delimiters().template to<std::vector<std::string>>(),
                   r.size() == 2 && r[0] == "," && r[1] == " ");
        // `.skip_empty()` drops the empty field between ',' and ' ': "Hi", "world".
        let_verify(auto s = sz::string_view_t("Hi, world")
                                .utf8_split_delimiters()
                                .skip_empty()
                                .template to<std::vector<std::string>>(),
                   s.size() == 2 && s[0] == "Hi" && s[1] == "world");
    }
}

#pragma endregion Unit

#pragma region Drivers

/** Drive the malformed-input safety probe through the delimiter dispatch point. */
void test_utf8_delimiters_safety(test_context_t &context) {
    check_utf8_delimiters_safety_(context, utf8_delimiters_dispatched);
}

/** Drive the serial-vs-dispatched UTF-8 delimiter differential. */
void test_utf8_delimiters_all(test_context_t &context) {
    check_utf8_delimiters_equivalence_(context, utf8_delimiters_dispatched);
}

#pragma endregion Drivers
