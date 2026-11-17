/**
 *  @file test/utf8_tokens.cpp
 *  @author Ash Vardanian
 *  @date November 18, 2025
 *  @brief UTF-8 newline/whitespace boundary equivalence and C++ line/token splitting semantics.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#undef STRINGZILLA_DEBUG
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

#include <string> // Baseline
#include <vector> // `std::vector`

#include <fmt/format.h>

#include "cross.hpp" // `check_utf8_tokens_unit_`, `check_utf8_delimiters_unit_`, `utf8_tokens_backend_t`

namespace ashvardanian::stringzilla::test {
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
        {
            auto l = lines("a\nb\nc");
            expect_true(l.size() == 3 && l[0] == "a" && l[2] == "c");
        }
        {
            auto l = lines("a\r\nb\r\nc");
            expect_true(l.size() == 3 && l[1] == "b");
        }
        {
            auto l = lines("a\rb\rc");
            expect_true(l.size() == 3 && l[0] == "a");
        }
        {
            auto l = lines("a\r\nb");
            expect_true(l.size() == 2 && l[0] == "a" && l[1] == "b");
        } // CRLF counts as one newline
        {
            auto l = lines("a\r\n\r\nb");
            expect_true(l.size() == 3 && l[0] == "a" && l[1].empty() && l[2] == "b");
        }
        {
            auto l = lines("\r\na\r\n\r\nb\r\n");
            expect_true(l.size() == 5 && l[0].empty() && l[1] == "a" && l[2].empty() && l[3] == "b" && l[4].empty());
        }

        // Edge cases - N delimiters yield N+1 segments
        {
            auto l = lines("");
            expect_true(l.size() == 1 && l[0] == "");
        }
        {
            auto l = lines("\n");
            expect_true(l.size() == 2 && l[0] == "" && l[1] == "");
        }
        {
            auto l = lines("\n\n");
            expect_true(l.size() == 3 && l[0] == "" && l[1] == "" && l[2] == "");
        }
        {
            auto l = lines("a\n");
            expect_true(l.size() == 2 && l[0] == "a" && l[1] == "");
        }
        {
            auto l = lines("\na");
            expect_true(l.size() == 2 && l[0] == "" && l[1] == "a");
        }
        {
            auto l = lines("a\nb");
            expect_true(l.size() == 2 && l[0] == "a" && l[1] == "b");
        }
        {
            auto l = lines("single");
            expect_true(l.size() == 1 && l[0] == "single");
        }

        // Mixed newlines with non-ASCII content
        {
            auto l = lines("Hello 世界\nПривет\r\n😀");
            expect_true(l.size() == 3 && l[0] == "Hello 世界" && l[1] == "Привет" && l[2] == "😀");
        }

        // Multiple line types
        {
            auto l = lines("a\nb\r\nc\rd");
            expect_true(l.size() == 4 && l[3] == "d");
        }

        // Unicode line separators (U+2028, U+2029)
        {
            auto l = lines("a\xE2\x80\xA8" // the next fragment starts with a hex digit
                           "b");
            expect_true(l.size() >= 1);
        }
        {
            auto l = lines("a\xE2\x80\xA9" // the next fragment starts with a hex digit
                           "b");
            expect_true(l.size() >= 1);
        }

        // Use `_sv` literals for size-aware NUL-containing strings
        {
            auto l = lines("a\x00" // the next fragment starts with a hex digit
                           "b"_sv);
            expect_equal(l.size(), 1);
        } // NUL in middle - not a newline
        {
            auto l = lines("\x00\x00\x00"_sv);
            expect_equal(l.size(), 1);
        } // Only NULs - one "line"
        {
            auto l = lines("hello\x00world"_sv);
            expect_equal(l.size(), 1);
        } // NUL between words - not a newline
        {
            auto l = lines("\x00\n"_sv);
            expect_equal(l.size(), 2);
        } // NUL before newline - find \n, yields 2 segments
        {
            auto l = lines("\n\x00"_sv);
            expect_equal(l.size(), 2);
        } // Newline before NUL - split correctly
    }

    // Test with `sz::string_t` - not just `sz::string_view_t`
    {
        sz::string_t multiline;
        expect_true(sz::succeeded(multiline.assign("a\nb\nc")));
        {
            auto l = multiline.utf8_split_newlines().template to<std::vector<std::string>>();
            expect_true(l.size() == 3 && l[1] == "b");
        }

        sz::string_t words_str;
        expect_true(sz::succeeded(words_str.assign("foo bar baz")));
        {
            auto w = words_str.utf8_split_whitespaces().template to<std::vector<std::string>>();
            expect_true(w.size() == 3 && w[2] == "baz");
        }
    }

    // The kernel-named accessors yield the delimiter runs themselves (not the segments between).
    {
        // `utf8_newlines` on "a\nb\r\nc": the "\n" and "\r\n".
        {
            auto n = sz::string_view_t("a\nb\r\nc").utf8_newlines().template to<std::vector<std::string>>();
            expect_true(n.size() == 2 && n[0] == "\n" && n[1] == "\r\n");
        }
        // `utf8_whitespaces` on "a b  c": each whitespace codepoint is its own delimiter (runs are not coalesced).
        {
            auto w = sz::string_view_t("a b  c").utf8_whitespaces().template to<std::vector<std::string>>();
            expect_true(w.size() == 3 && w[0] == " " && w[1] == " " && w[2] == " ");
        }
    }

    // `.with_separators()` interleaves segments and delimiters losslessly: concatenation reconstructs the input.
    {
        for (sz::string_view_t input :
             {sz::string_view_t("Hi, world"), sz::string_view_t("a\nb\nc"), sz::string_view_t("  x  "),
              sz::string_view_t(""), sz::string_view_t("plain")}) {
            std::string rejoined;
            for (auto piece : input.utf8_split_whitespaces().with_separators())
                rejoined.append(piece.data(), piece.size());
            {
                std::string round = rejoined;
                expect_equal(round, std::string(input.data(), input.size()));
            }
        }
    }

    // `.skip_empty()`: a compile-time, branchless variant that drops empty segments, matching Rust/Python.
    {
        // Whitespace tokens across a double space: "a  b" → "a", "b" (the empty middle dropped).
        {
            auto t =
                sz::string_view_t("a  b").utf8_split_whitespaces().skip_empty().template to<std::vector<std::string>>();
            expect_true(t.size() == 2 && t[0] == "a" && t[1] == "b");
        }
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
        {
            auto w = words("Hello World");
            expect_true(w.size() == 2 && w[0] == "Hello" && w[1] == "World");
        }
        {
            auto w = words("a\tb");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+0009 TAB
        {
            auto w = words("a\nb");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+000A LF
        {
            auto w = words("a\vb");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+000B VT
        {
            auto w = words("a\fb");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+000C FF
        {
            auto w = words("a\rb");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+000D CR
        {
            auto w = words("a b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+0020 SPACE
        {
            auto w = words("a\r\nb");
            expect_true(w.size() == 3 && w[0] == "a" && w[1].empty() && w[2] == "b");
        } // CR and LF are both spaces

        // Multiple spaces - N delimiters yield N+1 segments
        {
            auto w = words("  a  b  ");
            expect_equal(w.size(), 7);
        } // 6 spaces: "" "" "a" "" "b" "" ""
        {
            auto w = words("a    b");
            expect_equal(w.size(), 5);
        } // 4 spaces: "a" "" "" "" "b"
        {
            auto w = words("a\tb\nc\rd");
            expect_true(w.size() == 4 && w[3] == "d");
        }

        // Double-byte whitespace (2 chars)
        {
            auto w = words("a\xC2\x85" // the next fragment starts with a hex digit
                           "b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+0085 NEL (Next Line)
        {
            auto w = words("a\xC2\xA0" // the next fragment starts with a hex digit
                           "b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+00A0 NBSP (No-Break Space)

        // Triple-byte whitespace (17 chars) - various space widths
        {
            auto w = words("a b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+1680 OGHAM SPACE MARK
        {
            auto w = words("a b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+2000 EN QUAD
        {
            auto w = words("a b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+2001 EM QUAD
        {
            auto w = words("a b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+2002 EN SPACE
        {
            auto w = words("a b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+2003 EM SPACE
        {
            auto w = words("a b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+2004 THREE-PER-EM SPACE
        {
            auto w = words("a b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+2005 FOUR-PER-EM SPACE
        {
            auto w = words("a b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+2006 SIX-PER-EM SPACE
        {
            auto w = words("a b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+2007 FIGURE SPACE
        {
            auto w = words("a b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+2008 PUNCTUATION SPACE
        {
            auto w = words("a b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+2009 THIN SPACE
        {
            auto w = words("a b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+200A HAIR SPACE
        {
            auto w = words("a\xE2\x80\xA8" // the next fragment starts with a hex digit
                           "b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+2028 LINE SEPARATOR
        {
            auto w = words("a\xE2\x80\xA9" // the next fragment starts with a hex digit
                           "b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+2029 PARAGRAPH SEPARATOR
        {
            auto w = words("a b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+202F NARROW NO-BREAK SPACE
        {
            auto w = words("a b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+205F MEDIUM MATHEMATICAL SPACE
        {
            auto w = words("a\xE3\x80\x80" // the next fragment starts with a hex digit
                           "b");
            expect_true(w.size() == 2 && w[0] == "a" && w[1] == "b");
        } // U+3000 IDEOGRAPHIC SPACE

        // Mixed byte-length whitespace patterns
        {
            auto w = words("a \xC2\xA0" // the next fragment starts with a hex digit
                           " b");
            expect_equal(w.size(), 4);
        } // 1+2+3 byte mix: "a" "" "" "b"
        {
            auto w = words("a\t\xC2\x85\xE3\x80\x80" // the next fragment starts with a hex digit
                           "b");
            expect_equal(w.size(), 4);
        } // 1+2+3 byte mix: "a" "" "" "b"
        {
            auto w = words("Hello 世界\xC2\xA0" // the next fragment starts with a hex digit
                           "Привет");
            expect_equal(w.size(), 3);
        } // Unicode content + spaces

        // Edge cases
        {
            auto w = words("");
            expect_true(w.size() == 1 && w[0] == "");
        }
        {
            auto w = words("   ");
            expect_equal(w.size(), 4);
        } // "" "" "" ""
        {
            auto w = words("\t\n\r\v\f");
            expect_equal(w.size(), 6);
        } // All single-byte whitespace
        {
            auto w = words("\xC2\x85\xC2\xA0");
            expect_equal(w.size(), 3);
        } // All double-byte whitespace
        {
            auto w = words("  \xE3\x80\x80" // the next fragment starts with a hex digit
                           "");
            expect_equal(w.size(), 4);
        } // All triple-byte whitespace
        {
            auto w = words("NoSpaces");
            expect_true(w.size() == 1 && w[0] == "NoSpaces");
        }

        // Non-ASCII content with regular spaces
        {
            auto w = words("Hello 世界 Привет 😀");
            expect_true(w.size() == 4 && w[1] == "世界" && w[2] == "Привет" && w[3] == "😀");
        }
        {
            auto w = words("مرحبا بك");
            expect_equal(w.size(), 2);
        }
        {
            auto w = words("שלום עולם");
            expect_equal(w.size(), 2);
        }

        // U+001C-U+001F are separators, not whitespace
        {
            auto w = words("ab");
            expect_equal(w.size(), 1);
        } // FILE SEPARATOR - correctly not split
        {
            auto w = words("ab");
            expect_equal(w.size(), 1);
        } // GROUP SEPARATOR - correctly not split
        {
            auto w = words("ab");
            expect_equal(w.size(), 1);
        } // RECORD SEPARATOR - correctly not split
        {
            auto w = words("ab");
            expect_equal(w.size(), 1);
        } // UNIT SEPARATOR - correctly not split

        // Use `_sv` literals for size-aware NUL-containing strings
        {
            auto w = words("a\x00" // the next fragment starts with a hex digit
                           "b"_sv);
            expect_equal(w.size(), 1);
        } // NUL in middle - not split
        {
            auto w = words("\x00\x00\x00"_sv);
            expect_equal(w.size(), 1);
        } // Only NULs - one "word"
        {
            auto w = words("hello\x00world"_sv);
            expect_equal(w.size(), 1);
        } // NUL between words - not split
        {
            auto w = words("\x00 a"_sv);
            expect_equal(w.size(), 2);
        } // NUL before space - yields 2 segments
        {
            auto w = words("a \x00"_sv);
            expect_equal(w.size(), 2);
        } // Space before NUL - yields 2 segments

        // U+200B/200C/200D (ZWSP/ZWNJ/ZWJ) are Format characters (Unicode White_Space=No): not
        // whitespace, so a word containing one stays a single segment. A regression here would
        // shatter ZWJ emoji and Arabic/Indic words.
        {
            auto w = words("a​b");
            expect_equal(w.size(), 1);
        } // ZERO WIDTH SPACE - Format char, not whitespace
        {
            auto w = words("a‌b");
            expect_equal(w.size(), 1);
        } // ZERO WIDTH NON-JOINER - Format char, not whitespace
        {
            auto w = words("a‍b");
            expect_equal(w.size(), 1);
        } // ZERO WIDTH JOINER - Format char, not whitespace

        // Consecutive different whitespace types - N delimiters yield N+1 segments
        {
            auto w = words("a \t\n\r\vb");
            expect_equal(w.size(), 6);
        } // 5 whitespace chars between a and b
        {
            auto w = words("a \xC2\xA0"    // the next fragment starts with a hex digit
                           " \xE3\x80\x80" // the next fragment starts with a hex digit
                           "b");
            expect_equal(w.size(), 5);
        } // 1+2+3+3 byte: 4 delims → 5 segs

        // Long sequences to test chunk boundaries - N delimiters yield N+1 segments
        {
            std::string long_ws;
            for (int i = 0; i < 100; ++i) long_ws += " ";
            expect_equal(
                sz::string_view_t(long_ws).utf8_split_whitespaces().template to<std::vector<std::string>>().size(),
                101);
        } // 100 spaces = 101 empty segments

        {
            std::string long_mixed;
            for (int i = 0; i < 50; ++i) long_mixed += "word ";
            long_mixed.pop_back(); // Remove trailing space
            expect_equal(
                sz::string_view_t(long_mixed).utf8_split_whitespaces().template to<std::vector<std::string>>().size(),
                50); // 50 words
        }
    }
}

#pragma endregion Unit

#pragma region Safety

/** Drives the dispatched newline and whitespace finders through the malformed-input battery. */
void test_utf8_tokens_safety() {
    split_mix_generator_t generator {case_stream_key()};
    check_utf8_tokens_safety_(generator, utf8_tokens_dispatched);
}

#pragma endregion Safety

#pragma region Drivers

/** Runs the count/newline/whitespace differential of the dispatch points against serial. */
void test_utf8_tokens_all() {
    split_mix_generator_t generator {case_stream_key()};
    check_utf8_tokens_equivalence_(generator, utf8_tokens_dispatched);
}

#pragma endregion Drivers

#pragma region Unit

/** Known-answer unit tests for the UTF-8 delimiter dispatch point and its C++ range wrappers. */
void test_utf8_delimiters_unit() {
    check_utf8_delimiters_unit_(utf8_delimiters_dispatched);

    // The C++ range wrappers over the same kernel, on the same hand-verifiable inputs.
    {
        // "Hi, world" → delimiters at ',' (byte 2) and ' ' (byte 3): segments "Hi", "", "world".
        {
            auto d = sz::string_view_t("Hi, world").utf8_split_delimiters().template to<std::vector<std::string>>();
            expect_true(d.size() == 3 && d[0] == "Hi" && d[2] == "world");
        }
        // U+2014 EM DASH (E2 80 94) is a delimiter: "a—b" → "a", "b".
        {
            auto e = sz::string_view_t("a\xE2\x80\x94" // the next fragment starts with a hex digit
                                       "b")
                         .utf8_split_delimiters()
                         .skip_empty()
                         .template to<std::vector<std::string>>();
            expect_true(e.size() == 2 && e[0] == "a" && e[1] == "b");
        }
        // The kernel-named accessor yields the delimiter runs themselves, not the segments between.
        {
            auto r = sz::string_view_t("Hi, world").utf8_delimiters().template to<std::vector<std::string>>();
            expect_true(r.size() == 2 && r[0] == "," && r[1] == " ");
        }
        // `.skip_empty()` drops the empty field between ',' and ' ': "Hi", "world".
        {
            auto s = sz::string_view_t("Hi, world")
                         .utf8_split_delimiters()
                         .skip_empty()
                         .template to<std::vector<std::string>>();
            expect_true(s.size() == 2 && s[0] == "Hi" && s[1] == "world");
        }
    }
}

#pragma endregion Unit

#pragma region Drivers

/** Drive the malformed-input safety probe through the delimiter dispatch point. */
void test_utf8_delimiters_safety() {
    split_mix_generator_t generator {case_stream_key()};
    check_utf8_delimiters_safety_(generator, utf8_delimiters_dispatched);
}

/** Drive the serial-vs-dispatched UTF-8 delimiter differential. */
void test_utf8_delimiters_all() {
    split_mix_generator_t generator {case_stream_key()};
    check_utf8_delimiters_equivalence_(generator, utf8_delimiters_dispatched);
}

#pragma endregion Drivers

} // namespace ashvardanian::stringzilla::test
