/**
 *  @brief  UTF-8 newline/whitespace boundary equivalence and C++ line/token splitting semantics.
 *  @file   scripts/test_utf8_tokens.cpp
 *  @author Ash Vardanian
 *  @date June 16, 2026
 */
#undef NDEBUG // ! Enable all assertions for testing

/**
 *  The Visual C++ run-time library detects incorrect iterator use,
 *  and asserts and displays a dialog box at run time on Windows.
 */
#if !defined(_ITERATOR_DEBUG_LEVEL) || _ITERATOR_DEBUG_LEVEL == 0
#define _ITERATOR_DEBUG_LEVEL 1
#endif

/**
 *  ! Overload the following with caution.
 *  ! Those parameters must never be explicitly set during releases,
 *  ! but they come handy during development, if you want to validate
 *  ! different ISA-specific implementations.

 #define SZ_USE_WESTMERE 0
 #define SZ_USE_HASWELL 0
 #define SZ_USE_GOLDMONT 0
 #define SZ_USE_SKYLAKE 0
 #define SZ_USE_ICELAKE 0
 #define SZ_USE_NEON 0
 #define SZ_USE_SVE 0
 #define SZ_USE_SVE2 0
 */
#define SZ_USE_MISALIGNED_LOADS 0
#if defined(SZ_DEBUG)
#undef SZ_DEBUG
#endif
#define SZ_DEBUG 1 // ! Enforce aggressive logging in this translation unit

/**
 *  Make sure to include the StringZilla headers before anything else,
 *  to intercept missing `#include` directives and other issues.
 */
#include <stringzilla/stringzilla.h>   // Primary C API
#include <stringzilla/stringzilla.hpp> // C++ string class replacement

#if defined(__SANITIZE_ADDRESS__)
#include <sanitizer/asan_interface.h> // We use ASAN API to poison memory addresses
#endif

#include <cassert> // C-style assertions
#include <cstdio>  // `std::printf`
#include <cstdlib> // `std::getenv`, `std::strtoul`
#include <cstring> // `std::memcpy`

#include <algorithm>  // `std::transform`
#include <functional> // `std::function` (C++11-clean type erasure for the matcher callable)
#include <iterator>   // `std::distance`
#include <random>     // `std::random_device`
#include <string>     // Baseline
#include <vector>     // `std::vector`

#if !SZ_IS_CPP11_
#error "This test requires C++11 or later."
#endif

#include "stringzilla.hpp" // `global_random_generator`, `random_string`

namespace sz = ashvardanian::stringzilla;
using namespace sz::scripts;
using sz::literals::operator""_sv; // for `sz::string_view`

#pragma region Helpers

/** @brief Prints one labeled hex dump line to `stderr`; used by the malformed-input safety test below. */
static void print_utf8_test_bytes_(char const *label, char const *bytes, std::size_t length) {
    std::fprintf(stderr, "  %s (%zu bytes): ", label, length);
    for (std::size_t index = 0; index < length; ++index) std::fprintf(stderr, "%02X ", (unsigned char)bytes[index]);
    std::fprintf(stderr, "\n");
}

/**
 *  @brief Runs one UTF-8 backend's counting and boundary-finding kernels over the known-answer anchors
 *         and asserts the produced codepoint count and the emitted newline/whitespace (offset, length)
 *         spans match the expected lists exactly.
 *
 *  Mirrors `check_sha256_unit_` in `test_hash.cpp`: the caller drives it once per backend (dispatched,
 *  serial, and each natively-compiled kernel), so a wrong constant shared by the serial-vs-SIMD agreement
 *  tests is still caught against these external ground-truth vectors.
 *
 *  @param count                 Codepoint counter under test.
 *  @param newlines              Newline boundary finder under test.
 *  @param whitespaces           Whitespace boundary finder under test.
 *  @param count_text            Anchor text whose codepoints are counted.
 *  @param count_length          Byte length of @p count_text.
 *  @param expected_count        Expected codepoint count of @p count_text.
 *  @param newline_text          Anchor text scanned for newline boundaries.
 *  @param newline_length        Byte length of @p newline_text.
 *  @param expected_newlines     Expected (offset, length) pairs of the newline matches.
 *  @param whitespace_text       Anchor text scanned for whitespace boundaries.
 *  @param whitespace_length     Byte length of @p whitespace_text.
 *  @param expected_whitespaces  Expected (offset, length) pairs of the whitespace matches.
 */
static void check_utf8_unit_(                                                             //
    sz_utf8_count_t count, sz_utf8_segmenter_t newlines, sz_utf8_segmenter_t whitespaces, //
    sz_cptr_t count_text, sz_size_t count_length, sz_size_t expected_count,               //
    sz_cptr_t newline_text, sz_size_t newline_length,                                     //
    std::vector<std::pair<sz_size_t, sz_size_t>> const &expected_newlines,                //
    sz_cptr_t whitespace_text, sz_size_t whitespace_length,                               //
    std::vector<std::pair<sz_size_t, sz_size_t>> const &expected_whitespaces) {

    assert(count(count_text, count_length) == expected_count);

    auto check_boundaries = [](sz_utf8_segmenter_t finder, sz_cptr_t text, sz_size_t length,
                               std::vector<std::pair<sz_size_t, sz_size_t>> const &expected) {
        sz_size_t found_offsets[16], found_lengths[16], consumed = 0;
        sz_size_t const found = finder(text, length, found_offsets, found_lengths, 16u, &consumed);
        assert(found == expected.size());
        for (sz_size_t index = 0; index != found; ++index) {
            assert(found_offsets[index] == expected[index].first);
            assert(found_lengths[index] == expected[index].second);
        }
    };
    check_boundaries(newlines, newline_text, newline_length, expected_newlines);
    check_boundaries(whitespaces, whitespace_text, whitespace_length, expected_whitespaces);
}

#pragma endregion // Helpers

#pragma region Unit

/**
 *  @brief Known-answer coverage for UTF-8 newline/whitespace boundary detection and the C++ line/token
 *         splitting iterators.
 *
 *  Exercises the boundary finders through the dispatched C API (automatic kernel resolution) and through
 *  the natively-compiled backend kernels directly (manual propagation to a specific kernel), so a
 *  regression that the serial-vs-SIMD agreement tests would miss - because both share a wrong constant -
 *  is still caught against an external ground truth. The C++ `utf8_lines`/`utf8_tokens` checks assert
 *  against literal expected segment lists, never against another backend.
 */
void test_utf8_tokens_unit() {
    std::printf("  - testing UTF-8 newline/whitespace known-answer vectors...\n");

    // The mixed-script anchor: "aß中" is `a` (1 byte) + `ß` U+00DF (2 bytes) + `中` U+4E2D (3 bytes),
    // so 6 bytes encode exactly 3 codepoints {0x61, 0xDF, 0x4E2D}.
    char const mixed[] = "a\xC3\x9F\xE4\xB8\xAD"; // "aß中"
    sz_size_t const mixed_length = (sz_size_t)(sizeof(mixed) - 1);
    assert(mixed_length == 6u);

    // `sz_utf8_newlines`: in "a\nb\r\nc" the `\n` is a length-1 newline at byte 1, and the `\r\n` is a
    // single length-2 newline at byte 3 (CRLF merges into one match).
    char const newline_text[] = "a\nb\r\nc";
    sz_size_t const newline_length = (sz_size_t)(sizeof(newline_text) - 1);
    std::vector<std::pair<sz_size_t, sz_size_t>> const newline_spans = {{1u, 1u}, {3u, 2u}};

    // `sz_utf8_whitespaces`: in "a b\tc" the space is a length-1 match at byte 1, the tab at byte 3
    // (there is no CRLF merging in the whitespace set - each codepoint is its own match).
    char const whitespace_text[] = "a b\tc";
    sz_size_t const whitespace_length = (sz_size_t)(sizeof(whitespace_text) - 1);
    std::vector<std::pair<sz_size_t, sz_size_t>> const whitespace_spans = {{1u, 1u}, {3u, 1u}};

    // `sz_utf8_count` (6 bytes, 3 codepoints) plus the newline/whitespace boundary anchors, driven through
    // the dispatched (automatic kernel), serial, and each natively-compiled backend.
    check_utf8_unit_(sz_utf8_count, sz_utf8_newlines, sz_utf8_whitespaces, // Dispatched
                     mixed, mixed_length, 3u, newline_text, newline_length, newline_spans, whitespace_text,
                     whitespace_length, whitespace_spans);
    check_utf8_unit_(sz_utf8_count_serial, sz_utf8_newlines_serial, sz_utf8_whitespaces_serial, // serial
                     mixed, mixed_length, 3u, newline_text, newline_length, newline_spans, whitespace_text,
                     whitespace_length, whitespace_spans);
#if SZ_USE_HASWELL
    check_utf8_unit_(sz_utf8_count_haswell, sz_utf8_newlines_haswell, sz_utf8_whitespaces_haswell, // haswell
                     mixed, mixed_length, 3u, newline_text, newline_length, newline_spans, whitespace_text,
                     whitespace_length, whitespace_spans);
#endif
#if SZ_USE_ICELAKE
    check_utf8_unit_(sz_utf8_count_icelake, sz_utf8_newlines_icelake, sz_utf8_whitespaces_icelake, // icelake
                     mixed, mixed_length, 3u, newline_text, newline_length, newline_spans, whitespace_text,
                     whitespace_length, whitespace_spans);
#endif
#if SZ_USE_NEON
    check_utf8_unit_(sz_utf8_count_neon, sz_utf8_newlines_neon, sz_utf8_whitespaces_neon, // neon
                     mixed, mixed_length, 3u, newline_text, newline_length, newline_spans, whitespace_text,
                     whitespace_length, whitespace_spans);
#endif
#if SZ_USE_SVE2
    check_utf8_unit_(sz_utf8_count_sve2, sz_utf8_newlines_sve2, sz_utf8_whitespaces_sve2, // sve2
                     mixed, mixed_length, 3u, newline_text, newline_length, newline_spans, whitespace_text,
                     whitespace_length, whitespace_spans);
#endif
#if SZ_USE_V128
    check_utf8_unit_(sz_utf8_count_v128, sz_utf8_newlines_v128, sz_utf8_whitespaces_v128, // v128
                     mixed, mixed_length, 3u, newline_text, newline_length, newline_spans, whitespace_text,
                     whitespace_length, whitespace_spans);
#endif
#if SZ_USE_RVV
    check_utf8_unit_(sz_utf8_count_rvv, sz_utf8_newlines_rvv, sz_utf8_whitespaces_rvv, // rvv
                     mixed, mixed_length, 3u, newline_text, newline_length, newline_spans, whitespace_text,
                     whitespace_length, whitespace_spans);
#endif
#if SZ_USE_POWERVSX
    check_utf8_unit_(sz_utf8_count_powervsx, sz_utf8_newlines_powervsx, sz_utf8_whitespaces_powervsx, // powervsx
                     mixed, mixed_length, 3u, newline_text, newline_length, newline_spans, whitespace_text,
                     whitespace_length, whitespace_spans);
#endif
#if SZ_USE_LASX
    check_utf8_unit_(sz_utf8_count_lasx, sz_utf8_newlines_lasx, sz_utf8_whitespaces_lasx, // lasx
                     mixed, mixed_length, 3u, newline_text, newline_length, newline_spans, whitespace_text,
                     whitespace_length, whitespace_spans);
#endif

    // Split by Unicode newlines
    {
        auto lines = [](sz::string_view t) { return t.utf8_lines().template to<std::vector<std::string>>(); };

        // Basic newline types
        let_assert(auto l = lines("a\nb\nc"), l.size() == 3 && l[0] == "a" && l[2] == "c");
        let_assert(auto l = lines("a\r\nb\r\nc"), l.size() == 3 && l[1] == "b");
        let_assert(auto l = lines("a\rb\rc"), l.size() == 3 && l[0] == "a");
        let_assert(auto l = lines("a\r\nb"), l.size() == 2 && l[0] == "a" && l[1] == "b"); // CRLF counts as one newline
        let_assert(auto l = lines("a\r\n\r\nb"), l.size() == 3 && l[0] == "a" && l[1].empty() && l[2] == "b");
        let_assert(auto l = lines("\r\na\r\n\r\nb\r\n"),
                   l.size() == 5 && l[0].empty() && l[1] == "a" && l[2].empty() && l[3] == "b" && l[4].empty());

        // Edge cases - N delimiters yield N+1 segments
        let_assert(auto l = lines(""), l.size() == 1 && l[0] == "");
        let_assert(auto l = lines("\n"), l.size() == 2 && l[0] == "" && l[1] == "");
        let_assert(auto l = lines("\n\n"), l.size() == 3 && l[0] == "" && l[1] == "" && l[2] == "");
        let_assert(auto l = lines("a\n"), l.size() == 2 && l[0] == "a" && l[1] == "");
        let_assert(auto l = lines("\na"), l.size() == 2 && l[0] == "" && l[1] == "a");
        let_assert(auto l = lines("a\nb"), l.size() == 2 && l[0] == "a" && l[1] == "b");
        let_assert(auto l = lines("single"), l.size() == 1 && l[0] == "single");

        // Mixed newlines with non-ASCII content
        let_assert(auto l = lines("Hello 世界\nПривет\r\n😀"),
                   l.size() == 3 && l[0] == "Hello 世界" && l[1] == "Привет" && l[2] == "😀");

        // Multiple line types
        let_assert(auto l = lines("a\nb\r\nc\rd"), l.size() == 4 && l[3] == "d");

        // Unicode line separators (U+2028, U+2029)
        let_assert(auto l = lines("a\xE2\x80\xA8" "b"), l.size() >= 1);
        let_assert(auto l = lines("a\xE2\x80\xA9" "b"), l.size() >= 1);

        // Use `_sv` literals for size-aware NUL-containing strings
        let_assert(auto l = lines("a\x00" "b"_sv),
                   l.size() == 1);                                      // NUL in middle - NOT a newline
        let_assert(auto l = lines("\x00\x00\x00"_sv), l.size() == 1);   // Only NULs - one "line"
        let_assert(auto l = lines("hello\x00world"_sv), l.size() == 1); // NUL between words - NOT a newline
        let_assert(auto l = lines("\x00\n"_sv), l.size() == 2); // NUL before newline - find \n, yields 2 segments
        let_assert(auto l = lines("\n\x00"_sv), l.size() == 2); // Newline before NUL - split correctly
    }

    // Split by Unicode whitespace (25 total Unicode White_Space characters)
    {
        auto words = [](sz::string_view t) { return t.utf8_tokens().template to<std::vector<std::string>>(); };

        // Basic ASCII whitespace (6 single-byte chars)
        let_assert(auto w = words("Hello World"), w.size() == 2 && w[0] == "Hello" && w[1] == "World");
        let_assert(auto w = words("a\tb"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+0009 TAB
        let_assert(auto w = words("a\nb"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+000A LF
        let_assert(auto w = words("a\vb"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+000B VT
        let_assert(auto w = words("a\fb"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+000C FF
        let_assert(auto w = words("a\rb"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+000D CR
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b");  // U+0020 SPACE
        let_assert(auto w = words("a\r\nb"),
                   w.size() == 3 && w[0] == "a" && w[1].empty() && w[2] == "b"); // CR and LF are both spaces

        // Multiple spaces - N delimiters yield N+1 segments
        let_assert(auto w = words("  a  b  "), w.size() == 7); // 6 spaces: "" "" "a" "" "b" "" ""
        let_assert(auto w = words("a    b"), w.size() == 5);   // 4 spaces: "a" "" "" "" "b"
        let_assert(auto w = words("a\tb\nc\rd"), w.size() == 4 && w[3] == "d");

        // Double-byte whitespace (2 chars)
        let_assert(auto w = words("a\xC2\x85" "b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+0085 NEL (Next Line)
        let_assert(auto w = words("a\xC2\xA0" "b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+00A0 NBSP (No-Break Space)

        // Triple-byte whitespace (17 chars) - various space widths
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+1680 OGHAM SPACE MARK
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2000 EN QUAD
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2001 EM QUAD
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2002 EN SPACE
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2003 EM SPACE
        let_assert(auto w = words("a b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b");                        // U+2004 THREE-PER-EM SPACE
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2005 FOUR-PER-EM SPACE
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2006 SIX-PER-EM SPACE
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2007 FIGURE SPACE
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2008 PUNCTUATION SPACE
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2009 THIN SPACE
        let_assert(auto w = words("a b"), w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+200A HAIR SPACE
        let_assert(auto w = words("a\xE2\x80\xA8" "b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2028 LINE SEPARATOR
        let_assert(auto w = words("a\xE2\x80\xA9" "b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+2029 PARAGRAPH SEPARATOR
        let_assert(auto w = words("a b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+202F NARROW NO-BREAK SPACE
        let_assert(auto w = words("a b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+205F MEDIUM MATHEMATICAL SPACE
        let_assert(auto w = words("a\xE3\x80\x80" "b"),
                   w.size() == 2 && w[0] == "a" && w[1] == "b"); // U+3000 IDEOGRAPHIC SPACE

        // Mixed byte-length whitespace patterns
        let_assert(auto w = words("a \xC2\xA0" " b"), w.size() == 4);                // 1+2+3 byte mix: "a" "" "" "b"
        let_assert(auto w = words("a\t\xC2\x85" "\xE3\x80\x80" "b"), w.size() == 4); // 1+2+3 byte mix: "a" "" "" "b"
        let_assert(auto w = words("Hello 世界\xC2\xA0" "Привет"), w.size() == 3);    // Unicode content + spaces

        // Edge cases
        let_assert(auto w = words(""), w.size() == 1 && w[0] == "");
        let_assert(auto w = words("   "), w.size() == 4);                    // "" "" "" ""
        let_assert(auto w = words("\t\n\r\v\f"), w.size() == 6);             // All single-byte whitespace
        let_assert(auto w = words("\xC2\x85" "\xC2\xA0" ""), w.size() == 3); // All double-byte whitespace
        let_assert(auto w = words("  \xE3\x80\x80" ""), w.size() == 4);      // All triple-byte whitespace
        let_assert(auto w = words("NoSpaces"), w.size() == 1 && w[0] == "NoSpaces");

        // Non-ASCII content with regular spaces
        let_assert(auto w = words("Hello 世界 Привет 😀"),
                   w.size() == 4 && w[1] == "世界" && w[2] == "Привет" && w[3] == "😀");
        let_assert(auto w = words("مرحبا بك"), w.size() == 2);
        let_assert(auto w = words("שלום עולם"), w.size() == 2);

        // U+001C-U+001F are separators, not whitespace
        let_assert(auto w = words("ab"), w.size() == 1); // FILE SEPARATOR - correctly NOT split
        let_assert(auto w = words("ab"), w.size() == 1); // GROUP SEPARATOR - correctly NOT split
        let_assert(auto w = words("ab"), w.size() == 1); // RECORD SEPARATOR - correctly NOT split
        let_assert(auto w = words("ab"), w.size() == 1); // UNIT SEPARATOR - correctly NOT split

        // Use `_sv` literals for size-aware NUL-containing strings
        let_assert(auto w = words("a\x00" "b"_sv),
                   w.size() == 1);                                      // NUL in middle - NOT split
        let_assert(auto w = words("\x00\x00\x00"_sv), w.size() == 1);   // Only NULs - one "word"
        let_assert(auto w = words("hello\x00world"_sv), w.size() == 1); // NUL between words - NOT split
        let_assert(auto w = words("\x00 a"_sv), w.size() == 2);         // NUL before space - yields 2 segments
        let_assert(auto w = words("a \x00"_sv), w.size() == 2);         // Space before NUL - yields 2 segments

        // U+200B-U+200D are format characters per Unicode, but implementation treats them as whitespace
        // Note: This may be intentional for compatibility, but differs from Unicode "White_Space" property
        let_assert(auto w = words("a​b"), w.size() == 2); // ZERO WIDTH SPACE
        let_assert(auto w = words("a‌b"), w.size() == 2); // ZERO WIDTH NON-JOINER
        let_assert(auto w = words("a‍b"), w.size() == 2); // ZERO WIDTH JOINER

        // Consecutive different whitespace types - N delimiters yield N+1 segments
        let_assert(auto w = words("a \t\n\r\vb"), w.size() == 6); // 5 whitespace chars between a and b
        let_assert(auto w = words("a \xC2\xA0" " \xE3\x80\x80" "b"), w.size() == 5); // 1+2+3+3 byte: 4 delims -> 5 segs

        // Long sequences to test chunk boundaries - N delimiters yield N+1 segments
        scope_assert(std::string long_ws, for (int i = 0; i < 100; ++i) long_ws += " ",
                     sz::string_view(long_ws).utf8_tokens().template to<std::vector<std::string>>().size() ==
                         101); // 100 spaces = 101 empty segments

        scope_assert(
            std::string long_mixed,
            {
                for (int i = 0; i < 50; ++i) long_mixed += "word ";
                long_mixed.pop_back();
            }, // Remove trailing space
            sz::string_view(long_mixed).utf8_tokens().template to<std::vector<std::string>>().size() == 50); // 50 words
    }

    // Test with `sz::string` - not just `sz::string_view`
    {
        sz::string multiline = "a\nb\nc";
        let_assert(auto l = multiline.utf8_lines().template to<std::vector<std::string>>(),
                   l.size() == 3 && l[1] == "b");

        sz::string words_str = "foo bar baz";
        let_assert(auto w = words_str.utf8_tokens().template to<std::vector<std::string>>(),
                   w.size() == 3 && w[2] == "baz");
    }

    // `utf8_delimiters`: split on any punctuation/symbol/separator, the superset of whitespace tokens.
    {
        // "Hi, world" -> delimiters at ',' (byte 2) and ' ' (byte 3): segments "Hi", "", "world".
        let_assert(auto d = sz::string_view("Hi, world").utf8_delimiters().template to<std::vector<std::string>>(),
                   d.size() == 3 && d[0] == "Hi" && d[2] == "world");
        // U+2014 EM DASH (E2 80 94) is a delimiter: "a—b" -> "a", "b".
        let_assert(
            auto e = sz::string_view("a\xE2\x80\x94" "b").utf8_delimiters().template to<std::vector<std::string>>(),
            e.size() == 2 && e[0] == "a" && e[1] == "b");
    }
}

#pragma endregion // Unit

#pragma region Equivalence

/** @brief Wraps the UTF-8 counting and boundary-finding kernels of one backend by their pointers. */
template <sz_utf8_count_t count_, sz_utf8_segmenter_t newlines_, sz_utf8_segmenter_t whitespaces_>
struct utf8_from_sz_ {
    sz_size_t count(sz_cptr_t text, sz_size_t length) const noexcept { return count_(text, length); }
    sz_size_t newlines(sz_cptr_t text, sz_size_t length, sz_size_t *offsets, sz_size_t *lengths, //
                       sz_size_t capacity, sz_size_t *bytes_consumed) const noexcept {
        return newlines_(text, length, offsets, lengths, capacity, bytes_consumed);
    }
    sz_size_t whitespaces(sz_cptr_t text, sz_size_t length, sz_size_t *offsets, sz_size_t *lengths, //
                          sz_size_t capacity, sz_size_t *bytes_consumed) const noexcept {
        return whitespaces_(text, length, offsets, lengths, capacity, bytes_consumed);
    }
};

/**
 *  @brief Tests UTF-8 count/newline/whitespace functions across different SIMD backends against serial.
 *
 *  Generates random strings containing:
 *  - ASCII content (1-byte)
 *  - Multi-byte UTF-8 characters (2, 3, 4-byte) - correct and broken ones
 *  - All 25 Unicode White_Space characters (including all newlines) + CRLF sequences - correct and partial ones
 *
 *  For each generated string, compares:
 *  - sz_utf8_count: character counting
 *  - sz_utf8_find_newline: newline detection (position and matched length)
 *  - sz_utf8_find_whitespace: whitespace detection (position and matched length)
 *
 *  @param reference       Serial reference backend bundle (counting + newline/whitespace boundaries).
 *  @param candidate       ISA-specific backend bundle under test (counting + newline/whitespace boundaries).
 *  @param min_text_length Minimum byte length of each generated string.
 *  @param min_iterations  Number of random strings to generate and check.
 */
template <typename reference_, typename candidate_>
void test_utf8_tokens_equivalence(reference_ reference, candidate_ candidate, //
                                  std::size_t min_text_length = 4000,
                                  std::size_t min_iterations = scale_iterations(10000)) {

    // A boundary finder bound to one backend, type-erased so the helpers below stay C++11-clean (no generic
    // lambda `auto` parameters, which are a C++14 feature).
    using token_matcher_t =
        std::function<sz_size_t(sz_cptr_t, sz_size_t, sz_size_t *, sz_size_t *, sz_size_t, sz_size_t *)>;

    // Enumerate every delimiter via repeated streaming calls (resuming from `bytes_consumed`), so a small
    // `capacity` exercises the SIMD vector loop, its resume logic, and the serial tail handoff.
    auto enumerate = [](token_matcher_t const &matcher, sz_cptr_t data, sz_size_t length, sz_size_t capacity,
                        std::vector<sz_size_t> &offsets, std::vector<sz_size_t> &lengths) {
        offsets.clear(), lengths.clear();
        std::vector<sz_size_t> batch_offsets(capacity), batch_lengths(capacity);
        sz_size_t base = 0;
        while (base < length) {
            sz_size_t consumed = 0;
            sz_size_t got = matcher(data + base, length - base, batch_offsets.data(), batch_lengths.data(), capacity,
                                    &consumed);
            for (sz_size_t i = 0; i < got; ++i)
                offsets.push_back(base + batch_offsets[i]), lengths.push_back(batch_lengths[i]);
            if (got == 0 && consumed == 0) break;
            base += consumed;
        }
    };

    // Reconstruct the SEGMENTS the C++/Python/Rust split iterators would yield (gap before each delimiter, plus the
    // trailing segment to end-of-text), advancing the suffix by `bytes_consumed`. This is the consumer's view: it
    // catches a `bytes_consumed` that overshoots the last emitted delimiter (which the raw-delimiter enumeration
    // above cannot see, since the skipped span has no delimiters), e.g. a batch that fills exactly at a window edge.
    auto reconstruct_segments = [](token_matcher_t const &matcher, sz_cptr_t data, sz_size_t length, sz_size_t capacity,
                                   std::vector<sz_size_t> &seg_offsets, std::vector<sz_size_t> &seg_lengths) {
        seg_offsets.clear(), seg_lengths.clear();
        std::vector<sz_size_t> batch_offsets(capacity), batch_lengths(capacity);
        sz_size_t suffix = 0;
        for (;;) {
            sz_size_t region = length - suffix, consumed = 0;
            sz_size_t delimiters = matcher(data + suffix, region, batch_offsets.data(), batch_lengths.data(), capacity,
                                           &consumed);
            sz_size_t previous_end = 0;
            for (sz_size_t i = 0; i < delimiters; ++i) {
                seg_offsets.push_back(suffix + previous_end), seg_lengths.push_back(batch_offsets[i] - previous_end);
                previous_end = batch_offsets[i] + batch_lengths[i];
            }
            if (consumed == region) { // exhausted: the trailing segment runs to end-of-text
                seg_offsets.push_back(suffix + previous_end), seg_lengths.push_back(region - previous_end);
                break;
            }
            suffix += consumed;
            if (consumed == 0) break;
        }
    };

    // Adapt the bundle methods to the plain boundary-finder signature the enumerators expect.
    auto reference_newlines = [&](sz_cptr_t data, sz_size_t length, sz_size_t *offsets, sz_size_t *lengths,
                                  sz_size_t capacity, sz_size_t *bytes_consumed) {
        return reference.newlines(data, length, offsets, lengths, capacity, bytes_consumed);
    };
    auto candidate_newlines = [&](sz_cptr_t data, sz_size_t length, sz_size_t *offsets, sz_size_t *lengths,
                                  sz_size_t capacity, sz_size_t *bytes_consumed) {
        return candidate.newlines(data, length, offsets, lengths, capacity, bytes_consumed);
    };
    auto reference_whitespaces = [&](sz_cptr_t data, sz_size_t length, sz_size_t *offsets, sz_size_t *lengths,
                                     sz_size_t capacity, sz_size_t *bytes_consumed) {
        return reference.whitespaces(data, length, offsets, lengths, capacity, bytes_consumed);
    };
    auto candidate_whitespaces = [&](sz_cptr_t data, sz_size_t length, sz_size_t *offsets, sz_size_t *lengths,
                                     sz_size_t capacity, sz_size_t *bytes_consumed) {
        return candidate.whitespaces(data, length, offsets, lengths, capacity, bytes_consumed);
    };

    auto check = [&](sz_cptr_t data, sz_size_t length) {
        // Test `sz_utf8_count` equivalence
        sz_size_t count_result_reference = reference.count(data, length);
        sz_size_t count_result_candidate = candidate.count(data, length);
        assert(count_result_reference == count_result_candidate);

        // Sweep capacities: one huge (one-shot), the awkward 65/63 straddling the 64-byte AVX-512 window, the
        // binding default 16, and tiny 3/1 - stressing the per-window / mid-window capacity cut at every boundary.
        sz_size_t const capacities[] = {length + 64, 65, 63, 16, 3, 1};
        std::vector<sz_size_t> reference_offsets, reference_lengths, candidate_offsets, candidate_lengths;
        for (sz_size_t capacity : capacities) {
            if (capacity == 0) continue;
            enumerate(reference_newlines, data, length, capacity, reference_offsets, reference_lengths);
            enumerate(candidate_newlines, data, length, capacity, candidate_offsets, candidate_lengths);
            assert(reference_offsets == candidate_offsets && "Mismatch in newline offsets");
            assert(reference_lengths == candidate_lengths && "Mismatch in newline lengths");

            enumerate(reference_whitespaces, data, length, capacity, reference_offsets, reference_lengths);
            enumerate(candidate_whitespaces, data, length, capacity, candidate_offsets, candidate_lengths);
            assert(reference_offsets == candidate_offsets && "Mismatch in whitespace offsets");
            assert(reference_lengths == candidate_lengths && "Mismatch in whitespace lengths");

            // Segment-level (iterator) equivalence: catches a `bytes_consumed` overshoot at a window-aligned fill.
            reconstruct_segments(reference_newlines, data, length, capacity, reference_offsets, reference_lengths);
            reconstruct_segments(candidate_newlines, data, length, capacity, candidate_offsets, candidate_lengths);
            assert(reference_offsets == candidate_offsets && reference_lengths == candidate_lengths &&
                   "Mismatch in newline segments");
            reconstruct_segments(reference_whitespaces, data, length, capacity, reference_offsets, reference_lengths);
            reconstruct_segments(candidate_whitespaces, data, length, capacity, candidate_offsets, candidate_lengths);
            assert(reference_offsets == candidate_offsets && reference_lengths == candidate_lengths &&
                   "Mismatch in whitespace segments");
        }
    };

    // Strings that shouldn't affect the control flow
    static char const *utf8_content[] = {
        // Various ASCII strings
        "",
        "a",
        "hello",
        "012",
        "3456789",
        // 2-byte Cyrillic П (U+041F), Armenian Ս (U+054D), and Greek Pi π (U+03C0)
        "\xD0\x9F",
        "\xD5\xA5",
        "\xCF\x80",
        // 3-byte characters
        "\xE0\xA4\xB9", // Hindi ह (U+0939)
        "\xE1\x88\xB4", // Ethiopic ሴ (U+1234)
        "\xE2\x9C\x94", // Check mark ✔ (U+2714)
        // 4-byte emojis: U+1F600 (😀), U+1F601 (😁), U+1F602 (😂)
        "\xF0\x9F\x98\x80",
        "\xF0\x9F\x98\x81",
        "\xF0\x9F\x98\x82",
        // Characters with bytes in 0x80-0x8F range (tests unsigned comparison in SIMD)
        "\xE2\x82\x80", // U+2080 SUBSCRIPT ZERO (has 0x80 suffix, NOT whitespace)
        "\xE2\x84\x8A", // U+210A SCRIPT SMALL G (has 0x8A like HAIR SPACE suffix)
        "\xE2\x84\x8D", // U+210D DOUBLE-STRUCK H (has 0x8D suffix)
        // Near-miss characters (same prefix as whitespace but different suffix)
        "\xE2\x80\xB0", // U+2030 PER MILLE SIGN (E2 80 prefix like whitespace range)
        "\xE2\x80\xBB", // U+203B REFERENCE MARK (E2 80 prefix)
        "\xE2\x81\xA0", // U+2060 WORD JOINER (E2 81 prefix like MMSP)
        "\xE3\x80\x81", // U+3001 IDEOGRAPHIC COMMA (E3 80 prefix like IDEOGRAPHIC SPACE)
        "\xE3\x80\x82", // U+3002 IDEOGRAPHIC FULL STOP
        // More 4-byte sequences for boundary handling
        "\xF0\x9F\x8E\x89", // U+1F389 PARTY POPPER 🎉
        "\xF0\x9F\x92\xA9", // U+1F4A9 PILE OF POO 💩
    };

    // Special characters that will affect control flow
    static char const *special_chars[26] = {
        "\x09",         "\x0A",         "\x0B",         "\x0C",         "\x0D",         " ", // 1-byte (6)
        "\xC2\x85",     "\xC2\xA0",     "\r\n",                                              // 2-byte (2)
        "\xE1\x9A\x80", "\xE2\x80\x80", "\xE2\x80\x81", "\xE2\x80\x82", "\xE2\x80\x83",      // 3-byte
        "\xE2\x80\x84", "\xE2\x80\x85", "\xE2\x80\x86", "\xE2\x80\x87", "\xE2\x80\x88",      // 3-byte
        "\xE2\x80\x89", "\xE2\x80\x8A", "\xE2\x80\xA8", "\xE2\x80\xA9", "\xE2\x80\xAF",      // 3-byte
        "\xE2\x81\x9F", "\xE3\x80\x80",                                                      // 3-byte
    };

    auto &rng = global_random_generator();
    std::size_t const utf8_content_count = sizeof(utf8_content) / sizeof(utf8_content[0]);
    std::size_t const special_delimiter_count = sizeof(special_chars) / sizeof(special_chars[0]);
    std::size_t const total_strings_to_sample = utf8_content_count + special_delimiter_count;
    std::uniform_int_distribution<std::size_t> content_dist(0, total_strings_to_sample - 1);

    // Generate and test many random strings
    for (std::size_t iteration = 0; iteration < min_iterations; ++iteration) {
        std::string text;

        // Build up a random string of at least `min_text_length` bytes
        while (text.size() < min_text_length) {
            std::size_t random_content_index = content_dist(rng);
            if (random_content_index < utf8_content_count) { text.append(utf8_content[random_content_index]); }
            else { text.append(special_chars[random_content_index - utf8_content_count]); }
        }
        check(text.data(), text.size());

        // Now, let's replace 10% of bytes in the sequence with a NUL character, thus breaking many valid codepoints
        std::size_t num_bytes_to_corrupt = text.size() / 10;
        std::uniform_int_distribution<std::size_t> byte_index_dist(0, text.size() - 1);
        for (std::size_t i = 0; i < num_bytes_to_corrupt; ++i) {
            std::size_t byte_index = byte_index_dist(rng);
            text[byte_index] = '\0';
        }
        check(text.data(), text.size());

        // Swap 10% of bytes at random positions, creating malformed UTF-8 sequences
        for (std::size_t i = 0; i < num_bytes_to_corrupt; ++i) {
            std::size_t byte_index_1 = byte_index_dist(rng);
            std::size_t byte_index_2 = byte_index_dist(rng);
            std::swap(text[byte_index_1], text[byte_index_2]);
        }
        check(text.data(), text.size());
    }

    // Re-run count/newline/whitespace equivalence with the input placed at every sub-cache-line byte offset,
    // so the SIMD load alignment - not just the content - is swept against the serial reference.
    std::string alignment_probe;
    while (alignment_probe.size() < 256) {
        std::size_t random_content_index = content_dist(rng);
        if (random_content_index < utf8_content_count) { alignment_probe.append(utf8_content[random_content_index]); }
        else { alignment_probe.append(special_chars[random_content_index - utf8_content_count]); }
    }
    for_each_cacheline_offset_(alignment_probe.size(), [&](sz_ptr_t buffer, std::size_t /*offset*/) {
        std::memcpy(buffer, alignment_probe.data(), alignment_probe.size());
        check(buffer, alignment_probe.size());
    });
}

#pragma endregion // Equivalence

#pragma region Safety

/**
 *  @brief Drives the UTF-8 newline/whitespace boundary finders through the malformed-input battery
 *         (named adversarial shapes, all 256 single bytes, all 65,536 byte pairs, and random garbage at
 *         every sub-cache-line alignment), asserting they survive, stay in bounds, and never report a
 *         `bytes_consumed` past the input.
 */
void test_utf8_tokens_safety() {
    std::printf("  - testing malformed-input safety of UTF-8 newline/whitespace kernels...\n");

    static constexpr std::size_t max_input_length = 70;

    // Drive every newline/whitespace boundary finder shipped on this target over one malformed input.
    auto check = [&](char const *input, std::size_t input_length) {
        sz_size_t boundary_offsets[max_input_length + 1], boundary_lengths[max_input_length + 1];
        auto check_boundaries = [&](sz_utf8_segmenter_t finder, char const *finder_name) {
            sz_size_t bytes_consumed = 0;
            sz_size_t const found = finder(input, (sz_size_t)input_length, boundary_offsets, boundary_lengths,
                                           (sz_size_t)(max_input_length + 1), &bytes_consumed);
            assert(bytes_consumed <= input_length && "Boundary finder consumed past the input");
            for (sz_size_t index = 0; index != found; ++index) {
                if (boundary_offsets[index] + boundary_lengths[index] <= input_length) continue;
                std::fprintf(stderr, "%s emitted out-of-bounds boundary (offset=%zu len=%zu, input=%zu)\n", finder_name,
                             (std::size_t)boundary_offsets[index], (std::size_t)boundary_lengths[index], input_length);
                print_utf8_test_bytes_("input", input, input_length);
                assert(false && "Boundary finder emitted a span outside the input");
            }
        };

        // Serial baseline and the dispatched (automatic kernel resolution) entry points face the same contract.
        check_boundaries(sz_utf8_newlines_serial, "serial newline finder");
        check_boundaries(sz_utf8_whitespaces_serial, "serial whitespace finder");
        check_boundaries(sz_utf8_newlines, "dispatched newline finder");
        check_boundaries(sz_utf8_whitespaces, "dispatched whitespace finder");
#if SZ_USE_HASWELL
        check_boundaries(sz_utf8_newlines_haswell, "haswell newline finder");
        check_boundaries(sz_utf8_whitespaces_haswell, "haswell whitespace finder");
#endif
#if SZ_USE_ICELAKE
        check_boundaries(sz_utf8_newlines_icelake, "icelake newline finder");
        check_boundaries(sz_utf8_whitespaces_icelake, "icelake whitespace finder");
#endif
#if SZ_USE_NEON
        check_boundaries(sz_utf8_newlines_neon, "neon newline finder");
        check_boundaries(sz_utf8_whitespaces_neon, "neon whitespace finder");
#endif
#if SZ_USE_SVE2
        check_boundaries(sz_utf8_newlines_sve2, "sve2 newline finder");
        check_boundaries(sz_utf8_whitespaces_sve2, "sve2 whitespace finder");
#endif
#if SZ_USE_V128
        check_boundaries(sz_utf8_newlines_v128, "v128 newline finder");
        check_boundaries(sz_utf8_whitespaces_v128, "v128 whitespace finder");
#endif
#if SZ_USE_RVV
        check_boundaries(sz_utf8_newlines_rvv, "rvv newline finder");
        check_boundaries(sz_utf8_whitespaces_rvv, "rvv whitespace finder");
#endif
#if SZ_USE_LASX
        check_boundaries(sz_utf8_newlines_lasx, "lasx newline finder");
        check_boundaries(sz_utf8_whitespaces_lasx, "lasx whitespace finder");
#endif
#if SZ_USE_POWERVSX
        check_boundaries(sz_utf8_newlines_powervsx, "powervsx newline finder");
        check_boundaries(sz_utf8_whitespaces_powervsx, "powervsx whitespace finder");
#endif
    };

    char input[max_input_length];

    // The named adversarial shapes the task calls out, exercised directly.
    check("\x80", 1);              // Lone continuation byte
    check("\xC0\x80", 2);          // Overlong encoding of NUL
    check("\xED\xA0\x80", 3);      // Surrogate-encoded codepoint (U+D800)
    check("hello\xF0\x9F\x98", 8); // Truncated 4-byte sequence at the very end

    // All 256 single bytes: truncated leads, stray continuations, 0xFE/0xFF.
    for (std::size_t byte = 0; byte != 256; ++byte) {
        input[0] = (char)byte;
        check(input, 1);
    }

    // All 65,536 byte pairs: every lead x continuation interaction, including overlong and surrogate shapes.
    for (std::size_t first_byte = 0; first_byte != 256; ++first_byte)
        for (std::size_t second_byte = 0; second_byte != 256; ++second_byte) {
            input[0] = (char)first_byte;
            input[1] = (char)second_byte;
            check(input, 2);
        }

    // Random garbage buffers spanning whole SIMD chunks, at every sub-cache-line alignment.
    std::size_t const random_inputs = scale_iterations(10000);
    auto &rng = global_random_generator();
    std::uniform_int_distribution<std::size_t> length_distribution(1, max_input_length);
    std::uniform_int_distribution<int> byte_distribution(0, 255);
    for (std::size_t iteration = 0; iteration != random_inputs; ++iteration) {
        std::size_t const input_length = length_distribution(rng);
        for (std::size_t index = 0; index != input_length; ++index) input[index] = (char)byte_distribution(rng);
        for_each_cacheline_offset_(input_length, [&](sz_ptr_t buffer, std::size_t /*offset*/) {
            std::memcpy(buffer, input, input_length);
            check(buffer, input_length);
        });
    }

    std::printf("    malformed-input safety passed!\n");
}

#pragma endregion // Safety

#pragma region Drivers

/** @brief Run the UTF-8 count/newline/whitespace differential against every compiled SIMD backend. */
void test_utf8_tokens_all() {
    using reference_t = utf8_from_sz_<sz_utf8_count_serial, sz_utf8_newlines_serial, sz_utf8_whitespaces_serial>;
#if SZ_USE_HASWELL
    test_utf8_tokens_equivalence(
        reference_t {}, utf8_from_sz_<sz_utf8_count_haswell, sz_utf8_newlines_haswell, sz_utf8_whitespaces_haswell> {});
#endif
#if SZ_USE_ICELAKE
    test_utf8_tokens_equivalence(
        reference_t {}, utf8_from_sz_<sz_utf8_count_icelake, sz_utf8_newlines_icelake, sz_utf8_whitespaces_icelake> {});
#endif
#if SZ_USE_NEON
    test_utf8_tokens_equivalence(reference_t {},
                                 utf8_from_sz_<sz_utf8_count_neon, sz_utf8_newlines_neon, sz_utf8_whitespaces_neon> {});
#endif
#if SZ_USE_SVE2
    test_utf8_tokens_equivalence(reference_t {},
                                 utf8_from_sz_<sz_utf8_count_sve2, sz_utf8_newlines_sve2, sz_utf8_whitespaces_sve2> {});
#endif
#if SZ_USE_V128
    test_utf8_tokens_equivalence(reference_t {},
                                 utf8_from_sz_<sz_utf8_count_v128, sz_utf8_newlines_v128, sz_utf8_whitespaces_v128> {});
#endif
#if SZ_USE_V128RELAXED
    test_utf8_tokens_equivalence(
        reference_t {}, utf8_from_sz_<sz_utf8_count_v128relaxed, sz_utf8_newlines_v128, sz_utf8_whitespaces_v128> {});
#endif
#if SZ_USE_RVV
    test_utf8_tokens_equivalence(reference_t {},
                                 utf8_from_sz_<sz_utf8_count_rvv, sz_utf8_newlines_rvv, sz_utf8_whitespaces_rvv> {});
#endif
#if SZ_USE_LASX
    test_utf8_tokens_equivalence(reference_t {},
                                 utf8_from_sz_<sz_utf8_count_lasx, sz_utf8_newlines_lasx, sz_utf8_whitespaces_lasx> {});
#endif
#if SZ_USE_POWERVSX
    test_utf8_tokens_equivalence(
        reference_t {},
        utf8_from_sz_<sz_utf8_count_powervsx, sz_utf8_newlines_powervsx, sz_utf8_whitespaces_powervsx> {});
#endif
}

#pragma endregion // Drivers

#pragma region Delimiters

/** @brief Append one codepoint to @p text as UTF-8 via `sz_rune_encode` (silently skips invalid runes). */
static void append_codepoint_(std::string &text, sz_rune_t codepoint) {
    sz_u8_t bytes[4];
    sz_rune_length_t const length = sz_rune_encode(codepoint, bytes);
    if (length == sz_rune_invalid_k) return;
    text.append((char const *)bytes, (std::size_t)length);
}

/**
 *  @brief Builds a random, well-formed UTF-8 string whose codepoints span all four byte-widths and every
 *         1->2->3->4 transition, mixing delimiter and non-delimiter codepoints so the scan kernels hit their
 *         mixed-width paths and resolve a match at every alignment.
 */
static std::string random_valid_utf8_(std::size_t target_codepoints, std::mt19937 &rng) {
    static struct {
        sz_rune_t low, high;
    } const ranges[] = {
        {0x000020u, 0x00007Eu}, // 1-byte ASCII printable (space + punctuation + letters + digits)
        {0x0000A1u, 0x0007FFu}, // 2-byte (Latin-1 symbols, Greek/Cyrillic letters)
        {0x000800u, 0x00CFFFu}, // 3-byte (general punctuation, CJK; stays below U+D800 surrogates)
        {0x010000u, 0x0EFFFFu}, // 4-byte (symbols, emoji; avoids the plane-ender noncharacters)
    };
    std::uniform_int_distribution<int> width_pick(0, 3);
    std::string text;
    text.reserve(target_codepoints * 4);
    for (std::size_t index = 0; index != target_codepoints; ++index) {
        int const width = width_pick(rng);
        std::uniform_int_distribution<sz_rune_t> codepoint(ranges[width].low, ranges[width].high);
        std::size_t const before = text.size();
        while (text.size() == before) append_codepoint_(text, codepoint(rng));
    }
    return text;
}

/**
 *  @brief Drain every delimiter a segmenter emits over the whole input, resuming via `bytes_consumed` so an
 *         arbitrarily small @p capacity yields the identical full match list. Offsets are absolute.
 */
static void drain_delimiters_(sz_utf8_segmenter_t finder, sz_cptr_t text, sz_size_t length, sz_size_t capacity,
                              std::vector<sz_size_t> &offsets, std::vector<sz_size_t> &lengths) {
    offsets.clear(), lengths.clear();
    std::vector<sz_size_t> offset_batch(capacity ? capacity : 1), length_batch(capacity ? capacity : 1);
    sz_size_t position = 0;
    while (position < length) {
        sz_size_t consumed = 0;
        sz_size_t const emitted = finder(text + position, length - position, offset_batch.data(), length_batch.data(),
                                         capacity, &consumed);
        for (sz_size_t index = 0; index != emitted; ++index)
            offsets.push_back(position + offset_batch[index]), lengths.push_back(length_batch[index]);
        if (consumed == 0) break; // No forward progress: stop rather than spin.
        position += consumed;
    }
}

/** @brief Known-answer unit tests for the UTF-8 delimiter segmenter on simple, hand-verifiable inputs. */
void test_utf8_delimiters_unit() {
    std::printf("  - testing UTF-8 delimiter known-answer vectors...\n");

    struct {
        char const *text;
        sz_size_t length, expected_offset, expected_length, expected_count;
    } const cases[] = {
        {"abc def", 7, 3, 1, 1},               // ASCII space (Zs) at byte 3
        {"hello", 5, 0, 0, 0},                 // all letters, no delimiter
        {"ab,", 3, 2, 1, 1},                   // ASCII comma (Po) at byte 2
        {"ab\xC2\xA0", 4, 2, 2, 1},            // U+00A0 NO-BREAK SPACE (Zs), 2 bytes at byte 2
        {"ab\xE2\x80\x94", 5, 2, 3, 1},        // U+2014 EM DASH (Pd), 3 bytes at byte 2
        {"ab\xF0\x9F\x98\x80", 6, 2, 4, 1},    // U+1F600 GRINNING FACE (So), 4 bytes at byte 2
        {"a\xC3\x9F\xE4\xB8\xAD", 6, 0, 0, 0}, // a + U+00DF + U+4E2D, all letters
        {"", 0, 0, 0, 0},                      // empty input
    };

    struct {
        sz_utf8_segmenter_t finder;
        char const *name;
    } const backends[] = {
        {sz_utf8_delimiters, "dispatched"},      {sz_utf8_delimiters_serial, "serial"},
#if SZ_USE_HASWELL
        {sz_utf8_delimiters_haswell, "haswell"},
#endif
#if SZ_USE_ICELAKE
        {sz_utf8_delimiters_icelake, "icelake"},
#endif
#if SZ_USE_NEON
        {sz_utf8_delimiters_neon, "neon"},
#endif
    };

    std::vector<sz_size_t> offsets, lengths;
    for (auto const &backend : backends) {
        sz_unused_(backend.name);
        for (auto const &one : cases) {
            drain_delimiters_(backend.finder, one.text, one.length, one.length + 1, offsets, lengths);
            assert(offsets.size() == one.expected_count && "Delimiter count mismatch");
            if (one.expected_count) {
                assert(offsets[0] == one.expected_offset && "Delimiter offset mismatch");
                assert(lengths[0] == one.expected_length && "Delimiter length mismatch");
            }
        }
    }
}

/**
 *  @brief Cross-checks the serial UTF-8 delimiter segmenter against a candidate SIMD backend on random,
 *         well-formed inputs: the full (offset, length) match list must agree, both in one shot and when the
 *         candidate is drained through a tiny capacity so its `bytes_consumed` resume path is exercised.
 */
static void test_utf8_delimiters_equivalence(sz_utf8_segmenter_t finder_serial, sz_utf8_segmenter_t finder_candidate,
                                             sz_size_t inputs) {
    auto &rng = global_random_generator();
    std::vector<sz_size_t> serial_offsets, serial_lengths, candidate_offsets, candidate_lengths, resumed_offsets,
        resumed_lengths;

    auto check = [&](std::string const &text) {
        sz_cptr_t const data = text.data();
        sz_size_t const length = (sz_size_t)text.size();
        drain_delimiters_(finder_serial, data, length, length + 1, serial_offsets, serial_lengths);
        drain_delimiters_(finder_candidate, data, length, length + 1, candidate_offsets, candidate_lengths);
        assert(candidate_offsets == serial_offsets && "Mismatch in delimiter offsets");
        assert(candidate_lengths == serial_lengths && "Mismatch in delimiter lengths");
        // Resume path: a capacity of 3 forces repeated re-entry; the accumulated list must still match.
        drain_delimiters_(finder_candidate, data, length, 3u, resumed_offsets, resumed_lengths);
        assert(resumed_offsets == serial_offsets && "Resume-path delimiter offsets diverged");
        assert(resumed_lengths == serial_lengths && "Resume-path delimiter lengths diverged");
    };

    sz_size_t const ladder[] = {0u, 1u, 2u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 64u, 65u, 100u, 200u};
    for (sz_size_t codepoints : ladder) check(random_valid_utf8_(codepoints, rng));

    std::uniform_int_distribution<std::size_t> codepoint_distribution(0, 96);
    for (sz_size_t iteration = 0; iteration != inputs; ++iteration) {
        std::string const text = random_valid_utf8_(codepoint_distribution(rng), rng);
        for_each_cacheline_offset_(text.size(), [&](sz_ptr_t buffer, std::size_t /*offset*/) {
            std::memcpy(buffer, text.data(), text.size());
            check(std::string(buffer, text.size()));
        });
    }
}

/** @brief Feeds malformed / invalid UTF-8 through one backend, asserting in-bounds, ascending, well-formed output. */
static void check_utf8_delimiters_safety_(sz_utf8_segmenter_t finder,
                                          std::size_t random_inputs = scale_iterations(10000)) {
    static constexpr std::size_t max_input_length = 70;
    std::vector<sz_size_t> offsets, lengths;

    auto check = [&](char const *input, std::size_t input_length) {
        drain_delimiters_(finder, input, (sz_size_t)input_length, (sz_size_t)input_length + 1, offsets, lengths);
        sz_size_t previous_end = 0;
        for (std::size_t index = 0; index != offsets.size(); ++index) {
            assert(lengths[index] >= 1u && lengths[index] <= 4u && "Delimiter matched an impossible byte length");
            assert(offsets[index] + lengths[index] <= input_length && "Delimiter match span outside the input");
            assert(offsets[index] >= previous_end && "Delimiter matches must be ascending and non-overlapping");
            previous_end = offsets[index] + lengths[index];
        }
    };

    char input[max_input_length];
    check("\x80", 1);              // Lone continuation byte
    check("\xC0\x80", 2);          // Overlong encoding of NUL
    check("\xED\xA0\x80", 3);      // Surrogate-encoded codepoint (U+D800)
    check("hello\xF0\x9F\x98", 8); // Truncated 4-byte sequence at the very end

    for (std::size_t byte = 0; byte != 256; ++byte) { input[0] = (char)byte, check(input, 1); }
    for (std::size_t first_byte = 0; first_byte != 256; ++first_byte)
        for (std::size_t second_byte = 0; second_byte != 256; ++second_byte) {
            input[0] = (char)first_byte, input[1] = (char)second_byte;
            check(input, 2);
        }

    auto &rng = global_random_generator();
    std::uniform_int_distribution<std::size_t> length_distribution(1, max_input_length);
    std::uniform_int_distribution<int> byte_distribution(0, 255);
    for (std::size_t iteration = 0; iteration != random_inputs; ++iteration) {
        std::size_t const input_length = length_distribution(rng);
        for (std::size_t index = 0; index != input_length; ++index) input[index] = (char)byte_distribution(rng);
        for_each_cacheline_offset_(input_length, [&](sz_ptr_t buffer, std::size_t /*offset*/) {
            std::memcpy(buffer, input, input_length);
            check(buffer, input_length);
        });
    }
}

/** @brief Drive the malformed-input safety probe through serial, dispatched, and every native backend. */
void test_utf8_delimiters_safety() {
    std::printf("  - testing malformed-input safety of UTF-8 delimiter kernels...\n");
    check_utf8_delimiters_safety_(sz_utf8_delimiters_serial);
    check_utf8_delimiters_safety_(sz_utf8_delimiters);
#if SZ_USE_HASWELL
    check_utf8_delimiters_safety_(sz_utf8_delimiters_haswell);
#endif
#if SZ_USE_ICELAKE
    check_utf8_delimiters_safety_(sz_utf8_delimiters_icelake);
#endif
#if SZ_USE_NEON
    check_utf8_delimiters_safety_(sz_utf8_delimiters_neon);
#endif
    std::printf("    malformed-input safety passed!\n");
}

/** @brief Drive the serial-vs-SIMD UTF-8 delimiter differential across every backend compiled on this target. */
void test_utf8_delimiters_all() {
    sz_size_t const inputs = (sz_size_t)scale_iterations(200);
    sz_unused_(inputs);
#if SZ_USE_HASWELL
    test_utf8_delimiters_equivalence(sz_utf8_delimiters_serial, sz_utf8_delimiters_haswell, inputs);
#endif
#if SZ_USE_ICELAKE
    test_utf8_delimiters_equivalence(sz_utf8_delimiters_serial, sz_utf8_delimiters_icelake, inputs);
#endif
#if SZ_USE_NEON
    test_utf8_delimiters_equivalence(sz_utf8_delimiters_serial, sz_utf8_delimiters_neon, inputs);
#endif
}

#pragma endregion // Delimiters
