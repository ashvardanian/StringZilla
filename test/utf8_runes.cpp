/**
 *  @file test/utf8_runes.cpp
 *  @author Ash Vardanian
 *  @date November 24, 2025
 *  @brief UTF-8 codepoint counting, nth-character finding, and streaming rune-unpacking tests.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

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

#include <cstring> // `std::memcpy`

#include <string> // Baseline
#include <vector> // `std::vector`

#include "cross.hpp" // `check_utf8_runes_unit_`, `utf8_runes_backend_t`, `random_valid_utf8_`

namespace ashvardanian::stringzilla::test {
using sz::literals::operator""_sv; // for `sz::string_view_t`

/** The dispatch points in the shape of a codepoint backend. */
static utf8_runes_backend_t const utf8_runes_dispatched = {"dispatched", cpu_best<sz_utf8_count_best>,
                                                           cpu_best<sz_utf8_seek_best>, cpu_best<sz_utf8_decode_best>,
                                                           cpu_best<sz_utf8_count_strings_best>};

#pragma region Unit

/**
 *  @brief Known-answer unit tests for the UTF-8 codepoint family on simple, hand-verifiable inputs.
 *
 *  Exercises each function through the dispatched C API and through the C++ @c sz::string_view_t
 *  wrappers, so a regression that the serial-vs-SIMD agreement tests would miss - because both
 *  share a wrong constant - is still caught against an external ground truth. The same anchor
 *  holds every capability's kernels in the cross files.
 */
void test_utf8_runes_unit() {
    // The count + find-nth + unpack known-answer through the dispatch points.
    check_utf8_runes_unit_(utf8_runes_dispatched);

    // C++ API: character counting vs byte length through the `sz::string_view_t` wrappers.
    expect_equal(sz::utf8_count(sz::string_view_t("a\xC3\x9F\xE4\xB8\xAD")), 3u); // `a` + U+00DF + U+4E2D
    expect_equal(sz::utf8_count("hello"_sv), 5);
    expect_equal("hello"_sv.size(), 5);
    expect_equal(sz::utf8_count("Hello World"_sv), 11);
    expect_equal(sz::utf8_count(sz::string_view_t("")), 0);
    expect_equal(sz::utf8_count(sz::string_view_t("Hello \xE4\xB8\x96\xE7\x95\x8C")), 8); // "Hello " (6) + 2 CJK chars
    expect_equal(sz::string_view_t("Hello \xE4\xB8\x96\xE7\x95\x8C").size(), 12);         // "Hello " (6) + 6 bytes
    expect_equal(sz::utf8_count(sz::string_view_t("Hello \xF0\x9F\x98\x80")), 7);         // "Hello " (6) + 1 emoji
    expect_equal(sz::string_view_t("Hello \xF0\x9F\x98\x80").size(), 10);                 // "Hello " (6) + 4 bytes
    expect_equal(sz::utf8_count(sz::string_view_t("\xF0\x9F\x98\x80\xF0\x9F\x98\x81\xF0\x9F\x98\x82")), 3);
    expect_equal(sz::string_view_t("\xF0\x9F\x98\x80\xF0\x9F\x98\x81\xF0\x9F\x98\x82").size(), 12);
    expect_equal(sz::utf8_count(sz::string_view_t("\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82")), 6);
    expect_equal(sz::string_view_t("\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82").size(), 12);

    // C++ API: byte offset of the nth character (and the npos beyond-end sentinel).
    {
        sz::string_view_t text = "Hello";
        expect_equal(sz::utf8_seek(text, 0), 0);
        expect_equal(sz::utf8_seek(text, 1), 1);
        expect_equal(sz::utf8_seek(text, 4), 4);
        expect_equal(sz::utf8_seek(text, 5), sz::string_view_t::npos);
        expect_equal(sz::utf8_seek(text, 100), sz::string_view_t::npos);
    }
    {
        sz::string_view_t text = "Hello \xE4\xB8\x96\xE7\x95\x8C";
        expect_equal(sz::utf8_seek(text, 0), 0); // 'H' at byte 0
        expect_equal(sz::utf8_seek(text, 5), 5); // ' ' at byte 5
        expect_equal(sz::utf8_seek(text, 6), 6); // '世' at byte 6
        expect_equal(sz::utf8_seek(text, 7), 9); // '界' at byte 9
        expect_equal(sz::utf8_seek(text, 8), sz::string_view_t::npos);
    }
    {
        sz::string_view_t text = "\xF0\x9F\x98\x80\xF0\x9F\x98\x81\xF0\x9F\x98\x82";
        expect_equal(sz::utf8_seek(text, 0), 0); // First emoji at byte 0
        expect_equal(sz::utf8_seek(text, 1), 4); // Second emoji at byte 4
        expect_equal(sz::utf8_seek(text, 2), 8); // Third emoji at byte 8
        expect_equal(sz::utf8_seek(text, 3), sz::string_view_t::npos);
    }

    // 64-byte chunk boundaries and batch limits, materialized via the vector wrapper.
    {
        // Critical 63, 64, 65 byte boundaries
        {
            std::string s63(63, 'x');
            expect_equal(sz::utf8_runes(sz::string_view_t(s63)).size(), 63);
        }
        {
            std::string s64(64, 'x');
            expect_equal(sz::utf8_runes(sz::string_view_t(s64)).size(), 64);
        }
        {
            std::string s65(65, 'x');
            expect_equal(sz::utf8_runes(sz::string_view_t(s65)).size(), 65);
        }

        // ASCII batch limit: 16 characters max per Ice Lake iteration
        {
            std::string s17(17, 'x');
            expect_equal(sz::utf8_runes(sz::string_view_t(s17)).size(), 17);
        }
        {
            std::string s20(20, 'x');
            expect_equal(sz::utf8_runes(sz::string_view_t(s20)).size(), 20);
        }

        // 2-byte batch limit: 32 characters (64 bytes) max per iteration
        {
            std::string cyr32;
            for (int i = 0; i < 32; ++i) cyr32 += "\xD0\x9F";
            expect_equal(sz::utf8_count(sz::string_view_t(cyr32)), 32);
        }
        {
            std::string cyr33;
            for (int i = 0; i < 33; ++i) cyr33 += "\xD0\x9F";
            expect_equal(sz::utf8_count(sz::string_view_t(cyr33)), 33);
        }

        // 3-byte batch limit: 16 characters (48 bytes) max per iteration
        {
            std::string cjk16;
            for (int i = 0; i < 16; ++i) cjk16 += "\xE4\xB8\x96";
            expect_equal(sz::utf8_count(sz::string_view_t(cjk16)), 16);
        }
        {
            std::string cjk17;
            for (int i = 0; i < 17; ++i) cjk17 += "\xE4\xB8\x96";
            expect_equal(sz::utf8_count(sz::string_view_t(cjk17)), 17);
        }

        // 4-byte batch limit: 16 characters (64 bytes) max per iteration
        {
            std::string emoji16;
            for (int i = 0; i < 16; ++i) emoji16 += "\xF0\x9F\x98\x80";
            expect_equal(sz::utf8_count(sz::string_view_t(emoji16)), 16);
        }
        {
            std::string emoji17;
            for (int i = 0; i < 17; ++i) emoji17 += "\xF0\x9F\x98\x80";
            expect_equal(sz::utf8_count(sz::string_view_t(emoji17)), 17);
        }

        // Asymmetric at chunk boundary: 60 ASCII + "ПП世" = 63 chars, 67 bytes
        {
            std::string boundary_asym(60, 'x');
            boundary_asym += "\xD0\x9F\xD0\x9F\xE4\xB8\x96";
            expect_equal(sz::utf8_count(sz::string_view_t(boundary_asym)), 63);
        }

        // Sequences exceeding batch limits
        {
            std::string cyr100;
            for (int i = 0; i < 100; ++i) cyr100 += "\xD0\x9F";
            expect_equal(sz::utf8_runes(sz::string_view_t(cyr100)).size(), 100);
        }
        {
            std::string cjk50;
            for (int i = 0; i < 50; ++i) cjk50 += "\xE4\xB8\x96";
            expect_equal(sz::utf8_runes(sz::string_view_t(cjk50)).size(), 50);
        }
        {
            std::string emoji50;
            for (int i = 0; i < 50; ++i) emoji50 += "\xF0\x9F\x98\x80";
            expect_equal(sz::utf8_runes(sz::string_view_t(emoji50)).size(), 50);
        }

        // Asymmetric overflow: 20x (2 ASCII + 3 Cyrillic) = 100 chars, 140 bytes
        {
            std::string overflow_asym;
            for (int i = 0; i < 20; ++i) overflow_asym += "aa\xD0\x9F\xD0\xA0\xD0\xA1";
            expect_equal(sz::utf8_count(sz::string_view_t(overflow_asym)), 100);
        }

        // Transitions at chunk boundaries
        {
            std::string boundary_test(63, 'x');
            boundary_test += "\xD0\x9F";
            expect_equal(sz::utf8_runes(sz::string_view_t(boundary_test)).size(), 64);
        }
        {
            std::string span_asym;
            {
                for (int i = 0; i < 30; ++i) span_asym += "aa";
                for (int i = 0; i < 8; ++i) span_asym += "\xD0\x9F\xD0\xA0\xD0\xA1";
            };
            expect_equal(sz::utf8_count(sz::string_view_t(span_asym)), 84);
        }
        {
            std::string exact_boundary(64, 'x');
            exact_boundary += "\xD0\x9F\xE4\xB8\x96\xF0\x9F\x98\x80";
            expect_equal(sz::utf8_count(sz::string_view_t(exact_boundary)), 67);
        }
    }

    // `+=` counts from the current rune, both inside the decoded batch and past it.
    {
        auto rune = sz::utf8_runes("abcdefgh"_sv).begin();
        expect_equal(*(++rune += 2), 'd');
        sz::utf8_runes_view<sz::string_view_t, 2> pairs("a\xD0\x9F\xE4\xB8\x96\xF0\x9F\x98\x80xy"_sv);
        auto paired = pairs.begin();
        expect_true(*(paired += 3) == 0x1F600 && *(paired += 2) == 'y' && ++paired == pairs.end());
    }
}

/**
 *  @brief Known-answer rune-iteration vectors spanning Unicode scripts and byte-width transitions.
 *
 *  Decodes hand-written samples of ASCII, CJK, Cyrillic, Arabic, Hebrew, Thai, Devanagari, emoji,
 *  the maximum codepoint U+10FFFF, Deseret, zero-width and combining marks through the C++
 *  @c utf8_runes wrapper, then walks every 1/2/3/4-byte neighbor pair, so a kernel that assumes a
 *  homogeneous byte-width run is caught here.
 */
void test_utf8_runes_scripts_unit() {
    // C++ API: codepoint iteration materialized as a vector, since every check below indexes by position.
    {
        auto runes_of = [](char const *t) {
            return sz::utf8_runes(sz::string_view_t(t)).template to<std::vector<sz_rune_t>>();
        };

        // Basic ASCII and edge cases
        {
            auto c = runes_of("Hello");
            expect_true(c.size() == 5 && c[0] == 'H' && c[4] == 'o');
        }
        {
            auto c = runes_of("");
            expect_equal(c.size(), 0);
        }
        {
            auto c = runes_of("A");
            expect_true(c.size() == 1 && c[0] == 'A');
        }

        // CJK (3-byte UTF-8)
        {
            auto c = runes_of("\xE4\xB8\x96\xE7\x95\x8C");
            expect_true(c.size() == 2 && c[0] == 0x4E16 && c[1] == 0x754C);
        }
        {
            auto c = runes_of("\xE4\xBD\xA0\xE5\xA5\xBD");
            expect_true(c.size() == 2 && c[0] == 0x4F60 && c[1] == 0x597D);
        }

        // Cyrillic (2-byte UTF-8)
        {
            auto c = runes_of("\xD0\x9F\xD1\x80\xD0\xB8\xD0\xB2\xD0\xB5\xD1\x82");
            expect_true(c.size() == 6 && c[0] == 0x041F && c[5] == 0x0442);
        }

        // Arabic/RTL (2-byte UTF-8)
        {
            auto c = runes_of("\xD9\x85\xD8\xB1\xD8\xAD\xD8\xA8\xD8\xA7");
            expect_true(c.size() == 5 && c[0] == 0x0645 && c[4] == 0x0627);
        }

        // Hebrew/RTL (2-byte UTF-8)
        {
            auto c = runes_of("\xD7\xA9\xD7\x9C\xD7\x95\xD7\x9D");
            expect_true(c.size() == 4 && c[0] == 0x05E9 && c[3] == 0x05DD);
        }

        // Thai (3-byte UTF-8)
        {
            auto c = runes_of("\xE0\xB8\xAA\xE0\xB8\xA7\xE0\xB8\xB1\xE0\xB8\xAA\xE0\xB8\x94\xE0\xB8\xB5");
            expect_true(c.size() == 6 && c[0] == 0x0E2A);
        }

        // Devanagari/Hindi (3-byte UTF-8)
        {
            auto c = runes_of("\xE0\xA4\xA8\xE0\xA4\xAE\xE0\xA4\xB8\xE0\xA5\x8D\xE0\xA4\xA4\xE0\xA5\x87");
            expect_true(c.size() == 6 && c[0] == 0x0928);
        }

        // Emoji: basic smileys (4-byte UTF-8)
        {
            auto c = runes_of("\xF0\x9F\x98\x80\xF0\x9F\x98\x81\xF0\x9F\x98\x82");
            expect_true(c.size() == 3 && c[0] == 0x1F600 && c[2] == 0x1F602);
        }

        // Emoji: with variation selector
        {
            auto c = runes_of("\xE2\x9D\xA4\xEF\xB8\x8F");
            expect_true(c.size() == 2 && c[0] == 0x2764 && c[1] == 0xFE0F);
        }

        // Emoji: various categories
        {
            auto c = runes_of("\xF0\x9F\x9A\x80\xF0\x9F\x8E\x89\xF0\x9F\x94\xA5");
            expect_true(c.size() == 3 && c[0] == 0x1F680);
        }

        // Maximum valid Unicode codepoint (U+10FFFF)
        {
            auto c = runes_of("\xF4\x8F\xBF\xBF");
            expect_true(c.size() == 1 && c[0] == 0x10FFFF);
        }

        // Deseret alphabet (4-byte UTF-8, U+10400 range)
        {
            auto c = runes_of("\xF0\x90\x90\xB7");
            expect_true(c.size() == 1 && c[0] == 0x10437);
        }

        // Mixed scripts
        {
            auto c = runes_of("Hello\xE4\xB8\x96\xE7\x95\x8C");
            expect_true(c.size() == 7 && c[4] == 'o' && c[5] == 0x4E16);
        }
        {
            auto c = runes_of("a\xF0\x90\x90\xB7" // the next fragment starts with a hex digit
                              "b");
            expect_true(c.size() == 3 && c[0] == 'a' && c[1] == 0x10437 && c[2] == 'b');
        }

        // Zero-width characters
        {
            auto c = runes_of("a\xE2\x80\x8B" // the next fragment starts with a hex digit
                              "b");
            expect_true(c.size() == 3 && c[0] == 'a' && c[1] == 0x200B && c[2] == 'b');
        }
        {
            auto c = runes_of("\xEF\xBB\xBF");
            expect_true(c.size() == 1 && c[0] == 0xFEFF);
        } // BOM

        // A truncated sequence at the end is one U+FFFD, not dropped
        {
            auto c = runes_of("a\xE2\x82");
            expect_true(c.size() == 2 && c[0] == 'a' && c[1] == 0xFFFD);
        }

        // Combining diacritics (e + combining acute) vs precomposed. Written as \xHH escapes so the decomposed
        // sequence cannot be NFC-composed away by an editor/normalizer.
        {
            auto c = runes_of("e\xCC\x81");
            expect_true(c.size() == 2 && c[0] == 'e' && c[1] == 0x0301);
        } // e + U+0301
        {
            auto c = runes_of("\xC3\xA9");
            expect_true(c.size() == 1 && c[0] == 0x00E9);
        } // precomposed U+00E9

        // Missing transitions: 1 → 2, 2 → 1, 2 → 3, 3 → 2, 2 → 4, 4 → 2, 3 → 4, 4 → 3
        {
            auto c = runes_of("a\xD0\x9F");
            expect_true(c.size() == 2 && c[0] == 'a' && c[1] == 0x041F);
        } // 1 → 2
        {
            auto c = runes_of("\xD0\x9F" // the next fragment starts with a hex digit
                              "a");
            expect_true(c.size() == 2 && c[0] == 0x041F && c[1] == 'a');
        } // 2 → 1
        {
            auto c = runes_of("\xD0\x9F\xE4\xB8\x96");
            expect_true(c.size() == 2 && c[0] == 0x041F && c[1] == 0x4E16);
        } // 2 → 3
        {
            auto c = runes_of("\xE4\xB8\x96\xD0\x9F");
            expect_true(c.size() == 2 && c[0] == 0x4E16 && c[1] == 0x041F);
        } // 3 → 2
        {
            auto c = runes_of("\xD0\x9F\xF0\x9F\x98\x80");
            expect_true(c.size() == 2 && c[0] == 0x041F && c[1] == 0x1F600);
        } // 2 → 4
        {
            auto c = runes_of("\xF0\x9F\x98\x80\xD0\x9F");
            expect_true(c.size() == 2 && c[0] == 0x1F600 && c[1] == 0x041F);
        } // 4 → 2
        {
            auto c = runes_of("\xE4\xB8\x96\xF0\x9F\x98\x80");
            expect_true(c.size() == 2 && c[0] == 0x4E16 && c[1] == 0x1F600);
        } // 3 → 4
        {
            auto c = runes_of("\xF0\x9F\x98\x80\xE4\xB8\x96");
            expect_true(c.size() == 2 && c[0] == 0x1F600 && c[1] == 0x4E16);
        } // 4 → 3

        // Extended transitions with same-length runs
        {
            auto c = runes_of("\xD0\x9F\xD0\xA0\xD0\xA1");
            expect_true(c.size() == 3 && c[0] == 0x041F && c[2] == 0x0421);
        } // 2 → 2 → 2
        {
            auto c = runes_of("\xE4\xB8\x96\xE7\x95\x8C\xE4\xBA\xBA");
            expect_true(c.size() == 3 && c[0] == 0x4E16 && c[2] == 0x4EBA);
        } // 3 → 3 → 3

        // Asymmetric alternating patterns - stress the homogeneity assumption
        {
            auto c = runes_of("xx\xD0\x9F\xD0\x9F\xD0\x9Fxx\xD0\x9F\xD0\x9F\xD0\x9F");
            expect_equal(c.size(), 10);
        } // 2 ASCII, 3 Cyrillic
        {
            auto c = runes_of("xxx\xD0\x9F\xD0\x9Fxxx\xD0\x9F\xD0\x9F");
            expect_equal(c.size(), 10);
        } // 3 ASCII, 2 Cyrillic
        {
            auto c = runes_of("xx\xE4\xB8\x96\xE4\xB8\x96\xE4\xB8\x96xx\xE4\xB8\x96\xE4\xB8\x96\xE4\xB8\x96");
            expect_equal(c.size(), 10);
        } // 2 ASCII, 3 CJK
        {
            auto c = runes_of(
                "\xD0\x9F\xD0\x9F\xE4\xB8\x96\xE4\xB8\x96\xE4\xB8\x96\xD0\x9F\xD0\x9F\xE4\xB8\x96\xE4\xB8" // one literal wrapped at the column limit
                "\x96\xE4\xB8\x96");
            expect_equal(c.size(), 10);
        } // 2 Cyrillic, 3 CJK
        {
            auto c = runes_of(
                "\xE4\xB8\x96\xE4\xB8\x96\xF0\x9F\x98\x80\xF0\x9F\x98\x80\xF0\x9F\x98\x80\xE4\xB8\x96\xE4" // one literal wrapped at the column limit
                "\xB8\x96\xF0\x9F\x98\x80\xF0\x9F\x98\x80\xF0\x9F\x98\x80");
            expect_equal(c.size(), 10);
        } // 2 CJK, 3 Emoji
        {
            auto c = runes_of("xxx\xF0\x9F\x98\x80\xF0\x9F\x98\x80xxx\xF0\x9F\x98\x80\xF0\x9F\x98\x80");
            expect_equal(c.size(), 10);
        } // 3 ASCII, 2 Emoji

        // Pathological mixed patterns
        {
            auto c = runes_of(
                "xx\xD0\x9F\xD0\x9F\xD0\x9F\xD0\x9F\xE4\xB8\x96\xE4\xB8\x96\xE4\xB8\x96\xE4\xB8\x96\xF0" // one literal wrapped at the column limit
                "\x9F\x98\x80\xF0\x9F\x98\x80\xF0\x9F\x98\x80\xF0\x9F\x98\x80\xF0\x9F\x98\x80");
            expect_equal(c.size(), 15);
        } // 2-4-4-5
        {
            auto c = runes_of(
                "xx\xD0\x9F\xD0\x9F\xD0\x9Fxx\xF0\x9F\x98\x80\xF0\x9F\x98\x80\xF0\x9F\x98\x80\xF0\x9F\x98" // one literal wrapped at the column limit
                "\x80\xE4\xB8\x96\xE4\xB8\x96\xE4\xB8\x96\xD0\x9F\xD0\x9F");
            expect_equal(c.size(), 16);
        } // 2-3-2-4-3-2

        // Extended asymmetric: 30x "xxППП" = 150 chars, 210 bytes (crosses multiple 64-byte chunks)
        {
            std::string asym_long;
            for (int i = 0; i < 30; ++i) asym_long += "xx\xD0\x9F\xD0\x9F\xD0\x9F";
            expect_equal(sz::utf8_count(sz::string_view_t(asym_long)), 150);
        }
    }
}

#pragma endregion Unit

#pragma region Equivalence

/** Large-buffer count agreement: a few hundred KB of mixed-width codepoints where the dispatched
 *  and C++ counts must equal the serial reference and the exact known total. */
static void check_utf8_runes_large_count_() {
    // Every repeat contributes one ASCII 'x', one 2-byte, one 3-byte, and one 4-byte codepoint - 4
    // codepoints in 10 bytes - so the total is exactly `repeats * 4`.
    char const unit[] = "x\xD0\x9F\xE4\xB8\xAD\xF0\x9F\x98\x80"; // 'x' + U+041F + U+4E2D + U+1F600, 10 bytes
    std::size_t const repeats = scaled_iterations(30000);        // ~300 KB at the default multiplier
    std::string mixed;
    mixed.reserve(repeats * (sizeof(unit) - 1));
    for (std::size_t repeat = 0; repeat != repeats; ++repeat) mixed.append(unit, sizeof(unit) - 1);

    sz_size_t const expected_codepoints = (sz_size_t)(repeats * 4);
    sz_size_t const count_serial = kernel_result<sz_size_t>(sz_utf8_count_serial, mixed.data(), mixed.size());
    expect_equal(count_serial, expected_codepoints);
    expect_equal(kernel_result<sz_size_t>(cpu_best<sz_utf8_count_best>, mixed.data(), mixed.size()), count_serial);
    expect_equal(sz::utf8_count(sz::string_view_t(mixed)), count_serial); // C++ wrapper matches serial
}

#pragma endregion Equivalence

#pragma region Safety

/** Drive the malformed-input safety probe through the dispatch points. */
void test_utf8_runes_safety() {
    split_mix_generator_t generator {case_stream_key()};
    check_utf8_runes_safety_(generator, utf8_runes_dispatched);
}

#pragma endregion Safety

#pragma region Drivers

/** Drives the serial-vs-dispatched UTF-8 codepoint differential (unpack + find-nth + count), plus
 *  the large-buffer count agreement. */
void test_utf8_runes_all() {
    split_mix_generator_t generator {case_stream_key()};
    check_utf8_runes_equivalence_(generator, utf8_runes_dispatched);

    // Large-buffer count agreement: serial == dispatched == C++ wrapper == known total.
    check_utf8_runes_large_count_();
}

#pragma endregion Drivers

} // namespace ashvardanian::stringzilla::test
